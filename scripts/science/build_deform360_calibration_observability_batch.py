#!/usr/bin/env python3
"""Build all ten Deform360 observability cases and report atomically.

The command accepts one strict portable specification that accounts for every
frozen calibration object. It routes each row through the existing claim-bearing
per-object producer, constructs the object-balanced report, writes a
content-addressed manifest and checksum inventory, reloads the complete staged
directory, and only then publishes it without replacement.

No confirmation payload or target outcome is admitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from bayesian_phystwin._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_execution import (
    load_deform360_stage0_selection,
)
from bayesian_phystwin.deform360_calibration_observability_case_builder import (
    build_evaluated_case_from_paths,
    build_technical_failure_case_from_paths,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM,
    DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID,
    Deform360CalibrationObservabilityCaseV1,
    build_report_from_paths,
    load_deform360_calibration_observability_case,
    load_deform360_calibration_observability_report,
    save_deform360_calibration_observability_case,
    save_deform360_calibration_observability_report,
)

SPEC_SCHEMA = "bayesian-phystwin.deform360-calibration-observability-batch-spec"
MANIFEST_SCHEMA = "bayesian-phystwin.deform360-calibration-observability-batch"
SCHEMA_VERSION = 1
SEMANTICS = "atomic-ten-object-calibration-observability-assembly-v1"
CLAIM_BOUNDARY = (
    "Calibration-only case assembly and observability-report evidence. A valid "
    "batch does not establish Deform360 accuracy, tactile benefit, provider "
    "competence, calibrated uncertainty, Causal4D benefit, deployment safety, "
    "or state of the art."
)
INSUFFICIENT_SUPPORT_EXIT_CODE = 3
CONTRACT_FAILURE_EXIT_CODE = 2

Mode = Literal["evaluated", "technical-failure"]

_MANIFEST_NAME = "batch-manifest.json"
_REPORT_NAME = "calibration-observability-report.json"
_CHECKSUMS_NAME = "SHA256SUMS"
_CASES_DIRECTORY = "cases"
_SPEC_SOURCE_KEY = "sources/batch/spec.json"
_QUERY_SOURCE_KEY = "sources/observability/shared/physical-query-jacobian.npy"
_SPEC_FIELDS = frozenset({"schema", "schema_version", "cases"})
_EVALUATED_FIELDS = frozenset(
    {
        "mode",
        "object_id",
        "reference_marginal_precision",
        "candidate_marginal_precision",
        "contact_anchor_artifact",
    }
)
_FAILURE_FIELDS = frozenset({"mode", "object_id", "failure_evidence", "failure_reason"})
_CASE_FILE_FIELDS = frozenset({"case_id", "path", "file_sha256"})
_REPORT_FIELDS = frozenset(
    {"report_id", "path", "file_sha256", "status", "support_gate"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "batch_id",
        "protocol_id",
        "spec_id",
        "spec_file_sha256",
        "implementation_revision",
        "physical_query_id",
        "case_files",
        "report",
        "source_artifacts",
        "information_boundary",
        "claim_boundary",
    }
)
_BOUNDARY = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}


@dataclass(frozen=True)
class CaseSpec:
    mode: Mode
    object_id: str
    reference_marginal_precision: str | None = None
    candidate_marginal_precision: str | None = None
    contact_anchor_artifact: str | None = None
    failure_evidence: str | None = None
    failure_reason: str | None = None

    def to_record(self) -> dict[str, object]:
        if self.mode == "evaluated":
            return {
                "mode": self.mode,
                "object_id": self.object_id,
                "reference_marginal_precision": self.reference_marginal_precision,
                "candidate_marginal_precision": self.candidate_marginal_precision,
                "contact_anchor_artifact": self.contact_anchor_artifact,
            }
        return {
            "mode": self.mode,
            "object_id": self.object_id,
            "failure_evidence": self.failure_evidence,
            "failure_reason": self.failure_reason,
        }


def _literal(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result != result.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return result


def _mode(value: object) -> Mode:
    if type(value) is not str or value not in {"evaluated", "technical-failure"}:
        raise ValueError("case mode must be evaluated or technical-failure")
    return cast(Mode, value)


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symlinks: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an ordinary directory: {path}")
    return resolved


def _read_bytes(path: str | Path, *, name: str) -> tuple[bytes, str]:
    source = _ordinary_file(path, name=name)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not data:
        raise ValueError(f"{name} must not be empty")
    return data, hashlib.sha256(data).hexdigest()


def _file_sha256(path: str | Path, *, name: str) -> str:
    return _read_bytes(path, name=name)[1]


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_object(path: str | Path, *, name: str) -> tuple[Mapping[str, Any], str]:
    data, digest = _read_bytes(path, name=name)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse {name}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be a JSON object")
    return value, digest


def _case_spec(value: object, *, index: int) -> CaseSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"specification case {index} must be a JSON object")
    mode = _mode(value.get("mode"))
    fields = _EVALUATED_FIELDS if mode == "evaluated" else _FAILURE_FIELDS
    require_exact_fields(value, expected=fields, name=f"specification case {index}")
    object_id = _literal(value["object_id"], name=f"cases[{index}].object_id")
    if mode == "evaluated":
        return CaseSpec(
            mode=mode,
            object_id=object_id,
            reference_marginal_precision=canonical_relative_posix_path(
                value["reference_marginal_precision"],
                name=f"cases[{index}].reference_marginal_precision",
            ),
            candidate_marginal_precision=canonical_relative_posix_path(
                value["candidate_marginal_precision"],
                name=f"cases[{index}].candidate_marginal_precision",
            ),
            contact_anchor_artifact=canonical_relative_posix_path(
                value["contact_anchor_artifact"],
                name=f"cases[{index}].contact_anchor_artifact",
            ),
        )
    return CaseSpec(
        mode=mode,
        object_id=object_id,
        failure_evidence=canonical_relative_posix_path(
            value["failure_evidence"],
            name=f"cases[{index}].failure_evidence",
        ),
        failure_reason=_literal(
            value["failure_reason"],
            name=f"cases[{index}].failure_reason",
        ),
    )


def _load_spec(path: str | Path) -> tuple[tuple[CaseSpec, ...], str, str]:
    value, file_digest = _load_object(path, name="batch specification")
    require_exact_fields(value, expected=_SPEC_FIELDS, name="batch specification")
    if value["schema"] != SPEC_SCHEMA:
        raise ValueError("batch specification schema changed")
    version = genuine_integer(value["schema_version"], name="schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        raise ValueError("batch specification version changed")
    raw = value["cases"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("batch specification cases must be a sequence")
    cases = tuple(
        sorted(
            (_case_spec(case, index=index) for index, case in enumerate(raw)),
            key=lambda case: case.object_id,
        )
    )
    if len(cases) != 2 * DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM:
        raise ValueError("batch specification must contain exactly ten cases")
    object_ids = [case.object_id for case in cases]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("batch specification repeats a physical object")
    semantic = {
        "schema": SPEC_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cases": [case.to_record() for case in cases],
    }
    return cases, content_id(semantic), file_digest


def _resolve(root: Path, relative: str, *, name: str) -> Path:
    canonical = canonical_relative_posix_path(relative, name=name)
    candidate = root.joinpath(*PurePosixPath(canonical).parts)
    resolved = _ordinary_file(candidate, name=name)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes the input root")
    return resolved


def _resolved_inputs(root: Path, case: CaseSpec) -> dict[str, Path]:
    if case.mode == "evaluated":
        assert case.reference_marginal_precision is not None
        assert case.candidate_marginal_precision is not None
        assert case.contact_anchor_artifact is not None
        return {
            "reference": _resolve(
                root,
                case.reference_marginal_precision,
                name=f"{case.object_id} reference precision",
            ),
            "candidate": _resolve(
                root,
                case.candidate_marginal_precision,
                name=f"{case.object_id} candidate precision",
            ),
            "anchor": _resolve(
                root,
                case.contact_anchor_artifact,
                name=f"{case.object_id} contact anchor",
            ),
        }
    assert case.failure_evidence is not None
    return {
        "failure": _resolve(
            root,
            case.failure_evidence,
            name=f"{case.object_id} failure evidence",
        )
    }


def _case_file(
    case: Deform360CalibrationObservabilityCaseV1,
    *,
    path: str,
    digest: str,
) -> dict[str, str]:
    return {
        "case_id": cast(str, case.case_id),
        "path": path,
        "file_sha256": digest,
    }


def _normalize_case_file(value: object, *, index: int) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"case file {index} must be a JSON object")
    require_exact_fields(value, expected=_CASE_FILE_FIELDS, name=f"case file {index}")
    case_id = sha256_digest(value["case_id"], name=f"case_files[{index}].case_id")
    path = canonical_relative_posix_path(
        value["path"],
        name=f"case_files[{index}].path",
    )
    if path != f"{_CASES_DIRECTORY}/{case_id}.json":
        raise ValueError("case file path is not content-addressed")
    return {
        "case_id": case_id,
        "path": path,
        "file_sha256": sha256_digest(
            value["file_sha256"],
            name=f"case_files[{index}].file_sha256",
        ),
    }


def _normalize_report(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest report must be a JSON object")
    require_exact_fields(value, expected=_REPORT_FIELDS, name="manifest report")
    path = canonical_relative_posix_path(value["path"], name="report.path")
    if path != _REPORT_NAME:
        raise ValueError("manifest report path changed")
    gate = value["support_gate"]
    if not isinstance(gate, Mapping):
        raise ValueError("manifest report support_gate must be a JSON object")
    normalized_gate = plain_json(
        frozen_finite_json_mapping(gate, name="manifest report support_gate")
    )
    return {
        "report_id": sha256_digest(value["report_id"], name="report.report_id"),
        "path": path,
        "file_sha256": sha256_digest(
            value["file_sha256"],
            name="report.file_sha256",
        ),
        "status": _literal(value["status"], name="report.status"),
        "support_gate": normalized_gate,
    }


def _validated_manifest(value: Mapping[str, Any]) -> dict[str, object]:
    require_exact_fields(value, expected=_MANIFEST_FIELDS, name="batch manifest")
    if value["schema"] != MANIFEST_SCHEMA:
        raise ValueError("batch manifest schema changed")
    version = genuine_integer(value["schema_version"], name="schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        raise ValueError("batch manifest version changed")
    if value["semantics"] != SEMANTICS:
        raise ValueError("batch manifest semantics changed")
    if value["protocol_id"] != DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID:
        raise ValueError("batch manifest protocol changed")
    if value["information_boundary"] != _BOUNDARY:
        raise ValueError("batch manifest information boundary changed")
    if value["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("batch manifest claim boundary changed")

    raw_case_files = value["case_files"]
    if isinstance(raw_case_files, (str, bytes)) or not isinstance(
        raw_case_files,
        Sequence,
    ):
        raise ValueError("manifest case_files must be a sequence")
    case_files = sorted(
        (
            _normalize_case_file(item, index=index)
            for index, item in enumerate(raw_case_files)
        ),
        key=lambda item: item["case_id"],
    )
    if case_files != list(raw_case_files):
        raise ValueError("manifest case_files are not in canonical order")
    if len(case_files) != 2 * DEFORM360_CALIBRATION_OBJECTS_PER_STRATUM:
        raise ValueError("manifest must contain exactly ten case files")
    if len({item["case_id"] for item in case_files}) != len(case_files):
        raise ValueError("manifest repeats a case identity")

    sources = value["source_artifacts"]
    if not isinstance(sources, Mapping) or set(sources) != {
        _SPEC_SOURCE_KEY,
        _QUERY_SOURCE_KEY,
    }:
        raise ValueError("manifest source_artifacts changed")
    source_artifacts = {
        key: sha256_digest(sources[key], name=f"source_artifacts.{key}")
        for key in sorted(sources)
    }
    spec_digest = sha256_digest(value["spec_file_sha256"], name="spec_file_sha256")
    if source_artifacts[_SPEC_SOURCE_KEY] != spec_digest:
        raise ValueError("manifest specification digest changed")

    identity: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "protocol_id": DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID,
        "spec_id": sha256_digest(value["spec_id"], name="spec_id"),
        "spec_file_sha256": spec_digest,
        "implementation_revision": exact_revision(
            value["implementation_revision"],
            name="implementation_revision",
        ),
        "physical_query_id": sha256_digest(
            value["physical_query_id"],
            name="physical_query_id",
        ),
        "case_files": case_files,
        "report": _normalize_report(value["report"]),
        "source_artifacts": source_artifacts,
        "information_boundary": dict(_BOUNDARY),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    expected_id = content_id(identity)
    if sha256_digest(value["batch_id"], name="batch_id") != expected_id:
        raise ValueError("batch manifest identity changed")
    return {**identity, "batch_id": expected_id}


def _manifest(
    *,
    spec_id: str,
    spec_digest: str,
    implementation_revision: str,
    physical_query_id: str,
    case_files: Sequence[Mapping[str, str]],
    report: Any,
    report_digest: str,
    query_digest: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "protocol_id": DEFORM360_CALIBRATION_OBSERVABILITY_PROTOCOL_ID,
        "spec_id": spec_id,
        "spec_file_sha256": spec_digest,
        "implementation_revision": implementation_revision,
        "physical_query_id": physical_query_id,
        "case_files": sorted(
            (dict(item) for item in case_files),
            key=lambda item: item["case_id"],
        ),
        "report": {
            "report_id": report.report_id,
            "path": _REPORT_NAME,
            "file_sha256": report_digest,
            "status": report.status,
            "support_gate": report.support_gate,
        },
        "source_artifacts": {
            _SPEC_SOURCE_KEY: spec_digest,
            _QUERY_SOURCE_KEY: query_digest,
        },
        "information_boundary": dict(_BOUNDARY),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return _validated_manifest({**identity, "batch_id": content_id(identity)})


def _write_checksums(root: Path, values: Mapping[str, str]) -> None:
    lines = [f"{values[path]}  {path}\n" for path in sorted(values)]
    with (root / _CHECKSUMS_NAME).open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())


def _read_checksums(path: Path) -> dict[str, str]:
    source = _ordinary_file(path, name="batch SHA256SUMS")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read batch SHA256SUMS") from error
    if not lines:
        raise ValueError("batch SHA256SUMS must not be empty")
    result: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("batch SHA256SUMS line is malformed")
        digest = sha256_digest(line[:64], name="SHA256SUMS digest")
        relative = canonical_relative_posix_path(
            line[66:],
            name="SHA256SUMS path",
        )
        if relative == _CHECKSUMS_NAME or relative in result:
            raise ValueError("batch SHA256SUMS repeats or self-hashes a path")
        result[relative] = digest
        order.append(relative)
    if order != sorted(order):
        raise ValueError("batch SHA256SUMS paths are not sorted")
    return result


def _directory_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in dirnames:
            path = current / name
            if path.is_symlink():
                raise ValueError("batch contains a symlinked directory")
            if path.relative_to(root).as_posix() != _CASES_DIRECTORY:
                raise ValueError("batch contains an unexpected directory")
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("batch contains a non-ordinary file")
            files.add(path.relative_to(root).as_posix())
    return files


def validate_directory(root: str | Path) -> dict[str, object]:
    """Reload and independently validate one complete published batch."""

    batch_root = _ordinary_directory(root, name="batch directory")
    manifest_value, _digest = _load_object(
        batch_root / _MANIFEST_NAME,
        name="batch manifest",
    )
    manifest = _validated_manifest(manifest_value)
    case_files = cast(list[dict[str, str]], manifest["case_files"])
    report_entry = cast(dict[str, object], manifest["report"])
    expected_files = {
        _MANIFEST_NAME,
        _REPORT_NAME,
        _CHECKSUMS_NAME,
        *(item["path"] for item in case_files),
    }
    if _directory_files(batch_root) != expected_files:
        raise ValueError("batch file set changed")
    expected_checksums = {
        _MANIFEST_NAME: _file_sha256(
            batch_root / _MANIFEST_NAME,
            name="batch manifest",
        ),
        _REPORT_NAME: cast(str, report_entry["file_sha256"]),
        **{item["path"]: item["file_sha256"] for item in case_files},
    }
    if _read_checksums(batch_root / _CHECKSUMS_NAME) != expected_checksums:
        raise ValueError("batch SHA256SUMS changed")
    for relative, digest in expected_checksums.items():
        if _file_sha256(batch_root / relative, name=relative) != digest:
            raise ValueError(f"batch file digest changed: {relative}")

    cases: list[Deform360CalibrationObservabilityCaseV1] = []
    for item in case_files:
        case = load_deform360_calibration_observability_case(batch_root / item["path"])
        if case.case_id != item["case_id"]:
            raise ValueError("manifest case identity differs from case file")
        cases.append(case)
    report = load_deform360_calibration_observability_report(batch_root / _REPORT_NAME)
    if report.report_id != report_entry["report_id"]:
        raise ValueError("manifest report identity differs from report file")
    if report.status != report_entry["status"]:
        raise ValueError("manifest report status differs from report file")
    if report.support_gate != report_entry["support_gate"]:
        raise ValueError("manifest support gate differs from report file")
    if report.implementation_revision != manifest["implementation_revision"]:
        raise ValueError("manifest implementation revision differs from report")
    if report.physical_query_id != manifest["physical_query_id"]:
        raise ValueError("manifest physical query differs from report")
    if report.metadata.get("batch_spec_id") != manifest["spec_id"]:
        raise ValueError("report does not bind the batch specification")
    if {case.case_id for case in cases} != {case.case_id for case in report.cases}:
        raise ValueError("report and batch contain different cases")
    for case, item in zip(cases, case_files, strict=True):
        logical = f"sources/observability/cases/{case.case_id}.json"
        if report.source_artifacts.get(logical) != item["file_sha256"]:
            raise ValueError("report case-source digest changed")

    query_digest = cast(dict[str, str], manifest["source_artifacts"])[_QUERY_SOURCE_KEY]
    for case in cases:
        matches = [
            digest
            for path, digest in case.source_artifacts.items()
            if path.endswith("/physical-query-jacobian.npy")
        ]
        if matches != [query_digest]:
            raise ValueError("case physical-query source differs from batch")
    return manifest


def _publish(staged: Path, output: Path) -> None:
    lock = output.parent / f".{output.name}.publish.lock"
    created = False
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created = True
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(staged, output)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            lock.unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> dict[str, object]:
    implementation = exact_revision(
        args.implementation_revision,
        name="implementation_revision",
    )
    specs, spec_id, spec_digest = _load_spec(args.batch_spec)
    input_root = _ordinary_directory(args.input_root, name="input root")
    protocol = _ordinary_file(args.stage0_protocol, name="Stage-0 protocol")
    selection_path = _ordinary_file(args.selection_lock, name="Stage-0 selection")
    selection = load_deform360_stage0_selection(
        selection_path,
        protocol_path=protocol,
    )
    expected_ids = {unit.object_id for unit in selection.calibration_units}
    if {case.object_id for case in specs} != expected_ids:
        raise ValueError("batch specification differs from the calibration cohort")
    resolved = {case.object_id: _resolved_inputs(input_root, case) for case in specs}
    query = _ordinary_file(args.query_jacobian, name="physical query Jacobian")

    output = Path(args.output_dir).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_directory(output.parent, name="output parent")
    if os.path.lexists(output):
        raise FileExistsError(output)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.batch.", dir=output.parent)
    )
    staged = workspace / "published"
    staged.mkdir()
    (staged / _CASES_DIRECTORY).mkdir()
    try:
        common: dict[str, Any] = {
            "source_protocol_path": args.source_protocol,
            "stage0_protocol_path": protocol,
            "selection_lock_path": selection_path,
            "visual_provider_lock_path": args.visual_provider_lock,
            "calibration_source_plan_path": args.calibration_source_plan,
            "calibration_source_download_path": args.calibration_source_download,
            "calibration_source_run_record_path": args.calibration_source_run_record,
            "calibration_source_result_path": args.calibration_source_result,
            "implementation_revision": implementation,
            "query_jacobian_path": query,
        }
        cases: list[Deform360CalibrationObservabilityCaseV1] = []
        case_paths: list[Path] = []
        case_files: list[dict[str, str]] = []
        for spec in specs:
            inputs = resolved[spec.object_id]
            if spec.mode == "evaluated":
                case = build_evaluated_case_from_paths(
                    **common,
                    object_id=spec.object_id,
                    reference_marginal_precision_path=inputs["reference"],
                    candidate_marginal_precision_path=inputs["candidate"],
                    contact_anchor_artifact_path=inputs["anchor"],
                )
            else:
                assert spec.failure_reason is not None
                case = build_technical_failure_case_from_paths(
                    **common,
                    object_id=spec.object_id,
                    failure_evidence_path=inputs["failure"],
                    failure_reason=spec.failure_reason,
                )
            case_id = cast(str, case.case_id)
            relative = f"{_CASES_DIRECTORY}/{case_id}.json"
            path = staged / relative
            save_deform360_calibration_observability_case(case, path)
            digest = _file_sha256(path, name=f"case {case_id}")
            cases.append(case)
            case_paths.append(path)
            case_files.append(_case_file(case, path=relative, digest=digest))

        query_ids = {case.physical_query_id for case in cases}
        query_digests = {
            digest
            for case in cases
            for source_path, digest in case.source_artifacts.items()
            if source_path.endswith("/physical-query-jacobian.npy")
        }
        if len(query_ids) != 1 or len(query_digests) != 1:
            raise ValueError("generated cases use different physical queries")
        query_id = next(iter(query_ids))
        query_digest = next(iter(query_digests))

        report = build_report_from_paths(
            selection_lock_path=selection_path,
            stage0_protocol_path=protocol,
            visual_provider_lock_path=args.visual_provider_lock,
            calibration_source_run_record_path=args.calibration_source_run_record,
            case_paths=case_paths,
            implementation_revision=implementation,
            physical_query_id=query_id,
            numerical_positive_tolerance=args.numerical_positive_tolerance,
            metadata={"batch_spec_id": spec_id},
        )
        report_path = staged / _REPORT_NAME
        save_deform360_calibration_observability_report(report, report_path)
        report_digest = _file_sha256(report_path, name="observability report")
        manifest = _manifest(
            spec_id=spec_id,
            spec_digest=spec_digest,
            implementation_revision=implementation,
            physical_query_id=query_id,
            case_files=case_files,
            report=report,
            report_digest=report_digest,
            query_digest=query_digest,
        )
        manifest_path = staged / _MANIFEST_NAME
        write_atomic_json(manifest, manifest_path, overwrite=False)
        checksums = {
            _MANIFEST_NAME: _file_sha256(manifest_path, name="batch manifest"),
            _REPORT_NAME: report_digest,
            **{
                item["path"]: item["file_sha256"]
                for item in cast(list[dict[str, str]], manifest["case_files"])
            },
        }
        _write_checksums(staged, checksums)
        validate_directory(staged)
        _publish(staged, output)
        return validate_directory(output)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-spec", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--stage0-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--calibration-source-plan", type=Path, required=True)
    parser.add_argument("--calibration-source-download", type=Path, required=True)
    parser.add_argument("--calibration-source-run-record", type=Path, required=True)
    parser.add_argument("--calibration-source-result", type=Path, required=True)
    parser.add_argument("--query-jacobian", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--numerical-positive-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = _run(args)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"passed": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return CONTRACT_FAILURE_EXIT_CODE
    report = cast(dict[str, object], manifest["report"])
    gate = cast(dict[str, object], report["support_gate"])
    summary = {
        "passed": gate["support_passed"],
        "batch_id": manifest["batch_id"],
        "report_id": report["report_id"],
        "status": report["status"],
        "support_gate": gate,
        "output": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if gate["support_passed"] is True:
        return 0
    return INSUFFICIENT_SUPPORT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
