#!/usr/bin/env python3
"""Build content-addressed release artifact-matrix evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from bayesian_phystwin.numerical_environment_v1 import (
    NumericalEnvironmentV1,
    validate_embedded_numerical_environment_v1,
)

LANE_RECEIPT_SCHEMA: Final = "bayesian-phystwin.release-artifact-validation"
MATRIX_EVIDENCE_SCHEMA: Final = "bayesian-phystwin.release-artifact-matrix-evidence"
RELEASE_EVIDENCE_SCHEMA: Final = "bayesian-phystwin.release-candidate-evidence"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ReleaseMatrixEvidenceError(ValueError):
    """Raised when release-matrix evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReleaseLaneV1:
    lane: str
    python_version: str
    artifact_kind: str
    numpy_version: str
    resolver_input: str


_RELEASE_LANES = (
    ReleaseLaneV1(
        "py310-wheel-floor",
        "3.10",
        "wheel",
        "1.23.5",
        "requirements/release-runtime-py310-floor.txt",
    ),
    ReleaseLaneV1(
        "py310-sdist-floor",
        "3.10",
        "sdist",
        "1.23.5",
        "requirements/release-runtime-py310-floor.txt",
    ),
    ReleaseLaneV1(
        "py312-wheel",
        "3.12",
        "wheel",
        "2.2.6",
        "requirements/release-runtime-py312.txt",
    ),
    ReleaseLaneV1(
        "py312-sdist",
        "3.12",
        "sdist",
        "2.2.6",
        "requirements/release-runtime-py312.txt",
    ),
    ReleaseLaneV1(
        "py314-wheel",
        "3.14",
        "wheel",
        "2.5.2",
        "requirements/release-runtime-py314.txt",
    ),
    ReleaseLaneV1(
        "py314-sdist",
        "3.14",
        "sdist",
        "2.5.2",
        "requirements/release-runtime-py314.txt",
    ),
)
LANES_BY_NAME: Final = {lane.lane: lane for lane in _RELEASE_LANES}
RELEASE_RESOLVER_INPUTS: Final = (
    "requirements/release-build.txt",
    "requirements/release-runtime-py310-floor.txt",
    "requirements/release-runtime-py312.txt",
    "requirements/release-runtime-py314.txt",
)
_SOURCE_CONTRACT_KEYS: Final = {
    RELEASE_RESOLVER_INPUTS[0]: "release_build_requirements",
    RELEASE_RESOLVER_INPUTS[1]: "release_runtime_py310_floor_requirements",
    RELEASE_RESOLVER_INPUTS[2]: "release_runtime_py312_requirements",
    RELEASE_RESOLVER_INPUTS[3]: "release_runtime_py314_requirements",
}


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _content_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_record(path: Path, *, reported_path: str | None = None) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ReleaseMatrixEvidenceError(f"cannot read {path}") from error
    return {
        "path": path.name if reported_path is None else reported_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ReleaseMatrixEvidenceError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    def pairs(pairs_: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs_:
            if key in result:
                raise ReleaseMatrixEvidenceError(f"{name} has duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReleaseMatrixEvidenceError(f"{name} has non-finite value {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseMatrixEvidenceError(f"cannot read {name}") from error
    return _mapping(value, name=name)


def _verified_record(
    path: Path,
    *,
    name: str,
    id_field: str,
    schema: str,
) -> Mapping[str, Any]:
    value = _load_json(path, name=name)
    if value.get("schema") != schema or value.get("schema_version") != 1:
        raise ReleaseMatrixEvidenceError(f"{name} schema changed")
    supplied_id = value.get(id_field)
    if type(supplied_id) is not str or _SHA256.fullmatch(supplied_id) is None:
        raise ReleaseMatrixEvidenceError(f"{name} ID is invalid")
    descriptor = dict(value)
    descriptor.pop(id_field)
    if _content_id(descriptor) != supplied_id:
        raise ReleaseMatrixEvidenceError(f"{name} ID does not match payload")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReleaseMatrixEvidenceError(f"refusing to overwrite {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _profile(path: Path, *, name: str) -> NumericalEnvironmentV1:
    runtime = _load_json(path, name=name)
    try:
        profile = validate_embedded_numerical_environment_v1(
            runtime,
            require_profile=True,
            require_dependency_lock=True,
        )
    except ValueError as error:
        raise ReleaseMatrixEvidenceError(f"{name} is invalid: {error}") from error
    if profile is None or profile.dependency_lock is None:
        raise ReleaseMatrixEvidenceError(f"{name} is incomplete")
    return profile


def _versions(profile: NumericalEnvironmentV1) -> dict[str, str]:
    return {item.name: item.version for item in profile.installed_distributions}


def build_lane_receipt(
    *,
    lane_name: str,
    release_evidence_path: Path,
    artifact_path: Path,
    runtime_fragment_path: Path,
    resolver_input_path: Path,
    source_revision: str,
    runtime_python_version: str | None = None,
    runtime_numpy_version: str | None = None,
    runtime_project_version: str | None = None,
) -> dict[str, object]:
    """Validate one installed artifact lane and return its receipt."""

    lane = LANES_BY_NAME.get(lane_name)
    if lane is None:
        raise ReleaseMatrixEvidenceError(f"unknown release lane: {lane_name!r}")
    if _GIT_SHA.fullmatch(source_revision) is None:
        raise ReleaseMatrixEvidenceError("source revision must be a full Git SHA")
    evidence = _verified_record(
        release_evidence_path,
        name="release evidence",
        id_field="evidence_id",
        schema=RELEASE_EVIDENCE_SCHEMA,
    )
    if evidence.get("source_revision") != source_revision:
        raise ReleaseMatrixEvidenceError("release evidence source revision changed")

    python_version = runtime_python_version or platform.python_version()
    numpy_version = runtime_numpy_version or importlib.metadata.version("numpy")
    project_version = runtime_project_version or importlib.metadata.version(
        "bayesian-phystwin"
    )
    if ".".join(python_version.split(".")[:2]) != lane.python_version:
        raise ReleaseMatrixEvidenceError(
            f"lane {lane.lane} requires Python {lane.python_version}"
        )
    if numpy_version != lane.numpy_version:
        raise ReleaseMatrixEvidenceError(
            f"lane {lane.lane} requires NumPy {lane.numpy_version}"
        )
    if project_version != evidence.get("project_version"):
        raise ReleaseMatrixEvidenceError("installed project version changed")

    expected_artifact = _mapping(
        _mapping(evidence.get("artifacts"), name="release artifacts").get(
            lane.artifact_kind
        ),
        name=f"release {lane.artifact_kind}",
    )
    artifact = _file_record(artifact_path)
    if artifact != expected_artifact:
        raise ReleaseMatrixEvidenceError("artifact digest does not match release evidence")

    resolver = _file_record(resolver_input_path)
    if resolver["path"] != Path(lane.resolver_input).name:
        raise ReleaseMatrixEvidenceError("lane resolver input filename changed")
    profile = _profile(runtime_fragment_path, name="numerical environment")
    lock = profile.dependency_lock.as_dict()
    if lock != {
        "name": resolver["path"],
        "sha256": resolver["sha256"],
        "size_bytes": resolver["size_bytes"],
    }:
        raise ReleaseMatrixEvidenceError("profile does not bind exact resolver input")
    if profile.python_version != python_version or profile.numpy_version != numpy_version:
        raise ReleaseMatrixEvidenceError("profile runtime versions changed")
    versions = _versions(profile)
    if versions.get("numpy") != numpy_version:
        raise ReleaseMatrixEvidenceError("profile lacks installed NumPy")
    if versions.get("bayesian-phystwin") != project_version:
        raise ReleaseMatrixEvidenceError("profile lacks installed BayesianPhysTwin")

    runtime = _file_record(runtime_fragment_path)
    descriptor: dict[str, object] = {
        "schema": LANE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "lane": lane.lane,
        "source_revision": source_revision,
        "release_evidence_id": evidence["evidence_id"],
        "project_version": project_version,
        "artifact_kind": lane.artifact_kind,
        "artifact": artifact,
        "python": {"requested": lane.python_version, "actual": python_version},
        "numpy": {"expected": lane.numpy_version, "actual": numpy_version},
        "dependency_lock": lock,
        "numerical_environment": {
            "profile_id": profile.profile_id,
            "runtime_fragment_sha256": runtime["sha256"],
            "runtime_fragment_size_bytes": runtime["size_bytes"],
            "installed_distributions_sha256": profile.installed_distributions_sha256,
            "numpy_configuration_sha256": profile.numpy_configuration_sha256,
        },
        "claim_boundary": (
            "Installed-artifact and numerical-runtime compatibility evidence only; "
            "not an accuracy, calibration, deployment, or scientific claim."
        ),
    }
    return {"receipt_id": _content_id(descriptor), **descriptor}


def build_matrix_evidence(
    *,
    release_evidence_path: Path,
    receipts_dir: Path,
    project_root: Path,
) -> dict[str, object]:
    """Validate the exact six-lane roster and return aggregate evidence."""

    evidence = _verified_record(
        release_evidence_path,
        name="release evidence",
        id_field="evidence_id",
        schema=RELEASE_EVIDENCE_SCHEMA,
    )
    receipt_paths = sorted(receipts_dir.rglob("validation-receipt-*.json"))
    if len(receipt_paths) != len(_RELEASE_LANES):
        raise ReleaseMatrixEvidenceError(
            f"expected {len(_RELEASE_LANES)} lane receipts, got {len(receipt_paths)}"
        )
    receipts: dict[str, Mapping[str, Any]] = {}
    for path in receipt_paths:
        receipt = _verified_record(
            path,
            name=f"lane receipt {path.name}",
            id_field="receipt_id",
            schema=LANE_RECEIPT_SCHEMA,
        )
        lane_name = receipt.get("lane")
        if type(lane_name) is not str or lane_name not in LANES_BY_NAME:
            raise ReleaseMatrixEvidenceError("lane receipt names an unknown lane")
        if lane_name in receipts:
            raise ReleaseMatrixEvidenceError(f"duplicate lane receipt: {lane_name}")
        receipts[lane_name] = receipt
    if set(receipts) != set(LANES_BY_NAME):
        raise ReleaseMatrixEvidenceError("lane receipt roster changed")

    source_contracts = _mapping(
        evidence.get("source_contracts"), name="source contracts"
    )
    resolvers: dict[str, dict[str, object]] = {}
    root = project_root.resolve()
    for relative in RELEASE_RESOLVER_INPUTS:
        record = _file_record(root / relative, reported_path=relative)
        if record != source_contracts.get(_SOURCE_CONTRACT_KEYS[relative]):
            raise ReleaseMatrixEvidenceError(
                f"release source contract does not bind {relative}"
            )
        resolvers[relative] = record

    summaries: dict[str, dict[str, object]] = {}
    artifacts = _mapping(evidence.get("artifacts"), name="release artifacts")
    for lane_name, receipt in sorted(receipts.items()):
        lane = LANES_BY_NAME[lane_name]
        if receipt.get("source_revision") != evidence.get("source_revision"):
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} source changed")
        if receipt.get("release_evidence_id") != evidence.get("evidence_id"):
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} evidence changed")
        if receipt.get("artifact_kind") != lane.artifact_kind:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} artifact kind changed")
        if receipt.get("artifact") != artifacts.get(lane.artifact_kind):
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} artifact changed")

        expected_lock = {
            "name": Path(lane.resolver_input).name,
            "sha256": resolvers[lane.resolver_input]["sha256"],
            "size_bytes": resolvers[lane.resolver_input]["size_bytes"],
        }
        if receipt.get("dependency_lock") != expected_lock:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} lock changed")
        profiles = list(
            receipts_dir.rglob(f"numerical-environment-{lane_name}.json")
        )
        resolver_copies = list(receipts_dir.rglob(f"resolver-input-{lane_name}.txt"))
        if len(profiles) != 1 or len(resolver_copies) != 1:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} evidence roster changed")
        copied = _file_record(resolver_copies[0])
        if (
            copied["sha256"] != expected_lock["sha256"]
            or copied["size_bytes"] != expected_lock["size_bytes"]
        ):
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} resolver copy changed")

        profile = _profile(profiles[0], name=f"lane {lane_name} profile")
        profile_file = _file_record(profiles[0])
        numerical = _mapping(
            receipt.get("numerical_environment"),
            name=f"lane {lane_name} numerical evidence",
        )
        expected_numerical = {
            "profile_id": profile.profile_id,
            "runtime_fragment_sha256": profile_file["sha256"],
            "runtime_fragment_size_bytes": profile_file["size_bytes"],
            "installed_distributions_sha256": profile.installed_distributions_sha256,
            "numpy_configuration_sha256": profile.numpy_configuration_sha256,
        }
        if dict(numerical) != expected_numerical:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} profile changed")
        if profile.dependency_lock is None:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} profile lacks lock")
        if profile.dependency_lock.as_dict() != expected_lock:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} profile lock changed")

        python_record = _mapping(receipt.get("python"), name="Python record")
        numpy_record = _mapping(receipt.get("numpy"), name="NumPy record")
        if python_record.get("requested") != lane.python_version:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} Python changed")
        if numpy_record != {"expected": lane.numpy_version, "actual": lane.numpy_version}:
            raise ReleaseMatrixEvidenceError(f"lane {lane_name} NumPy changed")
        summaries[lane_name] = {
            "receipt_id": receipt["receipt_id"],
            "artifact_kind": lane.artifact_kind,
            "artifact_sha256": _mapping(receipt["artifact"], name="artifact")[
                "sha256"
            ],
            "python_version": python_record["actual"],
            "numpy_version": lane.numpy_version,
            "resolver_input": lane.resolver_input,
            "resolver_input_sha256": expected_lock["sha256"],
            "numerical_environment_profile_id": profile.profile_id,
        }

    descriptor: dict[str, object] = {
        "schema": MATRIX_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "source_revision": evidence["source_revision"],
        "release_evidence_id": evidence["evidence_id"],
        "project_version": evidence["project_version"],
        "artifact_sha256": {
            kind: _mapping(artifacts[kind], name=kind)["sha256"]
            for kind in ("wheel", "sdist")
        },
        "resolver_inputs": resolvers,
        "lanes": summaries,
        "claim_boundary": (
            "Cross-version wheel/sdist and numerical-runtime compatibility evidence "
            "only; not publication, deployment, or empirical scientific evidence."
        ),
    }
    return {"matrix_evidence_id": _content_id(descriptor), **descriptor}


def write_matrix_summary(path: Path, evidence: Mapping[str, Any]) -> None:
    lanes = _mapping(evidence.get("lanes"), name="matrix lanes")
    lines = [
        "### Release artifact matrix evidence",
        "",
        f"- Version: `{evidence['project_version']}`",
        f"- Source: `{evidence['source_revision']}`",
        f"- Release evidence: `{evidence['release_evidence_id']}`",
        f"- Matrix evidence: `{evidence['matrix_evidence_id']}`",
        f"- Validated lanes: `{len(lanes)}`",
        "",
        "| Lane | Artifact | Python | NumPy |",
        "| --- | --- | --- | --- |",
    ]
    for name in sorted(lanes):
        lane = _mapping(lanes[name], name=name)
        lines.append(
            f"| `{name}` | `{lane['artifact_kind']}` | "
            f"`{lane['python_version']}` | `{lane['numpy_version']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lane = commands.add_parser("lane")
    lane.add_argument("--lane", required=True)
    lane.add_argument("--release-evidence", type=Path, required=True)
    lane.add_argument("--artifact", type=Path, required=True)
    lane.add_argument("--runtime-fragment", type=Path, required=True)
    lane.add_argument("--resolver-input", type=Path, required=True)
    lane.add_argument("--source-revision", required=True)
    lane.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--release-evidence", type=Path, required=True)
    aggregate.add_argument("--receipts-dir", type=Path, required=True)
    aggregate.add_argument("--project-root", type=Path, default=Path.cwd())
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--summary", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "lane":
            payload = build_lane_receipt(
                lane_name=args.lane,
                release_evidence_path=args.release_evidence,
                artifact_path=args.artifact,
                runtime_fragment_path=args.runtime_fragment,
                resolver_input_path=args.resolver_input,
                source_revision=args.source_revision,
            )
        else:
            payload = build_matrix_evidence(
                release_evidence_path=args.release_evidence,
                receipts_dir=args.receipts_dir,
                project_root=args.project_root,
            )
        _write_json(args.output, payload)
        if args.command == "aggregate" and args.summary is not None:
            write_matrix_summary(args.summary, payload)
    except (ReleaseMatrixEvidenceError, importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
