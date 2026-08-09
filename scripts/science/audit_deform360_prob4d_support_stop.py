#!/usr/bin/env python3
"""Audit a terminal pre-calibration Deform360 Prob4D support stop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from bayesian_phystwin._canonical_contracts import genuine_integer, plain_json
from bayesian_phystwin._portable_contracts import (
    exact_revision,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)
from bayesian_phystwin.deform360_public_contact_prefix import (
    _ordinary_directory,
    _ordinary_file,
)

AUDIT_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-support-stop-audit"
AUDIT_VERSION: Final = 1
PIPELINE_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt"
SUPPORT_SCHEMA: Final = "bayesian-phystwin.deform360-prob4d-source-support-receipt"
SUPPORT_STOP_STAGES: Final = {
    "metric_batch": "success",
    "support_gate": "failure",
    "samples": "skipped",
    "calibration": "skipped",
    "source_gate": "skipped",
}
CLOSED_BOUNDARY: Final = {
    "public_released_measurements_used": True,
    "new_measurements_required": False,
    "human_approval_required": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
}
CLAIM_BOUNDARY: Final = (
    "This audit establishes only that the frozen calibration-source pipeline "
    "stopped at its preregistered all-stream support gate with every stream "
    "accounted for and confirmation closed. It establishes no calibrated "
    "uncertainty, physical-query benefit, deployment safety, or state of the art."
)

_PIPELINE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "visual_production_result_id",
        "prob4d_revision",
        "motioncrafter_revision",
        "stage_outcomes",
        "stderr_sha256",
        "source_gate_result_id",
        "source_gate_passed",
        "confirmation_access_authorized",
        "information_boundary",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_revision",
        "metric_batch_step_outcome",
        "metric_batch_result_id",
        "metric_batch_status",
        "metric_batch_stderr_sha256",
        "new_measurements_required",
        "human_approval_required",
        "confirmation_payloads_opened",
        "target_outcomes_used",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(plain_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(root: Path) -> None:
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in members
    ]
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def _validate_checksums(root: Path) -> str:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("source artifact contains a symbolic link")
    checksum = _ordinary_file(root / "SHA256SUMS", name="source SHA256SUMS")
    observed = checksum.read_text(encoding="ascii").splitlines()
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    expected = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in members
    ]
    if observed != expected:
        raise ValueError("source artifact checksums changed")
    return _sha256(checksum)


def _load_metric_validator(repository: Path):  # type: ignore[no-untyped-def]
    script = _ordinary_file(
        repository / "scripts/science/materialize_deform360_prob4d_metric_batch.py",
        name="metric-batch validator",
    )
    spec = importlib.util.spec_from_file_location(
        "support_stop_metric_validator", script
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load metric-batch validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_deform360_prob4d_metric_batch", None)
    if not callable(validator):
        raise ValueError("metric-batch validator is missing")
    return validator


def _validate_metric_batch(source: Path, repository: Path) -> dict[str, Any]:
    support = _ordinary_directory(source / "metric-support", name="metric support")
    expected = {"metric-batch-result.json", "support-receipt.json"}
    observed = {path.name for path in support.iterdir()}
    if observed != expected:
        raise ValueError("compact metric-support roster changed")
    with tempfile.TemporaryDirectory(prefix="deform360-support-stop-") as temporary:
        staged = Path(temporary)
        shutil.copy2(support / "metric-batch-result.json", staged)
        _write_checksums(staged)
        result = _load_metric_validator(repository)(staged)
    if not isinstance(result, dict):
        raise ValueError("metric-batch validator returned a non-object")
    return result


def audit_support_stop(
    *,
    source_root: str | Path,
    output_directory: str | Path,
    repository_root: str | Path,
    source_run_id: int,
    source_run_attempt: int,
    source_run_conclusion: str,
    source_head_sha: str,
    source_artifact_id: int,
    source_artifact_name: str,
    auditor_revision: str,
    expected_production_result_id: str,
    expected_admission_id: str,
    expected_prob4d_revision: str,
    expected_motioncrafter_revision: str,
    expected_object_count: int,
    expected_admitted_stream_count: int,
) -> dict[str, Any]:
    """Validate and publish one immutable support-negative audit bundle."""

    source = _ordinary_directory(source_root, name="compact source artifact")
    repository = _ordinary_directory(repository_root, name="auditor repository")
    output = Path(output_directory).absolute()
    parent = _ordinary_directory(output.parent, name="audit output parent")
    output = parent / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit output already exists: {output}")

    run_id = genuine_integer(source_run_id, name="source run ID", minimum=1)
    run_attempt = genuine_integer(
        source_run_attempt, name="source run attempt", minimum=1
    )
    if run_attempt != 1:
        raise ValueError("only the first source workflow attempt is admissible")
    conclusion = nonempty_string(source_run_conclusion, name="source run conclusion")
    if conclusion != "failure":
        raise ValueError("a support stop must have workflow conclusion failure")
    head = exact_revision(source_head_sha, name="source head revision")
    auditor = exact_revision(auditor_revision, name="auditor revision")
    artifact_id = genuine_integer(
        source_artifact_id, name="source artifact ID", minimum=1
    )
    artifact_name = nonempty_string(source_artifact_name, name="source artifact name")
    if artifact_name != f"deform360-prob4d-source-gate-{run_id}-{run_attempt}":
        raise ValueError("source artifact name changed")

    expected_production = sha256_digest(
        expected_production_result_id, name="expected production result ID"
    )
    expected_admission = sha256_digest(
        expected_admission_id, name="expected admission ID"
    )
    expected_prob4d = exact_revision(
        expected_prob4d_revision, name="expected Prob4D revision"
    )
    expected_motioncrafter = exact_revision(
        expected_motioncrafter_revision, name="expected MotionCrafter revision"
    )
    expected_objects = genuine_integer(
        expected_object_count, name="expected object count", minimum=1
    )
    expected_streams = genuine_integer(
        expected_admitted_stream_count, name="expected admitted stream count", minimum=1
    )

    receipt: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_VERSION,
        "audit_status": "invalid",
        "source_workflow_run_id": run_id,
        "source_workflow_run_attempt": run_attempt,
        "source_workflow_conclusion": conclusion,
        "source_workflow_head_sha": head,
        "source_artifact_id": artifact_id,
        "source_artifact_name": artifact_name,
        "auditor_revision": auditor,
        "confirmation_access_authorized": False,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output.mkdir()
    try:
        receipt["source_sha256sums_sha256"] = _validate_checksums(source)
        pipeline_path = source / "pipeline-receipt.json"
        pipeline = _load_json(pipeline_path, name="pipeline receipt")
        require_exact_fields(
            pipeline, expected=_PIPELINE_FIELDS, name="pipeline receipt"
        )
        if not (
            pipeline["schema"] == PIPELINE_SCHEMA
            and pipeline["schema_version"] == 1
        ):
            raise ValueError("pipeline receipt contract changed")
        if (
            exact_revision(
                pipeline["implementation_revision"], name="pipeline revision"
            )
            != head
        ):
            raise ValueError("pipeline revision changed")
        if (
            sha256_digest(
                pipeline["visual_production_result_id"], name="production ID"
            )
            != expected_production
        ):
            raise ValueError("visual production result changed")
        if (
            exact_revision(pipeline["prob4d_revision"], name="Prob4D revision")
            != expected_prob4d
        ):
            raise ValueError("Prob4D revision changed")
        if (
            exact_revision(
                pipeline["motioncrafter_revision"], name="MotionCrafter revision"
            )
            != expected_motioncrafter
        ):
            raise ValueError("MotionCrafter revision changed")
        if (
            dict(_mapping(pipeline["stage_outcomes"], name="stage outcomes"))
            != SUPPORT_STOP_STAGES
        ):
            raise ValueError("pipeline is not a terminal support stop")
        if (
            dict(
                _mapping(
                    pipeline["information_boundary"], name="pipeline boundary"
                )
            )
            != CLOSED_BOUNDARY
        ):
            raise ValueError("pipeline information boundary changed")
        if any(
            pipeline[field] is not None
            for field in (
                "source_gate_result_id",
                "source_gate_passed",
                "confirmation_access_authorized",
            )
        ):
            raise ValueError("skipped source gate contains a decision")
        stderr = _mapping(pipeline["stderr_sha256"], name="stderr digests")
        if set(stderr) != {"metric-batch", "samples", "calibration", "source-gate"}:
            raise ValueError("pipeline stderr roster changed")
        for name, digest in stderr.items():
            sha256_digest(digest, name=f"stderr digest {name}")

        result = _validate_metric_batch(source, repository)
        exact = {
            "implementation_revision": head,
            "production_result_id": expected_production,
            "admission_id": expected_admission,
            "object_count": expected_objects,
            "admitted_stream_count": expected_streams,
            "supported_object_count": expected_objects,
            "plan_emitted": False,
            "plan_file": None,
        }
        if {key: result.get(key) for key in exact} != exact:
            raise ValueError("metric-batch frozen identity or cohort changed")
        supported = genuine_integer(
            result["supported_stream_count"], name="supported stream count", minimum=0
        )
        support_negative = genuine_integer(
            result["support_negative_stream_count"],
            name="support-negative stream count",
            minimum=0,
        )
        technical = genuine_integer(
            result["technical_failure_stream_count"],
            name="technical-failure stream count",
            minimum=0,
        )
        if supported + support_negative + technical != expected_streams:
            raise ValueError("metric-batch stream accounting changed")
        if support_negative == 0 and technical == 0:
            raise ValueError("failed support gate contains no retained negative")
        expected_status = (
            "technical-failures-retained" if technical else "support-negatives-retained"
        )
        if result["status"] != expected_status:
            raise ValueError("metric-batch terminal status changed")

        support_path = source / "metric-support/support-receipt.json"
        support = _load_json(support_path, name="support receipt")
        require_exact_fields(support, expected=_SUPPORT_FIELDS, name="support receipt")
        expected_support = {
            "schema": SUPPORT_SCHEMA,
            "schema_version": 1,
            "implementation_revision": head,
            "metric_batch_step_outcome": "success",
            "metric_batch_result_id": result["result_id"],
            "metric_batch_status": result["status"],
            "metric_batch_stderr_sha256": stderr["metric-batch"],
            "new_measurements_required": False,
            "human_approval_required": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        }
        if support != expected_support:
            raise ValueError("support receipt does not bind the frozen metric batch")

        receipt.update(
            {
                "audit_status": (
                    "validated-technical-negative"
                    if technical
                    else "validated-support-negative"
                ),
                "metric_batch_result_id": result["result_id"],
                "metric_batch_status": result["status"],
                "admitted_stream_count": expected_streams,
                "supported_stream_count": supported,
                "support_negative_stream_count": support_negative,
                "technical_failure_stream_count": technical,
                "supported_object_count": expected_objects,
                "pipeline_receipt_sha256": _sha256(pipeline_path),
                "support_receipt_sha256": _sha256(support_path),
                "metric_batch_result_sha256": _sha256(
                    source / "metric-support/metric-batch-result.json"
                ),
            }
        )
        shutil.copy2(pipeline_path, output)
        shutil.copytree(source / "metric-support", output / "metric-support")
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        receipt["error_sha256"] = hashlib.sha256(str(error).encode("utf-8")).hexdigest()

    _write_json(output / "independent-audit-receipt.json", receipt)
    _write_checksums(output)
    if receipt["audit_status"] == "invalid":
        raise ValueError("support-stop audit is invalid")
    return cast(dict[str, Any], plain_json(receipt))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument("--source-run-attempt", required=True, type=int)
    parser.add_argument("--source-run-conclusion", required=True)
    parser.add_argument("--source-head-sha", required=True)
    parser.add_argument("--source-artifact-id", required=True, type=int)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--auditor-revision", required=True)
    parser.add_argument("--expected-production-result-id", required=True)
    parser.add_argument("--expected-admission-id", required=True)
    parser.add_argument("--expected-prob4d-revision", required=True)
    parser.add_argument("--expected-motioncrafter-revision", required=True)
    parser.add_argument("--expected-object-count", required=True, type=int)
    parser.add_argument("--expected-admitted-stream-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        audit_support_stop(
            source_root=arguments.source_root,
            output_directory=arguments.output_dir,
            repository_root=arguments.repository_root,
            source_run_id=arguments.source_run_id,
            source_run_attempt=arguments.source_run_attempt,
            source_run_conclusion=arguments.source_run_conclusion,
            source_head_sha=arguments.source_head_sha,
            source_artifact_id=arguments.source_artifact_id,
            source_artifact_name=arguments.source_artifact_name,
            auditor_revision=arguments.auditor_revision,
            expected_production_result_id=arguments.expected_production_result_id,
            expected_admission_id=arguments.expected_admission_id,
            expected_prob4d_revision=arguments.expected_prob4d_revision,
            expected_motioncrafter_revision=arguments.expected_motioncrafter_revision,
            expected_object_count=arguments.expected_object_count,
            expected_admitted_stream_count=arguments.expected_admitted_stream_count,
        )
    except (OSError, ValueError) as error:
        print(f"support-stop audit failed: {type(error).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
