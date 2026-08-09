"""Independently validate the frozen Deform360 Prob4D support-negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_HEAD_SHA = "ded8910becbbffe958dfd18c84ad91069e7087a4"
PRODUCTION_RESULT_ID = (
    "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89"
)
ADMISSION_ID = "715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15"
PROB4D_REVISION = "25d90ef7f78ba4307f4555cb636d666004e1bf66"
MOTIONCRAFTER_REVISION = "9cb4e9679f5f34e249945544052464ef46324bc2"
METRIC_RESULT_ID = "f246394c84fd643b6ec8961dbcb2101a73c34e46d5eaf43961f28429aeb197eb"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

SOURCE_FILES = {
    "metric-support/metric-batch-result.json": (
        "679550aff53d3b615f63c66ee78318258893867511dd6c33100d1cf10c0f5be6"
    ),
    "metric-support/support-receipt.json": (
        "2c14774dd0f0f96301483a46da148de392442794329da6dcb97dd61e3ca7e07f"
    ),
    "pipeline-receipt.json": (
        "8588f6e7b3115808c49cc781a27093308b6011f528e3aabfae936daa17994dfd"
    ),
}
SOURCE_ARTIFACTS = {
    "metric-prior-policy.json": (
        "e2a88201a2a5a6a7d47f94994f7a9a9e2a5923d973ba09e11d49a813be0de0a7"
    ),
    "prepared-source-inventory.json": (
        "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
    ),
    "selection.json": (
        "4dd12af9889d64976095eb9e237eeb655f9675ff7d5940aa5dfc1d4ee11f295c"
    ),
    "visual-production-result.json": (
        "7af3d3bc17bfbe923c4c3754ff20179fd1d51417dd8a56d8475e4eecac66879b"
    ),
    "visual-provider-spec.json": (
        "9758ce5b358096ae83e8845c01abeb4f1e324b619929d250c4062bb14cba1cf8"
    ),
}
NEGATIVE_STREAMS = {
    ("026-sock-cloth", "brics-odroid-002_cam0"),
    ("036-napkin-cloth", "brics-odroid-025_cam0"),
    ("058-roll-napkin", "brics-odroid-002_cam0"),
    ("058-roll-napkin", "brics-odroid-007_cam1"),
    ("152-slime", "brics-odroid-002_cam0"),
    ("152-slime", "brics-odroid-012_cam1"),
    ("152-slime", "brics-odroid-016_cam0"),
    ("153-cake", "brics-odroid-002_cam0"),
    ("153-cake", "brics-odroid-016_cam0"),
    ("167-glove-gray-cloth", "brics-odroid-002_cam0"),
    ("167-glove-gray-cloth", "brics-odroid-007_cam1"),
}
BOUNDARY = {
    "calibration_robot_state_access_attempted": True,
    "calibration_robot_state_opened": True,
    "calibration_camera_calibration_opened": True,
    "calibration_camera_images_opened": False,
    "calibration_tactile_payloads_opened": False,
    "rendered_depth_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "future_frames_used": False,
    "replacement_allowed": False,
    "human_approval_required": False,
}
CLAIM_BOUNDARY = (
    "Source-only materialization of released Deform360 robot geometry for every "
    "sealed calibration camera. This artifact does not use new capture, require "
    "human approval, open confirmation payloads or future frames, evaluate "
    "calibration or transfer, authorize confirmation, or establish state of the art."
)
RESULT_FIELDS = {
    "schema",
    "schema_version",
    "semantics",
    "result_id",
    "implementation_revision",
    "production_result_id",
    "admission_id",
    "object_count",
    "admitted_stream_count",
    "supported_stream_count",
    "support_negative_stream_count",
    "technical_failure_stream_count",
    "supported_object_count",
    "plan_emitted",
    "plan_file",
    "status",
    "jobs",
    "source_artifacts",
    "information_boundary",
    "claim_boundary",
}
JOB_FIELDS = {
    "job_id",
    "object_id",
    "episode_id",
    "stratum",
    "camera_id",
    "output_relative_directory",
    "status",
    "metric_artifact_id",
    "projected_point_count",
    "failure_reason",
    "failure_detail_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _canonical_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}") from error
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value


def _hex(value: object, *, length: int, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be hexadecimal")
    _require(len(value) == length, f"{name} has the wrong length")
    _require(set(value) <= set("0123456789abcdef"), f"{name} is not lowercase hex")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="ascii",
    )


def _validate_pipeline(source: Path) -> dict[str, Any]:
    pipeline = _load(source / "pipeline-receipt.json", name="pipeline receipt")
    expected = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-pipeline-receipt",
        "schema_version": 1,
        "implementation_revision": SOURCE_HEAD_SHA,
        "visual_production_result_id": PRODUCTION_RESULT_ID,
        "prob4d_revision": PROB4D_REVISION,
        "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        "stage_outcomes": {
            "metric_batch": "success",
            "support_gate": "failure",
            "samples": "skipped",
            "calibration": "skipped",
            "source_gate": "skipped",
        },
        "stderr_sha256": {
            "metric-batch": EMPTY_SHA256,
            "samples": EMPTY_SHA256,
            "calibration": EMPTY_SHA256,
            "source-gate": EMPTY_SHA256,
        },
        "source_gate_result_id": None,
        "source_gate_passed": None,
        "confirmation_access_authorized": None,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "future_frames_used": False,
            "human_approval_required": False,
            "new_measurements_required": False,
            "public_released_measurements_used": True,
            "replacement_allowed": False,
            "target_outcomes_used": False,
        },
    }
    _require(pipeline == expected, "pipeline is not the frozen support terminal")
    return pipeline


def _validate_metric(source: Path) -> tuple[dict[str, Any], str]:
    metric = _load(
        source / "metric-support/metric-batch-result.json", name="metric result"
    )
    _require(set(metric) == RESULT_FIELDS, "metric result fields changed")
    identity = dict(metric)
    result_id = _hex(identity.pop("result_id"), length=64, name="metric result ID")
    _require(
        _canonical_id(identity) == result_id == METRIC_RESULT_ID,
        "result ID changed",
    )
    expected_scalars = {
        "schema": "bayesian-phystwin.deform360-prob4d-metric-batch",
        "schema_version": 1,
        "semantics": "all-sealed-calibration-streams-released-robot-gauge-v1",
        "implementation_revision": SOURCE_HEAD_SHA,
        "production_result_id": PRODUCTION_RESULT_ID,
        "admission_id": ADMISSION_ID,
        "object_count": 10,
        "admitted_stream_count": 324,
        "supported_stream_count": 313,
        "support_negative_stream_count": 11,
        "technical_failure_stream_count": 0,
        "supported_object_count": 10,
        "plan_emitted": False,
        "plan_file": None,
        "status": "support-negatives-retained",
        "source_artifacts": SOURCE_ARTIFACTS,
        "information_boundary": BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field, expected in expected_scalars.items():
        _require(metric[field] == expected, f"metric result changed: {field}")

    jobs = metric["jobs"]
    _require(
        isinstance(jobs, Sequence) and not isinstance(jobs, (str, bytes)),
        "metric jobs are invalid",
    )
    _require(len(jobs) == 324, "metric job roster changed")
    counts: Counter[str] = Counter()
    order: list[tuple[str, str, str]] = []
    supported_objects: set[str] = set()
    negative_streams: set[tuple[str, str]] = set()
    job_ids: set[str] = set()
    for index, row in enumerate(jobs):
        _require(isinstance(row, dict), f"metric job {index} is invalid")
        _require(set(row) == JOB_FIELDS, f"metric job {index} fields changed")
        job_id = _hex(row["job_id"], length=64, name=f"metric job {index} ID")
        _require(job_id not in job_ids, "metric job ID repeated")
        job_ids.add(job_id)
        object_id = row["object_id"]
        camera_id = row["camera_id"]
        _require(isinstance(object_id, str) and object_id, "invalid object ID")
        _require(isinstance(camera_id, str) and camera_id, "invalid camera ID")
        _require(
            type(row["episode_id"]) is int and row["episode_id"] >= 0,
            "invalid episode",
        )
        _require(isinstance(row["stratum"], str) and row["stratum"], "invalid stratum")
        _require(
            isinstance(row["output_relative_directory"], str)
            and row["output_relative_directory"]
            and ".." not in Path(row["output_relative_directory"]).parts,
            "invalid output directory",
        )
        order.append((object_id, camera_id, job_id))
        status = row["status"]
        _require(status in {"supported", "support-negative"}, "unexpected job status")
        counts[status] += 1
        if status == "supported":
            supported_objects.add(object_id)
            _hex(row["metric_artifact_id"], length=64, name="metric artifact ID")
            _require(
                type(row["projected_point_count"]) is int
                and row["projected_point_count"] > 0
                and row["failure_reason"] is None
                and row["failure_detail_sha256"] is None,
                "supported stream contains failure state",
            )
        else:
            negative_streams.add((object_id, camera_id))
            _require(
                row["metric_artifact_id"] is None
                and row["projected_point_count"] == 0
                and row["failure_reason"]
                == "released-robot-geometry-outside-fixed-camera-prefix"
                and row["failure_detail_sha256"] is None,
                "support-negative stream changed",
            )
    _require(order == sorted(order), "metric jobs are not sorted")
    _require(counts == {"supported": 313, "support-negative": 11}, "counts changed")
    _require(len(supported_objects) == 10, "supported object roster changed")
    _require(negative_streams == NEGATIVE_STREAMS, "support-negative roster changed")
    roster_id = _canonical_id(
        {
            "support_negative_streams": [
                {"object_id": object_id, "camera_id": camera_id}
                for object_id, camera_id in sorted(negative_streams)
            ]
        }
    )
    return metric, roster_id


def _validate_support_receipt(source: Path) -> dict[str, Any]:
    receipt = _load(
        source / "metric-support/support-receipt.json", name="support receipt"
    )
    expected = {
        "schema": "bayesian-phystwin.deform360-prob4d-source-support-receipt",
        "schema_version": 1,
        "implementation_revision": SOURCE_HEAD_SHA,
        "metric_batch_result_id": METRIC_RESULT_ID,
        "metric_batch_status": "support-negatives-retained",
        "metric_batch_step_outcome": "success",
        "metric_batch_stderr_sha256": EMPTY_SHA256,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "new_measurements_required": False,
        "human_approval_required": False,
    }
    _require(receipt == expected, "support receipt changed")
    return receipt


def audit_support_negative(
    *,
    source_root: str | Path,
    output_directory: str | Path,
    source_run_id: int,
    source_run_attempt: int,
    source_artifact_id: int,
    source_artifact_name: str,
    source_artifact_digest: str,
    auditor_revision: str,
    auditor_workflow: str | Path,
) -> dict[str, Any]:
    """Validate and copy the immutable compact early-terminal evidence."""

    source = Path(source_root).resolve()
    output = Path(output_directory).resolve()
    workflow = Path(auditor_workflow).resolve()
    _require(source.is_dir(), "source artifact directory is missing")
    _require(workflow.is_file(), "auditor workflow is missing")
    _hex(auditor_revision, length=40, name="auditor revision")
    _require(source_run_id == 31297018948, "source run changed")
    _require(source_run_attempt == 1, "source run attempt changed")
    _require(source_artifact_id == 9033414269, "source artifact changed")
    _require(
        source_artifact_name == "deform360-prob4d-source-gate-31297018948-1",
        "source artifact name changed",
    )
    _require(
        source_artifact_digest
        == "sha256:7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de",
        "source artifact digest changed",
    )
    _require(not output.exists(), "audit output already exists")
    output.mkdir(parents=True)
    receipt: dict[str, object] = {
        "schema": (
            "bayesian-phystwin.deform360-prob4d-support-negative-independent-audit"
        ),
        "schema_version": 1,
        "audit_status": "invalid",
        "terminal_stage": None,
        "source_workflow_run_id": source_run_id,
        "source_workflow_run_attempt": source_run_attempt,
        "source_workflow_conclusion": "failure",
        "source_workflow_head_sha": SOURCE_HEAD_SHA,
        "source_artifact_id": source_artifact_id,
        "source_artifact_name": source_artifact_name,
        "source_artifact_digest": source_artifact_digest,
        "auditor_revision": auditor_revision,
        "source_gate_result_id": None,
        "source_gate_passed": False,
        "confirmation_access_authorized": False,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
    }
    try:
        _require(
            not any(path.is_symlink() for path in source.rglob("*")),
            "source artifact contains a symbolic link",
        )
        observed = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file()
        }
        _require(
            observed == {"SHA256SUMS", *SOURCE_FILES},
            "source file roster changed",
        )
        checksum_lines = [
            f"{digest}  {relative}" for relative, digest in sorted(SOURCE_FILES.items())
        ]
        _require(
            (source / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            == checksum_lines,
            "source checksum manifest changed",
        )
        for relative, expected in SOURCE_FILES.items():
            _require(_sha256_file(source / relative) == expected, f"{relative} changed")

        _validate_pipeline(source)
        metric, roster_id = _validate_metric(source)
        _validate_support_receipt(source)
        shutil.copytree(source / "metric-support", output / "metric-support")
        shutil.copy2(source / "pipeline-receipt.json", output / "pipeline-receipt.json")
        receipt.update(
            {
                "audit_status": "validated-negative",
                "terminal_stage": "support-gate",
                "metric_batch_result_id": METRIC_RESULT_ID,
                "metric_batch_status": metric["status"],
                "admitted_stream_count": 324,
                "supported_stream_count": 313,
                "support_negative_stream_count": 11,
                "technical_failure_stream_count": 0,
                "support_negative_roster_id": roster_id,
                "source_artifact_checksums_sha256": _sha256_file(
                    source / "SHA256SUMS"
                ),
                "auditor_workflow_sha256": _sha256_file(workflow),
            }
        )
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        receipt["error_sha256"] = hashlib.sha256(
            str(error).encode("utf-8")
        ).hexdigest()
        _write_json(output / "independent-support-audit-receipt.json", receipt)
        _write_checksums(output)
        raise

    _write_json(output / "independent-support-audit-receipt.json", receipt)
    _write_checksums(output)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument("--source-run-attempt", required=True, type=int)
    parser.add_argument("--source-artifact-id", required=True, type=int)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--auditor-revision", required=True)
    parser.add_argument("--auditor-workflow", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = audit_support_negative(
        source_root=arguments.source_root,
        output_directory=arguments.output_directory,
        source_run_id=arguments.source_run_id,
        source_run_attempt=arguments.source_run_attempt,
        source_artifact_id=arguments.source_artifact_id,
        source_artifact_name=arguments.source_artifact_name,
        source_artifact_digest=arguments.source_artifact_digest,
        auditor_revision=arguments.auditor_revision,
        auditor_workflow=arguments.auditor_workflow,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
