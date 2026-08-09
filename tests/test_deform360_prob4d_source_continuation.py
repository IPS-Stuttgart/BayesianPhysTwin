from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/continue_deform360_prob4d_source_gate.py"
WORKFLOW = ROOT / ".github/workflows/deform360-prob4d-source-continuation.yml"
LAUNCHER = ROOT / (
    ".github/workflows/launch-deform360-prob4d-source-continuation-once.yml"
)
SPEC = importlib.util.spec_from_file_location("deform360_source_continuation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
continuation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuation
SPEC.loader.exec_module(continuation)

build_supported_stream_records = continuation.build_supported_stream_records
validate_supported_batch_against_lock = (
    continuation.validate_supported_batch_against_lock
)


def _job_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lock() -> dict[str, Any]:
    return {
        "cohort": {
            "exact_object_count": 2,
            "exact_stratum_counts": {"sheet": 1, "volumetric": 1},
            "minimum_metric_streams_per_object": 2,
        }
    }


def _batch(*, technical: bool = False, under_supported: bool = False) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for object_id, stratum in (
        ("object-a", "sheet"),
        ("object-b", "volumetric"),
    ):
        supported = 1 if under_supported and object_id == "object-b" else 3
        for camera_index in range(supported):
            camera_id = f"camera-{camera_index}"
            jobs.append(
                {
                    "job_id": _job_id(f"{object_id}:{camera_id}"),
                    "object_id": object_id,
                    "episode_id": 0,
                    "stratum": stratum,
                    "camera_id": camera_id,
                    "output_relative_directory": f"{object_id}/{camera_id}",
                    "status": "supported",
                    "failure_reason": None,
                }
            )
    jobs.append(
        {
            "job_id": _job_id("object-a:camera-negative"),
            "object_id": "object-a",
            "episode_id": 0,
            "stratum": "sheet",
            "camera_id": "camera-negative",
            "output_relative_directory": "object-a/camera-negative",
            "status": "technical-failure" if technical else "support-negative",
            "failure_reason": (
                "metric-materialization-failed"
                if technical
                else continuation.EXPECTED_SUPPORT_NEGATIVE_REASON
            ),
        }
    )
    supported_count = sum(row["status"] == "supported" for row in jobs)
    support_negative_count = sum(
        row["status"] == "support-negative" for row in jobs
    )
    technical_count = sum(row["status"] == "technical-failure" for row in jobs)
    return {
        "object_count": 2,
        "admitted_stream_count": len(jobs),
        "supported_stream_count": supported_count,
        "support_negative_stream_count": support_negative_count,
        "technical_failure_stream_count": technical_count,
        "supported_object_count": 2,
        "status": (
            "technical-failures-retained"
            if technical
            else "support-negatives-retained"
        ),
        "jobs": jobs,
    }


def test_support_continuation_uses_frozen_object_minimum() -> None:
    result = validate_supported_batch_against_lock(_batch(), _lock())

    assert result["minimum_metric_streams_per_object"] == 2
    assert result["supported_stream_counts_by_object"] == {
        "object-a": 3,
        "object-b": 3,
    }
    assert len(result["support_negatives"]) == 1
    assert result["support_negatives"][0]["reason"] == (
        continuation.EXPECTED_SUPPORT_NEGATIVE_REASON
    )


def test_support_continuation_rejects_object_below_frozen_minimum() -> None:
    with pytest.raises(ValueError, match="fewer supported streams"):
        validate_supported_batch_against_lock(
            _batch(under_supported=True),
            _lock(),
        )


def test_support_continuation_rejects_technical_failure() -> None:
    with pytest.raises(ValueError, match="technical failure"):
        validate_supported_batch_against_lock(_batch(technical=True), _lock())


def test_supported_stream_reconstruction_excludes_negative_rows(
    tmp_path: Path,
) -> None:
    batch = _batch()
    production_jobs = [
        {
            "job_id": row["job_id"],
            "object_id": row["object_id"],
            "episode_id": row["episode_id"],
            "stratum": row["stratum"],
            "camera_id": row["camera_id"],
            "output_relative_directory": row["output_relative_directory"],
        }
        for row in batch["jobs"]
    ]
    production = {"jobs": production_jobs}

    def load_prediction_seal(**arguments: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        row = arguments["row"]
        seal = {
            **row,
            "causal_prefix_frame_range_half_open": [0, 58],
        }
        return seal, {
            "path": f"{row['output_relative_directory']}/prediction.json",
            "sha256": "1" * 64,
            "byte_count": 1,
        }

    def metric_stream_records(**arguments: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        relative = Path(arguments["metric_directory"]).relative_to(
            arguments["metric_root"]
        )
        return (
            {
                "path": (relative / "metric-prefix.npz").as_posix(),
                "sha256": "2" * 64,
                "byte_count": 1,
            },
            {
                "path": (relative / "metric-calibration.json").as_posix(),
                "sha256": "3" * 64,
                "byte_count": 1,
            },
        )

    module = SimpleNamespace(
        _load_prediction_seal=load_prediction_seal,
        _metric_stream_records=metric_stream_records,
    )
    streams = build_supported_stream_records(
        module=module,
        batch=batch,
        production=production,
        production_root=tmp_path,
        prediction_root=tmp_path,
        metric_root=tmp_path,
    )

    assert len(streams) == batch["supported_stream_count"]
    assert all(stream["camera_id"] != "camera-negative" for stream in streams)
    assert all(stream["causal_frame_range_half_open"] == [0, 58] for stream in streams)


def test_continuation_workflow_is_reviewed_main_only_and_target_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert "pull_request:" in text
    assert "workflow_call:" in text
    assert "workflow_dispatch:" not in text
    assert "runs-on: self-hosted" in text
    assert "AUTHORIZED_RUNNER_NAME: workstation2" in text
    assert "SOURCE_METRIC_BATCH_RESULT_ID: f246394c84fd" in text
    assert "SOURCE_METRIC_BATCH_REVISION: ded8910becbb" in text
    assert "minimum_metric_streams_per_object" in text
    assert '"${SOURCE_METRIC_BATCH_ROOT}/metrics"' in text
    assert "adaptive-confirmation" not in text
    assert "confirmation_payloads_opened" in text
    assert "target_outcomes_used" in text
    assert "replacement_allowed" in text


def test_continuation_launcher_is_one_shot_and_has_no_manual_dispatch() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(document, dict)
    assert document["on"] == {
        "push": {
            "branches": ["main"],
            "paths": [
                ".github/workflows/"
                "launch-deform360-prob4d-source-continuation-once.yml"
            ],
        }
    }
    assert "workflow_dispatch:" not in text
    assert "pull_request:" not in text
    assert "uses: ./.github/workflows/deform360-prob4d-source-continuation.yml" in text
    assert "execute_authorized: true" in text
