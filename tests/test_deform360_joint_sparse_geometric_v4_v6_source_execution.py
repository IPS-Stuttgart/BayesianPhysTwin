from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_geometric_common_v4 import (
    METRIC_PLAN_SCHEMA,
    METRIC_PLAN_SEMANTICS,
    METRIC_PLAN_VERSION,
)
from scripts.science import materialize_deform360_v6_source_plan_inputs as module

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "source_prediction_execution.json"
)
WORKFLOW = ROOT / (".github/workflows/deform360-v6-source-prediction-evidence.yml")
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
AMENDMENT_ID = "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
V5_SOURCE_LOCK_ID = "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cohort() -> list[dict[str, Any]]:
    return [
        {
            "object_id": f"object-{index:02d}",
            "episode_id": index,
            "stratum": "sheet" if index < 5 else "volumetric",
        }
        for index in range(10)
    ]


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    results = tmp_path / "results"
    metric_root = results / "metric"
    prediction_root = results / "predictions"
    physical_root = results / "physical"
    for path in (metric_root / "metrics", prediction_root, physical_root):
        path.mkdir(parents=True)

    cases: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in _cohort():
        object_id = row["object_id"]
        episode_id = row["episode_id"]
        camera_ids = [f"camera-{index:02d}" for index in range(6)]
        excluded_camera = camera_ids[-1]
        exclusions.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": row["stratum"],
                "camera_id": excluded_camera,
                "job_id": f"excluded-{object_id}",
                "reason": "frozen-support-negative",
            }
        )
        streams: list[dict[str, Any]] = []
        for camera_id in camera_ids[:-1]:
            view = prediction_root / object_id / camera_id
            archive = view / "baseline_disjoint.npz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(f"archive:{object_id}:{camera_id}".encode())
            manifest = view / "predictions.json"
            _write_json(manifest, {"disjoint_baseline": archive.name})
            metric = metric_root / "metrics" / f"{object_id}-{camera_id}.npz"
            metric.write_bytes(f"metric:{object_id}:{camera_id}".encode())
            streams.append(
                {
                    "camera_id": camera_id,
                    "job_id": f"{object_id}-{camera_id}",
                    "prediction_manifest": {
                        "path": manifest.relative_to(prediction_root).as_posix(),
                        "sha256": _digest(manifest),
                        "byte_count": manifest.stat().st_size,
                    },
                    "metric_prefix": {
                        "path": metric.relative_to(metric_root / "metrics").as_posix(),
                        "sha256": _digest(metric),
                        "byte_count": metric.stat().st_size,
                    },
                    "metric_calibration": {
                        "path": "unused.json",
                        "sha256": "0" * 64,
                        "byte_count": 0,
                    },
                }
            )

        case_name = f"{object_id}-ep{episode_id:04d}"
        physical_directory = physical_root / case_name
        physical_directory.mkdir(parents=True)
        physical_archive = physical_directory / "prediction.npz"
        physical_archive.write_bytes(f"physical:{object_id}".encode())
        _write_json(
            physical_directory / "physical_prediction_manifest.json",
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "physical_mode": "persistence_fallback",
                "physical_prediction_archive": {
                    "file_sha256": _digest(physical_archive)
                },
            },
        )
        cases.append(
            {
                "case_id": case_name,
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": row["stratum"],
                "causal_frame_range_half_open": [0, 58],
                "streams": streams,
            }
        )

    plan = {
        "schema": METRIC_PLAN_SCHEMA,
        "schema_version": METRIC_PLAN_VERSION,
        "semantics": METRIC_PLAN_SEMANTICS,
        "plan_id": "1" * 64,
        "cases": cases,
        "excluded_streams": exclusions,
    }
    _write_json(metric_root / "metric-prefix-plan.json", plan)
    lock = {
        "execution_lock_id": V5_SOURCE_LOCK_ID,
        "cohort": {"development_objects": _cohort()},
    }
    return results, metric_root, prediction_root, physical_root, lock


def test_execution_amendment_is_content_addressed_and_target_closed() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = amendment.pop("amendment_id")

    assert declared == AMENDMENT_ID == content_id(amendment)
    assert amendment["upstream_evidence"] == {
        "supported_objects": 10,
        "technical_failures": 0,
        "v4_artifact_id": 9053374591,
        "v4_development_report_id": (
            "d7548059971ce0b7836240ec7d5dac4fc53776a613fff3fc1f6cf600b87e141c"
        ),
        "v4_materialization_id": (
            "9333d9970ad6ff8f9b3f32e1b2f922ad35649debb81e26b5bedc10f3be2ba7f6"
        ),
        "v4_status": "development-design-supported",
        "v4_workflow_run_id": 31363421947,
    }
    boundary = amendment["information_boundary"]
    assert not any(boundary.values())
    assert amendment["execution"]["runner_name"] == "workstation2"
    assert amendment["execution"]["v5_nested_prediction_record_count"] == 100
    assert amendment["source_cohort"]["object_count"] == 10
    assert amendment["source_cohort"]["replacement_allowed"] is False


def test_plan_materializer_uses_only_frozen_supported_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, metric_root, prediction_root, physical_root, lock = _fixture_roots(
        tmp_path
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "load_deform360_joint_sparse_source_execution_lock_v5",
        lambda _path: lock,
    )

    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema": "fixture-plan", "objects": kwargs["objects"]}

    monkeypatch.setattr(
        module,
        "build_deform360_joint_sparse_source_prediction_plan_v5",
        fake_builder,
    )
    output = tmp_path / "source-plan-inputs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize",
            "--execution-lock",
            "unused-lock.json",
            "--execution-amendment",
            str(AMENDMENT),
            "--metric-batch-root",
            str(metric_root),
            "--prediction-root",
            str(prediction_root),
            "--physical-work-root",
            str(physical_root),
            "--results-root",
            str(results),
            "--implementation-revision",
            "3" * 40,
            "--output",
            str(output),
        ],
    )

    assert module.main() == 0
    assert len(captured["objects"]) == 10
    for row in captured["objects"]:
        all_cameras = tuple(row["all_camera_ids"])
        reserved = tuple(row["reserved_endpoint_camera_ids"])
        expected = select_reserved_endpoint_views_v5(
            row["object_id"], all_cameras, count=2
        )
        assert reserved == expected
        assert 2 <= len(row["visual_windows"]) <= 8
        assert not (
            {value["camera_id"] for value in row["visual_windows"]} & set(reserved)
        )
        assert row["contact_prefix"] == {
            "status": "unavailable",
            "path": None,
            "manifest_file_sha256": None,
            "materialization_id": None,
            "unavailable_reason": ("released-tactile-robot-axis-identity-unavailable"),
        }
        for visual in row["visual_windows"]:
            assert not Path(visual["decoded_uniform"]["path"]).is_absolute()
            assert not Path(visual["metric_prefix"]["path"]).is_absolute()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["execution_amendment_id"] == AMENDMENT_ID
    assert result["information_boundary"] == {
        "development_suffix_opened": False,
        "future_object_observations_used_for_prediction": False,
        "v5_confirmation_payloads_opened": False,
        "v6_target_payloads_opened": False,
        "target_outcomes_used": False,
    }


def test_amendment_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    payload["information_boundary"]["v6_target_payloads_opened"] = True
    path = tmp_path / "changed.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="amendment identity changed"):
        module._load_amendment(path)


def test_workflow_runs_science_only_after_merge_on_workstation2() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    evidence = workflow["jobs"]["evidence"]
    assert evidence["runs-on"] == ["self-hosted", "Linux", "X64", "nvidia-smi"]
    assert "github.event_name == 'push'" in evidence["if"]
    assert "github.ref == 'refs/heads/main'" in evidence["if"]
    assert evidence["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "write",
    }
    assert "workflow_dispatch:" not in text
    assert "contents: write" not in text
    assert "git push" not in text


def test_runner_seals_predictions_without_opening_source_suffix() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'prediction_record_count") != 100' in text
    assert "source-prediction-evidence-sealed" in text
    assert "source-inputs-incomplete" in text
    assert "source-technical-failure-retained" in text
    assert "run_deform360_joint_sparse_source_predictions_v5.py" in text
    assert "run_deform360_fresh_object_session_source_v6.py" not in text
    assert 'development_suffix_opened": False' in text
    assert 'v6_target_payloads_opened": False' in text
    assert 'fresh_target_selection_authorized": False' in text
