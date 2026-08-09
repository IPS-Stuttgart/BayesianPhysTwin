from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.deform360_prob4d_metric_batch as metric_batch
from bayesian_phystwin.deform360_prob4d_metric_batch import (
    METRIC_BATCH_RESULT_FILENAME,
    METRIC_DIRECTORY_NAME,
    METRIC_PREFIX_PLAN_FILENAME,
    SUPPORT_NEGATIVE_DETAIL,
    materialize_deform360_prob4d_metric_batch,
    validate_deform360_prob4d_metric_batch,
)

PROCESSING_REVISION = "1" * 40
IMPLEMENTATION_REVISION = "2" * 40
PROB4D_REVISION = "3" * 40
MOTIONCRAFTER_REVISION = "4" * 40
DATASET_REVISION = "5" * 40
PRODUCTION_RESULT_ID = "6" * 64
ADMISSION_ID = "7" * 64
VISUAL_LOCK_ID = "8" * 64
MODEL_SET_ID = "9" * 64
PROTOCOL_ID = "public-source-protocol"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _job_id(object_id: str, camera_id: str) -> str:
    return hashlib.sha256(f"{object_id}:{camera_id}".encode()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Any]:
    production_root = tmp_path / "production"
    prediction_root = tmp_path / "predictions"
    processed_root = tmp_path / "processed"
    production_root.mkdir()
    prediction_root.mkdir()
    processed_root.mkdir()
    selected_rows = []
    inventory_rows = []
    production_jobs = []
    for object_index, stratum in enumerate(("sheet", "volumetric")):
        object_id = f"object-{object_index}"
        episode_id = object_index
        selected_rows.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
            }
        )
        inventory_rows.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
            }
        )
        for camera_index in range(2):
            camera_id = f"camera-{camera_index}"
            job_id = _job_id(object_id, camera_id)
            relative = (
                Path("objects") / object_id / f"episode_{episode_id:04d}" / camera_id
            )
            prediction_manifest = prediction_root / relative / "prediction.json"
            _write_json(prediction_manifest, {"job_id": job_id})
            seal = {
                "job_id": job_id,
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "camera_id": camera_id,
                "admission_id": ADMISSION_ID,
                "implementation_revision": IMPLEMENTATION_REVISION,
                "provider_revision": PROB4D_REVISION,
                "motioncrafter_revision": MOTIONCRAFTER_REVISION,
                "visual_provider_lock_id": VISUAL_LOCK_ID,
                "model_set_id": MODEL_SET_ID,
                "output_relative_directory": relative.as_posix(),
                "causal_prefix_frame_range_half_open": [0, 58],
                "prediction_manifest": {
                    "path": "prediction.json",
                    "sha256": _sha256(prediction_manifest),
                    "byte_count": prediction_manifest.stat().st_size,
                },
            }
            seal_path = production_root / relative / "prediction-seal.json"
            _write_json(seal_path, seal)
            production_jobs.append(
                {
                    "job_id": job_id,
                    "object_id": object_id,
                    "camera_id": camera_id,
                    "status": "succeeded",
                    "receipt": _record(seal_path, root=production_root),
                }
            )
    production = {
        "result_id": PRODUCTION_RESULT_ID,
        "admission_id": ADMISSION_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "provider_revision": PROB4D_REVISION,
        "motioncrafter_revision": MOTIONCRAFTER_REVISION,
        "visual_provider_lock_id": VISUAL_LOCK_ID,
        "model_set_id": MODEL_SET_ID,
        "object_count": 2,
        "camera_view_count": 4,
        "succeeded_job_count": 4,
        "technical_failure_job_count": 0,
        "status": "all-jobs-succeeded",
        "jobs": production_jobs,
    }
    production_path = tmp_path / "visual-production-result.json"
    _write_json(production_path, production)
    inventory = {
        "processing_revision": PROCESSING_REVISION,
        "objects": inventory_rows,
    }
    inventory_path = tmp_path / "prepared-source-inventory.json"
    _write_json(inventory_path, inventory)
    selection = {
        "protocol_id": PROTOCOL_ID,
        "dataset": {"resolved_revision": DATASET_REVISION},
        "selection": {"calibration": selected_rows},
    }
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    provider = {
        "protocol_id": PROTOCOL_ID,
        "provider": {"revision": PROB4D_REVISION},
        "motioncrafter": {
            "revision": MOTIONCRAFTER_REVISION,
            "height": 160,
            "width": 320,
        },
    }
    provider_path = tmp_path / "visual-provider.json"
    _write_json(provider_path, provider)
    policy = {
        "protocol_id": PROTOCOL_ID,
        "metric_source_kind": "released-deform360-robot-taxel-gauge-v1",
        "future_frames_used": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "human_approval_required": False,
    }
    policy_path = tmp_path / "metric-policy.json"
    _write_json(policy_path, policy)
    return {
        "prepared_source_inventory_path": inventory_path,
        "production_result_path": production_path,
        "production_root": production_root,
        "prediction_root": prediction_root,
        "processed_root": processed_root,
        "selection_path": selection_path,
        "visual_provider_spec_path": provider_path,
        "metric_prior_policy_path": policy_path,
        "expected_processing_revision": PROCESSING_REVISION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "output_directory": tmp_path / "batch",
    }


def _install_contract_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metric_batch,
        "validate_deform360_calibration_visual_production_result",
        lambda value: value,
    )
    monkeypatch.setattr(
        metric_batch,
        "validate_deform360_prepared_source_inventory",
        lambda value: value,
    )
    monkeypatch.setattr(
        metric_batch,
        "validate_deform360_calibration_visual_prediction_seal",
        lambda value: value,
    )


def _install_metric_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: Callable[[str, str], Exception | None] | None = None,
) -> None:
    def materialize(**arguments: Any) -> dict[str, Any]:
        object_id = str(arguments["object_id"])
        camera_id = str(arguments["camera_id"])
        if failure is not None and (error := failure(object_id, camera_id)) is not None:
            raise error
        output = Path(arguments["output_directory"])
        output.mkdir(parents=True)
        (output / "metric-prefix.npz").write_bytes(b"metric-prefix")
        _write_json(output / "metric-calibration.json", {"camera_id": camera_id})
        manifest = {
            "artifact_id": hashlib.sha256(
                f"metric:{object_id}:{camera_id}".encode()
            ).hexdigest(),
            "object_id": object_id,
            "episode_id": int(object_id.rsplit("-", 1)[1]),
            "stratum": "sheet" if object_id == "object-0" else "volumetric",
            "camera_id": camera_id,
            "causal_frame_range_half_open": [0, 58],
            "projected_point_count": 17,
        }
        _write_json(output / "metric-prefix.json", manifest)
        return manifest

    def validate(directory: str | Path) -> dict[str, Any]:
        return json.loads((Path(directory) / "metric-prefix.json").read_text())

    monkeypatch.setattr(
        metric_batch, "materialize_deform360_robot_metric_prefix", materialize
    )
    monkeypatch.setattr(
        metric_batch, "validate_deform360_robot_metric_prefix", validate
    )


def test_metric_batch_emits_complete_all_stream_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    _install_contract_stubs(monkeypatch)
    _install_metric_stub(monkeypatch)

    result = materialize_deform360_prob4d_metric_batch(**arguments)
    output = Path(arguments["output_directory"])
    plan = json.loads((output / METRIC_PREFIX_PLAN_FILENAME).read_text())

    assert result["status"] == "all-streams-supported"
    assert result["supported_stream_count"] == 4
    assert result["plan_emitted"] is True
    assert len(plan["cases"]) == 2
    assert all(len(case["streams"]) == 2 for case in plan["cases"])
    assert plan["information_boundary"] == {
        "calibration_payloads_opened": True,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
        "future_frames_used": False,
        "replacement_allowed": False,
    }
    assert result["information_boundary"]["human_approval_required"] is False
    assert result["information_boundary"]["confirmation_payloads_opened"] is False
    assert result["information_boundary"]["future_frames_used"] is False
    assert len(list((output / METRIC_DIRECTORY_NAME).rglob("metric-prefix.npz"))) == 4
    validate_deform360_prob4d_metric_batch(output)


def test_metric_batch_retains_support_negative_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    _install_contract_stubs(monkeypatch)
    _install_metric_stub(
        monkeypatch,
        failure=lambda object_id, camera_id: (
            ValueError(SUPPORT_NEGATIVE_DETAIL)
            if (object_id, camera_id) == ("object-1", "camera-1")
            else None
        ),
    )

    result = materialize_deform360_prob4d_metric_batch(**arguments)
    output = Path(arguments["output_directory"])

    assert result["status"] == "support-negatives-retained"
    assert result["support_negative_stream_count"] == 1
    assert result["supported_stream_count"] == 3
    assert result["plan_emitted"] is False
    assert not (output / METRIC_PREFIX_PLAN_FILENAME).exists()
    negative = next(
        row for row in result["jobs"] if row["status"] == "support-negative"
    )
    assert negative["failure_reason"] == (
        "released-robot-geometry-outside-fixed-camera-prefix"
    )
    assert negative["failure_detail_sha256"] is None
    assert result["information_boundary"]["replacement_allowed"] is False
    validate_deform360_prob4d_metric_batch(output)


def test_metric_batch_hashes_technical_failure_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    _install_contract_stubs(monkeypatch)
    secret_detail = "private runner path must not be copied"
    _install_metric_stub(
        monkeypatch,
        failure=lambda object_id, camera_id: (
            ValueError(secret_detail)
            if (object_id, camera_id) == ("object-0", "camera-0")
            else None
        ),
    )

    result = materialize_deform360_prob4d_metric_batch(**arguments)
    output = Path(arguments["output_directory"])
    serialized = (output / METRIC_BATCH_RESULT_FILENAME).read_text()

    assert result["status"] == "technical-failures-retained"
    assert result["technical_failure_stream_count"] == 1
    technical = next(
        row for row in result["jobs"] if row["status"] == "technical-failure"
    )
    assert len(technical["failure_detail_sha256"]) == 64
    assert secret_detail not in serialized
    assert result["plan_emitted"] is False
    validate_deform360_prob4d_metric_batch(output)


def test_metric_batch_rejects_changed_policy_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    _install_contract_stubs(monkeypatch)
    _install_metric_stub(monkeypatch)
    policy_path = Path(arguments["metric_prior_policy_path"])
    policy = json.loads(policy_path.read_text())
    policy["human_approval_required"] = True
    _write_json(policy_path, policy)

    with pytest.raises(ValueError, match="policy changed"):
        materialize_deform360_prob4d_metric_batch(**arguments)

    assert not Path(arguments["output_directory"]).exists()
