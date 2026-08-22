from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts.remote.run_deform360_matphys_source_endpoint_v1 import (
    EXPECTED_RUNTIME_IDENTITY,
    _validate_prediction_seals,
    _validate_protocol,
    _validate_runtime_identity,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _prediction_fixture(tmp_path: Path) -> dict[str, Path]:
    prefix = {
        "case": "case-a",
        "object_id": "object-a",
        "camera_records": [
            {"camera": "cam0"},
            {"camera": "cam1"},
            {"camera": "cam2"},
            {"camera": "cam3"},
        ],
        "action_window": {
            "selected_raw_frame_range_half_open": [10, 91],
            "prediction_raw_frame_range_half_open": [10, 86],
            "prefix_raw_frame_range_half_open": [10, 68],
        },
        "information_boundary": {
            "source_object_frames_after_prefix_read": False,
            "future_dense_reconstruction_read": False,
            "target_metric_read": False,
        },
    }
    deform = {
        "case": "case-a",
        "object_id": "object-a",
        "passed": True,
        "information_boundary": {
            "outcome_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    part = {
        "case_id": "case-a",
        "target_object_id": "object-a",
        "information_boundary": {
            "future_object_observations_read": False,
            "changes_frozen_deform_mean": False,
        },
        "camera_selection": {
            "mode": "explicit-disjoint-provider-panel",
            "camera_ids": ["cam0", "cam2"],
        },
    }
    warp = {
        "case_id": "case-a",
        "target_object_id": "object-a",
        "passed": True,
        "information_boundary": {
            "target_future_observations_used": False,
            "target_future_outcomes_opened": False,
        },
    }
    return {
        "prefix": _write(tmp_path / "prefix.json", prefix),
        "deform": _write(tmp_path / "deform.json", deform),
        "part": _write(tmp_path / "part.json", part),
        "warp": _write(tmp_path / "warp.json", warp),
    }


def test_prediction_seals_require_disjoint_provider_panel(tmp_path: Path) -> None:
    fixture = _prediction_fixture(tmp_path)

    result = _validate_prediction_seals(
        prefix_path=fixture["prefix"],
        deform_path=fixture["deform"],
        part_path=fixture["part"],
        warp_path=fixture["warp"],
        scoring_cameras=("cam1", "cam3"),
    )

    assert result["provider_camera_ids"] == ["cam0", "cam2"]
    assert result["scoring_camera_ids"] == ["cam1", "cam3"]


def test_prediction_seals_reject_shared_provider_camera(tmp_path: Path) -> None:
    fixture = _prediction_fixture(tmp_path)
    part = json.loads(fixture["part"].read_text(encoding="utf-8"))
    part["camera_selection"]["camera_ids"] = ["cam0", "cam1"]
    _write(fixture["part"], part)

    with pytest.raises(ValueError, match="registered partition"):
        _validate_prediction_seals(
            prefix_path=fixture["prefix"],
            deform_path=fixture["deform"],
            part_path=fixture["part"],
            warp_path=fixture["warp"],
            scoring_cameras=("cam1", "cam3"),
        )


def test_committed_protocol_binds_source_denominator_and_camera_panel() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "sota" / "matphys_surface_uq_source_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    scoring = tuple(value["camera_partition"]["scoring_camera_ids"])

    loaded = _validate_protocol(
        path,
        case_id="153-cake-ep0005",
        scoring_cameras=scoring,
    )

    assert loaded["source_panel"]["replacement_allowed"] is False
    assert loaded["source_amendments"][0]["source_outcome_or_metric_inspected"] is False
    assert loaded["source_amendments"][0]["prediction_or_gate_changed"] is False
    assert (
        loaded["source_amendments"][1]["source_reconstruction_or_metric_inspected"]
        is False
    )
    assert loaded["source_amendments"][1]["prediction_or_gate_changed"] is False
    assert (
        loaded["source_amendments"][2][
            "source_reconstructed_endpoint_or_metric_inspected"
        ]
        is False
    )
    assert loaded["source_amendments"][2]["prediction_or_gate_changed"] is False
    assert (
        loaded["source_amendments"][3][
            "source_reconstructed_endpoint_or_metric_inspected"
        ]
        is False
    )
    assert loaded["source_amendments"][3]["prediction_or_gate_changed"] is False
    assert loaded["source_amendments"][3]["camera_or_support_gate_changed"] is False
    assert loaded["scoring_reconstruction_runtime"] == EXPECTED_RUNTIME_IDENTITY
    assert loaded["outcome"]["robot_state_required_for_scoring_reconstruction"] is False
    assert loaded["outcome"]["urdf_gripper_mask_used"] is False
    assert loaded["outcome"]["gripper_pixels_excluded"] is False
    assert (
        loaded["information_boundary"]["robot_state_read_for_scoring_reconstruction"]
        is False
    )
    with pytest.raises(ValueError, match="denominator"):
        _validate_protocol(
            path,
            case_id="unregistered-case",
            scoring_cameras=scoring,
        )


def test_scoring_runtime_identity_is_exact_and_cuda_backed() -> None:
    observed = {**EXPECTED_RUNTIME_IDENTITY, "cuda_device_name": "Test GPU"}

    assert _validate_runtime_identity(observed) == observed

    for key, changed in (
        ("python_version", "3.12.0"),
        ("torch_version", "2.5.0+cu121"),
        ("torch_cuda_version", "12.4"),
        ("gsplat_cuda_backend_available", False),
        ("gsplat_camera_model_available", False),
        ("nerfstudio_splatfacto_available", False),
    ):
        mutated = {**observed, key: changed}
        with pytest.raises(ValueError, match=f"runtime {key} changed"):
            _validate_runtime_identity(mutated)


def test_scoring_runtime_identity_rejects_missing_or_extra_fields() -> None:
    observed = {**EXPECTED_RUNTIME_IDENTITY, "cuda_device_name": "Test GPU"}
    observed.pop("numpy_version")
    with pytest.raises(ValueError, match="runtime fields changed"):
        _validate_runtime_identity(observed)

    observed = {
        **EXPECTED_RUNTIME_IDENTITY,
        "cuda_device_name": "Test GPU",
        "unregistered": "value",
    }
    with pytest.raises(ValueError, match="runtime fields changed"):
        _validate_runtime_identity(observed)


def test_retained_failure_preserves_stage_and_camera_diagnostics() -> None:
    from scripts.remote import run_deform360_matphys_source_endpoint_v1 as runner

    source = inspect.getsource(runner.main)

    assert '"failure_stage": failure_stage' in source
    assert '"camera_records": camera_records' in source
    assert '"successful_mask_camera_count": len(support_by_camera)' in source
