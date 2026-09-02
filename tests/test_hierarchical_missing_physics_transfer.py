"""Regression tests for sealed hierarchical discrepancy transfer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "evaluate_hierarchical_missing_physics_transfer.py"


def _load_module():
    name = "hierarchical_transfer_evaluator"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_controlled_transfer_seals_diagnosis_before_target(tmp_path: Path) -> None:
    module = _load_module()
    result = module.run_controlled_smoke(tmp_path, seed=19, bootstrap=6)

    assert result["controlled_gate"]["passed"] is True
    assert result["controlled_gate"]["diagnosis_guided_beats_physical"] is True
    assert result["controlled_gate"]["diagnosis_guided_beats_all_components"] is True
    assert result["controlled_gate"]["shared_beats_wrong_diagnosis"] is True
    assert result["controlled_gate"]["sensor_physical_change_is_zero"] is True
    assert result["fallback_identity_violations"] == 0
    assert "shared_physics" in result["selected_groups_frozen_source_only"]
    assert result["information_boundary"]["target_used_for_group_selection"] is False
    assert result["information_boundary"]["target_used_for_coefficient_refit"] is False
    assert (tmp_path / "sealed_source_model.json").is_file()
    assert (tmp_path / "sealed_source_model.npz").is_file()
    assert (tmp_path / "sealed_source_model_source_result.json").is_file()


def test_sealed_model_detects_array_tampering(tmp_path: Path) -> None:
    module = _load_module()
    source, _target = module.make_controlled_transfer_panels(seed=23)
    source_npz, source_json = module.save_panel(source, tmp_path / "source")
    module.seal_source_model(
        source_npz,
        source_json,
        output_prefix=tmp_path / "model",
        bootstrap_repetitions=5,
        seed=23,
    )
    array_path = tmp_path / "model.npz"
    array_path.write_bytes(array_path.read_bytes() + b"tamper")
    with pytest.raises(module.TransferContractError, match="array hash mismatch"):
        module.load_sealed_model(tmp_path / "model")


def test_target_authorization_is_bound_to_source_model(tmp_path: Path) -> None:
    module = _load_module()
    source, target = module.make_controlled_transfer_panels(seed=29)
    source_npz, source_json = module.save_panel(source, tmp_path / "source")
    target_npz, target_json = module.save_panel(target, tmp_path / "target")
    module.seal_source_model(
        source_npz,
        source_json,
        output_prefix=tmp_path / "model",
        bootstrap_repetitions=5,
        seed=29,
    )
    sealed = module.load_sealed_model(tmp_path / "model")
    loaded_target = module.CORE.load_panel(target_npz, target_json)
    with pytest.raises(module.TransferContractError, match="another source-model seal"):
        module.evaluate_frozen_transfer(
            loaded_target,
            sealed,
            authorization={
                "target_scoring_authorized": True,
                "source_model_id": "0" * 64,
                "selection_frozen_before_target": True,
                "target_panel_sha256": loaded_target.metadata["panel_sha256"],
            },
        )


def test_sensor_discrepancy_cannot_be_authorized_for_physical_rollout(tmp_path: Path) -> None:
    module = _load_module()
    source, _target = module.make_controlled_transfer_panels(seed=31)
    model = module.CORE.fit_group_ard(source)
    with pytest.raises(module.TransferContractError, match="sensor discrepancy"):
        module.model_payload(
            model,
            selected_groups=["shared_physics", "sensor"],
            wrong_diagnosis_group=None,
            source_result_id="source",
            source_panel_sha256="panel",
            metadata={},
        )
