"""Regression tests for hierarchical missing-physics diagnosis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "hierarchical_missing_physics_diagnosis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hierarchical_diagnosis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compact_panel(module, seed: int = 3):
    rng = np.random.default_rng(seed)
    y_rows = []
    shared_rows = []
    object_rows = []
    backend_rows = []
    trajectories = []
    objects = []
    backends = []
    for object_index, object_name in enumerate(("DLO2", "DLO3")):
        for trajectory_index in range(3):
            backend_name = "deform"
            phase = 0.35 * trajectory_index + 0.2 * object_index
            for sample_index in range(14):
                t = sample_index / 13.0
                shared = np.array(
                    [np.sin(2.0 * np.pi * t + phase), np.cos(np.pi * t)]
                )
                object_block = np.array(
                    [shared[0] if object_index == 0 else 0.0,
                     shared[0] if object_index == 1 else 0.0]
                )
                backend_block = np.array([shared[1]])
                residual = np.array(
                    [
                        0.90 * shared[0] - 0.45 * shared[1]
                        + 0.08 * object_block[object_index],
                        -0.62 * shared[0] + 0.30 * shared[1],
                    ]
                )
                residual += rng.normal(scale=0.025, size=2)
                y_rows.append(residual)
                shared_rows.append(shared)
                object_rows.append(object_block)
                backend_rows.append(backend_block)
                trajectories.append(f"{object_name}:trajectory-{trajectory_index}")
                objects.append(object_name)
                backends.append(backend_name)
    return module.ResidualPanel(
        y=np.asarray(y_rows),
        blocks={
            "shared_physics": np.asarray(shared_rows),
            "object": np.asarray(object_rows),
            "backend": np.asarray(backend_rows),
        },
        trajectory_id=np.asarray(trajectories),
        object_id=np.asarray(objects),
        backend_id=np.asarray(backends),
        metadata={
            "transfer_eligibility": {
                "shared_physics": True,
                "object": False,
                "backend": False,
            },
            "minimum_bootstrap_diagnosis_frequency": 0.50,
            "source_gate": {
                "shared_vs_physical_min_relative_improvement": 0.10,
                "minimum_source_bootstrap_shared_diagnosis_frequency": 0.50,
                "minimum_complete_trajectory_win_fraction": 0.80,
                "maximum_worst_trajectory_ratio_vs_physical": 0.95,
            },
        },
    )


def test_contract_requires_shared_physics_block() -> None:
    module = _load_module()
    panel = module.ResidualPanel(
        y=np.ones((6, 1)),
        blocks={"object": np.ones((6, 1))},
        trajectory_id=np.asarray(["a", "a", "a", "b", "b", "b"]),
        object_id=np.asarray(["o"] * 6),
        backend_id=np.asarray(["b"] * 6),
        metadata={},
    )
    with pytest.raises(module.ContractError, match="shared_physics"):
        panel.validate()


def test_group_ard_recovers_shared_missing_physics() -> None:
    module = _load_module()
    panel = _compact_panel(module)
    model = module.fit_group_ard(panel, max_iterations=180)
    diagnostics = module.group_diagnostics(panel, model)
    physical_rmse = float(np.sqrt(np.mean(np.square(panel.y))))
    shared_prediction = model.predict(panel.blocks, active_groups=["shared_physics"])
    shared_rmse = float(np.sqrt(np.mean(np.square(panel.y - shared_prediction))))

    assert shared_rmse < 0.45 * physical_rmse
    assert diagnostics["shared_physics"]["source_supported"] is True
    assert diagnostics["shared_physics"]["effective_df_per_output"] > 0.10
    assert diagnostics["shared_physics"]["negative_log_score_increase_when_removed"] > 0.0


def test_source_evaluation_is_target_blind_and_fallback_exact() -> None:
    module = _load_module()
    panel = _compact_panel(module)
    result = module.evaluate_source_panel(
        panel,
        bootstrap_repetitions=4,
        seed=11,
    )

    assert result["information_boundary"]["target_outcomes_read"] is False
    assert result["information_boundary"]["target_group_selection"] is False
    assert result["source_gate"]["fallback_identity_violations"] == 0
    assert "shared_physics" in result["transferable_groups_selected_source_only"]
    assert result["aggregate"]["shared_physics"][
        "relative_improvement_vs_physical"
    ] > 0.10
    assert result["source_gate"]["passed"] is True


def test_sensor_group_cannot_be_declared_transferable_by_default() -> None:
    module = _load_module()
    assert module.DEFAULT_TRANSFER_ELIGIBILITY["sensor"] is False
    assert module.DEFAULT_TRANSFER_ELIGIBILITY["object"] is False
    assert module.DEFAULT_TRANSFER_ELIGIBILITY["backend"] is False
    assert module.DEFAULT_TRANSFER_ELIGIBILITY["shared_physics"] is True
