from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/science/run_interventional_cause_attribution_v1.py"
    )
    spec = importlib.util.spec_from_file_location("controlled_cause_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_study_separates_fit_from_transport() -> None:
    result = _module().run_study(
        trials_per_cause=100,
        seed=20260902,
        noise_standard_deviation=0.05,
    )

    assert result["source_action_identifiable_cause_count"] == 0
    assert result["all_interventions_identifiable_cause_count"] == 5
    assert result["aggregate"]["source_only_cause_accuracy"] == 0.2
    assert result["aggregate"]["multi_action_cause_accuracy"] >= 0.99
    assert (
        result["aggregate"]["multi_action_confirmation_rmse"]
        < result["aggregate"]["source_only_confirmation_rmse"]
    )
    assert result["undeclared_nuisance_control"]["declared_material_status"] == (
        "confounded"
    )
