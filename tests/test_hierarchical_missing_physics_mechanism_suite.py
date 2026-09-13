"""Regression tests for the six-mechanism discrepancy diagnosis suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_hierarchical_missing_physics_mechanism_suite.py"


def _load_module():
    name = "hierarchical_missing_physics_mechanism_suite"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_all_registered_mechanisms_are_diagnosed_and_scoped() -> None:
    module = _load_module()
    result = module.run_suite(seed=41, bootstrap=8)

    assert result["gate"]["passed"] is True
    assert result["correct_diagnoses"] == len(module.GROUPS)
    assert result["total_diagnoses"] == len(module.GROUPS)
    assert result["gate"]["shared_transfers_and_improves"] is True
    assert result["gate"]["all_nontransferable_mechanisms_rejected"] is True
    assert result["gate"]["all_rejected_mechanisms_emit_exact_fallback"] is True
    assert result["gate"]["sensor_never_changes_physical_rollout"] is True
    assert result["gate"]["fallback_identity_violations"] == 0
    assert result["information_boundary"]["protected_dlo4_dlo5_result_read"] is False


def test_only_shared_physics_is_cross_object_cross_backend_eligible() -> None:
    module = _load_module()
    result = module.run_suite(seed=43, bootstrap=6)
    for group, value in result["cases"].items():
        eligible = value["scope_evaluation"][
            "admissible_for_cross_object_cross_backend_physical_transfer"
        ]
        assert eligible is (group == "shared_physics")
        if group != "shared_physics":
            assert value["scope_evaluation"]["emitted_is_exact_fallback"] is True
