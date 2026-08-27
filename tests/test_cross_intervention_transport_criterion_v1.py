from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "science"
        / "run_cross_intervention_transport_criterion_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_cross_intervention_transport_criterion_v1",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _roster(tmp_path: Path) -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "cross_action_transport"
        / "causal4d_sloth_multi_action_v1_sparse_pairs.json"
    )
    target = tmp_path / "roster.json"
    target.write_bytes(source.read_bytes())
    return target


def test_controlled_helpfulness_checks_pass(tmp_path: Path) -> None:
    module = _load_module()
    result = module.run_study(
        roster=_roster(tmp_path),
        trials=300,
        bootstrap_replicates=300,
        seed=20260827,
        noise_scale=0.3,
        guard_threshold=0.25,
        harmful_gain_margin=0.25,
        discrepancy_shrink=0.5,
    )
    assert result["all_registered_checks_passed"] is True
    assert result["decision"] == (
        "criterion-useful-but-requires-declared-nuisance-and-conservative-guard"
    )
    assert all(result["registered_checks"].values())
    regimes = result["regimes"]
    assert regimes["source_local_discrepancy"]["claim_rates"]["source_only"] > 0.95
    assert (
        regimes["source_local_discrepancy"]["claim_rates"]["transport_only"]
        < 0.05
    )
    assert (
        regimes["action_aligned_declared_nuisance"]["claim_rates"]["full_protocol"]
        == 0.0
    )


def test_full_registered_run_is_deterministic(tmp_path: Path) -> None:
    module = _load_module()
    kwargs = {
        "roster": _roster(tmp_path),
        "trials": 300,
        "bootstrap_replicates": 300,
        "seed": 20260827,
        "noise_scale": 0.3,
        "guard_threshold": 0.25,
        "harmful_gain_margin": 0.25,
        "discrepancy_shrink": 0.5,
    }
    first = module.run_study(**kwargs)
    second = module.run_study(**kwargs)
    assert first == second
    assert first["result_id"] == second["result_id"]


def test_roster_binding_fails_closed(tmp_path: Path) -> None:
    module = _load_module()
    roster = json.loads(_roster(tmp_path).read_text(encoding="utf-8"))
    roster["causal4d_design_sha256"] = "0" * 64
    path = tmp_path / "wrong-roster.json"
    path.write_text(json.dumps(roster), encoding="utf-8")
    with pytest.raises(ValueError, match="registered Causal4D design"):
        module.run_study(
            roster=path,
            trials=100,
            bootstrap_replicates=100,
            seed=1,
            noise_scale=0.3,
            guard_threshold=0.25,
            harmful_gain_margin=0.25,
            discrepancy_shrink=0.5,
        )
