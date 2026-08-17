from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from bayesian_phystwin.v1 import load_observation_belief

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "ecosystem_minimal_v1.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ecosystem_minimal_v1", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_ecosystem_smoke_preserves_contract_and_fallback_identity(
    tmp_path: Path,
) -> None:
    example = _load_example()
    first = example.run_example(tmp_path / "first")
    example.run_example(tmp_path / "second")

    first_observation = load_observation_belief(
        tmp_path / "first" / "prob4d_observation_belief_v1.npz"
    )
    second_observation = load_observation_belief(
        tmp_path / "second" / "prob4d_observation_belief_v1.npz"
    )
    assert first_observation.artifact_id == second_observation.artifact_id
    assert first["prob4d_observation"]["roundtrip_verified"] is True
    assert first["bayesian_phystwin_routing"] == {
        "accepted_exact_candidate_identity": True,
        "fallback_exact_baseline_identity": True,
    }

    accepted = json.loads(
        (tmp_path / "first" / "accepted_decision.json").read_text(encoding="utf-8")
    )
    fallback = json.loads(
        (tmp_path / "first" / "fallback_decision.json").read_text(encoding="utf-8")
    )
    assert accepted["selected_role"] == "candidate"
    assert accepted["exact_candidate_identity"] is True
    assert fallback["selected_role"] == "baseline"
    assert fallback["exact_fallback_identity"] is True

    manifest = first["causal4d_provider_manifest"]
    assert manifest["provider_name"] == "bayesian-phystwin"
    assert manifest["provider_revision"] == "ecosystem-minimal-v1"
    assert manifest["schema_version"] == 1
