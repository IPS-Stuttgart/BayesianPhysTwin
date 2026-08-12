from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
INFERENCE_GUIDE_PATH = ROOT / "docs/inference_v1.md"
EXAMPLE_PATH = ROOT / "examples/guarded_inference_v1.py"


def test_public_onboarding_uses_versioned_integration_namespaces() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    guide = INFERENCE_GUIDE_PATH.read_text(encoding="utf-8")

    assert "from bayesian_phystwin import load_observation_belief" not in readme
    assert "from bayesian_phystwin.v1 import load_observation_belief" in readme
    assert "from bayesian_phystwin.inference.v1 import (" in readme
    assert "[Guarded inference API v1](docs/inference_v1.md)" in readme
    assert "Keep point-mean and covariance decisions separate" in guide
    assert "exact baseline Python object" in guide


def test_guarded_inference_example_proves_exact_object_routing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    payload = json.loads(capsys.readouterr().out)

    accepted = payload["accepted"]
    assert accepted["guard_accepted"] is True
    assert accepted["selection_reason"] == "guard-accepted"
    assert accepted["selected_candidate"] is True
    assert accepted["exact_fallback"] is False
    assert accepted["selected_belief_is_candidate_object"] is True
    assert accepted["selected_belief_is_baseline_object"] is False

    fallback = payload["fallback"]
    assert fallback["guard_accepted"] is False
    assert fallback["selection_reason"] == "regret-guard-rejected"
    assert fallback["selected_candidate"] is False
    assert fallback["exact_fallback"] is True
    assert fallback["selected_belief_is_candidate_object"] is False
    assert fallback["selected_belief_is_baseline_object"] is True

    for record in (accepted, fallback):
        assert len(record["selected_belief_id"]) == 64
        assert len(record["result_id"]) == 64
