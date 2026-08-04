from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_prob4d_bpt_controlled_decisive_v1.py"
SPEC = importlib.util.spec_from_file_location("controlled_decisive_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _protocol() -> dict:
    return json.loads(
        (ROOT / "protocols" / "prob4d_bpt_controlled_decisive_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_protocol_digest_is_frozen() -> None:
    assert MODULE._canonical_sha256(_protocol()) == (
        "921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111"
    )


def test_conditioned_gauge_prior_reduces_uncertainty() -> None:
    prior = np.eye(14, dtype=np.float64) * 4.0e-4
    prior[:7, 7:] = np.eye(7) * 1.0e-4
    prior[7:, :7] = np.eye(7) * 1.0e-4
    anchor_covariance = np.eye(7, dtype=np.float64) * 1.0e-6

    mean, posterior = MODULE._condition_gauge_prior(
        prior,
        np.linspace(-0.01, 0.01, 7),
        anchor_covariance,
    )

    assert mean.shape == (14,)
    assert posterior.shape == (14, 14)
    assert np.trace(posterior) < np.trace(prior)
    np.testing.assert_allclose(posterior, posterior.T, atol=0.0, rtol=0.0)


def test_generated_group_uses_persistent_ids_and_joint_gauge_prior() -> None:
    _, config = MODULE.load_protocol(
        ROOT / "protocols" / "prob4d_bpt_controlled_decisive_v1.json"
    )

    group = MODULE.generate_group(
        config.calibration_seed,
        "nominal_correlated",
        config,
        group_prefix="test",
    )

    expected_ids = np.tile(np.arange(config.point_count), config.frame_count)
    np.testing.assert_array_equal(group.stack.point_ids, expected_ids)
    assert group.stack.gauge_count == 2
    assert np.any(group.stack.gauge_prior_covariance[:7, 7:] != 0.0)
    assert group.stack.local_gauge_jacobian.shape[-1] == 7


def test_small_study_is_deterministic_and_exact_on_fallback() -> None:
    protocol = _protocol()
    _, original = MODULE.load_protocol(
        ROOT / "protocols" / "prob4d_bpt_controlled_decisive_v1.json"
    )
    config = replace(
        original,
        scenarios=original.scenarios[:3],
        calibration_groups_per_scenario=1,
        target_groups_per_scenario=1,
        bootstrap_resamples=20,
        guard_minimum_accepted_groups=1,
    )
    protocol["scenarios"] = list(config.scenarios)
    protocol["calibration"]["groups_per_scenario"] = 1
    protocol["target"]["groups_per_scenario"] = 1
    protocol["bootstrap"]["resamples"] = 20
    protocol["guard_calibration"]["minimum_accepted_groups"] = 1

    first, first_trials = MODULE.run_study(
        protocol,
        config,
        repository_revision=protocol["repository_pins"]["bayesian_phystwin_base"],
        prob4d_revision=protocol["repository_pins"]["prob4d"],
    )
    second, second_trials = MODULE.run_study(
        protocol,
        config,
        repository_revision=protocol["repository_pins"]["bayesian_phystwin_base"],
        prob4d_revision=protocol["repository_pins"]["prob4d"],
    )

    assert first["report_id"] == second["report_id"]
    assert first_trials == second_trials
    assert all(value.exact_fallback for value in first_trials)
    baseline = [
        value for value in first_trials if value.method_id == MODULE.BASELINE_METHOD
    ]
    assert baseline and not any(value.guard_accepted for value in baseline)


def test_self_hosted_workflow_is_read_only_and_pinned() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "prob4d-bpt-controlled-decisive.yml"
    ).read_text(encoding="utf-8")

    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "contents: write" not in workflow
    assert "push:" not in workflow
    assert "persist-credentials: false" in workflow
    pins = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    )
    assert all(pin in workflow for pin in pins)
