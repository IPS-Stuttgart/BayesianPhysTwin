import json
from pathlib import Path

from bayesian_phystwin.deform_dlo_longrun import (
    load_deform_dlo_longrun_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_longrun_v2.json"
PARITY = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo_longrun_posterior_v1"
    / "rollout_parity_smoke.json"
)


def test_longrun_posterior_operator_bank_is_frozen() -> None:
    protocol = load_deform_dlo_longrun_protocol(PROTOCOL)
    posterior = protocol["checkpoint_posterior_if_source_gate_passes"]

    assert posterior["operators"] == ["parameter_mean", "predictive_mean"]
    assert posterior["fallback"] == "selected_single_exact"
    assert posterior["validation_improvement_min"] == 0.01
    assert posterior["source_transfer_improvement_min"] == 0.01
    assert posterior["source_transfer_minimum_case_wins"] == 5
    assert posterior["coordinate_variance_floor_m2"] == 0.000025
    assert posterior["coordinate_interval_nominal_coverage"] == 0.9


def test_longrun_posterior_rollout_matches_parent() -> None:
    parity = json.loads(PARITY.read_text(encoding="utf-8"))

    assert parity["official_eval_read"] is False
    assert parity["source_test_opened"] is False
    assert parity["passed"] is True
    assert parity["model_abs_diff_m"] <= parity["tolerance_m"]
    assert parity["persistence_abs_diff_m"] <= parity["tolerance_m"]
