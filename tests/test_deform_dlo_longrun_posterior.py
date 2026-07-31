import json
from pathlib import Path

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    load_deform_longrun_posterior_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_longrun_posterior_v2.json"
SUPERSEDED_PROTOCOL = (
    REPOSITORY_ROOT / "configs" / "sota" / "deform_dlo_longrun_posterior_v1.json"
)
PARITY = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform_dlo_longrun_posterior_v1"
    / "rollout_parity_smoke.json"
)


def test_longrun_posterior_operator_bank_is_frozen() -> None:
    posterior = load_deform_longrun_posterior_protocol(PROTOCOL)

    assert posterior["operators"] == [
        "parameter_mean",
        "predictive_mean",
        "predictive_median",
    ]
    assert posterior["benchmark_point_loss"] == "coordinate-wise-l1"
    assert posterior["fallback"] == "selected_single_exact"
    assert posterior["validation_improvement_min"] == 0.01
    assert posterior["source_transfer_improvement_min"] == 0.01
    assert posterior["source_transfer_minimum_case_wins"] == 5
    assert posterior["coordinate_variance_floor_m2"] == 0.000025
    assert posterior["coordinate_interval_nominal_coverage"] == 0.9


def test_superseded_longrun_posterior_is_non_executable() -> None:
    try:
        load_deform_longrun_posterior_protocol(SUPERSEDED_PROTOCOL)
    except ValueError as error:
        assert "unsupported DEFORM long-run posterior schema" in str(error)
    else:
        raise AssertionError("superseded long-run posterior unexpectedly loaded")


def test_longrun_posterior_rollout_matches_parent() -> None:
    parity = json.loads(PARITY.read_text(encoding="utf-8"))

    assert parity["official_eval_read"] is False
    assert parity["source_test_opened"] is False
    assert parity["passed"] is True
    assert parity["model_abs_diff_m"] <= parity["tolerance_m"]
    assert parity["persistence_abs_diff_m"] <= parity["tolerance_m"]
