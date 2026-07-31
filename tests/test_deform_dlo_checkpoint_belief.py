import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bayesian_phystwin.deform_dlo_checkpoint_belief import (
    average_deform_checkpoint_states,
    build_deform_checkpoint_belief_arms,
    calibrate_deform_coordinate_variance,
    combine_deform_checkpoint_predictions,
    deform_prediction_records,
    evaluate_deform_checkpoint_belief_transfer,
    evaluate_deform_coordinate_uncertainty,
    load_deform_checkpoint_belief_protocol,
    select_deform_checkpoint_belief_arm,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform_dlo_checkpoint_belief_exploratory_v1.json"
)


def _validation_records() -> list[dict[str, float | int]]:
    return [
        {"update": 0, "validation_l1_m": 0.03},
        {"update": 40, "validation_l1_m": 0.018},
        {"update": 80, "validation_l1_m": 0.015},
        {"update": 160, "validation_l1_m": 0.013},
        {"update": 280, "validation_l1_m": 0.014},
    ]


def test_checkpoint_belief_protocol_is_explicitly_post_open() -> None:
    protocol = load_deform_checkpoint_belief_protocol(PROTOCOL)

    assert protocol["source_test_status"] == "post-open-exploratory-only"
    assert protocol["fresh_confirmation_dlo"] == "DLO2"
    assert protocol["validation_gate"]["fallback"] == "selected_single_exact"


def test_checkpoint_belief_protocol_rejects_dlo1_as_confirmation(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["fresh_confirmation_dlo"] = "DLO1"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh DLO2"):
        load_deform_checkpoint_belief_protocol(changed)


def test_checkpoint_belief_arms_use_validation_only() -> None:
    protocol = load_deform_checkpoint_belief_protocol(PROTOCOL)
    arms = build_deform_checkpoint_belief_arms(_validation_records(), protocol)

    assert arms["selected_single"] == {160: 1.0}
    assert arms["swa_tail_2"] == {160: 0.5, 280: 0.5}
    assert sum(arms["validation_softmax"].values()) == pytest.approx(1.0)
    assert arms["validation_softmax"][160] > arms["validation_softmax"][40]


def test_checkpoint_belief_average_preserves_discrete_state() -> None:
    averaged = average_deform_checkpoint_states(
        {
            40: {
                "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "counter": torch.tensor(7, dtype=torch.int64),
            },
            80: {
                "weight": torch.tensor([3.0, 7.0], dtype=torch.float32),
                "counter": torch.tensor(7, dtype=torch.int64),
            },
        },
        {40: 0.25, 80: 0.75},
    )

    assert torch.equal(averaged["weight"], torch.tensor([2.5, 6.0]))
    assert torch.equal(averaged["counter"], torch.tensor(7))


def test_checkpoint_belief_average_rejects_discrete_disagreement() -> None:
    with pytest.raises(ValueError, match="discrete checkpoint state"):
        average_deform_checkpoint_states(
            {
                40: {"counter": torch.tensor(1, dtype=torch.int64)},
                80: {"counter": torch.tensor(2, dtype=torch.int64)},
            },
            {40: 0.5, 80: 0.5},
        )


def test_checkpoint_belief_combines_predictive_moments() -> None:
    mean, variance = combine_deform_checkpoint_predictions(
        {
            40: np.zeros((1, 2, 1, 3), dtype=np.float32),
            80: np.full((1, 2, 1, 3), 2.0, dtype=np.float32),
        },
        {40: 0.25, 80: 0.75},
    )

    assert np.all(mean == pytest.approx(1.5))
    assert np.all(variance == pytest.approx(0.75))


def test_checkpoint_belief_prediction_records_match_exact_l1() -> None:
    targets = np.zeros((1, 6, 1, 3), dtype=float)
    predictions = np.arange(6, dtype=float)[None, :, None, None]
    predictions = np.repeat(predictions, 3, axis=-1)
    persistence = np.ones_like(targets)

    records = deform_prediction_records(
        predictions,
        targets,
        persistence,
        ["case"],
    )

    assert records[0]["model_l1_m"] == pytest.approx(2.5)
    assert records[0]["persistence_l1_m"] == pytest.approx(1.0)
    assert records[0]["early_l1_m"] == pytest.approx(0.5)
    assert records[0]["middle_l1_m"] == pytest.approx(2.5)
    assert records[0]["late_l1_m"] == pytest.approx(4.5)


def test_checkpoint_belief_variance_calibration_never_shrinks() -> None:
    predictions = np.zeros((1, 2, 1, 3), dtype=float)
    targets = np.ones_like(predictions)
    variance = np.zeros_like(predictions)

    scale = calibrate_deform_coordinate_variance(
        predictions,
        targets,
        variance,
        variance_floor_m2=0.25,
    )
    diagnostics = evaluate_deform_coordinate_uncertainty(
        predictions,
        targets,
        variance,
        variance_floor_m2=0.25,
        variance_scale=scale,
        nominal_coverage=0.9,
    )

    assert scale == pytest.approx(4.0)
    assert diagnostics["coordinate_coverage"] == pytest.approx(1.0)
    assert diagnostics["mean_interval_width_m"] > 3.0


def test_checkpoint_belief_selector_falls_back_exactly() -> None:
    selected = select_deform_checkpoint_belief_arm(
        {
            "selected_single": 0.012,
            "swa_tail_2": 0.01195,
            "swa_tail_3": 0.0122,
        },
        minimum_relative_improvement=0.01,
    )

    assert selected["selected_arm"] == "selected_single"
    assert selected["fallback_used"] is True


def test_checkpoint_belief_selector_accepts_registered_gain() -> None:
    selected = select_deform_checkpoint_belief_arm(
        {
            "selected_single": 0.012,
            "swa_tail_2": 0.0117,
            "swa_tail_3": 0.0118,
        },
        minimum_relative_improvement=0.01,
    )

    assert selected["selected_arm"] == "swa_tail_2"
    assert selected["fallback_used"] is False


def test_checkpoint_belief_transfer_uses_paired_cases() -> None:
    candidate = [
        {"name": "a", "model_l1_m": 0.009},
        {"name": "b", "model_l1_m": 0.011},
    ]
    baseline = [
        {"name": "a", "model_l1_m": 0.010},
        {"name": "b", "model_l1_m": 0.012},
    ]

    transfer = evaluate_deform_checkpoint_belief_transfer(candidate, baseline)

    assert transfer["relative_improvement"] == pytest.approx(1.0 / 11.0)
    assert transfer["wins"] == 2
    assert "post-open" in transfer["claim_boundary"]


def test_checkpoint_belief_transfer_rejects_duplicate_cases() -> None:
    with pytest.raises(ValueError, match="not unique"):
        evaluate_deform_checkpoint_belief_transfer(
            [
                {"name": "a", "model_l1_m": 0.009},
                {"name": "a", "model_l1_m": 0.010},
            ],
            [{"name": "a", "model_l1_m": 0.011}],
        )
