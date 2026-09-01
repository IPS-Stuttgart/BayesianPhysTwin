from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.query_quotient_belief_v1 import (
    aggregate_to_query_quotient,
)
from experiments.tracking_cloth_query_quotient_v1.run import (
    LIFT_NAMES,
    _aggregate,
    categorical_scores,
    observed_trajectory_endpoints,
    query_class_index,
    same_quotient_lifts,
    trajectory_endpoints,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "experiments" / "tracking_cloth_query_quotient_v1" / "protocol.json"
)


def test_protocol_freezes_public_data_and_closed_claim_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)

    assert protocol["dataset"]["csv_count"] == 120
    assert protocol["analysis"]["target_recording_count"] == 32
    assert protocol["analysis"]["specimen_count"] == 8
    assert protocol["information_boundary"]["public_real_measurements"]
    assert not protocol["information_boundary"][
        "twist_free_marker_outcomes_used_for_prediction"
    ]
    assert not protocol["information_boundary"][
        "query_partition_fit_to_twist_outcomes"
    ]
    assert protocol["information_boundary"]["twist_outcomes_used_for_scoring_only"]
    assert not protocol["information_boundary"]["fresh_confirmation_authorized"]
    assert not protocol["information_boundary"]["paper_claim_authorized"]


def test_query_partition_is_deterministic_equal_count_and_finite_diameter() -> None:
    values = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.4, 0.3, 0.6, 0.5])
    classes = query_class_index(values, class_count=3)

    np.testing.assert_array_equal(np.bincount(classes), np.array([3, 3, 3]))
    assert tuple(classes) == tuple(query_class_index(values, class_count=3))
    class_ranges = [np.ptp(values[classes == class_id]) for class_id in range(3)]
    assert all(value >= 0.0 for value in class_ranges)
    assert max(values[classes == 0]) <= min(values[classes == 1])
    assert max(values[classes == 1]) <= min(values[classes == 2])


def test_trajectory_endpoints_use_only_free_markers_after_cutoff() -> None:
    bank = np.zeros((3, 6, 4, 3), dtype=np.float64)
    for model in range(3):
        for frame in range(2, 6):
            bank[model, frame, 1:3, 0] = (model + 1) * (frame - 1)
        bank[model, :, (0, 3), :] = 1000.0

    endpoints, mid_index = trajectory_endpoints(
        bank,
        cutoff=1,
        corners=np.array([0, 3]),
    )

    assert mid_index == 3
    np.testing.assert_allclose(endpoints[:, 0], np.array([4.0, 8.0, 12.0]))
    np.testing.assert_allclose(endpoints[:, 1], np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(endpoints[:, 2], endpoints[:, 0])


def test_observed_endpoints_tolerate_missing_free_marker_values() -> None:
    truth = np.zeros((6, 4, 3), dtype=np.float64)
    truth[2:, 1:3, 0] = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
    truth[-1, 2] = np.nan

    endpoints = observed_trajectory_endpoints(
        truth,
        reference=np.zeros((4, 3)),
        cutoff=1,
        mid_index=3,
        corners=np.array([0, 3]),
    )

    np.testing.assert_allclose(endpoints, np.array([4.0, 2.0, 4.0]))


def test_same_quotient_lifts_share_query_belief_but_not_specificity() -> None:
    prior = np.array([0.04, 0.08, 0.12, 0.10, 0.14, 0.16, 0.09, 0.11, 0.16])
    full = np.array([0.02, 0.18, 0.05, 0.06, 0.24, 0.08, 0.03, 0.11, 0.23])
    classes = np.repeat(np.arange(3), 3)

    lifts, quotient = same_quotient_lifts(prior, full, classes)

    assert tuple(lifts) == LIFT_NAMES
    for weights in lifts.values():
        np.testing.assert_allclose(
            aggregate_to_query_quotient(weights, classes),
            quotient,
            atol=1.0e-12,
            rtol=0.0,
        )
    jeffrey = lifts["jeffrey_i_projection"]
    for class_id in range(3):
        members = classes == class_id
        np.testing.assert_allclose(
            jeffrey[members] / np.sum(jeffrey[members]),
            prior[members] / np.sum(prior[members]),
        )
    assert not np.allclose(
        lifts["prior_map_concentration"],
        lifts["reverse_prior_concentration"],
    )


def test_categorical_scores_are_properly_oriented() -> None:
    confident_correct = categorical_scores([0.05, 0.90, 0.05], 1)
    diffuse = categorical_scores([1.0 / 3.0] * 3, 1)
    confident_wrong = categorical_scores([0.90, 0.05, 0.05], 1)

    assert confident_correct["correct"] == 1
    assert confident_wrong["correct"] == 0
    assert confident_correct["nll"] < diffuse["nll"] < confident_wrong["nll"]
    assert confident_correct["brier"] < diffuse["brier"] < confident_wrong["brier"]


def _row(specimen: str, condition: int, offset: float) -> dict[str, object]:
    row: dict[str, object] = {
        "recording": f"{specimen}-{condition}.csv",
        "specimen": specimen,
        "material": specimen.split("_", maxsplit=1)[0],
        "size": specimen.split("_", maxsplit=1)[1],
        "speed": "fast" if condition < 2 else "slow",
        "grasp": "hands" if condition % 2 == 0 else "hanger",
        "observed_class": condition % 3,
        "prior_query_nll": 1.2 + offset,
        "posterior_query_nll": 0.9 + offset,
        "prior_query_brier": 0.7 + offset,
        "posterior_query_brier": 0.5 + offset,
        "prior_query_correct": 0,
        "posterior_query_correct": 1,
        "same_quotient_verified": 1,
        "maximum_query_class_diameter_mm": 2.0,
        "final_query_envelope_width_mm": 2.0,
        "mid_query_envelope_width_mm": 2.5,
        "peak_query_envelope_width_mm": 3.0,
        "final_query_envelope_covers_observation": 1,
        "mid_query_envelope_covers_observation": 1,
        "peak_query_envelope_covers_observation": 1,
        "stiffness_envelope_width": 1500.0,
        "damping_envelope_width": 7.5,
        "stiffness_decision_ambiguous": 1,
        "damping_decision_ambiguous": condition % 2,
        "complete_lift_decision_disagreement": 1,
    }
    for index, name in enumerate(LIFT_NAMES):
        row[f"{name}_final_rms_displacement_absolute_error_mm"] = (
            1.0 + index + offset
        )
        row[f"{name}_mid_rms_displacement_absolute_error_mm"] = (
            1.5 + index + offset
        )
        row[f"{name}_peak_rms_displacement_absolute_error_mm"] = (
            2.0 + index + offset
        )
        row[f"{name}_unsupported_specificity_nats"] = (
            0.0 if name == "jeffrey_i_projection" else 0.1 + index
        )
    return row


def test_aggregation_uses_specimens_not_recording_rows_as_replicates() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for material in ("cotton", "denim", "polyester", "wool"):
        for size in ("A2", "A3"):
            specimen = f"{material}_{size}"
            rows.extend(
                _row(specimen, condition, 0.01 * condition)
                for condition in range(4)
            )

    specimens, metrics = _aggregate(rows, protocol)

    assert len(specimens) == 8
    assert metrics["recording_count"] == 32
    assert metrics["specimen_count"] == 8
    assert metrics["query_evidence"]["posterior_nll_better_than_prior"]
    assert metrics["query_evidence"]["posterior_brier_better_than_prior"]
    assert metrics["mechanism_checks"]["all_complete_lifts_preserved_quotient"]
    assert metrics["mechanism_checks"][
        "jeffrey_unsupported_specificity_numerically_zero"
    ]


@pytest.mark.parametrize(
    "values,class_count",
    [
        ([0.1, 0.2], 3),
        ([0.1, np.nan, 0.3], 3),
        ([0.1, 0.2, 0.3, 0.4], 3),
    ],
)
def test_invalid_query_partitions_fail_closed(
    values: object,
    class_count: int,
) -> None:
    with pytest.raises(ValueError):
        query_class_index(values, class_count=class_count)
