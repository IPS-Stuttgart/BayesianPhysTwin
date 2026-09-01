from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tracking_cloth_query_quotient_v1.core import (
    SAME_QUOTIENT_LIFTS,
    categorical_scores,
    centered_shape_rms_m,
    observed_query_class,
    prior_aware_source_posterior,
    query_partition,
    registered_prior,
    same_quotient_lifts,
    trajectory_energy_score_mm,
    trajectory_mask,
    trajectory_rmse_mm,
    unsupported_specificity_nats,
)
from experiments.tracking_cloth_query_quotient_v1.run import (
    PROTOCOL_SCHEMA,
    load_protocols,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "tracking_cloth_query_quotient_v1"
    / "protocol.json"
)
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "tracking-cloth-query-quotient-v1.yml"
)
REQUEST = (
    ROOT
    / "run_requests"
    / "tracking-cloth-query-quotient-real-v1.json"
)


def test_protocol_extends_exact_base_without_widening_claims() -> None:
    protocol, base = load_protocols(PROTOCOL)

    assert protocol["schema"] == PROTOCOL_SCHEMA
    assert protocol["query"]["requested_class_count"] == 3
    assert base["dataset_record"] == "14644526"
    assert base["csv_count"] == 120
    assert protocol["information_boundary"]["public_real_measurements"]
    assert protocol["information_boundary"][
        "source_motion_only_for_belief_update"
    ]
    assert not protocol["information_boundary"][
        "target_outcomes_used_for_selection"
    ]
    assert not protocol["information_boundary"][
        "fresh_confirmation_authorized"
    ]
    assert not protocol["information_boundary"]["paper_claim_authorized"]


def test_registered_prior_is_nonuniform_product_prior() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prior = registered_prior(protocol)

    np.testing.assert_allclose(prior.sum(), 1.0)
    assert prior.shape == (9,)
    assert prior[4] == pytest.approx(0.36)
    assert prior[0] == pytest.approx(0.04)
    assert prior[4] > prior[1] > prior[0]


def test_prior_aware_source_posterior_uses_one_loss_per_recording() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prior = registered_prior(protocol)
    losses = np.full((4, 9), 4e-6)
    losses[:, 7] = 0.25e-6

    posterior, temperature = prior_aware_source_posterior(
        losses,
        prior,
        measurement_floor_m=0.001,
    )

    assert posterior.shape == (9,)
    np.testing.assert_allclose(posterior.sum(), 1.0)
    assert int(np.argmax(posterior)) == 7
    assert temperature == pytest.approx(1e-6)


def test_centered_query_is_translation_invariant() -> None:
    frames = 12
    markers = 5
    base = np.stack(
        [
            np.linspace(0.0, 0.4, markers),
            np.linspace(-0.1, 0.1, markers),
            np.linspace(0.2, -0.2, markers),
        ],
        axis=1,
    )
    trajectory = np.repeat(base[None], frames, axis=0)
    trajectory[:, 2, 1] += np.linspace(0.0, 0.08, frames)
    translation = np.zeros_like(trajectory)
    translation[:, :, 0] = np.linspace(0.0, 3.0, frames)[:, None]
    translation[:, :, 2] = np.linspace(0.0, -2.0, frames)[:, None]

    first = centered_shape_rms_m(
        trajectory,
        cutoff=3,
        corners=[0, 4],
        tail_fraction=0.5,
    )
    second = centered_shape_rms_m(
        trajectory + translation,
        cutoff=3,
        corners=[0, 4],
        tail_fraction=0.5,
    )

    assert first > 0.0
    assert second == pytest.approx(first, abs=1e-12)


def test_query_partition_is_contiguous_and_merges_tied_boundary() -> None:
    values = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2])
    classes, thresholds = query_partition(
        values,
        requested_class_count=3,
        minimum_gap_m=1e-12,
    )

    np.testing.assert_array_equal(classes, np.repeat(np.arange(3), 3))
    np.testing.assert_allclose(thresholds, np.array([1.6, 2.6]))
    assert observed_query_class(0.5, thresholds) == 0
    assert observed_query_class(2.0, thresholds) == 1
    assert observed_query_class(4.0, thresholds) == 2

    tied, tied_thresholds = query_partition(
        np.ones(9),
        requested_class_count=3,
        minimum_gap_m=1e-12,
    )
    np.testing.assert_array_equal(tied, np.zeros(9, dtype=np.int64))
    assert tied_thresholds.size == 0


def test_same_quotient_lifts_preserve_class_masses() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prior = registered_prior(protocol)
    posterior = np.array(
        [0.02, 0.04, 0.06, 0.08, 0.30, 0.10, 0.12, 0.18, 0.10]
    )
    classes = np.repeat(np.arange(3), 3)

    lifts = same_quotient_lifts(prior, posterior, classes)
    expected = np.array([0.12, 0.48, 0.40])

    assert tuple(lifts) == SAME_QUOTIENT_LIFTS
    for weights in lifts.values():
        np.testing.assert_allclose(
            np.bincount(classes, weights=weights, minlength=3),
            expected,
            atol=1e-12,
        )
    assert unsupported_specificity_nats(
        prior,
        lifts["jeffrey_i_projection"],
        classes,
    ) == pytest.approx(0.0, abs=1e-12)
    assert unsupported_specificity_nats(
        prior,
        lifts["prior_map_within_class"],
        classes,
    ) > 0.0
    assert unsupported_specificity_nats(
        prior,
        lifts["full_source_posterior"],
        classes,
    ) > 0.0


def test_categorical_score_depends_only_on_quotient() -> None:
    probabilities = np.array([0.15, 0.70, 0.15])
    first = categorical_scores(
        probabilities,
        1,
        probability_floor=1e-12,
    )
    second = categorical_scores(
        probabilities.copy(),
        1,
        probability_floor=1e-12,
    )

    assert first == second
    assert first["query_class_correct"] == 1
    assert first["query_log_score_nats"] == pytest.approx(-np.log(0.70))


def test_trajectory_scores_are_zero_for_exact_degenerate_belief() -> None:
    truth = np.zeros((8, 4, 3), dtype=np.float64)
    bank = np.stack([truth.copy(), np.ones_like(truth)])
    mask = trajectory_mask(
        truth,
        cutoff=1,
        corners=[0],
        time_stride=2,
    )

    assert trajectory_energy_score_mm(
        bank,
        [1.0, 0.0],
        truth,
        mask,
    ) == pytest.approx(0.0)
    assert trajectory_rmse_mm(
        bank,
        [1.0, 0.0],
        truth,
        mask,
    ) == pytest.approx(0.0)
    assert trajectory_energy_score_mm(
        bank,
        [0.5, 0.5],
        truth,
        mask,
    ) > 0.0


def test_result_validator_rejects_self_authorization() -> None:
    result = {
        "schema": (
            "bayesian-phystwin/"
            "tracking-cloth-query-quotient-result-v1"
        ),
        "schema_version": 1,
        "metrics": {
            "target_recordings": 32,
            "contract": {
                "target_cases": 32,
                "same_quotient_max_l1": 0.0,
                "same_quotient_query_log_score_max_spread": 0.0,
                "jeffrey_max_unsupported_specificity_nats": 0.0,
                "same_quotient_contract_passed": True,
            },
            "paper_claim_authorized": False,
        },
        "information_boundary": {
            "public_real_measurements": True,
            "raw_data_uploaded": False,
            "fresh_confirmation_authorized": False,
            "paper_claim_authorized": True,
        },
    }

    with pytest.raises(ValueError, match="self-authorized"):
        validate_result(result)


def test_workflow_is_file_triggered_and_bound_to_gpuserver4090() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert "experiment/tracking-cloth-query-quotient-real-v1" in workflow
    assert "run_requests/tracking-cloth-query-quotient-real-v1.json" in workflow
    assert "runs-on: [self-hosted, Linux, X64, gpuserver4090]" in workflow
    assert (
        request["dataset_root"]
        == "/home/github-runner/.cache/datasets/"
        "tracking-cloth-deformation-v1-zenodo-14644526"
    )
    assert request["runner_label"] == "gpuserver4090"
    assert request["mode"] == "evaluate"
    assert request["paper_claim_authorized"] is False
