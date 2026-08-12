"""Adversarial tests for fail-closed PhysTwin refit input contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from bayesian_phystwin.phystwin_prior_evaluation import (
    evaluate_phystwin_prior_arrays,
)
from bayesian_phystwin.phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    build_phystwin_track_objective,
    causal_markov_cue_reliability,
    evaluate_phystwin_trajectory,
    evaluate_phystwin_trajectory_splits,
    phystwin_tracking_metrics,
)


def _masks() -> tuple[np.ndarray, np.ndarray]:
    visible = np.array([[True, True], [True, False], [True, True]])
    motion_valid = np.array([[True, False], [False, False], [True, False]])
    return visible, motion_valid


@pytest.mark.parametrize("invalid", [False, 0, {}, object()])
def test_refit_objective_rejects_invalid_config_types(invalid: object) -> None:
    visible, motion_valid = _masks()

    with pytest.raises(TypeError, match="PhysTwinRefitReliabilityConfig"):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="mixture",
            config=cast("PhysTwinRefitReliabilityConfig", invalid),
        )


@pytest.mark.parametrize("invalid", [False, 0, [], object()])
def test_refit_objective_rejects_invalid_cue_mappings(invalid: object) -> None:
    visible, motion_valid = _masks()

    with pytest.raises(TypeError, match="cues must be a mapping"):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="mixture",
            cues=cast("Mapping[str, np.ndarray]", invalid),
        )


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("minimum_probability", False, TypeError),
        ("minimum_probability", np.nan, ValueError),
        ("confidence_power", -1.0, ValueError),
        ("boundary_scale", 0.0, ValueError),
        ("flow_scale", np.inf, ValueError),
        ("occlusion_probability", 1.1, ValueError),
        ("markov_inlier_persistence", 1.0, ValueError),
        ("markov_outlier_persistence", True, TypeError),
    ],
)
def test_refit_config_fails_closed_on_invalid_values(
    keyword: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        PhysTwinRefitReliabilityConfig(**{keyword: value})


def test_refit_masks_require_boolean_dtype() -> None:
    visible, motion_valid = _masks()

    with pytest.raises(TypeError, match="visible must contain only booleans"):
        build_phystwin_track_objective(
            visible.astype(np.int8),
            motion_valid,
            variant="hard",
        )
    with pytest.raises(TypeError, match="motion_valid must contain only booleans"):
        build_phystwin_track_objective(
            visible,
            motion_valid.astype(np.float64),
            variant="hard",
        )
    with pytest.raises(TypeError, match="mask must contain only booleans"):
        phystwin_tracking_metrics(
            np.zeros((3, 2, 3)),
            np.zeros((3, 2, 3)),
            visible.astype(np.int8),
        )


@pytest.mark.parametrize(
    "name",
    ["occluded", "forward_backward_valid", "multiview_valid"],
)
def test_boolean_cues_require_boolean_dtype(name: str) -> None:
    visible, motion_valid = _masks()

    with pytest.raises(TypeError, match=rf"cue {name} must contain only booleans"):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="cue",
            cues={name: np.ones(visible.shape, dtype=np.int8)},
        )


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("confidence", np.full((3, 2), 1.1), "must lie in"),
        ("visibility_probability", np.full((3, 2), -0.1), "must lie in"),
        ("flow_inconsistency", np.full((3, 2), -0.1), "nonnegative"),
        ("boundary_distance", np.full((3, 2), np.nan), "finite"),
    ],
)
def test_numeric_cues_reject_invalid_scientific_values(
    name: str,
    values: np.ndarray,
    message: str,
) -> None:
    visible, motion_valid = _masks()

    with pytest.raises(ValueError, match=message):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="cue",
            cues={name: values},
        )


def test_markov_reliability_rejects_probability_contract_violations() -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        causal_markov_cue_reliability(np.array([[0.5, 1.1]]))
    with pytest.raises(TypeError, match="real scalar"):
        causal_markov_cue_reliability(
            np.array([[0.5]]),
            inlier_persistence=True,
        )


def test_tracking_metrics_reject_nonfinite_coordinates() -> None:
    observed = np.zeros((2, 1, 3))
    trajectory = np.zeros_like(observed)
    mask = np.ones((2, 1), dtype=bool)
    observed[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="observed must contain finite values"):
        phystwin_tracking_metrics(observed, trajectory, mask)


def test_split_controls_require_literal_integer_contracts() -> None:
    visible, motion_valid = _masks()
    observed = np.zeros((3, 2, 3))
    trajectory = np.zeros_like(observed)

    with pytest.raises(TypeError, match="train_end_frame must be an integer"):
        evaluate_phystwin_trajectory(
            observed,
            trajectory,
            visible,
            motion_valid,
            train_end_frame=True,
        )
    with pytest.raises(TypeError, match="splits must be a mapping"):
        evaluate_phystwin_trajectory_splits(
            observed,
            trajectory,
            visible,
            motion_valid,
            splits=cast("Mapping[str, tuple[int, int]]", []),
        )
    with pytest.raises(TypeError, match="split test start must be an integer"):
        evaluate_phystwin_trajectory_splits(
            observed,
            trajectory,
            visible,
            motion_valid,
            splits={"test": (False, 3)},
        )


def test_prior_evaluation_uses_same_fail_closed_mask_and_config_boundary() -> None:
    visible, motion_valid = _masks()

    with pytest.raises(TypeError, match="visible must contain only booleans"):
        evaluate_phystwin_prior_arrays(
            visible.astype(np.int8),
            motion_valid,
            {},
        )
    with pytest.raises(TypeError, match="PhysTwinRefitReliabilityConfig"):
        evaluate_phystwin_prior_arrays(
            visible,
            motion_valid,
            {},
            config=cast("PhysTwinRefitReliabilityConfig", False),
        )
