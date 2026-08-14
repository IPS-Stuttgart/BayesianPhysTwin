"""Adversarial tests for fail-closed PhysTwin refit input contracts."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bayesian_phystwin.phystwin_prior_evaluation import (
    evaluate_phystwin_prior_arrays,
    evaluate_phystwin_prior_files,
    write_prior_evaluation,
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


def _metric_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.zeros((3, 2, 3), dtype=float)
    trajectory = np.zeros((3, 2, 3), dtype=float)
    mask = np.ones((3, 2), dtype=bool)
    return observed, trajectory, mask


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
        ("minimum_probability", np.array([0.1]), TypeError),
        ("minimum_probability", np.nan, ValueError),
        ("minimum_probability", -0.1, ValueError),
        ("minimum_probability", 0.5, ValueError),
        ("confidence_power", -1.0, ValueError),
        ("visibility_power", -1.0, ValueError),
        ("boundary_scale", 0.0, ValueError),
        ("flow_scale", np.inf, ValueError),
        ("forward_backward_scale_px", -1.0, ValueError),
        ("multiview_scale_px", "invalid", TypeError),
        ("occlusion_probability", -0.1, ValueError),
        ("occlusion_probability", 1.1, ValueError),
        ("markov_inlier_persistence", 0.0, ValueError),
        ("markov_inlier_persistence", 1.0, ValueError),
        ("markov_outlier_persistence", 0.0, ValueError),
        ("markov_outlier_persistence", 1.0, ValueError),
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


def test_refit_config_normalizes_valid_numpy_scalars() -> None:
    config = PhysTwinRefitReliabilityConfig(
        minimum_probability=np.float64(0.01),
        confidence_power=np.int64(2),
        visibility_power=np.float32(0.5),
        boundary_scale=None,
        flow_scale=np.float64(0.02),
        forward_backward_scale_px=np.int64(3),
        multiview_scale_px=np.float32(4.0),
        occlusion_probability=np.float64(0.2),
        markov_inlier_persistence=np.float64(0.8),
        markov_outlier_persistence=np.float64(0.7),
    )

    assert config.minimum_probability == pytest.approx(0.01)
    assert config.confidence_power == pytest.approx(2.0)
    assert config.visibility_power == pytest.approx(0.5)
    assert config.boundary_scale is None
    assert config.flow_scale == pytest.approx(0.02)
    assert config.forward_backward_scale_px == pytest.approx(3.0)
    assert config.multiview_scale_px == pytest.approx(4.0)


def test_refit_objective_rejects_nonstring_variant() -> None:
    visible, motion_valid = _masks()

    with pytest.raises(ValueError, match="variant must be one of"):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant=cast("str", 0),
        )


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


def test_refit_masks_require_registered_shapes() -> None:
    visible, motion_valid = _masks()

    with pytest.raises(ValueError, match="visible must have shape"):
        build_phystwin_track_objective(
            visible[..., None],
            motion_valid,
            variant="hard",
        )
    with pytest.raises(ValueError, match="motion_valid must have shape"):
        build_phystwin_track_objective(
            visible,
            motion_valid[:1],
            variant="hard",
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


def test_interframe_boolean_cues_align_to_target_frames() -> None:
    visible, motion_valid = _masks()
    interframe_valid = np.array([[True, False], [False, True]])
    interframe_error = np.ones((2, 2), dtype=float)

    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        variant="cue",
        cues={
            "forward_backward_valid": interframe_valid,
            "forward_backward_error_px": interframe_error,
        },
        config=PhysTwinRefitReliabilityConfig(
            boundary_scale=None,
            flow_scale=None,
            forward_backward_scale_px=1.0,
        ),
    )

    assert objective.weights[1, 0] == pytest.approx(np.exp(-1.0))
    assert objective.weights[2, 1] == pytest.approx(np.exp(-1.0))


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("confidence", np.full((3, 2), 1.1), "must lie in"),
        ("visibility_probability", np.full((3, 2), -0.1), "must lie in"),
        ("flow_inconsistency", np.full((3, 2), -0.1), "nonnegative"),
        ("boundary_distance", np.full((3, 2), np.nan), "finite"),
        ("confidence", np.full((3, 2), "bad"), "real numeric"),
    ],
)
def test_numeric_cues_reject_invalid_scientific_values(
    name: str,
    values: np.ndarray,
    message: str,
) -> None:
    visible, motion_valid = _masks()
    error = TypeError if values.dtype.kind not in "iuf" else ValueError

    with pytest.raises(error, match=message):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="cue",
            cues={name: values},
        )


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("confidence", np.ones((1, 1), dtype=float)),
        ("occluded", np.ones((1, 1), dtype=bool)),
    ],
)
def test_cues_require_frame_aligned_shapes(name: str, values: np.ndarray) -> None:
    visible, motion_valid = _masks()

    with pytest.raises(ValueError, match=rf"cue {name} must have shape"):
        build_phystwin_track_objective(
            visible,
            motion_valid,
            variant="cue",
            cues={name: values},
        )


@pytest.mark.parametrize("variant", ["visible", "markov_cue"])
def test_remaining_refit_variants_preserve_registered_support(variant: str) -> None:
    visible, motion_valid = _masks()

    objective = build_phystwin_track_objective(
        visible,
        motion_valid,
        variant=variant,
    )

    np.testing.assert_array_equal(objective.support, visible.astype(np.int32))
    assert np.all(objective.normalizer >= 1.0)


@pytest.mark.parametrize(
    ("prior", "kwargs", "message"),
    [
        (np.array([0.5]), {}, "shape"),
        (np.array([["bad"]]), {}, "real numeric"),
        (np.array([[0.5]]), {"inlier_persistence": 0.0}, "inlier_persistence"),
        (np.array([[0.5]]), {"outlier_persistence": 1.0}, "outlier_persistence"),
        (np.array([[0.5]]), {"probability_floor": 0.0}, "probability_floor"),
        (np.array([[0.5]]), {"probability_floor": 0.5}, "probability_floor"),
    ],
)
def test_markov_reliability_rejects_invalid_inputs(
    prior: np.ndarray,
    kwargs: dict[str, float],
    message: str,
) -> None:
    error = TypeError if prior.dtype.kind not in "iuf" else ValueError

    with pytest.raises(error, match=message):
        causal_markov_cue_reliability(prior, **kwargs)


def test_markov_reliability_rejects_probability_contract_violations() -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        causal_markov_cue_reliability(np.array([[0.5, 1.1]]))
    with pytest.raises(TypeError, match="real scalar"):
        causal_markov_cue_reliability(
            np.array([[0.5]]),
            inlier_persistence=True,
        )


def test_tracking_metrics_reject_nonfinite_coordinates() -> None:
    observed, trajectory, mask = _metric_arrays()
    observed[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="observed must contain finite values"):
        phystwin_tracking_metrics(observed, trajectory, mask)


@pytest.mark.parametrize(
    ("observed", "trajectory", "mask", "message"),
    [
        (np.zeros((3, 2, 2)), np.zeros((3, 2, 3)), np.ones((3, 2), bool), "observed"),
        (np.zeros((3, 2, 3)), np.zeros((3, 2, 2)), np.ones((3, 2), bool), "trajectory"),
        (
            np.zeros((3, 2, 3)),
            np.zeros((2, 2, 3)),
            np.ones((3, 2), bool),
            "fewer frames",
        ),
        (
            np.zeros((3, 2, 3)),
            np.zeros((3, 1, 3)),
            np.ones((3, 2), bool),
            "fewer vertices",
        ),
        (
            np.zeros((3, 2, 3)),
            np.zeros((3, 2, 3)),
            np.ones((2, 2), bool),
            "mask must match",
        ),
    ],
)
def test_tracking_metrics_reject_shape_mismatches(
    observed: np.ndarray,
    trajectory: np.ndarray,
    mask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        phystwin_tracking_metrics(observed, trajectory, mask)


def test_tracking_metrics_support_empty_registered_groups() -> None:
    observed, trajectory, mask = _metric_arrays()

    assert phystwin_tracking_metrics(observed, trajectory, ~mask) == {"count": 0}


def test_split_controls_require_literal_integer_contracts() -> None:
    visible, motion_valid = _masks()
    observed, trajectory, _ = _metric_arrays()

    with pytest.raises(TypeError, match="train_end_frame must be an integer"):
        evaluate_phystwin_trajectory(
            observed,
            trajectory,
            visible,
            motion_valid,
            train_end_frame=True,
        )
    with pytest.raises(TypeError, match="train_end_frame must be an integer"):
        evaluate_phystwin_trajectory(
            observed,
            trajectory,
            visible,
            motion_valid,
            train_end_frame=2.0,
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


@pytest.mark.parametrize("train_end", [1, 3])
def test_default_split_rejects_out_of_range_train_boundary(train_end: int) -> None:
    visible, motion_valid = _masks()
    observed, trajectory, _ = _metric_arrays()

    with pytest.raises(ValueError, match="between 2 and T-1"):
        evaluate_phystwin_trajectory(
            observed,
            trajectory,
            visible,
            motion_valid,
            train_end_frame=train_end,
        )


def test_default_split_requires_two_dimensional_visibility() -> None:
    visible, motion_valid = _masks()
    observed, trajectory, _ = _metric_arrays()

    with pytest.raises(ValueError, match="visible must have shape"):
        evaluate_phystwin_trajectory(
            observed,
            trajectory,
            visible[..., None],
            motion_valid,
            train_end_frame=2,
        )


@pytest.mark.parametrize(
    ("splits", "error", "message"),
    [
        ({"": (0, 1)}, TypeError, "nonempty strings"),
        ({cast("str", 1): (0, 1)}, TypeError, "nonempty strings"),
        ({"test": [0, 1]}, TypeError, "two-integer tuple"),
        ({"test": (0, 1, 2)}, TypeError, "two-integer tuple"),
        ({"test": (0, 1.0)}, TypeError, "stop must be an integer"),
        ({"test": (-1, 1)}, ValueError, "must satisfy"),
        ({"test": (1, 1)}, ValueError, "must satisfy"),
        ({"test": (1, 4)}, ValueError, "must satisfy"),
    ],
)
def test_named_splits_reject_invalid_schema(
    splits: object,
    error: type[Exception],
    message: str,
) -> None:
    visible, motion_valid = _masks()
    observed, trajectory, _ = _metric_arrays()

    with pytest.raises(error, match=message):
        evaluate_phystwin_trajectory_splits(
            observed,
            trajectory,
            visible,
            motion_valid,
            splits=cast("Mapping[str, tuple[int, int]]", splits),
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


def test_prior_file_evaluation_and_writer_preserve_validated_config(
    tmp_path: Path,
) -> None:
    visible, motion_valid = _masks()
    final_data_path = tmp_path / "final.pkl"
    cues_path = tmp_path / "cues.npz"
    output_path = tmp_path / "summary.json"
    with final_data_path.open("wb") as stream:
        pickle.dump(
            {
                "object_visibilities": visible,
                "object_motions_valid": motion_valid,
            },
            stream,
        )
    np.savez(
        cues_path,
        flow_inconsistency=np.zeros((2, 2), dtype=float),
    )
    config = PhysTwinRefitReliabilityConfig(
        boundary_scale=None,
        flow_scale=None,
    )

    summary = evaluate_phystwin_prior_files(
        final_data_path,
        cues_path,
        config=config,
    )
    write_prior_evaluation(summary, output_path)
    restored = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["config"]["reliability"]["flow_scale"] is None
    assert len(summary["inputs"]["final_data"]["sha256"]) == 64
    assert restored == summary
