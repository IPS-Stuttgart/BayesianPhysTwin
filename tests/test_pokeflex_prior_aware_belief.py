from dataclasses import replace

import numpy as np

from bayesian_phystwin.pokeflex_independent_depth import (
    PokeFlexIndependentDepthAnchor,
)
from bayesian_phystwin.pokeflex_prior_aware_belief import (
    PokeFlexPriorAwareConfigV1,
    build_pokeflex_prior_aware_frame_artifacts,
    infer_pokeflex_prior_aware_frame,
)


def _geometry() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 7), np.linspace(-0.02, 0.02, 5))
    z = 0.30 + 0.003 * np.sin(30.0 * x)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def _field(vertices: np.ndarray) -> np.ndarray:
    scale = vertices[:, 0] / np.max(np.abs(vertices[:, 0]))
    return (
        np.column_stack(
            (
                np.zeros(len(vertices)),
                scale,
                0.25 * np.sin(np.pi * scale),
            )
        )
        * 0.010
    )


def _anchor(
    vertices: np.ndarray,
    *,
    duplicate: int = 1,
    outlier_sensor: int | None = None,
) -> PokeFlexIndependentDepthAnchor:
    mode = _field(vertices) / 0.010
    selected = np.asarray([1, 4, 8, 11, 17, 20, 25, 31], dtype=np.int64)
    clouds = []
    sensors = []
    for sensor, bias in enumerate(([-0.002, 0.001, 0.0], [0.002, -0.001, 0.0])):
        points = vertices[selected] + 0.006 * mode[selected] + np.asarray(bias)
        if outlier_sensor == sensor:
            points = points.copy()
            points[0] += np.asarray([0.10, -0.08, 0.06])
        clouds.append(np.tile(points, (duplicate, 1)))
        sensors.append(np.full(len(points) * duplicate, sensor, dtype=np.int64))
    points = np.vstack(clouds)
    return PokeFlexIndependentDepthAnchor(
        take_id="FoamDice_T1",
        frame_id=10,
        causal_cutoff_frame=10,
        points_m=points,
        variance_m2=np.full(len(points), 0.002**2),
        sensor_index=np.concatenate(sensors),
        sensor_names=("realsense0", "realsense1"),
        calibration_sha256=("a" * 64, "b" * 64),
        metadata={
            "calibration_median_residual_m": [0.003, 0.004],
            "calibration_p90_residual_m": [0.005, 0.006],
        },
    )


def _config(**changes: object) -> PokeFlexPriorAwareConfigV1:
    return replace(
        PokeFlexPriorAwareConfigV1(
            minimum_points_per_sensor=4,
            effective_samples_per_sensor=8.0,
            state_prior_std_m=0.012,
            shared_bias_prior_std_m=0.004,
            view_bias_prior_std_m=0.003,
            maximum_state_rank=2,
            maximum_update_to_physical_response_ratio=20.0,
        ),
        **changes,
    )


def _artifacts(
    *,
    anchor: PokeFlexIndependentDepthAnchor | None = None,
    source_vertices: np.ndarray | None = None,
    config: PokeFlexPriorAwareConfigV1 | None = None,
):
    source = _geometry() if source_vertices is None else source_vertices
    target = source + np.asarray([0.0, 0.001, 0.0])
    field = _field(source)
    return build_pokeflex_prior_aware_frame_artifacts(
        anchor=_anchor(source) if anchor is None else anchor,
        baseline_source_vertices_m=source,
        baseline_target_vertices_m=target,
        source_correction_fields_m={"action-local": field},
        target_correction_fields_m={"action-local": field},
        baseline_belief_id="c" * 64,
        action_prefix_id="d" * 64,
        simulator_revision="pokeflex-checkpoint-test",
        source_revision="e" * 40,
        source_artifact_sha256="f" * 64,
        config=_config() if config is None else config,
    )


def test_assignment_probability_is_not_prior_reliability() -> None:
    source = _geometry()
    first = _artifacts(source_vertices=source)
    second = _artifacts(source_vertices=source + np.asarray([0.004, 0.0, 0.0]))

    assert not np.array_equal(
        first.observation_belief.association_probability,
        second.observation_belief.association_probability,
    )
    np.testing.assert_array_equal(
        first.observation_belief.prior_reliability,
        second.observation_belief.prior_reliability,
    )
    assert (
        first.observation_belief.metadata["assignment_probability_used_as_reliability"]
        is False
    )
    assert (
        first.observation_belief.metadata[
            "calibration_reliability_uses_state_innovation"
        ]
        is False
    )


def test_assignment_mixture_spread_enters_metric_covariance() -> None:
    single = _artifacts(config=_config(assignment_candidates=1))
    mixture = _artifacts(config=_config(assignment_candidates=4))

    single_trace = np.trace(
        single.observation_belief.local_covariance_m2, axis1=1, axis2=2
    )
    mixture_trace = np.trace(
        mixture.observation_belief.local_covariance_m2,
        axis1=1,
        axis2=2,
    )
    assert np.all(mixture_trace >= single_trace - 1e-15)
    assert np.any(mixture_trace > single_trace + 1e-10)


def test_duplicate_correlated_rows_do_not_create_arbitrary_precision() -> None:
    source = _geometry()
    config = _config()
    original = _artifacts(anchor=_anchor(source), config=config)
    duplicated = _artifacts(anchor=_anchor(source, duplicate=2), config=config)
    target = source + np.asarray([0.0, 0.001, 0.0])

    first = infer_pokeflex_prior_aware_frame(original, target, config=config)
    second = infer_pokeflex_prior_aware_frame(duplicated, target, config=config)

    assert first.result.inference_admissible
    assert second.result.inference_admissible
    assert original.observation_belief.observation_count == (
        duplicated.observation_belief.observation_count
    )
    first_variance = float(first.result.posterior_covariance[0, 0])
    second_variance = float(second.result.posterior_covariance[0, 0])
    assert second_variance == first_variance
    assert original.observation_belief.group_composite_weight[0] == 1.0
    assert duplicated.observation_belief.group_composite_weight[0] == 1.0
    assert duplicated.metadata["deduplicated_row_count"] == 16


def test_exact_duplicate_camera_is_suppressed() -> None:
    source = _geometry()
    anchor = _anchor(source)
    sensor_zero = anchor.points_m[anchor.sensor_index == 0]
    duplicated = replace(
        anchor,
        points_m=np.vstack((sensor_zero, sensor_zero)),
    )
    artifacts = _artifacts(anchor=duplicated)

    assert artifacts.metadata["duplicate_sensor_count"] == 1
    np.testing.assert_array_equal(
        np.unique(artifacts.observation_belief.view_indices),
        [0],
    )


def test_unknown_correlation_is_not_more_confident_than_naive_fusion() -> None:
    source = _geometry()
    config = _config()
    artifacts = _artifacts(config=config)
    target = source + np.asarray([0.0, 0.001, 0.0])
    conservative = infer_pokeflex_prior_aware_frame(artifacts, target, config=config)

    assert conservative.result.inference_admissible
    state_count = artifacts.linearization.state_jacobian.shape[2]
    conservative_trace = np.trace(
        conservative.result.posterior_covariance[:state_count, :state_count]
    )
    known_bias_trace = conservative.result.diagnostics[
        "known_bias_conditional_state_covariance_trace_m2"
    ]
    assert conservative_trace >= known_bias_trace
    assert (
        conservative.result.diagnostics["innovation_reprocessed_for_covariance_floor"]
        is False
    )


def test_gross_outlier_is_processed_once_by_grouped_robust_likelihood() -> None:
    source = _geometry()
    config = _config(maximum_association_m=0.20)
    clean = _artifacts(anchor=_anchor(source), config=config)
    contaminated = _artifacts(
        anchor=_anchor(source, outlier_sensor=0),
        config=config,
    )
    target = source + np.asarray([0.0, 0.001, 0.0])
    contaminated_result = infer_pokeflex_prior_aware_frame(
        contaminated,
        target,
        config=config,
    )

    np.testing.assert_array_equal(
        clean.observation_belief.prior_reliability,
        contaminated.observation_belief.prior_reliability,
    )
    assert contaminated_result.result.inference_admissible
    views = contaminated.observation_belief.view_indices
    first_weight = np.mean(contaminated_result.result.robust_weights[views == 0])
    second_weight = np.mean(contaminated_result.result.robust_weights[views == 1])
    assert first_weight < second_weight
    assert (
        contaminated_result.result.diagnostics["robust_likelihood"]
        == "grouped nominal/outlier Student-t mixture"
    )


def test_rejected_update_returns_exact_caller_baseline() -> None:
    source = _geometry()
    config = _config(
        maximum_state_update_m=0.0001,
        maximum_update_to_physical_response_ratio=0.01,
    )
    artifacts = _artifacts(config=config)
    baseline = source + np.asarray([0.0, 0.001, 0.0])
    inference = infer_pokeflex_prior_aware_frame(artifacts, baseline, config=config)

    assert not inference.result.inference_admissible
    assert inference.result.reason == "implausible-state-update"
    assert inference.select_or_exact_fallback(baseline) is baseline
