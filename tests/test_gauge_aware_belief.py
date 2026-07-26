import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareObservationBatch,
    decode_gauge_aware_query,
    select_gauge_aware_candidate,
    update_gauge_aware_belief,
)


def _design(mode: np.ndarray) -> np.ndarray:
    result = np.zeros((len(mode), 3, 1), dtype=np.float64)
    result[:, 0, 0] = mode
    return result


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batch(
    mode: np.ndarray,
    innovation_x: np.ndarray,
    *,
    gauge_mode: np.ndarray | None = None,
    groups: tuple[str, ...] | None = None,
    physical_response_scale_m: float = 0.05,
    anchor_state: np.ndarray | None = None,
    anchor_innovation_x: np.ndarray | None = None,
) -> GaugeAwareObservationBatch:
    count = len(mode)
    gauge = _empty_design(count) if gauge_mode is None else _design(gauge_mode)
    gauge_count = gauge.shape[2]
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = innovation_x
    kwargs = {}
    if anchor_state is not None:
        anchor_innovation = np.zeros((len(anchor_state), 3), dtype=np.float64)
        anchor_innovation[:, 0] = anchor_innovation_x
        kwargs = {
            "anchor_innovation_m": anchor_innovation,
            "anchor_covariance_m2": np.tile(
                np.eye(3) * 1e-8, (len(anchor_state), 1, 1)
            ),
            "anchor_state_jacobian": _design(anchor_state),
        }
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=_design(mode),
        gauge_jacobian=gauge,
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=_design(mode),
        gauge_prior_covariance=np.eye(gauge_count) * 0.01,
        correlation_group_ids=groups or tuple("window-0" for _ in range(count)),
        prior_reliability=np.ones(count),
        physical_response_scale_m=physical_response_scale_m,
        state_prior_covariance_m2=np.asarray([[0.01]]),
        **kwargs,
    )


def test_gauge_aware_update_separates_translation_gauge_from_local_state() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    batch = _batch(
        mode,
        0.01 * mode + 0.03,
        gauge_mode=np.ones_like(mode),
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=12.0,
        ),
    )

    assert result.accepted
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=2e-4)
    assert result.gauge_delta[0] == pytest.approx(0.03, abs=2e-4)
    assert result.identifiable_fractions[0] == pytest.approx(1.0)
    decoded = decode_gauge_aware_query(result, batch.query_state_jacobian)
    np.testing.assert_allclose(decoded[:, 0], 0.01 * mode, atol=2e-4)


def test_fully_gauge_confounded_query_falls_back_bit_exact() -> None:
    mode = np.ones(10)
    batch = _batch(mode, np.full(10, 0.03), gauge_mode=mode)

    result = update_gauge_aware_belief(batch)
    baseline = np.asarray([0.0, -0.0, 1.5], dtype=np.float32)
    selection = select_gauge_aware_candidate(
        baseline,
        np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        result,
    )

    assert not result.accepted
    assert result.reason == "no-identifiable-query-state"
    assert not selection.candidate_accepted
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.tobytes() == baseline.tobytes()


def test_independent_anchor_breaks_state_gauge_ambiguity() -> None:
    mode = np.ones(10)
    batch = _batch(
        mode,
        np.full(10, 0.03),
        gauge_mode=mode,
        anchor_state=np.ones(2),
        anchor_innovation_x=np.full(2, 0.01),
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=10.0,
        ),
    )

    assert result.accepted
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=3e-4)
    assert result.gauge_delta[0] == pytest.approx(0.02, abs=3e-4)
    assert result.diagnostics["independent_anchor_count"] == 2


def test_duplicate_rows_in_one_correlation_group_do_not_add_confidence() -> None:
    mode = np.linspace(-1.0, 1.0, 8)
    base = _batch(mode, 0.01 * mode)
    duplicated = _batch(
        np.tile(mode, 2),
        np.tile(0.01 * mode, 2),
        groups=tuple("window-0" for _ in range(16)),
    )
    config = GaugeAwareBeliefConfig(
        effective_samples_per_correlation_group=8.0,
    )

    base_result = update_gauge_aware_belief(base, config=config)
    duplicate_result = update_gauge_aware_belief(duplicated, config=config)

    assert base_result.accepted and duplicate_result.accepted
    assert duplicate_result.posterior_covariance[0, 0] == pytest.approx(
        base_result.posterior_covariance[0, 0], rel=1e-10
    )
    assert duplicate_result.diagnostics[
        "effective_observation_information_mass"
    ] == pytest.approx(8.0)


def test_student_t_update_downweights_one_factor_outlier() -> None:
    mode = np.linspace(-1.0, 1.0, 15)
    innovation = 0.01 * mode
    innovation[-1] += 0.20
    batch = _batch(mode, innovation)

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=15.0,
        ),
    )

    assert result.accepted
    assert result.robust_weights[-1] < 0.05
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=0.003)


def test_update_is_rejected_when_query_correction_exceeds_physical_response() -> None:
    mode = np.linspace(-1.0, 1.0, 12)
    batch = _batch(
        mode,
        0.08 * mode,
        physical_response_scale_m=0.01,
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=12.0,
            maximum_update_to_physical_response_ratio=2.0,
        ),
    )

    assert not result.accepted
    assert result.reason == "implausible-state-update"
    assert result.diagnostics["maximum_query_state_update_m"] > 0.02


def test_query_irrelevant_state_mode_is_not_updated() -> None:
    count = 14
    first = np.linspace(-1.0, 1.0, count)
    second = np.cos(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = first
    state[:, 1, 1] = second
    query = np.zeros((count, 3, 2), dtype=np.float64)
    query[:, 0, 0] = first
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.012 * first
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=_empty_design(count),
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=query,
        gauge_prior_covariance=np.zeros((0, 0)),
        correlation_group_ids=tuple("window-0" for _ in range(count)),
        prior_reliability=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.eye(2) * 0.01,
    )

    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(count),
        ),
    )

    assert result.accepted
    assert result.identifiable_state_transform.shape == (2, 1)
    assert result.state_coefficients[0] == pytest.approx(0.012, abs=2e-4)
    assert result.state_coefficients[1] == pytest.approx(0.0, abs=1e-12)
