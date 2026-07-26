from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import given, strategies as st

from bayesian_phystwin._gauge_aware_solver import _correlation_group_weights
from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefResult,
    select_gauge_aware_candidate,
)
from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)


def _belief() -> ObservationBeliefV1:
    local_covariance = np.repeat(
        np.eye(3, dtype=np.float64)[None],
        4,
        axis=0,
    ) * 1e-4
    factors = np.zeros((4, 3, 2), dtype=np.float64)
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="property-case",
        stream_id="prob4d:property-stream",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9], dtype=np.int64),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        frame_ids=np.asarray([8, 8, 9, 9], dtype=np.int64),
        entity_ids=np.asarray([0, 1, 0, 1], dtype=np.int64),
        view_indices=np.zeros(4, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        factor_group_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4, dtype=np.float64),
        local_covariance_m2=local_covariance,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


def _belief_result(*, inference_admissible: bool) -> GaugeAwareBeliefResult:
    return GaugeAwareBeliefResult(
        inference_admissible=inference_admissible,
        reason=("inference-admissible" if inference_admissible else "rejected"),
        state_coefficients=np.zeros(1, dtype=np.float64),
        gauge_delta=np.zeros(0, dtype=np.float64),
        shared_bias_coefficients=np.zeros(0, dtype=np.float64),
        view_bias_coefficients=np.zeros(0, dtype=np.float64),
        anchor_bias_coefficients=np.zeros(0, dtype=np.float64),
        posterior_covariance=np.eye(1, dtype=np.float64),
        identifiable_state_transform=np.ones((1, 1), dtype=np.float64),
        identifiable_fractions=np.ones(1, dtype=np.float64),
        query_sensitivity_fractions=np.ones(1, dtype=np.float64),
        robust_weights=np.ones(1, dtype=np.float64),
        anchor_robust_weights=np.zeros(0, dtype=np.float64),
        diagnostics={},
    )


@dataclass(frozen=True)
class _GuardDecision:
    selected_value: np.ndarray
    candidate_accepted: bool
    reason: str = "property-test"


_NONZERO_STEPS = st.one_of(
    st.integers(min_value=-10_000, max_value=-1),
    st.integers(min_value=1, max_value=10_000),
)
_FINITE_COORDINATE = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)
_PROBABILITY = st.one_of(
    st.just(0.0),
    st.floats(
        min_value=1e-6,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_POSITIVE_WEIGHT = st.floats(
    min_value=1e-6,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


@given(step=_NONZERO_STEPS)
def test_artifact_id_changes_after_any_numeric_payload_mutation(step: int) -> None:
    belief = _belief()
    changed_mean = belief.mean_xyz_m.copy()
    changed_mean[0, 0] += step * 1e-6

    changed = replace(belief, mean_xyz_m=changed_mean)

    assert changed.artifact_id != belief.artifact_id


@given(step=st.integers(min_value=-10_000, max_value=10_000))
def test_round_trip_preserves_content_address_and_readonly_arrays(step: int) -> None:
    belief = _belief()
    changed_mean = belief.mean_xyz_m.copy()
    changed_mean[0, 1] += step * 1e-6
    belief = replace(belief, mean_xyz_m=changed_mean)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "belief.npz"
        save_observation_belief(path, belief)
        restored = load_observation_belief(path)

    assert restored.artifact_id == belief.artifact_id
    restored_arrays = restored._arrays()
    for name, expected in belief._arrays().items():
        actual = restored_arrays[name]
        assert actual.flags.writeable is False
        np.testing.assert_array_equal(actual, expected)


@given(overrun=st.integers(min_value=0, max_value=100))
def test_exclusive_causal_cutoff_rejects_every_future_row(overrun: int) -> None:
    belief = _belief()
    frame_ids = belief.frame_ids.copy()
    frame_ids[-1] = belief.causal_frame_stop + overrun

    with pytest.raises(ValueError, match="causal boundary"):
        replace(belief, frame_ids=frame_ids)


@given(
    diagonal=st.floats(
        min_value=-1.0,
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_local_covariance_must_remain_positive_definite(diagonal: float) -> None:
    belief = _belief()
    covariance = belief.local_covariance_m2.copy()
    covariance[0, 0, 0] = diagonal

    with pytest.raises(ValueError, match="positive definite"):
        replace(belief, local_covariance_m2=covariance)


@given(
    values=st.lists(
        st.integers(min_value=-100_000, max_value=100_000),
        min_size=1,
        max_size=32,
    ),
    delta=st.integers(min_value=1, max_value=100),
    rejection_route=st.sampled_from(
        ("inference-rejected", "missing-guard", "guard-rejected")
    ),
)
def test_every_rejection_route_preserves_exact_baseline_bytes(
    values: list[int],
    delta: int,
    rejection_route: str,
) -> None:
    baseline = np.asarray(values, dtype=np.int32)
    candidate = baseline.copy()
    candidate[0] += delta

    if rejection_route == "inference-rejected":
        result = _belief_result(inference_admissible=False)
        decision = _GuardDecision(
            selected_value=candidate,
            candidate_accepted=True,
        )
    elif rejection_route == "missing-guard":
        result = _belief_result(inference_admissible=True)
        decision = None
    else:
        result = _belief_result(inference_admissible=True)
        decision = _GuardDecision(
            selected_value=baseline,
            candidate_accepted=False,
        )

    selection = select_gauge_aware_candidate(
        baseline,
        candidate,
        result,
        regret_decision=decision,
    )

    assert not selection.candidate_accepted
    assert selection.selected_value.shape == baseline.shape
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.tobytes() == baseline.tobytes()


@given(
    factors=st.lists(
        st.tuples(_PROBABILITY, _PROBABILITY, _POSITIVE_WEIGHT),
        min_size=1,
        max_size=32,
    ),
    effective_samples=st.integers(min_value=1, max_value=64),
)
def test_information_mass_combines_probability_terms_multiplicatively(
    factors: list[tuple[float, float, float]],
    effective_samples: int,
) -> None:
    reliability = np.asarray([value[0] for value in factors])
    nominal_probability = np.asarray([value[1] for value in factors])
    composite_weight = np.asarray([value[2] for value in factors])
    group_ids = ("shared-window",) * len(factors)

    actual, counts = _correlation_group_weights(
        group_ids,
        reliability,
        nominal_probability,
        composite_weight,
        float(effective_samples),
    )

    group_scale = min(float(effective_samples), float(len(factors))) / len(factors)
    expected = reliability * nominal_probability * composite_weight * group_scale
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert counts == {"shared-window": len(factors)}


@given(
    scale=st.floats(
        min_value=0.01,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    translation=st.tuples(
        _FINITE_COORDINATE,
        _FINITE_COORDINATE,
        _FINITE_COORDINATE,
    ),
)
def test_similarity_transform_preserves_covariance_contract(
    scale: float,
    translation: tuple[float, float, float],
) -> None:
    belief = _belief()
    transformed = belief.transformed(
        rotation=np.eye(3),
        translation_m=np.asarray(translation),
        scale=scale,
        stream_id="world",
    )

    np.testing.assert_allclose(
        transformed.local_covariance_m2,
        scale**2 * belief.local_covariance_m2,
    )
    assert np.all(
        np.linalg.eigvalsh(transformed.local_covariance_m2) > 0.0
    )
    assert transformed.local_covariance_m2.flags.writeable is False
    assert transformed.low_rank_factor_m.flags.writeable is False
    assert transformed.artifact_id != belief.artifact_id
