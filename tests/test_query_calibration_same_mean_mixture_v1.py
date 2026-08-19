from __future__ import annotations

import math

import numpy as np
import pytest

from bayesian_phystwin.predictive_query_mixture import (
    SameMeanGaussianMixtureCandidateV1,
    SameMeanGaussianMixtureSelectionV1,
    compose_candidate_same_mean_gaussian_mixture,
    compose_same_mean_gaussian_mixture,
    gaussian_mixture_moment_covariance,
    gaussian_mixture_negative_log_density,
    gaussian_mixture_rms_marginal_standard_deviation,
    group_gaussian_mixture_energy_score,
    group_gaussian_mixture_negative_log_score,
    select_same_mean_gaussian_mixture,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _mean(endpoint_count: int = 2, dimension: int = 1) -> np.ndarray:
    return np.zeros((endpoint_count, dimension), dtype=np.float64, order="C")


def _covariance(
    variance: float = 1.0,
    *,
    endpoint_count: int = 2,
    dimension: int = 1,
) -> np.ndarray:
    eye = np.eye(dimension, dtype=np.float64) * variance
    return np.repeat(eye[None, :, :], endpoint_count, axis=0)


def _prediction(
    *,
    probability: float = 0.9,
    nominal_variance: float = 1.0,
    tail_variance: float = 9.0,
):
    mean = _mean()
    prediction = compose_same_mean_gaussian_mixture(
        mean,
        _covariance(nominal_variance),
        _covariance(tail_variance),
        reference_predictor_id="last-residual-v1",
        nominal_covariance_id="structured-core-v1",
        tail_covariance_id="structured-tail-v1",
        nominal_probability=probability,
    )
    return mean, prediction


def test_composition_preserves_exact_mean_and_immutable_density_arrays() -> None:
    mean, prediction = _prediction()

    assert prediction.mean_m is mean
    assert prediction.record.mean_object_identity_preserved is True
    assert prediction.record.point_prediction_changed is False
    assert prediction.record.tail_dominates_nominal is True
    assert prediction.nominal_covariance_m2.flags.writeable is False
    assert prediction.tail_covariance_m2.flags.writeable is False
    assert prediction.nominal_probability.flags.writeable is False

    with pytest.raises(ValueError):
        prediction.nominal_covariance_m2.setflags(write=True)


def test_one_dimensional_log_density_matches_manual_mixture() -> None:
    _, prediction = _prediction(probability=0.8, tail_variance=4.0)
    residual = np.asarray([[0.0], [2.0]], dtype=np.float64)

    scores = gaussian_mixture_negative_log_density(residual, prediction)

    def gaussian_density(value: float, variance: float) -> float:
        return math.exp(-(value**2) / (2.0 * variance)) / math.sqrt(
            2.0 * math.pi * variance
        )

    expected = np.asarray(
        [
            -math.log(
                0.8 * gaussian_density(float(value), 1.0)
                + 0.2 * gaussian_density(float(value), 4.0)
            )
            for value in residual[:, 0]
        ]
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    assert group_gaussian_mixture_negative_log_score(
        residual,
        prediction,
    ) == pytest.approx(float(np.mean(expected)))


def test_moment_covariance_and_width_match_same_mean_mixture_identity() -> None:
    _, prediction = _prediction(
        probability=0.75,
        nominal_variance=1.0,
        tail_variance=9.0,
    )

    covariance = gaussian_mixture_moment_covariance(prediction)

    np.testing.assert_allclose(covariance[..., 0, 0], 3.0)
    assert covariance.flags.writeable is False
    assert gaussian_mixture_rms_marginal_standard_deviation(
        prediction
    ) == pytest.approx(math.sqrt(3.0))


def test_tail_must_be_broader_in_psd_order() -> None:
    mean = _mean(endpoint_count=1, dimension=2)
    nominal = np.asarray([[[2.0, 0.0], [0.0, 1.0]]])
    tail = np.asarray([[[1.0, 0.0], [0.0, 2.0]]])

    with pytest.raises(ValueError, match="dominate"):
        compose_same_mean_gaussian_mixture(
            mean,
            nominal,
            tail,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
        )


def test_density_requires_positive_definite_components_without_hidden_jitter() -> None:
    mean = _mean(endpoint_count=1, dimension=1)
    zero = _covariance(0.0, endpoint_count=1)

    with pytest.raises(ValueError, match="positive definite"):
        compose_same_mean_gaussian_mixture(
            mean,
            zero,
            zero,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
        )

    prediction = compose_same_mean_gaussian_mixture(
        mean,
        zero,
        zero,
        reference_predictor_id="reference",
        nominal_covariance_id="nominal",
        tail_covariance_id="tail",
        density_floor_variance_m2=1e-6,
    )
    np.testing.assert_allclose(prediction.nominal_covariance_m2, [[[1e-6]]])


def test_candidate_reference_reproduces_one_gaussian_exactly() -> None:
    mean = _mean()
    covariance = _covariance(2.0)
    reference = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.37,
        tail_covariance_scale=1.0,
        tail_isotropic_variance_m2=0.0,
    )
    assert reference.is_gaussian_reference is True

    prediction = compose_candidate_same_mean_gaussian_mixture(
        mean,
        covariance,
        reference,
        reference_predictor_id="last-residual-v1",
        nominal_covariance_id="core-v1",
    )

    np.testing.assert_array_equal(
        prediction.nominal_covariance_m2,
        prediction.tail_covariance_m2,
    )
    score = gaussian_mixture_negative_log_density(
        np.zeros_like(mean),
        prediction,
    )
    expected = 0.5 * math.log(2.0 * math.pi * 2.0)
    np.testing.assert_allclose(score, expected)


def test_candidate_identity_changes_with_tail_or_probability() -> None:
    first = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.9,
        tail_covariance_scale=4.0,
    )
    second = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.8,
        tail_covariance_scale=4.0,
    )
    third = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.9,
        tail_covariance_scale=9.0,
    )

    assert first.candidate_id != second.candidate_id
    assert first.candidate_id != third.candidate_id
    with pytest.raises(ValueError, match="candidate_id"):
        SameMeanGaussianMixtureCandidateV1(candidate_id=DIGEST_A)


def test_energy_score_is_deterministic_and_uses_caller_draws() -> None:
    _, prediction = _prediction(probability=0.75, tail_variance=4.0)
    residual = np.asarray([[0.0], [1.0]])
    normal_a = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
    normal_b = np.asarray([[0.5], [-0.5], [1.5], [-1.5]])
    uniform_a = np.asarray([0.1, 0.8, 0.2, 0.95])
    uniform_b = np.asarray([0.7, 0.85, 0.3, 0.99])

    first = group_gaussian_mixture_energy_score(
        residual,
        prediction,
        normal_draws_a=normal_a,
        component_uniforms_a=uniform_a,
        normal_draws_b=normal_b,
        component_uniforms_b=uniform_b,
    )
    second = group_gaussian_mixture_energy_score(
        residual,
        prediction,
        normal_draws_a=normal_a,
        component_uniforms_a=uniform_a,
        normal_draws_b=normal_b,
        component_uniforms_b=uniform_b,
    )

    assert first == second
    assert math.isfinite(first)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        group_gaussian_mixture_energy_score(
            residual,
            prediction,
            normal_draws_a=normal_a,
            component_uniforms_a=np.asarray([0.1, 1.0, 0.2, 0.3]),
            normal_draws_b=normal_b,
            component_uniforms_b=uniform_b,
        )


def test_source_selection_can_choose_a_broad_tail_without_changing_mean() -> None:
    reference = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.5,
        tail_covariance_scale=1.0,
    )
    mixture = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.9,
        tail_covariance_scale=25.0,
    )
    residual_groups = [
        np.asarray([[0.0], [0.2], [5.0]], dtype=np.float64),
        np.asarray([[0.1], [-0.2], [-4.5]], dtype=np.float64),
        np.asarray([[0.0], [0.3], [4.0]], dtype=np.float64),
    ]
    covariance_groups = [_covariance(1.0, endpoint_count=3) for _ in residual_groups]

    selection = select_same_mean_gaussian_mixture(
        development_group_ids=["object-c", "object-a", "object-b"],
        residual_groups=residual_groups,
        nominal_covariance_groups=covariance_groups,
        candidates=[mixture, reference],
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        development_evidence_id=DIGEST_D,
        reference_candidate_id=str(reference.candidate_id),
        maximum_worst_group_regret=0.0,
        maximum_width_ratio=3.0,
    )

    assert selection.selected_candidate_id == mixture.candidate_id
    assert selection.selected_reference is False
    assert selection.development_group_ids == (
        "object-a",
        "object-b",
        "object-c",
    )
    assert selection.group_negative_log_scores.flags.writeable is False
    assert selection.group_rms_marginal_standard_deviations.flags.writeable is False


def test_source_selection_retains_reference_when_one_group_is_harmed() -> None:
    reference = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.5,
        tail_covariance_scale=1.0,
    )
    broad = SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.1,
        tail_covariance_scale=100.0,
    )
    residual_groups = [
        np.zeros((4, 1), dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
    ]
    covariance_groups = [_covariance(endpoint_count=4) for _ in residual_groups]

    selection = select_same_mean_gaussian_mixture(
        development_group_ids=["a", "b"],
        residual_groups=residual_groups,
        nominal_covariance_groups=covariance_groups,
        candidates=[broad, reference],
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        development_evidence_id=DIGEST_D,
        reference_candidate_id=str(reference.candidate_id),
        maximum_worst_group_regret=0.0,
        maximum_width_ratio=20.0,
    )

    assert selection.selected_reference is True
    assert selection.selected_candidate_id == reference.candidate_id


def test_selection_rejects_target_use_and_forged_selected_candidate() -> None:
    candidate = SameMeanGaussianMixtureCandidateV1(
        tail_covariance_scale=1.0,
    )
    scores = np.asarray([[1.0, 1.0]])
    widths = np.asarray([[1.0, 1.0]])

    with pytest.raises(ValueError, match="target outcomes"):
        SameMeanGaussianMixtureSelectionV1(
            predictor_id=DIGEST_A,
            query_set_id=DIGEST_B,
            grouping_rule_id=DIGEST_C,
            development_evidence_id=DIGEST_D,
            development_group_ids=["a", "b"],
            candidates=[candidate],
            group_negative_log_scores=scores,
            group_rms_marginal_standard_deviations=widths,
            reference_candidate_id=str(candidate.candidate_id),
            selected_candidate_id=None,
            maximum_worst_group_regret=0.0,
            maximum_width_ratio=1.0,
            density_floor_variance_m2=0.0,
            grid_frozen_before_development_scores=True,
            target_outcomes_used=True,
        )

    with pytest.raises(ValueError, match="selected_candidate_id"):
        SameMeanGaussianMixtureSelectionV1(
            predictor_id=DIGEST_A,
            query_set_id=DIGEST_B,
            grouping_rule_id=DIGEST_C,
            development_evidence_id=DIGEST_D,
            development_group_ids=["a", "b"],
            candidates=[candidate],
            group_negative_log_scores=scores,
            group_rms_marginal_standard_deviations=widths,
            reference_candidate_id=str(candidate.candidate_id),
            selected_candidate_id=DIGEST_A,
            maximum_worst_group_regret=0.0,
            maximum_width_ratio=1.0,
            density_floor_variance_m2=0.0,
            grid_frozen_before_development_scores=True,
            target_outcomes_used=False,
        )


def test_input_contracts_fail_closed() -> None:
    mean = _mean()
    covariance = _covariance()

    with pytest.raises(TypeError, match="NumPy array"):
        compose_same_mean_gaussian_mixture(
            mean.tolist(),
            covariance,
            covariance,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
        )
    with pytest.raises(ValueError, match="dtype float64"):
        compose_same_mean_gaussian_mixture(
            mean.astype(np.float32),
            covariance,
            covariance,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
        )
    with pytest.raises(ValueError, match="strictly inside"):
        compose_same_mean_gaussian_mixture(
            mean,
            covariance,
            covariance,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
            nominal_probability=1.0,
        )
    with pytest.raises(ValueError, match="shape"):
        gaussian_mixture_negative_log_density(np.zeros((3, 1)), _prediction()[1])


def test_private_scalar_and_shape_validators_cover_fail_closed_edges() -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    for value in ("", " spaced "):
        with pytest.raises(ValueError, match="canonical string"):
            module._canonical_string(value, name="value")
    with pytest.raises(ValueError, match="single canonical line"):
        module._canonical_string("a\nb", name="value")
    with pytest.raises(ValueError, match="SHA-256"):
        module._sha256("abc", name="digest")
    for value in (True, float("inf")):
        with pytest.raises(ValueError, match="finite real"):
            module._finite_real(value, name="value")
    with pytest.raises(ValueError, match="at least"):
        module._finite_real(-1.0, name="value", minimum=0.0)
    with pytest.raises(ValueError, match="positive"):
        module._finite_real(0.0, name="value", strictly_positive=True)
    with pytest.raises(ValueError, match="strictly inside"):
        module._open_probability(0.0, name="probability")
    with pytest.raises(ValueError, match="integer shape"):
        module._shape("12", name="shape")
    with pytest.raises(ValueError, match="integer shape"):
        module._shape((), name="shape")
    with pytest.raises(ValueError, match="positive integer"):
        module._shape((True,), name="shape")
    with pytest.raises(ValueError, match="positive integer"):
        module._shape((0,), name="shape")


def test_private_json_and_array_validators_cover_invalid_values() -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    assert module._plain_json(np.int64(4)) == 4
    with pytest.raises(ValueError, match="finite JSON"):
        module._plain_json(float("nan"))
    with pytest.raises(ValueError, match="finite JSON"):
        module._plain_json({"bad": object()})
    with pytest.raises(ValueError, match="real numeric"):
        module._real_array(["x"], name="array")
    with pytest.raises(ValueError, match="finite"):
        module._real_array([float("inf")], name="array")


def test_reference_mean_and_covariance_validation_edges() -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    with pytest.raises(ValueError, match="shape"):
        module._reference_mean(np.empty((0,), dtype=np.float64))
    base = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        module._reference_mean(base[:, ::-1])
    nonfinite = np.zeros((1, 1), dtype=np.float64)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module._reference_mean(nonfinite)
    mean = _mean(endpoint_count=1, dimension=2)
    with pytest.raises(ValueError, match="shape"):
        module._covariance_stack(
            np.eye(2),
            mean_shape=mean.shape,
            name="covariance",
            tolerance=1e-10,
        )
    asymmetric = np.asarray([[[1.0, 1.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="symmetric"):
        module._covariance_stack(
            asymmetric,
            mean_shape=mean.shape,
            name="covariance",
            tolerance=1e-10,
        )
    with pytest.raises(ValueError, match="real numeric"):
        module._probability_schedule("x", shape=(1,))
    with pytest.raises(ValueError, match="broadcast"):
        module._probability_schedule(np.ones(2), shape=(3,))


def test_record_contract_rejects_shape_flags_probability_order_and_forgery() -> None:
    from bayesian_phystwin.predictive_query_mixture import (
        SameMeanGaussianMixtureRecordV1,
    )

    common = dict(
        reference_predictor_id="reference",
        nominal_covariance_id="nominal",
        tail_covariance_id="tail",
        mean_shape=(1, 1),
        covariance_shape=(1, 1, 1),
        reference_mean_sha256=DIGEST_A,
        nominal_covariance_sha256=DIGEST_B,
        tail_covariance_sha256=DIGEST_C,
        nominal_probability_sha256=DIGEST_D,
        minimum_nominal_probability=0.2,
        maximum_nominal_probability=0.8,
        density_floor_variance_m2=0.0,
        tail_dominates_nominal=True,
        mean_object_identity_preserved=True,
        point_prediction_changed=False,
    )
    with pytest.raises(ValueError, match="incompatible"):
        SameMeanGaussianMixtureRecordV1(**{**common, "covariance_shape": (1, 2, 2)})
    for field_name, expected in (
        ("tail_dominates_nominal", "tail_dominates"),
        ("mean_object_identity_preserved", "mean_object"),
        ("point_prediction_changed", "point_prediction"),
    ):
        with pytest.raises(ValueError, match=expected):
            SameMeanGaussianMixtureRecordV1(
                **{**common, field_name: not common[field_name]}
            )
    with pytest.raises(ValueError, match="must not be smaller"):
        SameMeanGaussianMixtureRecordV1(
            **{
                **common,
                "minimum_nominal_probability": 0.8,
                "maximum_nominal_probability": 0.2,
            }
        )
    with pytest.raises(ValueError, match="artifact_id"):
        SameMeanGaussianMixtureRecordV1(**common, artifact_id=DIGEST_A)


def test_overflow_type_and_shape_guards_are_explicit(monkeypatch) -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    mean = _mean(endpoint_count=1)
    huge = _covariance(np.finfo(np.float64).max, endpoint_count=1)
    with pytest.raises(ValueError, match="remain finite"):
        compose_same_mean_gaussian_mixture(
            mean,
            huge,
            huge,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
            density_floor_variance_m2=np.finfo(np.float64).max,
        )
    original = module._reference_mean
    monkeypatch.setattr(module, "_reference_mean", lambda value: value.copy())
    with pytest.raises(AssertionError, match="copied"):
        compose_same_mean_gaussian_mixture(
            mean,
            _covariance(endpoint_count=1),
            _covariance(endpoint_count=1),
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
            tail_covariance_id="tail",
        )
    monkeypatch.setattr(module, "_reference_mean", original)
    with pytest.raises(TypeError, match="candidate"):
        compose_candidate_same_mean_gaussian_mixture(
            mean,
            _covariance(endpoint_count=1),
            object(),
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
        )
    candidate = SameMeanGaussianMixtureCandidateV1()
    with pytest.raises(ValueError, match="shape"):
        compose_candidate_same_mean_gaussian_mixture(
            mean,
            np.eye(1),
            candidate,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
        )
    huge_candidate = SameMeanGaussianMixtureCandidateV1(
        tail_covariance_scale=np.finfo(np.float64).max,
    )
    with pytest.raises(ValueError, match="tail covariance"):
        compose_candidate_same_mean_gaussian_mixture(
            mean,
            huge,
            huge_candidate,
            reference_predictor_id="reference",
            nominal_covariance_id="nominal",
        )


def test_scoring_and_moment_type_and_numerical_guards(monkeypatch) -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    _, prediction = _prediction()
    with pytest.raises(TypeError, match="prediction"):
        gaussian_mixture_negative_log_density(np.zeros((2, 1)), object())
    monkeypatch.setattr(
        module,
        "_gaussian_log_density",
        lambda residual, covariance: np.full(residual.shape[:-1], np.inf),
    )
    with pytest.raises(ValueError, match="log density"):
        gaussian_mixture_negative_log_density(np.zeros((2, 1)), prediction)
    with pytest.raises(TypeError, match="prediction"):
        gaussian_mixture_moment_covariance(object())

    fake = module.SameMeanGaussianMixturePredictionV1(
        mean_m=prediction.mean_m,
        nominal_covariance_m2=np.full_like(
            prediction.nominal_covariance_m2,
            np.inf,
        ),
        tail_covariance_m2=prediction.tail_covariance_m2,
        nominal_probability=prediction.nominal_probability,
        record=prediction.record,
    )
    with pytest.raises(ValueError, match="moment covariance"):
        gaussian_mixture_moment_covariance(fake)
    negative = module.SameMeanGaussianMixturePredictionV1(
        mean_m=prediction.mean_m,
        nominal_covariance_m2=-prediction.nominal_covariance_m2,
        tail_covariance_m2=-prediction.tail_covariance_m2,
        nominal_probability=prediction.nominal_probability,
        record=prediction.record,
    )
    with pytest.raises(ValueError, match="marginal variances"):
        gaussian_mixture_rms_marginal_standard_deviation(negative)


def test_empty_group_and_draw_bank_validation(monkeypatch) -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    _, prediction = _prediction()
    monkeypatch.setattr(
        module,
        "gaussian_mixture_negative_log_density",
        lambda residual, value: np.asarray([], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="at least one endpoint"):
        group_gaussian_mixture_negative_log_score(np.zeros((2, 1)), prediction)

    with pytest.raises(ValueError, match="shape"):
        module._draw_bank(
            np.ones((1, 1)),
            np.zeros(1),
            dimension=1,
            name="a",
        )
    with pytest.raises(ValueError, match="one value"):
        module._draw_bank(
            np.ones((2, 1)),
            np.zeros(1),
            dimension=1,
            name="a",
        )
    with pytest.raises(ValueError, match="same size"):
        group_gaussian_mixture_energy_score(
            np.zeros((2, 1)),
            prediction,
            normal_draws_a=np.ones((2, 1)),
            component_uniforms_a=np.zeros(2),
            normal_draws_b=np.ones((3, 1)),
            component_uniforms_b=np.zeros(3),
        )


def test_group_and_candidate_sequence_validation_edges() -> None:
    import bayesian_phystwin.predictive_query_mixture as module

    with pytest.raises(ValueError, match="sequence"):
        module._canonical_group_ids("group", count=1)
    with pytest.raises(ValueError, match="length"):
        module._canonical_group_ids(["a"], count=2)
    with pytest.raises(ValueError, match="unique"):
        module._canonical_group_ids(["a", "a"], count=2)
    with pytest.raises(ValueError, match="sequence"):
        module._candidate_sequence("candidate")
    with pytest.raises(ValueError, match="must contain"):
        module._candidate_sequence([])
    with pytest.raises(ValueError, match="must contain"):
        module._candidate_sequence([object()])
    candidate = SameMeanGaussianMixtureCandidateV1()
    with pytest.raises(ValueError, match="unique"):
        module._candidate_sequence([candidate, candidate])


def test_selection_artifact_validates_arrays_reference_freeze_and_identity() -> None:
    candidate = SameMeanGaussianMixtureCandidateV1(tail_covariance_scale=1.0)
    common = dict(
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        development_evidence_id=DIGEST_D,
        development_group_ids=["a", "b"],
        candidates=[candidate],
        group_negative_log_scores=np.asarray([[1.0, 2.0]]),
        group_rms_marginal_standard_deviations=np.asarray([[1.0, 1.0]]),
        reference_candidate_id=str(candidate.candidate_id),
        selected_candidate_id=None,
        maximum_worst_group_regret=0.0,
        maximum_width_ratio=1.0,
        density_floor_variance_m2=0.0,
        grid_frozen_before_development_scores=True,
        target_outcomes_used=False,
    )
    for field_name, value, match in (
        ("group_negative_log_scores", np.asarray([1.0]), "group_negative"),
        (
            "group_rms_marginal_standard_deviations",
            np.asarray([1.0]),
            "group_rms_marginal",
        ),
        (
            "group_negative_log_scores",
            np.asarray([[np.nan, 1.0]]),
            "must be finite",
        ),
        (
            "group_rms_marginal_standard_deviations",
            np.asarray([[0.0, 1.0]]),
            "must be positive",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            SameMeanGaussianMixtureSelectionV1(**{**common, field_name: value})
    with pytest.raises(ValueError, match="must name one"):
        SameMeanGaussianMixtureSelectionV1(
            **{**common, "reference_candidate_id": DIGEST_A}
        )
    with pytest.raises(ValueError, match="grid must be frozen"):
        SameMeanGaussianMixtureSelectionV1(
            **{**common, "grid_frozen_before_development_scores": False}
        )
    valid = SameMeanGaussianMixtureSelectionV1(**common)
    assert valid.selected_candidate is candidate
    with pytest.raises(ValueError, match="artifact_id"):
        SameMeanGaussianMixtureSelectionV1(**common, artifact_id=DIGEST_A)


def test_selection_input_group_shapes_and_one_dimensional_residual_path() -> None:
    reference = SameMeanGaussianMixtureCandidateV1(tail_covariance_scale=1.0)
    common = dict(
        candidates=[reference],
        predictor_id=DIGEST_A,
        query_set_id=DIGEST_B,
        grouping_rule_id=DIGEST_C,
        development_evidence_id=DIGEST_D,
        reference_candidate_id=str(reference.candidate_id),
    )
    with pytest.raises(ValueError, match="equal nonzero"):
        select_same_mean_gaussian_mixture(
            development_group_ids=[],
            residual_groups=[],
            nominal_covariance_groups=[],
            **common,
        )
    one_dimensional = select_same_mean_gaussian_mixture(
        development_group_ids=["a"],
        residual_groups=[np.asarray([0.0])],
        nominal_covariance_groups=[np.asarray([[[1.0]]])],
        **common,
    )
    assert one_dimensional.selected_reference
    with pytest.raises(ValueError, match=r"shape \(M, D\)"):
        select_same_mean_gaussian_mixture(
            development_group_ids=["a"],
            residual_groups=[np.zeros((1, 1, 1))],
            nominal_covariance_groups=[np.asarray([[[1.0]]])],
            **common,
        )
    with pytest.raises(ValueError, match="must have shape"):
        select_same_mean_gaussian_mixture(
            development_group_ids=["a"],
            residual_groups=[np.asarray([[0.0]])],
            nominal_covariance_groups=[np.eye(1)],
            **common,
        )
