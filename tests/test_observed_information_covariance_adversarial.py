"""Adversarial coverage for exact observed-information covariance contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import test_observed_information_covariance as cases

import bayesian_phystwin.observed_information_covariance as target
from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin.observed_information_covariance import (
    ObservedInformationCovarianceResultV1,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)


@pytest.fixture(scope="module")
def base_case() -> tuple[
    GaugeAwareObservationBatch,
    GaugeAwareBeliefResult,
    ObservedInformationCovarianceResultV1,
]:
    return cases._analysis()


def _reject_result(
    analysis: ObservedInformationCovarianceResultV1,
    message: str,
    **changes: Any,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(analysis, artifact_id=None, **changes)


def _anchor_batch(*, with_bias: bool = False) -> GaugeAwareObservationBatch:
    base = cases._batch()
    count = 4
    anchor_state = np.zeros((count, 3, 1), dtype=np.float64)
    anchor_state[:, 0, 0] = 1.0
    anchor_innovation = np.zeros((count, 3), dtype=np.float64)
    anchor_innovation[:, 0] = 0.011
    values: dict[str, Any] = {
        "anchor_innovation_m": anchor_innovation,
        "anchor_covariance_m2": np.tile(
            np.eye(3, dtype=np.float64) * 1e-6,
            (count, 1, 1),
        ),
        "anchor_state_jacobian": anchor_state,
        "anchor_correlation_group_ids": ("anchor-a",) * count,
        "anchor_prior_reliability": np.ones(count, dtype=np.float64),
        "anchor_prior_nominal_probability": np.full(
            count,
            0.9,
            dtype=np.float64,
        ),
        "anchor_composite_weight": np.ones(count, dtype=np.float64),
    }
    if with_bias:
        anchor_bias = np.zeros((count, 3, 1), dtype=np.float64)
        anchor_bias[:, 1, 0] = 1.0
        values.update(
            anchor_bias_jacobian=anchor_bias,
            anchor_bias_prior_covariance=np.asarray([[1e-4]], dtype=np.float64),
        )
    return replace(base, **values)


def _run(
    batch: GaugeAwareObservationBatch,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> tuple[GaugeAwareBeliefResult, ObservedInformationCovarianceResultV1]:
    cfg = cases._config() if config is None else config
    result = update_prior_aware_gauge_belief(batch, config=cfg)
    assert result.inference_admissible
    analysis = target.observed_information_covariance_from_prior_aware_result(
        batch,
        result,
        config=cfg,
    )
    return result, analysis


def test_private_validation_helpers_reject_noncanonical_values() -> None:
    for value in (None, "", " padded ", 1):
        with pytest.raises(ValueError, match="canonical string"):
            target._canonical_string(value, name="value")

    with pytest.raises(ValueError, match="real matrix"):
        target._finite_matrix([[object()]], name="matrix")
    for value in (np.asarray([1.0]), np.asarray([[np.inf]])):
        with pytest.raises(ValueError, match="finite matrix"):
            target._finite_matrix(value, name="matrix")

    with pytest.raises(ValueError, match="square"):
        target._symmetric_matrix(np.ones((1, 2)), name="matrix")
    with pytest.raises(ValueError, match="symmetric"):
        target._symmetric_matrix(
            np.asarray([[1.0, 2.0], [3.0, 1.0]]),
            name="matrix",
        )

    with pytest.raises(ValueError, match="real vector"):
        target._finite_vector([object()], name="vector")
    for value in (np.asarray([[1.0]]), np.asarray([np.nan])):
        with pytest.raises(ValueError, match="finite vector"):
            target._finite_vector(value, name="vector")

    with pytest.raises(ValueError, match="unique"):
        target._group_ids(("duplicate", "duplicate"), name="groups")

    for value in (True, np.bool_(False), "1"):
        with pytest.raises(ValueError, match="finite real number"):
            target._finite_real(value, name="value", minimum=1.0)
    for value in (np.inf, 0.5):
        with pytest.raises(ValueError, match="at least 1.0"):
            target._finite_real(value, name="value", minimum=1.0)


def test_result_contract_rejects_matrix_and_mapping_drift(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    _, _, analysis = base_case
    dimension = analysis.reduced_dimension
    retained = analysis.state_mapping.shape[1]

    _reject_result(
        analysis,
        "shapes changed",
        working_information=np.eye(dimension + 1),
    )
    _reject_result(
        analysis,
        "must be nonempty",
        working_information=np.empty((0, 0)),
        observed_information=np.empty((0, 0)),
        reduced_covariance=np.empty((0, 0)),
    )
    _reject_result(
        analysis,
        "working_information must be positive definite",
        working_information=-np.eye(dimension),
    )
    _reject_result(
        analysis,
        "does not invert",
        reduced_covariance=np.asarray(analysis.reduced_covariance) * 2.0,
    )
    _reject_result(
        analysis,
        "row count must match",
        state_mapping=np.zeros(
            (len(analysis.state_prior_covariance) + 1, retained),
            dtype=np.float64,
        ),
    )
    _reject_result(
        analysis,
        "retains more directions than the prior supports",
        state_mapping=np.zeros(
            (len(analysis.state_prior_covariance), dimension + 1),
            dtype=np.float64,
        ),
    )
    _reject_result(
        analysis,
        "full_covariance does not match",
        full_covariance=np.asarray(analysis.full_covariance) * 2.0,
    )


def test_result_contract_rejects_group_and_condition_drift(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    _, _, analysis = base_case

    _reject_result(
        analysis,
        "must be unique",
        observation_group_ids=("duplicate", "duplicate"),
    )
    _reject_result(
        analysis,
        "canonical string",
        anchor_group_ids=(" padded ",),
    )
    _reject_result(
        analysis,
        "observation group arrays changed length",
        observation_group_power=np.zeros(0, dtype=np.float64),
    )
    _reject_result(
        analysis,
        "anchor group arrays changed length",
        anchor_group_ids=("anchor",),
    )
    _reject_result(
        analysis,
        "group powers must be nonnegative",
        observation_group_power=-np.ones(
            len(analysis.observation_group_ids),
            dtype=np.float64,
        ),
    )
    _reject_result(
        analysis,
        "expected precisions must be nonnegative",
        observation_group_expected_precision=-np.ones(
            len(analysis.observation_group_ids),
            dtype=np.float64,
        ),
    )
    _reject_result(
        analysis,
        "condition_number does not match",
        condition_number=analysis.condition_number * 2.0,
    )


def test_result_contract_rejects_semantic_and_identity_drift(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    _, _, analysis = base_case
    semantics = analysis.covariance_semantics

    _reject_result(
        analysis,
        "must be a PosteriorCovarianceSemanticsV1",
        covariance_semantics=object(),
    )
    working_semantics = PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=analysis.full_dimension,
        likelihood_power_semantics=semantics.likelihood_power_semantics,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=False,
        group_score_correction=False,
        calibrated=False,
    )
    _reject_result(
        analysis,
        "method must be observed information",
        covariance_semantics=working_semantics,
    )
    _reject_result(
        analysis,
        "dimension changed",
        covariance_semantics=replace(
            semantics,
            dimension=semantics.dimension + 1,
            artifact_id=None,
        ),
    )

    inexact = replace(semantics, artifact_id=None)
    object.__setattr__(inexact, "mixture_curvature_exact", False)
    _reject_result(
        analysis,
        "declare exact mixture curvature",
        covariance_semantics=inexact,
    )

    corrected = replace(semantics, artifact_id=None)
    object.__setattr__(corrected, "group_score_correction", True)
    _reject_result(
        analysis,
        "flags are inconsistent",
        covariance_semantics=corrected,
    )

    with pytest.raises(ValueError):
        replace(analysis, metadata={"nonfinite": np.nan}, artifact_id=None)
    with pytest.raises(ValueError, match="artifact_id"):
        replace(analysis, artifact_id="Z" * 64)

    record = analysis.to_record()
    assert record["artifact_id"] == analysis.artifact_id
    assert record["full_dimension"] == analysis.full_dimension


def test_public_builder_rejects_invalid_inputs_and_lineage(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    batch, result, _ = base_case

    with pytest.raises(TypeError, match="batch"):
        target.observed_information_covariance_from_prior_aware_result(
            object(),
            result,
        )
    with pytest.raises(TypeError, match="result"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            object(),
        )

    inadmissible = replace(result)
    object.__setattr__(inadmissible, "inference_admissible", False)
    with pytest.raises(ValueError, match="admissible result"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            inadmissible,
            config=cases._config(),
        )

    missing_metadata = replace(batch)
    object.__setattr__(missing_metadata, "prior_nominal_probability", None)
    with pytest.raises(ValueError, match="observation mixture metadata"):
        target.observed_information_covariance_from_prior_aware_result(
            missing_metadata,
            result,
            config=cases._config(),
        )

    wrong_dimension = replace(result)
    object.__setattr__(
        wrong_dimension,
        "state_coefficients",
        np.append(np.asarray(result.state_coefficients), 0.0),
    )
    with pytest.raises(ValueError, match="dimensions do not match"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            wrong_dimension,
            config=cases._config(),
        )

    no_retained_state = replace(result)
    object.__setattr__(
        no_retained_state,
        "identifiable_state_transform",
        np.zeros((len(result.state_coefficients), 0), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="no retained state direction"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            no_retained_state,
            config=cases._config(),
        )


def test_public_builder_rejects_covariance_and_diagnostic_drift(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    batch, result, _ = base_case

    wrong_covariance = replace(result)
    object.__setattr__(
        wrong_covariance,
        "posterior_covariance",
        np.asarray(result.posterior_covariance) * 2.0,
    )
    with pytest.raises(ValueError, match="working covariance"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            wrong_covariance,
            config=cases._config(),
        )

    for key, message in (
        (
            "exact_reduced_mixture_hessian_minimum_eigenvalue",
            "minimum Hessian eigenvalue",
        ),
        (
            "exact_reduced_mixture_hessian_maximum_eigenvalue",
            "maximum Hessian eigenvalue",
        ),
    ):
        diagnostics = dict(result.diagnostics)
        diagnostics[key] = -1.0
        changed = replace(result, diagnostics=diagnostics)
        with pytest.raises(ValueError, match=message):
            target.observed_information_covariance_from_prior_aware_result(
                batch,
                changed,
                config=cases._config(),
            )

    strict_condition = replace(
        cases._config(),
        maximum_condition_number=1.0,
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            result,
            config=strict_condition,
        )


def test_public_builder_rejects_nonpositive_exact_curvature(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, result, _ = base_case
    diagnostics = {
        key: value
        for key, value in result.diagnostics.items()
        if key
        not in {
            "exact_reduced_mixture_hessian_minimum_eigenvalue",
            "exact_reduced_mixture_hessian_maximum_eigenvalue",
        }
    }
    without_exact_diagnostics = replace(result, diagnostics=diagnostics)
    original = target._student_t_mixture_statistics

    def excessive_negative_curvature(*args: Any, **kwargs: Any) -> Any:
        statistics = original(*args, **kwargs)
        return replace(
            statistics,
            expected_precision_derivative=-1e12,
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            target,
            "_student_t_mixture_statistics",
            excessive_negative_curvature,
        )
        with pytest.raises(ValueError, match="not positive definite"):
            target.observed_information_covariance_from_prior_aware_result(
                batch,
                without_exact_diagnostics,
                config=cases._config(),
            )


def test_no_nuisance_and_zero_reliability_groups_are_supported() -> None:
    base = cases._batch()
    count = len(base.innovation_m)
    no_nuisance = replace(
        base,
        gauge_jacobian=cases._empty(count),
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
    )
    _, no_nuisance_analysis = _run(no_nuisance)
    assert no_nuisance_analysis.reduced_dimension == 1

    reliability = np.ones(count, dtype=np.float64)
    reliability[-4:] = 0.0
    zero_group = replace(base, prior_reliability=reliability)
    _, zero_group_analysis = _run(zero_group)
    assert zero_group_analysis.observation_group_expected_precision[-1] == 0.0
    assert zero_group_analysis.observation_group_precision_derivative[-1] == 0.0


def test_anchor_paths_and_anchor_tampering_are_checked() -> None:
    batch = _anchor_batch()
    result, analysis = _run(batch)

    assert analysis.anchor_group_ids == ("anchor-a",)
    assert len(analysis.anchor_group_expected_precision) == 1

    missing_anchor_metadata = replace(batch)
    object.__setattr__(
        missing_anchor_metadata,
        "anchor_prior_nominal_probability",
        None,
    )
    with pytest.raises(ValueError, match="anchor mixture metadata"):
        target.observed_information_covariance_from_prior_aware_result(
            missing_anchor_metadata,
            result,
            config=cases._config(),
        )

    wrong_anchor_weights = replace(result)
    object.__setattr__(
        wrong_anchor_weights,
        "anchor_robust_weights",
        np.asarray(result.anchor_robust_weights) * 0.5,
    )
    with pytest.raises(ValueError, match="anchor robust weights"):
        target.observed_information_covariance_from_prior_aware_result(
            batch,
            wrong_anchor_weights,
            config=cases._config(),
        )


def test_anchor_bias_path_is_supported() -> None:
    _, analysis = _run(_anchor_batch(with_bias=True))

    assert analysis.full_dimension >= analysis.reduced_dimension
    assert analysis.metadata["working_covariance_kind"] is not None


def test_default_metadata_path_remains_noncalibrated(
    base_case: tuple[
        GaugeAwareObservationBatch,
        GaugeAwareBeliefResult,
        ObservedInformationCovarianceResultV1,
    ],
) -> None:
    batch, result, _ = base_case

    analysis = target.observed_information_covariance_from_prior_aware_result(
        batch,
        result,
        config=cases._config(),
    )

    assert analysis.covariance_semantics.method == "laplace_observed_information"
    assert not analysis.covariance_semantics.calibrated
    assert "study" not in analysis.metadata
