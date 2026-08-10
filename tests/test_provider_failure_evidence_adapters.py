from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2
from bayesian_phystwin._gauge_aware_contracts import (
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    PriorAwareGaugeBeliefResultV2,
    update_prior_aware_gauge_belief_v2,
)
from bayesian_phystwin.prospective_prob4d_update import ClaimBearingProb4DUpdateV1
from bayesian_phystwin.provider_failure_decomposition import (
    ProviderFailureSignalsV1,
    analyze_provider_failure_evidence,
    decompose_provider_failure,
)
from bayesian_phystwin.provider_failure_evidence_adapters import (
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA,
    CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION,
    build_provider_failure_payload_from_claim_bearing_updates,
    provider_failure_evidence_from_claim_bearing_update,
)

OBSERVATION_ID = "a" * 64
LINEARIZATION_ID = "b" * 64
PROVIDER_ID = "c" * 64
CALIBRATION_IDS = {"gauge": "d" * 64, "point": "e" * 64}


def _lineage(provider_id: str = PROVIDER_ID) -> dict[str, object]:
    return {
        "observation_artifact_id": OBSERVATION_ID,
        "linearization_artifact_id": LINEARIZATION_ID,
        "prob4d_claim_bearing_provider_manifest_id": provider_id,
        "prob4d_claim_bearing_calibration_artifact_ids": CALIBRATION_IDS,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _batch(provider_id: str = PROVIDER_ID) -> GaugeAwareObservationBatch:
    count = 12
    mode = np.linspace(-1.0, 1.0, count)
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.006 * mode
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = mode
    local_gauge = np.zeros((count, 3, 1), dtype=np.float64)
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.repeat(
            (np.eye(3) * 0.01)[None], count, axis=0
        ),
        state_jacobian=state,
        gauge_jacobian=local_gauge,
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.asarray([[0.09]]),
        correlation_group_ids=("group-0",) * count,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        physical_response_scale_m=1.0,
        state_prior_covariance_m2=np.asarray([[0.04]]),
        metadata=_lineage(provider_id),
    )


def _config(**changes: object) -> PriorAwareGaugeConfigV1:
    values: dict[str, object] = {
        "effective_samples_per_correlation_group": 12.0,
        "maximum_iterations": 100,
        "convergence_tolerance": 1.0e-12,
        "minimum_conditional_information_fraction": 0.0,
        "minimum_identifiable_fraction": 1.0e-8,
        "minimum_query_sensitivity_fraction": 0.0,
        "maximum_state_update_m": 1.0,
        "maximum_update_to_physical_response_ratio": 100.0,
    }
    values.update(changes)
    return replace(PriorAwareGaugeConfigV1(), **values)


def _diagnostics() -> dict[str, object]:
    return {
        "robust_likelihood_objective": "exact-group-mixture-gradient",
        "mixture_fixed_point_converged": True,
        "mixture_solution_delta": 0.0,
        "mixture_stationarity_norm": 0.0,
        "exact_reduced_mixture_hessian_minimum_eigenvalue": 1.0,
        "exact_reduced_mixture_hessian_maximum_eigenvalue": 2.0,
        "exact_reduced_mixture_hessian_positive_definite": True,
    }


def _underlying_result(
    batch: GaugeAwareObservationBatch,
    *,
    admissible: bool,
    reason: str,
) -> GaugeAwareBeliefResult:
    return GaugeAwareBeliefResult(
        inference_admissible=admissible,
        reason=reason,
        state_coefficients=np.asarray([0.005 if admissible else 0.0]),
        gauge_delta=np.zeros(1),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.eye(2) * 0.01,
        identifiable_state_transform=(
            np.ones((1, 1)) if admissible else np.zeros((1, 0))
        ),
        identifiable_fractions=np.ones(1) if admissible else np.zeros(0),
        query_sensitivity_fractions=np.ones(1) if admissible else np.zeros(0),
        robust_weights=(
            np.ones(len(batch.innovation_m))
            if admissible
            else np.zeros(len(batch.innovation_m))
        ),
        anchor_robust_weights=np.zeros(0),
        diagnostics=_diagnostics() if admissible else {},
        input_lineage=batch.metadata,
    )


def _claim_update(
    result: GaugeAwareBeliefResult,
    *,
    provider_id: str = PROVIDER_ID,
) -> ClaimBearingProb4DUpdateV1:
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=OBSERVATION_ID,
        linearization_artifact_id=LINEARIZATION_ID,
        provider_manifest_id=provider_id,
        calibration_artifact_ids=CALIBRATION_IDS,
        runtime_revision_source="independent-vcs-check",
        runtime_revision_independently_verified=True,
    )


def _accepted_update(provider_id: str = PROVIDER_ID) -> ClaimBearingProb4DUpdateV1:
    batch = _batch(provider_id)
    return _claim_update(
        update_prior_aware_gauge_belief_v2(batch, config=_config()),
        provider_id=provider_id,
    )


def _numerical_rejection_update() -> ClaimBearingProb4DUpdateV1:
    batch = _batch()
    result = update_prior_aware_gauge_belief_v2(
        batch,
        config=_config(maximum_iterations=1, convergence_tolerance=1.0e-15),
    )
    assert result.reason == "strict-v2-fixed-point-not-converged"
    return _claim_update(result)


def _underlying_rejection_update(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> ClaimBearingProb4DUpdateV1:
    batch = _batch()
    underlying = _underlying_result(batch, admissible=False, reason=reason)
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )
    return _claim_update(
        strict_v2.update_prior_aware_gauge_belief_v2(batch, config=_config())
    )


def test_adapter_derives_accepted_evidence_and_binds_all_identities() -> None:
    update = _accepted_update()

    evidence = provider_failure_evidence_from_claim_bearing_update(
        "accepted-case",
        update,
    )

    assert evidence.accepted is True
    assert evidence.signals == ProviderFailureSignalsV1(
        technical_valid=True,
        provider_support_complete=True,
        numerically_converged=True,
        query_identifiable=True,
        physical_guard_passed=True,
    )
    assert evidence.metrics["claim_bearing_update_id"] == update.update_id
    assert evidence.metrics["claim_bearing_admission_id"] == update.admission_id
    assert evidence.metrics["claim_bearing_inference_result_id"] == (
        update.inference_result_id
    )
    assert evidence.metrics["provider_manifest_id"] == PROVIDER_ID
    assert evidence.metrics["strict_admission_certificate"] == (
        update.result.diagnostics["strict_admission_certificate"]
    )
    assert decompose_provider_failure(evidence).primary_category == "accepted"


def test_adapter_localizes_strict_numerical_rejection_conservatively() -> None:
    update = _numerical_rejection_update()

    evidence = provider_failure_evidence_from_claim_bearing_update(
        "numerical-case",
        update,
    )
    attribution = decompose_provider_failure(evidence)

    assert evidence.signals == ProviderFailureSignalsV1(
        technical_valid=True,
        provider_support_complete=True,
        numerically_converged=False,
        query_identifiable=True,
        physical_guard_passed=True,
    )
    assert evidence.signals.covariance_calibrated is None
    assert evidence.signals.material_identity_reliable is None
    assert evidence.signals.robust_support_sufficient is None
    assert attribution.primary_category == "numerical-non-convergence"


@pytest.mark.parametrize(
    ("reason", "signal_name", "category"),
    [
        (
            "no-observation-support",
            "provider_support_complete",
            "unsupported-provider-geometry",
        ),
        (
            "no-identifiable-query-state",
            "query_identifiable",
            "unidentifiable-physical-query",
        ),
        (
            "implausible-state-update",
            "physical_guard_passed",
            "physical-model-or-readout-mismatch",
        ),
    ],
)
def test_adapter_preserves_underlying_semantic_rejections(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    signal_name: str,
    category: str,
) -> None:
    update = _underlying_rejection_update(monkeypatch, reason)

    evidence = provider_failure_evidence_from_claim_bearing_update("case", update)
    values = evidence.signals.to_dict()

    assert values[signal_name] is False
    assert values["technical_valid"] is True
    assert values["numerically_converged"] is None
    assert decompose_provider_failure(evidence).primary_category == category


def test_unknown_underlying_rejection_remains_explicitly_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update = _underlying_rejection_update(monkeypatch, "new-unclassified-reason")

    evidence = provider_failure_evidence_from_claim_bearing_update("unknown", update)
    attribution = decompose_provider_failure(evidence)

    assert evidence.signals.technical_valid is True
    assert sum(value is not None for value in evidence.signals.to_dict().values()) == 1
    assert attribution.primary_category == "unresolved-rejection"
    assert attribution.classification_complete is False


def test_source_signals_fill_unknowns_but_cannot_override_derived_facts() -> None:
    update = _numerical_rejection_update()

    evidence = provider_failure_evidence_from_claim_bearing_update(
        "case",
        update,
        source_signals={
            "covariance_calibrated": False,
            "material_identity_reliable": True,
        },
        metrics={"source_gate_id": "gate-v1"},
    )
    attribution = decompose_provider_failure(evidence)

    assert evidence.signals.covariance_calibrated is False
    assert evidence.signals.material_identity_reliable is True
    assert evidence.metrics["source_gate_id"] == "gate-v1"
    assert attribution.failed_categories == (
        "numerical-non-convergence",
        "provider-covariance-miscalibration",
    )

    with pytest.raises(ValueError, match="contradicts immutable"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            source_signals={"numerically_converged": True},
        )
    with pytest.raises(TypeError, match="source_signals"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            source_signals=cast(Any, []),
        )


def test_accepted_update_rejects_externally_supplied_failed_gate() -> None:
    with pytest.raises(ValueError, match="accepted case"):
        provider_failure_evidence_from_claim_bearing_update(
            "accepted",
            _accepted_update(),
            source_signals=ProviderFailureSignalsV1(covariance_calibrated=False),
        )


def test_adapter_rejects_non_strict_updates_and_invalid_metrics() -> None:
    batch = _batch()
    non_strict = _claim_update(
        _underlying_result(batch, admissible=True, reason="accepted")
    )

    with pytest.raises(TypeError, match="PriorAwareGaugeBeliefResultV2"):
        provider_failure_evidence_from_claim_bearing_update("case", non_strict)
    with pytest.raises(TypeError, match="ClaimBearingProb4DUpdateV1"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            cast(Any, object()),
        )

    update = _numerical_rejection_update()
    with pytest.raises(ValueError, match="adapter-owned"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            metrics={"claim_bearing_update_id": "replacement"},
        )
    with pytest.raises(TypeError, match="metrics must be a mapping"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            metrics=cast(Any, []),
        )
    with pytest.raises(ValueError, match="literal string keys"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            metrics=cast(Any, {1: "value"}),
        )
    with pytest.raises(ValueError, match="finite JSON values"):
        provider_failure_evidence_from_claim_bearing_update(
            "case",
            update,
            metrics={"bad": float("nan")},
        )


@pytest.mark.parametrize(
    ("certificate", "message"),
    [
        (None, "lacks an admission certificate"),
        (
            {
                "passed": 1,
                "underlying_inference_admissible": True,
                "reason": "strict-admission-passed",
            },
            "passed",
        ),
        (
            {
                "passed": True,
                "underlying_inference_admissible": 1,
                "reason": "strict-admission-passed",
            },
            "underlying_inference_admissible",
        ),
        (
            {
                "passed": True,
                "underlying_inference_admissible": True,
                "reason": "",
            },
            "reason",
        ),
    ],
)
def test_adapter_defensively_rejects_malformed_strict_result(
    certificate: object,
    message: str,
) -> None:
    update = _accepted_update()
    fake = object.__new__(PriorAwareGaugeBeliefResultV2)
    object.__setattr__(fake, "diagnostics", {"strict_admission_certificate": certificate})
    object.__setattr__(update, "result", fake)

    with pytest.raises(ValueError, match=message):
        provider_failure_evidence_from_claim_bearing_update("case", update)


def test_payload_builder_preserves_order_and_is_directly_analyzable() -> None:
    accepted = _accepted_update()
    numerical = _numerical_rejection_update()

    payload = build_provider_failure_payload_from_claim_bearing_updates(
        [("accepted", accepted), ("numerical", numerical)],
        source_signals_by_case={
            "numerical": {"covariance_calibrated": False}
        },
        metrics_by_case={"numerical": {"session_id": "s02"}},
        metadata={"split": "source-only"},
    )
    report = cast(dict[str, Any], analyze_provider_failure_evidence(payload))

    assert payload["provider_id"] == PROVIDER_ID
    assert [record["case_id"] for record in payload["records"]] == [
        "accepted",
        "numerical",
    ]
    metadata = cast(dict[str, Any], payload["metadata"])
    assert metadata["adapter_schema"] == (
        CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_SCHEMA
    )
    assert metadata["adapter_schema_version"] == (
        CLAIM_BEARING_PROVIDER_FAILURE_ADAPTER_VERSION
    )
    assert metadata["record_update_ids"] == [
        {"case_id": "accepted", "update_id": accepted.update_id},
        {"case_id": "numerical", "update_id": numerical.update_id},
    ]
    assert report["accepted_count"] == 1
    assert report["classified_rejection_count"] == 1
    assert report["primary_category_counts"]["numerical-non-convergence"] == 1
    assert report["any_category_counts"]["provider-covariance-miscalibration"] == 1


def test_payload_builder_requires_one_provider_identity() -> None:
    first = _accepted_update()
    second = _accepted_update("f" * 64)

    with pytest.raises(ValueError, match="same provider_manifest_id"):
        build_provider_failure_payload_from_claim_bearing_updates(
            [("first", first), ("second", second)]
        )


def test_payload_builder_rejects_invalid_case_shapes_and_ids() -> None:
    update = _accepted_update()

    with pytest.raises(TypeError, match="cases must be a sequence"):
        build_provider_failure_payload_from_claim_bearing_updates(cast(Any, "case"))
    with pytest.raises(ValueError, match="must not be empty"):
        build_provider_failure_payload_from_claim_bearing_updates([])
    with pytest.raises(TypeError, match="two-item sequence"):
        build_provider_failure_payload_from_claim_bearing_updates(cast(Any, [1]))
    with pytest.raises(ValueError, match="case_id and update"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cast(Any, [("case",)])
        )
    with pytest.raises(ValueError, match="invalid case_id"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cast(Any, [("", update)])
        )
    with pytest.raises(TypeError, match="ClaimBearingProb4DUpdateV1"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cast(Any, [("case", object())])
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_provider_failure_payload_from_claim_bearing_updates(
            [("case", update), ("case", update)]
        )


def test_payload_builder_rejects_unknown_or_reserved_side_inputs() -> None:
    update = _numerical_rejection_update()
    cases = [("case", update)]

    with pytest.raises(ValueError, match="unknown case IDs"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            source_signals_by_case={"other": {}},
        )
    with pytest.raises(ValueError, match="unknown case IDs"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            metrics_by_case={"other": {}},
        )
    with pytest.raises(ValueError, match="metadata cannot replace"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            metadata={"adapter_schema": "replacement"},
        )
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            metadata=cast(Any, []),
        )
    with pytest.raises(ValueError, match="literal string keys"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            source_signals_by_case=cast(Any, {1: {}}),
        )
    with pytest.raises(TypeError, match="metrics must be a mapping"):
        build_provider_failure_payload_from_claim_bearing_updates(
            cases,
            metrics_by_case=cast(Any, {"case": []}),
        )
