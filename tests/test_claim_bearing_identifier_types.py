from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._canonical_contracts import frozen_finite_json_mapping
from bayesian_phystwin.gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from bayesian_phystwin.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
    validate_prob4d_provider_attestation,
)

_REVISION = "1" * 40


def _provider_manifest() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.3.0",
        "provider_revision": _REVISION,
        "provider_api_version": 2,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    return {
        "manifest_id": compute_prob4d_provider_manifest_id(descriptor),
        **descriptor,
    }


def _provider_attestation() -> dict[str, object]:
    manifest = _provider_manifest()
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": _REVISION,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "5" * 64,
            "point_artifact_id": "6" * 64,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": _REVISION,
            "observed_revision": _REVISION,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def test_literal_provider_attestation_identifiers_validate() -> None:
    validated = validate_prob4d_provider_attestation(
        _provider_attestation(),
        source_revision=_REVISION,
        require_claim_bearing=True,
    )
    assert validated["provider_revision"] == _REVISION


def test_provider_attestation_rejects_integer_revision_that_stringifies_to_hex() -> (
    None
):
    attestation = _provider_attestation()
    attestation["provider_revision"] = int(_REVISION)

    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            attestation,
            source_revision=_REVISION,
            require_claim_bearing=True,
        )


def test_provider_attestation_rejects_integer_source_revision() -> None:
    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            _provider_attestation(),
            source_revision=int(_REVISION),  # type: ignore[arg-type]
            require_claim_bearing=True,
        )


def test_provider_attestation_rejects_integer_calibration_digest() -> None:
    attestation = deepcopy(_provider_attestation())
    calibration = attestation["calibration_artifact_ids"]
    assert isinstance(calibration, dict)
    calibration["gauge_artifact_id"] = int("5" * 64)

    with pytest.raises(ValueError, match="literal"):
        validate_prob4d_provider_attestation(
            attestation,
            source_revision=_REVISION,
            require_claim_bearing=True,
        )


def test_claim_metadata_rejects_falsey_non_mapping_root() -> None:
    for value in ([], (), "", 0, False):
        with pytest.raises(ValueError, match="metadata must be a mapping"):
            frozen_finite_json_mapping(value)  # type: ignore[arg-type]


def test_claim_metadata_rejects_non_string_keys_recursively() -> None:
    with pytest.raises(ValueError, match="literal string object keys"):
        frozen_finite_json_mapping({"nested": [{1: "value"}]})


def _gauge_design(count: int, width: int) -> np.ndarray:
    result = np.zeros((count, 3, width), dtype=np.float64)
    if width:
        result[:, 0, 0] = 1.0
    return result


def _gauge_batch(*, with_anchor: bool = False) -> GaugeAwareObservationBatch:
    anchor: dict[str, object] = {}
    if with_anchor:
        anchor = {
            "anchor_innovation_m": np.zeros((2, 3), dtype=np.float64),
            "anchor_covariance_m2": np.tile(
                np.eye(3, dtype=np.float64) * 1e-6,
                (2, 1, 1),
            ),
            "anchor_state_jacobian": _gauge_design(2, 1),
            "anchor_correlation_group_ids": ("anchor", "anchor"),
        }
    return GaugeAwareObservationBatch(
        innovation_m=np.zeros((2, 3), dtype=np.float64),
        observation_covariance_m2=np.tile(
            np.eye(3, dtype=np.float64) * 1e-6,
            (2, 1, 1),
        ),
        state_jacobian=_gauge_design(2, 1),
        gauge_jacobian=_gauge_design(2, 0),
        shared_bias_jacobian=_gauge_design(2, 0),
        view_bias_jacobian=_gauge_design(2, 0),
        query_state_jacobian=_gauge_design(1, 1),
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=("window", "window"),
        prior_reliability=np.ones(2, dtype=np.float64),
        physical_response_scale_m=0.05,
        **anchor,  # type: ignore[arg-type]
    )


class _ProviderFinalMode:
    def __str__(self) -> str:
        return COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL


def test_gauge_batch_preserves_repeated_literal_group_ids() -> None:
    batch = _gauge_batch(with_anchor=True)
    assert batch.correlation_group_ids == ("window", "window")
    assert batch.anchor_correlation_group_ids == ("anchor", "anchor")


def test_gauge_batch_rejects_non_tuple_correlation_group_ids() -> None:
    with pytest.raises(TypeError, match="tuple of exact strings"):
        replace(
            _gauge_batch(),
            correlation_group_ids=["window", "window"],  # type: ignore[arg-type]
        )


def test_gauge_batch_rejects_nonliteral_correlation_group_ids() -> None:
    with pytest.raises(TypeError, match="exact string"):
        replace(
            _gauge_batch(),
            correlation_group_ids=(1, "1"),  # type: ignore[arg-type]
        )


def test_gauge_batch_rejects_nonliteral_anchor_group_ids() -> None:
    with pytest.raises(TypeError, match="exact string"):
        replace(
            _gauge_batch(with_anchor=True),
            anchor_correlation_group_ids=(1, "1"),  # type: ignore[arg-type]
        )


def test_gauge_batch_rejects_string_like_weight_modes() -> None:
    with pytest.raises(
        TypeError,
        match="composite_weight_mode must be an exact string",
    ):
        replace(
            _gauge_batch(),
            composite_weight_mode=_ProviderFinalMode(),  # type: ignore[arg-type]
        )
    with pytest.raises(
        TypeError,
        match="anchor_composite_weight_mode must be an exact string",
    ):
        replace(
            _gauge_batch(),
            anchor_composite_weight_mode=(
                _ProviderFinalMode()  # type: ignore[arg-type]
            ),
        )
