from __future__ import annotations

import numpy as np
import pytest

import bayesian_phystwin.causal4d_belief_provider_v2 as provider_v2
import bayesian_phystwin.causal4d_belief_provider_v3 as provider_v3
from bayesian_phystwin.dynamic_endpoint_model_average import (
    DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION,
    DynamicEndpointModelAverageConfigV2,
    PersistenceEndpointComponentV2,
)


def test_provider_v3_infers_default_dynamic_endpoint() -> None:
    posterior = provider_v3.infer_dynamic_bayesian_anchor_endpoint(
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        end_frame=2,
    )

    assert posterior.update_count[0] == 2
    assert len(posterior.config.components) == 7


def test_provider_v3_accepts_explicit_config_and_rejects_wrong_type() -> None:
    config = DynamicEndpointModelAverageConfigV2(
        components=(PersistenceEndpointComponentV2(),),
    )
    posterior = provider_v3.infer_dynamic_bayesian_anchor_endpoint(
        np.array([[[0.01, 0.0, 0.0]]]),
        np.ones((1, 1), dtype=bool),
        end_frame=1,
        config=config,
    )
    assert np.array_equal(posterior.mean_m, [[[0.01, 0.0, 0.0]]][0])

    with pytest.raises(TypeError, match="config"):
        provider_v3.infer_dynamic_bayesian_anchor_endpoint(
            np.zeros((1, 1, 3)),
            np.ones((1, 1), dtype=bool),
            end_frame=1,
            config=object(),  # type: ignore[arg-type]
        )


def test_manifest_declares_dynamic_and_recursive_capabilities() -> None:
    manifest = provider_v3.causal4d_belief_provider_v3_manifest(
        provider_revision="revision-test",
    )

    assert manifest["schema_version"] == 3
    assert manifest["provider_revision"] == "revision-test"
    capabilities = manifest["capabilities"]
    assert "exact_last_residual_component" in capabilities
    assert "robust_damped_trend_components" in capabilities
    assert "fail_closed_dynamic_covariance" in capabilities
    assert "claim_bearing_prob4d_recursive_stream" in capabilities
    schemas = manifest["artifact_schema_versions"]
    assert (
        schemas["DynamicEndpointPrediction"]
        == DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION
    )
    assert schemas["ModelAveragedEndpointConfig"] == 1
    metadata = manifest["metadata"]
    assert "independent calibration" in metadata["raw_covariance_claim"]
    assert "provider-v2 Prob4D" in metadata["recursive_stream_claim"]


def test_manifest_revision_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAYESIAN_PHYSTWIN_REVISION", "environment-revision")
    from_environment = provider_v3.causal4d_belief_provider_v3_manifest()
    assert from_environment["provider_revision"] == "environment-revision"

    monkeypatch.delenv("BAYESIAN_PHYSTWIN_REVISION")
    monkeypatch.setattr(
        provider_v3,
        "installed_distribution_revision",
        lambda _: "installed-revision",
    )
    from_install = provider_v3.causal4d_belief_provider_v3_manifest()
    assert from_install["provider_revision"] == "installed-revision"

    monkeypatch.setattr(
        provider_v3,
        "installed_distribution_revision",
        lambda _: None,
    )
    source_tree = provider_v3.causal4d_belief_provider_v3_manifest()
    assert source_tree["provider_revision"] == "unversioned-install"


def test_provider_v3_retains_v2_recursive_surface() -> None:
    assert (
        provider_v3.ClaimBearingProb4DStreamRunV1
        is provider_v2.ClaimBearingProb4DStreamRunV1
    )
    assert (
        provider_v3.apply_claim_bearing_prob4d_stream_update
        is provider_v2.apply_claim_bearing_prob4d_stream_update
    )
    assert (
        provider_v3.predict_horizon_conditioned_endpoint
        is provider_v2.predict_horizon_conditioned_endpoint
    )
