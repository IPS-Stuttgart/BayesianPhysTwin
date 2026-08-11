from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    CameraRecorderFamilyMapV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    build_residual_history_adapter,
    deterministic_disjoint_camera_partition,
    run_source_only_residual_history_dry_run,
)
from bayesian_phystwin.endpoint_model_average import (
    DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _policy() -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=3,
        minimum_cameras_per_role=2,
        minimum_camera_families_per_role=2,
    )


def _family_map() -> CameraRecorderFamilyMapV1:
    return CameraRecorderFamilyMapV1(
        source_inventory_id=_digest("source-inventory"),
        bindings=tuple(
            (f"camera-{index:02d}", f"recorder-{index:02d}")
            for index in range(8)
        ),
    )


def _manifests(
    family_map: CameraRecorderFamilyMapV1,
) -> tuple[ReconstructionManifestV1, ReconstructionManifestV1]:
    partition = deterministic_disjoint_camera_partition(
        family_map,
        policy=_policy(),
    )
    provider = ReconstructionManifestV1(
        role="provider",
        source_inventory_id=family_map.source_inventory_id,
        reconstruction_artifact_id=_digest("provider-reconstruction"),
        implementation_revision="1" * 40,
        configuration_id=_digest("provider-config"),
        input_camera_ids=partition.provider_camera_ids,
        input_source_artifact_ids=tuple(
            _digest(f"provider-source:{camera}")
            for camera in partition.provider_camera_ids
        ),
        parent_reconstruction_artifact_ids=(_digest("provider-parent"),),
    )
    scoring = ReconstructionManifestV1(
        role="scoring",
        source_inventory_id=family_map.source_inventory_id,
        reconstruction_artifact_id=_digest("scoring-reconstruction"),
        implementation_revision="2" * 40,
        configuration_id=_digest("scoring-config"),
        input_camera_ids=partition.scoring_camera_ids,
        input_source_artifact_ids=tuple(
            _digest(f"scoring-source:{camera}")
            for camera in partition.scoring_camera_ids
        ),
        parent_reconstruction_artifact_ids=(_digest("scoring-parent"),),
    )
    return provider, scoring


def _arrays() -> dict[str, np.ndarray]:
    physical_prefix = np.zeros((3, 4, 3), dtype=np.float64)
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((3, 4), dtype=bool)
    validity[0] = True
    observation[0] = 0.01
    validity[1, 2:] = True
    observation[1, 2:] = 0.2
    validity[2, :2] = True
    observation[2, :2] = 0.03
    return {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": np.zeros((3, 4, 3), dtype=np.float64),
        "physical_covariance": np.zeros((3, 4, 3, 3), dtype=np.float64),
        "frame_indices": np.asarray([12, 13, 14], dtype=np.int64),
        "material_ids": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "future_frame_indices": np.asarray([15, 16, 17], dtype=np.int64),
        "horizon_bins": np.asarray([0, 1, 2], dtype=np.int64),
    }


def _registered_mean(arrays: dict[str, np.ndarray]) -> np.ndarray:
    result = np.array(arrays["physical_future"], copy=True, order="C")
    for material_index in range(arrays["validity"].shape[1]):
        support = np.flatnonzero(arrays["validity"][:, material_index])
        if len(support):
            frame = int(support[-1])
            result[:, material_index] += (
                arrays["observation"][frame, material_index]
                - arrays["physical_prefix"][frame, material_index]
            )
    return result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    registered_mean: np.ndarray | None = None,
    provider: ReconstructionManifestV1 | None = None,
    scoring: ReconstructionManifestV1 | None = None,
):
    family_map = _family_map()
    default_provider, default_scoring = _manifests(family_map)
    return run_source_only_residual_history_dry_run(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["physical_covariance"],
        (
            _registered_mean(arrays)
            if registered_mean is None
            else registered_mean
        ),
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_frame_indices=arrays["future_frame_indices"],
        future_horizon_bins=arrays["horizon_bins"],
        camera_recorder_family_map=family_map,
        provider_reconstruction_manifest=(
            default_provider if provider is None else provider
        ),
        scoring_reconstruction_manifest=(
            default_scoring if scoring is None else scoring
        ),
        source_unit_id="opened-source-unit",
        policy=_policy(),
    )


def _expected_unscaled_donor(arrays: dict[str, np.ndarray]) -> np.ndarray:
    residual = np.zeros_like(arrays["physical_prefix"])
    residual[arrays["validity"]] = (
        arrays["observation"][arrays["validity"]]
        - arrays["physical_prefix"][arrays["validity"]]
    )
    posterior = infer_model_averaged_endpoint(
        residual,
        arrays["validity"],
        end_frame=len(arrays["frame_indices"]),
        config=DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1,
    )
    steps = arrays["future_frame_indices"] - arrays["frame_indices"][-1]
    return np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=int(step),
            ).covariance_m2
            for step in steps
        ]
    )


def test_explicit_camera_partition_is_deterministic_and_disjoint() -> None:
    family_map = _family_map()
    first = deterministic_disjoint_camera_partition(family_map, policy=_policy())
    second = deterministic_disjoint_camera_partition(family_map, policy=_policy())
    assert first.partition_id == second.partition_id
    assert set(first.provider_camera_ids).isdisjoint(first.scoring_camera_ids)
    assert set(first.provider_family_ids).isdisjoint(first.scoring_family_ids)
    assert set(first.provider_camera_ids) | set(first.scoring_camera_ids) == set(
        family_map.camera_ids
    )


def test_adapter_uses_zero_only_missingness_and_complete_support() -> None:
    arrays = _arrays()
    family_map = _family_map()
    provider, scoring = _manifests(family_map)
    adapter = build_residual_history_adapter(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        camera_recorder_family_map=family_map,
        provider_reconstruction_manifest=provider,
        scoring_reconstruction_manifest=scoring,
        source_unit_id="opened-source-unit",
        policy=_policy(),
    )
    np.testing.assert_array_equal(adapter.observed_validity, arrays["validity"])
    assert np.all(adapter.residual_history_m[~arrays["validity"]] == 0.0)
    assert adapter.valid_observation_count_by_material == (2, 2, 2, 2)
    assert adapter.supported_material_count == 4
    assert adapter.unsupported_material_count == 0


def test_accepted_path_preserves_mean_and_reproduces_registered_donor() -> None:
    arrays = _arrays()
    registered = _registered_mean(arrays)
    unscaled = _expected_unscaled_donor(arrays)
    result = _run(arrays, registered_mean=registered)
    assert result.accepted
    assert result.hybrid is not None
    assert result.mean_m is registered
    assert result.hybrid.mean_m is registered
    assert result.hybrid.record.reference_predictor_id == (
        REGISTERED_REFERENCE_PREDICTOR_ID
    )
    assert result.hybrid.record.covariance_donor_id == (
        REGISTERED_COVARIANCE_DONOR_ID
    )
    np.testing.assert_allclose(result.covariance_m2[0], 8.0 * unscaled[0])
    np.testing.assert_allclose(result.covariance_m2[1], 16.0 * unscaled[1])
    np.testing.assert_allclose(result.covariance_m2[2], 16.0 * unscaled[2])
    assert result.decision.endpoint_contract_version == (
        MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION
    )
    assert result.decision.endpoint_config_id
    assert result.decision.endpoint_posterior_id
    assert len(result.decision.endpoint_prediction_ids) == 3
    assert result.decision.unscaled_donor_covariance_sha256


def test_intermittent_final_missingness_uses_last_valid_residual() -> None:
    arrays = _arrays()
    registered = _registered_mean(arrays)
    result = _run(arrays, registered_mean=registered)
    expected = np.asarray(
        [
            [0.03, 0.03, 0.03],
            [0.03, 0.03, 0.03],
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        result.mean_m,
        np.broadcast_to(expected, result.mean_m.shape),
    )


def test_registered_mean_mismatch_fails_before_admission() -> None:
    arrays = _arrays()
    registered = _registered_mean(arrays)
    registered[0, 0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="causal last-valid mean"):
        _run(arrays, registered_mean=registered)


def test_one_unsupported_material_returns_exact_whole_case_fallback() -> None:
    arrays = _arrays()
    arrays["validity"][1:, 3] = False
    arrays["observation"][1:, 3] = np.nan
    result = _run(arrays, registered_mean=_registered_mean(arrays))
    assert not result.accepted
    assert result.mean_m is arrays["physical_future"]
    assert result.covariance_m2 is arrays["physical_covariance"]
    assert result.decision.fallback_reasons == (
        "insufficient-per-material-support",
    )
    assert result.decision.unsupported_material_count == 1
    assert result.decision.endpoint_posterior_id
    assert len(result.decision.endpoint_prediction_ids) == 3


def test_reconstruction_source_bytes_and_lineage_must_be_disjoint() -> None:
    arrays = _arrays()
    family_map = _family_map()
    provider, scoring = _manifests(family_map)
    shared_source = replace(
        scoring,
        input_source_artifact_ids=(
            provider.input_source_artifact_ids[0],
            *scoring.input_source_artifact_ids[1:],
        ),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="share source bytes"):
        _run(arrays, provider=provider, scoring=shared_source)
    shared_lineage = replace(
        scoring,
        parent_reconstruction_artifact_ids=(
            provider.reconstruction_artifact_id,
        ),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="lineages overlap"):
        _run(arrays, provider=provider, scoring=shared_lineage)


def test_registered_policy_and_public_signature_are_frozen() -> None:
    policy = _policy()
    assert policy.covariance_scales == REGISTERED_COVARIANCE_SCALES
    with pytest.raises(ValueError, match="registered schedule"):
        ResidualHistoryDryRunPolicyV1(
            covariance_scales=(8.0, 8.0, 16.0)
        )
    parameters = inspect.signature(
        run_source_only_residual_history_dry_run
    ).parameters
    assert "donor_covariance_m2" not in parameters
    assert "future_frame_indices" in parameters
    assert "registered_last_residual_mean_m" in parameters


def test_decision_identity_is_tamper_evident() -> None:
    decision = _run(_arrays()).decision
    with pytest.raises(ValueError, match="endpoint model contract"):
        replace(
            decision,
            endpoint_contract_version=2,
            decision_id=None,
        )
    with pytest.raises(ValueError, match="decision_id"):
        replace(decision, decision_id="0" * 64)
