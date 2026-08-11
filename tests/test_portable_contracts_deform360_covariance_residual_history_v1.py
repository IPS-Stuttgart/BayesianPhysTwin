from __future__ import annotations

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


def _policy() -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_cameras_per_role=2,
        minimum_camera_families_per_role=2,
    )


def _family_map(*, reverse: bool = False) -> CameraRecorderFamilyMapV1:
    bindings = tuple(
        (f"camera-{index:02d}", f"physical-recorder-{index:02d}")
        for index in range(8)
    )
    if reverse:
        bindings = tuple(reversed(bindings))
    return CameraRecorderFamilyMapV1(
        source_inventory_id="a" * 64,
        bindings=bindings,
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
        reconstruction_artifact_id="b" * 64,
        implementation_revision="1" * 40,
        configuration_id="c" * 64,
        input_camera_ids=partition.provider_camera_ids,
        input_source_artifact_ids=("d" * 64,),
    )
    scoring = ReconstructionManifestV1(
        role="scoring",
        source_inventory_id=family_map.source_inventory_id,
        reconstruction_artifact_id="e" * 64,
        implementation_revision="2" * 40,
        configuration_id="f" * 64,
        input_camera_ids=partition.scoring_camera_ids,
        input_source_artifact_ids=("0" * 64,),
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

    physical_future = np.zeros((3, 4, 3), dtype=np.float64)
    physical_covariance = np.zeros((3, 4, 3, 3), dtype=np.float64)
    donor_covariance = np.broadcast_to(
        0.001 * np.eye(3),
        (3, 4, 3, 3),
    ).copy()
    return {
        "physical_prefix": physical_prefix,
        "observation": observation,
        "validity": validity,
        "physical_future": physical_future,
        "physical_covariance": physical_covariance,
        "donor_covariance": donor_covariance,
        "frame_indices": np.asarray([0, 7, 14], dtype=np.int64),
        "material_ids": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "horizon_bins": np.asarray([0, 1, 2], dtype=np.int64),
    }


def _registered_mean(arrays: dict[str, np.ndarray]) -> np.ndarray:
    residual = np.zeros_like(arrays["physical_prefix"])
    validity = arrays["validity"]
    residual[validity] = (
        arrays["observation"][validity] - arrays["physical_prefix"][validity]
    )
    endpoint = np.zeros((residual.shape[1], 3), dtype=np.float64)
    for material_index in range(residual.shape[1]):
        support = np.flatnonzero(validity[:, material_index])
        if len(support):
            endpoint[material_index] = residual[support[-1], material_index]
    result = np.array(arrays["physical_future"], copy=True, order="C")
    result += endpoint[None, ...]
    return result


def _run(
    arrays: dict[str, np.ndarray],
    *,
    registered_mean: np.ndarray | None = None,
    family_map: CameraRecorderFamilyMapV1 | None = None,
    provider: ReconstructionManifestV1 | None = None,
    scoring: ReconstructionManifestV1 | None = None,
):
    selected_map = _family_map() if family_map is None else family_map
    default_provider, default_scoring = _manifests(selected_map)
    selected_provider = default_provider if provider is None else provider
    selected_scoring = default_scoring if scoring is None else scoring
    selected_mean = (
        _registered_mean(arrays) if registered_mean is None else registered_mean
    )
    return run_source_only_residual_history_dry_run(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["physical_covariance"],
        selected_mean,
        arrays["donor_covariance"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_bins=arrays["horizon_bins"],
        camera_recorder_family_map=selected_map,
        provider_reconstruction_manifest=selected_provider,
        scoring_reconstruction_manifest=selected_scoring,
        source_unit_id="opened-source-object-session-001",
        policy=_policy(),
    )


def test_partition_is_deterministic_and_recorder_family_disjoint() -> None:
    forward_map = _family_map()
    reverse_map = _family_map(reverse=True)
    forward = deterministic_disjoint_camera_partition(
        forward_map,
        policy=_policy(),
    )
    reverse = deterministic_disjoint_camera_partition(
        reverse_map,
        policy=_policy(),
    )

    assert forward_map.map_id == reverse_map.map_id
    assert forward.partition_id == reverse.partition_id
    assert set(forward.provider_camera_ids).isdisjoint(forward.scoring_camera_ids)
    assert set(forward.provider_family_ids).isdisjoint(forward.scoring_family_ids)
    assert set(forward.provider_camera_ids) | set(forward.scoring_camera_ids) == set(
        forward_map.camera_ids
    )


def test_adapter_retains_explicit_missingness_and_provenance() -> None:
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
        source_unit_id="opened-source-object-session-001",
        policy=_policy(),
    )

    np.testing.assert_array_equal(adapter.observed_validity, arrays["validity"])
    assert np.all(adapter.residual_history_m[~arrays["validity"]] == 0.0)
    assert adapter.valid_observation_count_by_material == (2, 2, 2, 2)
    assert adapter.supported_material_count == 4
    assert adapter.unsupported_material_count == 0
    assert adapter.partition.family_map is family_map
    assert adapter.provider_reconstruction_manifest is provider
    assert adapter.scoring_reconstruction_manifest is scoring
    assert not adapter.residual_history_m.flags.writeable
    assert not adapter.observed_validity.flags.writeable
    assert not adapter.supported_material_mask.flags.writeable


def test_accepted_candidate_preserves_registered_mean_object_identity() -> None:
    arrays = _arrays()
    registered_mean = _registered_mean(arrays)
    result = _run(arrays, registered_mean=registered_mean)

    expected_endpoint = np.asarray(
        [
            [0.03, 0.03, 0.03],
            [0.03, 0.03, 0.03],
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    assert result.accepted
    assert result.hybrid is not None
    assert result.mean_m is registered_mean
    assert result.hybrid.mean_m is registered_mean
    assert result.decision.hybrid_registered_mean_identity_preserved
    assert result.decision.unsupported_material_count == 0
    np.testing.assert_array_equal(
        result.mean_m,
        np.broadcast_to(expected_endpoint, result.mean_m.shape),
    )
    descriptor = result.decision.descriptor()
    assert descriptor["reference_predictor_id"] == REGISTERED_REFERENCE_PREDICTOR_ID
    assert descriptor["covariance_donor_id"] == REGISTERED_COVARIANCE_DONOR_ID
    assert descriptor["reference_mean_semantics"] == (
        "exact-caller-owned-last-residual-mean-verified-against-causal-history"
    )


def test_candidate_uses_the_registered_horizon_scales() -> None:
    arrays = _arrays()
    result = _run(arrays)

    assert REGISTERED_COVARIANCE_SCALES == (8.0, 16.0, 16.0)
    for horizon, scale in enumerate(REGISTERED_COVARIANCE_SCALES):
        np.testing.assert_allclose(
            result.covariance_m2[horizon],
            scale * arrays["donor_covariance"][horizon],
        )


def test_registered_mean_bytes_must_match_causal_history() -> None:
    arrays = _arrays()
    registered_mean = _registered_mean(arrays)
    registered_mean[0, 0, 0] += 1e-12

    with pytest.raises(ValueError, match="differs from the causal last-valid mean"):
        _run(arrays, registered_mean=registered_mean)


def test_insufficient_per_material_support_returns_exact_physical_objects() -> None:
    arrays = _arrays()
    arrays["validity"][1:, 3] = False
    arrays["observation"][1:, 3] = np.nan
    registered_mean = _registered_mean(arrays)
    result = _run(arrays, registered_mean=registered_mean)

    assert not result.accepted
    assert result.mean_m is arrays["physical_future"]
    assert result.covariance_m2 is arrays["physical_covariance"]
    assert result.decision.fallback_reasons == (
        "insufficient-per-material-support",
    )
    assert result.decision.valid_observation_count_by_material == (2, 2, 2, 1)
    assert result.decision.unsupported_material_count == 1


def test_invalid_donor_covariance_returns_exact_physical_objects() -> None:
    arrays = _arrays()
    arrays["donor_covariance"][..., 0, 0] = -1.0
    result = _run(arrays)

    assert not result.accepted
    assert result.mean_m is arrays["physical_future"]
    assert result.covariance_m2 is arrays["physical_covariance"]
    assert result.decision.fallback_reasons == ("covariance-contract-rejection",)


def test_reconstruction_source_bytes_and_lineage_must_be_disjoint() -> None:
    arrays = _arrays()
    family_map = _family_map()
    provider, scoring = _manifests(family_map)
    overlapping = replace(
        scoring,
        input_source_artifact_ids=provider.input_source_artifact_ids,
        manifest_id=None,
    )

    with pytest.raises(ValueError, match="share source bytes"):
        _run(
            arrays,
            family_map=family_map,
            provider=provider,
            scoring=overlapping,
        )

    overlapping_lineage = replace(
        scoring,
        parent_reconstruction_artifact_ids=(
            provider.reconstruction_artifact_id,
        ),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="lineages overlap"):
        _run(
            arrays,
            family_map=family_map,
            provider=provider,
            scoring=overlapping_lineage,
        )


def test_valid_rows_must_be_finite() -> None:
    arrays = _arrays()
    arrays["observation"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid provider observations"):
        _run(arrays)


def test_registered_policy_values_cannot_be_retuned() -> None:
    with pytest.raises(ValueError, match="registered schedule"):
        ResidualHistoryDryRunPolicyV1(
            minimum_prefix_frames=2,
            minimum_cameras_per_role=2,
            minimum_camera_families_per_role=2,
            covariance_scales=(1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="registered value"):
        ResidualHistoryDryRunPolicyV1(
            minimum_prefix_frames=2,
            minimum_valid_observations_per_material=1,
            minimum_cameras_per_role=2,
            minimum_camera_families_per_role=2,
        )


def test_camera_partition_rejects_inadequate_family_support() -> None:
    family_map = CameraRecorderFamilyMapV1(
        source_inventory_id="a" * 64,
        bindings=(
            ("camera-a", "recorder-a"),
            ("camera-b", "recorder-b"),
        ),
    )
    with pytest.raises(ValueError, match="minimum support"):
        deterministic_disjoint_camera_partition(
            family_map,
            policy=_policy(),
        )
