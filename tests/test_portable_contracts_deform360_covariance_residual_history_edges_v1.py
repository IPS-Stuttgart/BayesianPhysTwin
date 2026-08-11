from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.deform360_covariance_residual_history_v1 as public_api
from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    CameraRecorderFamilyMapV1,
    ReconstructionManifestV1,
    ResidualHistoryDryRunPolicyV1,
    build_residual_history_adapter,
    deterministic_disjoint_camera_partition,
    run_source_only_residual_history_dry_run,
    validate_reconstruction_separation,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _policy() -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_cameras_per_role=2,
        minimum_camera_families_per_role=2,
    )


def _family_map(
    *,
    count: int = 8,
    inventory_id: str | None = None,
    reverse: bool = False,
) -> CameraRecorderFamilyMapV1:
    bindings = tuple(
        (f"camera-{index:02d}", f"recorder-{index:02d}")
        for index in range(count)
    )
    if reverse:
        bindings = tuple(reversed(bindings))
    return CameraRecorderFamilyMapV1(
        source_inventory_id=(
            _digest("inventory") if inventory_id is None else inventory_id
        ),
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
        reconstruction_artifact_id=_digest("provider-reconstruction"),
        implementation_revision="1" * 40,
        configuration_id=_digest("provider-configuration"),
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
        configuration_id=_digest("scoring-configuration"),
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
        "donor_covariance": np.broadcast_to(
            0.001 * np.eye(3),
            (3, 4, 3, 3),
        ).copy(),
        "frame_indices": np.asarray([0, 7, 14], dtype=np.int64),
        "material_ids": np.asarray([10, 11, 12, 13], dtype=np.int64),
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
    registered_mean: object | None = None,
    family_map: CameraRecorderFamilyMapV1 | None = None,
    provider: ReconstructionManifestV1 | None = None,
    scoring: ReconstructionManifestV1 | None = None,
):
    selected_map = _family_map() if family_map is None else family_map
    default_provider, default_scoring = _manifests(selected_map)
    selected_mean = (
        _registered_mean(arrays) if registered_mean is None else registered_mean
    )
    return run_source_only_residual_history_dry_run(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["physical_covariance"],
        selected_mean,  # type: ignore[arg-type]
        arrays["donor_covariance"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_bins=arrays["horizon_bins"],
        camera_recorder_family_map=selected_map,
        provider_reconstruction_manifest=(
            default_provider if provider is None else provider
        ),
        scoring_reconstruction_manifest=(
            default_scoring if scoring is None else scoring
        ),
        source_unit_id="opened-source-unit",
        policy=_policy(),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"minimum_prefix_frames": 1},
        {"minimum_valid_observations_per_material": 1},
        {"minimum_cameras_per_role": 1},
        {"minimum_camera_families_per_role": 1},
        {"covariance_scales": (8.0, 8.0, 16.0)},
        {"covariance_scales": [8.0, 16.0, 16.0]},
    ],
)
def test_policy_rejects_unregistered_or_unsupported_values(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ResidualHistoryDryRunPolicyV1(**arguments)  # type: ignore[arg-type]


def test_policy_and_family_map_ids_are_tamper_evident() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        replace(_policy(), policy_id="0" * 64)
    with pytest.raises(ValueError, match="map_id"):
        replace(_family_map(), map_id="0" * 64)


def test_family_map_is_order_invariant_and_queryable() -> None:
    forward = _family_map()
    reverse = _family_map(reverse=True)
    assert forward.map_id == reverse.map_id
    assert forward.bindings == reverse.bindings
    assert forward.family_for_camera("camera-03") == "recorder-03"
    with pytest.raises(KeyError):
        forward.family_for_camera("missing")


@pytest.mark.parametrize(
    "bindings",
    [
        (),
        (("camera", "family"), ("camera", "other")),
        (["camera", "family"],),
        ((" camera", "family"),),
    ],
)
def test_family_map_rejects_malformed_bindings(bindings: object) -> None:
    with pytest.raises(ValueError):
        CameraRecorderFamilyMapV1(
            source_inventory_id=_digest("inventory"),
            bindings=bindings,  # type: ignore[arg-type]
        )


def test_explicit_family_map_keeps_related_streams_in_one_role() -> None:
    family_map = CameraRecorderFamilyMapV1(
        source_inventory_id=_digest("inventory"),
        bindings=(
            ("arbitrary-left", "recorder-a"),
            ("unrelated-right", "recorder-a"),
            ("camera-b", "recorder-b"),
            ("camera-c", "recorder-c"),
            ("camera-d", "recorder-d"),
            ("camera-e", "recorder-e"),
        ),
    )
    partition = deterministic_disjoint_camera_partition(
        family_map,
        policy=_policy(),
    )
    related = {"arbitrary-left", "unrelated-right"}
    assert related <= set(partition.provider_camera_ids) or related <= set(
        partition.scoring_camera_ids
    )


def test_partition_rejects_insufficient_support_and_tamper() -> None:
    with pytest.raises(ValueError, match="minimum support"):
        deterministic_disjoint_camera_partition(
            _family_map(count=3),
            policy=_policy(),
        )
    valid = deterministic_disjoint_camera_partition(_family_map(), policy=_policy())
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            valid,
            scoring_camera_ids=(
                valid.provider_camera_ids[0],
                *valid.scoring_camera_ids,
            ),
            partition_id=None,
        )
    with pytest.raises(ValueError, match="exhaust"):
        replace(
            valid,
            scoring_camera_ids=valid.scoring_camera_ids[:-1],
            partition_id=None,
        )
    with pytest.raises(ValueError, match="partition_id"):
        replace(valid, partition_id="0" * 64)


@pytest.mark.parametrize(
    "arguments",
    [
        {"role": "unknown"},
        {"implementation_revision": "x" * 40},
        {"input_camera_ids": ()},
        {"input_source_artifact_ids": ()},
    ],
)
def test_reconstruction_manifest_rejects_malformed_identity(
    arguments: dict[str, object],
) -> None:
    provider, _scoring = _manifests(_family_map())
    with pytest.raises(ValueError):
        replace(provider, manifest_id=None, **arguments)


def test_reconstruction_manifest_rejects_self_parent_and_tamper() -> None:
    provider, _scoring = _manifests(_family_map())
    with pytest.raises(ValueError, match="own parent"):
        replace(
            provider,
            parent_reconstruction_artifact_ids=(
                provider.reconstruction_artifact_id,
            ),
            manifest_id=None,
        )
    with pytest.raises(ValueError, match="manifest_id"):
        replace(provider, manifest_id="0" * 64)


def test_reconstruction_separation_rejects_wrong_identity_sets() -> None:
    family_map = _family_map()
    partition = deterministic_disjoint_camera_partition(family_map, policy=_policy())
    provider, scoring = _manifests(family_map)
    with pytest.raises(ValueError, match="roles changed"):
        validate_reconstruction_separation(partition, scoring, provider)
    wrong_inventory = replace(
        scoring,
        source_inventory_id=_digest("other-inventory"),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="different source inventory"):
        validate_reconstruction_separation(partition, provider, wrong_inventory)
    wrong_cameras = replace(
        scoring,
        input_camera_ids=scoring.input_camera_ids[:-1],
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="camera set differs"):
        validate_reconstruction_separation(partition, provider, wrong_cameras)


def test_reconstruction_separation_rejects_shared_bytes_and_lineage() -> None:
    family_map = _family_map()
    partition = deterministic_disjoint_camera_partition(family_map, policy=_policy())
    provider, scoring = _manifests(family_map)
    shared_bytes = replace(
        scoring,
        input_source_artifact_ids=(
            provider.input_source_artifact_ids[0],
            *scoring.input_source_artifact_ids[1:],
        ),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="share source bytes"):
        validate_reconstruction_separation(partition, provider, shared_bytes)
    shared_lineage = replace(
        scoring,
        parent_reconstruction_artifact_ids=(
            provider.reconstruction_artifact_id,
        ),
        manifest_id=None,
    )
    with pytest.raises(ValueError, match="lineages overlap"):
        validate_reconstruction_separation(partition, provider, shared_lineage)
    with pytest.raises(TypeError, match="partition"):
        validate_reconstruction_separation("bad", provider, scoring)  # type: ignore[arg-type]


def test_hidden_invalid_observations_do_not_change_adapter_identity() -> None:
    first_arrays = _arrays()
    second_arrays = _arrays()
    second_arrays["observation"][~second_arrays["validity"]] = 1.0e40
    family_map = _family_map()
    provider, scoring = _manifests(family_map)
    keyword = {
        "frame_indices": first_arrays["frame_indices"],
        "material_ids": first_arrays["material_ids"],
        "camera_recorder_family_map": family_map,
        "provider_reconstruction_manifest": provider,
        "scoring_reconstruction_manifest": scoring,
        "source_unit_id": "opened-source-unit",
        "policy": _policy(),
    }
    first = build_residual_history_adapter(
        first_arrays["physical_prefix"],
        first_arrays["observation"],
        first_arrays["validity"],
        **keyword,
    )
    second = build_residual_history_adapter(
        second_arrays["physical_prefix"],
        second_arrays["observation"],
        second_arrays["validity"],
        **keyword,
    )
    assert second.observation_prefix_sha256 == first.observation_prefix_sha256
    assert second.residual_history_sha256 == first.residual_history_sha256
    assert second.adapter_id == first.adapter_id


def test_adapter_rejects_frames_materials_validity_and_shapes() -> None:
    arrays = _arrays()
    arrays["frame_indices"] = np.asarray([0, 14, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(arrays)
    arrays = _arrays()
    arrays["material_ids"][-1] = arrays["material_ids"][-2]
    with pytest.raises(ValueError, match="material_ids must be unique"):
        _run(arrays)
    arrays = _arrays()
    arrays["validity"] = arrays["validity"].astype(np.int64)
    with pytest.raises(ValueError, match="Boolean array"):
        _run(arrays)
    arrays = _arrays()
    arrays["observation"] = arrays["observation"][:, :3]
    with pytest.raises(ValueError, match="shape differs"):
        _run(arrays)


@pytest.mark.parametrize(
    "registered",
    [
        "not-an-array",
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.float64),
        np.asfortranarray(np.zeros((3, 4, 3))),
        np.full((3, 4, 3), np.inf),
    ],
)
def test_registered_mean_contract_is_strict(registered: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _run(_arrays(), registered_mean=registered)


@pytest.mark.parametrize(
    "physical_future",
    [
        "not-an-array",
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.float64),
        np.asfortranarray(np.zeros((3, 4, 3))),
        np.full((3, 4, 3), np.inf),
    ],
)
def test_physical_future_contract_is_strict(physical_future: object) -> None:
    arrays = _arrays()
    arrays["physical_future"] = physical_future  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError)):
        _run(
            arrays,
            registered_mean=np.zeros((3, 4, 3), dtype=np.float64),
        )


def test_physical_covariance_contract_is_strict() -> None:
    arrays = _arrays()
    arrays["physical_covariance"] = arrays["physical_covariance"].astype(
        np.float32
    )
    with pytest.raises(ValueError, match="dtype float64"):
        _run(arrays)
    arrays = _arrays()
    arrays["physical_covariance"][..., 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        _run(arrays)
    arrays = _arrays()
    arrays["physical_covariance"][..., 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _run(arrays)


@pytest.mark.parametrize(
    "bins",
    [
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0]),
    ],
)
def test_horizon_contract_is_strict(bins: np.ndarray) -> None:
    arrays = _arrays()
    arrays["horizon_bins"] = bins
    with pytest.raises(ValueError):
        _run(arrays)


def test_artifact_and_decision_ids_are_tamper_evident() -> None:
    result = _run(_arrays())
    with pytest.raises(ValueError, match="adapter_id"):
        replace(result.adapter, adapter_id="0" * 64)
    with pytest.raises(ValueError, match="decision_id"):
        replace(result.decision, decision_id="0" * 64)
    with pytest.raises(ValueError, match="supported_material_count"):
        replace(
            result.decision,
            supported_material_count=3,
            decision_id=None,
        )


def test_public_api_is_registered_and_target_agnostic() -> None:
    parameters = inspect.signature(
        run_source_only_residual_history_dry_run
    ).parameters
    assert "registered_last_residual_mean_m" in parameters
    assert "reference_predictor_id" not in parameters
    assert "covariance_donor_id" not in parameters
    assert not hasattr(public_api, "TARGET_QUARANTINE_ROOT")
    assert not hasattr(public_api, "assert_outside_target_quarantine")
    assert not hasattr(public_api, "camera_hardware_family")
