from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    ResidualHistoryDryRunPolicyV1,
    build_residual_history_adapter,
    deterministic_disjoint_camera_partition,
    run_source_only_residual_history_dry_run,
)


def _policy(
    *,
    minimum_count: int = 2,
    minimum_fraction: float = 0.5,
) -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_final_observed_count=minimum_count,
        minimum_final_observed_fraction=minimum_fraction,
        minimum_cameras_per_role=2,
        minimum_camera_families_per_role=2,
        covariance_scales=(8.0, 16.0, 16.0),
    )


def _camera_ids() -> list[str]:
    return [f"recorder-{index:02d}_cam0" for index in range(8)]


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


def _run(
    arrays: dict[str, np.ndarray],
    *,
    policy: ResidualHistoryDryRunPolicyV1 | None = None,
):
    selected = _policy() if policy is None else policy
    partition = deterministic_disjoint_camera_partition(
        _camera_ids(),
        policy=selected,
    )
    return run_source_only_residual_history_dry_run(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        arrays["physical_future"],
        arrays["physical_covariance"],
        arrays["donor_covariance"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_bins=arrays["horizon_bins"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="opened-source-object-session-001",
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        policy=selected,
    )


def test_partition_is_deterministic_and_recorder_family_disjoint() -> None:
    policy = _policy()
    forward = deterministic_disjoint_camera_partition(_camera_ids(), policy=policy)
    reverse = deterministic_disjoint_camera_partition(
        list(reversed(_camera_ids())),
        policy=policy,
    )

    assert forward.partition_id == reverse.partition_id
    assert set(forward.provider_camera_ids).isdisjoint(forward.scoring_camera_ids)
    assert set(forward.provider_family_ids).isdisjoint(forward.scoring_family_ids)
    assert set(forward.provider_camera_ids) | set(forward.scoring_camera_ids) == set(
        _camera_ids()
    )


def test_adapter_retains_explicit_missingness_without_filling() -> None:
    arrays = _arrays()
    policy = _policy()
    partition = deterministic_disjoint_camera_partition(_camera_ids(), policy=policy)
    adapter = build_residual_history_adapter(
        arrays["physical_prefix"],
        arrays["observation"],
        arrays["validity"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="opened-source-object-session-001",
        policy=policy,
    )

    np.testing.assert_array_equal(adapter.observed_validity, arrays["validity"])
    assert np.all(adapter.residual_history_m[~arrays["validity"]] == 0.0)
    assert not adapter.residual_history_m.flags.writeable
    assert not adapter.observed_validity.flags.writeable


def test_covariance_only_mean_uses_each_materials_last_valid_residual() -> None:
    arrays = _arrays()
    result = _run(arrays)

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
    assert result.mean_m is result.hybrid.mean_m
    np.testing.assert_allclose(
        result.mean_m,
        np.broadcast_to(expected_endpoint, result.mean_m.shape),
    )
    assert result.decision.descriptor()["reference_mean_semantics"] == (
        "physical-future-plus-last-valid-causal-residual-per-material-identity"
    )


def test_covariance_only_candidate_uses_frozen_horizon_scales() -> None:
    arrays = _arrays()
    result = _run(arrays)

    np.testing.assert_allclose(
        result.covariance_m2[0],
        8.0 * arrays["donor_covariance"][0],
    )
    np.testing.assert_allclose(
        result.covariance_m2[1],
        16.0 * arrays["donor_covariance"][1],
    )
    np.testing.assert_allclose(
        result.covariance_m2[2],
        16.0 * arrays["donor_covariance"][2],
    )


def test_insufficient_final_support_returns_exact_physical_objects() -> None:
    arrays = _arrays()
    result = _run(arrays, policy=_policy(minimum_count=3))

    assert not result.accepted
    assert result.mean_m is arrays["physical_future"]
    assert result.covariance_m2 is arrays["physical_covariance"]
    assert result.decision.fallback_reasons == ("minimum-final-observed-count",)


def test_invalid_donor_covariance_returns_exact_physical_objects() -> None:
    arrays = _arrays()
    arrays["donor_covariance"][..., 0, 0] = -1.0
    result = _run(arrays)

    assert not result.accepted
    assert result.mean_m is arrays["physical_future"]
    assert result.covariance_m2 is arrays["physical_covariance"]
    assert result.decision.fallback_reasons == ("covariance-contract-rejection",)


def test_valid_rows_must_be_finite() -> None:
    arrays = _arrays()
    arrays["observation"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid provider observations"):
        _run(arrays)


def test_camera_partition_rejects_inadequate_family_support() -> None:
    with pytest.raises(ValueError, match="minimum camera support"):
        deterministic_disjoint_camera_partition(
            ["recorder-a_cam0", "recorder-b_cam0"],
            policy=_policy(),
        )
