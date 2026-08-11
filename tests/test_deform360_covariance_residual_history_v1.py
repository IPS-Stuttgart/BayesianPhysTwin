from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_covariance_residual_history_v1 import (
    TARGET_QUARANTINE_ROOT,
    ResidualHistoryDryRunPolicyV1,
    assert_outside_target_quarantine,
    build_residual_history_adapter,
    deterministic_disjoint_camera_partition,
    run_source_only_residual_history_dry_run,
)
from scripts.science.run_deform360_covariance_residual_history_dry_run_v1 import (
    load_locked_policy,
    run_dry_run,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_covariance_residual_history_dry_run_v1.json"
)


def _policy(
    *,
    observed_count: int = 9,
    observed_fraction: float = 0.5,
    cameras_per_role: int = 4,
    families_per_role: int = 3,
) -> ResidualHistoryDryRunPolicyV1:
    return ResidualHistoryDryRunPolicyV1(
        minimum_prefix_frames=2,
        minimum_final_observed_count=observed_count,
        minimum_final_observed_fraction=observed_fraction,
        minimum_cameras_per_role=cameras_per_role,
        minimum_camera_families_per_role=families_per_role,
        covariance_scales=(8.0, 16.0, 16.0),
    )


def _camera_ids() -> list[str]:
    result = [f"brics-odroid-{index:03d}_cam0" for index in range(1, 17)]
    result.extend(
        [
            "brics-odroid-007_cam1",
            "brics-odroid-008_cam1",
            "brics-odroid-010_cam1",
            "brics-odroid-012_cam1",
        ]
    )
    return result


def _camera_partition(
    policy: ResidualHistoryDryRunPolicyV1,
):
    return deterministic_disjoint_camera_partition(_camera_ids(), policy=policy)


def _arrays(
    *,
    material_count: int = 18,
    final_observed_count: int = 12,
) -> dict[str, np.ndarray]:
    prefix_count = 3
    future_count = 3
    physical_prefix = np.zeros(
        (prefix_count, material_count, 3),
        dtype=np.float64,
    )
    observation = np.full_like(physical_prefix, np.nan)
    validity = np.zeros((prefix_count, material_count), dtype=bool)
    validity[0, :] = True
    validity[1, ::2] = True
    validity[-1, :final_observed_count] = True
    observation[validity] = physical_prefix[validity] + np.array([0.01, -0.02, 0.03])
    physical_future = np.zeros(
        (future_count, material_count, 3),
        dtype=np.float64,
    )
    fallback_covariance = np.zeros(
        (future_count, material_count, 3, 3),
        dtype=np.float64,
    )
    donor_covariance = np.broadcast_to(
        0.001 * np.eye(3),
        (future_count, material_count, 3, 3),
    ).copy()
    return {
        "physical_prefix_m": physical_prefix,
        "provider_observation_prefix_m": observation,
        "provider_validity": validity,
        "physical_future_m": physical_future,
        "physical_fallback_covariance_m2": fallback_covariance,
        "donor_covariance_m2": donor_covariance,
        "frame_indices": np.array([0, 7, 14], dtype=np.int64),
        "material_ids": np.arange(material_count, dtype=np.int64),
        "future_horizon_bins": np.array([0, 1, 2], dtype=np.int64),
    }


def _run(
    arrays: dict[str, np.ndarray],
    *,
    policy: ResidualHistoryDryRunPolicyV1 | None = None,
):
    selected_policy = _policy() if policy is None else policy
    partition = _camera_partition(selected_policy)
    return run_source_only_residual_history_dry_run(
        arrays["physical_prefix_m"],
        arrays["provider_observation_prefix_m"],
        arrays["provider_validity"],
        arrays["physical_future_m"],
        arrays["physical_fallback_covariance_m2"],
        arrays["donor_covariance_m2"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        future_horizon_bins=arrays["future_horizon_bins"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="opened-source-object-session-001",
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
        policy=selected_policy,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_camera_partition_is_order_invariant_exhaustive_and_family_disjoint() -> None:
    policy = _policy()
    forward = deterministic_disjoint_camera_partition(
        _camera_ids(),
        policy=policy,
    )
    reverse = deterministic_disjoint_camera_partition(
        list(reversed(_camera_ids())),
        policy=policy,
    )

    assert forward.partition_id == reverse.partition_id
    assert set(forward.provider_camera_ids).isdisjoint(forward.scoring_camera_ids)
    assert set(forward.provider_camera_ids) | set(forward.scoring_camera_ids) == set(
        _camera_ids()
    )
    assert set(forward.provider_family_ids).isdisjoint(forward.scoring_family_ids)
    for family in ("brics-odroid-007", "brics-odroid-008"):
        provider = family in forward.provider_family_ids
        scoring = family in forward.scoring_family_ids
        assert provider != scoring


def test_camera_partition_fails_when_a_role_lacks_support() -> None:
    with pytest.raises(ValueError, match="minimum camera support"):
        deterministic_disjoint_camera_partition(
            [f"device-{index}_cam0" for index in range(6)],
            policy=_policy(cameras_per_role=4, families_per_role=2),
        )


def test_adapter_preserves_validity_and_never_fills_missing_rows() -> None:
    arrays = _arrays(final_observed_count=10)
    arrays["provider_validity"][1, -1] = True
    arrays["provider_observation_prefix_m"][1, -1] = [0.5, 0.4, 0.3]
    partition = _camera_partition(_policy())
    adapter = build_residual_history_adapter(
        arrays["physical_prefix_m"],
        arrays["provider_observation_prefix_m"],
        arrays["provider_validity"],
        frame_indices=arrays["frame_indices"],
        material_ids=arrays["material_ids"],
        camera_ids=_camera_ids(),
        provider_camera_ids=partition.provider_camera_ids,
        scoring_camera_ids=partition.scoring_camera_ids,
        provider_reconstruction_artifact_id="a" * 64,
        scoring_reconstruction_artifact_id="b" * 64,
        source_unit_id="source-unit",
        policy=_policy(),
    )

    np.testing.assert_array_equal(
        adapter.observed_validity,
        arrays["provider_validity"],
    )
    assert np.all(adapter.residual_history_m[-1, -1] == 0.0)
    np.testing.assert_allclose(
        adapter.residual_history_m[1, -1],
        [0.5, 0.4, 0.3],
    )
    assert not adapter.residual_history_m.flags.writeable
    assert not adapter.observed_validity.flags.writeable


def test_adapter_rejects_nonfinite_valid_observation_and_shared_artifact() -> None:
    arrays = _arrays()
    partition = _camera_partition(_policy())
    arrays["provider_observation_prefix_m"][-1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="valid provider observations"):
        build_residual_history_adapter(
            arrays["physical_prefix_m"],
            arrays["provider_observation_prefix_m"],
            arrays["provider_validity"],
            frame_indices=arrays["frame_indices"],
            material_ids=arrays["material_ids"],
            camera_ids=_camera_ids(),
            provider_camera_ids=partition.provider_camera_ids,
            scoring_camera_ids=partition.scoring_camera_ids,
            provider_reconstruction_artifact_id="a" * 64,
            scoring_reconstruction_artifact_id="b" * 64,
            source_unit_id="source-unit",
            policy=_policy(),
        )

    arrays = _arrays()
    with pytest.raises(ValueError, match="artifacts must differ"):
        build_residual_history_adapter(
            arrays["physical_prefix_m"],
            arrays["provider_observation_prefix_m"],
            arrays["provider_validity"],
            frame_indices=arrays["frame_indices"],
            material_ids=arrays["material_ids"],
            camera_ids=_camera_ids(),
            provider_camera_ids=partition.provider_camera_ids,
            scoring_camera_ids=partition.scoring_camera_ids,
            provider_reconstruction_artifact_id="a" * 64,
            scoring_reconstruction_artifact_id="a" * 64,
            source_unit_id="source-unit",
            policy=_policy(),
        )


def test_adapter_rejects_declared_camera_rosters_that_do_not_match_partition() -> None:
    arrays = _arrays()
    policy = _policy()
    partition = _camera_partition(policy)
    wrong_provider = tuple(
        sorted((*partition.provider_camera_ids[:-1], partition.scoring_camera_ids[0]))
    )
    with pytest.raises(ValueError, match="declared provider cameras"):
        build_residual_history_adapter(
            arrays["physical_prefix_m"],
            arrays["provider_observation_prefix_m"],
            arrays["provider_validity"],
            frame_indices=arrays["frame_indices"],
            material_ids=arrays["material_ids"],
            camera_ids=_camera_ids(),
            provider_camera_ids=wrong_provider,
            scoring_camera_ids=partition.scoring_camera_ids,
            provider_reconstruction_artifact_id="a" * 64,
            scoring_reconstruction_artifact_id="b" * 64,
            source_unit_id="source-unit",
            policy=policy,
        )


def test_admitted_dry_run_uses_last_valid_residual_and_frozen_scales() -> None:
    arrays = _arrays(final_observed_count=12)
    arrays["provider_observation_prefix_m"][1, 12:] = 10.0
    arrays["provider_validity"][1, 12:] = True

    result = _run(arrays)

    assert result.accepted
    assert result.hybrid is not None
    assert result.mean_m is result.hybrid.mean_m
    assert result.decision.hybrid_reference_mean_identity_preserved
    assert result.decision.descriptor()["reference_mean_semantics"] == (
        "physical-future-plus-last-valid-causal-residual-per-material-identity"
    )
    np.testing.assert_allclose(
        result.mean_m[:, :12],
        np.broadcast_to(np.array([0.01, -0.02, 0.03]), (3, 12, 3)),
    )
    np.testing.assert_allclose(result.mean_m[:, 12:], 10.0)
    np.testing.assert_allclose(
        result.covariance_m2[0],
        8.0 * arrays["donor_covariance_m2"][0],
    )
    np.testing.assert_allclose(
        result.covariance_m2[1],
        16.0 * arrays["donor_covariance_m2"][1],
    )
    np.testing.assert_allclose(
        result.covariance_m2[2],
        16.0 * arrays["donor_covariance_m2"][2],
    )


def test_low_final_count_returns_exact_physical_fallback_objects() -> None:
    arrays = _arrays(final_observed_count=8)

    result = _run(arrays)

    assert not result.accepted
    assert result.mean_m is arrays["physical_future_m"]
    assert result.covariance_m2 is arrays["physical_fallback_covariance_m2"]
    assert result.decision.fallback_reasons == (
        "minimum-final-observed-count",
        "minimum-final-observed-fraction",
    )


def test_low_final_fraction_returns_exact_fallback_even_when_count_passes() -> None:
    arrays = _arrays(material_count=24, final_observed_count=10)

    result = _run(
        arrays,
        policy=_policy(observed_count=9, observed_fraction=0.5),
    )

    assert not result.accepted
    assert result.decision.fallback_reasons == ("minimum-final-observed-fraction",)
    assert result.mean_m is arrays["physical_future_m"]
    assert result.covariance_m2 is arrays["physical_fallback_covariance_m2"]


def test_covariance_rejection_returns_exact_fallback() -> None:
    arrays = _arrays()
    arrays["donor_covariance_m2"][..., 0, 0] = -1.0

    result = _run(arrays)

    assert not result.accepted
    assert result.decision.fallback_reasons == ("covariance-contract-rejection",)
    assert result.mean_m is arrays["physical_future_m"]
    assert result.covariance_m2 is arrays["physical_fallback_covariance_m2"]


def test_physical_fallback_covariance_itself_must_be_valid() -> None:
    arrays = _arrays()
    arrays["physical_fallback_covariance_m2"][..., 0, 0] = -1.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        _run(arrays)


def test_frame_identity_and_horizon_contracts_fail_closed() -> None:
    arrays = _arrays()
    arrays["frame_indices"] = np.array([0, 14, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(arrays)

    arrays = _arrays()
    arrays["material_ids"][-1] = arrays["material_ids"][-2]
    with pytest.raises(ValueError, match="material_ids must be unique"):
        _run(arrays)

    arrays = _arrays()
    arrays["future_horizon_bins"][-1] = 3
    with pytest.raises(ValueError, match="early/middle/late"):
        _run(arrays)


def test_target_quarantine_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="unopened target quarantine"):
        assert_outside_target_quarantine(TARGET_QUARANTINE_ROOT / "camera.mp4")
    assert assert_outside_target_quarantine("/tmp/opened-source.npz") == Path(
        "/tmp/opened-source.npz"
    )


def test_locked_policy_is_content_addressed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    protocol, policy = load_locked_policy(PROTOCOL)
    assert protocol["lock_sha256"]
    assert policy.covariance_scales == (8.0, 16.0, 16.0)

    tampered = dict(protocol)
    tampered["policy"] = dict(protocol["policy"])
    tampered["policy"]["minimum_final_observed_count"] = 8
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest changed"):
        load_locked_policy(path)


def test_cli_dry_run_publishes_no_clobber_source_only_receipt(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    protocol_partition = _camera_partition(
        _policy(cameras_per_role=8, families_per_role=4)
    )
    archive_path = tmp_path / "source.npz"
    np.savez_compressed(archive_path, **arrays)
    manifest = {
        "schema": ("bayesian-phystwin/deform360-covariance-residual-history-source-v1"),
        "schema_version": 1,
        "source_unit_id": "opened-source-object-session-001",
        "archive": {
            "path": str(archive_path),
            "sha256": _file_sha256(archive_path),
        },
        "camera_ids": _camera_ids(),
        "provider_camera_ids": list(protocol_partition.provider_camera_ids),
        "scoring_camera_ids": list(protocol_partition.scoring_camera_ids),
        "provider_reconstruction_artifact_id": "a" * 64,
        "scoring_reconstruction_artifact_id": "b" * 64,
        "information_boundary": {
            "opened_source_only": True,
            "fresh_target_payload_opened": False,
            "fresh_target_prediction_opened": False,
            "fresh_target_outcome_opened": False,
        },
    }
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"

    payload = run_dry_run(
        protocol_path=PROTOCOL,
        source_manifest_path=manifest_path,
        output_dir=output,
        implementation_revision="c" * 40,
    )

    assert payload["decision"]["accepted"] is True
    assert payload["information_boundary"]["fresh_target_quarantine_read"] is False
    assert (output / "dry_run_arrays.npz").is_file()
    assert (output / "dry_run_result.json").is_file()
    with pytest.raises(FileExistsError):
        run_dry_run(
            protocol_path=PROTOCOL,
            source_manifest_path=manifest_path,
            output_dir=output,
            implementation_revision="c" * 40,
        )
