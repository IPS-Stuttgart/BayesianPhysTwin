from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.cli.deform360_adaptive_covariance_confirmation as confirmation_cli
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_failure as failure
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_measurement as measurement
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_prediction as prediction
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal as seal
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    write_confirmation_cohort_lock,
)
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_MOLMOMOTION_REVISION,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
    ALLTRACKER_SOURCE_TREE,
    RawCameraObservationConfig,
)
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
)


H1 = "a" * 40
H2 = "b" * 40


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: dict[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _external_identity(
    lock: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches = [
        {
            "case": case_id,
            "object_id": record["object_id"],
            "episode_id": episode["episode_id"],
            "episode_key": f"{record['object_id']}/{episode['episode_id']}",
            "stratum": stratum,
            "role": "calibration",
        }
        for stratum, records in lock["cohort"].items()
        for record in records
        for episode in record["episodes"]
        if episode["case_id"] == case_id
    ]
    assert len(matches) == 1
    return matches[0]


def _write_staged_case(
    tmp_path: Path,
    *,
    point_count: int = 24,
) -> tuple[Path, Path, dict[str, Any], str]:
    lock_path = tmp_path / "lock.json"
    lock = write_confirmation_cohort_lock(lock_path, H1)
    case_id = lock["selected_case_ids"][0]
    staged = tmp_path / case_id
    staged.mkdir()
    action = staged / failure.KNOWN_ACTION_RELATIVE_PATH
    action.parent.mkdir()
    action.write_bytes(b"known-action")
    splat = staged / failure.FRAME_ZERO_SPLAT_RELATIVE_PATH
    splat.parent.mkdir(parents=True)
    splat.write_bytes(b"sealed-splat")
    hull_points = np.column_stack(
        (
            np.linspace(-0.1, 0.1, point_count),
            np.linspace(0.0, 0.05, point_count),
            np.full(point_count, 2.0),
        )
    ).astype(np.float32)
    hull_colors = np.full_like(hull_points, 0.5)
    geometry = staged / failure.FRAME_ZERO_ARCHIVE_FILENAME
    np.savez_compressed(geometry, points_m=hull_points, colors=hull_colors)
    identity = _external_identity(lock, case_id)
    prefix: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwarePredictionPrefix",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "inputs_sha256": {
            "protocol": _file_sha256(lock_path),
            "source_preparation_manifest": "9" * 64,
        },
        "staged_robot_sha256": {"known_action": _file_sha256(action)},
        "information_boundary": {
            "source_object_frames_after_prefix_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    prefix["result_sha256"] = _self_hash(prefix, "result_sha256")
    prefix_path = staged / failure.PREDICTION_PREFIX_MANIFEST_FILENAME
    prefix_path.write_text(
        json.dumps(prefix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360BiasAwareFrameZeroReconstruction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": lock["artifact_sha256"],
        **identity,
        "inputs_sha256": {
            "prediction_prefix_manifest": _file_sha256(prefix_path),
        },
        "outputs_sha256": {
            "frame_zero_splat": _file_sha256(splat),
            "frame_zero_points": _file_sha256(geometry),
        },
        "material_point_count": len(hull_points),
        "material_identity_sha256": measurement._external_array_sha256(hull_points),
        "material_point_source": "strict-multiview-visual-hull-surface",
        "physical_policy": "persistence_only",
        "fallback_source_config_sha256": "1" * 64,
        "fallback_source_config_file_sha256": "2" * 64,
        "fallback_diagnostics": {
            "decision": "strict_hull_fallback",
            "original_point_count": 8,
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    frame["result_sha256"] = _self_hash(frame, "result_sha256")
    (staged / failure.FRAME_ZERO_MANIFEST_FILENAME).write_text(
        json.dumps(frame, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock_path, staged, lock, case_id


def test_frame_zero_adapter_restores_native_splat_identities_and_preserves_hull(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    lock_path, staged, lock, _ = _write_staged_case(tmp_path)
    seeded = np.column_stack(
        (
            np.linspace(-0.2, 0.2, 19),
            np.linspace(-0.03, 0.03, 19),
            np.full(19, 1.5),
        )
    ).astype(np.float32)
    colors = np.full_like(seeded, 0.25)
    fake_stage = SimpleNamespace(
        CROP_HALF_EXTENT_M=0.5,
        SEED_POINT_COUNT=5000,
        seed_points_from_splat=lambda *_args, **_kwargs: (
            seeded.copy(),
            colors.copy(),
        ),
    )
    monkeypatch.setattr(
        failure,
        "validate_deform360_execution_repository",
        lambda _path: {},
    )
    monkeypatch.setattr(
        failure,
        "_import_pinned_pcd_stage",
        lambda _path: fake_stage,
    )
    repository = tmp_path / "deform360"
    repository.mkdir()

    adapted = failure.adapt_frame_zero_original_splat_identity_persistence(
        lock_path,
        H2,
        staged,
        repository,
        expected_h1=H1,
    )

    with np.load(
        staged / failure.FRAME_ZERO_ARCHIVE_FILENAME,
        allow_pickle=False,
    ) as stored:
        np.testing.assert_array_equal(stored["points_m"], seeded)
        np.testing.assert_array_equal(stored["colors"], colors)
    marker = adapted[measurement.IDENTITY_PERSISTENCE_ADAPTER_KEY]
    assert adapted["material_point_source"] == (measurement.IDENTITY_PERSISTENCE_POLICY)
    assert adapted["physical_policy"] == "persistence_only"
    assert adapted["fallback_diagnostics"] == {
        "decision": "strict_hull_fallback",
        "original_point_count": 8,
    }
    assert marker["previous_material"]["source"] == (
        "strict-multiview-visual-hull-surface"
    )
    assert marker["adapted_material"]["point_count"] == len(seeded)
    assert marker["implementation_commit_h1"] == H1
    assert marker["cohort_lock_commit_h2"] == H2
    assert marker["cohort_lock_artifact_sha256"] == lock["artifact_sha256"]
    assert (
        failure.confirmation_frame_zero_physical_policy(
            adapted,
            original_policy=lambda _value: "automatic_twin",
        )
        == "persistence_only"
    )
    assert (
        failure.validate_original_splat_identity_persistence_manifest(
            lock_path,
            H2,
            staged,
            expected_h1=H1,
        )
        == adapted
    )
    assert (
        failure.validate_native_original_splat_frame_zero(
            lock_path,
            H2,
            staged,
            expected_h1=H1,
        )
        == adapted
    )


def test_native_identity_gate_rejects_bare_strict_hull(tmp_path: Path) -> None:
    lock_path, staged, _, _ = _write_staged_case(tmp_path)

    with pytest.raises(ValueError, match="native original-Splat identities"):
        failure.validate_native_original_splat_frame_zero(
            lock_path,
            H2,
            staged,
            expected_h1=H1,
        )


def test_frame_zero_adapter_rejects_too_few_native_identities_without_mutation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    lock_path, staged, _, _ = _write_staged_case(tmp_path)
    original_geometry = (staged / failure.FRAME_ZERO_ARCHIVE_FILENAME).read_bytes()
    original_manifest = (staged / failure.FRAME_ZERO_MANIFEST_FILENAME).read_bytes()
    seeded = np.zeros((16, 3), dtype=np.float32)
    fake_stage = SimpleNamespace(
        CROP_HALF_EXTENT_M=0.5,
        SEED_POINT_COUNT=5000,
        seed_points_from_splat=lambda *_args, **_kwargs: (
            seeded,
            seeded.copy(),
        ),
    )
    monkeypatch.setattr(
        failure,
        "validate_deform360_execution_repository",
        lambda _path: {},
    )
    monkeypatch.setattr(
        failure,
        "_import_pinned_pcd_stage",
        lambda _path: fake_stage,
    )
    repository = tmp_path / "deform360"
    repository.mkdir()

    with pytest.raises(ValueError, match="invalid shape or support"):
        failure.adapt_frame_zero_original_splat_identity_persistence(
            lock_path,
            H2,
            staged,
            repository,
            expected_h1=H1,
        )

    assert (
        staged / failure.FRAME_ZERO_ARCHIVE_FILENAME
    ).read_bytes() == original_geometry
    assert (
        staged / failure.FRAME_ZERO_MANIFEST_FILENAME
    ).read_bytes() == original_manifest


def test_identity_policy_hook_delegates_and_rejects_marker_tamper() -> None:
    assert (
        failure.confirmation_frame_zero_physical_policy(
            {"physical_policy": "automatic_twin"},
            original_policy=lambda _value: "automatic_twin",
        )
        == "automatic_twin"
    )
    malformed = {
        measurement.IDENTITY_PERSISTENCE_ADAPTER_KEY: {
            "artifact_kind": measurement.IDENTITY_PERSISTENCE_ADAPTER_KIND,
        },
        "physical_policy": "persistence_only",
    }
    with pytest.raises(ValueError, match="marker is inconsistent"):
        failure.confirmation_frame_zero_physical_policy(
            malformed,
            original_policy=lambda _value: "automatic_twin",
        )


def test_materialize_retained_failure_cli_dispatches_every_bound_path(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    observed: dict[str, Any] = {}

    def materialize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"case_id": "case-01", "failure_code": "resource_exhaustion"}

    monkeypatch.setattr(
        confirmation_cli,
        "materialize_and_seal_retained_confirmation_failure",
        materialize,
    )
    monkeypatch.setattr(
        confirmation_cli,
        "validate_confirmation_h2_production_entrypoint",
        lambda *args, **kwargs: observed.update(
            {"provenance": {"args": args, "kwargs": kwargs}}
        ),
    )
    confirmation_cli.main(
        [
            "materialize-retained-failure",
            "--adapter-repo",
            "/adapter",
            "--external-execution-repo",
            "/external",
            "--lock",
            "/lock.json",
            "--h2-commit",
            H2,
            "--expected-h1",
            H1,
            "--case-id",
            "case-01",
            "--staged-case-dir",
            "/staged/case-01",
            "--processed-episode-dir",
            "/staged/case-01/processed",
            "--source-custody-seal",
            "/custody/case-01.json",
            "--physical-work-dir",
            "/work/case-01",
            "--backbone-dir",
            "/backbone/case-01",
            "--measurement-output-dir",
            "/measurement/case-01",
            "--case-output-dir",
            "/cases/case-01",
            "--failure-code",
            "resource_exhaustion",
        ],
        source_bootstrap_file="/adapter/scripts/remote/"
        "run_deform360_adaptive_confirmation_cli.py",
    )

    assert observed == {
        "provenance": {
            "args": ("/adapter", "/lock.json", H2),
            "kwargs": {
                "expected_h1": H1,
                "entrypoint_file": confirmation_cli.__file__,
                "entrypoint_repository_path": (
                    confirmation_cli.ENTRYPOINT_REPOSITORY_PATH
                ),
                "source_bootstrap_file": (
                    "/adapter/scripts/remote/run_deform360_adaptive_confirmation_cli.py"
                ),
                "source_bootstrap_repository_path": (
                    confirmation_cli.SOURCE_BOOTSTRAP_REPOSITORY_PATH
                ),
            },
        },
        "args": (
            "/adapter",
            "/external",
            "/lock.json",
            H2,
            "case-01",
            "/staged/case-01",
            "/staged/case-01/processed",
            "/custody/case-01.json",
            "/work/case-01",
            "/backbone/case-01",
            "/measurement/case-01",
            "/cases/case-01",
            "resource_exhaustion",
        ),
        "kwargs": {"expected_h1": H1},
    }
    assert json.loads(capsys.readouterr().out) == {
        "case_id": "case-01",
        "failure_code": "resource_exhaustion",
    }


def test_every_confirmation_subcommand_requires_h2_repository_binding() -> None:
    parser = confirmation_cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, confirmation_cli.argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "measurement",
        "source-custody",
        "materialize-retained-failure",
        "predict",
        "retain-failure",
        "validate-case",
        "barrier",
        "validate-barrier",
        "compatibility",
        "validate-compatibility",
        "evaluate",
    }
    for name, command in subparsers.choices.items():
        required = {
            action.dest
            for action in command._actions
            if getattr(action, "required", False)
        }
        assert {"adapter_repo", "lock", "h2_commit", "expected_h1"} <= required, name


def test_confirmation_cli_rejects_forged_scoring_loader_before_evaluation(
    monkeypatch: Any,
) -> None:
    def forged_loader(*_args: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(
        confirmation_cli,
        "validate_confirmation_h2_production_entrypoint",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        confirmation_cli,
        "_case_dirs",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        confirmation_cli,
        "build_confirmation_case_target_loader",
        lambda *_args, **_kwargs: forged_loader,
    )
    monkeypatch.setattr(
        confirmation_cli,
        "evaluate_adaptive_covariance_confirmation",
        lambda *_args, **_kwargs: pytest.fail(
            "production evaluator must not receive a forged loader"
        ),
    )

    with pytest.raises(
        ValueError,
        match="not issued by the frozen scoring factory",
    ):
        confirmation_cli.main(
            [
                "evaluate",
                "--adapter-repo",
                "/adapter",
                "--lock",
                "/lock.json",
                "--h2-commit",
                H2,
                "--expected-h1",
                H1,
                "--barrier",
                "/barrier.json",
                "--case-root",
                "/cases",
                "--measurement-root",
                "/measurements",
                "--compatibility-root",
                "/compatibility",
                "--authorized-future-root",
                "/future",
                "--authorized-outcome-root",
                "/outcome",
                "--output",
                "/result.json",
            ],
            source_bootstrap_file="/adapter/scripts/remote/"
            "run_deform360_adaptive_confirmation_cli.py",
        )


def _archive_record(
    path: Path,
    roles: tuple[str, ...],
    relative_path: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {
            role: {
                "dtype": np.asarray(stored[role]).dtype.str,
                "shape": list(np.asarray(stored[role]).shape),
                "array_sha256": seal.array_sha256(np.asarray(stored[role])),
            }
            for role in roles
        }
    return {
        "relative_path": relative_path,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "arrays": arrays,
    }


def _write_retained_input_package(
    tmp_path: Path,
) -> tuple[
    Path,
    str,
    Path,
    dict[int, Path],
    dict[int, Path],
    np.ndarray,
]:
    lock_path, staged, lock, case_id = _write_staged_case(tmp_path)
    frame_manifest_path = staged / failure.FRAME_ZERO_MANIFEST_FILENAME
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    frame_manifest["material_point_source"] = "original-splat"
    frame_manifest["result_sha256"] = _self_hash(frame_manifest, "result_sha256")
    frame_manifest_path.write_text(
        json.dumps(frame_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with np.load(
        staged / failure.FRAME_ZERO_ARCHIVE_FILENAME,
        allow_pickle=False,
    ) as stored:
        frame_zero = np.asarray(stored["points_m"]).copy()
    point_count = len(frame_zero)
    persistence = np.repeat(frame_zero[None], 76, axis=0)
    physical = tmp_path / "physical.npz"
    physical_arrays = {
        "action_support": np.zeros(point_count, dtype=np.float32),
        "driven_readout_m": persistence.copy(),
        "frame_zero_points_m": frame_zero.copy(),
        "persistence_m": persistence.copy(),
        "prediction_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
    }
    np.savez_compressed(physical, **physical_arrays)
    external_hashes = {
        role: measurement._external_array_sha256(physical_arrays[role])
        for role in measurement.EXTERNAL_PHYSICAL_ARRAY_ROLES
    }
    cameras = {
        4: tuple(f"camera-{index:02d}" for index in range(4)),
        8: tuple(f"camera-{index:02d}" for index in range(8)),
    }
    centers = np.arange(16, dtype=np.int64)
    measurements: dict[int, Path] = {}
    uncertainties: dict[int, Path] = {}
    outputs: dict[str, Any] = {}
    for budget in (4, 8):
        root = tmp_path / f"budget-{budget}"
        root.mkdir()
        observed = np.full_like(persistence, np.nan)
        validity = np.zeros(persistence.shape[:2], dtype=bool)
        covariance = np.full(
            (*persistence.shape[:2], 3, 3),
            np.nan,
            dtype=np.float32,
        )
        covariance_validity = np.zeros(persistence.shape[:2], dtype=bool)
        measurement_path = root / measurement.MEASUREMENT_ARCHIVE_FILENAME
        uncertainty_path = root / measurement.UNCERTAINTY_ARCHIVE_FILENAME
        np.savez_compressed(
            measurement_path,
            measurement_m=observed,
            measurement_validity=validity,
            center_ids=centers,
            selected_cameras=np.asarray(cameras[budget]),
            update_frames=np.asarray((19, 38, 57), dtype=np.int64),
        )
        np.savez_compressed(
            uncertainty_path,
            measurement_covariance_m2=covariance,
            measurement_covariance_valid=covariance_validity,
        )
        measurements[budget] = measurement_path
        uncertainties[budget] = uncertainty_path
        outputs[str(budget)] = {
            "measurement_archive": _archive_record(
                measurement_path,
                measurement.MEASUREMENT_ARRAY_ROLES,
                f"budget-{budget}/{measurement.MEASUREMENT_ARCHIVE_FILENAME}",
            ),
            "uncertainty_archive": _archive_record(
                uncertainty_path,
                measurement.UNCERTAINTY_ARRAY_ROLES,
                f"budget-{budget}/{measurement.UNCERTAINTY_ARCHIVE_FILENAME}",
            ),
        }
    failure_code = "resource_exhaustion"
    empty_covariance = np.full(
        (76, point_count, 3, 3),
        np.nan,
        dtype=np.float32,
    )
    empty_covariance_validity = np.zeros((76, point_count), dtype=bool)
    updates = []
    prefix_hashes = {camera: {} for camera in cameras[8]}
    for frame in (19, 38, 57):
        trackers = []
        for index, camera in enumerate(cameras[8]):
            digest = hashlib.sha256(f"{frame}:{camera}".encode()).hexdigest()
            prefix_hashes[camera][str(frame)] = digest
            trackers.append(
                {
                    "prefix_frame_range_half_open": [0, frame + 1],
                    "maximum_video_frame_read": frame,
                    "decoded_frame_count": frame + 1,
                    "decoded_rgb_prefix_sha256": digest,
                    "original_image_shape": [480, 640],
                    "camera": camera,
                    "query_ids": centers.tolist(),
                    "execution_role": (
                        "adaptive_first_four"
                        if index < 4
                        else "adaptive_eight_escalation"
                    ),
                    "execution_index_within_update": index,
                    "four_view_decision_already_materialized": index >= 4,
                    "camera_stream_attempted": True,
                    "tracker_inference_executed": False,
                    "dynamic_observation_available": False,
                    "failure_code": failure_code,
                }
            )
        current_reliability = {
            **prediction.normalized_covariance_dispersion(
                empty_covariance,
                empty_covariance_validity,
                centers,
                frame,
                frame_zero,
                quantile=(FROZEN_ADAPTIVE_COVARIANCE_CONFIG.covariance_quantile),
            ),
            "reliable": False,
        }
        updates.append(
            {
                "frame": frame,
                "four_view_decision_materialized_before_shadow_extra_four": True,
                "four_view_reliable_before_shadow": False,
                "offline_shadow_extra_four_tracked": False,
                "adaptive_route": "physical_prior_fallback",
                "adaptive_charged_camera_streams": 8,
                "budget_reliability": {
                    "4": copy.deepcopy(current_reliability),
                    "8": copy.deepcopy(current_reliability),
                },
                "tracker": trackers,
                "centers": {
                    str(budget): [
                        {
                            "center_id": int(center),
                            "measurement_available": False,
                            "covariance_valid": False,
                            "decision": (
                                "retained_technical_failure_measurement_unavailable"
                            ),
                            "failure_code": failure_code,
                        }
                        for center in centers
                    ]
                    for budget in (4, 8)
                },
            }
        )
    selected_inputs = {
        camera: {
            "video": {
                "path": str(staged / "prefix" / camera / "undistorted.mp4"),
                "decoded_prefix_sha256_by_update": prefix_hashes[camera],
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_mask": {
                "path": str(staged / "prefix" / camera / "mask_refined.h5"),
                "frame_zero_array_sha256": "1" * 64,
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_depth": {
                "path": str(staged / "prefix" / camera / "rendered_depth.h5"),
                "frame_zero_array_sha256": "2" * 64,
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
        }
        for camera in cameras[8]
    }
    prefix_manifest = staged / failure.PREDICTION_PREFIX_MANIFEST_FILENAME
    frame_manifest = staged / failure.FRAME_ZERO_MANIFEST_FILENAME
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    for budget in (4, 8):
        shutil_source = tmp_path / f"budget-{budget}"
        shutil_destination = nested_root / f"budget-{budget}"
        shutil_destination.mkdir()
        for source in shutil_source.iterdir():
            (shutil_destination / source.name).write_bytes(source.read_bytes())
        measurements[budget] = (
            shutil_destination / measurement.MEASUREMENT_ARCHIVE_FILENAME
        )
        uncertainties[budget] = (
            shutil_destination / measurement.UNCERTAINTY_ARCHIVE_FILENAME
        )
        outputs[str(budget)] = {
            "measurement_archive": _archive_record(
                measurements[budget],
                measurement.MEASUREMENT_ARRAY_ROLES,
                f"budget-{budget}/{measurement.MEASUREMENT_ARCHIVE_FILENAME}",
            ),
            "uncertainty_archive": _archive_record(
                uncertainties[budget],
                measurement.UNCERTAINTY_ARRAY_ROLES,
                f"budget-{budget}/{measurement.UNCERTAINTY_ARCHIVE_FILENAME}",
            ),
        }
    manifest: dict[str, Any] = {
        "schema_version": measurement.SCHEMA_VERSION,
        "artifact_kind": measurement.ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_identity": measurement._case_identity(lock, case_id),
        "lock_binding": {
            "implementation_commit_h1": H1,
            "cohort_lock_commit_h2": H2,
            "cohort_lock_artifact_sha256": lock["artifact_sha256"],
            "cohort_lock_file_sha256": _file_sha256(lock_path),
        },
        "config": {
            "observation": asdict(RawCameraObservationConfig(selected_camera_count=8)),
            "uncertainty": asdict(RawCameraUncertaintyConfig()),
            "adaptive_routing": asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG),
        },
        "plan": {
            "candidate_ids": centers.tolist(),
            "center_ids": centers.tolist(),
            "camera_activation_order": list(cameras[8]),
            "selected_cameras_by_budget": {
                str(budget): list(cameras[budget]) for budget in (4, 8)
            },
            "selection_score": {
                "4": [16, 16, 64, 1.0],
                "8": [16, 16, 128, 1.0],
            },
        },
        "inputs": {
            "physical_backbone": {
                "external_backbone_seal_file_sha256": "3" * 64,
                "external_backbone_seal_result_sha256": "4" * 64,
                "external_physical_manifest_file_sha256": "5" * 64,
                "external_physical_manifest_result_sha256": "6" * 64,
                "physical_archive_file_sha256": _file_sha256(physical),
                "physical_archive_array_sha256": external_hashes,
            },
            "physical_archive": {
                "sha256": _file_sha256(physical),
                "frame_zero_array_sha256": raw_array_sha256(frame_zero),
            },
            "intrinsics_sha256": "7" * 64,
            "extrinsics_sha256": "8" * 64,
            "selected_camera_prefixes_and_frame_zero": selected_inputs,
            "source_stage_lineage": {
                "prediction_prefix_manifest": {
                    "path": str(prefix_manifest),
                    "file_sha256": _file_sha256(prefix_manifest),
                    "result_sha256": json.loads(prefix_manifest.read_text())[
                        "result_sha256"
                    ],
                },
                "frame_zero_manifest": {
                    "path": str(frame_manifest),
                    "file_sha256": _file_sha256(frame_manifest),
                    "result_sha256": json.loads(frame_manifest.read_text())[
                        "result_sha256"
                    ],
                },
                "source_preparation_manifest_file_sha256": "9" * 64,
                "source_custody_seal": {
                    "path": str(staged / "source-custody.json"),
                    "file_sha256": "a" * 64,
                    "artifact_sha256": "b" * 64,
                },
            },
            "retained_failure_source": {
                "failure_code": failure_code,
                "prediction_prefix_manifest": {
                    "path": str(prefix_manifest),
                    "file_sha256": _file_sha256(prefix_manifest),
                    "result_sha256": json.loads(prefix_manifest.read_text())[
                        "result_sha256"
                    ],
                },
                "frame_zero_manifest": {
                    "path": str(frame_manifest),
                    "file_sha256": _file_sha256(frame_manifest),
                    "result_sha256": json.loads(frame_manifest.read_text())[
                        "result_sha256"
                    ],
                },
                "processed_prefix_episode": {
                    "path": str(staged / failure.PROCESSED_PREFIX_RELATIVE_PATH),
                    "intrinsics_file_sha256": "7" * 64,
                    "extrinsics_file_sha256": "8" * 64,
                },
                "dynamic_point_observations_available": False,
            },
        },
        "tracker": {
            "name": "AllTracker",
            "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
            "source_tree": ALLTRACKER_SOURCE_TREE,
            "runtime_source_sha256": ALLTRACKER_RUNTIME_SOURCE_SHA256,
            "checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
            "device": "not-executed",
            "execution_status": "retained_technical_failure",
            "failure_code": failure_code,
            "inference_executed": False,
        },
        "updates": updates,
        "outputs": outputs,
        "camera_accounting": dict(measurement.RETAINED_FAILURE_CAMERA_ACCOUNTING),
        "information_boundary": copy.deepcopy(
            prediction._MEASUREMENT_INFORMATION_BOUNDARY
        ),
    }
    manifest["artifact_sha256"] = prediction._manifest_artifact_sha256(manifest)
    manifest_path = nested_root / measurement.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        lock_path,
        case_id,
        physical,
        measurements,
        uncertainties,
        persistence,
    )


def raw_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_retained_input_manifest_seals_six_persistence_roles_and_rejects_observation(
    tmp_path: Path,
) -> None:
    (
        lock_path,
        case_id,
        physical,
        measurements,
        uncertainties,
        persistence,
    ) = _write_retained_input_package(tmp_path)
    manifest = tmp_path / "nested" / measurement.MANIFEST_FILENAME
    output = tmp_path / "cases" / case_id

    prediction.seal_retained_confirmation_failure(
        lock_path,
        H2,
        case_id,
        output,
        physical,
        measurements,
        uncertainties,
        "resource_exhaustion",
        measurement_manifest=manifest,
        expected_h1=H1,
    )

    with np.load(output / seal.ARRAY_ARCHIVE_FILENAME, allow_pickle=False) as stored:
        for role in seal.ARRAY_ROLES:
            np.testing.assert_array_equal(stored[role], persistence)
    diagnostic = json.loads(
        (output / seal.DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    assert diagnostic["technical_disposition"]["fallback_label"] == "persistence_only"
    assert all(
        update["route"] == "physical_prior_fallback"
        and update["selected_backbone"] == "persistence"
        and update["tracked_camera_count"] == 8
        and update["state_updated"] is False
        for update in diagnostic["covariance_routing"]["updates"]
    )

    with np.load(measurements[8], allow_pickle=False) as stored:
        changed = {name: np.asarray(stored[name]).copy() for name in stored.files}
    changed["measurement_m"][19, 0] = np.array([0.0, 0.0, 2.0])
    changed["measurement_validity"][19, 0] = True
    np.savez_compressed(measurements[8], **changed)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["outputs"]["8"]["measurement_archive"] = _archive_record(
        measurements[8],
        measurement.MEASUREMENT_ARRAY_ROLES,
        f"budget-8/{measurement.MEASUREMENT_ARCHIVE_FILENAME}",
    )
    payload["artifact_sha256"] = prediction._manifest_artifact_sha256(payload)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="available dynamic observation"):
        prediction.seal_retained_confirmation_failure(
            lock_path,
            H2,
            case_id,
            tmp_path / "tampered" / case_id,
            physical,
            measurements,
            uncertainties,
            "resource_exhaustion",
            measurement_manifest=manifest,
            expected_h1=H1,
        )
