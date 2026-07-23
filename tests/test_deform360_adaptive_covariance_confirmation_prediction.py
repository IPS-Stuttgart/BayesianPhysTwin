from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_measurement as measurement
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_prediction as prediction
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal as seal
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
    normalized_covariance_dispersion,
    predict_adaptive_covariance_selected_backbone_rbf,
)
from bayesian_phystwin.deform360_held_online_prefix import (
    FRAME_COUNT,
    HELD_RBF_CONFIG,
    UPDATE_FRAMES,
    predict_support_gated_selected_backbone_rbf,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _external_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _case_identity(lock_payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        {
            "case_id": case_id,
            "stratum": stratum,
            "object_id": record["object_id"],
            "episode_id": episode["episode_id"],
        }
        for stratum, records in lock_payload["cohort"].items()
        for record in records
        for episode in record["episodes"]
        if episode["case_id"] == case_id
    ]
    assert len(matches) == 1
    return matches[0]


def _archive_record(
    path: Path, roles: tuple[str, ...], relative: str
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
        "relative_path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "arrays": arrays,
    }


def _rewrite_manifest(
    path: Path,
    mutate: Any,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["artifact_sha256"] = prediction._manifest_artifact_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_lock(tmp_path: Path) -> tuple[Path, list[str]]:
    path = tmp_path / "lock" / "confirmation.json"
    payload = lock.write_confirmation_cohort_lock(path, H1)
    return path, list(payload["selected_case_ids"])


def _fixture_arrays(
    *,
    persistence_equal: bool = False,
) -> dict[str, Any]:
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.0, 0.04, 5),
            np.linspace(0.0, 0.03, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    frame_zero = np.column_stack((grid, np.zeros(len(grid)))).astype(np.float32)
    physical = np.repeat(frame_zero[None], FRAME_COUNT, axis=0)
    if not persistence_equal:
        physical[:, :, 0] += np.arange(FRAME_COUNT, dtype=np.float32)[
            :, None
        ] * np.float32(2e-4)
    persistence = np.repeat(frame_zero[None], FRAME_COUNT, axis=0)
    driven = physical.copy()
    zero_action = persistence.copy()
    action_support = np.ones(len(frame_zero), dtype=np.float32)
    centers = np.arange(16, dtype=np.int64)
    measurements: dict[int, np.ndarray] = {}
    measurement_validity: dict[int, np.ndarray] = {}
    covariance: dict[int, np.ndarray] = {}
    covariance_validity: dict[int, np.ndarray] = {}
    cameras = {
        4: tuple(f"camera-{index:02d}" for index in range(4)),
        8: tuple(f"camera-{index:02d}" for index in range(8)),
    }
    for budget in prediction.CAMERA_BUDGETS:
        observed = np.full_like(physical, np.nan)
        valid = np.zeros(physical.shape[:2], dtype=bool)
        cov = np.full((*physical.shape[:2], 3, 3), np.nan, dtype=np.float64)
        cov_valid = np.zeros(physical.shape[:2], dtype=bool)
        for update in UPDATE_FRAMES:
            observed[update, centers] = physical[update, centers]
            observed[update, centers, 2] += np.float32(0.001)
            valid[update, centers] = True
            cov[update, centers] = np.eye(3) * 1e-12
            cov_valid[update, centers] = True
        measurements[budget] = observed
        measurement_validity[budget] = valid
        covariance[budget] = cov
        covariance_validity[budget] = cov_valid
    return {
        "physical": physical,
        "persistence": persistence,
        "frame_zero": frame_zero,
        "driven": driven,
        "zero_action": zero_action,
        "action_support": action_support,
        "centers": centers,
        "cameras": cameras,
        "measurements": measurements,
        "measurement_validity": measurement_validity,
        "covariance": covariance,
        "covariance_validity": covariance_validity,
    }


def _write_inputs(
    root: Path,
    lock_path: Path,
    case_id: str,
    *,
    persistence_equal: bool = False,
) -> tuple[Path, Path, dict[int, Path], dict[int, Path], dict[str, Any]]:
    arrays = _fixture_arrays(persistence_equal=persistence_equal)
    root.mkdir(parents=True)
    physical_path = root / "physical.npz"
    np.savez_compressed(
        physical_path,
        action_support=arrays["action_support"],
        driven_readout_m=arrays["driven"],
        frame_zero_points_m=arrays["frame_zero"],
        persistence_m=arrays["persistence"],
        prediction_m=arrays["physical"],
        zero_action_readout_m=arrays["zero_action"],
    )
    measurements: dict[int, Path] = {}
    uncertainties: dict[int, Path] = {}
    for budget in prediction.CAMERA_BUDGETS:
        budget_root = root / f"budget-{budget}"
        budget_root.mkdir()
        measurement_path = budget_root / measurement.MEASUREMENT_ARCHIVE_FILENAME
        np.savez_compressed(
            measurement_path,
            measurement_m=arrays["measurements"][budget],
            measurement_validity=arrays["measurement_validity"][budget],
            center_ids=arrays["centers"],
            selected_cameras=np.asarray(arrays["cameras"][budget]),
            update_frames=np.asarray(UPDATE_FRAMES, dtype=np.int64),
        )
        uncertainty_path = budget_root / measurement.UNCERTAINTY_ARCHIVE_FILENAME
        np.savez_compressed(
            uncertainty_path,
            measurement_covariance_m2=arrays["covariance"][budget],
            measurement_covariance_valid=arrays["covariance_validity"][budget],
        )
        measurements[budget] = measurement_path
        uncertainties[budget] = uncertainty_path

    updates: list[dict[str, Any]] = []
    cameras8 = arrays["cameras"][8]
    prefix_hashes: dict[str, dict[str, str]] = {camera: {} for camera in cameras8}
    for frame in UPDATE_FRAMES:
        reliability: dict[str, Any] = {}
        for budget in prediction.CAMERA_BUDGETS:
            result = normalized_covariance_dispersion(
                arrays["covariance"][budget],
                arrays["covariance_validity"][budget],
                arrays["centers"],
                frame,
                arrays["frame_zero"],
                quantile=FROZEN_ADAPTIVE_COVARIANCE_CONFIG.covariance_quantile,
            )
            normalized = result["normalized_covariance_dispersion"]
            reliable = (
                result["valid_covariance_center_count"]
                >= FROZEN_ADAPTIVE_COVARIANCE_CONFIG.minimum_valid_covariance_centers
                and normalized is not None
                and normalized
                <= FROZEN_ADAPTIVE_COVARIANCE_CONFIG.maximum_normalized_covariance_dispersion
            )
            reliability[str(budget)] = {
                **result,
                "reliable": bool(reliable),
            }
        four_reliable = reliability["4"]["reliable"]
        eight_reliable = reliability["8"]["reliable"]
        route = (
            "4_view_rbf"
            if four_reliable
            else ("8_view_rbf" if eight_reliable else "physical_prior_fallback")
        )
        trackers = []
        for camera_index, camera in enumerate(cameras8):
            prefix_sha = hashlib.sha256(
                f"{case_id}:{frame}:{camera}".encode()
            ).hexdigest()
            prefix_hashes[camera][str(frame)] = prefix_sha
            trackers.append(
                {
                    "maximum_video_frame_read": frame,
                    "decoded_rgb_prefix_sha256": prefix_sha,
                    "camera": camera,
                    "query_ids": arrays["centers"].tolist(),
                    "execution_role": (
                        "adaptive_first_four"
                        if camera_index < 4
                        else (
                            "fixed_eight_shadow_after_four_decision"
                            if four_reliable
                            else "adaptive_eight_escalation"
                        )
                    ),
                    "execution_index_within_update": camera_index,
                    "four_view_decision_already_materialized": (camera_index >= 4),
                }
            )
        updates.append(
            {
                "frame": frame,
                "four_view_decision_materialized_before_shadow_extra_four": True,
                "four_view_reliable_before_shadow": four_reliable,
                "offline_shadow_extra_four_tracked": True,
                "adaptive_route": route,
                "adaptive_charged_camera_streams": (4 if four_reliable else 8),
                "budget_reliability": reliability,
                "tracker": trackers,
                "centers": {
                    str(budget): [
                        {"center_id": int(center)} for center in arrays["centers"]
                    ]
                    for budget in prediction.CAMERA_BUDGETS
                },
            }
        )
    selected_inputs = {
        camera: {
            "video": {
                "path": str(root / camera / "undistorted.mp4"),
                "decoded_prefix_sha256_by_update": prefix_hashes[camera],
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_mask": {
                "path": str(root / camera / "mask_refined.h5"),
                "frame_zero_array_sha256": hashlib.sha256(
                    f"{case_id}:{camera}:mask".encode()
                ).hexdigest(),
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
            "frame_zero_depth": {
                "path": str(root / camera / "rendered_depth.h5"),
                "frame_zero_array_sha256": hashlib.sha256(
                    f"{case_id}:{camera}:depth".encode()
                ).hexdigest(),
                "only_index_read": 0,
                "whole_file_hashed_or_read": False,
            },
        }
        for camera in cameras8
    }
    with np.load(physical_path, allow_pickle=False) as stored:
        external_hashes = {
            role: _external_array_sha256(np.asarray(stored[role]))
            for role in prediction.PHYSICAL_ARRAY_ROLES
        }
    lock_payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=H1,
    )
    manifest_payload: dict[str, Any] = {
        "schema_version": measurement.SCHEMA_VERSION,
        "artifact_kind": measurement.ARTIFACT_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "case_identity": _case_identity(lock_payload, case_id),
        "lock_binding": {
            "implementation_commit_h1": H1,
            "cohort_lock_commit_h2": H2,
            "cohort_lock_artifact_sha256": lock_payload["artifact_sha256"],
            "cohort_lock_file_sha256": _sha256(lock_path),
        },
        "config": {
            "observation": asdict(RawCameraObservationConfig(selected_camera_count=8)),
            "uncertainty": asdict(RawCameraUncertaintyConfig()),
            "adaptive_routing": asdict(FROZEN_ADAPTIVE_COVARIANCE_CONFIG),
        },
        "plan": {
            "candidate_ids": arrays["centers"].tolist(),
            "center_ids": arrays["centers"].tolist(),
            "camera_activation_order": list(cameras8),
            "selected_cameras_by_budget": {
                str(budget): list(arrays["cameras"][budget])
                for budget in prediction.CAMERA_BUDGETS
            },
            "selection_score": {
                "4": [16, 16, 64, 1.0],
                "8": [16, 16, 128, 1.0],
            },
        },
        "inputs": {
            "physical_backbone": {
                "external_backbone_seal_file_sha256": "1" * 64,
                "external_backbone_seal_result_sha256": "2" * 64,
                "external_physical_manifest_file_sha256": "3" * 64,
                "external_physical_manifest_result_sha256": "4" * 64,
                "physical_archive_file_sha256": _sha256(physical_path),
                "physical_archive_array_sha256": external_hashes,
            },
            "physical_archive": {
                "sha256": _sha256(physical_path),
                "frame_zero_array_sha256": _measurement_array_sha256(
                    arrays["frame_zero"]
                ),
            },
            "intrinsics_sha256": "5" * 64,
            "extrinsics_sha256": "6" * 64,
            "selected_camera_prefixes_and_frame_zero": selected_inputs,
            "source_stage_lineage": {
                "prediction_prefix_manifest": {
                    "path": str(root / "prediction_prefix_manifest.json"),
                    "file_sha256": "9" * 64,
                    "result_sha256": "a" * 64,
                },
                "frame_zero_manifest": {
                    "path": str(root / "frame_zero_reconstruction_manifest.json"),
                    "file_sha256": "b" * 64,
                    "result_sha256": "c" * 64,
                },
                "source_preparation_manifest_file_sha256": "d" * 64,
                "source_custody_seal": {
                    "path": str(root / "source-custody.json"),
                    "file_sha256": "e" * 64,
                    "artifact_sha256": "f" * 64,
                },
            },
        },
        "tracker": {
            "name": "AllTracker",
            "molmomotion_revision": ALLTRACKER_MOLMOMOTION_REVISION,
            "source_tree": ALLTRACKER_SOURCE_TREE,
            "runtime_source_sha256": ALLTRACKER_RUNTIME_SOURCE_SHA256,
            "checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
            "device": "synthetic-cpu",
        },
        "updates": updates,
        "outputs": {
            str(budget): {
                "measurement_archive": _archive_record(
                    measurements[budget],
                    prediction.MEASUREMENT_ARRAY_ROLES,
                    f"budget-{budget}/{measurement.MEASUREMENT_ARCHIVE_FILENAME}",
                ),
                "uncertainty_archive": _archive_record(
                    uncertainties[budget],
                    prediction.UNCERTAINTY_ARRAY_ROLES,
                    (f"budget-{budget}/{measurement.UNCERTAINTY_ARCHIVE_FILENAME}"),
                ),
            }
            for budget in prediction.CAMERA_BUDGETS
        },
        "camera_accounting": copy.deepcopy(prediction._CAMERA_ACCOUNTING),
        "information_boundary": copy.deepcopy(
            prediction._MEASUREMENT_INFORMATION_BOUNDARY
        ),
    }
    manifest_payload["artifact_sha256"] = prediction._manifest_artifact_sha256(
        manifest_payload
    )
    manifest_path = root / measurement.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return physical_path, manifest_path, measurements, uncertainties, arrays


def _assemble(
    lock_path: Path,
    case_id: str,
    output: Path,
    physical: Path,
    manifest: Path,
    measurements: dict[int, Path],
    uncertainties: dict[int, Path],
) -> dict[str, Any]:
    return prediction.assemble_and_seal_confirmation_prediction(
        lock_path,
        H2,
        case_id,
        output,
        physical,
        measurements,
        uncertainties,
        measurement_manifest=manifest,
        expected_h1=H1,
    )


def _refresh_manifest_outputs(
    manifest_path: Path,
    measurements: dict[int, Path],
    uncertainties: dict[int, Path],
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["outputs"] = {
            str(budget): {
                "measurement_archive": _archive_record(
                    measurements[budget],
                    prediction.MEASUREMENT_ARRAY_ROLES,
                    f"budget-{budget}/{measurement.MEASUREMENT_ARCHIVE_FILENAME}",
                ),
                "uncertainty_archive": _archive_record(
                    uncertainties[budget],
                    prediction.UNCERTAINTY_ARRAY_ROLES,
                    (f"budget-{budget}/{measurement.UNCERTAINTY_ARCHIVE_FILENAME}"),
                ),
            }
            for budget in prediction.CAMERA_BUDGETS
        }

    _rewrite_manifest(manifest_path, mutate)


def test_success_computes_exact_fixed_and_adaptive_arms_and_hashes_inputs(
    tmp_path: Path,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical_path, manifest_path, measurements, uncertainties, inputs = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )
    output = tmp_path / "cases" / cases[0]

    record = _assemble(
        lock_path,
        cases[0],
        output,
        physical_path,
        manifest_path,
        measurements,
        uncertainties,
    )

    with np.load(output / seal.ARRAY_ARCHIVE_FILENAME, allow_pickle=False) as stored:
        sealed = {key: np.asarray(stored[key]) for key in stored.files}
    fixed4, _, _ = predict_support_gated_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["measurements"][4],
        inputs["measurement_validity"][4],
        center_ids=inputs["centers"],
        rbf_config=HELD_RBF_CONFIG,
    )
    fixed8, _, _ = predict_support_gated_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["measurements"][8],
        inputs["measurement_validity"][8],
        center_ids=inputs["centers"],
        rbf_config=HELD_RBF_CONFIG,
    )
    adaptive, selected_raw, _ = predict_adaptive_covariance_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["cameras"],
        inputs["measurements"],
        inputs["measurement_validity"],
        inputs["covariance"],
        inputs["covariance_validity"],
        center_ids=inputs["centers"],
        config=FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
        rbf_config=HELD_RBF_CONFIG,
    )
    np.testing.assert_array_equal(sealed["physical_prior_m"], inputs["physical"])
    np.testing.assert_array_equal(sealed["persistence_m"], inputs["persistence"])
    np.testing.assert_array_equal(sealed["fixed_4_rbf_prediction_m"], fixed4)
    np.testing.assert_array_equal(sealed["fixed_8_rbf_prediction_m"], fixed8)
    np.testing.assert_array_equal(sealed["adaptive_prediction_m"], adaptive)
    np.testing.assert_array_equal(
        sealed["selected_raw_prediction_m"],
        selected_raw,
    )
    diagnostic = json.loads((output / seal.DIAGNOSTIC_FILENAME).read_text())
    disposition = diagnostic["technical_disposition"]
    assert disposition["status"] == "prediction_complete"
    hashes = disposition["causal_input_hashes"]
    assert (
        hashes["physical_archive"]["file_sha256"]
        == hashlib.sha256(physical_path.read_bytes()).hexdigest()
    )
    assert set(hashes["measurement_archives"]) == {"4", "8"}
    assert set(hashes["uncertainty_archives"]) == {"4", "8"}
    assert hashes["nested_measurement_manifest"]["file_sha256"] == _sha256(
        manifest_path
    )
    assert "path" not in hashes["physical_archive"]
    assert record == seal.validate_confirmation_case_seal(
        output,
        lock_path,
        H2,
        expected_case_id=cases[0],
        expected_h1=H1,
    )


def test_retained_failure_rejects_successful_measurement_carrier(
    tmp_path: Path,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical_path, manifest_path, measurements, uncertainties, _ = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )

    with pytest.raises(
        ValueError,
        match="exact failed-measurement carrier",
    ):
        prediction.seal_retained_confirmation_failure(
            lock_path,
            H2,
            cases[0],
            tmp_path / "cases" / cases[0],
            physical_path,
            measurements,
            uncertainties,
            "prediction_runtime_failure",
            measurement_manifest=manifest_path,
            expected_h1=H1,
        )


def test_wrong_centers_and_nonnested_camera_plan_fail_closed(
    tmp_path: Path,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical_path, manifest_path, measurements, uncertainties, inputs = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )
    np.savez_compressed(
        measurements[8],
        measurement_m=inputs["measurements"][8],
        measurement_validity=inputs["measurement_validity"][8],
        center_ids=inputs["centers"][::-1],
        selected_cameras=np.asarray(inputs["cameras"][8]),
        update_frames=np.asarray(UPDATE_FRAMES, dtype=np.int64),
    )
    _refresh_manifest_outputs(manifest_path, measurements, uncertainties)
    with pytest.raises(ValueError, match="center IDs changed"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "wrong-centers" / cases[0],
            physical_path,
            manifest_path,
            measurements,
            uncertainties,
        )

    physical_path, manifest_path, measurements, uncertainties, inputs = _write_inputs(
        tmp_path / "nonnested-inputs",
        lock_path,
        cases[0],
    )
    nonnested = tuple(inputs["cameras"][8][4:] + inputs["cameras"][8][:4])
    np.savez_compressed(
        measurements[8],
        measurement_m=inputs["measurements"][8],
        measurement_validity=inputs["measurement_validity"][8],
        center_ids=inputs["centers"],
        selected_cameras=np.asarray(nonnested),
        update_frames=np.asarray(UPDATE_FRAMES, dtype=np.int64),
    )
    _refresh_manifest_outputs(manifest_path, measurements, uncertainties)
    with pytest.raises(ValueError, match="strict ordered prefix"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "nonnested" / cases[0],
            physical_path,
            manifest_path,
            measurements,
            uncertainties,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("case", "binds another case"),
        ("h1", "lock binding changed"),
        ("h2", "lock binding changed"),
        ("physical", "binds another physical archive"),
        ("measurement_file", "measurement_archive manifest binding changed"),
        ("measurement_array", "measurement_archive manifest binding changed"),
        ("boundary", "crossed the target boundary"),
        ("config", "configuration changed"),
        ("tracker_source", "tracker provenance changed"),
        ("source_custody_schema", "source-custody lineage changed"),
    ),
)
def test_manifest_identity_hashes_arrays_and_target_free_boundary_are_exact(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical, manifest, measurements, uncertainties, _ = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )

    def mutate(payload: dict[str, Any]) -> None:
        if tamper == "case":
            payload["case_identity"] = _case_identity(
                lock.load_confirmation_cohort_lock(lock_path),
                cases[1],
            )
        elif tamper == "h1":
            payload["lock_binding"]["implementation_commit_h1"] = "c" * 40
        elif tamper == "h2":
            payload["lock_binding"]["cohort_lock_commit_h2"] = "c" * 40
        elif tamper == "physical":
            payload["inputs"]["physical_archive"]["sha256"] = "9" * 64
        elif tamper == "measurement_file":
            payload["outputs"]["4"]["measurement_archive"]["sha256"] = "9" * 64
        elif tamper == "measurement_array":
            payload["outputs"]["4"]["measurement_archive"]["arrays"]["measurement_m"][
                "array_sha256"
            ] = "9" * 64
        elif tamper == "boundary":
            payload["information_boundary"]["future_geometry_read"] = True
        elif tamper == "config":
            payload["config"]["observation"]["selected_camera_count"] = 4
        elif tamper == "tracker_source":
            payload["tracker"]["runtime_source_sha256"] = "9" * 64
        elif tamper == "source_custody_schema":
            del payload["inputs"]["source_stage_lineage"]["source_custody_seal"][
                "artifact_sha256"
            ]
        else:  # pragma: no cover - guarded by parametrization
            raise AssertionError(tamper)

    _rewrite_manifest(manifest, mutate)
    with pytest.raises(ValueError, match=message):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / f"tamper-{tamper}" / cases[0],
            physical,
            manifest,
            measurements,
            uncertainties,
        )


def test_same_shape_archives_or_manifest_from_another_case_are_rejected(
    tmp_path: Path,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical_a, manifest_a, measurements_a, uncertainties_a, arrays_a = _write_inputs(
        tmp_path / "case-a", lock_path, cases[0]
    )
    physical_b, manifest_b, measurements_b, uncertainties_b, arrays_b = _write_inputs(
        tmp_path / "case-b", lock_path, cases[1]
    )
    assert arrays_a["physical"].shape == arrays_b["physical"].shape
    assert arrays_a["measurements"][4].shape == arrays_b["measurements"][4].shape

    with pytest.raises(
        ValueError,
        match="manifest binding changed|another physical|outside its manifest package",
    ):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "swapped-archives" / cases[0],
            physical_b,
            manifest_a,
            measurements_b,
            uncertainties_b,
        )
    with pytest.raises(ValueError, match="binds another case"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "swapped-manifest" / cases[0],
            physical_a,
            manifest_b,
            measurements_a,
            uncertainties_a,
        )


def test_manifest_self_hash_and_duplicate_json_keys_fail_closed(
    tmp_path: Path,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical, manifest, measurements, uncertainties, _ = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "9" * 64
    manifest.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest checksum changed"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "wrong-self-hash" / cases[0],
            physical,
            manifest,
            measurements,
            uncertainties,
        )

    (
        physical,
        manifest,
        measurements,
        uncertainties,
        _,
    ) = _write_inputs(tmp_path / "duplicate", lock_path, cases[0])
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        original.replace(
            '"schema_version": 1,',
            '"schema_version": 1, "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "duplicate-key" / cases[0],
            physical,
            manifest,
            measurements,
            uncertainties,
        )


def test_archive_hash_mutation_between_load_and_seal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, cases = _write_lock(tmp_path)
    physical_path, manifest_path, measurements, uncertainties, _ = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )
    output = tmp_path / "cases" / cases[0]
    original = prediction.predict_support_gated_selected_backbone_rbf
    call_count = 0

    def mutate_after_snapshot(*args: Any, **kwargs: Any):
        nonlocal call_count
        result = original(*args, **kwargs)
        call_count += 1
        if call_count == 1:
            measurements[4].write_bytes(
                measurements[4].read_bytes() + b"changed-after-load"
            )
        return result

    monkeypatch.setattr(
        prediction,
        "predict_support_gated_selected_backbone_rbf",
        mutate_after_snapshot,
    )
    with pytest.raises(ValueError, match="archive changed after target-free loading"):
        _assemble(
            lock_path,
            cases[0],
            output,
            physical_path,
            manifest_path,
            measurements,
            uncertainties,
        )
    assert not output.exists()


def test_archive_paths_are_regular_distinct_and_signature_has_no_leakage(
    tmp_path: Path,
) -> None:
    for function in (
        prediction.assemble_and_seal_confirmation_prediction,
        prediction.seal_retained_confirmation_failure,
    ):
        parameters = inspect.signature(function).parameters
        assert parameters["measurement_manifest"].default is inspect.Parameter.empty
        assert all(
            "target" not in name and "outcome" not in name and "metric" not in name
            for name in parameters
        )

    lock_path, cases = _write_lock(tmp_path)
    physical_path, manifest_path, measurements, uncertainties, _ = _write_inputs(
        tmp_path / "inputs",
        lock_path,
        cases[0],
    )
    symlink = tmp_path / "physical-link.npz"
    symlink.symlink_to(physical_path)
    with pytest.raises(ValueError, match="symlink"):
        _assemble(
            lock_path,
            cases[0],
            tmp_path / "symlink" / cases[0],
            symlink,
            manifest_path,
            measurements,
            uncertainties,
        )
