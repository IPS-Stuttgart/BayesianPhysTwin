#!/usr/bin/env python3
"""Seal and score the prospective PokeFlex conservative-shrinkage target."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "remote"))

from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _load_mesh,
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _correction_field_variants,
    _load_official_template,
)

from bayesian_phystwin.pokeflex_action_robust_scale import (  # noqa: E402
    ACTION_ROBUST_SCALE_FILE_SHA256,
    load_action_robust_scale_calibration,
    validate_action_robust_scale_calibration,
)
from bayesian_phystwin.pokeflex_action_robust_scale import (  # noqa: E402
    BASE_EFFECTIVE_SCALE as ACTION_ROBUST_BASE_EFFECTIVE_SCALE,
)
from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    ACTION_ROBUST_PROTOCOLS,
    CHECKPOINT_SHA256,
    SELECTED_ARM,
    SOURCE_RESULT_SHA256,
    TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1,
    TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
    TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
    TARGET_PROTOCOL_OFFICIAL13_PUBLIC_V1,
    TARGET_PROTOCOL_OFFICIAL18_V1,
    TARGET_PROTOCOL_V2,
    UPSTREAM_COMMIT,
    action_field_history_is_supported,
    build_prediction_barrier,
    evaluate_target_metrics,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
    prediction_seal_sha256,
    score_one_prediction,
    target_take_ids_for_protocol,
    validate_prediction_barrier,
    validate_prediction_seal,
)
from bayesian_phystwin.pokeflex_fresh12_staging import (  # noqa: E402
    STAGE_MANIFEST_NAME,
    validate_pokeflex_fresh12_stage_manifest,
    validate_staged_file,
)
from bayesian_phystwin.pokeflex_instance_shrinkage import (  # noqa: E402
    BASE_EFFECTIVE_SCALE,
    INSTANCE_SCALE_CALIBRATION_FILE_SHA256,
    load_instance_scale_calibration,
    validate_instance_scale_calibration,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)


def _git_output(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(checkout), *arguments),
        text=True,
    ).strip()


def _git_revision(checkout: Path) -> str:
    return _git_output(checkout, "rev-parse", "HEAD")


def _git_clean(checkout: Path) -> bool:
    return not bool(_git_output(checkout, "status", "--porcelain"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _input_record(path: Path, take_root: Path) -> dict[str, object]:
    return {
        "path_relative_to_take": str(path.resolve().relative_to(take_root.resolve())),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _predict(
    take_root: Path,
    output_dir: Path,
    protocol_path: Path,
    source_result_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    source_stage_manifest: Path | None,
    instance_scale_calibration: Path | None,
    action_robust_scale_calibration: Path | None,
) -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(protocol_path)
    if protocol["protocol_id"] not in {
        TARGET_PROTOCOL_V2,
        TARGET_PROTOCOL_OFFICIAL18_V1,
        TARGET_PROTOCOL_OFFICIAL13_PUBLIC_V1,
        TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
        TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
        TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
        TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1,
    }:
        raise ValueError("new predictions require a robot-history-aware protocol")
    target_take_ids = target_take_ids_for_protocol(protocol)
    take_root = take_root.resolve()
    output_dir = output_dir.resolve()
    if take_root.name not in target_take_ids:
        raise ValueError(f"take is outside the target lock: {take_root.name}")
    source_archive_record = None
    stage = None
    if protocol["protocol_id"] in {
        TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
        TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
        TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
    }:
        if source_stage_manifest is None:
            raise ValueError("fresh public prediction requires its stage manifest")
        stage = validate_pokeflex_fresh12_stage_manifest(
            source_stage_manifest,
            protocol,
            expected_take_id=take_root.name,
        )
        if stage["path"].parent != take_root:
            raise ValueError("fresh stage manifest is outside the take root")
        stage_payload = stage["payload"]
        source_archive_record = {
            "source_archive_name": stage_payload["archive_name"],
            "source_archive_sha256": stage_payload["archive_sha256"],
            "source_stage_manifest_name": Path(source_stage_manifest).name,
            "source_stage_manifest_sha256": stage["stage_manifest_sha256"],
            "source_stage_manifest_file_sha256": file_sha256(
                Path(source_stage_manifest)
            ),
        }
    object_name, _, _ = take_root.name.rpartition("_T")
    correction_multiplier = 1.0
    calibration_record = None
    if protocol["protocol_id"] == TARGET_PROTOCOL_INSTANCE_FRESH12_V2:
        if instance_scale_calibration is None:
            raise ValueError("instance protocol requires its scale calibration")
        if (
            file_sha256(instance_scale_calibration)
            != INSTANCE_SCALE_CALIBRATION_FILE_SHA256
        ):
            raise ValueError("instance scale calibration bytes changed")
        calibration = load_instance_scale_calibration(instance_scale_calibration)
        validation = validate_instance_scale_calibration(calibration)
        correction_multiplier = float(validation["multipliers"][object_name])
        protocol_multiplier = float(
            protocol["method"]["instance_scale_calibration"]["multipliers"][object_name]
        )
        if correction_multiplier != protocol_multiplier:
            raise ValueError("protocol and calibration multiplier differ")
        calibration_record = {
            "instance_scale_calibration_sha256": calibration["calibration_sha256"],
            "instance_scale_calibration_file_sha256": file_sha256(
                instance_scale_calibration
            ),
            "correction_multiplier": correction_multiplier,
            "effective_scale": BASE_EFFECTIVE_SCALE * correction_multiplier,
        }
    elif instance_scale_calibration is not None:
        raise ValueError("instance scale calibration is not allowed for this protocol")
    if protocol["protocol_id"] in ACTION_ROBUST_PROTOCOLS:
        if action_robust_scale_calibration is None:
            raise ValueError("action-robust protocol requires its scale calibration")
        if (
            file_sha256(action_robust_scale_calibration)
            != ACTION_ROBUST_SCALE_FILE_SHA256
        ):
            raise ValueError("action-robust scale calibration bytes changed")
        calibration = load_action_robust_scale_calibration(
            action_robust_scale_calibration
        )
        validation = validate_action_robust_scale_calibration(calibration)
        correction_multiplier = float(validation["multipliers"].get(object_name, 1.0))
        protocol_multiplier = float(
            protocol["method"]["action_robust_scale_calibration"]["multipliers"][
                object_name
            ]
        )
        if correction_multiplier != protocol_multiplier:
            raise ValueError("protocol and action-robust multiplier differ")
        calibration_record = {
            "action_robust_scale_calibration_sha256": calibration["calibration_sha256"],
            "action_robust_scale_calibration_file_sha256": file_sha256(
                action_robust_scale_calibration
            ),
            "correction_multiplier": correction_multiplier,
            "effective_scale": (
                ACTION_ROBUST_BASE_EFFECTIVE_SCALE * correction_multiplier
            ),
        }
    elif action_robust_scale_calibration is not None:
        raise ValueError(
            "action-robust scale calibration is not allowed for this protocol"
        )
    if output_dir.exists():
        raise FileExistsError(f"prediction output already exists: {output_dir}")
    if file_sha256(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source result bytes changed")
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("prediction implementation checkout is dirty")
    implementation_revision = _git_revision(REPOSITORY_ROOT)
    if _git_revision(upstream_checkout) != UPSTREAM_COMMIT:
        raise ValueError("upstream PokeFlex checkout changed")
    if not _git_clean(upstream_checkout):
        raise ValueError("upstream PokeFlex checkout is dirty")
    checkpoint_hashes = {
        filename: file_sha256(checkpoint_root / filename)
        for filename in CHECKPOINT_SHA256
    }
    if checkpoint_hashes != CHECKPOINT_SHA256:
        raise ValueError("released checkpoint bytes changed")

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    if not robot_by_frame:
        raise ValueError("robot trajectory is empty")
    active = [
        frame
        for frame, record in sorted(robot_by_frame.items())
        if float(record["forces"][1]) > 3.0
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )
    frame_limit = max(robot_by_frame)
    if frame_limit < 6:
        raise ValueError("target take is shorter than the checkpoint history")

    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
    )
    views_by_frame: dict[int, tuple[np.ndarray, ...]] = {}
    features_by_frame: dict[int, object] = {}
    preprocessing_by_frame: dict[int, object] = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        feature, preprocessing = checkpoint.encode_frame(views)
        views_by_frame[frame] = views
        features_by_frame[frame] = feature
        preprocessing_by_frame[frame] = preprocessing

    predictions_by_frame = {}
    for frame in range(6, frame_limit + 1):
        history = range(frame - checkpoint.history_frame_count, frame)
        predictions_by_frame[frame] = checkpoint.predict_from_encoded_history(
            [features_by_frame[index] for index in history],
            [preprocessing_by_frame[index] for index in history],
        )

    config = PokeFlexBayesianRegistrationConfig(residual_geometry="point_to_point")
    updates_by_frame = {}
    corrections_by_frame: dict[int, np.ndarray] = {}
    for source_frame in range(6, frame_limit):
        source_prior = predictions_by_frame[source_frame].vertices_m
        action_supported = float(robot_by_frame[source_frame]["forces"][1]) > 3.0
        update = register_pokeflex_graph_posterior(
            source_prior,
            views_by_frame[source_frame],
            action_supported=action_supported,
            prior_faces=template_faces,
            config=config,
        )
        updates_by_frame[source_frame] = update
        corrections_by_frame[source_frame] = update.posterior_vertices_m - source_prior

    target_frames = np.arange(6, frame_limit + 1, dtype=np.int64)
    baseline_rows = []
    global_candidate_rows = []
    candidate_rows = []
    update_supported = []
    update_accepted = []
    action_supported_rows = []
    robot_history_supported_rows = []
    correction_rms = []
    diagnostics = []
    for target_frame in target_frames:
        target_frame_int = int(target_frame)
        source_frame = target_frame_int - 1
        target_prior = predictions_by_frame[target_frame_int].vertices_m
        accepted = source_frame in updates_by_frame and bool(
            updates_by_frame[source_frame].accepted
        )
        source_record = robot_by_frame.get(source_frame, {})
        source_forces = np.asarray(source_record.get("forces"), dtype=np.float64)
        action_supported = bool(
            source_forces.ndim == 1
            and len(source_forces) >= 2
            and np.isfinite(source_forces[1])
            and source_forces[1] > 3.0
        )
        robot_history_supported = action_field_history_is_supported(
            robot_by_frame,
            source_frame,
        )
        supported = bool(accepted and action_supported and robot_history_supported)
        if supported:
            source_prior = predictions_by_frame[source_frame].vertices_m
            correction = corrections_by_frame[source_frame]
            fields = _correction_field_variants(
                source_prior,
                target_prior,
                correction,
                ("action_local_state_relative_0.4",),
                previous_correction=corrections_by_frame.get(source_frame - 1),
                tool_positions=np.asarray(
                    [
                        robot_by_frame[frame]["T_WT"]
                        for frame in range(max(1, source_frame - 3), source_frame + 1)
                    ],
                    dtype=np.float64,
                )[:, :3, 3],
                end_effector_positions=np.asarray(
                    [
                        robot_by_frame[frame]["T_WE"]
                        for frame in range(max(1, source_frame - 3), source_frame + 1)
                    ],
                    dtype=np.float64,
                )[:, :3, 3],
                force_vectors=np.asarray(
                    [
                        robot_by_frame[frame]["forces"][:3]
                        for frame in range(max(1, source_frame - 3), source_frame + 1)
                    ],
                    dtype=np.float64,
                ),
            )
            field = fields["action_local_state_relative_0.4"]
            global_candidate = target_prior + BASE_EFFECTIVE_SCALE * field
            candidate = (
                target_prior + BASE_EFFECTIVE_SCALE * correction_multiplier * field
            )
            field_rms = float(np.sqrt(np.mean(np.sum(np.square(field), axis=1))))
        else:
            global_candidate = target_prior.copy()
            candidate = target_prior.copy()
            field_rms = 0.0
        if not supported and not np.array_equal(candidate, target_prior):
            raise AssertionError(
                "unsupported target prediction changed checkpoint bytes"
            )
        baseline_rows.append(target_prior)
        global_candidate_rows.append(global_candidate)
        candidate_rows.append(candidate)
        update_supported.append(supported)
        update_accepted.append(accepted)
        action_supported_rows.append(action_supported)
        robot_history_supported_rows.append(robot_history_supported)
        correction_rms.append(field_rms)
        update = updates_by_frame.get(source_frame)
        diagnostics.append(
            {
                "target_frame": target_frame_int,
                "source_frame": source_frame,
                "accepted": accepted,
                "action_supported": action_supported,
                "robot_history_supported": robot_history_supported,
                "update_supported": supported,
                "reason": (
                    "missing-required-action-history"
                    if not robot_history_supported
                    else (
                        update.reason
                        if update is not None
                        else "no-five-frame-source-prior"
                    )
                ),
                "association_count": (
                    int(update.diagnostics.get("association_count", 0))
                    if update is not None
                    else 0
                ),
                "correction_field_rms_m": field_rms,
            }
        )

    baseline = np.asarray(baseline_rows, dtype=np.float64)
    global_candidate = np.asarray(global_candidate_rows, dtype=np.float64)
    candidate = np.asarray(candidate_rows, dtype=np.float64)
    supported_array = np.asarray(update_supported, dtype=np.bool_)
    fallback_mismatches = int(
        np.sum(
            np.any(
                candidate[~supported_array].view(np.uint64)
                != baseline[~supported_array].view(np.uint64),
                axis=(1, 2),
            )
        )
    )
    if fallback_mismatches:
        raise AssertionError("one or more fallback frames changed checkpoint bytes")
    global_fallback_mismatches = int(
        np.sum(
            np.any(
                global_candidate[~supported_array].view(np.uint64)
                != baseline[~supported_array].view(np.uint64),
                axis=(1, 2),
            )
        )
    )
    if global_fallback_mismatches:
        raise AssertionError(
            "one or more global fallback frames changed checkpoint bytes"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    npz_path = output_dir / "prediction.npz"
    prediction_arrays = {
        "baseline_vertices_m": baseline,
        "candidate_vertices_m": candidate,
        "faces": np.asarray(template_faces, dtype=np.int64),
        "target_frames": target_frames,
        "source_frames": target_frames - 1,
        "history_start_frames": target_frames - 5,
        "history_end_frames": target_frames - 1,
        "update_supported": supported_array,
        "update_accepted": np.asarray(update_accepted, dtype=np.bool_),
        "action_supported": np.asarray(action_supported_rows, dtype=np.bool_),
        "robot_history_supported": np.asarray(
            robot_history_supported_rows,
            dtype=np.bool_,
        ),
        "correction_rms_m": np.asarray(correction_rms, dtype=np.float64),
    }
    if protocol["protocol_id"] in {
        TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
        TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
        TARGET_PROTOCOL_ACTION_ROBUST_OFFICIAL13_PUBLIC_V1,
    }:
        prediction_arrays["global_candidate_vertices_m"] = global_candidate
    np.savez_compressed(
        npz_path,
        **prediction_arrays,
    )

    input_paths = [robot_path, template_path]
    input_paths.extend(
        take_root / "kinect" / str(camera) / "camera_parameters.json"
        for camera in (0, 1)
    )
    input_paths.extend(
        take_root / "kinect" / str(camera) / "depth" / f"{frame:05d}.png"
        for camera in (0, 1)
        for frame in range(1, frame_limit)
    )
    input_records = [_input_record(path, take_root) for path in input_paths]
    if stage is not None:
        expected_files = stage["files_by_path"]
        for path in input_paths:
            validate_staged_file(path, take_root, expected_files)

    seal: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkagePredictionSeal",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "selected_arm": SELECTED_ARM,
        "take_id": take_root.name,
        "object_name": object_name,
        "implementation_revision": implementation_revision,
        "implementation_clean": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_hashes,
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": len(target_frames),
        "supported_frame_count": int(np.sum(supported_array)),
        "fallback_frame_count": int(np.sum(~supported_array)),
        "fallback_mismatch_count": fallback_mismatches,
        "missing_robot_history_frame_count": int(
            np.sum(~np.asarray(robot_history_supported_rows, dtype=np.bool_))
        ),
        "template_frame": template_frame,
        "template_preprocessing": template_preprocessing,
        "causal_history": "each prediction f uses Kinect frames f-5 through f-1",
        "future_mesh_read": False,
        "future_mesh_read_count": 0,
        "inputs": input_records,
        "updates": diagnostics,
    }
    if source_archive_record is not None:
        seal.update(source_archive_record)
    if calibration_record is not None:
        seal.update(calibration_record)
        seal["global_fallback_mismatch_count"] = global_fallback_mismatches
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    _write_json(output_dir / "seal.json", seal)
    print(
        json.dumps(
            {
                "take_id": take_root.name,
                "prediction_npz_sha256": seal["prediction_npz_sha256"],
                "seal_sha256": seal["seal_sha256"],
                "predicted_frame_count": seal["predicted_frame_count"],
                "supported_frame_count": seal["supported_frame_count"],
                "future_mesh_read": False,
            },
            indent=2,
        )
    )


def _barrier(prediction_root: Path, output: Path, protocol_path: Path) -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(protocol_path)
    target_take_ids = target_take_ids_for_protocol(protocol)
    seal_paths = [
        prediction_root / take_id / "seal.json" for take_id in target_take_ids
    ]
    payload = build_prediction_barrier(seal_paths, protocol)
    _write_json(output, payload)
    print(json.dumps(payload, indent=2))


def _locate_take(dataset_root: Path, take_id: str) -> Path:
    candidates = sorted(path for path in dataset_root.rglob(take_id) if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"expected one target take root for {take_id}, found {len(candidates)}"
        )
    return candidates[0]


def _target_mesh_loader(
    take_root: Path,
    staged_files: Mapping[str, Mapping[str, object]] | None = None,
):
    def load(frame: int) -> tuple[np.ndarray, np.ndarray]:
        path = take_root / "meshes" / f"mesh-f{frame:05d}.obj"
        if staged_files is not None:
            validate_staged_file(path, take_root, staged_files)
        mesh = _load_mesh(path)
        return (
            np.asarray(mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(mesh.faces, dtype=np.int64),
        )

    return load


def _score(
    dataset_root: Path,
    prediction_root: Path,
    barrier_path: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = load_pokeflex_shrinkage_target_protocol(protocol_path)
    target_take_ids = target_take_ids_for_protocol(protocol)
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    validate_prediction_barrier(barrier, protocol)
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("target scorer checkout is dirty")
    if _git_revision(REPOSITORY_ROOT) != barrier["implementation_revision"]:
        raise ValueError("target scorer revision differs from prediction revision")
    barrier_predictions = {str(row["take_id"]): row for row in barrier["predictions"]}
    per_object = []
    for take_id in target_take_ids:
        archive = validate_prediction_seal(
            prediction_root / take_id / "seal.json",
            protocol,
        )
        barrier_prediction = barrier_predictions[take_id]
        if file_sha256(archive.seal_path) != barrier_prediction["seal_file_sha256"]:
            raise ValueError(f"prediction seal changed after barrier: {take_id}")
        if file_sha256(archive.npz_path) != barrier_prediction["prediction_npz_sha256"]:
            raise ValueError(f"prediction archive changed after barrier: {take_id}")
        take_root = _locate_take(dataset_root, take_id)
        staged_files = None
        if protocol["protocol_id"] in {
            TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
            TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
            TARGET_PROTOCOL_ACTION_ROBUST_FRESH6_V3,
        }:
            seal_payload = json.loads(archive.seal_path.read_text(encoding="utf-8"))
            stage_path = take_root / STAGE_MANIFEST_NAME
            stage = validate_pokeflex_fresh12_stage_manifest(
                stage_path,
                protocol,
                expected_take_id=take_id,
            )
            if (
                file_sha256(stage_path)
                != seal_payload["source_stage_manifest_file_sha256"]
                or stage["stage_manifest_sha256"]
                != seal_payload["source_stage_manifest_sha256"]
            ):
                raise ValueError(f"stage manifest changed after prediction: {take_id}")
            staged_files = stage["files_by_path"]
        robot_path = take_root / "robot_data.json"
        if staged_files is not None:
            validate_staged_file(robot_path, take_root, staged_files)
        robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
        active_frames = [
            int(row["frame"])
            for row in robot_records
            if float(row["forces"][1]) > 3.0 and int(row["frame"]) >= 6
        ]

        row = score_one_prediction(
            archive,
            active_frames,
            _target_mesh_loader(take_root, staged_files),
            protocol,
        )
        row["prediction_seal_sha256"] = file_sha256(archive.seal_path)
        row["robot_sha256"] = file_sha256(robot_path)
        per_object.append(row)
    aggregate = evaluate_target_metrics(per_object, protocol)
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkageTargetResult",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "barrier_file_sha256": file_sha256(barrier_path),
        "barrier_sha256": barrier["barrier_sha256"],
        "selected_arm": SELECTED_ARM,
        "target_meshes_opened_after_complete_barrier": True,
        "target_object_count": len(per_object),
        "objects": per_object,
        "aggregate": aggregate,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(output, result)
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_conservative_shrinkage_target_v2.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("take_root", type=Path)
    predict.add_argument("output_dir", type=Path)
    predict.add_argument("--source-result", type=Path, required=True)
    predict.add_argument("--upstream-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint-root", type=Path, required=True)
    predict.add_argument("--source-stage-manifest", type=Path)
    predict.add_argument("--instance-scale-calibration", type=Path)
    predict.add_argument("--action-robust-scale-calibration", type=Path)

    barrier = subparsers.add_parser("barrier")
    barrier.add_argument("prediction_root", type=Path)
    barrier.add_argument("output", type=Path)

    score = subparsers.add_parser("score")
    score.add_argument("dataset_root", type=Path)
    score.add_argument("prediction_root", type=Path)
    score.add_argument("barrier", type=Path)
    score.add_argument("output", type=Path)

    args = parser.parse_args()
    if args.command == "predict":
        _predict(
            args.take_root,
            args.output_dir,
            args.protocol,
            args.source_result,
            args.upstream_checkout,
            args.checkpoint_root,
            args.source_stage_manifest,
            args.instance_scale_calibration,
            args.action_robust_scale_calibration,
        )
    elif args.command == "barrier":
        _barrier(args.prediction_root, args.output, args.protocol)
    else:
        _score(
            args.dataset_root,
            args.prediction_root,
            args.barrier,
            args.output,
            args.protocol,
        )


if __name__ == "__main__":
    main()
