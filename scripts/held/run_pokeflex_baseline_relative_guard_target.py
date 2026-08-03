#!/usr/bin/env python3
"""Stage, seal, and score the prospective guarded PokeFlex update."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "remote"))

from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _correction_field_variants,
    _load_official_template,
)

from bayesian_phystwin.pokeflex_baseline_relative_guard import (  # noqa: E402
    apply_baseline_relative_guard,
    baseline_relative_guard_decision,
    certificate_from_payload,
    extract_baseline_relative_guard_features,
)
from bayesian_phystwin.pokeflex_baseline_relative_guard_target import (  # noqa: E402
    CERTIFICATE_SHA256,
    CHECKPOINT_SHA256,
    SOURCE_RESULT_SHA256,
    TARGET_TAKE_IDS,
    UPSTREAM_COMMIT,
    build_prediction_barrier,
    canonical_payload_sha256,
    evaluate_target_metrics,
    file_sha256,
    load_protocol,
    score_one_prediction,
    seal_sha256,
    validate_prediction_barrier,
    validate_prediction_seal,
)
from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    action_field_history_is_supported,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)


def _git_output(checkout: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(checkout), *arguments), text=True
    ).strip()


def _git_revision(checkout: Path) -> str:
    return _git_output(checkout, "rev-parse", "HEAD")


def _git_clean(checkout: Path) -> bool:
    return not bool(_git_output(checkout, "status", "--porcelain"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input_record(path: Path, take_root: Path) -> dict[str, object]:
    return {
        "path_relative_to_take": str(path.resolve().relative_to(take_root.resolve())),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _zip_member(zip_file: zipfile.ZipFile, take_id: str, relative: str) -> bytes:
    name = f"{take_id}/{relative}"
    try:
        return zip_file.read(name)
    except KeyError as error:
        raise ValueError(f"required archive member is missing: {name}") from error


def _stage(zip_path: Path, output_dir: Path, protocol_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    take_id = zip_path.stem
    if take_id not in TARGET_TAKE_IDS:
        raise ValueError("archive is outside the target cohort")
    if output_dir.exists():
        raise FileExistsError(f"staging output already exists: {output_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        robot_bytes = _zip_member(archive, take_id, "robot_data.json")
        robot_records = json.loads(robot_bytes)
        robot_by_frame = {int(row["frame"]): row for row in robot_records}
        if not robot_by_frame:
            raise ValueError("robot trajectory is empty")
        active = [
            frame
            for frame, row in sorted(robot_by_frame.items())
            if float(row["forces"][1]) > 3.0
        ]
        template_frame = _template_frame(active)
        frame_limit = max(robot_by_frame)
        authorized: dict[str, bytes] = {
            "robot_data.json": robot_bytes,
            f"meshes/mesh-f{template_frame:05d}.obj": _zip_member(
                archive, take_id, f"meshes/mesh-f{template_frame:05d}.obj"
            ),
        }
        for camera in (0, 1):
            parameter_bytes = _zip_member(
                archive, take_id, f"kinect/{camera}/camera_parameters.json"
            )
            parameters = json.loads(parameter_bytes)
            if np.asarray(parameters.get("depth_intrinsics")).shape != (3, 3):
                raise ValueError(f"invalid Kinect depth intrinsics for camera {camera}")
            if np.asarray(parameters.get("depth_extrinsics")).shape != (4, 4):
                raise ValueError(f"invalid Kinect depth extrinsics for camera {camera}")
            authorized[f"kinect/{camera}/camera_parameters.json"] = parameter_bytes
            for frame in range(1, frame_limit):
                output_name = f"kinect/{camera}/depth/{frame:05d}.png"
                authorized[output_name] = _zip_member(
                    archive, take_id, f"kinect/{camera}/depth/{frame:05d}.png"
                )
    output_dir.mkdir(parents=True, exist_ok=False)
    members = []
    for relative, content in sorted(authorized.items()):
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        members.append(
            {
                "path_relative_to_take": relative,
                "sha256": _bytes_sha256(content),
                "byte_count": len(content),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexGuardCausalInputManifest",
        "protocol_sha256": protocol["protocol_sha256"],
        "take_id": take_id,
        "source_archive_path": str(zip_path.resolve()),
        "frame_limit": frame_limit,
        "template_frame": template_frame,
        "authorized_members": members,
        "authorized_member_count": len(members),
        "template_mesh_read_count": 1,
        "future_target_mesh_read_count": 0,
        "target_frame_observation_read_count": 0,
        "claim_boundary": (
            "Only robot metadata, two causal depth streams, camera calibration, "
            "and the explicit upstream template mesh were materialized."
        ),
    }
    manifest["manifest_sha256"] = canonical_payload_sha256(
        manifest, digest_field="manifest_sha256"
    )
    _write_json(output_dir / "causal_input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "take_id": take_id,
                "authorized_member_count": len(members),
                "future_target_mesh_read_count": 0,
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
        )
    )


def _validate_causal_input_manifest(take_root: Path, protocol: dict[str, object]) -> None:
    path = take_root / "causal_input_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = canonical_payload_sha256(payload, digest_field="manifest_sha256")
    if payload.get("manifest_sha256") != observed:
        raise ValueError("causal input manifest checksum mismatch")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("causal input protocol changed")
    if payload.get("take_id") != take_root.name:
        raise ValueError("causal input take changed")
    if int(payload.get("future_target_mesh_read_count", -1)) != 0:
        raise ValueError("causal input staging read a future target mesh")
    if int(payload.get("target_frame_observation_read_count", -1)) != 0:
        raise ValueError("causal input staging read a target observation")
    for record in payload["authorized_members"]:
        member = take_root / str(record["path_relative_to_take"])
        if not member.is_file() or file_sha256(member) != record["sha256"]:
            raise ValueError(f"causal input member changed: {member}")


def _predict(
    take_root: Path,
    output_dir: Path,
    protocol_path: Path,
    source_result_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    take_root = take_root.resolve()
    output_dir = output_dir.resolve()
    if take_root.name not in TARGET_TAKE_IDS:
        raise ValueError("take is outside the target lock")
    if output_dir.exists():
        raise FileExistsError(f"prediction output already exists: {output_dir}")
    _validate_causal_input_manifest(take_root, protocol)
    if file_sha256(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source result bytes changed")
    if not _git_clean(ROOT):
        raise ValueError("prediction implementation checkout is dirty")
    implementation_revision = _git_revision(ROOT)
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
    certificate = certificate_from_payload(
        protocol["development_guard"]["certificate"]
    )

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(row["frame"]): row for row in robot_records}
    active = [
        frame
        for frame, row in sorted(robot_by_frame.items())
        if float(row["forces"][1]) > 3.0
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )
    frame_limit = max(robot_by_frame)
    if frame_limit < 6:
        raise ValueError("target take is shorter than checkpoint history")
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

    registration_config = PokeFlexBayesianRegistrationConfig(
        residual_geometry="point_to_point"
    )
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
            config=registration_config,
        )
        updates_by_frame[source_frame] = update
        corrections_by_frame[source_frame] = update.posterior_vertices_m - source_prior

    target_frames = np.arange(6, frame_limit + 1, dtype=np.int64)
    baseline_rows = []
    candidate_rows = []
    raw_support_rows = []
    guard_support_rows = []
    guard_accepted_rows = []
    update_accepted_rows = []
    action_supported_rows = []
    robot_supported_rows = []
    association_counts = []
    raw_rms_rows = []
    field_rms_rows = []
    predicted_regret_rows = []
    upper_regret_rows = []
    diagnostics = []
    for target_frame in target_frames:
        target_frame_int = int(target_frame)
        source_frame = target_frame_int - 1
        target_prior = predictions_by_frame[target_frame_int].vertices_m
        update = updates_by_frame.get(source_frame)
        update_accepted = bool(update is not None and update.accepted)
        source_record = robot_by_frame.get(source_frame, {})
        source_forces = np.asarray(source_record.get("forces"), dtype=np.float64)
        action_supported = bool(
            source_forces.ndim == 1
            and len(source_forces) >= 2
            and np.isfinite(source_forces[1])
            and source_forces[1] > 3.0
        )
        robot_supported = action_field_history_is_supported(
            robot_by_frame, source_frame
        )
        raw_supported = bool(update_accepted and action_supported and robot_supported)
        association_count = (
            int(update.diagnostics.get("association_count", 0))
            if update is not None
            else 0
        )
        raw_rms = 0.0
        field_rms = 0.0
        guard_decision = {
            "accepted": False,
            "in_source_support": False,
            "predicted_regret_mm": None,
            "upper_regret_mm": None,
            "reason": "raw-update-unavailable",
        }
        if raw_supported:
            source_prior = predictions_by_frame[source_frame].vertices_m
            raw_correction = corrections_by_frame[source_frame]
            previous_raw = corrections_by_frame.get(
                source_frame - 1, np.zeros_like(raw_correction)
            )
            guard_features = extract_baseline_relative_guard_features(
                raw_correction,
                source_prior,
                target_prior,
                previous_raw,
                association_count=association_count,
            )
            guard_decision = baseline_relative_guard_decision(
                certificate, guard_features
            )
            fields = _correction_field_variants(
                source_prior,
                target_prior,
                raw_correction,
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
            raw_candidate = target_prior + 0.125 * field
            candidate, repeated_decision = apply_baseline_relative_guard(
                target_prior, raw_candidate, certificate, guard_features
            )
            if repeated_decision != guard_decision:
                raise AssertionError("guard decision changed during application")
            raw_rms = float(
                np.sqrt(np.mean(np.sum(np.square(raw_correction), axis=1)))
            )
            field_rms = float(np.sqrt(np.mean(np.sum(np.square(field), axis=1))))
        else:
            candidate = target_prior.copy()
        guard_accepted = bool(guard_decision["accepted"])
        if not guard_accepted and not np.array_equal(candidate, target_prior):
            raise AssertionError("guard fallback changed checkpoint bytes")
        baseline_rows.append(target_prior)
        candidate_rows.append(candidate)
        raw_support_rows.append(raw_supported)
        guard_support_rows.append(bool(guard_decision["in_source_support"]))
        guard_accepted_rows.append(guard_accepted)
        update_accepted_rows.append(update_accepted)
        action_supported_rows.append(action_supported)
        robot_supported_rows.append(robot_supported)
        association_counts.append(association_count)
        raw_rms_rows.append(raw_rms)
        field_rms_rows.append(field_rms)
        predicted_regret_rows.append(
            float(guard_decision["predicted_regret_mm"])
            if guard_decision["predicted_regret_mm"] is not None
            else np.nan
        )
        upper_regret_rows.append(
            float(guard_decision["upper_regret_mm"])
            if guard_decision["upper_regret_mm"] is not None
            else np.nan
        )
        diagnostics.append(
            {
                "target_frame": target_frame_int,
                "source_frame": source_frame,
                "raw_update_supported": raw_supported,
                "guard_in_source_support": guard_decision["in_source_support"],
                "guard_accepted": guard_accepted,
                "guard_reason": guard_decision["reason"],
                "guard_predicted_regret_mm": guard_decision[
                    "predicted_regret_mm"
                ],
                "guard_upper_regret_mm": guard_decision["upper_regret_mm"],
                "association_count": association_count,
            }
        )

    baseline = np.asarray(baseline_rows, dtype=np.float64)
    candidate = np.asarray(candidate_rows, dtype=np.float64)
    accepted_array = np.asarray(guard_accepted_rows, dtype=np.bool_)
    fallback_mismatches = int(
        np.sum(
            np.any(
                candidate[~accepted_array].view(np.uint64)
                != baseline[~accepted_array].view(np.uint64),
                axis=(1, 2),
            )
        )
    )
    if fallback_mismatches:
        raise AssertionError("one or more guard fallbacks changed checkpoint bytes")
    output_dir.mkdir(parents=True, exist_ok=False)
    npz_path = output_dir / "prediction.npz"
    np.savez_compressed(
        npz_path,
        baseline_vertices_m=baseline,
        candidate_vertices_m=candidate,
        faces=np.asarray(template_faces, dtype=np.int64),
        target_frames=target_frames,
        source_frames=target_frames - 1,
        history_start_frames=target_frames - 5,
        history_end_frames=target_frames - 1,
        raw_update_supported=np.asarray(raw_support_rows, dtype=np.bool_),
        guard_in_source_support=np.asarray(guard_support_rows, dtype=np.bool_),
        guard_accepted=accepted_array,
        update_accepted=np.asarray(update_accepted_rows, dtype=np.bool_),
        action_supported=np.asarray(action_supported_rows, dtype=np.bool_),
        robot_history_supported=np.asarray(robot_supported_rows, dtype=np.bool_),
        association_count=np.asarray(association_counts, dtype=np.int64),
        raw_correction_rms_m=np.asarray(raw_rms_rows, dtype=np.float64),
        correction_field_rms_m=np.asarray(field_rms_rows, dtype=np.float64),
        guard_predicted_regret_mm=np.asarray(
            predicted_regret_rows, dtype=np.float64
        ),
        guard_upper_regret_mm=np.asarray(upper_regret_rows, dtype=np.float64),
    )
    input_paths = [
        robot_path,
        template_path,
        take_root / "causal_input_manifest.json",
    ]
    input_paths.extend(
        take_root / "kinect" / str(camera) / "camera_parameters.json"
        for camera in (0, 1)
    )
    input_paths.extend(
        take_root / "kinect" / str(camera) / "depth" / f"{frame:05d}.png"
        for camera in (0, 1)
        for frame in range(1, frame_limit)
    )
    seal: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardPredictionSeal",
        "protocol_sha256": protocol["protocol_sha256"],
        "certificate_sha256": CERTIFICATE_SHA256,
        "take_id": take_root.name,
        "object_name": take_root.name.rpartition("_T")[0],
        "implementation_revision": implementation_revision,
        "implementation_clean": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_hashes,
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": len(target_frames),
        "raw_supported_frame_count": int(np.sum(raw_support_rows)),
        "guard_accepted_frame_count": int(np.sum(accepted_array)),
        "fallback_frame_count": int(np.sum(~accepted_array)),
        "fallback_mismatch_count": fallback_mismatches,
        "template_frame": template_frame,
        "template_preprocessing": template_preprocessing,
        "causal_history": "each prediction f uses Kinect frames f-5 through f-1",
        "future_mesh_read": False,
        "future_mesh_read_count": 0,
        "inputs": [_input_record(path, take_root) for path in input_paths],
        "updates": diagnostics,
    }
    seal["seal_sha256"] = seal_sha256(seal)
    _write_json(output_dir / "seal.json", seal)
    print(
        json.dumps(
            {
                "take_id": take_root.name,
                "seal_sha256": seal["seal_sha256"],
                "raw_supported_frame_count": seal["raw_supported_frame_count"],
                "guard_accepted_frame_count": seal["guard_accepted_frame_count"],
                "future_mesh_read": False,
            },
            indent=2,
        )
    )


def _barrier(prediction_root: Path, output: Path, protocol_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    payload = build_prediction_barrier(
        [prediction_root / take_id / "seal.json" for take_id in TARGET_TAKE_IDS],
        protocol,
    )
    _write_json(output, payload)
    print(json.dumps(payload, indent=2))


def _find_take_zip(dataset_root: Path, take_id: str) -> Path:
    candidates = sorted(dataset_root.rglob(f"{take_id}.zip"))
    if len(candidates) != 1:
        raise ValueError(f"expected one archive for {take_id}, found {len(candidates)}")
    return candidates[0]


def _mesh_from_bytes(value: bytes) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(io.BytesIO(value), file_type="obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("target OBJ did not decode to one triangle mesh")
    return (
        np.asarray(mesh.vertices, dtype=np.float64) / 1000.0,
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _score(
    dataset_root: Path,
    prediction_root: Path,
    barrier_path: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    validate_prediction_barrier(barrier, protocol)
    if not _git_clean(ROOT):
        raise ValueError("target scorer checkout is dirty")
    if _git_revision(ROOT) != barrier["implementation_revision"]:
        raise ValueError("target scorer revision differs from predictions")
    barrier_by_take = {str(row["take_id"]): row for row in barrier["predictions"]}
    per_take = []
    for take_id in TARGET_TAKE_IDS:
        archive = validate_prediction_seal(
            prediction_root / take_id / "seal.json", protocol
        )
        record = barrier_by_take[take_id]
        if file_sha256(archive.seal_path) != record["seal_file_sha256"]:
            raise ValueError(f"prediction seal changed after barrier: {take_id}")
        if file_sha256(archive.npz_path) != record["prediction_npz_sha256"]:
            raise ValueError(f"prediction archive changed after barrier: {take_id}")
        zip_path = _find_take_zip(dataset_root, take_id)
        with zipfile.ZipFile(zip_path) as zip_file:
            robot_bytes = _zip_member(zip_file, take_id, "robot_data.json")
            robot_records = json.loads(robot_bytes)
            active_frames = [
                int(row["frame"])
                for row in robot_records
                if float(row["forces"][1]) > 3.0 and int(row["frame"]) >= 6
            ]

            def mesh_loader(
                frame: int, current_take_id: str = take_id
            ) -> tuple[np.ndarray, np.ndarray]:
                return _mesh_from_bytes(
                    _zip_member(
                        zip_file,
                        current_take_id,
                        f"meshes/mesh-f{frame:05d}.obj",
                    )
                )

            row = score_one_prediction(
                archive, active_frames, mesh_loader, protocol
            )
        row["prediction_seal_sha256"] = file_sha256(archive.seal_path)
        row["robot_member_sha256"] = _bytes_sha256(robot_bytes)
        row["source_archive_path"] = str(zip_path.resolve())
        per_take.append(row)
    aggregate = evaluate_target_metrics(per_take, protocol)
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardTargetResult",
        "protocol_sha256": protocol["protocol_sha256"],
        "certificate_sha256": CERTIFICATE_SHA256,
        "barrier_file_sha256": file_sha256(barrier_path),
        "barrier_sha256": barrier["barrier_sha256"],
        "target_meshes_opened_after_complete_barrier": True,
        "target_object_count": len(per_take),
        "objects": per_take,
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
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_baseline_relative_guard_public_paired_v2.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("zip_path", type=Path)
    stage.add_argument("output_dir", type=Path)
    predict = subparsers.add_parser("predict")
    predict.add_argument("take_root", type=Path)
    predict.add_argument("output_dir", type=Path)
    predict.add_argument("--source-result", type=Path, required=True)
    predict.add_argument("--upstream-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint-root", type=Path, required=True)
    barrier = subparsers.add_parser("barrier")
    barrier.add_argument("prediction_root", type=Path)
    barrier.add_argument("output", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("dataset_root", type=Path)
    score.add_argument("prediction_root", type=Path)
    score.add_argument("barrier", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        _stage(args.zip_path, args.output_dir, args.protocol)
    elif args.command == "predict":
        _predict(
            args.take_root,
            args.output_dir,
            args.protocol,
            args.source_result,
            args.upstream_checkout,
            args.checkpoint_root,
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
