#!/usr/bin/env python3
"""Stage, seal, and score the five prospective PokeFlex V5 predictions."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "remote"))

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (  # noqa: E402
    load_archived_public13_result,
    load_official18_v4_protocol,
    validate_author_source_manifest,
)
from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    CHECKPOINT_SHA256,
    UPSTREAM_COMMIT,
    action_field_history_is_supported,
)
from bayesian_phystwin.pokeflex_missing5_completion_v5 import (  # noqa: E402
    validate_completion_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    INPUT_STAGE_KIND,
    PREDICTION_SEAL_KIND,
    RESULT_KIND,
    TARGET_TAKE_IDS,
    build_prediction_barrier,
    evaluate_result,
    file_sha256,
    input_stage_sha256,
    load_execution_protocol,
    prediction_seal_sha256,
    result_sha256,
    score_one_prediction,
    validate_input_stage,
    validate_prediction_barrier,
    validate_prediction_seal,
    validate_result,
    verify_implementation_files,
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


def _template_frame(active_frames: list[int]) -> int:
    if not active_frames:
        raise ValueError("take has no active deformation frames")
    if active_frames[0] != 1:
        return 1
    previous = active_frames[0]
    for frame in active_frames[1:]:
        if frame - previous > 5:
            return int((frame + previous) / 2)
        previous = frame
    raise ValueError("upstream template-selection rule found no inactive gap")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_protocols(
    execution_path: Path,
    completion_path: Path,
    parent_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    parent = load_official18_v4_protocol(parent_path)
    completion = _load_json(completion_path)
    validate_completion_protocol(completion)
    execution = load_execution_protocol(execution_path, completion, parent)
    verify_implementation_files(execution, REPOSITORY_ROOT)
    return execution, completion, parent


def _source_row(
    source_manifest: Mapping[str, object], take_id: str
) -> Mapping[str, object]:
    rows = {str(row["take_id"]): row for row in source_manifest["takes"]}
    if set(rows) != set(TARGET_TAKE_IDS):
        raise ValueError("source manifest cohort changed")
    return rows[take_id]


def _copy_member(
    archive: ZipFile,
    member: str,
    destination: Path,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with archive.open(member) as source, destination.open("xb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
            digest.update(block)
            byte_count += len(block)
    if byte_count <= 0:
        raise ValueError(f"prediction input is empty: {member}")
    return {
        "archive_member": member,
        "staged_relative_path": member,
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _stage(
    source_archive: Path,
    output_dir: Path,
    source_manifest_path: Path,
    execution_path: Path,
    completion_path: Path,
    parent_path: Path,
) -> None:
    execution, completion, parent = _load_protocols(
        execution_path,
        completion_path,
        parent_path,
    )
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("prediction implementation checkout is dirty")
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    source_archive = source_archive.resolve()
    take_id = source_archive.stem
    if take_id not in TARGET_TAKE_IDS:
        raise ValueError(f"source archive is outside target cohort: {take_id}")
    source = _source_row(source_manifest, take_id)
    if source_archive.name != source["source_payload_name"]:
        raise ValueError("source archive name changed")
    if file_sha256(source_archive) != source["source_payload_sha256"]:
        raise ValueError("source archive bytes changed")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"prediction stage already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    try:
        with ZipFile(source_archive) as archive:
            names = {info.filename for info in archive.infolist() if not info.is_dir()}
            robot_member = f"{take_id}/robot_data.json"
            if robot_member not in names:
                raise ValueError("source archive is missing robot_data.json")
            robot_bytes = archive.read(robot_member)
            robot_records = json.loads(robot_bytes.decode("utf-8"))
            robot_by_frame = {int(record["frame"]): record for record in robot_records}
            if not robot_by_frame:
                raise ValueError("robot trajectory is empty")
            active = [
                frame
                for frame, record in sorted(robot_by_frame.items())
                if float(record["forces"][1]) > 3.0
            ]
            template_frame = _template_frame(active)
            frame_limit = max(robot_by_frame)
            if frame_limit != int(source["episode_length"]):
                raise ValueError("robot and mesh episode lengths differ")
            if frame_limit < 6:
                raise ValueError("target take is shorter than checkpoint history")
            template_member = f"{take_id}/meshes/mesh-f{template_frame:05d}.obj"
            allowed_members = {robot_member, template_member}
            allowed_members.update(
                f"{take_id}/kinect/{camera}/camera_parameters.json" for camera in (0, 1)
            )
            allowed_members.update(
                f"{take_id}/kinect/{camera}/depth/{frame:05d}.png"
                for camera in (0, 1)
                for frame in range(1, frame_limit)
            )
            missing = sorted(allowed_members - names)
            if missing:
                raise ValueError(
                    f"source archive is missing prediction inputs: {missing[:3]}"
                )
            inputs = [
                _copy_member(archive, member, temporary / member)
                for member in sorted(allowed_members)
            ]
        stage: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": INPUT_STAGE_KIND,
            "execution_protocol_sha256": execution["execution_protocol_sha256"],
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "take_id": take_id,
            "object_name": take_id.rpartition("_T")[0],
            "source_archive_name": source_archive.name,
            "source_archive_sha256": source["source_payload_sha256"],
            "source_member_manifest_sha256": source["member_manifest_sha256"],
            "frame_limit": frame_limit,
            "template_frame": template_frame,
            "authorized_template_member": template_member,
            "authorized_template_mesh_decoded": True,
            "authorized_template_mesh_decode_count": 1,
            "future_target_mesh_member_decoded_count": 0,
            "target_metric_computed": False,
            "inputs": inputs,
            "implementation_revision": _git_revision(REPOSITORY_ROOT),
            "implementation_clean": True,
            "held_v8_accessed": False,
            "input_stage_sha256": "",
        }
        stage["input_stage_sha256"] = input_stage_sha256(stage)
        validate_input_stage(
            stage,
            execution,
            completion,
            parent,
            source_manifest,
        )
        _write_json(temporary / "stage.json", stage)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "take_id": take_id,
                "stage": str(output_dir),
                "input_stage_sha256": stage["input_stage_sha256"],
                "template_frame": template_frame,
                "authorized_template_mesh_count": 1,
                "future_target_mesh_member_decoded_count": 0,
            },
            indent=2,
        )
    )


def _validate_staged_files(stage_dir: Path, stage: Mapping[str, object]) -> Path:
    expected = {"stage.json"}
    for row in stage["inputs"]:
        relative = str(row["staged_relative_path"])
        expected.add(relative)
        path = stage_dir / relative
        if not path.is_file():
            raise ValueError(f"staged prediction input is missing: {relative}")
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"staged prediction input changed: {relative}")
        if path.stat().st_size != int(row["byte_count"]):
            raise ValueError(f"staged prediction input size changed: {relative}")
    observed = {
        str(path.relative_to(stage_dir))
        for path in stage_dir.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise ValueError("prediction stage contains an unauthorized file")
    return stage_dir / str(stage["take_id"])


def _compute_predictions(
    take_root: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    *,
    global_scale: float,
    v4_scale: float,
    v5_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    from run_pokeflex_bayesian_registration_smoke import _view_points
    from run_pokeflex_checkpoint_registration_independent_depth import (
        _correction_field_variants,
        _load_official_template,
    )

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
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
    candidate_rows = {"global": [], "v4": [], "v5": []}
    update_supported = []
    update_accepted = []
    action_supported_rows = []
    robot_history_supported_rows = []
    correction_rms = []
    diagnostics = []
    for target_frame in target_frames:
        target = int(target_frame)
        source_frame = target - 1
        target_prior = predictions_by_frame[target].vertices_m
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
            candidates = {
                "global": target_prior + global_scale * field,
                "v4": target_prior + v4_scale * field,
                "v5": target_prior + v5_scale * field,
            }
            field_rms = float(np.sqrt(np.mean(np.sum(np.square(field), axis=1))))
        else:
            candidates = {name: target_prior.copy() for name in candidate_rows}
            field_rms = 0.0
        baseline_rows.append(target_prior)
        for name in candidate_rows:
            candidate_rows[name].append(candidates[name])
        update_supported.append(supported)
        update_accepted.append(accepted)
        action_supported_rows.append(action_supported)
        robot_history_supported_rows.append(robot_history_supported)
        correction_rms.append(field_rms)
        update = updates_by_frame.get(source_frame)
        diagnostics.append(
            {
                "target_frame": target,
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
    supported_array = np.asarray(update_supported, dtype=np.bool_)
    arrays: dict[str, np.ndarray] = {
        "baseline_vertices_m": baseline,
        "global_vertices_m": np.asarray(candidate_rows["global"], dtype=np.float64),
        "v4_vertices_m": np.asarray(candidate_rows["v4"], dtype=np.float64),
        "v5_vertices_m": np.asarray(candidate_rows["v5"], dtype=np.float64),
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
    mismatch_counts = {}
    for name in ("global", "v4", "v5"):
        values = arrays[f"{name}_vertices_m"]
        mismatch_counts[name] = int(
            np.sum(
                np.any(
                    values[~supported_array].view(np.uint64)
                    != baseline[~supported_array].view(np.uint64),
                    axis=(1, 2),
                )
            )
        )
        if mismatch_counts[name]:
            raise AssertionError(f"one or more {name} fallback frames changed bytes")
    metadata = {
        "template_frame": template_frame,
        "template_preprocessing": template_preprocessing,
        "frame_limit": frame_limit,
        "diagnostics": diagnostics,
        "fallback_mismatch_counts": mismatch_counts,
        "supported_frame_count": int(np.sum(supported_array)),
    }
    return arrays, metadata


def _predict(
    stage_dir: Path,
    output_dir: Path,
    source_manifest_path: Path,
    execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
) -> None:
    execution, completion, parent = _load_protocols(
        execution_path,
        completion_path,
        parent_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
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
    stage_dir = stage_dir.resolve()
    stage_path = stage_dir / "stage.json"
    stage = _load_json(stage_path)
    stage_validation = validate_input_stage(
        stage,
        execution,
        completion,
        parent,
        source_manifest,
    )
    if stage.get("implementation_revision") != implementation_revision:
        raise ValueError("input stage and prediction revisions differ")
    take_root = _validate_staged_files(stage_dir, stage)
    take_id = str(stage_validation["take_id"])
    method = execution["method"]
    arrays, metadata = _compute_predictions(
        take_root,
        upstream_checkout.resolve(),
        checkpoint_root.resolve(),
        global_scale=float(method["global_effective_scale"]),
        v4_scale=float(method["v4_effective_scales"][take_id]),
        v5_scale=float(method["v5_effective_scales"][take_id]),
    )
    if metadata["template_frame"] != stage_validation["template_frame"]:
        raise ValueError("staged and computed template frames differ")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"prediction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        npz_path = output_dir / "prediction.npz"
        np.savez_compressed(npz_path, **arrays)
        copied_stage = output_dir / "input_stage_manifest.json"
        copied_stage.write_bytes(stage_path.read_bytes())
        seal: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": PREDICTION_SEAL_KIND,
            "execution_protocol_sha256": execution["execution_protocol_sha256"],
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "take_id": take_id,
            "object_name": take_id.rpartition("_T")[0],
            "implementation_revision": implementation_revision,
            "implementation_clean": True,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": checkpoint_hashes,
            "input_stage_manifest": copied_stage.name,
            "input_stage_manifest_file_sha256": file_sha256(copied_stage),
            "input_stage_sha256": stage["input_stage_sha256"],
            "prediction_npz": npz_path.name,
            "prediction_npz_sha256": file_sha256(npz_path),
            "predicted_frame_count": len(arrays["target_frames"]),
            "supported_frame_count": metadata["supported_frame_count"],
            "global_fallback_mismatch_count": metadata["fallback_mismatch_counts"][
                "global"
            ],
            "v4_fallback_mismatch_count": metadata["fallback_mismatch_counts"]["v4"],
            "v5_fallback_mismatch_count": metadata["fallback_mismatch_counts"]["v5"],
            "global_effective_scale": method["global_effective_scale"],
            "v4_effective_scale": method["v4_effective_scales"][take_id],
            "v5_effective_scale": method["v5_effective_scales"][take_id],
            "template_frame": metadata["template_frame"],
            "template_preprocessing": metadata["template_preprocessing"],
            "authorized_template_mesh_read": True,
            "authorized_template_mesh_read_count": 1,
            "future_target_mesh_read": False,
            "future_target_mesh_read_count": 0,
            "future_observation_used": False,
            "target_metric_computed": False,
            "causal_history": "each prediction f uses Kinect depth frames f-5 through f-1",
            "updates": metadata["diagnostics"],
            "held_v8_accessed": False,
            "seal_sha256": "",
        }
        seal["seal_sha256"] = prediction_seal_sha256(seal)
        _write_json(output_dir / "seal.json", seal)
        validate_prediction_seal(
            output_dir / "seal.json",
            execution,
            completion,
            parent,
            source_manifest,
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "take_id": take_id,
                "prediction_npz_sha256": seal["prediction_npz_sha256"],
                "seal_sha256": seal["seal_sha256"],
                "predicted_frame_count": seal["predicted_frame_count"],
                "supported_frame_count": seal["supported_frame_count"],
                "future_target_mesh_read": False,
            },
            indent=2,
        )
    )


def _barrier(
    prediction_root: Path,
    output: Path,
    source_manifest_path: Path,
    execution_path: Path,
    completion_path: Path,
    parent_path: Path,
) -> None:
    execution, completion, parent = _load_protocols(
        execution_path,
        completion_path,
        parent_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    seal_paths = [
        prediction_root / take_id / "seal.json" for take_id in TARGET_TAKE_IDS
    ]
    payload = build_prediction_barrier(
        seal_paths,
        execution,
        completion,
        parent,
        source_manifest,
    )
    _write_json(output, payload)
    print(json.dumps(payload, indent=2))


def _mesh_from_bytes(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(io.BytesIO(payload), file_type="obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("expected one triangle target mesh")
    return (
        np.asarray(mesh.vertices, dtype=np.float64) / 1000.0,
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _locate_archive(source_root: Path, take_id: str) -> Path:
    matches = sorted(source_root.rglob(f"{take_id}.zip"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one source archive for {take_id}, found {len(matches)}"
        )
    return matches[0]


def _score(
    source_root: Path,
    prediction_root: Path,
    barrier_path: Path,
    output: Path,
    source_manifest_path: Path,
    execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    public13_path: Path,
) -> None:
    execution, completion, parent = _load_protocols(
        execution_path,
        completion_path,
        parent_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    barrier = _load_json(barrier_path)
    validate_prediction_barrier(barrier, execution, source_manifest)
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("target scorer checkout is dirty")
    if _git_revision(REPOSITORY_ROOT) != barrier["implementation_revision"]:
        raise ValueError("target scorer revision differs from prediction revision")
    barrier_rows = {str(row["take_id"]): row for row in barrier["predictions"]}
    objects = []
    for take_id in TARGET_TAKE_IDS:
        archive = validate_prediction_seal(
            prediction_root / take_id / "seal.json",
            execution,
            completion,
            parent,
            source_manifest,
        )
        barrier_row = barrier_rows[take_id]
        if file_sha256(archive.seal_path) != barrier_row["seal_file_sha256"]:
            raise ValueError(f"prediction seal changed after barrier: {take_id}")
        if file_sha256(archive.npz_path) != barrier_row["prediction_npz_sha256"]:
            raise ValueError(f"prediction archive changed after barrier: {take_id}")
        source_archive = _locate_archive(source_root, take_id)
        source = _source_row(source_manifest, take_id)
        if file_sha256(source_archive) != source["source_payload_sha256"]:
            raise ValueError(f"source archive changed after prediction: {take_id}")
        seal = _load_json(archive.seal_path)
        stage = _load_json(archive.seal_path.parent / seal["input_stage_manifest"])
        input_by_member = {str(row["archive_member"]): row for row in stage["inputs"]}
        target_mesh_records = []
        with ZipFile(source_archive) as source_zip:
            robot_member = f"{take_id}/robot_data.json"
            robot_bytes = source_zip.read(robot_member)
            if (
                hashlib.sha256(robot_bytes).hexdigest()
                != input_by_member[robot_member]["sha256"]
            ):
                raise ValueError(f"robot input changed after prediction: {take_id}")
            robot_records = json.loads(robot_bytes.decode("utf-8"))
            active_frames = [
                int(row["frame"])
                for row in robot_records
                if float(row["forces"][1]) > 3.0 and int(row["frame"]) >= 6
            ]

            def mesh_loader(
                frame: int,
                *,
                bound_take_id: str = take_id,
                records: list[dict[str, object]] = target_mesh_records,
            ) -> tuple[np.ndarray, np.ndarray]:
                member = f"{bound_take_id}/meshes/mesh-f{frame:05d}.obj"
                payload = source_zip.read(member)
                records.append(
                    {
                        "target_frame": frame,
                        "archive_member": member,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "byte_count": len(payload),
                    }
                )
                return _mesh_from_bytes(payload)

            row = score_one_prediction(
                archive,
                active_frames,
                mesh_loader,
                execution,
            )
        row["prediction_seal_file_sha256"] = file_sha256(archive.seal_path)
        row["source_archive_sha256"] = source["source_payload_sha256"]
        row["target_meshes"] = target_mesh_records
        objects.append(row)
    public13 = load_archived_public13_result(public13_path, parent)
    aggregate = evaluate_result(objects, public13, execution, completion)
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
        "completion_protocol_sha256": completion["protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "prediction_barrier_sha256": barrier["prediction_barrier_sha256"],
        "prediction_barrier_file_sha256": file_sha256(barrier_path),
        "prediction_barrier_passed": True,
        "target_mesh_access_before_barrier": False,
        "target_meshes_opened_after_complete_barrier": True,
        "future_observation_used_for_prediction": False,
        "parameter_selection_from_this_cohort": False,
        "replacement_used": False,
        "target_adaptation_used": False,
        "objects": objects,
        "aggregate": aggregate,
        "held_v8_accessed": False,
        "result_sha256": "",
    }
    result["result_sha256"] = result_sha256(result)
    validate_result(
        result,
        public13,
        execution,
        completion,
        barrier,
        source_manifest,
    )
    _write_json(output, result)
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
        ),
    )
    parser.add_argument(
        "--completion-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_missing5_scale_completion_v5.json"
        ),
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_official18_v4.json"
        ),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("stage")
    stage.add_argument("source_archive", type=Path)
    stage.add_argument("output_dir", type=Path)

    predict = commands.add_parser("predict")
    predict.add_argument("stage_dir", type=Path)
    predict.add_argument("output_dir", type=Path)
    predict.add_argument("--upstream-checkout", type=Path, required=True)
    predict.add_argument("--checkpoint-root", type=Path, required=True)

    barrier = commands.add_parser("barrier")
    barrier.add_argument("prediction_root", type=Path)
    barrier.add_argument("output", type=Path)

    score = commands.add_parser("score")
    score.add_argument("source_root", type=Path)
    score.add_argument("prediction_root", type=Path)
    score.add_argument("barrier", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument(
        "--public13-result",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "sota"
            / "pokeflex_action_robust_all18_v4_public13_retrospective"
            / "result.json"
        ),
    )

    args = parser.parse_args()
    common = (
        args.source_manifest,
        args.execution_protocol,
        args.completion_protocol,
        args.parent_protocol,
    )
    if args.command == "stage":
        _stage(args.source_archive, args.output_dir, *common)
    elif args.command == "predict":
        _predict(
            args.stage_dir,
            args.output_dir,
            *common,
            args.upstream_checkout,
            args.checkpoint_root,
        )
    elif args.command == "barrier":
        _barrier(args.prediction_root, args.output, *common)
    else:
        _score(
            args.source_root,
            args.prediction_root,
            args.barrier,
            args.output,
            *common,
            args.public13_result,
        )


if __name__ == "__main__":
    main()
