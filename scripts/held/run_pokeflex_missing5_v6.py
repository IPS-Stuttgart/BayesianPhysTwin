#!/usr/bin/env python3
"""Augment sealed PokeFlex V5 predictions and score V6 after its barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "held"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "remote"))

from run_pokeflex_missing5_v5 import (  # noqa: E402
    _git_clean,
    _git_revision,
    _load_json,
    _locate_archive,
    _mesh_from_bytes,
    _source_row,
    _validate_staged_files,
    _write_json,
)

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
    action_field_history_is_supported,
)
from bayesian_phystwin.pokeflex_missing5_completion_v5 import (  # noqa: E402
    validate_completion_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    TARGET_TAKE_IDS,
    PredictionArchiveV5,
    validate_input_stage,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    load_execution_protocol as load_v5_execution_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    validate_prediction_seal as validate_v5_prediction_seal,
)
from bayesian_phystwin.pokeflex_missing5_execution_v6 import (  # noqa: E402
    PREDICTION_SEAL_KIND,
    RESULT_KIND,
    apply_causal_scale_sequence,
    build_prediction_barrier,
    evaluate_result,
    file_sha256,
    load_execution_protocol,
    prediction_seal_sha256,
    result_sha256,
    score_one_prediction,
    validate_prediction_barrier,
    validate_prediction_seal,
    validate_result,
    verify_implementation_files,
)


def _load_protocols(
    execution_path: Path,
    v5_execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    model_path: Path,
    source_result_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    parent = load_official18_v4_protocol(parent_path)
    completion = _load_json(completion_path)
    validate_completion_protocol(completion)
    v5_execution = load_v5_execution_protocol(
        v5_execution_path,
        completion,
        parent,
    )
    execution, model, source_result = load_execution_protocol(
        execution_path,
        v5_execution,
        completion,
        parent,
        model_path,
        source_result_path,
    )
    verify_implementation_files(execution, REPOSITORY_ROOT)
    return execution, v5_execution, completion, parent, model, source_result


def _rms_field(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _field_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    first_flat = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= 1e-12:
        return None
    return float(np.dot(first_flat, second_flat) / denominator)


def _prefix_update_rows(
    take_root: Path,
    parent_archive: PredictionArchiveV5,
    stage: Mapping[str, object],
) -> list[dict[str, object]]:
    from run_pokeflex_bayesian_registration_smoke import _view_points
    from run_pokeflex_checkpoint_registration_independent_depth import (
        _load_official_template,
    )

    robot_records = json.loads(
        (take_root / "robot_data.json").read_text(encoding="utf-8")
    )
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    template_path = (
        take_root / str(stage["authorized_template_member"]).split("/", maxsplit=1)[1]
    )
    template_vertices, _, _ = _load_official_template(template_path)
    frames = np.asarray(parent_archive.target_frames, dtype=np.int64)
    baseline = np.asarray(parent_archive.baseline_vertices_m, dtype=np.float64)
    supported_parent = np.asarray(parent_archive.update_supported, dtype=np.bool_)
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    config = PokeFlexBayesianRegistrationConfig(residual_geometry="point_to_point")
    rows = []
    for target_index, frame_value in enumerate(frames):
        target_frame = int(frame_value)
        source_frame = target_frame - 1
        if source_frame not in frame_to_index:
            row: dict[str, object] = {
                "target_frame": target_frame,
                "source_frame": source_frame,
                "accepted": False,
                "registration_reason": "no-five-frame-source-prior",
                "action_supported": False,
                "robot_history_supported": False,
                "rms_update_m": 0.0,
                "prior_motion_rms_m": 0.0,
                "correction_to_prior_motion_ratio": 0.0,
                "correction_prior_motion_cosine": None,
                "association_count": 0,
            }
            supported = False
        else:
            source_index = frame_to_index[source_frame]
            source_prior = baseline[source_index]
            target_prior = baseline[target_index]
            views = tuple(
                _view_points(take_root, source_frame, camera, template_vertices)
                for camera in (0, 1)
            )
            update = register_pokeflex_graph_posterior(
                source_prior,
                views,
                action_supported=(
                    float(robot_by_frame[source_frame]["forces"][1]) > 3.0
                ),
                prior_faces=np.asarray(parent_archive.faces, dtype=np.int64),
                config=config,
            )
            source_forces = np.asarray(
                robot_by_frame.get(source_frame, {}).get("forces"),
                dtype=np.float64,
            )
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
            supported = bool(
                update.accepted and action_supported and robot_history_supported
            )
            correction = update.posterior_vertices_m - source_prior
            prior_motion = target_prior - source_prior
            prior_motion_rms = _rms_field(prior_motion)
            row = {
                "target_frame": target_frame,
                "source_frame": source_frame,
                "accepted": bool(update.accepted),
                "registration_reason": update.reason,
                "action_supported": action_supported,
                "robot_history_supported": robot_history_supported,
                "rms_update_m": float(update.diagnostics.get("rms_update_m", 0.0)),
                "prior_motion_rms_m": prior_motion_rms,
                "correction_to_prior_motion_ratio": (
                    _rms_field(correction) / max(prior_motion_rms, 1e-12)
                ),
                "correction_prior_motion_cosine": _field_cosine(
                    correction,
                    prior_motion,
                ),
                "association_count": int(
                    update.diagnostics.get("association_count", 0)
                ),
            }
        if supported != bool(supported_parent[target_index]):
            raise ValueError(
                f"replayed prefix support differs from sealed V5: {target_frame}"
            )
        rows.append(row)
    return rows


def _augment(
    stage_dir: Path,
    parent_v5_prediction_dir: Path,
    output_dir: Path,
    source_manifest_path: Path,
    execution_path: Path,
    v5_execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    model_path: Path,
    source_result_path: Path,
) -> None:
    (
        execution,
        v5_execution,
        completion,
        parent,
        model,
        source_result,
    ) = _load_protocols(
        execution_path,
        v5_execution_path,
        completion_path,
        parent_path,
        model_path,
        source_result_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("V6 prediction implementation checkout is dirty")
    revision = _git_revision(REPOSITORY_ROOT)
    parent_seal_path = parent_v5_prediction_dir.resolve() / "seal.json"
    parent_archive = validate_v5_prediction_seal(
        parent_seal_path,
        v5_execution,
        completion,
        parent,
        source_manifest,
    )
    if parent_archive.implementation_revision != revision:
        raise ValueError("V5 and V6 prediction revisions differ")
    stage_dir = stage_dir.resolve()
    stage_path = stage_dir / "stage.json"
    stage = _load_json(stage_path)
    stage_validation = validate_input_stage(
        stage,
        v5_execution,
        completion,
        parent,
        source_manifest,
    )
    if stage_validation["take_id"] != parent_archive.take_id:
        raise ValueError("V5 stage and prediction takes differ")
    copied_stage_path = parent_archive.seal_path.parent / "input_stage_manifest.json"
    if copied_stage_path.read_bytes() != stage_path.read_bytes():
        raise ValueError("V5 prediction and supplied input stage differ")
    take_root = _validate_staged_files(stage_dir, stage)
    update_rows = _prefix_update_rows(take_root, parent_archive, stage)
    arrays, decisions = apply_causal_scale_sequence(
        model,
        object_name=parent_archive.take_id.rpartition("_T")[0],
        baseline_vertices_m=parent_archive.baseline_vertices_m,
        v5_vertices_m=parent_archive.v5_vertices_m,
        target_frames=parent_archive.target_frames,
        update_supported=parent_archive.update_supported,
        update_rows=update_rows,
    )
    unsupported = ~parent_archive.update_supported
    rejected = parent_archive.update_supported & ~arrays["candidate_admitted"]
    unsupported_mismatch = int(
        np.sum(
            np.any(
                arrays["v6_vertices_m"][unsupported].view(np.uint64)
                != parent_archive.baseline_vertices_m[unsupported].view(np.uint64),
                axis=(1, 2),
            )
        )
    )
    rejected_mismatch = int(
        np.sum(
            np.any(
                arrays["v6_vertices_m"][rejected].view(np.uint64)
                != parent_archive.v5_vertices_m[rejected].view(np.uint64),
                axis=(1, 2),
            )
        )
    )
    if unsupported_mismatch or rejected_mismatch:
        raise AssertionError("V6 exact fallback changed bytes")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"V6 prediction output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        npz_path = output_dir / "prediction.npz"
        np.savez_compressed(
            npz_path,
            v6_vertices_m=arrays["v6_vertices_m"],
            target_frames=arrays["target_frames"],
            selected_scale=arrays["selected_scale"],
            candidate_admitted=arrays["candidate_admitted"],
            predicted_lower_gain_mm=arrays["predicted_lower_gain_mm"],
            minimum_source_distance=arrays["minimum_source_distance"],
            support_radius=arrays["support_radius"],
        )
        seal: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": PREDICTION_SEAL_KIND,
            "execution_protocol_sha256": execution["execution_protocol_sha256"],
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "take_id": parent_archive.take_id,
            "object_name": parent_archive.take_id.rpartition("_T")[0],
            "implementation_revision": revision,
            "implementation_clean": True,
            "parent_v5_execution_protocol_sha256": v5_execution[
                "execution_protocol_sha256"
            ],
            "parent_v5_seal_sha256": _load_json(parent_seal_path)["seal_sha256"],
            "parent_v5_seal_file_sha256": file_sha256(parent_seal_path),
            "parent_v5_prediction_npz_sha256": file_sha256(parent_archive.npz_path),
            "causal_scale_model_sha256": execution["causal_scale_model_sha256"],
            "causal_scale_model_file_sha256": file_sha256(model_path),
            "source_result_sha256": execution["source_result_sha256"],
            "source_result_file_sha256": file_sha256(source_result_path),
            "input_stage_sha256": stage["input_stage_sha256"],
            "prediction_npz": npz_path.name,
            "prediction_npz_sha256": file_sha256(npz_path),
            "predicted_frame_count": len(arrays["target_frames"]),
            "candidate_admission_count": int(np.sum(arrays["candidate_admitted"])),
            "unsupported_fallback_mismatch_count": unsupported_mismatch,
            "rejected_fallback_mismatch_count": rejected_mismatch,
            "future_observation_used": False,
            "future_target_mesh_read": False,
            "future_target_mesh_read_count": 0,
            "target_metric_computed": False,
            "causal_history": (
                "every V6 decision uses only the V5 f-5 through f-1 stage and "
                "the realized prefix update at f-1"
            ),
            "decisions": decisions,
            "held_v8_accessed": False,
            "seal_sha256": "",
        }
        seal["seal_sha256"] = prediction_seal_sha256(seal)
        _write_json(output_dir / "seal.json", seal)
        validate_prediction_seal(
            output_dir / "seal.json",
            parent_seal_path,
            execution,
            v5_execution,
            completion,
            parent,
            source_manifest,
            model,
            source_result,
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "take_id": parent_archive.take_id,
                "prediction_npz_sha256": seal["prediction_npz_sha256"],
                "seal_sha256": seal["seal_sha256"],
                "candidate_admission_count": seal["candidate_admission_count"],
                "future_target_mesh_read": False,
            },
            indent=2,
        )
    )


def _barrier(
    parent_v5_prediction_root: Path,
    prediction_root: Path,
    output: Path,
    source_manifest_path: Path,
    execution_path: Path,
    v5_execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    model_path: Path,
    source_result_path: Path,
) -> None:
    (
        execution,
        v5_execution,
        completion,
        parent,
        model,
        source_result,
    ) = _load_protocols(
        execution_path,
        v5_execution_path,
        completion_path,
        parent_path,
        model_path,
        source_result_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    seals = [prediction_root / take_id / "seal.json" for take_id in TARGET_TAKE_IDS]
    parent_seals = [
        parent_v5_prediction_root / take_id / "seal.json" for take_id in TARGET_TAKE_IDS
    ]
    payload = build_prediction_barrier(
        seals,
        parent_seals,
        execution,
        v5_execution,
        completion,
        parent,
        source_manifest,
        model,
        source_result,
    )
    _write_json(output, payload)
    print(json.dumps(payload, indent=2))


def _score(
    source_root: Path,
    parent_v5_prediction_root: Path,
    prediction_root: Path,
    barrier_path: Path,
    output: Path,
    source_manifest_path: Path,
    execution_path: Path,
    v5_execution_path: Path,
    completion_path: Path,
    parent_path: Path,
    model_path: Path,
    source_result_path: Path,
    public13_path: Path,
) -> None:
    (
        execution,
        v5_execution,
        completion,
        parent,
        model,
        source_result,
    ) = _load_protocols(
        execution_path,
        v5_execution_path,
        completion_path,
        parent_path,
        model_path,
        source_result_path,
    )
    source_manifest = _load_json(source_manifest_path)
    validate_author_source_manifest(source_manifest, parent)
    barrier = _load_json(barrier_path)
    validate_prediction_barrier(barrier, execution, source_manifest)
    if not _git_clean(REPOSITORY_ROOT):
        raise ValueError("V6 target scorer checkout is dirty")
    if _git_revision(REPOSITORY_ROOT) != barrier["implementation_revision"]:
        raise ValueError("V6 scorer revision differs from prediction revision")
    barrier_rows = {str(row["take_id"]): row for row in barrier["predictions"]}
    objects = []
    for take_id in TARGET_TAKE_IDS:
        parent_seal_path = parent_v5_prediction_root / take_id / "seal.json"
        archive = validate_prediction_seal(
            prediction_root / take_id / "seal.json",
            parent_seal_path,
            execution,
            v5_execution,
            completion,
            parent,
            source_manifest,
            model,
            source_result,
        )
        barrier_row = barrier_rows[take_id]
        if file_sha256(archive.seal_path) != barrier_row["seal_file_sha256"]:
            raise ValueError(f"V6 seal changed after barrier: {take_id}")
        if file_sha256(archive.npz_path) != barrier_row["prediction_npz_sha256"]:
            raise ValueError(f"V6 prediction changed after barrier: {take_id}")
        source_archive = _locate_archive(source_root, take_id)
        source = _source_row(source_manifest, take_id)
        if file_sha256(source_archive) != source["source_payload_sha256"]:
            raise ValueError(f"source archive changed after V6 prediction: {take_id}")
        parent_seal = _load_json(parent_seal_path)
        stage = _load_json(
            parent_seal_path.parent / str(parent_seal["input_stage_manifest"])
        )
        input_by_member = {str(row["archive_member"]): row for row in stage["inputs"]}
        target_mesh_records: list[dict[str, object]] = []
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
                v5_execution,
            )
        row["prediction_seal_file_sha256"] = file_sha256(archive.seal_path)
        row["parent_v5_prediction_seal_file_sha256"] = file_sha256(
            archive.parent_v5.seal_path
        )
        row["source_archive_sha256"] = source["source_payload_sha256"]
        row["target_meshes"] = target_mesh_records
        objects.append(row)
    public13 = load_archived_public13_result(public13_path, parent)
    aggregate = evaluate_result(objects, public13, execution)
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
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
            REPOSITORY_ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v6.json"
        ),
    )
    parser.add_argument(
        "--v5-execution-protocol",
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
    parser.add_argument(
        "--causal-scale-model",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_missing5_causal_scale_v6.json"
        ),
    )
    parser.add_argument(
        "--source-result",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "sota"
            / "pokeflex_missing5_causal_scale_v6"
            / "source_result.json"
        ),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    augment = commands.add_parser("augment")
    augment.add_argument("stage_dir", type=Path)
    augment.add_argument("parent_v5_prediction_dir", type=Path)
    augment.add_argument("output_dir", type=Path)

    barrier = commands.add_parser("barrier")
    barrier.add_argument("parent_v5_prediction_root", type=Path)
    barrier.add_argument("prediction_root", type=Path)
    barrier.add_argument("output", type=Path)

    score = commands.add_parser("score")
    score.add_argument("source_root", type=Path)
    score.add_argument("parent_v5_prediction_root", type=Path)
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
        args.v5_execution_protocol,
        args.completion_protocol,
        args.parent_protocol,
        args.causal_scale_model,
        args.source_result,
    )
    if args.command == "augment":
        _augment(
            args.stage_dir,
            args.parent_v5_prediction_dir,
            args.output_dir,
            *common,
        )
    elif args.command == "barrier":
        _barrier(
            args.parent_v5_prediction_root,
            args.prediction_root,
            args.output,
            *common,
        )
    else:
        _score(
            args.source_root,
            args.parent_v5_prediction_root,
            args.prediction_root,
            args.barrier,
            args.output,
            *common,
            args.public13_result,
        )


if __name__ == "__main__":
    main()
