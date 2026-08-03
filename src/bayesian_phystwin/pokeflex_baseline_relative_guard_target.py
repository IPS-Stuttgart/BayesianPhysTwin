"""Prospective custody for the guarded PokeFlex state update."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_baseline_relative_guard import (
    FEATURE_NAMES,
    certificate_from_payload,
)
from .pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    PUBLISHED_KINECT_CD_UL1_MM,
    SELECTED_ARM,
    SOURCE_PROTOCOL_SHA256,
    SOURCE_RESULT_SHA256,
    UPSTREAM_COMMIT,
    cd_ul1_mm,
    file_sha256,
    official_volumetric_jaccard,
    paired_object_bootstrap_upper_difference,
    surface_sample,
)

PROTOCOL_ID = "pokeflex-baseline-relative-guard-public-paired-v2"
PROTOCOL_KIND = "PokeFlexBaselineRelativeGuardTargetProtocol"
PROTOCOL_FILE_SHA256 = (
    "355e6a8850781c90f92e5a74c673f00d4d71e46bcde3f7728330c6fb4bbe3985"
)
SELECTION_MANIFEST_SHA256 = (
    "418b80936fe98cc1007c7884ebcc8c10ed011a7487aa63e681b6a6e59c854d5f"
)
SELECTION_FILE_SHA256 = (
    "53d155d6818bdcb17e5fc18f5ebf8e544724410f306030647e10b0f21bf5c207"
)
DEVELOPMENT_EVALUATION_SHA256 = (
    "49007cc03f2ed10e59e3aa2588f2a5130b70d047e919d75200c2769143bf3c71"
)
DEVELOPMENT_COMMIT = "ece028ae8ce6cdfbd0c15af31482dffb6033edf7"
CERTIFICATE_SHA256 = (
    "5eeb2350900334d487db5630f3a39a186a2e388be41334349cbcff613cdeb8a5"
)
TARGET_TAKE_IDS = (
    "3dPrintedCylinder_T3",
    "3dPrintedPizza_T5",
    "3dPrintedPyramid_T1",
    "Beanbag_T4",
    "FoamCylinder_T7",
    "FoamHalfSphere_T4",
    "Pillow_T6",
    "PlushDice_T5",
    "PlushMoon_T3",
    "PlushTurtle_T1",
    "PlushVolleyball_T5",
    "Sponge_T3",
)
TARGET_OBJECTS = tuple(value.rpartition("_T")[0] for value in TARGET_TAKE_IDS)
MINIMUM_WIN_COUNT = 10
MINIMUM_SUPPORTED_OBJECT_COUNT = 10


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_payload_sha256(
    payload: Mapping[str, Any], *, digest_field: str
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def protocol_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="protocol_sha256")


def certificate_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def seal_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="seal_sha256")


def barrier_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="barrier_sha256")


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete lock before any selected target mesh is opened."""

    _require(payload.get("schema_version") == 1, "protocol schema changed")
    _require(payload.get("artifact_kind") == PROTOCOL_KIND, "protocol kind changed")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    _require(
        payload.get("protocol_sha256") == protocol_sha256(payload),
        "protocol checksum mismatch",
    )

    source = payload.get("source_gate")
    _require(isinstance(source, Mapping), "source gate is missing")
    _require(source.get("protocol_sha256") == SOURCE_PROTOCOL_SHA256, "source changed")
    _require(source.get("result_sha256") == SOURCE_RESULT_SHA256, "source result changed")
    _require(source.get("passed") is True, "source gate did not pass")
    _require(source.get("selected_arm") == SELECTED_ARM, "source arm changed")

    selection = payload.get("selection_audit")
    _require(isinstance(selection, Mapping), "selection audit is missing")
    _require(
        selection.get("manifest_sha256") == SELECTION_MANIFEST_SHA256,
        "selection manifest changed",
    )
    _require(
        selection.get("manifest_file_sha256") == SELECTION_FILE_SHA256,
        "selection file bytes changed",
    )
    _require(int(selection.get("public_take_count", -1)) == 116, "take inventory changed")
    _require(
        int(selection.get("referenced_take_count", -1)) == 73,
        "reference inventory changed",
    )
    _require(
        int(selection.get("eligible_object_count", -1)) == len(TARGET_OBJECTS),
        "eligible object count changed",
    )

    development = payload.get("development_guard")
    _require(isinstance(development, Mapping), "development guard is missing")
    _require(
        development.get("evaluation_sha256") == DEVELOPMENT_EVALUATION_SHA256,
        "development evaluation changed",
    )
    _require(development.get("git_commit") == DEVELOPMENT_COMMIT, "guard commit changed")
    certificate = development.get("certificate")
    _require(isinstance(certificate, Mapping), "guard certificate is missing")
    _require(
        certificate_sha256(certificate) == CERTIFICATE_SHA256,
        "guard certificate changed",
    )
    _require(
        development.get("certificate_sha256") == CERTIFICATE_SHA256,
        "guard certificate checksum field changed",
    )
    certificate_from_payload(certificate)
    _require(
        tuple(development.get("feature_names", ())) == FEATURE_NAMES,
        "guard feature schema changed",
    )

    cohort = payload.get("target_cohort")
    _require(isinstance(cohort, Mapping), "target cohort is missing")
    _require(tuple(cohort.get("take_ids", ())) == TARGET_TAKE_IDS, "target takes changed")
    _require(tuple(cohort.get("objects", ())) == TARGET_OBJECTS, "target objects changed")
    _require(cohort.get("replacement_allowed") is False, "replacement was enabled")

    method = payload.get("method")
    _require(isinstance(method, Mapping), "method lock is missing")
    _require(method.get("selected_arm") == SELECTED_ARM, "candidate arm changed")
    _require(method.get("field") == "action_local_state_relative_0.4", "field changed")
    _require(float(method.get("scale", -1.0)) == 0.125, "candidate scale changed")
    _require(
        method.get("guard_rule")
        == "admit iff in source support and calibrated upper regret is below zero",
        "guard rule changed",
    )
    _require(
        method.get("fallback") == "byte-identical released checkpoint",
        "fallback changed",
    )

    upstream = payload.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream lock is missing")
    _require(upstream.get("code_commit") == UPSTREAM_COMMIT, "upstream changed")
    _require(
        dict(upstream.get("checkpoint_sha256", {})) == CHECKPOINT_SHA256,
        "checkpoint bytes changed",
    )

    custody = payload.get("custody")
    _require(isinstance(custody, Mapping), "custody lock is missing")
    _require(
        custody.get("prediction_and_scoring_are_separate") is True,
        "prediction/scoring separation changed",
    )
    _require(
        int(custody.get("required_prediction_seal_count", -1)) == len(TARGET_TAKE_IDS),
        "prediction barrier count changed",
    )
    _require(
        custody.get("target_mesh_access_before_barrier") == "forbidden",
        "target custody weakened",
    )
    _require(
        custody.get("prediction_observation_history") == "f-5 through f-1",
        "causal history changed",
    )

    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation lock is missing")
    _require(evaluation.get("primary_metric") == "CD_UL1_mm", "metric changed")
    _require(int(evaluation.get("surface_sample_count", -1)) == 10000, "samples changed")
    _require(int(evaluation.get("surface_sample_seed", -1)) == 20260720, "seed changed")
    _require(
        evaluation.get("aggregation")
        == "equal scored frames within each take, then equal physical objects",
        "aggregation changed",
    )

    paired = payload.get("gates", {}).get("paired_transfer", {})
    _require(
        float(paired.get("relative_CD_UL1_improvement_above", -1.0)) == 0.0,
        "improvement gate changed",
    )
    _require(
        float(paired.get("bootstrap_upper_difference_mm_below", 1.0)) == 0.0,
        "bootstrap gate changed",
    )
    _require(
        float(paired.get("maximum_per_object_relative_regression", 1.0)) == 0.0,
        "regression gate changed",
    )
    _require(
        int(paired.get("minimum_object_win_count", -1)) == MINIMUM_WIN_COUNT,
        "win-count gate changed",
    )
    _require(
        int(paired.get("minimum_supported_object_count", -1))
        == MINIMUM_SUPPORTED_OBJECT_COUNT,
        "support gate changed",
    )
    _require(int(paired.get("bootstrap_replicates", -1)) == 20000, "bootstrap changed")
    _require(int(paired.get("bootstrap_seed", -1)) == 20260720, "bootstrap seed changed")
    _require(
        float(paired.get("bootstrap_upper_quantile", -1.0)) == 0.975,
        "bootstrap quantile changed",
    )
    return {
        "passed": True,
        "protocol_sha256": payload["protocol_sha256"],
        "target_take_ids": TARGET_TAKE_IDS,
    }


def load_protocol(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(file_sha256(path) == PROTOCOL_FILE_SHA256, "protocol file bytes changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(payload)
    return payload


@dataclass(frozen=True)
class PredictionArchive:
    take_id: str
    seal_path: Path
    npz_path: Path
    implementation_revision: str
    baseline_vertices_m: np.ndarray
    candidate_vertices_m: np.ndarray
    faces: np.ndarray
    target_frames: np.ndarray
    update_supported: np.ndarray


def validate_prediction_seal(
    seal_path: Path, protocol: Mapping[str, Any]
) -> PredictionArchive:
    validate_protocol(protocol)
    seal_path = Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal.get("schema_version") == 1, "prediction schema changed")
    _require(
        seal.get("artifact_kind") == "PokeFlexBaselineRelativeGuardPredictionSeal",
        "prediction kind changed",
    )
    _require(seal.get("seal_sha256") == seal_sha256(seal), "seal checksum mismatch")
    _require(seal.get("protocol_sha256") == protocol["protocol_sha256"], "protocol changed")
    _require(
        seal.get("certificate_sha256") == CERTIFICATE_SHA256,
        "prediction certificate changed",
    )
    _require(seal.get("future_mesh_read") is False, "prediction read future mesh")
    _require(int(seal.get("future_mesh_read_count", -1)) == 0, "future access recorded")
    _require(seal.get("implementation_clean") is True, "implementation was dirty")
    revision = str(seal.get("implementation_revision", ""))
    _require(len(revision) == 40, "prediction revision is invalid")
    take_id = str(seal.get("take_id", ""))
    _require(take_id in TARGET_TAKE_IDS, "prediction take is outside cohort")
    _require(
        seal.get("object_name") == take_id.rpartition("_T")[0],
        "prediction object changed",
    )
    _require(dict(seal.get("checkpoint_sha256", {})) == CHECKPOINT_SHA256, "checkpoint changed")
    _require(seal.get("upstream_commit") == UPSTREAM_COMMIT, "upstream changed")

    npz_path = seal_path.parent / str(seal.get("prediction_npz", ""))
    _require(npz_path.is_file(), "prediction archive is missing")
    _require(file_sha256(npz_path) == seal.get("prediction_npz_sha256"), "archive changed")
    required = {
        "baseline_vertices_m",
        "candidate_vertices_m",
        "faces",
        "target_frames",
        "source_frames",
        "history_start_frames",
        "history_end_frames",
        "raw_update_supported",
        "guard_in_source_support",
        "guard_accepted",
        "update_accepted",
        "action_supported",
        "robot_history_supported",
        "association_count",
        "raw_correction_rms_m",
        "correction_field_rms_m",
        "guard_predicted_regret_mm",
        "guard_upper_regret_mm",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(set(archive.files) == required, "prediction array schema changed")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    baseline = np.asarray(arrays["baseline_vertices_m"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_vertices_m"], dtype=np.float64)
    faces = np.asarray(arrays["faces"])
    frames = np.asarray(arrays["target_frames"])
    raw_supported = np.asarray(arrays["raw_update_supported"])
    in_support = np.asarray(arrays["guard_in_source_support"])
    accepted = np.asarray(arrays["guard_accepted"])
    predicted_regret = np.asarray(arrays["guard_predicted_regret_mm"])
    upper_regret = np.asarray(arrays["guard_upper_regret_mm"])
    _require(baseline.ndim == 3 and baseline.shape[-1] == 3, "baseline shape changed")
    _require(candidate.shape == baseline.shape, "candidate shape changed")
    _require(np.all(np.isfinite(baseline)), "baseline is non-finite")
    _require(np.all(np.isfinite(candidate)), "candidate is non-finite")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "faces shape changed")
    _require(np.issubdtype(faces.dtype, np.integer), "faces are not integer")
    _require(np.array_equal(arrays["source_frames"], frames - 1), "source frame changed")
    _require(np.array_equal(arrays["history_start_frames"], frames - 5), "history start changed")
    _require(np.array_equal(arrays["history_end_frames"], frames - 1), "history end changed")
    _require(len(frames) == len(baseline), "frame count changed")
    _require(int(frames[0]) == 6 and np.all(np.diff(frames) == 1), "frames are not contiguous")
    for value, name in (
        (raw_supported, "raw support"),
        (in_support, "source support"),
        (accepted, "guard acceptance"),
    ):
        _require(value.shape == frames.shape and value.dtype == np.bool_, f"{name} changed")
    _require(not np.any(accepted & ~raw_supported), "guard accepted unavailable update")
    _require(not np.any(accepted & ~in_support), "guard accepted out-of-support update")
    _require(predicted_regret.shape == frames.shape, "predicted regret shape changed")
    _require(upper_regret.shape == frames.shape, "upper regret shape changed")
    _require(
        np.all(np.isfinite(predicted_regret[raw_supported])),
        "available update has non-finite predicted regret",
    )
    _require(
        np.all(np.isnan(predicted_regret[~raw_supported])),
        "unavailable update has a predicted regret",
    )
    _require(
        np.all(np.isfinite(upper_regret[in_support])),
        "in-support update has non-finite upper regret",
    )
    _require(
        np.all(np.isnan(upper_regret[~in_support])),
        "out-of-support update has an upper regret",
    )
    _require(
        np.all(upper_regret[accepted] < 0.0),
        "guard accepted nonnegative upper regret",
    )
    _require(np.array_equal(candidate[~accepted], baseline[~accepted]), "fallback is not exact")
    _require(int(seal.get("fallback_mismatch_count", -1)) == 0, "fallback mismatch recorded")
    _require(int(seal.get("predicted_frame_count", -1)) == len(frames), "seal frame count changed")
    _require(int(seal.get("guard_accepted_frame_count", -1)) == int(np.sum(accepted)), "acceptance count changed")
    return PredictionArchive(
        take_id=take_id,
        seal_path=seal_path,
        npz_path=npz_path,
        implementation_revision=revision,
        baseline_vertices_m=baseline,
        candidate_vertices_m=candidate,
        faces=np.asarray(faces, dtype=np.int64),
        target_frames=np.asarray(frames, dtype=np.int64),
        update_supported=np.asarray(accepted, dtype=np.bool_),
    )


def build_prediction_barrier(
    seal_paths: Sequence[Path], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    validate_protocol(protocol)
    archives = [validate_prediction_seal(path, protocol) for path in seal_paths]
    _require(len(archives) == len(TARGET_TAKE_IDS), "prediction barrier is incomplete")
    by_take = {archive.take_id: archive for archive in archives}
    _require(len(by_take) == len(archives), "duplicate prediction seal")
    _require(tuple(sorted(by_take)) == tuple(sorted(TARGET_TAKE_IDS)), "seal cohort changed")
    revisions = {archive.implementation_revision for archive in archives}
    _require(len(revisions) == 1, "prediction revisions differ")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardPredictionBarrier",
        "protocol_sha256": protocol["protocol_sha256"],
        "certificate_sha256": CERTIFICATE_SHA256,
        "implementation_revision": next(iter(revisions)),
        "prediction_count": len(archives),
        "target_take_ids": list(TARGET_TAKE_IDS),
        "predictions": [
            {
                "take_id": take_id,
                "seal_path": str(by_take[take_id].seal_path),
                "seal_file_sha256": file_sha256(by_take[take_id].seal_path),
                "prediction_npz_sha256": file_sha256(by_take[take_id].npz_path),
            }
            for take_id in TARGET_TAKE_IDS
        ],
        "target_mesh_opened": False,
        "scoring_authorized": True,
    }
    payload["barrier_sha256"] = barrier_sha256(payload)
    return payload


def validate_prediction_barrier(
    barrier: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    validate_protocol(protocol)
    _require(
        barrier.get("artifact_kind")
        == "PokeFlexBaselineRelativeGuardPredictionBarrier",
        "barrier kind changed",
    )
    _require(barrier.get("barrier_sha256") == barrier_sha256(barrier), "barrier checksum changed")
    _require(barrier.get("protocol_sha256") == protocol["protocol_sha256"], "barrier protocol changed")
    _require(barrier.get("certificate_sha256") == CERTIFICATE_SHA256, "barrier certificate changed")
    _require(int(barrier.get("prediction_count", -1)) == len(TARGET_TAKE_IDS), "barrier count changed")
    _require(tuple(barrier.get("target_take_ids", ())) == TARGET_TAKE_IDS, "barrier cohort changed")
    _require(barrier.get("target_mesh_opened") is False, "barrier reports target access")
    _require(barrier.get("scoring_authorized") is True, "barrier did not authorize scoring")
    predictions = barrier.get("predictions")
    _require(isinstance(predictions, list), "barrier predictions are missing")
    _require(tuple(row.get("take_id") for row in predictions) == TARGET_TAKE_IDS, "barrier order changed")
    return {"passed": True, "barrier_sha256": barrier["barrier_sha256"]}


def score_one_prediction(
    archive: PredictionArchive,
    active_frames: Sequence[int],
    mesh_loader: Callable[[int], tuple[np.ndarray, np.ndarray]],
    protocol: Mapping[str, Any],
    *,
    jaccard: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
    | None = None,
) -> dict[str, Any]:
    validate_protocol(protocol)
    active = tuple(sorted(int(frame) for frame in active_frames if int(frame) >= 6))
    _require(bool(active), "target take has no scored action frames")
    frame_to_index = {int(frame): index for index, frame in enumerate(archive.target_frames)}
    _require(all(frame in frame_to_index for frame in active), "prediction frame is missing")
    evaluation = protocol["evaluation"]
    count = int(evaluation["surface_sample_count"])
    seed = int(evaluation["surface_sample_seed"])
    jaccard_function = jaccard or (
        lambda pv, pf, tv, tf: official_volumetric_jaccard(
            pv,
            pf,
            tv,
            tf,
            engine=str(evaluation["jaccard_boolean_backend"]),
            process=evaluation.get("jaccard_mesh_processing") == "trimesh_default",
        )
    )
    rows = []
    for frame in active:
        index = frame_to_index[frame]
        target_vertices, target_faces = mesh_loader(frame)
        target_sample = surface_sample(target_vertices, target_faces, count, seed + frame)
        baseline_sample = surface_sample(archive.baseline_vertices_m[index], archive.faces, count, seed + frame)
        candidate_sample = surface_sample(archive.candidate_vertices_m[index], archive.faces, count, seed + frame)
        row: dict[str, Any] = {
            "target_frame": frame,
            "update_supported": bool(archive.update_supported[index]),
            "baseline_CD_UL1_mm": cd_ul1_mm(baseline_sample, target_sample),
            "candidate_CD_UL1_mm": cd_ul1_mm(candidate_sample, target_sample),
            "baseline_jaccard": None,
            "candidate_jaccard": None,
            "baseline_jaccard_error": None,
            "candidate_jaccard_error": None,
        }
        for prefix, vertices in (
            ("baseline", archive.baseline_vertices_m[index]),
            ("candidate", archive.candidate_vertices_m[index]),
        ):
            try:
                row[f"{prefix}_jaccard"] = jaccard_function(
                    vertices, archive.faces, target_vertices, target_faces
                )
            except Exception as error:
                row[f"{prefix}_jaccard_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    baseline_errors = np.asarray([row["baseline_CD_UL1_mm"] for row in rows])
    candidate_errors = np.asarray([row["candidate_CD_UL1_mm"] for row in rows])
    baseline_mean = float(np.mean(baseline_errors))
    candidate_mean = float(np.mean(candidate_errors))
    _require(baseline_mean > 0.0, "baseline target error is zero")
    baseline_jaccard = [row["baseline_jaccard"] for row in rows if row["baseline_jaccard"] is not None]
    candidate_jaccard = [row["candidate_jaccard"] for row in rows if row["candidate_jaccard"] is not None]
    return {
        "object_name": archive.take_id.rpartition("_T")[0],
        "take_id": archive.take_id,
        "scored_frame_count": len(rows),
        "supported_frame_count": int(sum(row["update_supported"] for row in rows)),
        "baseline_mean_CD_UL1_mm": baseline_mean,
        "candidate_mean_CD_UL1_mm": candidate_mean,
        "relative_CD_UL1_improvement": float((baseline_mean - candidate_mean) / baseline_mean),
        "baseline_jaccard_valid_count": len(baseline_jaccard),
        "candidate_jaccard_valid_count": len(candidate_jaccard),
        "baseline_mean_jaccard_valid": float(np.mean(baseline_jaccard)) if baseline_jaccard else None,
        "candidate_mean_jaccard_valid": float(np.mean(candidate_jaccard)) if candidate_jaccard else None,
        "frames": rows,
    }


def evaluate_target_metrics(
    per_take: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    validate_protocol(protocol)
    _require(len(per_take) == len(TARGET_TAKE_IDS), "target result is incomplete")
    by_take = {str(row["take_id"]): row for row in per_take}
    _require(len(by_take) == len(per_take), "target result contains duplicates")
    _require(tuple(sorted(by_take)) == tuple(sorted(TARGET_TAKE_IDS)), "result cohort changed")
    ordered = [by_take[take_id] for take_id in TARGET_TAKE_IDS]
    _require(tuple(str(row["object_name"]) for row in ordered) == TARGET_OBJECTS, "result objects changed")
    baseline = np.asarray([row["baseline_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64)
    candidate = np.asarray([row["candidate_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64)
    _require(np.all(np.isfinite(baseline)) and np.all(baseline > 0.0), "baseline scores are invalid")
    _require(np.all(np.isfinite(candidate)), "candidate scores are invalid")
    difference = candidate - baseline
    relative_by_object = -difference / baseline
    tolerance = 1e-12
    wins = int(np.sum(difference < -tolerance))
    losses = int(np.sum(difference > tolerance))
    ties = len(difference) - wins - losses
    supported_objects = int(sum(int(row["supported_frame_count"]) > 0 for row in ordered))
    paired = protocol["gates"]["paired_transfer"]
    upper = paired_object_bootstrap_upper_difference(
        difference,
        replicates=int(paired["bootstrap_replicates"]),
        seed=int(paired["bootstrap_seed"]),
        upper_quantile=float(paired["bootstrap_upper_quantile"]),
    )
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    improvement = float((baseline_mean - candidate_mean) / baseline_mean)
    paired_pass = bool(
        improvement > float(paired["relative_CD_UL1_improvement_above"])
        and upper < float(paired["bootstrap_upper_difference_mm_below"])
        and float(np.min(relative_by_object))
        >= -float(paired["maximum_per_object_relative_regression"])
        and wins >= int(paired["minimum_object_win_count"])
        and supported_objects >= int(paired["minimum_supported_object_count"])
    )
    total_frames = sum(int(row["scored_frame_count"]) for row in ordered)
    valid_jaccard_count = sum(int(row["candidate_jaccard_valid_count"]) for row in ordered)
    valid_jaccard = [
        float(row["candidate_mean_jaccard_valid"])
        for row in ordered
        if row["candidate_mean_jaccard_valid"] is not None
    ]
    return {
        "target_take_count": len(TARGET_TAKE_IDS),
        "baseline_object_balanced_CD_UL1_mm": baseline_mean,
        "candidate_object_balanced_CD_UL1_mm": candidate_mean,
        "object_balanced_relative_CD_UL1_improvement": improvement,
        "object_win_count": wins,
        "object_tie_count": ties,
        "object_loss_count": losses,
        "minimum_per_object_relative_improvement": float(np.min(relative_by_object)),
        "supported_object_count": supported_objects,
        "bootstrap_upper_candidate_minus_baseline_CD_UL1_mm": upper,
        "candidate_object_balanced_jaccard_valid": float(np.mean(valid_jaccard)) if valid_jaccard else None,
        "candidate_jaccard_valid_fraction": float(valid_jaccard_count / total_frames),
        "published_kinect_CD_UL1_mm_context_only": PUBLISHED_KINECT_CD_UL1_MM,
        "candidate_below_published_context_reference": bool(candidate_mean < PUBLISHED_KINECT_CD_UL1_MM),
        "published_reference_is_gating": False,
        "paired_transfer_passed": paired_pass,
        "all_target_gates_passed": paired_pass,
    }
