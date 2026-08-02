"""Prospective target custody for conservative PokeFlex shrinkage.

The target workflow is deliberately split into prediction, an all-case barrier,
and scoring.  Prediction archives contain no target-frame mesh information.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TARGET_OBJECTS = (
    "3dPrintedCylinder",
    "3dPrintedPizza",
    "FoamHalfSphere",
    "Pillow",
    "PlushDice",
    "PlushTurtle",
    "PlushVolleyball",
    "Sponge",
)
TARGET_TAKE_IDS = tuple(f"{name}_T2" for name in TARGET_OBJECTS)
TARGET_PROTOCOL_V1 = "pokeflex-conservative-shrinkage-target-v1"
TARGET_PROTOCOL_V2 = "pokeflex-conservative-shrinkage-target-v2"
TARGET_PROTOCOL_OFFICIAL18_V1 = "pokeflex-conservative-shrinkage-official18-v1"
OFFICIAL18_TARGET_TAKE_IDS = (
    "MemoryFoam_T2",
    "PlushVolleyball_T4",
    "FoamHalfSphere_T3",
    "3dPrintedBunny_T1",
    "3dPrintedPyramid_T6",
    "FoamDice_T3",
    "PlushMoon_T1",
    "PlushOctopus_T6",
    "PlushDice_T8",
    "PlushTurtle_T3",
    "Pillow_T8",
    "3dPrintedCylinder_T7",
    "Beanbag_T6",
    "3dPrintedHeart_T14",
    "FoamCylinder_T1",
    "ToiletPaperRoll_T1",
    "Sponge_T10",
    "3dPrintedPizza_T13",
)
OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS = (
    "FoamDice_T3",
    "PlushOctopus_T6",
    "ToiletPaperRoll_T1",
)
OFFICIAL18_PROSPECTIVE_TAKE_IDS = tuple(
    take_id
    for take_id in OFFICIAL18_TARGET_TAKE_IDS
    if take_id not in OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS
)
OFFICIAL_EVALUATOR_SHA256 = (
    "ea1854ba5224b8aec2e8ba6b80fb762eba7314b925e87ca7775d810003615b60"
)
PUBLISHED_KINECT_CD_UL1_MM = 6.498
PUBLISHED_KINECT_JACCARD = 0.820
SOURCE_PROTOCOL_SHA256 = (
    "73b69d3efae27d5afe511bc795c3e270546722e410aaca698db5afcc90ed23e9"
)
SOURCE_RESULT_SHA256 = (
    "0075c331fc23ffadb2e9ebdd4b58093c76d25ce39c2bcf33e84d80d50a338bda"
)
SELECTED_ARM = "checkpoint_action_local_state_relative_0.4_residual_scale_0.125"
UPSTREAM_COMMIT = "aaa8726072834a95bbe97e1a113588968c36e185"
CHECKPOINT_SHA256 = {
    "attention_model.pth": (
        "51181c22d7ad9fcc194a48411fda64759bdfb491c73abfa94f63d0a7167284fe"
    ),
    "decoder.pth": ("34a29ab89912ffdd0ea2a4436bcaca0e843d2c51a19f77c88844702b596b46cf"),
    "pointcloud_encoder.pth": (
        "3053f0656e4ca61645aa194e2d33540f68953efe9fd0cbab062ec561c405609b"
    ),
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    """Hash one immutable artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any], *, digest_field: str) -> str:
    """Hash canonical JSON after removing its self-referential digest field."""

    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def target_protocol_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="protocol_sha256")


def _take_identity(take_id: str) -> tuple[str, str]:
    object_name, separator, take_number = take_id.rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid target take id")
    return object_name, f"T{take_number}"


def target_take_ids_for_protocol(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the immutable cohort associated with a target protocol."""

    if protocol.get("protocol_id") == TARGET_PROTOCOL_OFFICIAL18_V1:
        return OFFICIAL18_TARGET_TAKE_IDS
    return TARGET_TAKE_IDS


def protocol_requires_robot_history(protocol_id: str) -> bool:
    """Return whether prediction custody includes explicit robot-history support."""

    return protocol_id in {TARGET_PROTOCOL_V2, TARGET_PROTOCOL_OFFICIAL18_V1}


def action_field_history_is_supported(
    robot_by_frame: Mapping[int, Mapping[str, Any]],
    source_frame: int,
) -> bool:
    """Return whether the frozen action-local field has all required robot poses."""

    for frame in range(max(1, source_frame - 3), source_frame + 1):
        record = robot_by_frame.get(frame)
        if record is None:
            return False
        for key in ("T_WT", "T_WE"):
            transform = np.asarray(record.get(key), dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                return False
        forces = np.asarray(record.get("forces"), dtype=np.float64)
        if forces.ndim != 1 or len(forces) < 3 or not np.all(np.isfinite(forces[:3])):
            return False
    return True


def validate_pokeflex_shrinkage_target_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact target lock before any target member is opened."""

    _require(payload.get("schema_version") == 1, "target schema changed")
    _require(
        payload.get("artifact_kind") == "PokeFlexConservativeShrinkageTargetProtocol",
        "target artifact kind changed",
    )
    protocol_id = payload.get("protocol_id")
    _require(
        protocol_id
        in {
            TARGET_PROTOCOL_V1,
            TARGET_PROTOCOL_V2,
            TARGET_PROTOCOL_OFFICIAL18_V1,
        },
        "target protocol id changed",
    )
    observed = target_protocol_sha256(payload)
    _require(payload.get("protocol_sha256") == observed, "protocol checksum mismatch")

    source = payload.get("source_gate")
    _require(isinstance(source, Mapping), "source gate is missing")
    _require(
        source.get("protocol_sha256") == SOURCE_PROTOCOL_SHA256,
        "source protocol changed",
    )
    _require(
        source.get("result_sha256") == SOURCE_RESULT_SHA256,
        "source result changed",
    )
    _require(source.get("passed") is True, "source gate did not pass")
    _require(source.get("selected_arm") == SELECTED_ARM, "selected arm changed")

    cohort = payload.get("target_cohort")
    _require(isinstance(cohort, Mapping), "target cohort is missing")
    if protocol_id == TARGET_PROTOCOL_OFFICIAL18_V1:
        _require(
            tuple(cohort.get("take_ids", ())) == OFFICIAL18_TARGET_TAKE_IDS,
            "official target take cohort changed",
        )
        _require(
            tuple(cohort.get("prospective_take_ids", ()))
            == OFFICIAL18_PROSPECTIVE_TAKE_IDS,
            "prospective target take cohort changed",
        )
        _require(
            tuple(cohort.get("development_overlap_take_ids", ()))
            == OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS,
            "development-overlap cohort changed",
        )
    else:
        _require(
            tuple(cohort.get("objects", ())) == TARGET_OBJECTS,
            "target object cohort changed",
        )
        _require(cohort.get("take") == "T2", "target take changed")
    _require(cohort.get("replacement_allowed") is False, "replacement was enabled")

    method = payload.get("method")
    _require(isinstance(method, Mapping), "method lock is missing")
    _require(method.get("selected_arm") == SELECTED_ARM, "target arm changed")
    _require(method.get("field") == "action_local_state_relative_0.4", "field changed")
    _require(float(method.get("scale", -1.0)) == 0.125, "scale changed")
    _require(
        method.get("unsupported_frame_action") == "byte-identical released checkpoint",
        "fallback changed",
    )
    if protocol_requires_robot_history(str(protocol_id)):
        _require(
            method.get("missing_required_robot_pose_action")
            == "mark update unsupported and return byte-identical released checkpoint",
            "missing-pose fallback changed",
        )
    if protocol_id == TARGET_PROTOCOL_V2:
        amendment = payload.get("preoutcome_amendment")
        _require(isinstance(amendment, Mapping), "pre-outcome amendment is missing")
        _require(
            amendment.get("supersedes_protocol_sha256")
            == "7662ec3d92e2ae1d6872e32c218baaae27926924c730178d7477f98c684ff277",
            "superseded protocol changed",
        )
        _require(
            amendment.get("target_mesh_outcome_opened") is False,
            "amendment followed target outcome access",
        )
        _require(
            amendment.get("uniform_eight_take_rerun_required") is True,
            "uniform rerun requirement changed",
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
        int(custody.get("required_prediction_seal_count", -1))
        == len(target_take_ids_for_protocol(payload)),
        "prediction barrier count changed",
    )
    _require(
        custody.get("target_mesh_access_before_barrier") == "forbidden",
        "target mesh custody weakened",
    )
    _require(
        custody.get("prediction_observation_history") == "f-5 through f-1",
        "causal history changed",
    )

    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation lock is missing")
    _require(evaluation.get("primary_metric") == "CD_UL1_mm", "primary metric changed")
    _require(
        int(evaluation.get("surface_sample_count", -1)) == 10000, "sample count changed"
    )
    _require(
        int(evaluation.get("surface_sample_seed", -1)) == 20260720,
        "sample seed changed",
    )
    _require(
        evaluation.get("jaccard_definition")
        == "trimesh boolean intersection volume divided by union volume",
        "Jaccard definition changed",
    )
    _require(
        evaluation.get("jaccard_boolean_backend") == "manifold",
        "Jaccard backend changed",
    )
    if protocol_id == TARGET_PROTOCOL_OFFICIAL18_V1:
        _require(
            evaluation.get("aggregation")
            == "equal scored frames over the exact official 18-take split; object-balanced values are diagnostic",
            "official aggregation changed",
        )
        _require(
            evaluation.get("jaccard_role")
            == "non-gating reproducibility diagnostic because the released target meshes are not guaranteed volumetric",
            "official Jaccard role changed",
        )
        official = payload.get("official_reference")
        _require(isinstance(official, Mapping), "official reference is missing")
        _require(
            official.get("code_commit") == UPSTREAM_COMMIT,
            "official evaluator commit changed",
        )
        _require(
            official.get("evaluator_sha256") == OFFICIAL_EVALUATOR_SHA256,
            "official evaluator bytes changed",
        )
        _require(
            float(official.get("published_kinect_CD_UL1_mm", -1.0))
            == PUBLISHED_KINECT_CD_UL1_MM,
            "published CD reference changed",
        )
        _require(
            float(official.get("published_kinect_jaccard", -1.0))
            == PUBLISHED_KINECT_JACCARD,
            "published Jaccard reference changed",
        )
    else:
        _require(
            float(evaluation.get("minimum_candidate_jaccard_valid_fraction", -1.0))
            == 1.0,
            "Jaccard validity gate changed",
        )

    gates = payload.get("gates")
    _require(isinstance(gates, Mapping), "target gates are missing")
    direct = gates.get("direct_metric_reference")
    paired = gates.get("paired_transfer")
    _require(isinstance(direct, Mapping), "direct gate is missing")
    _require(isinstance(paired, Mapping), "paired gate is missing")
    if protocol_id == TARGET_PROTOCOL_OFFICIAL18_V1:
        reproduction = gates.get("baseline_reproduction")
        _require(isinstance(reproduction, Mapping), "reproduction gate is missing")
        _require(
            float(reproduction.get("maximum_relative_CD_UL1_error", -1.0)) == 0.05,
            "baseline reproduction tolerance changed",
        )
        _require(
            float(direct.get("candidate_CD_UL1_mm_below", -1.0))
            == PUBLISHED_KINECT_CD_UL1_MM,
            "official candidate CD gate changed",
        )
        _require(
            direct.get("jaccard_is_gating") is False,
            "official Jaccard unexpectedly became gating",
        )
    else:
        _require(
            float(direct.get("CD_UL1_mm_below", -1.0)) == 6.498,
            "CD gate changed",
        )
        _require(
            float(direct.get("jaccard_not_below", -1.0)) == 0.82,
            "Jaccard gate changed",
        )
    _require(
        float(paired.get("relative_CD_UL1_improvement_above", -1.0)) == 0.0,
        "paired improvement gate changed",
    )
    _require(
        float(paired.get("bootstrap_upper_difference_mm_below", 1.0)) == 0.0,
        "bootstrap gate changed",
    )
    _require(
        float(paired.get("maximum_per_object_relative_regression", 1.0)) == 0.0,
        "object-regression gate changed",
    )
    _require(
        int(paired.get("bootstrap_replicates", -1)) == 20000, "bootstrap count changed"
    )
    _require(
        int(paired.get("bootstrap_seed", -1)) == 20260720, "bootstrap seed changed"
    )
    _require(
        float(paired.get("bootstrap_upper_quantile", -1.0)) == 0.975,
        "bootstrap quantile changed",
    )
    return {
        "passed": True,
        "protocol_sha256": observed,
        "protocol_id": protocol_id,
        "target_take_ids": target_take_ids_for_protocol(payload),
        "selected_arm": SELECTED_ARM,
    }


def load_pokeflex_shrinkage_target_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pokeflex_shrinkage_target_protocol(payload)
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


def prediction_seal_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="seal_sha256")


def _load_prediction_arrays(
    path: Path,
    *,
    protocol_id: str,
) -> dict[str, np.ndarray]:
    required = {
        "baseline_vertices_m",
        "candidate_vertices_m",
        "faces",
        "target_frames",
        "source_frames",
        "history_start_frames",
        "history_end_frames",
        "update_supported",
        "update_accepted",
        "action_supported",
        "correction_rms_m",
    }
    if protocol_requires_robot_history(protocol_id):
        required.add("robot_history_supported")
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == required, "prediction array schema changed")
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_prediction_seal(
    seal_path: Path,
    protocol: Mapping[str, Any],
) -> PredictionArchive:
    """Validate one prediction without opening any target mesh."""

    validate_pokeflex_shrinkage_target_protocol(protocol)
    seal_path = Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal.get("schema_version") == 1, "prediction seal schema changed")
    _require(
        seal.get("artifact_kind") == "PokeFlexConservativeShrinkagePredictionSeal",
        "prediction seal kind changed",
    )
    _require(
        seal.get("seal_sha256") == prediction_seal_sha256(seal),
        "prediction seal checksum mismatch",
    )
    _require(
        seal.get("protocol_sha256") == protocol["protocol_sha256"],
        "prediction protocol changed",
    )
    _require(
        seal.get("source_result_sha256") == SOURCE_RESULT_SHA256,
        "prediction source result changed",
    )
    _require(seal.get("selected_arm") == SELECTED_ARM, "prediction arm changed")
    _require(seal.get("future_mesh_read") is False, "prediction read a future mesh")
    _require(
        int(seal.get("future_mesh_read_count", -1)) == 0,
        "prediction reports future mesh access",
    )
    _require(seal.get("implementation_clean") is True, "implementation was dirty")
    revision = seal.get("implementation_revision")
    _require(isinstance(revision, str) and len(revision) == 40, "revision is invalid")
    take_id = seal.get("take_id")
    target_take_ids = target_take_ids_for_protocol(protocol)
    _require(take_id in target_take_ids, "prediction take is outside target cohort")
    object_name, take = _take_identity(str(take_id))
    _require(seal.get("object_name") == object_name, "prediction object changed")
    protocol_id = str(protocol["protocol_id"])
    if protocol_id != TARGET_PROTOCOL_OFFICIAL18_V1:
        _require(take == "T2", "prediction take changed")
    _require(
        dict(seal.get("checkpoint_sha256", {})) == CHECKPOINT_SHA256,
        "prediction checkpoint changed",
    )
    _require(
        seal.get("upstream_commit") == UPSTREAM_COMMIT, "prediction upstream changed"
    )

    npz_path = seal_path.parent / str(seal.get("prediction_npz", ""))
    _require(npz_path.is_file(), "prediction archive is missing")
    _require(
        file_sha256(npz_path) == seal.get("prediction_npz_sha256"),
        "prediction archive checksum mismatch",
    )
    arrays = _load_prediction_arrays(npz_path, protocol_id=protocol_id)
    baseline = np.asarray(arrays["baseline_vertices_m"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_vertices_m"], dtype=np.float64)
    faces = np.asarray(arrays["faces"])
    frames = np.asarray(arrays["target_frames"])
    source_frames = np.asarray(arrays["source_frames"])
    starts = np.asarray(arrays["history_start_frames"])
    ends = np.asarray(arrays["history_end_frames"])
    supported = np.asarray(arrays["update_supported"])
    _require(
        baseline.ndim == 3 and baseline.shape[-1] == 3,
        "baseline vertices must be FxNx3",
    )
    _require(candidate.shape == baseline.shape, "candidate vertex shape changed")
    _require(np.all(np.isfinite(baseline)), "baseline contains non-finite values")
    _require(np.all(np.isfinite(candidate)), "candidate contains non-finite values")
    _require(faces.ndim == 2 and faces.shape[1] == 3, "faces must be Mx3")
    _require(np.issubdtype(faces.dtype, np.integer), "faces must be integer")
    _require(len(frames) == len(baseline), "target frame count changed")
    _require(np.array_equal(source_frames, frames - 1), "source frame is not f-1")
    _require(np.array_equal(starts, frames - 5), "history does not start at f-5")
    _require(np.array_equal(ends, frames - 1), "history does not end at f-1")
    _require(np.all(np.diff(frames) == 1), "prediction frames are not contiguous")
    _require(int(frames[0]) == 6, "prediction does not begin at frame six")
    _require(supported.shape == frames.shape, "support shape changed")
    _require(supported.dtype == np.bool_, "support mask must be Boolean")
    if protocol_requires_robot_history(protocol_id):
        robot_supported = np.asarray(arrays["robot_history_supported"])
        _require(robot_supported.shape == frames.shape, "robot support shape changed")
        _require(robot_supported.dtype == np.bool_, "robot support must be Boolean")
        _require(
            not np.any(supported & ~robot_supported),
            "prediction used incomplete robot history",
        )
        _require(
            int(seal.get("missing_robot_history_frame_count", -1))
            == int(np.sum(~robot_supported)),
            "missing robot-history count changed",
        )
    _require(
        np.array_equal(candidate[~supported], baseline[~supported]),
        "unsupported prediction is not an exact fallback",
    )
    _require(
        int(seal.get("fallback_mismatch_count", -1)) == 0,
        "prediction fallback mismatch was recorded",
    )
    _require(
        int(seal.get("predicted_frame_count", -1)) == len(frames), "frame count changed"
    )
    return PredictionArchive(
        take_id=str(take_id),
        seal_path=seal_path,
        npz_path=npz_path,
        implementation_revision=revision,
        baseline_vertices_m=baseline,
        candidate_vertices_m=candidate,
        faces=np.asarray(faces, dtype=np.int64),
        target_frames=np.asarray(frames, dtype=np.int64),
        update_supported=np.asarray(supported, dtype=np.bool_),
    )


def barrier_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, digest_field="barrier_sha256")


def build_prediction_barrier(
    seal_paths: Sequence[Path],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every registered prediction before target scoring is authorized."""

    validate_pokeflex_shrinkage_target_protocol(protocol)
    target_take_ids = target_take_ids_for_protocol(protocol)
    archives = [validate_prediction_seal(path, protocol) for path in seal_paths]
    _require(len(archives) == len(target_take_ids), "prediction barrier is incomplete")
    by_take = {archive.take_id: archive for archive in archives}
    _require(len(by_take) == len(archives), "duplicate prediction seal")
    _require(
        tuple(sorted(by_take)) == tuple(sorted(target_take_ids)),
        "target seal set changed",
    )
    revisions = {archive.implementation_revision for archive in archives}
    _require(len(revisions) == 1, "prediction revisions differ")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkagePredictionBarrier",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "selected_arm": SELECTED_ARM,
        "implementation_revision": next(iter(revisions)),
        "prediction_count": len(archives),
        "target_take_ids": list(target_take_ids),
        "predictions": [
            {
                "take_id": take_id,
                "seal_path": str(by_take[take_id].seal_path),
                "seal_file_sha256": file_sha256(by_take[take_id].seal_path),
                "prediction_npz_sha256": file_sha256(by_take[take_id].npz_path),
            }
            for take_id in target_take_ids
        ],
        "target_mesh_opened": False,
        "scoring_authorized": True,
    }
    payload["barrier_sha256"] = barrier_sha256(payload)
    return payload


def validate_prediction_barrier(
    barrier: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pokeflex_shrinkage_target_protocol(protocol)
    target_take_ids = target_take_ids_for_protocol(protocol)
    _require(
        barrier.get("artifact_kind")
        == "PokeFlexConservativeShrinkagePredictionBarrier",
        "barrier kind changed",
    )
    _require(
        barrier.get("barrier_sha256") == barrier_sha256(barrier),
        "barrier checksum mismatch",
    )
    _require(
        barrier.get("protocol_sha256") == protocol["protocol_sha256"],
        "barrier protocol changed",
    )
    _require(
        barrier.get("source_result_sha256") == SOURCE_RESULT_SHA256,
        "barrier source changed",
    )
    _require(barrier.get("selected_arm") == SELECTED_ARM, "barrier arm changed")
    _require(
        int(barrier.get("prediction_count", -1)) == len(target_take_ids),
        "barrier count changed",
    )
    _require(
        tuple(barrier.get("target_take_ids", ())) == target_take_ids,
        "barrier cohort changed",
    )
    _require(
        barrier.get("target_mesh_opened") is False, "barrier reports target access"
    )
    _require(
        barrier.get("scoring_authorized") is True, "barrier did not authorize scoring"
    )
    predictions = barrier.get("predictions")
    _require(
        isinstance(predictions, list) and len(predictions) == len(target_take_ids),
        "barrier prediction inventory changed",
    )
    _require(
        tuple(row.get("take_id") for row in predictions) == target_take_ids,
        "barrier prediction order changed",
    )
    return {"passed": True, "barrier_sha256": barrier["barrier_sha256"]}


def surface_sample(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample a mesh surface deterministically using the registered scorer rule."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = points[np.asarray(faces, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    _require(np.all(np.isfinite(areas)), "mesh has non-finite area")
    _require(float(np.sum(areas)) > 0.0, "mesh has zero surface area")
    generator = np.random.default_rng(seed)
    face_indices = generator.choice(len(faces), size=count, p=areas / np.sum(areas))
    first = generator.random(count)
    second = generator.random(count)
    reflected = first + second > 1.0
    first[reflected] = 1.0 - first[reflected]
    second[reflected] = 1.0 - second[reflected]
    chosen = triangles[face_indices]
    return (
        chosen[:, 0]
        + first[:, None] * (chosen[:, 1] - chosen[:, 0])
        + second[:, None] * (chosen[:, 2] - chosen[:, 0])
    )


def cd_ul1_mm(prediction: np.ndarray, target: np.ndarray) -> float:
    """Official one-sided nearest-neighbor L1 surface distance in millimetres."""

    from scipy.spatial import cKDTree

    indices = cKDTree(target).query(prediction, k=1)[1]
    return float(1000.0 * np.mean(np.sum(np.abs(prediction - target[indices]), axis=1)))


def official_volumetric_jaccard(
    prediction_vertices_m: np.ndarray,
    prediction_faces: np.ndarray,
    target_vertices_m: np.ndarray,
    target_faces: np.ndarray,
    *,
    engine: str = "manifold",
) -> float:
    """Compute PokeFlex's boolean-volume Jaccard with an explicit backend."""

    import trimesh

    available = set(trimesh.boolean.engines_available)
    _require(
        engine in available,
        f"required trimesh boolean backend is unavailable: {engine}",
    )
    prediction = trimesh.Trimesh(
        vertices=np.asarray(prediction_vertices_m, dtype=np.float64),
        faces=np.asarray(prediction_faces, dtype=np.int64),
        process=False,
    )
    target = trimesh.Trimesh(
        vertices=np.asarray(target_vertices_m, dtype=np.float64),
        faces=np.asarray(target_faces, dtype=np.int64),
        process=False,
    )
    union = prediction.union(target, engine=engine)
    intersection = prediction.intersection(target, engine=engine)
    union_volume = float(union.volume)
    intersection_volume = float(intersection.volume)
    _require(np.isfinite(union_volume) and union_volume > 0.0, "invalid union volume")
    _require(
        np.isfinite(intersection_volume) and intersection_volume >= 0.0,
        "invalid intersection volume",
    )
    value = intersection_volume / union_volume
    _require(np.isfinite(value) and 0.0 <= value <= 1.0 + 1e-9, "invalid Jaccard value")
    return float(min(value, 1.0))


def paired_object_bootstrap_upper_difference(
    differences_mm: Sequence[float],
    *,
    replicates: int,
    seed: int,
    upper_quantile: float,
) -> float:
    """Bootstrap the object-balanced candidate-minus-baseline difference."""

    values = np.asarray(differences_mm, dtype=np.float64)
    _require(values.ndim == 1 and len(values) >= 2, "bootstrap requires objects")
    _require(np.all(np.isfinite(values)), "bootstrap differences are non-finite")
    _require(replicates >= 1000, "bootstrap replicate count is too small")
    _require(0.5 < upper_quantile < 1.0, "bootstrap quantile is invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    draws = np.mean(values[indices], axis=1)
    return float(np.quantile(draws, upper_quantile))


def _evaluate_official18_metrics(
    per_take: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exact-split reproduction and prospective-transfer gates."""

    _require(
        len(per_take) == len(OFFICIAL18_TARGET_TAKE_IDS),
        "official target result set is incomplete",
    )
    by_take = {str(row["take_id"]): row for row in per_take}
    _require(
        tuple(sorted(by_take)) == tuple(sorted(OFFICIAL18_TARGET_TAKE_IDS)),
        "official target result cohort changed",
    )
    ordered = [by_take[take_id] for take_id in OFFICIAL18_TARGET_TAKE_IDS]
    for row in ordered:
        frames = row.get("frames")
        _require(isinstance(frames, list), "official frame scores are missing")
        _require(
            len(frames) == int(row["scored_frame_count"]),
            "official scored-frame inventory changed",
        )

    baseline_frames = np.asarray(
        [frame["baseline_CD_UL1_mm"] for row in ordered for frame in row["frames"]],
        dtype=np.float64,
    )
    candidate_frames = np.asarray(
        [frame["candidate_CD_UL1_mm"] for row in ordered for frame in row["frames"]],
        dtype=np.float64,
    )
    _require(len(baseline_frames) > 0, "official target has no scored frames")
    _require(np.all(np.isfinite(baseline_frames)), "baseline scores are non-finite")
    _require(np.all(np.isfinite(candidate_frames)), "candidate scores are non-finite")
    baseline_global = float(np.mean(baseline_frames))
    candidate_global = float(np.mean(candidate_frames))
    _require(baseline_global > 0.0, "baseline official score is zero")

    baseline_object = np.asarray(
        [row["baseline_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    candidate_object = np.asarray(
        [row["candidate_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    _require(np.all(np.isfinite(baseline_object)), "baseline object scores are invalid")
    _require(
        np.all(np.isfinite(candidate_object)), "candidate object scores are invalid"
    )

    prospective = [by_take[take_id] for take_id in OFFICIAL18_PROSPECTIVE_TAKE_IDS]
    prospective_baseline = np.asarray(
        [row["baseline_mean_CD_UL1_mm"] for row in prospective], dtype=np.float64
    )
    prospective_candidate = np.asarray(
        [row["candidate_mean_CD_UL1_mm"] for row in prospective], dtype=np.float64
    )
    prospective_baseline_mean = float(np.mean(prospective_baseline))
    prospective_candidate_mean = float(np.mean(prospective_candidate))
    _require(prospective_baseline_mean > 0.0, "prospective baseline score is zero")
    prospective_relative = float(
        (prospective_baseline_mean - prospective_candidate_mean)
        / prospective_baseline_mean
    )
    prospective_object_relative = (
        prospective_baseline - prospective_candidate
    ) / prospective_baseline

    gates = protocol["gates"]
    paired = gates["paired_transfer"]
    upper_difference = paired_object_bootstrap_upper_difference(
        prospective_candidate - prospective_baseline,
        replicates=int(paired["bootstrap_replicates"]),
        seed=int(paired["bootstrap_seed"]),
        upper_quantile=float(paired["bootstrap_upper_quantile"]),
    )
    published = float(
        protocol["official_reference"]["published_kinect_CD_UL1_mm"]
    )
    reproduction_relative_error = abs(baseline_global - published) / published
    reproduction_pass = bool(
        reproduction_relative_error
        <= float(gates["baseline_reproduction"]["maximum_relative_CD_UL1_error"])
    )
    candidate_reference_pass = bool(
        candidate_global
        < float(gates["direct_metric_reference"]["candidate_CD_UL1_mm_below"])
    )
    paired_pass = bool(
        prospective_relative
        > float(paired["relative_CD_UL1_improvement_above"])
        and upper_difference < float(paired["bootstrap_upper_difference_mm_below"])
        and float(np.min(prospective_object_relative))
        >= -float(paired["maximum_per_object_relative_regression"])
    )

    candidate_jaccard = [
        float(frame["candidate_jaccard"])
        for row in ordered
        for frame in row["frames"]
        if frame["candidate_jaccard"] is not None
    ]
    total_frames = len(candidate_frames)
    direct_pass = bool(reproduction_pass and candidate_reference_pass)
    return {
        "published_kinect_CD_UL1_mm": published,
        "baseline_official_split_global_CD_UL1_mm": baseline_global,
        "candidate_official_split_global_CD_UL1_mm": candidate_global,
        "official_split_global_relative_CD_UL1_improvement": float(
            (baseline_global - candidate_global) / baseline_global
        ),
        "baseline_reproduction_relative_error": float(reproduction_relative_error),
        "baseline_reproduction_passed": reproduction_pass,
        "candidate_below_published_reference_passed": candidate_reference_pass,
        "baseline_official_split_object_balanced_CD_UL1_mm": float(
            np.mean(baseline_object)
        ),
        "candidate_official_split_object_balanced_CD_UL1_mm": float(
            np.mean(candidate_object)
        ),
        "full18_object_win_count": int(np.sum(candidate_object < baseline_object)),
        "prospective_take_count": len(OFFICIAL18_PROSPECTIVE_TAKE_IDS),
        "development_overlap_take_count": len(
            OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS
        ),
        "prospective_object_balanced_baseline_CD_UL1_mm": prospective_baseline_mean,
        "prospective_object_balanced_candidate_CD_UL1_mm": prospective_candidate_mean,
        "prospective_object_balanced_relative_CD_UL1_improvement": prospective_relative,
        "prospective_object_win_count": int(
            np.sum(prospective_candidate < prospective_baseline)
        ),
        "prospective_minimum_per_object_relative_improvement": float(
            np.min(prospective_object_relative)
        ),
        "prospective_bootstrap_upper_candidate_minus_baseline_CD_UL1_mm": (
            upper_difference
        ),
        "candidate_global_jaccard_valid": (
            float(np.mean(candidate_jaccard)) if candidate_jaccard else None
        ),
        "candidate_jaccard_valid_fraction": float(
            len(candidate_jaccard) / total_frames
        ),
        "jaccard_is_gating": False,
        "direct_metric_reference_passed": direct_pass,
        "paired_transfer_passed": paired_pass,
        "all_target_gates_passed": bool(direct_pass and paired_pass),
    }


def evaluate_target_metrics(
    per_object: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate already-scored target objects and apply the frozen gates."""

    validate_pokeflex_shrinkage_target_protocol(protocol)
    if protocol["protocol_id"] == TARGET_PROTOCOL_OFFICIAL18_V1:
        return _evaluate_official18_metrics(per_object, protocol)
    _require(len(per_object) == len(TARGET_OBJECTS), "target result set is incomplete")
    by_object = {str(row["object_name"]): row for row in per_object}
    _require(
        tuple(sorted(by_object)) == tuple(sorted(TARGET_OBJECTS)),
        "target result cohort changed",
    )
    ordered = [by_object[name] for name in TARGET_OBJECTS]
    baseline = np.asarray(
        [row["baseline_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    candidate = np.asarray(
        [row["candidate_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    _require(np.all(np.isfinite(baseline)), "baseline target scores are non-finite")
    _require(np.all(np.isfinite(candidate)), "candidate target scores are non-finite")
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    relative = float((baseline_mean - candidate_mean) / baseline_mean)
    object_relative = (baseline - candidate) / baseline
    evaluation = protocol["evaluation"]
    candidate_valid = sum(int(row["candidate_jaccard_valid_count"]) for row in ordered)
    total_frames = sum(int(row["scored_frame_count"]) for row in ordered)
    candidate_jaccard_valid_fraction = candidate_valid / total_frames
    valid_object_jaccard = [
        float(row["candidate_mean_jaccard_valid"])
        for row in ordered
        if row["candidate_mean_jaccard_valid"] is not None
    ]
    candidate_jaccard = (
        float(np.mean(valid_object_jaccard)) if valid_object_jaccard else None
    )
    paired = protocol["gates"]["paired_transfer"]
    upper_difference = paired_object_bootstrap_upper_difference(
        candidate - baseline,
        replicates=int(paired["bootstrap_replicates"]),
        seed=int(paired["bootstrap_seed"]),
        upper_quantile=float(paired["bootstrap_upper_quantile"]),
    )
    direct = protocol["gates"]["direct_metric_reference"]
    direct_pass = bool(
        candidate_mean < float(direct["CD_UL1_mm_below"])
        and candidate_jaccard_valid_fraction
        >= float(evaluation["minimum_candidate_jaccard_valid_fraction"])
        and candidate_jaccard is not None
        and candidate_jaccard >= float(direct["jaccard_not_below"])
    )
    paired_pass = bool(
        relative > float(paired["relative_CD_UL1_improvement_above"])
        and upper_difference < float(paired["bootstrap_upper_difference_mm_below"])
        and float(np.min(object_relative))
        >= -float(paired["maximum_per_object_relative_regression"])
    )
    return {
        "baseline_object_balanced_CD_UL1_mm": baseline_mean,
        "candidate_object_balanced_CD_UL1_mm": candidate_mean,
        "object_balanced_relative_CD_UL1_improvement": relative,
        "object_win_count": int(np.sum(candidate < baseline)),
        "minimum_per_object_relative_improvement": float(np.min(object_relative)),
        "bootstrap_upper_candidate_minus_baseline_CD_UL1_mm": upper_difference,
        "candidate_object_balanced_jaccard_valid": candidate_jaccard,
        "candidate_jaccard_valid_fraction": float(candidate_jaccard_valid_fraction),
        "direct_metric_reference_passed": direct_pass,
        "paired_transfer_passed": paired_pass,
        "all_target_gates_passed": bool(direct_pass and paired_pass),
    }


def score_one_prediction(
    archive: PredictionArchive,
    active_frames: Sequence[int],
    mesh_loader: Callable[[int], tuple[np.ndarray, np.ndarray]],
    protocol: Mapping[str, Any],
    *,
    jaccard: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
    | None = None,
) -> dict[str, Any]:
    """Open target meshes only after custody checks and score one object."""

    validate_pokeflex_shrinkage_target_protocol(protocol)
    active = tuple(sorted(int(frame) for frame in active_frames if int(frame) >= 6))
    _require(bool(active), "target take has no scored action frames")
    frame_to_index = {
        int(frame): index for index, frame in enumerate(archive.target_frames)
    }
    _require(
        all(frame in frame_to_index for frame in active), "prediction frame is missing"
    )
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
        )
    )
    rows = []
    for frame in active:
        index = frame_to_index[frame]
        target_vertices, target_faces = mesh_loader(frame)
        target_sample = surface_sample(
            target_vertices, target_faces, count, seed + frame
        )
        baseline_sample = surface_sample(
            archive.baseline_vertices_m[index], archive.faces, count, seed + frame
        )
        candidate_sample = surface_sample(
            archive.candidate_vertices_m[index], archive.faces, count, seed + frame
        )
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
        try:
            row["baseline_jaccard"] = jaccard_function(
                archive.baseline_vertices_m[index],
                archive.faces,
                target_vertices,
                target_faces,
            )
        except Exception as error:
            row["baseline_jaccard_error"] = f"{type(error).__name__}: {error}"
        try:
            row["candidate_jaccard"] = jaccard_function(
                archive.candidate_vertices_m[index],
                archive.faces,
                target_vertices,
                target_faces,
            )
        except Exception as error:
            row["candidate_jaccard_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    baseline_errors = np.asarray([row["baseline_CD_UL1_mm"] for row in rows])
    candidate_errors = np.asarray([row["candidate_CD_UL1_mm"] for row in rows])
    baseline_mean = float(np.mean(baseline_errors))
    candidate_mean = float(np.mean(candidate_errors))
    _require(baseline_mean > 0.0, "baseline target error is zero")
    baseline_jaccard = [
        row["baseline_jaccard"] for row in rows if row["baseline_jaccard"] is not None
    ]
    candidate_jaccard = [
        row["candidate_jaccard"] for row in rows if row["candidate_jaccard"] is not None
    ]
    object_name, _ = _take_identity(archive.take_id)
    return {
        "object_name": object_name,
        "take_id": archive.take_id,
        "scored_frame_count": len(rows),
        "supported_frame_count": int(sum(row["update_supported"] for row in rows)),
        "baseline_mean_CD_UL1_mm": baseline_mean,
        "candidate_mean_CD_UL1_mm": candidate_mean,
        "relative_CD_UL1_improvement": float(
            (baseline_mean - candidate_mean) / baseline_mean
        ),
        "baseline_jaccard_valid_count": len(baseline_jaccard),
        "candidate_jaccard_valid_count": len(candidate_jaccard),
        "baseline_mean_jaccard_valid": (
            float(np.mean(baseline_jaccard)) if baseline_jaccard else None
        ),
        "candidate_mean_jaccard_valid": (
            float(np.mean(candidate_jaccard)) if candidate_jaccard else None
        ),
        "frames": rows,
    }
