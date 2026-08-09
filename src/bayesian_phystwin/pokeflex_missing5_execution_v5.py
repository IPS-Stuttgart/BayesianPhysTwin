"""Custody-safe execution and scoring for the five PokeFlex V5 targets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_action_robust_all18 import SOURCE_FIELD
from .pokeflex_action_robust_official18_v4 import (
    EXPECTED_PROTOCOL_SHA256 as PARENT_PROTOCOL_SHA256,
)
from .pokeflex_action_robust_official18_v4 import (
    PUBLIC13_RESULT_FILE_SHA256,
    validate_author_source_manifest,
    validate_official18_v4_protocol,
)
from .pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
    OFFICIAL18_TARGET_TAKE_IDS,
    PUBLISHED_KINECT_CD_UL1_MM,
    UPSTREAM_COMMIT,
    cd_ul1_mm,
    paired_object_bootstrap_upper_difference,
    surface_sample,
)
from .pokeflex_missing5_completion_v5 import (
    TARGET_MULTIPLIERS,
    validate_completion_protocol,
)
from .pokeflex_missing5_completion_v5 import (
    protocol_sha256 as completion_protocol_sha256,
)

EXECUTION_PROTOCOL_ID = "pokeflex-missing5-execution-v5"
EXECUTION_PROTOCOL_KIND = "PokeFlexMissingFiveV5ExecutionProtocol"
INPUT_STAGE_KIND = "PokeFlexMissingFiveV5PredictionInputStage"
PREDICTION_SEAL_KIND = "PokeFlexMissingFiveV5PredictionSeal"
PREDICTION_BARRIER_KIND = "PokeFlexMissingFiveV5PredictionBarrier"
RESULT_KIND = "PokeFlexMissingFiveV5ProspectiveResult"

TARGET_TAKE_IDS = tuple(OFFICIAL18_MISSING_PUBLIC_TAKE_IDS)
BASE_EFFECTIVE_SCALE = 0.125
IMPLEMENTATION_FILE_PATHS = (
    "scripts/held/run_pokeflex_missing5_v5.py",
    "scripts/remote/run_pokeflex_bayesian_registration_smoke.py",
    "scripts/remote/run_pokeflex_checkpoint_registration_independent_depth.py",
    "src/bayesian_phystwin/pokeflex_bayesian_registration.py",
    "src/bayesian_phystwin/pokeflex_missing5_execution_v5.py",
    "src/bayesian_phystwin/pokeflex_released_checkpoint.py",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def execution_protocol_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "execution_protocol_sha256")


def input_stage_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "input_stage_sha256")


def prediction_seal_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "seal_sha256")


def prediction_barrier_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "prediction_barrier_sha256")


def result_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "result_sha256")


def build_execution_protocol(
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    *,
    locked_at_utc: str,
    implementation_file_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Build the pre-archive-access execution lock."""

    validate_completion_protocol(completion_protocol)
    validate_official18_v4_protocol(parent_protocol)
    _require(
        set(implementation_file_sha256s) == set(IMPLEMENTATION_FILE_PATHS),
        "implementation file inventory changed",
    )
    _require(
        all(_is_sha256(value) for value in implementation_file_sha256s.values()),
        "implementation file hash is invalid",
    )
    v4_multipliers = parent_protocol["method"]["multipliers"]
    v4_scales = {
        take_id: BASE_EFFECTIVE_SCALE
        * float(v4_multipliers[take_id.rpartition("_T")[0]])
        for take_id in TARGET_TAKE_IDS
    }
    v5_scales = {
        take_id: BASE_EFFECTIVE_SCALE * float(TARGET_MULTIPLIERS[take_id])
        for take_id in TARGET_TAKE_IDS
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": EXECUTION_PROTOCOL_KIND,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "locked_at_utc": locked_at_utc,
        "completion_protocol_sha256": completion_protocol["protocol_sha256"],
        "parent_v4_protocol_sha256": parent_protocol["protocol_sha256"],
        "target_take_ids": list(TARGET_TAKE_IDS),
        "method": {
            "field": SOURCE_FIELD,
            "global_effective_scale": BASE_EFFECTIVE_SCALE,
            "v4_effective_scales": v4_scales,
            "v5_effective_scales": v5_scales,
            "unsupported_frame_action": "byte-identical released checkpoint",
        },
        "prediction_input_boundary": {
            "allowed": [
                "robot_data.json",
                "two Kinect camera-parameter files",
                "Kinect depth frames strictly before each predicted frame",
                "exactly one upstream-selected template mesh",
            ],
            "authorized_template_mesh_count_per_take": 1,
            "template_selection_rule": (
                "frame 1 when deformation begins later; otherwise midpoint of the "
                "first inactive gap longer than five frames"
            ),
            "template_role": "explicit published-task input, not a scored outcome",
            "future_target_mesh_member_decoding_before_barrier": "forbidden",
            "target_metric_before_barrier": "forbidden",
        },
        "artifacts": {
            "input_stage_kind": INPUT_STAGE_KIND,
            "prediction_seal_kind": PREDICTION_SEAL_KIND,
            "prediction_barrier_kind": PREDICTION_BARRIER_KIND,
            "result_kind": RESULT_KIND,
        },
        "upstream": {
            "commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": dict(CHECKPOINT_SHA256),
        },
        "evaluation": {
            "metric": "CD_UL1_mm",
            "surface_sample_count": 10_000,
            "surface_sample_seed": 20_260_720,
            "public13_result_file_sha256": PUBLIC13_RESULT_FILE_SHA256,
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20_260_806,
            "bootstrap_upper_quantile": 0.975,
        },
        "custody": {
            "required_prediction_count": 5,
            "all_prediction_revisions_must_match": True,
            "prediction_and_scoring_are_separate": True,
            "target_mesh_access_before_barrier": "forbidden",
            "replacement_allowed": False,
            "target_adaptation": "forbidden",
        },
        "implementation_file_sha256s": dict(
            sorted(implementation_file_sha256s.items())
        ),
        "held_v8_accessed": False,
        "execution_protocol_sha256": "",
    }
    payload["execution_protocol_sha256"] = execution_protocol_sha256(payload)
    validate_execution_protocol(
        payload,
        completion_protocol,
        parent_protocol,
        bind_registered_digest=False,
    )
    return payload


def validate_execution_protocol(
    payload: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate the execution lock and its causal input boundary."""

    validate_completion_protocol(completion_protocol)
    validate_official18_v4_protocol(parent_protocol)
    _require(payload.get("schema_version") == 1, "execution schema changed")
    _require(
        payload.get("artifact_kind") == EXECUTION_PROTOCOL_KIND,
        "execution kind changed",
    )
    _require(
        payload.get("protocol_id") == EXECUTION_PROTOCOL_ID, "execution id changed"
    )
    observed = execution_protocol_sha256(payload)
    _require(
        payload.get("execution_protocol_sha256") == observed,
        "execution checksum changed",
    )
    if bind_registered_digest:
        from .pokeflex_missing5_execution_v5_lock import (
            EXPECTED_EXECUTION_PROTOCOL_SHA256,
        )

        _require(
            observed == EXPECTED_EXECUTION_PROTOCOL_SHA256,
            "registered execution lock changed",
        )
    _require(
        payload.get("completion_protocol_sha256")
        == completion_protocol_sha256(completion_protocol),
        "completion protocol changed",
    )
    _require(
        payload.get("parent_v4_protocol_sha256") == PARENT_PROTOCOL_SHA256,
        "parent V4 protocol changed",
    )
    _require(
        tuple(payload.get("target_take_ids", ())) == TARGET_TAKE_IDS,
        "target cohort changed",
    )
    method = payload.get("method")
    _require(isinstance(method, Mapping), "execution method is missing")
    assert isinstance(method, Mapping)
    _require(method.get("field") == SOURCE_FIELD, "correction field changed")
    _require(
        float(method.get("global_effective_scale", -1.0)) == BASE_EFFECTIVE_SCALE,
        "global scale changed",
    )
    expected_v5 = {
        take_id: BASE_EFFECTIVE_SCALE * TARGET_MULTIPLIERS[take_id]
        for take_id in TARGET_TAKE_IDS
    }
    _require(
        dict(method.get("v5_effective_scales", {})) == expected_v5, "V5 scales changed"
    )
    expected_v4 = {
        take_id: BASE_EFFECTIVE_SCALE
        * float(parent_protocol["method"]["multipliers"][take_id.rpartition("_T")[0]])
        for take_id in TARGET_TAKE_IDS
    }
    _require(
        dict(method.get("v4_effective_scales", {})) == expected_v4, "V4 scales changed"
    )
    boundary = payload.get("prediction_input_boundary")
    _require(isinstance(boundary, Mapping), "prediction boundary is missing")
    assert isinstance(boundary, Mapping)
    _require(
        int(boundary.get("authorized_template_mesh_count_per_take", -1)) == 1,
        "template-mesh allowance changed",
    )
    _require(
        boundary.get("future_target_mesh_member_decoding_before_barrier")
        == "forbidden",
        "future-mesh custody weakened",
    )
    custody = payload.get("custody")
    _require(isinstance(custody, Mapping), "execution custody is missing")
    assert isinstance(custody, Mapping)
    _require(
        int(custody.get("required_prediction_count", -1)) == 5, "barrier count changed"
    )
    _require(
        custody.get("target_mesh_access_before_barrier") == "forbidden",
        "target custody weakened",
    )
    _require(
        custody.get("target_adaptation") == "forbidden", "target adaptation enabled"
    )
    files = payload.get("implementation_file_sha256s")
    _require(isinstance(files, Mapping), "implementation file hashes are missing")
    assert isinstance(files, Mapping)
    _require(
        set(files) == set(IMPLEMENTATION_FILE_PATHS), "implementation inventory changed"
    )
    _require(
        all(_is_sha256(value) for value in files.values()),
        "implementation hash is invalid",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 boundary changed")
    return {
        "passed": True,
        "execution_protocol_sha256": observed,
        "target_take_ids": TARGET_TAKE_IDS,
    }


def verify_implementation_files(
    payload: Mapping[str, Any], repository_root: Path
) -> None:
    """Require the checked-out implementation bytes frozen by the execution lock."""

    expected = payload["implementation_file_sha256s"]
    for relative_path in IMPLEMENTATION_FILE_PATHS:
        path = Path(repository_root) / relative_path
        _require(path.is_file(), f"implementation file is missing: {relative_path}")
        _require(
            file_sha256(path) == expected[relative_path],
            f"implementation file changed: {relative_path}",
        )


def load_execution_protocol(
    path: Path,
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_execution_protocol(payload, completion_protocol, parent_protocol)
    return payload


def _source_row(source_manifest: Mapping[str, Any], take_id: str) -> Mapping[str, Any]:
    rows = {str(row["take_id"]): row for row in source_manifest["takes"]}
    _require(set(rows) == set(TARGET_TAKE_IDS), "source manifest cohort changed")
    return rows[take_id]


def validate_input_stage(
    payload: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one prediction-only extraction without reading staged payloads."""

    validate_execution_protocol(
        execution_protocol, completion_protocol, parent_protocol
    )
    source_validation = validate_author_source_manifest(
        source_manifest, parent_protocol
    )
    _require(payload.get("schema_version") == 1, "input-stage schema changed")
    _require(
        payload.get("artifact_kind") == INPUT_STAGE_KIND, "input-stage kind changed"
    )
    observed = input_stage_sha256(payload)
    _require(
        payload.get("input_stage_sha256") == observed, "input-stage checksum changed"
    )
    _require(
        payload.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "input-stage execution protocol changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_validation["source_manifest_sha256"],
        "input-stage source manifest changed",
    )
    take_id = str(payload.get("take_id", ""))
    _require(take_id in TARGET_TAKE_IDS, "input-stage take is outside target cohort")
    source = _source_row(source_manifest, take_id)
    _require(
        payload.get("source_archive_name") == source["source_payload_name"],
        "source archive name changed",
    )
    _require(
        payload.get("source_archive_sha256") == source["source_payload_sha256"],
        "source archive changed",
    )
    _require(
        payload.get("source_member_manifest_sha256")
        == source["member_manifest_sha256"],
        "source member inventory changed",
    )
    _require(
        int(payload.get("frame_limit", 0)) == int(source["episode_length"]),
        "episode length changed",
    )
    template_frame = int(payload.get("template_frame", -1))
    expected_template = f"{take_id}/meshes/mesh-f{template_frame:05d}.obj"
    _require(
        payload.get("authorized_template_member") == expected_template,
        "template member changed",
    )
    _require(
        payload.get("authorized_template_mesh_decoded") is True,
        "template was not staged",
    )
    _require(
        int(payload.get("authorized_template_mesh_decode_count", -1)) == 1,
        "template count changed",
    )
    _require(
        int(payload.get("future_target_mesh_member_decoded_count", -1)) == 0,
        "future target mesh was decoded",
    )
    _require(
        payload.get("target_metric_computed") is False, "target metric was computed"
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    records = payload.get("inputs")
    _require(isinstance(records, list), "staged inputs are missing")
    assert isinstance(records, list)
    paths = [str(row.get("archive_member", "")) for row in records]
    _require(len(paths) == len(set(paths)), "staged input member is duplicated")
    expected = {f"{take_id}/robot_data.json", expected_template}
    expected.update(
        f"{take_id}/kinect/{camera}/camera_parameters.json" for camera in (0, 1)
    )
    expected.update(
        f"{take_id}/kinect/{camera}/depth/{frame:05d}.png"
        for camera in (0, 1)
        for frame in range(1, int(payload["frame_limit"]))
    )
    _require(set(paths) == expected, "staged input inventory changed")
    mesh_members = [path for path in paths if f"{take_id}/meshes/" in path]
    _require(
        mesh_members == [expected_template],
        "more than one mesh entered prediction stage",
    )
    for row in records:
        _require(_is_sha256(row.get("sha256")), "staged input is unbound")
        _require(int(row.get("byte_count", 0)) > 0, "staged input is empty")
        _require(
            row.get("staged_relative_path") == row.get("archive_member"),
            "staged path changed",
        )
    return {
        "input_stage_sha256": observed,
        "take_id": take_id,
        "template_frame": template_frame,
        "input_count": len(records),
    }


@dataclass(frozen=True)
class PredictionArchiveV5:
    take_id: str
    seal_path: Path
    npz_path: Path
    implementation_revision: str
    baseline_vertices_m: np.ndarray
    global_vertices_m: np.ndarray
    v4_vertices_m: np.ndarray
    v5_vertices_m: np.ndarray
    faces: np.ndarray
    target_frames: np.ndarray
    update_supported: np.ndarray


def _load_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "baseline_vertices_m",
        "global_vertices_m",
        "v4_vertices_m",
        "v5_vertices_m",
        "faces",
        "target_frames",
        "source_frames",
        "history_start_frames",
        "history_end_frames",
        "update_supported",
        "update_accepted",
        "action_supported",
        "robot_history_supported",
        "correction_rms_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == required, "prediction array schema changed")
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_prediction_seal(
    seal_path: Path,
    execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> PredictionArchiveV5:
    """Validate a sealed prediction without reading a future target mesh."""

    validate_execution_protocol(
        execution_protocol, completion_protocol, parent_protocol
    )
    validate_author_source_manifest(source_manifest, parent_protocol)
    seal_path = Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal.get("schema_version") == 1, "prediction seal schema changed")
    _require(
        seal.get("artifact_kind") == PREDICTION_SEAL_KIND,
        "prediction seal kind changed",
    )
    _require(
        seal.get("seal_sha256") == prediction_seal_sha256(seal),
        "prediction seal checksum mismatch",
    )
    _require(
        seal.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "prediction execution protocol changed",
    )
    take_id = str(seal.get("take_id", ""))
    _require(take_id in TARGET_TAKE_IDS, "prediction take is outside target cohort")
    _require(seal.get("future_observation_used") is False, "future observation leaked")
    _require(
        seal.get("future_target_mesh_read") is False, "future target mesh was read"
    )
    _require(
        int(seal.get("future_target_mesh_read_count", -1)) == 0,
        "future target mesh access was recorded",
    )
    _require(
        seal.get("authorized_template_mesh_read") is True,
        "authorized template was not read",
    )
    _require(
        int(seal.get("authorized_template_mesh_read_count", -1)) == 1,
        "template read count changed",
    )
    _require(
        seal.get("target_metric_computed") is False,
        "target metric was computed during prediction",
    )
    _require(seal.get("implementation_clean") is True, "prediction checkout was dirty")
    revision = seal.get("implementation_revision")
    _require(_is_revision(revision), "prediction revision is invalid")
    _require(seal.get("upstream_commit") == UPSTREAM_COMMIT, "upstream commit changed")
    _require(
        dict(seal.get("checkpoint_sha256", {})) == CHECKPOINT_SHA256,
        "checkpoint bytes changed",
    )
    _require(seal.get("held_v8_accessed") is False, "held-v8 was accessed")

    stage_path = seal_path.parent / str(seal.get("input_stage_manifest", ""))
    _require(stage_path.is_file(), "input-stage manifest is missing")
    _require(
        file_sha256(stage_path) == seal.get("input_stage_manifest_file_sha256"),
        "input-stage manifest bytes changed",
    )
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage_validation = validate_input_stage(
        stage,
        execution_protocol,
        completion_protocol,
        parent_protocol,
        source_manifest,
    )
    _require(
        stage_validation["take_id"] == take_id, "prediction and input-stage take differ"
    )
    _require(
        stage_validation["input_stage_sha256"] == seal.get("input_stage_sha256"),
        "prediction input stage changed",
    )
    _require(
        seal.get("template_frame") == stage_validation["template_frame"],
        "prediction template frame changed",
    )

    npz_path = seal_path.parent / str(seal.get("prediction_npz", ""))
    _require(npz_path.is_file(), "prediction archive is missing")
    _require(
        file_sha256(npz_path) == seal.get("prediction_npz_sha256"),
        "prediction archive checksum mismatch",
    )
    arrays = _load_prediction_arrays(npz_path)
    baseline = np.asarray(arrays["baseline_vertices_m"], dtype=np.float64)
    candidates = {
        "global": np.asarray(arrays["global_vertices_m"], dtype=np.float64),
        "v4": np.asarray(arrays["v4_vertices_m"], dtype=np.float64),
        "v5": np.asarray(arrays["v5_vertices_m"], dtype=np.float64),
    }
    _require(
        baseline.ndim == 3 and baseline.shape[-1] == 3,
        "baseline vertices must be FxNx3",
    )
    _require(np.all(np.isfinite(baseline)), "baseline contains non-finite values")
    for name, values in candidates.items():
        _require(values.shape == baseline.shape, f"{name} candidate shape changed")
        _require(
            np.all(np.isfinite(values)), f"{name} candidate contains non-finite values"
        )
    faces = np.asarray(arrays["faces"])
    _require(faces.ndim == 2 and faces.shape[1] == 3, "faces must be Mx3")
    _require(np.issubdtype(faces.dtype, np.integer), "faces must be integer")
    frames = np.asarray(arrays["target_frames"], dtype=np.int64)
    _require(len(frames) == len(baseline), "prediction frame count changed")
    _require(
        len(frames) > 0 and int(frames[0]) == 6,
        "prediction does not begin at frame six",
    )
    _require(np.all(np.diff(frames) == 1), "prediction frames are not contiguous")
    _require(
        np.array_equal(arrays["source_frames"], frames - 1), "source frame is not f-1"
    )
    _require(
        np.array_equal(arrays["history_start_frames"], frames - 5),
        "history does not start at f-5",
    )
    _require(
        np.array_equal(arrays["history_end_frames"], frames - 1),
        "history does not end at f-1",
    )
    supported = np.asarray(arrays["update_supported"])
    _require(
        supported.shape == frames.shape and supported.dtype == np.bool_,
        "support mask changed",
    )
    robot_supported = np.asarray(arrays["robot_history_supported"])
    _require(
        robot_supported.shape == frames.shape and robot_supported.dtype == np.bool_,
        "robot support mask changed",
    )
    _require(
        not np.any(supported & ~robot_supported),
        "prediction used incomplete robot history",
    )
    for name, values in candidates.items():
        _require(
            np.array_equal(values[~supported], baseline[~supported]),
            f"unsupported {name} prediction is not exact fallback",
        )
        _require(
            int(seal.get(f"{name}_fallback_mismatch_count", -1)) == 0,
            f"{name} fallback mismatch was recorded",
        )
    _require(
        int(seal.get("predicted_frame_count", -1)) == len(frames),
        "predicted frame count changed",
    )
    _require(
        int(seal.get("supported_frame_count", -1)) == int(np.sum(supported)),
        "supported frame count changed",
    )
    expected_scales = execution_protocol["method"]
    _require(
        float(seal.get("global_effective_scale", -1.0))
        == float(expected_scales["global_effective_scale"]),
        "global prediction scale changed",
    )
    _require(
        float(seal.get("v4_effective_scale", -1.0))
        == float(expected_scales["v4_effective_scales"][take_id]),
        "V4 prediction scale changed",
    )
    _require(
        float(seal.get("v5_effective_scale", -1.0))
        == float(expected_scales["v5_effective_scales"][take_id]),
        "V5 prediction scale changed",
    )
    return PredictionArchiveV5(
        take_id=take_id,
        seal_path=seal_path,
        npz_path=npz_path,
        implementation_revision=str(revision),
        baseline_vertices_m=baseline,
        global_vertices_m=candidates["global"],
        v4_vertices_m=candidates["v4"],
        v5_vertices_m=candidates["v5"],
        faces=np.asarray(faces, dtype=np.int64),
        target_frames=frames,
        update_supported=np.asarray(supported, dtype=np.bool_),
    )


def build_prediction_barrier(
    seal_paths: Sequence[Path],
    execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require five immutable prediction archives before scoring is authorized."""

    archives = [
        validate_prediction_seal(
            path,
            execution_protocol,
            completion_protocol,
            parent_protocol,
            source_manifest,
        )
        for path in seal_paths
    ]
    _require(len(archives) == len(TARGET_TAKE_IDS), "prediction barrier is incomplete")
    by_take = {archive.take_id: archive for archive in archives}
    _require(len(by_take) == len(archives), "prediction seal is duplicated")
    _require(set(by_take) == set(TARGET_TAKE_IDS), "prediction seal cohort changed")
    revisions = {archive.implementation_revision for archive in archives}
    _require(len(revisions) == 1, "prediction revisions differ")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_BARRIER_KIND,
        "execution_protocol_sha256": execution_protocol["execution_protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "implementation_revision": next(iter(revisions)),
        "implementation_checkout_clean": True,
        "prediction_count": len(archives),
        "target_take_ids": list(TARGET_TAKE_IDS),
        "predictions": [
            {
                "take_id": take_id,
                "seal_sha256": json.loads(
                    by_take[take_id].seal_path.read_text(encoding="utf-8")
                )["seal_sha256"],
                "seal_file_sha256": file_sha256(by_take[take_id].seal_path),
                "prediction_npz_sha256": file_sha256(by_take[take_id].npz_path),
            }
            for take_id in TARGET_TAKE_IDS
        ],
        "authorized_template_mesh_count": len(archives),
        "future_target_mesh_accessed": False,
        "target_metric_computed": False,
        "scoring_authorized": True,
        "held_v8_accessed": False,
        "prediction_barrier_sha256": "",
    }
    payload["prediction_barrier_sha256"] = prediction_barrier_sha256(payload)
    validate_prediction_barrier(payload, execution_protocol, source_manifest)
    return payload


def validate_prediction_barrier(
    payload: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, "barrier schema changed")
    _require(
        payload.get("artifact_kind") == PREDICTION_BARRIER_KIND, "barrier kind changed"
    )
    _require(
        payload.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "barrier execution protocol changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_manifest["source_manifest_sha256"],
        "barrier source manifest changed",
    )
    observed = prediction_barrier_sha256(payload)
    _require(
        payload.get("prediction_barrier_sha256") == observed,
        "barrier checksum mismatch",
    )
    _require(
        _is_revision(payload.get("implementation_revision")),
        "barrier revision is invalid",
    )
    _require(
        payload.get("implementation_checkout_clean") is True,
        "barrier checkout was dirty",
    )
    _require(int(payload.get("prediction_count", -1)) == 5, "barrier count changed")
    _require(
        tuple(payload.get("target_take_ids", ())) == TARGET_TAKE_IDS,
        "barrier cohort changed",
    )
    rows = payload.get("predictions")
    _require(
        isinstance(rows, list) and len(rows) == 5, "barrier predictions are incomplete"
    )
    assert isinstance(rows, list)
    _require(
        tuple(row.get("take_id") for row in rows) == TARGET_TAKE_IDS,
        "barrier prediction order changed",
    )
    for row in rows:
        _require(_is_sha256(row.get("seal_sha256")), "barrier seal is unbound")
        _require(
            _is_sha256(row.get("seal_file_sha256")), "barrier seal file is unbound"
        )
        _require(
            _is_sha256(row.get("prediction_npz_sha256")),
            "barrier prediction is unbound",
        )
    _require(
        int(payload.get("authorized_template_mesh_count", -1)) == 5,
        "barrier template count changed",
    )
    _require(
        payload.get("future_target_mesh_accessed") is False,
        "target mesh opened before barrier",
    )
    _require(
        payload.get("target_metric_computed") is False,
        "target metric computed before barrier",
    )
    _require(
        payload.get("scoring_authorized") is True, "barrier did not authorize scoring"
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    return {"passed": True, "prediction_barrier_sha256": observed}


def score_one_prediction(
    archive: PredictionArchiveV5,
    active_frames: Sequence[int],
    mesh_loader: Callable[[int], tuple[np.ndarray, np.ndarray]],
    execution_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one sealed archive after the complete barrier has passed."""

    active = tuple(sorted(int(frame) for frame in active_frames if int(frame) >= 6))
    _require(bool(active), "target take has no scored action frames")
    frame_to_index = {
        int(frame): index for index, frame in enumerate(archive.target_frames)
    }
    _require(
        all(frame in frame_to_index for frame in active), "prediction frame is missing"
    )
    evaluation = execution_protocol["evaluation"]
    count = int(evaluation["surface_sample_count"])
    seed = int(evaluation["surface_sample_seed"])
    rows = []
    for frame in active:
        index = frame_to_index[frame]
        target_vertices, target_faces = mesh_loader(frame)
        target_sample = surface_sample(
            target_vertices, target_faces, count, seed + frame
        )
        scores = {}
        for name, vertices in (
            ("baseline", archive.baseline_vertices_m),
            ("global", archive.global_vertices_m),
            ("v4", archive.v4_vertices_m),
            ("v5", archive.v5_vertices_m),
        ):
            sample = surface_sample(vertices[index], archive.faces, count, seed + frame)
            scores[f"{name}_CD_UL1_mm"] = cd_ul1_mm(sample, target_sample)
        rows.append(
            {
                "target_frame": frame,
                "update_supported": bool(archive.update_supported[index]),
                **scores,
            }
        )
    result: dict[str, Any] = {
        "take_id": archive.take_id,
        "object_name": archive.take_id.rpartition("_T")[0],
        "scored_frame_count": len(rows),
        "supported_frame_count": sum(bool(row["update_supported"]) for row in rows),
        "frames": rows,
    }
    for name in ("baseline", "global", "v4", "v5"):
        result[f"{name}_mean_CD_UL1_mm"] = float(
            np.mean([row[f"{name}_CD_UL1_mm"] for row in rows])
        )
    return result


def evaluate_result(
    objects: Sequence[Mapping[str, Any]],
    public13: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate prospective V5-vs-V4 and combined official-18 gates."""

    _require(
        tuple(row.get("take_id") for row in objects) == TARGET_TAKE_IDS,
        "result cohort changed",
    )
    prospective_v4 = np.asarray(
        [row["v4_mean_CD_UL1_mm"] for row in objects], dtype=np.float64
    )
    prospective_v5 = np.asarray(
        [row["v5_mean_CD_UL1_mm"] for row in objects], dtype=np.float64
    )
    prospective_relative = (prospective_v4 - prospective_v5) / prospective_v4
    evaluation = execution_protocol["evaluation"]

    def upper(values: Sequence[float]) -> float:
        return paired_object_bootstrap_upper_difference(
            values,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
            upper_quantile=float(evaluation["bootstrap_upper_quantile"]),
        )

    prospective_upper = upper(prospective_v5 - prospective_v4)
    public_by_take = {str(row["take_id"]): row for row in public13["objects"]}
    prospective_by_take = {str(row["take_id"]): row for row in objects}
    _require(
        set(public_by_take) | set(prospective_by_take)
        == set(OFFICIAL18_TARGET_TAKE_IDS),
        "official-18 inventory changed",
    )
    v4_frame_values = []
    v5_frame_values = []
    v4_object_values = []
    v5_object_values = []
    for take_id in OFFICIAL18_TARGET_TAKE_IDS:
        if take_id in public_by_take:
            row = public_by_take[take_id]
            v4_frames = [float(frame["v4_all18_CD_UL1_mm"]) for frame in row["frames"]]
            v5_frames = list(v4_frames)
            v4_object = float(row["v4_all18_mean_CD_UL1_mm"])
            v5_object = v4_object
        else:
            row = prospective_by_take[take_id]
            v4_frames = [float(frame["v4_CD_UL1_mm"]) for frame in row["frames"]]
            v5_frames = [float(frame["v5_CD_UL1_mm"]) for frame in row["frames"]]
            v4_object = float(row["v4_mean_CD_UL1_mm"])
            v5_object = float(row["v5_mean_CD_UL1_mm"])
        v4_frame_values.extend(v4_frames)
        v5_frame_values.extend(v5_frames)
        v4_object_values.append(v4_object)
        v5_object_values.append(v5_object)
    v4_frames_array = np.asarray(v4_frame_values, dtype=np.float64)
    v5_frames_array = np.asarray(v5_frame_values, dtype=np.float64)
    v4_objects_array = np.asarray(v4_object_values, dtype=np.float64)
    v5_objects_array = np.asarray(v5_object_values, dtype=np.float64)
    official_upper = upper(v5_objects_array - v4_objects_array)
    gates = completion_protocol["gates"]
    prospective_gate = gates["prospective_v5_vs_v4"]
    prospective_improvement = float(
        (np.mean(prospective_v4) - np.mean(prospective_v5)) / np.mean(prospective_v4)
    )
    prospective_passed = bool(
        prospective_improvement
        > float(prospective_gate["object_balanced_relative_improvement_above"])
        and float(np.min(prospective_relative))
        >= float(prospective_gate["minimum_per_object_relative_improvement"])
        and prospective_upper
        < float(prospective_gate["paired_bootstrap_upper_difference_mm_below"])
    )
    official_gate = gates["official18"]
    v4_mean = float(np.mean(v4_frames_array))
    v5_mean = float(np.mean(v5_frames_array))
    official_passed = bool(
        (not bool(official_gate["v5_below_v4"]) or v5_mean < v4_mean)
        and (
            not bool(official_gate["v5_below_published_6_498_mm"])
            or v5_mean < PUBLISHED_KINECT_CD_UL1_MM
        )
        and official_upper
        < float(official_gate["paired_bootstrap_upper_v5_minus_v4_mm_below"])
    )
    return {
        "prospective_take_count": 5,
        "prospective_v4_object_balanced_CD_UL1_mm": float(np.mean(prospective_v4)),
        "prospective_v5_object_balanced_CD_UL1_mm": float(np.mean(prospective_v5)),
        "prospective_v5_vs_v4_relative_improvement": prospective_improvement,
        "prospective_v5_vs_v4_win_count": int(np.sum(prospective_v5 < prospective_v4)),
        "prospective_minimum_per_object_relative_improvement": float(
            np.min(prospective_relative)
        ),
        "prospective_bootstrap_upper_v5_minus_v4_mm": prospective_upper,
        "prospective_v5_vs_v4_gate_passed": prospective_passed,
        "official_take_count": 18,
        "official_scored_frame_count": len(v4_frame_values),
        "official18_v4_frame_balanced_CD_UL1_mm": v4_mean,
        "official18_v5_frame_balanced_CD_UL1_mm": v5_mean,
        "official18_v4_object_balanced_CD_UL1_mm": float(np.mean(v4_objects_array)),
        "official18_v5_object_balanced_CD_UL1_mm": float(np.mean(v5_objects_array)),
        "official18_bootstrap_upper_v5_minus_v4_mm": official_upper,
        "official18_below_published_6_498_mm": bool(
            v5_mean < PUBLISHED_KINECT_CD_UL1_MM
        ),
        "official18_gate_passed": official_passed,
        "all_v5_gates_passed": bool(prospective_passed and official_passed),
    }


def validate_result(
    payload: Mapping[str, Any],
    public13: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    barrier: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the sealed result and enforce its post-barrier boundary."""

    barrier_validation = validate_prediction_barrier(
        barrier,
        execution_protocol,
        source_manifest,
    )
    _require(payload.get("schema_version") == 1, "result schema changed")
    _require(payload.get("artifact_kind") == RESULT_KIND, "result kind changed")
    observed = result_sha256(payload)
    _require(payload.get("result_sha256") == observed, "result checksum mismatch")
    _require(
        payload.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "result execution protocol changed",
    )
    _require(
        payload.get("completion_protocol_sha256")
        == completion_protocol["protocol_sha256"],
        "result completion protocol changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_manifest["source_manifest_sha256"],
        "result source manifest changed",
    )
    _require(
        payload.get("prediction_barrier_sha256")
        == barrier_validation["prediction_barrier_sha256"],
        "result barrier changed",
    )
    _require(payload.get("prediction_barrier_passed") is True, "result barrier failed")
    _require(
        payload.get("target_mesh_access_before_barrier") is False,
        "result opened a target mesh before the barrier",
    )
    _require(
        payload.get("target_meshes_opened_after_complete_barrier") is True,
        "result did not record post-barrier target access",
    )
    _require(
        payload.get("future_observation_used_for_prediction") is False,
        "result used a future observation",
    )
    _require(
        payload.get("parameter_selection_from_this_cohort") is False,
        "result selected parameters from target outcomes",
    )
    _require(payload.get("replacement_used") is False, "result replaced a target")
    _require(
        payload.get("target_adaptation_used") is False,
        "result adapted to target outcomes",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    objects = payload.get("objects")
    _require(
        isinstance(objects, list) and len(objects) == len(TARGET_TAKE_IDS),
        "result cohort is incomplete",
    )
    assert isinstance(objects, list)
    _require(
        tuple(row.get("take_id") for row in objects) == TARGET_TAKE_IDS,
        "result cohort order changed",
    )
    for row, take_id in zip(objects, TARGET_TAKE_IDS, strict=True):
        _require(
            row.get("object_name") == take_id.rpartition("_T")[0],
            "result object identity changed",
        )
        frames = row.get("frames")
        _require(isinstance(frames, list) and bool(frames), "result frames are missing")
        assert isinstance(frames, list)
        _require(
            len(frames) == int(row.get("scored_frame_count", -1)),
            "result frame count changed",
        )
        target_frames = [int(frame.get("target_frame", -1)) for frame in frames]
        _require(
            target_frames == sorted(set(target_frames)),
            "result frames are not unique and sorted",
        )
        for name in ("baseline", "global", "v4", "v5"):
            values = np.asarray(
                [frame.get(f"{name}_CD_UL1_mm") for frame in frames],
                dtype=np.float64,
            )
            _require(
                np.all(np.isfinite(values)) and np.all(values >= 0.0),
                "result contains an invalid score",
            )
            _require(
                float(row.get(f"{name}_mean_CD_UL1_mm", -1.0))
                == float(np.mean(values)),
                f"{name} result mean does not reproduce frames",
            )
        target_meshes = row.get("target_meshes")
        _require(
            isinstance(target_meshes, list) and len(target_meshes) == len(frames),
            "target-mesh evidence is incomplete",
        )
        for mesh, frame in zip(target_meshes, target_frames, strict=True):
            _require(
                mesh.get("archive_member") == f"{take_id}/meshes/mesh-f{frame:05d}.obj",
                "scored target mesh changed",
            )
            _require(_is_sha256(mesh.get("sha256")), "scored target mesh is unbound")
            _require(int(mesh.get("byte_count", 0)) > 0, "scored target mesh is empty")
    expected_aggregate = evaluate_result(
        objects,
        public13,
        execution_protocol,
        completion_protocol,
    )
    _require(payload.get("aggregate") == expected_aggregate, "result aggregate changed")
    return {
        "passed": True,
        "result_sha256": observed,
        "all_v5_gates_passed": bool(expected_aggregate["all_v5_gates_passed"]),
    }


__all__ = [
    "EXECUTION_PROTOCOL_ID",
    "EXECUTION_PROTOCOL_KIND",
    "IMPLEMENTATION_FILE_PATHS",
    "INPUT_STAGE_KIND",
    "PREDICTION_BARRIER_KIND",
    "PREDICTION_SEAL_KIND",
    "PredictionArchiveV5",
    "RESULT_KIND",
    "TARGET_TAKE_IDS",
    "build_execution_protocol",
    "build_prediction_barrier",
    "canonical_payload_sha256",
    "evaluate_result",
    "execution_protocol_sha256",
    "file_sha256",
    "input_stage_sha256",
    "load_execution_protocol",
    "prediction_barrier_sha256",
    "prediction_seal_sha256",
    "result_sha256",
    "score_one_prediction",
    "validate_execution_protocol",
    "validate_input_stage",
    "validate_prediction_barrier",
    "validate_prediction_seal",
    "validate_result",
    "verify_implementation_files",
]
