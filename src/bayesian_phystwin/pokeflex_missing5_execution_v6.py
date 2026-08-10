"""Custody-safe V6 augmentation of sealed PokeFlex missing-five predictions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_conservative_shrinkage_target import (
    OFFICIAL18_TARGET_TAKE_IDS,
    PUBLISHED_KINECT_CD_UL1_MM,
    cd_ul1_mm,
    paired_object_bootstrap_upper_difference,
    surface_sample,
)
from .pokeflex_missing5_causal_scale_v6 import (
    V5_EFFECTIVE_SCALES,
    causal_scale_vertices,
    select_causal_scale,
    validate_causal_scale_model,
)
from .pokeflex_missing5_execution_v5 import (
    TARGET_TAKE_IDS,
    PredictionArchiveV5,
    canonical_payload_sha256,
    file_sha256,
)
from .pokeflex_missing5_execution_v5 import (
    score_one_prediction as score_one_v5_prediction,
)
from .pokeflex_missing5_execution_v5 import (
    validate_execution_protocol as validate_v5_execution_protocol,
)
from .pokeflex_missing5_execution_v5 import (
    validate_prediction_seal as validate_v5_prediction_seal,
)

EXECUTION_PROTOCOL_ID = "pokeflex-missing5-execution-v6"
EXECUTION_PROTOCOL_KIND = "PokeFlexMissingFiveV6ExecutionProtocol"
PREDICTION_SEAL_KIND = "PokeFlexMissingFiveV6PredictionSeal"
PREDICTION_BARRIER_KIND = "PokeFlexMissingFiveV6PredictionBarrier"
RESULT_KIND = "PokeFlexMissingFiveV6ProspectiveResult"
SOURCE_RESULT_KIND = "PokeFlexMissingFiveCausalScaleV6SourceResult"

IMPLEMENTATION_FILE_PATHS = (
    "scripts/held/run_pokeflex_missing5_v6.py",
    "src/bayesian_phystwin/pokeflex_missing5_causal_scale_v6.py",
    "src/bayesian_phystwin/pokeflex_missing5_execution_v6.py",
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


def execution_protocol_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "execution_protocol_sha256")


def prediction_seal_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "seal_sha256")


def prediction_barrier_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "prediction_barrier_sha256")


def result_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(payload, "result_sha256")


def validate_source_result(
    payload: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the target-disjoint V6 source decision."""

    model_validation = validate_causal_scale_model(model)
    _require(payload.get("schema_version") == 1, "source result schema changed")
    _require(
        payload.get("artifact_kind") == SOURCE_RESULT_KIND, "source result changed"
    )
    observed = canonical_payload_sha256(payload, "result_sha256")
    _require(payload.get("result_sha256") == observed, "source result checksum changed")
    _require(
        payload.get("model_sha256") == model_validation["model_sha256"],
        "source result model changed",
    )
    _require(payload.get("source_gate", {}).get("passed") is True, "source gate failed")
    _require(
        payload.get("official_target_outcomes_used") is False,
        "source result used an official target outcome",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    return {"passed": True, "result_sha256": observed}


def build_execution_protocol(
    v5_execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    model: Mapping[str, Any],
    source_result: Mapping[str, Any],
    *,
    locked_at_utc: str,
    model_file_sha256: str,
    source_result_file_sha256: str,
    implementation_file_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Build the pre-target V6 augmentation lock."""

    validate_v5_execution_protocol(
        v5_execution_protocol, completion_protocol, parent_protocol
    )
    model_validation = validate_causal_scale_model(model)
    source_validation = validate_source_result(source_result, model)
    _require(_is_sha256(model_file_sha256), "model file hash is invalid")
    _require(
        _is_sha256(source_result_file_sha256), "source result file hash is invalid"
    )
    _require(
        set(implementation_file_sha256s) == set(IMPLEMENTATION_FILE_PATHS),
        "implementation inventory changed",
    )
    _require(
        all(_is_sha256(value) for value in implementation_file_sha256s.values()),
        "implementation file hash is invalid",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": EXECUTION_PROTOCOL_KIND,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "locked_at_utc": locked_at_utc,
        "parent_v5_execution_protocol_sha256": v5_execution_protocol[
            "execution_protocol_sha256"
        ],
        "causal_scale_model_sha256": model_validation["model_sha256"],
        "causal_scale_model_file_sha256": model_file_sha256,
        "source_result_sha256": source_validation["result_sha256"],
        "source_result_file_sha256": source_result_file_sha256,
        "target_take_ids": list(TARGET_TAKE_IDS),
        "method": {
            "parent_arm": "sealed V5 prediction",
            "promoted_objects": ["3dPrintedCylinder", "3dPrintedHeart"],
            "v5_effective_scales": {
                take_id: V5_EFFECTIVE_SCALES[take_id.rpartition("_T")[0]]
                for take_id in TARGET_TAKE_IDS
            },
            "allowed_decision_inputs": [
                "normalized target phase",
                "prefix update RMS",
                "physical-prior motion RMS",
                "update-to-motion RMS ratio",
                "update/prior-motion cosine",
            ],
            "rejected_frame_action": "byte-identical V5 prediction",
            "unsupported_frame_action": "byte-identical released checkpoint",
        },
        "evaluation": {
            "metric": "CD_UL1_mm",
            "surface_sample_count": int(
                v5_execution_protocol["evaluation"]["surface_sample_count"]
            ),
            "surface_sample_seed": int(
                v5_execution_protocol["evaluation"]["surface_sample_seed"]
            ),
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20_260_810,
            "bootstrap_upper_quantile": 0.975,
            "prospective_v6_vs_v5": {
                "object_balanced_relative_improvement_above": 0.0,
                "minimum_per_object_relative_improvement": 0.0,
                "paired_bootstrap_upper_difference_mm_at_most": 0.0,
            },
            "official18": {
                "v6_below_v5": True,
                "v6_below_published_6_498_mm": True,
                "paired_bootstrap_upper_v6_minus_v5_mm_at_most": 0.0,
            },
        },
        "custody": {
            "required_prediction_count": 5,
            "all_prediction_revisions_must_match": True,
            "parent_v5_prediction_required": True,
            "prediction_and_scoring_are_separate": True,
            "target_mesh_access_before_barrier": "forbidden",
            "replacement_allowed": False,
            "target_adaptation": "forbidden",
        },
        "implementation_file_sha256s": dict(
            sorted(implementation_file_sha256s.items())
        ),
        "official_target_outcomes_used_to_build_protocol": False,
        "held_v8_accessed": False,
        "execution_protocol_sha256": "",
    }
    payload["execution_protocol_sha256"] = execution_protocol_sha256(payload)
    validate_execution_protocol(
        payload,
        v5_execution_protocol,
        completion_protocol,
        parent_protocol,
        model,
        source_result,
        bind_registered_digest=False,
    )
    return payload


def validate_execution_protocol(
    payload: Mapping[str, Any],
    v5_execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    model: Mapping[str, Any],
    source_result: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate the V6 lock and every parent binding."""

    validate_v5_execution_protocol(
        v5_execution_protocol, completion_protocol, parent_protocol
    )
    model_validation = validate_causal_scale_model(model)
    source_validation = validate_source_result(source_result, model)
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
        from .pokeflex_missing5_execution_v6_lock import (
            EXPECTED_EXECUTION_PROTOCOL_SHA256,
        )

        _require(
            observed == EXPECTED_EXECUTION_PROTOCOL_SHA256, "registered V6 lock changed"
        )
    _require(
        payload.get("parent_v5_execution_protocol_sha256")
        == v5_execution_protocol["execution_protocol_sha256"],
        "parent V5 execution changed",
    )
    _require(
        payload.get("causal_scale_model_sha256") == model_validation["model_sha256"],
        "V6 model binding changed",
    )
    _require(
        payload.get("source_result_sha256") == source_validation["result_sha256"],
        "V6 source result binding changed",
    )
    _require(
        _is_sha256(payload.get("causal_scale_model_file_sha256")),
        "model file is unbound",
    )
    _require(
        _is_sha256(payload.get("source_result_file_sha256")),
        "source result file is unbound",
    )
    _require(
        tuple(payload.get("target_take_ids", ())) == TARGET_TAKE_IDS,
        "target cohort changed",
    )
    method = payload.get("method")
    _require(isinstance(method, Mapping), "V6 method is missing")
    assert isinstance(method, Mapping)
    expected_scales = {
        take_id: V5_EFFECTIVE_SCALES[take_id.rpartition("_T")[0]]
        for take_id in TARGET_TAKE_IDS
    }
    _require(method.get("v5_effective_scales") == expected_scales, "V5 scales changed")
    _require(
        method.get("rejected_frame_action") == "byte-identical V5 prediction",
        "rejected-frame fallback changed",
    )
    _require(
        method.get("unsupported_frame_action") == "byte-identical released checkpoint",
        "unsupported-frame fallback changed",
    )
    custody = payload.get("custody")
    _require(isinstance(custody, Mapping), "V6 custody policy is missing")
    assert isinstance(custody, Mapping)
    _require(custody.get("required_prediction_count") == 5, "barrier size changed")
    _require(custody.get("replacement_allowed") is False, "replacement was allowed")
    _require(
        custody.get("target_adaptation") == "forbidden", "target adaptation was allowed"
    )
    files = payload.get("implementation_file_sha256s")
    _require(isinstance(files, Mapping), "implementation hashes are missing")
    assert isinstance(files, Mapping)
    _require(
        set(files) == set(IMPLEMENTATION_FILE_PATHS), "implementation inventory changed"
    )
    _require(
        all(_is_sha256(value) for value in files.values()),
        "implementation hash is invalid",
    )
    _require(
        payload.get("official_target_outcomes_used_to_build_protocol") is False,
        "official target outcome entered the V6 lock",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    return {"passed": True, "execution_protocol_sha256": observed}


def verify_implementation_files(
    payload: Mapping[str, Any], repository_root: Path
) -> None:
    files = payload["implementation_file_sha256s"]
    for relative_path in IMPLEMENTATION_FILE_PATHS:
        path = repository_root / relative_path
        _require(path.is_file(), f"implementation file is missing: {relative_path}")
        _require(
            file_sha256(path) == files[relative_path],
            f"implementation file changed: {relative_path}",
        )


def load_execution_protocol(
    path: Path,
    v5_execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    model_path: Path,
    source_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the V6 lock, model, and source result with file bindings."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    source_result = json.loads(Path(source_result_path).read_text(encoding="utf-8"))
    validate_execution_protocol(
        payload,
        v5_execution_protocol,
        completion_protocol,
        parent_protocol,
        model,
        source_result,
    )
    _require(
        file_sha256(model_path) == payload["causal_scale_model_file_sha256"],
        "V6 model file changed",
    )
    _require(
        file_sha256(source_result_path) == payload["source_result_file_sha256"],
        "V6 source result file changed",
    )
    return payload, model, source_result


def apply_causal_scale_sequence(
    model: Mapping[str, Any],
    *,
    object_name: str,
    baseline_vertices_m: np.ndarray,
    v5_vertices_m: np.ndarray,
    target_frames: np.ndarray,
    update_supported: np.ndarray,
    update_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Apply V6 to a sealed V5 sequence using only prefix-derived rows."""

    validate_causal_scale_model(model)
    baseline = np.asarray(baseline_vertices_m, dtype=np.float64)
    v5 = np.asarray(v5_vertices_m, dtype=np.float64)
    frames = np.asarray(target_frames, dtype=np.int64)
    supported = np.asarray(update_supported, dtype=np.bool_)
    _require(baseline.ndim == 3 and baseline.shape[-1] == 3, "baseline shape changed")
    _require(v5.shape == baseline.shape, "V5 shape changed")
    _require(
        frames.shape == supported.shape == (len(baseline),), "V6 frame shape changed"
    )
    _require(
        np.all(np.isfinite(baseline)) and np.all(np.isfinite(v5)),
        "V6 input is non-finite",
    )
    rows = {int(row.get("target_frame", -1)): row for row in update_rows}
    _require(len(rows) == len(update_rows), "V6 update frame repeats")
    _require(
        set(rows) == set(int(frame) for frame in frames),
        "V6 update rows are incomplete",
    )
    v5_scale = V5_EFFECTIVE_SCALES[object_name]
    values = []
    scales = []
    admitted = []
    lower_gains = []
    distances = []
    radii = []
    diagnostics = []
    for index, frame_value in enumerate(frames):
        frame = int(frame_value)
        row = rows[frame]
        is_supported = bool(supported[index])
        _require(
            not is_supported or bool(row.get("accepted")),
            "supported V6 row lacks an accepted update",
        )
        decision = select_causal_scale(
            model,
            object_name=object_name,
            update=row,
            target_frame=frame,
            maximum_frame=int(frames[-1]),
            supported=is_supported,
        )
        correction_field = (
            (v5[index] - baseline[index]) / v5_scale
            if is_supported
            else np.zeros_like(baseline[index])
        )
        value = causal_scale_vertices(
            baseline[index],
            correction_field,
            v5[index],
            decision,
            supported=is_supported,
        )
        values.append(value)
        scales.append(decision.selected_scale)
        admitted.append(decision.admitted)
        lower_gains.append(
            np.nan
            if decision.predicted_lower_gain_mm is None
            else decision.predicted_lower_gain_mm
        )
        distances.append(
            np.nan
            if decision.minimum_source_distance is None
            else decision.minimum_source_distance
        )
        radii.append(
            np.nan if decision.support_radius is None else decision.support_radius
        )
        diagnostics.append(
            {
                **dict(row),
                "target_frame": frame,
                "update_supported": is_supported,
                "v5_effective_scale": decision.baseline_scale,
                "v6_candidate_effective_scale": decision.candidate_scale,
                "v6_selected_effective_scale": decision.selected_scale,
                "v6_candidate_admitted": decision.admitted,
                "v6_decision_reason": decision.reason,
                "v6_predicted_lower_gain_mm": decision.predicted_lower_gain_mm,
                "v6_minimum_source_distance": decision.minimum_source_distance,
                "v6_support_radius": decision.support_radius,
            }
        )
    arrays = {
        "v6_vertices_m": np.asarray(values, dtype=np.float64),
        "target_frames": frames.copy(),
        "selected_scale": np.asarray(scales, dtype=np.float64),
        "candidate_admitted": np.asarray(admitted, dtype=np.bool_),
        "predicted_lower_gain_mm": np.asarray(lower_gains, dtype=np.float64),
        "minimum_source_distance": np.asarray(distances, dtype=np.float64),
        "support_radius": np.asarray(radii, dtype=np.float64),
    }
    _require(
        np.array_equal(arrays["v6_vertices_m"][~supported], baseline[~supported]),
        "unsupported V6 frame changed the checkpoint",
    )
    rejected = supported & ~arrays["candidate_admitted"]
    _require(
        np.array_equal(arrays["v6_vertices_m"][rejected], v5[rejected]),
        "rejected V6 frame changed V5",
    )
    return arrays, diagnostics


@dataclass(frozen=True)
class PredictionArchiveV6:
    take_id: str
    seal_path: Path
    npz_path: Path
    implementation_revision: str
    parent_v5: PredictionArchiveV5
    v6_vertices_m: np.ndarray
    target_frames: np.ndarray
    selected_scale: np.ndarray
    candidate_admitted: np.ndarray


def _load_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "v6_vertices_m",
        "target_frames",
        "selected_scale",
        "candidate_admitted",
        "predicted_lower_gain_mm",
        "minimum_source_distance",
        "support_radius",
    }
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == required, "V6 prediction array schema changed")
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_prediction_seal(
    seal_path: Path,
    parent_v5_seal_path: Path,
    execution_protocol: Mapping[str, Any],
    v5_execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> PredictionArchiveV6:
    """Validate one V6 augmentation without reading a target mesh."""

    validate_execution_protocol(
        execution_protocol,
        v5_execution_protocol,
        completion_protocol,
        parent_protocol,
        model,
        source_result,
    )
    parent_archive = validate_v5_prediction_seal(
        parent_v5_seal_path,
        v5_execution_protocol,
        completion_protocol,
        parent_protocol,
        source_manifest,
    )
    seal_path = Path(seal_path).resolve()
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal.get("schema_version") == 1, "V6 seal schema changed")
    _require(seal.get("artifact_kind") == PREDICTION_SEAL_KIND, "V6 seal kind changed")
    _require(
        seal.get("seal_sha256") == prediction_seal_sha256(seal),
        "V6 seal checksum changed",
    )
    _require(
        seal.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "V6 seal execution changed",
    )
    _require(
        seal.get("source_manifest_sha256") == source_manifest["source_manifest_sha256"],
        "V6 seal source manifest changed",
    )
    _require(seal.get("take_id") == parent_archive.take_id, "V5/V6 take differs")
    _require(
        seal.get("object_name") == parent_archive.take_id.rpartition("_T")[0],
        "V6 seal object changed",
    )
    parent_seal = json.loads(Path(parent_v5_seal_path).read_text(encoding="utf-8"))
    _require(
        seal.get("parent_v5_seal_sha256") == parent_seal["seal_sha256"],
        "parent V5 seal digest changed",
    )
    _require(
        seal.get("parent_v5_seal_file_sha256") == file_sha256(parent_v5_seal_path),
        "parent V5 seal bytes changed",
    )
    _require(
        seal.get("parent_v5_prediction_npz_sha256")
        == file_sha256(parent_archive.npz_path),
        "parent V5 prediction bytes changed",
    )
    _require(
        seal.get("causal_scale_model_sha256")
        == execution_protocol["causal_scale_model_sha256"],
        "V6 seal model changed",
    )
    _require(
        seal.get("causal_scale_model_file_sha256")
        == execution_protocol["causal_scale_model_file_sha256"],
        "V6 seal model file changed",
    )
    _require(
        seal.get("source_result_sha256") == execution_protocol["source_result_sha256"],
        "V6 seal source result changed",
    )
    _require(
        seal.get("source_result_file_sha256")
        == execution_protocol["source_result_file_sha256"],
        "V6 seal source result file changed",
    )
    _require(
        seal.get("input_stage_sha256") == parent_seal["input_stage_sha256"],
        "V6 seal input stage changed",
    )
    _require(seal.get("future_observation_used") is False, "future observation leaked")
    _require(
        seal.get("future_target_mesh_read") is False,
        "target mesh opened during V6 prediction",
    )
    _require(
        int(seal.get("future_target_mesh_read_count", -1)) == 0,
        "V6 seal records target-mesh access",
    )
    _require(
        seal.get("target_metric_computed") is False,
        "target metric computed during V6 prediction",
    )
    _require(
        seal.get("implementation_clean") is True, "V6 prediction checkout was dirty"
    )
    revision = seal.get("implementation_revision")
    _require(_is_revision(revision), "V6 prediction revision is invalid")
    _require(
        revision == parent_archive.implementation_revision, "V5/V6 revisions differ"
    )
    _require(seal.get("held_v8_accessed") is False, "held-v8 was accessed")
    npz_path = seal_path.parent / str(seal.get("prediction_npz", ""))
    _require(npz_path.is_file(), "V6 prediction archive is missing")
    _require(
        file_sha256(npz_path) == seal.get("prediction_npz_sha256"),
        "V6 archive checksum changed",
    )
    arrays = _load_prediction_arrays(npz_path)
    frames = np.asarray(arrays["target_frames"], dtype=np.int64)
    _require(
        np.array_equal(frames, parent_archive.target_frames), "V6 target frames changed"
    )
    _require(
        int(seal.get("predicted_frame_count", -1)) == len(frames),
        "V6 predicted frame count changed",
    )
    diagnostics = seal.get("decisions")
    _require(isinstance(diagnostics, list), "V6 decisions are missing")
    assert isinstance(diagnostics, list)
    expected_arrays, expected_diagnostics = apply_causal_scale_sequence(
        model,
        object_name=parent_archive.take_id.rpartition("_T")[0],
        baseline_vertices_m=parent_archive.baseline_vertices_m,
        v5_vertices_m=parent_archive.v5_vertices_m,
        target_frames=parent_archive.target_frames,
        update_supported=parent_archive.update_supported,
        update_rows=diagnostics,
    )
    for name, expected in expected_arrays.items():
        observed = np.asarray(arrays[name])
        _require(
            np.array_equal(observed, expected, equal_nan=True),
            f"V6 {name} does not reproduce the sealed decisions",
        )
    _require(diagnostics == expected_diagnostics, "V6 decision record changed")
    admitted = np.asarray(arrays["candidate_admitted"], dtype=np.bool_)
    _require(
        int(seal.get("candidate_admission_count", -1)) == int(np.sum(admitted)),
        "V6 admission count changed",
    )
    _require(
        int(seal.get("unsupported_fallback_mismatch_count", -1)) == 0,
        "checkpoint fallback mismatch recorded",
    )
    _require(
        int(seal.get("rejected_fallback_mismatch_count", -1)) == 0,
        "V5 fallback mismatch recorded",
    )
    return PredictionArchiveV6(
        take_id=parent_archive.take_id,
        seal_path=seal_path,
        npz_path=npz_path,
        implementation_revision=str(revision),
        parent_v5=parent_archive,
        v6_vertices_m=np.asarray(arrays["v6_vertices_m"], dtype=np.float64),
        target_frames=frames,
        selected_scale=np.asarray(arrays["selected_scale"], dtype=np.float64),
        candidate_admitted=admitted,
    )


def build_prediction_barrier(
    seal_paths: Sequence[Path],
    parent_v5_seal_paths: Sequence[Path],
    execution_protocol: Mapping[str, Any],
    v5_execution_protocol: Mapping[str, Any],
    completion_protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Require all five V6 augmentations before target scoring."""

    _require(len(seal_paths) == len(parent_v5_seal_paths), "V5/V6 seal counts differ")
    archives = [
        validate_prediction_seal(
            seal_path,
            parent_path,
            execution_protocol,
            v5_execution_protocol,
            completion_protocol,
            parent_protocol,
            source_manifest,
            model,
            source_result,
        )
        for seal_path, parent_path in zip(seal_paths, parent_v5_seal_paths, strict=True)
    ]
    by_take = {archive.take_id: archive for archive in archives}
    _require(len(archives) == len(TARGET_TAKE_IDS), "V6 barrier is incomplete")
    _require(set(by_take) == set(TARGET_TAKE_IDS), "V6 barrier cohort changed")
    revisions = {archive.implementation_revision for archive in archives}
    _require(len(revisions) == 1, "V6 prediction revisions differ")
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
                "parent_v5_seal_file_sha256": file_sha256(
                    by_take[take_id].parent_v5.seal_path
                ),
                "parent_v5_prediction_npz_sha256": file_sha256(
                    by_take[take_id].parent_v5.npz_path
                ),
            }
            for take_id in TARGET_TAKE_IDS
        ],
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
    _require(payload.get("schema_version") == 1, "V6 barrier schema changed")
    _require(
        payload.get("artifact_kind") == PREDICTION_BARRIER_KIND,
        "V6 barrier kind changed",
    )
    _require(
        payload.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "V6 barrier execution changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_manifest["source_manifest_sha256"],
        "V6 barrier source manifest changed",
    )
    observed = prediction_barrier_sha256(payload)
    _require(
        payload.get("prediction_barrier_sha256") == observed,
        "V6 barrier checksum changed",
    )
    _require(
        _is_revision(payload.get("implementation_revision")),
        "V6 barrier revision is invalid",
    )
    _require(
        payload.get("implementation_checkout_clean") is True,
        "V6 barrier checkout was dirty",
    )
    _require(payload.get("prediction_count") == 5, "V6 barrier count changed")
    _require(
        tuple(payload.get("target_take_ids", ())) == TARGET_TAKE_IDS,
        "V6 barrier cohort changed",
    )
    rows = payload.get("predictions")
    _require(
        isinstance(rows, list) and len(rows) == 5,
        "V6 barrier predictions are incomplete",
    )
    assert isinstance(rows, list)
    _require(
        tuple(row.get("take_id") for row in rows) == TARGET_TAKE_IDS,
        "V6 barrier order changed",
    )
    for row in rows:
        for name in (
            "seal_sha256",
            "seal_file_sha256",
            "prediction_npz_sha256",
            "parent_v5_seal_file_sha256",
            "parent_v5_prediction_npz_sha256",
        ):
            _require(_is_sha256(row.get(name)), f"V6 barrier {name} is unbound")
    _require(
        payload.get("future_target_mesh_accessed") is False,
        "target mesh opened before V6 barrier",
    )
    _require(
        payload.get("target_metric_computed") is False,
        "target metric computed before V6 barrier",
    )
    _require(
        payload.get("scoring_authorized") is True,
        "V6 barrier did not authorize scoring",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    return {"passed": True, "prediction_barrier_sha256": observed}


def score_one_prediction(
    archive: PredictionArchiveV6,
    active_frames: Sequence[int],
    mesh_loader: Callable[[int], tuple[np.ndarray, np.ndarray]],
    execution_protocol: Mapping[str, Any],
    v5_execution_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Score V5 and V6 together after the V6 barrier passes."""

    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def cached_mesh_loader(frame: int) -> tuple[np.ndarray, np.ndarray]:
        if frame not in mesh_cache:
            mesh_cache[frame] = mesh_loader(frame)
        return mesh_cache[frame]

    base = score_one_v5_prediction(
        archive.parent_v5,
        active_frames,
        cached_mesh_loader,
        v5_execution_protocol,
    )
    frame_to_index = {
        int(frame): index for index, frame in enumerate(archive.target_frames)
    }
    evaluation = execution_protocol["evaluation"]
    count = int(evaluation["surface_sample_count"])
    seed = int(evaluation["surface_sample_seed"])
    for row in base["frames"]:
        frame = int(row["target_frame"])
        index = frame_to_index[frame]
        target_vertices, target_faces = cached_mesh_loader(frame)
        target_sample = surface_sample(
            target_vertices, target_faces, count, seed + frame
        )
        sample = surface_sample(
            archive.v6_vertices_m[index], archive.parent_v5.faces, count, seed + frame
        )
        row["v6_CD_UL1_mm"] = cd_ul1_mm(sample, target_sample)
        row["v6_candidate_admitted"] = bool(archive.candidate_admitted[index])
        row["v6_selected_scale"] = float(archive.selected_scale[index])
    base["v6_mean_CD_UL1_mm"] = float(
        np.mean([row["v6_CD_UL1_mm"] for row in base["frames"]])
    )
    base["v6_candidate_admission_count"] = sum(
        bool(row["v6_candidate_admitted"]) for row in base["frames"]
    )
    return base


def evaluate_result(
    objects: Sequence[Mapping[str, Any]],
    public13: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen V6-vs-V5 prospective and official-18 gates."""

    _require(
        tuple(row.get("take_id") for row in objects) == TARGET_TAKE_IDS,
        "V6 result cohort changed",
    )
    v5_prospective = np.asarray(
        [row["v5_mean_CD_UL1_mm"] for row in objects], dtype=np.float64
    )
    v6_prospective = np.asarray(
        [row["v6_mean_CD_UL1_mm"] for row in objects], dtype=np.float64
    )
    relative = (v5_prospective - v6_prospective) / v5_prospective
    evaluation = execution_protocol["evaluation"]

    def upper(values: Sequence[float]) -> float:
        return paired_object_bootstrap_upper_difference(
            values,
            replicates=int(evaluation["bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
            upper_quantile=float(evaluation["bootstrap_upper_quantile"]),
        )

    prospective_upper = upper(v6_prospective - v5_prospective)
    public_by_take = {str(row["take_id"]): row for row in public13["objects"]}
    prospective_by_take = {str(row["take_id"]): row for row in objects}
    _require(
        set(public_by_take) | set(prospective_by_take)
        == set(OFFICIAL18_TARGET_TAKE_IDS),
        "official-18 inventory changed",
    )
    v5_frames = []
    v6_frames = []
    v5_objects = []
    v6_objects = []
    for take_id in OFFICIAL18_TARGET_TAKE_IDS:
        if take_id in public_by_take:
            row = public_by_take[take_id]
            frames = [float(frame["v4_all18_CD_UL1_mm"]) for frame in row["frames"]]
            v5_frames.extend(frames)
            v6_frames.extend(frames)
            value = float(row["v4_all18_mean_CD_UL1_mm"])
            v5_objects.append(value)
            v6_objects.append(value)
        else:
            row = prospective_by_take[take_id]
            v5_frames.extend(float(frame["v5_CD_UL1_mm"]) for frame in row["frames"])
            v6_frames.extend(float(frame["v6_CD_UL1_mm"]) for frame in row["frames"])
            v5_objects.append(float(row["v5_mean_CD_UL1_mm"]))
            v6_objects.append(float(row["v6_mean_CD_UL1_mm"]))
    v5_frames_array = np.asarray(v5_frames, dtype=np.float64)
    v6_frames_array = np.asarray(v6_frames, dtype=np.float64)
    v5_objects_array = np.asarray(v5_objects, dtype=np.float64)
    v6_objects_array = np.asarray(v6_objects, dtype=np.float64)
    official_upper = upper(v6_objects_array - v5_objects_array)
    prospective_gate = evaluation["prospective_v6_vs_v5"]
    prospective_improvement = float(
        (np.mean(v5_prospective) - np.mean(v6_prospective)) / np.mean(v5_prospective)
    )
    prospective_passed = bool(
        prospective_improvement
        > float(prospective_gate["object_balanced_relative_improvement_above"])
        and float(np.min(relative))
        >= float(prospective_gate["minimum_per_object_relative_improvement"])
        and prospective_upper
        <= float(prospective_gate["paired_bootstrap_upper_difference_mm_at_most"])
    )
    official_gate = evaluation["official18"]
    v5_mean = float(np.mean(v5_frames_array))
    v6_mean = float(np.mean(v6_frames_array))
    official_passed = bool(
        (not bool(official_gate["v6_below_v5"]) or v6_mean < v5_mean)
        and (
            not bool(official_gate["v6_below_published_6_498_mm"])
            or v6_mean < PUBLISHED_KINECT_CD_UL1_MM
        )
        and official_upper
        <= float(official_gate["paired_bootstrap_upper_v6_minus_v5_mm_at_most"])
    )
    return {
        "prospective_take_count": 5,
        "prospective_v5_object_balanced_CD_UL1_mm": float(np.mean(v5_prospective)),
        "prospective_v6_object_balanced_CD_UL1_mm": float(np.mean(v6_prospective)),
        "prospective_v6_vs_v5_relative_improvement": prospective_improvement,
        "prospective_v6_vs_v5_win_count": int(np.sum(v6_prospective < v5_prospective)),
        "prospective_v6_vs_v5_tie_count": int(np.sum(v6_prospective == v5_prospective)),
        "prospective_v6_vs_v5_regression_count": int(
            np.sum(v6_prospective > v5_prospective)
        ),
        "prospective_minimum_per_object_relative_improvement": float(np.min(relative)),
        "prospective_bootstrap_upper_v6_minus_v5_mm": prospective_upper,
        "prospective_v6_vs_v5_gate_passed": prospective_passed,
        "official_take_count": 18,
        "official_scored_frame_count": len(v5_frames),
        "official18_v5_frame_balanced_CD_UL1_mm": v5_mean,
        "official18_v6_frame_balanced_CD_UL1_mm": v6_mean,
        "official18_v5_object_balanced_CD_UL1_mm": float(np.mean(v5_objects_array)),
        "official18_v6_object_balanced_CD_UL1_mm": float(np.mean(v6_objects_array)),
        "official18_bootstrap_upper_v6_minus_v5_mm": official_upper,
        "official18_below_published_6_498_mm": bool(
            v6_mean < PUBLISHED_KINECT_CD_UL1_MM
        ),
        "official18_gate_passed": official_passed,
        "all_v6_gates_passed": bool(prospective_passed and official_passed),
    }


def validate_result(
    payload: Mapping[str, Any],
    public13: Mapping[str, Any],
    execution_protocol: Mapping[str, Any],
    barrier: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the V6 aggregate and enforce post-barrier target access."""

    barrier_validation = validate_prediction_barrier(
        barrier, execution_protocol, source_manifest
    )
    _require(payload.get("schema_version") == 1, "V6 result schema changed")
    _require(payload.get("artifact_kind") == RESULT_KIND, "V6 result kind changed")
    observed = result_sha256(payload)
    _require(payload.get("result_sha256") == observed, "V6 result checksum changed")
    _require(
        payload.get("execution_protocol_sha256")
        == execution_protocol["execution_protocol_sha256"],
        "V6 result execution changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_manifest["source_manifest_sha256"],
        "V6 result source manifest changed",
    )
    _require(
        payload.get("prediction_barrier_sha256")
        == barrier_validation["prediction_barrier_sha256"],
        "V6 result barrier changed",
    )
    _require(
        payload.get("prediction_barrier_passed") is True, "V6 result barrier failed"
    )
    _require(
        payload.get("target_mesh_access_before_barrier") is False,
        "target mesh opened before V6 barrier",
    )
    _require(
        payload.get("target_meshes_opened_after_complete_barrier") is True,
        "V6 result lacks post-barrier target access",
    )
    _require(
        payload.get("future_observation_used_for_prediction") is False,
        "future observation entered V6",
    )
    _require(
        payload.get("parameter_selection_from_this_cohort") is False,
        "V6 selected from target outcomes",
    )
    _require(payload.get("replacement_used") is False, "V6 replaced a target")
    _require(
        payload.get("target_adaptation_used") is False, "V6 adapted to target outcomes"
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    objects = payload.get("objects")
    _require(
        isinstance(objects, list) and len(objects) == 5,
        "V6 result cohort is incomplete",
    )
    assert isinstance(objects, list)
    _require(
        tuple(row.get("take_id") for row in objects) == TARGET_TAKE_IDS,
        "V6 result order changed",
    )
    for row, take_id in zip(objects, TARGET_TAKE_IDS, strict=True):
        _require(
            row.get("object_name") == take_id.rpartition("_T")[0],
            "V6 result object changed",
        )
        frames = row.get("frames")
        _require(
            isinstance(frames, list) and bool(frames), "V6 result frames are missing"
        )
        assert isinstance(frames, list)
        _require(
            len(frames) == int(row.get("scored_frame_count", -1)),
            "V6 result frame count changed",
        )
        target_frames = [int(frame.get("target_frame", -1)) for frame in frames]
        _require(
            target_frames == sorted(set(target_frames)),
            "V6 result frames are not unique and sorted",
        )
        for name in ("baseline", "global", "v4", "v5", "v6"):
            values = np.asarray(
                [frame.get(f"{name}_CD_UL1_mm") for frame in frames], dtype=np.float64
            )
            _require(
                np.all(np.isfinite(values)) and np.all(values >= 0.0),
                "V6 result score is invalid",
            )
            _require(
                float(row.get(f"{name}_mean_CD_UL1_mm", -1.0))
                == float(np.mean(values)),
                f"{name} mean changed",
            )
        target_meshes = row.get("target_meshes")
        _require(
            isinstance(target_meshes, list) and len(target_meshes) == len(frames),
            "V6 target-mesh evidence is incomplete",
        )
        assert isinstance(target_meshes, list)
        for mesh, frame in zip(target_meshes, target_frames, strict=True):
            _require(
                mesh.get("archive_member") == f"{take_id}/meshes/mesh-f{frame:05d}.obj",
                "V6 scored target mesh changed",
            )
            _require(_is_sha256(mesh.get("sha256")), "V6 target mesh is unbound")
            _require(int(mesh.get("byte_count", 0)) > 0, "V6 target mesh is empty")
    expected = evaluate_result(objects, public13, execution_protocol)
    _require(payload.get("aggregate") == expected, "V6 aggregate changed")
    return {
        "passed": True,
        "result_sha256": observed,
        "all_v6_gates_passed": bool(expected["all_v6_gates_passed"]),
    }


__all__ = [
    "EXECUTION_PROTOCOL_ID",
    "EXECUTION_PROTOCOL_KIND",
    "IMPLEMENTATION_FILE_PATHS",
    "PREDICTION_BARRIER_KIND",
    "PREDICTION_SEAL_KIND",
    "PredictionArchiveV6",
    "RESULT_KIND",
    "apply_causal_scale_sequence",
    "build_execution_protocol",
    "build_prediction_barrier",
    "evaluate_result",
    "execution_protocol_sha256",
    "load_execution_protocol",
    "prediction_barrier_sha256",
    "prediction_seal_sha256",
    "result_sha256",
    "score_one_prediction",
    "validate_execution_protocol",
    "validate_prediction_barrier",
    "validate_prediction_seal",
    "validate_result",
    "validate_source_result",
    "verify_implementation_files",
]
