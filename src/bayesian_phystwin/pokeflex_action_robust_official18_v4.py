"""Prospective completion protocol for PokeFlex's exact official 18-take split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
    OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
    OFFICIAL18_TARGET_TAKE_IDS,
    OFFICIAL_EVALUATOR_SHA256,
    PUBLISHED_KINECT_CD_UL1_MM,
    SELECTED_ARM,
    UPSTREAM_COMMIT,
    paired_object_bootstrap_upper_difference,
)

PROTOCOL_ID = "pokeflex-action-robust-official18-v4"
ARTIFACT_KIND = "PokeFlexActionRobustOfficial18V4Protocol"
COMPLETION_ARTIFACT_KIND = "PokeFlexActionRobustOfficial5V4ProspectiveResult"
SOURCE_MANIFEST_ARTIFACT_KIND = "PokeFlexOfficial5AuthorSourceManifest"
PREDICTION_BARRIER_ARTIFACT_KIND = "PokeFlexOfficial5V4PredictionBarrier"
EXPECTED_PROTOCOL_SHA256 = (
    "112a34e3af11f7fde44fa70de6b712d8b906ba81ea287b4897ad4a13692fed26"
)
CALIBRATION_SHA256 = "e94eeb9bdd2cc69e245b0bd48d843e5f64cb039e1eb02841e4a784cbe4dbc880"
CALIBRATION_FILE_SHA256 = (
    "00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110"
)
PUBLIC13_RESULT_FILE_SHA256 = (
    "9d6a3ce6e4d606485dcecfb12418199dc4bd3bbf43236e2d42f3f25f94a98a0e"
)
PUBLIC13_ARCHIVED_V3_FILE_SHA256 = (
    "619c46726aab0f7e81d2e943bd44820e521c9fe6285906add28af87203c15ebd"
)
ALL18_MULTIPLIERS = {
    "3dPrintedBunny": 3.0,
    "3dPrintedCylinder": 3.0,
    "3dPrintedHeart": 1.5,
    "3dPrintedPizza": 0.5,
    "3dPrintedPyramid": 1.0,
    "Beanbag": 4.0,
    "FoamCylinder": 3.0,
    "FoamDice": 4.0,
    "FoamHalfSphere": 2.0,
    "MemoryFoam": 1.5,
    "Pillow": 2.0,
    "PlushDice": 4.0,
    "PlushMoon": 4.0,
    "PlushOctopus": 3.0,
    "PlushTurtle": 4.0,
    "PlushVolleyball": 1.0,
    "Sponge": 1.5,
    "ToiletPaperRoll": 3.0,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one immutable file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    """Hash canonical JSON after excluding its self-referential digest."""

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
    """Return the canonical protocol digest."""

    return canonical_payload_sha256(payload, "protocol_sha256")


def completion_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical prospective-completion digest."""

    return canonical_payload_sha256(payload, "completion_sha256")


def source_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical pre-prediction source-manifest digest."""

    return canonical_payload_sha256(payload, "source_manifest_sha256")


def prediction_barrier_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical all-five prediction-barrier digest."""

    return canonical_payload_sha256(payload, "prediction_barrier_sha256")


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return _is_lower_hex(value, 64)


def validate_official18_v4_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable mixed-exposure official-18 completion lock."""

    _require(payload.get("schema_version") == 1, "protocol schema changed")
    _require(payload.get("artifact_kind") == ARTIFACT_KIND, "artifact kind changed")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    observed = protocol_sha256(payload)
    _require(payload.get("protocol_sha256") == observed, "protocol checksum mismatch")
    _require(observed == EXPECTED_PROTOCOL_SHA256, "registered protocol changed")

    cohort = payload.get("target_cohort")
    _require(isinstance(cohort, Mapping), "target cohort is missing")
    assert isinstance(cohort, Mapping)
    _require(
        tuple(cohort.get("take_ids", ())) == OFFICIAL18_TARGET_TAKE_IDS,
        "official 18-take cohort changed",
    )
    _require(
        tuple(cohort.get("previously_opened_take_ids", ()))
        == OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
        "opened public-13 inventory changed",
    )
    _require(
        tuple(cohort.get("prospective_take_ids", ()))
        == OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
        "prospective five-take inventory changed",
    )
    _require(cohort.get("replacement_allowed") is False, "replacement was enabled")

    method = payload.get("method")
    _require(isinstance(method, Mapping), "method lock is missing")
    assert isinstance(method, Mapping)
    _require(method.get("selected_arm") == SELECTED_ARM, "selected arm changed")
    _require(float(method.get("base_scale", -1.0)) == 0.125, "base scale changed")
    _require(
        method.get("calibration_sha256") == CALIBRATION_SHA256,
        "calibration changed",
    )
    _require(
        method.get("calibration_file_sha256") == CALIBRATION_FILE_SHA256,
        "calibration bytes changed",
    )
    _require(
        dict(method.get("multipliers", {})) == ALL18_MULTIPLIERS,
        "multiplier map changed",
    )
    _require(
        method.get("target_outcome_adaptation") == "forbidden", "adaptation enabled"
    )
    _require(
        method.get("unsupported_frame_action") == "byte-identical released checkpoint",
        "fallback changed",
    )

    archive = payload.get("archived_public13")
    _require(isinstance(archive, Mapping), "public-13 binding is missing")
    assert isinstance(archive, Mapping)
    _require(
        archive.get("result_file_sha256") == PUBLIC13_RESULT_FILE_SHA256,
        "public-13 result changed",
    )
    _require(
        archive.get("archived_v3_result_file_sha256")
        == PUBLIC13_ARCHIVED_V3_FILE_SHA256,
        "archived V3 result changed",
    )
    _require(archive.get("outcomes_previously_opened") is True, "exposure was hidden")

    upstream = payload.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream lock is missing")
    assert isinstance(upstream, Mapping)
    _require(upstream.get("code_commit") == UPSTREAM_COMMIT, "upstream commit changed")
    _require(
        upstream.get("evaluator_sha256") == OFFICIAL_EVALUATOR_SHA256,
        "official evaluator changed",
    )
    _require(
        dict(upstream.get("checkpoint_sha256", {})) == CHECKPOINT_SHA256,
        "checkpoint bytes changed",
    )

    custody = payload.get("custody")
    _require(isinstance(custody, Mapping), "custody lock is missing")
    assert isinstance(custody, Mapping)
    _require(
        int(custody.get("required_prospective_prediction_seal_count", -1)) == 5,
        "prediction barrier count changed",
    )
    _require(
        custody.get("target_mesh_access_before_barrier") == "forbidden",
        "target custody weakened",
    )
    _require(
        custody.get("all_prediction_revisions_must_match") is True,
        "mixed revisions allowed",
    )
    _require(
        custody.get("author_source_manifest_required") is True,
        "source manifest disabled",
    )
    _require(
        custody.get("author_source_manifest_artifact_kind")
        == SOURCE_MANIFEST_ARTIFACT_KIND,
        "source manifest contract changed",
    )
    _require(
        custody.get("prediction_barrier_artifact_kind")
        == PREDICTION_BARRIER_ARTIFACT_KIND,
        "prediction barrier contract changed",
    )

    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation lock is missing")
    assert isinstance(evaluation, Mapping)
    _require(evaluation.get("primary_metric") == "CD_UL1_mm", "metric changed")
    _require(
        int(evaluation.get("surface_sample_count", -1)) == 10000, "sample count changed"
    )
    _require(
        int(evaluation.get("surface_sample_seed", -1)) == 20260720,
        "sample seed changed",
    )
    _require(
        float(evaluation.get("published_kinect_CD_UL1_mm", -1.0))
        == PUBLISHED_KINECT_CD_UL1_MM,
        "published reference changed",
    )
    return {
        "protocol_sha256": observed,
        "target_take_count": len(OFFICIAL18_TARGET_TAKE_IDS),
        "prospective_take_count": len(OFFICIAL18_MISSING_PUBLIC_TAKE_IDS),
    }


def load_official18_v4_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the registered protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_official18_v4_protocol(payload)
    return payload


def load_archived_public13_result(
    path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the immutable opened public-13 component without re-scoring it."""

    validate_official18_v4_protocol(protocol)
    _require(
        file_sha256(path) == PUBLIC13_RESULT_FILE_SHA256, "public-13 bytes changed"
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("parameter_selection_from_this_cohort") is False,
        "public-13 selected parameters",
    )
    _require(
        payload.get("future_or_missing_official_takes_accessed") is False,
        "missing takes were accessed",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    _require(
        payload.get("calibration_sha256") == CALIBRATION_SHA256,
        "public-13 calibration changed",
    )
    _require(
        payload.get("calibration_file_sha256") == CALIBRATION_FILE_SHA256,
        "public-13 calibration bytes changed",
    )
    _require(
        tuple(row.get("take_id") for row in payload.get("objects", ()))
        == OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
        "public-13 result inventory changed",
    )
    return payload


def validate_author_source_manifest(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact author-supplied inputs before prediction begins."""

    validate_official18_v4_protocol(protocol)
    _require(payload.get("schema_version") == 1, "source manifest schema changed")
    _require(
        payload.get("artifact_kind") == SOURCE_MANIFEST_ARTIFACT_KIND,
        "source manifest kind changed",
    )
    _require(
        payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "source manifest protocol changed",
    )
    observed = source_manifest_sha256(payload)
    _require(
        payload.get("source_manifest_sha256") == observed,
        "source manifest checksum mismatch",
    )
    _require(
        payload.get("created_before_prediction") is True, "manifest followed prediction"
    )
    _require(
        payload.get("target_geometry_decoded") is False, "target geometry was decoded"
    )
    _require(
        payload.get("target_metric_computed") is False, "target metric was computed"
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    takes = payload.get("takes")
    _require(isinstance(takes, list) and len(takes) == 5, "source cohort is incomplete")
    assert isinstance(takes, list)
    for row, take_id in zip(takes, OFFICIAL18_MISSING_PUBLIC_TAKE_IDS, strict=True):
        _require(row.get("take_id") == take_id, "source take order changed")
        _require(
            _is_sha256(row.get("source_payload_sha256")), "source payload is unbound"
        )
        _require(
            _is_sha256(row.get("member_manifest_sha256")), "member manifest is unbound"
        )
        _require(
            row.get("official_take_identity_verified") is True,
            "take identity is unverified",
        )
        _require(
            row.get("required_streams_present") is True, "required stream is missing"
        )
        _require(
            row.get("camera_panel_sufficient") is True, "camera panel is insufficient"
        )
        _require(
            row.get("evaluator_compatible") is True, "take is evaluator-incompatible"
        )
        _require(int(row.get("episode_length", 0)) >= 7, "episode is too short")
    return {"source_manifest_sha256": observed, "take_count": len(takes)}


def validate_prediction_barrier(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all five sealed predictions before any target score is opened."""

    validate_official18_v4_protocol(protocol)
    source_validation = validate_author_source_manifest(source_manifest, protocol)
    _require(payload.get("schema_version") == 1, "barrier schema changed")
    _require(
        payload.get("artifact_kind") == PREDICTION_BARRIER_ARTIFACT_KIND,
        "barrier kind changed",
    )
    _require(
        payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "barrier protocol changed",
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_validation["source_manifest_sha256"],
        "barrier source manifest changed",
    )
    observed = prediction_barrier_sha256(payload)
    _require(
        payload.get("prediction_barrier_sha256") == observed,
        "barrier checksum mismatch",
    )
    _require(payload.get("passed") is True, "prediction barrier failed")
    _require(
        payload.get("implementation_checkout_clean") is True, "implementation was dirty"
    )
    _require(
        payload.get("target_mesh_accessed") is False, "target opened before barrier"
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    revision = payload.get("implementation_revision")
    _require(
        _is_lower_hex(revision, 40),
        "implementation revision is invalid",
    )
    predictions = payload.get("predictions")
    _require(
        isinstance(predictions, list) and len(predictions) == 5,
        "barrier cohort is incomplete",
    )
    assert isinstance(predictions, list)
    allowed_status = {
        "prediction_success",
        "exact_fallback",
        "retained_technical_failure",
    }
    for row, take_id in zip(
        predictions, OFFICIAL18_MISSING_PUBLIC_TAKE_IDS, strict=True
    ):
        _require(row.get("take_id") == take_id, "barrier take order changed")
        _require(
            row.get("implementation_revision") == revision,
            "prediction revision changed",
        )
        _require(
            row.get("future_observation_used") is False, "future observation leaked"
        )
        _require(row.get("status") in allowed_status, "prediction status is invalid")
        _require(_is_sha256(row.get("seal_sha256")), "prediction seal is unbound")
        _require(
            _is_sha256(row.get("prediction_file_sha256")), "prediction file is unbound"
        )
    return {"prediction_barrier_sha256": observed, "prediction_count": len(predictions)}


def _validate_scored_object(row: Mapping[str, Any], expected_take_id: str) -> None:
    _require(row.get("take_id") == expected_take_id, "completion take order changed")
    object_name = expected_take_id.rpartition("_T")[0]
    _require(
        row.get("object_name") == object_name, "completion object identity changed"
    )
    frames = row.get("frames")
    _require(isinstance(frames, list) and bool(frames), "completion frames are missing")
    assert isinstance(frames, list)
    _require(
        len(frames) == int(row.get("scored_frame_count", -1)), "frame count changed"
    )
    target_frames = [int(frame.get("target_frame", -1)) for frame in frames]
    _require(
        target_frames == sorted(set(target_frames)),
        "target frames are not unique and sorted",
    )
    for key, frame_key in (
        ("baseline_mean_CD_UL1_mm", "baseline_CD_UL1_mm"),
        ("global_mean_CD_UL1_mm", "global_CD_UL1_mm"),
        ("v4_all18_mean_CD_UL1_mm", "v4_all18_CD_UL1_mm"),
    ):
        values = np.asarray(
            [frame.get(frame_key) for frame in frames], dtype=np.float64
        )
        _require(
            np.all(np.isfinite(values)) and np.all(values >= 0.0), "invalid frame score"
        )
        _require(
            abs(float(row.get(key, -1.0)) - float(np.mean(values))) <= 1e-12,
            f"{key} does not reproduce frames",
        )


def validate_prospective_completion(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    prediction_barrier: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sealed five-take result before combining it with public-13."""

    validate_official18_v4_protocol(protocol)
    source_validation = validate_author_source_manifest(source_manifest, protocol)
    barrier_validation = validate_prediction_barrier(
        prediction_barrier,
        protocol,
        source_manifest,
    )
    _require(payload.get("schema_version") == 1, "completion schema changed")
    _require(
        payload.get("artifact_kind") == COMPLETION_ARTIFACT_KIND,
        "completion kind changed",
    )
    _require(
        payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "completion protocol changed",
    )
    observed = completion_sha256(payload)
    _require(
        payload.get("completion_sha256") == observed, "completion checksum mismatch"
    )
    _require(
        payload.get("source_manifest_sha256")
        == source_validation["source_manifest_sha256"],
        "completion source manifest changed",
    )
    _require(
        payload.get("prediction_barrier_sha256")
        == barrier_validation["prediction_barrier_sha256"],
        "completion barrier changed",
    )
    _require(
        payload.get("prediction_barrier_passed") is True, "prediction barrier failed"
    )
    _require(
        payload.get("target_mesh_access_before_barrier") is False,
        "target opened before barrier",
    )
    _require(
        payload.get("parameter_selection_from_this_cohort") is False,
        "target selected parameters",
    )
    _require(
        payload.get("future_observation_used_for_prediction") is False,
        "future observation leaked",
    )
    _require(payload.get("replacement_used") is False, "replacement was used")
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    objects = payload.get("objects")
    _require(
        isinstance(objects, list) and len(objects) == 5,
        "completion cohort is incomplete",
    )
    assert isinstance(objects, list)
    for row, take_id in zip(objects, OFFICIAL18_MISSING_PUBLIC_TAKE_IDS, strict=True):
        _validate_scored_object(row, take_id)
    return {"completion_sha256": observed, "prospective_take_count": len(objects)}


def _ordered_official18_objects(
    public13: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    rows = [*public13["objects"], *completion["objects"]]
    by_take = {str(row["take_id"]): row for row in rows}
    _require(len(by_take) == 18, "official result contains duplicate takes")
    _require(
        set(by_take) == set(OFFICIAL18_TARGET_TAKE_IDS),
        "official result cohort changed",
    )
    return [by_take[take_id] for take_id in OFFICIAL18_TARGET_TAKE_IDS]


def evaluate_official18_v4(
    public13_path: Path,
    source_manifest: Mapping[str, Any],
    prediction_barrier: Mapping[str, Any],
    completion: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine immutable public-13 scores with the sealed prospective five."""

    public13 = load_archived_public13_result(public13_path, protocol)
    validate_prospective_completion(
        completion,
        protocol,
        source_manifest,
        prediction_barrier,
    )
    ordered = _ordered_official18_objects(public13, completion)
    for row in ordered:
        _validate_scored_object(row, str(row["take_id"]))

    def frame_values(key: str) -> np.ndarray:
        return np.asarray(
            [frame[key] for row in ordered for frame in row["frames"]],
            dtype=np.float64,
        )

    baseline_frames = frame_values("baseline_CD_UL1_mm")
    global_frames = frame_values("global_CD_UL1_mm")
    candidate_frames = frame_values("v4_all18_CD_UL1_mm")
    baseline_object = np.asarray(
        [row["baseline_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    global_object = np.asarray(
        [row["global_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    candidate_object = np.asarray(
        [row["v4_all18_mean_CD_UL1_mm"] for row in ordered], dtype=np.float64
    )
    prospective = [
        ordered[OFFICIAL18_TARGET_TAKE_IDS.index(take_id)]
        for take_id in OFFICIAL18_MISSING_PUBLIC_TAKE_IDS
    ]
    prospective_baseline = np.asarray(
        [row["baseline_mean_CD_UL1_mm"] for row in prospective], dtype=np.float64
    )
    prospective_candidate = np.asarray(
        [row["v4_all18_mean_CD_UL1_mm"] for row in prospective], dtype=np.float64
    )
    gates = protocol["gates"]
    bootstrap = gates["bootstrap"]

    def upper(differences: Sequence[float]) -> float:
        return paired_object_bootstrap_upper_difference(
            differences,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
            upper_quantile=float(bootstrap["upper_quantile"]),
        )

    baseline_mean = float(np.mean(baseline_frames))
    global_mean = float(np.mean(global_frames))
    candidate_mean = float(np.mean(candidate_frames))
    prospective_relative = (
        prospective_baseline - prospective_candidate
    ) / prospective_baseline
    full_baseline_upper = upper(candidate_object - baseline_object)
    full_global_upper = upper(candidate_object - global_object)
    prospective_upper = upper(prospective_candidate - prospective_baseline)
    reproduction_passed = bool(
        abs(baseline_mean - PUBLISHED_KINECT_CD_UL1_MM)
        <= float(gates["baseline_reproduction"]["absolute_tolerance_mm"])
    )
    direct_gate = gates["direct_full18"]
    prospective_gate = gates["prospective_missing5"]
    direct_passed = bool(
        reproduction_passed
        and (
            not bool(direct_gate["candidate_below_published_reference"])
            or candidate_mean < PUBLISHED_KINECT_CD_UL1_MM
        )
        and (
            not bool(direct_gate["candidate_below_global"])
            or candidate_mean < global_mean
        )
        and full_baseline_upper
        < float(direct_gate["paired_bootstrap_upper_difference_mm_below"])
        and full_global_upper
        < float(direct_gate["paired_bootstrap_upper_difference_mm_below"])
    )
    prospective_passed = bool(
        float(
            (np.mean(prospective_baseline) - np.mean(prospective_candidate))
            / np.mean(prospective_baseline)
        )
        > float(prospective_gate["object_balanced_relative_improvement_above"])
        and float(np.min(prospective_relative))
        >= float(prospective_gate["minimum_per_object_relative_improvement"])
        and prospective_upper
        < float(prospective_gate["paired_bootstrap_upper_difference_mm_below"])
    )
    return {
        "official_take_count": 18,
        "prospective_take_count": 5,
        "scored_frame_count": int(len(baseline_frames)),
        "baseline_frame_balanced_CD_UL1_mm": baseline_mean,
        "global_frame_balanced_CD_UL1_mm": global_mean,
        "v4_frame_balanced_CD_UL1_mm": candidate_mean,
        "baseline_object_balanced_CD_UL1_mm": float(np.mean(baseline_object)),
        "global_object_balanced_CD_UL1_mm": float(np.mean(global_object)),
        "v4_object_balanced_CD_UL1_mm": float(np.mean(candidate_object)),
        "full18_v4_wins_vs_baseline": int(np.sum(candidate_object < baseline_object)),
        "full18_bootstrap_upper_v4_minus_baseline_mm": full_baseline_upper,
        "full18_bootstrap_upper_v4_minus_global_mm": full_global_upper,
        "prospective_v4_wins_vs_baseline": int(
            np.sum(prospective_candidate < prospective_baseline)
        ),
        "prospective_minimum_relative_improvement": float(np.min(prospective_relative)),
        "prospective_bootstrap_upper_v4_minus_baseline_mm": prospective_upper,
        "baseline_reproduction_passed": reproduction_passed,
        "direct_full18_benchmark_gate_passed": direct_passed,
        "prospective_missing5_transfer_gate_passed": prospective_passed,
        "published_full_split_improvement_authorized": bool(
            direct_passed and prospective_passed
        ),
        "fully_prospective_official18_claim_authorized": False,
    }
