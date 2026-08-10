"""Frozen causal per-frame scale policy for the missing PokeFlex targets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .pokeflex_action_robust_all18 import SOURCE_FIELD
from .pokeflex_missing5_scale import OFFICIAL_TARGET_TAKES, SOURCE_TAKES

MODEL_KIND = "PokeFlexMissingFiveCausalScaleV6Model"
MODEL_ID = "pokeflex-missing5-causal-scale-v6"
FEATURE_NAMES = (
    "normalized_target_phase",
    "log_rms_update_m",
    "log_prior_motion_rms_m",
    "log_correction_to_prior_motion_ratio",
    "correction_prior_motion_cosine",
)
V5_EFFECTIVE_SCALES = {
    "3dPrintedCylinder": 0.25,
    "3dPrintedHeart": 0.1875,
    "3dPrintedPizza": 0.125,
    "Pillow": 0.125,
    "Sponge": 0.125,
}
V6_CANDIDATE_SCALES = {
    "3dPrintedCylinder": 0.375,
    "3dPrintedHeart": 0.25,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def model_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical V6 model digest."""

    return _canonical_sha256(payload, "model_sha256")


def _finite_float(value: object, name: str) -> float:
    result = float(str(value))
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _positive_log(value: object, name: str) -> float:
    result = _finite_float(value, name)
    _require(result >= 0.0, f"{name} is negative")
    return math.log(max(result, 1e-12))


def causal_scale_feature(
    update: Mapping[str, Any],
    *,
    target_frame: int,
    maximum_frame: int,
) -> np.ndarray:
    """Extract the outcome-independent feature vector available before a target."""

    _require(maximum_frame >= 1, "maximum frame is invalid")
    _require(1 <= target_frame <= maximum_frame, "target frame is invalid")
    cosine_value = update.get("correction_prior_motion_cosine")
    cosine = 0.0 if cosine_value is None else _finite_float(cosine_value, "cosine")
    _require(-1.000001 <= cosine <= 1.000001, "cosine is outside [-1, 1]")
    feature = np.asarray(
        [
            target_frame / maximum_frame,
            _positive_log(update.get("rms_update_m", 0.0), "rms update"),
            _positive_log(
                update.get("prior_motion_rms_m", 0.0), "prior motion rms"
            ),
            _positive_log(
                update.get("correction_to_prior_motion_ratio", 0.0),
                "correction-to-prior ratio",
            ),
            cosine,
        ],
        dtype=np.float64,
    )
    _require(feature.shape == (len(FEATURE_NAMES),), "feature schema changed")
    _require(np.all(np.isfinite(feature)), "causal feature is non-finite")
    return feature


@dataclass(frozen=True)
class CausalScalePolicyConfig:
    """Fixed source-neighborhood and deployment settings."""

    neighbors_per_source_take: int = 20
    gain_margin_mm: float = 0.001
    support_distance_quantile: float = 0.90
    robust_scale_factor: float = 1.4826
    minimum_robust_scale: float = 1e-8

    def __post_init__(self) -> None:
        _require(self.neighbors_per_source_take >= 1, "neighbor count is invalid")
        _require(self.gain_margin_mm >= 0.0, "gain margin is negative")
        _require(
            0.0 < self.support_distance_quantile < 1.0,
            "support quantile is invalid",
        )
        _require(self.robust_scale_factor > 0.0, "robust scale factor is invalid")
        _require(self.minimum_robust_scale > 0.0, "minimum scale is invalid")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalScaleDecision:
    """One closed, target-free scale decision."""

    object_name: str
    baseline_scale: float
    candidate_scale: float
    selected_scale: float
    admitted: bool
    reason: str
    predicted_lower_gain_mm: float | None
    minimum_source_distance: float | None
    support_radius: float | None


def _object_name(take_id: str) -> str:
    name, separator, number = str(take_id).rpartition("_T")
    _require(bool(separator) and number.isdigit(), "invalid PokeFlex take id")
    return name


def _score_key(scale: float) -> str:
    return f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"


def extract_source_frame_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract causal features and source-only gains from one sealed V5 artifact."""

    _require(
        payload.get("artifact_kind")
        == "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "unexpected source artifact kind",
    )
    _require(payload.get("future_observation_used") is False, "future input was used")
    _require(
        payload.get("official_target_outcome_used") is False,
        "official target outcome was used",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    take = payload.get("take")
    _require(isinstance(take, Mapping), "source take metadata is missing")
    assert isinstance(take, Mapping)
    take_id = str(take.get("id", ""))
    object_name = _object_name(take_id)
    expected_takes = SOURCE_TAKES.get(object_name)
    _require(expected_takes is not None and take_id in expected_takes, "unknown source take")
    maximum_frame = int(take.get("maximum_frame", 0))
    _require(maximum_frame >= 6, "source maximum frame is invalid")
    updates_payload = payload.get("updates")
    _require(
        isinstance(updates_payload, Sequence)
        and not isinstance(updates_payload, (str, bytes)),
        "source updates are missing",
    )
    assert isinstance(updates_payload, Sequence) and not isinstance(
        updates_payload, (str, bytes)
    )
    updates: dict[int, Mapping[str, Any]] = {}
    for update in updates_payload:
        _require(isinstance(update, Mapping), "source update is invalid")
        assert isinstance(update, Mapping)
        target_frame = int(update.get("target_frame", -1))
        _require(target_frame not in updates, "source update frame repeats")
        updates[target_frame] = update
    targets = payload.get("targets")
    _require(
        isinstance(targets, list) and bool(targets), "source target rows are missing"
    )
    assert isinstance(targets, list)
    if object_name not in V6_CANDIDATE_SCALES:
        return []
    baseline_scale = V5_EFFECTIVE_SCALES[object_name]
    candidate_scale = V6_CANDIDATE_SCALES[object_name]
    rows: list[dict[str, Any]] = []
    for target in targets:
        _require(isinstance(target, Mapping), "source target row is invalid")
        assert isinstance(target, Mapping)
        target_frame = int(target.get("target_frame", -1))
        update = updates.get(target_frame)
        _require(update is not None, "source target has no causal update row")
        assert update is not None
        baseline = _finite_float(target.get(_score_key(baseline_scale)), "baseline score")
        candidate = _finite_float(
            target.get(_score_key(candidate_scale)), "candidate score"
        )
        _require(baseline > 0.0 and candidate >= 0.0, "source score is invalid")
        accepted = bool(update.get("accepted"))
        _require(
            not accepted or bool(update.get("action_supported")),
            "accepted source update lacks action support",
        )
        rows.append(
            {
                "take_id": take_id,
                "object_name": object_name,
                "target_frame": target_frame,
                "accepted": accepted,
                "features": causal_scale_feature(
                    update,
                    target_frame=target_frame,
                    maximum_frame=maximum_frame,
                ).tolist(),
                "baseline_CD_UL1_mm": baseline,
                "candidate_CD_UL1_mm": candidate,
                "candidate_gain_mm": baseline - candidate,
            }
        )
    _require(
        [row["target_frame"] for row in rows]
        == sorted({row["target_frame"] for row in rows}),
        "source target frames are not unique and sorted",
    )
    return rows


def fit_object_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: CausalScalePolicyConfig | None = None,
    expected_take_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit one deterministic group-conservative source-neighborhood model."""

    cfg = config or CausalScalePolicyConfig()
    accepted = [row for row in rows if bool(row.get("accepted"))]
    _require(bool(accepted), "accepted source row bank is empty")
    objects = {str(row.get("object_name", "")) for row in accepted}
    _require(len(objects) == 1, "source rows mix physical objects")
    object_name = next(iter(objects))
    _require(object_name in V6_CANDIDATE_SCALES, "object is not promoted by V6")
    take_ids = sorted({str(row.get("take_id", "")) for row in accepted})
    expected = (
        sorted(str(value) for value in expected_take_ids)
        if expected_take_ids is not None
        else list(SOURCE_TAKES[object_name])
    )
    _require(
        take_ids == expected and len(take_ids) >= 2, "source take partition changed"
    )
    features = np.asarray([row.get("features") for row in accepted], dtype=np.float64)
    _require(
        features.ndim == 2 and features.shape[1] == len(FEATURE_NAMES),
        "source feature bank shape changed",
    )
    _require(np.all(np.isfinite(features)), "source feature bank is non-finite")
    center = np.median(features, axis=0)
    mad = np.median(np.abs(features - center), axis=0)
    scale = np.where(
        mad > cfg.minimum_robust_scale,
        cfg.robust_scale_factor * mad,
        1.0,
    )
    standardized = (features - center) / scale
    group_labels = np.asarray([str(row["take_id"]) for row in accepted])
    distance2 = ((standardized[:, None] - standardized[None, :]) ** 2).sum(axis=2)
    distance2[group_labels[:, None] == group_labels[None, :]] = np.inf
    cross_group_distance = np.sqrt(np.min(distance2, axis=1))
    _require(
        np.all(np.isfinite(cross_group_distance)),
        "source groups have no cross-group support",
    )
    support_radius = float(
        np.quantile(cross_group_distance, cfg.support_distance_quantile)
    )
    groups = []
    for take_id in take_ids:
        indices = np.flatnonzero(group_labels == take_id)
        _require(
            len(indices) >= cfg.neighbors_per_source_take,
            "source group is smaller than the frozen neighbor count",
        )
        groups.append(
            {
                "take_id": take_id,
                "standardized_features": standardized[indices].tolist(),
                "candidate_gains_mm": [
                    _finite_float(accepted[index]["candidate_gain_mm"], "source gain")
                    for index in indices
                ],
            }
        )
    return {
        "object_name": object_name,
        "baseline_effective_scale": V5_EFFECTIVE_SCALES[object_name],
        "candidate_effective_scale": V6_CANDIDATE_SCALES[object_name],
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "support_radius": support_radius,
        "source_group_count": len(groups),
        "source_row_count": len(accepted),
        "groups": groups,
    }


def _validate_object_model(
    object_model: Mapping[str, Any], config: CausalScalePolicyConfig
) -> None:
    object_name = str(object_model.get("object_name", ""))
    _require(object_name in V6_CANDIDATE_SCALES, "model object is not promoted")
    _require(
        float(object_model.get("baseline_effective_scale", -1.0))
        == V5_EFFECTIVE_SCALES[object_name],
        "model baseline scale changed",
    )
    _require(
        float(object_model.get("candidate_effective_scale", -1.0))
        == V6_CANDIDATE_SCALES[object_name],
        "model candidate scale changed",
    )
    center = np.asarray(object_model.get("feature_center"), dtype=np.float64)
    scale = np.asarray(object_model.get("feature_scale"), dtype=np.float64)
    _require(
        center.shape == scale.shape == (len(FEATURE_NAMES),),
        "model feature transform changed",
    )
    _require(
        np.all(np.isfinite(center)) and np.all(np.isfinite(scale)) and np.all(scale > 0),
        "model feature transform is invalid",
    )
    radius = _finite_float(object_model.get("support_radius"), "support radius")
    _require(radius > 0.0, "support radius is not positive")
    groups = object_model.get("groups")
    _require(isinstance(groups, list), "model source groups are missing")
    assert isinstance(groups, list)
    take_ids = []
    for group in groups:
        _require(isinstance(group, Mapping), "model source group is invalid")
        assert isinstance(group, Mapping)
        take_ids.append(str(group.get("take_id", "")))
    _require(take_ids == list(SOURCE_TAKES[object_name]), "model source groups changed")
    row_count = 0
    for group in groups:
        _require(isinstance(group, Mapping), "model source group is invalid")
        assert isinstance(group, Mapping)
        features = np.asarray(group.get("standardized_features"), dtype=np.float64)
        gains = np.asarray(group.get("candidate_gains_mm"), dtype=np.float64)
        _require(
            features.ndim == 2 and features.shape[1] == len(FEATURE_NAMES),
            "model source feature shape changed",
        )
        _require(gains.shape == (len(features),), "model source gains changed")
        _require(
            len(features) >= config.neighbors_per_source_take,
            "model source group is too small",
        )
        _require(
            np.all(np.isfinite(features)) and np.all(np.isfinite(gains)),
            "model source rows are non-finite",
        )
        row_count += len(features)
    _require(
        int(object_model.get("source_group_count", -1)) == len(groups),
        "model source group count changed",
    )
    _require(
        int(object_model.get("source_row_count", -1)) == row_count,
        "model source row count changed",
    )


def validate_causal_scale_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete frozen V6 model and its target boundary."""

    _require(payload.get("schema_version") == 1, "V6 model schema changed")
    _require(payload.get("artifact_kind") == MODEL_KIND, "V6 model kind changed")
    _require(payload.get("model_id") == MODEL_ID, "V6 model id changed")
    _require(payload.get("model_sha256") == model_sha256(payload), "V6 model changed")
    _require(payload.get("feature_names") == list(FEATURE_NAMES), "features changed")
    _require(payload.get("official_target_outcomes_used") is False, "target was opened")
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    policy = payload.get("policy")
    _require(isinstance(policy, Mapping), "V6 policy is missing")
    assert isinstance(policy, Mapping)
    config = CausalScalePolicyConfig(**policy)
    objects = payload.get("objects")
    _require(isinstance(objects, Mapping), "V6 object models are missing")
    assert isinstance(objects, Mapping)
    _require(
        set(objects) == set(V6_CANDIDATE_SCALES), "V6 promoted object set changed"
    )
    for object_model in objects.values():
        _require(isinstance(object_model, Mapping), "V6 object model is invalid")
        assert isinstance(object_model, Mapping)
        _validate_object_model(object_model, config)
    fallback = payload.get("fallback_effective_scales")
    _require(fallback == V5_EFFECTIVE_SCALES, "V5 fallback scales changed")
    _require(
        payload.get("official_target_takes") == OFFICIAL_TARGET_TAKES,
        "official target partition changed",
    )
    return {
        "passed": True,
        "model_sha256": str(payload["model_sha256"]),
        "config": config,
        "objects": objects,
    }


def build_causal_scale_model(
    source_payloads: Sequence[Mapping[str, Any]],
    *,
    source_artifact_file_sha256s: Mapping[str, str],
    parent_bindings: Mapping[str, str],
    source_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deployment model after a separately computed source-only gate."""

    by_take: dict[str, Mapping[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for source_payload in source_payloads:
        take = source_payload.get("take")
        _require(isinstance(take, Mapping), "source take metadata is missing")
        assert isinstance(take, Mapping)
        take_id = str(take.get("id", ""))
        _require(take_id not in by_take, "source artifact repeats")
        by_take[take_id] = source_payload
        all_rows.extend(extract_source_frame_rows(source_payload))
    expected = sorted(take for takes in SOURCE_TAKES.values() for take in takes)
    _require(sorted(by_take) == expected, "source artifact cohort changed")
    _require(
        sorted(source_artifact_file_sha256s) == expected,
        "source artifact hash inventory changed",
    )
    for digest in source_artifact_file_sha256s.values():
        _require(
            len(str(digest)) == 64
            and all(char in "0123456789abcdef" for char in str(digest)),
            "source artifact digest is invalid",
        )
    _require(source_gate.get("passed") is True, "source gate did not pass")
    config = CausalScalePolicyConfig()
    objects = {
        object_name: fit_object_model(
            [row for row in all_rows if row["object_name"] == object_name],
            config=config,
        )
        for object_name in V6_CANDIDATE_SCALES
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": MODEL_KIND,
        "model_id": MODEL_ID,
        "claim_boundary": (
            "Target-disjoint source training only. This model is not an official-target "
            "result and may not be changed after an official target archive is opened."
        ),
        "feature_names": list(FEATURE_NAMES),
        "policy": config.as_dict(),
        "objects": objects,
        "fallback_effective_scales": dict(V5_EFFECTIVE_SCALES),
        "official_target_takes": dict(OFFICIAL_TARGET_TAKES),
        "source_artifact_file_sha256s": dict(sorted(source_artifact_file_sha256s.items())),
        "parent_bindings": dict(sorted(parent_bindings.items())),
        "source_gate": dict(source_gate),
        "official_target_outcomes_used": False,
        "held_v8_accessed": False,
        "model_sha256": "",
    }
    result["model_sha256"] = model_sha256(result)
    validate_causal_scale_model(result)
    return result


def _decision_from_feature(
    object_model: Mapping[str, Any],
    config: CausalScalePolicyConfig,
    feature: np.ndarray,
    *,
    supported: bool,
) -> CausalScaleDecision:
    object_name = str(object_model["object_name"])
    baseline = float(object_model["baseline_effective_scale"])
    candidate = float(object_model["candidate_effective_scale"])
    if not supported:
        return CausalScaleDecision(
            object_name,
            baseline,
            candidate,
            baseline,
            False,
            "unsupported-v5-exact-fallback",
            None,
            None,
            float(object_model["support_radius"]),
        )
    center = np.asarray(object_model["feature_center"], dtype=np.float64)
    scale = np.asarray(object_model["feature_scale"], dtype=np.float64)
    standardized = (feature - center) / scale
    group_predictions = []
    minimum_distance = math.inf
    for group in object_model["groups"]:
        source = np.asarray(group["standardized_features"], dtype=np.float64)
        gains = np.asarray(group["candidate_gains_mm"], dtype=np.float64)
        distance2 = np.sum(np.square(source - standardized[None, :]), axis=1)
        order = np.argsort(distance2, kind="stable")
        neighbors = order[: min(config.neighbors_per_source_take, len(order))]
        group_predictions.append(float(np.mean(gains[neighbors])))
        minimum_distance = min(minimum_distance, float(np.sqrt(distance2[order[0]])))
    lower_gain = min(group_predictions)
    radius = float(object_model["support_radius"])
    if minimum_distance > radius:
        reason = "outside-source-support-v5-exact-fallback"
        admitted = False
    elif lower_gain <= config.gain_margin_mm:
        reason = "source-lower-envelope-v5-exact-fallback"
        admitted = False
    else:
        reason = "source-lower-envelope-candidate-admitted"
        admitted = True
    return CausalScaleDecision(
        object_name,
        baseline,
        candidate,
        candidate if admitted else baseline,
        admitted,
        reason,
        lower_gain,
        minimum_distance,
        radius,
    )


def select_fitted_feature(
    object_model: Mapping[str, Any],
    config: CausalScalePolicyConfig,
    feature: np.ndarray,
    *,
    supported: bool,
) -> CausalScaleDecision:
    """Apply one already-fitted source model during source cross-validation."""

    values = np.asarray(feature, dtype=np.float64)
    _require(values.shape == (len(FEATURE_NAMES),), "causal feature shape changed")
    _require(np.all(np.isfinite(values)), "causal feature is non-finite")
    return _decision_from_feature(object_model, config, values, supported=supported)


def select_causal_scale(
    model: Mapping[str, Any],
    *,
    object_name: str,
    update: Mapping[str, Any],
    target_frame: int,
    maximum_frame: int,
    supported: bool,
) -> CausalScaleDecision:
    """Select a per-frame scale, failing closed to the immutable V5 arm."""

    validation = validate_causal_scale_model(model)
    baseline = V5_EFFECTIVE_SCALES.get(object_name)
    _require(baseline is not None, "object is outside the missing-five cohort")
    assert baseline is not None
    objects = validation["objects"]
    if object_name not in objects:
        return CausalScaleDecision(
            object_name,
            baseline,
            baseline,
            baseline,
            False,
            "object-not-promoted-v5-exact-fallback",
            None,
            None,
            None,
        )
    if not supported or not bool(update.get("accepted")):
        return _decision_from_feature(
            objects[object_name],
            validation["config"],
            np.zeros(len(FEATURE_NAMES), dtype=np.float64),
            supported=False,
        )
    try:
        feature = causal_scale_feature(
            update,
            target_frame=target_frame,
            maximum_frame=maximum_frame,
        )
    except (TypeError, ValueError, OverflowError):
        return CausalScaleDecision(
            object_name,
            baseline,
            float(objects[object_name]["candidate_effective_scale"]),
            baseline,
            False,
            "invalid-causal-feature-v5-exact-fallback",
            None,
            None,
            float(objects[object_name]["support_radius"]),
        )
    return _decision_from_feature(
        objects[object_name],
        validation["config"],
        feature,
        supported=True,
    )


def causal_scale_vertices(
    target_prior_m: np.ndarray,
    correction_field_m: np.ndarray,
    v5_vertices_m: np.ndarray,
    decision: CausalScaleDecision,
    *,
    supported: bool,
) -> np.ndarray:
    """Apply an admitted scale and preserve exact V5/checkpoint fallback bytes."""

    target_prior = np.asarray(target_prior_m, dtype=np.float64)
    correction = np.asarray(correction_field_m, dtype=np.float64)
    v5_vertices = np.asarray(v5_vertices_m, dtype=np.float64)
    _require(
        target_prior.ndim == 2
        and target_prior.shape[1] == 3
        and correction.shape == target_prior.shape
        and v5_vertices.shape == target_prior.shape,
        "V6 vertex fields must be Nx3 with shared topology",
    )
    _require(
        np.all(np.isfinite(target_prior))
        and np.all(np.isfinite(correction))
        and np.all(np.isfinite(v5_vertices)),
        "V6 vertex fields are non-finite",
    )
    if not supported:
        return target_prior.copy()
    if not decision.admitted:
        return v5_vertices.copy()
    return target_prior + decision.selected_scale * correction


__all__ = [
    "CausalScaleDecision",
    "CausalScalePolicyConfig",
    "FEATURE_NAMES",
    "MODEL_ID",
    "MODEL_KIND",
    "V5_EFFECTIVE_SCALES",
    "V6_CANDIDATE_SCALES",
    "build_causal_scale_model",
    "causal_scale_vertices",
    "causal_scale_feature",
    "extract_source_frame_rows",
    "fit_object_model",
    "model_sha256",
    "select_causal_scale",
    "select_fitted_feature",
    "validate_causal_scale_model",
]
