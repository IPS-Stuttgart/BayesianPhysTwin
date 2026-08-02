"""Prospective composition of pairwise camera updates and a tactile guard."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    SELECTED_BACKBONE_ARM,
    predict_dual_backbone_pairwise_rbf_arrays,
)
from .deform360_tactile_regret_guard import (
    TACTILE_REGRET_FEATURE_NAMES,
    TactileRegretGuardModel,
    apply_tactile_regret_guard,
)
from .phystwin_correspondence_gate import PairwiseCorrespondenceGateConfig
from .phystwin_online_belief import RecursiveRbfBeliefConfig

TACTILE_GUARDED_ARM = "dual_backbone_pairwise_tactile_guarded"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key, None)
    encoded = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tactile_guard_model_from_source_result(
    source_result: Mapping[str, Any],
) -> TactileRegretGuardModel:
    """Load the exact post-open source model without refitting it."""

    _require(
        source_result.get("artifact_kind")
        == "Deform360TactileRegretGuardSourceDiagnostic",
        "unexpected tactile source result",
    )
    _require(
        source_result.get("artifact_sha256")
        == _canonical_sha256(source_result, "artifact_sha256"),
        "tactile source result checksum changed",
    )
    _require(
        source_result.get("all_advancement_gates_passed") is True,
        "tactile source gates did not pass",
    )
    raw = source_result.get("full_source_model_for_future_lock", {})
    _require(
        tuple(raw.get("feature_names", ())) == TACTILE_REGRET_FEATURE_NAMES,
        "tactile source feature contract changed",
    )
    return TactileRegretGuardModel(
        feature_center=tuple(float(value) for value in raw["feature_center"]),
        feature_scale=tuple(float(value) for value in raw["feature_scale"]),
        coefficients=tuple(float(value) for value in raw["coefficients"]),
        ridge_penalty=float(raw["ridge_penalty"]),
        admission_threshold=float(raw["admission_threshold"]),
        source_object_count=int(raw["source_object_count"]),
        source_row_count=int(raw["source_row_count"]),
    )


def tactile_feature_matrix_from_artifact(
    feature_artifact: Mapping[str, Any],
    *,
    case_name: str,
    update_frames: Sequence[int] = (19, 38, 57),
) -> np.ndarray:
    """Read one case's source-independent feature rows from a sealed artifact."""

    _require(
        feature_artifact.get("artifact_kind") == "Deform360CausalTactileFeatureAuditV2"
        and feature_artifact.get("artifact_sha256")
        == _canonical_sha256(feature_artifact, "artifact_sha256"),
        "tactile feature artifact changed",
    )
    boundary = feature_artifact.get("information_boundary", {})
    _require(
        boundary.get("target_outcomes_read") is False
        and boundary.get("held_v8_read") is False
        and boundary.get("future_tactile_values_used_for_update") is False,
        "tactile feature artifact crossed its information boundary",
    )
    matches = [
        row for row in feature_artifact.get("cases", []) if row.get("case") == case_name
    ]
    _require(len(matches) == 1, "tactile feature case is missing or repeated")
    updates = {int(row["update_frame"]): row for row in matches[0]["updates"]}
    requested = tuple(int(value) for value in update_frames)
    _require(tuple(sorted(updates)) == requested, "tactile update schedule changed")
    rows = []
    for update in requested:
        row = updates[update]
        sensor_ratio = np.asarray(row["sensor_energy_over_initial"], dtype=np.float64)
        active_taxels = np.asarray(row["sensor_active_taxels"], dtype=np.float64)
        expanded = {
            **row,
            "sensor_ratio_min": float(np.min(sensor_ratio)),
            "sensor_ratio_max": float(np.max(sensor_ratio)),
            "sensor_ratio_std": float(np.std(sensor_ratio)),
            "active_taxel_mean": float(np.mean(active_taxels)),
            "active_taxel_std": float(np.std(active_taxels)),
        }
        rows.append([float(expanded[name]) for name in TACTILE_REGRET_FEATURE_NAMES])
    values = np.asarray(rows, dtype=np.float64)
    _require(np.all(np.isfinite(values)), "tactile feature matrix is invalid")
    return values


def predict_tactile_guarded_belief_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    tactile_feature_vectors: np.ndarray,
    tactile_model: TactileRegretGuardModel,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = (19, 38, 57),
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the frozen candidate and admit intervals using independent tactile cues."""

    camera_report, camera_arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical_prior_m,
        persistence_m,
        measurement_m,
        measurement_visibility,
        measurement_validity,
        center_ids=center_ids,
        update_frames=update_frames,
        gate_config=gate_config,
        belief_config=belief_config,
    )
    selected = camera_arrays[SELECTED_BACKBONE_ARM]
    candidate = camera_arrays[DUAL_BACKBONE_ARM]
    tactile_report, guarded = apply_tactile_regret_guard(
        selected,
        candidate,
        tactile_feature_vectors,
        tactile_model,
        update_frames=update_frames,
    )
    decisions = tactile_report["updates"]
    updates = tuple(int(value) for value in update_frames)
    for index, (update, decision) in enumerate(zip(updates, decisions, strict=True)):
        stop = updates[index + 1] if index + 1 < len(updates) else len(guarded)
        expected = candidate if decision["candidate_accepted"] else selected
        _require(
            np.array_equal(guarded[update + 1 : stop], expected[update + 1 : stop]),
            "tactile guarded interval changed its selected parent",
        )
    report = {
        "arm": TACTILE_GUARDED_ARM,
        "camera_candidate": camera_report,
        "tactile_guard": tactile_report,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_object_positions_read": False,
            "future_tactile_values_used_for_update": False,
            "camera_state_innovation_likelihood_count": 1,
            "tactile_features_use_camera_state_residual": False,
            "rejected_interval_is_bit_exact_selected_backbone": True,
        },
    }
    return report, {
        SELECTED_BACKBONE_ARM: selected,
        DUAL_BACKBONE_ARM: candidate,
        TACTILE_GUARDED_ARM: guarded,
    }


__all__ = [
    "TACTILE_GUARDED_ARM",
    "predict_tactile_guarded_belief_arrays",
    "tactile_feature_matrix_from_artifact",
    "tactile_guard_model_from_source_result",
]
