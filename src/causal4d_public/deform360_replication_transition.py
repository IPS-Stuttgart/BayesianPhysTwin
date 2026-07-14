"""Checksum-bound causal contact-transition artifacts for Deform360."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_replication_controls import (
    ContactTransitionEpisode,
    ContactTransitionFit,
    ContactTransitionModel,
)


TRANSITION_EPISODE_SCHEMA_VERSION = 1
TRANSITION_FIT_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _finite_diagnostics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_diagnostics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def build_transition_episode_artifact(
    episode: ContactTransitionEpisode,
    archive_path: str | Path,
    *,
    object_id: str,
    split: str,
    pooled_fit_result_sha256: str,
    pooled_candidate_index: int,
    prefix_geometry_result_sha256: str,
    visual_contact_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Archive one visual-rollout feature episode and oracle contact labels."""

    _require(split in {"source", "calibration"}, "transition split is invalid")
    for value, name in (
        (pooled_fit_result_sha256, "pooled fit"),
        (prefix_geometry_result_sha256, "prefix geometry"),
    ):
        _require(len(value) == 64, f"{name} checksum is invalid")
    output = Path(archive_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        openings_m=episode.openings_m,
        controller_positions_m=episode.controller_positions_m,
        predicted_object_positions_m=episode.predicted_object_positions_m,
        contact_active=episode.contact_active.astype(np.uint8),
    )
    payload: dict[str, Any] = {
        "schema_version": TRANSITION_EPISODE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ContactTransitionEpisode",
        "object_id": str(object_id),
        "episode_id": episode.episode_id,
        "split": split,
        "dt_seconds": episode.dt_seconds,
        "frame_count": len(episode.openings_m),
        "controller_count": episode.openings_m.shape[1],
        "object_node_count": episode.predicted_object_positions_m.shape[1],
        "pooled_fit_result_sha256": pooled_fit_result_sha256,
        "pooled_candidate_index": int(pooled_candidate_index),
        "prefix_geometry_result_sha256": prefix_geometry_result_sha256,
        "visual_contact_model": dict(visual_contact_model),
        "feature_rollout_policy": (
            "official Warp rollout under the source-fitted visual contact schedule"
        ),
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
        },
        "information_boundary": {
            "split": split,
            "future_tactile_read_as_training_label": True,
            "future_geometry_read_for_feature_rollout": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def load_transition_episode_artifact(
    payload: Mapping[str, Any],
) -> ContactTransitionEpisode:
    """Validate and load one transition-training episode."""

    _require(
        payload.get("schema_version") == TRANSITION_EPISODE_SCHEMA_VERSION,
        "transition-episode schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360ContactTransitionEpisode",
        "transition-episode kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "transition-episode checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False,
        "transition-training artifact crossed the target boundary",
    )
    archive = Path(payload["archive"]["path"])
    _require(
        archive.is_file() and _sha256_file(archive) == payload["archive"]["sha256"],
        "transition-episode archive changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        episode = ContactTransitionEpisode(
            episode_id=str(payload["episode_id"]),
            openings_m=np.asarray(stored["openings_m"], dtype=np.float64),
            controller_positions_m=np.asarray(
                stored["controller_positions_m"], dtype=np.float64
            ),
            predicted_object_positions_m=np.asarray(
                stored["predicted_object_positions_m"], dtype=np.float64
            ),
            contact_active=np.asarray(stored["contact_active"], dtype=bool),
            dt_seconds=float(payload["dt_seconds"]),
        )
    _require(len(episode.openings_m) == payload["frame_count"], "frame count changed")
    _require(
        episode.openings_m.shape[1] == payload["controller_count"],
        "controller count changed",
    )
    _require(
        episode.predicted_object_positions_m.shape[1]
        == payload["object_node_count"],
        "object node count changed",
    )
    return episode


def build_transition_fit_artifact(
    fit: ContactTransitionFit,
    source_artifacts: Sequence[Mapping[str, Any]],
    calibration_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal a source-fitted, calibration-selected transition model."""

    _require(len(source_artifacts) >= 2, "too few source transition artifacts")
    _require(
        len(calibration_artifacts) >= 1,
        "too few calibration transition artifacts",
    )
    for payload in (*source_artifacts, *calibration_artifacts):
        load_transition_episode_artifact(payload)
    _require(
        all(payload["split"] == "source" for payload in source_artifacts),
        "source transition split changed",
    )
    _require(
        all(payload["split"] == "calibration" for payload in calibration_artifacts),
        "calibration transition split changed",
    )
    payload: dict[str, Any] = {
        "schema_version": TRANSITION_FIT_SCHEMA_VERSION,
        "artifact_kind": "Deform360CausalContactTransitionFit",
        "model": fit.model.as_dict(),
        "source_metrics": _finite_diagnostics(fit.source_metrics),
        "calibration_metrics": _finite_diagnostics(fit.calibration_metrics),
        "candidate_table": _finite_diagnostics(fit.candidate_table),
        "source_episode_result_sha256": [
            value["result_sha256"] for value in source_artifacts
        ],
        "calibration_episode_result_sha256": [
            value["result_sha256"] for value in calibration_artifacts
        ],
        "source_episode_ids": [value["episode_id"] for value in source_artifacts],
        "calibration_episode_ids": [
            value["episode_id"] for value in calibration_artifacts
        ],
        "information_boundary": {
            "source_labels_used_for_coefficients": True,
            "calibration_labels_used_for_hyperparameters_only": True,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_transition_fit_artifact(
    payload: Mapping[str, Any],
) -> ContactTransitionModel:
    _require(
        payload.get("schema_version") == TRANSITION_FIT_SCHEMA_VERSION,
        "transition-fit schema changed",
    )
    _require(
        payload.get("artifact_kind") == "Deform360CausalContactTransitionFit",
        "transition-fit kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "transition-fit checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_labels_used_for_hyperparameters_only") is True
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False,
        "transition fit crossed its information boundary",
    )
    return ContactTransitionModel(**payload["model"])


def write_transition_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "build_transition_episode_artifact",
    "build_transition_fit_artifact",
    "load_transition_episode_artifact",
    "validate_transition_fit_artifact",
    "write_transition_artifact",
]
