"""Explicit evaluator boundary for Deform360 state-of-the-art comparisons.

The Deform360 paper publishes aggregate numbers but its released repository does
not currently include the world-model evaluator or split.  This module keeps an
independent development score useful without allowing it to silently become a
protocol-mismatched Table 4 comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION = 2
DEFORM360_EVALUATOR_CONTRACT_KIND = "Deform360EvaluatorContract"
DEFORM360_EPISODE_SCORE_KIND = "Deform360EpisodeScore"
DEFORM360_PANEL_SCORE_KIND = "Deform360PanelScore"

_CONTRACT_STATUSES = {
    "unresolved-non-authorizing",
    "independent-protocol",
    "official-parity",
}
_CHAMFER_DEFINITIONS = {
    "symmetric_mean_euclidean_m",
    "symmetric_mean_squared_euclidean_m2",
}
_TRACK_DEFINITIONS = {
    "mean_euclidean_m",
    "root_mean_squared_euclidean_m",
    "mean_squared_euclidean_m2",
}
_VISIBILITY_POLICIES = {
    "all_finite_material_points",
    "visible_and_finite_material_points",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deform360_evaluator_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical checksum used by evaluator-contract artifacts."""

    return _canonical_sha256(payload)


def _episode_key(object_id: str, episode_id: int) -> str:
    return f"{object_id}/{int(episode_id)}"


def validate_deform360_evaluator_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a contract without pretending unresolved fields are known."""

    _require(
        payload.get("schema_version")
        == DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION,
        "unsupported Deform360 evaluator-contract schema",
    )
    _require(
        payload.get("artifact_kind") == DEFORM360_EVALUATOR_CONTRACT_KIND,
        "unexpected Deform360 evaluator-contract kind",
    )
    _require(
        payload.get("result_sha256") == _canonical_sha256(payload),
        "Deform360 evaluator-contract checksum mismatch",
    )
    status = payload.get("status")
    _require(status in _CONTRACT_STATUSES, "invalid evaluator-contract status")
    unresolved = payload.get("unresolved_fields")
    _require(
        isinstance(unresolved, list)
        and all(isinstance(value, str) and value for value in unresolved),
        "unresolved_fields must be a list of nonempty paths",
    )
    dataset = payload.get("dataset")
    _require(isinstance(dataset, Mapping), "contract dataset is missing")
    _require(dataset.get("coordinate_unit") == "m", "coordinate unit must be metres")
    split = payload.get("split")
    _require(isinstance(split, Mapping), "contract split is missing")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), "contract metrics are missing")
    temporal = payload.get("temporal")
    _require(isinstance(temporal, Mapping), "contract temporal policy is missing")
    particles = payload.get("particles")
    _require(isinstance(particles, Mapping), "contract particle policy is missing")
    aggregation = payload.get("aggregation")
    _require(isinstance(aggregation, Mapping), "contract aggregation is missing")
    reference = payload.get("published_reference")
    _require(isinstance(reference, Mapping), "published reference is missing")
    _require(
        float(reference.get("future_chamfer_m", -1.0)) == 0.051
        and float(reference.get("future_track_error_m", -1.0)) == 0.079,
        "published Deform360 Table 4 reference changed",
    )
    if status == "official-parity":
        _validate_official_parity_contract(payload)
    return {
        "passed": True,
        "status": status,
        "official_table4_authorizing": status == "official-parity",
        "unresolved_field_count": len(unresolved),
        "result_sha256": payload["result_sha256"],
    }


def _validate_official_parity_contract(payload: Mapping[str, Any]) -> None:
    _require(not payload["unresolved_fields"], "official parity has unresolved fields")
    split = payload["split"]
    object_ids = split.get("object_ids")
    fit = split.get("fit_episode_ids_by_object")
    held = split.get("held_episode_ids_by_object")
    _require(
        isinstance(object_ids, list) and object_ids,
        "official parity requires the complete ordered object split",
    )
    _require(
        isinstance(fit, Mapping)
        and isinstance(held, Mapping)
        and set(fit) == set(held) == set(object_ids),
        "official split maps differ from object_ids",
    )
    _require(
        all(fit[object_id] and held[object_id] for object_id in object_ids),
        "official split contains an empty fit or held set",
    )
    temporal = payload["temporal"]
    for field in (
        "evaluation_start_frame",
        "evaluation_stop_frame_exclusive",
        "frame_stride",
    ):
        _require(isinstance(temporal.get(field), int), f"temporal.{field} is unresolved")
    _require(
        0 <= temporal["evaluation_start_frame"]
        < temporal["evaluation_stop_frame_exclusive"]
        and temporal["frame_stride"] >= 1,
        "official temporal range is invalid",
    )
    metrics = payload["metrics"]
    _require(
        metrics.get("chamfer", {}).get("definition") in _CHAMFER_DEFINITIONS,
        "official Chamfer definition is unresolved",
    )
    _require(
        metrics.get("chamfer", {}).get("visibility_policy")
        in _VISIBILITY_POLICIES,
        "official Chamfer visibility policy is unresolved",
    )
    _require(
        metrics.get("track", {}).get("definition") in _TRACK_DEFINITIONS,
        "official track definition is unresolved",
    )
    _require(
        metrics.get("track", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "official track visibility policy is unresolved",
    )
    _require(
        payload["aggregation"].get("panel")
        in {"object_balanced_mean", "episode_balanced_mean"},
        "official panel aggregation is unresolved",
    )
    identities = payload["particles"].get("identity_sha256_by_episode")
    expected_keys = {
        _episode_key(object_id, episode_id)
        for object_id in object_ids
        for episode_id in held[object_id]
    }
    _require(
        isinstance(identities, Mapping) and set(identities) == expected_keys,
        "official particle identities do not cover the complete held split",
    )
    _require(
        all(_valid_sha256(value) for value in identities.values()),
        "official particle identity checksum is invalid",
    )
    provenance = payload.get("evaluator_provenance")
    _require(isinstance(provenance, Mapping), "official evaluator provenance is missing")
    _require(
        provenance.get("released_by_deform360_authors") is True,
        "evaluator is not author-released",
    )
    _require(
        _valid_sha256(provenance.get("source_revision_sha256"))
        and _valid_sha256(provenance.get("entrypoint_sha256")),
        "official evaluator provenance checksum is invalid",
    )
    reproduction = provenance.get("particleformer_table4_reproduction")
    _require(
        isinstance(reproduction, Mapping)
        and reproduction.get("passed") is True
        and float(reproduction.get("future_chamfer_m", -1.0)) == 0.051
        and float(reproduction.get("future_track_error_m", -1.0)) == 0.079,
        "official evaluator has not reproduced the published reference row",
    )


def load_deform360_evaluator_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "evaluator contract must contain an object")
    validate_deform360_evaluator_contract(payload)
    return payload


def write_deform360_evaluator_contract(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_deform360_evaluator_contract(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _score_ready_contract(payload: Mapping[str, Any], episode_key: str) -> None:
    validate_deform360_evaluator_contract(payload)
    temporal = payload["temporal"]
    _require(
        all(
            isinstance(temporal.get(field), int)
            for field in (
                "evaluation_start_frame",
                "evaluation_stop_frame_exclusive",
                "frame_stride",
            )
        ),
        "temporal evaluator fields are unresolved",
    )
    metrics = payload["metrics"]
    _require(
        metrics.get("chamfer", {}).get("definition") in _CHAMFER_DEFINITIONS,
        "Chamfer definition is unresolved",
    )
    _require(
        metrics.get("chamfer", {}).get("visibility_policy")
        in _VISIBILITY_POLICIES,
        "Chamfer visibility policy is unresolved",
    )
    _require(
        metrics.get("track", {}).get("definition") in _TRACK_DEFINITIONS,
        "track definition is unresolved",
    )
    _require(
        metrics.get("track", {}).get("visibility_policy") in _VISIBILITY_POLICIES,
        "track visibility policy is unresolved",
    )
    identities = payload["particles"].get("identity_sha256_by_episode")
    _require(
        isinstance(identities, Mapping) and episode_key in identities,
        f"particle identity is unresolved for {episode_key}",
    )


def _chamfer(
    target: np.ndarray, prediction: np.ndarray, definition: str
) -> float:
    difference = target[:, None, :] - prediction[None, :, :]
    squared = np.sum(difference * difference, axis=2)
    if definition == "symmetric_mean_euclidean_m":
        distance = np.sqrt(squared)
    elif definition == "symmetric_mean_squared_euclidean_m2":
        distance = squared
    else:  # pragma: no cover - guarded by contract validation
        raise ValueError(f"unsupported Chamfer definition: {definition}")
    return 0.5 * (
        float(np.mean(np.min(distance, axis=0)))
        + float(np.mean(np.min(distance, axis=1)))
    )


def _track(displacement: np.ndarray, definition: str) -> float:
    squared = np.sum(displacement * displacement, axis=1)
    if definition == "mean_euclidean_m":
        return float(np.mean(np.sqrt(squared)))
    if definition == "root_mean_squared_euclidean_m":
        return float(np.sqrt(np.mean(squared)))
    if definition == "mean_squared_euclidean_m2":
        return float(np.mean(squared))
    raise ValueError(f"unsupported track definition: {definition}")


def score_deform360_episode(
    contract: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    particle_identity_sha256: str,
    target_m: np.ndarray,
    prediction_m: np.ndarray,
    visibility: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one episode only under the explicitly named metric contract."""

    episode_key = _episode_key(object_id, episode_id)
    _score_ready_contract(contract, episode_key)
    expected_identity = contract["particles"]["identity_sha256_by_episode"][episode_key]
    _require(
        particle_identity_sha256 == expected_identity,
        "particle identity differs from the evaluator contract",
    )
    target = np.asarray(target_m, dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    _require(
        target.shape == prediction.shape
        and target.ndim == 3
        and target.shape[2] == 3,
        "target and prediction must share shape (T,N,3)",
    )
    temporal = contract["temporal"]
    indices = np.arange(
        temporal["evaluation_start_frame"],
        temporal["evaluation_stop_frame_exclusive"],
        temporal["frame_stride"],
        dtype=np.int64,
    )
    _require(len(indices) and int(indices[-1]) < len(target), "evaluation horizon exceeds data")
    chamfer_visibility_policy = contract["metrics"]["chamfer"][
        "visibility_policy"
    ]
    track_visibility_policy = contract["metrics"]["track"]["visibility_policy"]
    if "visible_and_finite_material_points" in {
        chamfer_visibility_policy,
        track_visibility_policy,
    }:
        _require(visibility is not None, "visibility mask is required by the contract")
        visible = np.asarray(visibility, dtype=bool)
        _require(visible.shape == target.shape[:2], "visibility shape differs")
    else:
        visible = np.ones(target.shape[:2], dtype=bool)
    chamfer_definition = contract["metrics"]["chamfer"]["definition"]
    track_definition = contract["metrics"]["track"]["definition"]
    frame_chamfer: list[float] = []
    frame_track: list[float] = []
    frame_chamfer_counts: list[int] = []
    frame_track_counts: list[int] = []
    for frame_index in indices:
        finite = np.all(np.isfinite(target[frame_index]), axis=1) & np.all(
            np.isfinite(prediction[frame_index]), axis=1
        )
        chamfer_selected = finite.copy()
        if chamfer_visibility_policy == "visible_and_finite_material_points":
            chamfer_selected &= visible[frame_index]
        track_selected = finite.copy()
        if track_visibility_policy == "visible_and_finite_material_points":
            track_selected &= visible[frame_index]
        _require(
            np.any(chamfer_selected),
            f"no valid Chamfer particles at frame {frame_index}",
        )
        _require(
            np.any(track_selected),
            f"no valid track particles at frame {frame_index}",
        )
        frame_chamfer.append(
            _chamfer(
                target[frame_index, chamfer_selected],
                prediction[frame_index, chamfer_selected],
                chamfer_definition,
            )
        )
        frame_track.append(
            _track(
                prediction[frame_index, track_selected]
                - target[frame_index, track_selected],
                track_definition,
            )
        )
        frame_chamfer_counts.append(int(np.count_nonzero(chamfer_selected)))
        frame_track_counts.append(int(np.count_nonzero(track_selected)))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEFORM360_EPISODE_SCORE_KIND,
        "contract_result_sha256": contract["result_sha256"],
        "object_id": object_id,
        "episode_id": int(episode_id),
        "episode_key": episode_key,
        "particle_identity_sha256": particle_identity_sha256,
        "evaluated_frame_indices": indices.tolist(),
        "valid_chamfer_particle_count_by_frame": frame_chamfer_counts,
        "valid_track_particle_count_by_frame": frame_track_counts,
        "metrics": {
            "future_chamfer": float(np.mean(frame_chamfer)),
            "future_track_error": float(np.mean(frame_track)),
            "chamfer_definition": chamfer_definition,
            "chamfer_visibility_policy": chamfer_visibility_policy,
            "track_definition": track_definition,
            "track_visibility_policy": track_visibility_policy,
            "per_frame_chamfer": frame_chamfer,
            "per_frame_track_error": frame_track,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def aggregate_deform360_panel(
    contract: Mapping[str, Any], episode_scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate episodes without silently changing the declared replicate unit."""

    validate_deform360_evaluator_contract(contract)
    rows = tuple(episode_scores)
    _require(rows, "panel score requires at least one episode")
    observed_keys = {str(row.get("episode_key")) for row in rows}
    _require(len(observed_keys) == len(rows), "panel contains duplicate episodes")
    for row in rows:
        _require(
            row.get("artifact_kind") == DEFORM360_EPISODE_SCORE_KIND
            and row.get("result_sha256") == _canonical_sha256(row)
            and row.get("contract_result_sha256") == contract["result_sha256"],
            "panel contains an invalid or mismatched episode score",
        )
    held = contract["split"].get("held_episode_ids_by_object")
    if held:
        expected = {
            _episode_key(object_id, episode_id)
            for object_id, episode_ids in held.items()
            for episode_id in episode_ids
        }
        _require(observed_keys == expected, "panel differs from the declared held split")
    by_object: dict[str, dict[str, float | int]] = {}
    for object_id in sorted({str(row["object_id"]) for row in rows}):
        selected = [row for row in rows if row["object_id"] == object_id]
        by_object[object_id] = {
            "episode_count": len(selected),
            "future_chamfer": float(
                np.mean([row["metrics"]["future_chamfer"] for row in selected])
            ),
            "future_track_error": float(
                np.mean([row["metrics"]["future_track_error"] for row in selected])
            ),
        }
    panel_rule = contract["aggregation"].get("panel")
    if panel_rule == "object_balanced_mean":
        chamfer = float(np.mean([row["future_chamfer"] for row in by_object.values()]))
        track = float(
            np.mean([row["future_track_error"] for row in by_object.values()])
        )
    elif panel_rule == "episode_balanced_mean":
        chamfer = float(np.mean([row["metrics"]["future_chamfer"] for row in rows]))
        track = float(np.mean([row["metrics"]["future_track_error"] for row in rows]))
    else:
        raise ValueError("panel aggregation is unresolved")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DEFORM360_PANEL_SCORE_KIND,
        "contract_result_sha256": contract["result_sha256"],
        "episode_count": len(rows),
        "object_count": len(by_object),
        "aggregation": panel_rule,
        "metrics": {
            "future_chamfer": chamfer,
            "future_track_error": track,
            "by_object": by_object,
        },
        "episode_result_sha256": {
            str(row["episode_key"]): str(row["result_sha256"])
            for row in sorted(rows, key=lambda value: str(value["episode_key"]))
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def authorize_deform360_table4_claim(
    contract: Mapping[str, Any], panel_score: Mapping[str, Any]
) -> dict[str, Any]:
    """Authorize a direct Table 4 claim only after exact evaluator parity."""

    validation = validate_deform360_evaluator_contract(contract)
    _require(
        validation["official_table4_authorizing"],
        "direct Table 4 comparison refused: evaluator parity is not established",
    )
    _require(
        panel_score.get("artifact_kind") == DEFORM360_PANEL_SCORE_KIND
        and panel_score.get("result_sha256") == _canonical_sha256(panel_score)
        and panel_score.get("contract_result_sha256") == contract["result_sha256"],
        "panel score does not belong to the official evaluator contract",
    )
    reference = contract["published_reference"]
    chamfer = float(panel_score["metrics"]["future_chamfer"])
    track = float(panel_score["metrics"]["future_track_error"])
    gates = {
        "future_chamfer_below_particleformer": chamfer
        < float(reference["future_chamfer_m"]),
        "future_track_below_particleformer": track
        < float(reference["future_track_error_m"]),
    }
    return {
        "authorized": all(gates.values()),
        "protocol_parity_established": True,
        "gates": gates,
        "candidate": {
            "future_chamfer": chamfer,
            "future_track_error": track,
        },
        "reference": {
            "future_chamfer_m": float(reference["future_chamfer_m"]),
            "future_track_error_m": float(reference["future_track_error_m"]),
        },
        "contract_result_sha256": contract["result_sha256"],
        "panel_result_sha256": panel_score["result_sha256"],
    }


__all__ = [
    "DEFORM360_EVALUATOR_CONTRACT_KIND",
    "DEFORM360_EVALUATOR_CONTRACT_SCHEMA_VERSION",
    "aggregate_deform360_panel",
    "authorize_deform360_table4_claim",
    "deform360_evaluator_contract_sha256",
    "load_deform360_evaluator_contract",
    "score_deform360_episode",
    "validate_deform360_evaluator_contract",
    "write_deform360_evaluator_contract",
]
