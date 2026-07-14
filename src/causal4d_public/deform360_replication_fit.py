"""Source-only official-Warp grid fitting for the Deform360 replication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_phystwin_feasibility import (
    WarpRopeFeasibilityConfig,
    warp_rope_candidates,
)
from .deform360_replication_case import (
    build_replication_warp_observation,
    score_constant_persistence,
    score_replication_warp_prediction,
)
from .deform360_replication_contact import (
    ReplicationContactEpisode,
    ReplicationOpeningContactModel,
    contact_state_by_robot_axis,
)
from .deform360_replication_controls import select_pooling_controls
from .deform360_replication_warp import (
    OfficialWarpSparseGraphRunner,
    sparse_graph_strain_summary,
)


SOURCE_GRID_SCHEMA_VERSION = 3
POOLED_FIT_SCHEMA_VERSION = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def score_source_warp_candidate_grid(
    episode_dir: str | Path,
    episode: ReplicationContactEpisode,
    contact_model: ReplicationOpeningContactModel,
    stratum: str,
    raw_hull_frame_indices: np.ndarray,
    reference_hulls_m: Sequence[np.ndarray],
    reference_geometry_result_sha256: str,
    reference_geometry_total_frame_count: int,
    official_repo: str | Path,
    *,
    device: str = "cuda:0",
    config: WarpRopeFeasibilityConfig | None = None,
) -> dict[str, Any]:
    """Score the locked 200-candidate grid on one source episode."""

    cfg = config or WarpRopeFeasibilityConfig()
    _require(cfg.substeps == 128, "replication source grid requires 128 substeps")
    _require(
        len(reference_geometry_result_sha256) == 64,
        "source grid requires a geometry checksum",
    )
    _require(
        reference_geometry_total_frame_count >= len(reference_hulls_m) >= 2,
        "source grid geometry frame counts are invalid",
    )
    schedule = contact_state_by_robot_axis(
        episode, contact_model.tactile_group_to_robot_axis
    )
    observation = build_replication_warp_observation(
        episode_dir,
        episode.episode_id,
        stratum,
        raw_hull_frame_indices,
        reference_hulls_m,
        schedule,
    )
    runner = OfficialWarpSparseGraphRunner(
        official_repo, observation.case, cfg, device=device
    )
    rows = []
    for index, candidate in enumerate(warp_rope_candidates(cfg)):
        prediction = runner.rollout(candidate)
        metrics = score_replication_warp_prediction(observation, prediction)
        strain = sparse_graph_strain_summary(observation.case.graph, prediction)
        rows.append(
            {
                "candidate_index": index,
                "parameters": candidate.as_dict(),
                "mean_chamfer_m": _finite(float(metrics["mean_m"])),
                "late_chamfer_m": _finite(float(metrics["late_mean_m"])),
                "p99_relative_edge_strain": _finite(strain["p99"]),
                "maximum_relative_edge_strain": _finite(strain["maximum"]),
                "finite": bool(np.isfinite(float(metrics["mean_m"]))),
            }
        )
    persistence = score_constant_persistence(observation)
    payload = {
        "schema_version": SOURCE_GRID_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationSourceWarpCandidateGrid",
        "episode_id": episode.episode_id,
        "stratum": stratum,
        "config": asdict(cfg),
        "candidate_count": len(rows),
        "reference_geometry_result_sha256": reference_geometry_result_sha256,
        "reference_geometry_total_frame_count": reference_geometry_total_frame_count,
        "reference_geometry_available_frame_count": len(reference_hulls_m),
        "raw_hull_frame_indices": observation.raw_hull_frame_indices.astype(int).tolist(),
        "persistence": persistence,
        "contact_model": asdict(contact_model),
        "contact_associations": list(observation.contact_associations),
        "candidate_scores": rows,
        "information_boundary": {
            "source_episode_only": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_source_warp_candidate_grid(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == SOURCE_GRID_SCHEMA_VERSION, "source-grid schema changed")
    _require(payload.get("artifact_kind") == "Deform360ReplicationSourceWarpCandidateGrid", "source-grid kind changed")
    _require(payload.get("result_sha256") == _artifact_sha256(payload), "source-grid checksum mismatch")
    _require(
        isinstance(payload.get("reference_geometry_result_sha256"), str)
        and len(payload["reference_geometry_result_sha256"]) == 64,
        "source-grid geometry checksum is missing",
    )
    total = payload.get("reference_geometry_total_frame_count")
    available = payload.get("reference_geometry_available_frame_count")
    _require(
        isinstance(total, int)
        and isinstance(available, int)
        and total >= available >= 2,
        "source-grid geometry frame counts are invalid",
    )
    rows = payload.get("candidate_scores")
    _require(isinstance(rows, list) and len(rows) == 200, "source grid is not 200 candidates")
    _require([row["candidate_index"] for row in rows] == list(range(200)), "candidate indices changed")
    _require(payload["information_boundary"]["target_future_read"] is False, "source grid read a target")


def pool_source_warp_candidate_grids(
    source_grids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal pooled, single-source, and leave-one-source candidate identities."""

    _require(len(source_grids) >= 2, "pooling needs two source episodes")
    for payload in source_grids:
        validate_source_warp_candidate_grid(payload)
    episode_ids = [str(payload["episode_id"]) for payload in source_grids]
    _require(len(episode_ids) == len(set(episode_ids)), "source episode repeated")
    object_ids = {
        episode_id.split("/episode_", maxsplit=1)[0] for episode_id in episode_ids
    }
    _require(len(object_ids) == 1, "source grids mix object identities")
    object_id = next(iter(object_ids))
    scores = np.full((200, len(source_grids)), np.inf, dtype=np.float64)
    parameters: dict[int, dict[str, float]] = {}
    persistence = np.empty(len(source_grids), dtype=np.float64)
    maximum_p99_strain: float | None = None
    for source_index, payload in enumerate(source_grids):
        configured_limit = float(
            payload["config"]["maximum_p99_relative_edge_strain"]
        )
        if maximum_p99_strain is None:
            maximum_p99_strain = configured_limit
        else:
            _require(
                configured_limit == maximum_p99_strain,
                "source-grid strain limit changed",
            )
        persistence[source_index] = float(payload["persistence"]["mean_m"])
        for row in payload["candidate_scores"]:
            index = int(row["candidate_index"])
            if (
                row["mean_chamfer_m"] is not None
                and row["p99_relative_edge_strain"] is not None
                and float(row["p99_relative_edge_strain"])
                <= configured_limit
            ):
                scores[index, source_index] = float(row["mean_chamfer_m"])
            if index in parameters:
                _require(parameters[index] == row["parameters"], "candidate grid changed")
            else:
                parameters[index] = dict(row["parameters"])
    selection = select_pooling_controls(scores)
    pooled_scores = scores[selection.pooled_candidate_index]
    pooled_mean = float(np.mean(pooled_scores))
    persistence_mean = float(np.mean(persistence))
    leave_one_out = []
    for held_out in range(len(source_grids)):
        fit_indices = [index for index in range(len(source_grids)) if index != held_out]
        valid = np.all(np.isfinite(scores[:, fit_indices]), axis=1)
        _require(np.any(valid), "no leave-one-source candidate is finite")
        candidate = int(
            min(
                np.flatnonzero(valid),
                key=lambda index: (
                    float(np.mean(scores[index, fit_indices])),
                    int(index),
                ),
            )
        )
        value = float(scores[candidate, held_out])
        baseline = float(persistence[held_out])
        leave_one_out.append(
            {
                "held_out_episode_id": episode_ids[held_out],
                "selected_candidate_index": candidate,
                "candidate_chamfer_m": _finite(value),
                "persistence_chamfer_m": baseline,
                "win": bool(np.isfinite(value) and value < baseline),
            }
        )
    loo_win_fraction = float(np.mean([row["win"] for row in leave_one_out]))
    required = sorted(
        {
            selection.pooled_candidate_index,
            *selection.single_source_candidate_indices,
            *(row["selected_candidate_index"] for row in leave_one_out),
        }
    )
    payload = {
        "schema_version": POOLED_FIT_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReplicationPooledSourceWarpFit",
        "object_id": object_id,
        "source_episode_ids": episode_ids,
        "source_grid_result_sha256": [payload["result_sha256"] for payload in source_grids],
        "selection": selection.as_dict(),
        "candidate_quality_filter": {
            "finite_chamfer_required": True,
            "maximum_p99_relative_edge_strain": maximum_p99_strain,
            "pooled_candidate_must_pass_every_source_episode": True,
            "single_source_candidate_must_pass_its_fit_episode": True,
        },
        "pooled_source_mean_chamfer_m": pooled_mean,
        "persistence_source_mean_chamfer_m": persistence_mean,
        "pooled_source_relative_improvement_vs_persistence": (
            (persistence_mean - pooled_mean) / persistence_mean
        ),
        "leave_one_source": leave_one_out,
        "leave_one_source_win_fraction": loo_win_fraction,
        "sealed_candidate_indices": required,
        "sealed_candidate_parameters": {
            str(index): parameters[index] for index in required
        },
        "source_backend_competence": {
            "minimum_relative_improvement": 0.05,
            "minimum_leave_one_source_win_fraction": 0.60,
            "passed": bool(
                (persistence_mean - pooled_mean) / persistence_mean >= 0.05
                and loo_win_fraction >= 0.60
            ),
        },
        "information_boundary": {
            "source_candidate_scores_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_pooled_source_warp_fit(payload: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema_version") == POOLED_FIT_SCHEMA_VERSION,
        "pooled-fit schema changed",
    )
    _require(
        payload.get("artifact_kind")
        == "Deform360ReplicationPooledSourceWarpFit",
        "pooled-fit kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "pooled-fit checksum mismatch",
    )
    _require(
        len(payload.get("source_episode_ids", [])) >= 2,
        "pooled fit has too few sources",
    )
    _require(
        isinstance(payload.get("object_id"), str) and payload["object_id"],
        "pooled fit has no object identity",
    )
    quality = payload.get("candidate_quality_filter", {})
    _require(
        quality.get("finite_chamfer_required") is True
        and quality.get("maximum_p99_relative_edge_strain") == 0.5
        and quality.get("pooled_candidate_must_pass_every_source_episode") is True,
        "pooled-fit quality filter changed",
    )
    _require(
        payload["information_boundary"]["target_future_read"] is False,
        "pooled fit read target future data",
    )


def write_replication_fit_artifact(
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
    "pool_source_warp_candidate_grids",
    "score_source_warp_candidate_grid",
    "validate_pooled_source_warp_fit",
    "validate_source_warp_candidate_grid",
    "write_replication_fit_artifact",
]
