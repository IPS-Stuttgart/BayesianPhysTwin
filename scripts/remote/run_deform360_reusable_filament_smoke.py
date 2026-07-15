#!/usr/bin/env python3
"""Run a source-only reusable-rest-metric smoke test on Deform360 filaments."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_phystwin_feasibility import (
    WarpRopeFeasibilityConfig,
    warp_rope_candidates,
)
from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_case import (
    build_replication_warp_observation,
    score_constant_persistence,
    score_replication_warp_prediction,
)
from causal4d_public.deform360_replication_contact import (
    ReplicationOpeningContactModel,
    contact_state_by_robot_axis,
    load_replication_contact_episode,
)
from causal4d_public.deform360_replication_geometry import (
    load_replication_hull_archive,
)
from causal4d_public.deform360_replication_graph import (
    build_filament_sparse_graph,
)
from causal4d_public.deform360_replication_warp import (
    OfficialWarpSparseGraphRunner,
    sparse_graph_strain_summary,
)
from causal4d_public.deform360_reusable_twin import (
    fit_reusable_filament_twin,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--object-id", action="append", required=True)
    parser.add_argument("--rest-quantile", type=float, action="append")
    parser.add_argument("--candidate-index", type=int, action="append")
    parser.add_argument(
        "--initial-velocity-policy",
        choices=("zero", "contact-propagated"),
        action="append",
    )
    parser.add_argument("--legacy-fit-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _cohort_record(protocol: dict[str, Any], object_id: str) -> dict[str, Any]:
    matches = [
        record
        for record in protocol["config"]["cohort"]
        if record["object_id"] == object_id
    ]
    if len(matches) != 1:
        raise ValueError(f"object is not unique in protocol: {object_id}")
    if matches[0]["stratum"] != "filament":
        raise ValueError(f"smoke runner accepts only filaments: {object_id}")
    return matches[0]


def _load_contact_model(root: Path, object_id: str) -> ReplicationOpeningContactModel:
    path = root / "observations" / object_id / "contact_model.json"
    return ReplicationOpeningContactModel(
        **json.loads(path.read_text(encoding="utf-8"))
    )


def _load_source(
    root: Path, cohort: dict[str, Any], object_id: str, episode_index: int
) -> dict[str, Any]:
    episode_id = f"{object_id}/episode_{episode_index:04d}"
    metadata = cohort["episodes"][str(episode_index)]
    episode_dir = root / "aligned" / object_id / f"episode_{episode_index:04d}"
    episode = load_replication_contact_episode(
        episode_dir,
        episode_id=episode_id,
        bimanual=metadata["bimanual"] == "yes",
        nonprehensile=metadata["nonprehensile"] == "yes",
    )
    hull_path = (
        root
        / "observations"
        / object_id
        / f"episode_{episode_index:04d}"
        / "sampled_hulls.json"
    )
    hull_payload = json.loads(hull_path.read_text(encoding="utf-8"))
    frames, hulls = load_replication_hull_archive(hull_payload)
    available = np.asarray([len(hull) > 0 for hull in hulls], dtype=bool)
    if not available[0] or np.count_nonzero(available) < 2:
        raise ValueError(f"source hull is not forecastable: {episode_id}")
    return {
        "episode_id": episode_id,
        "episode_dir": episode_dir,
        "episode": episode,
        "frames": frames[available],
        "hulls": tuple(
            hull for hull, keep in zip(hulls, available, strict=True) if keep
        ),
        "geometry_sha256": hull_payload["result_sha256"],
    }


def _legacy_score(
    root: Path | None, object_id: str, episode_index: int, candidate_index: int
) -> float | None:
    if root is None:
        return None
    path = root / object_id / f"source_episode_{episode_index:04d}_grid.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["candidate_scores"][candidate_index]
    if row["candidate_index"] != candidate_index:
        raise ValueError("legacy candidate ordering changed")
    value = row["mean_chamfer_m"]
    return None if value is None else float(value)


def _run_object(
    protocol: dict[str, Any],
    root: Path,
    official_repo: Path,
    object_id: str,
    quantiles: tuple[float, ...],
    candidate_indices: tuple[int, ...],
    initial_velocity_policies: tuple[str, ...],
    legacy_fit_root: Path | None,
    device: str,
) -> dict[str, Any]:
    cohort = _cohort_record(protocol, object_id)
    source_indices = tuple(map(int, cohort["source_episode_ids"]))
    sources = [
        _load_source(root, cohort, object_id, episode_index)
        for episode_index in source_indices
    ]
    graphs = [build_filament_sparse_graph(source["hulls"][0]) for source in sources]
    model = _load_contact_model(root, object_id)
    twins = {
        quantile: fit_reusable_filament_twin(
            object_id,
            graphs,
            [source["episode_id"] for source in sources],
            [source["geometry_sha256"] for source in sources],
            rest_length_quantile=quantile,
        )
        for quantile in quantiles
    }
    candidates = warp_rope_candidates(WarpRopeFeasibilityConfig())
    rows = []
    for episode_index, source in zip(source_indices, sources, strict=True):
        schedule = contact_state_by_robot_axis(
            source["episode"], model.tactile_group_to_robot_axis
        )
        for quantile, twin in twins.items():
            for velocity_policy in initial_velocity_policies:
                started = time.time()
                observation = build_replication_warp_observation(
                    source["episode_dir"],
                    source["episode_id"],
                    "filament",
                    source["frames"],
                    source["hulls"],
                    schedule,
                    reusable_twin=twin,
                    initial_velocity_policy=velocity_policy,
                )
                runner = OfficialWarpSparseGraphRunner(
                    official_repo,
                    observation.case,
                    WarpRopeFeasibilityConfig(),
                    device=device,
                )
                persistence = score_constant_persistence(observation)
                for candidate_index in candidate_indices:
                    prediction = runner.rollout(candidates[candidate_index])
                    metrics = score_replication_warp_prediction(observation, prediction)
                    strain = sparse_graph_strain_summary(
                        observation.case.graph,
                        prediction,
                        rest_lengths_m=observation.case.object_rest_lengths_m,
                    )
                    stretch_strain = sparse_graph_strain_summary(
                        observation.case.graph,
                        prediction,
                        rest_lengths_m=observation.case.object_rest_lengths_m,
                        spring_family=0,
                    )
                    rows.append(
                        {
                            "episode_id": source["episode_id"],
                            "rest_length_quantile": quantile,
                            "initial_velocity_policy": velocity_policy,
                            "twin_result_sha256": twin.as_artifact()["result_sha256"],
                            "candidate_index": candidate_index,
                            "candidate_parameters": candidates[
                                candidate_index
                            ].as_dict(),
                            "mean_chamfer_m": (
                                float(metrics["mean_m"])
                                if np.isfinite(float(metrics["mean_m"]))
                                else None
                            ),
                            "late_chamfer_m": (
                                float(metrics["late_mean_m"])
                                if np.isfinite(float(metrics["late_mean_m"]))
                                else None
                            ),
                            "persistence_chamfer_m": float(persistence["mean_m"]),
                            "legacy_same_candidate_chamfer_m": _legacy_score(
                                legacy_fit_root,
                                object_id,
                                episode_index,
                                candidate_index,
                            ),
                            "p99_relative_edge_strain": (
                                float(strain["p99"])
                                if np.isfinite(strain["p99"])
                                else None
                            ),
                            "p99_relative_stretch_strain": (
                                float(stretch_strain["p99"])
                                if np.isfinite(stretch_strain["p99"])
                                else None
                            ),
                            "runtime_seconds": time.time() - started,
                        }
                    )
    summaries = []
    for quantile in quantiles:
        for velocity_policy in initial_velocity_policies:
            for candidate_index in candidate_indices:
                selected = [
                    row
                    for row in rows
                    if row["rest_length_quantile"] == quantile
                    and row["initial_velocity_policy"] == velocity_policy
                    and row["candidate_index"] == candidate_index
                ]
                values = np.asarray(
                    [
                        np.inf
                        if row["mean_chamfer_m"] is None
                        else row["mean_chamfer_m"]
                        for row in selected
                    ]
                )
                persistence = np.asarray(
                    [row["persistence_chamfer_m"] for row in selected]
                )
                legacy = np.asarray(
                    [
                        np.nan
                        if row["legacy_same_candidate_chamfer_m"] is None
                        else row["legacy_same_candidate_chamfer_m"]
                        for row in selected
                    ]
                )
                summaries.append(
                    {
                        "rest_length_quantile": quantile,
                        "initial_velocity_policy": velocity_policy,
                        "candidate_index": candidate_index,
                        "mean_chamfer_m": (
                            float(np.mean(values))
                            if np.all(np.isfinite(values))
                            else None
                        ),
                        "persistence_mean_chamfer_m": float(np.mean(persistence)),
                        "win_count_vs_persistence": int(
                            np.count_nonzero(values < persistence)
                        ),
                        "legacy_same_candidate_mean_chamfer_m": (
                            float(np.nanmean(legacy))
                            if np.any(np.isfinite(legacy))
                            else None
                        ),
                        "win_count_vs_legacy_same_candidate": int(
                            np.count_nonzero(values < legacy)
                        ),
                    }
                )
    return {
        "object_id": object_id,
        "source_episode_ids": [source["episode_id"] for source in sources],
        "twins": {
            str(quantile): twin.as_artifact() for quantile, twin in twins.items()
        },
        "rows": rows,
        "summaries": summaries,
    }


def main() -> None:
    args = _parse_args()
    protocol = load_deform360_replication_protocol(args.protocol)
    quantiles = tuple(args.rest_quantile or (0.0, 0.10, 0.25))
    candidate_indices = tuple(args.candidate_index or (21,))
    initial_velocity_policies = tuple(args.initial_velocity_policy or ("zero",))
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("rest quantile is repeated")
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("candidate index is repeated")
    if len(set(initial_velocity_policies)) != len(initial_velocity_policies):
        raise ValueError("initial velocity policy is repeated")
    if not all(0 <= index < 200 for index in candidate_indices):
        raise ValueError("candidate index is outside the locked grid")
    root = args.data_root.resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableFilamentSourceSmoke",
        "protocol_config_sha256": protocol["config_sha256"],
        "objects": [
            _run_object(
                protocol,
                root,
                args.official_phystwin_repo.resolve(),
                object_id,
                quantiles,
                candidate_indices,
                initial_velocity_policies,
                (
                    None
                    if args.legacy_fit_root is None
                    else args.legacy_fit_root.resolve()
                ),
                args.device,
            )
            for object_id in args.object_id
        ],
        "information_boundary": {
            "source_prefix_geometry_used_for_twin_fit": True,
            "source_future_geometry_used_for_diagnostic_scoring": True,
            "calibration_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
            "status": "exploratory-source-only",
        },
    }
    payload["result_sha256"] = _result_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "result_sha256": payload["result_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
