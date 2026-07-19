"""Open-panel diagnostic for topology-aware Deform360 belief decoding.

This module is intentionally restricted to the 27 already-open independent
source outcomes used by ``deform360_online_belief_evaluation``.  It never
discovers episodes from the dataset tree and refuses any case outside that
module's fixed source whitelist.  The KNN topology proxy is built from the
sealed physical prior at frame zero; future frames do not affect its edges.

The diagnostic reuses the frozen update acceptance and causal-continuation
decisions recorded by the audited Euclidean-RBF run.  Consequently it isolates
only the decoder metric: current Euclidean embedding versus fixed material
graph distance.  A KNN sweep is exploratory and must not be called held-out
evidence.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_online_belief_evaluation import (
    CENTER_COUNT,
    EXPECTED_SOURCE_EPISODES,
    PROTOCOL_ID as SOURCE_PROTOCOL_ID,
    UPDATE_FRAMES,
    score_deform360_hidden_trajectory,
)
from .phystwin_geodesic_belief import (
    MaterialGeodesicGraph,
    build_reference_knn_geodesic_graph,
    decode_recursive_geodesic_rbf_belief,
    geodesic_distances_to_centers_m,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


DIAGNOSTIC_PROTOCOL_ID = "deform360-open27-geodesic-rbf-development-v1"
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)


def _expected_cases() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _component_count(graph: MaterialGeodesicGraph) -> int:
    adjacency: list[list[int]] = [[] for _ in range(graph.node_count)]
    for left, right in graph.edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    visited = np.zeros(graph.node_count, dtype=bool)
    count = 0
    for seed in range(graph.node_count):
        if visited[seed]:
            continue
        count += 1
        visited[seed] = True
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return count


def _corrected_frame(
    prior_frame_m: np.ndarray,
    correction_m: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    return (
        np.asarray(prior_frame_m, dtype=float) + np.asarray(correction_m, dtype=float)
    ).astype(dtype, copy=False)


def _symmetric_set_chamfer_m(first_m: np.ndarray, second_m: np.ndarray) -> float:
    first = np.asarray(first_m, dtype=float)
    second = np.asarray(second_m, dtype=float)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1:] != (3,)
        or second.shape[1:] != (3,)
        or len(first) == 0
        or len(second) == 0
    ):
        raise ValueError("Chamfer inputs must have nonempty shape (N, 3)")
    distance = np.linalg.norm(first[:, None] - second[None], axis=2)
    return 0.5 * (
        float(np.mean(np.min(distance, axis=1)))
        + float(np.mean(np.min(distance, axis=0)))
    )


def evaluate_geodesic_decoder_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_records: Sequence[Mapping[str, Any]],
    belief_config: RecursiveRbfBeliefConfig,
    neighbor_count: int,
    scored_frames: Sequence[int],
    measurement_center_m: np.ndarray | None = None,
    measurement_available: np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Re-decode one open case with a frame-zero material-graph proxy."""

    prior_input = np.asarray(physical_prior_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence_input = np.asarray(persistence_m)
    persistence = np.asarray(persistence_input, dtype=float)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    frames = tuple(int(frame) for frame in scored_frames)
    if (
        prior.shape != persistence.shape
        or prior.shape != target.shape
        or prior.ndim != 3
        or prior.shape[2] != 3
    ):
        raise ValueError(
            "physical prior, persistence, and target must share shape (T, N, 3)"
        )
    if visible.shape != target.shape[:2] or valid.shape != target.shape[:2]:
        raise ValueError("visibility and validity must have shape (T, N)")
    if centers.shape != (CENTER_COUNT,) or len(np.unique(centers)) != len(centers):
        raise ValueError(f"center_ids must contain {CENTER_COUNT} unique IDs")
    if np.any(centers < 0) or np.any(centers >= target.shape[1]):
        raise ValueError("center ID exceeds the material trajectory")
    if not np.array_equal(
        prior_input[0].astype(np.float32), target[0].astype(np.float32)
    ):
        raise ValueError("frame-zero material identities differ")
    if tuple(int(record["frame"]) for record in update_records) != UPDATE_FRAMES:
        raise ValueError("update records differ from the fixed Deform360 schedule")
    if measurement_center_m is None:
        measurement = target[:, centers]
    else:
        measurement = np.asarray(measurement_center_m, dtype=float)
    if measurement_available is None:
        measurement_mask = (
            visible[:, centers]
            & valid[:, centers]
            & np.all(np.isfinite(measurement), axis=2)
        )
    else:
        measurement_mask = np.asarray(measurement_available, dtype=bool)
    if measurement.shape != (len(target), len(centers), 3):
        raise ValueError("measurement_center_m must have shape (T, K, 3)")
    if measurement_mask.shape != (len(target), len(centers)):
        raise ValueError("measurement_available must have shape (T, K)")

    graph = build_reference_knn_geodesic_graph(prior[0], neighbor_count=neighbor_count)
    center_distance = geodesic_distances_to_centers_m(graph, centers)
    risk_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=belief_config,
    )
    ungated_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=belief_config,
    )
    matched_euclidean_risk_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=belief_config,
    )
    matched_euclidean_ungated_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=belief_config,
    )
    backbones = {"physical_prior": prior, "persistence": persistence}
    selected_geodesic_beliefs = {
        name: initialize_recursive_rbf_belief(
            centers,
            trajectory[0, centers],
            trajectory[0],
            config=belief_config,
        )
        for name, trajectory in backbones.items()
    }
    selected_euclidean_beliefs = {
        name: initialize_recursive_rbf_belief(
            centers,
            trajectory[0, centers],
            trajectory[0],
            config=belief_config,
        )
        for name, trajectory in backbones.items()
    }
    output_dtype = prior_input.dtype
    risk = prior_input.copy()
    ungated = prior_input.copy()
    causal = prior_input.copy()
    frozen_current = prior_input.copy()
    matched_euclidean_risk = prior_input.copy()
    matched_euclidean_ungated = prior_input.copy()
    matched_euclidean_causal = prior_input.copy()
    selected_geodesic = prior_input.copy()
    selected_euclidean = prior_input.copy()
    selected_backbones: list[str | None] = []
    query_ids = np.arange(prior.shape[1], dtype=np.int64)

    for update_index, (update, record) in enumerate(
        zip(UPDATE_FRAMES, update_records, strict=True)
    ):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(prior)
        )
        available = (
            measurement_mask[update]
            & np.all(np.isfinite(measurement[update]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
        )
        if int(np.sum(available)) != int(record["available_center_count"]):
            raise ValueError("recorded centre support differs from the open outcome")
        residual = np.full((len(centers), 3), np.nan, dtype=float)
        residual[available] = (
            measurement[update, available] - prior[update, centers[available]]
        )
        if np.any(available):
            available_ids = centers[available]
            target_current = measurement[update, available]
            physical_current_chamfer = _symmetric_set_chamfer_m(
                prior[update, available_ids], target_current
            )
            persistence_current_chamfer = _symmetric_set_chamfer_m(
                persistence[update, available_ids], target_current
            )
            selected_backbone = (
                "physical_prior"
                if physical_current_chamfer <= persistence_current_chamfer
                else "persistence"
            )
            selected_trajectory = (
                prior if selected_backbone == "physical_prior" else persistence
            )
            for backbone_name, backbone in backbones.items():
                backbone_residual = np.full((len(centers), 3), np.nan, dtype=float)
                backbone_residual[available] = (
                    target_current - backbone[update, available_ids]
                )
                selected_geodesic_beliefs[backbone_name], _ = (
                    update_recursive_rbf_belief(
                        selected_geodesic_beliefs[backbone_name],
                        update,
                        backbone[update, centers],
                        backbone_residual,
                        available,
                        config=belief_config,
                    )
                )
                selected_euclidean_beliefs[backbone_name], _ = (
                    update_recursive_rbf_belief(
                        selected_euclidean_beliefs[backbone_name],
                        update,
                        backbone[update, centers],
                        backbone_residual,
                        available,
                        config=belief_config,
                    )
                )
            selected_geodesic_belief = selected_geodesic_beliefs[selected_backbone]
            selected_euclidean_belief = selected_euclidean_beliefs[selected_backbone]
            ungated_belief, _ = update_recursive_rbf_belief(
                ungated_belief,
                update,
                prior[update, centers],
                residual,
                available,
                config=belief_config,
            )
            matched_euclidean_ungated_belief, _ = update_recursive_rbf_belief(
                matched_euclidean_ungated_belief,
                update,
                prior[update, centers],
                residual,
                available,
                config=belief_config,
            )
            for frame in range(update + 1, stop):
                ungated_decoded = decode_recursive_geodesic_rbf_belief(
                    ungated_belief,
                    graph,
                    query_ids,
                    forecast_frames=frame - update,
                    config=belief_config,
                    distances_to_belief_centers_m=center_distance,
                )
                ungated[frame] = _corrected_frame(
                    prior[frame], ungated_decoded.mean_m, dtype=output_dtype
                )
                matched_euclidean_ungated_decoded = decode_recursive_rbf_belief(
                    matched_euclidean_ungated_belief,
                    prior[update],
                    forecast_frames=frame - update,
                    config=belief_config,
                )
                matched_euclidean_ungated[frame] = _corrected_frame(
                    prior[frame],
                    matched_euclidean_ungated_decoded.mean_m,
                    dtype=output_dtype,
                )
                selected_geodesic_decoded = decode_recursive_geodesic_rbf_belief(
                    selected_geodesic_belief,
                    graph,
                    query_ids,
                    forecast_frames=frame - update,
                    config=belief_config,
                    distances_to_belief_centers_m=center_distance,
                )
                selected_geodesic[frame] = _corrected_frame(
                    selected_trajectory[frame],
                    selected_geodesic_decoded.mean_m,
                    dtype=output_dtype,
                )
                selected_euclidean_decoded = decode_recursive_rbf_belief(
                    selected_euclidean_belief,
                    selected_trajectory[update],
                    forecast_frames=frame - update,
                    config=belief_config,
                )
                selected_euclidean[frame] = _corrected_frame(
                    selected_trajectory[frame],
                    selected_euclidean_decoded.mean_m,
                    dtype=output_dtype,
                )
            selected_backbones.append(selected_backbone)
        else:
            selected_backbones.append(None)
        accepted = bool(record["accepted"])
        if not accepted:
            if not np.array_equal(
                risk[update + 1 : stop], prior_input[update + 1 : stop]
            ):
                raise AssertionError("rejected geodesic interval changed the prior")
            continue

        risk_belief, _ = update_recursive_rbf_belief(
            risk_belief,
            update,
            prior[update, centers],
            residual,
            available,
            config=belief_config,
        )
        matched_euclidean_risk_belief, _ = update_recursive_rbf_belief(
            matched_euclidean_risk_belief,
            update,
            prior[update, centers],
            residual,
            available,
            config=belief_config,
        )
        frozen_decoded = decode_recursive_geodesic_rbf_belief(
            risk_belief,
            graph,
            query_ids,
            forecast_frames=0,
            config=belief_config,
            distances_to_belief_centers_m=center_distance,
        )
        frozen_state = _corrected_frame(
            prior[update], frozen_decoded.mean_m, dtype=output_dtype
        )
        matched_euclidean_frozen_decoded = decode_recursive_rbf_belief(
            matched_euclidean_risk_belief,
            prior[update],
            forecast_frames=0,
            config=belief_config,
        )
        matched_euclidean_frozen_state = _corrected_frame(
            prior[update],
            matched_euclidean_frozen_decoded.mean_m,
            dtype=output_dtype,
        )
        continuation_selected = bool(record["causal_continuation_selected"])
        for frame in range(update + 1, stop):
            decoded = decode_recursive_geodesic_rbf_belief(
                risk_belief,
                graph,
                query_ids,
                forecast_frames=frame - update,
                config=belief_config,
                distances_to_belief_centers_m=center_distance,
            )
            risk[frame] = _corrected_frame(
                prior[frame], decoded.mean_m, dtype=output_dtype
            )
            matched_euclidean_decoded = decode_recursive_rbf_belief(
                matched_euclidean_risk_belief,
                prior[update],
                forecast_frames=frame - update,
                config=belief_config,
            )
            matched_euclidean_risk[frame] = _corrected_frame(
                prior[frame],
                matched_euclidean_decoded.mean_m,
                dtype=output_dtype,
            )
            frozen_current[frame] = frozen_state
            causal[frame] = risk[frame] if continuation_selected else frozen_state
            matched_euclidean_causal[frame] = (
                matched_euclidean_risk[frame]
                if continuation_selected
                else matched_euclidean_frozen_state
            )

    trajectories = {
        "recursive_geodesic_rbf_ungated": ungated,
        "recursive_geodesic_rbf_risk_limited": risk,
        "recursive_geodesic_rbf_causal_continuation": causal,
        "geodesic_risk_limited_frozen_current_state": frozen_current,
        "matched_euclidean_rbf_ungated": matched_euclidean_ungated,
        "matched_euclidean_rbf_risk_limited": matched_euclidean_risk,
        "matched_euclidean_rbf_causal_continuation": matched_euclidean_causal,
        "selected_backbone_geodesic_rbf_ungated": selected_geodesic,
        "selected_backbone_euclidean_rbf_ungated": selected_euclidean,
    }
    scores = {
        name: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visible,
            valid,
            center_ids=centers,
            scored_frames=frames,
        )
        for name, trajectory in trajectories.items()
    }
    finite_distance = center_distance[np.isfinite(center_distance)]
    report: dict[str, object] = {
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "status": "post-hoc open-panel decoder diagnostic",
        "neighbor_count": int(neighbor_count),
        "center_ids": centers.tolist(),
        "belief_config": asdict(belief_config),
        "topology": {
            "construction": graph.construction,
            "frame": 0,
            "future_blind": True,
            "node_count": graph.node_count,
            "edge_count": len(graph.edges),
            "component_count": _component_count(graph),
            "finite_node_center_distance_fraction": float(
                len(finite_distance) / center_distance.size
            ),
            "median_finite_node_center_distance_m": float(np.median(finite_distance)),
        },
        "decision_source": (
            "frozen accepted/rejected and causal-continuation decisions from "
            "the audited Euclidean-RBF source run"
        ),
        "observed_backbone_selector": {
            "metric": "current observed-centre symmetric set Chamfer",
            "tie_break": "physical_prior",
            "belief_state_reference": (
                "one recursively updated discrepancy belief per backbone; "
                "the selected state is decoded without cross-backbone mixing"
            ),
            "selected_by_update": selected_backbones,
            "physical_prior_count": int(
                sum(value == "physical_prior" for value in selected_backbones)
            ),
            "persistence_count": int(
                sum(value == "persistence" for value in selected_backbones)
            ),
        },
        "scores": scores,
    }
    return report, trajectories


def _metric_value(score: Mapping[str, Any], metric: str) -> float:
    value = float(score[metric])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite {metric}")
    return value


def _aggregate(
    cases: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in METRICS:
        candidate_values = np.asarray(
            [_metric_value(case["scores"][candidate], metric) for case in cases]
        )
        baseline_values = np.asarray(
            [_metric_value(case["baseline_scores"][baseline], metric) for case in cases]
        )
        delta = candidate_values - baseline_values
        result[metric] = {
            "candidate_mean_m": float(np.mean(candidate_values)),
            "baseline_mean_m": float(np.mean(baseline_values)),
            "relative_change_percent": float(
                100.0 * (np.mean(candidate_values) / np.mean(baseline_values) - 1.0)
            ),
            "paired_win_count": int(np.sum(delta < 0.0)),
            "paired_tie_count": int(np.sum(delta == 0.0)),
            "paired_case_count": len(cases),
            "median_paired_delta_m": float(np.median(delta)),
        }
    return result


def run_open_deform360_geodesic_diagnostic(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    neighbor_counts: Sequence[int] = (2, 4, 6, 8, 12),
    belief_config_overrides: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Run a fail-closed KNN sweep on exactly the audited open 27 cases."""

    source = Path(input_dir).resolve()
    expected = _expected_cases()
    observed = tuple(
        path.stem
        for path in sorted(source.glob("*.json"))
        if path.name != "summary.json"
    )
    if observed != tuple(sorted(expected)):
        raise ValueError("input directory is not exactly the fixed open 27 panel")
    counts = tuple(int(value) for value in neighbor_counts)
    if (
        not counts
        or any(value < 1 for value in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("neighbor_counts must be unique positive integers")
    overrides = dict(belief_config_overrides or {})
    permitted_overrides = {"length_scale_fraction", "local_blend"}
    if not set(overrides).issubset(permitted_overrides):
        raise ValueError("only length_scale_fraction and local_blend may be overridden")

    by_count: dict[str, list[dict[str, object]]] = {str(count): [] for count in counts}
    input_hashes: dict[str, dict[str, str]] = {}
    for case in expected:
        report_path = source / f"{case}.json"
        archive_path = source / f"{case}.npz"
        source_report = json.loads(report_path.read_text())
        if source_report.get("protocol_id") != SOURCE_PROTOCOL_ID:
            raise ValueError(f"{case} does not use the audited source protocol")
        if source_report.get("case") != case:
            raise ValueError(f"{case} report identity changed")
        target_record = source_report.get("inputs", {}).get("target_data", {})
        target_path = Path(str(target_record.get("path", ""))).resolve()
        if (
            target_path.parent.name != case
            or "independent-source-v1" not in target_path.parts
        ):
            raise ValueError(f"{case} target path is outside the open source panel")
        expected_target_hash = str(target_record.get("sha256", ""))
        if not expected_target_hash or _sha256(target_path) != expected_target_hash:
            raise ValueError(f"{case} target checksum changed")
        target_payload = _load_pickle(target_path)
        if not isinstance(target_payload, Mapping):
            raise ValueError(f"{case} target payload is not a mapping")
        with np.load(archive_path, allow_pickle=False) as archive:
            prior = archive["physical_prior_m"]
            persistence = archive["persistence_m"]
            centers = archive["center_ids"]
            euclidean_ungated = archive["recursive_rbf_ungated_m"]
            euclidean_risk = archive["recursive_rbf_risk_limited_m"]
            euclidean_causal = archive["recursive_rbf_causal_continuation_m"]
        target = np.asarray(target_payload["object_points"], dtype=float)
        visibility = np.asarray(target_payload["object_visibilities"], dtype=bool)
        validity = np.asarray(target_payload["object_motions_valid"], dtype=bool)
        scored_frames = tuple(int(frame) for frame in source_report["scored_frames"])
        baseline_recomputed = {
            "recursive_rbf_ungated": score_deform360_hidden_trajectory(
                euclidean_ungated,
                target,
                visibility,
                validity,
                center_ids=centers,
                scored_frames=scored_frames,
            ),
            "recursive_rbf_risk_limited": score_deform360_hidden_trajectory(
                euclidean_risk,
                target,
                visibility,
                validity,
                center_ids=centers,
                scored_frames=scored_frames,
            ),
            "recursive_rbf_causal_continuation": score_deform360_hidden_trajectory(
                euclidean_causal,
                target,
                visibility,
                validity,
                center_ids=centers,
                scored_frames=scored_frames,
            ),
        }
        for arm, score in baseline_recomputed.items():
            for metric in METRICS:
                if not np.isclose(
                    _metric_value(score, metric),
                    _metric_value(source_report["scores"][arm], metric),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(f"{case} baseline score differs from its report")

        config = replace(
            RecursiveRbfBeliefConfig(**source_report["belief_config"]),
            **overrides,
        )
        input_hashes[case] = {
            "source_report_sha256": _sha256(report_path),
            "source_archive_sha256": _sha256(archive_path),
            "target_data_sha256": expected_target_hash,
        }
        for count in counts:
            report, _ = evaluate_geodesic_decoder_arrays(
                prior,
                persistence,
                target,
                visibility,
                validity,
                center_ids=centers,
                update_records=source_report["updates"],
                belief_config=config,
                neighbor_count=count,
                scored_frames=scored_frames,
            )
            by_count[str(count)].append(
                {
                    "case": case,
                    "object_id": source_report["object_id"],
                    "episode_id": source_report["episode_id"],
                    "topology": report["topology"],
                    "scores": report["scores"],
                    "baseline_scores": {
                        **baseline_recomputed,
                        "matched_euclidean_rbf_ungated": report["scores"][
                            "matched_euclidean_rbf_ungated"
                        ],
                        "matched_euclidean_rbf_risk_limited": report["scores"][
                            "matched_euclidean_rbf_risk_limited"
                        ],
                        "matched_euclidean_rbf_causal_continuation": report["scores"][
                            "matched_euclidean_rbf_causal_continuation"
                        ],
                        "selected_backbone_euclidean_rbf_ungated": report["scores"][
                            "selected_backbone_euclidean_rbf_ungated"
                        ],
                    },
                }
            )

    aggregate: dict[str, object] = {}
    for count in counts:
        cases = by_count[str(count)]
        aggregate[str(count)] = {
            "ungated_vs_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_ungated",
                baseline="recursive_rbf_ungated",
            ),
            "ungated_vs_hyperparameter_matched_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_ungated",
                baseline="matched_euclidean_rbf_ungated",
            ),
            "risk_limited_vs_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_risk_limited",
                baseline="recursive_rbf_risk_limited",
            ),
            "risk_limited_vs_hyperparameter_matched_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_risk_limited",
                baseline="matched_euclidean_rbf_risk_limited",
            ),
            "causal_continuation_vs_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_causal_continuation",
                baseline="recursive_rbf_causal_continuation",
            ),
            "causal_continuation_vs_hyperparameter_matched_euclidean": _aggregate(
                cases,
                candidate="recursive_geodesic_rbf_causal_continuation",
                baseline="matched_euclidean_rbf_causal_continuation",
            ),
            "selected_backbone_geodesic_vs_euclidean": _aggregate(
                cases,
                candidate="selected_backbone_geodesic_rbf_ungated",
                baseline="selected_backbone_euclidean_rbf_ungated",
            ),
            "connected_case_count": int(
                sum(case["topology"]["component_count"] == 1 for case in cases)
            ),
            "case_count": len(cases),
        }
    result: dict[str, object] = {
        "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
        "status": (
            "post-hoc exploratory development on already-open source outcomes; "
            "not held-out evidence"
        ),
        "input_protocol_id": SOURCE_PROTOCOL_ID,
        "case_count": len(expected),
        "cases": list(expected),
        "neighbor_counts": list(counts),
        "belief_config_overrides": overrides,
        "future_blind_graph_construction": True,
        "input_hashes": input_hashes,
        "aggregate": aggregate,
        "by_neighbor_count": by_count,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
