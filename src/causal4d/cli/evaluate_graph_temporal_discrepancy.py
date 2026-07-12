"""Evaluate graph-temporal discrepancy using only O-minus and an O-plus prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import PhysicalPosterior, load_contract
from causal4d.graph_temporal_discrepancy import (
    fit_graph_temporal_discrepancy,
    forecast_graph_temporal_discrepancy,
    graph_laplacian_basis,
)
from causal4d.physical_validation import physical_posterior_moments
from causal4d.real_calibration import (
    RealCalibrationCase,
    evaluate_real_prediction_case,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _state_moments(
    posterior: PhysicalPosterior,
    *,
    variance_floor_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    components = posterior.state_trajectories_m.astype(float)
    mean = np.einsum("k,ktnc->tnc", posterior.weights, components)
    centered = components - mean[None]
    variance = np.einsum(
        "k,ktnc->tnc",
        posterior.weights,
        np.square(centered),
    )
    return mean, variance + variance_floor_m2


def _graph_region_labels(
    springs: np.ndarray,
    *,
    object_spring_count: int,
    object_node_count: int,
    output_node_count: int,
) -> tuple[str, ...]:
    edges = np.asarray(springs, dtype=int)
    adjacency = [[] for _ in range(object_node_count)]
    for first, second in edges[:object_spring_count]:
        if first < object_node_count and second < object_node_count:
            adjacency[int(first)].append(int(second))
            adjacency[int(second)].append(int(first))
    seeds = set()
    for first, second in edges[object_spring_count:]:
        if first < object_node_count <= second:
            seeds.add(int(first))
        elif second < object_node_count <= first:
            seeds.add(int(second))
    distance = np.full(object_node_count, -1, dtype=int)
    queue = deque()
    for seed in sorted(seeds):
        distance[seed] = 0
        queue.append(seed)
    while queue:
        node = queue.popleft()
        if distance[node] >= 2:
            continue
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    labels = []
    for value in distance[:output_node_count]:
        if value == 0:
            labels.append("contact_attachment")
        elif 1 <= value <= 2:
            labels.append("contact_neighborhood")
        else:
            labels.append("far_graph")
    return tuple(labels)


def _evaluate_variant(
    identifier: str,
    mean: np.ndarray,
    variance: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    case_id: str,
    action_id: str,
    start_frame: int,
    labels: tuple[str, ...],
    confidence_level: float,
) -> dict:
    case = RealCalibrationCase(
        case_id=case_id,
        action_id=action_id,
        contact_region_id="released_graph_attachment",
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth,
        valid=valid,
        start_frame=start_frame,
        node_group_labels=labels,
    )
    result = evaluate_real_prediction_case(
        case,
        confidence_level=confidence_level,
    )
    return {
        "method": identifier,
        "groups": result["groups"],
        "worst_group_coverage": result["worst_group_coverage"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("optimal_params_pickle")
    parser.add_argument("parameter_profile_npz")
    parser.add_argument("output_json")
    parser.add_argument("--model-npz")
    parser.add_argument("--moments-npz")
    parser.add_argument("--node-groups-json")
    parser.add_argument(
        "--rank-candidates", type=int, nargs="+", default=(4, 8, 16, 32)
    )
    parser.add_argument("--o-plus-prefix-frames", type=int, default=6)
    parser.add_argument("--variance-floor-m2", type=float, default=2.5e-5)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    data = _load_pickle(args.final_data_pickle)
    optimal = _load_pickle(args.optimal_params_pickle)
    observed = np.asarray(data["object_points"], dtype=float)
    valid = _target_validity(
        np.asarray(data["object_visibilities"], dtype=bool),
        np.asarray(data["object_motions_valid"], dtype=bool),
    )
    structure = np.concatenate(
        (
            observed[0],
            np.asarray(data["surface_points"], dtype=float),
            np.asarray(data["interior_points"], dtype=float),
        ),
        axis=0,
    )
    graph = build_phystwin_spring_graph(
        structure,
        np.asarray(data["controller_points"][0], dtype=float),
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    maximum_rank = max(args.rank_candidates)
    basis, eigenvalues = graph_laplacian_basis(
        len(structure),
        graph.springs[: graph.num_object_springs],
        rank=maximum_rank,
    )
    with np.load(args.parameter_profile_npz, allow_pickle=False) as profile:
        profile_mean = np.asarray(profile["posterior_mean_trajectory"], dtype=float)
    endpoint = artifact.context.o_minus.frame_stop - 1
    if profile_mean.shape[:2] != (len(observed), len(structure)):
        raise ValueError("profile posterior mean does not match the PhysTwin case")
    o_minus_residual = (
        observed[1 : endpoint + 1] - profile_mean[1 : endpoint + 1, : len(observed[0])]
    )
    o_minus_valid = valid[1 : endpoint + 1]
    model = fit_graph_temporal_discrepancy(
        o_minus_residual,
        o_minus_valid,
        basis,
        eigenvalues,
        rank_candidates=args.rank_candidates,
    )

    state_mean, state_variance = _state_moments(
        artifact,
        variance_floor_m2=args.variance_floor_m2,
    )
    current_mean, current_variance = physical_posterior_moments(artifact)
    truth = observed[endpoint:, : state_mean.shape[1]]
    target_valid = valid[endpoint:, : state_mean.shape[1]]
    prefix_frame_count = args.o_plus_prefix_frames + 1
    prefix_residual = truth[:prefix_frame_count] - state_mean[:prefix_frame_count]
    prefix_valid = target_valid[:prefix_frame_count]
    graph_mean, graph_variance = forecast_graph_temporal_discrepancy(
        model,
        prefix_residual,
        prefix_valid,
        total_frame_count=len(state_mean),
        dynamics="learned",
    )
    persistent_mean, persistent_variance = forecast_graph_temporal_discrepancy(
        model,
        prefix_residual,
        prefix_valid,
        total_frame_count=len(state_mean),
        dynamics="persistence",
    )
    graph_mean = graph_mean[:, : state_mean.shape[1]]
    graph_variance = graph_variance[:, : state_mean.shape[1]]
    persistent_mean = persistent_mean[:, : state_mean.shape[1]]
    persistent_variance = persistent_variance[:, : state_mean.shape[1]]
    labels = _graph_region_labels(
        graph.springs,
        object_spring_count=graph.num_object_springs,
        object_node_count=len(structure),
        output_node_count=state_mean.shape[1],
    )
    variants = [
        _evaluate_variant(
            "current_random_walk_readout",
            current_mean,
            current_variance,
            truth,
            target_valid,
            case_id=artifact.context.case_id,
            action_id=artifact.context.u_cf.action_id,
            start_frame=prefix_frame_count,
            labels=labels,
            confidence_level=args.confidence_level,
        ),
        _evaluate_variant(
            "state_only",
            state_mean,
            state_variance,
            truth,
            target_valid,
            case_id=artifact.context.case_id,
            action_id=artifact.context.u_cf.action_id,
            start_frame=prefix_frame_count,
            labels=labels,
            confidence_level=args.confidence_level,
        ),
        _evaluate_variant(
            "graph_persistence",
            state_mean + persistent_mean,
            state_variance + persistent_variance,
            truth,
            target_valid,
            case_id=artifact.context.case_id,
            action_id=artifact.context.u_cf.action_id,
            start_frame=prefix_frame_count,
            labels=labels,
            confidence_level=args.confidence_level,
        ),
        _evaluate_variant(
            "graph_temporal",
            state_mean + graph_mean,
            state_variance + graph_variance,
            truth,
            target_valid,
            case_id=artifact.context.case_id,
            action_id=artifact.context.u_cf.action_id,
            start_frame=prefix_frame_count,
            labels=labels,
            confidence_level=args.confidence_level,
        ),
    ]
    output = Path(args.output_json)
    model_path = (
        Path(args.model_npz) if args.model_npz else output.with_suffix(".model.npz")
    )
    moments_path = (
        Path(args.moments_npz)
        if args.moments_npz
        else output.with_suffix(".moments.npz")
    )
    labels_path = (
        Path(args.node_groups_json)
        if args.node_groups_json
        else output.with_suffix(".node_groups.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    moments_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        basis=model.basis,
        eigenvalues=model.eigenvalues,
        transition=model.transition,
        innovation_covariance=model.innovation_covariance,
        projection_variance_m2=model.projection_variance_m2,
    )
    np.savez_compressed(
        moments_path,
        descriptor_json=np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": artifact.context.case_id,
                    "action_id": artifact.context.u_cf.action_id,
                    "endpoint_frame": endpoint,
                    "start_frame": prefix_frame_count,
                    "methods": [
                        "current_random_walk_readout",
                        "state_only",
                        "graph_persistence",
                        "graph_temporal",
                    ],
                    "future_labels_stored": False,
                },
                sort_keys=True,
            )
        ),
        current_random_walk_readout_mean_m=current_mean,
        current_random_walk_readout_variance_m2=current_variance,
        state_only_mean_m=state_mean,
        state_only_variance_m2=state_variance,
        graph_persistence_mean_m=state_mean + persistent_mean,
        graph_persistence_variance_m2=state_variance + persistent_variance,
        graph_temporal_mean_m=state_mean + graph_mean,
        graph_temporal_variance_m2=state_variance + graph_variance,
    )
    labels_path.write_text(json.dumps(list(labels), indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "evaluation": "causal4d_graph_temporal_discrepancy_v1",
        "case": artifact.context.case_id,
        "label_use": {
            "basis_and_dynamics_fit": "O-minus only",
            "rank_selection": "O-minus validation suffix only",
            "target_initialization": f"first {args.o_plus_prefix_frames} O-plus frames",
            "future": "evaluation only",
        },
        "model": {
            "selected_rank": model.selected_rank,
            "candidate_validation_rmse_m": [
                {"rank": rank, "rmse_m": score}
                for rank, score in model.candidate_validation_rmse_m
            ],
            "spectral_radius_before_clipping": model.spectral_radius_before_clipping,
            "spectral_radius": model.spectral_radius,
            "projection_variance_m2": model.projection_variance_m2.tolist(),
            "fit_frame_count": model.fit_frame_count,
            "model_npz": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "moments_npz": str(moments_path.resolve()),
            "moments_sha256": _sha256(moments_path),
        },
        "node_groups": {
            "path": str(labels_path.resolve()),
            "sha256": _sha256(labels_path),
            "counts": {label: labels.count(label) for label in sorted(set(labels))},
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("physical_posterior", args.physical_posterior_npz),
                ("final_data", args.final_data_pickle),
                ("optimal_params", args.optimal_params_pickle),
                ("parameter_profile", args.parameter_profile_npz),
            )
        },
        "variants": variants,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": result["case"],
                "selected_rank": model.selected_rank,
                "output": str(output.resolve()),
                "variants": {
                    value["method"]: value["groups"]["all"] for value in variants
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
