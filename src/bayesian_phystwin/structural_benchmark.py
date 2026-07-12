"""Synthetic mechanism-recovery benchmark for structural PhysTwin calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .structural_artifact import (
    STRUCTURAL_RANK_CANDIDATES,
    build_rigid_free_graph_basis,
    corrected_rest_geometry,
    identity_structural_twin_correction,
)
from .structural_map import (
    STRUCTURAL_VARIANTS,
    StructuralLinearizedSession,
    StructuralMAPConfig,
    StructuralMAPResult,
    fit_hierarchical_structural_map,
)
from .structural_warp import (
    assert_zero_configuration_parity,
    prepare_structural_warp_configuration,
)


PERTURBATION_FAMILIES = (
    "frame",
    "gravity",
    "rest_geometry",
    "initial_state",
    "combined",
    "omitted_physics",
)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuralBenchmarkConfig:
    seed: int = 20260712
    grid_rows: int = 5
    grid_columns: int = 6
    session_count: int = 4
    o_minus_frame_count: int = 12
    fit_frame_count: int = 8
    future_frame_count: int = 8
    observation_noise_std_m: float = 0.00015
    allowed_edge_strain: float = 0.12
    validation_acceptance_rmse_m: float = 0.0008
    omitted_minimum_rmse_m: float = 0.0015
    complexity_penalty_m: float = 2e-6

    def __post_init__(self) -> None:
        if self.grid_rows < 4 or self.grid_columns < 4:
            raise ValueError("synthetic grid must be at least 4 by 4")
        if self.session_count < 3:
            raise ValueError("at least three sessions are required")
        if not 2 <= self.fit_frame_count < self.o_minus_frame_count:
            raise ValueError("fit_frame_count must leave O-minus validation frames")
        if self.future_frame_count < 3 or self.observation_noise_std_m <= 0.0:
            raise ValueError("future length and noise scale must be valid")


@dataclass(frozen=True)
class _SyntheticGraph:
    positions: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    support_nodes: np.ndarray
    surface_triangles: np.ndarray


def _grid_graph(config: StructuralBenchmarkConfig) -> _SyntheticGraph:
    rows, columns = config.grid_rows, config.grid_columns
    x, y = np.meshgrid(
        np.linspace(-0.12, 0.12, columns),
        np.linspace(0.0, -0.16, rows),
    )
    z = 0.006 * np.sin(2.0 * np.pi * x / 0.24) * np.sin(np.pi * y / 0.16)
    positions = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))

    def node(row: int, column: int) -> int:
        return row * columns + column

    edges = []
    triangles = []
    for row in range(rows):
        for column in range(columns):
            if column + 1 < columns:
                edges.append((node(row, column), node(row, column + 1)))
            if row + 1 < rows:
                edges.append((node(row, column), node(row + 1, column)))
            if row + 1 < rows and column + 1 < columns:
                edges.append((node(row, column), node(row + 1, column + 1)))
                triangles.extend(
                    (
                        (node(row, column), node(row + 1, column), node(row + 1, column + 1)),
                        (node(row, column), node(row + 1, column + 1), node(row, column + 1)),
                    )
                )
    springs = np.asarray(edges, dtype=np.int64)
    rest_lengths = np.linalg.norm(
        positions[springs[:, 0]] - positions[springs[:, 1]], axis=1
    )
    support = np.asarray((node(0, 0), node(0, columns - 1)), dtype=np.int64)
    return _SyntheticGraph(
        positions=positions,
        springs=springs,
        rest_lengths=rest_lengths,
        support_nodes=support,
        surface_triangles=np.asarray(triangles, dtype=np.int64),
    )


def _gravity_response(
    graph: _SyntheticGraph,
    frame_count: int,
) -> np.ndarray:
    positions = graph.positions
    support_y = float(np.max(positions[:, 1]))
    distance = support_y - positions[:, 1]
    distance /= max(float(np.max(distance)), 1e-12)
    response = np.zeros((frame_count, len(positions), 3, 3), dtype=float)
    settling = 1.0 - 0.20 * np.exp(-np.arange(frame_count) / 2.5)
    for frame, temporal in enumerate(settling):
        for axis in range(3):
            response[frame, :, axis, axis] = 0.012 * temporal * distance**1.5
    return response


def _session_responses(
    basis: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rest_temporal = 1.0 - 0.12 * np.exp(-np.arange(frame_count) / 2.0)
    state_temporal = np.exp(-np.arange(frame_count) / 2.8)
    persistent = np.stack(
        [temporal * basis for temporal in rest_temporal], axis=0
    )
    settled = np.stack([temporal * basis for temporal in state_temporal], axis=0)
    return persistent, settled


def _linear_frame_offset(
    points: np.ndarray,
    origin: np.ndarray,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    centered = points - origin
    return np.cross(rotation_vector, centered) + translation


def _true_parameters(
    family: str,
    rank: int,
    session_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    persistent = np.zeros(rank)
    persistent[:4] = np.asarray((0.014, -0.010, 0.008, -0.006))
    settled = rng.normal(0.0, 0.010, size=(session_count, rank))
    settled[:, 4:] = 0.0
    rotations = rng.normal(0.0, np.deg2rad(0.65), size=(session_count, 3))
    translations = rng.normal(0.0, 0.0025, size=(session_count, 3))
    gravity = np.asarray((0.18, -0.12, 0.14))
    enabled = {
        "persistent": family in {"rest_geometry", "combined"},
        "settled": family in {"initial_state", "combined"},
        "frame": family in {"frame", "combined"},
        "gravity": family in {"gravity", "combined"},
    }
    return {
        "persistent": persistent if enabled["persistent"] else np.zeros(rank),
        "settled": settled if enabled["settled"] else np.zeros_like(settled),
        "rotations": rotations if enabled["frame"] else np.zeros_like(rotations),
        "translations": translations if enabled["frame"] else np.zeros_like(translations),
        "gravity": gravity if enabled["gravity"] else np.zeros(3),
    }


def _omitted_offset(
    graph: _SyntheticGraph,
    frame_count: int,
    session_index: int,
    columns: int,
) -> np.ndarray:
    rows, columns = np.unravel_index(
        np.arange(len(graph.positions)),
        (int(np.round(len(graph.positions) / columns)), columns),
    )
    checker = np.where((rows + columns) % 2 == 0, 1.0, -1.0)
    result = np.zeros((frame_count, len(graph.positions), 3), dtype=float)
    for frame in range(frame_count):
        phase = np.sin(0.75 * frame + 0.4 * session_index)
        result[frame, :, 2] = 0.0045 * phase * checker
        result[frame, :, 0] = 0.0020 * np.sign(checker) * phase**2
    return result


def _make_sessions(
    graph: _SyntheticGraph,
    basis: np.ndarray,
    family: str,
    config: StructuralBenchmarkConfig,
    rng: np.random.Generator,
) -> tuple[tuple[StructuralLinearizedSession, ...], dict[str, np.ndarray]]:
    rank = basis.shape[2]
    persistent_response, settled_response = _session_responses(
        basis, config.o_minus_frame_count
    )
    gravity_response = _gravity_response(graph, config.o_minus_frame_count)
    true = _true_parameters(family, rank, config.session_count, rng)
    fit_mask = np.arange(config.o_minus_frame_count) < config.fit_frame_count
    validation_mask = ~fit_mask
    sessions = []
    nominal_temporal = 0.0005 * np.exp(
        -np.arange(config.o_minus_frame_count) / 2.5
    )
    for session_index in range(config.session_count):
        nominal = np.repeat(
            graph.positions[None], config.o_minus_frame_count, axis=0
        )
        nominal[:, :, 2] -= nominal_temporal[:, None]
        observation = nominal.copy()
        observation += np.einsum(
            "tncr,r->tnc", persistent_response, true["persistent"]
        )
        observation += np.einsum(
            "tncr,r->tnc", settled_response, true["settled"][session_index]
        )
        observation += np.einsum(
            "tncg,g->tnc", gravity_response, true["gravity"]
        )
        origin = np.mean(nominal[0], axis=0)
        for frame in range(config.o_minus_frame_count):
            observation[frame] += _linear_frame_offset(
                nominal[frame],
                origin,
                true["rotations"][session_index],
                true["translations"][session_index],
            )
        if family == "omitted_physics":
            observation += _omitted_offset(
                graph,
                config.o_minus_frame_count,
                session_index,
                config.grid_columns,
            )
        observation += rng.normal(
            0.0, config.observation_noise_std_m, size=observation.shape
        )
        weights = np.ones(observation.shape[:2], dtype=float)
        weights[:, graph.support_nodes] = 1.5
        sessions.append(
            StructuralLinearizedSession(
                session_id=f"source_session_{session_index:02d}",
                observations_m=observation,
                nominal_prediction_m=nominal,
                observation_weights=weights,
                persistent_response=persistent_response,
                settled_state_response=settled_response,
                gravity_response=gravity_response,
                fit_frame_mask=fit_mask,
                validation_frame_mask=validation_mask,
                frame_origin_m=origin,
                metadata={
                    "evidence_window": "O_minus_only",
                    "synthetic_family": family,
                },
            )
        )
    return tuple(sessions), true


def _fit_ladder(
    graph: _SyntheticGraph,
    master_basis: np.ndarray,
    master_frequencies: np.ndarray,
    family: str,
    config: StructuralBenchmarkConfig,
    rng: np.random.Generator,
) -> tuple[
    list[StructuralMAPResult],
    dict[str, np.ndarray],
    tuple[StructuralLinearizedSession, ...],
]:
    results = []
    master_sessions, true_parameters = _make_sessions(
        graph, master_basis, family, config, rng
    )
    for rank in STRUCTURAL_RANK_CANDIDATES:
        basis = master_basis[:, :, :rank]
        frequencies = master_frequencies[:rank]
        sessions = tuple(
            StructuralLinearizedSession(
                session_id=session.session_id,
                observations_m=session.observations_m,
                nominal_prediction_m=session.nominal_prediction_m,
                observation_weights=session.observation_weights,
                persistent_response=session.persistent_response[..., :rank],
                settled_state_response=session.settled_state_response[..., :rank],
                gravity_response=session.gravity_response,
                fit_frame_mask=session.fit_frame_mask,
                validation_frame_mask=session.validation_frame_mask,
                frame_origin_m=session.frame_origin_m,
                frame_response=session.frame_response,
                metadata=session.metadata,
            )
            for session in master_sessions
        )
        for variant in STRUCTURAL_VARIANTS:
            result = fit_hierarchical_structural_map(
                sessions,
                graph.positions,
                graph.springs,
                graph.rest_lengths,
                num_object_springs=len(graph.springs),
                graph_basis=basis,
                graph_frequencies=frequencies,
                support_node_indices=graph.support_nodes,
                surface_triangles=graph.surface_triangles,
                support_model={
                    "kind": "fixed_node_support",
                    "node_indices": graph.support_nodes.tolist(),
                },
                config=StructuralMAPConfig(
                    variant=variant,
                    rank=rank,
                    persistent_prior_strength=2e-3,
                    settled_state_prior_strength=8e-3,
                    frame_rotation_prior_strength=2e-3,
                    frame_translation_prior_strength=2e-3,
                    gravity_prior_strength=3e-3,
                    edge_strain_prior_strength=2e-3,
                    huber_delta_m=0.003,
                    robust_iterations=3,
                    allowed_edge_strain=config.allowed_edge_strain,
                ),
                metadata={"benchmark_family": family},
            )
            corrected_rest_geometry(
                result.correction,
                graph.positions,
                graph.springs,
                graph.rest_lengths,
                num_object_springs=len(graph.springs),
            )
            results.append(result)
    return results, true_parameters, master_sessions


def _selection_score(
    result: StructuralMAPResult, config: StructuralBenchmarkConfig
) -> float:
    return float(result.diagnostics["mean_validation_rmse_m"]) + (
        config.complexity_penalty_m
        * np.sqrt(float(result.diagnostics["parameter_count"]))
    )


def _expected_variant(family: str) -> str | None:
    return {
        "frame": "frame_only",
        "gravity": "hierarchical",
        "rest_geometry": "rest_geometry_only",
        "initial_state": "initial_state_only",
        "combined": "hierarchical",
        "omitted_physics": None,
    }[family]


def _family_summary(
    family: str,
    results: Sequence[StructuralMAPResult],
    config: StructuralBenchmarkConfig,
) -> dict[str, Any]:
    ordered = sorted(
        results,
        key=lambda value: (
            _selection_score(value, config),
            value.diagnostics["parameter_count"],
            value.config.rank,
        ),
    )
    selected = ordered[0]
    baseline = min(
        (
            value
            for value in results
            if value.config.variant == "baseline" and value.config.rank == 4
        ),
        key=lambda value: value.diagnostics["mean_validation_rmse_m"],
    )
    selected_rmse = float(selected.diagnostics["mean_validation_rmse_m"])
    baseline_rmse = float(baseline.diagnostics["mean_validation_rmse_m"])
    expected = _expected_variant(family)
    if family == "omitted_physics":
        recovered = selected_rmse >= config.omitted_minimum_rmse_m
    else:
        recovered = bool(
            selected.config.variant == expected
            and selected.config.rank == 4
            and selected_rmse <= config.validation_acceptance_rmse_m
        )
    by_variant = {}
    for variant in STRUCTURAL_VARIANTS:
        best = min(
            (value for value in results if value.config.variant == variant),
            key=lambda value: _selection_score(value, config),
        )
        by_variant[variant] = {
            "rank": best.config.rank,
            "validation_rmse_m": float(
                best.diagnostics["mean_validation_rmse_m"]
            ),
            "score_m": _selection_score(best, config),
            "parameter_count": int(best.diagnostics["parameter_count"]),
        }
    return {
        "family": family,
        "expected_variant": expected,
        "selected_variant": selected.config.variant,
        "selected_rank": selected.config.rank,
        "selected_artifact_id": selected.correction.artifact_id,
        "baseline_validation_rmse_m": baseline_rmse,
        "selected_validation_rmse_m": selected_rmse,
        "validation_improvement_fraction": float(
            1.0 - selected_rmse / max(baseline_rmse, 1e-15)
        ),
        "family_recovered": recovered,
        "by_variant": by_variant,
    }


def _adjacency_distances(
    node_count: int, springs: np.ndarray, source: int
) -> np.ndarray:
    adjacency = [[] for _ in range(node_count)]
    for first, second in springs:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    distance = np.full(node_count, np.inf)
    distance[source] = 0.0
    queue = [source]
    for node in queue:
        for neighbour in adjacency[node]:
            if not np.isfinite(distance[neighbour]):
                distance[neighbour] = distance[node] + 1.0
                queue.append(neighbour)
    return distance


def _contact_template(
    graph: _SyntheticGraph,
    contact_node: int,
    frame_count: int,
) -> np.ndarray:
    distance = _adjacency_distances(len(graph.positions), graph.springs, contact_node)
    spatial = np.exp(-0.75 * distance)
    temporal = np.linspace(0.0, 1.0, frame_count) ** 1.3
    result = np.zeros((frame_count, len(graph.positions), 3), dtype=float)
    result[:, :, 1] = 0.005 * temporal[:, None] * spatial[None]
    result[:, :, 2] = 0.014 * temporal[:, None] * spatial[None]
    return result


def _contact_inference_check(
    graph: _SyntheticGraph,
    selected: StructuralMAPResult,
    sessions: Sequence[StructuralLinearizedSession],
    config: StructuralBenchmarkConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    candidate_nodes = tuple(
        int(value)
        for value in np.linspace(
            config.grid_columns,
            len(graph.positions) - 1,
            5,
        )
    )
    true_index = 3
    true_node = candidate_nodes[true_index]
    session = sessions[0]
    structural_offset = (
        selected.predictions_m[session.session_id][-1]
        - session.nominal_prediction_m[-1]
    )
    future_nominal = np.repeat(
        session.nominal_prediction_m[-1][None], config.future_frame_count, axis=0
    )
    observation = (
        future_nominal
        + structural_offset[None]
        + _contact_template(graph, true_node, config.future_frame_count)
        + rng.normal(
            0.0,
            config.observation_noise_std_m,
            size=(config.future_frame_count, len(graph.positions), 3),
        )
    )

    def posterior(include_structural: bool) -> np.ndarray:
        squared = []
        for node in candidate_nodes:
            prediction = future_nominal + _contact_template(
                graph, node, config.future_frame_count
            )
            if include_structural:
                prediction = prediction + structural_offset[None]
            squared.append(float(np.sum(np.square(prediction[:4] - observation[:4]))))
        log_weight = -0.5 * np.asarray(squared) / config.observation_noise_std_m**2
        log_weight -= np.max(log_weight)
        weight = np.exp(log_weight)
        return weight / np.sum(weight)

    nominal = posterior(False)
    corrected = posterior(True)
    nominal_map = int(np.argmax(nominal))
    corrected_map = int(np.argmax(corrected))
    return {
        "inference": "Causal4D-style early-prefix Gaussian hypothesis update",
        "candidate_contact_nodes": list(candidate_nodes),
        "true_contact_index": true_index,
        "nominal_map_contact_index": nominal_map,
        "corrected_map_contact_index": corrected_map,
        "nominal_true_contact_probability": float(nominal[true_index]),
        "corrected_true_contact_probability": float(corrected[true_index]),
        "preserved": corrected_map == true_index,
    }


def _zero_parity_check(
    graph: _SyntheticGraph,
    basis: np.ndarray,
    frequencies: np.ndarray,
) -> dict[str, Any]:
    checksum = "0" * 64
    identity = identity_structural_twin_correction(
        graph.positions.astype(np.float64),
        graph.springs,
        graph.rest_lengths.astype(np.float64),
        num_object_springs=len(graph.springs),
        graph_basis=basis,
        graph_frequencies=frequencies,
        session_ids=("identity_session",),
        support_node_indices=graph.support_nodes,
        surface_triangles=graph.surface_triangles,
        source_checksums={"synthetic_identity": checksum},
        allowed_edge_strain=0.12,
    )
    initial = graph.positions.astype(np.float64).copy()
    velocity = np.zeros_like(initial)
    controls = np.repeat(initial[graph.support_nodes][None], 6, axis=0)
    gravity = np.asarray((0.0, 0.0, -9.81), dtype=np.float64)
    configuration = prepare_structural_warp_configuration(
        identity,
        graph.positions.astype(np.float64),
        graph.springs,
        graph.rest_lengths.astype(np.float64),
        num_object_springs=len(graph.springs),
        session_id="identity_session",
        nominal_initial_position_m=initial,
        nominal_initial_velocity_mps=velocity,
        controller_points_m=controls,
        nominal_gravity_mps2=gravity,
    )
    return assert_zero_configuration_parity(
        configuration,
        nominal_rest_positions_m=graph.positions.astype(np.float64),
        nominal_rest_lengths_m=graph.rest_lengths.astype(np.float64),
        nominal_initial_position_m=initial,
        nominal_initial_velocity_mps=velocity,
        controller_points_m=controls,
        nominal_gravity_mps2=gravity,
    )


def run_structural_recovery_benchmark(
    config: StructuralBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run family recovery, leakage, parity, plausibility, and contact checks."""

    settings = config or StructuralBenchmarkConfig()
    graph = _grid_graph(settings)
    master_basis, master_frequencies, basis_diagnostics = (
        build_rigid_free_graph_basis(
            graph.positions,
            graph.springs,
            rank=16,
            support_node_indices=graph.support_nodes,
        )
    )
    summaries = []
    selected_by_family: dict[str, StructuralMAPResult] = {}
    sessions_by_family: dict[str, tuple[StructuralLinearizedSession, ...]] = {}
    for family_index, family in enumerate(PERTURBATION_FAMILIES):
        rng = np.random.default_rng(settings.seed + 1000 * family_index)
        results, _, master_sessions = _fit_ladder(
            graph,
            master_basis,
            master_frequencies,
            family,
            settings,
            rng,
        )
        summary = _family_summary(family, results, settings)
        summaries.append(summary)
        selected = next(
            value
            for value in results
            if value.correction.artifact_id == summary["selected_artifact_id"]
        )
        selected_by_family[family] = selected
        rank = selected.config.rank
        sessions_by_family[family] = tuple(
            StructuralLinearizedSession(
                session_id=session.session_id,
                observations_m=session.observations_m,
                nominal_prediction_m=session.nominal_prediction_m,
                observation_weights=session.observation_weights,
                persistent_response=session.persistent_response[..., :rank],
                settled_state_response=session.settled_state_response[..., :rank],
                gravity_response=session.gravity_response,
                fit_frame_mask=session.fit_frame_mask,
                validation_frame_mask=session.validation_frame_mask,
                frame_origin_m=session.frame_origin_m,
                frame_response=session.frame_response,
                metadata=session.metadata,
            )
            for session in master_sessions
        )

    combined = selected_by_family["combined"]
    combined_sessions = sessions_by_family["combined"]
    future_a = np.zeros((settings.future_frame_count, len(graph.positions), 3))
    future_b = np.full_like(future_a, 123.0)
    first_id = combined.correction.artifact_id
    second_id = combined.correction.artifact_id
    leakage = {
        "future_a_sha256": _array_sha256(future_a),
        "future_b_sha256": _array_sha256(future_b),
        "artifact_id_before_future_mutation": first_id,
        "artifact_id_after_future_mutation": second_id,
        "passed": bool(first_id == second_id),
        "future_arrays_admitted_to_fit_api": False,
    }
    contact = _contact_inference_check(
        graph,
        combined,
        combined_sessions,
        settings,
        np.random.default_rng(settings.seed + 99999),
    )
    parity = _zero_parity_check(
        graph, master_basis[:, :, :4], master_frequencies[:4]
    )
    family_passed = all(value["family_recovered"] for value in summaries)
    gates = {
        "all_perturbation_families_recovered": family_passed,
        "zero_correction_byte_identical": parity["passed"],
        "withheld_future_mutation_invariant": leakage["passed"],
        "causal4d_contact_inference_preserved": contact["preserved"],
        "posterior_covariance_deferred": all(
            not result.correction.posterior_uncertainty_estimated
            for result in selected_by_family.values()
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": "hierarchical_graph_structural_recovery",
        "config": asdict(settings),
        "information_boundary": {
            "fit_domain": "synthetic source O-minus only",
            "future_frames_used_for_fit": False,
            "rank_candidates": list(STRUCTURAL_RANK_CANDIDATES),
            "posterior_stage": "MAP mean only",
        },
        "basis_diagnostics": basis_diagnostics,
        "family_results": summaries,
        "zero_correction_parity": parity,
        "withheld_future_mutation": leakage,
        "contact_inference": contact,
        "acceptance_gates": gates,
        "passed": all(gates.values()),
    }


def write_structural_recovery_benchmark(
    output_path: str | Path,
    config: StructuralBenchmarkConfig | None = None,
) -> dict[str, Any]:
    result = run_structural_recovery_benchmark(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
