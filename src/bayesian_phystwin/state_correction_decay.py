"""Post-hoc decay diagnostics for frozen PhysTwin state-correction rollouts."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.dynamic_discrepancy import _array_sha256
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_structural_diagnostic import (
    _attachment_support_nodes,
    _graph_distance,
)


STATE_CORRECTION_DECAY_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_state(values: np.ndarray, name: str) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.ndim != 3 or state.shape[2] != 3:
        raise ValueError(f"{name} must have shape (frame, node, 3)")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be finite")
    return state


def _offset_exponential_fit(
    rms_m: np.ndarray,
    *,
    frame_dt_s: float,
    tail_fraction: float,
    minimum_excess_fraction: float,
) -> dict[str, Any]:
    """Fit decay after the maximum toward a robust empirical tail floor."""

    peak_offset = int(np.argmax(rms_m))
    tail_count = max(3, int(math.ceil(tail_fraction * len(rms_m))))
    tail_count = min(tail_count, len(rms_m))
    tail_floor_m = float(np.median(rms_m[-tail_count:]))
    excess = rms_m[peak_offset:] - tail_floor_m
    peak_excess = float(max(excess[0], 0.0))
    threshold = max(1.0e-12, minimum_excess_fraction * peak_excess)
    selected = np.flatnonzero(excess > threshold)
    result: dict[str, Any] = {
        "model": "tail_floor_plus_exponential_after_peak",
        "peak_frame_offset": peak_offset,
        "tail_fraction": tail_fraction,
        "tail_frame_count": tail_count,
        "tail_floor_m": tail_floor_m,
        "minimum_excess_fraction": minimum_excess_fraction,
        "fit_frame_count": int(len(selected)),
        "time_constant_s": None,
        "half_life_s": None,
        "log_space_r_squared": None,
        "adequate_single_decay": False,
    }
    if len(selected) < 3:
        result["failure_reason"] = "fewer_than_three_frames_above_tail_floor"
        return result

    times_s = frame_dt_s * selected.astype(float)
    log_excess = np.log(excess[selected])
    slope, intercept = np.polyfit(times_s, log_excess, 1)
    if slope >= 0.0:
        result["failure_reason"] = "nondecaying_log_linear_slope"
        return result
    fitted = intercept + slope * times_s
    residual_sum = float(np.sum(np.square(log_excess - fitted)))
    total_sum = float(np.sum(np.square(log_excess - np.mean(log_excess))))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    time_constant_s = float(-1.0 / slope)
    result.update(
        {
            "time_constant_s": time_constant_s,
            "half_life_s": float(math.log(2.0) * time_constant_s),
            "log_space_r_squared": r_squared,
            "adequate_single_decay": bool(r_squared >= 0.80),
        }
    )
    return result


def analyze_state_correction_decay(
    baseline_state_m: np.ndarray,
    corrected_state_m: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    frame_dt_s: float,
    tail_fraction: float = 0.20,
    minimum_excess_fraction: float = 0.05,
) -> dict[str, Any]:
    """Measure how an injected prefix-state perturbation evolves in Warp.

    The interval includes the injection state at ``start_frame`` followed by the
    untouched continuation. The diagnostic reports both total displacement from
    the nominal rollout and retention along the originally injected direction.
    """

    baseline = _finite_state(baseline_state_m, "baseline_state_m")
    corrected = _finite_state(corrected_state_m, "corrected_state_m")
    if baseline.shape != corrected.shape:
        raise ValueError("baseline and corrected states must have matching shapes")
    if not 0 <= start_frame < stop_frame <= len(baseline):
        raise ValueError("state-correction interval is invalid")
    if stop_frame - start_frame < 3:
        raise ValueError("state-correction interval must contain at least three frames")
    if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be positive and finite")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if not 0.0 < minimum_excess_fraction < 1.0:
        raise ValueError("minimum_excess_fraction must lie in (0, 1)")

    difference = corrected[start_frame:stop_frame] - baseline[start_frame:stop_frame]
    rms_m = np.sqrt(np.mean(np.sum(np.square(difference), axis=2), axis=1))
    initial = difference[0].reshape(-1)
    initial_energy = float(initial @ initial)
    if initial_energy <= 0.0:
        raise ValueError("the injected state correction has zero position magnitude")
    flattened = difference.reshape(len(difference), -1)
    aligned_retention = flattened @ initial / initial_energy
    orthogonal = flattened - aligned_retention[:, None] * initial[None, :]
    orthogonal_rms_m = np.sqrt(
        np.mean(np.sum(np.square(orthogonal.reshape(difference.shape)), axis=2), axis=1)
    )
    peak_offset = int(np.argmax(rms_m))
    decay_fit = _offset_exponential_fit(
        rms_m,
        frame_dt_s=frame_dt_s,
        tail_fraction=tail_fraction,
        minimum_excess_fraction=minimum_excess_fraction,
    )
    return {
        "schema_version": STATE_CORRECTION_DECAY_SCHEMA_VERSION,
        "analysis_kind": "prefix_state_correction_decay",
        "interval": {
            "start_frame_inclusive": int(start_frame),
            "stop_frame_exclusive": int(stop_frame),
            "frame_count": int(stop_frame - start_frame),
            "frame_dt_s": float(frame_dt_s),
        },
        "summary": {
            "initial_rms_m": float(rms_m[0]),
            "peak_rms_m": float(rms_m[peak_offset]),
            "peak_frame_offset": peak_offset,
            "final_rms_m": float(rms_m[-1]),
            "final_to_initial_ratio": float(rms_m[-1] / rms_m[0]),
            "final_to_peak_ratio": float(rms_m[-1] / rms_m[peak_offset]),
            "final_aligned_retention": float(aligned_retention[-1]),
            "final_orthogonal_rms_m": float(orthogonal_rms_m[-1]),
        },
        "decay_fit": decay_fit,
        "per_frame": {
            "frame": list(range(start_frame, stop_frame)),
            "elapsed_s": (frame_dt_s * np.arange(len(rms_m))).tolist(),
            "rms_m": rms_m.tolist(),
            "aligned_retention": aligned_retention.tolist(),
            "orthogonal_rms_m": orthogonal_rms_m.tolist(),
        },
        "claim_boundary": (
            "Post-hoc trajectory diagnostic only; it does not select a correction "
            "model or identify the physical source of discrepancy."
        ),
    }


def _retention_group(
    difference_m: np.ndarray,
    selected: np.ndarray,
    *,
    global_final_energy: float,
) -> dict[str, Any]:
    initial = difference_m[0, selected].reshape(-1)
    final = difference_m[-1, selected].reshape(-1)
    initial_energy = float(initial @ initial)
    final_energy = float(final @ final)
    return {
        "node_count": int(np.sum(selected)),
        "initial_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(difference_m[0, selected]), axis=1)))
        ),
        "final_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(difference_m[-1, selected]), axis=1)))
        ),
        "final_to_initial_rms_ratio": (
            float(np.sqrt(final_energy / initial_energy))
            if initial_energy > 0.0
            else None
        ),
        "final_aligned_retention": (
            float(final @ initial / initial_energy) if initial_energy > 0.0 else None
        ),
        "final_energy_fraction_of_global_difference": (
            float(final_energy / global_final_energy)
            if global_final_energy > 0.0
            else None
        ),
    }


def analyze_state_correction_modes(
    baseline_state_m: np.ndarray,
    corrected_state_m: np.ndarray,
    graph_basis: np.ndarray,
    graph_eigenvalues: np.ndarray,
    graph_distance_from_attachment: np.ndarray,
    attachment_nodes: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
    frame_dt_s: float,
) -> dict[str, Any]:
    """Resolve frozen state-correction retention by graph mode and constraint distance."""

    baseline = _finite_state(baseline_state_m, "baseline_state_m")
    corrected = _finite_state(corrected_state_m, "corrected_state_m")
    if baseline.shape != corrected.shape:
        raise ValueError("baseline and corrected states must have matching shapes")
    if not 0 <= start_frame < stop_frame <= len(baseline):
        raise ValueError("state-correction interval is invalid")
    basis = np.asarray(graph_basis, dtype=float)
    eigenvalues = np.asarray(graph_eigenvalues, dtype=float).reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != baseline.shape[1]:
        raise ValueError("graph_basis must cover every state node")
    if basis.shape[1] != len(eigenvalues) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("graph eigenvalues must match the graph basis")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-7, rtol=1e-7):
        raise ValueError("graph_basis must be orthonormal")
    distance = np.asarray(graph_distance_from_attachment, dtype=float).reshape(-1)
    if len(distance) != baseline.shape[1] or not np.all(np.isfinite(distance)):
        raise ValueError("every state node needs a finite attachment distance")
    attachments = np.unique(np.asarray(attachment_nodes, dtype=int).reshape(-1))
    if (
        len(attachments) == 0
        or np.any(attachments < 0)
        or np.any(attachments >= len(distance))
    ):
        raise ValueError("attachment_nodes must identify state nodes")
    if not np.all(distance[attachments] == 0.0):
        raise ValueError("attachment nodes must have zero graph distance")
    if frame_dt_s <= 0.0 or not np.isfinite(frame_dt_s):
        raise ValueError("frame_dt_s must be positive and finite")

    difference = corrected[start_frame:stop_frame] - baseline[start_frame:stop_frame]
    coefficients = np.einsum("nm,tnc->tmc", basis, difference)
    modal_energy = np.sum(np.square(coefficients), axis=2)
    total_energy = np.sum(np.square(difference), axis=(1, 2))
    in_basis_energy = np.sum(modal_energy, axis=1)
    initial_modal_energy = modal_energy[0]
    mode_records = []
    retained_direction_energy = []
    for mode in range(basis.shape[1]):
        initial = coefficients[0, mode]
        final = coefficients[-1, mode]
        initial_energy = float(initial_modal_energy[mode])
        retention = (
            float(final @ initial / initial_energy) if initial_energy > 1e-18 else None
        )
        retained_direction_energy.append(
            0.0 if retention is None else retention * retention * initial_energy
        )
        mode_records.append(
            {
                "mode": mode,
                "eigenvalue": float(eigenvalues[mode]),
                "initial_energy_m2": initial_energy,
                "final_energy_m2": float(modal_energy[-1, mode]),
                "final_to_initial_energy_ratio": (
                    float(modal_energy[-1, mode] / initial_energy)
                    if initial_energy > 1e-18
                    else None
                ),
                "final_directional_retention": retention,
                "final_energy_fraction_of_in_basis": (
                    float(modal_energy[-1, mode] / in_basis_energy[-1])
                    if in_basis_energy[-1] > 0.0
                    else None
                ),
            }
        )
    retained_total = float(np.sum(retained_direction_energy))
    for record, energy in zip(mode_records, retained_direction_energy, strict=True):
        record["retained_direction_energy_fraction"] = (
            float(energy / retained_total) if retained_total > 0.0 else None
        )

    near_cut = float(np.quantile(distance, 1.0 / 3.0))
    far_cut = float(np.quantile(distance, 2.0 / 3.0))
    if near_cut >= far_cut:
        raise ValueError("attachment-distance thirds are not separable")
    groups = {
        "near": distance <= near_cut,
        "middle": (distance > near_cut) & (distance < far_cut),
        "far": distance >= far_cut,
    }
    if any(not np.any(selected) for selected in groups.values()):
        raise ValueError("attachment-distance group is empty")
    global_final_energy = float(total_energy[-1])
    distance_groups = {
        name: {
            "minimum_hops": float(np.min(distance[selected])),
            "maximum_hops": float(np.max(distance[selected])),
            **_retention_group(
                difference,
                selected,
                global_final_energy=global_final_energy,
            ),
        }
        for name, selected in groups.items()
    }
    return {
        "schema_version": STATE_CORRECTION_DECAY_SCHEMA_VERSION,
        "analysis_kind": "prefix_state_correction_mode_and_constraint_retention",
        "interval": {
            "start_frame_inclusive": int(start_frame),
            "stop_frame_exclusive": int(stop_frame),
            "frame_count": int(stop_frame - start_frame),
            "frame_dt_s": float(frame_dt_s),
        },
        "constraint_coverage": {
            "attachment_node_count": int(len(attachments)),
            "attachment_node_fraction": float(len(attachments) / len(distance)),
            "mean_hops_to_attachment": float(np.mean(distance)),
            "median_hops_to_attachment": float(np.median(distance)),
            "p90_hops_to_attachment": float(np.quantile(distance, 0.90)),
            "fraction_within_5_hops": float(np.mean(distance <= 5.0)),
            "near_third_maximum_hops": near_cut,
            "far_third_minimum_hops": far_cut,
        },
        "modal_retention": {
            "modes": mode_records,
            "initial_in_basis_energy_fraction": float(
                in_basis_energy[0] / total_energy[0]
            ),
            "final_in_basis_energy_fraction": (
                float(in_basis_energy[-1] / total_energy[-1])
                if total_energy[-1] > 0.0
                else None
            ),
            "final_out_of_basis_energy_fraction": (
                float(
                    max(total_energy[-1] - in_basis_energy[-1], 0.0) / total_energy[-1]
                )
                if total_energy[-1] > 0.0
                else None
            ),
            "dominant_final_mode": int(np.argmax(modal_energy[-1])),
            "dominant_retained_direction_mode": int(
                np.argmax(retained_direction_energy)
            ),
        },
        "distance_resolved_retention": distance_groups,
        "per_frame": {
            "elapsed_s": (frame_dt_s * np.arange(len(difference))).tolist(),
            "total_energy_m2": total_energy.tolist(),
            "in_basis_energy_m2": in_basis_energy.tolist(),
            "modal_energy_m2": modal_energy.tolist(),
        },
        "claim_boundary": (
            "Trajectory-only post-hoc diagnostic. Mode and attachment-distance "
            "patterns cannot select a new mechanism or establish material constants."
        ),
    }


def audit_frozen_state_correction_modes(
    summary_json: str | Path,
    rollout_npz: str | Path,
    correction_json: str | Path,
    correction_npz: str | Path,
    final_data_pickle: str | Path,
    optimal_params_pickle: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Rebuild and verify the frozen graph, then write mode-resolved retention."""

    paths = {
        "summary_json": Path(summary_json),
        "rollout_npz": Path(rollout_npz),
        "correction_json": Path(correction_json),
        "correction_npz": Path(correction_npz),
        "final_data_pickle": Path(final_data_pickle),
        "optimal_params_pickle": Path(optimal_params_pickle),
    }
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    correction = json.loads(paths["correction_json"].read_text(encoding="utf-8"))
    with paths["final_data_pickle"].open("rb") as handle:
        data = pickle.load(handle)
    with paths["optimal_params_pickle"].open("rb") as handle:
        optimal = pickle.load(handle)
    structure = np.concatenate(
        (
            np.asarray(data["object_points"])[0],
            np.asarray(data["surface_points"]),
            np.asarray(data["interior_points"]),
        ),
        axis=0,
    )
    controller = np.asarray(data["controller_points"])[0]
    graph = build_phystwin_spring_graph(
        structure,
        controller,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    if _array_sha256(graph.springs) != summary["graph"]["springs_sha256"]:
        raise ValueError("rebuilt graph differs from the frozen localization graph")
    with np.load(paths["correction_npz"], allow_pickle=False) as artifact:
        basis = artifact["graph_basis"]
        eigenvalues = artifact["graph_eigenvalues"]
    if _array_sha256(basis) != summary["graph"]["basis_sha256"]:
        raise ValueError("correction basis differs from the frozen localization basis")
    attachments = _attachment_support_nodes(graph, len(structure))
    graph_distance = _graph_distance(
        len(structure),
        graph.springs[: graph.num_object_springs],
        attachments,
    )
    heldout_start, heldout_stop = summary["comparison_contract"][
        "common_heldout_continuation"
    ]
    with np.load(paths["rollout_npz"], allow_pickle=False) as rollout:
        result = analyze_state_correction_modes(
            rollout["mean_global__bpt_particle_baseline"],
            rollout["mean_global__prefix_state_position_velocity"],
            basis,
            eigenvalues,
            graph_distance,
            attachments,
            start_frame=int(heldout_start) - 1,
            stop_frame=int(heldout_stop),
            frame_dt_s=float(correction["frame_dt_s"]),
        )
    result.update(
        {
            "case": summary["case"],
            "experiment": summary["experiment"],
            "graph_verification": {
                "springs_sha256": summary["graph"]["springs_sha256"],
                "basis_sha256": summary["graph"]["basis_sha256"],
                "verified": True,
            },
            "information_boundary": {
                "future_residuals_opened": False,
                "manual_tracks_opened": False,
                "outcome_metrics_opened": False,
                "frame_zero_geometry_and_controller_attachments_only": True,
            },
            "source_checksums": {
                name: _file_sha256(path) for name, path in paths.items()
            },
        }
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def aggregate_state_correction_modes(
    result_jsons: list[str | Path], output_json: str | Path
) -> dict[str, Any]:
    """Summarize three frozen cases without inferential claims."""

    results = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in result_jsons
    ]
    if len(results) < 2 or len({result["case"] for result in results}) != len(results):
        raise ValueError("mode-retention aggregation needs distinct cases")
    coverage = np.asarray(
        [result["constraint_coverage"]["fraction_within_5_hops"] for result in results]
    )
    retention = np.asarray(
        [
            abs(result["distance_resolved_retention"]["far"]["final_aligned_retention"])
            for result in results
        ]
    )
    global_retention = np.asarray(
        [
            abs(
                json.loads(
                    Path(path)
                    .with_name("state_correction_decay.json")
                    .read_text(encoding="utf-8")
                )["summary"]["final_aligned_retention"]
            )
            for path in result_jsons
        ]
    )
    result = {
        "schema_version": STATE_CORRECTION_DECAY_SCHEMA_VERSION,
        "analysis_kind": "state_correction_mode_constraint_aggregate",
        "case_count": len(results),
        "cases": {
            item["case"]: {
                "constraint_coverage": item["constraint_coverage"],
                "modal_retention": item["modal_retention"],
                "distance_resolved_retention": item["distance_resolved_retention"],
            }
            for item in results
        },
        "descriptive_correlations": {
            "within_5_hops_vs_absolute_global_retention": float(
                np.corrcoef(coverage, global_retention)[0, 1]
            ),
            "within_5_hops_vs_absolute_far_graph_retention": float(
                np.corrcoef(coverage, retention)[0, 1]
            ),
        },
        "claim_boundary": (
            "Three-case descriptive trajectory audit only. Correlations have no "
            "sampling interpretation and cannot select the physical mechanism."
        ),
        "source_checksums": {
            Path(path).stem + f"_{index}": _file_sha256(Path(path))
            for index, path in enumerate(result_jsons)
        },
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def audit_frozen_state_correction_decay(
    summary_json: str | Path,
    rollout_npz: str | Path,
    correction_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Audit one frozen localization case and write a checksummed JSON result."""

    summary_path = Path(summary_json)
    rollout_path = Path(rollout_npz)
    correction_path = Path(correction_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    heldout_start, heldout_stop = summary["comparison_contract"][
        "common_heldout_continuation"
    ]
    with np.load(rollout_path, allow_pickle=False) as archive:
        result = analyze_state_correction_decay(
            archive["mean_global__bpt_particle_baseline"],
            archive["mean_global__prefix_state_position_velocity"],
            start_frame=int(heldout_start) - 1,
            stop_frame=int(heldout_stop),
            frame_dt_s=float(correction["frame_dt_s"]),
        )
    result.update(
        {
            "case": summary["case"],
            "experiment": summary["experiment"],
            "source_checksums": {
                "summary_json": _file_sha256(summary_path),
                "rollout_npz": _file_sha256(rollout_path),
                "correction_json": _file_sha256(correction_path),
            },
        }
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
