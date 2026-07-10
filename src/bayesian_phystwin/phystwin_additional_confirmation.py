"""Label-free confirmation of capped persistent discrepancy anchoring."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .phystwin_comparison import (
    _nearest_distances,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import _lock_protocol
from .phystwin_residual_dynamics import (
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


def _chamfer_by_frame(
    trajectory: np.ndarray,
    observed: np.ndarray,
    visible: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    values = np.empty(end_frame - start_frame, dtype=float)
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        current_observed = observed[frame, visible[frame]]
        distance, _ = _nearest_distances(
            trajectory[frame, :num_surface_points],
            current_observed,
            p=1,
        )
        values[output_index] = np.mean(distance)
    return values


def apply_persistent_residual_anchor(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
) -> dict[str, object]:
    """Hold the final valid training residual without labels or model selection."""

    if train_end_frame < 2:
        raise ValueError("train_end_frame must include at least two frames")
    if maximum_residual_m <= 0.0:
        raise ValueError("maximum_residual_m must be positive")
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    frame_count, original_count, _ = observed.shape
    if not train_end_frame < frame_count:
        raise ValueError("train_end_frame must precede future frames")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover the observations")
    baseline = baseline[:frame_count]
    residual = observed - baseline[:, :original_count]
    valid = _target_validity(visible, motion_valid)
    filled = _temporally_fill(residual, valid, train_end_frame)
    tracked_endpoint = filled[-1]
    future_count = frame_count - train_end_frame
    tracked_future = np.repeat(tracked_endpoint[None], future_count, axis=0)
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, interpolation_neighbors
    )
    correction = _lift_residual(
        tracked_future,
        baseline.shape[1],
        lift_indices,
        lift_weights,
        maximum_norm=maximum_residual_m,
    )
    corrected = baseline.copy()
    corrected[train_end_frame:] += correction
    num_surface_points = original_count + len(np.asarray(data["surface_points"]))
    baseline_cd = _chamfer_by_frame(
        baseline,
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=train_end_frame,
        end_frame=frame_count,
    )
    corrected_cd = _chamfer_by_frame(
        corrected,
        observed,
        visible,
        num_surface_points=num_surface_points,
        start_frame=train_end_frame,
        end_frame=frame_count,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(corrected.astype(np.float32), handle, protocol=pickle.HIGHEST_PROTOCOL)
    correction_norm = np.linalg.norm(correction, axis=2)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "train_end_frame": train_end_frame,
            "maximum_residual_m": maximum_residual_m,
            "interpolation_neighbors": interpolation_neighbors,
        },
        "contract": {
            "training_inputs": "released pseudo-measurements through train_end_frame",
            "future_inputs": "none",
            "manual_labels": "none",
            "selection": "none",
            "future_mean": "final temporally filled residual held constant",
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "baseline_trajectory": {
                "path": str(Path(baseline_trajectory_path).resolve()),
                "sha256": _sha256(baseline_trajectory_path),
            },
        },
        "future": {
            "frame_interval": [train_end_frame, frame_count],
            "baseline_chamfer_by_frame_m": baseline_cd.tolist(),
            "corrected_chamfer_by_frame_m": corrected_cd.tolist(),
            "baseline_chamfer_m": float(np.mean(baseline_cd)),
            "corrected_chamfer_m": float(np.mean(corrected_cd)),
            "percent_change": 100.0
            * (float(np.mean(corrected_cd)) / float(np.mean(baseline_cd)) - 1.0),
        },
        "correction": {
            "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
            "maximum_m": float(np.max(correction_norm, initial=0.0)),
            "saturated_fraction": float(
                np.mean(correction_norm >= 0.999 * maximum_residual_m)
            ),
        },
        "outputs": {"trajectory": str(trajectory_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    return summary


def run_additional_anchor_confirmation(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260710,
    force: bool = False,
) -> dict[str, object]:
    """Confirm the frozen label-free anchor on every additional cloth case."""

    root = Path(data_root)
    source_manifest_path = root / "additional_evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = tuple(str(case) for case in source_manifest["selected_cases"])
    clusters = {
        case: phystwin_physical_object_cluster(case) for case in selected
    }
    output = Path(output_dir)
    specification = {
        "method": "ungated capped persistent residual anchor",
        "maximum_residual_m": maximum_residual_m,
        "interpolation_neighbors": interpolation_neighbors,
        "future_inputs": "none",
        "manual_labels": "none",
        "model_selection": "none",
        "cohort": list(selected),
        "physical_object_clusters": clusters,
        "data_manifest": str(source_manifest_path.resolve()),
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "status": "untouched confirmatory additional release",
    }
    locked = _lock_protocol(output, specification)
    case_results: dict[str, object] = {}
    paired: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_start, train_end = (int(value) for value in split["train"])
        test_start, test_end = (int(value) for value in split["test"])
        if train_start != 0 or test_start != train_end or test_end != int(
            split["frame_len"]
        ):
            raise ValueError(f"unsupported split in {case}")
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "train_end_frame": train_end,
                "maximum_residual_m": maximum_residual_m,
                "interpolation_neighbors": interpolation_neighbors,
            }
            if summary["config"] != expected:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            summary = apply_persistent_residual_anchor(
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_output,
                train_end_frame=train_end,
                maximum_residual_m=maximum_residual_m,
                interpolation_neighbors=interpolation_neighbors,
            )
        future = summary["future"]
        case_results[case] = {
            "physical_object": clusters[case],
            "future_baseline_chamfer_m": future["baseline_chamfer_m"],
            "future_corrected_chamfer_m": future["corrected_chamfer_m"],
            "future_percent_change": future["percent_change"],
            "correction": summary["correction"],
        }
        paired[case] = (
            {
                "chamfer_distance_m": np.asarray(
                    future["baseline_chamfer_by_frame_m"], dtype=float
                )
            },
            {
                "chamfer_distance_m": np.asarray(
                    future["corrected_chamfer_by_frame_m"], dtype=float
                )
            },
        )
    bootstrap = paired_block_bootstrap(
        paired,
        samples=bootstrap_samples,
        block_length=bootstrap_block_length,
        seed=bootstrap_seed,
        clusters=clusters,
    )
    changes = np.array(
        [case_results[case]["future_percent_change"] for case in selected],
        dtype=float,
    )
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "improved_case_count": int(np.sum(changes < 0.0)),
        "median_percent_change": float(np.median(changes)),
        "worst_percent_change": float(np.max(changes)),
        "case_results": case_results,
        "bootstrap": bootstrap,
    }
    result_path = output / "additional_anchor_confirmation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
