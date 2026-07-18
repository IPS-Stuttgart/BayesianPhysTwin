"""Future-blind zero-order search over PhysTwin topology and spring fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_headless_refit import (
    HeadlessPhysTwinRefitConfig,
    run_headless_phystwin_refit,
)
from .phystwin_piecewise_topology import build_piecewise_topology_from_files


ZERO_ORDER_TOPOLOGY_CONTRACT = "phystwin-zero-order-topology-field-v1"
METRICS = ("chamfer_distance_m", "track_error_m")


@dataclass(frozen=True)
class ZeroOrderTopologySearchConfig:
    """Locked dimensions, bounds, and fit-only selection rule."""

    region_count: int = 5
    candidates_per_family: int = 8
    seed: int = 20260718
    radius_bounds: tuple[float, float] = (0.75, 1.25)
    neighbour_bounds: tuple[float, float] = (0.65, 1.35)
    object_log_scale_bounds: tuple[float, float] = (-0.35, 0.35)
    region_log_scale_bounds: tuple[float, float] = (-0.50, 0.50)
    controller_log_scale_bounds: tuple[float, float] = (-0.35, 0.35)
    minimum_fit_improvement: float = 0.01
    maximum_fit_metric_ratio: float = 1.02


@dataclass(frozen=True)
class TopologyFieldCandidate:
    """One simulator proposal in zero-order search coordinates."""

    candidate_id: str
    family: str
    radius_multipliers: tuple[float, ...]
    neighbour_multipliers: tuple[float, ...]
    object_log_scale: float
    controller_log_scale: float
    region_object_log_scales: tuple[float, ...]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scale(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError("search bounds must be finite and ordered")
    return lower + (upper - lower) * values


def _latin_hypercube(samples: int, dimensions: int, seed: int) -> np.ndarray:
    if samples < 1 or dimensions < 1:
        raise ValueError("Latin-hypercube dimensions must be positive")
    rng = np.random.default_rng(seed)
    jitter = rng.random((samples, dimensions))
    result = np.empty_like(jitter)
    for dimension in range(dimensions):
        result[:, dimension] = (
            rng.permutation(samples) + jitter[:, dimension]
        ) / samples
    return result


def generate_topology_field_candidates(
    config: ZeroOrderTopologySearchConfig,
) -> tuple[TopologyFieldCandidate, ...]:
    """Generate a deterministic identity plus topology/field/joint LHS bank."""

    if config.region_count < 2 or config.candidates_per_family < 1:
        raise ValueError("search needs multiple regions and positive family budgets")
    region_count = config.region_count
    ones = (1.0,) * region_count
    zeros = (0.0,) * region_count
    candidates = [
        TopologyFieldCandidate(
            candidate_id="exact_teacher",
            family="identity",
            radius_multipliers=ones,
            neighbour_multipliers=ones,
            object_log_scale=0.0,
            controller_log_scale=0.0,
            region_object_log_scales=zeros,
        )
    ]

    topology_lhs = _latin_hypercube(
        config.candidates_per_family,
        2 * region_count,
        config.seed,
    )
    field_lhs = _latin_hypercube(
        config.candidates_per_family,
        region_count + 2,
        config.seed + 1,
    )
    joint_lhs = _latin_hypercube(
        config.candidates_per_family,
        3 * region_count + 2,
        config.seed + 2,
    )
    for index in range(config.candidates_per_family):
        radius = _scale(topology_lhs[index, :region_count], config.radius_bounds)
        neighbour = _scale(
            topology_lhs[index, region_count:], config.neighbour_bounds
        )
        candidates.append(
            TopologyFieldCandidate(
                candidate_id=f"topology_{index:03d}",
                family="topology_only",
                radius_multipliers=tuple(float(value) for value in radius),
                neighbour_multipliers=tuple(float(value) for value in neighbour),
                object_log_scale=0.0,
                controller_log_scale=0.0,
                region_object_log_scales=zeros,
            )
        )

        field_region = _scale(
            field_lhs[index, :region_count], config.region_log_scale_bounds
        )
        field_object = float(
            _scale(
                field_lhs[index, region_count : region_count + 1],
                config.object_log_scale_bounds,
            )[0]
        )
        field_controller = float(
            _scale(
                field_lhs[index, region_count + 1 :],
                config.controller_log_scale_bounds,
            )[0]
        )
        candidates.append(
            TopologyFieldCandidate(
                candidate_id=f"field_{index:03d}",
                family="field_only",
                radius_multipliers=ones,
                neighbour_multipliers=ones,
                object_log_scale=field_object,
                controller_log_scale=field_controller,
                region_object_log_scales=tuple(
                    float(value) for value in field_region
                ),
            )
        )

        joint_radius = _scale(
            joint_lhs[index, :region_count], config.radius_bounds
        )
        joint_neighbour = _scale(
            joint_lhs[index, region_count : 2 * region_count],
            config.neighbour_bounds,
        )
        joint_region = _scale(
            joint_lhs[index, 2 * region_count : 3 * region_count],
            config.region_log_scale_bounds,
        )
        joint_object = float(
            _scale(
                joint_lhs[index, 3 * region_count : 3 * region_count + 1],
                config.object_log_scale_bounds,
            )[0]
        )
        joint_controller = float(
            _scale(
                joint_lhs[index, 3 * region_count + 1 :],
                config.controller_log_scale_bounds,
            )[0]
        )
        candidates.append(
            TopologyFieldCandidate(
                candidate_id=f"joint_{index:03d}",
                family="joint",
                radius_multipliers=tuple(float(value) for value in joint_radius),
                neighbour_multipliers=tuple(
                    float(value) for value in joint_neighbour
                ),
                object_log_scale=joint_object,
                controller_log_scale=joint_controller,
                region_object_log_scales=tuple(
                    float(value) for value in joint_region
                ),
            )
        )
    return tuple(candidates)


def select_topology_field_candidate(
    metrics_by_candidate: dict[str, dict[str, float]],
    candidates: tuple[TopologyFieldCandidate, ...],
    config: ZeroOrderTopologySearchConfig,
) -> dict[str, object]:
    """Select on fit metrics only, with a deterministic exact-teacher fallback."""

    if not candidates or candidates[0].candidate_id != "exact_teacher":
        raise ValueError("candidate bank must begin with exact_teacher")
    reference = metrics_by_candidate.get("exact_teacher")
    if reference is None:
        raise ValueError("candidate metrics omit exact_teacher")
    reference_values = np.asarray([float(reference[name]) for name in METRICS])
    if np.any(~np.isfinite(reference_values)) or np.any(reference_values <= 0.0):
        raise ValueError("exact-teacher metrics must be finite and positive")

    records: list[dict[str, object]] = []
    for candidate in candidates:
        raw = metrics_by_candidate.get(candidate.candidate_id)
        if raw is None:
            continue
        values = np.asarray([float(raw[name]) for name in METRICS])
        valid = bool(np.all(np.isfinite(values)) and np.all(values > 0.0))
        ratios = values / reference_values if valid else np.full(2, np.inf)
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "metric_ratios": {
                    name: float(value) for name, value in zip(METRICS, ratios)
                },
                "balanced_score": float(np.mean(ratios)),
                "valid": valid,
            }
        )
    if not records:
        raise ValueError("candidate search produced no metrics")
    best = min(records, key=lambda record: float(record["balanced_score"]))
    eligible = [
        record
        for record in records
        if record["candidate_id"] != "exact_teacher"
        and 1.0 - float(record["balanced_score"])
        >= config.minimum_fit_improvement
        and max(float(value) for value in record["metric_ratios"].values())
        <= config.maximum_fit_metric_ratio
    ]
    selected = (
        min(eligible, key=lambda record: float(record["balanced_score"]))
        if eligible
        else next(
            record for record in records if record["candidate_id"] == "exact_teacher"
        )
    )
    accepted = bool(eligible)
    selected_id = str(selected["candidate_id"])
    return {
        "selected_candidate_id": selected_id,
        "best_raw_candidate_id": str(best["candidate_id"]),
        "best_raw_family": str(best["family"]),
        "best_raw_balanced_fit_improvement": float(
            1.0 - float(best["balanced_score"])
        ),
        "best_raw_maximum_metric_ratio": max(
            float(value) for value in best["metric_ratios"].values()
        ),
        "selected_balanced_fit_improvement": float(
            1.0 - float(selected["balanced_score"])
        ),
        "candidate_accepted": accepted,
        "fallback": "exact_teacher" if not accepted else None,
        "records": records,
    }


def run_zero_order_topology_search(
    *,
    official_repo: str | Path,
    fit_final_data_path: str | Path,
    optimal_params_path: str | Path,
    checkpoint_path: str | Path,
    partition_path: str | Path,
    cues_path: str | Path,
    output_dir: str | Path,
    fit_end_frame: int,
    selection_start_frame: int | None = None,
    released_trajectory_path: str | Path,
    gt_track_path: str | Path,
    config: ZeroOrderTopologySearchConfig = ZeroOrderTopologySearchConfig(),
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build and score a deterministic bank using fit-prefix observations only."""

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"zero-order output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_start = (
        int(np.floor(0.75 * fit_end_frame))
        if selection_start_frame is None
        else int(selection_start_frame)
    )
    if not 1 < selection_start < fit_end_frame:
        raise ValueError("selection start must lie inside the fit prefix")
    candidates = generate_topology_field_candidates(config)
    plan = {
        "schema_version": 1,
        "contract": ZERO_ORDER_TOPOLOGY_CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "candidate_count": len(candidates),
        "selection_interval": [selection_start, fit_end_frame],
        "candidates": [asdict(candidate) for candidate in candidates],
        "inputs": {
            "fit_final_data": {
                "path": str(Path(fit_final_data_path).resolve()),
                "sha256": _sha256(fit_final_data_path),
            },
            "optimal_params": {
                "path": str(Path(optimal_params_path).resolve()),
                "sha256": _sha256(optimal_params_path),
            },
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "partition": {
                "path": str(Path(partition_path).resolve()),
                "sha256": _sha256(partition_path),
            },
            "gt_track_3d": {
                "path": str(Path(gt_track_path).resolve()),
                "sha256": _sha256(gt_track_path),
            },
        },
        "information_boundary": (
            "Every candidate sees only the fit-prefix artifact. Candidate "
            "selection uses official CD and track error on the locked late-fit "
            "interval; no source suffix or future observation is present."
        ),
        "future_observations_used": False,
    }
    plan_path = output / "search_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    results: dict[str, dict[str, object]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        candidate_root = output / "candidates" / candidate.candidate_id
        candidate_root.mkdir(parents=True)
        topology_path = candidate_root / "topology.npz"
        topology_summary = build_piecewise_topology_from_files(
            fit_final_data_path,
            optimal_params_path,
            checkpoint_path,
            partition_path,
            topology_path,
            radius_multipliers=candidate.radius_multipliers,
            neighbour_multipliers=candidate.neighbour_multipliers,
            object_log_scale=candidate.object_log_scale,
            controller_log_scale=candidate.controller_log_scale,
            preserve_total_object_stiffness=True,
            region_object_log_scales=candidate.region_object_log_scales,
        )
        topology_summary_path = candidate_root / "topology.json"
        topology_summary_path.write_text(
            json.dumps(topology_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostics = topology_summary["diagnostics"]
        if (
            diagnostics["object_component_count"] != 1
            or diagnostics["isolated_object_point_count"] != 0
        ):
            results[candidate.candidate_id] = {
                "status": "invalid_topology",
                "candidate": asdict(candidate),
                "topology": topology_summary,
            }
            continue
        run_summary = run_headless_phystwin_refit(
            official_repo=official_repo,
            final_data_path=fit_final_data_path,
            optimal_params_path=optimal_params_path,
            checkpoint_path=checkpoint_path,
            cues_path=cues_path,
            output_dir=candidate_root / "run",
            config=HeadlessPhysTwinRefitConfig(
                variant="hard",
                train_end_frame=fit_end_frame,
                fit_end_frame=selection_start,
                epochs=0,
                optimize_collision=False,
                spring_parameterization="dense",
                selection_metric="official_3d",
                deterministic_spring_forces=True,
                device=device,
            ),
            released_trajectory_path=released_trajectory_path,
            gt_track_path=gt_track_path,
            spring_topology_path=topology_path,
        )
        selection_metrics = {
            name: float(run_summary["official_evaluation"]["validation"][name])
            for name in METRICS
        }
        metrics[candidate.candidate_id] = selection_metrics
        results[candidate.candidate_id] = {
            "status": "evaluated",
            "candidate": asdict(candidate),
            "selection_metrics": selection_metrics,
            "selection_interval": [selection_start, fit_end_frame],
            "topology": topology_summary,
            "run_summary": {
                "path": str(candidate_root / "run" / "summary.json"),
                "sha256": _sha256(candidate_root / "run" / "summary.json"),
            },
        }

    selection = select_topology_field_candidate(metrics, candidates, config)
    selected_id = str(selection["selected_candidate_id"])
    selected = results[selected_id]
    summary = {
        "schema_version": 1,
        "contract": ZERO_ORDER_TOPOLOGY_CONTRACT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "selection": selection,
        "selection_interval": [selection_start, fit_end_frame],
        "selected_candidate": selected["candidate"],
        "selected_topology": selected["topology"]["artifact"],
        "candidate_results": results,
        "future_observations_used": False,
    }
    summary_path = output / "search_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary
