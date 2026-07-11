"""Matched full-horizon and future-third comparison of residual magnitude priors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_comparison import official_metrics_by_frame, paired_block_bootstrap
from .phystwin_confirmatory import DEVELOPMENT_CASES
from .phystwin_horizon_analysis import HORIZON_LABELS, METRICS, split_future_horizon
from .phystwin_residual_dynamics import _load_pickle, _sha256


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro"}
    }


def _paired_summary(
    pairs: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, object]:
    return _compact_bootstrap(
        paired_block_bootstrap(
            pairs,
            samples=samples,
            block_length=block_length,
            seed=seed,
        )
    )


def compare_residual_magnitude_methods(
    data_root: str | Path,
    shrinkage_run_dir: str | Path,
    cap_control_run_dir: str | Path,
    output_path: str | Path,
    *,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
) -> dict[str, Any]:
    """Compare hierarchical shrinkage with matched 10 and 30 mm hard caps."""

    if bootstrap_samples < 1 or bootstrap_block_length < 1:
        raise ValueError("bootstrap settings must be positive")
    root = Path(data_root)
    shrinkage_root = Path(shrinkage_run_dir)
    cap_root = Path(cap_control_run_dir)
    cases = tuple(DEVELOPMENT_CASES)
    methods = ("hard_cap_10mm", "hard_cap_30mm", "hierarchical_shrinkage")
    whole_pairs = {method: {} for method in methods}
    direct_pairs: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    horizon_pairs = {
        method: {horizon: {} for horizon in HORIZON_LABELS}
        for method in methods
    }
    direct_horizon_pairs = {horizon: {} for horizon in HORIZON_LABELS}
    case_results: dict[str, Any] = {}
    for case in cases:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, frame_count = (int(value) for value in split["test"])
        final_path = case_dir / "final_data.pkl"
        track_path = case_dir / "gt_track_3d.pkl"
        trajectory_paths = {
            "released": case_dir / "inference.pkl",
            "hard_cap_10mm": cap_root / case / "10mm" / "trajectory.pkl",
            "hard_cap_30mm": cap_root / case / "30mm" / "trajectory.pkl",
            "hierarchical_shrinkage": (
                shrinkage_root / "cases" / case / "trajectory.pkl"
            ),
        }
        final_data = _load_pickle(final_path)
        observed = np.asarray(final_data["object_points"], dtype=float)
        visible = np.asarray(final_data["object_visibilities"], dtype=bool)
        tracks = np.asarray(_load_pickle(track_path), dtype=float)
        surface_count = observed.shape[1] + len(final_data["surface_points"])
        metrics = {
            method: official_metrics_by_frame(
                np.asarray(_load_pickle(path), dtype=float),
                observed,
                visible,
                tracks,
                num_surface_points=surface_count,
                start_frame=train_end,
                end_frame=frame_count,
            )
            for method, path in trajectory_paths.items()
        }
        horizons = split_future_horizon(frame_count - train_end)
        case_methods = {}
        for method in methods:
            whole_pairs[method][case] = (metrics["released"], metrics[method])
            case_methods[method] = {
                "whole_future": {
                    metric: {
                        "released_mean_m": float(np.mean(metrics["released"][metric])),
                        "candidate_mean_m": float(np.mean(metrics[method][metric])),
                        "percent_change": 100.0
                        * (
                            float(np.mean(metrics[method][metric]))
                            / float(np.mean(metrics["released"][metric]))
                            - 1.0
                        ),
                    }
                    for metric in METRICS
                },
                "horizons": {},
            }
            for horizon, indexes in horizons.items():
                horizon_pairs[method][horizon][case] = (
                    {
                        metric: metrics["released"][metric][indexes]
                        for metric in METRICS
                    },
                    {metric: metrics[method][metric][indexes] for metric in METRICS},
                )
                case_methods[method]["horizons"][horizon] = {
                    metric: {
                        "released_mean_m": float(
                            np.mean(metrics["released"][metric][indexes])
                        ),
                        "candidate_mean_m": float(
                            np.mean(metrics[method][metric][indexes])
                        ),
                        "percent_change": 100.0
                        * (
                            float(np.mean(metrics[method][metric][indexes]))
                            / float(np.mean(metrics["released"][metric][indexes]))
                            - 1.0
                        ),
                    }
                    for metric in METRICS
                }
        direct_pairs[case] = (
            metrics["hard_cap_10mm"], metrics["hierarchical_shrinkage"]
        )
        for horizon, indexes in horizons.items():
            direct_horizon_pairs[horizon][case] = (
                {
                    metric: metrics["hard_cap_10mm"][metric][indexes]
                    for metric in METRICS
                },
                {
                    metric: metrics["hierarchical_shrinkage"][metric][indexes]
                    for metric in METRICS
                },
            )
        case_results[case] = {
            "future_interval": [train_end, frame_count],
            "methods": case_methods,
            "inputs": {
                "final_data": {
                    "path": str(final_path.resolve()),
                    "sha256": _sha256(final_path),
                },
                "gt_track_3d": {
                    "path": str(track_path.resolve()),
                    "sha256": _sha256(track_path),
                },
                "trajectories": {
                    method: {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                    }
                    for method, path in trajectory_paths.items()
                },
            },
        }

    methods_vs_released = {
        method: {
            "whole_future": _paired_summary(
                whole_pairs[method],
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed + method_index,
            ),
            "horizons": {
                horizon: _paired_summary(
                    horizon_pairs[method][horizon],
                    samples=bootstrap_samples,
                    block_length=bootstrap_block_length,
                    seed=bootstrap_seed + method_index * 10 + horizon_index,
                )
                for horizon_index, horizon in enumerate(HORIZON_LABELS)
            },
        }
        for method_index, method in enumerate(methods)
    }
    shrinkage_vs_10mm = {
        "whole_future": _paired_summary(
            direct_pairs,
            samples=bootstrap_samples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed + 100,
        ),
        "horizons": {
            horizon: _paired_summary(
                direct_horizon_pairs[horizon],
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed + 110 + horizon_index,
            )
            for horizon_index, horizon in enumerate(HORIZON_LABELS)
        },
    }
    result = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory development-cohort comparison",
        "cases": list(cases),
        "contract": {
            "future_interval": "released test frames, never used for selection",
            "horizon_bins": "three contiguous count-balanced thirds per interaction",
            "bootstrap": {
                "samples": bootstrap_samples,
                "moving_block_length_frames": bootstrap_block_length,
                "interaction_weighting": "equal case",
            },
            "shrinkage_selection": "outer leave-one-interaction-out for shared settings; local scale from validation only",
            "controls": "per-case validation-selected action residual with matched 10 mm or 30 mm pointwise hard cap",
        },
        "case_results": case_results,
        "methods_vs_released": methods_vs_released,
        "hierarchical_shrinkage_vs_hard_cap_10mm": shrinkage_vs_10mm,
        "source_summaries": {
            "hierarchical_shrinkage": {
                "path": str(
                    (shrinkage_root / "hierarchical_shrinkage_summary.json").resolve()
                ),
                "sha256": _sha256(
                    shrinkage_root / "hierarchical_shrinkage_summary.json"
                ),
            },
            "hard_cap_controls": {
                case: {
                    label: {
                        "path": str((cap_root / case / label / "summary.json").resolve()),
                        "sha256": _sha256(cap_root / case / label / "summary.json"),
                    }
                    for label in ("10mm", "30mm")
                }
                for case in cases
            },
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["output_path"] = str(output.resolve())
    return result
