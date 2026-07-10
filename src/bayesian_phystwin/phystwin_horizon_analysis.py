"""Post-hoc horizon analysis for PhysTwin residual anchoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from .phystwin_comparison import (
    _moving_block_indices,
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_residual_dynamics import (
    _clip_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


HORIZON_LABELS = ("early", "middle", "late")
METRICS = ("chamfer_distance_m", "track_error_m")


def split_future_horizon(frame_count: int) -> dict[str, np.ndarray]:
    """Split a future interval into three contiguous count-balanced thirds."""

    if frame_count < len(HORIZON_LABELS):
        raise ValueError("future interval must contain at least three frames")
    indexes = np.array_split(np.arange(frame_count), len(HORIZON_LABELS))
    return dict(zip(HORIZON_LABELS, indexes, strict=True))


def centered_spatial_correlation(
    endpoint_residual: np.ndarray,
    future_residual: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Correlate two 3D residual fields after removing global translation."""

    endpoint = np.asarray(endpoint_residual, dtype=float)
    future = np.asarray(future_residual, dtype=float)
    support = np.asarray(valid, dtype=bool)
    if endpoint.shape != future.shape or endpoint.ndim != 2 or endpoint.shape[1] != 3:
        raise ValueError("residual fields must have matching shape (N, 3)")
    if support.shape != (len(endpoint),):
        raise ValueError("valid must select residual-field points")
    if np.sum(support) < 2:
        raise ValueError("spatial correlation requires at least two valid points")
    endpoint_values = endpoint[support]
    future_values = future[support]
    if not np.all(np.isfinite(endpoint_values)) or not np.all(
        np.isfinite(future_values)
    ):
        raise ValueError("spatial correlation values must be finite")
    endpoint_centered = endpoint_values - np.mean(endpoint_values, axis=0)
    future_centered = future_values - np.mean(future_values, axis=0)
    denominator = float(
        np.linalg.norm(endpoint_centered) * np.linalg.norm(future_centered)
    )
    if denominator <= 1e-15:
        raise ValueError("spatial correlation is undefined for a constant field")
    return float(np.sum(endpoint_centered * future_centered) / denominator)


def _mean_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def bootstrap_case_frame_mean(
    cases: Mapping[str, np.ndarray],
    *,
    samples: int,
    block_length: int,
    seed: int,
    clusters: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Bootstrap a scalar frame series with equal case and object weighting."""

    if not cases:
        raise ValueError("at least one case is required")
    if samples < 1 or block_length < 1:
        raise ValueError("samples and block_length must be positive")
    rng = np.random.default_rng(seed)
    case_names = tuple(cases)
    observed: dict[str, float] = {}
    case_draws: dict[str, np.ndarray] = {}
    per_case: dict[str, object] = {}
    for case, raw_values in cases.items():
        values = np.asarray(raw_values, dtype=float)
        if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{case} values must be a finite nonempty vector")
        observed[case] = float(np.mean(values))
        draws = np.empty(samples, dtype=float)
        for sample in range(samples):
            indexes = _moving_block_indices(len(values), block_length, rng)
            draws[sample] = np.mean(values[indexes])
        case_draws[case] = draws
        per_case[case] = {
            "observed_mean": observed[case],
            "frame_bootstrap_mean": _mean_interval(draws),
        }

    macro_draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.integers(0, len(case_names), size=len(case_names))
        macro_draws[sample] = np.mean(
            [case_draws[case_names[index]][sample] for index in selected]
        )
    result: dict[str, object] = {
        "samples": samples,
        "block_length": block_length,
        "seed": seed,
        "per_case": per_case,
        "macro": {
            "observed_equal_case_mean": float(np.mean(list(observed.values()))),
            "case_and_frame_bootstrap_mean": _mean_interval(macro_draws),
        },
    }
    if clusters is not None:
        if set(clusters) != set(case_names):
            raise ValueError("clusters must assign every case exactly once")
        grouped: dict[str, list[str]] = {}
        for case in case_names:
            grouped.setdefault(str(clusters[case]), []).append(case)
        cluster_names = tuple(grouped)
        cluster_observed = {
            cluster: float(np.mean([observed[case] for case in grouped[cluster]]))
            for cluster in cluster_names
        }
        cluster_draws = np.empty(samples, dtype=float)
        for sample in range(samples):
            selected = rng.integers(
                0, len(cluster_names), size=len(cluster_names)
            )
            cluster_draws[sample] = np.mean(
                [
                    np.mean(
                        [
                            case_draws[case][sample]
                            for case in grouped[cluster_names[index]]
                        ]
                    )
                    for index in selected
                ]
            )
        result["cluster_macro"] = {
            "cluster_count": len(cluster_names),
            "case_counts": {
                cluster: len(grouped[cluster]) for cluster in cluster_names
            },
            "observed_equal_cluster_mean": float(
                np.mean(list(cluster_observed.values()))
            ),
            "cluster_and_frame_bootstrap_mean": _mean_interval(cluster_draws),
        }
    return result


def _compact_paired_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro", "cluster_macro"}
    }


def run_phystwin_horizon_analysis(
    data_root: str | Path,
    action_run_dir: str | Path,
    persistent_run_dir: str | Path,
    output_path: str | Path,
    *,
    maximum_residual_m: float = 0.01,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Analyze future thirds and endpoint residual persistence on 19 cases."""

    if maximum_residual_m <= 0.0:
        raise ValueError("maximum_residual_m must be positive")
    root = Path(data_root)
    action_root = Path(action_run_dir)
    persistent_root = Path(persistent_run_dir)
    action_summary_path = action_root / "confirmatory_summary.json"
    persistent_summary_path = persistent_root / "baseline_confirmation_summary.json"
    action_summary = json.loads(action_summary_path.read_text(encoding="utf-8"))
    persistent_summary = json.loads(
        persistent_summary_path.read_text(encoding="utf-8")
    )
    cases = tuple(
        str(case)
        for case in action_summary["cohorts"]["confirmation"]["cases"]
    )
    if len(cases) != 19:
        raise ValueError("horizon analysis expects the 19-case confirmation cohort")
    persistent_cases = tuple(
        str(case)
        for case in persistent_summary["methods"]["last_residual"]["confirmation"][
            "cases"
        ]
    )
    if set(persistent_cases) != set(cases):
        raise ValueError("action and persistent runs use different confirmation cases")
    clusters = {case: phystwin_physical_object_cluster(case) for case in cases}
    paired_vs_released = {
        method: {horizon: {} for horizon in HORIZON_LABELS}
        for method in ("action_arx", "persistent")
    }
    paired_direct = {horizon: {} for horizon in HORIZON_LABELS}
    correlations = {horizon: {} for horizon in HORIZON_LABELS}
    case_results: dict[str, object] = {}

    for case in cases:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        final_path = case_dir / "final_data.pkl"
        track_path = case_dir / "gt_track_3d.pkl"
        baseline_path = case_dir / "inference.pkl"
        action_path = action_root / "cases" / case / "trajectory.pkl"
        persistent_path = (
            persistent_root
            / "cases"
            / case
            / "last_residual"
            / "trajectory.pkl"
        )
        data = _load_pickle(final_path)
        tracks = np.asarray(_load_pickle(track_path), dtype=float)
        trajectories = {
            "released": np.asarray(_load_pickle(baseline_path), dtype=float),
            "action_arx": np.asarray(_load_pickle(action_path), dtype=float),
            "persistent": np.asarray(_load_pickle(persistent_path), dtype=float),
        }
        observed = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
        if future_end > len(observed):
            raise ValueError(f"future interval exceeds observations for {case}")
        surface_count = observed.shape[1] + len(data["surface_points"])
        metrics = {
            method: official_metrics_by_frame(
                trajectory,
                observed,
                visible,
                tracks,
                num_surface_points=surface_count,
                start_frame=train_end,
                end_frame=future_end,
            )
            for method, trajectory in trajectories.items()
        }
        residual = (
            observed
            - trajectories["released"][: len(observed), : observed.shape[1]]
        )
        valid = _target_validity(visible, motion_valid)
        endpoint = _temporally_fill(residual, valid, train_end)[-1]
        endpoint = _clip_residual(endpoint[None], maximum_residual_m)[0]
        training_support = np.any(valid[:train_end], axis=0)
        correlation_by_frame = np.empty(future_end - train_end, dtype=float)
        valid_count_by_frame = np.empty(future_end - train_end, dtype=int)
        for offset, frame in enumerate(range(train_end, future_end)):
            support = training_support & valid[frame]
            valid_count_by_frame[offset] = int(np.sum(support))
            correlation_by_frame[offset] = centered_spatial_correlation(
                endpoint,
                residual[frame],
                support,
            )

        action_case_summary = json.loads(
            (action_root / "cases" / case / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        persistent_case_summary = json.loads(
            (persistent_root / "cases" / case / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        case_horizons: dict[str, object] = {}
        for horizon, indexes in split_future_horizon(future_end - train_end).items():
            case_horizons[horizon] = {
                "frame_interval": [
                    train_end + int(indexes[0]),
                    train_end + int(indexes[-1]) + 1,
                ],
                "frame_count": len(indexes),
                "endpoint_residual_spatial_correlation": float(
                    np.mean(correlation_by_frame[indexes])
                ),
                "mean_valid_track_count": float(
                    np.mean(valid_count_by_frame[indexes])
                ),
                "methods": {},
            }
            correlations[horizon][case] = correlation_by_frame[indexes]
            for method in ("action_arx", "persistent"):
                paired_vs_released[method][horizon][case] = (
                    {
                        metric: metrics["released"][metric][indexes]
                        for metric in METRICS
                    },
                    {
                        metric: metrics[method][metric][indexes]
                        for metric in METRICS
                    },
                )
                case_horizons[horizon]["methods"][method] = {
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
            paired_direct[horizon][case] = (
                {
                    metric: metrics["action_arx"][metric][indexes]
                    for metric in METRICS
                },
                {
                    metric: metrics["persistent"][metric][indexes]
                    for metric in METRICS
                },
            )
        case_results[case] = {
            "physical_object": clusters[case],
            "action_accepted_on_validation": bool(
                action_case_summary["selection"]["accepted"]
            ),
            "persistent_accepted_on_validation": bool(
                persistent_case_summary["methods"]["last_residual"]["selection"][
                    "accepted"
                ]
            ),
            "horizons": case_horizons,
            "inputs": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in {
                    "final_data": final_path,
                    "gt_track_3d": track_path,
                    "released_trajectory": baseline_path,
                    "action_trajectory": action_path,
                    "persistent_trajectory": persistent_path,
                }.items()
            },
        }

    methods_vs_released = {
        method: {
            horizon: _compact_paired_bootstrap(
                paired_block_bootstrap(
                    paired_vs_released[method][horizon],
                    samples=bootstrap_samples,
                    block_length=bootstrap_block_length,
                    seed=bootstrap_seed,
                    clusters=clusters,
                )
            )
            for horizon in HORIZON_LABELS
        }
        for method in ("action_arx", "persistent")
    }
    persistent_vs_action = {
        horizon: _compact_paired_bootstrap(
            paired_block_bootstrap(
                paired_direct[horizon],
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed,
                clusters=clusters,
            )
        )
        for horizon in HORIZON_LABELS
    }
    correlation_summary = {
        horizon: bootstrap_case_frame_mean(
            correlations[horizon],
            samples=bootstrap_samples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed,
            clusters=clusters,
        )
        for horizon in HORIZON_LABELS
    }
    result = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-hoc mechanism analysis after main-cohort outcomes",
        "cohort": {
            "cases": list(cases),
            "case_count": len(cases),
            "physical_object_count": len(set(clusters.values())),
        },
        "source_protocols": {
            "action_arx": action_summary["protocol_id"],
            "persistent": persistent_summary["protocol_id"],
        },
        "contract": {
            "future_interval": "official released test interval",
            "horizon_bins": (
                "three contiguous count-balanced thirds within each case"
            ),
            "methods": (
                "saved validation-gated action ARX and last-residual trajectories"
            ),
            "correlation": (
                "normalized dot product between the 10 mm capped, temporally "
                "filled training-endpoint residual and each future residual "
                "field after subtracting each field's per-coordinate point mean"
            ),
            "correlation_support": (
                "tracks with training support and valid future pseudo-measurement"
            ),
            "bootstrap": {
                "samples": bootstrap_samples,
                "moving_block_length_frames": bootstrap_block_length,
                "seed": bootstrap_seed,
                "interaction_weighting": "equal case",
                "object_sensitivity": (
                    "equal physical object with equal interactions within object"
                ),
            },
        },
        "inputs": {
            "data_manifest": {
                "path": str((root / "evaluation_subset_manifest.json").resolve()),
                "sha256": _sha256(root / "evaluation_subset_manifest.json"),
            },
            "action_summary": {
                "path": str(action_summary_path.resolve()),
                "sha256": _sha256(action_summary_path),
            },
            "persistent_summary": {
                "path": str(persistent_summary_path.resolve()),
                "sha256": _sha256(persistent_summary_path),
            },
        },
        "case_results": case_results,
        "methods_vs_released": methods_vs_released,
        "persistent_vs_action": persistent_vs_action,
        "endpoint_residual_correlation": correlation_summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["output_path"] = str(output.resolve())
    return result
