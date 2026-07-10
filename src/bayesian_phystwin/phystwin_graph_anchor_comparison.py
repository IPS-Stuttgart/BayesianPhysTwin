"""Matched raw, kNN, and spring-graph endpoint-anchor comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_additional_confirmation import _chamfer_by_frame
from .phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from .phystwin_bayesian_anchor import robust_random_walk_endpoint
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from .phystwin_graph_discrepancy import (
    graph_discrepancy_diagnostics,
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .phystwin_residual_dynamics import (
    _clip_residual,
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
)


RAW_METHOD = "raw_per_point"
KNN_METHOD = "knn_lifted"


def graph_method_id(prior_strength: float) -> str:
    """Return a stable JSON/NPZ identifier for a graph strength."""

    value = format(float(prior_strength), ".12g")
    return "graph_smoothed_lambda_" + value.replace("-", "m").replace(".", "p")


def _metric_summary(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    result = {}
    for metric in baseline:
        baseline_values = np.asarray(baseline[metric], dtype=float)
        candidate_values = np.asarray(candidate[metric], dtype=float)
        baseline_mean = float(np.mean(baseline_values))
        candidate_mean = float(np.mean(candidate_values))
        result[metric] = {
            "baseline_by_frame_m": baseline_values.tolist(),
            "candidate_by_frame_m": candidate_values.tolist(),
            "baseline_mean_m": baseline_mean,
            "candidate_mean_m": candidate_mean,
            "percent_change": 100.0 * (candidate_mean / baseline_mean - 1.0),
        }
    return result


def _correction_summary(
    correction: np.ndarray,
    springs: np.ndarray,
    laplacian,
    *,
    maximum_residual_m: float,
) -> dict[str, float]:
    norm = np.linalg.norm(correction, axis=1)
    return {
        "rms_m": float(np.sqrt(np.mean(np.square(norm)))),
        "maximum_m": float(np.max(norm, initial=0.0)),
        "saturated_fraction": float(
            np.mean(norm >= 0.999 * maximum_residual_m)
        ),
        **graph_discrepancy_diagnostics(correction, springs, laplacian),
    }


def apply_graph_anchor_variants(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    prior_strengths: Iterable[float],
    gt_track_path: str | Path | None = None,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
    covariance_probes: int = 0,
    covariance_seed: int = 20260711,
) -> dict[str, object]:
    """Apply matched endpoint anchors and evaluate the official future interval."""

    strengths = tuple(sorted(set(float(value) for value in prior_strengths)))
    if not strengths or any(value <= 0.0 for value in strengths):
        raise ValueError("prior_strengths must contain positive values")
    if maximum_residual_m <= 0.0:
        raise ValueError("maximum_residual_m must be positive")
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    optimal = _load_pickle(optimal_params_path)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    surface_points = np.asarray(data["surface_points"], dtype=float)
    interior_points = np.asarray(data["interior_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not 2 <= train_end_frame < frame_count:
        raise ValueError("train_end_frame must leave future frames")
    if baseline.shape[0] < frame_count or baseline.shape[1] < original_count:
        raise ValueError("baseline trajectory does not cover observations")
    baseline = baseline[:frame_count]
    structure_points = np.concatenate(
        (observed[0], surface_points, interior_points), axis=0
    )
    if len(structure_points) != baseline.shape[1]:
        raise ValueError("released structure points and trajectory state disagree")
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    springs = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(len(structure_points), springs)
    try:
        from scipy.sparse.csgraph import connected_components
    except (ImportError, OSError) as error:
        raise RuntimeError("graph discrepancy smoothing requires scipy") from error
    component_count, component_labels = connected_components(
        laplacian, directed=False
    )

    residual = observed - baseline[:, :original_count]
    valid = _target_validity(visible, motion_valid)
    endpoint = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=train_end_frame,
        process_variance=FIXED_PROCESS_STD_M**2,
        observation_variance=FIXED_OBSERVATION_STD_M**2,
        initial_variance=FIXED_INITIAL_STD_M**2,
        inlier_prior=FIXED_INLIER_PRIOR,
        outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    )
    updated = endpoint.update_count > 0
    raw = np.zeros((baseline.shape[1], 3), dtype=float)
    raw[:original_count] = endpoint.mean
    raw = _clip_residual(raw[None], maximum_residual_m)[0]
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, interpolation_neighbors
    )
    knn = _lift_residual(
        endpoint.mean[None],
        baseline.shape[1],
        lift_indices,
        lift_weights,
        maximum_norm=maximum_residual_m,
    )[0]
    corrections = {RAW_METHOD: raw, KNN_METHOD: knn}
    graph_posteriors = {}
    for index, strength in enumerate(strengths):
        posterior = graph_smoothed_discrepancy_posterior(
            endpoint.mean,
            endpoint.variance,
            updated,
            laplacian,
            prior_strength=strength,
            covariance_probes=covariance_probes,
            covariance_seed=covariance_seed + index,
        )
        method = graph_method_id(strength)
        corrections[method] = _clip_residual(
            posterior.mean[None], maximum_residual_m
        )[0]
        graph_posteriors[method] = posterior

    num_surface_points = original_count + len(surface_points)
    gt_track = (
        None
        if gt_track_path is None
        else np.asarray(_load_pickle(gt_track_path), dtype=float)
    )

    def metrics_by_frame(trajectory: np.ndarray) -> dict[str, np.ndarray]:
        if gt_track is None:
            return {
                "chamfer_distance_m": _chamfer_by_frame(
                    trajectory,
                    observed,
                    visible,
                    num_surface_points=num_surface_points,
                    start_frame=train_end_frame,
                    end_frame=frame_count,
                )
            }
        return official_metrics_by_frame(
            trajectory,
            observed,
            visible,
            gt_track,
            num_surface_points=num_surface_points,
            start_frame=train_end_frame,
            end_frame=frame_count,
        )

    baseline_metrics = metrics_by_frame(baseline)
    method_results = {}
    archive_values = {
        "endpoint_mean": endpoint.mean,
        "endpoint_variance": endpoint.variance,
        "endpoint_update_count": endpoint.update_count,
        "springs": springs,
    }
    for method, correction in corrections.items():
        candidate = baseline.copy()
        candidate[train_end_frame:] += correction[None]
        method_result: dict[str, object] = {
            "future": _metric_summary(
                baseline_metrics,
                metrics_by_frame(candidate),
            ),
            "correction": _correction_summary(
                correction,
                springs,
                laplacian,
                maximum_residual_m=maximum_residual_m,
            ),
        }
        archive_values[f"correction__{method}"] = correction
        if method in graph_posteriors:
            posterior = graph_posteriors[method]
            covariance = None
            if posterior.marginal_variance is not None:
                marginal_std = np.sqrt(posterior.marginal_variance)
                covariance = {
                    "probe_count": covariance_probes,
                    "median_marginal_std_m": float(np.median(marginal_std)),
                    "p95_marginal_std_m": float(np.quantile(marginal_std, 0.95)),
                    "median_observed_marginal_std_m": float(
                        np.median(marginal_std[:original_count][updated])
                    ),
                    "median_untracked_marginal_std_m": (
                        None
                        if original_count == len(marginal_std)
                        else float(np.median(marginal_std[original_count:]))
                    ),
                    "negative_diagonal_estimate_fraction": (
                        posterior.covariance_negative_fraction
                    ),
                }
                archive_values[f"marginal_variance__{method}"] = (
                    posterior.marginal_variance
                )
            method_result["posterior"] = {
                "reference_variance_m2": posterior.reference_variance,
                "solve_iteration_maximum": max(
                    posterior.solve_iterations, default=0
                ),
                "solve_relative_residual_maximum": max(
                    posterior.solve_relative_residuals, default=0.0
                ),
                "covariance": covariance,
            }
        method_results[method] = method_result

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "graph_anchors.npz"
    np.savez_compressed(archive_path, **archive_values)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "train_end_frame": train_end_frame,
            "prior_strengths": list(strengths),
            "maximum_residual_m": maximum_residual_m,
            "interpolation_neighbors": interpolation_neighbors,
            "covariance_probes": covariance_probes,
            "covariance_seed": covariance_seed,
        },
        "contract": {
            "endpoint": "fixed robust Bayesian training posterior",
            "future_mean": "fixed endpoint correction held constant",
            "future_inputs": "none",
            "raw_untracked_nodes": "zero correction",
            "knn_untracked_nodes": "inverse-distance interpolation",
            "graph_prior": (
                "variance-whitened random-walk-normalized spring Laplacian; "
                "negative log prior lambda * ||L b||^2"
            ),
            "spatial_covariance": (
                "v_ref * (W + 2 lambda L.T L + ridge I)^-1"
            ),
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "final_data": Path(final_data_path),
                "baseline_trajectory": Path(baseline_trajectory_path),
                "optimal_params": Path(optimal_params_path),
                **(
                    {}
                    if gt_track_path is None
                    else {"gt_track_3d": Path(gt_track_path)}
                ),
            }.items()
        },
        "graph": {
            "node_count": len(structure_points),
            "tracked_node_count": original_count,
            "untracked_node_count": len(structure_points) - original_count,
            "spring_count": len(springs),
            "connected_component_count": int(component_count),
            "components_without_observation": int(
                sum(
                    not np.any(component_labels[:original_count] == component)
                    for component in range(component_count)
                )
            ),
            "optimal_parameters": {
                key: (
                    int(optimal[key])
                    if "neighbours" in key
                    else float(optimal[key])
                )
                for key in (
                    "object_radius",
                    "object_max_neighbours",
                    "controller_radius",
                    "controller_max_neighbours",
                )
            },
        },
        "endpoint_posterior": {
            "updated_track_count": int(np.sum(updated)),
            "median_std_m": float(np.median(np.sqrt(endpoint.variance[updated]))),
            "median_final_inlier_probability": float(
                np.median(endpoint.final_inlier_probability[updated])
            ),
        },
        "raw_knn_identical": bool(np.array_equal(raw, knn)),
        "methods": method_results,
        "outputs": {"anchors": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    return summary


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro", "cluster_macro"}
    }


def run_graph_anchor_comparison(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "all",
    cases: Iterable[str] | None = None,
    prior_strengths: Iterable[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    select_prior_strength: bool = False,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
    covariance_probes: int = 0,
    covariance_seed: int = 20260711,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
    force: bool = False,
) -> dict[str, object]:
    """Run the matched graph-anchor comparison on a released cohort."""

    root = Path(data_root)
    additional_manifest = root / "additional_evaluation_subset_manifest.json"
    main_manifest = root / "evaluation_subset_manifest.json"
    if additional_manifest.exists():
        manifest_path = additional_manifest
        is_additional = True
    elif main_manifest.exists():
        manifest_path = main_manifest
        is_additional = False
    else:
        raise FileNotFoundError("data root has no PhysTwin evaluation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in manifest["selected_cases"])
    if cases is not None:
        selected = tuple(dict.fromkeys(str(case) for case in cases))
    elif is_additional or cohort == "all":
        selected = available
    elif cohort == "development":
        selected = tuple(case for case in available if case in DEVELOPMENT_CASES)
    elif cohort == "confirmation":
        selected = tuple(case for case in available if case not in DEVELOPMENT_CASES)
    else:
        raise ValueError("cohort must be all, development, or confirmation")
    if is_additional and cohort != "all" and cases is None:
        raise ValueError("additional data supports only the all cohort")
    missing = sorted(set(selected) - set(available))
    if missing or not selected:
        raise ValueError("invalid selected cases: " + ", ".join(missing))
    strengths = tuple(sorted(set(float(value) for value in prior_strengths)))
    methods = (RAW_METHOD, KNN_METHOD) + tuple(
        graph_method_id(value) for value in strengths
    )
    clusters = {case: phystwin_physical_object_cluster(case) for case in selected}
    status = (
        "development tuning on designated cases"
        if select_prior_strength
        else "post-hoc frozen graph-prior evaluation"
    )
    output = Path(output_dir)
    specification = {
        "method": "matched fixed Bayesian endpoint spatial comparison",
        "cohort": cohort,
        "cases": list(selected),
        "dataset": "additional" if is_additional else "main",
        "prior_strengths": list(strengths),
        "select_prior_strength": select_prior_strength,
        "selection_rule": (
            "lowest equal-case mean graph/knn ratio over available official metrics"
            if select_prior_strength
            else "none"
        ),
        "fixed_filter": {
            "process_std_m": FIXED_PROCESS_STD_M,
            "observation_std_m": FIXED_OBSERVATION_STD_M,
            "initial_std_m": FIXED_INITIAL_STD_M,
            "inlier_prior": FIXED_INLIER_PRIOR,
            "outlier_variance_multiplier": FIXED_OUTLIER_VARIANCE_MULTIPLIER,
        },
        "maximum_residual_m": maximum_residual_m,
        "interpolation_neighbors": interpolation_neighbors,
        "covariance_probes": covariance_probes,
        "future_inputs": "none",
        "status": status,
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
        "data_manifest": str(manifest_path.resolve()),
    }
    locked = _lock_protocol(output, specification)
    case_results: dict[str, object] = {}
    paired_vs_released = {method: {} for method in methods}
    paired_vs_knn = {method: {} for method in methods if method.startswith("graph_")}
    paired_knn_vs_raw = {}
    for case in selected:
        case_dir = root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        train_end, future_end = (int(value) for value in split["test"])
        if future_end != int(split["frame_len"]):
            raise ValueError(f"future split does not end at frame_len for {case}")
        case_output = output / "cases" / case
        summary_path = case_output / "summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "train_end_frame": train_end,
                "prior_strengths": list(strengths),
                "maximum_residual_m": maximum_residual_m,
                "interpolation_neighbors": interpolation_neighbors,
                "covariance_probes": covariance_probes,
                "covariance_seed": covariance_seed,
            }
            if summary["config"] != expected:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            track_path = case_dir / "gt_track_3d.pkl"
            summary = apply_graph_anchor_variants(
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_output,
                train_end_frame=train_end,
                prior_strengths=strengths,
                gt_track_path=track_path if track_path.exists() else None,
                maximum_residual_m=maximum_residual_m,
                interpolation_neighbors=interpolation_neighbors,
                covariance_probes=covariance_probes,
                covariance_seed=covariance_seed,
            )
        case_results[case] = {
            "physical_object": clusters[case],
            "graph": summary["graph"],
            "raw_knn_identical": summary["raw_knn_identical"],
            "methods": {
                method: {
                    "future": summary["methods"][method]["future"],
                    "correction": summary["methods"][method]["correction"],
                    **(
                        {}
                        if "posterior" not in summary["methods"][method]
                        else {"posterior": summary["methods"][method]["posterior"]}
                    ),
                }
                for method in methods
            },
        }
        for method in methods:
            future = summary["methods"][method]["future"]
            baseline_metrics = {
                metric: np.asarray(values["baseline_by_frame_m"], dtype=float)
                for metric, values in future.items()
            }
            candidate_metrics = {
                metric: np.asarray(values["candidate_by_frame_m"], dtype=float)
                for metric, values in future.items()
            }
            paired_vs_released[method][case] = (
                baseline_metrics,
                candidate_metrics,
            )
            if method in paired_vs_knn:
                knn_future = summary["methods"][KNN_METHOD]["future"]
                paired_vs_knn[method][case] = (
                    {
                        metric: np.asarray(
                            knn_future[metric]["candidate_by_frame_m"], dtype=float
                        )
                        for metric in knn_future
                    },
                    candidate_metrics,
                )
        raw_future = summary["methods"][RAW_METHOD]["future"]
        knn_future = summary["methods"][KNN_METHOD]["future"]
        paired_knn_vs_raw[case] = (
            {
                metric: np.asarray(
                    raw_future[metric]["candidate_by_frame_m"], dtype=float
                )
                for metric in raw_future
            },
            {
                metric: np.asarray(
                    knn_future[metric]["candidate_by_frame_m"], dtype=float
                )
                for metric in knn_future
            },
        )

    comparisons = {
        method: _compact_bootstrap(
            paired_block_bootstrap(
                paired_vs_released[method],
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed,
                clusters=clusters,
            )
        )
        for method in methods
    }
    graph_vs_knn = {
        method: _compact_bootstrap(
            paired_block_bootstrap(
                paired,
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed,
                clusters=clusters,
            )
        )
        for method, paired in paired_vs_knn.items()
    }
    knn_vs_raw = _compact_bootstrap(
        paired_block_bootstrap(
            paired_knn_vs_raw,
            samples=bootstrap_samples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed,
            clusters=clusters,
        )
    )
    selection = None
    if select_prior_strength:
        scores = {}
        for strength in strengths:
            method = graph_method_id(strength)
            ratios = []
            for case in selected:
                graph_future = case_results[case]["methods"][method]["future"]
                knn_future = case_results[case]["methods"][KNN_METHOD]["future"]
                for metric in graph_future:
                    ratios.append(
                        float(graph_future[metric]["candidate_mean_m"])
                        / float(knn_future[metric]["candidate_mean_m"])
                    )
            scores[strength] = float(np.mean(ratios))
        selected_strength = min(scores, key=lambda value: (scores[value], value))
        selection = {
            "selected_prior_strength": selected_strength,
            "selected_method": graph_method_id(selected_strength),
            "mean_graph_to_knn_metric_ratio": scores[selected_strength],
            "candidates": [
                {
                    "prior_strength": strength,
                    "method": graph_method_id(strength),
                    "mean_graph_to_knn_metric_ratio": scores[strength],
                }
                for strength in strengths
            ],
        }
    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset": "additional" if is_additional else "main",
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "methods": list(methods),
        "case_results": case_results,
        "comparisons_vs_released": comparisons,
        "knn_vs_raw": knn_vs_raw,
        "graph_vs_knn": graph_vs_knn,
        "selection": selection,
    }
    result_path = output / "graph_anchor_comparison_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
