"""Post-hoc physical diagnosis of PhysTwin endpoint anchor fields."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_additional_confirmation import (
    SPATIAL_MODES,
    apply_endpoint_transform,
    fit_endpoint_transform,
)
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_graph import PhysTwinSpringGraphConfig, build_phystwin_spring_graph
from .phystwin_residual_dynamics import (
    _clip_residual,
    _lift_map,
    _lift_residual,
    _load_pickle,
    _sha256,
    _target_validity,
    _temporally_fill,
)


TRANSFORM_MODES = tuple(mode for mode in SPATIAL_MODES if mode != "per_point")


def _metric_summary(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric, baseline_values_raw in baseline.items():
        baseline_values = np.asarray(baseline_values_raw, dtype=float)
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


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=float)
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def transform_geometry_diagnostics(
    transform: dict[str, object],
) -> dict[str, object]:
    """Return interpretable rotation, scale, and strain diagnostics."""

    linear = np.asarray(transform["linear"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    singular_values = np.linalg.svd(linear, compute_uv=False)
    result: dict[str, object] = {
        "translation_m": translation.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "linear_determinant": float(np.linalg.det(linear)),
        "linear_singular_values": singular_values.tolist(),
        "linear_anisotropy_ratio": float(
            np.max(singular_values) / max(np.min(singular_values), 1e-15)
        ),
    }
    rotation = transform.get("rotation")
    if rotation is not None:
        result["rotation_angle_deg"] = _rotation_angle_degrees(
            np.asarray(rotation, dtype=float)
        )
    scale = transform.get("scale")
    if scale is not None:
        result["uniform_scale"] = float(scale)
        result["uniform_scale_percent_change"] = 100.0 * (float(scale) - 1.0)
    if transform["mode"] == "affine":
        left, _, right_transpose = np.linalg.svd(linear)
        polar_rotation = left @ right_transpose
        if np.linalg.det(polar_rotation) < 0.0:
            left[:, -1] *= -1.0
            polar_rotation = left @ right_transpose
        determinant = float(np.linalg.det(linear))
        result["polar_rotation_angle_deg"] = _rotation_angle_degrees(
            polar_rotation
        )
        result["isotropic_scale"] = (
            None if determinant <= 0.0 else float(np.cbrt(determinant))
        )
        result["linear_departure_from_identity_frobenius"] = float(
            np.linalg.norm(linear - np.eye(3), ord="fro")
        )
    return result


def _region_localization(
    norm: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    support = np.asarray(mask, dtype=bool)
    energy = np.square(np.asarray(norm, dtype=float))
    point_fraction = float(np.mean(support))
    energy_total = float(np.sum(energy))
    energy_fraction = (
        0.0 if energy_total <= 1e-30 else float(np.sum(energy[support]) / energy_total)
    )
    mean_near = None if not np.any(support) else float(np.mean(norm[support]))
    mean_far = None if np.all(support) else float(np.mean(norm[~support]))
    return {
        "point_count": int(np.sum(support)),
        "point_fraction": point_fraction,
        "residual_energy_fraction": energy_fraction,
        "energy_concentration_ratio": (
            None if point_fraction <= 0.0 else energy_fraction / point_fraction
        ),
        "mean_norm_near_m": mean_near,
        "mean_norm_far_m": mean_far,
        "near_to_far_mean_norm_ratio": (
            None
            if mean_near is None or mean_far is None or mean_far <= 1e-15
            else mean_near / mean_far
        ),
    }


def field_localization_diagnostics(
    points: np.ndarray,
    controller_points: np.ndarray,
    field: np.ndarray,
    *,
    controller_radius_m: float,
    ground_band_m: float,
) -> dict[str, object]:
    """Measure whether a field is concentrated near grasp or ground regions."""

    positions = np.asarray(points, dtype=float)
    controls = np.asarray(controller_points, dtype=float)
    values = np.asarray(field, dtype=float)
    if positions.shape != values.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("points and field must have matching shape (N, 3)")
    if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 1:
        raise ValueError("controller_points must have nonempty shape (C, 3)")
    if controller_radius_m <= 0.0 or ground_band_m <= 0.0:
        raise ValueError("localization radii must be positive")
    controller_distance = np.min(
        np.linalg.norm(positions[:, None] - controls[None], axis=2), axis=1
    )
    near_controller = controller_distance <= controller_radius_m
    ground_distance = np.abs(positions[:, 2])
    near_ground = ground_distance <= ground_band_m
    norm = np.linalg.norm(values, axis=1)
    return {
        "field_rms_m": float(np.sqrt(np.mean(np.square(norm)))),
        "field_median_norm_m": float(np.median(norm)),
        "controller_radius_m": controller_radius_m,
        "ground_plane": "z = 0 m",
        "ground_band_m": ground_band_m,
        "median_controller_distance_m": float(np.median(controller_distance)),
        "median_ground_distance_m": float(np.median(ground_distance)),
        "near_controller": _region_localization(norm, near_controller),
        "near_ground": _region_localization(norm, near_ground),
        "near_controller_or_ground": _region_localization(
            norm, near_controller | near_ground
        ),
    }


def field_graph_diagnostics(
    field: np.ndarray,
    springs: np.ndarray,
) -> dict[str, float | None]:
    """Measure graph roughness relative to field magnitude."""

    values = np.asarray(field, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("field must have shape (N, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) < 1:
        raise ValueError("springs must have nonempty shape (S, 2)")
    if np.any(edges < 0) or np.any(edges >= len(values)):
        raise ValueError("spring endpoint exceeds field")
    norm = np.linalg.norm(values, axis=1)
    field_rms = float(np.sqrt(np.mean(np.square(norm))))
    centered = values - np.mean(values, axis=0, keepdims=True)
    centered_norm = np.linalg.norm(centered, axis=1)
    centered_field_rms = float(
        np.sqrt(np.mean(np.square(centered_norm)))
    )
    edge_delta = values[edges[:, 0]] - values[edges[:, 1]]
    degree = np.zeros(len(values), dtype=float)
    neighbor_sum = np.zeros_like(values)
    np.add.at(degree, edges[:, 0], 1.0)
    np.add.at(degree, edges[:, 1], 1.0)
    np.add.at(neighbor_sum, edges[:, 0], values[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], values[edges[:, 0]])
    active = degree > 0.0
    laplacian_values = np.zeros_like(values)
    laplacian_values[active] = (
        values[active] - neighbor_sum[active] / degree[active, None]
    )
    edge_rms = float(
        np.sqrt(np.mean(np.sum(np.square(edge_delta), axis=1)))
    )
    laplacian_energy = float(
        np.mean(np.sum(np.square(laplacian_values), axis=1))
    )
    laplacian_rms = float(np.sqrt(laplacian_energy))
    centered_left = centered[edges[:, 0]].reshape(-1)
    centered_right = centered[edges[:, 1]].reshape(-1)
    correlation_denominator = float(
        np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
    )
    return {
        "field_rms_m": field_rms,
        "centered_field_rms_m": centered_field_rms,
        "edge_difference_rms_m": edge_rms,
        "laplacian_energy_m2_per_node": laplacian_energy,
        "edge_difference_to_field_rms_ratio": (
            0.0 if field_rms <= 1e-15 else edge_rms / field_rms
        ),
        "laplacian_rms_to_field_rms_ratio": (
            0.0 if field_rms <= 1e-15 else laplacian_rms / field_rms
        ),
        "centered_edge_difference_to_field_rms_ratio": (
            None
            if centered_field_rms <= 1e-15
            else edge_rms / centered_field_rms
        ),
        "centered_edge_vector_correlation": (
            None
            if correlation_denominator <= 1e-30
            else float(np.dot(centered_left, centered_right))
            / correlation_denominator
        ),
    }


def _endpoint_fit_diagnostics(
    endpoint_residual: np.ndarray,
    correction: np.ndarray,
    *,
    maximum_residual_m: float,
) -> dict[str, float]:
    target = np.asarray(endpoint_residual, dtype=float)
    fitted = np.asarray(correction, dtype=float)
    remaining = target - fitted
    total_sse = float(np.sum(np.square(target)))
    remaining_sse = float(np.sum(np.square(remaining)))
    norm = np.linalg.norm(fitted, axis=1)
    return {
        "endpoint_fit_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(remaining), axis=1)))
        ),
        "endpoint_discrepancy_sse_m2": total_sse,
        "endpoint_remaining_sse_m2": remaining_sse,
        "endpoint_sse_explained_fraction": (
            0.0 if total_sse <= 1e-30 else 1.0 - remaining_sse / total_sse
        ),
        "endpoint_correction_rms_m": float(
            np.sqrt(np.mean(np.square(norm)))
        ),
        "endpoint_correction_maximum_m": float(np.max(norm, initial=0.0)),
        "endpoint_correction_saturated_fraction": float(
            np.mean(norm >= 0.999 * maximum_residual_m)
        ),
    }


def analyze_spatial_modes_case(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    optimal_params_path: str | Path,
    gt_track_path: str | Path,
    output_dir: str | Path,
    *,
    train_end_frame: int,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
    ground_band_m: float = 0.01,
) -> dict[str, object]:
    """Fit all endpoint spatial modes and score the official future metrics."""

    if not 2 <= train_end_frame:
        raise ValueError("train_end_frame must include at least two frames")
    if maximum_residual_m <= 0.0 or interpolation_neighbors < 1:
        raise ValueError("correction settings must be positive")
    data = _load_pickle(final_data_path)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    optimal = _load_pickle(optimal_params_path)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    surface_points = np.asarray(data["surface_points"], dtype=float)
    interior_points = np.asarray(data["interior_points"], dtype=float)
    controller_points = np.asarray(data["controller_points"], dtype=float)
    frame_count, original_count, _ = observed.shape
    if not train_end_frame < frame_count:
        raise ValueError("train_end_frame must precede future frames")
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
    object_springs = graph.springs[: graph.num_object_springs]
    tracked_springs = object_springs[
        np.all(object_springs < original_count, axis=1)
    ]
    if len(tracked_springs) < 1:
        raise ValueError("tracked endpoint field has no reconstructed graph edges")
    residual = observed - baseline[:, :original_count]
    valid = _target_validity(visible, motion_valid)
    endpoint_residual = _temporally_fill(residual, valid, train_end_frame)[-1]
    endpoint_source = baseline[train_end_frame - 1, :original_count]
    endpoint_target = endpoint_source + endpoint_residual
    endpoint_controller = controller_points[train_end_frame - 1]
    num_surface_points = original_count + len(surface_points)

    def metrics_by_frame(trajectory: np.ndarray) -> dict[str, np.ndarray]:
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
    lift_indices, lift_weights = _lift_map(
        baseline[0], original_count, interpolation_neighbors
    )
    method_results: dict[str, object] = {}
    archive: dict[str, np.ndarray] = {
        "endpoint_residual": endpoint_residual,
        "tracked_springs": tracked_springs,
    }
    for mode in SPATIAL_MODES:
        transform = None
        if mode == "per_point":
            tracked_correction = _clip_residual(
                endpoint_residual[None], maximum_residual_m
            )[0]
            repeated = np.repeat(
                endpoint_residual[None], frame_count - train_end_frame, axis=0
            )
            future_correction = _lift_residual(
                repeated,
                baseline.shape[1],
                lift_indices,
                lift_weights,
                maximum_norm=maximum_residual_m,
            )
            geometry = None
            uncapped_fit = endpoint_residual
        else:
            transform = fit_endpoint_transform(
                endpoint_source, endpoint_target, mode=mode
            )
            uncapped_fit = (
                apply_endpoint_transform(endpoint_source, transform)
                - endpoint_source
            )
            tracked_correction = _clip_residual(
                uncapped_fit[None], maximum_residual_m
            )[0]
            transformed_future = apply_endpoint_transform(
                baseline[train_end_frame:], transform
            )
            future_correction = _clip_residual(
                transformed_future - baseline[train_end_frame:],
                maximum_residual_m,
            )
            geometry = transform_geometry_diagnostics(transform)
        candidate = baseline.copy()
        candidate[train_end_frame:] += future_correction
        remaining = endpoint_residual - tracked_correction
        correction_norm = np.linalg.norm(future_correction, axis=2)
        endpoint_fit = _endpoint_fit_diagnostics(
            endpoint_residual,
            tracked_correction,
            maximum_residual_m=maximum_residual_m,
        )
        uncapped_remaining = endpoint_residual - uncapped_fit
        total_sse = float(np.sum(np.square(endpoint_residual)))
        endpoint_fit["uncapped_endpoint_fit_rmse_m"] = float(
            np.sqrt(np.mean(np.sum(np.square(uncapped_remaining), axis=1)))
        )
        endpoint_fit["uncapped_endpoint_sse_explained_fraction"] = (
            0.0
            if total_sse <= 1e-30
            else 1.0 - float(np.sum(np.square(uncapped_remaining))) / total_sse
        )
        method_results[mode] = {
            "future": _metric_summary(baseline_metrics, metrics_by_frame(candidate)),
            "future_correction": {
                "rms_m": float(np.sqrt(np.mean(np.square(correction_norm)))),
                "maximum_m": float(np.max(correction_norm, initial=0.0)),
                "saturated_fraction": float(
                    np.mean(correction_norm >= 0.999 * maximum_residual_m)
                ),
            },
            "endpoint_fit": endpoint_fit,
            "remaining_graph": field_graph_diagnostics(
                remaining, tracked_springs
            ),
            "remaining_localization": field_localization_diagnostics(
                endpoint_source,
                endpoint_controller,
                remaining,
                controller_radius_m=float(optimal["controller_radius"]),
                ground_band_m=ground_band_m,
            ),
            "transform": (
                None
                if transform is None
                else {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in transform.items()
                }
            ),
            "geometry": geometry,
        }
        archive[f"endpoint_correction__{mode}"] = tracked_correction
        archive[f"endpoint_remaining__{mode}"] = remaining

    endpoint_norm = np.linalg.norm(endpoint_residual, axis=1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "spatial_fields.npz"
    np.savez_compressed(archive_path, **archive)
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "train_end_frame": train_end_frame,
            "maximum_residual_m": maximum_residual_m,
            "interpolation_neighbors": interpolation_neighbors,
            "ground_band_m": ground_band_m,
        },
        "contract": {
            "endpoint_anchor": "final temporally filled training residual",
            "spatial_modes": list(SPATIAL_MODES),
            "future_inputs": "none",
            "transform_fit": "released tracked endpoint correspondences only",
            "future_application": (
                "fixed transform or field, capped at maximum_residual_m"
            ),
            "ground_region": "absolute baseline endpoint z <= ground_band_m",
            "controller_region": "nearest endpoint controller within optimized controller radius",
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "final_data": Path(final_data_path),
                "baseline_trajectory": Path(baseline_trajectory_path),
                "optimal_params": Path(optimal_params_path),
                "gt_track_3d": Path(gt_track_path),
            }.items()
        },
        "graph": {
            "tracked_node_count": original_count,
            "tracked_spring_count": len(tracked_springs),
        },
        "endpoint_anchor": {
            "rms_m": float(np.sqrt(np.mean(np.square(endpoint_norm)))),
            "median_norm_m": float(np.median(endpoint_norm)),
            "p95_norm_m": float(np.quantile(endpoint_norm, 0.95)),
            "above_cap_fraction": float(
                np.mean(endpoint_norm >= maximum_residual_m)
            ),
            "graph": field_graph_diagnostics(
                endpoint_residual, tracked_springs
            ),
            "localization": field_localization_diagnostics(
                endpoint_source,
                endpoint_controller,
                endpoint_residual,
                controller_radius_m=float(optimal["controller_radius"]),
                ground_band_m=ground_band_m,
            ),
        },
        "methods": method_results,
        "outputs": {"fields": str(archive_path.resolve())},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["outputs"]["summary"] = str(summary_path.resolve())
    return summary


def _compact_bootstrap(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key in {"samples", "block_length", "seed", "macro", "cluster_macro"}
    }


def _distribution(
    values: Iterable[float | None],
) -> dict[str, float | int | None]:
    raw = tuple(values)
    finite = tuple(
        float(value)
        for value in raw
        if value is not None and np.isfinite(float(value))
    )
    if not finite:
        return {
            "count": 0,
            "missing_count": len(raw),
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(finite, dtype=float)
    return {
        "count": len(array),
        "missing_count": len(raw) - len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _physical_aggregate(
    case_results: dict[str, object],
    selected: tuple[str, ...],
) -> dict[str, object]:
    anchor = {
        "rms_m": _distribution(
            case_results[case]["endpoint_anchor"]["rms_m"] for case in selected
        ),
        "above_cap_fraction": _distribution(
            case_results[case]["endpoint_anchor"]["above_cap_fraction"]
            for case in selected
        ),
        "edge_difference_to_field_rms_ratio": _distribution(
            case_results[case]["endpoint_anchor"]["graph"][
                "edge_difference_to_field_rms_ratio"
            ]
            for case in selected
        ),
        "centered_edge_difference_to_field_rms_ratio": _distribution(
            case_results[case]["endpoint_anchor"]["graph"][
                "centered_edge_difference_to_field_rms_ratio"
            ]
            for case in selected
        ),
        "centered_edge_vector_correlation": _distribution(
            case_results[case]["endpoint_anchor"]["graph"][
                "centered_edge_vector_correlation"
            ]
            for case in selected
        ),
        "controller_energy_concentration_ratio": _distribution(
            case_results[case]["endpoint_anchor"]["localization"][
                "near_controller"
            ]["energy_concentration_ratio"]
            for case in selected
        ),
        "ground_energy_concentration_ratio": _distribution(
            case_results[case]["endpoint_anchor"]["localization"]["near_ground"][
                "energy_concentration_ratio"
            ]
            for case in selected
        ),
        "controller_or_ground_energy_concentration_ratio": _distribution(
            case_results[case]["endpoint_anchor"]["localization"][
                "near_controller_or_ground"
            ]["energy_concentration_ratio"]
            for case in selected
        ),
    }
    methods: dict[str, object] = {}
    for mode in SPATIAL_MODES:
        method: dict[str, object] = {
            "endpoint_sse_explained_fraction": _distribution(
                case_results[case]["methods"][mode]["endpoint_fit"][
                    "endpoint_sse_explained_fraction"
                ]
                for case in selected
            ),
            "uncapped_endpoint_sse_explained_fraction": _distribution(
                case_results[case]["methods"][mode]["endpoint_fit"][
                    "uncapped_endpoint_sse_explained_fraction"
                ]
                for case in selected
            ),
            "remaining_edge_difference_to_field_rms_ratio": _distribution(
                case_results[case]["methods"][mode]["remaining_graph"][
                    "edge_difference_to_field_rms_ratio"
                ]
                for case in selected
            ),
            "remaining_centered_edge_difference_to_field_rms_ratio": (
                _distribution(
                    case_results[case]["methods"][mode]["remaining_graph"][
                        "centered_edge_difference_to_field_rms_ratio"
                    ]
                    for case in selected
                )
            ),
            "remaining_centered_edge_vector_correlation": _distribution(
                case_results[case]["methods"][mode]["remaining_graph"][
                    "centered_edge_vector_correlation"
                ]
                for case in selected
            ),
            "remaining_controller_energy_concentration_ratio": _distribution(
                case_results[case]["methods"][mode]["remaining_localization"][
                    "near_controller"
                ]["energy_concentration_ratio"]
                for case in selected
            ),
            "remaining_ground_energy_concentration_ratio": _distribution(
                case_results[case]["methods"][mode]["remaining_localization"][
                    "near_ground"
                ]["energy_concentration_ratio"]
                for case in selected
            ),
            "remaining_controller_or_ground_energy_concentration_ratio": (
                _distribution(
                    case_results[case]["methods"][mode][
                        "remaining_localization"
                    ]["near_controller_or_ground"]["energy_concentration_ratio"]
                    for case in selected
                )
            ),
        }
        geometry_keys = {
            key
            for case in selected
            for key in (
                ()
                if case_results[case]["methods"][mode]["geometry"] is None
                else case_results[case]["methods"][mode]["geometry"].keys()
            )
            if isinstance(
                case_results[case]["methods"][mode]["geometry"].get(key),
                (int, float),
            )
            and case_results[case]["methods"][mode]["geometry"].get(key) is not None
        }
        if geometry_keys:
            method["geometry"] = {
                key: _distribution(
                    case_results[case]["methods"][mode]["geometry"][key]
                    for case in selected
                )
                for key in sorted(geometry_keys)
                if all(
                    key in case_results[case]["methods"][mode]["geometry"]
                    and case_results[case]["methods"][mode]["geometry"][key]
                    is not None
                    for case in selected
                )
            }
        methods[mode] = method
    return {"endpoint_anchor": anchor, "methods": methods}


def _gain_recovery(
    case_results: dict[str, object],
    selected: tuple[str, ...],
) -> dict[str, object]:
    metrics = tuple(
        case_results[selected[0]]["methods"]["per_point"]["future"]
    )
    output: dict[str, object] = {}
    for metric in metrics:
        normalized: dict[str, float] = {}
        for mode in SPATIAL_MODES:
            normalized[mode] = float(
                np.mean(
                    [
                        case_results[case]["methods"][mode]["future"][metric][
                            "candidate_mean_m"
                        ]
                        / case_results[case]["methods"][mode]["future"][metric][
                            "baseline_mean_m"
                        ]
                        for case in selected
                    ]
                )
            )
        denominator = 1.0 - normalized["per_point"]
        output[metric] = {
            mode: {
                "equal_case_normalized_error": normalized[mode],
                "fraction_of_per_point_gain": (
                    None
                    if abs(denominator) <= 1e-12
                    else (1.0 - normalized[mode]) / denominator
                ),
            }
            for mode in SPATIAL_MODES
        }
    return output


def run_spatial_mode_analysis(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str = "confirmation",
    cases: Iterable[str] | None = None,
    maximum_residual_m: float = 0.01,
    interpolation_neighbors: int = 4,
    ground_band_m: float = 0.01,
    bootstrap_samples: int = 10000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260711,
    force: bool = False,
) -> dict[str, object]:
    """Run all spatial controls on a locked subset of the main release."""

    root = Path(data_root)
    manifest_path = root / "evaluation_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in manifest["selected_cases"])
    if cases is not None:
        selected = tuple(dict.fromkeys(str(case) for case in cases))
    elif cohort == "all":
        selected = available
    elif cohort == "development":
        selected = tuple(case for case in available if case in DEVELOPMENT_CASES)
    elif cohort == "confirmation":
        selected = tuple(case for case in available if case not in DEVELOPMENT_CASES)
    else:
        raise ValueError("cohort must be all, development, or confirmation")
    missing = sorted(set(selected) - set(available))
    if missing or not selected:
        raise ValueError("invalid selected cases: " + ", ".join(missing))
    clusters = {case: phystwin_physical_object_cluster(case) for case in selected}
    output = Path(output_dir)
    specification = {
        "method": "main-cohort endpoint spatial-mode physical diagnosis",
        "status": "post-hoc mechanism analysis",
        "cohort": cohort,
        "cases": list(selected),
        "spatial_modes": list(SPATIAL_MODES),
        "endpoint_anchor": "final temporally filled training residual",
        "maximum_residual_m": maximum_residual_m,
        "interpolation_neighbors": interpolation_neighbors,
        "ground_band_m": ground_band_m,
        "future_inputs": "none",
        "model_selection": "none",
        "data_manifest": str(manifest_path.resolve()),
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": bootstrap_block_length,
            "seed": bootstrap_seed,
        },
    }
    locked = _lock_protocol(output, specification)
    case_results: dict[str, object] = {}
    paired_vs_released = {mode: {} for mode in SPATIAL_MODES}
    paired_vs_per_point = {mode: {} for mode in TRANSFORM_MODES}
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
                "maximum_residual_m": maximum_residual_m,
                "interpolation_neighbors": interpolation_neighbors,
                "ground_band_m": ground_band_m,
            }
            if summary["config"] != expected:
                raise RuntimeError(f"cached case uses a different protocol: {case}")
        else:
            summary = analyze_spatial_modes_case(
                case_dir / "final_data.pkl",
                case_dir / "inference.pkl",
                case_dir / "optimal_params.pkl",
                case_dir / "gt_track_3d.pkl",
                case_output,
                train_end_frame=train_end,
                maximum_residual_m=maximum_residual_m,
                interpolation_neighbors=interpolation_neighbors,
                ground_band_m=ground_band_m,
            )
        case_results[case] = {
            "physical_object": clusters[case],
            "endpoint_anchor": summary["endpoint_anchor"],
            "methods": summary["methods"],
        }
        per_point_future = summary["methods"]["per_point"]["future"]
        for mode in SPATIAL_MODES:
            future = summary["methods"][mode]["future"]
            baseline_metrics = {
                metric: np.asarray(values["baseline_by_frame_m"], dtype=float)
                for metric, values in future.items()
            }
            candidate_metrics = {
                metric: np.asarray(values["candidate_by_frame_m"], dtype=float)
                for metric, values in future.items()
            }
            paired_vs_released[mode][case] = (baseline_metrics, candidate_metrics)
            if mode in paired_vs_per_point:
                paired_vs_per_point[mode][case] = (
                    {
                        metric: np.asarray(
                            per_point_future[metric]["candidate_by_frame_m"],
                            dtype=float,
                        )
                        for metric in per_point_future
                    },
                    candidate_metrics,
                )

    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": specification["status"],
        "cohort": cohort,
        "case_count": len(selected),
        "physical_object_count": len(set(clusters.values())),
        "methods": list(SPATIAL_MODES),
        "case_results": case_results,
        "comparisons_vs_released": {
            mode: _compact_bootstrap(
                paired_block_bootstrap(
                    paired,
                    samples=bootstrap_samples,
                    block_length=bootstrap_block_length,
                    seed=bootstrap_seed,
                    clusters=clusters,
                )
            )
            for mode, paired in paired_vs_released.items()
        },
        "comparisons_vs_per_point": {
            mode: _compact_bootstrap(
                paired_block_bootstrap(
                    paired,
                    samples=bootstrap_samples,
                    block_length=bootstrap_block_length,
                    seed=bootstrap_seed,
                    clusters=clusters,
                )
            )
            for mode, paired in paired_vs_per_point.items()
        },
        "gain_recovery": _gain_recovery(case_results, selected),
        "physical_diagnostics": _physical_aggregate(case_results, selected),
    }
    result_path = output / "spatial_mode_analysis_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["summary_path"] = str(result_path.resolve())
    return result
