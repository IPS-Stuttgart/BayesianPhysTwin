"""Paired frame-level comparison of PhysTwin trajectory methods."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phystwin_official_evaluation import _nearest_distances


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_metrics_by_frame(
    vertices: np.ndarray,
    object_points: np.ndarray,
    object_visibilities: np.ndarray,
    gt_track_3d: np.ndarray,
    *,
    num_surface_points: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    """Return the per-frame values averaged by the official metrics."""

    trajectory = np.asarray(vertices, dtype=float)
    observed = np.asarray(object_points, dtype=float)
    visible = np.asarray(object_visibilities, dtype=bool)
    tracks = np.asarray(gt_track_3d, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("vertices must have shape (T, N, 3)")
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError("object_points must have shape (T, M, 3)")
    if visible.shape != observed.shape[:2]:
        raise ValueError("object_visibilities must match object_points")
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError("gt_track_3d must have shape (T, K, 3)")
    if not 0 <= start_frame < end_frame <= min(
        len(trajectory), len(observed), len(tracks)
    ):
        raise ValueError("invalid frame interval")

    initial_track_mask = np.isfinite(tracks[0]).all(axis=1)
    _, track_indices = _nearest_distances(
        trajectory[0],
        tracks[0, initial_track_mask],
        p=2,
    )
    chamfer = np.empty(end_frame - start_frame, dtype=float)
    track_error = np.empty_like(chamfer)
    for output_index, frame in enumerate(range(start_frame, end_frame)):
        current_observed = observed[frame, visible[frame]]
        distance, _ = _nearest_distances(
            trajectory[frame, :num_surface_points],
            current_observed,
            p=1,
        )
        chamfer[output_index] = np.mean(distance)
        current_tracks = tracks[frame, initial_track_mask]
        current_valid = np.isfinite(current_tracks).all(axis=1)
        if np.any(current_valid):
            residual = (
                trajectory[frame, track_indices][current_valid]
                - current_tracks[current_valid]
            )
            track_error[output_index] = np.mean(np.linalg.norm(residual, axis=1))
        else:
            track_error[output_index] = 0.0
    return {"chamfer_distance_m": chamfer, "track_error_m": track_error}


def _moving_block_indices(
    frame_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    effective_length = min(block_length, frame_count)
    block_count = int(np.ceil(frame_count / effective_length))
    starts = rng.integers(0, frame_count - effective_length + 1, size=block_count)
    indexes = np.concatenate(
        [np.arange(start, start + effective_length) for start in starts]
    )
    return indexes[:frame_count]


def _percent_change(candidate: np.ndarray, baseline: np.ndarray) -> float:
    return 100.0 * (float(np.mean(candidate)) / float(np.mean(baseline)) - 1.0)


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "probability_improved": float(np.mean(values < 0.0)),
    }


def phystwin_physical_object_cluster(case_name: str) -> str:
    """Map released interaction names to conservative physical-object groups."""

    for object_name in (
        "cloth_1",
        "cloth_3",
        "cloth_4",
        "sloth",
        "zebra",
        "dinosor",
        "rope_1",
        "rope_4",
        "rope",
        "weird_package",
    ):
        if object_name in case_name:
            return object_name
    if case_name == "single_lift_cloth":
        return "cloth"
    raise ValueError(f"cannot infer a released PhysTwin object for {case_name}")


def paired_block_bootstrap(
    cases: Mapping[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    *,
    samples: int = 10000,
    block_length: int = 5,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Bootstrap paired percent changes per case and across equal-weighted cases."""

    if not cases:
        raise ValueError("at least one case is required")
    if samples < 1 or block_length < 1:
        raise ValueError("samples and block_length must be positive")
    rng = np.random.default_rng(seed)
    metric_names = ("chamfer_distance_m", "track_error_m")
    per_case: dict[str, object] = {}
    bootstrap_by_case: dict[str, dict[str, np.ndarray]] = {}
    for case_name, (baseline, candidate) in cases.items():
        bootstrap_by_case[case_name] = {}
        case_summary: dict[str, object] = {}
        for metric in metric_names:
            baseline_values = np.asarray(baseline[metric], dtype=float)
            candidate_values = np.asarray(candidate[metric], dtype=float)
            if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 1:
                raise ValueError(f"{case_name} {metric} arrays must be paired vectors")
            draws = np.empty(samples, dtype=float)
            for sample in range(samples):
                indexes = _moving_block_indices(
                    len(baseline_values),
                    block_length,
                    rng,
                )
                draws[sample] = _percent_change(
                    candidate_values[indexes],
                    baseline_values[indexes],
                )
            bootstrap_by_case[case_name][metric] = draws
            case_summary[metric] = {
                "baseline_mean_m": float(np.mean(baseline_values)),
                "candidate_mean_m": float(np.mean(candidate_values)),
                "observed_percent_change": _percent_change(
                    candidate_values,
                    baseline_values,
                ),
                "bootstrap_percent_change": _interval(draws),
            }
        per_case[case_name] = case_summary

    case_names = tuple(cases)
    macro: dict[str, object] = {}
    for metric in metric_names:
        observed = np.array(
            [
                per_case[name][metric]["observed_percent_change"]
                for name in case_names
            ],
            dtype=float,
        )
        draws = np.empty(samples, dtype=float)
        for sample in range(samples):
            selected = rng.integers(0, len(case_names), size=len(case_names))
            draws[sample] = np.mean(
                [
                    bootstrap_by_case[case_names[index]][metric][sample]
                    for index in selected
                ]
            )
        macro[metric] = {
            "observed_macro_percent_change": float(np.mean(observed)),
            "case_and_frame_bootstrap_percent_change": _interval(draws),
        }
    result: dict[str, object] = {
        "samples": samples,
        "block_length": block_length,
        "seed": seed,
        "per_case": per_case,
        "macro": macro,
    }
    if clusters is not None:
        if set(clusters) != set(case_names):
            raise ValueError("clusters must assign every case exactly once")
        grouped: dict[str, list[str]] = {}
        for case_name in case_names:
            grouped.setdefault(str(clusters[case_name]), []).append(case_name)
        cluster_names = tuple(grouped)
        cluster_macro: dict[str, object] = {}
        for metric in metric_names:
            observed_by_cluster = np.array(
                [
                    np.mean(
                        [
                            per_case[case_name][metric]["observed_percent_change"]
                            for case_name in grouped[cluster]
                        ]
                    )
                    for cluster in cluster_names
                ],
                dtype=float,
            )
            draws = np.empty(samples, dtype=float)
            for sample in range(samples):
                selected = rng.integers(0, len(cluster_names), size=len(cluster_names))
                draws[sample] = np.mean(
                    [
                        np.mean(
                            [
                                bootstrap_by_case[case_name][metric][sample]
                                for case_name in grouped[cluster_names[index]]
                            ]
                        )
                        for index in selected
                    ]
                )
            cluster_macro[metric] = {
                "observed_equal_cluster_percent_change": float(
                    np.mean(observed_by_cluster)
                ),
                "cluster_and_frame_bootstrap_percent_change": _interval(draws),
            }
        result["cluster_macro"] = {
            "cluster_count": len(cluster_names),
            "case_counts": {
                cluster: len(grouped[cluster]) for cluster in cluster_names
            },
            "metrics": cluster_macro,
        }
    return result


def compare_phystwin_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    samples: int = 10000,
    block_length: int = 5,
    seed: int = 0,
    cluster_by_phystwin_object: bool = False,
) -> dict[str, object]:
    """Evaluate and bootstrap the baseline/candidate pairs in a JSON manifest."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw_cases: Sequence[Mapping[str, object]] = manifest.get("cases", ())
    if not raw_cases:
        raise ValueError("manifest must contain a nonempty cases list")
    cases: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    clusters: dict[str, str] = {}
    has_cluster = ["cluster" in case for case in raw_cases]
    if any(has_cluster) and not all(has_cluster):
        raise ValueError("manifest must assign either all clusters or none")
    inputs: dict[str, object] = {}
    for case in raw_cases:
        name = str(case["name"])
        if name in cases:
            raise ValueError(f"duplicate case name: {name}")
        final_path = Path(str(case["final_data"]))
        track_path = Path(str(case["gt_track_3d"]))
        baseline_path = Path(str(case["baseline_trajectory"]))
        candidate_path = Path(str(case["candidate_trajectory"]))
        final_data = _load_pickle(final_path)
        tracks = np.asarray(_load_pickle(track_path), dtype=float)
        baseline = np.asarray(_load_pickle(baseline_path), dtype=float)
        candidate = np.asarray(_load_pickle(candidate_path), dtype=float)
        observed = np.asarray(final_data["object_points"], dtype=float)
        visible = np.asarray(final_data["object_visibilities"], dtype=bool)
        surface_count = len(observed[0]) + len(final_data["surface_points"])
        start = int(case["start_frame"])
        end = int(case.get("end_frame", len(observed)))
        cases[name] = (
            official_metrics_by_frame(
                baseline,
                observed,
                visible,
                tracks,
                num_surface_points=surface_count,
                start_frame=start,
                end_frame=end,
            ),
            official_metrics_by_frame(
                candidate,
                observed,
                visible,
                tracks,
                num_surface_points=surface_count,
                start_frame=start,
                end_frame=end,
            ),
        )
        if all(has_cluster):
            clusters[name] = str(case["cluster"])
        inputs[name] = {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in {
                "final_data": final_path,
                "gt_track_3d": track_path,
                "baseline_trajectory": baseline_path,
                "candidate_trajectory": candidate_path,
            }.items()
        }
        inputs[name]["frame_interval"] = [start, end]
    if cluster_by_phystwin_object:
        if clusters:
            raise ValueError(
                "do not combine explicit clusters with PhysTwin object inference"
            )
        clusters = {
            case_name: phystwin_physical_object_cluster(case_name)
            for case_name in cases
        }
    if clusters:
        for case_name, cluster in clusters.items():
            inputs[case_name]["cluster"] = cluster
    result = {
        "schema_version": 1,
        "manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": _sha256(manifest_path),
        },
        "inputs": inputs,
        "cluster_weighting": (
            "equal physical object, with equal interactions within object"
            if clusters
            else None
        ),
        "bootstrap": paired_block_bootstrap(
            cases,
            samples=samples,
            block_length=block_length,
            seed=seed,
            clusters=clusters if clusters else None,
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
