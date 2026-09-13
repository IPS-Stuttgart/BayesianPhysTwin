#!/usr/bin/env python3
"""Retrospective nonlinear-query test on immutable DEFORM DLO4/DLO5 beliefs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

CONTRACT = "bayesian-phystwin.deform-dlo-bayesian-value-query-protocol"
COVARIANCE_KEY = "bayesian_covariance_m2__calibrated_full_coordinate_covariance_v1"
DLOS = ("DLO4", "DLO5")
QUERY_IDS = (
    "terminal_chord_distance_m",
    "late_chord_distance_m",
    "terminal_squared_chord_distance_m2",
    "late_squared_chord_distance_m2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def as_sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return cast(Sequence[Any], value)


def verify(path: Path, record: Mapping[str, Any], prefix: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = int(record[f"{prefix}_size_bytes"])
    expected_hash = str(record[f"{prefix}_sha256"])
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise RuntimeError(f"frozen file identity changed: {path}")


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_trajectory(path: Path, identity: Mapping[str, Any]) -> np.ndarray:
    if path.stat().st_size != int(identity["size_bytes"]):
        raise RuntimeError(f"trajectory size changed: {path}")
    if sha256(path) != identity["sha256"]:
        raise RuntimeError(f"trajectory hash changed: {path}")
    with path.open("rb") as stream:
        raw = pickle.load(stream)  # noqa: S301 - hash-bound public benchmark
    array = np.asarray(raw, dtype=np.float32)
    if array.shape != (500, 3, 12) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid DEFORM trajectory: {path}")
    array = np.transpose(array, (0, 2, 1)).astype(np.float64, copy=True)
    array[:, :, 2] = np.clip(array[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return array


def chord_query(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    squared: bool,
) -> np.ndarray:
    chord = right - left
    length = np.linalg.norm(chord, axis=-1)
    if np.any(length <= 1e-12):
        raise ValueError("degenerate clamp chord")
    direction = chord / length[..., None]
    offset = points - left
    perpendicular = offset - np.sum(offset * direction, axis=-1)[..., None] * direction
    distance2 = np.maximum(np.sum(perpendicular**2, axis=-1), 0.0)
    return distance2 if squared else np.sqrt(distance2)


def exact_squared_mean(
    mean: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    direction = right - left
    direction /= np.linalg.norm(direction, axis=-1)[..., None]
    plug_in = chord_query(mean, left, right, squared=True)
    trace = np.trace(covariance, axis1=-2, axis2=-1)
    along = np.einsum("ci,cij,cj->c", direction, covariance, direction)
    return np.maximum(plug_in + trace - along, 0.0)


def covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    values, vectors = np.linalg.eigh(symmetric)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"non-positive query covariance: {np.min(values)}")
    return vectors * np.sqrt(values)[:, None, :]


def crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    first = np.mean(np.abs(samples - truth[None]), axis=0)
    ordered = np.sort(samples, axis=0)
    ranks = np.arange(1, count + 1, dtype=np.float64)
    half_pairwise = np.sum(
        (2.0 * ranks - count - 1.0)[:, None] * ordered,
        axis=0,
    ) / count**2
    score = first - half_pairwise
    if np.any(score < -1e-12) or not np.all(np.isfinite(score)):
        raise FloatingPointError("invalid CRPS")
    return np.maximum(score, 0.0)


def query_cells(
    candidate: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
    frame_selection: str,
    late_fraction: float,
) -> dict[str, np.ndarray]:
    horizon = candidate.shape[0]
    if frame_selection == "terminal":
        frames = np.array([horizon - 1])
    elif frame_selection == "final-quarter":
        count = max(1, math.ceil(horizon * late_fraction))
        frames = np.arange(horizon - count, horizon)
    else:
        raise ValueError(frame_selection)
    nodes = np.arange(2, 10)
    frame_grid, node_grid = np.meshgrid(frames, nodes, indexing="ij")
    flat_frames, flat_nodes = frame_grid.ravel(), node_grid.ravel()

    def clamps(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = np.mean(array[frames][:, (0, 1)], axis=1)
        right = np.mean(array[frames][:, (10, 11)], axis=1)
        return np.repeat(left, 8, axis=0), np.repeat(right, 8, axis=0)

    candidate_left, candidate_right = clamps(candidate)
    target_left, target_right = clamps(target)
    return {
        "mean": candidate[flat_frames, flat_nodes],
        "truth_point": target[flat_frames, flat_nodes],
        "covariance": covariance[flat_frames, flat_nodes],
        "candidate_left": candidate_left,
        "candidate_right": candidate_right,
        "target_left": target_left,
        "target_right": target_right,
    }


def evaluate_query(
    dlo: str,
    name: str,
    query: Mapping[str, Any],
    candidate: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
    monte_carlo: Mapping[str, Any],
    late_fraction: float,
) -> dict[str, object]:
    query_id = str(query["query_id"])
    squared = "squared" in query_id
    cells = query_cells(
        candidate,
        target,
        covariance,
        str(query["frame_selection"]),
        late_fraction,
    )
    cell_count = len(cells["mean"])
    sample_count = int(monte_carlo["sample_count"])
    chunk_size = int(monte_carlo["cell_chunk_size"])
    lower, upper = map(float, monte_carlo["quantile_interval"])
    rng = np.random.default_rng(
        stable_seed(int(monte_carlo["base_seed"]), dlo, name, query_id)
    )
    metric_names = (
        "plugin_mae",
        "plugin_mse",
        "full_mean_mae",
        "full_mean_mse",
        "full_median_mae",
        "diag_mean_mae",
        "diag_mean_mse",
        "diag_median_mae",
        "full_crps",
        "diag_crps",
        "full_coverage",
        "diag_coverage",
        "full_width",
        "diag_width",
        "full_shift",
        "full_diag_mean_difference",
        "full_split_half_difference",
    )
    totals = {key: 0.0 for key in metric_names}
    for start in range(0, cell_count, chunk_size):
        stop = min(start + chunk_size, cell_count)
        selection = slice(start, stop)
        mean = cells["mean"][selection]
        truth_point = cells["truth_point"][selection]
        covariance_chunk = cells["covariance"][selection]
        left = cells["candidate_left"][selection]
        right = cells["candidate_right"][selection]
        truth = chord_query(
            truth_point,
            cells["target_left"][selection],
            cells["target_right"][selection],
            squared=squared,
        )
        plugin = chord_query(mean, left, right, squared=squared)
        root = covariance_root(covariance_chunk)
        variance = np.diagonal(covariance_chunk, axis1=-2, axis2=-1).copy()
        standard = rng.standard_normal((sample_count, stop - start, 3))
        full_points = mean[None] + np.einsum(
            "cij,scj->sci", root, standard, optimize=True
        )
        diag_points = mean[None] + standard * np.sqrt(variance)[None]
        full_samples = chord_query(
            full_points, left[None], right[None], squared=squared
        )
        diag_samples = chord_query(
            diag_points, left[None], right[None], squared=squared
        )
        if squared:
            full_mean = exact_squared_mean(mean, left, right, covariance_chunk)
            diag_covariance = np.zeros_like(covariance_chunk)
            coordinates = np.arange(3)
            diag_covariance[:, coordinates, coordinates] = variance
            diag_mean = exact_squared_mean(mean, left, right, diag_covariance)
        else:
            full_mean = np.mean(full_samples, axis=0)
            diag_mean = np.mean(diag_samples, axis=0)
        full_median = np.median(full_samples, axis=0)
        diag_median = np.median(diag_samples, axis=0)
        full_interval = np.quantile(full_samples, (lower, upper), axis=0)
        diag_interval = np.quantile(diag_samples, (lower, upper), axis=0)
        half = sample_count // 2
        values = {
            "plugin_mae": np.abs(plugin - truth),
            "plugin_mse": (plugin - truth) ** 2,
            "full_mean_mae": np.abs(full_mean - truth),
            "full_mean_mse": (full_mean - truth) ** 2,
            "full_median_mae": np.abs(full_median - truth),
            "diag_mean_mae": np.abs(diag_mean - truth),
            "diag_mean_mse": (diag_mean - truth) ** 2,
            "diag_median_mae": np.abs(diag_median - truth),
            "full_crps": crps(full_samples, truth),
            "diag_crps": crps(diag_samples, truth),
            "full_coverage": (truth >= full_interval[0]) & (truth <= full_interval[1]),
            "diag_coverage": (truth >= diag_interval[0]) & (truth <= diag_interval[1]),
            "full_width": full_interval[1] - full_interval[0],
            "diag_width": diag_interval[1] - diag_interval[0],
            "full_shift": np.abs(full_mean - plugin),
            "full_diag_mean_difference": np.abs(full_mean - diag_mean),
            "full_split_half_difference": np.abs(
                np.mean(full_samples[:half], axis=0)
                - np.mean(full_samples[half:], axis=0)
            ),
        }
        for key, value in values.items():
            array = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"non-finite {query_id}/{key}")
            totals[key] += float(np.sum(array))
    means = {key: value / cell_count for key, value in totals.items()}
    return {
        "dlo": dlo,
        "trajectory": name,
        "query_id": query_id,
        "query_unit": query["unit"],
        "cell_count": cell_count,
        "plugin_mae": means["plugin_mae"],
        "plugin_mse": means["plugin_mse"],
        "full_posterior_mean_mae": means["full_mean_mae"],
        "full_posterior_mean_mse": means["full_mean_mse"],
        "full_posterior_median_mae": means["full_median_mae"],
        "diagonal_posterior_mean_mae": means["diag_mean_mae"],
        "diagonal_posterior_mean_mse": means["diag_mean_mse"],
        "diagonal_posterior_median_mae": means["diag_median_mae"],
        "full_crps": means["full_crps"],
        "diagonal_crps": means["diag_crps"],
        "plugin_point_mass_crps": means["plugin_mae"],
        "full_coverage_90": means["full_coverage"],
        "diagonal_coverage_90": means["diag_coverage"],
        "full_interval_width": means["full_width"],
        "diagonal_interval_width": means["diag_width"],
        "mean_absolute_full_query_shift_from_plugin": means["full_shift"],
        "mean_absolute_full_diagonal_query_mean_difference": means[
            "full_diag_mean_difference"
        ],
        "full_split_half_mean_difference": means["full_split_half_difference"],
    }


def equal_dlo_mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(
        np.mean(
            [
                np.mean([float(row[key]) for row in rows if row["dlo"] == dlo])
                for dlo in DLOS
            ]
        )
    )


def contrast(
    rows: Sequence[Mapping[str, object]],
    candidate_key: str,
    reference_key: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    differences = {}
    for dlo in DLOS:
        selected = [row for row in rows if row["dlo"] == dlo]
        if len(selected) != 14:
            raise ValueError(f"expected 14 {dlo} trajectories")
        differences[dlo] = np.array(
            [float(row[candidate_key]) - float(row[reference_key]) for row in selected]
        )
    samples = []
    for dlo in DLOS:
        values = differences[dlo]
        indices = rng.integers(0, 14, size=(replicates, 14))
        samples.append(np.mean(values[indices], axis=1))
    bootstrap = np.mean(np.stack(samples, axis=1), axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    observed = float(np.mean([np.mean(differences[dlo]) for dlo in DLOS]))
    all_differences = np.concatenate([differences[dlo] for dlo in DLOS])
    candidate_mean = equal_dlo_mean(rows, candidate_key)
    reference_mean = equal_dlo_mean(rows, reference_key)
    return {
        "candidate_metric": candidate_key,
        "reference_metric": reference_key,
        "candidate_equal_dlo_mean": candidate_mean,
        "reference_equal_dlo_mean": reference_mean,
        "candidate_minus_reference_equal_dlo_mean": observed,
        "bootstrap_95_interval": [float(lower), float(upper)],
        "relative_improvement_pct": 100.0
        * (reference_mean - candidate_mean)
        / reference_mean,
        "trajectory_wins": int(np.sum(all_differences < -1e-15)),
        "trajectory_ties": int(np.sum(np.abs(all_differences) <= 1e-15)),
        "trajectory_losses": int(np.sum(all_differences > 1e-15)),
        "favorable_interval_excludes_zero": bool(upper < 0.0),
    }


def summarize(
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    uncertainty = as_mapping(protocol["uncertainty"], "uncertainty")
    replicates = int(uncertainty["bootstrap_replicates"])
    seed = int(uncertainty["bootstrap_seed"])
    results = {}
    for query_id in QUERY_IDS:
        selected = [row for row in rows if row["query_id"] == query_id]
        if len(selected) != 28:
            raise ValueError(f"expected 28 records for {query_id}")
        pairs = {
            "full_posterior_mean_mse_minus_plugin_mse": (
                "full_posterior_mean_mse",
                "plugin_mse",
            ),
            "full_posterior_median_mae_minus_plugin_mae": (
                "full_posterior_median_mae",
                "plugin_mae",
            ),
            "full_crps_minus_diagonal_marginal_matched_crps": (
                "full_crps",
                "diagonal_crps",
            ),
            "full_crps_minus_plugin_point_mass_crps": (
                "full_crps",
                "plugin_point_mass_crps",
            ),
        }
        results[query_id] = {
            "query_unit": selected[0]["query_unit"],
            "trajectory_count": 28,
            "cell_count": sum(int(row["cell_count"]) for row in selected),
            "equal_dlo_metrics": {
                key: equal_dlo_mean(selected, key)
                for key in (
                    "plugin_mae",
                    "plugin_mse",
                    "full_posterior_mean_mae",
                    "full_posterior_mean_mse",
                    "full_posterior_median_mae",
                    "diagonal_posterior_mean_mae",
                    "diagonal_posterior_mean_mse",
                    "diagonal_posterior_median_mae",
                    "full_crps",
                    "diagonal_crps",
                    "plugin_point_mass_crps",
                    "full_coverage_90",
                    "diagonal_coverage_90",
                    "full_interval_width",
                    "diagonal_interval_width",
                    "mean_absolute_full_query_shift_from_plugin",
                    "mean_absolute_full_diagonal_query_mean_difference",
                    "full_split_half_mean_difference",
                )
            },
            "contrasts": {
                name: contrast(
                    selected,
                    candidate,
                    reference,
                    replicates,
                    stable_seed(seed, query_id, name),
                )
                for name, (candidate, reference) in pairs.items()
            },
        }
    return results


def evaluate(protocol_path: Path, output_dir: Path) -> dict[str, object]:
    protocol = read_json(protocol_path)
    if (
        protocol.get("schema") != CONTRACT
        or protocol.get("schema_version") != 1
        or protocol.get("status")
        != "retrospective-diagnostic-frozen-before-execution"
    ):
        raise ValueError("protocol identity changed")
    queries = [
        as_mapping(value, "query")
        for value in as_sequence(protocol["queries"], "queries")
    ]
    if tuple(query["query_id"] for query in queries) != QUERY_IDS:
        raise ValueError("query roster changed")
    execution = as_mapping(protocol["execution"], "execution")
    if any(
        execution[key] is not False
        for key in (
            "retraining",
            "refitting",
            "new_predictions",
            "target_selection",
            "target_calibration",
            "case_replacement",
        )
    ):
        raise ValueError("no-refit boundary changed")
    cache_root = Path(execution["cache_root"]).resolve()
    dataset_root = Path(execution["dataset_root"]).resolve()
    if not cache_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("frozen cache or dataset root is missing")
    parent = as_mapping(protocol["parent_artifacts"], "parent artifacts")
    monte_carlo = as_mapping(protocol["monte_carlo"], "Monte Carlo")
    late_fraction = float(
        as_mapping(protocol["query_geometry"], "geometry")["late_frame_fraction"]
    )
    records = []
    verified = {}
    diagnostics = {
        "maximum_clamped_candidate_target_mismatch_m": 0.0,
        "minimum_internal_covariance_eigenvalue_m2": math.inf,
        "maximum_internal_covariance_asymmetry_m2": 0.0,
    }
    for dlo in DLOS:
        frozen = as_mapping(parent[dlo], f"{dlo} parent")
        prediction_path = Path(frozen["prediction_path"]).resolve()
        manifest_path = Path(frozen["eval_manifest_path"]).resolve()
        seal_path = Path(frozen["prediction_seal_path"]).resolve()
        for path in (prediction_path, manifest_path, seal_path):
            path.relative_to(cache_root)
        verify(prediction_path, frozen, "prediction")
        verify(manifest_path, frozen, "eval_manifest")
        verify(seal_path, frozen, "prediction_seal")
        manifest = read_json(manifest_path)
        seal = read_json(seal_path)
        if (
            manifest.get("contract") != "deform-dlo45-eval-manifest-v1"
            or manifest.get("dlo") != dlo
            or seal.get("contract") != "deform-dlo45-target-prediction-seal-v1"
            or seal.get("point_mean_count") != 1
            or as_mapping(seal["method_seal"], "method seal")["sha256"]
            != frozen["method_seal_sha256"]
        ):
            raise RuntimeError(f"{dlo} parent contract changed")
        names = [
            str(value)
            for value in as_sequence(manifest["ordered_names"], "names")
        ]
        identities = as_mapping(manifest["trajectories"], "trajectory identities")
        targets = []
        trajectory_hashes = []
        for name in names:
            identity = as_mapping(identities[name], name)
            path = Path(identity["path"]).resolve()
            path.relative_to(dataset_root / dlo / "eval")
            targets.append(load_trajectory(path, identity)[2:])
            trajectory_hashes.append({"name": name, "sha256": identity["sha256"]})
        target = np.stack(targets)
        with np.load(prediction_path, allow_pickle=False) as archive:
            if [str(value) for value in archive["names"].tolist()] != names:
                raise RuntimeError(f"{dlo} prediction order changed")
            candidate = np.asarray(archive["candidate"], dtype=np.float64)
            covariance = np.asarray(archive[COVARIANCE_KEY], dtype=np.float64)
        if candidate.shape != (14, 498, 12, 3) or target.shape != candidate.shape:
            raise ValueError(f"{dlo} prediction shape changed")
        if covariance.shape != (14, 498, 12, 3, 3):
            raise ValueError(f"{dlo} covariance shape changed")
        internal = covariance[:, :, 2:10]
        asymmetry = float(
            np.max(np.abs(internal - np.swapaxes(internal, -1, -2)))
        )
        minimum = float(
            np.min(
                np.linalg.eigvalsh(
                    0.5 * (internal + np.swapaxes(internal, -1, -2))
                )
            )
        )
        if asymmetry > 1e-12 or minimum <= 0.0:
            raise ValueError(f"{dlo} covariance is invalid")
        mismatch = float(
            np.max(
                np.abs(
                    candidate[:, :, (0, 1, 10, 11)]
                    - target[:, :, (0, 1, 10, 11)]
                )
            )
        )
        diagnostics["maximum_clamped_candidate_target_mismatch_m"] = max(
            diagnostics["maximum_clamped_candidate_target_mismatch_m"], mismatch
        )
        diagnostics["minimum_internal_covariance_eigenvalue_m2"] = min(
            diagnostics["minimum_internal_covariance_eigenvalue_m2"], minimum
        )
        diagnostics["maximum_internal_covariance_asymmetry_m2"] = max(
            diagnostics["maximum_internal_covariance_asymmetry_m2"], asymmetry
        )
        for index, name in enumerate(names):
            for query in queries:
                records.append(
                    evaluate_query(
                        dlo,
                        name,
                        query,
                        candidate[index],
                        target[index],
                        covariance[index],
                        monte_carlo,
                        late_fraction,
                    )
                )
        verified[dlo] = {
            "prediction_sha256": frozen["prediction_sha256"],
            "eval_manifest_sha256": frozen["eval_manifest_sha256"],
            "prediction_seal_sha256": frozen["prediction_seal_sha256"],
            "trajectory_hashes": trajectory_hashes,
        }
    query_results = summarize(records, protocol)
    favorable = {}
    contrast_names = (
        "full_posterior_mean_mse_minus_plugin_mse",
        "full_posterior_median_mae_minus_plugin_mae",
        "full_crps_minus_diagonal_marginal_matched_crps",
        "full_crps_minus_plugin_point_mass_crps",
    )
    for name in contrast_names:
        favorable[name] = sum(
            bool(
                query_results[query_id]["contrasts"][name][
                    "favorable_interval_excludes_zero"
                ]
            )
            for query_id in QUERY_IDS
        )
    interpretation = as_mapping(protocol["interpretation"], "interpretation")
    result = {
        "schema": "bayesian-phystwin.deform-dlo-bayesian-value-query-result-v1",
        "schema_version": 1,
        "protocol_sha256": sha256(protocol_path),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "github_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "runner_name": os.environ.get("RUNNER_NAME", ""),
        "parent_workflow_run_id": execution["parent_workflow_run_id"],
        "verified_parent": verified,
        "trajectory_count": 28,
        "trajectory_query_record_count": len(records),
        "statistical_unit": protocol["statistical_unit"],
        "point_prediction_unchanged": True,
        "coordinate_marginal_parity_max_abs": 0.0,
        "retraining": False,
        "refitting": False,
        "new_predictions": False,
        "target_selection": False,
        "target_calibration": False,
        "queries": query_results,
        "favorable_interval_counts_out_of_four_queries": favorable,
        "diagnostics": diagnostics,
        "retrospective_target_already_open": True,
        "claim_authorized": False,
        "claim_boundary": interpretation["positive_result_boundary"],
        "negative_result_boundary": interpretation["negative_result_boundary"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "trajectory_records.json", {"records": records})
    with (output_dir / "trajectory_records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Nonlinear-query diagnostic",
        "",
        "Negative differences favor the full Bayesian arm.",
        "",
    ]
    for query_id in QUERY_IDS:
        lines.append(f"## `{query_id}`")
        for name, value in query_results[query_id]["contrasts"].items():
            interval = value["bootstrap_95_interval"]
            lines.append(
                f"- `{name}`: {value['candidate_minus_reference_equal_dlo_mean']:.8g} "
                f"[{interval[0]:.8g}, {interval[1]:.8g}]"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def self_test() -> None:
    left = np.array([[0.0, 0.0, 0.0]])
    right = np.array([[1.0, 0.0, 0.0]])
    point = np.array([[0.5, 3.0, 4.0]])
    covariance = np.diag([1.0, 4.0, 9.0])[None]
    assert np.allclose(chord_query(point, left, right, squared=False), [5.0])
    assert np.allclose(exact_squared_mean(point, left, right, covariance), [38.0])
    draws = np.array([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]])
    assert np.allclose(crps(draws, np.array([3.5, 4.0])), [1.5, 1.0])
    root = covariance_root(covariance)
    assert np.allclose(root @ np.swapaxes(root, -1, -2), covariance)
    print("deform DLO Bayesian-value query self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/deform_dlo_bayesian_value_query_v1/protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/deform-dlo-bayesian-value-query-v1"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        protocol = read_json(args.protocol)
        if protocol.get("schema") != CONTRACT:
            raise ValueError("protocol contract changed")
        print(json.dumps(protocol, indent=2, sort_keys=True))
    else:
        evaluate(args.protocol.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
