"""Audit saved opened-data forecasts with an observation-space Gaussian solve.

This is a numerical verification of an already completed exploratory result,
not an independent empirical replication or authorization for new outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.deform_predictive_coupling import (
    ELLIPSOID_90_CHI2_3,
    METHODS,
    POLICIES,
    load_coupling_config,
)
from bayesian_phystwin_experiments.deform_sparse_observation_budget import (
    array_sha256,
    file_sha256,
    write_json,
)


def _frame(reference: np.ndarray) -> np.ndarray:
    points: np.ndarray = reference[0].astype(np.float64)
    x = points[10:12].mean(axis=0) - points[:2].mean(axis=0)
    x /= np.linalg.norm(x)
    y = 0.5 * (points[1] - points[0] + points[11] - points[10])
    y -= np.dot(y, x) * x
    if np.linalg.norm(y) <= 1e-10:
        y = np.array([0.0, 0.0, 1.0])
        if abs(float(y @ x)) > 1 - 1e-8:
            y = np.array([0.0, 1.0, 0.0])
        y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


def _posterior(
    observation_factor: np.ndarray,
    future_factor: np.ndarray,
    observation_noise: np.ndarray,
    future_floor: np.ndarray,
    selected: np.ndarray,
    innovation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = future_factor.shape[-1]
    coefficient = np.zeros(dimension)
    covariance = np.eye(dimension)
    if len(selected):
        jac = observation_factor[selected].reshape(-1, dimension)
        noise = np.zeros((3 * len(selected), 3 * len(selected)))
        for block, index in enumerate(selected):
            noise[3 * block : 3 * block + 3, 3 * block : 3 * block + 3] = (
                observation_noise[index]
            )
        marginal = jac @ jac.T + noise
        solved = np.linalg.solve(
            marginal, np.column_stack((innovation[selected].reshape(-1), jac))
        )
        coefficient = jac.T @ solved[:, 0]
        covariance -= jac.T @ solved[:, 1:]
    correction = future_factor @ coefficient
    blocks = (
        future_factor @ covariance @ np.swapaxes(future_factor, -1, -2) + future_floor
    )
    return correction, blocks


def _metrics(
    means: np.ndarray, covariance: np.ndarray, truth: np.ndarray, noise: float
) -> dict[str, np.ndarray]:
    difference = means - truth
    total = covariance + noise * np.eye(3)
    cholesky = np.linalg.cholesky(total)
    standardized = np.linalg.solve(cholesky, difference[..., None])[..., 0]
    nees = np.sum(standardized**2, axis=-1)
    logdet = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
    result = {
        "coordinate_l1_mm": np.mean(np.abs(difference), axis=(1, 2, 3)) * 1000,
        "point_rmse_mm": np.sqrt(np.mean(np.sum(difference**2, axis=-1), axis=(1, 2)))
        * 1000,
        "point_nees": nees.mean(axis=(1, 2)),
        "point_coverage_90": (nees <= ELLIPSOID_90_CHI2_3).mean(axis=(1, 2)),
        "gaussian_nll_per_point": (0.5 * (3 * np.log(2 * np.pi) + logdet + nees)).mean(
            axis=(1, 2)
        ),
        "ellipsoid_volume_mm3": (
            4 * np.pi / 3 * ELLIPSOID_90_CHI2_3**1.5 * np.exp(logdet / 2)
        ).mean(axis=(1, 2))
        * 1e9,
    }
    for label, indices in zip(
        ("early", "middle", "late"),
        np.array_split(np.arange(len(truth)), 3),
        strict=True,
    ):
        result[label + "_coordinate_l1_mm"] = (
            np.mean(np.abs(difference[:, indices]), axis=(1, 2, 3)) * 1000
        )
    return result


def verify(archive: Path, config_path: Path, root: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    config = load_coupling_config(config_path, repo)
    b = config.budget
    assert file_sha256(archive) == b.source_archive_sha256
    completion = json.loads((root / "run-complete.json").read_text())
    assert file_sha256(root / "results.json") == completion["results_sha256"]
    assert (
        file_sha256(root / "prediction-barrier.json")
        == completion["prediction_barrier_sha256"]
    )
    assert file_sha256(root / "predictive-coupling.png") == completion["plot_sha256"]
    barrier = json.loads((root / "prediction-barrier.json").read_text())
    result = json.loads((root / "results.json").read_text())
    input_manifest = json.loads((root / "input-manifest.json").read_text())
    assert result["source_revision"] == input_manifest["source_revision"]
    assert input_manifest["source_clean"] is True
    assert input_manifest["crossfit_source_future_outcomes_used"] is True
    for boundary_key in (
        "fresh_targets_accessed",
        "held_v8_accessed",
        "dlo4_dlo5_accessed",
        "held_own_future_input_to_predictor",
    ):
        assert input_manifest[boundary_key] is False
    assert input_manifest["config_sha256"] == file_sha256(config_path)
    assert input_manifest["module_sha256"] == file_sha256(
        repo / "src/bayesian_phystwin_experiments/deform_predictive_coupling.py"
    )
    with np.load(archive, allow_pickle=False) as data:
        names = data["names"].tolist()
        stop = b.forecast_end_exclusive - b.dataset_frame_offset
        reference = data["candidate_predictions"][:, :stop]
        variance = data["coordinate_variance_m2"][:, :stop]
        truth = data["targets"][:, :stop]
    held_names = sorted(set(names) - {config.design_case})
    assert len(held_names) == config.expected_trajectory_count - 1
    assert sorted(item["case"] for item in barrier["seals"]) == held_names
    assert barrier["case_count"] == len(held_names)
    local = np.stack(
        [
            (outcome.astype(float) - mean.astype(float)) @ _frame(mean)
            for outcome, mean in zip(truth, reference, strict=True)
        ]
    )
    local[:, :, [0, 1, 10, 11]] = 0
    frames = np.repeat(b.observation_frames, len(b.candidate_nodes))
    nodes = np.tile(b.candidate_nodes, len(b.observation_frames))
    offsets = frames - b.dataset_frame_offset
    start = b.prefix_end_exclusive - b.dataset_frame_offset
    counts = {
        "predictions": 0,
        "zero_budget_mean_checks": 0,
        "exact_guard_fallbacks": 0,
        "nonzero_guard_predictions": 0,
    }
    maximum_mean_error = 0.0
    maximum_covariance_error = 0.0
    case_metrics = {}
    for item in barrier["seals"]:
        name = item["case"]
        index = names.index(name)
        case_root = root / name.removesuffix(".pkl")
        assert file_sha256(case_root / "prediction-seal.json") == item["seal_sha256"]
        seal = json.loads((case_root / "prediction-seal.json").read_text())
        assert seal["held_name"] == name and seal["held_future_received"] is False
        source_names = sorted(set(names) - {name})
        assert seal["source_names"] == source_names
        validation_names = sorted(set(source_names) - {config.design_case})
        eligible = defaultdict(list)
        for entry in seal["guard"]:
            assert entry["validation_names"] == validation_names
            assert entry["validation_count"] == len(validation_names)
            passed = bool(
                entry["blend"] > 0
                and entry["budget"] > 0
                and max(entry["mean_l1_ratio"], entry["mean_rmse_ratio"])
                <= 1 - config.guard_minimum_mean_improvement
                and entry["joint_wins"] / len(validation_names)
                >= config.guard_minimum_joint_win_fraction
                and entry["worst_case_ratio"] <= config.guard_maximum_case_ratio
            )
            assert entry["eligible"] is passed
            if passed:
                eligible[(entry["policy"], entry["budget"])].append(
                    (
                        (entry["mean_l1_ratio"] + entry["mean_rmse_ratio"]) / 2,
                        entry["blend"],
                    )
                )
        source_indices = [names.index(source) for source in source_names]
        frame = _frame(reference[index])
        observation_errors = local[source_indices][:, offsets, nodes] @ frame.T
        future_errors = local[source_indices][:, start:, b.hidden_nodes] @ frame.T
        ref = reference[index, start:, b.hidden_nodes].transpose(1, 0, 2)
        base_covariance = variance[index, start:, b.hidden_nodes].transpose(1, 0, 2)[
            ..., :, None
        ] * np.eye(3)
        assert array_sha256(ref) == seal["reference_future_sha256"]
        assert array_sha256(base_covariance) == seal["baseline_covariance_sha256"]
        innovation = truth[index, offsets, nodes] - reference[index, offsets, nodes]
        models = {}
        for method, floor in (
            ("empirical_no_floor", 0.0),
            ("empirical_floor", config.floor_fraction),
            ("permuted_floor", config.floor_fraction),
        ):
            future = (
                np.roll(future_errors, 1, axis=0)
                if method == "permuted_floor"
                else future_errors
            )
            observation_factors = observation_errors.transpose(1, 2, 0) * np.sqrt(
                (1 - floor) / len(source_names)
            )
            future_factors = future.transpose(1, 2, 3, 0) * np.sqrt(
                (1 - floor) / len(source_names)
            )
            obs_noise = floor * np.mean(
                observation_errors[..., :, None] * observation_errors[..., None, :],
                axis=0,
            ) + b.measurement_std_m**2 * np.eye(3)
            future_floor = floor * np.mean(
                future[..., :, None] * future[..., None, :], axis=0
            )
            models[method] = (
                observation_factors,
                future_factors,
                obs_noise,
                future_floor,
            )
        basis = np.sin(
            np.pi * np.arange(1, 9)[:, None] * np.arange(1, b.graph_rank + 1) / 9
        ) / np.arange(1, b.graph_rank + 1)
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
        graph = np.zeros((stop, 12, 3, 3 * b.graph_rank))
        for coordinate in range(3):
            graph[:, 2:10, coordinate, coordinate::3] = (
                np.sqrt(variance[index, :, 2:10, coordinate])[..., None] * basis
            )
        models["graph_persistence"] = (
            graph[offsets, nodes],
            graph[start:, b.hidden_nodes],
            np.broadcast_to(b.measurement_std_m**2 * np.eye(3), (len(nodes), 3, 3)),
            np.zeros_like(base_covariance),
        )
        assert file_sha256(case_root / "predictions.npz") == seal["prediction_sha256"]
        with np.load(case_root / "predictions.npz", allow_pickle=False) as saved:
            means, covariance = saved["means"], saved["covariance_m2"]
        assert len(means) == len(seal["records"]) == len(covariance)
        expected_records = {
            (method, policy, count, repetition)
            for method in METHODS
            for policy in POLICIES
            for count in b.budgets
            for repetition in range(
                b.random_policy_repetitions if policy == "random" else 1
            )
        }
        expected_records.update(
            ("last_residual", "latest_uniform", count, 0) for count in b.budgets
        )
        expected_records.add(("unchanged_baseline", "none", 0, 0))
        assert len(seal["records"]) == len(expected_records)
        assert {
            (record["method"], record["policy"], record["budget"], record["repetition"])
            for record in seal["records"]
        } == expected_records
        cached = {}
        groups = defaultdict(list)
        for record_index, (record, mean, cov) in enumerate(
            zip(seal["records"], means, covariance, strict=True)
        ):
            selected = np.array(sorted(record["selected_indices"]), dtype=int)
            count = record["budget"]
            assert len(selected) == len(set(selected)) == count
            assert np.all(frames[selected] < b.prefix_end_exclusive)
            assert not set(nodes[selected]) & set(b.hidden_nodes)
            method = record["method"]
            if method == "source_guarded_floor":
                choices = eligible[(record["policy"], count)]
                assert record["blend"] == (min(choices)[1] if choices else 0)
            if method == "unchanged_baseline" or (
                method == "source_guarded_floor" and record["blend"] == 0
            ):
                expected_mean, expected_covariance = ref, base_covariance
            elif method == "last_residual":
                latest: dict[int, int] = {}
                for position in selected:
                    node = int(nodes[position])
                    if node in range(2, 10) and (
                        node not in latest or frames[position] > frames[latest[node]]
                    ):
                        latest[node] = position
                anchors = [1, *sorted(latest), 10]
                values = np.array(
                    [
                        np.zeros(3),
                        *[innovation[latest[node]] for node in sorted(latest)],
                        np.zeros(3),
                    ]
                )
                delta = np.column_stack(
                    [
                        np.interp(b.hidden_nodes, anchors, values[:, coordinate])
                        for coordinate in range(3)
                    ]
                )
                expected_mean, expected_covariance = ref + delta, base_covariance
            else:
                model_name = (
                    "empirical_floor" if method == "source_guarded_floor" else method
                )
                cache_key = (model_name, tuple(selected))
                if cache_key not in cached:
                    delta, conditional = _posterior(
                        *models[model_name], selected, innovation
                    )
                    if method == "graph_persistence":
                        conditional = np.diagonal(conditional, axis1=-2, axis2=-1)[
                            ..., :, None
                        ] * np.eye(3)
                    cached[cache_key] = (ref + delta, conditional)
                expected_mean, expected_covariance = cached[cache_key]
                if method == "source_guarded_floor":
                    blend = record["blend"]
                    delta = expected_mean - ref
                    expected_mean = ref + blend * delta
                    expected_covariance = (
                        (1 - blend) * base_covariance
                        + blend * expected_covariance
                        + blend
                        * (1 - blend)
                        * delta[..., :, None]
                        * delta[..., None, :]
                    )
            np.testing.assert_allclose(mean, expected_mean, atol=2e-10, rtol=1e-10)
            np.testing.assert_allclose(cov, expected_covariance, atol=2e-12, rtol=1e-8)
            maximum_mean_error = max(
                maximum_mean_error, float(np.max(np.abs(mean - expected_mean)))
            )
            maximum_covariance_error = max(
                maximum_covariance_error,
                float(np.max(np.abs(cov - expected_covariance))),
            )
            if count == 0:
                assert array_sha256(mean) == array_sha256(ref)
                counts["zero_budget_mean_checks"] += 1
            if method == "source_guarded_floor":
                if record["blend"] == 0:
                    assert array_sha256(mean) == array_sha256(ref)
                    assert array_sha256(cov) == array_sha256(base_covariance)
                    counts["exact_guard_fallbacks"] += 1
                else:
                    counts["nonzero_guard_predictions"] += 1
            counts["predictions"] += 1
            groups[(method, record["policy"], count)].append(record_index)
        metrics = _metrics(
            means,
            covariance,
            truth[index, start:, b.hidden_nodes].transpose(1, 0, 2),
            b.measurement_std_m**2,
        )
        for key, positions in groups.items():
            case_metrics[(name, *key)] = {
                metric: float(values[positions].mean())
                for metric, values in metrics.items()
            }
        print(f"Verified {name}: {len(means)} saved predictions", flush=True)
    for row in result["case_results"]:
        expected = case_metrics[
            (row["case"], row["method"], row["policy"], row["budget"])
        ]
        for metric, value in expected.items():
            np.testing.assert_allclose(row[metric], value, rtol=1e-9, atol=1e-9)
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap = rng.integers(
        0, len(held_names), (config.bootstrap_replicates, len(held_names))
    )
    for row in result["summaries"]:
        key = (row["method"], row["policy"], row["budget"])
        expected_cases = [case_metrics[(name, *key)] for name in held_names]
        assert row["case_count"] == len(held_names)
        for metric in expected_cases[0]:
            np.testing.assert_allclose(
                row[metric],
                np.mean([case[metric] for case in expected_cases]),
                rtol=1e-9,
                atol=1e-9,
            )
        for short, metric in (("l1", "coordinate_l1_mm"), ("rmse", "point_rmse_mm")):
            delta = np.array(
                [
                    case_metrics[(name, *key)][metric]
                    - case_metrics[(name, "unchanged_baseline", "none", 0)][metric]
                    for name in held_names
                ]
            )
            np.testing.assert_allclose(
                row[short + "_delta_mm"], delta.mean(), rtol=1e-9, atol=1e-9
            )
            np.testing.assert_allclose(
                row[short + "_delta_ci95_mm"],
                np.quantile(delta[bootstrap].mean(axis=1), [0.025, 0.975]),
                rtol=1e-9,
                atol=1e-9,
            )
    return {
        "schema": "deform-predictive-coupling-numerical-verification-v1",
        "status": "pass",
        "method": "observation-space-conditioning-and-cholesky-scoring",
        "independent_empirical_replication": False,
        "source_revision": result["source_revision"],
        "results_sha256": file_sha256(root / "results.json"),
        "verifier_sha256": file_sha256(Path(__file__)),
        "case_count": len(held_names),
        "physical_object_count": 1,
        "case_summary_count": len(case_metrics),
        "aggregate_count": len(result["summaries"]),
        **counts,
        "maximum_mean_difference_m": maximum_mean_error,
        "maximum_covariance_difference_m2": maximum_covariance_error,
        "fresh_targets_accessed": False,
        "held_v8_accessed": False,
        "dlo4_dlo5_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("verification records are write-once")
    report = verify(args.archive, args.config, args.run)
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
