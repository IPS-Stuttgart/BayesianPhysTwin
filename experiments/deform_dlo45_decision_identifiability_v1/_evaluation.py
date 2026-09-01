"""Source model selection and held-trajectory scoring."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._common import (
    ATOL,
    DLOS,
    INTERNAL,
    FloatArray,
    Model,
    Protocol,
    extract_observation,
    load_trajectory,
    partition_names,
    window_starts,
)
from ._model import build_pool, decide, fit_model


def evaluate_paths(
    paths: tuple[Path, ...],
    model: Model,
    protocol: Protocol,
) -> dict[str, object]:
    methods = (
        "fallback",
        "certificate",
        "jeffrey_point",
        "kernel_point",
        "map_point",
        "oracle",
    )
    squared_error: dict[str, list[float]] = {name: [] for name in methods}
    normalized_regret: dict[str, list[float]] = {name: [] for name in methods}
    action_counts = {
        name: np.zeros(len(model.action_scales), dtype=np.int64) for name in methods
    }
    per_trajectory: list[dict[str, object]] = []
    ambiguity: list[float] = []
    specificity: list[float] = []
    minimax_regret: list[float] = []
    exact_unique = 0
    tolerance_unique = 0
    decisions = 0
    for path in paths:
        trajectory = load_trajectory(path)
        local_error: dict[str, list[float]] = {name: [] for name in methods}
        for current in window_starts(protocol):
            observation = extract_observation(trajectory, current, protocol)
            decision = decide(observation.feature, model, protocol)
            # The target label is deliberately sliced only after every action has
            # been selected from the registered observation.
            truth = trajectory[
                current + 1 : current + 1 + protocol.horizon_frames,
                INTERNAL,
                :,
            ].copy()
            actual_residual = (truth - observation.baseline).reshape(
                -1
            ) / observation.length_scale
            actions = model.action_scales[:, None] * decision.correction[None, :]
            normalized_mse = np.mean(
                np.square(actual_residual[None, :] - actions), axis=1
            )
            physical_mse = normalized_mse * observation.length_scale**2
            oracle_action = int(np.argmin(physical_mse))
            selected_actions = {
                "fallback": 0,
                "certificate": decision.certificate_action,
                "jeffrey_point": decision.jeffrey_action,
                "kernel_point": decision.kernel_action,
                "map_point": decision.map_action,
                "oracle": oracle_action,
            }
            best = float(np.min(normalized_mse))
            denominator = max(float(normalized_mse[0]), model.loss_floor)
            for method, action in selected_actions.items():
                value = float(physical_mse[action])
                squared_error[method].append(value)
                local_error[method].append(value)
                normalized_regret[method].append(
                    (float(normalized_mse[action]) - best) / denominator
                )
                action_counts[method][action] += 1
            ambiguity.append(decision.ambiguity_width)
            specificity.append(decision.unsupported_specificity_nats)
            minimax_regret.append(decision.minimax_regret)
            exact_unique += int(np.count_nonzero(decision.robust_mask) == 1)
            tolerance_unique += int(np.count_nonzero(decision.tolerance_mask) == 1)
            decisions += 1
        baseline_rmse = math.sqrt(float(np.mean(local_error["fallback"])))
        trajectory_record: dict[str, object] = {
            "trajectory": path.name,
            "decision_count": len(local_error["fallback"]),
            "fallback_rmse_mm": 1000.0 * baseline_rmse,
        }
        for method in methods[1:]:
            rmse = math.sqrt(float(np.mean(local_error[method])))
            trajectory_record[f"{method}_rmse_mm"] = 1000.0 * rmse
            trajectory_record[f"{method}_ratio"] = rmse / max(baseline_rmse, 1e-12)
        per_trajectory.append(trajectory_record)
    aggregate: dict[str, object] = {}
    fallback_rmse = math.sqrt(float(np.mean(squared_error["fallback"])))
    for method in methods:
        rmse = math.sqrt(float(np.mean(squared_error[method])))
        regrets = np.asarray(normalized_regret[method])
        aggregate[method] = {
            "rmse_mm": 1000.0 * rmse,
            "rmse_ratio_to_fallback": rmse / max(fallback_rmse, 1e-12),
            "mean_normalized_regret": float(np.mean(regrets)),
            "p95_normalized_regret": float(np.quantile(regrets, 0.95)),
            "harm_fraction_vs_fallback": float(
                np.mean(
                    np.asarray(squared_error[method])
                    > np.asarray(squared_error["fallback"]) + ATOL
                )
            ),
            "action_counts": action_counts[method].tolist(),
        }
    certificate_ratios = [
        float(record["certificate_ratio"]) for record in per_trajectory
    ]
    return {
        "decision_count": decisions,
        "aggregate": aggregate,
        "per_trajectory": per_trajectory,
        "certificate_nonfallback_fraction": float(
            1.0 - action_counts["certificate"][0] / max(decisions, 1)
        ),
        "certificate_exact_unique_fraction": exact_unique / max(decisions, 1),
        "certificate_tolerance_unique_fraction": tolerance_unique / max(decisions, 1),
        "maximum_certificate_trajectory_ratio": max(certificate_ratios),
        "mean_ambiguity_width_normalized": float(np.mean(ambiguity)),
        "positive_ambiguity_fraction": float(np.mean(np.asarray(ambiguity) > ATOL)),
        "mean_kernel_unsupported_specificity_nats": float(np.mean(specificity)),
        "mean_minimax_worst_case_regret": float(np.mean(minimax_regret)),
    }


def hyperparameter_grid(protocol: Protocol) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "neighbors": neighbors,
            "cluster_count": clusters,
            "temperature_scale": temperature,
            "regret_tolerance": tolerance,
        }
        for neighbors in protocol.neighbor_grid
        for clusters in protocol.cluster_grid
        for temperature in protocol.temperature_grid
        for tolerance in protocol.regret_tolerance_grid
    )


def calibration_score(result: dict[str, object]) -> tuple[float, ...]:
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    certificate = aggregate["certificate"]
    assert isinstance(certificate, dict)
    ratio = float(certificate["rmse_ratio_to_fallback"])
    p95 = float(certificate["p95_normalized_regret"])
    worst = float(result["maximum_certificate_trajectory_ratio"])
    coverage = float(result["certificate_nonfallback_fraction"])
    safety_penalty = max(0.0, ratio - 1.0) + max(0.0, worst - 1.10)
    objective = ratio + 0.25 * max(0.0, p95) + 0.02 * (1.0 - coverage)
    return safety_penalty, objective, -coverage, ratio, worst


def choose_model_for_dlo(
    train_paths: tuple[Path, ...],
    dlo: str,
    protocol: Protocol,
) -> tuple[Model, dict[str, object]]:
    names = tuple(path.name for path in train_paths)
    split = partition_names(names, dlo, protocol)
    fit_features, fit_residuals, _ = build_pool(
        train_paths,
        split["fit"],
        protocol,
    )
    calibration_paths = tuple(
        path for path in train_paths if path.name in set(split["calibration"])
    )
    candidates: list[tuple[tuple[float, ...], dict[str, object]]] = []
    for settings in hyperparameter_grid(protocol):
        model = fit_model(
            fit_features,
            fit_residuals,
            cluster_count=int(settings["cluster_count"]),
            neighbors=int(settings["neighbors"]),
            temperature_scale=float(settings["temperature_scale"]),
            regret_tolerance=float(settings["regret_tolerance"]),
            protocol=protocol,
        )
        result = evaluate_paths(calibration_paths, model, protocol)
        candidates.append(
            (
                calibration_score(result),
                {"settings": settings, "calibration": result},
            )
        )
    candidates.sort(key=lambda item: item[0])
    selected = candidates[0][1]
    settings = selected["settings"]
    assert isinstance(settings, dict)
    refit_names = split["fit"] + split["calibration"]
    refit_features, refit_residuals, _ = build_pool(
        train_paths,
        refit_names,
        protocol,
    )
    source_test_model = fit_model(
        refit_features,
        refit_residuals,
        cluster_count=int(settings["cluster_count"]),
        neighbors=int(settings["neighbors"]),
        temperature_scale=float(settings["temperature_scale"]),
        regret_tolerance=float(settings["regret_tolerance"]),
        protocol=protocol,
    )
    source_test_paths = tuple(
        path for path in train_paths if path.name in set(split["source_test"])
    )
    source_test = evaluate_paths(source_test_paths, source_test_model, protocol)
    all_features, all_residuals, _ = build_pool(train_paths, names, protocol)
    target_model = fit_model(
        all_features,
        all_residuals,
        cluster_count=int(settings["cluster_count"]),
        neighbors=int(settings["neighbors"]),
        temperature_scale=float(settings["temperature_scale"]),
        regret_tolerance=float(settings["regret_tolerance"]),
        protocol=protocol,
    )
    aggregate = source_test["aggregate"]
    assert isinstance(aggregate, dict)
    certificate = aggregate["certificate"]
    assert isinstance(certificate, dict)
    gate = {
        "passed": bool(
            float(certificate["rmse_ratio_to_fallback"])
            <= protocol.source_gate_mean_ratio
            and float(source_test["maximum_certificate_trajectory_ratio"])
            <= protocol.source_gate_worst_trajectory_ratio
            and float(source_test["certificate_nonfallback_fraction"])
            >= protocol.source_gate_minimum_nonfallback_fraction
        ),
        "maximum_mean_rmse_ratio": protocol.source_gate_mean_ratio,
        "maximum_worst_trajectory_rmse_ratio": (
            protocol.source_gate_worst_trajectory_ratio
        ),
        "minimum_nonfallback_fraction": (
            protocol.source_gate_minimum_nonfallback_fraction
        ),
    }
    record = {
        "partition": {name: list(values) for name, values in split.items()},
        "selected_settings": settings,
        "calibration": selected["calibration"],
        "source_test": source_test,
        "source_gate": gate,
        "candidate_count": len(candidates),
    }
    return target_model, record


def save_models(path: Path, models: dict[str, Model]) -> None:
    arrays: dict[str, object] = {}
    for dlo, model in models.items():
        prefix = dlo.lower()
        arrays[f"{prefix}_features"] = model.features
        arrays[f"{prefix}_residuals"] = model.residuals
        arrays[f"{prefix}_class_labels"] = model.class_labels
        arrays[f"{prefix}_feature_mean"] = model.feature_mean
        arrays[f"{prefix}_feature_scale"] = model.feature_scale
        arrays[f"{prefix}_action_scales"] = model.action_scales
        arrays[f"{prefix}_scalars"] = np.asarray(
            (
                model.loss_floor,
                float(model.neighbors),
                model.temperature_scale,
                model.regret_tolerance,
            ),
            dtype=np.float64,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_models(path: Path) -> dict[str, Model]:
    models: dict[str, Model] = {}
    with np.load(path, allow_pickle=False) as archive:
        for dlo in DLOS:
            prefix = dlo.lower()
            scalars = np.asarray(archive[f"{prefix}_scalars"], dtype=np.float64)
            models[dlo] = Model(
                features=np.asarray(archive[f"{prefix}_features"], dtype=np.float64),
                residuals=np.asarray(archive[f"{prefix}_residuals"], dtype=np.float64),
                class_labels=np.asarray(
                    archive[f"{prefix}_class_labels"], dtype=np.int64
                ),
                feature_mean=np.asarray(
                    archive[f"{prefix}_feature_mean"], dtype=np.float64
                ),
                feature_scale=np.asarray(
                    archive[f"{prefix}_feature_scale"], dtype=np.float64
                ),
                action_scales=np.asarray(
                    archive[f"{prefix}_action_scales"], dtype=np.float64
                ),
                loss_floor=float(scalars[0]),
                neighbors=int(round(float(scalars[1]))),
                temperature_scale=float(scalars[2]),
                regret_tolerance=float(scalars[3]),
            )
    return models


def bootstrap_interval(
    values: FloatArray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(np.mean(values[sample]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
