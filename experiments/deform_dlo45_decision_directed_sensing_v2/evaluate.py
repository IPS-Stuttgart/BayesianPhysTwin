"""Decision-directed active sensing with competing source-fitted actions.

This source-test-only experiment uses existing DEFORM DLO4/DLO5 trajectories.
A source fit split defines a finite portfolio of competing future-shape
corrections. A disjoint source calibration split selects one sensor-likelihood
scale and one regret tolerance. A disjoint source-test split compares active
measurement policies. Official evaluation files are never exposed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from experiments.deform_dlo45_decision_directed_sensing_v1 import evaluate as core
from experiments.deform_dlo45_decision_identifiability_v1._model import (
    deterministic_kmeans,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-decision-directed-sensing-v2"
EXPECTED_POLICIES: Final = core.EXPECTED_POLICIES
ATOL: Final = 1e-12


@dataclass(frozen=True)
class V2Protocol:
    core_template: core.Protocol
    task_nodes: tuple[int, ...]
    query_projection_dimension: int
    likelihood_scales: tuple[float, ...]
    regret_tolerances: tuple[float, ...]
    calibration_budget: int
    maximum_nonfallback_harmful_fraction: float
    minimum_nonfallback_decisions: int
    claim_boundary: str

    def runtime(self, *, likelihood_scale: float, tolerance: float) -> core.Protocol:
        return dataclasses.replace(
            self.core_template,
            sensor_log_likelihood_scale=likelihood_scale,
            regret_tolerance=tolerance,
        )


@dataclass(frozen=True)
class CompetingContext:
    support_indices: IntArray
    base_logits: FloatArray
    support_sensor_features: FloatArray
    target_sensor_features: FloatArray
    support_residuals: FloatArray
    support_classes: IntArray
    support_state_representation: FloatArray
    support_query_representation: FloatArray
    fixed_actions: FloatArray
    relative_losses: FloatArray
    length_scale: float
    task_flat_indices: IntArray
    action_labels: tuple[str, ...]


class FrozenCase(NamedTuple):
    dlo: str
    trajectory: str
    current_frame: int
    context: CompetingContext
    target_residual: FloatArray


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> V2Protocol:
    raw = _read_json(path)
    if raw.get("contract") != CONTRACT or raw.get("schema_version") != 2:
        raise ValueError("unsupported decision-directed sensing v2 protocol")
    sections = {
        name: raw.get(name)
        for name in (
            "data",
            "windows",
            "source_split",
            "model",
            "calibration",
            "sensing",
            "evaluation",
        )
    }
    if not all(isinstance(value, dict) for value in sections.values()):
        raise ValueError("protocol sections must be JSON objects")
    data = sections["data"]
    windows = sections["windows"]
    split = sections["source_split"]
    model = sections["model"]
    calibration = sections["calibration"]
    sensing = sections["sensing"]
    evaluation = sections["evaluation"]
    assert isinstance(data, dict)
    assert isinstance(windows, dict)
    assert isinstance(split, dict)
    assert isinstance(model, dict)
    assert isinstance(calibration, dict)
    assert isinstance(sensing, dict)
    assert isinstance(evaluation, dict)

    if (
        tuple(data.get("dlos", ())) != core.DLOS
        or int(data.get("frame_count", -1)) != core.FRAME_COUNT
        or int(data.get("node_count", -1)) != core.NODE_COUNT
        or int(data.get("train_trajectory_count", -1)) != 56
        or int(data.get("evaluation_trajectory_count", -1)) != 14
        or tuple(data.get("known_endpoint_nodes", ())) != (0, 1, -2, -1)
        or tuple(data.get("candidate_internal_nodes", ())) != tuple(range(2, 10))
        or evaluation.get("primary_stage") != "source-test-only-competing-action-pilot"
        or evaluation.get("evaluation_split_opened") is not False
        or evaluation.get("target_tuning") is not False
        or evaluation.get("new_data_collection") is not False
    ):
        raise ValueError("frozen dataset or information boundary changed")

    fit_count = int(split["fit_count"])
    calibration_count = int(split["calibration_count"])
    source_test_count = int(split["source_test_count"])
    policies = tuple(str(value) for value in sensing["policies"])
    budgets = tuple(int(value) for value in sensing["measurement_budgets"])
    template = core.Protocol(
        first_current_frame=int(windows["first_current_frame"]),
        horizon_frames=int(windows["horizon_frames"]),
        stride_frames=int(windows["stride_frames"]),
        fit_count=fit_count,
        calibration_count=calibration_count,
        source_test_count=source_test_count,
        split_domain=str(split["domain_separator"]),
        support_neighbors=int(model["support_neighbors"]),
        response_clusters=int(model["response_clusters"]),
        action_scales=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        temperature_scale=float(model["temperature_scale"]),
        sensor_log_likelihood_scale=1.0,
        regret_tolerance=0.0,
        state_projection_dimension=int(model["state_projection_dimension"]),
        kmeans_iterations=int(model["kmeans_iterations"]),
        policies=policies,
        budgets=budgets,
        maximum_measurements=int(sensing["maximum_measurements"]),
        center_out_nodes=tuple(
            int(value) for value in sensing["center_out_internal_nodes"]
        ),
        random_seed=int(sensing["random_seed"]),
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )
    task_nodes = tuple(int(value) for value in data["task_internal_nodes"])
    likelihood_scales = tuple(
        float(value) for value in calibration["sensor_log_likelihood_scales"]
    )
    tolerances = tuple(float(value) for value in calibration["regret_tolerances"])
    result = V2Protocol(
        core_template=template,
        task_nodes=task_nodes,
        query_projection_dimension=int(model["query_projection_dimension"]),
        likelihood_scales=likelihood_scales,
        regret_tolerances=tolerances,
        calibration_budget=int(calibration["measurement_budget"]),
        maximum_nonfallback_harmful_fraction=float(
            calibration["maximum_nonfallback_harmful_fraction"]
        ),
        minimum_nonfallback_decisions=int(calibration["minimum_nonfallback_decisions"]),
        claim_boundary=str(raw["claim_boundary"]),
    )
    if (
        fit_count + calibration_count + source_test_count != 56
        or fit_count < 1
        or calibration_count < 1
        or source_test_count < 1
        or template.support_neighbors < 4
        or template.response_clusters < 2
        or template.response_clusters > template.support_neighbors
        or template.state_projection_dimension < 1
        or result.query_projection_dimension < 1
        or policies != EXPECTED_POLICIES
        or budgets[0] != 0
        or budgets[-1] != template.maximum_measurements
        or tuple(sorted(template.center_out_nodes)) != tuple(range(2, 10))
        or not task_nodes
        or len(set(task_nodes)) != len(task_nodes)
        or not set(task_nodes).issubset(set(range(2, 10)))
        or not likelihood_scales
        or any(value <= 0.0 for value in likelihood_scales)
        or tuple(sorted(likelihood_scales)) != likelihood_scales
        or not tolerances
        or any(value < 0.0 for value in tolerances)
        or tuple(sorted(tolerances)) != tolerances
        or result.calibration_budget not in budgets
        or not 0.0 <= result.maximum_nonfallback_harmful_fraction <= 1.0
        or result.minimum_nonfallback_decisions < 1
    ):
        raise ValueError("invalid frozen v2 protocol")
    return result


def task_flat_indices(protocol: V2Protocol) -> IntArray:
    local_nodes = [node - 2 for node in protocol.task_nodes]
    shaped = np.arange(
        protocol.core_template.horizon_frames * 8 * 3,
        dtype=np.int64,
    ).reshape(protocol.core_template.horizon_frames, 8, 3)
    return np.asarray(shaped[:, local_nodes, :].reshape(-1), dtype=np.int64)


def _project(values: FloatArray, dimension: int) -> FloatArray:
    centered = values - np.mean(values, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    count = min(dimension, right.shape[0])
    return np.asarray(centered @ right[:count].T, dtype=np.float64)


def fit_competing_source_model(
    base_features: FloatArray,
    sensor_features: FloatArray,
    residuals: FloatArray,
    protocol: V2Protocol,
) -> core.SourceModel:
    runtime = protocol.runtime(likelihood_scale=1.0, tolerance=0.0)
    base_model = core.fit_source_model(
        base_features,
        sensor_features,
        residuals,
        runtime,
    )
    task = residuals[:, task_flat_indices(protocol)]
    query_representation = _project(
        task,
        protocol.query_projection_dimension,
    )
    labels = deterministic_kmeans(
        query_representation,
        runtime.response_clusters,
        runtime.kmeans_iterations,
    )
    fallback_losses = np.mean(np.square(task), axis=1)
    loss_floor = max(float(np.quantile(fallback_losses, 0.05)) * 0.1, 1e-12)
    return dataclasses.replace(
        base_model,
        class_labels=np.asarray(labels, dtype=np.int64),
        query_representation=query_representation,
        loss_floor=loss_floor,
    )


def make_competing_context(
    observation: core.Observation,
    model: core.SourceModel,
    protocol: V2Protocol,
) -> CompetingContext:
    runtime = protocol.runtime(likelihood_scale=1.0, tolerance=0.0)
    base_pool = (model.base_features - model.base_mean) / model.base_scale
    base_query = (observation.base_feature - model.base_mean) / model.base_scale
    distances = np.mean(np.square(base_pool - base_query[None, :]), axis=1)
    count = min(runtime.support_neighbors, len(distances))
    support = np.argpartition(distances, count - 1)[:count]
    support = support[np.lexsort((support, distances[support]))]
    selected_distance = distances[support]
    positive = selected_distance[selected_distance > 0.0]
    bandwidth = (
        float(np.median(positive))
        if len(positive)
        else max(float(np.mean(selected_distance)), 1e-12)
    )
    bandwidth = max(bandwidth * runtime.temperature_scale, 1e-12)
    base_logits = -(selected_distance - float(np.min(selected_distance))) / bandwidth
    base_weights = core._softmax(base_logits)

    sensor_pool = (
        model.sensor_features - model.sensor_mean[None, :, :]
    ) / model.sensor_scale[None, :, :]
    target_sensor = (
        observation.sensor_features - model.sensor_mean
    ) / model.sensor_scale
    classes = core._local_classes(model.class_labels[support])
    residuals = model.residuals[support]

    actions = [np.zeros(residuals.shape[1], dtype=np.float64)]
    labels = ["physical_fallback"]
    for class_id in range(int(np.max(classes)) + 1):
        members = classes == class_id
        weights = base_weights[members]
        weights = weights / float(np.sum(weights))
        actions.append(np.einsum("i,id->d", weights, residuals[members]))
        labels.append(f"source_response_class_{class_id}")
    fixed_actions = np.asarray(actions, dtype=np.float64)

    indices = task_flat_indices(protocol)
    task_hypotheses = residuals[:, indices]
    task_actions = fixed_actions[:, indices]
    raw_losses = np.mean(
        np.square(task_hypotheses[:, None, :] - task_actions[None, :, :]),
        axis=2,
    )
    fallback_losses = np.mean(np.square(task_hypotheses), axis=1)
    relative_losses = raw_losses / (fallback_losses[:, None] + model.loss_floor)
    return CompetingContext(
        support_indices=np.asarray(support, dtype=np.int64),
        base_logits=np.asarray(base_logits, dtype=np.float64),
        support_sensor_features=np.asarray(
            sensor_pool[support],
            dtype=np.float64,
        ),
        target_sensor_features=np.asarray(target_sensor, dtype=np.float64),
        support_residuals=np.asarray(residuals, dtype=np.float64),
        support_classes=np.asarray(classes, dtype=np.int64),
        support_state_representation=np.asarray(
            model.state_representation[support],
            dtype=np.float64,
        ),
        support_query_representation=np.asarray(
            model.query_representation[support],
            dtype=np.float64,
        ),
        fixed_actions=fixed_actions,
        relative_losses=np.asarray(relative_losses, dtype=np.float64),
        length_scale=observation.length_scale,
        task_flat_indices=indices,
        action_labels=tuple(labels),
    )


def full_acquisition_path(
    policy: str,
    context: CompetingContext,
    key: str,
    runtime: core.Protocol,
) -> tuple[list[core.DecisionState], list[int]]:
    observations: dict[int, FloatArray] = {}
    states = [core.decision_state(context, observations, runtime)]
    selected: list[int] = []
    while len(selected) < runtime.maximum_measurements:
        remaining = tuple(index for index in range(8) if index not in observations)
        candidate = core.choose_candidate(
            policy,
            context,
            observations,
            remaining,
            key,
            runtime,
        )
        observations[candidate] = context.target_sensor_features[candidate]
        selected.append(candidate)
        states.append(core.decision_state(context, observations, runtime))
    return states, selected


def plan_from_path(
    policy: str,
    budget: int,
    states: list[core.DecisionState],
    selected: list[int],
    tolerance: float,
) -> dict[str, object]:
    allowed = min(budget, len(states) - 1)
    certified_step = next(
        (
            index
            for index in range(allowed + 1)
            if (states[index].certificate.minimax_worst_case_regret <= tolerance + ATOL)
        ),
        None,
    )
    if certified_step is None:
        action = 0
        sensor_count = allowed
        state = states[allowed]
        certified = False
    else:
        action = states[certified_step].certificate.minimax_action_index
        sensor_count = certified_step
        state = states[certified_step]
        certified = True
    return {
        "policy": policy,
        "budget": budget,
        "certified": certified,
        "action_index": int(action),
        "nonfallback": int(action) != 0,
        "sensor_count": sensor_count,
        "selected_internal_nodes": [
            selected[index] + 2 for index in range(sensor_count)
        ],
        "certificate_worst_case_regret": (state.certificate.minimax_worst_case_regret),
        "state_variance": state.state_variance,
        "query_variance": state.query_variance,
        "effective_hypothesis_count": state.effective_hypothesis_count,
        "posterior_entropy": state.posterior_entropy,
    }


def score_plan(
    plan: dict[str, object],
    context: CompetingContext,
    target_residual: FloatArray,
) -> dict[str, object]:
    task_truth = target_residual[context.task_flat_indices]
    task_actions = context.fixed_actions[:, context.task_flat_indices]
    task_mse = np.mean(
        np.square(task_truth[None, :] - task_actions),
        axis=1,
    )
    full_mse = np.mean(
        np.square(target_residual[None, :] - context.fixed_actions),
        axis=1,
    )
    action = int(plan["action_index"])
    scale = context.length_scale**2
    task_physical = task_mse * scale
    full_physical = full_mse * scale
    task_fallback = float(task_physical[0])
    task_selected = float(task_physical[action])
    full_fallback = float(full_physical[0])
    full_selected = float(full_physical[action])
    best = float(np.min(task_mse))
    denominator = max(float(task_mse[0]), ATOL)
    result = dict(plan)
    result.update(
        {
            "action_label": context.action_labels[action],
            "action_count": len(context.action_labels),
            "task_mse": task_selected,
            "task_fallback_mse": task_fallback,
            "full_mse": full_selected,
            "full_fallback_mse": full_fallback,
            "harmful_vs_task_fallback": bool(task_selected > task_fallback + ATOL),
            "normalized_realized_task_regret": (float(task_mse[action]) - best)
            / denominator,
        }
    )
    return result


def freeze_case(
    trajectory: FloatArray,
    current: int,
    dlo: str,
    trajectory_name: str,
    model: core.SourceModel,
    protocol: V2Protocol,
) -> tuple[core.Observation, CompetingContext, str]:
    observation = core.extract_endpoint_observation(
        trajectory,
        current,
        protocol.core_template,
    )
    context = make_competing_context(observation, model, protocol)
    key = f"{dlo}/{trajectory_name}/{current}"
    return observation, context, key


def score_frozen_case(
    trajectory: FloatArray,
    current: int,
    observation: core.Observation,
    context: CompetingContext,
    plans: list[dict[str, object]],
    dlo: str,
    trajectory_name: str,
    protocol: V2Protocol,
) -> list[dict[str, object]]:
    target = core.extract_target_residual(
        trajectory,
        current,
        observation,
        protocol.core_template,
    )
    rows = []
    for plan in plans:
        row = score_plan(plan, context, target)
        row.update(
            {
                "dlo": dlo,
                "trajectory": trajectory_name,
                "current_frame": current,
            }
        )
        rows.append(row)
    return rows


def evaluate_paths(
    paths: tuple[Path, ...],
    dlo: str,
    model: core.SourceModel,
    protocol: V2Protocol,
    *,
    likelihood_scale: float,
    tolerance: float,
    policies: tuple[str, ...],
    budgets: tuple[int, ...],
) -> list[dict[str, object]]:
    runtime = protocol.runtime(
        likelihood_scale=likelihood_scale,
        tolerance=0.0,
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        trajectory = core.load_trajectory(path)
        for current in core.window_starts(protocol.core_template):
            observation, context, key = freeze_case(
                trajectory,
                current,
                dlo,
                path.name,
                model,
                protocol,
            )
            plans: list[dict[str, object]] = []
            for policy in policies:
                states, selected = full_acquisition_path(
                    policy,
                    context,
                    key,
                    runtime,
                )
                for budget in budgets:
                    plans.append(
                        plan_from_path(
                            policy,
                            budget,
                            states,
                            selected,
                            tolerance,
                        )
                    )
            rows.extend(
                score_frozen_case(
                    trajectory,
                    current,
                    observation,
                    context,
                    plans,
                    dlo,
                    path.name,
                    protocol,
                )
            )
    return rows


def _equal_trajectory_metrics(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dlo"]), str(row["trajectory"]))
        groups.setdefault(key, []).append(row)
    trajectories = []
    for (dlo, trajectory), items in sorted(groups.items()):
        task = np.asarray([float(item["task_mse"]) for item in items])
        fallback = np.asarray([float(item["task_fallback_mse"]) for item in items])
        full = np.asarray([float(item["full_mse"]) for item in items])
        full_fallback = np.asarray([float(item["full_fallback_mse"]) for item in items])
        task_rmse = math.sqrt(float(np.mean(task)))
        fallback_rmse = math.sqrt(float(np.mean(fallback)))
        full_rmse = math.sqrt(float(np.mean(full)))
        full_fallback_rmse = math.sqrt(float(np.mean(full_fallback)))
        trajectories.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "decision_count": len(items),
                "task_rmse_mm": 1000.0 * task_rmse,
                "task_fallback_rmse_mm": 1000.0 * fallback_rmse,
                "task_relative_improvement": (
                    1.0 - task_rmse / max(fallback_rmse, ATOL)
                ),
                "full_rmse_mm": 1000.0 * full_rmse,
                "full_fallback_rmse_mm": 1000.0 * full_fallback_rmse,
                "full_relative_improvement": (
                    1.0 - full_rmse / max(full_fallback_rmse, ATOL)
                ),
                "nonfallback_fraction": float(
                    np.mean([bool(item["nonfallback"]) for item in items])
                ),
                "certified_fraction": float(
                    np.mean([bool(item["certified"]) for item in items])
                ),
                "harmful_fraction": float(
                    np.mean([bool(item["harmful_vs_task_fallback"]) for item in items])
                ),
                "mean_sensor_count": float(
                    np.mean([int(item["sensor_count"]) for item in items])
                ),
            }
        )
    task_improvements = np.asarray(
        [float(item["task_relative_improvement"]) for item in trajectories]
    )
    full_improvements = np.asarray(
        [float(item["full_relative_improvement"]) for item in trajectories]
    )
    nonfallback = [row for row in rows if bool(row["nonfallback"])]
    harmful_nonfallback = [
        row for row in nonfallback if bool(row["harmful_vs_task_fallback"])
    ]
    return {
        "decision_count": len(rows),
        "trajectory_count": len(trajectories),
        "task_pooled_rmse_mm": 1000.0
        * math.sqrt(float(np.mean([float(row["task_mse"]) for row in rows]))),
        "task_pooled_fallback_rmse_mm": 1000.0
        * math.sqrt(float(np.mean([float(row["task_fallback_mse"]) for row in rows]))),
        "task_mean_trajectory_improvement": float(np.mean(task_improvements)),
        "full_pooled_rmse_mm": 1000.0
        * math.sqrt(float(np.mean([float(row["full_mse"]) for row in rows]))),
        "full_pooled_fallback_rmse_mm": 1000.0
        * math.sqrt(float(np.mean([float(row["full_fallback_mse"]) for row in rows]))),
        "full_mean_trajectory_improvement": float(np.mean(full_improvements)),
        "nonfallback_fraction": float(
            np.mean([bool(row["nonfallback"]) for row in rows])
        ),
        "certified_fraction": float(np.mean([bool(row["certified"]) for row in rows])),
        "mean_sensor_count": float(np.mean([int(row["sensor_count"]) for row in rows])),
        "harmful_fraction": float(
            np.mean([bool(row["harmful_vs_task_fallback"]) for row in rows])
        ),
        "nonfallback_harmful_fraction": (
            len(harmful_nonfallback) / len(nonfallback) if nonfallback else 0.0
        ),
        "nonfallback_count": len(nonfallback),
        "mean_normalized_realized_task_regret": float(
            np.mean([float(row["normalized_realized_task_regret"]) for row in rows])
        ),
        "nonfallback_effective_hypothesis_count": (
            float(
                np.mean(
                    [float(row["effective_hypothesis_count"]) for row in nonfallback]
                )
            )
            if nonfallback
            else 0.0
        ),
        "nonfallback_state_variance": (
            float(np.mean([float(row["state_variance"]) for row in nonfallback]))
            if nonfallback
            else 0.0
        ),
        "per_trajectory": trajectories,
    }


def select_calibration(
    models: dict[str, core.SourceModel],
    calibration_paths: dict[str, tuple[Path, ...]],
    protocol: V2Protocol,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for scale in protocol.likelihood_scales:
        scale_rows: dict[float, list[dict[str, object]]] = {
            tolerance: [] for tolerance in protocol.regret_tolerances
        }
        for dlo in core.DLOS:
            runtime = protocol.runtime(
                likelihood_scale=scale,
                tolerance=0.0,
            )
            for path in calibration_paths[dlo]:
                trajectory = core.load_trajectory(path)
                for current in core.window_starts(protocol.core_template):
                    observation, context, key = freeze_case(
                        trajectory,
                        current,
                        dlo,
                        path.name,
                        models[dlo],
                        protocol,
                    )
                    states, selected = full_acquisition_path(
                        "decision_regret",
                        context,
                        key,
                        runtime,
                    )
                    plans = [
                        plan_from_path(
                            "decision_regret",
                            protocol.calibration_budget,
                            states,
                            selected,
                            tolerance,
                        )
                        for tolerance in protocol.regret_tolerances
                    ]
                    scored = score_frozen_case(
                        trajectory,
                        current,
                        observation,
                        context,
                        plans,
                        dlo,
                        path.name,
                        protocol,
                    )
                    for tolerance, row in zip(
                        protocol.regret_tolerances,
                        scored,
                        strict=True,
                    ):
                        row["likelihood_scale"] = scale
                        row["regret_tolerance"] = tolerance
                        scale_rows[tolerance].append(row)
        for tolerance, rows in scale_rows.items():
            metrics = _equal_trajectory_metrics(rows)
            eligible = bool(
                metrics["nonfallback_count"] >= protocol.minimum_nonfallback_decisions
                and metrics["nonfallback_harmful_fraction"]
                <= protocol.maximum_nonfallback_harmful_fraction
                and metrics["task_mean_trajectory_improvement"] > 0.0
            )
            candidate = {
                "likelihood_scale": scale,
                "regret_tolerance": tolerance,
                "eligible": eligible,
                "metrics": metrics,
            }
            candidates.append(candidate)
            all_rows.extend(rows)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    pool = eligible if eligible else candidates
    selected = max(
        pool,
        key=lambda candidate: (
            float(candidate["metrics"]["task_mean_trajectory_improvement"]),
            float(candidate["metrics"]["nonfallback_fraction"]),
            -float(candidate["metrics"]["mean_sensor_count"]),
            -float(candidate["metrics"]["nonfallback_harmful_fraction"]),
            -float(candidate["likelihood_scale"]),
            -float(candidate["regret_tolerance"]),
        ),
    )
    result = {
        "selected_likelihood_scale": selected["likelihood_scale"],
        "selected_regret_tolerance": selected["regret_tolerance"],
        "selected_candidate_eligible": selected["eligible"],
        "selection_pool_had_eligible_candidate": bool(eligible),
        "selection_objective": (
            "maximize equal-trajectory task RMSE improvement, then "
            "nonfallback fraction, then minimize measurements and harm"
        ),
        "candidates": candidates,
    }
    return result, all_rows


def _bootstrap_interval(
    values: FloatArray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(np.mean(values[sample]))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def aggregate_test_rows(
    rows: list[dict[str, object]],
    protocol: V2Protocol,
) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for policy_index, policy in enumerate(protocol.core_template.policies):
        policy_rows: dict[str, object] = {}
        for budget in protocol.core_template.budgets:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and row["budget"] == budget
            ]
            metrics = _equal_trajectory_metrics(selected)
            trajectory_values = np.asarray(
                [
                    float(item["task_relative_improvement"])
                    for item in metrics["per_trajectory"]
                ],
                dtype=np.float64,
            )
            metrics["task_trajectory_bootstrap_95_interval"] = list(
                _bootstrap_interval(
                    trajectory_values,
                    protocol.core_template.bootstrap_replicates,
                    protocol.core_template.bootstrap_seed + 100 * policy_index + budget,
                )
            )
            node_counts = {str(node): 0 for node in range(2, 10)}
            for row in selected:
                for node in row["selected_internal_nodes"]:
                    node_counts[str(node)] += 1
            metrics["selected_node_counts"] = node_counts
            policy_rows[str(budget)] = metrics
        aggregate[policy] = policy_rows
    return aggregate


def render_summary(
    result: dict[str, object],
    protocol: V2Protocol,
) -> str:
    calibration = result["calibration"]
    aggregate = result["aggregate"]
    assert isinstance(calibration, dict)
    assert isinstance(aggregate, dict)
    lines = [
        "# DEFORM decision-directed virtual sensing v2",
        "",
        "Status: **source-calibrated source-test exploratory result**",
        "",
        (
            "Selected on the disjoint source-calibration split: "
            f"likelihood scale `{calibration['selected_likelihood_scale']}`, "
            f"regret tolerance `{calibration['selected_regret_tolerance']}`."
        ),
        "",
        "| Policy | Budget | Task RMSE [mm] | Task improvement | "
        "Nonfallback | Mean measurements | Harm | "
        "Effective hypotheses when acting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shown_budgets = tuple(
        value for value in protocol.core_template.budgets if value in {0, 1, 2, 4, 8}
    )
    for policy in protocol.core_template.policies:
        policy_result = aggregate[policy]
        assert isinstance(policy_result, dict)
        for budget in shown_budgets:
            row = policy_result[str(budget)]
            assert isinstance(row, dict)
            lines.append(
                f"| {policy} | {budget} | "
                f"{float(row['task_pooled_rmse_mm']):.3f} | "
                f"{100.0 * float(row['task_mean_trajectory_improvement']):.2f}% | "
                f"{100.0 * float(row['nonfallback_fraction']):.1f}% | "
                f"{float(row['mean_sensor_count']):.2f} | "
                f"{100.0 * float(row['harmful_fraction']):.2f}% | "
                f"{float(row['nonfallback_effective_hypothesis_count']):.2f} |"
            )
    lines.extend(
        (
            "",
            "## Boundary",
            "",
            protocol.claim_boundary,
            "",
        )
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = load_protocol(protocol_path)

    models: dict[str, core.SourceModel] = {}
    split_records: dict[str, object] = {}
    source_records: dict[str, object] = {}
    calibration_paths: dict[str, tuple[Path, ...]] = {}
    source_test_paths: dict[str, tuple[Path, ...]] = {}

    for dlo in core.DLOS:
        train_paths = core.trajectory_paths(dataset_root, dlo, "train")
        names = tuple(path.name for path in train_paths)
        split = core.split_names(names, dlo, protocol.core_template)
        by_name = {path.name: path for path in train_paths}
        fit_paths = tuple(by_name[name] for name in split["fit"])
        calibration_paths[dlo] = tuple(by_name[name] for name in split["calibration"])
        source_test_paths[dlo] = tuple(by_name[name] for name in split["source_test"])
        base, sensors, residuals, manifest = core.build_arrays(
            fit_paths,
            split["fit"],
            protocol.core_template,
        )
        models[dlo] = fit_competing_source_model(
            base,
            sensors,
            residuals,
            protocol,
        )
        split_records[dlo] = {name: list(values) for name, values in split.items()}
        source_records[dlo] = {
            "fit_trajectory_count": len(fit_paths),
            "fit_window_count": len(base),
            "calibration_trajectory_count": len(calibration_paths[dlo]),
            "source_test_trajectory_count": len(source_test_paths[dlo]),
            "response_class_count": int(len(np.unique(models[dlo].class_labels))),
            "fit_manifest": manifest,
        }

    calibration, calibration_rows = select_calibration(
        models,
        calibration_paths,
        protocol,
    )
    selected_scale = float(calibration["selected_likelihood_scale"])
    selected_tolerance = float(calibration["selected_regret_tolerance"])

    test_rows: list[dict[str, object]] = []
    for dlo in core.DLOS:
        test_rows.extend(
            evaluate_paths(
                source_test_paths[dlo],
                dlo,
                models[dlo],
                protocol,
                likelihood_scale=selected_scale,
                tolerance=selected_tolerance,
                policies=protocol.core_template.policies,
                budgets=protocol.core_template.budgets,
            )
        )
    aggregate = aggregate_test_rows(test_rows, protocol)
    result: dict[str, object] = {
        "contract": CONTRACT,
        "schema_version": 2,
        "status": "source-calibrated-source-test-exploratory-result",
        "protocol_sha256": _sha256_file(protocol_path),
        "source_revision": args.source_revision,
        "dataset_root_name": dataset_root.name,
        "source_split": split_records,
        "source": source_records,
        "calibration": calibration,
        "accounting": {
            "fit_trajectories": 2 * protocol.core_template.fit_count,
            "calibration_trajectories": (2 * protocol.core_template.calibration_count),
            "source_test_trajectories": (2 * protocol.core_template.source_test_count),
            "source_test_decision_windows": (
                2
                * protocol.core_template.source_test_count
                * len(core.window_starts(protocol.core_template))
            ),
            "calibration_case_rows": len(calibration_rows),
            "source_test_case_rows": len(test_rows),
            "official_evaluation_files_opened": False,
            "future_internal_nodes_used_before_action_selection": False,
            "new_data_collected": False,
        },
        "aggregate": aggregate,
        "claim_boundary": protocol.claim_boundary,
    }
    result["result_id"] = _canonical_sha256(result)
    _write_json(output_dir / "result.json", result)
    with (output_dir / "calibration_cases.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in calibration_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "source_test_cases.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in test_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "SUMMARY.md").write_text(
        render_summary(result, protocol),
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
