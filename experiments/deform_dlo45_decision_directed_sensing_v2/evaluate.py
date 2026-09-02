"""Source-calibrated decision-directed sensing on DEFORM DLO4/DLO5.

The experiment exposes physically interpretable virtual measurements: the
recorded current line-relative position and one-frame velocity of one internal
DLO node.  A source-fitted endpoint-only model supplies a finite local support,
a response quotient, and genuinely competing future-shape actions.  The sensor
likelihood scale, action-prototype shrinkage, and decision-regret tolerance are
selected on disjoint source-calibration trajectories before source-test
outcomes are opened.

Every source-test acquisition path and action is frozen before future internal
node outcomes are sliced for scoring.  No new data are collected and official
evaluation files are absent from the runtime view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.query_decision_certificate_v1 import (
    QueryDecisionCertificateV1,
    query_decision_certificate,
)
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    ACTION_LEFT,
    ACTION_RIGHT,
    DLOS,
    FRAME_COUNT,
    INTERNAL,
    NODE_COUNT,
    load_trajectory,
    trajectory_paths,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import (
    deterministic_kmeans,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-decision-directed-sensing-v2"
EXPECTED_POLICIES: Final = (
    "decision_regret",
    "bayes_risk",
    "posterior_entropy",
    "class_entropy",
    "state_variance",
    "query_variance",
    "center_out",
    "random",
    "oracle_decision",
)
ADAPTIVE_POLICIES: Final = EXPECTED_POLICIES[:6]
ATOL: Final = 1e-12


@dataclass(frozen=True)
class Protocol:
    task_internal_nodes: tuple[int, ...]
    first_current_frame: int
    horizon_frames: int
    stride_frames: int
    fit_count: int
    calibration_count: int
    source_test_count: int
    split_domain: str
    support_neighbors: int
    response_clusters: int
    state_projection_dimension: int
    query_projection_dimension: int
    temperature_scale: float
    kmeans_iterations: int
    minimum_class_support: int
    sensor_log_likelihood_scales: tuple[float, ...]
    action_prototype_scales: tuple[float, ...]
    regret_tolerances: tuple[float, ...]
    calibration_budget: int
    maximum_nonfallback_harmful_fraction: float
    minimum_nonfallback_decisions: int
    selection_objective: str
    policies: tuple[str, ...]
    budgets: tuple[int, ...]
    maximum_measurements: int
    center_out_nodes: tuple[int, ...]
    measurement_costs: FloatArray
    random_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class SourceModel:
    base_features: FloatArray
    sensor_features: FloatArray
    full_residuals: FloatArray
    task_residuals: FloatArray
    class_labels: IntArray
    action_prototypes: FloatArray
    state_representation: FloatArray
    query_representation: FloatArray
    base_mean: FloatArray
    base_scale: FloatArray
    sensor_mean: FloatArray
    sensor_scale: FloatArray
    loss_floor: float
    class_counts: IntArray


class Observation(NamedTuple):
    base_feature: FloatArray
    sensor_features: FloatArray
    baseline: FloatArray
    length_scale: float


@dataclass(frozen=True)
class CaseContext:
    support_indices: IntArray
    base_logits: FloatArray
    support_sensor_features: FloatArray
    target_sensor_features: FloatArray
    support_classes: IntArray
    support_global_classes: IntArray
    support_state_representation: FloatArray
    support_query_representation: FloatArray
    support_task_residuals: FloatArray
    actions: FloatArray
    action_labels: tuple[str, ...]
    relative_losses: FloatArray
    length_scale: float


@dataclass(frozen=True)
class DecisionState:
    posterior_weights: FloatArray
    quotient_weights: FloatArray
    certificate: QueryDecisionCertificateV1
    minimax_action_index: int
    bayes_action_index: int
    bayes_risk: float
    state_variance: float
    query_variance: float
    effective_hypothesis_count: float
    effective_class_count: float
    posterior_entropy: float
    class_entropy: float


@dataclass(frozen=True)
class FrozenPlan:
    policy: str
    budget: int
    certified: bool
    action_index: int
    sensor_count: int
    measurement_cost: float
    selected_internal_nodes: tuple[int, ...]
    state: DecisionState


@dataclass(frozen=True)
class CalibrationChoice:
    sensor_log_likelihood_scale: float
    action_prototype_scale: float
    regret_tolerance: float
    gate_passed: bool
    summary: dict[str, object]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tuple_of(value: object, cast: type) -> tuple:
    if not isinstance(value, list):
        raise ValueError("protocol sequence must be a JSON array")
    return tuple(cast(item) for item in value)


def _float_array(value: object) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def load_protocol(path: Path) -> Protocol:
    raw = _read_json(path)
    if raw.get("contract") != CONTRACT or raw.get("schema_version") != 2:
        raise ValueError("unsupported decision-directed sensing protocol")
    data = raw.get("data")
    windows = raw.get("windows")
    split = raw.get("source_split")
    model = raw.get("model")
    calibration = raw.get("calibration")
    sensing = raw.get("sensing")
    evaluation = raw.get("evaluation")
    sections = (data, windows, split, model, calibration, sensing, evaluation)
    if not all(isinstance(value, dict) for value in sections):
        raise ValueError("protocol sections must be objects")
    assert isinstance(data, dict)
    assert isinstance(windows, dict)
    assert isinstance(split, dict)
    assert isinstance(model, dict)
    assert isinstance(calibration, dict)
    assert isinstance(sensing, dict)
    assert isinstance(evaluation, dict)
    if (
        tuple(data.get("dlos", ())) != DLOS
        or int(data.get("frame_count", -1)) != FRAME_COUNT
        or int(data.get("node_count", -1)) != NODE_COUNT
        or int(data.get("train_trajectory_count", -1)) != 56
        or int(data.get("evaluation_trajectory_count", -1)) != 14
        or tuple(data.get("known_endpoint_nodes", ())) != (0, 1, -2, -1)
        or tuple(data.get("candidate_internal_nodes", ())) != tuple(range(2, 10))
        or evaluation.get("primary_stage") != "source-test-only-competing-action-pilot"
        or evaluation.get("evaluation_split_opened") is not False
        or evaluation.get("target_tuning") is not False
        or evaluation.get("new_data_collection") is not False
    ):
        raise ValueError("frozen data or information boundary changed")
    costs = _float_array(sensing["measurement_costs_by_internal_node"])
    protocol = Protocol(
        task_internal_nodes=_tuple_of(data["task_internal_nodes"], int),
        first_current_frame=int(windows["first_current_frame"]),
        horizon_frames=int(windows["horizon_frames"]),
        stride_frames=int(windows["stride_frames"]),
        fit_count=int(split["fit_count"]),
        calibration_count=int(split["calibration_count"]),
        source_test_count=int(split["source_test_count"]),
        split_domain=str(split["domain_separator"]),
        support_neighbors=int(model["support_neighbors"]),
        response_clusters=int(model["response_clusters"]),
        state_projection_dimension=int(model["state_projection_dimension"]),
        query_projection_dimension=int(model["query_projection_dimension"]),
        temperature_scale=float(model["temperature_scale"]),
        kmeans_iterations=int(model["kmeans_iterations"]),
        minimum_class_support=int(model["minimum_class_support"]),
        sensor_log_likelihood_scales=_tuple_of(
            calibration["sensor_log_likelihood_scales"], float
        ),
        action_prototype_scales=_tuple_of(
            calibration["action_prototype_scales"], float
        ),
        regret_tolerances=_tuple_of(calibration["regret_tolerances"], float),
        calibration_budget=int(calibration["measurement_budget"]),
        maximum_nonfallback_harmful_fraction=float(
            calibration["maximum_nonfallback_harmful_fraction"]
        ),
        minimum_nonfallback_decisions=int(calibration["minimum_nonfallback_decisions"]),
        selection_objective=str(calibration["selection_objective"]),
        policies=_tuple_of(sensing["policies"], str),
        budgets=_tuple_of(sensing["measurement_budgets"], int),
        maximum_measurements=int(sensing["maximum_measurements"]),
        center_out_nodes=_tuple_of(sensing["center_out_internal_nodes"], int),
        measurement_costs=costs,
        random_seed=int(sensing["random_seed"]),
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )
    source_count = (
        protocol.fit_count + protocol.calibration_count + protocol.source_test_count
    )
    if (
        protocol.task_internal_nodes != (4, 5, 6, 7)
        or protocol.first_current_frame < 1
        or protocol.horizon_frames < 1
        or protocol.stride_frames < 1
        or source_count != 56
        or protocol.support_neighbors < 2
        or protocol.response_clusters < 2
        or protocol.state_projection_dimension < 1
        or protocol.query_projection_dimension < 1
        or protocol.temperature_scale <= 0.0
        or protocol.kmeans_iterations < 1
        or protocol.minimum_class_support < 1
        or not protocol.sensor_log_likelihood_scales
        or any(value <= 0.0 for value in protocol.sensor_log_likelihood_scales)
        or not protocol.action_prototype_scales
        or any(value <= 0.0 for value in protocol.action_prototype_scales)
        or not protocol.regret_tolerances
        or any(value < 0.0 for value in protocol.regret_tolerances)
        or protocol.calibration_budget not in protocol.budgets
        or not 0.0 <= protocol.maximum_nonfallback_harmful_fraction <= 1.0
        or protocol.minimum_nonfallback_decisions < 1
        or protocol.policies != EXPECTED_POLICIES
        or protocol.maximum_measurements != 8
        or protocol.budgets[0] != 0
        or protocol.budgets[-1] != protocol.maximum_measurements
        or tuple(sorted(protocol.center_out_nodes)) != tuple(range(2, 10))
        or costs.shape != (8,)
        or not np.all(np.isfinite(costs))
        or np.any(costs <= 0.0)
    ):
        raise ValueError("invalid frozen protocol")
    return protocol


def split_names(
    names: tuple[str, ...],
    dlo: str,
    protocol: Protocol,
) -> dict[str, tuple[str, ...]]:
    if len(names) != 56 or len(set(names)) != 56:
        raise ValueError(f"{dlo}: expected 56 unique training names")

    def key(name: str) -> tuple[bytes, str]:
        payload = (
            protocol.split_domain.encode()
            + b"\0"
            + dlo.encode()
            + b"\0"
            + name.encode()
        )
        return hashlib.sha256(payload).digest(), name

    ordered = tuple(sorted(names, key=key))
    first = protocol.fit_count
    second = first + protocol.calibration_count
    return {
        "fit": ordered[:first],
        "calibration": ordered[first:second],
        "source_test": ordered[second:],
    }


def window_starts(protocol: Protocol) -> tuple[int, ...]:
    stop = FRAME_COUNT - protocol.horizon_frames
    starts = tuple(range(protocol.first_current_frame, stop, protocol.stride_frames))
    if not starts:
        raise ValueError("protocol yields no windows")
    return starts


def _anchor_means(frame: FloatArray) -> tuple[FloatArray, FloatArray]:
    left = np.mean(frame[list(ACTION_LEFT)], axis=0)
    right = np.mean(frame[list(ACTION_RIGHT)], axis=0)
    return left, right


def _line_internal(left: FloatArray, right: FloatArray) -> FloatArray:
    count = NODE_COUNT - 4
    weights = np.linspace(
        1.0 / (count + 1),
        count / (count + 1),
        count,
        dtype=np.float64,
    )
    return (1.0 - weights[:, None]) * left + weights[:, None] * right


def extract_endpoint_observation(
    trajectory: FloatArray,
    current: int,
    protocol: Protocol,
) -> Observation:
    """Build endpoint-only input plus separately masked current-node readouts."""

    previous = trajectory[current - 1]
    present = trajectory[current]
    future_endpoints = trajectory[current + 1 : current + 1 + protocol.horizon_frames][
        :, [0, 1, NODE_COUNT - 2, NODE_COUNT - 1], :
    ]
    previous_left, previous_right = _anchor_means(previous)
    present_left, present_right = _anchor_means(present)
    future_left = np.mean(future_endpoints[:, :2, :], axis=1)
    future_right = np.mean(future_endpoints[:, 2:, :], axis=1)
    present_delta = present_right - present_left
    previous_delta = previous_right - previous_left
    length_scale = max(float(np.linalg.norm(present_delta)), 1e-6)
    previous_length = max(float(np.linalg.norm(previous_delta)), 1e-6)
    present_axis = present_delta / length_scale
    previous_axis = previous_delta / previous_length
    left_velocity = (present_left - previous_left) / length_scale
    right_velocity = (present_right - previous_right) / length_scale
    sample_indices = np.linspace(
        0,
        protocol.horizon_frames - 1,
        5,
        dtype=np.int64,
    )
    sampled_actions = (
        np.concatenate(
            (
                future_left[sample_indices] - present_left,
                future_right[sample_indices] - present_right,
            ),
            axis=1,
        )
        / length_scale
    )
    sampled_lengths = (
        np.linalg.norm(
            future_right[sample_indices] - future_left[sample_indices], axis=1
        )
        / length_scale
    )
    midpoint_velocity = (
        0.5
        * (present_left + present_right - previous_left - previous_right)
        / length_scale
    )
    base_feature = np.concatenate(
        (
            present_axis,
            previous_axis,
            left_velocity,
            right_velocity,
            sampled_actions.reshape(-1),
            sampled_lengths,
            np.asarray([previous_length / length_scale]),
            midpoint_velocity,
        )
    )

    present_line = _line_internal(present_left, present_right)
    previous_line = _line_internal(previous_left, previous_right)
    present_shape = (present[INTERNAL] - present_line) / length_scale
    previous_shape = (previous[INTERNAL] - previous_line) / length_scale
    shape_velocity = present_shape - previous_shape
    sensor_features = np.concatenate((present_shape, shape_velocity), axis=1)

    count = NODE_COUNT - 4
    weights = np.linspace(
        1.0 / (count + 1),
        count / (count + 1),
        count,
        dtype=np.float64,
    )
    baseline = (1.0 - weights[None, :, None]) * future_left[:, None, :] + weights[
        None, :, None
    ] * future_right[:, None, :]
    return Observation(
        base_feature=np.asarray(base_feature, dtype=np.float64),
        sensor_features=np.asarray(sensor_features, dtype=np.float64),
        baseline=np.asarray(baseline, dtype=np.float64),
        length_scale=length_scale,
    )


def extract_full_target_residual(
    trajectory: FloatArray,
    current: int,
    observation: Observation,
    protocol: Protocol,
) -> FloatArray:
    """Slice future internal-node outcomes after plans have been frozen."""

    truth = trajectory[
        current + 1 : current + 1 + protocol.horizon_frames,
        INTERNAL,
        :,
    ].copy()
    return np.asarray(
        ((truth - observation.baseline) / observation.length_scale).reshape(-1),
        dtype=np.float64,
    )


def task_residuals(full_residuals: FloatArray, protocol: Protocol) -> FloatArray:
    matrix = np.atleast_2d(full_residuals)
    shaped = matrix.reshape(
        matrix.shape[0],
        protocol.horizon_frames,
        NODE_COUNT - 4,
        3,
    )
    indices = np.asarray(
        [node - 2 for node in protocol.task_internal_nodes], dtype=np.int64
    )
    task = shaped[:, :, indices, :].reshape(matrix.shape[0], -1)
    if full_residuals.ndim == 1:
        return task[0]
    return task


def response_signature(task: FloatArray, protocol: Protocol) -> FloatArray:
    shaped = task.reshape(
        task.shape[0],
        protocol.horizon_frames,
        len(protocol.task_internal_nodes),
        3,
    )
    mean_all = np.mean(shaped, axis=(1, 2))
    final_mean = np.mean(shaped[:, -1], axis=1)
    per_node_mean = np.mean(shaped, axis=1).reshape(len(shaped), -1)
    per_node_final = shaped[:, -1].reshape(len(shaped), -1)
    quarter = max(protocol.horizon_frames // 4, 1)
    temporal_change = np.mean(shaped[:, -quarter:], axis=(1, 2)) - np.mean(
        shaped[:, :quarter], axis=(1, 2)
    )
    rms = np.sqrt(np.mean(np.square(shaped), axis=(1, 2, 3)))[:, None]
    return np.concatenate(
        (
            mean_all,
            final_mean,
            per_node_mean,
            per_node_final,
            temporal_change,
            rms,
        ),
        axis=1,
    )


def build_arrays(
    paths: tuple[Path, ...],
    names: tuple[str, ...],
    protocol: Protocol,
) -> tuple[FloatArray, FloatArray, FloatArray, list[dict[str, object]]]:
    wanted = set(names)
    base_features: list[FloatArray] = []
    sensor_features: list[FloatArray] = []
    residuals: list[FloatArray] = []
    records: list[dict[str, object]] = []
    for path in paths:
        if path.name not in wanted:
            continue
        trajectory = load_trajectory(path)
        count = 0
        for current in window_starts(protocol):
            observation = extract_endpoint_observation(trajectory, current, protocol)
            base_features.append(observation.base_feature)
            sensor_features.append(observation.sensor_features)
            residuals.append(
                extract_full_target_residual(
                    trajectory,
                    current,
                    observation,
                    protocol,
                )
            )
            count += 1
        records.append(
            {
                "trajectory": path.name,
                "sha256": _sha256_file(path),
                "windows": count,
            }
        )
    if len(records) != len(names):
        raise ValueError("source arrays omitted a requested trajectory")
    return (
        np.asarray(base_features, dtype=np.float64),
        np.asarray(sensor_features, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
        records,
    )


def _projection(values: FloatArray, dimension: int) -> FloatArray:
    centered = values - np.mean(values, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    selected = min(dimension, right.shape[0])
    return np.asarray(centered @ right[:selected].T, dtype=np.float64)


def fit_source_model(
    base_features: FloatArray,
    sensor_features: FloatArray,
    full_residuals: FloatArray,
    protocol: Protocol,
) -> SourceModel:
    task = task_residuals(full_residuals, protocol)
    labels = deterministic_kmeans(
        response_signature(task, protocol),
        protocol.response_clusters,
        protocol.kmeans_iterations,
    )
    class_count = int(np.max(labels)) + 1
    counts = np.bincount(labels, minlength=class_count).astype(np.int64)
    if class_count < 2 or np.any(counts < protocol.minimum_class_support):
        raise ValueError("source response quotient has inadequate class support")
    prototypes = np.stack(
        [np.mean(task[labels == class_id], axis=0) for class_id in range(class_count)]
    )
    fallback_losses = np.mean(np.square(task), axis=1)
    loss_floor = max(float(np.quantile(fallback_losses, 0.05)) * 0.1, 1e-12)
    return SourceModel(
        base_features=np.asarray(base_features, dtype=np.float64),
        sensor_features=np.asarray(sensor_features, dtype=np.float64),
        full_residuals=np.asarray(full_residuals, dtype=np.float64),
        task_residuals=np.asarray(task, dtype=np.float64),
        class_labels=np.asarray(labels, dtype=np.int64),
        action_prototypes=np.asarray(prototypes, dtype=np.float64),
        state_representation=_projection(
            full_residuals, protocol.state_projection_dimension
        ),
        query_representation=_projection(task, protocol.query_projection_dimension),
        base_mean=np.mean(base_features, axis=0),
        base_scale=np.maximum(np.std(base_features, axis=0), 1e-9),
        sensor_mean=np.mean(sensor_features, axis=0),
        sensor_scale=np.maximum(np.std(sensor_features, axis=0), 1e-9),
        loss_floor=loss_floor,
        class_counts=counts,
    )


def action_competition_summary(model: SourceModel) -> dict[str, object]:
    actions = np.concatenate(
        (
            np.zeros((1, model.action_prototypes.shape[1]), dtype=np.float64),
            model.action_prototypes,
        ),
        axis=0,
    )
    losses = np.mean(
        np.square(model.task_residuals[:, None, :] - actions[None, :, :]), axis=2
    )
    winners = np.argmin(losses, axis=1)
    counts = np.bincount(winners, minlength=len(actions)).astype(np.int64)
    distances = np.sqrt(
        np.mean(
            np.square(actions[:, None, :] - actions[None, :, :]),
            axis=2,
        )
    )
    upper = distances[np.triu_indices(len(actions), 1)]
    active = int(np.count_nonzero(counts))
    return {
        "action_count": len(actions),
        "pointwise_winner_counts": counts.tolist(),
        "pointwise_active_action_count": active,
        "minimum_pairwise_action_rmse": float(np.min(upper)),
        "median_pairwise_action_rmse": float(np.median(upper)),
        "prototype_norms": np.sqrt(
            np.mean(np.square(model.action_prototypes), axis=1)
        ).tolist(),
    }


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("posterior weights are degenerate")
    return weights / total


def _local_classes(labels: IntArray) -> tuple[IntArray, IntArray]:
    unique = np.unique(labels)
    remap = {int(value): index for index, value in enumerate(unique.tolist())}
    local = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    return local, np.asarray(unique, dtype=np.int64)


def _class_masses(weights: FloatArray, classes: IntArray) -> FloatArray:
    return np.bincount(
        classes,
        weights=weights,
        minlength=int(np.max(classes)) + 1,
    ).astype(np.float64)


def make_context(
    observation: Observation,
    model: SourceModel,
    action_prototype_scale: float,
    protocol: Protocol,
) -> CaseContext:
    base_pool = (model.base_features - model.base_mean) / model.base_scale
    base_query = (observation.base_feature - model.base_mean) / model.base_scale
    distances = np.mean(np.square(base_pool - base_query[None, :]), axis=1)
    count = min(protocol.support_neighbors, len(distances))
    support = np.argpartition(distances, count - 1)[:count]
    support = support[np.lexsort((support, distances[support]))]
    selected_distance = distances[support]
    positive = selected_distance[selected_distance > 0.0]
    bandwidth = (
        float(np.median(positive))
        if len(positive)
        else max(float(np.mean(selected_distance)), 1e-12)
    )
    bandwidth = max(bandwidth * protocol.temperature_scale, 1e-12)
    base_logits = -(selected_distance - float(np.min(selected_distance))) / bandwidth
    sensor_pool = (
        model.sensor_features - model.sensor_mean[None, :, :]
    ) / model.sensor_scale[None, :, :]
    target_sensor = (
        observation.sensor_features - model.sensor_mean
    ) / model.sensor_scale
    global_classes = model.class_labels[support]
    classes, _ = _local_classes(global_classes)
    actions = np.concatenate(
        (
            np.zeros((1, model.task_residuals.shape[1]), dtype=np.float64),
            action_prototype_scale * model.action_prototypes,
        ),
        axis=0,
    )
    selected_residuals = model.task_residuals[support]
    raw_losses = np.mean(
        np.square(selected_residuals[:, None, :] - actions[None, :, :]), axis=2
    )
    fallback_losses = np.mean(np.square(selected_residuals), axis=1)
    relative_losses = raw_losses / (fallback_losses[:, None] + model.loss_floor)
    labels = ("fallback",) + tuple(
        f"response_prototype_{index}"
        for index in range(model.action_prototypes.shape[0])
    )
    return CaseContext(
        support_indices=np.asarray(support, dtype=np.int64),
        base_logits=np.asarray(base_logits, dtype=np.float64),
        support_sensor_features=np.asarray(sensor_pool[support], dtype=np.float64),
        target_sensor_features=np.asarray(target_sensor, dtype=np.float64),
        support_classes=classes,
        support_global_classes=np.asarray(global_classes, dtype=np.int64),
        support_state_representation=np.asarray(
            model.state_representation[support], dtype=np.float64
        ),
        support_query_representation=np.asarray(
            model.query_representation[support], dtype=np.float64
        ),
        support_task_residuals=np.asarray(selected_residuals, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        action_labels=labels,
        relative_losses=np.asarray(relative_losses, dtype=np.float64),
        length_scale=observation.length_scale,
    )


def posterior_weights(
    context: CaseContext,
    observations: dict[int, FloatArray],
    sensor_log_likelihood_scale: float,
) -> FloatArray:
    logits = context.base_logits.copy()
    for sensor_index, value in observations.items():
        difference = context.support_sensor_features[:, sensor_index, :] - value
        logits -= sensor_log_likelihood_scale * np.mean(np.square(difference), axis=1)
    return _softmax(logits)


def _weighted_variance(values: FloatArray, weights: FloatArray) -> float:
    mean = np.einsum("i,id->d", weights, values)
    second = np.einsum("i,id->d", weights, np.square(values))
    variance = np.maximum(second - np.square(mean), 0.0)
    return float(np.mean(variance))


def _entropy(weights: FloatArray) -> float:
    positive = weights[weights > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def decision_state(
    context: CaseContext,
    observations: dict[int, FloatArray],
    sensor_log_likelihood_scale: float,
) -> DecisionState:
    weights = posterior_weights(context, observations, sensor_log_likelihood_scale)
    quotient = _class_masses(weights, context.support_classes)
    prior = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    certificate = query_decision_certificate(
        prior,
        quotient,
        context.support_classes,
        context.relative_losses,
        regret_tolerance=0.0,
    )
    expected_losses = np.einsum("i,ia->a", weights, context.relative_losses)
    bayes_action = int(np.argmin(expected_losses))
    class_square_sum = float(np.sum(np.square(quotient)))
    return DecisionState(
        posterior_weights=weights,
        quotient_weights=quotient,
        certificate=certificate,
        minimax_action_index=certificate.minimax_action_index,
        bayes_action_index=bayes_action,
        bayes_risk=float(expected_losses[bayes_action]),
        state_variance=_weighted_variance(
            context.support_state_representation, weights
        ),
        query_variance=_weighted_variance(
            context.support_query_representation, weights
        ),
        effective_hypothesis_count=float(1.0 / np.sum(np.square(weights))),
        effective_class_count=(
            float(1.0 / class_square_sum) if class_square_sum > 0.0 else 0.0
        ),
        posterior_entropy=_entropy(weights),
        class_entropy=_entropy(quotient),
    )


def _metric_value(state: DecisionState, metric: str) -> float:
    if metric == "decision_regret":
        return state.certificate.minimax_worst_case_regret
    if metric == "bayes_risk":
        return state.bayes_risk
    if metric == "posterior_entropy":
        return state.posterior_entropy
    if metric == "class_entropy":
        return state.class_entropy
    if metric == "state_variance":
        return state.state_variance
    if metric == "query_variance":
        return state.query_variance
    raise ValueError(f"unsupported expected metric: {metric}")


def expected_candidate_metric(
    context: CaseContext,
    observations: dict[int, FloatArray],
    candidate: int,
    metric: str,
    sensor_log_likelihood_scale: float,
) -> float:
    current = posterior_weights(context, observations, sensor_log_likelihood_scale)
    total = 0.0
    for outcome_index, probability in enumerate(current):
        if probability <= 0.0:
            continue
        hypothetical = dict(observations)
        hypothetical[candidate] = context.support_sensor_features[
            outcome_index, candidate
        ]
        state = decision_state(
            context,
            hypothetical,
            sensor_log_likelihood_scale,
        )
        total += float(probability) * _metric_value(state, metric)
    return total


def _stable_random_order(key: str, protocol: Protocol) -> tuple[int, ...]:
    seed_bytes = hashlib.sha256(f"{protocol.random_seed}\0{key}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.permutation(8))


def choose_candidate(
    policy: str,
    context: CaseContext,
    observations: dict[int, FloatArray],
    remaining: tuple[int, ...],
    key: str,
    sensor_log_likelihood_scale: float,
    protocol: Protocol,
) -> int:
    if policy in ADAPTIVE_POLICIES:
        current = decision_state(context, observations, sensor_log_likelihood_scale)
        current_value = _metric_value(current, policy)
        scores: list[tuple[float, float, float, int]] = []
        for candidate in remaining:
            expected = expected_candidate_metric(
                context,
                observations,
                candidate,
                policy,
                sensor_log_likelihood_scale,
            )
            cost = float(protocol.measurement_costs[candidate])
            gain_per_cost = (current_value - expected) / cost
            scores.append((-gain_per_cost, expected, cost, candidate))
        return min(scores)[-1]
    if policy == "oracle_decision":
        scores: list[tuple[float, float, int]] = []
        for candidate in remaining:
            hypothetical = dict(observations)
            hypothetical[candidate] = context.target_sensor_features[candidate]
            state = decision_state(context, hypothetical, sensor_log_likelihood_scale)
            scores.append(
                (
                    state.certificate.minimax_worst_case_regret,
                    float(protocol.measurement_costs[candidate]),
                    candidate,
                )
            )
        return min(scores)[-1]
    if policy == "center_out":
        order = tuple(node - 2 for node in protocol.center_out_nodes)
    elif policy == "random":
        order = _stable_random_order(key, protocol)
    else:
        raise ValueError(f"unsupported policy: {policy}")
    for candidate in order:
        if candidate in remaining:
            return candidate
    raise RuntimeError("fixed policy has no remaining candidate")


def acquisition_path(
    policy: str,
    context: CaseContext,
    key: str,
    sensor_log_likelihood_scale: float,
    protocol: Protocol,
) -> tuple[list[DecisionState], list[int]]:
    """Generate the full path; tolerance changes only the stopping rule."""

    observations: dict[int, FloatArray] = {}
    states = [decision_state(context, observations, sensor_log_likelihood_scale)]
    selected: list[int] = []
    while len(selected) < protocol.maximum_measurements:
        remaining = tuple(index for index in range(8) if index not in observations)
        candidate = choose_candidate(
            policy,
            context,
            observations,
            remaining,
            key,
            sensor_log_likelihood_scale,
            protocol,
        )
        observations[candidate] = context.target_sensor_features[candidate]
        selected.append(candidate)
        states.append(
            decision_state(context, observations, sensor_log_likelihood_scale)
        )
    return states, selected


def _budget_plan(
    policy: str,
    budget: int,
    states: list[DecisionState],
    selected: list[int],
    regret_tolerance: float,
    protocol: Protocol,
) -> FrozenPlan:
    allowed = min(budget, len(states) - 1)
    certified_step = next(
        (
            index
            for index in range(allowed + 1)
            if states[index].certificate.minimax_worst_case_regret
            <= regret_tolerance + ATOL
        ),
        None,
    )
    if certified_step is None:
        action = 0
        sensor_count = allowed
        state = states[allowed]
        certified = False
    else:
        action = states[certified_step].minimax_action_index
        sensor_count = certified_step
        state = states[certified_step]
        certified = True
    selected_nodes = tuple(selected[index] + 2 for index in range(sensor_count))
    cost = float(sum(protocol.measurement_costs[node - 2] for node in selected_nodes))
    return FrozenPlan(
        policy=policy,
        budget=budget,
        certified=certified,
        action_index=action,
        sensor_count=sensor_count,
        measurement_cost=cost,
        selected_internal_nodes=selected_nodes,
        state=state,
    )


def score_plan(
    plan: FrozenPlan,
    context: CaseContext,
    target_task_residual: FloatArray,
    model: SourceModel,
) -> dict[str, object]:
    normalized_mse = np.mean(
        np.square(target_task_residual[None, :] - context.actions), axis=1
    )
    physical_mse = normalized_mse * context.length_scale**2
    action = plan.action_index
    fallback = float(physical_mse[0])
    selected_mse = float(physical_mse[action])
    target_relative_losses = normalized_mse / (
        float(normalized_mse[0]) + model.loss_floor
    )
    best = float(np.min(target_relative_losses))
    realized_regret = float(target_relative_losses[action]) - best
    certificate_radius = plan.state.certificate.minimax_worst_case_regret
    pointwise_best = int(np.argmin(normalized_mse))
    certificate_evaluable = plan.certified
    certificate_excess = (
        realized_regret - certificate_radius if certificate_evaluable else 0.0
    )
    return {
        "policy": plan.policy,
        "budget": plan.budget,
        "certified": plan.certified,
        "action_index": action,
        "action_label": context.action_labels[action],
        "nonfallback": action != 0,
        "sensor_count": plan.sensor_count,
        "measurement_cost": plan.measurement_cost,
        "selected_internal_nodes": list(plan.selected_internal_nodes),
        "certificate_worst_case_regret": certificate_radius,
        "physical_task_mse": selected_mse,
        "fallback_task_mse": fallback,
        "harmful_vs_fallback": bool(selected_mse > fallback + ATOL),
        "normalized_realized_regret": realized_regret,
        "certificate_target_evaluable": certificate_evaluable,
        "certificate_excess_regret": certificate_excess,
        "certificate_target_violation": bool(
            certificate_evaluable and realized_regret > certificate_radius + ATOL
        ),
        "pointwise_best_action_index": pointwise_best,
        "pointwise_action_correct": action == pointwise_best,
        "state_variance": plan.state.state_variance,
        "query_variance": plan.state.query_variance,
        "effective_hypothesis_count": plan.state.effective_hypothesis_count,
        "effective_class_count": plan.state.effective_class_count,
        "posterior_entropy": plan.state.posterior_entropy,
        "class_entropy": plan.state.class_entropy,
        "bayes_risk": plan.state.bayes_risk,
    }


def _evaluate_calibration_trajectory(
    path: Path,
    dlo: str,
    model: SourceModel,
    protocol: Protocol,
) -> list[dict[str, object]]:
    trajectory = load_trajectory(path)
    rows: list[dict[str, object]] = []
    for current in window_starts(protocol):
        observation = extract_endpoint_observation(trajectory, current, protocol)
        key = f"calibration/{dlo}/{path.name}/{current}"
        frozen: list[tuple[float, float, float, CaseContext, FrozenPlan]] = []
        for prototype_scale in protocol.action_prototype_scales:
            context = make_context(observation, model, prototype_scale, protocol)
            for sensor_scale in protocol.sensor_log_likelihood_scales:
                states, selected = acquisition_path(
                    "decision_regret",
                    context,
                    key,
                    sensor_scale,
                    protocol,
                )
                for tolerance in protocol.regret_tolerances:
                    plan = _budget_plan(
                        "decision_regret",
                        protocol.calibration_budget,
                        states,
                        selected,
                        tolerance,
                        protocol,
                    )
                    frozen.append(
                        (
                            sensor_scale,
                            prototype_scale,
                            tolerance,
                            context,
                            plan,
                        )
                    )

        full_target = extract_full_target_residual(
            trajectory, current, observation, protocol
        )
        target_task = task_residuals(full_target, protocol)
        for sensor_scale, prototype_scale, tolerance, context, plan in frozen:
            record = score_plan(plan, context, target_task, model)
            record.update(
                {
                    "dlo": dlo,
                    "trajectory": path.name,
                    "current_frame": current,
                    "sensor_log_likelihood_scale": sensor_scale,
                    "action_prototype_scale": prototype_scale,
                    "regret_tolerance": tolerance,
                }
            )
            rows.append(record)
    return rows


def _evaluate_source_test_trajectory(
    path: Path,
    dlo: str,
    model: SourceModel,
    choice: CalibrationChoice,
    protocol: Protocol,
) -> list[dict[str, object]]:
    trajectory = load_trajectory(path)
    rows: list[dict[str, object]] = []
    for current in window_starts(protocol):
        observation = extract_endpoint_observation(trajectory, current, protocol)
        context = make_context(
            observation,
            model,
            choice.action_prototype_scale,
            protocol,
        )
        key = f"source-test/{dlo}/{path.name}/{current}"

        # Freeze every acquisition path and action before target outcomes are
        # sliced.  The diagnostic oracle may inspect only current-prefix sensor
        # values, never future internal-node outcomes.
        plans: list[FrozenPlan] = []
        for policy in protocol.policies:
            states, selected = acquisition_path(
                policy,
                context,
                key,
                choice.sensor_log_likelihood_scale,
                protocol,
            )
            for budget in protocol.budgets:
                plans.append(
                    _budget_plan(
                        policy,
                        budget,
                        states,
                        selected,
                        choice.regret_tolerance,
                        protocol,
                    )
                )

        full_target = extract_full_target_residual(
            trajectory, current, observation, protocol
        )
        target_task = task_residuals(full_target, protocol)
        for plan in plans:
            record = score_plan(plan, context, target_task, model)
            record.update(
                {
                    "dlo": dlo,
                    "trajectory": path.name,
                    "current_frame": current,
                }
            )
            rows.append(record)
    return rows


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


def _trajectory_aggregate(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], FloatArray, FloatArray]:
    by_trajectory: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dlo"]), str(row["trajectory"]))
        by_trajectory.setdefault(key, []).append(row)
    records: list[dict[str, object]] = []
    physical_values: list[float] = []
    fallback_values: list[float] = []
    for (dlo, trajectory), items in sorted(by_trajectory.items()):
        physical = np.asarray([float(item["physical_task_mse"]) for item in items])
        fallback = np.asarray([float(item["fallback_task_mse"]) for item in items])
        physical_values.extend(physical.tolist())
        fallback_values.extend(fallback.tolist())
        rmse = math.sqrt(float(np.mean(physical)))
        fallback_rmse = math.sqrt(float(np.mean(fallback)))
        nonfallback = [item for item in items if bool(item["nonfallback"])]
        records.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "decision_count": len(items),
                "task_rmse_mm": 1000.0 * rmse,
                "fallback_task_rmse_mm": 1000.0 * fallback_rmse,
                "relative_improvement": 1.0 - rmse / max(fallback_rmse, ATOL),
                "nonfallback_fraction": float(
                    np.mean([bool(item["nonfallback"]) for item in items])
                ),
                "certified_fraction": float(
                    np.mean([bool(item["certified"]) for item in items])
                ),
                "mean_sensor_count": float(
                    np.mean([int(item["sensor_count"]) for item in items])
                ),
                "mean_measurement_cost": float(
                    np.mean([float(item["measurement_cost"]) for item in items])
                ),
                "harmful_fraction": float(
                    np.mean([bool(item["harmful_vs_fallback"]) for item in items])
                ),
                "nonfallback_harmful_fraction": (
                    float(
                        np.mean(
                            [bool(item["harmful_vs_fallback"]) for item in nonfallback]
                        )
                    )
                    if nonfallback
                    else 0.0
                ),
            }
        )
    return (
        records,
        np.asarray(physical_values, dtype=np.float64),
        np.asarray(fallback_values, dtype=np.float64),
    )


def summarize_rows(
    rows: list[dict[str, object]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    trajectories, physical, fallback = _trajectory_aggregate(rows)
    improvements = np.asarray(
        [float(item["relative_improvement"]) for item in trajectories],
        dtype=np.float64,
    )
    nonfallback = [item for item in rows if bool(item["nonfallback"])]
    certificate_evaluable = [
        item for item in rows if bool(item["certificate_target_evaluable"])
    ]
    node_counts = {str(node): 0 for node in range(2, 10)}
    action_counts: dict[str, int] = {}
    for item in rows:
        for node in item["selected_internal_nodes"]:
            node_counts[str(node)] += 1
        label = str(item["action_label"])
        action_counts[label] = action_counts.get(label, 0) + 1
    interval = _bootstrap_interval(improvements, bootstrap_replicates, bootstrap_seed)
    return {
        "decision_count": len(rows),
        "trajectory_count": len(trajectories),
        "pooled_task_rmse_mm": 1000.0 * math.sqrt(float(np.mean(physical))),
        "pooled_fallback_task_rmse_mm": 1000.0 * math.sqrt(float(np.mean(fallback))),
        "mean_trajectory_improvement": float(np.mean(improvements)),
        "trajectory_bootstrap_95_interval": list(interval),
        "nonfallback_count": len(nonfallback),
        "nonfallback_fraction": float(
            np.mean([bool(item["nonfallback"]) for item in rows])
        ),
        "certified_fraction": float(
            np.mean([bool(item["certified"]) for item in rows])
        ),
        "mean_sensor_count": float(
            np.mean([int(item["sensor_count"]) for item in rows])
        ),
        "mean_measurement_cost": float(
            np.mean([float(item["measurement_cost"]) for item in rows])
        ),
        "harmful_fraction": float(
            np.mean([bool(item["harmful_vs_fallback"]) for item in rows])
        ),
        "nonfallback_harmful_fraction": (
            float(np.mean([bool(item["harmful_vs_fallback"]) for item in nonfallback]))
            if nonfallback
            else 0.0
        ),
        "mean_normalized_realized_regret": float(
            np.mean([float(item["normalized_realized_regret"]) for item in rows])
        ),
        "certified_mean_certificate_excess_regret": (
            float(
                np.mean(
                    [
                        float(item["certificate_excess_regret"])
                        for item in certificate_evaluable
                    ]
                )
            )
            if certificate_evaluable
            else 0.0
        ),
        "certified_target_violation_fraction": (
            float(
                np.mean(
                    [
                        bool(item["certificate_target_violation"])
                        for item in certificate_evaluable
                    ]
                )
            )
            if certificate_evaluable
            else 0.0
        ),
        "pointwise_action_accuracy": float(
            np.mean([bool(item["pointwise_action_correct"]) for item in rows])
        ),
        "nonfallback_effective_hypothesis_count": (
            float(
                np.mean(
                    [float(item["effective_hypothesis_count"]) for item in nonfallback]
                )
            )
            if nonfallback
            else 0.0
        ),
        "nonfallback_effective_class_count": (
            float(
                np.mean([float(item["effective_class_count"]) for item in nonfallback])
            )
            if nonfallback
            else 0.0
        ),
        "nonfallback_state_ambiguous_fraction": (
            float(
                np.mean(
                    [
                        float(item["effective_hypothesis_count"]) > 1.5
                        for item in nonfallback
                    ]
                )
            )
            if nonfallback
            else 0.0
        ),
        "selected_node_counts": node_counts,
        "action_counts": action_counts,
        "per_trajectory": trajectories,
    }


def aggregate_calibration_rows(
    rows: list[dict[str, object]],
    protocol: Protocol,
) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["sensor_log_likelihood_scale"]),
            float(row["action_prototype_scale"]),
            float(row["regret_tolerance"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for index, (key, selected) in enumerate(sorted(grouped.items())):
        sensor_scale, prototype_scale, tolerance = key
        summary = summarize_rows(
            selected,
            protocol.bootstrap_replicates,
            protocol.bootstrap_seed + index,
        )
        eligible = bool(
            int(summary["nonfallback_count"]) >= protocol.minimum_nonfallback_decisions
            and float(summary["nonfallback_harmful_fraction"])
            <= protocol.maximum_nonfallback_harmful_fraction + ATOL
            and float(summary["mean_trajectory_improvement"]) > 0.0
        )
        summary.update(
            {
                "sensor_log_likelihood_scale": sensor_scale,
                "action_prototype_scale": prototype_scale,
                "regret_tolerance": tolerance,
                "eligible": eligible,
            }
        )
        summaries.append(summary)
    return summaries


def select_calibration(
    summaries: list[dict[str, object]],
) -> CalibrationChoice:
    if not summaries:
        raise ValueError("empty calibration grid")
    eligible = [item for item in summaries if bool(item["eligible"])]
    pool = eligible if eligible else summaries

    def key(item: dict[str, object]) -> tuple[float, ...]:
        return (
            -float(item["mean_trajectory_improvement"]),
            -float(item["nonfallback_fraction"]),
            float(item["mean_measurement_cost"]),
            float(item["nonfallback_harmful_fraction"]),
            float(item["regret_tolerance"]),
            float(item["action_prototype_scale"]),
            float(item["sensor_log_likelihood_scale"]),
        )

    selected = min(pool, key=key)
    return CalibrationChoice(
        sensor_log_likelihood_scale=float(selected["sensor_log_likelihood_scale"]),
        action_prototype_scale=float(selected["action_prototype_scale"]),
        regret_tolerance=float(selected["regret_tolerance"]),
        gate_passed=bool(eligible),
        summary=selected,
    )


def aggregate_source_test_rows(
    rows: list[dict[str, object]],
    protocol: Protocol,
) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    counter = 0
    for policy in protocol.policies:
        policy_result: dict[str, object] = {}
        for budget in protocol.budgets:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and row["budget"] == budget
            ]
            policy_result[str(budget)] = summarize_rows(
                selected,
                protocol.bootstrap_replicates,
                protocol.bootstrap_seed + 1000 + counter,
            )
            counter += 1
        aggregate[policy] = policy_result
    return aggregate


def _paired_stratified_interval(
    records: list[tuple[str, float]],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_dlo: dict[str, FloatArray] = {}
    for dlo in DLOS:
        by_dlo[dlo] = np.asarray(
            [value for record_dlo, value in records if record_dlo == dlo],
            dtype=np.float64,
        )
        if len(by_dlo[dlo]) == 0:
            raise ValueError("paired bootstrap is missing a DLO stratum")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        values: list[float] = []
        for dlo in DLOS:
            stratum = by_dlo[dlo]
            sample = rng.integers(0, len(stratum), size=len(stratum))
            values.extend(stratum[sample].tolist())
        estimates[index] = float(np.mean(values))
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def paired_comparisons(
    aggregate: dict[str, object],
    protocol: Protocol,
) -> dict[str, object]:
    result: dict[str, object] = {}
    decision_all = aggregate["decision_regret"]
    assert isinstance(decision_all, dict)
    baselines = tuple(
        policy
        for policy in protocol.policies
        if policy not in {"decision_regret", "oracle_decision"}
    )
    counter = 0
    for baseline in baselines:
        baseline_all = aggregate[baseline]
        assert isinstance(baseline_all, dict)
        by_budget: dict[str, object] = {}
        for budget in protocol.budgets:
            decision = decision_all[str(budget)]
            other = baseline_all[str(budget)]
            assert isinstance(decision, dict)
            assert isinstance(other, dict)
            decision_records = {
                (str(item["dlo"]), str(item["trajectory"])): item
                for item in decision["per_trajectory"]
            }
            other_records = {
                (str(item["dlo"]), str(item["trajectory"])): item
                for item in other["per_trajectory"]
            }
            if set(decision_records) != set(other_records):
                raise ValueError("paired trajectory roster changed")
            improvement_differences: list[tuple[str, float]] = []
            cost_savings: list[tuple[str, float]] = []
            wins = ties = losses = 0
            for key in sorted(decision_records):
                dlo, _ = key
                difference = float(
                    decision_records[key]["relative_improvement"]
                ) - float(other_records[key]["relative_improvement"])
                improvement_differences.append((dlo, difference))
                saving = float(other_records[key]["mean_measurement_cost"]) - float(
                    decision_records[key]["mean_measurement_cost"]
                )
                cost_savings.append((dlo, saving))
                if difference > ATOL:
                    wins += 1
                elif difference < -ATOL:
                    losses += 1
                else:
                    ties += 1
            improvement_values = np.asarray(
                [value for _, value in improvement_differences], dtype=np.float64
            )
            cost_values = np.asarray(
                [value for _, value in cost_savings], dtype=np.float64
            )
            improvement_interval = _paired_stratified_interval(
                improvement_differences,
                protocol.bootstrap_replicates,
                protocol.bootstrap_seed + 2000 + counter,
            )
            cost_interval = _paired_stratified_interval(
                cost_savings,
                protocol.bootstrap_replicates,
                protocol.bootstrap_seed + 3000 + counter,
            )
            by_budget[str(budget)] = {
                "mean_trajectory_improvement_advantage": float(
                    np.mean(improvement_values)
                ),
                "improvement_advantage_bootstrap_95_interval": list(
                    improvement_interval
                ),
                "mean_measurement_cost_saving": float(np.mean(cost_values)),
                "measurement_cost_saving_bootstrap_95_interval": list(cost_interval),
                "trajectory_wins_ties_losses": [wins, ties, losses],
                "decision_nonfallback_fraction_advantage": float(
                    decision["nonfallback_fraction"]
                )
                - float(other["nonfallback_fraction"]),
                "decision_nonfallback_harmful_fraction_advantage": float(
                    other["nonfallback_harmful_fraction"]
                )
                - float(decision["nonfallback_harmful_fraction"]),
            }
            counter += 1
        result[baseline] = by_budget
    return result


def cost_rmse_frontiers(
    aggregate: dict[str, object],
    protocol: Protocol,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for policy in protocol.policies:
        policy_rows = aggregate[policy]
        assert isinstance(policy_rows, dict)
        points = []
        for budget in protocol.budgets:
            row = policy_rows[str(budget)]
            assert isinstance(row, dict)
            points.append(
                {
                    "budget": budget,
                    "mean_measurement_cost": float(row["mean_measurement_cost"]),
                    "pooled_task_rmse_mm": float(row["pooled_task_rmse_mm"]),
                    "mean_trajectory_improvement": float(
                        row["mean_trajectory_improvement"]
                    ),
                    "nonfallback_fraction": float(row["nonfallback_fraction"]),
                    "nonfallback_harmful_fraction": float(
                        row["nonfallback_harmful_fraction"]
                    ),
                }
            )
        frontier = []
        for point in points:
            dominated = any(
                other["mean_measurement_cost"] <= point["mean_measurement_cost"] + ATOL
                and other["pooled_task_rmse_mm"] <= point["pooled_task_rmse_mm"] + ATOL
                and (
                    other["mean_measurement_cost"]
                    < point["mean_measurement_cost"] - ATOL
                    or other["pooled_task_rmse_mm"]
                    < point["pooled_task_rmse_mm"] - ATOL
                )
                for other in points
            )
            if not dominated:
                frontier.append(point)
        result[policy] = frontier
    return result


def result_classification(
    calibration: CalibrationChoice,
    aggregate: dict[str, object],
    comparisons: dict[str, object],
    protocol: Protocol,
) -> dict[str, object]:
    budget = str(protocol.calibration_budget)
    decision = aggregate["decision_regret"][budget]
    assert isinstance(decision, dict)
    positive_intervals = []
    for baseline, values in comparisons.items():
        assert isinstance(values, dict)
        row = values[budget]
        assert isinstance(row, dict)
        interval = row["improvement_advantage_bootstrap_95_interval"]
        if float(interval[0]) > 0.0:
            positive_intervals.append(baseline)
    substantial_ambiguity = bool(
        float(decision["nonfallback_effective_hypothesis_count"]) > 2.0
        and float(decision["nonfallback_state_ambiguous_fraction"]) > 0.5
    )
    beats_state = bool(
        float(
            comparisons["state_variance"][budget][
                "mean_trajectory_improvement_advantage"
            ]
        )
        > 0.0
    )
    beats_entropy = bool(
        float(
            comparisons["posterior_entropy"][budget][
                "mean_trajectory_improvement_advantage"
            ]
        )
        > 0.0
    )
    if (
        calibration.gate_passed
        and substantial_ambiguity
        and positive_intervals
        and (beats_state or beats_entropy)
    ):
        label = "strong-positive-source-test-mechanism"
    elif calibration.gate_passed and substantial_ambiguity:
        label = "positive-source-test-mechanism"
    elif calibration.gate_passed:
        label = "calibration-passed-without-ambiguity-advantage"
    else:
        label = "negative-calibration-gate"
    return {
        "label": label,
        "calibration_gate_passed": calibration.gate_passed,
        "paired_positive_baselines_at_calibration_budget": positive_intervals,
        "substantial_state_ambiguity_when_acting": substantial_ambiguity,
        "beats_state_variance_at_calibration_budget": beats_state,
        "beats_posterior_entropy_at_calibration_budget": beats_entropy,
    }


def render_summary(
    result: dict[str, object],
    protocol: Protocol,
) -> str:
    selected = result["selected_calibration"]
    aggregate = result["aggregate"]
    comparisons = result["paired_comparisons"]
    classification = result["classification"]
    assert isinstance(selected, dict)
    assert isinstance(aggregate, dict)
    assert isinstance(comparisons, dict)
    assert isinstance(classification, dict)
    lines = [
        "# DEFORM decision-directed virtual sensing v2",
        "",
        f"Status: **{result['status']}**",
        "",
        "The official DLO4/DLO5 evaluation files were not mounted. All virtual "
        "measurements reveal already recorded current-prefix internal-node values.",
        "",
        "## Source calibration",
        "",
        f"- Gate passed: **{selected['gate_passed']}**",
        f"- Sensor log-likelihood scale: "
        f"{float(selected['sensor_log_likelihood_scale']):.4g}",
        f"- Action-prototype scale: {float(selected['action_prototype_scale']):.4g}",
        f"- Regret tolerance: {float(selected['regret_tolerance']):.4g}",
        f"- Classification: **{classification['label']}**",
        "",
        "## Source-test operating points",
        "",
        "| Policy | Budget | Task RMSE [mm] | Equal-trajectory improvement | "
        "Nonfallback | Mean measurements | Nonfallback harm | "
        "Effective hypotheses when acting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shown_budgets = tuple(value for value in protocol.budgets if value in {0, 2, 4, 8})
    for policy in protocol.policies:
        policy_result = aggregate[policy]
        assert isinstance(policy_result, dict)
        for budget in shown_budgets:
            row = policy_result[str(budget)]
            assert isinstance(row, dict)
            lines.append(
                f"| {policy} | {budget} | "
                f"{float(row['pooled_task_rmse_mm']):.3f} | "
                f"{100.0 * float(row['mean_trajectory_improvement']):.2f}% | "
                f"{100.0 * float(row['nonfallback_fraction']):.1f}% | "
                f"{float(row['mean_sensor_count']):.2f} | "
                f"{100.0 * float(row['nonfallback_harmful_fraction']):.2f}% | "
                f"{float(row['nonfallback_effective_hypothesis_count']):.2f} |"
            )
    lines.extend(
        (
            "",
            "## Paired comparison at the calibration budget",
            "",
            "| Baseline | Improvement advantage | 95% paired bootstrap interval | "
            "Measurement-cost saving | Wins/ties/losses |",
            "|---|---:|---:|---:|---:|",
        )
    )
    budget_key = str(protocol.calibration_budget)
    for baseline, values in comparisons.items():
        assert isinstance(values, dict)
        row = values[budget_key]
        assert isinstance(row, dict)
        interval = row["improvement_advantage_bootstrap_95_interval"]
        wins = row["trajectory_wins_ties_losses"]
        lines.append(
            f"| {baseline} | "
            f"{100.0 * float(row['mean_trajectory_improvement_advantage']):.2f} pp | "
            f"[{100.0 * float(interval[0]):.2f}, "
            f"{100.0 * float(interval[1]):.2f}] pp | "
            f"{float(row['mean_measurement_cost_saving']):.3f} | "
            f"{wins[0]}/{wins[1]}/{wins[2]} |"
        )
    lines.extend(
        (
            "",
            "## Boundary",
            "",
            str(result["claim_boundary"]),
            "",
        )
    )
    return "\n".join(lines)


def _compact_result(result: dict[str, object], protocol: Protocol) -> dict[str, object]:
    aggregate = result["aggregate"]
    comparisons = result["paired_comparisons"]
    assert isinstance(aggregate, dict)
    assert isinstance(comparisons, dict)
    budget = str(protocol.calibration_budget)
    operating_points = {
        policy: aggregate[policy][budget] for policy in protocol.policies
    }
    compact_points = {}
    for policy, row in operating_points.items():
        compact_points[policy] = {
            key: row[key]
            for key in (
                "pooled_task_rmse_mm",
                "mean_trajectory_improvement",
                "trajectory_bootstrap_95_interval",
                "nonfallback_fraction",
                "mean_sensor_count",
                "mean_measurement_cost",
                "nonfallback_harmful_fraction",
                "mean_normalized_realized_regret",
                "nonfallback_effective_hypothesis_count",
                "nonfallback_state_ambiguous_fraction",
            )
        }
    compact_comparisons = {
        baseline: values[budget] for baseline, values in comparisons.items()
    }
    return {
        "contract": result["contract"],
        "schema_version": result["schema_version"],
        "status": result["status"],
        "result_id": result["result_id"],
        "classification": result["classification"],
        "selected_calibration": result["selected_calibration"],
        "calibration_budget": protocol.calibration_budget,
        "operating_points": compact_points,
        "paired_comparisons": compact_comparisons,
        "accounting": result["accounting"],
        "claim_boundary": result["claim_boundary"],
    }


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = load_protocol(protocol_path)

    models: dict[str, SourceModel] = {}
    path_sets: dict[str, dict[str, tuple[Path, ...]]] = {}
    source_records: dict[str, object] = {}
    split_records: dict[str, object] = {}
    for dlo in DLOS:
        train_paths = trajectory_paths(dataset_root, dlo, "train")
        names = tuple(path.name for path in train_paths)
        split = split_names(names, dlo, protocol)
        fit_base, fit_sensors, fit_residuals, fit_manifest = build_arrays(
            train_paths, split["fit"], protocol
        )
        model = fit_source_model(
            fit_base,
            fit_sensors,
            fit_residuals,
            protocol,
        )
        models[dlo] = model
        path_sets[dlo] = {
            name: tuple(path for path in train_paths if path.name in set(values))
            for name, values in split.items()
        }
        for name, expected in (
            ("fit", protocol.fit_count),
            ("calibration", protocol.calibration_count),
            ("source_test", protocol.source_test_count),
        ):
            if len(path_sets[dlo][name]) != expected:
                raise ValueError(f"{dlo}: incomplete {name} roster")
        source_records[dlo] = {
            "fit_trajectory_count": protocol.fit_count,
            "fit_window_count": len(fit_base),
            "calibration_trajectory_count": protocol.calibration_count,
            "source_test_trajectory_count": protocol.source_test_count,
            "response_class_count": int(len(model.class_counts)),
            "response_class_counts": model.class_counts.tolist(),
            "loss_floor": model.loss_floor,
            "action_competition": action_competition_summary(model),
            "fit_manifest": fit_manifest,
        }
        split_records[dlo] = {name: list(values) for name, values in split.items()}

    calibration_rows: list[dict[str, object]] = []
    for dlo in DLOS:
        for path in path_sets[dlo]["calibration"]:
            calibration_rows.extend(
                _evaluate_calibration_trajectory(
                    path,
                    dlo,
                    models[dlo],
                    protocol,
                )
            )
    calibration_grid = aggregate_calibration_rows(calibration_rows, protocol)
    choice = select_calibration(calibration_grid)

    source_test_rows: list[dict[str, object]] = []
    for dlo in DLOS:
        for path in path_sets[dlo]["source_test"]:
            source_test_rows.extend(
                _evaluate_source_test_trajectory(
                    path,
                    dlo,
                    models[dlo],
                    choice,
                    protocol,
                )
            )
    aggregate = aggregate_source_test_rows(source_test_rows, protocol)
    comparisons = paired_comparisons(aggregate, protocol)
    frontiers = cost_rmse_frontiers(aggregate, protocol)
    classification = result_classification(choice, aggregate, comparisons, protocol)
    status = (
        "source-test-only-exploratory-result"
        if choice.gate_passed
        else "source-test-only-negative-calibration-gate"
    )
    result: dict[str, object] = {
        "contract": CONTRACT,
        "schema_version": 2,
        "status": status,
        "protocol_sha256": _sha256_file(protocol_path),
        "source_revision": args.source_revision,
        "dataset_root_name": dataset_root.name,
        "source_split": split_records,
        "source": source_records,
        "selected_calibration": {
            "sensor_log_likelihood_scale": (choice.sensor_log_likelihood_scale),
            "action_prototype_scale": choice.action_prototype_scale,
            "regret_tolerance": choice.regret_tolerance,
            "gate_passed": choice.gate_passed,
            "selection_objective": protocol.selection_objective,
            "calibration_summary": choice.summary,
        },
        "calibration_grid": calibration_grid,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "cost_rmse_frontiers": frontiers,
        "classification": classification,
        "accounting": {
            "fit_trajectories": 2 * protocol.fit_count,
            "fit_windows": 2 * protocol.fit_count * len(window_starts(protocol)),
            "calibration_trajectories": 2 * protocol.calibration_count,
            "calibration_windows": 2
            * protocol.calibration_count
            * len(window_starts(protocol)),
            "source_test_trajectories": 2 * protocol.source_test_count,
            "source_test_decision_windows": 2
            * protocol.source_test_count
            * len(window_starts(protocol)),
            "calibration_base_configurations": len(
                protocol.sensor_log_likelihood_scales
            )
            * len(protocol.action_prototype_scales),
            "calibration_grid_configurations": len(calibration_grid),
            "calibration_case_rows": len(calibration_rows),
            "policies": len(protocol.policies),
            "budgets": len(protocol.budgets),
            "source_test_case_rows": len(source_test_rows),
            "official_evaluation_files_opened": False,
            "future_internal_nodes_used_before_action_selection": False,
            "new_data_collected": False,
        },
        "claim_boundary": _read_json(protocol_path)["claim_boundary"],
    }
    result["result_id"] = _canonical_sha256(result)
    _write_json(output_dir / "result.json", result)
    _write_json(output_dir / "compact_result.json", _compact_result(result, protocol))
    _write_json(output_dir / "calibration_grid.json", calibration_grid)
    with (output_dir / "source_test_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in source_test_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "calibration_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in calibration_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "SUMMARY.md").write_text(
        render_summary(result, protocol), encoding="utf-8"
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
