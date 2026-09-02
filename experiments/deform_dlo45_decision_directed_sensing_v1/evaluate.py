"""Decision-directed virtual sensing on existing DEFORM DLO4/DLO5 data.

The experiment masks eight internal-node readouts from an endpoint-only physical
forecast.  It then reveals recorded prefix readouts one at a time.  Candidate
readouts are selected either by expected certified decision regret, state
variance, query variance, fixed order, random order, or a target-value oracle.
No future internal node is exposed before every action is selected.
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

CONTRACT: Final = "deform-dlo45-decision-directed-sensing-v1"
EXPECTED_POLICIES: Final = (
    "decision_regret",
    "state_variance",
    "query_variance",
    "center_out",
    "random",
    "oracle_decision",
)
ATOL: Final = 1e-12


@dataclass(frozen=True)
class Protocol:
    first_current_frame: int
    horizon_frames: int
    stride_frames: int
    fit_count: int
    calibration_count: int
    source_test_count: int
    split_domain: str
    support_neighbors: int
    response_clusters: int
    action_scales: FloatArray
    temperature_scale: float
    sensor_log_likelihood_scale: float
    regret_tolerance: float
    state_projection_dimension: int
    kmeans_iterations: int
    policies: tuple[str, ...]
    budgets: tuple[int, ...]
    maximum_measurements: int
    center_out_nodes: tuple[int, ...]
    random_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class SourceModel:
    base_features: FloatArray
    sensor_features: FloatArray
    residuals: FloatArray
    class_labels: IntArray
    state_representation: FloatArray
    query_representation: FloatArray
    base_mean: FloatArray
    base_scale: FloatArray
    sensor_mean: FloatArray
    sensor_scale: FloatArray
    loss_floor: float


class Observation(NamedTuple):
    base_feature: FloatArray
    sensor_features: FloatArray
    residual: FloatArray
    length_scale: float


@dataclass(frozen=True)
class CaseContext:
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
    target_residual: FloatArray
    length_scale: float


@dataclass(frozen=True)
class DecisionState:
    posterior_weights: FloatArray
    certificate: QueryDecisionCertificateV1
    certified: bool
    action_index: int
    state_variance: float
    query_variance: float
    effective_hypothesis_count: float
    posterior_entropy: float


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


def _float_array(value: object) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def load_protocol(path: Path) -> Protocol:
    raw = _read_json(path)
    if raw.get("contract") != CONTRACT or raw.get("schema_version") != 1:
        raise ValueError("unsupported decision-directed sensing protocol")
    data = raw.get("data")
    windows = raw.get("windows")
    split = raw.get("source_split")
    model = raw.get("model")
    sensing = raw.get("sensing")
    evaluation = raw.get("evaluation")
    if not all(
        isinstance(value, dict)
        for value in (data, windows, split, model, sensing, evaluation)
    ):
        raise ValueError("protocol sections must be objects")
    assert isinstance(data, dict)
    assert isinstance(windows, dict)
    assert isinstance(split, dict)
    assert isinstance(model, dict)
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
        or evaluation.get("primary_stage") != "source-test-only-pilot"
        or evaluation.get("evaluation_split_opened") is not False
        or evaluation.get("target_tuning") is not False
        or evaluation.get("new_data_collection") is not False
    ):
        raise ValueError("frozen data or information boundary changed")
    protocol = Protocol(
        first_current_frame=int(windows["first_current_frame"]),
        horizon_frames=int(windows["horizon_frames"]),
        stride_frames=int(windows["stride_frames"]),
        fit_count=int(split["fit_count"]),
        calibration_count=int(split["calibration_count"]),
        source_test_count=int(split["source_test_count"]),
        split_domain=str(split["domain_separator"]),
        support_neighbors=int(model["support_neighbors"]),
        response_clusters=int(model["response_clusters"]),
        action_scales=_float_array(model["action_scales"]),
        temperature_scale=float(model["temperature_scale"]),
        sensor_log_likelihood_scale=float(
            model["sensor_log_likelihood_scale"]
        ),
        regret_tolerance=float(model["regret_tolerance"]),
        state_projection_dimension=int(model["state_projection_dimension"]),
        kmeans_iterations=int(model["kmeans_iterations"]),
        policies=tuple(str(value) for value in sensing["policies"]),
        budgets=tuple(int(value) for value in sensing["measurement_budgets"]),
        maximum_measurements=int(sensing["maximum_measurements"]),
        center_out_nodes=tuple(
            int(value) for value in sensing["center_out_internal_nodes"]
        ),
        random_seed=int(sensing["random_seed"]),
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )
    if (
        protocol.first_current_frame < 1
        or protocol.horizon_frames < 1
        or protocol.stride_frames < 1
        or protocol.fit_count + protocol.calibration_count
        + protocol.source_test_count
        != 56
        or protocol.support_neighbors < 2
        or protocol.response_clusters < 2
        or protocol.action_scales.shape != (3,)
        or not np.array_equal(protocol.action_scales, [0.0, 0.5, 1.0])
        or protocol.temperature_scale <= 0.0
        or protocol.sensor_log_likelihood_scale <= 0.0
        or protocol.regret_tolerance < 0.0
        or protocol.state_projection_dimension < 1
        or protocol.policies != EXPECTED_POLICIES
        or protocol.maximum_measurements != 8
        or protocol.budgets[0] != 0
        or protocol.budgets[-1] != protocol.maximum_measurements
        or tuple(sorted(protocol.center_out_nodes)) != tuple(range(2, 10))
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
    starts = tuple(
        range(
            protocol.first_current_frame,
            stop,
            protocol.stride_frames,
        )
    )
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
    previous = trajectory[current - 1]
    present = trajectory[current]
    future = trajectory[
        current + 1 : current + 1 + protocol.horizon_frames
    ]
    previous_left, previous_right = _anchor_means(previous)
    present_left, present_right = _anchor_means(present)
    future_left = np.mean(future[:, list(ACTION_LEFT), :], axis=1)
    future_right = np.mean(future[:, list(ACTION_RIGHT), :], axis=1)
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
    sampled_actions = np.concatenate(
        (
            future_left[sample_indices] - present_left,
            future_right[sample_indices] - present_right,
        ),
        axis=1,
    ) / length_scale
    sampled_lengths = np.linalg.norm(
        future_right[sample_indices] - future_left[sample_indices],
        axis=1,
    ) / length_scale
    midpoint_velocity = 0.5 * (
        present_left
        + present_right
        - previous_left
        - previous_right
    ) / length_scale
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
    sensor_features = np.concatenate(
        (present_shape, shape_velocity),
        axis=1,
    )

    count = NODE_COUNT - 4
    weights = np.linspace(
        1.0 / (count + 1),
        count / (count + 1),
        count,
        dtype=np.float64,
    )
    baseline = (
        (1.0 - weights[None, :, None]) * future_left[:, None, :]
        + weights[None, :, None] * future_right[:, None, :]
    )
    truth = future[:, INTERNAL, :]
    residual = ((truth - baseline) / length_scale).reshape(-1)
    return Observation(
        base_feature=np.asarray(base_feature, dtype=np.float64),
        sensor_features=np.asarray(sensor_features, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        length_scale=length_scale,
    )


def response_signature(
    residuals: FloatArray,
    protocol: Protocol,
) -> FloatArray:
    shaped = residuals.reshape(
        residuals.shape[0],
        protocol.horizon_frames,
        NODE_COUNT - 4,
        3,
    )
    mean_all = np.mean(shaped, axis=(1, 2))
    final_mean = np.mean(shaped[:, -1], axis=1)
    rms = np.sqrt(np.mean(np.square(shaped), axis=(1, 2, 3)))[:, None]
    return np.concatenate((mean_all, final_mean, rms), axis=1)


def query_representation(
    residuals: FloatArray,
    protocol: Protocol,
) -> FloatArray:
    shaped = residuals.reshape(
        residuals.shape[0],
        protocol.horizon_frames,
        NODE_COUNT - 4,
        3,
    )
    return np.mean(shaped, axis=(1, 2))


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
            observation = extract_endpoint_observation(
                trajectory,
                current,
                protocol,
            )
            base_features.append(observation.base_feature)
            sensor_features.append(observation.sensor_features)
            residuals.append(observation.residual)
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


def fit_source_model(
    base_features: FloatArray,
    sensor_features: FloatArray,
    residuals: FloatArray,
    protocol: Protocol,
) -> SourceModel:
    base_mean = np.mean(base_features, axis=0)
    base_scale = np.maximum(np.std(base_features, axis=0), 1e-9)
    sensor_mean = np.mean(sensor_features, axis=0)
    sensor_scale = np.maximum(np.std(sensor_features, axis=0), 1e-9)
    centered = residuals - np.mean(residuals, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    dimension = min(protocol.state_projection_dimension, right.shape[0])
    state_representation = centered @ right[:dimension].T
    classes = deterministic_kmeans(
        response_signature(residuals, protocol),
        protocol.response_clusters,
        protocol.kmeans_iterations,
    )
    fallback_losses = np.mean(np.square(residuals), axis=1)
    loss_floor = max(float(np.quantile(fallback_losses, 0.05)) * 0.1, 1e-12)
    return SourceModel(
        base_features=base_features,
        sensor_features=sensor_features,
        residuals=residuals,
        class_labels=classes,
        state_representation=state_representation,
        query_representation=query_representation(residuals, protocol),
        base_mean=base_mean,
        base_scale=base_scale,
        sensor_mean=sensor_mean,
        sensor_scale=sensor_scale,
        loss_floor=loss_floor,
    )


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("posterior weights are degenerate")
    return weights / total


def _local_classes(labels: IntArray) -> IntArray:
    unique = np.unique(labels)
    remap = {int(value): index for index, value in enumerate(unique.tolist())}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int64)


def _class_masses(weights: FloatArray, classes: IntArray) -> FloatArray:
    return np.bincount(
        classes,
        weights=weights,
        minlength=int(np.max(classes)) + 1,
    ).astype(np.float64)


def _jeffrey_weights(weights: FloatArray, classes: IntArray) -> FloatArray:
    masses = _class_masses(weights, classes)
    sizes = np.bincount(classes, minlength=len(masses)).astype(np.float64)
    return masses[classes] / sizes[classes]


def make_context(
    observation: Observation,
    model: SourceModel,
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
    base_logits = -(
        selected_distance - float(np.min(selected_distance))
    ) / bandwidth
    sensor_pool = (
        model.sensor_features - model.sensor_mean[None, :, :]
    ) / model.sensor_scale[None, :, :]
    target_sensor = (
        observation.sensor_features - model.sensor_mean
    ) / model.sensor_scale
    classes = _local_classes(model.class_labels[support])
    base_weights = _softmax(base_logits)
    jeffrey = _jeffrey_weights(base_weights, classes)
    residuals = model.residuals[support]
    correction = np.einsum("i,id->d", jeffrey, residuals)
    actions = protocol.action_scales[:, None] * correction[None, :]
    raw_losses = np.mean(
        np.square(residuals[:, None, :] - actions[None, :, :]),
        axis=2,
    )
    fallback_losses = np.mean(np.square(residuals), axis=1)
    relative_losses = raw_losses / (
        fallback_losses[:, None] + model.loss_floor
    )
    return CaseContext(
        support_indices=np.asarray(support, dtype=np.int64),
        base_logits=np.asarray(base_logits, dtype=np.float64),
        support_sensor_features=np.asarray(sensor_pool[support], dtype=np.float64),
        target_sensor_features=np.asarray(target_sensor, dtype=np.float64),
        support_residuals=np.asarray(residuals, dtype=np.float64),
        support_classes=classes,
        support_state_representation=np.asarray(
            model.state_representation[support],
            dtype=np.float64,
        ),
        support_query_representation=np.asarray(
            model.query_representation[support],
            dtype=np.float64,
        ),
        fixed_actions=np.asarray(actions, dtype=np.float64),
        relative_losses=np.asarray(relative_losses, dtype=np.float64),
        target_residual=observation.residual,
        length_scale=observation.length_scale,
    )


def posterior_weights(
    context: CaseContext,
    observations: dict[int, FloatArray],
    protocol: Protocol,
) -> FloatArray:
    logits = context.base_logits.copy()
    for sensor_index, value in observations.items():
        difference = context.support_sensor_features[:, sensor_index, :] - value
        logits -= protocol.sensor_log_likelihood_scale * np.mean(
            np.square(difference),
            axis=1,
        )
    return _softmax(logits)


def _weighted_variance(values: FloatArray, weights: FloatArray) -> float:
    mean = np.einsum("i,id->d", weights, values)
    second = np.einsum("i,id->d", weights, np.square(values))
    variance = np.maximum(second - np.square(mean), 0.0)
    return float(np.mean(variance))


def decision_state(
    context: CaseContext,
    observations: dict[int, FloatArray],
    protocol: Protocol,
) -> DecisionState:
    weights = posterior_weights(context, observations, protocol)
    quotient = _class_masses(weights, context.support_classes)
    prior = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    certificate = query_decision_certificate(
        prior,
        quotient,
        context.support_classes,
        context.relative_losses,
        regret_tolerance=protocol.regret_tolerance,
    )
    certified = bool(
        certificate.minimax_worst_case_regret
        <= protocol.regret_tolerance + ATOL
    )
    action = certificate.minimax_action_index if certified else 0
    positive = weights[weights > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return DecisionState(
        posterior_weights=weights,
        certificate=certificate,
        certified=certified,
        action_index=action,
        state_variance=_weighted_variance(
            context.support_state_representation,
            weights,
        ),
        query_variance=_weighted_variance(
            context.support_query_representation,
            weights,
        ),
        effective_hypothesis_count=float(1.0 / np.sum(np.square(weights))),
        posterior_entropy=entropy,
    )


def expected_candidate_metric(
    context: CaseContext,
    observations: dict[int, FloatArray],
    candidate: int,
    metric: str,
    protocol: Protocol,
) -> float:
    current = posterior_weights(context, observations, protocol)
    total = 0.0
    for outcome_index, probability in enumerate(current):
        if probability <= 0.0:
            continue
        hypothetical = dict(observations)
        hypothetical[candidate] = context.support_sensor_features[
            outcome_index,
            candidate,
        ]
        state = decision_state(context, hypothetical, protocol)
        if metric == "decision_regret":
            value = state.certificate.minimax_worst_case_regret
        elif metric == "state_variance":
            value = state.state_variance
        elif metric == "query_variance":
            value = state.query_variance
        else:
            raise ValueError(f"unsupported expected metric: {metric}")
        total += float(probability) * float(value)
    return total


def _stable_random_order(key: str, protocol: Protocol) -> tuple[int, ...]:
    seed_bytes = hashlib.sha256(
        f"{protocol.random_seed}\0{key}".encode()
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.permutation(8))


def choose_candidate(
    policy: str,
    context: CaseContext,
    observations: dict[int, FloatArray],
    remaining: tuple[int, ...],
    key: str,
    protocol: Protocol,
) -> int:
    if policy in {
        "decision_regret",
        "state_variance",
        "query_variance",
    }:
        scores = [
            (
                expected_candidate_metric(
                    context,
                    observations,
                    candidate,
                    policy,
                    protocol,
                ),
                candidate,
            )
            for candidate in remaining
        ]
        return min(scores)[1]
    if policy == "oracle_decision":
        scores = []
        for candidate in remaining:
            hypothetical = dict(observations)
            hypothetical[candidate] = context.target_sensor_features[candidate]
            state = decision_state(context, hypothetical, protocol)
            scores.append(
                (
                    state.certificate.minimax_worst_case_regret,
                    candidate,
                )
            )
        return min(scores)[1]
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
    protocol: Protocol,
) -> tuple[list[DecisionState], list[int]]:
    observations: dict[int, FloatArray] = {}
    states = [decision_state(context, observations, protocol)]
    selected: list[int] = []
    while (
        len(selected) < protocol.maximum_measurements
        and not states[-1].certified
    ):
        remaining = tuple(
            index for index in range(8) if index not in observations
        )
        candidate = choose_candidate(
            policy,
            context,
            observations,
            remaining,
            key,
            protocol,
        )
        observations[candidate] = context.target_sensor_features[candidate]
        selected.append(candidate)
        states.append(decision_state(context, observations, protocol))
    return states, selected


def _budget_record(
    policy: str,
    budget: int,
    states: list[DecisionState],
    selected: list[int],
    context: CaseContext,
) -> dict[str, object]:
    allowed = min(budget, len(states) - 1)
    certified_step = next(
        (index for index in range(allowed + 1) if states[index].certified),
        None,
    )
    if certified_step is None:
        action = 0
        sensor_count = allowed
        state = states[allowed]
        certified = False
    else:
        action = states[certified_step].action_index
        sensor_count = certified_step
        state = states[certified_step]
        certified = True
    normalized_mse = np.mean(
        np.square(
            context.target_residual[None, :]
            - context.fixed_actions,
        ),
        axis=1,
    )
    physical_mse = normalized_mse * context.length_scale**2
    fallback = float(physical_mse[0])
    selected_mse = float(physical_mse[action])
    best = float(np.min(normalized_mse))
    denominator = max(float(normalized_mse[0]), ATOL)
    return {
        "policy": policy,
        "budget": budget,
        "certified": certified,
        "action_index": action,
        "nonfallback": action != 0,
        "sensor_count": sensor_count,
        "selected_internal_nodes": [
            selected[index] + 2 for index in range(sensor_count)
        ],
        "physical_mse": selected_mse,
        "fallback_mse": fallback,
        "harmful_vs_fallback": bool(selected_mse > fallback + ATOL),
        "normalized_realized_regret": (
            float(normalized_mse[action]) - best
        ) / denominator,
        "certificate_worst_case_regret": (
            state.certificate.minimax_worst_case_regret
        ),
        "state_variance": state.state_variance,
        "query_variance": state.query_variance,
        "effective_hypothesis_count": state.effective_hypothesis_count,
        "posterior_entropy": state.posterior_entropy,
    }


def evaluate_trajectory(
    path: Path,
    dlo: str,
    model: SourceModel,
    protocol: Protocol,
) -> list[dict[str, object]]:
    trajectory = load_trajectory(path)
    rows: list[dict[str, object]] = []
    for current in window_starts(protocol):
        observation = extract_endpoint_observation(
            trajectory,
            current,
            protocol,
        )
        context = make_context(observation, model, protocol)
        key = f"{dlo}/{path.name}/{current}"
        for policy in protocol.policies:
            states, selected = acquisition_path(
                policy,
                context,
                key,
                protocol,
            )
            for budget in protocol.budgets:
                record = _budget_record(
                    policy,
                    budget,
                    states,
                    selected,
                    context,
                )
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


def aggregate_rows(
    rows: list[dict[str, object]],
    protocol: Protocol,
) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for policy in protocol.policies:
        policy_result: dict[str, object] = {}
        for budget in protocol.budgets:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and row["budget"] == budget
            ]
            by_trajectory: dict[tuple[str, str], list[dict[str, object]]] = {}
            for row in selected:
                key = (str(row["dlo"]), str(row["trajectory"]))
                by_trajectory.setdefault(key, []).append(row)
            trajectory_records: list[dict[str, object]] = []
            improvements: list[float] = []
            for (dlo, trajectory), items in sorted(by_trajectory.items()):
                physical = np.asarray(
                    [float(item["physical_mse"]) for item in items]
                )
                fallback = np.asarray(
                    [float(item["fallback_mse"]) for item in items]
                )
                rmse = math.sqrt(float(np.mean(physical)))
                fallback_rmse = math.sqrt(float(np.mean(fallback)))
                improvement = 1.0 - rmse / max(fallback_rmse, ATOL)
                improvements.append(improvement)
                trajectory_records.append(
                    {
                        "dlo": dlo,
                        "trajectory": trajectory,
                        "decision_count": len(items),
                        "rmse_mm": 1000.0 * rmse,
                        "fallback_rmse_mm": 1000.0 * fallback_rmse,
                        "relative_improvement": improvement,
                        "nonfallback_fraction": float(
                            np.mean([bool(item["nonfallback"]) for item in items])
                        ),
                        "certified_fraction": float(
                            np.mean([bool(item["certified"]) for item in items])
                        ),
                        "mean_sensor_count": float(
                            np.mean([int(item["sensor_count"]) for item in items])
                        ),
                        "harmful_fraction": float(
                            np.mean(
                                [
                                    bool(item["harmful_vs_fallback"])
                                    for item in items
                                ]
                            )
                        ),
                    }
                )
            physical_all = np.asarray(
                [float(item["physical_mse"]) for item in selected]
            )
            fallback_all = np.asarray(
                [float(item["fallback_mse"]) for item in selected]
            )
            nonfallback = [item for item in selected if item["nonfallback"]]
            improvement_array = np.asarray(improvements, dtype=np.float64)
            interval = _bootstrap_interval(
                improvement_array,
                protocol.bootstrap_replicates,
                protocol.bootstrap_seed + budget,
            )
            node_counts = {str(node): 0 for node in range(2, 10)}
            for item in selected:
                for node in item["selected_internal_nodes"]:
                    node_counts[str(node)] += 1
            policy_result[str(budget)] = {
                "decision_count": len(selected),
                "trajectory_count": len(trajectory_records),
                "pooled_rmse_mm": 1000.0
                * math.sqrt(float(np.mean(physical_all))),
                "pooled_fallback_rmse_mm": 1000.0
                * math.sqrt(float(np.mean(fallback_all))),
                "mean_trajectory_improvement": float(
                    np.mean(improvement_array)
                ),
                "trajectory_bootstrap_95_interval": list(interval),
                "nonfallback_fraction": float(
                    np.mean([bool(item["nonfallback"]) for item in selected])
                ),
                "certified_fraction": float(
                    np.mean([bool(item["certified"]) for item in selected])
                ),
                "mean_sensor_count": float(
                    np.mean([int(item["sensor_count"]) for item in selected])
                ),
                "harmful_fraction": float(
                    np.mean(
                        [
                            bool(item["harmful_vs_fallback"])
                            for item in selected
                        ]
                    )
                ),
                "nonfallback_harmful_fraction": (
                    float(
                        np.mean(
                            [
                                bool(item["harmful_vs_fallback"])
                                for item in nonfallback
                            ]
                        )
                    )
                    if nonfallback
                    else 0.0
                ),
                "mean_normalized_realized_regret": float(
                    np.mean(
                        [
                            float(item["normalized_realized_regret"])
                            for item in selected
                        ]
                    )
                ),
                "nonfallback_effective_hypothesis_count": (
                    float(
                        np.mean(
                            [
                                float(item["effective_hypothesis_count"])
                                for item in nonfallback
                            ]
                        )
                    )
                    if nonfallback
                    else 0.0
                ),
                "nonfallback_state_variance": (
                    float(
                        np.mean(
                            [float(item["state_variance"]) for item in nonfallback]
                        )
                    )
                    if nonfallback
                    else 0.0
                ),
                "selected_node_counts": node_counts,
                "per_trajectory": trajectory_records,
            }
        aggregate[policy] = policy_result
    return aggregate


def render_summary(result: dict[str, object], protocol: Protocol) -> str:
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    lines = [
        "# DEFORM decision-directed virtual sensing pilot",
        "",
        "Status: **source-test-only exploratory result**",
        "",
        "All measurements are masked/revealed from existing recorded prefix "
        "nodes. Official DLO4/DLO5 evaluation files were not opened.",
        "",
        "| Policy | Budget | RMSE [mm] | Improvement | Nonfallback | "
        "Mean measurements | Harm | Effective hypotheses when acting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shown_budgets = tuple(
        value for value in protocol.budgets if value in {0, 2, 4, 8}
    )
    for policy in protocol.policies:
        policy_result = aggregate[policy]
        assert isinstance(policy_result, dict)
        for budget in shown_budgets:
            row = policy_result[str(budget)]
            assert isinstance(row, dict)
            lines.append(
                f"| {policy} | {budget} | "
                f"{float(row['pooled_rmse_mm']):.3f} | "
                f"{100.0 * float(row['mean_trajectory_improvement']):.2f}% | "
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
            str(result["claim_boundary"]),
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
    all_rows: list[dict[str, object]] = []
    source_records: dict[str, object] = {}
    split_records: dict[str, object] = {}
    for dlo in DLOS:
        train_paths = trajectory_paths(dataset_root, dlo, "train")
        names = tuple(path.name for path in train_paths)
        split = split_names(names, dlo, protocol)
        model_names = split["fit"] + split["calibration"]
        base, sensors, residuals, manifest = build_arrays(
            train_paths,
            model_names,
            protocol,
        )
        model = fit_source_model(base, sensors, residuals, protocol)
        test_paths = tuple(
            path for path in train_paths if path.name in set(split["source_test"])
        )
        if len(test_paths) != protocol.source_test_count:
            raise ValueError("source-test roster is incomplete")
        for path in test_paths:
            all_rows.extend(evaluate_trajectory(path, dlo, model, protocol))
        source_records[dlo] = {
            "model_trajectory_count": len(model_names),
            "model_window_count": len(base),
            "source_test_trajectory_count": len(test_paths),
            "source_test_window_count": len(test_paths)
            * len(window_starts(protocol)),
            "response_class_count": int(len(np.unique(model.class_labels))),
            "loss_floor": model.loss_floor,
            "model_manifest": manifest,
        }
        split_records[dlo] = {
            name: list(values) for name, values in split.items()
        }
    aggregate = aggregate_rows(all_rows, protocol)
    result: dict[str, object] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": "source-test-only-exploratory-result",
        "protocol_sha256": _sha256_file(protocol_path),
        "source_revision": args.source_revision,
        "dataset_root_name": dataset_root.name,
        "source_split": split_records,
        "source": source_records,
        "accounting": {
            "source_test_trajectories": 2 * protocol.source_test_count,
            "source_test_decision_windows": 2
            * protocol.source_test_count
            * len(window_starts(protocol)),
            "policies": len(protocol.policies),
            "budgets": len(protocol.budgets),
            "case_rows": len(all_rows),
            "official_evaluation_files_opened": False,
            "future_internal_nodes_used_before_action_selection": False,
            "new_data_collected": False,
        },
        "aggregate": aggregate,
        "claim_boundary": _read_json(protocol_path)["claim_boundary"],
    }
    result["result_id"] = _canonical_sha256(result)
    _write_json(output_dir / "result.json", result)
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )
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
