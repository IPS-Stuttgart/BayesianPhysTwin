"""Shared contracts and observation construction for DLO4/DLO5 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONTRACT: Final = "deform-dlo45-decision-identifiability-v1"
REQUEST_CONTRACT: Final = "deform-dlo45-decision-identifiability-request-v1"
DLOS: Final = ("DLO4", "DLO5")
FRAME_COUNT: Final = 500
NODE_COUNT: Final = 12
INTERNAL: Final = slice(2, -2)
ACTION_LEFT: Final = (0, 1)
ACTION_RIGHT: Final = (-2, -1)
ATOL: Final = 1e-12


@dataclass(frozen=True)
class Protocol:
    prefix_frames: int
    horizon_frames: int
    stride_frames: int
    action_scales: tuple[float, ...]
    neighbor_grid: tuple[int, ...]
    cluster_grid: tuple[int, ...]
    temperature_grid: tuple[float, ...]
    regret_tolerance_grid: tuple[float, ...]
    kmeans_iterations: int
    source_fit_count: int
    source_calibration_count: int
    source_test_count: int
    partition_domain: str
    source_gate_mean_ratio: float
    source_gate_worst_trajectory_ratio: float
    source_gate_minimum_nonfallback_fraction: float
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class Model:
    features: FloatArray
    residuals: FloatArray
    class_labels: IntArray
    feature_mean: FloatArray
    feature_scale: FloatArray
    loss_floor: float
    neighbors: int
    temperature_scale: float
    regret_tolerance: float
    action_scales: FloatArray


class Observation(NamedTuple):
    feature: FloatArray
    baseline: FloatArray
    length_scale: float


class Decision(NamedTuple):
    certificate_action: int
    jeffrey_action: int
    kernel_action: int
    map_action: int
    correction: FloatArray
    worst_case_regret: FloatArray
    minimax_regret: float
    robust_mask: npt.NDArray[np.bool_]
    tolerance_mask: npt.NDArray[np.bool_]
    ambiguity_width: float
    unsupported_specificity_nats: float
    neighbor_count: int
    quotient_class_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tuple_of(value: object, cast: type) -> tuple:
    if not isinstance(value, list):
        raise ValueError("protocol grids must be JSON arrays")
    return tuple(cast(item) for item in value)


def load_protocol(path: Path) -> Protocol:
    value = read_json(path)
    if value.get("contract") != CONTRACT or value.get("schema_version") != 1:
        raise ValueError("unsupported decision-identifiability protocol")
    data = value.get("data")
    windows = value.get("windows")
    decision = value.get("decision")
    split = value.get("source_split")
    gate = value.get("source_gate")
    bootstrap = value.get("bootstrap")
    if not all(
        isinstance(item, dict)
        for item in (data, windows, decision, split, gate, bootstrap)
    ):
        raise ValueError("protocol sections must be JSON objects")
    assert isinstance(data, dict)
    assert isinstance(windows, dict)
    assert isinstance(decision, dict)
    assert isinstance(split, dict)
    assert isinstance(gate, dict)
    assert isinstance(bootstrap, dict)
    if (
        tuple(data.get("dlos", ())) != DLOS
        or int(data.get("frame_count", -1)) != FRAME_COUNT
        or int(data.get("node_count", -1)) != NODE_COUNT
        or int(data.get("train_trajectory_count", -1)) != 56
        or int(data.get("eval_trajectory_count", -1)) != 14
        or tuple(data.get("known_action_nodes", ())) != (0, 1, -2, -1)
        or decision.get("loss") != "fallback-normalized-trajectory-mse"
        or decision.get("fallback_action_index") != 0
        or decision.get("target_tuning") is not False
        or decision.get("target_retries") is not False
    ):
        raise ValueError("frozen data or decision contract changed")
    protocol = Protocol(
        prefix_frames=int(windows["prefix_frames"]),
        horizon_frames=int(windows["horizon_frames"]),
        stride_frames=int(windows["stride_frames"]),
        action_scales=tuple_of(decision["action_scales"], float),
        neighbor_grid=tuple_of(decision["neighbor_grid"], int),
        cluster_grid=tuple_of(decision["cluster_grid"], int),
        temperature_grid=tuple_of(decision["temperature_scale_grid"], float),
        regret_tolerance_grid=tuple_of(
            decision["regret_tolerance_grid"], float
        ),
        kmeans_iterations=int(decision["kmeans_iterations"]),
        source_fit_count=int(split["fit_count"]),
        source_calibration_count=int(split["calibration_count"]),
        source_test_count=int(split["source_test_count"]),
        partition_domain=str(split["domain_separator"]),
        source_gate_mean_ratio=float(gate["maximum_mean_rmse_ratio"]),
        source_gate_worst_trajectory_ratio=float(
            gate["maximum_worst_trajectory_rmse_ratio"]
        ),
        source_gate_minimum_nonfallback_fraction=float(
            gate["minimum_nonfallback_fraction"]
        ),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    if (
        protocol.prefix_frames < 2
        or protocol.horizon_frames < 1
        or protocol.stride_frames < 1
        or not protocol.action_scales
        or protocol.action_scales[0] != 0.0
        or any(value < 0.0 for value in protocol.regret_tolerance_grid)
        or (
            protocol.source_fit_count
            + protocol.source_calibration_count
            + protocol.source_test_count
            != 56
        )
    ):
        raise ValueError("invalid frozen protocol")
    return protocol


def validate_request(path: Path) -> dict[str, object]:
    request = read_json(path)
    if (
        request.get("contract") != REQUEST_CONTRACT
        or request.get("schema_version") != 1
        or request.get("status") != "authorized"
        or tuple(request.get("dlos", ())) != DLOS
        or request.get("target_tuning") is not False
        or request.get("target_retries") is not False
        or not isinstance(request.get("run_key"), str)
        or not str(request["run_key"]).strip()
    ):
        raise ValueError("invalid target-evaluation request")
    return request


def trajectory_paths(root: Path, dlo: str, partition: str) -> tuple[Path, ...]:
    paths = tuple(sorted((root / dlo / partition).glob("*.pkl")))
    expected = 56 if partition == "train" else 14
    if len(paths) != expected:
        raise ValueError(
            f"{dlo}/{partition}: expected {expected} files, got {len(paths)}"
        )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise ValueError(f"{dlo}/{partition}: missing or empty trajectory")
    return paths


def load_trajectory(path: Path) -> FloatArray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    array = np.asarray(raw, dtype=np.float64)
    expected = (FRAME_COUNT, 3, NODE_COUNT)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: expected finite {expected}, got {array.shape}")
    nodes = np.transpose(array, (0, 2, 1)).copy()
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return nodes


def file_manifest(paths: tuple[Path, ...]) -> dict[str, object]:
    return {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def partition_names(
    names: tuple[str, ...], dlo: str, protocol: Protocol
) -> dict[str, tuple[str, ...]]:
    if len(names) != 56 or len(set(names)) != 56:
        raise ValueError(f"{dlo}: expected 56 unique training names")

    def key(name: str) -> tuple[bytes, str]:
        payload = (
            protocol.partition_domain.encode("utf-8")
            + b"\0"
            + dlo.encode("utf-8")
            + b"\0"
            + name.encode("utf-8")
        )
        return hashlib.sha256(payload).digest(), name

    ordered = tuple(sorted(names, key=key))
    first = protocol.source_fit_count
    second = first + protocol.source_calibration_count
    return {
        "fit": ordered[:first],
        "calibration": ordered[first:second],
        "source_test": ordered[second:],
    }


def window_starts(protocol: Protocol) -> tuple[int, ...]:
    first = protocol.prefix_frames - 1
    stop = FRAME_COUNT - protocol.horizon_frames
    starts = tuple(range(first, stop, protocol.stride_frames))
    if not starts:
        raise ValueError("protocol yields no evaluation windows")
    return starts


def anchor_means(nodes: FloatArray) -> tuple[FloatArray, FloatArray]:
    left = np.mean(nodes[..., list(ACTION_LEFT), :], axis=-2)
    right = np.mean(nodes[..., list(ACTION_RIGHT), :], axis=-2)
    return left, right


def observation_from_parts(
    prefix: FloatArray,
    future_action_nodes: FloatArray,
    protocol: Protocol,
) -> Observation:
    expected_prefix = (protocol.prefix_frames, NODE_COUNT, 3)
    expected_action = (protocol.horizon_frames, 4, 3)
    if prefix.shape != expected_prefix or future_action_nodes.shape != expected_action:
        raise ValueError("observation arrays have incorrect shapes")
    current = prefix[-1]
    previous = prefix[-2]
    current_left, current_right = anchor_means(current[None, ...])
    previous_left, previous_right = anchor_means(previous[None, ...])
    current_left = current_left[0]
    current_right = current_right[0]
    previous_left = previous_left[0]
    previous_right = previous_right[0]
    future_left = np.mean(future_action_nodes[:, :2, :], axis=1)
    future_right = np.mean(future_action_nodes[:, 2:, :], axis=1)
    length_scale = float(np.linalg.norm(current_right - current_left))
    length_scale = max(length_scale, 1e-6)
    internal_count = NODE_COUNT - 4
    weights = np.linspace(
        1.0 / (internal_count + 1),
        internal_count / (internal_count + 1),
        internal_count,
        dtype=np.float64,
    )
    blend_weights = weights[None, :, None]
    current_line = (
        (1.0 - blend_weights[0]) * current_left
        + blend_weights[0] * current_right
    )
    current_internal = current[INTERNAL]
    previous_internal = previous[INTERNAL]
    previous_line = (
        (1.0 - blend_weights[0]) * previous_left
        + blend_weights[0] * previous_right
    )
    shape = (current_internal - current_line) / length_scale
    shape_velocity = (
        (current_internal - current_line)
        - (previous_internal - previous_line)
    ) / length_scale
    left_displacement = future_left - current_left
    right_displacement = future_right - current_right
    anchor_displacement = (
        (1.0 - blend_weights) * left_displacement[:, None, :]
        + blend_weights * right_displacement[:, None, :]
    )
    decay = 0.85
    steps = np.arange(1, protocol.horizon_frames + 1, dtype=np.float64)
    if math.isclose(decay, 1.0):
        velocity_factor = steps
    else:
        velocity_factor = decay * (1.0 - np.power(decay, steps)) / (1.0 - decay)
    baseline = (
        current_internal[None, ...]
        + anchor_displacement
        + velocity_factor[:, None, None] * shape_velocity[None, ...] * length_scale
    )
    sample_indices = np.linspace(
        0,
        protocol.horizon_frames - 1,
        5,
        dtype=np.int64,
    )
    action_feature = np.concatenate(
        (
            left_displacement[sample_indices],
            right_displacement[sample_indices],
        ),
        axis=1,
    ) / length_scale
    prefix_line_left, prefix_line_right = anchor_means(prefix)
    prefix_midpoint = 0.5 * (prefix_line_left + prefix_line_right)
    midpoint_velocity = (
        prefix_midpoint[-1] - prefix_midpoint[-2]
    ) / length_scale
    feature = np.concatenate(
        (
            shape.reshape(-1),
            shape_velocity.reshape(-1),
            action_feature.reshape(-1),
            midpoint_velocity.reshape(-1),
        )
    )
    return Observation(feature, baseline, length_scale)


def extract_observation(
    trajectory: FloatArray,
    current: int,
    protocol: Protocol,
) -> Observation:
    prefix = trajectory[
        current - protocol.prefix_frames + 1 : current + 1
    ].copy()
    future = trajectory[
        current + 1 : current + 1 + protocol.horizon_frames
    ]
    future_action = np.concatenate(
        (future[:, :2, :], future[:, -2:, :]),
        axis=1,
    ).copy()
    return observation_from_parts(prefix, future_action, protocol)


def source_window(
    trajectory: FloatArray,
    current: int,
    protocol: Protocol,
) -> tuple[FloatArray, FloatArray]:
    observation = extract_observation(trajectory, current, protocol)
    truth = trajectory[
        current + 1 : current + 1 + protocol.horizon_frames,
        INTERNAL,
        :,
    ]
    residual = (truth - observation.baseline) / observation.length_scale
    return observation.feature, residual.reshape(-1)


