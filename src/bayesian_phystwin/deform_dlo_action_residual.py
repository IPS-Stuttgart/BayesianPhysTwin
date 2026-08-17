"""Action-conditioned residual analog posterior for DEFORM rollouts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

DEFORM_ACTION_RESIDUAL_PROTOCOL_SCHEMA_VERSION = 1
DEFORM_ACTION_RESIDUAL_PROTOCOL_CONTRACT = "deform-dlo-action-residual-source-v3"


def load_deform_action_residual_protocol(path: str | Path) -> dict[str, object]:
    """Load the DLO1-only development protocol for the residual posterior."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_ACTION_RESIDUAL_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM action-residual protocol schema")
    if payload.get("contract") != DEFORM_ACTION_RESIDUAL_PROTOCOL_CONTRACT:
        raise ValueError("unsupported DEFORM action-residual protocol contract")
    if (
        payload.get("dlo_type") != "DLO1"
        or payload.get("source_test_status") != "post-open-development-only"
        or payload.get("official_eval_policy") != "forbidden"
        or payload.get("fresh_confirmation_dlo") != "DLO2"
    ):
        raise ValueError("action-residual development must remain DLO1-only")
    boundaries = payload.get("information_boundary")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("dlo2_training_read") is not False
        or boundaries.get("dlo2_source_outcome_read") is not False
        or boundaries.get("official_eval_read") is not False
    ):
        raise ValueError("action-residual protocol does not seal future data")
    for key in ("longrun_protocol", "longrun_result", "source_manifest"):
        identity = payload.get(key)
        if (
            not isinstance(identity, Mapping)
            or not str(identity.get("repository_path", identity.get("path", "")))
            or len(str(identity.get("sha256", ""))) != 64
        ):
            raise ValueError("action-residual parent identity is invalid")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("action-residual protocol omits its baseline")
    for key in ("validation_l1_m", "source_test_l1_m"):
        value = float(baseline.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("action-residual baseline metric is invalid")
    tolerance = float(baseline.get("reproduction_tolerance_m", math.nan))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("action-residual baseline tolerance is invalid")
    descriptor = payload.get("descriptor")
    if (
        not isinstance(descriptor, Mapping)
        or int(descriptor.get("sample_count", -1)) < 2
        or descriptor.get("query_evidence")
        != "observed-initial-state-and-known-clamped-action-only"
        or descriptor.get("frame") != "initial-action-gravity-frame-v1"
    ):
        raise ValueError("action-residual descriptor contract is invalid")
    posterior = payload.get("posterior")
    if (
        not isinstance(posterior, Mapping)
        or posterior.get("duplicate_policy") != "collapse-exact-descriptor-clusters"
        or posterior.get("uncertainty") != "unshrunk-donor-mixture-spread-plus-floor"
    ):
        raise ValueError("action-residual posterior contract is invalid")
    floor = float(posterior.get("coordinate_variance_floor_m2", math.nan))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("action-residual variance floor is invalid")
    bank = payload.get("candidate_bank")
    if not isinstance(bank, Mapping):
        raise ValueError("action-residual protocol omits its candidate bank")
    counts = tuple(int(value) for value in bank.get("neighbor_counts", ()))
    scales = tuple(float(value) for value in bank.get("length_scale_multipliers", ()))
    shrinkages = tuple(float(value) for value in bank.get("shrinkages", ()))
    if (
        not counts
        or tuple(sorted(set(counts))) != counts
        or any(value < 1 for value in counts)
        or not scales
        or any(not math.isfinite(value) or value <= 0.0 for value in scales)
        or not shrinkages
        or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 for value in shrinkages
        )
    ):
        raise ValueError("action-residual candidate bank is invalid")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("action-residual protocol omits its gates")
    for stage in ("validation", "source_test"):
        gate = gates.get(stage)
        if not isinstance(gate, Mapping):
            raise ValueError("action-residual gate is invalid")
        improvement = float(gate.get("minimum_relative_improvement", math.nan))
        ratio = float(gate.get("maximum_case_ratio", math.nan))
        wins = int(gate.get("minimum_case_wins", -1))
        if (
            not math.isfinite(improvement)
            or not 0.0 < improvement < 1.0
            or not math.isfinite(ratio)
            or ratio < 1.0
            or wins < 1
        ):
            raise ValueError("action-residual gate threshold is invalid")
    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def _trajectory_batch(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 4
        or array.shape[0] == 0
        or array.shape[1] < 2
        or array.shape[2] < 5
        or array.shape[3] != 3
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} must be a finite trajectory batch")
    return array


def _clamped_indices(node_count: int) -> np.ndarray:
    if node_count < 5:
        raise ValueError("DEFORM action residual requires at least five nodes")
    return np.asarray((0, 1, node_count - 2, node_count - 1), dtype=np.int64)


def _unit(vector: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot construct {label} from degenerate geometry")
    return vector / norm


def _initial_frames(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_count = trajectories.shape[2]
    clamped = _clamped_indices(node_count)
    initial = trajectories[:, 0]
    centers = np.mean(initial[:, clamped], axis=1)
    frames = []
    for points in initial:
        left = np.mean(points[clamped[:2]], axis=0)
        right = np.mean(points[clamped[2:]], axis=0)
        x_axis = _unit(right - left, label="primary action frame")
        secondary = 0.5 * (
            (points[clamped[1]] - points[clamped[0]])
            + (points[clamped[3]] - points[clamped[2]])
        )
        secondary = secondary - np.dot(secondary, x_axis) * x_axis
        if np.linalg.norm(secondary) <= 1e-10:
            # Gravity is part of the DEFORM world contract and resolves the
            # otherwise unobservable roll of a straight initial cable.
            fallback = np.asarray((0.0, 0.0, 1.0))
            if abs(float(np.dot(fallback, x_axis))) > 1.0 - 1e-8:
                fallback = np.asarray((0.0, 1.0, 0.0))
            secondary = fallback - np.dot(fallback, x_axis) * x_axis
        y_axis = _unit(secondary, label="secondary action frame")
        z_axis = _unit(np.cross(x_axis, y_axis), label="action-frame normal")
        y_axis = np.cross(z_axis, x_axis)
        frames.append(np.stack((x_axis, y_axis, z_axis), axis=1))
    return centers, np.stack(frames)


def build_deform_action_descriptors(
    trajectories: np.ndarray,
    *,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid-invariant initial-state and known-action descriptors."""

    array = _trajectory_batch(trajectories, label="trajectories")
    if sample_count < 2 or sample_count > array.shape[1]:
        raise ValueError("action descriptor sample count is invalid")
    centers, frames = _initial_frames(array)
    centered = array - centers[:, None, None, :]
    canonical = np.einsum("ntvi,nij->ntvj", centered, frames)
    clamped = _clamped_indices(array.shape[2])
    sample_indices = np.rint(np.linspace(0, array.shape[1] - 1, sample_count)).astype(
        np.int64
    )
    sampled_action = canonical[:, sample_indices][:, :, clamped]
    action_displacement = sampled_action - sampled_action[:, :1]
    action_velocity = np.diff(sampled_action, axis=1, prepend=sampled_action[:, :1])
    initial_shape = canonical[:, 0]
    descriptors = np.concatenate(
        (
            initial_shape.reshape(array.shape[0], -1),
            action_displacement.reshape(array.shape[0], -1),
            action_velocity.reshape(array.shape[0], -1),
        ),
        axis=1,
    )
    return descriptors, frames


def _standardize_descriptors(
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = np.mean(descriptors, axis=0)
    scale = np.std(descriptors, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (descriptors - location) / scale, location, scale


def _collapse_duplicate_donors(
    descriptors: np.ndarray,
    corrections: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...], np.ndarray]:
    groups: dict[bytes, list[int]] = {}
    for index, descriptor in enumerate(descriptors):
        key = np.round(descriptor, decimals=12).tobytes()
        groups.setdefault(key, []).append(index)
    grouped_descriptors = []
    grouped_corrections = []
    grouped_names = []
    grouped_sizes = []
    for indices in groups.values():
        grouped_descriptors.append(np.mean(descriptors[indices], axis=0))
        grouped_corrections.append(np.mean(corrections[indices], axis=0))
        grouped_names.append(tuple(str(names[index]) for index in indices))
        grouped_sizes.append(len(indices))
    return (
        np.stack(grouped_descriptors),
        np.stack(grouped_corrections),
        tuple(grouped_names),
        np.asarray(grouped_sizes, dtype=np.int64),
    )


def _reference_distance(descriptors: np.ndarray) -> float:
    if descriptors.shape[0] < 2:
        return 1.0
    difference = descriptors[:, None, :] - descriptors[None, :, :]
    distances = np.sqrt(np.mean(np.square(difference), axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    value = float(np.median(nearest))
    return value if math.isfinite(value) and value > 1e-12 else 1.0


def fit_deform_action_residual(
    trajectories: np.ndarray,
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    names: Sequence[str],
    *,
    sample_count: int,
    variance_floor_m2: float,
) -> dict[str, object]:
    """Fit a donor posterior using only source trajectories and their residuals."""

    full = _trajectory_batch(trajectories, label="fit trajectories")
    baseline = _trajectory_batch(baseline_predictions, label="fit predictions")
    observed = _trajectory_batch(targets, label="fit targets")
    if (
        baseline.shape != observed.shape
        or full.shape[0] != baseline.shape[0]
        or full.shape[2:] != baseline.shape[2:]
        or len(names) != full.shape[0]
        or len(set(names)) != len(names)
    ):
        raise ValueError("DEFORM action-residual fit arrays do not align")
    if not math.isfinite(variance_floor_m2) or variance_floor_m2 <= 0.0:
        raise ValueError("action-residual variance floor must be positive")
    raw_descriptors, frames = build_deform_action_descriptors(
        full,
        sample_count=sample_count,
    )
    descriptors, location, scale = _standardize_descriptors(raw_descriptors)
    residual = observed - baseline
    clamped = _clamped_indices(baseline.shape[2])
    residual[:, :, clamped] = 0.0
    canonical_residual = np.einsum("ntvi,nij->ntvj", residual, frames)
    (
        donor_descriptors,
        donor_corrections,
        donor_names,
        donor_cluster_sizes,
    ) = _collapse_duplicate_donors(descriptors, canonical_residual, names)
    return {
        "schema_version": 1,
        "contract": "deform-dlo-action-residual-model-v1",
        "sample_count": sample_count,
        "node_count": baseline.shape[2],
        "prediction_horizon": baseline.shape[1],
        "descriptor_location": location,
        "descriptor_scale": scale,
        "donor_descriptors": donor_descriptors,
        "donor_corrections_canonical": donor_corrections,
        "donor_names": donor_names,
        "donor_cluster_sizes": donor_cluster_sizes,
        "reference_distance": _reference_distance(donor_descriptors),
        "variance_floor_m2": variance_floor_m2,
    }


def _validated_model_array(
    model: Mapping[str, object], key: str, *, ndim: int
) -> np.ndarray:
    array = np.asarray(model.get(key), dtype=np.float64)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"action-residual model {key} is invalid")
    return array


def predict_deform_action_residual(
    model: Mapping[str, object],
    trajectories: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    neighbor_count: int,
    length_scale_multiplier: float,
    shrinkage: float,
) -> dict[str, object]:
    """Apply the residual mixture without using the query innovation."""

    full = _trajectory_batch(trajectories, label="query trajectories")
    baseline = _trajectory_batch(baseline_predictions, label="query predictions")
    if (
        full.shape[0] != baseline.shape[0]
        or full.shape[2:] != baseline.shape[2:]
        or int(cast(Any, model.get("node_count", -1))) != baseline.shape[2]
        or int(cast(Any, model.get("prediction_horizon", -1))) != baseline.shape[1]
    ):
        raise ValueError("DEFORM action-residual query arrays do not align")
    donors = _validated_model_array(model, "donor_descriptors", ndim=2)
    corrections = _validated_model_array(model, "donor_corrections_canonical", ndim=4)
    location = _validated_model_array(model, "descriptor_location", ndim=1)
    scale = _validated_model_array(model, "descriptor_scale", ndim=1)
    if (
        donors.shape[0] != corrections.shape[0]
        or donors.shape[1] != location.size
        or scale.shape != location.shape
        or corrections.shape[1:] != baseline.shape[1:]
    ):
        raise ValueError("DEFORM action-residual model arrays do not align")
    if neighbor_count < 1 or neighbor_count > donors.shape[0]:
        raise ValueError("action-residual neighbor count is invalid")
    if not math.isfinite(length_scale_multiplier) or length_scale_multiplier <= 0.0:
        raise ValueError("action-residual length scale is invalid")
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("action-residual shrinkage is invalid")
    raw_query, frames = build_deform_action_descriptors(
        full,
        sample_count=int(cast(Any, model.get("sample_count", -1))),
    )
    query = (raw_query - location) / scale
    distance = np.sqrt(
        np.mean(np.square(query[:, None, :] - donors[None, :, :]), axis=2)
    )
    order = np.argsort(distance, axis=1, kind="stable")[:, :neighbor_count]
    selected_distance = np.take_along_axis(distance, order, axis=1)
    reference = float(cast(Any, model.get("reference_distance", math.nan)))
    length_scale = reference * length_scale_multiplier
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("action-residual effective length scale is invalid")
    logits = -0.5 * np.square(selected_distance / length_scale)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    selected_corrections = corrections[order]
    correction_canonical = np.einsum("nk,nktvc->ntvc", weights, selected_corrections)
    difference = selected_corrections - correction_canonical[:, None]
    variance_canonical = np.einsum("nk,nktvc->ntvc", weights, np.square(difference))
    correction_global = np.einsum("ntvj,nij->ntvi", correction_canonical, frames)
    rotation_squared = np.square(frames)
    variance_global = np.einsum("ntvj,nij->ntvi", variance_canonical, rotation_squared)
    variance_floor = float(cast(Any, model.get("variance_floor_m2", math.nan)))
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("action-residual model variance floor is invalid")
    variance_global = variance_global + variance_floor
    candidate = baseline + shrinkage * correction_global
    clamped = _clamped_indices(baseline.shape[2])
    candidate[:, :, clamped] = baseline[:, :, clamped]
    variance_global[:, :, clamped] = 0.0
    effective_sample_size = 1.0 / np.sum(np.square(weights), axis=1)
    return {
        "predictions": candidate,
        "coordinate_variance_m2": variance_global,
        "neighbor_indices": order,
        "neighbor_distances": selected_distance,
        "weights": weights,
        "effective_sample_size": effective_sample_size,
        "correction_l2_m": np.sqrt(
            np.mean(np.square(shrinkage * correction_global), axis=(1, 2, 3))
        ),
    }


def deform_action_residual_records(
    predictions: np.ndarray,
    targets: np.ndarray,
    baseline_predictions: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, object]]:
    """Return paired trajectory-level L1 records."""

    predicted = _trajectory_batch(predictions, label="candidate predictions")
    observed = _trajectory_batch(targets, label="targets")
    baseline = _trajectory_batch(baseline_predictions, label="baseline predictions")
    if (
        predicted.shape != observed.shape
        or predicted.shape != baseline.shape
        or len(names) != predicted.shape[0]
    ):
        raise ValueError("action-residual metric arrays do not align")
    candidate_error = np.mean(np.abs(predicted - observed), axis=(1, 2, 3))
    baseline_error = np.mean(np.abs(baseline - observed), axis=(1, 2, 3))
    records = []
    for index, name in enumerate(names):
        baseline_value = float(baseline_error[index])
        candidate_value = float(candidate_error[index])
        records.append(
            {
                "name": str(name),
                "baseline_l1_m": baseline_value,
                "candidate_l1_m": candidate_value,
                "candidate_to_baseline_ratio": (
                    candidate_value / baseline_value
                    if baseline_value > 0.0
                    else math.inf
                ),
                "improved": candidate_value < baseline_value,
            }
        )
    return records


def summarize_deform_action_residual_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize one arm without pooling dense coordinates as independent cases."""

    if not records:
        raise ValueError("action-residual summary requires trajectory records")
    baseline = np.asarray([record["baseline_l1_m"] for record in records], dtype=float)
    candidate = np.asarray(
        [record["candidate_l1_m"] for record in records], dtype=float
    )
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or np.any(baseline <= 0.0)
    ):
        raise ValueError("action-residual records contain invalid errors")
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    return {
        "trajectory_count": len(records),
        "baseline_mean_l1_m": baseline_mean,
        "candidate_mean_l1_m": candidate_mean,
        "relative_improvement": 1.0 - candidate_mean / baseline_mean,
        "wins": int(np.sum(candidate < baseline)),
        "ties": int(np.sum(candidate == baseline)),
        "maximum_case_ratio": float(np.max(candidate / baseline)),
    }


def select_deform_action_residual_arm(
    arm_records: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    minimum_relative_improvement: float,
    minimum_case_wins: int,
    maximum_case_ratio: float,
) -> dict[str, object]:
    """Select on validation only, otherwise require exact baseline fallback."""

    if not arm_records:
        raise ValueError("action-residual selector contains no arms")
    summaries = {
        name: summarize_deform_action_residual_records(records)
        for name, records in arm_records.items()
    }
    eligible = {
        name: summary
        for name, summary in summaries.items()
        if float(cast(Any, summary["relative_improvement"]))
        >= minimum_relative_improvement
        and int(cast(Any, summary["wins"])) >= minimum_case_wins
        and float(cast(Any, summary["maximum_case_ratio"])) <= maximum_case_ratio
    }
    if not eligible:
        return {
            "selected_arm": "baseline_exact",
            "fallback_used": True,
            "summaries": summaries,
        }
    selected = min(
        eligible,
        key=lambda name: (
            float(cast(Any, eligible[name]["candidate_mean_l1_m"])),
            name,
        ),
    )
    return {
        "selected_arm": selected,
        "fallback_used": False,
        "selected_summary": eligible[selected],
        "summaries": summaries,
    }


def serialize_deform_action_residual_model(
    model: Mapping[str, object],
) -> dict[str, Any]:
    """Return NPZ-compatible arrays without serializing Python objects."""

    names = model.get("donor_names")
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError("action-residual model donor names are invalid")
    donor_names = cast(Sequence[Sequence[str]], names)
    return {
        "schema_version": np.asarray(
            [int(cast(Any, model["schema_version"]))], dtype=np.int64
        ),
        "sample_count": np.asarray(
            [int(cast(Any, model["sample_count"]))], dtype=np.int64
        ),
        "node_count": np.asarray([int(cast(Any, model["node_count"]))], dtype=np.int64),
        "prediction_horizon": np.asarray(
            [int(cast(Any, model["prediction_horizon"]))], dtype=np.int64
        ),
        "descriptor_location": np.asarray(model["descriptor_location"]),
        "descriptor_scale": np.asarray(model["descriptor_scale"]),
        "donor_descriptors": np.asarray(model["donor_descriptors"]),
        "donor_corrections_canonical": np.asarray(model["donor_corrections_canonical"]),
        "donor_cluster_sizes": np.asarray(model["donor_cluster_sizes"]),
        "donor_names_json": np.asarray(
            [json.dumps([list(group) for group in donor_names], separators=(",", ":"))]
        ),
        "reference_distance": np.asarray(
            [float(cast(Any, model["reference_distance"]))], dtype=np.float64
        ),
        "variance_floor_m2": np.asarray(
            [float(cast(Any, model["variance_floor_m2"]))], dtype=np.float64
        ),
    }
