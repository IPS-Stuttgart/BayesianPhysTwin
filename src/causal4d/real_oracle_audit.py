"""Holdout-only oracle gaps and variance diagnostics for real Causal4D cases."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.contracts import PhysicalPosterior, TwinBelief
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    nominal_contact_hypotheses,
)
from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class HoldoutOracleProtocol:
    """An explicit information boundary for label-selected diagnostics."""

    start_frame: int
    stop_frame: int
    selection_metric: str = "track_error_m"
    label_use: str = "diagnostic_only"

    def __post_init__(self) -> None:
        if self.start_frame < 1 or self.stop_frame <= self.start_frame:
            raise ValueError("oracle holdout must be a nonempty post-endpoint interval")
        if self.selection_metric not in {"track_error_m", "coordinate_rmse_m"}:
            raise ValueError("unsupported oracle selection metric")
        if self.label_use != "diagnostic_only":
            raise ValueError("holdout labels may only be used for diagnostic oracles")


def _validate_bank_belief(bank: JointRolloutBank, belief: TwinBelief) -> None:
    expected = (len(bank.parameter_weights), bank.node_count, bank.coordinate_count)
    if belief.discrepancy_mean_m[:, : bank.node_count].shape != expected:
        raise ValueError("TwinBelief discrepancy does not match the rollout bank")
    if belief.discrepancy_variance_m2[:, : bank.node_count].shape != expected:
        raise ValueError("TwinBelief discrepancy variance does not match the bank")
    if not np.array_equal(belief.theta, bank.parameter_particles):
        raise ValueError("TwinBelief theta does not match the rollout bank")
    if not np.array_equal(belief.weights, bank.parameter_weights):
        raise ValueError("TwinBelief weights do not match the rollout bank")


def released_phystwin_prediction(
    released_trajectory_m: np.ndarray,
    *,
    endpoint_frame: int,
    frame_count: int,
    node_count: int,
) -> np.ndarray:
    """Extract the released nominal rollout without consuming holdout labels."""

    trajectory = np.asarray(released_trajectory_m, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError("released trajectory must have shape (T, N, 3)")
    stop = endpoint_frame + frame_count
    if endpoint_frame < 0 or stop > len(trajectory) or node_count > trajectory.shape[1]:
        raise ValueError("released trajectory does not cover the requested rollout")
    return trajectory[endpoint_frame:stop, :node_count].copy()


def bpt_nominal_prediction(
    bank: JointRolloutBank,
    belief: TwinBelief,
    o_plus_prefix_m: np.ndarray,
    *,
    prefix_mask: np.ndarray,
    config: FactualAbductionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition theta/delta on O+ while fixing realized intervention z to nominal.

    Only the supplied prefix can enter the update. The held-out suffix is not an
    argument to this function.
    """

    _validate_bank_belief(bank, belief)
    prefix = np.asarray(o_plus_prefix_m, dtype=float)
    mask = np.asarray(prefix_mask, dtype=bool)
    if prefix.ndim != 3 or prefix.shape[1:] != bank.trajectories.shape[3:]:
        raise ValueError("O+ prefix must have shape (F, N, C) matching the bank")
    if mask.shape not in {prefix.shape, prefix.shape[:2]}:
        raise ValueError("prefix mask must have shape (F, N) or (F, N, C)")
    if not 2 <= len(prefix) < bank.frame_count:
        raise ValueError("O+ prefix must reveal evidence and leave a holdout")

    observations = np.full(bank.trajectories.shape[2:], np.nan, dtype=float)
    observations[: len(prefix)] = prefix
    padded_mask = np.zeros(bank.trajectories.shape[2:4], dtype=bool)
    if mask.ndim == 3:
        mask = np.all(mask, axis=2)
    padded_mask[: len(prefix)] = mask

    nominal = nominal_contact_hypotheses(bank)
    base_weights = np.zeros_like(bank.prior_joint_weights)
    action_weights = bank.hypothesis_prior_weights[nominal]
    action_weights = action_weights / np.sum(action_weights)
    base_weights[nominal] = action_weights[:, None] * belief.weights[None]
    weights = bank.update_from_observations(
        observations,
        prefix_frame_count=len(prefix),
        scale_m=config.observation_scale_m,
        likelihood_power=config.likelihood_power,
        dynamic_likelihood_weight=config.dynamic_likelihood_weight,
        degrees_of_freedom=config.degrees_of_freedom,
        mask=padded_mask,
        base_weights=base_weights,
        particle_discrepancy_m=belief.discrepancy_mean_m[:, : bank.node_count],
        particle_discrepancy_variance_m2=belief.discrepancy_variance_m2[
            :, : bank.node_count
        ],
    )
    prediction = np.zeros(bank.trajectories.shape[2:], dtype=float)
    discrepancy = belief.discrepancy_mean_m[:, : bank.node_count]
    for hypothesis_index, particle_index in zip(*np.nonzero(weights), strict=True):
        weight = float(weights[hypothesis_index, particle_index])
        prediction += weight * (
            bank.trajectories[hypothesis_index, particle_index]
            + discrepancy[particle_index]
        )
    return prediction, weights


def causal4d_posterior_prediction(posterior: PhysicalPosterior) -> np.ndarray:
    """Moment-match a Causal4D posterior without accessing evaluation labels."""

    return np.einsum(
        "k,ktnc->tnc",
        posterior.weights,
        posterior.readout_trajectories_m.astype(float),
    )


def verify_nested_rollout_banks(
    current: JointRolloutBank,
    expanded: JointRolloutBank,
    *,
    absolute_tolerance_m: float = 1e-8,
) -> dict[str, Any]:
    """Verify that proposal expansion preserves every current-bank rollout."""

    if absolute_tolerance_m < 0.0:
        raise ValueError("bank nesting tolerance must be nonnegative")
    if current.trajectories.shape[1:] != expanded.trajectories.shape[1:]:
        raise ValueError("nested banks must share particle and rollout shapes")
    if not np.array_equal(current.parameter_particles, expanded.parameter_particles):
        raise ValueError("nested banks use different parameter particles")
    if not np.array_equal(current.parameter_weights, expanded.parameter_weights):
        raise ValueError("nested banks use different parameter weights")
    expanded_indices = {
        hypothesis_id: index
        for index, hypothesis_id in enumerate(expanded.hypothesis_ids)
    }
    missing = [
        hypothesis_id
        for hypothesis_id in current.hypothesis_ids
        if hypothesis_id not in expanded_indices
    ]
    if missing:
        raise ValueError(f"expanded bank is missing current hypotheses: {missing}")
    maximum_difference = 0.0
    mapped_indices = []
    for current_index, hypothesis_id in enumerate(current.hypothesis_ids):
        expanded_index = expanded_indices[hypothesis_id]
        mapped_indices.append(expanded_index)
        current_metadata = current.hypothesis_metadata[current_index]
        expanded_metadata = expanded.hypothesis_metadata[expanded_index]
        current_action = {
            key: value
            for key, value in current_metadata["action"].items()
            if key != "prior_weight"
        }
        expanded_action = {
            key: value
            for key, value in expanded_metadata["action"].items()
            if key != "prior_weight"
        }
        current_contact = {
            key: value
            for key, value in current_metadata["contact"].items()
            if key != "contact_prior_weight"
        }
        expanded_contact = {
            key: value
            for key, value in expanded_metadata["contact"].items()
            if key != "contact_prior_weight"
        }
        if current_action != expanded_action or current_contact != expanded_contact:
            raise ValueError(f"nested hypothesis metadata changed for {hypothesis_id}")
        difference = float(
            np.max(
                np.abs(
                    current.trajectories[current_index].astype(float)
                    - expanded.trajectories[expanded_index].astype(float)
                )
            )
        )
        maximum_difference = max(maximum_difference, difference)
    if maximum_difference > absolute_tolerance_m:
        raise ValueError(
            "expanded bank changed a current rollout: "
            f"max |difference|={maximum_difference:.9g} m"
        )
    return {
        "verified": True,
        "current_hypothesis_count": len(current.hypothesis_ids),
        "expanded_hypothesis_count": len(expanded.hypothesis_ids),
        "current_to_expanded_indices": mapped_indices,
        "maximum_absolute_trajectory_difference_m": maximum_difference,
        "absolute_tolerance_m": absolute_tolerance_m,
    }


def _valid_mask(
    truth_m: np.ndarray,
    mask: np.ndarray,
    protocol: HoldoutOracleProtocol,
) -> np.ndarray:
    truth = np.asarray(truth_m, dtype=float)
    supplied = np.asarray(mask, dtype=bool)
    if truth.ndim != 3 or truth.shape[2] != 3:
        raise ValueError("truth must have shape (T, N, 3)")
    if supplied.shape == truth.shape:
        supplied = np.all(supplied, axis=2)
    if supplied.shape != truth.shape[:2]:
        raise ValueError("mask must have shape (T, N) or (T, N, 3)")
    if protocol.stop_frame > len(truth):
        raise ValueError("oracle protocol extends beyond available truth")
    valid = supplied & np.all(np.isfinite(truth), axis=2)
    selected = np.zeros_like(valid)
    selected[protocol.start_frame : protocol.stop_frame] = valid[
        protocol.start_frame : protocol.stop_frame
    ]
    if not np.any(selected):
        raise ValueError("holdout contains no valid tracked points")
    return selected


def evaluate_prediction(
    prediction_m: np.ndarray,
    truth_m: np.ndarray,
    mask: np.ndarray,
    protocol: HoldoutOracleProtocol,
) -> dict[str, Any]:
    """Evaluate a fixed prediction; this function never selects a model."""

    prediction = np.asarray(prediction_m, dtype=float)
    truth = np.asarray(truth_m, dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")
    valid = _valid_mask(truth, mask, protocol)
    residual = prediction - truth
    vectors = residual[valid]
    final_frame = protocol.stop_frame - 1
    final_valid = valid[final_frame]
    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(vectors)))),
        "track_error_m": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "fde_m": (
            float(
                np.mean(
                    np.linalg.norm(residual[final_frame, final_valid], axis=1)
                )
            )
            if np.any(final_valid)
            else None
        ),
        "valid_point_frames": int(np.sum(valid)),
        "evaluation_frame_interval": [protocol.start_frame, protocol.stop_frame],
    }


def _constant_biases(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    cap_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = truth - prediction
    global_bias = np.mean(residual[valid], axis=0)
    counts = np.sum(valid, axis=0)
    point_bias = np.zeros(prediction.shape[1:], dtype=float)
    np.divide(
        np.sum(residual * valid[:, :, None], axis=0),
        counts[:, None],
        out=point_bias,
        where=counts[:, None] > 0,
    )
    norms = np.linalg.norm(point_bias, axis=1)
    scale = np.ones_like(norms)
    selected = norms > cap_m
    scale[selected] = cap_m / norms[selected]
    return global_bias, point_bias * scale[:, None], point_bias


def _flat_metric_columns(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_coordinate_rmse_m": metrics["coordinate_rmse_m"],
        f"{prefix}_track_error_m": metrics["track_error_m"],
        f"{prefix}_fde_m": metrics["fde_m"],
    }


def _best_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_prefix: str,
    selection_metric: str,
) -> dict[str, Any]:
    metric_key = f"{metric_prefix}_{selection_metric}"
    coordinate_key = f"{metric_prefix}_coordinate_rmse_m"
    best = min(
        rows,
        key=lambda row: (
            float(row[metric_key]),
            float(row[coordinate_key]),
            str(row["component_id"]),
        ),
    )
    return {
        "component_id": best["component_id"],
        "hypothesis_id": best["hypothesis_id"],
        "particle_id": best["particle_id"],
        "hypothesis_index": best["hypothesis_index"],
        "particle_index": best["particle_index"],
        "contact": best["contact"],
        "action": best["action"],
        "metrics": {
            "coordinate_rmse_m": best[coordinate_key],
            "track_error_m": best[metric_key.replace(selection_metric, "track_error_m")],
            "fde_m": best[f"{metric_prefix}_fde_m"],
            "valid_point_frames": best["valid_point_frames"],
            "evaluation_frame_interval": best["evaluation_frame_interval"],
        },
    }


def audit_oracle_bank(
    bank: JointRolloutBank,
    belief: TwinBelief,
    truth_m: np.ndarray,
    mask: np.ndarray,
    protocol: HoldoutOracleProtocol,
    *,
    bank_name: str,
    discrepancy_cap_m: float = 0.01,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select holdout-labeled component and discrepancy ceilings.

    All selections and fitted corrections in this function are deliberately
    in-sample holdout oracles and are invalid as deployable predictors.
    """

    if discrepancy_cap_m <= 0.0:
        raise ValueError("discrepancy cap must be positive")
    _validate_bank_belief(bank, belief)
    truth = np.asarray(truth_m, dtype=float)
    if truth.shape != bank.trajectories.shape[2:]:
        raise ValueError("truth must match rollout bank trajectory shape")
    valid_full = _valid_mask(truth, mask, protocol)
    frame_slice = slice(protocol.start_frame, protocol.stop_frame)
    truth_selected = truth[frame_slice]
    valid_selected = valid_full[frame_slice]
    discrepancy = belief.discrepancy_mean_m[:, : bank.node_count]
    rows: list[dict[str, Any]] = []
    for hypothesis_index, (hypothesis_id, metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        for particle_index, particle_id in enumerate(belief.particle_ids):
            state = bank.trajectories[hypothesis_index, particle_index].astype(float)
            readout = state + discrepancy[particle_index]
            state_metrics = evaluate_prediction(state, truth, mask, protocol)
            readout_metrics = evaluate_prediction(readout, truth, mask, protocol)
            global_bias, capped_bias, uncapped_bias = _constant_biases(
                readout[frame_slice],
                truth_selected,
                valid_selected,
                cap_m=discrepancy_cap_m,
            )
            global_metrics = evaluate_prediction(
                readout + global_bias,
                truth,
                mask,
                protocol,
            )
            capped_metrics = evaluate_prediction(
                readout + capped_bias[None],
                truth,
                mask,
                protocol,
            )
            uncapped_metrics = evaluate_prediction(
                readout + uncapped_bias[None],
                truth,
                mask,
                protocol,
            )
            contact = dict(metadata["contact"])
            action = dict(metadata["action"])
            visible_nodes = np.any(valid_selected, axis=0)
            visible_bias_norms = np.linalg.norm(uncapped_bias[visible_nodes], axis=1)
            rows.append(
                {
                    "bank": bank_name,
                    "component_id": f"{hypothesis_id}::{particle_id}",
                    "hypothesis_id": hypothesis_id,
                    "particle_id": particle_id,
                    "hypothesis_index": hypothesis_index,
                    "particle_index": particle_index,
                    "contact": contact,
                    "action": action,
                    "valid_point_frames": readout_metrics["valid_point_frames"],
                    "evaluation_frame_interval": readout_metrics[
                        "evaluation_frame_interval"
                    ],
                    **_flat_metric_columns("state", state_metrics),
                    **_flat_metric_columns("readout", readout_metrics),
                    **_flat_metric_columns("global_translation", global_metrics),
                    **_flat_metric_columns("point_capped", capped_metrics),
                    **_flat_metric_columns("point_uncapped", uncapped_metrics),
                    "oracle_global_bias_x_m": float(global_bias[0]),
                    "oracle_global_bias_y_m": float(global_bias[1]),
                    "oracle_global_bias_z_m": float(global_bias[2]),
                    "oracle_point_bias_mean_norm_m": float(
                        np.mean(visible_bias_norms)
                    ),
                    "oracle_point_bias_p95_norm_m": float(
                        np.quantile(visible_bias_norms, 0.95)
                    ),
                    "oracle_point_bias_capped_fraction": float(
                        np.mean(visible_bias_norms > discrepancy_cap_m)
                    ),
                    "oracle_point_bias_valid_node_count": int(
                        np.sum(visible_nodes)
                    ),
                }
            )
    report = {
        "bank": bank_name,
        "label_use": protocol.label_use,
        "deployable": False,
        "selection_metric": protocol.selection_metric,
        "component_count": len(rows),
        "discrepancy_cap_m": discrepancy_cap_m,
        "best": {
            "state_only": _best_component(
                rows,
                metric_prefix="state",
                selection_metric=protocol.selection_metric,
            ),
            "discrepancy_aware": _best_component(
                rows,
                metric_prefix="readout",
                selection_metric=protocol.selection_metric,
            ),
            "global_translation": _best_component(
                rows,
                metric_prefix="global_translation",
                selection_metric=protocol.selection_metric,
            ),
            "per_node_constant_capped": _best_component(
                rows,
                metric_prefix="point_capped",
                selection_metric=protocol.selection_metric,
            ),
            "per_node_constant_uncapped": _best_component(
                rows,
                metric_prefix="point_uncapped",
                selection_metric=protocol.selection_metric,
            ),
        },
    }
    return report, rows


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.einsum("k,km->m", weights, values)
    return float(np.mean(np.einsum("k,km->m", weights, np.square(values - mean))))


def _weighted_cross_covariance(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    left_mean = np.einsum("k,km->m", weights, left)
    right_mean = np.einsum("k,km->m", weights, right)
    return float(
        np.mean(
            np.einsum(
                "k,km->m",
                weights,
                (left - left_mean) * (right - right_mean),
            )
        )
    )


def _explained_group_variance(
    values: np.ndarray,
    weights: np.ndarray,
    group_keys: Sequence[tuple[Any, ...]],
) -> float:
    if not group_keys or len(group_keys[0]) == 0:
        return 0.0
    overall = np.einsum("k,km->m", weights, values)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)
    result = 0.0
    for indices in groups.values():
        selected = np.asarray(indices, dtype=int)
        group_weight = float(np.sum(weights[selected]))
        if group_weight <= 0.0:
            continue
        conditional = np.einsum(
            "k,km->m",
            weights[selected] / group_weight,
            values[selected],
        )
        result += group_weight * float(np.mean(np.square(conditional - overall)))
    return result


def _shapley_family_variance(
    values: np.ndarray,
    weights: np.ndarray,
    family_keys: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, float], dict[str, float], float]:
    names = tuple(family_keys)
    count = len(names)
    subset_values: dict[tuple[str, ...], float] = {}
    for subset_size in range(count + 1):
        for subset in itertools.combinations(names, subset_size):
            keys = [
                tuple(family_keys[name][index] for name in subset)
                for index in range(len(weights))
            ]
            subset_values[subset] = _explained_group_variance(values, weights, keys)
    shapley: dict[str, float] = {}
    for name in names:
        others = tuple(candidate for candidate in names if candidate != name)
        contribution = 0.0
        for subset_size in range(len(others) + 1):
            coefficient = (
                math.factorial(subset_size)
                * math.factorial(count - subset_size - 1)
                / math.factorial(count)
            )
            for subset in itertools.combinations(others, subset_size):
                ordered = tuple(candidate for candidate in names if candidate in subset)
                with_name = tuple(
                    candidate
                    for candidate in names
                    if candidate in set(subset) | {name}
                )
                contribution += coefficient * (
                    subset_values[with_name] - subset_values[ordered]
                )
        shapley[name] = float(contribution)
    main_effects = {name: subset_values[(name,)] for name in names}
    full = subset_values[names]
    return shapley, main_effects, full


def _variance_entry(value: float, total: float) -> dict[str, float]:
    signed_root = math.copysign(math.sqrt(abs(value)) * 1000.0, value)
    return {
        "variance_m2": float(value),
        "signed_root_equivalent_mm": float(signed_root),
        "fraction_of_total_predictive_variance": (
            float(value / total) if total > 0.0 else 0.0
        ),
    }


def _variance_window(
    posterior: PhysicalPosterior,
    truth: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
    *,
    variance_floor_m2: float,
) -> dict[str, Any]:
    coordinate_mask = np.repeat(valid[frame_indices, :, None], 3, axis=2)
    states = posterior.state_trajectories_m[:, frame_indices].astype(float)
    readouts = posterior.readout_trajectories_m[:, frame_indices].astype(float)
    state_values = states[:, coordinate_mask]
    readout_values = readouts[:, coordinate_mask]
    delta_values = readout_values - state_values
    truth_values = truth[frame_indices][coordinate_mask]
    weights = posterior.weights

    state_epistemic = _weighted_variance(state_values, weights)
    delta_mean_epistemic = _weighted_variance(delta_values, weights)
    state_delta_cross = 2.0 * _weighted_cross_covariance(
        state_values,
        delta_values,
        weights,
    )
    readout_epistemic = _weighted_variance(readout_values, weights)

    conditional_static = np.maximum(
        posterior.readout_variance_m2.astype(float) - variance_floor_m2,
        0.0,
    )
    conditional = np.broadcast_to(
        conditional_static[:, None],
        (
            len(weights),
            len(frame_indices),
            posterior.readout_trajectories_m.shape[2],
            3,
        ),
    )[:, coordinate_mask]
    conditional_delta = float(
        np.mean(np.einsum("k,km->m", weights, conditional))
    )
    total_predictive = readout_epistemic + conditional_delta + variance_floor_m2
    mean_prediction = np.einsum("k,km->m", weights, readout_values)
    empirical_mse = float(np.mean(np.square(mean_prediction - truth_values)))

    family_keys: dict[str, Sequence[Any]] = {
        "theta": [int(value) for value in posterior.twin_particle_indices],
        "phi": [tuple(map(float, row)) for row in posterior.phi],
        "kappa": [tuple(map(float, row)) for row in posterior.kappa_cf],
    }
    shapley, main_effects, grouped_state = _shapley_family_variance(
        state_values,
        weights,
        family_keys,
    )
    unallocated_state = state_epistemic - grouped_state
    contributions = {
        "theta_shapley": _variance_entry(shapley["theta"], total_predictive),
        "phi_shapley": _variance_entry(shapley["phi"], total_predictive),
        "kappa_shapley": _variance_entry(shapley["kappa"], total_predictive),
        "unallocated_state_support": _variance_entry(
            unallocated_state,
            total_predictive,
        ),
        "discrepancy_mean_epistemic": _variance_entry(
            delta_mean_epistemic,
            total_predictive,
        ),
        "state_discrepancy_cross": _variance_entry(
            state_delta_cross,
            total_predictive,
        ),
        "discrepancy_conditional": _variance_entry(
            conditional_delta,
            total_predictive,
        ),
        "conditional_simulator_observation_noise": _variance_entry(
            variance_floor_m2,
            total_predictive,
        ),
    }
    allocated_total = sum(
        entry["variance_m2"] for entry in contributions.values()
    )
    return {
        "frame_interval": [int(frame_indices[0]), int(frame_indices[-1] + 1)],
        "valid_coordinate_count": int(np.sum(coordinate_mask)),
        "state_epistemic_variance_m2": state_epistemic,
        "readout_epistemic_variance_m2": readout_epistemic,
        "main_effect_variance_m2": {
            name: float(value) for name, value in main_effects.items()
        },
        "contributions": contributions,
        "total_predictive_variance_m2": total_predictive,
        "total_predictive_sd_mm": math.sqrt(total_predictive) * 1000.0,
        "empirical_residual_mse_m2": empirical_mse,
        "empirical_residual_rmse_mm": math.sqrt(empirical_mse) * 1000.0,
        "residual_mse_to_predictive_variance_ratio": (
            empirical_mse / total_predictive if total_predictive > 0.0 else None
        ),
        "closure": {
            "state_family_absolute_error_m2": float(
                abs(
                    state_epistemic
                    - sum(shapley.values())
                    - unallocated_state
                )
            ),
            "readout_algebra_absolute_error_m2": float(
                abs(
                    readout_epistemic
                    - state_epistemic
                    - delta_mean_epistemic
                    - state_delta_cross
                )
            ),
            "predictive_allocation_absolute_error_m2": float(
                abs(total_predictive - allocated_total)
            ),
        },
    }


def variance_decomposition(
    posterior: PhysicalPosterior,
    truth_m: np.ndarray,
    mask: np.ndarray,
    protocol: HoldoutOracleProtocol,
    *,
    variance_floor_m2: float,
) -> dict[str, Any]:
    """Allocate predictive variance without assuming independent latent families."""

    if variance_floor_m2 <= 0.0:
        raise ValueError("variance floor must be positive")
    truth = np.asarray(truth_m, dtype=float)
    if truth.shape != posterior.readout_trajectories_m.shape[1:]:
        raise ValueError("truth must match the PhysicalPosterior trajectories")
    valid = _valid_mask(truth, mask, protocol)
    frame_indices = np.arange(protocol.start_frame, protocol.stop_frame)
    chunks = np.array_split(frame_indices, 3)
    if any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("variance audit requires at least three holdout frames")
    return {
        "method": "weighted_shapley_variance_of_conditional_means",
        "latent_families": ["theta", "phi", "kappa"],
        "discrepancy_handling": (
            "delta mean, delta conditional variance, and the state-delta cross "
            "term are reported separately"
        ),
        "conditional_noise_note": (
            "The configured variance floor is a combined simulator/observation "
            "noise proxy; replay and observation noise are not separately "
            "identified by this audit."
        ),
        "all_holdout": _variance_window(
            posterior,
            truth,
            valid,
            frame_indices,
            variance_floor_m2=variance_floor_m2,
        ),
        "horizon": {
            label: _variance_window(
                posterior,
                truth,
                valid,
                chunk,
                variance_floor_m2=variance_floor_m2,
            )
            for label, chunk in zip(("early", "middle", "late"), chunks, strict=True)
        },
    }


def oracle_gap_report(
    current_causal4d_metrics: Mapping[str, Any],
    current_oracle: Mapping[str, Any],
    expanded_oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute inference, proposal, and model gaps under both error metrics."""

    result: dict[str, Any] = {}

    def fractions(values: Mapping[str, float], total: float) -> dict[str, float | None]:
        if abs(total) <= np.finfo(float).eps:
            return {name: None for name in values}
        return {name: float(value / total) for name, value in values.items()}

    for metric in ("track_error_m", "coordinate_rmse_m"):
        posterior_error = float(current_causal4d_metrics[metric])
        current_error = float(
            current_oracle["best"]["discrepancy_aware"]["metrics"][metric]
        )
        expanded_error = float(
            expanded_oracle["best"]["discrepancy_aware"]["metrics"][metric]
        )
        ceiling_error = float(
            expanded_oracle["best"]["per_node_constant_uncapped"]["metrics"][
                metric
            ]
        )
        capped_error = float(
            expanded_oracle["best"]["per_node_constant_capped"]["metrics"][metric]
        )
        inference_gap = posterior_error - current_error
        proposal_gap = current_error - expanded_error
        model_gap = expanded_error - ceiling_error
        capped_model_gap = expanded_error - capped_error
        total_headroom = posterior_error - ceiling_error
        capped_total_headroom = posterior_error - capped_error
        gaps = {
            "inference_gap": inference_gap,
            "proposal_gap": proposal_gap,
            "model_gap": model_gap,
        }
        result[metric] = {
            "current_causal4d_posterior": posterior_error,
            "current_bank_oracle": current_error,
            "expanded_bank_oracle": expanded_error,
            "oracle_discrepancy_ceiling": ceiling_error,
            "capped_oracle_discrepancy_ceiling": capped_error,
            **gaps,
            "capped_model_gap": capped_model_gap,
            "total_diagnostic_headroom": total_headroom,
            "capped_total_diagnostic_headroom": capped_total_headroom,
            "fraction_of_total_diagnostic_headroom": fractions(
                gaps,
                total_headroom,
            ),
            "fraction_of_capped_diagnostic_headroom": fractions(
                {
                    "inference_gap": inference_gap,
                    "proposal_gap": proposal_gap,
                    "capped_model_gap": capped_model_gap,
                },
                capped_total_headroom,
            ),
            "dominant_gap": max(gaps, key=gaps.get),
        }
    result["definitions"] = {
        "inference_gap": "Causal4D posterior error - current-bank oracle error",
        "proposal_gap": "current-bank oracle error - expanded-bank oracle error",
        "model_gap": "expanded-bank oracle error - labeled discrepancy ceiling error",
    }
    return result


def protocol_dict(protocol: HoldoutOracleProtocol) -> dict[str, Any]:
    """Return JSON-ready protocol metadata."""

    return asdict(protocol)
