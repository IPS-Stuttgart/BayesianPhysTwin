"""Evaluation of MolmoMotion evidence over real PhysTwin rollout banks."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.molmo_adapter import MolmoForecastBundle
from causal4d.rollout_bank import (
    JointRolloutBank,
    PhysicalTrajectoryDistribution,
    SparseTrajectoryEvidence,
)


def _coordinate_mask(mask: np.ndarray, coordinate_count: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("trajectory mask must have shape (T, N)")
    return np.repeat(values[:, :, None], coordinate_count, axis=2)


def physical_trajectory_metrics(
    prediction: PhysicalTrajectoryDistribution,
    truth_m: np.ndarray,
    mask: np.ndarray,
    *,
    start_frame: int,
    node_indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    truth = np.asarray(truth_m, dtype=float)
    if truth.shape != prediction.mean.shape:
        raise ValueError("truth and prediction must have matching shapes")
    if not 0 <= start_frame < len(truth):
        raise ValueError("start_frame must lie inside the trajectory")
    nodes = (
        np.arange(truth.shape[1], dtype=int)
        if node_indices is None
        else np.asarray(node_indices, dtype=int)
    )
    if nodes.ndim != 1 or not len(nodes) or np.any(nodes < 0) or np.any(nodes >= truth.shape[1]):
        raise ValueError("node_indices must identify available nodes")
    valid_nodes = np.asarray(mask, dtype=bool)[start_frame:, nodes]
    residual = prediction.mean[start_frame:, nodes] - truth[start_frame:, nodes]
    coordinate_valid = _coordinate_mask(valid_nodes, truth.shape[2])
    finite = np.isfinite(residual) & np.isfinite(truth[start_frame:, nodes])
    coordinate_valid &= finite
    if not np.any(coordinate_valid):
        raise ValueError("metric window has no valid coordinates")
    coordinate_rmse = float(
        np.sqrt(np.mean(np.square(residual[coordinate_valid])))
    )
    norm = np.linalg.norm(residual, axis=2)
    valid_norm = valid_nodes & np.all(finite, axis=2)
    vector_rmse = float(np.sqrt(np.mean(np.square(norm[valid_norm]))))
    ade = float(np.mean(norm[valid_norm]))
    valid_frame_indices = np.flatnonzero(np.any(valid_norm, axis=1))
    last = int(valid_frame_indices[-1])
    fde = float(np.mean(norm[last, valid_norm[last]]))
    frame_count = len(norm)
    thirds = np.array_split(np.arange(frame_count), 3)

    def segment_rmse(frames: np.ndarray) -> float:
        selected_valid = valid_norm[frames]
        if not np.any(selected_valid):
            return float("nan")
        return float(np.sqrt(np.mean(np.square(norm[frames][selected_valid]))))

    result: dict[str, float | int] = {
        "start_frame": int(start_frame),
        "valid_node_frames": int(np.sum(valid_norm)),
        "coordinate_rmse_m": coordinate_rmse,
        "vector_rmse_m": vector_rmse,
        "ade_m": ade,
        "fde_m": fde,
        "early_vector_rmse_m": segment_rmse(thirds[0]),
        "middle_vector_rmse_m": segment_rmse(thirds[1]),
        "late_vector_rmse_m": segment_rmse(thirds[2]),
    }
    if prediction.interval_lower is not None and prediction.interval_upper is not None:
        lower = prediction.interval_lower[start_frame:, nodes]
        upper = prediction.interval_upper[start_frame:, nodes]
        selected_truth = truth[start_frame:, nodes]
        covered = (selected_truth >= lower) & (selected_truth <= upper)
        result["coordinate_coverage"] = float(np.mean(covered[coordinate_valid]))
        result["mean_interval_width_m"] = float(
            np.mean((upper - lower)[coordinate_valid])
        )
    return result


def molmo_sparse_evidence(
    bundle: MolmoForecastBundle,
    forecast_id: str,
    bank: JointRolloutBank,
    *,
    scale_m: float,
    likelihood_weight: float,
    degrees_of_freedom: float = 3.0,
) -> SparseTrajectoryEvidence:
    if forecast_id not in bundle.forecast_ids:
        raise ValueError(f"unknown MolmoMotion forecast id {forecast_id!r}")
    forecast_index = bundle.forecast_ids.index(forecast_id)
    available = min(bundle.future_horizon, bank.frame_count - 1)
    if available < 1:
        raise ValueError("MolmoMotion and rollout bank have no overlapping future")
    # Both released datasets are frame-indexed. Do not stretch a short learned
    # forecast across a longer physical horizon; score only its supported prefix.
    future_world = bundle.future_world_m[forecast_index, :, :available]
    return SparseTrajectoryEvidence(
        positions_m=np.transpose(future_world, (1, 0, 2)),
        node_indices=bundle.query.node_indices,
        rollout_frame_indices=np.arange(1, available + 1, dtype=float),
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
        likelihood_weight=likelihood_weight,
        compare_displacements=True,
        anchor_positions_m=bundle.query.anchor_positions_world_m,
        anchor_rollout_frame=0,
        source=f"MolmoMotion:{forecast_id}",
    )


def _action_marginal(
    bank: JointRolloutBank,
    weights: np.ndarray,
) -> dict[str, float]:
    hypothesis_weights = bank.hypothesis_marginal(weights)
    result: dict[str, float] = {}
    for metadata, weight in zip(
        bank.hypothesis_metadata, hypothesis_weights, strict=True
    ):
        action_id = str(metadata["action"]["proposal_id"])
        result[action_id] = result.get(action_id, 0.0) + float(weight)
    return dict(sorted(result.items()))


def _contact_shift_marginal(
    bank: JointRolloutBank,
    weights: np.ndarray,
) -> dict[str, float]:
    hypothesis_weights = bank.hypothesis_marginal(weights)
    result: dict[str, float] = {}
    for metadata, weight in zip(
        bank.hypothesis_metadata, hypothesis_weights, strict=True
    ):
        key = ",".join(map(str, metadata["contact"]["attachment_shifts"]))
        result[key] = result.get(key, 0.0) + float(weight)
    return dict(sorted(result.items()))


def _oracle_joint_weights(
    bank: JointRolloutBank,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    start_frame: int,
) -> np.ndarray:
    valid = _coordinate_mask(mask[start_frame:], bank.coordinate_count)
    count = float(np.sum(valid))
    errors = np.empty(bank.prior_joint_weights.shape, dtype=float)
    selected_truth = truth[start_frame:]
    for hypothesis in range(bank.trajectories.shape[0]):
        for particle in range(bank.trajectories.shape[1]):
            residual = (
                bank.trajectories[hypothesis, particle, start_frame:]
                - selected_truth
            )
            errors[hypothesis, particle] = float(
                np.sum(np.where(valid, np.square(residual), 0.0)) / count
            )
    best = np.unravel_index(int(np.argmin(errors)), errors.shape)
    weights = np.zeros_like(bank.prior_joint_weights)
    weights[best] = 1.0
    return weights


def evaluate_phystwin_rollout_bank(
    bank: JointRolloutBank,
    bank_manifest: dict[str, Any],
    final_data_path: str | Path,
    molmo: MolmoForecastBundle,
    *,
    observation_fraction: float = 0.20,
    observation_scale_m: float = 0.006,
    observation_likelihood_power: float = 8.0,
    dynamic_likelihood_weight: float = 0.5,
    molmo_scale_m: float = 0.10,
    molmo_likelihood_weight: float = 12.0,
) -> dict[str, Any]:
    """Compare prior, Molmo, online, and combined posteriors on held-out frames."""

    if not 0.10 <= observation_fraction <= 0.20:
        raise ValueError("observation_fraction must lie in [0.10, 0.20]")
    with Path(final_data_path).open("rb") as handle:
        data = pickle.load(handle)
    train_end = int(bank_manifest["train_end_frame"])
    truth = np.asarray(data["object_points"], dtype=float)[train_end - 1 :]
    mask = (
        np.asarray(data["object_visibilities"], dtype=bool)
        & np.asarray(data["object_motions_valid"], dtype=bool)
    )[train_end - 1 :]
    if truth.shape != bank.trajectories.shape[2:]:
        raise ValueError("rollout bank and held-out PhysTwin observations differ in shape")
    if molmo.query.case_name != bank_manifest["case"]:
        raise ValueError("MolmoMotion and rollout artifacts refer to different cases")
    if molmo.query.t0_frame != train_end - 1:
        raise ValueError("MolmoMotion t0 does not match the rollout endpoint")
    prefix = max(3, int(np.ceil(observation_fraction * bank.frame_count)))
    prefix = min(prefix, bank.frame_count - 1)
    prior_weights = bank.prior_joint_weights
    online_weights = bank.update_from_observations(
        truth,
        prefix_frame_count=prefix,
        scale_m=observation_scale_m,
        likelihood_power=observation_likelihood_power,
        dynamic_likelihood_weight=dynamic_likelihood_weight,
        mask=mask,
    )
    weight_sets: dict[str, np.ndarray] = {
        "physics_prior": prior_weights,
        "online_prefix": online_weights,
    }
    evidence_diagnostics: dict[str, Any] = {}
    for forecast_id in molmo.forecast_ids:
        evidence = molmo_sparse_evidence(
            molmo,
            forecast_id,
            bank,
            scale_m=molmo_scale_m,
            likelihood_weight=molmo_likelihood_weight,
        )
        molmo_weights = bank.update_from_sparse_evidence(evidence)
        combined_weights = bank.update_from_observations(
            truth,
            prefix_frame_count=prefix,
            scale_m=observation_scale_m,
            likelihood_power=observation_likelihood_power,
            dynamic_likelihood_weight=dynamic_likelihood_weight,
            mask=mask,
            base_weights=molmo_weights,
        )
        weight_sets[f"molmo_{forecast_id}"] = molmo_weights
        weight_sets[f"molmo_{forecast_id}_plus_online"] = combined_weights
        available = len(evidence.rollout_frame_indices)
        query_truth = truth[1 : available + 1, evidence.node_indices]
        target = evidence.positions_m
        target_displacement = target - evidence.anchor_positions_m[None]
        truth_displacement = query_truth - truth[0, evidence.node_indices][None]
        residual = target_displacement - truth_displacement
        evidence_diagnostics[forecast_id] = {
            "caption": molmo.captions[molmo.forecast_ids.index(forecast_id)],
            "compared_future_frames": available,
            "displacement_ade_m": float(
                np.mean(np.linalg.norm(residual, axis=2))
            ),
            "displacement_fde_m": float(
                np.mean(np.linalg.norm(residual[-1], axis=1))
            ),
        }
    oracle_weights = _oracle_joint_weights(bank, truth, mask, start_frame=prefix)
    weight_sets["oracle_rollout"] = oracle_weights

    methods: dict[str, Any] = {}
    for method, weights in weight_sets.items():
        prediction = bank.predictive_distribution(
            weights,
            method=method,
            include_intervals=False,
        )
        methods[method] = {
            "future_after_online_prefix": physical_trajectory_metrics(
                prediction,
                truth,
                mask,
                start_frame=prefix,
            ),
            "query_future_after_online_prefix": physical_trajectory_metrics(
                prediction,
                truth,
                mask,
                start_frame=prefix,
                node_indices=molmo.query.node_indices,
            ),
            "full_future": physical_trajectory_metrics(
                prediction,
                truth,
                mask,
                start_frame=1,
            ),
            "action_marginal": _action_marginal(bank, weights),
            "attachment_shift_marginal": _contact_shift_marginal(bank, weights),
            "parameter_marginal": bank.parameter_marginal(weights).tolist(),
            "joint_effective_sample_size": float(
                1.0 / np.sum(np.square(weights))
            ),
        }

    true_id = "instruction" if "instruction" in molmo.forecast_ids else molmo.forecast_ids[0]
    true_method = f"molmo_{true_id}"
    prior_rmse = float(
        methods["physics_prior"]["future_after_online_prefix"]["vector_rmse_m"]
    )
    molmo_rmse = float(
        methods[true_method]["future_after_online_prefix"]["vector_rmse_m"]
    )
    online_rmse = float(
        methods["online_prefix"]["future_after_online_prefix"]["vector_rmse_m"]
    )
    combined_rmse = float(
        methods[f"{true_method}_plus_online"]["future_after_online_prefix"][
            "vector_rmse_m"
        ]
    )
    controls = [
        forecast_id
        for forecast_id in molmo.forecast_ids
        if forecast_id != true_id
    ]
    control_comparison = {
        forecast_id: {
            "vector_rmse_m": methods[f"molmo_{forecast_id}"][
                "future_after_online_prefix"
            ]["vector_rmse_m"],
            "instruction_minus_control_rmse_m": molmo_rmse
            - methods[f"molmo_{forecast_id}"]["future_after_online_prefix"][
                "vector_rmse_m"
            ],
        }
        for forecast_id in controls
    }
    return {
        "experiment": "causal4d-phystwin-molmo-pilot-v1",
        "case": bank_manifest["case"],
        "action_setting": bank_manifest.get("action_setting", "unspecified"),
        "contracts": {
            "physics_backend": "official PhysTwin Warp simulator",
            "parameter_support": "Bayesian-PhysTwin weighted profile particles",
            "contact_support": "Causal4D attachment/gain/delay/slip/rotation hypotheses",
            "molmo_role": "robust proposal/ranking evidence over rollouts",
            "online_update": f"prefix frames [1, {prefix}) only",
            "evaluation": f"future frames [{prefix}, {bank.frame_count})",
            "molmo_statistical_role": "product of experts, not an independent likelihood",
        },
        "configuration": {
            "observation_fraction": observation_fraction,
            "observation_scale_m": observation_scale_m,
            "observation_likelihood_power": observation_likelihood_power,
            "dynamic_likelihood_weight": dynamic_likelihood_weight,
            "molmo_scale_m": molmo_scale_m,
            "molmo_likelihood_weight": molmo_likelihood_weight,
        },
        "evidence_diagnostics": evidence_diagnostics,
        "methods": methods,
        "headline": {
            "instruction_forecast_id": true_id,
            "physics_prior_vector_rmse_m": prior_rmse,
            "molmo_vector_rmse_m": molmo_rmse,
            "molmo_percent_change_vs_prior": 100.0 * (molmo_rmse / prior_rmse - 1.0),
            "online_vector_rmse_m": online_rmse,
            "molmo_plus_online_vector_rmse_m": combined_rmse,
            "combined_percent_change_vs_online": 100.0
            * (combined_rmse / online_rmse - 1.0),
            "molmo_improves_prior": molmo_rmse < prior_rmse,
            "molmo_improves_online": combined_rmse < online_rmse,
            "language_controls": control_comparison,
        },
        "source_manifests": {
            "rollout_bank": bank_manifest,
            "molmo": molmo.metadata(),
        },
    }


def write_phystwin_evaluation(path: str | Path, result: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
