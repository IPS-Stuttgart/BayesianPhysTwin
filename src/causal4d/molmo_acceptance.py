"""Locked competence checks before MolmoMotion can reweight physical rollouts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.molmo_adapter import MolmoForecastBundle, camera_to_world_points
from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class MolmoAcceptanceThresholds:
    """Precommitted gates for a simulator-independent semantic competence test."""

    expected_forecast_fps: float = 15.0
    maximum_ade_ratio_vs_zero: float = 0.95
    maximum_ade_ratio_vs_constant_velocity: float = 0.95
    minimum_motion_scale_ratio: float = 0.5
    maximum_motion_scale_ratio: float = 2.0
    maximum_anchor_alignment_rmse_m: float = 1e-5
    maximum_first_step_error_m: float = 0.02
    maximum_frame_transform_error_m: float = 1e-5
    required_correct_action_rank: int = 1
    ranking_top_k: int = 2
    minimum_paraphrase_top_k_recall: float = 2.0 / 3.0
    minimum_prompt_top1_agreement: float = 2.0 / 3.0
    maximum_prompt_pairwise_motion_ratio: float = 0.5
    minimum_query_subset_top1_agreement: float = 0.75
    minimum_query_subset_top_k_recall: float = 0.75
    minimum_paraphrases: int = 3
    minimum_independent_cases: int = 3
    minimum_case_pass_fraction: float = 0.8
    semantic_scale_m: float = 0.10
    semantic_degrees_of_freedom: float = 3.0

    def __post_init__(self) -> None:
        positive = (
            self.expected_forecast_fps,
            self.maximum_ade_ratio_vs_zero,
            self.maximum_ade_ratio_vs_constant_velocity,
            self.minimum_motion_scale_ratio,
            self.maximum_motion_scale_ratio,
            self.maximum_anchor_alignment_rmse_m,
            self.maximum_first_step_error_m,
            self.maximum_frame_transform_error_m,
            self.semantic_scale_m,
            self.semantic_degrees_of_freedom,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Molmo acceptance thresholds must be finite and positive")
        if self.minimum_motion_scale_ratio > self.maximum_motion_scale_ratio:
            raise ValueError("motion-scale interval is empty")
        probabilities = (
            self.minimum_paraphrase_top_k_recall,
            self.minimum_prompt_top1_agreement,
            self.minimum_query_subset_top1_agreement,
            self.minimum_query_subset_top_k_recall,
            self.minimum_case_pass_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Molmo acceptance proportions must lie in [0, 1]")
        integers = (
            self.required_correct_action_rank,
            self.ranking_top_k,
            self.minimum_paraphrases,
            self.minimum_independent_cases,
        )
        if any(value < 1 for value in integers):
            raise ValueError("Molmo acceptance count thresholds must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> MolmoAcceptanceThresholds:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "unknown Molmo acceptance thresholds: "
                + ", ".join(sorted(unknown))
            )
        return cls(**dict(values))

    def as_dict(self) -> dict[str, float | int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def load_molmo_acceptance_result(path: str | Path) -> dict[str, Any]:
    """Load and validate the decision needed to unlock positive beta values."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Molmo acceptance result schema_version must be 1")
    expected_id = payload.get("acceptance_result_id")
    if not isinstance(expected_id, str) or expected_id != molmo_acceptance_result_id(
        payload
    ):
        raise ValueError("Molmo acceptance result digest does not match")
    decision = payload.get("decision")
    if not isinstance(decision, dict) or not isinstance(
        decision.get("accepted_for_semantic_reweighting"),
        bool,
    ):
        raise ValueError("Molmo acceptance result has no typed acceptance decision")
    return payload


def molmo_acceptance_result_id(payload: Mapping[str, Any]) -> str:
    """Hash a result without its self-referential digest field."""

    canonical = {
        key: value for key, value in payload.items() if key != "acceptance_result_id"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gate_beta_candidates(
    candidates: Sequence[float],
    acceptance_result: Mapping[str, Any] | None,
) -> tuple[float, ...]:
    """Permit positive beta candidates only after independent acceptance."""

    requested = tuple(sorted(set(map(float, candidates))))
    if not requested or 0.0 not in requested or any(value < 0.0 for value in requested):
        raise ValueError("beta candidates must be nonnegative and include zero")
    accepted = bool(
        acceptance_result
        and acceptance_result["decision"]["accepted_for_semantic_reweighting"]
    )
    return requested if accepted else (0.0,)


def _trajectory_metrics(
    prediction_m: np.ndarray,
    truth_m: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction_m, dtype=float)
    truth = np.asarray(truth_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if prediction.shape != truth.shape or mask.shape != truth.shape[:2]:
        raise ValueError("trajectory metric inputs have incompatible shapes")
    finite = np.all(np.isfinite(prediction) & np.isfinite(truth), axis=2)
    mask &= finite
    if not np.any(mask):
        raise ValueError("trajectory metric has no valid point-frames")
    errors = np.linalg.norm(prediction - truth, axis=2)
    valid_frames = np.flatnonzero(np.any(mask, axis=1))
    last = int(valid_frames[-1])
    return {
        "ade_m": float(np.mean(errors[mask])),
        "vector_rmse_m": float(np.sqrt(np.mean(np.square(errors[mask])))),
        "fde_m": float(np.mean(errors[last, mask[last]])),
    }


def _motion_rms(
    positions_m: np.ndarray,
    anchor_m: np.ndarray,
    valid: np.ndarray,
) -> float:
    displacement = np.asarray(positions_m, dtype=float) - np.asarray(anchor_m)[None]
    norms_squared = np.sum(np.square(displacement), axis=2)
    return float(np.sqrt(np.mean(norms_squared[np.asarray(valid, dtype=bool)])))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("Molmo acceptance ratio requires a positive denominator")
    return float(numerator / denominator)


def _forecast_truth(
    bundle: MolmoForecastBundle,
    object_points_m: np.ndarray,
    validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(object_points_m, dtype=float)
    valid = np.asarray(validity, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("observed points and validity have incompatible shapes")
    query = bundle.query
    offsets = np.arange(1, bundle.future_horizon + 1) * query.frame_stride
    frame_indices = query.t0_frame + offsets
    frame_indices = frame_indices[frame_indices < len(points)]
    if not len(frame_indices):
        raise ValueError("Molmo forecast has no sampled held-out target frames")
    nodes = query.node_indices
    truth = points[frame_indices[:, None], nodes[None]]
    truth_valid = valid[frame_indices[:, None], nodes[None]]
    return frame_indices, truth, truth_valid


def forecast_competence_diagnostics(
    bundle: MolmoForecastBundle,
    forecast_id: str,
    object_points_m: np.ndarray,
    validity: np.ndarray,
    thresholds: MolmoAcceptanceThresholds,
) -> dict[str, Any]:
    """Compare a raw Molmo forecast with zero and constant-velocity baselines."""

    if forecast_id not in bundle.forecast_ids:
        raise ValueError(f"unknown Molmo forecast id {forecast_id!r}")
    frame_indices, truth, truth_valid = _forecast_truth(
        bundle,
        object_points_m,
        validity,
    )
    forecast_index = bundle.forecast_ids.index(forecast_id)
    forecast = np.transpose(
        bundle.future_world_m[forecast_index, :, : len(frame_indices)],
        (1, 0, 2),
    )
    anchor = bundle.query.anchor_positions_world_m
    zero = np.repeat(anchor[None], len(frame_indices), axis=0)
    history = bundle.query.points_3d_world_history_m
    velocity = history[-1] - history[-2] if len(history) >= 2 else np.zeros_like(anchor)
    constant_velocity = anchor[None] + (
        np.arange(1, len(frame_indices) + 1)[:, None, None] * velocity[None]
    )
    metrics = {
        "molmo": _trajectory_metrics(forecast, truth, truth_valid),
        "zero_motion": _trajectory_metrics(zero, truth, truth_valid),
        "constant_velocity": _trajectory_metrics(
            constant_velocity,
            truth,
            truth_valid,
        ),
    }
    predicted_motion = _motion_rms(forecast, anchor, truth_valid)
    true_motion = _motion_rms(truth, anchor, truth_valid)
    motion_ratio = _safe_ratio(predicted_motion, true_motion)
    object_points = np.asarray(object_points_m, dtype=float)
    observed_anchor = object_points[bundle.query.t0_frame, bundle.query.node_indices]
    anchor_rmse = float(np.sqrt(np.mean(np.square(anchor - observed_anchor))))
    first_frame = int(np.flatnonzero(np.any(truth_valid, axis=1))[0])
    first_valid = truth_valid[first_frame]
    first_step_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    (forecast[first_frame] - anchor)[first_valid]
                    - (truth[first_frame] - anchor)[first_valid]
                )
            )
        )
    )
    transformed = camera_to_world_points(
        bundle.future_camera_m[forecast_index],
        bundle.query.camera_to_world,
    )
    transform_error = float(
        np.max(np.abs(transformed - bundle.future_world_m[forecast_index]))
    )
    expected_stride = bundle.query.source_fps / thresholds.expected_forecast_fps
    temporal_contract = bool(
        np.isclose(bundle.query.forecast_fps, thresholds.expected_forecast_fps)
        and np.isclose(expected_stride, bundle.query.frame_stride)
        and np.all(np.diff(bundle.query.history_frame_indices) == bundle.query.frame_stride)
    )
    gates = {
        "beats_zero_motion": bool(
            metrics["molmo"]["ade_m"]
            <= thresholds.maximum_ade_ratio_vs_zero * metrics["zero_motion"]["ade_m"]
        ),
        "beats_constant_velocity": bool(
            metrics["molmo"]["ade_m"]
            <= thresholds.maximum_ade_ratio_vs_constant_velocity
            * metrics["constant_velocity"]["ade_m"]
        ),
        "motion_scale": bool(
            thresholds.minimum_motion_scale_ratio
            <= motion_ratio
            <= thresholds.maximum_motion_scale_ratio
        ),
        "query_anchor": bool(
            anchor_rmse <= thresholds.maximum_anchor_alignment_rmse_m
            and first_step_error <= thresholds.maximum_first_step_error_m
            and transform_error <= thresholds.maximum_frame_transform_error_m
        ),
        "temporal_sampling": temporal_contract,
    }
    return {
        "forecast_id": forecast_id,
        "caption": bundle.captions[forecast_index],
        "target_frame_indices": frame_indices.tolist(),
        "compared_future_frames": len(frame_indices),
        "metrics": metrics,
        "ratios": {
            "ade_vs_zero_motion": _safe_ratio(
                metrics["molmo"]["ade_m"],
                metrics["zero_motion"]["ade_m"],
            ),
            "ade_vs_constant_velocity": _safe_ratio(
                metrics["molmo"]["ade_m"],
                metrics["constant_velocity"]["ade_m"],
            ),
            "predicted_to_true_motion_scale": motion_ratio,
        },
        "motion": {
            "predicted_rms_m": predicted_motion,
            "true_rms_m": true_motion,
        },
        "anchor_and_frame": {
            "query_anchor_alignment_rmse_m": anchor_rmse,
            "first_step_displacement_error_m": first_step_error,
            "camera_to_world_max_abs_error_m": transform_error,
        },
        "temporal_sampling": {
            "source_fps": bundle.query.source_fps,
            "forecast_fps": bundle.query.forecast_fps,
            "frame_stride": bundle.query.frame_stride,
            "expected_forecast_fps": thresholds.expected_forecast_fps,
        },
        "gates": gates,
    }


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _component_semantic_scores(
    bank: JointRolloutBank,
    bundle: MolmoForecastBundle,
    forecast_id: str,
    query_indices: np.ndarray,
    thresholds: MolmoAcceptanceThresholds,
) -> np.ndarray:
    forecast_index = bundle.forecast_ids.index(forecast_id)
    available = min(
        bundle.future_horizon,
        (bank.frame_count - 1) // bundle.query.frame_stride,
    )
    frames = np.arange(1, available + 1) * bundle.query.frame_stride
    nodes = bundle.query.node_indices[query_indices]
    predicted = bank.trajectories[:, :, frames[:, None], nodes[None]]
    predicted_anchor = bank.trajectories[:, :, 0, nodes]
    predicted_displacement = predicted - predicted_anchor[:, :, None]
    target = np.transpose(
        bundle.future_world_m[forecast_index, query_indices, :available],
        (1, 0, 2),
    )
    target_displacement = (
        target - bundle.query.anchor_positions_world_m[query_indices][None]
    )
    standardized = (
        predicted_displacement - target_displacement[None, None]
    ) / thresholds.semantic_scale_m
    terms = -0.5 * (thresholds.semantic_degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / thresholds.semantic_degrees_of_freedom
    )
    return np.mean(terms, axis=(2, 3, 4))


def _action_ranking(
    bank: JointRolloutBank,
    component_scores: np.ndarray,
    correct_action_id: str,
    top_k: int,
) -> dict[str, Any]:
    action_ids = tuple(
        str(metadata["action"]["proposal_id"])
        for metadata in bank.hypothesis_metadata
    )
    scores: dict[str, float] = {}
    prior = bank.prior_joint_weights
    for action_id in sorted(set(action_ids)):
        hypothesis_mask = np.asarray([value == action_id for value in action_ids])
        selected_prior = prior[hypothesis_mask]
        selected_prior /= np.sum(selected_prior)
        values = np.log(np.maximum(selected_prior, 1e-300)) + component_scores[
            hypothesis_mask
        ]
        scores[action_id] = _logsumexp(values.reshape(-1))
    ranking = sorted(scores, key=lambda action: (-scores[action], action))
    rank = ranking.index(correct_action_id) + 1
    return {
        "ranking": ranking,
        "action_log_scores": scores,
        "top1_action_id": ranking[0],
        "correct_action_rank": rank,
        "correct_action_in_top_k": rank <= top_k,
    }


def ranking_and_stability_diagnostics(
    bank: JointRolloutBank,
    bank_manifest: Mapping[str, Any],
    bundle: MolmoForecastBundle,
    paraphrase_forecast_ids: Sequence[str],
    truth_motion_rms_m: float,
    thresholds: MolmoAcceptanceThresholds,
) -> dict[str, Any]:
    """Test language ranking across prompts and leave-one-query-out subsets."""

    prompt_ids = tuple(dict.fromkeys(map(str, paraphrase_forecast_ids)))
    missing = set(prompt_ids) - set(bundle.forecast_ids)
    if missing:
        raise ValueError(
            "unknown paraphrase forecast ids: " + ", ".join(sorted(missing))
        )
    observed = [
        str(entry["proposal_id"])
        for entry in bank_manifest.get("action_proposals", [])
        if bool(entry.get("future_action_observed"))
    ]
    if len(observed) != 1:
        raise ValueError("ranking benchmark requires exactly one labeled correct action")
    correct_action_id = observed[0]
    point_count = len(bundle.query.node_indices)
    subsets = {"all": np.arange(point_count, dtype=int)}
    if point_count >= 3:
        subsets.update(
            {
                f"leave_out_query_{index}": np.delete(np.arange(point_count), index)
                for index in range(point_count)
            }
        )
    results: dict[str, Any] = {}
    prompt_full_top1 = []
    prompt_top_k = []
    subset_top1_agreements = []
    subset_top_k_recalls = []
    for prompt_id in prompt_ids:
        prompt_results = {}
        for subset_id, query_indices in subsets.items():
            component_scores = _component_semantic_scores(
                bank,
                bundle,
                prompt_id,
                query_indices,
                thresholds,
            )
            prompt_results[subset_id] = _action_ranking(
                bank,
                component_scores,
                correct_action_id,
                thresholds.ranking_top_k,
            )
        full = prompt_results["all"]
        leave_out = [
            result for subset_id, result in prompt_results.items() if subset_id != "all"
        ]
        top1_agreement = (
            float(
                np.mean(
                    [
                        result["top1_action_id"] == full["top1_action_id"]
                        for result in leave_out
                    ]
                )
            )
            if leave_out
            else 0.0
        )
        top_k_recall = (
            float(np.mean([result["correct_action_in_top_k"] for result in leave_out]))
            if leave_out
            else 0.0
        )
        results[prompt_id] = {
            "caption": bundle.captions[bundle.forecast_ids.index(prompt_id)],
            "subsets": prompt_results,
            "leave_one_query_out_top1_agreement": top1_agreement,
            "leave_one_query_out_correct_top_k_recall": top_k_recall,
        }
        prompt_full_top1.append(full["top1_action_id"])
        prompt_top_k.append(bool(full["correct_action_in_top_k"]))
        subset_top1_agreements.append(top1_agreement)
        subset_top_k_recalls.append(top_k_recall)

    mode_count = max(Counter(prompt_full_top1).values(), default=0)
    prompt_top1_agreement = mode_count / max(len(prompt_full_top1), 1)
    paraphrase_top_k_recall = float(np.mean(prompt_top_k)) if prompt_top_k else 0.0
    pairwise = {}
    pairwise_ratios = []
    for left_id, right_id in combinations(prompt_ids, 2):
        left_index = bundle.forecast_ids.index(left_id)
        right_index = bundle.forecast_ids.index(right_id)
        left = bundle.future_world_m[left_index] - bundle.query.anchor_positions_world_m[:, None]
        right = bundle.future_world_m[right_index] - bundle.query.anchor_positions_world_m[:, None]
        pairwise_rms = float(np.sqrt(np.mean(np.sum(np.square(left - right), axis=2))))
        ratio = pairwise_rms / max(truth_motion_rms_m, 1e-12)
        pairwise[f"{left_id}__{right_id}"] = {
            "displacement_vector_rms_m": pairwise_rms,
            "ratio_to_true_motion_rms": ratio,
        }
        pairwise_ratios.append(ratio)
    maximum_pairwise_ratio = max(pairwise_ratios, default=0.0)
    minimum_subset_top1 = min(subset_top1_agreements, default=0.0)
    minimum_subset_top_k = min(subset_top_k_recalls, default=0.0)
    primary_rank = (
        results[prompt_ids[0]]["subsets"]["all"]["correct_action_rank"]
        if prompt_ids
        else None
    )
    gates = {
        "correct_rollout_ranking": bool(
            primary_rank is not None
            and primary_rank <= thresholds.required_correct_action_rank
            and paraphrase_top_k_recall >= thresholds.minimum_paraphrase_top_k_recall
        ),
        "prompt_and_query_stability": bool(
            len(prompt_ids) >= thresholds.minimum_paraphrases
            and prompt_top1_agreement >= thresholds.minimum_prompt_top1_agreement
            and maximum_pairwise_ratio
            <= thresholds.maximum_prompt_pairwise_motion_ratio
            and minimum_subset_top1
            >= thresholds.minimum_query_subset_top1_agreement
            and minimum_subset_top_k
            >= thresholds.minimum_query_subset_top_k_recall
        ),
    }
    return {
        "correct_action_id": correct_action_id,
        "semantic_action_prior_removed": True,
        "physical_bank_contract": (
            "all candidates completed PhysTwin simulation; hardware feasibility is not asserted"
        ),
        "prompt_ids": list(prompt_ids),
        "prompt_results": results,
        "summary": {
            "primary_correct_action_rank": primary_rank,
            "paraphrase_correct_top_k_recall": paraphrase_top_k_recall,
            "prompt_top1_mode_agreement": prompt_top1_agreement,
            "maximum_prompt_pairwise_motion_ratio": maximum_pairwise_ratio,
            "minimum_query_subset_top1_agreement": minimum_subset_top1,
            "minimum_query_subset_correct_top_k_recall": minimum_subset_top_k,
        },
        "prompt_pairwise_motion": pairwise,
        "gates": gates,
    }


def evaluate_molmo_acceptance_case(
    *,
    case_id: str,
    bundle: MolmoForecastBundle,
    object_points_m: np.ndarray,
    validity: np.ndarray,
    bank: JointRolloutBank,
    bank_manifest: Mapping[str, Any],
    primary_forecast_id: str,
    paraphrase_forecast_ids: Sequence[str],
    thresholds: MolmoAcceptanceThresholds,
) -> dict[str, Any]:
    if bundle.query.case_name != bank_manifest.get("case", bundle.query.case_name):
        raise ValueError("Molmo forecast and physical rollout bank cases differ")
    direct = forecast_competence_diagnostics(
        bundle,
        primary_forecast_id,
        object_points_m,
        validity,
        thresholds,
    )
    ranking = ranking_and_stability_diagnostics(
        bank,
        bank_manifest,
        bundle,
        paraphrase_forecast_ids,
        direct["motion"]["true_rms_m"],
        thresholds,
    )
    gates = {**direct["gates"], **ranking["gates"]}
    return {
        "case_id": case_id,
        "checkpoint": bundle.checkpoint,
        "primary_forecast_id": primary_forecast_id,
        "direct_forecast": direct,
        "ranking_and_stability": ranking,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def aggregate_molmo_acceptance(
    cases: Sequence[Mapping[str, Any]],
    thresholds: MolmoAcceptanceThresholds,
) -> dict[str, Any]:
    values = tuple(cases)
    if not values:
        raise ValueError("Molmo acceptance requires at least one evaluated case")
    passed = sum(bool(case["passed"]) for case in values)
    pass_fraction = passed / len(values)
    sufficient_cases = len(values) >= thresholds.minimum_independent_cases
    accepted = bool(
        sufficient_cases and pass_fraction >= thresholds.minimum_case_pass_fraction
    )
    reasons = []
    if not accepted:
        if not sufficient_cases:
            reasons.append("insufficient_independent_source_cases")
        failed_gates = sorted(
            {
                gate
                for case in values
                for gate, status in case["gates"].items()
                if not status
            }
        )
        reasons.extend(f"failed_gate:{gate}" for gate in failed_gates)
    return {
        "accepted_for_semantic_reweighting": accepted,
        "decision": (
            "semantic_reweighting_may_enter_source_beta_selection"
            if accepted
            else "keep_beta_zero_and_exclude_semantic_improvement_claim"
        ),
        "independent_case_count": len(values),
        "minimum_independent_cases": thresholds.minimum_independent_cases,
        "passed_case_count": passed,
        "case_pass_fraction": pass_fraction,
        "minimum_case_pass_fraction": thresholds.minimum_case_pass_fraction,
        "safe_fallback_frequency": 1.0 - pass_fraction,
        "blocking_reasons": reasons,
    }
