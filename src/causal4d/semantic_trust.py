"""Source-validated and OOD-gated trust for semantic trajectory priors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal4d.contracts import PhysicalPosterior, TaskPosterior
from causal4d.semantic_posterior import (
    SparseSemanticEvidence,
    build_task_posterior,
    query_point_readout,
    task_posterior_mean,
)


@dataclass(frozen=True)
class SemanticValidationCase:
    """One source-only case used to select semantic trust."""

    case_id: str
    physical: PhysicalPosterior
    evidence: SparseSemanticEvidence
    truth_m: np.ndarray
    mask: np.ndarray | None = None
    start_frame: int = 1

    def __post_init__(self) -> None:
        truth = np.asarray(self.truth_m, dtype=float)
        if not self.case_id:
            raise ValueError("semantic validation case_id must be nonempty")
        if truth.shape != self.physical.readout_trajectories_m.shape[1:]:
            raise ValueError("semantic validation truth must match physical trajectories")
        if not 0 <= self.start_frame < len(truth):
            raise ValueError("semantic validation start_frame is invalid")
        mask = None
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape == truth.shape:
                mask = np.all(mask, axis=2)
            if mask.shape != truth.shape[:2]:
                raise ValueError("semantic validation mask has an invalid shape")
        object.__setattr__(self, "truth_m", truth)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class SemanticTrustCalibration:
    """Trust settings selected without reading any target future."""

    selected_beta: float
    beta_candidates: tuple[float, ...]
    source_case_ids: tuple[str, ...]
    source_mean_rmse_m: tuple[float, ...]
    physical_mean_rmse_m: float
    maximum_support_distance_m: float
    maximum_anchor_error_m: float
    minimum_semantic_motion_m: float
    minimum_motion_ratio: float
    maximum_motion_ratio: float
    minimum_relative_improvement: float

    def __post_init__(self) -> None:
        if not self.beta_candidates or 0.0 not in self.beta_candidates:
            raise ValueError("beta_candidates must include zero")
        if len(self.source_mean_rmse_m) != len(self.beta_candidates):
            raise ValueError("source RMSE must identify every beta candidate")
        if self.selected_beta not in self.beta_candidates:
            raise ValueError("selected_beta must be a beta candidate")
        if not self.source_case_ids or len(set(self.source_case_ids)) != len(
            self.source_case_ids
        ):
            raise ValueError("source_case_ids must be nonempty and unique")
        positive = (
            self.maximum_support_distance_m,
            self.maximum_anchor_error_m,
            self.minimum_semantic_motion_m,
            self.maximum_motion_ratio,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("semantic trust thresholds must be finite and positive")
        if not 0.0 <= self.minimum_motion_ratio < self.maximum_motion_ratio:
            raise ValueError("semantic motion-ratio thresholds are invalid")

    @property
    def calibration_id(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SemanticTrustDecision:
    """Target-side decision based only on source settings and OOD diagnostics."""

    calibration_id: str
    selected_beta: float
    applied_beta: float
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, float]


def _valid_coordinates(case: SemanticValidationCase) -> np.ndarray:
    valid = np.all(np.isfinite(case.truth_m), axis=2)
    if case.mask is not None:
        valid &= case.mask
    valid[: case.start_frame] = False
    if not np.any(valid):
        raise ValueError(f"source case {case.case_id!r} has no valid target coordinates")
    return np.repeat(valid[:, :, None], 3, axis=2)


def _case_rmse(case: SemanticValidationCase, prediction: np.ndarray) -> float:
    valid = _valid_coordinates(case)
    return float(np.sqrt(np.mean(np.square((prediction - case.truth_m)[valid]))))


def semantic_ood_diagnostics(
    physical: PhysicalPosterior,
    evidence: SparseSemanticEvidence,
) -> dict[str, float]:
    """Measure target forecast motion and distance from physical support."""

    predicted = query_point_readout(
        physical,
        evidence.node_indices,
        evidence.physical_frame_indices,
    ).astype(float)
    semantic = evidence.positions_m.astype(float)
    physical_anchor = physical.readout_trajectories_m[
        :, evidence.anchor_physical_frame, evidence.node_indices
    ].astype(float)
    if evidence.compare_displacements:
        predicted = predicted - physical_anchor[:, None]
        semantic = semantic - evidence.anchor_positions_m[None]
    else:
        predicted = predicted - predicted[:, :1]
        semantic = semantic - semantic[:1]
    valid = np.asarray(evidence.valid, dtype=bool)
    semantic_motion = float(
        np.sqrt(np.mean(np.square(semantic[valid])))
    )
    physical_mean = np.einsum("k,kfqc->fqc", physical.weights, predicted)
    physical_motion = float(np.sqrt(np.mean(np.square(physical_mean[valid]))))
    component_distance = np.empty(len(physical.weights), dtype=float)
    for component in range(len(physical.weights)):
        component_distance[component] = np.sqrt(
            np.mean(np.square((predicted[component] - semantic)[valid]))
        )
    weighted_anchor = np.einsum("k,kqc->qc", physical.weights, physical_anchor)
    anchor_error = (
        float(
            np.mean(
                np.linalg.norm(
                    weighted_anchor - evidence.anchor_positions_m,
                    axis=1,
                )
            )
        )
        if evidence.anchor_positions_m is not None
        else 0.0
    )
    return {
        "semantic_motion_rms_m": semantic_motion,
        "physical_motion_rms_m": physical_motion,
        "semantic_to_physical_motion_ratio": semantic_motion
        / max(physical_motion, 1e-12),
        "minimum_physical_support_distance_m": float(np.min(component_distance)),
        "anchor_error_m": anchor_error,
    }


def fit_semantic_trust_calibration(
    source_cases: Sequence[SemanticValidationCase],
    *,
    beta_candidates: Sequence[float] = (0.0, 1.0, 3.0, 6.0, 12.0),
    minimum_relative_improvement: float = 0.0,
    support_margin: float = 1.5,
) -> SemanticTrustCalibration:
    """Select beta and OOD thresholds using source futures only."""

    cases = tuple(source_cases)
    candidates = tuple(sorted(set(map(float, beta_candidates))))
    if not cases:
        raise ValueError("semantic trust calibration requires source cases")
    if not candidates or candidates[0] != 0.0 or any(value < 0.0 for value in candidates):
        raise ValueError("beta candidates must be nonnegative and include zero")
    if minimum_relative_improvement < 0.0 or support_margin < 1.0:
        raise ValueError("calibration margins are invalid")
    physical_errors = [
        _case_rmse(
            case,
            np.einsum(
                "k,ktnc->tnc",
                case.physical.weights,
                case.physical.readout_trajectories_m,
            ),
        )
        for case in cases
    ]
    errors_by_beta = []
    for beta in candidates:
        errors = []
        for case in cases:
            task = build_task_posterior(case.physical, case.evidence, beta=beta)
            errors.append(_case_rmse(case, task_posterior_mean(case.physical, task)))
        errors_by_beta.append(float(np.mean(errors)))
    physical_mean = float(np.mean(physical_errors))
    best_index = min(
        range(len(candidates)),
        key=lambda index: (errors_by_beta[index], candidates[index]),
    )
    relative_improvement = 1.0 - errors_by_beta[best_index] / physical_mean
    selected_beta = (
        candidates[best_index]
        if relative_improvement >= minimum_relative_improvement
        else 0.0
    )
    diagnostics = [semantic_ood_diagnostics(case.physical, case.evidence) for case in cases]
    semantic_motion = [value["semantic_motion_rms_m"] for value in diagnostics]
    motion_ratios = [value["semantic_to_physical_motion_ratio"] for value in diagnostics]
    support = [value["minimum_physical_support_distance_m"] for value in diagnostics]
    anchor = [value["anchor_error_m"] for value in diagnostics]
    return SemanticTrustCalibration(
        selected_beta=selected_beta,
        beta_candidates=candidates,
        source_case_ids=tuple(case.case_id for case in cases),
        source_mean_rmse_m=tuple(errors_by_beta),
        physical_mean_rmse_m=physical_mean,
        maximum_support_distance_m=max(max(support) * support_margin, 1e-4),
        maximum_anchor_error_m=max(max(anchor) * support_margin + 0.005, 0.005),
        minimum_semantic_motion_m=max(min(semantic_motion) * 0.25, 1e-4),
        minimum_motion_ratio=max(min(motion_ratios) * 0.5, 0.05),
        maximum_motion_ratio=max(max(motion_ratios) * 2.0, 2.0),
        minimum_relative_improvement=minimum_relative_improvement,
    )


def apply_adaptive_semantic_trust(
    physical: PhysicalPosterior,
    evidence: SparseSemanticEvidence,
    calibration: SemanticTrustCalibration,
) -> tuple[TaskPosterior, SemanticTrustDecision]:
    """Gate source-selected beta using target-side, label-free OOD checks."""

    diagnostics = semantic_ood_diagnostics(physical, evidence)
    reasons = []
    if calibration.selected_beta == 0.0:
        reasons.append("no_source_validation_gain")
    if diagnostics["semantic_motion_rms_m"] < calibration.minimum_semantic_motion_m:
        reasons.append("static_semantic_forecast")
    ratio = diagnostics["semantic_to_physical_motion_ratio"]
    if ratio < calibration.minimum_motion_ratio:
        reasons.append("semantic_motion_too_small")
    if ratio > calibration.maximum_motion_ratio:
        reasons.append("semantic_motion_too_large")
    if (
        diagnostics["minimum_physical_support_distance_m"]
        > calibration.maximum_support_distance_m
    ):
        reasons.append("outside_physical_support")
    if diagnostics["anchor_error_m"] > calibration.maximum_anchor_error_m:
        reasons.append("anchor_misalignment")
    applied_beta = 0.0 if reasons else calibration.selected_beta
    task = build_task_posterior(physical, evidence, beta=applied_beta)
    if applied_beta == 0.0 and not np.array_equal(task.task_weights, physical.weights):
        raise RuntimeError("semantic rejection failed to preserve physical weights")
    decision = SemanticTrustDecision(
        calibration_id=calibration.calibration_id,
        selected_beta=calibration.selected_beta,
        applied_beta=applied_beta,
        accepted=not reasons,
        reasons=tuple(reasons),
        diagnostics=diagnostics,
    )
    return task, decision


def save_semantic_trust_calibration(
    path: str | Path,
    calibration: SemanticTrustCalibration,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(calibration), "calibration_id": calibration.calibration_id}
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_semantic_trust_calibration(path: str | Path) -> SemanticTrustCalibration:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_id = str(payload.pop("calibration_id"))
    payload["beta_candidates"] = tuple(map(float, payload["beta_candidates"]))
    payload["source_case_ids"] = tuple(map(str, payload["source_case_ids"]))
    payload["source_mean_rmse_m"] = tuple(map(float, payload["source_mean_rmse_m"]))
    calibration = SemanticTrustCalibration(**payload)
    if calibration.calibration_id != expected_id:
        raise ValueError("semantic trust calibration digest does not match")
    return calibration
