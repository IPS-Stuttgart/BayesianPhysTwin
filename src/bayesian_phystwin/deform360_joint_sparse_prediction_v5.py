"""Numerical prediction kernel for the Deform360 joint-sparse v5 study.

This module deliberately contains no dataset I/O and no outcome scoring.  It
consumes one already validated, causal observation batch and propagates the
posterior state coefficients through a caller-supplied physical Jacobian.  The
separation keeps future geometry out of prediction construction and makes exact
fallback testable before the source suffix is opened.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import GaugeAwareBeliefResult, GaugeAwareObservationBatch
from ._portable_contracts import source_artifact_mapping
from ._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from .prior_aware_gauge_belief import update_prior_aware_gauge_belief

B0_PHYSICAL_FALLBACK: Final = "B0_physical_fallback"
B1_LAST_CAUSAL_RESIDUAL: Final = "B1_last_causal_residual"
V1_VISUAL_GUARDED: Final = "V1_joint_sparse_visual_guarded"
T1_CONTACT_ONLY: Final = "T1_contact_anchor_only"
VT2_VISUOTACTILE_UNGUARDED: Final = "VT2_joint_sparse_visuotactile_unguarded"
VT3_VISUOTACTILE_ANCHOR_BIAS: Final = (
    "VT3_joint_sparse_visuotactile_anchor_bias"
)
RAW_METHOD_IDS: Final = (
    B0_PHYSICAL_FALLBACK,
    B1_LAST_CAUSAL_RESIDUAL,
    V1_VISUAL_GUARDED,
    T1_CONTACT_ONLY,
    VT2_VISUOTACTILE_UNGUARDED,
    VT3_VISUOTACTILE_ANCHOR_BIAS,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly_float64(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "iuf", f"{name} must be real")
    result = np.asarray(raw, dtype=np.float64, order="C")
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_floating(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.dtype in {np.dtype(np.float32), np.dtype(np.float64)},
        f"{name} must use float32 or float64",
    )
    result = np.array(raw, dtype=raw.dtype, order="C", copy=True)
    _require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _optional_array_sha256(value: np.ndarray | None) -> str | None:
    return None if value is None else _array_sha256(value)


def _observation_batch_descriptor(
    batch: GaugeAwareObservationBatch,
) -> dict[str, Any]:
    arrays = {
        "innovation_m": batch.innovation_m,
        "observation_covariance_m2": batch.observation_covariance_m2,
        "state_jacobian": batch.state_jacobian,
        "gauge_jacobian": batch.gauge_jacobian,
        "shared_bias_jacobian": batch.shared_bias_jacobian,
        "view_bias_jacobian": batch.view_bias_jacobian,
        "query_state_jacobian": batch.query_state_jacobian,
        "gauge_prior_covariance": batch.gauge_prior_covariance,
        "prior_reliability": batch.prior_reliability,
        "prior_nominal_probability": batch.prior_nominal_probability,
        "composite_weight": batch.composite_weight,
        "association_probability": batch.association_probability,
    }
    optional_arrays = {
        "state_prior_covariance_m2": batch.state_prior_covariance_m2,
        "anchor_innovation_m": batch.anchor_innovation_m,
        "anchor_covariance_m2": batch.anchor_covariance_m2,
        "anchor_state_jacobian": batch.anchor_state_jacobian,
        "anchor_prior_reliability": batch.anchor_prior_reliability,
        "anchor_prior_nominal_probability": batch.anchor_prior_nominal_probability,
        "anchor_composite_weight": batch.anchor_composite_weight,
        "anchor_bias_jacobian": batch.anchor_bias_jacobian,
        "anchor_bias_prior_covariance": batch.anchor_bias_prior_covariance,
    }
    return {
        "schema": "bayesian-phystwin.gauge-aware-observation-batch-binding",
        "schema_version": 1,
        "array_sha256": {
            name: _array_sha256(np.asarray(value))
            for name, value in sorted(arrays.items())
        },
        "optional_array_sha256": {
            name: _optional_array_sha256(value)
            for name, value in sorted(optional_arrays.items())
        },
        "correlation_group_ids": list(batch.correlation_group_ids),
        "anchor_correlation_group_ids": (
            None
            if batch.anchor_correlation_group_ids is None
            else list(batch.anchor_correlation_group_ids)
        ),
        "physical_response_scale_m": float(batch.physical_response_scale_m),
        "composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        "metadata": plain_json(batch.metadata),
    }


def _belief_result_descriptor(result: GaugeAwareBeliefResult) -> dict[str, Any]:
    arrays = {
        "state_coefficients": result.state_coefficients,
        "gauge_delta": result.gauge_delta,
        "shared_bias_coefficients": result.shared_bias_coefficients,
        "view_bias_coefficients": result.view_bias_coefficients,
        "anchor_bias_coefficients": result.anchor_bias_coefficients,
        "posterior_covariance": result.posterior_covariance,
        "identifiable_state_transform": result.identifiable_state_transform,
        "identifiable_fractions": result.identifiable_fractions,
        "query_sensitivity_fractions": result.query_sensitivity_fractions,
        "robust_weights": result.robust_weights,
        "anchor_robust_weights": result.anchor_robust_weights,
    }
    return {
        "inference_admissible": result.inference_admissible,
        "reason": result.reason,
        "array_sha256": {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        },
        "diagnostics": plain_json(result.diagnostics),
        "input_lineage": plain_json(result.input_lineage),
    }


@dataclass(frozen=True, slots=True)
class Deform360JointSparsePredictionInputV5:
    """One outcome-blind object prediction problem.

    ``future_state_jacobian_m`` maps the inferred state coefficients into the
    physical graph at every registered frame.  It may depend on the known robot
    action and the physical rollout, but never on a future object observation.
    """

    object_id: str
    episode_id: int
    stratum: str
    physical_prediction_m: np.ndarray
    persistence_m: np.ndarray
    last_causal_residual_m: np.ndarray
    future_state_jacobian_m: np.ndarray
    observation_batch: GaugeAwareObservationBatch
    causal_frame_stop: int
    evaluation_frame_range_half_open: tuple[int, int]
    factor_admitted: bool
    physical_mode: str
    source_artifact_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        _require(
            isinstance(self.object_id, str) and bool(self.object_id),
            "object_id must be nonempty",
        )
        _require(
            type(self.episode_id) is int and self.episode_id >= 0,
            "episode_id must be a nonnegative integer",
        )
        _require(self.stratum in {"sheet", "volumetric"}, "invalid stratum")
        _require(type(self.factor_admitted) is bool, "factor_admitted must be Boolean")
        _require(
            isinstance(self.physical_mode, str) and bool(self.physical_mode),
            "physical_mode must be nonempty",
        )
        _require(
            isinstance(self.observation_batch, GaugeAwareObservationBatch),
            "observation_batch must be a GaugeAwareObservationBatch",
        )
        _require(
            type(self.causal_frame_stop) is int and self.causal_frame_stop >= 1,
            "causal_frame_stop must be a positive integer",
        )
        _require(
            type(self.evaluation_frame_range_half_open) is tuple
            and len(self.evaluation_frame_range_half_open) == 2
            and all(
                type(value) is int
                for value in self.evaluation_frame_range_half_open
            ),
            "evaluation frame range must be a pair of exact integers",
        )

        physical = _readonly_floating(
            self.physical_prediction_m,
            name="physical_prediction_m",
            ndim=3,
        )
        persistence = _readonly_floating(
            self.persistence_m,
            name="persistence_m",
            ndim=3,
        )
        residual = _readonly_float64(
            self.last_causal_residual_m,
            name="last_causal_residual_m",
            ndim=2,
        )
        propagation = _readonly_float64(
            self.future_state_jacobian_m,
            name="future_state_jacobian_m",
            ndim=4,
        )
        _require(
            physical.shape == persistence.shape
            and physical.shape[0] > 0
            and physical.shape[1] > 0
            and physical.shape[2] == 3,
            "physical and persistence trajectories must have matching shape (T,N,3)",
        )
        _require(
            physical.dtype == persistence.dtype,
            "physical and persistence trajectories must use the same dtype",
        )
        evaluation_start, evaluation_stop = self.evaluation_frame_range_half_open
        _require(
            self.causal_frame_stop == evaluation_start
            and evaluation_start < evaluation_stop <= len(physical),
            "evaluation must begin at the causal cutoff and stay in the trajectory",
        )
        _require(
            self.physical_mode in {"warp_twin", "persistence_fallback"},
            "physical_mode changed",
        )
        if self.physical_mode == "persistence_fallback":
            _require(
                np.array_equal(physical, persistence),
                "persistence fallback must be byte-equivalent to persistence",
            )
        _require(
            residual.shape == physical.shape[1:],
            "last causal residual must have shape (N,3)",
        )
        state_count = self.observation_batch.state_jacobian.shape[2]
        _require(
            propagation.shape == (*physical.shape, state_count),
            "future state Jacobian must have shape (T,N,3,S)",
        )
        _require(
            np.array_equal(
                propagation[: self.causal_frame_stop],
                np.zeros_like(propagation[: self.causal_frame_stop]),
            ),
            "state propagation must be exactly zero before the causal cutoff",
        )
        endpoint_query = propagation[evaluation_start:evaluation_stop].reshape(
            -1, 3, state_count
        )
        _require(
            self.observation_batch.query_state_jacobian.shape == endpoint_query.shape
            and np.array_equal(
                self.observation_batch.query_state_jacobian,
                endpoint_query,
            ),
            "observation query Jacobian must exactly bind the propagated trajectory",
        )
        _require(
            isinstance(self.source_artifact_ids, Mapping)
            and bool(self.source_artifact_ids),
            "source_artifact_ids must be a nonempty mapping",
        )
        sources = source_artifact_mapping(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )

        object.__setattr__(self, "physical_prediction_m", physical)
        object.__setattr__(self, "persistence_m", persistence)
        object.__setattr__(self, "last_causal_residual_m", residual)
        object.__setattr__(self, "future_state_jacobian_m", propagation)
        object.__setattr__(
            self,
            "evaluation_frame_range_half_open",
            (evaluation_start, evaluation_stop),
        )
        object.__setattr__(self, "source_artifact_ids", sources)

    @property
    def input_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian-phystwin.deform360-joint-sparse-prediction-input",
                "schema_version": 1,
                "object_id": self.object_id,
                "episode_id": self.episode_id,
                "stratum": self.stratum,
                "factor_admitted": self.factor_admitted,
                "physical_mode": self.physical_mode,
                "causal_frame_stop": self.causal_frame_stop,
                "evaluation_frame_range_half_open": list(
                    self.evaluation_frame_range_half_open
                ),
                "physical_prediction_sha256": _array_sha256(
                    self.physical_prediction_m
                ),
                "persistence_sha256": _array_sha256(self.persistence_m),
                "last_causal_residual_sha256": _array_sha256(
                    self.last_causal_residual_m
                ),
                "future_state_jacobian_sha256": _array_sha256(
                    self.future_state_jacobian_m
                ),
                "observation_batch": _observation_batch_descriptor(
                    self.observation_batch
                ),
                "source_artifact_ids": dict(self.source_artifact_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class Deform360JointSparsePredictionResultV5:
    """Raw method trajectories and prefix-only diagnostics for one object."""

    input_id: str
    trajectories_m: Mapping[str, np.ndarray]
    inference_results: Mapping[str, GaugeAwareBeliefResult]
    risk_score: float
    predicted_loss_features_m: Mapping[str, float]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(
            isinstance(self.input_id, str) and len(self.input_id) == 64,
            "input_id must be a SHA-256 digest",
        )
        _require(
            set(self.trajectories_m) == set(RAW_METHOD_IDS),
            "raw trajectory roster changed",
        )
        shape: tuple[int, ...] | None = None
        dtype: np.dtype[Any] | None = None
        trajectories: dict[str, np.ndarray] = {}
        for method_id in RAW_METHOD_IDS:
            array = _readonly_floating(
                self.trajectories_m[method_id],
                name=f"trajectory {method_id}",
                ndim=3,
            )
            shape = array.shape if shape is None else shape
            dtype = array.dtype if dtype is None else dtype
            _require(array.shape == shape, "raw trajectory shapes differ")
            _require(array.dtype == dtype, "raw trajectory dtypes differ")
            trajectories[method_id] = array
        _require(
            set(self.inference_results)
            == {
                V1_VISUAL_GUARDED,
                T1_CONTACT_ONLY,
                VT2_VISUOTACTILE_UNGUARDED,
                VT3_VISUOTACTILE_ANCHOR_BIAS,
            },
            "inference result roster changed",
        )
        _require(
            np.isfinite(self.risk_score) and self.risk_score >= 0.0,
            "risk_score must be finite and nonnegative",
        )
        _require(
            set(self.predicted_loss_features_m) == set(RAW_METHOD_IDS),
            "predicted-loss feature roster changed",
        )
        features: dict[str, float] = {}
        for method_id, value in self.predicted_loss_features_m.items():
            _require(
                np.isfinite(value) and value >= 0.0,
                f"predicted-loss feature {method_id} is invalid",
            )
            features[method_id] = float(value)
        _require(
            isinstance(self.diagnostics, Mapping),
            "diagnostics must be a mapping",
        )
        diagnostics = frozen_finite_json_mapping(
            self.diagnostics,
            name="prediction diagnostics",
        )
        object.__setattr__(
            self,
            "trajectories_m",
            MappingProxyType(trajectories),
        )
        object.__setattr__(
            self,
            "inference_results",
            MappingProxyType(dict(self.inference_results)),
        )
        object.__setattr__(self, "risk_score", float(self.risk_score))
        object.__setattr__(
            self,
            "predicted_loss_features_m",
            MappingProxyType(features),
        )
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def result_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "bayesian-phystwin.deform360-joint-sparse-prediction-result",
                "schema_version": 1,
                "input_id": self.input_id,
                "trajectory_sha256": {
                    method_id: _array_sha256(self.trajectories_m[method_id])
                    for method_id in RAW_METHOD_IDS
                },
                "inference": {
                    method_id: _belief_result_descriptor(result)
                    for method_id, result in sorted(self.inference_results.items())
                },
                "risk_score": self.risk_score,
                "predicted_loss_features_m": dict(self.predicted_loss_features_m),
                "diagnostics": plain_json(self.diagnostics),
            }
        )


def _without_anchor(batch: GaugeAwareObservationBatch) -> GaugeAwareObservationBatch:
    return replace(
        batch,
        anchor_innovation_m=None,
        anchor_covariance_m2=None,
        anchor_state_jacobian=None,
        anchor_correlation_group_ids=None,
        anchor_prior_reliability=None,
        anchor_prior_nominal_probability=None,
        anchor_composite_weight=None,
        anchor_bias_jacobian=None,
        anchor_bias_prior_covariance=None,
    )


def _without_anchor_bias(
    batch: GaugeAwareObservationBatch,
) -> GaugeAwareObservationBatch:
    return replace(
        batch,
        anchor_bias_jacobian=None,
        anchor_bias_prior_covariance=None,
    )


def _contact_only(batch: GaugeAwareObservationBatch) -> GaugeAwareObservationBatch:
    return replace(
        _without_anchor_bias(batch),
        prior_reliability=np.zeros(len(batch.innovation_m), dtype=np.float64),
    )


def _propagate(
    baseline: np.ndarray,
    jacobian: np.ndarray,
    result: GaugeAwareBeliefResult,
    *,
    frame_range: tuple[int, int],
) -> np.ndarray:
    output = baseline.copy()
    if not result.inference_admissible:
        return output
    start, stop = frame_range
    correction = np.einsum(
        "tncs,s->tnc",
        jacobian[start:stop],
        result.state_coefficients,
        optimize=True,
    )
    output[start:stop] = np.asarray(
        baseline[start:stop] + correction,
        dtype=baseline.dtype,
    )
    return output


def _query_uncertainty_rms_m(
    jacobian: np.ndarray,
    result: GaugeAwareBeliefResult,
) -> float:
    state_count = jacobian.shape[-1]
    covariance = np.asarray(result.posterior_covariance[:state_count, :state_count])
    variances = np.einsum(
        "tnci,ij,tncj->tnc",
        jacobian,
        covariance,
        jacobian,
        optimize=True,
    )
    return float(np.sqrt(np.mean(np.maximum(variances, 0.0))))


def _correction_rms_m(candidate: np.ndarray, baseline: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(candidate - baseline), axis=2))))


def _normalized_rejection(result: GaugeAwareBeliefResult) -> float:
    weights = np.concatenate((result.robust_weights, result.anchor_robust_weights))
    if not len(weights):
        return 1.0
    finite = weights[np.isfinite(weights)]
    if not len(finite):
        return 1.0
    return float(1.0 - np.clip(np.mean(finite), 0.0, 1.0))


def run_deform360_joint_sparse_prediction_v5(
    problem: Deform360JointSparsePredictionInputV5,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> Deform360JointSparsePredictionResultV5:
    """Produce every registered raw arm from causal evidence only."""

    if not isinstance(problem, Deform360JointSparsePredictionInputV5):
        raise TypeError("problem must be a Deform360JointSparsePredictionInputV5")
    cfg = config or PriorAwareGaugeConfigV1(
        effective_samples_per_anchor_correlation_group=1.0
    )
    batch = problem.observation_batch
    contact_available = batch.anchor_innovation_m is not None
    visual_batch = _without_anchor(batch)
    joint_batch = _without_anchor_bias(batch)
    contact_batch = _contact_only(batch)

    visual_result = update_prior_aware_gauge_belief(visual_batch, config=cfg)
    contact_result = update_prior_aware_gauge_belief(contact_batch, config=cfg)
    joint_result = update_prior_aware_gauge_belief(joint_batch, config=cfg)
    biased_result = update_prior_aware_gauge_belief(batch, config=cfg)
    results = {
        V1_VISUAL_GUARDED: visual_result,
        T1_CONTACT_ONLY: contact_result,
        VT2_VISUOTACTILE_UNGUARDED: joint_result,
        VT3_VISUOTACTILE_ANCHOR_BIAS: biased_result,
    }

    baseline = np.asarray(problem.physical_prediction_m)
    evaluation_range = problem.evaluation_frame_range_half_open
    evaluation_start, evaluation_stop = evaluation_range
    residual_trajectory = baseline.copy()
    residual_trajectory[evaluation_start:evaluation_stop] = np.asarray(
        baseline[evaluation_start:evaluation_stop]
        + problem.last_causal_residual_m[None],
        dtype=baseline.dtype,
    )
    trajectories: dict[str, np.ndarray] = {
        B0_PHYSICAL_FALLBACK: baseline.copy(),
        B1_LAST_CAUSAL_RESIDUAL: residual_trajectory,
    }
    for method_id, result in results.items():
        eligible = result.inference_admissible and (
            problem.factor_admitted or method_id == T1_CONTACT_ONLY
        )
        if method_id in {
            T1_CONTACT_ONLY,
            VT2_VISUOTACTILE_UNGUARDED,
            VT3_VISUOTACTILE_ANCHOR_BIAS,
        }:
            eligible = eligible and contact_available
        trajectories[method_id] = (
            _propagate(
                baseline,
                problem.future_state_jacobian_m,
                result,
                frame_range=evaluation_range,
            )
            if eligible
            else baseline.copy()
        )

    response_scale = max(
        float(batch.physical_response_scale_m),
        float(np.finfo(float).eps),
    )
    joint_trajectory = trajectories[VT2_VISUOTACTILE_UNGUARDED]
    joint_correction = _correction_rms_m(
        joint_trajectory[evaluation_start:evaluation_stop],
        baseline[evaluation_start:evaluation_stop],
    )
    joint_uncertainty = _query_uncertainty_rms_m(
        problem.future_state_jacobian_m[evaluation_start:evaluation_stop],
        joint_result,
    )
    disagreement = float(
        np.linalg.norm(visual_result.state_coefficients - contact_result.state_coefficients)
        / np.sqrt(len(joint_result.state_coefficients))
    )
    risk_score = (
        joint_correction / response_scale
        + joint_uncertainty / response_scale
        + disagreement / response_scale
        + _normalized_rejection(joint_result)
        + (
            0.0
            if problem.factor_admitted
            and contact_available
            and joint_result.inference_admissible
            else 4.0
        )
    )

    features: dict[str, float] = {}
    for method_id in RAW_METHOD_IDS:
        correction = _correction_rms_m(
            trajectories[method_id][evaluation_start:evaluation_stop],
            baseline[evaluation_start:evaluation_stop],
        )
        if method_id in results:
            uncertainty = _query_uncertainty_rms_m(
                problem.future_state_jacobian_m[evaluation_start:evaluation_stop],
                results[method_id],
            )
        else:
            uncertainty = 0.0
        features[method_id] = float(np.hypot(correction, uncertainty))

    diagnostics = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-prediction-diagnostics",
        "schema_version": 1,
        "information_boundary": {
            "confirmation_payloads_opened": False,
            "future_object_observations_used": False,
            "public_released_prefix_measurements_used": True,
            "known_robot_action_used": True,
        },
        "factor_admitted": problem.factor_admitted,
        "contact_available": contact_available,
        "physical_mode": problem.physical_mode,
        "causal_frame_stop": problem.causal_frame_stop,
        "evaluation_frame_range_half_open": list(evaluation_range),
        "risk_components": {
            "joint_correction_rms_m": joint_correction,
            "joint_query_uncertainty_rms_m": joint_uncertainty,
            "visual_contact_state_disagreement_m": disagreement,
            "joint_robust_rejection_fraction": _normalized_rejection(joint_result),
            "physical_response_scale_m": response_scale,
        },
        "inference": {
            method_id: {
                "admissible": result.inference_admissible,
                "reason": result.reason,
                "exact_fallback": bool(
                    np.array_equal(trajectories[method_id], baseline)
                ),
            }
            for method_id, result in results.items()
        },
    }
    return Deform360JointSparsePredictionResultV5(
        input_id=problem.input_id,
        trajectories_m=trajectories,
        inference_results=results,
        risk_score=risk_score,
        predicted_loss_features_m=features,
        diagnostics=diagnostics,
    )


__all__ = [
    "B0_PHYSICAL_FALLBACK",
    "B1_LAST_CAUSAL_RESIDUAL",
    "Deform360JointSparsePredictionInputV5",
    "Deform360JointSparsePredictionResultV5",
    "RAW_METHOD_IDS",
    "T1_CONTACT_ONLY",
    "V1_VISUAL_GUARDED",
    "VT2_VISUOTACTILE_UNGUARDED",
    "VT3_VISUOTACTILE_ANCHOR_BIAS",
    "run_deform360_joint_sparse_prediction_v5",
]
