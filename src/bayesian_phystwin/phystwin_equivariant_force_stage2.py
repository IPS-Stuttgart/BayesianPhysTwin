"""Frozen execution semantics for the equivariant-force official-Warp gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from .phystwin_bayesian_anchor import robust_random_walk_endpoint
from .phystwin_graph_discrepancy import (
    graph_smoothed_discrepancy_posterior,
    normalized_spring_laplacian,
)
from .phystwin_residual_dynamics import _clip_residual, _sha256


EQUIVARIANT_FORCE_STAGE2_CONTRACT = (
    "phystwin-equivariant-force-official-warp-stage2-v1"
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class EquivariantForceStage2Protocol:
    """Validated Stage-2 amendment bound to one immutable source protocol."""

    payload: Mapping[str, Any]
    source_protocol_sha256: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Stage2FrameIntervals:
    """Frame indices under the frame-zero initial-state convention."""

    initial_state_frame: int
    simulator_step_frames: tuple[int, ...]
    readout_fit_frames: tuple[int, ...]
    scoring_frames: tuple[int, ...]


@dataclass(frozen=True)
class GraphPersistenceFit:
    """One prefix-only graph-persistence readout fit."""

    correction_m: np.ndarray
    endpoint_mean_m: np.ndarray
    endpoint_variance_m2: np.ndarray
    endpoint_update_count: np.ndarray
    correction_rms_m: float
    laplacian_energy_m2: float
    direct_support_count: int


def stage2_frame_intervals(
    fit_end_frame: int,
    train_end_frame: int,
) -> Stage2FrameIntervals:
    """Resolve the exclusive fit boundary without an off-by-one choice."""

    if not 3 <= fit_end_frame < train_end_frame:
        raise ValueError("Stage 2 requires 3 <= fit_end < train_end")
    return Stage2FrameIntervals(
        initial_state_frame=0,
        simulator_step_frames=tuple(range(1, train_end_frame)),
        readout_fit_frames=tuple(range(0, fit_end_frame)),
        scoring_frames=tuple(range(fit_end_frame, train_end_frame)),
    )


def load_equivariant_force_stage2_protocol(
    path: str | Path,
    *,
    source_protocol_path: str | Path | None = None,
) -> EquivariantForceStage2Protocol:
    """Load the pre-Stage-1 amendment and verify its source binding."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported equivariant-force Stage-2 schema")
    if payload.get("contract") != EQUIVARIANT_FORCE_STAGE2_CONTRACT:
        raise ValueError("unsupported equivariant-force Stage-2 contract")
    if payload.get("stage_1_outcome_observed_before_lock") is not False:
        raise ValueError("Stage-2 execution must be locked before Stage 1")
    if payload.get("target_artifacts_opened") is not False:
        raise ValueError("Stage-2 execution contract crossed the target boundary")

    source_digest = payload.get("source_protocol_sha256")
    source_archive_digest = payload.get("source_episode_archive_sha256")
    if not _valid_sha256(source_digest) or not _valid_sha256(
        source_archive_digest
    ):
        raise ValueError("Stage-2 contract omits its source protocol SHA-256")
    source_digest = str(source_digest)
    source_payload = None
    if source_protocol_path is not None:
        if _sha256(source_protocol_path) != source_digest:
            raise ValueError("Stage-2 contract is bound to another source protocol")
        source_payload = json.loads(
            Path(source_protocol_path).read_text(encoding="utf-8")
        )

    rollout = payload.get("rollout")
    ensemble = payload.get("ensemble")
    readout = payload.get("readout_refit")
    if not all(
        isinstance(value, Mapping) for value in (rollout, ensemble, readout)
    ):
        raise ValueError("Stage-2 contract omits execution settings")
    required_rollout = {
        "initial_state_frame": 0,
        "first_simulator_step_frame": 1,
        "fit_end_is_exclusive": True,
        "same_initial_state": True,
        "same_simulator_configuration": True,
        "physical_parameters": "released_optimal_phystwin_parameters",
        "reference_force": "exact_zero",
        "candidate_force_location": "inside_official_warp",
        "candidate_force_active_during_prefix": True,
        "candidate_force_active_during_scoring": True,
        "candidate_force_interval": "[1, train_end_frame)",
        "reference_force_interval": "exact_zero_on_[1, train_end_frame)",
        "score_interval": "[fit_end_frame, train_end_frame)",
        "state_at_fit_end_is_propagated_from_prefix": True,
        "future_observations_used_for_initialization_or_force": False,
    }
    if any(rollout.get(key) != value for key, value in required_rollout.items()):
        raise ValueError("Stage-2 rollout semantics changed")
    if ensemble.get("aggregation") != (
        "arithmetic_mean_force_field_per_frame_float64_then_float32"
    ):
        raise ValueError("Stage-2 seed aggregation changed")
    if ensemble.get("member_pairing") != (
        "each_seed_model_with_its_own_prefix_adapted_latent"
    ):
        raise ValueError("Stage-2 seed-latent pairing changed")
    if ensemble.get("seed_selection") != "forbidden":
        raise ValueError("Stage-2 seed selection must remain forbidden")
    if ensemble.get("force_bound_preserved_by_convex_averaging") is not True:
        raise ValueError("Stage-2 ensemble must preserve the force bound")
    seeds = tuple(int(value) for value in ensemble.get("seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage-2 seeds must be nonempty and unique")
    if source_payload is not None:
        source_seeds = tuple(
            int(value) for value in source_payload["training"]["seeds"]
        )
        if seeds != source_seeds:
            raise ValueError("Stage-2 seeds differ from the source protocol")

    required_readout = {
        "fit_interval": "[0, fit_end_frame)",
        "fit_separately_per_arm": True,
        "future_frames_used": False,
        "robust_process_std_m": FIXED_PROCESS_STD_M,
        "robust_observation_std_m": FIXED_OBSERVATION_STD_M,
        "robust_initial_std_m": FIXED_INITIAL_STD_M,
        "robust_inlier_prior": FIXED_INLIER_PRIOR,
        "robust_outlier_variance_multiplier": (
            FIXED_OUTLIER_VARIANCE_MULTIPLIER
        ),
        "graph": "normalized_released_spring_laplacian",
        "maximum_residual_m": 0.01,
        "graph_prior_strength": 0.3,
        "correction_application": (
            "persist_fitted_field_over_[fit_end_frame,train_end_frame)"
        ),
        "shrinkage_definition": (
            "1-candidate_correction_rms/reference_correction_rms"
        ),
        "unsupported_reference_cases_count_toward_shrinkage_gate": False,
        "laplacian_energy_is_diagnostic_only": True,
    }
    if any(readout.get(key) != value for key, value in required_readout.items()):
        raise ValueError("Stage-2 readout-refit semantics changed")
    if float(readout.get("minimum_reference_rms_m", 0.0)) <= 0.0:
        raise ValueError("Stage-2 readout support floor must be positive")
    primary = payload.get("primary_comparison")
    parity = payload.get("zero_force_parity")
    if not isinstance(primary, Mapping) or not isinstance(parity, Mapping):
        raise ValueError("Stage-2 contract omits controls")
    required_primary = {
        "candidate": (
            "equivariant_force_replay_plus_separately_refit_graph_persistence"
        ),
        "reference": (
            "exact_zero_force_replay_plus_separately_refit_graph_persistence"
        ),
        "only_arm_difference_before_readout_refit": (
            "equivariant_generalized_force"
        ),
        "equal_case_aggregation": True,
        "prefix_state_injection": "diagnostic_control_only",
    }
    if any(primary.get(key) != value for key, value in required_primary.items()):
        raise ValueError("Stage-2 primary comparison changed")
    if parity.get("required") is not True or parity.get("bitwise") is not True:
        raise ValueError("Stage-2 exact zero-force parity is required")
    return EquivariantForceStage2Protocol(
        payload=payload,
        source_protocol_sha256=source_digest,
        seeds=seeds,
    )


def fit_prefix_graph_persistence(
    observed_prefix_m: np.ndarray,
    physical_prefix_m: np.ndarray,
    valid_prefix: np.ndarray,
    object_edges: np.ndarray,
    *,
    graph_prior_strength: float = 0.3,
    maximum_residual_m: float = 0.01,
) -> GraphPersistenceFit:
    """Fit the frozen graph readout using one arm's prefix and no future."""

    observed = np.asarray(observed_prefix_m, dtype=float)
    physical = np.asarray(physical_prefix_m, dtype=float)
    valid = np.asarray(valid_prefix, dtype=bool)
    edges = np.asarray(object_edges, dtype=np.int64)
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("observed_prefix_m must have shape (T,M,3)")
    if physical.ndim != 3 or physical.shape[-1] != 3:
        raise ValueError("physical_prefix_m must have shape (T,N,3)")
    if len(observed) != len(physical) or observed.shape[1] > physical.shape[1]:
        raise ValueError("observed and physical prefixes disagree")
    if valid.shape != observed.shape[:2]:
        raise ValueError("valid_prefix must match observed points")
    if len(observed) < 3:
        raise ValueError("graph-persistence fitting requires at least three frames")
    if graph_prior_strength <= 0.0 or maximum_residual_m <= 0.0:
        raise ValueError("graph-persistence scales must be positive")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(physical)):
        raise ValueError("graph-persistence prefixes must be finite")

    observed_count = observed.shape[1]
    residual = observed - physical[:, :observed_count]
    endpoint = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=len(observed),
        process_variance=FIXED_PROCESS_STD_M**2,
        observation_variance=FIXED_OBSERVATION_STD_M**2,
        initial_variance=FIXED_INITIAL_STD_M**2,
        inlier_prior=FIXED_INLIER_PRIOR,
        outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    )
    updated = endpoint.update_count > 0
    if not np.any(updated):
        raise ValueError("graph-persistence prefix has no supported observation")
    laplacian = normalized_spring_laplacian(physical.shape[1], edges)
    posterior = graph_smoothed_discrepancy_posterior(
        endpoint.mean,
        endpoint.variance,
        updated,
        laplacian,
        prior_strength=graph_prior_strength,
    )
    correction = _clip_residual(
        posterior.mean[None],
        maximum_residual_m,
    )[0]
    node_norm = np.linalg.norm(correction, axis=1)
    laplacian_field = laplacian @ correction
    return GraphPersistenceFit(
        correction_m=np.asarray(correction, dtype=np.float32),
        endpoint_mean_m=np.asarray(endpoint.mean, dtype=np.float32),
        endpoint_variance_m2=np.asarray(endpoint.variance, dtype=np.float32),
        endpoint_update_count=np.asarray(endpoint.update_count, dtype=np.int32),
        correction_rms_m=float(np.sqrt(np.mean(np.square(node_norm)))),
        laplacian_energy_m2=float(np.mean(np.square(laplacian_field))),
        direct_support_count=int(np.sum(updated)),
    )


def readout_correction_shrinkage(
    candidate_correction_m: np.ndarray,
    reference_correction_m: np.ndarray,
    *,
    minimum_reference_rms_m: float,
) -> dict[str, float | bool]:
    """Measure held-out mechanism explanation in readout-amplitude units."""

    candidate = np.asarray(candidate_correction_m, dtype=float)
    reference = np.asarray(reference_correction_m, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("candidate and reference corrections must share (N,3)")
    if candidate.shape[1] != 3:
        raise ValueError("correction fields must contain 3-vectors")
    if minimum_reference_rms_m <= 0.0:
        raise ValueError("minimum_reference_rms_m must be positive")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(reference)):
        raise ValueError("correction fields must be finite")
    candidate_rms = float(
        np.sqrt(np.mean(np.sum(np.square(candidate), axis=1)))
    )
    reference_rms = float(
        np.sqrt(np.mean(np.sum(np.square(reference), axis=1)))
    )
    supported = reference_rms >= minimum_reference_rms_m
    if supported:
        shrinkage = 1.0 - candidate_rms / reference_rms
    elif candidate_rms <= minimum_reference_rms_m:
        shrinkage = 0.0
    else:
        shrinkage = 1.0 - candidate_rms / minimum_reference_rms_m
    return {
        "candidate_correction_rms_m": candidate_rms,
        "reference_correction_rms_m": reference_rms,
        "readout_correction_shrinkage": float(shrinkage),
        "reference_supported": bool(supported),
    }
