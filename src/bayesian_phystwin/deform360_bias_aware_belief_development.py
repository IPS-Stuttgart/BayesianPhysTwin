"""Open-source development adapter for guarded bias-aware Deform360 belief.

Candidate construction accepts no target. Open source outcomes enter only in
the separate scoring and leave-one-object-out regret-calibration stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    apply_group_regret_bound,
    apply_regret_guard,
    build_physical_response_basis,
    decode_bias_aware_state,
    fit_source_regret_certificate,
    fit_source_group_regret_bound,
    restrict_state_basis_to_identifiable_subspace,
    update_bias_aware_state,
)
from .deform360_online_belief_evaluation import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_SOURCE_EPISODES,
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _load_pickle,
    _physical_object_cluster_bootstrap,
    _resolve_prediction_archive,
    _sha256,
    _validate_deform360_outcome_manifest,
    robust_huber_continuation_gain,
    score_deform360_hidden_trajectory,
)


PROTOCOL_ID = "deform360-bias-aware-guarded-belief-open27-v4-development"
FEATURE_NAMES = (
    "physical_response_rms_over_scale",
    "observed_motion_rms_over_scale",
    "innovation_rms_over_scale",
    "correction_rms_over_scale",
    "mean_prior_reliability",
    "causal_physical_agreement_gain",
    "maximum_state_bias_subspace_cosine",
    "maximum_state_bias_posterior_correlation",
)


@dataclass(frozen=True)
class Deform360BiasAwareDevelopmentConfig:
    """Frozen choices for one source-development candidate construction."""

    update_frames: tuple[int, ...] = UPDATE_FRAMES
    minimum_available_center_count: int = 9
    minimum_motion_center_count: int = 3
    physical_response_rank: int = 4
    minimum_physical_response_m: float = 0.0005
    minimum_observed_motion_m: float = 0.0005
    minimum_physical_agreement_gain: float = 0.40
    minimum_identifiable_fraction: float = 0.10
    observation_variance_floor_m2: float = 0.005**2
    reprojection_scale_px: float = 3.0
    regret_nominal_coverage: float = 0.90
    regret_minimum_improvement_m: float = 0.000005
    regret_ridge_penalty: float = 10.0
    regret_support_margin_std: float = 0.0
    state_update: BiasAwareStateUpdateConfig = field(
        default_factory=BiasAwareStateUpdateConfig
    )

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.update_frames))) != self.update_frames:
            raise ValueError("update_frames must be strictly increasing")
        if not self.update_frames:
            raise ValueError("update_frames must not be empty")
        if self.minimum_available_center_count < 1:
            raise ValueError("minimum available centre count must be positive")
        if self.minimum_motion_center_count < 1:
            raise ValueError("minimum motion centre count must be positive")
        if self.physical_response_rank < 1:
            raise ValueError("physical response rank must be positive")
        positive = (
            self.minimum_physical_response_m,
            self.minimum_observed_motion_m,
            self.minimum_identifiable_fraction,
            self.observation_variance_floor_m2,
            self.reprojection_scale_px,
            self.regret_ridge_penalty,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("development scales must be positive")
        if self.minimum_identifiable_fraction > 1.0:
            raise ValueError("minimum identifiable fraction exceeds one")
        if not 0.0 <= self.minimum_physical_agreement_gain <= 1.0:
            raise ValueError("minimum physical agreement gain must lie in [0, 1]")
        if not 0.0 < self.regret_nominal_coverage < 1.0:
            raise ValueError("regret coverage must lie in (0, 1)")
        if self.regret_minimum_improvement_m < 0.0:
            raise ValueError("minimum regret improvement is negative")
        if self.regret_support_margin_std < 0.0:
            raise ValueError("regret support margin is negative")


def _validate_prediction_arrays(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    center_ids: np.ndarray,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    update_frames: tuple[int, ...],
) -> None:
    if baseline_m.ndim != 3 or baseline_m.shape[2] != 3:
        raise ValueError("baseline must have shape (T, N, 3)")
    if physical_response_m.shape != baseline_m.shape:
        raise ValueError("physical response must match baseline")
    if frame_zero_points_m.shape != baseline_m.shape[1:]:
        raise ValueError("frame-zero point shape changed")
    if action_support.shape != (baseline_m.shape[1],):
        raise ValueError("action support shape changed")
    if measurement_m.shape != baseline_m.shape:
        raise ValueError("measurement must match baseline")
    for name, value in (
        ("measurement_visibility", measurement_visibility),
        ("measurement_validity", measurement_validity),
    ):
        if value.shape != baseline_m.shape[:2]:
            raise ValueError(f"{name} shape changed")
    if center_ids.ndim != 1 or len(center_ids) != len(np.unique(center_ids)):
        raise ValueError("center_ids must be a unique vector")
    if np.any(center_ids < 0) or np.any(center_ids >= baseline_m.shape[1]):
        raise ValueError("centre ID exceeds trajectory")
    expected_update_shape = (len(update_frames), len(center_ids))
    if prior_reliability.shape != expected_update_shape:
        raise ValueError("prior reliability shape changed")
    if observation_variance_m2.shape != expected_update_shape:
        raise ValueError("observation variance shape changed")
    if update_frames[-1] >= len(baseline_m):
        raise ValueError("update frame exceeds trajectory")
    finite_inputs = (
        baseline_m,
        physical_response_m,
        frame_zero_points_m,
        action_support,
        prior_reliability,
        observation_variance_m2,
    )
    if any(not np.all(np.isfinite(value)) for value in finite_inputs):
        raise ValueError("prediction input contains non-finite values")
    if np.any((action_support < 0.0) | (action_support > 1.0)):
        raise ValueError("action support must lie in [0, 1]")
    if np.any((prior_reliability < 0.0) | (prior_reliability > 1.0)):
        raise ValueError("prior reliability must lie in [0, 1]")
    if np.any(observation_variance_m2 <= 0.0):
        raise ValueError("observation variance must be positive")


def _spatial_bias_basis(frame_zero_points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(points), 0), dtype=np.float64)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular_values[0]
    count = int(np.sum(singular_values > tolerance))
    basis = left[:, :count].copy()
    for mode in range(count):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
        basis[:, mode] /= np.max(np.abs(basis[:, mode]))
    return basis


def _radial_rms(value_m: np.ndarray) -> float:
    value = np.asarray(value_m, dtype=np.float64)
    if not len(value):
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _object_scale_m(frame_zero_points_m: np.ndarray) -> float:
    centered = frame_zero_points_m - np.median(frame_zero_points_m, axis=0)
    return max(1e-6, float(2.0 * np.max(np.linalg.norm(centered, axis=1))))


def predict_bias_aware_candidate_arrays(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    config: Deform360BiasAwareDevelopmentConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build an unguarded candidate using no target or future observation."""

    cfg = config or Deform360BiasAwareDevelopmentConfig()
    baseline_input = np.asarray(baseline_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    response = np.asarray(physical_response_m, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    variance = np.asarray(observation_variance_m2, dtype=np.float64)
    _validate_prediction_arrays(
        baseline,
        response,
        frame_zero,
        support,
        measurement,
        visible,
        valid,
        centers,
        reliability,
        variance,
        cfg.update_frames,
    )
    candidate = baseline_input.copy()
    spatial_bias = _spatial_bias_basis(frame_zero)
    object_scale = _object_scale_m(frame_zero)
    support_mask = support > 0.0
    previous_update = 0
    update_records: list[dict[str, Any]] = []

    for update_index, update in enumerate(cfg.update_frames):
        stop = (
            cfg.update_frames[update_index + 1]
            if update_index + 1 < len(cfg.update_frames)
            else len(baseline)
        )
        candidate[update + 1 : stop] = baseline_input[update + 1 : stop]
        available = (
            visible[update, centers]
            & valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & (reliability[update_index] > 0.0)
        )
        previous_available = (
            visible[previous_update, centers]
            & valid[previous_update, centers]
            & np.all(np.isfinite(measurement[previous_update, centers]), axis=1)
        )
        motion_available = available & previous_available
        physical_delta = response[update] - response[previous_update]
        physical_response_rms = _radial_rms(physical_delta[support_mask])
        observed_motion_rms = _radial_rms(
            measurement[update, centers[motion_available]]
            - measurement[previous_update, centers[motion_available]]
        )
        physical_agreement_gain = robust_huber_continuation_gain(
            physical_delta[centers[motion_available]],
            measurement[update, centers[motion_available]]
            - measurement[previous_update, centers[motion_available]],
            minimum_point_count=cfg.minimum_motion_center_count,
            fallback=0.0,
        )
        dynamic_selected = (
            int(np.sum(available)) >= cfg.minimum_available_center_count
            and int(np.sum(motion_available)) >= cfg.minimum_motion_center_count
            and physical_response_rms >= cfg.minimum_physical_response_m
            and observed_motion_rms >= cfg.minimum_observed_motion_m
            and physical_agreement_gain >= cfg.minimum_physical_agreement_gain
        )
        record: dict[str, Any] = {
            "frame": update,
            "interval_end_exclusive": stop,
            "available_center_count": int(np.sum(available)),
            "motion_center_count": int(np.sum(motion_available)),
            "physical_response_rms_m": physical_response_rms,
            "observed_motion_rms_m": observed_motion_rms,
            "causal_physical_agreement_gain": physical_agreement_gain,
            "dynamic_window_selected": dynamic_selected,
            "candidate_available": False,
            "decision": "dynamic-evidence-exact-baseline-fallback",
            "bit_exact_baseline_fallback": True,
            "feature_names": list(FEATURE_NAMES),
            "features": None,
        }
        if dynamic_selected:
            try:
                physical_basis = build_physical_response_basis(
                    response[: update + 1],
                    action_support=support,
                    rank=cfg.physical_response_rank,
                    minimum_response_m=cfg.minimum_physical_response_m,
                )
                observed_ids = centers[available]
                observation_state = physical_basis.basis[observed_ids]
                observation_bias = np.column_stack(
                    (spatial_bias[observed_ids], np.ones(len(observed_ids)))
                )
                identifiable = restrict_state_basis_to_identifiable_subspace(
                    physical_basis.basis,
                    observation_state,
                    observation_bias,
                    minimum_identifiable_fraction=(
                        cfg.minimum_identifiable_fraction
                    ),
                )
                innovation = (
                    measurement[update, observed_ids]
                    - baseline[update, observed_ids]
                )
                update_result = update_bias_aware_state(
                    innovation[None],
                    np.ones((1, len(observed_ids)), dtype=bool),
                    identifiable.observation_basis,
                    spatial_bias[observed_ids],
                    prior_reliability=reliability[update_index, available][None],
                    observation_variance_m2=variance[
                        update_index, available
                    ][None],
                    config=cfg.state_update,
                )
                if not update_result.accepted:
                    raise ValueError(update_result.reason)
                correction = decode_bias_aware_state(
                    update_result, identifiable.query_basis
                )
                correction *= physical_agreement_gain
                candidate[update + 1 : stop] = (
                    baseline[update + 1 : stop] + correction[None]
                ).astype(baseline_input.dtype, copy=False)
                innovation_rms = _radial_rms(innovation)
                correction_rms = _radial_rms(correction)
                features = np.asarray(
                    [
                        physical_response_rms / object_scale,
                        observed_motion_rms / object_scale,
                        innovation_rms / object_scale,
                        correction_rms / object_scale,
                        float(np.mean(reliability[update_index, available])),
                        physical_agreement_gain,
                        float(update_result.diagnostics["state_bias_subspace_cosine"]),
                        float(
                            update_result.diagnostics[
                                "maximum_state_bias_posterior_correlation"
                            ]
                        ),
                    ],
                    dtype=np.float64,
                )
                record.update(
                    {
                        "candidate_available": True,
                        "decision": "bias-aware-candidate-available",
                        "bit_exact_baseline_fallback": False,
                        "features": features.tolist(),
                        "physical_basis_rank": int(
                            physical_basis.basis.shape[1]
                        ),
                        "identifiable_basis_rank": int(
                            identifiable.query_basis.shape[1]
                        ),
                        "minimum_identifiable_fraction": float(
                            np.min(identifiable.identifiable_fractions)
                        ),
                        "maximum_correction_m": float(
                            np.max(np.linalg.norm(correction, axis=1))
                        ),
                        "state_update_diagnostics": update_result.diagnostics,
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                record["decision"] = (
                    "bias-aware-update-exact-baseline-fallback"
                )
                record["fallback_reason"] = f"{type(error).__name__}: {error}"
        if not record["candidate_available"] and not np.array_equal(
            candidate[update + 1 : stop], baseline_input[update + 1 : stop]
        ):
            raise AssertionError("candidate fallback changed the exact baseline")
        update_records.append(record)
        previous_update = update

    report = {
        "protocol_id": PROTOCOL_ID,
        "arm": "bias_aware_state_candidate_unguarded",
        "config": asdict(cfg),
        "feature_names": list(FEATURE_NAMES),
        "center_ids": centers.tolist(),
        "updates": update_records,
        "candidate_update_count": int(
            sum(bool(record["candidate_available"]) for record in update_records)
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "physical_response_frames_by_update": (
                "causal prefix [0, update] only"
            ),
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
        },
    }
    return report, candidate


@dataclass
class _OpenSourceCase:
    case: str
    object_id: str
    center_ids: np.ndarray
    baseline: np.ndarray
    candidate: np.ndarray
    target: np.ndarray
    visibility: np.ndarray
    validity: np.ndarray
    report: dict[str, Any]
    guarded: np.ndarray | None = None
    group_guarded: np.ndarray | None = None


def _expected_case_names() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _load_source_target_pickle(path: Path) -> Any:
    """Read NumPy-2 pickles under the validated NumPy-1 server runtime."""

    try:
        return _load_pickle(path)
    except ModuleNotFoundError as error:
        if error.name != "numpy._core.numeric":
            raise
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
        return _load_pickle(path)


def _scored_frames(
    frame_count: int, update_frames: tuple[int, ...]
) -> tuple[int, ...]:
    frames: list[int] = []
    for index, update in enumerate(update_frames):
        stop = (
            update_frames[index + 1]
            if index + 1 < len(update_frames)
            else frame_count
        )
        frames.extend(range(update + 1, stop))
    return tuple(frames)


def _source_reliability_and_variance(
    measurement_arrays: Mapping[str, np.ndarray],
    uncertainty_arrays: Mapping[str, np.ndarray],
    *,
    update_frames: tuple[int, ...],
    center_ids: np.ndarray,
    config: Deform360BiasAwareDevelopmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    selected_camera_count = len(measurement_arrays["selected_cameras"])
    if selected_camera_count < 2:
        raise ValueError("source measurement has fewer than two cameras")
    inlier_count = np.asarray(
        measurement_arrays["triangulation_inlier_view_count"], dtype=np.float64
    )
    reprojection = np.asarray(
        measurement_arrays["triangulation_median_reprojection_px"],
        dtype=np.float64,
    )
    expected = (len(update_frames), len(center_ids))
    if inlier_count.shape != expected or reprojection.shape != expected:
        raise ValueError("source triangulation diagnostic shape changed")
    redundancy = np.clip(
        (inlier_count - 1.0) / (selected_camera_count - 1.0), 0.0, 1.0
    )
    geometry = np.exp(
        -0.5 * np.square(reprojection / config.reprojection_scale_px)
    )
    reliability = redundancy * geometry
    reliability[~np.isfinite(reliability)] = 0.0

    covariance = np.asarray(
        uncertainty_arrays["measurement_covariance_m2"], dtype=np.float64
    )
    covariance_valid = np.asarray(
        uncertainty_arrays["measurement_covariance_valid"], dtype=bool
    )
    if covariance.shape[-2:] != (3, 3):
        raise ValueError("source covariance must end in (3, 3)")
    variance = np.empty(expected, dtype=np.float64)
    for update_index, update in enumerate(update_frames):
        selected = covariance[update, center_ids]
        isotropic = np.trace(selected, axis1=1, axis2=2) / 3.0
        valid_covariance = covariance_valid[update, center_ids] & np.isfinite(
            isotropic
        )
        reliability[update_index, ~valid_covariance] = 0.0
        variance[update_index] = np.where(
            valid_covariance,
            np.maximum(isotropic, config.observation_variance_floor_m2),
            config.observation_variance_floor_m2,
        )
    return reliability, variance


def _load_open_source_case(
    source_case_dir: Path,
    measurement_case_dir: Path,
    uncertainty_case_dir: Path,
    selected_baseline_path: Path,
    config: Deform360BiasAwareDevelopmentConfig,
) -> _OpenSourceCase:
    seal_path = source_case_dir / "prediction_seal.json"
    outcome_path = source_case_dir / "outcome.json"
    target_path = source_case_dir / "target_data.pkl"
    measurement_path = measurement_case_dir / "measurement.npz"
    uncertainty_path = uncertainty_case_dir / "measurement_cycle_uncertainty.npz"
    for path in (
        seal_path,
        outcome_path,
        target_path,
        measurement_path,
        uncertainty_path,
        selected_baseline_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    _validate_deform360_outcome_manifest(
        seal_path, target_path, seal, outcome
    )
    prediction_path = _resolve_prediction_archive(source_case_dir, seal)
    with np.load(prediction_path, allow_pickle=False) as stored:
        required = {
            "prediction_m",
            "persistence_m",
            "driven_readout_m",
            "zero_action_readout_m",
            "action_support",
            "frame_zero_points_m",
        }
        if not required.issubset(stored.files):
            raise ValueError("sealed source archive lacks bias-aware inputs")
        physical_response = (
            np.asarray(stored["driven_readout_m"], dtype=np.float64)
            - np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
        )
        action_support = np.asarray(stored["action_support"], dtype=np.float64)
        frame_zero = np.asarray(
            stored["frame_zero_points_m"], dtype=np.float64
        )
    with np.load(selected_baseline_path, allow_pickle=False) as stored:
        if "selected_raw_backbone" not in stored.files:
            raise ValueError("selected baseline archive lacks selected_raw_backbone")
        baseline = np.asarray(stored["selected_raw_backbone"]).copy()
    with np.load(measurement_path, allow_pickle=False) as stored:
        measurement_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(uncertainty_path, allow_pickle=False) as stored:
        uncertainty_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    update_frames = tuple(
        int(value) for value in measurement_arrays["update_frames"]
    )
    if update_frames != config.update_frames:
        raise ValueError("source measurement update frames changed")
    center_ids = np.asarray(measurement_arrays["center_ids"], dtype=np.int64)
    reliability, variance = _source_reliability_and_variance(
        measurement_arrays,
        uncertainty_arrays,
        update_frames=update_frames,
        center_ids=center_ids,
        config=config,
    )
    target_free_report, candidate = predict_bias_aware_candidate_arrays(
        baseline,
        physical_response,
        frame_zero,
        action_support,
        np.asarray(measurement_arrays["measurement_m"]),
        np.asarray(measurement_arrays["measurement_visibility"], dtype=bool),
        np.asarray(measurement_arrays["measurement_validity"], dtype=bool),
        center_ids=center_ids,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=config,
    )
    target_data = _load_source_target_pickle(target_path)
    target = np.asarray(target_data["object_points"])
    visibility = np.asarray(target_data["object_visibilities"], dtype=bool)
    validity = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if target.shape != baseline.shape:
        raise ValueError("source target and selected baseline shapes differ")
    if not np.array_equal(target[0].astype(np.float32), frame_zero.astype(np.float32)):
        raise ValueError("source frame-zero target differs from sealed prediction")
    scored_frames = _scored_frames(len(target), config.update_frames)
    baseline_score = score_deform360_hidden_trajectory(
        baseline,
        target,
        visibility,
        validity,
        center_ids=center_ids,
        scored_frames=scored_frames,
    )
    candidate_score = score_deform360_hidden_trajectory(
        candidate,
        target,
        visibility,
        validity,
        center_ids=center_ids,
        scored_frames=scored_frames,
    )
    interval_outcomes = []
    for update_record in target_free_report["updates"]:
        update = int(update_record["frame"])
        stop = int(update_record["interval_end_exclusive"])
        interval_frames = tuple(range(update + 1, stop))
        baseline_interval = score_deform360_hidden_trajectory(
            baseline,
            target,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=interval_frames,
        )
        candidate_interval = score_deform360_hidden_trajectory(
            candidate,
            target,
            visibility,
            validity,
            center_ids=center_ids,
            scored_frames=interval_frames,
        )
        identity_regret = (
            candidate_interval[PRIMARY_METRICS[0]]
            - baseline_interval[PRIMARY_METRICS[0]]
        )
        chamfer_regret = (
            candidate_interval[PRIMARY_METRICS[1]]
            - baseline_interval[PRIMARY_METRICS[1]]
        )
        interval_outcomes.append(
            {
                "frame": update,
                "identity_regret_m": float(identity_regret),
                "chamfer_regret_m": float(chamfer_regret),
                "worst_primary_regret_m": float(
                    max(identity_regret, chamfer_regret)
                ),
            }
        )
    case_report = {
        "protocol_id": PROTOCOL_ID,
        "case": source_case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "target_free_prediction": target_free_report,
        "source_outcome": {
            "scores": {
                "selected_raw_baseline": baseline_score,
                "bias_aware_candidate_unguarded": candidate_score,
            },
            "intervals": interval_outcomes,
            "claim_boundary": (
                "already-open source outcome used only for development scoring "
                "and object-held-out regret calibration"
            ),
        },
        "input_sha256": {
            "prediction_seal": _sha256(seal_path),
            "prediction_archive": _sha256(prediction_path),
            "source_outcome": _sha256(outcome_path),
            "source_target": _sha256(target_path),
            "measurement": _sha256(measurement_path),
            "uncertainty": _sha256(uncertainty_path),
            "selected_baseline": _sha256(selected_baseline_path),
        },
    }
    return _OpenSourceCase(
        case=source_case_dir.name,
        object_id=str(seal["object_id"]),
        center_ids=center_ids,
        baseline=baseline,
        candidate=candidate,
        target=target,
        visibility=visibility,
        validity=validity,
        report=case_report,
    )


def _training_rows(
    cases: Sequence[_OpenSourceCase], held_object_id: str | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[list[float]] = []
    regret: list[float] = []
    groups: list[str] = []
    for case in cases:
        if held_object_id is not None and case.object_id == held_object_id:
            continue
        updates = case.report["target_free_prediction"]["updates"]
        outcomes = case.report["source_outcome"]["intervals"]
        for update, outcome in zip(updates, outcomes, strict=True):
            if not update["candidate_available"]:
                continue
            features.append(update["features"])
            regret.append(outcome["worst_primary_regret_m"])
            groups.append(case.object_id)
    if not features:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), []
    return np.asarray(features), np.asarray(regret), groups


def _fit_full_source_deployment_lock(
    cases: Sequence[_OpenSourceCase],
    config: Deform360BiasAwareDevelopmentConfig,
) -> dict[str, Any]:
    """Freeze the source-only selector consumed by a future evaluation."""

    features, regret, groups = _training_rows(cases, None)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise ValueError("full-source lock has fewer than three eligible objects")
    bound = fit_source_group_regret_bound(
        regret,
        groups,
        nominal_coverage=config.regret_nominal_coverage,
        within_group_coverage=1.0,
        minimum_improvement_m=config.regret_minimum_improvement_m,
    )
    group_scores = {
        group: float(score)
        for group, score in zip(
            unique_groups, bound.group_scores_m.tolist(), strict=True
        )
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "source-fitted-prospective-candidate-lock",
        "selector": (
            "target-free physical-response eligibility followed by one frozen "
            "source-group regret decision"
        ),
        "config": asdict(config),
        "candidate_interval_count": int(len(features)),
        "eligible_source_object_ids": unique_groups,
        "source_group_count": len(unique_groups),
        "source_group_worst_regret_m": group_scores,
        "upper_regret_m": bound.upper_regret_m,
        "minimum_improvement_m": bound.minimum_improvement_m,
        "candidate_certified": bound.candidate_certified,
        "requested_nominal_coverage": bound.nominal_coverage,
        "finite_sample_rank": bound.finite_sample_rank,
        "finite_sample_coverage": bound.finite_sample_coverage,
        "fresh_accuracy_evaluation_allowed": bound.candidate_certified,
        "calibrated_90_percent_claim_allowed": bool(
            bound.finite_sample_coverage >= 0.90
        ),
        "fallback": "bit-exact selected raw baseline",
        "information_boundary": {
            "source_outcomes_used_to_fit_lock": True,
            "prospective_outcomes_used_to_construct_candidate": False,
            "future_observations_used_to_construct_candidate": False,
            "eligibility_is_target_free": True,
        },
        "claim_boundary": (
            "This lock was selected on already-open source outcomes. Its finite-"
            "sample level is conditional on the frozen eligibility rule and the "
            "eligible source-object exchangeability assumption. It permits only "
            "a fresh accuracy/non-regression evaluation, not a 90% safety or "
            "calibration claim."
        ),
    }


def _aggregate_scores(
    cases: Sequence[_OpenSourceCase], arm: str
) -> dict[str, dict[str, float]]:
    by_object: dict[str, list[_OpenSourceCase]] = {}
    for case in cases:
        by_object.setdefault(case.object_id, []).append(case)
    result: dict[str, dict[str, float]] = {}
    for metric in PRIMARY_METRICS:
        episode_values = [
            float(case.report["source_outcome"]["scores"][arm][metric])
            for case in cases
        ]
        object_values = [
            float(
                np.mean(
                    [
                        member.report["source_outcome"]["scores"][arm][metric]
                        for member in members
                    ]
                )
            )
            for members in by_object.values()
        ]
        result[metric] = {
            "episode_mean_m": float(np.mean(episode_values)),
            "object_balanced_mean_m": float(np.mean(object_values)),
        }
    return result


def _apply_cross_fitted_regret_guard(
    cases: Sequence[_OpenSourceCase],
    config: Deform360BiasAwareDevelopmentConfig,
) -> dict[str, Any]:
    decision_records: list[dict[str, Any]] = []
    group_decision_records: list[dict[str, Any]] = []
    for held_object in sorted({case.object_id for case in cases}):
        features, regret, groups = _training_rows(cases, held_object)
        unique_groups = len(set(groups))
        certificate = None
        group_bound = None
        if unique_groups >= 3 and len(features):
            certificate = fit_source_regret_certificate(
                features,
                regret,
                groups,
                nominal_coverage=config.regret_nominal_coverage,
                within_group_coverage=1.0,
                minimum_improvement=config.regret_minimum_improvement_m,
                ridge_penalty=config.regret_ridge_penalty,
                support_margin_std=config.regret_support_margin_std,
            )
            group_bound = fit_source_group_regret_bound(
                regret,
                groups,
                nominal_coverage=config.regret_nominal_coverage,
                within_group_coverage=1.0,
                minimum_improvement_m=config.regret_minimum_improvement_m,
            )
        for case in cases:
            if case.object_id != held_object:
                continue
            guarded = case.baseline.copy()
            group_guarded = case.baseline.copy()
            updates = case.report["target_free_prediction"]["updates"]
            outcomes = case.report["source_outcome"]["intervals"]
            case_decisions = []
            case_group_decisions = []
            for update, outcome in zip(updates, outcomes, strict=True):
                start = int(update["frame"]) + 1
                stop = int(update["interval_end_exclusive"])
                if certificate is None or not update["candidate_available"]:
                    guarded[start:stop] = case.baseline[start:stop]
                    reason = (
                        "insufficient-cross-fit-source-exact-baseline-fallback"
                        if certificate is None
                        else "candidate-unavailable-exact-baseline-fallback"
                    )
                    accepted = False
                    predicted_regret = None
                    upper_regret = None
                    in_source_support = None
                else:
                    decision = apply_regret_guard(
                        case.baseline[start:stop],
                        case.candidate[start:stop],
                        np.asarray(update["features"]),
                        certificate,
                    )
                    guarded[start:stop] = decision.selected_value
                    reason = decision.reason
                    accepted = decision.candidate_accepted
                    predicted_regret = decision.predicted_regret
                    upper_regret = (
                        decision.upper_regret
                        if np.isfinite(decision.upper_regret)
                        else None
                    )
                    in_source_support = certificate.in_source_support(
                        np.asarray(update["features"])
                    )
                if not accepted and not np.array_equal(
                    guarded[start:stop], case.baseline[start:stop]
                ):
                    raise AssertionError("regret guard violated exact fallback")
                record = {
                    "case": case.case,
                    "object_id": case.object_id,
                    "frame": int(update["frame"]),
                    "candidate_available": bool(update["candidate_available"]),
                    "candidate_accepted": accepted,
                    "reason": reason,
                    "predicted_regret_m": predicted_regret,
                    "upper_regret_m": upper_regret,
                    "upper_regret_is_infinite": bool(
                        update["candidate_available"]
                        and upper_regret is None
                        and certificate is not None
                    ),
                    "in_source_support": in_source_support,
                    "actual_worst_primary_regret_m": outcome[
                        "worst_primary_regret_m"
                    ],
                    "bit_exact_baseline_fallback": bool(
                        accepted
                        or np.array_equal(
                            guarded[start:stop], case.baseline[start:stop]
                        )
                    ),
                }
                case_decisions.append(record)
                decision_records.append(record)
                if group_bound is None or not update["candidate_available"]:
                    group_guarded[start:stop] = case.baseline[start:stop]
                    group_reason = (
                        "insufficient-source-groups-exact-baseline-fallback"
                        if group_bound is None
                        else "candidate-unavailable-exact-baseline-fallback"
                    )
                    group_accepted = False
                    upper_group_regret = None
                    finite_sample_coverage = None
                    finite_sample_rank = None
                    source_group_count = unique_groups
                else:
                    group_decision = apply_group_regret_bound(
                        case.baseline[start:stop],
                        case.candidate[start:stop],
                        group_bound,
                    )
                    group_guarded[start:stop] = group_decision.selected_value
                    group_reason = group_decision.reason
                    group_accepted = group_decision.candidate_accepted
                    upper_group_regret = group_bound.upper_regret_m
                    finite_sample_coverage = group_bound.finite_sample_coverage
                    finite_sample_rank = group_bound.finite_sample_rank
                    source_group_count = len(group_bound.group_scores_m)
                if not group_accepted and not np.array_equal(
                    group_guarded[start:stop], case.baseline[start:stop]
                ):
                    raise AssertionError(
                        "source-group regret bound violated exact fallback"
                    )
                group_record = {
                    "case": case.case,
                    "object_id": case.object_id,
                    "frame": int(update["frame"]),
                    "candidate_available": bool(update["candidate_available"]),
                    "candidate_accepted": group_accepted,
                    "reason": group_reason,
                    "upper_regret_m": upper_group_regret,
                    "requested_nominal_coverage": (
                        config.regret_nominal_coverage
                    ),
                    "finite_sample_rank": finite_sample_rank,
                    "finite_sample_coverage": finite_sample_coverage,
                    "source_group_count": source_group_count,
                    "actual_worst_primary_regret_m": outcome[
                        "worst_primary_regret_m"
                    ],
                    "bit_exact_baseline_fallback": bool(
                        group_accepted
                        or np.array_equal(
                            group_guarded[start:stop], case.baseline[start:stop]
                        )
                    ),
                }
                case_group_decisions.append(group_record)
                group_decision_records.append(group_record)
            case.guarded = guarded
            case.group_guarded = group_guarded
            score = score_deform360_hidden_trajectory(
                guarded,
                case.target,
                case.visibility,
                case.validity,
                center_ids=case.center_ids,
                scored_frames=_scored_frames(len(guarded), config.update_frames),
            )
            case.report["source_outcome"]["scores"][
                "bias_aware_candidate_guarded_cross_fit"
            ] = score
            case.report["source_outcome"]["cross_fitted_guard"] = case_decisions
            group_score = score_deform360_hidden_trajectory(
                group_guarded,
                case.target,
                case.visibility,
                case.validity,
                center_ids=case.center_ids,
                scored_frames=_scored_frames(
                    len(group_guarded), config.update_frames
                ),
            )
            case.report["source_outcome"]["scores"][
                "bias_aware_group_bound_guarded_cross_fit"
            ] = group_score
            case.report["source_outcome"]["cross_fitted_group_regret_bound"] = (
                case_group_decisions
            )

    finite_bound = [
        record
        for record in decision_records
        if record["upper_regret_m"] is not None
    ]
    accepted = [record for record in decision_records if record["candidate_accepted"]]
    coverage = (
        None
        if not finite_bound
        else float(
            np.mean(
                [
                    record["actual_worst_primary_regret_m"]
                    <= record["upper_regret_m"]
                    for record in finite_bound
                ]
            )
        )
    )
    harmful_rate = (
        None
        if not accepted
        else float(
            np.mean(
                [
                    record["actual_worst_primary_regret_m"] > 0.0
                    for record in accepted
                ]
            )
        )
    )
    group_accepted_records = [
        record
        for record in group_decision_records
        if record["candidate_accepted"]
    ]
    group_harmful_rate = (
        None
        if not group_accepted_records
        else float(
            np.mean(
                [
                    record["actual_worst_primary_regret_m"] > 0.0
                    for record in group_accepted_records
                ]
            )
        )
    )
    finite_group_coverages = [
        float(record["finite_sample_coverage"])
        for record in group_decision_records
        if record["finite_sample_coverage"] is not None
    ]
    return {
        "decisions": decision_records,
        "candidate_available_count": int(
            sum(record["candidate_available"] for record in decision_records)
        ),
        "accepted_count": len(accepted),
        "finite_bound_count": len(finite_bound),
        "finite_bound_coverage": coverage,
        "accepted_harmful_rate": harmful_rate,
        "exact_fallback_count": int(
            sum(
                not record["candidate_accepted"]
                and record["bit_exact_baseline_fallback"]
                for record in decision_records
            )
        ),
        "source_group_bound": {
            "decisions": group_decision_records,
            "accepted_count": len(group_accepted_records),
            "accepted_harmful_rate": group_harmful_rate,
            "exact_fallback_count": int(
                sum(
                    not record["candidate_accepted"]
                    and record["bit_exact_baseline_fallback"]
                    for record in group_decision_records
                )
            ),
            "minimum_finite_sample_coverage": (
                None
                if not finite_group_coverages
                else float(np.min(finite_group_coverages))
            ),
            "maximum_finite_sample_coverage": (
                None
                if not finite_group_coverages
                else float(np.max(finite_group_coverages))
            ),
            "coverage_claim_boundary": (
                "The requested 90% level is not attainable with three or four "
                "training object groups; exact finite-sample resolution is "
                "reported per decision."
            ),
        },
    }


def evaluate_deform360_bias_aware_development(
    source_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    selected_baseline_root: str | Path,
    output_dir: str | Path,
    *,
    config: Deform360BiasAwareDevelopmentConfig | None = None,
) -> dict[str, Any]:
    """Run object-cross-fitted development on the already-open 27 episodes."""

    cfg = config or Deform360BiasAwareDevelopmentConfig()
    source = Path(source_root).resolve()
    measurement = Path(measurement_root).resolve()
    uncertainty = Path(uncertainty_root).resolve()
    selected_baseline = Path(selected_baseline_root).resolve()
    output = Path(output_dir).resolve()
    cases: list[_OpenSourceCase] = []
    for case_name in _expected_case_names():
        cases.append(
            _load_open_source_case(
                source / case_name,
                measurement / case_name,
                uncertainty / case_name,
                selected_baseline / f"{case_name}.npz",
                cfg,
            )
        )
    guard = _apply_cross_fitted_regret_guard(cases, cfg)
    arms = (
        "selected_raw_baseline",
        "bias_aware_candidate_unguarded",
        "bias_aware_candidate_guarded_cross_fit",
        "bias_aware_group_bound_guarded_cross_fit",
    )
    aggregate = {arm: _aggregate_scores(cases, arm) for arm in arms}
    groups = {case.case: case.object_id for case in cases}
    comparisons: dict[str, dict[str, Any]] = {}
    for arm in arms[1:]:
        comparisons[arm] = {}
        for metric in PRIMARY_METRICS:
            differences = {
                case.case: float(
                    case.report["source_outcome"]["scores"][arm][metric]
                    - case.report["source_outcome"]["scores"][
                        "selected_raw_baseline"
                    ][metric]
                )
                for case in cases
            }
            comparison = _physical_object_cluster_bootstrap(
                differences,
                groups,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            baseline_mean = aggregate["selected_raw_baseline"][metric][
                "object_balanced_mean_m"
            ]
            comparison["object_balanced_relative_change"] = (
                comparison["object_balanced_mean_difference_m"] / baseline_mean
            )
            comparison["episode_win_count"] = int(
                sum(value < 0.0 for value in differences.values())
            )
            comparison["episode_tie_count"] = int(
                sum(value == 0.0 for value in differences.values())
            )
            comparison["episode_count"] = len(differences)
            comparisons[arm][metric] = comparison

    guarded_comparison = comparisons[
        "bias_aware_group_bound_guarded_cross_fit"
    ]
    group_guard = guard["source_group_bound"]
    harmful_rate = group_guard["accepted_harmful_rate"]
    source_gates = {
        "guard_accepts_at_least_one_update": group_guard["accepted_count"] > 0,
        "object_balanced_identity_improves": (
            guarded_comparison[PRIMARY_METRICS[0]][
                "object_balanced_mean_difference_m"
            ]
            < 0.0
        ),
        "object_balanced_chamfer_improves": (
            guarded_comparison[PRIMARY_METRICS[1]][
                "object_balanced_mean_difference_m"
            ]
            < 0.0
        ),
        "accepted_harmful_rate_at_most_ten_percent": (
            harmful_rate is not None and harmful_rate <= 0.10
        ),
        "all_rejections_are_exact_fallback": (
            group_guard["exact_fallback_count"]
            == len(group_guard["decisions"]) - group_guard["accepted_count"]
        ),
    }
    deployment_lock = _fit_full_source_deployment_lock(cases, cfg)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for case in cases:
        report_path = output / f"{case.case}.json"
        arrays_path = output / f"{case.case}.npz"
        report_path.write_text(
            json.dumps(case.report, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        if case.guarded is None or case.group_guarded is None:
            raise AssertionError("cross-fitted guarded trajectory is missing")
        np.savez_compressed(
            arrays_path,
            selected_raw_baseline=case.baseline,
            bias_aware_candidate_unguarded=case.candidate,
            bias_aware_candidate_guarded_cross_fit=case.guarded,
            bias_aware_group_bound_guarded_cross_fit=case.group_guarded,
        )
        artifacts.append(
            {
                "case": case.case,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
    deployment_lock_path = output / "prospective_lock.json"
    deployment_lock_path.write_text(
        json.dumps(
            deployment_lock,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "already-open-source-development",
        "config": asdict(cfg),
        "case_count": len(cases),
        "object_count": len(set(groups.values())),
        "feature_names": list(FEATURE_NAMES),
        "aggregate": aggregate,
        "comparisons_to_selected_raw_baseline": comparisons,
        "cross_fitted_regret_guard": guard,
        "source_transfer_gates": source_gates,
        "larger_preregistered_run_justified": bool(all(source_gates.values())),
        "prospective_candidate_lock": {
            "path": deployment_lock_path.name,
            "sha256": _sha256(deployment_lock_path),
            "candidate_certified": deployment_lock["candidate_certified"],
            "finite_sample_coverage": deployment_lock[
                "finite_sample_coverage"
            ],
            "calibrated_90_percent_claim_allowed": deployment_lock[
                "calibrated_90_percent_claim_allowed"
            ],
        },
        "calibrated_90_percent_claim_ready": bool(
            group_guard["minimum_finite_sample_coverage"] is not None
            and group_guard["minimum_finite_sample_coverage"] >= 0.90
        ),
        "artifacts": artifacts,
        "input_roots": {
            "sealed_source": str(source),
            "prefix_measurement": str(measurement),
            "prefix_uncertainty": str(uncertainty),
            "selected_raw_baseline": str(selected_baseline),
        },
        "claim_boundary": (
            "All outcomes were open before this method was designed. This run may "
            "calibrate and reject a future protocol, but cannot confirm accuracy, "
            "calibration, safety, or state of the art."
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "Deform360BiasAwareDevelopmentConfig",
    "FEATURE_NAMES",
    "PROTOCOL_ID",
    "evaluate_deform360_bias_aware_development",
    "predict_bias_aware_candidate_arrays",
]
