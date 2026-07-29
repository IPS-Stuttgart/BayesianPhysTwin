"""Synthetic positive and placebo controls for the frozen V12 update."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
    evaluate_causal_response_admission,
)
from .deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    build_causal_response_measurements,
    predict_causal_response_candidate,
)
from .deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
)

CONTRACT = "deform360-causal-response-synthetic-controls-v12"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-synthetic-controls-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponseSyntheticConfig:
    """Frozen size and outcome boundary for synthetic controls."""

    trial_count: int = 12
    node_count: int = 12
    frame_count: int = 14
    birth_frame: int = 2
    update_frame: int = 8
    observation_standard_deviation_m: float = 0.001

    def __post_init__(self) -> None:
        _require(self.trial_count >= 4, "too few synthetic trials")
        _require(self.node_count >= 6, "too few synthetic nodes")
        _require(
            0 <= self.birth_frame < self.update_frame < self.frame_count - 1,
            "synthetic frame contract is invalid",
        )
        _require(
            np.isfinite(self.observation_standard_deviation_m)
            and self.observation_standard_deviation_m > 0.0,
            "synthetic observation scale is invalid",
        )


@dataclass(frozen=True)
class CausalResponseSyntheticTrial:
    """One sealed synthetic prediction followed by its synthetic outcome."""

    trial_index: int
    control_kind: str
    admitted: bool
    admission_reason: str
    candidate_applied: bool
    exact_baseline_fallback: bool
    baseline_future_rmse_m: float | None
    candidate_future_rmse_m: float | None

    def __post_init__(self) -> None:
        _require(self.trial_index >= 0, "synthetic trial index is negative")
        _require(
            self.control_kind
            in {"positive-nonrigid", "placebo-rigid", "placebo-cross-panel"},
            "synthetic control kind is invalid",
        )
        _require(bool(self.admission_reason.strip()), "admission reason is empty")
        for value in (
            self.baseline_future_rmse_m,
            self.candidate_future_rmse_m,
        ):
            _require(
                value is None or (np.isfinite(value) and value >= 0.0),
                "synthetic outcome is invalid",
            )
        _require(
            (self.baseline_future_rmse_m is None)
            == (self.candidate_future_rmse_m is None),
            "synthetic outcome pair is incomplete",
        )


@dataclass(frozen=True)
class CausalResponseSyntheticResult:
    """Checksummed detection-power and placebo-specificity evidence."""

    config: CausalResponseSyntheticConfig
    trials: tuple[CausalResponseSyntheticTrial, ...]
    positive_detection_count: int
    placebo_admission_count: int
    placebo_exact_fallback_count: int
    mean_positive_baseline_rmse_m: float
    mean_positive_candidate_rmse_m: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        expected = 2 * self.config.trial_count
        _require(len(self.trials) == expected, "synthetic trial count changed")
        _require(
            self.positive_detection_count >= 0
            and self.placebo_admission_count >= 0
            and self.placebo_exact_fallback_count >= 0,
            "synthetic aggregate count is negative",
        )
        _require(
            np.isfinite(self.mean_positive_baseline_rmse_m)
            and np.isfinite(self.mean_positive_candidate_rmse_m)
            and self.mean_positive_baseline_rmse_m >= 0.0
            and self.mean_positive_candidate_rmse_m >= 0.0,
            "synthetic aggregate RMSE is invalid",
        )
        _require(
            len(self.artifact_sha256) == 64
            and all(
                character in "0123456789abcdef" for character in self.artifact_sha256
            ),
            "synthetic result digest is invalid",
        )

    @property
    def positive_detection_rate(self) -> float:
        return self.positive_detection_count / self.config.trial_count

    @property
    def placebo_false_admission_rate(self) -> float:
        return self.placebo_admission_count / self.config.trial_count

    @property
    def positive_improvement_fraction(self) -> float:
        return 1.0 - (
            self.mean_positive_candidate_rmse_m
            / max(self.mean_positive_baseline_rmse_m, 1e-12)
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseSyntheticControls",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "trials": [asdict(trial) for trial in self.trials],
            "positive_detection_count": self.positive_detection_count,
            "positive_detection_rate": self.positive_detection_rate,
            "placebo_admission_count": self.placebo_admission_count,
            "placebo_false_admission_rate": self.placebo_false_admission_rate,
            "placebo_exact_fallback_count": self.placebo_exact_fallback_count,
            "mean_positive_baseline_rmse_m": self.mean_positive_baseline_rmse_m,
            "mean_positive_candidate_rmse_m": self.mean_positive_candidate_rmse_m,
            "positive_improvement_fraction": self.positive_improvement_fraction,
            "information_boundary": {
                "real_object_observation_read": False,
                "real_identity_or_metric_read": False,
                "synthetic_mechanism_defined_before_prediction": True,
                "synthetic_future_instantiated_and_scored_after_prediction": True,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def _baseline_and_residual(
    config: CausalResponseSyntheticConfig,
    trial_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = config.node_count
    x = np.linspace(-0.05, 0.05, node_count)
    y = 0.02 * np.sin(np.linspace(0.0, 2.0 * np.pi, node_count))
    frame_zero = np.column_stack((x, y, np.full(node_count, 0.8)))
    phase = 0.15 * trial_index
    per_frame = np.column_stack(
        (
            0.0005 * np.cos(np.linspace(phase, np.pi + phase, node_count)),
            0.0004 * np.sin(np.linspace(phase, np.pi + phase, node_count)),
            0.00015 * np.cos(np.linspace(0.0, 2.0 * np.pi, node_count) + phase),
        )
    )
    baseline = np.stack(
        [frame_zero + frame * per_frame for frame in range(config.frame_count)]
    )
    action_displacement = baseline[config.update_frame] - baseline[config.birth_frame]
    local_scale = 0.4 + 1.1 * (
        0.5
        + 0.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, node_count) + 0.3 * trial_index)
    )
    residual = 1.0 * action_displacement * local_scale[:, None]
    return baseline, residual


def _observations(
    baseline: np.ndarray,
    residual: np.ndarray,
    config: CausalResponseSyntheticConfig,
    *,
    residual_multiplier: float,
    global_translation_m: np.ndarray | None = None,
) -> DirectDepthEndpointObservations:
    frames = np.asarray(
        [config.birth_frame, config.update_frame],
        dtype=np.int64,
    )
    entity_ids = np.arange(config.node_count, dtype=np.int64)
    points = baseline[frames].copy()
    points[1] += residual_multiplier * residual
    if global_translation_m is not None:
        points[1] += np.asarray(global_translation_m, dtype=np.float64)
    variance = config.observation_standard_deviation_m**2
    covariance = np.repeat(
        (variance * np.eye(3))[None, None],
        2 * config.node_count,
        axis=0,
    ).reshape(2, config.node_count, 3, 3)
    return DirectDepthEndpointObservations(
        endpoint_frames=frames,
        entity_ids=entity_ids,
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=np.ones((2, config.node_count), dtype=bool),
        association_probability=np.full((2, config.node_count), 0.9),
        support_count=np.full((2, config.node_count), 3, dtype=np.int64),
        maximum_view_scatter_m=np.zeros((2, config.node_count)),
        config=DirectDepthEndpointConfig(),
    )


def _evaluate_trial(
    config: CausalResponseSyntheticConfig,
    trial_index: int,
    control_kind: str,
) -> CausalResponseSyntheticTrial:
    baseline, residual = _baseline_and_residual(config, trial_index)
    if control_kind == "positive-nonrigid":
        proposal = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=1.0,
        )
        validation = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=0.95,
        )
    elif control_kind == "placebo-rigid":
        translation = np.asarray([0.012, -0.008, 0.004])
        proposal = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=0.0,
            global_translation_m=translation,
        )
        validation = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=0.0,
            global_translation_m=translation,
        )
    else:
        proposal = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=1.0,
        )
        validation = _observations(
            baseline,
            residual,
            config,
            residual_multiplier=-0.75,
        )
    admission = evaluate_causal_response_admission(
        f"synthetic-{trial_index:02d}",
        baseline,
        proposal,
        validation,
        np.full(config.node_count, 0.8),
        proposal_camera_ids=("proposal-0", "proposal-1", "proposal-2"),
        validation_camera_ids=("validation-0", "validation-1", "validation-2"),
        tactile_contact_probability=1.0,
        actuator_displacement_m=0.01,
        config=CausalResponseAdmissionConfig(),
    )
    measurements = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
    )
    report, arrays = predict_causal_response_candidate(
        baseline,
        measurements,
        admission,
    )
    exact_fallback = (
        arrays[CANDIDATE_ARM].dtype == arrays[BASELINE_ARM].dtype
        and arrays[CANDIDATE_ARM].shape == arrays[BASELINE_ARM].shape
        and arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes()
    )
    if control_kind == "positive-nonrigid":
        synthetic_truth = baseline.copy()
        synthetic_truth[config.update_frame + 1 :] += residual
        future = slice(config.update_frame + 1, None)
        baseline_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(arrays[BASELINE_ARM][future] - synthetic_truth[future])
                )
            )
        )
        candidate_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(arrays[CANDIDATE_ARM][future] - synthetic_truth[future])
                )
            )
        )
    else:
        baseline_rmse = None
        candidate_rmse = None
    return CausalResponseSyntheticTrial(
        trial_index=trial_index,
        control_kind=control_kind,
        admitted=admission.admitted,
        admission_reason=admission.reason,
        candidate_applied=bool(report["candidate_applied"]),
        exact_baseline_fallback=exact_fallback,
        baseline_future_rmse_m=baseline_rmse,
        candidate_future_rmse_m=candidate_rmse,
    )


def run_causal_response_synthetic_controls(
    config: CausalResponseSyntheticConfig | None = None,
) -> CausalResponseSyntheticResult:
    """Run matched positive and placebo controls without any real-data access."""

    cfg = config or CausalResponseSyntheticConfig()
    positive = tuple(
        _evaluate_trial(cfg, index, "positive-nonrigid")
        for index in range(cfg.trial_count)
    )
    placebo = tuple(
        _evaluate_trial(
            cfg,
            index,
            "placebo-rigid" if index % 2 == 0 else "placebo-cross-panel",
        )
        for index in range(cfg.trial_count)
    )
    baseline_rmse = np.asarray(
        [trial.baseline_future_rmse_m for trial in positive],
        dtype=np.float64,
    )
    candidate_rmse = np.asarray(
        [trial.candidate_future_rmse_m for trial in positive],
        dtype=np.float64,
    )
    provisional = CausalResponseSyntheticResult(
        config=cfg,
        trials=positive + placebo,
        positive_detection_count=sum(trial.candidate_applied for trial in positive),
        placebo_admission_count=sum(trial.admitted for trial in placebo),
        placebo_exact_fallback_count=sum(
            trial.exact_baseline_fallback for trial in placebo
        ),
        mean_positive_baseline_rmse_m=float(np.mean(baseline_rmse)),
        mean_positive_candidate_rmse_m=float(np.mean(candidate_rmse)),
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = CausalResponseSyntheticResult(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "synthetic result changed after construction",
    )
    return result


def write_causal_response_synthetic_result(
    path: str | Path,
    result: CausalResponseSyntheticResult,
) -> None:
    """Write one immutable synthetic-control result."""

    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "synthetic result checksum changed",
    )
    output = Path(path)
    _require(not output.exists(), "synthetic result output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CONTRACT",
    "CausalResponseSyntheticConfig",
    "CausalResponseSyntheticResult",
    "CausalResponseSyntheticTrial",
    "run_causal_response_synthetic_controls",
    "write_causal_response_synthetic_result",
]
