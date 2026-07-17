"""Trust-gated reusable PhysTwin response for Deform360 episode generalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _candidate_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-6))


@dataclass(frozen=True)
class ReusableTwinTrustDecision:
    alpha: float
    raw_alpha: float
    closure_accepted: bool
    closure_value: float


@dataclass(frozen=True)
class Deform360ReusableTwinTrustCandidate:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: float
    closure_feature: str
    closure_mode: str
    closure_threshold: float | None
    reference_response_alpha: float
    maximum_alpha: float
    result_sha256: str

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        _require(feature_count > 0, "trust candidate has no features")
        _require(
            self.coefficients.shape == (feature_count + 1,),
            "trust coefficient shape does not match feature count",
        )
        _require(
            self.feature_mean.shape == (feature_count,),
            "trust feature mean shape does not match feature count",
        )
        _require(
            self.feature_scale.shape == (feature_count,),
            "trust feature scale shape does not match feature count",
        )
        _require(
            np.all(np.isfinite(self.coefficients)), "trust coefficients are not finite"
        )
        _require(
            np.all(np.isfinite(self.feature_mean)), "trust feature mean is not finite"
        )
        _require(
            np.all(np.isfinite(self.feature_scale))
            and np.all(self.feature_scale > 0.0),
            "trust feature scale must be finite and positive",
        )
        _require(self.ridge > 0.0, "trust ridge must be positive")
        _require(
            self.reference_response_alpha > 0.0,
            "reference response alpha must be positive",
        )
        _require(self.maximum_alpha >= 0.0, "maximum alpha must be nonnegative")
        _require(
            self.closure_mode in {"accept_all", "accept_none", "threshold"},
            "unknown closure mode",
        )
        if self.closure_mode == "threshold":
            _require(
                self.closure_threshold is not None
                and np.isfinite(self.closure_threshold),
                "threshold closure mode requires a finite threshold",
            )
        else:
            _require(
                self.closure_threshold is None,
                "non-threshold closure mode cannot carry a threshold",
            )

    def decide(self, features: Mapping[str, float]) -> ReusableTwinTrustDecision:
        missing = [name for name in self.feature_names if name not in features]
        _require(not missing, f"trust features are missing {missing}")
        _require(
            self.closure_feature in features,
            f"closure feature {self.closure_feature!r} is missing",
        )
        vector = np.asarray([features[name] for name in self.feature_names], dtype=float)
        _require(np.all(np.isfinite(vector)), "trust features are not finite")
        normalized = (vector - self.feature_mean) / self.feature_scale
        raw_alpha = float(self.coefficients[0] + normalized @ self.coefficients[1:])
        clipped_alpha = float(np.clip(raw_alpha, 0.0, self.maximum_alpha))
        closure_value = float(features[self.closure_feature])
        _require(np.isfinite(closure_value), "closure feature is not finite")
        if self.closure_mode == "accept_all":
            closure_accepted = True
        elif self.closure_mode == "accept_none":
            closure_accepted = False
        else:
            assert self.closure_threshold is not None
            closure_accepted = closure_value >= self.closure_threshold
        return ReusableTwinTrustDecision(
            alpha=clipped_alpha if closure_accepted else 0.0,
            raw_alpha=raw_alpha,
            closure_accepted=closure_accepted,
            closure_value=closure_value,
        )


def load_reusable_twin_trust_candidate(
    path: str | Path,
) -> Deform360ReusableTwinTrustCandidate:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "trust artifact must contain an object")
    if payload.get("artifact_kind") == "Deform360SameObjectTrustDiagnosis":
        payload = payload.get("full_source_candidate")
        _require(isinstance(payload, dict), "diagnosis lacks full-source candidate")
    _require(
        payload.get("artifact_kind") == "Deform360ReusableTwinTrustCandidate",
        "unexpected trust artifact kind",
    )
    result_sha256 = str(payload.get("result_sha256", ""))
    _require(result_sha256 == _candidate_sha256(payload), "trust checksum mismatch")
    closure_rule = payload.get("closure_rule")
    _require(isinstance(closure_rule, dict), "trust closure rule is missing")
    threshold = closure_rule.get("threshold")
    return Deform360ReusableTwinTrustCandidate(
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        ridge=float(payload["ridge"]),
        closure_feature=str(payload["closure_feature"]),
        closure_mode=str(closure_rule["mode"]),
        closure_threshold=None if threshold is None else float(threshold),
        reference_response_alpha=float(payload["reference_response_alpha"]),
        maximum_alpha=float(payload["maximum_alpha"]),
        result_sha256=result_sha256,
    )


def deform360_closure_features(
    openings_m: np.ndarray, frame_count: int
) -> dict[str, float]:
    openings = np.asarray(openings_m, dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        openings.ndim == 2 and len(openings) >= frame_count,
        "robot openings must contain the prediction horizon",
    )
    low = np.quantile(openings, 0.1, axis=0)
    high = np.quantile(openings, 0.9, axis=0)
    scale = np.maximum(high - low, 1e-9)
    closure = np.clip((high - openings[:frame_count]) / scale, 0.0, 1.0)
    minimum = np.min(closure, axis=1)
    return {
        "mean_closure": float(np.mean(closure)),
        "mean_minimum_gripper_closure": float(np.mean(minimum)),
        "all_grippers_closed_fraction_075": float(np.mean(minimum >= 0.75)),
    }


def deform360_robot_action_features(
    actions_m: np.ndarray, openings_m: np.ndarray
) -> dict[str, float]:
    action = np.asarray(actions_m, dtype=np.float64)
    openings = np.asarray(openings_m, dtype=np.float64)
    if action.ndim == 3 and action.shape[1:] == (5, 3):
        action = action[:, None]
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        action.ndim == 4 and action.shape[2:] == (5, 3),
        f"unexpected robot action shape: {action.shape}",
    )
    _require(
        openings.shape == action.shape[:2],
        f"robot opening shape {openings.shape} does not match {action.shape[:2]}",
    )
    centres = np.mean(action, axis=2)
    order = np.argsort(centres[0, :, 0])
    centres = centres[:, order]
    openings = openings[:, order]
    steps = np.diff(centres, axis=0)
    delta = centres[-1] - centres[0]
    axis_path = np.sum(np.abs(steps), axis=0)
    path = np.sum(np.linalg.norm(steps, axis=-1), axis=0)
    features: dict[str, float] = {
        "gripper_count": float(centres.shape[1]),
        "bimanual": float(centres.shape[1] == 2),
        "mean_gripper_path_m": float(np.mean(path)),
        "max_gripper_path_m": float(np.max(path)),
        "mean_endpoint_displacement_m": float(np.mean(np.linalg.norm(delta, axis=-1))),
        "mean_vertical_displacement_m": float(np.mean(delta[:, 2])),
        "mean_absolute_vertical_displacement_m": float(np.mean(np.abs(delta[:, 2]))),
        "mean_horizontal_displacement_m": float(
            np.mean(np.linalg.norm(delta[:, :2], axis=-1))
        ),
        "mean_vertical_path_m": float(np.mean(axis_path[:, 2])),
        "mean_horizontal_path_m": float(
            np.mean(np.linalg.norm(axis_path[:, :2], axis=-1))
        ),
        "mean_opening_start_m": float(np.mean(openings[0])),
        "mean_opening_end_m": float(np.mean(openings[-1])),
        "mean_opening_change_m": float(np.mean(openings[-1] - openings[0])),
        "minimum_opening_m": float(np.min(openings)),
    }
    features["vertical_to_horizontal_path_ratio"] = _safe_ratio(
        features["mean_vertical_path_m"], features["mean_horizontal_path_m"]
    )
    for gripper_index in range(min(2, centres.shape[1])):
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            features[f"gripper_{gripper_index}_delta_{axis_name}_m"] = float(
                delta[gripper_index, axis_index]
            )
            features[f"gripper_{gripper_index}_axis_path_{axis_name}_m"] = float(
                axis_path[gripper_index, axis_index]
            )
    for gripper_index in range(centres.shape[1], 2):
        for axis_name in ("x", "y", "z"):
            features[f"gripper_{gripper_index}_delta_{axis_name}_m"] = 0.0
            features[f"gripper_{gripper_index}_axis_path_{axis_name}_m"] = 0.0
    if centres.shape[1] == 2:
        separation = np.linalg.norm(centres[:, 1] - centres[:, 0], axis=-1)
        features.update(
            {
                "gripper_separation_start_m": float(separation[0]),
                "gripper_separation_end_m": float(separation[-1]),
                "gripper_separation_change_m": float(separation[-1] - separation[0]),
                "gripper_separation_range_m": float(np.ptp(separation)),
                "gripper_delta_dot_m2": float(np.dot(delta[0], delta[1])),
            }
        )
    else:
        features.update(
            {
                "gripper_separation_start_m": 0.0,
                "gripper_separation_end_m": 0.0,
                "gripper_separation_change_m": 0.0,
                "gripper_separation_range_m": 0.0,
                "gripper_delta_dot_m2": 0.0,
            }
        )
    return features


def deform360_response_features(
    response_m: np.ndarray,
    persistence_m: np.ndarray,
    robot_features: Mapping[str, float],
) -> dict[str, float]:
    response = np.asarray(response_m, dtype=np.float64)
    persistence = np.asarray(persistence_m, dtype=np.float64)
    _require(
        response.ndim == 3 and response.shape[-1] == 3,
        "response must have shape (frames, points, 3)",
    )
    _require(persistence.shape == response.shape, "persistence and response differ")
    _require(len(response) >= 76, "trust model requires at least 76 response frames")
    relative = response - response[0:1]
    displacement = np.linalg.norm(relative, axis=-1)
    frame_rms = np.sqrt(np.mean(relative**2, axis=(1, 2)))
    geometry = persistence[0]
    extents = np.ptp(geometry, axis=0)
    singular_values = np.linalg.svd(
        geometry - np.mean(geometry, axis=0, keepdims=True),
        full_matrices=False,
        compute_uv=False,
    )
    pca_scales = singular_values / np.sqrt(max(len(geometry) - 1, 1))
    features = {
        "response_rms_m": float(np.sqrt(np.mean(relative[1:76] ** 2))),
        "response_mean_displacement_m": float(np.mean(displacement[1:76])),
        "response_early_mean_displacement_m": float(np.mean(displacement[1:26])),
        "response_late_mean_displacement_m": float(np.mean(displacement[51:76])),
        "response_endpoint_mean_displacement_m": float(np.mean(displacement[75])),
        "response_p90_displacement_m": float(np.quantile(displacement[1:76], 0.9)),
        "response_max_displacement_m": float(np.max(displacement[1:76])),
        "response_frame_rms_growth_m": float(frame_rms[75] - frame_rms[1]),
        "geometry_extent_x_m": float(extents[0]),
        "geometry_extent_y_m": float(extents[1]),
        "geometry_extent_z_m": float(extents[2]),
        "geometry_diagonal_m": float(np.linalg.norm(extents)),
        "geometry_pca_0_m": float(pca_scales[0]),
        "geometry_pca_1_m": float(pca_scales[1]),
        "geometry_pca_2_m": float(pca_scales[2]),
    }
    features["response_late_to_early_ratio"] = _safe_ratio(
        features["response_late_mean_displacement_m"],
        features["response_early_mean_displacement_m"],
    )
    features["response_to_action_path_ratio"] = _safe_ratio(
        features["response_mean_displacement_m"],
        float(robot_features["mean_gripper_path_m"]),
    )
    features["response_to_geometry_ratio"] = _safe_ratio(
        features["response_mean_displacement_m"], features["geometry_diagonal_m"]
    )
    return features


def build_deform360_trust_features(
    actions_m: np.ndarray,
    openings_m: np.ndarray,
    response_m: np.ndarray,
    persistence_m: np.ndarray,
) -> dict[str, float]:
    features = deform360_closure_features(openings_m, len(persistence_m))
    robot_features = deform360_robot_action_features(actions_m, openings_m)
    features.update(robot_features)
    features.update(
        deform360_response_features(response_m, persistence_m, robot_features)
    )
    return features


__all__ = [
    "Deform360ReusableTwinTrustCandidate",
    "ReusableTwinTrustDecision",
    "build_deform360_trust_features",
    "deform360_closure_features",
    "deform360_response_features",
    "deform360_robot_action_features",
    "load_reusable_twin_trust_candidate",
]
