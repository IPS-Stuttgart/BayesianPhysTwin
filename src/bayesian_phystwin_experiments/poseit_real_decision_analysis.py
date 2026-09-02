"""Frozen fit, abstention, and object-level analysis for the PoseIt study."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .poseit_real_decision_protocol import (
    CALIBRATION_COUNT,
    CONFIRMATION_COUNT,
    FIT_COUNT,
    MANDATORY_ANCHOR,
    SELECTABLE_POSES,
    SOURCE_TEST_COUNT,
    canonical_object_token,
)
from .poseit_real_decision_selectors import (
    POSE_COUNT,
    PoseItGaussianState,
    PoseItPolicyTrace,
    pose_stability_index,
    stability_probabilities,
    trace_policy,
    trace_probe_order,
)

FloatArray: TypeAlias = NDArray[np.float64]
PCA_COMPONENT_COUNT = 8
COVARIANCE_DIAGONAL_SHRINKAGE = 0.25
COVARIANCE_JITTER_FRACTION = 1e-8
CERTIFICATE_COVERAGE = 0.8
CERTIFICATE_THRESHOLD = 0.5
FIXED_ORDER_COUNT = 256
FIXED_ORDER_DOMAIN = "poseit-real-decision-fixed-probe-order-v1"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260902


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _family_token(value: str) -> str:
    token = str(value).strip()
    _require(bool(token), "family token is empty")
    return token


def _readonly_vector(value: Sequence[float] | FloatArray) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    _require(vector.ndim == 1 and bool(len(vector)), "pose feature must be a vector")
    _require(bool(np.all(np.isfinite(vector))), "pose feature is non-finite")
    result = vector.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PoseItFeatureFamily:
    """Pre-shake records for one object, grasp location, and gripper force."""

    object_token: str
    family_token: str
    pre_shake_features: Mapping[int, Sequence[float] | FloatArray]

    def __post_init__(self) -> None:
        object_token = canonical_object_token(self.object_token)
        family_token = _family_token(self.family_token)
        features = {
            int(pose): _readonly_vector(values)
            for pose, values in self.pre_shake_features.items()
        }
        _require(bool(features), "feature family is empty")
        _require(MANDATORY_ANCHOR in features, "mandatory anchor is absent")
        _require(
            set(features) <= set(range(1, POSE_COUNT + 1)),
            "feature family contains an unregistered pose",
        )
        dimensions = {len(values) for values in features.values()}
        _require(len(dimensions) == 1, "raw feature dimension changed within family")
        canonical = {pose: features[pose] for pose in sorted(features)}
        object.__setattr__(self, "object_token", object_token)
        object.__setattr__(self, "family_token", family_token)
        object.__setattr__(self, "pre_shake_features", MappingProxyType(canonical))

    @property
    def available_poses(self) -> tuple[int, ...]:
        return tuple(self.pre_shake_features)

    @property
    def raw_feature_dimension(self) -> int:
        return len(next(iter(self.pre_shake_features.values())))

    @property
    def key(self) -> tuple[str, str]:
        return self.object_token, self.family_token


@dataclass(frozen=True)
class PoseItLabeledFamily:
    """A feature family plus shake outcomes used only by fit or scoring code."""

    features: PoseItFeatureFamily
    shake_stable: Mapping[int, bool]

    def __post_init__(self) -> None:
        labels: dict[int, bool] = {}
        for pose, stable in self.shake_stable.items():
            _require(
                isinstance(stable, (bool, np.bool_)),
                "shake stability label is not boolean",
            )
            labels[int(pose)] = bool(stable)
        _require(
            set(labels) == set(self.features.available_poses),
            "shake-label roster differs from structurally available poses",
        )
        object.__setattr__(
            self,
            "shake_stable",
            MappingProxyType({pose: labels[pose] for pose in sorted(labels)}),
        )

    @property
    def object_token(self) -> str:
        return self.features.object_token

    @property
    def family_token(self) -> str:
        return self.features.family_token

    @property
    def key(self) -> tuple[str, str]:
        return self.features.key


@dataclass(frozen=True)
class PoseItFeatureProjector:
    """Fit-only standardization and deterministic eight-component PCA."""

    center: FloatArray
    scale: FloatArray
    components: FloatArray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        _require(center.ndim == 1 and bool(len(center)), "projector center is invalid")
        _require(scale.shape == center.shape, "projector scale shape changed")
        _require(
            components.shape == (PCA_COMPONENT_COUNT, len(center)),
            "projector component shape changed",
        )
        _require(bool(np.all(np.isfinite(center))), "projector center is non-finite")
        _require(bool(np.all(np.isfinite(scale))), "projector scale is non-finite")
        _require(bool(np.all(scale > 0.0)), "projector scale is not positive")
        _require(
            bool(np.all(np.isfinite(components))),
            "projector component is non-finite",
        )
        _require(
            np.allclose(
                components @ components.T,
                np.eye(PCA_COMPONENT_COUNT),
                rtol=0.0,
                atol=1e-10,
            ),
            "projector components are not orthonormal",
        )
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "scale", scale.copy())
        object.__setattr__(self, "components", components.copy())

    @classmethod
    def fit(cls, families: Sequence[PoseItFeatureFamily]) -> PoseItFeatureProjector:
        _require(bool(families), "fit feature-family roster is empty")
        dimensions = {family.raw_feature_dimension for family in families}
        _require(len(dimensions) == 1, "raw feature dimension changed across fit data")
        raw_dimension = next(iter(dimensions))
        _require(
            raw_dimension >= PCA_COMPONENT_COUNT,
            "raw feature dimension is below the frozen PCA dimension",
        )
        records = np.stack(
            [
                family.pre_shake_features[pose]
                for family in families
                for pose in family.available_poses
            ]
        )
        _require(
            len(records) > PCA_COMPONENT_COUNT,
            "too few fit records for frozen PCA",
        )
        center = np.mean(records, axis=0)
        scale = np.std(records, axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        standardized = (records - center) / scale
        _, _, right = np.linalg.svd(standardized, full_matrices=False)
        _require(
            len(right) >= PCA_COMPONENT_COUNT,
            "fit matrix rank surface is below frozen PCA dimension",
        )
        components = np.asarray(right[:PCA_COMPONENT_COUNT], dtype=np.float64).copy()
        for index in range(PCA_COMPONENT_COUNT):
            pivot = int(np.argmax(np.abs(components[index])))
            if components[index, pivot] < 0.0:
                components[index] *= -1.0
        return cls(center=center, scale=scale, components=components)

    def transform(self, values: Sequence[float] | FloatArray) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        _require(vector.shape == self.center.shape, "raw feature shape changed")
        _require(bool(np.all(np.isfinite(vector))), "raw feature is non-finite")
        return np.asarray(
            self.components @ ((vector - self.center) / self.scale),
            dtype=np.float64,
        )

    def transform_family(self, family: PoseItFeatureFamily) -> dict[int, FloatArray]:
        _require(
            family.raw_feature_dimension == len(self.center),
            "family raw feature dimension changed",
        )
        return {
            pose: self.transform(values)
            for pose, values in family.pre_shake_features.items()
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "components": self.components.tolist(),
            "component_count": PCA_COMPONENT_COUNT,
            "fit_only": True,
            "outcome_used": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoseItFeatureProjector:
        _require(
            payload.get("component_count") == PCA_COMPONENT_COUNT,
            "projector component count changed",
        )
        _require(payload.get("fit_only") is True, "projector is not fit-only")
        _require(payload.get("outcome_used") is False, "outcome entered projector")
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            components=np.asarray(payload["components"], dtype=np.float64),
        )


@dataclass(frozen=True)
class PoseItGaussianTwin:
    """Empirical Gaussian twin fitted only on complete fit-object families."""

    projector: PoseItFeatureProjector
    mean: FloatArray
    covariance: FloatArray
    covariance_diagonal_shrinkage: float = COVARIANCE_DIAGONAL_SHRINKAGE
    jitter_fraction_of_median_variance: float = COVARIANCE_JITTER_FRACTION

    def __post_init__(self) -> None:
        state = PoseItGaussianState(
            mean=np.asarray(self.mean, dtype=np.float64),
            covariance=np.asarray(self.covariance, dtype=np.float64),
            feature_dimension=PCA_COMPONENT_COUNT,
        )
        _require(
            self.covariance_diagonal_shrinkage == COVARIANCE_DIAGONAL_SHRINKAGE,
            "covariance shrinkage changed",
        )
        _require(
            self.jitter_fraction_of_median_variance == COVARIANCE_JITTER_FRACTION,
            "covariance jitter changed",
        )
        eigenvalues = np.linalg.eigvalsh(state.covariance)
        _require(float(np.min(eigenvalues)) > 0.0, "twin covariance is not positive")
        object.__setattr__(self, "mean", state.mean)
        object.__setattr__(self, "covariance", state.covariance)

    @classmethod
    def fit(cls, families: Sequence[PoseItLabeledFamily]) -> PoseItGaussianTwin:
        _validate_unique_families(families)
        objects = {family.object_token for family in families}
        _require(len(objects) == FIT_COUNT, "fit object count changed")
        complete = set(range(1, POSE_COUNT + 1))
        _require(
            all(
                set(family.features.available_poses) == complete for family in families
            ),
            "fit family is structurally incomplete",
        )
        projector = PoseItFeatureProjector.fit([family.features for family in families])
        rows: list[FloatArray] = []
        for family in families:
            transformed = projector.transform_family(family.features)
            blocks: list[FloatArray] = []
            for pose in range(1, POSE_COUNT + 1):
                latent = 1.0 if family.shake_stable[pose] else -1.0
                blocks.append(
                    np.concatenate(
                        (transformed[pose], np.asarray([latent], dtype=np.float64))
                    )
                )
            rows.append(np.concatenate(blocks))
        matrix = np.stack(rows)
        _require(len(matrix) >= 3, "too few complete fit families")
        mean = np.mean(matrix, axis=0)
        sample_covariance = np.asarray(
            np.cov(matrix, rowvar=False, ddof=1), dtype=np.float64
        )
        diagonal = np.diag(np.diag(sample_covariance))
        covariance = (
            1.0 - COVARIANCE_DIAGONAL_SHRINKAGE
        ) * sample_covariance + COVARIANCE_DIAGONAL_SHRINKAGE * diagonal
        positive = np.diag(covariance)[np.diag(covariance) > 0.0]
        _require(bool(len(positive)), "fit covariance has no positive variance")
        jitter = max(
            COVARIANCE_JITTER_FRACTION * float(np.median(positive)),
            np.finfo(np.float64).eps,
        )
        covariance = covariance + jitter * np.eye(len(covariance))
        return cls(projector=projector, mean=mean, covariance=covariance)

    def prior(self, available_poses: Sequence[int]) -> PoseItGaussianState:
        return PoseItGaussianState(
            mean=self.mean,
            covariance=self.covariance,
            feature_dimension=PCA_COMPONENT_COUNT,
            available_poses=tuple(sorted(int(pose) for pose in available_poses)),
        )

    def transform_family(self, family: PoseItFeatureFamily) -> dict[int, FloatArray]:
        return self.projector.transform_family(family)

    def as_dict(self) -> dict[str, Any]:
        return {
            "projector": self.projector.as_dict(),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "covariance_diagonal_shrinkage": self.covariance_diagonal_shrinkage,
            "jitter_fraction_of_median_variance": (
                self.jitter_fraction_of_median_variance
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoseItGaussianTwin:
        projector = payload.get("projector")
        if not isinstance(projector, Mapping):
            raise ValueError("projector artifact is missing")
        return cls(
            projector=PoseItFeatureProjector.from_dict(
                cast(Mapping[str, Any], projector)
            ),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
            covariance_diagonal_shrinkage=float(
                payload["covariance_diagonal_shrinkage"]
            ),
            jitter_fraction_of_median_variance=float(
                payload["jitter_fraction_of_median_variance"]
            ),
        )


def _validate_unique_families(families: Sequence[PoseItLabeledFamily]) -> None:
    _require(bool(families), "labeled family roster is empty")
    keys = [family.key for family in families]
    _require(len(keys) == len(set(keys)), "labeled family repeated")


def registered_fixed_probe_orders() -> tuple[tuple[int, ...], ...]:
    """Return the 256 source-independent hash-derived fixed orders."""

    orders: list[tuple[int, ...]] = []
    for index in range(FIXED_ORDER_COUNT):

        def key(pose: int, *, order_index: int = index) -> tuple[str, int]:
            message = f"{FIXED_ORDER_DOMAIN}\0{order_index}\0{pose}".encode()
            return hashlib.sha256(message).hexdigest(), pose

        orders.append(tuple(sorted(SELECTABLE_POSES, key=key)))
    _require(len(set(orders)) == FIXED_ORDER_COUNT, "fixed probe orders collide")
    return tuple(orders)


def fixed_probe_order_roster_sha256() -> str:
    payload = json.dumps(
        registered_fixed_probe_orders(),
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _registered_traces(
    twin: PoseItGaussianTwin,
    family: PoseItFeatureFamily,
) -> tuple[tuple[str, PoseItPolicyTrace], ...]:
    transformed = twin.transform_family(family)
    prior = twin.prior(family.available_poses)
    traces: list[tuple[str, PoseItPolicyTrace]] = [
        (
            "decision_directed",
            trace_policy(prior, transformed, selector="decision_directed"),
        ),
        (
            "system_identification",
            trace_policy(prior, transformed, selector="system_identification"),
        ),
        (
            "lexicographic_fixed",
            trace_probe_order(
                prior,
                transformed,
                prior.action_poses,
                selector="lexicographic_fixed",
            ),
        ),
    ]
    available = set(prior.action_poses)
    fixed_trace_cache: dict[tuple[int, ...], PoseItPolicyTrace] = {}
    for index, full_order in enumerate(registered_fixed_probe_orders()):
        order = tuple(pose for pose in full_order if pose in available)
        trace = fixed_trace_cache.get(order)
        if trace is None:
            trace = trace_probe_order(
                prior,
                transformed,
                order,
                selector=f"hash_fixed_{index:03d}",
            )
            fixed_trace_cache[order] = trace
        named_trace = PoseItPolicyTrace(
            selector=f"hash_fixed_{index:03d}",
            selected_poses=trace.selected_poses,
            states=trace.states,
        )
        traces.append(
            (
                f"hash_fixed_{index:03d}",
                named_trace,
            )
        )
    return tuple(traces)


def _object_shortfall_score(
    twin: PoseItGaussianTwin,
    families: Sequence[PoseItLabeledFamily],
) -> float:
    score = 0.0
    for family in families:
        for _, trace in _registered_traces(twin, family.features):
            for state in trace.states:
                poses = state.action_poses
                latent_mean, latent_standard_deviation = _latent_moments(state, poses)
                observed_latent = np.asarray(
                    [1.0 if family.shake_stable[pose] else -1.0 for pose in poses],
                    dtype=np.float64,
                )
                score = max(
                    score,
                    float(
                        np.max(
                            np.maximum(
                                0.0,
                                (latent_mean - observed_latent)
                                / latent_standard_deviation,
                            )
                        )
                    ),
                )
    _require(np.isfinite(score), "calibration score is non-finite")
    return score


def calibrate_shared_stability_shortfall(
    twin: PoseItGaussianTwin,
    calibration_families: Sequence[PoseItLabeledFamily],
) -> tuple[float, tuple[float, ...], int]:
    """Calibrate one simultaneous lower-bound shortfall at object level."""

    _validate_unique_families(calibration_families)
    grouped = _group_families_by_object(calibration_families)
    _require(len(grouped) == CALIBRATION_COUNT, "calibration object count changed")
    scores = tuple(
        _object_shortfall_score(twin, grouped[object_token])
        for object_token in sorted(grouped)
    )
    rank = min(len(scores), math.ceil((len(scores) + 1) * CERTIFICATE_COVERAGE))
    _require(rank == 5, "finite-sample calibration rank changed")
    shortfall = float(np.sort(np.asarray(scores, dtype=np.float64))[rank - 1])
    return shortfall, scores, rank


@dataclass(frozen=True)
class PoseItDecisionMethod:
    """Frozen twin and shared object-calibrated abstention certificate."""

    twin: PoseItGaussianTwin
    stability_multiplier: float
    calibration_scores: tuple[float, ...]
    calibration_rank: int

    def __post_init__(self) -> None:
        _require(
            np.isfinite(self.stability_multiplier) and self.stability_multiplier >= 0.0,
            "stability multiplier is invalid",
        )
        _require(
            len(self.calibration_scores) == CALIBRATION_COUNT,
            "calibration score count changed",
        )
        _require(
            all(
                np.isfinite(score) and score >= 0.0 for score in self.calibration_scores
            ),
            "calibration score is invalid",
        )
        _require(self.calibration_rank == 5, "calibration rank changed")

    @classmethod
    def fit(
        cls,
        fit_families: Sequence[PoseItLabeledFamily],
        calibration_families: Sequence[PoseItLabeledFamily],
    ) -> PoseItDecisionMethod:
        fit_objects = {family.object_token for family in fit_families}
        calibration_objects = {family.object_token for family in calibration_families}
        _require(
            not (fit_objects & calibration_objects),
            "fit and calibration objects overlap",
        )
        twin = PoseItGaussianTwin.fit(fit_families)
        shortfall, scores, rank = calibrate_shared_stability_shortfall(
            twin, calibration_families
        )
        return cls(
            twin=twin,
            stability_multiplier=shortfall,
            calibration_scores=scores,
            calibration_rank=rank,
        )

    def lower_stability_bounds(
        self,
        state: PoseItGaussianState,
        poses: Sequence[int] | None = None,
    ) -> FloatArray:
        selected = (
            state.action_poses if poses is None else tuple(int(pose) for pose in poses)
        )
        mean, standard_deviation = _latent_moments(state, selected)
        return _normal_cdf(mean / standard_deviation - self.stability_multiplier)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PoseItRealDecisionMethodV1",
            "twin": self.twin.as_dict(),
            "stability_multiplier": self.stability_multiplier,
            "calibration_scores": list(self.calibration_scores),
            "calibration_rank": self.calibration_rank,
            "fixed_probe_order_roster_sha256": fixed_probe_order_roster_sha256(),
            "confirmation_opened": False,
            "held_v8_accessed": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoseItDecisionMethod:
        _require(
            payload.get("artifact_kind") == "PoseItRealDecisionMethodV1",
            "method artifact kind changed",
        )
        _require(payload.get("confirmation_opened") is False, "confirmation opened")
        _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
        _require(
            payload.get("fixed_probe_order_roster_sha256")
            == fixed_probe_order_roster_sha256(),
            "fixed probe-order roster changed",
        )
        twin = payload.get("twin")
        if not isinstance(twin, Mapping):
            raise ValueError("twin artifact is missing")
        return cls(
            twin=PoseItGaussianTwin.from_dict(cast(Mapping[str, Any], twin)),
            stability_multiplier=float(payload["stability_multiplier"]),
            calibration_scores=tuple(
                float(value) for value in payload["calibration_scores"]
            ),
            calibration_rank=int(payload["calibration_rank"]),
        )


def _group_families_by_object(
    families: Sequence[PoseItLabeledFamily],
) -> dict[str, tuple[PoseItLabeledFamily, ...]]:
    grouped: dict[str, list[PoseItLabeledFamily]] = {}
    for family in families:
        grouped.setdefault(family.object_token, []).append(family)
    return {
        object_token: tuple(sorted(rows, key=lambda row: row.family_token))
        for object_token, rows in grouped.items()
    }


def _normal_cdf(value: FloatArray) -> FloatArray:
    values = np.asarray(value, dtype=np.float64)
    return np.asarray(
        0.5
        * (
            1.0
            + np.asarray(
                [math.erf(float(item) / math.sqrt(2.0)) for item in values.ravel()],
                dtype=np.float64,
            ).reshape(values.shape)
        ),
        dtype=np.float64,
    )


def _latent_moments(
    state: PoseItGaussianState,
    poses: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    selected = tuple(int(pose) for pose in poses)
    _require(set(selected) <= set(state.action_poses), "latent pose is unavailable")
    indices = np.asarray(
        [pose_stability_index(pose, state.feature_dimension) for pose in selected],
        dtype=int,
    )
    mean = state.mean[indices]
    variance = np.diag(state.covariance[np.ix_(indices, indices)])
    _require(bool(np.all(variance > 0.0)), "latent predictive variance vanished")
    return mean, np.sqrt(variance)


def _evaluate_state(
    method: PoseItDecisionMethod,
    state: PoseItGaussianState,
    shake_stable: Mapping[int, bool],
) -> dict[str, Any]:
    poses = state.action_poses
    probabilities = stability_probabilities(state, poses)
    lower = method.lower_stability_bounds(state, poses)
    latent_mean, latent_standard_deviation = _latent_moments(state, poses)
    lower_latent = latent_mean - method.stability_multiplier * latent_standard_deviation
    certified = tuple(
        pose
        for pose, bound in zip(poses, lower, strict=True)
        if float(bound) >= CERTIFICATE_THRESHOLD
    )
    selected = certified[0] if certified else None
    selected_stable = None if selected is None else bool(shake_stable[selected])
    realized_utility = (
        0.0 if selected_stable is None else (1.0 if selected_stable else -1.0)
    )
    stable_poses = tuple(pose for pose in poses if shake_stable[pose])
    oracle = stable_poses[0] if stable_poses else None
    oracle_utility = 1.0 if oracle is not None else 0.0
    observed_latent = np.asarray(
        [1.0 if shake_stable[pose] else -1.0 for pose in poses]
    )
    covered = bool(np.all(lower_latent <= observed_latent + 1e-12))
    unsafe = bool(selected is not None and not selected_stable)
    return {
        "selected_pose": selected,
        "certified_poses": list(certified),
        "oracle_pose": oracle,
        "realized_utility": realized_utility,
        "oracle_utility": oracle_utility,
        "regret": oracle_utility - realized_utility,
        "abstained": selected is None,
        "unsafe": unsafe,
        "false_safe": unsafe,
        "simultaneous_stability_covered": covered,
        "posterior_stability_probability": {
            str(pose): float(value)
            for pose, value in zip(poses, probabilities, strict=True)
        },
        "lower_stability_bound": {
            str(pose): float(value) for pose, value in zip(poses, lower, strict=True)
        },
        "lower_latent_stability": {
            str(pose): float(value)
            for pose, value in zip(poses, lower_latent, strict=True)
        },
    }


def _trace_result(
    method: PoseItDecisionMethod,
    trace: PoseItPolicyTrace,
    family: PoseItLabeledFamily,
) -> dict[str, Any]:
    budgets = [
        _evaluate_state(method, state, family.shake_stable) for state in trace.states
    ]
    regrets = np.asarray([record["regret"] for record in budgets], dtype=np.float64)
    return {
        "selector": trace.selector,
        "selected_poses": list(trace.selected_poses),
        "budgets": budgets,
        "regret_auc": float(np.trapezoid(regrets, dx=1.0)),
        "simultaneous_stability_covered_all_budgets": bool(
            all(record["simultaneous_stability_covered"] for record in budgets)
        ),
    }


def evaluate_poseit_family(
    method: PoseItDecisionMethod,
    family: PoseItLabeledFamily,
) -> dict[str, Any]:
    """Evaluate every frozen policy on one already-authorized family."""

    traces = _registered_traces(method.twin, family.features)
    results = {name: _trace_result(method, trace, family) for name, trace in traces}
    fixed = [results[f"hash_fixed_{index:03d}"] for index in range(FIXED_ORDER_COUNT)]
    fixed_budget_regret = np.asarray(
        [[float(budget["regret"]) for budget in result["budgets"]] for result in fixed],
        dtype=np.float64,
    )
    return {
        "object_token": family.object_token,
        "family_token": family.family_token,
        "decision_directed": results["decision_directed"],
        "system_identification": results["system_identification"],
        "lexicographic_fixed": results["lexicographic_fixed"],
        "hash_fixed_order_mean": {
            "order_count": FIXED_ORDER_COUNT,
            "order_roster_sha256": fixed_probe_order_roster_sha256(),
            "mean_budget_regret": np.mean(fixed_budget_regret, axis=0).tolist(),
            "mean_regret_auc": float(
                np.mean([float(result["regret_auc"]) for result in fixed])
            ),
            "simultaneous_stability_covered_all_orders_and_budgets": bool(
                all(
                    result["simultaneous_stability_covered_all_budgets"]
                    for result in fixed
                )
            ),
        },
    }


def _bootstrap_mean_interval(values: FloatArray) -> tuple[float, float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPETITIONS, len(values)),
    )
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _one_sided_exact_sign_flip_p(differences: FloatArray) -> float:
    _require(
        differences.shape == (CONFIRMATION_COUNT,),
        "confirmation contrast object count changed",
    )
    observed = float(np.mean(differences))
    count = 0
    total = 1 << len(differences)
    for mask in range(total):
        signs = np.asarray(
            [
                1.0 if not (mask & (1 << index)) else -1.0
                for index in range(len(differences))
            ]
        )
        permuted = float(np.mean(signs * differences))
        count += int(permuted <= observed + 1e-15)
    return count / total


def summarize_poseit_evaluation(
    family_results: Sequence[Mapping[str, Any]],
    *,
    expected_object_count: int,
    confirmation: bool = False,
) -> dict[str, Any]:
    """Aggregate family outcomes within objects before cross-object inference."""

    _require(bool(family_results), "family result roster is empty")
    keys = [
        (str(record["object_token"]), str(record["family_token"]))
        for record in family_results
    ]
    _require(len(keys) == len(set(keys)), "family result repeated")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in family_results:
        grouped.setdefault(str(record["object_token"]), []).append(record)
    _require(len(grouped) == expected_object_count, "evaluation object count changed")
    if confirmation:
        _require(
            expected_object_count == CONFIRMATION_COUNT,
            "confirmation object count changed",
        )

    object_rows: list[dict[str, Any]] = []
    for object_token in sorted(grouped):
        rows = grouped[object_token]
        decision_auc = float(
            np.mean([float(row["decision_directed"]["regret_auc"]) for row in rows])
        )
        identification_auc = float(
            np.mean([float(row["system_identification"]["regret_auc"]) for row in rows])
        )
        covered = all(
            bool(row[selector]["simultaneous_stability_covered_all_budgets"])
            for row in rows
            for selector in (
                "decision_directed",
                "system_identification",
                "lexicographic_fixed",
            )
        ) and all(
            bool(
                row["hash_fixed_order_mean"][
                    "simultaneous_stability_covered_all_orders_and_budgets"
                ]
            )
            for row in rows
        )
        object_rows.append(
            {
                "object_token": object_token,
                "family_count": len(rows),
                "decision_directed_regret_auc": decision_auc,
                "system_identification_regret_auc": identification_auc,
                "paired_auc_difference": decision_auc - identification_auc,
                "simultaneous_stability_covered": covered,
            }
        )

    decision_auc = np.asarray(
        [row["decision_directed_regret_auc"] for row in object_rows],
        dtype=np.float64,
    )
    identification_auc = np.asarray(
        [row["system_identification_regret_auc"] for row in object_rows],
        dtype=np.float64,
    )
    differences = decision_auc - identification_auc
    decision_budgets = [
        cast(Mapping[str, Any], budget)
        for row in family_results
        for budget in cast(Mapping[str, Any], row["decision_directed"])["budgets"]
    ]
    identification_budgets = [
        cast(Mapping[str, Any], budget)
        for row in family_results
        for budget in cast(Mapping[str, Any], row["system_identification"])["budgets"]
    ]
    nonabstaining = [record for record in decision_budgets if not record["abstained"]]
    false_safe_rate = (
        0.0
        if not nonabstaining
        else float(np.mean([bool(record["false_safe"]) for record in nonabstaining]))
    )
    mean_identification = float(np.mean(identification_auc))
    relative_improvement = (
        0.0
        if mean_identification <= 0.0
        else (mean_identification - float(np.mean(decision_auc))) / mean_identification
    )
    selected = {
        int(pose)
        for row in family_results
        for pose in cast(Mapping[str, Any], row["decision_directed"])["selected_poses"]
    }
    interval = _bootstrap_mean_interval(differences)
    summary: dict[str, Any] = {
        "object_count": len(object_rows),
        "family_count": len(family_results),
        "objects": object_rows,
        "decision_directed_mean_regret_auc": float(np.mean(decision_auc)),
        "system_identification_mean_regret_auc": mean_identification,
        "paired_mean_auc_difference": float(np.mean(differences)),
        "paired_mean_auc_difference_bootstrap_95": list(interval),
        "relative_auc_improvement": relative_improvement,
        "object_improvement_count": int(np.count_nonzero(differences < -1e-12)),
        "object_tie_count": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "object_regression_count": int(np.count_nonzero(differences > 1e-12)),
        "object_level_certificate_coverage": float(
            np.mean([row["simultaneous_stability_covered"] for row in object_rows])
        ),
        "decision_directed_false_safe_rate": false_safe_rate,
        "decision_directed_unsafe_action_rate": float(
            np.mean([bool(record["unsafe"]) for record in decision_budgets])
        ),
        "system_identification_unsafe_action_rate": float(
            np.mean([bool(record["unsafe"]) for record in identification_budgets])
        ),
        "decision_directed_certified_action_coverage": float(
            np.mean([not bool(record["abstained"]) for record in decision_budgets])
        ),
        "decision_directed_abstention_rate": float(
            np.mean([bool(record["abstained"]) for record in decision_budgets])
        ),
        "selected_probe_count": len(selected),
        "selected_poses": sorted(selected),
    }
    if confirmation:
        summary["one_sided_exact_paired_sign_flip_p"] = _one_sided_exact_sign_flip_p(
            differences
        )
    return summary


def poseit_source_promotion_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen five-object source gate without authorizing confirmation."""

    _require(
        int(summary["object_count"]) == SOURCE_TEST_COUNT,
        "source-test object count changed",
    )
    checks = {
        "relative_auc_improvement": float(summary["relative_auc_improvement"]) >= 0.05,
        "object_improvement_count": int(summary["object_improvement_count"]) >= 4,
        "object_level_certificate_coverage": float(
            summary["object_level_certificate_coverage"]
        )
        >= 0.8,
        "false_safe_rate": float(summary["decision_directed_false_safe_rate"]) <= 0.2,
        "unsafe_action_rate_noninferiority": float(
            summary["decision_directed_unsafe_action_rate"]
        )
        <= float(summary["system_identification_unsafe_action_rate"]) + 0.1,
        "selected_probe_diversity": int(summary["selected_probe_count"]) >= 2,
    }
    return {
        "artifact_kind": "PoseItRealDecisionSourceGateV1",
        "checks": checks,
        "passed": all(checks.values()),
        "target_authorized": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }


def poseit_confirmation_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen one-shot six-object confirmation result."""

    _require(
        int(summary["object_count"]) == CONFIRMATION_COUNT,
        "confirmation object count changed",
    )
    checks = {
        "decision_directed_auc_below_system_identification": float(
            summary["decision_directed_mean_regret_auc"]
        )
        < float(summary["system_identification_mean_regret_auc"]),
        "relative_auc_improvement": float(summary["relative_auc_improvement"]) >= 0.05,
        "one_sided_exact_paired_sign_flip": float(
            summary["one_sided_exact_paired_sign_flip_p"]
        )
        <= 0.05,
        "object_level_certificate_coverage": float(
            summary["object_level_certificate_coverage"]
        )
        >= 0.8,
        "false_safe_rate": float(summary["decision_directed_false_safe_rate"]) <= 0.2,
        "unsafe_action_rate_noninferiority": float(
            summary["decision_directed_unsafe_action_rate"]
        )
        <= float(summary["system_identification_unsafe_action_rate"]) + 0.1,
    }
    return {
        "artifact_kind": "PoseItRealDecisionConfirmationGateV1",
        "checks": checks,
        "passed": all(checks.values()),
        "attempt_limit": 1,
        "retry_authorized": False,
        "held_v8_accessed": False,
    }


__all__ = [
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "CERTIFICATE_COVERAGE",
    "CERTIFICATE_THRESHOLD",
    "COVARIANCE_DIAGONAL_SHRINKAGE",
    "COVARIANCE_JITTER_FRACTION",
    "FIXED_ORDER_COUNT",
    "FIXED_ORDER_DOMAIN",
    "PCA_COMPONENT_COUNT",
    "PoseItDecisionMethod",
    "PoseItFeatureFamily",
    "PoseItFeatureProjector",
    "PoseItGaussianTwin",
    "PoseItLabeledFamily",
    "calibrate_shared_stability_shortfall",
    "evaluate_poseit_family",
    "fixed_probe_order_roster_sha256",
    "poseit_confirmation_gate",
    "poseit_source_promotion_gate",
    "registered_fixed_probe_orders",
    "summarize_poseit_evaluation",
]
