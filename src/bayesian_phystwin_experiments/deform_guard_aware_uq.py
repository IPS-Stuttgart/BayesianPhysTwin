"""Fixed-mean Gaussian second-moment reporting for a guarded DEFORM forecast."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform_multiobject_restart import load_protocol as load_parent
from .deform_state_restart import array_digest, file_digest
from .deform_weak_constraint_belief import calibration_scales, scaled_covariance

EXPERIMENT = "deform-guard-aware-uq-v1"
PROTOCOL = "configs/sota/deform_guard_aware_uq_v1.json"
RAW_ARMS = (
    "isotropic",
    "guard_scaled",
    "shadow",
    "fixed_mean_bridge",
    "rotated_bridge",
)
FAMILIES = (*RAW_ARMS, "source_full")
VARIANTS = ("moment", "conformal")
PRIMARY = "fixed_mean_bridge__moment"
ROTATION = np.asarray(((0.0, 1, 0), (0.0, 0, 1), (1.0, 0, 0)))


@dataclass(frozen=True)
class GuardUQConfig:
    floor_std_m: float = 0.003
    source_full_ridge_std_m: float = 0.000001
    bootstrap_seed: int = 260835
    bootstrap_replicates: int = 10000

    def __post_init__(self) -> None:
        if (
            any(
                not np.isfinite(x) or x <= 0
                for x in (self.floor_std_m, self.source_full_ridge_std_m)
            )
            or self.bootstrap_replicates < 1
        ):
            raise ValueError(
                "positive finite uncertainty and bootstrap scales required"
            )


DEFAULT_CONFIG = GuardUQConfig()


def load_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text())
    expected = {
        "schema": EXPERIMENT,
        "scope": "exploratory-only-on-the-already-open-three-object-roster",
        "point_mean": "previous_paired_8",
        "shadow_belief": "strong_8",
        "primary_arm": PRIMARY,
        "run_root": "/home/florianpfaff/source-only/deform-guard-aware-uq-v1/run-v1",
        "attempt_ledger": "/home/florianpfaff/source-only/deform-guard-aware-uq-v1/prediction_attempt.json",
        "raw_covariance_arms": list(RAW_ARMS),
        "source_fitted_comparator": "source_full",
        "calibration_variants": list(VARIANTS),
        "config": asdict(GuardUQConfig()),
        "input_root": "/home/florianpfaff/source-only/deform-weak-belief-b87eea1e/run-v1",
        "input_source_revision": "b87eea1e9477ad2b5c4445691f8b1af1b353a701",
        "input_prediction_barrier_sha256": "1fe14811ffc26731584e0c601003bacb9e5634a52cd456250621c6f260700045",
        "input_verification_path": "/home/florianpfaff/source-only/deform-weak-belief-b87eea1e/verification_v1_1.json",
        "input_verification_sha256": "82190bec5ee0b442a0c66af5b19a7d627ff55d09ffad045f588f67c79ee0b4d9",
        "all_raw_predictions_sealed_before_new_calibration": True,
        "calibration_sealed_before_transfer_metrics": True,
        "shared_point_mean_only": True,
        "new_native_rollouts": False,
        "new_observation_queries": False,
        "future_free_node_truth_is_model_input": False,
        "future_clamped_actions_known_under_parent_contract": True,
        "protected_data_access": False,
        "new_official_evaluation": False,
        "population_coverage_guarantee": False,
        "exact_posterior_claim": False,
        "automatic_promotion": False,
        "outcome_publication": "local-or-private-paper-evidence-only",
        "technical_failure_policy": "retain and stop; no replacement or outcome-driven rerun",
        "calibration": {
            "object": "DLO2",
            "case_count": 13,
            "excluded_design_case": "103.pkl",
            "horizons": [40, 40, 40],
            "source_full": "equal-case-uncentered-error-second-moment-plus-fixed-ridge",
            "moment": "mean-source-NEES-divided-by-three",
            "conformal": "source-trajectory-90th-percentile-higher-then-maximum-divided-by-chi2_3_90",
            "outer_rank": 13,
        },
        "gate": {
            "objects": ["DLO1", "DLO3"],
            "nll_ci95_upper_below_zero_against": ["isotropic", "source_full", "shadow"],
            "minimum_nll_wins_per_object": 5,
            "wins_compared_against": ["isotropic", "source_full"],
            "coverage_range": [0.8, 0.98],
            "volume_nonincreasing_against": ["isotropic", "source_full"],
            "point_arrays_byte_identical": True,
            "all_required_checks_on_both_objects": True,
            "secondary_arms_cannot_rescue_primary": True,
        },
    }
    if set(value) != {*expected, "parent_protocol"} or any(
        value[k] != v for k, v in expected.items()
    ):
        raise ValueError("frozen uncertainty method or information boundary changed")
    parent = value["parent_protocol"]
    if (
        parent
        != {
            "path": "configs/sota/deform_multiobject_state_restart_v1.json",
            "sha256": "6ea78a8d4c412d1fbd3423e388bbfd85b78696e619a29ab6a961ab807e08af17",
        }
        or file_digest(root / parent["path"]) != parent["sha256"]
    ):
        raise ValueError("opened cohort, incumbent, or checkpoint contract changed")
    return value, load_parent(root / parent["path"])


def validate_covariance(covariance: np.ndarray) -> None:
    if (
        covariance.shape[-2:] != (3, 3)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=0, atol=1e-12)
    ):
        raise ValueError("finite symmetric metric 3x3 covariances required")
    np.linalg.cholesky(covariance)


def fixed_mean_second_moment(
    shadow_mean: np.ndarray, covariance: np.ndarray, deployed_mean: np.ndarray
) -> np.ndarray:
    if (
        shadow_mean.shape != deployed_mean.shape
        or shadow_mean.shape[-1:] != (3,)
        or covariance.shape != (*shadow_mean.shape, 3)
        or not np.isfinite(shadow_mean).all()
        or not np.isfinite(deployed_mean).all()
    ):
        raise ValueError(
            "shadow and deployed means must share finite metric identities"
        )
    validate_covariance(covariance)
    bias = shadow_mean.astype(np.float64) - deployed_mean.astype(np.float64)
    result = covariance + bias[..., :, None] * bias[..., None, :]
    validate_covariance(result)
    return result


def build_prediction(
    incumbent: np.ndarray,
    deployed_mean: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    posterior: np.ndarray,
    gains: np.ndarray,
    *,
    registered_mean_sha256: str,
    config: GuardUQConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    batch = len(deployed_mean)
    if (
        deployed_mean.ndim != 4
        or deployed_mean.shape[1] != 120
        or deployed_mean.shape[-1] != 3
        or incumbent.shape != deployed_mean.shape
        or response.shape != (*deployed_mean.shape, 24)
        or coefficients.shape != (batch, 27)
        or posterior.shape != (batch, 27, 27)
        or gains.shape != (batch,)
        or array_digest(deployed_mean) != registered_mean_sha256
    ):
        raise ValueError(
            "registered mean, response, or joint state/bias dimensions differ"
        )
    if any(
        not np.isfinite(x).all()
        for x in (incumbent, deployed_mean, response, coefficients, posterior, gains)
    ) or np.any((gains < 0) | (gains > 1)):
        raise ValueError("nonfinite belief or invalid mean guard")
    if not np.allclose(posterior, posterior.swapaxes(-1, -2), atol=1e-12, rtol=0) or (
        np.linalg.eigvalsh(posterior).min() < -1e-9
    ):
        raise ValueError("joint posterior must retain a symmetric PSD physical block")
    mu = incumbent.astype(np.float64) + np.einsum(
        "btnik,bk->btni", response, coefficients[:, :24]
    )
    tangent = np.einsum(
        "btnik,bkl,btnjl->btnij", response, posterior[:, :24, :24], response
    )
    tangent = (tangent + tangent.swapaxes(-1, -2)) / 2
    isotropic = np.broadcast_to(
        config.floor_std_m**2 * np.eye(3), (*deployed_mean.shape, 3)
    ).copy()
    covariance = tangent + isotropic
    bridge = fixed_mean_second_moment(mu, covariance, deployed_mean)
    result = {
        "mean": deployed_mean,
        "shadow_mean": mu,
        "isotropic": isotropic,
        "guard_scaled": tangent * gains[:, None, None, None, None] ** 2 + isotropic,
        "shadow": covariance,
        "fixed_mean_bridge": bridge,
        "rotated_bridge": np.einsum("ij,...jk,lk->...il", ROTATION, bridge, ROTATION),
    }
    for arm in RAW_ARMS:
        validate_covariance(result[arm])
    if array_digest(result["mean"]) != registered_mean_sha256:
        raise ValueError("uncertainty construction changed the deployed mean")
    return result


def source_full_matrices(
    error: np.ndarray, *, object_name: str, config: GuardUQConfig = DEFAULT_CONFIG
) -> np.ndarray:
    if (
        object_name != "DLO2"
        or error.shape != (13, 120, 4, 3)
        or not np.isfinite(error).all()
    ):
        raise ValueError(
            "source covariance requires exactly thirteen DLO2 trajectories"
        )
    result = []
    for frames in np.array_split(np.arange(120), 3):
        e = error[:, frames]
        per_case = np.einsum("btni,btnj->bij", e, e) / (len(frames) * 4)
        result.append(
            per_case.mean(axis=0) + config.source_full_ridge_std_m**2 * np.eye(3)
        )
    value = np.stack(result)
    validate_covariance(value)
    return value


def expand_source_full(matrices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if (
        matrices.shape != (3, 3, 3)
        or len(shape) != 5
        or shape[1] != 120
        or shape[-2:] != (3, 3)
    ):
        raise ValueError("source full covariance needs three aligned forecast horizons")
    validate_covariance(matrices)
    result = np.empty(shape)
    for frames, matrix in zip(np.array_split(np.arange(120), 3), matrices, strict=True):
        result[:, frames] = matrix
    return result


def calibrate_source(
    error: np.ndarray, raw: dict[str, np.ndarray], *, object_name: str
) -> dict[str, Any]:
    if (
        object_name != "DLO2"
        or error.shape != (13, 120, 4, 3)
        or set(raw) != set(RAW_ARMS)
    ):
        raise ValueError("calibration input or source-only denominator changed")
    matrices = source_full_matrices(error, object_name=object_name)
    covariances = {
        **raw,
        "source_full": expand_source_full(matrices, (*error.shape, 3)),
    }
    return {
        "source_full_matrices_m2": matrices.tolist(),
        "scales": {
            arm: calibration_scales(error, covariance, object_name=object_name)
            for arm, covariance in covariances.items()
        },
    }


def calibrated_covariance(
    raw: dict[str, np.ndarray], calibration: dict[str, Any], family: str, variant: str
) -> np.ndarray:
    if family not in FAMILIES or variant not in VARIANTS:
        raise ValueError("unregistered covariance arm")
    covariance = (
        expand_source_full(
            np.asarray(calibration["source_full_matrices_m2"]), raw["isotropic"].shape
        )
        if family == "source_full"
        else raw[family]
    )
    return scaled_covariance(covariance, calibration["scales"][family][variant])


def primary_decision(
    results: dict[str, Any],
    *,
    mean_identity: bool,
    accounted_cases: int,
    config: GuardUQConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if set(results) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("all opened objects are required for a decision")
    objects = {}
    for name in ("DLO1", "DLO3"):
        arms = results[name]["uq"]
        if set(arms) != {f"{a}__{b}" for a in FAMILIES for b in VARIANTS}:
            raise ValueError("all frozen same-mean controls are required")
        candidate = arms[PRIMARY]
        values = np.asarray(candidate["per_case"]["nll"])
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("complete eight-trajectory transfer denominator required")
        checks = {
            "point_mean_byte_identical": mean_identity is True,
            "all_30_predictions_accounted": accounted_cases == 30,
            "coverage_between_80_and_98_percent": 0.8
            <= candidate["summary"]["coverage_90"]
            <= 0.98,
        }
        intervals = {}
        draws = np.random.default_rng(config.bootstrap_seed).integers(
            0, 8, (config.bootstrap_replicates, 8)
        )
        for comparator in ("isotropic", "source_full", "shadow"):
            control = arms[comparator + "__moment"]
            other = np.asarray(control["per_case"]["nll"])
            if other.shape != (8,) or not np.isfinite(other).all():
                raise ValueError("incomplete comparator trajectory denominator")
            delta = values - other
            interval = np.quantile(delta[draws].mean(axis=1), (0.025, 0.975)).tolist()
            intervals[comparator] = interval
            checks["nll_ci95_upper_negative_vs_" + comparator] = interval[1] < 0
            if comparator != "shadow":
                checks["at_least_five_nll_wins_vs_" + comparator] = (
                    int(np.sum(delta < 0)) >= 5
                )
                checks["volume_nonincreasing_vs_" + comparator] = (
                    candidate["summary"]["ellipsoid_volume_mm3"]
                    <= control["summary"]["ellipsoid_volume_mm3"]
                )
        objects[name] = {"checks": checks, "nll_delta_ci95": intervals}
    return {
        "primary_arm": PRIMARY,
        "objects": objects,
        "development_advancement_gate_passed": all(
            all(x["checks"].values()) for x in objects.values()
        ),
        "point_mean_changed": False,
        "secondary_arms_cannot_rescue_primary": True,
        "automatic_target_authorization": False,
        "incumbent_promoted": False,
        "population_confirmation_or_sota_claim": False,
    }
