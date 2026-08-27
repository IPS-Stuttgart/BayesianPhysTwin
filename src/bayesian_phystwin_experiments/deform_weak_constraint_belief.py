"""Weak-constraint prefix inference and marginal calibration for opened DEFORM data.

This is an approximate local Gaussian smoother, not a new constitutive model.
The frozen native mean, object-specific checkpoints, and old results are immutable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform_forecast_sensing import (
    LockedQueryBank,
    SensingConfig,
    infer_coefficients,
    material_basis,
    query_pairs,
)
from .deform_multiobject_restart import load_protocol as load_parent
from .deform_state_restart import (
    RestartConfig,
    file_digest,
    interpolate_material_residual,
)

PROTOCOL = "configs/sota/deform_weak_constraint_belief_v1.json"
EXPERIMENT = "deform-weak-constraint-belief-v1"
NATIVE_ARMS = ("strong_8", "weak_8", "strong_16", "weak_16")
ARMS = (
    "incumbent",
    "previous_paired_8",
    "ols_physical_16",
    "ols_readout_16",
    "periodic_pose_16",
    *NATIVE_ARMS,
)
HORIZONS = ("early", "middle", "late")
CHI2_3_90 = 6.251388631170325
CALIBRATION_FAMILIES = {
    "weak_16_shaped": "weak_16",
    "weak_16_isotropic": "weak_16",
    "previous_paired_8_isotropic": "previous_paired_8",
}


@dataclass(frozen=True)
class BeliefConfig:
    anchor_frame: int = 25
    observation_frames: tuple[int, ...] = (25, 33, 41, 49)
    position_std_m: float = 0.01
    velocity_std_m_s: float = 0.1
    process_velocity_std_m_s: float = 0.05
    process_position_velocity_ratio_s: float = 0.04
    shared_bias_std_m: float = 0.005
    measurement_std_m: float = 0.001
    finite_difference_fraction: float = 0.1
    maximum_total_position_increment_m: float = 0.03
    maximum_total_velocity_increment_m_s: float = 0.3
    covariance_floor_std_m: float = 0.003
    calibration_bootstrap_replicates: int = 10000
    calibration_bootstrap_seed: int = 260834

    def __post_init__(self) -> None:
        if self.anchor_frame != 25 or self.observation_frames != (25, 33, 41, 49):
            raise ValueError("only the fixed public prefix is allowed")
        scales = (
            self.position_std_m,
            self.velocity_std_m_s,
            self.process_velocity_std_m_s,
            self.process_position_velocity_ratio_s,
            self.shared_bias_std_m,
            self.measurement_std_m,
            self.finite_difference_fraction,
            self.maximum_total_position_increment_m,
            self.maximum_total_velocity_increment_m_s,
            self.covariance_floor_std_m,
        )
        if any(not np.isfinite(x) or x <= 0 for x in scales):
            raise ValueError("belief scales must be positive and finite")
        if self.calibration_bootstrap_replicates < 1:
            raise ValueError("trajectory bootstrap must be nonempty")


def config_record(config: BeliefConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config)))


def load_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text())
    expected = {
        "schema": EXPERIMENT,
        "scope": "exploratory-only-on-the-already-open-three-object-roster",
        "primary_arm": "weak_16",
        "calibration_object": "DLO2",
        "calibration_case_count": 13,
        "transfer_objects": ["DLO1", "DLO3"],
        "arms": list(ARMS),
        "belief": config_record(BeliefConfig()),
        "all_predictions_sealed_before_new_metrics": True,
        "calibration_sealed_before_transfer_metrics": True,
        "protected_data_access": False,
        "new_official_evaluation": False,
        "checkpoint_or_readout_refitting": False,
        "automatic_promotion": False,
        "future_free_node_truth_is_model_input": False,
        "observation_schedule_adaptation": False,
        "population_coverage_guarantee": False,
    }
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("frozen method, denominator, or boundary changed")
    parent_spec = value["parent_protocol"]
    parent_path = root / parent_spec["path"]
    if file_digest(parent_path) != parent_spec["sha256"]:
        raise ValueError("parent checkpoints, roster, or readout changed")
    if value["point_gate"] != {
        "minimum_rmse_gain_over_previous_and_ols_percent": 2.0,
        "minimum_joint_wins_over_previous_per_object": 5,
        "maximum_case_rmse_ratio_to_incumbent": 1.05,
        "late_rmse_nonincreasing_over_incumbent": True,
        "must_beat_all_matched_controls_on_both_metrics": True,
    }:
        raise ValueError("point advancement gate changed")
    if value["calibration"] != {
        "primary": "moment",
        "secondary": "conformal",
        "nominal_coverage": 0.9,
        "session_inner_quantile": 0.9,
        "quantile_method": "higher",
        "outer_rank": 13,
        "horizon_frames": [40, 40, 40],
        "minimum_coverage": 0.8,
        "maximum_coverage": 0.98,
        "same_mean_isotropic_nll_delta_ci95_upper_below_zero": True,
        "volume_no_larger_than_same_mean_isotropic": True,
    }:
        raise ValueError("calibration mechanism or gate changed")
    return value, load_parent(parent_path)


def impulse_basis(
    rod: RestartConfig, config: BeliefConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Four native injection times, 60 standardized physical coefficients."""
    basis = material_basis(rod)
    rank = len(basis)
    pose = np.zeros((4, 5 * rank, rod.node_count, 3))
    velocity = np.zeros_like(pose)
    pose[0, :rank] = config.position_std_m * basis
    velocity[0, rank : 2 * rank] = config.velocity_std_m_s * basis
    for step in range(1, 4):
        columns = slice((step + 1) * rank, (step + 2) * rank)
        velocity[step, columns] = config.process_velocity_std_m_s * basis
        pose[step, columns] = (
            config.process_position_velocity_ratio_s * velocity[step, columns]
        )
    return pose, velocity


def arm_columns(arm: str) -> tuple[int, ...]:
    if arm not in NATIVE_ARMS:
        raise ValueError("unknown strong/weak constraint arm")
    if arm.startswith("strong"):
        return tuple(range(24))
    return tuple(range(60)) if arm == "weak_16" else (*range(24), *range(48, 60))


def arm_schedule(arm: str) -> tuple[int, ...]:
    arm_columns(arm)
    return tuple(range(0 if arm.endswith("16") else 8, 16))


def validate_response(response: np.ndarray, rod: RestartConfig) -> None:
    if (
        response.shape != (145, rod.node_count, 3, 60)
        or not np.isfinite(response).all()
    ):
        raise ValueError("response must bind causal time, metric units, and identity")
    if np.any(response[:, rod.clamped_nodes]):
        raise ValueError("a prescribed clamp cannot carry a correction")
    for step, frame in enumerate((33, 41, 49), 1):
        if np.any(response[: frame - 25, :, :, (step + 1) * 12 : (step + 2) * 12]):
            raise ValueError("a process impulse has a pre-impulse response")


def query_design(
    response: np.ndarray, arm: str, rod: RestartConfig, config: BeliefConfig
) -> np.ndarray:
    validate_response(response, rod)
    columns = list(arm_columns(arm))
    rows = np.stack(
        [response[t - 25, n][:, columns] for t, n in query_pairs(rod, SensingConfig())]
    )
    nuisance = np.broadcast_to(config.shared_bias_std_m * np.eye(3), (16, 3, 3))
    return np.concatenate((rows, nuisance), axis=-1)


def infer_prefix(
    response: np.ndarray,
    reference: np.ndarray,
    observations: np.ndarray,
    arm: str,
    rod: RestartConfig,
    config: BeliefConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = query_pairs(rod, SensingConfig())
    schedule = arm_schedule(arm)
    bank = LockedQueryBank(observations, pairs, schedule)
    result = infer_coefficients(
        query_design(response, arm, rod, config),
        reference,
        bank,
        pairs,
        schedule,
        config.measurement_std_m,
    )
    if bank.access_log != [pairs[i] for i in schedule]:
        raise ValueError("measurement reveal differs from the frozen schedule")
    return result


def limit_impulses(
    pose: np.ndarray, velocity: np.ndarray, config: BeliefConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        pose.ndim != 4
        or velocity.shape != pose.shape
        or pose.shape[-1] != 3
        or not np.isfinite(pose).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("physical impulse arrays must be finite batch/time/node/3")
    displacement = np.linalg.norm(pose, axis=-1).sum(axis=1).max(axis=1)
    speed = np.linalg.norm(velocity, axis=-1).sum(axis=1).max(axis=1)
    gain = 1 / np.maximum(
        1,
        np.maximum(
            displacement / config.maximum_total_position_increment_m,
            speed / config.maximum_total_velocity_increment_m_s,
        ),
    )
    return pose * gain[:, None, None, None], velocity * gain[:, None, None, None], gain


def physical_impulses(
    coefficients: np.ndarray, arm: str, rod: RestartConfig, config: BeliefConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = list(arm_columns(arm))
    value = np.asarray(coefficients, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[1] != len(columns) + 3
        or not np.isfinite(value).all()
    ):
        raise ValueError("physical coefficients and shared bias must remain joint")
    pose, velocity = impulse_basis(rod, config)
    return limit_impulses(
        np.einsum("bk,tknd->btnd", value[:, :-3], pose[:, columns]),
        np.einsum("bk,tknd->btnd", value[:, :-3], velocity[:, columns]),
        config,
    )


def marginal_covariance(
    response: np.ndarray,
    posterior: np.ndarray,
    gains: np.ndarray,
    arm: str,
    config: BeliefConfig,
) -> np.ndarray:
    columns = list(arm_columns(arm))
    if (
        response.ndim != 5
        or response.shape[1] != 145
        or response.shape[-2:] != (3, 60)
        or posterior.shape != (len(response), len(columns) + 3, len(columns) + 3)
        or gains.shape != (len(response),)
        or not np.isfinite(response).all()
        or not np.isfinite(posterior).all()
        or not np.isfinite(gains).all()
        or np.any((gains < 0) | (gains > 1))
    ):
        raise ValueError("tangent covariance requires aligned finite physical inputs")
    if np.linalg.eigvalsh(posterior).min() < -1e-9:
        raise ValueError("latent posterior is not PSD")
    g = response[:, 25:, :, :, columns]
    covariance = np.einsum("btnik,bkl,btnjl->btnij", g, posterior[:, :-3, :-3], g)
    covariance *= gains[:, None, None, None, None] ** 2
    covariance += config.covariance_floor_std_m**2 * np.eye(3)
    return (covariance + covariance.swapaxes(-2, -1)) / 2


def ols_endpoint(
    incumbent: np.ndarray,
    observations: np.ndarray,
    rod: RestartConfig,
    config: BeliefConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if observations.shape != (len(incumbent), 16, 3):
        raise ValueError("OLS control must use the same sixteen observations")
    pairs = query_pairs(rod, SensingConfig())
    reference = np.stack([incumbent[:, t, n] for t, n in pairs], axis=1)
    residual = (observations.astype(np.float64) - reference).reshape(-1, 4, 4, 3)
    time = (np.asarray(config.observation_frames) - 49) * rod.dt_s
    design = np.stack((np.ones(4), time), axis=1)
    coefficients = np.linalg.lstsq(
        design, residual.transpose(1, 0, 2, 3).reshape(4, -1), rcond=None
    )[0].reshape(2, len(incumbent), 4, 3)
    pose = interpolate_material_residual(coefficients[0], rod)[:, None]
    velocity = interpolate_material_residual(coefficients[1], rod)[:, None]
    pose, velocity, gain = limit_impulses(pose, velocity, config)
    return pose[:, 0], velocity[:, 0], gain


def gaussian_events(error: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Three-dimensional marginal scores; no cross-identity independence claim."""
    if (
        error.shape[-1:] != (3,)
        or covariance.shape != (*error.shape, 3)
        or not np.isfinite(error).all()
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.swapaxes(-2, -1), atol=1e-12, rtol=0)
    ):
        raise ValueError("marginal covariance and error must align in metres/metres^2")
    np.linalg.cholesky(covariance)
    _, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, error[..., None])[..., 0]
    nees = np.einsum("...i,...i->...", error, solved)
    return {
        "nll": (3 * np.log(2 * np.pi) + logdet + nees) / 2,
        "nees": nees,
        "coverage_90": (nees <= CHI2_3_90).astype(float),
        "ellipsoid_volume_mm3": 4
        * np.pi
        / 3
        * CHI2_3_90**1.5
        * np.exp(logdet / 2)
        * 1e9,
        "geometric_full_width_mm": 2 * np.sqrt(CHI2_3_90) * np.exp(logdet / 6) * 1000,
    }


def calibration_scales(
    error: np.ndarray, covariance: np.ndarray, *, object_name: str
) -> dict[str, list[float]]:
    if object_name != "DLO2" or error.shape != (13, 120, 4, 3):
        raise ValueError("calibration is restricted to thirteen opened source cases")
    nees = gaussian_events(error, covariance)["nees"]
    result: dict[str, list[float]] = {"moment": [], "conformal": []}
    for frames in np.array_split(np.arange(120), 3):
        values = nees[:, frames].reshape(13, -1)
        result["moment"].append(max(1e-6, float(values.mean(axis=1).mean() / 3)))
        scores = np.quantile(values, 0.9, axis=1, method="higher")
        # ceil((13 + 1) * .9) = 13: the maximum source-trajectory score.
        result["conformal"].append(max(1e-6, float(np.sort(scores)[12] / CHI2_3_90)))
    return result


def scaled_covariance(covariance: np.ndarray, scales: list[float]) -> np.ndarray:
    if (
        covariance.ndim != 5
        or covariance.shape[1] != 120
        or covariance.shape[-2:] != (3, 3)
        or len(scales) != 3
        or any(not np.isfinite(x) or x <= 0 for x in scales)
    ):
        raise ValueError("three positive fixed horizon scales are required")
    result = covariance.copy()
    for frames, scale in zip(np.array_split(np.arange(120), 3), scales, strict=True):
        result[:, frames] *= scale
    return result


def summarize_uq(error: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    if error.ndim != 4 or error.shape[1:] != (120, 4, 3):
        raise ValueError(
            "UQ estimand is complete hidden-identity future per trajectory"
        )
    events = gaussian_events(error, covariance)
    per_case = {key: value.mean(axis=(1, 2)).tolist() for key, value in events.items()}
    summary = {key: float(np.mean(value)) for key, value in per_case.items()}
    horizons = {}
    for label, frames in zip(HORIZONS, np.array_split(np.arange(120), 3), strict=True):
        horizons[label] = {
            key: float(value[:, frames].mean()) for key, value in events.items()
        }
    return {"per_case": per_case, "summary": summary, "horizons": horizons}


def primary_decision(results: dict[str, Any], config: BeliefConfig) -> dict[str, Any]:
    if set(results) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("all opened objects are required")
    points, uncertainty = {}, {}
    for name in ("DLO1", "DLO3"):
        scores = results[name]["point"]
        summary = scores["summaries"]
        candidate, old, base = (
            summary[k] for k in ("weak_16", "previous_paired_8", "incumbent")
        )
        metrics = ("coordinate_l1_mm", "point_rmse_mm")
        controls = (
            "strong_16",
            "ols_physical_16",
            "ols_readout_16",
            "periodic_pose_16",
        )
        joint = sum(
            all(a[k] < b[k] for k in metrics)
            for a, b in zip(
                scores["per_case"]["weak_16"],
                scores["per_case"]["previous_paired_8"],
                strict=True,
            )
        )
        points[name] = {
            "at_least_2percent_rmse_gain_over_previous": candidate[metrics[1]]
            <= 0.98 * old[metrics[1]],
            "at_least_2percent_rmse_gain_over_matched_ols": candidate[metrics[1]]
            <= 0.98 * summary["ols_physical_16"][metrics[1]],
            "both_metrics_better_than_all_matched_controls": all(
                candidate[k] < summary[arm][k] for arm in controls for k in metrics
            ),
            "both_metrics_better_than_previous": all(
                candidate[k] < old[k] for k in metrics
            ),
            "late_rmse_nonincreasing_over_incumbent": candidate["late"][metrics[1]]
            <= base["late"][metrics[1]],
            "at_least_five_joint_wins_over_previous": joint >= 5,
            "worst_case_rmse_ratio_at_most_1_05": candidate[
                metrics[1] + "_worst_case_ratio"
            ]
            <= 1.05,
        }
        shaped = results[name]["uq"]["weak_16_shaped__moment"]
        isotropic = results[name]["uq"]["weak_16_isotropic__moment"]
        delta = np.asarray(shaped["per_case"]["nll"]) - isotropic["per_case"]["nll"]
        draws = np.random.default_rng(config.calibration_bootstrap_seed).integers(
            0, len(delta), (config.calibration_bootstrap_replicates, len(delta))
        )
        ci = np.quantile(delta[draws].mean(axis=1), (0.025, 0.975)).tolist()
        uq_checks = {
            "same_mean_nll_delta_ci95_upper_below_zero": ci[1] < 0,
            "coverage_between_80_and_98_percent": 0.8
            <= shaped["summary"]["coverage_90"]
            <= 0.98,
            "volume_not_larger_than_same_mean_isotropic": shaped["summary"][
                "ellipsoid_volume_mm3"
            ]
            <= isotropic["summary"]["ellipsoid_volume_mm3"],
        }
        uncertainty[name] = {"checks": uq_checks, "same_mean_nll_delta_ci95": ci}
    point_pass = all(all(v.values()) for v in points.values())
    uq_pass = all(all(v["checks"].values()) for v in uncertainty.values())
    return {
        "primary_arm": "weak_16",
        "point_checks": points,
        "uncertainty_checks": uncertainty,
        "point_gate_passed": point_pass,
        "uncertainty_gate_passed": uq_pass,
        "development_advancement_gate_passed": point_pass and uq_pass,
        "secondary_arms_cannot_rescue_primary": True,
        "automatic_target_authorization": False,
        "incumbent_promoted": False,
        "population_confirmation_or_sota_claim": False,
    }
