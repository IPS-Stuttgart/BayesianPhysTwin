"""Covariance-gated raw-camera belief diagnostics on the open Deform360 panel.

Measurement and uncertainty artifacts are checksum-verified before the open
target is loaded.  The gate uses only a current unordered observation set, the
current physical/persistence candidates, and the causal measurement covariance.
When it abstains, both candidate arms copy the selected raw backbone interval
bit-for-bit.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cpd_registration import NonrigidCpdConfig, fit_nonrigid_cpd
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import (
    PRIMARY_METRICS,
    UPDATE_FRAMES,
    _post_update_scored_frames,
    _physical_object_cluster_bootstrap,
    score_deform360_hidden_trajectory,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    _canonical_sha256,
    _load_measurement_artifact,
    _load_open_case_for_evaluation,
    _sha256,
    _validate_prediction_seal,
    expected_open_case_names,
)
from .deform360_raw_camera_uncertainty import (
    UNCERTAINTY_ARCHIVE_FILENAME,
    UNCERTAINTY_MANIFEST_FILENAME,
    UNCERTAINTY_PROTOCOL_ID,
)
from .deform360_recursive_cpd_diagnostic import INDEPENDENT_ARM
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)

GATED_EVALUATION_PROTOCOL_ID = (
    "deform360-open27-raw-camera-covariance-gated-rbf-cpd-v1-development"
)
CHI2_DF3_50 = 2.3659738843753377
CHI2_DF3_90 = 6.251388631170325
CHI2_DF3_95 = 7.814727903251179
CHI2_DF3_99 = 11.344866730144373
GATE_THRESHOLDS = {
    "ungated": -np.inf,
    "chi2_df3_50": CHI2_DF3_50,
    "chi2_df3_90": CHI2_DF3_90,
    "chi2_df3_95": CHI2_DF3_95,
    "chi2_df3_99": CHI2_DF3_99,
}
RBF_ARM_PREFIX = "selected_backbone_euclidean_rbf"
CPD_ARM_PREFIX = INDEPENDENT_ARM
SELECTED_BACKBONE_ARM = "selected_raw_backbone"
LEGACY_SELECTED_BACKBONE_ARM = "selected_raw_backbone_legacy_physical_default"
MINIMUM_SELECTOR_SUPPORT = 3


def _load_uncertainty_artifact(
    measurement_dir: Path,
    uncertainty_dir: Path,
    expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = uncertainty_dir / UNCERTAINTY_MANIFEST_FILENAME
    archive_path = uncertainty_dir / UNCERTAINTY_ARCHIVE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    claimed = unsigned.pop("result_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("uncertainty manifest content checksum changed")
    if manifest.get("protocol_id") != UNCERTAINTY_PROTOCOL_ID:
        raise ValueError("uncertainty protocol ID changed")
    if manifest.get("artifact_kind") != (
        "Deform360CausalRawCameraMeasurementUncertainty"
    ):
        raise ValueError("unsupported uncertainty artifact")
    for key in ("object_id", "episode_id", "episode_key"):
        if manifest.get(key) != expected_identity.get(key):
            raise ValueError(f"uncertainty {key} differs from prediction seal")
    boundary = manifest.get("information_boundary", {})
    if (
        boundary.get("target_data_read") is not False
        or boundary.get("outcome_manifest_read") is not False
    ):
        raise ValueError("uncertainty artifact crossed the target boundary")
    if manifest.get("output", {}).get("archive_sha256") != _sha256(archive_path):
        raise ValueError("uncertainty archive checksum changed")
    inputs = manifest.get("inputs", {})
    if inputs.get("measurement_manifest", {}).get("sha256") != _sha256(
        measurement_dir / MANIFEST_FILENAME
    ):
        raise ValueError("uncertainty binds a different measurement manifest")
    if inputs.get("measurement_archive", {}).get("sha256") != _sha256(
        measurement_dir / MEASUREMENT_FILENAME
    ):
        raise ValueError("uncertainty binds a different measurement archive")
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def _load_cycle_artifact(
    measurement_dir: Path,
    uncertainty_dir: Path,
    cycle_dir: Path,
    expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # Imported lazily to keep the causal builder modules acyclic.
    from .deform360_raw_camera_cycle_uncertainty import (
        CYCLE_ARCHIVE_FILENAME,
        CYCLE_MANIFEST_FILENAME,
        CYCLE_PROTOCOL_ID,
    )

    manifest_path = cycle_dir / CYCLE_MANIFEST_FILENAME
    archive_path = cycle_dir / CYCLE_ARCHIVE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    claimed = unsigned.pop("result_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("cycle manifest content checksum changed")
    if manifest.get("protocol_id") != CYCLE_PROTOCOL_ID:
        raise ValueError("cycle protocol ID changed")
    if manifest.get("artifact_kind") != "Deform360CausalRawCameraCycleUncertainty":
        raise ValueError("unsupported cycle uncertainty artifact")
    for key in ("object_id", "episode_id", "episode_key"):
        if manifest.get(key) != expected_identity.get(key):
            raise ValueError(f"cycle uncertainty {key} differs from prediction seal")
    boundary = manifest.get("information_boundary", {})
    if (
        boundary.get("target_data_read") is not False
        or boundary.get("outcome_manifest_read") is not False
    ):
        raise ValueError("cycle uncertainty crossed the target boundary")
    if manifest.get("output", {}).get("archive_sha256") != _sha256(archive_path):
        raise ValueError("cycle uncertainty archive checksum changed")
    inputs = manifest.get("inputs", {})
    if inputs.get("measurement_manifest", {}).get("sha256") != _sha256(
        measurement_dir / MANIFEST_FILENAME
    ):
        raise ValueError("cycle uncertainty binds a different measurement manifest")
    if inputs.get("uncertainty_manifest", {}).get("sha256") != _sha256(
        uncertainty_dir / UNCERTAINTY_MANIFEST_FILENAME
    ):
        raise ValueError("cycle uncertainty binds a different uncertainty manifest")
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    return manifest, arrays


def covariance_innovation_gate(
    observation_m: np.ndarray,
    backbone_m: np.ndarray,
    covariance_m2: np.ndarray,
    covariance_valid: np.ndarray,
    *,
    threshold: float = CHI2_DF3_95,
    minimum_valid_count: int = 3,
) -> dict[str, Any]:
    """Correspondence-free median Mahalanobis innovation gate."""

    observation = np.asarray(observation_m, dtype=float)
    backbone = np.asarray(backbone_m, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float)
    valid = np.asarray(covariance_valid, dtype=bool).copy()
    if observation.ndim != 2 or observation.shape[1] != 3 or not len(observation):
        raise ValueError("observation_m must have nonempty shape (K, 3)")
    if backbone.ndim != 2 or backbone.shape[1] != 3 or not len(backbone):
        raise ValueError("backbone_m must have nonempty shape (M, 3)")
    if covariance.shape != (len(observation), 3, 3):
        raise ValueError("covariance_m2 must have shape (K, 3, 3)")
    if valid.shape != (len(observation),):
        raise ValueError("covariance_valid must have shape (K,)")
    valid &= np.all(np.isfinite(observation), axis=1)
    valid &= np.all(np.isfinite(covariance), axis=(1, 2))
    distances = np.linalg.norm(observation[:, None, :] - backbone[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    residual = observation - backbone[nearest]
    squared_mahalanobis: list[float] = []
    condition_exclusion_count = 0
    for index in np.flatnonzero(valid):
        eigenvalues = np.linalg.eigvalsh(covariance[index])
        if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
            condition_exclusion_count += 1
            continue
        try:
            solved = np.linalg.solve(covariance[index], residual[index])
        except np.linalg.LinAlgError:
            condition_exclusion_count += 1
            continue
        value = float(residual[index] @ solved)
        if np.isfinite(value) and value >= 0.0:
            squared_mahalanobis.append(value)
    statistic = (
        None
        if not squared_mahalanobis
        else float(np.median(np.asarray(squared_mahalanobis)))
    )
    accepted = (
        len(squared_mahalanobis) >= minimum_valid_count
        and statistic is not None
        and statistic > threshold
    )
    return {
        "accepted": bool(accepted),
        "decision": (
            "accepted"
            if accepted
            else "insufficient_valid_covariance"
            if len(squared_mahalanobis) < minimum_valid_count
            else "covariance_gate_rejected"
        ),
        "median_squared_mahalanobis_innovation": statistic,
        "threshold": None if np.isneginf(threshold) else float(threshold),
        "valid_count": len(squared_mahalanobis),
        "condition_number_exclusion_count": condition_exclusion_count,
        "nearest_backbone_distance_median_m": float(
            np.median(np.min(distances, axis=1))
        ),
        "nearest_backbone_distance_maximum_m": float(np.max(np.min(distances, axis=1))),
    }


def _arm_name(prefix: str, threshold_name: str) -> str:
    return f"{prefix}_{threshold_name}"


def evaluate_covariance_gated_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    measurement_m: np.ndarray,
    measurement_validity: np.ndarray,
    measurement_covariance_m2: np.ndarray,
    measurement_covariance_valid: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: Sequence[int],
    rbf_config: RecursiveRbfBeliefConfig | None = None,
    cpd_config: NonrigidCpdConfig | None = None,
    gate_thresholds: Mapping[str, float] = GATE_THRESHOLDS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate selected-backbone RBF and CPD arms under fixed gates."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    prior = np.asarray(prior_input, dtype=float)
    persistence = np.asarray(persistence_input, dtype=float)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    measurement = np.asarray(measurement_m, dtype=float)
    measurement_mask = np.asarray(measurement_validity, dtype=bool)
    covariance = np.asarray(measurement_covariance_m2, dtype=float)
    covariance_mask = np.asarray(measurement_covariance_valid, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    if prior.shape != persistence.shape or prior.shape != target.shape:
        raise ValueError("prior, persistence, and target shapes differ")
    if measurement.shape != prior.shape or measurement_mask.shape != prior.shape[:2]:
        raise ValueError("measurement arrays differ from trajectory shape")
    if covariance.shape != prior.shape[:2] + (3, 3):
        raise ValueError("measurement covariance shape differs from trajectory")
    if covariance_mask.shape != prior.shape[:2]:
        raise ValueError("covariance validity shape differs from trajectory")
    if visible.shape != prior.shape[:2] or valid.shape != prior.shape[:2]:
        raise ValueError("target visibility/validity shape differs from trajectory")
    if not np.array_equal(
        prior_input[0].astype(np.float32), target[0].astype(np.float32)
    ):
        raise ValueError("frame-zero material identities differ")
    if len(centers) != len(np.unique(centers)):
        raise ValueError("center IDs must be unique")
    thresholds = {str(name): float(value) for name, value in gate_thresholds.items()}
    if "ungated" not in thresholds or not np.isneginf(thresholds["ungated"]):
        raise ValueError("gate thresholds must include ungated=-inf")
    rbf_cfg = rbf_config or RecursiveRbfBeliefConfig(local_blend=1.0)
    cpd_cfg = cpd_config or NonrigidCpdConfig()
    output_dtype = prior_input.dtype
    selected_raw = prior_input.copy()
    legacy_selected_raw = prior_input.copy()
    rbf_trajectories = {name: prior_input.copy() for name in thresholds}
    cpd_trajectories = {name: prior_input.copy() for name in thresholds}
    backbones = {"physical_prior": prior, "persistence": persistence}
    rbf_states = {
        threshold_name: {
            backbone_name: initialize_recursive_rbf_belief(
                centers,
                trajectory[0, centers],
                trajectory[0],
                config=rbf_cfg,
            )
            for backbone_name, trajectory in backbones.items()
        }
        for threshold_name in thresholds
    }
    update_records: list[dict[str, Any]] = []

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(prior)
        )
        available = (
            measurement_mask[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        observed = measurement[update, available_ids]
        selector_support_sufficient = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if selector_support_sufficient:
            chamfer = {
                name: _symmetric_set_chamfer_m(
                    trajectory[update, available_ids], observed
                )
                for name, trajectory in backbones.items()
            }
            selected_name = min(
                ("physical_prior", "persistence"),
                key=lambda name: (
                    chamfer[name],
                    0 if name == "physical_prior" else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(
                        trajectory[update, available_ids], observed
                    )
                    for name, trajectory in backbones.items()
                }
                if len(available_ids)
                else {"physical_prior": None, "persistence": None}
            )
            selected_name = "persistence"
        selected = backbones[selected_name]
        selected_raw[update + 1 : stop] = selected[update + 1 : stop]
        legacy_selected = (
            backbones["physical_prior"] if not selector_support_sufficient else selected
        )
        legacy_selected_raw[update + 1 : stop] = legacy_selected[update + 1 : stop]
        if not np.array_equal(
            selected_raw[update + 1 : stop], selected[update + 1 : stop]
        ):
            raise AssertionError("selected raw backbone fallback is not bit-exact")
        if len(available_ids):
            gate_by_name = {
                name: covariance_innovation_gate(
                    observed,
                    selected[update, available_ids],
                    covariance[update, available_ids],
                    covariance_mask[update, available_ids],
                    threshold=threshold,
                )
                for name, threshold in thresholds.items()
            }
            gate_by_name["ungated"].update(
                {
                    "accepted": selector_support_sufficient,
                    "decision": (
                        "accepted_without_covariance_gate"
                        if selector_support_sufficient
                        else "insufficient_selector_support"
                    ),
                }
            )
            if not selector_support_sufficient:
                for gate in gate_by_name.values():
                    gate.update(
                        {
                            "accepted": False,
                            "decision": "insufficient_selector_support",
                        }
                    )
        else:
            gate_by_name = {
                name: {
                    "accepted": False,
                    "decision": "insufficient_valid_covariance",
                    "median_squared_mahalanobis_innovation": None,
                    "threshold": None if np.isneginf(threshold) else threshold,
                    "valid_count": 0,
                    "condition_number_exclusion_count": 0,
                    "nearest_backbone_distance_median_m": None,
                    "nearest_backbone_distance_maximum_m": None,
                }
                for name, threshold in thresholds.items()
            }
        residual_by_backbone: dict[str, np.ndarray] = {}
        for backbone_name, trajectory in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=float)
            residual[available] = observed - trajectory[update, available_ids]
            residual_by_backbone[backbone_name] = residual
        transform = None
        fit_error = None
        if len(available_ids) >= 3:
            try:
                transform = fit_nonrigid_cpd(
                    selected[update, available_ids],
                    observed,
                    config=cpd_cfg,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                fit_error = f"{type(error).__name__}: {error}"

        threshold_records: dict[str, Any] = {}
        for name in thresholds:
            gate = gate_by_name[name]
            rbf_trajectories[name][update + 1 : stop] = selected[update + 1 : stop]
            cpd_trajectories[name][update + 1 : stop] = selected[update + 1 : stop]
            rbf_applied = False
            cpd_applied = False
            if gate["accepted"]:
                for backbone_name, trajectory in backbones.items():
                    posterior, _ = update_recursive_rbf_belief(
                        rbf_states[name][backbone_name],
                        update,
                        trajectory[update, centers],
                        residual_by_backbone[backbone_name],
                        available,
                        config=rbf_cfg,
                    )
                    rbf_states[name][backbone_name] = posterior
                selected_posterior = rbf_states[name][selected_name]
                for frame in range(update + 1, stop):
                    decoded = decode_recursive_rbf_belief(
                        selected_posterior,
                        selected[update],
                        forecast_frames=frame - update,
                        config=rbf_cfg,
                    )
                    rbf_trajectories[name][frame] = (
                        selected[frame].astype(float) + decoded.mean_m
                    ).astype(output_dtype, copy=False)
                rbf_applied = True
                if transform is not None:
                    for frame in range(update + 1, stop):
                        cpd_trajectories[name][frame] = transform.transform(
                            selected[frame]
                        ).astype(output_dtype, copy=False)
                    cpd_applied = True
            if not rbf_applied and not np.array_equal(
                rbf_trajectories[name][update + 1 : stop],
                selected[update + 1 : stop],
            ):
                raise AssertionError("RBF abstention fallback is not bit-exact")
            if not cpd_applied and not np.array_equal(
                cpd_trajectories[name][update + 1 : stop],
                selected[update + 1 : stop],
            ):
                raise AssertionError("CPD abstention fallback is not bit-exact")
            threshold_records[name] = {
                **gate,
                "selected_backbone": selected_name,
                "fallback_backbone": selected_name,
                "rbf_correction_applied": rbf_applied,
                "cpd_correction_applied": cpd_applied,
                "cpd_fit_available": transform is not None,
                "rbf_state_update_count_by_backbone": {
                    backbone_name: int(np.max(state.update_count))
                    for backbone_name, state in rbf_states[name].items()
                },
            }
        update_records.append(
            {
                "frame": int(update),
                "stop_frame_exclusive": int(stop),
                "available_center_count": int(np.sum(available)),
                "selected_backbone": selected_name,
                "selector_support_sufficient": selector_support_sufficient,
                "selector_decision": (
                    "current_observation_chamfer"
                    if selector_support_sufficient
                    else "insufficient_support_persistence_default"
                ),
                "legacy_physical_default_backbone": (
                    "physical_prior"
                    if not selector_support_sufficient
                    else selected_name
                ),
                "current_observation_chamfer_m": chamfer,
                "cpd_fit_error": fit_error,
                "cpd_effective_correspondence_count": (
                    None
                    if transform is None
                    else float(transform.effective_correspondence_count)
                ),
                "gates": threshold_records,
            }
        )

    trajectories: dict[str, np.ndarray] = {
        "physical_prior": prior_input.copy(),
        "persistence": persistence_input.copy(),
        SELECTED_BACKBONE_ARM: selected_raw,
        LEGACY_SELECTED_BACKBONE_ARM: legacy_selected_raw,
    }
    trajectories.update(
        {
            _arm_name(RBF_ARM_PREFIX, name): trajectory
            for name, trajectory in rbf_trajectories.items()
        }
    )
    trajectories.update(
        {
            _arm_name(CPD_ARM_PREFIX, name): trajectory
            for name, trajectory in cpd_trajectories.items()
        }
    )
    scores = {
        arm: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visible,
            valid,
            center_ids=centers,
            scored_frames=tuple(int(frame) for frame in scored_frames),
        )
        for arm, trajectory in trajectories.items()
    }
    report = {
        "protocol_id": GATED_EVALUATION_PROTOCOL_ID,
        "center_ids": centers.tolist(),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": [int(frame) for frame in scored_frames],
        "rbf_config": asdict(rbf_cfg),
        "cpd_config": asdict(cpd_cfg),
        "gate_thresholds": {
            name: None if np.isneginf(value) else value
            for name, value in thresholds.items()
        },
        "gate_contract": {
            "statistic": (
                "median per-observation squared Mahalanobis innovation to the "
                "nearest point in the selected current backbone set"
            ),
            "primary_threshold": CHI2_DF3_95,
            "primary_threshold_label": "chi2_df3_95",
            "acceptance_rule": "statistic strictly greater than threshold",
            "minimum_valid_covariance_count": 3,
            "abstention": (
                "bit-exact selected current-observation physical or persistence "
                "backbone for the complete interval"
            ),
            "abstention_scope": (
                "correction abstention, not observation abstention: the same "
                "current measurement still selects physical versus persistence"
            ),
            "safety_claim": False,
        },
        "observed_backbone_selector": {
            "metric": "current observed-centre symmetric set Chamfer",
            "tie_break": "physical_prior",
            "minimum_reliable_support": MINIMUM_SELECTOR_SUPPORT,
            "insufficient_support_default": "persistence",
            "insufficient_support_rule_status": (
                "frozen on the open development panel before held-target use"
            ),
            "selected_by_update": [
                record["selected_backbone"] for record in update_records
            ],
            "physical_prior_count": int(
                sum(
                    record["selected_backbone"] == "physical_prior"
                    for record in update_records
                )
            ),
            "persistence_count": int(
                sum(
                    record["selected_backbone"] == "persistence"
                    for record in update_records
                )
            ),
            "insufficient_support_count": int(
                sum(
                    not record["selector_support_sufficient"]
                    for record in update_records
                )
            ),
            "legacy_physical_default_ablation_arm": LEGACY_SELECTED_BACKBONE_ARM,
        },
        "updates": update_records,
        "scores": scores,
    }
    return report, trajectories


def _covariance_calibration(
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    measurement: np.ndarray,
    measurement_validity: np.ndarray,
    covariance: np.ndarray,
    covariance_validity: np.ndarray,
    centers: np.ndarray,
) -> dict[str, Any]:
    errors: list[float] = []
    squared_mahalanobis: list[float] = []
    predicted_rms: list[float] = []
    nll: list[float] = []
    for frame in UPDATE_FRAMES:
        supported = (
            visibility[frame, centers]
            & validity[frame, centers]
            & measurement_validity[frame, centers]
            & covariance_validity[frame, centers]
        )
        for point_id in centers[supported]:
            delta = measurement[frame, point_id] - target[frame, point_id]
            matrix = covariance[frame, point_id]
            sign, logdet = np.linalg.slogdet(matrix)
            if sign <= 0.0 or not np.isfinite(logdet):
                continue
            try:
                solved = np.linalg.solve(matrix, delta)
            except np.linalg.LinAlgError:
                continue
            d2 = float(delta @ solved)
            if not np.isfinite(d2) or d2 < 0.0:
                continue
            errors.append(float(np.linalg.norm(delta)))
            squared_mahalanobis.append(d2)
            predicted_rms.append(float(np.sqrt(np.trace(matrix) / 3.0)))
            nll.append(float(0.5 * (d2 + logdet + 3.0 * np.log(2.0 * np.pi))))
    values = np.asarray(squared_mahalanobis)
    error_values = np.asarray(errors)
    predicted_values = np.asarray(predicted_rms)
    if not len(values):
        raise ValueError("no target-visible covariance measurements for calibration")
    correlation_value = float(
        np.corrcoef(predicted_values, error_values)[0, 1]
        if np.std(predicted_values) > 0.0 and np.std(error_values) > 0.0
        else np.nan
    )
    return {
        "count": len(values),
        "coverage": {
            "chi2_df3_50": float(np.mean(values <= CHI2_DF3_50)),
            "chi2_df3_90": float(np.mean(values <= CHI2_DF3_90)),
            "chi2_df3_95": float(np.mean(values <= CHI2_DF3_95)),
            "chi2_df3_99": float(np.mean(values <= CHI2_DF3_99)),
        },
        "squared_mahalanobis_median": float(np.median(values)),
        "squared_mahalanobis_p90": float(np.quantile(values, 0.9)),
        "mean_gaussian_nll": float(np.mean(nll)),
        "measurement_error_mean_m": float(np.mean(error_values)),
        "measurement_error_median_m": float(np.median(error_values)),
        "measurement_error_p90_m": float(np.quantile(error_values, 0.9)),
        "measurement_error_maximum_m": float(np.max(error_values)),
        "predicted_rms_standard_deviation_mean_m": float(np.mean(predicted_values)),
        "predicted_rms_standard_deviation_median_m": float(np.median(predicted_values)),
        "predicted_rms_vs_error_pearson": (
            correlation_value if np.isfinite(correlation_value) else None
        ),
        "claim_boundary": (
            "post-hoc target-open calibration audit only; no covariance scaling "
            "or gate threshold was fit to these errors"
        ),
    }


def evaluate_covariance_gated_case(
    panel_case_dir: str | Path,
    measurement_dir: str | Path,
    uncertainty_dir: str | Path,
    cycle_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one prebuilt measurement/uncertainty pair against an open target."""

    case_dir = Path(panel_case_dir).resolve()
    if case_dir.name not in expected_open_case_names():
        raise ValueError("case is outside the explicit outcome-open panel")
    seal_path = case_dir / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _validate_prediction_seal(seal)
    measurement_manifest, measurement_arrays = _load_measurement_artifact(
        case_dir,
        Path(measurement_dir).resolve(),
        seal,
    )
    uncertainty_manifest, uncertainty_arrays = _load_uncertainty_artifact(
        Path(measurement_dir).resolve(),
        Path(uncertainty_dir).resolve(),
        seal,
    )
    cycle_manifest = None
    cycle_arrays = None
    if cycle_dir is not None:
        cycle_manifest, cycle_arrays = _load_cycle_artifact(
            Path(measurement_dir).resolve(),
            Path(uncertainty_dir).resolve(),
            Path(cycle_dir).resolve(),
            seal,
        )
    # Only now, after both causal artifacts and their hashes are verified, may
    # the already-open development target be loaded for scoring/calibration.
    open_seal, prior, persistence, target, visibility, validity = (
        _load_open_case_for_evaluation(case_dir)
    )
    if open_seal != seal:
        raise ValueError("prediction seal changed while opening the outcome")
    centers = np.asarray(measurement_arrays["center_ids"], dtype=np.int64)
    measurement = np.asarray(measurement_arrays["measurement_m"], dtype=float)
    measurement_validity = np.asarray(
        measurement_arrays["measurement_validity"], dtype=bool
    )
    covariance_source_arrays = (
        uncertainty_arrays if cycle_arrays is None else cycle_arrays
    )
    covariance = np.asarray(
        covariance_source_arrays["measurement_covariance_m2"], dtype=float
    )
    covariance_validity = np.asarray(
        covariance_source_arrays["measurement_covariance_valid"], dtype=bool
    )
    scored_frames = _post_update_scored_frames(len(target))
    algorithm_report, trajectories = evaluate_covariance_gated_arrays(
        prior,
        persistence,
        target,
        visibility,
        validity,
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        center_ids=centers,
        scored_frames=scored_frames,
    )
    calibration = _covariance_calibration(
        target,
        visibility,
        validity,
        measurement,
        measurement_validity,
        covariance,
        covariance_validity,
        centers,
    )
    report = {
        **algorithm_report,
        "case": case_dir.name,
        "object_id": str(seal["object_id"]),
        "episode_id": int(seal["episode_id"]),
        "measurement_manifest_sha256": _sha256(
            Path(measurement_dir) / MANIFEST_FILENAME
        ),
        "measurement_result_sha256": measurement_manifest["result_sha256"],
        "uncertainty_manifest_sha256": _sha256(
            Path(uncertainty_dir) / UNCERTAINTY_MANIFEST_FILENAME
        ),
        "uncertainty_result_sha256": uncertainty_manifest["result_sha256"],
        "covariance_source": (
            "jacobian_plus_leave_one_camera_out"
            if cycle_manifest is None
            else "forward_backward_cycle_inflated_jacobian_plus_leave_one_camera_out"
        ),
        "cycle_manifest_sha256": (
            None
            if cycle_dir is None
            else _sha256(
                Path(cycle_dir) / "measurement_cycle_uncertainty_manifest.json"
            )
        ),
        "cycle_result_sha256": (
            None if cycle_manifest is None else cycle_manifest["result_sha256"]
        ),
        "covariance_calibration": calibration,
        "information_boundary": {
            "measurement_and_uncertainty_verified_before_target_open": True,
            "measurement_builder_target_read": False,
            "uncertainty_builder_target_read": False,
            "target_role": "scoring and explicitly labeled calibration audit only",
        },
    }
    return report, trajectories


def evaluate_covariance_gated_cohort(
    panel_root: str | Path,
    measurement_root: str | Path,
    uncertainty_root: str | Path,
    output_dir: str | Path,
    *,
    cycle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the complete open-27 cohort and emit risk-coverage summaries."""

    panel = Path(panel_root).resolve()
    measurements = Path(measurement_root).resolve()
    uncertainties = Path(uncertainty_root).resolve()
    cycles = None if cycle_root is None else Path(cycle_root).resolve()
    output = Path(output_dir).resolve()
    missing = [
        case
        for case in expected_open_case_names()
        if not (measurements / case / MANIFEST_FILENAME).is_file()
        or not (uncertainties / case / UNCERTAINTY_MANIFEST_FILENAME).is_file()
        or (
            cycles is not None
            and not (
                cycles / case / "measurement_cycle_uncertainty_manifest.json"
            ).is_file()
        )
    ]
    if missing:
        raise FileNotFoundError(f"missing measurement/uncertainty artifacts: {missing}")
    output.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    for case in expected_open_case_names():
        report, trajectories = evaluate_covariance_gated_case(
            panel / case,
            measurements / case,
            uncertainties / case,
            None if cycles is None else cycles / case,
        )
        report_path = output / f"{case}.json"
        archive_path = output / f"{case}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(archive_path, **trajectories)
        groups[case] = str(report["object_id"])
        artifacts.append(
            {
                "case": case,
                "report_sha256": _sha256(report_path),
                "archive_sha256": _sha256(archive_path),
            }
        )
        reports.append(report)
    arms = tuple(reports[0]["scores"])
    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in arms
    }
    primary_rbf = _arm_name(RBF_ARM_PREFIX, "chi2_df3_95")
    primary_cpd = _arm_name(CPD_ARM_PREFIX, "chi2_df3_95")
    comparisons: dict[str, Any] = {}
    for candidate in (primary_rbf, primary_cpd):
        for baseline in ("physical_prior", "persistence", SELECTED_BACKBONE_ARM):
            for metric in PRIMARY_METRICS:
                differences = {
                    report["case"]: float(
                        report["scores"][candidate][metric]
                        - report["scores"][baseline][metric]
                    )
                    for report in reports
                }
                comparison = _physical_object_cluster_bootstrap(differences, groups)
                comparison["episode_wins"] = int(
                    np.sum(np.asarray(list(differences.values())) < 0.0)
                )
                comparison["per_object_mean_difference_m"] = {
                    object_id: float(
                        np.mean(
                            [
                                differences[case]
                                for case, group in groups.items()
                                if group == object_id
                            ]
                        )
                    )
                    for object_id in sorted(set(groups.values()))
                }
                comparison["relative_change"] = (
                    aggregate[candidate][metric] / aggregate[baseline][metric] - 1.0
                )
                comparisons[f"{candidate}:vs:{baseline}:{metric}"] = comparison
    risk_coverage: dict[str, Any] = {}
    for threshold_name in GATE_THRESHOLDS:
        accepted = [
            bool(update["gates"][threshold_name]["accepted"])
            for report in reports
            for update in report["updates"]
        ]
        rbf_applied = [
            bool(update["gates"][threshold_name]["rbf_correction_applied"])
            for report in reports
            for update in report["updates"]
        ]
        cpd_applied = [
            bool(update["gates"][threshold_name]["cpd_correction_applied"])
            for report in reports
            for update in report["updates"]
        ]
        risk_coverage[threshold_name] = {
            "threshold": (
                None
                if np.isneginf(GATE_THRESHOLDS[threshold_name])
                else GATE_THRESHOLDS[threshold_name]
            ),
            "accepted_update_count": int(np.sum(accepted)),
            "update_count": len(accepted),
            "correction_coverage": float(np.mean(accepted)),
            "rbf_applied_update_count": int(np.sum(rbf_applied)),
            "rbf_applied_coverage": float(np.mean(rbf_applied)),
            "cpd_applied_update_count": int(np.sum(cpd_applied)),
            "cpd_applied_coverage": float(np.mean(cpd_applied)),
            "rbf_scores": aggregate[_arm_name(RBF_ARM_PREFIX, threshold_name)],
            "cpd_scores": aggregate[_arm_name(CPD_ARM_PREFIX, threshold_name)],
        }
    calibration_values = [report["covariance_calibration"] for report in reports]
    calibration_count = int(sum(value["count"] for value in calibration_values))
    calibration = {
        "visible_point_update_count": calibration_count,
        "case_mean_coverage": {
            label: float(
                np.mean([value["coverage"][label] for value in calibration_values])
            )
            for label in ("chi2_df3_50", "chi2_df3_90", "chi2_df3_95", "chi2_df3_99")
        },
        "point_pooled_coverage": {
            label: float(
                sum(
                    value["count"] * value["coverage"][label]
                    for value in calibration_values
                )
                / calibration_count
            )
            for label in (
                "chi2_df3_50",
                "chi2_df3_90",
                "chi2_df3_95",
                "chi2_df3_99",
            )
        },
        "point_pooled_measurement_error_mean_m": float(
            sum(
                value["count"] * value["measurement_error_mean_m"]
                for value in calibration_values
            )
            / calibration_count
        ),
        "point_pooled_predicted_rms_standard_deviation_mean_m": float(
            sum(
                value["count"] * value["predicted_rms_standard_deviation_mean_m"]
                for value in calibration_values
            )
            / calibration_count
        ),
        "mean_case_measurement_error_m": float(
            np.mean([value["measurement_error_mean_m"] for value in calibration_values])
        ),
        "maximum_case_measurement_error_m": float(
            np.max(
                [value["measurement_error_maximum_m"] for value in calibration_values]
            )
        ),
        "mean_case_predicted_rms_standard_deviation_m": float(
            np.mean(
                [
                    value["predicted_rms_standard_deviation_mean_m"]
                    for value in calibration_values
                ]
            )
        ),
        "per_case": {
            report["case"]: report["covariance_calibration"] for report in reports
        },
        "claim_boundary": (
            "target-open calibration audit; no error-derived covariance inflation "
            "or threshold selection"
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": GATED_EVALUATION_PROTOCOL_ID,
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "covariance_source": reports[0]["covariance_source"],
        "observed_backbone_selector_counts": {
            "physical_prior": int(
                sum(
                    report["observed_backbone_selector"]["physical_prior_count"]
                    for report in reports
                )
            ),
            "persistence": int(
                sum(
                    report["observed_backbone_selector"]["persistence_count"]
                    for report in reports
                )
            ),
            "insufficient_support": int(
                sum(
                    report["observed_backbone_selector"]["insufficient_support_count"]
                    for report in reports
                )
            ),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "risk_coverage": risk_coverage,
        "covariance_calibration": calibration,
        "per_case_raw_quality": {
            report["case"]: {
                "measurement_error_mean_m": report["covariance_calibration"][
                    "measurement_error_mean_m"
                ],
                "measurement_error_p90_m": report["covariance_calibration"][
                    "measurement_error_p90_m"
                ],
                "measurement_error_maximum_m": report["covariance_calibration"][
                    "measurement_error_maximum_m"
                ],
                "predicted_rms_standard_deviation_mean_m": report[
                    "covariance_calibration"
                ]["predicted_rms_standard_deviation_mean_m"],
                "chi2_df3_95_coverage": report["covariance_calibration"]["coverage"][
                    "chi2_df3_95"
                ],
                "available_center_count_by_update": [
                    update["available_center_count"] for update in report["updates"]
                ],
                "selected_backbone_by_update": [
                    update["selected_backbone"] for update in report["updates"]
                ],
                "chi2_df3_95_gate_accepted_by_update": [
                    bool(update["gates"]["chi2_df3_95"]["accepted"])
                    for update in report["updates"]
                ],
                "primary_rbf_identity_delta_vs_selected_raw_m": float(
                    report["scores"][primary_rbf]["post_update_hidden_identity_rmse_m"]
                    - report["scores"][SELECTED_BACKBONE_ARM][
                        "post_update_hidden_identity_rmse_m"
                    ]
                ),
                "primary_cpd_identity_delta_vs_selected_raw_m": float(
                    report["scores"][primary_cpd]["post_update_hidden_identity_rmse_m"]
                    - report["scores"][SELECTED_BACKBONE_ARM][
                        "post_update_hidden_identity_rmse_m"
                    ]
                ),
            }
            for report in reports
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "outcome-open development comparison on reconstructed proxy targets; "
            "not official Deform360, PointMotionBench, or held-target SOTA evidence"
        ),
    }
    summary["result_sha256"] = _canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "CHI2_DF3_95",
    "GATED_EVALUATION_PROTOCOL_ID",
    "GATE_THRESHOLDS",
    "covariance_innovation_gate",
    "evaluate_covariance_gated_arrays",
    "evaluate_covariance_gated_case",
    "evaluate_covariance_gated_cohort",
]
