"""Matched-comparator, recording-split GP development study (issue #945)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from .gp_discrepancy_v1 import (
    Array,
    GPConfig,
    RecordingDiscrepancyGP,
    chain_modes,
    empirical_lowrank,
    frame_block_diagonal,
    gaussian_metrics,
)


@dataclass(frozen=True)
class StudyData:
    features: Array  # recordings x forecast times x causal features
    times: Array  # forecast times, in normalized horizon units
    ids: np.ndarray
    split: np.ndarray  # fit / select / calibrate / score, whole recordings
    baseline: Array  # recordings x times x vertices x 3, native predictor
    ridge: Array  # exact existing ridge comparator on the same inputs
    truth: Array
    basis: Array  # vertices x graph modes; clamped rows exactly zero
    frames: Array  # recording x 3 x 3; columns are canonical basis axes
    kind: str  # controlled-synthetic or retrospective-development

    def validate(self) -> None:
        if self.kind not in {"controlled-synthetic", "retrospective-development"}:
            raise ValueError("this study cannot authorize confirmatory or target evidence")
        if self.features.ndim != 3 or self.baseline.ndim != 4:
            raise ValueError("invalid study array dimensions")
        recordings, horizon, _ = self.features.shape
        if self.baseline.shape[:2] != (recordings, horizon) or self.baseline.shape[-1] != 3:
            raise ValueError("features and point predictions do not align")
        if self.ridge.shape != self.baseline.shape or self.truth.shape != self.baseline.shape:
            raise ValueError("point arrays do not align")
        if self.times.shape != (horizon,) or horizon < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("forecast times must be strictly increasing")
        for name in ("features", "times", "baseline", "ridge", "truth", "basis", "frames"):
            value = getattr(self, name)
            if 0 in value.shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains empty or nonfinite values")
        if self.ids.shape != (recordings,) or self.ids.dtype.kind != "U" or len(set(self.ids.tolist())) != recordings:
            raise ValueError("recording identities must be unique Unicode strings")
        if any(not name.strip() for name in self.ids.tolist()):
            raise ValueError("empty recording identity")
        if self.frames.shape != (recordings, 3, 3):
            raise ValueError("canonical frames do not align")
        if not np.allclose(np.swapaxes(self.frames, -1, -2) @ self.frames, np.eye(3), atol=1e-10) or not np.allclose(np.linalg.det(self.frames), 1.0, atol=1e-10):
            raise ValueError("canonical frames must be proper orthogonal matrices")
        required = {"fit", "select", "calibrate", "score"}
        if self.split.shape != (recordings,) or set(self.split.tolist()) != required:
            raise ValueError("four disjoint recording-level splits are required")
        if any(np.sum(self.split == role) < 2 for role in required):
            raise ValueError("each split requires at least two recordings")
        vertices = self.baseline.shape[2]
        if self.basis.ndim != 2 or self.basis.shape[0] != vertices:
            raise ValueError("graph basis and vertex count differ")
        if not np.allclose(self.basis.T @ self.basis, np.eye(self.basis.shape[1]), atol=1e-12):
            raise ValueError("graph modes must be orthonormal")
        clamped = np.all(self.basis == 0, axis=1)
        if not np.array_equal(self.baseline[:, :, clamped], self.ridge[:, :, clamped]):
            raise ValueError("existing comparator changes clamped coordinates")


def _modes(points: Array, basis: Array, frames: Array | None = None) -> Array:
    if frames is not None:
        points = points @ frames if frames.ndim == 2 else np.einsum("stvi,sij->stvj", points, frames)
    coefficients = np.einsum("...vc,vr->...rc", points, basis)
    return coefficients.reshape(*points.shape[:-2], -1)


def _points(coefficients: Array, basis: Array, frames: Array | None = None) -> Array:
    reshaped = coefficients.reshape(*coefficients.shape[:-1], basis.shape[1], 3)
    points = np.einsum("...rc,vr->...vc", reshaped, basis)
    if frames is not None:
        points = points @ frames.T if frames.ndim == 2 else np.einsum("stvj,sij->stvi", points, frames)
    return points


def _digest(array: Array) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _paired_summary(values: Array, seed: int = 945) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean_difference": float(values.mean()),
            "recording_bootstrap_95_low": float(np.quantile(draws, 0.025)),
            "recording_bootstrap_95_high": float(np.quantile(draws, 0.975))}


def run_study(data: StudyData, protocol: dict) -> tuple[dict, dict[str, Array]]:
    """Select on select, fit uncertainty on calibrate, then score once.

    All partitions are development partitions. Score outcomes cannot affect
    model selection, scale fitting, query threshold, or predictive means.
    """
    data.validate()
    fit = np.flatnonzero(data.split == "fit")
    select = np.flatnonzero(data.split == "select")
    calibrate = np.flatnonzero(data.split == "calibrate")
    score = np.flatnonzero(data.split == "score")
    selected_times = np.unique(np.linspace(0, len(data.times) - 1, protocol["train_times"], dtype=int))
    train_features = data.features[fit][:, selected_times].reshape(len(fit) * len(selected_times), -1)
    train_response = _modes((data.truth[fit] - data.ridge[fit])[:, selected_times], data.basis, data.frames[fit]).reshape(len(train_features), -1)
    train_time = np.tile(data.times[selected_times], len(fit))
    train_ids = np.repeat(data.ids[fit], len(selected_times))
    selection_records = []
    fitted = []
    for item in protocol["candidates"]:
        config = GPConfig(**item, max_rows=protocol["max_rows"], output_scale_floor=protocol["output_scale_floor_m"])
        model = RecordingDiscrepancyGP.fit(train_features, train_time, train_ids, train_response, config)
        errors = []
        for index in select:
            prediction = model.predict(data.features[index], data.times, np.repeat(data.ids[index], len(data.times)))
            candidate = data.ridge[index] + _points(prediction.mean, data.basis, data.frames[index])
            errors.append(float(np.mean(np.abs(candidate - data.truth[index]))))
        selection_records.append({"config": asdict(config), "select_coordinate_l1_m": float(np.mean(errors))})
        fitted.append(model)
    selected = min(range(len(fitted)), key=lambda i: selection_records[i]["select_coordinate_l1_m"])
    model = fitted[selected]
    ridge_select_l1 = float(np.mean(np.abs(data.ridge[select] - data.truth[select])))
    accept_mean = selection_records[selected]["select_coordinate_l1_m"] < ridge_select_l1
    means, gp_means, raw_covariances = {}, {}, {}
    for index in np.r_[calibrate, score]:
        prediction = model.predict(data.features[index], data.times, np.repeat(data.ids[index], len(data.times)))
        gp_means[int(index)] = data.ridge[index] + _points(prediction.mean, data.basis, data.frames[index])
        means[int(index)] = gp_means[int(index)] if accept_mean else data.ridge[index].copy()
        raw_covariances[int(index)] = prediction.joint_covariance()
    calibration_error = np.stack([_modes(data.truth[i] - means[int(i)], data.basis, data.frames[i]).ravel() for i in calibrate])
    scale = float(np.mean([gaussian_metrics(error, raw_covariances[int(i)])["normalized_nees"]
                           for i, error in zip(calibrate, calibration_error, strict=True)]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("calibration cannot identify a positive covariance scale")
    floor = protocol["variance_floor_m2"]
    diagonal = np.diag(np.mean(calibration_error ** 2, axis=0) + floor)
    lowrank = empirical_lowrank(calibration_error, min(protocol["lowrank_rank"], *calibration_error.shape),
                                protocol["lowrank_diagonal_fraction"], floor)
    modes_per_frame = data.basis.shape[1] * 3
    calibration_truth = _modes(data.truth[calibrate], data.basis, data.frames[calibrate])
    threshold = float(np.quantile(calibration_truth[:, -1, 0] - calibration_truth[:, 0, 0], 0.75))
    query = np.zeros(calibration_error.shape[1])
    query[0], query[-modes_per_frame] = -1.0, 1.0
    signs = np.repeat((-1.0) ** np.arange(len(data.times)), modes_per_frame)
    point_records, covariance_records = [], []
    score_means = np.stack([means[int(i)] for i in score])
    score_gp_means = np.stack([gp_means[int(i)] for i in score])
    # This digest is computed before scoring outcomes. It is a reproducibility
    # seal, not a claim that development score cases were previously untouched.
    seal = _digest(score_means)
    for position, index in enumerate(score):
        truth = data.truth[index]
        point_records.append({"recording_id": str(data.ids[index]),
                              "baseline_l1_m": float(np.mean(np.abs(data.baseline[index] - truth))),
                              "existing_ridge_l1_m": float(np.mean(np.abs(data.ridge[index] - truth))),
                              "ridge_plus_gp_l1_m": float(np.mean(np.abs(score_gp_means[position] - truth))),
                              "selected_mean_l1_m": float(np.mean(np.abs(score_means[position] - truth)))})
        mean_modes = _modes(score_means[position], data.basis, data.frames[index]).ravel()
        truth_modes = _modes(truth, data.basis, data.frames[index]).ravel()
        error = truth_modes - mean_modes
        full = raw_covariances[int(index)] * scale
        covariances = {"gp_raw": raw_covariances[int(index)], "gp_scaled": full,
                       "gp_frame_block": frame_block_diagonal(full, modes_per_frame),
                       "gp_temporal_sign_control": signs[:, None] * full * signs[None, :],
                       "calibrated_diagonal": diagonal, "empirical_lowrank": lowrank}
        event = float(query @ truth_modes > threshold)
        for name, covariance in covariances.items():
            metrics = gaussian_metrics(error, covariance)
            query_variance = float(query @ covariance @ query)
            if query_variance <= 0:
                raise ValueError("nonpositive deformation-contrast variance")
            probability = 0.5 * math.erfc((threshold - float(query @ mean_modes)) / math.sqrt(2 * query_variance))
            covariance_records.append({"recording_id": str(data.ids[index]), "method": name,
                                       **metrics, "contrast_brier": (probability - event) ** 2})
    point_keys = ("baseline_l1_m", "existing_ridge_l1_m", "ridge_plus_gp_l1_m", "selected_mean_l1_m")
    point_summary = {key: float(np.mean([item[key] for item in point_records])) for key in point_keys}
    covariance_summary = {}
    metric_keys = ("nll_per_dimension", "normalized_nees", "marginal_90_coverage", "mean_full_90_width", "contrast_brier")
    for method in sorted({item["method"] for item in covariance_records}):
        records = [item for item in covariance_records if item["method"] == method]
        covariance_summary[method] = {key: float(np.mean([item[key] for item in records])) for key in metric_keys}
    paired = {}
    gp_nll = np.array([item["nll_per_dimension"] for item in covariance_records if item["method"] == "gp_scaled"])
    for control in ("gp_frame_block", "empirical_lowrank", "calibrated_diagonal"):
        other = np.array([item["nll_per_dimension"] for item in covariance_records if item["method"] == control])
        paired[f"gp_scaled_minus_{control}"] = _paired_summary(gp_nll - other)
    result = {
        "study": "gp-discrepancy-development-v1", "evidence_kind": data.kind,
        "statistical_unit": "complete recording", "official_evaluation_opened": False,
        "physical_state_correction_claim": False, "real_improvement_established": False,
        "covariance_space": "fixed graph-mode coordinates; not full-state calibration",
        "lowrank_comparator": "empirical residual second-moment control, not the frozen Deform360 implementation",
        "splits": {role: data.ids[data.split == role].tolist() for role in ("fit", "select", "calibrate", "score")},
        "train_rows": len(train_features), "fit_time_indices": selected_times.tolist(),
        "selection": selection_records, "selected_config": asdict(model.config),
        "gp_mean_accepted_on_select": bool(accept_mean), "existing_ridge_select_l1_m": ridge_select_l1,
        "covariance_scale_from_calibration": scale,
        "contrast_threshold_from_calibration_m": threshold,
        "prediction_sha256": seal, "point_summary": point_summary,
        "covariance_summary": covariance_summary, "paired_nll": paired,
        "point_records": point_records, "covariance_records": covariance_records,
    }
    arrays = {"ids": data.ids[score], "times": data.times, "selected_predictions": score_means,
              "ridge_plus_gp_predictions": score_gp_means, "existing_ridge_predictions": data.ridge[score],
              "baseline_predictions": data.baseline[score]}
    return result, arrays


def synthetic_data(seed: int, adverse: bool = False) -> StudyData:
    """Nonlinear deterministic bias plus OU and oscillatory execution errors.

    The data generator is not the fitted Matérn model. The adverse arm shifts a
    hidden regime on score recordings, without providing its label as a feature.
    """
    rng = np.random.default_rng(seed)
    recordings, horizon, vertices, rank = 48, 18, 9, 2
    time = np.linspace(0, 1, horizon)
    split = np.repeat(["fit", "select", "calibrate", "score"], [24, 8, 8, 8])
    context = rng.uniform(-1.2, 1.2, size=recordings)
    features = np.stack([np.tile(time, (recordings, 1)), np.repeat(context[:, None], horizon, axis=1)], axis=-1)
    basis = chain_modes(vertices, rank)
    modes = rank * 3
    baseline = np.zeros((recordings, horizon, vertices, 3))
    baseline[:, :, :, 0] = np.linspace(0, 0.4, vertices)[None, None, :]
    baseline[:, :, :, 1] = context[:, None, None] * time[None, :, None] * 0.02
    deterministic = np.sin(2.5 * context[:, None]) * (0.004 + 0.009 * time[None, :])
    deterministic = deterministic[:, :, None] * np.linspace(0.7, 1.3, modes)[None, None, :]
    ou = np.exp(-np.abs(time[:, None] - time[None, :]) / 0.22)
    root = np.linalg.cholesky(ou + np.eye(horizon) * 0.15)
    execution = np.stack([rng.standard_normal((recordings, horizon)) @ root.T for _ in range(modes)], axis=-1) * 0.0015
    execution += rng.normal(0, 0.0007, (recordings, 1, modes)) * np.sin(2 * np.pi * time)[None, :, None]
    residual = deterministic + execution
    if adverse:
        residual[split == "score"] += 0.009 * np.sign(context[split == "score", None, None]) * (time[None, :, None] > 0.45)
    truth = baseline + _points(residual, basis)
    # A controlled ridge comparator only; the native-DEFORM adapter below uses
    # the repository's actual per-node ridge implementation instead.
    design = np.concatenate([np.ones((recordings, horizon, 1)), features,
                             (features[:, :, :1] * features[:, :, 1:])], axis=-1)
    fit = split == "fit"
    train_x = design[fit].reshape(-1, design.shape[-1])
    train_y = residual[fit].reshape(-1, modes)
    penalty = np.eye(train_x.shape[1]) * 0.01
    penalty[0, 0] = 0
    coefficient = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)
    ridge = baseline + _points(design @ coefficient, basis)
    return StudyData(features, time, np.array([f"synthetic-{seed}-{i:02d}" for i in range(recordings)]),
                     split, baseline, ridge, truth, basis, np.tile(np.eye(3), (recordings, 1, 1)), "controlled-synthetic")
