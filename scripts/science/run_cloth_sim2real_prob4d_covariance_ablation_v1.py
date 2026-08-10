#!/usr/bin/env python3
"""Retrospective real-cloth attribution of Prob4D covariance treatments.

The exact frozen Cloth Sim2Real split and physical baselines are reused. Each
case first writes five prediction seals from the real prefix. Only then are the
future point clouds opened for scoring. Because repeat-2 outcomes were opened
in the original campaign, this is real-physical retrospective attribution, not
fresh confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

SCHEMA: Final = "bayesian_phystwin.cloth_sim2real_covariance_attribution"
ABLATION_SCHEMA: Final = "bayesian_phystwin.prob4d_covariance_ablation"
ABLATION_ID: Final = "cloth-sim2real-retrospective-prob4d-covariance-v1"
LEGACY_REVISION: Final = "94fe5d0012e66b93d73bf9474df2b28f2a83d153"
BENCHMARK_REVISION: Final = "178a9b9722191c51cf0dcbc3cf0dc03701b09eb3"
DATASET_SHA256: Final = (
    "268d07d90da770278106028b3c704340bbac48dbd03bb4afd563630fb6de7ec"
)
NORMAL_90: Final = 1.6448536269514722
TREATMENTS: Final = (
    "full_joint",
    "block_diagonal",
    "independent_rows",
    "shared_uncertainty_removed",
    "shared_uncertainty_underreported",
)
METHODS: Final = {
    name: "cloth-prob4d-" + name.replace("_", "-") for name in TREATMENTS
}
PARAMETERS: Final = {
    "full_joint": (1.0, True, "persistent"),
    "block_diagonal": (1.0, False, "frame_blocks"),
    "independent_rows": (0.0, False, "independent_marginals"),
    "shared_uncertainty_removed": (0.0, False, "removed"),
    "shared_uncertainty_underreported": (0.5, True, "persistent_half"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _json_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe(case_id: str) -> str:
    return case_id.replace("/", "_")


def _legacy() -> dict[str, Any]:
    from bayesian_phystwin.cloth_sim2real_belief import (
        ClothReadoutBeliefConfig,
        associate_dense_cloud,
        directed_l1_chamfer_m,
        load_binary_little_endian_ply_xyz,
        mesh_edges_from_faces,
        sample_physical_rollout,
        symmetric_l1_chamfer_m,
        symmetric_l2_hausdorff_m,
    )
    from bayesian_phystwin.phystwin_graph_discrepancy import (
        normalized_spring_laplacian,
    )
    from bayesian_phystwin.pseudo_measurements import PseudoMeasurementBatch
    from bayesian_phystwin.robust_likelihood import (
        RobustLikelihoodConfig,
        robust_mixture_likelihood,
    )

    return {
        "ClothReadoutBeliefConfig": ClothReadoutBeliefConfig,
        "associate_dense_cloud": associate_dense_cloud,
        "directed_l1_chamfer_m": directed_l1_chamfer_m,
        "load_binary_little_endian_ply_xyz": load_binary_little_endian_ply_xyz,
        "mesh_edges_from_faces": mesh_edges_from_faces,
        "sample_physical_rollout": sample_physical_rollout,
        "symmetric_l1_chamfer_m": symmetric_l1_chamfer_m,
        "symmetric_l2_hausdorff_m": symmetric_l2_hausdorff_m,
        "normalized_spring_laplacian": normalized_spring_laplacian,
        "PseudoMeasurementBatch": PseudoMeasurementBatch,
        "RobustLikelihoodConfig": RobustLikelihoodConfig,
        "robust_mixture_likelihood": robust_mixture_likelihood,
    }


@dataclass(frozen=True)
class CovariancePolicy:
    treatment: str
    shared_uncertainty_scale: float
    gauge_factors_enabled: bool
    construction: str

    @classmethod
    def from_treatment(cls, treatment: str) -> CovariancePolicy:
        _require(treatment in PARAMETERS, f"unknown treatment {treatment}")
        scale, gauge, construction = PARAMETERS[treatment]
        return cls(treatment, float(scale), bool(gauge), str(construction))


@dataclass(frozen=True)
class FitStatistics:
    residual_mean_rows_m: np.ndarray
    row_precision_per_m2: np.ndarray
    vertex_mean_m: np.ndarray
    local_mean_variance_m2: np.ndarray
    robust_probability: np.ndarray
    assignment_entropy: np.ndarray


def build_fit_statistics(
    physical: np.ndarray,
    clouds: Sequence[np.ndarray],
    *,
    config: Any,
    api: Mapping[str, Any],
) -> FitStatistics:
    """Reproduce the frozen robust vertex mean before shared covariance."""

    residuals = []
    variances = []
    reliabilities = []
    entropies = []
    for state, cloud in zip(physical, clouds, strict=True):
        association = api["associate_dense_cloud"](
            state,
            cloud,
            candidate_count=config.candidate_count,
            sensor_std_m=config.sensor_std_m,
        )
        residuals.append(association.observed_points_m - state)
        variances.append(association.variance_m2)
        reliabilities.append(association.prior_reliability)
        entropies.append(association.assignment_entropy)
    residual = np.stack(residuals)
    variance = np.stack(variances)
    reliability = np.stack(reliabilities)
    center = np.median(residual, axis=0)
    batch = api["PseudoMeasurementBatch"](
        observed=residual.reshape(-1, 3),
        predicted=np.broadcast_to(center, residual.shape).reshape(-1, 3),
        variance=variance.reshape(-1),
    )
    robust = api["robust_mixture_likelihood"](
        batch,
        prior_reliability=reliability.reshape(-1),
        config=api["RobustLikelihoodConfig"](
            outlier_variance_multiplier=config.outlier_variance_multiplier,
            model_discrepancy_variance=config.model_discrepancy_std_m**2,
        ),
    ).posterior_inlier_probability.reshape(residual.shape[:2])
    weight = robust * reliability / variance
    weight_sum = np.sum(weight, axis=0)
    mean = np.sum(weight[:, :, None] * residual, axis=0)
    mean /= np.maximum(weight_sum[:, None], 1e-15)
    raw_effective = np.square(weight_sum) / np.maximum(
        np.sum(np.square(weight), axis=0),
        1e-15,
    )
    effective = np.minimum(raw_effective, config.effective_fit_frames)
    scatter = np.sum(
        weight * np.mean(np.square(residual - mean[None]), axis=2),
        axis=0,
    ) / np.maximum(weight_sum, 1e-15)
    assignment = np.sum(weight * variance, axis=0)
    assignment /= np.maximum(weight_sum, 1e-15)
    local_variance = (scatter + assignment) / np.maximum(effective, 1.0)
    row_precision = weight / np.maximum(weight_sum[None], 1e-15)
    row_precision /= local_variance[None]
    reconstructed = np.sum(row_precision[:, :, None] * residual, axis=0)
    reconstructed /= np.sum(row_precision, axis=0)[:, None]
    _require(np.allclose(mean, reconstructed), "row normalization changed means")
    return FitStatistics(
        residual,
        row_precision,
        mean,
        local_variance,
        robust,
        np.stack(entropies),
    )


def covariance_normal_equations(
    statistics: FitStatistics,
    policy: CovariancePolicy,
    *,
    shared_bias_variance_m2: float,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Compute X'R^-1X and X'R^-1y without materializing row covariance."""

    residual = statistics.residual_mean_rows_m
    precision = statistics.row_precision_per_m2
    frame_count, node_count = precision.shape
    information = np.zeros((node_count, node_count))
    rhs = np.zeros((node_count, 3))
    diagonal = np.diag_indices(node_count)
    shared = shared_bias_variance_m2 * policy.shared_uncertainty_scale
    if policy.construction == "removed":
        weights = np.sum(precision, axis=0)
        information[diagonal] = weights
        rhs = np.sum(precision[:, :, None] * residual, axis=0)
    elif policy.construction == "independent_marginals":
        weights = 1.0 / (1.0 / precision + shared_bias_variance_m2)
        information[diagonal] = np.sum(weights, axis=0)
        rhs = np.sum(weights[:, :, None] * residual, axis=0)
    elif policy.construction == "frame_blocks":
        for frame in range(frame_count):
            weights = precision[frame]
            denominator = 1.0 / shared + float(np.sum(weights))
            information[diagonal] += weights
            information -= np.outer(weights, weights) / denominator
            weighted = weights[:, None] * residual[frame]
            rhs += weighted
            rhs -= weights[:, None] * np.sum(weighted, axis=0) / denominator
    elif policy.construction in {"persistent", "persistent_half"}:
        weights = np.sum(precision, axis=0)
        denominator = 1.0 / shared + float(np.sum(weights))
        information[diagonal] = weights
        information -= np.outer(weights, weights) / denominator
        weighted = np.sum(precision[:, :, None] * residual, axis=0)
        rhs = weighted - weights[:, None] * np.sum(weighted, axis=0) / denominator
    else:
        raise AssertionError(policy.construction)
    information = 0.5 * (information + information.T)
    minimum = float(np.min(np.linalg.eigvalsh(information)))
    tolerance = 1e-8 * max(1.0, float(np.max(np.diag(information))))
    _require(minimum >= -tolerance, "covariance information is indefinite")
    ones = np.ones(node_count)
    return information, rhs, {
        "minimum_information_eigenvalue": minimum,
        "trace_information": float(np.trace(information)),
        "constant_mode_information": float(ones @ information @ ones),
    }


def _posterior_solution(
    information: np.ndarray,
    rhs: np.ndarray,
    laplacian_square: np.ndarray,
    *,
    reference_variance_m2: float,
    prior_strength: float,
    covariance: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, Mapping[str, object]]:
    from scipy.linalg import cho_factor, cho_solve

    size = len(information)
    precision = (
        reference_variance_m2 * information
        + 2.0 * prior_strength * laplacian_square
        + 1e-8 * np.eye(size)
    )
    factor = cho_factor(0.5 * (precision + precision.T), check_finite=False)
    mean = cho_solve(factor, reference_variance_m2 * rhs, check_finite=False)
    variance = None
    if covariance:
        inverse = cho_solve(factor, np.eye(size), check_finite=False)
        variance = np.maximum(reference_variance_m2 * np.diag(inverse), 0.0)
    diagonal = np.abs(np.diag(factor[0]))
    estimate = float(np.square(np.max(diagonal) / np.min(diagonal)))
    return mean, variance, {"posterior_precision_condition_estimate": estimate}


def _cap(value: np.ndarray, maximum: float) -> np.ndarray:
    result = np.asarray(value).copy()
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    result *= np.minimum(1.0, maximum / np.maximum(norm, 1e-15))
    return result


def fit_variant(
    statistics: FitStatistics,
    physical_validation: np.ndarray,
    clouds: Sequence[np.ndarray],
    faces: np.ndarray,
    *,
    policy: CovariancePolicy,
    config: Any,
    api: Mapping[str, Any],
) -> Mapping[str, Any]:
    information, rhs, diagnostics = covariance_normal_equations(
        statistics,
        policy,
        shared_bias_variance_m2=config.shared_bias_std_m**2,
    )
    edges = api["mesh_edges_from_faces"](faces, len(information))
    laplacian = api["normalized_spring_laplacian"](len(information), edges)
    laplacian_square = np.asarray((laplacian.T @ laplacian).toarray())
    reference = float(
        np.median(
            statistics.local_mean_variance_m2 + config.shared_bias_std_m**2
        )
    )
    ones = np.ones(len(information))
    translation = (ones @ rhs) / max(float(ones @ information @ ones), 1e-15)
    candidates = {
        "baseline": np.zeros_like(statistics.vertex_mean_m),
        "global": _cap(
            np.broadcast_to(translation, statistics.vertex_mean_m.shape),
            config.maximum_correction_m,
        ),
    }
    for strength in config.graph_prior_strengths:
        mean, _, _ = _posterior_solution(
            information,
            rhs,
            laplacian_square,
            reference_variance_m2=reference,
            prior_strength=float(strength),
        )
        for scale in config.correction_scales:
            candidates[f"graph_l{strength:g}_s{scale:g}"] = _cap(
                float(scale) * mean,
                config.maximum_correction_m,
            )
    metric = api["symmetric_l1_chamfer_m"]
    baseline_scores = np.asarray(
        [
            metric(state, cloud)
            for state, cloud in zip(
                physical_validation,
                clouds,
                strict=True,
            )
        ]
    )
    rows = []
    for name, correction in candidates.items():
        scores = np.asarray(
            [
                metric(state + correction, cloud)
                for state, cloud in zip(
                    physical_validation,
                    clouds,
                    strict=True,
                )
            ]
        )
        rows.append(
            {
                "name": name,
                "mean": float(np.mean(scores)),
                "maximum": float(np.max(scores)),
                "win_fraction": float(np.mean(scores < baseline_scores)),
            }
        )
    rows.sort(key=lambda row: (row["mean"], row["name"]))
    best = rows[0]
    baseline_mean = float(np.mean(baseline_scores))
    improvement = (baseline_mean - best["mean"]) / max(baseline_mean, 1e-15)
    worst_ratio = best["maximum"] / max(float(np.max(baseline_scores)), 1e-15)
    accepted = bool(
        best["name"] != "baseline"
        and improvement >= config.minimum_validation_improvement
        and best["win_fraction"] >= config.minimum_validation_win_fraction
        and worst_ratio <= config.maximum_validation_worst_ratio
    )
    selected = str(best["name"])
    correction = candidates[selected] if accepted else candidates["baseline"]
    if selected.startswith("graph_l"):
        parts = selected.split("_")
        strength = float(parts[1][1:])
        scale = float(parts[2][1:])
        _, variance, solve = _posterior_solution(
            information,
            rhs,
            laplacian_square,
            reference_variance_m2=reference,
            prior_strength=strength,
            covariance=True,
        )
        _require(variance is not None, "graph covariance was not returned")
        variance = scale**2 * variance
        diagnostics = {**diagnostics, **solve}
    elif selected == "global":
        variance = np.full(
            len(information),
            1.0 / max(float(ones @ information @ ones), 1e-15),
        )
    else:
        variance = np.full(len(information), reference)
    return {
        "selected_name": selected if accepted else "baseline",
        "selected_before_gate": selected,
        "accepted": accepted,
        "reason": "accepted" if accepted else "prefix-validation-gate-failed",
        "correction_m": correction,
        "variance_m2": np.repeat(np.maximum(variance, 1e-15)[:, None], 3, axis=1),
        "risk_score": float(best["mean"] / max(baseline_mean, 1e-15)),
        "validation_improvement": float(improvement),
        "validation_win_fraction": float(best["win_fraction"]),
        "validation_worst_ratio": float(worst_ratio),
        "scores": rows,
        "diagnostics": diagnostics,
    }


def _cloud_dir(root: Path, case_id: str) -> Path:
    if (root / "Benchmarking_cloth").is_dir():
        root = root / "Benchmarking_cloth"
    path = root / case_id / "cloud"
    _require(path.is_dir(), f"missing cloud directory {path}")
    return path


def _clouds(path: Path, start: int, stop: int, loader: Any) -> list[np.ndarray]:
    return [loader(path / f"{index:05d}.ply") for index in range(start, stop)]


def _legacy_case(root: Path, split: str, case_id: str) -> tuple[Path, Path]:
    case_root = root / f"{split}_transfer" / _safe(case_id)
    seal = case_root / "prediction_seal.json"
    result = case_root / "result.json"
    _require(seal.is_file() and result.is_file(), "legacy case artifacts missing")
    return seal, result


def _seal_case(
    case: Mapping[str, Any],
    *,
    dataset_root: Path,
    baseline_root: Path,
    legacy_root: Path,
    output_root: Path,
    config: Any,
    api: Mapping[str, Any],
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, Any]], list[np.ndarray]]:
    case_id = str(case["case_id"])
    split = str(case["split"])
    baseline_path = baseline_root / f"{_safe(case_id)}.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        sampled, sampled_indices = api["sample_physical_rollout"](
            baseline["vertices_m"],
            int(case["frame_count"]),
        )
        faces = np.asarray(baseline["faces"], dtype=np.int64)
    fit_stop = int(case["fit_stop_frame"])
    branch = int(case["branch_frame"])
    path = _cloud_dir(dataset_root, case_id)
    prefix = _clouds(
        path,
        0,
        branch + 1,
        api["load_binary_little_endian_ply_xyz"],
    )
    statistics = build_fit_statistics(
        sampled[:fit_stop],
        prefix[:fit_stop],
        config=config,
        api=api,
    )
    physical_future = sampled[branch + 1 :]
    legacy_seal_path, legacy_result_path = _legacy_case(
        legacy_root,
        split,
        case_id,
    )
    legacy_seal = _read_json(legacy_seal_path)
    case_root = output_root / split / _safe(case_id)
    case_root.mkdir(parents=True)
    physical_path = case_root / "physical_future.npy"
    np.save(physical_path, physical_future, allow_pickle=False)
    _require(
        _file_hash(physical_path) == legacy_seal["physical_future_sha256"],
        "regenerated physical future differs from the frozen campaign",
    )
    common = {
        "case_id": case_id,
        "split": split,
        "fit_frames": [0, fit_stop - 1],
        "validation_frames": [fit_stop, branch],
        "future_frames": [branch + 1, int(case["frame_count"]) - 1],
        "sampled_physical_sha256": _array_hash(sampled),
        "sampled_indices_sha256": _array_hash(sampled_indices),
        "faces_sha256": _array_hash(faces),
        "residual_mean_rows_sha256": _array_hash(
            statistics.residual_mean_rows_m
        ),
        "vertex_mean_sha256": _array_hash(statistics.vertex_mean_m),
        "row_precision_sha256": _array_hash(statistics.row_precision_per_m2),
        "legacy_seal_sha256": _file_hash(legacy_seal_path),
        "legacy_result_sha256": _file_hash(legacy_result_path),
    }
    predictions = {}
    for treatment in TREATMENTS:
        policy = CovariancePolicy.from_treatment(treatment)
        fit = fit_variant(
            statistics,
            sampled[fit_stop : branch + 1],
            prefix[fit_stop : branch + 1],
            faces,
            policy=policy,
            config=config,
            api=api,
        )
        candidate = (
            physical_future + fit["correction_m"][None]
            if fit["accepted"]
            else physical_future
        )
        _require(
            fit["accepted"] or np.array_equal(candidate, physical_future),
            "fallback is not exact",
        )
        treatment_root = case_root / treatment
        treatment_root.mkdir()
        candidate_path = treatment_root / "candidate_future.npy"
        variance_path = treatment_root / "variance_m2.npy"
        np.save(candidate_path, candidate, allow_pickle=False)
        np.save(variance_path, fit["variance_m2"], allow_pickle=False)
        covariance = {
            "policy": asdict(policy),
            "row_precision_sha256": common["row_precision_sha256"],
            "diagnostics": fit["diagnostics"],
        }
        seal = {
            "schema": SCHEMA,
            "schema_version": 1,
            "artifact_kind": "ClothCovariancePredictionSeal",
            "ablation_id": ABLATION_ID,
            "case_id": case_id,
            "split": split,
            "method": METHODS[treatment],
            "treatment": treatment,
            "policy": asdict(policy),
            "selected_name": fit["selected_name"],
            "selected_before_gate": fit["selected_before_gate"],
            "accepted": fit["accepted"],
            "reason": fit["reason"],
            "risk_score": fit["risk_score"],
            "validation_improvement": fit["validation_improvement"],
            "validation_win_fraction": fit["validation_win_fraction"],
            "validation_worst_ratio": fit["validation_worst_ratio"],
            "candidate_future_sha256": _file_hash(candidate_path),
            "variance_sha256": _file_hash(variance_path),
            "covariance_artifact_sha256": _json_hash(covariance),
            "common_inputs": common,
            "future_outcomes_read": False,
            "retrospective_rerun": True,
        }
        seal_path = treatment_root / "prediction_seal.json"
        _write_once(seal_path, seal)
        predictions[treatment] = {
            "fit": fit,
            "candidate_path": candidate_path,
            "variance_path": variance_path,
            "seal_path": seal_path,
        }
    future = _clouds(
        path,
        branch + 1,
        int(case["frame_count"]),
        api["load_binary_little_endian_ply_xyz"],
    )
    return common, predictions, future


def _score_case(
    case: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    future: Sequence[np.ndarray],
    *,
    output_root: Path,
    legacy_root: Path,
    multiplier_by_treatment: Mapping[str, float],
    config: Any,
    api: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    case_id = str(case["case_id"])
    split = str(case["split"])
    case_root = output_root / split / _safe(case_id)
    physical = np.load(case_root / "physical_future.npy", allow_pickle=False)
    legacy_result = _read_json(_legacy_case(legacy_root, split, case_id)[1])
    scored = {}
    for treatment, prediction in predictions.items():
        seal = _read_json(prediction["seal_path"])
        _require(seal["future_outcomes_read"] is False, "invalid prediction seal")
        candidate = np.load(prediction["candidate_path"], allow_pickle=False)
        variance = np.load(prediction["variance_path"], allow_pickle=False)
        physical_symmetric = []
        candidate_symmetric = []
        physical_directed = []
        candidate_directed = []
        physical_hausdorff = []
        candidate_hausdorff = []
        raw_coverage = []
        reported_coverage = []
        interval_width = []
        standardized = []
        multiplier = float(multiplier_by_treatment[treatment])
        for horizon, (base, estimate, observation) in enumerate(
            zip(physical, candidate, future, strict=True),
            start=1,
        ):
            physical_symmetric.append(
                api["symmetric_l1_chamfer_m"](base, observation)
            )
            candidate_symmetric.append(
                api["symmetric_l1_chamfer_m"](estimate, observation)
            )
            physical_directed.append(
                api["directed_l1_chamfer_m"](base, observation)
            )
            candidate_directed.append(
                api["directed_l1_chamfer_m"](estimate, observation)
            )
            physical_hausdorff.append(
                api["symmetric_l2_hausdorff_m"](base, observation)
            )
            candidate_hausdorff.append(
                api["symmetric_l2_hausdorff_m"](estimate, observation)
            )
            association = api["associate_dense_cloud"](
                estimate,
                observation,
                candidate_count=config.candidate_count,
                sensor_std_m=config.sensor_std_m,
            )
            raw_variance = (
                variance
                + horizon * config.forecast_process_std_m_per_sqrt_frame**2
                + association.variance_m2[:, None]
            )
            residual = association.observed_points_m - estimate
            standardized.append(np.abs(residual) / np.sqrt(raw_variance))
            raw_half = NORMAL_90 * np.sqrt(raw_variance)
            reported_half = multiplier * raw_half
            raw_coverage.append(float(np.mean(np.abs(residual) <= raw_half)))
            reported_coverage.append(
                float(np.mean(np.abs(residual) <= reported_half))
            )
            interval_width.append(float(np.mean(2.0 * reported_half)))
        physical_symmetric_array = np.asarray(physical_symmetric)
        candidate_symmetric_array = np.asarray(candidate_symmetric)
        physical_directed_array = np.asarray(physical_directed)
        candidate_directed_array = np.asarray(candidate_directed)
        physical_hausdorff_array = np.asarray(physical_hausdorff)
        candidate_hausdorff_array = np.asarray(candidate_hausdorff)

        def improvement(base: np.ndarray, estimate: np.ndarray) -> float:
            return float((np.mean(base) - np.mean(estimate)) / np.mean(base))

        metrics = {
            "physical_symmetric_l1_chamfer_m": float(
                np.mean(physical_symmetric_array)
            ),
            "candidate_symmetric_l1_chamfer_m": float(
                np.mean(candidate_symmetric_array)
            ),
            "symmetric_relative_improvement": improvement(
                physical_symmetric_array,
                candidate_symmetric_array,
            ),
            "physical_directed_l1_chamfer_m": float(
                np.mean(physical_directed_array)
            ),
            "candidate_directed_l1_chamfer_m": float(
                np.mean(candidate_directed_array)
            ),
            "directed_relative_improvement": improvement(
                physical_directed_array,
                candidate_directed_array,
            ),
            "physical_symmetric_l2_hausdorff_m": float(
                np.mean(physical_hausdorff_array)
            ),
            "candidate_symmetric_l2_hausdorff_m": float(
                np.mean(candidate_hausdorff_array)
            ),
            "hausdorff_relative_improvement": improvement(
                physical_hausdorff_array,
                candidate_hausdorff_array,
            ),
            "raw_90_coordinate_coverage": float(np.mean(raw_coverage)),
            "reported_90_coordinate_coverage": float(
                np.mean(reported_coverage)
            ),
            "mean_90_interval_width_m": float(np.mean(interval_width)),
            "trial_coordinate_abs_standardized_q90": float(
                np.quantile(
                    np.concatenate([value.ravel() for value in standardized]),
                    0.90,
                )
            ),
        }
        for key in (
            "physical_symmetric_l1_chamfer_m",
            "physical_directed_l1_chamfer_m",
            "physical_symmetric_l2_hausdorff_m",
        ):
            _require(
                np.isclose(metrics[key], legacy_result["metrics"][key]),
                f"physical reproduction changed for {case_id}/{key}",
            )
        result = {
            "schema": SCHEMA,
            "schema_version": 1,
            "artifact_kind": "ClothCovariancePredictionResult",
            "ablation_id": ABLATION_ID,
            "case_id": case_id,
            "split": split,
            "method": METHODS[treatment],
            "treatment": treatment,
            "accepted": bool(seal["accepted"]),
            "selected_name": seal["selected_name"],
            "risk_score": float(seal["risk_score"]),
            "calibration_std_multiplier": multiplier,
            "prediction_seal_sha256": _file_hash(prediction["seal_path"]),
            "legacy_result_sha256": _file_hash(
                _legacy_case(legacy_root, split, case_id)[1]
            ),
            "metrics": metrics,
            "future_outcomes_opened_after_prediction_seal": True,
            "retrospective_rerun": True,
        }
        _write_once(
            prediction["seal_path"].parent / "result.json",
            result,
        )
        scored[treatment] = result
    return scored


def _task_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, object]:
    _require(len(rows) == 3, "task summary requires three trials")
    improvements = [row["metrics"]["symmetric_relative_improvement"] for row in rows]

    def mean(name: str) -> float:
        return float(np.mean([row["metrics"][name] for row in rows]))

    return {
        "accepted_case_count": int(np.sum([row["accepted"] for row in rows])),
        "symmetric_win_count": int(np.sum(np.asarray(improvements) > 0.0)),
        "physical_symmetric_l1_chamfer_m": mean(
            "physical_symmetric_l1_chamfer_m"
        ),
        "candidate_symmetric_l1_chamfer_m": mean(
            "candidate_symmetric_l1_chamfer_m"
        ),
        "object_balanced_symmetric_relative_improvement": float(
            np.mean(improvements)
        ),
        "object_balanced_directed_relative_improvement": mean(
            "directed_relative_improvement"
        ),
        "object_balanced_hausdorff_relative_improvement": mean(
            "hausdorff_relative_improvement"
        ),
        "raw_90_coordinate_coverage": mean("raw_90_coordinate_coverage"),
        "reported_90_coordinate_coverage": mean(
            "reported_90_coordinate_coverage"
        ),
        "mean_90_interval_width_m": mean("mean_90_interval_width_m"),
    }


def _invariants(
    common: Mapping[str, Mapping[str, object]],
    target_ids: Sequence[str],
    manifest: Mapping[str, Any],
    config: Any,
    script: Path,
) -> Mapping[str, str]:
    means = {
        case_id: {
            "rows": row["residual_mean_rows_sha256"],
            "vertices": row["vertex_mean_sha256"],
        }
        for case_id, row in sorted(common.items())
    }
    physical = {
        case_id: {
            key: row[key]
            for key in (
                "sampled_physical_sha256",
                "sampled_indices_sha256",
                "faces_sha256",
                "fit_frames",
                "validation_frames",
                "future_frames",
            )
        }
        for case_id, row in sorted(common.items())
    }
    risk = {
        "graph_prior_strengths": list(config.graph_prior_strengths),
        "correction_scales": list(config.correction_scales),
        "minimum_validation_improvement": config.minimum_validation_improvement,
        "minimum_validation_win_fraction": config.minimum_validation_win_fraction,
        "maximum_validation_worst_ratio": config.maximum_validation_worst_ratio,
    }
    partition = {
        "manifest_sha256": _json_hash(manifest),
        "calibration_repeat": 1,
        "target_repeat": 2,
        "target_case_ids": list(sorted(target_ids)),
        "calibration_rule": "maximum-six-trial-q90-normalized-residual",
    }
    software = {
        "legacy_revision": LEGACY_REVISION,
        "benchmark_revision": BENCHMARK_REVISION,
        "script_sha256": _file_hash(script),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
    }
    return {
        "observation_mean_sha256": _json_hash(means),
        "scored_units_sha256": _json_hash(list(sorted(target_ids))),
        "physical_linearization_sha256": _json_hash(physical),
        "fallback_policy_sha256": _json_hash("exact-physical-fallback"),
        "risk_policy_sha256": _json_hash(risk),
        "calibration_partition_sha256": _json_hash(partition),
        "software_stack_sha256": _json_hash(software),
    }


def _evidence_records(
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[Mapping[str, object]]:
    metrics = (
        (
            "future_symmetric_l1_chamfer_m",
            "candidate_symmetric_l1_chamfer_m",
            "physical_symmetric_l1_chamfer_m",
        ),
        (
            "future_directed_l1_chamfer_m",
            "candidate_directed_l1_chamfer_m",
            "physical_directed_l1_chamfer_m",
        ),
        (
            "future_symmetric_l2_hausdorff_m",
            "candidate_symmetric_l2_hausdorff_m",
            "physical_symmetric_l2_hausdorff_m",
        ),
    )
    records = []
    for case_id, variants in sorted(results.items()):
        task = case_id.split("/", 1)[1]
        for treatment in TREATMENTS:
            row = variants[treatment]
            for metric, loss_key, fallback_key in metrics:
                loss = float(row["metrics"][loss_key])
                fallback = float(row["metrics"][fallback_key])
                accepted = bool(row["accepted"])
                records.append(
                    {
                        "unit_id": case_id,
                        "group_id": case_id,
                        "metric": metric,
                        "method": METHODS[treatment],
                        "loss": loss,
                        "fallback_loss": fallback,
                        "risk_score": float(row["risk_score"]),
                        "accepted": accepted,
                        "deployed_loss": loss if accepted else fallback,
                        "horizon": task,
                        "reliability": max(
                            0.0,
                            min(1.0, 1.0 - float(row["risk_score"])),
                        ),
                    }
                )
    return records


def run(args: argparse.Namespace) -> int:
    api = _legacy()
    manifest = _read_json(args.manifest.resolve())
    _require(
        manifest.get("artifact_kind") == "ClothSim2RealDatasetManifest",
        "dataset manifest identity changed",
    )
    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = api["ClothReadoutBeliefConfig"]()
    cases = [
        case
        for case in manifest["cases"]
        if case["split"] in {"calibration", "target"}
    ]
    _require(len(cases) == 12, "expected twelve calibration/target cases")
    common = {}
    sealed = {"calibration": {}, "target": {}}
    future = {"calibration": {}, "target": {}}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        split = str(case["split"])
        case_id = str(case["case_id"])
        case_common, predictions, observations = _seal_case(
            case,
            dataset_root=args.dataset_root.resolve(),
            baseline_root=args.baseline_root.resolve(),
            legacy_root=args.frozen_results_root.resolve(),
            output_root=output,
            config=config,
            api=api,
        )
        common[case_id] = case_common
        sealed[split][case_id] = predictions
        future[split][case_id] = observations
    calibration = {}
    unit_multiplier = {name: 1.0 for name in TREATMENTS}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        if case["split"] != "calibration":
            continue
        case_id = str(case["case_id"])
        calibration[case_id] = _score_case(
            case,
            sealed["calibration"][case_id],
            future["calibration"][case_id],
            output_root=output,
            legacy_root=args.frozen_results_root.resolve(),
            multiplier_by_treatment=unit_multiplier,
            config=config,
            api=api,
        )
    multipliers = {}
    for treatment in TREATMENTS:
        requirements = [
            max(
                1.0,
                rows[treatment]["metrics"][
                    "trial_coordinate_abs_standardized_q90"
                ]
                / NORMAL_90,
            )
            for rows in calibration.values()
        ]
        multipliers[treatment] = float(max(requirements))
    target = {}
    for case in sorted(cases, key=lambda row: row["case_id"]):
        if case["split"] != "target":
            continue
        case_id = str(case["case_id"])
        target[case_id] = _score_case(
            case,
            sealed["target"][case_id],
            future["target"][case_id],
            output_root=output,
            legacy_root=args.frozen_results_root.resolve(),
            multiplier_by_treatment=multipliers,
            config=config,
            api=api,
        )
    target_ids = tuple(sorted(target))
    invariant = _invariants(
        common,
        target_ids,
        manifest,
        config,
        Path(__file__).resolve(),
    )
    variants = []
    summaries = {}
    for treatment in TREATMENTS:
        policy = CovariancePolicy.from_treatment(treatment)
        covariance = {
            case_id: _read_json(
                output
                / "target"
                / _safe(case_id)
                / treatment
                / "prediction_seal.json"
            )["covariance_artifact_sha256"]
            for case_id in target_ids
        }
        covariance_hash = _json_hash(
            {"policy": asdict(policy), "cases": covariance}
        )
        run_manifest = {
            "ablation_id": ABLATION_ID,
            "treatment": treatment,
            "covariance_artifact_sha256": covariance_hash,
            "invariants": invariant,
            "calibration_multiplier": multipliers[treatment],
        }
        variants.append(
            {
                "method": METHODS[treatment],
                "treatment": treatment,
                "shared_uncertainty_scale": policy.shared_uncertainty_scale,
                "gauge_factors_enabled": policy.gauge_factors_enabled,
                "run_manifest_sha256": _json_hash(run_manifest),
                "covariance_artifact_sha256": covariance_hash,
                **invariant,
            }
        )
        rows = [target[case_id][treatment] for case_id in target_ids]
        summaries[treatment] = {
            "policy": asdict(policy),
            "calibration_std_multiplier": multipliers[treatment],
            "dynamic_primary": _task_summary(
                [row for row in rows if row["case_id"].endswith("/dynamic")]
            ),
            "quasi_static_secondary": _task_summary(
                [
                    row
                    for row in rows
                    if row["case_id"].endswith("/quasi_static")
                ]
            ),
            "case_results": {row["case_id"]: row for row in rows},
        }
    ablation = {
        "schema": ABLATION_SCHEMA,
        "schema_version": 1,
        "ablation_id": ABLATION_ID,
        "reference_treatment": "independent_rows",
        "locked_factors": {
            "dataset_id": f"zenodo-13823986-{DATASET_SHA256}",
            "split_id": "cloth-repeat1-calibration-repeat2-target",
            "registered_statistical_unit": "physical cloth acquisition trial",
            "source_or_calibration_policy_frozen": True,
            "allowed_variant_difference": "prob4d-covariance-treatment-only",
            "retrospective_target_rerun": True,
        },
        "variants": variants,
        "evidence": {
            "contract": "bayesian-phystwin-decisive-evidence-v1",
            "schema_version": 1,
            "protocol_id": ABLATION_ID,
            "statistical_unit": "physical cloth acquisition trial",
            "claim_boundary": (
                "retrospective real-physical covariance attribution on the "
                "previously opened repeat-2 target; not fresh confirmation"
            ),
            "reference_method": METHODS["independent_rows"],
            "records": _evidence_records(target),
        },
        "metadata": {
            "dataset_doi": "10.5281/zenodo.13823986",
            "legacy_revision": LEGACY_REVISION,
            "benchmark_revision": BENCHMARK_REVISION,
            "calibration_multipliers": multipliers,
            "future_outcomes_opened_after_new_prediction_seals": True,
            "original_target_outcomes_previously_opened": True,
            "claim_authorized": False,
        },
    }
    _write_once(output / "ablation_input.json", ablation)
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "artifact_kind": "ClothCovarianceAttributionResult",
        "ablation_id": ABLATION_ID,
        "dataset_sha256": DATASET_SHA256,
        "invariant_digests": invariant,
        "calibration_multipliers": multipliers,
        "variant_summaries": summaries,
        "target_case_ids": list(target_ids),
        "fresh_confirmation": False,
        "retrospective_target_rerun": True,
        "future_outcomes_opened_after_new_prediction_seals": True,
        "claim_authorized": False,
        "scientific_boundary": (
            "Real physical covariance attribution under the frozen Cloth "
            "Sim2Real split and causal prefix. Repeat-2 was already opened, so "
            "the result is not an independent confirmation claim."
        ),
    }
    result["result_id"] = _json_hash(result)
    _write_once(output / "scientific_result.json", result)
    print(json.dumps({"result_id": result["result_id"]}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--frozen-results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
