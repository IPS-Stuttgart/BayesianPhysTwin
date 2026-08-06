#!/usr/bin/env python3
"""Seal TAPNext++ sparse-assimilation predictions before future evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_additional_bayesian_confirmation import (
    FIXED_INITIAL_STD_M,
    FIXED_INLIER_PRIOR,
    FIXED_OBSERVATION_STD_M,
    FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    FIXED_PROCESS_STD_M,
)
from bayesian_phystwin.phystwin_bayesian_anchor import (
    robust_random_walk_endpoint,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_graph_discrepancy import (
    normalized_spring_laplacian,
)
from bayesian_phystwin.phystwin_residual_dynamics import (
    _clip_residual,
    _target_validity,
)
from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.tapnextpp_sparse_assimilation import (
    SparseAssimilationConfig,
    associate_sparse_observations,
    build_sparse_graph_update,
    robust_metric_random_walk_endpoint,
)

PREDICTION_FILENAME = "tapnextpp_sparse_assimilation_prediction.npz"
REPORT_FILENAME = "tapnextpp_sparse_assimilation_prediction_report.json"
SEAL_FILENAME = "tapnextpp_sparse_assimilation_prediction_seal.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.asarray(pickle.load(stream), dtype=np.float64)


def _case_record(manifest: dict[str, Any], case_name: str) -> dict[str, Any]:
    records = [
        record
        for record in manifest["case_records"]
        if record.get("case") == case_name
    ]
    _require(len(records) == 1, f"source manifest does not bind {case_name} once")
    return records[0]


def _dense_persistence(
    baseline: np.ndarray,
    prefix_points: np.ndarray,
    prefix_visible: np.ndarray,
    prefix_motion_valid: np.ndarray,
    *,
    original_count: int,
    cap_quantile: float,
    cap_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    residual = prefix_points - baseline[: len(prefix_points), :original_count]
    valid = _target_validity(prefix_visible, prefix_motion_valid)
    endpoint = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=len(prefix_points),
        process_variance=FIXED_PROCESS_STD_M**2,
        observation_variance=FIXED_OBSERVATION_STD_M**2,
        initial_variance=FIXED_INITIAL_STD_M**2,
        inlier_prior=FIXED_INLIER_PRIOR,
        outlier_variance_multiplier=FIXED_OUTLIER_VARIANCE_MULTIPLIER,
    )
    correction = np.zeros(baseline.shape[1:], dtype=np.float64)
    variance = np.full(
        baseline.shape[1],
        FIXED_INITIAL_STD_M**2,
        dtype=np.float64,
    )
    correction[:original_count] = endpoint.mean
    variance[:original_count] = endpoint.variance
    updated = endpoint.update_count > 0
    if np.any(updated):
        reference = float(
            np.quantile(
                np.linalg.norm(endpoint.mean[updated], axis=1),
                cap_quantile,
            )
        )
        cap = max(cap_multiplier * reference, 1e-6)
        correction = _clip_residual(correction[None], cap)[0]
    else:
        reference = 0.0
        cap = None
        correction.fill(0.0)
    diagnostics = {
        "updated_identity_count": int(np.sum(updated)),
        "update_count": int(np.sum(endpoint.update_count)),
        "median_final_inlier_probability": (
            None
            if not np.any(updated)
            else float(np.median(endpoint.final_inlier_probability[updated]))
        ),
        "reference_norm_m": reference,
        "maximum_correction_m": cap,
    }
    return correction, variance, diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--physical-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def predict_case(
    protocol_path: str | Path,
    source_manifest_path: str | Path,
    case_name: str,
    prediction_input_path: str | Path,
    physical_trajectory_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build and seal one future prediction without reading its outcome."""

    protocol_file = Path(protocol_path).resolve()
    manifest_file = Path(source_manifest_path).resolve()
    input_file = Path(prediction_input_path).resolve()
    physical_file = Path(physical_trajectory_path).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    protocol = _load_json(protocol_file)
    _require(
        protocol.get("status") == "locked-before-future-assimilation-outcome",
        "protocol is not prediction-locked",
    )
    manifest = _load_json(manifest_file)
    _require(
        manifest.get("result_sha256") == canonical_sha256(manifest),
        "source manifest hash changed",
    )
    _require(
        manifest.get("protocol_sha256") == file_sha256(protocol_file),
        "source manifest binds another protocol",
    )
    record = _case_record(manifest, case_name)
    _require(
        file_sha256(input_file) == record["prediction_input"]["sha256"],
        "prediction input changed after staging",
    )
    _require(
        file_sha256(physical_file) == record["physical_trajectory"]["sha256"],
        "physical trajectory changed after staging",
    )
    baseline = _load_pickle_array(physical_file)
    with np.load(input_file, allow_pickle=False) as stored:
        prefix_points = np.asarray(stored["prefix_object_points_m"], np.float64)
        prefix_visible = np.asarray(stored["prefix_object_visibilities"], bool)
        prefix_motion_valid = np.asarray(stored["prefix_motion_valid"], bool)
        structure_points = np.asarray(stored["structure_points_m"], np.float64)
        original_count = int(stored["original_point_count"])
        surface_count = int(stored["surface_point_count"])
        train_end = int(stored["train_end_frame_exclusive"])
        future_end = int(stored["future_end_frame_exclusive"])
        graph_config = PhysTwinSpringGraphConfig(
            object_radius=float(stored["object_radius"]),
            object_max_neighbours=int(stored["object_max_neighbours"]),
            controller_radius=float(stored["controller_radius"]),
            controller_max_neighbours=int(stored["controller_max_neighbours"]),
        )
        provider_points = np.asarray(stored["provider_points_world_m"], np.float64)
        provider_support = np.asarray(stored["provider_support"], bool)
        provider_reliability = np.asarray(
            stored["provider_prior_reliability"],
            np.float64,
        )
        provider_covariance = np.asarray(
            stored["provider_covariance_m2"],
            np.float64,
        )
        provider_ids = np.asarray(stored["provider_identity_ids"], np.int64)
        source_start = int(stored["provider_source_frame_start"])
        source_end = int(stored["provider_source_frame_end_exclusive"])
        provider_gate_passed = bool(stored["provider_gate_passed"])
    _require(
        baseline.shape[:2] == (len(baseline), len(structure_points))
        and baseline.shape[2] == 3
        and len(baseline) >= future_end,
        "physical trajectory shape changed",
    )
    _require(len(prefix_points) == train_end, "prefix length changed")
    dense_config = protocol["dense_backbone"]
    dense_correction, dense_variance, dense_diagnostics = _dense_persistence(
        baseline,
        prefix_points,
        prefix_visible,
        prefix_motion_valid,
        original_count=original_count,
        cap_quantile=float(dense_config["relative_cap_quantile"]),
        cap_multiplier=float(dense_config["relative_cap_multiplier"]),
    )
    sparse_config = SparseAssimilationConfig(
        **protocol["sparse_assimilation_config"]
    )
    graph = build_phystwin_spring_graph(
        structure_points,
        None,
        config=graph_config,
    )
    springs = graph.springs[: graph.num_object_springs]
    laplacian = normalized_spring_laplacian(len(structure_points), springs)

    association = None
    endpoint = None
    update = None
    fallback_reason = None
    if provider_gate_passed:
        try:
            association = associate_sparse_observations(
                provider_points,
                provider_support,
                provider_reliability,
                provider_covariance,
                baseline[source_start:source_end],
                config=sparse_config,
            )
            endpoint = robust_metric_random_walk_endpoint(
                association.innovation_m,
                association.support,
                association.prior_reliability,
                association.covariance_m2,
                config=sparse_config,
            )
            update = build_sparse_graph_update(
                endpoint,
                association,
                dense_correction,
                laplacian,
                config=sparse_config,
            )
            if not update.accepted:
                fallback_reason = update.reason
        except ValueError as error:
            if "query is too far from the physical graph" not in str(error):
                raise
            fallback_reason = str(error)
    else:
        fallback_reason = "provider-transfer-gate-failed"

    if update is None or not update.accepted:
        direct_delta = np.zeros_like(dense_correction)
        graph_delta = np.zeros_like(dense_correction)
        graph_variance = np.zeros(len(dense_correction), dtype=np.float64)
        observed_nodes = np.empty(0, dtype=np.int64)
    else:
        direct_delta = update.direct_delta_m
        graph_delta = update.graph_delta_m
        graph_variance = update.graph_marginal_variance_m2
        observed_nodes = update.observed_nodes

    baseline_future = baseline[train_end:future_end]
    dense_future = baseline_future + dense_correction[None]
    direct_future = dense_future + direct_delta[None]
    graph_future = dense_future + graph_delta[None]
    if update is None or not update.accepted:
        _require(
            np.array_equal(direct_future, dense_future)
            and np.array_equal(graph_future, dense_future),
            "rejected sparse update did not fall back exactly",
        )
    future_count = future_end - train_end
    horizon = np.arange(1, future_count + 1, dtype=np.float64)
    noise_floor_variance = float(protocol["predictive_uq"]["noise_floor_std_m"]) ** 2
    dense_future_variance = (
        noise_floor_variance
        + dense_variance[None]
        + horizon[:, None] * FIXED_PROCESS_STD_M**2
    )
    direct_extra_variance = np.zeros(len(structure_points), dtype=np.float64)
    if update is not None and update.accepted and endpoint is not None:
        endpoint_scalar = np.linalg.eigvalsh(endpoint.covariance_m2)[:, -1]
        for identity, node in enumerate(association.map_indices):
            if node in observed_nodes:
                direct_extra_variance[node] = max(
                    direct_extra_variance[node],
                    float(endpoint_scalar[identity]),
                )
    sparse_elapsed = np.arange(
        train_end - source_end + 1,
        future_end - source_end + 1,
        dtype=np.float64,
    )
    direct_future_variance = (
        dense_future_variance
        + direct_extra_variance[None]
        + sparse_elapsed[:, None] * sparse_config.process_std_m**2
    )
    graph_future_variance = (
        dense_future_variance
        + graph_variance[None]
        + sparse_elapsed[:, None] * sparse_config.process_std_m**2
    )

    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        physical_frame_zero_m=baseline[0].astype(np.float32),
        physical_future_m=baseline_future.astype(np.float32),
        dense_persistence_future_m=dense_future.astype(np.float32),
        tapnext_direct_future_m=direct_future.astype(np.float32),
        tapnext_graph_future_m=graph_future.astype(np.float32),
        physical_variance_m2=np.full(
            (future_count, len(structure_points)),
            noise_floor_variance,
            dtype=np.float32,
        ),
        dense_persistence_variance_m2=dense_future_variance.astype(np.float32),
        tapnext_direct_variance_m2=direct_future_variance.astype(np.float32),
        tapnext_graph_variance_m2=graph_future_variance.astype(np.float32),
        provider_identity_ids=provider_ids,
        provider_source_frame_start=np.asarray(source_start, dtype=np.int64),
        provider_source_frame_end_exclusive=np.asarray(source_end, dtype=np.int64),
        train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
        future_end_frame_exclusive=np.asarray(future_end, dtype=np.int64),
        num_surface_points=np.asarray(
            original_count + surface_count,
            dtype=np.int64,
        ),
    )
    sparse_diagnostics: dict[str, Any] = {
        "provider_gate_passed": provider_gate_passed,
        "accepted": bool(update is not None and update.accepted),
        "fallback_reason": fallback_reason,
        "exact_dense_fallback": bool(
            np.array_equal(direct_future, dense_future)
            and np.array_equal(graph_future, dense_future)
        ),
    }
    if association is not None:
        sparse_diagnostics["association"] = {
            "map_indices": association.map_indices.tolist(),
            "source_distance_m": association.source_distance_m.tolist(),
            "entropy": association.entropy.tolist(),
            "candidate_indices": association.candidate_indices.tolist(),
            "candidate_probabilities": (
                association.candidate_probabilities.tolist()
            ),
            "prior_reliability_mean": float(
                np.mean(association.prior_reliability[association.support])
            ),
        }
    if endpoint is not None:
        sparse_diagnostics["robust_endpoint"] = {
            "update_count": endpoint.update_count.tolist(),
            "effective_row_count": endpoint.effective_row_count.tolist(),
            "temporal_covariance_inflation": (
                endpoint.temporal_covariance_inflation.tolist()
            ),
            "final_inlier_probability": (
                endpoint.final_inlier_probability.tolist()
            ),
        }
    if update is not None and update.accepted:
        sparse_diagnostics["graph_update"] = {
            "observed_nodes": update.observed_nodes.tolist(),
            "observed_delta_m": update.observed_delta_m.tolist(),
            "observed_variance_m2": update.observed_variance_m2.tolist(),
            "direct_delta_rms_m": float(
                np.sqrt(np.mean(np.sum(np.square(direct_delta), axis=1)))
            ),
            "graph_delta_rms_m": float(
                np.sqrt(np.mean(np.sum(np.square(graph_delta), axis=1)))
            ),
            "graph_solve_methods": list(update.graph_posterior.solve_methods),
            "graph_solve_maximum_relative_residual": float(
                max(update.graph_posterior.solve_relative_residuals)
            ),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationPrediction",
        "protocol_id": protocol["protocol_id"],
        "case": case_name,
        "arms": [
            "physical",
            "dense_persistence",
            "tapnext_direct",
            "tapnext_graph",
        ],
        "dense_backbone": dense_diagnostics,
        "sparse_update": sparse_diagnostics,
        "method_config": {
            "dense_backbone": dense_config,
            "sparse_assimilation": asdict(sparse_config),
            "predictive_uq": protocol["predictive_uq"],
        },
        "inputs": {
            "protocol_sha256": file_sha256(protocol_file),
            "source_manifest_sha256": file_sha256(manifest_file),
            "prediction_input_sha256": file_sha256(input_file),
            "physical_trajectory_sha256": file_sha256(physical_file),
        },
        "information_boundary": {
            "provider_source_prefix_read": True,
            "dense_released_prefix_read": True,
            "future_physical_rollout_read": True,
            "future_real_outcome_read": False,
            "manual_future_identity_read": False,
            "physical_innovation_used_for_prior_reliability": False,
            "innovation_processed_once_by_robust_mixture": True,
            "held_v8_accessed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / REPORT_FILENAME
    _write_json(report_path, report)
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationPredictionSeal",
        "prediction_archive_sha256": file_sha256(archive_path),
        "prediction_report_sha256": file_sha256(report_path),
    }
    seal["result_sha256"] = canonical_sha256(seal)
    _write_json(output / SEAL_FILENAME, seal)
    return report


def main() -> int:
    args = _parse_args()
    report = predict_case(
        args.protocol,
        args.source_manifest,
        args.case,
        args.prediction_input,
        args.physical_trajectory,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
