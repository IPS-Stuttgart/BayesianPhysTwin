"""Sealed prefix-only competence control for conditioned DINO correspondence.

The predictor receives one permitted manual identity query at frame 114, the
released PhysTwin trajectory over frames 114--120, and causal RGB-D for the
same interval. Manual targets after frame 114 are staged separately and cannot
be opened until the correspondence prediction has been hashed and sealed.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_conditioned_dino_correspondence import exact_fallback_points
from .phystwin_mvtracker_competence import (
    array_sha256,
    canonical_sha256,
    file_sha256,
)

PROTOCOL_ID = "phystwin-conditioned-dino-prefix-competence-v1"
CASE_NAME = "single_lift_cloth"
INPUT_FILENAME = "prediction_input.npz"
WITHHELD_FILENAME = "withheld_prefix_target.npz"
SOURCE_REPORT_FILENAME = "source_artifact_report.json"
PREDICTION_FILENAME = "conditioned_dino_prediction.npz"
PREDICTION_REPORT_FILENAME = "conditioned_dino_prediction_report.json"
PREDICTION_SEAL_FILENAME = "conditioned_dino_prediction_seal.json"
EVALUATION_FILENAME = "conditioned_dino_prefix_evaluation.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def _radial_rmse_m(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target,
        dtype=np.float64,
    )
    selected = error[np.asarray(mask, dtype=bool)]
    _require(len(selected) > 0, "RMSE requires at least one supported row")
    return float(np.sqrt(np.mean(np.sum(np.square(selected), axis=1))))


def _optional_radial_rmse_m(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if not np.any(np.asarray(mask, dtype=bool)):
        return None
    return _radial_rmse_m(prediction, target, mask)


@dataclass(frozen=True)
class PhysTwinConditionedDinoCompetenceConfig:
    """Frozen source-only competence choices."""

    case_name: str = CASE_NAME
    reference_frame: int = 114
    source_frame_end_exclusive: int = 121
    selected_identity_ids: tuple[int, ...] = (3, 4, 6, 8)
    selected_cameras: tuple[int, ...] = (0, 1, 2)
    minimum_supported_fraction: float = 0.50
    minimum_supported_gain_over_physical: float = 0.10
    minimum_overall_gain_over_physical: float = 0.0
    maximum_supported_rmse_m: float = 0.015
    maximum_endpoint_rmse_m: float = 0.015
    endpoint_frame_count: int = 2

    def __post_init__(self) -> None:
        _require(self.case_name == CASE_NAME, "source case is not frozen")
        _require(self.reference_frame >= 0, "reference frame must be nonnegative")
        _require(
            self.source_frame_end_exclusive > self.reference_frame + 1,
            "source interval is too short",
        )
        _require(
            len(self.selected_identity_ids) >= 3
            and len(set(self.selected_identity_ids)) == len(self.selected_identity_ids),
            "selected identities must be unique and contain at least three rows",
        )
        _require(
            all(identity >= 0 for identity in self.selected_identity_ids),
            "identity indices must be nonnegative",
        )
        _require(
            len(self.selected_cameras) >= 2
            and len(set(self.selected_cameras)) == len(self.selected_cameras),
            "selected cameras must be unique and multiview",
        )
        _require(
            0.0 < self.minimum_supported_fraction <= 1.0,
            "support threshold must lie in (0, 1]",
        )
        _require(
            0.0 <= self.minimum_supported_gain_over_physical < 1.0,
            "supported gain threshold must lie in [0, 1)",
        )
        _require(
            0.0 <= self.minimum_overall_gain_over_physical < 1.0,
            "overall gain threshold must lie in [0, 1)",
        )
        _require(
            self.maximum_supported_rmse_m > 0.0 and self.maximum_endpoint_rmse_m > 0.0,
            "RMSE thresholds must be positive",
        )
        _require(
            1 <= self.endpoint_frame_count < self.prefix_frame_count,
            "endpoint interval is invalid",
        )

    @property
    def prefix_frame_count(self) -> int:
        return self.source_frame_end_exclusive - self.reference_frame


def _nearest_vertex_indices(
    vertices: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = (
        np.asarray(vertices, dtype=np.float64)[None]
        - np.asarray(
            query_points,
            dtype=np.float64,
        )[:, None]
    )
    distance = np.linalg.norm(delta, axis=2)
    indices = np.argmin(distance, axis=1).astype(np.int64)
    return indices, distance[np.arange(len(indices)), indices]


def prepare_source_artifacts(
    manual_tracks_path: str | Path,
    split_path: str | Path,
    physical_trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    config: PhysTwinConditionedDinoCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Stage prediction-visible query/prior and separately withheld targets."""

    cfg = config or PhysTwinConditionedDinoCompetenceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "source artifact output already exists")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    _require(
        cfg.source_frame_end_exclusive <= train_end,
        "source interval crosses the released training boundary",
    )
    tracks = np.asarray(_load_pickle(manual_tracks_path), dtype=np.float64)
    physical = np.asarray(_load_pickle(physical_trajectory_path), dtype=np.float64)
    _require(
        tracks.ndim == 3 and tracks.shape[2] == 3,
        "manual tracks must have shape (T, K, 3)",
    )
    _require(
        physical.ndim == 3 and physical.shape[2] == 3,
        "physical trajectory must have shape (T, N, 3)",
    )
    _require(
        min(len(tracks), len(physical)) >= cfg.source_frame_end_exclusive,
        "source arrays are shorter than the frozen interval",
    )
    identity_ids = np.asarray(cfg.selected_identity_ids, dtype=np.int64)
    _require(
        int(np.max(identity_ids)) < tracks.shape[1],
        "selected identity exceeds the manual track array",
    )
    withheld = tracks[
        cfg.reference_frame : cfg.source_frame_end_exclusive,
        identity_ids,
    ].copy()
    query = withheld[0].copy()
    _require(
        np.all(np.isfinite(query)),
        "every frozen identity must be finite at the reference frame",
    )
    vertex_indices, initial_distance = _nearest_vertex_indices(
        physical[cfg.reference_frame],
        query,
    )
    _require(
        len(np.unique(vertex_indices)) == len(vertex_indices),
        "frozen identities map to duplicate physical vertices",
    )
    physical_window = physical[
        cfg.reference_frame : cfg.source_frame_end_exclusive,
        vertex_indices,
    ].copy()
    _require(
        np.all(np.isfinite(physical_window)),
        "physical prefix contains non-finite values",
    )

    input_dir = output / "prediction_input"
    withheld_dir = output / "withheld_evaluation"
    input_dir.mkdir(parents=True)
    withheld_dir.mkdir()
    input_path = input_dir / INPUT_FILENAME
    withheld_path = withheld_dir / WITHHELD_FILENAME
    np.savez_compressed(
        input_path,
        query_points_world_m=query.astype(np.float32),
        physical_points_world_m=physical_window.astype(np.float32),
        physical_vertex_indices=vertex_indices,
        identity_ids=identity_ids,
        reference_frame=np.asarray(cfg.reference_frame, dtype=np.int64),
        source_frame_end_exclusive=np.asarray(
            cfg.source_frame_end_exclusive,
            dtype=np.int64,
        ),
        train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
    )
    np.savez_compressed(
        withheld_path,
        target_tracks_world_m=withheld.astype(np.float32),
        identity_ids=identity_ids,
        reference_frame=np.asarray(cfg.reference_frame, dtype=np.int64),
        source_frame_end_exclusive=np.asarray(
            cfg.source_frame_end_exclusive,
            dtype=np.int64,
        ),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinConditionedDinoCompetenceSourceArtifacts",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "inputs": {
            "manual_tracks_sha256": file_sha256(manual_tracks_path),
            "split_sha256": file_sha256(split_path),
            "physical_trajectory_sha256": file_sha256(physical_trajectory_path),
            "released_train_end_frame_exclusive": train_end,
        },
        "prediction_input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
            "query_array_sha256": array_sha256(query.astype(np.float32)),
            "physical_array_sha256": array_sha256(physical_window.astype(np.float32)),
            "maximum_query_to_vertex_distance_m": float(np.max(initial_distance)),
        },
        "withheld_evaluation": {
            "path": str(withheld_path),
            "sha256": file_sha256(withheld_path),
            "target_array_sha256": array_sha256(withheld.astype(np.float32)),
        },
        "information_boundary": {
            "manual_source_file_loaded_during_staging": True,
            "prediction_input_manual_frames": [cfg.reference_frame],
            "prediction_input_physical_frame_range_half_open": [
                cfg.reference_frame,
                cfg.source_frame_end_exclusive,
            ],
            "withheld_manual_frame_range_half_open": [
                cfg.reference_frame,
                cfg.source_frame_end_exclusive,
            ],
            "withheld_artifact_available_to_prediction": False,
            "frame_at_or_after_train_end_retained": False,
        },
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / SOURCE_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_prediction_input(
    input_path: str | Path,
    expected_sha256: str,
    *,
    config: PhysTwinConditionedDinoCompetenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hash-validate and load the only arrays visible to prediction."""

    cfg = config or PhysTwinConditionedDinoCompetenceConfig()
    path = Path(input_path).resolve()
    _require(file_sha256(path) == expected_sha256, "prediction input hash changed")
    with np.load(path, allow_pickle=False) as stored:
        query = np.asarray(stored["query_points_world_m"], dtype=np.float32)
        physical = np.asarray(stored["physical_points_world_m"], dtype=np.float32)
        vertex_indices = np.asarray(
            stored["physical_vertex_indices"],
            dtype=np.int64,
        )
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        reference_frame = int(stored["reference_frame"])
        source_end = int(stored["source_frame_end_exclusive"])
        train_end = int(stored["train_end_frame_exclusive"])
    expected_ids = np.asarray(cfg.selected_identity_ids, dtype=np.int64)
    _require(
        query.shape == (len(expected_ids), 3),
        "query geometry differs from the protocol",
    )
    _require(
        physical.shape == (cfg.prefix_frame_count, len(expected_ids), 3),
        "physical geometry differs from the protocol",
    )
    _require(
        vertex_indices.shape == identity_ids.shape == expected_ids.shape
        and np.array_equal(identity_ids, expected_ids),
        "identity or vertex index contract differs",
    )
    _require(
        np.all(np.isfinite(query)) and np.all(np.isfinite(physical)),
        "prediction input contains non-finite values",
    )
    _require(
        reference_frame == cfg.reference_frame
        and source_end == cfg.source_frame_end_exclusive
        and source_end <= train_end,
        "prediction input interval differs from the protocol",
    )
    return query, physical, vertex_indices, identity_ids


def write_prediction_artifact(
    output_dir: str | Path,
    *,
    observed_points_world_m: np.ndarray,
    observation_covariance_world_m2: np.ndarray,
    prior_reliability: np.ndarray,
    accepted: np.ndarray,
    accepted_view_count: np.ndarray,
    physical_points_world_m: np.ndarray,
    identity_ids: np.ndarray,
    input_provenance: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    config: PhysTwinConditionedDinoCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Write one target-free observation prediction with exact fallback."""

    cfg = config or PhysTwinConditionedDinoCompetenceConfig()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "prediction output already exists")
    observed = np.asarray(observed_points_world_m, dtype=np.float32)
    covariance = np.asarray(
        observation_covariance_world_m2,
        dtype=np.float32,
    )
    reliability = np.asarray(prior_reliability, dtype=np.float32)
    accepted_mask = np.asarray(accepted, dtype=bool)
    view_count = np.asarray(accepted_view_count, dtype=np.int16)
    physical = np.asarray(physical_points_world_m, dtype=np.float32)
    ids = np.asarray(identity_ids, dtype=np.int64)
    expected_shape = (cfg.prefix_frame_count, len(ids))
    _require(
        observed.shape == physical.shape == (*expected_shape, 3),
        "point array geometry differs from the protocol",
    )
    _require(
        covariance.shape == (*expected_shape, 3, 3),
        "covariance geometry differs from the protocol",
    )
    _require(
        reliability.shape == accepted_mask.shape == view_count.shape == expected_shape,
        "observation diagnostics differ from the point arrays",
    )
    _require(
        np.array_equal(ids, np.asarray(cfg.selected_identity_ids)),
        "identity IDs differ from the protocol",
    )
    _require(np.all(np.isfinite(physical)), "physical fallback is non-finite")
    _require(
        np.all(np.isfinite(observed[accepted_mask])),
        "accepted observations must be finite",
    )
    _require(
        np.all(np.isfinite(covariance[accepted_mask])),
        "accepted covariances must be finite",
    )
    _require(
        np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    candidate = exact_fallback_points(physical, observed, accepted_mask)
    _require(
        np.array_equal(candidate[~accepted_mask], physical[~accepted_mask]),
        "rejected rows differ from exact physical fallback",
    )

    output.mkdir(parents=True)
    archive_path = output / PREDICTION_FILENAME
    np.savez_compressed(
        archive_path,
        observed_points_world_m=observed,
        observation_covariance_world_m2=covariance,
        prior_reliability=reliability,
        accepted=accepted_mask,
        accepted_view_count=view_count,
        physical_points_world_m=physical,
        candidate_points_world_m=candidate,
        identity_ids=ids,
    )
    scored_rows = np.arange(cfg.prefix_frame_count) > 0
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinConditionedDinoPrefixPrediction",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "inputs": dict(input_provenance),
        "runtime": dict(runtime_provenance),
        "implementation_sha256": dict(implementation_sha256),
        "diagnostics": {
            "accepted_fraction_after_reference": float(
                np.mean(accepted_mask[scored_rows])
            ),
            "accepted_view_count_min": int(np.min(view_count[scored_rows])),
            "accepted_view_count_max": int(np.max(view_count[scored_rows])),
            "prior_reliability_mean": float(np.mean(reliability[scored_rows])),
        },
        "output": {
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "rgb_depth_frame_range_half_open": [
                cfg.reference_frame,
                cfg.source_frame_end_exclusive,
            ],
            "manual_identity_frames_read": [cfg.reference_frame],
            "withheld_prefix_target_read": False,
            "frame_at_or_after_train_end_read": False,
            "state_innovation_used_in_prior_reliability": False,
            "rejected_rows_use_exact_physical_fallback": True,
        },
        "claim_boundary": (
            "one already-open source competence control for material "
            "correspondence; not simulator assimilation, confirmation, "
            "calibration, or state-of-the-art evidence"
        ),
    }
    report["result_sha256"] = canonical_sha256(report)
    report_path = output / PREDICTION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def seal_prediction(prediction_dir: str | Path) -> dict[str, Any]:
    """Seal a target-free prediction before the prefix target is opened."""

    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("artifact_kind") == "PhysTwinConditionedDinoPrefixPrediction",
        "prediction report kind is invalid",
    )
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "prediction report self-hash changed",
    )
    archive_sha256 = file_sha256(archive_path)
    _require(
        report.get("output", {}).get("archive_sha256") == archive_sha256,
        "prediction archive hash differs from its report",
    )
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinConditionedDinoPrefixPredictionSeal",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "prediction_report_sha256": file_sha256(report_path),
        "prediction_archive_sha256": archive_sha256,
        "prediction_result_sha256": report["result_sha256"],
        "information_boundary": {
            "prediction_hashed_before_withheld_prefix_scoring": True,
            "future_outcome_scoring_authorized": False,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal)
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    _require(not seal_path.exists(), "prediction seal already exists")
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return seal


def evaluate_competence(
    prediction_dir: str | Path,
    withheld_prefix_path: str | Path,
    expected_withheld_sha256: str,
    output_path: str | Path,
    *,
    config: PhysTwinConditionedDinoCompetenceConfig | None = None,
) -> dict[str, Any]:
    """Open the staged target and evaluate the frozen correspondence method."""

    cfg = config or PhysTwinConditionedDinoCompetenceConfig()
    prediction = Path(prediction_dir).resolve()
    report_path = prediction / PREDICTION_REPORT_FILENAME
    archive_path = prediction / PREDICTION_FILENAME
    seal_path = prediction / PREDICTION_SEAL_FILENAME
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(
        seal.get("artifact_kind") == "PhysTwinConditionedDinoPrefixPredictionSeal",
        "prediction seal kind is invalid",
    )
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "prediction seal self-hash changed",
    )
    _require(
        seal.get("prediction_report_sha256") == file_sha256(report_path)
        and seal.get("prediction_archive_sha256") == file_sha256(archive_path),
        "sealed prediction files changed",
    )
    withheld_path = Path(withheld_prefix_path).resolve()
    _require(
        file_sha256(withheld_path) == expected_withheld_sha256,
        "withheld prefix target hash changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        observed = np.asarray(stored["observed_points_world_m"], dtype=np.float32)
        candidate = np.asarray(stored["candidate_points_world_m"], dtype=np.float32)
        physical = np.asarray(stored["physical_points_world_m"], dtype=np.float32)
        accepted = np.asarray(stored["accepted"], dtype=bool)
        identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
    with np.load(withheld_path, allow_pickle=False) as stored:
        target = np.asarray(stored["target_tracks_world_m"], dtype=np.float32)
        target_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        reference_frame = int(stored["reference_frame"])
        source_end = int(stored["source_frame_end_exclusive"])
    expected_shape = (
        cfg.prefix_frame_count,
        len(cfg.selected_identity_ids),
        3,
    )
    _require(
        observed.shape
        == candidate.shape
        == physical.shape
        == target.shape
        == expected_shape,
        "prediction and target geometry differs from the protocol",
    )
    _require(
        np.array_equal(identity_ids, target_ids)
        and np.array_equal(
            identity_ids,
            np.asarray(cfg.selected_identity_ids),
        ),
        "prediction and target identities differ",
    )
    _require(
        reference_frame == cfg.reference_frame
        and source_end == cfg.source_frame_end_exclusive,
        "withheld target interval differs from the protocol",
    )

    score_rows = np.arange(cfg.prefix_frame_count) > 0
    target_valid = np.all(np.isfinite(target), axis=2)
    scored = target_valid & score_rows[:, None]
    supported = scored & accepted
    supported_fraction = float(np.sum(supported) / max(np.sum(scored), 1))
    candidate_rmse = _radial_rmse_m(candidate, target, scored)
    physical_rmse = _radial_rmse_m(physical, target, scored)
    persistence = np.repeat(target[:1], cfg.prefix_frame_count, axis=0)
    persistence_rmse = _radial_rmse_m(persistence, target, scored)
    supported_candidate_rmse = _optional_radial_rmse_m(
        observed,
        target,
        supported,
    )
    supported_physical_rmse = _optional_radial_rmse_m(
        physical,
        target,
        supported,
    )
    supported_gain = (
        (supported_physical_rmse - supported_candidate_rmse) / supported_physical_rmse
        if supported_physical_rmse is not None
        and supported_candidate_rmse is not None
        and supported_physical_rmse > 0.0
        else -1.0
    )
    overall_gain = (
        (physical_rmse - candidate_rmse) / physical_rmse
        if physical_rmse > 0.0
        else -1.0
    )
    endpoint_rows = np.arange(cfg.prefix_frame_count) >= (
        cfg.prefix_frame_count - cfg.endpoint_frame_count
    )
    endpoint_mask = target_valid & endpoint_rows[:, None]
    endpoint_rmse = _radial_rmse_m(candidate, target, endpoint_mask)
    gates = {
        "supported_fraction": (supported_fraction >= cfg.minimum_supported_fraction),
        "supported_gain_over_physical": (
            supported_gain >= cfg.minimum_supported_gain_over_physical
        ),
        "overall_nonregression": (
            overall_gain >= cfg.minimum_overall_gain_over_physical
        ),
        "supported_identity_rmse": (
            supported_candidate_rmse is not None
            and supported_candidate_rmse <= cfg.maximum_supported_rmse_m
        ),
        "endpoint_rmse": endpoint_rmse <= cfg.maximum_endpoint_rmse_m,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinConditionedDinoPrefixCompetenceResult",
        "protocol_id": PROTOCOL_ID,
        "case": CASE_NAME,
        "config": asdict(cfg),
        "metrics": {
            "supported_fraction": supported_fraction,
            "supported_point_frame_count": int(np.sum(supported)),
            "eligible_point_frame_count": int(np.sum(scored)),
            "candidate_identity_rmse_m": candidate_rmse,
            "physical_identity_rmse_m": physical_rmse,
            "persistence_identity_rmse_m": persistence_rmse,
            "overall_gain_over_physical": overall_gain,
            "supported_candidate_identity_rmse_m": supported_candidate_rmse,
            "supported_physical_identity_rmse_m": supported_physical_rmse,
            "supported_gain_over_physical": supported_gain,
            "candidate_endpoint_rmse_m": endpoint_rmse,
        },
        "gates": gates,
        "competence_gate_passed": passed,
        "decision": (
            "advance-to-separately-locked-source-panel"
            if passed
            else "stop-conditioned-dino-correspondence-route"
        ),
        "inputs": {
            "prediction_seal_sha256": file_sha256(seal_path),
            "withheld_prefix_sha256": expected_withheld_sha256,
        },
        "information_boundary": {
            "prediction_sealed_before_target_open": True,
            "scored_manual_frame_range_half_open": [
                cfg.reference_frame + 1,
                cfg.source_frame_end_exclusive,
            ],
            "frame_at_or_after_train_end_scored": False,
            "future_simulator_outcome_read": False,
        },
        "claim_boundary": (
            "one prefix-only competence control on an already-open source "
            "interaction; not a Bayesian-PhysTwin gain or state-of-the-art result"
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    output = Path(output_path).resolve()
    _require(not output.exists(), "evaluation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "CASE_NAME",
    "EVALUATION_FILENAME",
    "INPUT_FILENAME",
    "PREDICTION_FILENAME",
    "PREDICTION_REPORT_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "PROTOCOL_ID",
    "PhysTwinConditionedDinoCompetenceConfig",
    "evaluate_competence",
    "prepare_source_artifacts",
    "seal_prediction",
    "validate_prediction_input",
    "write_prediction_artifact",
]
