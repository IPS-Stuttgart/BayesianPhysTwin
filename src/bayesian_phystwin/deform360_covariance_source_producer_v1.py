"""Exact, target-closed producer for the frozen Deform360 covariance source gate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    source_artifact_mapping,
)
from .deform360_covariance_source_inventory_v1 import (
    CROSSREPO_BINDING_ID,
    PAPER_PROTOCOL_ID,
    SOFTWARE_PROTOCOL_ID,
    validate_covariance_source_inventory_v1,
)
from .deform360_fresh_object_session_public_inputs_v6_1 import (
    prepare_deform360_disjoint_visual_window_v6_1,
)
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparsePrefixFitV5,
    Deform360JointSparseVisualWindowRowsV5,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_joint_sparse_source_runner_v5 import (
    _load_physical_archive,
    _ordinary_root,
    _verified_file,
    validate_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_receipt_v5,
)
from .deform360_registered_residual_history_v1 import (
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    ResidualHistorySourceProvenanceV1,
    run_registered_residual_history_v1,
)
from .physical_rollout_v1 import write_deterministic_npz

PRODUCER_SCHEMA: Final = "bayesian-phystwin.deform360-covariance-source-producer-v1"
PRODUCER_SCHEMA_VERSION: Final = 1
UNIT_MANIFEST_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-source-unit-prediction-v1"
)
UNIT_MANIFEST_VERSION: Final = 1
PANEL_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-source-panel-receipt-v1"
)
PANEL_RECEIPT_VERSION: Final = 1
TECHNICAL_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-source-technical-receipt-v1"
)
TECHNICAL_RECEIPT_VERSION: Final = 1

SELECTION_SHA256: Final = (
    "4dd12af9889d64976095eb9e237eeb655f9675ff7d5940aa5dfc1d4ee11f295c"
)
SOURCE_EXECUTION_LOCK_ID: Final = (
    "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"
)
SOURCE_EXECUTION_LOCK_FILE_SHA256: Final = (
    "429a2c6e745a223ef5cae3c22ab991c09201121943624a9574036fec69b2c33a"
)
UPSTREAM_REVISION: Final = "6bb16bb307349c50535b1b368c60dfb4d5d17ab9"
UPSTREAM_SOURCE_PLAN_ID: Final = (
    "6c55447a79200521c6781cf43fa1f1809006f43a190a7da23183a597c52b0067"
)
UPSTREAM_SOURCE_PLAN_FILE_SHA256: Final = (
    "fa59b4655e061d181bf95d2f3e9a6ad83290032bb0a54e870afb2b3cadb7bd23"
)
UPSTREAM_PREDICTION_BATCH_ID: Final = (
    "c030edb49f035fd918c4be060533ec580dc5d470a4e4ced901b9f44a9412c9ed"
)
UPSTREAM_PREDICTION_BATCH_FILE_SHA256: Final = (
    "c99575e7af7aa2654d69b46843a550fef51f40c56c9ba48a5cb4f7fb39f44b57"
)
UPSTREAM_PREDICTION_RECEIPT_ID: Final = (
    "2c4ad8a00c8267d52ed0d453815f7d5493e9bf55546250d6002d7cf7a89aa12c"
)
UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256: Final = (
    "2d645fe6c477ed3f2c4cb342a4de8def657cfcc33ac87d1c8399dac4dfa3bb9c"
)
UPSTREAM_EXECUTION_RECEIPT_ID: Final = (
    "cf3ebb9e69eb3c15051ba4ae39e2d0338ec244e0c49e587a277f7b36344c5f3d"
)
UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256: Final = (
    "f1cd4ccfb8281a167718a30e5a6af1caaf740ba7a9d49081638efaabdeaf8441"
)

PREFIX_RANGE: Final = (0, 58)
FUTURE_RANGE: Final = (58, 76)
OBSERVATION_STD_M: Final = 0.005
COVARIANCE_EIGENVALUE_FLOOR_M2: Final = 1e-12
ASSOCIATION_CANDIDATE_COUNT: Final = 4
ASSOCIATION_SCALE_M: Final = 0.010
MAXIMUM_ASSOCIATION_DISTANCE_M: Final = 0.040
ASSOCIATION_ENTROPY_STRENGTH: Final = 0.5
OVERLAP_DISAGREEMENT_SCALE_M: Final = 0.015
BOUNDARY_RELIABILITY_SCALE_PIXELS: Final = 8.0
BOUNDARY_RELIABILITY_FLOOR: Final = 0.25
MINIMUM_EFFECTIVE_OBSERVATION_POWER: Final = 1e-6

_INFORMATION_BOUNDARY: Final = {
    "all_100_predictions_sealed": True,
    "confirmation_outcomes_used": False,
    "confirmation_payloads_opened": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_used": False,
    "source_suffix_used": False,
    "target_informed_selection_used": False,
}
_TECHNICAL_CODES: Final = frozenset(
    {
        "input-contract-failure",
        "inventory-contract-failure",
        "output-publication-failure",
        "provider-materialization-failure",
        "runtime-identity-failure",
        "unexpected-runtime-failure",
    }
)
_IMPLEMENTATION_PATHS: Final = (
    "src/bayesian_phystwin/deform360_covariance_source_inventory_v1.py",
    "src/bayesian_phystwin/deform360_covariance_source_producer_v1.py",
    "src/bayesian_phystwin/deform360_fresh_object_session_public_inputs_v6_1.py",
    "src/bayesian_phystwin/deform360_joint_sparse_public_inputs_v5.py",
    "src/bayesian_phystwin/deform360_registered_residual_history_v1/_common.py",
    "src/bayesian_phystwin/deform360_registered_residual_history_v1/_decision.py",
    "src/bayesian_phystwin/deform360_registered_residual_history_v1/_execution.py",
    "src/bayesian_phystwin/deform360_registered_residual_history_v1/_provenance.py",
    "src/bayesian_phystwin_experiments/deform360_covariance_only_source_gate_v1.py",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_revision(value: object, *, name: str) -> str:
    _require(type(value) is str and len(value) == 40, f"{name} must be a Git SHA-1")
    result = cast(str, value)
    _require(
        all(character in "0123456789abcdef" for character in result),
        f"{name} must be lowercase hexadecimal",
    )
    return result


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_file() and not requested.is_symlink(), f"invalid {name}")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical path")
    return resolved


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_dir() and not requested.is_symlink(), f"invalid {name}")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical path")
    return resolved


def _verified_exact_file(path: str | Path, *, expected: str, name: str) -> Path:
    source = _ordinary_file(path, name=name)
    _require(_sha256_file(source) == expected, f"{name} SHA-256 changed")
    return source


def _positive_covariance(value: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (value + np.swapaxes(value, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, COVARIANCE_EIGENVALUE_FLOOR_M2)
    result = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        eigenvalues,
        eigenvectors,
        optimize=True,
    )
    return 0.5 * (result + np.swapaxes(result, -1, -2))


def _nearest_neighbors(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    count: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    _require(1 <= count <= len(reference), "invalid association candidate count")
    distances: np.ndarray = np.empty((len(query), count), dtype=np.float64)
    indices: np.ndarray = np.empty((len(query), count), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        squared = np.sum(np.square(query[start:stop, None] - reference[None]), axis=2)
        local = np.argpartition(squared, kth=count - 1, axis=1)[:, :count]
        local_squared = np.take_along_axis(squared, local, axis=1)
        order = np.argsort(local_squared, axis=1, kind="mergesort")
        local = np.take_along_axis(local, order, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(np.take_along_axis(squared, local, axis=1))
    return distances, indices


def _prior_reliability(
    confidence: np.ndarray,
    mask_distance_pixels: np.ndarray,
    overlap_disagreement_m: np.ndarray,
) -> np.ndarray:
    _require(
        confidence.shape == mask_distance_pixels.shape == overlap_disagreement_m.shape
        and np.all(np.isfinite(confidence))
        and np.all(np.isfinite(mask_distance_pixels))
        and np.all(np.isfinite(overlap_disagreement_m))
        and np.all((confidence >= 0.0) & (confidence <= 1.0))
        and np.all(mask_distance_pixels >= 0.0)
        and np.all(overlap_disagreement_m >= 0.0),
        "residual-independent reliability cues changed",
    )
    boundary = BOUNDARY_RELIABILITY_FLOOR + (1.0 - BOUNDARY_RELIABILITY_FLOOR) * (
        1.0 - np.exp(-mask_distance_pixels / BOUNDARY_RELIABILITY_SCALE_PIXELS)
    )
    overlap = np.exp(
        -0.5 * np.square(overlap_disagreement_m / OVERLAP_DISAGREEMENT_SCALE_M)
    )
    return np.clip(confidence * boundary * overlap, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CovarianceSourceResidualHistoryV1:
    """One correlation-safe, prefix-only residual observation per frame/node."""

    residual_m: np.ndarray
    valid: np.ndarray
    observation_covariance_m2: np.ndarray
    prior_reliability: np.ndarray

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual_m, dtype=np.float64, order="C")
        valid = np.asarray(self.valid, dtype=np.bool_, order="C")
        covariance = np.asarray(
            self.observation_covariance_m2, dtype=np.float64, order="C"
        )
        reliability = np.asarray(self.prior_reliability, dtype=np.float64, order="C")
        _require(
            residual.ndim == 3
            and residual.shape[2] == 3
            and valid.shape == residual.shape[:2]
            and covariance.shape == (*valid.shape, 3, 3)
            and reliability.shape == valid.shape,
            "residual-history arrays have inconsistent shapes",
        )
        _require(
            np.all(np.isfinite(residual))
            and np.all(np.isfinite(covariance))
            and np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0)),
            "residual-history arrays must be finite",
        )
        _require(
            np.array_equal(residual[~valid], np.zeros((np.sum(~valid), 3)))
            and np.array_equal(
                covariance[~valid],
                np.zeros((np.sum(~valid), 3, 3)),
            )
            and np.array_equal(reliability[~valid], np.zeros(np.sum(~valid))),
            "unsupported residual rows must remain exact zero",
        )
        if np.any(valid):
            _require(
                np.allclose(
                    covariance[valid],
                    np.swapaxes(covariance[valid], -1, -2),
                    atol=1e-14,
                    rtol=0.0,
                )
                and np.min(np.linalg.eigvalsh(covariance[valid]))
                >= -COVARIANCE_EIGENVALUE_FLOOR_M2,
                "observation covariance must be symmetric positive semidefinite",
            )
        for array in (residual, valid, covariance, reliability):
            array.setflags(write=False)
        object.__setattr__(self, "residual_m", residual)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "observation_covariance_m2", covariance)
        object.__setattr__(self, "prior_reliability", reliability)


def estimate_covariance_source_residual_history_v1(
    *,
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
    physical_prediction_m: object,
    causal_frame_stop: int = PREFIX_RANGE[1],
) -> CovarianceSourceResidualHistoryV1:
    """Associate prefix observations without residual-derived reliability.

    Candidate-specific innovations are formed before graph aggregation. Unknown-
    correlation rows are combined with normalized covariance-intersection weights,
    so duplicating an identical camera or pixel block cannot add precision.
    """

    windows = tuple(visual_windows)
    _require(
        type(causal_frame_stop) is int
        and causal_frame_stop == PREFIX_RANGE[1]
        and bool(windows)
        and all(
            isinstance(window, Deform360JointSparseVisualWindowRowsV5)
            for window in windows
        ),
        "invalid causal visual-window panel",
    )
    _require(
        all(np.all(window.frame_indices < causal_frame_stop) for window in windows),
        "causal history received a post-cutoff observation",
    )
    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[0] >= causal_frame_stop
        and physical.shape[1] >= 1
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical prediction must have finite shape (T,N,3)",
    )
    frames = np.concatenate([window.frame_indices for window in windows])
    points = np.concatenate([window.point_world_m for window in windows])
    point_covariance = np.concatenate(
        [window.point_covariance_m2 for window in windows]
    )
    confidence = np.concatenate([window.source_confidence for window in windows])
    mask_distance = np.concatenate([window.mask_distance_pixels for window in windows])
    disagreement = np.concatenate([window.overlap_disagreement_m for window in windows])
    node_count = physical.shape[1]
    residual = np.zeros((causal_frame_stop, node_count, 3), dtype=np.float64)
    valid = np.zeros((causal_frame_stop, node_count), dtype=np.bool_)
    covariance = np.zeros((causal_frame_stop, node_count, 3, 3), dtype=np.float64)
    reliability = np.zeros((causal_frame_stop, node_count), dtype=np.float64)
    count = min(ASSOCIATION_CANDIDATE_COUNT, node_count)
    for frame in np.unique(frames):
        selected = np.flatnonzero(frames == frame)
        reference = physical[int(frame)]
        distance, indices = _nearest_neighbors(reference, points[selected], count=count)
        shifted = np.square(distance / ASSOCIATION_SCALE_M)
        shifted -= shifted[:, :1]
        assignment = np.exp(np.clip(-0.5 * shifted, -700.0, 0.0))
        assignment /= np.sum(assignment, axis=1, keepdims=True)
        entropy = -np.sum(
            np.where(
                assignment > 0.0,
                assignment * np.log(np.maximum(assignment, 1e-300)),
                0.0,
            ),
            axis=1,
        )
        if count > 1:
            entropy /= np.log(count)
        association_power = np.exp(
            -0.5 * np.square(distance[:, 0] / MAXIMUM_ASSOCIATION_DISTANCE_M)
            - ASSOCIATION_ENTROPY_STRENGTH * entropy
        )
        association_power[distance[:, 0] > MAXIMUM_ASSOCIATION_DISTANCE_M] = 0.0
        cue_reliability = _prior_reliability(
            confidence[selected],
            mask_distance[selected],
            disagreement[selected],
        )
        row_power = cue_reliability * association_power
        admitted = row_power >= MINIMUM_EFFECTIVE_OBSERVATION_POWER
        candidate_points = reference[indices]
        predicted_mixture = np.sum(assignment[..., None] * candidate_points, axis=1)
        offsets = candidate_points - predicted_mixture[:, None, :]
        assignment_spread = np.einsum(
            "nk,nki,nkj->nij",
            assignment,
            offsets,
            offsets,
            optimize=True,
        )
        row_covariance = _positive_covariance(
            point_covariance[selected] + assignment_spread
        )
        candidate_residual = points[selected, None, :] - candidate_points
        candidate_weight = row_power[:, None] * assignment
        candidate_weight[~admitted] = 0.0
        flat_node = indices.reshape(-1)
        flat_weight = candidate_weight.reshape(-1)
        flat_prior_reliability = np.repeat(cue_reliability, count)
        flat_residual = candidate_residual.reshape(-1, 3)
        flat_covariance = np.repeat(row_covariance, count, axis=0)
        for node in np.unique(flat_node[flat_weight > 0.0]):
            chosen = (flat_node == node) & (flat_weight > 0.0)
            weights = flat_weight[chosen]
            weights /= np.sum(weights)
            observations = flat_residual[chosen]
            observation_covariance = flat_covariance[chosen]
            identity = np.broadcast_to(
                np.eye(3, dtype=np.float64), observation_covariance.shape
            )
            precision = np.linalg.solve(observation_covariance, identity)
            fused_precision = np.einsum("n,nij->ij", weights, precision, optimize=True)
            fused_covariance = _positive_covariance(
                np.linalg.solve(fused_precision, np.eye(3, dtype=np.float64))
            )
            fused_mean = fused_covariance @ np.einsum(
                "n,nij,nj->i",
                weights,
                precision,
                observations,
                optimize=True,
            )
            residual_offset = observations - fused_mean
            residual_spread = np.einsum(
                "n,ni,nj->ij",
                weights,
                residual_offset,
                residual_offset,
                optimize=True,
            )
            index = int(node)
            valid[int(frame), index] = True
            residual[int(frame), index] = fused_mean
            covariance[int(frame), index] = _positive_covariance(
                fused_covariance + residual_spread
            ) / np.max(flat_weight[chosen])
            reliability[int(frame), index] = float(
                np.sum(weights * flat_prior_reliability[chosen])
            )
    return CovarianceSourceResidualHistoryV1(
        residual_m=residual,
        valid=valid,
        observation_covariance_m2=covariance,
        prior_reliability=reliability,
    )


@dataclass(frozen=True, slots=True)
class CovarianceSourceUnitInputsV1:
    """Fully hashed prefix inputs for one registered source unit."""

    object_id: str
    episode: int
    stratum: str
    raw_prefix_range_half_open: tuple[int, int]
    physical_mode: str
    physical_archive_path: Path
    visual_inputs: tuple[tuple[str, Path, Path], ...]
    reserved_scoring_camera_ids: tuple[str, ...]
    source_artifacts: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        _require(
            type(self.object_id) is str and bool(self.object_id),
            "source object ID is invalid",
        )
        _require(
            type(self.episode) is int and self.episode >= 0,
            "source episode is invalid",
        )
        _require(self.stratum in {"sheet", "volumetric"}, "source stratum changed")
        _require(
            self.raw_prefix_range_half_open[1] - self.raw_prefix_range_half_open[0]
            == PREFIX_RANGE[1],
            "source raw prefix must contain 58 frames",
        )
        _require(
            self.physical_mode in {"warp_twin", "persistence_fallback"},
            "source physical mode changed",
        )
        _require(
            self.physical_archive_path.is_file()
            and not self.physical_archive_path.is_symlink(),
            "source physical archive is unavailable",
        )
        cameras = tuple(camera for camera, _visual, _metric in self.visual_inputs)
        _require(
            cameras == tuple(sorted(set(cameras))) and len(cameras) >= 2,
            "provider camera panel is incomplete or unordered",
        )
        _require(
            self.reserved_scoring_camera_ids
            == tuple(sorted(set(self.reserved_scoring_camera_ids)))
            and len(self.reserved_scoring_camera_ids) == 2
            and not set(cameras).intersection(self.reserved_scoring_camera_ids),
            "provider and scoring camera panels are not disjoint",
        )
        artifacts = cast(
            Mapping[str, Mapping[str, Any]], plain_json(self.source_artifacts)
        )
        _require(bool(artifacts), "source artifact roster is empty")
        object.__setattr__(self, "source_artifacts", artifacts)


def _runtime_identity(
    *,
    repository_root: Path,
    implementation_revision: str,
) -> dict[str, Any]:
    revision = _canonical_revision(
        implementation_revision,
        name="implementation_revision",
    )
    root = _ordinary_directory(repository_root, name="repository root")
    source_files: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = _ordinary_file(root / relative, name=f"implementation file {relative}")
        source_files[relative] = _sha256_file(path)
    try:
        distribution_version = importlib.metadata.version("bayesian-phystwin")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = "source-tree-uninstalled"
    identity: dict[str, Any] = {
        "schema": PRODUCER_SCHEMA,
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "implementation_revision": revision,
        "distribution": {
            "name": "bayesian-phystwin",
            "version": distribution_version,
        },
        "environment": {
            "byteorder": sys.byteorder,
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "system": platform.system(),
        },
        "numerical_runtime": {
            "float64_epsilon": float(np.finfo(np.float64).eps),
            "numpy_version": np.__version__,
        },
        "source_files_sha256": source_files,
    }
    return cast(
        dict[str, Any],
        plain_json({**identity, "runtime_id": content_id(identity)}),
    )


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    source = _ordinary_file(path, name="consumed source input")
    if root is None:
        label = source.as_posix()
    else:
        try:
            label = source.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                "consumed source input escapes its admitted root"
            ) from error
    return {
        "path": label,
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _validate_execution_receipt(path: Path) -> dict[str, Any]:
    source = _verified_exact_file(
        path,
        expected=UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        name="upstream execution receipt",
    )
    receipt = load_strict_json_object(source, label="upstream execution receipt")
    declared = receipt.get("receipt_id")
    identity = {key: item for key, item in receipt.items() if key != "receipt_id"}
    _require(
        declared == UPSTREAM_EXECUTION_RECEIPT_ID == content_id(identity),
        "upstream execution receipt identity changed",
    )
    boundary = cast(Mapping[str, Any], receipt.get("information_boundary"))
    _require(
        receipt.get("schema")
        == "bayesian-phystwin.deform360-v6-source-prediction-execution-receipt"
        and receipt.get("schema_version") == 1
        and receipt.get("source_revision") == UPSTREAM_REVISION
        and receipt.get("status") == "source-prediction-evidence-sealed"
        and receipt.get("terminal_stage") == "completed"
        and receipt.get("physical_manifest_count") == 10
        and receipt.get("source_prediction_seal_count") == 100
        and receipt.get("exit_code") == 0
        and receipt.get("error") is None
        and boundary.get("development_suffix_opened") is False
        and boundary.get("v5_confirmation_payloads_opened") is False
        and boundary.get("v5_confirmation_outcomes_used") is False
        and boundary.get("v6_target_payloads_opened") is False
        and boundary.get("v6_target_outcomes_used") is False,
        "upstream execution receipt no longer proves a target-closed source run",
    )
    artifacts = cast(Mapping[str, Mapping[str, Any]], receipt.get("artifacts"))
    expected_artifacts = {
        "prediction_batch": UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        "prediction_receipt": UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        "source_plan": UPSTREAM_SOURCE_PLAN_FILE_SHA256,
    }
    _require(
        all(
            cast(Mapping[str, Any], artifacts.get(name)).get("sha256") == digest
            for name, digest in expected_artifacts.items()
        ),
        "upstream execution artifact binding changed",
    )
    return receipt


def _validate_physical_manifest(
    path: Path,
    *,
    object_id: str,
    episode: int,
    stratum: str,
    physical_mode: str,
    physical_archive: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_strict_json_object(path, label="physical prediction manifest")
    boundary = cast(Mapping[str, Any], manifest.get("information_boundary"))
    archive = cast(Mapping[str, Any], manifest.get("physical_prediction_archive"))
    _require(
        manifest.get("artifact_kind")
        == "Deform360BiasAwareProspectivePhysicalPrediction"
        and manifest.get("schema_version") == 1
        and manifest.get("object_id") == object_id
        and manifest.get("episode_id") == episode
        and manifest.get("stratum") == stratum
        and manifest.get("physical_mode") == physical_mode
        and manifest.get("passed") is True
        and archive.get("file_sha256") == physical_archive.get("sha256")
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("outcome_read") is False,
        "physical prediction manifest changed or crossed its boundary",
    )
    return manifest


def _resolve_source_inputs(
    *,
    source_plan: Mapping[str, Any],
    input_root: Path,
    upstream_run_root: Path,
    upstream_evidence_root: Path,
    common_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[CovarianceSourceUnitInputsV1, ...]:
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
    )

    rows = {
        cast(str, raw["object_id"]): cast(Mapping[str, Any], raw)
        for raw in cast(Sequence[Mapping[str, Any]], source_plan["objects"])
    }
    _require(
        set(rows) == {item[0] for item in SOURCE_ROSTER}, "source plan roster changed"
    )
    resolved: list[CovarianceSourceUnitInputsV1] = []
    for object_id, episode, stratum in SOURCE_ROSTER:
        row = rows[object_id]
        _require(
            row.get("episode_id") == episode and row.get("stratum") == stratum,
            "source plan unit identity changed",
        )
        physical_record = cast(Mapping[str, Any], row["physical"])
        physical_path = _verified_file(
            input_root,
            physical_record,
            name=f"physical archive for {object_id}",
        )
        _require(
            physical_path == upstream_run_root
            or upstream_run_root in physical_path.parents,
            "physical archive escapes the registered upstream run root",
        )
        case_id = f"{object_id}-ep{episode:04d}"
        manifest_path = _ordinary_file(
            upstream_evidence_root
            / "physical-manifests"
            / case_id
            / "physical_prediction_manifest.json",
            name=f"physical manifest for {object_id}",
        )
        _validate_physical_manifest(
            manifest_path,
            object_id=object_id,
            episode=episode,
            stratum=stratum,
            physical_mode=cast(str, physical_record["physical_mode"]),
            physical_archive=physical_record,
        )
        artifacts: dict[str, Mapping[str, Any]] = dict(common_artifacts)
        artifacts[f"physical-manifests/{case_id}.json"] = _file_record(manifest_path)
        artifacts[f"physical/{object_id}.npz"] = _file_record(
            physical_path,
            root=input_root,
        )
        visual_inputs: list[tuple[str, Path, Path]] = []
        for visual in cast(Sequence[Mapping[str, Any]], row["visual_windows"]):
            camera = cast(str, visual["camera_id"])
            visual_path = _verified_file(
                input_root,
                cast(Mapping[str, Any], visual["decoded_uniform"]),
                name=f"disjoint visual prefix for {object_id}/{camera}",
            )
            metric_path = _verified_file(
                input_root,
                cast(Mapping[str, Any], visual["metric_prefix"]),
                name=f"metric prefix for {object_id}/{camera}",
            )
            artifacts[f"visual/{object_id}/{camera}/baseline_disjoint.npz"] = (
                _file_record(
                    visual_path,
                    root=input_root,
                )
            )
            artifacts[f"visual/{object_id}/{camera}/metric-prefix.npz"] = _file_record(
                metric_path,
                root=input_root,
            )
            visual_inputs.append((camera, visual_path, metric_path))
        prefix = cast(Sequence[int], row["raw_prefix_range_half_open"])
        resolved.append(
            CovarianceSourceUnitInputsV1(
                object_id=object_id,
                episode=episode,
                stratum=stratum,
                raw_prefix_range_half_open=(int(prefix[0]), int(prefix[1])),
                physical_mode=cast(str, physical_record["physical_mode"]),
                physical_archive_path=physical_path,
                visual_inputs=tuple(sorted(visual_inputs, key=lambda item: item[0])),
                reserved_scoring_camera_ids=tuple(
                    sorted(cast(Sequence[str], row["reserved_endpoint_camera_ids"]))
                ),
                source_artifacts=artifacts,
            )
        )
    return tuple(resolved)


def _planned_source_provenance(
    unit: CovarianceSourceUnitInputsV1,
    *,
    source_inventory_id: str,
    implementation_revision: str,
) -> ResidualHistorySourceProvenanceV1:
    provider_cameras = tuple(camera for camera, _visual, _metric in unit.visual_inputs)
    provider_inputs = tuple(
        sorted(
            {
                cast(str, record["sha256"])
                for name, record in unit.source_artifacts.items()
                if name.startswith(f"visual/{unit.object_id}/")
            }
        )
    )
    provider_configuration = {
        "association_candidate_count": ASSOCIATION_CANDIDATE_COUNT,
        "association_entropy_strength": ASSOCIATION_ENTROPY_STRENGTH,
        "association_scale_m": ASSOCIATION_SCALE_M,
        "maximum_association_distance_m": MAXIMUM_ASSOCIATION_DISTANCE_M,
        "provider_camera_family_ids": list(provider_cameras),
        "source_unit": f"{unit.object_id}#{unit.episode}",
    }
    scoring_configuration = {
        "future_range_half_open": list(FUTURE_RANGE),
        "reserved_scoring_camera_ids": list(unit.reserved_scoring_camera_ids),
        "source_unit": f"{unit.object_id}#{unit.episode}",
        "status": "reserved-before-source-suffix-opening",
    }
    scoring_plan_artifacts = tuple(
        sorted(
            content_id(
                {
                    "camera_id": camera,
                    "role": "reserved-disjoint-source-scoring-camera",
                    "software_protocol_id": SOFTWARE_PROTOCOL_ID,
                    "source_unit": f"{unit.object_id}#{unit.episode}",
                }
            )
            for camera in unit.reserved_scoring_camera_ids
        )
    )
    return ResidualHistorySourceProvenanceV1(
        source_inventory_id=source_inventory_id,
        provider_reconstruction_id=content_id(
            {"configuration": provider_configuration, "inputs": list(provider_inputs)}
        ),
        scoring_reconstruction_id=content_id(scoring_configuration),
        provider_implementation_revision=implementation_revision,
        scoring_implementation_revision=implementation_revision,
        provider_configuration_id=content_id(provider_configuration),
        scoring_configuration_id=content_id(scoring_configuration),
        provider_camera_family_ids=provider_cameras,
        scoring_camera_family_ids=unit.reserved_scoring_camera_ids,
        provider_input_artifact_ids=provider_inputs,
        scoring_input_artifact_ids=scoring_plan_artifacts,
        metadata={
            "scoring_artifacts_opened": False,
            "scoring_identity_role": "reserved-plan-not-materialized-outcome",
            "source_suffix_used": False,
        },
    )


def _reference_mean(
    physical_future_m: np.ndarray,
    history: CovarianceSourceResidualHistoryV1,
) -> np.ndarray:
    last = np.zeros((history.residual_m.shape[1], 3), dtype=np.float64)
    for node in range(history.residual_m.shape[1]):
        support = np.flatnonzero(history.valid[:, node])
        if len(support):
            last[node] = history.residual_m[int(support[-1]), node]
    return np.asarray(physical_future_m + last[None], dtype=np.float64, order="C")


def _observation_prefix(
    physical_prefix_m: np.ndarray,
    history: CovarianceSourceResidualHistoryV1,
) -> np.ndarray:
    observation = np.full(physical_prefix_m.shape, np.nan, dtype=np.float64, order="C")
    observation[history.valid] = (
        physical_prefix_m[history.valid] + history.residual_m[history.valid]
    )
    return observation


def _covariance_diagnostics(covariance: np.ndarray) -> dict[str, Any]:
    _require(
        covariance.dtype == np.dtype(np.float64)
        and covariance.ndim == 4
        and covariance.shape[-2:] == (3, 3)
        and covariance.flags.c_contiguous,
        "prediction covariance representation changed",
    )
    finite = bool(np.all(np.isfinite(covariance)))
    symmetric = bool(
        finite
        and np.allclose(
            covariance,
            np.swapaxes(covariance, -1, -2),
            atol=1e-14,
            rtol=0.0,
        )
    )
    minimum = (
        float(np.min(np.linalg.eigvalsh(covariance))) if symmetric else float("nan")
    )
    psd = bool(symmetric and minimum >= -COVARIANCE_EIGENVALUE_FLOOR_M2)
    _require(finite and symmetric and psd, "prediction covariance is not finite PSD")
    return {
        "finite": finite,
        "minimum_eigenvalue_m2": minimum,
        "positive_semidefinite": psd,
        "symmetric": symmetric,
    }


def _unit_arrays(
    unit: CovarianceSourceUnitInputsV1,
    *,
    source_inventory_id: str,
    implementation_revision: str,
    fit: Deform360JointSparsePrefixFitV5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    physical_raw, _persistence = _load_physical_archive(
        unit.physical_archive_path,
        physical_mode=unit.physical_mode,
    )
    physical = np.asarray(physical_raw, dtype=np.float64, order="C")
    windows: list[Deform360JointSparseVisualWindowRowsV5] = []
    source_hashes = {
        name: cast(str, record["sha256"])
        for name, record in unit.source_artifacts.items()
    }
    for camera, visual_path, metric_path in unit.visual_inputs:
        rows, _gauge = prepare_deform360_disjoint_visual_window_v6_1(
            camera_id=camera,
            disjoint_motioncrafter_path=visual_path,
            metric_prefix_path=metric_path,
            raw_prefix_range_half_open=unit.raw_prefix_range_half_open,
            fit=fit,
            source_artifact_ids=source_hashes,
        )
        windows.append(rows)
    history = estimate_covariance_source_residual_history_v1(
        visual_windows=windows,
        physical_prediction_m=physical,
    )
    physical_prefix = np.asarray(
        physical[slice(*PREFIX_RANGE)], dtype=np.float64, order="C"
    )
    physical_future = np.asarray(
        physical[slice(*FUTURE_RANGE)], dtype=np.float64, order="C"
    )
    registered_mean = _reference_mean(physical_future, history)
    reference_covariance = np.zeros(
        (*registered_mean.shape, 3),
        dtype=np.float64,
        order="C",
    )
    observation = _observation_prefix(physical_prefix, history)
    provenance = _planned_source_provenance(
        unit,
        source_inventory_id=source_inventory_id,
        implementation_revision=implementation_revision,
    )
    prediction = run_registered_residual_history_v1(
        physical_prefix,
        observation,
        history.valid,
        physical_future,
        registered_mean,
        reference_covariance,
        source_unit_id=f"{unit.object_id}#{unit.episode}",
        provenance=provenance,
        metadata={
            "provider_observation_covariance_sha256": _array_sha256(
                history.observation_covariance_m2
            ),
            "provider_prior_reliability_sha256": _array_sha256(
                history.prior_reliability
            ),
            "provider_row_covariance_role": (
                "admission-and-audit-only-frozen-donor-uses-registered-endpoint-model"
            ),
        },
    )
    _require(
        prediction.mean_m is registered_mean
        and prediction.mean_m.tobytes(order="C") == registered_mean.tobytes(order="C"),
        "covariance candidate changed or copied the registered mean",
    )
    arrays = {
        "covariance_m2": np.asarray(
            prediction.covariance_m2, dtype=np.float64, order="C"
        ),
        "mean_m": registered_mean,
        "observation_covariance_m2": history.observation_covariance_m2,
        "prior_reliability": history.prior_reliability,
        "residual_history_m": history.residual_m,
        "residual_valid": history.valid,
    }
    decision = prediction.decision.descriptor()
    metadata = {
        "accepted": prediction.accepted,
        "decision": decision,
        "diagnostic_code": (
            "accepted"
            if prediction.accepted
            else "+".join(cast(Sequence[str], decision["fallback_reasons"]))
        ),
        "exact_fallback": not prediction.accepted,
        "provenance": provenance.descriptor(),
    }
    return arrays, cast(dict[str, Any], plain_json(metadata))


def _array_descriptors(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "c_contiguous": bool(np.asarray(value).flags.c_contiguous),
            "dtype": np.asarray(value).dtype.str,
            "sha256": _array_sha256(np.asarray(value)),
            "shape": list(np.asarray(value).shape),
            "units": (
                "m^2"
                if "covariance" in name
                else "m"
                if name.endswith("_m")
                else "dimensionless"
            ),
        }
        for name, value in sorted(arrays.items())
    }


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(plain_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_unit_artifact(
    directory: Path,
    *,
    unit: CovarianceSourceUnitInputsV1,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source_inventory_id: str,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    archive_path = directory / "prediction-arrays.npz"
    write_deterministic_npz(archive_path, arrays)
    descriptors = _array_descriptors(arrays)
    covariance = np.asarray(arrays["covariance_m2"])
    mean = np.asarray(arrays["mean_m"])
    decision = cast(Mapping[str, Any], metadata["decision"])
    mean_digest = _array_sha256(mean)
    identity: dict[str, Any] = {
        "schema": UNIT_MANIFEST_SCHEMA,
        "schema_version": UNIT_MANIFEST_VERSION,
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": CROSSREPO_BINDING_ID,
        "source_inventory_id": source_inventory_id,
        "runtime_id": runtime["runtime_id"],
        "object_id": unit.object_id,
        "episode": unit.episode,
        "stratum": unit.stratum,
        "physical_mode": unit.physical_mode,
        "raw_prefix_range_half_open": list(unit.raw_prefix_range_half_open),
        "prefix_range_half_open": list(PREFIX_RANGE),
        "future_range_half_open": list(FUTURE_RANGE),
        "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
        "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
        "early_middle_late_covariance_scales": list(REGISTERED_COVARIANCE_SCALES),
        "observation_std_m": OBSERVATION_STD_M,
        "covariance_eigenvalue_floor_m2": COVARIANCE_EIGENVALUE_FLOOR_M2,
        "accepted": metadata["accepted"],
        "exact_fallback": metadata["exact_fallback"],
        "diagnostic_code": metadata["diagnostic_code"],
        "technical_failure": False,
        "technical_failure_code": None,
        "mean_sha256": mean_digest,
        "reference_mean_sha256": mean_digest,
        "mean_byte_identity": True,
        "mean_object_identity_at_execution": True,
        "covariance_sha256": _array_sha256(covariance),
        "covariance_diagnostics": _covariance_diagnostics(covariance),
        "reference_fallback_covariance_sha256": _array_sha256(
            np.zeros_like(covariance)
        ),
        "archive": {
            "file_sha256": _sha256_file(archive_path),
            "path": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
        },
        "arrays": descriptors,
        "decision": decision,
        "provenance": metadata["provenance"],
        "source_artifacts": unit.source_artifacts,
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    manifest = cast(
        dict[str, Any],
        plain_json({**identity, "manifest_id": content_id(identity)}),
    )
    manifest_path = directory / "prediction-manifest.json"
    _write_json_once(manifest_path, manifest)
    checksums = (
        f"{_sha256_file(archive_path)}  {archive_path.name}\n"
        f"{_sha256_file(manifest_path)}  {manifest_path.name}\n"
    )
    checksums_path = directory / "SHA256SUMS"
    with checksums_path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(checksums)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _validate_unit_artifact(directory: Path) -> dict[str, Any]:
    manifest_path = _ordinary_file(
        directory / "prediction-manifest.json",
        name="unit prediction manifest",
    )
    archive_path = _ordinary_file(
        directory / "prediction-arrays.npz",
        name="unit prediction archive",
    )
    manifest = load_strict_json_object(manifest_path, label="unit prediction manifest")
    declared = manifest.get("manifest_id")
    identity = {key: item for key, item in manifest.items() if key != "manifest_id"}
    _require(
        manifest.get("schema") == UNIT_MANIFEST_SCHEMA
        and manifest.get("schema_version") == UNIT_MANIFEST_VERSION
        and declared == content_id(identity),
        "unit prediction manifest identity changed",
    )
    archive = cast(Mapping[str, Any], manifest["archive"])
    _require(
        archive.get("file_sha256") == _sha256_file(archive_path)
        and archive.get("size_bytes") == archive_path.stat().st_size,
        "unit prediction archive bytes changed",
    )
    try:
        with np.load(archive_path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load unit prediction archive") from error
    descriptors = cast(Mapping[str, Mapping[str, Any]], manifest["arrays"])
    _require(set(arrays) == set(descriptors), "unit prediction array roster changed")
    for name, value in arrays.items():
        descriptor = descriptors[name]
        _require(
            descriptor.get("sha256") == _array_sha256(value)
            and descriptor.get("shape") == list(value.shape)
            and descriptor.get("dtype") == value.dtype.str,
            f"unit prediction array changed: {name}",
        )
    mean = np.asarray(arrays["mean_m"])
    covariance = np.asarray(arrays["covariance_m2"])
    _require(
        manifest.get("mean_sha256")
        == manifest.get("reference_mean_sha256")
        == _array_sha256(mean)
        and manifest.get("mean_byte_identity") is True,
        "unit covariance prediction changed its point mean",
    )
    _covariance_diagnostics(np.asarray(covariance, dtype=np.float64, order="C"))
    if manifest.get("exact_fallback"):
        _require(
            np.array_equal(covariance, np.zeros_like(covariance)),
            "ordinary rejection is not exact covariance fallback",
        )
    return manifest


def _prediction_record(
    *,
    outer_fold_index: int,
    unit_index: int,
    unit_manifest: Mapping[str, Any],
    unit_manifest_file_sha256: str,
    unit_archive_file_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
    )

    outer_object, outer_episode, outer_stratum = SOURCE_ROSTER[outer_fold_index]
    object_id, episode, stratum = SOURCE_ROSTER[unit_index]
    _require(
        unit_manifest.get("object_id") == object_id
        and unit_manifest.get("episode") == episode
        and unit_manifest.get("stratum") == stratum,
        "unit artifact is assigned to another source record",
    )
    exact_fallback = cast(bool, unit_manifest["exact_fallback"])
    input_files = [
        {"logical_name": name, **cast(Mapping[str, Any], record)}
        for name, record in sorted(
            cast(
                Mapping[str, Mapping[str, Any]], unit_manifest["source_artifacts"]
            ).items()
        )
    ]
    payload: dict[str, Any] = {
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": CROSSREPO_BINDING_ID,
        "selection_sha256": SELECTION_SHA256,
        "runtime_id": runtime["runtime_id"],
        "implementation_revision": runtime["implementation_revision"],
        "distribution": runtime["distribution"],
        "environment": runtime["environment"],
        "numerical_runtime": runtime["numerical_runtime"],
        "outer_fold_index": outer_fold_index,
        "outer_fold_context": {
            "episode": outer_episode,
            "object_id": outer_object,
            "stratum": outer_stratum,
        },
        "source_unit_index": unit_index,
        "object_id": object_id,
        "episode": episode,
        "stratum": stratum,
        "prefix_range_half_open": list(PREFIX_RANGE),
        "future_range_half_open": list(FUTURE_RANGE),
        "future_horizon_steps": list(range(1, FUTURE_RANGE[1] - FUTURE_RANGE[0] + 1)),
        "future_horizon_bins": ["early"] * 6 + ["middle"] * 6 + ["late"] * 6,
        "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
        "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
        "early_middle_late_covariance_scales": list(REGISTERED_COVARIANCE_SCALES),
        "observation_std_m": OBSERVATION_STD_M,
        "covariance_eigenvalue_floor_m2": COVARIANCE_EIGENVALUE_FLOOR_M2,
        "mean_sha256": unit_manifest["mean_sha256"],
        "reference_mean_sha256": unit_manifest["reference_mean_sha256"],
        "mean_dtype": cast(Mapping[str, Any], unit_manifest["arrays"])["mean_m"][
            "dtype"
        ],
        "mean_shape": cast(Mapping[str, Any], unit_manifest["arrays"])["mean_m"][
            "shape"
        ],
        "mean_c_contiguous": True,
        "mean_bytes_identical": True,
        "covariance_sha256": unit_manifest["covariance_sha256"],
        "covariance_shape": cast(Mapping[str, Any], unit_manifest["arrays"])[
            "covariance_m2"
        ]["shape"],
        "covariance_dtype": cast(Mapping[str, Any], unit_manifest["arrays"])[
            "covariance_m2"
        ]["dtype"],
        "covariance_units": "m^2",
        "covariance_diagnostics": unit_manifest["covariance_diagnostics"],
        "disposition": "exact_fallback" if exact_fallback else "candidate",
        "exact_fallback": exact_fallback,
        "exact_fallback_reference_identity": (
            unit_manifest["reference_fallback_covariance_sha256"]
            if exact_fallback
            else None
        ),
        "technical_failure": False,
        "technical_failure_code": None,
        "diagnostic_code": unit_manifest["diagnostic_code"],
        "input_files": input_files,
        "unit_manifest_id": unit_manifest["manifest_id"],
        "unit_manifest_file_sha256": unit_manifest_file_sha256,
        "prediction_payload_sha256": unit_archive_file_sha256,
        "source_suffix_used": False,
        "confirmation_outcomes_used": False,
    }
    return cast(
        dict[str, Any],
        plain_json({**payload, "prediction_id": content_id(payload)}),
    )


def _publish_records_and_batch(
    output_root: Path,
    *,
    unit_manifests: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
        seal_prediction_batch,
        validate_prediction_batch,
    )

    records: list[dict[str, Any]] = []
    record_digests: dict[str, str] = {}
    record_root = output_root / "records"
    record_root.mkdir(parents=True, exist_ok=False)
    for outer_fold_index in range(len(SOURCE_ROSTER)):
        for unit_index, manifest in enumerate(unit_manifests):
            object_id, episode, _stratum = SOURCE_ROSTER[unit_index]
            unit_directory = (
                output_root
                / "unit-artifacts"
                / f"{unit_index:02d}-{object_id}-ep{episode:04d}"
            )
            record = _prediction_record(
                outer_fold_index=outer_fold_index,
                unit_index=unit_index,
                unit_manifest=manifest,
                unit_manifest_file_sha256=_sha256_file(
                    unit_directory / "prediction-manifest.json"
                ),
                unit_archive_file_sha256=_sha256_file(
                    unit_directory / "prediction-arrays.npz"
                ),
                runtime=runtime,
            )
            path = record_root / f"{outer_fold_index:02d}-{unit_index:02d}.json"
            _write_json_once(path, record)
            records.append(record)
            record_digests[path.name] = _sha256_file(path)
    source_units = [
        {"object_id": object_id, "episode": episode, "stratum": stratum}
        for object_id, episode, stratum in SOURCE_ROSTER
    ]
    selected = {
        f"{object_id}#{episode}": records[index * len(SOURCE_ROSTER) + index][
            "prediction_id"
        ]
        for index, (object_id, episode, _stratum) in enumerate(SOURCE_ROSTER)
    }
    batch = seal_prediction_batch(
        {
            "schema": ("bayesian-phystwin.deform360-covariance-only-source-batch"),
            "schema_version": 1,
            "software_protocol_id": SOFTWARE_PROTOCOL_ID,
            "paper_protocol_id": PAPER_PROTOCOL_ID,
            "candidate": {
                "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
                "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
                "early_middle_late_covariance_scales": list(
                    REGISTERED_COVARIANCE_SCALES
                ),
                "observation_std_m": OBSERVATION_STD_M,
                "point_prediction_change_allowed": False,
            },
            "source_units": source_units,
            "records": records,
            "scoring_prediction_by_source_unit": selected,
            "information_boundary": {
                "sealed_before_source_suffix": True,
                "source_suffix_used": False,
                "confirmation_payloads_opened": False,
                "confirmation_predictions_run": False,
                "confirmation_outcomes_used": False,
                "replacement_used": False,
                "target_informed_selection_used": False,
            },
        }
    )
    validate_prediction_batch(batch)
    batch_path = output_root / "source-prediction-batch.json"
    _write_json_once(batch_path, batch)
    return batch, record_digests


def publish_covariance_source_panel_v1(
    *,
    protocol_path: str | Path,
    selection_path: str | Path,
    crossrepo_binding_path: str | Path,
    source_execution_lock_path: str | Path,
    source_inventory_path: str | Path,
    upstream_source_plan_path: str | Path,
    upstream_prediction_batch_path: str | Path,
    upstream_prediction_receipt_path: str | Path,
    upstream_execution_receipt_path: str | Path,
    upstream_run_root: str | Path,
    input_root: str | Path,
    forbidden_confirmation_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Publish the complete 10-by-10 source barrier and stop before scoring."""

    revision = _canonical_revision(
        implementation_revision,
        name="implementation_revision",
    )
    protocol_file = _ordinary_file(protocol_path, name="software protocol")
    selection_file = _verified_exact_file(
        selection_path,
        expected=SELECTION_SHA256,
        name="selection lock",
    )
    binding_file = _ordinary_file(
        crossrepo_binding_path,
        name="cross-repository binding",
    )
    source_lock_file = _verified_exact_file(
        source_execution_lock_path,
        expected=SOURCE_EXECUTION_LOCK_FILE_SHA256,
        name="source execution lock",
    )
    inventory_file = _ordinary_file(source_inventory_path, name="source inventory")
    inventory = validate_covariance_source_inventory_v1(
        load_strict_json_object(inventory_file, label="source inventory")
    )
    _require(
        inventory.get("implementation_revision") == revision,
        "source inventory was produced by another implementation revision",
    )
    forbidden = _ordinary_directory(
        forbidden_confirmation_root,
        name="forbidden confirmation root",
    )
    admitted_input_root = _ordinary_root(input_root)
    run_root = _ordinary_directory(upstream_run_root, name="upstream run root")
    output = Path(output_root).absolute()
    _require(not output.exists(), "refusing to overwrite a source prediction panel")
    for admitted in (admitted_input_root, run_root, output.parent.resolve(strict=True)):
        _require(
            admitted != forbidden
            and forbidden not in admitted.parents
            and admitted not in forbidden.parents,
            "source producer path overlaps the forbidden confirmation root",
        )
    runtime = _runtime_identity(
        repository_root=Path(repository_root),
        implementation_revision=revision,
    )
    source_lock = load_deform360_joint_sparse_source_execution_lock_v5(source_lock_file)
    _require(
        source_lock.get("execution_lock_id") == SOURCE_EXECUTION_LOCK_ID,
        "source execution lock identity changed",
    )
    plan_file = _verified_exact_file(
        upstream_source_plan_path,
        expected=UPSTREAM_SOURCE_PLAN_FILE_SHA256,
        name="upstream source plan",
    )
    batch_file = _verified_exact_file(
        upstream_prediction_batch_path,
        expected=UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        name="upstream prediction batch",
    )
    prediction_receipt_file = _verified_exact_file(
        upstream_prediction_receipt_path,
        expected=UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        name="upstream prediction receipt",
    )
    execution_receipt_file = _verified_exact_file(
        upstream_execution_receipt_path,
        expected=UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        name="upstream execution receipt",
    )
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        load_strict_json_object(plan_file, label="upstream source plan"),
        lock=source_lock,
    )
    upstream_batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_file, label="upstream prediction batch"),
        source_lock,
    )
    upstream_receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
        load_strict_json_object(
            prediction_receipt_file,
            label="upstream prediction receipt",
        ),
        lock=source_lock,
        plan=plan,
        prediction_batch=upstream_batch,
        prediction_batch_file_sha256=_sha256_file(batch_file),
    )
    _validate_execution_receipt(execution_receipt_file)
    _require(
        plan.get("plan_id") == UPSTREAM_SOURCE_PLAN_ID
        and plan.get("implementation_revision") == UPSTREAM_REVISION
        and upstream_batch.get("prediction_batch_id") == UPSTREAM_PREDICTION_BATCH_ID
        and upstream_receipt.get("receipt_id") == UPSTREAM_PREDICTION_RECEIPT_ID,
        "upstream source evidence identity changed",
    )
    common_artifacts: dict[str, Mapping[str, Any]] = {
        "contracts/software-protocol.json": _file_record(protocol_file),
        "contracts/selection-lock.json": _file_record(selection_file),
        "contracts/crossrepo-binding.json": _file_record(binding_file),
        "contracts/source-execution-lock.json": _file_record(source_lock_file),
        "inventory/source-input-inventory.json": _file_record(inventory_file),
        "upstream/execution-receipt.json": _file_record(execution_receipt_file),
        "upstream/source-plan.json": _file_record(plan_file),
        "upstream/source-prediction-batch.json": _file_record(batch_file),
        "upstream/source-prediction-receipt.json": _file_record(
            prediction_receipt_file
        ),
    }
    units = _resolve_source_inputs(
        source_plan=plan,
        input_root=admitted_input_root,
        upstream_run_root=run_root,
        upstream_evidence_root=execution_receipt_file.parent,
        common_artifacts=common_artifacts,
    )
    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
    )

    fit = Deform360JointSparsePrefixFitV5(
        fit_object_ids=tuple(sorted(item[0] for item in SOURCE_ROSTER)),
        source_artifact_ids=source_artifact_mapping(
            {
                name: cast(str, record["sha256"])
                for name, record in common_artifacts.items()
            },
            name="covariance source producer fit inputs",
        ),
    )
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        unit_root = temporary / "unit-artifacts"
        unit_root.mkdir()
        unit_manifests: list[dict[str, Any]] = []
        for unit_index, unit in enumerate(units):
            arrays, metadata = _unit_arrays(
                unit,
                source_inventory_id=cast(str, inventory["inventory_id"]),
                implementation_revision=revision,
                fit=fit,
            )
            directory = (
                unit_root / f"{unit_index:02d}-{unit.object_id}-ep{unit.episode:04d}"
            )
            manifest = _publish_unit_artifact(
                directory,
                unit=unit,
                arrays=arrays,
                metadata=metadata,
                runtime=runtime,
                source_inventory_id=cast(str, inventory["inventory_id"]),
            )
            unit_manifests.append(_validate_unit_artifact(directory))
            _require(manifest == unit_manifests[-1], "unit artifact changed on reload")
        batch, record_digests = _publish_records_and_batch(
            temporary,
            unit_manifests=unit_manifests,
            runtime=runtime,
        )
        unit_manifest_digests = {
            f"{index:02d}": _sha256_file(
                temporary
                / "unit-artifacts"
                / f"{index:02d}-{unit.object_id}-ep{unit.episode:04d}"
                / "prediction-manifest.json"
            )
            for index, unit in enumerate(units)
        }
        receipt_identity: dict[str, Any] = {
            "schema": PANEL_RECEIPT_SCHEMA,
            "schema_version": PANEL_RECEIPT_VERSION,
            "status": "source-prediction-barrier-sealed",
            "software_protocol_id": SOFTWARE_PROTOCOL_ID,
            "paper_protocol_id": PAPER_PROTOCOL_ID,
            "crossrepo_binding_id": CROSSREPO_BINDING_ID,
            "source_inventory_id": inventory["inventory_id"],
            "runtime_id": runtime["runtime_id"],
            "implementation_revision": revision,
            "upstream_execution_receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID,
            "prediction_batch_id": batch["batch_id"],
            "prediction_batch_file_sha256": _sha256_file(
                temporary / "source-prediction-batch.json"
            ),
            "prediction_record_count": 100,
            "unit_artifact_count": 10,
            "candidate_unit_count": sum(
                not cast(bool, manifest["exact_fallback"])
                for manifest in unit_manifests
            ),
            "exact_fallback_unit_count": sum(
                cast(bool, manifest["exact_fallback"]) for manifest in unit_manifests
            ),
            "technical_failure_count": 0,
            "unit_manifest_file_sha256": unit_manifest_digests,
            "record_file_sha256": record_digests,
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "source_suffix_scoring_authorized": False,
            "confirmation_prediction_authorized": False,
            "confirmation_outcome_opening_authorized": False,
            "claim_authorized": False,
        }
        receipt = cast(
            dict[str, Any],
            plain_json(
                {**receipt_identity, "receipt_id": content_id(receipt_identity)}
            ),
        )
        _write_json_once(temporary / "source-panel-receipt.json", receipt)
        os.replace(temporary, output)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_covariance_source_panel_v1(output_root: str | Path) -> dict[str, Any]:
    """Rehash the complete source barrier without opening any source suffix."""

    from bayesian_phystwin_experiments.deform360_covariance_only_source_gate_v1 import (
        SOURCE_ROSTER,
        validate_prediction_batch,
    )

    root = _ordinary_directory(output_root, name="source prediction panel")
    receipt_path = _ordinary_file(
        root / "source-panel-receipt.json",
        name="source panel receipt",
    )
    batch_path = _ordinary_file(
        root / "source-prediction-batch.json",
        name="source prediction batch",
    )
    receipt = load_strict_json_object(receipt_path, label="source panel receipt")
    batch = validate_prediction_batch(
        load_strict_json_object(batch_path, label="source prediction batch")
    )
    identity = {key: item for key, item in receipt.items() if key != "receipt_id"}
    _require(
        receipt.get("schema") == PANEL_RECEIPT_SCHEMA
        and receipt.get("schema_version") == PANEL_RECEIPT_VERSION
        and receipt.get("receipt_id") == content_id(identity)
        and receipt.get("prediction_batch_id") == batch.get("batch_id")
        and receipt.get("prediction_batch_file_sha256") == _sha256_file(batch_path)
        and receipt.get("prediction_record_count") == 100
        and receipt.get("unit_artifact_count") == 10
        and receipt.get("technical_failure_count") == 0
        and receipt.get("information_boundary") == _INFORMATION_BOUNDARY,
        "source panel receipt changed",
    )
    manifests = cast(Mapping[str, str], receipt["unit_manifest_file_sha256"])
    records = cast(Mapping[str, str], receipt["record_file_sha256"])
    _require(
        set(manifests) == {f"{index:02d}" for index in range(10)}
        and set(records)
        == {
            f"{outer:02d}-{unit:02d}.json" for outer in range(10) for unit in range(10)
        },
        "source panel receipt roster changed",
    )
    unit_documents: list[dict[str, Any]] = []
    for index, (object_id, episode, _stratum) in enumerate(SOURCE_ROSTER):
        directory = root / "unit-artifacts" / f"{index:02d}-{object_id}-ep{episode:04d}"
        unit_document = _validate_unit_artifact(directory)
        unit_documents.append(unit_document)
        _require(
            _sha256_file(directory / "prediction-manifest.json")
            == manifests[f"{index:02d}"],
            "unit manifest differs from the panel receipt",
        )
    batch_records = cast(Sequence[Mapping[str, Any]], batch["records"])
    for record_index, (name, digest) in enumerate(sorted(records.items())):
        path = _ordinary_file(root / "records" / name, name="source prediction record")
        _require(_sha256_file(path) == digest, "source prediction record bytes changed")
        record = load_strict_json_object(path, label="source prediction record")
        expected_record = batch_records[record_index]
        unit_index = record_index % len(SOURCE_ROSTER)
        unit_document = unit_documents[unit_index]
        _require(
            record == expected_record
            and record.get("unit_manifest_id") == unit_document.get("manifest_id")
            and record.get("unit_manifest_file_sha256")
            == manifests[f"{unit_index:02d}"],
            "source prediction record is not the batch-bound unit record",
        )
    return receipt


def build_covariance_source_technical_receipt_v1(
    *,
    implementation_revision: str,
    terminal_stage: str,
    diagnostic_code: str,
    retained_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a bounded technical receipt; it never substitutes for a barrier."""

    revision = _canonical_revision(
        implementation_revision,
        name="implementation_revision",
    )
    _require(
        diagnostic_code in _TECHNICAL_CODES,
        "technical diagnostic code is outside the bounded vocabulary",
    )
    _require(
        type(terminal_stage) is str
        and bool(terminal_stage)
        and terminal_stage == terminal_stage.strip(),
        "terminal stage is invalid",
    )
    artifacts = {} if retained_artifacts is None else dict(retained_artifacts)
    _require(
        all(
            type(name) is str
            and type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for name, digest in artifacts.items()
        ),
        "technical receipt artifact digest changed",
    )
    identity: dict[str, Any] = {
        "schema": TECHNICAL_RECEIPT_SCHEMA,
        "schema_version": TECHNICAL_RECEIPT_VERSION,
        "status": "source-technical-negative",
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "crossrepo_binding_id": CROSSREPO_BINDING_ID,
        "implementation_revision": revision,
        "terminal_stage": terminal_stage,
        "diagnostic_code": diagnostic_code,
        "retained_artifacts": dict(sorted(artifacts.items())),
        "prediction_record_count": 0,
        "complete_barrier": False,
        "information_boundary": {
            **_INFORMATION_BOUNDARY,
            "all_100_predictions_sealed": False,
        },
        "source_suffix_scoring_authorized": False,
        "confirmation_prediction_authorized": False,
        "confirmation_outcome_opening_authorized": False,
        "claim_authorized": False,
    }
    return cast(
        dict[str, Any],
        plain_json({**identity, "receipt_id": content_id(identity)}),
    )


__all__ = [
    "COVARIANCE_EIGENVALUE_FLOOR_M2",
    "CovarianceSourceResidualHistoryV1",
    "CovarianceSourceUnitInputsV1",
    "FUTURE_RANGE",
    "OBSERVATION_STD_M",
    "PREFIX_RANGE",
    "build_covariance_source_technical_receipt_v1",
    "estimate_covariance_source_residual_history_v1",
    "publish_covariance_source_panel_v1",
    "validate_covariance_source_panel_v1",
]
