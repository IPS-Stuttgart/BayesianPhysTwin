"""Outcome-closed D1 candidate artifacts for the Deform360 v6.1 source gate.

This module consumes only the already released causal prefix and the sealed
physical/source predictions.  It deliberately has no endpoint, suffix, or
target interface.  The public Deform360 release does not identify tactile
channels with robot axes, so the three VT1 covariance variants remain explicit
unavailable carriers rather than being silently replaced by visual evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    D1_NATIVE,
    VARIANT_IDS,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)
from .deform360_fresh_object_session_source_v6_1 import (
    NESTED_REPAIR_ID,
    UPSTREAM_PREDICTION_BATCH_ID,
    UPSTREAM_REVISION,
)
from .deform360_joint_sparse_materializer_v5 import (
    Deform360JointSparseVisualWindowRowsV5,
)
from .dynamic_endpoint_model_average import (
    DynamicEndpointModelAverageConfigV2,
    infer_dynamic_endpoint_model_average,
    predict_dynamic_endpoint_model_average,
)

CANDIDATE_AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-producer-amendment"
)
CANDIDATE_AMENDMENT_ID: Final = (
    "593ca6c08ee5430ad37bc0cc5bb3d1b79d77a049714cb36a8c78696f3c68cfee"
)
CANDIDATE_AMENDMENT_FILE_SHA256: Final = (
    "6b5b5b99bea29c2927d52fb6ae52623f1d154a038932b6922900b709c30b10db"
)
EXECUTION_LOCK_ID: Final = (
    "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189"
)
EXECUTION_LOCK_FILE_SHA256: Final = (
    "429a2c6e745a223ef5cae3c22ab991c09201121943624a9574036fec69b2c33a"
)
UPSTREAM_SOURCE_PLAN_ID: Final = (
    "d9b9e4df9d020e8ae076f407f61d5e1f328c68d2f4fe4d8e4ad1688d2d253100"
)
UPSTREAM_SOURCE_PLAN_FILE_SHA256: Final = (
    "08863166df11033f4a968c94a4cb5bd02869175ce3bf1c1859c8ac49be371991"
)
UPSTREAM_PREDICTION_BATCH_FILE_SHA256: Final = (
    "5b40c7d0a71a841c937875a2fafc8d224d23204f367ab092f7f8ff92883bfd87"
)
UPSTREAM_PREDICTION_RECEIPT_ID: Final = (
    "04a5a8b71603b66850e35122405bc24c5de1e766c14cc2b58974f1ea97fb49ef"
)
UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256: Final = (
    "584597aa3025444acd334fb50c8acee26abe0af44c4dee7537aa37be37155803"
)
UPSTREAM_EXECUTION_RECEIPT_ID: Final = (
    "a408e44eaecf9e63311a2f1a6f511f130e586031e8a0e8e795d58fa5696e3026"
)
UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256: Final = (
    "5f8e8668a056cd21dbe1bfb6bf17ae8fdba201861ca32677b99ba704758e8255"
)
CANDIDATE_ARTIFACT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-artifact"
)
CANDIDATE_ARTIFACT_VERSION: Final = 1
CANDIDATE_ARCHIVE_FILENAME: Final = "candidate-arrays.npz"
CANDIDATE_SEAL_FILENAME: Final = "candidate-seal.json"
CANDIDATE_CHECKSUMS_FILENAME: Final = "SHA256SUMS"
CAUSAL_FRAME_STOP: Final = 58
EVALUATION_RANGE: Final = (58, 76)
ASSOCIATION_CANDIDATE_COUNT: Final = 4
ASSOCIATION_SCALE_M: Final = 0.010
MAXIMUM_ASSOCIATION_DISTANCE_M: Final = 0.040
ASSOCIATION_ENTROPY_STRENGTH: Final = 0.5
OVERLAP_DISAGREEMENT_SCALE_M: Final = 0.015
BOUNDARY_RELIABILITY_SCALE_PIXELS: Final = 8.0
BOUNDARY_RELIABILITY_FLOOR: Final = 0.25
BASELINE_RAW_STD_M: Final = 0.010
MINIMUM_COVARIANCE_EIGENVALUE_M2: Final = 1e-12
MINIMUM_EFFECTIVE_OBSERVATION_POWER: Final = 1e-6
SUPPORT_UPDATE_TARGET: Final = 3
PUBLIC_TACTILE_UNAVAILABLE_REASON: Final = (
    "released-tactile-robot-axis-identity-unavailable"
)

_CANDIDATE_IDS: Final = (B0, B1, D1_NATIVE)
_ARRAY_NAMES: Final = frozenset(
    {
        *(f"trajectory__{variant_id}" for variant_id in _CANDIDATE_IDS),
        *(f"covariance__{variant_id}" for variant_id in _CANDIDATE_IDS),
        "residual_history_m",
        "residual_valid",
        "observation_covariance_m2",
        "posterior_update_count",
        "posterior_final_nominal_probability",
        "posterior_component_weights",
        "posterior_component_log_evidence",
        "posterior_component_state_mean",
        "posterior_component_state_covariance",
    }
)
_ARTIFACT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_amendment_id",
        "nested_repair_id",
        "upstream_prediction_batch_id",
        "upstream_revision",
        "candidate_revision",
        "outer_held_out_object_id",
        "object_id",
        "episode_id",
        "stratum",
        "fit_object_ids",
        "risk_score",
        "technical_failure",
        "technical_failure_id",
        "variant_artifacts",
        "archive",
        "source_artifacts",
        "information_boundary",
        "candidate_artifact_id",
    }
)
_VARIANT_ARTIFACT_FIELDS: Final = frozenset(
    {
        "available",
        "prediction_artifact_id",
        "fit_artifact_id",
        "covariance_artifact_id",
        "unavailable_reason",
    }
)
_ARCHIVE_FIELDS: Final = frozenset(
    {"path", "file_sha256", "byte_count", "array_sha256"}
)
_INFORMATION_BOUNDARY: Final = {
    "source_suffix_opened": False,
    "future_object_observations_used_for_prediction": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "existing_source_provider_products_reused": True,
    "prob4d_pipeline_artifacts_reused": True,
    "prob4d_decoded_uniform_fusion_used": False,
    "motioncrafter_disjoint_baseline_used": True,
    "new_prob4d_inference_run": False,
    "new_motioncrafter_inference_run": False,
    "public_tactile_axis_identity_available": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(value),
        version=(2, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, value in sorted(arrays.items()):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(value))
        raw.flush()
        os.fsync(raw.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                plain_json(value),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != CANDIDATE_CHECKSUMS_FILENAME
    )
    with (root / CANDIDATE_CHECKSUMS_FILENAME).open(
        "w", encoding="ascii", newline="\n"
    ) as stream:
        for path in files:
            stream.write(f"{_file_sha256(path)}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_checksums(root: Path) -> None:
    expected = "".join(
        f"{_file_sha256(root / name)}  {name}\n"
        for name in sorted((CANDIDATE_ARCHIVE_FILENAME, CANDIDATE_SEAL_FILENAME))
    )
    try:
        actual = (root / CANDIDATE_CHECKSUMS_FILENAME).read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read candidate checksums") from error
    _require(actual == expected, "candidate SHA256SUMS changed")


def _canonical_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(sorted(nonempty_string(item, name=name) for item in values))
    _require(len(result) == len(set(result)), f"{name} repeat")
    _require(
        all(item == item.strip() and "\x00" not in item for item in result),
        f"{name} are not canonical",
    )
    return result


def load_deform360_v61_candidate_amendment(path: str | Path) -> Mapping[str, Any]:
    """Load the source-outcome-closed candidate producer amendment."""

    amendment = load_strict_json_object(
        path, label="Deform360 v6.1 candidate amendment"
    )
    _require(
        amendment.get("schema") == CANDIDATE_AMENDMENT_SCHEMA
        and amendment.get("schema_version") == 1,
        "candidate amendment schema changed",
    )
    declared = sha256_digest(amendment.get("amendment_id"), name="amendment_id")
    identity = {key: item for key, item in amendment.items() if key != "amendment_id"}
    _require(declared == content_id(identity), "candidate amendment identity changed")
    _require(declared == CANDIDATE_AMENDMENT_ID, "candidate amendment changed")
    boundary = cast(Mapping[str, Any], amendment.get("information_boundary"))
    _require(
        boundary.get("source_outcomes_used") is False
        and boundary.get("source_suffix_opened") is False
        and boundary.get("target_payloads_used") is False,
        "candidate amendment crossed its information boundary",
    )
    producer = cast(Mapping[str, Any], amendment.get("candidate_producer"))
    _require(
        producer.get("d1_evidence_pooling") == "object"
        and producer.get("d1_evidence_pooling_scope")
        == "within-target-object-graph-nodes-only"
        and producer.get("cross_object_parameters_fitted") is False
        and producer.get("nested_fit_roster_role") == "eligibility-and-provenance-only"
        and producer.get("missing_node_policy") == "invalid-no-nearest-fill"
        and producer.get("risk_score")
        == "one-minus-mean-clipped-update-count-over-three"
        and producer.get("public_vt1_policy")
        == "explicit-unavailable-exact-b0-fallback",
        "candidate producer semantics changed",
    )
    _require(
        producer.get("visual_provider_product")
        == "motioncrafter-baseline-disjoint-from-overlapping-window-bundle"
        and producer.get("upstream_source_plan_legacy_field_name") == "decoded_uniform"
        and producer.get("prob4d_pipeline_artifacts_reused") is True
        and producer.get("prob4d_decoded_uniform_fusion_used") is False
        and producer.get("new_prob4d_inference_run") is False
        and producer.get("new_motioncrafter_inference_run") is False,
        "candidate visual-provider provenance changed",
    )
    _require(
        boundary.get("motioncrafter_disjoint_baseline_used") is True
        and boundary.get("prob4d_pipeline_artifacts_reused") is True
        and boundary.get("prob4d_decoded_uniform_fusion_used") is False,
        "candidate information-boundary provenance changed",
    )
    _require(
        producer.get("metric_observation_variance")
        == "largest-eigenvalue-m2-added-once-to-robust-mixture-likelihood"
        and producer.get("unknown_correlation_fusion")
        == "per-frame-node-covariance-intersection-with-disagreement-spread"
        and producer.get("assignment_mixture_spread")
        == "included-in-metric-observation-covariance"
        and producer.get("state_innovation_processing")
        == "once-through-dynamic-robust-mixture-without-pre-filter-clipping",
        "candidate uncertainty or innovation semantics changed",
    )
    base_lock = amendment.get("base_execution_lock")
    _require(isinstance(base_lock, Mapping), "candidate execution lock is missing")
    _require(
        base_lock
        == {
            "execution_lock_id": EXECUTION_LOCK_ID,
            "file_sha256": EXECUTION_LOCK_FILE_SHA256,
            "path": (
                "protocols/locks/"
                "deform360_official_hub_joint_sparse_source_execution_v5.json"
            ),
        },
        "candidate execution lock changed",
    )
    upstream = amendment.get("upstream")
    _require(isinstance(upstream, Mapping), "candidate upstream binding is missing")
    _require(
        upstream
        == {
            "execution_receipt_file_sha256": UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
            "execution_receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID,
            "prediction_batch_file_sha256": UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
            "prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
            "prediction_receipt_file_sha256": UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
            "prediction_receipt_id": UPSTREAM_PREDICTION_RECEIPT_ID,
            "revision": UPSTREAM_REVISION,
            "source_plan_file_sha256": UPSTREAM_SOURCE_PLAN_FILE_SHA256,
            "source_plan_id": UPSTREAM_SOURCE_PLAN_ID,
        },
        "candidate upstream binding changed",
    )
    return amendment


@dataclass(frozen=True, slots=True)
class Deform360CausalGraphResidualHistoryV61:
    """One residual vector per directly supported graph node and prefix frame."""

    residual_m: np.ndarray
    valid: np.ndarray
    observation_covariance_m2: np.ndarray

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual_m, dtype=np.float64)
        valid = np.asarray(self.valid)
        covariance = np.asarray(self.observation_covariance_m2, dtype=np.float64)
        _require(
            residual.ndim == 3
            and residual.shape[2] == 3
            and valid.dtype.kind == "b"
            and valid.shape == residual.shape[:2],
            "causal residual history shape changed",
        )
        _require(
            covariance.shape == (*valid.shape, 3, 3)
            and np.all(np.isfinite(covariance))
            and np.allclose(
                covariance,
                np.swapaxes(covariance, -1, -2),
                atol=1e-14,
                rtol=0.0,
            )
            and np.min(np.linalg.eigvalsh(covariance[valid]), initial=0.0) >= -1e-12,
            "causal observation covariance changed",
        )
        _require(np.all(np.isfinite(residual)), "causal residual history is non-finite")
        _require(
            np.array_equal(residual[~valid], np.zeros((np.sum(~valid), 3))),
            "unsupported causal residuals must remain exact zero",
        )
        _require(
            np.array_equal(covariance[~valid], np.zeros((np.sum(~valid), 3, 3))),
            "unsupported causal covariance must remain exact zero",
        )
        residual = np.array(residual, copy=True, order="C")
        valid = np.array(valid, dtype=np.bool_, copy=True, order="C")
        covariance = np.array(covariance, copy=True, order="C")
        residual.setflags(write=False)
        valid.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "residual_m", residual)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "observation_covariance_m2", covariance)


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


def _prior_perception_reliability(
    source_confidence: np.ndarray,
    mask_distance_pixels: np.ndarray,
    overlap_disagreement_m: np.ndarray,
) -> np.ndarray:
    """Return residual-independent perception reliability in [0, 1]."""

    confidence = np.asarray(source_confidence, dtype=np.float64)
    distance = np.asarray(mask_distance_pixels, dtype=np.float64)
    disagreement = np.asarray(overlap_disagreement_m, dtype=np.float64)
    _require(
        confidence.shape == distance.shape == disagreement.shape
        and np.all(np.isfinite(confidence))
        and np.all(np.isfinite(distance))
        and np.all(np.isfinite(disagreement))
        and np.all((confidence >= 0.0) & (confidence <= 1.0))
        and np.all(distance >= 0.0)
        and np.all(disagreement >= 0.0),
        "perception reliability cues changed",
    )
    boundary = BOUNDARY_RELIABILITY_FLOOR + (1.0 - BOUNDARY_RELIABILITY_FLOOR) * (
        1.0 - np.exp(-distance / BOUNDARY_RELIABILITY_SCALE_PIXELS)
    )
    overlap = np.exp(-0.5 * np.square(disagreement / OVERLAP_DISAGREEMENT_SCALE_M))
    return np.clip(confidence * boundary * overlap, 0.0, 1.0)


def estimate_deform360_causal_graph_residual_history_v6_1(
    *,
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
    physical_prediction_m: object,
    causal_frame_stop: int = CAUSAL_FRAME_STOP,
) -> Deform360CausalGraphResidualHistoryV61:
    """Aggregate causal pixels once per frame/node without nearest filling.

    Geometry determines soft association. Reliability uses only source
    confidence, mask distance, and overlap disagreement; association entropy is
    a separate generalized-Bayes power. The state innovation never enters either
    prior quantity. Unknown-correlation rows are fused by covariance intersection,
    so duplication changes neither precision nor the number of filter updates.
    """

    _require(
        type(causal_frame_stop) is int and causal_frame_stop >= 1,
        "causal_frame_stop must be a positive integer",
    )
    windows = tuple(visual_windows)
    _require(
        bool(windows)
        and all(
            isinstance(window, Deform360JointSparseVisualWindowRowsV5)
            for window in windows
        ),
        "visual_windows must contain validated v5 rows",
    )
    _require(
        all(np.all(window.frame_indices < causal_frame_stop) for window in windows),
        "causal residual history received a post-cutoff observation",
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
    confidence = np.concatenate([window.source_confidence for window in windows])
    mask_distance = np.concatenate([window.mask_distance_pixels for window in windows])
    disagreement = np.concatenate([window.overlap_disagreement_m for window in windows])
    point_covariance = np.concatenate(
        [window.point_covariance_m2 for window in windows]
    )
    node_count = physical.shape[1]
    residual = np.zeros((causal_frame_stop, node_count, 3), dtype=np.float64)
    valid = np.zeros((causal_frame_stop, node_count), dtype=np.bool_)
    covariance = np.zeros((causal_frame_stop, node_count, 3, 3), dtype=np.float64)
    count = min(ASSOCIATION_CANDIDATE_COUNT, node_count)
    for frame in np.unique(frames):
        selected = np.flatnonzero(frames == frame)
        reference = physical[int(frame)]
        distance, indices = _nearest_neighbors(reference, points[selected], count=count)
        shifted = np.square(distance / ASSOCIATION_SCALE_M)
        shifted -= shifted[:, :1]
        assignment = np.exp(np.clip(-0.5 * shifted, -700.0, 0.0))
        assignment /= np.sum(assignment, axis=1, keepdims=True)
        positive = assignment > 0.0
        entropy = -np.sum(
            np.where(
                positive,
                assignment * np.log(np.maximum(assignment, 1e-300)),
                0.0,
            ),
            axis=1,
        )
        if count > 1:
            entropy /= np.log(count)
        distance_probability = np.exp(
            -0.5 * np.square(distance[:, 0] / MAXIMUM_ASSOCIATION_DISTANCE_M)
        )
        distance_probability[distance[:, 0] > MAXIMUM_ASSOCIATION_DISTANCE_M] = 0.0
        association_probability = distance_probability * np.exp(
            -ASSOCIATION_ENTROPY_STRENGTH * entropy
        )
        prior_reliability = _prior_perception_reliability(
            confidence[selected],
            mask_distance[selected],
            disagreement[selected],
        )
        # Association is a separate generalized-Bayes power.  It may depend on
        # candidate geometry, but it never changes prior perception reliability.
        effective_row_weight = prior_reliability * association_probability
        candidate_points = reference[indices]
        predicted = np.sum(assignment[..., None] * candidate_points, axis=1)
        row_residual = points[selected] - predicted
        association_offset = candidate_points - predicted[:, None, :]
        mixture_spread = np.einsum(
            "nk,nki,nkj->nij",
            assignment,
            association_offset,
            association_offset,
            optimize=True,
        )
        row_covariance = _positive_covariance(
            point_covariance[selected] + mixture_spread
        )
        admitted = effective_row_weight >= MINIMUM_EFFECTIVE_OBSERVATION_POWER
        contribution_weight = effective_row_weight[:, None] * assignment
        contribution_weight[~admitted] = 0.0
        flat_node = indices.reshape(-1)
        flat_weight = contribution_weight.reshape(-1)
        flat_residual = np.repeat(row_residual, count, axis=0)
        flat_covariance = np.repeat(row_covariance, count, axis=0)
        direct = np.zeros(node_count, dtype=np.bool_)
        frame_residual = np.zeros((node_count, 3), dtype=np.float64)
        frame_covariance = np.zeros((node_count, 3, 3), dtype=np.float64)
        for node in np.unique(flat_node[flat_weight > 0.0]):
            chosen = (flat_node == node) & (flat_weight > 0.0)
            weights = flat_weight[chosen]
            weights /= np.sum(weights)
            observations = flat_residual[chosen]
            observation_covariance = flat_covariance[chosen]
            precision = np.linalg.inv(observation_covariance)
            fused_precision = np.einsum("n,nij->ij", weights, precision, optimize=True)
            fused_covariance = _positive_covariance(np.linalg.inv(fused_precision))
            fused_mean = fused_covariance @ np.einsum(
                "n,nij,nj->i",
                weights,
                precision,
                observations,
                optimize=True,
            )
            disagreement_offset = observations - fused_mean
            disagreement_spread = np.einsum(
                "n,ni,nj->ij",
                weights,
                disagreement_offset,
                disagreement_offset,
                optimize=True,
            )
            direct[int(node)] = True
            frame_residual[int(node)] = fused_mean
            frame_covariance[int(node)] = _positive_covariance(
                fused_covariance + disagreement_spread
            ) / np.max(flat_weight[chosen])
        if not np.any(direct):
            continue
        residual[int(frame)] = frame_residual
        valid[int(frame)] = direct
        covariance[int(frame)] = frame_covariance
    return Deform360CausalGraphResidualHistoryV61(
        residual_m=residual,
        valid=valid,
        observation_covariance_m2=covariance,
    )


def _positive_covariance(value: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (value + np.swapaxes(value, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, MINIMUM_COVARIANCE_EIGENVALUE_M2)
    result = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        eigenvalues,
        eigenvectors,
        optimize=True,
    )
    return 0.5 * (result + np.swapaxes(result, -1, -2))


@dataclass(frozen=True, slots=True)
class Deform360V61CandidateArrays:
    """Raw point trajectories, covariance, and replay-complete D1 state."""

    arrays: Mapping[str, np.ndarray]
    risk_score: float

    def __post_init__(self) -> None:
        _require(set(self.arrays) == _ARRAY_NAMES, "candidate array roster changed")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.arrays.items():
            array = np.array(value, copy=True, order="C")
            _require(
                array.dtype.kind in "fbiu" and np.all(np.isfinite(array)),
                f"candidate array {name} is invalid",
            )
            array.setflags(write=False)
            normalized[name] = array
        trajectories = [normalized[f"trajectory__{item}"] for item in _CANDIDATE_IDS]
        _require(
            all(array.ndim == 3 and array.shape[2] == 3 for array in trajectories)
            and all(array.shape == trajectories[0].shape for array in trajectories),
            "candidate trajectory shapes changed",
        )
        node_count = trajectories[0].shape[1]
        horizon = EVALUATION_RANGE[1] - EVALUATION_RANGE[0]
        for item in _CANDIDATE_IDS:
            covariance = normalized[f"covariance__{item}"]
            _require(
                covariance.shape == (horizon, node_count, 3, 3)
                and np.allclose(
                    covariance,
                    np.swapaxes(covariance, -1, -2),
                    atol=1e-14,
                    rtol=0.0,
                )
                and np.min(np.linalg.eigvalsh(covariance)) > 0.0,
                f"candidate covariance {item} changed",
            )
        residual = normalized["residual_history_m"]
        valid = normalized["residual_valid"]
        observation_covariance = normalized["observation_covariance_m2"]
        _require(
            residual.shape == (CAUSAL_FRAME_STOP, node_count, 3)
            and valid.shape == residual.shape[:2]
            and valid.dtype.kind == "b",
            "candidate residual history changed",
        )
        _require(
            observation_covariance.shape == (CAUSAL_FRAME_STOP, node_count, 3, 3)
            and np.allclose(
                observation_covariance,
                np.swapaxes(observation_covariance, -1, -2),
                atol=1e-14,
                rtol=0.0,
            ),
            "candidate observation covariance changed",
        )
        _require(
            not isinstance(self.risk_score, (bool, np.bool_))
            and np.isfinite(self.risk_score)
            and 0.0 <= self.risk_score <= 1.0,
            "candidate risk score must lie in [0,1]",
        )
        object.__setattr__(self, "arrays", MappingProxyType(normalized))
        object.__setattr__(self, "risk_score", float(self.risk_score))


def build_deform360_v61_candidate_arrays(
    *,
    physical_prediction_m: object,
    b0_trajectory_m: object,
    b1_trajectory_m: object,
    visual_windows: Sequence[Deform360JointSparseVisualWindowRowsV5],
) -> Deform360V61CandidateArrays:
    """Build B0/B1 unchanged and D1 from the causal prefix only."""

    physical = np.asarray(physical_prediction_m)
    b0 = np.asarray(b0_trajectory_m)
    b1 = np.asarray(b1_trajectory_m)
    _require(
        physical.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and b0.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and b1.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and physical.shape == b0.shape == b1.shape
        and physical.ndim == 3
        and physical.shape[0] >= EVALUATION_RANGE[1]
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(b0))
        and np.all(np.isfinite(b1)),
        "candidate source trajectories changed",
    )
    history = estimate_deform360_causal_graph_residual_history_v6_1(
        visual_windows=visual_windows,
        physical_prediction_m=physical,
    )
    config = DynamicEndpointModelAverageConfigV2(evidence_pooling="object")
    posterior = infer_dynamic_endpoint_model_average(
        history.residual_m,
        history.valid,
        end_frame=CAUSAL_FRAME_STOP,
        config=config,
        observation_variance_m2=np.max(
            np.linalg.eigvalsh(history.observation_covariance_m2), axis=-1
        ),
    )
    correction: list[np.ndarray] = []
    covariance: list[np.ndarray] = []
    for horizon in range(1, EVALUATION_RANGE[1] - EVALUATION_RANGE[0] + 1):
        prediction = predict_dynamic_endpoint_model_average(
            posterior,
            horizon_steps=horizon,
        )
        correction.append(np.asarray(prediction.mean_m, dtype=np.float64))
        covariance.append(
            _positive_covariance(np.asarray(prediction.covariance_m2, dtype=np.float64))
        )
    d1 = np.asarray(physical, dtype=np.float64).copy()
    start, stop = EVALUATION_RANGE
    d1[start:stop] += np.stack(correction)
    node_count = physical.shape[1]
    baseline_covariance = np.broadcast_to(
        BASELINE_RAW_STD_M**2 * np.eye(3, dtype=np.float64),
        (stop - start, node_count, 3, 3),
    ).copy()
    support = np.minimum(
        np.asarray(posterior.update_count, dtype=np.float64) / SUPPORT_UPDATE_TARGET,
        1.0,
    )
    risk = float(1.0 - np.mean(support))
    arrays: dict[str, np.ndarray] = {
        f"trajectory__{B0}": np.array(b0, copy=True, order="C"),
        f"trajectory__{B1}": np.array(b1, copy=True, order="C"),
        f"trajectory__{D1_NATIVE}": d1,
        f"covariance__{B0}": baseline_covariance,
        f"covariance__{B1}": baseline_covariance.copy(),
        f"covariance__{D1_NATIVE}": np.stack(covariance),
        "residual_history_m": history.residual_m,
        "residual_valid": history.valid,
        "observation_covariance_m2": history.observation_covariance_m2,
        "posterior_update_count": posterior.update_count,
        "posterior_final_nominal_probability": posterior.final_nominal_probability,
        "posterior_component_weights": posterior.component_weights,
        "posterior_component_log_evidence": posterior.component_log_evidence,
        "posterior_component_state_mean": posterior.component_state_mean,
        "posterior_component_state_covariance": posterior.component_state_covariance,
    }
    return Deform360V61CandidateArrays(arrays=arrays, risk_score=risk)


def build_deform360_v61_technical_fallback_arrays(
    *,
    physical_prediction_m: object,
    b0_trajectory_m: object,
    b1_trajectory_m: object,
) -> Deform360V61CandidateArrays:
    """Build an auditable exact-B0 D1 carrier after a prefix-only failure."""

    physical = np.asarray(physical_prediction_m)
    b0 = np.asarray(b0_trajectory_m)
    b1 = np.asarray(b1_trajectory_m)
    _require(
        physical.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and b0.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and b1.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and physical.shape == b0.shape == b1.shape
        and physical.ndim == 3
        and physical.shape[0] >= EVALUATION_RANGE[1]
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(b0))
        and np.all(np.isfinite(b1)),
        "technical fallback trajectories changed",
    )
    node_count = physical.shape[1]
    history = np.zeros((CAUSAL_FRAME_STOP, node_count, 3), dtype=np.float64)
    valid = np.zeros((CAUSAL_FRAME_STOP, node_count), dtype=np.bool_)
    posterior = infer_dynamic_endpoint_model_average(
        history,
        valid,
        end_frame=CAUSAL_FRAME_STOP,
        config=DynamicEndpointModelAverageConfigV2(evidence_pooling="object"),
    )
    horizon = EVALUATION_RANGE[1] - EVALUATION_RANGE[0]
    baseline_covariance = np.broadcast_to(
        BASELINE_RAW_STD_M**2 * np.eye(3, dtype=np.float64),
        (horizon, node_count, 3, 3),
    ).copy()
    return Deform360V61CandidateArrays(
        arrays={
            f"trajectory__{B0}": np.array(b0, copy=True, order="C"),
            f"trajectory__{B1}": np.array(b1, copy=True, order="C"),
            f"trajectory__{D1_NATIVE}": np.array(b0, copy=True, order="C"),
            f"covariance__{B0}": baseline_covariance,
            f"covariance__{B1}": baseline_covariance.copy(),
            f"covariance__{D1_NATIVE}": baseline_covariance.copy(),
            "residual_history_m": history,
            "residual_valid": valid,
            "observation_covariance_m2": np.zeros(
                (CAUSAL_FRAME_STOP, node_count, 3, 3), dtype=np.float64
            ),
            "posterior_update_count": posterior.update_count,
            "posterior_final_nominal_probability": (
                posterior.final_nominal_probability
            ),
            "posterior_component_weights": posterior.component_weights,
            "posterior_component_log_evidence": posterior.component_log_evidence,
            "posterior_component_state_mean": posterior.component_state_mean,
            "posterior_component_state_covariance": (
                posterior.component_state_covariance
            ),
        },
        risk_score=1.0,
    )


def _fit_artifact_id(variant_id: str, fit_object_ids: Sequence[str]) -> str:
    return content_id(
        {
            "schema": "bayesian-phystwin.deform360-v6-candidate-fit-v1",
            "candidate_amendment_id": CANDIDATE_AMENDMENT_ID,
            "variant_id": variant_id,
            "fit_object_ids": list(fit_object_ids),
            "cross_object_parameters_fitted": False,
            "dynamic_endpoint_evidence_pooling": (
                "object" if variant_id == D1_NATIVE else None
            ),
            "source_suffix_outcomes_used": False,
        }
    )


def build_deform360_v61_candidate_seal(
    *,
    arrays: Deform360V61CandidateArrays,
    archive_file_sha256: str,
    archive_byte_count: int,
    candidate_revision: str,
    outer_held_out_object_id: str,
    object_id: str,
    episode_id: int,
    stratum: str,
    fit_object_ids: Sequence[str],
    source_artifacts: Mapping[str, str],
    technical_failure: bool = False,
    technical_failure_id: str | None = None,
) -> dict[str, Any]:
    """Build one outcome-free candidate seal around deterministic arrays."""

    if not isinstance(arrays, Deform360V61CandidateArrays):
        raise TypeError("arrays must be Deform360V61CandidateArrays")
    revision = exact_revision(candidate_revision, name="candidate_revision")
    outer_id = nonempty_string(
        outer_held_out_object_id, name="outer_held_out_object_id"
    )
    unit_id = nonempty_string(object_id, name="object_id")
    _require(
        outer_id == outer_id.strip()
        and unit_id == unit_id.strip()
        and "\x00" not in outer_id + unit_id,
        "candidate identities are not canonical",
    )
    _require(type(episode_id) is int and episode_id >= 0, "episode_id changed")
    _require(stratum in {"sheet", "volumetric"}, "stratum changed")
    _require(type(technical_failure) is bool, "technical_failure must be Boolean")
    failure_id = (
        sha256_digest(technical_failure_id, name="technical_failure_id")
        if technical_failure
        else None
    )
    _require(
        (technical_failure and failure_id is not None)
        or (not technical_failure and technical_failure_id is None),
        "technical failure identity changed",
    )
    fit_ids = _canonical_ids(fit_object_ids, name="fit object ID")
    array_ids = {name: _array_sha256(value) for name, value in arrays.arrays.items()}
    variants: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        if variant_id in {VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH}:
            variants[variant_id] = {
                "available": False,
                "prediction_artifact_id": None,
                "fit_artifact_id": None,
                "covariance_artifact_id": None,
                "unavailable_reason": PUBLIC_TACTILE_UNAVAILABLE_REASON,
            }
            continue
        variant_fit_ids = fit_ids if variant_id == D1_NATIVE else ()
        variants[variant_id] = {
            "available": True,
            "prediction_artifact_id": array_ids[f"trajectory__{variant_id}"],
            "fit_artifact_id": _fit_artifact_id(variant_id, variant_fit_ids),
            "covariance_artifact_id": array_ids[f"covariance__{variant_id}"],
            "unavailable_reason": None,
        }
    identity: dict[str, Any] = {
        "schema": CANDIDATE_ARTIFACT_SCHEMA,
        "schema_version": CANDIDATE_ARTIFACT_VERSION,
        "candidate_amendment_id": CANDIDATE_AMENDMENT_ID,
        "nested_repair_id": NESTED_REPAIR_ID,
        "upstream_prediction_batch_id": UPSTREAM_PREDICTION_BATCH_ID,
        "upstream_revision": UPSTREAM_REVISION,
        "candidate_revision": revision,
        "outer_held_out_object_id": outer_id,
        "object_id": unit_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "fit_object_ids": list(fit_ids),
        "risk_score": arrays.risk_score,
        "technical_failure": technical_failure,
        "technical_failure_id": failure_id,
        "variant_artifacts": variants,
        "archive": {
            "path": CANDIDATE_ARCHIVE_FILENAME,
            "file_sha256": sha256_digest(
                archive_file_sha256, name="archive_file_sha256"
            ),
            "byte_count": archive_byte_count,
            "array_sha256": array_ids,
        },
        "source_artifacts": dict(
            source_artifact_mapping(source_artifacts, name="source_artifacts")
        ),
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    _require(
        type(archive_byte_count) is int and archive_byte_count > 0,
        "archive byte count changed",
    )
    return {**identity, "candidate_artifact_id": content_id(identity)}


def publish_deform360_v61_candidate_artifact(
    arrays: Deform360V61CandidateArrays,
    output_directory: str | Path,
    **seal_arguments: Any,
) -> dict[str, Any]:
    """Atomically publish one deterministic candidate artifact."""

    destination = Path(output_directory).absolute()
    _require(
        not destination.is_symlink()
        and not any(parent.is_symlink() for parent in destination.parents),
        "candidate output path is invalid",
    )
    if destination.exists():
        seal, existing = load_deform360_v61_candidate_artifact(destination)
        expected = build_deform360_v61_candidate_seal(
            arrays=arrays,
            archive_file_sha256=cast(str, seal["archive"]["file_sha256"]),
            archive_byte_count=cast(int, seal["archive"]["byte_count"]),
            **seal_arguments,
        )
        _require(seal == expected, "existing candidate artifact differs")
        _require(
            all(
                np.array_equal(existing.arrays[name], arrays.arrays[name])
                for name in _ARRAY_NAMES
            ),
            "existing candidate arrays differ",
        )
        return seal
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        archive_path = temporary / CANDIDATE_ARCHIVE_FILENAME
        _write_deterministic_npz(archive_path, arrays.arrays)
        seal = build_deform360_v61_candidate_seal(
            arrays=arrays,
            archive_file_sha256=_file_sha256(archive_path),
            archive_byte_count=archive_path.stat().st_size,
            **seal_arguments,
        )
        _write_json(temporary / CANDIDATE_SEAL_FILENAME, seal)
        _write_checksums(temporary)
        os.replace(temporary, destination)
        return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_seal(
    seal: Mapping[str, Any],
    *,
    archive_path: Path,
) -> dict[str, Any]:
    require_exact_fields(seal, expected=_ARTIFACT_FIELDS, name="candidate seal")
    _require(
        seal.get("schema") == CANDIDATE_ARTIFACT_SCHEMA
        and seal.get("schema_version") == CANDIDATE_ARTIFACT_VERSION,
        "candidate artifact schema changed",
    )
    declared = sha256_digest(
        seal.get("candidate_artifact_id"), name="candidate_artifact_id"
    )
    body = {key: item for key, item in seal.items() if key != "candidate_artifact_id"}
    _require(declared == content_id(body), "candidate artifact identity changed")
    _require(
        seal.get("candidate_amendment_id") == CANDIDATE_AMENDMENT_ID
        and seal.get("nested_repair_id") == NESTED_REPAIR_ID
        and seal.get("upstream_prediction_batch_id") == UPSTREAM_PREDICTION_BATCH_ID
        and seal.get("upstream_revision") == UPSTREAM_REVISION,
        "candidate artifact lineage changed",
    )
    exact_revision(seal.get("candidate_revision"), name="candidate_revision")
    _require(
        seal.get("information_boundary") == _INFORMATION_BOUNDARY,
        "candidate artifact crossed its information boundary",
    )
    _canonical_ids(
        cast(Sequence[str], seal.get("fit_object_ids")), name="fit object ID"
    )
    _require(
        not isinstance(seal.get("risk_score"), bool)
        and np.isfinite(float(cast(float, seal.get("risk_score"))))
        and 0.0 <= float(cast(float, seal.get("risk_score"))) <= 1.0,
        "candidate risk score changed",
    )
    technical_failure = seal.get("technical_failure")
    _require(type(technical_failure) is bool, "technical failure flag changed")
    if technical_failure:
        sha256_digest(seal.get("technical_failure_id"), name="technical_failure_id")
    else:
        _require(
            seal.get("technical_failure_id") is None,
            "successful candidate carries a failure identity",
        )
    variants = cast(Mapping[str, Any], seal.get("variant_artifacts"))
    require_exact_fields(variants, expected=frozenset(VARIANT_IDS), name="variants")
    for variant_id in VARIANT_IDS:
        row = cast(Mapping[str, Any], variants[variant_id])
        require_exact_fields(row, expected=_VARIANT_ARTIFACT_FIELDS, name=variant_id)
        if variant_id in _CANDIDATE_IDS:
            _require(row.get("available") is True, "required candidate unavailable")
            for field in (
                "prediction_artifact_id",
                "fit_artifact_id",
                "covariance_artifact_id",
            ):
                sha256_digest(row.get(field), name=f"{variant_id}.{field}")
            _require(
                row.get("unavailable_reason") is None,
                "available candidate has an unavailable reason",
            )
        else:
            _require(
                row
                == {
                    "available": False,
                    "prediction_artifact_id": None,
                    "fit_artifact_id": None,
                    "covariance_artifact_id": None,
                    "unavailable_reason": PUBLIC_TACTILE_UNAVAILABLE_REASON,
                },
                "public VT1 unavailability changed",
            )
    source_artifact_mapping(
        cast(Mapping[str, str], seal.get("source_artifacts")),
        name="source_artifacts",
    )
    archive = cast(Mapping[str, Any], seal.get("archive"))
    require_exact_fields(archive, expected=_ARCHIVE_FIELDS, name="archive")
    _require(
        archive.get("path") == CANDIDATE_ARCHIVE_FILENAME
        and archive_path.stat().st_size == archive.get("byte_count")
        and _file_sha256(archive_path) == archive.get("file_sha256"),
        "candidate archive bytes changed",
    )
    declared_arrays = cast(Mapping[str, str], archive.get("array_sha256"))
    _require(set(declared_arrays) == _ARRAY_NAMES, "candidate array roster changed")
    return cast(dict[str, Any], plain_json(seal))


def load_deform360_v61_candidate_artifact(
    directory: str | Path,
) -> tuple[dict[str, Any], Deform360V61CandidateArrays]:
    """Load and fully revalidate one candidate artifact."""

    requested = Path(directory).absolute()
    _require(
        requested.is_dir() and not requested.is_symlink(),
        "candidate artifact root is invalid",
    )
    root = requested.resolve(strict=True)
    expected_files = {
        CANDIDATE_ARCHIVE_FILENAME,
        CANDIDATE_SEAL_FILENAME,
        CANDIDATE_CHECKSUMS_FILENAME,
    }
    _require(
        {path.name for path in root.iterdir()} == expected_files
        and all(path.is_file() and not path.is_symlink() for path in root.iterdir()),
        "candidate artifact is incomplete",
    )
    archive_path = root / CANDIDATE_ARCHIVE_FILENAME
    seal = _validate_seal(
        load_strict_json_object(root / CANDIDATE_SEAL_FILENAME, label="candidate seal"),
        archive_path=archive_path,
    )
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == _ARRAY_NAMES, "candidate array roster changed"
            )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load candidate archive") from error
    declared = cast(Mapping[str, str], seal["archive"]["array_sha256"])
    _require(
        all(_array_sha256(arrays[name]) == declared[name] for name in _ARRAY_NAMES),
        "candidate array bytes changed",
    )
    result = Deform360V61CandidateArrays(
        arrays=arrays,
        risk_score=float(seal["risk_score"]),
    )
    variants = cast(Mapping[str, Mapping[str, Any]], seal["variant_artifacts"])
    for variant_id in _CANDIDATE_IDS:
        _require(
            variants[variant_id]["prediction_artifact_id"]
            == _array_sha256(result.arrays[f"trajectory__{variant_id}"])
            and variants[variant_id]["covariance_artifact_id"]
            == _array_sha256(result.arrays[f"covariance__{variant_id}"]),
            "candidate variant artifact identity changed",
        )
    _validate_checksums(root)
    return seal, result


def raw_variants_from_deform360_v61_candidate_seal(
    seal: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert one validated candidate seal into the v6.1 raw variant rows."""

    variants = cast(Mapping[str, Mapping[str, Any]], seal["variant_artifacts"])
    fit_ids = cast(Sequence[str], seal["fit_object_ids"])
    result: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        row = variants[variant_id]
        challenger_fit = fit_ids if variant_id not in {B0, B1} else ()
        result[variant_id] = {
            "available": row["available"],
            "prediction_artifact_id": row["prediction_artifact_id"],
            "fit_artifact_id": row["fit_artifact_id"],
            "fit_object_ids": list(challenger_fit),
            "covariance_artifact_id": row["covariance_artifact_id"],
            "risk_score": seal["risk_score"] if variant_id == D1_NATIVE else None,
            "unavailable_reason": row["unavailable_reason"],
        }
    return result


__all__ = [
    "CANDIDATE_AMENDMENT_ID",
    "CANDIDATE_AMENDMENT_FILE_SHA256",
    "CANDIDATE_AMENDMENT_SCHEMA",
    "CANDIDATE_ARCHIVE_FILENAME",
    "CANDIDATE_CHECKSUMS_FILENAME",
    "CANDIDATE_SEAL_FILENAME",
    "EXECUTION_LOCK_FILE_SHA256",
    "EXECUTION_LOCK_ID",
    "Deform360CausalGraphResidualHistoryV61",
    "Deform360V61CandidateArrays",
    "UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256",
    "UPSTREAM_EXECUTION_RECEIPT_ID",
    "UPSTREAM_PREDICTION_BATCH_FILE_SHA256",
    "UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256",
    "UPSTREAM_PREDICTION_RECEIPT_ID",
    "UPSTREAM_SOURCE_PLAN_FILE_SHA256",
    "UPSTREAM_SOURCE_PLAN_ID",
    "build_deform360_v61_candidate_arrays",
    "build_deform360_v61_candidate_seal",
    "build_deform360_v61_technical_fallback_arrays",
    "estimate_deform360_causal_graph_residual_history_v6_1",
    "load_deform360_v61_candidate_amendment",
    "load_deform360_v61_candidate_artifact",
    "publish_deform360_v61_candidate_artifact",
    "raw_variants_from_deform360_v61_candidate_seal",
]
