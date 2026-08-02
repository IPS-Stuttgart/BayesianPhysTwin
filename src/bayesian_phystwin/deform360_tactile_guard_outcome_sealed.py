"""Prospective outcome-sealed transfer for the Deform360 tactile guard.

The cohort used here is identity-known: its prefixes were processed by the
frozen V14 source study.  The V14 custody record proves that no future object
observation, identity trajectory, or metric was opened.  This module binds
that record, produces target-free guarded trajectories, and requires an
all-case seal before any outcome reader may be authorized.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_pairwise_regret_guard import (
    DUAL_BACKBONE_ARM,
    SELECTED_BACKBONE_ARM,
    predict_dual_backbone_pairwise_rbf_arrays,
)
from .deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
)
from .deform360_tactile_features import canonical_artifact_sha256
from .deform360_tactile_regret_guard import (
    TACTILE_REGRET_FEATURE_NAMES,
    TactileRegretGuardModel,
    apply_tactile_regret_guard,
)
from .phystwin_correspondence_gate import PairwiseCorrespondenceGateConfig
from .phystwin_online_belief import RecursiveRbfBeliefConfig

PROTOCOL_ID = "deform360-tactile-regret-guard-outcome-sealed-v1"
PROTOCOL_ARTIFACT_KIND = "Deform360TactileGuardOutcomeSealedProtocolV1"
BACKBONE_ARTIFACT_KIND = "Deform360TactileGuardBackboneSealV1"
PREDICTION_ARTIFACT_KIND = "Deform360TactileGuardPredictionSealV1"
UNSEALABLE_ARTIFACT_KIND = "Deform360TactileGuardUnsealableCaseV1"
BARRIER_ARTIFACT_KIND = "Deform360TactileGuardPredictionBarrierV1"
V14_DECISION_ARTIFACT_KIND = "Deform360CausalResponseDirectDepthSourceDecisionV14"
V14_PHYSICAL_ARTIFACT_KIND = "Deform360CausalResponseDirectDepthPhysicalBackboneV14"
V14_DECISION_NAMESPACE = b"deform360-causal-response-direct-depth-source-decision-v14\0"
V14_PHYSICAL_NAMESPACE = b"deform360-causal-response-direct-depth-physical-v14\0"
TACTILE_FEATURE_ARTIFACT_KIND = "Deform360CausalTactileFeatureAuditV2"
CLAIM_LABEL = "prospective_outcome_sealed_identity_known_transfer"
UPDATE_FRAMES = (19, 38, 57)
EXPECTED_FRAME_COUNT = 76
EXPECTED_CASE_COUNT = 12
ORDINARY_STATUS = "ordinary_prediction_sealed"
TECHNICAL_FALLBACK_STATUS = "technical_failure_exact_baseline_fallback_sealed"
UNSEALABLE_STATUS = "unsealable"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_oid(path: str | Path) -> str:
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git identity


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def canonical_sha256(payload: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def namespaced_canonical_sha256(
    payload: Mapping[str, Any],
    *,
    namespace: bytes,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        namespace
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON object expected: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _valid_revision(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX40.fullmatch(value))


def _validate_case_records(cases: Any) -> tuple[dict[str, Any], ...]:
    _require(isinstance(cases, list) and len(cases) == EXPECTED_CASE_COUNT, "cohort size changed")
    normalized: list[dict[str, Any]] = []
    for raw in cases:
        _require(isinstance(raw, Mapping), "cohort case must be an object")
        record = dict(raw)
        _require(
            isinstance(record.get("case"), str)
            and isinstance(record.get("object_id"), str)
            and int(record.get("episode_id", -1)) >= 0
            and int(record.get("queue_rank", -1)) >= 1
            and _valid_digest(record.get("case_hash"))
            and _valid_digest(record.get("object_hash"))
            and _valid_digest(record.get("metadata_sha256")),
            "cohort case is malformed",
        )
        expected_name = (
            f"{record['object_id']}-ep{int(record['episode_id']):04d}"
        )
        _require(record["case"] == expected_name, "cohort case name changed")
        normalized.append(record)
    _require(
        len({row["case"] for row in normalized}) == EXPECTED_CASE_COUNT
        and len({row["object_hash"] for row in normalized}) == EXPECTED_CASE_COUNT
        and len({int(row["queue_rank"]) for row in normalized})
        == EXPECTED_CASE_COUNT,
        "cohort identities are not unique",
    )
    _require(
        [int(row["queue_rank"]) for row in normalized]
        == sorted(int(row["queue_rank"]) for row in normalized),
        "cohort cases must be queue-rank ordered",
    )
    return tuple(normalized)


def load_outcome_sealed_protocol(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the frozen contract and optionally verify its method sources."""

    protocol_path = Path(path).resolve()
    payload = load_json(protocol_path)
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == PROTOCOL_ARTIFACT_KIND
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("protocol_sha256")
        == canonical_sha256(payload, digest_key="protocol_sha256"),
        "outcome-sealed protocol identity changed",
    )
    claim = payload.get("claim_boundary", {})
    _require(
        claim.get("label") == CLAIM_LABEL
        and claim.get("fresh_object_identity_confirmation") is False
        and claim.get("identity_known_before_lock") is True
        and claim.get("future_outcomes_sealed_before_prediction") is True
        and claim.get("state_of_the_art_confirmation") is False,
        "claim boundary changed",
    )
    cohort = payload.get("cohort", {})
    cases = _validate_case_records(cohort.get("cases"))
    _require(
        cohort.get("case_count") == EXPECTED_CASE_COUNT
        and cohort.get("physical_object_count") == EXPECTED_CASE_COUNT
        and cohort.get("replacement_policy") == "none"
        and cohort.get("technical_failures_remain_in_denominator") is True,
        "cohort accounting changed",
    )
    v14 = cohort.get("v14_custody", {})
    _require(
        _valid_revision(v14.get("repository_revision"))
        and _valid_revision(v14.get("source_decision_git_blob_oid"))
        and _valid_revision(v14.get("staging_queue_git_blob_oid"))
        and _valid_digest(v14.get("source_decision_artifact_sha256"))
        and v14.get("source_outcome_authorized") is False
        and v14.get("future_identity_or_metric_read") is False
        and v14.get("future_object_observation_read") is False
        and v14.get("maximum_object_observation_frame") == UPDATE_FRAMES[-1],
        "V14 custody contract changed",
    )
    method = payload.get("method", {})
    _require(
        method.get("candidate") == DUAL_BACKBONE_ARM
        and method.get("baseline") == SELECTED_BACKBONE_ARM
        and method.get("guard") == "causal_tactile_regret_guard"
        and method.get("update_frames") == list(UPDATE_FRAMES)
        and method.get("technical_failure_fallback") == "bit-exact persistence"
        and method.get("target_fitting") is False,
        "method contract changed",
    )
    tactile_source = method.get("tactile_source_model", {})
    _require(
        tactile_source.get("artifact_kind")
        == "Deform360TactileRegretGuardSourceDiagnostic"
        and _valid_digest(tactile_source.get("artifact_sha256"))
        and _valid_digest(tactile_source.get("model_sha256"))
        and tactile_source.get("feature_names")
        == list(TACTILE_REGRET_FEATURE_NAMES),
        "tactile source model contract changed",
    )
    source_hashes = method.get("source_sha256", {})
    _require(
        isinstance(source_hashes, Mapping)
        and source_hashes
        and all(_valid_digest(value) for value in source_hashes.values()),
        "method source hashes are malformed",
    )
    if repository_root is not None:
        root = Path(repository_root).resolve()
        observed = {
            name: file_sha256(root / name) for name in source_hashes
        }
        _require(observed == dict(source_hashes), "frozen method source tree changed")
    gates = payload.get("advancement_gates", {})
    _require(
        gates.get("minimum_joint_case_wins") == 2
        and gates.get("maximum_joint_case_regressions") == 0
        and gates.get("minimum_identity_improvement_percent") == 1.0
        and gates.get("minimum_chamfer_improvement_percent") == 1.0
        and gates.get("object_cluster_ci_upper_must_be_nonpositive") is True,
        "advancement gates changed",
    )
    custody = payload.get("custody", {})
    _require(
        custody.get("outcome_before_barrier") is False
        and custody.get("failed_case_replacement") is False
        and custody.get("unsealable_case_blocks_outcome") is True
        and custody.get("future_tactile_values_used") is False,
        "outcome custody changed",
    )
    payload["cohort"]["cases"] = [dict(row) for row in cases]
    payload["config_file_sha256"] = file_sha256(protocol_path)
    return payload


def validate_v14_custody(
    protocol: Mapping[str, Any],
    *,
    source_decision_path: str | Path,
    staging_queue_path: str | Path,
) -> dict[str, Any]:
    """Verify that the bound V14 cohort still has sealed future outcomes."""

    decision_path = Path(source_decision_path).resolve()
    queue_path = Path(staging_queue_path).resolve()
    custody = protocol["cohort"]["v14_custody"]
    _require(
        git_blob_oid(decision_path) == custody["source_decision_git_blob_oid"]
        and git_blob_oid(queue_path) == custody["staging_queue_git_blob_oid"],
        "V14 custody files differ from the bound Git blobs",
    )
    decision = load_json(decision_path)
    _require(
        decision.get("artifact_kind") == V14_DECISION_ARTIFACT_KIND
        and decision.get("artifact_sha256")
        == namespaced_canonical_sha256(
            decision,
            namespace=V14_DECISION_NAMESPACE,
            digest_key="artifact_sha256",
        )
        and decision.get("artifact_sha256")
        == custody["source_decision_artifact_sha256"],
        "V14 source decision is incompatible",
    )
    boundary = decision.get("information_boundary", {})
    gate = decision.get("outcome_blind_gate", {})
    _require(
        boundary.get("future_identity_or_metric_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("source_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False
        and boundary.get("maximum_object_observation_frame") == UPDATE_FRAMES[-1]
        and gate.get("source_outcome_authorized") is False
        and gate.get("sealed_prediction_or_exact_fallback_count")
        == EXPECTED_CASE_COUNT
        and gate.get("technical_failure_count") == 0,
        "V14 outcome boundary is not sealed",
    )
    queue = load_json(queue_path)
    candidates = {
        int(row["queue_rank"]): row for row in queue.get("candidates", [])
    }
    prediction_by_rank = {
        int(row["queue_rank"]): row for row in decision.get("predictions", [])
    }
    expected = protocol["cohort"]["cases"]
    _require(
        set(prediction_by_rank) == {int(row["queue_rank"]) for row in expected},
        "V14 decision names another prediction cohort",
    )
    for record in expected:
        rank = int(record["queue_rank"])
        _require(rank in candidates, "V14 queue rank is missing")
        candidate = candidates[rank]
        prediction = prediction_by_rank[rank]
        _require(
            candidate.get("object_id") == record["object_id"]
            and int(candidate.get("episode_id", -1)) == int(record["episode_id"])
            and candidate.get("category") == record["category"]
            and candidate.get("metadata_sha256") == record["metadata_sha256"]
            and prediction.get("case_hash") == record["case_hash"]
            and prediction.get("object_hash") == record["object_hash"]
            and prediction.get("status") == "exact_baseline_fallback_sealed",
            "V14 case lineage changed",
        )
    return decision


def protocol_case(
    protocol: Mapping[str, Any],
    *,
    queue_rank: int | None = None,
    case: str | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in protocol["cohort"]["cases"]
        if (queue_rank is None or int(row["queue_rank"]) == int(queue_rank))
        and (case is None or row["case"] == case)
    ]
    _require(len(matches) == 1, "case is outside the outcome-sealed cohort")
    return dict(matches[0])


def _source_model_sha256(model: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(model),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_frozen_tactile_model(
    source_result_path: str | Path,
    protocol: Mapping[str, Any],
) -> TactileRegretGuardModel:
    result = load_json(source_result_path)
    source = protocol["method"]["tactile_source_model"]
    _require(
        result.get("artifact_kind") == source["artifact_kind"]
        and result.get("artifact_sha256")
        == canonical_sha256(result, digest_key="artifact_sha256")
        and result.get("artifact_sha256") == source["artifact_sha256"]
        and result.get("all_advancement_gates_passed") is True,
        "tactile source result is incompatible",
    )
    raw_model = result.get("full_source_model_for_future_lock")
    _require(isinstance(raw_model, Mapping), "tactile source model is missing")
    _require(
        _source_model_sha256(raw_model) == source["model_sha256"]
        and raw_model.get("feature_names") == list(TACTILE_REGRET_FEATURE_NAMES),
        "tactile source model changed",
    )
    return TactileRegretGuardModel(
        feature_center=tuple(float(value) for value in raw_model["feature_center"]),
        feature_scale=tuple(float(value) for value in raw_model["feature_scale"]),
        coefficients=tuple(float(value) for value in raw_model["coefficients"]),
        ridge_penalty=float(raw_model["ridge_penalty"]),
        admission_threshold=float(raw_model["admission_threshold"]),
        source_object_count=int(raw_model["source_object_count"]),
        source_row_count=int(raw_model["source_row_count"]),
    )


def validate_v14_physical_manifest(
    manifest: Mapping[str, Any],
    case_record: Mapping[str, Any],
) -> None:
    _require(
        manifest.get("artifact_kind") == V14_PHYSICAL_ARTIFACT_KIND
        and manifest.get("artifact_sha256")
        == namespaced_canonical_sha256(
            manifest,
            namespace=V14_PHYSICAL_NAMESPACE,
            digest_key="artifact_sha256",
        )
        and manifest.get("case_hash") == case_record["case_hash"]
        and manifest.get("object_hash") == case_record["object_hash"]
        and manifest.get("metadata_sha256") == case_record["metadata_sha256"]
        and int(manifest.get("queue_rank", -1)) == int(case_record["queue_rank"])
        and manifest.get("category") == case_record["category"]
        and manifest.get("physical_admitted") is True,
        "V14 physical carrier is incompatible",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("object_observation_frames_used") == [0]
        and boundary.get("prefix_or_future_object_geometry_read") is False
        and boundary.get("prefix_or_future_object_rgb_read") is False
        and boundary.get("prefix_or_future_object_track_read") is False
        and boundary.get("prefix_or_future_tactile_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 physical carrier crossed the outcome boundary",
    )


def stage_v14_physical_backbone(
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    physical_manifest_path: str | Path,
    physical_archive_path: str | Path,
    queue_rank: int,
) -> dict[str, Any]:
    """Convert a sealed V14 carrier to the common prediction interface."""

    protocol = load_outcome_sealed_protocol(protocol_path)
    record = protocol_case(protocol, queue_rank=queue_rank)
    manifest_path = Path(physical_manifest_path).resolve()
    archive_path = Path(physical_archive_path).resolve()
    manifest = load_json(manifest_path)
    validate_v14_physical_manifest(manifest, record)
    _require(
        file_sha256(archive_path)
        == manifest.get("physical_archive", {}).get("file_sha256"),
        "V14 physical archive checksum changed",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        physical = np.asarray(stored["physical_prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_prediction_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    _require(
        physical.shape == persistence.shape
        and physical.shape == (EXPECTED_FRAME_COUNT, len(frame_zero), 3)
        and len(frame_zero) >= 128
        and np.array_equal(physical[0], frame_zero)
        and np.array_equal(persistence[0], frame_zero)
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(persistence)),
        "V14 physical arrays violate the registered backbone contract",
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "backbone output already exists")
    output.mkdir(parents=True)
    staged_archive = output / "prediction.npz"
    archive_arrays = {
        "prediction_m": physical,
        "persistence_m": persistence,
        "frame_zero_points_m": frame_zero,
    }
    np.savez_compressed(staged_archive, **archive_arrays)
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BACKBONE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "case": record["case"],
        "object_id": record["object_id"],
        "episode_id": int(record["episode_id"]),
        "episode_key": (
            f"{record['object_id']}/episode_{int(record['episode_id']):04d}"
        ),
        "category": record["category"],
        "queue_rank": int(record["queue_rank"]),
        "case_hash": record["case_hash"],
        "object_hash": record["object_hash"],
        "prediction_archive": {
            "path": staged_archive.name,
            "file_sha256": file_sha256(staged_archive),
            "array_sha256": {
                name: array_sha256(value)
                for name, value in sorted(archive_arrays.items())
            },
        },
        "parent_v14_physical": {
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_artifact_sha256": manifest["artifact_sha256"],
            "archive_file_sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_tactile_read": False,
            "outcome_manifest_read": False,
            "held_v8_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    seal["result_sha256"] = canonical_sha256(seal, digest_key="result_sha256")
    write_json(output / "prediction_seal.json", seal)
    return seal


def validate_backbone_seal(
    seal: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    case_dir: str | Path | None = None,
) -> None:
    record = protocol_case(protocol, case=str(seal.get("case")))
    _require(
        seal.get("schema_version") == 1
        and seal.get("artifact_kind") == BACKBONE_ARTIFACT_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256") == protocol["config_file_sha256"]
        and seal.get("case_hash") == record["case_hash"]
        and seal.get("object_hash") == record["object_hash"]
        and int(seal.get("queue_rank", -1)) == int(record["queue_rank"])
        and seal.get("result_sha256")
        == canonical_sha256(seal, digest_key="result_sha256"),
        "outcome-sealed backbone seal is incompatible",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("held_v8_read") is False,
        "outcome-sealed backbone crossed its boundary",
    )
    if case_dir is not None:
        root = Path(case_dir).resolve()
        archive = root / Path(str(seal["prediction_archive"]["path"])).name
        _require(
            archive.is_file()
            and file_sha256(archive) == seal["prediction_archive"]["file_sha256"],
            "staged backbone archive changed",
        )


def _tactile_case_features(
    artifact: Mapping[str, Any],
    *,
    case: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _require(
        artifact.get("artifact_kind") == TACTILE_FEATURE_ARTIFACT_KIND
        and artifact.get("artifact_sha256") == canonical_artifact_sha256(artifact),
        "causal tactile feature artifact is incompatible",
    )
    boundary = artifact.get("information_boundary", {})
    _require(
        boundary.get("target_outcomes_read") is False
        and boundary.get("held_v8_read") is False
        and boundary.get("future_tactile_values_used_for_update") is False
        and boundary.get("each_update_uses_tactile_at_or_before_update") is True
        and boundary.get("episode_wide_tactile_normalization_used") is False,
        "tactile feature artifact crossed its boundary",
    )
    rows = [row for row in artifact.get("cases", []) if row.get("case") == case]
    _require(len(rows) == 1, "tactile artifact does not contain exactly one case")
    updates = rows[0].get("updates", [])
    _require(
        [int(row.get("update_frame", -1)) for row in updates]
        == list(UPDATE_FRAMES),
        "tactile update frames changed",
    )
    vectors = np.asarray(
        [
            [float(row[name]) for name in TACTILE_REGRET_FEATURE_NAMES]
            for row in updates
        ],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(vectors)), "tactile features are non-finite")
    return vectors, [dict(row) for row in updates]


def _load_prediction_inputs(
    backbone_dir: Path,
    measurement_dir: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    backbone_path = backbone_dir / "prediction_seal.json"
    backbone = load_json(backbone_path)
    validate_backbone_seal(backbone, protocol=protocol, case_dir=backbone_dir)
    physical_archive = backbone_dir / Path(
        str(backbone["prediction_archive"]["path"])
    ).name
    with np.load(physical_archive, allow_pickle=False) as stored:
        physical_arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    manifest_path = measurement_dir / MANIFEST_FILENAME
    measurement = load_json(manifest_path)
    _require(
        measurement.get("artifact_kind") == "Deform360CausalRawCameraMeasurement"
        and measurement.get("protocol_id") == PROTOCOL_ID
        and measurement.get("case") == backbone["case"]
        and measurement.get("result_sha256")
        == canonical_sha256(measurement, digest_key="result_sha256"),
        "causal camera measurement is incompatible",
    )
    boundary = measurement.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("future_reconstruction_after_frame_zero_read") is False
        and boundary.get("maximum_video_frame_read_by_update")
        == list(UPDATE_FRAMES),
        "camera measurement crossed its outcome boundary",
    )
    measurement_path = measurement_dir / MEASUREMENT_FILENAME
    _require(
        file_sha256(measurement_path)
        == measurement.get("output", {}).get("measurement_archive_sha256"),
        "causal camera measurement archive changed",
    )
    with np.load(measurement_path, allow_pickle=False) as stored:
        measurement_arrays = {
            name: np.asarray(stored[name]).copy() for name in stored.files
        }
    return backbone, physical_arrays, measurement, measurement_arrays


def build_guarded_prediction(
    output_dir: str | Path,
    *,
    repository_root: str | Path,
    protocol_path: str | Path,
    backbone_dir: str | Path,
    measurement_dir: str | Path,
    tactile_feature_path: str | Path,
    source_result_path: str | Path,
) -> dict[str, Any]:
    """Build and seal one target-free tactile-guarded trajectory."""

    protocol = load_outcome_sealed_protocol(
        protocol_path,
        repository_root=repository_root,
    )
    backbone_root = Path(backbone_dir).resolve()
    measurement_root = Path(measurement_dir).resolve()
    backbone, physical_arrays, measurement, measurement_arrays = (
        _load_prediction_inputs(backbone_root, measurement_root, protocol)
    )
    tactile_path = Path(tactile_feature_path).resolve()
    tactile_artifact = load_json(tactile_path)
    tactile_features, tactile_diagnostics = _tactile_case_features(
        tactile_artifact,
        case=backbone["case"],
    )
    model = load_frozen_tactile_model(source_result_path, protocol)
    gate = PairwiseCorrespondenceGateConfig(**protocol["method"]["pairwise_gate"])
    belief = RecursiveRbfBeliefConfig(**protocol["method"]["recursive_rbf"])
    method_report, candidate_arrays = predict_dual_backbone_pairwise_rbf_arrays(
        physical_arrays["prediction_m"],
        physical_arrays["persistence_m"],
        measurement_arrays["measurement_m"],
        measurement_arrays["measurement_visibility"],
        measurement_arrays["measurement_validity"],
        center_ids=measurement_arrays["center_ids"],
        update_frames=UPDATE_FRAMES,
        gate_config=gate,
        belief_config=belief,
    )
    guard_report, guarded = apply_tactile_regret_guard(
        candidate_arrays[SELECTED_BACKBONE_ARM],
        candidate_arrays[DUAL_BACKBONE_ARM],
        tactile_features,
        model,
        update_frames=UPDATE_FRAMES,
    )
    output = Path(output_dir).resolve()
    _require(not output.exists(), "guarded prediction output already exists")
    output.mkdir(parents=True)
    archive_path = output / "guarded_prediction.npz"
    archive_arrays = {
        "physical_prior_m": physical_arrays["prediction_m"],
        "persistence_m": physical_arrays["persistence_m"],
        "selected_baseline_m": candidate_arrays[SELECTED_BACKBONE_ARM],
        "raw_candidate_m": candidate_arrays[DUAL_BACKBONE_ARM],
        "guarded_prediction_m": guarded,
        "center_ids": measurement_arrays["center_ids"],
    }
    np.savez_compressed(archive_path, **archive_arrays)
    accepted_count = sum(
        bool(row["candidate_accepted"]) for row in guard_report["updates"]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360TactileGuardPredictionReportV1",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "case": backbone["case"],
        "object_id": backbone["object_id"],
        "episode_id": int(backbone["episode_id"]),
        "queue_rank": int(backbone["queue_rank"]),
        "case_hash": backbone["case_hash"],
        "object_hash": backbone["object_hash"],
        "status": ORDINARY_STATUS,
        "accepted_update_count": accepted_count,
        "candidate_report": method_report,
        "guard_report": guard_report,
        "tactile_update_diagnostics": tactile_diagnostics,
        "gate_config": asdict(gate),
        "belief_config": asdict(belief),
        "prediction_archive": {
            "path": archive_path.name,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(value)
                for name, value in sorted(archive_arrays.items())
            },
        },
        "inputs": {
            "backbone_seal": {
                "path": str(backbone_root / "prediction_seal.json"),
                "file_sha256": file_sha256(backbone_root / "prediction_seal.json"),
                "result_sha256": backbone["result_sha256"],
            },
            "measurement_manifest": {
                "path": str(measurement_root / MANIFEST_FILENAME),
                "file_sha256": file_sha256(measurement_root / MANIFEST_FILENAME),
                "result_sha256": measurement["result_sha256"],
            },
            "tactile_features": {
                "path": str(tactile_path),
                "file_sha256": file_sha256(tactile_path),
                "artifact_sha256": tactile_artifact["artifact_sha256"],
            },
            "source_tactile_result": {
                "path": str(Path(source_result_path).resolve()),
                "artifact_sha256": protocol["method"]["tactile_source_model"][
                    "artifact_sha256"
                ],
            },
        },
        "information_boundary": {
            "object_rgb_frames_read": [0, UPDATE_FRAMES[-1]],
            "causal_rgb_prefix_updates_used": list(UPDATE_FRAMES),
            "causal_tactile_prefix_updates_used": list(UPDATE_FRAMES),
            "future_target_read": False,
            "future_object_observation_read": False,
            "future_tactile_values_used": False,
            "outcome_manifest_read": False,
            "held_v8_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    report["result_sha256"] = canonical_sha256(report, digest_key="result_sha256")
    report_path = write_json(output / "guarded_prediction_report.json", report)
    seal = _build_prediction_seal(
        protocol,
        report,
        report_path=report_path,
        archive_path=archive_path,
    )
    write_json(output / "prediction_seal.json", seal)
    return seal


def _build_prediction_seal(
    protocol: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    report_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "case": report["case"],
        "object_id": report["object_id"],
        "episode_id": int(report["episode_id"]),
        "queue_rank": int(report["queue_rank"]),
        "case_hash": report["case_hash"],
        "object_hash": report["object_hash"],
        "status": report["status"],
        "accepted_update_count": int(report.get("accepted_update_count", 0)),
        "prediction_archive": {
            "path": archive_path.name,
            "file_sha256": file_sha256(archive_path),
        },
        "prediction_report": {
            "path": report_path.name,
            "file_sha256": file_sha256(report_path),
            "result_sha256": report["result_sha256"],
        },
        "information_boundary": dict(report["information_boundary"]),
    }
    seal["result_sha256"] = canonical_sha256(seal, digest_key="result_sha256")
    return seal


def build_technical_fallback(
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    backbone_dir: str | Path,
    failure_stage: str,
    failure_type: str,
    failure_message: str,
    failed_input_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal an explicit persistence fallback without hiding the failure."""

    protocol = load_outcome_sealed_protocol(protocol_path)
    backbone_root = Path(backbone_dir).resolve()
    backbone = load_json(backbone_root / "prediction_seal.json")
    validate_backbone_seal(backbone, protocol=protocol, case_dir=backbone_root)
    archive = backbone_root / Path(str(backbone["prediction_archive"]["path"])).name
    with np.load(archive, allow_pickle=False) as stored:
        persistence = np.asarray(stored["persistence_m"]).copy()
    output = Path(output_dir).resolve()
    _require(not output.exists(), "technical fallback output already exists")
    output.mkdir(parents=True)
    archive_path = output / "guarded_prediction.npz"
    archive_arrays = {
        "persistence_m": persistence,
        "selected_baseline_m": persistence.copy(),
        "raw_candidate_m": persistence.copy(),
        "guarded_prediction_m": persistence.copy(),
    }
    np.savez_compressed(archive_path, **archive_arrays)
    _require(
        np.array_equal(
            archive_arrays["guarded_prediction_m"],
            archive_arrays["selected_baseline_m"],
        ),
        "technical fallback changed persistence",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360TactileGuardPredictionReportV1",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "case": backbone["case"],
        "object_id": backbone["object_id"],
        "episode_id": int(backbone["episode_id"]),
        "queue_rank": int(backbone["queue_rank"]),
        "case_hash": backbone["case_hash"],
        "object_hash": backbone["object_hash"],
        "status": TECHNICAL_FALLBACK_STATUS,
        "accepted_update_count": 0,
        "failure": {
            "stage": str(failure_stage),
            "type": str(failure_type),
            "message": str(failure_message),
            "input_sha256": dict(failed_input_sha256 or {}),
        },
        "bit_exact_persistence_fallback": True,
        "prediction_archive": {
            "path": archive_path.name,
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(value)
                for name, value in sorted(archive_arrays.items())
            },
        },
        "information_boundary": {
            "object_rgb_frames_read": [],
            "causal_rgb_prefix_updates_used": [],
            "causal_tactile_prefix_updates_used": [],
            "future_target_read": False,
            "future_object_observation_read": False,
            "future_tactile_values_used": False,
            "outcome_manifest_read": False,
            "held_v8_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    report["result_sha256"] = canonical_sha256(report, digest_key="result_sha256")
    report_path = write_json(output / "guarded_prediction_report.json", report)
    seal = _build_prediction_seal(
        protocol,
        report,
        report_path=report_path,
        archive_path=archive_path,
    )
    write_json(output / "prediction_seal.json", seal)
    return seal


def validate_prediction_seal(
    seal: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    prediction_dir: str | Path | None = None,
) -> None:
    record = protocol_case(protocol, case=str(seal.get("case")))
    _require(
        seal.get("schema_version") == 1
        and seal.get("artifact_kind") == PREDICTION_ARTIFACT_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("protocol_config_sha256") == protocol["config_file_sha256"]
        and seal.get("claim_label") == CLAIM_LABEL
        and seal.get("case_hash") == record["case_hash"]
        and seal.get("object_hash") == record["object_hash"]
        and int(seal.get("queue_rank", -1)) == int(record["queue_rank"])
        and seal.get("status") in {ORDINARY_STATUS, TECHNICAL_FALLBACK_STATUS}
        and seal.get("result_sha256")
        == canonical_sha256(seal, digest_key="result_sha256"),
        "guarded prediction seal is incompatible",
    )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("future_target_read") is False
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_tactile_values_used") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("held_v8_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True,
        "guarded prediction crossed its outcome boundary",
    )
    if prediction_dir is not None:
        root = Path(prediction_dir).resolve()
        archive = root / Path(str(seal["prediction_archive"]["path"])).name
        report_path = root / Path(str(seal["prediction_report"]["path"])).name
        _require(
            archive.is_file()
            and report_path.is_file()
            and file_sha256(archive) == seal["prediction_archive"]["file_sha256"]
            and file_sha256(report_path) == seal["prediction_report"]["file_sha256"],
            "guarded prediction files changed",
        )
        report = load_json(report_path)
        _require(
            report.get("result_sha256")
            == canonical_sha256(report, digest_key="result_sha256")
            and report.get("result_sha256")
            == seal["prediction_report"]["result_sha256"]
            and report.get("status") == seal["status"],
            "guarded prediction report changed",
        )
        with np.load(archive, allow_pickle=False) as stored:
            guarded = np.asarray(stored["guarded_prediction_m"])
            baseline = np.asarray(stored["selected_baseline_m"])
        if seal["status"] == TECHNICAL_FALLBACK_STATUS:
            _require(
                np.array_equal(guarded, baseline),
                "technical failure did not preserve its exact baseline",
            )


def write_unsealable_case(
    path: str | Path,
    *,
    protocol_path: str | Path,
    case: str,
    failure_stage: str,
    failure_type: str,
    failure_message: str,
) -> dict[str, Any]:
    protocol = load_outcome_sealed_protocol(protocol_path)
    record = protocol_case(protocol, case=case)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": UNSEALABLE_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "case": record["case"],
        "queue_rank": int(record["queue_rank"]),
        "case_hash": record["case_hash"],
        "object_hash": record["object_hash"],
        "status": UNSEALABLE_STATUS,
        "failure": {
            "stage": str(failure_stage),
            "type": str(failure_type),
            "message": str(failure_message),
        },
        "replacement_authorized": False,
        "future_outcome_read": False,
        "held_v8_read": False,
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(path, payload)
    return payload


def build_prediction_barrier(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    prediction_root: str | Path,
) -> dict[str, Any]:
    """Seal complete accounting; any unsealable case keeps the barrier closed."""

    protocol = load_outcome_sealed_protocol(protocol_path)
    root = Path(prediction_root).resolve()
    records = []
    ordinary = technical = unsealable = 0
    for case_record in protocol["cohort"]["cases"]:
        case = str(case_record["case"])
        case_dir = root / case
        seal_path = case_dir / "prediction_seal.json"
        failure_path = case_dir / "unsealable.json"
        _require(
            seal_path.is_file() ^ failure_path.is_file(),
            f"case must have exactly one disposition: {case}",
        )
        if seal_path.is_file():
            seal = load_json(seal_path)
            validate_prediction_seal(
                seal,
                protocol=protocol,
                prediction_dir=case_dir,
            )
            status = str(seal["status"])
            ordinary += int(status == ORDINARY_STATUS)
            technical += int(status == TECHNICAL_FALLBACK_STATUS)
            records.append(
                {
                    "case": case,
                    "status": status,
                    "file_sha256": file_sha256(seal_path),
                    "result_sha256": seal["result_sha256"],
                }
            )
        else:
            failure = load_json(failure_path)
            _require(
                failure.get("artifact_kind") == UNSEALABLE_ARTIFACT_KIND
                and failure.get("protocol_id") == PROTOCOL_ID
                and failure.get("protocol_config_sha256")
                == protocol["config_file_sha256"]
                and failure.get("case") == case
                and failure.get("status") == UNSEALABLE_STATUS
                and failure.get("future_outcome_read") is False
                and failure.get("held_v8_read") is False
                and failure.get("replacement_authorized") is False
                and failure.get("result_sha256")
                == canonical_sha256(failure, digest_key="result_sha256"),
                "unsealable disposition is incompatible",
            )
            unsealable += 1
            records.append(
                {
                    "case": case,
                    "status": UNSEALABLE_STATUS,
                    "file_sha256": file_sha256(failure_path),
                    "result_sha256": failure["result_sha256"],
                }
            )
    barrier_passed = (
        ordinary + technical == EXPECTED_CASE_COUNT and unsealable == 0
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BARRIER_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_file_sha256"],
        "claim_label": CLAIM_LABEL,
        "total_locked_case_count": EXPECTED_CASE_COUNT,
        "ordinary_successful_prediction_count": ordinary,
        "retained_technical_failure_count": technical,
        "unsealable_case_count": unsealable,
        "replacement_count": 0,
        "records": records,
        "barrier_passed": barrier_passed,
        "outcome_authorized": barrier_passed,
        "information_boundary": {
            "future_target_read": False,
            "future_object_observation_read": False,
            "outcome_manifest_read": False,
            "held_v8_read": False,
            "all_dispositions_hashed_before_outcome": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    destination = Path(output_path).resolve()
    _require(not destination.exists(), "prediction barrier already exists")
    write_json(destination, payload)
    return payload


def validate_prediction_barrier(
    barrier_path: str | Path,
    *,
    protocol_path: str | Path,
    prediction_root: str | Path,
    require_passed: bool = True,
) -> dict[str, Any]:
    protocol = load_outcome_sealed_protocol(protocol_path)
    barrier = load_json(barrier_path)
    _require(
        barrier.get("artifact_kind") == BARRIER_ARTIFACT_KIND
        and barrier.get("protocol_id") == PROTOCOL_ID
        and barrier.get("protocol_config_sha256") == protocol["config_file_sha256"]
        and barrier.get("result_sha256")
        == canonical_sha256(barrier, digest_key="result_sha256")
        and barrier.get("total_locked_case_count") == EXPECTED_CASE_COUNT
        and barrier.get("replacement_count") == 0,
        "prediction barrier is incompatible",
    )
    if require_passed:
        _require(
            barrier.get("barrier_passed") is True
            and barrier.get("outcome_authorized") is True
            and barrier.get("unsealable_case_count") == 0,
            "prediction barrier did not authorize outcomes",
        )
    root = Path(prediction_root).resolve()
    for record in barrier.get("records", []):
        case_dir = root / str(record["case"])
        path = (
            case_dir / "unsealable.json"
            if record["status"] == UNSEALABLE_STATUS
            else case_dir / "prediction_seal.json"
        )
        _require(
            path.is_file()
            and file_sha256(path) == record["file_sha256"]
            and load_json(path).get("result_sha256") == record["result_sha256"],
            f"disposition changed after barrier: {record['case']}",
        )
    return barrier


__all__ = [
    "BACKBONE_ARTIFACT_KIND",
    "BARRIER_ARTIFACT_KIND",
    "CLAIM_LABEL",
    "EXPECTED_CASE_COUNT",
    "ORDINARY_STATUS",
    "PREDICTION_ARTIFACT_KIND",
    "PROTOCOL_ARTIFACT_KIND",
    "PROTOCOL_ID",
    "TECHNICAL_FALLBACK_STATUS",
    "UNSEALABLE_ARTIFACT_KIND",
    "UPDATE_FRAMES",
    "array_sha256",
    "build_guarded_prediction",
    "build_prediction_barrier",
    "build_technical_fallback",
    "canonical_sha256",
    "file_sha256",
    "git_blob_oid",
    "load_frozen_tactile_model",
    "load_outcome_sealed_protocol",
    "namespaced_canonical_sha256",
    "protocol_case",
    "stage_v14_physical_backbone",
    "validate_backbone_seal",
    "validate_prediction_barrier",
    "validate_prediction_seal",
    "validate_v14_custody",
    "write_json",
    "write_unsealable_case",
]
