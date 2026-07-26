"""Frozen artifacts for the fresh-object Deform360 pairwise-belief study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .deform360_fresh_source_lock import validate_fresh_cohort_lock


PROTOCOL_ID = "deform360-fresh-pairwise-belief-v1"
BACKBONE_SEAL_KIND = "Deform360FreshPairwiseBackboneSeal"
BELIEF_SEAL_KIND = "Deform360FreshPairwiseBeliefPredictionSeal"
COMPLETENESS_BARRIER_KIND = "Deform360FreshPairwiseCompletenessBarrier"
EXPECTED_FRAME_COUNT = 76
UPDATE_FRAMES = (19, 38, 57)
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


def array_sha256(value: Any) -> str:
    import numpy as np

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


def load_fresh_pairwise_protocol(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the frozen contract and optionally verify its method source tree."""

    protocol_path = Path(path).resolve()
    payload = load_json(protocol_path)
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    cohort = payload.get("cohort", {})
    _require(
        cohort.get("lock_artifact_kind") == "Deform360FreshObjectCohortLock"
        and _valid_digest(cohort.get("lock_sha256"))
        and cohort.get("case_count") == 12
        and cohort.get("physical_object_count") == 12
        and cohort.get("replacement_policy") == "none",
        "cohort contract changed",
    )
    method = payload.get("method", {})
    _require(
        isinstance(method.get("development_commit"), str)
        and bool(_HEX40.fullmatch(method["development_commit"]))
        and method.get("selected_arm")
        == "raw_selected_backbone_full_blend_rbf_pairwise_clique"
        and method.get("update_frames") == list(UPDATE_FRAMES)
        and method.get("rejection_policy") == "bit-exact selected raw backbone",
        "method contract changed",
    )
    source_hashes = method.get("source_sha256")
    _require(
        isinstance(source_hashes, Mapping)
        and source_hashes
        and all(_valid_digest(value) for value in source_hashes.values()),
        "method source hashes are malformed",
    )
    physical = payload.get("physical_backbone", {})
    _require(
        physical.get("canonical_observed_node_count") == 384
        and physical.get("minimum_observed_node_count") == 128
        and physical.get("length_scale_m") == 0.12
        and physical.get("action_response") == 0.9
        and physical.get("autonomous_drift_response") == 0.0,
        "physical-backbone contract changed",
    )
    action_window = physical.get("known_action_window", {})
    _require(
        action_window.get("staged_frame_count") == 81
        and action_window.get("prediction_frame_range_half_open") == [0, 76]
        and action_window.get("tracking_tail_frames_skipped") == 5,
        "known-action window contract changed",
    )
    custody = payload.get("custody", {})
    _require(
        custody.get("outcome_before_barrier") is False
        and custody.get("technical_failure_is_model_prediction") is False
        and custody.get("failed_case_replacement") is False,
        "custody contract changed",
    )
    if repository_root is not None:
        root = Path(repository_root).resolve()
        observed = {
            name: file_sha256(root / "src" / "bayesian_phystwin" / name)
            for name in source_hashes
        }
        _require(observed == dict(source_hashes), "frozen method source tree changed")
    payload["config_file_sha256"] = file_sha256(protocol_path)
    return payload


def load_bound_cohort(
    cohort_path: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    cohort = load_json(cohort_path)
    validate_fresh_cohort_lock(cohort)
    _require(
        cohort["cohort_lock_sha256"] == protocol["cohort"]["lock_sha256"],
        "protocol refers to another cohort lock",
    )
    _require(
        len(cohort["cases"]) == protocol["cohort"]["case_count"],
        "cohort size changed",
    )
    return cohort


def fresh_case_record(
    cohort: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    matches = [
        case
        for case in cohort["cases"]
        if case["object_id"] == object_id
        and int(case["episode_id"]) == int(episode_id)
    ]
    _require(len(matches) == 1, "case is outside the frozen fresh cohort")
    return dict(matches[0])


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file() and not source.is_symlink(), f"missing file: {source}")
    return {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "size_bytes": source.stat().st_size,
    }


def build_backbone_seal(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    cohort_path: str | Path,
    case_record: Mapping[str, Any],
    admission_path: str | Path,
    prediction_archive: str | Path,
    physical_manifest: str | Path,
) -> dict[str, Any]:
    """Seal a frame-zero physical/persistence backbone before RGB tracking."""

    protocol = load_fresh_pairwise_protocol(protocol_path)
    cohort = load_bound_cohort(cohort_path, protocol)
    expected = fresh_case_record(
        cohort,
        object_id=str(case_record["object_id"]),
        episode_id=int(case_record["episode_id"]),
    )
    _require(dict(case_record) == expected, "case record differs from cohort lock")
    admission = load_json(admission_path)
    _require(
        admission.get("accepted") is True
        and admission.get("admission_sha256") == expected["admission_sha256"],
        "source admission differs from cohort lock",
    )
    manifest = load_json(physical_manifest)
    _require(
        manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("case") == expected["case"]
        and manifest.get("result_sha256")
        == canonical_sha256(manifest, digest_key="result_sha256")
        and manifest.get("passed") is True,
        "physical manifest is incompatible",
    )
    archive_record = _bound_file(prediction_archive)
    _require(
        manifest.get("physical_prediction_archive", {}).get("file_sha256")
        == archive_record["file_sha256"],
        "physical manifest refers to another archive",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BACKBONE_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": file_sha256(protocol_path),
        "cohort_lock_sha256": cohort["cohort_lock_sha256"],
        "case": expected["case"],
        "object_id": expected["object_id"],
        "episode_id": int(expected["episode_id"]),
        "episode_key": f"{expected['object_id']}/episode_{int(expected['episode_id']):04d}",
        "category": expected["category"],
        "admission_sha256": expected["admission_sha256"],
        "prediction_archive": archive_record,
        "physical_manifest": _bound_file(physical_manifest),
        "inputs": {
            "protocol": _bound_file(protocol_path),
            "cohort_lock": _bound_file(cohort_path),
            "source_admission": _bound_file(admission_path),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "outcome_manifest_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(output_path, payload)
    return payload


def validate_backbone_seal(
    seal: Mapping[str, Any],
    *,
    protocol_config_sha256: str | None = None,
    cohort_lock_sha256: str | None = None,
) -> None:
    _require(
        seal.get("schema_version") == 1
        and seal.get("artifact_kind") == BACKBONE_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("result_sha256")
        == canonical_sha256(seal, digest_key="result_sha256"),
        "fresh backbone seal is incompatible",
    )
    if protocol_config_sha256 is not None:
        _require(
            seal.get("protocol_config_sha256") == protocol_config_sha256,
            "backbone seal uses another protocol config",
        )
    if cohort_lock_sha256 is not None:
        _require(
            seal.get("cohort_lock_sha256") == cohort_lock_sha256,
            "backbone seal uses another cohort",
        )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True,
        "fresh backbone crossed its information boundary",
    )


def build_belief_prediction_seal(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    cohort_path: str | Path,
    backbone_seal_path: str | Path,
    measurement_manifest_path: str | Path,
    prediction_archive_path: str | Path,
    prediction_report_path: str | Path,
) -> dict[str, Any]:
    """Bind one target-free online-belief prediction after causal RGB tracking."""

    protocol = load_fresh_pairwise_protocol(protocol_path)
    cohort = load_bound_cohort(cohort_path, protocol)
    backbone = load_json(backbone_seal_path)
    validate_backbone_seal(
        backbone,
        protocol_config_sha256=file_sha256(protocol_path),
        cohort_lock_sha256=cohort["cohort_lock_sha256"],
    )
    record = fresh_case_record(
        cohort,
        object_id=str(backbone["object_id"]),
        episode_id=int(backbone["episode_id"]),
    )
    measurement = load_json(measurement_manifest_path)
    _require(
        measurement.get("artifact_kind") == "Deform360CausalRawCameraMeasurement"
        and measurement.get("protocol_id") == PROTOCOL_ID
        and measurement.get("case") == record["case"]
        and measurement.get("result_sha256")
        == canonical_sha256(measurement, digest_key="result_sha256"),
        "causal measurement manifest is incompatible",
    )
    measurement_boundary = measurement.get("information_boundary", {})
    _require(
        measurement_boundary.get("target_data_read") is False
        and measurement_boundary.get("outcome_manifest_read") is False
        and measurement_boundary.get("future_reconstruction_after_frame_zero_read")
        is False,
        "measurement crossed the prediction boundary",
    )
    report = load_json(prediction_report_path)
    _require(
        report.get("artifact_kind") == "Deform360FreshPairwiseBeliefPrediction"
        and report.get("protocol_id") == PROTOCOL_ID
        and report.get("case") == record["case"]
        and report.get("result_sha256")
        == canonical_sha256(report, digest_key="result_sha256"),
        "belief prediction report is incompatible",
    )
    report_boundary = report.get("information_boundary", {})
    _require(
        report_boundary.get("future_target_read") is False
        and report_boundary.get("outcome_manifest_read") is False,
        "belief prediction crossed the outcome boundary",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": BELIEF_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": file_sha256(protocol_path),
        "cohort_lock_sha256": cohort["cohort_lock_sha256"],
        "case": record["case"],
        "object_id": record["object_id"],
        "episode_id": int(record["episode_id"]),
        "episode_key": (
            f"{record['object_id']}/episode_{int(record['episode_id']):04d}"
        ),
        "category": record["category"],
        "selected_arm": protocol["method"]["selected_arm"],
        "prediction_archive": _bound_file(prediction_archive_path),
        "prediction_report": _bound_file(prediction_report_path),
        "inputs": {
            "protocol": _bound_file(protocol_path),
            "cohort_lock": _bound_file(cohort_path),
            "backbone_seal": _bound_file(backbone_seal_path),
            "measurement_manifest": _bound_file(measurement_manifest_path),
        },
        "information_boundary": {
            "causal_rgb_prefix_updates_used": list(UPDATE_FRAMES),
            "future_target_read": False,
            "outcome_manifest_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(output_path, payload)
    return payload


def validate_belief_prediction_seal(
    seal: Mapping[str, Any],
    *,
    protocol_config_sha256: str | None = None,
    cohort_lock_sha256: str | None = None,
) -> None:
    _require(
        seal.get("schema_version") == 1
        and seal.get("artifact_kind") == BELIEF_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID
        and seal.get("result_sha256")
        == canonical_sha256(seal, digest_key="result_sha256"),
        "fresh belief prediction seal is incompatible",
    )
    if protocol_config_sha256 is not None:
        _require(
            seal.get("protocol_config_sha256") == protocol_config_sha256,
            "belief seal uses another protocol config",
        )
    if cohort_lock_sha256 is not None:
        _require(
            seal.get("cohort_lock_sha256") == cohort_lock_sha256,
            "belief seal uses another cohort",
        )
    boundary = seal.get("information_boundary", {})
    _require(
        boundary.get("causal_rgb_prefix_updates_used") == list(UPDATE_FRAMES)
        and boundary.get("future_target_read") is False
        and boundary.get("outcome_manifest_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True,
        "belief prediction seal crossed its information boundary",
    )


def build_completeness_barrier(
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    cohort_path: str | Path,
    prediction_root: str | Path,
) -> dict[str, Any]:
    """Require one valid ordinary prediction seal for every locked object."""

    protocol = load_fresh_pairwise_protocol(protocol_path)
    cohort = load_bound_cohort(cohort_path, protocol)
    root = Path(prediction_root).resolve()
    expected = tuple(str(case["case"]) for case in cohort["cases"])
    records = []
    for case in expected:
        path = root / case / "belief_prediction_seal.json"
        _require(path.is_file(), f"missing belief prediction seal: {case}")
        seal = load_json(path)
        validate_belief_prediction_seal(
            seal,
            protocol_config_sha256=file_sha256(protocol_path),
            cohort_lock_sha256=cohort["cohort_lock_sha256"],
        )
        _require(seal.get("case") == case, "belief seal case changed")
        records.append(
            {
                "case": case,
                "seal_file_sha256": file_sha256(path),
                "seal_result_sha256": seal["result_sha256"],
            }
        )
    observed = sorted(
        path.parent.name
        for path in root.glob("*/belief_prediction_seal.json")
        if path.is_file()
    )
    _require(observed == sorted(expected), "prediction root contains another cohort")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": COMPLETENESS_BARRIER_KIND,
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": file_sha256(protocol_path),
        "cohort_lock_sha256": cohort["cohort_lock_sha256"],
        "expected_case_count": len(expected),
        "ordinary_prediction_count": len(records),
        "retained_technical_failure_count": 0,
        "unsealable_case_count": 0,
        "replacement_count": 0,
        "records": records,
        "barrier_passed": True,
        "information_boundary": {
            "future_target_read": False,
            "outcome_manifest_read": False,
            "all_predictions_hashed_before_outcome": True,
        },
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(output_path, payload)
    return payload


__all__ = [
    "BACKBONE_SEAL_KIND",
    "BELIEF_SEAL_KIND",
    "COMPLETENESS_BARRIER_KIND",
    "EXPECTED_FRAME_COUNT",
    "PROTOCOL_ID",
    "UPDATE_FRAMES",
    "array_sha256",
    "build_backbone_seal",
    "build_belief_prediction_seal",
    "build_completeness_barrier",
    "canonical_sha256",
    "file_sha256",
    "fresh_case_record",
    "load_bound_cohort",
    "load_fresh_pairwise_protocol",
    "load_json",
    "validate_backbone_seal",
    "validate_belief_prediction_seal",
    "write_json",
]
