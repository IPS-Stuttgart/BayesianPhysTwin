from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_tactile_features import (
    canonical_artifact_sha256,
)
from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (
    BACKBONE_ARTIFACT_KIND,
    CLAIM_LABEL,
    ORDINARY_STATUS,
    PROTOCOL_ARTIFACT_KIND,
    PROTOCOL_ID,
    TECHNICAL_FALLBACK_STATUS,
    V14_DECISION_NAMESPACE,
    V14_PHYSICAL_NAMESPACE,
    build_guarded_prediction,
    build_prediction_barrier,
    build_technical_fallback,
    canonical_sha256,
    canonical_text_sha256,
    file_sha256,
    git_blob_oid,
    load_frozen_tactile_model,
    load_outcome_sealed_protocol,
    namespaced_canonical_sha256,
    stage_v14_physical_backbone,
    validate_prediction_barrier,
    validate_prediction_seal,
    validate_v14_custody,
    write_unsealable_case,
)
from bayesian_phystwin.deform360_tactile_regret_guard import (
    TACTILE_REGRET_FEATURE_NAMES,
)
from bayesian_phystwin.phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
)
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig

REPO = Path(__file__).resolve().parents[1]
MODULE = "src/bayesian_phystwin/deform360_tactile_guard_outcome_sealed.py"
TACTILE_MODULE = "src/bayesian_phystwin/deform360_tactile_regret_guard.py"
PAIRWISE_MODULE = "src/bayesian_phystwin/deform360_pairwise_regret_guard.py"
REGISTERED_PROTOCOL = (
    REPO / "configs/sota/deform360_tactile_guard_outcome_sealed_v1.json"
)
REGISTERED_TACTILE_MANIFEST = (
    REPO
    / "configs/sota/deform360_tactile_guard_outcome_sealed_v1_tactile_manifest.json"
)
REGISTERED_SOURCE_RESULT = (
    REPO
    / "results/sota/diagnostics/deform360_tactile_regret_guard_source_v1/result.json"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _model_payload() -> dict[str, object]:
    count = len(TACTILE_REGRET_FEATURE_NAMES)
    return {
        "admission_threshold": 0.7,
        "coefficients": [1.0] + [0.0] * count,
        "feature_center": [0.0] * count,
        "feature_scale": [1.0] * count,
        "feature_names": list(TACTILE_REGRET_FEATURE_NAMES),
        "ridge_penalty": 10.0,
        "score_semantics": "linear source-fitted benefit score; not a probability",
        "source_object_count": 17,
        "source_row_count": 117,
    }


def _model_sha256(model: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            model,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _case_records() -> list[dict[str, object]]:
    return [
        {
            "case": f"object-{rank:02d}-ep0000",
            "object_id": f"object-{rank:02d}",
            "episode_id": 0,
            "queue_rank": rank,
            "category": ("sheet" if rank % 2 else "complex"),
            "case_hash": hashlib.sha256(f"case-{rank}".encode()).hexdigest(),
            "object_hash": hashlib.sha256(f"object-{rank}".encode()).hexdigest(),
            "metadata_sha256": hashlib.sha256(f"meta-{rank}".encode()).hexdigest(),
        }
        for rank in range(1, 13)
    ]


def _write_custody_files(
    root: Path,
    cases: list[dict[str, object]],
) -> tuple[Path, Path, dict[str, object]]:
    queue = {
        "artifact_kind": "Deform360FreshSourceStagingQueue",
        "candidates": [
            {
                "object_id": row["object_id"],
                "episode_id": row["episode_id"],
                "queue_rank": row["queue_rank"],
                "category": row["category"],
                "metadata_sha256": row["metadata_sha256"],
            }
            for row in cases
        ],
    }
    queue_path = root / "queue.json"
    _write_json(queue_path, queue)
    decision: dict[str, object] = {
        "artifact_kind": "Deform360CausalResponseDirectDepthSourceDecisionV14",
        "decision": "close_v14_without_source_outcome_reveal",
        "information_boundary": {
            "future_identity_or_metric_read": False,
            "future_object_observation_read": False,
            "source_outcome_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
            "maximum_object_observation_frame": 57,
        },
        "outcome_blind_gate": {
            "source_outcome_authorized": False,
            "sealed_prediction_or_exact_fallback_count": 12,
            "technical_failure_count": 0,
        },
        "predictions": [
            {
                "queue_rank": row["queue_rank"],
                "case_hash": row["case_hash"],
                "object_hash": row["object_hash"],
                "status": "exact_baseline_fallback_sealed",
            }
            for row in cases
        ],
    }
    decision["artifact_sha256"] = namespaced_canonical_sha256(
        decision,
        namespace=V14_DECISION_NAMESPACE,
        digest_key="artifact_sha256",
    )
    decision_path = root / "decision.json"
    _write_json(decision_path, decision)
    return decision_path, queue_path, decision


def _write_protocol_bundle(root: Path) -> dict[str, Path]:
    cases = _case_records()
    decision_path, queue_path, decision = _write_custody_files(root, cases)
    model = _model_payload()
    source_result: dict[str, object] = {
        "artifact_kind": "Deform360TactileRegretGuardSourceDiagnostic",
        "all_advancement_gates_passed": True,
        "full_source_model_for_future_lock": model,
    }
    source_result["artifact_sha256"] = canonical_sha256(
        source_result, digest_key="artifact_sha256"
    )
    source_result_path = root / "source_result.json"
    _write_json(source_result_path, source_result)
    protocol: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "claim_boundary": {
            "label": CLAIM_LABEL,
            "fresh_object_identity_confirmation": False,
            "identity_known_before_lock": True,
            "future_outcomes_sealed_before_prediction": True,
            "state_of_the_art_confirmation": False,
        },
        "cohort": {
            "case_count": 12,
            "physical_object_count": 12,
            "replacement_policy": "none",
            "technical_failures_remain_in_denominator": True,
            "cases": cases,
            "v14_custody": {
                "repository_revision": "a" * 40,
                "source_decision_git_blob_oid": git_blob_oid(decision_path),
                "staging_queue_git_blob_oid": git_blob_oid(queue_path),
                "source_decision_artifact_sha256": decision["artifact_sha256"],
                "source_outcome_authorized": False,
                "future_identity_or_metric_read": False,
                "future_object_observation_read": False,
                "maximum_object_observation_frame": 57,
            },
        },
        "method": {
            "candidate": "dual_backbone_pairwise_consensus_rbf",
            "baseline": "selected_raw_backbone",
            "guard": "causal_tactile_regret_guard",
            "update_frames": [19, 38, 57],
            "technical_failure_fallback": "bit-exact persistence",
            "target_fitting": False,
            "pairwise_gate": asdict(PairwiseCorrespondenceGateConfig()),
            "recursive_rbf": asdict(
                RecursiveRbfBeliefConfig(
                    length_scale_fraction=0.10,
                    local_blend=1.0,
                )
            ),
            "tactile_source_model": {
                "artifact_kind": source_result["artifact_kind"],
                "artifact_sha256": source_result["artifact_sha256"],
                "model_sha256": _model_sha256(model),
                "feature_names": list(TACTILE_REGRET_FEATURE_NAMES),
            },
            "source_text_sha256": {
                MODULE: canonical_text_sha256(REPO / MODULE),
                TACTILE_MODULE: canonical_text_sha256(REPO / TACTILE_MODULE),
                PAIRWISE_MODULE: canonical_text_sha256(REPO / PAIRWISE_MODULE),
            },
        },
        "advancement_gates": {
            "minimum_joint_case_wins": 2,
            "maximum_joint_case_regressions": 0,
            "minimum_identity_improvement_percent": 1.0,
            "minimum_chamfer_improvement_percent": 1.0,
            "object_cluster_ci_upper_must_be_nonpositive": True,
        },
        "custody": {
            "outcome_before_barrier": False,
            "failed_case_replacement": False,
            "unsealable_case_blocks_outcome": True,
            "future_tactile_values_used": False,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(
        protocol, digest_key="protocol_sha256"
    )
    protocol_path = root / "protocol.json"
    _write_json(protocol_path, protocol)
    return {
        "protocol": protocol_path,
        "decision": decision_path,
        "queue": queue_path,
        "source_result": source_result_path,
    }


def _write_physical_carrier(
    root: Path,
    case: dict[str, object],
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    frame_zero = np.zeros((128, 3), dtype=np.float32)
    frame_zero[:, 0] = np.linspace(0.0, 0.2, len(frame_zero))
    physical = np.repeat(frame_zero[None], 76, axis=0)
    persistence = physical.copy()
    physical[:, :, 1] += np.linspace(0.0, 0.02, 76)[:, None]
    archive = root / "v14_physical.npz"
    np.savez_compressed(
        archive,
        physical_prediction_m=physical,
        persistence_prediction_m=persistence,
        frame_zero_points_m=frame_zero,
    )
    manifest: dict[str, object] = {
        "artifact_kind": "Deform360CausalResponseDirectDepthPhysicalBackboneV14",
        "case_hash": case["case_hash"],
        "object_hash": case["object_hash"],
        "metadata_sha256": case["metadata_sha256"],
        "queue_rank": case["queue_rank"],
        "category": case["category"],
        "physical_admitted": True,
        "physical_archive": {"file_sha256": file_sha256(archive)},
        "information_boundary": {
            "identity_or_metric_outcome_read": False,
            "object_observation_frames_used": [0],
            "prefix_or_future_object_geometry_read": False,
            "prefix_or_future_object_rgb_read": False,
            "prefix_or_future_object_track_read": False,
            "prefix_or_future_tactile_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    manifest["artifact_sha256"] = namespaced_canonical_sha256(
        manifest,
        namespace=V14_PHYSICAL_NAMESPACE,
        digest_key="artifact_sha256",
    )
    manifest_path = root / "v14_physical.json"
    _write_json(manifest_path, manifest)
    return manifest_path, archive


def _stage_backbone(root: Path, bundle: dict[str, Path], rank: int = 1) -> Path:
    protocol = load_outcome_sealed_protocol(bundle["protocol"])
    case = protocol["cohort"]["cases"][rank - 1]
    manifest, archive = _write_physical_carrier(root, case)
    output = root / f"backbone-{rank}"
    seal = stage_v14_physical_backbone(
        output,
        protocol_path=bundle["protocol"],
        physical_manifest_path=manifest,
        physical_archive_path=archive,
        queue_rank=rank,
    )
    assert seal["artifact_kind"] == BACKBONE_ARTIFACT_KIND
    return output


def _write_measurement(root: Path, backbone_dir: Path) -> Path:
    seal = json.loads((backbone_dir / "prediction_seal.json").read_text())
    with np.load(backbone_dir / "prediction.npz") as stored:
        shape = stored["prediction_m"].shape
        frame_zero = stored["frame_zero_points_m"].copy()
    centers = np.arange(16, dtype=np.int64)
    measurement = np.full(shape, np.nan, dtype=np.float32)
    visible = np.zeros(shape[:2], dtype=bool)
    valid = np.zeros(shape[:2], dtype=bool)
    measurement[0, centers] = frame_zero[centers]
    visible[0, centers] = True
    valid[0, centers] = True
    for frame in (19, 38, 57):
        measurement[frame, centers] = frame_zero[centers] + (0.0, 0.005, 0.0)
        visible[frame, centers] = True
        valid[frame, centers] = True
    output = root / "measurement"
    output.mkdir()
    archive = output / "measurement.npz"
    np.savez_compressed(
        archive,
        measurement_m=measurement,
        measurement_visibility=visible,
        measurement_validity=valid,
        center_ids=centers,
    )
    manifest: dict[str, object] = {
        "artifact_kind": "Deform360CausalRawCameraMeasurement",
        "protocol_id": PROTOCOL_ID,
        "case": seal["case"],
        "output": {"measurement_archive_sha256": file_sha256(archive)},
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_reconstruction_after_frame_zero_read": False,
            "maximum_video_frame_read_by_update": [19, 38, 57],
        },
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    _write_json(output / "measurement_manifest.json", manifest)
    return output


def _write_tactile_features(
    path: Path,
    *,
    case: str,
    future_marker: str = "first",
) -> None:
    updates = []
    for frame in (19, 38, 57):
        row = {name: 1.0 for name in TACTILE_REGRET_FEATURE_NAMES}
        row["update_frame"] = frame
        updates.append(row)
    payload: dict[str, object] = {
        "artifact_kind": "Deform360CausalTactileFeatureAuditV2",
        "schema_version": 2,
        "information_boundary": {
            "target_outcomes_read": False,
            "held_v8_read": False,
            "future_tactile_values_used_for_update": False,
            "each_update_uses_tactile_at_or_before_update": True,
            "episode_wide_tactile_normalization_used": False,
        },
        "cases": [{"case": case, "updates": updates}],
        "future_payload_custody_marker": future_marker,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    _write_json(path, payload)


def test_v14_custody_is_bound_and_outcome_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = _write_protocol_bundle(tmp_path)
    protocol = load_outcome_sealed_protocol(bundle["protocol"], repository_root=REPO)
    decision = validate_v14_custody(
        protocol,
        source_decision_path=bundle["decision"],
        staging_queue_path=bundle["queue"],
    )
    assert decision["outcome_blind_gate"]["source_outcome_authorized"] is False

    decision["information_boundary"]["future_object_observation_read"] = True
    decision["artifact_sha256"] = namespaced_canonical_sha256(
        decision,
        namespace=V14_DECISION_NAMESPACE,
        digest_key="artifact_sha256",
    )
    _write_json(bundle["decision"], decision)
    protocol["cohort"]["v14_custody"]["source_decision_git_blob_oid"] = git_blob_oid(
        bundle["decision"]
    )
    protocol["cohort"]["v14_custody"]["source_decision_artifact_sha256"] = (
        decision["artifact_sha256"]
    )
    with pytest.raises(ValueError, match="outcome boundary"):
        validate_v14_custody(
            protocol,
            source_decision_path=bundle["decision"],
            staging_queue_path=bundle["queue"],
        )


def test_registered_protocol_binds_sources_model_and_tactile_manifest() -> None:
    protocol = load_outcome_sealed_protocol(
        REGISTERED_PROTOCOL,
        repository_root=REPO,
    )
    model = load_frozen_tactile_model(REGISTERED_SOURCE_RESULT, protocol)
    tactile_manifest = json.loads(REGISTERED_TACTILE_MANIFEST.read_text())

    assert len(protocol["cohort"]["cases"]) == 12
    assert protocol["claim_boundary"]["label"] == CLAIM_LABEL
    assert model.source_object_count == 17
    assert tactile_manifest["artifact_sha256"] == canonical_artifact_sha256(
        tactile_manifest
    )
    assert len(tactile_manifest["cases"]) == 12


def test_source_text_identity_is_checkout_line_ending_invariant(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "source-lf.py"
    crlf = tmp_path / "source-crlf.py"
    lf.write_bytes(b"alpha = 1\nbeta = 2\n")
    crlf.write_bytes(b"alpha = 1\r\nbeta = 2\r\n")

    assert canonical_text_sha256(lf) == canonical_text_sha256(crlf)


def test_guarded_prediction_is_target_free_and_future_marker_invariant(
    tmp_path: Path,
) -> None:
    bundle = _write_protocol_bundle(tmp_path)
    backbone = _stage_backbone(tmp_path, bundle)
    measurement = _write_measurement(tmp_path, backbone)
    case = json.loads((backbone / "prediction_seal.json").read_text())["case"]
    first_features = tmp_path / "tactile-first.json"
    second_features = tmp_path / "tactile-second.json"
    _write_tactile_features(first_features, case=case, future_marker="first")
    _write_tactile_features(second_features, case=case, future_marker="mutated")

    first = tmp_path / "prediction-first"
    second = tmp_path / "prediction-second"
    first_seal = build_guarded_prediction(
        first,
        repository_root=REPO,
        protocol_path=bundle["protocol"],
        backbone_dir=backbone,
        measurement_dir=measurement,
        tactile_feature_path=first_features,
        source_result_path=bundle["source_result"],
    )
    second_seal = build_guarded_prediction(
        second,
        repository_root=REPO,
        protocol_path=bundle["protocol"],
        backbone_dir=backbone,
        measurement_dir=measurement,
        tactile_feature_path=second_features,
        source_result_path=bundle["source_result"],
    )
    assert first_seal["status"] == second_seal["status"] == ORDINARY_STATUS
    with np.load(first / "guarded_prediction.npz") as left, np.load(
        second / "guarded_prediction.npz"
    ) as right:
        assert np.array_equal(
            left["guarded_prediction_m"], right["guarded_prediction_m"]
        )
    protocol = load_outcome_sealed_protocol(bundle["protocol"])
    validate_prediction_seal(first_seal, protocol=protocol, prediction_dir=first)


def test_technical_failure_preserves_persistence_bit_exactly(tmp_path: Path) -> None:
    bundle = _write_protocol_bundle(tmp_path)
    backbone = _stage_backbone(tmp_path, bundle)
    output = tmp_path / "fallback"
    seal = build_technical_fallback(
        output,
        protocol_path=bundle["protocol"],
        backbone_dir=backbone,
        failure_stage="camera_measurement",
        failure_type="RuntimeError",
        failure_message="synthetic provider failure",
    )
    assert seal["status"] == TECHNICAL_FALLBACK_STATUS
    with np.load(output / "guarded_prediction.npz") as stored:
        assert np.array_equal(
            stored["guarded_prediction_m"], stored["persistence_m"]
        )
    protocol = load_outcome_sealed_protocol(bundle["protocol"])
    validate_prediction_seal(seal, protocol=protocol, prediction_dir=output)


def test_barrier_reports_four_counts_and_blocks_unsealable_case(
    tmp_path: Path,
) -> None:
    bundle = _write_protocol_bundle(tmp_path)
    protocol = load_outcome_sealed_protocol(bundle["protocol"])
    passed_root = tmp_path / "passed"
    for rank, record in enumerate(protocol["cohort"]["cases"], start=1):
        case_root = tmp_path / f"case-{rank}"
        backbone = _stage_backbone(case_root, bundle, rank=rank)
        build_technical_fallback(
            passed_root / record["case"],
            protocol_path=bundle["protocol"],
            backbone_dir=backbone,
            failure_stage="synthetic",
            failure_type="RuntimeError",
            failure_message="registered fallback",
        )
    barrier_path = tmp_path / "barrier.json"
    barrier = build_prediction_barrier(
        barrier_path,
        protocol_path=bundle["protocol"],
        prediction_root=passed_root,
    )
    assert barrier["barrier_passed"] is True
    assert barrier["ordinary_successful_prediction_count"] == 0
    assert barrier["retained_technical_failure_count"] == 12
    assert barrier["unsealable_case_count"] == 0
    assert barrier["total_locked_case_count"] == 12
    validate_prediction_barrier(
        barrier_path,
        protocol_path=bundle["protocol"],
        prediction_root=passed_root,
    )

    blocked_root = tmp_path / "blocked"
    for case_dir in passed_root.iterdir():
        target = blocked_root / case_dir.name
        target.mkdir(parents=True)
        for source in case_dir.iterdir():
            (target / source.name).write_bytes(source.read_bytes())
    first_case = protocol["cohort"]["cases"][0]["case"]
    for path in (blocked_root / first_case).iterdir():
        path.unlink()
    write_unsealable_case(
        blocked_root / first_case / "unsealable.json",
        protocol_path=bundle["protocol"],
        case=first_case,
        failure_stage="backbone",
        failure_type="ValueError",
        failure_message="synthetic unsealable case",
    )
    blocked_path = tmp_path / "blocked-barrier.json"
    blocked = build_prediction_barrier(
        blocked_path,
        protocol_path=bundle["protocol"],
        prediction_root=blocked_root,
    )
    assert blocked["barrier_passed"] is False
    assert blocked["unsealable_case_count"] == 1
    with pytest.raises(ValueError, match="did not authorize outcomes"):
        validate_prediction_barrier(
            blocked_path,
            protocol_path=bundle["protocol"],
            prediction_root=blocked_root,
        )
