from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    BACKBONE_SEAL_KIND,
    BELIEF_SEAL_KIND,
    build_belief_prediction_seal,
    build_backbone_seal,
    build_completeness_barrier,
    canonical_sha256,
    file_sha256,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    validate_backbone_seal,
    validate_belief_prediction_seal,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "configs/sota/deform360_fresh_pairwise_belief_v1.json"
COHORT = (
    REPO
    / "results/sota/deform360_fresh_source_lock_v1"
    / "deform360_fresh_object_cohort_lock_v1.json"
)


def test_frozen_protocol_binds_method_sources_and_cohort() -> None:
    protocol = load_fresh_pairwise_protocol(PROTOCOL, repository_root=REPO)
    cohort = load_bound_cohort(COHORT, protocol)

    assert len(cohort["cases"]) == 12
    assert protocol["method"]["selected_arm"].endswith("pairwise_clique")
    assert protocol["physical_backbone"]["canonical_observed_node_count"] == 384


def test_backbone_seal_binds_physical_archive(tmp_path: Path) -> None:
    protocol = load_fresh_pairwise_protocol(PROTOCOL)
    cohort = load_bound_cohort(COHORT, protocol)
    case = cohort["cases"][0]
    admission = {
        "accepted": True,
        "admission_sha256": case["admission_sha256"],
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    archive = tmp_path / "prediction.npz"
    np.savez_compressed(archive, prediction_m=np.zeros((76, 128, 3)))
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "case": case["case"],
        "passed": True,
        "physical_prediction_archive": {
            "file_sha256": __import__("hashlib").sha256(
                archive.read_bytes()
            ).hexdigest()
        },
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    manifest_path = tmp_path / "physical.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    seal = build_backbone_seal(
        tmp_path / "prediction_seal.json",
        protocol_path=PROTOCOL,
        cohort_path=COHORT,
        case_record=case,
        admission_path=admission_path,
        prediction_archive=archive,
        physical_manifest=manifest_path,
    )

    assert seal["artifact_kind"] == BACKBONE_SEAL_KIND
    validate_backbone_seal(seal)


def _write_belief_inputs(
    root: Path,
    *,
    case: dict[str, object],
    protocol: dict[str, object],
    cohort_sha256: str,
) -> None:
    root.mkdir(parents=True)
    backbone = {
        "schema_version": 1,
        "artifact_kind": BACKBONE_SEAL_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": file_sha256(PROTOCOL),
        "cohort_lock_sha256": cohort_sha256,
        "case": case["case"],
        "object_id": case["object_id"],
        "episode_id": case["episode_id"],
        "episode_key": f"{case['object_id']}/episode_{int(case['episode_id']):04d}",
        "category": case["category"],
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "outcome_manifest_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    backbone["result_sha256"] = canonical_sha256(
        backbone, digest_key="result_sha256"
    )
    backbone_path = root / "prediction_seal.json"
    backbone_path.write_text(json.dumps(backbone), encoding="utf-8")
    measurement = {
        "artifact_kind": "Deform360CausalRawCameraMeasurement",
        "protocol_id": protocol["protocol_id"],
        "case": case["case"],
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_reconstruction_after_frame_zero_read": False,
        },
    }
    measurement["result_sha256"] = canonical_sha256(
        measurement, digest_key="result_sha256"
    )
    measurement_path = root / "measurement_manifest.json"
    measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
    prediction = root / "belief_prediction.npz"
    np.savez_compressed(prediction, candidate_m=np.zeros((76, 128, 3)))
    report = {
        "artifact_kind": "Deform360FreshPairwiseBeliefPrediction",
        "protocol_id": protocol["protocol_id"],
        "case": case["case"],
        "information_boundary": {
            "future_target_read": False,
            "outcome_manifest_read": False,
        },
    }
    report["result_sha256"] = canonical_sha256(
        report, digest_key="result_sha256"
    )
    report_path = root / "belief_prediction_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    seal = build_belief_prediction_seal(
        root / "belief_prediction_seal.json",
        protocol_path=PROTOCOL,
        cohort_path=COHORT,
        backbone_seal_path=backbone_path,
        measurement_manifest_path=measurement_path,
        prediction_archive_path=prediction,
        prediction_report_path=report_path,
    )
    assert seal["artifact_kind"] == BELIEF_SEAL_KIND
    validate_belief_prediction_seal(seal)


def test_completeness_barrier_requires_exact_locked_cohort(tmp_path: Path) -> None:
    protocol = load_fresh_pairwise_protocol(PROTOCOL)
    cohort = load_bound_cohort(COHORT, protocol)
    prediction_root = tmp_path / "predictions"
    for case in cohort["cases"]:
        _write_belief_inputs(
            prediction_root / str(case["case"]),
            case=dict(case),
            protocol=protocol,
            cohort_sha256=str(cohort["cohort_lock_sha256"]),
        )

    barrier = build_completeness_barrier(
        tmp_path / "barrier.json",
        protocol_path=PROTOCOL,
        cohort_path=COHORT,
        prediction_root=prediction_root,
    )

    assert barrier["barrier_passed"] is True
    assert barrier["ordinary_prediction_count"] == 12
    assert barrier["retained_technical_failure_count"] == 0
