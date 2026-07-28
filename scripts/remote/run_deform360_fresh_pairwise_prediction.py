#!/usr/bin/env python3
"""Seal one target-free fresh-object pairwise online-belief prediction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_fresh_pairwise_prediction import (
    CANDIDATE_ARM,
    SELECTED_RAW_ARM,
    predict_fresh_pairwise_arrays,
)
from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    array_sha256,
    build_belief_prediction_seal,
    canonical_sha256,
    file_sha256,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    load_json,
    validate_backbone_seal,
    write_json,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
)
from bayesian_phystwin.phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
)
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    protocol = load_fresh_pairwise_protocol(
        args.protocol,
        repository_root=repo,
    )
    cohort = load_bound_cohort(args.cohort_lock, protocol)
    backbone_dir = args.backbone_case_dir.resolve()
    backbone_seal_path = backbone_dir / "prediction_seal.json"
    backbone = load_json(backbone_seal_path)
    validate_backbone_seal(
        backbone,
        protocol_config_sha256=protocol["config_file_sha256"],
        cohort_lock_sha256=cohort["cohort_lock_sha256"],
    )
    measurement_dir = args.measurement_dir.resolve()
    measurement_manifest_path = measurement_dir / MANIFEST_FILENAME
    measurement_manifest = load_json(measurement_manifest_path)
    _require(
        measurement_manifest.get("protocol_id") == protocol["protocol_id"]
        and measurement_manifest.get("case") == backbone["case"]
        and measurement_manifest.get("result_sha256")
        == canonical_sha256(measurement_manifest, digest_key="result_sha256"),
        "measurement manifest is incompatible",
    )
    boundary = measurement_manifest.get("information_boundary", {})
    _require(
        boundary.get("target_data_read") is False
        and boundary.get("outcome_manifest_read") is False,
        "measurement crossed the target boundary",
    )
    physical_archive = Path(backbone["prediction_archive"]["path"])
    if not physical_archive.is_file():
        physical_archive = backbone_dir / physical_archive.name
    _require(
        file_sha256(physical_archive)
        == backbone["prediction_archive"]["file_sha256"],
        "physical archive checksum changed",
    )
    measurement_archive = measurement_dir / MEASUREMENT_FILENAME
    _require(
        file_sha256(measurement_archive)
        == measurement_manifest["output"]["measurement_archive_sha256"],
        "measurement archive checksum changed",
    )
    with np.load(physical_archive, allow_pickle=False) as stored:
        physical = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
    with np.load(measurement_archive, allow_pickle=False) as stored:
        measurement = np.asarray(stored["measurement_m"]).copy()
        measurement_visibility = np.asarray(
            stored["measurement_visibility"], dtype=bool
        )
        measurement_validity = np.asarray(
            stored["measurement_validity"], dtype=bool
        )
        center_ids = np.asarray(stored["center_ids"], dtype=np.int64)

    gate = PairwiseCorrespondenceGateConfig(
        **protocol["method"]["pairwise_gate"]
    )
    belief = RecursiveRbfBeliefConfig(**protocol["method"]["recursive_rbf"])
    method_report, arrays = predict_fresh_pairwise_arrays(
        physical,
        persistence,
        measurement,
        measurement_visibility,
        measurement_validity,
        center_ids=center_ids,
        update_frames=tuple(protocol["method"]["update_frames"]),
        gate_config=gate,
        belief_config=belief,
    )
    output = args.output_dir.resolve()
    _require(not output.exists(), "belief output directory already exists")
    output.mkdir(parents=True)
    archive_path = output / "belief_prediction.npz"
    archive_arrays = {
        "physical_prior_m": arrays["physical_prior"],
        "persistence_m": arrays["persistence"],
        "selected_raw_backbone_m": arrays[SELECTED_RAW_ARM],
        "candidate_m": arrays[CANDIDATE_ARM],
        "center_ids": center_ids,
    }
    np.savez_compressed(archive_path, **archive_arrays)
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshPairwiseBeliefPrediction",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": protocol["config_file_sha256"],
        "cohort_lock_sha256": cohort["cohort_lock_sha256"],
        "case": backbone["case"],
        "object_id": backbone["object_id"],
        "episode_id": int(backbone["episode_id"]),
        "category": backbone["category"],
        "selected_arm": CANDIDATE_ARM,
        "method_report": method_report,
        "gate_config": asdict(gate),
        "belief_config": asdict(belief),
        "prediction_archive": {
            "path": str(archive_path),
            "file_sha256": file_sha256(archive_path),
            "array_sha256": {
                name: array_sha256(value)
                for name, value in sorted(archive_arrays.items())
            },
        },
        "inputs": {
            "backbone_seal": {
                "path": str(backbone_seal_path),
                "file_sha256": file_sha256(backbone_seal_path),
            },
            "measurement_manifest": {
                "path": str(measurement_manifest_path),
                "file_sha256": file_sha256(measurement_manifest_path),
            },
            "measurement_archive": {
                "path": str(measurement_archive),
                "file_sha256": file_sha256(measurement_archive),
            },
        },
        "information_boundary": {
            "future_target_read": False,
            "outcome_manifest_read": False,
            "causal_rgb_prefix_updates_used": list(
                protocol["method"]["update_frames"]
            ),
            "prediction_hashed_before_future_outcome_scoring": True,
        },
    }
    report["result_sha256"] = canonical_sha256(
        report, digest_key="result_sha256"
    )
    report_path = write_json(output / "belief_prediction_report.json", report)
    seal = build_belief_prediction_seal(
        output / "belief_prediction_seal.json",
        protocol_path=args.protocol,
        cohort_path=args.cohort_lock,
        backbone_seal_path=backbone_seal_path,
        measurement_manifest_path=measurement_manifest_path,
        prediction_archive_path=archive_path,
        prediction_report_path=report_path,
    )
    print(
        json.dumps(
            {
                "case": backbone["case"],
                "prediction_report_sha256": report["result_sha256"],
                "belief_prediction_seal_sha256": seal["result_sha256"],
                "selected_backbone_by_update": [
                    item["selected_backbone"]
                    for item in method_report["updates"]
                ],
                "pairwise_gate_accepted_by_update": [
                    item["selected_pairwise_gate"]["accepted"]
                    for item in method_report["updates"]
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
