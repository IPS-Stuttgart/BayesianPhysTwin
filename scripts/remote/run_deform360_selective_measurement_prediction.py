#!/usr/bin/env python3
"""Build sealed sparse measurements and predictions without opening targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_raw_camera_observation import (
    MANIFEST_FILENAME,
    MEASUREMENT_FILENAME,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    VIRTUAL_SENSING_SEAL_FILENAME,
    build_selective_raw_camera_measurement_case,
    build_selective_virtual_sensing_prediction_case,
    selective_case_records,
    validate_selective_prediction_seal,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    records = selective_case_records(args.protocol)
    selected = [
        record
        for index, record in enumerate(records)
        if index % args.shard_count == args.shard_index
    ]
    config = RawCameraObservationConfig()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.checkpoint,
        device=args.device,
        config=config,
    )
    rows = []
    try:
        for record in selected:
            case = str(record["case"])
            backbone = args.backbone_root.resolve() / case
            processed = (
                args.staged_root.resolve() / case / "prefix" / "episode_0000"
            )
            measurement = args.measurement_root.resolve() / case
            prediction = args.prediction_root.resolve() / case
            if not measurement.exists():
                build_selective_raw_camera_measurement_case(
                    args.protocol,
                    backbone,
                    processed,
                    measurement,
                    runtime,
                    config=config,
                )
            elif not all(
                (measurement / name).is_file()
                for name in (MANIFEST_FILENAME, MEASUREMENT_FILENAME)
            ):
                raise ValueError(f"incomplete existing measurement: {measurement}")
            if not prediction.exists():
                seal = build_selective_virtual_sensing_prediction_case(
                    args.protocol, backbone, measurement, prediction
                )
            else:
                seal_path = prediction / VIRTUAL_SENSING_SEAL_FILENAME
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
                validate_selective_prediction_seal(
                    seal,
                    protocol_path=args.protocol,
                    prediction_dir=prediction,
                )
            rows.append(
                {
                    "case": case,
                    "measurement_manifest_sha256": _sha256(
                        measurement / MANIFEST_FILENAME
                    ),
                    "measurement_archive_sha256": _sha256(
                        measurement / MEASUREMENT_FILENAME
                    ),
                    "prediction_seal_sha256": _sha256(
                        prediction / VIRTUAL_SENSING_SEAL_FILENAME
                    ),
                    "prediction_result_sha256": seal["result_sha256"],
                }
            )
    finally:
        runtime.close()
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360SelectivePredictionShard",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "case_count": len(rows),
        "cases": rows,
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
        },
    }
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    output = args.prediction_root / f"prediction-shard-{args.shard_index:02d}.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
