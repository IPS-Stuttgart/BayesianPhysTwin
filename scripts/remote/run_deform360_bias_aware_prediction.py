#!/usr/bin/env python3
"""Seal one sparse-camera bias-aware prediction without opening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    build_prospective_bias_aware_prediction_case,
    build_prospective_raw_camera_measurement_case,
    prospective_case_record,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    load_bias_aware_prospective_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_uncertainty import (
    build_prospective_raw_camera_cycle_case,
    build_prospective_raw_camera_uncertainty_case,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_clean_repository(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _parse_case(case: str) -> tuple[str, int]:
    object_id, separator, episode = case.rpartition("-ep")
    _require(separator == "-ep" and object_id and episode.isdigit(), "invalid case")
    return object_id, int(episode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--uncertainty-root", type=Path, required=True)
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--alltracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    protocol_path = args.protocol.resolve()
    load_bias_aware_prospective_protocol(protocol_path)
    backbone = args.backbone_case_dir.resolve()
    object_id, episode_id = _parse_case(backbone.name)
    record = prospective_case_record(
        protocol_path, object_id=object_id, episode_id=episode_id
    )
    processed = args.processed_episode_dir.resolve()
    _require(processed.name == "episode_0000", "processed prefix episode changed")
    case = str(record["case"])
    measurement_dir = args.measurement_root.resolve() / case
    uncertainty_dir = args.uncertainty_root.resolve() / case
    cycle_dir = args.cycle_root.resolve() / case
    prediction_dir = args.prediction_root.resolve() / case
    config = RawCameraObservationConfig()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.alltracker_checkpoint,
        device=args.device,
        config=config,
    )
    try:
        measurement = build_prospective_raw_camera_measurement_case(
            protocol_path,
            backbone,
            processed,
            measurement_dir,
            runtime,
            config=config,
        )
        uncertainty = build_prospective_raw_camera_uncertainty_case(
            protocol_path,
            backbone,
            processed,
            measurement_dir,
            uncertainty_dir,
            runtime,
        )
        cycle = build_prospective_raw_camera_cycle_case(
            protocol_path,
            backbone,
            processed,
            measurement_dir,
            uncertainty_dir,
            cycle_dir,
            runtime,
        )
    finally:
        runtime.close()
    prediction = build_prospective_bias_aware_prediction_case(
        protocol_path,
        backbone,
        measurement_dir,
        cycle_dir,
        args.source_lock.resolve(),
        prediction_dir,
    )
    output = {
        "case": case,
        "role": record["role"],
        "code_revision": revision,
        "measurement_result_sha256": measurement["result_sha256"],
        "uncertainty_result_sha256": uncertainty["result_sha256"],
        "cycle_result_sha256": cycle["result_sha256"],
        "prediction_result_sha256": prediction["result_sha256"],
        "information_boundary": {
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_particle_tracks_read": False,
            "target_metric_read": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
