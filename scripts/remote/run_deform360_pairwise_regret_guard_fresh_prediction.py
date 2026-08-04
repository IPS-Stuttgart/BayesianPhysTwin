#!/usr/bin/env python3
"""Build one frozen raw-prefix guarded prediction before outcome opening."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_artifacts import (
    FAILURE_SEAL_FILENAME,
    PHYSICAL_SEAL_FILENAME,
    build_fresh_guarded_prediction,
    build_fresh_runtime_failure_seal,
    fresh_case_records,
    validate_fresh_physical_seal,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    validate_fresh_processing_protocol,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    validate_fresh_technical_lock,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_case_with_contract,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--processing-protocol", type=Path, required=True)
    parser.add_argument("--physical-case-dir", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-qualification", type=Path, required=True)
    parser.add_argument("--alltracker-source", type=Path, required=True)
    parser.add_argument("--alltracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    lock_path = args.technical_lock.resolve()
    protocol_path = args.processing_protocol.resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    validate_fresh_technical_lock(lock)
    validate_fresh_processing_protocol(protocol)
    physical_root = args.physical_case_dir.resolve()
    physical_seal_path = physical_root / PHYSICAL_SEAL_FILENAME
    physical_seal = _load_json(physical_seal_path)
    validate_fresh_physical_seal(
        physical_seal,
        case_dir=physical_root,
        lock=lock,
        protocol=protocol,
    )
    case = str(physical_seal["case"])
    expected_cases = tuple(str(row["case"]) for row in fresh_case_records(lock))
    _require(case in expected_cases, "physical case is outside the technical lock")
    processed = args.processed_episode_dir.resolve()
    _require(
        processed.parent.name == physical_seal["object_id"]
        and int(processed.name.removeprefix("episode_"))
        == int(physical_seal["episode_id"]),
        "processed source differs from the physical seal",
    )
    measurement_dir = args.measurement_root.resolve() / case
    prediction_dir = args.prediction_root.resolve() / case
    _require(not prediction_dir.exists(), "prediction disposition already exists")
    config = RawCameraObservationConfig()

    def validate(seal: dict[str, object]) -> None:
        validate_fresh_physical_seal(
            seal, case_dir=physical_root, lock=lock, protocol=protocol
        )

    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source.resolve(),
        args.alltracker_checkpoint.resolve(),
        device=args.device,
        config=config,
    )
    try:
        measurement = build_raw_camera_measurement_case_with_contract(
            physical_root,
            processed,
            measurement_dir,
            runtime,
            protocol_id=str(lock["protocol_id"]),
            expected_case_names=expected_cases,
            prediction_seal_validator=validate,
            claim_boundary=(
                "single-object multi-action technical replication; raw RGB "
                "prefix only, with future object outcomes still sealed"
            ),
            config=config,
        )
    except Exception as exc:
        prediction_dir.mkdir(parents=True, exist_ok=False)
        failure = build_fresh_runtime_failure_seal(
            lock_path,
            prediction_dir / FAILURE_SEAL_FILENAME,
            object_id=str(physical_seal["object_id"]),
            episode_id=int(physical_seal["episode_id"]),
            stage="prefix-camera-measurement",
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_files={
                "physical_seal": physical_seal_path,
                "technical_lock": lock_path,
                "processing_protocol": protocol_path,
            },
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2
    finally:
        runtime.close()
    try:
        prediction = build_fresh_guarded_prediction(
            lock_path,
            protocol_path,
            physical_root,
            measurement_dir,
            args.source_qualification.resolve(),
            prediction_dir,
        )
    except Exception as exc:
        prediction_dir.mkdir(parents=True, exist_ok=False)
        failure = build_fresh_runtime_failure_seal(
            lock_path,
            prediction_dir / FAILURE_SEAL_FILENAME,
            object_id=str(physical_seal["object_id"]),
            episode_id=int(physical_seal["episode_id"]),
            stage="guarded-prediction",
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_files={
                "physical_seal": physical_seal_path,
                "measurement_manifest": measurement_dir / "measurement_manifest.json",
                "measurement_archive": measurement_dir / "measurement.npz",
                "source_qualification": args.source_qualification.resolve(),
            },
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(
        json.dumps(
            {
                "case": case,
                "code_revision": revision,
                "measurement_result_sha256": measurement["result_sha256"],
                "prediction_result_sha256": prediction["result_sha256"],
                "accepted_interval_count": prediction["accepted_interval_count"],
                "information_boundary": {
                    "future_object_rgb_read": False,
                    "future_object_geometry_read": False,
                    "target_metric_read": False,
                    "held_v8_access": False,
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
