"""CLI for fresh Deform360 source admission and cohort locking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_fresh_source_lock import (
    FreshSourceAdmissionConfig,
    build_fresh_cohort_lock,
    build_fresh_source_admission,
    build_object_exclusion_manifest,
    write_fresh_source_artifact,
)


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    admit = subparsers.add_parser("admit")
    admit.add_argument("episode_dir", type=Path)
    admit.add_argument("raw_metadata", type=Path)
    admit.add_argument("output", type=Path)
    admit.add_argument("--object-id", required=True)
    admit.add_argument("--episode-id", required=True, type=int)
    admit.add_argument("--category", required=True)
    admit.add_argument("--minimum-camera-count", type=int, default=3)
    admit.add_argument("--minimum-point-count", type=int, default=128)
    admit.add_argument("--maximum-point-count", type=int, default=10_000)
    admit.add_argument("--required-frame-count", type=int, default=76)
    admit.add_argument("--update-frames", type=int, nargs="+", default=[19, 38, 57])
    admit.add_argument("--minimum-test-frame-count", type=int, default=8)

    exclude = subparsers.add_parser("exclude")
    exclude.add_argument("output", type=Path)
    exclude.add_argument("--owner", required=True)
    exclude.add_argument("--object-id", action="append", required=True)
    exclude.add_argument("--source-sha256", action="append", required=True)

    lock = subparsers.add_parser("lock")
    lock.add_argument("output", type=Path)
    lock.add_argument("--admission", type=Path, action="append", required=True)
    lock.add_argument("--exclusion", type=Path, action="append", required=True)
    lock.add_argument("--cohort-size", type=int, required=True)
    lock.add_argument("--method-commit", required=True)
    lock.add_argument("--method-config-sha256", required=True)
    lock.add_argument("--parity-contract", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "admit":
        artifact = build_fresh_source_admission(
            args.episode_dir,
            args.raw_metadata,
            object_id=args.object_id,
            episode_id=args.episode_id,
            category=args.category,
            config=FreshSourceAdmissionConfig(
                minimum_camera_count=args.minimum_camera_count,
                minimum_point_count=args.minimum_point_count,
                maximum_point_count=args.maximum_point_count,
                required_frame_count=args.required_frame_count,
                update_frames=tuple(args.update_frames),
                minimum_test_frame_count=args.minimum_test_frame_count,
            ),
        )
    elif args.command == "exclude":
        artifact = build_object_exclusion_manifest(
            args.object_id,
            owner=args.owner,
            source_artifact_sha256s=args.source_sha256,
        )
    else:
        contract = _json(args.parity_contract)
        artifact = build_fresh_cohort_lock(
            [_json(path) for path in args.admission],
            [_json(path) for path in args.exclusion],
            cohort_size=args.cohort_size,
            method_commit=args.method_commit,
            method_config_sha256=args.method_config_sha256,
            parity_contract=contract,
        )
    write_fresh_source_artifact(artifact, args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
