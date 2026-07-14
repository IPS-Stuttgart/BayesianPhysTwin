#!/usr/bin/env python3
"""Record a checksummed, fail-closed Deform360 source-stage failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_backend import (
    build_source_stage_failure_artifact,
    load_backend_policy,
    validate_source_stage_failure_artifact,
    write_source_stage_failure_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backend-policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("source-geometry", "source-grid", "source-pooling"),
        required=True,
    )
    parser.add_argument("--failed-episode-id", type=int)
    parser.add_argument("--completed-episode-id", type=int, action="append")
    parser.add_argument("--error-type", required=True)
    parser.add_argument("--error-message", required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_deform360_replication_protocol(args.protocol)
    backend_policy = load_backend_policy(args.backend_policy)
    matches = [
        row
        for row in protocol["config"]["cohort"]
        if row["object_id"] == args.object_id
    ]
    if len(matches) != 1:
        raise ValueError("failure object is outside the locked cohort")
    source_ids = list(map(int, matches[0]["source_episode_ids"]))
    completed_ids = set(args.completed_episode_id or [])
    if args.stage == "source-pooling":
        if args.failed_episode_id is not None or completed_ids != set(source_ids):
            raise ValueError("source pooling requires every source grid")
    elif args.failed_episode_id not in source_ids:
        raise ValueError("failed episode is outside the locked source split")
    if not completed_ids.issubset(source_ids) or args.failed_episode_id in completed_ids:
        raise ValueError("completed source episodes are inconsistent")
    statuses = [
        {
            "episode_id": f"{args.object_id}/episode_{episode_id:04d}",
            "status": (
                "failed"
                if episode_id == args.failed_episode_id
                else "completed"
                if episode_id in completed_ids
                else "not-attempted"
            ),
        }
        for episode_id in source_ids
    ]

    data_root = args.data_root.resolve()
    evidence = []
    for raw_path in args.evidence:
        path = raw_path.resolve()
        try:
            relative = path.relative_to(data_root)
        except ValueError as error:
            raise ValueError("failure evidence must be inside the data root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = build_source_stage_failure_artifact(
        protocol,
        backend_policy,
        object_id=args.object_id,
        stage=args.stage,
        failed_episode_id=(
            None
            if args.failed_episode_id is None
            else f"{args.object_id}/episode_{args.failed_episode_id:04d}"
        ),
        error_type=args.error_type,
        error_message=args.error_message,
        episode_status=statuses,
        evidence=evidence,
    )
    write_source_stage_failure_artifact(args.output, payload)
    print(
        json.dumps(validate_source_stage_failure_artifact(payload), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
