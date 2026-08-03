#!/usr/bin/env python3
"""Probe or certify the frozen fresh-processing CUDA runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_runtime import (
    certify_fresh_runtime,
    collect_fresh_runtime_identity,
    validate_fresh_runtime_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--processing-protocol", type=Path)
    parser.add_argument("--failed-artifact", type=Path)
    parser.add_argument("--failed-log", type=Path)
    return parser


def _revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parser().parse_args()
    revision = _revision(args.repo.resolve())
    if args.amendment is None:
        identity = collect_fresh_runtime_identity()
        write_json_artifact(identity, args.output)
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    required = (
        args.processing_protocol,
        args.failed_artifact,
        args.failed_log,
    )
    if any(path is None for path in required):
        raise ValueError("certification requires all frozen source bindings")
    amendment = validate_fresh_runtime_sources(
        args.amendment,
        args.processing_protocol,
        args.failed_artifact,
        args.failed_log,
    )
    certificate = certify_fresh_runtime(amendment, validator_commit=revision)
    write_json_artifact(certificate, args.output)
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
