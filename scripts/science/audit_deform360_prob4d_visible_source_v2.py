#!/usr/bin/env python3
"""Audit the exact compact artifact from the frozen public Prob4D v2 source run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_prob4d_visible_source_audit import (
    audit_deform360_prob4d_visible_source_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validator-revision", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-run-attempt", type=int, required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = audit_deform360_prob4d_visible_source_v2(
        source_root=arguments.source_root,
        output_directory=arguments.output_dir,
        validator_revision=arguments.validator_revision,
        source_run_id=arguments.source_run_id,
        source_run_attempt=arguments.source_run_attempt,
        source_artifact_id=arguments.source_artifact_id,
        source_artifact_name=arguments.source_artifact_name,
        source_artifact_digest=arguments.source_artifact_digest,
    )
    print(json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
