#!/usr/bin/env python3
"""Isolated x0-only query worker for Deform360 held v8.

The command line deliberately has no target, outcome, visibility, validity,
or score argument.  This process imports only the frozen query implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one held-v8 x0-only query")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--official-query-manifest", required=True)
    parser.add_argument("--frozen-field-manifest", required=True)
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--output-seal", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if os.path.lexists("/nonexistent") or os.path.lexists(
        "/nonexistent/bpt-held-v82-pycache"
    ):
        raise RuntimeError("reserved held-v8 bytecode prefix is available")
    forbidden = sorted(
        name
        for name in sys.modules
        if "outcome" in name or name.endswith("deform360_held_v8_scoring")
    )
    if forbidden:
        raise RuntimeError(f"future-bearing module preloaded in x0 worker: {forbidden}")
    source = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(source))
    from bayesian_phystwin import deform360_held_v8_query_artifacts as query

    lock_sha256 = hashlib.sha256(Path(arguments.lock).read_bytes()).hexdigest()
    query.write_queried_prediction_artifact(
        arguments.output_archive,
        arguments.output_seal,
        lock_path=arguments.lock,
        lock_sha256=lock_sha256,
        frozen_field_manifest_path=arguments.frozen_field_manifest,
        official_query_manifest_path=arguments.official_query_manifest,
    )
    forbidden = sorted(
        name
        for name in sys.modules
        if name.endswith("deform360_held_v8_outcome_artifacts")
        or name.endswith("deform360_held_v8_scoring")
        or name.endswith("deform360_held_v8_score_artifacts")
    )
    if forbidden:
        raise RuntimeError(f"future-bearing module imported in x0 worker: {forbidden}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
