#!/usr/bin/env python3
"""Run the frozen two-group Genesis MPM source-physics qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.genesis_mpm_source_qualification_v1 import (
    run_genesis_mpm_source_qualification_v1,
)


def _group_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group roots use GROUP_ID=/absolute/path")
    group_id, raw_path = value.split("=", 1)
    if not group_id or group_id.strip() != group_id:
        raise argparse.ArgumentTypeError("group ID is not canonical")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("group root must be absolute")
    return group_id, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--group-root", action="append", type=_group_root, required=True
    )
    args = parser.parse_args()
    roots: dict[str, Path] = {}
    for group_id, path in args.group_root:
        if group_id in roots:
            parser.error(f"duplicate group root: {group_id}")
        roots[group_id] = path
    result = run_genesis_mpm_source_qualification_v1(
        protocol_path=args.protocol,
        group_roots=roots,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
