#!/usr/bin/env python3
"""Run the frozen source-only SOFA FEM keyed-Dirichlet v2 qualification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.sofa_fem_source_qualification_v2 import (
    run_sofa_fem_source_qualification_v2,
)


def _group_root(value: str) -> tuple[str, Path]:
    group_id, separator, root = value.partition("=")
    if not separator or not group_id or group_id.strip() != group_id or not root:
        raise argparse.ArgumentTypeError("group roots use GROUP_ID=/absolute/path")
    path = Path(root)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("group roots must be absolute")
    return group_id, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distribution-archive", type=Path, required=True)
    parser.add_argument("--sofa-root", type=Path, required=True)
    parser.add_argument(
        "--group-root",
        action="append",
        type=_group_root,
        required=True,
        help="One GROUP_ID=/absolute/path entry for every frozen source group.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = dict(args.group_root)
    if len(roots) != len(args.group_root):
        raise ValueError("group-root entries must be unique")
    result = run_sofa_fem_source_qualification_v2(
        protocol_path=args.protocol,
        group_roots=roots,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        distribution_archive=args.distribution_archive,
        sofa_root=args.sofa_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
