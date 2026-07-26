"""Print the installed Bayesian-PhysTwin Causal4D provider manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bayesian_phystwin.causal4d_provider_v1 import provider_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-revision",
        help="explicit source revision; defaults to the current Git checkout",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = provider_manifest(args.provider_revision).as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
