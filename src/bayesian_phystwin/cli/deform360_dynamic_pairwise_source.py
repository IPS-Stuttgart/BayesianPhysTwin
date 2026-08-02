"""CLI for the opened-source dynamic pairwise-belief evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_dynamic_pairwise_source import (
    evaluate_dynamic_pairwise_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen dynamic observation arm on Open27."
    )
    parser.add_argument("source_root")
    parser.add_argument("measurement_root")
    parser.add_argument("output_dir")
    parser.add_argument("--transfer-manifest-sha256")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_dynamic_pairwise_source(
        args.source_root,
        args.measurement_root,
        args.output_dir,
        transfer_manifest_sha256=args.transfer_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
