"""CLI for source-only native PGRD temporal-head training."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_pgrd_native import (
    NativePGRDTrainingConfig,
    run_native_pgrd_development,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PGRD's temporal head on PhysTwin prefixes and run the frozen "
            "three-action development gate."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("output_dir")
    parser.add_argument("--pgrd-checkout", required=True)
    parser.add_argument("--pgrd-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source-case-limit", type=int)
    args = parser.parse_args()
    summary = run_native_pgrd_development(
        args.protocol,
        args.output_dir,
        pgrd_checkout=args.pgrd_checkout,
        pgrd_checkpoint=args.pgrd_checkpoint,
        device=args.device,
        config=NativePGRDTrainingConfig(epochs=args.epochs),
        source_case_limit=args.source_case_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
