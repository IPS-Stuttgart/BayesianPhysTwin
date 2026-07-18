"""CLI for the locked multi-case PhysTwin residual confirmation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_confirmatory import (
    run_phystwin_confirmatory_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen residual protocol on untouched PhysTwin cases."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    summary = run_phystwin_confirmatory_benchmark(
        args.data_root,
        args.output_dir,
        cases=args.cases,
        force=args.force,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
