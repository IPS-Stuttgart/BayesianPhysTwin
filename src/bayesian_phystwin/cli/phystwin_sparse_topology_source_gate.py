"""CLI for the fixed sparse-topology source-transfer gate."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_sparse_topology_source_gate import (
    run_sparse_topology_source_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate a fixed sparse PhysTwin topology on transfer prefixes."
    )
    parser.add_argument("source_root")
    parser.add_argument("output")
    parser.add_argument("protocol")
    args = parser.parse_args()
    result = run_sparse_topology_source_gate(
        args.source_root,
        args.output,
        args.protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
