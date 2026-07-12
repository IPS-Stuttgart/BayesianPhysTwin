"""CLI for locking the canonical material graph before data collection."""

from __future__ import annotations

import argparse
import json

from causal4d.rest_geometry_protocol import write_rest_geometry_registration


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register one canonical object-only PhysTwin graph inside the "
            "same-object dataset root before confirmatory collection."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("dataset_root")
    parser.add_argument("canonical_graph")
    args = parser.parse_args()
    result = write_rest_geometry_registration(
        args.protocol,
        args.dataset_root,
        args.canonical_graph,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
