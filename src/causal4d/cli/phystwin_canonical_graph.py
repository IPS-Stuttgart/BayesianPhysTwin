"""CLI for freezing a canonical object-only PhysTwin material graph."""

from __future__ import annotations

import argparse
import json

from causal4d.phystwin_canonical_graph import (
    build_canonical_material_graph_from_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze one PhysTwin object's vertices, springs, rest lengths, and "
            "masses before same-object multi-action acquisition."
        )
    )
    parser.add_argument("final_data")
    parser.add_argument("optimal_params")
    parser.add_argument("output_npz")
    args = parser.parse_args()
    result = build_canonical_material_graph_from_case(
        args.final_data,
        args.optimal_params,
        args.output_npz,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
