"""CLI for the frozen query-field open27 development decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_query_field_development import (
    write_query_field_development_decision,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a frozen target-query field using exactly the audited, "
            "already-open Deform360 source-27 development panel."
        )
    )
    parser.add_argument("source_root")
    parser.add_argument("audited_run_dir")
    parser.add_argument("output_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = write_query_field_development_decision(
        args.source_root,
        args.audited_run_dir,
        args.output_json,
    )
    output = Path(args.output_json).resolve()
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "selected_candidate_id": decision["selection"]["selected_candidate_id"],
                "selected_objective_m": decision["selection"]["selected_objective_m"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
