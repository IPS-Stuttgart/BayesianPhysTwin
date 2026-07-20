"""CLI for the open27 proper-Kabsch residual-field development ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_online_belief_evaluation import _sha256
from bayesian_phystwin.deform360_rigid_residual_development import (
    write_rigid_residual_development_ablation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen total-displacement and proper-Kabsch-plus-residual "
            "Gaussian fields on exactly the audited open27 development panel."
        )
    )
    parser.add_argument("source_root")
    parser.add_argument("audited_run_dir")
    parser.add_argument("output_json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = write_rigid_residual_development_ablation(
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
                "operator_decision": decision["operator_decision"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
