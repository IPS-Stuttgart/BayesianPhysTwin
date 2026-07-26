#!/usr/bin/env python3
"""Open and score the fresh pairwise outcomes after the sealed barrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_fresh_pairwise_outcome import (
    evaluate_fresh_pairwise_outcomes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = evaluate_fresh_pairwise_outcomes(
        repository_root=args.repo,
        protocol_path=args.protocol,
        cohort_path=args.cohort_lock,
        admission_root=args.admission_root,
        prediction_root=args.prediction_root,
        processed_root=args.processed_root,
        barrier_path=args.barrier,
        analysis_path=args.analysis_lock,
        output_dir=args.output_dir,
        operator_path=Path(__file__),
    )
    print(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "transfer_gate_passed": summary["transfer_gate"]["passed"],
                "official_sota_claim_allowed": summary["official_sota_claim"][
                    "allowed"
                ],
                "result_sha256": summary["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
