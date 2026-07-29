#!/usr/bin/env python3
"""Seal the outcome-free V12 synthetic positive and placebo controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_synthetic import (
    run_causal_response_synthetic_controls,
    write_causal_response_synthetic_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_causal_response_synthetic_controls()
    write_causal_response_synthetic_result(args.output, result)
    print(
        json.dumps(
            {
                "artifact_sha256": result.artifact_sha256,
                "positive_detection_rate": result.positive_detection_rate,
                "placebo_false_admission_rate": (result.placebo_false_admission_rate),
                "positive_improvement_fraction": (result.positive_improvement_fraction),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
