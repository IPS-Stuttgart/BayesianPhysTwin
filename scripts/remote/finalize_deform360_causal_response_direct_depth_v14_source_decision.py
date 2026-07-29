#!/usr/bin/env python3
"""Finalize the outcome-blind V14 prediction barrier and early source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_source_decision_v14 import (
    finalize_v14_source_prediction_decision,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--prediction-runtime", type=Path, required=True)
    parser.add_argument("--admission-prelock", type=Path, required=True)
    parser.add_argument("--physical-prelock", type=Path, required=True)
    parser.add_argument("--spatial-support-runtime-v2", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = finalize_v14_source_prediction_decision(
        repository=args.repo,
        method_protocol_path=args.method_protocol,
        source_lock_path=args.source_lock,
        prediction_runtime_path=args.prediction_runtime,
        admission_custody_path=args.admission_prelock,
        physical_custody_path=args.physical_prelock,
        spatial_support_runtime_path=args.spatial_support_runtime_v2,
        prediction_root=args.prediction_root,
        output_path=args.output,
    )
    gate = result["outcome_blind_gate"]
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "candidate_applied_object_count": gate[
                    "candidate_applied_object_count"
                ],
                "decision": result["decision"],
                "event_admitted_object_count": gate["event_admitted_object_count"],
                "sealed_prediction_or_exact_fallback_count": gate[
                    "sealed_prediction_or_exact_fallback_count"
                ],
                "source_outcome_authorized": gate["source_outcome_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
