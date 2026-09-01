#!/usr/bin/env python3
"""Build the compact public-real-data query-competence evidence record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.public_real_query_competence_v1 import (
    build_public_real_query_competence_evidence_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--deform360-result", type=Path, required=True)
    parser.add_argument("--tracking-protocol", type=Path, required=True)
    parser.add_argument("--tracking-metrics", type=Path, required=True)
    parser.add_argument("--tracking-specimen-scores", type=Path, required=True)
    parser.add_argument("--pokeflex-same-profile", type=Path, required=True)
    parser.add_argument("--pokeflex-independent-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol = load_strict_json_object(args.protocol, label="study protocol")
    result = build_public_real_query_competence_evidence_v1(
        protocol=protocol,
        deform360_result_path=args.deform360_result,
        tracking_protocol_path=args.tracking_protocol,
        tracking_metrics_path=args.tracking_metrics,
        tracking_specimen_scores_path=args.tracking_specimen_scores,
        pokeflex_same_profile_path=args.pokeflex_same_profile,
        pokeflex_independent_object_path=args.pokeflex_independent_object,
    )
    write_atomic_json(result, args.output, overwrite=args.overwrite)
    deform360 = result.get("deform360")
    tracking_cloth = result.get("tracking_cloth")
    if not isinstance(deform360, Mapping) or not isinstance(tracking_cloth, Mapping):
        raise RuntimeError("evidence builder returned malformed dataset summaries")
    summary = {
        "artifact_id": result["artifact_id"],
        "deform360": deform360["exact_context_policy"],
        "tracking_cloth": tracking_cloth["exact_context_policy"],
        "pokeflex": result["pokeflex"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
