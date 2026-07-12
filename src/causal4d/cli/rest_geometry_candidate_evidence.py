"""CLI for extracting fit-only candidate evidence from Warp summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d.real_protocol import load_protocol
from causal4d.rest_geometry_cross_action import (
    build_candidate_evidence_from_case_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a pre-holdout candidate score artifact from one or more "
            "per-frame-mode rest-geometry Warp summaries."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("execution_id")
    parser.add_argument("output")
    parser.add_argument("summary", nargs="+")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    summaries = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.summary
    ]
    evidence = build_candidate_evidence_from_case_summaries(
        protocol,
        args.execution_id,
        summaries,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
