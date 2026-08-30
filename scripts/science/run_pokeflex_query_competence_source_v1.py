#!/usr/bin/env python3
"""Fit and gate the PokeFlex risk score without opening validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin_experiments.pokeflex_query_competence_v1 import (
    file_sha256,
    load_protocol_v1,
    run_source_stage_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    protocol = load_protocol_v1(args.protocol)
    result = run_source_stage_v1(protocol, args.artifact_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError("existing PokeFlex source result differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": file_sha256(args.output),
                "protocol_sha256": protocol["protocol_sha256"],
                "source_gate_passed": result["source_gate_passed"],
                "validation_authorized": result["validation_authorized"],
                "validation_take_count_opened": result["validation_take_count_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
