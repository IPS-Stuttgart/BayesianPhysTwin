#!/usr/bin/env python3
"""Run the one frozen PokeFlex retrospective validation after source passage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin_experiments.pokeflex_query_competence_v1 import (
    bound_execution_path_v1,
    consume_stage_attempt_v1,
    file_sha256,
    load_protocol_v1,
    run_validation_stage_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("source_result", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-file-sha256", required=True)
    args = parser.parse_args()

    protocol = load_protocol_v1(args.protocol)
    expected_source = bound_execution_path_v1(protocol, stage="source", kind="result")
    if args.source_result.resolve() != expected_source:
        raise ValueError("PokeFlex source result path changed")
    if file_sha256(args.source_result) != args.expected_source_file_sha256:
        raise ValueError("PokeFlex source result file identity changed")
    consume_stage_attempt_v1(
        protocol,
        protocol_file_sha256=file_sha256(args.protocol),
        stage="validation",
        output=args.output,
        source_file_sha256=args.expected_source_file_sha256,
    )
    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    result = run_validation_stage_v1(protocol, source_result, args.artifact_root)
    write_atomic_json(result, args.output, overwrite=False)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": file_sha256(args.output),
                "primary_gate_passed": result["primary_gate_passed"],
                "prospective_confirmation": result["prospective_confirmation"],
                "retrospective_data": result["retrospective_data"],
                "validation_take_count": result["validation_take_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
