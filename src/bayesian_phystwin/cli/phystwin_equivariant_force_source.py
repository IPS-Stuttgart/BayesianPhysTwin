"""Build and cross-fit the source-only equivariant-force experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_equivariant_force_gate import (
    evaluate_equivariant_force_official_warp_gate,
    write_equivariant_force_official_warp_gate,
)
from bayesian_phystwin.phystwin_equivariant_force_source import (
    build_equivariant_force_source_episodes,
    load_equivariant_force_source_protocol,
    run_equivariant_force_source_competence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-episodes")
    build.add_argument("data_root")
    build.add_argument("protocol_json")
    build.add_argument("output_dir")
    gate = subparsers.add_parser("source-competence")
    gate.add_argument("episode_root")
    gate.add_argument("protocol_json")
    gate.add_argument("output_dir")
    gate.add_argument("--device")
    decide = subparsers.add_parser("official-warp-gate")
    decide.add_argument("records_json")
    decide.add_argument("competence_summary_json")
    decide.add_argument("protocol_json")
    decide.add_argument("output_json")
    args = parser.parse_args()

    if args.command == "build-episodes":
        result = build_equivariant_force_source_episodes(
            args.data_root,
            args.protocol_json,
            args.output_dir,
        )
    elif args.command == "source-competence":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("source competence requires torch") from error
        result = run_equivariant_force_source_competence(
            args.episode_root,
            args.protocol_json,
            args.output_dir,
            torch,
            device=args.device,
        )
    else:
        records_payload = json.loads(
            Path(args.records_json).read_text(encoding="utf-8")
        )
        competence = json.loads(
            Path(args.competence_summary_json).read_text(encoding="utf-8")
        )
        result = evaluate_equivariant_force_official_warp_gate(
            records_payload["records"],
            load_equivariant_force_source_protocol(args.protocol_json),
            force_target_competence_passed=bool(
                competence["force_target_competence_passed"]
            ),
        )
        result["artifact"] = write_equivariant_force_official_warp_gate(
            args.output_json,
            result,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
