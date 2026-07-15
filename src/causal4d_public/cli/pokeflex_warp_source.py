"""Run the locked development-only PokeFlex official-Warp backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.pokeflex_warp_source import (
    load_warp_policy,
    run_pokeflex_warp_source_backend,
    validate_warp_artifact,
    write_warp_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("source_qa_json")
    parser.add_argument("official_phystwin_repo")
    parser.add_argument("output_json")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        source_qa = json.loads(
            Path(args.source_qa_json).read_text(encoding="utf-8")
        )
        config = load_warp_policy(args.policy)
        result = run_pokeflex_warp_source_backend(
            args.dataset_root,
            source_qa,
            args.official_phystwin_repo,
            config,
            device=args.device,
        )
        write_warp_artifact(args.output_json, result)
        validation = validate_warp_artifact(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **validation,
                "gates": result["gates"],
                "pooling_controls": {
                    "leave_one_out_persistence_win_fraction": result[
                        "pooling_controls"
                    ]["leave_one_out_persistence_win_fraction"],
                    "pooled_vs_single_source_win_fraction": result[
                        "pooling_controls"
                    ]["pooled_vs_single_source_win_fraction"],
                },
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["source_backend_admitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
