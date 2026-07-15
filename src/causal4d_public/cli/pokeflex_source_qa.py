"""Run the locked source-only PokeFlex geometry and intervention audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.pokeflex_source_qa import (
    load_source_qa_policy,
    run_pokeflex_source_qa,
    validate_source_qa_artifact,
    write_source_qa_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("preflight_json")
    parser.add_argument("output_json")
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    try:
        preflight = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
        config = load_source_qa_policy(args.policy)
        result = run_pokeflex_source_qa(args.dataset_root, preflight, config)
        write_source_qa_artifact(args.output_json, result)
        validation = validate_source_qa_artifact(result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **validation,
                "capability_gates": result["capability_gates"],
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["source_qa_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
