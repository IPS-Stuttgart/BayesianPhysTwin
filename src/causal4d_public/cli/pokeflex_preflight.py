"""Run an outcome-free compatibility audit on a PokeFlex dataset root."""

from __future__ import annotations

import argparse
import json

from causal4d_public.pokeflex import (
    load_readiness_config,
    preflight_pokeflex_dataset,
    validate_preflight_result,
    write_preflight_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = load_readiness_config(args.config)
        result = preflight_pokeflex_dataset(args.dataset_root, config)
        write_preflight_result(args.output_json, result)
        validation = validate_preflight_result(result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **validation,
                "capability_gates": result["capability_gates"],
                "split_counts": result["metadata_only_split"]["split_counts"],
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["preflight_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
