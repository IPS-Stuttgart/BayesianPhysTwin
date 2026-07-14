"""Build the post-seal full-tactile contact upper-bound rope rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_predict import (
    build_target_oracle_tactile_rope_prediction,
    write_target_oracle_tactile_rope_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("held_out_prediction_seal_json")
    parser.add_argument("target_contact_oracle_json")
    parser.add_argument("shared_fit_json")
    parser.add_argument("prefix_geometry_json")
    parser.add_argument("output_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        artifacts = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in (
                args.held_out_prediction_seal_json,
                args.target_contact_oracle_json,
                args.shared_fit_json,
                args.prefix_geometry_json,
            )
        ]
        result = build_target_oracle_tactile_rope_prediction(
            args.processed_root,
            protocol,
            artifacts[0],
            artifacts[1],
            artifacts[2],
            artifacts[3],
        )
        write_target_oracle_tactile_rope_prediction(args.output_json, result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "result_sha256": result["result_sha256"],
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
