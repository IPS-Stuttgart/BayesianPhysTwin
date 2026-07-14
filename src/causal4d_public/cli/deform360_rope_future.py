"""Reconstruct target-future rope geometry after predictions are sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_future import (
    build_target_future_rope_geometry,
    validate_target_future_rope_geometry,
    write_target_future_rope_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("held_out_prediction_seal_json")
    parser.add_argument("prefix_geometry_json")
    parser.add_argument("suffix_mask_audit_json")
    parser.add_argument("output_archive_npz")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        artifacts = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in (
                args.held_out_prediction_seal_json,
                args.prefix_geometry_json,
                args.suffix_mask_audit_json,
            )
        ]
        result = build_target_future_rope_geometry(
            args.processed_root,
            protocol,
            artifacts[0],
            artifacts[1],
            artifacts[2],
            args.output_archive_npz,
        )
        write_target_future_rope_geometry(args.output_audit_json, result)
        validation = validate_target_future_rope_geometry(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {**validation, "output": args.output_audit_json},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
