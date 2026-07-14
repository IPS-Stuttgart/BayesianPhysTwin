"""Reconstruct the contact-sealed Deform360 target rope prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_prefix import (
    build_target_prefix_rope_geometry,
    validate_target_prefix_rope_geometry,
    write_target_prefix_rope_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("prefix_mask_audit_json")
    parser.add_argument("output_archive_npz")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-sequence-json", action="append", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        mask_audit = json.loads(
            Path(args.prefix_mask_audit_json).read_text(encoding="utf-8")
        )
        source_sequences = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in args.source_sequence_json
        ]
        artifact = build_target_prefix_rope_geometry(
            args.processed_root,
            protocol,
            mask_audit,
            source_sequences,
            args.output_archive_npz,
        )
        write_target_prefix_rope_geometry(args.output_audit_json, artifact)
        validation = validate_target_prefix_rope_geometry(artifact)
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
