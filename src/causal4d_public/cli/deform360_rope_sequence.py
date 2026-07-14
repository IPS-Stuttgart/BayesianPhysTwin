"""Extract one source Deform360 rope centerline sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import (
    load_deform360_protocol_config,
    validate_deform360_preflight,
)
from causal4d_public.deform360_rope_sequence import (
    RopeCenterlineSequenceConfig,
    run_source_rope_centerline_sequence,
    validate_rope_sequence_artifact,
    write_rope_sequence_artifact,
)
from causal4d_public.deform360_sam2_views import load_sam2_view_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("episode_index", type=int)
    parser.add_argument("source_view_audit_json")
    parser.add_argument("preflight_json")
    parser.add_argument("output_archive_npz")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mask-audit-json", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--minimum-sync-reliability", type=float, default=0.85)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        view_audit = load_sam2_view_audit(args.source_view_audit_json)
        mask_audit = json.loads(Path(args.mask_audit_json).read_text(encoding="utf-8"))
        preflight = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
        validate_deform360_preflight(preflight)
        sequence_config = RopeCenterlineSequenceConfig(
            source_episode_index=args.episode_index,
            frame_start=args.frame_start,
            frame_stop_exclusive=args.frame_stop,
            frame_stride=args.frame_stride,
            minimum_synchronization_reliability=(args.minimum_sync_reliability),
        )
        result = run_source_rope_centerline_sequence(
            args.processed_root,
            protocol,
            view_audit,
            mask_audit,
            preflight,
            args.output_archive_npz,
            sequence_config=sequence_config,
        )
        write_rope_sequence_artifact(args.output_audit_json, result)
        validation = validate_rope_sequence_artifact(result)
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
