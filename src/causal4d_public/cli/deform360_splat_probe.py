"""Run the source-only Deform360 thin-rope Splatfacto probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import (
    load_deform360_protocol_config,
    validate_deform360_preflight,
)
from causal4d_public.deform360_sam2_views import load_sam2_view_audit
from causal4d_public.deform360_splat_probe import (
    ThinRopeSplatProbeConfig,
    run_source_splat_probe,
    validate_splat_probe_artifact,
    write_splat_probe_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("source_view_audit_json")
    parser.add_argument("preflight_json")
    parser.add_argument("output_dir")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--minimum-sync-reliability", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        view_audit = load_sam2_view_audit(args.source_view_audit_json)
        preflight = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
        validate_deform360_preflight(preflight)
        probe_config = ThinRopeSplatProbeConfig(
            training_iterations=args.iterations,
            minimum_synchronization_reliability=(args.minimum_sync_reliability),
        )
        result = run_source_splat_probe(
            args.processed_root,
            protocol,
            view_audit,
            preflight,
            args.output_dir,
            config=probe_config,
            overwrite=args.overwrite,
        )
        write_splat_probe_artifact(args.output_audit_json, result)
        validation = validate_splat_probe_artifact(result)
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
