"""Seal visual-only and tactile-conditioned Deform360 rope predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_predict import (
    build_and_seal_target_rope_predictions,
    write_target_rope_prediction_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("contact_prediction_seal_json")
    parser.add_argument("shared_fit_json")
    parser.add_argument("prefix_geometry_json")
    parser.add_argument("prediction_archive_npz")
    parser.add_argument("prediction_seal_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        artifacts = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in (
                args.contact_prediction_seal_json,
                args.shared_fit_json,
                args.prefix_geometry_json,
            )
        ]
        seal = build_and_seal_target_rope_predictions(
            args.processed_root,
            protocol,
            artifacts[0],
            artifacts[1],
            artifacts[2],
            args.prediction_archive_npz,
        )
        write_target_rope_prediction_seal(args.prediction_seal_json, seal)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "result_sha256": seal["result_sha256"],
                "prediction_shape": seal["prediction_shape"],
                "output": args.prediction_seal_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
