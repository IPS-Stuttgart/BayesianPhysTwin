"""Fit and cross-validate shared Deform360 rope dynamics on source actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_fit import (
    build_forward_rope_fit_artifact,
    validate_forward_rope_fit_artifact,
    write_forward_rope_fit_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_json", nargs="+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        protocol = load_deform360_protocol_config(args.config)
        observations = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in args.observation_json
        ]
        artifact = build_forward_rope_fit_artifact(protocol, observations)
        write_forward_rope_fit_artifact(args.output, artifact)
        validation = validate_forward_rope_fit_artifact(artifact)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps({**validation, "output": args.output}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
