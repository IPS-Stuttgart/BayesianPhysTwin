#!/usr/bin/env python3
"""Generate or verify the retained PokeFlex source-target transfer result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bayesian_phystwin_experiments import (  # noqa: E402
    pokeflex_source_target_transfer_v1 as transfer,
)

DEFAULT_OUTPUT = (
    ROOT / "results/analysis/pokeflex_source_target_transfer_v1/result.json"
)


def _render() -> str:
    result = transfer.build_from_repository_root(ROOT)
    return json.dumps(
        result.to_record(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = _render()
    output = arguments.output
    if arguments.check:
        if not output.is_file():
            print(f"missing generated result: {output}", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != rendered:
            print(f"generated result is stale: {output}", file=sys.stderr)
            return 1
        print(output)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
