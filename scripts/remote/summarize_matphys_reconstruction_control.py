#!/usr/bin/env python3
"""Write the locked MatPhys all-frame capacity decision from terminal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.matphys_reconstruction_control import (
    build_matphys_reconstruction_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit")
    parser.add_argument("terminal_checkpoint")
    parser.add_argument("export_manifest")
    parser.add_argument("released_phystwin_metrics")
    parser.add_argument("output_json")
    args = parser.parse_args()
    result = build_matphys_reconstruction_result(
        args.audit,
        args.terminal_checkpoint,
        args.export_manifest,
        args.released_phystwin_metrics,
    )
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "result_path": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
