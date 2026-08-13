#!/usr/bin/env python3
"""Analyze the frozen MatPhys/CoTracker selected-overlay source report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bayesian_phystwin.phystwin_matphys_cotracker_selected_overlay_analysis import (
    analyze_matphys_cotracker_selected_overlay_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = analyze_matphys_cotracker_selected_overlay_report(report)
    result["input"] = {
        "path": str(report_path),
        "sha256": _sha256(report_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
