#!/usr/bin/env python3
"""Analyze the locked exploratory CoTracker3 multiview-priority comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bayesian_phystwin.phystwin_multiview_priority_analysis as analysis_module
from bayesian_phystwin.phystwin_multiview_priority_analysis import (
    analyze_multiview_priority_results,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()

    result = analyze_multiview_priority_results(
        _load_json(args.source_result),
        _load_json(args.candidate_result),
        _load_json(args.protocol),
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in (
            ("source_result", args.source_result),
            ("candidate_result", args.candidate_result),
            ("protocol", args.protocol),
        )
    }
    module_path = Path(str(analysis_module.__file__)).resolve()
    script_path = Path(__file__).resolve()
    result["analysis_software"] = {
        "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        "script": {"path": str(script_path), "sha256": _sha256(script_path)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "aggregate": result["aggregate"]["metrics"],
                "gates": result["gates"],
                "fresh_evaluation_justified": (
                    result["fresh_evaluation_justified"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
