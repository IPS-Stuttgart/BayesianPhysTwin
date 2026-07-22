#!/usr/bin/env python3
"""Evaluate source-only PokeFlex RealSense selector artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_independent_depth_evaluation import (  # noqa: E402
    evaluate_locked_independent_depth_source_validation,
    load_and_evaluate_independent_depth_artifacts,
)
from bayesian_phystwin.pokeflex_independent_depth_protocol import (  # noqa: E402
    load_pokeflex_independent_depth_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact(result: dict[str, object]) -> None:
    for take in result.get("takes", []):
        take["competence"].pop("rows", None)
        take["selector"].pop("rows", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-anchor-improvement-mm", type=float, default=0.0)
    parser.add_argument(
        "--maximum-calibration-median-residual-mm", type=float, default=10.0
    )
    parser.add_argument(
        "--source-validation-protocol",
        type=Path,
        help="Evaluate the exact registered source panel and emit its gate decision",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit frame-level rows while retaining take/object summaries",
    )
    args = parser.parse_args()
    if args.source_validation_protocol is not None:
        payloads = [
            json.loads(path.resolve().read_text(encoding="utf-8"))
            for path in args.artifacts
        ]
        protocol = load_pokeflex_independent_depth_protocol(
            args.source_validation_protocol
        )["payload"]
        result = evaluate_locked_independent_depth_source_validation(
            payloads, protocol
        )
        result["sources"] = [
            {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for path in args.artifacts
        ]
    else:
        result = load_and_evaluate_independent_depth_artifacts(
            args.artifacts,
            minimum_anchor_improvement_mm=args.minimum_anchor_improvement_mm,
            maximum_calibration_median_residual_mm=(
                args.maximum_calibration_median_residual_mm
            ),
        )
        result["source_sha256"] = {
            str(path.resolve()): _sha256(path.resolve()) for path in args.artifacts
        }
    if args.compact:
        _compact(result)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing evaluation artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "competence": result["competence"],
                "object_balanced_selector": result["object_balanced_selector"],
                **(
                    {"registered_gate": result["registered_gate"]}
                    if "registered_gate" in result
                    else {}
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
