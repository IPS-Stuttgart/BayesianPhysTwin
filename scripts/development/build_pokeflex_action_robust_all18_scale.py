#!/usr/bin/env python3
"""Build the source-only all-18 PokeFlex robust-scale extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_all18 import (  # noqa: E402
    build_all18_calibration,
    load_all18_source_protocol,
    load_source_artifacts,
    validate_all18_calibration,
    validate_all18_source_protocol,
)
from bayesian_phystwin.pokeflex_action_robust_scale import (  # noqa: E402
    load_action_robust_scale_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_all18_source_v4.json"
        ),
    )
    parser.add_argument(
        "--parent-calibration",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_scale_v3.json"
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace calibration: {args.output}")
    protocol = load_all18_source_protocol(args.protocol)
    validation = validate_all18_source_protocol(protocol)
    parent = load_action_robust_scale_calibration(args.parent_calibration)
    artifacts, digests = load_source_artifacts(
        args.source_artifact_root,
        validation["selected_take_ids"],
    )
    calibration = build_all18_calibration(
        parent,
        protocol,
        artifacts,
        source_artifact_file_sha256s=digests,
    )
    validate_all18_calibration(calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration_sha256": calibration["calibration_sha256"],
                "source_gate": calibration["source_gate"],
                "new_multipliers": {
                    name: row["multiplier"]
                    for name, row in calibration["new_objects"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
