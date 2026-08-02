#!/usr/bin/env python3
"""Build one frozen tactile-guarded prediction without opening its outcome."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (  # noqa: E402
    build_guarded_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--tactile-features", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_guarded_prediction(
        args.output_dir,
        repository_root=args.repo,
        protocol_path=args.protocol,
        backbone_dir=args.backbone_case_dir,
        measurement_dir=args.measurement_dir,
        tactile_feature_path=args.tactile_features,
        source_result_path=args.source_result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
