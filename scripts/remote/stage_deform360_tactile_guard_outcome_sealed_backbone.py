#!/usr/bin/env python3
"""Stage one sealed V14 physical carrier for the tactile-guard protocol."""

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
    stage_v14_physical_backbone,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--physical-manifest", type=Path, required=True)
    parser.add_argument("--physical-archive", type=Path, required=True)
    parser.add_argument("--queue-rank", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage_v14_physical_backbone(
        args.output_dir,
        protocol_path=args.protocol,
        physical_manifest_path=args.physical_manifest,
        physical_archive_path=args.physical_archive,
        queue_rank=args.queue_rank,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
