#!/usr/bin/env python3
"""Stage one registered fresh PokeFlex archive without decoding geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_fresh12_staging import (  # noqa: E402
    stage_pokeflex_fresh12_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_conservative_shrinkage_fresh12_public_v1.json"
        ),
    )
    args = parser.parse_args()
    result = stage_pokeflex_fresh12_archive(
        args.archive,
        args.destination_root,
        args.protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
