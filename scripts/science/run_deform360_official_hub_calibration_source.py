#!/usr/bin/env python3
"""Open and prepare only the locked Deform360 calibration source objects."""

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.science.deform360_calibration_source.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
