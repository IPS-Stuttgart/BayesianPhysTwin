#!/usr/bin/env python3
"""Run the frozen Cloth Sim2Real covariance study with its dataset lock.

The original experiment implementation was reviewed with a malformed, truncated
archive identifier. This launcher binds the authoritative Zenodo SHA-256 before
calling that implementation and fails closed if the implementation no longer
contains either the known malformed value or the corrected value.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

EXPECTED_DATASET_SHA256: Final = (
    "268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f"
)
LEGACY_MALFORMED_DATASET_SHA256: Final = (
    "268d07d90da770278106028b3c704340bbac48dbd03bb4afd563630fb6de7ec"
)
IMPLEMENTATION_PATH: Final = Path(__file__).with_name(
    "run_cloth_sim2real_prob4d_covariance_ablation_v1.py"
)


def _load_implementation() -> ModuleType:
    if len(EXPECTED_DATASET_SHA256) != 64 or any(
        character not in "0123456789abcdef" for character in EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError("authoritative dataset SHA-256 is malformed")
    spec = importlib.util.spec_from_file_location(
        "_cloth_sim2real_prob4d_covariance_ablation_v1",
        IMPLEMENTATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the covariance-ablation implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    recorded = getattr(module, "DATASET_SHA256", None)
    if recorded not in {
        EXPECTED_DATASET_SHA256,
        LEGACY_MALFORMED_DATASET_SHA256,
    }:
        raise RuntimeError(
            "implementation dataset identity differs from the reviewed lock"
        )
    module.__dict__["DATASET_SHA256"] = EXPECTED_DATASET_SHA256
    return module


_IMPLEMENTATION = _load_implementation()

for _export_name in dir(_IMPLEMENTATION):
    if _export_name.startswith("__") or _export_name in {
        "DATASET_SHA256",
        "main",
    }:
        continue
    globals()[_export_name] = getattr(_IMPLEMENTATION, _export_name)

DATASET_SHA256: Final = EXPECTED_DATASET_SHA256


def main() -> int:
    """Run the reviewed implementation with the authoritative dataset lock."""

    return int(_IMPLEMENTATION.main())


if __name__ == "__main__":
    raise SystemExit(main())
