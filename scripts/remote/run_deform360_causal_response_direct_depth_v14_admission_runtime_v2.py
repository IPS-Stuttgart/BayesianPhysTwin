#!/usr/bin/env python3
"""Apply the checksum-only V14 amendment before source admission."""

from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_method_hash_runtime_v2 import (
    run_v14_method_hash_amended_parent,
)

if __name__ == "__main__":
    raise SystemExit(
        run_v14_method_hash_amended_parent(
            role="admission",
            wrapper_path=Path(__file__),
        )
    )
