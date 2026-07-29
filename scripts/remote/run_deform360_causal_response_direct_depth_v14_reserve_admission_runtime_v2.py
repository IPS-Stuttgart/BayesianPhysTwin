#!/usr/bin/env python3
"""Apply the checksum-only repair before V14 reserve source admission."""

from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_admission_runtime_v2 import (
    run_v14_reserve_admission_runtime_v2,
)

if __name__ == "__main__":
    raise SystemExit(run_v14_reserve_admission_runtime_v2(wrapper_path=Path(__file__)))
