#!/usr/bin/env python3
"""Finalize the mixed-prelock V14 source panel without outcome access."""

from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_source_finalizer_v14 import (
    run_v14_reserve_source_finalizer,
)

if __name__ == "__main__":
    raise SystemExit(run_v14_reserve_source_finalizer(wrapper_path=Path(__file__)))
