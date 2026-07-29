#!/usr/bin/env python3
"""Run V14 with the registered sparse-spatial crash-to-rejection fix."""

from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_spatial_support_runtime_v2 import (
    run_v14_sparse_spatial_support_prediction,
)

if __name__ == "__main__":
    raise SystemExit(
        run_v14_sparse_spatial_support_prediction(wrapper_path=Path(__file__))
    )
