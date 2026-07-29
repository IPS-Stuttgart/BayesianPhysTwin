#!/usr/bin/env python3
"""Run the frozen V14 estimator through mixed admission/physical custody."""

from __future__ import annotations

from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_prediction_v14 import (
    run_v14_reserve_prediction,
)

if __name__ == "__main__":
    raise SystemExit(run_v14_reserve_prediction(wrapper_path=Path(__file__)))
