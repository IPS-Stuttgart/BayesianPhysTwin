#!/usr/bin/env python3
"""Hosted-workflow path shim for the compression experiment."""

from pathlib import Path
import runpy


_TARGET = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "science"
    / "run_deform360_decision_sufficient_compression_v1.py"
)
runpy.run_path(str(_TARGET), run_name="__main__")
