#!/usr/bin/env python3
"""Apply explicit float narrowing for the visual-admission parser."""

from __future__ import annotations

from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py"
)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = """    result = float(value)\n    if not math.isfinite(result) or result <= 0.0:\n"""
    new = """    result: float = float(value)\n    if not math.isfinite(result) or result <= 0.0:\n"""
    if text.count(old) != 1:
        raise SystemExit("positive-number parser shape changed")
    TARGET.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
