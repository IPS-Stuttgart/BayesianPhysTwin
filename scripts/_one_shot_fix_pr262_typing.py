#!/usr/bin/env python3
"""Prepare the exact validated PR 262 source tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGET = (
    ROOT
    / "src/bayesian_phystwin/deform360_calibration_visual_execution_admission.py"
)
TEST_TARGET = ROOT / "tests/test_deform360_calibration_visual_execution_admission.py"


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{name} shape changed")
    return text.replace(old, new)


def main() -> None:
    source = SOURCE_TARGET.read_text(encoding="utf-8")
    source = _replace_once(
        source,
        """    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
""",
        """    result: float = float(value)
    if not math.isfinite(result) or result <= 0.0:
""",
        name="positive-number parser",
    )
    SOURCE_TARGET.write_text(source, encoding="utf-8")

    tests = TEST_TARGET.read_text(encoding="utf-8")
    tests = _replace_once(
        tests,
        """    for plan_object in plan[\"objects\"]:
        object_id = plan_object[\"object_id\"]
        cameras = []
""",
        """    for plan_object in plan[\"objects\"]:
        object_id = plan_object[\"object_id\"]
        selected_stop = plan_object[\"selected_source_frame_range_half_open\"][1]
        if type(selected_stop) is not int:
            raise AssertionError(\"selected frame stop must be an integer\")
        aligned_frame_count = max(140, selected_stop)
        cameras = []
""",
        name="visual-admission inventory fixture header",
    )
    tests = _replace_once(
        tests,
        '                    "frame_count": 140,\n',
        '                    "frame_count": aligned_frame_count,\n',
        name="camera frame-count fixture",
    )
    tests = _replace_once(
        tests,
        '                "aligned_frame_count": 140,\n',
        '                "aligned_frame_count": aligned_frame_count,\n',
        name="inventory aligned-frame-count fixture",
    )
    tests = _replace_once(
        tests,
        '                    "raw_frame_count": 140,\n',
        '                    "raw_frame_count": aligned_frame_count,\n',
        name="inventory raw-frame-count fixture",
    )
    TEST_TARGET.write_text(tests, encoding="utf-8")


if __name__ == "__main__":
    main()
