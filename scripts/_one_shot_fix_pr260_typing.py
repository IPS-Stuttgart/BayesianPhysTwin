#!/usr/bin/env python3
"""Apply explicit runtime narrowing to the visual-production plan parser."""

from __future__ import annotations

from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / "src/bayesian_phystwin/deform360_calibration_visual_production_plan.py"
)


def _replace(old: str, new: str, *, expected_count: int = 1) -> None:
    text = TARGET.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    TARGET.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        '''    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2,
        f"{name} must contain two integer bounds",
    )
''',
        '''    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain two integer bounds")
''',
    )
    _replace(
        '''    _require(isinstance(value, Mapping), "camera plan must be a JSON object")
    record = dict(value)
''',
        '''    if not isinstance(value, Mapping):
        raise ValueError("camera plan must be a JSON object")
    record = dict(value)
''',
    )


if __name__ == "__main__":
    main()
