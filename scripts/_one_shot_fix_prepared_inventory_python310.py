#!/usr/bin/env python3
"""Use portable anonymous temporary files for verified parse snapshots."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/science/inventory_deform360_calibration_prepared_source.py"
DOC = ROOT / "docs/deform360_calibration_prepared_inventory.md"


def _replace(
    path: Path,
    old: str,
    new: str,
    *,
    expected_count: int = 1,
) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        SOURCE,
        "_SNAPSHOT_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024\n",
        "",
    )
    _replace(
        SOURCE,
        '''with tempfile.SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    ) as snapshot:''',
        '''with tempfile.TemporaryFile(mode="w+b") as snapshot:''',
        expected_count=3,
    )
    _replace(
        DOC,
        "Small\nsnapshots stay in memory and larger snapshots may spool to an anonymous temporary\nfile; neither case publishes local paths.\n",
        "Verified parse snapshots use anonymous temporary files on every supported Python\nversion; those files are never named in or retained by the published inventory.\n",
    )


if __name__ == "__main__":
    main()
