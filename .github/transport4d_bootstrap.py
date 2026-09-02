#!/usr/bin/env python3
"""Materialize the reviewed Transport4D source bundle on its staging branch."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path


def main() -> None:
    root = Path.cwd().resolve()
    part_root = root / ".github" / "transport4d_bundle"
    parts = sorted(part_root.glob("part-*.txt"))
    if len(parts) != 4:
        raise SystemExit("Transport4D bootstrap bundle is incomplete")
    encoded = "".join(path.read_text(encoding="utf-8") for path in parts)
    payload = json.loads(
        zlib.decompress(base64.b85decode(encoded)).decode("utf-8")
    )
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("invalid Transport4D bootstrap payload")
    for relative, content in sorted(payload.items()):
        if not isinstance(relative, str) or not isinstance(content, str):
            raise SystemExit("invalid Transport4D bootstrap entry")
        path = (root / relative).resolve()
        if root not in path.parents or path == root:
            raise SystemExit(f"unsafe Transport4D path: {relative}")
        if path.exists():
            raise SystemExit(f"Transport4D target already exists: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
