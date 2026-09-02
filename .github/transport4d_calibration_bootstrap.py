#!/usr/bin/env python3
"""Materialize the reviewed Transport4D calibration bundle."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

ENCODED_SHA256 = "13b2d1bd083bfa1d19c3a061e6c0d4b80ef2ce0bf336cb3af4616a77eb9e9027"


def main() -> None:
    root = Path.cwd().resolve()
    part_root = root / ".github" / "transport4d_calibration_bundle_b64"
    parts = sorted(part_root.glob("part-*.txt"))
    if len(parts) != 7:
        raise SystemExit("Transport4D calibration bootstrap bundle is incomplete")
    encoded = "".join(path.read_text(encoding="utf-8") for path in parts)
    if hashlib.sha256(encoded.encode("ascii")).hexdigest() != ENCODED_SHA256:
        raise SystemExit("Transport4D calibration bootstrap bundle digest mismatch")
    payload = json.loads(
        zlib.decompress(base64.b64decode(encoded, validate=True)).decode("utf-8")
    )
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("invalid Transport4D calibration bootstrap payload")
    for relative, content in sorted(payload.items()):
        if not isinstance(relative, str) or not isinstance(content, str):
            raise SystemExit("invalid Transport4D calibration bootstrap entry")
        path = (root / relative).resolve()
        if root not in path.parents or path == root:
            raise SystemExit(f"unsafe Transport4D calibration path: {relative}")
        if path.exists():
            raise SystemExit(f"Transport4D calibration target already exists: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
