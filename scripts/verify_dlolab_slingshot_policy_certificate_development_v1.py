#!/usr/bin/env python3
"""Verify the opened-world Slingshot policy-certificate diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/source/dlolab_slingshot_policy_certificate_development_v1/summary.json"
BUILDER = ROOT / "scripts/audit_dlolab_slingshot_policy_certificate_development_v1.py"
EXPECTED_ARTIFACT_ID = "ecf2a3375c38f31e2a371236ab6643083b9a12ffb9c48dfb292041ecad5e3bc0"
EXPECTED_FILE_SHA256 = "914a5f1978eac4a088a0f67c187e2969acc6c158cb95fff5e542363f9be5135f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_id(value: dict[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key != "artifact_id"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    if _sha256(SUMMARY) != EXPECTED_FILE_SHA256:
        raise ValueError("summary file identity changed")
    if summary.get("artifact_id") != EXPECTED_ARTIFACT_ID:
        raise ValueError("summary artifact identity changed")
    if _canonical_id(summary) != EXPECTED_ARTIFACT_ID:
        raise ValueError("summary canonical identity is invalid")
    for name, expected in summary["source_sha256"].items():
        if _sha256(ROOT / name) != expected:
            raise ValueError(f"bound source changed: {name}")

    with tempfile.TemporaryDirectory() as directory:
        reproduced = Path(directory) / "summary.json"
        subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--source-root",
                str(args.source_root.resolve()),
                "--output",
                str(reproduced),
            ],
            cwd=ROOT,
            check=True,
        )
        if reproduced.read_bytes() != SUMMARY.read_bytes():
            raise ValueError("diagnostic does not reproduce byte-for-byte")
    print(
        json.dumps(
            {
                "artifact_id": EXPECTED_ARTIFACT_ID,
                "file_sha256": EXPECTED_FILE_SHA256,
                "verification_passed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
