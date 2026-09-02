#!/usr/bin/env python3
"""Install the common-comparator capsule into current CI manifests, then self-delete."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_MANIFEST = ROOT / ".github/quality/test-suites.json"
WORKFLOW = ROOT / ".github/workflows/install-query-probe-certificate-v2.yml"
SCRIPT = Path(__file__).resolve()
TEST_PATH = "tests/test_query_probe_certificate_v1.py"


def main() -> None:
    value = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    tests = value["suites"]["stable-core-coverage"]
    if TEST_PATH not in tests:
        anchor = "tests/test_query_quotient_belief_v1.py"
        try:
            index = tests.index(anchor) + 1
        except ValueError as exc:
            raise SystemExit(f"missing manifest anchor: {anchor}") from exc
        tests.insert(index, TEST_PATH)
    if tests.count(TEST_PATH) != 1:
        raise SystemExit("query-probe focused test is not registered exactly once")
    TEST_MANIFEST.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    WORKFLOW.unlink()
    SCRIPT.unlink()


if __name__ == "__main__":
    main()
