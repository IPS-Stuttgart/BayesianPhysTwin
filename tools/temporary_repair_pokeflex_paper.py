#!/usr/bin/env python3
"""Finish the temporary deterministic repair for PokeFlex paper-artifact CI."""

from __future__ import annotations

from pathlib import Path


def validate_isolated_artifact_runtime() -> None:
    path = Path(".github/workflows/pokeflex-same-object-paper-artifacts.yml")
    text = path.read_text(encoding="utf-8")
    marker = "\n  artifacts:\n"
    if text.count(marker) != 1:
        raise SystemExit("unexpected paper-artifact job boundary")
    artifact_job = text.split(marker, 1)[1]
    if "actions/setup-python" in artifact_job:
        raise SystemExit("self-hosted artifact job still uses setup-python")
    required = (
        "Initialize isolated artifact paths and Python",
        "Validate frozen input custody before runtime installation",
        "Create isolated released-checkpoint runtime",
        '"${POKEFLEX_BOOTSTRAP_PYTHON}" -m venv --clear "${POKEFLEX_VENV}"',
        '"${POKEFLEX_VENV}/bin/python" -m pip check',
        "--no-cache-dir",
    )
    missing = [term for term in required if term not in artifact_job]
    if missing:
        raise SystemExit(f"isolated artifact runtime is incomplete: {missing}")


def repair_core_coverage() -> None:
    path = Path(".github/workflows/tests.yml")
    lines = path.read_text(encoding="utf-8").splitlines()
    target = "tests/test_pokeflex_same_object_reporting.py"
    target_count = sum(line.strip().rstrip(" \\") == target for line in lines)
    if target_count == 0:
        inserted = 0
        index = 0
        while index < len(lines):
            if lines[index].strip().rstrip(" \\") == "tests/test_quality_invariants.py":
                indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
                if not lines[index].rstrip().endswith("\\"):
                    lines[index] = lines[index].rstrip() + " \\"
                lines.insert(index + 1, indent + target)
                inserted += 1
                index += 1
            index += 1
        if inserted != 2:
            raise SystemExit(
                f"expected two stable/core reporting-test insertions, found {inserted}"
            )
    final_count = sum(line.strip().rstrip(" \\") == target for line in lines)
    if final_count != 2:
        raise SystemExit(
            f"PokeFlex reporting test must run in both core lists, found {final_count}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    validate_isolated_artifact_runtime()
    repair_core_coverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
