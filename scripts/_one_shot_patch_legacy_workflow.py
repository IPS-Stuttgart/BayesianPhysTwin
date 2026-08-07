#!/usr/bin/env python3
"""Extend the legacy-artifact matrix over verified-payload propagation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/legacy-artifact-contract.yml"


def _replace(old: str, new: str, *, expected_count: int) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    WORKFLOW.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        """      - "src/bayesian_phystwin/legacy_artifacts.py"
      - "src/bayesian_phystwin/causal4d_artifacts_v1.py"
""",
        """      - "src/bayesian_phystwin/legacy_artifacts.py"
      - "src/bayesian_phystwin/phystwin_raw_cues.py"
      - "src/bayesian_phystwin/causal4d_artifacts_v1.py"
""",
        expected_count=2,
    )
    _replace(
        """      - "tests/test_legacy_artifacts.py"
      - "tests/test_causal4d_artifacts_v2.py"
""",
        """      - "tests/test_legacy_artifacts.py"
      - "tests/test_legacy_artifacts_snapshot.py"
      - "tests/test_causal4d_artifacts_v2.py"
      - "tests/test_phystwin_raw_cues.py"
""",
        expected_count=2,
    )
    _replace(
        'run: python -m pip install -e ".[dev]"',
        'run: python -m pip install -e ".[dev,graph]"',
        expected_count=2,
    )
    _replace(
        """          src/bayesian_phystwin/legacy_artifacts.py
          src/bayesian_phystwin/causal4d_artifacts_v1.py
""",
        """          src/bayesian_phystwin/legacy_artifacts.py
          src/bayesian_phystwin/phystwin_raw_cues.py
          src/bayesian_phystwin/causal4d_artifacts_v1.py
""",
        expected_count=3,
    )
    _replace(
        """          tests/test_legacy_artifacts.py
          tests/test_causal4d_artifacts_v2.py
""",
        """          tests/test_legacy_artifacts.py
          tests/test_legacy_artifacts_snapshot.py
          tests/test_causal4d_artifacts_v2.py
          tests/test_phystwin_raw_cues.py
""",
        expected_count=3,
    )


if __name__ == "__main__":
    main()
