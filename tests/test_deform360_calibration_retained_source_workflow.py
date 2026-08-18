"""Exercise the retired retained-source launcher from archived bytes."""

from __future__ import annotations

from pathlib import Path

from tools.quality.retired_workflow_contract_tests import (
    expose_tests,
    load_retired_contract_test,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/github-actions/retired-one-shot-v1"
_ARCHIVED = load_retired_contract_test(
    archived_test=ARCHIVE / "contract-tests" / Path(__file__).name,
    original_test=Path(__file__).resolve(),
    replacements={
        "LAUNCHER": (ARCHIVE / "launch-deform360-calibration-retained-source-once.yml")
    },
)
expose_tests(globals(), _ARCHIVED)
