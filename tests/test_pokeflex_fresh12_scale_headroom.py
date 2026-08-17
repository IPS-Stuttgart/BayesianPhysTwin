import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "development" / "audit_pokeflex_fresh12_scale_headroom.py"


def _module():
    spec = importlib.util.spec_from_file_location("fresh12_scale_headroom", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_bank_requires_exact_fallback_and_sealed_scale() -> None:
    module = _module()

    assert module._parse_multipliers("2,0,1,0.5") == (0.0, 0.5, 1.0, 2.0)
    with pytest.raises(ValueError, match="zero and one"):
        module._parse_multipliers("0,2")
    with pytest.raises(ValueError, match="repeats"):
        module._parse_multipliers("0,1,1")


def test_scale_summary_counts_wins_ties_and_losses() -> None:
    module = _module()
    rows = [
        {"mean_CD_UL1_mm_by_multiplier": {"0.0": 2.0, "1.0": 1.0}},
        {"mean_CD_UL1_mm_by_multiplier": {"0.0": 2.0, "1.0": 2.0}},
        {"mean_CD_UL1_mm_by_multiplier": {"0.0": 2.0, "1.0": 3.0}},
    ]

    summary = module._summarize(rows, (0.0, 1.0))[1]

    assert summary["object_win_count"] == 1
    assert summary["object_tie_count"] == 1
    assert summary["object_loss_count"] == 1
    assert summary["object_balanced_relative_improvement"] == 0.0
