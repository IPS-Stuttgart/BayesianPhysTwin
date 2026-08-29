from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.dlolab_wrapping_certified_guard_v9 import (
    clopper_pearson_upper,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_certified_guard_calibration_v9",
    ROOT / "scripts/audit_dlolab_wrapping_certified_guard_calibration_v9.py",
)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def test_exact_binomial_upper_bound_matches_registered_calibration() -> None:
    assert clopper_pearson_upper(2, 144, confidence=0.95) == pytest.approx(
        0.04307319681566585,
        abs=1e-15,
    )
    assert clopper_pearson_upper(0, 144, confidence=0.95) == pytest.approx(
        0.020588792299495895,
        abs=1e-15,
    )


@pytest.mark.parametrize(
    ("harm_count", "world_count", "confidence"),
    [(-1, 144, 0.95), (145, 144, 0.95), (0, 0, 0.95), (0, 144, 1.0)],
)
def test_invalid_binomial_inputs_are_rejected(
    harm_count: int,
    world_count: int,
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="binomial confidence-bound"):
        clopper_pearson_upper(harm_count, world_count, confidence=confidence)


def test_v8_calibration_certificate_is_exact_and_does_not_reclassify_v8() -> None:
    result = audit_module.audit()
    assert result["certificate_passed"]
    assert result["harm_count"] == 2
    assert result["one_sided_exact_clopper_pearson_upper"] < 0.05
    assert result["v8_strict_source_gate_passed"] is False
    assert result["v8_strict_gate_reclassified"] is False
    assert result["candidate_threshold_selected_from_v8_outcomes"] is False
    assert result["lead_is_not_v9_evidence"]
    assert result["v9_fresh_replication_automatically_authorized"] is False
