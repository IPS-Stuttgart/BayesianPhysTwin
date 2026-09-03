from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_audit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "science"
        / "audit_deform360_dependence_matched_coverage_v1.py"
    )
    spec = importlib.util.spec_from_file_location("matched_coverage_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matched_counts_and_better_ranking() -> None:
    audit = load_audit_module()
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_)
    full = np.asarray(
        [0.05, 0.90, 0.10, 0.80, 0.15, 0.70, 0.20, 0.60, 0.25, 0.30]
    )
    weak = np.asarray(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    )
    grid = (0.2, 0.4, 0.6, 0.8)
    full_result = audit.matched_curve(full, labels, grid, 0.1)
    weak_result = audit.matched_curve(weak, labels, grid, 0.1)
    assert full_result["normalized_selective_risk_auc"] < weak_result[
        "normalized_selective_risk_auc"
    ]
    assert [point["accepted_count"] for point in full_result["curve"]] == [
        point["accepted_count"] for point in weak_result["curve"]
    ]


def test_stable_time_index_tie_break() -> None:
    audit = load_audit_module()
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool_)
    result = audit.matched_curve(np.zeros(10), labels, (0.2, 0.4), 0.1)
    expected = audit.array_digest(np.asarray([0, 1], dtype=np.int64))
    assert result["curve"][0]["accepted_index_sha256"] == expected
