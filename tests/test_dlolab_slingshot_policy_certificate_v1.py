from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_policy_certificate_v1 import (
    DEVELOPMENT_COUNT,
    leave_one_out_capacity_diagnostic,
)


def _development_fixture() -> dict[str, object]:
    observation = np.zeros((DEVELOPMENT_COUNT, 3, 4, 3), dtype=np.float64)
    observation[:, 0, 0, 0] = np.arange(DEVELOPMENT_COUNT)
    losses = np.tile(np.arange(7, dtype=np.float64), (DEVELOPMENT_COUNT, 1))
    losses[:, 5] = -1.0
    gains = np.zeros((DEVELOPMENT_COUNT, 7), dtype=np.float64)
    gains[:, 4] = np.linspace(-0.01, 0.02, DEVELOPMENT_COUNT)
    return {
        "case_ids": tuple(f"case-{index:02d}" for index in range(DEVELOPMENT_COUNT)),
        "observations": observation,
        "expected_losses": losses,
        "action_gains": gains,
    }


def test_capacity_diagnostic_is_explicitly_nonprospective() -> None:
    result = leave_one_out_capacity_diagnostic(**_development_fixture())

    assert result["status"] == "retrospective_leave_one_out_capacity_diagnostic_only"
    assert result["prospective_coverage_claim"] is False
    assert result["closed_288_world_panel_used"] is False
    assert result["case_count"] == DEVELOPMENT_COUNT
    assert len(result["neighbor_ids"]) == DEVELOPMENT_COUNT


def test_capacity_diagnostic_requires_complete_unique_denominator() -> None:
    fixture = _development_fixture()
    fixture["case_ids"] = tuple("same" for _ in range(DEVELOPMENT_COUNT))

    with pytest.raises(ValueError, match="complete finite"):
        leave_one_out_capacity_diagnostic(**fixture)


def test_committed_development_summary_is_canonical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "results/source/dlolab_slingshot_policy_certificate_development_v1/summary.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = result.pop("artifact_id")
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert expected == "ecf2a3375c38f31e2a371236ab6643083b9a12ffb9c48dfb292041ecad5e3bc0"
    assert hashlib.sha256(canonical).hexdigest() == expected
    assert result["accepted_count"] == 6
    assert result["harmful_guarded_count"] == 0
    assert result["closed_288_world_panel_used"] is False
