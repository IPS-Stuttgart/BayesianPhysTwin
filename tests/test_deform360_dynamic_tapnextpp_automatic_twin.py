from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/remote/build_deform360_dynamic_tapnextpp_automatic_twin.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform360_dynamic_tapnextpp_automatic_twin",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_protocol_snapshot_binds_candidate_coordinates() -> None:
    snapshot = MODULE._protocol_snapshot(
        {
            "state_update": {
                "physical_backbone": "selected physical backbone",
                "candidate_coordinates": "recursive readout discrepancy",
            }
        }
    )

    assert snapshot == {
        "physical_backbone": "selected physical backbone",
        "candidate_coordinates": "recursive readout discrepancy",
    }


def test_protocol_snapshot_rejects_obsolete_state_coordinates() -> None:
    with pytest.raises(ValueError, match="candidate-coordinate"):
        MODULE._protocol_snapshot(
            {
                "state_update": {
                    "physical_backbone": "selected physical backbone",
                    "state_coordinates": "obsolete key",
                }
            }
        )
