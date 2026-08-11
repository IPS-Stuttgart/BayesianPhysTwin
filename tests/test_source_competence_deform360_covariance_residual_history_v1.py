"""Register residual-history contracts in stable-core coverage."""

from pathlib import Path

from scripts.science.run_deform360_covariance_residual_history_dry_run_v1 import (
    load_locked_policy,
)
from test_deform360_covariance_residual_history_real_camera_roster_v1 import *  # noqa: F403
from test_deform360_covariance_residual_history_v1 import *  # noqa: F403

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_covariance_residual_history_dry_run_v1.json"
)


def test_source_only_lock_does_not_bind_or_authorize_a_target_roster() -> None:
    protocol, _ = load_locked_policy(PROTOCOL)

    assert "target_binding" not in protocol
    target = protocol["target_boundary"]
    assert target["roster_and_custody_owned_by"] == (
        "separately reviewed target protocol"
    )
    assert target["target_roster_identity_bound_here"] is False
    assert target["target_payload_access_authorized"] is False
    assert target["target_prediction_authorized"] is False
    assert target["target_outcome_access_authorized"] is False
