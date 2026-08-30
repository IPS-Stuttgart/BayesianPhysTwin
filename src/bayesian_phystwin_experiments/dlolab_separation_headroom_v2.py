"""Versioned custody wrapper for the DLO-Lab separation headroom screen."""

from __future__ import annotations

from typing import Any, cast

from .dlolab_separation_headroom_v1 import (
    ACTION_ANGLES_DEG,
    ACTION_NAMES,
    MEMORY_NAMES,
    NATIVE_MACROS,
    NATIVE_STEPS,
    NATIVE_STEPS_PER_MACRO,
    NUMERIC_REWARD_MARGIN_M,
    OBSERVED_NODES,
    PREFIX_FRAMES,
    PREFIX_MACROS,
    PREFIX_STEPS,
    UNIQUE_ACTION_COUNT,
    action_bank,
    development_metrics,
    native_qa,
    native_reward,
    task,
    worlds,
)
from .dlolab_separation_headroom_v1 import (
    protocol as protocol_v1,
)

PARENT_LAUNCH_FAILURE = {
    "schema": "dlolab-separation-headroom-launch-failure-v1",
    "source_revision": "6c3d6a78b183ae28ecb93f5ad2be209424aa339e",
    "registered_output_root": (
        "/home/fpfaff/source-only/dlolab-separation-headroom-development-v1"
    ),
    "root_created": True,
    "root_entry_count": 0,
    "lock_published": False,
    "worker_claims_published": 0,
    "native_worlds_started": 0,
    "failure_boundary": "runtime_identity_before_lock_publication",
    "reproduced_error": (
        "ValueError: registered CPU/software-rendering environment required"
    ),
    "retry_authorized": False,
    "protected_data_read": False,
}


def protocol() -> dict[str, Any]:
    """Return v1 science unchanged with corrected one-shot launch custody."""
    value = protocol_v1()
    value["schema"] = "dlolab-separation-headroom-development-v2"
    value["parent_launch_failure"] = dict(PARENT_LAUNCH_FAILURE)
    value["runtime_preflight_before_attempt_consumption"] = True
    value["external_write_once_attempt_ledger"] = True
    return cast(dict[str, Any], value)


__all__ = [
    "ACTION_ANGLES_DEG",
    "ACTION_NAMES",
    "MEMORY_NAMES",
    "NATIVE_MACROS",
    "NATIVE_STEPS",
    "NATIVE_STEPS_PER_MACRO",
    "NUMERIC_REWARD_MARGIN_M",
    "OBSERVED_NODES",
    "PARENT_LAUNCH_FAILURE",
    "PREFIX_FRAMES",
    "PREFIX_MACROS",
    "PREFIX_STEPS",
    "UNIQUE_ACTION_COUNT",
    "action_bank",
    "development_metrics",
    "native_qa",
    "native_reward",
    "protocol",
    "task",
    "worlds",
]
