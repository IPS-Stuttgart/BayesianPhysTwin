from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

RESULT = Path(
    "results/sota/pokeflex_conservative_shrinkage_source_v1/source_result.json"
)


def test_frozen_source_result_passes_registered_gates() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))

    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "0075c331fc23ffadb2e9ebdd4b58093c76d25ce39c2bcf33e84d80d50a338bda"
    )
    assert payload["source_gate_passed"]
    assert payload["take_count"] == 27
    assert payload["object_count"] == 9
    assert payload["selected_arm"] == (
        "checkpoint_action_local_state_relative_0.4_residual_scale_0.125"
    )
    assert payload["selected_result"]["object_win_count"] == 9
    assert payload["selected_result"][
        "object_balanced_relative_improvement"
    ] == pytest.approx(0.01286289679729027)
    assert payload["cross_fitted"]["stable_selection"]
    assert payload["cross_fitted"]["held_object_win_count"] == 9
    assert payload["fallback"]["unsupported_frame_count"] == 131
    assert payload["fallback"]["selected_arm_metric_mismatch_count"] == 0
    assert payload["target_objects_opened"] is False
