from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "diagnose_deform360_pairwise_regret_guard.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_deform360_pairwise_regret_guard",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture() -> tuple[np.ndarray, ...]:
    frames = 76
    nodes = 4
    physical = np.zeros((frames, nodes, 3), dtype=np.float64)
    baseline = np.zeros_like(physical)
    candidate = np.zeros_like(physical)
    target = np.zeros_like(physical)
    visibility = np.ones((frames, nodes), dtype=bool)
    validity = np.ones((frames, nodes), dtype=bool)
    for frame in range(frames):
        physical[frame, :, 0] = frame * 0.001
    for update_index, update in enumerate(MODULE.UPDATE_FRAMES):
        stop = (
            MODULE.UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(MODULE.UPDATE_FRAMES)
            else frames
        )
        candidate[update + 1 : stop, :, 0] = 0.002
        target[update, :, 0] = 0.002
    return physical, baseline, candidate, target, visibility, validity


def test_privileged_dense_action_guard_admits_supported_intervals() -> None:
    physical, baseline, candidate, target, visibility, validity = _fixture()

    guarded, decisions = MODULE._apply_privileged_dense_action_guard(
        physical,
        baseline,
        candidate,
        target,
        visibility,
        validity,
    )

    assert len(decisions) == len(MODULE.UPDATE_FRAMES)
    assert all(decision["candidate_accepted"] for decision in decisions)
    for update_index, update in enumerate(MODULE.UPDATE_FRAMES):
        stop = (
            MODULE.UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(MODULE.UPDATE_FRAMES)
            else len(guarded)
        )
        np.testing.assert_array_equal(
            guarded[update + 1 : stop],
            candidate[update + 1 : stop],
        )


def test_privileged_dense_action_guard_rejection_is_exact_fallback() -> None:
    physical, baseline, candidate, target, visibility, validity = _fixture()
    target.fill(0.0)

    guarded, decisions = MODULE._apply_privileged_dense_action_guard(
        physical,
        baseline,
        candidate,
        target,
        visibility,
        validity,
    )

    assert all(not decision["candidate_accepted"] for decision in decisions)
    assert all(
        decision["bit_exact_baseline_fallback"] for decision in decisions
    )
    np.testing.assert_array_equal(guarded, baseline)
