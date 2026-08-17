from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.material_trajectory_producer_v1 import (
    DrakeDeformableBodyReplayV1,
)


class _DrakeBody:
    def __init__(self, positions: np.ndarray, expected_context: object) -> None:
        self.positions = positions
        self.expected_context = expected_context

    def GetPositions(self, context: object) -> np.ndarray:
        assert context is self.expected_context
        return self.positions


def test_drake_replay_uses_default_synchronization_and_transposes_positions() -> None:
    context = object()
    advances: list[str] = []
    native = np.arange(15, dtype=np.float64).reshape(3, 5)
    replay = DrakeDeformableBodyReplayV1(
        deformable_body=_DrakeBody(native, context),
        plant_context_callback=lambda: context,
        advance_callback=lambda: advances.append("advance"),
        context="drake-scene",
    )

    assert replay.synchronize() is None
    portable = replay.get_material_positions_m()
    np.testing.assert_array_equal(portable, native.T)
    assert portable.shape == (5, 3)
    assert portable.flags.c_contiguous
    assert replay.context == "drake-scene"

    replay.step()
    assert advances == ["advance"]


def test_drake_replay_rejects_missing_surface_and_noncallable_callbacks() -> None:
    with pytest.raises(TypeError, match="must expose GetPositions"):
        DrakeDeformableBodyReplayV1(
            deformable_body=object(),
            plant_context_callback=lambda: object(),
            advance_callback=lambda: None,
        )

    context = object()
    body = _DrakeBody(np.zeros((3, 2), dtype=np.float64), context)
    with pytest.raises(TypeError, match="plant_context_callback must be callable"):
        DrakeDeformableBodyReplayV1(
            deformable_body=body,
            plant_context_callback=cast(Any, None),
            advance_callback=lambda: None,
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    (
        (np.zeros((2, 5), dtype=np.float64), r"shape \(3,N\)"),
        (np.zeros((3, 5), dtype=np.int64), "floating positions"),
        (
            np.array(
                [[0.0, np.nan], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float64,
            ),
            "non-finite positions",
        ),
    ),
)
def test_drake_replay_rejects_invalid_native_position_matrices(
    positions: np.ndarray,
    message: str,
) -> None:
    context = object()
    replay = DrakeDeformableBodyReplayV1(
        deformable_body=_DrakeBody(positions, context),
        plant_context_callback=lambda: context,
        advance_callback=lambda: None,
    )

    with pytest.raises(ValueError, match=message):
        replay.get_material_positions_m()
