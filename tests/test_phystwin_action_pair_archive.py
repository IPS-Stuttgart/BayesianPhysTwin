from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    PHYSICAL_ARRAY_NAMES,
)
from bayesian_phystwin.phystwin_action_pair_archive import (
    PHYSTWIN_ACTION_PAIR_CONTRACT,
    PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M,
    build_phystwin_action_pair_arrays,
    phystwin_graph_action_support,
    write_phystwin_action_pair_archive,
)


def _graph() -> tuple[np.ndarray, np.ndarray]:
    springs = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    rest_lengths = np.array([0.1, 0.1, 0.02], dtype=np.float32)
    return springs, rest_lengths


def test_graph_action_support_decays_from_controller_attachment() -> None:
    springs, lengths = _graph()

    support = phystwin_graph_action_support(
        springs,
        lengths,
        object_point_count=3,
        object_spring_count=2,
    )

    expected = np.exp(
        -np.array([0.2, 0.1, 0.0]) / PHYSTWIN_ACTION_SUPPORT_LENGTH_SCALE_M
    )
    np.testing.assert_allclose(support, expected, rtol=1e-6)


def test_action_pair_archive_preserves_real_replays(tmp_path: Path) -> None:
    springs, lengths = _graph()
    frame_zero = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    driven = np.repeat(frame_zero[None], 4, axis=0)
    held = driven.copy()
    driven[1:, :, 1] += np.array([0.01, 0.02, 0.03])[:, None]
    held[1:, :, 2] -= 0.001

    arrays = build_phystwin_action_pair_arrays(
        driven,
        held,
        springs=springs,
        rest_lengths_m=lengths,
        object_spring_count=2,
    )

    assert set(arrays) == PHYSICAL_ARRAY_NAMES
    assert np.array_equal(arrays["prediction_m"], driven)
    assert np.array_equal(arrays["driven_readout_m"], driven)
    assert np.array_equal(arrays["zero_action_readout_m"], held)
    assert np.array_equal(
        arrays["persistence_m"], np.repeat(frame_zero[None], 4, axis=0)
    )

    manifest = write_phystwin_action_pair_archive(tmp_path / "pair.npz", arrays)
    assert manifest["contract"] == PHYSTWIN_ACTION_PAIR_CONTRACT
    assert manifest["action_support"]["residual_independent"] is True
    with np.load(manifest["path"], allow_pickle=False) as stored:
        assert set(stored.files) == PHYSICAL_ARRAY_NAMES
        assert np.array_equal(stored["zero_action_readout_m"], held)


def test_action_pair_rejects_different_initial_states() -> None:
    springs, lengths = _graph()
    driven = np.zeros((3, 3, 3), dtype=np.float32)
    held = driven.copy()
    held[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="frame-zero"):
        build_phystwin_action_pair_arrays(
            driven,
            held,
            springs=springs,
            rest_lengths_m=lengths,
            object_spring_count=2,
        )


def test_action_support_rejects_disconnected_object_graph() -> None:
    springs = np.array([[0, 1], [1, 3]], dtype=np.int32)
    lengths = np.ones(2, dtype=np.float32)

    with pytest.raises(ValueError, match="complete object graph"):
        phystwin_graph_action_support(
            springs,
            lengths,
            object_point_count=3,
            object_spring_count=1,
        )
