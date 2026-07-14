from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_contact import (
    evaluate_target_contact_oracle,
    fit_contact_model,
    seal_target_contact_predictions,
    validate_contact_artifact,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_001_rope_v1.json"
    )


def _metadata(raw: Path) -> None:
    raw.mkdir(parents=True)
    sequences = {}
    actions = (
        "move edge",
        "move center",
        "lift edge",
        "lift center",
        "curl edge",
        "lift both edges",
        "move both edges",
        "push both edges",
        "curl both edges",
        "lift middle",
    )
    for index, action in enumerate(actions):
        sequences[str(index)] = {
            "action": action,
            "bimanual": "yes" if index >= 5 else "no",
            "nonprehensile": "yes" if index == 7 else "no",
        }
    (raw / "metadata.json").write_text(
        json.dumps({"object": "001-rope", "sequences": sequences}) + "\n",
        encoding="utf-8",
    )


def _tactile_values(
    frame_count: int, start: int, end: int, *, persistent: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((frame_count, 16, 32), dtype=np.float32)
    right = np.zeros_like(left)
    if persistent:
        left[:, 0, 0] = 0.25
        right[:, 0, 1] = 0.25
    else:
        left[start : end + 1, 0, 0] = 1.0
        right[start : end + 1, 0, 1] = 1.0
    return left, right


def _write_episode(
    root: Path,
    index: int,
    *,
    target_suffix_contact: bool = True,
) -> None:
    frame_count = 40
    bimanual = index >= 5
    episode = root / f"episode_{index:04d}"
    robot = episode / "robot"
    robot.mkdir(parents=True)
    if bimanual:
        openings = np.full((frame_count, 2), 0.10, dtype=np.float64)
        openings[8:30, 0] = 0.05
        openings[12:32, 1] = 0.05
    else:
        openings = np.full(frame_count, 0.10, dtype=np.float64)
        openings[10:31] = 0.05
    np.savez(robot / "robot.npz", openings=openings, bimanual=bimanual)

    tactile: dict[str, tuple[np.ndarray, np.ndarray]]
    if bimanual:
        tactile = {
            "brics-odroid_tactilel": _tactile_values(frame_count, 10, 29),
            "brics-odroid_tactiler": _tactile_values(frame_count, 14, 31),
        }
    else:
        tactile = {
            "brics-odroid_tactilel": _tactile_values(
                frame_count, 0, frame_count - 1, persistent=True
            ),
            "brics-odroid_tactiler": _tactile_values(frame_count, 12, 30),
        }
    if index == 6 and not target_suffix_contact:
        for group, (left, right) in tactile.items():
            del group
            left[22:] = 0.0
            right[22:] = 0.0
    for group, (left, right) in tactile.items():
        for side, values in (("left", left), ("right", right)):
            sensor = episode / f"{group}_{side}"
            sensor.mkdir()
            np.save(sensor / "synced_tactile.npy", values)


def _write_fit_episodes(root: Path) -> None:
    for index in (0, 1, 2, 3, 4, 5, 7, 8, 9):
        _write_episode(root, index)


def test_contact_fit_does_not_touch_target(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "001-rope"
    processed = tmp_path / "processed"
    _metadata(raw)
    _write_fit_episodes(processed)
    config = load_deform360_protocol_config(_config_path())

    model = fit_contact_model(raw, processed, config)

    validation = validate_contact_artifact(model, expected_kind="Deform360ContactModel")
    assert validation["passed"] is True
    assert model["information_boundary"]["target_robot_read"] is False
    assert model["information_boundary"]["target_tactile_read"] is False
    assert model["inputs"]["target_episode_ids_touched"] == []
    assert model["model"]["tactile_group_to_robot_axis"] == {
        "brics-odroid_tactilel": 0,
        "brics-odroid_tactiler": 1,
    }


def test_target_seal_is_invariant_to_unread_tactile_suffix(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "001-rope"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _metadata(raw)
    _write_fit_episodes(first)
    _write_fit_episodes(second)
    _write_episode(first, 6, target_suffix_contact=True)
    _write_episode(second, 6, target_suffix_contact=False)
    config = load_deform360_protocol_config(_config_path())
    model = fit_contact_model(raw, first, config)

    first_seal = seal_target_contact_predictions(first, config, model)
    second_seal = seal_target_contact_predictions(second, config, model)

    assert first_seal == second_seal
    assert first_seal["information_boundary"]["target_tactile_prefix_read"] is True
    assert first_seal["information_boundary"]["target_tactile_oracle_read"] is False
    assert first_seal["target_prefix"]["start_frame"] == 14
    assert first_seal["target_prefix"]["stop_frame_exclusive"] == 20
    assert first_seal["inputs"]["target_tactile_full_file_hashes_computed"] is False


def test_oracle_requires_downstream_prediction_seal(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "001-rope"
    processed = tmp_path / "processed"
    _metadata(raw)
    _write_fit_episodes(processed)
    _write_episode(processed, 6)
    config = load_deform360_protocol_config(_config_path())
    model = fit_contact_model(raw, processed, config)
    seal = seal_target_contact_predictions(processed, config, model)

    with pytest.raises(ValueError, match="downstream held-out prediction seal"):
        evaluate_target_contact_oracle(
            processed,
            config,
            model,
            seal,
            held_out_prediction_seal_sha256="not-sealed",
        )

    result = evaluate_target_contact_oracle(
        processed,
        config,
        model,
        seal,
        held_out_prediction_seal_sha256="a" * 64,
    )

    assert result["information_boundary"]["target_tactile_oracle_read"] is True
    assert result["episode_union"]["oracle_tactile"]["f1"] == 1.0
    assert result["episode_union"]["tactile_conditioned_z"]["f1"] > 0.8
