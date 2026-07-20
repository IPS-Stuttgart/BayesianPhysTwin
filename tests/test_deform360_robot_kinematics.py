from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    artifact_sha256,
    load_robot_kinematics_archive,
    select_robot_kinematics_window,
    slice_robot_kinematics,
    validate_robot_kinematics_arrays,
    validate_robot_kinematics_selection_audit,
    validate_selected_robot_kinematics_bundle,
)


def _robot_arrays(
    frame_count: int,
    *,
    bimanual: bool = False,
) -> dict[str, np.ndarray]:
    gripper_count = 2 if bimanual else 1
    transforms = np.tile(
        np.eye(4, dtype=np.float64),
        (frame_count, gripper_count, 1, 1),
    )
    openings = np.full((frame_count, gripper_count), 0.05, dtype=np.float64)
    actions = np.zeros((frame_count, gripper_count, 5, 3), dtype=np.float64)
    actions[..., 1:4, :] = transforms[..., :3, :3]
    actions[..., 4, 0] = openings
    if not bimanual:
        transforms = transforms[:, 0]
        openings = openings[:, 0]
        actions = actions[:, 0]
    return {
        "format_version": np.asarray(1, dtype=np.uint16),
        "actions": actions,
        "T_worlds": transforms,
        "openings": openings,
        "bimanual": np.asarray(bimanual, dtype=np.bool_),
    }


def _set_translation(
    arrays: dict[str, np.ndarray],
    translations: np.ndarray,
) -> None:
    transforms = arrays["T_worlds"]
    actions = arrays["actions"]
    transforms[..., :3, 3] = translations
    actions[..., 0, :] = translations


def _state(arrays: dict[str, np.ndarray]):
    return validate_robot_kinematics_arrays(**arrays)


def _write_robot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def test_contract_is_canonical_and_tworlds_specific() -> None:
    assert ROBOT_KINEMATICS_WINDOW_POLICY_ID.endswith("closed-path-v1")
    assert (
        ROBOT_KINEMATICS_WINDOW_CONTRACT["selection_inputs"][0]
        == "robot.npz:T_worlds[..., :3, 3]"
    )
    assert (
        "not a delta command"
        in ROBOT_KINEMATICS_WINDOW_CONTRACT["trajectory_semantics"]
    )
    assert len(ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256) == 64


def test_monomanual_window_has_exact_candidates_digest_and_earliest_tie() -> None:
    arrays = _robot_arrays(95)
    translations = np.zeros((95, 3), dtype=np.float64)
    translations[:, 0] = np.arange(95, dtype=np.float64)
    _set_translation(arrays, translations)
    state = _state(arrays)

    audit = select_robot_kinematics_window(state)
    repeated = select_robot_kinematics_window(state)

    assert state.eef_translations_world_m.shape == (95, 1, 3)
    assert state.canonical_openings_m.shape == (95, 1)
    assert audit == repeated
    assert audit["candidate_count"] == 2
    assert [record["frame_range_half_open"] for record in audit["candidates"]] == [
        [8, 89],
        [14, 95],
    ]
    assert audit["selected_candidate_index"] == 0
    assert audit["selected_raw_frame_range_half_open"] == [8, 89]
    assert audit["prediction_raw_frame_range_half_open"] == [8, 84]
    assert audit["selected_score"][
        "per_gripper_closed_weighted_translation_path_length_m"
    ] == [80.0]
    assert audit["artifact_sha256"] == artifact_sha256(audit)
    assert len(audit["candidate_records_sha256"]) == 64


def test_bimanual_paths_are_scored_before_the_gripper_mean() -> None:
    arrays = _robot_arrays(95, bimanual=True)
    translations = np.zeros((95, 2, 3), dtype=np.float64)
    translations[:, 0, 0] = np.arange(95, dtype=np.float64)
    translations[:, 1, 0] = -np.arange(95, dtype=np.float64)
    _set_translation(arrays, translations)

    audit = select_robot_kinematics_window(_state(arrays))

    assert audit["gripper_count"] == 2
    assert audit["selected_score"][
        "per_gripper_closed_weighted_translation_path_length_m"
    ] == [80.0, 80.0]
    assert (
        audit["selected_score"]["mean_closed_weighted_translation_path_length_m"]
        == 80.0
    )
    assert np.allclose(np.mean(translations, axis=1), 0.0)


def test_rotation_changes_do_not_enter_translation_path_score() -> None:
    arrays = _robot_arrays(95)
    translations = np.zeros((95, 3), dtype=np.float64)
    translations[:, 0] = np.arange(95, dtype=np.float64) * 0.001
    _set_translation(arrays, translations)
    reference = select_robot_kinematics_window(_state(arrays))

    rotated = deepcopy(arrays)
    angles = np.linspace(0.0, np.pi, 95)
    rotations = np.zeros((95, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = np.cos(angles)
    rotations[:, 0, 1] = -np.sin(angles)
    rotations[:, 1, 0] = np.sin(angles)
    rotations[:, 1, 1] = np.cos(angles)
    rotations[:, 2, 2] = 1.0
    rotated["T_worlds"][:, :3, :3] = rotations
    rotated["actions"][:, 1:4, :] = rotations
    observed = select_robot_kinematics_window(_state(rotated))

    assert (
        observed["selected_raw_frame_range_half_open"]
        == reference["selected_raw_frame_range_half_open"]
    )
    assert [
        record["mean_closed_weighted_translation_path_length_m"]
        for record in observed["candidates"]
    ] == [
        record["mean_closed_weighted_translation_path_length_m"]
        for record in reference["candidates"]
    ]


def test_closure_confidence_is_per_gripper_and_constant_spans_are_one() -> None:
    arrays = _robot_arrays(95, bimanual=True)
    translations = np.zeros((95, 2, 3), dtype=np.float64)
    translations[:, :, 0] = np.arange(95, dtype=np.float64)[:, None] * 0.001
    _set_translation(arrays, translations)
    arrays["openings"][:, 1] = np.linspace(0.01, 0.11, 95)
    arrays["actions"][..., 4, :] = 0.0
    arrays["actions"][..., 4, 0] = arrays["openings"]

    audit = select_robot_kinematics_window(_state(arrays))
    closure = audit["closure_confidence"]

    assert closure["varying_opening_by_gripper"] == [False, True]
    assert closure["q10_opening_m"][0] == pytest.approx(0.05)
    assert closure["q90_opening_m"][0] == pytest.approx(0.05)
    first_candidate = audit["candidates"][0][
        "per_gripper_closed_weighted_translation_path_length_m"
    ]
    assert first_candidate[0] == pytest.approx(0.080)
    assert 0.0 < first_candidate[1] < first_candidate[0]


@pytest.mark.parametrize(
    ("frame_count", "expected_starts"),
    [
        (89, [8]),
        (95, [8, 14]),
    ],
)
def test_candidate_grid_includes_the_last_complete_window(
    frame_count: int,
    expected_starts: list[int],
) -> None:
    audit = select_robot_kinematics_window(_state(_robot_arrays(frame_count)))
    assert [record["frame_range_half_open"][0] for record in audit["candidates"]] == (
        expected_starts
    )


def test_candidate_grid_rejects_no_complete_window() -> None:
    with pytest.raises(ValueError, match="shorter than the requested window"):
        select_robot_kinematics_window(_state(_robot_arrays(80)))
    with pytest.raises(ValueError, match="no complete candidate"):
        select_robot_kinematics_window(_state(_robot_arrays(88)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda values: values.update(format_version=np.asarray(1, dtype=np.int64)),
            "uint16 scalar",
        ),
        (
            lambda values: values.update(bimanual=np.asarray(0, dtype=np.int64)),
            "bool scalar",
        ),
        (
            lambda values: values.update(actions=values["actions"].astype(np.float32)),
            "actions must have dtype float64",
        ),
        (
            lambda values: values["T_worlds"].__setitem__((0, 3, 3), 2.0),
            "homogeneous bottom row",
        ),
        (
            lambda values: values["T_worlds"].__setitem__((0, 0, 0), 2.0),
            "rotation is not orthonormal",
        ),
        (
            lambda values: values["actions"].__setitem__((0, 0, 0), 1.0),
            "row 0 does not match",
        ),
        (
            lambda values: values["actions"].__setitem__((0, 1, 0), 0.0),
            "rows 1:4 do not match",
        ),
        (
            lambda values: values["actions"].__setitem__((0, 4, 0), 0.2),
            "row 4 does not match",
        ),
        (
            lambda values: values["openings"].__setitem__(0, -0.1),
            "non-negative",
        ),
    ],
)
def test_strict_robot_contract_rejects_schema_and_parity_changes(
    mutation,
    message: str,
) -> None:
    arrays = _robot_arrays(95)
    mutation(arrays)
    with pytest.raises(ValueError, match=message):
        _state(arrays)


def test_archive_loader_requires_exact_fields_and_no_symlink(tmp_path: Path) -> None:
    arrays = _robot_arrays(95)
    source = tmp_path / "robot.npz"
    _write_robot(source, arrays)
    loaded = load_robot_kinematics_archive(source, expected_frame_count=95)
    assert loaded.frame_count == 95

    extra = tmp_path / "extra.npz"
    _write_robot(extra, {**arrays, "unexpected": np.asarray(1)})
    with pytest.raises(ValueError, match="field set changed"):
        load_robot_kinematics_archive(extra)

    link = tmp_path / "link.npz"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_robot_kinematics_archive(link)


def test_selection_audit_recomputation_rejects_deep_tamper() -> None:
    state = _state(_robot_arrays(95))
    audit = select_robot_kinematics_window(state)
    assert validate_robot_kinematics_selection_audit(audit, state) == audit

    tampered = deepcopy(audit)
    tampered["candidates"][0]["mean_closed_weighted_translation_path_length_m"] = 1.0
    tampered["artifact_sha256"] = artifact_sha256(tampered)
    with pytest.raises(ValueError, match="selection audit changed"):
        validate_robot_kinematics_selection_audit(tampered, state)


def test_selected_bundle_must_equal_the_exact_source_slice(tmp_path: Path) -> None:
    arrays = _robot_arrays(100, bimanual=True)
    translations = np.zeros((100, 2, 3), dtype=np.float64)
    translations[:, 0, 0] = np.arange(100, dtype=np.float64) * 0.001
    translations[:, 1, 1] = np.arange(100, dtype=np.float64) * 0.002
    _set_translation(arrays, translations)
    source = _state(arrays)
    selected = slice_robot_kinematics(source, start_frame=8, frame_count=76)
    selected_path = tmp_path / "selected.npz"
    _write_robot(selected_path, selected.archive_arrays())

    audit = validate_selected_robot_kinematics_bundle(
        selected_path,
        source_state=source,
        prediction_start_frame=8,
    )
    assert audit["exact_source_slice"] is True
    assert audit["prediction_raw_frame_range_half_open"] == [8, 84]
    assert audit["artifact_sha256"] == artifact_sha256(audit)

    changed = selected.archive_arrays()
    changed["T_worlds"][0, 0, 0, 3] += 0.001
    changed["actions"][0, 0, 0, 0] += 0.001
    changed_path = tmp_path / "changed.npz"
    _write_robot(changed_path, changed)
    with pytest.raises(ValueError, match="exact source slice: T_worlds"):
        validate_selected_robot_kinematics_bundle(
            changed_path,
            source_state=source,
            prediction_start_frame=8,
        )


def test_validated_arrays_are_immutable_copies() -> None:
    arrays = _robot_arrays(95)
    state = _state(arrays)
    arrays["T_worlds"][0, 0, 3] = 10.0
    assert state.T_worlds[0, 0, 3] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        state.T_worlds[0, 0, 3] = 1.0
