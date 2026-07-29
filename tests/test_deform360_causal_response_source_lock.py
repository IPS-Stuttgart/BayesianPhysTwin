from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    REQUIRED_SOURCE_ROLES,
    CausalResponseSourceCameraRecord,
    evaluate_causal_response_source_preflight,
)
from bayesian_phystwin.deform360_causal_response_source_lock import (
    REQUIRED_EXCLUSION_SCOPES,
    CausalResponseSourceCase,
    build_causal_response_source_lock,
    validate_causal_response_source_lock,
    write_causal_response_source_lock,
)


def _preflights():
    cameras = tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=76,
            mask_frame_count=76,
            calibration_valid=True,
            frame_zero_projected_support_count=16,
        )
        for camera in REGISTERED_CAMERA_IDS
    )
    sources = {
        role: f"{index + 500:064x}"
        for index, role in enumerate(sorted(REQUIRED_SOURCE_ROLES))
    }
    return tuple(
        evaluate_causal_response_source_preflight(
            object_id=f"fresh-object-{index}",
            episode_id=1,
            category="cloth",
            bimanual_value="no",
            episode_frame_count=76,
            robot_frame_count=76,
            tactile_frame_count=76,
            physical_node_count=256,
            camera_records=cameras,
            source_sha256=sources,
        )
        for index in range(12)
    )


def _cases(
    preflights=None,
) -> tuple[CausalResponseSourceCase, ...]:
    selected_preflights = _preflights() if preflights is None else preflights
    return tuple(
        CausalResponseSourceCase(
            case_id=f"fresh-object-{index}/episode-0001",
            case_hash=preflight.case_hash,
            object_hash=preflight.object_hash,
            metadata_sha256=f"{index + 200:064x}",
            source_preflight_sha256=preflight.artifact_sha256,
            fold=index % 3,
        )
        for index, preflight in enumerate(selected_preflights)
    )


def _manifests() -> dict[str, str]:
    return {
        scope: f"{index + 300:064x}"
        for index, scope in enumerate(sorted(REQUIRED_EXCLUSION_SCOPES))
    }


def _build(
    cases: tuple[CausalResponseSourceCase, ...] | None = None,
    *,
    manifests: dict[str, str] | None = None,
    excluded: tuple[str, ...] = (),
    preflights=None,
):
    selected_preflights = _preflights() if preflights is None else preflights
    return build_causal_response_source_lock(
        _cases(selected_preflights) if cases is None else cases,
        protocol_id="deform360-causal-response-depth-v12",
        repository_revision="a" * 40,
        method_config_sha256="b" * 64,
        exclusion_manifest_sha256=(_manifests() if manifests is None else manifests),
        excluded_object_hashes=excluded,
        selection_metadata_sha256="c" * 64,
        source_preflights=selected_preflights,
    )


def test_source_lock_is_deterministic_balanced_and_round_trips(
    tmp_path: Path,
) -> None:
    first = _build()
    second = _build(tuple(reversed(_cases())))
    output = tmp_path / "source_lock.json"

    assert first.artifact_sha256 == second.artifact_sha256
    assert [case.fold for case in first.cases].count(0) == 4
    assert [case.fold for case in first.cases].count(1) == 4
    assert [case.fold for case in first.cases].count(2) == 4
    write_causal_response_source_lock(output, first)
    loaded = validate_causal_response_source_lock(output)
    assert loaded.descriptor() == first.descriptor()


def test_source_lock_rejects_overlap_with_any_excluded_object() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        _build(excluded=(_cases()[4].object_hash,))


def test_source_lock_rejects_a_missing_held_campaign_scope() -> None:
    manifests = _manifests()
    del manifests["held_v8_all_attempts"]

    with pytest.raises(ValueError, match="required exclusion scopes"):
        _build(manifests=manifests)


def test_source_lock_rejects_duplicated_physical_objects() -> None:
    cases = list(_cases())
    cases[1] = replace(
        cases[1],
        object_hash=cases[0].object_hash,
    )

    with pytest.raises(ValueError, match="duplicated"):
        _build(tuple(cases))


def test_source_lock_rejects_a_missing_or_rejected_preflight() -> None:
    preflights = _preflights()

    with pytest.raises(ValueError, match="preflight set"):
        _build(cases=_cases(preflights), preflights=preflights[:-1])

    rejected = evaluate_causal_response_source_preflight(
        object_id="fresh-object-0",
        episode_id=1,
        category="cloth",
        bimanual_value="yess",
        episode_frame_count=76,
        robot_frame_count=76,
        tactile_frame_count=76,
        physical_node_count=256,
        camera_records=preflights[0].camera_records,
        source_sha256=preflights[0].source_sha256,
    )
    replaced = (rejected, *preflights[1:])
    with pytest.raises(ValueError, match="accepted V12 preflight"):
        _build(preflights=replaced)
