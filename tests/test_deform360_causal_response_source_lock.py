from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_source_lock import (
    REQUIRED_EXCLUSION_SCOPES,
    CausalResponseSourceCase,
    build_causal_response_source_lock,
    deform360_object_hash,
    validate_causal_response_source_lock,
    write_causal_response_source_lock,
)


def _cases() -> tuple[CausalResponseSourceCase, ...]:
    return tuple(
        CausalResponseSourceCase(
            case_id=f"fresh-object-{index}/episode-0001",
            case_hash=f"{index + 100:064x}",
            object_hash=deform360_object_hash(f"fresh-object-{index}"),
            metadata_sha256=f"{index + 200:064x}",
            fold=index % 3,
        )
        for index in range(12)
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
):
    return build_causal_response_source_lock(
        _cases() if cases is None else cases,
        protocol_id="deform360-causal-response-depth-v12",
        repository_revision="a" * 40,
        method_config_sha256="b" * 64,
        exclusion_manifest_sha256=(_manifests() if manifests is None else manifests),
        excluded_object_hashes=excluded,
        selection_metadata_sha256="c" * 64,
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
