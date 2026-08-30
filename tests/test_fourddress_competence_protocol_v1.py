from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.fourddress_competence_protocol_v1 import (
    CERTIFICATION_COUNT,
    FOURDDRESS_REVISION,
    HOOD_REVISION,
    METHOD_SELECTION_COUNT,
    build_fourddress_participant_split_v1,
    load_fourddress_competence_feasibility_v1,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/locks/fourddress_query_competence_feasibility_v1.json"


def _payload() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _reseal(value: dict[str, Any]) -> None:
    descriptor = dict(value)
    descriptor.pop("protocol_id", None)
    value["protocol_id"] = content_id(descriptor)


def _write(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_access_closed_feasibility_protocol_loads() -> None:
    protocol = load_fourddress_competence_feasibility_v1(PROTOCOL)

    assert protocol.value["upstreams"]["dataset_revision"] == FOURDDRESS_REVISION
    assert protocol.value["upstreams"]["simulator_revision"] == HOOD_REVISION
    assert protocol.source_execution_authorized is False
    assert protocol.certification_execution_authorized is False
    assert len(protocol.unresolved_prerequisites) == 7
    assert protocol.value["information_boundary"]["physical_outcomes_read"] is False
    assert "participant_ids" not in json.dumps(protocol.value)


def test_protocol_id_binds_every_field() -> None:
    value = _payload()
    declared = value.pop("protocol_id")
    assert declared == content_id(value)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("schema", "wrong-schema", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("protocol_id", "0" * 64, "protocol_id changed"),
    ],
)
def test_top_level_identity_substitutions_are_rejected(
    tmp_path: Path,
    field: str,
    replacement: object,
    match: str,
) -> None:
    value = _payload()
    value[field] = replacement
    with pytest.raises(ValueError, match=match):
        load_fourddress_competence_feasibility_v1(_write(tmp_path, value))


def test_protocol_rejects_nonmapping_sections(tmp_path: Path) -> None:
    value = _payload()
    value["upstreams"] = []
    _reseal(value)
    with pytest.raises(ValueError, match="upstreams must be a mapping"):
        load_fourddress_competence_feasibility_v1(_write(tmp_path, value))


@pytest.mark.parametrize(
    ("section", "field", "replacement", "match"),
    [
        ("upstreams", "simulator_revision", "0" * 40, "upstreams changed"),
        (
            "access_audit",
            "dataset_payload_found",
            True,
            "access audit changed",
        ),
        (
            "information_boundary",
            "source_execution_authorized",
            True,
            "information boundary changed",
        ),
        (
            "unresolved_prerequisites",
            "participant_split_id",
            "0" * 64,
            "unresolved prerequisites changed",
        ),
    ],
)
def test_resealed_mutations_remain_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    match: str,
) -> None:
    value = _payload()
    value[section][field] = replacement
    _reseal(value)
    with pytest.raises(ValueError, match=match):
        load_fourddress_competence_feasibility_v1(_write(tmp_path, value))


def test_names_only_split_is_deterministic_disjoint_and_order_invariant() -> None:
    participants = [f"participant-{index:02d}" for index in range(32)]
    first = build_fourddress_participant_split_v1(participants)
    second = build_fourddress_participant_split_v1(list(reversed(participants)))

    assert first == second
    assert len(first.method_selection_participants) == METHOD_SELECTION_COUNT
    assert len(first.certification_participants) == CERTIFICATION_COUNT
    assert set(first.method_selection_participants).isdisjoint(
        first.certification_participants
    )
    assert set(first.method_selection_participants) | set(
        first.certification_participants
    ) == set(participants)
    assert first.split_id == content_id(first.descriptor())


def test_split_rejects_wrong_count_duplicates_and_noncanonical_ids() -> None:
    participants = [f"participant-{index:02d}" for index in range(32)]
    with pytest.raises(ValueError, match="32 entries"):
        build_fourddress_participant_split_v1(participants[:-1])

    duplicated = participants.copy()
    duplicated[-1] = duplicated[0]
    with pytest.raises(ValueError, match="unique"):
        build_fourddress_participant_split_v1(duplicated)

    noncanonical = participants.copy()
    noncanonical[-1] = " participant-31"
    with pytest.raises(ValueError, match="canonical"):
        build_fourddress_participant_split_v1(noncanonical)


def test_split_rejects_a_scalar_string() -> None:
    with pytest.raises(ValueError, match="sequence of strings"):
        build_fourddress_participant_split_v1("participant-00")
