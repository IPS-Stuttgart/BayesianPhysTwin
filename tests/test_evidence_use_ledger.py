from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.evidence_use_ledger import (
    EVIDENCE_USE_LEDGER_METADATA_KEY,
    EVIDENCE_USE_SEMANTICS,
    EvidenceUseLedgerV1,
    EvidenceUseV1,
    attach_evidence_use_ledger,
    evidence_use_from_deform360_contact_anchor,
    load_evidence_use_ledger,
    save_evidence_use_ledger,
)


def _use(index: int = 0, **updates: Any) -> EvidenceUseV1:
    values: dict[str, Any] = {
        "evidence_artifact_id": f"{index + 1:064x}",
        "raw_factor_id": f"{index + 101:064x}",
        "raw_factor_sha256": f"{index + 201:064x}",
        "source_repository": "brownu/deform360",
        "source_revision": "a" * 40,
        "source_artifacts": {f"raw/object-{index}/factor.npy": f"{index + 301:064x}"},
        "sensor_family": "tactile",
        "stream_id": f"tactile-{index}",
        "clock_id": "robot-clock",
        "causal_frame_start": 1,
        "causal_frame_stop": 8,
        "correlation_group_ids": (f"contact-{index}",),
        "inference_role": "state_update",
        "metadata": {"nested": {"values": [1, 2]}},
    }
    values.update(updates)
    return EvidenceUseV1(**values)


def _ledger(*entries: EvidenceUseV1, **updates: Any) -> EvidenceUseLedgerV1:
    values: dict[str, Any] = {
        "protocol_id": "deform360-official-hub-visuotactile-v1",
        "case_id": "200-fresh-object:episode-3",
        "causal_frame_stop": 8,
        "entries": entries,
        "metadata": {"stage": "calibration"},
    }
    values.update(updates)
    return EvidenceUseLedgerV1(**values)


def test_evidence_use_is_content_addressed_owned_and_canonical() -> None:
    source_artifacts = {"raw/object/factor.npy": "f" * 64}
    metadata = {"nested": {"values": [1, 2]}}
    use = _use(
        source_artifacts=source_artifacts,
        correlation_group_ids=("z", "a"),
        metadata=metadata,
    )
    entry_id = use.entry_id

    source_artifacts["raw/object/factor.npy"] = "e" * 64
    metadata["nested"]["values"].append(3)

    assert use.entry_id == entry_id
    assert use.correlation_group_ids == ("a", "z")
    assert use.metadata["nested"]["values"] == [1, 2]
    assert use.to_record()["semantics"] == EVIDENCE_USE_SEMANTICS
    with pytest.raises(TypeError, match="immutable"):
        use.metadata["nested"]["values"].append(3)


def test_ledger_identity_is_order_invariant_and_append_only() -> None:
    first = _use(0)
    second = _use(1)

    left = _ledger(first, second)
    right = _ledger(second, first)

    assert left.ledger_id == right.ledger_id
    assert [entry.entry_id for entry in left.entries] == sorted(
        [first.entry_id, second.entry_id]
    )
    assert left.summary()["role_counts"] == {"state_update": 2}
    empty = _ledger()
    appended = empty.append(first)
    assert empty.entries == ()
    assert appended.entries == (first,)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ((_use(0), _use(0)), "duplicate evidence-use"),
        (
            (_use(0), _use(1, evidence_artifact_id=f"{1:064x}")),
            "duplicate evidence artifact",
        ),
        (
            (_use(0), _use(1, raw_factor_id=f"{101:064x}")),
            "duplicate raw-factor identity",
        ),
        (
            (_use(0), _use(1, raw_factor_sha256=f"{201:064x}")),
            "relabelled",
        ),
        ((_use(0, causal_frame_stop=9),), "crosses the ledger causal prefix"),
        (
            (
                _use(0, correlation_group_ids=("shared",)),
                _use(
                    1,
                    correlation_group_ids=("shared",),
                    inference_role="contact_abduction",
                ),
            ),
            "across state and intervention",
        ),
        (
            (
                _use(
                    0,
                    correlation_group_ids=("shared",),
                    inference_role="joint_state_intervention_update",
                ),
                _use(
                    1,
                    correlation_group_ids=("shared",),
                    inference_role="state_update",
                ),
            ),
            "joint evidence",
        ),
    ],
)
def test_ledger_rejects_duplicate_or_cross_stage_evidence(
    entries: tuple[EvidenceUseV1, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ledger(*entries)


def test_ledger_allows_joint_processing_within_one_intervention_stage() -> None:
    ledger = _ledger(
        _use(
            0,
            correlation_group_ids=("shared",),
            inference_role="actuator_abduction",
        ),
        _use(
            1,
            correlation_group_ids=("shared",),
            inference_role="contact_abduction",
        ),
    )

    assert len(ledger.entries) == 2


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"evidence_artifact_id": "bad"}, "evidence_artifact_id"),
        ({"raw_factor_id": "bad"}, "raw_factor_id"),
        ({"raw_factor_sha256": "bad"}, "raw_factor_sha256"),
        ({"source_repository": "deform360"}, "owner/name"),
        ({"source_revision": "main"}, "source_revision"),
        ({"source_artifacts": {}}, "must not be empty"),
        ({"sensor_family": 1}, "sensor_family"),
        ({"causal_frame_start": True}, "causal_frame_start"),
        ({"causal_frame_start": 8}, "interval"),
        ({"correlation_group_ids": ()}, "must not be empty"),
        ({"correlation_group_ids": ("a", "a")}, "duplicates"),
        ({"inference_role": "unknown"}, "unsupported"),
        ({"metadata": {"bad": float("nan")}}, "finite JSON"),
    ],
)
def test_evidence_use_rejects_malformed_inputs(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _use(**updates)


def test_ledger_roundtrip_rejects_duplicate_keys_and_tampering(tmp_path: Path) -> None:
    ledger = _ledger(_use(0), _use(1))
    path = tmp_path / "ledger.json"
    save_evidence_use_ledger(ledger, path)

    loaded = load_evidence_use_ledger(path)
    assert loaded == ledger
    assert loaded.ledger_id == ledger.ledger_id
    with pytest.raises(FileExistsError):
        save_evidence_use_ledger(ledger, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["case_id"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger_id"):
        load_evidence_use_ledger(path)

    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_evidence_use_ledger(path)


@dataclass(frozen=True)
class _Batch:
    metadata: dict[str, object]


@dataclass(frozen=True)
class _Anchor:
    artifact_id: str = "a" * 64
    object_id: str = "200-fresh-object"
    episode_id: int = 3
    causal_frame_stop: int = 8
    source_revision: str = "b" * 40
    source_artifacts: dict[str, str] = field(
        default_factory=lambda: {"raw/200-fresh-object/tactile.npy": "c" * 64}
    )
    sensor_names: tuple[str, ...] = ("tactile-a", "tactile-a")
    correlation_group_ids: tuple[str, ...] = ("contact-a", "contact-a")


def test_contact_anchor_helper_and_batch_attachment_preserve_full_ledger() -> None:
    entry = evidence_use_from_deform360_contact_anchor(
        _Anchor(),
        raw_factor_id="d" * 64,
        raw_factor_sha256="e" * 64,
        stream_id="episode-3-contact",
        clock_id="robot-clock",
        causal_frame_start=4,
    )
    ledger = _ledger(entry)
    batch = _Batch(metadata={"observation_causal_frame_stop": 8})

    attached = attach_evidence_use_ledger(batch, ledger)

    record = attached.metadata[EVIDENCE_USE_LEDGER_METADATA_KEY]
    assert record["ledger_id"] == ledger.ledger_id
    assert record["entries"][0]["evidence_artifact_id"] == "a" * 64
    assert record["entries"][0]["metadata"]["object_id"] == "200-fresh-object"

    with pytest.raises(ValueError, match="already contains"):
        attach_evidence_use_ledger(attached, ledger)
    with pytest.raises(ValueError, match="causal cutoffs differ"):
        attach_evidence_use_ledger(
            replace(batch, metadata={"observation_causal_frame_stop": 7}),
            ledger,
        )


def test_noninference_roles_do_not_create_cross_stage_conflicts() -> None:
    ledger = _ledger(
        _use(
            0,
            correlation_group_ids=("shared",),
            inference_role="calibration_only",
        ),
        _use(
            1,
            correlation_group_ids=("shared",),
            inference_role="evaluation_only",
        ),
    )

    assert len(ledger.entries) == 2


def test_evidence_use_record_validation_is_fail_closed() -> None:
    use = _use()
    base = use.to_record()

    with pytest.raises(ValueError, match="JSON object"):
        EvidenceUseV1.from_mapping([])

    mutations = (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
        ("semantics", "changed", "semantics changed"),
        ("entry_id", "f" * 64, "entry_id"),
    )
    for key, value, message in mutations:
        record = dict(base)
        record[key] = value
        with pytest.raises(ValueError, match=message):
            EvidenceUseV1.from_mapping(record)

    record = dict(base)
    record["extra"] = True
    with pytest.raises(ValueError, match="fields changed"):
        EvidenceUseV1.from_mapping(record)


def test_ledger_constructor_and_record_validation_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="sequence"):
        _ledger(entries="bad")
    with pytest.raises(ValueError, match="contain EvidenceUseV1"):
        _ledger(entries=(object(),))

    ledger = _ledger(_use())
    base = ledger.to_record()
    with pytest.raises(ValueError, match="JSON object"):
        EvidenceUseLedgerV1.from_mapping([])

    mutations = (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
        ("semantics", "changed", "semantics changed"),
        ("claim_boundary", "changed", "claim boundary changed"),
        ("entries", {}, "entries must be a JSON array"),
        ("ledger_id", "f" * 64, "ledger_id"),
    )
    for key, value, message in mutations:
        record = dict(base)
        record[key] = value
        with pytest.raises(ValueError, match=message):
            EvidenceUseLedgerV1.from_mapping(record)


def test_helper_rejects_reserved_or_nonmapping_metadata() -> None:
    anchor = _Anchor()
    common = {
        "raw_factor_id": "d" * 64,
        "raw_factor_sha256": "e" * 64,
        "stream_id": "stream",
        "clock_id": "clock",
    }
    with pytest.raises(ValueError, match="must be a mapping"):
        evidence_use_from_deform360_contact_anchor(
            anchor,
            **common,
            metadata=[1],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="reserved"):
        evidence_use_from_deform360_contact_anchor(
            anchor,
            **common,
            metadata={"object_id": "changed"},
        )


def test_attachment_and_save_type_boundaries_are_fail_closed(tmp_path: Path) -> None:
    ledger = _ledger(_use())
    attached = attach_evidence_use_ledger(_Batch(metadata={}), ledger)
    assert attached.metadata[EVIDENCE_USE_LEDGER_METADATA_KEY]["ledger_id"] == (
        ledger.ledger_id
    )

    with pytest.raises(TypeError, match="EvidenceUseLedgerV1"):
        attach_evidence_use_ledger(
            _Batch(metadata={}),
            object(),  # type: ignore[arg-type]
        )

    @dataclass(frozen=True)
    class _BadBatch:
        metadata: list[object]

    with pytest.raises(ValueError, match="metadata must be a mapping"):
        attach_evidence_use_ledger(
            _BadBatch(metadata=[]),  # type: ignore[arg-type]
            ledger,
        )
    with pytest.raises(TypeError, match="EvidenceUseLedgerV1"):
        save_evidence_use_ledger(
            object(),  # type: ignore[arg-type]
            tmp_path / "bad.json",
        )


def test_strict_loader_and_atomic_overwrite_boundaries(tmp_path: Path) -> None:
    ledger = _ledger(_use())
    path = tmp_path / "ledger.json"
    save_evidence_use_ledger(ledger, path)
    save_evidence_use_ledger(ledger, path, overwrite=True)
    assert load_evidence_use_ledger(path).ledger_id == ledger.ledger_id

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_evidence_use_ledger(path)
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_evidence_use_ledger(path)
    with pytest.raises(ValueError, match="cannot read"):
        load_evidence_use_ledger(tmp_path / "missing.json")


def test_source_artifact_mapping_rejects_nonmapping_and_noncanonical_paths() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _use(source_artifacts=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical paths"):
        _use(source_artifacts={" bad ": "a" * 64})
