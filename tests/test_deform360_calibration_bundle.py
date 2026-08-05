from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    DEFORM360_CALIBRATION_STATUS,
    Deform360CalibrationArtifactRefV1,
    Deform360CalibrationBundleV1,
    Deform360CohortUnitV1,
    load_deform360_calibration_bundle,
    save_deform360_calibration_bundle,
    verify_deform360_confirmation_gate,
)


def _unit(index: int, *, role: str, stratum: str) -> Deform360CohortUnitV1:
    object_id = f"{index:03d}-{role}-{stratum}"
    return Deform360CohortUnitV1(
        object_id=object_id,
        episode_id=index,
        stratum=stratum,
        metadata_path=f"raw/{object_id}/metadata.json",
        metadata_sha256=f"{index + 1000:064x}",
    )


def _units(*, role: str, count: int) -> tuple[Deform360CohortUnitV1, ...]:
    result: list[Deform360CohortUnitV1] = []
    offset = 0 if role == "calibration" else 100
    for stratum_index, stratum in enumerate(("sheet", "volumetric")):
        for local_index in range(count):
            index = offset + stratum_index * 20 + local_index + 1
            result.append(_unit(index, role=role, stratum=stratum))
    return tuple(result)


def _artifacts(
    calibration_units: tuple[Deform360CohortUnitV1, ...],
) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in calibration_units)
    return tuple(
        Deform360CalibrationArtifactRefV1(
            role=role,
            artifact_id=f"{index + 1:064x}",
            implementation_revision="a" * 40,
            selection_evidence_id=f"{index + 101:064x}",
            selected_candidate_id=f"candidate-{index}",
            candidate_count=index + 2,
            calibration_group_ids=groups,
            source_artifacts={f"calibration/{role}.json": f"{index + 201:064x}"},
            metadata={"selection_rule": "source-only"},
        )
        for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)
    )


def _bundle(**updates: Any) -> Deform360CalibrationBundleV1:
    calibration_units = _units(role="calibration", count=5)
    values: dict[str, Any] = {
        "selection_artifact_sha256": "1" * 64,
        "content_selection_sha256": "2" * 64,
        "dataset_revision": "3" * 40,
        "processing_revision": "4" * 40,
        "implementation_revision": "5" * 40,
        "calibration_units": calibration_units,
        "confirmation_units": _units(role="confirmation", count=6),
        "calibration_artifacts": _artifacts(calibration_units),
        "evidence_use_ledger_id": "6" * 64,
        "source_artifacts": {
            "protocols/deform360_official_hub_visuotactile_v1.json": "7" * 64,
            "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json": "8"
            * 64,
        },
        "metadata": {"sealed_by": "calibration-workflow"},
    }
    values.update(updates)
    return Deform360CalibrationBundleV1(**values)


def test_bundle_is_order_invariant_content_addressed_and_immutable() -> None:
    bundle = _bundle()
    reversed_bundle = _bundle(
        calibration_units=tuple(reversed(bundle.calibration_units)),
        confirmation_units=tuple(reversed(bundle.confirmation_units)),
        calibration_artifacts=tuple(reversed(bundle.calibration_artifacts)),
    )

    assert reversed_bundle.bundle_id == bundle.bundle_id
    assert reversed_bundle.confirmation_opening_token == (
        bundle.confirmation_opening_token
    )
    assert bundle.status == DEFORM360_CALIBRATION_STATUS
    assert bundle.summary()["calibration_object_count"] == 10
    assert bundle.summary()["confirmation_object_count"] == 12
    assert bundle.summary()["target_outcomes_used"] is False
    with pytest.raises(TypeError, match="immutable"):
        bundle.metadata["new"] = True


def test_confirmation_gate_requires_every_reviewed_identity() -> None:
    bundle = _bundle()

    token = verify_deform360_confirmation_gate(
        bundle,
        expected_bundle_id=bundle.bundle_id,
        expected_selection_artifact_sha256=bundle.selection_artifact_sha256,
        expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
    )

    assert token == bundle.confirmation_opening_token
    with pytest.raises(ValueError, match="bundle identity"):
        verify_deform360_confirmation_gate(
            bundle,
            expected_bundle_id="9" * 64,
            expected_selection_artifact_sha256=bundle.selection_artifact_sha256,
            expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
        )
    with pytest.raises(ValueError, match="selection artifact"):
        verify_deform360_confirmation_gate(
            bundle,
            expected_bundle_id=bundle.bundle_id,
            expected_selection_artifact_sha256="9" * 64,
            expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
        )
    with pytest.raises(ValueError, match="ledger"):
        verify_deform360_confirmation_gate(
            bundle,
            expected_bundle_id=bundle.bundle_id,
            expected_selection_artifact_sha256=bundle.selection_artifact_sha256,
            expected_evidence_use_ledger_id="9" * 64,
        )


def test_bundle_roundtrip_revalidates_nested_ids_and_strict_json(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.json"
    save_deform360_calibration_bundle(bundle, path)

    loaded = load_deform360_calibration_bundle(path)
    assert loaded == bundle
    assert loaded.bundle_id == bundle.bundle_id
    with pytest.raises(FileExistsError):
        save_deform360_calibration_bundle(bundle, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration_artifacts"][0]["selected_candidate_id"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ref_id"):
        load_deform360_calibration_bundle(path)

    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_calibration_bundle(path)


def test_unit_rejects_path_escape_and_tampered_identity() -> None:
    with pytest.raises(ValueError, match="confined"):
        Deform360CohortUnitV1(
            object_id="object",
            episode_id=1,
            stratum="sheet",
            metadata_path="../object/metadata.json",
            metadata_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="raw/<object>"):
        Deform360CohortUnitV1(
            object_id="object",
            episode_id=1,
            stratum="sheet",
            metadata_path="raw/other/metadata.json",
            metadata_sha256="a" * 64,
        )

    unit = _unit(1, role="calibration", stratum="sheet")
    record = unit.to_record()
    record["episode_id"] = 2
    with pytest.raises(ValueError, match="unit_id"):
        Deform360CohortUnitV1.from_mapping(record)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"target_outcomes_used": True}, "target outcomes"),
        ({"confirmation_payload_used": True}, "confirmation payloads"),
        ({"candidate_count": True}, "candidate_count"),
        ({"candidate_count": 0}, "candidate_count"),
        ({"role": "unknown"}, "unsupported"),
        ({"calibration_group_ids": ()}, "must not be empty"),
        ({"source_artifacts": {}}, "must not be empty"),
        ({"metadata": {"bad": float("nan")}}, "finite JSON"),
    ],
)
def test_calibration_artifact_rejects_malformed_inputs(
    updates: dict[str, Any],
    message: str,
) -> None:
    calibration_units = _units(role="calibration", count=5)
    values: dict[str, Any] = {
        "role": DEFORM360_CALIBRATION_ROLES[0],
        "artifact_id": "1" * 64,
        "implementation_revision": "2" * 40,
        "selection_evidence_id": "3" * 64,
        "selected_candidate_id": "candidate",
        "candidate_count": 2,
        "calibration_group_ids": tuple(unit.object_id for unit in calibration_units),
        "source_artifacts": {"selection.json": "4" * 64},
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        Deform360CalibrationArtifactRefV1(**values)


def test_bundle_rejects_incomplete_or_leaky_calibration() -> None:
    bundle = _bundle()

    with pytest.raises(ValueError, match="roles are incomplete"):
        _bundle(calibration_artifacts=bundle.calibration_artifacts[:-1])
    with pytest.raises(ValueError, match="duplicate calibration role"):
        _bundle(
            calibration_artifacts=(
                *bundle.calibration_artifacts[:-1],
                bundle.calibration_artifacts[0],
            )
        )
    with pytest.raises(ValueError, match="does not retain every"):
        changed = replace(
            bundle.calibration_artifacts[0],
            calibration_group_ids=bundle.calibration_artifacts[0].calibration_group_ids[
                :-1
            ],
        )
        _bundle(calibration_artifacts=(changed, *bundle.calibration_artifacts[1:]))
    with pytest.raises(ValueError, match="overlap"):
        _bundle(
            confirmation_units=(
                bundle.calibration_units[:1] + bundle.confirmation_units[1:]
            )
        )
    with pytest.raises(ValueError, match="exactly 5"):
        _bundle(calibration_units=bundle.calibration_units[:-1])
    with pytest.raises(ValueError, match="confirmation payload"):
        _bundle(confirmation_payload_opened=True)
    with pytest.raises(ValueError, match="target outcomes"):
        _bundle(target_outcomes_used=True)
    with pytest.raises(ValueError, match="may not be replaced"):
        _bundle(replacement_allowed=True)
    with pytest.raises(ValueError, match="protocol_id changed"):
        _bundle(protocol_id="different")


def test_cohort_unit_and_nested_record_validation_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="stratum"):
        _unit(1, role="calibration", stratum="filament")

    unit = _unit(1, role="calibration", stratum="sheet")
    base = unit.to_record()
    with pytest.raises(ValueError, match="JSON object"):
        Deform360CohortUnitV1.from_mapping([])
    for key, value, message in (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
    ):
        record = dict(base)
        record[key] = value
        with pytest.raises(ValueError, match=message):
            Deform360CohortUnitV1.from_mapping(record)

    artifact = _artifacts(_units(role="calibration", count=5))[0]
    artifact_base = artifact.to_record()
    with pytest.raises(ValueError, match="JSON object"):
        Deform360CalibrationArtifactRefV1.from_mapping([])
    for key, value, message in (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
        ("semantics", "changed", "semantics changed"),
        ("ref_id", "f" * 64, "ref_id"),
    ):
        record = dict(artifact_base)
        record[key] = value
        with pytest.raises(ValueError, match=message):
            Deform360CalibrationArtifactRefV1.from_mapping(record)


def test_bundle_rejects_invalid_containers_and_repeated_units() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="calibration_artifacts must be a sequence"):
        _bundle(calibration_artifacts="bad")
    with pytest.raises(ValueError, match="must contain"):
        _bundle(calibration_artifacts=(*bundle.calibration_artifacts[:-1], object()))
    with pytest.raises(ValueError, match="calibration_units must be a sequence"):
        _bundle(calibration_units="bad")
    with pytest.raises(ValueError, match="must contain Deform360CohortUnitV1"):
        _bundle(calibration_units=())
    with pytest.raises(ValueError, match="repeats an object"):
        _bundle(
            calibration_units=(
                *bundle.calibration_units[:-1],
                bundle.calibration_units[0],
            )
        )


def test_bundle_record_validation_is_fail_closed() -> None:
    bundle = _bundle()
    base = bundle.to_record()
    with pytest.raises(ValueError, match="JSON object"):
        Deform360CalibrationBundleV1.from_mapping([])

    mutations = (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "schema_version changed"),
        ("semantics", "changed", "semantics changed"),
        ("dataset_repository", "other/repo", "dataset repository changed"),
        ("processing_repository", "other/repo", "processing repository changed"),
        ("claim_boundary", "changed", "claim boundary changed"),
        ("calibration_units", {}, "calibration_units must be a JSON array"),
        ("confirmation_units", {}, "confirmation_units must be a JSON array"),
        ("calibration_artifacts", {}, "calibration_artifacts must be a JSON array"),
        ("bundle_id", "f" * 64, "bundle_id"),
        ("confirmation_opening_token", "f" * 64, "confirmation_opening_token"),
    )
    for key, value, message in mutations:
        record = dict(base)
        record[key] = value
        with pytest.raises(ValueError, match=message):
            Deform360CalibrationBundleV1.from_mapping(record)


def test_bundle_gate_and_save_type_boundaries_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Deform360CalibrationBundleV1"):
        verify_deform360_confirmation_gate(
            object(),  # type: ignore[arg-type]
            expected_bundle_id="1" * 64,
            expected_selection_artifact_sha256="2" * 64,
            expected_evidence_use_ledger_id="3" * 64,
        )
    with pytest.raises(TypeError, match="Deform360CalibrationBundleV1"):
        save_deform360_calibration_bundle(
            object(),  # type: ignore[arg-type]
            tmp_path / "bad.json",
        )


def test_bundle_atomic_overwrite_and_loader_root_boundaries(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.json"
    save_deform360_calibration_bundle(bundle, path)
    save_deform360_calibration_bundle(bundle, path, overwrite=True)
    assert load_deform360_calibration_bundle(path).bundle_id == bundle.bundle_id

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_deform360_calibration_bundle(path)
