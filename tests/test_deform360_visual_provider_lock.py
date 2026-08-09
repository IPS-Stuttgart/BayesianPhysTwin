from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_visual_provider_lock import (
    DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID,
    DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT,
    DEFORM360_FINITE_GROUP_CONFORMAL_RANK,
    DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID,
    DEFORM360_VISUOTACTILE_PROTOCOL_ID,
    Deform360VisualCalibrationLockV1,
    Deform360VisualProviderLockV1,
    load_deform360_visual_calibration_lock,
    load_deform360_visual_provider_lock,
    save_deform360_visual_calibration_lock,
    save_deform360_visual_provider_lock,
)


def _provider_lock(**updates: object) -> Deform360VisualProviderLockV1:
    values: dict[str, object] = {
        "provider_revision": "1" * 40,
        "provider_manifest_id": "2" * 64,
        "provider_attestation_sha256": "3" * 64,
        "motioncrafter_revision": "4" * 40,
        "model_set_id": "5" * 64,
        "root_seed": 20260805,
        "seed_policy": "per-object-derived-seed-v1",
        "window_size": 25,
        "overlap": 8,
        "height": 320,
        "width": 640,
        "storage_dtype": "float32",
        "initial_metric_frame_prior_id": "6" * 64,
        "additional_metric_anchor_policy": "none",
        "max_gauge_rank": 64,
        "minimum_retained_gauge_trace": 0.999,
        "metadata": {"selection_role": "calibration-and-confirmation"},
    }
    values.update(updates)
    return Deform360VisualProviderLockV1(**values)  # type: ignore[arg-type]


def _calibration_lock(**updates: object) -> Deform360VisualCalibrationLockV1:
    values: dict[str, object] = {
        "visual_provider_lock_id": "a" * 64,
        "selection_lock_id": "b" * 64,
        "calibration_object_ids": tuple(f"object-{index}" for index in range(10)),
        "visual_calibration_id": "c" * 64,
        "contact_anchor_calibration_id": "d" * 64,
        "guard_calibration_id": "e" * 64,
        "interval_calibration_id": "f" * 64,
        "calibration_design_id": (DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID),
        "calibration_group_count": 10,
        "conformal_rank": 10,
        "metadata": {"finite_sample_miscoverage_increment": 1.0 / 11.0},
    }
    values.update(updates)
    return Deform360VisualCalibrationLockV1(**values)  # type: ignore[arg-type]


def test_visual_provider_lock_is_content_addressed_and_roundtrips(tmp_path) -> None:
    lock = _provider_lock()
    path = tmp_path / "visual-provider-lock.json"
    save_deform360_visual_provider_lock(path, lock)
    loaded = load_deform360_visual_provider_lock(path)

    assert loaded == lock
    assert loaded.artifact_id == lock.artifact_id
    assert loaded.to_record()["provider_api_version"] == 2
    assert loaded.to_record()["stream_contract_version"] == 2
    assert loaded.to_record()["full_joint_gauge_covariance"] is True
    assert loaded.to_record()["persistent_material_identities"] is True
    assert loaded.to_record()["target_outcomes_used"] is False
    assert loaded.metadata["selection_role"] == "calibration-and-confirmation"


def test_visual_provider_lock_supports_declared_optional_modes() -> None:
    lock = _provider_lock(
        storage_dtype="float64",
        additional_metric_anchor_policy="independent_sparse",
        max_gauge_rank=None,
    )

    assert lock.storage_dtype == "float64"
    assert lock.additional_metric_anchor_policy == "independent_sparse"
    assert lock.max_gauge_rank is None


def test_visual_provider_lock_rejects_target_access_and_ambiguous_configuration() -> (
    None
):
    cases = [
        ({"selected_raw_payloads_opened": True}, "unopened selected payloads"),
        ({"target_outcomes_used": True}, "target blind"),
        ({"overlap": 25}, "smaller than window_size"),
        ({"max_gauge_rank": 0}, "max_gauge_rank"),
        ({"minimum_retained_gauge_trace": 0.0}, "retained_gauge_trace"),
        ({"provider_revision": int("1" * 40)}, "literal"),
        ({"provider_manifest_id": int("2" * 64)}, "literal"),
        ({"additional_metric_anchor_policy": "ambiguous"}, "unsupported"),
        ({"storage_dtype": "float16"}, "storage_dtype"),
    ]
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            _provider_lock(**updates)


def test_visual_provider_lock_rejects_malformed_descriptors() -> None:
    cases = [
        {"protocol_id": ""},
        {"amendment_id": ""},
        {"provider_repository": ""},
        {"motioncrafter_repository": ""},
        {"provider_repository": "another/Prob4D"},
        {"motioncrafter_repository": "another/MotionCrafter"},
        {"provider_attestation_sha256": int("3" * 64)},
        {"motioncrafter_revision": int("4" * 40)},
        {"model_set_id": int("5" * 64)},
        {"root_seed": True},
        {"seed_policy": ""},
        {"window_size": 1},
        {"overlap": -1},
        {"height": 0},
        {"width": 0},
        {"minimum_retained_gauge_trace": True},
        {"minimum_retained_gauge_trace": "0.9"},
        {"minimum_retained_gauge_trace": float("nan")},
        {"minimum_retained_gauge_trace": 1.1},
        {"selected_raw_payloads_opened": 1},
        {"target_outcomes_used": 0},
    ]
    for updates in cases:
        with pytest.raises(ValueError):
            _provider_lock(**updates)


def test_visual_provider_mapping_rejects_schema_and_semantic_drift() -> None:
    lock = _provider_lock()
    malformed_records: list[object] = [[], {**lock.to_record(), "extra": True}]

    missing = lock.to_record()
    missing.pop("metadata")
    malformed_records.append(missing)

    replacements = [
        ("schema", "another-schema"),
        ("schema_version", 2),
        ("semantics", "another-semantics"),
        ("provider_api_version", 3),
        ("stream_contract_version", 3),
        ("full_joint_gauge_covariance", False),
        ("persistent_material_identities", False),
        ("causal_cutoff_convention", "inclusive"),
        ("artifact_id", int("1" * 64)),
    ]
    for field_name, invalid in replacements:
        record = lock.to_record()
        record[field_name] = invalid
        malformed_records.append(record)

    for record in malformed_records:
        with pytest.raises(ValueError):
            Deform360VisualProviderLockV1.from_mapping(record)


def test_visual_provider_lock_rejects_tampering_and_duplicate_json(tmp_path) -> None:
    lock = _provider_lock()
    record = lock.to_record()
    record["window_size"] = 26
    with pytest.raises(ValueError, match="artifact_id"):
        Deform360VisualProviderLockV1.from_mapping(record)

    for field_name, invalid in (
        ("provider_api_version", 2.0),
        ("stream_contract_version", 2.0),
        ("full_joint_gauge_covariance", 1),
        ("persistent_material_identities", 1),
    ):
        malformed = lock.to_record()
        malformed[field_name] = invalid
        with pytest.raises(ValueError, match=field_name):
            Deform360VisualProviderLockV1.from_mapping(malformed)

    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"x","schema":"y"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_visual_provider_lock(path)


def test_strict_visual_lock_loader_rejects_invalid_documents(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_deform360_visual_provider_lock(tmp_path / "missing.json")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        load_deform360_visual_provider_lock(invalid_json)

    list_root = tmp_path / "list.json"
    list_root.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_deform360_visual_provider_lock(list_root)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_deform360_visual_provider_lock(nonfinite)


def test_visual_calibration_lock_binds_object_level_finite_sample_rank(
    tmp_path,
) -> None:
    lock = _calibration_lock()
    path = tmp_path / "calibration-lock.json"
    save_deform360_visual_calibration_lock(path, lock)
    loaded = load_deform360_visual_calibration_lock(path)

    assert loaded == lock
    assert loaded.calibration_design_id == DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID
    assert (
        loaded.calibration_group_count == DEFORM360_FINITE_GROUP_CALIBRATION_GROUP_COUNT
    )
    assert loaded.conformal_rank == DEFORM360_FINITE_GROUP_CONFORMAL_RANK
    assert loaded.confirmation_payloads_opened is False
    assert loaded.target_outcomes_used is False


def test_visual_calibration_lock_rejects_target_access_and_pseudoreplication() -> None:
    cases = [
        ({"confirmation_payloads_opened": True}, "unopened confirmation"),
        ({"target_outcomes_used": True}, "target blind"),
        ({"calibration_group_count": 9}, "number of calibration objects"),
        ({"conformal_rank": 11}, "registered finite-group"),
        ({"calibration_design_id": "9" * 64}, "finite-group design"),
        (
            {
                "calibration_object_ids": (
                    "object-0",
                    "object-0",
                ),
                "calibration_group_count": 2,
                "conformal_rank": 2,
            },
            "unique",
        ),
        ({"visual_provider_lock_id": int("1" * 64)}, "literal"),
    ]
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            _calibration_lock(**updates)


def test_visual_calibration_lock_rejects_malformed_descriptors() -> None:
    cases = [
        {"protocol_id": ""},
        {"amendment_id": ""},
        {"selection_lock_id": int("2" * 64)},
        {"calibration_object_ids": ()},
        {"visual_calibration_id": int("3" * 64)},
        {"contact_anchor_calibration_id": int("4" * 64)},
        {"guard_calibration_id": int("5" * 64)},
        {"interval_calibration_id": int("6" * 64)},
        {"calibration_design_id": int("7" * 64)},
        {"calibration_group_count": True},
        {"conformal_rank": 0},
        {"confirmation_payloads_opened": 0},
        {"target_outcomes_used": 1},
    ]
    for updates in cases:
        with pytest.raises(ValueError):
            _calibration_lock(**updates)


def test_visual_calibration_mapping_rejects_schema_and_semantic_drift() -> None:
    lock = _calibration_lock()
    malformed_records: list[object] = [[], {**lock.to_record(), "extra": True}]

    missing = lock.to_record()
    missing.pop("metadata")
    malformed_records.append(missing)

    replacements = [
        ("schema", "another-schema"),
        ("schema_version", 2),
        ("semantics", "another-semantics"),
        ("artifact_id", int("7" * 64)),
    ]
    for field_name, invalid in replacements:
        record = lock.to_record()
        record[field_name] = invalid
        malformed_records.append(record)

    tampered = lock.to_record()
    tampered["conformal_rank"] = 9
    malformed_records.append(tampered)

    for record in malformed_records:
        with pytest.raises(ValueError):
            Deform360VisualCalibrationLockV1.from_mapping(record)


def test_visual_lock_records_are_bound_to_the_registered_protocol() -> None:
    provider = _provider_lock()
    calibration = _calibration_lock()

    assert provider.protocol_id == DEFORM360_VISUOTACTILE_PROTOCOL_ID
    assert provider.amendment_id == DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID
    assert calibration.protocol_id == DEFORM360_VISUOTACTILE_PROTOCOL_ID
    assert calibration.amendment_id == DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID

    with pytest.raises(ValueError, match="protocol_id"):
        replace(provider, protocol_id="another-protocol")
    with pytest.raises(ValueError, match="amendment_id"):
        replace(calibration, amendment_id="another-amendment")


def test_serialized_visual_provider_lock_is_finite_json(tmp_path) -> None:
    path = tmp_path / "visual-provider-lock.json"
    save_deform360_visual_provider_lock(path, _provider_lock())
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["artifact_id"] == _provider_lock().artifact_id


def test_target_blind_protocol_amendment_binds_stage_order_and_boundaries() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))

    assert amendment["amendment_id"] == DEFORM360_VISUAL_PROVIDER_AMENDMENT_ID
    assert amendment["parent_protocol"]["id"] == DEFORM360_VISUOTACTILE_PROTOCOL_ID
    assert amendment["visual_provider_lock"]["schema_version"] == 1
    assert amendment["calibration_lock"]["schema_version"] == 1
    design = amendment["finite_group_calibration_design"]
    assert design["artifact_id"] == DEFORM360_FINITE_GROUP_CALIBRATION_DESIGN_ID
    assert design["must_be_bound_by_calibration_lock"] is True
    design_record = json.loads(
        (Path(__file__).resolve().parents[1] / design["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert design_record["artifact_id"] == design["artifact_id"]
    assert amendment["status"] == "locked-before-selected-calibration-payload-access"
    assert not any(amendment["information_boundary"].values())
    assert amendment["stage_order"].index(
        "create and commit one exact visual-provider lock before downloading "
        "selected calibration payloads"
    ) < amendment["stage_order"].index("download and process calibration objects only")


def test_visual_lock_writes_are_non_overwriting(tmp_path: Path) -> None:
    provider_path = tmp_path / "provider.json"
    provider = _provider_lock()
    save_deform360_visual_provider_lock(provider_path, provider)
    with pytest.raises(FileExistsError):
        save_deform360_visual_provider_lock(provider_path, provider)
    save_deform360_visual_provider_lock(
        provider_path,
        provider,
        overwrite=True,
    )
    assert load_deform360_visual_provider_lock(provider_path) == provider

    calibration_path = tmp_path / "calibration.json"
    calibration = _calibration_lock()
    save_deform360_visual_calibration_lock(calibration_path, calibration)
    with pytest.raises(FileExistsError):
        save_deform360_visual_calibration_lock(calibration_path, calibration)
    save_deform360_visual_calibration_lock(
        calibration_path,
        calibration,
        overwrite=True,
    )
    assert load_deform360_visual_calibration_lock(calibration_path) == calibration

    with pytest.raises(TypeError, match="Deform360VisualProviderLockV1"):
        save_deform360_visual_provider_lock(
            tmp_path / "bad-provider.json",
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Deform360VisualCalibrationLockV1"):
        save_deform360_visual_calibration_lock(
            tmp_path / "bad-calibration.json",
            object(),  # type: ignore[arg-type]
        )
