import hashlib
import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_action_robust_all18 import (
    EXPECTED_ALL18_OBJECTS,
    NEW_OBJECTS,
    SOURCE_FIELD,
    build_all18_calibration,
    calibration_sha256,
    protocol_sha256,
    source_row_from_smoke,
    validate_all18_calibration,
    validate_all18_source_protocol,
)
from bayesian_phystwin.pokeflex_action_robust_scale import (
    BASE_EFFECTIVE_SCALE,
    CANDIDATE_MULTIPLIERS,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / "pokeflex_action_robust_all18_source_v4.json"
PARENT = ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_v3.json"
CALIBRATION = ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_all18_v4.json"
REMOTE_WRAPPER = (
    ROOT / "scripts" / "remote" / "run_pokeflex_action_robust_all18_source.py"
)


def _smoke(take_id: str, protocol_digest: str, *, best: float = 2.0) -> dict:
    aggregates = {}
    for multiplier in CANDIDATE_MULTIPLIERS:
        scale = BASE_EFFECTIVE_SCALE * multiplier
        error = 10.0 if multiplier == 1.0 else 10.4
        if multiplier == best:
            error = 9.8
        aggregates[f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"] = {
            "mean_CD_UL1_mm": error
        }
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "all18_source_protocol_sha256": protocol_digest,
        "take": {"id": take_id},
        "future_observation_used": False,
        "correction_fields": [SOURCE_FIELD],
        "aggregates": aggregates,
        "updates": [
            {"accepted": True, "action_supported": True},
            {"accepted": False, "action_supported": True},
        ],
    }


def test_frozen_source_protocol_is_exact_and_target_disjoint() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    validation = validate_all18_source_protocol(payload)

    assert validation["protocol_sha256"] == (
        "7a7b291418964ed7ccaf54f2eb4e2db25badf35edc3cb68d4ca484e0b0a6ed03"
    )
    assert validation["protocol_sha256"] == protocol_sha256(payload)
    assert len(validation["selected_take_ids"]) == 12
    for row in payload["source_selection"]["objects"].values():
        assert row["official_target_take_id"] not in row["selected_take_ids"]


def test_protocol_rejects_official_target_even_with_valid_checksum() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    row = payload["source_selection"]["objects"]["FoamDice"]
    row["eligible_take_ids"][0] = "FoamDice_T3"
    row["selected_take_ids"][0] = "FoamDice_T3"
    payload["protocol_sha256"] = protocol_sha256(payload)

    with pytest.raises(ValueError, match="official target"):
        validate_all18_source_protocol(payload)


def test_source_row_requires_protocol_binding_and_causal_prediction() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    take_id = "FoamDice_T1"
    smoke = _smoke(take_id, protocol["protocol_sha256"])

    row = source_row_from_smoke(
        smoke,
        expected_take_id=take_id,
        expected_protocol_sha256=protocol["protocol_sha256"],
    )

    assert row["supported_frame_count"] == 1
    assert row["mean_CD_UL1_mm_by_multiplier"]["2.0"] == 9.8

    leaked = deepcopy(smoke)
    leaked["future_observation_used"] = True
    with pytest.raises(ValueError, match="leaked"):
        source_row_from_smoke(
            leaked,
            expected_take_id=take_id,
            expected_protocol_sha256=protocol["protocol_sha256"],
        )


def test_builder_extends_parent_without_changing_parent_rows() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    validation = validate_all18_source_protocol(protocol)
    artifacts = {
        take_id: _smoke(take_id, protocol["protocol_sha256"])
        for take_id in validation["selected_take_ids"]
    }
    source_hashes = {
        take_id: f"{index:064x}"
        for index, take_id in enumerate(validation["selected_take_ids"], start=1)
    }

    calibration = build_all18_calibration(
        parent,
        protocol,
        artifacts,
        source_artifact_file_sha256s=source_hashes,
    )
    result = validate_all18_calibration(calibration)

    assert result["passed"] is True
    assert set(result["multipliers"]) == set(EXPECTED_ALL18_OBJECTS)
    assert set(calibration["new_objects"]) == set(NEW_OBJECTS)
    assert calibration["source_gate"]["adjusted_new_object_count"] == 6
    assert calibration["source_gate"]["source_action_regression_count"] == 0
    assert calibration["source_artifact_file_sha256s"] == source_hashes
    for object_name, row in parent["objects"].items():
        assert calibration["objects"][object_name] == row


def test_registered_all18_calibration_is_exact_and_source_bound() -> None:
    raw = CALIBRATION.read_bytes()
    calibration = json.loads(raw)

    result = validate_all18_calibration(calibration)

    assert result["calibration_sha256"] == (
        "e94eeb9bdd2cc69e245b0bd48d843e5f64cb039e1eb02841e4a784cbe4dbc880"
    )
    assert hashlib.sha256(raw).hexdigest() == (
        "00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110"
    )
    assert len(calibration["source_artifact_file_sha256s"]) == 12

    wrong_inventory = deepcopy(calibration)
    source_hashes = wrong_inventory["source_artifact_file_sha256s"]
    source_hashes["unrelated_T1"] = source_hashes.pop(next(iter(source_hashes)))
    wrong_inventory["calibration_sha256"] = calibration_sha256(wrong_inventory)
    with pytest.raises(ValueError, match="hash inventory changed"):
        validate_all18_calibration(wrong_inventory)


def test_builder_fails_closed_when_fewer_than_three_new_objects_transfer() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    validation = validate_all18_source_protocol(protocol)
    selected = list(validation["selected_take_ids"])
    adjusted_objects = set(NEW_OBJECTS[:2])
    artifacts = {
        take_id: _smoke(
            take_id,
            protocol["protocol_sha256"],
            best=(2.0 if take_id.rpartition("_T")[0] in adjusted_objects else 1.0),
        )
        for take_id in selected
    }
    source_hashes = {
        take_id: f"{index:064x}" for index, take_id in enumerate(selected, start=1)
    }

    with pytest.raises(ValueError, match="extension gate failed"):
        build_all18_calibration(
            parent,
            protocol,
            artifacts,
            source_artifact_file_sha256s=source_hashes,
        )


def test_remote_wrapper_adds_zero_control_and_isolates_source_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_loader(_path):
        return {"payload": {"cohort": {"development_objects": ["FoamDice"]}}}

    def fake_run_smoke(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["authorized_objects"] = tuple(
            fake_module.load_pokeflex_registration_protocol(None)["payload"]["cohort"][
                "development_objects"
            ]
        )
        return {
            "schema_version": 1,
            "artifact_kind": ("PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke"),
        }

    fake_module = types.ModuleType("run_pokeflex_checkpoint_registration_smoke")
    fake_module.run_smoke = fake_run_smoke
    fake_module.load_pokeflex_registration_protocol = fake_loader
    monkeypatch.setitem(
        sys.modules,
        "run_pokeflex_checkpoint_registration_smoke",
        fake_module,
    )
    spec = importlib.util.spec_from_file_location(
        "all18_source_wrapper", REMOTE_WRAPPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    take_root = tmp_path / "3dPrintedBunny_T4"
    take_root.mkdir()
    output = tmp_path / "source.json"
    upstream = tmp_path / "upstream"
    checkpoint = tmp_path / "checkpoint"
    upstream.mkdir()
    checkpoint.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(REMOTE_WRAPPER),
            str(take_root),
            str(output),
            "--upstream-checkout",
            str(upstream),
            "--checkpoint-root",
            str(checkpoint),
        ],
    )

    module.main()

    scales = captured["kwargs"]["correction_scales"]
    assert scales == (0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5)
    assert "3dPrintedBunny" in captured["authorized_objects"]
    assert "additional_authorized_take_ids" not in captured["kwargs"]
    assert fake_module.load_pokeflex_registration_protocol is fake_loader
