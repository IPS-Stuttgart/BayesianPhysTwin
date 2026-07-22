from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import pytest

import bayesian_phystwin.deform360_held_v8_protocol as protocol


def _write_json(path: Path, value: dict[str, Any], *, seal: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if seal:
        path.chmod(0o400)


def _bound_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _artifact(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["artifact_sha256"] = protocol.held_artifact_sha256(value)
    _write_json(path, value)
    return value


def _lock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    disclosed_payloads = {
        "v7_outcome_withdrawal_report": b"withdrawn-v7\n",
        "retired_case_official_target": b"retired-target\n",
        "retired_case_online_prediction": b"retired-online\n",
        "retired_case_online_prediction_seal": b"retired-seal\n",
    }
    disclosed_paths: dict[str, Path] = {}
    disclosed_specs: dict[str, tuple[int, str]] = {}
    for name, payload in disclosed_payloads.items():
        path = lineage / f"{name}.bin"
        path.write_bytes(payload)
        path.chmod(0o400)
        disclosed_paths[name] = path
        disclosed_specs[name] = (len(payload), hashlib.sha256(payload).hexdigest())
    withdrawal = disclosed_paths["v7_outcome_withdrawal_report"]
    monkeypatch.setattr(protocol, "V7_DISCLOSED_FILE_SPECS", disclosed_specs)
    monkeypatch.setattr(
        protocol,
        "V7_WITHDRAWAL_REPORT_FILE_SHA256",
        disclosed_specs["v7_outcome_withdrawal_report"][1],
    )

    development = lineage / "open27-decision.json"
    development.write_text("frozen-open27-decision\n", encoding="utf-8")
    development.chmod(0o400)
    development_sha = hashlib.sha256(development.read_bytes()).hexdigest()
    monkeypatch.setattr(
        protocol,
        "OPEN27_DEVELOPMENT_DECISION_FILE_SHA256",
        development_sha,
    )
    monkeypatch.setitem(
        protocol.FROZEN_FIELD_CONTRACT,
        "open27_development_decision_file_sha256",
        development_sha,
    )

    config = {"prediction_frame_count": 76, "test_fixture": True}
    root = tmp_path / "held-v8"
    fresh_root_capability = protocol.prepare_fresh_held_root(root)
    disclosure = root / "post-withdrawal-development-use-disclosure.json"
    disclosure_value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": protocol.POST_WITHDRAWAL_DISCLOSURE_KIND,
        "protocol_id": protocol.PROTOCOL_ID,
        "disclosed_v7_files": {
            name: {
                **_bound_file(disclosed_paths[name]),
                "mode_octal": "0400",
            }
            for name in sorted(disclosed_paths)
        },
        "post_withdrawal_development": {
            **protocol.POST_WITHDRAWAL_DEVELOPMENT_HASHES,
            "retired_official_target_opened_by_development_process": True,
            "retired_online_prediction_opened_by_development_process": True,
            "future_coordinates_or_masks_may_have_been_read": True,
            "derived_metrics_may_have_been_computed": True,
            "field_hypothesis_was_subsequently_reselected_on_independent_open27": True,
        },
        "retirement": {
            "exact_episode": protocol.RETIRED_V7_CASE_NAME,
            "replacement_episode": protocol.FRESH_REPLACEMENT_CASE_NAME,
            "replacement_search_excluded_entire_002_rope_silk_object": True,
            "reason": (
                "the exact held-v7 episode was exposed after formal withdrawal; "
                "the replacement was selected outside that object's episodes"
            ),
        },
        "v8_reuse_boundary": {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "all_v8_predictions_targets_queries_and_scores_must_be_fresh": True,
        },
        "claim_boundary": (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8 into an official Deform360 "
            "state-of-the-art comparison."
        ),
    }
    _artifact(disclosure, disclosure_value)
    lock = root / "calibration-lock.json"
    protocol.create_calibration_protocol_lock(
        lock,
        held_root=root,
        fresh_root_capability=fresh_root_capability,
        immutable_bindings={
            "frame_zero_default_config": protocol.held_contract_sha256(config),
            "test_operator_source": "a" * 64,
        },
        v7_withdrawal_report_path=withdrawal,
        post_withdrawal_disclosure_path=disclosure,
        development_decision_path=development,
    )
    return lock


def _first_barrier_artifacts(
    tmp_path: Path,
    lock: Path,
    *,
    role: str = "calibration",
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], Path | None]:
    cases = (
        protocol.CALIBRATION_CASE_NAMES
        if role == "calibration"
        else protocol.CONFIRMATION_CASE_NAMES
    )
    physical: dict[str, Path] = {}
    online: dict[str, Path] = {}
    fields: dict[str, Path] = {}
    for case_name in cases:
        case_root = tmp_path / role / case_name
        physical_path = case_root / "physical.json"
        online_path = case_root / "online.json"
        field_path = case_root / "field.json"
        physical_value = _artifact(
            physical_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "role": role,
                "lock": _bound_file(lock),
                "kind": "test-physical",
            },
        )
        online_value = _artifact(
            online_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "role": role,
                "lock": _bound_file(lock),
                "kind": "test-online",
            },
        )
        _artifact(
            field_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "online_prediction_seal": _bound_file(online_path),
                "online_prediction_seal_artifact_sha256": online_value[
                    "artifact_sha256"
                ],
                "kind": "test-frozen-field",
            },
        )
        assert physical_value["artifact_sha256"]
        physical[case_name] = physical_path
        online[case_name] = online_path
        fields[case_name] = field_path
    source_path: Path | None = None
    if role == "calibration":
        source_path = tmp_path / role / "replacement-aligned-source.json"
        _artifact(
            source_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": protocol.FRESH_REPLACEMENT_CASE_NAME,
                "source_permit": protocol.replacement_source_permit_evidence(lock),
                "kind": "test-aligned-replacement-source",
            },
        )
    return physical, online, fields, source_path


def _second_barrier_artifacts(
    tmp_path: Path,
    lock: Path,
    *,
    role: str = "calibration",
) -> tuple[dict[str, Path], dict[str, Path]]:
    cases = (
        protocol.CALIBRATION_CASE_NAMES
        if role == "calibration"
        else protocol.CONFIRMATION_CASE_NAMES
    )
    queries: dict[str, Path] = {}
    queried: dict[str, Path] = {}
    for case_name in cases:
        case_root = tmp_path / role / case_name
        query_path = case_root / "official-x0.json"
        queried_path = case_root / "queried.json"
        query_value = _artifact(
            query_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "kind": "test-official-x0",
            },
        )
        _artifact(
            queried_path,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "case_name": case_name,
                "lock": _bound_file(lock),
                "official_query_manifest": _bound_file(query_path),
                "official_query_manifest_artifact_sha256": query_value[
                    "artifact_sha256"
                ],
                "kind": "test-queried-prediction",
            },
        )
        queries[case_name] = query_path
        queried[case_name] = queried_path
    return queries, queried


def _validator(
    path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str,
    expected_role: str,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["lock"] == _bound_file(Path(lock_path))
    assert value["case_name"] == expected_case_name
    if "role" in value:
        assert value["role"] == expected_role
    assert value["artifact_sha256"] == protocol.held_artifact_sha256(value)
    return value


def _field_validator(
    path: str | Path,
    *,
    lock_path: str | Path,
    expected_case_name: str,
) -> dict[str, Any]:
    return _validator(
        path,
        lock_path,
        expected_case_name=expected_case_name,
        expected_role="calibration",
    )


def _source_validator(
    path: str | Path,
    *,
    expected_source_permit: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["source_permit"] == expected_source_permit
    assert value["artifact_sha256"] == protocol.held_artifact_sha256(value)
    return value


def test_lock_replaces_only_retired_case_and_binds_frozen_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = _lock_fixture(tmp_path, monkeypatch)
    lock = protocol.validate_protocol_lock(lock_path)

    assert len(protocol.CALIBRATION_CASE_NAMES) == 15
    assert protocol.RETIRED_V7_CASE_NAME not in protocol.CALIBRATION_CASE_NAMES
    assert (
        protocol.CALIBRATION_CASE_NAMES.count(protocol.FRESH_REPLACEMENT_CASE_NAME) == 1
    )
    assert protocol.CONFIRMATION_CASE_NAMES == (
        "002-rope-silk-ep0001",
        "081-stripe-rope-ep0005",
        "085-scarf-cloth-ep0002",
        "083-blanket-cloth-ep0007",
        "092-squirrel-ep0001",
        "170-spider-ep0006",
    )
    assert lock["frozen_field_contract"]["operator_id"] == (
        "gaussian-knn-normalized-v1"
    )
    assert lock["frozen_field_contract"]["neighbor_count"] == 4
    assert lock["frozen_field_contract"]["length_scale_fraction"] == 0.05
    assert lock["frozen_field_contract"]["support_radius_fraction"] == 0.5
    assert lock["frozen_field_contract"]["frame_indices"] == list(range(76))
    assert lock["freshness_and_reuse"]["v7_execution_artifacts_reused"] is False

    source_permit = protocol.authorize_replacement_source_acquisition(lock_path)
    source_evidence = protocol.consume_replacement_source_acquisition_capability(
        source_permit,
        case_name=protocol.FRESH_REPLACEMENT_CASE_NAME,
        operation=protocol.REPLACEMENT_SOURCE_OPERATION,
    )
    assert source_evidence == protocol.replacement_source_permit_evidence(lock_path)
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_replacement_source_acquisition_capability(
            source_permit,
            case_name=protocol.FRESH_REPLACEMENT_CASE_NAME,
            operation=protocol.REPLACEMENT_SOURCE_OPERATION,
        )

    with pytest.raises(ValueError, match="must be absent"):
        protocol.prepare_fresh_held_root(tmp_path / "held-v8")


def test_barrier_one_is_complete_cohort_case_specific_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    incomplete = dict(fields)
    incomplete.pop(protocol.CALIBRATION_CASE_NAMES[-1])

    with pytest.raises(ValueError, match="every exact cohort"):
        protocol.authorize_target_reconstruction_capabilities(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=incomplete,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
            replacement_source_validator=_source_validator,
        )

    capabilities = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    permit = capabilities[case_name]
    with pytest.raises(ValueError, match="case or operation"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.FUTURE_SCORE_OPERATION,
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(permit)

    evidence = protocol.consume_case_capability(
        permit,
        case_name=case_name,
        operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
    )
    assert evidence["single_use_consumed"] is True
    assert evidence["process_local_capability"] is True
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )


def test_barrier_replay_spends_capability_when_a_seal_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    capabilities = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[3]
    permit = capabilities[case_name]
    changed = online[protocol.CALIBRATION_CASE_NAMES[-1]]
    original = changed.read_bytes()
    changed.chmod(0o600)
    changed.write_bytes(original + b" ")
    changed.chmod(0o400)

    with pytest.raises((AssertionError, json.JSONDecodeError, ValueError)):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )
    changed.chmod(0o600)
    changed.write_bytes(original)
    changed.chmod(0o400)
    with pytest.raises(ValueError, match="already consumed"):
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )


def test_future_target_stays_closed_until_all_x0_queries_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    reconstruction = protocol.authorize_target_reconstruction_capabilities(
        lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        replacement_aligned_source_manifest_path=source,
        role="calibration",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
        replacement_source_validator=_source_validator,
    )
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    target_opened = False

    def try_open_future(permit: object) -> None:
        nonlocal target_opened
        protocol.consume_case_capability(
            permit,
            case_name=case_name,
            operation=protocol.FUTURE_SCORE_OPERATION,
        )
        target_opened = True

    with pytest.raises(ValueError, match="case or operation"):
        try_open_future(reconstruction[case_name])
    assert target_opened is False

    queries, queried = _second_barrier_artifacts(tmp_path, lock)
    incomplete = dict(queried)
    incomplete.pop(protocol.CALIBRATION_CASE_NAMES[-1])
    with pytest.raises(ValueError, match="every exact cohort"):
        protocol.authorize_future_score_capabilities(
            lock,
            official_query_manifest_paths=queries,
            queried_prediction_seal_paths=incomplete,
            role="calibration",
            official_query_validator=_validator,
            queried_prediction_validator=_validator,
        )
    assert target_opened is False

    with pytest.raises(ValueError, match="all reconstruction capabilities"):
        protocol.authorize_future_score_capabilities(
            lock,
            official_query_manifest_paths=queries,
            queried_prediction_seal_paths=queried,
            role="calibration",
            official_query_validator=_validator,
            queried_prediction_validator=_validator,
        )
    for reconstruction_case, permit in reconstruction.items():
        protocol.consume_case_capability(
            permit,
            case_name=reconstruction_case,
            operation=protocol.TARGET_RECONSTRUCTION_OPERATION,
        )

    scoring = protocol.authorize_future_score_capabilities(
        lock,
        official_query_manifest_paths=queries,
        queried_prediction_seal_paths=queried,
        role="calibration",
        official_query_validator=_validator,
        queried_prediction_validator=_validator,
    )
    try_open_future(scoring[case_name])
    assert target_opened is True


def _gate_decision(
    path: Path,
    lock: Path,
    *,
    passed: bool,
) -> None:
    score = path.parent / ("score-go.json" if passed else "score-no.json")
    _write_json(score, {"sealed": True})
    value: dict[str, Any] = {
        "schema_version": protocol.SCHEMA_VERSION,
        "artifact_kind": protocol.CALIBRATION_DECISION_KIND,
        "protocol_id": protocol.PROTOCOL_ID,
        "role": "calibration",
        "lock": _bound_file(lock),
        "ordered_case_names": list(protocol.CALIBRATION_CASE_NAMES),
        "barrier_two_sha256": "b" * 64,
        "score_evidence": _bound_file(score),
        "gate_result": {
            "gate": "v8-calibration-go-no-go-v1",
            "passed": passed,
        },
        "decision": "GO" if passed else "NO-GO",
    }
    _artifact(path, value)


def test_confirmation_is_inaccessible_until_a_sealed_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_lock = _lock_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="until calibration GO"):
        protocol.validate_first_cohort_barrier(
            calibration_lock,
            physical_seal_paths={},
            online_seal_paths={},
            frozen_field_manifest_paths={},
            role="confirmation",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
        )

    no_go = tmp_path / "held-v8" / "calibration-no-go.json"
    _gate_decision(no_go, calibration_lock, passed=False)
    with pytest.raises(ValueError, match="after calibration NO-GO"):
        protocol.create_confirmation_protocol_lock(
            tmp_path / "held-v8" / "confirmation-lock.json",
            calibration_lock,
            no_go,
        )

    go = tmp_path / "held-v8" / "calibration-go.json"
    _gate_decision(go, calibration_lock, passed=True)
    confirmation_lock = tmp_path / "held-v8" / "confirmation-lock.json"
    protocol.create_confirmation_protocol_lock(
        confirmation_lock,
        calibration_lock,
        go,
    )
    assert (
        protocol.locked_case_names(confirmation_lock, role="confirmation")
        == protocol.CONFIRMATION_CASE_NAMES
    )
    physical, online, fields, source = _first_barrier_artifacts(
        tmp_path / "post-go", confirmation_lock, role="confirmation"
    )
    assert source is None
    evidence = protocol.validate_first_cohort_barrier(
        confirmation_lock,
        physical_seal_paths=physical,
        online_seal_paths=online,
        frozen_field_manifest_paths=fields,
        role="confirmation",
        physical_validator=_validator,
        online_validator=_validator,
        frozen_field_validator=_field_validator,
    )
    assert evidence.role == "confirmation"
    assert evidence.ordered_case_names == protocol.CONFIRMATION_CASE_NAMES


def test_v7_execution_paths_are_never_admitted_to_a_v8_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    physical, online, fields, source = _first_barrier_artifacts(tmp_path, lock)
    case_name = protocol.CALIBRATION_CASE_NAMES[0]
    v7_root = tmp_path / "held-v7" / case_name
    v7_physical = v7_root / "physical.json"
    v7_physical.parent.mkdir(parents=True)
    v7_physical.write_bytes(physical[case_name].read_bytes())
    v7_physical.chmod(0o400)
    physical[case_name] = v7_physical

    with pytest.raises(ValueError, match="held-v7 execution artifact"):
        protocol.validate_first_cohort_barrier(
            lock,
            physical_seal_paths=physical,
            online_seal_paths=online,
            frozen_field_manifest_paths=fields,
            replacement_aligned_source_manifest_path=source,
            role="calibration",
            physical_validator=_validator,
            online_validator=_validator,
            frozen_field_validator=_field_validator,
            replacement_source_validator=_source_validator,
        )


def test_v8_seal_creators_freeze_fresh_builder_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _lock_fixture(tmp_path, monkeypatch)
    case_name = protocol.CALIBRATION_CASE_NAMES[1]
    case_root = tmp_path / "held-v8" / "calibration" / case_name
    frame_zero = case_root / "frame-zero.json"
    frame_zero_value = _artifact(
        frame_zero,
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "case_name": case_name,
            "role": "calibration",
            "artifact_kind": protocol.FRAME_ZERO_KIND,
        },
    )

    def validate_frame_zero(
        path: str | Path,
        lock_path: str | Path,
        *,
        expected_case_name: str | None = None,
        expected_role: str | None = None,
    ) -> dict[str, Any]:
        assert Path(path) == frame_zero
        assert Path(lock_path) == lock
        assert expected_case_name in {None, case_name}
        assert expected_role in {None, "calibration"}
        return frame_zero_value

    monkeypatch.setattr(
        protocol,
        "validate_frame_zero_bundle_manifest",
        validate_frame_zero,
    )
    physical_artifacts: dict[str, Path] = {}
    for role in protocol.PHYSICAL_ARTIFACT_ROLES:
        path = case_root / "physical" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))
        path.chmod(0o600)
        physical_artifacts[role] = path
    physical_seal = case_root / "physical" / "seal.json"
    protocol.create_physical_prior_seal(
        physical_seal,
        lock,
        frame_zero,
        physical_artifacts,
        case_name=case_name,
        role="calibration",
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o400 for path in physical_artifacts.values()
    )

    prefix = case_root / "online" / "prefix.json"
    prefix.parent.mkdir(parents=True)
    protocol.create_prefix_stage_authorization(prefix, lock, physical_seal)
    online_artifacts: dict[str, Path] = {}
    for role in protocol.ONLINE_ARTIFACT_ROLES:
        path = case_root / "online" / f"{role}.bin"
        path.write_bytes(role.encode("ascii"))
        path.chmod(0o600)
        online_artifacts[role] = path
    online_seal = case_root / "online" / "seal.json"
    protocol.create_online_prediction_seal(
        online_seal,
        lock,
        prefix,
        online_artifacts,
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o400 for path in online_artifacts.values()
    )
