# ruff: noqa: F403, F405
from deform360_calibration_acquisition_test_support import *

def test_ledger_retains_technical_failure_and_covers_all_ten_objects() -> None:
    plan = _plan()
    cases = [
        _failed_case(plan, plan.calibration_units[0]),
        *(
            _prepared_case(plan, unit)
            for unit in plan.calibration_units[1:]
        ),
    ]
    ledger = build_calibration_evidence_ledger(plan, cases)

    assert len(ledger.entries) == 10
    assert {entry.inference_role for entry in ledger.entries} == {"calibration_only"}
    assert {
        str(entry.metadata["object_id"]) for entry in ledger.entries
    } == {unit.object_id for unit in plan.calibration_units}
    failures = [
        entry
        for entry in ledger.entries
        if entry.metadata["technical_failure_retained"] is True
    ]
    assert len(failures) == 1
    assert failures[0].sensor_family == "technical-failure"


def test_raw_factor_identity_does_not_depend_on_processing_status() -> None:
    plan = _plan()
    unit = plan.calibration_units[0]
    prepared_cases = [_prepared_case(plan, item) for item in plan.calibration_units]
    failed_cases = [
        _failed_case(plan, unit),
        *(_prepared_case(plan, item) for item in plan.calibration_units[1:]),
    ]
    prepared_ledger = build_calibration_evidence_ledger(plan, prepared_cases)
    failed_ledger = build_calibration_evidence_ledger(plan, failed_cases)
    prepared_entry = next(
        entry
        for entry in prepared_ledger.entries
        if entry.metadata["object_id"] == unit.object_id
    )
    failed_entry = next(
        entry
        for entry in failed_ledger.entries
        if entry.metadata["object_id"] == unit.object_id
    )
    assert prepared_entry.raw_factor_sha256 == failed_entry.raw_factor_sha256
    assert prepared_entry.raw_factor_id == failed_entry.raw_factor_id
    assert prepared_entry.evidence_artifact_id != failed_entry.evidence_artifact_id


def test_result_validates_complete_accounting_and_rejects_duplicate_cases() -> None:
    plan = _plan()
    cases = [_prepared_case(plan, unit) for unit in plan.calibration_units]
    ledger = build_calibration_evidence_ledger(plan, cases)
    result = build_calibration_acquisition_result(
        plan,
        cases,
        ledger,
        source_artifacts={"acquisition-plan.json": _digest("plan-file")},
    )
    assert validate_calibration_acquisition_result(result) == result
    assert result["status"] == "complete"

    mutated = dict(result)
    case_ids = list(mutated["case_ids"])
    case_ids[-1] = case_ids[0]
    mutated["case_ids"] = case_ids
    descriptor = dict(mutated)
    descriptor.pop("result_id")
    mutated["result_id"] = content_id(descriptor)
    with pytest.raises(ValueError, match="unique"):
        validate_calibration_acquisition_result(mutated)


def test_runtime_cli_has_no_confirmation_or_target_surface() -> None:
    module = _script_module()
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "open_calibration_payloads" in destinations
    assert not any(
        forbidden in destination
        for destination in destinations
        for forbidden in ("confirmation", "target", "outcome")
    )

    required = [
        "--repository",
        ".",
        "--stage0-selection",
        "stage0.json",
        "--protocol",
        "protocol.json",
        "--visual-provider-lock",
        "provider.json",
        "--deform360-checkout",
        "deform360",
        "--data-root",
        "data",
        "--output-dir",
        "output",
        "--implementation-revision",
        "a" * 40,
    ]
    with pytest.raises(SystemExit, match="open-calibration-payloads"):
        module.main(required)


def test_released_bimanual_metadata_strings_are_supported() -> None:
    module = _script_module()
    assert module._LOCAL_PROCESSING_EPISODE_INDEX == 0
    metadata = {
        "sequences": {
            "2": {"bimanual": "yes"},
            "3": {"bimanual": "no"},
        }
    }
    assert module._metadata_bimanual(metadata, 2) is True
    assert module._metadata_bimanual(metadata, 3) is False

