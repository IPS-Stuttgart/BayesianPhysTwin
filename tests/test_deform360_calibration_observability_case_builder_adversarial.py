"""Adversarial branches for Deform360 observability case production."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_deform360_calibration_observability_case_builder as case_inputs

import bayesian_phystwin.deform360_calibration_observability_case_builder as builder
from bayesian_phystwin.deform360_calibration_observability_report import (
    load_deform360_calibration_observability_case,
)


def _successful_record() -> dict[str, Any]:
    return {
        "status": "succeeded",
        "exit_code": 0,
        "confirmation_boundary_verified": True,
        "confirmation_payloads_opened": False,
        "source_locks_valid": True,
        "plan_valid": True,
        "download_valid": True,
        "result_valid": True,
        "support_gate": {"support_passed": True},
    }


def test_ordinary_file_rejects_missing_directory_and_parent_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        builder._ordinary_file(tmp_path / "missing", name="source")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="ordinary file"):
        builder._ordinary_file(directory, name="source")

    real = tmp_path / "real"
    real.mkdir()
    (real / "value.txt").write_text("value\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        builder._ordinary_file(linked / "value.txt", name="source")


def test_read_ordinary_bytes_reports_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content\n", encoding="utf-8")

    def fail_read(_path: Path) -> bytes:
        raise OSError("synthetic read failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", fail_read)
        with pytest.raises(ValueError, match="cannot read source"):
            builder._read_ordinary_bytes(source, name="source")


def test_matrix_loader_rejects_wrong_suffix_scalar_and_non_numeric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_suffix = tmp_path / "matrix.bin"
    with wrong_suffix.open("wb") as stream:
        np.save(stream, np.eye(2), allow_pickle=False)
    with pytest.raises(ValueError, match="ordinary .npy file"):
        builder._load_npy_matrix(wrong_suffix, name="matrix")

    numeric = tmp_path / "numeric.npy"
    np.save(numeric, np.eye(2), allow_pickle=False)
    with monkeypatch.context() as patch:
        patch.setattr(builder.np, "load", lambda *_args, **_kwargs: 1.5)
        with pytest.raises(ValueError, match="one NumPy array"):
            builder._load_npy_matrix(numeric, name="matrix")

    non_numeric = tmp_path / "non-numeric.npy"
    np.save(non_numeric, np.asarray([["not-numeric"]]), allow_pickle=False)
    with pytest.raises(ValueError, match="real numeric values"):
        builder._load_npy_matrix(non_numeric, name="matrix")


@pytest.mark.parametrize(
    "matrix",
    (
        np.asarray([1.0, 2.0]),
        np.empty((0, 2)),
        np.asarray([[1.0, np.nan]]),
    ),
)
def test_matrix_loader_rejects_invalid_shape_empty_and_nonfinite(
    tmp_path: Path,
    matrix: np.ndarray,
) -> None:
    path = tmp_path / "invalid.npy"
    np.save(path, matrix, allow_pickle=False)

    with pytest.raises(ValueError, match="finite nonempty matrix"):
        builder._load_npy_matrix(path, name="matrix")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"status": "failed"}, "did not succeed"),
        ({"exit_code": 1}, "did not succeed"),
        ({"confirmation_boundary_verified": False}, "boundary is unverified"),
        ({"confirmation_payloads_opened": True}, "reports confirmation access"),
        ({"source_locks_valid": False}, "invalid source_locks_valid"),
        ({"plan_valid": False}, "invalid plan_valid"),
        ({"download_valid": False}, "invalid download_valid"),
        ({"result_valid": False}, "invalid result_valid"),
        ({"support_gate": None}, "support gate did not pass"),
        (
            {"support_gate": {"support_passed": False}},
            "support gate did not pass",
        ),
    ),
)
def test_successful_run_guard_rejects_each_invalid_terminal_state(
    mutation: dict[str, Any],
    message: str,
) -> None:
    record = _successful_record()
    record.update(mutation)

    with pytest.raises(ValueError, match=message):
        builder._require_successful_run(record)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ({}, "lacks artifact"),
        ({"artifact": "1" * 63}, "lacks artifact"),
        ({"artifact": "z" * 64}, "invalid artifact"),
    ),
)
def test_required_digest_rejects_missing_length_and_non_hex(
    value: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        builder._required_digest(value, "artifact", name="record")


def test_required_digest_accepts_literal_sha256() -> None:
    digest = "a" * 64
    assert (
        builder._required_digest({"artifact": digest}, "artifact", name="record")
        == digest
    )


def test_result_row_rejects_malformed_missing_duplicate_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="rows are missing"):
        builder._result_row(
            {"objects": "not-a-list"},
            object_id="object",
            episode_id=0,
            stratum="sheet",
        )

    with pytest.raises(ValueError, match="does not contain one object row"):
        builder._result_row(
            {"objects": []},
            object_id="object",
            episode_id=0,
            stratum="sheet",
        )

    row = {"object_id": "object", "episode_id": 0, "stratum": "sheet"}
    with pytest.raises(ValueError, match="does not contain one object row"):
        builder._result_row(
            {"objects": [row, dict(row)]},
            object_id="object",
            episode_id=0,
            stratum="sheet",
        )

    with pytest.raises(ValueError, match="changed object identity"):
        builder._result_row(
            {"objects": [row]},
            object_id="object",
            episode_id=1,
            stratum="sheet",
        )

    assert builder._result_row(
        {"objects": [row]},
        object_id="object",
        episode_id=0,
        stratum="sheet",
    ) == row


def test_context_rejects_surrounding_object_whitespace(tmp_path: Path) -> None:
    inputs = case_inputs._inputs(tmp_path)

    with pytest.raises(ValueError, match="surrounding whitespace"):
        case_inputs._evaluated(inputs, object_id=" cal-sheet-0")


@pytest.mark.parametrize(
    ("stage", "message"),
    (
        ("source", "source locks are invalid"),
        ("plan", "plan is invalid"),
        ("download", "download is invalid"),
        ("result", "result is invalid"),
    ),
)
def test_context_rejects_each_invalid_revalidated_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    message: str,
) -> None:
    inputs = case_inputs._inputs(tmp_path)

    with monkeypatch.context() as patch:
        if stage == "source":
            patch.setattr(
                builder,
                "source_lock_summary",
                lambda **_kwargs: (
                    {"source_locks_valid": False},
                    {},
                    frozenset(),
                ),
            )
        elif stage == "plan":
            patch.setattr(
                builder,
                "plan_summary",
                lambda *_args, **_kwargs: (
                    {"plan_valid": False},
                    frozenset(),
                    frozenset(),
                ),
            )
        elif stage == "download":
            patch.setattr(
                builder,
                "download_summary",
                lambda *_args, **_kwargs: {"download_valid": False},
            )
        else:
            patch.setattr(
                builder,
                "result_summary",
                lambda *_args, **_kwargs: {"result_valid": False},
            )
        with pytest.raises(ValueError, match=message):
            case_inputs._evaluated(inputs)


def test_context_detects_result_change_after_summary_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = case_inputs._inputs(tmp_path)
    original = builder.load_json_object

    def changed_result(path: Path) -> tuple[dict[str, Any], str]:
        value, digest = original(path)
        if path.resolve() == inputs.chain.result_path.resolve():
            return value, ("0" * 64 if digest != "0" * 64 else "1" * 64)
        return value, digest

    with monkeypatch.context() as patch:
        patch.setattr(builder, "load_json_object", changed_result)
        with pytest.raises(ValueError, match="changed after validation"):
            case_inputs._evaluated(inputs)


def test_failure_builder_rejects_whitespace_reason(tmp_path: Path) -> None:
    inputs = case_inputs._inputs(tmp_path)

    with pytest.raises(ValueError, match="surrounding whitespace"):
        builder.build_technical_failure_case_from_paths(
            **case_inputs._common(inputs, object_id="cal-sheet-0"),
            failure_evidence_path=inputs.failure_evidence_path,
            failure_reason=" whitespace ",
        )


def test_cli_publishes_technical_failure_case(tmp_path: Path) -> None:
    inputs = case_inputs._inputs(tmp_path)
    output = tmp_path / "technical-failure.json"
    arguments = [
        "technical-failure",
        "--source-protocol",
        str(inputs.chain.source_protocol_path),
        "--stage0-protocol",
        str(inputs.chain.stage0_protocol_path),
        "--selection-lock",
        str(inputs.chain.selection_path),
        "--visual-provider-lock",
        str(inputs.chain.provider_path),
        "--calibration-source-plan",
        str(inputs.chain.plan_path),
        "--calibration-source-download",
        str(inputs.chain.download_path),
        "--calibration-source-run-record",
        str(inputs.run_record_path),
        "--calibration-source-result",
        str(inputs.chain.result_path),
        "--object-id",
        "cal-sheet-0",
        "--implementation-revision",
        case_inputs.IMPLEMENTATION_REVISION,
        "--query-jacobian",
        str(inputs.query_path),
        "--failure-evidence",
        str(inputs.failure_evidence_path),
        "--failure-reason",
        "observability factorization failed",
        "--output",
        str(output),
    ]

    assert case_inputs.CLI.main(arguments) == 0
    case = load_deform360_calibration_observability_case(output)
    assert case.status == "technical_failure_without_replacement"
    assert case.failure_reason == "observability factorization failed"
