from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    load_poseit_range_transport_lock,
    poseit_mapping_constraints_file_sha256,
    poseit_method_lock_file_sha256,
    poseit_protocol_file_sha256,
    poseit_range_transport_lock_config_sha256,
    poseit_range_transport_lock_file_sha256,
)
from bayesian_phystwin_experiments.poseit_remote_archive import (
    RemoteArchiveExpectation,
    RemoteHashProgress,
    RemoteHashResult,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/acquire_poseit_gelsight_range_hash_v1.py"
PROTOCOL = ROOT / "protocols/poseit_real_decision_probe_v1.json"
MAPPING = (
    ROOT
    / "protocols"
    / "poseit_real_decision_probe_v1_preaccess_mapping_constraints.json"
)
METHOD = ROOT / "protocols/poseit_real_decision_probe_v1_method_lock.json"
TRANSPORT = ROOT / "protocols/poseit_real_decision_probe_v1_range_transport_lock.json"
CORE = ROOT / "src/bayesian_phystwin_experiments/poseit_remote_archive.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "poseit_range_hash_acquisition", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(
    tmp_path: Path,
    *,
    expected_transport_sha256: str | None = None,
    hasher: object | None = None,
) -> dict[str, object]:
    acquisition = _module()

    def exact_hasher(
        expectation: RemoteArchiveExpectation, **kwargs: object
    ) -> RemoteHashResult:
        progress = kwargs["progress"]
        assert callable(progress)
        chunks = (
            expectation.size_bytes + expectation.chunk_size_bytes - 1
        ) // expectation.chunk_size_bytes
        progress(
            RemoteHashProgress(
                completed_chunks=chunks,
                total_chunks=chunks,
                bytes_hashed=expectation.size_bytes,
                archive_size_bytes=expectation.size_bytes,
                transport_attempts=chunks,
            )
        )
        return RemoteHashResult(
            archive_sha256="a" * 64,
            archive_size_bytes=expectation.size_bytes,
            chunk_size_bytes=expectation.chunk_size_bytes,
            chunk_count=chunks,
            transport_attempts=chunks,
        )

    return acquisition._acquire_range_hash(
        tmp_path / "receipt.json",
        tmp_path / "progress.json",
        PROTOCOL,
        MAPPING,
        METHOD,
        TRANSPORT,
        CORE,
        expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
        expected_mapping_constraints_sha256=(
            poseit_mapping_constraints_file_sha256(MAPPING)
        ),
        expected_method_lock_sha256=poseit_method_lock_file_sha256(METHOD),
        expected_transport_lock_sha256=(
            expected_transport_sha256
            or poseit_range_transport_lock_file_sha256(TRANSPORT)
        ),
        hasher=hasher or exact_hasher,
    )


def test_range_transport_lock_is_parent_and_core_bound() -> None:
    lock = load_poseit_range_transport_lock(
        TRANSPORT,
        parent_protocol_path=PROTOCOL,
        mapping_constraints_path=MAPPING,
        method_lock_path=METHOD,
        range_transport_core_path=CORE,
    )

    assert lock["source_identity"]["archive_size_bytes"] == 905_738_058_282
    assert lock["execution"]["chunk_size_bytes"] == 33_554_432
    assert lock["execution"]["max_workers"] == 8
    assert lock["boundaries"]["archive_payload_retained"] is False
    assert (
        poseit_range_transport_lock_config_sha256(lock)
        == "3829b4fe9fa87964eb127838645288a9619df2fc695bfc6ca884602e5b1ac6da"
    )


def test_range_hash_receipt_is_content_bound_and_retains_no_archive(
    tmp_path: Path,
) -> None:
    result = _call(tmp_path)
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))

    assert receipt == result
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id")
    assert receipt_id == content_id(identity)
    assert receipt["archive_sha256"] == "a" * 64
    assert receipt["archive_size_bytes"] == 905_738_058_282
    assert receipt["archive_bytes_retained"] is False
    assert receipt["zip_central_directory_parsed"] is False
    assert receipt["archive_member_names_opened"] is False
    assert receipt["phase_labels_opened"] is False
    assert receipt["shake_outcomes_opened"] is False
    assert receipt["confirmation_opened"] is False
    assert receipt["held_v8_accessed"] is False
    assert progress["authoritative_result"] is True
    assert progress["receipt_id"] == receipt_id


def test_range_hash_rejects_transport_lock_drift_before_hash(tmp_path: Path) -> None:
    acquisition = _module()

    with pytest.raises(ValueError, match="transport-lock file SHA-256"):
        acquisition._acquire_range_hash(
            tmp_path / "receipt.json",
            tmp_path / "progress.json",
            PROTOCOL,
            MAPPING,
            METHOD,
            TRANSPORT,
            CORE,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(METHOD),
            expected_transport_lock_sha256="b" * 64,
            hasher=lambda expectation, **kwargs: pytest.fail("hashing was opened"),
        )


def test_range_hash_is_write_once(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("reserved", encoding="utf-8")
    acquisition = _module()

    with pytest.raises(ValueError, match="receipt already exists"):
        acquisition._acquire_range_hash(
            receipt,
            tmp_path / "progress.json",
            PROTOCOL,
            MAPPING,
            METHOD,
            TRANSPORT,
            CORE,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(METHOD),
            expected_transport_lock_sha256=(
                poseit_range_transport_lock_file_sha256(TRANSPORT)
            ),
            hasher=lambda expectation, **kwargs: pytest.fail("hashing was opened"),
        )


def test_range_hash_rejects_incomplete_hasher_result(tmp_path: Path) -> None:
    def incomplete(
        expectation: RemoteArchiveExpectation, **kwargs: object
    ) -> RemoteHashResult:
        return RemoteHashResult(
            archive_sha256="a" * 64,
            archive_size_bytes=expectation.size_bytes - 1,
            chunk_size_bytes=expectation.chunk_size_bytes,
            chunk_count=1,
            transport_attempts=1,
        )

    with pytest.raises(ValueError, match="wrong archive size"):
        _call(tmp_path, hasher=incomplete)

    assert not (tmp_path / "receipt.json").exists()


def test_terminal_response_failure_preserves_progress_without_receipt(
    tmp_path: Path,
) -> None:
    def failed_hasher(
        expectation: RemoteArchiveExpectation, **kwargs: object
    ) -> RemoteHashResult:
        progress = kwargs["progress"]
        assert callable(progress)
        chunks = (
            expectation.size_bytes + expectation.chunk_size_bytes - 1
        ) // expectation.chunk_size_bytes
        progress(
            RemoteHashProgress(
                completed_chunks=2,
                total_chunks=chunks,
                bytes_hashed=2 * expectation.chunk_size_bytes,
                archive_size_bytes=expectation.size_bytes,
                transport_attempts=3,
            )
        )
        raise ValueError("PoseIt range request did not return HTTP 206")

    with pytest.raises(ValueError, match="did not return HTTP 206"):
        _call(tmp_path, hasher=failed_hasher)

    assert not (tmp_path / "receipt.json").exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_chunks"] == 2
    assert progress["transport_attempts"] == 3
    for key in (
        "authoritative_result",
        "archive_member_names_opened",
        "member_payload_bytes_opened",
        "phase_labels_opened",
        "shake_outcomes_opened",
        "confirmation_opened",
        "held_v8_accessed",
    ):
        assert progress[key] is False


def test_registered_replacement_preserves_failure_and_exact_transport() -> None:
    replacement = json.loads(
        (
            ROOT / "protocols/poseit_real_decision_probe_v1_range_restart_v2.json"
        ).read_text(encoding="utf-8")
    )
    for binding in replacement["parent_artifacts"].values():
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == (binding["file_sha256"])
        )
    failure_root = ROOT / "evidence/poseit-real-decision-v1/range-hash-v1-failure"
    terminal = json.loads(
        (failure_root / "terminal_observation.json").read_text(encoding="utf-8")
    )
    for name, expected in terminal["retained_metadata"].items():
        assert (
            hashlib.sha256((failure_root / name).read_bytes()).hexdigest() == expected
        )
    progress = json.loads(
        (failure_root / "progress-v1.json").read_text(encoding="utf-8")
    )
    assert progress["authoritative_result"] is False
    assert terminal["archive_sha256"] is None
    assert terminal["completion_receipt_exists"] is False
    assert terminal["failure"]["actual_http_status"] is None
    assert (
        terminal["last_persisted_progress"]["total_issued_transport_attempts"] is None
    )
    assert terminal["structure_access_authorized"] is False
    assert terminal["scientific_result"] is False
    assert terminal["failure"]["message"] in (failure_root / "run-v1.log").read_text(
        encoding="utf-8"
    )
    headers = json.loads(
        (failure_root / "recovery_headers.json").read_text(encoding="utf-8")
    )
    assert headers["frozen_identity_headers_valid"] is True
    assert headers["response_body_read"] is False

    execution = replacement["execution"]
    command = execution["command"]
    assert command[:7] == [
        "flock",
        "-n",
        execution["lock_path"],
        "env",
        "PYTHONPATH=src",
        "python3",
        "scripts/science/acquire_poseit_gelsight_range_hash_v1.py",
    ]
    arguments = dict(zip(command[7::2], command[8::2], strict=True))
    for argument, path in (
        ("protocol", PROTOCOL),
        ("mapping-constraints", MAPPING),
        ("method-lock", METHOD),
        ("transport-lock", TRANSPORT),
    ):
        assert arguments[f"--{argument}"] == str(path.relative_to(ROOT))
        assert (
            arguments[f"--expected-{argument}-sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    assert arguments["--receipt"] == execution["receipt_path"]
    assert arguments["--progress"] == execution["progress_path"]
    assert arguments["--range-transport-core"] == str(CORE.relative_to(ROOT))
    assert execution["start_byte"] == 0
    assert execution["process_attempt_limit_for_this_record"] == 1
    assert execution["concurrent_process_limit"] == 1
    assert execution["overwrite_existing_files"] is False
    assert execution["scientific_method_changed"] is False
    assert Path(execution["progress_path"]).name == "progress-v2.json"
    assert Path(execution["log_path"]).name == "run-v2.log"
    assert not any(replacement["boundaries"].values())


def test_replacement_launch_metadata_binds_attempt_and_is_not_completion() -> None:
    root = ROOT / "evidence/poseit-real-decision-v1/range-hash-v2-launch"
    attempt_path = root / "attempt-v2.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    launch = json.loads((root / "launch-v2.json").read_text(encoding="utf-8"))
    record_path = ROOT / "protocols/poseit_real_decision_probe_v1_range_restart_v2.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert launch["attempt_ledger_sha256"] == hashlib.sha256(
        attempt_path.read_bytes()
    ).hexdigest()
    assert attempt["record_file_sha256"] == record_sha256
    assert launch["record_file_sha256"] == record_sha256
    assert attempt["command"] == record["execution"]["command"]
    assert attempt["cwd"] == record["execution"]["cwd"]
    assert attempt["attempt"] == 1
    assert attempt["start_byte"] == 0
    assert attempt["matching_acquisition_processes_before_launch"] == 0
    assert attempt["original_failure_artifacts_verified"] is True
    assert attempt["completion_receipt_absent_before_launch"] is True
    assert attempt["structure_access_authorized"] is False
    assert attempt["scientific_method_changed"] is False
    assert launch["completion_receipt"] is False
    assert launch["scientific_result"] is False
