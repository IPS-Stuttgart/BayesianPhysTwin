from __future__ import annotations

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
