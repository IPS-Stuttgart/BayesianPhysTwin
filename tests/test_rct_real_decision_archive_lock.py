from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/science/build_rct_real_decision_archive_lock_v1.py"
PROTOCOL_PATH = ROOT / "protocols/rct_real_decision_probe_v1.json"
CLARIFICATION_PATH = (
    ROOT / "protocols/rct_real_decision_probe_v1_preoutcome_clarification.json"
)
AMENDMENT_V2_PATH = (
    ROOT / "protocols/rct_real_decision_probe_v1_preoutcome_amendment_v2.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rct_archive_lock", BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_lock_hashes_without_opening_force_member(tmp_path: Path) -> None:
    builder = _module()
    archive = tmp_path / "rct_dataset.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "metadata/force_metadata.csv",
            "material_id,position,sensor,z_frame,raw_fz\nSECRET_FORCE_OUTCOME",
        )
        bundle.writestr("README.md", "metadata only")

    lock = builder._build_lock(
        archive,
        PROTOCOL_PATH,
        CLARIFICATION_PATH,
        AMENDMENT_V2_PATH,
        expected_archive_size=archive.stat().st_size,
    )

    identity = dict(lock)
    declared = identity.pop("lock_id")
    assert declared == content_id(identity)
    assert lock["force_metadata_member"] == "metadata/force_metadata.csv"
    assert lock["force_metadata_header_columns"] == [
        "material_id",
        "position",
        "sensor",
        "z_frame",
        "raw_fz",
    ]
    assert lock["archive_integrity_verified"] is True
    assert lock["force_metadata_header_opened"] is True
    assert lock["force_metadata_content_opened"] is False
    assert lock["confirmation_opened"] is False
    assert lock["held_v8_accessed"] is False


def test_archive_lock_rejects_ambiguous_force_members(tmp_path: Path) -> None:
    builder = _module()
    archive = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a/force_metadata.csv", "one")
        bundle.writestr("b/force_metadata.csv", "two")

    with pytest.raises(ValueError, match="exactly one"):
        builder._force_metadata_member(archive)


def test_archive_lock_rejects_wrong_byte_size(tmp_path: Path) -> None:
    builder = _module()
    archive = tmp_path / "rct_dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("force_metadata.csv", "metadata")

    with pytest.raises(ValueError, match="byte size"):
        builder._build_lock(
            archive,
            PROTOCOL_PATH,
            CLARIFICATION_PATH,
            AMENDMENT_V2_PATH,
            expected_archive_size=archive.stat().st_size + 1,
        )
