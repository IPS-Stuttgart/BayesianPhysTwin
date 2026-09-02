from __future__ import annotations

import importlib.util
import json
import stat
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.poseit_real_decision_protocol import (
    poseit_mapping_constraints_file_sha256,
    poseit_method_lock_file_sha256,
    poseit_protocol_file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/science/build_poseit_archive_structure_lock_v1.py"
PROTOCOL_PATH = ROOT / "protocols/poseit_real_decision_probe_v1.json"
MAPPING_CONSTRAINTS_PATH = (
    ROOT
    / "protocols"
    / "poseit_real_decision_probe_v1_preaccess_mapping_constraints.json"
)
METHOD_LOCK_PATH = ROOT / "protocols" / "poseit_real_decision_probe_v1_method_lock.json"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("poseit_archive_lock", BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("object-a/grasp-1/force-80/pose-01/metadata.json", "SECRET")
        bundle.writestr("object-a/grasp-1/force-80/pose-01/sensor.npy", b"PAYLOAD")
        bundle.writestr("object-b/grasp-2/force-40/pose-16/label.csv", "Pass")
    return path


def test_structure_lock_never_opens_a_member_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")

    def reject_member_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("member payload was opened")

    monkeypatch.setattr(zipfile.ZipFile, "open", reject_member_open)
    lock, private_bytes = builder._build_artifacts(
        archive,
        PROTOCOL_PATH,
        MAPPING_CONSTRAINTS_PATH,
        METHOD_LOCK_PATH,
        expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
        expected_mapping_constraints_sha256=(
            poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
        ),
        expected_method_lock_sha256=poseit_method_lock_file_sha256(METHOD_LOCK_PATH),
    )

    private = json.loads(private_bytes)
    lock_identity = dict(lock)
    lock_id = lock_identity.pop("lock_id")
    private_identity = dict(private)
    private_id = private_identity.pop("manifest_id")
    assert lock_id == content_id(lock_identity)
    assert private_id == content_id(private_identity)
    assert lock["structure"]["regular_member_count"] == 3
    assert lock["structure"]["top_level_component_count"] == 2
    assert lock["member_payload_bytes_opened"] is False
    assert lock["mapping_constraints_file_sha256"] == (
        poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
    )
    assert lock["method_lock_file_sha256"] == poseit_method_lock_file_sha256(
        METHOD_LOCK_PATH
    )
    assert lock["member_payload_integrity_verified"] is False
    assert lock["phase_labels_opened"] is False
    assert lock["sensor_payloads_opened"] is False
    assert lock["object_roles_assigned"] is False
    assert lock["confirmation_opened"] is False
    assert lock["held_v8_accessed"] is False
    assert [record["name"] for record in private["members"]] == sorted(
        record["name"] for record in private["members"]
    )


def test_structure_lock_rejects_protocol_drift(tmp_path: Path) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")

    with pytest.raises(ValueError, match="protocol file SHA-256"):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256="a" * 64,
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(
                METHOD_LOCK_PATH
            ),
        )


def test_structure_lock_rejects_mapping_constraint_drift(tmp_path: Path) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")

    with pytest.raises(ValueError, match="mapping-constraint file SHA-256"):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
            expected_mapping_constraints_sha256="a" * 64,
            expected_method_lock_sha256=poseit_method_lock_file_sha256(
                METHOD_LOCK_PATH
            ),
        )


def test_structure_lock_rejects_method_lock_drift(tmp_path: Path) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")

    with pytest.raises(ValueError, match="method-lock file SHA-256"):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
            ),
            expected_method_lock_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("member_name", "message"),
    (
        ("../outside", "traverses"),
        ("/absolute", "absolute"),
        ("folder\\member", "backslash"),
        ("C:/member", "drive prefix"),
    ),
)
def test_structure_lock_rejects_unsafe_member_paths(
    tmp_path: Path, member_name: str, message: str
) -> None:
    builder = _module()
    archive = tmp_path / "gelsight.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member_name, "metadata")

    with pytest.raises(ValueError, match=message):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(
                METHOD_LOCK_PATH
            ),
        )


def test_structure_lock_rejects_duplicate_members(tmp_path: Path) -> None:
    builder = _module()
    archive = tmp_path / "gelsight.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("same/name", "one")
            bundle.writestr("same/name", "two")

    with pytest.raises(ValueError, match="duplicate"):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(
                METHOD_LOCK_PATH
            ),
        )


def test_structure_lock_rejects_links(tmp_path: Path) -> None:
    builder = _module()
    archive = tmp_path / "gelsight.zip"
    link = zipfile.ZipInfo("linked")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "target")

    with pytest.raises(ValueError, match="link or special"):
        builder._build_artifacts(
            archive,
            PROTOCOL_PATH,
            MAPPING_CONSTRAINTS_PATH,
            METHOD_LOCK_PATH,
            expected_protocol_sha256=poseit_protocol_file_sha256(PROTOCOL_PATH),
            expected_mapping_constraints_sha256=(
                poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH)
            ),
            expected_method_lock_sha256=poseit_method_lock_file_sha256(
                METHOD_LOCK_PATH
            ),
        )


def test_cli_outputs_are_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")
    output = tmp_path / "lock.json"
    private = tmp_path / "private.json"
    output.write_text("reserved", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(BUILDER_PATH),
            "--archive",
            str(archive),
            "--protocol",
            str(PROTOCOL_PATH),
            "--expected-protocol-sha256",
            poseit_protocol_file_sha256(PROTOCOL_PATH),
            "--mapping-constraints",
            str(MAPPING_CONSTRAINTS_PATH),
            "--expected-mapping-constraints-sha256",
            poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH),
            "--method-lock",
            str(METHOD_LOCK_PATH),
            "--expected-method-lock-sha256",
            poseit_method_lock_file_sha256(METHOD_LOCK_PATH),
            "--private-member-manifest",
            str(private),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="already exists"):
        builder.main()


def test_cli_writes_content_bound_structure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _module()
    archive = _archive(tmp_path / "gelsight.zip")
    output = tmp_path / "lock.json"
    private = tmp_path / "private.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(BUILDER_PATH),
            "--archive",
            str(archive),
            "--protocol",
            str(PROTOCOL_PATH),
            "--expected-protocol-sha256",
            poseit_protocol_file_sha256(PROTOCOL_PATH),
            "--mapping-constraints",
            str(MAPPING_CONSTRAINTS_PATH),
            "--expected-mapping-constraints-sha256",
            poseit_mapping_constraints_file_sha256(MAPPING_CONSTRAINTS_PATH),
            "--method-lock",
            str(METHOD_LOCK_PATH),
            "--expected-method-lock-sha256",
            poseit_method_lock_file_sha256(METHOD_LOCK_PATH),
            "--private-member-manifest",
            str(private),
            "--output",
            str(output),
        ],
    )

    assert builder.main() == 0
    lock = json.loads(output.read_text(encoding="utf-8"))
    private_bytes = private.read_bytes()
    assert lock["private_member_manifest_sha256"] == builder._sha256_bytes(
        private_bytes
    )
    assert lock["archive_sha256"] == builder._sha256(archive)
    assert lock["member_payload_bytes_opened"] is False
