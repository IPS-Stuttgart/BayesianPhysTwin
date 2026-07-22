from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/held/run_deform360_v8_replacement_source.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "held_v8_source_launcher", LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_paths_and_environment_are_exact() -> None:
    launcher = _load_launcher()
    paths = launcher.formal_source_paths()
    held = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8")

    assert launcher.HELD_ROOT == held
    assert launcher.CALIBRATION_LOCK == held / "calibration-lock.json"
    assert paths.source_root == held / "replacement-source"
    assert paths.download_root == paths.source_root / "download"
    assert paths.aligned_root == paths.source_root / "aligned"
    assert (
        paths.inventory_manifest
        == paths.source_root / "manifests/remote-inventory.json"
    )
    assert (
        paths.content_manifest
        == paths.source_root / "manifests/downloaded-content.json"
    )
    assert (
        paths.aligned_source_manifest
        == paths.source_root / "manifests/aligned-source.json"
    )
    assert launcher.PROCESSING_CODE == Path(
        "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
        "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
    )
    assert launcher.PROCESSING_REVISION == "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
    assert launcher.PROCESSING_TREE == "c566ed29db7e0fd6a4cb768d840a4aa662864680"
    assert launcher.EXPECTED_HOST == "workstation2"

    code = held / ("code-" + "1" * 40)
    environment = launcher.normalized_environment(code)
    assert environment[launcher.CODE_ENVIRONMENT_KEY] == str(code)
    assert environment[launcher.NORMALIZED_MARKER] == "1"
    assert environment["HF_HOME"] == str(paths.hf_home)
    assert environment["TMPDIR"] == str(paths.temporary_root)
    assert "HF_HUB_OFFLINE" not in environment
    assert "TRANSFORMERS_OFFLINE" not in environment
    assert not any("PROXY" in key.upper() for key in environment)


def test_git_tree_canonicalization_matches_frozen_algorithm() -> None:
    launcher = _load_launcher()
    unsorted_raw = (
        b"100644 blob " + b"1" * 40 + b"\tsrc/a.py\0"
        b"100755 blob " + b"2" * 40 + b"\tscripts/run.py\0"
    )
    raw = (
        b"100755 blob " + b"2" * 40 + b"\tscripts/run.py\0"
        b"100644 blob " + b"1" * 40 + b"\tsrc/a.py\0"
    )
    records = launcher.parse_git_tree(raw)
    expected = [
        {
            "mode": "100755",
            "type": "blob",
            "object_id": "2" * 40,
            "path": "scripts/run.py",
        },
        {
            "mode": "100644",
            "type": "blob",
            "object_id": "1" * 40,
            "path": "src/a.py",
        },
    ]
    assert records == expected
    canonical = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    assert (
        launcher.git_tree_records_sha256(records)
        == hashlib.sha256(canonical).hexdigest()
    )

    with pytest.raises(ValueError, match="not a blob"):
        launcher.parse_git_tree(b"040000 tree " + b"3" * 40 + b"\tsrc\0")
    with pytest.raises(ValueError, match="order changed"):
        launcher.parse_git_tree(unsorted_raw)


def test_lock_must_bind_every_deployed_source_and_tree() -> None:
    launcher = _load_launcher()
    expected = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in launcher._REQUIRED_BINDING_KEYS
    }
    lock = {"immutable_bindings": {**expected, "another_frozen_input": "f" * 64}}
    launcher.validate_required_bindings(lock, expected)

    changed = dict(expected)
    changed["method_deployed_snapshot_tree"] = "0" * 64
    with pytest.raises(ValueError, match="deployment binding changed"):
        launcher.validate_required_bindings(lock, changed)


def test_deployed_git_provenance_requires_detached_clean_nonwritable_tree(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    build = tmp_path / "build"
    build.mkdir()
    subprocess.run(["git", "init", "-q", str(build)], check=True)
    subprocess.run(["git", "-C", str(build), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(build), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (build / "a.txt").write_text("a\n", encoding="utf-8")
    scripts = build / "scripts"
    scripts.mkdir()
    executable = scripts / "run.py"
    executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    executable.chmod(0o755)
    subprocess.run(["git", "-C", str(build), "add", "."], check=True)
    subprocess.run(["git", "-C", str(build), "commit", "-qm", "fixture"], check=True)
    subprocess.run(["git", "-C", str(build), "checkout", "-q", "--detach"], check=True)
    head = subprocess.run(
        ["git", "-C", str(build), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code = tmp_path / f"code-{head}"
    build.rename(code)
    for directory, directories, files in os.walk(code):
        Path(directory).chmod(0o555)
        for name in directories:
            (Path(directory) / name).chmod(0o555)
        for name in files:
            path = Path(directory) / name
            mode = stat.S_IMODE(os.lstat(path).st_mode)
            path.chmod(0o555 if mode & 0o111 else 0o444)
    try:
        provenance = launcher.deployed_git_provenance(code)
        assert provenance["head"] == head
        assert (
            provenance["head_text_sha256"]
            == hashlib.sha256(head.lower().encode("ascii")).hexdigest()
        )
        assert provenance["tree_records_sha256"] == launcher.git_tree_records_sha256(
            provenance["tree_records"]
        )
        assert [record["path"] for record in provenance["tree_records"]] == [
            "a.txt",
            "scripts/run.py",
        ]
    finally:
        for directory, directories, files in os.walk(code, topdown=False):
            for name in files:
                (Path(directory) / name).chmod(0o600)
            for name in directories:
                (Path(directory) / name).chmod(0o700)
            Path(directory).chmod(0o700)


def test_pinned_python_and_runtime_bindings_are_replayed(tmp_path: Path) -> None:
    launcher = _load_launcher()
    target = tmp_path / "python3.12"
    target.write_bytes(b"pinned Python fixture")
    target.chmod(0o500)
    lexical = tmp_path / "python"
    lexical.symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    source_module = SimpleNamespace(HF_DATASET_REVISION="f" * 40)
    bindings = {
        "pinned_python_executable_target": digest,
        "deform360_processing_head_text_sha256": hashlib.sha256(
            launcher.PROCESSING_REVISION.encode("ascii")
        ).hexdigest(),
        "hf_dataset_revision_text_sha256": hashlib.sha256(
            source_module.HF_DATASET_REVISION.encode("ascii")
        ).hexdigest(),
    }
    lock = {"immutable_bindings": bindings}

    launcher.validate_pinned_python(
        lock,
        launcher=lexical,
        expected_link_target=str(target),
        expected_resolved=target,
        expected_sha256=digest,
    )
    launcher.validate_runtime_bindings(lock, source_module=source_module)
    bindings["hf_dataset_revision_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime binding changed"):
        launcher.validate_runtime_bindings(lock, source_module=source_module)


def test_launcher_issues_and_consumes_source_capability_in_one_call(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    events: list[str] = []
    permit = object()
    expected_permit = {
        "case_name": "072-cotton-clohesline-ep0003",
        "operation": "acquire-aligned-replacement-source-v1",
    }

    def authorize(lock: Path) -> object:
        assert lock == launcher.CALIBRATION_LOCK
        events.append("authorize")
        return permit

    def evidence(lock: Path) -> dict[str, str]:
        assert lock == launcher.CALIBRATION_LOCK
        events.append("evidence")
        return expected_permit

    def consume(value: object, *, case_name: str, operation: str) -> dict[str, str]:
        assert value is permit
        assert case_name == expected_permit["case_name"]
        assert operation == expected_permit["operation"]
        events.append("consume-before-network")
        return expected_permit

    protocol = SimpleNamespace(
        authorize_replacement_source_acquisition=authorize,
        replacement_source_permit_evidence=evidence,
        consume_replacement_source_acquisition_capability=consume,
    )

    class ReplacementPaths:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    paths = launcher.FormalSourcePaths(
        source_root=tmp_path,
        download_root=tmp_path / "download",
        aligned_root=tmp_path / "aligned",
        manifest_root=tmp_path / "manifests",
        inventory_manifest=tmp_path / "manifests/inventory.json",
        content_manifest=tmp_path / "manifests/content.json",
        aligned_source_manifest=tmp_path / "manifests/aligned.json",
        temporary_root=tmp_path / "tmp",
        hf_home=tmp_path / "hf-home",
        cache_root=tmp_path / "cache",
        matplotlib_root=tmp_path / "matplotlib",
    )

    def acquire(
        replacement_paths: ReplacementPaths,
        *,
        source_permit: object,
        consume_source_permit: Any,
        expected_source_permit: dict[str, str],
        revision_reader: Any,
    ) -> Path:
        events.append("operator-enter")
        assert source_permit is permit
        assert expected_source_permit == expected_permit
        assert revision_reader is launcher.validate_processing_revision
        observed = consume_source_permit(
            source_permit,
            case_name=expected_permit["case_name"],
            operation=expected_permit["operation"],
        )
        assert observed == expected_permit
        events.append("network-would-start")
        assert (
            replacement_paths.aligned_source_manifest == paths.aligned_source_manifest
        )
        return replacement_paths.aligned_source_manifest

    source = SimpleNamespace(
        ReplacementSourcePaths=ReplacementPaths,
        acquire_and_align_replacement_source=acquire,
    )
    result = launcher.invoke_formal_operator(
        protocol_module=protocol,
        source_module=source,
        paths=paths,
    )

    assert result == paths.aligned_source_manifest
    assert events == [
        "authorize",
        "evidence",
        "operator-enter",
        "consume-before-network",
        "network-would-start",
    ]


def test_launcher_source_has_no_reuse_or_override_switch() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "formal replacement source root already exists; reuse is forbidden" in text
    assert "replacement-source launcher takes no arguments" in text
    assert "authorize_replacement_source_acquisition" in text
    assert "consume_replacement_source_acquisition_capability" in text
    assert "ALIGNED_SOURCE_MANIFEST" in text
    assert "HF_HUB_OFFLINE" not in text
