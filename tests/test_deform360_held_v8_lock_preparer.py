from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "held" / "prepare_deform360_v8_lock.py"
)
_SPEC = importlib.util.spec_from_file_location("deform360_v8_lock_preparer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
preparer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preparer)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "held-test",
            "GIT_AUTHOR_EMAIL": "held-test@example.invalid",
            "GIT_COMMITTER_NAME": "held-test",
            "GIT_COMMITTER_EMAIL": "held-test@example.invalid",
        },
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    (root / "scripts" / "held").mkdir(parents=True)
    operator = root / "scripts" / "held" / "prepare_deform360_v8_lock.py"
    operator.write_text("# frozen operator\n", encoding="utf-8")
    (root / "method.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "frozen test")
    return root


def _make_tree_writable(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(Path(current) / name, 0o600, follow_symlinks=False)
        for name in directories:
            os.chmod(Path(current) / name, 0o700, follow_symlinks=False)
    os.chmod(root, 0o700, follow_symlinks=False)


def test_repository_tree_binding_and_dirty_rejection(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    first = preparer._validate_repository(root)
    second = preparer._validate_repository(root)
    assert first["head"] == _git(root, "rev-parse", "HEAD")
    assert first["tree_sha256"] == second["tree_sha256"]
    assert first["head_text_sha256"] == preparer._sha256_text(first["head"])

    (root / "method.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is not completely clean"):
        preparer._validate_repository(root)


def test_staged_clone_is_independent_clean_and_read_only(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    source_provenance = preparer._validate_repository(source)
    stage = tmp_path / "stage"
    staged = preparer._clone_staged_deployment(source, source_provenance["head"], stage)
    try:
        assert staged["tree_sha256"] == source_provenance["tree_sha256"]
        assert _git(stage, "remote") == ""
        preparer._require_deployed_read_only(stage)
        assert not any(
            os.lstat(path).st_mode & 0o222 for path in [stage, *stage.rglob("*")]
        )
    finally:
        _make_tree_writable(stage)
        shutil.rmtree(stage)


def test_successor_lock_binds_attempt_one_preoutcome_withdrawal() -> None:
    pointer = preparer._EXPECTED_EXTERNAL_FILES[
        "v8_attempt1_preoutcome_withdrawal_pointer"
    ]
    report = preparer._EXPECTED_EXTERNAL_FILES[
        "v8_attempt1_preoutcome_withdrawal_report"
    ]

    assert pointer == (
        preparer._V8_ATTEMPT1_WITHDRAWAL_POINTER,
        "f7af6d1adf8541fd015cbe5336da97e013777c1bb711deaa01d9a84a49c81daa",
        0o400,
    )
    assert report == (
        preparer._V8_ATTEMPT1_WITHDRAWAL_REPORT,
        "c04a6e7a95d958950ea7e7c05e7e2b98ee4516c01f03e9284f85ccccf0f6873b",
        0o400,
    )


def test_prospective_bindings_include_named_deployment_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    operator = root / "scripts" / "held" / "prepare_deform360_v8_lock.py"
    sealed = tmp_path / "sealed.json"
    sealed.write_text("sealed\n", encoding="utf-8")
    os.chmod(sealed, 0o400)
    sealed_sha = preparer._sha256_file(sealed, role="test sealed", required_mode=0o400)
    fake_protocol = SimpleNamespace(
        FROZEN_FIELD_CONTRACT={"field": "frozen"},
        PRIMARY_METHOD={"method": "frozen"},
        held_contract_sha256=lambda value: preparer.hashlib.sha256(
            preparer._canonical_bytes(value)
        ).hexdigest(),
    )
    fake_replacement = SimpleNamespace(
        PROCESSING_CODE_REVISION="1" * 40,
        HF_DATASET_REVISION="2" * 40,
        REPLACEMENT_SOURCE_INVENTORY_CONTRACT={"source": "frozen"},
    )
    monkeypatch.setattr(preparer, "__file__", str(operator))
    monkeypatch.setattr(
        preparer,
        "_EXPECTED_EXTERNAL_FILES",
        {"sealed_parent": (sealed, sealed_sha, 0o400)},
    )
    monkeypatch.setattr(
        preparer,
        "_LOCAL_BINDING_FILES",
        {"held_v8_lock_preparer_source": "scripts/held/prepare_deform360_v8_lock.py"},
    )
    monkeypatch.setattr(preparer, "_validate_pinned_python", lambda: "9" * 64)
    monkeypatch.setattr(
        preparer,
        "_inherited_v7_bindings",
        lambda: {"frame_zero_default_config": "8" * 64},
    )
    monkeypatch.setattr(
        preparer, "_import_v8_modules", lambda code: (fake_protocol, fake_replacement)
    )
    monkeypatch.setattr(preparer, "_processing_revision", lambda: "1" * 40)

    bindings, provenance = preparer.prospective_bindings(root)

    assert bindings["sealed_parent"] == sealed_sha
    assert bindings["pinned_python_executable_target"] == "9" * 64
    assert bindings["frame_zero_default_config"] == "8" * 64
    assert "v7_inherited_immutable_bindings_contract" in bindings
    assert bindings["method_deployed_snapshot_tree"] == provenance["tree_sha256"]
    assert bindings["method_head_text_sha256"] == provenance["head_text_sha256"]
    assert "replacement_source_inventory_contract" in bindings
    assert all(len(value) == 64 for value in bindings.values())
