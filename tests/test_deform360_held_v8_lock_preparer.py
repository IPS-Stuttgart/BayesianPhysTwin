from __future__ import annotations

import ast
import importlib.util
import json
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
        REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256="7" * 64,
        query_artifacts=SimpleNamespace(
            CENTER_EXCLUSION_CONTRACT_SHA256="5" * 64
        ),
        frame_zero_assets=SimpleNamespace(
            EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256="6" * 64
        ),
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
    monkeypatch.setattr(
        preparer, "_validate_attempt2_operator_source_lineage", lambda _bindings: None
    )
    monkeypatch.setattr(
        preparer, "_validate_attempt3_archive_lineage", lambda _bindings: None
    )
    monkeypatch.setattr(
        preparer, "_validate_admission_replay_source_lineage", lambda _bindings: None
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
    assert bindings["replacement_automatic_twin_admission_contract"] == "7" * 64
    assert bindings["frame_zero_exact_eight_subset_bounded_audit_contract"] == "6" * 64
    assert bindings["center_exclusion_contract"] == "5" * 64
    assert bindings["v8_attempt3_postseal_noncode_inventory"] == (
        preparer._V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
    )
    assert bindings["v8_attempt3_postseal_noncode_inventory_contract"] == (
        preparer.hashlib.sha256(
            preparer._canonical_bytes(
                preparer._attempt3_archive_inventory_contract()
            )
        ).hexdigest()
    )
    assert all(len(value) == 64 for value in bindings.values())


def test_attempt_three_binds_attempt_two_lineage_and_operator_sources() -> None:
    expected = preparer._EXPECTED_EXTERNAL_FILES
    assert expected["v8_attempt2_preoutcome_withdrawal_pointer"] == (
        preparer._V8_ATTEMPT2_WITHDRAWAL_POINTER,
        "007d3fbde0dc93dc350661aafdd5d08d1398aa8d1f164e17bf295521fc40463a",
        0o400,
    )
    assert expected["v8_attempt2_preoutcome_withdrawal_report"] == (
        preparer._V8_ATTEMPT2_WITHDRAWAL_REPORT,
        "5830f9bfe8d29d5a09f64afbcaeabadc3acb7c8fdf820c1aeb68a6601055a895",
        0o400,
    )
    assert expected["v8_attempt2_withdrawal_integrity_completion"] == (
        preparer._V8_ATTEMPT2_INTEGRITY_COMPLETION,
        "21e7695af5f610193502ecb6e7e6c647d853bde34daa1c5f362e990dffdf56a7",
        0o400,
    )
    expected_attempt2_artifacts = {
        "v8_attempt2_preoutcome_withdrawal_pointer": (
            "9063011657b955902d1cf7d85a4253eee65caa430a41edae2709a18032baf99c"
        ),
        "v8_attempt2_preoutcome_withdrawal_report": (
            "457c6a64c0208b91ee5eb0f8038d22ae7eda743e29fb60a4bcb4ef1a2861b147"
        ),
        "v8_attempt2_withdrawal_integrity_completion": (
            "eb3a6c092a84dd95f516770d9837711a4f5b1eb58a28fee84c6df0bddb4999b0"
        ),
        "v8_attempt2_manifest_scale_diagnostic": (
            "96f7edc666cda3cf84c6121623028c290b577ceec62cc104a41780b7bb6560ce"
        ),
        "v8_attempt2_admission_compatibility_diagnostic": (
            "e659ceb9b4120c9a2e0c2bf33cbc8478bfc0157ed9b4f9415c3ebef194ea3f80"
        ),
    }
    assert {
        name: preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[name]
        for name in expected_attempt2_artifacts
    } == expected_attempt2_artifacts
    assert preparer._LOCAL_BINDING_FILES["frame_zero_builder_source"] == (
        "src/bayesian_phystwin/deform360_frame_zero_assets.py"
    )
    assert "held_v8_attempt2_withdrawal_operator_source" in (
        preparer._LOCAL_BINDING_FILES
    )
    assert "held_v8_attempt2_withdrawal_integrity_completion_operator_source" in (
        preparer._LOCAL_BINDING_FILES
    )


def test_attempt_four_binds_exact_attempt_three_lineage() -> None:
    expected = preparer._EXPECTED_EXTERNAL_FILES
    assert expected["v8_attempt3_postbarrier_withdrawal_report"] == (
        preparer._V8_ATTEMPT3_WITHDRAWAL_REPORT,
        "6d9c62606d18744d275df51fd08e041205bf15b38175d74c69690eafd511054b",
        0o400,
    )
    assert expected["v8_attempt3_postbarrier_withdrawal_pointer"] == (
        preparer._V8_ATTEMPT3_WITHDRAWAL_POINTER,
        "75acc7e9535f41528d22739ae8eeb5a0a2247c0fe63c097ad1da2859d7b33246",
        0o400,
    )
    assert expected["v8_attempt3_withdrawal_integrity_completion"] == (
        preparer._V8_ATTEMPT3_INTEGRITY_COMPLETION,
        "f3d1e8a6670484c81ac04743bcdb020cdee3fba02229a64844a8a9c9f4b8b989",
        0o400,
    )
    assert preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
        "v8_attempt3_postbarrier_withdrawal_report"
    ] == "4b7404961fa13b418265f76827dda356fb6ad019db764c6302f49e8149d05de2"
    assert preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
        "v8_attempt3_postbarrier_withdrawal_pointer"
    ] == "6ef596a63029d7fa8346141bb52c72d99062e201a12b7c9baf4fca7330baca64"
    assert preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
        "v8_attempt3_withdrawal_integrity_completion"
    ] == "9ec2989e3000464a0f72b038e26fe407403e02721e21c19ae4fb9123c6a7cf8c"
    assert preparer._attempt3_archive_inventory_contract() == {
        "archive_path": str(preparer._V8_ATTEMPT3_ARCHIVE),
        "postseal_noncode_entry_count": 1466,
        "postseal_noncode_inventory_sha256": (
            "5d398e998e2b738db545ffefd254712c6822017cfc5be6e7de435d5883c8c4c8"
        ),
    }
    relative_operator = preparer._LOCAL_BINDING_FILES[
        "held_v8_attempt3_withdrawal_operator_source"
    ]
    assert relative_operator == (
        "scripts/held/seal_deform360_v8_attempt3_outcome_failure.py"
    )
    assert preparer._sha256_file(
        Path(__file__).parents[1] / relative_operator,
        role="attempt-3 withdrawal operator source",
    ) == "bc6efe5660c90828be13fb9221472c5e37261e5041509ff61403ea89ef3e9648"


def test_attempt_four_replay_uses_new_fail_closed_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert str(preparer._V8_ADMISSION_REPLAY_ROOT).endswith(
        "bpt-held-v8.1-attempt-4-admission-wrapper-scratch-20260722"
    )
    assert preparer._V8_ADMISSION_REPLAY_REPORT_FILE_SHA256 is None
    assert preparer._V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256 is None
    assert preparer._V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256 is None
    assert preparer._V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256 is None
    monkeypatch.setattr(
        preparer,
        "_EXPECTED_EXTERNAL_FILES",
        {
            "v8_external_admission_metadata_only_replay": (
                preparer._V8_ADMISSION_REPLAY_REPORT,
                None,
                0o400,
            )
        },
    )
    monkeypatch.setattr(
        preparer,
        "_EXPECTED_EXTERNAL_ARTIFACT_SHA256",
        {"v8_external_admission_metadata_only_replay": None},
    )
    with pytest.raises(ValueError, match="placeholder is not populated"):
        preparer._external_bindings()


def test_attempt_three_archive_lineage_matches_local_operator_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "held-v8-attempt-3-withdrawn-postbarrier"
    archive.mkdir()
    report = archive / "execution-withdrawal-postbarrier-attempt3.json"
    pointer = tmp_path / "held-v8-attempt-3-withdrawal-pointer.json"
    completion = tmp_path / "held-v8-attempt-3-withdrawal-completion.json"
    operator = {"sha256": preparer._V8_ATTEMPT3_OPERATOR_SOURCE_SHA256}
    common = {
        "protocol_id": preparer._V8_ATTEMPT3_PROTOCOL_ID,
        "execution_attempt": preparer._V8_ATTEMPT3_EXECUTION_ATTEMPT,
        "disposition": preparer._V8_ATTEMPT3_DISPOSITION,
        "executed_withdrawal_operator_source": operator,
    }
    report_value = {
        **common,
        "status": preparer._V8_ATTEMPT3_WITHDRAWAL_STATUS,
        "immutable_archive_path": str(archive),
        "expected_postseal_inventory": {
            "entry_count": preparer._V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT,
            "inventory_sha256": preparer._V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256,
        },
    }
    completion_value = {
        **common,
        "status": preparer._V8_ATTEMPT3_COMPLETION_STATUS,
        "archive_path": str(archive),
        "archive_fully_nonwritable": True,
        "archive_root_mode_octal": "0500",
        "postseal_noncode_entry_count": preparer._V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT,
        "postseal_noncode_inventory_sha256": (
            preparer._V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
        ),
        "withdrawal_report_file_sha256": preparer._V8_ATTEMPT3_REPORT_FILE_SHA256,
        "withdrawal_report_artifact_sha256": (
            preparer._V8_ATTEMPT3_REPORT_ARTIFACT_SHA256
        ),
    }
    pointer_value = {
        **completion_value,
        "status": preparer._V8_ATTEMPT3_WITHDRAWAL_STATUS,
        "withdrawal_integrity_completion": {
            "path": str(completion),
            "file_sha256": preparer._V8_ATTEMPT3_COMPLETION_FILE_SHA256,
            "artifact_sha256": preparer._V8_ATTEMPT3_COMPLETION_ARTIFACT_SHA256,
        },
    }
    for path, value in (
        (report, report_value),
        (pointer, pointer_value),
        (completion, completion_value),
    ):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        os.chmod(path, 0o400)
    os.chmod(archive, 0o500)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_ARCHIVE", archive)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_WITHDRAWAL_REPORT", report)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_WITHDRAWAL_POINTER", pointer)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_INTEGRITY_COMPLETION", completion)
    bindings = {
        "held_v8_attempt3_withdrawal_operator_source": (
            preparer._V8_ATTEMPT3_OPERATOR_SOURCE_SHA256
        )
    }

    preparer._validate_attempt3_archive_lineage(bindings)

    os.chmod(pointer, 0o600)
    pointer_value["postseal_noncode_entry_count"] = 1465
    pointer.write_text(json.dumps(pointer_value) + "\n", encoding="utf-8")
    os.chmod(pointer, 0o400)
    with pytest.raises(ValueError, match="pointer archive or report lineage"):
        preparer._validate_attempt3_archive_lineage(bindings)
    with pytest.raises(ValueError, match="observed executed source"):
        preparer._validate_attempt3_archive_lineage(
            {"held_v8_attempt3_withdrawal_operator_source": "0" * 64}
        )


def test_lock_creation_passes_all_attempt_three_lineage_paths() -> None:
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_calibration_protocol_lock"
    ]
    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "attempt3_withdrawal_report_path",
        "attempt3_withdrawal_pointer_path",
        "attempt3_withdrawal_integrity_completion_path",
    } <= keyword_names


def test_attempt_two_completion_binds_the_exact_local_operator_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion.json"
    completion.write_text(
        '{"operator_source_bindings":{'
        '"attempt2_integrity_completion_operator":{"sha256":"' + "b" * 64 + '"},'
        '"attempt2_withdrawal_operator":{"sha256":"' + "a" * 64 + '"}}}\n',
        encoding="utf-8",
    )
    os.chmod(completion, 0o400)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT2_INTEGRITY_COMPLETION", completion)
    bindings = {
        "held_v8_attempt2_withdrawal_operator_source": "a" * 64,
        "held_v8_attempt2_withdrawal_integrity_completion_operator_source": "b" * 64,
    }
    preparer._validate_attempt2_operator_source_lineage(bindings)
    bindings["held_v8_attempt2_withdrawal_operator_source"] = "c" * 64
    with pytest.raises(ValueError, match="executed operator source"):
        preparer._validate_attempt2_operator_source_lineage(bindings)


def test_admission_replay_binds_the_exact_adapter_and_protocol_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = tmp_path / "admission-code-binding.json"
    binding.write_text(
        '{"local_worktree_at_replay":{'
        '"adapter_source_sha256":"'
        + "a" * 64
        + '","protocol_source_sha256":"'
        + "b" * 64
        + '"}}\n',
        encoding="utf-8",
    )
    os.chmod(binding, 0o400)
    monkeypatch.setattr(preparer, "_V8_ADMISSION_REPLAY_CODE_BINDING", binding)
    bindings = {
        "held_v8_builder_adapter_source": "a" * 64,
        "held_v8_protocol_source": "b" * 64,
    }
    preparer._validate_admission_replay_source_lineage(bindings)
    bindings["held_v8_protocol_source"] = "c" * 64
    with pytest.raises(ValueError, match="real pinned-upstream replay"):
        preparer._validate_admission_replay_source_lineage(bindings)
