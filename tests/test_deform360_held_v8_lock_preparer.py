from __future__ import annotations

import ast
import hashlib
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


def _make_tree_read_only(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(current) / name
            mode = 0o500 if os.lstat(path).st_mode & 0o111 else 0o400
            os.chmod(path, mode, follow_symlinks=False)
        for name in directories:
            os.chmod(Path(current) / name, 0o500, follow_symlinks=False)
    os.chmod(root, 0o500, follow_symlinks=False)


def _signed_artifact(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["artifact_sha256"] = hashlib.sha256(
        preparer._canonical_bytes(result)
    ).hexdigest()
    return result


def _write_replay_file(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_admission_replay_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str | None = None,
) -> tuple[dict[str, str], SimpleNamespace]:
    root.mkdir()
    boundary = {
        "contact_conditioned_action_result_sha256": None,
        "contact_conditioned_action_used": False,
        "future_object_tracks_present": False,
        "future_robot_action_available": True,
        "object_observation_frames_used": [0],
        "post_initial_object_observation_used": False,
        "prediction_only_input_required": True,
        "simulator_residual_used": False,
        "target_access": False,
    }
    metrics = {"passed": True, "finite": True, "readout_rmse_m": 0.001}
    summary_result_sha256 = "d" * 64
    graph = {"node_count": 4, "edge_count": 3}
    capacity_diagnostic = {"capacity_is_a_maximum": True}
    prediction_input_validation = {"frame_count": 76}
    outputs = {
        "episode_graph.npz": _write_replay_file(
            root / "episode_graph.npz", b"graph"
        ),
        "simulator_final_data.pkl": _write_replay_file(
            root / "simulator_final_data.pkl", b"simulator"
        ),
        "state_artifact.npz": _write_replay_file(
            root / "state_artifact.npz", b"state"
        ),
    }
    summary_output_sha256 = {
        "episode_graph": outputs["episode_graph.npz"]["sha256"],
        "simulator_final_data": outputs["simulator_final_data.pkl"]["sha256"],
        "state_artifact": outputs["state_artifact.npz"]["sha256"],
    }
    if mutation == "summary_output":
        summary_output_sha256["episode_graph"] = "0" * 64
    summary = {
        "passed": True,
        "result_sha256": summary_result_sha256,
        "state_metrics": metrics,
        "information_boundary": boundary,
        "graph": graph,
        "capacity_diagnostic": capacity_diagnostic,
        "prediction_input_validation": prediction_input_validation,
        "output_sha256": summary_output_sha256,
    }
    outputs["twin_summary.json"] = _write_replay_file(
        root / "twin_summary.json",
        (json.dumps(summary, sort_keys=True) + "\n").encode(),
    )
    success_stdout = _write_replay_file(root / "stdout.log", b"success\n")
    success_stderr = _write_replay_file(root / "stderr.log", b"")
    cross_stdout = _write_replay_file(root / "cross-auth/stdout.log", b"")
    cross_stderr_payload = (
        b"different rejection\n"
        if mutation == "cross_log_marker"
        else (
            preparer._V81_CROSS_AUTHORIZATION_STDERR_MARKER.encode("utf-8")
            + b"\n"
        )
    )
    cross_stderr = _write_replay_file(
        root / "cross-auth/stderr.log", cross_stderr_payload
    )
    if mutation == "cross_file":
        _write_replay_file(root / "cross-auth/episode_graph.npz", b"unexpected")
    bootstrap = "exact v8.1 bootstrap"
    source_binding = {
        "git_head": "1" * 40,
        "adapter_source_sha256": "a" * 64,
        "protocol_source_sha256": "b" * 64,
        "replay_operator_source_sha256": "c" * 64,
        "exact_child_bootstrap_sha256": hashlib.sha256(
            bootstrap.encode("utf-8")
        ).hexdigest(),
        "uncommitted_correction_present": False,
        "external_runtime": {
            "python": {},
            "upstream": {},
            "deform360": {},
        },
    }
    if mutation == "bootstrap":
        source_binding["exact_child_bootstrap_sha256"] = "0" * 64
    elif mutation == "uncommitted":
        source_binding["uncommitted_correction_present"] = True
    elif mutation == "source_sha":
        source_binding["adapter_source_sha256"] = "0" * 64
    contract_sha256 = "e" * 64
    report_unsigned: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": preparer._V81_ADMISSION_REPLAY_REPORT_KIND,
        "protocol_id": preparer._V81_PROTOCOL_ID,
        "execution_attempt": preparer._V81_EXECUTION_ATTEMPT,
        "case_name": "072-cotton-clohesline-ep0003",
        "role": "calibration",
        "development_replay_only": True,
        "formal_outcome_evidence": False,
        "source_evidence": {
            "future_object_observation_used": False,
            "source_used_for_numerical_replay": "prediction_only_input_only",
        },
        "admission": {
            "protocol_id": "deform360-held-v8-replacement-admission-v1",
            "contract_sha256": contract_sha256,
            "exact_case_only": True,
            "target_access": False,
        },
        "successful_replay": {
            "exit_code": 0,
            "hook_restoration_guard_completed": True,
            "summary_result_sha256": summary_result_sha256,
            "validator_result_sha256": summary_result_sha256,
            "graph": graph,
            "capacity_diagnostic": capacity_diagnostic,
            "prediction_input_validation": prediction_input_validation,
            "state_metrics": metrics,
            "information_boundary": boundary,
            "outputs": outputs,
            "stdout_log": success_stdout,
            "stderr_log": success_stderr,
        },
        "cross_authorization_rejection": {
            "attempted_case_name": preparer._V81_CROSS_AUTHORIZATION_CASE_NAME,
            "exit_code": 1,
            "rejected": True,
            "numerical_output_count": 0,
            "stderr_marker": preparer._V81_CROSS_AUTHORIZATION_STDERR_MARKER,
            "stderr_marker_present": True,
            "stdout_log": cross_stdout,
            "stderr_log": cross_stderr,
        },
        "information_boundary": {
            "official_target_created": False,
            "official_target_read": False,
            "query_created": False,
            "query_read": False,
            "score_created": False,
            "score_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "confirmation_accessed": False,
        },
        "local_source_at_replay": source_binding,
    }
    if mutation == "report_kind":
        report_unsigned["artifact_kind"] = "Deform360HeldV8ExternalReplay"
    elif mutation == "case_role":
        report_unsigned["role"] = "confirmation"
    elif mutation == "contract":
        report_unsigned["admission"]["contract_sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "formal_evidence":
        report_unsigned["formal_outcome_evidence"] = True
    elif mutation == "top_boundary":
        report_unsigned["information_boundary"]["query_read"] = True  # type: ignore[index]
    elif mutation == "metrics":
        report_unsigned["successful_replay"]["state_metrics"]["finite"] = False  # type: ignore[index]
    elif mutation == "successful_boundary":
        report_unsigned["successful_replay"]["information_boundary"][  # type: ignore[index]
            "target_access"
        ] = True
    elif mutation == "copied_graph":
        report_unsigned["successful_replay"]["graph"] = {  # type: ignore[index]
            "node_count": 5,
            "edge_count": 3,
        }
    elif mutation == "validator_result":
        report_unsigned["successful_replay"][  # type: ignore[index]
            "validator_result_sha256"
        ] = "0" * 64
    elif mutation == "cross_output":
        report_unsigned["cross_authorization_rejection"][  # type: ignore[index]
            "numerical_output_count"
        ] = 1
    elif mutation == "cross_exit":
        report_unsigned["cross_authorization_rejection"]["exit_code"] = 2  # type: ignore[index]
    elif mutation == "cross_marker_field":
        report_unsigned["cross_authorization_rejection"][  # type: ignore[index]
            "stderr_marker"
        ] = "different rejection"
    elif mutation == "cross_marker_flag":
        report_unsigned["cross_authorization_rejection"][  # type: ignore[index]
            "stderr_marker_present"
        ] = False
    elif mutation == "output_record":
        report_unsigned["successful_replay"]["outputs"]["episode_graph.npz"][  # type: ignore[index]
            "sha256"
        ] = "0" * 64

    report = _signed_artifact(report_unsigned)
    report_path = root / "metadata-only-replay-report.json"
    report_payload = (json.dumps(report, sort_keys=True) + "\n").encode()
    report_record = _write_replay_file(report_path, report_payload)
    code_unsigned: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": preparer._V81_ADMISSION_REPLAY_CODE_BINDING_KIND,
        "protocol_id": preparer._V81_PROTOCOL_ID,
        "execution_attempt": preparer._V81_EXECUTION_ATTEMPT,
        "admission_contract_sha256": contract_sha256,
        "formal_outcome_evidence": False,
        "target_query_score_or_outcome_accessed": False,
        "local_worktree_at_replay": source_binding,
        "replay_report": {
            **report_record,
            "artifact_sha256": report["artifact_sha256"],
        },
    }
    if mutation == "code_kind":
        code_unsigned["artifact_kind"] = "Deform360HeldV8ExternalBinding"
    elif mutation == "report_cross_binding":
        code_unsigned["replay_report"]["path"] = str(root / "other.json")  # type: ignore[index]
    elif mutation == "binding_boundary":
        code_unsigned["target_query_score_or_outcome_accessed"] = True

    code_binding = _signed_artifact(code_unsigned)
    code_path = root / "metadata-only-replay-code-binding.json"
    code_payload = (json.dumps(code_binding, sort_keys=True) + "\n").encode()
    _write_replay_file(code_path, code_payload)
    if mutation == "extra_root":
        _write_replay_file(root / "unexpected.txt", b"unexpected")
    elif mutation == "hardlink_output":
        os.link(root / "episode_graph.npz", root.parent / "hardlink-peer")
    elif mutation == "symlink_output":
        target = root.parent / "symlink-target"
        target.write_bytes(b"graph")
        (root / "episode_graph.npz").unlink()
        (root / "episode_graph.npz").symlink_to(target)
    elif mutation == "special_output":
        (root / "episode_graph.npz").unlink()
        os.mkfifo(root / "episode_graph.npz")
    for path in root.rglob("*"):
        os.chmod(path, 0o500 if path.is_dir() else 0o400)
    if mutation == "cross_mode":
        os.chmod(root / "cross-auth", 0o700)
    os.chmod(root, 0o500)

    monkeypatch.setattr(preparer, "_V8_ADMISSION_REPLAY_ROOT", root)
    monkeypatch.setattr(preparer, "_V8_ADMISSION_REPLAY_REPORT", report_path)
    monkeypatch.setattr(preparer, "_V8_ADMISSION_REPLAY_CODE_BINDING", code_path)
    monkeypatch.setattr(
        preparer,
        "_V8_ADMISSION_REPLAY_REPORT_FILE_SHA256",
        hashlib.sha256(report_payload).hexdigest(),
    )
    monkeypatch.setattr(
        preparer,
        "_V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256",
        report["artifact_sha256"],
    )
    monkeypatch.setattr(
        preparer,
        "_V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256",
        hashlib.sha256(code_payload).hexdigest(),
    )
    monkeypatch.setattr(
        preparer,
        "_V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256",
        code_binding["artifact_sha256"],
    )
    monkeypatch.setattr(preparer, "_validate_replay_source_commit", lambda *_: None)
    monkeypatch.setattr(preparer, "_validate_replay_external_runtime", lambda *_: None)
    bindings = {
        "held_v8_builder_adapter_source": "a" * 64,
        "held_v8_protocol_source": "b" * 64,
        "held_v81_external_admission_replay_operator_source": "c" * 64,
    }
    builders = SimpleNamespace(
        V8_EXTERNAL_CALIBRATION_CASE_NAME="072-cotton-clohesline-ep0003",
        V8_EXTERNAL_ADMISSION_PROTOCOL_ID=(
            "deform360-held-v8-replacement-admission-v1"
        ),
        V8_EXTERNAL_ADMISSION_CONTRACT_SHA256=contract_sha256,
        _V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP=bootstrap,
    )
    return bindings, builders


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
        query_artifacts=SimpleNamespace(CENTER_EXCLUSION_CONTRACT_SHA256="5" * 64),
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
    fake_builders = SimpleNamespace()
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
        preparer,
        "_validate_admission_replay_source_lineage",
        lambda _bindings, _builders, _code: None,
    )
    monkeypatch.setattr(preparer, "_validate_pinned_python", lambda: "9" * 64)
    monkeypatch.setattr(
        preparer,
        "_inherited_v7_bindings",
        lambda: {"frame_zero_default_config": "8" * 64},
    )
    monkeypatch.setattr(
        preparer,
        "_import_v8_modules",
        lambda code: (fake_protocol, fake_replacement, fake_builders),
    )
    monkeypatch.setattr(
        preparer,
        "_processing_revision",
        lambda: ("1" * 40, "3" * 40),
    )

    bindings, provenance = preparer.prospective_bindings(root)

    assert bindings["sealed_parent"] == sealed_sha
    assert bindings["pinned_python_executable_target"] == "9" * 64
    assert bindings["frame_zero_default_config"] == "8" * 64
    assert "v7_inherited_immutable_bindings_contract" in bindings
    assert bindings["method_deployed_snapshot_tree"] == provenance["tree_sha256"]
    assert bindings["method_head_text_sha256"] == provenance["head_text_sha256"]
    assert bindings["deform360_processing_tree_text_sha256"] == preparer._sha256_text(
        "3" * 40
    )
    assert "replacement_source_inventory_contract" in bindings
    assert bindings["replacement_automatic_twin_admission_contract"] == "7" * 64
    assert bindings["frame_zero_exact_eight_subset_bounded_audit_contract"] == "6" * 64
    assert bindings["center_exclusion_contract"] == "5" * 64
    assert bindings["v8_attempt3_postseal_noncode_inventory"] == (
        preparer._V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
    )
    assert bindings["v8_attempt3_postseal_noncode_inventory_contract"] == (
        preparer.hashlib.sha256(
            preparer._canonical_bytes(preparer._attempt3_archive_inventory_contract())
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
    assert (
        preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
            "v8_attempt3_postbarrier_withdrawal_report"
        ]
        == "4b7404961fa13b418265f76827dda356fb6ad019db764c6302f49e8149d05de2"
    )
    assert (
        preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
            "v8_attempt3_postbarrier_withdrawal_pointer"
        ]
        == "6ef596a63029d7fa8346141bb52c72d99062e201a12b7c9baf4fca7330baca64"
    )
    assert (
        preparer._EXPECTED_EXTERNAL_ARTIFACT_SHA256[
            "v8_attempt3_withdrawal_integrity_completion"
        ]
        == "9ec2989e3000464a0f72b038e26fe407403e02721e21c19ae4fb9123c6a7cf8c"
    )
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
    assert (
        preparer._sha256_file(
            Path(__file__).parents[1] / relative_operator,
            role="attempt-3 withdrawal operator source",
        )
        == "bc6efe5660c90828be13fb9221472c5e37261e5041509ff61403ea89ef3e9648"
    )


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
    monkeypatch.setattr(
        preparer,
        "_validate_attempt3_excluded_deployed_code",
        lambda _report: None,
    )
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


def test_attempt_three_excluded_deployment_is_exact_clean_git_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "attempt-3-archive"
    archive.mkdir()
    code = _repository(archive)
    deployed_name = "code-test-head"
    code.rename(archive / deployed_name)
    code = archive / deployed_name
    (code / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
    _git(code, "add", ".gitignore")
    _git(code, "commit", "-m", "bind ignored-file policy")
    head = _git(code, "rev-parse", "HEAD")
    records = preparer._parse_git_tree(
        preparer._run_isolated_filemode_git(
            code, ["ls-tree", "-r", "-z", "HEAD"]
        ).stdout
    )
    manifest_sha256 = hashlib.sha256(
        preparer._canonical_bytes(records)
    ).hexdigest()
    binding = {
        "path": deployed_name,
        "git_head": head,
        "head_text_sha256": preparer._sha256_text(head),
        "git_tree_record_count": len(records),
        "git_tree_manifest_sha256": manifest_sha256,
    }
    report = {
        "deployed_code": binding,
        "expected_postseal_inventory": {
            "excluded_deployed_code_directory": deployed_name,
        },
    }
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_ARCHIVE", archive)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_DEPLOYED_CODE_NAME", deployed_name)
    monkeypatch.setattr(preparer, "_V8_ATTEMPT3_DEPLOYED_HEAD", head)
    monkeypatch.setattr(
        preparer,
        "_V8_ATTEMPT3_DEPLOYED_HEAD_TEXT_SHA256",
        preparer._sha256_text(head),
    )
    monkeypatch.setattr(
        preparer,
        "_V8_ATTEMPT3_DEPLOYED_TREE_MANIFEST_SHA256",
        manifest_sha256,
    )
    monkeypatch.setattr(
        preparer, "_V8_ATTEMPT3_DEPLOYED_TREE_RECORD_COUNT", len(records)
    )

    preparer._validate_attempt3_excluded_deployed_code(report)

    (code / "ignored.tmp").write_text("hidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty or has untracked"):
        preparer._validate_attempt3_excluded_deployed_code(report)
    (code / "ignored.tmp").unlink()
    (code / "untracked.tmp").write_text("visible\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty or has untracked"):
        preparer._validate_attempt3_excluded_deployed_code(report)
    (code / "untracked.tmp").unlink()

    os.chmod(code / "method.py", 0o700)
    with pytest.raises(ValueError, match="tracked blob changed"):
        preparer._validate_attempt3_excluded_deployed_code(report)
    os.chmod(code / "method.py", 0o600)
    changed_report = {**report, "deployed_code": {**binding, "git_head": "0" * 40}}
    with pytest.raises(ValueError, match="report binding changed"):
        preparer._validate_attempt3_excluded_deployed_code(changed_report)


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


def test_replay_source_commit_is_clean_ancestor_with_exact_replayed_blobs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--initial-branch=main")
    payloads = {
        "held_v8_builder_adapter_source": b"# replayed adapter\n",
        "held_v8_protocol_source": b"# replayed protocol\n",
        "held_v81_external_admission_replay_operator_source": (
            b"# replayed operator\n"
        ),
    }
    for local_name, payload in payloads.items():
        path = source / preparer._LOCAL_BINDING_FILES[local_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    preparer_path = source / "scripts/held/prepare_deform360_v8_lock.py"
    preparer_path.write_text("# digest pins pending\n", encoding="utf-8")
    (source / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "replayed source")
    replay_head = _git(source, "rev-parse", "HEAD")
    local_bindings = {
        local_name: hashlib.sha256(payload).hexdigest()
        for local_name, payload in payloads.items()
    }
    tested = {
        "git_head": replay_head,
        "adapter_source_sha256": local_bindings[
            "held_v8_builder_adapter_source"
        ],
        "protocol_source_sha256": local_bindings["held_v8_protocol_source"],
        "replay_operator_source_sha256": local_bindings[
            "held_v81_external_admission_replay_operator_source"
        ],
    }
    preparer_path.write_text("# digest pins populated\n", encoding="utf-8")
    _git(source, "add", str(preparer_path.relative_to(source)))
    _git(source, "commit", "-m", "populate replay digest pins")

    preparer._validate_replay_source_commit(tested, local_bindings, source)

    changed_bindings = dict(local_bindings)
    changed_bindings["held_v8_protocol_source"] = "0" * 64
    with pytest.raises(ValueError, match="replayed clean-source commit"):
        preparer._validate_replay_source_commit(tested, changed_bindings, source)
    (source / "ignored.tmp").write_text("hidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current clean source"):
        preparer._validate_replay_source_commit(tested, local_bindings, source)
    (source / "ignored.tmp").unlink()
    (source / "unrelated.txt").write_text("not a digest pin\n", encoding="utf-8")
    _git(source, "add", "unrelated.txt")
    _git(source, "commit", "-m", "unrelated post-replay change")
    with pytest.raises(ValueError, match="confined to preparer digest pins"):
        preparer._validate_replay_source_commit(tested, local_bindings, source)


def test_admission_replay_semantics_and_cross_bindings_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings, builders = _write_admission_replay_fixture(
        tmp_path / "replay", monkeypatch
    )

    preparer._validate_admission_replay_source_lineage(
        bindings, builders, tmp_path
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("report_kind", "report identity"),
        ("code_kind", "code-binding identity"),
        ("case_role", "report identity"),
        ("contract", "current exact-case contract"),
        ("formal_evidence", "evidence boundary"),
        ("binding_boundary", "code-binding identity or boundary"),
        ("report_cross_binding", "report-to-code-binding lineage"),
        ("bootstrap", "bootstrap or committed-source boundary"),
        ("uncommitted", "bootstrap or committed-source boundary"),
        ("source_sha", "real pinned-upstream replay"),
        ("top_boundary", "target/query/score/outcome boundary"),
        ("metrics", "finite passing metrics"),
        ("successful_boundary", "information boundary"),
        ("summary_output", "diagnostics or output hashes"),
        ("copied_graph", "diagnostics or output hashes"),
        ("validator_result", "bound twin summary"),
        ("cross_output", "cross-authorization"),
        ("cross_exit", "cross-authorization"),
        ("cross_marker_field", "cross-authorization"),
        ("cross_marker_flag", "cross-authorization"),
        ("cross_log_marker", "lacks the exact admission-rejection marker"),
        ("cross_file", "cross-authorization"),
        ("output_record", "differs from its replay binding"),
        ("extra_root", "root allowlist"),
        ("hardlink_output", "not a regular file"),
        ("symlink_output", "not a regular file"),
        ("special_output", "not a regular file"),
        ("cross_mode", "entry is not a directory"),
    ],
)
def test_admission_replay_semantic_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    bindings, builders = _write_admission_replay_fixture(
        tmp_path / "replay",
        monkeypatch,
        mutation=mutation,
    )

    with pytest.raises(ValueError, match=message):
        preparer._validate_admission_replay_source_lineage(
            bindings, builders, tmp_path
        )
