from __future__ import annotations

from argparse import Namespace
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/held/run_deform360_v81_attempt5.py"


def _load_launcher() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "held_v81_attempt5_launcher",
        LAUNCHER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def launcher() -> ModuleType:
    module = _load_launcher()
    yield module
    module._ACTIVE_PHASE = None


def _write_artifact(
    module: ModuleType, path: Path, value: dict[str, Any]
) -> dict[str, Any]:
    artifact = module._signed_artifact(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o400)
    return artifact


def _record(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "artifact_sha256": artifact["artifact_sha256"],
    }


def _pin_fixture(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    head: str,
) -> None:
    qualification_base = tmp_path / "qualification-base"
    qualification_base.mkdir()
    qualification = qualification_base / f"qualification-{head}"
    qualification.mkdir()
    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.setattr(module, "QUALIFICATION_BASE", qualification_base)
    monkeypatch.setattr(module, "QUALIFICATION_ROOT_PREFIX", "qualification-")
    monkeypatch.setattr(module, "REPLAY_ROOT", replay)

    evidence_path = qualification / "resource-lifecycle-qualification.json"
    attempt_path = qualification / "qualification-attempt.json"
    manifest_path = qualification / "equivalence/repeat-manifest.json"
    analysis_path = qualification / "equivalence/analysis-result.json"
    evidence = _write_artifact(
        module,
        evidence_path,
        {
            "status": "qualified",
            "passed": True,
            "admission": {"decision": "admitted"},
        },
    )
    attempt = _write_artifact(module, attempt_path, {"kind": "attempt"})
    manifest = _write_artifact(module, manifest_path, {"kind": "manifest"})
    analysis = _write_artifact(module, analysis_path, {"kind": "analysis"})
    completion_path = Path(f"{qualification}-integrity-completion.json")
    _write_artifact(
        module,
        completion_path,
        {
            "status": "qualification-integrity-complete",
            "terminal_outcome": "qualified",
            "admission_eligible": True,
            "qualification_root": str(qualification),
            "source_code": {"source_head": head},
            "qualification_evidence": _record(evidence_path, evidence),
            "qualification_attempt": _record(attempt_path, attempt),
            "repeat_manifest": _record(manifest_path, manifest),
            "equivalence_result": _record(analysis_path, analysis),
        },
    )

    source = {"git_head": head}
    report_path = replay / "metadata-only-replay-report.json"
    report = _write_artifact(
        module,
        report_path,
        {
            "artifact_kind": "Deform360HeldV81ExternalAdmissionMetadataOnlyReplay",
            "protocol_id": module.PROTOCOL_ID,
            "execution_attempt": module.EXECUTION_ATTEMPT,
            "development_replay_only": True,
            "formal_outcome_evidence": False,
            "local_source_at_replay": source,
        },
    )
    _write_artifact(
        module,
        replay / "metadata-only-replay-code-binding.json",
        {
            "artifact_kind": "Deform360HeldV81ExternalAdmissionReplayCodeBinding",
            "protocol_id": module.PROTOCOL_ID,
            "execution_attempt": module.EXECUTION_ATTEMPT,
            "formal_outcome_evidence": False,
            "target_query_score_or_outcome_accessed": False,
            "local_worktree_at_replay": source,
            "replay_report": _record(report_path, report),
        },
    )
    qualification.chmod(0o500)
    replay.chmod(0o500)


def _h2_assignments(module: ModuleType) -> dict[str, str]:
    paths = {
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ROOT": (
            "/mnt/corsair/florianpfaff/bpt-resource-lifecycle-qualification-" + "a" * 40
        ),
        "_RESOURCE_LIFECYCLE_QUALIFICATION_EVIDENCE": (
            "/mnt/corsair/florianpfaff/bpt-resource-lifecycle-qualification-"
            + "a" * 40
            + "/resource-lifecycle-qualification.json"
        ),
        "_RESOURCE_LIFECYCLE_QUALIFICATION_COMPLETION": (
            "/mnt/corsair/florianpfaff/bpt-resource-lifecycle-qualification-"
            + "a" * 40
            + "-integrity-completion.json"
        ),
    }
    result: dict[str, str] = {}
    digest_index = 1
    for name in module.H2_PIN_ASSIGNMENT_NAMES:
        if name in paths:
            result[name] = paths[name]
        else:
            result[name] = f"{digest_index:064x}"
            digest_index += 1
    return result


def _git_for_h2(root: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "h2-gate-test",
        "GIT_AUTHOR_EMAIL": "h2-gate@example.invalid",
        "GIT_COMMITTER_NAME": "h2-gate-test",
        "GIT_COMMITTER_EMAIL": "h2-gate@example.invalid",
    }
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    ).stdout.strip()


def _h2_gate_repository(
    module: ModuleType,
    tmp_path: Path,
    *,
    mutation: str | None = None,
) -> tuple[Path, str, str, dict[str, Any]]:
    repository = tmp_path / "h2-repository"
    repository.mkdir()
    _git_for_h2(repository, "init", "--initial-branch=main")
    preparer = repository / module.H2_PREPARER_RELATIVE
    lineage_test = repository / module.H2_LINEAGE_TEST_RELATIVE
    preparer.parent.mkdir(parents=True)
    lineage_test.parent.mkdir(parents=True)
    preparer_lines = ["from pathlib import Path", "", "UNCHANGED_BEFORE = 1"]
    for name in module.H2_PIN_ASSIGNMENT_NAMES:
        annotation = (
            "Path | None" if name in module.H2_PATH_ASSIGNMENT_NAMES else "str | None"
        )
        preparer_lines.append(f"{name}: {annotation} = None")
    preparer_lines.extend(["UNCHANGED_AFTER = 2", ""])
    preparer.write_text("\n".join(preparer_lines), encoding="utf-8")
    lineage_test.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "",
                f"def {module.H1_LINEAGE_TEST_FUNCTION}() -> None:",
                "    assert True",
                "",
                "",
                "UNCHANGED_TEST_SENTINEL = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _git_for_h2(repository, "add", ".")
    _git_for_h2(repository, "commit", "-m", "H1")
    h1 = _git_for_h2(repository, "rev-parse", "HEAD")
    assignments = _h2_assignments(module)
    transition = module._h2_transition_material(repository, h1, assignments)
    payloads = transition["_expected_payloads"]
    preparer_payload = payloads[module.H2_PREPARER_RELATIVE.as_posix()]
    test_payload = payloads[module.H2_LINEAGE_TEST_RELATIVE.as_posix()]
    if mutation == "wrong-pin":
        wrong = dict(assignments)
        digest_name = next(
            name
            for name in module.H2_PIN_ASSIGNMENT_NAMES
            if name not in module.H2_PATH_ASSIGNMENT_NAMES
        )
        wrong[digest_name] = "f" * 64
        h1_preparer = subprocess.run(
            [
                "/usr/bin/git",
                "-C",
                str(repository),
                "show",
                f"{h1}:{module.H2_PREPARER_RELATIVE.as_posix()}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        preparer_payload = module._expected_h2_preparer_source(h1_preparer, wrong)
    if mutation == "extra-preparer-statement":
        preparer_payload += b"\nUNAUTHORIZED_PREPARER_BEHAVIOR = True\n"
    if mutation == "extra-test-statement":
        test_payload += b"\nUNAUTHORIZED_TEST_BEHAVIOR = True\n"
    preparer.write_bytes(preparer_payload)
    lineage_test.write_bytes(test_payload)
    if mutation == "mode-change":
        preparer.chmod(0o755)
    if mutation == "extra-path":
        (repository / "unrelated.py").write_text(
            "UNAUTHORIZED = True\n", encoding="utf-8"
        )
    _git_for_h2(repository, "add", ".")
    _git_for_h2(repository, "commit", "-m", "H2")
    h2 = _git_for_h2(repository, "rev-parse", "HEAD")
    if mutation == "merge-parent":
        _git_for_h2(repository, "checkout", "-b", "side", h1)
        _git_for_h2(repository, "commit", "--allow-empty", "-m", "side")
        _git_for_h2(repository, "checkout", "main")
        _git_for_h2(repository, "merge", "--no-ff", "side", "-m", "merge H2")
        h2 = _git_for_h2(repository, "rev-parse", "HEAD")
    pins = module._signed_artifact(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldV81Attempt5H2Pins",
            "protocol_id": module.PROTOCOL_ID,
            "execution_attempt": module.EXECUTION_ATTEMPT,
            "h1_source_head": h1,
            "preparer_assignments": assignments,
        }
    )
    return repository, h1, h2, pins


def test_parser_exposes_all_phases(launcher: ModuleType) -> None:
    parser = launcher._parser()
    for command in ("qualify", "replay", "emit-h2-pins", "prepare"):
        parsed = parser.parse_args([command, "--code-root", "/tmp/code"])
        assert parsed.command == command
    for command in ("calibrate", "confirm"):
        parsed = parser.parse_args(
            [
                command,
                "--code-root",
                "/tmp/code",
                "--cotracker-repo",
                "/tmp/cotracker",
                "--cotracker-checkpoint",
                "/tmp/cotracker.pth",
            ]
        )
        assert parsed.command == command


def test_h2_acceptance_requires_exact_generated_transition(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, h1, h2, pins = _h2_gate_repository(launcher, tmp_path)
    monkeypatch.setattr(launcher, "_collect_h2_pins", lambda head: pins)

    report = launcher._validate_h2_acceptance(repository, h2)

    assert report["h1_head"] == h1
    assert report["h2_head"] == h2
    assert report["changed_paths"] == sorted(launcher.H2_CHANGED_PATHS)
    assert report["exact_17_assignment_transition"] is True
    assert report["no_other_preparer_source_change"] is True
    assert report["lineage_test_exact_generated_replacement"] is True
    assert report["single_parent_h2"] is True
    assert "patch" not in report
    json.dumps(report)

    transition = launcher._h2_transition_material(
        repository,
        h1,
        pins["preparer_assignments"],
    )
    for payload in transition["_expected_payloads"].values():
        ast.parse(payload.decode("utf-8"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong-pin", "H2 source differs"),
        ("extra-preparer-statement", "H2 source differs"),
        ("extra-test-statement", "H2 source differs"),
        ("extra-path", "H2 changed paths differ"),
        ("mode-change", "regular non-executable blob"),
        ("merge-parent", "single-parent H2"),
    ],
)
def test_h2_acceptance_rejects_non_generated_or_non_adjacent_commits(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    repository, _h1, h2, pins = _h2_gate_repository(
        launcher,
        tmp_path,
        mutation=mutation,
    )
    monkeypatch.setattr(launcher, "_collect_h2_pins", lambda head: pins)

    with pytest.raises(RuntimeError, match=message):
        launcher._validate_h2_acceptance(repository, h2)


def test_qualification_commands_are_frozen(launcher: ModuleType) -> None:
    code = Path("/absolute/code")
    head = "a" * 40
    qualify, seal = launcher._qualification_commands(code, head)
    assert qualify[:5] == [
        str(launcher.PINNED_PYTHON),
        "-I",
        "-B",
        str(code / "scripts/development/qualify_deform360_resource_lifecycle.py"),
        "run",
    ]
    assert qualify[qualify.index("--cuda-device") + 1] == "1"
    assert qualify[qualify.index("--ab-repeat-count") + 1] == "5"
    assert qualify[qualify.index("--soak-fit-count") + 1] == "243"
    assert qualify[qualify.index("--output-dir") + 1] == str(
        launcher._qualification_root(head)
    )
    assert seal == [
        str(launcher.PINNED_PYTHON),
        "-I",
        "-B",
        str(code / "scripts/held/seal_deform360_resource_lifecycle_qualification.py"),
        "--qualification-root",
        str(launcher._qualification_root(head)),
    ]


def test_qualify_preserves_semantic_rc_and_always_seals(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "b" * 40
    code = tmp_path / "code"
    code.mkdir()
    orchestration = tmp_path / "orchestration"
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", orchestration)
    monkeypatch.setattr(launcher, "_require_host", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_verify_repository",
        lambda _path: (code, head),
    )
    monkeypatch.setattr(launcher, "_verify_pinned_git_repository", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_canonical_directory", lambda path, **_k: Path(path))
    monkeypatch.setattr(launcher, "_require_executable", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_require_gpu_idle", lambda _indices: None)
    monkeypatch.setattr(
        launcher,
        "_set_soft_nofile",
        lambda: {"before_soft": 1024, "hard": 4096, "after_soft": 1024},
    )
    monkeypatch.setattr(
        launcher, "_validate_qualification_completion", lambda *a, **k: None
    )
    observed: list[str] = []
    returncodes = iter((3, 0))

    def fake_step(context: Any, name: str, *_args: Any, **_kwargs: Any) -> int:
        observed.append(name)
        return next(returncodes)

    monkeypatch.setattr(launcher, "_run_logged_step", fake_step)
    result = launcher.run_qualify(Namespace(code_root=code))
    assert result.returncode == 3
    assert observed == ["qualification", "qualification-sealer"]
    assert Path(result.payload["phase_summary"]).stat().st_mode & 0o777 == 0o400


def test_qualify_technical_failure_is_not_sealed(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / "code"
    code.mkdir()
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", tmp_path / "orchestration")
    monkeypatch.setattr(launcher, "_require_host", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_verify_repository",
        lambda _path: (code, "1" * 40),
    )
    monkeypatch.setattr(launcher, "_verify_pinned_git_repository", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_canonical_directory", lambda path, **_k: Path(path))
    monkeypatch.setattr(launcher, "_require_executable", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_require_gpu_idle", lambda _indices: None)
    monkeypatch.setattr(launcher, "_set_soft_nofile", lambda: {})
    observed: list[str] = []

    def fake_step(_context: Any, name: str, *_args: Any, **_kwargs: Any) -> int:
        observed.append(name)
        return 2

    monkeypatch.setattr(launcher, "_run_logged_step", fake_step)
    with pytest.raises(RuntimeError, match="failed technically"):
        launcher.run_qualify(Namespace(code_root=code))
    assert observed == ["qualification"]
    launcher._finish_failed_phase(RuntimeError("expected test failure"))


def test_h2_pin_report_is_deterministic_and_complete(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "c" * 40
    _pin_fixture(launcher, tmp_path, monkeypatch, head=head)
    first = launcher._collect_h2_pins(head)
    second = launcher._collect_h2_pins(head)
    assert first == second
    assert len(first["preparer_assignments"]) == 17
    assert (
        first["artifact_sha256"]
        == hashlib.sha256(
            launcher._canonical_bytes(
                {key: value for key, value in first.items() if key != "artifact_sha256"}
            )
        ).hexdigest()
    )
    assert first["preparer_assignments"][
        "_RESOURCE_LIFECYCLE_QUALIFICATION_ROOT"
    ].endswith(head)


def test_logged_step_seals_log_and_exit_code(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", tmp_path / "orchestration")
    context = launcher._new_phase("test", "d" * 40)
    returncode = launcher._run_logged_step(
        context,
        "echo",
        ["/usr/bin/printf", "launcher-log"],
        cwd=tmp_path,
    )
    assert returncode == 0
    assert (context.root / "echo.log").read_bytes() == b"launcher-log"
    assert (context.root / "echo.exit.code").read_text(encoding="ascii") == "0\n"
    assert (context.root / "echo.log").stat().st_mode & 0o777 == 0o400
    assert (context.root / "echo.exit.code").stat().st_mode & 0o777 == 0o400
    launcher._finish_phase(context, returncode=0, status="complete")


def test_concurrent_shards_require_both_successes(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", tmp_path / "orchestration")
    context = launcher._new_phase("shards", "e" * 40)
    with pytest.raises(RuntimeError, match="both formal shards"):
        launcher._run_concurrent_steps(
            context,
            [
                ("shard-0", ["/usr/bin/true"]),
                ("shard-1", ["/usr/bin/false"]),
            ],
            cwd=tmp_path,
            environment=launcher._child_environment(),
        )
    assert (context.root / "shard-0.exit.code").read_text().strip() == "0"
    assert (context.root / "shard-1.exit.code").read_text().strip() == "1"
    launcher._finish_failed_phase(RuntimeError("expected test failure"))


def test_soft_nofile_is_exact_and_preserves_hard(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = [(4096, 8192)]

    monkeypatch.setattr(
        launcher.resource,
        "getrlimit",
        lambda _kind: state[-1],
    )

    def setrlimit(_kind: int, value: tuple[int, int]) -> None:
        state.append(value)

    monkeypatch.setattr(launcher.resource, "setrlimit", setrlimit)
    result = launcher._set_soft_nofile()
    assert state[-1] == (1024, 8192)
    assert result == {"before_soft": 4096, "hard": 8192, "after_soft": 1024}


def test_pinned_repository_allows_only_reserved_external_pycache(
    launcher: ModuleType,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "runtime"
    repository.mkdir()
    git_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "attempt5-test",
        "GIT_AUTHOR_EMAIL": "attempt5@example.invalid",
        "GIT_COMMITTER_NAME": "attempt5-test",
        "GIT_COMMITTER_EMAIL": "attempt5@example.invalid",
    }

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["/usr/bin/git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=git_environment,
        ).stdout.strip()

    git("init", "--initial-branch=main")
    (repository / ".gitignore").write_text(
        "__pycache__/\ncheckpoints/*.pt\n*.bin\n",
        encoding="utf-8",
    )
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "runtime")
    head = git("rev-parse", "HEAD")
    cache = repository / "package/__pycache__/module.cpython-312.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"not loaded under the reserved prefix")
    checkpoint = repository / "checkpoints/model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    launcher._verify_pinned_git_repository(
        repository,
        head,
        label="test runtime",
        allowed_ignored=frozenset({"checkpoints/model.pt"}),
        allow_external_pycache=True,
    )
    bad = repository / "unexpected.bin"
    bad.write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="not an adjacent"):
        launcher._verify_pinned_git_repository(
            repository,
            head,
            label="test runtime",
            allowed_ignored=frozenset({"checkpoints/model.pt"}),
            allow_external_pycache=True,
        )


def test_parent_writable_descriptor_below_held_root_is_rejected(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = tmp_path / "held-v8"
    held.mkdir()
    target = held / "bad.log"
    monkeypatch.setattr(launcher, "HELD_ROOT", held)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        with pytest.raises(RuntimeError, match="writable descriptor"):
            launcher._validate_no_writable_held_descriptors()
    finally:
        os.close(descriptor)


def test_role_freshness_includes_sealed_confirmation_source_boundary(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = tmp_path / "held-v8"
    held.mkdir()
    code = held / ("code-" + "a" * 40)
    code.mkdir()
    monkeypatch.setattr(launcher, "HELD_ROOT", held)
    monkeypatch.setattr(
        launcher, "CONFIRMATION_SOURCE_ROOT", held / "confirmation-source"
    )
    monkeypatch.setattr(
        launcher,
        "CONFIRMATION_SOURCE_RUNTIME_ROOT",
        held / ".confirmation-source-runtime",
    )
    monkeypatch.setattr(
        launcher, "TERMINAL_COMPLETION", tmp_path / "terminal-completion.json"
    )
    monkeypatch.setattr(launcher, "_canonical_directory", lambda *a, **k: held)

    def fake_read(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
        if Path(path).name == "calibration-gate-decision.json":
            return (
                {
                    "protocol_id": launcher.PROTOCOL_ID,
                    "role": "calibration",
                    "decision": "GO",
                },
                b"",
            )
        if Path(path).name == "calibration-outcome-integrity-completion.json":
            return (
                {
                    "status": "role-outcome-integrity-complete",
                    "role": "calibration",
                    "terminal_outcome": "GO",
                },
                b"",
            )
        return {}, b""

    monkeypatch.setattr(launcher, "_read_json_artifact", fake_read)
    checked: list[Path] = []
    monkeypatch.setattr(
        launcher,
        "_require_fresh",
        lambda path, *, label: checked.append(Path(path)),
    )

    launcher._precheck_calibration_freshness(code)

    assert held / "confirmation-source" in checked
    assert held / ".confirmation-source-runtime" in checked

    checked.clear()
    launcher._precheck_confirmation_go(code)

    assert held / "confirmation-source" in checked
    assert held / ".confirmation-source-runtime" in checked


def test_calibration_orders_source_shards_outcome_and_preserves_no_go(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / ("code-" + "2" * 40)
    code.mkdir()
    cotracker = tmp_path / "cotracker"
    cotracker.mkdir()
    checkpoint = tmp_path / "cotracker.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", tmp_path / "orchestration")
    monkeypatch.setattr(launcher, "_require_host", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_require_deployed_repository",
        lambda _path: (code, "2" * 40),
    )
    monkeypatch.setattr(
        launcher,
        "_runtime_arguments",
        lambda _arguments: (cotracker, checkpoint),
    )
    monkeypatch.setattr(launcher, "_require_runtime_resources", lambda *a: None)
    monkeypatch.setattr(launcher, "_precheck_calibration_freshness", lambda _code: None)
    monkeypatch.setattr(
        launcher,
        "_validate_no_writable_held_descriptors",
        lambda: {},
    )
    monkeypatch.setattr(launcher, "_require_gpu_idle", lambda _indices: None)
    monkeypatch.setattr(launcher, "_read_json_artifact", lambda *a, **k: ({}, b""))
    monkeypatch.setattr(launcher, "_set_soft_nofile", lambda: {})
    observed: list[str] = []

    def fake_step(_context: Any, name: str, *_args: Any, **_kwargs: Any) -> int:
        observed.append(name)
        return 3 if name == "calibration-outcome" else 0

    def fake_shards(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        observed.append("both-shards")
        return {"calibration-shard-0": 0, "calibration-shard-1": 0}

    def fake_completion(
        role: str, *, terminal_expected: bool, expected_outcome: str
    ) -> None:
        observed.append(f"completion:{role}:{terminal_expected}:{expected_outcome}")

    monkeypatch.setattr(launcher, "_run_logged_step", fake_step)
    monkeypatch.setattr(launcher, "_run_concurrent_steps", fake_shards)
    monkeypatch.setattr(launcher, "_require_role_completion", fake_completion)
    result = launcher.run_calibrate(
        Namespace(
            code_root=code,
            cotracker_repo=cotracker,
            cotracker_checkpoint=checkpoint,
        )
    )
    assert result.returncode == 3
    assert observed == [
        "replacement-source",
        "both-shards",
        "calibration-outcome",
        "completion:calibration:True:NO-GO",
    ]


def test_confirmation_orders_promotion_source_shards_and_outcome(
    launcher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = tmp_path / ("code-" + "3" * 40)
    code.mkdir()
    cotracker = tmp_path / "cotracker"
    cotracker.mkdir()
    checkpoint = tmp_path / "cotracker.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(launcher, "ORCHESTRATION_ROOT", tmp_path / "orchestration")
    monkeypatch.setattr(launcher, "_require_host", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_require_deployed_repository",
        lambda _path: (code, "3" * 40),
    )
    monkeypatch.setattr(
        launcher,
        "_runtime_arguments",
        lambda _arguments: (cotracker, checkpoint),
    )
    monkeypatch.setattr(launcher, "_require_runtime_resources", lambda *a: None)
    monkeypatch.setattr(launcher, "_precheck_confirmation_go", lambda _code: None)
    monkeypatch.setattr(
        launcher,
        "_validate_no_writable_held_descriptors",
        lambda: {},
    )
    monkeypatch.setattr(launcher, "_require_gpu_idle", lambda _indices: None)
    monkeypatch.setattr(launcher, "_require_fresh", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_set_soft_nofile", lambda: {})
    observed: list[str] = []

    def fake_artifact(path: Path, **_kwargs: Any) -> tuple[dict[str, Any], bytes]:
        if Path(path) == launcher.CONFIRMATION_SOURCE_MANIFEST:
            return (
                {
                    "schema_version": 1,
                    "artifact_kind": "Deform360HeldV8ConfirmationAlignedSourceCohort",
                    "protocol_id": launcher.PROTOCOL_ID,
                    "status": "confirmation-source-cohort-complete",
                    "role": "confirmation",
                },
                b"",
            )
        return {}, b""

    def fake_step(_context: Any, name: str, *_args: Any, **_kwargs: Any) -> int:
        observed.append(name)
        return 4 if name == "confirmation-outcome" else 0

    def fake_shards(
        _context: Any,
        commands: list[tuple[str, list[str]]],
        **_kwargs: Any,
    ) -> dict[str, int]:
        observed.append("both-shards")
        assert all(
            command[-1] == str(launcher.CONFIRMATION_SOURCE_MANIFEST)
            for _name, command in commands
        )
        return {"confirmation-shard-0": 0, "confirmation-shard-1": 0}

    def fake_completion(
        role: str, *, terminal_expected: bool, expected_outcome: str
    ) -> None:
        observed.append(f"completion:{role}:{terminal_expected}:{expected_outcome}")

    monkeypatch.setattr(launcher, "_read_json_artifact", fake_artifact)
    monkeypatch.setattr(launcher, "_run_logged_step", fake_step)
    monkeypatch.setattr(launcher, "_run_concurrent_steps", fake_shards)
    monkeypatch.setattr(launcher, "_require_role_completion", fake_completion)
    result = launcher.run_confirm(
        Namespace(
            code_root=code,
            cotracker_repo=cotracker,
            cotracker_checkpoint=checkpoint,
        )
    )
    assert result.returncode == 4
    assert observed == [
        "confirmation-promotion",
        "confirmation-source",
        "both-shards",
        "confirmation-outcome",
        "completion:confirmation:True:NOT-CONFIRMED",
    ]


def test_outcome_command_and_launcher_binding_are_exact(launcher: ModuleType) -> None:
    code = launcher.HELD_ROOT / ("code-" + "f" * 40)
    command = launcher._outcome_command(
        code,
        "calibration",
        cotracker_repo=Path("/models/cotracker"),
        cotracker_checkpoint=Path("/models/cotracker.pth"),
        replacement_manifest=launcher.HELD_ROOT / "replacement-source/manifest.json",
    )
    assert command[:5] == [
        str(launcher.PINNED_PYTHON),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={launcher.PYCACHE_PREFIX}",
    ]
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--ffmpeg") + 1] == "ffmpeg"

    confirmation_command = launcher._outcome_command(
        code,
        "confirmation",
        cotracker_repo=Path("/models/cotracker"),
        cotracker_checkpoint=Path("/models/cotracker.pth"),
        confirmation_source_manifest=launcher.CONFIRMATION_SOURCE_MANIFEST,
    )
    assert confirmation_command[
        confirmation_command.index("--confirmation-source-manifest") + 1
    ] == str(launcher.CONFIRMATION_SOURCE_MANIFEST)

    from bayesian_phystwin import deform360_held_v8_outcome_driver as driver

    assert driver._DEPLOYMENT_BINDINGS["held_v81_attempt5_launcher_source"] == (
        "scripts/held/run_deform360_v81_attempt5.py"
    )
