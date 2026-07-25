from __future__ import annotations

import builtins
import errno
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "held"
    / "replay_deform360_v81_external_admission.py"
)


def _module(*, install_source_modules: bool = True):
    spec = importlib.util.spec_from_file_location("v81_admission_replay", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if install_source_modules:
        source_root = str(_SCRIPT.parents[2] / "src")
        sys.path.insert(0, source_root)
        try:
            from bayesian_phystwin import deform360_held_v8_builders as builders
            from bayesian_phystwin import deform360_held_v8_protocol as protocol
        finally:
            assert sys.path[0] == source_root
            del sys.path[0]

        module.builders = builders
        module.protocol = protocol
    return module


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ["PATH"],
        },
    )
    return result.stdout.strip()


def _source_checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkout"
    operator = root / "scripts/held/replay_deform360_v81_external_admission.py"
    protocol = root / "src/bayesian_phystwin/deform360_held_v8_protocol.py"
    adapter = root / "src/bayesian_phystwin/deform360_held_v8_builders.py"
    package = root / "src/bayesian_phystwin/__init__.py"
    for path in (operator, protocol, adapter, package):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n*.py[co]\n", encoding="utf-8")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.name", "Replay Test")
    _git(root, "config", "user.email", "replay@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")
    return root, operator


def test_operator_import_is_stdlib_only_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ):
        if name == "bayesian_phystwin" or name.startswith("bayesian_phystwin."):
            attempted.append(name)
            raise AssertionError("project import occurred before preflight")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _module(install_source_modules=False)

    assert module.builders is None
    assert module.protocol is None
    assert attempted == []


def test_replay_rejects_wrong_interpreter_before_source_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(install_source_modules=False)
    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            isolated=1,
            ignore_environment=1,
            no_user_site=1,
            dont_write_bytecode=1,
        ),
        executable="/wrong/python",
        _base_executable=str(module.PINNED_PYTHON_TARGET),
        prefix=str(module.PINNED_PYTHON_RUNTIME),
        base_prefix=str(module.PINNED_PYTHON_BASE_PREFIX),
    )
    monkeypatch.setattr(module, "sys", fake_sys)

    with pytest.raises(ValueError, match="not executing through the pinned"):
        module._require_isolated_launch()


def test_attempt5_replay_is_historical_under_current_protocol() -> None:
    module = _module()
    assert module.protocol.PROTOCOL_ID == "deform360-held-online-belief-v8.2"
    assert module.protocol.EXECUTION_ATTEMPT == 1
    assert str(module.ROOT).endswith(
        "bpt-held-v8.1-attempt-5-admission-wrapper-scratch-20260722"
    )
    assert module.SOURCE_INPUT.name == "prediction_only_input.pkl"
    assert module.CASE_NAME == "072-cotton-clohesline-ep0003"
    assert module.WRONG_EPISODE_ID == 4


def test_replay_rejects_ignored_module_bytecode_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(install_source_modules=False)
    root, operator = _source_checkout(tmp_path)
    bytecode = root / "src/bayesian_phystwin/__pycache__/deform360_held_v8_protocol.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"stale bytecode must never execute")
    imports: list[object] = []
    monkeypatch.setattr(
        module,
        "_load_verified_source_modules",
        lambda preflight: imports.append(preflight),
    )

    with pytest.raises(ValueError, match="contains ignored files"):
        module._bootstrap_source_modules(operator_path=operator)

    assert imports == []


@pytest.mark.parametrize("rewrite", ["replace", "grafts"])
def test_replay_rejects_git_rewrite_state_before_head_or_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewrite: str,
) -> None:
    module = _module(install_source_modules=False)
    root, operator = _source_checkout(tmp_path)
    if rewrite == "replace":
        head = _git(root, "rev-parse", "HEAD")
        _git(root, "update-ref", f"refs/replace/{head}", head)
        expected = "replacement refs"
    else:
        (root / ".git/info/grafts").write_text("# forbidden\n", encoding="utf-8")
        expected = "grafts file"

    calls: list[tuple[str, ...]] = []
    original_run_git = module._run_git

    def recording_git(checkout: Path, arguments: list[str]) -> str:
        calls.append(tuple(arguments))
        return original_run_git(checkout, arguments)

    imports: list[object] = []
    monkeypatch.setattr(module, "_run_git", recording_git)
    monkeypatch.setattr(
        module,
        "_load_verified_source_modules",
        lambda preflight: imports.append(preflight),
    )

    with pytest.raises(ValueError, match=expected):
        module._bootstrap_source_modules(operator_path=operator)

    assert ("rev-parse", "HEAD") not in calls
    assert ("rev-parse", "HEAD^{tree}") not in calls
    assert imports == []


def test_replay_git_commands_disable_replacement_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(install_source_modules=False)
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._run_git(tmp_path, ["status", "--porcelain=v1"]) == ""
    assert observed["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert observed["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
    assert observed["command"] == [
        "git",
        "-C",
        str(tmp_path),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "status",
        "--porcelain=v1",
    ]


@pytest.mark.parametrize(
    ("flag", "tag"),
    [("--assume-unchanged", "h"), ("--skip-worktree", "S")],
)
def test_replay_rejects_hidden_tracked_drift_before_head_or_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    tag: str,
) -> None:
    module = _module(install_source_modules=False)
    root, operator = _source_checkout(tmp_path)
    relative = "src/bayesian_phystwin/deform360_held_v8_protocol.py"
    _git(root, "update-index", flag, relative)
    (root / relative).write_text("raise RuntimeError('must not execute')\n")
    assert any(
        line.startswith(f"{tag} ")
        for line in _git(root, "ls-files", "-v", "--", relative).splitlines()
    )

    calls: list[tuple[str, ...]] = []
    original_run_git = module._run_git

    def recording_git(checkout: Path, arguments: list[str]) -> str:
        calls.append(tuple(arguments))
        return original_run_git(checkout, arguments)

    imports: list[object] = []
    monkeypatch.setattr(module, "_run_git", recording_git)
    monkeypatch.setattr(
        module,
        "_load_verified_source_modules",
        lambda preflight: imports.append(preflight),
    )

    with pytest.raises(ValueError, match="non-ordinary index flags"):
        module._bootstrap_source_modules(operator_path=operator)

    assert ("rev-parse", "HEAD") not in calls
    assert ("rev-parse", "HEAD^{tree}") not in calls
    assert imports == []


def test_replay_requires_sealed_admissible_qualification_for_exact_h1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    head = "a" * 40
    monkeypatch.setattr(module, "QUALIFICATION_BASE", tmp_path)
    calls: list[dict[str, object]] = []

    def validate(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "resource_lifecycle_qualification_integrity": {
                "source_head": head,
                "terminal_outcome": "qualified",
                "admission_eligible": True,
                "generator_profile": "same-as-analyzer",
                "physical_gpu_index": 1,
                "analyzer_source_sha256": (
                    module.protocol.RESOURCE_LIFECYCLE_ANALYZER_SOURCE_SHA256
                ),
            }
        }

    monkeypatch.setattr(
        module.protocol,
        "validate_resource_lifecycle_qualification_lineage",
        validate,
    )
    lineage = module._qualification_lineage({"git_head": head})
    root = tmp_path / f"{module.QUALIFICATION_ROOT_PREFIX}{head}"
    assert (
        lineage["resource_lifecycle_qualification_integrity"]["admission_eligible"]
        is True
    )
    assert calls == [
        {
            "evidence_path": root / "resource-lifecycle-qualification.json",
            "completion_path": Path(f"{root}-integrity-completion.json"),
            "verify_content_inventory": True,
            "require_admission": True,
        }
    ]

    monkeypatch.setattr(
        module.protocol,
        "validate_resource_lifecycle_qualification_lineage",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("incomplete qualification root")
        ),
    )
    with pytest.raises(ValueError, match="incomplete qualification root"):
        module._qualification_lineage({"git_head": head})


def test_replay_child_environment_matches_the_clean_env_i_boundary() -> None:
    module = _module()

    assert module._environment() == {
        "HOME": "/home/florianpfaff",
        "USER": "florianpfaff",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
    }


def test_child_uses_the_original_remote_home_as_its_explicit_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, b"stdout", b"stderr")

    isolated_subprocess = SimpleNamespace(
        DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE,
        run=fake_run,
    )
    monkeypatch.setattr(module, "subprocess", isolated_subprocess)
    monkeypatch.setattr(module, "_command", lambda _root, episode_id: [str(episode_id)])

    result = module._run_child(tmp_path, episode_id=module.EPISODE_ID)
    assert result.returncode == 0
    assert observed["cwd"] == Path("/home/florianpfaff")
    assert observed["env"] == module._environment()
    assert observed["timeout"] == 1_800


def test_exclusive_writer_removes_a_partial_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "partial.json"
    original_write = module.os.write
    calls = 0

    def interrupted_write(descriptor: int, payload: memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, payload[:1])
        raise OSError("injected write failure")

    isolated_os = SimpleNamespace(
        O_WRONLY=module.os.O_WRONLY,
        O_CREAT=module.os.O_CREAT,
        O_EXCL=module.os.O_EXCL,
        O_NOFOLLOW=getattr(module.os, "O_NOFOLLOW", 0),
        open=module.os.open,
        write=interrupted_write,
        fsync=module.os.fsync,
        close=module.os.close,
    )
    monkeypatch.setattr(module, "os", isolated_os)
    with pytest.raises(OSError, match="injected write failure"):
        module._write_new(output, b"two or more bytes")

    assert not output.exists()


def test_replay_tree_is_an_exact_allowlist(tmp_path: Path) -> None:
    module = _module()
    cross = tmp_path / "cross-auth"
    cross.mkdir()
    for name in module.SUCCESS_OUTPUT_NAMES | {"stdout.log", "stderr.log"}:
        (tmp_path / name).write_bytes(b"sealed later")
    for name in ("stdout.log", "stderr.log"):
        (cross / name).write_bytes(b"log")

    module._require_exact_replay_tree(tmp_path, reports_written=False)
    unexpected = cross / "unrelated-child-output.txt"
    unexpected.write_bytes(b"must be rejected")
    with pytest.raises(ValueError, match="cross-authorization replay entries"):
        module._require_exact_replay_tree(tmp_path, reports_written=False)
    unexpected.unlink()

    for name in (module.REPORT_NAME, module.CODE_BINDING_NAME):
        (tmp_path / name).write_bytes(b"metadata")
    module._require_exact_replay_tree(tmp_path, reports_written=True)
    (tmp_path / "unexpected-root-entry").write_bytes(b"must be rejected")
    with pytest.raises(ValueError, match="replay root entries"):
        module._require_exact_replay_tree(tmp_path, reports_written=True)


def test_replay_publication_is_no_replace_and_postseal_verifiable(
    tmp_path: Path,
) -> None:
    module = _module()
    stage = tmp_path / "stage"
    destination = tmp_path / "published"
    cross = stage / "cross-auth"
    cross.mkdir(parents=True)
    for name in module.SUCCESS_OUTPUT_NAMES | {
        "stdout.log",
        "stderr.log",
        module.REPORT_NAME,
        module.CODE_BINDING_NAME,
    }:
        (stage / name).write_bytes(name.encode("utf-8"))
    for name in ("stdout.log", "stderr.log"):
        (cross / name).write_bytes(name.encode("utf-8"))
    expected = module._replay_content_inventory(stage)
    module._seal_tree(stage, seal_root=False)
    module._rename_noreplace(stage, destination)
    destination.chmod(0o500)

    module._require_exact_replay_tree(destination, reports_written=True)
    module._require_sealed_replay_tree(destination)
    assert module._replay_content_inventory(destination) == expected

    raced = tmp_path / "raced"
    raced.mkdir()
    another = tmp_path / "another"
    another.mkdir()
    with pytest.raises(OSError) as raised:
        module._rename_noreplace(another, raced)
    assert raised.value.errno == errno.EEXIST
    assert another.is_dir()


def test_cross_authorization_requires_the_exact_gate_failure(tmp_path: Path) -> None:
    module = _module()
    marker = module.CROSS_AUTHORIZATION_REJECTION_MARKER.encode("utf-8")
    exact = subprocess.CompletedProcess([], 1, b"", b"prefix " + marker + b"\n")

    module._validate_cross_authorization_rejection(
        exact, tmp_path, ("state_artifact.npz",)
    )
    for wrong in (
        subprocess.CompletedProcess([], 2, b"", marker),
        subprocess.CompletedProcess([], 1, b"", b"unrelated import failure"),
    ):
        with pytest.raises(ValueError, match="exact authorization gate"):
            module._validate_cross_authorization_rejection(
                wrong, tmp_path, ("state_artifact.npz",)
            )

    (tmp_path / "state_artifact.npz").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="exact authorization gate"):
        module._validate_cross_authorization_rejection(
            exact, tmp_path, ("state_artifact.npz",)
        )


def test_success_validator_requires_and_returns_filterable_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    outputs = {
        name: tmp_path / name
        for name in (
            "episode_graph.npz",
            "simulator_final_data.pkl",
            "state_artifact.npz",
            "twin_summary.json",
        )
    }
    for index, path in enumerate(outputs.values()):
        path.write_bytes(f"output-{index}".encode())
    output_sha256 = {
        "episode_graph": hashlib.sha256(
            outputs["episode_graph.npz"].read_bytes()
        ).hexdigest(),
        "simulator_final_data": hashlib.sha256(
            outputs["simulator_final_data.pkl"].read_bytes()
        ).hexdigest(),
        "state_artifact": hashlib.sha256(
            outputs["state_artifact.npz"].read_bytes()
        ).hexdigest(),
    }
    boundary = dict(module.SUCCESS_INFORMATION_BOUNDARY)
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360AutomaticEpisodeTwin",
        "protocol_id": module.builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID,
        "protocol_config_sha256": (
            module.builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
        ),
        "object_id": module.OBJECT_ID,
        "episode_id": module.EPISODE_ID,
        "phase": "calibration",
        "passed": True,
        "result_sha256": "result-sha",
        "input_sha256": {"episode_final_data": module.SOURCE_INPUT_SHA256},
        "output_sha256": output_sha256,
        "state_metrics": {"passed": True, "finite": True},
        "information_boundary": boundary,
        "graph": {"node_count": 1024},
        "capacity_diagnostic": {"passed": True},
        "prediction_input_validation": {"passed": True},
    }
    monkeypatch.setattr(
        module.builders.physical,
        "_load_json",
        lambda _path: summary,
    )
    monkeypatch.setattr(
        module.builders.physical,
        "_upstream_result_sha256",
        lambda _summary: "result-sha",
    )

    validated = module._validate_successful_replay(outputs)
    assert validated["state_metrics"] == {"passed": True, "finite": True}
    assert validated["information_boundary"] == boundary
    assert validated["graph"] == {"node_count": 1024}

    del summary["prediction_input_validation"]
    with pytest.raises(ValueError, match="diagnostics are incomplete"):
        module._validate_successful_replay(outputs)
