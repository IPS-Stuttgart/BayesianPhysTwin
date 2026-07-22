from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "held"
    / "replay_deform360_v81_external_admission.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("v81_admission_replay", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
