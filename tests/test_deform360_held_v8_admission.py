from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bayesian_phystwin import deform360_held_physical_prior as physical
from bayesian_phystwin import deform360_held_v8_builders as builders
from bayesian_phystwin import deform360_held_v8_protocol as protocol


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _runtime_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    upstream = tmp_path / "upstream"
    script = (
        upstream / "scripts" / "remote" / "build_deform360_automatic_episode_twin.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("raise RuntimeError('test script was not meant to run')\n")
    source = upstream / "src"
    source.mkdir()
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    return upstream, script, deform360


def _automatic_twin_arguments(
    upstream: Path,
    *,
    object_id: str,
    episode_id: int,
    phase: str = "calibration",
) -> list[str]:
    return [
        "--repo",
        str(upstream),
        "--object-id",
        object_id,
        "--episode-id",
        str(episode_id),
        "--phase",
        phase,
        "--episode-final-data",
        "/case/prediction.pkl",
        "--episode-graph",
        "/case/graph.npz",
        "--simulator-final-data",
        "/case/simulator.pkl",
        "--state-artifact",
        "/case/state.npz",
        "--summary",
        "/case/twin.json",
        "--prediction-only-input",
        "--canonical-node-count",
        str(physical.CANONICAL_NODE_COUNT),
        "--source-admission-passed",
    ]


def _parent_argv(
    upstream: Path,
    deform360: Path,
    *,
    role: str = "calibration",
) -> list[str]:
    return [
        "deform360-held-v8-physical",
        "--case-name",
        builders.V8_EXTERNAL_CALIBRATION_CASE_NAME,
        "--role",
        role,
        "--lock",
        "/held/calibration-lock.json",
        "--frame-zero-manifest",
        "/held/frame-zero.manifest.json",
        "--upstream-repo",
        str(upstream),
        "--deform360-repo",
        str(deform360),
    ]


def _allow_parent_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        protocol,
        "validate_protocol_lock",
        lambda _path: {
            "stage": "calibration",
            "confirmation_access_authorized": False,
            "calibration_case_whitelist": [builders.V8_EXTERNAL_CALIBRATION_CASE_NAME],
            "immutable_bindings": {
                "replacement_automatic_twin_admission_contract": (
                    builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
                )
            },
        },
    )
    monkeypatch.setattr(
        protocol,
        "validate_frame_zero_bundle_manifest",
        lambda *_args, **_kwargs: {
            "object_id": builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
            "episode_id": builders.V8_EXTERNAL_CALIBRATION_EPISODE_ID,
        },
    )


def test_replacement_admission_has_a_distinct_canonical_identity() -> None:
    contract = protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT
    assert contract["case_name"] == builders.V8_EXTERNAL_CALIBRATION_CASE_NAME
    assert contract["role"] == "calibration"
    assert contract["target_access"] is False
    assert contract["numerical_method_changed"] is False
    assert contract["protocol_id"] != physical.AUTOMATIC_TWIN_PROTOCOL_ID
    assert (
        contract["inherited_numerical_protocol_id"]
        == physical.AUTOMATIC_TWIN_PROTOCOL_ID
    )
    assert (
        contract["inherited_numerical_config_sha256"]
        == physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
    )
    assert _canonical_sha256(contract) == (
        protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
    )


def test_all_legacy_automatic_twin_commands_are_byte_identical(tmp_path: Path) -> None:
    upstream, script, deform360 = _runtime_layout(tmp_path)
    arguments = _automatic_twin_arguments(
        upstream,
        object_id="083-blanket-cloth",
        episode_id=3,
    )
    expected = builders._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    )
    observed = builders._v8_isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    )
    assert observed == expected

    # Even another episode of object 072 is not silently admitted.
    arguments = _automatic_twin_arguments(
        upstream,
        object_id=builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
        episode_id=4,
    )
    assert builders._v8_isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    ) == builders._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    )


def test_exact_command_changes_only_the_process_local_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, script, deform360 = _runtime_layout(tmp_path)
    arguments = _automatic_twin_arguments(
        upstream,
        object_id=builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
        episode_id=builders.V8_EXTERNAL_CALIBRATION_EPISODE_ID,
    )
    _allow_parent_validation(monkeypatch)
    monkeypatch.setattr(sys, "argv", _parent_argv(upstream, deform360))
    baseline = builders._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    )
    observed = builders._v8_isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(upstream / "src", deform360),
        arguments=arguments,
    )
    changed = [
        index
        for index, (before, after) in enumerate(zip(baseline, observed, strict=True))
        if before != after
    ]
    assert changed == [baseline.index(physical._ISOLATED_RUNPY_BOOTSTRAP)]
    assert observed[changed[0]] == builders._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP
    assert "--held-calibration-lock" not in observed
    assert observed[-len(arguments) :] == arguments


@pytest.mark.parametrize(
    ("role", "phase", "message"),
    (
        ("confirmation", "calibration", "another case or role"),
        ("calibration", "source", "identity changed"),
    ),
)
def test_exact_command_rejects_wrong_role_or_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    phase: str,
    message: str,
) -> None:
    upstream, script, deform360 = _runtime_layout(tmp_path)
    arguments = _automatic_twin_arguments(
        upstream,
        object_id=builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
        episode_id=builders.V8_EXTERNAL_CALIBRATION_EPISODE_ID,
        phase=phase,
    )
    _allow_parent_validation(monkeypatch)
    monkeypatch.setattr(sys, "argv", _parent_argv(upstream, deform360, role=role))
    with pytest.raises(RuntimeError, match=message):
        builders._v8_isolated_runpy_command(
            sys.executable,
            script,
            import_roots=(upstream / "src", deform360),
            arguments=arguments,
        )


def _write_fake_panel(source: Path) -> None:
    package = source / "causal4d_public"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "deform360_dense_reusable_panel.py").write_text(
        """
def validate_dense_reusable_panel_config(payload):
    if payload != {"frozen": True}:
        raise ValueError("bad payload")
    return {
        "passed": True,
        "protocol_id": "deform360-dense-reusable-panel-v1",
        "config_sha256": "1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd",
    }

def authorize_dense_panel_episode(*args, **kwargs):
    raise ValueError("object is outside the dense panel")
""".lstrip()
    )


def _run_bootstrap_script(
    tmp_path: Path,
    script_source: str,
) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "upstream" / "src"
    source.mkdir(parents=True)
    _write_fake_panel(source)
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    script = (
        source.parent
        / "scripts"
        / "remote"
        / "build_deform360_automatic_episode_twin.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(script_source)
    command = builders._V7_PHYSICAL_ISOLATED_RUNPY_COMMAND(
        sys.executable,
        script,
        import_roots=(source, deform360),
        arguments=(),
    )
    command = [
        (
            builders._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP
            if value == physical._ISOLATED_RUNPY_BOOTSTRAP
            else value
        )
        for value in command
    ]
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env={"PATH": os.environ["PATH"]},
    )


def test_child_bootstrap_emits_truthful_v8_authorization(tmp_path: Path) -> None:
    completed = _run_bootstrap_script(
        tmp_path,
        """
import json
from causal4d_public.deform360_dense_reusable_panel import authorize_dense_panel_episode

result = authorize_dense_panel_episode(
    {"frozen": True},
    object_id="072-cotton-clohesline",
    episode_id=3,
    phase="calibration",
    source_admission_passed=True,
)
print(json.dumps(result, sort_keys=True))
""".lstrip(),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["protocol_id"] == builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID
    assert result["config_sha256"] == builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
    assert result["target_access"] is False
    assert (
        result["inherited_numerical_protocol_id"] == physical.AUTOMATIC_TWIN_PROTOCOL_ID
    )


def test_child_bootstrap_rejects_cross_case_and_hook_tampering(tmp_path: Path) -> None:
    wrong_case = _run_bootstrap_script(
        tmp_path / "wrong",
        """
from causal4d_public.deform360_dense_reusable_panel import authorize_dense_panel_episode
authorize_dense_panel_episode(
    {"frozen": True}, object_id="083-blanket-cloth", episode_id=3,
    phase="calibration", source_admission_passed=True,
)
""".lstrip(),
    )
    assert wrong_case.returncode != 0
    assert "outside the exact v8 external calibration admission" in wrong_case.stderr

    tampered = _run_bootstrap_script(
        tmp_path / "tampered",
        """
from causal4d_public import deform360_dense_reusable_panel as panel
panel.authorize_dense_panel_episode = lambda *args, **kwargs: {}
""".lstrip(),
    )
    assert tampered.returncode != 0
    assert "hooks changed during execution" in tampered.stderr


def test_fallback_protocol_identity_is_exact_and_always_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "summary.json"

    def validate_summary(*_args: object, **_kwargs: object) -> dict[str, str]:
        value = json.loads(summary.read_text())
        if (
            value["protocol_id"] != physical.AUTOMATIC_TWIN_PROTOCOL_ID
            or value["protocol_config_sha256"]
            != physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        ):
            raise ValueError("automatic twin protocol identity changed")
        return value

    monkeypatch.setattr(
        builders,
        "_V7_PHYSICAL_INADMISSIBLE_TWIN_VALIDATOR",
        validate_summary,
    )
    arguments = ("prediction", "simulator", "graph", "state", summary)
    identity = {
        "case_name": builders.V8_EXTERNAL_CALIBRATION_CASE_NAME,
        "object_id": builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
        "episode_id": builders.V8_EXTERNAL_CALIBRATION_EPISODE_ID,
        "role": "calibration",
    }

    summary.write_text(
        json.dumps(
            {
                "protocol_id": builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID,
                "protocol_config_sha256": (
                    builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
                ),
            }
        )
    )
    result = builders._v8_validate_inadmissible_automatic_twin(*arguments, **identity)
    assert result["protocol_id"] == builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID
    assert physical.AUTOMATIC_TWIN_PROTOCOL_ID == (
        builders._V7_AUTOMATIC_TWIN_PROTOCOL_ID
    )
    assert physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 == (
        builders._V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
    )

    summary.write_text(
        json.dumps(
            {
                "protocol_id": physical.AUTOMATIC_TWIN_PROTOCOL_ID,
                "protocol_config_sha256": (
                    physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="protocol identity changed"):
        builders._v8_validate_inadmissible_automatic_twin(*arguments, **identity)
    assert physical.AUTOMATIC_TWIN_PROTOCOL_ID == (
        builders._V7_AUTOMATIC_TWIN_PROTOCOL_ID
    )
    assert physical.AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 == (
        builders._V7_AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
    )


def test_successful_archive_requires_truthful_exact_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    def build_archive(*_args: object, **kwargs: object) -> dict[str, bool]:
        delegated.append(str(kwargs["case_name"]))
        return {"delegated": True}

    monkeypatch.setattr(
        builders,
        "_V7_PHYSICAL_PREDICTION_ARCHIVE_BUILDER",
        build_archive,
    )
    summary = tmp_path / "summary.json"
    twin = {
        "protocol_id": builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID,
        "protocol_config_sha256": builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256,
        "object_id": builders.V8_EXTERNAL_CALIBRATION_OBJECT_ID,
        "episode_id": builders.V8_EXTERNAL_CALIBRATION_EPISODE_ID,
        "phase": "calibration",
        "passed": True,
    }
    twin["result_sha256"] = physical._upstream_result_sha256(twin)
    summary.write_text(json.dumps(twin))
    positional = (
        "prediction",
        "simulator",
        "graph",
        "readout",
        summary,
        "driven",
        "zero",
        "archive",
        "manifest",
    )
    keywords = {
        "frame_zero_manifest_path": "frame-zero",
        "lock_path": "lock",
        "case_name": builders.V8_EXTERNAL_CALIBRATION_CASE_NAME,
        "role": "calibration",
        "runtime_provenance": {},
        "stage_runtime_seconds": {},
    }
    assert builders._v8_build_physical_prediction_archive(*positional, **keywords) == {
        "delegated": True
    }
    assert delegated == [builders.V8_EXTERNAL_CALIBRATION_CASE_NAME]

    twin["protocol_id"] = physical.AUTOMATIC_TWIN_PROTOCOL_ID
    twin["result_sha256"] = physical._upstream_result_sha256(twin)
    summary.write_text(json.dumps(twin))
    with pytest.raises(ValueError, match="lacks exact v8 admission"):
        builders._v8_build_physical_prediction_archive(*positional, **keywords)
    assert delegated == [builders.V8_EXTERNAL_CALIBRATION_CASE_NAME]

    # Legacy cases remain a direct delegation, including when no summary exists.
    keywords["case_name"] = "083-blanket-cloth-ep0003"
    assert builders._v8_build_physical_prediction_archive(
        *(*positional[:4], tmp_path / "absent.json", *positional[5:]),
        **keywords,
    ) == {"delegated": True}


def test_physical_context_installs_and_restores_only_v8_fallback_adapter() -> None:
    original = physical._validate_inadmissible_automatic_twin
    original_archive = physical.build_physical_prediction_archive
    with builders.explicit_v8_builder_context("physical"):
        assert (
            physical._validate_inadmissible_automatic_twin
            is builders._v8_validate_inadmissible_automatic_twin
        )
        assert (
            physical.build_physical_prediction_archive
            is builders._v8_build_physical_prediction_archive
        )
    assert physical._validate_inadmissible_automatic_twin is original
    assert physical.build_physical_prediction_archive is original_archive
