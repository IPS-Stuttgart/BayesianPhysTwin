from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import content_id
from scripts.remote import run_deform360_v6_stage_selector_identity_repair as module

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "stage_selector_identity_repair.json"
)
ACTIVE_RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
PHYSICAL_WRAPPER = (
    ROOT / "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
)
STAGE_SCRIPT = ROOT / "scripts/remote/stage_deform360_bias_aware_prediction_prefix.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == module.REPAIR_ID == content_id(payload)
    assert module.load_stage_selector_identity_repair(AMENDMENT)["repair_id"] == (
        module.REPAIR_ID
    )
    assert payload["correction"] == {
        "application": "process-local-loaded-module-constant-only",
        "consumer_field": "GENERIC_SELECTOR_SHA256",
        "consumer_file_sha256": module.STAGE_SCRIPT_SHA256,
        "consumer_path": f"scripts/remote/{module.STAGE_SCRIPT}",
        "corrected_expected_sha256": module.CORRECTED_SELECTOR_SHA256,
        "previous_expected_sha256": module.PREVIOUS_SELECTOR_SHA256,
        "selector_byte_count": module.SELECTOR_BYTE_COUNT,
        "selector_path": module.SELECTOR_RELATIVE_PATH.as_posix(),
        "selector_repository": "IPS-Stuttgart/Causal4D",
        "selector_repository_revision": module.CAUSAL4D_REVISION,
        "selector_semantics": "deform360-object-sam2-generic-selector",
    }
    assert not any(payload["information_boundary"].values())
    scope = payload["repair_scope"]
    assert scope["runtime_expected_identity_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "runtime_expected_identity_only"
    )


def test_repair_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    payload["information_boundary"]["v6_target_payloads_opened"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity repair changed"):
        module.load_stage_selector_identity_repair(changed)


def test_checksum_bound_consumers_remain_exact() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = lock["physical_baseline"]["source_files_sha256"]

    assert _digest(PHYSICAL_WRAPPER) == module.PHYSICAL_WRAPPER_SHA256
    assert _digest(STAGE_SCRIPT) == module.STAGE_SCRIPT_SHA256
    assert (
        expected["scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"]
        == module.PHYSICAL_WRAPPER_SHA256
    )
    assert (
        expected["scripts/remote/stage_deform360_bias_aware_prediction_prefix.py"]
        == module.STAGE_SCRIPT_SHA256
    )


def _runtime_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    remote = repository / "scripts" / "remote"
    remote.mkdir(parents=True)
    (remote / "run_deform360_joint_sparse_physical_source_v5.py").write_text(
        "physical\n",
        encoding="utf-8",
    )
    (remote / module.STAGE_SCRIPT).write_text("stage\n", encoding="utf-8")
    lock = repository / "lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    selector_repository = tmp_path / "causal4d"
    (selector_repository / ".git").mkdir(parents=True)
    selector = selector_repository / module.SELECTOR_RELATIVE_PATH
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"x" * module.SELECTOR_BYTE_COUNT)
    return repository, lock, selector


def test_main_patches_only_the_loaded_selector_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, lock, selector = _runtime_tree(tmp_path)
    marker = tmp_path / "activation.json"
    observed: dict[str, Any] = {}
    stage = SimpleNamespace(
        GENERIC_SELECTOR_SHA256=module.PREVIOUS_SELECTOR_SHA256,
        SAM2_REPOSITORY_REVISION=module.SAM2_REVISION,
    )

    def stage_main() -> int:
        observed["argv"] = list(sys.argv)
        observed["selector_sha256"] = stage.GENERIC_SELECTOR_SHA256
        return 0

    stage.main = stage_main
    stage_arguments = [
        "--protocol",
        str(lock),
        "--generic-selector-source",
        str(selector),
        "--source-aligned-root",
        "source",
    ]
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: (
            SimpleNamespace(
                activation_marker=marker,
                execution_lock=lock,
                execution_repo=repository,
                runtime_repair=AMENDMENT,
                stage="stage-prefix",
            ),
            stage_arguments,
        ),
    )
    monkeypatch.setattr(
        module,
        "validate_joint_sparse_physical_execution_v5",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module, "_load_stage", lambda _path: stage)
    monkeypatch.setattr(
        module,
        "patch_joint_sparse_physical_stage_v5",
        lambda *_args, **_kwargs: observed.setdefault("patched", True),
    )
    monkeypatch.setattr(
        module,
        "activate_joint_sparse_physical_runtime_v5",
        nullcontext,
    )
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda _root, *args: (
            module.CAUSAL4D_REVISION if args == ("rev-parse", "HEAD") else ""
        ),
    )

    def fake_digest(path: Path) -> str:
        if path.name == "run_deform360_joint_sparse_physical_source_v5.py":
            return module.PHYSICAL_WRAPPER_SHA256
        if path.name == module.STAGE_SCRIPT:
            return module.STAGE_SCRIPT_SHA256
        if path == selector:
            return module.CORRECTED_SELECTOR_SHA256
        raise AssertionError(path)

    monkeypatch.setattr(module, "_file_sha256", fake_digest)

    assert module.main() == 0
    assert observed == {
        "argv": [
            str(repository / "scripts" / "remote" / module.STAGE_SCRIPT),
            *stage_arguments,
        ],
        "patched": True,
        "selector_sha256": module.CORRECTED_SELECTOR_SHA256,
    }
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "application": "process-local-loaded-module-constant-only",
        "consumer_file_sha256": module.STAGE_SCRIPT_SHA256,
        "corrected_expected_sha256": module.CORRECTED_SELECTOR_SHA256,
        "previous_expected_sha256": module.PREVIOUS_SELECTOR_SHA256,
        "repair_id": module.REPAIR_ID,
        "selector_file_sha256": module.CORRECTED_SELECTOR_SHA256,
    }


def test_main_rejects_a_changed_locked_stage_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, lock, selector = _runtime_tree(tmp_path)
    stage = SimpleNamespace(
        GENERIC_SELECTOR_SHA256=module.CORRECTED_SELECTOR_SHA256,
        SAM2_REPOSITORY_REVISION=module.SAM2_REVISION,
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: (
            SimpleNamespace(
                activation_marker=tmp_path / "activation.json",
                execution_lock=lock,
                execution_repo=repository,
                runtime_repair=AMENDMENT,
                stage="stage-prefix",
            ),
            [
                "--protocol",
                str(lock),
                "--generic-selector-source",
                str(selector),
            ],
        ),
    )
    monkeypatch.setattr(
        module,
        "validate_joint_sparse_physical_execution_v5",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module, "_load_stage", lambda _path: stage)
    monkeypatch.setattr(
        module,
        "_git_output",
        lambda _root, *args: (
            module.CAUSAL4D_REVISION if args == ("rev-parse", "HEAD") else ""
        ),
    )
    monkeypatch.setattr(
        module,
        "_file_sha256",
        lambda path: (
            module.PHYSICAL_WRAPPER_SHA256
            if path.name == "run_deform360_joint_sparse_physical_source_v5.py"
            else module.STAGE_SCRIPT_SHA256
            if path.name == module.STAGE_SCRIPT
            else module.CORRECTED_SELECTOR_SHA256
        ),
    )

    with pytest.raises(ValueError, match="superseded selector digest"):
        module.main()


def test_active_runner_records_the_process_local_repair() -> None:
    text = ACTIVE_RUNNER.read_text(encoding="utf-8")
    helper = ROOT / "scripts/remote/run_deform360_v6_stage_selector_identity_repair.py"

    assert f'STAGE_SELECTOR_REPAIR_ID="{module.REPAIR_ID}"' in text
    assert "run_deform360_v6_stage_selector_identity_repair.py" in text
    assert f'STAGE_SELECTOR_HELPER_SHA256="{_digest(helper)}"' in text
    assert 'BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"' in text
    assert 'BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"' in text
    assert '"runtime_stage_selector_consumer_identity_repair"' in text
    subprocess_result = subprocess.run(
        ["bash", "-n", str(ACTIVE_RUNNER)],
        cwd=ROOT,
        check=False,
    )
    assert subprocess_result.returncode == 0
