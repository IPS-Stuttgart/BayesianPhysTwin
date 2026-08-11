from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
SCIENCE_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)

AUTHORITATIVE_ADMISSION_RUN_ID = "31272512658"
AUTHORITATIVE_IMPLEMENTATION_REVISION = "e190c94014e6024e324d860618662526af6ea682"
PREPARED_INVENTORY_ID = (
    "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
)
PREPARED_INVENTORY_FILE_SHA256 = (
    "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
)
PREVIOUS_RUNNER_REVISION = "07fab9fcd3dae6ab0ec05c56ef565ab16d4466a5"
PREVIOUS_RUNNER_BLOB_SHA = "bf670c99351c9c2ed6dd3cdea9aeb106c1ffb4ca"
PREVIOUS_RUNNER_SHA256 = (
    "75a40281f69c4f99843cc59ca04107e7dda86289a3804d6edb12c88ab8d9e6fb"
)
STAGE_PREFIX_REPAIR_ID = (
    "048733975c44dfc9cf7b1c5bcfa6985327aaba650560305fbbff9c2ec6449c75"
)
STAGE_PREFIX_COMMAND = (
    "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
)


def _git_blob_sha(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _compatibility_shim(tmp_path: Path) -> Path:
    runner = RUNNER.read_text(encoding="utf-8")
    marker = 'cat > "${COMPATIBILITY_PYTHON_SHIM}" <<\'SH\'\n'
    start = runner.index(marker) + len(marker)
    stop = runner.index('\nSH\nchmod 700 "${COMPATIBILITY_PYTHON_SHIM}"', start)
    path = tmp_path / "compatibility-shim.sh"
    path.write_text(runner[start:stop] + "\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "fake-python.sh"
    path.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def test_runtime_reuses_the_frozen_prepared_inventory_identity() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["physical_baseline"]["prepared_source_inventory"] == {
        "file_sha256": PREPARED_INVENTORY_FILE_SHA256,
        "inventory_id": PREPARED_INVENTORY_ID,
    }

    runner = RUNNER.read_text(encoding="utf-8")
    assert (
        f'PREPARED_INVENTORY_IMPLEMENTATION_REVISION="'
        f'{AUTHORITATIVE_IMPLEMENTATION_REVISION}"'
    ) in runner
    assert f'PREPARED_INVENTORY_ID="{PREPARED_INVENTORY_ID}"' in runner
    assert (
        f'PREPARED_INVENTORY_FILE_SHA256="{PREPARED_INVENTORY_FILE_SHA256}"' in runner
    )
    assert (
        f'PREPARED_INVENTORY_ADMISSION_RUN_ID="{AUTHORITATIVE_ADMISSION_RUN_ID}"'
        in runner
    )


def test_runtime_rewrites_only_the_inventory_generator_revision() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    science = SCIENCE_RUNNER.read_text(encoding="utf-8")
    inventory_pattern = re.compile(
        r"scripts/science/inventory_deform360_calibration_prepared_source\.py"
        r"[\s\S]*?--implementation-revision \"\$\{BPT_SOURCE_SHA\}\""
        r"[\s\S]*?--output \"\$\{RUN_ROOT\}/prepared-source-inventory\.json\""
    )

    assert science.count('--implementation-revision "${BPT_SOURCE_SHA}"') == 2
    assert len(inventory_pattern.findall(science)) == 1
    assert (
        'target="scripts/science/inventory_deform360_calibration_prepared_source.py"'
        in runner
    )
    assert "inventory_pattern = re.compile(" in runner
    assert "len(inventory_pattern.findall(runner)) != 1" in runner
    assert 'rewritten+=("$1" "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION}")' in runner
    assert 'if [[ "${replacements}" -ne 1 ]]' in runner
    assert 'exec "${REAL_BPT_PYTHON}" "$@"' in runner
    assert 'BPT_PYTHON="${PYTHON_SHIM}"' in runner
    assert '"runtime_prepared_inventory_identity"' in runner


def test_stage_prefix_repair_delegates_the_exact_reviewed_runner() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert f'PREVIOUS_RUNNER_REVISION="{PREVIOUS_RUNNER_REVISION}"' in runner
    assert f'PREVIOUS_RUNNER_BLOB_SHA="{PREVIOUS_RUNNER_BLOB_SHA}"' in runner
    assert f'PREVIOUS_RUNNER_SHA256="{PREVIOUS_RUNNER_SHA256}"' in runner
    assert 'git show "${PREVIOUS_RUNNER_REVISION}:${PREVIOUS_RUNNER_PATH}"' in runner
    assert 'git hash-object "${DELEGATED_RUNNER}"' in runner
    assert 'sha256sum "${DELEGATED_RUNNER}"' in runner


def test_stage_prefix_repair_is_content_addressed_and_target_closed() -> None:
    repair = {
        "schema": (
            "bayesian-phystwin.deform360-v6-"
            "stage-prefix-argument-compatibility-repair"
        ),
        "schema_version": 1,
        "failed_execution": {
            "workflow_run_id": 31_510_971_371,
            "artifact_id": 9_109_136_220,
            "artifact_sha256": (
                "7e4bd7ba33db2985a2b8e768c1a489487d89b86f736276ed1d25d6cf9b3c73a1"
            ),
            "receipt_id": (
                "ea3856ed0084efd5e13357df877bc1e3bc0a64257c043a35490fda65054660b5"
            ),
            "source_revision": (
                "b0f6b46991a20c54260baf58ddf62fbb6dab7813"
            ),
            "terminal_stage": "stage-prefix:026-sock-cloth-ep0007",
            "exit_code": 2,
        },
        "rewrite": {
            "command": STAGE_PREFIX_COMMAND,
            "stage": "stage-prefix",
            "removed_legacy_arguments": ["--repo", "--role"],
            "required_execution_repo_alias_equality": True,
            "required_role": "calibration",
            "all_other_arguments_preserved": True,
            "other_stages_unchanged": True,
        },
        "information_boundary": {
            "claim_authorized": False,
            "development_suffix_opened": False,
            "replacement_allowed": False,
            "v5_confirmation_payloads_opened": False,
            "v6_fresh_target_selected": False,
            "v6_target_outcomes_used": False,
            "v6_target_payloads_opened": False,
        },
    }
    observed = hashlib.sha256(
        json.dumps(
            repair,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    runner = RUNNER.read_text(encoding="utf-8")

    assert observed == STAGE_PREFIX_REPAIR_ID
    assert f'STAGE_PREFIX_COMPATIBILITY_REPAIR_ID="{observed}"' in runner
    assert '"runtime_stage_prefix_compatibility"' in runner
    assert not any(repair["information_boundary"].values())


def test_stage_prefix_repair_matches_one_frozen_legacy_call() -> None:
    science = SCIENCE_RUNNER.read_text(encoding="utf-8")
    start = science.index('set_stage "stage-prefix:${case_id}"')
    stop = science.index('set_stage "frame-zero:${case_id}"', start)
    block = science[start:stop]

    assert block.count(STAGE_PREFIX_COMMAND) == 1
    assert block.count('--execution-repo "${GITHUB_WORKSPACE}"') == 1
    assert block.count('--repo "${GITHUB_WORKSPACE}"') == 1
    assert block.count("--role calibration") == 1


def test_stage_prefix_repair_removes_only_redundant_arguments(tmp_path: Path) -> None:
    shim = _compatibility_shim(tmp_path)
    fake = _fake_python(tmp_path)
    env = {**os.environ, "REAL_BPT_PYTHON": str(fake)}
    command = [
        str(shim),
        STAGE_PREFIX_COMMAND,
        "--execution-repo",
        "/exact/repository",
        "--execution-lock",
        "lock.json",
        "--stage",
        "stage-prefix",
        "--repo",
        "/exact/repository",
        "--protocol",
        "lock.json",
        "--role",
        "calibration",
        "--output-root",
        "output",
    ]
    completed = subprocess.run(
        command,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        STAGE_PREFIX_COMMAND,
        "--execution-repo",
        "/exact/repository",
        "--execution-lock",
        "lock.json",
        "--stage",
        "stage-prefix",
        "--protocol",
        "lock.json",
        "--output-root",
        "output",
    ]


def test_stage_prefix_repair_preserves_other_stages(tmp_path: Path) -> None:
    shim = _compatibility_shim(tmp_path)
    fake = _fake_python(tmp_path)
    env = {**os.environ, "REAL_BPT_PYTHON": str(fake)}
    arguments = [
        STAGE_PREFIX_COMMAND,
        "--execution-repo",
        "/exact/repository",
        "--stage",
        "physical-prior",
        "--repo",
        "/exact/repository",
        "--protocol",
        "lock.json",
    ]
    completed = subprocess.run(
        [str(shim), *arguments],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == arguments


def test_stage_prefix_repair_rejects_mismatched_repository_alias(
    tmp_path: Path,
) -> None:
    shim = _compatibility_shim(tmp_path)
    fake = _fake_python(tmp_path)
    env = {**os.environ, "REAL_BPT_PYTHON": str(fake)}
    completed = subprocess.run(
        [
            str(shim),
            STAGE_PREFIX_COMMAND,
            "--execution-repo",
            "/exact/repository",
            "--stage",
            "stage-prefix",
            "--repo",
            "/different/repository",
            "--role",
            "calibration",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr.strip() == "stage-prefix repository aliases differ"
