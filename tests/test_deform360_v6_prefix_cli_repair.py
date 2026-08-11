from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_prefix_cli_repair.json"
)
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
ARCHIVED_RUNNER = ROOT / (
    "scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v2.sh"
)

REPAIR_ID = "88441357317afa7280513e67fe081dc3fafcd463e5cd3a0e2d32520a50db31ae"
ARCHIVED_RUNNER_BLOB_SHA = "42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"
PHYSICAL_WRAPPER = "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"


def _content_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _python_shim() -> str:
    runner = RUNNER.read_text(encoding="utf-8")
    marker = "cat > \"${PYTHON_SHIM}\" <<'SH'\n"
    assert runner.count(marker) == 1
    remainder = runner.split(marker, 1)[1]
    body, _ = remainder.split("\nSH\n", 1)
    return body + "\n"


def _source_preflight_python() -> str:
    runner = RUNNER.read_text(encoding="utf-8")
    marker = "export SCIENCE_RUNNER\n\"${BPT_PYTHON}\" - <<'PY'\n"
    assert runner.count(marker) == 1
    remainder = runner.split(marker, 1)[1]
    body, _ = remainder.split("\nPY\n", 1)
    return body + "\n"


def _run_shim(
    tmp_path: Path,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    capture_path = tmp_path / "captured.json"
    target = tmp_path / PHYSICAL_WRAPPER
    target.parent.mkdir(parents=True)
    target.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(\n"
        "    json.dumps(sys.argv[1:]), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    shim = tmp_path / "python-shim"
    shim.write_text(_python_shim(), encoding="utf-8")
    shim.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "GITHUB_WORKSPACE": str(tmp_path),
            "PREPARED_INVENTORY_IMPLEMENTATION_REVISION": "e190c940",
            "REAL_BPT_PYTHON": sys.executable,
        }
    )
    return subprocess.run(
        [str(shim), PHYSICAL_WRAPPER, *arguments],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_prefix_cli_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == REPAIR_ID == _content_id(payload)
    assert payload["schema"] == (
        "bayesian-phystwin.deform360-v6-source-runtime-prefix-cli-repair"
    )
    assert payload["schema_version"] == 1
    assert payload["failed_execution_evidence"]["workflow_run_id"] == 31_510_971_371
    assert payload["failed_execution_evidence"]["workflow_run_attempt"] == 1
    assert payload["failed_execution_evidence"]["artifact_name"] == (
        "deform360-v6-source-prediction-evidence-31510971371-1"
    )
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["repair_scope"]["argument_adapter_only"] is True
    assert payload["repair_scope"]["stage_implementation_changed"] is False
    assert payload["information_boundary"]
    assert not any(payload["information_boundary"].values())
    assert not payload["execution_authorization"][
        "fresh_target_payload_access_authorized"
    ]


def test_prefix_cli_repair_leaves_archived_runner_unchanged() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    archived = ARCHIVED_RUNNER.read_text(encoding="utf-8")

    assert _git_blob_sha(ARCHIVED_RUNNER) == ARCHIVED_RUNNER_BLOB_SHA
    assert '--stage stage-prefix \\\n    --repo "${GITHUB_WORKSPACE}" \\' in archived
    assert '--protocol "${LOCK_PATH}" \\\n    --role calibration \\' in archived
    assert f'PREFIX_CLI_REPAIR_ID="{REPAIR_ID}"' in runner
    assert 'receipt["runtime_prefix_cli_repair_id"]' in runner


def test_source_preflight_accepts_bound_prefix_cli_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    environment = {
        "PREFIX_CLI_REPAIR_ID": REPAIR_ID,
        "PREFIX_CLI_REPAIR_PATH": str(REPAIR.relative_to(ROOT)),
        "PREPARED_INVENTORY_ADMISSION_RUN_ID": "31272512658",
        "PREPARED_INVENTORY_FILE_SHA256": (
            "4da96c4f636d195f7aea5d971fbd83bd3b0f35b1c66a77af68007bbd08a69007"
        ),
        "PREPARED_INVENTORY_ID": (
            "6994aa621b38dc8fb21cd38e43363bde3ea12dd644532addeecfc07a30f84e7b"
        ),
        "PREPARED_INVENTORY_IMPLEMENTATION_REVISION": (
            "e190c94014e6024e324d860618662526af6ea682"
        ),
        "SCIENCE_RUNNER": str(ARCHIVED_RUNNER.relative_to(ROOT)),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    exec(compile(_source_preflight_python(), str(RUNNER), "exec"), {})


def test_shim_removes_only_obsolete_stage_prefix_arguments(tmp_path: Path) -> None:
    result = _run_shim(
        tmp_path,
        [
            "--execution-repo",
            str(tmp_path),
            "--execution-lock",
            "lock.json",
            "--stage",
            "stage-prefix",
            "--repo",
            str(tmp_path),
            "--protocol",
            "lock.json",
            "--role",
            "calibration",
            "--object-id",
            "source-object",
        ],
    )

    assert result.returncode == 0, result.stderr
    captured = json.loads((tmp_path / "captured.json").read_text(encoding="utf-8"))
    assert captured == [
        "--execution-repo",
        str(tmp_path),
        "--execution-lock",
        "lock.json",
        "--stage",
        "stage-prefix",
        "--protocol",
        "lock.json",
        "--object-id",
        "source-object",
    ]


def test_shim_preserves_non_prefix_arguments(tmp_path: Path) -> None:
    arguments = [
        "--execution-repo",
        str(tmp_path),
        "--stage",
        "physical-prior",
        "--repo",
        str(tmp_path),
        "--protocol",
        "lock.json",
    ]

    result = _run_shim(tmp_path, arguments)

    assert result.returncode == 0, result.stderr
    captured = json.loads((tmp_path / "captured.json").read_text(encoding="utf-8"))
    assert captured == arguments


def test_shim_rejects_changed_stage_prefix_repository(tmp_path: Path) -> None:
    result = _run_shim(
        tmp_path,
        [
            "--stage",
            "stage-prefix",
            "--repo",
            str(tmp_path / "different"),
            "--role",
            "calibration",
        ],
    )

    assert result.returncode == 2
    assert "does not match the execution repository" in result.stderr
    assert not (tmp_path / "captured.json").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            [
                "--stage",
                "stage-prefix",
                "--repo",
                "{workspace}",
            ],
            "compatibility arguments are not unique",
        ),
        (
            [
                "--stage",
                "stage-prefix",
                "--repo",
                "{workspace}",
                "--role",
                "calibration",
                "--role",
                "calibration",
            ],
            "compatibility arguments are not unique",
        ),
        (
            [
                "--stage",
                "stage-prefix",
                "--stage",
                "stage-prefix",
                "--repo",
                "{workspace}",
                "--role",
                "calibration",
            ],
            "stage binding is not unique",
        ),
    ],
)
def test_shim_rejects_missing_or_duplicate_compatibility_bindings(
    tmp_path: Path,
    arguments: list[str],
    message: str,
) -> None:
    expanded = [
        str(tmp_path) if value == "{workspace}" else value for value in arguments
    ]

    result = _run_shim(tmp_path, expanded)

    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "captured.json").exists()
