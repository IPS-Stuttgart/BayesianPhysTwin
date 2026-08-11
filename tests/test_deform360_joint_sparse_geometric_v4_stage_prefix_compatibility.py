from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
PINNED_WRAPPER = (
    ROOT / "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
)
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
PHYSICAL_TARGET = (
    "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
)
COMPATIBILITY_MODE = "frozen-stage-prefix-redundant-context-removal-v1"


def _extract_python_shim() -> str:
    text = ACTIVE_RUNNER.read_text(encoding="utf-8")
    marker = "cat > \"${PYTHON_SHIM}\" <<'SH'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nSH\n", start)
    return text[start:end] + "\n"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _stage_arguments(
    repository: Path,
    *,
    stage: str = "stage-prefix",
    role: str = "calibration",
) -> list[str]:
    return [
        PHYSICAL_TARGET,
        "--execution-repo",
        str(repository),
        "--execution-lock",
        str(LOCK),
        "--stage",
        stage,
        "--repo",
        str(repository),
        "--protocol",
        str(LOCK),
        "--role",
        role,
        "--source-aligned-root",
        "/retained/source",
        "--object-id",
        "026-sock-cloth",
    ]


def _run_shim(
    tmp_path: Path,
    arguments: list[str],
) -> tuple[subprocess.CompletedProcess[str], list[str] | None, Path]:
    shim = tmp_path / "python-shim.sh"
    fake_python = tmp_path / "fake-python"
    capture = tmp_path / "captured.json"
    marker = tmp_path / "compatibility-marker.txt"
    _write_executable(shim, _extract_python_shim())
    _write_executable(
        fake_python,
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(\n"
        "    json.dumps(sys.argv[1:]), encoding='utf-8'\n"
        ")\n",
    )
    environment = {
        **os.environ,
        "CAPTURE": str(capture),
        "PREPARED_INVENTORY_IMPLEMENTATION_REVISION": (
            "e190c94014e6024e324d860618662526af6ea682"
        ),
        "REAL_BPT_PYTHON": str(fake_python),
        "STAGE_PREFIX_COMPATIBILITY_MARKER": str(marker),
        "STAGE_PREFIX_COMPATIBILITY_MODE": COMPATIBILITY_MODE,
    }
    result = subprocess.run(
        [str(shim), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    captured = (
        json.loads(capture.read_text(encoding="utf-8")) if capture.is_file() else None
    )
    return result, captured, marker


def test_checksum_bound_physical_wrapper_remains_exact() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = lock["physical_baseline"]["source_files_sha256"][
        "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
    ]

    assert hashlib.sha256(PINNED_WRAPPER.read_bytes()).hexdigest() == expected
    assert expected == "061fea23aeb83cbaeada9335417d99795de886c8ee6c6ae1013bddfe79bddb37"


def test_stage_prefix_adapter_removes_only_redundant_legacy_context(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    original = _stage_arguments(repository)

    result, captured, marker = _run_shim(tmp_path, original)

    assert result.returncode == 0, result.stderr
    assert captured == [
        PHYSICAL_TARGET,
        "--execution-repo",
        str(repository),
        "--execution-lock",
        str(LOCK),
        "--stage",
        "stage-prefix",
        "--protocol",
        str(LOCK),
        "--source-aligned-root",
        "/retained/source",
        "--object-id",
        "026-sock-cloth",
    ]
    assert original == _stage_arguments(repository)
    assert marker.read_text(encoding="utf-8") == f"{COMPATIBILITY_MODE}\n"


def test_other_physical_source_stages_are_passed_through_unchanged(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    original = _stage_arguments(repository, stage="physical-prior")

    result, captured, marker = _run_shim(tmp_path, original)

    assert result.returncode == 0, result.stderr
    assert captured == original
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong-repository", "legacy stage-prefix repository changed"),
        ("wrong-role", "legacy stage-prefix role changed"),
        ("missing-role", "legacy stage-prefix role binding is not unique"),
        ("duplicate-role", "legacy stage-prefix role binding is not unique"),
        ("duplicate-stage", "physical-source stage binding is not unique"),
    ],
)
def test_stage_prefix_context_drift_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    arguments = _stage_arguments(repository)
    if mutation == "wrong-repository":
        other = tmp_path / "other-repository"
        other.mkdir()
        index = arguments.index("--repo")
        arguments[index + 1] = str(other)
    elif mutation == "wrong-role":
        arguments[arguments.index("--role") + 1] = "target"
    elif mutation == "missing-role":
        index = arguments.index("--role")
        del arguments[index : index + 2]
    elif mutation == "duplicate-role":
        arguments.extend(["--role", "calibration"])
    elif mutation == "duplicate-stage":
        arguments.extend(["--stage", "stage-prefix"])
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    result, captured, marker = _run_shim(tmp_path, arguments)

    assert result.returncode == 2
    assert message in result.stderr
    assert captured is None
    assert not marker.exists()


def test_active_runner_records_adapter_provenance_and_is_valid_shell() -> None:
    text = ACTIVE_RUNNER.read_text(encoding="utf-8")

    assert f'STAGE_PREFIX_COMPATIBILITY_MODE="{COMPATIBILITY_MODE}"' in text
    assert 'physical_target="' + PHYSICAL_TARGET + '"' in text
    assert "stage_prefix_pattern = re.compile(" in text
    assert "len(stage_prefix_pattern.findall(runner)) != 1" in text
    assert '"runtime_stage_prefix_cli_compatibility"' in text
    assert '"activated": activated' in text
    subprocess.run(["bash", "-n", str(ACTIVE_RUNNER)], cwd=ROOT, check=True)
