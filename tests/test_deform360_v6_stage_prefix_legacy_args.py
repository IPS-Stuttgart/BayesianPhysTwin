from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
PHYSICAL_TARGET = "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
INVENTORY_TARGET = "scripts/science/inventory_deform360_calibration_prepared_source.py"
STAGE_SELECTOR_HELPER = (
    "scripts/remote/run_deform360_v6_stage_selector_identity_repair.py"
)
STAGE_SELECTOR_REPAIR = (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "stage_selector_identity_repair.json"
)


def _patch_python(text: str) -> str:
    marker = "\"${BPT_PYTHON}\" - <<'PY'\n"
    assert text.count(marker) == 1
    remainder = text.split(marker, 1)[1]
    body, tail = remainder.split("\nPY\n\nchmod 700", 1)
    assert tail
    return body + "\n"


def _literal(patch: str, name: str) -> str:
    tree = ast.parse(patch)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            continue
        if statement.targets[0].id == name:
            value = ast.literal_eval(statement.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing {name} literal")


def _materialize_shim(tmp_path: Path) -> Path:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    patch = _patch_python(launcher)
    old = _literal(patch, "old")
    base = tmp_path / "base.sh"
    output = tmp_path / "patched.sh"
    base.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"${PYTHON_SHIM}\" <<'SH'\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${REAL_BPT_PYTHON:?REAL_BPT_PYTHON is required}"\n'
        ': "${PREPARED_INVENTORY_IMPLEMENTATION_REVISION:?required}"\n\n'
        f"{old}\n"
        "SH\n"
        'chmod 700 "${PYTHON_SHIM}"\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "BASE_LAUNCHER": str(base),
            "PATCHED_LAUNCHER": str(output),
            "PATCH_ID_VALUE": "fixture-stage-selector-repair",
        }
    )
    subprocess.run(
        [sys.executable, "-c", patch],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    patched = output.read_text(encoding="utf-8")
    marker = "cat > \"${PYTHON_SHIM}\" <<'SH'\n"
    body = patched.split(marker, 1)[1].split("\nSH\nchmod 700", 1)[0]
    shim = tmp_path / "python-shim.sh"
    shim.write_text(body + "\n", encoding="utf-8")
    shim.chmod(0o700)
    return shim


def _fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "capture-python"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(\n"
        "    json.dumps(sys.argv[1:]), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _run_shim(
    tmp_path: Path,
    shim: Path,
    arguments: list[str],
) -> tuple[subprocess.CompletedProcess[str], list[str] | None]:
    capture = tmp_path / "arguments.json"
    environment = os.environ.copy()
    environment.update(
        {
            "CAPTURE_PATH": str(capture),
            "DEFORM360_V6_STAGE_SELECTOR_ACTIVATION_MARKER": str(
                tmp_path / "activation.json"
            ),
            "DEFORM360_V6_STAGE_SELECTOR_HELPER_PATH": STAGE_SELECTOR_HELPER,
            "DEFORM360_V6_STAGE_SELECTOR_REPAIR_PATH": STAGE_SELECTOR_REPAIR,
            "GITHUB_WORKSPACE": str(tmp_path / "exact-worktree"),
            "PREPARED_INVENTORY_IMPLEMENTATION_REVISION": "a" * 40,
            "REAL_BPT_PYTHON": str(_fake_python(tmp_path)),
        }
    )
    completed = subprocess.run(
        [str(shim), *arguments],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    captured = (
        json.loads(capture.read_text(encoding="utf-8")) if capture.exists() else None
    )
    return completed, captured


def _stage_prefix_arguments(worktree: str) -> list[str]:
    return [
        PHYSICAL_TARGET,
        "--execution-repo",
        worktree,
        "--execution-lock",
        "lock.json",
        "--stage",
        "stage-prefix",
        "--repo",
        worktree,
        "--protocol",
        "lock.json",
        "--role",
        "calibration",
        "--source-aligned-root",
        "source",
    ]


def test_stage_prefix_routes_exact_strict_arguments_through_selector_helper(
    tmp_path: Path,
) -> None:
    shim = _materialize_shim(tmp_path)
    worktree = str(tmp_path / "exact-worktree")
    arguments = _stage_prefix_arguments(worktree)

    completed, captured = _run_shim(tmp_path, shim, arguments)

    assert completed.returncode == 0
    assert captured == [
        STAGE_SELECTOR_HELPER,
        "--runtime-repair",
        STAGE_SELECTOR_REPAIR,
        "--activation-marker",
        str(tmp_path / "activation.json"),
        "--execution-repo",
        worktree,
        "--execution-lock",
        "lock.json",
        "--stage",
        "stage-prefix",
        "--protocol",
        "lock.json",
        "--source-aligned-root",
        "source",
    ]


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        (("--role", "target"), "role must remain calibration"),
        (("--repo", "/another/worktree"), "repo differs from the exact worktree"),
    ],
)
def test_stage_prefix_repair_rejects_changed_bindings(
    tmp_path: Path,
    changed: tuple[str, str],
    message: str,
) -> None:
    shim = _materialize_shim(tmp_path)
    worktree = str(tmp_path / "exact-worktree")
    arguments = _stage_prefix_arguments(worktree)
    index = arguments.index(changed[0])
    arguments[index + 1] = changed[1]

    completed, captured = _run_shim(tmp_path, shim, arguments)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert captured is None


def test_stage_prefix_repair_rejects_duplicate_binding(tmp_path: Path) -> None:
    shim = _materialize_shim(tmp_path)
    worktree = str(tmp_path / "exact-worktree")
    arguments = _stage_prefix_arguments(worktree) + ["--role", "calibration"]

    completed, captured = _run_shim(tmp_path, shim, arguments)

    assert completed.returncode == 2
    assert "bindings are not unique" in completed.stderr
    assert captured is None


def test_other_physical_stages_pass_through_unchanged(tmp_path: Path) -> None:
    shim = _materialize_shim(tmp_path)
    arguments = [
        PHYSICAL_TARGET,
        "--execution-repo",
        "repo",
        "--execution-lock",
        "lock.json",
        "--stage",
        "frame-zero",
        "--protocol",
        "lock.json",
    ]

    completed, captured = _run_shim(tmp_path, shim, arguments)

    assert completed.returncode == 0
    assert captured == arguments


def test_inventory_revision_rewrite_is_preserved(tmp_path: Path) -> None:
    shim = _materialize_shim(tmp_path)
    arguments = [
        INVENTORY_TARGET,
        "--implementation-revision",
        "b" * 40,
        "--output",
        "inventory.json",
    ]

    completed, captured = _run_shim(tmp_path, shim, arguments)

    assert completed.returncode == 0
    assert captured == [
        INVENTORY_TARGET,
        "--implementation-revision",
        "a" * 40,
        "--output",
        "inventory.json",
    ]


def test_launcher_preserves_predecessor_blob_and_records_new_repair() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"' in text
    assert 'BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"' in text
    assert 'PATCH_ID="deform360-v6-stage-selector-consumer-identity-v1"' in text
    assert (
        'STAGE_SELECTOR_REPAIR_ID="'
        'aea2506a8c648fcbaad460ae6eb0311801466015268271c5492bac9a6e1d2bae"' in text
    )
    assert STAGE_SELECTOR_HELPER in text
    assert '"runtime_stage_selector_consumer_identity_repair"' in text
    assert 'BASE_REVISION="b0f6b46991a20c54260baf58ddf62fbb6dab7813"' in text
    assert 'SCIENCE_RUNNER_BLOB_SHA="42dd4f3e0d05f18b9ff0a0bdcf90fbd282f0f6f1"' in text
    assert "source.count(old) != 1" in text
    assert "patched.count(new) != 1" in text
