from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "stage_selector_identity_repair.json"
)
WRAPPER = "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
PREVIOUS = "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
CORRECTED = "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
REPAIR_ID = "6e31a60bced8ce5d407fe3572eb6b0f5cfc314775cdac1f1c8ff5a1e5d076b11"


def _patch_python() -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    marker = (
        'BASE_LAUNCHER="${base_launcher}" \\\n'
        'PATCHED_LAUNCHER="${patched_launcher}" \\\n'
        'PATCH_ID_VALUE="${PATCH_ID}" \\\n'
        "\"${BPT_PYTHON}\" - <<'PY'\n"
    )
    assert text.count(marker) == 1
    remainder = text.split(marker, 1)[1]
    body, tail = remainder.split("\nPY\n\nchmod 700", 1)
    assert tail
    return body + "\n"


def _bootstrap_python() -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    marker = "cat > \"${selector_bootstrap}\" <<'PY'\n"
    assert text.count(marker) == 1
    remainder = text.split(marker, 1)[1]
    body, tail = remainder.split('\nPY\nchmod 600 "${selector_bootstrap}"', 1)
    assert tail
    return body + "\n"


def _literal(source: str, name: str) -> str:
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = ast.literal_eval(statement.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing {name} literal")


def _write_wrapper(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "from __future__ import annotations\n"
        "import argparse\n"
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "def _load_stage(path: Path, stage: str):\n"
        "    spec = importlib.util.spec_from_file_location('_fixture_stage', path)\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n"
        "def main() -> int:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--stage', required=True)\n"
        "    parser.add_argument('--stage-script', type=Path, required=True)\n"
        "    parser.add_argument('--output', type=Path, required=True)\n"
        "    args = parser.parse_args()\n"
        "    stage = _load_stage(args.stage_script, args.stage)\n"
        "    args.output.write_text(stage.GENERIC_SELECTOR_SHA256)\n"
        "    return 0\n",
        encoding="utf-8",
    )


def _run_bootstrap(
    tmp_path: Path,
    *,
    selector_sha256: str,
) -> subprocess.CompletedProcess[str]:
    wrapper = tmp_path / WRAPPER
    _write_wrapper(wrapper)
    stage = tmp_path / "stage.py"
    stage.write_text(
        f'GENERIC_SELECTOR_SHA256 = "{selector_sha256}"\n',
        encoding="utf-8",
    )
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text(_bootstrap_python(), encoding="utf-8")
    marker = tmp_path / "marker.json"
    output = tmp_path / "observed.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "STAGE_SELECTOR_REPAIR_ID": REPAIR_ID,
            "STAGE_SELECTOR_REPAIR_MARKER": str(marker),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(bootstrap),
            WRAPPER,
            "--stage",
            "stage-prefix",
            "--stage-script",
            str(stage),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    if result.returncode == 0:
        assert output.read_text(encoding="utf-8") == CORRECTED
        assert json.loads(marker.read_text(encoding="utf-8")) == {
            "corrected_sha256": CORRECTED,
            "previous_sha256": PREVIOUS,
            "repair_id": REPAIR_ID,
        }
    else:
        assert not output.exists()
        assert not marker.exists()
    return result


def test_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert declared == REPAIR_ID == hashlib.sha256(canonical).hexdigest()
    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31513816637
    assert failed["artifact_id"] == 9110649986
    assert failed["execution_receipt_id"] == (
        "741968a414984dca9c8c2dab2efbe716a151877d7ac7946830240bd292a47eee"
    )
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0
    assert not any(payload["information_boundary"].values())


def test_outer_patch_delegates_only_stage_prefix_through_bootstrap(
    tmp_path: Path,
) -> None:
    source = _patch_python()
    old = _literal(source, "old")
    new = _literal(source, "new")
    base = tmp_path / "base.sh"
    output = tmp_path / "patched.sh"
    base.write_text(f"#!/usr/bin/env bash\n{old}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "BASE_LAUNCHER": str(base),
            "PATCHED_LAUNCHER": str(output),
            "PATCH_ID_VALUE": "fixture-selector-repair",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    patched = output.read_text(encoding="utf-8")
    assert patched.count(new) == 1
    assert old not in patched
    assert '"${STAGE_SELECTOR_BOOTSTRAP}" "${rewritten[@]}"' in patched


def test_outer_patch_rejects_nonunique_source_block(tmp_path: Path) -> None:
    source = _patch_python()
    old = _literal(source, "old")
    base = tmp_path / "base.sh"
    output = tmp_path / "patched.sh"
    base.write_text(f"{old}\n{old}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "BASE_LAUNCHER": str(base),
            "PATCHED_LAUNCHER": str(output),
            "PATCH_ID_VALUE": "fixture-selector-repair",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "source block changed" in result.stderr
    assert not output.exists()


def test_bootstrap_changes_only_the_expected_stage_constant(
    tmp_path: Path,
) -> None:
    result = _run_bootstrap(tmp_path, selector_sha256=PREVIOUS)

    assert result.returncode == 0


def test_bootstrap_rejects_unexpected_stage_identity(tmp_path: Path) -> None:
    result = _run_bootstrap(tmp_path, selector_sha256="f" * 64)

    assert result.returncode != 0
    assert "frozen stage selector identity changed" in result.stderr


def test_launcher_pins_the_complete_preceding_runtime_lineage() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"' in text
    assert 'BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"' in text
    assert f'REPAIR_ID="{REPAIR_ID}"' in text
    assert f'PREVIOUS_STAGE_SELECTOR_SHA256="{PREVIOUS}"' in text
    assert f'CORRECTED_STAGE_SELECTOR_SHA256="{CORRECTED}"' in text
    assert 'PATCH_ID="deform360-v6-stage-selector-identity-v1"' in text
    assert "stage selector repair is restricted to stage-prefix" in text
    assert "source.count(old) != 1" in text
    assert "patched.count(new) != 1" in text
