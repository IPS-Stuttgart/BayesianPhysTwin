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
REPAIR_RUNNER = (
    ROOT
    / "scripts/remote/"
    "run_deform360_joint_sparse_physical_source_v5_selector_repair.py"
)
REPAIR = (
    ROOT
    / "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_stage_selector_binding_repair.json"
)
EXPECTED_REPAIR_ID = (
    "001910b84ded7b3f860aa208b87fedf51605fb977af8aab8df3b7e1fa45eeb67"
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


def test_launcher_layers_selector_repair_on_exact_reviewed_base() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'BASE_REVISION="dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"' in text
    assert 'BASE_LAUNCHER_BLOB_SHA="365c5ba0143ba38f1e3d4beac9fdcca1fa63a884"' in text
    assert 'REPAIR_RUNNER_BLOB_SHA="05c897ebcc152397074dd735be861121616d87a9"' in text
    assert 'PATCH_ID="deform360-v6-stage-prefix-selector-binding-v1"' in text
    assert f'REPAIR_ID="{EXPECTED_REPAIR_ID}"' in text
    assert 'BASE_REVISION="b0f6b46991a20c54260baf58ddf62fbb6dab7813"' in text
    assert 'PATCH_ID="deform360-v6-stage-prefix-obsolete-arguments-v1"' in text
    assert "source.count(old) != 1" in text
    assert "patched.count(new) != 1" in text


def test_launcher_patch_replaces_only_the_stage_prefix_tail(tmp_path: Path) -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    patch = _patch_python(launcher)
    old = _literal(patch, "old")
    source = tmp_path / "base.sh"
    output = tmp_path / "patched.sh"
    source.write_text(f"prefix\n{old}\nsuffix\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "BASE_LAUNCHER": str(source),
            "PATCHED_LAUNCHER": str(output),
            "PATCH_ID_VALUE": "fixture-selector-binding",
            "REPAIR_RUNNER_PATH_VALUE": (
                "scripts/remote/"
                "run_deform360_joint_sparse_physical_source_v5_selector_repair.py"
            ),
            "REPAIR_RUNNER_BLOB_SHA_VALUE": "a" * 40,
            "REPAIR_ID_VALUE": "b" * 64,
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
    assert old not in patched
    assert patched.count("rewritten[0]=\"${repair_runner}\"") == 1
    assert patched.count("selector repair runner byte identity changed") == 1
    assert patched.count("export DEFORM360_V6_SELECTOR_REPAIR_ID") == 1
    assert patched.startswith("# runtime compatibility patch: fixture-selector-binding\n")


def test_repair_record_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(REPAIR.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")
    observed = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert declared == observed == EXPECTED_REPAIR_ID
    failure = payload["failed_execution"]
    assert failure["workflow_run_id"] == 31513816637
    assert failure["artifact_id"] == 9110649986
    assert failure["terminal_stage"] == "stage-prefix:026-sock-cloth-ep0007"
    assert failure["physical_manifest_count"] == 0
    assert failure["source_prediction_seal_count"] == 0
    assert failure["error"] == "ValueError: selector changed"

    boundary = payload["information_boundary"]
    assert not any(boundary.values())
    scope = payload["scientific_scope"]
    assert not any(scope.values())
    authorization = payload["execution_authorization"]
    assert authorization["required_physical_manifest_count"] == 10
    assert authorization["required_source_prediction_seal_count"] == 100
    assert authorization["fresh_target_selection_authorized"] is False
    assert authorization["fresh_target_payload_access_authorized"] is False


def test_repair_runner_preserves_original_locked_stage_bytes() -> None:
    source = REPAIR_RUNNER.read_text(encoding="utf-8")

    assert "validate_joint_sparse_physical_execution_v5(" in source
    assert 'STAGE_GIT_BLOB_SHA1 = "188e39f28099f8862c1d0cad66761bcf5d5fb955"' in source
    assert "module.GENERIC_SELECTOR_SHA256 = CORRECTED_SELECTOR_SHA256" in source
    assert "write_text" not in source
    assert "shutil" not in source
    assert "target" not in source.lower()
