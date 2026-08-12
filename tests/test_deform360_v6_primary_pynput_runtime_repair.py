from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_primary_pynput_runtime.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)


def test_primary_pynput_runtime_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["failed_execution_evidence"]["error_type"] == ("ModuleNotFoundError")
    assert payload["repair_scope"][
        "primary_pynput_headless_import_dependency_completed"
    ]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "primary_pynput_headless_import_dependency_completed"
    )

    correction = payload["correction"]
    assert correction["backend"] == "dummy"
    assert correction["interactive_keyboard_control_selected"] is False
    assert correction["pynput"] == {
        "distribution_type": "wheel",
        "filename": "pynput-1.8.2-py2.py3-none-any.whl",
        "sha256": ("8cc38cf13a6ab2749cb375678be8a0fd705d7ce49c8001ff5db4007a723bbef1"),
        "version": "1.8.2",
    }
    assert correction["evdev"]["distribution_type"] == "sdist"
    assert correction["evdev"]["sha256"] == (
        "5d3278892ce1f92a74d6bf888cc8525d9f68af85dbe336c95d1c87fb8f423069"
    )


def test_workflow_hash_pins_probes_and_receipts_primary_pynput_runtime() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    correction = payload["correction"]

    assert f"PRIMARY_PYNPUT_RUNTIME_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"PRIMARY_PYNPUT_RUNTIME_REPAIR_SHA256: {amendment_digest}" in workflow
    assert f'PRIMARY_PYNPUT_VERSION: "{correction["pynput"]["version"]}"' in workflow
    assert correction["pynput"]["sha256"] in workflow
    assert correction["evdev"]["sha256"] in workflow
    assert correction["python_xlib"]["sha256"] in workflow
    assert correction["six"]["sha256"] in workflow
    assert correction["frozen_official_environment_sha256"] in workflow
    assert correction["frozen_trainer_sha256"] in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert "--no-build-isolation" in workflow
    assert "--only-binary=pynput,python-xlib,six" in workflow
    assert "--no-binary=evdev" in workflow
    assert "--require-hashes" in workflow
    assert "PYNPUT_BACKEND: dummy" in workflow
    assert "from pynput import keyboard" in workflow
    assert '"pynput.keyboard._dummy" not in sys.modules' in workflow
    assert "from qqtt.engine.trainer_warp import InvPhyTrainerWarp" in workflow
    assert "runtime_primary_pynput_import_dependency" in workflow
    assert '"interactive_keyboard_control_selected": False' in workflow
