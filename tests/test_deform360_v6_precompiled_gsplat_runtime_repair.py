from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "precompiled_gsplat_runtime.json"
)
LOCK = ROOT / "requirements/locks/deform360-v6-gsplat-pt24cu121-py310.txt"
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"


def _evidence_steps() -> list[dict[str, Any]]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return list(payload["jobs"]["evidence"]["steps"])


def _step(name: str) -> dict[str, Any]:
    matches = [step for step in _evidence_steps() if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_precompiled_gsplat_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")
    correction = payload["correction"]

    assert declared == content_id(payload)
    assert correction == {
        "backend_probe": (
            "CameraModelType.PINHOLE plus synthetic forward-backward rasterization"
        ),
        "gsplat_base_version": "1.4.0",
        "gsplat_distribution_version": "1.4.0+pt24cu121",
        "gsplat_extension_sha256": (
            "e0b664c9d6f355e611bdfa720103b86b399ded3dcc5ecfaf59eaade992f1359b"
        ),
        "gsplat_wheel_byte_count": 14805290,
        "gsplat_wheel_sha256": (
            "2efb8b8f4ad3275db05707fa6f9cf110482e7fd269c78a4cc7dc5b08cfc957ff"
        ),
        "gsplat_wheel_url": (
            "https://github.com/nerfstudio-project/gsplat/releases/download/"
            "v1.4.0/gsplat-1.4.0%2Bpt24cu121-cp310-cp310-linux_x86_64.whl"
        ),
        "jit_compilation_used": False,
        "nvcc_required": False,
        "python_major_minor": "3.10",
        "runtime_lock_path": (
            "requirements/locks/deform360-v6-gsplat-pt24cu121-py310.txt"
        ),
        "runtime_lock_sha256": (
            "2878efd3b13aed196df63a1dff45e0d3abe24ee2ed2d379ceb32c97ce2a49b61"
        ),
        "system_site_packages": False,
        "torch_cuda_version": "12.1",
        "torch_version": "2.4.0+cu121",
        "torchvision_version": "0.19.0+cu121",
    }
    assert not any(payload["information_boundary"].values())
    scope = payload["repair_scope"]
    assert scope["precompiled_gsplat_runtime_installed"]
    assert scope["dependency_scope_repair_superseded"]
    assert scope["system_site_packages_removed"]
    assert scope["scientific_code_changed"] is False
    assert scope["prob4d_role_changed"] is False


def test_cuda_runtime_lock_and_official_wheel_are_exact() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    correction = payload["correction"]
    lock_bytes = LOCK.read_bytes()
    lock_text = lock_bytes.decode("utf-8")

    assert hashlib.sha256(lock_bytes).hexdigest() == correction["runtime_lock_sha256"]
    assert lock_text.count("torch==2.4.0+cu121") == 1
    assert lock_text.count("torchvision==0.19.0+cu121") == 1
    assert lock_text.count("nvidia-cuda-runtime-cu12==12.1.105") == 1
    assert lock_text.count("nvidia-cudnn-cu12==9.1.0.70") == 1

    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert f"PRECOMPILED_GSPLAT_REPAIR_ID: {payload['repair_id']}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert str(LOCK.relative_to(ROOT)) in workflow
    assert correction["gsplat_wheel_url"] in workflow
    assert correction["gsplat_wheel_sha256"] in workflow
    assert str(correction["gsplat_wheel_byte_count"]) in workflow
    assert correction["gsplat_extension_sha256"] in workflow


def test_gpu_preflight_loads_and_executes_the_compiled_backend() -> None:
    runtime = str(_step("Build isolated GPU source runtime")["run"])
    setup = _step("Set up Python")

    assert setup["with"]["python-version"] == "3.10"
    assert 'python -m venv --copies "${runtime}"' in runtime
    assert "--system-site-packages" not in runtime
    assert (
        '"${runtime_python}" -m pip install -r "${CUDA_RUNTIME_LOCK_PATH}"' in runtime
    )
    assert '"${runtime_python}" -m pip install --no-deps "${gsplat_wheel}"' in runtime
    assert 'test "${pip_check_status}" -eq 0' in runtime
    assert 'test "${pip_check_output}" = "No broken requirements found."' in runtime
    assert "from gsplat.cuda._backend import _C" in runtime
    assert "CameraModelType.PINHOLE" in runtime
    assert "extension_sha256" in runtime
    assert "rasterization(" in runtime
    assert "(render.sum() + alpha.sum()).backward()" in runtime
    assert "gsplat CUDA forward-backward probe failed" in runtime
    assert "nvcc" not in runtime


def test_success_receipt_records_the_runtime_without_changing_science() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    runner = RUNNER.read_text(encoding="utf-8")

    assert f'PRECOMPILED_GSPLAT_REPAIR_ID="{payload["repair_id"]}"' in runner
    assert str(AMENDMENT.relative_to(ROOT)) in runner
    assert 'receipt["runtime_precompiled_gsplat_repair"]' in runner
    assert '"activated": True' in runner
    assert '"jit_compilation_used": False' in runner
    assert '"nvcc_required": False' in runner
    assert '"superseded_by_repair_id"' in runner
