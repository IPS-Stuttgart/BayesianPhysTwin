from __future__ import annotations

import json
import tomllib
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "gsplat_ninja_runtime.json"
)
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"


def test_gsplat_ninja_runtime_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert declared == (
        "e64ff97aa573023502ac363a68231732c32e8af18b3257f7bd60b5f607c6ae25"
    )
    assert payload["predecessor_cuda_host_compiler_repair_id"] == (
        "01a5b25972e5b254bfd0ed40fadfd3417532519869d70f404acedf64b98147e0"
    )
    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31576200607
    assert failed["artifact_id"] == 9133317708
    assert failed["artifact_digest"] == (
        "sha256:d4727c8ad965bc005e6ce9412a56779331258eac7fdde837caf15467a7c576b1"
    )
    assert failed["execution_receipt_id"] == (
        "51d39d1bd939bd8b04d003dfefca89c7cb09548561fa401067859944b9388320"
    )
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0
    assert failed["error_message"] == "Ninja is required to load C++ extensions"
    assert payload["correction"] == {
        "dependency_surface": "bayesian-phystwin[vision]",
        "distribution": "ninja",
        "distribution_version": "1.13.0",
        "executable": "ninja",
        "gsplat_version": "1.4.0",
        "installation": "isolated-runtime-pip-resolution",
        "jit_consumer": "torch.utils.cpp_extension",
        "required_before_native_backend_probe": True,
    }
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["ninja_runtime_dependency_added"] is True
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "ninja_runtime_dependency_added"
    )


def test_vision_extra_pins_ninja_before_the_native_backend_probe() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    vision = project["project"]["optional-dependencies"]["vision"]
    assert vision == [
        "h5py>=3.10",
        "ninja==1.13.0",
        "opencv-python-headless>=4.8",
    ]

    workflow = WORKFLOW.read_text(encoding="utf-8")
    install = '"${runtime_python}" -m pip install -e ".[graph,vision]"'
    probe = "from gsplat.cuda._backend import _C as gsplat_cuda_backend"
    assert install in workflow
    assert probe in workflow
    assert workflow.index(install) < workflow.index(probe)
