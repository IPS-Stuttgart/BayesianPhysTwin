from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "official_phystwin_py310_runtime.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"


def test_official_phystwin_py310_runtime_repair_is_content_addressed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["failed_execution_evidence"]["status"] == (
        "source-technical-failure-retained"
    )
    assert payload["repair_scope"]["official_phystwin_py310_runtime_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "official_phystwin_py310_runtime_completed"
    )

    correction = payload["correction"]
    assert correction["dispatcher_stages"] == ["frame-zero", "physical-prior"]
    assert correction["python_version"] == "3.10"
    assert correction["torch_version"] == "2.4.0+cu121"
    assert correction["build"] == {
        "cuda_home": "/usr/local/cuda",
        "cuda_toolkit_version": "12.9.86",
        "max_jobs": 2,
        "nvcc_prepend_flags": "-include cstdint",
        "source_copy_outside_frozen_checkout": True,
        "torch_cuda_arch_list": "8.9",
    }
    assert correction["pytorch3d"]["version"] == "0.7.8"
    assert correction["pytorch3d"]["sha256"] == (
        "fa593e799a020a60507d2d92b5d60e4543b78af0372bd37136dfa81535dbfc6c"
    )
    assert correction["vendored_extensions"] == {
        "diff_gaussian_rasterization": {
            "git_tree_oid": "18ffea5595189505e24e2af360aa570de56e2466"
        },
        "simple_knn": {"git_tree_oid": "9be2526499e5f907e47de8229b68d0bf785fb913"},
    }


def test_workflow_locks_builds_probes_and_receipts_official_runtime() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    correction = payload["correction"]
    amendment_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()

    assert f"OFFICIAL_PHYSTWIN_RUNTIME_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"OFFICIAL_PHYSTWIN_RUNTIME_REPAIR_SHA256: {amendment_digest}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert correction["pytorch3d"]["url"] in workflow
    assert correction["pytorch3d"]["sha256"] in workflow
    for extension in correction["vendored_extensions"].values():
        assert extension["git_tree_oid"] in workflow
    for dependency in correction["dependencies"].values():
        assert dependency["sha256"] in workflow
    assert 'OFFICIAL_PHYSTWIN_NVCC_PREPEND_FLAGS: "-include cstdint"' in workflow
    assert 'OFFICIAL_PHYSTWIN_TORCH_CUDA_ARCH_LIST: "8.9"' in workflow
    assert "from simple_knn._C import distCUDA2" in workflow
    assert "GaussianRasterizationSettings" in workflow
    assert "from qqtt.engine.trainer_warp import InvPhyTrainerWarp" in workflow
    assert "runtime_official_phystwin_py310" in workflow
    assert "BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER" in workflow
    assert "BPT_OFFICIAL_PHYSTWIN_BOOTSTRAP_MARKER" in workflow


def test_dispatcher_binds_physical_prior_to_official_runtime() -> None:
    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert payload["repair_id"] in dispatcher
    assert 'readonly PHYSICAL_PRIOR_STAGE="physical-prior"' in dispatcher
    assert 'if [[ "${stage_value}" == "${PHYSICAL_PRIOR_STAGE}" ]]' in dispatcher
    assert 'exec "${BPT_FRAME_ZERO_PYTHON}" "${arguments[@]}"' in dispatcher
    assert "mark_official_phystwin_runtime" in dispatcher
