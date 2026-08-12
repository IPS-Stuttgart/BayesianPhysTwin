from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_gsplat_cuda_runtime.json"
)
BOOTSTRAP = ROOT / "scripts/ci/bootstrap_deform360_v6_gsplat_cuda.sh"
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"


def test_gsplat_cuda_runtime_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert declared == (
        "44da91d95947d07d9d930bd0c707d16da9555bc7b9ea3042fcf0a88444ec3bb4"
    )
    assert payload["correction"]["cuda_toolkit_release"] == "12.1.1"
    assert payload["correction"]["torch_distribution_version"] == "2.5.1+cu121"
    assert payload["correction"]["torch_cuda_version"] == "12.1"
    assert payload["correction"]["gsplat_backend_probe"] == {
        "compiled_backend_required": True,
        "required_attribute": "CameraModelType",
        "torch_extensions_directory_scope": "workflow-run-and-attempt-local",
    }
    assert payload["correction"]["nvidia_redistributable_components"] == [
        {
            "component": "cuda_cccl",
            "relative_path": (
                "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.1.109-archive.tar.xz"
            ),
            "sha256": (
                "b84ef3ec3dc1b4891267be25846f0c3ed7f9fa84154d59eba805402b86991baa"
            ),
            "version": "12.1.109",
        },
        {
            "component": "cuda_cudart",
            "relative_path": (
                "cuda_cudart/linux-x86_64/"
                "cuda_cudart-linux-x86_64-12.1.105-archive.tar.xz"
            ),
            "sha256": (
                "6096ec878c8c443258d39c6e9cf2decef127f8aa8da594fdc5a336d047ab6bd9"
            ),
            "version": "12.1.105",
        },
        {
            "component": "cuda_nvcc",
            "relative_path": (
                "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.1.105-archive.tar.xz"
            ),
            "sha256": (
                "0b85f7eee17788abbd170b0b493c74ce2e9fd5a9604461b99c2c378165e1083b"
            ),
            "version": "12.1.105",
        },
    ]
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["cuda_compiler_runtime_completed"] is True
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "cuda_compiler_runtime_completed"
    )


def test_cuda_bootstrap_is_checksum_pinned_and_fails_before_science() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    relative_amendment = str(AMENDMENT.relative_to(ROOT))
    relative_bootstrap = str(BOOTSTRAP.relative_to(ROOT))

    assert (
        "GSPLAT_CUDA_RUNTIME_REPAIR_ID: "
        "44da91d95947d07d9d930bd0c707d16da9555bc7b9ea3042fcf0a88444ec3bb4" in workflow
    )
    assert (
        "GSPLAT_CUDA_RUNTIME_REPAIR_SHA256: "
        "fb532bf9626c0ba48cb9c7e4aca80488e12f255d18765a31bf4f4324deb385c7" in workflow
    )
    assert relative_amendment in workflow
    assert relative_bootstrap in workflow
    call = f'source "{relative_bootstrap}" "${{runtime}}"'
    assert call in workflow
    assert "from gsplat.cuda._backend import _C as gsplat_cuda_backend" in workflow
    assert 'hasattr(gsplat_cuda_backend, "CameraModelType")' in workflow
    assert workflow.index(call) < (
        workflow.index('-e "./_deform360_physical[processing]"')
    )
    assert (
        "RUNTIME_LOG: ${{ runner.temp }}/"
        "deform360-v6-source-runtime-bootstrap.log" in workflow
    )
    assert 'runtime_target="${logs}/runtime-bootstrap.log"' in workflow

    for token in (
        "cuda_cccl-linux-x86_64-12.1.109-archive.tar.xz",
        "b84ef3ec3dc1b4891267be25846f0c3ed7f9fa84154d59eba805402b86991baa",
        "cuda_cudart-linux-x86_64-12.1.105-archive.tar.xz",
        "6096ec878c8c443258d39c6e9cf2decef127f8aa8da594fdc5a336d047ab6bd9",
        "cuda_nvcc-linux-x86_64-12.1.105-archive.tar.xz",
        "0b85f7eee17788abbd170b0b493c74ce2e9fd5a9604461b99c2c378165e1083b",
        "sha256sum --check",
        "CUDA_HOME",
        "TORCH_EXTENSIONS_DIR",
        "release 12.1",
        "cuda-runtime-probe.cu",
    ):
        assert token in bootstrap

    runtime_call = workflow.index(call)
    science_call = workflow.index(
        "bash scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
    )
    assert runtime_call < science_call
