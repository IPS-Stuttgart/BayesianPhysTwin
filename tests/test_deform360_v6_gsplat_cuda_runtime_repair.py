from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_gsplat_cuda_runtime.json"
)
HOST_COMPILER_AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_cuda_host_compiler.json"
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


def test_cuda_host_compiler_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(HOST_COMPILER_AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert declared == (
        "e935a990cd380b10f225617d4b439ff609593d63a93e44c27e8fcba5e1dec721"
    )
    assert payload["predecessor_gsplat_cuda_runtime_repair_id"] == (
        "44da91d95947d07d9d930bd0c707d16da9555bc7b9ea3042fcf0a88444ec3bb4"
    )
    assert payload["failed_execution_evidence"] == {
        "artifact_digest": (
            "sha256:e563bba93ef78aa45714ade7114842944ee7dcc07d29442d94cc5857b987bc43"
        ),
        "artifact_id": 9131235897,
        "error_message": (
            "unsupported GNU version! gcc versions later than 12 are not supported!"
        ),
        "exit_code": 1,
        "physical_manifest_count": 0,
        "source_prediction_seal_count": 0,
        "source_revision": "88c2af78fd76252896d9fe00ed8b2400a14ade5e",
        "terminal_stage": "build-isolated-gpu-source-runtime",
        "workflow_run_attempt": 1,
        "workflow_run_id": 31570771026,
    }
    assert payload["correction"] == {
        "compiler_family": "GNU",
        "compiler_major_version": 12,
        "compiler_paths": {
            "cc": "/usr/bin/gcc-12",
            "cxx": "/usr/bin/g++-12",
        },
        "cuda_host_compiler_environment": [
            "CC",
            "CXX",
            "CUDAHOSTCXX",
            "NVCC_CCBIN",
        ],
        "nvcc_compiler_bindir_probe_required": True,
        "unsupported_compiler_override_allowed": False,
    }
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["host_compiler_binding_added"] is True
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "host_compiler_binding_added"
    )


def test_cuda_bootstrap_is_checksum_pinned_and_fails_before_science() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    relative_amendment = str(AMENDMENT.relative_to(ROOT))
    relative_host_amendment = str(HOST_COMPILER_AMENDMENT.relative_to(ROOT))
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
    assert relative_host_amendment in bootstrap
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
        "/usr/bin/gcc-12",
        "/usr/bin/g++-12",
        "CUDAHOSTCXX",
        "NVCC_CCBIN",
        "--compiler-bindir",
        "e935a990cd380b10f225617d4b439ff609593d63a93e44c27e8fcba5e1dec721",
        "8d9663ecd6665fc4c5fcd2b31907200a768ced90e0abb03c006cb04c9bc0a281",
    ):
        assert token in bootstrap
    assert "allow-unsupported-compiler" not in bootstrap

    runtime_call = workflow.index(call)
    science_call = workflow.index(
        "bash scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
    )
    assert runtime_call < science_call
