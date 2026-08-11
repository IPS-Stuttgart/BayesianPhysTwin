from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "frame_zero_cuda_runtime.json"
)
NAMESPACE_AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "dispatch_namespace_repair.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"
PHYSICAL_TARGET = "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"


def _stub(path: Path, route: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' '{route}' > \"${{ROUTE_LOG}}\"\n"
        'if [[ -n "${REPAIR_ID_LOG:-}" ]]; then\n'
        '  printf \'%s\\n\' "${REPAIR_ID-}" > "${REPAIR_ID_LOG}"\n'
        "fi\n"
        'printf \'%s\\n\' "$@" >> "${ROUTE_LOG}"\n',
        encoding="utf-8",
    )
    path.chmod(0o700)


def _dispatch(
    tmp_path: Path,
    arguments: list[str],
    *,
    environment_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    primary = tmp_path / "primary.sh"
    frame_zero = tmp_path / "frame-zero.sh"
    route_log = tmp_path / "route.log"
    marker = tmp_path / "frame-zero-runtime.json"
    _stub(primary, "primary")
    _stub(frame_zero, "frame-zero")
    environment = {
        **os.environ,
        "BPT_PRIMARY_PYTHON": str(primary),
        "BPT_FRAME_ZERO_PYTHON": str(frame_zero),
        "BPT_FRAME_ZERO_RUNTIME_MARKER": str(marker),
        "ROUTE_LOG": str(route_log),
    }
    environment.update(environment_overrides or {})
    result = subprocess.run(
        ["bash", str(DISPATCHER), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    routed = (
        route_log.read_text(encoding="utf-8").splitlines() if route_log.exists() else []
    )
    return result, routed, marker


def test_frame_zero_cuda_runtime_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["frame_zero_cuda_runtime_completed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "frame_zero_cuda_runtime_completed"
    )
    correction = payload["correction"]
    assert correction["dispatcher_stage"] == "frame-zero"
    assert correction["frame_zero_runtime_isolated_from_sam2"]
    assert correction["cuda_toolkit_or_jit_build_required"] is False
    assert correction["frame_zero_python_version"] == "3.10"
    assert correction["frame_zero_torch_version"] == "2.4.0+cu121"
    assert correction["gsplat_version"] == "1.4.0+pt24cu121"


def test_workflow_binds_exact_precompiled_cuda_runtime() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    file_digest = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()

    assert f"FRAME_ZERO_RUNTIME_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"FRAME_ZERO_RUNTIME_REPAIR_SHA256: {file_digest}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert 'FRAME_ZERO_PYTHON_VERSION: "3.10"' in workflow
    assert "FRAME_ZERO_TORCH_VERSION: 2.4.0+cu121" in workflow
    assert "FRAME_ZERO_TORCHVISION_VERSION: 0.19.0+cu121" in workflow
    assert "FRAME_ZERO_GSPLAT_VERSION: 1.4.0+pt24cu121" in workflow
    assert "https://download.pytorch.org/whl/cu121" in workflow
    assert "https://docs.gsplat.studio/whl/pt24cu121" in workflow
    assert 'hasattr(_C, "CameraModelType")' in workflow
    assert "nvcc" not in workflow
    assert "cuda-toolkit" not in workflow
    assert "BPT_PYTHON=${dispatcher}" in workflow
    assert '--system-site-packages "${frame_zero_runtime}"' not in workflow
    frame_zero_venv = (
        '"${FRAME_ZERO_BOOTSTRAP_PYTHON}" -m venv \\\n'
        '              --copies "${frame_zero_runtime}"'
    )
    assert frame_zero_venv in workflow


def test_dispatcher_routes_only_frame_zero_to_precompiled_runtime(
    tmp_path: Path,
) -> None:
    primary, primary_route, primary_marker = _dispatch(
        tmp_path / "primary",
        ["-c", "print('primary')"],
    )
    assert primary.returncode == 0
    assert primary_route == ["primary", "-c", "print('primary')"]
    assert not primary_marker.exists()

    frame, frame_route, frame_marker = _dispatch(
        tmp_path / "frame",
        [PHYSICAL_TARGET, "--stage", "frame-zero", "--protocol", "lock.json"],
    )
    assert frame.returncode == 0
    assert frame_route[0] == "frame-zero"
    assert frame_route[1:] == [
        PHYSICAL_TARGET,
        "--stage",
        "frame-zero",
        "--protocol",
        "lock.json",
    ]
    marker = json.loads(frame_marker.read_text(encoding="utf-8"))
    assert marker == {
        "repair_id": (
            "6524b544bb59d06fee3388906d680b8f1436a0c6a36555cd8f3de0c76074deb8"
        ),
        "stage": "frame-zero",
    }

    physical_prior, route, marker = _dispatch(
        tmp_path / "physical-prior",
        [PHYSICAL_TARGET, "--stage", "physical-prior"],
    )
    assert physical_prior.returncode == 0
    assert route == ["primary", PHYSICAL_TARGET, "--stage", "physical-prior"]
    assert not marker.exists()


def test_dispatcher_rejects_ambiguous_physical_stage(tmp_path: Path) -> None:
    result, route, marker = _dispatch(
        tmp_path,
        [PHYSICAL_TARGET, "--stage", "frame-zero", "--stage", "physical-prior"],
    )

    assert result.returncode == 2
    assert "stage binding is not unique" in result.stderr
    assert route == []
    assert not marker.exists()


def test_dispatcher_preserves_inherited_selector_repair_identity(
    tmp_path: Path,
) -> None:
    repair_id_log = tmp_path / "selector-repair-id.log"
    selector_repair_id = (
        "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
    )

    result, route, marker = _dispatch(
        tmp_path,
        ["-c", "print('selector verifier')"],
        environment_overrides={
            "REPAIR_ID": selector_repair_id,
            "REPAIR_ID_LOG": str(repair_id_log),
        },
    )

    assert result.returncode == 0
    assert route == ["primary", "-c", "print('selector verifier')"]
    assert repair_id_log.read_text(encoding="utf-8").strip() == selector_repair_id
    assert not marker.exists()


def test_dispatch_namespace_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(NAMESPACE_AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["correction"] == {
        "corrected_local_name": "FRAME_ZERO_DISPATCH_REPAIR_ID",
        "inherited_environment_key_preserved": "REPAIR_ID",
        "previous_local_name": "REPAIR_ID",
        "value_changed": False,
    }
    assert payload["repair_scope"]["dispatcher_namespace_corrected"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "dispatcher_namespace_corrected"
    )


def test_workflow_binds_and_probes_dispatch_namespace_repair() -> None:
    payload = json.loads(NAMESPACE_AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(NAMESPACE_AMENDMENT.read_bytes()).hexdigest()
    dispatcher_digest = hashlib.sha256(DISPATCHER.read_bytes()).hexdigest()

    assert f"DISPATCH_NAMESPACE_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"DISPATCH_NAMESPACE_REPAIR_SHA256: {amendment_digest}" in workflow
    assert f"DISPATCHER_SHA256: {dispatcher_digest}" in workflow
    assert str(NAMESPACE_AMENDMENT.relative_to(ROOT)) in workflow
    assert 'REPAIR_ID="selector-repair-environment-probe-v1"' in workflow
    assert "runtime_dispatch_namespace_repair" in workflow

    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    assert 'readonly FRAME_ZERO_DISPATCH_REPAIR_ID="' in dispatcher
    assert 'readonly REPAIR_ID="' not in dispatcher
