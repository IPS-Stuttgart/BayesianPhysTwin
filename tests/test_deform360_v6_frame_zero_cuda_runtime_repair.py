from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_frame_zero_initializer import (
    FrameZeroInitializerConfig,
)

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
FALLBACK_CONFIG_AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_"
    "fallback_config_routing_repair.json"
)
WORKFLOW = ROOT / (
    ".github/workflows/deform360-v6-source-prediction-evidence-dual-runtime.yml"
)
DISPATCHER = ROOT / "scripts/ci/dispatch_deform360_v6_source_python.sh"
PHYSICAL_TARGET = "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
FALLBACK_CONFIG_FLAG = "--persistence-fallback-source-config"
PREVIOUS_FALLBACK_CONFIG = (
    "configs/sota/deform360_reconstruction_failure_persistence_fallback_v1.json"
)
CORRECTED_FALLBACK_CONFIG = (
    "configs/sota/deform360_frame_zero_initializer_source_v1.json"
)


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
) -> tuple[subprocess.CompletedProcess[str], list[str], Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    primary = tmp_path / "primary.sh"
    frame_zero = tmp_path / "frame-zero.sh"
    route_log = tmp_path / "route.log"
    marker = tmp_path / "frame-zero-runtime.json"
    fallback_config_marker = tmp_path / "frame-zero-fallback-config-repair.json"
    official_phystwin_marker = tmp_path / "official-phystwin-runtime.json"
    _stub(primary, "primary")
    _stub(frame_zero, "frame-zero")
    environment = {
        **os.environ,
        "BPT_PRIMARY_PYTHON": str(primary),
        "BPT_FRAME_ZERO_PYTHON": str(frame_zero),
        "BPT_FRAME_ZERO_RUNTIME_MARKER": str(marker),
        "BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER": str(fallback_config_marker),
        "BPT_OFFICIAL_PHYSTWIN_RUNTIME_MARKER": str(official_phystwin_marker),
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
    return result, routed, marker, fallback_config_marker


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
    assert 'OFFICIAL_PHYSTWIN_NVCC_PREPEND_FLAGS: "-include cstdint"' in workflow
    assert "OFFICIAL_PHYSTWIN_CUDA_HOME: /usr/local/cuda" in workflow
    assert "BPT_PYTHON=${dispatcher}" in workflow
    assert '--system-site-packages "${frame_zero_runtime}"' not in workflow
    frame_zero_venv = (
        '"${FRAME_ZERO_BOOTSTRAP_PYTHON}" -m venv \\\n'
        '              --copies "${frame_zero_runtime}"'
    )
    assert frame_zero_venv in workflow


def test_dispatcher_routes_frame_zero_and_physical_prior_to_py310_runtime(
    tmp_path: Path,
) -> None:
    primary, primary_route, primary_marker, primary_config_marker = _dispatch(
        tmp_path / "primary",
        ["-c", "print('primary')"],
    )
    assert primary.returncode == 0
    assert primary_route == ["primary", "-c", "print('primary')"]
    assert not primary_marker.exists()
    assert not primary_config_marker.exists()

    frame, frame_route, frame_marker, frame_config_marker = _dispatch(
        tmp_path / "frame",
        [
            PHYSICAL_TARGET,
            "--stage",
            "frame-zero",
            "--protocol",
            "lock.json",
            FALLBACK_CONFIG_FLAG,
            PREVIOUS_FALLBACK_CONFIG,
        ],
    )
    assert frame.returncode == 0
    assert frame_route[0] == "frame-zero"
    assert frame_route[1:] == [
        PHYSICAL_TARGET,
        "--stage",
        "frame-zero",
        "--protocol",
        "lock.json",
        FALLBACK_CONFIG_FLAG,
        CORRECTED_FALLBACK_CONFIG,
    ]
    marker = json.loads(frame_marker.read_text(encoding="utf-8"))
    assert marker == {
        "repair_id": (
            "6524b544bb59d06fee3388906d680b8f1436a0c6a36555cd8f3de0c76074deb8"
        ),
        "stage": "frame-zero",
    }
    config_marker = json.loads(frame_config_marker.read_text(encoding="utf-8"))
    assert config_marker == {
        "corrected_config": CORRECTED_FALLBACK_CONFIG,
        "previous_config": PREVIOUS_FALLBACK_CONFIG,
        "repair_id": (
            "df4fd52c65acc25c70c4cde650dd021f704e799dceda3323f3aa28af6fd99e0e"
        ),
        "stage": "frame-zero",
    }

    physical_prior, route, marker, config_marker = _dispatch(
        tmp_path / "physical-prior",
        [PHYSICAL_TARGET, "--stage", "physical-prior"],
    )
    assert physical_prior.returncode == 0
    assert route == ["frame-zero", PHYSICAL_TARGET, "--stage", "physical-prior"]
    assert not marker.exists()
    assert not config_marker.exists()
    official_marker = tmp_path / "physical-prior/official-phystwin-runtime.json"
    assert json.loads(official_marker.read_text(encoding="utf-8")) == {
        "repair_id": (
            "72db4752194340a4e8122332ec7483e7d397240c714b3aeec771b1e043369deb"
        ),
        "stage": "physical-prior",
    }


def test_dispatcher_rejects_ambiguous_physical_stage(tmp_path: Path) -> None:
    result, route, marker, config_marker = _dispatch(
        tmp_path,
        [PHYSICAL_TARGET, "--stage", "frame-zero", "--stage", "physical-prior"],
    )

    assert result.returncode == 2
    assert "stage binding is not unique" in result.stderr
    assert route == []
    assert not marker.exists()
    assert not config_marker.exists()


def test_dispatcher_rejects_changed_official_runtime_marker(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    official_marker = tmp_path / "official-phystwin-runtime.json"
    official_marker.write_text("{}\n", encoding="utf-8")

    result, route, marker, config_marker = _dispatch(
        tmp_path,
        [PHYSICAL_TARGET, "--stage", "physical-prior"],
    )

    assert result.returncode == 2
    assert "official PhysTwin runtime marker changed" in result.stderr
    assert route == []
    assert not marker.exists()
    assert not config_marker.exists()


def test_dispatcher_preserves_inherited_selector_repair_identity(
    tmp_path: Path,
) -> None:
    repair_id_log = tmp_path / "selector-repair-id.log"
    selector_repair_id = (
        "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
    )

    result, route, marker, config_marker = _dispatch(
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
    assert not config_marker.exists()


def test_dispatcher_rejects_missing_or_changed_fallback_config(tmp_path: Path) -> None:
    missing, route, marker, config_marker = _dispatch(
        tmp_path / "missing",
        [PHYSICAL_TARGET, "--stage", "frame-zero"],
    )
    assert missing.returncode == 2
    assert "fallback config binding is not unique" in missing.stderr
    assert route == []
    assert not marker.exists()
    assert not config_marker.exists()

    changed, route, marker, config_marker = _dispatch(
        tmp_path / "changed",
        [
            PHYSICAL_TARGET,
            "--stage",
            "frame-zero",
            FALLBACK_CONFIG_FLAG,
            CORRECTED_FALLBACK_CONFIG,
        ],
    )
    assert changed.returncode == 2
    assert "no longer matches the retained failure" in changed.stderr
    assert route == []
    assert not marker.exists()
    assert not config_marker.exists()


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


def test_fallback_config_routing_repair_is_content_addressed_and_closed() -> None:
    payload = json.loads(FALLBACK_CONFIG_AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert not any(payload["information_boundary"].values())
    assert payload["failed_execution_evidence"]["physical_manifest_count"] == 0
    assert payload["failed_execution_evidence"]["source_prediction_seal_count"] == 0
    assert payload["repair_scope"]["fallback_config_argument_routed"]
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "fallback_config_argument_routed"
    )

    correction = payload["correction"]
    previous = ROOT / correction["previous_config"]["path"]
    corrected = ROOT / correction["corrected_config"]["path"]
    assert (
        hashlib.sha256(previous.read_bytes()).hexdigest()
        == (correction["previous_config"]["file_sha256"])
    )
    assert (
        hashlib.sha256(corrected.read_bytes()).hexdigest()
        == (correction["corrected_config"]["file_sha256"])
    )

    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
    corrected_payload = json.loads(corrected.read_text(encoding="utf-8"))
    for config_payload, expected in (
        (previous_payload, correction["previous_config"]),
        (corrected_payload, correction["corrected_config"]),
    ):
        canonical = json.dumps(
            config_payload["config"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        canonical_digest = hashlib.sha256(canonical).hexdigest()
        assert canonical_digest == config_payload["config_sha256"]
        assert canonical_digest == expected["canonical_config_sha256"]
        assert (
            isinstance(config_payload["config"].get("method"), dict)
            is expected["method_config_present"]
        )

    method = corrected_payload["config"]["method"]
    fields = FrameZeroInitializerConfig.__dataclass_fields__
    initializer = FrameZeroInitializerConfig(**{name: method[name] for name in fields})
    assert initializer.minimum_original_point_count == 128
    assert initializer.minimum_fallback_point_count == 128


def test_workflow_binds_and_receipts_fallback_config_routing_repair() -> None:
    payload = json.loads(FALLBACK_CONFIG_AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    amendment_digest = hashlib.sha256(
        FALLBACK_CONFIG_AMENDMENT.read_bytes()
    ).hexdigest()
    dispatcher_digest = hashlib.sha256(DISPATCHER.read_bytes()).hexdigest()

    assert f"FALLBACK_CONFIG_ROUTE_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"FALLBACK_CONFIG_ROUTE_REPAIR_SHA256: {amendment_digest}" in workflow
    assert f"DISPATCHER_SHA256: {dispatcher_digest}" in workflow
    assert str(FALLBACK_CONFIG_AMENDMENT.relative_to(ROOT)) in workflow
    assert "BPT_FRAME_ZERO_FALLBACK_CONFIG_REPAIR_MARKER" in workflow
    assert "runtime_fallback_config_routing_repair" in workflow

    dispatcher = DISPATCHER.read_text(encoding="utf-8")
    correction = payload["correction"]
    assert payload["repair_id"] in dispatcher
    assert correction["previous_config"]["path"] in dispatcher
    assert correction["corrected_config"]["path"] in dispatcher
    assert correction["previous_config"]["file_sha256"] in dispatcher
    assert correction["corrected_config"]["file_sha256"] in dispatcher
