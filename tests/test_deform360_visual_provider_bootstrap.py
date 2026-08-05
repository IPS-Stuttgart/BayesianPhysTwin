from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "science"
    / "bootstrap_deform360_visual_provider_models.py"
)
SPEC = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_official_hub_visuotactile_v1_visual_provider_spec.json"
)


def _module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "deform360_visual_provider_bootstrap", SCRIPT
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_bootstrap_groups_shared_motioncrafter_snapshot() -> None:
    module = _module()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    groups = module._exact_source_groups(spec)

    motion_revision = "fc7b18d5657184607bf4501b02d64ada7540b4e3"
    assert (
        "TencentARC/MotionCrafter",
        motion_revision,
    ) in groups
    assert groups[("TencentARC/MotionCrafter", motion_revision)] == (
        "geometry_motion_vae/config.json",
        "geometry_motion_vae/diffusion_pytorch_model.safetensors",
        "unet_determ/config.json",
        "unet_determ/diffusion_pytorch_model.safetensors",
    )
    assert len(groups) == 3


def test_bootstrap_downloads_only_exact_declared_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[dict[str, Any]] = []

    def snapshot_download(**arguments: Any) -> str:
        calls.append(arguments)
        repository = arguments["repo_id"]
        revision = arguments["revision"]
        cache = Path(arguments["cache_dir"])
        snapshot = (
            cache
            / ("models--" + repository.replace("/", "--"))
            / "snapshots"
            / revision
        )
        for relative in arguments["allow_patterns"]:
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        return str(snapshot)

    fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    result = module.bootstrap_exact_model_snapshots(
        spec_path=SPEC,
        cache_directory=tmp_path / "cache",
        token="test-token",
    )

    assert result["source_count"] == 3
    assert result["selected_raw_payloads_opened"] is False
    assert result["calibration_payloads_opened"] is False
    assert result["target_outcomes_used"] is False
    assert len(calls) == 3
    for call in calls:
        assert len(call["revision"]) == 40
        assert call["token"] == "test-token"
        assert call["allow_patterns"]
        assert all("*" not in path for path in call["allow_patterns"])


def test_complete_cache_skips_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cache = tmp_path / "cache"
    for (repository, revision), members in module._exact_source_groups(spec).items():
        snapshot = (
            cache
            / ("models--" + repository.replace("/", "--"))
            / "snapshots"
            / revision
        )
        for relative in members:
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")

    def unexpected_download(**arguments: Any) -> str:
        raise AssertionError(arguments)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=unexpected_download),
    )
    result = module.bootstrap_exact_model_snapshots(
        spec_path=SPEC,
        cache_directory=cache,
    )

    assert all(not item["download_performed"] for item in result["sources"])


def test_bootstrap_rejects_noncanonical_member(tmp_path: Path) -> None:
    module = _module()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["motioncrafter"]["model_sources"]["unet"]["required_members"] = [
        "../weights"
    ]

    with pytest.raises(ValueError, match="canonical relative path"):
        module._exact_source_groups(spec)


def test_bootstrap_cli_has_no_dataset_surface() -> None:
    module = _module()
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "dataset_root" not in destinations
    assert "confirmation_root" not in destinations
    assert "target" not in destinations
    assert {"spec", "cache_dir", "output"}.issubset(destinations)
