from __future__ import annotations

from pathlib import Path
import sys

from bayesian_phystwin import deform360_held_physical_prior as physical
from bayesian_phystwin import deform360_held_v8_builders as builders


def _automatic_twin_arguments(object_id: str, episode_id: int) -> list[str]:
    return [
        "--repo",
        builders.V8_UPSTREAM_ROOT,
        "--object-id",
        object_id,
        "--episode-id",
        str(episode_id),
        "--phase",
        "calibration",
        "--source-admission-passed",
        "--prediction-only-input",
    ]


def test_only_external_calibration_case_receives_held_lock(
    monkeypatch,
) -> None:
    lock = "/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v8/calibration-lock.json"
    monkeypatch.setattr(sys, "argv", ["physical", "--lock", lock])
    script = Path(builders.V8_UPSTREAM_ROOT) / (
        "scripts/remote/build_deform360_automatic_episode_twin.py"
    )
    external = builders._v8_isolated_runpy_command(
        "/usr/bin/python3",
        script,
        import_roots=(Path(builders.V8_UPSTREAM_ROOT) / "src",),
        arguments=_automatic_twin_arguments("072-cotton-clohesline", 3),
    )
    ordinary = builders._v8_isolated_runpy_command(
        "/usr/bin/python3",
        script,
        import_roots=(Path(builders.V8_UPSTREAM_ROOT) / "src",),
        arguments=_automatic_twin_arguments("083-blanket-cloth", 3),
    )
    assert external[-2:] == ["--held-calibration-lock", lock]
    assert "--held-calibration-lock" not in ordinary


def test_physical_context_installs_and_restores_external_runtime_contract() -> None:
    original_command = physical._isolated_runpy_command
    original_files = physical.UPSTREAM_FILE_SHA256
    with builders.explicit_v8_builder_context("physical"):
        assert physical._isolated_runpy_command is builders._v8_isolated_runpy_command
        assert physical.UPSTREAM_FILE_SHA256 is builders.V8_UPSTREAM_FILE_SHA256
        assert (
            physical.UPSTREAM_RUNTIME_BUNDLE_CONTRACT
            is builders.V8_UPSTREAM_RUNTIME_BUNDLE_CONTRACT
        )
        assert (
            physical.HELD_PHYSICAL_NUMERIC_CONTRACT
            is builders.V8_HELD_PHYSICAL_NUMERIC_CONTRACT
        )
    assert physical._isolated_runpy_command is original_command
    assert physical.UPSTREAM_FILE_SHA256 is original_files


def test_external_runtime_contract_binds_authorizer_and_builder() -> None:
    assert builders.V8_UPSTREAM_FILE_SHA256[
        "scripts/remote/build_deform360_automatic_episode_twin.py"
    ] == "8b8763905bb92092066503ac54f0cadb457dc5a6c10484a4e520801fe7268fa5"
    assert builders.V8_UPSTREAM_FILE_SHA256[
        "src/causal4d_public/deform360_external_calibration.py"
    ] == "4dcbb6f663a6d6989ce25ab78a53e0d3a5b412cce990dcd7c904b450ab8dceae"
    assert (
        builders.V8_UPSTREAM_LOCK_BINDING_BY_PATH[
            "src/causal4d_public/deform360_external_calibration.py"
        ]
        == "upstream_external_calibration_source"
    )
