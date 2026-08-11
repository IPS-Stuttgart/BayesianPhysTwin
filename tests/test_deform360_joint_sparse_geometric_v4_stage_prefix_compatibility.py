from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/remote/run_deform360_joint_sparse_physical_source_v5.py"
LOCK = (
    ROOT
    / "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)


def _load_runner() -> ModuleType:
    name = "_test_deform360_joint_sparse_physical_source_v5_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_stage_prefix_arguments(*, repository: Path = ROOT) -> list[str]:
    return [
        "--repo",
        str(repository),
        "--protocol",
        str(LOCK),
        "--role",
        "calibration",
        "--source-aligned-root",
        "/retained/source",
        "--object-id",
        "026-sock-cloth",
    ]


def test_stage_prefix_removes_only_exact_legacy_context_arguments() -> None:
    module = _load_runner()
    original = _legacy_stage_prefix_arguments()

    normalized = module._normalize_stage_arguments(
        "stage-prefix",
        original,
        repository=ROOT.resolve(),
    )

    assert original == _legacy_stage_prefix_arguments()
    assert "--repo" not in normalized
    assert "--role" not in normalized
    assert normalized == [
        "--protocol",
        str(LOCK),
        "--source-aligned-root",
        "/retained/source",
        "--object-id",
        "026-sock-cloth",
    ]


def test_non_stage_prefix_arguments_are_not_rewritten() -> None:
    module = _load_runner()
    original = ["--protocol", str(LOCK), "--repo", str(ROOT)]

    normalized = module._normalize_stage_arguments(
        "physical-prior",
        original,
        repository=ROOT.resolve(),
    )

    assert normalized == original
    assert normalized is not original


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (
            _legacy_stage_prefix_arguments(repository=Path("/different/repository")),
            "legacy stage-prefix repository changed",
        ),
        (
            [
                value if value != "calibration" else "target"
                for value in _legacy_stage_prefix_arguments()
            ],
            "legacy stage-prefix role changed",
        ),
        (
            [
                value
                for value in _legacy_stage_prefix_arguments()
                if value not in {"--role", "calibration"}
            ],
            "expected one --role argument",
        ),
    ],
)
def test_stage_prefix_legacy_context_drift_fails_closed(
    arguments: list[str],
    match: str,
) -> None:
    module = _load_runner()

    with pytest.raises(ValueError, match=match):
        module._normalize_stage_arguments(
            "stage-prefix",
            arguments,
            repository=ROOT.resolve(),
        )


def test_runner_main_passes_only_current_stage_prefix_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: (
            SimpleNamespace(
                execution_repo=ROOT,
                execution_lock=LOCK,
                stage="stage-prefix",
            ),
            _legacy_stage_prefix_arguments(),
        ),
    )
    monkeypatch.setattr(
        module,
        "validate_joint_sparse_physical_execution_v5",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "patch_joint_sparse_physical_stage_v5",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "activate_joint_sparse_physical_runtime_v5",
        nullcontext,
    )

    class Stage:
        @staticmethod
        def main() -> int:
            observed["argv"] = list(sys.argv)
            return 0

    monkeypatch.setattr(module, "_load_stage", lambda *_args: Stage())

    assert module.main() == 0
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert "--repo" not in argv
    assert "--role" not in argv
    assert argv[0].endswith("stage_deform360_bias_aware_prediction_prefix.py")
    assert argv[1:] == [
        "--protocol",
        str(LOCK),
        "--source-aligned-root",
        "/retained/source",
        "--object-id",
        "026-sock-cloth",
    ]
