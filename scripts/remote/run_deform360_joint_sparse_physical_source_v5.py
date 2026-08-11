#!/usr/bin/env python3
"""Run one checksum-bound physical-source stage for public Deform360 v5."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from bayesian_phystwin.deform360_joint_sparse_physical_source_v5 import (
    activate_joint_sparse_physical_runtime_v5,
    patch_joint_sparse_physical_stage_v5,
    validate_joint_sparse_physical_execution_v5,
)

STAGE_SCRIPTS = {
    "stage-prefix": "stage_deform360_bias_aware_prediction_prefix.py",
    "frame-zero": "run_deform360_bias_aware_frame_zero.py",
    "automatic-twin": "build_deform360_bias_aware_automatic_twin.py",
    "physical-prior": "run_deform360_bias_aware_physical_prior.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _argument_value(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    _require(len(matches) == 1, f"expected one {option} argument")
    index = matches[0]
    _require(index + 1 < len(arguments), f"{option} has no value")
    return arguments[index + 1]


def _remove_option(arguments: list[str], option: str) -> tuple[str, list[str]]:
    value = _argument_value(arguments, option)
    index = arguments.index(option)
    return value, [*arguments[:index], *arguments[index + 2 :]]


def _normalize_stage_arguments(
    stage: str,
    arguments: list[str],
    *,
    repository: Path,
) -> list[str]:
    """Translate the frozen stage-prefix call to its current strict CLI.

    The archived source runner carries two legacy context options. Their values
    are independently bound by the outer execution wrapper and the v5 source
    lock, so accept and remove them only when both exact legacy values match.
    """

    normalized = list(arguments)
    if stage != "stage-prefix":
        return normalized
    legacy_repository, normalized = _remove_option(normalized, "--repo")
    _require(
        Path(legacy_repository).resolve() == repository,
        "legacy stage-prefix repository changed",
    )
    legacy_role, normalized = _remove_option(normalized, "--role")
    _require(legacy_role == "calibration", "legacy stage-prefix role changed")
    return normalized


def _load_stage(path: Path, stage: str) -> ModuleType:
    name = f"_deform360_joint_sparse_physical_v5_{stage.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load stage")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_SCRIPTS), required=True)
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve()
    execution_lock = args.execution_lock.resolve()
    validate_joint_sparse_physical_execution_v5(
        execution_lock,
        repository=repository,
    )
    stage_arguments = _normalize_stage_arguments(
        args.stage,
        stage_arguments,
        repository=repository,
    )
    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    _require(protocol_path == execution_lock, "stage protocol must be the v5 lock")
    script = repository / "scripts" / "remote" / STAGE_SCRIPTS[args.stage]
    module = _load_stage(script, args.stage)
    with activate_joint_sparse_physical_runtime_v5():
        patch_joint_sparse_physical_stage_v5(
            module,
            stage=args.stage,
            repository=repository,
            execution_lock=execution_lock,
        )
        previous = sys.argv
        sys.argv = [str(script), *stage_arguments]
        try:
            return int(module.main())
        finally:
            sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
