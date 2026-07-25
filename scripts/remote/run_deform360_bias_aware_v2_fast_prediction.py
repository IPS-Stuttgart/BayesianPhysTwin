#!/usr/bin/env python3
"""Run the frozen v2 prediction with an exact camera-search accelerator."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from bayesian_phystwin import deform360_raw_camera_observation as raw_camera
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    activate_v2_prediction_runtime,
    patch_v2_stage_module,
    validate_v2_execution_lock,
)
from bayesian_phystwin.deform360_exact_camera_selector import (
    select_frame_zero_observation_plan_exact_fast,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _argument_value(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    _require(len(matches) == 1, f"expected one {option} argument")
    index = matches[0]
    _require(index + 1 < len(arguments), f"{option} has no value")
    return arguments[index + 1]


def _load_prediction_stage(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_deform360_bias_aware_v2_fast_prediction",
        path,
    )
    _require(spec is not None and spec.loader is not None, "cannot load stage")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--selector-cache", type=Path, required=True)
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve()
    execution_lock = args.execution_lock.resolve()
    validate_v2_execution_lock(execution_lock, repository=repository)
    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    load_bias_aware_prospective_v2_protocol(protocol_path, root=repository)
    script = repository / "scripts/remote/run_deform360_bias_aware_prediction.py"
    module = _load_prediction_stage(script)
    patch_v2_stage_module(
        module,
        stage="prediction",
        repository=repository,
        execution_lock=execution_lock,
    )
    original_selector = raw_camera.select_frame_zero_observation_plan

    def exact_selector(*selector_args, **selector_kwargs):
        return select_frame_zero_observation_plan_exact_fast(
            *selector_args,
            cache_dir=args.selector_cache,
            **selector_kwargs,
        )

    raw_camera.select_frame_zero_observation_plan = exact_selector
    previous = sys.argv
    sys.argv = [str(script), *stage_arguments]
    try:
        with activate_v2_prediction_runtime():
            return int(module.main())
    finally:
        sys.argv = previous
        raw_camera.select_frame_zero_observation_plan = original_selector


if __name__ == "__main__":
    raise SystemExit(main())
