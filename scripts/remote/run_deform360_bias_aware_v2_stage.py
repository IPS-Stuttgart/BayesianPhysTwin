#!/usr/bin/env python3
"""Run one frozen v1 prediction stage under the additive prospective v2 lock."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    activate_v2_prediction_runtime,
    patch_v2_stage_module,
    validate_v2_execution_lock,
)


STAGE_SCRIPTS = {
    "prepare-source": "prepare_deform360_bias_aware_source.py",
    "stage-prefix": "stage_deform360_bias_aware_prediction_prefix.py",
    "frame-zero": "run_deform360_bias_aware_frame_zero.py",
    "automatic-twin": "build_deform360_bias_aware_automatic_twin.py",
    "physical-prior": "run_deform360_bias_aware_physical_prior.py",
    "prediction": "run_deform360_bias_aware_prediction.py",
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


def _load_stage(path: Path, stage: str):
    name = f"_deform360_bias_aware_v2_{stage.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load stage")
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
    validate_v2_execution_lock(execution_lock, repository=repository)
    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    load_bias_aware_prospective_v2_protocol(protocol_path, root=repository)
    script = repository / "scripts" / "remote" / STAGE_SCRIPTS[args.stage]
    module = _load_stage(script, args.stage)
    with activate_v2_prediction_runtime():
        patch_v2_stage_module(
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
