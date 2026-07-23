#!/usr/bin/env python3
"""Run one checksum-bound target-free Deform360 execution stage under H2."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import ModuleType

ENTRYPOINT_REPOSITORY_PATH = (
    "scripts/remote/run_deform360_adaptive_confirmation_external_stage.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_adapter_python_caches(adapter: Path) -> None:
    """Scan the adapter with stdlib only, before any adapter import."""

    _require(
        adapter.is_dir()
        and not adapter.is_symlink()
        and adapter.resolve(strict=True) == adapter,
        "adapter repository is invalid",
    )
    for root in (adapter / "src", adapter / "scripts"):
        _require(
            root.is_dir() and not root.is_symlink(),
            f"adapter Python source root is invalid: {root}",
        )
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    observed = entry.stat(follow_symlinks=False)
                    _require(
                        not stat.S_ISLNK(observed.st_mode),
                        f"adapter Python tree contains a symlink: {path}",
                    )
                    if stat.S_ISDIR(observed.st_mode):
                        _require(
                            entry.name != "__pycache__",
                            f"adapter Python bytecode cache is forbidden: {path}",
                        )
                        pending.append(path)
                    elif stat.S_ISREG(observed.st_mode):
                        _require(
                            path.suffix.lower() not in {".pyc", ".pyo"},
                            f"adapter Python bytecode is forbidden: {path}",
                        )
                    else:
                        _require(
                            False,
                            f"adapter Python tree contains a special file: {path}",
                        )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--adapter-repo", type=Path, required=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--h2-commit", required=True)
    parser.add_argument("--expected-h1", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare-source",
            "stage-prefix",
            "frame-zero",
            "automatic-twin",
            "physical-prior",
        ),
        required=True,
    )
    return parser.parse_known_args()


def _reject_reserved_option_abbreviations(
    arguments: list[str],
    reserved: set[str],
) -> None:
    """Forbid argparse prefixes of paths injected by this wrapper."""

    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        _require(
            not any(
                candidate.startswith(option) and candidate != option
                for candidate in reserved
            ),
            f"reserved option abbreviation is forbidden: {option}",
        )


def _load_stage(path: Path, stage: str) -> ModuleType:
    name = f"_adaptive_confirmation_external_{stage.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load stage")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _remove_bound_path_option(
    arguments: list[str],
    option: str,
    expected: Path,
) -> list[str]:
    """Remove at most one stage option after verifying its exact bound path."""

    result: list[str] = []
    observed: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            _require(index + 1 < len(arguments), f"{option} has no value")
            observed.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith(f"{option}="):
            observed.append(value.split("=", 1)[1])
            index += 1
            continue
        result.append(value)
        index += 1
    _require(len(observed) <= 1, f"{option} is duplicated")
    if observed:
        supplied = Path(observed[0]).absolute()
        _require(
            supplied.resolve(strict=True) == expected.resolve(strict=True),
            f"{option} differs from the wrapper-bound path",
        )
    return result


def _required_path_option(arguments: list[str], option: str) -> Path:
    observed: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            _require(index + 1 < len(arguments), f"{option} has no value")
            observed.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith(f"{option}="):
            observed.append(value.split("=", 1)[1])
        index += 1
    _require(len(observed) == 1, f"{option} must occur exactly once")
    path = Path(observed[0]).absolute()
    _require(path.resolve(strict=True) == path, f"{option} is noncanonical")
    return path


def main() -> int:
    args, stage_arguments = _parse_args()
    # Adapter authorization below requires a source-only checkout.  Prevent
    # these deferred adapter imports from materializing ignored bytecode.
    sys.dont_write_bytecode = True
    reserved_stage_options = {"--protocol"}
    if args.stage in {"prepare-source", "physical-prior"}:
        reserved_stage_options.add("--repo")
    if args.stage in {"prepare-source", "frame-zero", "physical-prior"}:
        reserved_stage_options.add("--deform360-repo")
    _reject_reserved_option_abbreviations(
        stage_arguments,
        reserved_stage_options,
    )
    adapter = args.adapter_repo.absolute()
    execution = args.execution_repo.absolute()
    deform360 = args.deform360_repo.absolute()
    _require(
        adapter.is_dir()
        and execution.is_dir()
        and deform360.is_dir()
        and not adapter.is_symlink()
        and not execution.is_symlink(),
        "execution repository is invalid",
    )
    _reject_adapter_python_caches(adapter)
    sys.path[:0] = [str(adapter / "src"), str(execution / "src")]
    from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
        STAGE_SCRIPTS,
        activate_confirmation_external_runtime,
        load_confirmation_execution_protocol,
        patch_confirmation_stage_module,
        validate_confirmation_h2_production_entrypoint,
        validate_external_execution_repository,
        validate_external_module_provenance,
        validate_deform360_execution_repository,
    )
    from bayesian_phystwin.deform360_adaptive_covariance_confirmation_failure import (
        adapt_frame_zero_original_splat_identity_persistence,
        confirmation_frame_zero_physical_policy,
    )

    lock = args.lock.absolute()
    validate_confirmation_h2_production_entrypoint(
        adapter,
        lock,
        args.h2_commit,
        expected_h1=args.expected_h1,
        entrypoint_file=__file__,
        entrypoint_repository_path=ENTRYPOINT_REPOSITORY_PATH,
    )
    protocol = load_confirmation_execution_protocol(lock)
    h1 = protocol["payload"]["two_commit_freeze"]["implementation_commit_h1"]
    _require(
        h1 == args.expected_h1
        and len(args.h2_commit) == 40
        and args.h2_commit != h1
        and all(character in "0123456789abcdef" for character in args.h2_commit),
        "declared H1/H2 commits are invalid",
    )
    validate_external_execution_repository(execution)
    validate_deform360_execution_repository(deform360)
    stage_arguments = _remove_bound_path_option(
        stage_arguments,
        "--protocol",
        lock,
    )
    stage_prefix_arguments = ["--protocol", str(lock)]
    if args.stage in {"prepare-source", "physical-prior"}:
        stage_arguments = _remove_bound_path_option(
            stage_arguments,
            "--repo",
            execution,
        )
        stage_prefix_arguments.extend(["--repo", str(execution)])
    sys.path.insert(0, str(deform360))
    script = execution / "scripts" / "remote" / STAGE_SCRIPTS[args.stage]
    exit_code: int
    with activate_confirmation_external_runtime(execution):
        module = _load_stage(script, args.stage)
        validate_external_module_provenance(execution)
        patch_confirmation_stage_module(
            module,
            stage=args.stage,
            adapter_repository=adapter,
            execution_repository=execution,
            deform360_repository=deform360,
            lock_path=lock,
            h2_commit=args.h2_commit,
            expected_h1=args.expected_h1,
        )
        if args.stage == "physical-prior":
            original_policy = module.frame_zero_physical_policy

            def adapted_physical_policy(manifest: object) -> str:
                return confirmation_frame_zero_physical_policy(
                    manifest,
                    original_policy=original_policy,
                )

            module.frame_zero_physical_policy = adapted_physical_policy
        previous = sys.argv
        deform360_stage_argument = (
            ["--deform360-repo", str(deform360)]
            if args.stage in {"prepare-source", "frame-zero", "physical-prior"}
            else []
        )
        sys.argv = [
            str(script),
            *stage_prefix_arguments,
            *deform360_stage_argument,
            *stage_arguments,
        ]
        try:
            exit_code = int(module.main())
        finally:
            sys.argv = previous
    if args.stage == "frame-zero" and exit_code == 0:
        staged_case_dir = _required_path_option(
            stage_arguments,
            "--staged-case-dir",
        )
        adapt_frame_zero_original_splat_identity_persistence(
            lock,
            args.h2_commit,
            staged_case_dir,
            deform360,
            expected_h1=h1,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
