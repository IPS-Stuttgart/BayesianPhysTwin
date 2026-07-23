#!/usr/bin/env python3
"""Run one checksum-bound post-barrier Deform360 outcome stage under H2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys

ENTRYPOINT_REPOSITORY_PATH = (
    "scripts/remote/run_deform360_adaptive_confirmation_outcome_stage.py"
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
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--compatibility-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("authorized-future", "authorized-outcome"),
        required=True,
    )
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    # Adapter authorization below requires a source-only checkout.  Prevent
    # these deferred adapter imports from materializing ignored bytecode.
    sys.dont_write_bytecode = True
    adapter = args.adapter_repo.absolute()
    _reject_adapter_python_caches(adapter)
    sys.path.insert(0, str(adapter / "src"))

    from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
        load_confirmation_cohort_lock,
    )
    from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
        validate_confirmation_h2_production_entrypoint,
    )
    from bayesian_phystwin.deform360_adaptive_covariance_confirmation_outcome_adapter import (
        run_confirmation_outcome_stage,
    )

    validate_confirmation_h2_production_entrypoint(
        adapter,
        args.lock,
        args.h2_commit,
        expected_h1=args.expected_h1,
        entrypoint_file=__file__,
        entrypoint_repository_path=ENTRYPOINT_REPOSITORY_PATH,
    )
    lock = load_confirmation_cohort_lock(
        args.lock,
        expected_implementation_commit_h1=args.expected_h1,
    )
    case_ids = tuple(lock["selected_case_ids"])
    case_seal_dirs = {
        case_id: args.case_root.absolute() / case_id for case_id in case_ids
    }
    nested_measurement_dirs = {
        case_id: args.measurement_root.absolute() / case_id for case_id in case_ids
    }
    return run_confirmation_outcome_stage(
        args.stage,
        stage_arguments,
        adapter_repository=adapter,
        execution_repository=args.execution_repo.absolute(),
        deform360_repository=args.deform360_repo.absolute(),
        lock_path=args.lock.absolute(),
        h2_commit=args.h2_commit,
        barrier_path=args.barrier.absolute(),
        case_seal_dirs=case_seal_dirs,
        nested_measurement_dirs=nested_measurement_dirs,
        compatibility_root=args.compatibility_root.absolute(),
        expected_h1=args.expected_h1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
