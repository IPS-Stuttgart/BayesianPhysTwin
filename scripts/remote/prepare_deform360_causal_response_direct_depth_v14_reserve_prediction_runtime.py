#!/usr/bin/env python3
"""Lock the mixed-custody V14 prediction runtime before any prefix scan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_prediction_v14 import (
    PARENT_PREDICTION_RUNNER,
    PARENT_RUNTIME_BUILDER,
    build_v14_reserve_prediction_runtime,
    write_v14_reserve_prediction_runtime,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    validate_adaptive_direct_depth_source_lock_v14,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--admission-prelock", type=Path, required=True)
    parser.add_argument("--physical-prelock", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    revision = _git_output(repository, "rev-parse", "HEAD")
    _require(
        not _git_output(
            repository,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ),
        "V14 reserve prediction-runtime repository is dirty",
    )
    source_lock_path = args.source_lock.resolve()
    source_lock = validate_adaptive_direct_depth_source_lock_v14(source_lock_path)
    _require(
        _git_output(
            repository,
            "merge-base",
            "--is-ancestor",
            source_lock.repository_revision,
            revision,
        )
        == "",
        "V14 reserve source-lock revision is not an ancestor",
    )
    implementation_paths = {
        "prediction_module": (
            repository / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_prediction_v14.py"
        ),
        "prediction_runner": repository / PARENT_PREDICTION_RUNNER,
        "preflight_module": (
            repository / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_preflight.py"
        ),
        "runtime_builder": repository / PARENT_RUNTIME_BUILDER,
        "reserve_prediction_module": (
            repository / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_reserve_prediction_v14.py"
        ),
        "reserve_prediction_runner": (
            repository / "scripts/remote/"
            "run_deform360_causal_response_direct_depth_v14_reserve_prediction.py"
        ),
        "reserve_runtime_builder": Path(__file__).resolve(),
    }
    payload = build_v14_reserve_prediction_runtime(
        repository_revision=revision,
        repository=repository,
        method_protocol_path=args.method_protocol.resolve(),
        source_lock_path=source_lock_path,
        admission_custody_path=args.admission_prelock.resolve(),
        physical_custody_path=args.physical_prelock.resolve(),
        admission_root=args.admission_root.resolve(),
        physical_root=args.physical_root.resolve(),
        implementation_paths=implementation_paths,
    )
    output = args.output.resolve()
    write_v14_reserve_prediction_runtime(
        output,
        payload,
        repository=repository,
        method_protocol_path=args.method_protocol.resolve(),
        source_lock_path=source_lock_path,
        admission_custody_path=args.admission_prelock.resolve(),
        physical_custody_path=args.physical_prelock.resolve(),
    )
    print(
        json.dumps(
            {
                "case_count": len(payload["cases"]),
                "config_sha256": payload["config_sha256"],
                "output": str(output),
                "repository_revision": revision,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
