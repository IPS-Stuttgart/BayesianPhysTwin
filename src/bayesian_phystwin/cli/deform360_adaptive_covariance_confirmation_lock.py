"""Create the metadata-only Deform360 adaptive-covariance cohort lock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h1_lock_generation_entrypoint,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    write_confirmation_cohort_lock,
)

ENTRYPOINT_REPOSITORY_PATH = (
    "src/bayesian_phystwin/cli/deform360_adaptive_covariance_confirmation_lock.py"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-repo",
        required=True,
        help="exact clean H1 checkout in which the canonical lock will be written",
    )
    parser.add_argument(
        "--implementation-commit-h1",
        required=True,
        help="full lowercase 40-hex implementation commit fixed before selection",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="absent JSON path to create atomically for the H2 lock commit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_confirmation_h1_lock_generation_entrypoint(
        args.adapter_repo,
        args.output,
        args.implementation_commit_h1,
        entrypoint_file=__file__,
        entrypoint_repository_path=ENTRYPOINT_REPOSITORY_PATH,
    )
    payload = write_confirmation_cohort_lock(
        args.output,
        args.implementation_commit_h1,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "case_count": payload["case_count"],
                "implementation_commit_h1": args.implementation_commit_h1,
                "output": args.output,
                "protocol_id": payload["protocol_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
