"""Download the exact H2-locked fresh Deform360 object panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.deform360_adaptive_covariance_confirmation_download import (
    download_confirmation_panel_by_object,
    write_confirmation_download_manifest,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h2_production_entrypoint,
)

ENTRYPOINT_REPOSITORY_PATH = (
    "src/bayesian_phystwin/cli/deform360_adaptive_covariance_confirmation_download.py"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--h2-commit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-h1", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--object-delay-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_confirmation_h2_production_entrypoint(
        args.adapter_repo,
        args.lock,
        args.h2_commit,
        expected_h1=args.expected_h1,
        entrypoint_file=__file__,
        entrypoint_repository_path=ENTRYPOINT_REPOSITORY_PATH,
    )
    from huggingface_hub import HfApi, hf_hub_download

    result = download_confirmation_panel_by_object(
        args.lock,
        args.h2_commit,
        args.output_root,
        max_workers=args.max_workers,
        object_delay_seconds=args.object_delay_seconds,
        list_repo_tree=HfApi().list_repo_tree,
        hub_download=hf_hub_download,
        expected_h1=args.expected_h1,
    )
    write_confirmation_download_manifest(args.manifest, result)
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "object_count": result["object_count"],
                "output_root": args.output_root,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
