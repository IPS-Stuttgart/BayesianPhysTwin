from __future__ import annotations

import argparse
from pathlib import Path

from causal4d_public.deform360_reusable_sota_download import (
    development_download_plan,
    download_development_panel,
    write_download_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the locked Deform360 reusable-SOTA development panel."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = development_download_plan(args.protocol)
    if args.dry_run:
        print(plan)
        return
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "development download requires the optional huggingface_hub package"
        ) from error
    manifest = download_development_panel(
        args.protocol,
        args.output_root,
        max_workers=args.workers,
        snapshot_download=snapshot_download,
    )
    write_download_manifest(args.manifest, manifest)
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
