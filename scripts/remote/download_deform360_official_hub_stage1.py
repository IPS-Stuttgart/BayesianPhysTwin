#!/usr/bin/env python3
"""Preflight and optionally download the locked Deform360 calibration payload."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bayesian_phystwin.deform360_official_hub_stage1 import (
    HubFileRecord,
    build_official_hub_stage1_preflight,
    download_official_hub_stage1,
    load_official_hub_stage1_lock,
    validate_official_hub_stage1_preflight,
    write_official_hub_stage1_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument(
        "--preflight-input",
        type=Path,
        help="reuse an already sealed preflight without listing Hub object trees",
    )
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retry-attempts", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="use Hugging Face Xet instead of the rate-limit-resistant HTTP path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if (args.download_root is None) != (args.download_manifest is None):
        raise ValueError(
            "--download-root and --download-manifest must be supplied together"
        )
    lock = load_official_hub_stage1_lock(
        args.repository,
        args.protocol,
        args.selection,
    )
    if args.preflight_input is not None:
        try:
            preflight = json.loads(args.preflight_input.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot load the sealed Stage-1 preflight") from error
        if not isinstance(preflight, dict):
            raise ValueError("sealed Stage-1 preflight must be a JSON object")
        validate_official_hub_stage1_preflight(preflight, lock=lock)
    else:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Stage 1 requires the optional huggingface_hub package"
            ) from error

        api = HfApi()
        tree_by_object: dict[str, tuple[HubFileRecord, ...]] = {}
        metadata_by_object: dict[str, bytes] = {}
        for selected in lock.calibration:
            prefix = f"raw/{selected.object_id}/"
            entries = api.list_repo_tree(
                repo_id=lock.dataset_repository,
                path_in_repo=f"raw/{selected.object_id}",
                recursive=True,
                expand=True,
                repo_type="dataset",
                revision=lock.dataset_revision,
            )
            files: list[HubFileRecord] = []
            for entry in entries:
                blob_id = getattr(entry, "blob_id", None)
                if blob_id is None:
                    continue
                path = str(getattr(entry, "path", ""))
                if not path.startswith(prefix):
                    raise ValueError(f"Hub listing escaped calibration object: {path}")
                lfs = getattr(entry, "lfs", None)
                files.append(
                    HubFileRecord(
                        path=path,
                        size=int(getattr(entry, "size", -1)),
                        blob_id=str(blob_id),
                        lfs_sha256=(None if lfs is None else str(lfs.sha256)),
                    )
                )
            tree_by_object[selected.object_id] = tuple(files)
            metadata_cache = Path(
                hf_hub_download(
                    repo_id=lock.dataset_repository,
                    filename=selected.metadata_path,
                    repo_type="dataset",
                    revision=lock.dataset_revision,
                    cache_dir=(None if args.cache_dir is None else str(args.cache_dir)),
                )
            )
            metadata_by_object[selected.object_id] = metadata_cache.read_bytes()

        preflight = build_official_hub_stage1_preflight(
            lock,
            tree_by_object=tree_by_object,
            metadata_bytes_by_object=metadata_by_object,
        )
    write_official_hub_stage1_manifest(args.preflight_manifest, preflight)
    summary: dict[str, object] = {
        "preflight_manifest": str(args.preflight_manifest.resolve()),
        "preflight_sha256": preflight["preflight_sha256"],
        "object_count": preflight["object_count"],
        "file_count": preflight["file_count"],
        "total_bytes": preflight["total_bytes"],
        "downloaded": False,
    }
    if args.download_root is not None and args.download_manifest is not None:
        if args.retry_attempts < 1:
            raise ValueError("--retry-attempts must be positive")
        if args.retry_base_seconds < 0:
            raise ValueError("--retry-base-seconds must be non-negative")
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError as error:
            raise RuntimeError(
                "Stage 1 download requires the optional huggingface_hub package"
            ) from error

        def download(path: str) -> str:
            for attempt in range(args.retry_attempts):
                try:
                    return hf_hub_download(
                        repo_id=lock.dataset_repository,
                        filename=path,
                        repo_type="dataset",
                        revision=lock.dataset_revision,
                        cache_dir=(
                            None if args.cache_dir is None else str(args.cache_dir)
                        ),
                    )
                except HfHubHTTPError as error:
                    status = getattr(getattr(error, "response", None), "status_code", 0)
                    retryable = status == 429 or status >= 500
                    if not retryable or attempt + 1 == args.retry_attempts:
                        raise
                    delay = min(
                        args.retry_base_seconds * (2**attempt),
                        120.0,
                    )
                    time.sleep(delay)
            raise AssertionError("unreachable retry loop")

        result = download_official_hub_stage1(
            preflight,
            args.download_root,
            lock=lock,
            hub_download=download,
            max_workers=args.max_workers,
        )
        write_official_hub_stage1_manifest(args.download_manifest, result)
        summary.update(
            {
                "downloaded": True,
                "download_manifest": str(args.download_manifest.resolve()),
                "download_sha256": result["download_sha256"],
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
