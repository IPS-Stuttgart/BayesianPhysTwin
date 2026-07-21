#!/usr/bin/env python3
"""Download only the three fresh calibration objects authorized by v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from bayesian_phystwin.deform360_bias_aware_prospective_v2_download import (
    download_bias_aware_prospective_v2_fresh_by_object,
    write_bias_aware_prospective_v2_download_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--object-delay-seconds", type=float, default=2.0)
    args = parser.parse_args()
    result = download_bias_aware_prospective_v2_fresh_by_object(
        args.protocol,
        args.output_root,
        max_workers=args.max_workers,
        object_delay_seconds=args.object_delay_seconds,
        list_repo_tree=HfApi().list_repo_tree,
        hub_download=hf_hub_download,
    )
    write_bias_aware_prospective_v2_download_manifest(args.manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
