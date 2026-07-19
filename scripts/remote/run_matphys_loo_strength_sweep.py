#!/usr/bin/env python3
"""Replay and seal the LOO MatPhys spring-strength family."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_loo_strength_sweep import run_loo_strength_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spring_field_manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--python", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cues-root", required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--overlay-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpu_ids = tuple(value.strip() for value in args.gpu_ids.split(",") if value.strip())
    result = run_loo_strength_sweep(
        args.spring_field_manifest,
        args.output_dir,
        python=args.python,
        official_repo=args.official_repo,
        data_root=args.data_root,
        cues_root=args.cues_root,
        gpu_ids=gpu_ids,
        overlay_workers=args.overlay_workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
