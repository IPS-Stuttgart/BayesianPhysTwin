#!/usr/bin/env python3
"""Build the outcome-blind RGBench online-belief dataset lock."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bayesian_phystwin.rgbench_protocol import (
    DATASET_REVISION,
    PAPER_GARMENTS,
    RGBENCH_COMMIT,
    build_rgbbench_dataset_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("preparing the RGBench lock requires PyYAML") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return payload


def main() -> None:
    args = _parse_args()
    benchmark = args.benchmark_root.resolve()
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if git_head != RGBENCH_COMMIT:
        raise ValueError(f"RGBench checkout is {git_head}, expected {RGBENCH_COMMIT}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite dataset lock: {args.output}")
    experiment_library = _load_yaml(
        benchmark / "configs/experiment_library.yaml"
    )
    mesh_relative_paths: dict[str, str] = {}
    for garment in PAPER_GARMENTS:
        cloth_parameters = _load_yaml(
            benchmark / f"configs/cloth_params/{garment}.yaml"
        )
        mesh_path = cloth_parameters.get("cloth_model_file_name")
        if not isinstance(mesh_path, str) or not mesh_path:
            raise ValueError(f"{garment} has no cloth_model_file_name")
        mesh_relative_paths[garment] = mesh_path
    manifest = build_rgbbench_dataset_manifest(
        args.dataset_root.resolve(),
        benchmark,
        experiment_library=experiment_library,
        mesh_relative_paths=mesh_relative_paths,
        dataset_revision=args.dataset_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(manifest.artifact_sha256)


if __name__ == "__main__":
    main()
