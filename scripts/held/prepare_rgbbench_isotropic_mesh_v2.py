#!/usr/bin/env python3
"""Build the target-free RGBench v2 physical mesh manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from bayesian_phystwin.rgbench_isotropic_mesh import (
    RGBenchIsotropicMeshConfig,
    build_isotropic_mesh_artifact,
    build_isotropic_mesh_manifest,
    write_json_once,
)
from bayesian_phystwin.rgbench_online_belief import sha256_file
from bayesian_phystwin.rgbench_protocol import (
    DATASET_REVISION,
    PAPER_GARMENTS,
    RGBENCH_COMMIT,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("preparing RGBench v2 requires PyYAML") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return payload


def _source_mesh_by_garment(
    dataset_manifest: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    cases = dataset_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("dataset manifest has no cases")
    result: dict[str, tuple[str, str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("dataset manifest contains a non-object case")
        garment = str(case["garment"])
        value = (
            str(case["mesh_relative_path"]),
            str(case["mesh_sha256"]),
        )
        previous = result.setdefault(garment, value)
        if previous != value:
            raise ValueError(f"{garment} has inconsistent source meshes")
    if set(result) != set(PAPER_GARMENTS):
        raise ValueError("dataset manifest garment cohort changed")
    return result


def main() -> None:
    args = _parse_args()
    benchmark = args.benchmark_root.resolve()
    dataset = args.dataset_root.resolve()
    manifest_path = args.dataset_manifest.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    incomplete = output.with_name(output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"stale incomplete output root exists: {incomplete}")
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if git_head != RGBENCH_COMMIT:
        raise ValueError(f"RGBench checkout is {git_head}, expected {RGBENCH_COMMIT}")
    dataset_manifest = _load_json(manifest_path)
    if (
        dataset_manifest.get("artifact_kind") != "RGBenchDatasetManifest"
        or dataset_manifest.get("rgbbench_commit") != RGBENCH_COMMIT
        or dataset_manifest.get("dataset_revision") != DATASET_REVISION
    ):
        raise ValueError("dataset manifest provenance changed")
    source_meshes = _source_mesh_by_garment(dataset_manifest)
    config = RGBenchIsotropicMeshConfig()
    artifact_paths: list[Path] = []
    incomplete.mkdir(parents=True)
    for garment in PAPER_GARMENTS:
        source_relative, expected_sha256 = source_meshes[garment]
        source_mesh = dataset / "meshes" / source_relative
        if sha256_file(source_mesh) != expected_sha256:
            raise ValueError(f"{garment} source mesh changed after dataset lock")
        parameters_relative = f"configs/cloth_params/{garment}.yaml"
        parameters_path = benchmark / parameters_relative
        parameters = _load_yaml(parameters_path)
        configured_mesh = parameters.get("cloth_model_file_name")
        if configured_mesh != source_relative:
            raise ValueError(f"{garment} configured mesh changed")
        pins = parameters.get("shoulder_index")
        if (
            not isinstance(pins, list)
            or len(pins) != 2
            or not all(isinstance(value, int) for value in pins)
        ):
            raise ValueError(f"{garment} has invalid fling pin indices")
        derived_relative = f"meshes/{garment}.obj"
        artifact = build_isotropic_mesh_artifact(
            garment=garment,
            source_mesh=source_mesh,
            source_mesh_relative_path=source_relative,
            cloth_parameters=parameters_path,
            cloth_parameters_relative_path=parameters_relative,
            source_fling_pin_indices=(pins[0], pins[1]),
            derived_mesh=incomplete / derived_relative,
            derived_mesh_relative_path=derived_relative,
            config=config,
        )
        artifact_path = incomplete / f"artifacts/{garment}.json"
        write_json_once(artifact_path, artifact.descriptor())
        artifact_paths.append(artifact_path)

    mesh_manifest = build_isotropic_mesh_manifest(
        tuple(artifact_paths),
        root=incomplete,
        rgbbench_commit=RGBENCH_COMMIT,
        dataset_revision=DATASET_REVISION,
        dataset_manifest_artifact_sha256=str(
            dataset_manifest["artifact_sha256"]
        ),
        dataset_manifest_file_sha256=sha256_file(manifest_path),
    )
    write_json_once(incomplete / "manifest.json", mesh_manifest.descriptor())
    incomplete.replace(output)
    print(mesh_manifest.artifact_sha256)


if __name__ == "__main__":
    main()
