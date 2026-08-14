#!/usr/bin/env python3
"""Audit descriptive graph-part variation in a MatPhys reconstruction export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_matphys_causal import (
    _configure_matphys_imports,
    _install_torchvision_nms_stub,
    _source_commit,
)
from run_matphys_reconstruction_control import _install_warp_warn_compatibility

from bayesian_phystwin.matphys_part_model import summarize_part_spring_field


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validated_identity(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file() or _identity(path)["sha256"] != value.get("sha256"):
        raise ValueError(f"{label} identity changed")
    return path


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_manifest")
    parser.add_argument("proxy_summary")
    parser.add_argument("output_json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--experiments-optimization-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = Path(args.export_manifest).resolve()
    proxy_path = Path(args.proxy_summary).resolve()
    output_path = Path(args.output_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    matphys_root = Path(args.matphys_root).resolve()
    data_root = Path(args.data_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError("export manifest omits its case")
    case_name = str(case.get("name", ""))
    records = proxy.get("cases")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("part audit requires one proxy case")
    if records[0].get("name") != case_name:
        raise ValueError("proxy and export cases differ")
    train_ready_path = _validated_identity(
        records[0].get("train_ready"), label="train-ready proxy"
    )
    train_ready = torch.load(
        train_ready_path, map_location="cpu", weights_only=False
    )
    part_count = int(train_ready["num_parts"])
    if part_count < 2 or tuple(train_ready["part_features"].shape)[0] != part_count:
        raise ValueError("proxy does not contain multiple complete parts")

    if _validated_identity(
        manifest.get("checkpoint"), label="export checkpoint"
    ) != checkpoint_path:
        raise ValueError("checkpoint differs from the reconstruction export")
    source_commit = str(manifest.get("source_commit", ""))
    if _source_commit(matphys_root) != source_commit:
        raise ValueError("MatPhys source revision differs from the export")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_model_args = checkpoint.get("args")
    if not isinstance(raw_model_args, dict):
        raise ValueError("checkpoint omits its model arguments")
    model_args = dict(raw_model_args)
    model_args.update(
        {
            "base_path": str(data_root),
            "experiments_dir": str(Path(args.experiments_dir).resolve()),
            "experiments_optimization_dir": str(
                Path(args.experiments_optimization_dir).resolve()
            ),
            "case_to_material": str(proxy_path.parent / "case_to_material.json"),
            "results_dir": str(proxy_path.parent / "results"),
            "sem_cache_dir": str(proxy_path.parent / "semantic_cache"),
            "gaussian_root": "__disabled__",
            "device": str(args.device),
            "rank": 0,
        }
    )
    os.chdir(matphys_root)
    _configure_matphys_imports(matphys_root)
    _install_warp_warn_compatibility()
    from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset

    namespace = SimpleNamespace(**model_args)
    dataset = MaterialParamDataset(
        MaterialDatasetConfig(
            base_path=namespace.base_path,
            sem_cache_dir=namespace.sem_cache_dir,
            experiments_dir=namespace.experiments_dir,
            experiments_optimization_dir=namespace.experiments_optimization_dir,
            case_to_material_path=namespace.case_to_material,
            results_dir=namespace.results_dir,
            use_knn_topology=namespace.use_knn_topology,
            object_knn=namespace.object_knn,
            object_radius=namespace.object_radius,
            object_max_neighbours=namespace.object_max_neighbours,
            controller_radius=namespace.controller_radius,
            controller_max_neighbours=namespace.controller_max_neighbours,
        )
    )
    matches = [sample for sample in dataset.samples if sample["case_name"] == case_name]
    if len(matches) != 1:
        raise ValueError("audited proxy must produce exactly one requested sample")
    sample = matches[0]
    object_part_index = np.asarray(sample["edge_part_idx"], dtype=np.int64)
    controller_part_index = np.asarray(sample["ctrl_part_idx"], dtype=np.int64)

    spring_path = _validated_identity(case.get("spring_field"), label="spring field")
    spring = np.load(spring_path, allow_pickle=False)
    combined = np.concatenate((object_part_index, controller_part_index))
    if len(combined) != len(spring) or set(combined.tolist()) != set(range(part_count)):
        raise ValueError("edge-part indices do not cover the exported spring field")
    summary = summarize_part_spring_field(spring, combined)
    result = {
        "schema_version": 1,
        "contract": "matphys-reconstruction-part-spring-field-audit-v1",
        "claim_boundary": (
            "Descriptive all-frame reconstruction diagnostic only. This audit does "
            "not alter the capacity gate or authorize causal, predictive, transfer, "
            "calibration, or state-of-the-art claims."
        ),
        "case": case_name,
        "artifacts": {
            "export_manifest": _identity(manifest_path),
            "proxy_summary": _identity(proxy_path),
            "train_ready": _identity(train_ready_path),
            "spring_field": _identity(spring_path),
            "checkpoint": _identity(checkpoint_path),
        },
        "matphys_source_commit": source_commit,
        "part_count": part_count,
        "object_spring_count": int(len(object_part_index)),
        "controller_spring_count": int(len(controller_part_index)),
        "edge_part_index_sha256": hashlib.sha256(
            np.ascontiguousarray(combined.astype("<i8", copy=False)).tobytes()
        ).hexdigest(),
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "audit_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    _install_torchvision_nms_stub()
    main()
