"""Causal input and provenance helpers for the public MatPhys release."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION = 1
_EXTERNAL_BACKBONE_SHARED_FIELDS = (
    "name",
    "source_repository",
    "source_commit",
    "future_observations_used",
    "coordinate_frame",
    "vertex_contract",
    "proxy_contract",
    "claim_boundary",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_uniform_frame_indices(
    frame_count: int,
    evidence_end_frame_exclusive: int,
    sample_count: int,
) -> np.ndarray:
    """Select uniform video inputs without touching the held-out interval."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 1 <= evidence_end_frame_exclusive <= frame_count:
        raise ValueError("evidence end must lie inside the available video")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.linspace(
        0,
        evidence_end_frame_exclusive - 1,
        sample_count,
        dtype=int,
    )


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare_global_material_proxy(
    data_root: str | Path,
    case_names: Sequence[str],
    source_mapping_path: str | Path,
    output_root: str | Path,
    *,
    num_materials: int = 10,
    semantic_dimension: int = 1024,
) -> dict[str, object]:
    """Create deterministic one-part inputs for missing MatPhys release assets.

    MatPhys does not publish its per-case ``node_sem.npz`` and
    ``train_ready.pt`` artifacts. The released simplified decoder does not use
    ``z_sem`` directly, so zeros are sufficient for that unused field. The
    material proxy preserves the repository's declared object-class label as a
    one-hot, single-part distribution. This is an explicit global-material
    ablation, not a reproduction of MatPhys's part decomposition.
    """

    if num_materials < 1 or semantic_dimension < 1:
        raise ValueError("proxy dimensions must be positive")
    root = Path(data_root).resolve()
    mapping_path = Path(source_mapping_path).resolve()
    destination = Path(output_root).resolve()
    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    case_to_label = raw_mapping.get("case_to_material", raw_mapping)
    class_to_id = raw_mapping.get("class_to_id", {})
    if not class_to_id:
        labels = sorted(set(case_to_label.values()))
        class_to_id = {label: index for index, label in enumerate(labels)}
    selected_mapping: dict[str, str | int] = {}
    case_records: list[dict[str, object]] = []

    import torch

    for case in case_names:
        if case not in case_to_label:
            raise ValueError(f"MatPhys material mapping omits {case}")
        label = case_to_label[case]
        material_id = int(class_to_id[label]) if isinstance(label, str) else int(label)
        if not 0 <= material_id < num_materials:
            raise ValueError(f"{case}: material id exceeds decoder dimensions")
        final_data_path = root / case / "final_data.pkl"
        data = _load_pickle(final_data_path)
        structure_points = np.concatenate(
            (
                np.asarray(data["object_points"])[0],
                np.asarray(data["surface_points"]),
                np.asarray(data["interior_points"]),
            ),
            axis=0,
        ).astype(np.float32)
        node_sem_path = destination / "semantic_cache" / f"{case}_node_sem.npz"
        node_sem_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            node_sem_path,
            node_sem=np.zeros(
                (len(structure_points), semantic_dimension), dtype=np.float32
            ),
        )
        material_distribution = torch.zeros(
            (1, num_materials), dtype=torch.float32
        )
        material_distribution[0, material_id] = 1.0
        train_ready_path = destination / "results" / case / "train" / "train_ready.pt"
        train_ready_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "part_assignments": torch.zeros(
                    len(structure_points), dtype=torch.long
                ),
                "material_distributions": material_distribution,
                "part_features": torch.zeros(
                    (1, semantic_dimension), dtype=torch.float32
                ),
                "xyz": torch.from_numpy(structure_points),
                "proxy_contract": "global-onehot-single-part-v1",
            },
            train_ready_path,
        )
        selected_mapping[case] = label
        case_records.append(
            {
                "name": case,
                "material_label": label,
                "material_id": material_id,
                "structure_point_count": len(structure_points),
                "node_sem": {
                    "path": str(node_sem_path),
                    "sha256": sha256_file(node_sem_path),
                },
                "train_ready": {
                    "path": str(train_ready_path),
                    "sha256": sha256_file(train_ready_path),
                },
            }
        )

    proxy_mapping_path = destination / "case_to_material.json"
    proxy_mapping_path.write_text(
        json.dumps(
            {
                "class_to_id": class_to_id,
                "case_to_material": selected_mapping,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "contract": "global-onehot-single-part-v1",
        "claim_boundary": (
            "Proxy for unpublished MatPhys preprocessing artifacts; not a "
            "part-level MatPhys reproduction."
        ),
        "source_mapping": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
        },
        "mapping": {
            "path": str(proxy_mapping_path),
            "sha256": sha256_file(proxy_mapping_path),
        },
        "semantic_cache_dir": str(destination / "semantic_cache"),
        "results_dir": str(destination / "results"),
        "cases": case_records,
    }
    summary_path = destination / "proxy_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def write_causal_training_audit(
    checkpoint_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    source_repository: str,
    source_commit: str,
    data_root: str | Path,
    accessed_frame_indices: Mapping[str, Sequence[int]],
    objective_end_frames_exclusive: Mapping[str, int],
    split_by_case: Mapping[str, Mapping[str, object]],
    proxy_summary_path: str | Path,
    parameterization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind checkpoint bytes to an observed-frame access log."""

    if not source_repository or not source_commit:
        raise ValueError("source repository and commit are required")
    cases: list[dict[str, object]] = []
    for case, raw_indices in sorted(accessed_frame_indices.items()):
        if case not in split_by_case:
            raise ValueError(f"missing split for accessed case {case}")
        train_end = int(split_by_case[case]["train"][1])
        if case not in objective_end_frames_exclusive:
            raise ValueError(f"missing objective boundary for accessed case {case}")
        objective_end = int(objective_end_frames_exclusive[case])
        if not 1 <= objective_end <= train_end:
            raise ValueError(f"{case}: objective accessed a future frame")
        indices = sorted(set(int(index) for index in raw_indices))
        if not indices:
            raise ValueError(f"no video frames were recorded for {case}")
        if indices[0] < 0 or indices[-1] >= train_end:
            raise ValueError(f"{case}: checkpoint training accessed a future frame")
        cases.append(
            {
                "name": case,
                "train_end_frame_exclusive": train_end,
                "accessed_frame_indices": indices,
                "maximum_accessed_frame": indices[-1],
                "objective_frame_interval": [1, objective_end],
                "maximum_objective_frame": objective_end - 1,
            }
        )
    checkpoints = []
    for raw_path in checkpoint_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"path": str(path), "sha256": sha256_file(path)})
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    proxy_path = Path(proxy_summary_path).resolve()
    audit = {
        "schema_version": MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION,
        "future_observations_used": False,
        "video_sampling": "uniform-within-released-training-prefix-v1",
        "optimization_and_checkpoint_selection": "released-prefix-only-v1",
        "checkpoint_policy": "fixed-terminal-epoch-v1",
        "fit_all_frames": False,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "data_root": str(Path(data_root).resolve()),
        "proxy": {"path": str(proxy_path), "sha256": sha256_file(proxy_path)},
        "checkpoints": checkpoints,
        "cases": cases,
    }
    if parameterization is not None:
        audit["parameterization"] = dict(parameterization)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit["audit_path"] = str(destination.resolve())
    audit["audit_sha256"] = sha256_file(destination)
    return audit


def validate_causal_training_audit(
    audit_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Require a future-blind audit that binds the exact exported checkpoint."""

    source = Path(audit_path).resolve()
    audit = json.loads(source.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MATPHYS_CAUSAL_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported MatPhys causal-audit schema")
    if audit.get("future_observations_used") is not False:
        raise ValueError("MatPhys checkpoint audit does not forbid future observations")
    if audit.get("fit_all_frames") is not False:
        raise ValueError("MatPhys checkpoint audit permits fit_all_frames")
    if (
        audit.get("optimization_and_checkpoint_selection")
        != "released-prefix-only-v1"
    ):
        raise ValueError("MatPhys checkpoint audit does not bind objective access")
    if audit.get("checkpoint_policy") != "fixed-terminal-epoch-v1":
        raise ValueError("MatPhys checkpoint audit does not use the terminal epoch")
    checkpoint = Path(checkpoint_path).resolve()
    identity = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    if identity not in audit.get("checkpoints", []):
        raise ValueError("checkpoint bytes are not bound by the causal training audit")
    for case in audit.get("cases", []):
        train_end = int(case["train_end_frame_exclusive"])
        indices = [int(index) for index in case["accessed_frame_indices"]]
        if not indices or min(indices) < 0 or max(indices) >= train_end:
            raise ValueError(f"{case.get('name')}: audit contains future video access")
        objective_start, objective_end = (
            int(value) for value in case["objective_frame_interval"]
        )
        if objective_start != 1 or not objective_start <= objective_end <= train_end:
            raise ValueError(
                f"{case.get('name')}: audit contains future objective access"
            )
        if int(case["maximum_objective_frame"]) != objective_end - 1:
            raise ValueError(f"{case.get('name')}: objective audit is inconsistent")
    parameterization = audit.get("parameterization")
    if parameterization is not None:
        if not isinstance(parameterization, dict):
            raise ValueError("MatPhys parameterization audit must be an object")
        from .matphys_teacher_residual import (
            TEACHER_PARAMETERIZATION,
            validate_matphys_teacher_manifest,
        )

        if parameterization.get("name") != TEACHER_PARAMETERIZATION:
            raise ValueError("unsupported MatPhys parameterization audit")
        scale = float(parameterization.get("residual_log_scale", -1.0))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("invalid teacher residual scale")
        teacher = parameterization.get("teacher")
        if not isinstance(teacher, dict):
            raise ValueError("teacher parameterization omits its source manifest")
        validate_matphys_teacher_manifest(teacher)
    return {
        **audit,
        "audit_path": str(source),
        "audit_sha256": sha256_file(source),
    }


def merge_matphys_external_manifests(
    manifest_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    case_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Merge per-case MatPhys exports without erasing their provenance."""

    if not manifest_paths:
        raise ValueError("at least one external-backbone manifest is required")
    components: list[dict[str, object]] = []
    cases_by_name: dict[str, dict[str, object]] = {}
    shared_backbone: dict[str, object] | None = None
    for raw_path in manifest_paths:
        source = Path(raw_path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"{source}: unsupported external manifest schema")
        backbone = payload.get("backbone")
        raw_cases = payload.get("cases")
        if not isinstance(backbone, dict) or not isinstance(raw_cases, list):
            raise ValueError(f"{source}: malformed external manifest")
        identity = {field: backbone.get(field) for field in _EXTERNAL_BACKBONE_SHARED_FIELDS}
        if shared_backbone is None:
            shared_backbone = identity
        elif identity != shared_backbone:
            raise ValueError("component manifests describe incompatible backbones")
        component_cases: list[str] = []
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError(f"{source}: malformed case entry")
            name = str(raw_case.get("name", ""))
            if not name or name in cases_by_name:
                raise ValueError("component case names must be nonempty and unique")
            trajectory = Path(str(raw_case.get("trajectory", "")))
            if not trajectory.is_absolute():
                trajectory = source.parent / trajectory
            normalized = {**raw_case, "trajectory": str(trajectory.resolve())}
            cases_by_name[name] = normalized
            component_cases.append(name)
        components.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "cases": component_cases,
                "checkpoint": backbone.get("checkpoint"),
                "causal_training_audit": backbone.get("causal_training_audit"),
            }
        )

    requested_order = (
        tuple(str(case) for case in case_order)
        if case_order is not None
        else tuple(cases_by_name)
    )
    if len(requested_order) != len(set(requested_order)):
        raise ValueError("case order must not contain duplicates")
    if set(requested_order) != set(cases_by_name):
        raise ValueError("case order must contain every component case exactly once")
    assert shared_backbone is not None
    merged = {
        "schema_version": 1,
        "backbone": {
            **shared_backbone,
            "training_scope": "independent-per-case-fixed-terminal-v1",
            "component_manifests": components,
        },
        "cases": [cases_by_name[name] for name in requested_order],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **merged,
        "manifest_path": str(destination),
        "manifest_sha256": sha256_file(destination),
    }
