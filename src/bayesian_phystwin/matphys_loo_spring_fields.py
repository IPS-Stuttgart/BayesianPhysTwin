"""Portable spring-field artifacts for object-disjoint MatPhys folds."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .matphys_causal_bridge import (
    sha256_file,
    validate_matphys_fresh_fold_initialization,
    validate_source_supervised_training_audit,
)


LOO_WORKSPACE_CONTRACT = "matphys-object-disjoint-loo-workspace-v1"
LOO_SPRING_FIELDS_CONTRACT = "matphys-object-disjoint-loo-spring-fields-v1"
_SHARED_BACKBONE_FIELDS = (
    "source_repository",
    "source_commit",
    "coordinate_frame",
    "vertex_contract",
    "proxy_contract",
    "claim_boundary",
)


def _resolve_path(base: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base.parent / path
    return path.resolve()


def _validated_identity(base: Path, value: object, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a file identity")
    path = _resolve_path(base, value.get("path"))
    expected = str(value.get("sha256", ""))
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"{label} bytes changed")
    return path


def _copy_exact(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise RuntimeError(f"portable artifact differs at {destination}")
    else:
        shutil.copy2(source, destination)
    return {"path": str(destination), "sha256": digest}


def _relative_identity(path: Path, manifest_path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(manifest_path.parent)),
        "sha256": sha256_file(path),
    }


def _candidate_identity(
    export_path: Path,
    case_entry: dict[str, Any],
    case: str,
) -> tuple[Path, dict[str, Any]]:
    summary = case_entry.get("spring_field_summary")
    complete = summary.get("complete_spring_y") if isinstance(summary, dict) else None
    if not isinstance(complete, dict):
        raise ValueError(f"{case}: export omits complete_spring_y")
    path = _validated_identity(export_path, complete, f"{case}.complete_spring_y")
    count = int(complete.get("count", -1))
    if count < 1:
        raise ValueError(f"{case}: invalid spring-field count")
    return path, complete


def collect_loo_spring_fields(
    workspace_manifest: str | Path,
    output_dir: str | Path,
    *,
    fold_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Validate completed folds and copy their fields into a portable bundle."""

    workspace_path = Path(workspace_manifest).resolve()
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    if workspace.get("contract") != LOO_WORKSPACE_CONTRACT:
        raise ValueError("unsupported MatPhys LOO workspace")
    if workspace.get("future_opened") is not False:
        raise ValueError("MatPhys LOO workspace has opened future metrics")
    protocol_path = _validated_identity(
        workspace_path, workspace.get("protocol"), "protocol"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    case_order = tuple(str(case) for case in protocol["case_order"])
    selected = None if fold_indices is None else {int(index) for index in fold_indices}
    if selected is not None and len(selected) != len(tuple(fold_indices or ())):
        raise ValueError("fold indices must be unique")

    output = Path(output_dir).resolve()
    manifest_path = output / "loo_spring_fields.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    copied_protocol = output / "provenance" / "protocol.json"
    _copy_exact(protocol_path, copied_protocol)
    records: dict[str, dict[str, object]] = {}
    shared_backbone: dict[str, object] | None = None
    collected_folds: list[int] = []
    available_folds = {int(fold["fold_index"]) for fold in workspace["folds"]}
    if selected is not None and not selected <= available_folds:
        raise ValueError("requested fold is absent from the workspace")

    for fold in workspace["folds"]:
        index = int(fold["fold_index"])
        if selected is not None and index not in selected:
            continue
        fold_root = Path(str(fold["root"])).resolve()
        export_path = fold_root / "matphys_export" / "external_backbone_manifest.json"
        if not export_path.is_file():
            raise FileNotFoundError(export_path)
        export = json.loads(export_path.read_text(encoding="utf-8"))
        backbone = export.get("backbone")
        raw_cases = export.get("cases")
        if export.get("schema_version") != 1 or not isinstance(backbone, dict):
            raise ValueError(f"fold {index}: malformed export manifest")
        if backbone.get("future_observations_used") is not False:
            raise ValueError(f"fold {index}: export used future observations")
        if not isinstance(raw_cases, list):
            raise ValueError(f"fold {index}: export cases must be a list")
        names = [str(entry.get("name", "")) for entry in raw_cases]
        expected_names = [str(case) for case in fold["target_cases"]]
        if names != expected_names:
            raise ValueError(f"fold {index}: target cases changed")
        identity = {field: backbone.get(field) for field in _SHARED_BACKBONE_FIELDS}
        if shared_backbone is None:
            shared_backbone = identity
        elif identity != shared_backbone:
            raise ValueError("LOO folds describe incompatible MatPhys backbones")

        checkpoint = _validated_identity(
            export_path, backbone.get("checkpoint"), "checkpoint"
        )
        audit = _validated_identity(
            export_path,
            backbone.get("causal_training_audit"),
            "source-supervised training audit",
        )
        validated_audit = validate_source_supervised_training_audit(audit, checkpoint)
        expected_initialization = protocol.get("source_training", {}).get(
            "initialization"
        )
        if expected_initialization is not None:
            actual_initialization = validated_audit.get("parameterization", {}).get(
                "initialization"
            )
            try:
                expected_clean = validate_matphys_fresh_fold_initialization(
                    expected_initialization
                )
                actual_clean = validate_matphys_fresh_fold_initialization(
                    actual_initialization
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"fold {index}: training initialization is not clean"
                ) from error
            if actual_clean != expected_clean:
                raise ValueError(f"fold {index}: training initialization differs")
        provenance_root = output / "provenance" / f"fold_{index:02d}"
        copied_export = provenance_root / "external_backbone_manifest.json"
        copied_audit = provenance_root / "source_supervised_training_audit.json"
        _copy_exact(export_path, copied_export)
        _copy_exact(audit, copied_audit)

        for raw_entry in raw_cases:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"fold {index}: malformed case entry")
            case = str(raw_entry["name"])
            if case in records:
                raise ValueError(f"duplicate held-out case {case}")
            source_field, complete = _candidate_identity(export_path, raw_entry, case)
            copied_field = output / "cases" / case / "candidate_spring_y.npy"
            _copy_exact(source_field, copied_field)
            evidence_end = int(raw_entry.get("evidence_end_frame_exclusive", -1))
            if evidence_end < 1:
                raise ValueError(f"{case}: invalid evidence boundary")
            records[case] = {
                "name": case,
                "fold_index": index,
                "held_out_object": str(fold["held_out_object"]),
                "evidence_end_frame_exclusive": evidence_end,
                "candidate_spring_y": {
                    **_relative_identity(copied_field, manifest_path),
                    "count": int(complete["count"]),
                },
                "source_checkpoint": {
                    "path_at_collection": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                },
                "source_training_audit": _relative_identity(
                    copied_audit, manifest_path
                ),
                "source_export_manifest": _relative_identity(
                    copied_export, manifest_path
                ),
            }
        collected_folds.append(index)

    if not records:
        raise ValueError("no completed folds were selected")
    assert shared_backbone is not None
    ordered_cases = [records[case] for case in case_order if case in records]
    payload = {
        "schema_version": 1,
        "contract": LOO_SPRING_FIELDS_CONTRACT,
        "future_observations_used": False,
        "complete_cohort": len(ordered_cases) == len(case_order),
        "protocol": _relative_identity(copied_protocol, manifest_path),
        "protocol_id": protocol["protocol_id"],
        "case_order": [record["name"] for record in ordered_cases],
        "fold_indices": collected_folds,
        "backbone": shared_backbone,
        "cases": ordered_cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_loo_spring_fields(
    manifest: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Validate a portable spring-field manifest and resolve field paths."""

    source = Path(manifest).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("contract") != LOO_SPRING_FIELDS_CONTRACT:
        raise ValueError("unsupported LOO spring-field contract")
    if payload.get("future_observations_used") is not False:
        raise ValueError("LOO spring fields must be future blind")
    _validated_identity(source, payload.get("protocol"), "protocol")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("LOO spring-field manifest has no cases")
    names = [str(entry.get("name", "")) for entry in raw_cases]
    if names != [str(name) for name in payload.get("case_order", ())]:
        raise ValueError("LOO spring-field case order changed")
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("LOO spring-field cases must be nonempty and unique")
    if require_complete and payload.get("complete_cohort") is not True:
        raise ValueError("LOO spring-field cohort is incomplete")
    resolved_cases = []
    for entry in raw_cases:
        field = _validated_identity(
            source,
            entry.get("candidate_spring_y"),
            f"{entry['name']}.candidate_spring_y",
        )
        resolved_cases.append({**entry, "candidate_spring_y_path": str(field)})
    return {
        **payload,
        "cases": resolved_cases,
        "manifest": {"path": str(source), "sha256": sha256_file(source)},
    }


def merge_loo_spring_field_bundles(
    manifests: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    """Merge disjoint host bundles into one canonical full-cohort artifact."""

    if not manifests:
        raise ValueError("at least one spring-field bundle is required")
    validated = [
        validate_loo_spring_fields(path, require_complete=False) for path in manifests
    ]
    first = validated[0]
    protocol_hash = str(first["protocol"]["sha256"])
    protocol_id = str(first["protocol_id"])
    backbone = first["backbone"]
    all_cases: dict[str, dict[str, object]] = {}
    for bundle in validated:
        if str(bundle["protocol"]["sha256"]) != protocol_hash:
            raise ValueError("spring-field bundles use different protocols")
        if bundle["protocol_id"] != protocol_id or bundle["backbone"] != backbone:
            raise ValueError("spring-field bundles describe incompatible experiments")
        for entry in bundle["cases"]:
            name = str(entry["name"])
            if name in all_cases:
                raise ValueError(f"duplicate spring field for {name}")
            all_cases[name] = entry

    first_manifest = Path(str(first["manifest"]["path"]))
    first_protocol = _validated_identity(first_manifest, first["protocol"], "protocol")
    protocol = json.loads(first_protocol.read_text(encoding="utf-8"))
    canonical_order = [str(case) for case in protocol["case_order"]]
    if set(all_cases) != set(canonical_order):
        missing = sorted(set(canonical_order) - set(all_cases))
        extra = sorted(set(all_cases) - set(canonical_order))
        raise ValueError(f"merged cohort mismatch; missing={missing}, extra={extra}")

    output = Path(output_dir).resolve()
    manifest_path = output / "loo_spring_fields.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    copied_protocol = output / "provenance" / "protocol.json"
    _copy_exact(first_protocol, copied_protocol)
    components = []
    component_roots: dict[str, Path] = {}
    source_lookup = {
        str(entry["name"]): (bundle, entry)
        for bundle in validated
        for entry in bundle["cases"]
    }
    for index, bundle in enumerate(validated):
        source_manifest = Path(str(bundle["manifest"]["path"]))
        component_root = output / "provenance" / "components" / f"component_{index:02d}"
        if component_root.exists():
            raise FileExistsError(component_root)
        component_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_manifest.parent, component_root)
        copied = component_root / source_manifest.name
        if sha256_file(copied) != sha256_file(source_manifest):
            raise RuntimeError("component bundle changed while being copied")
        component_roots[str(source_manifest)] = component_root
        components.append(
            {
                **_relative_identity(copied, manifest_path),
                "cases": [str(entry["name"]) for entry in bundle["cases"]],
            }
        )

    merged_cases = []
    for case in canonical_order:
        bundle, entry = source_lookup[case]
        source_manifest = Path(str(bundle["manifest"]["path"]))
        source_field = _validated_identity(
            source_manifest,
            entry["candidate_spring_y"],
            f"{case}.candidate_spring_y",
        )
        copied_field = output / "cases" / case / "candidate_spring_y.npy"
        _copy_exact(source_field, copied_field)
        component_root = component_roots[str(source_manifest)]
        source_audit = _validated_identity(
            source_manifest,
            entry["source_training_audit"],
            f"{case}.source_training_audit",
        )
        source_export = _validated_identity(
            source_manifest,
            entry["source_export_manifest"],
            f"{case}.source_export_manifest",
        )
        copied_audit = component_root / source_audit.relative_to(source_manifest.parent)
        copied_export = component_root / source_export.relative_to(
            source_manifest.parent
        )
        merged_cases.append(
            {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "candidate_spring_y",
                    "candidate_spring_y_path",
                    "source_training_audit",
                    "source_export_manifest",
                }
            }
            | {
                "candidate_spring_y": {
                    **_relative_identity(copied_field, manifest_path),
                    "count": int(entry["candidate_spring_y"]["count"]),
                },
                "source_training_audit": _relative_identity(
                    copied_audit, manifest_path
                ),
                "source_export_manifest": _relative_identity(
                    copied_export, manifest_path
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "contract": LOO_SPRING_FIELDS_CONTRACT,
        "future_observations_used": False,
        "complete_cohort": True,
        "protocol": _relative_identity(copied_protocol, manifest_path),
        "protocol_id": protocol_id,
        "case_order": canonical_order,
        "fold_indices": sorted({int(entry["fold_index"]) for entry in merged_cases}),
        "backbone": backbone,
        "components": components,
        "cases": merged_cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_loo_spring_fields(manifest_path)
    return {
        **payload,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
