"""Validation helpers for object-disjoint MatPhys evaluation protocols."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .matphys_causal_bridge import (
    sha256_file,
    validate_matphys_fresh_fold_initialization,
)
from .matphys_graph_parts import materialize_compact_graph_proxy_subset
from .phystwin_comparison import phystwin_physical_object_cluster
from .phystwin_sota_comparison import PHYSTWIN_TABLE1_CASES


def load_matphys_loo_protocol(path: str | Path) -> dict[str, object]:
    """Load and validate one exhaustive leave-one-physical-object-out protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported MatPhys LOO protocol schema")
    case_order = tuple(str(case) for case in payload.get("case_order", ()))
    if case_order != PHYSTWIN_TABLE1_CASES:
        raise ValueError("MatPhys LOO case order differs from the official cohort")
    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("MatPhys LOO protocol contains no folds")
    training = payload.get("source_training")
    if not isinstance(training, Mapping):
        raise ValueError("MatPhys LOO protocol omits source training")
    initialization = training.get("initialization")
    legacy_checkpoint_hash = training.get("initialization_checkpoint_sha256")
    if initialization is None:
        if not isinstance(legacy_checkpoint_hash, str) or not legacy_checkpoint_hash:
            raise ValueError("MatPhys LOO protocol omits its initialization contract")
    else:
        try:
            clean_initialization = validate_matphys_fresh_fold_initialization(
                initialization
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "object-disjoint training has an invalid fresh initialization"
            ) from error
        random_seed = training.get("random_seed")
        if clean_initialization["random_seed"] != random_seed:
            raise ValueError("fresh-fold initialization seed differs from training")
        if legacy_checkpoint_hash is not None:
            raise ValueError("fresh-fold training must not declare a checkpoint hash")

    target_to_object: dict[str, str] = {}
    fold_objects: set[str] = set()
    normalized_folds: list[dict[str, object]] = []
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, Mapping):
            raise ValueError("MatPhys LOO fold must be an object")
        object_name = str(raw_fold.get("object", ""))
        targets = tuple(str(case) for case in raw_fold.get("targets", ()))
        if not object_name or object_name in fold_objects:
            raise ValueError("MatPhys LOO fold objects must be nonempty and unique")
        if not targets or len(targets) != len(set(targets)):
            raise ValueError("MatPhys LOO fold targets must be nonempty and unique")
        for case in targets:
            if case in target_to_object:
                raise ValueError(f"MatPhys LOO target appears twice: {case}")
            if phystwin_physical_object_cluster(case) != object_name:
                raise ValueError(
                    f"MatPhys LOO target is in the wrong object fold: {case}"
                )
            target_to_object[case] = object_name
        fold_objects.add(object_name)
        sources = tuple(case for case in case_order if case not in targets)
        if any(
            phystwin_physical_object_cluster(case) == object_name for case in sources
        ):
            raise ValueError(f"MatPhys LOO source leaks target object: {object_name}")
        normalized_folds.append(
            {"object": object_name, "targets": targets, "sources": sources}
        )

    if tuple(case for case in case_order if case in target_to_object) != case_order:
        missing = sorted(set(case_order) - set(target_to_object))
        raise ValueError(f"MatPhys LOO folds do not cover the cohort: {missing}")
    result = dict(payload)
    result["folds"] = normalized_folds
    result["protocol_path"] = str(source)
    return result


def _write_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing locked MatPhys LOO artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def prepare_matphys_loo_workspace(
    protocol_path: str | Path,
    proxy_summary_paths: Sequence[str | Path],
    output_root: str | Path,
) -> dict[str, object]:
    """Materialize byte-bound source and target proxies for every LOO fold."""

    protocol = load_matphys_loo_protocol(protocol_path)
    destination = Path(output_root).resolve()
    protocol_source = Path(str(protocol["protocol_path"]))
    protocol_identity = {
        "path": str(protocol_source),
        "sha256": sha256_file(protocol_source),
    }
    fold_records = []
    for index, fold in enumerate(protocol["folds"]):
        object_name = str(fold["object"])
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", object_name).strip("_")
        if not slug:
            raise ValueError(f"invalid MatPhys LOO object name: {object_name}")
        fold_root = destination / f"fold_{index:02d}_{slug}"
        sources = tuple(str(case) for case in fold["sources"])
        targets = tuple(str(case) for case in fold["targets"])
        source_proxy = materialize_compact_graph_proxy_subset(
            proxy_summary_paths,
            fold_root / "source_proxy",
            sources,
        )
        target_proxy = materialize_compact_graph_proxy_subset(
            proxy_summary_paths,
            fold_root / "target_proxy",
            targets,
        )
        registration_path = fold_root / "registered_split.json"
        _write_exact_json(
            registration_path,
            {
                "schema_version": 1,
                "contract": "object-disjoint-matphys-source-target-split-v1",
                "protocol": protocol_identity,
                "fold_index": index,
                "held_out_object": object_name,
                "source_cases": list(sources),
                "target_cases": list(targets),
                "future_opened": False,
            },
        )
        source_summary = Path(str(source_proxy["summary_path"]))
        target_summary = Path(str(target_proxy["summary_path"]))
        fold_record = {
            "fold_index": index,
            "held_out_object": object_name,
            "root": str(fold_root),
            "source_cases": list(sources),
            "target_cases": list(targets),
            "registration": {
                "path": str(registration_path),
                "sha256": sha256_file(registration_path),
            },
            "source_proxy": {
                "path": str(source_summary),
                "sha256": sha256_file(source_summary),
            },
            "target_proxy": {
                "path": str(target_summary),
                "sha256": sha256_file(target_summary),
            },
        }
        _write_exact_json(fold_root / "fold_manifest.json", fold_record)
        fold_records.append(fold_record)
    manifest = {
        "schema_version": 1,
        "contract": "matphys-object-disjoint-loo-workspace-v1",
        "claim_boundary": str(protocol["claim_boundary"]),
        "protocol": protocol_identity,
        "future_opened": False,
        "folds": fold_records,
    }
    manifest_path = destination / "loo_workspace_manifest.json"
    _write_exact_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
