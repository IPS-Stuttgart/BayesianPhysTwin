#!/usr/bin/env python3
"""Freeze the hash-only 78-take PokeFlex competence protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin_experiments.pokeflex_query_competence_v1 import (
    BOOTSTRAP_REPLICATES,
    CLAIM_BOUNDARY,
    CONTEXT_FEATURES,
    FAILED_ARCHIVE_SHA256,
    FAILED_ATTEMPT_FILE_SHA256,
    FAILED_IMPLEMENTATION_COMMIT,
    FAILED_PARENT_PUBLIC78_PROTOCOL_SHA256,
    FAILED_PROTOCOL_FILE_SHA256,
    FAILED_PROTOCOL_ID,
    FAILED_PROTOCOL_SHA256,
    FAILURE_RECEIPT_PATH,
    FORBIDDEN_BOUNDARIES,
    HARM_MARGIN_RELATIVE,
    IMPLEMENTATION_MODULE_PATH,
    MINIMUM_ACCEPTED_OBJECTS,
    MINIMUM_OBJECT_BALANCED_COVERAGE,
    PARENT_PUBLIC78_PROTOCOL_SHA256,
    PRIMARY_FEATURES,
    PROTOCOL_ID,
    SCHEMA,
    SCHEMA_VERSION,
    SOURCE_ARTIFACT_ROLE,
    SPLIT_NAMESPACE,
    TARGET_HARM_PROBABILITY,
    THRESHOLD_GRID,
    _canonical_json_sha256,
    deterministic_split_v1,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    args = parser.parse_args()

    if len(args.implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.implementation_commit
    ):
        raise ValueError("implementation commit must be a lowercase full SHA-1")
    repository_root = Path(__file__).resolve().parents[2]
    implementation_module = repository_root / IMPLEMENTATION_MODULE_PATH
    if not args.execution_root.is_absolute():
        raise ValueError("execution root must be absolute")
    commit_prefix = args.implementation_commit[:8]

    paths = sorted(args.artifact_root.glob("*.json"))
    if len(paths) != 78:
        raise ValueError("expected exactly 78 retained PokeFlex artifacts")
    inventory = {
        path.stem: {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    }
    split = deterministic_split_v1(inventory)
    for take_id in split["risk_train"] + split["threshold_select"]:
        artifact = json.loads((args.artifact_root / f"{take_id}.json").read_text())
        if artifact.get("public_transfer_protocol_sha256") != (
            PARENT_PUBLIC78_PROTOCOL_SHA256
        ):
            raise ValueError("source metadata parent protocol changed")
        if artifact.get("retrospective_prediction_role") != SOURCE_ARTIFACT_ROLE:
            raise ValueError("source metadata retrospective role changed")
        if artifact.get("future_observation_used") is not False:
            raise ValueError("source metadata future-observation boundary changed")
    failure_receipt = repository_root / FAILURE_RECEIPT_PATH
    protocol: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": args.locked_at_utc,
        "claim_boundary": CLAIM_BOUNDARY,
        "parent_public78_protocol_sha256": PARENT_PUBLIC78_PROTOCOL_SHA256,
        "source_artifact_role": SOURCE_ARTIFACT_ROLE,
        "implementation": {
            "git_commit": args.implementation_commit,
            "module_path": IMPLEMENTATION_MODULE_PATH,
            "module_sha256": file_sha256(implementation_module),
        },
        "execution": {
            "root": str(args.execution_root),
            "source_attempt_filename": f"source-attempt-{commit_prefix}.json",
            "source_result_filename": f"source-result-{commit_prefix}.json",
            "validation_attempt_filename": (f"validation-attempt-{commit_prefix}.json"),
            "validation_result_filename": f"validation-result-{commit_prefix}.json",
        },
        "artifact_inventory": inventory,
        "split": {name: list(values) for name, values in split.items()},
        "method": {
            "split_namespace": SPLIT_NAMESPACE,
            "primary_arm": "model_disagreement_only",
            "primary_feature_names": list(PRIMARY_FEATURES),
            "context_feature_names": list(CONTEXT_FEATURES),
            "candidate": (
                "frozen object-scale causal action-local correction from the "
                "parent public78 protocol"
            ),
            "fallback": "byte-identical released checkpoint prediction",
            "harm_margin_relative": HARM_MARGIN_RELATIVE,
            "threshold_grid": list(THRESHOLD_GRID),
            "risk_fit": "deterministic object-balanced L2 logistic regression",
            "uncertainty": (
                f"{BOOTSTRAP_REPLICATES} replicate physical-object cluster bootstrap"
            ),
        },
        "gates": {
            "target_harm_probability": TARGET_HARM_PROBABILITY,
            "minimum_object_balanced_coverage": (MINIMUM_OBJECT_BALANCED_COVERAGE),
            "minimum_accepted_objects": MINIMUM_ACCEPTED_OBJECTS,
            "harm_upper_bound": "object-cluster percentile 95% upper <= 0.10",
            "policy_regret": "object-cluster percentile 95% upper < 0",
            "source_requirement": (
                "primary arm must pass on the 18 threshold-selection takes before "
                "any of the 42 validation artifacts are opened"
            ),
        },
        "forbidden": list(FORBIDDEN_BOUNDARIES),
        "replacement": {
            "kind": "source-independent pre-analysis metadata correction",
            "failed_protocol_id": FAILED_PROTOCOL_ID,
            "failed_protocol_sha256": FAILED_PROTOCOL_SHA256,
            "failed_protocol_file_sha256": FAILED_PROTOCOL_FILE_SHA256,
            "failed_implementation_commit": FAILED_IMPLEMENTATION_COMMIT,
            "failed_archive_sha256": FAILED_ARCHIVE_SHA256,
            "failed_attempt_file_sha256": FAILED_ATTEMPT_FILE_SHA256,
            "failed_stage": "source",
            "failed_source_result_written": False,
            "failed_validation_artifacts_opened": 0,
            "scientific_quantities_computed": False,
            "failure_receipt_path": FAILURE_RECEIPT_PATH,
            "failure_receipt_file_sha256": file_sha256(failure_receipt),
            "correction": {
                "field": "parent_public78_protocol_sha256",
                "from": FAILED_PARENT_PUBLIC78_PROTOCOL_SHA256,
                "to": PARENT_PUBLIC78_PROTOCOL_SHA256,
            },
            "invariants": [
                "artifact inventory",
                "18/18/42 take split",
                "split namespace",
                "candidate and exact fallback",
                "features and risk fit",
                "threshold grid",
                "bootstrap and gates",
                "claim boundary",
            ],
            "authorization": (
                "exactly one replacement source attempt in a new execution root; "
                "no further replacement or retry"
            ),
        },
    }
    protocol["protocol_sha256"] = _canonical_json_sha256(protocol)
    rendered = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError("existing PokeFlex competence protocol differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_count": len(inventory),
                "output": str(args.output.resolve()),
                "protocol_sha256": protocol["protocol_sha256"],
                "split_counts": {name: len(values) for name, values in split.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
