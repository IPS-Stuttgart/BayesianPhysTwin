#!/usr/bin/env python3
"""Reproduce frozen Deform360 means and cross-fit physical-query covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import deform360_exact_confirmation_capture_v7 as exact_capture
import deform360_query_covariance_v7 as query_cov
import numpy as np

SCHEMA = "bayesian-phystwin/deform360-exact-same-mean-query-covariance-v7"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_exact_export(
    artifact_root: Path,
    confirmation_root: Path,
    frozen_root: Path,
    data_root: Path,
    output_root: Path,
    maximum_queries: int,
    fold_count: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reproduction = exact_capture.reproduce_exact_confirmation(
        artifact_root,
        confirmation_root,
        frozen_root,
        data_root,
    )
    reference = reproduction["reference"]
    reproduced = reproduction["reproduced"]
    captures = reproduction["captures"]
    dimensions = {int(row["pooled_field_dimension"]) for row in reproduced["objects"]}
    if len(dimensions) != 1:
        raise ValueError("pooled tactile field dimension is not constant")
    field_dimension = dimensions.pop()
    query = query_cov.cosine_query_matrix(field_dimension, maximum_queries)
    residual_blocks: list[np.ndarray] = []
    covariance_blocks: list[np.ndarray] = []
    object_groups: list[str] = []
    target_episode_ids: list[int] = []
    window_indices: list[int] = []
    object_records: list[dict[str, Any]] = []
    for row, captured in zip(reproduced["objects"], captures, strict=True):
        errors = captured["errors"]
        if errors.shape != (
            int(row["forecast_window_count"]),
            int(row["pooled_field_dimension"]),
        ):
            raise ValueError(
                f"captured target residual shape changed for {row['object_id']}"
            )
        projected_residuals = errors @ query.T
        projected_covariance = query_cov.project_compact_covariance(
            captured["diagonal"],
            captured["factor"],
            captured["multiplier"],
            query,
        )
        window_count = errors.shape[0]
        residual_blocks.append(projected_residuals)
        covariance_blocks.append(
            np.broadcast_to(
                projected_covariance,
                (window_count, *projected_covariance.shape),
            ).copy()
        )
        object_groups.extend([str(row["object_id"])] * window_count)
        target_episode_ids.extend([int(row["target_episode_id"])] * window_count)
        window_indices.extend(range(window_count))
        object_records.append(
            {
                "object_id": str(row["object_id"]),
                "target_episode_id": int(row["target_episode_id"]),
                "forecast_window_count": window_count,
                "field_dimension": field_dimension,
                "query_dimension": int(query.shape[0]),
                "source_covariance_rank": int(captured["factor"].shape[1]),
                "source_covariance_multiplier": captured["multiplier"],
                "target_full_field_joint_nanees": float(
                    row["uncertainty"]["joint_nanees"]
                ),
            }
        )
    residuals = np.concatenate(residual_blocks)
    covariances = np.concatenate(covariance_blocks)
    groups = np.asarray(object_groups)
    target_ids = np.asarray(target_episode_ids, dtype=np.int32)
    windows = np.asarray(window_indices, dtype=np.int32)
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(covariances)):
        raise ValueError("exported query sufficient statistics are nonfinite")

    statistics_path = output_root / "query-sufficient-statistics-v7.npz"
    np.savez_compressed(
        statistics_path,
        query_residuals=residuals,
        query_covariances=covariances,
        object_ids=groups,
        target_episode_ids=target_ids,
        window_indices=windows,
        query_matrix=query,
    )
    study_result = query_cov.study(residuals, covariances, groups, fold_count)
    statistics_manifest = {
        "schema": "bayesian-phystwin/deform360-query-sufficient-statistics-v7",
        "statistics_file": statistics_path.name,
        "statistics_sha256": sha256_file(statistics_path),
        "case_count": int(residuals.shape[0]),
        "object_count": len(set(groups.tolist())),
        "field_dimension": field_dimension,
        "query_dimension": int(query.shape[0]),
        "query_family": "first normalized one-dimensional DCT-II pooled-field modes",
        "query_matrix_sha256": query_cov.canonical_digest(query.tolist()),
        "common_origin_translation": {
            "truth_representation": "projected target residual",
            "mean_representation": "exact zero after equal truth-and-mean translation",
            "gaussian_residual_and_covariance_scores_unchanged": True,
        },
        "object_records": object_records,
        "paper_claim_authorized": False,
    }
    write_json(output_root / "statistics-manifest.json", statistics_manifest)
    result = {
        "schema": SCHEMA,
        "schema_version": 7,
        "status": "completed",
        "source_confirmation": {
            "run_id": 33335779766,
            "artifact_id": 9738998271,
            "artifact_sha256": (
                "e98f9e2687f568d0d0fcabec9ce0393a7e1b34ca3019acb1e14fdf894885a948"
            ),
            "result_sha256": reference["result_sha256"],
            "confirmation_revision": reproduction["confirmation_revision"],
            "frozen_mean_method_revision": reproduction["frozen_revision"],
        },
        "exact_reproduction": {
            "object_count": len(reproduced["objects"]),
            "one_capture_per_object": True,
            "scientific_payload_matches_immutable_artifact": True,
            "maximum_absolute_numeric_difference": reproduction[
                "maximum_absolute_numeric_difference"
            ],
        },
        "same_mean_contract": {
            "exact_frozen_v3_predictive_mean_reused": True,
            "predictive_mean_changed": False,
            "only_covariance_arm_changes": True,
            "capture_location": "final target probabilistic scoring call",
            "random_number_stream_preserved": True,
            "common_origin_translation_applied_after_residual_formation": True,
        },
        "query_contract": {
            "field_dimension": field_dimension,
            "query_dimension": int(query.shape[0]),
            "family": "low-frequency normalized DCT-II pooled tactile-field modes",
            "query_matrix_sha256": statistics_manifest["query_matrix_sha256"],
        },
        "dependence_controls": {
            "diagonal_arm_preserves_each_query_variance": True,
            "permuted_arm_preserves_each_query_variance": True,
            "permuted_arm_changes_only_query_correlation_assignment": True,
        },
        "study": study_result,
        "original_full_field_uncertainty": reference["summary"]["uncertainty"],
        "statistics": {
            "file": statistics_path.name,
            "sha256": statistics_manifest["statistics_sha256"],
            "case_count": statistics_manifest["case_count"],
        },
        "classification": "retrospective grouped cross-fitted real-output development",
        "information_boundary": {
            "existing_public_dataset_numeric_payload_reopened": True,
            "exact_precommitted_92_object_roster_reused": True,
            "outcome_dependent_object_filtering": False,
            "new_measurements_collected": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "predictive_mean_retuned": False,
            "globally_fresh_confirmation": False,
        },
        "claim_boundary": (
            "This can establish calibrated same-mean query covariance and the value "
            "of query dependence on the existing Deform360 confirmation. It cannot "
            "establish a fresh final confirmation, geometric 4-D prediction, strict "
            "counterfactual causality, closed-loop robot benefit, or SOTA mean error."
        ),
        "paper_claim_authorized": False,
        "globally_fresh_confirmation_authorized": False,
        "strict_counterfactual_claim_authorized": False,
    }
    result["result_sha256"] = query_cov.canonical_digest(result)
    write_json(output_root / "result.json", result)
    reproduction_matches = result["exact_reproduction"][
        "scientific_payload_matches_immutable_artifact"
    ]
    report_lines = [
        "# Exact same-mean Deform360 query covariance v7",
        "",
        f"- Exact 92-object reproduction: **{reproduction_matches}**",
        f"- Query cases: **{study_result['case_count']}**",
        f"- Independent objects: **{study_result['independent_group_count']}**",
        f"- Query dimension: **{study_result['query_dimension']}**",
        f"- Superior target passed: **{study_result['superior_target_passed']}**",
        "",
        "| Arm | NLL/dim | nANEES | Ellipsoid coverage | Marginal coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in study_result["metrics"].items():
        report_lines.append(
            f"| `{name}` | {value['nll_per_dimension']:.8g} | "
            f"{value['normalized_anees']:.8g} | "
            f"{value['ellipsoid_coverage']:.4f} | "
            f"{value['marginal_coverage']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "All covariance arms use the exact same frozen predictive mean. Object IDs",
            "define the independent cross-fitting groups. No paper claim is "
            "authorized.",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return result


def self_test() -> None:
    query_cov.self_test()
    exact_capture.self_test()
    print("exact same-mean export v7 self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-artifact-root", type=Path)
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--maximum-queries", type=int, default=12)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.confirmation_artifact_root,
        args.confirmation_root,
        args.frozen_root,
        args.data_root,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("all root arguments are required unless --self-test is used")
    result = run_exact_export(
        args.confirmation_artifact_root,
        args.confirmation_root,
        args.frozen_root,
        args.data_root,
        args.output_root,
        args.maximum_queries,
        args.fold_count,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "superior_target_passed": result["study"]["superior_target_passed"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
