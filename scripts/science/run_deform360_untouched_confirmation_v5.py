#!/usr/bin/env python3
"""Run the exact frozen Deform360 v3 predictor on every untouched-ready object.

The object roster and target episodes are bound by a metadata/file-identity-only
readiness artifact created before these numeric payloads are opened. This
wrapper imports the exact v3 evaluator from a separate checkout of its frozen
source revision. It does not alter the point predictor, horizon, candidate
family, model averaging, preprocessing, source fitting, or action-shuffle
control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-untouched-confirmation-result-v5"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-untouched-confirmation-protocol-v5"
READINESS_SCHEMA = "bayesian-phystwin/deform360-untouched-readiness-v5"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def exact_one_sided_sign_pvalue(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0 or wins <= losses:
        return 1.0
    numerator = sum(math.comb(count, index) for index in range(wins, count + 1))
    return float(numerator / (2**count))


def paired_statistics(rows: list[dict[str, Any]], comparator: str) -> dict[str, Any]:
    metric = "active_field_rmse"
    differences = np.asarray(
        [
            row["metrics"]["bayesian_action_ensemble"][metric]
            - row["metrics"][comparator][metric]
            for row in rows
        ],
        dtype=np.float64,
    )
    wins = int(np.sum(differences < 0.0))
    ties = int(np.sum(differences == 0.0))
    losses = int(np.sum(differences > 0.0))
    return {
        "object_count": len(rows),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "standard_error_of_mean_difference": float(
            np.std(differences, ddof=1) / math.sqrt(len(differences))
            if len(differences) > 1
            else 0.0
        ),
        "object_wins": wins,
        "object_ties": ties,
        "object_losses": losses,
        "win_fraction_excluding_ties": float(
            wins / (wins + losses) if wins + losses else 0.0
        ),
        "exact_one_sided_sign_test_pvalue": exact_one_sided_sign_pvalue(wins, losses),
        "minimum_difference": float(np.min(differences)),
        "maximum_difference": float(np.max(differences)),
    }


def selection_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": row["object_id"],
        "metadata_sha256": row["metadata_sha256"],
        "complete_episode_ids": row["complete_episode_ids"],
        "target_episode_id": row["target_episode_id"],
        "target_action": row["target_action"],
        "robot_files": row["robot_files"],
        "tactile_groups": [
            {
                "directory": group["directory"],
                "recording_count": group["recording_count"],
                "recordings": group["recordings"],
            }
            for group in row["tactile_groups"]
        ],
    }


def verify_readiness(
    readiness: dict[str, Any],
    protocol: dict[str, Any],
    readiness_path: Path,
) -> list[dict[str, Any]]:
    binding = protocol["readiness_binding"]
    if readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("unexpected readiness schema")
    if sha256_file(readiness_path) != binding["readiness_file_sha256"]:
        raise ValueError("readiness file bytes changed")
    stored_result_digest = readiness.get("result_sha256")
    unsigned = dict(readiness)
    unsigned.pop("result_sha256", None)
    if canonical_digest(unsigned) != stored_result_digest:
        raise ValueError("readiness result digest is invalid")
    if stored_result_digest != binding["readiness_result_sha256"]:
        raise ValueError("readiness result binding changed")
    manifest = readiness.get("selection_manifest")
    if not isinstance(manifest, list):
        raise ValueError("readiness selection manifest is absent")
    if canonical_digest(manifest) != readiness.get("selection_manifest_sha256"):
        raise ValueError("readiness selection manifest digest is invalid")
    if (
        readiness.get("selection_manifest_sha256")
        != binding["selection_manifest_sha256"]
    ):
        raise ValueError("selection manifest binding changed")
    boundary = readiness["information_boundary"]
    for key in (
        "robot_numeric_payloads_opened",
        "tactile_numeric_payloads_opened",
        "target_outcomes_scored",
        "camera_pixels_opened",
        "geometry_or_point_cloud_opened",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"readiness numeric boundary changed: {key}")
    object_ids = [str(row["object_id"]) for row in manifest]
    expected = list(map(str, protocol["eligible_object_ids"]))
    if object_ids != expected:
        raise ValueError("readiness and confirmation object rosters disagree")
    required = int(protocol["confirmation_decision"]["required_object_count"])
    if len(object_ids) != required:
        raise ValueError("unexpected untouched confirmation object count")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("untouched confirmation roster contains duplicates")
    return manifest


def validate_frozen_method(
    frozen_root: Path,
    protocol: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    binding = protocol["development_method_binding"]
    expected_revision = str(binding["development_source_revision"])
    revision = git_output(frozen_root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(
            f"frozen method checkout changed: expected {expected_revision}, got {revision}"
        )

    development_path = frozen_root / str(binding["protocol_path"])
    implementation_path = frozen_root / str(binding["implementation_path"])
    base_path = frozen_root / str(binding["base_implementation_path"])
    if sha256_file(development_path) != binding["protocol_file_sha256"]:
        raise ValueError("frozen development protocol bytes changed")
    implementation_blob = git_output(
        frozen_root, "hash-object", str(implementation_path)
    )
    if implementation_blob != binding["implementation_git_blob_sha1"]:
        raise ValueError("frozen v3 implementation blob changed")
    base_blob = git_output(frozen_root, "hash-object", str(base_path))
    if base_blob != binding["base_implementation_git_blob_sha1"]:
        raise ValueError("frozen v2 base implementation blob changed")

    development = read_json(development_path)
    if development.get("schema") != (
        "bayesian-phystwin/deform360-action-kernel-protocol-v3"
    ):
        raise ValueError("frozen v3 protocol schema changed")
    horizon = int(development["shared_preprocessing"]["forecast_horizon_frames"])
    if horizon != 32:
        raise ValueError("frozen forecast horizon changed")
    base_protocol = read_json(
        frozen_root / str(development["shared_preprocessing"]["base_protocol_path"])
    )
    v3 = load_module(implementation_path, "deform360_action_kernel_v3_exact_v5")
    return v3, development, base_protocol


def validate_protocol(protocol: dict[str, Any], root: Path) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected untouched confirmation protocol schema")
    if protocol.get("status") != "frozen-before-untouched-numeric-payload-access":
        raise ValueError("untouched confirmation protocol is not frozen")
    if Path(str(protocol["dataset_root"])) != root:
        raise ValueError("dataset root changed")
    selection = protocol["selection"]
    if selection.get("include_every_readiness_eligible_object") is not True:
        raise ValueError("all readiness-eligible objects must be included")
    if selection.get("replacement_allowed") is not False:
        raise ValueError("confirmation replacement must be disabled")
    if selection.get("partial_roster_result_allowed") is not False:
        raise ValueError("partial untouched confirmation is not allowed")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("confirmation protocol self-authorized a paper claim")


def action_family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["target_action_family"])].append(row)
    result: dict[str, Any] = {}
    for family, family_rows in sorted(groups.items()):
        result[family] = {
            "object_count": len(family_rows),
            "versus_persistence": paired_statistics(family_rows, "persistence"),
            "versus_shuffled_action_control": paired_statistics(
                family_rows, "shuffled_action_control"
            ),
        }
    return result


def confirmation_decision(
    summary: dict[str, Any],
    robust: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rule = protocol["confirmation_decision"]
    persistence = summary["comparisons"]["persistence"]
    shuffled = summary["comparisons"]["shuffled_action_control"]
    persistence_robust = robust["versus_persistence"]
    shuffled_robust = robust["versus_shuffled_action_control"]
    gates = {
        "complete_precommitted_roster": (
            int(summary["object_count"]) == int(rule["required_object_count"])
        ),
        "mean_superiority_to_persistence": (
            float(persistence["ensemble_minus_comparator"]) < 0.0
        ),
        "bootstrap_superiority_to_persistence": (
            float(persistence["object_bootstrap_95_interval"][1]) < 0.0
        ),
        "majority_object_wins_to_persistence": (
            int(persistence["object_wins"]) > int(persistence["object_losses"])
        ),
        "sign_test_superiority_to_persistence": (
            float(persistence_robust["exact_one_sided_sign_test_pvalue"])
            < float(rule["maximum_sign_test_pvalue"])
        ),
        "mean_relation_breaking_control": (
            float(shuffled["ensemble_minus_comparator"]) < 0.0
        ),
        "bootstrap_relation_breaking_control": (
            float(shuffled["object_bootstrap_95_interval"][1]) < 0.0
        ),
        "majority_relation_breaking_wins": (
            int(shuffled["object_wins"]) > int(shuffled["object_losses"])
        ),
        "sign_test_relation_breaking_control": (
            float(shuffled_robust["exact_one_sided_sign_test_pvalue"])
            < float(rule["maximum_sign_test_pvalue"])
        ),
    }
    point_supported = all(gates.values())

    uncertainty = summary["uncertainty"]
    uncertainty_diagnostic = {
        "marginal_coverage_near_90": (
            0.80 <= float(uncertainty["marginal_90_coverage"]) <= 0.98
        ),
        "joint_coverage_near_90": (
            0.75 <= float(uncertainty["joint_90_ellipsoid_coverage"]) <= 0.98
        ),
        "joint_nanees_reasonable": (0.5 <= float(uncertainty["joint_nanees"]) <= 2.0),
    }
    return {
        "gates": gates,
        "point_confirmation_supported": point_supported,
        "probabilistic_confirmation_supported": all(uncertainty_diagnostic.values()),
        "uncertainty_diagnostic": uncertainty_diagnostic,
        "icra_evidence_materially_strengthened": point_supported,
        "paper_claim_authorized": False,
        "globally_fresh_confirmation_authorized": False,
        "strict_counterfactual_claim_authorized": False,
    }


def write_object_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "object_id",
        "target_episode_id",
        "target_action",
        "target_action_family",
        "forecast_window_count",
        "persistence_active_rmse",
        "ensemble_active_rmse",
        "shuffled_action_active_rmse",
        "ensemble_minus_persistence",
        "relative_change_to_persistence",
        "ensemble_minus_shuffled_action",
        "relative_change_to_shuffled_action",
        "guard_accepts",
        "joint_nanees",
        "joint_90_ellipsoid_coverage",
        "marginal_90_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            persistence = row["metrics"]["persistence"]["active_field_rmse"]
            ensemble = row["metrics"]["bayesian_action_ensemble"]["active_field_rmse"]
            shuffled = row["metrics"]["shuffled_action_control"]["active_field_rmse"]
            writer.writerow(
                {
                    "object_id": row["object_id"],
                    "target_episode_id": row["target_episode_id"],
                    "target_action": row["target_action"],
                    "target_action_family": row["target_action_family"],
                    "forecast_window_count": row["forecast_window_count"],
                    "persistence_active_rmse": persistence,
                    "ensemble_active_rmse": ensemble,
                    "shuffled_action_active_rmse": shuffled,
                    "ensemble_minus_persistence": ensemble - persistence,
                    "relative_change_to_persistence": (
                        (ensemble - persistence) / persistence
                    ),
                    "ensemble_minus_shuffled_action": ensemble - shuffled,
                    "relative_change_to_shuffled_action": (
                        (ensemble - shuffled) / shuffled
                    ),
                    "guard_accepts": row["guard_accepts"],
                    "joint_nanees": row["uncertainty"]["joint_nanees"],
                    "joint_90_ellipsoid_coverage": row["uncertainty"][
                        "joint_90_ellipsoid_coverage"
                    ],
                    "marginal_90_coverage": row["uncertainty"]["marginal_90_coverage"],
                }
            )


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["confirmation_decision"]
    robust = result["robust_statistics"]
    lines = [
        "# Deform360 study-relative untouched-object confirmation v5",
        "",
        f"- Status: **{result['status']}**",
        f"- Precommitted untouched objects scored: **{summary['object_count']}**",
        f"- Horizon: **{summary['primary_horizon_frames']} frames**",
        "- Exact frozen v3 point method reused: **yes**",
        "- Point confirmation supported: "
        f"**{str(decision['point_confirmation_supported']).lower()}**",
        "- Probabilistic confirmation supported: "
        f"**{str(decision['probabilistic_confirmation_supported']).lower()}**",
        "",
        "## Object-balanced point results",
        "",
        "| Method | Active RMSE | All-field RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        lines.append(
            f"| `{method}` | {values['active_field_rmse']:.8g} | "
            f"{values['field_rmse']:.8g} | {values['field_mae']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Primary paired contrasts",
            "",
            "Negative differences favor the frozen Bayesian action ensemble.",
            "",
            "| Comparator | Difference | Relative | 95% object bootstrap | W/T/L | "
            "Exact one-sided sign p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator in ("persistence", "shuffled_action_control"):
        contrast = summary["comparisons"][comparator]
        stats = robust[f"versus_{comparator}"]
        interval = contrast["object_bootstrap_95_interval"]
        lines.append(
            f"| `{comparator}` | {contrast['ensemble_minus_comparator']:.8g} | "
            f"{contrast['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | "
            f"{contrast['object_wins']}/{contrast['object_ties']}/"
            f"{contrast['object_losses']} | "
            f"{stats['exact_one_sided_sign_test_pvalue']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen confirmation gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    for name, passed in decision["gates"].items():
        lines.append(f"| `{name}` | {str(bool(passed)).lower()} |")
    lines.extend(
        [
            "",
            "## Target-action-family robustness",
            "",
            "| Action family | Objects | Ensemble-persistence difference | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, values in result["action_family_summary"].items():
        stats = values["versus_persistence"]
        lines.append(
            f"| `{family}` | {values['object_count']} | "
            f"{stats['mean_difference']:.8g} | "
            f"{stats['object_wins']}/{stats['object_ties']}/"
            f"{stats['object_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Probabilistic diagnostics",
            "",
            "| Diagnostic | Value |",
            "|---|---:|",
        ]
    )
    for name, value in summary["uncertainty"].items():
        lines.append(f"| `{name}` | {value:.8g} |")
    lines.extend(
        [
            "",
            "The object roster, target episode identities, metadata hashes, and carrier",
            "file identities were fixed by the successful v5 readiness artifact before",
            "numeric robot/tactile access. Every eligible object was required; replacement,",
            "partial-roster reporting, and outcome-dependent filtering were prohibited.",
            "The evaluator was imported from the exact v3 development revision and its",
            "protocol and implementation blob identities were rechecked before execution.",
            "",
            "This is a much larger confirmation on released real measurements that were",
            "untouched by the v3/v4 study. It is not a newly collected dataset, zero-shot",
            "unseen-object transfer, dense 4-D geometry validation, strict individual",
            "counterfactual evidence, or an automatic paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    protocol_path: Path,
    readiness_path: Path,
    data_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    data_root = data_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    validate_protocol(protocol, data_root)
    readiness = read_json(readiness_path)
    manifest = verify_readiness(readiness, protocol, readiness_path)
    v3, development, base_protocol = validate_frozen_method(frozen_root, protocol)

    audit_path = Path(__file__).with_name(
        "audit_deform360_untouched_confirmation_v5.py"
    )
    audit = load_module(audit_path, "deform360_untouched_audit_v5_for_confirmation")
    minimum = int(protocol["selection"]["minimum_complete_episodes_per_object"])

    current_manifest: list[dict[str, Any]] = []
    for expected in manifest:
        current = audit.inspect_object(data_root, str(expected["object_id"]), minimum)
        if not current.get("eligible"):
            raise ValueError(
                f"precommitted object lost carrier eligibility: {expected['object_id']}"
            )
        projection = selection_projection(current)
        if projection != expected:
            raise ValueError(
                f"precommitted metadata/carrier identity changed: {expected['object_id']}"
            )
        current_manifest.append(projection)
    expected_manifest_digest = protocol["readiness_binding"][
        "selection_manifest_sha256"
    ]
    if canonical_digest(current_manifest) != expected_manifest_digest:
        raise ValueError("recomputed pre-access manifest digest changed")

    rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(manifest, start=1):
        object_id = str(expected["object_id"])
        print(f"[{index}/{len(manifest)}] evaluating {object_id}", flush=True)
        descriptors = v3.base.discover_object(data_root, object_id, minimum)
        descriptor_ids = [int(item.episode_id) for item in descriptors]
        if descriptor_ids != list(expected["complete_episode_ids"]):
            raise ValueError(f"numeric evaluator roster changed for {object_id}")
        row = v3.evaluate_object(descriptors, development, base_protocol, rng)
        if int(row["target_episode_id"]) != int(expected["target_episode_id"]):
            raise ValueError(f"target episode changed for {object_id}")
        row["study_relative_untouched_confirmation_object"] = True
        row["readiness_manifest_bound"] = True
        row["exact_frozen_v3_method_unchanged"] = True
        rows.append(row)

    summary = v3.aggregate(rows, development)
    robust = {
        "versus_persistence": paired_statistics(rows, "persistence"),
        "versus_shuffled_action_control": paired_statistics(
            rows, "shuffled_action_control"
        ),
    }
    decision = confirmation_decision(summary, robust, protocol)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 5,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "dataset_root": str(data_root),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "development_method_binding": protocol["development_method_binding"],
        "readiness_binding": protocol["readiness_binding"],
        "selection_manifest_recomputed_sha256": canonical_digest(current_manifest),
        "information_boundary": {
            "confirmation_protocol_frozen_before_numeric_access": True,
            "readiness_selected_every_eligible_object": True,
            "metadata_and_carrier_identities_reverified_before_numeric_access": True,
            "exact_frozen_v3_point_method_reused": True,
            "eligible_robot_numeric_payloads_opened": True,
            "eligible_tactile_numeric_payloads_opened": True,
            "future_robot_trajectory_is_intervention_input": True,
            "target_tactile_opened_after_source_fit": True,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
            "outcome_dependent_object_filtering": False,
        },
        "summary": summary,
        "robust_statistics": robust,
        "action_family_summary": action_family_summary(rows),
        "confirmation_decision": decision,
        "objects": rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    result = run(
        args.protocol,
        args.readiness_json,
        args.data_root,
        args.frozen_root,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_object_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["robust_statistics"], indent=2, sort_keys=True))
    print(json.dumps(result["confirmation_decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
