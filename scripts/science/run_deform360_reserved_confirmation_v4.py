#!/usr/bin/env python3
"""Run the frozen Deform360 action-kernel predictor on reserved objects.

The confirmation protocol binds the exact v3 development protocol and evaluator
before any reserved numeric payload is opened.  This wrapper changes only the
object roster and result classification; all model, kernel, ridge, guard,
normalization, bias, covariance, and target-selection semantics are delegated to
the byte-bound development evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-reserved-confirmation-result-v4"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "run_deform360_action_kernel_v3.py"
v3 = load_module(V3_PATH, "deform360_action_kernel_v3_frozen")
base = v3.base


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise base.EvaluationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(
    confirmation: dict[str, Any],
    development: dict[str, Any],
    base_protocol: dict[str, Any],
    root: Path,
) -> None:
    if confirmation.get("schema") != (
        "bayesian-phystwin/deform360-reserved-confirmation-protocol-v4"
    ):
        raise base.EvaluationError("unexpected confirmation protocol schema")
    if confirmation.get("status") != (
        "frozen-after-development-before-reserved-payload-access"
    ):
        raise base.EvaluationError("confirmation protocol was not frozen")
    if Path(str(confirmation["dataset_root"])) != root:
        raise base.EvaluationError("confirmation dataset root changed")

    binding = confirmation["development_method_binding"]
    development_path = Path(str(binding["protocol_path"]))
    if sha256_file(development_path) != binding["protocol_file_sha256"]:
        raise base.EvaluationError("frozen development protocol bytes changed")
    if (
        git_blob_sha1(Path(str(binding["implementation_path"])))
        != binding["implementation_git_blob_sha1"]
    ):
        raise base.EvaluationError("v3 implementation blob changed")
    if (
        git_blob_sha1(Path(str(binding["base_implementation_path"])))
        != binding["base_implementation_git_blob_sha1"]
    ):
        raise base.EvaluationError("v2 base implementation blob changed")
    source_revision = str(binding["development_source_revision"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_revision, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise base.EvaluationError("development source is not an ancestor")

    if development.get("schema") != (
        "bayesian-phystwin/deform360-action-kernel-protocol-v3"
    ):
        raise base.EvaluationError("development protocol schema changed")
    if list(development["development_object_ids"]) != list(
        base_protocol["development_object_ids"]
    ):
        raise base.EvaluationError("development/base object rosters disagree")
    reserved = set(map(str, development["reserved_object_ids"]))
    eligible = list(map(str, confirmation["eligible_reserved_object_ids"]))
    ineligible = list(map(str, confirmation["ineligible_reserved_object_ids"]))
    if len(eligible) != len(set(eligible)) or len(ineligible) != len(set(ineligible)):
        raise base.EvaluationError("confirmation rosters contain duplicates")
    if set(eligible) & set(ineligible):
        raise base.EvaluationError("eligible and ineligible rosters overlap")
    if set(eligible) | set(ineligible) != reserved:
        raise base.EvaluationError("confirmation rosters do not partition reserved set")
    if set(eligible) & set(development["development_object_ids"]):
        raise base.EvaluationError("confirmation and development rosters overlap")
    if confirmation["selection"].get("replacement_allowed") is not False:
        raise base.EvaluationError("confirmation replacement must be disabled")
    if (
        confirmation["information_boundary"].get(
            "protocol_frozen_before_reserved_numeric_payload_access"
        )
        is not True
    ):
        raise base.EvaluationError("reserved pre-access freeze is absent")
    if confirmation.get("paper_claim_authorized") is not False:
        raise base.EvaluationError("protocol self-authorized a paper claim")
    if confirmation.get("fresh_confirmation_authorized") is not False:
        raise base.EvaluationError("protocol self-authorized global freshness")


def confirmation_decision(
    summary: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    rule = protocol["confirmation_decision"]
    persistence = summary["comparisons"]["persistence"]
    shuffled = summary["comparisons"]["shuffled_action_control"]
    object_count = int(summary["object_count"])

    gates = {
        "complete_eligible_roster": object_count == int(rule["required_object_count"]),
        "mean_superiority_to_persistence": (
            float(persistence["ensemble_minus_comparator"]) < 0.0
        ),
        "bootstrap_superiority_to_persistence": (
            float(persistence["object_bootstrap_95_interval"][1]) < 0.0
        ),
        "minimum_object_wins": (
            int(persistence["object_wins"]) >= int(rule["minimum_object_wins"])
        ),
        "maximum_object_losses": (
            int(persistence["object_losses"]) <= int(rule["maximum_object_losses"])
        ),
        "relation_breaking_control": (
            float(shuffled["ensemble_minus_comparator"]) < 0.0
            and float(shuffled["object_bootstrap_95_interval"][1]) < 0.0
        ),
        "nontrivial_guard_acceptance": (
            float(summary["guard_acceptance_fraction"])
            >= float(rule["minimum_guard_acceptance_fraction"])
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
    probabilistic_supported = all(uncertainty_diagnostic.values())
    return {
        "gates": gates,
        "point_confirmation_supported": point_supported,
        "probabilistic_confirmation_supported": probabilistic_supported,
        "uncertainty_diagnostic": uncertainty_diagnostic,
        "icra_promising_real_action_forecasting_evidence": point_supported,
        "paper_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "strict_counterfactual_claim_authorized": False,
    }


def make_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["confirmation_decision"]
    lines = [
        "# Deform360 reserved-object action confirmation v4",
        "",
        f"- Status: **{result['status']}**",
        f"- Reserved objects scored: **{summary['object_count']}**",
        f"- Horizon: **{summary['primary_horizon_frames']} frames**",
        f"- Guard acceptance: **{summary['guard_acceptance_fraction']:.1%}**",
        f"- Point confirmation supported: **{str(decision['point_confirmation_supported']).lower()}**",
        f"- Probabilistic confirmation supported: **{str(decision['probabilistic_confirmation_supported']).lower()}**",
        "",
        "## Object-balanced point results",
        "",
        "| Method | Active RMSE | All-field RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in v3.METHODS:
        value = summary["methods"][method]
        lines.append(
            f"| `{method}` | {value['active_field_rmse']:.8g} | "
            f"{value['field_rmse']:.8g} | {value['field_mae']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Confirmation contrasts",
            "",
            "Negative values favor the frozen Bayesian action ensemble.",
            "",
            "| Comparator | Difference | Relative | 95% bootstrap | W/T/L | Worst regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator, value in summary["comparisons"].items():
        interval = value["object_bootstrap_95_interval"]
        lines.append(
            f"| `{comparator}` | {value['ensemble_minus_comparator']:.8g} | "
            f"{value['relative_change']:+.2%} | "
            f"[{interval[0]:.8g}, {interval[1]:.8g}] | "
            f"{value['object_wins']}/{value['object_ties']}/{value['object_losses']} | "
            f"{value['worst_object_regret']:.8g} |"
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
            "The exact v3 development predictor and hyperparameters were byte-bound before",
            "any reserved numeric payload was opened. Each target episode was selected from",
            "metadata and carrier identities; all other complete episodes of the same object",
            "were used for source fitting. Future robot motion was supplied as the registered",
            "intervention input, while future tactile response was used only for scoring.",
            "",
            "This is experiment-internal confirmation on released measurements. It is not",
            "globally fresh, dense 4-D geometric validation, a strict individual",
            "counterfactual, a safety result, or an automatic paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(protocol_path: Path, root: Path) -> dict[str, Any]:
    confirmation = read_json(protocol_path)
    binding = confirmation["development_method_binding"]
    development_path = Path(str(binding["protocol_path"]))
    development = read_json(development_path)
    base_protocol = read_json(
        Path(str(development["shared_preprocessing"]["base_protocol_path"]))
    )
    root = root.resolve(strict=True)
    validate_protocol(confirmation, development, base_protocol, root)

    minimum_episodes = int(
        confirmation["selection"]["minimum_complete_episodes_per_object"]
    )
    descriptor_map: dict[str, list[Any]] = {}
    for object_id in confirmation["eligible_reserved_object_ids"]:
        descriptors = base.discover_object(root, str(object_id), minimum_episodes)
        if len(descriptors) < minimum_episodes:
            raise base.EvaluationError(
                f"eligible reserved object lost required carriers: {object_id}"
            )
        descriptor_map[str(object_id)] = descriptors
    if set(descriptor_map) != set(confirmation["eligible_reserved_object_ids"]):
        raise base.EvaluationError("eligible reserved roster was not completed")

    rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    for object_id in confirmation["eligible_reserved_object_ids"]:
        row = v3.evaluate_object(
            descriptor_map[str(object_id)], development, base_protocol, rng
        )
        row["confirmation_object"] = True
        row["development_method_unchanged"] = True
        row["reserved_numeric_payload_opened"] = True
        rows.append(row)

    summary = v3.aggregate(rows, development)
    decision = confirmation_decision(summary, confirmation)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 4,
        "status": "complete",
        "protocol_id": confirmation["protocol_id"],
        "dataset_root": str(root),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "method_binding": binding,
        "readiness_binding": confirmation["readiness_binding"],
        "information_boundary": {
            "protocol_frozen_before_reserved_numeric_payload_access": True,
            "exact_development_method_reused": True,
            "eligible_reserved_robot_trajectories_opened": True,
            "eligible_reserved_tactile_responses_opened": True,
            "future_robot_trajectory_is_intervention_input": True,
            "target_tactile_opened_after_source_fit": True,
            "ineligible_reserved_payloads_opened": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
        },
        "summary": summary,
        "confirmation_decision": decision,
        "objects": rows,
        "protocol": confirmation,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.protocol, args.data_root)
    write_json(args.output_json, result)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["confirmation_decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
