"""Maintained target-closed CLI for the active shake-to-twist cloth pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .active_probe_evaluation import (
    InputTemplate,
    SourceOutcome,
    fit_leave_one_material_out,
    replay_held_specimen,
)
from .active_probe_run import (
    arm_specs,
    belief_digest,
    build_belief_arms,
    loss_vector,
)
from .data import (
    Inputs,
    audit_dataset,
    digest,
    infer_source_scale,
    input_view,
    object_digest,
    read_prefix,
    scoring_view,
    write_json,
)
from .model import predict, score
from .run import METRICS, implementation, now, save_csv

HERE = Path(__file__).resolve().parent
SOURCE_FIT_SCHEMA = "tracking-cloth-active-source-fit-v1"
PREDICTION_SEAL_SCHEMA = "tracking-cloth-active-prediction-seal-v1"


def _source_worker(args: tuple[Any, dict[str, Any], float]) -> SourceOutcome:
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    prediction = predict(inputs, protocol)
    truth = scoring_view(case, inputs)
    return SourceOutcome(
        case.path.name,
        case.material,
        case.size,
        case.condition,
        prediction,
        truth,
    )


def _template_worker(args: tuple[Any, dict[str, Any], float]) -> InputTemplate:
    case, protocol, scale = args
    prediction = predict(input_view(case, protocol, scale), protocol)
    return InputTemplate(
        case.path.name,
        case.material,
        case.size,
        case.condition,
        prediction,
    )


def _map_workers(function: Any, tasks: list[Any], workers: int) -> list[Any]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(function, tasks))
    return [function(task) for task in tasks]


def _record_id(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return object_digest(payload)


def _verify_record_id(value: dict[str, Any], field: str) -> None:
    identifier = value.get(field)
    if not isinstance(identifier, str) or identifier != _record_id(value, field):
        raise ValueError(f"{field} does not bind the complete record")


def _selection_rows(specimens: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for specimen, replay in sorted(specimens.items()):
        for policy, states in replay["policy_states"].items():
            for budget, state in states.items():
                steps = state["steps"]
                rows.append(
                    {
                        "specimen": specimen,
                        "held_material": replay["held_material"],
                        "policy": policy,
                        "budget": int(budget),
                        "selected_actions": ";".join(state["selected_actions"]),
                        "first_action": (
                            state["selected_actions"][0]
                            if state["selected_actions"]
                            else ""
                        ),
                        "final_entropy": (
                            steps[-1]["entropy_after"] if steps else ""
                        ),
                        "final_target_model_spread_m2": (
                            steps[-1]["target_model_spread_after"] if steps else ""
                        ),
                    }
                )
    return rows


def _report_header(protocol: dict[str, Any]) -> list[str]:
    return [
        "# Task-directed active Bayesian physical twin: public-data pilot",
        "",
        f"Study: `{protocol['study_id']}`",
        "",
        "The source stage uses four leave-one-material-out folds. In each fold,",
        "24 other-material Shake outcomes fit the model belief, while 24",
        "other-material Twist input predictions define the downstream query.",
        "Held-material Twist inputs and all Twist outcomes are excluded from",
        "probe selection. Recorded Shake outcomes are requested only after the",
        "fixed-order, parameter-information, or task-directed policy selects them.",
        "",
        "This is a reduced spring-mesh retrospective pilot, not an online robot",
        "controller, a physical-safety policy, a PhysTwin/FEM reproduction, or",
        "an unseen-object validation. No paper claim is self-authorized.",
        "",
    ]


def prepare(
    root: Path,
    output: Path,
    protocol: dict[str, Any],
    stage: str,
    workers: int = 1,
) -> None:
    """Audit data, freeze source-only policy state, and optionally seal targets."""

    if stage not in ("inventory", "source", "predict"):
        raise ValueError("unknown active-probe preparation stage")
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset must be disjoint directory trees")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "active_protocol.json", protocol)
    manifest = {
        "created_at": now(),
        "protocol_id": object_digest(protocol),
        "implementation_sha256": implementation(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "target_numeric_outcomes_read": False,
        "evidence_class": protocol["evidence_class"],
        "paper_claim_authorized": False,
    }
    write_json(output / "active_run_manifest.json", manifest)
    cases, inventory = audit_dataset(root, protocol)
    write_json(output / "active_dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(inventory["included_license_text"])
    report = _report_header(protocol)
    (output / "active_report.md").write_text("\n".join(report))
    if stage == "inventory":
        return

    source_cases = [case for case in cases if case.motion == protocol["source_motion"]]
    target_cases = [case for case in cases if case.motion == protocol["target_motion"]]
    if len(source_cases) != 32 or len(target_cases) != 32:
        raise ValueError("the active pilot requires 32 Shake and 32 Twist records")
    scales = [
        infer_source_scale(case, read_prefix(case, protocol["prefix_seconds"])[1])
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("source recordings disagree about metric coordinate units")
    scale = scales[0]
    source_records = _map_workers(
        _source_worker,
        [(case, protocol, scale) for case in source_cases],
        workers,
    )
    target_templates = _map_workers(
        _template_worker,
        [(case, protocol, scale) for case in target_cases],
        workers,
    )
    source_records = sorted(source_records, key=lambda record: record.recording)
    target_templates = sorted(target_templates, key=lambda record: record.recording)

    folds = {}
    specimens = {}
    for held_material in protocol["materials"]:
        fold = fit_leave_one_material_out(
            held_material=held_material,
            source_outcomes=[
                record
                for record in source_records
                if record.material != held_material
            ],
            target_templates=[
                record
                for record in target_templates
                if record.material != held_material
            ],
            protocol=protocol,
        )
        folds[held_material] = fold
        for size in protocol["sizes"]:
            specimen = f"{held_material}_{size}"
            candidates = [
                record
                for record in source_records
                if record.material == held_material and record.size == size
            ]
            if len(candidates) != 4:
                raise ValueError("held specimen lacks the four registered probes")
            observed_losses = {
                record.condition: loss_vector(record.prediction, record.truth)
                for record in candidates
            }
            specimens[specimen] = replay_held_specimen(
                specimen=specimen,
                fold=fold,
                observed_losses=observed_losses,
                protocol=protocol,
            )

    source_fit: dict[str, Any] = {
        "schema": SOURCE_FIT_SCHEMA,
        "fitted_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "implementation_sha256": implementation(),
        "coordinate_scale_to_m": scale,
        "folds": folds,
        "specimens": specimens,
        "source_shake_outcomes_read": 32,
        "target_twist_input_templates_read": 32,
        "target_twist_outcomes_read": False,
        "each_fold_excludes_its_held_material": True,
        "selection_consumes_only_selected_probe_outcomes": True,
        "held_material_twist_inputs_used_for_selection": False,
        "held_material_twist_outcomes_used_for_selection": False,
        "cross_validation_scope": (
            "logical leave-one-material-out folds over a shared public dataset; "
            "not four temporally fresh experiments"
        ),
        "paper_claim_authorized": False,
    }
    source_fit["source_fit_id"] = _record_id(source_fit, "source_fit_id")
    write_json(output / "active_source_fit.json", source_fit)
    save_csv(output / "active_probe_selections.csv", _selection_rows(specimens))
    divergence = sum(
        int(
            replay["policy_states"]["task_directed"]["1"]["selected_actions"]
            != replay["policy_states"]["parameter_information"]["1"]
            ["selected_actions"]
        )
        for replay in specimens.values()
    )
    report.extend(
        [
            "## Source-only policy freeze",
            "",
            f"Coordinate scale to metres: `{scale}`.",
            "Four complete leave-one-material-out folds and eight specimen",
            "probe replays were content-addressed before held-target prediction.",
            (
                "Task-directed and parameter-information first probes differ "
                f"for {divergence}/8 specimens."
            ),
            "K=0 and K=4 beliefs are canonical identical endpoints across policies.",
            "No Twist free-marker outcome has been evaluated.",
            "",
        ]
    )
    (output / "active_report.md").write_text("\n".join(report))
    if stage == "source":
        return

    private = output / "private_active_predictions"
    private.mkdir(mode=0o700)
    arms = arm_specs(protocol)
    predictions = {}
    cases_by_name = {case.path.name: case for case in target_cases}
    for template in target_templates:
        case = cases_by_name[template.recording]
        prediction = template.prediction
        beliefs = build_belief_arms(
            prediction,
            prefix_last=prediction.inputs.prefix[-1],
            boundary=prediction.inputs.boundary,
            fold=folds[case.material],
            specimen=specimens[case.specimen],
            protocol=protocol,
        )
        _assert_endpoint_parity(beliefs, protocol)
        arrays = {f"{arm}_mean": beliefs[arm][0] for arm in arms}
        arrays.update({f"{arm}_variance": beliefs[arm][1] for arm in arms})
        arrays.update(
            {
                "times": prediction.inputs.times,
                "order": prediction.inputs.order,
                "corners": prediction.inputs.corners,
                "cutoff": np.array(prediction.inputs.cutoff),
                "scale": np.array(scale),
            }
        )
        artifact = private / f"{case.path.stem}.npz"
        np.savez_compressed(artifact, **arrays)
        predictions[case.path.name] = {
            "artifact": str(artifact.relative_to(output)),
            "sha256": digest(artifact),
            "specimen": case.specimen,
            "held_material": case.material,
            "fold_id": folds[case.material]["fold_id"],
            "specimen_replay_id": specimens[case.specimen]["specimen_replay_id"],
            "belief_ids": {
                arm: belief_digest(*beliefs[arm]) for arm in arms
            },
            "corner_raw_column_indices": prediction.inputs.order[
                prediction.inputs.corners
            ].tolist(),
            "causal_cutoff_seconds": float(
                prediction.inputs.times[prediction.inputs.cutoff]
            ),
        }
    if len(predictions) != 32:
        raise ValueError("refusing an incomplete active target prediction seal")
    seal: dict[str, Any] = {
        "schema": PREDICTION_SEAL_SCHEMA,
        "sealed_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "source_fit_sha256": digest(output / "active_source_fit.json"),
        "implementation_sha256": implementation(),
        "arms": list(arms),
        "predictions": predictions,
        "future_free_marker_outcomes_read": False,
        "future_driven_corner_coordinates_used": True,
        "initialization_prefix_all_markers_used": True,
        "held_fold_selection_excluded_held_target_inputs": True,
        "prior_public_outcome_exposure": (
            "unknown; cross-validated public-data pilot, not fresh confirmation"
        ),
    }
    seal["prediction_seal_id"] = _record_id(seal, "prediction_seal_id")
    write_json(output / "active_prediction_seal.json", seal)
    report.extend(
        [
            "## Active target beliefs sealed",
            "",
            "All 32 Twist targets and all 18 complete belief arms are sealed",
            "before scoring. The private trajectory arrays remain local.",
            "",
        ]
    )
    (output / "active_report.md").write_text("\n".join(report))


def _assert_endpoint_parity(
    beliefs: dict[str, tuple[np.ndarray, np.ndarray]],
    protocol: dict[str, Any],
) -> None:
    policies = tuple(protocol["probe_policies"])
    full_budget = max(int(value) for value in protocol["probe_budgets"])
    for budget in (0, full_budget):
        reference = beliefs[f"{policies[0]}_k{budget}"]
        for policy in policies[1:]:
            candidate = beliefs[f"{policy}_k{budget}"]
            if not all(
                np.array_equal(left, right)
                for left, right in zip(reference, candidate, strict=True)
            ):
                raise ValueError(f"K={budget} complete-belief endpoint parity failed")


def _paired_contrast(
    *,
    table: list[dict[str, Any]],
    specimens: list[str],
    candidate: str,
    comparator: str,
    protocol: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    differences = np.array(
        [
            next(
                row["rmse_mm"]
                for row in table
                if row["specimen"] == specimen and row["arm"] == candidate
            )
            - next(
                row["rmse_mm"]
                for row in table
                if row["specimen"] == specimen and row["arm"] == comparator
            )
            for specimen in specimens
        ]
    )
    resamples = rng.integers(
        0,
        len(specimens),
        size=(protocol["bootstrap_repetitions"], len(specimens)),
    )
    material_differences = np.array(
        [
            np.mean(
                [
                    differences[index]
                    for index, specimen in enumerate(specimens)
                    if specimen.startswith(f"{material}_")
                ]
            )
            for material in protocol["materials"]
        ]
    )
    material_samples = rng.integers(
        0,
        len(protocol["materials"]),
        size=(protocol["bootstrap_repetitions"], len(protocol["materials"])),
    )
    return {
        "candidate": candidate,
        "comparator": comparator,
        "candidate_minus_comparator_rmse_mm": float(differences.mean()),
        "specimen_bootstrap_95_interval_mm": np.quantile(
            differences[resamples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "material_cluster_sensitivity_95_interval_mm": np.quantile(
            material_differences[material_samples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "specimen_wins": int((differences < 0).sum()),
        "specimen_ties": int((differences == 0).sum()),
        "specimen_losses": int((differences > 0).sum()),
        "worst_specimen_regret_mm": float(differences.max()),
    }


def aggregate(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    source_fit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate equal-record and equal-specimen active-probe evidence."""

    arms = arm_specs(protocol)
    specimens = sorted({row["specimen"] for row in rows})
    if len(specimens) != 8 or len(rows) != 32 * len(arms):
        raise ValueError("incomplete active roster; no partial pooled result")
    if set(source_fit.get("specimens", {})) != set(specimens):
        raise ValueError("source-fit and target specimen rosters disagree")
    table = []
    for specimen in specimens:
        for arm in arms:
            subset = [
                row
                for row in rows
                if row["specimen"] == specimen and row["arm"] == arm
            ]
            if len(subset) != 4 or len({row["recording"] for row in subset}) != 4:
                raise ValueError("missing or duplicate target speed/grasp condition")
            table.append(
                {
                    "specimen": specimen,
                    "material": subset[0]["material"],
                    "arm": arm,
                    **{
                        metric: float(np.mean([row[metric] for row in subset]))
                        for metric in METRICS
                    },
                }
            )
    summary = {
        arm: {
            metric: float(
                np.mean([row[metric] for row in table if row["arm"] == arm])
            )
            for metric in METRICS
        }
        for arm in arms
    }
    primary_budget = int(protocol["primary_budget"])
    candidate = f"task_directed_k{primary_budget}"
    comparisons = (
        f"parameter_information_k{primary_budget}",
        f"fixed_order_k{primary_budget}",
        "task_directed_k0",
    )
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    contrasts = {
        comparator: _paired_contrast(
            table=table,
            specimens=specimens,
            candidate=candidate,
            comparator=comparator,
            protocol=protocol,
            rng=rng,
        )
        for comparator in comparisons
    }
    for policy in protocol["probe_policies"]:
        zero = summary[f"{policy}_k0"]
        full = summary[f"{policy}_k4"]
        if zero != summary["fixed_order_k0"] or full != summary["fixed_order_k4"]:
            raise ValueError("aggregated K=0/K=4 endpoint parity failed")

    first_actions = {
        policy: {
            action: sum(
                int(
                    replay["policy_states"][policy]["1"]["selected_actions"]
                    == [action]
                )
                for replay in source_fit["specimens"].values()
            )
            for action in protocol["probe_conditions"]
        }
        for policy in protocol["probe_policies"]
    }
    divergent = sum(
        int(
            replay["policy_states"]["task_directed"]["1"]["selected_actions"]
            != replay["policy_states"]["parameter_information"]["1"]
            ["selected_actions"]
        )
        for replay in source_fit["specimens"].values()
    )
    harmful_records = {}
    for policy in protocol["probe_policies"]:
        updated = f"{policy}_k{primary_budget}"
        baseline = f"{policy}_k0"
        by_recording = {}
        for row in rows:
            if row["arm"] == updated:
                base = next(
                    other["rmse_mm"]
                    for other in rows
                    if other["recording"] == row["recording"]
                    and other["arm"] == baseline
                )
                by_recording[row["recording"]] = row["rmse_mm"] > base
        harmful_records[policy] = int(sum(by_recording.values()))

    return table, {
        "arms": summary,
        "primary_candidate": candidate,
        "primary_contrasts": contrasts,
        "budget_curves": {
            policy: {
                str(budget): summary[f"{policy}_k{budget}"]
                for budget in protocol["probe_budgets"]
            }
            for policy in protocol["probe_policies"]
        },
        "first_probe_counts": first_actions,
        "task_vs_parameter_first_probe_disagreements": divergent,
        "harmful_primary_budget_records_vs_k0": harmful_records,
        "endpoint_parity": {"K0": True, "K4": True},
        "inferential_unit": (
            "8 material-size specimens; 4-material cluster sensitivity reported"
        ),
        "aggregation": (
            "equal recordings within specimen, then equal specimens; "
            "no frame pseudoreplication"
        ),
        "interval_interpretation": (
            "exploratory paired percentile bootstrap; not simultaneous; "
            "small specimen/material counts"
        ),
    }


def score_run(root: Path, output: Path) -> None:
    """Open Twist outcomes only after verifying the complete active belief seal."""

    root, output = root.resolve(strict=True), output.resolve(strict=True)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset must be disjoint directory trees")
    if (output / "active_target_access.json").exists():
        raise ValueError("this active pilot already started target scoring")
    protocol = json.loads((output / "active_protocol.json").read_text())
    source_fit = json.loads((output / "active_source_fit.json").read_text())
    seal = json.loads((output / "active_prediction_seal.json").read_text())
    _verify_record_id(source_fit, "source_fit_id")
    _verify_record_id(seal, "prediction_seal_id")
    if (
        seal["protocol_id"] != object_digest(protocol)
        or seal["implementation_sha256"] != implementation()
    ):
        raise ValueError("protocol or implementation changed after active sealing")
    if seal["source_fit_sha256"] != digest(output / "active_source_fit.json"):
        raise ValueError("active source fit changed after target sealing")
    cases, inventory = audit_dataset(root, protocol)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("dataset changed after active target sealing")
    private = (output / "private_active_predictions").resolve()
    for entry in seal["predictions"].values():
        path = (output / entry["artifact"]).resolve()
        if not path.is_relative_to(private) or digest(path) != entry["sha256"]:
            raise ValueError("active prediction artifact identity mismatch")
    write_json(
        output / "active_target_access.json",
        {
            "started_at": now(),
            "active_prediction_seal_sha256": digest(
                output / "active_prediction_seal.json"
            ),
            "authorized_recordings": sorted(seal["predictions"]),
            "purpose": "registered cross-validated active-probe pilot scoring",
        },
    )

    arms = tuple(seal["arms"])
    if arms != arm_specs(protocol):
        raise ValueError("sealed active arm roster differs from the protocol")
    rows = []
    for case in (case for case in cases if case.motion == protocol["target_motion"]):
        entry = seal["predictions"][case.path.name]
        with np.load(output / entry["artifact"], allow_pickle=False) as arrays:
            inputs = Inputs(
                arrays["times"],
                np.empty((0, case.markers, 3)),
                np.empty((0, 2, 3)),
                arrays["order"],
                arrays["corners"],
                int(arrays["cutoff"]),
                float(arrays["times"][0]),
                float(arrays["scale"]),
            )
            truth = scoring_view(case, inputs)
            case_scores = {}
            for arm in arms:
                mean = arrays[f"{arm}_mean"]
                variance = arrays[f"{arm}_variance"]
                if belief_digest(mean, variance) != entry["belief_ids"][arm]:
                    raise ValueError("sealed complete belief identity mismatch")
                case_scores[arm] = score(mean, variance, truth, inputs)
                rows.append(
                    {
                        "recording": case.path.name,
                        "specimen": case.specimen,
                        "material": case.material,
                        "speed": case.speed,
                        "grasp": case.grasp,
                        "arm": arm,
                        **case_scores[arm],
                    }
                )
            for budget in (0, 4):
                reference = case_scores[f"fixed_order_k{budget}"]
                for policy in ("parameter_information", "task_directed"):
                    if case_scores[f"{policy}_k{budget}"] != reference:
                        raise ValueError("active endpoint score parity failed")

    table, metrics = aggregate(rows, protocol, source_fit)
    metrics.update(
        {
            "target_recordings": 32,
            "active_arms": len(arms),
            "target_outcomes_opened_after_complete_seal": True,
            "evidence_class": protocol["evidence_class"],
            "paper_claim_authorized": False,
        }
    )
    save_csv(output / "active_target_scores.csv", rows)
    save_csv(output / "active_specimen_scores.csv", table)
    write_json(output / "active_metrics.json", metrics)
    manifest = json.loads((output / "active_run_manifest.json").read_text())
    manifest.update(
        {
            "completed_at": now(),
            "target_numeric_outcomes_read": True,
            "prediction_seal_sha256": digest(
                output / "active_prediction_seal.json"
            ),
            "metrics_sha256": digest(output / "active_metrics.json"),
            "status": "completed-active-pilot-not-claim-promoted",
        }
    )
    write_json(output / "active_run_manifest.json", manifest)

    report = (output / "active_report.md").read_text()
    report += "\n## Held-out Twist results\n\n"
    report += (
        "| Arm | Specimen-balanced RMSE [mm] | Coordinate NLL | "
        "90% coverage | Full width [mm] |\n"
    )
    report += "| --- | ---: | ---: | ---: | ---: |\n"
    for arm in arms:
        values = metrics["arms"][arm]
        report += (
            f"| {arm} | {values['rmse_mm']:.4f} | "
            f"{values['coordinate_nll']:.4f} | "
            f"{100 * values['coordinate_90_coverage']:.2f}% | "
            f"{values['mean_full_90_width_mm']:.4f} |\n"
        )
    report += "\n### Registered primary contrast\n\n"
    for comparator, values in metrics["primary_contrasts"].items():
        report += (
            f"- task_directed_k1 minus {comparator}: "
            f"{values['candidate_minus_comparator_rmse_mm']:.4f} mm; "
            f"wins/ties/losses {values['specimen_wins']}/"
            f"{values['specimen_ties']}/{values['specimen_losses']}.\n"
        )
    report += (
        "\nK=0 and K=4 endpoints are exact across policies. Intervals resample "
        "eight specimens and separately four material clusters; they are "
        "exploratory and non-simultaneous. Diagonal Gaussian scores do not "
        "validate joint trajectory covariance. The study is retrospective over "
        "recorded probes and does not establish online safety or fresh-data "
        "confirmation.\n"
    )
    (output / "active_report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("inventory", "source", "predict", "score"),
        default="source",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        if args.stage == "score":
            score_run(args.dataset_root, args.output)
        else:
            protocol = json.loads((HERE / "active_probe_protocol.json").read_text())
            prepare(args.dataset_root, args.output, protocol, args.stage, args.workers)
    except Exception as exc:
        if args.output.is_dir() and not args.output.resolve().is_relative_to(
            args.dataset_root.resolve()
        ):
            write_json(
                args.output / "active_failure.json",
                {
                    "failed_at": now(),
                    "stage": args.stage,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "target_scoring_started": (
                        args.output / "active_target_access.json"
                    ).exists(),
                    "scientific_decision": "not-evaluated-or-incomplete; no claim",
                },
            )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
