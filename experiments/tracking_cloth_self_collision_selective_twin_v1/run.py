"""Three-stage self-collision confirmation for selective digital twins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    Case,
    InputView,
    audit_dataset,
    object_digest,
    prediction_input,
    scoring_truth,
    write_json,
)
from .model import PhysicsFit, all_predictions, fit_physics
from .selection import (
    SELECTOR_ARMS,
    apply_policy,
    confirmation_gate,
    fit_cross_material_policies,
    incremental_summary,
    score_case,
    source_gate,
    summarize_policy_rows,
)

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(json.dumps(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fit_key(case: Case) -> str:
    return f"{case.material}|{case.interaction}"


def _input_descriptor(inputs: InputView) -> dict[str, Any]:
    return {
        "case_id": inputs.case.case_id,
        "recording": inputs.case.path.name,
        "material": inputs.case.material,
        "interaction": inputs.case.interaction,
        "repetition": inputs.case.repetition,
        "times_sha256": array_digest(inputs.times),
        "cloth_prefix_sha256": array_digest(inputs.cloth_prefix),
        "rod_prefix_sha256": array_digest(inputs.rod_prefix),
        "cutoff": inputs.cutoff,
        "scale": inputs.scale,
        "cloth_indices": inputs.cloth_indices.tolist(),
        "cloth_order": inputs.cloth_order.tolist(),
        "rod_indices": inputs.rod_indices.tolist(),
        "marker_count": inputs.marker_count,
        "initial_diameter_m": inputs.initial_diameter_m,
        "future_coordinates_read": False,
    }


def _prediction_key(case: Case, arm: str) -> str:
    safe_case = case.case_id.replace(":", "__")
    return f"{safe_case}___{arm}"


def _write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                item[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                item[key] = value
        flattened.append(item)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != (
        "bayesian-phystwin.tracking-cloth-self-collision-selective-twin.v1"
    ):
        raise ValueError("protocol schema changed")
    if protocol.get("fit_repetition") != 1:
        raise ValueError("fit repetition must remain rep1")
    if protocol.get("selection_repetition") != 2:
        raise ValueError("selection repetition must remain rep2")
    if protocol.get("confirmation_repetition") != 3:
        raise ValueError("confirmation repetition must remain rep3")
    boundary = protocol["information_boundary"]
    required_true = (
        "fresh_confirmation",
        "rep3_numeric_outcomes_may_not_be_read_before_joint_prediction_seal",
        "rep3_future_timestamps_may_define_prediction_grid",
    )
    if any(boundary.get(name) is not True for name in required_true):
        raise ValueError("required information boundary weakened")
    future_coordinates = boundary.get(
        "rep3_future_cloth_or_rod_coordinates_used_for_prediction"
    )
    if future_coordinates is not False:
        raise ValueError("future rep3 coordinates must remain excluded")
    if boundary.get("paper_claim_authorized") is not False:
        raise ValueError("protocol cannot authorize a paper claim")


def source_stage(dataset_root: Path, output: Path, protocol_path: Path) -> None:
    protocol = load_json(protocol_path)
    _verify_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    cases, inventory = audit_dataset(dataset_root, protocol)
    write_json(output / "protocol.json", protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"], encoding="utf-8"
    )

    rep1 = [case for case in cases if case.repetition == 1]
    rep2 = [case for case in cases if case.repetition == 2]
    if len(rep1) != 12 or len(rep2) != 12:
        raise ValueError("rep1/rep2 roster changed")

    fits: dict[str, Any] = {}
    fit_inputs: list[dict[str, Any]] = []
    for case in rep1:
        inputs = prediction_input(case, protocol)
        truth = scoring_truth(case, inputs)
        fit = fit_physics(inputs, truth, protocol)
        key = _fit_key(case)
        if key in fits:
            raise ValueError("duplicate rep1 fit key")
        fits[key] = fit.record()
        fit_inputs.append(_input_descriptor(inputs))
    fit_record = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-fits.v1",
        "schema_version": 1,
        "inventory_id": inventory["inventory_id"],
        "fit_repetition": 1,
        "fits": fits,
        "fit_inputs": fit_inputs,
        "rep3_outcomes_read": False,
    }
    fit_record["fits_id"] = object_digest(fit_record)
    write_json(output / "physics_fits.json", fit_record)

    rep2_rows: list[dict[str, Any]] = []
    selection_inputs: list[dict[str, Any]] = []
    for case in rep2:
        inputs = prediction_input(case, protocol)
        truth = scoring_truth(case, inputs)
        fit = PhysicsFit.from_record(fits[_fit_key(case)])
        predictions = all_predictions(inputs, fit, protocol)
        rep2_rows.extend(score_case(predictions, truth, inputs, protocol))
        selection_inputs.append(_input_descriptor(inputs))
    policy = fit_cross_material_policies(rep2_rows, protocol)
    policy["inventory_id"] = inventory["inventory_id"]
    policy["fits_id"] = fit_record["fits_id"]
    policy["selection_inputs"] = selection_inputs
    policy["policy_id"] = object_digest(
        {name: value for name, value in policy.items() if name != "policy_id"}
    )
    write_json(output / "policy.json", policy)

    policy_rows = apply_policy(rep2_rows, policy)
    summaries = summarize_policy_rows(policy_rows, protocol)
    incremental = incremental_summary(policy_rows, protocol)
    gate = source_gate(summaries, incremental, protocol)
    result = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-source-result.v1",
        "schema_version": 1,
        "inventory_id": inventory["inventory_id"],
        "fits_id": fit_record["fits_id"],
        "policy_id": policy["policy_id"],
        "summaries": summaries,
        "incremental": incremental,
        "source_gate": gate,
        "rep3_numeric_outcomes_read": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = object_digest(result)
    write_json(output / "source_result.json", result)
    _write_flat_csv(output / "rep2_query_cases.csv", rep2_rows)
    _write_flat_csv(output / "rep2_policy_cases.csv", policy_rows)
    report = [
        "# Tracking Cloth self-collision source gate",
        "",
        f"Decision: **{'pass' if gate['pass'] else 'fail'}**",
        "",
        "Rep1 fits the contact-model bank. Rep2 selects the leave-one-material-out",
        "query/interaction/horizon handoff. Rep3 remains numerically unopened.",
        "",
        "## Frozen source criteria",
        "",
    ]
    report.extend(
        f"- `{name}`: **{'pass' if passed else 'fail'}**"
        for name, passed in gate["criteria"].items()
    )
    report.extend(
        [
            "",
            "## Incremental source result",
            "",
            "- Physics minus matched residual: "
            f"{incremental['physics_minus_residual_mm']:.6f} mm",
            "- Incremental relative gain: "
            f"{100 * incremental['incremental_relative_gain']:.3f}%",
            "- Physics coverage: "
            f"{100 * summaries['physics_enabled']['physics_coverage']:.3f}%",
            f"- Materials nonpositive: {incremental['materials_nonpositive']}/4",
            "",
            protocol["claim_boundary"],
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        output / "run_manifest.json",
        {
            "schema": "bayesian-phystwin.tracking-cloth-self-collision-source-run.v1",
            "created_at": now(),
            "stage": "source",
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "result_id": result["result_id"],
            "rep3_numeric_outcomes_read": False,
        },
    )


def _verify_source(
    source: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_result = load_json(source / "source_result.json")
    fits = load_json(source / "physics_fits.json")
    policy = load_json(source / "policy.json")
    if source_result["source_gate"]["pass"] is not True:
        raise ValueError("source gate did not pass; rep3 is not authorized")
    if source_result["fits_id"] != fits["fits_id"]:
        raise ValueError("source fit identity changed")
    if source_result["policy_id"] != policy["policy_id"]:
        raise ValueError("source policy identity changed")
    if source_result["claim_boundary"] != protocol["claim_boundary"]:
        raise ValueError("source claim boundary changed")
    return source_result, fits, policy


def predict_stage(
    dataset_root: Path,
    source: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = load_json(protocol_path)
    _verify_protocol(protocol)
    source_result, fit_record, policy = _verify_source(source, protocol)
    output.mkdir(parents=True, exist_ok=False)
    cases, inventory = audit_dataset(dataset_root, protocol)
    if inventory["inventory_id"] != source_result["inventory_id"]:
        raise ValueError("dataset inventory differs from source stage")
    rep3 = [case for case in cases if case.repetition == 3]
    if len(rep3) != 12:
        raise ValueError("rep3 roster changed")

    arrays: dict[str, np.ndarray] = {}
    mappings: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for case in rep3:
        inputs = prediction_input(case, protocol)
        input_records.append(_input_descriptor(inputs))
        fit = PhysicsFit.from_record(fit_record["fits"][_fit_key(case)])
        predictions = all_predictions(inputs, fit, protocol)
        for arm, prediction in predictions.items():
            key = _prediction_key(case, arm)
            arrays[key] = prediction
            mappings.append(
                {
                    "case_id": case.case_id,
                    "arm": arm,
                    "array_key": key,
                    "shape": list(prediction.shape),
                    "sha256": array_digest(prediction),
                }
            )

    private_path = output / "private_predictions.npz"
    np.savez_compressed(private_path, **arrays)
    seal = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-prediction-seal.v1",
        "schema_version": 1,
        "inventory_id": inventory["inventory_id"],
        "source_result_id": source_result["result_id"],
        "fits_id": fit_record["fits_id"],
        "policy_id": policy["policy_id"],
        "prediction_file_sha256": file_digest(private_path),
        "prediction_arrays": mappings,
        "prediction_inputs": input_records,
        "rep3_prediction_count": len(rep3),
        "rep3_future_cloth_or_rod_coordinates_read": False,
        "rep3_numeric_outcomes_read": False,
        "created_at": now(),
    }
    seal["seal_id"] = object_digest(seal)
    write_json(output / "prediction_seal.json", seal)
    write_json(output / "protocol.json", protocol)
    write_json(output / "dataset_manifest.json", inventory)
    shutil.copy2(source / "source_result.json", output / "source_result.json")
    shutil.copy2(source / "physics_fits.json", output / "physics_fits.json")
    shutil.copy2(source / "policy.json", output / "policy.json")
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"], encoding="utf-8"
    )
    (output / "report.md").write_text(
        "# Tracking Cloth self-collision prediction seal\n\n"
        "Sealed predictions: "
        f"**{len(mappings)} arrays over {len(rep3)} rep3 recordings**.\n\n"
        "No repetition-3 cloth or rod coordinate after the causal prefix was "
        "numerically read before this seal.\n",
        encoding="utf-8",
    )


def _verify_prediction_inputs(
    cases: list[Case], protocol: dict[str, Any], seal: dict[str, Any]
) -> dict[str, InputView]:
    expected = {record["case_id"]: record for record in seal["prediction_inputs"]}
    result: dict[str, InputView] = {}
    for case in cases:
        inputs = prediction_input(case, protocol)
        descriptor = _input_descriptor(inputs)
        if descriptor != expected.get(case.case_id):
            raise ValueError(f"prediction input changed for {case.case_id}")
        result[case.case_id] = inputs
    return result


def score_stage(
    dataset_root: Path,
    prediction: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = load_json(protocol_path)
    _verify_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    seal = load_json(prediction / "prediction_seal.json")
    policy = load_json(prediction / "policy.json")
    source_result = load_json(prediction / "source_result.json")
    private_path = prediction / "private_predictions.npz"
    if file_digest(private_path) != seal["prediction_file_sha256"]:
        raise ValueError("private prediction file differs from the published seal")
    if seal["policy_id"] != policy["policy_id"]:
        raise ValueError("prediction policy identity changed")
    if seal["source_result_id"] != source_result["result_id"]:
        raise ValueError("prediction source identity changed")

    cases, inventory = audit_dataset(dataset_root, protocol)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("confirmation inventory changed")
    rep3 = [case for case in cases if case.repetition == 3]
    inputs_by_case = _verify_prediction_inputs(rep3, protocol, seal)
    array_records = {
        (item["case_id"], item["arm"]): item for item in seal["prediction_arrays"]
    }
    rows: list[dict[str, Any]] = []
    with np.load(private_path, allow_pickle=False) as arrays:
        for case in rep3:
            inputs = inputs_by_case[case.case_id]
            predictions: dict[str, np.ndarray] = {}
            required_arms = sorted(
                {name for selector in SELECTOR_ARMS.values() for name in selector}
            )
            for arm in required_arms:
                item = array_records[(case.case_id, arm)]
                value = np.asarray(arrays[item["array_key"]], dtype=float)
                shape_changed = list(value.shape) != item["shape"]
                digest_changed = array_digest(value) != item["sha256"]
                if shape_changed or digest_changed:
                    raise ValueError(
                        f"sealed prediction changed for {case.case_id}/{arm}"
                    )
                predictions[arm] = value
            # This is the first numerical access to rep3 future cloth outcomes.
            truth = scoring_truth(case, inputs)
            rows.extend(score_case(predictions, truth, inputs, protocol))

    policy_rows = apply_policy(rows, policy)
    summaries = summarize_policy_rows(policy_rows, protocol, seed_offset=1000)
    incremental = incremental_summary(policy_rows, protocol, seed_offset=1000)
    gate = confirmation_gate(summaries, incremental, protocol)
    decision = (
        "fresh-repetition-confirmation-pass"
        if gate["pass"]
        else "fresh-repetition-confirmation-fail"
    )
    result = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-confirmation.v1",
        "schema_version": 1,
        "decision": decision,
        "inventory_id": inventory["inventory_id"],
        "source_result_id": source_result["result_id"],
        "policy_id": policy["policy_id"],
        "prediction_seal_id": seal["seal_id"],
        "summaries": summaries,
        "incremental": incremental,
        "confirmation_gate": gate,
        "rep3_numeric_outcomes_read_after_joint_prediction_seal": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = object_digest(result)
    write_json(output / "result.json", result)
    write_json(output / "prediction_seal.json", seal)
    write_json(output / "protocol.json", protocol)
    write_json(output / "dataset_manifest.json", inventory)
    _write_flat_csv(output / "rep3_query_cases.csv", rows)
    _write_flat_csv(output / "rep3_policy_cases.csv", policy_rows)
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"], encoding="utf-8"
    )

    physics = summaries["physics_enabled"]
    residual = summaries["matched_residual"]
    report = [
        "# Fresh Tracking Cloth self-collision confirmation",
        "",
        f"Decision: **{decision}**",
        "",
        "The handoff policy was selected on rep2 and all rep3 predictions were",
        "jointly sealed before rep3 future cloth outcomes were numerically read.",
        "",
        "## Primary matched comparison",
        "",
        "| Selector | Loss [mm] | Nonfallback coverage | Physics coverage "
        "| Practical harm |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| matched residual | {residual['selected_loss_mm']:.4f} | "
        f"{100 * residual['nonfallback_coverage']:.2f}% | 0.00% | "
        f"{100 * residual['selected_practical_harm_fraction_all_cases']:.2f}% |",
        f"| physics enabled | {physics['selected_loss_mm']:.4f} | "
        f"{100 * physics['nonfallback_coverage']:.2f}% | "
        f"{100 * physics['physics_coverage']:.2f}% | "
        f"{100 * physics['selected_practical_harm_fraction_all_cases']:.2f}% |",
        "",
        "Physics-enabled minus matched residual: "
        f"**{incremental['physics_minus_residual_mm']:.6f} mm**.",
        "Incremental relative gain: "
        f"**{100 * incremental['incremental_relative_gain']:.3f}%**.",
        "Material-bootstrap 95% interval: "
        f"**[{incremental['material_bootstrap_95_interval_mm'][0]:.6f}, "
        f"{incremental['material_bootstrap_95_interval_mm'][1]:.6f}] mm**.",
        "Physics-use harmful fraction: "
        f"**{100 * physics['physics_practical_harm_fraction']:.2f}%**; "
        "exact one-sided 95% binomial endpoint: "
        f"**{100 * physics['physics_harm_upper_95']:.2f}%**.",
        "",
        "## Frozen confirmation criteria",
        "",
    ]
    report.extend(
        f"- `{name}`: **{'pass' if passed else 'fail'}**"
        for name, passed in gate["criteria"].items()
    )
    report.extend(["", "## Boundary", "", protocol["claim_boundary"]])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        output / "run_manifest.json",
        {
            "schema": "bayesian-phystwin.tracking-cloth-self-collision-score-run.v1",
            "created_at": now(),
            "stage": "score",
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "result_id": result["result_id"],
            "prediction_seal_id": seal["seal_id"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("source", "predict", "score"), required=True
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    if args.stage == "source":
        source_stage(args.dataset_root, args.output, args.protocol)
    elif args.stage == "predict":
        if args.source is None:
            parser.error("--source is required for predict")
        predict_stage(args.dataset_root, args.source, args.output, args.protocol)
    else:
        if args.prediction is None:
            parser.error("--prediction is required for score")
        score_stage(args.dataset_root, args.prediction, args.output, args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
