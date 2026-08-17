#!/usr/bin/env python3
"""Build and audit the target-disjoint PokeFlex causal-scale V6 model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_missing5_causal_scale_v6 import (  # noqa: E402
    V6_CANDIDATE_SCALES,
    CausalScalePolicyConfig,
    build_causal_scale_model,
    extract_source_frame_rows,
    fit_object_model,
    select_fitted_feature,
)
from bayesian_phystwin.pokeflex_missing5_scale import (  # noqa: E402
    SOURCE_TAKES,
    file_sha256,
)

RESULT_KIND = "PokeFlexMissingFiveCausalScaleV6SourceResult"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _source_payloads(
    source_root: Path, v5_result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_hashes = dict(v5_result.get("artifact_file_sha256s", {}))
    expected_takes = sorted(take for takes in SOURCE_TAKES.values() for take in takes)
    _require(sorted(expected_hashes) == expected_takes, "V5 source inventory changed")
    payloads = []
    observed_hashes = {}
    for take_id in expected_takes:
        path = source_root / f"{take_id}.json"
        _require(path.is_file(), f"missing source artifact: {take_id}")
        digest = file_sha256(path)
        _require(
            digest == expected_hashes[take_id], f"source artifact changed: {take_id}"
        )
        payload = _load_json(path)
        _require(payload.get("take", {}).get("id") == take_id, "source take changed")
        payloads.append(payload)
        observed_hashes[take_id] = digest
    return payloads, observed_hashes


def _rows_by_object(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        object_name: [] for object_name in V6_CANDIDATE_SCALES
    }
    for payload in payloads:
        for row in extract_source_frame_rows(payload):
            rows[str(row["object_name"])].append(row)
    for object_name, object_rows in rows.items():
        _require(bool(object_rows), f"source rows are missing: {object_name}")
    return rows


def _evaluate_fold(
    object_rows: Sequence[Mapping[str, Any]],
    held_take_ids: set[str],
    config: CausalScalePolicyConfig,
) -> dict[str, dict[str, Any]]:
    take_ids = sorted({str(row["take_id"]) for row in object_rows})
    training_take_ids = [
        take_id for take_id in take_ids if take_id not in held_take_ids
    ]
    training = [row for row in object_rows if row["take_id"] in training_take_ids]
    model = fit_object_model(
        training,
        config=config,
        expected_take_ids=training_take_ids,
    )
    result = {}
    for held_take_id in sorted(held_take_ids):
        held = [row for row in object_rows if row["take_id"] == held_take_id]
        baseline = np.asarray(
            [row["baseline_CD_UL1_mm"] for row in held], dtype=np.float64
        )
        gains = np.asarray([row["candidate_gain_mm"] for row in held], dtype=np.float64)
        admitted = np.asarray(
            [
                select_fitted_feature(
                    model,
                    config,
                    np.asarray(row["features"], dtype=np.float64),
                    supported=bool(row["accepted"]),
                ).admitted
                for row in held
            ],
            dtype=np.bool_,
        )
        absolute_gain = float(np.mean(np.where(admitted, gains, 0.0)))
        relative_gain = absolute_gain / float(np.mean(baseline))
        result[held_take_id] = {
            "relative_gain": relative_gain,
            "absolute_gain_mm": absolute_gain,
            "admitted_frame_count": int(np.sum(admitted)),
        }
    return result


def _leave_one_out(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
    config: CausalScalePolicyConfig,
) -> list[dict[str, Any]]:
    leave_one_out = []
    for object_name, object_rows in rows_by_object.items():
        take_ids = sorted({str(row["take_id"]) for row in object_rows})
        for take_id in take_ids:
            result = _evaluate_fold(object_rows, {take_id}, config)[take_id]
            leave_one_out.append(
                {"object_name": object_name, "held_take_id": take_id, **result}
            )
    return leave_one_out


def _cross_validation(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
    config: CausalScalePolicyConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    leave_one_out = _leave_one_out(rows_by_object, config)
    leave_two_summary = {}
    for object_name, object_rows in rows_by_object.items():
        take_ids = sorted({str(row["take_id"]) for row in object_rows})
        leave_two_rows = []
        for held_pair in itertools.combinations(take_ids, 2):
            results = _evaluate_fold(object_rows, set(held_pair), config)
            for take_id in held_pair:
                leave_two_rows.append(results[take_id])
        gains = np.asarray(
            [row["relative_gain"] for row in leave_two_rows], dtype=np.float64
        )
        leave_two_summary[object_name] = {
            "evaluation_count": len(gains),
            "strict_win_count": int(np.sum(gains > 1e-12)),
            "regression_count": int(np.sum(gains < -1e-12)),
            "mean_relative_gain": float(np.mean(gains)),
            "minimum_relative_gain": float(np.min(gains)),
        }
    return leave_one_out, leave_two_summary


def _sensitivity(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    summaries = []
    for neighbors, margin, distance_quantile in itertools.product(
        (10, 15, 20, 30),
        (0.001, 0.003, 0.005),
        (0.85, 0.90, 0.95),
    ):
        config = CausalScalePolicyConfig(
            neighbors_per_source_take=neighbors,
            gain_margin_mm=margin,
            support_distance_quantile=distance_quantile,
        )
        loo = _leave_one_out(rows_by_object, config)
        gains = np.asarray([row["relative_gain"] for row in loo], dtype=np.float64)
        summaries.append(
            {
                "neighbors_per_source_take": neighbors,
                "gain_margin_mm": margin,
                "support_distance_quantile": distance_quantile,
                "mean_relative_gain": float(np.mean(gains)),
                "minimum_relative_gain": float(np.min(gains)),
                "regression_count": int(np.sum(gains < -1e-12)),
                "admitted_frame_count": sum(
                    int(row["admitted_frame_count"]) for row in loo
                ),
            }
        )
    return {
        "configuration_count": len(summaries),
        "regressing_configuration_count": sum(
            row["regression_count"] > 0 for row in summaries
        ),
        "minimum_relative_gain": min(row["minimum_relative_gain"] for row in summaries),
        "minimum_mean_relative_gain": min(
            row["mean_relative_gain"] for row in summaries
        ),
        "maximum_mean_relative_gain": max(
            row["mean_relative_gain"] for row in summaries
        ),
        "minimum_admitted_frame_count": min(
            row["admitted_frame_count"] for row in summaries
        ),
        "maximum_admitted_frame_count": max(
            row["admitted_frame_count"] for row in summaries
        ),
        "configurations": summaries,
    }


def _permuted_rows(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    result = {}
    for object_name, object_rows in rows_by_object.items():
        copied = [deepcopy(dict(row)) for row in object_rows]
        for take_id in sorted({str(row["take_id"]) for row in copied}):
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(copied)
                    if row["take_id"] == take_id and bool(row["accepted"])
                ],
                dtype=np.int64,
            )
            gains = np.asarray(
                [copied[index]["candidate_gain_mm"] for index in indices],
                dtype=np.float64,
            )
            gains = gains[rng.permutation(len(gains))]
            for index, gain in zip(indices, gains, strict=True):
                copied[int(index)]["candidate_gain_mm"] = float(gain)
        result[object_name] = copied
    return result


def _permutation_control(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_loo: Sequence[Mapping[str, Any]],
    config: CausalScalePolicyConfig,
) -> dict[str, Any]:
    observed = np.asarray(
        [row["relative_gain"] for row in observed_loo], dtype=np.float64
    )
    replicate_count = 1000
    records = []
    for seed in range(replicate_count):
        permuted = _permuted_rows(rows_by_object, seed)
        loo = _leave_one_out(permuted, config)
        gains = np.asarray([row["relative_gain"] for row in loo], dtype=np.float64)
        records.append(
            (
                float(np.mean(gains)),
                float(np.min(gains)),
                int(np.sum(gains < -1e-12)),
                int(np.sum(gains > 1e-12)),
            )
        )
    means = np.asarray([row[0] for row in records], dtype=np.float64)
    minimums = np.asarray([row[1] for row in records], dtype=np.float64)
    regressions = np.asarray([row[2] for row in records], dtype=np.int64)
    wins = np.asarray([row[3] for row in records], dtype=np.int64)
    observed_mean = float(np.mean(observed))
    observed_minimum = float(np.min(observed))
    both = (means >= observed_mean) & (minimums >= observed_minimum)
    return {
        "replicate_count": replicate_count,
        "seed_rule": "integers 0 through 999 inclusive",
        "permutation_unit": "accepted frame gains within each physical take",
        "observed_mean_relative_gain": observed_mean,
        "observed_minimum_relative_gain": observed_minimum,
        "null_mean_relative_gain_median": float(np.quantile(means, 0.50)),
        "null_mean_relative_gain_q95": float(np.quantile(means, 0.95)),
        "null_minimum_relative_gain_median": float(np.quantile(minimums, 0.50)),
        "null_minimum_relative_gain_q95": float(np.quantile(minimums, 0.95)),
        "zero_regression_count": int(np.sum(regressions == 0)),
        "all_take_win_count": int(np.sum(wins == len(observed))),
        "mean_at_least_observed_count": int(np.sum(means >= observed_mean)),
        "minimum_at_least_observed_count": int(np.sum(minimums >= observed_minimum)),
        "mean_and_minimum_at_least_observed_count": int(np.sum(both)),
    }


def _positive_controls(
    rows_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
    config: CausalScalePolicyConfig,
) -> dict[str, Any]:
    controls = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        synthetic = {}
        for object_name, object_rows in rows_by_object.items():
            copied = []
            for row in object_rows:
                value = deepcopy(dict(row))
                phase = float(value["features"][0])
                if phase >= 0.42:
                    region = 1
                    signal = 0.012
                elif phase <= 0.26:
                    region = -1
                    signal = -0.012
                else:
                    region = 0
                    signal = 0.0
                value["synthetic_region"] = region
                value["baseline_CD_UL1_mm"] = 5.0
                value["candidate_gain_mm"] = signal + float(rng.normal(0.0, 0.0005))
                copied.append(value)
            synthetic[object_name] = copied
        loo = _leave_one_out(synthetic, config)
        gains = np.asarray([row["relative_gain"] for row in loo], dtype=np.float64)
        harmful_selected = 0
        for object_rows in synthetic.values():
            take_ids = sorted({str(row["take_id"]) for row in object_rows})
            for held_take_id in take_ids:
                training_take_ids = [
                    value for value in take_ids if value != held_take_id
                ]
                model = fit_object_model(
                    [row for row in object_rows if row["take_id"] in training_take_ids],
                    config=config,
                    expected_take_ids=training_take_ids,
                )
                for row in object_rows:
                    if row["take_id"] != held_take_id:
                        continue
                    decision = select_fitted_feature(
                        model,
                        config,
                        np.asarray(row["features"], dtype=np.float64),
                        supported=bool(row["accepted"]),
                    )
                    harmful_selected += int(
                        decision.admitted and int(row["synthetic_region"]) == -1
                    )
        passed = bool(
            np.all(gains > 1e-12) and harmful_selected == 0 and len(gains) == 12
        )
        controls.append(
            {
                "control_index": seed + 1,
                "passed": passed,
                "mean_relative_gain": float(np.mean(gains)),
                "minimum_relative_gain": float(np.min(gains)),
                "harmful_region_admission_count": harmful_selected,
            }
        )
    return {
        "control_count": len(controls),
        "passed_count": sum(row["passed"] for row in controls),
        "harmful_region_admission_count": sum(
            row["harmful_region_admission_count"] for row in controls
        ),
        "minimum_relative_gain": min(row["minimum_relative_gain"] for row in controls),
        "controls": controls,
    }


def _parent_bindings(
    v5_source_result_path: Path,
    v5_source_result: Mapping[str, Any],
    module_path: Path,
) -> dict[str, str]:
    paths = {
        "v5_source_result_file_sha256": v5_source_result_path,
        "v5_source_protocol_file_sha256": (
            ROOT / "configs" / "sota" / "pokeflex_missing5_scale_source_v5.json"
        ),
        "v5_completion_protocol_file_sha256": (
            ROOT / "configs" / "sota" / "pokeflex_missing5_scale_completion_v5.json"
        ),
        "v5_execution_protocol_file_sha256": (
            ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
        ),
        "v6_policy_module_file_sha256": module_path,
        "v6_source_builder_file_sha256": Path(__file__),
    }
    bindings = {name: file_sha256(path) for name, path in paths.items()}
    bindings["v5_source_result_sha256"] = str(v5_source_result["result_sha256"])
    bindings["v5_source_protocol_sha256"] = str(v5_source_result["protocol_sha256"])
    return bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_artifact_dir", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("output_result", type=Path)
    parser.add_argument(
        "--v5-source-result",
        type=Path,
        default=(
            ROOT
            / "results"
            / "sota"
            / "pokeflex_missing5_scale_source_v5"
            / "source_result.json"
        ),
    )
    args = parser.parse_args()

    v5_result_path = args.v5_source_result.resolve()
    v5_result = _load_json(v5_result_path)
    _require(v5_result.get("official_target_outcomes_used") is False, "target opened")
    _require(v5_result.get("held_v8_accessed") is False, "held-v8 was accessed")
    payloads, artifact_hashes = _source_payloads(
        args.source_artifact_dir.resolve(), v5_result
    )
    rows_by_object = _rows_by_object(payloads)
    config = CausalScalePolicyConfig()
    leave_one_out, leave_two = _cross_validation(rows_by_object, config)
    sensitivity = _sensitivity(rows_by_object)
    permutation = _permutation_control(rows_by_object, leave_one_out, config)
    positive = _positive_controls(rows_by_object, config)
    loo_gains = np.asarray(
        [row["relative_gain"] for row in leave_one_out], dtype=np.float64
    )
    leave_two_regressions = sum(
        int(row["regression_count"]) for row in leave_two.values()
    )
    source_gate = {
        "leave_one_out_take_count": len(leave_one_out),
        "leave_one_out_strict_win_count": int(np.sum(loo_gains > 1e-12)),
        "leave_one_out_regression_count": int(np.sum(loo_gains < -1e-12)),
        "leave_one_out_mean_relative_gain": float(np.mean(loo_gains)),
        "leave_one_out_minimum_relative_gain": float(np.min(loo_gains)),
        "leave_two_out_evaluation_count": sum(
            int(row["evaluation_count"]) for row in leave_two.values()
        ),
        "leave_two_out_regression_count": leave_two_regressions,
        "sensitivity_regressing_configuration_count": sensitivity[
            "regressing_configuration_count"
        ],
        "permutation_mean_and_minimum_at_least_observed_count": permutation[
            "mean_and_minimum_at_least_observed_count"
        ],
        "positive_controls_passed": positive["passed_count"],
        "positive_control_harmful_region_admission_count": positive[
            "harmful_region_admission_count"
        ],
    }
    source_gate["passed"] = bool(
        source_gate["leave_one_out_take_count"] == 12
        and source_gate["leave_one_out_strict_win_count"] == 12
        and source_gate["leave_one_out_regression_count"] == 0
        and source_gate["leave_two_out_evaluation_count"] == 60
        and source_gate["leave_two_out_regression_count"] == 0
        and source_gate["sensitivity_regressing_configuration_count"] == 0
        and source_gate["permutation_mean_and_minimum_at_least_observed_count"] == 0
        and source_gate["positive_controls_passed"] == 12
        and source_gate["positive_control_harmful_region_admission_count"] == 0
    )
    bindings = _parent_bindings(
        v5_result_path,
        v5_result,
        ROOT / "src" / "bayesian_phystwin" / "pokeflex_missing5_causal_scale_v6.py",
    )
    model = build_causal_scale_model(
        payloads,
        source_artifact_file_sha256s=artifact_hashes,
        parent_bindings=bindings,
        source_gate=source_gate,
    )
    _write_json(args.output_model.resolve(), model)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "protocol_id": "pokeflex-missing5-causal-scale-v6-source",
        "claim_boundary": (
            "Exploratory target-disjoint source calibration. The five official "
            "target archives and outcomes remained unavailable and unopened."
        ),
        "policy": config.as_dict(),
        "promoted_transitions": {
            "3dPrintedCylinder": "effective scale 0.25 to 0.375",
            "3dPrintedHeart": "effective scale 0.1875 to 0.25",
        },
        "exact_v5_fallback_objects": ["3dPrintedPizza", "Pillow", "Sponge"],
        "source_gate": source_gate,
        "leave_one_out": leave_one_out,
        "leave_two_out": leave_two,
        "sensitivity": sensitivity,
        "permutation_control": permutation,
        "positive_controls": positive,
        "model_sha256": model["model_sha256"],
        "model_file_sha256": file_sha256(args.output_model.resolve()),
        "source_artifact_file_sha256s": artifact_hashes,
        "parent_bindings": bindings,
        "official_target_outcomes_used": False,
        "held_v8_accessed": False,
        "result_sha256": "",
    }
    result["result_sha256"] = _canonical_sha256(result, "result_sha256")
    _write_json(args.output_result.resolve(), result)
    print(
        json.dumps(
            {
                "model_sha256": model["model_sha256"],
                "result_sha256": result["result_sha256"],
                "source_gate_passed": source_gate["passed"],
                "official_target_outcomes_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
