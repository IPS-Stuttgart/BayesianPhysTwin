#!/usr/bin/env python3
"""Exact conditional sign-flip analysis of PokeFlex held-out transfer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin/pokeflex-source-selected-transfer-sign-test-v1"
)
RESULT_SCHEMA: Final = (
    "bayesian-phystwin/pokeflex-source-selected-transfer-sign-test-result-v1"
)


class AnalysisError(ValueError):
    """Raised when an immutable analysis contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except AnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_id(payload: Mapping[str, object]) -> str:
    value = dict(payload)
    value.pop("result_id", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 1, "unsupported protocol version")
    require(
        protocol.get("protocol_id")
        == "pokeflex-source-selected-transfer-sign-test-v1",
        "protocol ID changed",
    )
    multipliers = protocol.get("source_selected_multiplier")
    require(type(multipliers) is dict and len(multipliers) == 13, "multiplier map changed")
    adjusted = [key for key, value in multipliers.items() if float(value) != 1.0]
    require(len(adjusted) == 6, "adjusted-object count changed")
    test = protocol.get("test")
    require(type(test) is dict, "test contract is missing")
    require(test.get("expected_assignment_count") == 64, "assignment count changed")
    boundary = protocol.get("information_boundary")
    require(
        boundary
        == {
            "fresh_confirmation": False,
            "all_thirteen_target_outcomes_previously_opened": True,
            "source_selection_used_target_outcomes": False,
            "new_model_prediction": False,
            "target_data_reopened": False,
            "checked_in_result_only": True,
            "paper_claim_authorized": False,
        },
        "information boundary changed",
    )
    return protocol


def build_result(protocol_path: Path, repository_revision: str) -> dict[str, object]:
    protocol = validate_protocol(protocol_path)
    repository_root = protocol_path.resolve().parents[1]
    registered = protocol["input"]
    target_path = (repository_root / registered["target_result"]).resolve()
    require(target_path.is_file(), "registered target result is missing")
    observed_sha256 = sha256_file(target_path)
    require(
        observed_sha256 == registered["target_result_sha256"],
        "registered target result bytes changed",
    )
    target = read_json(target_path)
    require(
        target.get("barrier_sha256") == registered["target_result_barrier_sha256"],
        "prediction barrier identity changed",
    )
    require(target.get("target_outcomes_opened_previously") is True, "history changed")
    objects = target.get("objects")
    require(type(objects) is list and len(objects) == 13, "target cohort changed")
    by_take = {
        str(row["take_id"]): row
        for row in objects
        if type(row) is dict and type(row.get("take_id")) is str
    }
    multipliers = {
        str(key): float(value)
        for key, value in protocol["source_selected_multiplier"].items()
    }
    require(set(by_take) == set(multipliers), "target take inventory changed")

    rows: list[dict[str, object]] = []
    gains: list[float] = []
    for take_id, multiplier in multipliers.items():
        row = by_take[take_id]
        candidate = float(row["candidate_mean_CD_UL1_mm"])
        global_candidate = float(row["global_candidate_mean_CD_UL1_mm"])
        require(math.isfinite(candidate) and math.isfinite(global_candidate), "non-finite score")
        gain = global_candidate - candidate
        registered_difference = -float(row["action_robust_minus_global_mean_CD_UL1_mm"])
        require(
            math.isclose(gain, registered_difference, rel_tol=0.0, abs_tol=1e-12),
            f"registered contrast changed: {take_id}",
        )
        adjusted = multiplier != 1.0
        if adjusted:
            require(not math.isclose(gain, 0.0, rel_tol=0.0, abs_tol=1e-12), f"zero adjusted gain: {take_id}")
            gains.append(gain)
        else:
            require(math.isclose(gain, 0.0, rel_tol=0.0, abs_tol=1e-12), f"fallback mismatch: {take_id}")
        rows.append(
            {
                "take_id": take_id,
                "source_selected_multiplier": multiplier,
                "adjusted": adjusted,
                "global_candidate_mean_CD_UL1_mm": global_candidate,
                "same_object_candidate_mean_CD_UL1_mm": candidate,
                "heldout_incremental_gain_mm": gain,
                "heldout_relative_incremental_gain": (
                    gain / global_candidate if global_candidate > 0.0 else None
                ),
            }
        )

    require(len(gains) == 6, "adjusted held-out contrast count changed")
    observed_statistic = float(sum(gains))
    assignments: list[float] = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(gains)):
        assignments.append(float(sum(sign * gain for sign, gain in zip(signs, gains, strict=True))))
    require(len(assignments) == 64, "sign-flip assignment count changed")
    tolerance = 1e-12
    no_worse = sum(value >= observed_statistic - tolerance for value in assignments)
    strictly_better = sum(value > observed_statistic + tolerance for value in assignments)
    win_count = sum(gain > 0.0 for gain in gains)
    loss_count = sum(gain < 0.0 for gain in gains)
    require(win_count + loss_count == len(gains), "adjusted contrast has a tie")

    all_candidate = [float(by_take[take]["candidate_mean_CD_UL1_mm"]) for take in multipliers]
    all_global = [float(by_take[take]["global_candidate_mean_CD_UL1_mm"]) for take in multipliers]
    candidate_mean = float(sum(all_candidate) / len(all_candidate))
    global_mean = float(sum(all_global) / len(all_global))
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": repository_revision,
        "input_target_result_sha256": observed_sha256,
        "physical_target_take_count": len(rows),
        "source_selected_non_global_take_count": len(gains),
        "per_take": rows,
        "primary": {
            "observed_sum_heldout_incremental_gain_mm": observed_statistic,
            "observed_mean_heldout_incremental_gain_mm": float(
                observed_statistic / len(gains)
            ),
            "win_count": win_count,
            "loss_count": loss_count,
            "sign_flip_assignment_count": len(assignments),
            "strictly_better_assignment_count": strictly_better,
            "no_worse_assignment_count": no_worse,
            "exact_one_sided_sign_flip_p_value": float(no_worse / len(assignments)),
            "exact_rank_larger_is_better": strictly_better + 1,
            "passes_registered_alpha_0_05": bool(no_worse / len(assignments) <= 0.05),
        },
        "secondary": {
            "minimum_adjusted_gain_mm": float(min(gains)),
            "maximum_adjusted_gain_mm": float(max(gains)),
            "same_object_candidate_object_balanced_CD_UL1_mm": candidate_mean,
            "global_candidate_object_balanced_CD_UL1_mm": global_mean,
            "all_13_relative_improvement_over_global": float(
                (global_mean - candidate_mean) / global_mean
            ),
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    return result


def write_outputs(output: Path, result: Mapping[str, object]) -> None:
    require(not output.exists(), "output path already exists")
    output.mkdir(parents=True)
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    primary = result["primary"]
    secondary = result["secondary"]
    report = [
        "# PokeFlex source-selected transfer sign test v1",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Source-selected non-global objects: `{result['source_selected_non_global_take_count']}`",
        f"- Held-out wins/losses: `{primary['win_count']}/{primary['loss_count']}`",
        f"- Mean incremental gain: `{primary['observed_mean_heldout_incremental_gain_mm']:.9f} mm`",
        f"- Exact sign-flip p-value: `{primary['exact_one_sided_sign_flip_p_value']:.8f}`",
        f"- Exact rank: `{primary['exact_rank_larger_is_better']}/{primary['sign_flip_assignment_count']}`",
        f"- All-13 relative improvement over global: `{100.0 * secondary['all_13_relative_improvement_over_global']:.6f}%`",
        f"- Registered 0.05 gate: `{str(primary['passes_registered_alpha_0_05']).lower()}`",
        "",
        str(result["claim_boundary"]),
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def self_test() -> None:
    gains = [1.0, 2.0, 3.0]
    values = [
        sum(sign * gain for sign, gain in zip(signs, gains, strict=True))
        for signs in itertools.product((-1.0, 1.0), repeat=3)
    ]
    require(len(values) == 8, "self-test sign count changed")
    require(sum(value >= 6.0 for value in values) == 1, "self-test exact tail changed")
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repository-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "self-test":
        self_test()
        return 0
    require(
        len(arguments.repository_revision) == 40
        and all(character in "0123456789abcdef" for character in arguments.repository_revision),
        "repository revision must be a full lowercase SHA",
    )
    result = build_result(arguments.protocol, arguments.repository_revision)
    write_outputs(arguments.output, result)
    print(json.dumps({"result_id": result["result_id"], **result["primary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as error:
        print(f"PokeFlex sign test failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
