#!/usr/bin/env python3
"""Compute the exact source-selected PokeFlex held-out sign-flip diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when frozen input or analysis contracts differ."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_id(payload: Mapping[str, object]) -> str:
    value = dict(payload)
    value.pop("result_id", None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def run(protocol_path: Path, revision: str) -> dict[str, object]:
    protocol = read_json(protocol_path.resolve())
    require(
        protocol.get("schema")
        == "bayesian-phystwin/pokeflex-source-selected-transfer-sign-test-v1",
        "protocol schema changed",
    )
    require(protocol.get("schema_version") == 1, "protocol version changed")
    require(
        protocol.get("status") == "retrospective-post-outcome-diagnostic",
        "protocol status changed",
    )
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

    repository_root = protocol_path.resolve().parents[1]
    registered = protocol["input"]
    target_path = (repository_root / registered["target_result"]).resolve()
    require(target_path.is_file(), "target result is missing")
    observed_sha = file_sha256(target_path)
    require(observed_sha == registered["target_result_sha256"], "target bytes changed")
    target = read_json(target_path)
    require(
        target.get("barrier_sha256") == registered["target_result_barrier_sha256"],
        "prediction barrier identity changed",
    )
    require("retrospective" in str(target.get("claim_boundary", "")).casefold(), "retrospective boundary missing")

    raw_objects = target.get("objects")
    require(type(raw_objects) is list and len(raw_objects) == 13, "target cohort changed")
    objects = {
        str(row["take_id"]): row
        for row in raw_objects
        if type(row) is dict and type(row.get("take_id")) is str
    }
    multipliers = {
        str(key): float(value)
        for key, value in protocol["source_selected_multiplier"].items()
    }
    require(set(objects) == set(multipliers), "target identity roster changed")

    rows: list[dict[str, object]] = []
    adjusted_gains: list[float] = []
    candidate_scores: list[float] = []
    global_scores: list[float] = []
    for take_id, multiplier in multipliers.items():
        row = objects[take_id]
        candidate = float(row["candidate_mean_CD_UL1_mm"])
        global_score = float(row["global_candidate_mean_CD_UL1_mm"])
        require(math.isfinite(candidate) and math.isfinite(global_score), "non-finite score")
        gain = global_score - candidate
        registered_gain = -float(row["action_robust_minus_global_mean_CD_UL1_mm"])
        require(
            math.isclose(gain, registered_gain, rel_tol=0.0, abs_tol=1e-12),
            f"contrast mismatch: {take_id}",
        )
        adjusted = multiplier != 1.0
        if adjusted:
            require(abs(gain) > 1e-12, f"adjusted object has zero gain: {take_id}")
            adjusted_gains.append(gain)
        else:
            require(abs(gain) <= 1e-12, f"global fallback is not an exact tie: {take_id}")
        candidate_scores.append(candidate)
        global_scores.append(global_score)
        rows.append(
            {
                "take_id": take_id,
                "source_selected_multiplier": multiplier,
                "adjusted": adjusted,
                "global_candidate_mean_CD_UL1_mm": global_score,
                "same_object_candidate_mean_CD_UL1_mm": candidate,
                "heldout_incremental_gain_mm": gain,
                "heldout_relative_incremental_gain": gain / global_score,
            }
        )

    require(len(adjusted_gains) == 6, "adjusted-object count changed")
    observed = float(sum(adjusted_gains))
    null_statistics = [
        float(sum(sign * gain for sign, gain in zip(signs, adjusted_gains, strict=True)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(adjusted_gains))
    ]
    require(len(null_statistics) == 64, "sign-flip count changed")
    tolerance = 1e-12
    no_worse = sum(value >= observed - tolerance for value in null_statistics)
    strictly_better = sum(value > observed + tolerance for value in null_statistics)
    win_count = sum(gain > 0.0 for gain in adjusted_gains)
    loss_count = sum(gain < 0.0 for gain in adjusted_gains)
    require(win_count + loss_count == 6, "adjusted contrast contains a tie")

    candidate_mean = float(sum(candidate_scores) / len(candidate_scores))
    global_mean = float(sum(global_scores) / len(global_scores))
    payload: dict[str, object] = {
        "schema": "bayesian-phystwin/pokeflex-source-selected-transfer-sign-test-result-v1",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": revision,
        "input_target_result_sha256": observed_sha,
        "physical_target_take_count": 13,
        "source_selected_non_global_take_count": 6,
        "per_take": rows,
        "primary": {
            "observed_sum_heldout_incremental_gain_mm": observed,
            "observed_mean_heldout_incremental_gain_mm": observed / 6.0,
            "win_count": win_count,
            "loss_count": loss_count,
            "sign_flip_assignment_count": 64,
            "strictly_better_assignment_count": strictly_better,
            "no_worse_assignment_count": no_worse,
            "exact_one_sided_sign_flip_p_value": no_worse / 64.0,
            "exact_rank_larger_is_better": strictly_better + 1,
            "passes_registered_alpha_0_05": no_worse / 64.0 <= 0.05,
        },
        "secondary": {
            "minimum_adjusted_gain_mm": min(adjusted_gains),
            "maximum_adjusted_gain_mm": max(adjusted_gains),
            "same_object_candidate_object_balanced_CD_UL1_mm": candidate_mean,
            "global_candidate_object_balanced_CD_UL1_mm": global_mean,
            "all_13_relative_improvement_over_global": (
                global_mean - candidate_mean
            )
            / global_mean,
        },
        "information_boundary": boundary,
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["result_id"] = result_id(payload)
    return payload


def write_outputs(output: Path, payload: Mapping[str, object]) -> None:
    require(not output.exists(), "output path already exists")
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    primary = payload["primary"]
    secondary = payload["secondary"]
    lines = [
        "# PokeFlex source-selected held-out transfer sign test",
        "",
        f"- Result ID: `{payload['result_id']}`",
        f"- Adjusted objects: `{payload['source_selected_non_global_take_count']}`",
        f"- Held-out wins/losses: `{primary['win_count']}/{primary['loss_count']}`",
        f"- Mean incremental gain: `{primary['observed_mean_heldout_incremental_gain_mm']:.9f} mm`",
        f"- Exact sign-flip p-value: `{primary['exact_one_sided_sign_flip_p_value']:.8f}`",
        f"- Exact rank: `{primary['exact_rank_larger_is_better']}/{primary['sign_flip_assignment_count']}`",
        f"- All-13 improvement over global: `{100.0 * secondary['all_13_relative_improvement_over_global']:.6f}%`",
        f"- Registered 0.05 gate: `{str(primary['passes_registered_alpha_0_05']).lower()}`",
        "",
        str(payload["claim_boundary"]),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = [
        f"{file_sha256(path)}  {path.name}"
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def self_test() -> None:
    gains = (1.0, 2.0, 3.0)
    values = [
        sum(sign * gain for sign, gain in zip(signs, gains, strict=True))
        for signs in itertools.product((-1.0, 1.0), repeat=3)
    ]
    require(len(values) == 8, "self-test assignment count")
    require(sum(value >= 6.0 for value in values) == 1, "self-test tail count")
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "run"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-revision")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "self-test":
        self_test()
        return 0
    require(arguments.protocol is not None, "--protocol is required")
    require(arguments.output is not None, "--output is required")
    revision = arguments.repository_revision
    require(
        type(revision) is str
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "--repository-revision must be a full lowercase SHA",
    )
    payload = run(arguments.protocol, revision)
    write_outputs(arguments.output, payload)
    print(json.dumps({"result_id": payload["result_id"], **payload["primary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"PokeFlex sign test failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
