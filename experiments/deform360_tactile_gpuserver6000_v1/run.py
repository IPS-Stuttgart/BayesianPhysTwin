#!/usr/bin/env python3
"""Evaluate the exact synchronized Deform360 tactile roster on gpuserver6000."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.deform360_real_v1.run import (
    Carrier,
    Profile,
    aggregate,
    content_id,
    evaluate_sequence,
    load_protocol,
    load_tactile,
    save_csv,
    write_json,
)


def _exact_paths(
    root: Path, protocol: Mapping[str, Any]
) -> list[tuple[int, str, Path]]:
    roster = protocol.get("exact_tactile_roster")
    if not isinstance(roster, Mapping):
        raise ValueError("exact_tactile_roster is missing")
    required = {
        "source_object_id",
        "episode_ids",
        "sensor_names",
        "relative_path_template",
        "expected_case_count",
    }
    if set(roster) != required:
        raise ValueError("exact tactile roster keys changed")
    source_object = str(roster["source_object_id"])
    if source_object != "001-rope":
        raise ValueError("source object differs from the reviewed 001-rope case")
    episodes = tuple(int(value) for value in roster["episode_ids"])
    if episodes != tuple(range(10)):
        raise ValueError("episode roster must remain exactly 0 through 9")
    sensors = tuple(str(value) for value in roster["sensor_names"])
    if len(sensors) != 4 or len(set(sensors)) != len(sensors):
        raise ValueError("sensor roster must contain four unique sensors")
    template = str(roster["relative_path_template"])
    rows = [
        (
            episode_id,
            sensor_name,
            root
            / template.format(
                episode_id=episode_id,
                sensor_name=sensor_name,
            ),
        )
        for episode_id in episodes
        for sensor_name in sensors
    ]
    if len(rows) != int(roster["expected_case_count"]):
        raise ValueError("exact tactile case count changed")
    return rows


def _report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]["raw_tactile_field"]
    errors = summary["mean_primary_error"]
    calibration = summary["bayesian_calibration"]
    return "\n".join(
        [
            "# Deform360 gpuserver6000 synchronized-tactile diagnostic",
            "",
            f"Protocol: `{result['protocol_id']}`",
            f"Revision: `{result.get('revision')}`",
            f"Dataset root: `{result['data_root']}`",
            "",
            "## Exact evaluated roster",
            "",
            f"- Source object: `{result['source_object_id']}`",
            f"- Episodes: `{result['episode_ids']}`",
            f"- Sensors: `{result['sensor_names']}`",
            f"- Evaluated cases: `{summary['case_count']}`",
            "",
            "## Rolling tactile-field results",
            "",
            f"- Persistence RMSE: `{errors['persistence']:.6g}`",
            f"- Last-residual RMSE: `{errors['last_residual']:.6g}`",
            f"- Bayesian RMSE: `{errors['bayesian']:.6g}`",
            (
                "- Bayesian minus best baseline: "
                f"`{summary['bayesian_minus_best_baseline']:.6g}`"
            ),
            (
                "- Episode-balanced Bayesian minus best baseline: "
                f"`{summary['object_balanced_bayesian_minus_best_baseline']:.6g}`"
            ),
            (
                "- Joint normalized NEES: "
                f"`{calibration['joint_nees_normalized']:.6g}`"
            ),
            (
                "- Marginal 90% coverage: "
                f"`{calibration['marginal_90_coverage']:.6g}`"
            ),
            "",
            "## Interpretation boundary",
            "",
            "This evaluates synchronized measured tactile-field dynamics only. It is",
            "not a 3-D/4-D geometry result, not a causal-intervention validation, not",
            "a uniform Deform360 benchmark, and not a fresh-confirmation or paper claim.",
            "The three public objects with ambiguous tactile baselines remain excluded.",
            "",
        ]
    )


def run(
    *,
    data_root: Path,
    protocol_path: Path,
    output_dir: Path,
    profile_name: str,
    revision: str | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    root = data_root.expanduser().resolve(strict=True)
    expected_root = Path(protocol["dataset_root"]).resolve()
    if root != expected_root:
        raise ValueError(f"dataset root changed: {root} != {expected_root}")
    if profile_name not in protocol["profiles"]:
        raise ValueError("unknown evaluation profile")
    profile = Profile.from_mapping(protocol["profiles"][profile_name])
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset roots must be disjoint")
    output.mkdir(parents=True)
    shutil.copy2(protocol_path, output / "protocol.json")

    roster = _exact_paths(root, protocol)
    identities_before: dict[str, tuple[int, int]] = {}
    inventory_rows: list[dict[str, Any]] = []
    for episode_id, sensor_name, path in roster:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"exact synchronized tactile file is missing: {path}")
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        identities_before[relative] = (stat.st_size, stat.st_mtime_ns)
        inventory_rows.append(
            {
                "episode_id": episode_id,
                "sensor_name": sensor_name,
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    write_json(
        output / "carrier_inventory.json",
        {
            "selection_uses_payload_values": False,
            "selection_uses_achieved_scores": False,
            "source_object_id": protocol["exact_tactile_roster"]["source_object_id"],
            "reserved_objects": protocol["reserved_objects"],
            "expected_case_count": len(roster),
            "selected": inventory_rows,
        },
    )

    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for index, (episode_id, sensor_name, path) in enumerate(roster):
        try:
            carrier = Carrier(
                "tactile",
                f"episode_{episode_id:04d}",
                path,
            )
            data = load_tactile(carrier, profile, root)
            metrics, steps, _ = evaluate_sequence(data, protocol["model"])
            case = {
                "case_id": f"case-{index:03d}",
                "kind": "synchronized_tactile",
                "object_id": f"episode_{episode_id:04d}",
                "source_object_id": protocol["exact_tactile_roster"][
                    "source_object_id"
                ],
                "episode_id": episode_id,
                "sensor_name": sensor_name,
                "representation": data.representation,
                "unit": data.unit,
                "primary_metric": data.primary_metric,
                "frame_count": int(data.values.shape[0]),
                "channel_count": int(data.values.shape[1]),
                "step_count": len(steps),
                "metrics": metrics,
                "provenance": dict(data.metadata),
            }
            cases.append(case)
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "source_object_id": case["source_object_id"],
                    "episode_id": episode_id,
                    "sensor_name": sensor_name,
                    "representation": case["representation"],
                    "unit": case["unit"],
                    **metrics,
                }
            )
            for row in steps:
                step_rows.append(
                    {
                        "case_id": case["case_id"],
                        "episode_id": episode_id,
                        "sensor_name": sensor_name,
                        **row,
                    }
                )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            np.linalg.LinAlgError,
        ) as error:
            failures.append(
                {
                    "episode_id": episode_id,
                    "sensor_name": sensor_name,
                    "relative_path": path.relative_to(root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if failures or len(cases) != len(roster):
        write_json(output / "failures.json", failures)
        raise RuntimeError(
            f"exact tactile roster was not fully evaluated: {len(cases)}/{len(roster)}"
        )

    for _, _, path in roster:
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        if identities_before[relative] != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"tactile source changed while being evaluated: {relative}")

    exact = protocol["exact_tactile_roster"]
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-tactile-dynamics-result-v1",
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "profile": profile_name,
        "revision": revision,
        "data_root": str(root),
        "source_object_id": exact["source_object_id"],
        "episode_ids": exact["episode_ids"],
        "sensor_names": exact["sensor_names"],
        "official_processing_revision": protocol["official_processing_revision"],
        "information_boundary": protocol["information_boundary"],
        "selection": {
            "selected_count": len(roster),
            "evaluated_count": len(cases),
            "failure_count": 0,
            "reserved_object_overlap": [],
        },
        "method": {
            "baselines": ["persistence", "last_residual"],
            "candidate": "causal-prefix Gibbs mixture over finite velocity lags",
            "uncertainty": (
                "diagonal residual covariance plus low-rank between-model spread"
            ),
            "velocity_lags": protocol["model"]["velocity_lags"],
        },
        "summary": aggregate(cases, protocol["analysis"]),
        "cases": cases,
        "failures": [],
        "claim_authorized": False,
        "fresh_confirmation_authorized": False,
    }
    result["result_sha256"] = content_id(result)
    write_json(output / "result.json", result)
    save_csv(output / "case_metrics.csv", case_rows)
    save_csv(output / "step_metrics.csv", step_rows)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="pilot")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run(
        data_root=args.data_root,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        profile_name=args.profile,
        revision=os.environ.get("GITHUB_SHA"),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
