#!/usr/bin/env python3
"""Evaluate no-refit DEFORM residual coefficients on sealed PyElastica rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.deform_dlo_cross_backend_transfer_v1 import (
    evaluate_cross_backend_transfer,
    load_cross_backend_transfer_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "run"),
        default="run",
    )
    return parser.parse_args()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve(strict=True)
    expected_size = int(cast(Any, identity.get("size_bytes", -1)))
    expected_digest = str(identity.get("sha256", ""))
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size differs")
    if sha256_file(path) != expected_digest:
        raise ValueError(f"{label} digest differs")
    return path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path, *, expected_count: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source manifest must be a JSON object")
    split = _mapping(value.get("split"), label="source split")
    names = split.get("source_test")
    trajectories = value.get("trajectories")
    if (
        value.get("dlo_type") != "DLO3"
        or value.get("partition") != "train"
        or value.get("official_eval_read") is not False
        or not isinstance(names, list)
        or len(names) != expected_count
        or len(set(names)) != expected_count
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(trajectories, Mapping)
    ):
        raise ValueError("DLO3 source manifest differs")
    return value


def _load_trajectory(path: Path, *, frame_count: int, node_count: int) -> np.ndarray:
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    array = np.asarray(raw, dtype=np.float32)
    expected = (frame_count, 3, node_count)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"{path}: invalid DLO trajectory")
    nodes = np.transpose(array, (0, 2, 1)).astype(np.float64, copy=True)
    nodes[:, :, 2] = np.clip(nodes[:, :, 2], 2e-3 + 1e-6, 10000.0)
    return nodes


def _load_source_trajectories(
    manifest: Mapping[str, object],
    names: Sequence[str],
    *,
    frame_count: int,
    node_count: int,
) -> np.ndarray:
    identities = _mapping(manifest.get("trajectories"), label="trajectories")
    rows = []
    for name in names:
        identity = identities.get(name)
        path = _verified_file(identity, label=f"source trajectory {name}")
        normalized = path.as_posix()
        if "/DLO3/train/" not in normalized or "/eval/" in normalized:
            raise ValueError(f"source trajectory left DLO3/train: {path}")
        rows.append(
            _load_trajectory(
                path,
                frame_count=frame_count,
                node_count=node_count,
            )
        )
    return np.stack(rows)


def _load_model(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return deserialize_deform_local_residual_model(archive)


def _write_csv(
    path: Path,
    names: Sequence[str],
    result: Mapping[str, object],
) -> None:
    primary = _mapping(
        result["primary_vs_raw_pyelastica"],
        label="primary comparison",
    )
    specific = _mapping(
        result["pyelastica_specific_vs_raw_pyelastica"],
        label="specific comparison",
    )
    direct_cases = cast(Sequence[Mapping[str, object]], primary["cases"])
    specific_cases = cast(Sequence[Mapping[str, object]], specific["cases"])
    seed_summaries = _mapping(
        result["individual_seed_vs_raw_pyelastica"],
        label="seed summaries",
    )
    seed_cases = {
        seed: cast(
            Sequence[Mapping[str, object]],
            _mapping(summary, label=f"seed {seed}")["cases"],
        )
        for seed, summary in seed_summaries.items()
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "name",
                "raw_pyelastica_l1_m",
                "pyelastica_specific_l1_m",
                "deform_no_refit_seed_42_l1_m",
                "deform_no_refit_seed_43_l1_m",
                "deform_no_refit_seed_44_l1_m",
                "deform_no_refit_equal_seed_l1_m",
            ],
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "name": name,
                    "raw_pyelastica_l1_m": direct_cases[index][
                        "baseline_l1_m"
                    ],
                    "pyelastica_specific_l1_m": specific_cases[index][
                        "candidate_l1_m"
                    ],
                    "deform_no_refit_seed_42_l1_m": seed_cases["42"][index][
                        "candidate_l1_m"
                    ],
                    "deform_no_refit_seed_43_l1_m": seed_cases["43"][index][
                        "candidate_l1_m"
                    ],
                    "deform_no_refit_seed_44_l1_m": seed_cases["44"][index][
                        "candidate_l1_m"
                    ],
                    "deform_no_refit_equal_seed_l1_m": direct_cases[index][
                        "candidate_l1_m"
                    ],
                }
            )


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    methods = _mapping(result["methods"], label="methods")
    primary = _mapping(
        result["primary_vs_raw_pyelastica"],
        label="primary comparison",
    )
    gate = _mapping(result["promotion_gate"], label="promotion gate")
    specific = _mapping(
        result["pyelastica_specific_vs_raw_pyelastica"],
        label="specific comparison",
    )
    specific_improvement = 100.0 * float(
        cast(Any, specific["relative_improvement"])
    )
    interval = cast(Sequence[float], primary["object_bootstrap_95_interval_m"])
    lines = [
        "# DLO3 no-refit cross-backend coefficient transfer",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Source trajectories: **{result['source_trajectory_count']}**",
        "- Source panel: DLO3 train/source-test; official evaluation unopened",
        "",
        "## Mean coordinate L1",
        "",
        "| Method | Mean L1 (mm) |",
        "|---|---:|",
    ]
    for name, value in methods.items():
        lines.append(f"| `{name}` | {1000.0 * float(cast(Any, value)):.4f} |")
    lines.extend(
        [
            "",
            "## Primary no-refit comparison",
            "",
            (
                "- Equal-seed DEFORM coefficient transfer versus raw PyElastica: "
                f"**{100.0 * float(cast(Any, primary['relative_improvement'])):.2f}%** "
                "relative improvement."
            ),
            (
                "- Trajectory wins/ties/losses: "
                f"**{primary['wins']}/{primary['ties']}/{primary['losses']}**."
            ),
            (
                "- Paired trajectory-bootstrap 95% interval for candidate minus "
                f"backend: **[{1000.0 * interval[0]:.4f}, "
                f"{1000.0 * interval[1]:.4f}] mm**."
            ),
            (
                "- Maximum candidate/backend trajectory ratio: "
                f"**{float(cast(Any, primary['maximum_case_ratio'])):.4f}**."
            ),
            (
                "- Improving DEFORM source models: "
                f"**{gate['improving_seed_models']}/3**."
            ),
            "",
            "## Backend-specific reference",
            "",
            (
                "- PyElastica-specific refit versus raw PyElastica: "
                f"**{specific_improvement:.2f}%**."
            ),
            (
                "- Fraction of its gain retained without coefficient refitting: "
                f"**{result['backend_specific_gain_retained_fraction']}**."
            ),
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_cross_backend_transfer_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = _mapping(protocol["artifacts"], label="artifacts")
    source_manifest_path = _verified_file(
        artifacts["source_manifest"],
        label="source manifest",
    )
    predictions_path = _verified_file(
        artifacts["pyelastica_source_predictions"],
        label="PyElastica source predictions",
    )
    raw_models = cast(
        Sequence[Mapping[str, object]],
        artifacts["deform_local_residual_models"],
    )
    model_paths = {
        int(cast(Any, identity["seed"])): _verified_file(
            identity,
            label=f"DEFORM seed {identity['seed']} model",
        )
        for identity in raw_models
    }
    source_panel = _mapping(protocol["source_panel"], label="source panel")
    expected_count = int(cast(Any, source_panel["trajectory_count"]))
    manifest = _load_manifest(
        source_manifest_path,
        expected_count=expected_count,
    )
    names = cast(list[str], _mapping(manifest["split"], label="split")["source_test"])
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-cross-backend-transfer-preflight-v1",
        "mode": args.mode,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(source_manifest_path),
        "pyelastica_source_predictions": _identity(predictions_path),
        "deform_local_residual_models": {
            str(seed): _identity(path) for seed, path in model_paths.items()
        },
        "source_names": names,
        "source_numeric_payload_opened": False,
        "dlo3_official_evaluation_read": False,
        "dlo4_or_dlo5_read": False,
        "paper_claim_authorized": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-cross-backend-transfer-method-seal-v1",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(source_manifest_path),
        "pyelastica_source_predictions": _identity(predictions_path),
        "deform_local_residual_models": {
            str(seed): _identity(path) for seed, path in model_paths.items()
        },
        "source_names": names,
        "primary_arm": "equal-seed-no-refit-transfer",
        "shrinkage": evaluation["shrinkage"],
        "source_numeric_payload_opened": False,
        "dlo3_official_evaluation_read": False,
        "dlo4_or_dlo5_read": False,
        "target_side_selection": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    with np.load(predictions_path, allow_pickle=False) as archive:
        prediction_names = [
            str(value) for value in np.asarray(archive["names"]).tolist()
        ]
        if prediction_names != names:
            raise ValueError("sealed PyElastica prediction order differs")
        pyelastica_backend = np.asarray(archive["backend"], dtype=np.float64)
        pyelastica_specific = np.asarray(
            archive["candidate"],
            dtype=np.float64,
        )

    trajectories = _load_source_trajectories(
        manifest,
        names,
        frame_count=int(cast(Any, source_panel["frame_count"])),
        node_count=int(cast(Any, source_panel["node_count"])),
    )
    truth = trajectories[:, 2:]
    initial, action = deform_causal_inputs(trajectories)
    shrinkage = float(cast(Any, evaluation["shrinkage"]))
    transferred = {
        seed: predict_deform_local_residual(
            _load_model(path),
            initial,
            action,
            pyelastica_backend,
            shrinkage=shrinkage,
        )["predictions"]
        for seed, path in model_paths.items()
    }
    result = evaluate_cross_backend_transfer(
        names=names,
        truth=truth,
        pyelastica_backend=pyelastica_backend,
        pyelastica_specific_candidate=pyelastica_specific,
        transferred_predictions=transferred,
        protocol=protocol,
    )
    result.update(
        {
            "protocol": _identity(protocol_path),
            "method_seal": _identity(method_seal_path),
            "source_manifest": _identity(source_manifest_path),
            "pyelastica_source_predictions": _identity(predictions_path),
            "deform_local_residual_models": {
                str(seed): _identity(path) for seed, path in model_paths.items()
            },
            "source_payload_loaded_after_method_seal": True,
        }
    )
    _write_json(output_root / "result.json", result)
    _write_csv(output_root / "trajectory-results.csv", names, result)
    _write_report(output_root / "report.md", result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "primary": result["primary_vs_raw_pyelastica"],
                "gain_retained_fraction": result[
                    "backend_specific_gain_retained_fraction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
