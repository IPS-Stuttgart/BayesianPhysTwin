#!/usr/bin/env python3
"""Evaluate one-scalar transport of a fixed DEFORM residual field to PyElastica."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo3_cross_backend_transfer_v1 as direct_runtime

from bayesian_phystwin_experiments.deform_dlo_cross_backend_scalar_v1 import (
    evaluate_cross_backend_scalar_transport,
    load_cross_backend_scalar_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_cross_backend_transfer_v1 import (
    load_cross_backend_transfer_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    predict_deform_local_residual,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    return parser.parse_args()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


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
        or value.get("official_eval_read") is not False
        or not isinstance(names, list)
        or len(names) != expected_count
        or len(set(names)) != expected_count
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(trajectories, Mapping)
        or any(name not in trajectories for name in names)
    ):
        raise ValueError("DLO3 source manifest differs")
    return value


def _write_csv(path: Path, result: Mapping[str, object]) -> None:
    cases = cast(Sequence[Mapping[str, object]], result["cases"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "name",
                "fold_scalar",
                "alignment_cosine",
                "baseline_l1_m",
                "direct_l1_m",
                "scalar_l1_m",
            ],
        )
        writer.writeheader()
        writer.writerows(cases)


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    methods = _mapping(result["methods"], label="methods")
    scalar = _mapping(result["scalar_vs_raw_pyelastica"], label="scalar summary")
    direct = _mapping(result["direct_vs_raw_pyelastica"], label="direct summary")
    alignment = _mapping(result["directional_alignment"], label="alignment")
    fold_scalars = _mapping(result["fold_scalars"], label="fold scalars")
    claim_ladder = _mapping(result["claim_ladder"], label="claim ladder")
    interval = cast(Sequence[float], scalar["object_bootstrap_95_interval_m"])
    lines = [
        "# DLO3 cross-backend one-scalar transport",
        "",
        f"- Decision: **{result['decision']}**",
        "- Interpretation: retrospective complete-trajectory cross-validation",
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
            "## Claim ladder",
            "",
            (
                "- Exact equal-seed no-refit point transfer: "
                f"**{claim_ladder['exact_no_refit_point_transfer_supported']}**"
            ),
            (
                "- Cross-validated one-scalar point transfer: "
                f"**{claim_ladder['one_scalar_cross_validated_point_transfer_supported']}**"
            ),
            (
                "- Directional alignment: "
                f"**{claim_ladder['directional_alignment_supported']}**"
            ),
            (
                "- Shared residual geometry: "
                f"**{claim_ladder['shared_residual_geometry_supported']}**"
            ),
            "",
            "## Registered scalar gate",
            "",
            (
                "- Relative improvement over raw PyElastica: "
                f"**{100.0 * float(cast(Any, scalar['relative_improvement'])):.2f}%**"
            ),
            (
                "- Wins/ties/losses: "
                f"**{scalar['wins']}/{scalar['ties']}/{scalar['losses']}**"
            ),
            (
                "- Maximum trajectory ratio: "
                f"**{float(cast(Any, scalar['maximum_case_ratio'])):.4f}**"
            ),
            (
                "- Fixed-fold paired bootstrap interval, candidate minus raw: "
                f"**[{1000.0 * interval[0]:.4f}, {1000.0 * interval[1]:.4f}] mm**"
            ),
            "",
            "## Residual geometry",
            "",
            (
                "- Positive trajectory alignments: "
                f"**{alignment['positive_cases']}/8**"
            ),
            (
                "- Median alignment cosine: "
                f"**{float(cast(Any, alignment['median_cosine'])):.4f}**"
            ),
            (
                "- Fold-scalar minimum/median/maximum: "
                f"**{float(cast(Any, fold_scalars['minimum'])):.4f} / "
                f"{float(cast(Any, fold_scalars['median'])):.4f} / "
                f"{float(cast(Any, fold_scalars['maximum'])):.4f}**"
            ),
            "",
            "## Direct-transfer reference",
            "",
            (
                "- Direct no-refit relative improvement: "
                f"**{100.0 * float(cast(Any, direct['relative_improvement'])):.2f}%**"
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
    protocol = load_cross_backend_scalar_protocol(protocol_path)
    parent_record = _mapping(protocol["parent"], label="parent")
    repository_root = protocol_path.parent.parent
    parent_protocol_path = (
        repository_root / str(parent_record["protocol"])
    ).resolve(strict=True)
    parent_protocol = load_cross_backend_transfer_protocol(parent_protocol_path)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts = _mapping(parent_protocol["artifacts"], label="parent artifacts")
    source_manifest_path = direct_runtime._verified_file(
        artifacts["source_manifest"], label="source manifest"
    )
    predictions_path = direct_runtime._verified_file(
        artifacts["pyelastica_source_predictions"],
        label="PyElastica source predictions",
    )
    raw_models = cast(
        Sequence[Mapping[str, object]],
        artifacts["deform_local_residual_models"],
    )
    model_paths = {
        int(cast(Any, identity["seed"])): direct_runtime._verified_file(
            identity,
            label=f"DEFORM seed {identity['seed']} model",
        )
        for identity in raw_models
    }
    panel = _mapping(protocol["source_panel"], label="source panel")
    manifest = _load_manifest(
        source_manifest_path,
        expected_count=int(cast(Any, panel["trajectory_count"])),
    )
    names = cast(list[str], _mapping(manifest["split"], label="split")["source_test"])

    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-cross-backend-scalar-preflight-v1",
        "mode": args.mode,
        "protocol": direct_runtime._identity(protocol_path),
        "parent_protocol": direct_runtime._identity(parent_protocol_path),
        "source_manifest": direct_runtime._identity(source_manifest_path),
        "pyelastica_source_predictions": direct_runtime._identity(predictions_path),
        "deform_local_residual_models": {
            str(seed): direct_runtime._identity(path)
            for seed, path in model_paths.items()
        },
        "source_names": names,
        "source_numeric_payload_opened": False,
        "same_trajectory_label_used_for_its_scalar": False,
        "dlo3_official_evaluation_read": False,
        "dlo4_or_dlo5_read": False,
        "paper_claim_authorized": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    transport = _mapping(protocol["transport"], label="transport")
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-cross-backend-scalar-method-seal-v1",
        "protocol": direct_runtime._identity(protocol_path),
        "parent_protocol": direct_runtime._identity(parent_protocol_path),
        "source_manifest": direct_runtime._identity(source_manifest_path),
        "pyelastica_source_predictions": direct_runtime._identity(predictions_path),
        "deform_local_residual_models": {
            str(seed): direct_runtime._identity(path)
            for seed, path in model_paths.items()
        },
        "source_names": names,
        "operator": transport["operator"],
        "minimum_scalar": transport["minimum_scalar"],
        "maximum_scalar": transport["maximum_scalar"],
        "same_trajectory_label_used_for_its_scalar": False,
        "source_numeric_payload_opened": False,
        "dlo3_official_evaluation_read": False,
        "dlo4_or_dlo5_read": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    with np.load(predictions_path, allow_pickle=False) as archive:
        prediction_names = [
            str(value) for value in np.asarray(archive["names"]).tolist()
        ]
        if prediction_names != names:
            raise ValueError("sealed PyElastica prediction order differs")
        baseline = np.asarray(archive["backend"], dtype=np.float64)
        pyelastica_specific = np.asarray(archive["candidate"], dtype=np.float64)

    trajectories = direct_runtime._load_source_trajectories(
        manifest,
        names,
        frame_count=int(cast(Any, panel["frame_count"])),
        node_count=int(cast(Any, panel["node_count"])),
    )
    truth = trajectories[:, 2:]
    initial, action = deform_causal_inputs(trajectories)
    shrinkage = float(cast(Any, parent_record["direct_shrinkage"]))
    transferred = [
        np.asarray(
            predict_deform_local_residual(
                direct_runtime._load_model(model_paths[seed]),
                initial,
                action,
                baseline,
                shrinkage=shrinkage,
            )["predictions"],
            dtype=np.float64,
        )
        for seed in (42, 43, 44)
    ]
    direct_prediction = np.mean(np.stack(transferred), axis=0)
    result = evaluate_cross_backend_scalar_transport(
        names=names,
        truth=truth,
        baseline=baseline,
        direct_prediction=direct_prediction,
        pyelastica_specific_candidate=pyelastica_specific,
        protocol=protocol,
    )
    result.update(
        {
            "protocol": direct_runtime._identity(protocol_path),
            "parent_protocol": direct_runtime._identity(parent_protocol_path),
            "method_seal": direct_runtime._identity(method_seal_path),
            "source_manifest": direct_runtime._identity(source_manifest_path),
            "pyelastica_source_predictions": direct_runtime._identity(predictions_path),
            "deform_local_residual_models": {
                str(seed): direct_runtime._identity(path)
                for seed, path in model_paths.items()
            },
            "source_payload_loaded_after_method_seal": True,
        }
    )
    _write_json(output_root / "result.json", result)
    _write_csv(output_root / "trajectory-results.csv", result)
    _write_report(output_root / "report.md", result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "claim_ladder": result["claim_ladder"],
                "scalar": result["scalar_vs_raw_pyelastica"],
                "alignment": result["directional_alignment"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
