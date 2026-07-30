#!/usr/bin/env python3
"""Evaluate causal PokeFlex translation discrepancy by object-held-out folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_action_discrepancy import (  # noqa: E402
    fit_translation_ridge,
)

ARTIFACT_KIND = "PokeFlexActionDiscrepancyCrossObjectDiagnosticV1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _score_translation(
    baseline_samples_m: np.ndarray,
    target_samples_m: np.ndarray,
    translation_m: np.ndarray,
    scale: float,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    result = []
    for baseline, target, translation in zip(
        baseline_samples_m,
        target_samples_m,
        translation_m,
        strict=True,
    ):
        candidate = baseline + scale * translation[None]
        index = cKDTree(target).query(candidate, k=1)[1]
        result.append(
            1000.0
            * np.mean(
                np.sum(
                    np.abs(candidate - target[np.asarray(index, dtype=np.int64)]),
                    axis=1,
                )
            )
        )
    return np.asarray(result, dtype=np.float64)


def _object_balanced(
    rows: list[dict[str, Any]],
    metric: str,
) -> float:
    by_take: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_take[(row["object"], row["take"])].append(float(row[metric]))
    by_object: dict[str, list[float]] = defaultdict(list)
    for (object_name, _), values in by_take.items():
        by_object[object_name].append(float(np.mean(values)))
    return float(np.mean([np.mean(take_values) for take_values in by_object.values()]))


def _run(arguments: argparse.Namespace) -> int:
    _require(
        arguments.acknowledge_opened_outcomes,
        "evaluation requires --acknowledge-opened-outcomes",
    )
    scales = tuple(float(value) for value in arguments.scales)
    _require(0.0 in scales, "scale bank must include exact fallback zero")
    artifact_paths = [path.resolve() for path in arguments.artifact]
    _require(artifact_paths, "at least one opened-take artifact is required")
    takes: list[dict[str, Any]] = []
    objects: set[str] = set()
    for path in artifact_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        _require(
            metadata.get("artifact_kind") == "PokeFlexActionDiscrepancyOpenedTakeV1",
            f"unexpected artifact kind: {path}",
        )
        _require(
            metadata["take"]["role"] in {"development", "opened_calibration"},
            "artifact role is not opened development/calibration",
        )
        npz_path = Path(metadata["arrays"]["path"])
        _require(_sha256(npz_path) == metadata["arrays"]["sha256"], "NPZ hash changed")
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        frame_count = len(arrays["target_frame"])
        _require(
            arrays["causal_features"].shape[0] == frame_count
            and arrays["oracle_translation_m"].shape == (frame_count, 3)
            and arrays["baseline_samples_m"].shape[0] == frame_count
            and arrays["target_samples_m"].shape == arrays["baseline_samples_m"].shape,
            "take-array shapes are inconsistent",
        )
        object_name = str(metadata["take"]["object"])
        objects.add(object_name)
        takes.append(
            {
                "metadata_path": path,
                "metadata_sha256": _sha256(path),
                "object": object_name,
                "take": str(metadata["take"]["take"]),
                "role": str(metadata["take"]["role"]),
                **arrays,
            }
        )
    _require(len(objects) >= 3, "cross-object evaluation needs at least three objects")

    rows: list[dict[str, Any]] = []
    fold_records = []
    for held_object in sorted(objects):
        training = [take for take in takes if take["object"] != held_object]
        held = [take for take in takes if take["object"] == held_object]
        feature = np.concatenate([take["causal_features"] for take in training])
        target = np.concatenate([take["oracle_translation_m"] for take in training])
        if arguments.feature_family == "constant":
            feature = np.empty((len(feature), 0), dtype=np.float64)
        groups = np.concatenate(
            [np.repeat(take["object"], len(take["target_frame"])) for take in training]
        )
        model = fit_translation_ridge(
            feature,
            target,
            groups,
            ridge_penalty=arguments.ridge_penalty,
            maximum_translation_m=arguments.maximum_translation_mm / 1000.0,
        )
        fold_scale_metrics = {}
        for take in held:
            held_features = take["causal_features"]
            if arguments.feature_family == "constant":
                held_features = np.empty(
                    (len(held_features), 0),
                    dtype=np.float64,
                )
            predicted = model.predict(held_features)
            errors = {
                scale: (
                    np.asarray(take["baseline_cd_ul1_mm"], dtype=np.float64)
                    if scale == 0.0
                    else _score_translation(
                        take["baseline_samples_m"],
                        take["target_samples_m"],
                        predicted,
                        scale,
                    )
                )
                for scale in scales
            }
            oracle_error = np.asarray(
                take["oracle_translation_cd_ul1_mm"],
                dtype=np.float64,
            )
            for index, frame in enumerate(take["target_frame"]):
                row = {
                    "object": held_object,
                    "take": take["take"],
                    "role": take["role"],
                    "target_frame": int(frame),
                    "baseline_cd_ul1_mm": float(take["baseline_cd_ul1_mm"][index]),
                    "oracle_translation_cd_ul1_mm": float(oracle_error[index]),
                    "predicted_translation_norm_mm": float(
                        1000.0 * np.linalg.norm(predicted[index])
                    ),
                }
                for scale, value in errors.items():
                    row[f"scale_{scale:g}_cd_ul1_mm"] = float(value[index])
                rows.append(row)
        for scale in scales:
            current = [row for row in rows if row["object"] == held_object]
            fold_scale_metrics[f"{scale:g}"] = {
                "baseline_mean_cd_ul1_mm": float(
                    np.mean([row["baseline_cd_ul1_mm"] for row in current])
                ),
                "candidate_mean_cd_ul1_mm": float(
                    np.mean([row[f"scale_{scale:g}_cd_ul1_mm"] for row in current])
                ),
            }
        fold_records.append(
            {
                "held_object": held_object,
                "training_object_count": len(objects) - 1,
                "held_take_count": len(held),
                "scales": fold_scale_metrics,
            }
        )

    baseline = _object_balanced(rows, "baseline_cd_ul1_mm")
    oracle = _object_balanced(rows, "oracle_translation_cd_ul1_mm")
    scale_records = {}
    for scale in scales:
        metric = f"scale_{scale:g}_cd_ul1_mm"
        candidate = _object_balanced(rows, metric)
        object_changes = {}
        for object_name in sorted(objects):
            current = [row for row in rows if row["object"] == object_name]
            object_baseline = float(
                np.mean([row["baseline_cd_ul1_mm"] for row in current])
            )
            object_candidate = float(np.mean([row[metric] for row in current]))
            object_changes[object_name] = {
                "baseline_mean_cd_ul1_mm": object_baseline,
                "candidate_mean_cd_ul1_mm": object_candidate,
                "relative_improvement": (
                    (object_baseline - object_candidate) / max(object_baseline, 1e-15)
                ),
            }
        improvements = [
            value["relative_improvement"] for value in object_changes.values()
        ]
        scale_records[f"{scale:g}"] = {
            "object_balanced_cd_ul1_mm": candidate,
            "relative_improvement": (baseline - candidate) / max(baseline, 1e-15),
            "object_wins": int(np.sum(np.asarray(improvements) > 0.0)),
            "worst_object_relative_improvement": float(np.min(improvements)),
            "objects": object_changes,
        }
    best_scale = min(
        scales,
        key=lambda value: (
            scale_records[f"{value:g}"]["object_balanced_cd_ul1_mm"],
            value,
        ),
    )
    best = scale_records[f"{best_scale:g}"]
    gate_checks = {
        "minimum_object_balanced_improvement": (
            best["relative_improvement"] >= arguments.minimum_relative_improvement
        ),
        "minimum_object_wins": best["object_wins"] >= arguments.minimum_object_wins,
        "maximum_object_regression": (
            best["worst_object_relative_improvement"]
            >= -arguments.maximum_object_regression
        ),
    }
    result = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "claim_boundary": (
            "Post-open cross-object diagnostic on development and previously "
            "opened calibration objects. The best scale is selected after seeing "
            "these outcomes; this cannot authorize or confirm a target claim."
        ),
        "configuration": {
            "feature_family": arguments.feature_family,
            "ridge_penalty": arguments.ridge_penalty,
            "maximum_translation_mm": arguments.maximum_translation_mm,
            "scales": list(scales),
            "aggregation": "equal frames within take, takes within object, objects",
        },
        "source": {
            "artifact_count": len(takes),
            "object_count": len(objects),
            "row_count": len(rows),
            "artifacts": [
                {
                    "path": str(take["metadata_path"]),
                    "sha256": take["metadata_sha256"],
                    "role": take["role"],
                }
                for take in takes
            ],
            "sealed_target_object_accessed": False,
        },
        "baseline_object_balanced_cd_ul1_mm": baseline,
        "per_frame_oracle_translation_object_balanced_cd_ul1_mm": oracle,
        "per_frame_oracle_translation_relative_improvement": (
            (baseline - oracle) / max(baseline, 1e-15)
        ),
        "scales": scale_records,
        "post_open_best_scale": best_scale,
        "post_open_best": best,
        "diagnostic_gate_checks": gate_checks,
        "diagnostic_gate_passed": all(gate_checks.values()),
        "folds": fold_records,
        "rows": rows,
    }
    _write_json_once(arguments.output.resolve(), result)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-family",
        choices=("causal", "constant"),
        default="causal",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--maximum-translation-mm", type=float, default=10.0)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-object-wins", type=int, default=7)
    parser.add_argument("--maximum-object-regression", type=float, default=0.10)
    parser.add_argument("--acknowledge-opened-outcomes", action="store_true")
    return parser


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
