#!/usr/bin/env python3
"""Run a post-open leave-one-repeat-out cloth discrepancy diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.cloth_action_phase_diagnostic import (
    apply_action_phase_correction,
    apply_action_phase_translation_delta,
    fit_action_phase_profile,
    fit_action_phase_translation_delta,
    projected_residual_scale,
)
from bayesian_phystwin.cloth_sim2real_belief import (
    ClothReadoutBeliefConfig,
    associate_dense_cloud,
    directed_l1_chamfer_m,
    load_binary_little_endian_ply_xyz,
    symmetric_l1_chamfer_m,
)

METHOD_ID = "cloth-sim2real-action-phase-diagnostic-v1"
CLOTHS = ("chequered_rag", "cotton_rag", "linen_rag")
REPEATS = (0, 1, 2)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cloud_dir(dataset_root: Path, case_name: str) -> Path:
    root = dataset_root
    if (root / "Benchmarking_cloth").is_dir():
        root = root / "Benchmarking_cloth"
    result = root / case_name / "dynamic" / "cloud"
    _require(result.is_dir(), f"point-cloud directory does not exist: {result}")
    return result


def _case_table(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "ClothSim2RealDatasetManifest",
        "dataset manifest kind changed",
    )
    return {str(case["case_id"]): case for case in payload["cases"]}


def _relative_improvement(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline_mean = float(np.mean(baseline))
    return float(
        (baseline_mean - float(np.mean(candidate))) / max(baseline_mean, 1e-15)
    )


def _scalar_relative_improvement(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(baseline, 1e-15))


def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _run(args: argparse.Namespace) -> int:
    _require(
        args.acknowledge_opened_outcomes,
        "this diagnostic requires --acknowledge-opened-outcomes",
    )
    manifest_path = args.manifest.resolve()
    cases = _case_table(manifest_path)
    dataset_root = args.dataset_root.resolve()
    run_roots = {
        0: args.source_run_root.resolve(),
        1: args.calibration_run_root.resolve(),
        2: args.target_run_root.resolve(),
    }
    config = ClothReadoutBeliefConfig()
    records: dict[tuple[str, int], dict[str, Any]] = {}

    for cloth in CLOTHS:
        for repeat in REPEATS:
            case_name = f"{cloth}_{repeat}"
            case_id = f"{case_name}/dynamic"
            _require(case_id in cases, f"manifest is missing {case_id}")
            descriptor = cases[case_id]
            branch = int(descriptor["branch_frame"])
            prediction = run_roots[repeat] / "predictions" / f"{case_name}_dynamic"
            seal_path = prediction / "prediction_seal.json"
            physical_path = prediction / "physical_future.npy"
            belief_path = prediction / "readout_belief.npz"
            _require(
                seal_path.is_file()
                and physical_path.is_file()
                and belief_path.is_file(),
                f"prediction artifacts are incomplete for {case_id}",
            )
            physical = np.load(physical_path, allow_pickle=False)
            with np.load(belief_path, allow_pickle=False) as belief:
                correction = np.asarray(
                    belief["correction_m"],
                    dtype=np.float64,
                )
            clouds = [
                load_binary_little_endian_ply_xyz(
                    _cloud_dir(dataset_root, case_name) / f"{frame:05d}.ply"
                )
                for frame in range(
                    branch + 1,
                    int(descriptor["frame_count"]),
                )
            ]
            _require(
                len(physical) == len(clouds),
                f"future length changed for {case_id}",
            )
            profile: np.ndarray | None = None
            residual_field: np.ndarray | None = None
            if np.any(correction):
                values = []
                residuals = []
                for state, cloud in zip(physical, clouds, strict=True):
                    association = associate_dense_cloud(
                        state,
                        cloud,
                        candidate_count=config.candidate_count,
                        sensor_std_m=config.sensor_std_m,
                    )
                    residuals.append(association.observed_points_m - state)
                    values.append(
                        projected_residual_scale(
                            correction,
                            state,
                            association.observed_points_m,
                        )
                    )
                profile = np.asarray(values, dtype=np.float64)
                residual_field = np.asarray(residuals, dtype=np.float64)
            records[(cloth, repeat)] = {
                "case_id": case_id,
                "physical": physical,
                "correction": correction,
                "clouds": clouds,
                "profile": profile,
                "residual_field": residual_field,
                "prediction_seal_sha256": _sha256(seal_path),
            }

    case_results: dict[str, Any] = {}
    for cloth in CLOTHS:
        for repeat in REPEATS:
            record = records[(cloth, repeat)]
            correction = record["correction"]
            own_profile = record["profile"]
            if own_profile is None:
                case_results[record["case_id"]] = {
                    "evaluated": False,
                    "reason": "held-prefix-correction-rejected",
                    "exact_fallback": True,
                    "prediction_seal_sha256": record["prediction_seal_sha256"],
                }
                continue
            training_profiles = [
                records[(cloth, other)]["profile"]
                for other in REPEATS
                if other != repeat and records[(cloth, other)]["profile"] is not None
            ]
            training_records = [
                records[(cloth, other)]
                for other in REPEATS
                if other != repeat
                and records[(cloth, other)]["residual_field"] is not None
            ]
            _require(
                training_profiles and len(training_records) == len(training_profiles),
                f"no disjoint training profile is available for {record['case_id']}",
            )
            scale = fit_action_phase_profile(
                training_profiles,
                target_length=len(record["physical"]),
            )
            phase_candidate = apply_action_phase_correction(
                record["physical"],
                correction,
                scale,
            )
            translation_delta = fit_action_phase_translation_delta(
                [
                    training_record["residual_field"]
                    for training_record in training_records
                ],
                [training_record["correction"] for training_record in training_records],
                target_length=len(record["physical"]),
                maximum_translation_m=config.maximum_correction_m,
            )
            translation_candidate = apply_action_phase_translation_delta(
                record["physical"],
                correction,
                translation_delta,
                maximum_correction_m=config.maximum_correction_m,
            )
            persistent_candidate = apply_action_phase_correction(
                record["physical"],
                correction,
                np.ones(len(record["physical"]), dtype=np.float64),
            )
            physical_symmetric = []
            persistent_symmetric = []
            phase_symmetric = []
            translation_symmetric = []
            physical_directed = []
            persistent_directed = []
            phase_directed = []
            translation_directed = []
            for physical, persistent, phase, translation, cloud in zip(
                record["physical"],
                persistent_candidate,
                phase_candidate,
                translation_candidate,
                record["clouds"],
                strict=True,
            ):
                physical_symmetric.append(symmetric_l1_chamfer_m(physical, cloud))
                persistent_symmetric.append(symmetric_l1_chamfer_m(persistent, cloud))
                phase_symmetric.append(symmetric_l1_chamfer_m(phase, cloud))
                translation_symmetric.append(symmetric_l1_chamfer_m(translation, cloud))
                physical_directed.append(directed_l1_chamfer_m(physical, cloud))
                persistent_directed.append(directed_l1_chamfer_m(persistent, cloud))
                phase_directed.append(directed_l1_chamfer_m(phase, cloud))
                translation_directed.append(directed_l1_chamfer_m(translation, cloud))
            metrics = {
                name: np.asarray(values, dtype=np.float64)
                for name, values in (
                    ("physical_symmetric", physical_symmetric),
                    ("persistent_symmetric", persistent_symmetric),
                    ("phase_symmetric", phase_symmetric),
                    ("translation_symmetric", translation_symmetric),
                    ("physical_directed", physical_directed),
                    ("persistent_directed", persistent_directed),
                    ("phase_directed", phase_directed),
                    ("translation_directed", translation_directed),
                )
            }
            means = {name: float(np.mean(values)) for name, values in metrics.items()}
            horizon_indices = np.array_split(
                np.arange(len(record["physical"])),
                3,
            )
            case_results[record["case_id"]] = {
                "evaluated": True,
                "training_repeat_count": len(training_profiles),
                "prediction_seal_sha256": record["prediction_seal_sha256"],
                "profile_correlation": _profile_correlation(scale, own_profile),
                "physical_symmetric_l1_chamfer_m": means["physical_symmetric"],
                "persistent_symmetric_l1_chamfer_m": means["persistent_symmetric"],
                "action_phase_symmetric_l1_chamfer_m": means["phase_symmetric"],
                "translation_delta_symmetric_l1_chamfer_m": means[
                    "translation_symmetric"
                ],
                "persistent_symmetric_relative_improvement": (
                    _relative_improvement(
                        metrics["physical_symmetric"],
                        metrics["persistent_symmetric"],
                    )
                ),
                "action_phase_symmetric_relative_improvement": (
                    _relative_improvement(
                        metrics["physical_symmetric"],
                        metrics["phase_symmetric"],
                    )
                ),
                "action_phase_vs_persistent_symmetric_relative_improvement": (
                    _scalar_relative_improvement(
                        means["persistent_symmetric"],
                        means["phase_symmetric"],
                    )
                ),
                "translation_delta_vs_persistent_symmetric_relative_improvement": (
                    _scalar_relative_improvement(
                        means["persistent_symmetric"],
                        means["translation_symmetric"],
                    )
                ),
                "physical_directed_l1_chamfer_m": means["physical_directed"],
                "persistent_directed_l1_chamfer_m": means["persistent_directed"],
                "action_phase_directed_l1_chamfer_m": means["phase_directed"],
                "translation_delta_directed_l1_chamfer_m": means[
                    "translation_directed"
                ],
                "persistent_directed_relative_improvement": (
                    _relative_improvement(
                        metrics["physical_directed"],
                        metrics["persistent_directed"],
                    )
                ),
                "action_phase_directed_relative_improvement": (
                    _relative_improvement(
                        metrics["physical_directed"],
                        metrics["phase_directed"],
                    )
                ),
                "action_phase_vs_persistent_directed_relative_improvement": (
                    _scalar_relative_improvement(
                        means["persistent_directed"],
                        means["phase_directed"],
                    )
                ),
                "translation_delta_vs_persistent_directed_relative_improvement": (
                    _scalar_relative_improvement(
                        means["persistent_directed"],
                        means["translation_directed"],
                    )
                ),
                "horizons": [
                    {
                        "name": name,
                        "persistent_directed_relative_improvement": (
                            _relative_improvement(
                                metrics["physical_directed"][indices],
                                metrics["persistent_directed"][indices],
                            )
                        ),
                        "action_phase_directed_relative_improvement": (
                            _relative_improvement(
                                metrics["physical_directed"][indices],
                                metrics["phase_directed"][indices],
                            )
                        ),
                        "translation_delta_directed_relative_improvement": (
                            _relative_improvement(
                                metrics["physical_directed"][indices],
                                metrics["translation_directed"][indices],
                            )
                        ),
                    }
                    for name, indices in zip(
                        ("early", "middle", "late"),
                        horizon_indices,
                        strict=True,
                    )
                ],
            }

    evaluated = [value for value in case_results.values() if value["evaluated"]]
    target = [
        value
        for case_id, value in case_results.items()
        if "_2/dynamic" in case_id and value["evaluated"]
    ]

    def ratio_of_means(
        values: list[dict[str, Any]],
        baseline_key: str,
        candidate_key: str,
    ) -> float:
        return _scalar_relative_improvement(
            float(np.mean([value[baseline_key] for value in values])),
            float(np.mean([value[candidate_key] for value in values])),
        )

    payload = {
        "schema_version": 1,
        "artifact_kind": "ClothSim2RealActionPhaseDiagnostic",
        "method_id": METHOD_ID,
        "manifest_sha256": _sha256(manifest_path),
        "configuration": {
            "profile_coordinate": "normalized future horizon",
            "smoothing_window_frames": 9,
            "maximum_absolute_scale": 1.5,
            "translation_statistic": "coordinatewise node median",
            "maximum_total_node_correction_m": config.maximum_correction_m,
            "training_policy": "leave-one-repeat-out, rejected profiles omitted",
        },
        "cases": case_results,
        "aggregate": {
            "evaluated_case_count": len(evaluated),
            "target_repeat_evaluated_cloth_count": len(target),
            "directed_phase_beats_persistence_count": int(
                np.sum(
                    [
                        value["action_phase_directed_l1_chamfer_m"]
                        < value["persistent_directed_l1_chamfer_m"]
                        for value in evaluated
                    ]
                )
            ),
            "symmetric_phase_beats_persistence_count": int(
                np.sum(
                    [
                        value["action_phase_symmetric_l1_chamfer_m"]
                        < value["persistent_symmetric_l1_chamfer_m"]
                        for value in evaluated
                    ]
                )
            ),
            "directed_translation_beats_persistence_count": int(
                np.sum(
                    [
                        value["translation_delta_directed_l1_chamfer_m"]
                        < value["persistent_directed_l1_chamfer_m"]
                        for value in evaluated
                    ]
                )
            ),
            "symmetric_translation_beats_persistence_count": int(
                np.sum(
                    [
                        value["translation_delta_symmetric_l1_chamfer_m"]
                        < value["persistent_symmetric_l1_chamfer_m"]
                        for value in evaluated
                    ]
                )
            ),
            "mean_persistent_symmetric_relative_improvement": float(
                np.mean(
                    [
                        value["persistent_symmetric_relative_improvement"]
                        for value in evaluated
                    ]
                )
            ),
            "mean_action_phase_symmetric_relative_improvement": float(
                np.mean(
                    [
                        value["action_phase_symmetric_relative_improvement"]
                        for value in evaluated
                    ]
                )
            ),
            "ratio_of_means_action_phase_vs_persistent_symmetric_improvement": (
                ratio_of_means(
                    evaluated,
                    "persistent_symmetric_l1_chamfer_m",
                    "action_phase_symmetric_l1_chamfer_m",
                )
            ),
            "ratio_of_means_translation_vs_persistent_symmetric_improvement": (
                ratio_of_means(
                    evaluated,
                    "persistent_symmetric_l1_chamfer_m",
                    "translation_delta_symmetric_l1_chamfer_m",
                )
            ),
            "mean_persistent_directed_relative_improvement": float(
                np.mean(
                    [
                        value["persistent_directed_relative_improvement"]
                        for value in evaluated
                    ]
                )
            ),
            "mean_action_phase_directed_relative_improvement": float(
                np.mean(
                    [
                        value["action_phase_directed_relative_improvement"]
                        for value in evaluated
                    ]
                )
            ),
            "ratio_of_means_action_phase_vs_persistent_directed_improvement": (
                ratio_of_means(
                    evaluated,
                    "persistent_directed_l1_chamfer_m",
                    "action_phase_directed_l1_chamfer_m",
                )
            ),
            "ratio_of_means_translation_vs_persistent_directed_improvement": (
                ratio_of_means(
                    evaluated,
                    "persistent_directed_l1_chamfer_m",
                    "translation_delta_directed_l1_chamfer_m",
                )
            ),
            "target_mean_persistent_directed_relative_improvement": float(
                np.mean(
                    [
                        value["persistent_directed_relative_improvement"]
                        for value in target
                    ]
                )
            ),
            "target_mean_action_phase_directed_relative_improvement": float(
                np.mean(
                    [
                        value["action_phase_directed_relative_improvement"]
                        for value in target
                    ]
                )
            ),
            "target_physical_directed_l1_chamfer_m": float(
                np.mean([value["physical_directed_l1_chamfer_m"] for value in target])
            ),
            "target_persistent_directed_l1_chamfer_m": float(
                np.mean([value["persistent_directed_l1_chamfer_m"] for value in target])
            ),
            "target_action_phase_directed_l1_chamfer_m": float(
                np.mean(
                    [value["action_phase_directed_l1_chamfer_m"] for value in target]
                )
            ),
            "target_translation_delta_directed_l1_chamfer_m": float(
                np.mean(
                    [
                        value["translation_delta_directed_l1_chamfer_m"]
                        for value in target
                    ]
                )
            ),
            "target_ratio_of_means_action_phase_vs_persistent_directed_improvement": (
                ratio_of_means(
                    target,
                    "persistent_directed_l1_chamfer_m",
                    "action_phase_directed_l1_chamfer_m",
                )
            ),
            "target_ratio_of_means_translation_vs_persistent_directed_improvement": (
                ratio_of_means(
                    target,
                    "persistent_directed_l1_chamfer_m",
                    "translation_delta_directed_l1_chamfer_m",
                )
            ),
        },
        "claim_boundary": (
            "Post-open leave-one-repeat-out mechanism diagnostic. Future "
            "outcomes from disjoint opened repeats train the scalar and "
            "translation profiles. No independent confirmation or SOTA claim "
            "is permitted."
        ),
    }
    _write_json_once(args.output.resolve(), payload)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--calibration-run-root", type=Path, required=True)
    parser.add_argument("--target-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-opened-outcomes", action="store_true")
    parser.set_defaults(function=_run)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
