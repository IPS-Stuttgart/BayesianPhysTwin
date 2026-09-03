"""Independent read-only rescore of the retained DLO4/DLO5 headline result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.deform_dlo45_frozen_v1 import core as frozen_core

DLOS = ("DLO4", "DLO5")
EXPECTED = {
    "DLO4": {
        "physical_mean_l1_m": 0.009954855613252654,
        "candidate_mean_l1_m": 0.00895317154556605,
    },
    "DLO5": {
        "physical_mean_l1_m": 0.008050080783977409,
        "candidate_mean_l1_m": 0.007826805297017976,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalize_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def case_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target arrays do not align")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction or target contains non-finite values")
    return np.mean(np.abs(prediction - target), axis=(1, 2, 3))


def comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    candidate_case = case_errors(candidate, target)
    baseline_case = case_errors(baseline, target)
    if np.any(baseline_case <= 0.0):
        raise ValueError("baseline case error is not positive")
    ratio = candidate_case / baseline_case
    return {
        "candidate_mean_l1_m": float(np.mean(candidate_case)),
        "baseline_mean_l1_m": float(np.mean(baseline_case)),
        "relative_improvement": float(
            1.0 - np.mean(candidate_case) / np.mean(baseline_case)
        ),
        "wins": int(np.count_nonzero(candidate_case < baseline_case)),
        "ties": int(np.count_nonzero(candidate_case == baseline_case)),
        "losses": int(np.count_nonzero(candidate_case > baseline_case)),
        "worst_candidate_to_baseline_ratio": float(np.max(ratio)),
        "case_names": names,
        "candidate_case_l1_m": candidate_case.tolist(),
        "baseline_case_l1_m": baseline_case.tolist(),
        "case_ratios": ratio.tolist(),
    }


def bootstrap_equal_dlo(
    candidate: dict[str, list[float]],
    baseline: dict[str, list[float]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    observed_candidate = []
    observed_baseline = []
    arrays = {}
    for dlo in DLOS:
        cand = np.asarray(candidate[dlo], dtype=np.float64)
        base = np.asarray(baseline[dlo], dtype=np.float64)
        if cand.ndim != 1 or cand.shape != base.shape or cand.size != 14:
            raise ValueError("bootstrap case roster differs")
        arrays[dlo] = cand, base
        observed_candidate.append(float(np.mean(cand)))
        observed_baseline.append(float(np.mean(base)))
    for index in range(replicates):
        candidate_means = []
        baseline_means = []
        for cand, base in arrays.values():
            draw = rng.integers(0, cand.size, size=cand.size)
            candidate_means.append(float(np.mean(cand[draw])))
            baseline_means.append(float(np.mean(base[draw])))
        samples[index] = 1.0 - np.mean(candidate_means) / np.mean(baseline_means)
    observed = 1.0 - np.mean(observed_candidate) / np.mean(observed_baseline)
    return {
        "relative_improvement": float(observed),
        "bootstrap_low": float(np.quantile(samples, 0.025)),
        "bootstrap_high": float(np.quantile(samples, 0.975)),
    }


def load_target(dataset_root: Path, dlo: str, names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    paths = sorted((dataset_root / dlo / "eval").glob("*.pkl"))
    by_name = {path.name: path for path in paths}
    if len(paths) != 14 or len(by_name) != 14 or set(names) != set(by_name):
        raise ValueError(f"{dlo} public evaluation roster differs")
    trajectories = []
    identities = []
    for name in names:
        path = by_name[name]
        trajectory = frozen_core.source_runtime._load_trajectory(
            path, frame_count=500, node_count=12
        )
        trajectories.append(trajectory[2:])
        identities.append(identity(path))
    return np.stack(trajectories), identities


def run(parent_root: Path, dataset_root: Path, output: Path) -> int:
    parent_root = parent_root.resolve()
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    joint_seal = parent_root / "joint" / "joint_prediction_seal.json"
    if not joint_seal.is_file():
        raise FileNotFoundError(joint_seal)

    results: dict[str, Any] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for dlo in DLOS:
        prediction_path = parent_root / f"{dlo.lower()}-target" / "target_predictions.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as archive:
            required = {
                "names",
                "physical",
                "compute_matched_physical",
                "candidate",
            }
            if not required.issubset(archive.files):
                raise ValueError(
                    f"{dlo} prediction keys differ: {sorted(archive.files)}"
                )
            names = [normalize_name(value) for value in archive["names"].tolist()]
            physical = np.asarray(archive["physical"], dtype=np.float64)
            compute_matched = np.asarray(
                archive["compute_matched_physical"], dtype=np.float64
            )
            candidate = np.asarray(archive["candidate"], dtype=np.float64)
        if len(names) != 14 or len(set(names)) != 14:
            raise ValueError(f"{dlo} retained target names differ")
        target, target_identities = load_target(dataset_root, dlo, names)
        if physical.shape != target.shape or compute_matched.shape != target.shape or candidate.shape != target.shape:
            raise ValueError(f"{dlo} retained prediction shape differs")
        if target.shape != (14, 498, 12, 3):
            raise ValueError(f"{dlo} target shape differs: {target.shape}")

        versus_physical = comparison(candidate, physical, target, names)
        versus_compute = comparison(candidate, compute_matched, target, names)
        compute_versus_physical = comparison(
            compute_matched, physical, target, names
        )
        clamped = np.asarray((0, 1, 10, 11), dtype=np.int64)
        clamped_max_abs = float(
            np.max(np.abs(candidate[:, :, clamped] - physical[:, :, clamped]))
        )
        expected = EXPECTED[dlo]
        reproduction = {
            "physical_mean_abs_difference_m": abs(
                versus_physical["baseline_mean_l1_m"]
                - expected["physical_mean_l1_m"]
            ),
            "candidate_mean_abs_difference_m": abs(
                versus_physical["candidate_mean_l1_m"]
                - expected["candidate_mean_l1_m"]
            ),
            "tolerance_m": 1e-12,
        }
        reproduction["passed"] = (
            reproduction["physical_mean_abs_difference_m"]
            <= reproduction["tolerance_m"]
            and reproduction["candidate_mean_abs_difference_m"]
            <= reproduction["tolerance_m"]
        )
        gate = {
            "registered_means_reproduced": reproduction["passed"],
            "candidate_beats_physical": versus_physical["relative_improvement"] > 0.0,
            "candidate_beats_compute_matched": versus_compute[
                "relative_improvement"
            ]
            > 0.0,
            "candidate_wins_all_cases_vs_physical": versus_physical["wins"] == 14,
            "clamped_nodes_byte_equal": clamped_max_abs == 0.0,
        }
        gate["passed"] = all(gate.values())
        results[dlo] = {
            "prediction_archive": identity(prediction_path),
            "target_files": target_identities,
            "versus_physical": versus_physical,
            "versus_compute_matched": versus_compute,
            "compute_matched_versus_physical": compute_versus_physical,
            "clamped_node_max_abs_difference_m": clamped_max_abs,
            "registered_mean_reproduction": reproduction,
            "gate": gate,
        }
        for arm, summary in (
            ("candidate_vs_physical", versus_physical),
            ("candidate_vs_compute_matched", versus_compute),
            ("compute_matched_vs_physical", compute_versus_physical),
        ):
            for name, candidate_error, baseline_error, ratio in zip(
                summary["case_names"],
                summary["candidate_case_l1_m"],
                summary["baseline_case_l1_m"],
                summary["case_ratios"],
                strict=True,
            ):
                trajectory_rows.append(
                    {
                        "dlo": dlo,
                        "arm": arm,
                        "trajectory": name,
                        "candidate_l1_mm": 1000.0 * float(candidate_error),
                        "baseline_l1_mm": 1000.0 * float(baseline_error),
                        "ratio": float(ratio),
                    }
                )

    candidate_by_dlo = {
        dlo: results[dlo]["versus_physical"]["candidate_case_l1_m"]
        for dlo in DLOS
    }
    physical_by_dlo = {
        dlo: results[dlo]["versus_physical"]["baseline_case_l1_m"]
        for dlo in DLOS
    }
    compute_by_dlo = {
        dlo: results[dlo]["versus_compute_matched"]["baseline_case_l1_m"]
        for dlo in DLOS
    }
    equal_physical = bootstrap_equal_dlo(
        candidate_by_dlo,
        physical_by_dlo,
        seed=20260903,
        replicates=10000,
    )
    equal_compute = bootstrap_equal_dlo(
        candidate_by_dlo,
        compute_by_dlo,
        seed=20260904,
        replicates=10000,
    )
    equal_physical["candidate_mean_l1_m"] = float(
        np.mean(
            [results[dlo]["versus_physical"]["candidate_mean_l1_m"] for dlo in DLOS]
        )
    )
    equal_physical["baseline_mean_l1_m"] = float(
        np.mean(
            [results[dlo]["versus_physical"]["baseline_mean_l1_m"] for dlo in DLOS]
        )
    )
    equal_physical["wins"] = sum(
        results[dlo]["versus_physical"]["wins"] for dlo in DLOS
    )
    equal_compute["candidate_mean_l1_m"] = equal_physical[
        "candidate_mean_l1_m"
    ]
    equal_compute["baseline_mean_l1_m"] = float(
        np.mean(
            [
                results[dlo]["versus_compute_matched"]["baseline_mean_l1_m"]
                for dlo in DLOS
            ]
        )
    )
    equal_compute["wins"] = sum(
        results[dlo]["versus_compute_matched"]["wins"] for dlo in DLOS
    )
    primary_gate = {
        "each_dlo_passed": all(results[dlo]["gate"]["passed"] for dlo in DLOS),
        "equal_dlo_beats_physical": equal_physical["relative_improvement"] > 0.0,
        "equal_dlo_beats_compute_matched": equal_compute["relative_improvement"]
        > 0.0,
    }
    primary_gate["passed"] = all(primary_gate.values())
    record = {
        "schema_version": 1,
        "contract": "deform-dlo-posthoc-headline-audit-v1",
        "status": "completed",
        "evidence_class": "independent-rescore-of-already-open-retained-targets",
        "parent_workflow_run_id": 33361441865,
        "joint_prediction_seal": identity(joint_seal),
        "new_data_collected": False,
        "backbone_training_updates": 0,
        "target_selection": False,
        "target_calibration": False,
        "dlos": results,
        "equal_dlo": {
            "candidate_vs_physical": equal_physical,
            "candidate_vs_compute_matched": equal_compute,
        },
        "primary_gate": primary_gate,
        "claim_boundary": (
            "This independently rescored retained DLO4/DLO5 predictions against "
            "the verified public trajectories. It is retrospective confirmation "
            "of arithmetic and custody, not a new prospective target evaluation, "
            "unseen-object transfer, physical-state identification, calibration, "
            "deployment safety, or robot-control evidence."
        ),
    }
    write_json(output / "result.json", record)
    with (output / "trajectory_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dlo",
                "arm",
                "trajectory",
                "candidate_l1_mm",
                "baseline_l1_mm",
                "ratio",
            ),
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    lines = [
        "# DEFORM DLO4/DLO5 post-hoc headline audit",
        "",
        "Independent read-only rescore of the retained protected-run arrays against the public evaluation trajectories.",
        "",
        "| Panel | Physical (mm) | Compute matched (mm) | Adapter (mm) | Gain vs physical | Gain vs compute | Wins vs physical | Wins vs compute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dlo in DLOS:
        physical = results[dlo]["versus_physical"]
        compute = results[dlo]["versus_compute_matched"]
        lines.append(
            f"| {dlo} | {1000.0 * physical['baseline_mean_l1_m']:.4f} | "
            f"{1000.0 * compute['baseline_mean_l1_m']:.4f} | "
            f"{1000.0 * physical['candidate_mean_l1_m']:.4f} | "
            f"{100.0 * physical['relative_improvement']:.2f}% | "
            f"{100.0 * compute['relative_improvement']:.2f}% | "
            f"{physical['wins']}/14 | {compute['wins']}/14 |"
        )
    lines.append(
        f"| Equal-DLO | {1000.0 * equal_physical['baseline_mean_l1_m']:.4f} | "
        f"{1000.0 * equal_compute['baseline_mean_l1_m']:.4f} | "
        f"{1000.0 * equal_physical['candidate_mean_l1_m']:.4f} | "
        f"{100.0 * equal_physical['relative_improvement']:.2f}% | "
        f"{100.0 * equal_compute['relative_improvement']:.2f}% | "
        f"{equal_physical['wins']}/28 | {equal_compute['wins']}/28 |"
    )
    lines.extend(
        [
            "",
            f"Primary gate: **{'PASS' if primary_gate['passed'] else 'FAIL'}**.",
            "",
            f"Equal-DLO gain vs physical, complete-trajectory bootstrap: "
            f"{100.0 * equal_physical['relative_improvement']:.2f}% "
            f"[95%: {100.0 * equal_physical['bootstrap_low']:.2f}%, "
            f"{100.0 * equal_physical['bootstrap_high']:.2f}%].",
            "",
            "No new data were collected and no backbone update was executed. The target outcomes were already open before this audit.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"primary_gate": primary_gate, "equal_dlo": record["equal_dlo"]}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.parent_root, args.dataset_root, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
