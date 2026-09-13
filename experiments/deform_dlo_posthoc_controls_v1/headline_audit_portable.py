"""Portable read-only DLO4/DLO5 rescore from a retained Actions artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def trajectory_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    candidates: list[object] = [payload]
    if isinstance(payload, dict):
        for key in (
            "trajectory",
            "positions",
            "position",
            "vertices",
            "states",
            "state",
            "data",
        ):
            if key in payload:
                candidates.insert(0, payload[key])
        candidates.extend(payload.values())
    elif isinstance(payload, (tuple, list)):
        candidates.extend(payload)
    for candidate in candidates:
        try:
            array = np.asarray(candidate, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if array.shape == (500, 12, 3) and np.isfinite(array).all():
            return array
        if array.shape == (12, 500, 3) and np.isfinite(array).all():
            return np.swapaxes(array, 0, 1)
    raise ValueError(f"cannot decode a 500x12x3 trajectory from {path}")


def load_target(
    dataset_root: Path, dlo: str, names: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    paths = sorted((dataset_root / dlo / "eval").glob("*.pkl"))
    by_name = {path.name: path for path in paths}
    if len(paths) != 14 or len(by_name) != 14 or set(names) != set(by_name):
        raise ValueError(f"{dlo} public evaluation roster differs")
    trajectories = []
    identities = []
    for name in names:
        path = by_name[name]
        trajectories.append(trajectory_array(path)[2:])
        identities.append(identity(path))
    return np.stack(trajectories), identities


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
    ratios = candidate_case / baseline_case
    return {
        "candidate_mean_l1_m": float(np.mean(candidate_case)),
        "baseline_mean_l1_m": float(np.mean(baseline_case)),
        "relative_improvement": float(
            1.0 - np.mean(candidate_case) / np.mean(baseline_case)
        ),
        "wins": int(np.count_nonzero(candidate_case < baseline_case)),
        "ties": int(np.count_nonzero(candidate_case == baseline_case)),
        "losses": int(np.count_nonzero(candidate_case > baseline_case)),
        "worst_candidate_to_baseline_ratio": float(np.max(ratios)),
        "case_names": names,
        "candidate_case_l1_m": candidate_case.tolist(),
        "baseline_case_l1_m": baseline_case.tolist(),
        "case_ratios": ratios.tolist(),
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
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_candidate = []
    observed_baseline = []
    for dlo in DLOS:
        cand = np.asarray(candidate[dlo], dtype=np.float64)
        base = np.asarray(baseline[dlo], dtype=np.float64)
        if cand.shape != (14,) or base.shape != (14,):
            raise ValueError("bootstrap trajectory roster differs")
        arrays[dlo] = cand, base
        observed_candidate.append(float(np.mean(cand)))
        observed_baseline.append(float(np.mean(base)))
    for index in range(replicates):
        candidate_means = []
        baseline_means = []
        for cand, base in arrays.values():
            draw = rng.integers(0, 14, size=14)
            candidate_means.append(float(np.mean(cand[draw])))
            baseline_means.append(float(np.mean(base[draw])))
        samples[index] = 1.0 - np.mean(candidate_means) / np.mean(baseline_means)
    observed = 1.0 - np.mean(observed_candidate) / np.mean(observed_baseline)
    return {
        "relative_improvement": float(observed),
        "bootstrap_low": float(np.quantile(samples, 0.025)),
        "bootstrap_high": float(np.quantile(samples, 0.975)),
    }


def load_prediction_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "names",
            "physical",
            "compute_matched_physical",
            "candidate",
        }
        if not required.issubset(archive.files):
            raise ValueError(
                f"prediction archive {path} lacks keys; found {sorted(archive.files)}"
            )
        names = [normalize_name(value) for value in archive["names"].tolist()]
        result = {
            "names": names,
            "physical": np.asarray(archive["physical"], dtype=np.float64),
            "compute_matched_physical": np.asarray(
                archive["compute_matched_physical"], dtype=np.float64
            ),
            "candidate": np.asarray(archive["candidate"], dtype=np.float64),
        }
    if len(names) != 14 or len(set(names)) != 14:
        raise ValueError(f"prediction archive {path} has an invalid name roster")
    return result


def discover_archives(
    artifact_root: Path, dataset_root: Path
) -> dict[str, tuple[Path, dict[str, Any], np.ndarray, list[dict[str, Any]]]]:
    raw_paths = sorted(artifact_root.rglob("target_predictions.npz"))
    if not raw_paths:
        raw_paths = sorted(artifact_root.rglob("*.npz"))
    unique_paths = []
    seen_hashes: set[str] = set()
    for path in raw_paths:
        digest = sha256(path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique_paths.append(path)
    matches: dict[str, list[tuple[Path, dict[str, Any], np.ndarray, list[dict[str, Any]]]]] = {
        dlo: [] for dlo in DLOS
    }
    failures = []
    for path in unique_paths:
        try:
            prediction = load_prediction_archive(path)
        except Exception as error:
            failures.append({"path": str(path), "reason": repr(error)})
            continue
        for dlo in DLOS:
            try:
                target, target_ids = load_target(
                    dataset_root, dlo, prediction["names"]
                )
            except Exception:
                continue
            physical_mean = float(
                np.mean(case_errors(prediction["physical"], target))
            )
            if abs(physical_mean - EXPECTED[dlo]["physical_mean_l1_m"]) <= 1e-12:
                matches[dlo].append((path, prediction, target, target_ids))
    selected = {}
    for dlo in DLOS:
        if len(matches[dlo]) != 1:
            raise ValueError(
                f"expected exactly one {dlo} target archive, found {len(matches[dlo])}; "
                f"candidate paths={list(map(str, unique_paths))}; failures={failures}"
            )
        selected[dlo] = matches[dlo][0]
    if selected["DLO4"][0].resolve() == selected["DLO5"][0].resolve():
        raise ValueError("one prediction archive was assigned to both DLOs")
    return selected


def run(artifact_root: Path, dataset_root: Path, output: Path) -> int:
    artifact_root = artifact_root.resolve()
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = discover_archives(artifact_root, dataset_root)
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dlo in DLOS:
        path, prediction, target, target_ids = selected[dlo]
        physical = prediction["physical"]
        compute = prediction["compute_matched_physical"]
        candidate = prediction["candidate"]
        names = prediction["names"]
        if (
            physical.shape != (14, 498, 12, 3)
            or compute.shape != physical.shape
            or candidate.shape != physical.shape
            or target.shape != physical.shape
        ):
            raise ValueError(f"{dlo} retained shape differs")
        versus_physical = comparison(candidate, physical, target, names)
        versus_compute = comparison(candidate, compute, target, names)
        compute_versus_physical = comparison(compute, physical, target, names)
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
        clamped = np.asarray((0, 1, 10, 11), dtype=np.int64)
        clamped_max_abs = float(
            np.max(np.abs(candidate[:, :, clamped] - physical[:, :, clamped]))
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
            "prediction_archive": identity(path),
            "target_files": target_ids,
            "versus_physical": versus_physical,
            "versus_compute_matched": versus_compute,
            "compute_matched_versus_physical": compute_versus_physical,
            "registered_mean_reproduction": reproduction,
            "clamped_node_max_abs_difference_m": clamped_max_abs,
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
                rows.append(
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
        candidate_by_dlo, physical_by_dlo, seed=20260903, replicates=10000
    )
    equal_compute = bootstrap_equal_dlo(
        candidate_by_dlo, compute_by_dlo, seed=20260904, replicates=10000
    )
    equal_physical.update(
        {
            "candidate_mean_l1_m": float(
                np.mean(
                    [
                        results[dlo]["versus_physical"]["candidate_mean_l1_m"]
                        for dlo in DLOS
                    ]
                )
            ),
            "baseline_mean_l1_m": float(
                np.mean(
                    [
                        results[dlo]["versus_physical"]["baseline_mean_l1_m"]
                        for dlo in DLOS
                    ]
                )
            ),
            "wins": int(
                sum(results[dlo]["versus_physical"]["wins"] for dlo in DLOS)
            ),
        }
    )
    equal_compute.update(
        {
            "candidate_mean_l1_m": equal_physical["candidate_mean_l1_m"],
            "baseline_mean_l1_m": float(
                np.mean(
                    [
                        results[dlo]["versus_compute_matched"]["baseline_mean_l1_m"]
                        for dlo in DLOS
                    ]
                )
            ),
            "wins": int(
                sum(results[dlo]["versus_compute_matched"]["wins"] for dlo in DLOS)
            ),
        }
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
        "source_parent_workflow_run_id": 33361441865,
        "source_parent_target_artifact_id": 9809452574,
        "artifact_download_root": identity(artifact_root),
        "dataset_root": str(dataset_root),
        "runner_tag": "gpuserver6000",
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
            "matching public trajectories on a second server. It is retrospective "
            "confirmation of arithmetic, file identity, and cross-server dataset "
            "agreement, not a new prospective target evaluation, unseen-object "
            "transfer, physical-state identification, calibration, deployment "
            "safety, or robot-control evidence."
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
        writer.writerows(rows)
    lines = [
        "# Cross-server DEFORM DLO4/DLO5 headline audit",
        "",
        "The sealed parent prediction artifact was downloaded from GitHub Actions and rescored against the matching public DEFORM copy on `gpuserver6000`.",
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
            f"Equal-DLO gain versus physical: {100.0 * equal_physical['relative_improvement']:.2f}% "
            f"(complete-trajectory bootstrap 95% interval "
            f"[{100.0 * equal_physical['bootstrap_low']:.2f}%, "
            f"{100.0 * equal_physical['bootstrap_high']:.2f}%]).",
            "",
            "No new data were collected and no backbone update was executed. The target outcomes were already open before this audit.",
        ]
    )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"primary_gate": primary_gate, "equal_dlo": record["equal_dlo"]}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.artifact_root, args.dataset_root, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
