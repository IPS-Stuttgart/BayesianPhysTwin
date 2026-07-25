"""Post-opening SOTA report for the sealed object-disjoint MatPhys study."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .matphys_causal_bridge import sha256_file
from .phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from .phystwin_horizon_analysis import HORIZON_LABELS, split_future_horizon
from .phystwin_sota_comparison import PHYSTWIN_TABLE1_CASES


MATPHYS_LOO_SOTA_REPORT_CONTRACT = "matphys-object-disjoint-loo-sota-report-v1"
_METRICS = ("chamfer_distance_m", "track_error_m")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _validated_identity(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file identity")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(value.get("sha256", "")):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _case_metrics(
    data_root: Path,
    case: str,
    trajectory_path: Path,
    *,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    data = _load_pickle(data_root / case / "final_data.pkl")
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    tracks = np.asarray(_load_pickle(data_root / case / "gt_track_3d.pkl"))
    trajectory = np.asarray(_load_pickle(trajectory_path), dtype=float)
    return official_metrics_by_frame(
        trajectory,
        observed,
        visible,
        tracks,
        num_surface_points=observed.shape[1] + len(np.asarray(data["surface_points"])),
        start_frame=start,
        end_frame=end,
    )


def _equal_case_means(
    values: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, float]:
    return {
        metric: float(
            np.mean([np.mean(case_values[metric]) for case_values in values.values()])
        )
        for metric in _METRICS
    }


def _percent_change(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("identity-arm metric must be positive")
    return 100.0 * (candidate / baseline - 1.0)


def build_matphys_loo_sota_report(
    data_root: str | Path,
    selection_summary: str | Path,
    future_summary: str | Path,
    output_path: str | Path,
    *,
    published_chamfer_m: float = 0.008,
    published_track_m: float = 0.015,
    bootstrap_samples: int = 10_000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260719,
) -> dict[str, object]:
    """Report a frozen selection after, and only after, its future was opened."""

    if published_chamfer_m <= 0.0 or published_track_m <= 0.0:
        raise ValueError("published references must be positive")
    selection_path = Path(selection_summary).resolve()
    future_path = Path(future_summary).resolve()
    selection_hash = sha256_file(selection_path)
    future_hash = sha256_file(future_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    future = json.loads(future_path.read_text(encoding="utf-8"))
    if selection.get("future_metrics_opened") is not False:
        raise ValueError("selection must be a sealed prefix-only artifact")
    if future.get("future_metrics_opened") is not True:
        raise ValueError("future summary has not opened the held-out interval")
    bound_selection = future.get("contract", {}).get("selection_summary")
    if not isinstance(bound_selection, Mapping):
        raise ValueError("future summary does not bind its selection artifact")
    if str(bound_selection.get("sha256", "")) != selection_hash:
        raise ValueError("future summary was opened from a different selection")
    if future.get("contract", {}).get("selection_protocol_id") != selection.get(
        "protocol_id"
    ):
        raise ValueError("future and selection protocol ids differ")

    selection_cases = selection.get("case_results")
    future_cases = future.get("case_results")
    if not isinstance(selection_cases, Mapping) or not isinstance(
        future_cases, Mapping
    ):
        raise ValueError("selection and future summaries must contain case results")
    case_order = tuple(str(case) for case in selection_cases)
    if case_order != PHYSTWIN_TABLE1_CASES or tuple(future_cases) != case_order:
        raise ValueError("report requires the ordered official 22-case cohort")
    reference_family = str(selection.get("contract", {}).get("reference_family", ""))
    if not reference_family:
        raise ValueError("selection omits the identity reference family")

    root = Path(data_root).resolve()
    baseline_by_case: dict[str, dict[str, np.ndarray]] = {}
    selected_by_case: dict[str, dict[str, np.ndarray]] = {}
    inputs: dict[str, object] = {}
    selected_family_counts: dict[str, int] = {}
    selected_method_counts: dict[str, int] = {}
    per_case: dict[str, object] = {}
    for case in case_order:
        selected_record = selection_cases[case]
        future_record = future_cases[case]
        if not isinstance(selected_record, Mapping) or not isinstance(
            future_record, Mapping
        ):
            raise ValueError(f"{case}: malformed case record")
        family = str(selected_record.get("selected_family", ""))
        method = str(selected_record.get("selected_within_family_method", ""))
        selected_family_counts[family] = selected_family_counts.get(family, 0) + 1
        selected_method_counts[method] = selected_method_counts.get(method, 0) + 1
        if future_record.get("selected_family") != family:
            raise ValueError(f"{case}: opened family differs from sealed selection")
        raw_family_outputs = selected_record.get("family_outputs")
        if not isinstance(raw_family_outputs, Mapping):
            raise ValueError(f"{case}: family outputs are missing")
        baseline_path = _validated_identity(
            raw_family_outputs.get(reference_family),
            label=f"{case}.{reference_family}",
        )
        selected_path = _validated_identity(
            selected_record.get("output"), label=f"{case}.selected"
        )
        opened_selected = _validated_identity(
            future_record.get("selected_output"), label=f"{case}.opened_selected"
        )
        if sha256_file(selected_path) != sha256_file(opened_selected):
            raise ValueError(f"{case}: opened output differs from sealed output")

        split_path = root / case / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        start, end = (int(value) for value in split["test"])
        if start != int(selected_record["train_end_frame_exclusive"]):
            raise ValueError(f"{case}: selection and benchmark split differ")
        if end > int(selected_record["frame_count"]):
            raise ValueError(f"{case}: selected trajectory does not cover the future")
        baseline = _case_metrics(root, case, baseline_path, start=start, end=end)
        candidate = _case_metrics(root, case, selected_path, start=start, end=end)
        baseline_by_case[case] = baseline
        selected_by_case[case] = candidate
        case_summary: dict[str, object] = {
            "selected_family": family,
            "selected_within_family_method": method,
            "frame_interval": [start, end],
            "metrics": {},
            "horizons": {},
        }
        for metric in _METRICS:
            baseline_mean = float(np.mean(baseline[metric]))
            candidate_mean = float(np.mean(candidate[metric]))
            case_summary["metrics"][metric] = {
                "identity_mean_m": baseline_mean,
                "selected_mean_m": candidate_mean,
                "percent_change": _percent_change(candidate_mean, baseline_mean),
            }
        for horizon, indexes in split_future_horizon(end - start).items():
            case_summary["horizons"][horizon] = {
                metric: {
                    "identity_mean_m": float(np.mean(baseline[metric][indexes])),
                    "selected_mean_m": float(np.mean(candidate[metric][indexes])),
                }
                for metric in _METRICS
            }
        per_case[case] = case_summary
        inputs[case] = {
            "split": _identity(split_path),
            "final_data": _identity(root / case / "final_data.pkl"),
            "gt_track_3d": _identity(root / case / "gt_track_3d.pkl"),
            "identity_trajectory": _identity(baseline_path),
            "selected_trajectory": _identity(selected_path),
        }

    selected_mean = _equal_case_means(selected_by_case)
    identity_mean = _equal_case_means(baseline_by_case)
    opened_mean = future.get("comparison", {}).get("selected_equal_case_mean")
    if not isinstance(opened_mean, Mapping) or any(
        not np.isclose(selected_mean[metric], float(opened_mean[metric]), atol=1e-12)
        for metric in _METRICS
    ):
        raise ValueError("recomputed selected means differ from the future opener")

    clusters = {case: phystwin_physical_object_cluster(case) for case in case_order}
    paired = {
        case: (baseline_by_case[case], selected_by_case[case]) for case in case_order
    }
    bootstrap = paired_block_bootstrap(
        paired,
        samples=bootstrap_samples,
        block_length=bootstrap_block_length,
        seed=bootstrap_seed,
        clusters=clusters,
    )
    horizon_summary: dict[str, object] = {}
    for horizon_index, horizon in enumerate(HORIZON_LABELS):
        horizon_pairs = {}
        horizon_baseline = {}
        horizon_selected = {}
        for case in case_order:
            indexes = split_future_horizon(len(baseline_by_case[case][_METRICS[0]]))[
                horizon
            ]
            baseline = {
                metric: baseline_by_case[case][metric][indexes] for metric in _METRICS
            }
            candidate = {
                metric: selected_by_case[case][metric][indexes] for metric in _METRICS
            }
            horizon_pairs[case] = (baseline, candidate)
            horizon_baseline[case] = baseline
            horizon_selected[case] = candidate
        horizon_summary[horizon] = {
            "identity_equal_case_mean": _equal_case_means(horizon_baseline),
            "selected_equal_case_mean": _equal_case_means(horizon_selected),
            "paired_bootstrap": paired_block_bootstrap(
                horizon_pairs,
                samples=bootstrap_samples,
                block_length=bootstrap_block_length,
                seed=bootstrap_seed + horizon_index + 1,
                clusters=clusters,
            ),
        }

    worst_cases = {}
    for metric in _METRICS:
        changes = {
            case: float(per_case[case]["metrics"][metric]["percent_change"])
            for case in case_order
        }
        worst = max(changes, key=changes.get)
        best = min(changes, key=changes.get)
        worst_cases[metric] = {
            "improved_case_count": sum(value < 0.0 for value in changes.values()),
            "worst_case": worst,
            "worst_percent_change": changes[worst],
            "best_case": best,
            "best_percent_change": changes[best],
        }

    references = {
        "chamfer_distance_m": published_chamfer_m,
        "track_error_m": published_track_m,
    }
    metric_passes = {
        metric: selected_mean[metric] < reference
        for metric, reference in references.items()
    }
    if sha256_file(selection_path) != selection_hash:
        raise RuntimeError("selection summary changed during reporting")
    if sha256_file(future_path) != future_hash:
        raise RuntimeError("future summary changed during reporting")
    for case, case_inputs in inputs.items():
        for label, identity in case_inputs.items():
            _validated_identity(identity, label=f"{case}.{label}")
    report = {
        "schema_version": 1,
        "contract": MATPHYS_LOO_SOTA_REPORT_CONTRACT,
        "future_metrics_opened": True,
        "claim_boundary": (
            "Retrospective official PhysTwin benchmark with object-disjoint source "
            "training and sealed within-run future opening; not untouched external "
            "confirmation."
        ),
        "selection_observation_boundary": (
            "The permitted past validation prefix includes released manual 3D track "
            "labels. No future manual track is used, but this is online-supervised "
            "selection rather than a label-free deployment result."
        ),
        "selection_summary": {"path": str(selection_path), "sha256": selection_hash},
        "future_summary": {"path": str(future_path), "sha256": future_hash},
        "data_root": str(root),
        "reference_family": reference_family,
        "selection_counts": {
            "family": selected_family_counts,
            "within_family_method": selected_method_counts,
        },
        "point_estimates": {
            "identity_equal_case_mean": identity_mean,
            "selected_equal_case_mean": selected_mean,
            "all_family_equal_case_means": future["comparison"][
                "family_equal_case_means"
            ],
        },
        "published_rounded_reference": {
            "source": "MatPhys Table 1",
            "values_m": references,
            "comparison_note": (
                "Strict comparison to published rounded values; this does not "
                "reproduce MatPhys under the local evaluator."
            ),
        },
        "sota_point_estimate_gate": {
            "metric_passes": metric_passes,
            "passed": all(metric_passes.values()),
            "margin_m": {
                metric: references[metric] - selected_mean[metric]
                for metric in _METRICS
            },
        },
        "paired_vs_identity": bootstrap,
        "future_horizons": horizon_summary,
        "worst_cases": worst_cases,
        "per_case": per_case,
        "uncertainty_claim": {
            "status": "not established by this deterministic family report",
            "note": (
                "Object-clustered paired uncertainty concerns point-estimate change; "
                "NEES and predictive coverage require a separately validated "
                "predictive covariance."
            ),
        },
        "inputs": inputs,
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["output_path"] = str(destination)
    report["output_sha256"] = sha256_file(destination)
    return report
