"""Matched residual-overlay audit for frozen MatPhys/PhysTwin trajectories."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin.phystwin_comparison import (
    official_metrics_by_frame,
    paired_block_bootstrap,
    phystwin_physical_object_cluster,
)
from bayesian_phystwin.phystwin_horizon_analysis import (
    HORIZON_LABELS,
    split_future_horizon,
)
from bayesian_phystwin.phystwin_sota_comparison import PHYSTWIN_TABLE1_CASES

MATPHYS_RESIDUAL_OVERLAY_AUDIT_CONTRACT = "matphys-residual-overlay-audit-v1"
METHODS = ("backbone", "bayesian_anchor", "last_residual", "validation_selected")
METRICS = ("chamfer_distance_m", "track_error_m")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one ordinary file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return cast(dict[str, object], payload)


def _file_identity(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    return {"path": str(source), "sha256": sha256_file(source)}


def _validated_identity(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a file identity")
    source = Path(str(value.get("path", ""))).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sha256_file(source) != str(value.get("sha256", "")):
        raise ValueError(f"{label} SHA-256 mismatch")
    return source


def _load_pickle_array(path: Path, *, require_finite: bool) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    result = np.asarray(value)
    if result.ndim != 3 or result.shape[-1] != 3:
        raise ValueError(f"{path} is not a trajectory array")
    if require_finite and not np.isfinite(result).all():
        raise ValueError(f"{path} contains non-finite prediction coordinates")
    return result


def _case_metrics(
    data_root: Path,
    case: str,
    trajectory_path: Path,
    *,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    with (data_root / case / "final_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{case}: final_data.pkl is malformed")
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    tracks = _load_pickle_array(
        data_root / case / "gt_track_3d.pkl", require_finite=False
    )
    trajectory: np.ndarray = _load_pickle_array(
        trajectory_path, require_finite=True
    ).astype(float, copy=False)
    return official_metrics_by_frame(
        trajectory,
        observed,
        visible,
        tracks,
        num_surface_points=observed.shape[1] + len(np.asarray(data["surface_points"])),
        start_frame=start,
        end_frame=end,
    )


def _mean_metrics(values: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(np.mean(values[metric])) for metric in METRICS}


def _percent_change(candidate: float, baseline: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("backbone metric must be finite and positive")
    return 100.0 * (candidate / baseline - 1.0)


def _row_methods(row: Mapping[str, object]) -> Mapping[str, Mapping[str, float]]:
    methods = row.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("MatPhys overlay row omits method metrics")
    return cast(Mapping[str, Mapping[str, float]], methods)


def summarize_matphys_overlay_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize matched per-case means without selecting on future outcomes."""

    if not rows:
        raise ValueError("MatPhys overlay audit requires at least one case")
    method_means: dict[str, dict[str, float]] = {}
    for method in METHODS:
        method_means[method] = {
            metric: float(
                np.mean([float(_row_methods(row)[method][metric]) for row in rows])
            )
            for metric in METRICS
        }
    baseline = method_means["backbone"]
    comparisons: dict[str, object] = {}
    for method in METHODS[1:]:
        changes = {}
        wins = {}
        worst_ratios = {}
        for metric in METRICS:
            changes[metric] = _percent_change(
                method_means[method][metric], baseline[metric]
            )
            ratios = [
                float(_row_methods(row)[method][metric])
                / float(_row_methods(row)["backbone"][metric])
                for row in rows
            ]
            wins[metric] = int(sum(ratio < 1.0 for ratio in ratios))
            worst_ratios[metric] = float(max(ratios))
        joint_wins = sum(
            all(
                float(_row_methods(row)[method][metric])
                < float(_row_methods(row)["backbone"][metric])
                for metric in METRICS
            )
            for row in rows
        )
        comparisons[method] = {
            "percent_change_vs_backbone": changes,
            "case_wins": wins,
            "joint_case_wins": int(joint_wins),
            "worst_case_ratio": worst_ratios,
        }
    return {
        "case_count": len(rows),
        "equal_case_mean": method_means,
        "comparisons_vs_backbone": comparisons,
    }


def _paired_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    bootstrap_samples: int,
    bootstrap_block_length: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    paired = {}
    clusters = {}
    for row in rows:
        case = str(row["case"])
        frames = cast(Mapping[str, Mapping[str, np.ndarray]], row["frame_metrics"])
        paired[case] = (frames["backbone"], frames[method])
        clusters[case] = phystwin_physical_object_cluster(case)
    return cast(
        dict[str, object],
        paired_block_bootstrap(
            paired,
            samples=bootstrap_samples,
            block_length=bootstrap_block_length,
            seed=bootstrap_seed,
            clusters=clusters,
        ),
    )


def _horizon_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for label in HORIZON_LABELS:
        baseline_cases = []
        candidate_cases = []
        for row in rows:
            frames = cast(Mapping[str, Mapping[str, np.ndarray]], row["frame_metrics"])
            frame_count = len(frames["backbone"][METRICS[0]])
            indexes = split_future_horizon(frame_count)[label]
            baseline_cases.append(
                {
                    metric: float(np.mean(frames["backbone"][metric][indexes]))
                    for metric in METRICS
                }
            )
            candidate_cases.append(
                {
                    metric: float(np.mean(frames[method][metric][indexes]))
                    for metric in METRICS
                }
            )
        result[label] = {
            "backbone_equal_case_mean": {
                metric: float(np.mean([case[metric] for case in baseline_cases]))
                for metric in METRICS
            },
            "candidate_equal_case_mean": {
                metric: float(np.mean([case[metric] for case in candidate_cases]))
                for metric in METRICS
            },
        }
    return result


def build_matphys_residual_overlay_audit(
    data_root: str | Path,
    selection_summary: str | Path,
    future_summary: str | Path,
    output_path: str | Path,
    *,
    expected_selection_sha256: str | None = None,
    expected_future_sha256: str | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_block_length: int = 5,
    bootstrap_seed: int = 20260822,
) -> dict[str, object]:
    """Audit frozen residual corrections on selected MatPhys physical families."""

    selection_path = Path(selection_summary).resolve()
    future_path = Path(future_summary).resolve()
    selection_sha256 = sha256_file(selection_path)
    future_sha256 = sha256_file(future_path)
    if expected_selection_sha256 not in (None, selection_sha256):
        raise ValueError("selection summary differs from the registered artifact")
    if expected_future_sha256 not in (None, future_sha256):
        raise ValueError("future summary differs from the registered artifact")
    selection = _load_json(selection_path)
    future = _load_json(future_path)
    if selection.get("future_metrics_opened") is not False:
        raise ValueError("family selection must precede future opening")
    if future.get("future_metrics_opened") is not True:
        raise ValueError("future summary has not opened the evaluation interval")
    future_contract = future.get("contract")
    if not isinstance(future_contract, Mapping):
        raise ValueError("future summary omits its contract")
    bound_selection = future_contract.get("selection_summary")
    if (
        not isinstance(bound_selection, Mapping)
        or bound_selection.get("sha256") != selection_sha256
    ):
        raise ValueError("future summary was opened from a different selection")

    selection_contract = selection.get("contract")
    selection_cases = selection.get("case_results")
    future_cases = future.get("case_results")
    if not isinstance(selection_contract, Mapping):
        raise ValueError("MatPhys family selection contract is malformed")
    if not isinstance(selection_cases, Mapping) or not isinstance(
        future_cases, Mapping
    ):
        raise ValueError("MatPhys family artifacts are malformed")
    case_order = tuple(str(case) for case in selection_cases)
    if case_order != PHYSTWIN_TABLE1_CASES or tuple(future_cases) != case_order:
        raise ValueError("MatPhys audit requires the ordered official 22-case cohort")
    family_identities = selection_contract.get("families")
    if not isinstance(family_identities, Mapping):
        raise ValueError("selection artifact omits family summaries")
    family_summaries = {
        str(name): _load_json(_validated_identity(identity, label=f"family.{name}"))
        for name, identity in family_identities.items()
    }

    root = Path(data_root).resolve()
    rows = []
    input_identities: dict[str, object] = {}
    for case in case_order:
        selected_record = cast(Mapping[str, object], selection_cases[case])
        opened_record = cast(Mapping[str, object], future_cases[case])
        family = str(selected_record.get("selected_family", ""))
        if family not in family_summaries:
            raise ValueError(f"{case}: selected family is unknown")
        family_cases = family_summaries[family].get("case_results")
        if not isinstance(family_cases, Mapping) or case not in family_cases:
            raise ValueError(f"{case}: selected family case record is missing")
        family_record = cast(Mapping[str, object], family_cases[case])
        outputs = family_record.get("outputs")
        selector = family_record.get("selector")
        if not isinstance(outputs, Mapping) or not isinstance(selector, Mapping):
            raise ValueError(f"{case}: selected family output is malformed")
        if family_record.get("future_metrics_opened") is not False:
            raise ValueError(f"{case}: family output opened future metrics")
        if selector.get("selected_method") != selected_record.get(
            "selected_within_family_method"
        ):
            raise ValueError(f"{case}: within-family selections differ")
        if opened_record.get("selected_family") != family:
            raise ValueError(f"{case}: opened family differs from selection")

        split_path = root / case / "split.json"
        split = _load_json(split_path)
        start, end = (
            int(cast(Any, value)) for value in cast(Sequence[object], split["test"])
        )
        if start != int(cast(Any, selected_record["train_end_frame_exclusive"])):
            raise ValueError(f"{case}: evaluation boundary differs from selection")
        method_paths = {
            method: Path(str(outputs[method])).resolve() for method in METHODS
        }
        if any(not path.is_file() for path in method_paths.values()):
            raise FileNotFoundError(f"{case}: one or more method outputs are missing")
        if sha256_file(method_paths["backbone"]) != family_record.get(
            "backbone_sha256"
        ):
            raise ValueError(f"{case}: backbone identity changed")
        selected_output = _validated_identity(
            selected_record.get("output"), label=f"{case}.selected_output"
        )
        opened_output = _validated_identity(
            opened_record.get("selected_output"), label=f"{case}.opened_output"
        )
        if not (
            sha256_file(selected_output)
            == sha256_file(opened_output)
            == sha256_file(method_paths["validation_selected"])
        ):
            raise ValueError(f"{case}: operational output identity changed")

        frame_metrics = {
            method: _case_metrics(root, case, path, start=start, end=end)
            for method, path in method_paths.items()
        }
        methods = {
            method: _mean_metrics(values) for method, values in frame_metrics.items()
        }
        opened_metrics = opened_record.get("selected_future_metrics")
        if not isinstance(opened_metrics, Mapping) or any(
            not np.isclose(
                methods["validation_selected"][metric],
                float(opened_metrics[metric]),
                atol=1e-12,
            )
            for metric in METRICS
        ):
            raise ValueError(f"{case}: recomputed operational metrics changed")
        rows.append(
            {
                "case": case,
                "selected_family": family,
                "selected_within_family_method": str(selector["selected_method"]),
                "frame_interval": [start, end],
                "methods": methods,
                "frame_metrics": frame_metrics,
            }
        )
        input_identities[case] = {
            "split": _file_identity(split_path),
            "final_data": _file_identity(root / case / "final_data.pkl"),
            "gt_track_3d": _file_identity(root / case / "gt_track_3d.pkl"),
            "methods": {
                method: _file_identity(path) for method, path in method_paths.items()
            },
        }

    accepted = [row for row in rows if row["selected_family"] != "alpha_0000"]
    if not accepted:
        raise ValueError("selection contains no nonzero MatPhys family")
    accepted_summary = summarize_matphys_overlay_rows(accepted)
    all_summary = summarize_matphys_overlay_rows(rows)
    for summary_rows, summary, seed_offset in (
        (accepted, accepted_summary, 0),
        (rows, all_summary, 100),
    ):
        summary["paired_bootstrap"] = {
            method: _paired_summary(
                summary_rows,
                method=method,
                bootstrap_samples=bootstrap_samples,
                bootstrap_block_length=bootstrap_block_length,
                bootstrap_seed=bootstrap_seed + seed_offset + index,
            )
            for index, method in enumerate(METHODS[1:])
        }
        summary["horizons"] = {
            method: _horizon_summary(summary_rows, method=method)
            for method in METHODS[1:]
        }

    public_rows = []
    for row in rows:
        public_rows.append(
            {key: value for key, value in row.items() if key != "frame_metrics"}
        )
    report = {
        "schema_version": 1,
        "contract": MATPHYS_RESIDUAL_OVERLAY_AUDIT_CONTRACT,
        "claim_boundary": (
            "Post-open exploratory audit of already frozen PhysTwin-22 MatPhys "
            "families and residual overlays; not an independent MatPhys reproduction, "
            "fresh confirmation, or state-of-the-art claim."
        ),
        "future_metrics_opened": True,
        "selection_changed": False,
        "methods": {
            "backbone": "raw prefix-selected physical family",
            "bayesian_anchor": "frozen causal Bayesian residual anchor",
            "last_residual": "frozen causal last-residual control",
            "validation_selected": "pre-existing prefix-selected operational overlay",
        },
        "selection_summary": {
            "path": str(selection_path),
            "sha256": selection_sha256,
        },
        "future_summary": {"path": str(future_path), "sha256": future_sha256},
        "primary_nonzero_matphys_subset": accepted_summary,
        "secondary_full_fallback_stack": all_summary,
        "selected_family_counts": {
            family: sum(row["selected_family"] == family for row in rows)
            for family in family_summaries
        },
        "per_case": public_rows,
        "inputs": input_identities,
        "interpretation_boundary": {
            "matphys_role": (
                "MatPhys proposes an object-disjoint spring field; official PhysTwin/Warp "
                "produces the physical trajectory."
            ),
            "online_supervision": (
                "The frozen overlay selector uses permitted prefix manual 3D tracks."
            ),
            "bayesian_point_novelty": (
                "The last-residual control is retained because uncertainty does not by "
                "itself imply a better point mean."
            ),
            "calibration": "not established by this point-estimate audit",
        },
    }
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **report,
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
    }
