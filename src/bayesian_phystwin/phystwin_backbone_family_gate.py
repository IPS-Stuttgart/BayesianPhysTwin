"""Causal validation gate across PhysTwin-compatible backbone families."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phystwin_confirmation_lock import exclusively_owned_confirmation_output
from .phystwin_confirmatory import DEVELOPMENT_CASES, _lock_protocol
from .phystwin_official_evaluation import evaluate_official_phystwin_interval
from .phystwin_sota_comparison import PHYSTWIN_TABLE1_CASES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalized_validation_score(
    metrics: Mapping[str, object],
    reference: Mapping[str, object],
) -> float:
    """Balance CD and track error against one common family reference."""

    values = []
    for name in ("chamfer_distance_m", "track_error_m"):
        denominator = float(reference[name])
        numerator = float(metrics[name])
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            raise ValueError("family-gate metrics must be finite")
        if denominator <= 0.0 or numerator < 0.0:
            raise ValueError("family-gate metrics must have positive references")
        values.append(numerator / denominator)
    return float(np.mean(values))


def choose_backbone_family(
    validation_metrics: Mapping[str, Mapping[str, object]],
    reference_metrics: Mapping[str, object],
) -> tuple[str, dict[str, float]]:
    """Choose the minimum common validation score, preserving tie order."""

    if not validation_metrics:
        raise ValueError("at least one backbone family is required")
    scores = {
        family: normalized_validation_score(metrics, reference_metrics)
        for family, metrics in validation_metrics.items()
    }
    selected = min(scores, key=scores.get)
    return selected, scores


def _baseline_validation_metrics(
    data_root: Path,
    case: str,
    trajectory_path: Path,
    *,
    fit_end: int,
    train_end: int,
) -> dict[str, object]:
    final_data = _load_pickle(data_root / case / "final_data.pkl")
    trajectory = np.asarray(_load_pickle(trajectory_path), dtype=float)
    object_points = np.asarray(final_data["object_points"], dtype=float)
    visibility = np.asarray(final_data["object_visibilities"], dtype=bool)
    tracks = np.asarray(_load_pickle(data_root / case / "gt_track_3d.pkl"))
    surface_count = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    return evaluate_official_phystwin_interval(
        trajectory,
        object_points,
        visibility,
        tracks,
        num_surface_points=surface_count,
        start_frame=fit_end,
        end_frame=train_end,
    )


def _selected_validation_metrics(
    case_result: Mapping[str, object],
    baseline_validation: Mapping[str, object],
) -> dict[str, object]:
    method = str(case_result["selector"]["selected_method"])
    outputs = case_result["outputs"]
    if method == "backbone":
        return dict(baseline_validation)
    if method == "bayesian_anchor":
        summary_path = Path(str(outputs["bayesian_anchor"])).parent / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return dict(summary["selection"]["selected_candidate"]["official_evaluation"])
    if method == "last_residual":
        summary_path = (
            Path(str(outputs["last_residual"])).parent.parent / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return dict(
            summary["methods"]["last_residual"]["selection"][
                "selected_candidate"
            ]["official_evaluation"]
        )
    raise ValueError(f"unsupported within-family method: {method}")


def _future_metrics(
    data_root: Path,
    case: str,
    trajectory_path: Path,
    *,
    train_end: int,
    frame_count: int,
) -> dict[str, object]:
    final_data = _load_pickle(data_root / case / "final_data.pkl")
    object_points = np.asarray(final_data["object_points"], dtype=float)
    return evaluate_official_phystwin_interval(
        np.asarray(_load_pickle(trajectory_path), dtype=float),
        object_points,
        np.asarray(final_data["object_visibilities"], dtype=bool),
        np.asarray(_load_pickle(data_root / case / "gt_track_3d.pkl")),
        num_surface_points=object_points.shape[1]
        + len(np.asarray(final_data["surface_points"])),
        start_frame=train_end,
        end_frame=frame_count,
    )


def _copy_exact(source: Path, destination: Path) -> str:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    source_hash = _sha256_file(source)
    if _sha256_file(destination) != source_hash:
        raise RuntimeError("family-gate trajectory changed while being staged")
    return source_hash


@exclusively_owned_confirmation_output
def run_backbone_family_gate(
    data_root: str | Path,
    output_dir: str | Path,
    family_summaries: Mapping[str, str | Path],
    *,
    case_names: Sequence[str] | None = None,
    development_smoke: bool = False,
) -> dict[str, object]:
    """Select one future-blind backbone family on the permitted validation split."""

    if len(family_summaries) < 2:
        raise ValueError("the backbone-family gate requires at least two families")
    if len(family_summaries) != len(set(family_summaries)):
        raise ValueError("backbone family names must be unique")
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    loaded: dict[str, dict[str, object]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for family, raw_path in family_summaries.items():
        path = Path(raw_path).resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary.get("case_results"), dict):
            raise ValueError(f"{family}: overlay summary has no case results")
        loaded[family] = summary
        provenance[family] = {"path": str(path), "sha256": _sha256_file(path)}

    requested = tuple(case_names or PHYSTWIN_TABLE1_CASES)
    if len(requested) != len(set(requested)) or not requested:
        raise ValueError("case names must be nonempty and unique")
    if development_smoke:
        expected = tuple(case for case in DEVELOPMENT_CASES if case in requested)
        if requested != expected:
            raise ValueError("development cases must be an ordered declared subset")
    elif requested != PHYSTWIN_TABLE1_CASES:
        raise ValueError("full family gating requires the ordered 22-case cohort")
    for family, summary in loaded.items():
        missing = [case for case in requested if case not in summary["case_results"]]
        if missing:
            raise ValueError(f"{family}: overlay summary omits {missing}")

    specification = {
        "method": "causal validation gate across backbone families",
        "families": provenance,
        "cases": list(requested),
        "reference_family": next(iter(family_summaries)),
        "score": (
            "equal weight mean of validation CD and track error, each normalized "
            "to the reference family's raw backbone"
        ),
        "within_family_candidate": "existing validation-selected trajectory",
        "tie_break": "family declaration order; reference family declared first",
        "information_boundary": "selection ends at released training boundary",
        "status": (
            "development-only integration smoke; not cohort evidence"
            if development_smoke
            else "exploratory on the previously examined PhysTwin cohort"
        ),
    }
    locked = _lock_protocol(output, specification)
    reference_family = next(iter(family_summaries))
    case_results: dict[str, dict[str, object]] = {}
    selection_counts = {family: 0 for family in family_summaries}
    for case in requested:
        first = loaded[reference_family]["case_results"][case]
        fit_end = int(first["fit_end_frame_exclusive"])
        train_end = int(first["train_end_frame_exclusive"])
        frame_count = int(first["frame_count"])
        validation_by_family: dict[str, dict[str, object]] = {}
        trajectory_by_family: dict[str, Path] = {}
        method_by_family: dict[str, str] = {}
        for family, summary in loaded.items():
            result = summary["case_results"][case]
            boundaries = (
                int(result["fit_end_frame_exclusive"]),
                int(result["train_end_frame_exclusive"]),
                int(result["frame_count"]),
            )
            if boundaries != (fit_end, train_end, frame_count):
                raise ValueError(f"{case}: family split boundaries disagree")
            backbone_path = Path(str(result["outputs"]["backbone"])).resolve()
            baseline_validation = _baseline_validation_metrics(
                root,
                case,
                backbone_path,
                fit_end=fit_end,
                train_end=train_end,
            )
            validation_by_family[family] = _selected_validation_metrics(
                result, baseline_validation
            )
            trajectory_by_family[family] = Path(
                str(result["outputs"]["validation_selected"])
            ).resolve()
            method_by_family[family] = str(result["selector"]["selected_method"])
        reference_validation = _baseline_validation_metrics(
            root,
            case,
            Path(str(first["outputs"]["backbone"])).resolve(),
            fit_end=fit_end,
            train_end=train_end,
        )
        selected_family, scores = choose_backbone_family(
            validation_by_family, reference_validation
        )
        selection_counts[selected_family] += 1
        selected_source = trajectory_by_family[selected_family]
        staged = output / "cases" / case / "trajectory.pkl"
        selected_hash = _copy_exact(selected_source, staged)
        family_test = {
            family: _future_metrics(
                root,
                case,
                trajectory,
                train_end=train_end,
                frame_count=frame_count,
            )
            for family, trajectory in trajectory_by_family.items()
        }
        case_results[case] = {
            "selected_family": selected_family,
            "selected_within_family_method": method_by_family[selected_family],
            "validation_scores": scores,
            "validation_metrics": validation_by_family,
            "reference_raw_validation_metrics": reference_validation,
            "test_metrics_by_family": family_test,
            "selected_test_metrics": family_test[selected_family],
            "output": {"path": str(staged), "sha256": selected_hash},
        }

    metrics = ("chamfer_distance_m", "track_error_m")
    selected_mean = {
        metric: float(
            np.mean(
                [result["selected_test_metrics"][metric] for result in case_results.values()]
            )
        )
        for metric in metrics
    }
    family_means = {
        family: {
            metric: float(
                np.mean(
                    [
                        result["test_metrics_by_family"][family][metric]
                        for result in case_results.values()
                    ]
                )
            )
            for metric in metrics
        }
        for family in family_summaries
    }
    summary = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": specification,
        "selection_counts": selection_counts,
        "case_results": case_results,
        "comparison": {
            "case_count": len(requested),
            "selected_equal_case_mean": selected_mean,
            "family_equal_case_means": family_means,
        },
    }
    summary_path = output / "backbone_family_gate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary
