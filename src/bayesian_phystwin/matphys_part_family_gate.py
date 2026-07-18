"""Disjoint fit/validation gate for causal MatPhys graph-part residuals."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .matphys_causal_bridge import validate_causal_training_audit
from .matphys_graph_parts import GRAPH_PART_PROXY_CONTRACT
from .matphys_part_model import PART_AWARE_MODEL_CONTRACT
from .phystwin_confirmation_lock import exclusively_owned_confirmation_output
from .phystwin_confirmatory import _lock_protocol
from .phystwin_official_evaluation import evaluate_official_phystwin_interval


PART_FAMILY_GATE_CONTRACT = "causal-matphys-part-family-fit-validation-v1"
PAIRED_VALIDATION_CONTRACT = "paired-fit-validation-trajectories-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _resolve(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validated_identity(
    manifest_path: Path,
    identity: object,
    *,
    label: str,
) -> Path:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} must be a file identity")
    path = _resolve(manifest_path, identity.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(identity.get("sha256", ""))
    if not expected or _sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def choose_part_family(
    teacher_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    *,
    minimum_relative_score_improvement: float = 0.0,
    maximum_metric_regression: float = 0.0,
) -> dict[str, object]:
    """Accept a learned residual only when its disjoint validation transfers."""

    if minimum_relative_score_improvement < 0.0:
        raise ValueError("minimum score improvement must be nonnegative")
    if maximum_metric_regression < 0.0:
        raise ValueError("maximum metric regression must be nonnegative")
    ratios: dict[str, float] = {}
    for metric in ("chamfer_distance_m", "track_error_m"):
        teacher = float(teacher_metrics[metric])
        candidate = float(candidate_metrics[metric])
        if not np.isfinite(teacher) or not np.isfinite(candidate):
            raise ValueError("family-gate metrics must be finite")
        if teacher <= 0.0 or candidate < 0.0:
            raise ValueError("family-gate reference metrics must be positive")
        ratios[metric] = candidate / teacher
    candidate_score = float(np.mean(list(ratios.values())))
    relative_improvement = 1.0 - candidate_score
    no_metric_regression = all(
        ratio <= 1.0 + maximum_metric_regression for ratio in ratios.values()
    )
    accepted = bool(
        relative_improvement > minimum_relative_score_improvement
        and no_metric_regression
    )
    return {
        "selected_family": "learned_part_residual" if accepted else "exact_teacher",
        "learned_accepted": accepted,
        "candidate_metric_ratios": ratios,
        "candidate_normalized_score": candidate_score,
        "relative_score_improvement": relative_improvement,
        "no_metric_regression": no_metric_regression,
    }


def _official_metrics(
    case_root: Path,
    trajectory: np.ndarray,
    *,
    fit_end: int,
    train_end: int,
) -> dict[str, object]:
    final_data = _load_pickle(case_root / "final_data.pkl")
    object_points = np.asarray(final_data["object_points"], dtype=float)
    surface_count = object_points.shape[1] + len(
        np.asarray(final_data["surface_points"])
    )
    return evaluate_official_phystwin_interval(
        trajectory,
        object_points,
        np.asarray(final_data["object_visibilities"], dtype=bool),
        np.asarray(_load_pickle(case_root / "gt_track_3d.pkl"), dtype=float),
        num_surface_points=surface_count,
        start_frame=fit_end,
        end_frame=train_end,
    )


def _validate_spring_summary(
    manifest_path: Path,
    case_entry: Mapping[str, object],
    residual_log_scale: float,
) -> dict[str, object]:
    path = _validated_identity(
        manifest_path,
        case_entry.get("spring_field_summary"),
        label=f"{case_entry.get('name')}.spring_field_summary",
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    overall = summary.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("spring summary omits overall statistics")
    minimum = float(overall["minimum"])
    maximum = float(overall["maximum"])
    lower = float(np.exp(-residual_log_scale))
    upper = float(np.exp(residual_log_scale))
    tolerance = 1e-5
    if minimum < lower - tolerance or maximum > upper + tolerance:
        raise ValueError("learned spring field exceeds its teacher-centered bound")
    by_part = summary.get("by_part")
    if not isinstance(by_part, list) or len(by_part) < 2:
        raise ValueError("part-aware spring summary needs multiple parts")
    part_means = np.asarray([float(record["mean"]) for record in by_part])
    if not np.all(np.isfinite(part_means)):
        raise ValueError("part spring ratios must be finite")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "ratio_bounds": [minimum, maximum],
        "part_mean_range": float(np.max(part_means) - np.min(part_means)),
        "part_means": part_means.tolist(),
    }


@exclusively_owned_confirmation_output
def run_matphys_part_family_gate(
    data_root: str | Path,
    output_dir: str | Path,
    candidate_manifest: str | Path,
    *,
    case_names: Sequence[str] | None = None,
    minimum_relative_score_improvement: float = 0.0,
    maximum_metric_regression: float = 0.0,
    required_learned_case_count: int = 1,
    source_protocol: str | Path | None = None,
) -> dict[str, object]:
    """Gate learned graph parts without loading or scoring future trajectories."""

    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    manifest_path = Path(candidate_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported candidate manifest schema")
    backbone = manifest.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("candidate manifest omits its backbone")
    if backbone.get("future_observations_used") is not False:
        raise ValueError("candidate backbone does not forbid future observations")
    if backbone.get("proxy_contract") != GRAPH_PART_PROXY_CONTRACT:
        raise ValueError("family gate requires the causal graph-part proxy")
    parameterization = backbone.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise ValueError("candidate manifest omits teacher parameterization")
    if parameterization.get("part_model_contract") != PART_AWARE_MODEL_CONTRACT:
        raise ValueError("candidate manifest uses an unsupported part adapter")
    residual_log_scale = float(parameterization.get("residual_log_scale", -1.0))
    if not np.isfinite(residual_log_scale) or residual_log_scale < 0.0:
        raise ValueError("candidate manifest has an invalid residual bound")

    checkpoint = _validated_identity(
        manifest_path,
        backbone.get("checkpoint"),
        label="candidate checkpoint",
    )
    audit_path = _validated_identity(
        manifest_path,
        backbone.get("causal_training_audit"),
        label="candidate causal audit",
    )
    audit = validate_causal_training_audit(audit_path, checkpoint)
    audit_cases = {str(record["name"]): record for record in audit["cases"]}

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("candidate manifest contains no cases")
    entries: dict[str, Mapping[str, object]] = {}
    for entry in raw_cases:
        if not isinstance(entry, Mapping):
            raise ValueError("candidate case entry must be an object")
        name = str(entry.get("name", ""))
        if not name or name in entries:
            raise ValueError("candidate case names must be nonempty and unique")
        entries[name] = entry
    requested = tuple(case_names or entries)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested cases must be nonempty and unique")
    if not 0 <= required_learned_case_count <= len(requested):
        raise ValueError("required learned-case count exceeds the source panel")
    if tuple(entries) != requested:
        raise ValueError("candidate manifest order differs from the requested source panel")
    if set(audit_cases) != set(requested):
        raise ValueError("candidate audit and manifest cases disagree")

    protocol_identity = None
    if source_protocol is not None:
        protocol_path = Path(source_protocol).resolve()
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        protocol_identity = {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        }
    specification = {
        "contract": PART_FAMILY_GATE_CONTRACT,
        "candidate_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "cases": list(requested),
        "selection_interval": "[fit_end, released_train_end)",
        "future_trajectory_access": "forbidden; gate reads truncated validation artifacts only",
        "reference": "paired exact released-PhysTwin teacher rerun",
        "minimum_relative_score_improvement": minimum_relative_score_improvement,
        "maximum_metric_regression": maximum_metric_regression,
        "required_learned_case_count": required_learned_case_count,
        "fallback": "exact_teacher",
        "status": "source/development family selection; not future evidence",
        "source_protocol": protocol_identity,
    }
    locked = _lock_protocol(output, specification)
    case_results: dict[str, dict[str, object]] = {}
    for case in requested:
        entry = entries[case]
        audit_case = audit_cases[case]
        split_path = root / case / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_start, train_end = (int(value) for value in split["train"])
        if train_start != 0:
            raise ValueError(f"{case}: family gate requires a zero-based train split")
        fit_end = int(audit_case["evidence_end_frame_exclusive"])
        if not 1 <= fit_end < train_end:
            raise ValueError(f"{case}: no disjoint validation suffix remains")
        if audit_case.get("validation_frame_interval") != [fit_end, train_end]:
            raise ValueError(f"{case}: causal audit validation interval changed")
        validation = entry.get("causal_validation")
        if not isinstance(validation, Mapping):
            raise ValueError(f"{case}: export omits paired causal validation")
        if validation.get("contract") != PAIRED_VALIDATION_CONTRACT:
            raise ValueError(f"{case}: unsupported paired validation contract")
        if (
            int(validation.get("fit_end_frame_exclusive", -1)) != fit_end
            or int(validation.get("validation_end_frame_exclusive", -1))
            != train_end
        ):
            raise ValueError(f"{case}: paired validation boundaries changed")
        candidate_path = _validated_identity(
            manifest_path,
            validation.get("candidate"),
            label=f"{case}.candidate_validation",
        )
        teacher_path = _validated_identity(
            manifest_path,
            validation.get("teacher"),
            label=f"{case}.teacher_validation",
        )
        candidate = np.asarray(_load_pickle(candidate_path), dtype=float)
        teacher = np.asarray(_load_pickle(teacher_path), dtype=float)
        if candidate.shape != teacher.shape or candidate.ndim != 3:
            raise ValueError(f"{case}: paired validation trajectories disagree")
        if candidate.shape[0] != train_end or candidate.shape[2] != 3:
            raise ValueError(f"{case}: validation artifacts are not prefix-truncated")
        if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(teacher)):
            raise ValueError(f"{case}: validation trajectories contain non-finite values")
        teacher_metrics = _official_metrics(
            root / case,
            teacher,
            fit_end=fit_end,
            train_end=train_end,
        )
        candidate_metrics = _official_metrics(
            root / case,
            candidate,
            fit_end=fit_end,
            train_end=train_end,
        )
        decision = choose_part_family(
            teacher_metrics,
            candidate_metrics,
            minimum_relative_score_improvement=minimum_relative_score_improvement,
            maximum_metric_regression=maximum_metric_regression,
        )
        selected_identity = (
            validation["candidate"]
            if decision["learned_accepted"]
            else validation["teacher"]
        )
        case_results[case] = {
            "fit_end_frame_exclusive": fit_end,
            "validation_end_frame_exclusive": train_end,
            "teacher_validation_metrics": teacher_metrics,
            "candidate_validation_metrics": candidate_metrics,
            "selected_validation_metrics": (
                candidate_metrics if decision["learned_accepted"] else teacher_metrics
            ),
            "decision": decision,
            "selected_validation_trajectory": dict(selected_identity),
            "spring_field": _validate_spring_summary(
                manifest_path,
                entry,
                residual_log_scale,
            ),
            "inputs": {
                "split": {"path": str(split_path), "sha256": _sha256_file(split_path)},
                "final_data": {
                    "path": str((root / case / "final_data.pkl").resolve()),
                    "sha256": _sha256_file(root / case / "final_data.pkl"),
                },
                "gt_track_3d": {
                    "path": str((root / case / "gt_track_3d.pkl").resolve()),
                    "sha256": _sha256_file(root / case / "gt_track_3d.pkl"),
                },
            },
        }

    selected_count = sum(
        int(result["decision"]["learned_accepted"])
        for result in case_results.values()
    )
    aggregate: dict[str, dict[str, float]] = {}
    for family, metrics_key in (
        ("teacher", "teacher_validation_metrics"),
        ("candidate", "candidate_validation_metrics"),
        ("selected", "selected_validation_metrics"),
    ):
        aggregate[family] = {
            metric: float(
                np.mean(
                    [
                        float(result[metrics_key][metric])
                        for result in case_results.values()
                    ]
                )
            )
            for metric in ("chamfer_distance_m", "track_error_m")
        }
    aggregate_both_improved = all(
        aggregate["selected"][metric] < aggregate["teacher"][metric]
        for metric in ("chamfer_distance_m", "track_error_m")
    )
    source_gate_passed = bool(
        selected_count >= required_learned_case_count
        and aggregate_both_improved
    )
    summary = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": specification,
        "learned_acceptance_count": selected_count,
        "teacher_fallback_count": len(case_results) - selected_count,
        "aggregate_validation_metrics": aggregate,
        "aggregate_both_metrics_improved": aggregate_both_improved,
        "source_gate_passed": source_gate_passed,
        "case_results": case_results,
        "future_metrics_opened": False,
    }
    summary_path = output / "matphys_part_family_gate.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary
