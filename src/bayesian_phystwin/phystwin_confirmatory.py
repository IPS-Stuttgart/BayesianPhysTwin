"""Locked multi-case confirmation of the constrained PhysTwin residual model."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .phystwin_comparison import compare_phystwin_manifest
from .phystwin_confirmation_lock import exclusively_owned_confirmation_output
from .phystwin_residual_dynamics import (
    PhysTwinResidualDynamicsConfig,
    fit_action_conditioned_residual_dynamics,
)


DEVELOPMENT_CASES = (
    "single_lift_sloth",
    "double_lift_sloth",
    "double_stretch_sloth",
)


@dataclass(frozen=True)
class PhysTwinConfirmatoryProtocol:
    """Preregistered choices for the untouched-case benchmark."""

    fit_fraction: float = 0.75
    rank_candidates: tuple[int, ...] = (1, 2, 4, 8)
    persistence_candidates: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.0)
    ridge_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)
    projection_ridge: float = 1e-6
    interpolation_neighbors: int = 4
    maximum_state_multiplier: float = 1.5
    maximum_residual_m: float = 0.01
    minimum_validation_improvement: float = 0.0
    bootstrap_samples: int = 10000
    bootstrap_block_length: int = 5
    bootstrap_seed: int = 20260710
    development_cases: tuple[str, ...] = DEVELOPMENT_CASES


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _protocol_id(specification: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(specification).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


_CASE_INPUT_FILES = (
    ("final_data", "final_data.pkl"),
    ("baseline_trajectory", "inference.pkl"),
    ("gt_track_3d", "gt_track_3d.pkl"),
)


def _case_source_identity(case_dir: Path) -> dict[str, object]:
    """Capture every source that can affect one fitted case."""

    return {
        "split_sha256": _sha256_file(case_dir / "split.json"),
        "implementation_sha256": _implementation_sha256(),
        "inputs": {
            name: {
                "path": str((case_dir / filename).resolve()),
                "sha256": _sha256_file(case_dir / filename),
            }
            for name, filename in _CASE_INPUT_FILES
        },
    }


def _summary_body_sha256(summary: dict[str, object]) -> str:
    body = {key: value for key, value in summary.items() if key != "cache_identity"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _case_cache_identity(
    case_dir: Path,
    case_output: Path,
    summary: dict[str, object],
    *,
    source_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    outputs = {
        path.relative_to(case_output).as_posix(): _sha256_file(path)
        for path in sorted(case_output.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }
    if not outputs:
        raise RuntimeError(f"case output contains no fitted artifacts: {case_output}")
    source = (
        _case_source_identity(case_dir)
        if source_identity is None
        else source_identity
    )
    return {
        "schema_version": 3,
        **source,
        "summary_body_sha256": _summary_body_sha256(summary),
        "outputs": outputs,
    }


def _seal_case_cache(
    summary: dict[str, object],
    case_dir: Path,
    case_output: Path,
    *,
    expected_source_identity: dict[str, object] | None = None,
) -> None:
    current_source_identity = _case_source_identity(case_dir)
    if (
        expected_source_identity is not None
        and current_source_identity != expected_source_identity
    ):
        raise RuntimeError("case sources changed while the fit was running")
    recorded_inputs = summary.get("inputs")
    if not isinstance(recorded_inputs, dict):
        raise RuntimeError("fitted case does not record input identities")
    for name, expected_identity in current_source_identity["inputs"].items():
        if recorded_inputs.get(name) != expected_identity:
            raise RuntimeError(f"fitted case recorded a different input: {name}")
    summary["cache_identity"] = _case_cache_identity(
        case_dir,
        case_output,
        summary,
        source_identity=current_source_identity,
    )
    path = case_output / "summary.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_cached_case(
    summary: dict[str, object],
    expected_config: dict[str, object],
    case_dir: Path,
    case_output: Path,
    case: str,
) -> None:
    """Reject a cached fit unless its protocol and source artifacts still match."""

    if summary.get("schema_version") != 1 or _canonical_json(
        summary.get("config")
    ) != _canonical_json(expected_config):
        raise RuntimeError(f"cached case uses a different protocol: {case}")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"cached case does not record input identities: {case}")
    expected_source_identity = _case_source_identity(case_dir)
    for name, expected_identity in expected_source_identity["inputs"].items():
        identity = inputs.get(name)
        if identity != expected_identity:
            raise RuntimeError(f"cached case input changed: {case}: {name}")
    if summary.get("cache_identity") != _case_cache_identity(
        case_dir,
        case_output,
        summary,
        source_identity=expected_source_identity,
    ):
        raise RuntimeError(
            f"cached case summary, implementation, source, or output changed: {case}"
        )


def _lock_protocol(output: Path, specification: dict[str, object]) -> dict[str, object]:
    path = output / "locked_protocol.json"
    protocol_id = _protocol_id(specification)
    normalized_specification = json.loads(_canonical_json(specification))
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        if locked.get("protocol_id") != protocol_id or locked.get(
            "specification"
        ) != normalized_specification:
            raise RuntimeError(
                "output directory already contains a different locked protocol"
            )
        return locked
    output.mkdir(parents=True, exist_ok=True)
    locked = {
        "schema_version": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol_id,
        "specification": normalized_specification,
    }
    path.write_text(
        json.dumps(locked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return locked


def _split_for_case(case_dir: Path, fit_fraction: float) -> tuple[int, int, int]:
    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    train_start, train_end = (int(value) for value in split["train"])
    test_start, test_end = (int(value) for value in split["test"])
    if train_start != 0 or test_start != train_end or test_end != int(
        split["frame_len"]
    ):
        raise ValueError(f"unsupported noncontiguous split in {case_dir.name}")
    fit_end = train_start + math.floor(fit_fraction * (train_end - train_start))
    if not 2 < fit_end < train_end:
        raise ValueError(f"split is too short in {case_dir.name}")
    return fit_end, train_end, test_end


def _comparison_manifest(
    data_root: Path,
    output: Path,
    cases: Iterable[str],
) -> dict[str, object]:
    entries = []
    for case in cases:
        case_dir = data_root / case
        split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
        entries.append(
            {
                "name": case,
                "final_data": str((case_dir / "final_data.pkl").resolve()),
                "gt_track_3d": str((case_dir / "gt_track_3d.pkl").resolve()),
                "baseline_trajectory": str((case_dir / "inference.pkl").resolve()),
                "candidate_trajectory": str(
                    (output / "cases" / case / "trajectory.pkl").resolve()
                ),
                "start_frame": int(split["test"][0]),
                "end_frame": int(split["test"][1]),
            }
        )
    return {"schema_version": 1, "cases": entries}


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials < 1:
        return 1.0
    probability = sum(
        math.comb(trials, count) for count in range(0, min(successes, trials - successes) + 1)
    ) / 2**trials
    return min(1.0, 2.0 * probability)


def _case_readout(summary: dict[str, object]) -> dict[str, object]:
    baseline = summary["test"]["baseline_official_evaluation"]
    corrected = summary["test"]["corrected_official_evaluation"]
    changes = {
        metric: 100.0 * (float(corrected[metric]) / float(baseline[metric]) - 1.0)
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    return {
        "accepted_on_validation": summary["selection"]["accepted"],
        "validation_relative_improvement": summary["selection"][
            "relative_improvement"
        ],
        "selected_candidate": summary["selection"]["selected_candidate"],
        "future_baseline": baseline,
        "future_corrected": corrected,
        "future_percent_change": changes,
        "correction": summary["correction"],
    }


def _cohort_readout(
    cases: tuple[str, ...],
    case_results: dict[str, dict[str, object]],
    comparison: dict[str, object],
) -> dict[str, object]:
    changes = {
        metric: np.array(
            [
                case_results[case]["future_percent_change"][metric]
                for case in cases
            ],
            dtype=float,
        )
        for metric in ("chamfer_distance_m", "track_error_m")
    }
    both = sum(
        case_results[case]["future_percent_change"]["chamfer_distance_m"] < 0.0
        and case_results[case]["future_percent_change"]["track_error_m"] < 0.0
        for case in cases
    )
    either_metric_improved = {
        metric: int(np.sum(values < 0.0)) for metric, values in changes.items()
    }
    return {
        "cases": list(cases),
        "case_count": len(cases),
        "validation_acceptance_count": sum(
            bool(case_results[case]["accepted_on_validation"]) for case in cases
        ),
        "improved_case_count": either_metric_improved,
        "improved_on_both_count": both,
        "sign_test_p_two_sided": {
            metric: _binomial_two_sided(count, len(cases))
            for metric, count in either_metric_improved.items()
        },
        "median_percent_change": {
            metric: float(np.median(values)) for metric, values in changes.items()
        },
        "bootstrap": comparison["bootstrap"],
    }


def _fit_case(
    job: tuple[
        Path,
        Path,
        str,
        PhysTwinConfirmatoryProtocol,
        bool,
    ],
) -> tuple[str, dict[str, object]]:
    """Fit one independent case so full-cohort runs can use multiple processes."""

    root, output, case, config, force = job
    case_dir = root / case
    source_identity = _case_source_identity(case_dir)
    fit_end, train_end, frame_count = _split_for_case(
        case_dir, config.fit_fraction
    )
    case_output = output / "cases" / case
    summary_path = case_output / "summary.json"
    residual_config = PhysTwinResidualDynamicsConfig(
        fit_end_frame=fit_end,
        train_end_frame=train_end,
        rank_candidates=config.rank_candidates,
        persistence_candidates=config.persistence_candidates,
        ridge_candidates=config.ridge_candidates,
        projection_ridge=config.projection_ridge,
        interpolation_neighbors=config.interpolation_neighbors,
        maximum_state_multiplier=config.maximum_state_multiplier,
        maximum_residual_m=config.maximum_residual_m,
        minimum_validation_improvement=config.minimum_validation_improvement,
    )
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_cached_case(
            summary, asdict(residual_config), case_dir, case_output, case
        )
    else:
        summary = fit_action_conditioned_residual_dynamics(
            case_dir / "final_data.pkl",
            case_dir / "inference.pkl",
            case_dir / "gt_track_3d.pkl",
            case_output,
            config=residual_config,
        )
        _seal_case_cache(
            summary,
            case_dir,
            case_output,
            expected_source_identity=source_identity,
        )
    return case, {
        "fit_end_frame": fit_end,
        "train_end_frame": train_end,
        "frame_count": frame_count,
        **_case_readout(summary),
    }


@exclusively_owned_confirmation_output
def run_phystwin_confirmatory_benchmark(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    protocol: PhysTwinConfirmatoryProtocol | None = None,
    cases: Iterable[str] | None = None,
    force: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """Run the locked residual protocol and report untouched cases separately."""

    if workers < 1:
        raise ValueError("workers must be positive")
    config = PhysTwinConfirmatoryProtocol() if protocol is None else protocol
    if not 0.0 < config.fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    root = Path(data_root)
    source_manifest_path = root / "evaluation_subset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    available = tuple(str(case) for case in source_manifest["selected_cases"])
    if len(available) != len(set(available)):
        raise ValueError("the data manifest contains duplicate cases")
    selected = available if cases is None else tuple(dict.fromkeys(cases))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError("cases absent from data manifest: " + ", ".join(missing))
    development = tuple(case for case in selected if case in config.development_cases)
    confirmation = tuple(case for case in selected if case not in config.development_cases)
    if not confirmation:
        raise ValueError("the selected cohort contains no confirmatory cases")
    output = Path(output_dir)
    specification = {
        "method": "validation-gated action-conditioned low-rank residual",
        "protocol": asdict(config),
        "data_manifest": {
            "path": str(source_manifest_path.resolve()),
            "selected_cases": list(selected),
        },
        "cohorts": {
            "development": list(development),
            "confirmation": list(confirmation),
        },
        "split_rule": "fit_end=floor(0.75*released_train_end); future=official_test",
    }
    locked = _lock_protocol(output, specification)

    jobs = [(root, output, case, config, force) for case in selected]
    if workers == 1:
        fitted_cases = list(map(_fit_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fitted_cases = list(executor.map(_fit_case, jobs))
    case_results = dict(fitted_cases)

    comparisons: dict[str, dict[str, object]] = {}
    cohorts = {"confirmation": confirmation, "all": selected}
    if development:
        cohorts["development"] = development
    for cohort_name, cohort_cases in cohorts.items():
        manifest = _comparison_manifest(root, output, cohort_cases)
        manifest_path = output / f"comparison_{cohort_name}_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparisons[cohort_name] = compare_phystwin_manifest(
            manifest_path,
            output / f"comparison_{cohort_name}.json",
            samples=config.bootstrap_samples,
            block_length=config.bootstrap_block_length,
            seed=config.bootstrap_seed,
        )

    result = {
        "schema_version": 1,
        "protocol_id": locked["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_results": case_results,
        "cohorts": {
            name: _cohort_readout(cohort_cases, case_results, comparisons[name])
            for name, cohort_cases in cohorts.items()
        },
    }
    result_path = output / "confirmatory_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(result_path.resolve())
    return result
