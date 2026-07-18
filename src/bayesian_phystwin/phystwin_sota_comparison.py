"""Absolute PhysTwin benchmark aggregation for external table comparisons."""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import platform
from pathlib import Path
from typing import Mapping
import warnings

import numpy as np

from .phystwin_confirmatory import DEVELOPMENT_CASES, _implementation_sha256
from .phystwin_official_evaluation import official_phystwin_metrics_by_frame


PHYSTWIN_TABLE1_CASES = (
    "double_lift_cloth_1",
    "double_lift_cloth_3",
    "double_lift_sloth",
    "double_lift_zebra",
    "double_stretch_sloth",
    "double_stretch_zebra",
    "rope_double_hand",
    "single_clift_cloth_1",
    "single_clift_cloth_3",
    "single_lift_cloth",
    "single_lift_cloth_1",
    "single_lift_cloth_3",
    "single_lift_cloth_4",
    "single_lift_dinosor",
    "single_lift_rope",
    "single_lift_sloth",
    "single_lift_zebra",
    "single_push_rope",
    "single_push_rope_1",
    "single_push_rope_4",
    "single_push_sloth",
    "weird_package",
)


def _load_pickle(payload: bytes) -> object:
    return pickle.loads(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_file(path: Path) -> tuple[bytes, dict[str, str]]:
    """Read the exact bytes used by evaluation and bind their path identity."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = resolved.read_bytes()
    return payload, {
        "path": str(resolved),
        "sha256": _payload_sha256(payload),
    }


def _bind_snapshot(
    path: Path,
    bound_inputs: dict[str, dict[str, str]],
) -> tuple[bytes, dict[str, str]]:
    payload, identity = _snapshot_file(path)
    previous = bound_inputs.setdefault(identity["path"], identity)
    if previous != identity:
        raise RuntimeError(
            f"input path changed between uses during aggregation: {identity['path']}"
        )
    return payload, identity


def _require_bound_inputs_unchanged(
    bound_inputs: Mapping[str, Mapping[str, str]],
) -> None:
    for path_text, expected in bound_inputs.items():
        path = Path(path_text)
        try:
            current = {"path": path_text, "sha256": _sha256(path)}
        except FileNotFoundError as error:
            raise RuntimeError(
                f"input disappeared during aggregation: {path_text}"
            ) from error
        if current != expected:
            raise RuntimeError(f"input changed during aggregation: {path_text}")


def _module_file_identity(module_name: str) -> dict[str, str] | None:
    module = importlib.import_module(module_name)
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        return None
    path = Path(module_path).resolve()
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": _sha256(path)}


def _evaluator_identity() -> dict[str, object]:
    """Bind source plus the optional nearest-neighbor runtime implementation."""

    scipy_version: str | None = None
    backend = "numpy_chunked_fallback"
    backend_module: dict[str, str] | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            import scipy
            from scipy.spatial import cKDTree
    except (ImportError, OSError, ValueError, Warning):
        pass
    else:
        scipy_version = str(scipy.__version__)
        backend = "scipy.spatial.cKDTree"
        backend_module = _module_file_identity(cKDTree.__module__)

    evaluator_path = Path(
        official_phystwin_metrics_by_frame.__code__.co_filename
    ).resolve()
    return {
        "package_implementation_sha256": _implementation_sha256(),
        "source_files": {
            "aggregator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "official_evaluator": {
                "path": str(evaluator_path),
                "sha256": _sha256(evaluator_path),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "scipy": scipy_version,
            "nearest_neighbor_backend": backend,
            "nearest_neighbor_module": backend_module,
        },
    }


def _trajectory_path(template: str, case: str) -> Path:
    if "{case}" not in template:
        raise ValueError("each method path template must contain '{case}'")
    return Path(template.format(case=case))


def _aggregate_cohort(
    case_names: tuple[str, ...],
    per_case: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not case_names:
        raise ValueError("a benchmark cohort cannot be empty")
    metric_names = ("chamfer_distance_m", "track_error_m")
    result: dict[str, object] = {
        "case_count": len(case_names),
        "cases": list(case_names),
        "frame_count": int(
            sum(int(per_case[case]["frame_count"]) for case in case_names)
        ),
    }
    for metric in metric_names:
        case_means = np.asarray(
            [per_case[case]["means"][metric] for case in case_names],
            dtype=float,
        )
        frame_values = np.concatenate(
            [
                np.asarray(per_case[case]["frames"][metric], dtype=float)
                for case in case_names
            ]
        )
        result[metric] = {
            "equal_case_mean_m": float(np.mean(case_means)),
            "frame_weighted_mean_m": float(np.mean(frame_values)),
        }
    return result


def aggregate_phystwin_sota_comparison(
    data_root: str | Path,
    methods: Mapping[str, str],
    output_path: str | Path,
    *,
    hash_inputs: bool = True,
) -> dict[str, object]:
    """Aggregate absolute released-metric results over all 22 benchmark cases.

    ``methods`` maps a display name to a trajectory path template containing
    ``{case}``. Both the table-compatible all-case cohort and the untouched
    19-case cohort are emitted. No external paper number is embedded or treated
    as a locally reproduced result.
    """

    if not methods:
        raise ValueError("at least one trajectory method is required")
    for template in methods.values():
        if "{case}" not in template:
            raise ValueError("each method path template must contain '{case}'")
    if not hash_inputs:
        raise ValueError(
            "input identity hashing is mandatory for a scientific comparison"
        )
    evaluator_identity = _evaluator_identity()
    bound_inputs: dict[str, dict[str, str]] = {}
    root = Path(data_root)
    manifest_path = root / "evaluation_subset_manifest.json"
    manifest_payload, manifest_identity = _bind_snapshot(
        manifest_path,
        bound_inputs,
    )
    manifest = json.loads(manifest_payload.decode("utf-8"))
    selected = tuple(str(case) for case in manifest["selected_cases"])
    if len(selected) != len(set(selected)):
        raise ValueError("the data manifest contains duplicate cases")
    if selected != PHYSTWIN_TABLE1_CASES:
        missing = sorted(set(PHYSTWIN_TABLE1_CASES) - set(selected))
        extra = sorted(set(selected) - set(PHYSTWIN_TABLE1_CASES))
        raise ValueError(
            "the selected cases do not match the ordered official PhysTwin "
            f"Table-1 cohort; missing={missing}, extra={extra}"
        )
    available = tuple(str(case) for case in manifest.get("available_cases", ()))
    if available != PHYSTWIN_TABLE1_CASES:
        raise ValueError(
            "the data manifest does not identify the ordered official 22-case cohort"
        )
    missing_development = sorted(set(DEVELOPMENT_CASES) - set(selected))
    if missing_development:
        raise ValueError(
            "the benchmark manifest is missing development cases: "
            + ", ".join(missing_development)
        )
    development = tuple(case for case in selected if case in DEVELOPMENT_CASES)
    confirmation = tuple(case for case in selected if case not in DEVELOPMENT_CASES)

    normalized_methods = {str(name): template for name, template in methods.items()}
    if len(normalized_methods) != len(methods):
        raise ValueError("method names must remain unique when converted to strings")
    per_method_case: dict[str, dict[str, dict[str, object]]] = {
        name: {} for name in normalized_methods
    }
    for case in selected:
        case_root = root / case
        final_path = case_root / "final_data.pkl"
        track_path = case_root / "gt_track_3d.pkl"
        split_path = case_root / "split.json"
        final_payload, final_identity = _bind_snapshot(final_path, bound_inputs)
        track_payload, track_identity = _bind_snapshot(track_path, bound_inputs)
        split_payload, split_identity = _bind_snapshot(split_path, bound_inputs)
        trajectory_snapshots = {
            method_name: _bind_snapshot(
                _trajectory_path(template, case),
                bound_inputs,
            )
            for method_name, template in normalized_methods.items()
        }

        data = _load_pickle(final_payload)
        tracks = np.asarray(_load_pickle(track_payload), dtype=float)
        split = json.loads(split_payload.decode("utf-8"))
        observed = np.asarray(data["object_points"], dtype=float)
        visible = np.asarray(data["object_visibilities"], dtype=bool)
        surface_count = len(observed[0]) + len(data["surface_points"])
        start, end = (int(value) for value in split["test"])
        shared_inputs: dict[str, object] = {
            "final_data": final_identity,
            "gt_track_3d": track_identity,
            "split": split_identity,
        }

        for method_name, template in normalized_methods.items():
            trajectory_payload, trajectory_identity = trajectory_snapshots[method_name]
            trajectory = np.asarray(_load_pickle(trajectory_payload), dtype=float)
            frame_len = int(split["frame_len"])
            if trajectory.ndim != 3 or trajectory.shape[2] != 3:
                raise ValueError(
                    f"{method_name} trajectory for {case} must have shape (T, N, 3)"
                )
            if len(trajectory) != frame_len:
                raise ValueError(
                    f"{method_name} trajectory for {case} has {len(trajectory)} "
                    f"frames; expected the complete {frame_len}-frame sequence"
                )
            if not np.isfinite(trajectory).all():
                raise ValueError(
                    f"{method_name} trajectory for {case} contains non-finite values"
                )
            metrics = official_phystwin_metrics_by_frame(
                trajectory,
                observed,
                visible,
                tracks,
                num_surface_points=surface_count,
                start_frame=start,
                end_frame=end,
            )
            inputs = {
                **shared_inputs,
                "trajectory": trajectory_identity,
            }
            per_method_case[method_name][case] = {
                "frame_interval": [start, end],
                "frame_count": end - start,
                "means": {
                    metric: float(np.mean(values)) for metric, values in metrics.items()
                },
                "frames": {
                    metric: np.asarray(values, dtype=float).tolist()
                    for metric, values in metrics.items()
                },
                "inputs": inputs,
            }

    method_results: dict[str, object] = {}
    for method_name, template in normalized_methods.items():
        per_case = per_method_case[method_name]
        method_results[method_name] = {
            "path_template": template,
            "cohorts": {
                "all_22_table_compatible": _aggregate_cohort(selected, per_case),
                "development_3": _aggregate_cohort(development, per_case),
                "confirmation_19": _aggregate_cohort(confirmation, per_case),
            },
            "per_case": per_case,
        }

    _require_bound_inputs_unchanged(bound_inputs)
    if _evaluator_identity() != evaluator_identity:
        raise RuntimeError("evaluator implementation changed during aggregation")

    result = {
        "schema_version": 2,
        "metric_contract": {
            "chamfer_distance_m": "released one-way L1 nearest-surface distance",
            "track_error_m": "released initial-nearest-node Euclidean track error",
            "aggregation": ["equal case", "frame weighted"],
            "external_reference_policy": (
                "published numbers are not embedded or labeled as reproduced"
            ),
            "input_identity_policy": (
                "metrics use pre-load byte snapshots; every bound path must retain "
                "the same SHA-256 through aggregation"
            ),
        },
        "data_manifest": {
            **manifest_identity,
            "selected_cases": list(selected),
        },
        "evaluator_identity": evaluator_identity,
        "development_cases": list(DEVELOPMENT_CASES),
        "methods": method_results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["output_path"] = str(output.resolve())
    return result
