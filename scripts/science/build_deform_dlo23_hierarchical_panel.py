#!/usr/bin/env python3
"""Build a source-only hierarchical discrepancy panel from a frozen DLO2/DLO3 plan.

Only the exact SHA-256-bound carriers selected by
``plan_deform_dlo23_hierarchical_panel.py`` are opened.  The builder preserves
complete trajectory identity, computes residuals as registered, derives
source-only mechanics/action/contact blocks, and emits the provider-neutral NPZ
contract consumed by ``hierarchical_missing_physics_diagnosis.py``.

No DLO4/DLO5 path or protected target result is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SOURCE_DLOS = ("DLO2", "DLO3")
FORBIDDEN_TOKENS = ("dlo4", "dlo_4", "dlo-4", "dlo5", "dlo_5", "dlo-5", "dlo45", "33361441865")
MAX_FEATURES_PER_RAW_BLOCK = 64


class BuildError(RuntimeError):
    """Raised when the frozen adapter cannot be executed exactly."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbid(path: Path) -> None:
    lower = path.as_posix().lower()
    if any(token in lower for token in FORBIDDEN_TOKENS):
        raise BuildError(f"protected target path is forbidden: {path}")


def load_record(record: Mapping[str, Any] | None) -> np.ndarray | None:
    if record is None:
        return None
    path = Path(record["file"]).resolve(strict=True)
    forbid(path)
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"carrier must be a real file: {path}")
    key = str(record["key"])
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise BuildError(f"array {key!r} is absent from {path}")
            value = np.asarray(archive[key])
    else:
        raise BuildError(f"unsupported numeric carrier: {path}")
    if list(value.shape) != list(record["shape"]):
        raise BuildError(
            f"array shape changed for {path}::{key}: {value.shape} != {record['shape']}"
        )
    if str(value.dtype) != str(record["dtype"]):
        raise BuildError(
            f"array dtype changed for {path}::{key}: {value.dtype} != {record['dtype']}"
        )
    return value


def load_trajectory_ids(record: Mapping[str, Any]) -> np.ndarray:
    value = load_record(record)
    if value is None:
        raise BuildError("trajectory ID carrier is required")
    if value.dtype.kind not in "USbiuf":
        raise BuildError("trajectory IDs must be non-pickle strings or numeric values")
    return np.asarray(value).astype(str)


def choose_output_axis(array: np.ndarray) -> int:
    if array.ndim < 2:
        raise BuildError(f"residual carrier must be at least 2-D, got {array.shape}")
    if 1 <= array.shape[-1] <= 32:
        return array.ndim - 1
    raise BuildError(
        f"cannot identify a bounded residual output axis in shape {array.shape}"
    )


def _largest_matching_prefix(feature_shape: tuple[int, ...], leading: tuple[int, ...]) -> int:
    maximum = min(len(feature_shape), len(leading))
    for length in range(maximum, -1, -1):
        if feature_shape[:length] == leading[:length]:
            return length
    return 0


def align_feature_block(
    feature: np.ndarray,
    *,
    y_shape: tuple[int, ...],
    output_axis: int,
    label: str,
) -> np.ndarray:
    """Broadcast a lower-rate feature over residual geometry dimensions.

    ``y_shape`` uses its final axis as residual output dimension. A feature may
    share any leading prefix (for example trajectory/time) and use its remaining
    axes as feature dimensions. Missing residual-leading axes are broadcast.
    """

    if output_axis != len(y_shape) - 1:
        raise BuildError("only a final residual output axis is supported")
    leading = tuple(y_shape[:-1])
    value = np.asarray(feature)
    if value.ndim == 0:
        value = value.reshape(1)
    prefix = _largest_matching_prefix(tuple(value.shape), leading)
    if prefix == 0 and value.shape and value.shape[0] == int(np.prod(leading)):
        value = value.reshape(*leading, *value.shape[1:])
        prefix = len(leading)
    feature_shape = tuple(value.shape[prefix:])
    if not feature_shape:
        feature_shape = (1,)
        value = value.reshape(*value.shape, 1)
    if math.prod(feature_shape) > MAX_FEATURES_PER_RAW_BLOCK:
        # Preserve deterministic leading channels; never choose channels by fit.
        flat = value.reshape(*value.shape[:prefix], -1)
        flat = flat[..., :MAX_FEATURES_PER_RAW_BLOCK]
        value = flat
        feature_shape = (flat.shape[-1],)
    reshape = tuple(value.shape[:prefix]) + (1,) * (len(leading) - prefix) + feature_shape
    value = value.reshape(reshape)
    target_shape = leading + feature_shape
    try:
        broadcast = np.broadcast_to(value, target_shape)
    except ValueError as error:
        raise BuildError(
            f"cannot align {label} shape {feature.shape} to residual shape {y_shape}"
        ) from error
    return np.asarray(broadcast, dtype=float).reshape(int(np.prod(leading)), -1)


def trajectory_row_ids(
    ids: np.ndarray,
    *,
    y_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    leading = tuple(y_shape[:-1])
    count = int(np.prod(leading))
    flat = ids.reshape(-1).astype(str)
    if flat.size == count:
        return flat, "per-residual-row"
    if flat.size == leading[0]:
        repeats = int(np.prod(leading[1:])) if len(leading) > 1 else 1
        return np.repeat(flat, repeats), "per-leading-trajectory"
    if ids.shape == leading:
        return ids.reshape(-1).astype(str), "leading-grid"
    raise BuildError(
        f"trajectory ID shape {ids.shape} cannot identify residual rows {y_shape}"
    )


def deterministic_subsample(
    trajectory_ids: np.ndarray,
    *,
    max_rows_per_trajectory: int,
) -> np.ndarray:
    selected: list[int] = []
    for trajectory in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory)
        if len(indices) <= max_rows_per_trajectory:
            selected.extend(indices.tolist())
            continue
        positions = np.linspace(
            0, len(indices) - 1, max_rows_per_trajectory, dtype=int
        )
        selected.extend(indices[positions].tolist())
    return np.asarray(sorted(selected), dtype=int)


def local_difference_features(
    state_rows: np.ndarray,
    trajectory_ids: np.ndarray,
) -> np.ndarray:
    first = np.zeros_like(state_rows)
    second = np.zeros_like(state_rows)
    for trajectory in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory)
        values = state_rows[indices]
        first_values = np.diff(values, axis=0, prepend=values[[0]])
        second_values = np.diff(
            first_values, axis=0, prepend=first_values[[0]]
        )
        first[indices] = first_values
        second[indices] = second_values
    norm = np.linalg.norm(state_rows, axis=1, keepdims=True)
    speed = np.linalg.norm(first, axis=1, keepdims=True)
    acceleration = np.linalg.norm(second, axis=1, keepdims=True)
    return np.concatenate(
        (state_rows, first, second, norm, speed, acceleration), axis=1
    )


def object_features(
    shared: np.ndarray,
    object_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    objects = sorted(np.unique(object_ids.astype(str)).tolist())
    if objects != list(SOURCE_DLOS):
        raise BuildError(f"expected exactly DLO2/DLO3, got {objects}")
    one_hot = np.column_stack(
        [(object_ids.astype(str) == value).astype(float) for value in objects]
    )
    interaction_source = shared[:, : min(4, shared.shape[1])]
    interactions = np.concatenate(
        [one_hot[:, [index]] * interaction_source for index in range(len(objects))],
        axis=1,
    )
    return np.concatenate((one_hot, interactions), axis=1), objects


def infer_backend(path: Path) -> str:
    lower = path.as_posix().lower()
    if "pyelastica" in lower or "alternate" in lower:
        return "alternate"
    if "deform" in lower:
        return "deform"
    return "unknown-source-backend"


def finite_filter(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    masks = []
    for value in arrays.values():
        if value.dtype.kind in "biufc":
            masks.append(np.all(np.isfinite(value), axis=1))
    if not masks:
        raise BuildError("no numerical rows to validate")
    return np.logical_and.reduce(masks)


def build(plan: Mapping[str, Any], *, max_rows_per_trajectory: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if plan.get("ready_for_panel_build") is not True:
        raise BuildError("source plan did not authorize panel construction")
    boundary = plan.get("information_boundary", {})
    for key in (
        "dlo4_payload_read",
        "dlo5_payload_read",
        "protected_parent_target_result_read",
        "target_outcome_used_for_semantic_mapping",
        "ambiguous_carrier_auto_selected",
    ):
        if boundary.get(key) is not False:
            raise BuildError(f"source plan does not prove custody: {key}")
    if sorted(plan.get("selected_carriers", {})) != list(SOURCE_DLOS):
        raise BuildError("source plan must select exactly DLO2 and DLO3")

    y_parts = []
    shared_parts = []
    object_ids_parts = []
    trajectory_parts = []
    backend_parts = []
    action_parts = []
    contact_parts = []
    source_records = []
    trajectory_modes = {}

    for dlo in SOURCE_DLOS:
        option = plan["selected_carriers"][dlo]
        carrier = Path(option["carrier_file"]).resolve(strict=True)
        forbid(carrier)
        if sha256(carrier) != option["carrier_sha256"]:
            raise BuildError(f"selected carrier changed after planning: {carrier}")

        residual = load_record(option.get("residual"))
        observation = load_record(option.get("observation"))
        physical = load_record(option.get("physical"))
        state = load_record(option.get("state"))
        action = load_record(option.get("action"))
        contact = load_record(option.get("contact"))
        ids_raw = load_trajectory_ids(option["trajectory_id"])

        if physical is None:
            raise BuildError(f"{dlo} lacks a physical/state feature carrier")
        if option["mode"] == "explicit-residual":
            if residual is None:
                raise BuildError(f"{dlo} explicit-residual mode lacks residual array")
            y_raw = np.asarray(residual, dtype=float)
        elif option["mode"] == "observation-minus-physical":
            if observation is None:
                raise BuildError(f"{dlo} observation-minus-physical lacks observation")
            try:
                y_raw = np.asarray(observation, dtype=float) - np.asarray(physical, dtype=float)
            except ValueError as error:
                raise BuildError(f"{dlo} observation/physical arrays do not align") from error
        else:
            raise BuildError(f"unregistered residual mode: {option['mode']}")

        output_axis = choose_output_axis(y_raw)
        y_shape = tuple(int(value) for value in y_raw.shape)
        y_rows = y_raw.reshape(-1, y_shape[-1])
        trajectory_ids, trajectory_mode = trajectory_row_ids(ids_raw, y_shape=y_shape)
        trajectory_ids = np.asarray(
            [f"{dlo}:{value}" for value in trajectory_ids], dtype=str
        )
        trajectory_modes[dlo] = trajectory_mode

        state_source = np.asarray(state if state is not None else physical)
        physical_rows = align_feature_block(
            state_source,
            y_shape=y_shape,
            output_axis=output_axis,
            label=f"{dlo} physical/state",
        )
        shared_rows = local_difference_features(physical_rows, trajectory_ids)
        action_rows = (
            align_feature_block(
                np.asarray(action),
                y_shape=y_shape,
                output_axis=output_axis,
                label=f"{dlo} action",
            )
            if action is not None
            else np.empty((len(y_rows), 0), dtype=float)
        )
        contact_rows = (
            align_feature_block(
                np.asarray(contact),
                y_shape=y_shape,
                output_axis=output_axis,
                label=f"{dlo} contact",
            )
            if contact is not None
            else np.empty((len(y_rows), 0), dtype=float)
        )
        if not (
            len(y_rows)
            == len(trajectory_ids)
            == len(shared_rows)
            == len(action_rows)
            == len(contact_rows)
        ):
            raise BuildError(f"{dlo} aligned row counts differ")

        object_ids = np.full(len(y_rows), dlo, dtype=str)
        backend = infer_backend(carrier)
        backend_ids = np.full(len(y_rows), backend, dtype=str)
        arrays_for_finite = {"y": y_rows, "shared": shared_rows}
        if action_rows.shape[1]:
            arrays_for_finite["action"] = action_rows
        if contact_rows.shape[1]:
            arrays_for_finite["contact"] = contact_rows
        mask = finite_filter(arrays_for_finite)
        if np.sum(mask) < 64:
            raise BuildError(f"{dlo} has too few finite source rows")

        y_parts.append(y_rows[mask])
        shared_parts.append(shared_rows[mask])
        action_parts.append(action_rows[mask])
        contact_parts.append(contact_rows[mask])
        trajectory_parts.append(trajectory_ids[mask])
        object_ids_parts.append(object_ids[mask])
        backend_parts.append(backend_ids[mask])
        source_records.append(
            {
                "dlo": dlo,
                "carrier_file": carrier.as_posix(),
                "carrier_sha256": option["carrier_sha256"],
                "mode": option["mode"],
                "raw_residual_shape": list(y_shape),
                "finite_rows": int(np.sum(mask)),
                "discarded_nonfinite_rows": int(len(mask) - np.sum(mask)),
                "trajectory_identity_mode": trajectory_mode,
                "backend_identity": backend,
                "physical_feature_dimensions": int(shared_rows.shape[1]),
                "action_feature_dimensions": int(action_rows.shape[1]),
                "contact_feature_dimensions": int(contact_rows.shape[1]),
            }
        )

    y = np.concatenate(y_parts, axis=0)
    shared = np.concatenate(shared_parts, axis=0)
    action = np.concatenate(action_parts, axis=0)
    contact = np.concatenate(contact_parts, axis=0)
    trajectory_ids = np.concatenate(trajectory_parts)
    object_ids = np.concatenate(object_ids_parts)
    backend_ids = np.concatenate(backend_parts)

    object_block, object_order = object_features(shared, object_ids)
    selected = deterministic_subsample(
        trajectory_ids, max_rows_per_trajectory=max_rows_per_trajectory
    )
    y = y[selected]
    shared = shared[selected]
    object_block = object_block[selected]
    action = action[selected]
    contact = contact[selected]
    trajectory_ids = trajectory_ids[selected]
    object_ids = object_ids[selected]
    backend_ids = backend_ids[selected]

    blocks: dict[str, np.ndarray] = {
        "shared_physics": shared,
        "object": object_block,
    }
    if action.shape[1]:
        blocks["actuation"] = action
    if contact.shape[1]:
        blocks["contact"] = contact
    unique_backends = sorted(np.unique(backend_ids).tolist())
    if len(unique_backends) > 1:
        blocks["backend"] = np.column_stack(
            [(backend_ids == value).astype(float) for value in unique_backends]
        )

    arrays: dict[str, np.ndarray] = {
        "y": np.asarray(y, dtype=float),
        "trajectory_id": trajectory_ids.astype(str),
        "object_id": object_ids.astype(str),
        "backend_id": backend_ids.astype(str),
    }
    arrays.update({f"block__{name}": np.asarray(value, dtype=float) for name, value in blocks.items()})

    trajectory_counts = {
        value: int(np.sum(trajectory_ids == value))
        for value in sorted(np.unique(trajectory_ids).tolist())
    }
    manifest: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform-dlo23-hierarchical-residual-panel",
        "schema_version": 1,
        "source_plan_id": plan["plan_id"],
        "source_census_id": plan["source_census_id"],
        "source_records": source_records,
        "residual_definition": {
            dlo: plan["selected_carriers"][dlo]["residual_definition"]
            for dlo in SOURCE_DLOS
        },
        "feature_semantics": {
            "shared_physics": "source physical/state carrier, local first and second ordered differences, and their norms; no target-derived feature selection",
            "object": "DLO2/DLO3 one-hot identity plus interactions with the first four frozen shared-state channels",
            "actuation": "explicit action carrier aligned by shared leading trajectory/time dimensions" if "actuation" in blocks else "unavailable",
            "contact": "explicit contact/force carrier aligned by shared leading trajectory/time dimensions" if "contact" in blocks else "unavailable",
            "backend": "one-hot source backend identity" if "backend" in blocks else "not identifiable from one source backend and therefore omitted",
            "sensor": "omitted: no explicit non-physical sensor-calibration carrier was selected; sensor discrepancy cannot be invented from the residual target"
        },
        "source_only_standardization": True,
        "max_rows_per_trajectory": max_rows_per_trajectory,
        "sample_count": int(len(y)),
        "output_dimensions": int(y.shape[1]),
        "trajectory_count": int(len(trajectory_counts)),
        "trajectory_counts": trajectory_counts,
        "object_order": object_order,
        "backend_order": unique_backends,
        "block_dimensions": {name: int(value.shape[1]) for name, value in blocks.items()},
        "transfer_eligibility": {
            name: name == "shared_physics" for name in blocks
        },
        "minimum_bootstrap_diagnosis_frequency": 0.70,
        "source_gate": {
            "shared_vs_physical_min_relative_improvement": 0.01,
            "minimum_source_bootstrap_shared_diagnosis_frequency": 0.70,
            "minimum_complete_trajectory_win_fraction": 0.60,
            "maximum_worst_trajectory_ratio_vs_physical": 1.10
        },
        "information_boundary": {
            "dlo2_payload_read": True,
            "dlo3_payload_read": True,
            "dlo4_path_read": False,
            "dlo4_payload_read": False,
            "dlo5_path_read": False,
            "dlo5_payload_read": False,
            "protected_parent_target_result_read": False,
            "target_outcome_used_for_feature_choice": False,
            "sensor_features_derived_from_residual_target": False,
            "frames_or_nodes_declared_independent_units": False
        }
    }
    array_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        for key, value in arrays.items()
    }
    manifest["array_content_sha256"] = array_hashes
    manifest["panel_id"] = canonical_hash(manifest)
    manifest["panel_sha256"] = manifest["panel_id"]
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--max-rows-per-trajectory", type=int, default=2000)
    arguments = parser.parse_args()
    if arguments.max_rows_per_trajectory < 64:
        raise SystemExit("--max-rows-per-trajectory must be at least 64")
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    arrays, manifest = build(
        plan, max_rows_per_trajectory=arguments.max_rows_per_trajectory
    )
    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = arguments.output_prefix.with_suffix(".npz")
    json_path = arguments.output_prefix.with_suffix(".json")
    np.savez_compressed(npz_path, **arrays)
    manifest["serialized_npz_sha256"] = sha256(npz_path)
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "panel_id": manifest["panel_id"],
                "npz_sha256": manifest["serialized_npz_sha256"],
                "sample_count": manifest["sample_count"],
                "trajectory_count": manifest["trajectory_count"],
                "object_order": manifest["object_order"],
                "backend_order": manifest["backend_order"],
                "block_dimensions": manifest["block_dimensions"],
                "source_records": manifest["source_records"],
                "information_boundary": manifest["information_boundary"]
            },
            indent=2,
            sort_keys=True,
            allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
