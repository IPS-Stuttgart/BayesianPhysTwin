"""Source-only table-collision query-directed probe feasibility study.

This one-shot development study uses only the Tracking Cloth table-collision
records. Half-lay trajectories are source/probe outcomes. Full-lay records are
read through a strict input view containing a short causal prefix and the future
trajectories of the two detected grasped corners; no post-prefix full-lay free
marker coordinate is read. The output decides whether a separately sealed target
evaluation is scientifically justified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import traceback
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .active_probe import simulate_policy, weights_from_records

MATERIALS = ("cotton", "denim", "polyester", "wool")
FRICTIONS = ("low_friction", "high_friction")
ACTIONS = tuple(f"half_lay_{friction}" for friction in FRICTIONS)
QUERIES = tuple(f"full_lay_{friction}" for friction in FRICTIONS)
POLICIES = ("fixed_order", "parameter_information", "task_directed")
TABLE_NAME = re.compile(
    r"^(cotton|denim|polyester|wool)_a2_(half_lay|full_lay)_"
    r"(low_friction|high_friction)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TableCase:
    path: Path
    material: str
    lay: str
    friction: str

    @property
    def action(self) -> str:
        return f"{self.lay}_{self.friction}"


@dataclass(frozen=True)
class TableInputs:
    times: np.ndarray
    prefix: np.ndarray
    boundary: np.ndarray
    order: np.ndarray
    corners: np.ndarray
    cutoff: int
    scale: float
    table_z: float


@dataclass(frozen=True)
class Parameter:
    stiffness: float
    damping: float
    mu_low: float
    mu_high: float

    def friction(self, regime: str) -> float:
        if regime == "low_friction":
            return self.mu_low
        if regime == "high_friction":
            return self.mu_high
        raise ValueError(f"unknown friction regime: {regime}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parameter_bank(protocol: Mapping[str, Any]) -> tuple[Parameter, ...]:
    bank = tuple(
        Parameter(*map(float, values))
        for values in itertools.product(
            protocol["stiffness_per_mass"],
            protocol["damping_per_mass"],
            protocol["mu_low"],
            protocol["mu_high"],
        )
    )
    if len(bank) != 81:
        raise ValueError("the frozen source study requires exactly 81 models")
    return bank


def audit_archive(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    csvs = [path for path in files if path.suffix.lower() == ".csv"]
    if len(csvs) != int(protocol["csv_count"]):
        raise ValueError(f"expected {protocol['csv_count']} CSVs, found {len(csvs)}")
    archives = [path for path in files if path.suffix.lower() == ".zip"]
    matching = [path for path in archives if md5(path) == protocol["archive_md5"]]
    if len(matching) != 1:
        raise ValueError("expected one retained archive matching the published MD5")
    archive = matching[0]
    hashes = {path.name.lower(): sha256(path) for path in csvs}
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("dataset archive integrity check failed")
        entries = [
            entry
            for entry in zipped.infolist()
            if entry.filename.lower().endswith(".csv")
        ]
        if len(entries) != len(csvs):
            raise ValueError("archive and extracted CSV counts disagree")
        seen: set[str] = set()
        for entry in entries:
            name = Path(entry.filename).name.lower()
            if name in seen or name not in hashes:
                raise ValueError("ambiguous archive CSV identity")
            seen.add(name)
            if hashlib.sha256(zipped.read(entry)).hexdigest() != hashes[name]:
                raise ValueError(f"extracted bytes differ from archive for {name}")
    licenses = [path for path in files if path.name.lower() == "license.txt"]
    if len(licenses) != 1:
        raise ValueError("expected exactly one included License.txt")
    license_text = licenses[0].read_text(encoding="utf-8-sig")
    if not any(
        token in license_text.lower().replace(" ", "")
        for token in ("by-nc-sa", "noncommercial", "non-commercial")
    ):
        raise ValueError("included license differs from the frozen NC policy")
    table_cases: list[TableCase] = []
    for path in csvs:
        match = TABLE_NAME.fullmatch(path.name)
        if match:
            material, lay, friction = match.groups()
            table_cases.append(
                TableCase(path, material.lower(), lay.lower(), friction.lower())
            )
    expected = set(itertools.product(MATERIALS, ("half_lay", "full_lay"), FRICTIONS))
    actual = {(case.material, case.lay, case.friction) for case in table_cases}
    if actual != expected or len(table_cases) != 16:
        raise ValueError("complete 16-record table-collision factorial is required")
    inventory: dict[str, Any] = {
        "dataset_record": protocol["dataset_record"],
        "archive_name": archive.name,
        "archive_md5": protocol["archive_md5"],
        "archive_sha256": sha256(archive),
        "csv_count": len(csvs),
        "table_record_count": len(table_cases),
        "half_lay_source_count": 8,
        "full_lay_target_count": 8,
        "table_csv_sha256": {
            case.path.name: hashes[case.path.name.lower()] for case in table_cases
        },
        "license_sha256": sha256(licenses[0]),
        "license_policy": protocol["license_policy"],
        "included_license_text": license_text,
        "full_lay_post_prefix_free_marker_outcomes_read": False,
    }
    inventory["inventory_id"] = object_digest(inventory)
    return {
        "cases": sorted(table_cases, key=lambda case: case.path.name),
        "inventory": inventory,
    }


def numeric_rows(path: Path, markers: int = 20) -> Iterable[tuple[float, list[str]]]:
    width = 2 + 3 * markers
    started = False
    previous_time = -math.inf
    previous_frame = -math.inf
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                frame = float(row[0])
                timestamp = float(row[1])
            except (ValueError, IndexError):
                if started:
                    raise ValueError(
                        f"nonnumeric row after data start in {path.name}"
                    ) from None
                continue
            started = True
            if (
                not np.isfinite((frame, timestamp)).all()
                or frame != int(frame)
                or frame <= previous_frame
                or timestamp <= previous_time
            ):
                raise ValueError(f"invalid frame/time order in {path.name}")
            if len(row) < width or any(cell.strip() for cell in row[width:]):
                raise ValueError(f"unexpected table CSV width in {path.name}")
            previous_frame = frame
            previous_time = timestamp
            yield timestamp, row[2:width]
    if not started:
        raise ValueError(f"no numeric rows in {path.name}")


def positions(cells: Sequence[str], indices: Sequence[int]) -> np.ndarray:
    result = []
    for marker in indices:
        values = []
        for dimension in range(3):
            text = cells[3 * int(marker) + dimension].strip()
            values.append(float(text) if text else np.nan)
        triple = np.asarray(values, dtype=np.float64)
        if np.any(np.isinf(triple)):
            raise ValueError("infinite marker coordinate")
        result.append(triple if np.all(np.isfinite(triple)) else np.full(3, np.nan))
    return np.asarray(result)


def infer_scale(initial: np.ndarray) -> float:
    diameter = float(
        np.max(np.linalg.norm(initial[:, None, :] - initial[None, :, :], axis=2))
    )
    nominal = math.hypot(0.42, 0.594)
    allowed = [
        scale for scale in (1.0, 0.01, 0.001) if 0.4 < diameter * scale / nominal < 1.6
    ]
    if len(allowed) != 1:
        raise ValueError(f"ambiguous coordinate scale from diameter {diameter}")
    return allowed[0]


def _layout_candidate(initial: np.ndarray, row_axis: int, row_sign: int, col_sign: int):
    centered = initial - initial.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vh[:2].T
    row_values = row_sign * plane[:, row_axis]
    col_values = col_sign * plane[:, 1 - row_axis]
    row_order = np.argsort(row_values, kind="stable")
    groups = row_order.reshape(5, 4)
    ordered_groups = [
        group[np.argsort(col_values[group], kind="stable")] for group in groups
    ]
    order = np.concatenate(ordered_groups)
    grid = initial[order].reshape(5, 4, 3)
    projected = plane[order].reshape(5, 4, 2)
    row_centers = np.mean(row_sign * projected[:, :, row_axis], axis=1)
    col_centers = np.mean(col_sign * projected[:, :, 1 - row_axis], axis=0)
    row_steps = np.diff(row_centers)
    col_steps = np.diff(col_centers)
    if np.any(row_steps <= 0.0) or np.any(col_steps <= 0.0):
        return math.inf, order, singular
    row_cv = float(np.std(row_steps) / max(np.mean(row_steps), 1e-12))
    col_cv = float(np.std(col_steps) / max(np.mean(col_steps), 1e-12))
    row_lengths = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    col_lengths = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    adjacency_cv = float(
        np.std(np.concatenate((row_lengths.ravel(), col_lengths.ravel())))
        / max(
            np.mean(np.concatenate((row_lengths.ravel(), col_lengths.ravel()))), 1e-12
        )
    )
    long_span = float(np.linalg.norm(grid[-1].mean(axis=0) - grid[0].mean(axis=0)))
    short_span = float(
        np.linalg.norm(grid[:, -1].mean(axis=0) - grid[:, 0].mean(axis=0))
    )
    ratio_penalty = abs(
        math.log(max(long_span, 1e-12) / max(short_span, 1e-12))
        - math.log(0.594 / 0.42)
    )
    plane_penalty = float(singular[2] / max(singular[1], 1e-12))
    score = row_cv + col_cv + adjacency_cv + ratio_penalty + 2.0 * plane_penalty
    return score, order, singular


def planar_layout(initial: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if initial.shape != (20, 3) or not np.all(np.isfinite(initial)):
        raise ValueError("layout requires one complete 20-marker frame")
    candidates = []
    for row_axis in (0, 1):
        for row_sign in (-1, 1):
            for col_sign in (-1, 1):
                candidates.append(
                    _layout_candidate(initial, row_axis, row_sign, col_sign)
                )
    score, order, singular = min(candidates, key=lambda item: item[0])
    if not np.isfinite(score) or score > 1.8:
        raise ValueError(f"unsupported initial table-cloth layout score {score:.6f}")
    return order, {
        "layout_score": float(score),
        "planarity_ratio": float(singular[2] / max(singular[1], 1e-12)),
    }


def choose_grasped_edge(
    prefix_times: np.ndarray, ordered_prefix: np.ndarray, table_z: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if ordered_prefix.ndim != 3 or ordered_prefix.shape[1:] != (20, 3):
        raise ValueError("ordered prefix has the wrong shape")
    edges = {
        "row0": (0, 3),
        "row4": (16, 19),
        "col0": (0, 16),
        "col3": (3, 19),
    }
    dt = np.diff(prefix_times)
    if len(dt) < 3 or np.any(dt <= 0.0):
        raise ValueError("insufficient prefix for grasped-edge detection")
    initial_span = float(np.ptp(ordered_prefix[0, :, 2]))
    scale = max(initial_span, 0.05)
    diagnostics: dict[str, Any] = {}
    scored = []
    for name, pair in edges.items():
        points = ordered_prefix[:, pair, :]
        separation = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
        relative_velocity = np.diff(points[:, 1] - points[:, 0], axis=0) / dt[:, None]
        centroid = points.mean(axis=1)
        centroid_velocity = np.diff(centroid, axis=0) / dt[:, None]
        separation_cv = float(np.std(separation) / max(np.mean(separation), 1e-12))
        relative_speed = float(np.sqrt(np.mean(np.sum(relative_velocity**2, axis=1))))
        centroid_speed = float(np.sqrt(np.mean(np.sum(centroid_velocity**2, axis=1))))
        height = float(np.mean(points[0, :, 2]) - table_z)
        rigidity = separation_cv + 0.05 * relative_speed / max(centroid_speed, 0.02)
        height_bonus = height / scale
        score = rigidity - 0.35 * height_bonus - 0.20 * (centroid_speed / 0.10)
        diagnostics[name] = {
            "ordered_corner_indices": list(pair),
            "initial_height_above_table_m": height,
            "separation_cv": separation_cv,
            "relative_speed_m_s": relative_speed,
            "centroid_speed_m_s": centroid_speed,
            "selection_score": score,
        }
        scored.append((score, name, pair))
    score, name, pair = min(scored)
    second = sorted(scored)[1][0]
    if second - score < 1e-5:
        raise ValueError(
            "grasped edge is not uniquely identified from the causal prefix"
        )
    return np.asarray(pair, dtype=int), {
        "selected_edge": name,
        "score_margin": float(second - score),
        "candidates": diagnostics,
    }


def _complete_prefix(path: Path, seconds: float):
    times: list[float] = []
    values: list[np.ndarray] = []
    start = None
    previous = None
    for timestamp, cells in numeric_rows(path):
        if start is None:
            start = timestamp
        if timestamp - start > seconds + 1e-9:
            break
        current = positions(cells, range(20))
        if previous is not None:
            current = np.where(np.isfinite(current), current, previous)
        if np.all(np.isfinite(current)):
            previous = current.copy()
        times.append(timestamp)
        values.append(current)
    raw = np.asarray(values)
    complete = np.flatnonzero(np.isfinite(raw).all(axis=(1, 2)))
    if len(complete) == 0:
        raise ValueError(f"no complete causal prefix frame in {path.name}")
    first = int(complete[0])
    return np.asarray(times[first:]), raw[first:]


def input_view(
    case: TableCase, protocol: Mapping[str, Any]
) -> tuple[TableInputs, dict[str, Any]]:
    prefix_seconds = float(protocol["prefix_seconds"])
    prefix_times_raw, prefix_raw = _complete_prefix(case.path, prefix_seconds)
    scale = infer_scale(prefix_raw[0])
    prefix_raw = prefix_raw * scale
    order, layout_diagnostic = planar_layout(prefix_raw[0])
    ordered_prefix_raw = prefix_raw[:, order]
    table_z = float(protocol["table_z_m"])
    corners, edge_diagnostic = choose_grasped_edge(
        prefix_times_raw, ordered_prefix_raw, table_z
    )
    raw_corners = order[corners]

    start_time = float(prefix_times_raw[0])
    end_time = start_time + prefix_seconds + float(protocol["forecast_seconds"])
    stride = int(protocol["sample_stride"])
    times: list[float] = []
    prefix: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    last_all = None
    last_boundary = None
    row_index = 0
    for timestamp, cells in numeric_rows(case.path):
        if timestamp < start_time - 1e-9:
            continue
        if timestamp > end_time + 1e-8:
            break
        selected = row_index % stride == 0
        row_index += 1
        if not selected:
            continue
        corner_values = positions(cells, raw_corners) * scale
        if last_boundary is not None:
            corner_values = np.where(
                np.isfinite(corner_values), corner_values, last_boundary
            )
        if not np.all(np.isfinite(corner_values)):
            raise ValueError(f"missing initial driven corner in {case.path.name}")
        last_boundary = corner_values.copy()
        times.append(timestamp)
        boundary.append(corner_values)
        if timestamp <= start_time + prefix_seconds + 1e-8:
            all_values = positions(cells, order) * scale
            if last_all is not None:
                all_values = np.where(np.isfinite(all_values), all_values, last_all)
            if not np.all(np.isfinite(all_values)):
                raise ValueError(f"nonfinite causal prefix in {case.path.name}")
            last_all = all_values.copy()
            prefix.append(all_values)
    times_array = np.asarray(times)
    prefix_array = np.asarray(prefix)
    boundary_array = np.asarray(boundary)
    if len(prefix_array) < 5 or len(times_array) <= len(prefix_array) + 10:
        raise ValueError(f"insufficient prefix/forecast samples in {case.path.name}")
    dt = np.diff(times_array)
    expected_dt = stride / float(protocol["frame_rate_hz"])
    if not np.allclose(dt, expected_dt, rtol=0.05, atol=1e-4):
        raise ValueError(f"unexpected sampling cadence in {case.path.name}")
    if times_array[-1] < end_time - 2.0 * expected_dt:
        raise ValueError(f"recording does not cover frozen horizon: {case.path.name}")
    inputs = TableInputs(
        times=times_array,
        prefix=prefix_array,
        boundary=boundary_array,
        order=order,
        corners=corners,
        cutoff=len(prefix_array) - 1,
        scale=scale,
        table_z=table_z,
    )
    diagnostic = {
        **layout_diagnostic,
        **edge_diagnostic,
        "coordinate_scale_to_m": scale,
        "prefix_samples": len(prefix_array),
        "total_samples": len(times_array),
        "initial_min_z_m": float(np.min(prefix_array[0, :, 2])),
        "initial_max_z_m": float(np.max(prefix_array[0, :, 2])),
        "raw_corner_indices": raw_corners.tolist(),
    }
    return inputs, diagnostic


def scoring_view(case: TableCase, inputs: TableInputs) -> np.ndarray:
    rows: list[np.ndarray] = []
    index = 0
    for timestamp, cells in numeric_rows(case.path):
        if index == len(inputs.times):
            break
        if abs(timestamp - inputs.times[index]) <= 1e-7:
            rows.append(positions(cells, inputs.order) * inputs.scale)
            index += 1
    if index != len(inputs.times):
        raise ValueError(f"scoring timestamps do not reproduce {case.path.name}")
    return np.asarray(rows)


def graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    links: list[tuple[int, int]] = []
    weights: list[float] = []
    for row in range(5):
        for col in range(4):
            for dr, dc, weight in (
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 0.35),
                (1, -1, 0.35),
                (0, 2, 0.08),
                (2, 0, 0.08),
            ):
                other_row = row + dr
                other_col = col + dc
                if 0 <= other_row < 5 and 0 <= other_col < 4:
                    links.append((row * 4 + col, other_row * 4 + other_col))
                    weights.append(weight)
    edge_array = np.asarray(links, dtype=int)
    incidence = np.zeros((20, len(links)), dtype=np.float64)
    incidence[edge_array[:, 0], np.arange(len(links))] = 1.0
    incidence[edge_array[:, 1], np.arange(len(links))] = -1.0
    return edge_array, np.asarray(weights, dtype=np.float64), incidence


def estimate_velocity(times: np.ndarray, positions_value: np.ndarray) -> np.ndarray:
    count = min(5, len(times))
    selected_times = times[-count:]
    selected_positions = positions_value[-count:]
    centered = selected_times - selected_times.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("degenerate prefix timestamps")
    return np.einsum("t,tnd->nd", centered, selected_positions) / denominator


def rollout_bank(
    inputs: TableInputs,
    parameters: Sequence[Parameter],
    friction_regime: str,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    links, relative_k, incidence = graph()
    left, right = links.T
    model_count = len(parameters)
    stiffness = np.asarray([value.stiffness for value in parameters])[:, None, None]
    damping = np.asarray([value.damping for value in parameters])[:, None, None]
    friction = np.asarray([value.friction(friction_regime) for value in parameters])[
        :, None
    ]
    x = np.broadcast_to(inputs.prefix[-1], (model_count, 20, 3)).copy()
    initial_velocity = estimate_velocity(
        inputs.times[: inputs.cutoff + 1], inputs.prefix
    )
    v = np.broadcast_to(initial_velocity, x.shape).copy()
    rest = np.linalg.norm(inputs.prefix[0, right] - inputs.prefix[0, left], axis=1)
    if np.min(rest) <= 1e-6:
        raise ValueError("degenerate spring rest length")
    result = np.empty((model_count, len(inputs.times), 20, 3), dtype=np.float64)
    result[:, : inputs.cutoff + 1] = inputs.prefix[None, ...]
    substeps = int(protocol["integration_substeps"])
    gravity = float(protocol["gravity_m_s2"])
    origin = inputs.prefix[0].mean(axis=0)
    for time_index in range(inputs.cutoff + 1, len(inputs.times)):
        full_dt = float(inputs.times[time_index] - inputs.times[time_index - 1])
        dt = full_dt / substeps
        boundary_velocity = (
            inputs.boundary[time_index] - inputs.boundary[time_index - 1]
        ) / full_dt
        for substep in range(1, substeps + 1):
            delta = x[:, right] - x[:, left]
            lengths = np.linalg.norm(delta, axis=2)
            extension = lengths - rest[None, :]
            edge_force = (
                stiffness
                * relative_k[None, :, None]
                * extension[:, :, None]
                / np.maximum(lengths[:, :, None], 1e-9)
                * delta
            )
            acceleration = -damping * v
            acceleration[:, :, 2] -= gravity
            acceleration += np.einsum("ne,med->mnd", incidence, edge_force)
            v += dt * acceleration
            x += dt * v

            contact = x[:, :, 2] < inputs.table_z
            if np.any(contact):
                x[:, :, 2] = np.maximum(x[:, :, 2], inputs.table_z)
                downward = contact & (v[:, :, 2] < 0.0)
                v[:, :, 2][downward] = 0.0
                tangential = v[:, :, :2]
                speed = np.linalg.norm(tangential, axis=2)
                reduction = friction * gravity * dt
                shrink = np.maximum(0.0, 1.0 - reduction / np.maximum(speed, 1e-12))
                shrink = np.where(contact, shrink, 1.0)
                v[:, :, :2] *= shrink[:, :, None]

            fraction = substep / substeps
            driven = (1.0 - fraction) * inputs.boundary[
                time_index - 1
            ] + fraction * inputs.boundary[time_index]
            x[:, inputs.corners] = driven[None, :, :]
            v[:, inputs.corners] = boundary_velocity[None, :, :]
        if (
            not np.all(np.isfinite(x))
            or np.max(np.linalg.norm(x - origin[None, None, :], axis=2)) > 5.0
        ):
            raise ValueError("numerically invalid table-contact rollout")
        result[:, time_index] = x
    return result


def active_mask(inputs: TableInputs) -> np.ndarray:
    mask = np.ones((len(inputs.times), 20), dtype=bool)
    mask[: inputs.cutoff + 1] = False
    mask[:, inputs.corners] = False
    if not np.any(mask):
        raise ValueError("empty scored table-collision mask")
    return mask


def loss_vector(bank: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(truth).all(axis=2)
    if not np.any(valid):
        raise ValueError("no evaluable source table samples")
    selected = bank[:, valid, :]
    observed = truth[valid]
    if not np.all(np.isfinite(selected)):
        raise ValueError("nonfinite model prediction on evaluable samples")
    error = selected - observed[None, ...]
    return np.mean(np.sum(error * error, axis=2), axis=1)


def pairwise_mse(bank: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(bank[:, mask, :], dtype=np.float64).reshape(len(bank), -1)
    if selected.shape[1] == 0 or not np.all(np.isfinite(selected)):
        raise ValueError("empty or nonfinite model disagreement trajectory")
    squared = np.sum(selected * selected, axis=1)
    distance = (squared[:, None] + squared[None, :] - 2.0 * (selected @ selected.T)) / (
        selected.shape[1] / 3.0
    )
    distance = np.maximum(0.5 * (distance + distance.T), 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def weighted_mean(bank: np.ndarray, weights: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.einsum("m,mtnd->tnd", probabilities, bank)


def rmse_mm(mean: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(truth).all(axis=2)
    error = mean[valid] - truth[valid]
    return 1000.0 * float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def persistence_mean(inputs: TableInputs) -> np.ndarray:
    mean = np.broadcast_to(inputs.prefix[-1], (len(inputs.times), 20, 3)).copy()
    mean[:, inputs.corners] = inputs.boundary
    return mean


def source_temperature(losses: np.ndarray, floor_m: float) -> float:
    if losses.ndim != 2 or np.any(losses < 0.0):
        raise ValueError("source losses must be a nonnegative matrix")
    return max(float(np.median(np.min(losses, axis=1))), floor_m * floor_m)


def average_distances(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("at least one disagreement matrix is required")
    result = np.mean(np.stack(values), axis=0)
    result = np.maximum(0.5 * (result + result.T), 0.0)
    np.fill_diagonal(result, 0.0)
    return result


def run_source(
    root: Path, output: Path, protocol: Mapping[str, Any], workers: int
) -> None:
    del (
        workers
    )  # vectorized NumPy bank; one process avoids nested BLAS oversubscription.
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output and dataset roots must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_archive(root, protocol)
    cases: list[TableCase] = audit["cases"]
    inventory = audit["inventory"]
    write_json(output / "protocol.json", dict(protocol))
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(inventory["included_license_text"])

    parameters = parameter_bank(protocol)
    case_records: dict[str, dict[str, Any]] = {}
    geometry: dict[str, Any] = {}
    for case in cases:
        inputs, diagnostic = input_view(case, protocol)
        bank = rollout_bank(inputs, parameters, case.friction, protocol)
        mask = active_mask(inputs)
        record: dict[str, Any] = {
            "case": case,
            "inputs": inputs,
            "bank": bank,
            "mask": mask,
            "distance": pairwise_mse(bank, mask),
        }
        if case.lay == "half_lay":
            truth = scoring_view(case, inputs)
            record["truth"] = truth
            record["loss"] = loss_vector(bank, truth, mask)
            record["persistence_rmse_mm"] = rmse_mm(
                persistence_mean(inputs), truth, mask
            )
        else:
            # Deliberately do not call scoring_view for full-lay target records.
            record["truth"] = None
            record["loss"] = None
        case_records[case.path.name] = record
        geometry[case.path.name] = diagnostic

    folds: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for held_material in MATERIALS:
        training = [
            record
            for record in case_records.values()
            if record["case"].lay == "half_lay"
            and record["case"].material != held_material
        ]
        if len(training) != 6:
            raise ValueError("incomplete leave-one-material-out half-lay source fold")
        loss_matrix = np.stack([record["loss"] for record in training])
        temperature = source_temperature(
            loss_matrix, float(protocol["measurement_floor_m"])
        )
        prior = weights_from_records(loss_matrix, temperature)
        probe_distances = {
            action: average_distances(
                [
                    record["distance"]
                    for record in training
                    if record["case"].action == action
                ]
            )
            for action in ACTIONS
        }
        target_distances = {
            query: average_distances(
                [
                    record["distance"]
                    for record in case_records.values()
                    if record["case"].action == query
                    and record["case"].material != held_material
                ]
            )
            for query in QUERIES
        }
        held_sources = {
            record["case"].action: record
            for record in case_records.values()
            if record["case"].lay == "half_lay"
            and record["case"].material == held_material
        }
        if set(held_sources) != set(ACTIONS):
            raise ValueError("held material lacks the two registered half-lay probes")
        observed_losses = {action: held_sources[action]["loss"] for action in ACTIONS}
        fold_decisions: dict[str, Any] = {}
        for query in QUERIES:
            query_states: dict[str, Any] = {}
            for policy in POLICIES:
                states = simulate_policy(
                    policy=policy,
                    initial_weights=prior,
                    probe_distances=probe_distances,
                    target_distance=target_distances[query],
                    observed_losses=observed_losses,
                    temperature=temperature,
                    fixed_order=ACTIONS,
                    budgets=(0, 1, 2),
                )
                state = states[1]
                selected = state.selected_actions[0]
                query_states[policy] = {
                    "selected_action": selected,
                    "weights": state.weights.tolist(),
                    "utilities": state.steps[0]["utilities"],
                    "entropy_before": state.steps[0]["entropy_before"],
                    "entropy_after": state.steps[0]["entropy_after"],
                    "target_model_spread_before": state.steps[0][
                        "target_model_spread_before"
                    ],
                    "target_model_spread_after": state.steps[0][
                        "target_model_spread_after"
                    ],
                }
                unselected = next(action for action in ACTIONS if action != selected)
                evaluation = held_sources[unselected]
                truth = evaluation["truth"]
                mask = evaluation["mask"]
                prior_rmse = rmse_mm(
                    weighted_mean(evaluation["bank"], prior), truth, mask
                )
                updated_rmse = rmse_mm(
                    weighted_mean(evaluation["bank"], state.weights), truth, mask
                )
                persistence_rmse = float(evaluation["persistence_rmse_mm"])
                transfer_rows.append(
                    {
                        "held_material": held_material,
                        "query": query,
                        "policy": policy,
                        "selected_probe": selected,
                        "unselected_source_evaluation": unselected,
                        "prior_rmse_mm": prior_rmse,
                        "updated_rmse_mm": updated_rmse,
                        "persistence_rmse_mm": persistence_rmse,
                        "updated_minus_prior_mm": updated_rmse - prior_rmse,
                        "updated_minus_persistence_mm": updated_rmse - persistence_rmse,
                    }
                )
            fold_decisions[query] = query_states
            decisions.append(
                {
                    "held_material": held_material,
                    "query": query,
                    "fixed_order": query_states["fixed_order"]["selected_action"],
                    "parameter_information": query_states["parameter_information"][
                        "selected_action"
                    ],
                    "task_directed": query_states["task_directed"]["selected_action"],
                    "task_vs_parameter_disagree": (
                        query_states["task_directed"]["selected_action"]
                        != query_states["parameter_information"]["selected_action"]
                    ),
                }
            )
        folds[held_material] = {
            "training_recordings": [record["case"].path.name for record in training],
            "temperature_m2": temperature,
            "prior_weights": prior.tolist(),
            "probe_distance_matrices_m2": {
                key: value.tolist() for key, value in probe_distances.items()
            },
            "target_distance_matrices_m2": {
                key: value.tolist() for key, value in target_distances.items()
            },
            "decisions": fold_decisions,
            "half_lay_source_outcomes_used": True,
            "full_lay_prefix_and_future_driven_corners_used": True,
            "full_lay_post_prefix_free_marker_outcomes_used": False,
        }

    task_rows = [row for row in transfer_rows if row["policy"] == "task_directed"]
    disagreement_count = sum(row["task_vs_parameter_disagree"] for row in decisions)
    query_switch_materials = sum(
        next(
            row["task_directed"]
            for row in decisions
            if row["held_material"] == material
            and row["query"] == "full_lay_low_friction"
        )
        != next(
            row["task_directed"]
            for row in decisions
            if row["held_material"] == material
            and row["query"] == "full_lay_high_friction"
        )
        for material in MATERIALS
    )
    task_mean_prior = float(np.mean([row["prior_rmse_mm"] for row in task_rows]))
    task_mean_updated = float(np.mean([row["updated_rmse_mm"] for row in task_rows]))
    task_mean_persistence = float(
        np.mean([row["persistence_rmse_mm"] for row in task_rows])
    )
    transfer_wins_prior = sum(
        row["updated_rmse_mm"] < row["prior_rmse_mm"] for row in task_rows
    )
    transfer_wins_persistence = sum(
        row["updated_rmse_mm"] < row["persistence_rmse_mm"] for row in task_rows
    )
    gate = {
        "minimum_task_vs_parameter_disagreements": int(
            protocol["minimum_task_vs_parameter_disagreements"]
        ),
        "task_vs_parameter_disagreements": int(disagreement_count),
        "minimum_query_switch_materials": int(
            protocol["minimum_query_switch_materials"]
        ),
        "query_switch_materials": int(query_switch_materials),
        "task_transfer_mean_prior_rmse_mm": task_mean_prior,
        "task_transfer_mean_updated_rmse_mm": task_mean_updated,
        "task_transfer_mean_persistence_rmse_mm": task_mean_persistence,
        "task_transfer_wins_vs_prior": int(transfer_wins_prior),
        "task_transfer_wins_vs_persistence": int(transfer_wins_persistence),
        "minimum_transfer_wins_vs_prior": int(
            protocol["minimum_transfer_wins_vs_prior"]
        ),
    }
    gate["selection_divergence_passes"] = bool(
        disagreement_count >= gate["minimum_task_vs_parameter_disagreements"]
        and query_switch_materials >= gate["minimum_query_switch_materials"]
    )
    gate["source_transfer_passes"] = bool(
        transfer_wins_prior >= gate["minimum_transfer_wins_vs_prior"]
        and task_mean_updated <= task_mean_prior
    )
    gate["authorize_separate_target_protocol"] = bool(
        gate["selection_divergence_passes"] and gate["source_transfer_passes"]
    )

    source_fit = {
        "schema": "tracking-cloth-table-query-source-v1",
        "created_at": now(),
        "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"],
        "parameter_count": len(parameters),
        "parameters": [parameter.__dict__ for parameter in parameters],
        "folds": folds,
        "geometry_diagnostics": geometry,
        "full_lay_target_input_contract": (
            "short causal all-marker prefix plus future detected grasped-corner "
            "trajectories only"
        ),
        "full_lay_post_prefix_free_marker_outcomes_read": False,
        "paper_claim_authorized": False,
    }
    write_json(output / "source_fit.json", source_fit)
    write_json(output / "source_gate.json", gate)

    fieldnames = list(decisions[0])
    with (output / "decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    with (output / "source_transfer.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transfer_rows[0]))
        writer.writeheader()
        writer.writerows(transfer_rows)

    report = [
        "# Tracking Cloth table-query active-probe source feasibility v1",
        "",
        "This run opened the eight half-lay source/probe trajectories. It read",
        "each full-lay target only through a registered causal prefix and the",
        "future trajectories of the detected grasped corners. It did not read any",
        "post-prefix full-lay free-marker coordinate.",
        "",
        "## Selection",
        "",
        "| Held material | Query | Parameter-information | Task-directed | Different |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in decisions:
        report.append(
            f"| {row['held_material']} | {row['query']} | "
            f"{row['parameter_information']} | {row['task_directed']} | "
            f"{'yes' if row['task_vs_parameter_disagree'] else 'no'} |"
        )
    report.extend(
        [
            "",
            "## Source-only gate",
            "",
            f"- Task-vs-parameter K=1 disagreements: `{disagreement_count}/8`.",
            f"- Materials whose task-directed choice changes with query: `{query_switch_materials}/4`.",
            f"- Task-directed unselected-probe transfer wins vs prior: `{transfer_wins_prior}/8`.",
            f"- Task-directed mean prior/updated/persistence RMSE: "
            f"`{task_mean_prior:.3f} / {task_mean_updated:.3f} / "
            f"{task_mean_persistence:.3f} mm`.",
            f"- Selection-divergence gate: `{'pass' if gate['selection_divergence_passes'] else 'fail'}`.",
            f"- Source-transfer gate: `{'pass' if gate['source_transfer_passes'] else 'fail'}`.",
            f"- Separate target protocol authorized by this source gate: "
            f"`{'yes' if gate['authorize_separate_target_protocol'] else 'no'}`.",
            "",
            "This is source-only development evidence. It does not authorize a",
            "paper claim, online safety claim, calibrated joint uncertainty claim,",
            "or target scoring outside a separately frozen and reviewed protocol.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n")
    write_json(
        output / "run_manifest.json",
        {
            "schema": "tracking-cloth-table-query-source-run-v1",
            "completed_at": now(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "protocol_sha256": sha256(output / "protocol.json"),
            "source_fit_sha256": sha256(output / "source_fit.json"),
            "source_gate_sha256": sha256(output / "source_gate.json"),
            "full_lay_post_prefix_free_marker_outcomes_read": False,
            "paper_claim_authorized": False,
        },
    )


def self_test() -> None:
    protocol = {
        "stiffness_per_mass": [80.0, 240.0, 720.0],
        "damping_per_mass": [0.5, 2.0, 8.0],
        "mu_low": [0.03, 0.15, 0.45],
        "mu_high": [0.3, 0.8, 1.5],
    }
    bank = parameter_bank(protocol)
    assert len(bank) == 81
    assert bank[0].friction("low_friction") == 0.03
    points = np.array(
        [[0.105 * col, 0.1485 * row, 0.1] for row in range(5) for col in range(4)],
        dtype=float,
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    order, diagnostic = planar_layout(points @ rotation.T)
    assert set(order.tolist()) == set(range(20))
    assert diagnostic["layout_score"] < 1.8
    times = np.linspace(0.0, 0.4, 9)
    prefix = np.broadcast_to(points[None, ...], (9, 20, 3)).copy()
    prefix[:, [0, 3], 2] += np.linspace(0.0, 0.04, 9)[:, None]
    corners, edge = choose_grasped_edge(times, prefix, 0.0)
    assert set(corners.tolist()) == {0, 3}, edge
    toy_bank = np.stack((prefix, prefix + 0.01, prefix - 0.01))
    mask = np.ones((9, 20), dtype=bool)
    distance = pairwise_mse(toy_bank, mask)
    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.dataset_root is None or args.output is None or args.protocol is None:
        parser.error("--dataset-root, --output and --protocol are required")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        protocol = json.loads(args.protocol.read_text())
        run_source(args.dataset_root, args.output, protocol, args.workers)
    except Exception as exc:
        traceback.print_exc()
        if args.output is not None and args.output.is_dir():
            write_json(
                args.output / "failure.json",
                {
                    "failed_at": now(),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "full_lay_post_prefix_free_marker_outcomes_read": False,
                    "scientific_decision": "incomplete; no target authorization",
                },
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
