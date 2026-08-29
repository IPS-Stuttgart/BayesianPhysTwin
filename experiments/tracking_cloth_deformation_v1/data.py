"""Read-only archive audit and causal CSV views for Zenodo 14644526.

No MATLAB code is executed. The parser follows the paper's Frame, Time, XYZ
layout. A target input view converts future corner columns only; future free
marker values are converted exclusively by the post-seal scoring reader.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FREE_NAME = re.compile(
    r"^(cotton|denim|polyester|wool)_(A2|A3)_(shake|twist)_"
    r"(fast|slow)_(hands|hanger)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Case:
    path: Path
    material: str
    size: str
    motion: str
    speed: str
    grasp: str

    @property
    def specimen(self) -> str:
        return f"{self.material}_{self.size}"

    @property
    def condition(self) -> str:
        return f"{self.speed}_{self.grasp}"

    @property
    def markers(self) -> int:
        return 20 if self.size == "A2" else 12


def digest(path: Path, algorithm: str = "sha256") -> str:
    # MD5 is required only to match the publisher's archive checksum.
    h = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def audit_dataset(
    root: Path, protocol: dict[str, Any]
) -> tuple[list[Case], dict[str, Any]]:
    """Hash bytes, verify extraction against ZIP, and freeze the filename roster.

    Hashing/ZIP integrity reads bytes but does not interpret target measurements.
    Files are never changed, downloaded, extracted, or chmod'ed by this runner.
    """
    root = root.resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset entry escapes the cache: {path.name}")
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    if len(csvs) != protocol["csv_count"]:
        raise ValueError(f"Expected {protocol['csv_count']} CSVs, found {len(csvs)}")
    if len({p.name.lower() for p in csvs}) != len(csvs):
        raise ValueError("Ambiguous duplicate CSV basenames")
    archives = [p for p in files if p.suffix.lower() == ".zip"]
    matching = [p for p in archives if digest(p, "md5") == protocol["archive_md5"]]
    if len(matching) != 1:
        raise ValueError("Expected one retained ZIP matching the published MD5")
    archive = matching[0]
    readers = [p for p in files if p.name.lower() == "read_data.m"]
    licenses = [p for p in files if p.name.lower() == "license.txt"]
    if len(readers) != 4 or len(licenses) != 1:
        raise ValueError("Expected four read_data.m readers and one License.txt")
    license_text = licenses[0].read_text(encoding="utf-8-sig")
    normalized = license_text.lower().replace(" ", "")
    if not any(
        s in normalized for s in ("by-nc-sa", "noncommercial", "non-commercial")
    ):
        raise ValueError(
            "Included license differs from the declared noncommercial policy"
        )
    hashes = {p.name.lower(): digest(p) for p in csvs}
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise ValueError(f"ZIP integrity failure: {bad}")
        entries = [z for z in zipped.infolist() if z.filename.lower().endswith(".csv")]
        if len(entries) != len(csvs):
            raise ValueError("ZIP and extracted CSV inventories disagree")
        seen = set()
        for entry in entries:
            name = Path(entry.filename).name.lower()
            if name in seen or name not in hashes:
                raise ValueError("Ambiguous ZIP CSV identity")
            seen.add(name)
            if hashlib.sha256(zipped.read(entry)).hexdigest() != hashes[name]:
                raise ValueError(f"Extracted bytes differ from verified ZIP: {name}")
    cases = []
    for path in csvs:
        match = FREE_NAME.fullmatch(path.name)
        if match:
            material, size, motion, speed, grasp = match.groups()
            cases.append(
                Case(
                    path,
                    material.lower(),
                    size.upper(),
                    motion.lower(),
                    speed.lower(),
                    grasp.lower(),
                )
            )
    expected = set(
        itertools.product(
            protocol["materials"],
            protocol["sizes"],
            ["shake", "twist"],
            protocol["speeds"],
            protocol["grasps"],
        )
    )
    actual = {(c.material, c.size, c.motion, c.speed, c.grasp) for c in cases}
    if actual != expected or len(cases) != 64:
        raise ValueError("The complete 64-recording free-hanging factorial is required")
    inventory = {
        "dataset_record": protocol["dataset_record"],
        "archive_name": archive.name,
        "archive_md5": protocol["archive_md5"],
        "archive_sha256": digest(archive),
        "csv_count": len(csvs),
        "source_count": 32,
        "target_count": 32,
        "unused_count": len(csvs) - 64,
        "csv_sha256": hashes,
        "reader_sha256": {str(p.relative_to(root)): digest(p) for p in readers},
        "license_sha256": digest(licenses[0]),
        "included_license_text": license_text,
        "license_policy": protocol["license_policy"],
        "target_numeric_outcomes_read": False,
    }
    inventory["inventory_id"] = object_digest(inventory)
    return sorted(cases, key=lambda c: c.path.name), inventory


def _rows(path: Path, markers: int):
    """Yield the documented numeric Frame, Time, XYZ rows, preserving blanks."""
    width = 2 + 3 * markers
    started = False
    last_t = -np.inf
    last_frame = -np.inf
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                frame, time = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                if started:
                    raise ValueError(
                        f"Nonnumeric row after data start: {path.name}"
                    ) from None
                continue
            started = True
            if (
                not np.isfinite([frame, time]).all()
                or frame != int(frame)
                or frame <= last_frame
                or time <= last_t
            ):
                raise ValueError(f"Invalid frame/time order: {path.name}")
            if len(row) < width or any(cell.strip() for cell in row[width:]):
                raise ValueError(
                    f"Expected Frame, Time and {markers} XYZ triplets: {path.name}"
                )
            last_t, last_frame = time, frame
            yield time, row[2:width]
    if not started:
        raise ValueError(f"No numeric rows in {path.name}")


def _positions(cells: list[str], indices: np.ndarray) -> np.ndarray:
    values = []
    for marker in indices:
        triple = [
            float(cells[3 * int(marker) + d])
            if cells[3 * int(marker) + d].strip()
            else np.nan
            for d in range(3)
        ]
        if np.isinf(triple).any():
            raise ValueError("Infinite coordinate")
        values.append(triple if np.isfinite(triple).all() else [np.nan] * 3)
    return np.asarray(values, dtype=float)


def read_prefix(case: Case, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    start = None
    for time, cells in _rows(case.path, case.markers):
        if start is None:
            start = time
        if time - start > seconds + 1e-8:
            break
        times.append(time)
        values.append(_positions(cells, np.arange(case.markers)))
    return np.asarray(times), np.asarray(values)


def infer_source_scale(case: Case, positions: np.ndarray) -> float:
    """Resolve m/cm/mm from source-only initial geometry; never from a target."""
    complete = np.flatnonzero(np.isfinite(positions).all(axis=(1, 2)))
    if not len(complete):
        raise ValueError(f"No complete initialization frame: {case.path.name}")
    first = positions[complete[0]]
    diameter = np.linalg.norm(first[:, None] - first[None, :], axis=2).max()
    nominal = np.hypot(0.42, 0.594) if case.size == "A2" else np.hypot(0.297, 0.42)
    allowed = [s for s in (1.0, 0.01, 0.001) if 0.4 < diameter * s / nominal < 1.6]
    if len(allowed) != 1:
        raise ValueError(
            "Ambiguous coordinate units; inspect source readers before revising protocol"
        )
    return allowed[0]


def layout(initial: np.ndarray, size: str) -> tuple[np.ndarray, np.ndarray]:
    """Initial-frame grid only: four/five descending-z rows, three/four columns.

    A deliberately explicit assumption of the first pilot, not claimed to be the
    official MATLAB marker-ordering implementation. Ambiguous geometry fails.
    """
    rows, cols = (5, 4) if size == "A2" else (4, 3)
    xy = initial[:, :2] - initial[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    horizontal = vh[0]
    horizontal *= 1 if horizontal[np.argmax(abs(horizontal))] >= 0 else -1
    levels = np.argsort(-initial[:, 2], kind="stable").reshape(rows, cols)
    order = np.concatenate(
        [row[np.argsort(xy[row] @ horizontal, kind="stable")] for row in levels]
    )
    grid = initial[order].reshape(rows, cols, 3)
    vertical_steps = grid[:-1, :, 2].mean(axis=1) - grid[1:, :, 2].mean(axis=1)
    horizontal_steps = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    if (
        np.min(vertical_steps) < 0.025
        or np.min(horizontal_steps) < 0.025
        or np.max(horizontal_steps) > 0.25
    ):
        raise ValueError("Initial grid/corner assignment is unsupported")
    # This pilot assumes a near-vertical regular initial mesh, not arbitrary folds.
    if np.max(np.ptp(grid[:, :, 2], axis=1)) > 0.75 * np.median(vertical_steps):
        raise ValueError(
            "Initial row ordering is ambiguous; no outcome-based remapping"
        )
    return order, np.array([0, cols - 1], dtype=int)


@dataclass(frozen=True)
class Inputs:
    times: np.ndarray
    prefix: np.ndarray
    boundary: np.ndarray
    order: np.ndarray
    corners: np.ndarray
    cutoff: int
    initial_time: float
    scale: float


def input_view(case: Case, protocol: dict[str, Any], scale: float) -> Inputs:
    """Only prefix XYZ and future prescribed-corner XYZ enter this view."""
    seconds = protocol["prefix_seconds"]
    time0, pos0 = read_prefix(case, seconds)
    complete = np.flatnonzero(np.isfinite(pos0).all(axis=(1, 2)))
    if not len(complete):
        raise ValueError(f"No complete prefix frame: {case.path.name}")
    first = int(complete[0])
    if time0[first] - time0[0] > protocol["initial_complete_frame_deadline_seconds"]:
        raise ValueError("Late complete initialization frame")
    initial_time = float(time0[first])
    order, corners = layout(pos0[first] * scale, case.size)
    start_time = float(time0[0])
    stride = protocol["sample_stride"]
    end = start_time + seconds + protocol["forecast_seconds"]
    times, prefix, boundary = [], [], []
    last_boundary = None
    row_index = 0
    for time, cells in _rows(case.path, case.markers):
        if time < initial_time - 1e-9:
            continue
        if time > end + 1e-8:
            break
        use = row_index % stride == 0
        row_index += 1
        if not use:
            continue
        b = _positions(cells, order[corners]) * scale
        if last_boundary is not None:
            b = np.where(np.isfinite(b), b, last_boundary)
        if not np.isfinite(b).all():
            raise ValueError("Missing initial driven corner")
        last_boundary = b.copy()
        times.append(time)
        boundary.append(b)
        if time <= start_time + seconds + 1e-8:
            p = _positions(cells, order) * scale
            if prefix:
                p = np.where(np.isfinite(p), p, prefix[-1])
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite causal initialization")
            prefix.append(p)
    times_array = np.asarray(times)
    if len(prefix) < 5 or len(times) <= len(prefix) + 5:
        raise ValueError("Insufficient prefix or forecast frames")
    dt = np.diff(times_array)
    if not np.allclose(dt, stride / 120.0, rtol=0.05, atol=1e-4):
        raise ValueError("Sampling does not match the frozen 120 Hz/stride contract")
    if times_array[-1] < end - 2 * stride / 120.0:
        raise ValueError("Recording does not cover the complete frozen horizon")
    return Inputs(
        times_array,
        np.asarray(prefix),
        np.asarray(boundary),
        order,
        corners,
        len(prefix) - 1,
        initial_time,
        scale,
    )


def scoring_view(case: Case, inputs: Inputs) -> np.ndarray:
    """Called only by source training or after the complete target prediction seal."""
    rows = []
    index = 0
    for time, cells in _rows(case.path, case.markers):
        if index == len(inputs.times):
            break
        if abs(time - inputs.times[index]) <= 1e-7:
            rows.append(_positions(cells, inputs.order) * inputs.scale)
            index += 1
    if index != len(inputs.times):
        raise ValueError("Scoring timestamps do not reproduce the sealed input view")
    return np.asarray(rows)
