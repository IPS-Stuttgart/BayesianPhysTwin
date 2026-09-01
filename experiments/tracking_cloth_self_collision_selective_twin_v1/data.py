"""Causal views for the Tracking Cloth self-collision repetition panel."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SELF_NAME = re.compile(
    r"^(cotton|denim|polyester|wool)_a2_"
    r"(four_corners_normal|four_corners_parallel|two_corners_normal)_"
    r"rep([123])\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Case:
    path: Path
    material: str
    interaction: str
    repetition: int

    @property
    def case_id(self) -> str:
        return f"{self.material}:{self.interaction}:rep{self.repetition}"


@dataclass(frozen=True)
class InputView:
    case: Case
    times: np.ndarray
    cloth_prefix: np.ndarray
    rod_prefix: np.ndarray
    cutoff: int
    scale: float
    cloth_indices: np.ndarray
    cloth_order: np.ndarray
    rod_indices: np.ndarray
    marker_count: int
    initial_diameter_m: float


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_dataset(
    root: Path, protocol: dict[str, Any]
) -> tuple[list[Case], dict[str, Any]]:
    """Verify the public archive and freeze the 36-file self-collision roster.

    Archive hashing and byte comparison do not interpret numeric trajectory
    values. The target repetition remains numerically unopened.
    """

    root = root.resolve(strict=True)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("dataset entry escapes the cache")
    csvs = [path for path in files if path.suffix.lower() == ".csv"]
    if len(csvs) != int(protocol["expected_csv_count"]):
        raise ValueError(
            f"expected {protocol['expected_csv_count']} CSVs, found {len(csvs)}"
        )
    if len({path.name.lower() for path in csvs}) != len(csvs):
        raise ValueError("ambiguous duplicate CSV basename")

    archives = [path for path in files if path.suffix.lower() == ".zip"]
    matching = [
        path
        for path in archives
        if digest(path, "md5") == protocol["archive_md5"]
        and digest(path) == protocol["archive_sha256"]
    ]
    if len(matching) != 1:
        raise ValueError("expected exactly one publisher archive with frozen hashes")
    archive = matching[0]
    hashes = {path.name.lower(): digest(path) for path in csvs}
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad is not None:
            raise ValueError(f"ZIP integrity failure: {bad}")
        entries = [
            entry
            for entry in zipped.infolist()
            if entry.filename.lower().endswith(".csv")
        ]
        if len(entries) != len(csvs):
            raise ValueError("archive and extracted CSV inventories disagree")
        seen: set[str] = set()
        for entry in entries:
            name = Path(entry.filename).name.lower()
            if name in seen or name not in hashes:
                raise ValueError("ambiguous archive CSV identity")
            seen.add(name)
            if hashlib.sha256(zipped.read(entry)).hexdigest() != hashes[name]:
                raise ValueError(f"extracted bytes differ from archive: {name}")

    cases: list[Case] = []
    for path in csvs:
        match = SELF_NAME.fullmatch(path.name)
        if match:
            material, interaction, repetition = match.groups()
            cases.append(
                Case(
                    path=path,
                    material=material.lower(),
                    interaction=interaction.lower(),
                    repetition=int(repetition),
                )
            )
    expected = set(
        itertools.product(
            protocol["materials"],
            protocol["interactions"],
            (1, 2, 3),
        )
    )
    actual = {(case.material, case.interaction, case.repetition) for case in cases}
    if actual != expected or len(cases) != int(
        protocol["expected_self_collision_count"]
    ):
        raise ValueError("complete 4x3x3 self-collision factorial is required")

    license_files = [path for path in files if path.name.lower() == "license.txt"]
    if len(license_files) != 1:
        raise ValueError("expected one included License.txt")
    license_text = license_files[0].read_text(encoding="utf-8-sig")
    normalized = license_text.lower().replace(" ", "")
    license_tokens = ("by-nc-sa", "noncommercial", "non-commercial")
    if not any(token in normalized for token in license_tokens):
        raise ValueError(
            "included license does not match the retained noncommercial policy"
        )

    inventory = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-inventory.v1",
        "dataset_record": protocol["dataset_record"],
        "archive_name": archive.name,
        "archive_md5": digest(archive, "md5"),
        "archive_sha256": digest(archive),
        "csv_count": len(csvs),
        "self_collision_count": len(cases),
        "rep1_count": sum(case.repetition == 1 for case in cases),
        "rep2_count": sum(case.repetition == 2 for case in cases),
        "rep3_count": sum(case.repetition == 3 for case in cases),
        "self_collision_sha256": {
            case.path.name.lower(): hashes[case.path.name.lower()] for case in cases
        },
        "all_csv_sha256": hashes,
        "license_sha256": digest(license_files[0]),
        "included_license_text": license_text,
        "numeric_rep3_outcomes_read": False,
    }
    inventory["inventory_id"] = object_digest(inventory)
    return sorted(cases, key=lambda item: item.path.name), inventory


def _row_stream(path: Path) -> Iterable[tuple[float, float, list[str], int]]:
    """Yield frame, time, coordinate cells, and marker count.

    Coordinate strings are not converted here. This permits the prediction
    view to read future timestamps without numerically opening future positions.
    """

    started = False
    marker_count: int | None = None
    last_frame = -np.inf
    last_time = -np.inf
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                frame = float(row[0])
                time = float(row[1])
            except (ValueError, IndexError):
                if started:
                    raise ValueError(
                        f"nonnumeric row after data start: {path.name}"
                    ) from None
                continue
            started = True
            if not np.isfinite([frame, time]).all() or frame != int(frame):
                raise ValueError(f"invalid frame/time value: {path.name}")
            if frame <= last_frame or time <= last_time:
                raise ValueError(f"nonmonotone frame/time order: {path.name}")
            cells = row[2:]
            if marker_count is None:
                # Some writers leave one or two delimiter-only cells after the
                # final XYZ triplet. Remove only those modulo-three extras; do
                # not erase a genuinely missing final marker.
                while len(cells) % 3 and cells and not cells[-1].strip():
                    cells.pop()
                if len(cells) % 3:
                    raise ValueError(
                        f"coordinate columns are not XYZ triplets: {path.name}"
                    )
                marker_count = len(cells) // 3
                if marker_count < 22:
                    raise ValueError(
                        f"expected cloth plus rod markers, found {marker_count}: "
                        f"{path.name}"
                    )
            expected_width = 3 * marker_count
            if len(cells) < expected_width:
                cells.extend([""] * (expected_width - len(cells)))
            elif len(cells) > expected_width:
                if any(cell.strip() for cell in cells[expected_width:]):
                    raise ValueError(f"unexpected trailing columns: {path.name}")
                cells = cells[:expected_width]
            count = marker_count
            last_frame, last_time = frame, time
            yield frame, time, cells, count
    if not started:
        raise ValueError(f"no numeric rows in {path.name}")


def _positions(cells: list[str], indices: np.ndarray) -> np.ndarray:
    values = []
    for marker in indices:
        triple = []
        for dimension in range(3):
            cell = cells[3 * int(marker) + dimension].strip()
            triple.append(float(cell) if cell else np.nan)
        if np.isinf(triple).any():
            raise ValueError("infinite coordinate")
        values.append(triple if np.isfinite(triple).all() else [np.nan] * 3)
    return np.asarray(values, dtype=float)


def _scale_for_a2(points: np.ndarray) -> float:
    diameter = float(
        np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2))
    )
    nominal = float(np.hypot(0.42, 0.594))
    candidates = [
        scale
        for scale in (1.0, 0.01, 0.001)
        if 0.35 < diameter * scale / nominal < 1.8
    ]
    if len(candidates) != 1:
        raise ValueError("ambiguous A2 coordinate scale")
    return candidates[0]


def _grid_order_score(points_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Order twenty initially near-planar markers as a 5x4 grid."""

    centered = points_m - points_m.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= 1e-8:
        raise ValueError("degenerate cloth initialization")
    projected = centered @ basis[:2].T
    best: tuple[np.ndarray, float] | None = None
    for swap in (False, True):
        uv = projected[:, ::-1] if swap else projected
        for row_sign in (-1.0, 1.0):
            for col_sign in (-1.0, 1.0):
                local = uv * np.array([col_sign, row_sign])
                row_sorted = np.argsort(local[:, 1], kind="stable")
                chunks = row_sorted.reshape(5, 4)
                order = np.concatenate(
                    [
                        chunk[np.argsort(local[chunk, 0], kind="stable")]
                        for chunk in chunks
                    ]
                )
                grid = points_m[order].reshape(5, 4, 3)
                horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
                vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
                if np.min(horizontal) <= 1e-5 or np.min(vertical) <= 1e-5:
                    continue
                h_cv = float(np.std(horizontal) / np.mean(horizontal))
                v_cv = float(np.std(vertical) / np.mean(vertical))
                diagonal = float(np.linalg.norm(grid[-1, -1] - grid[0, 0]))
                score = h_cv + v_cv + 0.1 / max(diagonal, 1e-6)
                if best is None or score < best[1]:
                    best = (order, score)
    if best is None or best[1] > 2.5:
        raise ValueError("unsupported initial cloth grid")
    return best


def _partition_markers(
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    marker_count = len(initial)
    candidate_cloth_sets = [np.arange(20), np.arange(marker_count - 20, marker_count)]
    unique: list[np.ndarray] = []
    for candidate in candidate_cloth_sets:
        if not any(np.array_equal(candidate, existing) for existing in unique):
            unique.append(candidate)

    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None
    for cloth_indices in unique:
        cloth = initial[cloth_indices]
        if not np.isfinite(cloth).all():
            continue
        try:
            scale = _scale_for_a2(cloth)
            order, score = _grid_order_score(cloth * scale)
        except ValueError:
            continue
        cloth_set = set(int(index) for index in cloth_indices)
        remaining = np.asarray(
            [index for index in range(marker_count) if index not in cloth_set],
            dtype=int,
        )
        if len(remaining) < 2:
            continue
        extra = initial[remaining] * scale
        distances = np.linalg.norm(extra[:, None] - extra[None, :], axis=2)
        flat = int(np.argmax(distances))
        left, right = np.unravel_index(flat, distances.shape)
        rod_indices = remaining[np.asarray([left, right])]
        rod_length = float(distances[left, right])
        # The exact rod length is not used for fitting, but the pair must be
        # distinctly longer than the cloth's local grid spacing.
        grid = cloth[order].reshape(5, 4, 3) * scale
        spacing = float(
            np.median(
                np.concatenate(
                    [
                        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                    ]
                )
            )
        )
        if rod_length < 3.0 * spacing:
            continue
        total_score = score + 0.05 / max(rod_length, 1e-6)
        candidate_result = (cloth_indices, order, rod_indices, scale, total_score)
        if best is None or total_score < best[4]:
            best = candidate_result
    if best is None:
        raise ValueError("could not separate the 20 cloth markers from rod markers")
    return best[:4]


def _causal_fill(values: list[np.ndarray]) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    for index in range(len(output)):
        if index == 0 and not np.isfinite(output[index]).all():
            raise ValueError("initial prefix frame is incomplete")
        if index:
            output[index] = np.where(
                np.isfinite(output[index]), output[index], output[index - 1]
            )
    if not np.isfinite(output).all():
        raise ValueError("prefix remains nonfinite after causal carry-forward")
    return output


def prediction_input(case: Case, protocol: dict[str, Any]) -> InputView:
    """Read a causal prefix and future timestamps, but no future coordinates."""

    rows = list(_row_stream(case.path))
    first_time = rows[0][1]
    deadline = first_time + float(protocol["initial_complete_frame_deadline_seconds"])
    initial_row = None
    all_indices = np.arange(rows[0][3], dtype=int)
    for _, time, cells, count in rows:
        if count != rows[0][3]:
            raise ValueError("marker count changed")
        if time > deadline + 1e-9:
            break
        values = _positions(cells, all_indices)
        if np.isfinite(values).all():
            initial_row = (time, values)
            break
    if initial_row is None:
        raise ValueError(f"no complete initialization frame: {case.path.name}")
    start_time, initial = initial_row
    cloth_indices, cloth_order, rod_indices, scale = _partition_markers(initial)

    prefix_end = start_time + float(protocol["prefix_seconds"])
    forecast_end = prefix_end + float(protocol["forecast_seconds"])
    stride = int(protocol["sample_stride"])
    selected_times: list[float] = []
    cloth_prefix: list[np.ndarray] = []
    rod_prefix: list[np.ndarray] = []
    numeric_index = 0
    cutoff = -1
    for _, time, cells, _ in rows:
        if time < start_time - 1e-9:
            continue
        if time > forecast_end + 1e-8:
            break
        use = numeric_index % stride == 0
        numeric_index += 1
        if not use:
            continue
        selected_times.append(time)
        if time <= prefix_end + 1e-8:
            cloth = _positions(cells, cloth_indices)[cloth_order] * scale
            rod = _positions(cells, rod_indices) * scale
            cloth_prefix.append(cloth)
            rod_prefix.append(rod)
            cutoff = len(selected_times) - 1
        # For future rows only time is converted. Coordinate cells remain text.

    times = np.asarray(selected_times, dtype=float)
    if cutoff < 4 or len(times) <= cutoff + 5:
        raise ValueError("insufficient causal prefix or forecast")
    cloth_array = _causal_fill(cloth_prefix)
    rod_array = _causal_fill(rod_prefix)
    dt = np.diff(times)
    expected_dt = stride / 120.0
    if not np.allclose(dt, expected_dt, rtol=0.05, atol=1e-4):
        raise ValueError("sampling does not match the frozen 120 Hz contract")
    if times[-1] < forecast_end - 2 * expected_dt:
        raise ValueError("recording does not cover the complete forecast horizon")
    rod_lengths = np.linalg.norm(rod_array[:, 1] - rod_array[:, 0], axis=1)
    if np.mean(rod_lengths) <= 0 or np.std(rod_lengths) / np.mean(rod_lengths) > 0.08:
        raise ValueError("rod endpoint pair is not length-stable in the prefix")
    initial_diameter = float(
        np.max(
            np.linalg.norm(
                cloth_array[0, :, None] - cloth_array[0, None, :], axis=2
            )
        )
    )
    return InputView(
        case=case,
        times=times,
        cloth_prefix=cloth_array,
        rod_prefix=rod_array,
        cutoff=cutoff,
        scale=scale,
        cloth_indices=cloth_indices,
        cloth_order=cloth_order,
        rod_indices=rod_indices,
        marker_count=len(initial),
        initial_diameter_m=initial_diameter,
    )


def scoring_truth(case: Case, inputs: InputView) -> np.ndarray:
    """Read cloth outcomes at the sealed timestamps after target authorization."""

    truth: list[np.ndarray] = []
    time_index = 0
    for _, time, cells, count in _row_stream(case.path):
        if count != inputs.marker_count:
            raise ValueError("marker count differs from the sealed prediction input")
        if time_index == len(inputs.times):
            break
        if abs(time - inputs.times[time_index]) <= 1e-7:
            cloth = _positions(cells, inputs.cloth_indices)[inputs.cloth_order]
            truth.append(cloth * inputs.scale)
            time_index += 1
    if time_index != len(inputs.times):
        raise ValueError("truth timestamps do not reproduce the prediction grid")
    return np.asarray(truth, dtype=float)
