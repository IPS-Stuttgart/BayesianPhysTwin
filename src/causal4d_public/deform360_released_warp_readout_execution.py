"""Execute and score the locked released-particle official-Warp source audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_phystwin_feasibility import (
    OFFICIAL_SIMULATOR_RELATIVE_PATH,
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
)
from .deform360_released_warp_readout import (
    PINNED_OFFICIAL_PHYSTWIN_COMMIT,
    load_released_warp_readout_protocol,
    released_pcd_manifest,
    validate_released_warp_readout_protocol,
)
from .deform360_replication_graph import Deform360SparseGraph
from .deform360_replication_warp import (
    Deform360WarpForecastCase,
    OfficialWarpSparseGraphRunner,
    sparse_graph_strain_summary,
)

RELEASED_WARP_PREDICTION_SCHEMA_VERSION = 1
RELEASED_WARP_PREDICTION_KIND = "Deform360ReleasedWarpReadoutPrediction"
RELEASED_WARP_SCORE_SCHEMA_VERSION = 1
RELEASED_WARP_SCORE_KIND = "Deform360ReleasedWarpReadoutScore"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _ordered_identity_sha256(points_m: np.ndarray) -> str:
    points = np.ascontiguousarray(points_m, dtype="<f4")
    digest = hashlib.sha256(b"deform360-ordered-material-identity-v1\0")
    digest.update(_canonical_bytes(list(points.shape)))
    digest.update(b"\0")
    digest.update(points.tobytes(order="C"))
    return digest.hexdigest()


def _load_released_points(path: Path) -> np.ndarray:
    _require(path.is_file(), f"released particle frame is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        _require("pts" in archive.files, f"released frame has no pts field: {path}")
        points = np.asarray(archive["pts"], dtype=np.float64)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and len(points) >= 1
        and np.all(np.isfinite(points)),
        f"released frame points are invalid: {path}",
    )
    return points


@dataclass(frozen=True)
class PolylineMaterialAssociation:
    """Fixed association from ordered particles to a sparse rope polyline."""

    segment_indices: np.ndarray
    barycentric_coordinates: np.ndarray
    local_offsets_m: np.ndarray

    def __post_init__(self) -> None:
        segments = np.asarray(self.segment_indices, dtype=np.int32)
        barycentric = np.asarray(self.barycentric_coordinates, dtype=np.float64)
        offsets = np.asarray(self.local_offsets_m, dtype=np.float64)
        _require(segments.ndim == 1, "association segments must be one-dimensional")
        _require(
            barycentric.shape == segments.shape,
            "association barycentric coordinates differ from segments",
        )
        _require(
            offsets.shape == (len(segments), 3),
            "association offsets must have shape (N,3)",
        )
        _require(
            np.all(segments >= 0)
            and np.all(np.isfinite(barycentric))
            and np.all((0.0 <= barycentric) & (barycentric <= 1.0))
            and np.all(np.isfinite(offsets)),
            "association contains invalid values",
        )
        for name, values in (
            ("segment_indices", segments),
            ("barycentric_coordinates", barycentric),
            ("local_offsets_m", offsets),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def associate_particles_to_polyline(
    particles_m: np.ndarray,
    polyline_m: np.ndarray,
) -> PolylineMaterialAssociation:
    """Associate each particle with the closest clamped polyline segment."""

    particles = np.asarray(particles_m, dtype=np.float64)
    polyline = np.asarray(polyline_m, dtype=np.float64)
    _require(
        particles.ndim == 2
        and particles.shape[1] == 3
        and len(particles) >= 1,
        "particles must have shape (N,3)",
    )
    _require(
        polyline.ndim == 2
        and polyline.shape[1] == 3
        and len(polyline) >= 2,
        "polyline must have shape (M,3)",
    )
    _require(
        np.all(np.isfinite(particles)) and np.all(np.isfinite(polyline)),
        "association inputs must be finite",
    )
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    _require(
        np.all(squared_lengths > 1e-12),
        "polyline contains a degenerate segment",
    )
    relative = particles[:, None, :] - starts[None, :, :]
    barycentric = np.einsum("nsd,sd->ns", relative, vectors)
    barycentric /= squared_lengths[None, :]
    barycentric = np.clip(barycentric, 0.0, 1.0)
    projected = starts[None] + barycentric[..., None] * vectors[None]
    squared_distance = np.sum((particles[:, None] - projected) ** 2, axis=2)
    segments = np.argmin(squared_distance, axis=1).astype(np.int32)
    row = np.arange(len(particles))
    selected_barycentric = barycentric[row, segments]
    selected_projection = projected[row, segments]
    return PolylineMaterialAssociation(
        segment_indices=segments,
        barycentric_coordinates=selected_barycentric,
        local_offsets_m=particles - selected_projection,
    )


def minimum_rotation_matrix(
    source_vector: np.ndarray,
    target_vector: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return the deterministic minimum rotation from source to target."""

    source = np.asarray(source_vector, dtype=np.float64)
    target = np.asarray(target_vector, dtype=np.float64)
    _require(source.shape == target.shape == (3,), "rotation vectors must be 3D")
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "rotation vectors must be finite",
    )
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= epsilon or target_norm <= epsilon:
        return np.eye(3, dtype=np.float64)
    source_unit = source / source_norm
    target_unit = target / target_norm
    cosine = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    cross = np.cross(source_unit, target_unit)
    sine = float(np.linalg.norm(cross))
    if sine > epsilon:
        skew = np.asarray(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))
    if cosine >= 0.0:
        return np.eye(3, dtype=np.float64)
    basis = np.zeros(3, dtype=np.float64)
    basis[int(np.argmin(np.abs(source_unit)))] = 1.0
    axis = np.cross(source_unit, basis)
    axis /= np.linalg.norm(axis)
    return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)


def lift_sparse_polyline_to_particles(
    sparse_trajectory_m: np.ndarray,
    origin_polyline_m: np.ndarray,
    association: PolylineMaterialAssociation,
    *,
    rotate_offsets: bool,
) -> np.ndarray:
    """Read a sparse rope trajectory out at fixed ordered particle identities."""

    trajectory = np.asarray(sparse_trajectory_m, dtype=np.float64)
    origin = np.asarray(origin_polyline_m, dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[1:] == origin.shape
        and origin.ndim == 2
        and origin.shape[1] == 3,
        "sparse trajectory and origin polyline differ",
    )
    segments = association.segment_indices
    _require(
        np.all(segments + 1 < len(origin)),
        "association segment lies outside the sparse polyline",
    )
    barycentric = association.barycentric_coordinates
    interpolated = (
        (1.0 - barycentric)[None, :, None] * trajectory[:, segments]
        + barycentric[None, :, None] * trajectory[:, segments + 1]
    )
    if not rotate_offsets:
        return interpolated + association.local_offsets_m[None]
    source_tangents = origin[1:] - origin[:-1]
    target_tangents = trajectory[:, 1:] - trajectory[:, :-1]
    rotations = np.empty(
        (len(trajectory), len(source_tangents), 3, 3),
        dtype=np.float64,
    )
    for frame in range(len(trajectory)):
        for segment in range(len(source_tangents)):
            rotations[frame, segment] = minimum_rotation_matrix(
                source_tangents[segment],
                target_tangents[frame, segment],
            )
    selected_rotations = rotations[:, segments]
    transported = np.einsum(
        "tnij,nj->tni",
        selected_rotations,
        association.local_offsets_m,
    )
    return interpolated + transported


def _chain_graph(positions_m: np.ndarray) -> Deform360SparseGraph:
    positions = np.asarray(positions_m, dtype=np.float64)
    node_count = len(positions)
    stretch = [(index, index + 1) for index in range(node_count - 1)]
    bend = [(index, index + 2) for index in range(node_count - 2)]
    edges = np.asarray(stretch + bend, dtype=np.int32)
    families = np.concatenate(
        (
            np.zeros(len(stretch), dtype=np.int8),
            np.ones(len(bend), dtype=np.int8),
        )
    )
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=edges,
        spring_families=families,
        masses=np.ones(node_count, dtype=np.float64),
        stratum="filament",
        diagnostics={"construction": "locked ordered 21-node source centerline"},
    )


def build_matched_origin_warp_cases(
    source_npz_path: str | Path,
    episode_record: Mapping[str, Any],
    *,
    dt_seconds: float,
) -> tuple[Deform360WarpForecastCase, Deform360WarpForecastCase, dict[str, Any]]:
    """Build finite- and zero-velocity cases without reading future object state."""

    source_path = Path(source_npz_path)
    _require(source_path.is_file(), f"source observation is missing: {source_path}")
    with np.load(source_path, allow_pickle=False) as archive:
        required = {
            "frame_indices",
            "positions_m",
            "controller_positions_m",
            "contact_active",
            "contact_node_indices",
            "contact_offsets_m",
        }
        _require(set(archive.files) == required, "source observation fields changed")
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        positions = np.asarray(archive["positions_m"], dtype=np.float64)
        controllers = np.asarray(
            archive["controller_positions_m"],
            dtype=np.float64,
        )
        contact_active = np.asarray(archive["contact_active"], dtype=bool)
        contact_nodes = tuple(
            map(int, np.asarray(archive["contact_node_indices"]).tolist())
        )
        contact_offsets = np.asarray(
            archive["contact_offsets_m"],
            dtype=np.float64,
        )
    frame_lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
    origin_frame = int(episode_record["matched_origin_frame"])
    previous_frame = int(episode_record["previous_state_frame"])
    rollout_frames = [
        origin_frame,
        *map(int, episode_record["evaluation_frames"]),
    ]
    _require(
        previous_frame in frame_lookup
        and all(frame in frame_lookup for frame in rollout_frames),
        "source observation misses a locked frame",
    )
    origin_index = frame_lookup[origin_frame]
    previous_index = frame_lookup[previous_frame]
    rollout_indices = np.asarray(
        [frame_lookup[frame] for frame in rollout_frames],
        dtype=np.int64,
    )
    graph = _chain_graph(positions[origin_index])
    initial_velocity = (
        positions[origin_index] - positions[previous_index]
    ) / float(dt_seconds)
    selected_controllers = controllers[rollout_indices]
    origin_contact = contact_active[origin_index]
    static_contact = np.repeat(
        origin_contact[None],
        len(rollout_frames),
        axis=0,
    )
    contact_rest = np.maximum(np.linalg.norm(contact_offsets, axis=1), 0.001)
    common = {
        "episode_id": f"001-rope/episode_{int(episode_record['episode_id']):04d}",
        "graph": graph,
        "controller_positions_m": selected_controllers,
        "contact_active": static_contact,
        "contact_node_indices": contact_nodes,
        "contact_rest_lengths_m": contact_rest,
        "dt_seconds": float(dt_seconds),
        "support_height_m": float(np.min(graph.positions_m[:, 1])),
    }
    finite_case = Deform360WarpForecastCase(
        **common,
        initial_velocities_m_s=initial_velocity,
    )
    zero_case = Deform360WarpForecastCase(
        **common,
        initial_velocities_m_s=np.zeros_like(initial_velocity),
    )
    diagnostics = {
        "rollout_frames": rollout_frames,
        "origin_source_index": int(origin_index),
        "previous_source_index": int(previous_index),
        "initial_velocity_rms_m_s": float(
            np.sqrt(np.mean(np.square(initial_velocity)))
        ),
        "initial_velocity_maximum_m_s": float(
            np.max(np.linalg.norm(initial_velocity, axis=1))
        ),
        "origin_contact_active": origin_contact.astype(int).tolist(),
        "future_contact_active_used": False,
        "contact_node_indices": list(contact_nodes),
    }
    return finite_case, zero_case, diagnostics


def _candidate(
    protocol: Mapping[str, Any],
    candidate_index: int,
) -> WarpRopeCandidate:
    parameters = protocol["config"]["prior_source_gate"]["candidate_parameters"][
        str(int(candidate_index))
    ]
    return WarpRopeCandidate(
        stretch_spring_y=float(parameters["stretch_spring_y"]),
        bend_spring_y=float(parameters["bend_spring_y"]),
        controller_spring_y=float(parameters["controller_spring_y"]),
        ground_friction=float(parameters["ground_friction"]),
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    _require(path.suffix == ".npz", "prediction archive must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def run_released_warp_readout_predictions(
    protocol_path: str | Path,
    *,
    official_repo: str | Path,
    source_observation_root: str | Path,
    released_object_root: str | Path,
    output_archive_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Seal all predictions before any released future particle is loaded."""

    protocol = load_released_warp_readout_protocol(protocol_path)
    config = protocol["config"]
    official = Path(official_repo).resolve()
    _require(official.is_dir(), "official PhysTwin repository is missing")
    official_head = _git_head(official)
    _require(
        official_head == PINNED_OFFICIAL_PHYSTWIN_COMMIT,
        "official PhysTwin checkout differs from the lock",
    )
    simulator_source = official / OFFICIAL_SIMULATOR_RELATIVE_PATH
    _require(simulator_source.is_file(), "official Warp simulator source is missing")
    source_root = Path(source_observation_root)
    released_root = Path(released_object_root)
    warp_config = WarpRopeFeasibilityConfig()
    arrays: dict[str, np.ndarray] = {}
    episode_rows = []
    origin_files_read: list[str] = []
    for record in config["episodes"]:
        episode_id = int(record["episode_id"])
        source_npz = (
            source_root
            / f"deform360_001_rope_source{episode_id}_observation_v5.npz"
        )
        _require(
            _sha256_file(source_npz) == record["source_npz_sha256"],
            f"episode {episode_id} source archive checksum changed",
        )
        finite_case, zero_case, case_diagnostics = build_matched_origin_warp_cases(
            source_npz,
            record,
            dt_seconds=float(config["temporal_policy"]["simulator_dt_seconds"]),
        )
        origin_frame = int(record["matched_origin_frame"])
        origin_path = (
            released_root
            / f"episode_{episode_id}"
            / "pcd_clean"
            / f"{origin_frame:06d}.npz"
        )
        origin_particles = _load_released_points(origin_path)
        origin_files_read.append(
            origin_path.relative_to(released_root).as_posix()
        )
        association = associate_particles_to_polyline(
            origin_particles,
            finite_case.graph.positions_m,
        )
        loo_index = int(record["loo_candidate_index"])
        pooled_index = int(config["prior_source_gate"]["pooled_candidate_index"])
        loo_candidate = _candidate(protocol, loo_index)
        pooled_candidate = _candidate(protocol, pooled_index)

        finite_runner = OfficialWarpSparseGraphRunner(
            official,
            finite_case,
            warp_config,
            device=device,
        )
        loo_sparse = finite_runner.rollout(loo_candidate)
        loo_repeat_sparse = finite_runner.rollout(loo_candidate)
        if pooled_index == loo_index:
            pooled_sparse = loo_sparse.copy()
        else:
            pooled_sparse = finite_runner.rollout(pooled_candidate)
        zero_runner = OfficialWarpSparseGraphRunner(
            official,
            zero_case,
            warp_config,
            device=device,
        )
        loo_zero_sparse = zero_runner.rollout(loo_candidate)

        loo_rotated = lift_sparse_polyline_to_particles(
            loo_sparse,
            finite_case.graph.positions_m,
            association,
            rotate_offsets=True,
        )
        loo_fixed = lift_sparse_polyline_to_particles(
            loo_sparse,
            finite_case.graph.positions_m,
            association,
            rotate_offsets=False,
        )
        pooled_rotated = lift_sparse_polyline_to_particles(
            pooled_sparse,
            finite_case.graph.positions_m,
            association,
            rotate_offsets=True,
        )
        zero_rotated = lift_sparse_polyline_to_particles(
            loo_zero_sparse,
            finite_case.graph.positions_m,
            association,
            rotate_offsets=True,
        )
        persistence = np.repeat(
            origin_particles[None],
            len(loo_sparse),
            axis=0,
        )
        prefix = f"episode_{episode_id:04d}"
        episode_arrays = {
            f"{prefix}_frame_indices": np.asarray(
                case_diagnostics["rollout_frames"],
                dtype=np.int32,
            ),
            f"{prefix}_origin_particles_m": origin_particles.astype(np.float32),
            f"{prefix}_association_segments": (
                association.segment_indices.astype(np.int32)
            ),
            f"{prefix}_association_barycentric": (
                association.barycentric_coordinates.astype(np.float32)
            ),
            f"{prefix}_association_offsets_m": (
                association.local_offsets_m.astype(np.float32)
            ),
            f"{prefix}_loo_finite_sparse_m": loo_sparse.astype(np.float32),
            f"{prefix}_loo_finite_repeat_sparse_m": (
                loo_repeat_sparse.astype(np.float32)
            ),
            f"{prefix}_loo_zero_sparse_m": loo_zero_sparse.astype(np.float32),
            f"{prefix}_pooled_finite_sparse_m": pooled_sparse.astype(np.float32),
            f"{prefix}_loo_finite_rotated_m": loo_rotated.astype(np.float32),
            f"{prefix}_loo_finite_fixed_m": loo_fixed.astype(np.float32),
            f"{prefix}_loo_zero_rotated_m": zero_rotated.astype(np.float32),
            f"{prefix}_pooled_finite_rotated_m": pooled_rotated.astype(np.float32),
            f"{prefix}_persistence_m": persistence.astype(np.float32),
        }
        arrays.update(episode_arrays)
        repeat_rmse = float(
            np.sqrt(np.mean(np.square(loo_sparse - loo_repeat_sparse)))
        )
        strains = {
            "loo_finite": sparse_graph_strain_summary(
                finite_case.graph,
                loo_sparse,
                rest_lengths_m=finite_case.object_rest_lengths_m,
                spring_family=0,
            ),
            "loo_zero": sparse_graph_strain_summary(
                zero_case.graph,
                loo_zero_sparse,
                rest_lengths_m=zero_case.object_rest_lengths_m,
                spring_family=0,
            ),
            "pooled_finite": sparse_graph_strain_summary(
                finite_case.graph,
                pooled_sparse,
                rest_lengths_m=finite_case.object_rest_lengths_m,
                spring_family=0,
            ),
        }
        episode_rows.append(
            {
                "episode_id": episode_id,
                "action": record["action"],
                "frame_indices": case_diagnostics["rollout_frames"],
                "loo_candidate_index": loo_index,
                "pooled_candidate_index": pooled_index,
                "origin_particle_count": len(origin_particles),
                "origin_particle_identity_sha256": _ordered_identity_sha256(
                    origin_particles
                ),
                "origin_pcd_relative_path": origin_path.relative_to(
                    released_root
                ).as_posix(),
                "origin_pcd_sha256": _sha256_file(origin_path),
                "association": {
                    "particle_count": len(association.segment_indices),
                    "direct_support_fraction": 1.0,
                    "segment_count": len(np.unique(association.segment_indices)),
                    "maximum_offset_m": float(
                        np.max(np.linalg.norm(association.local_offsets_m, axis=1))
                    ),
                },
                "case_diagnostics": case_diagnostics,
                "repeat_rollout_rmse_m": repeat_rmse,
                "strain": strains,
                "all_prediction_arrays_finite": bool(
                    all(np.all(np.isfinite(value)) for value in episode_arrays.values())
                ),
            }
        )

    archive = _write_npz(Path(output_archive_path).resolve(), arrays)
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": RELEASED_WARP_PREDICTION_SCHEMA_VERSION,
        "artifact_kind": RELEASED_WARP_PREDICTION_KIND,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "prediction_code_commit": _git_head(repository_root),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "official_phystwin": {
            "commit": official_head,
            "simulator_source_sha256": _sha256_file(simulator_source),
        },
        "device": device,
        "episodes": episode_rows,
        "prediction_archive": {
            "path": archive.name,
            "sha256": _sha256_file(archive),
            "bytes": archive.stat().st_size,
            "keys": sorted(arrays),
        },
        "information_boundary": {
            "source_episode_ids": config["development_source_episode_ids"],
            "forbidden_episode_ids": config["forbidden_episode_ids"],
            "released_particle_files_read": origin_files_read,
            "released_future_particle_file_count_read": 0,
            "future_contact_active_used": False,
            "future_controller_motion_used": True,
            "dense_scores_computed": False,
            "held_v8_access": False,
        },
        "claim_boundary": config["claim_boundary"]["description"],
    }


def write_released_warp_prediction_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Seal a prediction artifact around its already-written archive."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    artifact["result_sha256"] = _artifact_sha256(artifact)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def validate_released_warp_prediction_artifact(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    artifact_directory: str | Path,
    verify_archive: bool = True,
) -> dict[str, Any]:
    """Validate the sealed no-outcome prediction artifact."""

    validate_released_warp_readout_protocol(protocol)
    _require(
        payload.get("schema_version") == RELEASED_WARP_PREDICTION_SCHEMA_VERSION,
        "unsupported released-Warp prediction schema",
    )
    _require(
        payload.get("artifact_kind") == RELEASED_WARP_PREDICTION_KIND,
        "unexpected released-Warp prediction kind",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "released-Warp prediction checksum mismatch",
    )
    _require(
        payload.get("protocol_config_sha256") == protocol["config_sha256"],
        "prediction uses a different protocol lock",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("released_future_particle_file_count_read") == 0
        and boundary.get("future_contact_active_used") is False
        and boundary.get("dense_scores_computed") is False
        and boundary.get("held_v8_access") is False,
        "prediction information boundary changed",
    )
    expected_ids = protocol["config"]["development_source_episode_ids"]
    _require(
        [row["episode_id"] for row in payload["episodes"]] == expected_ids,
        "prediction episode order changed",
    )
    archive = Path(artifact_directory) / payload["prediction_archive"]["path"]
    if verify_archive:
        _require(archive.is_file(), "prediction archive is missing")
        _require(
            _sha256_file(archive) == payload["prediction_archive"]["sha256"],
            "prediction archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                sorted(stored.files) == payload["prediction_archive"]["keys"],
                "prediction archive keys changed",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "episode_count": len(expected_ids),
        "prediction_archive_sha256": payload["prediction_archive"]["sha256"],
    }


def load_released_warp_prediction_artifact(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    verify_archive: bool = True,
) -> dict[str, Any]:
    prediction_path = Path(path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "prediction artifact must contain an object")
    validate_released_warp_prediction_artifact(
        payload,
        protocol=protocol,
        artifact_directory=prediction_path.parent,
        verify_archive=verify_archive,
    )
    return payload


def symmetric_chamfer_distance_m(
    reference_points_m: np.ndarray,
    prediction_points_m: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Exact symmetric mean Euclidean Chamfer with bounded working memory."""

    reference = np.asarray(reference_points_m, dtype=np.float64)
    prediction = np.asarray(prediction_points_m, dtype=np.float64)
    _require(
        reference.ndim == prediction.ndim == 2
        and reference.shape[1] == prediction.shape[1] == 3
        and len(reference) >= 1
        and len(prediction) >= 1,
        "Chamfer inputs must be nonempty 3D point sets",
    )
    _require(
        np.all(np.isfinite(reference)) and np.all(np.isfinite(prediction)),
        "Chamfer inputs must be finite",
    )
    _require(chunk_size >= 1, "Chamfer chunk size must be positive")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        reference_to_prediction = cKDTree(prediction).query(
            reference,
            k=1,
            eps=0.0,
            workers=1,
        )[0]
        prediction_to_reference = cKDTree(reference).query(
            prediction,
            k=1,
            eps=0.0,
            workers=1,
        )[0]
        return 0.5 * (
            float(np.mean(reference_to_prediction))
            + float(np.mean(prediction_to_reference))
        )

    def directed(source: np.ndarray, target: np.ndarray) -> float:
        minima = np.empty(len(source), dtype=np.float64)
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            difference = source[start:stop, None] - target[None]
            squared = np.einsum("nmd,nmd->nm", difference, difference)
            minima[start:stop] = np.sqrt(np.min(squared, axis=1))
        return float(np.mean(minima))

    return 0.5 * (directed(reference, prediction) + directed(prediction, reference))


def _trajectory_metrics(
    target_m: np.ndarray,
    prediction_m: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target_m, dtype=np.float64)
    prediction = np.asarray(prediction_m, dtype=np.float64)
    _require(
        target.shape == prediction.shape
        and target.ndim == 3
        and target.shape[2] == 3
        and len(target) >= 3,
        "target and prediction trajectories differ",
    )
    identity_per_frame = np.mean(
        np.linalg.norm(prediction - target, axis=2),
        axis=1,
    )
    chamfer_per_frame = np.asarray(
        [
            symmetric_chamfer_distance_m(reference, forecast)
            for reference, forecast in zip(target, prediction, strict=True)
        ],
        dtype=np.float64,
    )
    horizons = {}
    for label, indices in zip(
        ("early", "middle", "late"),
        np.array_split(np.arange(len(target)), 3),
        strict=True,
    ):
        horizons[label] = {
            "frame_count": len(indices),
            "chamfer_m": float(np.mean(chamfer_per_frame[indices])),
            "identity_error_m": float(np.mean(identity_per_frame[indices])),
        }
    return {
        "frame_count": len(target),
        "chamfer_m": float(np.mean(chamfer_per_frame)),
        "identity_error_m": float(np.mean(identity_per_frame)),
        "endpoint_chamfer_m": float(chamfer_per_frame[-1]),
        "endpoint_identity_error_m": float(identity_per_frame[-1]),
        "per_frame_chamfer_m": chamfer_per_frame.tolist(),
        "per_frame_identity_error_m": identity_per_frame.tolist(),
        "horizon": horizons,
    }


def score_released_warp_readout_predictions(
    protocol_path: str | Path,
    prediction_artifact_path: str | Path,
    *,
    released_object_root: str | Path,
) -> dict[str, Any]:
    """Open released future particles only after validating the prediction seal."""

    protocol = load_released_warp_readout_protocol(protocol_path)
    prediction_path = Path(prediction_artifact_path)
    prediction = load_released_warp_prediction_artifact(
        prediction_path,
        protocol=protocol,
    )
    released_root = Path(released_object_root)
    observed_manifest = released_pcd_manifest(released_root, protocol)
    locked_manifest = protocol["config"]["author_release"]["pcd_frame_manifest"]
    _require(
        observed_manifest["file_count"] == locked_manifest["file_count"]
        and observed_manifest["sha256"] == locked_manifest["sha256"],
        "released particle manifest differs from the lock",
    )
    archive_path = prediction_path.parent / prediction["prediction_archive"]["path"]
    episode_results = []
    arm_names = (
        "constant_persistence",
        "loo_finite_velocity_rotated_offset",
        "loo_zero_velocity_rotated_offset",
        "pooled_finite_velocity_rotated_offset",
        "loo_finite_velocity_fixed_offset",
    )
    suffix_by_arm = {
        "constant_persistence": "persistence_m",
        "loo_finite_velocity_rotated_offset": "loo_finite_rotated_m",
        "loo_zero_velocity_rotated_offset": "loo_zero_rotated_m",
        "pooled_finite_velocity_rotated_offset": "pooled_finite_rotated_m",
        "loo_finite_velocity_fixed_offset": "loo_finite_fixed_m",
    }
    with np.load(archive_path, allow_pickle=False) as stored:
        for record in protocol["config"]["episodes"]:
            episode_id = int(record["episode_id"])
            prefix = f"episode_{episode_id:04d}"
            origin = np.asarray(
                stored[f"{prefix}_origin_particles_m"],
                dtype=np.float64,
            )
            targets = []
            target_files = []
            for frame in map(int, record["evaluation_frames"]):
                path = (
                    released_root
                    / f"episode_{episode_id}"
                    / "pcd_clean"
                    / f"{frame:06d}.npz"
                )
                points = _load_released_points(path)
                _require(
                    points.shape == origin.shape,
                    f"episode {episode_id} ordered particle count changed",
                )
                targets.append(points)
                target_files.append(path.relative_to(released_root).as_posix())
            target = np.stack(targets)
            metrics = {}
            for arm in arm_names:
                full_prediction = np.asarray(
                    stored[f"{prefix}_{suffix_by_arm[arm]}"],
                    dtype=np.float64,
                )
                _require(
                    len(full_prediction) == len(target) + 1,
                    f"episode {episode_id} {arm} frame count changed",
                )
                metrics[arm] = _trajectory_metrics(target, full_prediction[1:])
            prediction_row = next(
                row
                for row in prediction["episodes"]
                if int(row["episode_id"]) == episode_id
            )
            episode_results.append(
                {
                    "episode_id": episode_id,
                    "action": record["action"],
                    "evaluation_frames": record["evaluation_frames"],
                    "target_files_read_post_seal": target_files,
                    "particle_count": len(origin),
                    "metrics": metrics,
                    "prediction_diagnostics": {
                        "repeat_rollout_rmse_m": prediction_row[
                            "repeat_rollout_rmse_m"
                        ],
                        "strain": prediction_row["strain"],
                        "direct_support_fraction": prediction_row["association"][
                            "direct_support_fraction"
                        ],
                    },
                }
            )

    panel = {}
    for arm in arm_names:
        panel[arm] = {
            metric: float(
                np.mean(
                    [
                        episode["metrics"][arm][metric]
                        for episode in episode_results
                    ]
                )
            )
            for metric in (
                "chamfer_m",
                "identity_error_m",
                "endpoint_chamfer_m",
                "endpoint_identity_error_m",
            )
        }
        panel[arm]["episode_chamfer_wins_vs_persistence"] = int(
            np.count_nonzero(
                [
                    episode["metrics"][arm]["chamfer_m"]
                    < episode["metrics"]["constant_persistence"]["chamfer_m"]
                    for episode in episode_results
                ]
            )
        )
    primary_name = "loo_finite_velocity_rotated_offset"
    persistence_name = "constant_persistence"
    primary = panel[primary_name]
    persistence = panel[persistence_name]
    chamfer_improvement = 1.0 - primary["chamfer_m"] / persistence["chamfer_m"]
    identity_ratio = primary["identity_error_m"] / persistence["identity_error_m"]
    maximum_strain = float(
        max(
            episode["prediction_diagnostics"]["strain"]["loo_finite"]["p99"]
            for episode in episode_results
        )
    )
    maximum_repeat = float(
        max(
            episode["prediction_diagnostics"]["repeat_rollout_rmse_m"]
            for episode in episode_results
        )
    )
    gate = protocol["config"]["transfer_gate"]
    gates = {
        "mean_chamfer_improvement": {
            "observed": chamfer_improvement,
            "required_minimum": gate[
                "minimum_mean_chamfer_improvement_vs_matched_persistence"
            ],
            "passed": bool(
                chamfer_improvement
                >= gate[
                    "minimum_mean_chamfer_improvement_vs_matched_persistence"
                ]
            ),
        },
        "episode_chamfer_wins": {
            "observed": primary["episode_chamfer_wins_vs_persistence"],
            "required_minimum": gate["minimum_episode_chamfer_wins"],
            "passed": bool(
                primary["episode_chamfer_wins_vs_persistence"]
                >= gate["minimum_episode_chamfer_wins"]
            ),
        },
        "mean_identity_error_ratio": {
            "observed": identity_ratio,
            "required_maximum": gate[
                "maximum_mean_identity_error_ratio_vs_matched_persistence"
            ],
            "passed": bool(
                identity_ratio
                <= gate[
                    "maximum_mean_identity_error_ratio_vs_matched_persistence"
                ]
            ),
        },
        "p99_relative_edge_strain": {
            "observed_maximum": maximum_strain,
            "required_maximum": gate["maximum_p99_relative_edge_strain"],
            "passed": bool(
                maximum_strain <= gate["maximum_p99_relative_edge_strain"]
            ),
        },
        "all_states_finite": {
            "passed": bool(
                all(
                    row["all_prediction_arrays_finite"]
                    for row in prediction["episodes"]
                )
            )
        },
    }
    gate_passed = all(result["passed"] for result in gates.values())
    return {
        "schema_version": RELEASED_WARP_SCORE_SCHEMA_VERSION,
        "artifact_kind": RELEASED_WARP_SCORE_KIND,
        "protocol_id": protocol["config"]["protocol_id"],
        "protocol_config_sha256": protocol["config_sha256"],
        "prediction_result_sha256": prediction["result_sha256"],
        "prediction_archive_sha256": prediction["prediction_archive"]["sha256"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "episode_results": episode_results,
        "panel": panel,
        "transfer_gate": {
            "passed": gate_passed,
            "criteria": gates,
            "maximum_repeat_rollout_rmse_m": maximum_repeat,
            "decision": (
                "justify_separate_fresh_object_preregistration"
                if gate_passed
                else "stop_released_particle_readout_route"
            ),
        },
        "unavailable_metrics": {
            "nees": "no predictive covariance in this audit",
            "coverage": "no predictive covariance in this audit",
            "archived_long_rollout": (
                "diagnostic origin is unmatched and no equivalent dense "
                "origin association is sealed"
            ),
        },
        "information_boundary": {
            "prediction_validated_before_future_open": True,
            "future_particles_opened_for_scoring_only": True,
            "candidate_reselection_from_scores": False,
            "source_only": True,
            "held_v8_access": False,
        },
        "claim_boundary": protocol["config"]["claim_boundary"]["description"],
    }


def write_released_warp_score_artifact(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    artifact["result_sha256"] = _artifact_sha256(artifact)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def validate_released_warp_score_artifact(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_released_warp_readout_protocol(protocol)
    _require(
        payload.get("schema_version") == RELEASED_WARP_SCORE_SCHEMA_VERSION,
        "unsupported released-Warp score schema",
    )
    _require(
        payload.get("artifact_kind") == RELEASED_WARP_SCORE_KIND,
        "unexpected released-Warp score kind",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "released-Warp score checksum mismatch",
    )
    _require(
        payload.get("protocol_config_sha256") == protocol["config_sha256"],
        "score uses a different protocol lock",
    )
    _require(
        payload.get("information_boundary", {}).get("held_v8_access") is False,
        "score crossed the held-v8 boundary",
    )
    return {
        "passed": True,
        "transfer_gate_passed": bool(payload["transfer_gate"]["passed"]),
        "result_sha256": payload["result_sha256"],
    }


__all__ = [
    "PolylineMaterialAssociation",
    "associate_particles_to_polyline",
    "build_matched_origin_warp_cases",
    "lift_sparse_polyline_to_particles",
    "load_released_warp_prediction_artifact",
    "minimum_rotation_matrix",
    "run_released_warp_readout_predictions",
    "score_released_warp_readout_predictions",
    "symmetric_chamfer_distance_m",
    "validate_released_warp_prediction_artifact",
    "validate_released_warp_score_artifact",
    "write_released_warp_prediction_artifact",
    "write_released_warp_score_artifact",
]
