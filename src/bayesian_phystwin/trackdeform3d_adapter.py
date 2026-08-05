"""Target-free admission helpers for the public TrackDeform3D release.

The adapter inspects archive headers and source hashes without decoding RGB-D,
mask, pose, or future keypoint values.  Evaluator trajectories are intentionally
handled by a later, separately sealed stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy_format

TrackDeform3DObjectKind = Literal["dlo", "bdlo", "fabric", "cloth"]

_MASK_MEMBER_BY_KIND: dict[TrackDeform3DObjectKind, tuple[str, str]] = {
    "dlo": ("masks/masks.npz", "masks"),
    "bdlo": ("masks/masks.npz", "masks"),
    "fabric": ("fg_mask.npz", "fg_mask"),
    "cloth": ("fg_masks/masks.npz", "masks"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NpzMemberHeader:
    """One NPY member header read without materializing its payload."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class TrackDeform3DChunkAdmission:
    """Outcome-blind description of one public sample chunk."""

    schema_version: int
    object_kind: TrackDeform3DObjectKind
    chunk_name: str
    frame_count: int
    image_height: int
    image_width: int
    rgbd_sha256: str
    masks_sha256: str
    left_arm_poses_sha256: str
    right_arm_poses_sha256: str
    calibration_sha256: str
    mask_relative_path: str
    rgbd_members: tuple[NpzMemberHeader, ...]
    mask_members: tuple[NpzMemberHeader, ...]
    left_pose_members: tuple[NpzMemberHeader, ...]
    right_pose_members: tuple[NpzMemberHeader, ...]
    calibration_members: tuple[NpzMemberHeader, ...]
    information_boundary: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "rgbd_members",
            "mask_members",
            "left_pose_members",
            "right_pose_members",
            "calibration_members",
        ):
            payload[key] = [member.to_dict() for member in getattr(self, key)]
        return payload


def _read_npz_headers(path: Path) -> tuple[NpzMemberHeader, ...]:
    _require(path.is_file(), f"missing archive: {path}")
    headers: list[NpzMemberHeader] = []
    with ZipFile(path) as archive:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if member.filename.endswith(".npy")
            ),
            key=lambda member: member.filename,
        )
        _require(bool(members), f"archive contains no NPY members: {path}")
        for member in members:
            with archive.open(member) as stream:
                version = npy_format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(
                        stream
                    )
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(
                        stream
                    )
                else:
                    raise ValueError(
                        f"unsupported NPY version {version} in {path}:{member.filename}"
                    )
            headers.append(
                NpzMemberHeader(
                    name=member.filename.removesuffix(".npy"),
                    shape=tuple(int(value) for value in shape),
                    dtype=np.dtype(dtype).str,
                    fortran_order=bool(fortran_order),
                )
            )
    return tuple(headers)


def _member_map(
    members: tuple[NpzMemberHeader, ...],
) -> dict[str, NpzMemberHeader]:
    result = {member.name: member for member in members}
    _require(len(result) == len(members), "duplicate NPZ member names")
    return result


def _validate_pose_members(members: tuple[NpzMemberHeader, ...], *, name: str) -> int:
    indexed: list[tuple[int, NpzMemberHeader]] = []
    for member in members:
        _require(member.name.startswith("arr_"), f"{name} member name changed")
        suffix = member.name.removeprefix("arr_")
        _require(suffix.isdigit(), f"{name} pose index is not numeric")
        _require(member.shape == (7,), f"{name} pose shape changed")
        _require(np.dtype(member.dtype).kind == "f", f"{name} pose dtype changed")
        indexed.append((int(suffix), member))
    indexed.sort(key=lambda item: item[0])
    _require(
        [index for index, _ in indexed] == list(range(len(indexed))),
        f"{name} pose indices are not contiguous",
    )
    return len(indexed)


def inspect_trackdeform3d_chunk(
    chunk_dir: str | Path,
    calibration_path: str | Path,
    *,
    object_kind: TrackDeform3DObjectKind,
) -> TrackDeform3DChunkAdmission:
    """Validate one TrackDeform3D chunk without decoding observation values."""

    _require(object_kind in _MASK_MEMBER_BY_KIND, "unsupported object kind")
    chunk = Path(chunk_dir)
    calibration = Path(calibration_path)
    _require(chunk.is_dir(), f"missing chunk directory: {chunk}")
    mask_relative_path, mask_key = _MASK_MEMBER_BY_KIND[object_kind]
    rgbd_path = chunk / "rgbd.npz"
    mask_path = chunk / mask_relative_path
    left_path = chunk / "left_arm_poses.npz"
    right_path = chunk / "right_arm_poses.npz"

    rgbd_members = _read_npz_headers(rgbd_path)
    mask_members = _read_npz_headers(mask_path)
    left_members = _read_npz_headers(left_path)
    right_members = _read_npz_headers(right_path)
    calibration_members = _read_npz_headers(calibration)

    rgbd = _member_map(rgbd_members)
    _require(set(rgbd) == {"color", "depth"}, "RGB-D member set changed")
    color = rgbd["color"]
    depth = rgbd["depth"]
    _require(len(color.shape) == 4 and color.shape[-1] == 3, "color shape changed")
    _require(len(depth.shape) == 3, "depth shape changed")
    _require(color.shape[:3] == depth.shape, "color and depth shapes differ")
    _require(np.dtype(color.dtype) == np.dtype("uint8"), "color dtype changed")
    _require(np.dtype(depth.dtype) == np.dtype("uint16"), "depth dtype changed")

    masks = _member_map(mask_members)
    _require(set(masks) == {mask_key}, "mask member set changed")
    mask = masks[mask_key]
    _require(mask.shape == depth.shape, "mask and depth shapes differ")
    _require(np.dtype(mask.dtype).kind in {"b", "i", "u"}, "mask dtype changed")

    left_count = _validate_pose_members(left_members, name="left")
    right_count = _validate_pose_members(right_members, name="right")
    frame_count = int(color.shape[0])
    _require(left_count == frame_count, "left pose frame count changed")
    _require(right_count == frame_count, "right pose frame count changed")
    _require(frame_count >= 2, "chunk is too short")

    calibration_map = _member_map(calibration_members)
    _require(
        set(calibration_map) == {"K", "T_left_base2cam", "T_right_base2cam"},
        "calibration member set changed",
    )
    _require(calibration_map["K"].shape == (3, 3), "intrinsics shape changed")
    _require(
        calibration_map["T_left_base2cam"].shape == (4, 4),
        "left transform shape changed",
    )
    _require(
        calibration_map["T_right_base2cam"].shape == (4, 4),
        "right transform shape changed",
    )
    _require(
        all(np.dtype(member.dtype).kind == "f" for member in calibration_members),
        "calibration dtype changed",
    )

    return TrackDeform3DChunkAdmission(
        schema_version=1,
        object_kind=object_kind,
        chunk_name=chunk.name,
        frame_count=frame_count,
        image_height=int(color.shape[1]),
        image_width=int(color.shape[2]),
        rgbd_sha256=_file_sha256(rgbd_path),
        masks_sha256=_file_sha256(mask_path),
        left_arm_poses_sha256=_file_sha256(left_path),
        right_arm_poses_sha256=_file_sha256(right_path),
        calibration_sha256=_file_sha256(calibration),
        mask_relative_path=mask_relative_path,
        rgbd_members=rgbd_members,
        mask_members=mask_members,
        left_pose_members=left_members,
        right_pose_members=right_members,
        calibration_members=calibration_members,
        information_boundary={
            "rgbd_values_decoded": False,
            "mask_values_decoded": False,
            "pose_values_decoded": False,
            "keypoint_trajectories_read": False,
            "future_outcomes_read": False,
        },
    )


def deterministic_observed_identity_ids(
    frame_zero_points_m: np.ndarray,
    observed_count: int,
) -> np.ndarray:
    """Select a spatially spread identity subset from frame-zero geometry only."""

    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)")
    _require(np.all(np.isfinite(points)), "frame-zero points are not finite")
    _require(1 <= observed_count < len(points), "observed count is invalid")

    center = np.mean(points, axis=0)
    distance_to_center = np.linalg.norm(points - center, axis=1)
    first = int(np.argmax(distance_to_center))
    selected = [first]
    minimum_distance = np.linalg.norm(points - points[first], axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < observed_count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(points - points[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    return np.asarray(selected, dtype=np.int64)


__all__ = [
    "NpzMemberHeader",
    "TrackDeform3DChunkAdmission",
    "TrackDeform3DObjectKind",
    "deterministic_observed_identity_ids",
    "inspect_trackdeform3d_chunk",
]
