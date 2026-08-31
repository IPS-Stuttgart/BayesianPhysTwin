from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match in {path}, observed {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


RUN = "experiments/deform360_real_v1/run.py"
TEST = "tests/test_deform360_real_v1.py"

replace_once(
    RUN,
    """import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
""",
    """import re
import shutil
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
""",
)

replace_once(
    RUN,
    """    named_npz = 0
    named_pcd_dirs = 0
    named_tactile = 0
""",
    """    named_npz = 0
    named_pcd_dirs = 0
    named_pcd_archives = 0
    named_tactile = 0
""",
)

replace_once(
    RUN,
    """            lowered = name.lower()
            if lowered.endswith(".npz"):
                named_npz += 1
""",
    """            lowered = name.lower()
            if lowered == "pcd_clean.tar":
                named_pcd_archives += 1
                pcd.append(Carrier("pcd_clean_tar", object_id, path))
            elif lowered.endswith(".npz"):
                named_npz += 1
""",
)

replace_once(
    RUN,
    """        "named_npz_files": named_npz,
        "named_pcd_clean_directories": named_pcd_dirs,
        "named_tactile_data_files": named_tactile,
        "candidate_counts": {
            "pcd_clean": len(pcd),
            "trajectory_npz": len(fixed),
            "tactile": len(tactile),
        },
""",
    """        "named_npz_files": named_npz,
        "named_pcd_clean_directories": named_pcd_dirs,
        "named_pcd_clean_archives": named_pcd_archives,
        "named_tactile_data_files": named_tactile,
        "candidate_counts": {
            "pcd_clean": sum(carrier.kind == "pcd_clean" for carrier in pcd),
            "pcd_clean_tar": sum(carrier.kind == "pcd_clean_tar" for carrier in pcd),
            "trajectory_npz": len(fixed),
            "tactile": len(tactile),
        },
""",
)

replace_once(
    RUN,
    """def load_pcd_sequence(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
""",
    """def _load_cloud_bytes(payload: bytes, max_points: int) -> np.ndarray:
    with np.load(BytesIO(payload), allow_pickle=False) as stored:
        if "pts" in stored.files:
            value = np.asarray(stored["pts"], dtype=np.float64)
        else:
            candidate = next(
                (
                    np.asarray(stored[key], dtype=np.float64)
                    for key in stored.files
                    if np.asarray(stored[key]).ndim == 2
                    and np.asarray(stored[key]).shape[1] == 3
                    and np.asarray(stored[key]).dtype.kind in "iuf"
                ),
                None,
            )
            if candidate is None:
                raise ValueError("point-cloud frame has no numeric (N,3) array")
            value = candidate
    value = value[np.all(np.isfinite(value), axis=1)]
    if len(value) < 4:
        raise ValueError("point-cloud frame has fewer than four finite points")
    return value[_indices(len(value), max_points)]


def load_pcd_tar_sequence(
    carrier: Carrier,
    profile: Profile,
    root: Path,
) -> SequenceData:
    before = carrier.path.stat()
    indexed_members: list[tuple[int, str, tarfile.TarInfo]] = []
    clouds: list[np.ndarray] = []
    selected_names: list[str] = []
    with tarfile.open(carrier.path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            match = FRAME_PATTERN.fullmatch(Path(member.name).name)
            if match is None:
                continue
            indexed_members.append((int(match.group(1)), member.name, member))
        indexed_members.sort(key=lambda item: (item[0], item[1]))
        frame_ids = [item[0] for item in indexed_members]
        if len(frame_ids) < 4:
            raise ValueError("pcd_clean.tar contains fewer than four frames")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("pcd_clean.tar contains duplicate frame identities")
        for _, name, member in indexed_members[: profile.max_frames]:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"failed to read point-cloud member: {name}")
            clouds.append(_load_cloud_bytes(stream.read(), profile.max_points))
            selected_names.append(name)
    after = carrier.path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("pcd_clean.tar changed while being read")

    cloud_tuple = tuple(clouds)
    centroids = np.asarray(
        [np.mean(cloud, axis=0) for cloud in cloud_tuple],
        dtype=np.float64,
    )
    return SequenceData(
        values=centroids[:, None, :],
        valid=np.ones((len(centroids), 1), dtype=bool),
        representation="pcd_clean_centroid_3d",
        unit="m",
        primary_metric="centroid_error_mm",
        metadata={
            "relative_path": carrier.path.relative_to(root).as_posix(),
            "source_frame_count": len(indexed_members),
            "evaluated_frame_count": len(cloud_tuple),
            "points_per_frame": [len(cloud) for cloud in cloud_tuple],
            "files": [file_identity(carrier.path)],
            "selected_archive_members": selected_names,
            "identity_scope": (
                "archive identity plus selected member names; point payloads "
                "remain runner-local"
            ),
        },
        clouds=cloud_tuple,
    )


def load_pcd_sequence(carrier: Carrier, profile: Profile, root: Path) -> SequenceData:
""",
)

replace_once(
    RUN,
    """    if carrier.kind == "pcd_clean":
        return load_pcd_sequence(carrier, profile, root)
    if carrier.kind == "tactile":
""",
    """    if carrier.kind == "pcd_clean":
        return load_pcd_sequence(carrier, profile, root)
    if carrier.kind == "pcd_clean_tar":
        return load_pcd_tar_sequence(carrier, profile, root)
    if carrier.kind == "tactile":
""",
)

replace_once(
    TEST,
    """import importlib.util
import json
import sys
from pathlib import Path
""",
    """import importlib.util
import json
import sys
import tarfile
from io import BytesIO
from pathlib import Path
""",
)

replace_once(
    TEST,
    """def test_headerless_tactile_fallback_is_real_measurement_carrier(
""",
    """def test_official_pcd_clean_tar_sequence_is_scored(tmp_path: Path) -> None:
    root = tmp_path / "deform360"
    episode = root / "processed-repository" / "001-rope" / "episode_0000"
    episode.mkdir(parents=True)
    archive_path = episode / "pcd_clean.tar"
    base = moving_points(frames=1, points=20)[0]
    with tarfile.open(archive_path, mode="w") as archive:
        for frame in range(24):
            points = base + np.array([frame * 0.001, 0.0, 0.0])
            buffer = BytesIO()
            np.savez_compressed(buffer, pts=points)
            payload = buffer.getvalue()
            member = tarfile.TarInfo(name=f"pcd_clean/{frame:06d}.npz")
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))

    result = module.run(
        data_root=root,
        protocol_path=save_protocol(tmp_path, root),
        output_dir=tmp_path / "output",
        profile_name="pilot",
        revision=None,
    )

    case = result["cases"][0]
    assert case["kind"] == "pcd_clean_tar"
    assert case["representation"] == "pcd_clean_centroid_3d"
    assert case["metrics"]["last_residual_chamfer_mm"] < 1e-8
    assert case["metrics"]["bayesian_chamfer_mm"] < 1e-8
    inventory = result["selection"]["inventory"]
    assert inventory["named_pcd_clean_archives"] == 1
    assert inventory["candidate_counts"]["pcd_clean_tar"] == 1


def test_headerless_tactile_fallback_is_real_measurement_carrier(
""",
)
