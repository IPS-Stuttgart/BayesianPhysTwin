from __future__ import annotations

import os
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_dataset_containment import (
    validate_aligned_episode,
    validate_regular_file_nofollow,
)


OBJECT = "083-blanket-cloth"
EPISODE = 0


def _episode(root: Path) -> Path:
    episode = root / OBJECT / "episode_0000"
    episode.mkdir(parents=True)
    return episode


def test_exact_aligned_layout_accepts_only_regular_canonical_children(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aligned"
    episode = _episode(root)
    video = episode / "camera-00" / "undistorted.mp4"
    video.parent.mkdir()
    video.write_bytes(b"rgb")

    layout = validate_aligned_episode(
        episode,
        object_id=OBJECT,
        episode_id=EPISODE,
        aligned_root=root,
    )

    assert layout.aligned_root == root
    assert layout.file("camera-00", "undistorted.mp4", label="camera") == video


def test_object_symlink_cannot_escape_aligned_root(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    root.mkdir()
    outside = _episode(tmp_path / "outside")
    (root / OBJECT).symlink_to(outside.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="object directory.*symlink"):
        validate_aligned_episode(
            root / OBJECT / "episode_0000",
            object_id=OBJECT,
            episode_id=EPISODE,
            aligned_root=root,
        )


def test_episode_symlink_cannot_escape_object_directory(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    object_dir = root / OBJECT
    object_dir.mkdir(parents=True)
    outside = _episode(tmp_path / "outside")
    (object_dir / "episode_0000").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="episode directory.*symlink"):
        validate_aligned_episode(
            object_dir / "episode_0000",
            object_id=OBJECT,
            episode_id=EPISODE,
            aligned_root=root,
        )


@pytest.mark.parametrize("linked_component", ["camera", "robot"])
def test_camera_or_robot_directory_symlink_is_rejected(
    tmp_path: Path, linked_component: str
) -> None:
    root = tmp_path / "aligned"
    episode = _episode(root)
    layout = validate_aligned_episode(
        episode,
        object_id=OBJECT,
        episode_id=EPISODE,
        aligned_root=root,
    )
    outside = tmp_path / f"outside-{linked_component}"
    outside.mkdir()
    filename = "undistorted.mp4" if linked_component == "camera" else "robot.npz"
    (outside / filename).write_bytes(b"source")
    directory = "camera-00" if linked_component == "camera" else "robot"
    (episode / directory).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="directory.*symlink"):
        layout.file(directory, filename, label=linked_component)


def test_dataset_file_symlink_is_rejected_before_read(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    episode = _episode(root)
    camera = episode / "camera-00"
    camera.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"future payload")
    linked = camera / "undistorted.mp4"
    linked.symlink_to(outside)
    layout = validate_aligned_episode(
        episode,
        object_id=OBJECT,
        episode_id=EPISODE,
        aligned_root=root,
    )

    with pytest.raises(ValueError, match="file is a symlink"):
        layout.file("camera-00", "undistorted.mp4", label="camera video")


def test_optional_file_does_not_hide_dangling_camera_symlink(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    episode = _episode(root)
    (episode / "camera-00").symlink_to(
        tmp_path / "missing-camera", target_is_directory=True
    )
    layout = validate_aligned_episode(
        episode,
        object_id=OBJECT,
        episode_id=EPISODE,
        aligned_root=root,
    )

    with pytest.raises(ValueError, match="directory is a symlink"):
        layout.optional_file(
            "camera-00", "undistorted.mp4", label="optional camera video"
        )


def test_bound_file_rejects_symlinked_ancestor_and_noncanonical_spelling(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "input.bin"
    source.write_bytes(b"payload")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="ancestor is a symlink"):
        validate_regular_file_nofollow(
            linked_parent / "input.bin", label="bound dataset input"
        )
    with pytest.raises(ValueError, match="parent traversal"):
        validate_regular_file_nofollow(
            os.fspath(real / ".." / "real" / "input.bin"),
            label="bound dataset input",
        )


def test_wrong_object_or_episode_is_rejected_lexically(tmp_path: Path) -> None:
    root = tmp_path / "aligned"
    episode = _episode(root)

    with pytest.raises(ValueError, match="exact authorized"):
        validate_aligned_episode(
            episode,
            object_id="092-squirrel",
            episode_id=EPISODE,
            aligned_root=root,
        )
    with pytest.raises(ValueError, match="exact authorized"):
        validate_aligned_episode(
            episode,
            object_id=OBJECT,
            episode_id=1,
            aligned_root=root,
        )
