"""Canonical, symlink-free access to one aligned Deform360 episode.

The held protocol passes an exact ``aligned/<object>/episode_NNNN`` path to the
Python stages.  Resolving that path before checking its lexical ancestry would
allow an intermediate object or episode symlink to escape the aligned root.
This module keeps the lexical path intact, checks every directory with
``lstat``, and opens regular files once with ``O_NOFOLLOW`` before callers hand
their paths to NumPy, OpenCV, or JSON readers.

The formal operators independently bind the inferred ``aligned_root`` to the
one preregistered dataset root.  These helpers enforce the complementary
in-process guarantee: every dataset input is the exact expected child of that
root and no checked component is a symlink.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _leaf(value: str, *, label: str) -> str:
    _require(bool(value), f"{label} is empty")
    _require(
        value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value,
        f"{label} is not one lexical path component",
    )
    return value


def lexical_absolute_path(path: str | os.PathLike[str], *, label: str) -> Path:
    """Return an absolute lexical path without resolving any component."""

    value = Path(os.fspath(path))
    _require(value.is_absolute(), f"{label} path is not absolute")
    _require(".." not in value.parts, f"{label} path contains parent traversal")
    # ``Path`` normalizes redundant separators and ``.``.  Requiring the
    # normalized spelling prevents a different lexical route to the same file.
    _require(
        os.path.normpath(os.fspath(path)) == os.fspath(value),
        f"{label} path is not lexically normalized",
    )
    return value


def validate_directory_nofollow(
    path: str | os.PathLike[str], *, label: str
) -> Path:
    """Require an existing canonical directory with no symlink component."""

    value = lexical_absolute_path(path, label=label)
    try:
        observed = os.lstat(value)
    except OSError as error:
        raise ValueError(f"{label} directory is missing: {value}") from error
    _require(not stat.S_ISLNK(observed.st_mode), f"{label} directory is a symlink")
    _require(stat.S_ISDIR(observed.st_mode), f"{label} is not a regular directory")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} directory cannot be resolved: {value}") from error
    _require(resolved == value, f"{label} directory or an ancestor is a symlink")
    return value


def validate_regular_file_nofollow(
    path: str | os.PathLike[str],
    *,
    label: str,
    expected_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Require a canonical regular file and probe it with ``O_NOFOLLOW``.

    The descriptor is intentionally not returned because several consumers
    require a filesystem path.  Comparing ``lstat`` and ``fstat`` makes this a
    useful fail-closed guard against direct symlinks and ordinary replacement
    races without pretending to provide an ``openat2``-style lifetime lock.
    """

    value = lexical_absolute_path(path, label=label)
    if expected_path is not None:
        expected = lexical_absolute_path(expected_path, label=f"expected {label}")
        _require(value == expected, f"{label} is outside the exact aligned path")
    try:
        before = os.lstat(value)
    except OSError as error:
        raise ValueError(f"{label} file is missing: {value}") from error
    _require(not stat.S_ISLNK(before.st_mode), f"{label} file is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} file cannot be resolved: {value}") from error
    _require(resolved == value, f"{label} file or an ancestor is a symlink")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(value, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened without following links") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode), f"{label} opened as a non-regular file")
        _require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed during no-follow validation",
        )
    finally:
        os.close(descriptor)
    return value


@dataclass(frozen=True)
class AlignedEpisodeLayout:
    """Validated lexical layout rooted at one exact aligned dataset root."""

    aligned_root: Path
    object_dir: Path
    episode_dir: Path
    object_id: str
    episode_id: int

    def directory(self, *parts: str, label: str) -> Path:
        checked = tuple(_leaf(str(part), label=label) for part in parts)
        current = self.episode_dir
        for part in checked:
            current = validate_directory_nofollow(current / part, label=label)
        return current

    def file(self, *parts: str, label: str) -> Path:
        _require(bool(parts), f"{label} path is empty")
        checked = tuple(_leaf(str(part), label=label) for part in parts)
        current = self.episode_dir
        for part in checked[:-1]:
            current = validate_directory_nofollow(current / part, label=label)
        expected = current / checked[-1]
        return validate_regular_file_nofollow(
            expected,
            label=label,
            expected_path=self.episode_dir.joinpath(*checked),
        )

    def validate_file(
        self,
        path: str | os.PathLike[str],
        *parts: str,
        label: str,
    ) -> Path:
        _require(bool(parts), f"{label} expected path is empty")
        checked = tuple(_leaf(str(part), label=label) for part in parts)
        current = self.episode_dir
        for part in checked[:-1]:
            current = validate_directory_nofollow(current / part, label=label)
        expected = current / checked[-1]
        return validate_regular_file_nofollow(
            path,
            label=label,
            expected_path=expected,
        )

    def optional_file(self, *parts: str, label: str) -> Path | None:
        _require(bool(parts), f"{label} path is empty")
        checked = tuple(_leaf(str(part), label=label) for part in parts)
        current = self.episode_dir
        for part in checked[:-1]:
            candidate = current / part
            if not os.path.lexists(candidate):
                return None
            current = validate_directory_nofollow(candidate, label=label)
        expected = current / checked[-1]
        if not os.path.lexists(expected):
            return None
        return validate_regular_file_nofollow(
            expected,
            label=label,
            expected_path=self.episode_dir.joinpath(*checked),
        )


def validate_aligned_episode(
    episode_dir: str | os.PathLike[str],
    *,
    object_id: str,
    episode_id: int,
    aligned_root: str | os.PathLike[str] | None = None,
) -> AlignedEpisodeLayout:
    """Validate exact ``root/object/episode_NNNN`` lexical ancestry.

    When only an episode path is available, the aligned root is inferred
    lexically (never through ``resolve``).  Formal callers already pin that
    inferred root externally; accepting it explicitly is useful for tests and
    callers that possess the root as a separate capability.
    """

    _require(type(episode_id) is int and episode_id >= 0, "invalid episode id")
    object_name = _leaf(str(object_id), label="object id")
    episode_name = f"episode_{episode_id:04d}"
    episode = lexical_absolute_path(episode_dir, label="aligned episode")
    _require(len(episode.parents) >= 2, "aligned episode has no aligned root")
    inferred_root = episode.parents[1]
    root = lexical_absolute_path(
        inferred_root if aligned_root is None else aligned_root,
        label="aligned root",
    )
    expected_object = root / object_name
    expected_episode = expected_object / episode_name
    _require(
        episode == expected_episode,
        "aligned episode is not the exact authorized object/episode path",
    )
    root = validate_directory_nofollow(root, label="aligned root")
    object_dir = validate_directory_nofollow(
        expected_object, label="aligned object directory"
    )
    episode = validate_directory_nofollow(
        expected_episode, label="aligned episode directory"
    )
    return AlignedEpisodeLayout(
        aligned_root=root,
        object_dir=object_dir,
        episode_dir=episode,
        object_id=object_name,
        episode_id=episode_id,
    )


def layout_from_dataset_file(
    path: str | os.PathLike[str],
    *,
    object_id: str,
    episode_id: int,
    relative_parts: Sequence[str],
    label: str,
) -> AlignedEpisodeLayout:
    """Recover and validate an episode layout from an exact bound file path."""

    checked = tuple(_leaf(str(part), label=label) for part in relative_parts)
    _require(bool(checked), f"{label} relative path is empty")
    value = lexical_absolute_path(path, label=label)
    episode = value
    for _ in checked:
        episode = episode.parent
    layout = validate_aligned_episode(
        episode,
        object_id=object_id,
        episode_id=episode_id,
    )
    layout.validate_file(value, *checked, label=label)
    return layout


__all__ = [
    "AlignedEpisodeLayout",
    "layout_from_dataset_file",
    "lexical_absolute_path",
    "validate_aligned_episode",
    "validate_directory_nofollow",
    "validate_regular_file_nofollow",
]
