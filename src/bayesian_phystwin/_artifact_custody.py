"""Shared fail-closed primitives for content-addressed artifact custody."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Final

_CHECKSUM_BLOCK_SIZE: Final = 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one ordinary file."""

    source = ordinary_file(path, name="artifact file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHECKSUM_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def ordinary_file(path: str | Path, *, name: str) -> Path:
    """Resolve one regular file without traversing symlinks."""

    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def ordinary_directory(path: str | Path, *, name: str) -> Path:
    """Resolve one directory without traversing symlinks."""

    source = Path(path).absolute()
    _require(
        source.is_dir()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink directory",
    )
    return source.resolve(strict=True)


def regular_file_roster(root: str | Path) -> frozenset[str]:
    """Return the exact regular-file roster and reject any symlink below root."""

    directory = ordinary_directory(root, name="artifact bundle")
    descendants = tuple(directory.rglob("*"))
    _require(
        not any(path.is_symlink() for path in descendants),
        "artifact bundle must not contain symlinks",
    )
    return frozenset(
        path.relative_to(directory).as_posix() for path in descendants if path.is_file()
    )


def new_staging_directory(output_dir: str | Path) -> tuple[Path, Path]:
    """Create a same-parent staging directory for one no-overwrite publication."""

    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not output.parent.is_symlink()
        and not any(parent.is_symlink() for parent in output.parent.parents),
        "artifact output path must not traverse symlinks",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    ).resolve(strict=True)
    return output, staging


def publish_staging_directory(staging: str | Path, output: str | Path) -> None:
    """Atomically publish a staging directory without replacing an output."""

    staged = ordinary_directory(staging, name="artifact staging directory")
    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    _require(
        staged.parent == target.parent.resolve(strict=True),
        "artifact staging and output directories must share one parent",
    )
    os.replace(staged, target)


def copy_file_exact(source: str | Path, destination: str | Path) -> Path:
    """Copy one regular file once and verify byte identity."""

    original = ordinary_file(source, name="artifact source file")
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not target.parent.is_symlink()
        and not any(parent.is_symlink() for parent in target.parent.parents),
        "artifact destination path must not traverse symlinks",
    )
    with original.open("rb") as source_stream, target.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream, length=_CHECKSUM_BLOCK_SIZE)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    _require(
        original.stat().st_size == target.stat().st_size
        and file_sha256(original) == file_sha256(target),
        "artifact copy is not byte exact",
    )
    return target.resolve(strict=True)


def _relative_file(root: Path, relative_path: str) -> Path:
    if type(relative_path) is not str or not relative_path:
        raise ValueError("checksum path must be a nonempty string")
    pure = PurePosixPath(relative_path)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == relative_path
        and ".." not in pure.parts
        and "." not in pure.parts,
        "checksum path must be a canonical relative POSIX path",
    )
    return ordinary_file(
        root / Path(*pure.parts), name=f"checksum file {relative_path}"
    )


def checksum_manifest_text(root: str | Path, relative_paths: Iterable[str]) -> str:
    """Build a canonical checksum manifest for a declared file roster."""

    directory = ordinary_directory(root, name="checksum root")
    paths = tuple(sorted(relative_paths))
    _require(len(paths) == len(set(paths)), "checksum paths must be unique")
    return "".join(
        f"{file_sha256(_relative_file(directory, path))}  {path}\n" for path in paths
    )


def write_checksum_manifest(
    root: str | Path,
    relative_paths: Iterable[str],
    *,
    filename: str = "SHA256SUMS",
) -> Path:
    """Write one canonical checksum manifest without overwriting."""

    directory = ordinary_directory(root, name="checksum root")
    target = directory / filename
    text = checksum_manifest_text(directory, relative_paths)
    with target.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return target.resolve(strict=True)


def validate_checksum_manifest(
    root: str | Path,
    relative_paths: Iterable[str],
    *,
    filename: str = "SHA256SUMS",
) -> Path:
    """Validate one canonical checksum manifest exactly."""

    directory = ordinary_directory(root, name="checksum root")
    manifest = ordinary_file(directory / filename, name="checksum manifest")
    expected = checksum_manifest_text(directory, relative_paths)
    _require(
        manifest.read_text(encoding="ascii") == expected,
        "artifact checksum manifest changed",
    )
    return manifest


__all__ = [
    "checksum_manifest_text",
    "copy_file_exact",
    "file_sha256",
    "new_staging_directory",
    "ordinary_directory",
    "ordinary_file",
    "publish_staging_directory",
    "regular_file_roster",
    "validate_checksum_manifest",
    "write_checksum_manifest",
]
