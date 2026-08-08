#!/usr/bin/env python3
"""Replace path-reopened prepared inventory reads with stable snapshots."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/science/inventory_deform360_calibration_prepared_source.py"


def _replace(old: str, new: str, *, expected_count: int = 1) -> None:
    text = TARGET.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    TARGET.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        "import argparse\nimport hashlib\nimport json\nimport sys\n",
        "import argparse\nimport hashlib\nimport json\nimport os\nimport stat\nimport sys\nimport tempfile\n",
    )
    _replace("from typing import Any\n", "from typing import Any, BinaryIO\n")
    _replace("    load_strict_json_object,\n", "")
    _replace(
        """def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
""",
        """_READ_CHUNK_SIZE_BYTES = 1024 * 1024
_SNAPSHOT_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_file(
    path: Path,
    *,
    root: Path,
    name: str,
    sink: BinaryIO | None = None,
) -> dict[str, object]:
    # Hash one descriptor-stable regular file and optionally retain its bytes.
    ordinary = _ordinary_file(path, root=root, name=name)
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    try:
        descriptor = os.open(ordinary, flags)
    except OSError as error:
        raise ValueError(f"cannot open prepared file: {name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"prepared path is not a regular file: {name}")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(descriptor, _READ_CHUNK_SIZE_BYTES)
            if not block:
                break
            digest.update(block)
            if sink is not None:
                sink.write(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read prepared file: {name}") from error
    finally:
        os.close(descriptor)

    if _file_identity(before) != _file_identity(after) or byte_count != after.st_size:
        raise ValueError(f"prepared file changed while reading: {name}")
    if sink is not None:
        sink.flush()
        sink.seek(0)
    return {
        "path": _portable_path(ordinary, root=root),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }
""",
    )
    _replace(
        """def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    ordinary = _ordinary_file(path, root=root, name=_portable_path(path, root=root))
    return {
        "path": _portable_path(ordinary, root=root),
        "sha256": _sha256_file(ordinary),
        "byte_count": ordinary.stat().st_size,
    }
""",
        """def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    return _read_stable_file(
        path,
        root=root,
        name=_portable_path(path, root=root),
    )
""",
    )
    _replace(
        """def _npy_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    record = _file_record(path, root=root)
    _require(record["sha256"] == expected_sha256, f"prepared array changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        contract = _numeric_contract(array, name=str(path))
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {**record, **contract}
""",
        """def _npy_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    with tempfile.SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    ) as snapshot:
        record = _read_stable_file(
            path,
            root=root,
            name=_portable_path(path, root=root),
            sink=snapshot,
        )
        _require(
            record["sha256"] == expected_sha256,
            f"prepared array changed: {path}",
        )
        try:
            array = np.load(snapshot, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"prepared array is invalid: {path}") from error
        if not isinstance(array, np.ndarray):
            raise ValueError(f"prepared array is not an NPY payload: {path}")
        contract = _numeric_contract(array, name=str(path))
    return {**record, **contract}
""",
    )
    _replace(
        """def _npz_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    record = _file_record(path, root=root)
    _require(record["sha256"] == expected_sha256, f"prepared archive changed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        names = tuple(sorted(stored.files))
        missing = sorted(_REQUIRED_ROBOT_ARRAYS - set(names))
        if missing:
            raise ValueError(f"robot archive is missing arrays: {missing}")
        arrays = {
            name: _numeric_contract(stored[name], name=f"{path}:{name}")
            for name in names
        }
    return {**record, "arrays": arrays}
""",
        """def _npz_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    with tempfile.SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    ) as snapshot:
        record = _read_stable_file(
            path,
            root=root,
            name=_portable_path(path, root=root),
            sink=snapshot,
        )
        _require(
            record["sha256"] == expected_sha256,
            f"prepared archive changed: {path}",
        )
        try:
            with np.load(snapshot, allow_pickle=False) as stored:
                names = tuple(sorted(stored.files))
                missing = sorted(_REQUIRED_ROBOT_ARRAYS - set(names))
                if missing:
                    raise ValueError(f"robot archive is missing arrays: {missing}")
                arrays = {
                    name: _numeric_contract(stored[name], name=f"{path}:{name}")
                    for name in names
                }
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(
                "robot archive is missing arrays"
            ):
                raise
            raise ValueError(f"prepared archive is invalid: {path}") from error
    return {**record, "arrays": arrays}
""",
    )
    _replace(
        """def _load_camera_metadata(path: Path) -> Mapping[str, Any]:
    metadata = load_strict_json_object(path, label="prepared camera metadata")
    if metadata.get("schema") != "deform360.camera-alignment/v1":
        raise ValueError(f"prepared camera metadata schema changed: {path}")
    return metadata
""",
        """def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate prepared camera metadata key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite prepared camera metadata value: {value}")


def _load_camera_metadata(stream: BinaryIO, *, path: Path) -> Mapping[str, Any]:
    try:
        metadata = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"prepared camera metadata is invalid: {path}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"prepared camera metadata must be an object: {path}")
    if metadata.get("schema") != "deform360.camera-alignment/v1":
        raise ValueError(f"prepared camera metadata schema changed: {path}")
    return metadata
""",
    )
    _replace(
        """    metadata_file = _file_record(camera_dir / "metadata.json", root=root)
    metadata = _load_camera_metadata(camera_dir / "metadata.json")
""",
        """    metadata_path = camera_dir / "metadata.json"
    with tempfile.SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    ) as snapshot:
        metadata_file = _read_stable_file(
            metadata_path,
            root=root,
            name=_portable_path(metadata_path, root=root),
            sink=snapshot,
        )
        metadata = _load_camera_metadata(snapshot, path=metadata_path)
""",
    )


if __name__ == "__main__":
    main()
