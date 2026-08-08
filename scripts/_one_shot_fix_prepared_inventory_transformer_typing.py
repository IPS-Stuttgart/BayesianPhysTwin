#!/usr/bin/env python3
"""Widen the generated stable-snapshot sink contract before finalization."""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).with_name(
    "_one_shot_harden_prepared_inventory_snapshot.py"
)


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
        '_replace("from typing import Any\\n", "from typing import Any, BinaryIO\\n")',
        '_replace("from typing import Any\\n", "from typing import Any, Protocol\\n")',
    )
    _replace(
        """_SNAPSHOT_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


def _file_identity""",
        """_SNAPSHOT_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


class _BinarySink(Protocol):
    def write(self, data: bytes, /) -> int: ...

    def flush(self) -> None: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...


def _file_identity""",
    )
    _replace("sink: BinaryIO | None = None", "sink: _BinarySink | None = None")
    _replace(
        """def _load_camera_metadata(stream: BinaryIO, *, path: Path) -> Mapping[str, Any]:
    try:
        metadata = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"prepared camera metadata is invalid: {path}") from error
""",
        """def _load_camera_metadata(payload: bytes, *, path: Path) -> Mapping[str, Any]:
    try:
        metadata = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"prepared camera metadata is invalid: {path}") from error
""",
    )
    _replace(
        "metadata = _load_camera_metadata(snapshot, path=metadata_path)",
        "metadata = _load_camera_metadata(snapshot.read(), path=metadata_path)",
    )


if __name__ == "__main__":
    main()
