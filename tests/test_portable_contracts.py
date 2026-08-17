from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import bayesian_phystwin._portable_contracts as portable


def _temporary_publication_files(destination: Path) -> list[Path]:
    return list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_json_uses_canonical_pretty_encoding(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "artifact.json"

    portable.write_atomic_json(
        {"z": [3, 2, 1], "a": {"value": 4}},
        destination,
        overwrite=False,
    )

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "value": 4\n  },\n  "z": [\n    3,\n    2,\n    1\n  ]\n}\n'
    )
    assert _temporary_publication_files(destination) == []


def test_atomic_json_nonreplacement_preserves_existing_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("existing-bytes\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        portable.write_atomic_json(
            {"replacement": True},
            destination,
            overwrite=False,
        )

    assert destination.read_text(encoding="utf-8") == "existing-bytes\n"
    assert _temporary_publication_files(destination) == []


def test_atomic_json_nonreplacement_is_safe_under_concurrent_writers(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    writer_count = 16
    barrier = threading.Barrier(writer_count)
    payloads = [
        {"writer": index, "values": [index, index + 1, index + 2]}
        for index in range(writer_count)
    ]

    def publish(index: int) -> bool:
        barrier.wait()
        try:
            portable.write_atomic_json(
                payloads[index],
                destination,
                overwrite=False,
            )
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        outcomes = list(executor.map(publish, range(writer_count)))

    assert sum(outcomes) == 1
    assert json.loads(destination.read_text(encoding="utf-8")) in payloads
    assert _temporary_publication_files(destination) == []


def test_atomic_json_explicit_overwrite_replaces_complete_artifact(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    portable.write_atomic_json({"revision": 1}, destination, overwrite=False)

    portable.write_atomic_json({"revision": 2}, destination, overwrite=True)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"revision": 2}
    assert _temporary_publication_files(destination) == []


def test_atomic_json_cleans_temporary_after_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"

    def reject_link(_source: object, _destination: object) -> None:
        raise PermissionError("hard-link publication denied")

    monkeypatch.setattr(portable.os, "link", reject_link)
    with pytest.raises(PermissionError, match="hard-link publication denied"):
        portable.write_atomic_json({"value": 1}, destination, overwrite=False)

    assert not destination.exists()
    assert _temporary_publication_files(destination) == []


def test_atomic_json_cleans_temporary_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"

    def reject_replace(_source: object, _destination: object) -> None:
        raise PermissionError("replacement publication denied")

    monkeypatch.setattr(portable.os, "replace", reject_replace)
    with pytest.raises(PermissionError, match="replacement publication denied"):
        portable.write_atomic_json({"value": 1}, destination, overwrite=True)

    assert not destination.exists()
    assert _temporary_publication_files(destination) == []


def test_directory_fsync_is_best_effort_when_open_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_open(_path: object, _flags: int) -> int:
        raise PermissionError("directory descriptors unsupported")

    monkeypatch.setattr(portable.os, "open", reject_open)
    portable._fsync_directory(tmp_path)


def test_directory_fsync_closes_descriptor_after_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []

    def reject_fsync(_descriptor: int) -> None:
        raise OSError("directory fsync unsupported")

    monkeypatch.setattr(portable.os, "open", lambda _path, _flags: 41)
    monkeypatch.setattr(portable.os, "fsync", reject_fsync)
    monkeypatch.setattr(portable.os, "close", closed.append)

    portable._fsync_directory(tmp_path)

    assert closed == [41]
