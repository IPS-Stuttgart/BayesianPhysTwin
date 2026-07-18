import json
import struct
import zlib
from pathlib import Path

import pytest

from scripts.remote.run_phystwin_render_sweep import (
    _complete_output,
    _expected_frames,
    _output_lock,
    _renderer_code_sha256,
    _replace_directory,
    _tree_sha256,
    _write_json_atomic,
)


def _png(width: int, height: int) -> bytes:
    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload))
        )

    rows = b"".join(b"\x00" + b"\x00\x00\x00\xff" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_render_completion_requires_every_expected_frame(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"frame_len": 3}), encoding="utf-8")
    count, expected = _expected_frames(split)
    frames = tmp_path / "output" / "case" / "0"
    frames.mkdir(parents=True)
    for index in range(3):
        (frames / f"{index:05d}.png").write_bytes(_png(8, 6))

    assert count == 3
    assert _complete_output(tmp_path / "output", "case", expected)

    (frames / "00002.png").unlink()
    assert not _complete_output(tmp_path / "output", "case", expected)


def test_render_completion_rejects_corrupt_or_inconsistent_pngs(tmp_path: Path) -> None:
    frames = tmp_path / "output" / "case" / "0"
    frames.mkdir(parents=True)
    (frames / "00000.png").write_bytes(_png(8, 6))
    (frames / "00001.png").touch()
    assert not _complete_output(
        tmp_path / "output", "case", {"00000.png", "00001.png"}
    )

    (frames / "00001.png").write_bytes(_png(7, 6))
    assert not _complete_output(
        tmp_path / "output", "case", {"00000.png", "00001.png"}
    )


def test_tree_hash_binds_relative_names_and_contents(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    (root / "a").write_bytes(b"same")
    first = _tree_sha256(root)

    (root / "a").rename(root / "b")
    assert _tree_sha256(root) != first

    renamed = _tree_sha256(root)
    (root / "b").write_bytes(b"different")
    assert _tree_sha256(root) != renamed


def test_renderer_code_hash_binds_transitive_source(tmp_path: Path) -> None:
    package = tmp_path / "gaussian_splatting" / "scene"
    package.mkdir(parents=True)
    (tmp_path / "gs_render_dynamics.py").write_text("import scene\n")
    dependency = package / "model.py"
    dependency.write_text("VALUE = 1\n")
    first = _renderer_code_sha256(tmp_path)

    dependency.write_text("VALUE = 2\n")

    assert _renderer_code_sha256(tmp_path) != first


def test_atomic_manifest_write_replaces_valid_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json_atomic(manifest, {"status": "in_progress", "cases": {}})
    _write_json_atomic(manifest, {"status": "complete", "cases": {"a": {}}})

    assert json.loads(manifest.read_text()) == {
        "status": "complete",
        "cases": {"a": {}},
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_directory_replace_removes_stale_render_tail(tmp_path: Path) -> None:
    target = tmp_path / "case"
    (target / "0").mkdir(parents=True)
    (target / "0" / "00000.png").write_bytes(b"old")
    (target / "0" / "00001.png").write_bytes(b"stale-tail")
    staged = tmp_path / "staged" / "case"
    (staged / "0").mkdir(parents=True)
    (staged / "0" / "00000.png").write_bytes(b"new")

    _replace_directory(staged, target)

    assert (target / "0" / "00000.png").read_bytes() == b"new"
    assert not (target / "0" / "00001.png").exists()


def test_output_lock_rejects_a_second_sweep(tmp_path: Path) -> None:
    with _output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another render sweep"):
            with _output_lock(tmp_path):
                pass
