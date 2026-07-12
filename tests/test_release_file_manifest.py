import json
from pathlib import Path

from scripts.release.capture_file_manifest import (
    capture_manifest,
    verify_manifest,
)
from scripts.release.verify_result_bundle import verify_bundle


def test_manifest_capture_and_source_archive_verification(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    archive = tmp_path / "archive.bin"
    source.write_bytes(b"causal4d milestone")
    archive.write_bytes(source.read_bytes())
    specification = {
        "schema_version": 1,
        "milestone": "unit",
        "captured_at": "2026-07-12T00:00:00+02:00",
        "host": "unit-host",
        "entries": [
            {
                "id": "artifact",
                "category": "result",
                "source_path": str(source),
                "archive_path": str(archive),
            }
        ],
    }
    manifest = capture_manifest(specification)
    assert manifest["entry_count"] == 1
    assert manifest["entries"][0]["archive_verified"]
    assert verify_manifest(manifest, location="source")["passed"]
    assert verify_manifest(manifest, location="archive")["passed"]


def test_manifest_verification_detects_changed_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    manifest = capture_manifest(
        {
            "schema_version": 1,
            "milestone": "unit",
            "captured_at": "2026-07-12T00:00:00+02:00",
            "host": "unit-host",
            "entries": [
                {
                    "id": "artifact",
                    "category": "result",
                    "source_path": str(source),
                }
            ],
        }
    )
    source.write_bytes(b"after")
    result = verify_manifest(manifest, location="source")
    assert not result["passed"]
    assert result["failures"][0]["reason"] == "checksum_mismatch"


def test_capture_cli_writes_stable_json(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable")
    specification = tmp_path / "spec.json"
    specification.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "milestone": "unit",
                "captured_at": "2026-07-12T00:00:00+02:00",
                "host": "unit-host",
                "entries": [
                    {
                        "id": "artifact",
                        "category": "result",
                        "source_path": str(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    from scripts.release.capture_file_manifest import main

    assert main(["capture", str(specification), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["entry_count"] == 1


def test_result_bundle_verifier_uses_emitted_hashes(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = result_dir / "summary.json"
    result.write_text("{}\n", encoding="utf-8")
    from scripts.release.verify_result_bundle import sha256

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark": "unit",
                "artifacts": {
                    "summary.json": {
                        "bytes": result.stat().st_size,
                        "sha256": sha256(result),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert verify_bundle(manifest, result_dir)["passed"]
    result.write_text("changed\n", encoding="utf-8")
    assert not verify_bundle(manifest, result_dir)["passed"]


def test_result_bundle_verifier_accepts_bounded_numeric_drift(tmp_path: Path) -> None:
    expected_dir = tmp_path / "expected"
    result_dir = tmp_path / "result"
    expected_dir.mkdir()
    result_dir.mkdir()
    expected = expected_dir / "summary.json"
    actual = result_dir / "summary.json"
    expected.write_text('{"metric": 0.123456789}\n', encoding="utf-8")
    actual.write_text('{"metric": 0.1234567890001}\n', encoding="utf-8")
    from scripts.release.verify_result_bundle import sha256

    manifest = expected_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark": "unit",
                "artifacts": {
                    "summary.json": {
                        "bytes": expected.stat().st_size,
                        "sha256": sha256(expected),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    strict = verify_bundle(manifest, result_dir)
    tolerant = verify_bundle(manifest, result_dir, numeric_atol=1e-12)
    assert not strict["passed"]
    assert tolerant["passed"]
    assert tolerant["tolerance_matches"] == ["summary.json"]


def test_result_bundle_verifier_rejects_structural_drift(tmp_path: Path) -> None:
    expected_dir = tmp_path / "expected"
    result_dir = tmp_path / "result"
    expected_dir.mkdir()
    result_dir.mkdir()
    expected = expected_dir / "summary.json"
    actual = result_dir / "summary.json"
    expected.write_text('{"metric": 1.0, "status": "pass"}\n', encoding="utf-8")
    actual.write_text('{"metric": 1.0, "status": "fail"}\n', encoding="utf-8")
    from scripts.release.verify_result_bundle import sha256

    manifest = expected_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "benchmark": "unit",
                "artifacts": {
                    "summary.json": {
                        "bytes": expected.stat().st_size,
                        "sha256": sha256(expected),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = verify_bundle(manifest, result_dir, numeric_atol=1e-12)
    assert not result["passed"]
    assert "status" in result["failures"][0]["semantic_difference"]
