"""Contracts for exact isolated Deform360 runtime validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.verify_deform360_runtime_lock import (
    RuntimeLockError,
    load_lock,
    main,
    validate_runtime,
)


def _distribution(site: Path, name: str, version: str) -> None:
    normalized = name.replace("-", "_")
    metadata = site / f"{normalized}-{version}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_runtime_lock_accepts_exact_third_party_and_declared_local_packages(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text("numpy==2.2.6\npytest==9.1.1\n", encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    _distribution(site, "numpy", "2.2.6")
    _distribution(site, "pytest", "9.1.1")
    _distribution(site, "bayesian-phystwin", "0.4.0")

    report = validate_runtime(
        lock_path=lock,
        site=site,
        allowed_local_names={"bayesian-phystwin"},
        require_complete=True,
    )

    assert report["passed"] is True
    assert report["local_packages"] == {"bayesian-phystwin": "0.4.0"}
    assert report["unpinned_packages"] == {}
    assert report["version_mismatches"] == {}
    assert report["missing_locked_packages"] == {}


def test_runtime_lock_reports_unpinned_mismatched_and_missing_packages(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text("numpy==2.2.6\npytest==9.1.1\n", encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    _distribution(site, "numpy", "2.3.0")
    _distribution(site, "foreign", "1.0")

    report = validate_runtime(
        lock_path=lock,
        site=site,
        allowed_local_names=set(),
        require_complete=True,
    )

    assert report["passed"] is False
    assert report["version_mismatches"] == {
        "numpy": {"installed": "2.3.0", "locked": "2.2.6"}
    }
    assert report["unpinned_packages"] == {"foreign": "1.0"}
    assert report["missing_locked_packages"] == {"pytest": "9.1.1"}


def test_runtime_lock_rejects_ranges_and_conflicting_pins(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(RuntimeLockError, match="exact"):
        load_lock(lock)

    lock.write_text("numpy==2.2.6\nNumPy==2.3.0\n", encoding="utf-8")
    with pytest.raises(RuntimeLockError, match="conflicting"):
        load_lock(lock)


def test_runtime_lock_cli_publishes_negative_evidence(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text("numpy==2.2.6\n", encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    _distribution(site, "numpy", "2.3.0")
    output = tmp_path / "validation.json"

    status = main(
        [
            "--lock",
            str(lock),
            "--site",
            str(site),
            "--require-complete",
            "--output",
            str(output),
        ]
    )

    assert status == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["lock_sha256"]
