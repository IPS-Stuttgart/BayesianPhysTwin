from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/release/bind_release_matrix_contracts.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bind_release_matrix_contracts", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_sources(root: Path) -> None:
    for index, relative in enumerate(tool.CONTRACTS.values()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"contract {index}: {relative}\n", encoding="utf-8")


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    dist = tmp_path / "dist"
    root.mkdir()
    dist.mkdir()
    _write_sources(root)
    sdist = dist / "bayesian_phystwin-0.4.0.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        for relative in tool.CONTRACTS.values():
            _add_bytes(
                archive,
                f"bayesian_phystwin-0.4.0/{relative}",
                (root / relative).read_bytes(),
            )
    sdist_bytes = sdist.read_bytes()
    descriptor: dict[str, Any] = {
        "schema": tool.SCHEMA,
        "schema_version": 1,
        "project_version": "0.4.0",
        "artifacts": {
            "sdist": {
                "path": sdist.name,
                "sha256": hashlib.sha256(sdist_bytes).hexdigest(),
                "size_bytes": len(sdist_bytes),
            }
        },
        "source_contracts": {"existing": {"path": "existing"}},
    }
    evidence = {"evidence_id": _canonical_id(descriptor), **descriptor}
    evidence_path = tmp_path / "base-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return root, dist, evidence_path


def test_binds_exact_source_and_sdist_contracts(tmp_path: Path) -> None:
    root, dist, evidence_path = _fixture(tmp_path)

    evidence = tool.bind_release_matrix_contracts(
        evidence_path,
        dist_dir=dist,
        project_root=root,
    )

    assert set(tool.CONTRACTS) <= set(evidence["source_contracts"])
    assert evidence["source_contracts"]["existing"] == {"path": "existing"}
    descriptor = dict(evidence)
    supplied_id = descriptor.pop("evidence_id")
    assert supplied_id == _canonical_id(descriptor)


def test_rejects_tampered_base_evidence(tmp_path: Path) -> None:
    root, dist, evidence_path = _fixture(tmp_path)
    payload = json.loads(evidence_path.read_text())
    payload["project_version"] = "0.4.1"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.ReleaseMatrixBindingError, match="ID does not match"):
        tool.bind_release_matrix_contracts(
            evidence_path,
            dist_dir=dist,
            project_root=root,
        )


def test_rejects_source_distribution_member_drift(tmp_path: Path) -> None:
    root, dist, evidence_path = _fixture(tmp_path)
    sdist = next(dist.glob("*.tar.gz"))
    with tarfile.open(sdist, mode="w:gz") as archive:
        for relative in tool.CONTRACTS.values():
            data = b"drifted\n" if relative.endswith("release-build.txt") else (
                root / relative
            ).read_bytes()
            _add_bytes(archive, f"bayesian_phystwin-0.4.0/{relative}", data)
    sdist_bytes = sdist.read_bytes()
    payload = json.loads(evidence_path.read_text())
    payload["artifacts"]["sdist"] = {
        "path": sdist.name,
        "sha256": hashlib.sha256(sdist_bytes).hexdigest(),
        "size_bytes": len(sdist_bytes),
    }
    descriptor = dict(payload)
    descriptor.pop("evidence_id")
    payload["evidence_id"] = _canonical_id(descriptor)
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.ReleaseMatrixBindingError, match="member changed"):
        tool.bind_release_matrix_contracts(
            evidence_path,
            dist_dir=dist,
            project_root=root,
        )


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    tool.write_evidence(output, {"value": 1})
    with pytest.raises(tool.ReleaseMatrixBindingError, match="refusing to overwrite"):
        tool.write_evidence(output, {"value": 2})
