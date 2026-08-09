from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci/inventory_deform360_v4_results.py"


def module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "inventory_deform360_v4_results", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_inventory_is_results_only_bounded_and_content_addressed(
    tmp_path: Path,
) -> None:
    inventory_module = module()
    results = tmp_path / "results"
    raw = tmp_path / "raw"
    adaptive = tmp_path / "adaptive"
    results.mkdir()
    raw.mkdir()
    adaptive.mkdir()

    descriptor = results / "run" / "tree-sparse-observation-manifest.json"
    write_json(
        descriptor,
        {
            "schema": "prob4d.claim-bearing-tree-sparse-observation-envelope",
            "schema_version": 1,
            "observation_artifact_id": "1" * 64,
            "linearization_artifact_id": "2" * 64,
            "object_id": "object-a",
            "status": "complete",
        },
    )
    binary = results / "run" / "physical-linearization.npz"
    binary.write_bytes(b"not-opened-as-an-archive")
    ignored = results / "run" / "notes.txt"
    ignored.write_text("not a scientific descriptor", encoding="utf-8")
    symlink = results / "run" / "raw-link"
    symlink.symlink_to(raw, target_is_directory=True)

    first = inventory_module.inventory_results(
        results,
        forbidden_roots=(raw, adaptive),
        maximum_depth=5,
        maximum_entries=100,
        maximum_candidates=20,
        maximum_json_bytes=1024 * 1024,
    )
    second = inventory_module.inventory_results(
        results,
        forbidden_roots=(raw, adaptive),
        maximum_depth=5,
        maximum_entries=100,
        maximum_candidates=20,
        maximum_json_bytes=1024 * 1024,
    )

    assert first == second
    assert first["inventory_id"] == second["inventory_id"]
    assert first["counts"] == {
        "entry_count": 5,
        "directory_count": 2,
        "regular_file_count": 3,
        "symlink_count": 1,
        "other_count": 0,
        "candidate_count": 2,
    }
    assert [row["path"] for row in first["candidates"]] == [
        "run/physical-linearization.npz",
        "run/tree-sparse-observation-manifest.json",
    ]
    json_record = first["candidates"][1]
    assert json_record["json_status"] == "parsed"
    assert json_record["selected_fields"]["object_id"] == "object-a"
    assert first["candidate_schema_counts"] == {
        "prob4d.claim-bearing-tree-sparse-observation-envelope": 1
    }
    assert first["information_boundary"]["binary_scientific_payloads_loaded"] is False
    assert (
        first["information_boundary"]["adaptive_confirmation_payloads_opened"] is False
    )


def test_inventory_rejects_overlapping_or_exhausted_scope(tmp_path: Path) -> None:
    inventory_module = module()
    results = tmp_path / "results"
    results.mkdir()
    (results / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps"):
        inventory_module.inventory_results(
            results,
            forbidden_roots=(results,),
            maximum_depth=2,
            maximum_entries=10,
            maximum_candidates=10,
            maximum_json_bytes=1024,
        )

    for index in range(3):
        (results / f"result-{index}.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate bound"):
        inventory_module.inventory_results(
            results,
            forbidden_roots=(),
            maximum_depth=2,
            maximum_entries=10,
            maximum_candidates=2,
            maximum_json_bytes=1024,
        )
    with pytest.raises(ValueError, match="entry bound"):
        inventory_module.inventory_results(
            results,
            forbidden_roots=(),
            maximum_depth=2,
            maximum_entries=1,
            maximum_candidates=10,
            maximum_json_bytes=1024,
        )


def test_inventory_records_malformed_json_without_opening_binary_payloads(
    tmp_path: Path,
) -> None:
    inventory_module = module()
    results = tmp_path / "results"
    results.mkdir()
    (results / "provider-result.json").write_text(
        '{"schema":"x","schema":"y"}\n', encoding="utf-8"
    )
    (results / "factor-samples.npz").write_bytes(b"arbitrary-binary-bytes")

    value = inventory_module.inventory_results(
        results,
        forbidden_roots=(),
        maximum_depth=2,
        maximum_entries=10,
        maximum_candidates=10,
        maximum_json_bytes=1024,
    )
    records = {row["path"]: row for row in value["candidates"]}
    assert records["provider-result.json"]["json_status"] == (
        "unreadable-or-out-of-bound"
    )
    assert "json_error_sha256" in records["provider-result.json"]
    assert records["factor-samples.npz"] == {
        "path": "factor-samples.npz",
        "byte_count": len(b"arbitrary-binary-bytes"),
        "suffix": ".npz",
    }


def test_cli_publishes_once_and_rejects_replacement(tmp_path: Path) -> None:
    inventory_module = module()
    results = tmp_path / "results"
    raw = tmp_path / "raw"
    adaptive = tmp_path / "adaptive"
    results.mkdir()
    raw.mkdir()
    adaptive.mkdir()
    write_json(results / "source-gate" / "pipeline-receipt.json", {"schema": "receipt"})
    output = tmp_path / "inventory.json"

    arguments = [
        "--root",
        str(results),
        "--forbidden-root",
        str(raw),
        "--forbidden-root",
        str(adaptive),
        "--output",
        str(output),
        "--maximum-depth",
        "4",
        "--maximum-entries",
        "20",
        "--maximum-candidates",
        "20",
    ]
    assert inventory_module.main(arguments) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["counts"]["candidate_count"] == 1
    with pytest.raises(ValueError, match="already exists"):
        inventory_module.main(arguments)


def test_results_root_symlink_is_rejected(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("symlink permissions are platform-dependent")
    inventory_module = module()
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        inventory_module.inventory_results(
            linked,
            forbidden_roots=(),
            maximum_depth=2,
            maximum_entries=10,
            maximum_candidates=10,
            maximum_json_bytes=1024,
        )
