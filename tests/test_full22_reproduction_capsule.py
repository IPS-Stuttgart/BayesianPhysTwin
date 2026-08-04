from __future__ import annotations

import hashlib
import importlib.util
import json
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "reproductions" / "full22_anchor_v1" / "reproduce.py"
)
EXPECTED_PATH = SCRIPT_PATH.with_name("expected_metrics.json")


def _load_capsule() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full22_reproduction_capsule", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(expected: dict[str, object]) -> dict[str, object]:
    methods: dict[str, object] = {}
    expected_methods = expected["methods"]
    assert isinstance(expected_methods, dict)
    for method, method_record in expected_methods.items():
        assert isinstance(method_record, dict)
        equal_case = method_record["equal_case"]
        frame_weighted = method_record["frame_weighted"]
        assert isinstance(equal_case, dict)
        assert isinstance(frame_weighted, dict)
        methods[method] = {
            "cohorts": {
                "all_22_table_compatible": {
                    metric: {
                        "equal_case_mean_m": equal_case[metric],
                        "frame_weighted_mean_m": frame_weighted[metric],
                    }
                    for metric in equal_case
                }
            }
        }
    return {"schema_version": 2, "methods": methods}


def _write_data_root(
    root: Path,
    capsule: ModuleType,
    *,
    manifest_name: str = "trajectory_evaluation_manifest.json",
) -> tuple[Path, dict[str, Any], str]:
    cases = [f"case-{index:02d}" for index in range(22)]
    records: dict[str, Any] = {}
    for case in cases:
        files: dict[str, Any] = {}
        case_dir = root / case
        case_dir.mkdir(parents=True)
        for filename in capsule.REQUIRED_DATA_FILENAMES:
            payload = f"{case}:{filename}\n".encode()
            path = case_dir / filename
            path.write_bytes(payload)
            source = "experiments" if filename == "inference.pkl" else "data"
            files[filename] = {
                "archive_member": f"{source}/{case}/{filename}",
                "bytes": len(payload),
                "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "reused": False,
            }
        records[case] = {"files": files}
    manifest: dict[str, Any] = {
        "schema": "test-source-manifest",
        "schema_version": 1,
        "created_at_utc": "volatile",
        "sources": {
            "data": "https://example.test/data.zip",
            "experiments": "https://example.test/experiments.zip",
            "optimization": "https://example.test/ignored.zip",
        },
        "available_cases": cases,
        "selected_cases": cases,
        "cases": records,
    }
    path = root / manifest_name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    identity = capsule._canonical_sha256(capsule._normalized_data_manifest(manifest))
    return path, manifest, identity


def test_full22_expected_metrics_are_verified() -> None:
    capsule = _load_capsule()
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))

    report = capsule.verify_comparison(_comparison(expected), expected)

    assert report["status"] == "verified"
    assert report["check_count"] == 8
    assert all(record["passed"] for record in report["checks"])


def test_full22_metric_drift_fails_closed() -> None:
    capsule = _load_capsule()
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    comparison = _comparison(expected)
    comparison["methods"]["bayesian_anchor"]["cohorts"]["all_22_table_compatible"][
        "track_error_m"
    ]["equal_case_mean_m"] += 1e-4

    with pytest.raises(ValueError, match="metric verification failed"):
        capsule.verify_comparison(comparison, expected)


def test_confirmation_summary_requires_protocol_and_complete_cohort() -> None:
    capsule = _load_capsule()
    summary = {
        "protocol_id": capsule.EXPECTED_PROTOCOL_ID,
        "case_results": {f"case-{index:02d}": {} for index in range(22)},
    }
    capsule.verify_confirmation_summary(summary)

    summary["protocol_id"] = "changed"
    with pytest.raises(ValueError, match="protocol ID changed"):
        capsule.verify_confirmation_summary(summary)


def test_data_manifest_identity_ignores_retrieval_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _load_capsule()
    manifest, payload, identity = _write_data_root(tmp_path, capsule)
    monkeypatch.setattr(capsule, "EXPECTED_DATA_MANIFEST_IDENTITY_SHA256", identity)

    assert capsule.validate_data_root(tmp_path) == (manifest, identity)

    payload["created_at_utc"] = "different"
    for case in payload["selected_cases"]:
        for record in payload["cases"][case]["files"].values():
            record["reused"] = True
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    assert capsule.validate_data_root(tmp_path) == (manifest, identity)


def test_data_manifest_identity_binds_order_and_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _load_capsule()
    manifest, payload, identity = _write_data_root(
        tmp_path,
        capsule,
        manifest_name="evaluation_subset_manifest.json",
    )
    monkeypatch.setattr(capsule, "EXPECTED_DATA_MANIFEST_IDENTITY_SHA256", identity)

    payload["available_cases"] = list(reversed(payload["selected_cases"]))
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="available_cases"):
        capsule.validate_data_root(tmp_path)

    payload["available_cases"] = payload["selected_cases"]
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    damaged = tmp_path / payload["selected_cases"][0] / "split.json"
    damaged.write_bytes(b"changed but not declared\n")
    with pytest.raises(ValueError, match="data file (size|digest) changed"):
        capsule.validate_data_root(tmp_path)


def test_multiple_manifests_must_agree_semantically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = _load_capsule()
    _, payload, identity = _write_data_root(tmp_path, capsule)
    monkeypatch.setattr(capsule, "EXPECTED_DATA_MANIFEST_IDENTITY_SHA256", identity)
    second = tmp_path / "evaluation_subset_manifest.json"
    payload["cases"][payload["selected_cases"][0]]["files"]["split.json"][
        "archive_member"
    ] = "changed/member"
    second.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="multiple data manifests disagree"):
        capsule.validate_data_root(tmp_path)


def test_manifest_command_binds_claim_protocol_and_outputs(tmp_path: Path) -> None:
    capsule = _load_capsule()
    command = capsule._manifest_command(tmp_path, tmp_path / "source", "run command")

    assert "bpt.full22_anchor_released_contract" in command
    assert capsule.EXPECTED_PROTOCOL_ID in command
    assert "data_identity=input/data_identity.json" in command
    assert "full22_comparison=full22_comparison.json" in command
    assert "verification=verification.json" in command


@pytest.mark.parametrize("relation", ("same", "ancestor", "descendant"))
def test_output_path_rejects_protected_overlaps(
    tmp_path: Path,
    relation: str,
) -> None:
    capsule = _load_capsule()
    protected = tmp_path / "protected"
    protected.mkdir()
    output = {
        "same": protected,
        "ancestor": tmp_path,
        "descendant": protected / "bundle",
    }[relation]

    with pytest.raises(ValueError, match="must not overlap"):
        capsule._validate_output_path(output, (protected,))


def test_output_path_rejects_existing_file(tmp_path: Path) -> None:
    capsule = _load_capsule()
    protected = tmp_path / "protected"
    protected.mkdir()
    output = tmp_path / "bundle"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        capsule._validate_output_path(output, (protected,))


def test_force_preserves_output_when_source_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule = _load_capsule()
    source = tmp_path / "source"
    data = tmp_path / "data"
    output = tmp_path / "output"
    source.mkdir()
    data.mkdir()
    output.mkdir()
    marker_path = output / "must-survive.txt"
    marker_path.write_text("evidence", encoding="utf-8")

    def reject_source(_source: Path) -> None:
        raise ValueError("invalid source")

    monkeypatch.setattr(capsule, "validate_source_checkout", reject_source)
    args = SimpleNamespace(
        source_checkout=source,
        data_root=data,
        output_dir=output,
        workers=1,
        force=True,
    )

    with pytest.raises(ValueError, match="invalid source"):
        capsule.reproduce(args)
    assert marker_path.read_text(encoding="utf-8") == "evidence"
