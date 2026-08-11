from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from bayesian_phystwin import numerical_environment_v1 as numerical_environment
from bayesian_phystwin.numerical_environment_v1 import (
    DependencyLockV1,
    InstalledDistributionV1,
    NumericalEnvironmentV1,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/release/build_release_matrix_evidence.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_release_matrix_evidence",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _file_record(path: Path, *, relative: str | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name if relative is None else relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_resolvers(root: Path) -> None:
    contents = {
        "requirements/release-build.txt": "pip==26.1.2\n",
        "requirements/release-runtime-py310-floor.txt": "numpy==1.23.5\n",
        "requirements/release-runtime-py312.txt": "numpy==2.2.6\n",
        "requirements/release-runtime-py314.txt": "numpy==2.5.2\n",
    }
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _release_evidence(
    root: Path,
    wheel: Path,
    sdist: Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    keys = {
        "requirements/release-build.txt": "release_build_requirements",
        "requirements/release-runtime-py310-floor.txt": (
            "release_runtime_py310_floor_requirements"
        ),
        "requirements/release-runtime-py312.txt": (
            "release_runtime_py312_requirements"
        ),
        "requirements/release-runtime-py314.txt": (
            "release_runtime_py314_requirements"
        ),
    }
    descriptor: dict[str, Any] = {
        "schema": "bayesian-phystwin.release-candidate-evidence",
        "schema_version": 1,
        "source_revision": source_revision,
        "project_version": "0.4.0",
        "artifacts": {
            "wheel": _file_record(wheel),
            "sdist": _file_record(sdist),
        },
        "source_contracts": {
            key: _file_record(root / relative, relative=relative)
            for relative, key in keys.items()
        },
    }
    return {"evidence_id": tool._content_id(descriptor), **descriptor}


def _profile(
    resolver: Path,
    *,
    python_version: str,
    numpy_version: str,
) -> NumericalEnvironmentV1:
    resolver_bytes = resolver.read_bytes()
    controls = {
        name: None for name in numerical_environment._EXECUTION_CONTROL_NAMES
    }
    distributions = tuple(
        sorted(
            (
                InstalledDistributionV1(
                    name="bayesian-phystwin",
                    version="0.4.0",
                ),
                InstalledDistributionV1(name="numpy", version=numpy_version),
            )
        )
    )
    return NumericalEnvironmentV1(
        python_implementation="CPython",
        python_version=python_version,
        python_compiler="test compiler",
        numpy_version=numpy_version,
        numpy_configuration_text="test build configuration\n",
        scipy_version=None,
        logical_cpu_count=4,
        byte_order="little",
        execution_controls=controls,
        installed_distributions=distributions,
        dependency_lock=DependencyLockV1(
            name=resolver.name,
            sha256=hashlib.sha256(resolver_bytes).hexdigest(),
            size_bytes=len(resolver_bytes),
        ),
    )


def _write_profile(path: Path, profile: NumericalEnvironmentV1) -> None:
    path.write_text(
        json.dumps(
            {
                numerical_environment.NUMERICAL_ENVIRONMENT_RUNTIME_KEY: (
                    profile.as_dict()
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    root = tmp_path / "project"
    root.mkdir()
    _write_resolvers(root)
    wheel = tmp_path / "bayesian_phystwin-0.4.0-py3-none-any.whl"
    sdist = tmp_path / "bayesian_phystwin-0.4.0.tar.gz"
    wheel.write_bytes(b"wheel bytes")
    sdist.write_bytes(b"sdist bytes")
    source_revision = "a" * 40
    evidence = _release_evidence(
        root,
        wheel,
        sdist,
        source_revision=source_revision,
    )
    evidence_path = tmp_path / "release-evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, wheel, sdist, evidence_path, source_revision


def _receipt_for_lane(
    tmp_path: Path,
    *,
    lane_name: str,
    root: Path,
    wheel: Path,
    sdist: Path,
    evidence_path: Path,
    source_revision: str,
    retained_dir: Path | None = None,
) -> dict[str, object]:
    lane = tool.LANES_BY_NAME[lane_name]
    resolver = root / lane.resolver_input
    actual_python = f"{lane.python_version}.19"
    profile = _profile(
        resolver,
        python_version=actual_python,
        numpy_version=lane.numpy_version,
    )
    profile_root = tmp_path if retained_dir is None else retained_dir
    profile_root.mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / f"numerical-environment-{lane_name}.json"
    _write_profile(profile_path, profile)
    if retained_dir is not None:
        (retained_dir / f"resolver-input-{lane_name}.txt").write_bytes(
            resolver.read_bytes()
        )
    return tool.build_lane_receipt(
        lane_name=lane_name,
        release_evidence_path=evidence_path,
        artifact_path=wheel if lane.artifact_kind == "wheel" else sdist,
        runtime_fragment_path=profile_path,
        resolver_input_path=resolver,
        source_revision=source_revision,
        runtime_python_version=actual_python,
        runtime_numpy_version=lane.numpy_version,
        runtime_project_version="0.4.0",
    )


def test_builds_lane_receipt_with_exact_artifact_and_runtime(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)

    receipt = _receipt_for_lane(
        tmp_path,
        lane_name="py310-wheel-floor",
        root=root,
        wheel=wheel,
        sdist=sdist,
        evidence_path=evidence_path,
        source_revision=source_revision,
    )

    assert receipt["artifact_kind"] == "wheel"
    assert receipt["python"] == {"requested": "3.10", "actual": "3.10.19"}
    assert receipt["numpy"] == {"expected": "1.23.5", "actual": "1.23.5"}
    descriptor = dict(receipt)
    supplied_id = descriptor.pop("receipt_id")
    assert supplied_id == tool._content_id(descriptor)


def test_lane_receipt_rejects_artifact_and_resolver_drift(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)
    wheel.write_bytes(b"tampered wheel")

    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="artifact digest"):
        _receipt_for_lane(
            tmp_path,
            lane_name="py310-wheel-floor",
            root=root,
            wheel=wheel,
            sdist=sdist,
            evidence_path=evidence_path,
            source_revision=source_revision,
        )

    wheel.write_bytes(b"wheel bytes")
    resolver = root / "requirements/release-runtime-py310-floor.txt"
    receipt_profile = _profile(
        resolver,
        python_version="3.10.19",
        numpy_version="1.23.5",
    )
    profile_path = tmp_path / "stale-profile.json"
    _write_profile(profile_path, receipt_profile)
    resolver.write_text("numpy==1.23.4\n", encoding="utf-8")
    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="exact resolver input"):
        tool.build_lane_receipt(
            lane_name="py310-wheel-floor",
            release_evidence_path=evidence_path,
            artifact_path=wheel,
            runtime_fragment_path=profile_path,
            resolver_input_path=resolver,
            source_revision=source_revision,
            runtime_python_version="3.10.19",
            runtime_numpy_version="1.23.5",
            runtime_project_version="0.4.0",
        )


def test_builds_complete_six_lane_matrix_evidence(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for lane_name in tool.LANES_BY_NAME:
        receipt = _receipt_for_lane(
            tmp_path,
            lane_name=lane_name,
            root=root,
            wheel=wheel,
            sdist=sdist,
            evidence_path=evidence_path,
            source_revision=source_revision,
            retained_dir=receipts,
        )
        (receipts / f"validation-receipt-{lane_name}.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    evidence = tool.build_matrix_evidence(
        release_evidence_path=evidence_path,
        receipts_dir=receipts,
        project_root=root,
    )

    assert len(evidence["lanes"]) == 6
    assert set(evidence["artifact_sha256"]) == {"wheel", "sdist"}
    descriptor = dict(evidence)
    supplied_id = descriptor.pop("matrix_evidence_id")
    assert supplied_id == tool._content_id(descriptor)

    summary = tmp_path / "summary.md"
    tool.write_matrix_summary(summary, evidence)
    text = summary.read_text(encoding="utf-8")
    assert "Validated lanes: `6`" in text
    assert "py314-sdist" in text


def test_matrix_rejects_missing_or_tampered_receipt(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    receipt = _receipt_for_lane(
        tmp_path,
        lane_name="py310-wheel-floor",
        root=root,
        wheel=wheel,
        sdist=sdist,
        evidence_path=evidence_path,
        source_revision=source_revision,
    )
    path = receipts / "validation-receipt-py310-wheel-floor.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="expected 6"):
        tool.build_matrix_evidence(
            release_evidence_path=evidence_path,
            receipts_dir=receipts,
            project_root=root,
        )

    for lane_name in tool.LANES_BY_NAME:
        value = _receipt_for_lane(
            tmp_path,
            lane_name=lane_name,
            root=root,
            wheel=wheel,
            sdist=sdist,
            evidence_path=evidence_path,
            source_revision=source_revision,
            retained_dir=receipts,
        )
        (receipts / f"validation-receipt-{lane_name}.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )
    tampered_path = receipts / "validation-receipt-py314-sdist.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["numpy"]["actual"] = "0.0.0"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        tool.ReleaseMatrixEvidenceError,
        match="ID does not match payload",
    ):
        tool.build_matrix_evidence(
            release_evidence_path=evidence_path,
            receipts_dir=receipts,
            project_root=root,
        )


def test_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    tool._write_json(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="refusing to overwrite"):
        tool._write_json(output, {"value": 2})


def test_unknown_lane_and_runtime_version_fail_closed(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)
    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="unknown release lane"):
        tool.build_lane_receipt(
            lane_name="unknown",
            release_evidence_path=evidence_path,
            artifact_path=wheel,
            runtime_fragment_path=tmp_path / "missing.json",
            resolver_input_path=root / "requirements/release-build.txt",
            source_revision=source_revision,
        )

    lane = tool.LANES_BY_NAME["py310-wheel-floor"]
    resolver = root / lane.resolver_input
    profile_path = tmp_path / "profile.json"
    _write_profile(
        profile_path,
        _profile(
            resolver,
            python_version="3.10.19",
            numpy_version=lane.numpy_version,
        ),
    )
    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="requires Python"):
        tool.build_lane_receipt(
            lane_name=lane.lane,
            release_evidence_path=evidence_path,
            artifact_path=wheel,
            runtime_fragment_path=profile_path,
            resolver_input_path=resolver,
            source_revision=source_revision,
            runtime_python_version="3.12.0",
            runtime_numpy_version=lane.numpy_version,
            runtime_project_version="0.4.0",
        )


def test_matrix_rejects_source_resolver_contract_drift(tmp_path: Path) -> None:
    root, wheel, sdist, evidence_path, source_revision = _fixture(tmp_path)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for lane_name in tool.LANES_BY_NAME:
        value = _receipt_for_lane(
            tmp_path,
            lane_name=lane_name,
            root=root,
            wheel=wheel,
            sdist=sdist,
            evidence_path=evidence_path,
            source_revision=source_revision,
            retained_dir=receipts,
        )
        (receipts / f"validation-receipt-{lane_name}.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )
    (root / "requirements/release-build.txt").write_text(
        "pip==26.1.1\n",
        encoding="utf-8",
    )

    with pytest.raises(tool.ReleaseMatrixEvidenceError, match="does not bind"):
        tool.build_matrix_evidence(
            release_evidence_path=evidence_path,
            receipts_dir=receipts,
            project_root=root,
        )
