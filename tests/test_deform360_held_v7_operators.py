from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ROOT = ROOT / "scripts" / "held"
PREPARER = OPERATOR_ROOT / "prepare_deform360_v7_lock.py"
V6_WITHDRAWAL_SEALER = OPERATOR_ROOT / "seal_deform360_v6_execution_withdrawal.py"
V7_OPERATORS = (
    OPERATOR_ROOT / "run_deform360_v7_calibration_case.sh",
    OPERATOR_ROOT / "run_deform360_v7_calibration_shard.sh",
    OPERATOR_ROOT / "run_deform360_v7_calibration_outcomes.py",
    OPERATOR_ROOT / "run_deform360_v7_confirmation_case.sh",
    OPERATOR_ROOT / "run_deform360_v7_confirmation_shard.sh",
    OPERATOR_ROOT / "run_deform360_v7_confirmation_outcomes.py",
)


def _load_operator(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(module, unsigned: dict[str, object]) -> tuple[dict[str, object], bytes]:
    artifact = {**unsigned, "artifact_sha256": module._canonical_sha256(unsigned)}
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return artifact, payload


def _array_values(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"readonly -a {re.escape(name)}=\((.*?)\n\)",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return tuple(re.findall(r'"([^"\n]+)"', match.group(1)))


def test_v7_binding_classification_and_lineage_are_exact() -> None:
    preparer = _load_operator("deform360_v7_lock_preparer_bindings", PREPARER)
    assert preparer.EXPECTED_V6_LOCK_FILE_SHA256 == (
        "0436d6467d14e1caa42a0895c5cede13e97d96bd9c01fdea6b40557d6f539f8c"
    )
    assert preparer.EXPECTED_V6_LOCK_ARTIFACT_SHA256 == (
        "b334f7fcb28753ab4526465e5ca6e69397fa4cc65d1db6b8d9a68c090c6bb3db"
    )
    assert preparer.EXPECTED_V6_LOCK_BINDING_COUNT == 113
    assert preparer.EXPECTED_V6_REPORT_FILE_SHA256 == (
        "8a428535708057ff1c944b8ab81c93b3309539ae9d3dffb469ddc2b9f79de504"
    )
    assert preparer.EXPECTED_V6_REPORT_ARTIFACT_SHA256 == (
        "383d2d72ba148703482df76cdbf89ad8d43c6a5026b89325984a5d786748c843"
    )
    assert preparer.EXPECTED_V6_REPORT_SIZE_BYTES == 16_780
    assert preparer.EXPECTED_V7_BINDING_COUNT == 118
    assert preparer.EXPECTED_V7_MIGRATION_KEY_COUNT == 27
    assert preparer._PYCACHE_PREFIX == "/nonexistent/bpt-held-v7-pycache"
    assert (
        preparer.LOCAL_FILE_BINDINGS["held_outcome_cuda_smoke_operator_source"]
        == "src/bayesian_phystwin/deform360_held_gsplat_runtime.py"
    )
    assert "held_outcome_cuda_smoke_contract" in preparer.LOCAL_CONTRACT_BINDING_KEYS

    lineage = {
        "v1_preoutcome_feasibility_report",
        "v2_design_withdrawal_report",
        "v3_prelock_boundary_incident_report",
        "v4_execution_withdrawal_report",
        "v5_outcome_withdrawal_report",
        "v6_outcome_withdrawal_report",
    }
    runtime_evidence = {
        "held_gsplat_runtime_supplement_manifest",
        "held_outcome_cuda_smoke_evidence",
    }
    groups = (
        set(preparer.INHERITED_EXTERNAL_BINDING_KEYS),
        set(preparer.V7_PINNED_EXTERNAL_BINDING_KEYS),
        set(preparer.LOCAL_FILE_BINDINGS),
        set(preparer.LOCAL_CONTRACT_BINDING_KEYS),
        set(preparer.METHOD_PROVENANCE_BINDING_KEYS),
        runtime_evidence,
        lineage,
    )
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert group.isdisjoint(other)
    assert len(set().union(*groups)) == preparer.EXPECTED_V7_BINDING_COUNT
    assert len(preparer.V7_ONLY_BINDING_KEYS) == (
        preparer.EXPECTED_V7_MIGRATION_KEY_COUNT
    )
    assert lineage | runtime_evidence <= preparer.V7_ONLY_BINDING_KEYS


def test_exact_v6_withdrawal_report_is_accepted_and_tampering_rejected() -> None:
    preparer = _load_operator("deform360_v7_lock_preparer_report", PREPARER)
    sealer = _load_operator(
        "deform360_v6_withdrawal_fixture_for_v7", V6_WITHDRAWAL_SEALER
    )
    report, payload = sealer._artifact(sealer.expected_unsigned_report())
    assert len(payload) == preparer.EXPECTED_V6_REPORT_SIZE_BYTES
    assert report["artifact_sha256"] == preparer.EXPECTED_V6_REPORT_ARTIFACT_SHA256

    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        held_v6 = Path(temporary).resolve()
        report_path = held_v6 / "v6-outcome-withdrawal-report.json"
        report_path.write_bytes(payload)
        report_path.chmod(0o400)
        held_v6.chmod(0o500)
        try:
            preparer._CANONICAL_HELD_V6_ROOT = held_v6
            preparer._CANONICAL_V6_OUTCOME_WITHDRAWAL_REPORT = report_path
            assert preparer._validate_v6_outcome_withdrawal_report(report_path) == (
                preparer.EXPECTED_V6_REPORT_FILE_SHA256
            )

            held_v6.chmod(0o700)
            report_path.chmod(0o600)
            report_path.write_bytes(payload[:-1] + b" \n")
            report_path.chmod(0o400)
            held_v6.chmod(0o500)
            with pytest.raises(ValueError, match="file checksum changed"):
                preparer._validate_v6_outcome_withdrawal_report(report_path)
        finally:
            held_v6.chmod(0o700)
            report_path.chmod(0o600)


def test_runtime_supplement_and_deployed_smoke_evidence_are_strict() -> None:
    preparer = _load_operator("deform360_v7_runtime_evidence", PREPARER)
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        root = Path(temporary).resolve()
        supplement_root = root / "supplement"
        supplement_root.mkdir()
        extension = supplement_root / "gsplat_cuda.so"
        extension.write_bytes(b"fixed fake extension")
        extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
        manifest_path = supplement_root / "runtime-supplement-manifest.json"
        extension_contract = {
            "base_runtime_manifest_sha256": (
                preparer.EXPECTED_V5_RUNTIME_MANIFEST_FILE_SHA256
            ),
            "base_runtime_pip_freeze_sha256": (preparer.EXPECTED_V5_PIP_FREEZE_SHA256),
            "canonical_path": os.fspath(extension),
            "file_mode_octal": "0444",
            "file_size_bytes": len(b"fixed fake extension"),
            "parent_mode_octal": "0555",
            "sha256": extension_sha256,
        }
        extension_contract_sha256 = preparer._canonical_sha256(extension_contract)
        manifest, manifest_payload = _artifact(
            preparer,
            {
                "artifact_kind": ("Deform360HeldGsplatRuntimeSupplementManifestV1"),
                "extension_contract": extension_contract,
                "extension_contract_sha256": extension_contract_sha256,
                "schema_version": 1,
            },
        )
        manifest_path.write_bytes(manifest_payload)
        extension.chmod(0o444)
        manifest_path.chmod(0o400)
        supplement_root.chmod(0o555)

        evidence_path = root / "gsplat-runtime-smoke-evidence.json"
        contract_sha256 = "c" * 64
        operator_sha256 = "d" * 64
        head = "e" * 40
        smoke_common = {
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "compute_capability": "8.9",
            "contract_sha256": contract_sha256,
            "extension_loaded_and_retained": True,
            "extension_path": os.fspath(extension),
            "extension_sha256": extension_sha256,
            "gpu_name": "NVIDIA RTX 6000 Ada Generation",
            "gsplat_version": "1.4.0",
            "logical_device": "cuda:0",
            "ninja_visible": False,
            "nvcc_visible": False,
            "predicates": {
                "alpha_shape": [1, 16, 16, 1],
                "backward_complete": True,
                "cuda_synchronized": True,
                "forward_finite_nonempty_nonzero": True,
                "gradient_groups_finite_and_nonzero": [
                    "colors",
                    "means",
                    "opacities",
                    "quats",
                    "scales",
                ],
                "positive_radius_count": 2,
                "render_shape": [1, 16, 16, 3],
            },
            "python_version": "3.12",
            "schema_version": 1,
            "target_or_outcome_path_accessed": False,
            "torch_cuda_version": "12.1",
            "torch_version": "2.4.0+cu121",
        }
        smoke_zero, _ = _artifact(preparer, {**smoke_common, "physical_gpu_index": 0})
        smoke_one, _ = _artifact(preparer, {**smoke_common, "physical_gpu_index": 1})
        evidence, evidence_payload = _artifact(
            preparer,
            {
                "artifact_kind": "Deform360HeldGsplatRuntimeSmokeEvidenceV1",
                "contract_sha256": contract_sha256,
                "deployed_method_head": head,
                "operator_source_sha256": operator_sha256,
                "runtime_supplement_manifest_sha256": hashlib.sha256(
                    manifest_payload
                ).hexdigest(),
                "schema_version": 1,
                "smokes": [smoke_zero, smoke_one],
            },
        )
        evidence_path.write_bytes(evidence_payload)
        evidence_path.chmod(0o400)

        preparer._CANONICAL_GSPLAT_RUNTIME_SUPPLEMENT_MANIFEST = manifest_path
        preparer._CANONICAL_GSPLAT_CUDA_EXTENSION = extension
        preparer._EXPECTED_GSPLAT_CUDA_EXTENSION_SHA256 = extension_sha256
        preparer._EXPECTED_GSPLAT_CUDA_EXTENSION_SIZE_BYTES = len(
            b"fixed fake extension"
        )
        preparer._CANONICAL_GSPLAT_RUNTIME_SMOKE_EVIDENCE = evidence_path
        observed_manifest_sha256, observed_manifest = (
            preparer._validate_runtime_supplement_manifest(
                manifest_path,
                expected_extension_contract=extension_contract,
                expected_extension_contract_sha256=extension_contract_sha256,
            )
        )
        assert observed_manifest == manifest
        assert observed_manifest_sha256 == hashlib.sha256(manifest_payload).hexdigest()
        assert preparer._validate_runtime_smoke_evidence(
            evidence_path,
            expected_contract_sha256=contract_sha256,
            supplement_manifest_sha256=observed_manifest_sha256,
            deployed_method_head=head,
            operator_source_sha256=operator_sha256,
        ) == (hashlib.sha256(evidence_payload).hexdigest(), evidence)

        evidence_path.chmod(0o600)
        tampered = evidence_payload.replace(
            b'"physical_gpu_index": 1', b'"physical_gpu_index": 0'
        )
        evidence_path.write_bytes(tampered)
        evidence_path.chmod(0o400)
        with pytest.raises(ValueError, match="identity or checksum changed"):
            preparer._validate_runtime_smoke_evidence(
                evidence_path,
                expected_contract_sha256=contract_sha256,
                supplement_manifest_sha256=observed_manifest_sha256,
                deployed_method_head=head,
                operator_source_sha256=operator_sha256,
            )


def test_v7_operator_bundle_is_fresh_bound_and_smoke_first() -> None:
    expected_preparer_sha256 = hashlib.sha256(PREPARER.read_bytes()).hexdigest()
    observed_preparer_sha256: list[str] = []
    for path in V7_OPERATORS:
        assert path.stat().st_mode & stat.S_IXUSR
        source = path.read_text(encoding="utf-8")
        assert "held-v7" in source
        assert "/nonexistent/bpt-held-v7-pycache" in source
        assert "--v6-lock" in source
        assert "--v6-outcome-withdrawal-report" in source
        assert "--gsplat-runtime-supplement-manifest" in source
        assert "--gsplat-runtime-smoke-evidence" in source
        assert "held-v6/calibration/cases" not in source
        match = re.search(
            r"(?:readonly )?EXPECTED_LOCK_OPERATOR_SHA256\s*=\s*"
            r"(?:\(\s*)?[\"']([^\"']+)",
            source,
        )
        assert match is not None
        observed_preparer_sha256.append(match.group(1))
        if path.suffix == ".sh":
            subprocess.run(["/bin/bash", "-n", str(path)], check=True)
    assert observed_preparer_sha256 == [expected_preparer_sha256] * len(V7_OPERATORS)

    for outcome_driver in V7_OPERATORS[2::3]:
        source = outcome_driver.read_text(encoding="utf-8")
        execute = source[source.index("def _execute_driver(") :]
        assert execute.index("smoke_gsplat_runtime()") < execute.index(
            "protocol.authorize_outcome_phase("
        )
        assert 'getattr(gsplat_runtime, "load_and_smoke_gsplat_runtime"' in source
        assert "except Exception as error:" in source
        completed = subprocess.run(
            [sys.executable, str(outcome_driver), "--self-check"],
            check=True,
            capture_output=True,
            text=True,
        )
        events = [json.loads(line)["event"] for line in completed.stdout.splitlines()]
        assert events[0] == "GSPLAT_RUNTIME_SMOKE_VALIDATED"
        assert events[1].endswith("COHORT_BARRIER_VALIDATED")
        assert events[-1] == "SELF_CHECK_PASSED"


def test_v7_case_cohorts_match_frozen_v6_without_reuse() -> None:
    for role in ("calibration", "confirmation"):
        v7_shard = (OPERATOR_ROOT / f"run_deform360_v7_{role}_shard.sh").read_text(
            encoding="utf-8"
        )
        v6_shard = (OPERATOR_ROOT / f"run_deform360_v6_{role}_shard.sh").read_text(
            encoding="utf-8"
        )
        for array_name in (
            "ALL_CASE_SPECS",
            "SHARD_0_CASE_SPECS",
            "SHARD_1_CASE_SPECS",
        ):
            assert _array_values(v7_shard, array_name) == _array_values(
                v6_shard, array_name
            )
        v7_case = (OPERATOR_ROOT / f"run_deform360_v7_{role}_case.sh").read_text(
            encoding="utf-8"
        )
        assert all(
            case in v7_case for case in _array_values(v7_shard, "ALL_CASE_SPECS")
        )
        assert (
            'readonly HELD="/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7"'
            in v7_case
        )
        assert 'readonly RUN="$HELD/' in v7_case
