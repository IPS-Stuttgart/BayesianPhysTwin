from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts" / "held" / "prepare_deform360_v6_lock.py"
V5_WITHDRAWAL_SEALER = (
    ROOT / "scripts" / "held" / "seal_deform360_v5_execution_withdrawal.py"
)
V6_OPERATORS = (
    ROOT / "scripts" / "held" / "run_deform360_v6_calibration_case.sh",
    ROOT / "scripts" / "held" / "run_deform360_v6_calibration_shard.sh",
    ROOT / "scripts" / "held" / "run_deform360_v6_calibration_outcomes.py",
    ROOT / "scripts" / "held" / "run_deform360_v6_confirmation_case.sh",
    ROOT / "scripts" / "held" / "run_deform360_v6_confirmation_shard.sh",
    ROOT / "scripts" / "held" / "run_deform360_v6_confirmation_outcomes.py",
)


def _load_operator(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v6_binding_classification_is_exact() -> None:
    preparer = _load_operator("deform360_v6_lock_preparer_bindings", PREPARER)
    assert preparer.EXPECTED_V5_LOCK_FILE_SHA256 == (
        "a917650499b047bdcd7d7baf57212ff82a9e277867bb3ba5389b1a0c126d950e"
    )
    assert preparer.EXPECTED_V5_LOCK_ARTIFACT_SHA256 == (
        "cfb13c88220e5abbe83937d20e125ed7c22727bcce5c4acbf58df0f2b07d440d"
    )
    assert preparer.EXPECTED_V5_LOCK_BINDING_COUNT == 112
    assert preparer.EXPECTED_V6_BINDING_COUNT == 113
    assert preparer.EXPECTED_V6_MIGRATION_KEY_COUNT == 22
    assert preparer._PYCACHE_PREFIX == "/nonexistent/bpt-held-v6-pycache"

    assert (
        preparer.LOCAL_FILE_BINDINGS["held_protocol_lock_operator_source"]
        == "scripts/held/prepare_deform360_v6_lock.py"
    )
    for role in ("calibration", "confirmation"):
        assert (
            f"run_deform360_v6_{role}_case.sh"
            in preparer.LOCAL_FILE_BINDINGS[f"held_{role}_case_runner_source"]
        )
        assert (
            f"run_deform360_v6_{role}_shard.sh"
            in preparer.LOCAL_FILE_BINDINGS[f"held_{role}_shard_runner_source"]
        )
        assert (
            f"run_deform360_v6_{role}_outcomes.py"
            in preparer.LOCAL_FILE_BINDINGS[f"held_{role}_outcome_driver_source"]
        )

    lineage = {
        "v1_preoutcome_feasibility_report",
        "v2_design_withdrawal_report",
        "v3_prelock_boundary_incident_report",
        "v4_execution_withdrawal_report",
        "v5_outcome_withdrawal_report",
    }
    groups = (
        set(preparer.INHERITED_EXTERNAL_BINDING_KEYS),
        set(preparer.V6_PINNED_EXTERNAL_BINDING_KEYS),
        set(preparer.LOCAL_FILE_BINDINGS),
        set(preparer.LOCAL_CONTRACT_BINDING_KEYS),
        set(preparer.METHOD_PROVENANCE_BINDING_KEYS),
        lineage,
    )
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert group.isdisjoint(other)
    assert len(set().union(*groups)) == preparer.EXPECTED_V6_BINDING_COUNT
    assert "v5_outcome_withdrawal_report" in preparer.V6_ONLY_BINDING_KEYS
    assert len(preparer.V6_ONLY_BINDING_KEYS) == (
        preparer.EXPECTED_V6_MIGRATION_KEY_COUNT
    )


def test_exact_v5_withdrawal_report_is_accepted_and_tampering_rejected() -> None:
    preparer = _load_operator("deform360_v6_lock_preparer_report", PREPARER)
    sealer = _load_operator(
        "deform360_v5_withdrawal_fixture_for_v6", V5_WITHDRAWAL_SEALER
    )
    report, payload = sealer._artifact(sealer.expected_unsigned_report())
    assert len(payload) == preparer.EXPECTED_V5_REPORT_SIZE_BYTES
    assert report["artifact_sha256"] == preparer.EXPECTED_V5_REPORT_ARTIFACT_SHA256

    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        held_v5 = Path(temporary).resolve()
        report_path = held_v5 / "v5-outcome-withdrawal-report.json"
        report_path.write_bytes(payload)
        report_path.chmod(0o400)
        held_v5.chmod(0o500)
        try:
            preparer._CANONICAL_HELD_V5_ROOT = held_v5
            preparer._CANONICAL_V5_OUTCOME_WITHDRAWAL_REPORT = report_path
            assert preparer._validate_v5_outcome_withdrawal_report(report_path) == (
                preparer.EXPECTED_V5_REPORT_FILE_SHA256
            )

            held_v5.chmod(0o700)
            report_path.chmod(0o600)
            report_path.write_bytes(
                payload.replace(
                    b'"outcome_created_count": 0', b'"outcome_created_count": 1'
                )
            )
            report_path.chmod(0o400)
            held_v5.chmod(0o500)
            with pytest.raises(ValueError, match="file checksum changed"):
                preparer._validate_v5_outcome_withdrawal_report(report_path)
        finally:
            held_v5.chmod(0o700)
            report_path.chmod(0o600)


def test_v6_operator_preparer_digest_and_barrier_self_checks_are_final() -> None:
    expected = hashlib.sha256(PREPARER.read_bytes()).hexdigest()
    assert expected == (
        "7622e8a4338c9a76da3d114bde4fa5407374396f4943a838b13dd2214eebe329"
    )
    observed: list[str] = []
    for path in V6_OPERATORS:
        assert path.stat().st_mode & stat.S_IXUSR
        source = path.read_text(encoding="utf-8")
        match = re.search(
            r"(?:readonly )?EXPECTED_LOCK_OPERATOR_SHA256\s*=\s*"
            r"(?:\(\s*)?[\"']([^\"']+)",
            source,
        )
        assert match is not None
        observed.append(match.group(1))
        if path.suffix == ".sh":
            subprocess.run(["/bin/bash", "-n", str(path)], check=True)
    assert observed == [expected] * len(V6_OPERATORS)

    for path in V6_OPERATORS[2::3]:
        completed = subprocess.run(
            [sys.executable, str(path), "--self-check"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "SELF_CHECK_PASSED" in completed.stdout
