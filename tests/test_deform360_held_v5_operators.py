from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ROOT = ROOT / "scripts" / "held"
PREPARER = OPERATOR_ROOT / "prepare_deform360_v5_lock.py"
CASE_RUNNER = OPERATOR_ROOT / "run_deform360_v5_calibration_case.sh"
SHARD_RUNNER = OPERATOR_ROOT / "run_deform360_v5_calibration_shard.sh"
OUTCOME_DRIVER = OPERATOR_ROOT / "run_deform360_v5_calibration_outcomes.py"
CONFIRMATION_CASE_RUNNER = OPERATOR_ROOT / "run_deform360_v5_confirmation_case.sh"
CONFIRMATION_SHARD_RUNNER = OPERATOR_ROOT / "run_deform360_v5_confirmation_shard.sh"
CONFIRMATION_OUTCOME_DRIVER = (
    OPERATOR_ROOT / "run_deform360_v5_confirmation_outcomes.py"
)
V4_WITHDRAWAL_SEALER = OPERATOR_ROOT / "seal_deform360_v4_execution_withdrawal.py"

EXPECTED_CASE_SPECS = (
    "002-rope-silk-ep0003:002-rope-silk:0003",
    "002-rope-silk-ep0004:002-rope-silk:0004",
    "002-rope-silk-ep0008:002-rope-silk:0008",
    "083-blanket-cloth-ep0000:083-blanket-cloth:0000",
    "083-blanket-cloth-ep0003:083-blanket-cloth:0003",
    "083-blanket-cloth-ep0006:083-blanket-cloth:0006",
    "085-scarf-cloth-ep0000:085-scarf-cloth:0000",
    "085-scarf-cloth-ep0005:085-scarf-cloth:0005",
    "085-scarf-cloth-ep0007:085-scarf-cloth:0007",
    "092-squirrel-ep0002:092-squirrel:0002",
    "092-squirrel-ep0003:092-squirrel:0003",
    "092-squirrel-ep0006:092-squirrel:0006",
    "170-spider-ep0002:170-spider:0002",
    "170-spider-ep0004:170-spider:0004",
    "170-spider-ep0007:170-spider:0007",
)
EXPECTED_CONFIRMATION_CASE_SPECS = (
    "002-rope-silk-ep0001:002-rope-silk:0001",
    "081-stripe-rope-ep0005:081-stripe-rope:0005",
    "085-scarf-cloth-ep0002:085-scarf-cloth:0002",
    "083-blanket-cloth-ep0007:083-blanket-cloth:0007",
    "092-squirrel-ep0001:092-squirrel:0001",
    "170-spider-ep0006:170-spider:0006",
)


def _load_operator(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _array_values(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"readonly -a {re.escape(name)}=\((.*?)\n\)",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return tuple(re.findall(r'"([^"\n]+)"', match.group(1)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_operator_sources_parse_and_shells_are_hardened() -> None:
    for path in (PREPARER, OUTCOME_DRIVER, CONFIRMATION_OUTCOME_DRIVER):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (
        CASE_RUNNER,
        SHARD_RUNNER,
        CONFIRMATION_CASE_RUNNER,
        CONFIRMATION_SHARD_RUNNER,
    ):
        subprocess.run(["/bin/bash", "-n", str(path)], check=True)
        source = path.read_text(encoding="utf-8")
        assert source.startswith("#!/bin/bash\n")
        assert "-perm /222" in source
        assert "-perm /022" not in source
        assert "BPT_HELD_V2" not in source
        assert "/held-v2" not in source
        assert "--v2-lock" not in source
        assert "run_deform360_v2" not in source
        assert "run_deform360_v3_calibration" not in source
        assert "run_deform360_v5_" in source
        assert "PYTHONPYCACHEPREFIX" in source
        assert "/nonexistent/bpt-held-v5-pycache" in source
        assert "/home/florianpfaff/.venvs/bpt-gpu" not in source
        assert "PATH=/usr/local/bin:/usr/bin:/bin" in source
        assert (
            "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
            "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466d"
            "db7b5f1e9c1b0de1137f004/bin/python"
        ) in source
        if " -I " in source:
            assert '-B -X "pycache_prefix=$PYCACHE_PREFIX" -I' in source
    for path in (
        CASE_RUNNER,
        SHARD_RUNNER,
        OUTCOME_DRIVER,
        CONFIRMATION_CASE_RUNNER,
        CONFIRMATION_SHARD_RUNNER,
        CONFIRMATION_OUTCOME_DRIVER,
    ):
        source = path.read_text(encoding="utf-8")
        assert "--v4-lock" in source
        assert "--v4-execution-withdrawal-report" in source
        assert "v4-execution-withdrawal-report.json" in source
    shard_source = SHARD_RUNNER.read_text(encoding="utf-8")
    assert "SHARD_INDEX" in shard_source
    assert "CUDA_DEVICE" in shard_source
    assert '/bin/bash "$CASE_RUNNER" "$CUDA_DEVICE"' in shard_source
    assert "env -i" in shard_source
    for path in (CASE_RUNNER, CONFIRMATION_CASE_RUNNER):
        source = path.read_text(encoding="utf-8")
        assert "PYTHONPATH=" not in source
        assert "runpy.run_module" in source
        assert '"$PY" -I -B -X "pycache_prefix=$PYCACHE_PREFIX"' in source
        assert 'cd -- "$CODE"' in source
        assert 'readonly OBJECT_DIR="$ALIGNED/$OBJECT"' in source


def test_exact_calibration_cohort_and_disjoint_shards() -> None:
    source = SHARD_RUNNER.read_text(encoding="utf-8")
    all_cases = _array_values(source, "ALL_CASE_SPECS")
    shard_zero = _array_values(source, "SHARD_0_CASE_SPECS")
    shard_one = _array_values(source, "SHARD_1_CASE_SPECS")
    assert all_cases == EXPECTED_CASE_SPECS
    assert len(shard_zero) == 8
    assert len(shard_one) == 7
    assert set(shard_zero).isdisjoint(shard_one)
    assert set(shard_zero) | set(shard_one) == set(all_cases)
    case_source = CASE_RUNNER.read_text(encoding="utf-8")
    assert all(case in case_source for case in EXPECTED_CASE_SPECS)


def test_exact_confirmation_cohort_and_disjoint_shards() -> None:
    source = CONFIRMATION_SHARD_RUNNER.read_text(encoding="utf-8")
    all_cases = _array_values(source, "ALL_CASE_SPECS")
    shard_zero = _array_values(source, "SHARD_0_CASE_SPECS")
    shard_one = _array_values(source, "SHARD_1_CASE_SPECS")
    assert all_cases == EXPECTED_CONFIRMATION_CASE_SPECS
    assert len(shard_zero) == len(shard_one) == 3
    assert set(shard_zero).isdisjoint(shard_one)
    assert set(shard_zero) | set(shard_one) == set(all_cases)
    case_source = CONFIRMATION_CASE_RUNNER.read_text(encoding="utf-8")
    assert all(case in case_source for case in EXPECTED_CONFIRMATION_CASE_SPECS)
    assert "--role confirmation" in case_source
    assert "--role calibration" not in case_source


def test_verification_precedes_episode_access_and_formal_shard_claim() -> None:
    case_source = CASE_RUNNER.read_text(encoding="utf-8")
    assert case_source.index("--verify-existing-lock") < case_source.index(
        '[[ -d "$EPDIR"'
    )
    shard_source = SHARD_RUNNER.read_text(encoding="utf-8")
    claim = shard_source.index('mkdir -- "$SHARD_CLAIM"')
    assert shard_source.index("--verify-existing-lock") < claim
    assert shard_source.index("# Refuse a partial/reused v5 shard") < claim
    assert claim < shard_source.index('echo "SHARD_START')
    confirmation_case = CONFIRMATION_CASE_RUNNER.read_text(encoding="utf-8")
    assert confirmation_case.index("--verify-existing-lock") < confirmation_case.index(
        '[[ -d "$EPDIR"'
    )
    confirmation_shard = CONFIRMATION_SHARD_RUNNER.read_text(encoding="utf-8")
    confirmation_claim = confirmation_shard.index('mkdir -- "$SHARD_CLAIM"')
    assert confirmation_shard.index("--verify-existing-lock") < confirmation_claim
    assert confirmation_shard.index("load_held_protocol_lock") < confirmation_claim


def test_case_verifier_uses_exact_preparer_environment() -> None:
    case_source = CASE_RUNNER.read_text(encoding="utf-8")
    verifier = case_source[
        case_source.index(
            "# Recompute the full deployed Git snapshot"
        ) : case_source.index("readonly REVERIFIED_LOCK_SHA256=")
    ]
    for assignment in (
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1",
        "PYTHONHASHSEED=0",
        "GIT_OPTIONAL_LOCKS=0",
    ):
        assert assignment in verifier
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in case_source


def test_frame_zero_build_uses_exact_pinned_semantic_runtime() -> None:
    source = CASE_RUNNER.read_text(encoding="utf-8")
    assert (
        'readonly SEMANTIC_MODEL="/mnt/corsair/florianpfaff/model-cache/'
        'siglip2-base-patch16-224-75de2d55"' in source
    )
    assert (
        'readonly SEMANTIC_MODEL_LOCK="/mnt/corsair/florianpfaff/'
        'bpt-framezero-field-dev-20260720/scratch_siglip2_model_lock.json"' in source
    )
    frame_zero = source[
        source.index('CURRENT_PHASE="frame-zero-build"') : source.index(
            "readonly FZ_MANIFEST="
        )
    ]
    assert '--semantic-model "$SEMANTIC_MODEL"' in frame_zero
    assert '--semantic-model-lock "$SEMANTIC_MODEL_LOCK"' in frame_zero
    assert '--deform360-code "$DEFORM360"' in frame_zero
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source


def test_shards_are_bound_to_one_host_and_exact_gpu_mapping() -> None:
    source = SHARD_RUNNER.read_text(encoding="utf-8")
    assert '"$(hostname)" == "workstation2"' in source
    assert '"$SHARD_INDEX" == "0"' in source
    assert '"$CUDA_DEVICE" == "0"' in source
    assert '"$CUDA_DEVICE" == "1"' in source


def test_binding_classification_is_exact_and_pairwise_disjoint() -> None:
    preparer = _load_operator("deform360_v5_preparer_bindings", PREPARER)
    assert preparer.EXPECTED_V5_BINDING_COUNT == 112
    assert preparer.EXPECTED_V5_MIGRATION_KEY_COUNT == 21
    expected_sources = {
        "deform360_dataset_containment_source": (
            "src/bayesian_phystwin/deform360_dataset_containment.py"
        ),
        "deform360_robot_kinematics_source": (
            "src/bayesian_phystwin/deform360_robot_kinematics.py"
        ),
        "frame_zero_semantic_gate_source": (
            "src/bayesian_phystwin/deform360_frame_zero_semantic_gate.py"
        ),
        "held_protocol_lock_operator_source": (
            "scripts/held/prepare_deform360_v5_lock.py"
        ),
        "held_calibration_case_runner_source": (
            "scripts/held/run_deform360_v5_calibration_case.sh"
        ),
        "held_calibration_shard_runner_source": (
            "scripts/held/run_deform360_v5_calibration_shard.sh"
        ),
        "held_calibration_outcome_driver_source": (
            "scripts/held/run_deform360_v5_calibration_outcomes.py"
        ),
        "held_confirmation_case_runner_source": (
            "scripts/held/run_deform360_v5_confirmation_case.sh"
        ),
        "held_confirmation_shard_runner_source": (
            "scripts/held/run_deform360_v5_confirmation_shard.sh"
        ),
        "held_confirmation_outcome_driver_source": (
            "scripts/held/run_deform360_v5_confirmation_outcomes.py"
        ),
    }
    for key, path in expected_sources.items():
        assert preparer.LOCAL_FILE_BINDINGS[key] == path
    assert "robot_kinematics_window_contract" in preparer.LOCAL_CONTRACT_BINDING_KEYS
    assert "frame_zero_semantic_gate_contract" in preparer.LOCAL_CONTRACT_BINDING_KEYS
    assert preparer.V5_PINNED_EXTERNAL_BINDING_KEYS == {
        "frame_zero_siglip2_model_tree",
        "frame_zero_siglip2_revision_literal",
        "frame_zero_siglip2_transformers_sources",
        "held_frozen_runtime_manifest",
        "python_pip_freeze_sorted",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in preparer.V5_PINNED_EXTERNAL_BINDING_VALUES.values()
    )
    assert "v2_design_withdrawal_report" in preparer.V5_ONLY_BINDING_KEYS
    assert "v3_prelock_boundary_incident_report" in preparer.V5_ONLY_BINDING_KEYS
    assert "v4_execution_withdrawal_report" in preparer.V5_ONLY_BINDING_KEYS
    assert "held_frozen_runtime_manifest" in preparer.V5_ONLY_BINDING_KEYS
    assert "v2_calibration_lock" not in preparer.V5_ONLY_BINDING_KEYS
    groups = (
        set(preparer.INHERITED_EXTERNAL_BINDING_KEYS),
        set(preparer.V5_PINNED_EXTERNAL_BINDING_KEYS),
        set(preparer.LOCAL_FILE_BINDINGS),
        set(preparer.LOCAL_CONTRACT_BINDING_KEYS),
        set(preparer.METHOD_PROVENANCE_BINDING_KEYS),
        {
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
            "v4_execution_withdrawal_report",
        },
    )
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert group.isdisjoint(other)
    assert len(set().union(*groups)) == preparer.EXPECTED_V5_BINDING_COUNT
    assert (
        len(preparer.V5_ONLY_BINDING_KEYS) == preparer.EXPECTED_V5_MIGRATION_KEY_COUNT
    )


def test_preparer_rejects_semantic_export_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin.deform360_frame_zero_semantic_gate import (
        FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
        FRAME_ZERO_SIGLIP2_MODEL_REVISION,
        FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256,
        FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256,
    )

    preparer = _load_operator("deform360_v5_preparer_semantic_exports", PREPARER)
    exported = {
        "model_tree_sha256": FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256,
        "model_revision": FRAME_ZERO_SIGLIP2_MODEL_REVISION,
        "transformers_source_aggregate_sha256": (
            FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256
        ),
        "semantic_gate_contract_sha256": FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256,
    }
    expected_external = {
        "frame_zero_siglip2_model_tree": FRAME_ZERO_SIGLIP2_MODEL_TREE_SHA256,
        "frame_zero_siglip2_revision_literal": hashlib.sha256(
            FRAME_ZERO_SIGLIP2_MODEL_REVISION.encode("ascii")
        ).hexdigest(),
        "frame_zero_siglip2_transformers_sources": (
            FRAME_ZERO_SIGLIP2_TRANSFORMERS_SOURCE_AGGREGATE_SHA256
        ),
    }
    assert {
        key: preparer.V5_PINNED_EXTERNAL_BINDING_VALUES[key]
        for key in expected_external
    } == expected_external
    assert (
        preparer.V5_PINNED_SEMANTIC_GATE_CONTRACT_SHA256
        == FRAME_ZERO_SEMANTIC_GATE_CONTRACT_SHA256
    )
    preparer._require_deployed_semantic_binding_exports(**exported)

    for key in expected_external:
        drifted = dict(expected_external)
        drifted[key] = "0" * 64
        monkeypatch.setattr(preparer, "V5_PINNED_EXTERNAL_BINDING_VALUES", drifted)
        with pytest.raises(ValueError, match="SigLIP2 bindings diverge"):
            preparer._require_deployed_semantic_binding_exports(**exported)

    monkeypatch.setattr(
        preparer,
        "V5_PINNED_EXTERNAL_BINDING_VALUES",
        expected_external,
    )
    monkeypatch.setattr(
        preparer,
        "V5_PINNED_SEMANTIC_GATE_CONTRACT_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="contract binding diverges"):
        preparer._require_deployed_semantic_binding_exports(**exported)


def test_exact_v2_withdrawal_report_identity() -> None:
    preparer = _load_operator("deform360_v5_preparer_report_identity", PREPARER)
    report, encoded, file_sha256, artifact_sha256 = (
        preparer._expected_v2_withdrawal_report()
    )
    assert (
        file_sha256
        == "a7cf04337dbdccc1e3e2165f89b7c51bb25b53bc3c89dc54ddbdf7b5df5dadb3"
    )
    assert artifact_sha256 == (
        "30fe54df7db030aa34481fb87852dc7517a1f2d8f7dd3bb42641208fd0573a08"
    )
    assert hashlib.sha256(encoded).hexdigest() == file_sha256
    assert report["disposition"] == "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION"
    assert report["protocol_id"] == "deform360-held-online-belief-v2"
    assert report["replacement_protocol_id"] == "deform360-held-online-belief-v3"
    assert set(report["execution_counts"].values()) == {0}
    assert set(report["reuse"].values()) == {False}
    boundary = report["information_boundary"]
    assert boundary["source_only_evidence"] is True
    assert set(
        value for key, value in boundary.items() if key != "source_only_evidence"
    ) == {False}


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "disposition", "ABANDONED_PREOUTCOME"),
        ("execution_counts", "prediction_count", 1),
        ("existence_evidence", "held_root_exists", True),
        ("reuse", "v2_artifacts_reused_by_v3", True),
        ("information_boundary", "target_or_outcome_path_accessed", True),
    ],
)
def test_v2_withdrawal_report_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str | None,
    field: str,
    value: object,
) -> None:
    preparer = _load_operator(f"deform360_v5_preparer_tamper_{field}", PREPARER)
    report_path = tmp_path / "v2-design-withdrawal-report.json"
    absent_v2_root = tmp_path / "absent-held-v2"
    monkeypatch.setattr(preparer, "_CANONICAL_V2_WITHDRAWAL_REPORT", report_path)
    monkeypatch.setattr(preparer, "_CANONICAL_HELD_V2_ROOT", absent_v2_root)
    monkeypatch.setattr(preparer, "_require_exact_mode_0400", lambda *_args: None)
    report, _encoded, _file_sha256, _artifact_sha256 = (
        preparer._expected_v2_withdrawal_report()
    )
    changed = copy.deepcopy(report)
    if section is None:
        changed[field] = value
    else:
        changed[section][field] = value
    changed["artifact_sha256"] = preparer._artifact_sha256(changed)
    report_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.chmod(0o400)
    with pytest.raises(ValueError, match="not exact canonical JSON"):
        preparer._validate_v2_withdrawal_report(report_path)


def test_exact_v3_boundary_incident_report_identity_and_scope() -> None:
    preparer = _load_operator("deform360_v5_preparer_v3_incident_identity", PREPARER)
    report, encoded, file_sha256, artifact_sha256 = (
        preparer._expected_v3_boundary_incident_report()
    )
    assert file_sha256 == (
        "b344a99cb6c4de4fe16b186f85f914dc7d2a3e049eac90b3ae40b56381c4505d"
    )
    assert artifact_sha256 == (
        "58d087993ec069e6d0dc05d507c2edd0833a41d05bc2135f3191264d61d5327e"
    )
    assert hashlib.sha256(encoded).hexdigest() == file_sha256
    assert report["protocol_id"] == "deform360-held-online-belief-v3"
    assert report["replacement_protocol_id"] == "deform360-held-online-belief-v4"
    assert report["disposition"] == "WITHDRAWN_BEFORE_LOCK_AND_PREDICTION"
    assert set(report["formal_protocol_execution_counts"].values()) == {0}
    assert (
        "excludes the separately disclosed rg content scanner"
        in report["formal_protocol_execution_scope"]
    )
    assert report["existence_evidence"] == {
        "canonical_held_root": (
            "/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v3"
        ),
        "evidence_scope": "filesystem existence check after the pre-lock incident",
        "held_root_exists": False,
    }
    incident = report["incident"]
    assert incident["search"] == {
        "program": "rg",
        "mode": "-l",
        "search_terms": [
            "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
            "facebook/cotracker3-scaled-offline",
        ],
        "search_roots": [
            "/mnt/corsair/florianpfaff/bpt-online-belief-v1",
            "/mnt/corsair/florianpfaff/deform360-processing-deps",
            "/mnt/lexar4tb/datasets/deform360",
        ],
        "stdout_consumer": "head",
        "stdout_maximum_line_count": 100,
    }
    assert incident["scanner_scope"] == {
        "may_have_opened_any_regular_file_under_search_roots": True,
        "protected_file_open_status": "NOT_CLAIMED",
    }
    assert incident["returned_output"] == {
        "only_matching_absolute_filenames": True,
        "included_unrelated_171_outcome_or_log_paths": True,
        "payload_bytes_returned": False,
        "metrics_returned": False,
        "labels_returned": False,
        "arrays_returned": False,
        "payload_values_returned": False,
    }
    assert report["information_boundary"] == {
        "content_scanner_may_have_opened_any_regular_file_under_search_roots": True,
        "protected_file_open_status": "NOT_CLAIMED",
        "held_cohort_payload_content_or_value_returned_to_research_agent": False,
        "outcome_metric_label_array_or_value_returned_to_research_agent": False,
        "method_or_gate_choice_used_outcome_values": False,
        "stdout_was_filename_only": True,
    }
    assert "outcome_payloads_accessed" not in json.dumps(report, sort_keys=True)


def test_exact_v4_execution_withdrawal_report_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = _load_operator("deform360_v5_preparer_v4_withdrawal", PREPARER)
    sealer = _load_operator("deform360_v4_withdrawal_fixture", V4_WITHDRAWAL_SEALER)
    report, payload = sealer._artifact(sealer.expected_unsigned_report())
    report_path = tmp_path / "v4-execution-withdrawal-report.json"
    report_path.write_bytes(payload)
    report_path.chmod(0o400)
    monkeypatch.setattr(
        preparer,
        "_CANONICAL_V4_EXECUTION_WITHDRAWAL_REPORT",
        report_path,
    )

    assert hashlib.sha256(payload).hexdigest() == (
        preparer.EXPECTED_V4_REPORT_FILE_SHA256
    )
    assert report["artifact_sha256"] == (preparer.EXPECTED_V4_REPORT_ARTIFACT_SHA256)
    assert preparer._validate_v4_execution_withdrawal_report(report_path) == (
        preparer.EXPECTED_V4_REPORT_FILE_SHA256
    )
    assert report["execution_counts"]["formal_physical_prediction_count"] == 0
    assert report["execution_counts"]["formal_online_prediction_count"] == 0
    assert report["reuse"]["v5_requires_fresh_absent_held_root"] is True


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("disposition",), "ABANDONED_PREOUTCOME"),
        (("formal_protocol_execution_counts", "prediction_count"), 1),
        (("existence_evidence", "held_root_exists"), True),
        (
            (
                "incident",
                "scanner_scope",
                "may_have_opened_any_regular_file_under_search_roots",
            ),
            False,
        ),
        (("incident", "returned_output", "payload_bytes_returned"), True),
        (
            ("information_boundary", "method_or_gate_choice_used_outcome_values"),
            True,
        ),
    ],
)
def test_v3_boundary_incident_report_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    preparer = _load_operator(
        "deform360_v5_preparer_v3_incident_tamper_" + "_".join(field_path),
        PREPARER,
    )
    report_path = tmp_path / "v3-prelock-boundary-incident-report.json"
    monkeypatch.setattr(preparer, "_CANONICAL_V3_BOUNDARY_INCIDENT_REPORT", report_path)
    monkeypatch.setattr(
        preparer, "_CANONICAL_HELD_V3_ROOT", tmp_path / "absent-held-v3"
    )
    monkeypatch.setattr(preparer, "_require_exact_mode_0400", lambda *_args: None)
    report, _encoded, _file_sha256, _artifact_sha256 = (
        preparer._expected_v3_boundary_incident_report()
    )
    changed = copy.deepcopy(report)
    target = changed
    for component in field_path[:-1]:
        target = target[component]
    target[field_path[-1]] = value
    changed["artifact_sha256"] = preparer._artifact_sha256(changed)
    report_path.write_text(
        json.dumps(changed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.chmod(0o400)
    with pytest.raises(ValueError, match="not exact canonical JSON"):
        preparer._validate_v3_boundary_incident_report(report_path)


def test_exact_mode_0400_check_rejects_every_write_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = _load_operator("deform360_v5_preparer_mode", PREPARER)
    monkeypatch.setattr(
        preparer.os,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFREG | 0o400),
    )
    preparer._require_exact_mode_0400(Path("unused"), "artifact")
    monkeypatch.setattr(
        preparer.os,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFREG | 0o600),
    )
    with pytest.raises(ValueError, match="mode is not exactly 0400"):
        preparer._require_exact_mode_0400(Path("unused"), "artifact")


def test_outcome_environment_is_an_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _load_operator("deform360_v5_outcome_environment", OUTCOME_DRIVER)
    assert outcomes.EXPECTED_V5_BINDING_COUNT == 112
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("BASH_ENV", "/untrusted/bash-env")
    monkeypatch.setenv("TARGET_DATA_PATH", "/protected")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    environment = outcomes._minimal_environment(include_outcome_runtime=True)
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == ("/nonexistent/bpt-held-v5-pycache")
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert "PYTHONPATH" not in environment
    assert "BASH_ENV" not in environment
    assert "TARGET_DATA_PATH" not in environment
    lock_environment = outcomes._lock_verifier_environment()
    assert lock_environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert "PYTHONPATH" not in lock_environment
    assert "BASH_ENV" not in lock_environment
    preparer = _load_operator("deform360_v5_preparer_environment", PREPARER)
    assert lock_environment == preparer._PINNED_OPERATOR_ENVIRONMENT
    assert preparer._PYCACHE_PREFIX == "/nonexistent/bpt-held-v5-pycache"

    confirmation = _load_operator(
        "deform360_v5_confirmation_outcome_environment",
        CONFIRMATION_OUTCOME_DRIVER,
    )
    confirmation_environment = confirmation._minimal_environment(
        include_outcome_runtime=True
    )
    assert confirmation_environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert confirmation_environment["PYTHONSAFEPATH"] == "1"


def test_isolated_python_bootstrap_ignores_cwd_and_pythonpath(
    tmp_path: Path,
) -> None:
    malicious = tmp_path / "malicious"
    malicious.mkdir()
    sentinel = tmp_path / "startup-executed"
    (malicious / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    fake_package = malicious / "bayesian_phystwin"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text(
        "raise RuntimeError('malicious cwd package imported')\n",
        encoding="utf-8",
    )
    script = (
        "import pathlib,sys; root=sys.argv.pop(1); "
        "sys.path.insert(0,root); import bayesian_phystwin; "
        "print(pathlib.Path(bayesian_phystwin.__file__).resolve())"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")],
        check=False,
        cwd=malicious,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(malicious)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()
    assert Path(completed.stdout.strip()).is_relative_to(ROOT / "src")


@pytest.mark.parametrize(
    "driver",
    (OUTCOME_DRIVER, CONFIRMATION_OUTCOME_DRIVER),
)
def test_outcome_driver_rejects_symlinked_dataset_ancestry(
    tmp_path: Path,
    driver: Path,
) -> None:
    outcomes = _load_operator(f"symlink_ancestry_{driver.stem}", driver)
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    outside_object = tmp_path / "outside" / "object"
    (outside_object / "episode_0001").mkdir(parents=True)
    (aligned / "object").symlink_to(outside_object, target_is_directory=True)
    with pytest.raises(ValueError, match="linked, absent, or non-canonical"):
        outcomes._aligned_episode_path(str(aligned), "object-ep0001")


@pytest.mark.parametrize(
    ("driver", "event_prefix"),
    (
        (OUTCOME_DRIVER, "CALIBRATION"),
        (CONFIRMATION_OUTCOME_DRIVER, "CONFIRMATION"),
    ),
)
def test_outcome_callback_rejects_replaced_root_inode(
    tmp_path: Path,
    driver: Path,
    event_prefix: str,
) -> None:
    outcomes = _load_operator(f"root_inode_{driver.stem}", driver)
    root = tmp_path / "outcomes"
    root.mkdir()
    identity = outcomes._require_directory_identity(root, None, "outcomes root")
    called: list[bool] = []
    callback = outcomes._ProgressCallback(
        case_name="object-ep0001",
        operation="create",
        callback=lambda: called.append(True),
        outcomes_root=root,
        outcomes_root_identity=identity,
    )
    root.rename(tmp_path / "old-outcomes")
    root.mkdir()
    with pytest.raises(ValueError, match="inode changed"):
        callback()
    assert called == []
    assert event_prefix in driver.read_text(encoding="utf-8")


def test_isolated_operators_cannot_consult_adjacent_pyc() -> None:
    for path in (PREPARER, OUTCOME_DRIVER, CONFIRMATION_OUTCOME_DRIVER):
        source = path.read_text(encoding="utf-8")
        assert '"/nonexistent/bpt-held-v5-pycache"' in source
    for path in (OUTCOME_DRIVER, CONFIRMATION_OUTCOME_DRIVER):
        source = path.read_text(encoding="utf-8")
        assert '"-B"' in source
        assert '"-X"' in source
        assert 'f"pycache_prefix={_PYCACHE_PREFIX}"' in source
        assert "sys.flags.dont_write_bytecode == 1" in source
        assert "sys.pycache_prefix == _PYCACHE_PREFIX" in source
    preparer_source = PREPARER.read_text(encoding="utf-8")
    assert "sys.flags.dont_write_bytecode == 1" in preparer_source
    assert "sys.pycache_prefix == _PYCACHE_PREFIX" in preparer_source


def test_outcome_barrier_self_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(OUTCOME_DRIVER), "--self-check"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"event": "SELF_CHECK_PASSED"' in completed.stdout


def test_confirmation_outcome_barrier_self_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONFIRMATION_OUTCOME_DRIVER), "--self-check"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"event": "SELF_CHECK_PASSED"' in completed.stdout
    assert '"case_count": 6' in completed.stdout
    assert '"decision": "CONFIRMED"' in completed.stdout
    assert '"decision": "NOT_CONFIRMED"' in completed.stdout


def test_confirmation_promotion_is_write_once_and_target_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _load_operator(
        "deform360_v5_confirmation_promotion",
        CONFIRMATION_OUTCOME_DRIVER,
    )
    root = tmp_path / "held-v5"
    root.mkdir()
    calibration_lock = root / "calibration-lock.json"
    calibration_lock.write_text("calibration\n", encoding="utf-8")
    calibration_lock.chmod(0o400)
    calibration_root = root / "calibration"
    calibration_root.mkdir()
    calibration_decision = calibration_root / "calibration-gate-decision.json"
    calibration_decision.write_text("GO\n", encoding="utf-8")
    calibration_decision.chmod(0o400)
    confirmation_lock = root / "confirmation-lock.json"
    monkeypatch.setattr(outcomes, "_CANONICAL_HELD_ROOT", root)
    monkeypatch.setattr(outcomes, "_CANONICAL_CALIBRATION_LOCK", calibration_lock)
    monkeypatch.setattr(
        outcomes,
        "_CANONICAL_CALIBRATION_DECISION",
        calibration_decision,
    )
    monkeypatch.setattr(
        outcomes,
        "_CANONICAL_CONFIRMATION_LOCK",
        confirmation_lock,
    )
    frozen: list[tuple[Path, int, bool]] = []
    monkeypatch.setattr(
        outcomes.os,
        "chmod",
        lambda path, mode, *, follow_symlinks: frozen.append(
            (path, mode, follow_symlinks)
        ),
    )
    monkeypatch.setattr(outcomes, "_require_mode_0400", lambda *_args: None)
    arguments = outcomes.DriverArguments(
        deployed_code="unused",
        held_root=str(root),
        calibration_lock=str(calibration_lock),
        calibration_decision=str(calibration_decision),
        confirmation_lock=str(confirmation_lock),
        online_seals=(),
        promote_only=True,
        dry_run_barrier_only=False,
    )
    calls: list[tuple[Path, str, str]] = []

    class Protocol:
        @staticmethod
        def create_confirmation_protocol_lock(
            output: Path,
            parent: str,
            decision: str,
        ) -> dict[str, object]:
            calls.append((output, parent, decision))
            value = {
                "stage": "confirmation",
                "confirmation_access_authorized": True,
                "artifact_sha256": "a" * 64,
            }
            output.write_text(json.dumps(value), encoding="utf-8")
            return value

        @staticmethod
        def load_held_protocol_lock(path: Path) -> dict[str, object]:
            return json.loads(path.read_text(encoding="utf-8"))

    assert outcomes._promote_confirmation_lock(arguments, Protocol()) == 0
    assert calls == [
        (confirmation_lock, str(calibration_lock), str(calibration_decision))
    ]
    assert frozen == [(confirmation_lock, 0o400, False)]
    with pytest.raises(ValueError, match="already exists"):
        outcomes._promote_confirmation_lock(arguments, Protocol())


def test_noncanonical_seal_is_rejected_before_authorizer(tmp_path: Path) -> None:
    outcomes = _load_operator("deform360_v5_outcome_bad_seal", OUTCOME_DRIVER)
    arguments = outcomes._mock_arguments(tmp_path / "held", dry=True)
    first_case = f"mock-object-{0:02d}-ep{0:04d}"
    wrong = tmp_path / "protected" / "outcome.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("must not be opened\n", encoding="utf-8")
    assignments = list(arguments.online_seals)
    assignments[0] = f"{first_case}={wrong}"
    arguments = outcomes.DriverArguments(
        **{
            **arguments.__dict__,
            "online_seals": tuple(assignments),
        }
    )
    events: list[str] = []

    class Protocol:
        PROTOCOL_ID = outcomes.EXPECTED_PROTOCOL_ID
        CALIBRATION_CASE_NAMES = tuple(
            f"mock-object-{index:02d}-ep{index:04d}" for index in range(15)
        )

        @staticmethod
        def authorize_outcome_phase(*_args, **_kwargs):
            events.append("authorize")
            raise AssertionError("authorizer must not see a noncanonical path")

    with pytest.raises(ValueError, match="canonical held path"):
        outcomes._execute_driver(
            arguments,
            Protocol(),
            lambda: (_ for _ in ()).throw(AssertionError("post API imported")),
        )
    assert events == []


def test_fresh_outcome_root_rejects_symlink_escape(tmp_path: Path) -> None:
    outcomes = _load_operator("deform360_v5_outcome_symlink", OUTCOME_DRIVER)
    calibration = tmp_path / "held" / "calibration"
    calibration.mkdir(parents=True)
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    outcome_root = calibration / "outcomes"
    outcome_root.symlink_to(escaped, target_is_directory=True)
    layout = outcomes.HeldLayout(
        root=tmp_path / "held",
        lock=tmp_path / "held" / "calibration-lock.json",
        seal_paths={},
        outcomes_root=outcome_root,
        evidence_path=calibration / "calibration-score-evidence.json",
        decision_path=calibration / "calibration-gate-decision.json",
    )
    with pytest.raises(ValueError, match="outcome root already exists"):
        outcomes._prepare_fresh_gate_outputs_after_barrier(layout)
    assert list(escaped.iterdir()) == []


def test_bootstrap_preparer_digest_is_finalized_consistently() -> None:
    expected = _sha256(PREPARER)
    observed: list[str] = []
    for path in (
        CASE_RUNNER,
        SHARD_RUNNER,
        OUTCOME_DRIVER,
        CONFIRMATION_CASE_RUNNER,
        CONFIRMATION_SHARD_RUNNER,
        CONFIRMATION_OUTCOME_DRIVER,
    ):
        match = re.search(
            r"(?:readonly )?EXPECTED_LOCK_OPERATOR_SHA256\s*=\s*"
            r'(?:\(\s*)?["\']([^"\']+)',
            path.read_text(encoding="utf-8"),
        )
        assert match is not None
        observed.append(match.group(1))
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in observed)
    assert observed == [expected] * 6


def test_v5_protocol_interface_is_exact_when_active() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from bayesian_phystwin import deform360_held_protocol as protocol

    preparer = _load_operator("deform360_v5_preparer_protocol_interface", PREPARER)
    if protocol.PROTOCOL_ID != "deform360-held-online-belief-v5":
        pytest.skip("historical v5 operators are no longer the active protocol")
    required = set(protocol.REQUIRED_IMMUTABLE_BINDING_KEYS)
    expected = (
        set(preparer.INHERITED_EXTERNAL_BINDING_KEYS)
        | set(preparer.V5_PINNED_EXTERNAL_BINDING_KEYS)
        | set(preparer.LOCAL_FILE_BINDINGS)
        | set(preparer.LOCAL_CONTRACT_BINDING_KEYS)
        | set(preparer.METHOD_PROVENANCE_BINDING_KEYS)
        | {
            "v1_preoutcome_feasibility_report",
            "v2_design_withdrawal_report",
            "v3_prelock_boundary_incident_report",
            "v4_execution_withdrawal_report",
        }
    )
    assert required == expected
    assert len(required) == preparer.EXPECTED_V5_BINDING_COUNT


def test_v5_lock_creation_requires_a_fresh_root() -> None:
    source = PREPARER.read_text(encoding="utf-8")
    assert "if not arguments.verify_existing_lock:" in source
    assert "[entry.name for entry in root_entries] == [code.name]" in source
    assert (
        "fresh held-v5 root contains anything except the deployed code snapshot"
        in source
    )
