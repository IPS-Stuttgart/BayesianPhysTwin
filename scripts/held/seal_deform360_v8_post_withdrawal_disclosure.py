#!/usr/bin/env python3
"""Seal the cumulative v8.1 recovery disclosure before attempt 5.

This operator reads no protected array, image, mask, metric, prediction, or
score payload.  It parses only already-sealed metadata JSON, checks immutable
byte identities and archive metadata, then writes one fixed report for the
prospective lock.  It bars reuse of every v7, attempt-3, and attempt-4
execution product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping


PROTOCOL_ID = "deform360-held-online-belief-v8.1"
ARTIFACT_KIND = "Deform360HeldV81Attempt5RecoveryDisclosure"
_QUALIFICATION_KIND = "Deform360ResourceLifecycleQualificationEvidenceV2"
_QUALIFICATION_COMPLETION_KIND = (
    "Deform360ResourceLifecycleQualificationIntegrityCompletionV2"
)
_QUALIFICATION_ID = "deform360-nerfstudio-resource-lifecycle-qualification-v2"
_QUALIFICATION_ANALYZER_SHA256 = (
    "43056e39ff7ea5f760f18420784db0edbb75523031dba7f3a19eca0c6951c128"
)

_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_V7_ROOT = _BASE / "held-v7"
_V8_ROOT = _BASE / "held-v8"
_OUTPUT = _V8_ROOT / "post-withdrawal-development-use-disclosure.json"
_ATTEMPT3_ARCHIVE = _BASE / "held-v8-attempt-3-withdrawn-postbarrier"
_ATTEMPT3_REPORT = _ATTEMPT3_ARCHIVE / "execution-withdrawal-postbarrier-attempt3.json"
_ATTEMPT3_POINTER = _BASE / "held-v8-attempt-3-withdrawal-pointer.json"
_ATTEMPT3_COMPLETION = _BASE / "held-v8-attempt-3-withdrawal-integrity-completion.json"
_ATTEMPT4_ARCHIVE = _BASE / "held-v8-attempt-4-withdrawn-postbarrier"
_ATTEMPT4_REPORT = _ATTEMPT4_ARCHIVE / "execution-withdrawal-postbarrier-attempt4.json"
_ATTEMPT4_POINTER = _BASE / "held-v8-attempt-4-withdrawal-pointer.json"
_ATTEMPT4_COMPLETION = _BASE / "held-v8-attempt-4-withdrawal-integrity-completion.json"
_ATTEMPT4_LAUNCHER = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-orchestration/"
    "calibration-outcome-c88168c-20260722T1847"
)

_V7_FILE_NAMES = frozenset(
    {
        "v7_outcome_withdrawal_report",
        "retired_case_official_target",
        "retired_case_online_prediction",
        "retired_case_online_prediction_seal",
    }
)
_ATTEMPT3_FILE_NAMES = frozenset(
    {
        "v8_attempt3_withdrawal_report",
        "v8_attempt3_withdrawal_pointer",
        "v8_attempt3_withdrawal_integrity_completion",
    }
)
_ATTEMPT4_FILE_NAMES = frozenset(
    {
        "v8_attempt4_withdrawal_report",
        "v8_attempt4_withdrawal_pointer",
        "v8_attempt4_withdrawal_integrity_completion",
    }
)
_EXPECTED_FILES: Mapping[str, tuple[Path, int | None, str]] = {
    "v7_outcome_withdrawal_report": (
        _V7_ROOT / "v7-outcome-withdrawal-report.json",
        10_295,
        "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3",
    ),
    "retired_case_official_target": (
        _V7_ROOT / "calibration/outcomes/002-rope-silk-ep0003/official_target.npz",
        536_992,
        "850a894f1e1eb447fddbb877ac2fbf38225e97514a1218cc7ea1182212f471a8",
    ),
    "retired_case_online_prediction": (
        _V7_ROOT
        / "calibration/cases/002-rope-silk-ep0003/online/online_prediction.npz",
        994_650,
        "ecae2a595b50c91bf842c3e86eb38559eec0ad43aeeba40da2dd8a9098a31f8d",
    ),
    "retired_case_online_prediction_seal": (
        _V7_ROOT
        / "calibration/cases/002-rope-silk-ep0003/online/online_prediction_seal.json",
        3_684,
        "afac640547cf4f0de1f168dd4642b841ee96cc274b5e47401aadd4e361255814",
    ),
    "v8_attempt3_withdrawal_report": (
        _ATTEMPT3_REPORT,
        None,
        "6d9c62606d18744d275df51fd08e041205bf15b38175d74c69690eafd511054b",
    ),
    "v8_attempt3_withdrawal_pointer": (
        _ATTEMPT3_POINTER,
        None,
        "75acc7e9535f41528d22739ae8eeb5a0a2247c0fe63c097ad1da2859d7b33246",
    ),
    "v8_attempt3_withdrawal_integrity_completion": (
        _ATTEMPT3_COMPLETION,
        None,
        "f3d1e8a6670484c81ac04743bcdb020cdee3fba02229a64844a8a9c9f4b8b989",
    ),
    "v8_attempt4_withdrawal_report": (
        _ATTEMPT4_REPORT,
        1_101_388,
        "24c7c7f154c6985c5c5832222a0872d62798e282af3c0f7494e70b8dfc100b5a",
    ),
    "v8_attempt4_withdrawal_pointer": (
        _ATTEMPT4_POINTER,
        3_828,
        "3de7c79bf4d4949100f6bd90b1bc6da306d4b57090b70ef7606accefc9901665",
    ),
    "v8_attempt4_withdrawal_integrity_completion": (
        _ATTEMPT4_COMPLETION,
        3_419,
        "315c62fa0e4b621e07db053950e9d26ab1abcb6a2f71a9347ec8d1526d8ad984",
    ),
}

_ATTEMPT4_ARCHIVE_INTEGRITY = {
    "path": str(_ATTEMPT4_ARCHIVE),
    "root_mode_octal": "0500",
    "fully_nonwritable": True,
    "postseal_noncode_inventory_sha256": (
        "1ab11d7a3e841530e0d8c994327b9eca26a20a896f73cfa3d76e5c6935cdca5c"
    ),
    "postseal_noncode_entry_count": 1915,
}
_ATTEMPT4_STATUS = (
    "withdrawn-postbarrier-during-third-target-reconstruction-before-barrier2-or-score"
)
_ATTEMPT4_DISPOSITION = (
    "WITHDRAWN_AFTER_TWO_TARGET_X0_QUERY_PAIRS_DURING_THIRD_TARGET_"
    "RECONSTRUCTION_BEFORE_SECOND_BARRIER_OR_SCORE"
)
_ATTEMPT4_REPORT_ARTIFACT_SHA256 = (
    "3e2f7be514d0ab2776905f3bae7fe5e474b5fdc57a7c64e59de33adf97f79c5a"
)
_ATTEMPT4_POINTER_ARTIFACT_SHA256 = (
    "3bd025ec4ac6fd9a7b57f7ccacf4f44cee3b6aa0c763dc081f54474b129af4b2"
)
_ATTEMPT4_COMPLETION_ARTIFACT_SHA256 = (
    "62128be06dfb1e181c3d6cd849ccd34c5cd37e3769c6b917811676a05da37332"
)

_ATTEMPT3_ARCHIVE_INTEGRITY = {
    "path": str(_ATTEMPT3_ARCHIVE),
    "root_mode_octal": "0500",
    "fully_nonwritable": True,
    "postseal_noncode_inventory_sha256": (
        "5d398e998e2b738db545ffefd254712c6822017cfc5be6e7de435d5883c8c4c8"
    ),
    "postseal_noncode_entry_count": 1466,
}

_ATTEMPT3_DEPLOYED_CODE_BINDING = {
    "path": "code-9ad7ad2b385f7abc5e8c42081a41018980dd3827",
    "git_head": "9ad7ad2b385f7abc5e8c42081a41018980dd3827",
    "head_text_sha256": (
        "b5e33f85b96a0026147040044c288ef5c6ff3e60ca9b74743f904b49f78b79f1"
    ),
    "git_tree_record_count": 950,
    "git_tree_manifest_sha256": (
        "445f325dca5710c9873951445cb26107966e5344333edd8a69ac380e50e09546"
    ),
}

_POST_WITHDRAWAL_DEVELOPMENT_HASHES = {
    "scratch_frozen_field_source_sha256": (
        "e106611d9f5e9c6125b5c4c1704db06703108f1ce635d55e6e15d8c8b3a32822"
    ),
    "scratch_query_development_source_sha256": (
        "3f008ef9f9b6fe52c6a36e1939a56ec35e160912efae44ba5a12d11a59a572ae"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _artifact(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    unsigned = dict(value)
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    signed = {**unsigned, "artifact_sha256": digest}
    return signed, _canonical_json(signed)


def _bind_expected_regular_file(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str,
    role: str,
) -> dict[str, Any]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = os.lstat(absolute)
    _require(not stat.S_ISLNK(before.st_mode), f"{role} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{role} is not a regular file")
    _require(before.st_nlink == 1, f"{role} is hard-linked")
    _require(absolute.resolve() == absolute, f"{role} path is non-canonical")
    _require(
        stat.S_IMODE(before.st_mode) == 0o400,
        f"{role} mode is not exactly 0400",
    )
    if expected_size is not None:
        _require(before.st_size == expected_size, f"{role} size changed")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            f"{role} changed while opening",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(absolute)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ),
        f"{role} changed while hashing",
    )
    observed_sha256 = digest.hexdigest()
    _require(observed_sha256 == expected_sha256, f"{role} SHA-256 changed")
    return {
        "path": os.fspath(absolute),
        "sha256": observed_sha256,
        "size_bytes": before.st_size,
        "mode_octal": "0400",
    }


def _load_bound_json(
    path: Path, record: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
    finally:
        os.close(descriptor)
    _require(
        len(payload) == record.get("size_bytes")
        and hashlib.sha256(payload).hexdigest() == record.get("sha256"),
        f"{role} changed while loading",
    )
    try:
        artifact = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(artifact, dict), f"{role} is not a JSON object")
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256", None)
    observed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _require(claimed == observed, f"{role} artifact signature changed")
    return artifact


def _bind_attempt4_lineage(
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = _load_bound_json(
        _ATTEMPT4_REPORT,
        bindings["v8_attempt4_withdrawal_report"],
        role="attempt-4 withdrawal report",
    )
    pointer = _load_bound_json(
        _ATTEMPT4_POINTER,
        bindings["v8_attempt4_withdrawal_pointer"],
        role="attempt-4 withdrawal pointer",
    )
    completion = _load_bound_json(
        _ATTEMPT4_COMPLETION,
        bindings["v8_attempt4_withdrawal_integrity_completion"],
        role="attempt-4 withdrawal completion",
    )
    expected = (
        (
            report,
            "Deform360HeldV81Attempt4PostBarrierWithdrawalReport",
            _ATTEMPT4_STATUS,
            _ATTEMPT4_REPORT_ARTIFACT_SHA256,
        ),
        (
            pointer,
            "Deform360HeldV81Attempt4WithdrawalPointer",
            _ATTEMPT4_STATUS,
            _ATTEMPT4_POINTER_ARTIFACT_SHA256,
        ),
        (
            completion,
            "Deform360HeldV81Attempt4WithdrawalIntegrityCompletion",
            "withdrawal-integrity-complete",
            _ATTEMPT4_COMPLETION_ARTIFACT_SHA256,
        ),
    )
    for artifact, kind, status, digest in expected:
        _require(
            artifact.get("schema_version") == 1
            and artifact.get("artifact_kind") == kind
            and artifact.get("protocol_id") == PROTOCOL_ID
            and artifact.get("execution_attempt") == 4
            and artifact.get("status") == status
            and artifact.get("disposition") == _ATTEMPT4_DISPOSITION
            and artifact.get("artifact_sha256") == digest,
            "attempt-4 withdrawal identity changed",
        )
    archive_state = os.lstat(_ATTEMPT4_ARCHIVE)
    _require(
        stat.S_ISDIR(archive_state.st_mode)
        and not stat.S_ISLNK(archive_state.st_mode)
        and stat.S_IMODE(archive_state.st_mode) == 0o500,
        "attempt-4 archive is not immutable",
    )
    launcher = report.get("durable_launcher_evidence")
    _require(
        isinstance(launcher, Mapping)
        and launcher.get("path") == str(_ATTEMPT4_LAUNCHER)
        and launcher.get("exact_file_allowlist") == ["exit.code", "output.log"]
        and launcher.get("output_log", {}).get("sha256")
        == "9153b50771d8818384d96a77f3502dbbc9494136f679fd25aa6e8208f73bd3e8"
        and launcher.get("exit_code", {}).get("sha256")
        == "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3",
        "attempt-4 launcher binding changed",
    )
    launcher_state = os.lstat(_ATTEMPT4_LAUNCHER)
    _require(
        stat.S_ISDIR(launcher_state.st_mode)
        and not stat.S_ISLNK(launcher_state.st_mode)
        and stat.S_IMODE(launcher_state.st_mode) == 0o500
        and _ATTEMPT4_LAUNCHER.resolve() == _ATTEMPT4_LAUNCHER
        and sorted(child.name for child in _ATTEMPT4_LAUNCHER.iterdir())
        == ["exit.code", "output.log"],
        "attempt-4 launcher is not an immutable allowlist",
    )
    for name, size in (("exit.code", 2), ("output.log", 1_168_519_909)):
        source = _ATTEMPT4_LAUNCHER / name
        observed = os.lstat(source)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and stat.S_IMODE(observed.st_mode) == 0o400
            and observed.st_nlink == 1
            and observed.st_size == size,
            f"attempt-4 launcher {name} metadata changed",
        )
    execution = report.get("execution_boundary")
    information = report.get("information_boundary")
    _require(
        isinstance(execution, Mapping)
        and execution.get("first_cohort_barrier_validated_count") == 1
        and execution.get("official_target_archive_count") == 2
        and execution.get("official_x0_archive_count") == 2
        and execution.get("queried_prediction_seal_count") == 2
        and execution.get("partial_reconstruction_count") == 1
        and execution.get("second_cohort_barrier_validated_count") == 0
        and execution.get("score_evidence_count") == 0
        and execution.get("gate_decision_count") == 0
        and execution.get("confirmation_lock_count") == 0
        and isinstance(information, Mapping)
        and information.get("first_complete_cohort_barrier_crossed") is True
        and information.get("second_complete_cohort_barrier_crossed") is False
        and information.get("score_created_or_read") is False
        and information.get("gate_decision_created_or_read") is False
        and information.get("confirmation_created_or_read") is False,
        "attempt-4 outcome boundary changed",
    )
    return dict(launcher), dict(execution), dict(information)


def _bind_resource_qualification(
    evidence_path: Path,
    completion_path: Path,
    *,
    evidence_sha256: str,
    completion_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evidence_record = _bind_expected_regular_file(
        evidence_path,
        expected_size=None,
        expected_sha256=evidence_sha256,
        role="resource lifecycle qualification",
    )
    completion_record = _bind_expected_regular_file(
        completion_path,
        expected_size=None,
        expected_sha256=completion_sha256,
        role="resource lifecycle qualification completion",
    )
    evidence = _load_bound_json(
        evidence_path, evidence_record, role="resource qualification"
    )
    completion = _load_bound_json(
        completion_path, completion_record, role="resource qualification completion"
    )
    _require(
        evidence.get("schema_version") == 2
        and evidence.get("artifact_kind") == _QUALIFICATION_KIND
        and evidence.get("qualification_id") == _QUALIFICATION_ID
        and evidence.get("status") == "qualified"
        and evidence.get("passed") is True
        and evidence.get("generator_profile") == "same-as-analyzer"
        and evidence.get("physical_gpu_index") == 1
        and evidence.get("host") == "workstation2"
        and evidence.get("phase") == "all"
        and evidence.get("information_boundary", {}).get("formal_held_path_accepted")
        is False
        and evidence.get("information_boundary", {}).get(
            "formal_target_or_outcome_array_read"
        )
        is False,
        "resource qualification identity changed",
    )
    admission = evidence.get("admission")
    _require(
        admission
        == {
            "decision": "admitted",
            "terminal": True,
            "analyzer_outcome": "accepted",
            "analyzer_no_go_interpretation": None,
            "wrapper_inequivalence_proven": False,
            "retry_permitted": False,
            "in_place_reuse_permitted": False,
        },
        "resource qualification is not admission eligible",
    )
    _require(
        completion.get("schema_version") == 2
        and completion.get("artifact_kind") == _QUALIFICATION_COMPLETION_KIND
        and completion.get("qualification_id") == _QUALIFICATION_ID
        and completion.get("status") == "qualification-integrity-complete"
        and completion.get("passed") is True
        and completion.get("terminal_outcome") == "qualified"
        and completion.get("admission_eligible") is True,
        "resource qualification completion changed",
    )
    root = evidence_path.parent
    _require(
        root.parent == Path("/mnt/corsair/florianpfaff")
        and root.name.startswith("bpt-resource-lifecycle-qualification-")
        and completion_path == Path(f"{root}-integrity-completion.json")
        and completion.get("qualification_root") == str(root)
        and completion.get("qualification_root_mode_octal") == "0500"
        and completion.get("qualification_tree_fully_nonwritable") is True,
        "resource qualification root binding changed",
    )

    def bind_signed_completion_input(
        field: str, path: Path, *, role: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        declared = completion.get(field)
        _require(isinstance(declared, Mapping), f"{role} completion binding is absent")
        _require(
            declared.get("path") == str(path)
            and isinstance(declared.get("size_bytes"), int)
            and not isinstance(declared.get("size_bytes"), bool)
            and isinstance(declared.get("sha256"), str)
            and len(str(declared["sha256"])) == 64
            and isinstance(declared.get("artifact_sha256"), str)
            and len(str(declared["artifact_sha256"])) == 64,
            f"{role} completion binding changed",
        )
        observed = _bind_expected_regular_file(
            path,
            expected_size=int(declared["size_bytes"]),
            expected_sha256=str(declared["sha256"]),
            role=role,
        )
        artifact = _load_bound_json(path, observed, role=role)
        _require(
            artifact.get("artifact_sha256") == declared.get("artifact_sha256"),
            f"{role} artifact cross-link changed",
        )
        return observed, artifact

    attempt_path = root / "qualification-attempt.json"
    manifest_path = root / "equivalence/repeat-manifest.json"
    result_path = root / "equivalence/analysis-result.json"
    attempt_record, attempt = bind_signed_completion_input(
        "qualification_attempt", attempt_path, role="resource qualification attempt"
    )
    manifest_record, manifest = bind_signed_completion_input(
        "repeat_manifest", manifest_path, role="resource qualification repeat manifest"
    )
    result_record, result = bind_signed_completion_input(
        "equivalence_result", result_path, role="resource qualification result"
    )
    completion_evidence = completion.get("qualification_evidence")
    _require(
        isinstance(completion_evidence, Mapping)
        and completion_evidence.get("path") == evidence_record["path"]
        and completion_evidence.get("sha256") == evidence_record["sha256"]
        and completion_evidence.get("size_bytes") == evidence_record["size_bytes"]
        and completion_evidence.get("artifact_sha256") == evidence["artifact_sha256"],
        "resource qualification evidence completion cross-link changed",
    )
    _require(
        attempt.get("artifact_kind")
        == "Deform360ResourceLifecycleQualificationAttemptV2"
        and attempt.get("qualification_id") == _QUALIFICATION_ID
        and attempt.get("state") == "canonical-root-consumed-at-creation"
        and attempt.get("output_root") == str(root)
        and attempt.get("code_revision")
        == evidence.get("runtime_bindings", {}).get("code", {}).get("head")
        and attempt.get("root_consumption_policy")
        == evidence.get("root_consumption_policy"),
        "resource qualification attempt marker changed",
    )
    _require(
        manifest.get("artifact_kind") == "Deform360ResourceLifecycleRepeatManifestV1"
        and manifest.get("analysis_id")
        == "deform360-resource-lifecycle-distributional-equivalence-v1"
        and manifest.get("expected_environment", {}).get("generator_profile")
        == "same-as-analyzer"
        and manifest.get("expected_environment", {}).get("physical_gpu_index") == 1,
        "resource qualification analyzer manifest changed",
    )
    decision = result.get("decision")
    _require(
        result.get("artifact_kind")
        == "Deform360ResourceLifecycleDistributionalEquivalenceV1"
        and result.get("analysis_id")
        == "deform360-resource-lifecycle-distributional-equivalence-v1"
        and result.get("generator_profile") == "same-as-analyzer"
        and result.get("physical_gpu_index") == 1
        and isinstance(decision, Mapping)
        and decision.get("accepted") is True
        and decision.get("acceptance_basis")
        in {
            "exact-structured-array-equality",
            "secondary-distributional-envelope",
        }
        and completion.get("equivalence_decision") == decision,
        "resource qualification analyzer result changed",
    )
    analyzer_source = evidence.get("runtime_bindings", {}).get("analyzer_source")
    _require(
        isinstance(analyzer_source, Mapping)
        and analyzer_source.get("sha256") == _QUALIFICATION_ANALYZER_SHA256
        and completion.get("analyzer_source", {}).get("sha256")
        == _QUALIFICATION_ANALYZER_SHA256,
        "resource qualification frozen analyzer changed",
    )
    _require(
        completion.get("information_boundary")
        == {
            "formal_held_path_accessed": False,
            "formal_target_query_prediction_or_score_deserialized": False,
            "public_development_dataset_only": True,
            "scientific_method_selected_from_qualification": False,
        },
        "resource qualification completion boundary changed",
    )
    inventory = completion.get("sealed_content_inventory")
    observed_metadata = _qualification_metadata_inventory(root)
    _require(
        isinstance(inventory, Mapping)
        and isinstance(inventory.get("inventory_sha256"), str)
        and len(inventory["inventory_sha256"]) == 64
        and inventory.get("entry_count") == observed_metadata["entry_count"]
        and inventory.get("metadata_inventory_sha256")
        == observed_metadata["metadata_inventory_sha256"],
        "resource qualification tree metadata changed",
    )
    return (
        {
            "resource_lifecycle_qualification_attempt": {
                **attempt_record,
                "artifact_sha256": attempt["artifact_sha256"],
            },
            "resource_lifecycle_qualification_evidence": {
                **evidence_record,
                "artifact_sha256": evidence["artifact_sha256"],
            },
            "resource_lifecycle_qualification_repeat_manifest": {
                **manifest_record,
                "artifact_sha256": manifest["artifact_sha256"],
            },
            "resource_lifecycle_qualification_equivalence_result": {
                **result_record,
                "artifact_sha256": result["artifact_sha256"],
            },
            "resource_lifecycle_qualification_integrity_completion": {
                **completion_record,
                "artifact_sha256": completion["artifact_sha256"],
            },
        },
        {
            "root": str(root),
            "root_mode_octal": "0500",
            "fully_nonwritable": True,
            **completion["sealed_content_inventory"],
            **completion["source_code"],
            "terminal_outcome": completion["terminal_outcome"],
            "admission_eligible": completion["admission_eligible"],
            "generator_profile": evidence["generator_profile"],
            "physical_gpu_index": evidence["physical_gpu_index"],
            "equivalence_acceptance_basis": decision["acceptance_basis"],
            "analyzer_source_sha256": _QUALIFICATION_ANALYZER_SHA256,
        },
    )


def _qualification_metadata_inventory(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    states: dict[Path, tuple[int, ...]] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_state = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_state.st_mode)
            and not stat.S_ISLNK(current_state.st_mode)
            and stat.S_IMODE(current_state.st_mode) == 0o500,
            f"resource qualification directory is not sealed: {current_path}",
        )
        states[current_path] = _stable_inventory_state(current_state)
        directories[:] = sorted(directories)
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"resource qualification directory is not sealed: {child}",
            )
            states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "type": "directory",
                    "mode_octal": "0500",
                }
            )
        for name in sorted(files):
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o400
                and observed.st_nlink == 1,
                f"resource qualification file is not sealed: {child}",
            )
            states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "type": "file",
                    "mode_octal": "0400",
                    "size_bytes": observed.st_size,
                }
            )
    for path, before in states.items():
        _require(
            _stable_inventory_state(os.lstat(path)) == before,
            f"resource qualification metadata changed while scanning: {path}",
        )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "entry_count": len(rows),
        "metadata_inventory_sha256": hashlib.sha256(
            json.dumps(
                {"rows": rows},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _bind_attempt3_archive_integrity() -> dict[str, Any]:
    archive = Path(os.path.abspath(os.fspath(_ATTEMPT3_ARCHIVE)))
    observed = os.lstat(archive)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o500
        and archive.resolve() == archive,
        "attempt-3 archive is not a canonical mode-0500 directory",
    )
    candidates: list[Path] = []
    for child in archive.iterdir():
        if not child.name.startswith("code-"):
            continue
        suffix = child.name.removeprefix("code-")
        if len(suffix) in {40, 64} and all(
            character in "0123456789abcdef" for character in suffix
        ):
            candidates.append(child)
    _require(len(candidates) == 1, "attempt-3 deployed-code directory is not unique")
    deployed_code = candidates[0]
    code_state = os.lstat(deployed_code)
    _require(
        stat.S_ISDIR(code_state.st_mode)
        and not stat.S_ISLNK(code_state.st_mode)
        and stat.S_IMODE(code_state.st_mode) == 0o500
        and deployed_code.resolve() == deployed_code,
        "attempt-3 deployed-code directory changed",
    )
    _require(
        _attempt3_repository_binding(deployed_code) == _ATTEMPT3_DEPLOYED_CODE_BINDING,
        "attempt-3 deployed-code repository binding changed",
    )
    observed_inventory = _observed_attempt3_noncode_inventory(
        archive, deployed_code=deployed_code
    )
    _require(
        observed_inventory
        == {
            "entry_count": _ATTEMPT3_ARCHIVE_INTEGRITY["postseal_noncode_entry_count"],
            "inventory_sha256": _ATTEMPT3_ARCHIVE_INTEGRITY[
                "postseal_noncode_inventory_sha256"
            ],
        },
        "attempt-3 archive inventory changed",
    )
    return {
        **_ATTEMPT3_ARCHIVE_INTEGRITY,
        "path": str(archive),
        "postseal_noncode_entry_count": observed_inventory["entry_count"],
        "postseal_noncode_inventory_sha256": observed_inventory["inventory_sha256"],
    }


def _stable_inventory_state(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_uid,
        observed.st_gid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _attempt3_deployed_head(code: Path) -> str:
    git_directory = code / ".git"
    git_state = os.lstat(git_directory)
    _require(
        stat.S_ISDIR(git_state.st_mode)
        and not stat.S_ISLNK(git_state.st_mode)
        and stat.S_IMODE(git_state.st_mode) == 0o500,
        "attempt-3 deployed-code Git directory changed",
    )
    head_path = git_directory / "HEAD"
    before = os.lstat(head_path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o400
        and before.st_size <= 256,
        "attempt-3 deployed-code HEAD changed",
    )
    descriptor = os.open(
        head_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_inventory_state(opened) == _stable_inventory_state(before),
            "attempt-3 deployed-code HEAD changed while opening",
        )
        payload = bytearray()
        while block := os.read(descriptor, 256):
            payload.extend(block)
            _require(len(payload) <= 256, "attempt-3 deployed-code HEAD is oversized")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(head_path)
    _require(
        _stable_inventory_state(before)
        == _stable_inventory_state(after)
        == _stable_inventory_state(current),
        "attempt-3 deployed-code HEAD changed while reading",
    )
    try:
        return bytes(payload).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("attempt-3 deployed-code HEAD is not ASCII") from error


def _run_attempt3_git(code: Path, arguments: list[str]) -> bytes:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fileMode=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(code),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    _require(
        completed.returncode == 0,
        "attempt-3 deployed-code git "
        + " ".join(arguments)
        + " failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    return completed.stdout


def _parse_attempt3_git_tree(raw: bytes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        _require(bool(separator) and bool(path_bytes), "malformed deployed Git tree")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed deployed Git tree header")
        try:
            mode, kind, object_id = (field.decode("ascii") for field in fields)
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("deployed Git tree is not canonical text") from error
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id),
            f"unsupported attempt-3 deployed-code entry: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe attempt-3 deployed-code path",
        )
        rows.append({"mode": mode, "type": kind, "object_id": object_id, "path": path})
    _require(bool(rows), "attempt-3 deployed Git tree is empty")
    _require(
        [row["path"] for row in rows] == sorted(row["path"] for row in rows),
        "attempt-3 deployed Git tree is not sorted",
    )
    return rows


def _attempt3_worktree_blob_oid(path: Path, *, object_id: str) -> str:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o400,
        f"attempt-3 tracked file is not sealed mode 0400: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_inventory_state(opened) == _stable_inventory_state(before),
            f"attempt-3 tracked file changed while opening: {path}",
        )
        algorithm = "sha1" if len(object_id) == 40 else "sha256"
        digest = hashlib.new(algorithm)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        _stable_inventory_state(before)
        == _stable_inventory_state(after)
        == _stable_inventory_state(current),
        f"attempt-3 tracked file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _attempt3_repository_binding(code: Path) -> dict[str, Any]:
    git_directory = code / ".git"
    git_state = os.lstat(git_directory)
    _require(
        stat.S_ISDIR(git_state.st_mode)
        and not stat.S_ISLNK(git_state.st_mode)
        and stat.S_IMODE(git_state.st_mode) == 0o500,
        "attempt-3 deployed-code Git directory changed",
    )
    top = _run_attempt3_git(code, ["rev-parse", "--show-toplevel"])
    _require(
        top.decode("utf-8").strip() == str(code),
        "attempt-3 deployed Git top level changed",
    )
    head = _run_attempt3_git(code, ["rev-parse", "HEAD"]).decode("ascii").strip()
    _require(
        head == _attempt3_deployed_head(code),
        "attempt-3 deployed-code checkout changed",
    )
    _require(
        _run_attempt3_git(code, ["status", "--porcelain=v1", "--untracked-files=all"])
        == b"",
        "attempt-3 deployed worktree content changed",
    )
    # Deliberately omit every exclude option.  Unlike `git status`, this also
    # exposes files matched by .gitignore and repository-local exclude rules.
    _require(
        _run_attempt3_git(code, ["ls-files", "--others", "-z"]) == b"",
        "attempt-3 deployed worktree has untracked or ignored files",
    )
    _require(
        _run_attempt3_git(code, ["rev-parse", "--is-shallow-repository"])
        .decode("ascii")
        .strip()
        == "false",
        "attempt-3 deployed repository is shallow",
    )
    _run_attempt3_git(code, ["fsck", "--full", "--no-dangling"])
    rows = _parse_attempt3_git_tree(
        _run_attempt3_git(code, ["ls-tree", "-r", "-z", "HEAD"])
    )
    tracked_paths = {str(row["path"]) for row in rows}
    tracked_directories = {
        parent.as_posix()
        for path in (Path(relative) for relative in tracked_paths)
        for parent in path.parents
        if parent != Path(".")
    }
    for row in rows:
        path = code / row["path"]
        _require(
            _attempt3_worktree_blob_oid(path, object_id=row["object_id"])
            == row["object_id"],
            f"attempt-3 tracked file content changed: {path}",
        )
    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for current, directories, files in os.walk(code, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(code)
        if relative_parent == Path("."):
            directories[:] = sorted(name for name in directories if name != ".git")
        else:
            directories[:] = sorted(directories)
        for name in directories:
            path = current_path / name
            observed = os.lstat(path)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"attempt-3 deployed worktree directory changed: {path}",
            )
            actual_directories.add(path.relative_to(code).as_posix())
        for name in sorted(files):
            path = current_path / name
            observed = os.lstat(path)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o400,
                f"attempt-3 deployed worktree file changed: {path}",
            )
            actual_paths.add(path.relative_to(code).as_posix())
    _require(
        actual_paths == tracked_paths and actual_directories == tracked_directories,
        "attempt-3 deployed worktree path set changed",
    )
    canonical_rows = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "path": code.name,
        "git_head": head,
        "head_text_sha256": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "git_tree_record_count": len(rows),
        "git_tree_manifest_sha256": hashlib.sha256(canonical_rows).hexdigest(),
    }


def _attempt3_inventory_file_row(path: Path, *, relative: Path) -> dict[str, Any]:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o400,
        f"attempt-3 archive file is not a sealed mode-0400 file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and _stable_inventory_state(opened) == _stable_inventory_state(before),
            f"attempt-3 archive file changed while opening: {path}",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        _stable_inventory_state(before)
        == _stable_inventory_state(after)
        == _stable_inventory_state(current),
        f"attempt-3 archive file changed while hashing: {path}",
    )
    return {
        "path": relative.as_posix(),
        "type": "file",
        "mode_octal": "0400",
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _observed_attempt3_noncode_inventory(
    archive: Path, *, deployed_code: Path
) -> dict[str, Any]:
    directory_states: dict[Path, tuple[int, ...]] = {}
    rows: list[dict[str, Any]] = []
    report_relative = Path(_ATTEMPT3_REPORT.name)
    for current, directories, files in os.walk(
        archive, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_state = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_state.st_mode)
            and not stat.S_ISLNK(current_state.st_mode)
            and stat.S_IMODE(current_state.st_mode) == 0o500,
            f"attempt-3 archive directory is not sealed mode 0500: {current_path}",
        )
        directory_states[current_path] = _stable_inventory_state(current_state)
        relative_parent = current_path.relative_to(archive)
        directories[:] = sorted(
            name
            for name in directories
            if not (
                relative_parent == Path(".") and current_path / name == deployed_code
            )
        )
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"attempt-3 archive directory is not sealed mode 0500: {child}",
            )
            directory_states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": child.relative_to(archive).as_posix(),
                    "type": "directory",
                    "mode_octal": "0500",
                }
            )
        for name in sorted(files):
            child = current_path / name
            relative = child.relative_to(archive)
            if relative == report_relative:
                continue
            rows.append(_attempt3_inventory_file_row(child, relative=relative))
    for path, expected in directory_states.items():
        _require(
            _stable_inventory_state(os.lstat(path)) == expected,
            f"attempt-3 archive directory changed while hashing: {path}",
        )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len({str(row["path"]) for row in rows}) == len(rows),
        "attempt-3 archive inventory has a duplicate path",
    )
    canonical_rows = json.dumps(
        {"rows": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(canonical_rows).hexdigest(),
    }


def expected_unsigned_report(
    bindings: Mapping[str, Mapping[str, Any]],
    archive_integrity: Mapping[str, Any],
    *,
    attempt4_launcher: Mapping[str, Any],
    attempt4_execution: Mapping[str, Any],
    attempt4_information: Mapping[str, Any],
    qualification_files: Mapping[str, Mapping[str, Any]],
    qualification_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        set(bindings) == set(_EXPECTED_FILES),
        "disclosure input binding set changed",
    )
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "disclosed_v7_files": {
            name: dict(bindings[name]) for name in sorted(_V7_FILE_NAMES)
        },
        "disclosed_v8_attempt3_files": {
            name: dict(bindings[name]) for name in sorted(_ATTEMPT3_FILE_NAMES)
        },
        "disclosed_v8_attempt4_files": {
            name: dict(bindings[name]) for name in sorted(_ATTEMPT4_FILE_NAMES)
        },
        "v8_attempt3_archive_integrity": dict(archive_integrity),
        "v8_attempt4_archive_integrity": dict(_ATTEMPT4_ARCHIVE_INTEGRITY),
        "v8_attempt4_launcher_integrity": {
            **dict(attempt4_launcher),
            "root_mode_octal": "0500",
            "fully_nonwritable": True,
        },
        "v8_attempt4_execution_boundary": {
            "calibration_result": "NO_CALIBRATION_RESULT",
            "first_complete_cohort_barrier_crossed": True,
            "completed_target_x0_queried_pairs": 2,
            "partial_third_target_reconstruction": True,
            "second_complete_cohort_barrier_crossed": False,
            "score_evidence_count": 0,
            "gate_decision_count": 0,
            "confirmation_accessed": False,
            "report_execution_boundary": dict(attempt4_execution),
            "report_information_boundary": dict(attempt4_information),
        },
        "resource_lifecycle_qualification_files": {
            name: dict(value) for name, value in sorted(qualification_files.items())
        },
        "resource_lifecycle_qualification_integrity": dict(qualification_integrity),
        "v8_attempt3_revision_basis": {
            "official_x0_geometry_used_to_diagnose_exclusion_liveness": True,
            "future_target_coordinates_masks_or_scores_used_for_revision": False,
            "queried_prediction_score_or_gate_existed": False,
            "revision": (
                "replace exact-one-per-center matching with the inclusive 15 mm "
                "x0-only radius union"
            ),
        },
        "post_withdrawal_development": {
            **_POST_WITHDRAWAL_DEVELOPMENT_HASHES,
            "retired_official_target_opened_by_development_process": True,
            "retired_online_prediction_opened_by_development_process": True,
            "future_coordinates_or_masks_may_have_been_read": True,
            "derived_metrics_may_have_been_computed": True,
            "field_hypothesis_was_subsequently_reselected_on_independent_open27": True,
        },
        "attempt4_technical_failure_development": {
            "durable_launcher_log_used_for_fixed_marker_and_traceback_diagnosis": True,
            "too_many_open_files_diagnosed": True,
            "formal_target_query_prediction_or_score_array_deserialized": False,
            "attempt4_score_gate_or_confirmation_existed": False,
            "scientific_method_or_threshold_selected_from_attempt4_outcomes": False,
            "repair_scope": (
                "per-fit Nerfstudio resource lifecycle plus a post-case file-"
                "descriptor growth guard"
            ),
        },
        "retirement": {
            "exact_episode": "002-rope-silk-ep0003",
            "replacement_episode": "072-cotton-clohesline-ep0003",
            "replacement_search_excluded_entire_002_rope_silk_object": True,
            "reason": (
                "the exact held-v7 episode was exposed after formal withdrawal; "
                "the replacement was selected outside that object's episodes"
            ),
        },
        "v8_1_reuse_boundary": {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "v8_attempt3_predictions_reused": False,
            "v8_attempt3_source_manifests_reused": False,
            "v8_attempt3_frozen_fields_reused": False,
            "v8_attempt3_target_artifacts_reused": False,
            "v8_attempt3_official_x0_query_artifacts_reused": False,
            "v8_attempt3_queried_prediction_artifacts_reused": False,
            "v8_attempt3_score_or_gate_artifacts_reused": False,
            "v8_attempt3_partial_artifacts_reused": False,
            "v8_attempt4_predictions_reused": False,
            "v8_attempt4_source_manifests_reused": False,
            "v8_attempt4_frozen_fields_reused": False,
            "v8_attempt4_target_artifacts_reused": False,
            "v8_attempt4_official_x0_query_artifacts_reused": False,
            "v8_attempt4_queried_prediction_artifacts_reused": False,
            "v8_attempt4_score_or_gate_artifacts_reused": False,
            "v8_attempt4_partial_artifacts_reused": False,
            "all_v8_1_attempt5_predictions_targets_queries_and_scores_fresh": True,
            "full_15_case_fresh_rerun_required": True,
        },
        "claim_boundary": (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8.1 into an official Deform360 "
            "state-of-the-art comparison."
        ),
    }


def build_report(
    *,
    qualification_evidence: Path,
    qualification_completion: Path,
    qualification_evidence_sha256: str,
    qualification_completion_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    bindings = {
        name: _bind_expected_regular_file(
            path,
            expected_size=size,
            expected_sha256=sha256,
            role=name.replace("_", " "),
        )
        for name, (path, size, sha256) in _EXPECTED_FILES.items()
    }
    launcher, execution, information = _bind_attempt4_lineage(bindings)
    qualification_files, qualification_integrity = _bind_resource_qualification(
        qualification_evidence,
        qualification_completion,
        evidence_sha256=qualification_evidence_sha256,
        completion_sha256=qualification_completion_sha256,
    )
    return _artifact(
        expected_unsigned_report(
            bindings,
            _bind_attempt3_archive_integrity(),
            attempt4_launcher=launcher,
            attempt4_execution=execution,
            attempt4_information=information,
            qualification_files=qualification_files,
            qualification_integrity=qualification_integrity,
        )
    )


def _write_once(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute.parent.is_dir(), "held-v8 root does not exist")
    _require(not absolute.parent.is_symlink(), "held-v8 root is a symlink")
    _require(
        absolute.parent.resolve() == absolute.parent, "held-v8 root is non-canonical"
    )
    if os.path.lexists(absolute):
        before = os.lstat(absolute)
        _require(
            stat.S_ISREG(before.st_mode)
            and not stat.S_ISLNK(before.st_mode)
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o400,
            "existing disclosure is not a sealed regular file",
        )
        descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            observed = b""
            while block := os.read(descriptor, 1024 * 1024):
                observed += block
        finally:
            os.close(descriptor)
        _require(observed == payload, "existing disclosure payload changed")
        return
    descriptor = os.open(
        absolute,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(absolute, 0o400, follow_symlinks=False)
        directory = os.open(
            absolute.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        absolute.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-evidence", type=Path, required=True)
    parser.add_argument("--qualification-completion", type=Path, required=True)
    parser.add_argument("--qualification-evidence-sha256", required=True)
    parser.add_argument("--qualification-completion-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    _require(_V8_ROOT == _OUTPUT.parent, "disclosure output root changed")
    _, payload = build_report(
        qualification_evidence=arguments.qualification_evidence,
        qualification_completion=arguments.qualification_completion,
        qualification_evidence_sha256=arguments.qualification_evidence_sha256,
        qualification_completion_sha256=arguments.qualification_completion_sha256,
    )
    _write_once(_OUTPUT, payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
