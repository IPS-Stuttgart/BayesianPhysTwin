#!/usr/bin/env python3
"""Prepare the prospective Deform360 held-v8 lock and immutable deployment.

The formal root must not exist when this operator starts.  A complete clean
Git deployment is staged on the same filesystem, the required disclosure and
calibration lock are written into a newly created root, and only then is the
already-validated deployment atomically renamed below that root.  Thus no
prediction, source acquisition, target reconstruction, or score can precede
the immutable calibration lock.

Run ``--preflight`` first.  It performs every read-only provenance check and
prints the exact prospective bindings without creating the formal root.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any


_HELD_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
_HELD_ROOT = _HELD_BASE / "held-v8"
_LOCK_PATH = _HELD_ROOT / "calibration-lock.json"
_DISCLOSURE_PATH = _HELD_ROOT / "post-withdrawal-development-use-disclosure.json"
_V7_LOCK = _HELD_BASE / "held-v7" / "calibration-lock.json"
_V7_WITHDRAWAL = _HELD_BASE / "held-v7" / "v7-outcome-withdrawal-report.json"
_V7_RUNTIME_SMOKE = _HELD_BASE / "held-v7" / "gsplat-runtime-smoke-evidence.json"
_V8_ATTEMPT1_WITHDRAWAL_POINTER = (
    _HELD_BASE / "held-v8-attempt-1-withdrawal-pointer.json"
)
_V8_ATTEMPT1_WITHDRAWAL_REPORT = (
    _HELD_BASE
    / "held-v8-attempt-1-withdrawn-preoutcome"
    / "execution-withdrawal-preoutcome.json"
)
_V8_ATTEMPT2_ARCHIVE = _HELD_BASE / "held-v8-attempt-2-withdrawn-preoutcome"
_V8_ATTEMPT2_WITHDRAWAL_POINTER = (
    _HELD_BASE / "held-v8-attempt-2-withdrawal-pointer.json"
)
_V8_ATTEMPT2_WITHDRAWAL_REPORT = (
    _V8_ATTEMPT2_ARCHIVE / "execution-withdrawal-preoutcome-attempt2.json"
)
_V8_ATTEMPT2_INTEGRITY_COMPLETION = (
    _HELD_BASE / "held-v8-attempt-2-withdrawal-integrity-completion.json"
)
_V8_ATTEMPT2_MANIFEST_SCALE_DIAGNOSTIC = (
    _V8_ATTEMPT2_ARCHIVE / "prewithdrawal-072-manifest-scale-diagnostic.json"
)
_V8_ATTEMPT2_ADMISSION_DIAGNOSTIC = (
    _V8_ATTEMPT2_ARCHIVE / "prewithdrawal-072-admission-compatibility-diagnostic.json"
)
_V8_ATTEMPT2_FAILURE_LOG = (
    _V8_ATTEMPT2_ARCHIVE
    / "calibration"
    / "logs"
    / "072-cotton-clohesline-ep0003.physical.failed.log"
)
_V8_ATTEMPT3_ARCHIVE = _HELD_BASE / "held-v8-attempt-3-withdrawn-postbarrier"
_V8_ATTEMPT3_WITHDRAWAL_REPORT = (
    _V8_ATTEMPT3_ARCHIVE / "execution-withdrawal-postbarrier-attempt3.json"
)
_V8_ATTEMPT3_WITHDRAWAL_POINTER = (
    _HELD_BASE / "held-v8-attempt-3-withdrawal-pointer.json"
)
_V8_ATTEMPT3_INTEGRITY_COMPLETION = (
    _HELD_BASE / "held-v8-attempt-3-withdrawal-integrity-completion.json"
)
_V8_ATTEMPT3_REPORT_FILE_SHA256 = (
    "6d9c62606d18744d275df51fd08e041205bf15b38175d74c69690eafd511054b"
)
_V8_ATTEMPT3_REPORT_ARTIFACT_SHA256 = (
    "4b7404961fa13b418265f76827dda356fb6ad019db764c6302f49e8149d05de2"
)
_V8_ATTEMPT3_COMPLETION_FILE_SHA256 = (
    "f3d1e8a6670484c81ac04743bcdb020cdee3fba02229a64844a8a9c9f4b8b989"
)
_V8_ATTEMPT3_COMPLETION_ARTIFACT_SHA256 = (
    "9ec2989e3000464a0f72b038e26fe407403e02721e21c19ae4fb9123c6a7cf8c"
)
_V8_ATTEMPT3_POINTER_FILE_SHA256 = (
    "75acc7e9535f41528d22739ae8eeb5a0a2247c0fe63c097ad1da2859d7b33246"
)
_V8_ATTEMPT3_POINTER_ARTIFACT_SHA256 = (
    "6ef596a63029d7fa8346141bb52c72d99062e201a12b7c9baf4fca7330baca64"
)
_V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256 = (
    "5d398e998e2b738db545ffefd254712c6822017cfc5be6e7de435d5883c8c4c8"
)
_V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT = 1466
_V8_ATTEMPT3_DEPLOYED_CODE_NAME = "code-9ad7ad2b385f7abc5e8c42081a41018980dd3827"
_V8_ATTEMPT3_DEPLOYED_HEAD = "9ad7ad2b385f7abc5e8c42081a41018980dd3827"
_V8_ATTEMPT3_DEPLOYED_HEAD_TEXT_SHA256 = (
    "b5e33f85b96a0026147040044c288ef5c6ff3e60ca9b74743f904b49f78b79f1"
)
_V8_ATTEMPT3_DEPLOYED_TREE_MANIFEST_SHA256 = (
    "445f325dca5710c9873951445cb26107966e5344333edd8a69ac380e50e09546"
)
_V8_ATTEMPT3_DEPLOYED_TREE_RECORD_COUNT = 950
_V8_ATTEMPT3_OPERATOR_SOURCE_SHA256 = (
    "bc6efe5660c90828be13fb9221472c5e37261e5041509ff61403ea89ef3e9648"
)
_V8_ATTEMPT3_PROTOCOL_ID = "deform360-held-online-belief-v8"
_V8_ATTEMPT3_EXECUTION_ATTEMPT = 3
_V8_ATTEMPT3_WITHDRAWAL_STATUS = (
    "withdrawn-postbarrier-before-queried-prediction-or-score"
)
_V8_ATTEMPT3_COMPLETION_STATUS = "withdrawal-integrity-complete"
_V8_ATTEMPT3_DISPOSITION = (
    "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_SEAL_OR_SCORE"
)

# Attempt 4 requires a fresh replay against its exact adapter and protocol
# sources.  These deliberately empty digest pins make preflight fail closed
# until that replay has been executed and sealed at the new versioned root.
_V8_ADMISSION_REPLAY_ROOT = Path(
    "/mnt/corsair/florianpfaff/"
    "bpt-held-v8.1-attempt-4-admission-wrapper-scratch-20260722"
)
_V8_ADMISSION_REPLAY_REPORT = (
    _V8_ADMISSION_REPLAY_ROOT / "metadata-only-replay-report.json"
)
_V8_ADMISSION_REPLAY_CODE_BINDING = (
    _V8_ADMISSION_REPLAY_ROOT / "metadata-only-replay-code-binding.json"
)
_V8_ADMISSION_REPLAY_REPORT_FILE_SHA256: str | None = (
    "bdb9d2d577e2eed87531c29f7bba83cfbe0a7fc42ee7f0b3d203e6af038152a7"
)
_V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256: str | None = (
    "79824d3c7884fdb968fee4dd6573b12fc6ebbead59f5f5e8bb94181fa2788eb5"
)
_V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256: str | None = (
    "81d9a2ec3154082ffa2853a4ae5357bc4609e5a83f1358d19c2a4b89b33e6981"
)
_V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256: str | None = (
    "badfc0ba1317c54878d0a172701c86cbf91a67294b04befcc1f31d5c7aa3c31a"
)
_V81_ADMISSION_REPLAY_REPORT_KIND = (
    "Deform360HeldV81ExternalAdmissionMetadataOnlyReplay"
)
_V81_ADMISSION_REPLAY_CODE_BINDING_KIND = (
    "Deform360HeldV81ExternalAdmissionReplayCodeBinding"
)
_V81_PROTOCOL_ID = "deform360-held-online-belief-v8.1"
_V81_EXECUTION_ATTEMPT = 4
_V81_CROSS_AUTHORIZATION_CASE_NAME = "072-cotton-clohesline-ep0004"
_V81_CROSS_AUTHORIZATION_STDERR_MARKER = (
    "outside the exact v8 external calibration admission"
)
_V81_REPLAY_OUTPUT_NAMES = frozenset(
    {
        "episode_graph.npz",
        "simulator_final_data.pkl",
        "state_artifact.npz",
        "twin_summary.json",
    }
)
_V81_REPLAY_ROOT_FILE_NAMES = _V81_REPLAY_OUTPUT_NAMES | frozenset(
    {
        "stdout.log",
        "stderr.log",
        "metadata-only-replay-report.json",
        "metadata-only-replay-code-binding.json",
    }
)
_V81_REPLAY_CROSS_AUTHORIZATION_FILE_NAMES = frozenset({"stdout.log", "stderr.log"})
_V81_ALLOWED_POST_REPLAY_VALIDATION_PATHS = frozenset(
    {
        "scripts/held/prepare_deform360_v8_lock.py",
        "tests/test_deform360_held_v8_lock_preparer.py",
    }
)
_V81_PINNED_PYTHON_LAUNCHER_TARGET = "/usr/bin/python3"
_V81_PINNED_PYTHON_TARGET = Path("/usr/bin/python3.12")
_V81_PINNED_PYTHON_TARGET_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
_V81_PYTHON_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
_V81_PYTHON_TREE_MANIFEST_SHA256 = (
    "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61"
)
_V81_UPSTREAM_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "Bayesian-PhysTwin-upstream-58ab4808e59d"
)
_V81_UPSTREAM_HEAD = "58ab4808e59da811dd1a2c66ac628fe4ea2faeab"
_V81_UPSTREAM_TREE = "2b35d539be7a17b2de2c644b46c267b16ce26bf0"
_V81_UPSTREAM_BUILDER = (
    _V81_UPSTREAM_ROOT
    / "scripts"
    / "remote"
    / "build_deform360_automatic_episode_twin.py"
)
_V81_UPSTREAM_BUILDER_SHA256 = (
    "dd43bfeaa0ddb53252e3b2d9c907c147379b2cce6b4c5d5dfa14f310fdacfa9a"
)
_V81_UPSTREAM_AUTHORIZER = (
    _V81_UPSTREAM_ROOT / "src" / "causal4d_public" / "deform360_dense_reusable_panel.py"
)
_V81_UPSTREAM_AUTHORIZER_SHA256 = (
    "0861831b9ab3cf6d64833efe533073f4f444f2315c04057377f243efffd8b17e"
)
_V81_DEFORM360_HEAD = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
_V81_DEFORM360_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
_V81_DEFORM360_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    f"Deform360-processing-{_V81_DEFORM360_HEAD}"
)
_OPEN27_DECISION = (
    _HELD_BASE
    / "runs"
    / "deform360-query-field-open27-v1-development"
    / "decision.json"
)
_RIGID_RESIDUAL_DECISION = (
    _HELD_BASE
    / "runs"
    / "deform360-rigid-residual-open27-v1-development"
    / "decision.json"
)
_GSPLAT_SUPPLEMENT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v7-runtimes/"
    "gsplat-cuda-py312-cu121-"
    "2dd5e0c2a349619e1afc3dd041086eca900b387602bc76627b7f54264fffec64/"
    "runtime-supplement-manifest.json"
)
_RUNTIME_ROOT = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v5-runtimes/"
    "bpt-gpu-pip-4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)
_PINNED_PYTHON = _RUNTIME_ROOT / "bin" / "python"
_PYTHON_FREEZE = Path(f"{_RUNTIME_ROOT}.freeze.sorted.txt")
_PYTHON_TREE_MANIFEST = Path(f"{_RUNTIME_ROOT}.tree-manifest.json")
_SEMANTIC_MODEL_LOCK = Path(
    "/mnt/corsair/florianpfaff/bpt-framezero-field-dev-20260720/"
    "scratch_siglip2_model_lock.json"
)
_ALLTRACKER_CHECKPOINT = Path("/mnt/corsair/florianpfaff/model-cache/alltracker.pth")
_SAM2_CHECKPOINT = Path(
    "/mnt/lexar4tb/datasets/deform360/sam2-2b90b9f5/checkpoints/sam2.1_hiera_small.pt"
)
_DEFORM360_CODE = Path(
    "/mnt/corsair/florianpfaff/bpt-held-v81-runtimes/"
    "Deform360-processing-0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
)

_EXPECTED_EXTERNAL_FILES: Mapping[str, tuple[Path, str | None, int | None]] = {
    "v7_calibration_lock_file": (
        _V7_LOCK,
        "b464d7cfda3b4ad94f57ffd46267b3b50d8dc65e2ff8dfec2befc7953718aca7",
        0o400,
    ),
    "v7_withdrawal_report": (
        _V7_WITHDRAWAL,
        "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3",
        0o400,
    ),
    "v7_gsplat_runtime_smoke_evidence": (
        _V7_RUNTIME_SMOKE,
        "c5f0218962e1c18748f52d423c11804864e2695a719f00ff63452cebdbde029c",
        0o400,
    ),
    "v8_attempt1_preoutcome_withdrawal_pointer": (
        _V8_ATTEMPT1_WITHDRAWAL_POINTER,
        "f7af6d1adf8541fd015cbe5336da97e013777c1bb711deaa01d9a84a49c81daa",
        0o400,
    ),
    "v8_attempt1_preoutcome_withdrawal_report": (
        _V8_ATTEMPT1_WITHDRAWAL_REPORT,
        "c04a6e7a95d958950ea7e7c05e7e2b98ee4516c01f03e9284f85ccccf0f6873b",
        0o400,
    ),
    "v8_attempt2_preoutcome_withdrawal_pointer": (
        _V8_ATTEMPT2_WITHDRAWAL_POINTER,
        "007d3fbde0dc93dc350661aafdd5d08d1398aa8d1f164e17bf295521fc40463a",
        0o400,
    ),
    "v8_attempt2_preoutcome_withdrawal_report": (
        _V8_ATTEMPT2_WITHDRAWAL_REPORT,
        "5830f9bfe8d29d5a09f64afbcaeabadc3acb7c8fdf820c1aeb68a6601055a895",
        0o400,
    ),
    "v8_attempt2_withdrawal_integrity_completion": (
        _V8_ATTEMPT2_INTEGRITY_COMPLETION,
        "21e7695af5f610193502ecb6e7e6c647d853bde34daa1c5f362e990dffdf56a7",
        0o400,
    ),
    "v8_attempt2_manifest_scale_diagnostic": (
        _V8_ATTEMPT2_MANIFEST_SCALE_DIAGNOSTIC,
        "3166d488258f1f62535c87813bbd895c9e4ba9855d43fa4393b8795f85c78973",
        0o400,
    ),
    "v8_attempt2_admission_compatibility_diagnostic": (
        _V8_ATTEMPT2_ADMISSION_DIAGNOSTIC,
        "ba45b56d1e127099d7ef1a910d199cc0f6c9dd698b7f785828163bc28904e2fb",
        0o400,
    ),
    "v8_attempt2_failure_log": (
        _V8_ATTEMPT2_FAILURE_LOG,
        "e296021c5b647d5e26cbf8cecd2e3fc46ebed97026a2564224a54f0fcd156b1c",
        0o400,
    ),
    "v8_attempt3_postbarrier_withdrawal_report": (
        _V8_ATTEMPT3_WITHDRAWAL_REPORT,
        _V8_ATTEMPT3_REPORT_FILE_SHA256,
        0o400,
    ),
    "v8_attempt3_postbarrier_withdrawal_pointer": (
        _V8_ATTEMPT3_WITHDRAWAL_POINTER,
        _V8_ATTEMPT3_POINTER_FILE_SHA256,
        0o400,
    ),
    "v8_attempt3_withdrawal_integrity_completion": (
        _V8_ATTEMPT3_INTEGRITY_COMPLETION,
        _V8_ATTEMPT3_COMPLETION_FILE_SHA256,
        0o400,
    ),
    "v8_external_admission_metadata_only_replay": (
        _V8_ADMISSION_REPLAY_REPORT,
        _V8_ADMISSION_REPLAY_REPORT_FILE_SHA256,
        0o400,
    ),
    "v8_external_admission_replay_code_binding": (
        _V8_ADMISSION_REPLAY_CODE_BINDING,
        _V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256,
        0o400,
    ),
    "gsplat_runtime_supplement_manifest": (
        _GSPLAT_SUPPLEMENT,
        "87532ef68494442e2ab54885abbd760b7331ea8a83fa72110ea93589a60b1eee",
        0o400,
    ),
    "pinned_python_freeze": (
        _PYTHON_FREEZE,
        "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004",
        0o400,
    ),
    "pinned_python_tree_manifest": (
        _PYTHON_TREE_MANIFEST,
        "8147db39bc3ab30943951ae5f304de48ffc819625d30a382d5305528b6601b61",
        0o400,
    ),
    "open27_development_decision": (
        _OPEN27_DECISION,
        "110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd",
        0o400,
    ),
    "rigid_residual_rejection_decision": (
        _RIGID_RESIDUAL_DECISION,
        "b72faf6f7d4551622d6abbbd9521f05e46da7ef8cf4e9e17b161896889c7a2fa",
        0o400,
    ),
    "semantic_model_lock": (
        _SEMANTIC_MODEL_LOCK,
        "e5696dc4650194fe2d773a7c5a197862e9d87dda6d7ee5cc45401d5b71f55239",
        0o400,
    ),
    "alltracker_checkpoint": (
        _ALLTRACKER_CHECKPOINT,
        "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d",
        None,
    ),
    "sam2_checkpoint": (
        _SAM2_CHECKPOINT,
        "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        None,
    ),
}

_EXPECTED_EXTERNAL_ARTIFACT_SHA256: Mapping[str, str | None] = {
    "v8_attempt2_preoutcome_withdrawal_pointer": (
        "9063011657b955902d1cf7d85a4253eee65caa430a41edae2709a18032baf99c"
    ),
    "v8_attempt2_preoutcome_withdrawal_report": (
        "457c6a64c0208b91ee5eb0f8038d22ae7eda743e29fb60a4bcb4ef1a2861b147"
    ),
    "v8_attempt2_withdrawal_integrity_completion": (
        "eb3a6c092a84dd95f516770d9837711a4f5b1eb58a28fee84c6df0bddb4999b0"
    ),
    "v8_attempt2_manifest_scale_diagnostic": (
        "96f7edc666cda3cf84c6121623028c290b577ceec62cc104a41780b7bb6560ce"
    ),
    "v8_attempt2_admission_compatibility_diagnostic": (
        "e659ceb9b4120c9a2e0c2bf33cbc8478bfc0157ed9b4f9415c3ebef194ea3f80"
    ),
    "v8_attempt3_postbarrier_withdrawal_report": (_V8_ATTEMPT3_REPORT_ARTIFACT_SHA256),
    "v8_attempt3_postbarrier_withdrawal_pointer": (
        _V8_ATTEMPT3_POINTER_ARTIFACT_SHA256
    ),
    "v8_attempt3_withdrawal_integrity_completion": (
        _V8_ATTEMPT3_COMPLETION_ARTIFACT_SHA256
    ),
    "v8_external_admission_metadata_only_replay": (
        _V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256
    ),
    "v8_external_admission_replay_code_binding": (
        _V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256
    ),
}

_LOCAL_BINDING_FILES: Mapping[str, str] = {
    "held_v8_lock_preparer_source": "scripts/held/prepare_deform360_v8_lock.py",
    "held_v8_attempt2_withdrawal_operator_source": (
        "scripts/held/seal_deform360_v8_attempt2_withdrawal.py"
    ),
    "held_v8_attempt2_withdrawal_integrity_completion_operator_source": (
        "scripts/held/seal_deform360_v8_attempt2_withdrawal_completion.py"
    ),
    "held_v8_attempt3_withdrawal_operator_source": (
        "scripts/held/seal_deform360_v8_attempt3_outcome_failure.py"
    ),
    "held_v81_external_admission_replay_operator_source": (
        "scripts/held/replay_deform360_v81_external_admission.py"
    ),
    "held_v8_disclosure_sealer_source": (
        "scripts/held/seal_deform360_v8_post_withdrawal_disclosure.py"
    ),
    "held_v8_replacement_source_acquisition_launcher_source": (
        "scripts/held/run_deform360_v8_replacement_source.py"
    ),
    "held_v8_calibration_case_runner_source": (
        "scripts/held/run_deform360_v8_calibration_case.sh"
    ),
    "held_v8_confirmation_case_runner_source": (
        "scripts/held/run_deform360_v8_confirmation_case.sh"
    ),
    "held_v8_common_case_runner_source": (
        "scripts/held/run_deform360_v8_case_common.sh"
    ),
    "held_v8_calibration_shard_runner_source": (
        "scripts/held/run_deform360_v8_calibration_shard.sh"
    ),
    "held_v8_confirmation_shard_runner_source": (
        "scripts/held/run_deform360_v8_confirmation_shard.sh"
    ),
    "held_v8_calibration_outcome_driver_source": (
        "scripts/held/run_deform360_v8_calibration_outcomes.py"
    ),
    "held_v8_confirmation_outcome_driver_source": (
        "scripts/held/run_deform360_v8_confirmation_outcomes.py"
    ),
    "held_v8_x0_query_worker_source": ("scripts/held/run_deform360_v8_x0_query.py"),
    "held_v8_protocol_source": ("src/bayesian_phystwin/deform360_held_v8_protocol.py"),
    "held_v8_replacement_source_operator_source": (
        "src/bayesian_phystwin/deform360_held_v8_replacement_source.py"
    ),
    "held_v8_builder_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_builders.py"
    ),
    "held_v8_outcome_driver_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"
    ),
    "held_v8_outcome_reconstruction_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_reconstruction.py"
    ),
    "held_v8_gsplat_runtime_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ),
    "held_v8_query_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_query_artifacts.py"
    ),
    "held_v8_outcome_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_artifacts.py"
    ),
    "held_v8_scoring_source": ("src/bayesian_phystwin/deform360_held_v8_scoring.py"),
    "held_v8_score_artifacts_source": (
        "src/bayesian_phystwin/deform360_held_v8_score_artifacts.py"
    ),
    "held_v8_frozen_query_field_source": (
        "src/bayesian_phystwin/deform360_frozen_query_field.py"
    ),
    # This intentionally overrides the inherited v7 identity: v8 keeps the
    # exhaustive optimizer and changes only its audit serialization schema.
    "frame_zero_builder_source": (
        "src/bayesian_phystwin/deform360_frame_zero_assets.py"
    ),
    "held_official_reconstruction_numerical_source": (
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
    ),
    "held_gsplat_runtime_source": (
        "src/bayesian_phystwin/deform360_held_gsplat_runtime.py"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> tuple[Path, bytes, os.stat_result]:
    source = _absolute(path)
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"{role} is not a regular file",
    )
    _require(source.resolve() == source, f"{role} has a symlinked ancestor")
    if required_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == required_mode,
            f"{role} mode changed",
        )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{role} changed while opening",
        )
        digest = hashlib.sha256()
        payload = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(source)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    _require(
        identity
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
    _require(digest.digest() == hashlib.sha256(payload).digest(), "hash state changed")
    return source, bytes(payload), after


def _sha256_file(
    path: str | Path, *, role: str, required_mode: int | None = None
) -> str:
    _, payload, _ = _read_file(path, role=role, required_mode=required_mode)
    return hashlib.sha256(payload).hexdigest()


def _run_git(
    root: Path, arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed


def _parse_git_tree(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        _require(bool(separator) and bool(path_bytes), "malformed Git tree record")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed Git tree header")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = path_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id),
            f"unsupported tracked entry: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe tracked path",
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(bool(records), "Git tree is empty")
    _require(
        [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "Git tree paths are not sorted",
    )
    return records


def _validate_repository(root: str | Path) -> dict[str, Any]:
    code = _absolute(root)
    observed = os.lstat(code)
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        "source code is not a real directory",
    )
    _require(code.resolve() == code, "source code has a symlinked ancestor")
    _require((code / ".git").is_dir(), "source code is not a non-bare Git repository")
    top = _run_git(code, ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    _require(top == str(code), "source Git top level changed")
    head = _run_git(code, ["rev-parse", "HEAD"]).stdout.decode().strip().lower()
    _require(
        len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        "source HEAD is invalid",
    )
    _require(
        _run_git(code, ["status", "--porcelain=v1", "--untracked-files=all"]).stdout
        == b"",
        "source worktree is not completely clean",
    )
    _require(
        _run_git(code, ["rev-parse", "--is-shallow-repository"]).stdout.decode().strip()
        == "false",
        "source repository is shallow",
    )
    _run_git(code, ["fsck", "--full", "--no-dangling"])
    records = _parse_git_tree(_run_git(code, ["ls-tree", "-r", "-z", "HEAD"]).stdout)
    _require(
        all((code / record["path"]).is_file() for record in records),
        "source tracked file is absent",
    )
    return {
        "root": code,
        "head": head,
        "head_text_sha256": _sha256_text(head),
        "tree_records": records,
        "tree_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }


def _require_deployed_read_only(code: Path) -> None:
    for root, directories, files in os.walk(code, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *files]:
            path = root_path / name
            observed = os.lstat(path)
            _require(
                not stat.S_ISLNK(observed.st_mode), "deployment contains a symlink"
            )
            _require(observed.st_mode & 0o222 == 0, f"deployment is writable: {path}")


def _make_read_only(code: Path) -> None:
    paths: list[Path] = []
    for root, directories, files in os.walk(code, topdown=False, followlinks=False):
        root_path = Path(root)
        paths.extend(root_path / name for name in files)
        paths.extend(root_path / name for name in directories)
    paths.append(code)
    for path in paths:
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), "deployment contains a symlink")
        if stat.S_ISDIR(observed.st_mode):
            mode = 0o555
        elif stat.S_ISREG(observed.st_mode):
            mode = 0o555 if observed.st_mode & 0o111 else 0o444
        else:
            raise ValueError(f"deployment contains a special file: {path}")
        os.chmod(path, mode, follow_symlinks=False)
    _require_deployed_read_only(code)


def _clone_staged_deployment(source: Path, head: str, stage: Path) -> dict[str, Any]:
    _require(not os.path.lexists(stage), "deployment stage already exists")
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-local", str(source), str(stage)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    _require(
        completed.returncode == 0,
        "failed to clone independent deployment: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    try:
        _run_git(stage, ["checkout", "--detach", head])
        _run_git(stage, ["remote", "remove", "origin"])
        observed = _validate_repository(stage)
        _require(observed["head"] == head, "staged deployment HEAD changed")
        _make_read_only(stage)
        return observed
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _external_bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (
        path,
        expected_sha256,
        required_mode,
    ) in _EXPECTED_EXTERNAL_FILES.items():
        _require(
            _valid_sha256(expected_sha256),
            f"{name} expected SHA-256 placeholder is not populated",
        )
        expected_artifact = _EXPECTED_EXTERNAL_ARTIFACT_SHA256.get(name)
        if name in _EXPECTED_EXTERNAL_ARTIFACT_SHA256:
            _require(
                _valid_sha256(expected_artifact),
                f"{name} expected artifact SHA-256 placeholder is not populated",
            )
        observed = _sha256_file(
            path, role=name.replace("_", " "), required_mode=required_mode
        )
        _require(observed == expected_sha256, f"{name} SHA-256 changed")
        result[name] = observed
        if expected_artifact is not None:
            _, payload, _ = _read_file(
                path,
                role=f"{name.replace('_', ' ')} artifact",
                required_mode=required_mode,
            )
            try:
                artifact = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name} is not canonical JSON") from error
            _require(
                isinstance(artifact, dict)
                and artifact.get("artifact_sha256") == expected_artifact,
                f"{name} artifact SHA-256 field changed",
            )
            unsigned = dict(artifact)
            unsigned.pop("artifact_sha256")
            _require(
                hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
                == expected_artifact,
                f"{name} canonical artifact SHA-256 changed",
            )
            result[f"{name}_artifact"] = expected_artifact
    result["pinned_python_executable_target"] = _validate_pinned_python()
    return result


def _inherited_v7_bindings() -> dict[str, str]:
    """Load numerical/runtime pins from the exact sealed v7 parent lock.

    These are source and runtime identities, not v7 predictions or outcomes.
    The v8-specific method/tree bindings are overlaid later.
    """

    _, payload, _ = _read_file(
        _V7_LOCK, role="sealed v7 calibration lock", required_mode=0o400
    )
    _require(
        hashlib.sha256(payload).hexdigest()
        == "b464d7cfda3b4ad94f57ffd46267b3b50d8dc65e2ff8dfec2befc7953718aca7",
        "sealed v7 calibration lock changed before inheritance",
    )
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sealed v7 calibration lock is not JSON") from error
    _require(
        isinstance(artifact, dict)
        and artifact.get("protocol_id") == "deform360-held-online-belief-v7",
        "sealed parent lock is not exact held v7",
    )
    raw = artifact.get("immutable_bindings")
    _require(isinstance(raw, dict) and bool(raw), "v7 immutable bindings are absent")
    bindings = {str(key): str(value) for key, value in raw.items()}
    _require(
        all(
            key
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for key, value in bindings.items()
        ),
        "v7 immutable binding is not named SHA-256",
    )
    return dict(sorted(bindings.items()))


def _validate_pinned_python() -> str:
    """Validate the exact venv launcher symlink and its system interpreter."""

    observed = os.lstat(_PINNED_PYTHON)
    _require(
        stat.S_ISLNK(observed.st_mode)
        and os.readlink(_PINNED_PYTHON) == "/usr/bin/python3",
        "pinned Python launcher symlink changed",
    )
    target = _PINNED_PYTHON.resolve(strict=True)
    _require(
        target == Path("/usr/bin/python3.12")
        and target.is_file()
        and not target.is_symlink()
        and os.access(target, os.X_OK),
        "pinned Python target changed or is not executable",
    )
    digest = _sha256_file(target, role="pinned Python executable target")
    _require(
        digest == "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
        "pinned Python executable target SHA-256 changed",
    )
    return digest


def _local_file_bindings(code: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in _LOCAL_BINDING_FILES.items():
        result[name] = _sha256_file(code / relative, role=name.replace("_", " "))
    return result


def _validate_attempt2_operator_source_lineage(
    local_bindings: Mapping[str, str],
) -> None:
    _, payload, _ = _read_file(
        _V8_ATTEMPT2_INTEGRITY_COMPLETION,
        role="attempt-2 withdrawal integrity completion",
        required_mode=0o400,
    )
    try:
        completion = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("attempt-2 integrity completion is not JSON") from error
    records = completion.get("operator_source_bindings")
    _require(isinstance(records, Mapping), "attempt-2 operator bindings are absent")
    expected = {
        "held_v8_attempt2_withdrawal_operator_source": ("attempt2_withdrawal_operator"),
        "held_v8_attempt2_withdrawal_integrity_completion_operator_source": (
            "attempt2_integrity_completion_operator"
        ),
    }
    for local_name, completion_name in expected.items():
        record = records.get(completion_name)
        _require(
            isinstance(record, Mapping)
            and record.get("sha256") == local_bindings.get(local_name),
            f"{local_name} differs from the executed operator source",
        )


def _attempt3_archive_inventory_contract() -> dict[str, object]:
    return {
        "archive_path": str(_V8_ATTEMPT3_ARCHIVE),
        "postseal_noncode_entry_count": _V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT,
        "postseal_noncode_inventory_sha256": (_V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256),
    }


def _load_attempt3_lineage_artifact(path: Path, *, role: str) -> dict[str, Any]:
    _, payload, _ = _read_file(path, role=role, required_mode=0o400)
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not JSON") from error
    _require(isinstance(artifact, dict), f"{role} is not a JSON object")
    return artifact


def _run_isolated_filemode_git(
    root: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[bytes]:
    return _run_git(root, ["-c", "core.fileMode=false", *arguments])


def _git_blob_object_id(payload: bytes, *, hexadecimal_length: int) -> str:
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    if hexadecimal_length == 40:
        return hashlib.sha1(framed).hexdigest()
    _require(hexadecimal_length == 64, "unsupported Git object hash length")
    return hashlib.sha256(framed).hexdigest()


def _validate_attempt3_excluded_deployed_code(
    report: Mapping[str, Any],
) -> None:
    expected_binding = {
        "path": _V8_ATTEMPT3_DEPLOYED_CODE_NAME,
        "git_head": _V8_ATTEMPT3_DEPLOYED_HEAD,
        "head_text_sha256": _V8_ATTEMPT3_DEPLOYED_HEAD_TEXT_SHA256,
        "git_tree_record_count": _V8_ATTEMPT3_DEPLOYED_TREE_RECORD_COUNT,
        "git_tree_manifest_sha256": _V8_ATTEMPT3_DEPLOYED_TREE_MANIFEST_SHA256,
    }
    inventory = report.get("expected_postseal_inventory")
    _require(
        report.get("deployed_code") == expected_binding
        and isinstance(inventory, Mapping)
        and inventory.get("excluded_deployed_code_directory")
        == _V8_ATTEMPT3_DEPLOYED_CODE_NAME,
        "attempt-3 excluded deployed-code report binding changed",
    )
    code = _V8_ATTEMPT3_ARCHIVE / _V8_ATTEMPT3_DEPLOYED_CODE_NAME
    observed = os.lstat(code)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and code.resolve() == code
        and (code / ".git").is_dir(),
        "attempt-3 excluded deployed code is not a canonical Git repository",
    )
    top = (
        _run_isolated_filemode_git(code, ["rev-parse", "--show-toplevel"])
        .stdout.decode("utf-8")
        .strip()
    )
    head = (
        _run_isolated_filemode_git(code, ["rev-parse", "HEAD"])
        .stdout.decode("ascii")
        .strip()
        .lower()
    )
    _require(
        top == str(code)
        and head == _V8_ATTEMPT3_DEPLOYED_HEAD
        and _sha256_text(head) == _V8_ATTEMPT3_DEPLOYED_HEAD_TEXT_SHA256,
        "attempt-3 excluded deployed-code top level or HEAD changed",
    )
    _require(
        _run_isolated_filemode_git(
            code,
            ["status", "--porcelain=v1", "--untracked-files=no"],
        ).stdout
        == b""
        and _run_isolated_filemode_git(
            code,
            ["ls-files", "--others", "--exclude-standard"],
        ).stdout
        == b""
        and _run_isolated_filemode_git(
            code,
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
        ).stdout
        == b"",
        "attempt-3 excluded deployed code is dirty or has untracked content",
    )
    _require(
        _run_isolated_filemode_git(code, ["rev-parse", "--is-shallow-repository"])
        .stdout.decode("ascii")
        .strip()
        == "false",
        "attempt-3 excluded deployed-code repository is shallow",
    )
    _run_isolated_filemode_git(code, ["fsck", "--full", "--no-dangling"])
    records = _parse_git_tree(
        _run_isolated_filemode_git(code, ["ls-tree", "-r", "-z", "HEAD"]).stdout
    )
    _require(
        len(records) == _V8_ATTEMPT3_DEPLOYED_TREE_RECORD_COUNT
        and hashlib.sha256(_canonical_bytes(records)).hexdigest()
        == _V8_ATTEMPT3_DEPLOYED_TREE_MANIFEST_SHA256,
        "attempt-3 excluded deployed-code Git tree manifest changed",
    )
    for record in records:
        tracked = code / record["path"]
        _, payload, tracked_stat = _read_file(
            tracked,
            role=f"attempt-3 deployed tracked blob {record['path']}",
        )
        _require(
            stat.S_IMODE(tracked_stat.st_mode) == 0o400
            and _git_blob_object_id(
                payload,
                hexadecimal_length=len(record["object_id"]),
            )
            == record["object_id"],
            f"attempt-3 deployed tracked blob changed: {record['path']}",
        )


def _validate_attempt3_archive_lineage(
    local_bindings: Mapping[str, str],
) -> None:
    local_operator = local_bindings.get("held_v8_attempt3_withdrawal_operator_source")
    _require(
        local_operator == _V8_ATTEMPT3_OPERATOR_SOURCE_SHA256,
        "attempt-3 withdrawal operator differs from the observed executed source",
    )

    observed_archive = os.lstat(_V8_ATTEMPT3_ARCHIVE)
    _require(
        stat.S_ISDIR(observed_archive.st_mode)
        and not stat.S_ISLNK(observed_archive.st_mode)
        and _V8_ATTEMPT3_ARCHIVE.resolve() == _V8_ATTEMPT3_ARCHIVE
        and stat.S_IMODE(observed_archive.st_mode) == 0o500,
        "attempt-3 archive is not the exact sealed directory",
    )

    report = _load_attempt3_lineage_artifact(
        _V8_ATTEMPT3_WITHDRAWAL_REPORT,
        role="attempt-3 post-barrier withdrawal report",
    )
    _validate_attempt3_excluded_deployed_code(report)
    pointer = _load_attempt3_lineage_artifact(
        _V8_ATTEMPT3_WITHDRAWAL_POINTER,
        role="attempt-3 post-barrier withdrawal pointer",
    )
    completion = _load_attempt3_lineage_artifact(
        _V8_ATTEMPT3_INTEGRITY_COMPLETION,
        role="attempt-3 withdrawal integrity completion",
    )

    for label, artifact, expected_status in (
        ("report", report, _V8_ATTEMPT3_WITHDRAWAL_STATUS),
        ("pointer", pointer, _V8_ATTEMPT3_WITHDRAWAL_STATUS),
        ("completion", completion, _V8_ATTEMPT3_COMPLETION_STATUS),
    ):
        _require(
            artifact.get("protocol_id") == _V8_ATTEMPT3_PROTOCOL_ID
            and artifact.get("execution_attempt") == _V8_ATTEMPT3_EXECUTION_ATTEMPT
            and artifact.get("status") == expected_status
            and artifact.get("disposition") == _V8_ATTEMPT3_DISPOSITION,
            f"attempt-3 {label} identity or disposition changed",
        )
        operator = artifact.get("executed_withdrawal_operator_source")
        _require(
            isinstance(operator, Mapping) and operator.get("sha256") == local_operator,
            f"attempt-3 {label} differs from the local executed operator source",
        )

    inventory = report.get("expected_postseal_inventory")
    _require(
        isinstance(inventory, Mapping)
        and report.get("immutable_archive_path") == str(_V8_ATTEMPT3_ARCHIVE)
        and inventory.get("entry_count") == _V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT
        and inventory.get("inventory_sha256") == _V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256,
        "attempt-3 report archive inventory changed",
    )
    for label, artifact in (("pointer", pointer), ("completion", completion)):
        _require(
            artifact.get("archive_path") == str(_V8_ATTEMPT3_ARCHIVE)
            and artifact.get("archive_fully_nonwritable") is True
            and artifact.get("archive_root_mode_octal") == "0500"
            and artifact.get("postseal_noncode_entry_count")
            == _V8_ATTEMPT3_ARCHIVE_ENTRY_COUNT
            and artifact.get("postseal_noncode_inventory_sha256")
            == _V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
            and artifact.get("withdrawal_report_file_sha256")
            == _V8_ATTEMPT3_REPORT_FILE_SHA256
            and artifact.get("withdrawal_report_artifact_sha256")
            == _V8_ATTEMPT3_REPORT_ARTIFACT_SHA256,
            f"attempt-3 {label} archive or report lineage changed",
        )
    completion_binding = pointer.get("withdrawal_integrity_completion")
    _require(
        isinstance(completion_binding, Mapping)
        and completion_binding.get("path") == str(_V8_ATTEMPT3_INTEGRITY_COMPLETION)
        and completion_binding.get("file_sha256") == _V8_ATTEMPT3_COMPLETION_FILE_SHA256
        and completion_binding.get("artifact_sha256")
        == _V8_ATTEMPT3_COMPLETION_ARTIFACT_SHA256,
        "attempt-3 pointer integrity-completion binding changed",
    )


def _load_admission_replay_json(
    path: Path,
    *,
    role: str,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    _, payload, observed = _read_file(path, role=role, required_mode=0o400)
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not JSON") from error
    _require(isinstance(artifact, dict), f"{role} is not a JSON object")
    return artifact, payload, observed


def _validate_replay_artifact_sha256(
    artifact: Mapping[str, Any],
    expected_sha256: object,
    *,
    role: str,
) -> None:
    _require(_valid_sha256(expected_sha256), f"{role} digest pin is not populated")
    unsigned = dict(artifact)
    observed = unsigned.pop("artifact_sha256", None)
    _require(
        observed == expected_sha256
        and hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() == expected_sha256,
        f"{role} artifact SHA-256 changed",
    )


def _validate_replay_bound_file(
    record: object,
    expected_path: Path,
    *,
    role: str,
) -> bytes:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "sha256", "size_bytes"}
        and record.get("path") == str(expected_path)
        and _valid_sha256(record.get("sha256"))
        and isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record["size_bytes"]) >= 0,
        f"{role} record changed",
    )
    _, payload, _ = _read_file(expected_path, role=role, required_mode=0o400)
    _require(
        len(payload) == record["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == record["sha256"],
        f"{role} differs from its replay binding",
    )
    return payload


def _validate_exact_replay_root_allowlist() -> None:
    expected_root = _V81_REPLAY_ROOT_FILE_NAMES | {"cross-auth"}
    observed_root = {entry.name for entry in os.scandir(_V8_ADMISSION_REPLAY_ROOT)}
    _require(
        observed_root == expected_root,
        "v8.1 admission replay root allowlist changed",
    )
    for name in sorted(_V81_REPLAY_ROOT_FILE_NAMES):
        observed = os.lstat(_V8_ADMISSION_REPLAY_ROOT / name)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and observed.st_nlink == 1,
            f"v8.1 replay root entry is not a regular file: {name}",
        )
    cross = _V8_ADMISSION_REPLAY_ROOT / "cross-auth"
    cross_stat = os.lstat(cross)
    _require(
        stat.S_ISDIR(cross_stat.st_mode)
        and not stat.S_ISLNK(cross_stat.st_mode)
        and stat.S_IMODE(cross_stat.st_mode) == 0o500,
        "v8.1 cross-authorization entry is not a directory",
    )
    observed_cross = {entry.name for entry in os.scandir(cross)}
    _require(
        observed_cross == _V81_REPLAY_CROSS_AUTHORIZATION_FILE_NAMES,
        "v8.1 cross-authorization allowlist changed",
    )
    for name in sorted(_V81_REPLAY_CROSS_AUTHORIZATION_FILE_NAMES):
        observed = os.lstat(cross / name)
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and observed.st_nlink == 1,
            f"v8.1 cross-authorization entry is not a regular file: {name}",
        )


def _validate_external_file_record(
    record: object,
    expected_path: Path,
    expected_sha256: str,
    *,
    role: str,
    required_mode: int | None = None,
) -> None:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "sha256", "size_bytes"}
        and record.get("path") == str(expected_path)
        and record.get("sha256") == expected_sha256
        and isinstance(record.get("size_bytes"), int)
        and not isinstance(record.get("size_bytes"), bool)
        and int(record["size_bytes"]) >= 0,
        f"{role} record changed",
    )
    _, payload, _ = _read_file(
        expected_path,
        role=role,
        required_mode=required_mode,
    )
    _require(
        len(payload) == record["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{role} differs from its runtime binding",
    )


def _require_immutable_repository_tree(root: Path, *, role: str) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            observed = os.lstat(path)
            _require(
                not stat.S_ISLNK(observed.st_mode)
                and (stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode))
                and observed.st_mode & 0o222 == 0,
                f"{role} contains a writable, symlinked, or special entry",
            )


def _validate_immutable_runtime_repository(
    record: object,
    *,
    expected_root: Path,
    expected_head: str,
    expected_tree: str,
    role: str,
) -> None:
    _require(
        isinstance(record, Mapping)
        and record.get("repository_root") == str(expected_root)
        and record.get("git_head") == expected_head
        and record.get("git_tree") == expected_tree
        and record.get("clean_tracked_and_untracked") is True
        and record.get("ignored_files_absent") is True
        and record.get("fully_nonwritable") is True,
        f"{role} identity record changed",
    )
    observed = os.lstat(expected_root)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and expected_root.resolve() == expected_root
        and observed.st_mode & 0o222 == 0
        and (expected_root / ".git").is_dir(),
        f"{role} is not an immutable canonical Git repository",
    )
    top = (
        _run_isolated_filemode_git(expected_root, ["rev-parse", "--show-toplevel"])
        .stdout.decode("utf-8")
        .strip()
    )
    head = (
        _run_isolated_filemode_git(expected_root, ["rev-parse", "HEAD"])
        .stdout.decode("ascii")
        .strip()
        .lower()
    )
    tree = (
        _run_isolated_filemode_git(expected_root, ["rev-parse", "HEAD^{tree}"])
        .stdout.decode("ascii")
        .strip()
        .lower()
    )
    _require(
        top == str(expected_root) and head == expected_head and tree == expected_tree,
        f"{role} Git top, HEAD, or tree changed",
    )
    _require(
        _run_isolated_filemode_git(
            expected_root,
            ["status", "--porcelain=v1", "--untracked-files=no"],
        ).stdout
        == b""
        and _run_isolated_filemode_git(
            expected_root,
            ["ls-files", "--others", "--exclude-standard"],
        ).stdout
        == b""
        and _run_isolated_filemode_git(
            expected_root,
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
        ).stdout
        == b"",
        f"{role} is dirty or contains ignored/untracked files",
    )
    _require(
        _run_isolated_filemode_git(
            expected_root, ["rev-parse", "--is-shallow-repository"]
        )
        .stdout.decode("ascii")
        .strip()
        == "false",
        f"{role} is shallow",
    )
    _run_isolated_filemode_git(expected_root, ["fsck", "--full", "--no-dangling"])
    _require_immutable_repository_tree(expected_root, role=role)


def _validate_replay_external_runtime(record: object) -> None:
    _require(
        isinstance(record, Mapping)
        and set(record) == {"python", "upstream", "deform360"},
        "v8.1 replay external-runtime record changed",
    )
    python = record.get("python")
    _require(
        isinstance(python, Mapping)
        and set(python)
        == {
            "launcher_path",
            "launcher_target",
            "executable_target",
            "environment_freeze",
            "tree_manifest",
        }
        and python.get("launcher_path") == str(_PINNED_PYTHON)
        and python.get("launcher_target") == _V81_PINNED_PYTHON_LAUNCHER_TARGET,
        "v8.1 replay pinned-Python identity changed",
    )
    launcher = os.lstat(_PINNED_PYTHON)
    _require(
        stat.S_ISLNK(launcher.st_mode)
        and os.readlink(_PINNED_PYTHON) == _V81_PINNED_PYTHON_LAUNCHER_TARGET
        and _PINNED_PYTHON.resolve(strict=True) == _V81_PINNED_PYTHON_TARGET,
        "v8.1 replay pinned-Python launcher changed",
    )
    _validate_external_file_record(
        python.get("executable_target"),
        _V81_PINNED_PYTHON_TARGET,
        _V81_PINNED_PYTHON_TARGET_SHA256,
        role="v8.1 replay pinned-Python executable",
    )
    _validate_external_file_record(
        python.get("environment_freeze"),
        _PYTHON_FREEZE,
        _V81_PYTHON_FREEZE_SHA256,
        role="v8.1 replay Python environment freeze",
        required_mode=0o400,
    )
    _validate_external_file_record(
        python.get("tree_manifest"),
        _PYTHON_TREE_MANIFEST,
        _V81_PYTHON_TREE_MANIFEST_SHA256,
        role="v8.1 replay Python tree manifest",
        required_mode=0o400,
    )

    upstream = record.get("upstream")
    _require(
        isinstance(upstream, Mapping)
        and set(upstream)
        == {
            "repository_root",
            "git_head",
            "git_tree",
            "clean_tracked_and_untracked",
            "ignored_files_absent",
            "fully_nonwritable",
            "automatic_twin_builder",
            "dense_panel_authorizer",
        },
        "v8.1 replay upstream runtime record changed",
    )
    _validate_immutable_runtime_repository(
        upstream,
        expected_root=_V81_UPSTREAM_ROOT,
        expected_head=_V81_UPSTREAM_HEAD,
        expected_tree=_V81_UPSTREAM_TREE,
        role="v8.1 replay immutable upstream",
    )
    _validate_external_file_record(
        upstream.get("automatic_twin_builder"),
        _V81_UPSTREAM_BUILDER,
        _V81_UPSTREAM_BUILDER_SHA256,
        role="v8.1 replay upstream automatic-twin builder",
    )
    _validate_external_file_record(
        upstream.get("dense_panel_authorizer"),
        _V81_UPSTREAM_AUTHORIZER,
        _V81_UPSTREAM_AUTHORIZER_SHA256,
        role="v8.1 replay upstream dense-panel authorizer",
    )

    deform360 = record.get("deform360")
    _require(
        isinstance(deform360, Mapping)
        and set(deform360)
        == {
            "repository_root",
            "git_head",
            "git_tree",
            "clean_tracked_and_untracked",
            "ignored_files_absent",
            "fully_nonwritable",
        },
        "v8.1 replay Deform360 runtime record changed",
    )
    _validate_immutable_runtime_repository(
        deform360,
        expected_root=_V81_DEFORM360_ROOT,
        expected_head=_V81_DEFORM360_HEAD,
        expected_tree=_V81_DEFORM360_TREE,
        role="v8.1 replay immutable Deform360 processing snapshot",
    )


def _validate_replay_source_commit(
    tested: Mapping[str, Any],
    local_bindings: Mapping[str, str],
    source_code: Path,
) -> None:
    head = tested.get("git_head")
    _require(
        isinstance(head, str)
        and len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        "v8.1 replay source commit is invalid",
    )
    top = (
        _run_git(source_code, ["rev-parse", "--show-toplevel"]).stdout.decode().strip()
    )
    current_head = _run_git(source_code, ["rev-parse", "HEAD"]).stdout.decode().strip()
    _run_git(source_code, ["cat-file", "-e", f"{head}^{{commit}}"])
    ancestor = _run_git(
        source_code,
        ["merge-base", "--is-ancestor", str(head), current_head],
        check=False,
    )
    _require(
        top == str(source_code)
        and ancestor.returncode == 0
        and _run_git(
            source_code,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ).stdout
        == b""
        and _run_git(
            source_code,
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
        ).stdout
        == b"",
        "v8.1 replay source commit is not the current clean source",
    )
    changed_paths = set(
        _run_git(
            source_code,
            ["diff", "--name-only", "-z", f"{head}..{current_head}"],
        )
        .stdout.rstrip(b"\0")
        .split(b"\0")
    ) - {b""}
    _require(
        changed_paths
        <= {path.encode("utf-8") for path in _V81_ALLOWED_POST_REPLAY_VALIDATION_PATHS},
        "post-replay source changes escaped replay-independent validation files",
    )
    for local_name, replay_name in (
        ("held_v8_builder_adapter_source", "adapter_source_sha256"),
        ("held_v8_protocol_source", "protocol_source_sha256"),
        (
            "held_v81_external_admission_replay_operator_source",
            "replay_operator_source_sha256",
        ),
    ):
        relative = _LOCAL_BINDING_FILES[local_name]
        committed = _run_git(source_code, ["show", f"{head}:{relative}"]).stdout
        digest = hashlib.sha256(committed).hexdigest()
        _require(
            digest == tested.get(replay_name) == local_bindings.get(local_name),
            f"{local_name} differs from the replayed clean-source commit",
        )


def _validate_admission_replay_source_lineage(
    local_bindings: Mapping[str, str],
    builders: Any,
    source_code: Path,
) -> None:
    root_stat = os.lstat(_V8_ADMISSION_REPLAY_ROOT)
    _require(
        stat.S_ISDIR(root_stat.st_mode)
        and not stat.S_ISLNK(root_stat.st_mode)
        and _V8_ADMISSION_REPLAY_ROOT.resolve() == _V8_ADMISSION_REPLAY_ROOT
        and stat.S_IMODE(root_stat.st_mode) == 0o500,
        "v8.1 admission replay root is not the exact sealed directory",
    )
    _validate_exact_replay_root_allowlist()
    report, report_payload, report_stat = _load_admission_replay_json(
        _V8_ADMISSION_REPLAY_REPORT,
        role="v8.1 admission replay report",
    )
    replay, replay_payload, _ = _load_admission_replay_json(
        _V8_ADMISSION_REPLAY_CODE_BINDING,
        role="v8.1 admission replay code binding",
    )
    _require(
        _valid_sha256(_V8_ADMISSION_REPLAY_REPORT_FILE_SHA256)
        and hashlib.sha256(report_payload).hexdigest()
        == _V8_ADMISSION_REPLAY_REPORT_FILE_SHA256,
        "v8.1 admission replay report file SHA-256 changed",
    )
    _require(
        _valid_sha256(_V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256)
        and hashlib.sha256(replay_payload).hexdigest()
        == _V8_ADMISSION_REPLAY_CODE_BINDING_FILE_SHA256,
        "v8.1 admission replay code-binding file SHA-256 changed",
    )
    _validate_replay_artifact_sha256(
        report,
        _V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256,
        role="v8.1 admission replay report",
    )
    _validate_replay_artifact_sha256(
        replay,
        _V8_ADMISSION_REPLAY_CODE_BINDING_ARTIFACT_SHA256,
        role="v8.1 admission replay code binding",
    )

    _require(
        report.get("schema_version") == 1
        and report.get("artifact_kind") == _V81_ADMISSION_REPLAY_REPORT_KIND
        and report.get("protocol_id") == _V81_PROTOCOL_ID
        and report.get("execution_attempt") == _V81_EXECUTION_ATTEMPT
        and report.get("case_name") == builders.V8_EXTERNAL_CALIBRATION_CASE_NAME
        and report.get("role") == "calibration"
        and report.get("development_replay_only") is True
        and report.get("formal_outcome_evidence") is False,
        "v8.1 admission replay report identity or evidence boundary changed",
    )
    _require(
        replay.get("schema_version") == 1
        and replay.get("artifact_kind") == _V81_ADMISSION_REPLAY_CODE_BINDING_KIND
        and replay.get("protocol_id") == _V81_PROTOCOL_ID
        and replay.get("execution_attempt") == _V81_EXECUTION_ATTEMPT
        and replay.get("formal_outcome_evidence") is False
        and replay.get("target_query_score_or_outcome_accessed") is False,
        "v8.1 admission replay code-binding identity or boundary changed",
    )

    current_contract = builders.V8_EXTERNAL_ADMISSION_CONTRACT_SHA256
    _require(_valid_sha256(current_contract), "current admission contract is invalid")
    admission = report.get("admission")
    _require(
        isinstance(admission, Mapping)
        and admission.get("protocol_id") == builders.V8_EXTERNAL_ADMISSION_PROTOCOL_ID
        and admission.get("contract_sha256") == current_contract
        and admission.get("exact_case_only") is True
        and admission.get("target_access") is False
        and replay.get("admission_contract_sha256") == current_contract,
        "v8.1 admission replay does not bind the current exact-case contract",
    )

    replay_report = replay.get("replay_report")
    _require(
        isinstance(replay_report, Mapping)
        and set(replay_report)
        == {
            "path",
            "sha256",
            "size_bytes",
            "artifact_sha256",
        }
        and replay_report.get("path") == str(_V8_ADMISSION_REPLAY_REPORT)
        and replay_report.get("sha256") == _V8_ADMISSION_REPLAY_REPORT_FILE_SHA256
        and replay_report.get("size_bytes") == report_stat.st_size
        and replay_report.get("artifact_sha256")
        == _V8_ADMISSION_REPLAY_REPORT_ARTIFACT_SHA256
        == report.get("artifact_sha256"),
        "v8.1 admission report-to-code-binding lineage changed",
    )

    tested = replay.get("local_worktree_at_replay")
    _require(
        isinstance(tested, Mapping) and report.get("local_source_at_replay") == tested,
        "replay-tested source binding is absent or differs across artifacts",
    )
    for local_name, replay_name in (
        ("held_v8_builder_adapter_source", "adapter_source_sha256"),
        ("held_v8_protocol_source", "protocol_source_sha256"),
        (
            "held_v81_external_admission_replay_operator_source",
            "replay_operator_source_sha256",
        ),
    ):
        _require(
            tested.get(replay_name) == local_bindings.get(local_name),
            f"{local_name} differs from the real pinned-upstream replay",
        )
    bootstrap_sha256 = hashlib.sha256(
        builders._V8_EXTERNAL_ADMISSION_RUNPY_BOOTSTRAP.encode("utf-8")
    ).hexdigest()
    _require(
        tested.get("exact_child_bootstrap_sha256") == bootstrap_sha256
        and tested.get("uncommitted_correction_present") is False,
        "replay bootstrap or committed-source boundary changed",
    )
    _validate_replay_source_commit(tested, local_bindings, source_code)
    _validate_replay_external_runtime(tested.get("external_runtime"))

    source_evidence = report.get("source_evidence")
    _require(
        isinstance(source_evidence, Mapping)
        and source_evidence.get("future_object_observation_used") is False
        and source_evidence.get("source_used_for_numerical_replay")
        == "prediction_only_input_only",
        "v8.1 admission replay source crossed its development-only boundary",
    )
    _require(
        report.get("information_boundary")
        == {
            "official_target_created": False,
            "official_target_read": False,
            "query_created": False,
            "query_read": False,
            "score_created": False,
            "score_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "confirmation_accessed": False,
        },
        "v8.1 replay target/query/score/outcome boundary changed",
    )

    successful = report.get("successful_replay")
    _require(isinstance(successful, Mapping), "successful replay evidence is absent")
    state_metrics = successful.get("state_metrics")
    _require(
        successful.get("exit_code") == 0
        and successful.get("hook_restoration_guard_completed") is True
        and _valid_sha256(successful.get("summary_result_sha256"))
        and isinstance(state_metrics, Mapping)
        and state_metrics.get("passed") is True
        and state_metrics.get("finite") is True
        and all(
            math.isfinite(float(value))
            for value in state_metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "v8.1 admission replay did not complete with finite passing metrics",
    )
    successful_boundary = {
        "contact_conditioned_action_result_sha256": None,
        "contact_conditioned_action_used": False,
        "future_object_tracks_present": False,
        "future_robot_action_available": True,
        "object_observation_frames_used": [0],
        "post_initial_object_observation_used": False,
        "prediction_only_input_required": True,
        "simulator_residual_used": False,
        "target_access": False,
    }
    _require(
        successful.get("information_boundary") == successful_boundary,
        "successful v8.1 replay information boundary changed",
    )
    outputs = successful.get("outputs")
    _require(
        isinstance(outputs, Mapping) and set(outputs) == _V81_REPLAY_OUTPUT_NAMES,
        "successful v8.1 replay output set changed",
    )
    output_payloads = {
        name: _validate_replay_bound_file(
            outputs[name],
            _V8_ADMISSION_REPLAY_ROOT / name,
            role=f"v8.1 admission replay output {name}",
        )
        for name in sorted(_V81_REPLAY_OUTPUT_NAMES)
    }
    _validate_replay_bound_file(
        successful.get("stdout_log"),
        _V8_ADMISSION_REPLAY_ROOT / "stdout.log",
        role="successful v8.1 admission replay stdout",
    )
    _validate_replay_bound_file(
        successful.get("stderr_log"),
        _V8_ADMISSION_REPLAY_ROOT / "stderr.log",
        role="successful v8.1 admission replay stderr",
    )
    try:
        summary = json.loads(output_payloads["twin_summary.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v8.1 replay twin summary is not JSON") from error
    expected_summary_outputs = {
        "episode_graph": outputs["episode_graph.npz"]["sha256"],
        "simulator_final_data": outputs["simulator_final_data.pkl"]["sha256"],
        "state_artifact": outputs["state_artifact.npz"]["sha256"],
    }
    _require(
        isinstance(summary, Mapping)
        and summary.get("passed") is True
        and summary.get("result_sha256")
        == successful.get("summary_result_sha256")
        == successful.get("validator_result_sha256")
        and summary.get("state_metrics") == state_metrics
        and summary.get("information_boundary") == successful_boundary,
        "v8.1 replay report differs from its bound twin summary",
    )
    _require(
        summary.get("output_sha256") == expected_summary_outputs
        and successful.get("graph") == summary.get("graph")
        and successful.get("capacity_diagnostic") == summary.get("capacity_diagnostic")
        and successful.get("prediction_input_validation")
        == summary.get("prediction_input_validation"),
        "v8.1 replay diagnostics or output hashes differ from the sealed summary",
    )

    rejection = report.get("cross_authorization_rejection")
    _require(
        isinstance(rejection, Mapping)
        and rejection.get("attempted_case_name") == _V81_CROSS_AUTHORIZATION_CASE_NAME
        and isinstance(rejection.get("exit_code"), int)
        and not isinstance(rejection.get("exit_code"), bool)
        and rejection.get("exit_code") == 1
        and rejection.get("rejected") is True
        and isinstance(rejection.get("numerical_output_count"), int)
        and not isinstance(rejection.get("numerical_output_count"), bool)
        and rejection.get("numerical_output_count") == 0
        and rejection.get("stderr_marker") == _V81_CROSS_AUTHORIZATION_STDERR_MARKER
        and rejection.get("stderr_marker_present") is True
        and all(
            not os.path.lexists(_V8_ADMISSION_REPLAY_ROOT / "cross-auth" / output_name)
            for output_name in _V81_REPLAY_OUTPUT_NAMES
        ),
        "v8.1 replay cross-authorization was not rejected before numerical output",
    )
    _validate_replay_bound_file(
        rejection.get("stdout_log"),
        _V8_ADMISSION_REPLAY_ROOT / "cross-auth" / "stdout.log",
        role="v8.1 cross-authorization stdout",
    )
    rejection_stderr = _validate_replay_bound_file(
        rejection.get("stderr_log"),
        _V8_ADMISSION_REPLAY_ROOT / "cross-auth" / "stderr.log",
        role="v8.1 cross-authorization stderr",
    )
    _require(
        _V81_CROSS_AUTHORIZATION_STDERR_MARKER.encode("utf-8") in rejection_stderr,
        "cross-authorization stderr lacks the exact admission-rejection marker",
    )


def _import_v8_modules(code: Path) -> tuple[Any, Any, Any]:
    source_root = code / "src"
    sys.path.insert(0, str(source_root))
    try:
        from bayesian_phystwin import deform360_held_v8_builders as builders
        from bayesian_phystwin import deform360_held_v8_protocol as protocol
        from bayesian_phystwin import (
            deform360_held_v8_replacement_source as replacement,
        )
    finally:
        sys.path.pop(0)
    for module, label in (
        (protocol, "protocol"),
        (replacement, "replacement"),
        (builders, "builders"),
    ):
        module_path = Path(module.__file__).resolve()
        _require(
            module_path.is_relative_to(source_root),
            f"{label} module imported outside the clean source tree",
        )
    return protocol, replacement, builders


def _processing_revision() -> tuple[str, str]:
    code = _absolute(_DEFORM360_CODE)
    _require(code.is_dir() and code.resolve() == code, "Deform360 code is absent")
    _require_deployed_read_only(code)
    _require(
        os.lstat(code).st_mode & 0o222 == 0,
        "Deform360 processing snapshot root is writable",
    )
    provenance = _validate_repository(code)
    _require(
        _run_git(
            code,
            ["ls-files", "--others", "--ignored", "--exclude-standard"],
        ).stdout
        == b"",
        "Deform360 processing snapshot contains ignored files",
    )
    tree = (
        _run_isolated_filemode_git(code, ["rev-parse", "HEAD^{tree}"])
        .stdout.decode("ascii")
        .strip()
        .lower()
    )
    _require(
        provenance["head"] == _V81_DEFORM360_HEAD and tree == _V81_DEFORM360_TREE,
        "Deform360 processing snapshot HEAD or tree changed",
    )
    return str(provenance["head"]), tree


def prospective_bindings(
    source_code: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    provenance = _validate_repository(source_code)
    code = provenance["root"]
    _require(
        Path(__file__).resolve()
        == code / "scripts" / "held" / "prepare_deform360_v8_lock.py",
        "lock preparer is not the tracked source operator",
    )
    external = _external_bindings()
    inherited = _inherited_v7_bindings()
    bindings = dict(inherited)
    bindings["v7_inherited_immutable_bindings_contract"] = hashlib.sha256(
        _canonical_bytes(inherited)
    ).hexdigest()
    bindings.update(external)
    local_bindings = _local_file_bindings(code)
    _validate_attempt2_operator_source_lineage(local_bindings)
    _validate_attempt3_archive_lineage(local_bindings)
    bindings.update(local_bindings)
    bindings["v8_attempt3_postseal_noncode_inventory"] = (
        _V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
    )
    bindings["v8_attempt3_postseal_noncode_inventory_contract"] = hashlib.sha256(
        _canonical_bytes(_attempt3_archive_inventory_contract())
    ).hexdigest()
    protocol, replacement, builders = _import_v8_modules(code)
    _validate_admission_replay_source_lineage(local_bindings, builders, code)
    processing_revision, processing_tree = _processing_revision()
    _require(
        processing_revision == replacement.PROCESSING_CODE_REVISION,
        "Deform360 processing revision changed",
    )
    bindings.update(
        {
            "method_deployed_snapshot_tree": provenance["tree_sha256"],
            "method_head_text_sha256": provenance["head_text_sha256"],
            "replacement_source_inventory_contract": protocol.held_contract_sha256(
                replacement.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
            ),
            "replacement_automatic_twin_admission_contract": (
                protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
            ),
            "frame_zero_exact_eight_subset_bounded_audit_contract": (
                protocol.frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256
            ),
            "frozen_query_field_contract": protocol.held_contract_sha256(
                protocol.FROZEN_FIELD_CONTRACT
            ),
            "center_exclusion_contract": (
                protocol.query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256
            ),
            "primary_method_contract": protocol.held_contract_sha256(
                protocol.PRIMARY_METHOD
            ),
            "deform360_processing_head_text_sha256": _sha256_text(processing_revision),
            "deform360_processing_tree_text_sha256": _sha256_text(processing_tree),
            "hf_dataset_revision_text_sha256": _sha256_text(
                replacement.HF_DATASET_REVISION
            ),
        }
    )
    _require(
        all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in bindings.values()
        ),
        "prospective binding is not SHA-256",
    )
    return dict(sorted(bindings.items())), provenance


def _disclosure_environment() -> dict[str, str]:
    return {
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
        "USER": "florianpfaff",
    }


def _seal_disclosure(source_code: Path) -> None:
    operator = (
        source_code
        / "scripts"
        / "held"
        / "seal_deform360_v8_post_withdrawal_disclosure.py"
    )
    completed = subprocess.run(
        [str(_PINNED_PYTHON), "-I", "-B", str(operator)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=source_code,
        env=_disclosure_environment(),
    )
    _require(
        completed.returncode == 0,
        "disclosure sealer failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def create_lock_and_deployment(source_code: str | Path) -> dict[str, Any]:
    _require(
        socket.gethostname() == "workstation2", "formal lock must run on workstation2"
    )
    _require(not os.path.lexists(_HELD_ROOT), "formal held-v8 root is not fresh")
    bindings, provenance = prospective_bindings(source_code)
    source = provenance["root"]
    head = provenance["head"]
    stage = _HELD_BASE / f".held-v8-code-stage-{head}"
    destination = _HELD_ROOT / f"code-{head}"
    staged = _clone_staged_deployment(source, head, stage)
    _require(
        staged["tree_sha256"] == provenance["tree_sha256"],
        "staged deployment tree differs from source",
    )
    protocol, _replacement, _builders = _import_v8_modules(source)
    capability = protocol.prepare_fresh_held_root(_HELD_ROOT)
    deployment_moved = False
    try:
        _seal_disclosure(source)
        disclosure_sha256 = _sha256_file(
            _DISCLOSURE_PATH,
            role="post-withdrawal disclosure",
            required_mode=0o400,
        )
        bindings["post_withdrawal_development_use_disclosure"] = disclosure_sha256
        final_external = _external_bindings()
        _require(
            all(bindings.get(name) == value for name, value in final_external.items()),
            "external immutable input changed during lock preparation",
        )
        lock = protocol.create_calibration_protocol_lock(
            _LOCK_PATH,
            held_root=_HELD_ROOT,
            fresh_root_capability=capability,
            immutable_bindings=bindings,
            v7_withdrawal_report_path=_V7_WITHDRAWAL,
            attempt3_withdrawal_report_path=_V8_ATTEMPT3_WITHDRAWAL_REPORT,
            attempt3_withdrawal_pointer_path=_V8_ATTEMPT3_WITHDRAWAL_POINTER,
            attempt3_withdrawal_integrity_completion_path=(
                _V8_ATTEMPT3_INTEGRITY_COMPLETION
            ),
            post_withdrawal_disclosure_path=_DISCLOSURE_PATH,
            development_decision_path=_OPEN27_DECISION,
        )
        _require(not os.path.lexists(destination), "deployment destination exists")
        # The Corsair filesystem refuses to rename a directory whose own
        # owner-write bit is absent, even though POSIX rename ordinarily only
        # requires write permission on the two parents.  Descendants remain
        # immutable; expose the staging root bit only for the atomic move and
        # remove it again before any deployed validation or execution.
        os.chmod(stage, 0o755, follow_symlinks=False)
        os.rename(stage, destination)
        deployment_moved = True
        os.chmod(destination, 0o555, follow_symlinks=False)
        _require_deployed_read_only(destination)
        deployed = _validate_repository(destination)
        _require(
            deployed["head"] == head
            and deployed["tree_sha256"] == bindings["method_deployed_snapshot_tree"],
            "deployed repository differs after atomic move",
        )
        validated = protocol.validate_protocol_lock(_LOCK_PATH)
        _require(validated == lock, "calibration lock changed after deployment")
        return {
            "operation": "created_held_v8_calibration_lock_and_deployment",
            "protocol_id": lock["protocol_id"],
            "lock_path": str(_LOCK_PATH),
            "lock_file_sha256": _sha256_file(
                _LOCK_PATH, role="calibration lock", required_mode=0o400
            ),
            "lock_artifact_sha256": lock["artifact_sha256"],
            "deployed_code": str(destination),
            "deployed_head": head,
            "deployed_tree_sha256": deployed["tree_sha256"],
            "binding_count": len(bindings),
            "formal_root_was_absent": True,
        }
    except BaseException:
        if not deployment_moved and os.path.lexists(stage):
            # The stage is outside the formal root and contains no outcome.
            # It is safe to remove; the formal root is deliberately retained
            # as incident evidence and must never be silently retried as v8.
            for root, directories, files in os.walk(stage, topdown=False):
                for name in files:
                    os.chmod(Path(root) / name, 0o600, follow_symlinks=False)
                for name in directories:
                    os.chmod(Path(root) / name, 0o700, follow_symlinks=False)
            os.chmod(stage, 0o700, follow_symlinks=False)
            shutil.rmtree(stage)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--create", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    _require(
        sys.flags.isolated == 1 and sys.dont_write_bytecode,
        "run lock preparation with Python -I -B",
    )
    if arguments.preflight:
        bindings, provenance = prospective_bindings(arguments.source_code)
        result = {
            "operation": "preflight_only",
            "formal_root_absent": not os.path.lexists(_HELD_ROOT),
            "source_head": provenance["head"],
            "source_tree_sha256": provenance["tree_sha256"],
            "prospective_binding_count": len(bindings) + 1,
            "prospective_bindings": bindings,
        }
    else:
        result = create_lock_and_deployment(arguments.source_code)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
