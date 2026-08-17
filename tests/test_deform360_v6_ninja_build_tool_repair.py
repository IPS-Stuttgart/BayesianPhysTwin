from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_ninja_build_tool.json"
)
BOOTSTRAP = ROOT / "scripts/ci/bootstrap_deform360_v6_ninja.sh"
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-prediction-evidence.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"


def test_ninja_build_tool_repair_is_content_addressed_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    declared = payload.pop("repair_id")

    assert declared == content_id(payload)
    assert declared == (
        "4cee24a0db69c0f8902e6f58a492a0186be7b17c7f19a3e74ef06f3d781a6292"
    )
    assert payload["predecessor_cuda_host_compiler_repair_id"] == (
        "01a5b25972e5b254bfd0ed40fadfd3417532519869d70f404acedf64b98147e0"
    )
    failed = payload["failed_execution_evidence"]
    assert failed["workflow_run_id"] == 31576200607
    assert failed["artifact_id"] == 9133317708
    assert failed["artifact_digest"] == (
        "sha256:d4727c8ad965bc005e6ce9412a56779331258eac7fdde837caf15467a7c576b1"
    )
    assert failed["execution_receipt_id"] == (
        "51d39d1bd939bd8b04d003dfefca89c7cb09548561fa401067859944b9388320"
    )
    assert failed["physical_manifest_count"] == 0
    assert failed["source_prediction_seal_count"] == 0
    assert failed["cuda_host_compiler_repair_activated"] is True
    correction = payload["correction"]
    assert correction["version"] == "1.13.0"
    assert correction["wheel_sha256"] == (
        "fb46acf6b93b8dd0322adc3a4945452a4e774b75b91293bafcc7b7f8e6517dfa"
    )
    assert correction["executable_sha256"] == (
        "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67"
    )
    assert correction["runtime_bin_prepended_to_path"] is True
    assert not any(payload["information_boundary"].values())
    assert payload["repair_scope"]["ninja_build_tool_completed"] is True
    assert all(
        value is False
        for key, value in payload["repair_scope"].items()
        if key != "ninja_build_tool_completed"
    )


def test_ninja_bootstrap_is_exact_and_precedes_gsplat_import() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    file_sha256 = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()

    assert file_sha256 == (
        "576b134583437f76d264a4814cd357e80f9f413bed911519ae83a0787e15e4c1"
    )
    assert f"NINJA_BUILD_TOOL_REPAIR_ID: {payload['repair_id']}" in workflow
    assert f"NINJA_BUILD_TOOL_REPAIR_SHA256: {file_sha256}" in workflow
    assert str(AMENDMENT.relative_to(ROOT)) in workflow
    assert str(BOOTSTRAP.relative_to(ROOT)) in workflow
    call = 'source "scripts/ci/bootstrap_deform360_v6_ninja.sh" "${runtime}"'
    assert workflow.count(call) == 1
    assert workflow.index(call) > workflow.index(
        '"${runtime_python}" -m pip install -e ".[graph,vision]"'
    )
    assert workflow.index(call) < workflow.index("from gsplat.cuda._backend")
    for token in (
        "ninja==1.13.0",
        payload["correction"]["wheel_sha256"],
        payload["correction"]["executable_sha256"],
        "--ignore-installed",
        "--no-deps",
        "--only-binary=:all:",
        "--require-hashes",
        "torch.utils.cpp_extension import is_ninja_available",
        'export PATH="${runtime}/bin:${PATH}"',
        'printf \'%s\\n\' "${runtime}/bin" >> "${GITHUB_PATH}"',
    ):
        assert token in bootstrap
    assert '"runtime_ninja_build_tool_repair"' in workflow
    assert '"runtime_ninja_build_tool_repair"' in runner


def test_ninja_bootstrap_does_not_weaken_information_boundaries() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    forbidden = (
        "development_suffix",
        "confirmation_payload",
        "target_payload",
        "target_outcome",
    )
    assert all(token not in bootstrap for token in forbidden)
