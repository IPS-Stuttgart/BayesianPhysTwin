from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / (".github/workflows/deform360-v61-source-scorer-cudnn-supply.yml")
RETIRED_WORKFLOW = ROOT / ".github/workflows/deform360-v61-source-scorer.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v61_source_scorer.sh"
DOCUMENT = ROOT / "docs/deform360_fresh_object_session_source_scoring_v6_1.md"
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_1_source_scoring.json"
)
RUNTIME_LOCK = ROOT / (
    "requirements/locks/deform360-v61-source-scorer-pt24cu121-py310.txt"
)
RUNTIME_REPAIR = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_1_cudnn_supply_runtime.json"
)


def test_workflow_runs_one_protected_source_execution_without_confirmation() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert not RETIRED_WORKFLOW.exists()
    assert "# workflow-lifecycle: temporary" in workflow
    assert "# workflow-issue: #645" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "inputs.execute_authorized == true" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "github.ref_protected == true" in workflow
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "deform360-v61-source-scorer-cudnn-supply-v1" in workflow
    assert "os.O_EXCL" in runner
    assert 'test ! -e "${output}"' in workflow
    assert 'test ! -e "${output}.claim"' in workflow
    assert 'test ! -e "${endpoint}"' in workflow
    assert "independent_confirmation_authorized=false" in workflow
    assert "confirmation_payloads_opened=false" in workflow
    assert "contents: write" not in workflow
    assert "git push" not in workflow


def test_authorized_dispatch_cannot_pass_after_runtime_or_admission_skip() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Require one complete one-shot terminal decision" in workflow
    assert 'test "${{ steps.runtime.outputs.exit_code }}" = "0"' in workflow
    assert 'test "${{ steps.admit.outputs.admitted }}" = "true"' in workflow
    assert 'test "${{ steps.execute.outcome }}" != "skipped"' in workflow
    assert 'test "${{ steps.retain.outputs.owns_run }}" = "true"' in workflow


def test_workflow_binds_the_exact_candidate_barrier_and_public_source() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for value in (
        "e8b962a8abf228114495683cfb9ba87ee802e7405ca92f1f28b5a76df3faa371",
        "c616fe1fbe19785452535772adfa937501a0fa35ab41b3c2fc995a968e60a8f1",
        "65747822fa8380296a572811772fce88b9275a7e1148a8015e1156f520f7e369",
        "db3cc4351436492db5962bc1e99f516adc38a5031140b675b45dc6d752b7559a",
        "d27674518f523db4fddb9cc108dd3d77321dddefeccc866b2b81044bf44ebee8",
        "d9b9e4df9d020e8ae076f407f61d5e1f328c68d2f4fe4d8e4ad1688d2d253100",
        "2eb8d12e2120d58d0d678c3771d29faaeb765497",
        "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317",
        "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
    ):
        assert value in workflow
    assert "candidate-panel-receipt.json" in workflow
    assert "raw-nested-prediction-batch.json" in workflow
    assert ".technical_failure_record_count'" in workflow
    assert "= 0" in workflow


def test_runtime_is_precompiled_hash_locked_and_never_jit_compiled() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    combined = f"{workflow}\n{runner}"

    assert "1.4.0+pt24cu121" in workflow
    assert (
        "2efb8b8f4ad3275db05707fa6f9cf110482e7fd269c78a4cc7dc5b08cfc957ff" in workflow
    )
    assert (
        "e0b664c9d6f355e611bdfa720103b86b399ded3dcc5ecfaf59eaade992f1359b" in workflow
    )
    assert 'pip install --no-deps "${wheel}"' in workflow
    assert 'pip install --no-deps "${cudnn_wheel}"' in workflow
    assert "nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl" in workflow
    assert (
        "165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f" in workflow
    )
    assert 'version("nvidia-cudnn-cu12") != "9.1.0.70"' in workflow
    assert workflow.index('pip install --no-deps "${cudnn_wheel}"') < workflow.index(
        'pip install -r "${CUDA_RUNTIME_LOCK_PATH}"'
    )
    assert "gsplat/csrc.so" in workflow
    assert "build.ninja" not in combined
    assert "nvcc" not in combined.lower()


def test_runner_scores_only_after_authorization_and_retains_failures() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    authorization = runner.index("  authorize \\")
    suffix_marker = runner.index("source-suffix-opened.txt")
    endpoint = runner.index("process_deform360_fresh_object_session_source_endpoint")
    score = runner.index("  score \\")
    assert authorization < suffix_marker < endpoint < score
    assert "worker 0 &" in runner
    assert "worker 1 &" in runner
    assert "source-scoring-receipt.json" in runner
    assert "source-score-exit-code.txt" in runner
    assert '"${score_status}" -ne 0 && "${score_status}" -ne 3' in runner
    for forbidden in (
        "confirmation-payload",
        "target-outcome",
        "held-v8",
        "export_prob4d_uniform.py",
        "new_motioncrafter",
    ):
        assert forbidden not in runner.lower()


def test_source_scoring_artifacts_are_packaged_and_documented() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    document = " ".join(DOCUMENT.read_text(encoding="utf-8").split())

    for relative in (
        "docs/deform360_fresh_object_session_source_scoring_v6_1.md",
        "protocols/amendments/deform360_official_hub_fresh_object_session_v6_1_source_scoring.json",
        "protocols/amendments/deform360_official_hub_fresh_object_session_v6_1_cudnn_supply_runtime.json",
        "requirements/locks/deform360-v61-source-scorer-pt24cu121-py310.txt",
        "scripts/ci/run_deform360_v61_source_scorer.sh",
        "scripts/remote/process_deform360_fresh_object_session_source_endpoint_v6_1.py",
        "scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py",
    ):
        assert f"include {relative}" in manifest
    assert "public real-world RGB recordings" in document
    assert "collects no new measurement and requires no human approval" in document
    assert "decoded-uniform Prob4D overlap fusion is unused" in document
    assert "never becomes a scored loss" in document
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == (
        "c616fe1fbe19785452535772adfa937501a0fa35ab41b3c2fc995a968e60a8f1"
    )
    assert hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest() == (
        "e46e32b809fd9438437cf0ff4138dccb119904b5f1d9f90900df99603f278af3"
    )


def test_cudnn_supply_repair_is_content_addressed_and_science_preserving() -> None:
    repair = json.loads(RUNTIME_REPAIR.read_text(encoding="utf-8"))
    declared = repair.pop("repair_id")

    assert declared == content_id(repair)
    assert declared == (
        "afc4753c60e48062b6ae3b0789a6d924bb832d2b8b186c4d450ee2ca75dbf0ca"
    )
    failure = repair["failed_execution_evidence"]
    assert failure["workflow_run_id"] == 31660983482
    assert failure["source_revision"] == ("9a18d3a4dd4aa95c69308f184c77958ddc4eec8d")
    assert failure["source_suffix_opened"] is False
    assert failure["durable_run_claim_created"] is False
    assert failure["artifact_count"] == 0
    assert repair["information_boundary"]["confirmation_payloads_opened"] is False
    assert repair["information_boundary"]["target_outcomes_opened"] is False
    assert repair["information_boundary"]["held_v8_artifacts_accessed"] is False

    scope = repair["repair_scope"]
    assert scope["supply_route_changed"] is True
    for field, changed in scope.items():
        if field != "supply_route_changed":
            assert changed is False, field

    scientific = repair["scientific_identity"]
    expected = {
        "protocols/amendments/deform360_official_hub_fresh_object_session_v6_1_source_scoring.json": "source_scoring_amendment_file_sha256",
        "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json": "execution_lock_file_sha256",
        "scripts/ci/run_deform360_v61_source_scorer.sh": "scorer_runner_sha256",
        "scripts/remote/process_deform360_fresh_object_session_source_endpoint_v6_1.py": "endpoint_processor_sha256",
        "scripts/science/run_deform360_fresh_object_session_source_scorer_v6_1.py": "scorer_cli_sha256",
        "src/bayesian_phystwin/deform360_fresh_object_session_source_scorer_v6_1.py": "scorer_library_sha256",
    }
    for relative, field in expected.items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == scientific[field], relative
