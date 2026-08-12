from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v61-candidate-producer.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v61_candidate_producer.sh"
DOCUMENT = ROOT / "docs/deform360_fresh_object_session_candidate_v6_1.md"
FROZEN_V5_PUBLIC_INPUTS = (
    ROOT / "src/bayesian_phystwin/deform360_joint_sparse_public_inputs_v5.py"
)


def test_workflow_runs_exactly_one_protected_prefix_only_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "# workflow-lifecycle: temporary" in text
    assert "# workflow-issue: #642" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "workflow_dispatch:" in text
    assert "inputs.execute_authorized == true" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.ref_protected == true" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "cancel-in-progress: false" in text
    assert "RUN_CLAIM" in text
    assert "os.O_EXCL" in runner
    assert "durable candidate root belongs to another run" in text
    assert "retain-failure" in text
    assert "preterminal-execution-receipt.json" in text
    assert 'test "${{ steps.retain.outputs.owns_run }}" = "true"' in text
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "git push" not in text


def test_workflow_binds_the_exact_sealed_upstream_and_amendment() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for value in (
        "593ca6c08ee5430ad37bc0cc5bb3d1b79d77a049714cb36a8c78696f3c68cfee",
        "6b5b5b99bea29c2927d52fb6ae52623f1d154a038932b6922900b709c30b10db",
        "76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189",
        "d9b9e4df9d020e8ae076f407f61d5e1f328c68d2f4fe4d8e4ad1688d2d253100",
        "5b1cdf3f047b52665650dcbf56d8ec205ced8788e2cdd0e528793a9ece9387f0",
        "04a5a8b71603b66850e35122405bc24c5de1e766c14cc2b58974f1ea97fb49ef",
        "a408e44eaecf9e63311a2f1a6f511f130e586031e8a0e8e795d58fa5696e3026",
    ):
        assert value in text
    assert 'test ! -e "${output}"' in text
    assert "${CANDIDATE_AMENDMENT_ID}/${GITHUB_SHA}" not in text


def test_v61_keeps_the_v5_public_input_implementation_byte_exact() -> None:
    assert hashlib.sha256(FROZEN_V5_PUBLIC_INPUTS.read_bytes()).hexdigest() == (
        "6ee7dcf93768151496c1b52b11d501418b85c3caa8ed0849bd1295c11cb3c9c7"
    )


def test_workflow_and_runner_have_no_suffix_or_provider_execution_interface() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    combined = f"{text}\n{runner}"

    assert "publish-panel" in runner
    assert "validate-panel" in runner
    assert "seal-execution" in runner
    for forbidden in (
        "deform360_joint_sparse_source_scoring_v5",
        "score_deform360_joint_sparse_source_v5_2.py",
        "materialize_deform360_joint_sparse_source_endpoint_plan_v5_2.py",
        "evaluate_deform360_fresh_object_session_source_v6",
        "run_deform360_joint_sparse_motioncrafter_source_v5.py",
        "export_prob4d_uniform.py",
        "confirmation-payload",
        "target-outcome",
    ):
        assert forbidden not in combined
    assert "source_suffix_opened=false" in text
    assert "prob4d_pipeline_artifacts_reused=true" in text
    assert "prob4d_decoded_uniform_fusion_used=false" in text
    assert "motioncrafter_disjoint_baseline_used=true" in text
    assert "new_prob4d_inference_run=false" in text
    assert "new_motioncrafter_inference_run=false" in text
    assert "public_tactile_axis_identity_available=false" in text


def test_document_states_public_data_and_no_human_approval_boundary() -> None:
    text = " ".join(DOCUMENT.read_text(encoding="utf-8").split())

    assert "public Deform360 real-world recordings" in text
    assert "No new physical measurement or human approval is required" in text
    assert "consumes `baseline_disjoint.npz`" in text
    assert "not Prob4D decoded-uniform overlap fusion" in text
    assert "does not authorize suffix scoring" in text
