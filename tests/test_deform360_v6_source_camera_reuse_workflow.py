from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-v6-source-camera-reuse.yml"
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_camera_reuse.sh"
DOCUMENT = ROOT / "docs/deform360_v6_source_camera_reuse.md"


def test_workflow_runs_empirical_reuse_only_by_authorized_protected_dispatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "permissions:\n  contents: read" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert text.index("name: Set up Python", text.index("execute:")) < text.index(
        "name: Materialize and seal the target-closed recovery panel"
    )
    assert "actions: read" in text
    assert "inputs.execute_authorized == true" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.ref_protected == true" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "cancel-in-progress: false" in text
    assert "execution-started.txt" in text
    assert "RUN_CLAIM" in text
    assert "os.O_EXCL" in RUNNER.read_text(encoding="utf-8")
    assert "durable execution root belongs to another run" in text
    assert 'test ! -L "${RUN_ROOT}"' in text
    assert text.index('test ! -L "${RUN_ROOT}"') < text.index('mkdir "${compact}"')
    assert "retain-failure" in text
    assert "root.rglob" in (
        ROOT / "scripts/science/materialize_deform360_v6_source_camera_reuse.py"
    ).read_text(encoding="utf-8")
    assert "source-camera-reuse-technical-failure-retained" in text
    assert "preterminal-execution-receipt.json" in text
    assert 'if [[ "${status}" != "0" && -f "${receipt}" ]]' in text
    assert text.count("if: always() && steps.retain.outputs.owns_run == 'true'") == 2
    assert 'test "${{ steps.retain.outputs.owns_run }}" = "true"' in text
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "git push" not in text


def test_workflow_pins_one_successful_base_artifact_and_existing_products() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'BASE_RUN_ID: "31585420194"' in text
    assert 'BASE_ARTIFACT_ID: "9137481740"' in text
    assert "6bb16bb307349c50535b1b368c60dfb4d5d17ab9" in text
    assert "d811ff1ea4d6ad22a6e7476d2911602af1ee71f81339c0737e1cc60fd5883f9d" in text
    assert "cf3ebb9e69eb3c15051ba4ae39e2d0338ec244e0c49e587a277f7b36344c5f3d" in text
    assert "f1cd4ccfb8281a167718a30e5a6af1caaf740ba7a9d49081638efaabdeaf8441" in text
    assert "5cc43432eb509b98442d289ec884b30780ff26c76ab8654826d000bb4832e3b3" in text
    assert "c8b26c0ea5dd26cd0406282a0cc01659afc557cbda74615b1702a35a9c970180" in text
    assert "146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89" in text
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in text
    assert (
        'test "$(jq -r \'.digest\' <<<"${artifact}")" = "${BASE_ARTIFACT_DIGEST}"'
        in text
    )
    assert 'test ! -e "${output}"' in text
    assert "${AMENDMENT_ID}/${GITHUB_SHA}" not in text
    assert "source-camera-reuse/${AMENDMENT_ID}" in text
    assert 'mkdir "${RUN_ROOT}"' in RUNNER.read_text(encoding="utf-8")


def test_runner_uses_only_prefix_reuse_and_seals_before_any_outcome_access() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    combined = f"{text}\n{workflow}"

    assert "audit-base" in text
    assert "rank-reuse" in text
    assert "build-combined-plan" in text
    assert "audit-combined" in text
    assert "build-lineage" in text
    assert "freeze-source-plan" in text
    assert "run_deform360_joint_sparse_source_predictions_v5_2.py" in text
    assert "seal-execution" in text
    assert "execution-started.txt" in text
    assert "score_deform360_joint_sparse_source_v5_2.py" not in combined
    assert (
        "materialize_deform360_joint_sparse_source_endpoint_plan_v5_2.py"
        not in combined
    )
    assert "run_deform360_joint_sparse_motioncrafter_source_v5.py" not in combined
    assert "export_prob4d_uniform.py" not in combined
    assert "source_suffix_opened=false" in workflow
    assert "new_provider_inference_run=false" in workflow
    assert "prob4d_used=false" in workflow
    assert "confirmation_payloads_opened=false" in workflow
    assert "target_outcomes_used=false" in workflow


def test_document_states_real_measurement_and_no_human_approval_boundary() -> None:
    text = " ".join(DOCUMENT.read_text(encoding="utf-8").split())

    assert "real-world robot manipulation recordings" in text
    assert "requires no new recording and no human approval" in text
    assert "A person cannot choose cameras" in text
    assert "No provider inference is run" in text
    assert "It authorizes neither independent confirmation nor a claim" in text
