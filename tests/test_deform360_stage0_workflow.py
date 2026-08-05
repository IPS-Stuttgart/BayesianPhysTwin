from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "deform360-official-hub-visuotactile.yml"
)


def test_self_hosted_stage0_publication_has_an_independent_contract_gate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    selection_job = workflow.split("\n  official-hub-selection:\n", 1)[1]

    assert "\n    needs: contracts\n" not in selection_job
    initialization = "- name: Initialize evidence directory"
    setup = "- name: Set up Python"
    validation = "- name: Revalidate target-blind implementation on publication runner"
    selection = "- name: Resolve official revision and select from names/metadata only"
    assert initialization in selection_job
    assert setup in selection_job
    assert validation in selection_job
    assert selection in selection_job
    assert selection_job.index(initialization) < selection_job.index(setup)
    assert selection_job.index(setup) < selection_job.index(validation)
    assert selection_job.index(validation) < selection_job.index(selection)

    for command in (
        "python -m ruff check",
        "python -m ruff format --check",
        "python -m pytest -q",
        "python -m compileall -q",
    ):
        assert command in selection_job
    for test_path in (
        "tests/test_deform360_contact_anchor.py",
        "tests/test_deform360_hub_selection.py",
        "tests/test_deform360_stage0_workflow.py",
    ):
        assert test_path in selection_job


def test_stage0_workflow_uses_portable_python_and_runner_scoped_artifacts() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    selection_job = workflow.split("\n  official-hub-selection:\n", 1)[1]
    job_header = selection_job.split("\n    steps:\n", 1)[0]
    setup_block = selection_job.split("\n      - name: Set up Python\n", 1)[1].split(
        "\n      - name: Install metadata-only client\n",
        1,
    )[0]

    evidence_declaration = (
        "EVIDENCE_DIR: ${{ runner.temp }}/deform360-official-hub-stage0"
    )
    setup_action = "uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    assert "runner.temp" not in job_header
    assert selection_job.count(evidence_declaration) >= 6
    assert setup_action in setup_block
    assert "timeout-minutes: 10" in setup_block
    assert "cache:" not in setup_block
    assert "cache-dependency-path:" not in setup_block
    assert "python3 -m venv" not in selection_job
    assert 'echo "EVIDENCE_DIR=' not in selection_job
    assert "path: ${{ runner.temp }}/deform360-official-hub-stage0" in selection_job


def test_stage0_workflow_keeps_hosted_and_self_hosted_validation() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("tests/test_deform360_stage0_workflow.py") >= 3
    assert "name: Contact-anchor and selection contracts" in workflow
    assert "name: Names and metadata only / workstation2" in workflow


def test_stage0_workflow_verifies_lock_without_mutating_the_branch() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    selection_job = workflow.split("\n  official-hub-selection:\n", 1)[1]

    assert "- name: Verify committed Stage-0 selection lock" in selection_job
    assert "if: github.event_name == 'pull_request'" in selection_job
    assert "lock-verification.json" in selection_job
    assert "merge-base" in selection_job
    assert "--is-ancestor" in selection_job
    assert "persist-credentials: true" not in selection_job
    assert "persist-credentials: false" in selection_job
    assert "contents: write" not in selection_job
    assert "git push" not in selection_job
    assert "git commit" not in selection_job
