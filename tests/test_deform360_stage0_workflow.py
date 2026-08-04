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
    validation = (
        "- name: Revalidate target-blind implementation on publication runner"
    )
    selection = "- name: Resolve official revision and select from names/metadata only"
    assert validation in selection_job
    assert selection in selection_job
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


def test_stage0_workflow_keeps_hosted_and_self_hosted_validation() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("tests/test_deform360_stage0_workflow.py") >= 3
    assert "name: Contact-anchor and selection contracts" in workflow
    assert "name: Names and metadata only / workstation2" in workflow
