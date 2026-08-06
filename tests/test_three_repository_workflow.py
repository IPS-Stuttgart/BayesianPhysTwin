from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "three-repository-golden-path.yml"
SCRIPT = ROOT / "scripts" / "run_three_repository_golden_path.sh"


def test_three_repository_workflow_uses_transfer_safe_repositories() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/Causal4D" in text
    assert "PROB4D_CHECKOUT_REPOSITORY: IPS-Stuttgart/Prob4D" in text
    assert "repository: ${{ env.PROB4D_CHECKOUT_REPOSITORY }}" in text
    assert "repository: FlorianPfaff/Prob4D" not in text
    assert "repository: FlorianPfaff/Causal4D" not in text


def test_three_repository_workflow_accepts_external_compatibility_dispatches() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository_dispatch:" in text
    assert "prob4d-compatibility" in text
    assert "causal4d-compatibility" in text
    assert "github.event.client_payload.prob4d_ref" in text
    assert "github.event.client_payload.causal4d_ref" in text


def test_three_repository_workflow_pins_external_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "actions/upload-artifact@v" not in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text


def test_three_repository_workflow_requires_public_prob4d_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Check out selected Prob4D producer" in text
    assert "PROB4D_READ_TOKEN" not in text
    assert "prob4d-access" not in text
    assert "https://api.github.com/repos/IPS-Stuttgart/Prob4D" not in text
    assert "steps.prob4d-access.outputs.available" not in text
    assert "permissions:\n  contents: read" in text
    assert text.count("persist-credentials: false") >= 8


def test_three_repository_workflow_uses_lock_and_nonblocking_canary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Resolve committed ecosystem lock" in text
    assert "data/ecosystem_compatibility_v1.json" in text
    assert "needs.resolve-lock.outputs.prob4d_ref" in text
    assert "needs.resolve-lock.outputs.causal4d_ref" in text
    assert "needs.resolve-lock.outputs.lock_enforced" in text
    assert "Latest Prob4D and Causal4D main canary" in text
    assert "continue-on-error: true" in text
    assert "THREE_REPOSITORY_REQUIRE_LOCKED_REVISIONS" in text
    assert "three-repository-compatibility.json" in text


def test_installed_wheel_script_runs_ecosystem_validation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"${TEST_VENV}/bin/bpt" "${compatibility_arguments[@]}"' in text
    assert "ecosystem validate" in text
    assert "--require-all" in text
    assert "--exact-versions" in text
    assert '--revision "prob4d=${PROB4D_REVISION}"' in text
    assert '--revision "causal4d=${CAUSAL4D_REVISION}"' in text
    assert "THREE_REPOSITORY_COMPATIBILITY_REPORT" in text


def test_three_repository_workflow_tracks_prospective_belief_surfaces() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required_paths = (
        "src/bayesian_phystwin/causal4d_belief_provider_v2.py",
        "src/bayesian_phystwin/endpoint_model_average.py",
        "src/bayesian_phystwin/prospective_prob4d_update.py",
        "tests/test_endpoint_model_average.py",
        "tests/test_prospective_prob4d_update.py",
        "docs/prospective_belief_updates_v1.md",
    )
    for path in required_paths:
        assert f'- "{path}"' in text
