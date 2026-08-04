from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "three-repository-golden-path.yml"


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
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text


def test_three_repository_workflow_probes_repository_access() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "id: prob4d-access" in text
    assert "https://api.github.com/repos/IPS-Stuttgart/Prob4D" in text
    assert "github.event_name != 'pull_request'" in text
    assert "github.event_name != 'push'" in text
    assert (
        "Scheduled, manual, and repository-dispatch validation remains fail-closed"
        in text
    )


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
