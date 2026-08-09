from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/inventory-deform360-v4-results-once.yml"


def test_inventory_workflow_is_results_only_and_self_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in text
    assert "issues: write" not in text
    assert "pull_request_target" not in text
    assert "runs-on: self-hosted" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "github.event_name != 'pull_request'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert '--root "${DEFORM360_RESULTS_ROOT}"' in text
    assert '--forbidden-root "${DEFORM360_OFFICIAL_RAW_ROOT}"' in text
    assert '--forbidden-root "${DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT}"' in text
    assert "binary_scientific_payloads_loaded=false" in text
    assert "raw_dataset_payloads_opened=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=false" in text


def test_inventory_workflow_does_not_install_on_the_self_hosted_runner() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    inventory_job = text.split("  inventory:\n", maxsplit=1)[1]
    assert "pip install" not in inventory_job
    assert "actions/setup-python" not in inventory_job
    assert "command -v python3 || command -v python" in inventory_job
    assert "scripts/ci/inventory_deform360_v4_results.py" in inventory_job
    assert "actions/upload-artifact@v7" in inventory_job


def test_inventory_trigger_is_path_bounded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "push:\n    branches: [main]\n    paths:" in text
    assert '      - ".github/workflows/inventory-deform360-v4-results-once.yml"' in text
    assert '      - "scripts/ci/inventory_deform360_v4_results.py"' in text
    assert "workflow_dispatch:" in text
