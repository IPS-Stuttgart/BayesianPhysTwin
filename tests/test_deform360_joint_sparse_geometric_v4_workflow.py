from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / ".github/workflows/launch-deform360-joint-sparse-geometric-v4-once.yml"
CONTRACTS = ROOT / ".github/workflows/deform360-joint-sparse-geometric-v4-contracts.yml"


def test_launch_is_protected_main_only_and_uses_the_sole_runner() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert "pull_request:" not in text
    assert "pull_request_target" not in text
    assert "workflow_dispatch:" not in text
    assert "push:\n    branches: [main]\n    paths:" in text
    assert "runs-on: self-hosted" in text
    assert 'test "${RUNNER_NAME}" = "${AUTHORIZED_RUNNER_NAME}"' in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.repository == 'IPS-Stuttgart/BayesianPhysTwin'" in text
    assert "contents: read" in text
    assert "issues: write" in text
    assert "contents: write" not in text
    assert "actions: write" not in text


def test_launch_passes_only_retained_result_roots_to_the_materializer() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    command = text.split(
        "scripts/science/materialize_deform360_joint_sparse_geometric_v4.py",
        maxsplit=1,
    )[1].split("--output-dir", maxsplit=1)[0]
    assert '--metric-batch-root "${METRIC_BATCH_ROOT}"' in command
    assert '--prediction-root "${PREDICTION_ROOT}"' in command
    assert (
        '--production-result "${PREDICTION_ROOT}/visual-production-result.json"'
        in command
    )
    assert "DEFORM360_OFFICIAL_RAW_ROOT" not in command
    assert "DEFORM360_ADAPTIVE_CONFIRMATION_RAW_ROOT" not in command
    assert "data-7fea8e2" not in command
    assert "adaptive-confirmation-download" not in command
    assert "point_map" not in text
    assert "prediction_point_values_used=false" in text
    assert "prediction_residuals_used=false" in text
    assert "adaptive_confirmation_payloads_opened=false" in text
    assert "confirmation_payloads_opened=false" in text
    assert "target_outcomes_used=false" in text
    assert "replacement_allowed=false" in text


def test_launch_materializes_then_evaluates_and_retains_both_terminals() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert "materialize_deform360_joint_sparse_geometric_v4.py" in text
    assert "evaluate_deform360_joint_sparse_observability_v4.py" in text
    assert 'test "${status}" -eq 0 -o "${status}" -eq 3' in text
    assert "materialization-result.json" in text
    assert "development-report.json" in text
    assert "execution-receipt.json" in text
    assert "actions/upload-artifact@v7" in text
    assert 'ISSUE_NUMBER: "148"' in text
    assert "confirmation_access_authorized" in text
    assert "published issue comment" in text
    assert "retention-days: 180" in text


def test_pull_request_contract_workflow_is_hosted_and_read_only() -> None:
    text = CONTRACTS.read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: self-hosted" not in text
    assert "permissions:\n  contents: read" in text
    assert "issues: write" not in text
    assert "contents: write" not in text
    assert "pull_request_target" not in text
    assert "materialize_deform360_joint_sparse_geometric_v4.py" in text
    assert "test_deform360_joint_sparse_geometric_v4.py" in text
    assert "test_pull_request_workflow_integrity.py" in text
