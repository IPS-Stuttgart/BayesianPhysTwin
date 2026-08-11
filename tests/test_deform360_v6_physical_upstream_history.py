from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/ci/run_deform360_v6_source_prediction_evidence.sh"
PINNED_REVISION = "9f69d5d6c5d81d6d6e8f123c18ddba73dc4afa65"
DIAGNOSTIC_RUN_ID = "31461017011"
DIAGNOSTIC_REPORT_ID = (
    "75c1be85233e1835dfef5a1227a28e8938995335ead701fe8d3dfd8b5960a087"
)

FROZEN_PATHS = {
    "scripts/remote/run_deform360_official_phystwin_smoke.py",
    "src/causal4d_public/deform360_reusable_graph.py",
    "src/causal4d_public/deform360_partial_graph_state.py",
    "src/causal4d_public/deform360_dense_reusable_panel.py",
    "src/causal4d_public/deform360_action_support.py",
    "src/causal4d_public/deform360_contact_conditioned_action.py",
    "src/causal4d_public/deform360_dense_source.py",
    "src/bayesian_phystwin/phystwin_graph.py",
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json",
    "configs/causal4d_public/deform360_independent_source_split_v1.json",
}


def test_physical_upstream_pin_records_unique_target_blind_diagnostic() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert f'PHYSICAL_UPSTREAM_REVISION="{PINNED_REVISION}"' in text
    assert f'PHYSICAL_UPSTREAM_DIAGNOSTIC_RUN_ID="{DIAGNOSTIC_RUN_ID}"' in text
    assert f'PHYSICAL_UPSTREAM_REPORT_ID="{DIAGNOSTIC_REPORT_ID}"' in text
    assert "unique-complete-history-exact-ten-file-sha256-match" in text
    assert '["git", "-C", str(repository), "show"' in text
    assert '"fetch",' in text
    assert '"--depth=1",' in text
    assert '"origin",\n            revision,' in text
    assert "rev-list" not in text
    assert "--all" not in text


def test_frozen_physical_upstream_pin_materializes_exact_registered_bytes(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").is_dir():
        pytest.skip("git repository is unavailable")

    output = tmp_path / "physical-upstream"
    env = dict(os.environ)
    env["BPT_PYTHON"] = sys.executable
    completed = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--materialize-physical-upstream",
            str(ROOT),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == PINNED_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", completed.stdout.strip())
    subprocess.run(
        ["git", "cat-file", "-e", f"{PINNED_REVISION}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    for relative in FROZEN_PATHS:
        path = output / relative
        assert path.is_file()
        assert not path.is_symlink()
