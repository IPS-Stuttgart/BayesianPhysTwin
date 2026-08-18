from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "evidence" / "public_claim_snapshot_v1.json"
RENDERER_PATH = ROOT / "scripts" / "render_public_claim_status.py"


def _load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_public_claim_status",
        RENDERER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_claim_snapshot_is_pinned_and_bounded() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    renderer = _load_renderer()
    renderer.validate_snapshot(snapshot, root=ROOT)

    claims = {claim["id"]: claim for claim in snapshot["claims"]}
    assert claims["full22_point_improvement"]["status"] == "confirmed_with_boundary"
    assert claims["unique_deterministic_winner"]["status"] == "not_confirmed"
    assert claims["raw_covariance_calibration"]["status"] == "refuted"
    assert (
        claims["covariance_only_independent_validation"]["status"] == "not_established"
    )
    assert claims["fresh_object_session_v61"]["status"] == "terminal_without_claim"
    assert claims["prob4d_real_provider_transfer"]["status"] == "not_established"
    assert claims["causal4d_downstream_benefit"]["status"] == "not_established"


def test_readme_public_claim_table_is_generated_from_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    renderer = _load_renderer()
    renderer.validate_snapshot(snapshot, root=ROOT)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = renderer.replace_status_block(
        readme,
        renderer.render_status_block(snapshot),
    )
    assert readme == expected
