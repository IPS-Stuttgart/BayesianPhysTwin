from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_GATE_SDIST_MEMBERS = frozenset(
    {
        "docs/deform360_prob4d_source_calibration.md",
        "docs/deform360_prob4d_sample_materializer.md",
        "docs/deform360_robot_metric_prefix.md",
        "docs/deform360_prob4d_source_gate.md",
        "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json",
        "protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json",
        "protocols/locks/deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json",
        "protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json",
        "protocols/locks/deform360_official_hub_prob4d_source_gate_v1.json",
        "scripts/science/fit_deform360_prob4d_source_calibration.py",
        "scripts/science/materialize_deform360_prob4d_calibration_samples.py",
        "scripts/science/materialize_deform360_robot_metric_prefix.py",
        "scripts/science/materialize_deform360_prob4d_metric_batch.py",
        "scripts/science/evaluate_deform360_prob4d_source_gate.py",
        "scripts/science/audit_deform360_prob4d_support_stop.py",
    }
)


def test_complete_prob4d_source_gate_chain_is_declared_for_sdist() -> None:
    declared = {
        line.removeprefix("include ")
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("include ")
    }

    assert SOURCE_GATE_SDIST_MEMBERS <= declared
    assert all((ROOT / relative).is_file() for relative in SOURCE_GATE_SDIST_MEMBERS)
