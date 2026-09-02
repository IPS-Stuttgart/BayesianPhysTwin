from __future__ import annotations

from scripts.science.run_target_directed_intervention_design_controlled_v1 import (
    run,
)


def test_target_directed_design_reduces_intervention_cost() -> None:
    result = run()

    assert result["decision"] == (
        "target-identification-cost-strictly-below-full-cause-identification"
    )
    assert all(result["checks"].values())
    metrics = result["metrics"]
    assert metrics["targets_identified"] == metrics["targets_total"] == 3
    assert metrics["zero_probe_targets"] == 1
    assert metrics["one_probe_targets"] == 2
    assert metrics["relative_mean_cost_reduction"] >= 2.0 / 3.0 - 1e-12
