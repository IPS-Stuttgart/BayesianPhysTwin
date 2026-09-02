from __future__ import annotations

from scripts.science.run_interventional_transport_quotient_controlled_v1 import run


def test_transport_can_be_known_before_cause_with_fewer_probes() -> None:
    result = run(trials=2_000, seed=20260902)

    assert result["decision"] == ("transport-known-before-cause-and-probe-use-reduced")
    assert all(result["checks"].values())
    metrics = result["metrics"]
    assert metrics["source_unique_cause_coverage"] == 0.0
    assert metrics["unprobed_sum_transport_coverage"] == 1.0
    assert metrics["false_unprobed_sensitive_transport_rate"] == 0.0
    assert metrics["transport_quotient_probe_rate"] == 0.5
    assert metrics["probe_rate_reduction"] == 0.5
    assert metrics["transport_quotient_rmse"] <= (
        metrics["full_cause_identification_rmse"] + 0.002
    )
