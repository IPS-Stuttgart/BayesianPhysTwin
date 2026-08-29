#!/usr/bin/env python3
"""Certify the frozen v8 guard on its already-open public-simulator panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)
from bayesian_phystwin_experiments.dlolab_wrapping_certified_guard_v9 import (
    CALIBRATION_CONFIDENCE,
    CALIBRATION_V8_RESULT_ID,
    HARM_RISK_BUDGET,
    REWARD_MARGIN,
    clopper_pearson_upper,
)

ROOT = Path(__file__).resolve().parents[1]
V8 = Path.home() / "source-only/dlolab-wrapping-risk-guard-source-v8"
OUTPUT = (
    ROOT
    / "results/sota/dlolab_wrapping_certified_guard_calibration_v9/summary.json"
)
V8_FILE_SHA256 = {
    "lock.json": "810e87d7f12ce6cc767c768f0c48f742ef78bc1c72dc4a477c9a29838b40080a",
    "decision-barrier.json": (
        "add2255132ba04e010983027d55b4a4ed03884f32271555954898e5e5bc664f1"
    ),
    "decisions/arrays.npz": (
        "040be358e130e1d9c887cdf1ecc438152aca6a031c03cdc2668cb8969caf9fdc"
    ),
    "decisions/seal.json": (
        "949dc378bd431faaa08e70a00afc173d39dfd80384ebed534c34cedf89650046"
    ),
    "generation/arrays.npz": (
        "d1e7f778f0ed0fdba6a34ad08f688f8ea9031f39f71f1bb4bf5dae25e84feb6d"
    ),
    "generation/seal.json": (
        "4bfb09e1173d1195e9ffb47c673db4eee34979c5936e10f63ec30a6fbc0d79b9"
    ),
    "result.json": "d750944098428b05322af47e8727b60cb263ad7b9c8cb4a9b0ef53d091070438",
}


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def audit() -> dict[str, Any]:
    if any(
        not (V8 / name).is_file()
        or (V8 / name).is_symlink()
        or file_digest(V8 / name) != digest
        for name, digest in V8_FILE_SHA256.items()
    ):
        raise ValueError("registered v8 calibration evidence changed")
    lock = read_record(V8 / "lock.json")
    result = read_record(V8 / "result.json")
    decision = _load_npz(V8 / "decisions" / "arrays.npz")["decisions"]
    reward = np.asarray(
        _load_npz(V8 / "generation" / "arrays.npz")["reward"],
        dtype=np.float64,
    )
    protocol = lock.get("protocol", {})
    if (
        result.get("artifact_id") != CALIBRATION_V8_RESULT_ID
        or result.get("status") != "complete"
        or result.get("source_gate_passed") is not False
        or result.get("ordinary_worlds") != 144
        or result.get("technical_failures") != 0
        or result.get("replacements") != 0
        or result.get("retry_authorized") is not False
        or result.get("protected_data_read") is not False
        or protocol.get("posterior_improvement_probability_threshold") != 0.975
        or protocol.get("world_count") != 144
        or protocol.get("method") != "baseline_relative_posterior_probability_guard"
        or decision.shape != (144, 4096, 7)
        or reward.shape != (144, 8)
        or decision.dtype.kind not in "iu"
        or not np.isfinite(reward).all()
    ):
        raise ValueError("complete failed v8 calibration result required")
    fixed = decision[:, :, 0]
    guard = decision[:, :, 2]
    if np.any(fixed != 4):
        raise ValueError("registered v8 fixed action changed")
    fixed_reward = np.take_along_axis(reward[:, None, :], fixed[:, :, None], axis=2)[
        :, :, 0
    ].mean(axis=1)
    guard_reward = np.take_along_axis(reward[:, None, :], guard[:, :, None], axis=2)[
        :, :, 0
    ].mean(axis=1)
    gain = guard_reward - fixed_reward
    harm_count = int(np.count_nonzero(gain < -REWARD_MARGIN))
    upper = clopper_pearson_upper(
        harm_count,
        gain.size,
        confidence=CALIBRATION_CONFIDENCE,
    )
    if (
        harm_count != 2
        or not np.isclose(gain.mean(), 0.005042156280882724, rtol=0, atol=1e-14)
        or not np.isclose(upper, 0.04307319681566585, rtol=0, atol=1e-15)
    ):
        raise ValueError("v8 risk-certificate arithmetic changed")
    return {
        "schema": "dlolab-wrapping-certified-guard-calibration-v9",
        "status": "post_open_calibration_certificate_passed",
        "v8_file_sha256": V8_FILE_SHA256,
        "v8_result_id": CALIBRATION_V8_RESULT_ID,
        "v8_strict_source_gate_passed": False,
        "v8_strict_gate_reclassified": False,
        "candidate_policy": "posterior_975_guard",
        "candidate_threshold_registered_before_v8_outcomes": True,
        "candidate_threshold_selected_from_v8_outcomes": False,
        "statistical_unit": "independent_public_simulator_world",
        "world_count": int(gain.size),
        "harm_event": "world_mean_reward_below_fixed_by_more_than_0_002",
        "harm_count": harm_count,
        "observed_harm_fraction": float(harm_count / gain.size),
        "confidence": CALIBRATION_CONFIDENCE,
        "one_sided_exact_clopper_pearson_upper": upper,
        "risk_budget": HARM_RISK_BUDGET,
        "certificate_passed": upper <= HARM_RISK_BUDGET,
        "mean_gain_over_fixed": float(gain.mean()),
        "distributional_assumption": (
            "independent_exchangeable_worlds_from_the_registered_stress_distribution"
        ),
        "risk_budget_frozen_for_v9_after_v8_opened": True,
        "lead_is_not_v9_evidence": True,
        "v9_fresh_replication_automatically_authorized": False,
        "official_benchmark_or_sota_claim": False,
        "real_robot_or_physical_safety_claim": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
        "new_recordings": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve() or args.output.exists():
        raise ValueError("fresh registered calibration output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = write_record(args.output, audit())
    print(record["artifact_id"])


if __name__ == "__main__":
    main()
