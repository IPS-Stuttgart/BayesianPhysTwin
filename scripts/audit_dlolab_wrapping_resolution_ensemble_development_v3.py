#!/usr/bin/env python3
"""Reproduce the post-open v2 lead used to specify the v3 controller."""

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
from bayesian_phystwin_experiments.dlolab_slingshot_process import load_native_bundle
from bayesian_phystwin_experiments.dlolab_wrapping_continuous_interp_v2 import (
    REWARD_MARGIN,
    SENSOR_DRAWS,
    SENSOR_SEED,
    WORLD_COUNT,
)
from bayesian_phystwin_experiments.dlolab_wrapping_resolution_ensemble_v3 import (
    ARM_NAMES,
    DEVELOPMENT_V2_RESULT_ID,
    infer_resolution_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
DEVELOPMENT = Path(
    "/home/fpfaff/source-only/dlolab-wrapping-continuous-interp-source-v2"
)
OUTPUT = (
    ROOT
    / "results/sota/dlolab_wrapping_resolution_ensemble_development_v3/summary.json"
)
PARENT_FILE_SHA256 = {
    "source-bank/arrays.npz": (
        "914bd948df92e8b829ac65ca8c075c789d122a63a9ec32807a302bef16e2271d"
    ),
    "source-bank/seal.json": (
        "143686ee40ddfb8456e23cded5c8225015e60bd678a4f77bd4074946c33fe14f"
    ),
}
DEVELOPMENT_FILE_SHA256 = {
    "decisions/arrays.npz": (
        "35cf83c309a2e82f3eb24baf2666ee73ebf17de74d1a1f71b8ef5262041e98dd"
    ),
    "generation/arrays.npz": (
        "2a7717f20fabaa3ae8eef4ee4004d1c078cc513117fdc3fce5f370d365dc9605"
    ),
    "result.json": ("716e8d65abeeddaf95935c08bc7269a07ff63cc6f24d457f801a9d5bfa7cbc98"),
}


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def audit() -> dict[str, Any]:
    for root, expected in (
        (PARENT, PARENT_FILE_SHA256),
        (DEVELOPMENT, DEVELOPMENT_FILE_SHA256),
    ):
        if any(
            not (root / name).is_file()
            or (root / name).is_symlink()
            or file_digest(root / name) != digest
            for name, digest in expected.items()
        ):
            raise ValueError("registered v2 development evidence changed")
    result = read_record(DEVELOPMENT / "result.json")
    if (
        result.get("artifact_id") != DEVELOPMENT_V2_RESULT_ID
        or result.get("status") != "complete"
        or result.get("source_gate_passed") is not False
    ):
        raise ValueError("complete failed v2 development result required")
    parent_seal = read_record(PARENT / "source-bank" / "seal.json")
    bank = load_native_bundle(PARENT / "source-bank", parent_seal["bundle"])
    old = _load_npz(DEVELOPMENT / "decisions" / "arrays.npz")
    generation = _load_npz(DEVELOPMENT / "generation" / "arrays.npz")
    inferred = infer_resolution_decisions(
        bank["prefix"],
        bank["reward"],
        old["truth_prefix_m"],
        sensor_draws=SENSOR_DRAWS,
        sensor_seed=SENSOR_SEED,
    )
    decisions = inferred["decisions"]
    old_decisions = old["decisions"]
    reproduction = {
        "fixed": bool(np.array_equal(decisions[:, :, 0], old_decisions[:, :, 0])),
        "finite_particle_bayes": bool(
            np.array_equal(decisions[:, :, 1], old_decisions[:, :, 1])
        ),
        "continuous_bayes": bool(
            np.array_equal(decisions[:, :, 2], old_decisions[:, :, 3])
        ),
        "continuous_map": bool(
            np.array_equal(decisions[:, :, 5], old_decisions[:, :, 2])
        ),
    }
    reward = np.asarray(generation["reward"], dtype=np.float64)
    if (
        not all(reproduction.values())
        or decisions.shape != (WORLD_COUNT, SENSOR_DRAWS, len(ARM_NAMES))
        or reward.shape != (WORLD_COUNT, 8)
    ):
        raise ValueError("v2 observation replay did not reproduce registered arms")
    selected = np.take_along_axis(reward[:, None, :], decisions, axis=2).mean(axis=1)
    fixed = selected[:, 0]
    arms: dict[str, Any] = {}
    for index, name in enumerate(ARM_NAMES):
        gain = selected[:, index] - fixed
        arms[name] = {
            "mean_native_reward": float(selected[:, index].mean()),
            "mean_gain_over_fixed": float(gain.mean()),
            "worlds_harmed_beyond_numeric_margin": int(
                np.count_nonzero(gain < -REWARD_MARGIN)
            ),
        }
    ensemble = selected[:, ARM_NAMES.index("equal_resolution_ensemble")]
    finite_gain = arms["finite_particle_bayes"]["mean_gain_over_fixed"]
    ensemble_gain = arms["equal_resolution_ensemble"]["mean_gain_over_fixed"]
    return {
        "schema": "dlolab-wrapping-resolution-ensemble-development-v3",
        "status": "post_open_development_diagnostic",
        "development_v2_result_id": DEVELOPMENT_V2_RESULT_ID,
        "parent_file_sha256": PARENT_FILE_SHA256,
        "development_file_sha256": DEVELOPMENT_FILE_SHA256,
        "model_resolution_weights": {"finite": 0.5, "continuous": 0.5},
        "registered_v2_arm_reproduction": reproduction,
        "arms": arms,
        "ensemble_mean_gain_vs_finite": float((ensemble - selected[:, 1]).mean()),
        "ensemble_mean_gain_vs_continuous": float((ensemble - selected[:, 2]).mean()),
        "finite_gain_fraction_retained": float(ensemble_gain / finite_gain),
        "ensemble_decisions_different_from_finite": int(
            np.count_nonzero(decisions[:, :, 3] != decisions[:, :, 1])
        ),
        "ensemble_decisions_different_from_continuous": int(
            np.count_nonzero(decisions[:, :, 3] != decisions[:, :, 2])
        ),
        "worlds": WORLD_COUNT,
        "sensor_draws_per_world": SENSOR_DRAWS,
        "lead_is_not_prospective_evidence": True,
        "v2_source_gate_reclassified": False,
        "future_experiment_authorized": False,
        "protected_data_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve() or args.output.exists():
        raise ValueError("fresh registered development diagnostic output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = write_record(args.output, audit())
    print(record["artifact_id"])


if __name__ == "__main__":
    main()
