#!/usr/bin/env python3
"""Specify the v4 chance guard from the already-open v2/v3 wrapping studies."""

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
    SENSOR_SEED as V2_SENSOR_SEED,
)
from bayesian_phystwin_experiments.dlolab_wrapping_resolution_ensemble_v3 import (
    SENSOR_SEED as V3_SENSOR_SEED,
)
from bayesian_phystwin_experiments.dlolab_wrapping_risk_guard_v4 import (
    DEVELOPMENT_V2_RESULT_ID,
    DEVELOPMENT_V3_RESULT_ID,
    REWARD_MARGIN,
    infer_risk_decisions,
    posterior_guard_actions,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1")
V2 = Path("/home/fpfaff/source-only/dlolab-wrapping-continuous-interp-source-v2")
V3 = Path("/home/fpfaff/source-only/dlolab-wrapping-resolution-ensemble-source-v3")
OUTPUT = ROOT / "results/sota/dlolab_wrapping_risk_guard_development_v4/summary.json"
PARENT_FILE_SHA256 = {
    "source-bank/arrays.npz": (
        "914bd948df92e8b829ac65ca8c075c789d122a63a9ec32807a302bef16e2271d"
    ),
    "source-bank/seal.json": (
        "143686ee40ddfb8456e23cded5c8225015e60bd678a4f77bd4074946c33fe14f"
    ),
}
DEVELOPMENT_FILE_SHA256 = {
    "v2": {
        "decisions/arrays.npz": (
            "35cf83c309a2e82f3eb24baf2666ee73ebf17de74d1a1f71b8ef5262041e98dd"
        ),
        "generation/arrays.npz": (
            "2a7717f20fabaa3ae8eef4ee4004d1c078cc513117fdc3fce5f370d365dc9605"
        ),
        "result.json": (
            "716e8d65abeeddaf95935c08bc7269a07ff63cc6f24d457f801a9d5bfa7cbc98"
        ),
    },
    "v3": {
        "decisions/arrays.npz": (
            "f812fc6e7249d5b29a1a1327ee29cb751299829f5489878cdffbf63075ef96f7"
        ),
        "generation/arrays.npz": (
            "9922b5eb64d0b9fab00874c48f65612146897140b594a590b2799170dd3459b7"
        ),
        "result.json": (
            "bfb02757a9879a0263388635e5fc5173cf806371af78511c1f3b50007539bc08"
        ),
    },
}
CANDIDATE_PROBABILITIES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _check_files(root: Path, expected: dict[str, str]) -> None:
    if any(
        not (root / name).is_file()
        or (root / name).is_symlink()
        or file_digest(root / name) != digest
        for name, digest in expected.items()
    ):
        raise ValueError("registered wrapping development evidence changed")


def _metrics(
    decisions: np.ndarray[Any, Any], reward: np.ndarray[Any, Any]
) -> dict[str, Any]:
    selected = np.take_along_axis(reward[:, None, :], decisions[:, :, None], axis=2)[
        :, :, 0
    ].mean(axis=1)
    fixed = reward[:, 4]
    gain = selected - fixed
    return {
        "world_reward": selected,
        "gain": gain,
        "mean_native_reward": float(selected.mean()),
        "mean_gain_over_fixed": float(gain.mean()),
        "worlds_harmed_beyond_numeric_margin": int(
            np.count_nonzero(gain < -REWARD_MARGIN)
        ),
        "mean_downside_below_fixed": float(np.maximum(-gain, 0).mean()),
        "nonfixed_sensor_decisions": int(np.count_nonzero(decisions != 4)),
    }


def _cohort(
    name: str,
    root: Path,
    *,
    result_id: str,
    sensor_seed: int,
    old_columns: dict[str, int],
    bank: dict[str, np.ndarray[Any, Any]],
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    _check_files(root, DEVELOPMENT_FILE_SHA256[name])
    result = read_record(root / "result.json")
    if (
        result.get("artifact_id") != result_id
        or result.get("status") != "complete"
        or result.get("source_gate_passed") is not False
    ):
        raise ValueError("complete failed wrapping development result required")
    old = _load_npz(root / "decisions" / "arrays.npz")
    reward = np.asarray(
        _load_npz(root / "generation" / "arrays.npz")["reward"], dtype=np.float64
    )
    inferred = infer_risk_decisions(
        bank["prefix"],
        bank["reward"],
        old["truth_prefix_m"],
        sensor_draws=old["decisions"].shape[1],
        sensor_seed=sensor_seed,
    )
    decisions = inferred["decisions"]
    reproduction = {
        old_name: bool(
            np.array_equal(
                decisions[:, :, new_column], old["decisions"][:, :, old_column]
            )
        )
        for old_name, (new_column, old_column) in {
            "fixed": (0, old_columns["fixed"]),
            "finite_particle_bayes": (5, old_columns["finite_particle_bayes"]),
            "continuous_bayes": (1, old_columns["continuous_bayes"]),
            "continuous_map": (6, old_columns["continuous_map"]),
        }.items()
    }
    if not all(reproduction.values()):
        raise ValueError("registered wrapping decisions were not reproduced")
    expected = inferred["continuous_posterior_expected_reward"]
    probability = inferred["continuous_posterior_improvement_probability"]
    candidates: dict[float, dict[str, Any]] = {}
    for threshold in CANDIDATE_PROBABILITIES:
        candidate = np.empty(expected.shape[:2], dtype=np.int64)
        for world in range(expected.shape[0]):
            candidate[world] = posterior_guard_actions(
                expected[world],
                probability[world],
                threshold=float(threshold),
                fixed_action=4,
            )
        candidates[threshold] = _metrics(candidate, reward)
    return (
        {
            "worlds": int(reward.shape[0]),
            "sensor_draws_per_world": int(decisions.shape[1]),
            "registered_arm_reproduction": reproduction,
            "continuous_bayes": _metrics(decisions[:, :, 1], reward),
            "finite_particle_bayes": _metrics(decisions[:, :, 5], reward),
            "continuous_map": _metrics(decisions[:, :, 6], reward),
        },
        candidates,
    )


def _public_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item for key, item in value.items() if key not in {"world_reward", "gain"}
    }


def audit() -> dict[str, Any]:
    _check_files(PARENT, PARENT_FILE_SHA256)
    seal = read_record(PARENT / "source-bank" / "seal.json")
    bank = load_native_bundle(PARENT / "source-bank", seal["bundle"])
    v2, v2_candidates = _cohort(
        "v2",
        V2,
        result_id=DEVELOPMENT_V2_RESULT_ID,
        sensor_seed=V2_SENSOR_SEED,
        old_columns={
            "fixed": 0,
            "finite_particle_bayes": 1,
            "continuous_bayes": 3,
            "continuous_map": 2,
        },
        bank=bank,
    )
    v3, v3_candidates = _cohort(
        "v3",
        V3,
        result_id=DEVELOPMENT_V3_RESULT_ID,
        sensor_seed=V3_SENSOR_SEED,
        old_columns={
            "fixed": 0,
            "finite_particle_bayes": 1,
            "continuous_bayes": 2,
            "continuous_map": 5,
        },
        bank=bank,
    )
    candidate_sweep: dict[str, Any] = {}
    for threshold in CANDIDATE_PROBABILITIES:
        rows = [v2_candidates[threshold], v3_candidates[threshold]]
        reward = np.concatenate([row["world_reward"] for row in rows])
        gain = np.concatenate([row["gain"] for row in rows])
        candidate_sweep[str(threshold)] = {
            "mean_native_reward": float(reward.mean()),
            "mean_gain_over_fixed": float(gain.mean()),
            "worlds_harmed_beyond_numeric_margin": int(
                np.count_nonzero(gain < -REWARD_MARGIN)
            ),
            "mean_downside_below_fixed": float(np.maximum(-gain, 0).mean()),
            "worlds": int(gain.size),
        }
    selected = candidate_sweep[str(0.975)]
    if selected["worlds_harmed_beyond_numeric_margin"] != 0 or any(
        candidate_sweep[str(value)]["worlds_harmed_beyond_numeric_margin"] == 0
        for value in CANDIDATE_PROBABILITIES
        if value < 0.975
    ):
        raise ValueError("registered chance-threshold selection changed")
    return {
        "schema": "dlolab-wrapping-risk-guard-development-v4",
        "status": "post_open_development_diagnostic",
        "parent_file_sha256": PARENT_FILE_SHA256,
        "development_file_sha256": DEVELOPMENT_FILE_SHA256,
        "development_result_ids": {
            "v2": DEVELOPMENT_V2_RESULT_ID,
            "v3": DEVELOPMENT_V3_RESULT_ID,
        },
        "cohorts": {
            "v2": {
                key: _public_metrics(value) if isinstance(value, dict) else value
                for key, value in v2.items()
            },
            "v3": {
                key: _public_metrics(value) if isinstance(value, dict) else value
                for key, value in v3.items()
            },
        },
        "candidate_probability_sweep": candidate_sweep,
        "selected_probability": 0.975,
        "selection_rule": (
            "smallest_registered_probability_with_zero_harmed_worlds_across_v2_v3"
        ),
        "selected_development_metrics": selected,
        "fresh_stress_panel_normalized_log_ranges": {
            "stretching": [0.60, 0.995],
            "bending": [0.02, 0.70],
        },
        "stress_panel_reason": (
            "source-and-open-development action-switch region; prospective worlds unseen"
        ),
        "lead_is_not_prospective_evidence": True,
        "development_v2_v3_results_reclassified": False,
        "future_experiment_automatically_authorized": False,
        "protected_data_read": False,
        "held_v8_read": False,
        "dlo4_dlo5_read": False,
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
