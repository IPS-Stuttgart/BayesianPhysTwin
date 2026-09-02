"""Generate controlled evidence for adaptive stochastic policy-tree certificates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.adaptive_stochastic_policy_tree_v1 import (
    ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY,
    AdaptivePolicyTreeCertificateV1,
    adaptive_stochastic_policy_tree_certificate,
)

CONTRACT = "adaptive-stochastic-policy-tree-controlled-v1"


def _binary_sensor(values: np.ndarray, accuracy: float) -> np.ndarray:
    result = np.empty((values.size, 2), dtype=np.float64)
    result[:, 0] = np.where(values == 0, accuracy, 1.0 - accuracy)
    result[:, 1] = 1.0 - result[:, 0]
    return result


def _binary_information_nats(accuracy: float) -> float:
    error = 1.0 - accuracy
    entropy = 0.0
    for probability in (error, 1.0 - error):
        entropy -= probability * math.log(probability)
    return math.log(2.0) - entropy


def _problem(maximum_depth: int) -> AdaptivePolicyTreeCertificateV1:
    hypotheses = np.asarray(
        list(itertools.product((0, 1), repeat=4)),
        dtype=np.int64,
    )
    route = hypotheses[:, 0]
    x_value = hypotheses[:, 1]
    y_value = hypotheses[:, 2]
    nuisance = hypotheses[:, 3]
    target = np.where(route == 0, x_value, y_value)
    terminal_losses = np.empty((hypotheses.shape[0], 3), dtype=np.float64)
    terminal_losses[:, 0] = target
    terminal_losses[:, 1] = 1 - target
    terminal_losses[:, 2] = 0.45
    return adaptive_stochastic_policy_tree_certificate(
        np.ones(hypotheses.shape[0]),
        [1.0],
        np.zeros(hypotheses.shape[0], dtype=np.int64),
        terminal_losses,
        [
            _binary_sensor(route, 0.98),
            _binary_sensor(x_value, 0.95),
            _binary_sensor(y_value, 0.95),
            _binary_sensor(nuisance, 0.999),
        ],
        [0.025, 0.036, 0.036, 0.001],
        fallback_action_index=2,
        maximum_depth=maximum_depth,
        regret_tolerance=0.20,
        probe_names=["route", "x", "y", "nuisance"],
        max_policy_count=5000,
        max_raw_tree_count=500000,
    )


def _best_fixed_two_probe_regret(
    certificate: AdaptivePolicyTreeCertificateV1,
) -> float:
    eligible: list[int] = []
    for index, policy in enumerate(certificate.policies):
        if policy.mode == "act" or policy.depth < 2:
            eligible.append(index)
            continue
        if policy.depth != 2 or not policy.children:
            continue
        if all(child.mode == "sense" for child in policy.children):
            second_probes = {child.probe_index for child in policy.children}
            if len(second_probes) == 1:
                eligible.append(index)
    return float(np.min(certificate.worst_case_regret[eligible]))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_result() -> dict[str, Any]:
    depth_zero = _problem(0)
    depth_one = _problem(1)
    depth_two = _problem(2)
    selected = depth_two.output_policy
    if (
        depth_zero.output_mode != "fallback"
        or depth_one.output_mode != "fallback"
        or depth_two.output_mode != "sense"
        or selected.probe_index != 0
        or tuple(child.probe_index for child in selected.children) != (1, 2)
    ):
        raise RuntimeError("controlled strict-separation construction changed")

    adaptive_regret = float(
        depth_two.worst_case_regret[depth_two.output_policy_index]
    )
    direct_regret = float(
        depth_zero.worst_case_regret[depth_zero.fallback_policy_index]
    )
    one_probe_regret = float(
        depth_one.worst_case_regret[depth_one.fallback_policy_index]
    )
    fixed_two_probe_regret = _best_fixed_two_probe_regret(depth_two)
    information = {
        "route": _binary_information_nats(0.98),
        "x": _binary_information_nats(0.95),
        "y": _binary_information_nats(0.95),
        "nuisance": _binary_information_nats(0.999),
    }
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "problem": {
            "hypothesis_count": 16,
            "latent_bits": ["route", "x", "y", "nuisance"],
            "terminal_actions": ["action_0", "action_1", "fallback"],
            "probe_names": list(depth_two.probe_names),
            "probe_costs": depth_two.probe_costs.tolist(),
            "regret_tolerance": depth_two.regret_tolerance,
            "all_probe_outcomes_have_full_hypothesis_support": True,
        },
        "information_nats": information,
        "highest_information_probe": max(information, key=information.get),
        "policies": {
            "direct_only": {
                "output_mode": depth_zero.output_mode,
                "worst_case_regret": direct_regret,
            },
            "at_most_one_probe": {
                "output_mode": depth_one.output_mode,
                "worst_case_regret": one_probe_regret,
            },
            "fixed_nonadaptive_two_probe": {
                "best_worst_case_regret": fixed_two_probe_regret,
            },
            "adaptive_depth_two": {
                "output_mode": depth_two.output_mode,
                "worst_case_regret": adaptive_regret,
                "first_probe": depth_two.probe_names[selected.probe_index],
                "second_probe_by_first_outcome": [
                    depth_two.probe_names[child.probe_index]
                    for child in selected.children
                ],
                "uses_highest_information_nuisance_probe": (
                    "nuisance" in selected.canonical_key
                ),
                "minimum_expected_loss_over_hypotheses": float(
                    np.min(selected.expected_loss_by_hypothesis)
                ),
                "maximum_expected_loss_over_hypotheses": float(
                    np.max(selected.expected_loss_by_hypothesis)
                ),
                "raw_tree_count": depth_two.raw_tree_count,
                "retained_policy_count": depth_two.policy_count,
                "loss_equivalent_tree_count": (
                    depth_two.loss_equivalent_tree_count
                ),
                "dominance_pruned_tree_count": (
                    depth_two.dominance_pruned_tree_count
                ),
                "structure": selected.structure(),
            },
        },
        "strict_separation": {
            "direct_and_one_probe_fall_back": (
                depth_zero.used_fallback and depth_one.used_fallback
            ),
            "fixed_two_probe_cannot_beat_fallback": (
                fixed_two_probe_regret >= direct_regret - 1e-12
            ),
            "adaptive_policy_is_certified": not depth_two.used_fallback,
            "complete_state_remains_unidentified": True,
            "adaptive_regret_reduction_vs_fallback_fraction": (
                1.0 - adaptive_regret / direct_regret
            ),
        },
        "interpretation": (
            "The complete adaptive tree is certified before sensing. It first "
            "measures which latent task coordinate matters and then acquires only "
            "that coordinate. A cheaper and more accurate nuisance probe has the "
            "largest state-information gain but is never selected. Full-support "
            "noise leaves every physical hypothesis possible at every leaf."
        ),
        "claim_boundary": ADAPTIVE_STOCHASTIC_POLICY_TREE_CLAIM_BOUNDARY,
    }
    result["result_id"] = _canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    result = build_result()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
