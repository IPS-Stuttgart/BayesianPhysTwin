"""Controlled active decision-identification mechanism.

The experiment contrasts three probes over four latent states:

* no probe: no state or decision information;
* decision probe: identifies which terminal action is optimal but leaves two
  latent states unresolved;
* state probe: identifies the complete state at higher cost.

The exact active certificate should select the cheaper decision probe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bayesian_phystwin.active_decision_probe_v1 import (
    ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY,
    active_decision_probe_certificate,
    decision_probe_candidate,
    select_minimum_cost_decision_probe,
)


def entropy_bits(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def expected_posterior_state_entropy(
    prior: np.ndarray,
    likelihood: np.ndarray,
) -> float:
    outcome_probability = prior @ likelihood
    result = 0.0
    for outcome, probability in enumerate(outcome_probability):
        if probability <= 0.0:
            continue
        posterior = prior * likelihood[:, outcome] / probability
        result += float(probability) * entropy_bits(posterior)
    return result


def run() -> dict[str, object]:
    prior = np.full(4, 0.25, dtype=np.float64)
    quotient = np.asarray([1.0], dtype=np.float64)
    classes = np.zeros(4, dtype=np.int64)
    losses = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    no_probe_likelihood = np.ones((4, 1), dtype=np.float64)
    decision_probe_likelihood = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    state_probe_likelihood = np.eye(4, dtype=np.float64)

    probes = (
        decision_probe_candidate(
            "no_probe",
            0.0,
            no_probe_likelihood,
            losses,
        ),
        decision_probe_candidate(
            "decision_probe",
            1.0,
            decision_probe_likelihood,
            losses,
        ),
        decision_probe_candidate(
            "state_probe",
            4.0,
            state_probe_likelihood,
            losses,
        ),
    )
    selection = select_minimum_cost_decision_probe(
        prior,
        quotient,
        classes,
        probes,
        regret_tolerance=0.0,
    )
    records: list[dict[str, object]] = []
    likelihoods = (
        no_probe_likelihood,
        decision_probe_likelihood,
        state_probe_likelihood,
    )
    for probe, likelihood, certificate in zip(
        probes,
        likelihoods,
        selection.certificates,
        strict=True,
    ):
        records.append(
            {
                "name": probe.name,
                "cost": probe.cost,
                "outcome_count": int(likelihood.shape[1]),
                "minimax_worst_case_regret": (
                    certificate.minimax_worst_case_regret
                ),
                "terminal_policy": (
                    certificate.minimax_terminal_policy.tolist()
                ),
                "expected_posterior_state_entropy_bits": (
                    expected_posterior_state_entropy(prior, likelihood)
                ),
                "state_identified": math.isclose(
                    expected_posterior_state_entropy(prior, likelihood),
                    0.0,
                    abs_tol=1e-12,
                ),
                "decision_identified": (
                    certificate.minimax_worst_case_regret <= 1e-12
                ),
            }
        )

    result = {
        "schema": "bayesian-phystwin/active-decision-probe-controlled-v1",
        "schema_version": 1,
        "current_state_entropy_bits": entropy_bits(prior),
        "registered_regret_tolerance": 0.0,
        "probes": records,
        "selection": selection.summary(),
        "claim_boundary": ACTIVE_DECISION_PROBE_CLAIM_BOUNDARY,
    }
    if selection.selected_probe_name != "decision_probe":
        raise RuntimeError("controlled mechanism did not select decision probe")
    if records[1]["state_identified"] is not False:
        raise RuntimeError("decision probe unexpectedly identified full state")
    if records[1]["decision_identified"] is not True:
        raise RuntimeError("decision probe failed to identify the decision")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"result mismatch: {args.check}")
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
