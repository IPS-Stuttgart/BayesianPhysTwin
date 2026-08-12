"""Minimal accepted-update and exact-fallback examples for inference.v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from bayesian_phystwin.inference.v1 import (
    CompleteBeliefGuardDecisionV1,
    finalize_guarded_update,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExampleBelief:
    artifact_id: str


@dataclass(frozen=True)
class ExampleInference:
    candidate_id: str
    inference_admissible: bool


def _run_case(*, accepted: bool) -> dict[str, object]:
    label = "accepted" if accepted else "fallback"
    baseline = ExampleBelief(_digest(f"{label}:baseline"))
    candidate = ExampleBelief(_digest(f"{label}:candidate"))
    inference = ExampleInference(_digest(f"{label}:inference"), True)
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("common-domain"),
        certificate_id=_digest(f"{label}:certificate"),
        inference_admissible=True,
        regret_guard_accepted=accepted,
        reason=f"example-{label}",
        metadata={"example": label},
    )
    result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        decision,
        metadata={"example": label},
    )
    return {
        "case": label,
        "guard_accepted": accepted,
        "selection_reason": result.selection.reason,
        "selected_candidate": result.selected_candidate,
        "exact_fallback": result.exact_fallback,
        "selected_belief_id": result.selected_belief.artifact_id,
        "selected_belief_is_candidate_object": result.selected_belief is candidate,
        "selected_belief_is_baseline_object": result.selected_belief is baseline,
        "result_id": result.artifact_id,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "accepted": _run_case(accepted=True),
                "fallback": _run_case(accepted=False),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
