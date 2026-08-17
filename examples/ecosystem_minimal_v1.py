#!/usr/bin/env python3
"""Run a tiny Prob4D -> BayesianPhysTwin -> Causal4D contract smoke test.

The example verifies portable observation serialization, an accepted guarded
candidate, exact caller-owned fallback, and the Causal4D provider manifest. It
uses synthetic values and therefore establishes no real-provider or downstream
scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.causal4d_provider_v1 import causal4d_provider_manifest
from bayesian_phystwin.inference.v1 import (
    CompleteBeliefGuardDecisionV1,
    finalize_guarded_update,
)
from bayesian_phystwin.v1 import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)


@dataclass(frozen=True)
class ExampleCompleteBelief:
    """Minimal complete-belief object used only to demonstrate routing."""

    artifact_id: str
    mean_xyz_m: tuple[tuple[float, float, float], ...]
    covariance_diag_m2: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ExampleInference:
    """Minimal admissible candidate inference for the stable router."""

    candidate_id: str
    inference_admissible: bool


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def build_prob4d_observation() -> ObservationBeliefV1:
    """Build a deterministic, Prob4D-compatible synthetic observation belief."""

    source_payload = b"prob4d-synthetic-contract-smoke-v1"
    local_covariance = np.repeat(
        np.diag([4.0e-6, 4.0e-6, 9.0e-6])[None, :, :],
        2,
        axis=0,
    )
    low_rank_factor = np.asarray(
        [
            [[1.0e-3], [0.0], [0.0]],
            [[1.0e-3], [0.0], [0.0]],
        ],
        dtype=np.float64,
    )
    return ObservationBeliefV1(
        case_id="synthetic-cloth-v1",
        stream_id="prob4d-contract-smoke-v1",
        causal_frame_stop=2,
        view_names=("cam0",),
        window_names=("prefix",),
        factor_names=("shared-x",),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="synthetic-contract-smoke-v1",
        source_artifact_sha256=hashlib.sha256(source_payload).hexdigest(),
        declared_frame_ids=np.asarray([0, 1], dtype=np.int64),
        mean_xyz_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        ),
        frame_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=np.asarray([0, 0], dtype=np.int64),
        view_indices=np.asarray([0, 0], dtype=np.int64),
        window_indices=np.asarray([0, 0], dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0], dtype=np.int64),
        factor_group_ids=np.asarray([0, 0], dtype=np.int64),
        prior_reliability=np.asarray([0.95, 0.90], dtype=np.float64),
        association_probability=np.asarray([1.0, 1.0], dtype=np.float64),
        local_covariance_m2=local_covariance,
        low_rank_factor_m=low_rank_factor,
        group_ids=np.asarray([0], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([0.95], dtype=np.float64),
        group_composite_weight=np.asarray([0.5], dtype=np.float64),
        metadata={
            "example_role": "Prob4D-compatible producer contract",
            "scientific_scope": "contract-smoke-only",
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_example(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_path = output_dir / "prob4d_observation_belief_v1.npz"
    observation = build_prob4d_observation()
    save_observation_belief(observation_path, observation)
    loaded = load_observation_belief(observation_path)
    if loaded.artifact_id != observation.artifact_id:
        raise RuntimeError("portable observation round trip changed artifact identity")

    baseline = ExampleCompleteBelief(
        artifact_id=_digest("belief-baseline-v1"),
        mean_xyz_m=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
        covariance_diag_m2=((9.0e-6, 9.0e-6, 16.0e-6),) * 2,
    )
    candidate = ExampleCompleteBelief(
        artifact_id=_digest("belief-candidate-v1"),
        mean_xyz_m=((0.001, 0.0, 0.0), (0.101, 0.0, 0.0)),
        covariance_diag_m2=((6.0e-6, 6.0e-6, 12.0e-6),) * 2,
    )
    inference = ExampleInference(
        candidate_id=_digest("candidate-inference-v1"),
        inference_admissible=True,
    )
    common_domain_id = _digest("ecosystem-minimal-common-domain-v1")

    accepted = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=common_domain_id,
        certificate_id=_digest("accepted-certificate-v1"),
        inference_admissible=True,
        regret_guard_accepted=True,
        reason="synthetic-contract-smoke-accepted",
        metadata={"example": "ecosystem-minimal-v1"},
    )
    accepted_result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        accepted,
        metadata={"example_path": "accepted"},
    )
    if accepted_result.selected_belief is not candidate:
        raise RuntimeError("accepted decision did not preserve candidate identity")

    rejected = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=common_domain_id,
        certificate_id=_digest("fallback-certificate-v1"),
        inference_admissible=True,
        regret_guard_accepted=False,
        reason="synthetic-unsupported-group",
        metadata={"example": "ecosystem-minimal-v1"},
    )
    fallback_result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        rejected,
        metadata={"example_path": "fallback"},
    )
    if fallback_result.selected_belief is not baseline:
        raise RuntimeError("rejected decision did not return the exact fallback object")

    accepted_payload = {
        "guard_decision_id": accepted.decision_id,
        "selection_id": accepted_result.selection.selection_id,
        "selected_role": "candidate",
        "selected_artifact_id": accepted_result.selected_belief.artifact_id,
        "exact_candidate_identity": (accepted_result.selected_belief is candidate),
    }
    fallback_payload = {
        "guard_decision_id": rejected.decision_id,
        "selection_id": fallback_result.selection.selection_id,
        "selected_role": "baseline",
        "selected_artifact_id": fallback_result.selected_belief.artifact_id,
        "exact_fallback_identity": fallback_result.selected_belief is baseline,
    }
    _write_json(output_dir / "accepted_decision.json", accepted_payload)
    _write_json(output_dir / "fallback_decision.json", fallback_payload)
    summary = {
        "contract": "bayesian-phystwin.ecosystem-contract-smoke",
        "contract_version": 1,
        "scientific_scope": (
            "Synthetic contract smoke only; no real Prob4D provider value or "
            "Causal4D downstream benefit is claimed."
        ),
        "prob4d_observation": {
            "artifact_id": loaded.artifact_id,
            "source_repository": loaded.source_repository,
            "roundtrip_verified": True,
        },
        "bayesian_phystwin_routing": {
            "accepted_exact_candidate_identity": (
                accepted_result.selected_belief is candidate
            ),
            "fallback_exact_baseline_identity": (
                fallback_result.selected_belief is baseline
            ),
        },
        "causal4d_provider_manifest": causal4d_provider_manifest(
            provider_revision="ecosystem-minimal-v1"
        ),
    }
    _write_json(output_dir / "ecosystem_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ecosystem-minimal-v1"),
    )
    return parser.parse_args()


def main() -> int:
    summary = run_example(parse_args().output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
