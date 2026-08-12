"""External-consumer type-check fixture for the installed public APIs."""

from dataclasses import dataclass
from pathlib import Path
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin import GaugeAwareBeliefConfig
from bayesian_phystwin.causal4d_provider_v1 import causal4d_provider_manifest
from bayesian_phystwin.inference.v1 import (
    ClaimBearingProb4DCandidateV1,
    CompleteBeliefGuardDecisionV1,
    GuardedUpdateResultV1,
    PhysicalLinearizationV1,
    finalize_guarded_update,
    infer_prob4d_candidate,
)
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.v1 import ObservationBeliefV1, load_observation_belief


@dataclass(frozen=True)
class ExternalBelief:
    artifact_id: str


def load_validated_observation(path: Path) -> ObservationBeliefV1:
    """Exercise the stable observation contract from an installed wheel."""

    return load_observation_belief(path)


def infer_candidate(
    observation: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    prediction: NDArray[np.float64],
) -> ClaimBearingProb4DCandidateV1:
    """Exercise the stable strict-candidate entry point."""

    return infer_prob4d_candidate(
        observation,
        linearization,
        physical_prediction_xyz_m=prediction,
    )


def validate_prob4d_observation(
    observation: ObservationBeliefV1,
) -> dict[str, object]:
    """Exercise the public Prob4D compatibility boundary."""

    return validate_prob4d_causal_observation_belief(observation)


def provider_manifest() -> dict[str, object]:
    """Exercise the public Causal4D compatibility boundary."""

    return causal4d_provider_manifest(provider_revision="0" * 40)


def finalize_candidate(
    inference: ClaimBearingProb4DCandidateV1,
    baseline: ExternalBelief,
    candidate: ExternalBelief,
    decision: CompleteBeliefGuardDecisionV1,
) -> GuardedUpdateResultV1[ExternalBelief]:
    """Exercise generic complete-belief routing from an installed wheel."""

    return finalize_guarded_update(
        inference,
        baseline,
        candidate,
        decision,
    )


config = GaugeAwareBeliefConfig()
assert_type(config, GaugeAwareBeliefConfig)
assert_type(config.maximum_iterations, int)
assert_type(load_validated_observation(Path("observation.npz")), ObservationBeliefV1)
assert_type(provider_manifest(), dict[str, object])
