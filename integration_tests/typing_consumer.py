"""External-consumer type-check fixture for the installed public APIs."""

from pathlib import Path
from typing import assert_type

from bayesian_phystwin import GaugeAwareBeliefConfig
from bayesian_phystwin.v1 import ObservationBeliefV1, load_observation_belief


def load_validated_observation(path: Path) -> ObservationBeliefV1:
    """Exercise the stable observation contract from an installed wheel."""

    return load_observation_belief(path)


config = GaugeAwareBeliefConfig()
assert_type(config, GaugeAwareBeliefConfig)
assert_type(config.maximum_iterations, int)
assert_type(load_validated_observation(Path("observation.npz")), ObservationBeliefV1)
