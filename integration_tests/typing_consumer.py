"""External-consumer type-check fixture for the installed public API."""

from pathlib import Path
from typing import assert_type

from bayesian_phystwin import (
    GaugeAwareBeliefConfig,
    ObservationBeliefV1,
    load_observation_belief,
)


def load_validated_observation(path: Path) -> ObservationBeliefV1:
    """Exercise the public observation contract without importing private modules."""

    return load_observation_belief(path)


config = GaugeAwareBeliefConfig()
assert_type(config, GaugeAwareBeliefConfig)
assert_type(config.maximum_iterations, int)
assert_type(load_validated_observation(Path("observation.npz")), ObservationBeliefV1)
