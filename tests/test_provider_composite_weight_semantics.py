from types import SimpleNamespace
from typing import cast

import pytest

from bayesian_phystwin.gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    _observation_composite_weight_mode,
)


def _belief(*, repository: str, metadata: dict[str, object]) -> ObservationBeliefV1:
    return cast(
        ObservationBeliefV1,
        SimpleNamespace(source_repository=repository, metadata=metadata),
    )


def test_explicit_provider_final_semantics_take_precedence() -> None:
    mode, source = _observation_composite_weight_mode(
        _belief(
            repository="another/provider",
            metadata={
                "group_composite_weight_semantics": (
                    "final-per-row-effective-sample-cap-v1"
                )
            },
        )
    )

    assert mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    assert source == "artifact-metadata"


def test_legacy_prob4d_effective_sample_metadata_is_provider_final() -> None:
    mode, source = _observation_composite_weight_mode(
        _belief(
            repository="FlorianPfaff/Prob4D",
            metadata={"effective_samples_per_group": 64.0},
        )
    )

    assert mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    assert source == "legacy-prob4d-export-metadata"


def test_unknown_prob4d_semantics_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported Prob4D"):
        _observation_composite_weight_mode(
            _belief(
                repository="FlorianPfaff/Prob4D",
                metadata={"group_composite_weight_semantics": "unknown-v99"},
            )
        )


def test_other_providers_retain_consumer_side_cap_by_default() -> None:
    mode, source = _observation_composite_weight_mode(
        _belief(repository="another/provider", metadata={})
    )

    assert mode == COMPOSITE_WEIGHT_MODE_CONSUMER_CAP
    assert source == "consumer-default"
