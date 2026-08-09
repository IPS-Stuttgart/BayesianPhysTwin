from __future__ import annotations

from typing import Any

import pytest

from bayesian_phystwin.observation_timing_interchange import (
    OBSERVATION_TIME_CORRECTION_CONVENTION,
    observation_timing_prior_from_payload,
    observation_timing_prior_payload,
)
from bayesian_phystwin.observation_timing_nuisance import ObservationTimingPrior


def _payload(**overrides: Any) -> dict[str, object]:
    value: dict[str, object] = {
        "clock_domain": "camera-hardware-clock",
        "mean_offset_s": -0.010,
        "standard_deviation_s": 0.0015,
        "source_artifact_id": "a" * 64,
        "offset_convention": OBSERVATION_TIME_CORRECTION_CONVENTION,
    }
    value.update(overrides)
    return value


def test_source_only_causal4d_style_payload_is_admitted() -> None:
    prior = observation_timing_prior_from_payload(_payload())

    assert prior == ObservationTimingPrior(
        clock_domain="camera-hardware-clock",
        mean_offset_s=-0.010,
        standard_deviation_s=0.0015,
        source_artifact_id="a" * 64,
    )


def test_round_trip_preserves_exact_sign_convention() -> None:
    prior = ObservationTimingPrior(
        clock_domain="camera-hardware-clock",
        mean_offset_s=-0.010,
        standard_deviation_s=0.0015,
        source_artifact_id="a" * 64,
    )

    payload = observation_timing_prior_payload(prior)
    assert payload["offset_convention"] == (
        "aligned_observation_time_s = observation_time_s + offset_s"
    )
    assert observation_timing_prior_from_payload(payload) == prior


def test_changed_sign_convention_fails_closed() -> None:
    with pytest.raises(ValueError, match="convention changed"):
        observation_timing_prior_from_payload(
            _payload(
                offset_convention=(
                    "observation_time_s = aligned_observation_time_s + offset_s"
                )
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in _payload().items() if key != "clock_domain"},
        {**_payload(), "extra": "not-admitted"},
    ],
)
def test_payload_schema_is_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="fields changed"):
        observation_timing_prior_from_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_offset_s", True, "must be finite"),
        ("standard_deviation_s", 0.0, "must be positive"),
        ("standard_deviation_s", float("nan"), "must be finite"),
        ("clock_domain", " camera ", "surrounding whitespace"),
        ("source_artifact_id", "", "must be nonempty"),
    ],
)
def test_invalid_scalar_payload_values_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        observation_timing_prior_from_payload(_payload(**{field: value}))
