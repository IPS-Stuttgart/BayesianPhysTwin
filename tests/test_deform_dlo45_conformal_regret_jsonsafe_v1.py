from __future__ import annotations

import json

import pytest

from experiments.deform_dlo45_decision_identifiability_v1.support_envelope_jsonsafe import (
    INFINITE_RADIUS_TOKEN,
    canonical_sha256,
    json_safe,
    radius_from_record,
)


def _source(radius: object, has_finite_radius: bool) -> dict[str, object]:
    return {
        "dlos": {
            "DLO4": {
                "envelopes": {
                    "0.100000": {
                        "base_certificate_selected_action": {
                            "radius": radius,
                            "has_finite_radius": has_finite_radius,
                        }
                    }
                }
            }
        }
    }


def test_positive_infinity_is_explicit_and_strict_json_safe() -> None:
    value = {
        "radius": float("inf"),
        "nested": [0.1, {"other": float("inf")}],
    }
    converted = json_safe(value)
    assert converted == {
        "radius": INFINITE_RADIUS_TOKEN,
        "nested": [0.1, {"other": INFINITE_RADIUS_TOKEN}],
    }
    json.dumps(converted, allow_nan=False)
    assert canonical_sha256(value) == canonical_sha256(converted)


def test_explicit_infinite_radius_round_trips_to_fail_closed_value() -> None:
    radius = radius_from_record(
        _source(INFINITE_RADIUS_TOKEN, False),
        dlo="DLO4",
        grouping="per_dlo",
        alpha_key="0.100000",
        envelope_kind="base_certificate_selected_action",
    )
    assert radius == float("inf")


def test_finite_radius_round_trips_without_change() -> None:
    radius = radius_from_record(
        _source(0.42, True),
        dlo="DLO4",
        grouping="per_dlo",
        alpha_key="0.100000",
        envelope_kind="base_certificate_selected_action",
    )
    assert radius == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("radius", "has_finite"),
    [
        (INFINITE_RADIUS_TOKEN, True),
        (None, False),
        (-1.0, True),
        (float("inf"), True),
        (0.1, False),
    ],
)
def test_inconsistent_radius_records_are_rejected(
    radius: object, has_finite: bool
) -> None:
    with pytest.raises(ValueError):
        radius_from_record(
            _source(radius, has_finite),
            dlo="DLO4",
            grouping="per_dlo",
            alpha_key="0.100000",
            envelope_kind="base_certificate_selected_action",
        )


def test_nan_is_never_silently_serialized() -> None:
    with pytest.raises(ValueError, match="NaN"):
        json_safe({"radius": float("nan")})
