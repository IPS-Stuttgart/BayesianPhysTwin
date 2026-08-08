from __future__ import annotations

from collections.abc import Mapping

import pytest

from bayesian_phystwin._canonical_contracts import (
    frozen_finite_json_mapping,
    plain_json,
)


class _StringSubclass(str):
    pass


@pytest.mark.parametrize("value", [[], (), "", 0, False])
def test_frozen_json_mapping_rejects_falsey_non_mappings(value: object) -> None:
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        frozen_finite_json_mapping(value)  # type: ignore[arg-type]


def test_frozen_json_mapping_preserves_none_as_empty_mapping() -> None:
    frozen = frozen_finite_json_mapping(None)

    assert isinstance(frozen, Mapping)
    assert dict(frozen) == {}


@pytest.mark.parametrize("key", [1, False, b"key", _StringSubclass("key")])
def test_plain_json_rejects_nonliteral_string_mapping_keys(key: object) -> None:
    with pytest.raises(ValueError, match="literal strings"):
        plain_json({key: "value"})


def test_frozen_json_mapping_rejects_nested_nonliteral_string_keys() -> None:
    with pytest.raises(ValueError, match="literal string object keys"):
        frozen_finite_json_mapping(
            {
                "outer": [
                    {
                        1: "would otherwise collide with the string key",
                        "1": "different value",
                    }
                ]
            }
        )


def test_canonical_string_keys_and_nested_sequences_still_round_trip() -> None:
    frozen = frozen_finite_json_mapping(
        {
            "outer": [{"key": 1}, {"flag": True}],
            "tuple": ("a", "b"),
        }
    )

    assert plain_json(frozen) == {
        "outer": [{"key": 1}, {"flag": True}],
        "tuple": ["a", "b"],
    }
