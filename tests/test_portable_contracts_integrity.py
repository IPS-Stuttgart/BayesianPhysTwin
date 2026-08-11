from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._canonical_contracts import (
    FrozenDict,
    FrozenList,
    frozen_finite_json_mapping,
    plain_json,
)
from bayesian_phystwin.physical_linearization import evaluate_nonlinear_closure


def _metadata() -> Mapping[str, Any]:
    return frozen_finite_json_mapping(
        {
            "nested": {
                "items": [1, {"accepted": True}],
            },
            "value": 2,
        }
    )


def _content_id(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plain_json(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_frozen_mapping_base_mutation_fails_closed() -> None:
    metadata = _metadata()
    before = _content_id(metadata)

    dict.__setitem__(metadata, "tampered", True)

    for operation in (
        lambda: metadata["value"],
        lambda: len(metadata),
        lambda: "value" in metadata,
        lambda: reversed(metadata),
        lambda: iter(metadata),
        lambda: metadata.items(),
        lambda: metadata.keys(),
        lambda: metadata.values(),
        lambda: metadata.get("value"),
        lambda: metadata.copy(),
        lambda: dict(metadata),
        lambda: metadata | {"other": 3},
        lambda: {"other": 3} | metadata,
        lambda: repr(metadata),
        lambda: metadata == {},
        lambda: metadata != {},
        lambda: plain_json(metadata),
        lambda: _content_id(metadata),
        lambda: json.dumps(metadata),
        lambda: json.dumps({"metadata": metadata}),
    ):
        with pytest.raises(RuntimeError, match="backing storage was mutated"):
            operation()
    assert before


def test_frozen_mapping_replacement_and_order_tampering_fail_closed() -> None:
    replaced = _metadata()
    dict.__setitem__(replaced, "value", int("2000"))
    with pytest.raises(RuntimeError, match="backing storage was mutated"):
        plain_json(replaced)

    reordered = _metadata()
    original = dict.__getitem__(reordered, "nested")
    dict.__delitem__(reordered, "nested")
    dict.__setitem__(reordered, "nested", original)
    with pytest.raises(RuntimeError, match="backing storage was mutated"):
        plain_json(reordered)


def test_frozen_nested_sequence_base_mutation_fails_closed() -> None:
    metadata = _metadata()
    items = metadata["nested"]["items"]

    list.append(items, "tampered")

    for operation in (
        lambda: iter(items),
        lambda: len(items),
        lambda: 1 in items,
        lambda: reversed(items),
        lambda: items[0],
        lambda: items.copy(),
        lambda: items.count(1),
        lambda: items.index(1),
        lambda: items + [3],
        lambda: [0] + items,
        lambda: items * 2,
        lambda: 2 * items,
        lambda: items < [2],
        lambda: items <= [2],
        lambda: items > [0],
        lambda: items >= [0],
        lambda: repr(items),
        lambda: items == [],
        lambda: items != [],
        lambda: plain_json(metadata),
        lambda: json.dumps(metadata),
        lambda: json.dumps({"metadata": metadata}),
    ):
        with pytest.raises(RuntimeError, match="backing storage was mutated"):
            operation()


def test_frozen_sequence_replacement_tampering_fails_closed() -> None:
    metadata = _metadata()
    items = metadata["nested"]["items"]
    list.__setitem__(items, 0, int("1000"))

    with pytest.raises(RuntimeError, match="backing storage was mutated"):
        plain_json(metadata)


def test_frozen_metadata_rejects_normal_nested_mutation() -> None:
    metadata = _metadata()
    nested = metadata["nested"]
    items = nested["items"]

    with pytest.raises(TypeError, match="metadata is immutable"):
        metadata["new"] = 1
    with pytest.raises(TypeError, match="metadata is immutable"):
        del metadata["value"]
    with pytest.raises(TypeError, match="metadata is immutable"):
        metadata.update({"new": 1})
    with pytest.raises(TypeError, match="metadata is immutable"):
        metadata |= {"new": 1}
    with pytest.raises(TypeError, match="metadata is immutable"):
        items.append(3)
    with pytest.raises(TypeError, match="metadata is immutable"):
        items[0] = 3
    with pytest.raises(TypeError, match="metadata is immutable"):
        del items[0]
    with pytest.raises(TypeError, match="metadata is immutable"):
        items += [3]
    with pytest.raises(TypeError, match="metadata is immutable"):
        items *= 2


def test_frozen_metadata_preserves_dict_list_and_json_compatibility() -> None:
    expected = {
        "nested": {
            "items": [1, {"accepted": True}],
        },
        "value": 2,
    }
    metadata = _metadata()
    items = metadata["nested"]["items"]

    assert isinstance(metadata, FrozenDict)
    assert isinstance(metadata, dict)
    assert isinstance(items, FrozenList)
    assert isinstance(items, list)
    assert metadata == expected
    assert expected == metadata
    assert items == expected["nested"]["items"]
    assert repr(items) == repr(expected["nested"]["items"])
    assert list(metadata) == ["nested", "value"]
    assert list(reversed(metadata)) == ["value", "nested"]
    assert "nested" in metadata
    assert "missing" not in metadata
    assert len(metadata) == 2
    assert metadata.get("missing", 3) == 3
    assert list(metadata.keys()) == ["nested", "value"]
    assert list(metadata.values())[1] == 2
    assert dict(metadata.items()) == expected
    assert metadata.copy() == expected
    assert metadata | {"extra": 3} == {**expected, "extra": 3}
    assert {"extra": 3} | metadata == {"extra": 3, **expected}
    assert len(items) == 2
    assert 1 in items
    assert list(reversed(items)) == list(reversed(expected["nested"]["items"]))
    assert items[:] == expected["nested"]["items"]
    assert items.copy() == expected["nested"]["items"]
    assert items.count(1) == 1
    assert items.index(1) == 0
    assert items.index(1, 0, 1) == 0
    assert items + [3] == [1, {"accepted": True}, 3]
    assert [0] + items == [0, 1, {"accepted": True}]
    assert items * 2 == expected["nested"]["items"] * 2
    assert 2 * items == expected["nested"]["items"] * 2
    assert items < [2]
    assert items <= expected["nested"]["items"]
    assert items > [0]
    assert items >= expected["nested"]["items"]
    with pytest.raises(TypeError):
        _ = metadata | []
    with pytest.raises(TypeError):
        _ = [] | metadata
    with pytest.raises(TypeError):
        _ = items + (3,)
    with pytest.raises(TypeError):
        _ = (0,) + items
    with pytest.raises(TypeError):
        _ = items * 1.5
    with pytest.raises(TypeError):
        _ = 1.5 * items
    with pytest.raises(TypeError):
        _ = items < (2,)
    assert json.loads(json.dumps(metadata)) == expected
    assert json.loads(json.dumps(plain_json(metadata))) == expected


def test_frozen_metadata_copy_and_plain_json_are_detached() -> None:
    metadata = _metadata()

    for detached in (
        copy.copy(metadata),
        copy.deepcopy(metadata),
        plain_json(metadata),
    ):
        detached["nested"]["items"].append("copy-only")
        assert "copy-only" not in metadata["nested"]["items"]


def test_nonlinear_closure_rejects_empty_query_evidence() -> None:
    empty = np.empty((0, 3), dtype=np.float64)

    with pytest.raises(ValueError, match=r"Q >= 1"):
        evaluate_nonlinear_closure(
            "0" * 64,
            baseline_query_m=empty,
            linearized_query_m=empty,
            nonlinear_query_m=empty,
            absolute_tolerance_m=0.0,
            relative_tolerance=0.0,
        )


def test_nonlinear_closure_still_accepts_nonempty_exact_replay() -> None:
    query = np.zeros((1, 3), dtype=np.float64)

    closure = evaluate_nonlinear_closure(
        "0" * 64,
        baseline_query_m=query,
        linearized_query_m=query,
        nonlinear_query_m=query,
        absolute_tolerance_m=0.0,
        relative_tolerance=0.0,
    )

    assert closure.candidate_valid
    assert closure.absolute_error_m == 0.0
    assert closure.relative_error == 0.0
