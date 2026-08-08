"""Shared canonicalization helpers for content-addressed public contracts."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np


class FrozenDict(dict):
    """Dict-compatible recursively immutable JSON mapping."""

    __slots__ = ()
    _MUTATORS = frozenset({"clear", "pop", "popitem", "setdefault", "update"})

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: object, value: object) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> None:
        self._immutable(key)

    def __ior__(self, other: object) -> FrozenDict:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __copy__(self) -> dict[str, Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return plain_json(self)


class FrozenList(list):
    """List-compatible recursively immutable JSON sequence."""

    __slots__ = ()
    _MUTATORS = frozenset(
        {"append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort"}
    )

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("metadata is immutable")

    def __setitem__(self, key: object, value: object) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> None:
        self._immutable(key)

    def __iadd__(self, other: object) -> FrozenList:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __imul__(self, other: object) -> FrozenList:
        self._immutable(other)
        return self

    def __copy__(self) -> list[Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return plain_json(self)


def plain_json(value: Any) -> Any:
    """Return ordinary JSON-compatible containers from frozen containers.

    JSON object keys are part of content-addressed contract identities. Reject
    non-literal string keys instead of coercing them with ``str()`` because
    coercion can collapse distinct Python mappings onto the same JSON object.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be literal strings")
            result[key] = plain_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze_json(item) for item in value)
    return value


def frozen_finite_json_mapping(
    values: Mapping[str, Any] | None,
    *,
    name: str = "metadata",
) -> Mapping[str, Any]:
    """Copy, canonicalize, validate, and recursively freeze a JSON mapping."""

    if values is None:
        source: Mapping[str, Any] = {}
    elif not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    else:
        source = values
    try:
        plain = plain_json(source)
    except ValueError as error:
        raise ValueError(f"{name} must use literal string object keys") from error
    try:
        normalized = json.loads(
            json.dumps(
                plain,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    return _freeze_json(normalized)


def literal_lower_hex(
    value: object,
    *,
    name: str,
    lengths: Collection[int],
) -> str:
    """Require a literal lowercase hexadecimal string of an allowed length.

    Content-addressed public contracts must not accept integers, bytes, or custom
    string subclasses merely because ``str(value)`` happens to resemble a digest.
    """

    allowed_lengths = frozenset(lengths)
    if not allowed_lengths or any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in allowed_lengths
    ):
        raise ValueError("lengths must contain positive integers")
    if type(value) is not str:
        raise ValueError(f"{name} must be a literal string")
    if len(value) not in allowed_lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        expected = ", ".join(str(length) for length in sorted(allowed_lengths))
        raise ValueError(
            f"{name} must be lowercase hexadecimal with length in {{{expected}}}"
        )
    return value


def canonical_relative_posix_path(value: object, *, name: str) -> str:
    """Require one portable, canonical, repository-relative POSIX path."""

    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
        raise ValueError(f"{name} must be a canonical relative POSIX path")

    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    canonical = path.as_posix()
    if canonical != value or canonical in {"", "."}:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return canonical


def genuine_boolean(value: object, *, name: str) -> bool:
    """Require a real Python or NumPy boolean and return a Python bool."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def genuine_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Require a real Python or NumPy integer without lossy coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        qualifier = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{qualifier}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def integer_array(values: object, *, name: str) -> np.ndarray:
    """Require an integer-typed array and return an owned canonical int64 copy."""

    raw = np.asarray(values)
    integer_dtype = np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(
        raw.dtype, np.bool_
    )
    if not integer_dtype:
        raise ValueError(f"{name} must contain integers")
    if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(
        raw > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} contains integers outside int64 range")
    return np.array(raw, dtype=np.int64, copy=True, order="C")


def canonical_string_tuple(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Copy and validate a descriptor sequence as an immutable string tuple."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


__all__ = [
    "FrozenDict",
    "FrozenList",
    "canonical_relative_posix_path",
    "canonical_string_tuple",
    "frozen_finite_json_mapping",
    "genuine_boolean",
    "genuine_integer",
    "integer_array",
    "literal_lower_hex",
    "plain_json",
]
