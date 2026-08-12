"""Fail-closed validation helpers for supported integration boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def lowercase_sha256(value: object, *, name: str) -> str:
    """Return one exact lowercase SHA-256 digest without coercion."""

    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def optional_instance(
    value: object,
    expected_type: type[T],
    *,
    name: str,
) -> T | None:
    """Accept ``None`` or one value of the exact supported runtime family."""

    if value is None:
        return None
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__} or None")
    return value


def instance_or_default(
    value: object,
    expected_type: type[T],
    default_factory: Callable[[], T],
    *,
    name: str,
) -> T:
    """Resolve an omitted value without treating arbitrary falsey objects as absent."""

    if value is None:
        resolved = default_factory()
        if not isinstance(resolved, expected_type):
            raise TypeError(
                f"{name} default factory must return a {expected_type.__name__}"
            )
        return resolved
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__} or None")
    return value


__all__ = [
    "instance_or_default",
    "lowercase_sha256",
    "optional_instance",
]
