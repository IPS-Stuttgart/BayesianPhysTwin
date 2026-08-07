from __future__ import annotations

import importlib

import pytest

from bayesian_phystwin._canonical_contracts import (
    canonical_relative_posix_path,
    literal_lower_hex,
)


def _expose_stable_coverage_tests(module_name: str, prefix: str) -> None:
    module = importlib.import_module(module_name)
    for name in dir(module):
        if name.startswith("test_"):
            globals()[f"test_stable_core_{prefix}_{name[5:]}"] = getattr(module, name)


for _module_name, _prefix in (
    (
        "test_deform360_calibration_source_run_record",
        "deform360_run_record",
    ),
    (
        "test_deform360_calibration_source_run_record_validation",
        "deform360_run_record_validation",
    ),
    (
        "test_render_deform360_calibration_source_run_record",
        "deform360_run_record_rendering",
    ),
):
    _expose_stable_coverage_tests(_module_name, _prefix)


class _StringSubclass(str):
    pass


def test_literal_lower_hex_accepts_only_literal_lowercase_strings() -> None:
    assert literal_lower_hex("a" * 40, name="revision", lengths={40, 64}) == ("a" * 40)
    assert literal_lower_hex("1" * 64, name="digest", lengths={64}) == "1" * 64

    rejected = [
        int("1" * 40),
        b"1" * 40,
        _StringSubclass("1" * 40),
        "A" * 40,
        "1" * 39,
        "1" * 40 + " ",
    ]
    for value in rejected:
        with pytest.raises(ValueError):
            literal_lower_hex(value, name="revision", lengths={40, 64})


def test_literal_lower_hex_rejects_invalid_length_contract() -> None:
    for lengths in (set(), {0}, {-1}, {True}):
        with pytest.raises(ValueError, match="positive integers"):
            literal_lower_hex("a", name="digest", lengths=lengths)


def test_canonical_relative_posix_path_is_portable_and_non_normalizing() -> None:
    value = "raw/object-1/tactile.npy"
    assert canonical_relative_posix_path(value, name="artifact path") == value

    rejected = [
        "",
        b"raw/object",
        "/absolute/path",
        "//server/share",
        "C:/windows/path",
        "raw\\windows",
        "raw/../escape",
        "./raw/object",
        "raw/./object",
        "raw//object",
        "raw/object/",
        "raw/\x00object",
    ]
    for value in rejected:
        with pytest.raises(ValueError, match="POSIX|literal"):
            canonical_relative_posix_path(value, name="artifact path")
