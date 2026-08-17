from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.cli.run_manifest import (
    _load_json_mapping,
    _load_repository_states,
)
from bayesian_phystwin.repository_provenance import default_runtime_environment


class _IntegerSubclass(int):
    pass


class _StringableName:
    def __str__(self) -> str:
        return "CUDA_VISIBLE_DEVICES"


def test_runtime_overrides_require_a_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        default_runtime_environment(overrides=cast(Any, []))


def test_runtime_overrides_reject_nonstring_keys_at_every_depth() -> None:
    with pytest.raises(ValueError, match="genuine string keys"):
        default_runtime_environment(overrides=cast(Any, {1: "value"}))
    with pytest.raises(ValueError, match="genuine string keys"):
        default_runtime_environment(overrides=cast(Any, {"gpu": {1: "value"}}))


def test_runtime_overrides_reject_scalar_subclasses_and_cycles() -> None:
    with pytest.raises(ValueError, match="non-JSON value"):
        default_runtime_environment(overrides={"count": _IntegerSubclass(1)})

    circular: dict[str, object] = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="circular mapping"):
        default_runtime_environment(overrides=circular)

    sequence: list[object] = []
    sequence.append(sequence)
    with pytest.raises(ValueError, match="circular sequence"):
        default_runtime_environment(overrides={"cycle": sequence})


def test_runtime_overrides_cannot_replace_inferred_fields() -> None:
    for field in (
        "python_version",
        "operating_system",
        "selected_environment",
    ):
        with pytest.raises(ValueError, match="cannot replace inferred fields"):
            default_runtime_environment(overrides={field: "forged"})


def test_runtime_overrides_retain_strict_additional_json() -> None:
    runtime = default_runtime_environment(
        overrides={
            "accelerator": {
                "model": "test-gpu",
                "versions": (1, 2, 3),
                "deterministic": True,
            }
        }
    )

    assert runtime["accelerator"] == {
        "deterministic": True,
        "model": "test-gpu",
        "versions": [1, 2, 3],
    }
    assert runtime["python_version"] != "forged"


@pytest.mark.parametrize(
    "name",
    (
        "",
        " CUDA_VISIBLE_DEVICES",
        "CUDA VISIBLE DEVICES",
        "1CUDA",
        "CUDA=0",
        cast(Any, _StringableName()),
    ),
)
def test_environment_variable_names_require_canonical_identifiers(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="canonical identifiers"):
        default_runtime_environment(environment_variables=(name,))


@pytest.mark.parametrize(
    "names",
    (
        cast(Any, "CUDA_VISIBLE_DEVICES"),
        cast(Any, b"CUDA_VISIBLE_DEVICES"),
    ),
)
def test_environment_variable_collection_must_be_a_sequence_not_a_scalar(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="sequence of identifiers"):
        default_runtime_environment(environment_variables=names)


def test_selected_environment_is_explicit_sorted_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BPT_TEST_ALPHA", "a")
    monkeypatch.setenv("BPT_TEST_BETA", "b")

    runtime = default_runtime_environment(
        environment_variables=(
            "BPT_TEST_BETA",
            "BPT_TEST_ALPHA",
            "BPT_TEST_BETA",
            "BPT_TEST_MISSING",
        )
    )

    assert runtime["selected_environment"] == {
        "BPT_TEST_ALPHA": "a",
        "BPT_TEST_BETA": "b",
    }


def test_cli_json_loader_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"gpu": 1, "gpu": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _load_json_mapping(duplicate, name="runtime")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"temperature": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _load_json_mapping(nonfinite, name="runtime")


def _write_related(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps([record]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("repository", 1, "name must be a genuine string"),
        ("revision", 1, "revision must be a genuine string"),
        ("role", 1, "role is unsupported"),
        ("dirty", 0, "dirty field must be boolean"),
    ),
)
def test_related_repository_loader_never_coerces_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    record: dict[str, object] = {
        "repository": "owner/repository",
        "revision": "a" * 40,
        "dirty": False,
        "role": "upstream",
    }
    record[field] = value

    with pytest.raises(ValueError, match=message):
        _load_repository_states(_write_related(tmp_path / "related.json", record))


def test_related_repository_loader_retains_exact_valid_record(tmp_path: Path) -> None:
    record: dict[str, object] = {
        "repository": "owner/repository",
        "revision": "a" * 40,
        "dirty": False,
        "role": "observation",
    }

    states = _load_repository_states(_write_related(tmp_path / "related.json", record))

    assert len(states) == 1
    assert states[0].as_dict() == record
