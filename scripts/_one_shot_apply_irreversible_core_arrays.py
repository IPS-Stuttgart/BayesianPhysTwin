#!/usr/bin/env python3
"""Apply irreversible byte-backed ownership to core claim-bearing arrays."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, text: str) -> None:
    (ROOT / relative).write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _append_once(text: str, marker: str, addition: str, *, label: str) -> str:
    if marker in text:
        raise SystemExit(f"{label}: patch marker already exists")
    if not text.endswith("\n"):
        text += "\n"
    return text + "\n" + addition.strip() + "\n"


def _patch_canonical_contracts() -> None:
    path = "src/bayesian_phystwin/_canonical_contracts.py"
    text = _read(path)
    if "def immutable_array(" in text:
        raise SystemExit("immutable_array already exists")
    helper = '''def immutable_array(
    values: object,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return a C-contiguous array backed by immutable ``bytes`` storage.

    ``array.setflags(write=False)`` is reversible for owning NumPy arrays. Public
    content-addressed contracts instead use a ``bytes`` owner so callers cannot
    re-enable write access and mutate a validated artifact in place.
    """

    array = np.array(values, dtype=dtype, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("contract arrays must not contain Python objects")
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)


def immutable_integer_array(values: object, *, name: str) -> np.ndarray:
    """Return a validated canonical int64 array with irreversible immutability."""

    return immutable_array(
        integer_array(values, name=name),
        dtype=np.dtype(np.int64),
    )


'''
    text = _replace_once(
        text,
        "def integer_array(values: object, *, name: str) -> np.ndarray:\n",
        helper + "def integer_array(values: object, *, name: str) -> np.ndarray:\n",
        label="canonical helper insertion",
    )
    text = _replace_once(
        text,
        '    "genuine_integer",\n    "integer_array",\n',
        '    "genuine_integer",\n    "immutable_array",\n    "immutable_integer_array",\n    "integer_array",\n',
        label="canonical __all__ insertion",
    )
    _write(path, text)


def _patch_observation_belief() -> None:
    path = "src/bayesian_phystwin/observation_belief.py"
    text = _read(path)
    text = _replace_once(
        text,
        "    genuine_integer,\n    integer_array,\n",
        "    genuine_integer,\n    immutable_array,\n    immutable_integer_array,\n    integer_array,\n",
        label="observation imports",
    )
    text = _replace_once(
        text,
        '''def _readonly_float(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _readonly_integer(values: object, *, name: str) -> np.ndarray:
    array = integer_array(values, name=name)
    array.setflags(write=False)
    return array
''',
        '''def _readonly_float(values: object) -> np.ndarray:
    return immutable_array(values, dtype=np.dtype(np.float64))


def _readonly_integer(values: object, *, name: str) -> np.ndarray:
    return immutable_integer_array(values, name=name)
''',
        label="observation readonly helpers",
    )
    _write(path, text)


def _patch_physical_linearization() -> None:
    path = "src/bayesian_phystwin/physical_linearization.py"
    text = _read(path)
    text = _replace_once(
        text,
        "    genuine_integer,\n    integer_array,\n",
        "    genuine_integer,\n    immutable_array,\n    immutable_integer_array,\n    integer_array,\n",
        label="linearization imports",
    )
    text = _replace_once(
        text,
        '''def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _readonly_integer(values: object, *, name: str) -> np.ndarray:
    result = integer_array(values, name=name)
    result.setflags(write=False)
    return result
''',
        '''def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    return immutable_array(values, dtype=dtype)


def _readonly_integer(values: object, *, name: str) -> np.ndarray:
    return immutable_integer_array(values, name=name)
''',
        label="linearization readonly helpers",
    )
    _write(path, text)


def _patch_tests() -> None:
    canonical_path = "tests/test_canonical_contracts.py"
    canonical = _read(canonical_path)
    canonical = _append_once(
        canonical,
        "test_immutable_array_uses_irreversible_bytes_backing",
        '''def test_immutable_array_uses_irreversible_bytes_backing() -> None:
    import numpy as np
    import pytest

    from bayesian_phystwin._canonical_contracts import (
        immutable_array,
        immutable_integer_array,
    )

    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    frozen = immutable_array(source, dtype=np.dtype(np.float64))
    source[0, 0] = 99.0

    assert frozen.flags.c_contiguous
    assert frozen.flags.writeable is False
    assert frozen.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError):
        frozen.setflags(write=True)

    integers = immutable_integer_array([1, 2, 3], name="identities")
    assert integers.dtype == np.dtype(np.int64)
    assert integers.flags.writeable is False
    with pytest.raises(ValueError):
        integers.setflags(write=True)

    with pytest.raises(TypeError, match="must not contain Python objects"):
        immutable_array(np.asarray([object()], dtype=object))
''',
        label="canonical immutable-array test",
    )
    _write(canonical_path, canonical)

    observation_path = "tests/test_observation_belief.py"
    observation = _read(observation_path)
    observation = _append_once(
        observation,
        "test_observation_payload_arrays_are_irreversibly_immutable",
        '''def test_observation_payload_arrays_are_irreversibly_immutable(
    tmp_path: Path,
) -> None:
    source = _belief()
    path = tmp_path / "immutable-observation.npz"
    save_observation_belief(path, source)

    for belief in (source, load_observation_belief(path)):
        artifact_id = belief.artifact_id
        for name, array in belief._arrays().items():
            assert array.flags.writeable is False, name
            with pytest.raises(ValueError):
                array.setflags(write=True)
        assert belief.artifact_id == artifact_id
''',
        label="observation irreversible test",
    )
    _write(observation_path, observation)

    linearization_path = "tests/test_prior_aware_and_linearization.py"
    linearization = _read(linearization_path)
    linearization = _append_once(
        linearization,
        "test_physical_linearization_arrays_are_irreversibly_immutable",
        '''def test_physical_linearization_arrays_are_irreversibly_immutable(
    tmp_path: Path,
) -> None:
    import numpy as np
    import pytest

    from bayesian_phystwin.physical_linearization import (
        PhysicalLinearizationV1,
        load_physical_linearization,
        save_physical_linearization,
    )

    source = PhysicalLinearizationV1(
        observation_artifact_id="a" * 64,
        baseline_belief_id="b" * 64,
        action_prefix_id="c" * 64,
        simulator_revision="d" * 40,
        frame_ids=np.asarray([0], dtype=np.int64),
        entity_ids=np.asarray([0], dtype=np.int64),
        view_indices=np.asarray([0], dtype=np.int64),
        window_indices=np.asarray([0], dtype=np.int64),
        state_jacobian=np.ones((1, 3, 1), dtype=np.float64),
        query_state_jacobian=np.ones((1, 3, 1), dtype=np.float64),
        physical_response_m=np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64),
    )
    path = tmp_path / "immutable-linearization.npz"
    save_physical_linearization(path, source)

    for linearization in (source, load_physical_linearization(path)):
        artifact_id = linearization.artifact_id
        for name, array in linearization.arrays().items():
            assert array.flags.writeable is False, name
            with pytest.raises(ValueError):
                array.setflags(write=True)
        assert linearization.artifact_id == artifact_id
''',
        label="linearization irreversible test",
    )
    _write(linearization_path, linearization)


def _patch_changelog() -> None:
    path = "CHANGELOG.md"
    text = _read(path)
    bullet = (
        "- Core content-addressed observation and physical-linearization arrays now "
        "use immutable bytes-backed NumPy storage, so callers cannot re-enable "
        "write access after validation.\n"
    )
    if bullet in text:
        raise SystemExit("changelog bullet already exists")
    text = _replace_once(
        text,
        "### Changed\n\n",
        "### Changed\n\n" + bullet,
        label="changelog insertion",
    )
    _write(path, text)


def main() -> int:
    _patch_canonical_contracts()
    _patch_observation_belief()
    _patch_physical_linearization()
    _patch_tests()
    _patch_changelog()

    workflow = ROOT / ".github/workflows/_one_shot_irreversible_core_arrays.yml"
    script = ROOT / "scripts/_one_shot_apply_irreversible_core_arrays.py"
    workflow.unlink()
    script.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
