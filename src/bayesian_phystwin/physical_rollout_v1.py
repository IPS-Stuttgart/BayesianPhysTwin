"""Simulator-neutral physical rollout arrays used by Bayesian-PhysTwin.

The contract intentionally contains only material-query trajectories and the
minimal controls needed by downstream belief updates.  Simulator-native state
belongs in a producer artifact, not in this portable interface.
"""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

PHYSICAL_ROLLOUT_ARRAY_NAMES: Final = frozenset(
    {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
    }
)

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def validate_physical_rollout_arrays(
    values: Mapping[str, npt.NDArray[Any]],
    *,
    expected_frame_count: int | None = None,
) -> dict[str, FloatArray]:
    """Validate and copy one simulator-neutral physical rollout.

    Positions are expressed in metres in a producer-declared world frame.
    Point index ``n`` must denote the same material query in every array and
    frame.  This function validates array semantics; the producer manifest
    binds the actual coordinate frame and simulator provenance.
    """

    _require(
        set(values) == set(PHYSICAL_ROLLOUT_ARRAY_NAMES),
        "physical rollout array roster changed",
    )
    arrays = {
        name: np.ascontiguousarray(np.asarray(values[name])).copy()
        for name in PHYSICAL_ROLLOUT_ARRAY_NAMES
    }
    prediction = arrays["prediction_m"]
    persistence = arrays["persistence_m"]
    driven = arrays["driven_readout_m"]
    zero = arrays["zero_action_readout_m"]
    support = arrays["action_support"]
    frame_zero = arrays["frame_zero_points_m"]

    _require(
        prediction.ndim == 3
        and prediction.shape[0] >= 2
        and prediction.shape[1] >= 1
        and prediction.shape[2] == 3,
        "physical prediction must have shape (T,N,3)",
    )
    if expected_frame_count is not None:
        _require(
            isinstance(expected_frame_count, int)
            and not isinstance(expected_frame_count, bool)
            and expected_frame_count >= 2,
            "expected_frame_count must be an integer >= 2",
        )
        _require(
            prediction.shape[0] == expected_frame_count,
            "physical rollout frame count changed",
        )
    _require(
        persistence.shape == prediction.shape
        and driven.shape == prediction.shape
        and zero.shape == prediction.shape
        and frame_zero.shape == prediction.shape[1:]
        and support.shape == (prediction.shape[1],),
        "physical rollout array shapes changed",
    )
    _require(
        all(np.issubdtype(value.dtype, np.floating) for value in arrays.values()),
        "physical rollout arrays must be floating point",
    )
    _require(
        len({value.dtype.str for value in arrays.values()}) == 1,
        "physical rollout dtypes differ",
    )
    _require(
        all(np.all(np.isfinite(value)) for value in arrays.values()),
        "physical rollout contains non-finite values",
    )
    _require(
        np.all((support >= 0.0) & (support <= 1.0)),
        "physical rollout action support is outside [0,1]",
    )
    _require(
        np.array_equal(
            persistence,
            np.repeat(frame_zero[None], prediction.shape[0], axis=0),
        ),
        "physical rollout persistence is not exact",
    )
    _require(
        np.array_equal(prediction[0], frame_zero)
        and np.array_equal(driven[0], frame_zero)
        and np.array_equal(zero[0], frame_zero),
        "physical rollout changed frame-zero material identity",
    )
    return arrays


def load_physical_rollout_archive(
    path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> dict[str, FloatArray]:
    """Load and validate a no-pickle physical rollout archive."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("physical rollout must be an ordinary file")
    try:
        with np.load(source, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load physical rollout archive") from error
    return validate_physical_rollout_arrays(
        arrays,
        expected_frame_count=expected_frame_count,
    )


def _npy_bytes(value: npt.NDArray[Any]) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(  # type: ignore[no-untyped-call]
        output,
        np.ascontiguousarray(value),
        version=(2, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def write_deterministic_npz(
    path: str | Path,
    arrays: Mapping[str, npt.NDArray[Any]],
) -> Path:
    """Write sorted, timestamp-independent uncompressed NPY members once."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb"
    with target.open(mode) as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, value in sorted(arrays.items()):
                info = zipfile.ZipInfo(
                    f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(np.asarray(value)))
        raw.flush()
        os.fsync(raw.fileno())
    return target
