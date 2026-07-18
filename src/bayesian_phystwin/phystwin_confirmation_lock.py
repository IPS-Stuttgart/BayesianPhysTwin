"""Exclusive ownership for long-running PhysTwin confirmation outputs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import fcntl
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")
_PathArgument = str | Path


@contextmanager
def confirmation_output_lock(output_dir: _PathArgument) -> Iterator[None]:
    """Own one confirmation output tree until the complete run has finished.

    The lock deliberately remains open in the parent while any process pool is
    alive. Forked workers may inherit that open-file description, but they never
    acquire or release it; executor shutdown therefore precedes lock release.
    If the parent dies while workers remain, their inherited descriptors keep a
    competing invocation out until those writers have also exited.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".phystwin_confirmation.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another PhysTwin confirmation owns {output.resolve()}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def exclusively_owned_confirmation_output(
    function: Callable[
        Concatenate[_PathArgument, _PathArgument, _P],
        _R,
    ],
) -> Callable[Concatenate[_PathArgument, _PathArgument, _P], _R]:
    """Hold the shared confirmation lock for a complete runner invocation."""

    @wraps(function)
    def locked(
        data_root: _PathArgument,
        output_dir: _PathArgument,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with confirmation_output_lock(output_dir):
            return function(data_root, output_dir, *args, **kwargs)

    return locked
