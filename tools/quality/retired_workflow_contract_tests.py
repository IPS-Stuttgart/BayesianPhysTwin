"""Load preserved historical workflow tests against retired workflow bytes."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import ModuleType


def load_retired_contract_test(
    *,
    archived_test: Path,
    original_test: Path,
    replacements: Mapping[str, object],
) -> ModuleType:
    """Execute an archived test source with its original repository location."""

    source = archived_test.read_text(encoding="utf-8")
    identity = hashlib.sha256(archived_test.as_posix().encode()).hexdigest()[:16]
    module_name = f"_retired_workflow_contract_{identity}"
    module = ModuleType(module_name)
    module.__file__ = str(original_test)
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, str(original_test), "exec"), module.__dict__)
    for name, value in replacements.items():
        if name not in module.__dict__:
            raise RuntimeError(f"archived contract test has no global {name!r}")
        module.__dict__[name] = value
    return module


def expose_tests(
    target: MutableMapping[str, object],
    module: ModuleType,
    *,
    exclude: frozenset[str] | None = None,
) -> None:
    """Expose archived test functions to pytest without collecting the archive."""

    excluded = frozenset() if exclude is None else exclude
    for name, value in vars(module).items():
        if name.startswith("test_") and name not in excluded and callable(value):
            target[name] = value
