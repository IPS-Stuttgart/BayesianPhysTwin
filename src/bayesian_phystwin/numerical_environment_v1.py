"""Content-addressed numerical runtime profiles for evidence-bearing runs.

The generic run manifest intentionally records portable runtime metadata. This
module adds an optional stricter profile for numerical evidence: the complete
installed distribution inventory, NumPy build configuration, known execution
controls, and an optional dependency-lock artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

NUMERICAL_ENVIRONMENT_SCHEMA = "bayesian_phystwin.numerical_environment"
NUMERICAL_ENVIRONMENT_SCHEMA_VERSION = 1
NUMERICAL_ENVIRONMENT_RUNTIME_KEY = "numerical_environment_v1"

_EXECUTION_CONTROL_NAMES = (
    "BLIS_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_CBWR",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "TF_DETERMINISTIC_OPS",
    "VECLIB_MAXIMUM_THREADS",
)
_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "schema_name",
        "schema_version",
        "python",
        "numpy",
        "scipy_version",
        "logical_cpu_count",
        "byte_order",
        "execution_controls",
        "installed_distributions",
        "installed_distributions_sha256",
        "dependency_lock",
    }
)
_PYTHON_FIELDS = frozenset({"implementation", "version", "compiler"})
_NUMPY_FIELDS = frozenset(
    {"version", "configuration_text", "configuration_sha256"}
)
_DISTRIBUTION_FIELDS = frozenset({"name", "version"})
_LOCK_FIELDS = frozenset({"name", "sha256", "size_bytes"})
_CANONICAL_DISTRIBUTION_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_DISTRIBUTIONS = 10_000
_MAX_RUNTIME_FRAGMENT_BYTES = 16 * 1024 * 1024


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unknown {unknown}")
    raise ValueError(f"{name} does not match schema: {', '.join(details)}")


def _require_text(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
    maximum_length: int = 4096,
) -> str:
    if type(value) is not str or value != value.strip():
        raise ValueError(f"{name} must be canonical text")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be nonempty")
    if len(value) > maximum_length:
        raise ValueError(f"{name} exceeds {maximum_length} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _require_multiline_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    if any(
        ord(character) < 32 and character not in "\r\n\t"
        for character in value
    ):
        raise ValueError(f"{name} contains a control character")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _normalize_numpy_configuration(value: str) -> str:
    text = _require_multiline_text(value, name="NumPy build configuration")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        lines = ["(no NumPy build configuration reported)"]
    result = "\n".join(lines) + "\n"
    if len(result.encode("utf-8")) > 1024 * 1024:
        raise ValueError("NumPy build configuration exceeds one MiB")
    return result


@dataclass(frozen=True, order=True)
class InstalledDistributionV1:
    """One exact installed distribution version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        name = _require_text(
            self.name,
            name="distribution name",
            maximum_length=256,
        )
        if _CANONICAL_DISTRIBUTION_NAME.fullmatch(name) is None:
            raise ValueError("distribution name must use canonical PEP-503 form")
        version = _require_text(
            self.version,
            name=f"{name} version",
            maximum_length=512,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class DependencyLockV1:
    """Content identity of the resolver input used for an evidence run."""

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        name = _require_text(
            self.name,
            name="dependency lock name",
            maximum_length=255,
        )
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("dependency lock name must be a portable basename")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "sha256",
            _require_sha256(self.sha256, name="dependency lock SHA-256"),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _require_integer(
                self.size_bytes,
                name="dependency lock size",
                minimum=0,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class NumericalEnvironmentV1:
    """Strict numerical environment identity for one evidence-producing run."""

    python_implementation: str
    python_version: str
    python_compiler: str
    numpy_version: str
    numpy_configuration_text: str
    scipy_version: str | None
    logical_cpu_count: int | None
    byte_order: str
    execution_controls: Mapping[str, str | None]
    installed_distributions: tuple[InstalledDistributionV1, ...]
    dependency_lock: DependencyLockV1 | None = None

    def __post_init__(self) -> None:
        implementation = _require_text(
            self.python_implementation,
            name="Python implementation",
        )
        python_version = _require_text(self.python_version, name="Python version")
        compiler = _require_text(
            self.python_compiler,
            name="Python compiler",
            allow_empty=True,
        )
        numpy_version = _require_text(self.numpy_version, name="NumPy version")
        scipy_version = self.scipy_version
        if scipy_version is not None:
            scipy_version = _require_text(scipy_version, name="SciPy version")
        cpu_count = self.logical_cpu_count
        if cpu_count is not None:
            cpu_count = _require_integer(
                cpu_count,
                name="logical CPU count",
                minimum=1,
            )
        if type(self.byte_order) is not str or self.byte_order not in {
            "little",
            "big",
        }:
            raise ValueError("byte order must be 'little' or 'big'")

        controls_mapping = _require_mapping(
            self.execution_controls,
            name="execution controls",
        )
        controls = dict(controls_mapping)
        if frozenset(controls) != frozenset(_EXECUTION_CONTROL_NAMES):
            raise ValueError(
                "execution controls must contain the complete registered set"
            )
        normalized_controls: dict[str, str | None] = {}
        for name in _EXECUTION_CONTROL_NAMES:
            value = controls[name]
            if value is not None:
                value = _require_text(value, name=f"execution control {name}")
                if len(value) > 256:
                    raise ValueError(f"execution control {name} is too long")
            normalized_controls[name] = cast(str | None, value)

        distributions = tuple(self.installed_distributions)
        if not distributions:
            raise ValueError("installed distributions must be nonempty")
        if len(distributions) > _MAX_DISTRIBUTIONS:
            raise ValueError("installed distribution inventory is too large")
        if any(type(item) is not InstalledDistributionV1 for item in distributions):
            raise ValueError(
                "installed distributions must contain InstalledDistributionV1"
            )
        if distributions != tuple(sorted(distributions)):
            raise ValueError("installed distributions must be sorted")
        names = [distribution.name for distribution in distributions]
        if len(names) != len(set(names)):
            raise ValueError("installed distribution names must be unique")
        versions = {
            distribution.name: distribution.version for distribution in distributions
        }
        if versions.get("numpy") != numpy_version:
            raise ValueError(
                "NumPy runtime version must match the installed distribution"
            )
        if versions.get("scipy") != scipy_version:
            raise ValueError(
                "SciPy runtime version must match the installed distribution"
            )
        lock = self.dependency_lock
        if lock is not None and type(lock) is not DependencyLockV1:
            raise ValueError("dependency_lock must be DependencyLockV1 or None")

        configuration = _normalize_numpy_configuration(
            self.numpy_configuration_text
        )
        object.__setattr__(self, "python_implementation", implementation)
        object.__setattr__(self, "python_version", python_version)
        object.__setattr__(self, "python_compiler", compiler)
        object.__setattr__(self, "numpy_version", numpy_version)
        object.__setattr__(self, "numpy_configuration_text", configuration)
        object.__setattr__(self, "scipy_version", scipy_version)
        object.__setattr__(self, "logical_cpu_count", cpu_count)
        object.__setattr__(self, "execution_controls", normalized_controls)
        object.__setattr__(self, "installed_distributions", distributions)

    @property
    def installed_distributions_sha256(self) -> str:
        payload = [item.as_dict() for item in self.installed_distributions]
        return _sha256_bytes(_canonical_json(payload))

    @property
    def numpy_configuration_sha256(self) -> str:
        return _sha256_bytes(self.numpy_configuration_text.encode("utf-8"))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": NUMERICAL_ENVIRONMENT_SCHEMA,
            "schema_version": NUMERICAL_ENVIRONMENT_SCHEMA_VERSION,
            "python": {
                "implementation": self.python_implementation,
                "version": self.python_version,
                "compiler": self.python_compiler,
            },
            "numpy": {
                "version": self.numpy_version,
                "configuration_text": self.numpy_configuration_text,
                "configuration_sha256": self.numpy_configuration_sha256,
            },
            "scipy_version": self.scipy_version,
            "logical_cpu_count": self.logical_cpu_count,
            "byte_order": self.byte_order,
            "execution_controls": dict(self.execution_controls),
            "installed_distributions": [
                item.as_dict() for item in self.installed_distributions
            ],
            "installed_distributions_sha256": (
                self.installed_distributions_sha256
            ),
            "dependency_lock": (
                None if self.dependency_lock is None else self.dependency_lock.as_dict()
            ),
        }

    @property
    def profile_id(self) -> str:
        return _sha256_bytes(_canonical_json(self.descriptor()))

    def as_dict(self) -> dict[str, object]:
        return {"profile_id": self.profile_id, **self.descriptor()}


def _capture_installed_distributions() -> tuple[InstalledDistributionV1, ...]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution is missing its Name metadata")
        name = _canonical_distribution_name(str(raw_name))
        version = str(distribution.version)
        previous = versions.get(name)
        if previous is not None and previous != version:
            raise ValueError(
                f"installed distribution {name!r} has conflicting versions"
            )
        versions[name] = version
    return tuple(
        InstalledDistributionV1(name=name, version=version)
        for name, version in sorted(versions.items())
    )


def _capture_numpy_configuration() -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    captured = stream.getvalue()
    if not captured:
        captured = "(no NumPy build configuration reported)\n"
    return _normalize_numpy_configuration(captured)


def capture_numerical_environment_v1(
    *,
    dependency_lock: str | Path | None = None,
    dependency_lock_name: str | None = None,
) -> NumericalEnvironmentV1:
    """Capture the exact installed numerical stack without arbitrary secrets."""

    distributions = _capture_installed_distributions()
    distribution_versions = {
        distribution.name: distribution.version for distribution in distributions
    }
    numpy_version = str(np.__version__)
    lock: DependencyLockV1 | None = None
    if dependency_lock is not None:
        path = Path(dependency_lock)
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name if dependency_lock_name is None else dependency_lock_name
        lock = DependencyLockV1(
            name=name,
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    elif dependency_lock_name is not None:
        raise ValueError("dependency_lock_name requires dependency_lock")

    return NumericalEnvironmentV1(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_compiler=platform.python_compiler(),
        numpy_version=numpy_version,
        numpy_configuration_text=_capture_numpy_configuration(),
        scipy_version=distribution_versions.get("scipy"),
        logical_cpu_count=os.cpu_count(),
        byte_order=sys.byteorder,
        execution_controls={
            name: os.environ.get(name) for name in _EXECUTION_CONTROL_NAMES
        },
        installed_distributions=distributions,
        dependency_lock=lock,
    )


def _distribution_from_dict(value: Mapping[str, Any]) -> InstalledDistributionV1:
    _require_exact_fields(
        value,
        expected=_DISTRIBUTION_FIELDS,
        name="installed distribution",
    )
    return InstalledDistributionV1(
        name=_require_text(value["name"], name="distribution name"),
        version=_require_text(value["version"], name="distribution version"),
    )


def _dependency_lock_from_dict(value: Mapping[str, Any]) -> DependencyLockV1:
    _require_exact_fields(value, expected=_LOCK_FIELDS, name="dependency lock")
    return DependencyLockV1(
        name=_require_text(value["name"], name="dependency lock name"),
        sha256=_require_sha256(value["sha256"], name="dependency lock SHA-256"),
        size_bytes=_require_integer(
            value["size_bytes"],
            name="dependency lock size",
            minimum=0,
        ),
    )


def _execution_controls_from_dict(
    value: Mapping[str, Any],
) -> dict[str, str | None]:
    controls: dict[str, str | None] = {}
    for name, raw_value in value.items():
        if raw_value is not None and type(raw_value) is not str:
            raise ValueError(f"execution control {name} must be text or null")
        controls[name] = cast(str | None, raw_value)
    return controls


def numerical_environment_from_dict(
    value: Mapping[str, Any],
    *,
    require_dependency_lock: bool = False,
) -> NumericalEnvironmentV1:
    """Validate one serialized numerical environment and its content identity."""

    payload = _require_mapping(value, name="numerical environment")
    _require_exact_fields(
        payload,
        expected=_PROFILE_FIELDS,
        name="numerical environment",
    )
    schema_name = _require_text(payload["schema_name"], name="schema name")
    if schema_name != NUMERICAL_ENVIRONMENT_SCHEMA:
        raise ValueError("unsupported numerical-environment schema")
    schema_version = _require_integer(
        payload["schema_version"],
        name="schema version",
        minimum=1,
    )
    if schema_version != NUMERICAL_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("unsupported numerical-environment version")
    expected_profile_id = _require_sha256(
        payload["profile_id"],
        name="numerical environment profile ID",
    )

    python_record = _require_mapping(payload["python"], name="Python record")
    _require_exact_fields(
        python_record,
        expected=_PYTHON_FIELDS,
        name="Python record",
    )
    numpy_record = _require_mapping(payload["numpy"], name="NumPy record")
    _require_exact_fields(
        numpy_record,
        expected=_NUMPY_FIELDS,
        name="NumPy record",
    )
    distributions = tuple(
        _distribution_from_dict(
            _require_mapping(item, name="installed distribution")
        )
        for item in _require_sequence(
            payload["installed_distributions"],
            name="installed distributions",
        )
    )
    dependency_lock_value = payload["dependency_lock"]
    dependency_lock_record = (
        None
        if dependency_lock_value is None
        else _dependency_lock_from_dict(
            _require_mapping(dependency_lock_value, name="dependency lock")
        )
    )
    if require_dependency_lock and dependency_lock_record is None:
        raise ValueError("numerical environment requires a dependency lock")

    cpu_value = payload["logical_cpu_count"]
    if cpu_value is not None:
        cpu_value = _require_integer(
            cpu_value,
            name="logical CPU count",
            minimum=1,
        )
    scipy_value = payload["scipy_version"]
    if scipy_value is not None:
        scipy_value = _require_text(scipy_value, name="SciPy version")

    profile = NumericalEnvironmentV1(
        python_implementation=_require_text(
            python_record["implementation"],
            name="Python implementation",
        ),
        python_version=_require_text(
            python_record["version"],
            name="Python version",
        ),
        python_compiler=_require_text(
            python_record["compiler"],
            name="Python compiler",
            allow_empty=True,
        ),
        numpy_version=_require_text(
            numpy_record["version"],
            name="NumPy version",
        ),
        numpy_configuration_text=_require_multiline_text(
            numpy_record["configuration_text"],
            name="NumPy build configuration",
        ),
        scipy_version=cast(str | None, scipy_value),
        logical_cpu_count=cast(int | None, cpu_value),
        byte_order=_require_text(payload["byte_order"], name="byte order"),
        execution_controls=_execution_controls_from_dict(
            _require_mapping(
                payload["execution_controls"],
                name="execution controls",
            )
        ),
        installed_distributions=distributions,
        dependency_lock=dependency_lock_record,
    )
    expected_distribution_digest = _require_sha256(
        payload["installed_distributions_sha256"],
        name="installed-distribution digest",
    )
    if profile.installed_distributions_sha256 != expected_distribution_digest:
        raise ValueError("installed-distribution digest does not match its payload")
    expected_configuration_digest = _require_sha256(
        numpy_record["configuration_sha256"],
        name="NumPy configuration digest",
    )
    if profile.numpy_configuration_sha256 != expected_configuration_digest:
        raise ValueError("NumPy configuration digest does not match its payload")
    if profile.profile_id != expected_profile_id:
        raise ValueError("numerical environment profile ID does not match its payload")
    return profile


def _strict_json_value(
    value: object,
    *,
    name: str,
    path: str,
    active_containers: set[int],
) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{name} contains a circular mapping at {path}")
        active_containers.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(
                        f"{name} requires literal string keys at {path}"
                    )
                result[key] = _strict_json_value(
                    item,
                    name=name,
                    path=f"{path}.{key}",
                    active_containers=active_containers,
                )
            return {key: result[key] for key in sorted(result)}
        finally:
            active_containers.remove(identity)
    if type(value) is list or type(value) is tuple:
        sequence = cast(Sequence[object], value)
        identity = id(sequence)
        if identity in active_containers:
            raise ValueError(f"{name} contains a circular sequence at {path}")
        active_containers.add(identity)
        try:
            return [
                _strict_json_value(
                    item,
                    name=name,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(sequence)
            ]
        finally:
            active_containers.remove(identity)
    raise ValueError(
        f"{name} contains a non-JSON value at {path}: "
        f"{type(value).__name__}"
    )


def _strict_json_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    detached = _strict_json_value(
        value,
        name=name,
        path="$",
        active_containers=set(),
    )
    if not isinstance(detached, dict):
        raise AssertionError("mapping validation did not return a dictionary")
    return detached


def embed_numerical_environment_v1(
    runtime_environment: Mapping[str, Any],
    profile: NumericalEnvironmentV1,
) -> dict[str, Any]:
    """Attach one validated profile to a detached runtime-environment mapping."""

    if type(profile) is not NumericalEnvironmentV1:
        raise ValueError("profile must be NumericalEnvironmentV1")
    detached = _strict_json_mapping(
        runtime_environment,
        name="runtime environment",
    )
    if NUMERICAL_ENVIRONMENT_RUNTIME_KEY in detached:
        raise ValueError("runtime environment already contains a numerical profile")
    detached[NUMERICAL_ENVIRONMENT_RUNTIME_KEY] = profile.as_dict()
    return dict(sorted(detached.items()))


def validate_embedded_numerical_environment_v1(
    runtime_environment: Mapping[str, Any],
    *,
    require_profile: bool = False,
    require_dependency_lock: bool = False,
) -> NumericalEnvironmentV1 | None:
    """Validate an optional profile nested in a run-manifest runtime mapping."""

    runtime = _require_mapping(runtime_environment, name="runtime environment")
    raw_profile = runtime.get(NUMERICAL_ENVIRONMENT_RUNTIME_KEY)
    if raw_profile is None:
        if require_profile or require_dependency_lock:
            raise ValueError("runtime environment requires a numerical profile")
        return None
    return numerical_environment_from_dict(
        _require_mapping(raw_profile, name="embedded numerical environment"),
        require_dependency_lock=require_dependency_lock,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_runtime_fragment(path: Path) -> Mapping[str, Any]:
    try:
        if path.stat().st_size > _MAX_RUNTIME_FRAGMENT_BYTES:
            raise ValueError("runtime fragment exceeds 16 MiB")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("runtime fragment cannot be read as UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("runtime fragment is malformed JSON") from error
    return _require_mapping(value, name="runtime fragment")


def _write_runtime_fragment(
    path: Path,
    profile: NumericalEnvironmentV1,
    *,
    force: bool,
) -> None:
    payload = {NUMERICAL_ENVIRONMENT_RUNTIME_KEY: profile.as_dict()}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="capture a runtime JSON fragment",
    )
    capture.add_argument("output", type=Path)
    capture.add_argument("--dependency-lock", type=Path)
    capture.add_argument("--dependency-lock-name")
    capture.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate",
        help="validate a runtime JSON fragment",
    )
    validate.add_argument("runtime_fragment", type=Path)
    validate.add_argument("--require-dependency-lock", action="store_true")
    return parser


def _capture_command(args: argparse.Namespace) -> int:
    profile = capture_numerical_environment_v1(
        dependency_lock=args.dependency_lock,
        dependency_lock_name=args.dependency_lock_name,
    )
    _write_runtime_fragment(args.output, profile, force=bool(args.force))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "profile_id": profile.profile_id,
                "distribution_count": len(profile.installed_distributions),
                "dependency_lock": (
                    None
                    if profile.dependency_lock is None
                    else profile.dependency_lock.as_dict()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    runtime = _load_runtime_fragment(args.runtime_fragment)
    profile = validate_embedded_numerical_environment_v1(
        runtime,
        require_profile=True,
        require_dependency_lock=bool(args.require_dependency_lock),
    )
    if profile is None:
        raise AssertionError("required numerical profile was not returned")
    print(
        json.dumps(
            {
                "status": "valid",
                "profile_id": profile.profile_id,
                "distribution_count": len(profile.installed_distributions),
                "dependency_lock_present": profile.dependency_lock is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Capture or validate a numerical runtime profile."""

    args = _build_parser().parse_args(argv)
    if args.command == "capture":
        return _capture_command(args)
    if args.command == "validate":
        return _validate_command(args)
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NUMERICAL_ENVIRONMENT_RUNTIME_KEY",
    "NUMERICAL_ENVIRONMENT_SCHEMA",
    "NUMERICAL_ENVIRONMENT_SCHEMA_VERSION",
    "DependencyLockV1",
    "InstalledDistributionV1",
    "NumericalEnvironmentV1",
    "capture_numerical_environment_v1",
    "embed_numerical_environment_v1",
    "main",
    "numerical_environment_from_dict",
    "validate_embedded_numerical_environment_v1",
]
