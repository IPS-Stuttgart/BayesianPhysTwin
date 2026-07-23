"""Post-barrier official-target scoring for adaptive-covariance confirmation.

The factory in this module returns the exact callback accepted by
``evaluate_adaptive_covariance_confirmation``.  Constructing the callback is
target-free: official future and outcome paths are retained as opaque path
bindings and are opened only when the evaluator invokes the callback after its
complete 34-case prediction barrier.

Each invocation revalidates the supplied barrier case, the corresponding case
seal, and the complete outcome-adapter authorization chain.  The three scored
arms use one frozen sparse 15 mm frame-zero identity transport and the frozen
post-update hidden-identity metrics.  Assimilation centres and route evidence
come only from the sealed target-free diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import io
import inspect
import json
import marshal
import os
from pathlib import Path
import re
import stat
from types import CodeType
from typing import Any
import weakref

import numpy as np

from .deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h2_loaded_runtime,
)
from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)
from .deform360_adaptive_covariance_confirmation_outcome_adapter import (
    COMPATIBILITY_MANIFEST_FILENAME,
    EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME,
    EXTERNAL_AUTHORIZED_OUTCOME_KIND,
    EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
    EXTERNAL_TARGET_ARCHIVE_FILENAME,
    ConfirmationNativeOfficialTarget,
    validate_confirmation_native_official_target,
)
from .deform360_adaptive_covariance_confirmation_seal import (
    ARRAY_ARCHIVE_FILENAME,
    ARRAY_ROLES,
    CASE_MANIFEST_FILENAME,
    DIAGNOSTIC_FILENAME,
    array_sha256,
    artifact_sha256,
    validate_confirmation_case_seal,
)
from .deform360_held_outcome_scoring import (
    OfficialTarget,
    scored_frames,
    transport_official_target,
)
from .deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)


FRAME_COUNT = 76
CENTER_COUNT = 16
UPDATE_FRAMES = (19, 38, 57)
ROUTES = ("4_view_rbf", "8_view_rbf", "physical_prior_fallback")
TARGET_ARRAY_ROLES = (
    "target_m",
    "target_visibility",
    "target_validity",
)
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
ARMS_TO_ARRAY_ROLES = {
    "adaptive": "adaptive_prediction_m",
    "fixed8": "fixed_8_rbf_prediction_m",
    "fixed4": "fixed_4_rbf_prediction_m",
}

_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
_BARRIER_CASE_KEYS = {
    "case_id",
    "manifest_file_sha256",
    "manifest_artifact_sha256",
    "prediction_archive_sha256",
    "diagnostic_file_sha256",
    "diagnostic_artifact_sha256",
}
_DIAGNOSTIC_KEYS = {
    "schema_version",
    "artifact_kind",
    "protocol_id",
    "case_identity",
    "nested_selected_cameras",
    "covariance_routing",
    "technical_disposition",
    "information_boundary",
    "artifact_sha256",
}
_OUTCOME_BOUNDARY = {
    "prediction_cohort_verified_before_target_construction": True,
    "future_tactile_read": False,
    "prediction_metric_computed": False,
}
_NATIVE_EVIDENCE_KEYS = {
    "schema_version",
    "protocol_id",
    "case_identity",
    "lock_binding",
    "prediction_barrier",
    "case_seal",
    "nested_measurement",
    "identity_persistence_adapter",
    "selected_cameras",
    "compatibility_manifest",
    "authorized_future_manifest",
    "authorized_outcome_manifest",
    "target_archive",
    "information_boundary",
}

NativeTargetValidator = Callable[..., ConfirmationNativeOfficialTarget]
SCORING_LOADER_ATTESTATION_KIND = (
    "Deform360AdaptiveCovarianceConfirmationScoringLoaderAttestationV1"
)
SCORING_LOADER_FACTORY_KIND = "h1-bound-native-official-target-scoring-loader-v1"
SCORING_SOURCE_REPOSITORY_PATH = (
    "src/bayesian_phystwin/deform360_adaptive_covariance_confirmation_scoring.py"
)
NATIVE_VALIDATOR_SOURCE_REPOSITORY_PATH = (
    "src/bayesian_phystwin/"
    "deform360_adaptive_covariance_confirmation_outcome_adapter.py"
)
_LOADER_ATTESTATION_ATTRIBUTE = "__confirmation_scoring_attestation__"
_LOADER_ATTESTATION_KEYS = {
    "schema_version",
    "artifact_kind",
    "factory_kind",
    "protocol_id",
    "implementation_commit_h1",
    "cohort_lock_commit_h2",
    "cohort_lock_artifact_sha256",
    "cohort_lock_file_sha256",
    "native_target_validator",
    "scoring_source",
    "loader_callable",
    "repository_provenance",
    "production_eligible",
    "ineligibility_reasons",
    "attestation_sha256",
}
_SOURCE_BINDING_KEYS = {
    "repository_path",
    "source_sha256",
    "canonical_adapter_source",
}
_VALIDATOR_BINDING_KEYS = {
    "module",
    "qualname",
    "is_exact_default",
    *_SOURCE_BINDING_KEYS,
}
_CALLABLE_BINDING_KEYS = {
    "module",
    "qualname",
    "code_sha256",
    "code_filename_repository_path",
    "closure_freevars",
    "closure_cell_count",
    "exact_factory_registry_binding_required",
}
_REPOSITORY_PROVENANCE_KEYS = {
    "validated_exact_clean_h2",
    "implementation_commit_h1",
    "cohort_lock_commit_h2",
    "cohort_lock_repository_path",
    "cohort_lock_file_sha256",
    "cohort_lock_artifact_sha256",
    "source_only_runtime",
    "adapter_python_bytecode_cache_absent",
    "python_source_sha256",
}
_EXACT_H2_RUNTIME_VALIDATOR = validate_confirmation_h2_loaded_runtime


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _RegisteredLoader:
    attestation: bytes
    loader_reference: weakref.ReferenceType[Callable[..., Mapping[str, Any]]]
    code: CodeType
    globals_identity: int
    closure_cell_identities: tuple[int, ...]
    defaults: object
    keyword_defaults: object


_LOADER_ATTESTATIONS: weakref.WeakKeyDictionary[
    Callable[..., Mapping[str, Any]], _RegisteredLoader
] = weakref.WeakKeyDictionary()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _attestation_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("attestation_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _code_sha256(code: CodeType) -> str:
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _source_binding(
    source_value: object,
    *,
    adapter_repository: Path,
    repository_path: str,
) -> dict[str, Any]:
    source = (
        Path(source_value).absolute()
        if isinstance(source_value, (str, os.PathLike))
        else None
    )
    expected = adapter_repository / repository_path
    canonical = bool(
        source is not None
        and source.is_file()
        and not source.is_symlink()
        and source.resolve(strict=True) == source
        and source == expected
    )
    digest = (
        hashlib.sha256(source.read_bytes()).hexdigest()
        if source is not None and source.is_file() and not source.is_symlink()
        else None
    )
    return {
        "repository_path": repository_path,
        "source_sha256": digest,
        "canonical_adapter_source": canonical,
    }


def _build_loader_attestation(
    *,
    adapter_repository: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    h2_commit: str,
    expected_h1: str,
    native_target_validator: NativeTargetValidator,
    target_loader: Callable[..., Mapping[str, Any]],
    repository_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scoring_source = _source_binding(
        __file__,
        adapter_repository=adapter_repository,
        repository_path=SCORING_SOURCE_REPOSITORY_PATH,
    )
    try:
        native_source_file = inspect.getsourcefile(native_target_validator)
    except TypeError:
        native_source_file = None
    native_source = _source_binding(
        native_source_file,
        adapter_repository=adapter_repository,
        repository_path=NATIVE_VALIDATOR_SOURCE_REPOSITORY_PATH,
    )
    exact_default = (
        native_target_validator is validate_confirmation_native_official_target
    )
    native_binding = {
        "module": getattr(native_target_validator, "__module__", None),
        "qualname": getattr(native_target_validator, "__qualname__", None),
        "is_exact_default": exact_default,
        **native_source,
    }
    callable_binding = {
        "module": getattr(target_loader, "__module__", None),
        "qualname": getattr(target_loader, "__qualname__", None),
        "code_sha256": _code_sha256(target_loader.__code__),
        "code_filename_repository_path": SCORING_SOURCE_REPOSITORY_PATH,
        "closure_freevars": list(target_loader.__code__.co_freevars),
        "closure_cell_count": len(target_loader.__closure__ or ()),
        "exact_factory_registry_binding_required": True,
    }
    if repository_provenance is None:
        repository_binding: dict[str, Any] = {
            "validated_exact_clean_h2": False,
            "implementation_commit_h1": None,
            "cohort_lock_commit_h2": None,
            "cohort_lock_repository_path": None,
            "cohort_lock_file_sha256": None,
            "cohort_lock_artifact_sha256": None,
            "source_only_runtime": False,
            "adapter_python_bytecode_cache_absent": False,
            "python_source_sha256": {},
        }
    else:
        repository_binding = {
            "validated_exact_clean_h2": True,
            "implementation_commit_h1": repository_provenance.get(
                "implementation_commit_h1"
            ),
            "cohort_lock_commit_h2": repository_provenance.get("cohort_lock_commit_h2"),
            "cohort_lock_repository_path": repository_provenance.get(
                "cohort_lock_repository_path"
            ),
            "cohort_lock_file_sha256": repository_provenance.get(
                "cohort_lock_file_sha256"
            ),
            "cohort_lock_artifact_sha256": repository_provenance.get(
                "cohort_lock_artifact_sha256"
            ),
            "source_only_runtime": repository_provenance.get("source_only_runtime"),
            "adapter_python_bytecode_cache_absent": repository_provenance.get(
                "adapter_python_bytecode_cache_absent"
            ),
            "python_source_sha256": repository_provenance.get("python_source_sha256"),
        }
    reasons: list[str] = []
    if repository_provenance is None:
        reasons.append("factory_not_bound_to_exact_clean_h2")
    if not exact_default:
        reasons.append("native_target_validator_is_not_exact_default")
    if not scoring_source["canonical_adapter_source"]:
        reasons.append("scoring_source_is_not_canonical_adapter_source")
    if not native_binding["canonical_adapter_source"]:
        reasons.append("native_validator_source_is_not_canonical_adapter_source")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SCORING_LOADER_ATTESTATION_KIND,
        "factory_kind": SCORING_LOADER_FACTORY_KIND,
        "protocol_id": PROTOCOL_ID,
        "implementation_commit_h1": expected_h1,
        "cohort_lock_commit_h2": h2_commit,
        "cohort_lock_artifact_sha256": lock["artifact_sha256"],
        "cohort_lock_file_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "native_target_validator": native_binding,
        "scoring_source": scoring_source,
        "loader_callable": callable_binding,
        "repository_provenance": repository_binding,
        "production_eligible": not reasons,
        "ineligibility_reasons": reasons,
    }
    payload["attestation_sha256"] = _attestation_sha256(payload)
    return payload


def validate_confirmation_case_target_loader_attestation(
    target_loader: Callable[[str, Path, Mapping[str, Any]], Mapping[str, Any]],
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_h1: str,
    require_production: bool = True,
) -> dict[str, Any]:
    """Validate a factory-issued loader attestation without trusting attributes."""

    _require(_is_sha1(expected_h1), "expected H1 is invalid")
    _require(
        _is_sha1(h2_commit) and h2_commit != expected_h1,
        "H2 commit is invalid",
    )
    try:
        registered = _LOADER_ATTESTATIONS.get(target_loader)
    except TypeError:
        registered = None
    _require(
        isinstance(registered, _RegisteredLoader),
        "target loader was not issued by the frozen scoring factory",
    )
    closure = getattr(target_loader, "__closure__", None) or ()
    _require(
        registered.loader_reference() is target_loader
        and getattr(target_loader, "__code__", None) is registered.code
        and id(getattr(target_loader, "__globals__", None))
        == registered.globals_identity
        and tuple(id(cell) for cell in closure) == registered.closure_cell_identities
        and getattr(target_loader, "__defaults__", None) is registered.defaults
        and getattr(target_loader, "__kwdefaults__", None)
        is registered.keyword_defaults,
        "target loader factory code or closure registry binding changed",
    )
    try:
        attestation = json.loads(registered.attestation.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("registered scoring attestation is invalid") from error
    _require(
        isinstance(attestation, dict) and set(attestation) == _LOADER_ATTESTATION_KEYS,
        "scoring attestation schema changed",
    )
    declared = getattr(target_loader, _LOADER_ATTESTATION_ATTRIBUTE, None)
    _require(
        declared == attestation,
        "target loader scoring attestation attribute changed",
    )
    _require(
        attestation["schema_version"] == 1
        and attestation["artifact_kind"] == SCORING_LOADER_ATTESTATION_KIND
        and attestation["factory_kind"] == SCORING_LOADER_FACTORY_KIND
        and attestation["protocol_id"] == PROTOCOL_ID,
        "scoring attestation identity changed",
    )
    _require(
        attestation["implementation_commit_h1"] == expected_h1
        and attestation["cohort_lock_commit_h2"] == h2_commit,
        "scoring attestation commit binding changed",
    )
    _require(
        attestation["attestation_sha256"] == _attestation_sha256(attestation),
        "scoring attestation checksum changed",
    )
    lock_source = Path(lock_path).absolute()
    lock = load_confirmation_cohort_lock(
        lock_source,
        expected_implementation_commit_h1=expected_h1,
    )
    _require(
        attestation["cohort_lock_artifact_sha256"] == lock["artifact_sha256"]
        and attestation["cohort_lock_file_sha256"]
        == hashlib.sha256(lock_source.read_bytes()).hexdigest(),
        "scoring attestation lock binding changed",
    )
    scoring_source = attestation["scoring_source"]
    native = attestation["native_target_validator"]
    callable_binding = attestation["loader_callable"]
    repository_binding = attestation["repository_provenance"]
    _require(
        isinstance(scoring_source, Mapping)
        and set(scoring_source) == _SOURCE_BINDING_KEYS
        and isinstance(native, Mapping)
        and set(native) == _VALIDATOR_BINDING_KEYS
        and isinstance(callable_binding, Mapping)
        and set(callable_binding) == _CALLABLE_BINDING_KEYS
        and isinstance(repository_binding, Mapping)
        and set(repository_binding) == _REPOSITORY_PROVENANCE_KEYS,
        "scoring source attestation schema changed",
    )
    _require(
        scoring_source["repository_path"] == SCORING_SOURCE_REPOSITORY_PATH
        and scoring_source["source_sha256"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "live scoring source differs from its loader attestation",
    )
    _require(
        callable_binding["module"] == target_loader.__module__
        and callable_binding["qualname"] == target_loader.__qualname__
        and callable_binding["code_sha256"] == _code_sha256(target_loader.__code__)
        and callable_binding["code_filename_repository_path"]
        == SCORING_SOURCE_REPOSITORY_PATH
        and callable_binding["closure_freevars"]
        == list(target_loader.__code__.co_freevars)
        and callable_binding["closure_cell_count"] == len(closure)
        and callable_binding["exact_factory_registry_binding_required"] is True,
        "target loader callable code or closure attestation changed",
    )
    if require_production:
        _require(
            attestation["production_eligible"] is True
            and attestation["ineligibility_reasons"] == []
            and repository_binding["validated_exact_clean_h2"] is True
            and scoring_source["canonical_adapter_source"] is True
            and native["canonical_adapter_source"] is True
            and native["is_exact_default"] is True
            and native["module"]
            == validate_confirmation_native_official_target.__module__
            and native["qualname"]
            == validate_confirmation_native_official_target.__qualname__,
            "target loader is not eligible for production confirmation scoring",
        )
        _require(
            validate_confirmation_h2_loaded_runtime is _EXACT_H2_RUNTIME_VALIDATOR,
            "production H2 runtime validator capability changed",
        )
        live_repository = validate_confirmation_h2_loaded_runtime(
            Path(__file__).parents[2],
            lock_source,
            h2_commit,
            expected_h1=expected_h1,
            source_file=__file__,
            source_repository_path=SCORING_SOURCE_REPOSITORY_PATH,
        )
        expected_repository_binding = {
            "validated_exact_clean_h2": True,
            "implementation_commit_h1": live_repository["implementation_commit_h1"],
            "cohort_lock_commit_h2": live_repository["cohort_lock_commit_h2"],
            "cohort_lock_repository_path": live_repository[
                "cohort_lock_repository_path"
            ],
            "cohort_lock_file_sha256": live_repository["cohort_lock_file_sha256"],
            "cohort_lock_artifact_sha256": live_repository[
                "cohort_lock_artifact_sha256"
            ],
            "source_only_runtime": live_repository["source_only_runtime"],
            "adapter_python_bytecode_cache_absent": live_repository[
                "adapter_python_bytecode_cache_absent"
            ],
            "python_source_sha256": live_repository["python_source_sha256"],
        }
        attested_sources = repository_binding["python_source_sha256"]
        live_sources = expected_repository_binding["python_source_sha256"]
        _require(
            isinstance(attested_sources, Mapping)
            and isinstance(live_sources, Mapping)
            and SCORING_SOURCE_REPOSITORY_PATH in attested_sources
            and NATIVE_VALIDATOR_SOURCE_REPOSITORY_PATH in attested_sources
            and all(
                live_sources.get(path) == digest
                for path, digest in attested_sources.items()
            ),
            "target loader committed Python source provenance changed",
        )
        _require(
            {
                key: value
                for key, value in repository_binding.items()
                if key != "python_source_sha256"
            }
            == {
                key: value
                for key, value in expected_repository_binding.items()
                if key != "python_source_sha256"
            },
            "target loader repository provenance changed",
        )
        default_source = inspect.getsourcefile(
            validate_confirmation_native_official_target
        )
        _require(
            native["repository_path"] == NATIVE_VALIDATOR_SOURCE_REPOSITORY_PATH
            and isinstance(default_source, str)
            and native["source_sha256"]
            == hashlib.sha256(Path(default_source).read_bytes()).hexdigest(),
            "live native target validator differs from its loader attestation",
        )
    return json.loads(_canonical_bytes(attestation).decode("utf-8"))


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON artifact has duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _snapshot_file(path: str | Path, *, label: str) -> _FileSnapshot:
    source = Path(path).absolute()
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"{label} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    _require(
        source.resolve(strict=True) == source,
        f"{label} has a symlinked or noncanonical path",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"{label} changed while reading",
        )
        _require(len(payload) == opened.st_size, f"{label} read was incomplete")
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=source,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=identity,
    )


def _require_unchanged(snapshot: _FileSnapshot, *, label: str) -> None:
    _require(
        _snapshot_file(snapshot.path, label=label) == snapshot,
        f"{label} changed during official scoring",
    )


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path).absolute()
    _require(
        root.resolve(strict=True) == root and root.is_dir() and not root.is_symlink(),
        f"{label} is not a canonical directory",
    )
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _load_npz_bytes(
    payload: bytes,
    *,
    expected_roles: tuple[str, ...],
    label: str,
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
            _require(
                len(stored.files) == len(expected_roles)
                and set(stored.files) == set(expected_roles),
                f"{label} roles changed",
            )
            result = {
                role: np.array(stored[role], copy=True, order="C")
                for role in expected_roles
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{label} is invalid") from error
    for value in result.values():
        value.setflags(write=False)
    return result


def _external_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _target_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    hashes = {role: _external_array_sha256(arrays[role]) for role in TARGET_ARRAY_ROLES}
    return hashlib.sha256(_canonical_bytes(hashes)).hexdigest()


def _result_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_barrier_case(
    case_id: str,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == _BARRIER_CASE_KEYS,
        f"{case_id} barrier case schema changed",
    )
    record = dict(value)
    _require(record["case_id"] == case_id, f"{case_id} barrier case ID changed")
    for key in _BARRIER_CASE_KEYS - {"case_id"}:
        _require(_is_sha256(record[key]), f"{case_id} {key} is invalid")
    return record


def _validate_case_inputs(
    case_id: str,
    case_root: Path,
    barrier_case: Mapping[str, Any],
    lock_path: Path,
    h2_commit: str,
    expected_h1: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    tuple[_FileSnapshot, _FileSnapshot, _FileSnapshot],
]:
    validated = validate_confirmation_case_seal(
        case_root,
        lock_path,
        h2_commit,
        expected_case_id=case_id,
        expected_h1=expected_h1,
    )
    _require(
        validated == barrier_case,
        f"{case_id} case seal differs from the evaluator barrier record",
    )
    manifest_snapshot = _snapshot_file(
        case_root / CASE_MANIFEST_FILENAME,
        label=f"{case_id} case prediction seal",
    )
    archive_snapshot = _snapshot_file(
        case_root / ARRAY_ARCHIVE_FILENAME,
        label=f"{case_id} sealed prediction archive",
    )
    diagnostic_snapshot = _snapshot_file(
        case_root / DIAGNOSTIC_FILENAME,
        label=f"{case_id} sealed target-free diagnostic",
    )
    _require(
        manifest_snapshot.sha256 == barrier_case["manifest_file_sha256"]
        and archive_snapshot.sha256 == barrier_case["prediction_archive_sha256"]
        and diagnostic_snapshot.sha256 == barrier_case["diagnostic_file_sha256"],
        f"{case_id} case files changed after barrier validation",
    )
    manifest = _load_json_bytes(
        manifest_snapshot.payload,
        label=f"{case_id} case prediction seal",
    )
    diagnostic = _load_json_bytes(
        diagnostic_snapshot.payload,
        label=f"{case_id} sealed target-free diagnostic",
    )
    _require(
        manifest.get("artifact_sha256")
        == barrier_case["manifest_artifact_sha256"]
        == artifact_sha256(manifest)
        and diagnostic.get("artifact_sha256")
        == barrier_case["diagnostic_artifact_sha256"]
        == artifact_sha256(diagnostic),
        f"{case_id} case artifact hash changed after barrier validation",
    )
    _require(
        set(diagnostic) == _DIAGNOSTIC_KEYS
        and diagnostic.get("protocol_id") == PROTOCOL_ID
        and diagnostic.get("case_identity", {}).get("case_id") == case_id,
        f"{case_id} sealed diagnostic schema changed",
    )
    arrays = _load_npz_bytes(
        archive_snapshot.payload,
        expected_roles=ARRAY_ROLES,
        label=f"{case_id} sealed prediction archive",
    )
    return (
        arrays,
        diagnostic,
        (manifest_snapshot, archive_snapshot, diagnostic_snapshot),
    )


def _validate_centers_and_updates(
    case_id: str,
    diagnostic: Mapping[str, Any],
    *,
    point_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    disposition = diagnostic["technical_disposition"]
    _require(
        isinstance(disposition, Mapping)
        and disposition.get("status")
        in {"prediction_complete", "retained_technical_failure"}
        and disposition.get("case_retained") is True
        and disposition.get("disposition_based_on_target_or_outcome") is False,
        f"{case_id} sealed technical disposition changed",
    )
    center_ids = disposition.get("center_ids")
    _require(
        isinstance(center_ids, list)
        and len(center_ids) == len(set(center_ids)) == CENTER_COUNT
        and all(
            type(center_id) is int and 0 <= center_id < point_count
            for center_id in center_ids
        ),
        f"{case_id} must have exactly 16 sealed assimilation center IDs",
    )
    retained_failure = disposition["status"] == "retained_technical_failure"
    failure_code = disposition.get("failure_code")
    if retained_failure:
        _require(
            isinstance(failure_code, str) and bool(failure_code),
            f"{case_id} retained technical failure has no failure code",
        )

    selected = diagnostic["nested_selected_cameras"]
    _require(
        isinstance(selected, Mapping)
        and set(selected) == {"4", "8"}
        and isinstance(selected["4"], list)
        and isinstance(selected["8"], list)
        and len(selected["4"]) == len(set(selected["4"])) == 4
        and len(selected["8"]) == len(set(selected["8"])) == 8
        and selected["8"][:4] == selected["4"]
        and all(isinstance(camera, str) and bool(camera) for camera in selected["8"]),
        f"{case_id} sealed nested camera panel changed",
    )
    routing = diagnostic["covariance_routing"]
    updates = routing.get("updates") if isinstance(routing, Mapping) else None
    _require(
        isinstance(updates, list) and len(updates) == len(UPDATE_FRAMES),
        f"{case_id} sealed route count changed",
    )
    result: list[dict[str, Any]] = []
    for frame, update in zip(UPDATE_FRAMES, updates, strict=True):
        _require(
            isinstance(update, Mapping)
            and update.get("frame") == frame
            and update.get("route") in ROUTES,
            f"{case_id} sealed update route changed",
        )
        route = str(update["route"])
        attempted = update.get("tracked_cameras")
        expected_attempted = selected["4"] if route == "4_view_rbf" else selected["8"]
        _require(
            attempted == expected_attempted,
            f"{case_id} sealed attempted camera order changed",
        )
        applied = update.get("rbf_correction_applied")
        state_updated = update.get("state_updated")
        _require(
            type(applied) is bool and type(state_updated) is bool,
            f"{case_id} sealed update flags changed",
        )
        if route == "physical_prior_fallback":
            _require(
                applied is False and state_updated is False,
                f"{case_id} sealed fallback updated visual state",
            )
            fallback_reason = (
                str(failure_code) if retained_failure else "covariance_abstention"
            )
        else:
            _require(
                applied is True and state_updated is True,
                f"{case_id} sealed accepted route did not update state",
            )
            fallback_reason = None
        result.append(
            {
                "update_frame": frame,
                "route": route,
                "attempted_camera_ids": list(attempted),
                "future_visual_update_applied": applied,
                "rbf_state_updated": state_updated,
                "fallback_reason": fallback_reason,
            }
        )
    centers = np.asarray(center_ids, dtype=np.int64)
    centers.setflags(write=False)
    return centers, result


def _case_identity(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    identity = diagnostic["case_identity"]
    _require(
        isinstance(identity, Mapping)
        and set(identity) == {"case_id", "stratum", "object_id", "episode_id"}
        and isinstance(identity["case_id"], str)
        and isinstance(identity["stratum"], str)
        and isinstance(identity["object_id"], str)
        and type(identity["episode_id"]) is int,
        "sealed case identity changed",
    )
    return dict(identity)


def _validate_native_evidence(
    native: ConfirmationNativeOfficialTarget,
    *,
    case_id: str,
    case_root: Path,
    barrier_case: Mapping[str, Any],
    identity: Mapping[str, Any],
    selected_cameras: list[str],
    prediction_arrays: Mapping[str, np.ndarray],
    compatibility_root: Path,
    lock_binding: Mapping[str, Any],
    barrier_snapshot: _FileSnapshot,
    barrier_artifact_sha256: str,
    future_manifest_snapshot: _FileSnapshot,
    outcome_manifest_snapshot: _FileSnapshot,
    target_archive_snapshot: _FileSnapshot,
) -> Mapping[str, Any]:
    _require(
        isinstance(native, ConfirmationNativeOfficialTarget),
        f"{case_id} native target validator returned another schema",
    )
    evidence = native.evidence
    _require(
        isinstance(evidence, Mapping) and set(evidence) == _NATIVE_EVIDENCE_KEYS,
        f"{case_id} native official target evidence schema changed",
    )
    _require(
        evidence["schema_version"] == 1
        and evidence["protocol_id"] == PROTOCOL_ID
        and evidence["case_identity"] == identity
        and evidence["lock_binding"] == lock_binding
        and evidence["selected_cameras"] == selected_cameras
        and evidence["information_boundary"]
        == {
            "native_official_arrays_returned": True,
            "metric_or_score_computed": False,
        },
        f"{case_id} native official target authorization changed",
    )
    nested_measurement = evidence["nested_measurement"]
    retained_source = (
        nested_measurement.get("retained_failure_source")
        if isinstance(nested_measurement, Mapping)
        else None
    )
    expected_identity_persistence = (
        retained_source.get("identity_persistence_adapter")
        if isinstance(retained_source, Mapping)
        else None
    )
    _require(
        isinstance(nested_measurement, Mapping)
        and evidence["identity_persistence_adapter"] == expected_identity_persistence,
        f"{case_id} native identity-persistence evidence changed",
    )
    _require(
        evidence["prediction_barrier"]
        == {
            "path": str(barrier_snapshot.path),
            "file_sha256": barrier_snapshot.sha256,
            "artifact_sha256": barrier_artifact_sha256,
        },
        f"{case_id} native target used another prediction barrier",
    )
    case_evidence = evidence["case_seal"]
    _require(
        isinstance(case_evidence, Mapping)
        and case_evidence.get("case_seal_root") == str(case_root)
        and case_evidence.get("case_seal_file_sha256")
        == barrier_case["manifest_file_sha256"]
        and case_evidence.get("case_seal_artifact_sha256")
        == barrier_case["manifest_artifact_sha256"]
        and case_evidence.get("prediction_archive_file_sha256")
        == barrier_case["prediction_archive_sha256"]
        and case_evidence.get("diagnostic_file_sha256")
        == barrier_case["diagnostic_file_sha256"]
        and case_evidence.get("diagnostic_artifact_sha256")
        == barrier_case["diagnostic_artifact_sha256"]
        and case_evidence.get("prediction_arrays")
        == {
            "adaptive_prediction_m": array_sha256(
                prediction_arrays["adaptive_prediction_m"]
            ),
            "selected_raw_prediction_m": array_sha256(
                prediction_arrays["selected_raw_prediction_m"]
            ),
        },
        f"{case_id} native target case-seal binding changed",
    )
    compatibility_record = evidence["compatibility_manifest"]
    _require(
        isinstance(compatibility_record, Mapping)
        and set(compatibility_record) == {"path", "file_sha256", "result_sha256"}
        and compatibility_record["path"]
        == str(compatibility_root / COMPATIBILITY_MANIFEST_FILENAME)
        and _is_sha256(compatibility_record["file_sha256"])
        and _is_sha256(compatibility_record["result_sha256"]),
        f"{case_id} compatibility authorization evidence changed",
    )
    expected_file_evidence = (
        (
            "authorized_future_manifest",
            future_manifest_snapshot,
        ),
        (
            "authorized_outcome_manifest",
            outcome_manifest_snapshot,
        ),
    )
    for key, snapshot in expected_file_evidence:
        record = evidence[key]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "file_sha256", "result_sha256"}
            and record["path"] == str(snapshot.path)
            and record["file_sha256"] == snapshot.sha256
            and _is_sha256(record["result_sha256"]),
            f"{case_id} {key} evidence changed",
        )
    target_arrays = {
        "target_m": np.asarray(native.target_m),
        "target_visibility": np.asarray(native.target_visibility),
        "target_validity": np.asarray(native.target_validity),
    }
    target_record = evidence["target_archive"]
    _require(
        isinstance(target_record, Mapping)
        and set(target_record) == {"path", "file_sha256", "arrays"}
        and target_record["path"] == str(target_archive_snapshot.path)
        and target_record["file_sha256"] == target_archive_snapshot.sha256
        and target_record["arrays"]
        == {
            role: _external_array_sha256(target_arrays[role])
            for role in TARGET_ARRAY_ROLES
        },
        f"{case_id} native target archive evidence changed",
    )
    return evidence


def _validate_target_inputs(
    case_id: str,
    future_root: Path,
    outcome_root: Path,
    native: ConfirmationNativeOfficialTarget,
    *,
    identity: Mapping[str, Any],
    selected_cameras: list[str],
    lock_artifact_sha256: str,
    sealed_frame_zero_m: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    tuple[_FileSnapshot, _FileSnapshot, _FileSnapshot],
    dict[str, Any],
]:
    future_snapshot = _snapshot_file(
        future_root / EXTERNAL_AUTHORIZED_FUTURE_MANIFEST_FILENAME,
        label=f"{case_id} authorized future manifest",
    )
    outcome_snapshot = _snapshot_file(
        outcome_root / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
        label=f"{case_id} authorized outcome manifest",
    )
    archive_snapshot = _snapshot_file(
        outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME,
        label=f"{case_id} native official target archive",
    )
    future = _load_json_bytes(
        future_snapshot.payload,
        label=f"{case_id} authorized future manifest",
    )
    outcome = _load_json_bytes(
        outcome_snapshot.payload,
        label=f"{case_id} authorized outcome manifest",
    )
    expected_external_identity = {
        "case": case_id,
        "object_id": identity["object_id"],
        "episode_id": identity["episode_id"],
        "episode_key": (f"{identity['object_id']}/{int(identity['episode_id'])}"),
        "stratum": identity["stratum"],
        "role": "calibration",
    }
    _require(
        outcome.get("schema_version") == 1
        and outcome.get("artifact_kind") == EXTERNAL_AUTHORIZED_OUTCOME_KIND
        and outcome.get("protocol_id") == PROTOCOL_ID
        and outcome.get("protocol_config_sha256") == lock_artifact_sha256
        and all(
            outcome.get(key) == value
            for key, value in expected_external_identity.items()
        )
        and outcome.get("target_frame_count") == FRAME_COUNT
        and outcome.get("cameras") == selected_cameras
        and outcome.get("information_boundary") == _OUTCOME_BOUNDARY
        and outcome.get("result_sha256") == _result_sha256(outcome),
        f"{case_id} authorized outcome manifest changed",
    )
    _require(
        future.get("result_sha256") == _result_sha256(future)
        and native.evidence["authorized_future_manifest"].get("result_sha256")
        == future["result_sha256"]
        and native.evidence["authorized_outcome_manifest"].get("result_sha256")
        == outcome["result_sha256"]
        and outcome.get("inputs_sha256", {}).get("authorized_future_manifest")
        == future_snapshot.sha256,
        f"{case_id} authorized outcome used another future manifest",
    )
    compatibility_evidence = native.evidence["compatibility_manifest"]
    _require(
        isinstance(compatibility_evidence, Mapping)
        and outcome.get("authorization", {}).get("prediction_cohort_result_sha256")
        == compatibility_evidence.get("result_sha256")
        and outcome.get("inputs_sha256", {}).get("prediction_cohort_seal")
        == compatibility_evidence.get("file_sha256"),
        f"{case_id} authorized outcome used another cohort authorization",
    )
    output = outcome.get("output")
    sealed_frame_zero = np.asarray(sealed_frame_zero_m)
    _require(
        sealed_frame_zero.ndim == 2
        and sealed_frame_zero.shape[0] > CENTER_COUNT
        and sealed_frame_zero.shape[1] == 3
        and np.all(np.isfinite(sealed_frame_zero)),
        f"{case_id} sealed frame zero is invalid",
    )
    _require(
        isinstance(output, Mapping)
        and output.get("target_archive")
        == str(outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME)
        and output.get("target_archive_sha256") == archive_snapshot.sha256
        and type(output.get("frame_zero_bit_exact_to_sealed_baseline")) is bool,
        f"{case_id} native target archive manifest binding changed",
    )
    arrays = _load_npz_bytes(
        archive_snapshot.payload,
        expected_roles=TARGET_ARRAY_ROLES,
        label=f"{case_id} native official target archive",
    )
    target = arrays["target_m"]
    visibility = arrays["target_visibility"]
    validity = arrays["target_validity"]
    _require(
        target.dtype == np.dtype(np.float32)
        and target.ndim == 3
        and target.shape[0] == FRAME_COUNT
        and target.shape[1] >= sealed_frame_zero.shape[0]
        and target.shape[2] == 3
        and np.all(np.isfinite(target))
        and visibility.dtype == np.dtype(bool)
        and visibility.shape == target.shape[:2]
        and validity.dtype == np.dtype(bool)
        and validity.shape == target.shape[:2]
        and np.all(visibility)
        and np.all(validity)
        and outcome.get("material_point_count") == target.shape[1]
        and output.get("target_array_sha256") == _external_array_sha256(target),
        f"{case_id} native official target arrays changed",
    )
    frame_zero_is_exact = target.shape[1] == sealed_frame_zero.shape[
        0
    ] and np.array_equal(target[0], sealed_frame_zero)
    _require(
        output["frame_zero_bit_exact_to_sealed_baseline"] is frame_zero_is_exact,
        f"{case_id} native target frame-zero identity declaration changed",
    )
    native_arrays = {
        "target_m": np.asarray(native.target_m),
        "target_visibility": np.asarray(native.target_visibility),
        "target_validity": np.asarray(native.target_validity),
    }
    _require(
        all(
            np.array_equal(arrays[role], native_arrays[role])
            and arrays[role].dtype == native_arrays[role].dtype
            for role in TARGET_ARRAY_ROLES
        ),
        f"{case_id} native validator arrays differ from the sealed archive",
    )
    return (
        arrays,
        (future_snapshot, outcome_snapshot, archive_snapshot),
        outcome,
    )


def _score_case(
    case_id: str,
    case_root: Path,
    barrier_case_value: Mapping[str, Any],
    *,
    adapter_repository: Path,
    lock_path: Path,
    h2_commit: str,
    barrier_path: Path,
    case_seal_dirs: Mapping[str, Path],
    nested_measurement_dirs: Mapping[str, Path],
    compatibility_root: Path,
    authorized_future_case_dir: Path,
    authorized_outcome_case_dir: Path,
    expected_h1: str,
    native_target_validator: NativeTargetValidator,
) -> dict[str, Any]:
    barrier_case = _validate_barrier_case(case_id, barrier_case_value)
    case_root = _canonical_directory(
        case_root,
        label=f"{case_id} sealed case directory",
    )
    future_root = _canonical_directory(
        authorized_future_case_dir,
        label=f"{case_id} authorized future directory",
    )
    outcome_root = _canonical_directory(
        authorized_outcome_case_dir,
        label=f"{case_id} authorized outcome directory",
    )
    _require(
        case_root.name == future_root.name == outcome_root.name == case_id
        and not _paths_overlap(case_root, future_root)
        and not _paths_overlap(case_root, outcome_root)
        and not _paths_overlap(future_root, outcome_root),
        f"{case_id} case/future/outcome path binding changed",
    )
    lock_snapshot = _snapshot_file(lock_path, label="H2 confirmation lock")
    barrier_snapshot = _snapshot_file(
        barrier_path,
        label="complete prediction barrier",
    )
    lock = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    _require(
        lock["two_commit_freeze"]["implementation_commit_h1"] == expected_h1,
        "H1 lock binding changed",
    )
    barrier = _load_json_bytes(
        barrier_snapshot.payload,
        label="complete prediction barrier",
    )
    _require(
        barrier.get("artifact_sha256") == artifact_sha256(barrier)
        and any(
            row == barrier_case
            for row in barrier.get("ordered_case_seals", [])
            if isinstance(row, Mapping)
        ),
        f"{case_id} evaluator barrier case is not in the sealed barrier",
    )
    prediction_arrays, diagnostic, case_snapshots = _validate_case_inputs(
        case_id,
        case_root,
        barrier_case,
        lock_path,
        h2_commit,
        expected_h1,
    )
    identity = _case_identity(diagnostic)
    centers, updates = _validate_centers_and_updates(
        case_id,
        diagnostic,
        point_count=prediction_arrays["physical_prior_m"].shape[1],
    )
    selected_cameras = list(diagnostic["nested_selected_cameras"]["8"])

    native = native_target_validator(
        adapter_repository,
        lock_path,
        h2_commit,
        barrier_path,
        case_seal_dirs,
        nested_measurement_dirs,
        compatibility_root,
        case_id,
        future_root,
        outcome_root,
        expected_h1=expected_h1,
    )
    _require(
        isinstance(native, ConfirmationNativeOfficialTarget)
        and isinstance(native.evidence, Mapping),
        f"{case_id} native target validator returned another schema",
    )
    _require_unchanged(lock_snapshot, label="H2 confirmation lock")
    _require_unchanged(barrier_snapshot, label="complete prediction barrier")
    target_arrays, target_snapshots, _ = _validate_target_inputs(
        case_id,
        future_root,
        outcome_root,
        native,
        identity=identity,
        selected_cameras=selected_cameras,
        lock_artifact_sha256=lock["artifact_sha256"],
        sealed_frame_zero_m=prediction_arrays["physical_prior_m"][0],
    )
    future_snapshot, outcome_snapshot, target_archive_snapshot = target_snapshots
    expected_lock_binding = {
        "implementation_commit_h1": expected_h1,
        "cohort_lock_commit_h2": h2_commit,
        "cohort_lock_artifact_sha256": lock["artifact_sha256"],
        "cohort_lock_file_sha256": lock_snapshot.sha256,
    }
    _validate_native_evidence(
        native,
        case_id=case_id,
        case_root=case_root,
        barrier_case=barrier_case,
        identity=identity,
        selected_cameras=selected_cameras,
        prediction_arrays=prediction_arrays,
        compatibility_root=compatibility_root,
        lock_binding=expected_lock_binding,
        barrier_snapshot=barrier_snapshot,
        barrier_artifact_sha256=barrier["artifact_sha256"],
        future_manifest_snapshot=future_snapshot,
        outcome_manifest_snapshot=outcome_snapshot,
        target_archive_snapshot=target_archive_snapshot,
    )

    transported = transport_official_target(
        prediction_arrays["physical_prior_m"][0],
        OfficialTarget(
            object_points=target_arrays["target_m"],
            object_visibilities=target_arrays["target_visibility"],
            object_motions_valid=target_arrays["target_validity"],
            provenance={
                "case_id": case_id,
                "target_archive_sha256": target_archive_snapshot.sha256,
            },
        ),
    )
    metric_frames = scored_frames()
    metrics: dict[str, dict[str, float]] = {}
    for arm, role in ARMS_TO_ARRAY_ROLES.items():
        detail = score_deform360_hidden_trajectory(
            prediction_arrays[role],
            transported.object_points,
            transported.object_visibilities,
            transported.object_motions_valid,
            center_ids=centers,
            scored_frames=metric_frames,
        )
        metrics[arm] = {metric: float(detail[metric]) for metric in METRICS}
    frame_zero = np.asarray(
        prediction_arrays["physical_prior_m"][0],
        dtype=np.float64,
    )
    frame_zero_scale_m = float(
        np.linalg.norm(np.max(frame_zero, axis=0) - np.min(frame_zero, axis=0))
    )
    _require(
        np.isfinite(frame_zero_scale_m) and frame_zero_scale_m > 0.0,
        f"{case_id} frame-zero bounding-box diagonal is invalid",
    )

    for snapshot, label in (
        (lock_snapshot, "H2 confirmation lock"),
        (barrier_snapshot, "complete prediction barrier"),
        (case_snapshots[0], f"{case_id} case prediction seal"),
        (case_snapshots[1], f"{case_id} sealed prediction archive"),
        (case_snapshots[2], f"{case_id} sealed target-free diagnostic"),
        (future_snapshot, f"{case_id} authorized future manifest"),
        (outcome_snapshot, f"{case_id} authorized outcome manifest"),
        (target_archive_snapshot, f"{case_id} native official target archive"),
    ):
        _require_unchanged(snapshot, label=label)
    return {
        "case_id": case_id,
        "diagnostic_file_sha256": barrier_case["diagnostic_file_sha256"],
        "diagnostic_artifact_sha256": barrier_case["diagnostic_artifact_sha256"],
        "target_file_sha256": target_archive_snapshot.sha256,
        "target_arrays_sha256": _target_arrays_sha256(target_arrays),
        "frame_zero_scale_m": frame_zero_scale_m,
        "metrics": metrics,
        "updates": updates,
    }


def build_confirmation_case_target_loader(
    adapter_repository: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    barrier_path: str | Path,
    case_seal_dirs: Mapping[str, str | Path],
    nested_measurement_dirs: Mapping[str, str | Path],
    compatibility_root: str | Path,
    authorized_future_case_dirs: Mapping[str, str | Path],
    authorized_outcome_case_dirs: Mapping[str, str | Path],
    *,
    expected_h1: str,
    native_target_validator: NativeTargetValidator = (
        validate_confirmation_native_official_target
    ),
    production_mode: bool = False,
) -> Callable[[str, Path, Mapping[str, Any]], Mapping[str, Any]]:
    """Build a post-barrier evaluator target-loader callback.

    It deliberately does not stat, resolve, or open an authorized future or
    official target path before the returned callback is invoked.  Production
    mode requires the exact clean, source-only H2 direct child and the committed
    native scorer.  The default is deliberately development-only so a direct
    API caller cannot accidentally mint a production confirmation callback.
    """

    _require(_is_sha1(expected_h1), "expected H1 is invalid")
    _require(_is_sha1(h2_commit), "H2 commit is invalid")
    _require(h2_commit != expected_h1, "H2 must differ from H1")
    _require(type(production_mode) is bool, "production mode must be Boolean")
    adapter_source = Path(adapter_repository).absolute()
    lock_source = Path(lock_path).absolute()
    repository_provenance: Mapping[str, Any] | None = None
    if production_mode:
        _require(
            native_target_validator is validate_confirmation_native_official_target,
            "production scoring requires the exact native target validator",
        )
        _require(
            validate_confirmation_h2_loaded_runtime is _EXACT_H2_RUNTIME_VALIDATOR,
            "production H2 runtime validator capability changed",
        )
        repository_provenance = validate_confirmation_h2_loaded_runtime(
            adapter_source,
            lock_source,
            h2_commit,
            expected_h1=expected_h1,
            source_file=__file__,
            source_repository_path=SCORING_SOURCE_REPOSITORY_PATH,
        )
    lock = load_confirmation_cohort_lock(
        lock_source,
        expected_implementation_commit_h1=expected_h1,
    )
    exact_cases = tuple(lock["selected_case_ids"])
    mappings = (
        case_seal_dirs,
        nested_measurement_dirs,
        authorized_future_case_dirs,
        authorized_outcome_case_dirs,
    )
    _require(
        all(
            isinstance(value, Mapping)
            and set(value) == set(exact_cases)
            and len(value) == len(exact_cases)
            for value in mappings
        ),
        "scoring loader requires exact H2 case closure for every path mapping",
    )
    bound_case_dirs = {
        case_id: Path(case_seal_dirs[case_id]).absolute() for case_id in exact_cases
    }
    bound_measurement_dirs = {
        case_id: Path(nested_measurement_dirs[case_id]).absolute()
        for case_id in exact_cases
    }
    bound_future_dirs = {
        case_id: Path(authorized_future_case_dirs[case_id]).absolute()
        for case_id in exact_cases
    }
    bound_outcome_dirs = {
        case_id: Path(authorized_outcome_case_dirs[case_id]).absolute()
        for case_id in exact_cases
    }
    barrier_source = Path(barrier_path).absolute()
    compatibility_source = Path(compatibility_root).absolute()

    def target_loader(
        case_id: str,
        case_seal_dir: Path,
        barrier_case: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _require(case_id in bound_case_dirs, "target-loader case is outside H2")
        _require(
            Path(case_seal_dir).absolute() == bound_case_dirs[case_id],
            f"{case_id} evaluator case directory differs from loader binding",
        )
        return _score_case(
            case_id,
            bound_case_dirs[case_id],
            barrier_case,
            adapter_repository=adapter_source,
            lock_path=lock_source,
            h2_commit=h2_commit,
            barrier_path=barrier_source,
            case_seal_dirs=bound_case_dirs,
            nested_measurement_dirs=bound_measurement_dirs,
            compatibility_root=compatibility_source,
            authorized_future_case_dir=bound_future_dirs[case_id],
            authorized_outcome_case_dir=bound_outcome_dirs[case_id],
            expected_h1=expected_h1,
            native_target_validator=native_target_validator,
        )

    attestation = _build_loader_attestation(
        adapter_repository=adapter_source,
        lock_path=lock_source,
        lock=lock,
        h2_commit=h2_commit,
        expected_h1=expected_h1,
        native_target_validator=native_target_validator,
        target_loader=target_loader,
        repository_provenance=repository_provenance,
    )
    serialized_attestation = _canonical_bytes(attestation)
    _LOADER_ATTESTATIONS[target_loader] = _RegisteredLoader(
        attestation=serialized_attestation,
        loader_reference=weakref.ref(target_loader),
        code=target_loader.__code__,
        globals_identity=id(target_loader.__globals__),
        closure_cell_identities=tuple(
            id(cell) for cell in (target_loader.__closure__ or ())
        ),
        defaults=target_loader.__defaults__,
        keyword_defaults=target_loader.__kwdefaults__,
    )
    setattr(
        target_loader,
        _LOADER_ATTESTATION_ATTRIBUTE,
        json.loads(serialized_attestation.decode("utf-8")),
    )
    return target_loader


__all__ = [
    "ARMS_TO_ARRAY_ROLES",
    "METRICS",
    "NATIVE_VALIDATOR_SOURCE_REPOSITORY_PATH",
    "SCORING_LOADER_ATTESTATION_KIND",
    "SCORING_LOADER_FACTORY_KIND",
    "SCORING_SOURCE_REPOSITORY_PATH",
    "TARGET_ARRAY_ROLES",
    "build_confirmation_case_target_loader",
    "validate_confirmation_case_target_loader_attestation",
]
