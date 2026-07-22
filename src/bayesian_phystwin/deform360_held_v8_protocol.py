"""Two-barrier information protocol for the Deform360 held-v8 study.

Held v8 has two deliberately different future-information capabilities.  A
complete fresh prediction cohort authorizes reconstruction of each official
target.  Reconstruction exposes an independently sealed frame-zero query, but
does not authorize reading a future target.  Only a second complete-cohort
barrier, after every query has been evaluated by the frozen field, authorizes
case-by-case scoring.

Capabilities in this module are Python object identities registered in this
process.  They are bound to one role, case, operation, lock digest, and cohort
barrier digest, are revalidated immediately before use, and are consumed even
when that final revalidation fails.  They are consequently neither serializable
nor reproducible from their audit fields.

This is a new v8 implementation.  It does not import or accept v7 protocol
seals.  The unchanged numerical builders may be explicitly wired to the
signature-compatible seal functions below in a fresh v8 worker process.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Literal, Protocol

from . import deform360_frame_zero_assets as frame_zero_assets
from . import deform360_held_v8_replacement_source as replacement_source
from . import deform360_held_v8_query_artifacts as query_artifacts


PROTOCOL_ID = "deform360-held-online-belief-v8.1"
EXECUTION_ATTEMPT = 4
SCHEMA_VERSION = 1
LOCK_KIND = "Deform360HeldOnlineBeliefLock"
FRAME_ZERO_KIND = "Deform360HeldFrameZeroBundle"
PHYSICAL_SEAL_KIND = "Deform360HeldPhysicalPriorSeal"
PREFIX_AUTHORIZATION_KIND = "Deform360HeldCausalPrefixAuthorization"
ONLINE_SEAL_KIND = "Deform360HeldOnlinePredictionSeal"
CALIBRATION_DECISION_KIND = "Deform360HeldV8CalibrationGateDecision"
POST_WITHDRAWAL_DISCLOSURE_KIND = (
    "Deform360HeldV81PostWithdrawalDevelopmentUseDisclosure"
)

FRAME_COUNT = 76
UPDATE_FRAMES = (19, 38, 57)
TARGET_RECONSTRUCTION_OPERATION = "create-official-target-v1"
FUTURE_SCORE_OPERATION = "read-official-target-for-score-v1"
REPLACEMENT_SOURCE_OPERATION = "acquire-aligned-replacement-source-v1"

V7_WITHDRAWAL_REPORT_FILE_SHA256 = (
    "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3"
)
V7_DISCLOSED_FILE_SPECS: Mapping[str, tuple[int, str]] = {
    "v7_outcome_withdrawal_report": (
        10_295,
        V7_WITHDRAWAL_REPORT_FILE_SHA256,
    ),
    "retired_case_official_target": (
        536_992,
        "850a894f1e1eb447fddbb877ac2fbf38225e97514a1218cc7ea1182212f471a8",
    ),
    "retired_case_online_prediction": (
        994_650,
        "ecae2a595b50c91bf842c3e86eb38559eec0ad43aeeba40da2dd8a9098a31f8d",
    ),
    "retired_case_online_prediction_seal": (
        3_684,
        "afac640547cf4f0de1f168dd4642b841ee96cc274b5e47401aadd4e361255814",
    ),
}
POST_WITHDRAWAL_DEVELOPMENT_HASHES = {
    "scratch_frozen_field_source_sha256": (
        "e106611d9f5e9c6125b5c4c1704db06703108f1ce635d55e6e15d8c8b3a32822"
    ),
    "scratch_query_development_source_sha256": (
        "3f008ef9f9b6fe52c6a36e1939a56ec35e160912efae44ba5a12d11a59a572ae"
    ),
}
OPEN27_DEVELOPMENT_DECISION_FILE_SHA256 = (
    "110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd"
)
ATTEMPT3_ARCHIVE_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/"
    "held-v8-attempt-3-withdrawn-postbarrier"
)
ATTEMPT3_WITHDRAWAL_REPORT_PATH = (
    ATTEMPT3_ARCHIVE_PATH / "execution-withdrawal-postbarrier-attempt3.json"
)
ATTEMPT3_WITHDRAWAL_POINTER_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/"
    "held-v8-attempt-3-withdrawal-pointer.json"
)
ATTEMPT3_WITHDRAWAL_INTEGRITY_COMPLETION_PATH = Path(
    "/mnt/corsair/florianpfaff/bpt-online-belief-v1/"
    "held-v8-attempt-3-withdrawal-integrity-completion.json"
)
ATTEMPT3_WITHDRAWAL_REPORT_FILE_SHA256 = (
    "6d9c62606d18744d275df51fd08e041205bf15b38175d74c69690eafd511054b"
)
ATTEMPT3_WITHDRAWAL_REPORT_ARTIFACT_SHA256 = (
    "4b7404961fa13b418265f76827dda356fb6ad019db764c6302f49e8149d05de2"
)
ATTEMPT3_WITHDRAWAL_COMPLETION_FILE_SHA256 = (
    "f3d1e8a6670484c81ac04743bcdb020cdee3fba02229a64844a8a9c9f4b8b989"
)
ATTEMPT3_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256 = (
    "9ec2989e3000464a0f72b038e26fe407403e02721e21c19ae4fb9123c6a7cf8c"
)
ATTEMPT3_WITHDRAWAL_POINTER_FILE_SHA256 = (
    "75acc7e9535f41528d22739ae8eeb5a0a2247c0fe63c097ad1da2859d7b33246"
)
ATTEMPT3_WITHDRAWAL_POINTER_ARTIFACT_SHA256 = (
    "6ef596a63029d7fa8346141bb52c72d99062e201a12b7c9baf4fca7330baca64"
)
ATTEMPT3_ARCHIVE_INVENTORY_SHA256 = (
    "5d398e998e2b738db545ffefd254712c6822017cfc5be6e7de435d5883c8c4c8"
)
ATTEMPT3_ARCHIVE_ENTRY_COUNT = 1466
ATTEMPT3_ARCHIVE_METADATA_INVENTORY_SHA256 = (
    "f5c35890f2c41b3258ba75ddd66352546d6ac8fb3470704c7369f0cc7970c4ab"
)
_ATTEMPT3_PROTOCOL_ID = "deform360-held-online-belief-v8"
_ATTEMPT3_EXECUTION_ATTEMPT = 3
_ATTEMPT3_STATUS = "withdrawn-postbarrier-before-queried-prediction-or-score"
_ATTEMPT3_DISPOSITION = (
    "WITHDRAWN_AFTER_TARGET_AND_X0_BEFORE_ANY_QUERIED_PREDICTION_SEAL_OR_SCORE"
)
RETIRED_V7_CASE_NAME = "002-rope-silk-ep0003"
FRESH_REPLACEMENT_CASE_NAME = "072-cotton-clohesline-ep0003"

# Object 072 is deliberately outside the five-object dense-v1 panel.  The
# automatic-twin numerics remain those of that frozen panel, but admitting the
# exact replacement calibration case is a new protocol decision and must not
# be mislabeled as a dense-v1 authorization.
REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT = {
    "schema_version": 1,
    "artifact_kind": "Deform360HeldV8ReplacementAutomaticTwinAdmissionContract",
    "protocol_id": "deform360-held-v8-replacement-automatic-twin-admission-v1",
    "held_protocol_id": PROTOCOL_ID,
    "case_name": FRESH_REPLACEMENT_CASE_NAME,
    "object_id": "072-cotton-clohesline",
    "episode_id": 3,
    "role": "calibration",
    "phase": "calibration",
    "source_admission_required": True,
    "prediction_only_input_required": True,
    "target_access": False,
    "inherited_numerical_protocol_id": "deform360-dense-reusable-panel-v1",
    "inherited_numerical_config_sha256": (
        "1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd"
    ),
    "numerical_method_changed": False,
    "admission_scope": "exact-case-only",
}

FROZEN_FIELD_CONTRACT = {
    "operator_id": "gaussian-knn-normalized-v1",
    "field_semantics": "total-displacement-from-frame-zero-v1",
    "neighbor_count": 4,
    "length_scale_fraction": 0.05,
    "support_radius_fraction": 0.50,
    "frame_index_rule": "numpy.arange(76,dtype=int64)",
    "frame_indices": list(range(FRAME_COUNT)),
    "center_exclusion": {
        **deepcopy(query_artifacts.CENTER_EXCLUSION_CONTRACT),
        "contract_sha256": query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
    },
    "open27_development_decision_file_sha256": (
        OPEN27_DEVELOPMENT_DECISION_FILE_SHA256
    ),
    "future_target_used_for_selection": False,
}

PRIMARY_METHOD = {
    "method_id": "shared-frozen-gaussian-query-field-v1",
    "operator_id": FROZEN_FIELD_CONTRACT["operator_id"],
    "neighbor_count": FROZEN_FIELD_CONTRACT["neighbor_count"],
    "length_scale_fraction": FROZEN_FIELD_CONTRACT["length_scale_fraction"],
    "support_radius_fraction": FROZEN_FIELD_CONTRACT["support_radius_fraction"],
    "center_exclusion_contract_sha256": query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
    "calibration_selects_method": False,
}

FRESHNESS_AND_REUSE_CONTRACT = {
    "held_v8_root_absent_before_attempt4_lock": True,
    "all_predictions_must_be_fresh_v8_1_attempt4_outputs": True,
    "all_targets_queries_and_scores_must_be_fresh_v8_1_attempt4_outputs": True,
    "v7_execution_artifacts_reused": False,
    "v7_prediction_artifacts_reused": False,
    "v7_target_or_query_artifacts_reused": False,
    "v8_attempt1_predictions_reused": False,
    "v8_attempt2_predictions_reused": False,
    "v8_attempt1_source_manifests_reused": False,
    "v8_attempt2_source_manifests_reused": False,
    "v8_attempt1_frozen_fields_reused": False,
    "v8_attempt2_frozen_fields_reused": False,
    "v8_attempt1_partial_artifacts_reused": False,
    "v8_attempt2_partial_artifacts_reused": False,
    "v8_attempt3_predictions_reused": False,
    "v8_attempt3_source_manifests_reused": False,
    "v8_attempt3_frozen_fields_reused": False,
    "v8_attempt3_target_artifacts_reused": False,
    "v8_attempt3_official_x0_query_artifacts_reused": False,
    "v8_attempt3_queried_prediction_artifacts_reused": False,
    "v8_attempt3_score_or_gate_artifacts_reused": False,
    "v8_attempt3_partial_artifacts_reused": False,
    "full_15_case_fresh_rerun_required": True,
}

PHYSICAL_ARTIFACT_ROLES = (
    "prediction_only_input",
    "prediction_only_summary",
    "physical_prediction_archive",
    "physical_prediction_manifest",
)
ONLINE_ARTIFACT_ROLES = (
    "measurement_archive",
    "measurement_manifest",
    "measurement_uncertainty_archive",
    "measurement_uncertainty_manifest",
    "cycle_uncertainty_archive",
    "cycle_uncertainty_manifest",
    "online_prediction_archive",
)


@dataclass(frozen=True)
class HeldCaseSpec:
    case_name: str
    object_id: str
    episode_id: int
    remote_inventory_sha256: str
    remote_file_count: int
    remote_total_bytes: int


# Byte-for-byte the v7 confirmation panel.  It remains inaccessible from a
# calibration lock and may only be reached through a validated GO decision.
CONFIRMATION_CASES = (
    HeldCaseSpec(
        "002-rope-silk-ep0001",
        "002-rope-silk",
        1,
        "b33791f6faa8d05717408d7b77cf1405083b614fe42ecef3a3538a0dc2008858",
        32,
        37_863_432,
    ),
    HeldCaseSpec(
        "081-stripe-rope-ep0005",
        "081-stripe-rope",
        5,
        "6055375fb66ea1e0732e808d855e4eecb66687f14dfd6a6a604d5d9a39a194e0",
        32,
        61_222_868,
    ),
    HeldCaseSpec(
        "085-scarf-cloth-ep0002",
        "085-scarf-cloth",
        2,
        "cb9ee9be4c99244e94f676a329b31ecb629c0afef9b7ffbe6060a6b061b81249",
        32,
        31_710_094,
    ),
    HeldCaseSpec(
        "083-blanket-cloth-ep0007",
        "083-blanket-cloth",
        7,
        "102f9edd98b6d3703c3d98625a358c7588d87c79024c795d233771e76b10be84",
        32,
        53_947_570,
    ),
    HeldCaseSpec(
        "092-squirrel-ep0001",
        "092-squirrel",
        1,
        "6f02afc8e8101fdc0e30ee171435162d1d6a4d648f5ee910070f711313d2b960",
        32,
        38_161_504,
    ),
    HeldCaseSpec(
        "170-spider-ep0006",
        "170-spider",
        6,
        "c19cb57b087aa98c5e792e8dfcb2e889cb4b2a52653a78a2cba6591a0fdc80a7",
        32,
        47_269_453,
    ),
)
CONFIRMATION_CASE_NAMES = tuple(case.case_name for case in CONFIRMATION_CASES)

CALIBRATION_CASE_NAMES = (
    FRESH_REPLACEMENT_CASE_NAME,
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
)

_SEALED_FILE_MODE = 0o400
_FILE_RECORD_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_DISCLOSED_FILE_RECORD_FIELDS = frozenset(
    {"path", "sha256", "size_bytes", "mode_octal"}
)
_SHA256_LENGTH = 64
_ROLE_VALUES = frozenset({"calibration", "confirmation"})
_LEGACY_PROTOCOL_ID = "deform360-held-online-belief-v7"
_CAPABILITY_AUTHORITY = object()
_LIVE_CAPABILITIES: dict[int, "_CapabilityState"] = {}
_ISSUED_BARRIERS: set[tuple[str, str, str, str]] = set()
_FRESH_ROOT_CAPABILITIES: dict[int, "_FreshRootState"] = {}
_COMPLETED_RECONSTRUCTION_BARRIERS: dict[tuple[str, str], str] = {}


class ArtifactValidator(Protocol):
    def __call__(
        self,
        path: str | Path,
        lock_path: str | Path,
        *,
        expected_case_name: str,
        expected_role: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CohortBarrierEvidence:
    """Pure result of replaying one complete-cohort barrier."""

    protocol_id: str
    barrier_number: int
    role: str
    operation: str
    lock_path: str
    lock_file_sha256: str
    lock_artifact_sha256: str
    ordered_case_names: tuple[str, ...]
    ordered_artifact_bindings: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    barrier_sha256: str


@dataclass(frozen=True, slots=True)
class _CaseCapability:
    role: str
    case_name: str
    operation: str
    lock_file_sha256: str
    lock_artifact_sha256: str
    cohort_barrier_sha256: str
    predecessor_barrier_sha256: str | None
    _nonce: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __reduce__(self) -> Any:
        raise TypeError("held-v8 capabilities cannot be serialized")


@dataclass
class _CapabilityState:
    capability: _CaseCapability
    revalidate: Callable[[], CohortBarrierEvidence]
    consumed: bool = False
    validated_consumption: bool = False


@dataclass(frozen=True, slots=True)
class _FreshRootCapability:
    held_root: str
    _nonce: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __reduce__(self) -> Any:
        raise TypeError("fresh-root capabilities cannot be serialized")


@dataclass
class _FreshRootState:
    capability: _FreshRootCapability
    consumed: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical JSON") from error


def held_artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(value))
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def held_contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256 = held_contract_sha256(
    REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT
)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _held_root(lock: Mapping[str, Any]) -> Path:
    value = lock.get("held_root")
    _require(isinstance(value, str) and value, "held-v8 lock root is missing")
    return _canonical_path(value)


def _require_current_execution_path(
    path: str | Path,
    *,
    lock: Mapping[str, Any],
    role: str,
    required_parent: str | Path | None = None,
) -> Path:
    """Require a fresh execution path below this lock's positive allowlist."""

    source = _canonical_path(path)
    held_root = _held_root(lock)
    parent = held_root if required_parent is None else _canonical_path(required_parent)
    _require(
        source.resolve() == source,
        f"{role} is not a canonical path in its current held-v8 subtree",
    )
    try:
        parent.relative_to(held_root)
        relative = source.relative_to(parent)
    except ValueError as error:
        raise ValueError(
            f"{role} is outside its exact current held-v8 subtree"
        ) from error
    _require(
        bool(relative.parts),
        f"{role} must be a file below its exact current held-v8 subtree",
    )
    if os.path.lexists(source):
        observed = os.lstat(source)
        if stat.S_ISREG(observed.st_mode):
            _require(
                observed.st_nlink == 1,
                f"{role} must be a single-link fresh current-execution file",
            )
    return source


def _case_root(lock: Mapping[str, Any], *, role: str, case_name: str) -> Path:
    _authorize_case(lock, case_name, role)
    return _held_root(lock) / role / "cases" / case_name


def _case_stage_root(
    lock: Mapping[str, Any],
    *,
    role: str,
    case_name: str,
    stage: str,
) -> Path:
    _require(stage and Path(stage).name == stage, "invalid held-v8 case stage")
    return _case_root(lock, role=role, case_name=case_name) / stage


def _read_regular_file(path: str | Path) -> tuple[Path, bytes, os.stat_result]:
    source = _canonical_path(path)
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"{source} is a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{source} is not a regular file")
    _require(source.resolve() == source, f"{source} has a symlinked ancestor")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{source} changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (opened.st_dev, opened.st_ino)
        _require(
            (after.st_dev, after.st_ino) == identity
            and (current.st_dev, current.st_ino) == identity
            and after.st_size == opened.st_size
            and after.st_mtime_ns == opened.st_mtime_ns
            and after.st_ctime_ns == opened.st_ctime_ns,
            f"{source} changed while reading",
        )
    finally:
        os.close(descriptor)
    return source, payload, after


def _bound_file(path: str | Path) -> dict[str, Any]:
    source, payload, observed = _read_regular_file(path)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": observed.st_size,
    }


def _sha256_file(path: str | Path) -> str:
    return str(_bound_file(path)["sha256"])


def _require_mode(path: str | Path, mode: int, *, role: str) -> None:
    source = _canonical_path(path)
    observed = os.lstat(source)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == mode,
        f"{role} must be a regular non-symlink file with mode {mode:04o}",
    )


def _seal_existing_regular_file(
    path: str | Path,
    *,
    role: str,
    lock: Mapping[str, Any],
    required_parent: str | Path,
) -> dict[str, Any]:
    """Freeze one fresh builder output before it enters a v8 seal."""

    source = _require_current_execution_path(
        path,
        lock=lock,
        role=role,
        required_parent=required_parent,
    )
    before = os.lstat(source)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and source.resolve() == source,
        f"{role} is not a canonical regular file",
    )
    os.chmod(source, _SEALED_FILE_MODE, follow_symlinks=False)
    _require_mode(source, _SEALED_FILE_MODE, role=role)
    return _bound_file(source)


def _validate_bound_file(
    record: object,
    *,
    role: str,
    required_mode: int | None = None,
    current_lock: Mapping[str, Any] | None = None,
    required_parent: str | Path | None = None,
) -> Path:
    _require(
        isinstance(record, Mapping) and set(record) == _FILE_RECORD_FIELDS,
        f"{role} file record fields changed",
    )
    path = record.get("path")
    _require(isinstance(path, str) and path, f"{role} path is missing")
    if current_lock is not None:
        _require_current_execution_path(
            path,
            lock=current_lock,
            role=role,
            required_parent=required_parent,
        )
    else:
        _require(
            required_parent is None,
            f"{role} containment requires a current held-v8 lock",
        )
    observed = _bound_file(path)
    _require(observed == dict(record), f"{role} file binding changed")
    if required_mode is not None:
        _require_mode(path, required_mode, role=role)
    return Path(observed["path"])


def _load_json(path: str | Path) -> dict[str, Any]:
    source, payload, _ = _read_regular_file(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not canonical JSON") from error
    _require(isinstance(value, dict), f"{source} must contain a JSON object")
    return value


def _prepare_parent(destination: Path) -> None:
    parent = destination.parent
    _require(parent.exists(), f"destination parent does not exist: {parent}")
    observed = os.lstat(parent)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and parent.resolve() == parent,
        "destination parent is not a canonical directory",
    )


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = _canonical_path(path)
    _prepare_parent(destination)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _SEALED_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, _SEALED_FILE_MODE, follow_symlinks=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _case_identity(case_name: str, role: str) -> dict[str, Any]:
    _require(role in _ROLE_VALUES, "unsupported cohort role")
    expected = (
        CALIBRATION_CASE_NAMES if role == "calibration" else CONFIRMATION_CASE_NAMES
    )
    _require(case_name in expected, f"case is outside the exact {role} cohort")
    _require(
        case_name
        not in (
            CONFIRMATION_CASE_NAMES if role == "calibration" else CALIBRATION_CASE_NAMES
        ),
        "case has ambiguous cohort authorization",
    )
    object_id, encoded_episode = case_name.rsplit("-ep", maxsplit=1)
    return {
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": int(encoded_episode),
        "role": role,
    }


def _expected_confirmation_payload() -> list[dict[str, Any]]:
    return [asdict(case) for case in CONFIRMATION_CASES]


def _load_exact_attempt3_artifact(
    path: str | Path,
    *,
    expected_path: Path,
    expected_file_sha256: str,
    expected_artifact_sha256: str,
    expected_kind: str,
    expected_status: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _canonical_path(path)
    _require(source == _canonical_path(expected_path), f"{role} path changed")
    _require_mode(source, _SEALED_FILE_MODE, role=role)
    record = _bound_file(source)
    _require(record["sha256"] == expected_file_sha256, f"{role} file hash changed")
    artifact = _load_json(source)
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("artifact_kind") == expected_kind
        and artifact.get("protocol_id") == _ATTEMPT3_PROTOCOL_ID
        and artifact.get("execution_attempt") == _ATTEMPT3_EXECUTION_ATTEMPT
        and artifact.get("status") == expected_status
        and artifact.get("disposition") == _ATTEMPT3_DISPOSITION
        and artifact.get("artifact_sha256") == expected_artifact_sha256
        and held_artifact_sha256(artifact) == expected_artifact_sha256,
        f"{role} artifact identity changed",
    )
    return record, artifact


def _validate_attempt3_archive(path: str | Path) -> Path:
    archive = _canonical_path(path)
    _require(
        archive == _canonical_path(ATTEMPT3_ARCHIVE_PATH),
        "attempt-3 archive path changed",
    )
    root_state = os.lstat(archive)
    _require(
        stat.S_ISDIR(root_state.st_mode)
        and not stat.S_ISLNK(root_state.st_mode)
        and stat.S_IMODE(root_state.st_mode) == 0o500
        and archive.resolve() == archive,
        "attempt-3 archive root is not canonical mode 0500",
    )
    return archive


def _stable_inventory_state(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_uid,
        observed.st_gid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _attempt3_inventory_file_row(path: Path, *, relative: Path) -> dict[str, Any]:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o400,
        f"attempt-3 archive file is not a sealed mode-0400 file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and _stable_inventory_state(opened) == _stable_inventory_state(before),
            f"attempt-3 archive file changed while opening: {path}",
        )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        _stable_inventory_state(before)
        == _stable_inventory_state(after)
        == _stable_inventory_state(current),
        f"attempt-3 archive file changed while hashing: {path}",
    )
    return {
        "path": relative.as_posix(),
        "type": "file",
        "mode_octal": "0400",
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _run_attempt3_git(code: Path, arguments: list[str]) -> bytes:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/florianpfaff",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fileMode=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(code),
            *arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    _require(
        completed.returncode == 0,
        "attempt-3 deployed-code git "
        + " ".join(arguments)
        + " failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    return completed.stdout


def _parse_attempt3_git_tree(raw: bytes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        _require(bool(separator) and bool(path_bytes), "malformed deployed Git tree")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed deployed Git tree header")
        try:
            mode, kind, object_id = (field.decode("ascii") for field in fields)
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("deployed Git tree is not canonical text") from error
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) in {40, 64}
            and all(character in "0123456789abcdef" for character in object_id),
            f"unsupported attempt-3 deployed-code entry: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe attempt-3 deployed-code path",
        )
        rows.append({"mode": mode, "type": kind, "object_id": object_id, "path": path})
    _require(bool(rows), "attempt-3 deployed Git tree is empty")
    _require(
        [row["path"] for row in rows] == sorted(row["path"] for row in rows),
        "attempt-3 deployed Git tree is not sorted",
    )
    return rows


def _attempt3_worktree_blob_oid(path: Path, *, object_id: str) -> str:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and stat.S_IMODE(before.st_mode) == 0o400,
        f"attempt-3 tracked file is not sealed mode 0400: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_inventory_state(opened) == _stable_inventory_state(before),
            f"attempt-3 tracked file changed while opening: {path}",
        )
        algorithm = "sha1" if len(object_id) == 40 else "sha256"
        digest = hashlib.new(algorithm)
        digest.update(f"blob {before.st_size}\0".encode("ascii"))
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        _stable_inventory_state(before)
        == _stable_inventory_state(after)
        == _stable_inventory_state(current),
        f"attempt-3 tracked file changed while hashing: {path}",
    )
    return digest.hexdigest()


def _attempt3_repository_binding(code: Path) -> dict[str, Any]:
    git_directory = code / ".git"
    git_state = os.lstat(git_directory)
    _require(
        stat.S_ISDIR(git_state.st_mode)
        and not stat.S_ISLNK(git_state.st_mode)
        and stat.S_IMODE(git_state.st_mode) == 0o500,
        "attempt-3 deployed-code Git directory changed",
    )
    top = _run_attempt3_git(code, ["rev-parse", "--show-toplevel"])
    _require(
        top.decode("utf-8").strip() == str(code),
        "attempt-3 deployed Git top level changed",
    )
    head = _run_attempt3_git(code, ["rev-parse", "HEAD"]).decode("ascii").strip()
    _require(
        len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        "attempt-3 deployed HEAD is invalid",
    )
    _require(
        _run_attempt3_git(code, ["status", "--porcelain=v1", "--untracked-files=all"])
        == b"",
        "attempt-3 deployed worktree content changed",
    )
    # Deliberately omit every exclude option. Unlike `git status`, this also
    # exposes files matched by .gitignore and repository-local exclude rules.
    _require(
        _run_attempt3_git(code, ["ls-files", "--others", "-z"]) == b"",
        "attempt-3 deployed worktree has untracked or ignored files",
    )
    _require(
        _run_attempt3_git(code, ["rev-parse", "--is-shallow-repository"])
        .decode("ascii")
        .strip()
        == "false",
        "attempt-3 deployed repository is shallow",
    )
    _run_attempt3_git(code, ["fsck", "--full", "--no-dangling"])
    rows = _parse_attempt3_git_tree(
        _run_attempt3_git(code, ["ls-tree", "-r", "-z", "HEAD"])
    )
    tracked_paths = {str(row["path"]) for row in rows}
    tracked_directories = {
        parent.as_posix()
        for path in (Path(relative) for relative in tracked_paths)
        for parent in path.parents
        if parent != Path(".")
    }
    for row in rows:
        path = code / row["path"]
        _require(
            _attempt3_worktree_blob_oid(path, object_id=row["object_id"])
            == row["object_id"],
            f"attempt-3 tracked file content changed: {path}",
        )
    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for current, directories, files in os.walk(code, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_parent = current_path.relative_to(code)
        if relative_parent == Path("."):
            directories[:] = sorted(name for name in directories if name != ".git")
        else:
            directories[:] = sorted(directories)
        for name in directories:
            path = current_path / name
            observed = os.lstat(path)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"attempt-3 deployed worktree directory changed: {path}",
            )
            actual_directories.add(path.relative_to(code).as_posix())
        for name in sorted(files):
            path = current_path / name
            observed = os.lstat(path)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o400,
                f"attempt-3 deployed worktree file changed: {path}",
            )
            actual_paths.add(path.relative_to(code).as_posix())
    _require(
        actual_paths == tracked_paths and actual_directories == tracked_directories,
        "attempt-3 deployed worktree path set changed",
    )
    return {
        "path": code.name,
        "git_head": head,
        "head_text_sha256": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "git_tree_record_count": len(rows),
        "git_tree_manifest_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _attempt3_deployed_code_directory(archive: Path, report: Mapping[str, Any]) -> Path:
    deployed = report.get("deployed_code")
    _require(isinstance(deployed, Mapping), "attempt-3 deployed-code binding is absent")
    name = deployed.get("path")
    head = deployed.get("git_head")
    _require(
        isinstance(name, str)
        and isinstance(head, str)
        and len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head)
        and name == f"code-{head}"
        and Path(name).name == name,
        "attempt-3 deployed-code identity changed",
    )
    code = archive / name
    observed = os.lstat(code)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o500
        and code.resolve() == code,
        "attempt-3 deployed-code directory changed",
    )
    candidates = []
    for child in archive.iterdir():
        if not child.name.startswith("code-"):
            continue
        suffix = child.name.removeprefix("code-")
        if len(suffix) in {40, 64} and all(
            character in "0123456789abcdef" for character in suffix
        ):
            candidates.append(child)
    _require(candidates == [code], "attempt-3 deployed-code directory is not unique")
    _require(
        deployed.get("head_text_sha256")
        == hashlib.sha256(head.encode("ascii")).hexdigest(),
        "attempt-3 deployed-code HEAD binding changed",
    )
    git_directory = code / ".git"
    git_state = os.lstat(git_directory)
    _require(
        stat.S_ISDIR(git_state.st_mode)
        and not stat.S_ISLNK(git_state.st_mode)
        and stat.S_IMODE(git_state.st_mode) == 0o500,
        "attempt-3 deployed-code Git directory changed",
    )
    head_path = git_directory / "HEAD"
    _require_mode(head_path, 0o400, role="attempt-3 deployed-code HEAD")
    _, head_payload, _ = _read_regular_file(head_path)
    try:
        observed_head = head_payload.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("attempt-3 deployed-code HEAD is not ASCII") from error
    _require(observed_head == head, "attempt-3 deployed-code checkout changed")
    _require(
        dict(deployed) == _attempt3_repository_binding(code),
        "attempt-3 deployed-code repository binding changed",
    )
    return code


def _observed_attempt3_noncode_inventory(
    archive: Path, *, deployed_code: Path
) -> dict[str, Any]:
    directory_states: dict[Path, tuple[int, ...]] = {}
    rows: list[dict[str, Any]] = []
    report_relative = Path(ATTEMPT3_WITHDRAWAL_REPORT_PATH.name)
    for current, directories, files in os.walk(
        archive, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_state = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_state.st_mode)
            and not stat.S_ISLNK(current_state.st_mode)
            and stat.S_IMODE(current_state.st_mode) == 0o500,
            f"attempt-3 archive directory is not sealed mode 0500: {current_path}",
        )
        directory_states[current_path] = _stable_inventory_state(current_state)
        relative_parent = current_path.relative_to(archive)
        directories[:] = sorted(
            name
            for name in directories
            if not (
                relative_parent == Path(".") and current_path / name == deployed_code
            )
        )
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"attempt-3 archive directory is not sealed mode 0500: {child}",
            )
            directory_states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": child.relative_to(archive).as_posix(),
                    "type": "directory",
                    "mode_octal": "0500",
                }
            )
        for name in sorted(files):
            child = current_path / name
            relative = child.relative_to(archive)
            if relative == report_relative:
                continue
            rows.append(_attempt3_inventory_file_row(child, relative=relative))
    for path, expected in directory_states.items():
        _require(
            _stable_inventory_state(os.lstat(path)) == expected,
            f"attempt-3 archive directory changed while hashing: {path}",
        )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len({str(row["path"]) for row in rows}) == len(rows),
        "attempt-3 archive inventory has a duplicate path",
    )
    return {
        "entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
    }


def _observed_attempt3_noncode_metadata_inventory(
    archive: Path, *, deployed_code: Path
) -> dict[str, Any]:
    states: dict[Path, tuple[int, ...]] = {}
    rows: list[dict[str, Any]] = []
    report_relative = Path(ATTEMPT3_WITHDRAWAL_REPORT_PATH.name)
    for current, directories, files in os.walk(
        archive, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_state = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_state.st_mode)
            and not stat.S_ISLNK(current_state.st_mode)
            and stat.S_IMODE(current_state.st_mode) == 0o500,
            f"attempt-3 archive directory is not sealed mode 0500: {current_path}",
        )
        states[current_path] = _stable_inventory_state(current_state)
        relative_parent = current_path.relative_to(archive)
        directories[:] = sorted(
            name
            for name in directories
            if not (
                relative_parent == Path(".") and current_path / name == deployed_code
            )
        )
        for name in directories:
            child = current_path / name
            observed = os.lstat(child)
            _require(
                stat.S_ISDIR(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o500,
                f"attempt-3 archive directory is not sealed mode 0500: {child}",
            )
            states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": child.relative_to(archive).as_posix(),
                    "type": "directory",
                    "mode_octal": "0500",
                    "mtime_ns": observed.st_mtime_ns,
                    "ctime_ns": observed.st_ctime_ns,
                }
            )
        for name in sorted(files):
            child = current_path / name
            relative = child.relative_to(archive)
            if relative == report_relative:
                continue
            observed = os.lstat(child)
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and stat.S_IMODE(observed.st_mode) == 0o400,
                f"attempt-3 archive file is not a sealed mode-0400 file: {child}",
            )
            states[child] = _stable_inventory_state(observed)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "type": "file",
                    "mode_octal": "0400",
                    "size_bytes": observed.st_size,
                    "mtime_ns": observed.st_mtime_ns,
                    "ctime_ns": observed.st_ctime_ns,
                }
            )
    for path, expected in states.items():
        _require(
            _stable_inventory_state(os.lstat(path)) == expected,
            f"attempt-3 archive entry changed while scanning metadata: {path}",
        )
    rows.sort(key=lambda row: str(row["path"]))
    _require(
        len({str(row["path"]) for row in rows}) == len(rows),
        "attempt-3 archive metadata inventory has a duplicate path",
    )
    return {
        "entry_count": len(rows),
        "metadata_inventory_sha256": hashlib.sha256(
            _canonical_bytes({"rows": rows})
        ).hexdigest(),
    }


def validate_attempt3_withdrawal_lineage(
    *,
    archive_path: str | Path,
    report_path: str | Path,
    pointer_path: str | Path,
    completion_path: str | Path,
    verify_content_inventory: bool = False,
) -> dict[str, Any]:
    """Validate the exact immutable attempt-3 post-barrier withdrawal chain."""

    archive = _validate_attempt3_archive(archive_path)
    report_record, report = _load_exact_attempt3_artifact(
        report_path,
        expected_path=ATTEMPT3_WITHDRAWAL_REPORT_PATH,
        expected_file_sha256=ATTEMPT3_WITHDRAWAL_REPORT_FILE_SHA256,
        expected_artifact_sha256=ATTEMPT3_WITHDRAWAL_REPORT_ARTIFACT_SHA256,
        expected_kind="Deform360HeldV8Attempt3PostBarrierWithdrawalReport",
        expected_status=_ATTEMPT3_STATUS,
        role="attempt-3 withdrawal report",
    )
    deployed_code = _attempt3_deployed_code_directory(archive, report)
    observed_metadata = _observed_attempt3_noncode_metadata_inventory(
        archive, deployed_code=deployed_code
    )
    _require(
        observed_metadata
        == {
            "entry_count": ATTEMPT3_ARCHIVE_ENTRY_COUNT,
            "metadata_inventory_sha256": (ATTEMPT3_ARCHIVE_METADATA_INVENTORY_SHA256),
        },
        "attempt-3 archive metadata inventory changed",
    )
    expected_inventory = {
        "entry_count": ATTEMPT3_ARCHIVE_ENTRY_COUNT,
        "inventory_sha256": ATTEMPT3_ARCHIVE_INVENTORY_SHA256,
    }
    if verify_content_inventory:
        observed_inventory = _observed_attempt3_noncode_inventory(
            archive, deployed_code=deployed_code
        )
        _require(
            observed_inventory == expected_inventory,
            "attempt-3 archive content inventory changed",
        )
    else:
        observed_inventory = expected_inventory
    completion_record, completion = _load_exact_attempt3_artifact(
        completion_path,
        expected_path=ATTEMPT3_WITHDRAWAL_INTEGRITY_COMPLETION_PATH,
        expected_file_sha256=ATTEMPT3_WITHDRAWAL_COMPLETION_FILE_SHA256,
        expected_artifact_sha256=ATTEMPT3_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256,
        expected_kind="Deform360HeldV8Attempt3WithdrawalIntegrityCompletion",
        expected_status="withdrawal-integrity-complete",
        role="attempt-3 withdrawal integrity completion",
    )
    pointer_record, pointer = _load_exact_attempt3_artifact(
        pointer_path,
        expected_path=ATTEMPT3_WITHDRAWAL_POINTER_PATH,
        expected_file_sha256=ATTEMPT3_WITHDRAWAL_POINTER_FILE_SHA256,
        expected_artifact_sha256=ATTEMPT3_WITHDRAWAL_POINTER_ARTIFACT_SHA256,
        expected_kind="Deform360HeldV8Attempt3WithdrawalPointer",
        expected_status=_ATTEMPT3_STATUS,
        role="attempt-3 withdrawal pointer",
    )

    _require(
        _canonical_path(report["immutable_archive_path"]) == archive
        and _canonical_path(completion["archive_path"]) == archive
        and _canonical_path(pointer["archive_path"]) == archive,
        "attempt-3 archive cross-link changed",
    )
    report_link = {
        "path": str(_canonical_path(report_path)),
        "size_bytes": report_record["size_bytes"],
        "file_sha256": report_record["sha256"],
        "artifact_sha256": ATTEMPT3_WITHDRAWAL_REPORT_ARTIFACT_SHA256,
    }
    report_cross_link = {
        "withdrawal_report_size_bytes": report_link["size_bytes"],
        "withdrawal_report_file_sha256": report_link["file_sha256"],
        "withdrawal_report_artifact_sha256": report_link["artifact_sha256"],
    }
    for artifact, role in ((completion, "completion"), (pointer, "pointer")):
        _require(
            all(artifact.get(key) == value for key, value in report_cross_link.items())
            and _canonical_path(artifact.get("withdrawal_report_path"))
            == _canonical_path(report_link["path"]),
            f"attempt-3 {role} report cross-link changed",
        )
    expected_completion_link = {
        "path": str(_canonical_path(completion_path)),
        "mode_octal": "0400",
        "size_bytes": completion_record["size_bytes"],
        "file_sha256": completion_record["sha256"],
        "artifact_sha256": ATTEMPT3_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256,
    }
    _require(
        pointer.get("withdrawal_integrity_completion") == expected_completion_link,
        "attempt-3 pointer completion cross-link changed",
    )
    _require(
        completion.get("pointer_contract")
        == {
            "path": str(_canonical_path(pointer_path)),
            "artifact_kind": "Deform360HeldV8Attempt3WithdrawalPointer",
            "pointer_must_bind_this_completion": True,
            "completion_does_not_predict_pointer_hash_to_avoid_circularity": True,
        },
        "attempt-3 completion pointer contract changed",
    )
    shared = {
        "archive_root_mode_octal": "0500",
        "archive_fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": ATTEMPT3_ARCHIVE_INVENTORY_SHA256,
        "postseal_noncode_entry_count": ATTEMPT3_ARCHIVE_ENTRY_COUNT,
        "independent_post_rename_integrity_verified": True,
    }
    _require(
        all(completion.get(key) == value for key, value in shared.items())
        and all(pointer.get(key) == value for key, value in shared.items()),
        "attempt-3 archive integrity evidence changed",
    )
    _require(
        report.get("executed_withdrawal_operator_source")
        == completion.get("executed_withdrawal_operator_source")
        == pointer.get("executed_withdrawal_operator_source")
        and report.get("deployed_code")
        == completion.get("deployed_code")
        == pointer.get("deployed_code"),
        "attempt-3 operator or deployed-code lineage changed",
    )
    execution = report.get("execution_boundary")
    information = report.get("information_boundary")
    _require(
        isinstance(execution, Mapping)
        and execution.get("online_prediction_seal_count") == 15
        and execution.get("frozen_field_manifest_count") == 15
        and execution.get("official_target_archive_count") == 1
        and execution.get("official_x0_archive_count") == 1
        and execution.get("queried_prediction_seal_count") == 0
        and execution.get("score_evidence_count") == 0
        and execution.get("gate_decision_count") == 0
        and execution.get("confirmation_lock_count") == 0
        and isinstance(information, Mapping)
        and information.get("first_complete_cohort_barrier_crossed") is True
        and information.get("queried_prediction_created_or_read") is False
        and information.get("score_created_or_read") is False
        and information.get("gate_decision_created_or_read") is False
        and information.get("confirmation_created_or_read") is False,
        "attempt-3 execution or information boundary changed",
    )
    _require(
        pointer.get("active_held_v8_root_absent_after_archive") is True
        and pointer.get("queried_prediction_seal_count") == 0
        and pointer.get("score_evidence_count") == 0
        and pointer.get("gate_decision_count") == 0
        and pointer.get("confirmation_accessed") is False,
        "attempt-3 pointer outcome boundary changed",
    )
    archive_integrity = {
        "path": str(archive),
        "root_mode_octal": "0500",
        "fully_nonwritable": True,
        "postseal_noncode_inventory_sha256": observed_inventory["inventory_sha256"],
        "postseal_noncode_entry_count": observed_inventory["entry_count"],
    }
    return {
        "v8_attempt3_withdrawal_report": report_record,
        "v8_attempt3_withdrawal_pointer": pointer_record,
        "v8_attempt3_withdrawal_integrity_completion": completion_record,
        "v8_attempt3_archive_integrity": archive_integrity,
    }


def validate_post_withdrawal_development_use_disclosure(
    path: str | Path,
) -> dict[str, Any]:
    """Validate the exact report emitted by the committed disclosure sealer."""

    _require_mode(path, _SEALED_FILE_MODE, role="post-withdrawal disclosure")
    artifact = _load_json(path)
    _require(
        set(artifact)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "disclosed_v7_files",
            "disclosed_v8_attempt3_files",
            "v8_attempt3_archive_integrity",
            "v8_attempt3_revision_basis",
            "post_withdrawal_development",
            "retirement",
            "v8_1_reuse_boundary",
            "claim_boundary",
            "artifact_sha256",
        },
        "post-withdrawal disclosure fields changed",
    )
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == POST_WITHDRAWAL_DISCLOSURE_KIND
        and artifact.get("protocol_id") == PROTOCOL_ID,
        "post-withdrawal disclosure identity changed",
    )
    disclosed = artifact.get("disclosed_v7_files")
    _require(
        isinstance(disclosed, Mapping)
        and set(disclosed) == set(V7_DISCLOSED_FILE_SPECS),
        "disclosed v7 file set changed",
    )
    for name, (expected_size, expected_sha256) in V7_DISCLOSED_FILE_SPECS.items():
        record = disclosed[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == _DISCLOSED_FILE_RECORD_FIELDS
            and record.get("mode_octal") == "0400",
            f"{name} disclosure record changed",
        )
        path_value = record.get("path")
        _require(isinstance(path_value, str) and path_value, f"{name} path is missing")
        _require_mode(path_value, _SEALED_FILE_MODE, role=name)
        observed = _bound_file(path_value)
        _require(
            observed
            == {
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            and record["sha256"] == expected_sha256
            and record["size_bytes"] == expected_size,
            f"{name} binding changed",
        )
    attempt3_disclosed = artifact.get("disclosed_v8_attempt3_files")
    expected_attempt3 = {
        "v8_attempt3_withdrawal_report": (
            ATTEMPT3_WITHDRAWAL_REPORT_FILE_SHA256,
            ATTEMPT3_WITHDRAWAL_REPORT_ARTIFACT_SHA256,
        ),
        "v8_attempt3_withdrawal_pointer": (
            ATTEMPT3_WITHDRAWAL_POINTER_FILE_SHA256,
            ATTEMPT3_WITHDRAWAL_POINTER_ARTIFACT_SHA256,
        ),
        "v8_attempt3_withdrawal_integrity_completion": (
            ATTEMPT3_WITHDRAWAL_COMPLETION_FILE_SHA256,
            ATTEMPT3_WITHDRAWAL_COMPLETION_ARTIFACT_SHA256,
        ),
    }
    _require(
        isinstance(attempt3_disclosed, Mapping)
        and set(attempt3_disclosed) == set(expected_attempt3),
        "disclosed attempt-3 file set changed",
    )
    for name, (
        expected_file_sha256,
        _expected_artifact_sha256,
    ) in expected_attempt3.items():
        record = attempt3_disclosed[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == _DISCLOSED_FILE_RECORD_FIELDS
            and record.get("mode_octal") == "0400",
            f"{name} disclosure record changed",
        )
        path_value = record.get("path")
        _require(isinstance(path_value, str) and path_value, f"{name} path is missing")
        _require_mode(path_value, _SEALED_FILE_MODE, role=name)
        observed = _bound_file(path_value)
        _require(
            observed
            == {
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            and record["sha256"] == expected_file_sha256,
            f"{name} binding changed",
        )
    attempt3_lineage = validate_attempt3_withdrawal_lineage(
        archive_path=ATTEMPT3_ARCHIVE_PATH,
        report_path=attempt3_disclosed["v8_attempt3_withdrawal_report"]["path"],
        pointer_path=attempt3_disclosed["v8_attempt3_withdrawal_pointer"]["path"],
        completion_path=attempt3_disclosed[
            "v8_attempt3_withdrawal_integrity_completion"
        ]["path"],
    )
    _require(
        artifact.get("v8_attempt3_archive_integrity")
        == attempt3_lineage["v8_attempt3_archive_integrity"],
        "disclosed attempt-3 archive integrity changed",
    )
    _require(
        artifact.get("v8_attempt3_revision_basis")
        == {
            "official_x0_geometry_used_to_diagnose_exclusion_liveness": True,
            "future_target_coordinates_masks_or_scores_used_for_revision": False,
            "queried_prediction_score_or_gate_existed": False,
            "revision": (
                "replace exact-one-per-center matching with the inclusive 15 mm "
                "x0-only radius union"
            ),
        },
        "attempt-3 revision basis disclosure changed",
    )
    _require(
        artifact.get("post_withdrawal_development")
        == {
            **POST_WITHDRAWAL_DEVELOPMENT_HASHES,
            "retired_official_target_opened_by_development_process": True,
            "retired_online_prediction_opened_by_development_process": True,
            "future_coordinates_or_masks_may_have_been_read": True,
            "derived_metrics_may_have_been_computed": True,
            "field_hypothesis_was_subsequently_reselected_on_independent_open27": True,
        },
        "post-withdrawal development disclosure changed",
    )
    _require(
        artifact.get("retirement")
        == {
            "exact_episode": RETIRED_V7_CASE_NAME,
            "replacement_episode": FRESH_REPLACEMENT_CASE_NAME,
            "replacement_search_excluded_entire_002_rope_silk_object": True,
            "reason": (
                "the exact held-v7 episode was exposed after formal withdrawal; "
                "the replacement was selected outside that object's episodes"
            ),
        },
        "post-withdrawal retirement changed",
    )
    _require(
        artifact.get("v8_1_reuse_boundary")
        == {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "v8_attempt3_predictions_reused": False,
            "v8_attempt3_source_manifests_reused": False,
            "v8_attempt3_frozen_fields_reused": False,
            "v8_attempt3_target_artifacts_reused": False,
            "v8_attempt3_official_x0_query_artifacts_reused": False,
            "v8_attempt3_queried_prediction_artifacts_reused": False,
            "v8_attempt3_score_or_gate_artifacts_reused": False,
            "v8_attempt3_partial_artifacts_reused": False,
            "all_v8_1_attempt4_predictions_targets_queries_and_scores_fresh": True,
            "full_15_case_fresh_rerun_required": True,
        },
        "v8.1 reuse boundary changed",
    )
    _require(
        artifact.get("claim_boundary")
        == (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8.1 into an official Deform360 "
            "state-of-the-art comparison."
        ),
        "disclosure claim boundary changed",
    )
    _require(
        artifact.get("artifact_sha256") == held_artifact_sha256(artifact),
        "post-withdrawal disclosure checksum changed",
    )
    return artifact


def prepare_fresh_held_root(held_root: str | Path) -> object:
    """Create an absent held-v8 root and return a process-local one-use proof."""

    root = _canonical_path(held_root)
    _require(not os.path.lexists(root), "held-v8 root must be absent before prepare")
    parent = root.parent
    _require(parent.exists() and parent.resolve() == parent, "held root parent changed")
    root.mkdir(mode=0o700)
    capability = _FreshRootCapability(
        held_root=str(root),
        _nonce=object(),
        _authority=_CAPABILITY_AUTHORITY,
    )
    _FRESH_ROOT_CAPABILITIES[id(capability)] = _FreshRootState(capability)
    return capability


def _consume_fresh_root_capability(capability: object, root: Path) -> None:
    _require(
        isinstance(capability, _FreshRootCapability),
        "calibration lock lacks a fresh-root capability",
    )
    state = _FRESH_ROOT_CAPABILITIES.get(id(capability))
    _require(
        state is not None
        and state.capability is capability
        and capability._authority is _CAPABILITY_AUTHORITY,
        "fresh-root capability is not live in this process",
    )
    _require(not state.consumed, "fresh-root capability was already consumed")
    state.consumed = True
    _require(
        capability.held_root == str(root),
        "fresh-root capability binds another root",
    )


def create_calibration_protocol_lock(
    output_path: str | Path,
    *,
    held_root: str | Path,
    fresh_root_capability: object,
    immutable_bindings: Mapping[str, str],
    v7_withdrawal_report_path: str | Path,
    post_withdrawal_disclosure_path: str | Path,
    development_decision_path: str | Path,
    attempt3_withdrawal_report_path: str | Path,
    attempt3_withdrawal_pointer_path: str | Path,
    attempt3_withdrawal_integrity_completion_path: str | Path,
) -> dict[str, Any]:
    """Create v8.1 attempt 4 only after proving its formal root was absent."""

    root = _canonical_path(held_root)
    output = _canonical_path(output_path)
    _require(output == root / "calibration-lock.json", "non-canonical lock path")
    _consume_fresh_root_capability(fresh_root_capability, root)
    _require(root.is_dir() and root.resolve() == root, "prepared held root changed")
    disclosure_expected = root / "post-withdrawal-development-use-disclosure.json"
    _require(
        _canonical_path(post_withdrawal_disclosure_path) == disclosure_expected,
        "disclosure is outside the freshly prepared held-v8 root",
    )
    _require(
        set(root.iterdir()) == {disclosure_expected},
        "fresh held-v8 root contains an artifact other than the disclosure",
    )
    try:
        bindings = {str(key): str(value) for key, value in immutable_bindings.items()}
        _require(
            bool(bindings)
            and all(key and _valid_sha256(value) for key, value in bindings.items()),
            "immutable bindings must be named SHA-256 values",
        )
        _require(
            bindings.get("replacement_automatic_twin_admission_contract")
            == REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256,
            "replacement automatic-twin admission contract is not locked",
        )
        _require(
            bindings.get("frame_zero_exact_eight_subset_bounded_audit_contract")
            == frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256,
            "frame-zero bounded subset-audit contract is not locked",
        )
        _require(
            bindings.get("center_exclusion_contract")
            == query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
            "center-exclusion contract is not independently locked",
        )
        _require_mode(
            v7_withdrawal_report_path,
            _SEALED_FILE_MODE,
            role="v7 withdrawal report",
        )
        withdrawal = _bound_file(v7_withdrawal_report_path)
        _require(
            withdrawal["sha256"] == V7_WITHDRAWAL_REPORT_FILE_SHA256,
            "v7 withdrawal report SHA-256 changed",
        )
        disclosure = validate_post_withdrawal_development_use_disclosure(
            post_withdrawal_disclosure_path
        )
        disclosed_withdrawal = disclosure["disclosed_v7_files"][
            "v7_outcome_withdrawal_report"
        ]
        _require(
            {key: disclosed_withdrawal[key] for key in ("path", "sha256", "size_bytes")}
            == withdrawal,
            "disclosure binds another v7 withdrawal report",
        )
        _require_mode(
            development_decision_path,
            _SEALED_FILE_MODE,
            role="open27 development decision",
        )
        development = _bound_file(development_decision_path)
        _require(
            development["sha256"] == OPEN27_DEVELOPMENT_DECISION_FILE_SHA256,
            "open27 development decision SHA-256 changed",
        )
        attempt3_lineage = validate_attempt3_withdrawal_lineage(
            archive_path=ATTEMPT3_ARCHIVE_PATH,
            report_path=attempt3_withdrawal_report_path,
            pointer_path=attempt3_withdrawal_pointer_path,
            completion_path=attempt3_withdrawal_integrity_completion_path,
            verify_content_inventory=True,
        )
        disclosed_attempt3 = disclosure["disclosed_v8_attempt3_files"]
        for name in (
            "v8_attempt3_withdrawal_report",
            "v8_attempt3_withdrawal_pointer",
            "v8_attempt3_withdrawal_integrity_completion",
        ):
            disclosed = disclosed_attempt3[name]
            _require(
                {key: disclosed[key] for key in ("path", "sha256", "size_bytes")}
                == attempt3_lineage[name],
                f"disclosure binds another {name}",
            )
        _require(
            disclosure["v8_attempt3_archive_integrity"]
            == attempt3_lineage["v8_attempt3_archive_integrity"],
            "disclosure binds another attempt-3 archive",
        )
        artifact: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": LOCK_KIND,
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": EXECUTION_ATTEMPT,
            "held_root": str(root),
            "cohort": _expected_confirmation_payload(),
            "case_whitelist": list(CONFIRMATION_CASE_NAMES),
            "calibration_case_whitelist": list(CALIBRATION_CASE_NAMES),
            "frame_count": FRAME_COUNT,
            "update_frames": list(UPDATE_FRAMES),
            "immutable_bindings": dict(sorted(bindings.items())),
            "lineage": {
                "v7_withdrawal_report": withdrawal,
                "post_withdrawal_development_use_disclosure": _bound_file(
                    post_withdrawal_disclosure_path
                ),
                "open27_development_decision": development,
                **attempt3_lineage,
            },
            "frozen_field_contract": deepcopy(FROZEN_FIELD_CONTRACT),
            "replacement_source_inventory_contract": deepcopy(
                replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
            ),
            "primary_method": deepcopy(PRIMARY_METHOD),
            "stage": "calibration",
            "confirmation_access_authorized": False,
            "parent_calibration_lock": None,
            "calibration_gate_evidence": None,
            "freshness_and_reuse": deepcopy(FRESHNESS_AND_REUSE_CONTRACT),
            "information_boundary": {
                "filesystem_case_discovery_permitted": False,
                "barrier_one_requires_physical_online_and_field_for_every_case": True,
                "target_reconstruction_before_barrier_one_permitted": False,
                "future_target_read_before_barrier_two_permitted": False,
                "confirmation_before_calibration_go_permitted": False,
            },
        }
        artifact["artifact_sha256"] = held_artifact_sha256(artifact)
        _write_new_json(output, artifact)
    except BaseException:
        if output.exists():
            os.chmod(output, 0o600, follow_symlinks=False)
            output.unlink()
        raise
    return validate_protocol_lock(output)


def validate_protocol_lock(path: str | Path) -> dict[str, Any]:
    _require_mode(path, _SEALED_FILE_MODE, role="held-v8 lock")
    artifact = _load_json(path)
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == LOCK_KIND
        and artifact.get("protocol_id") == PROTOCOL_ID
        and artifact.get("execution_attempt") == EXECUTION_ATTEMPT,
        "unsupported held-v8 lock",
    )
    root = _canonical_path(str(artifact.get("held_root")))
    _require(
        _canonical_path(path) == root / "calibration-lock.json"
        or (_canonical_path(path) == root / "confirmation-lock.json"),
        "lock is outside its bound held-v8 root",
    )
    _require(
        artifact.get("cohort") == _expected_confirmation_payload()
        and artifact.get("case_whitelist") == list(CONFIRMATION_CASE_NAMES)
        and artifact.get("calibration_case_whitelist") == list(CALIBRATION_CASE_NAMES),
        "held-v8 cohort changed",
    )
    _require(
        RETIRED_V7_CASE_NAME not in artifact["calibration_case_whitelist"]
        and artifact["calibration_case_whitelist"].count(FRESH_REPLACEMENT_CASE_NAME)
        == 1
        and len(artifact["calibration_case_whitelist"]) == 15,
        "held-v8 replacement is not exact",
    )
    _require(
        artifact.get("frame_count") == FRAME_COUNT
        and artifact.get("update_frames") == list(UPDATE_FRAMES)
        and artifact.get("frozen_field_contract") == FROZEN_FIELD_CONTRACT
        and artifact.get("replacement_source_inventory_contract")
        == replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
        and artifact.get("primary_method") == PRIMARY_METHOD,
        "held-v8 method or temporal contract changed",
    )
    bindings = artifact.get("immutable_bindings")
    _require(
        isinstance(bindings, Mapping)
        and bool(bindings)
        and all(
            isinstance(key, str) and key and _valid_sha256(value)
            for key, value in bindings.items()
        ),
        "held-v8 immutable bindings changed",
    )
    _require(
        bindings.get("replacement_automatic_twin_admission_contract")
        == REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256,
        "replacement automatic-twin admission contract changed",
    )
    _require(
        bindings.get("frame_zero_exact_eight_subset_bounded_audit_contract")
        == frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256,
        "frame-zero bounded subset-audit contract changed",
    )
    _require(
        bindings.get("center_exclusion_contract")
        == query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256,
        "center-exclusion contract changed",
    )
    lineage = artifact.get("lineage")
    _require(
        isinstance(lineage, Mapping)
        and set(lineage)
        == {
            "v7_withdrawal_report",
            "post_withdrawal_development_use_disclosure",
            "open27_development_decision",
            "v8_attempt3_withdrawal_report",
            "v8_attempt3_withdrawal_pointer",
            "v8_attempt3_withdrawal_integrity_completion",
            "v8_attempt3_archive_integrity",
        },
        "held-v8 lineage fields changed",
    )
    withdrawal_path = _validate_bound_file(
        lineage["v7_withdrawal_report"],
        role="v7 withdrawal report",
        required_mode=_SEALED_FILE_MODE,
    )
    _require(
        _sha256_file(withdrawal_path) == V7_WITHDRAWAL_REPORT_FILE_SHA256,
        "v7 withdrawal lineage changed",
    )
    disclosure_path = _validate_bound_file(
        lineage["post_withdrawal_development_use_disclosure"],
        role="post-withdrawal disclosure",
        required_mode=_SEALED_FILE_MODE,
    )
    disclosure = validate_post_withdrawal_development_use_disclosure(disclosure_path)
    disclosed_withdrawal = disclosure["disclosed_v7_files"][
        "v7_outcome_withdrawal_report"
    ]
    _require(
        {key: disclosed_withdrawal[key] for key in ("path", "sha256", "size_bytes")}
        == lineage["v7_withdrawal_report"],
        "post-withdrawal disclosure lineage changed",
    )
    development = _validate_bound_file(
        lineage["open27_development_decision"],
        role="open27 development decision",
        required_mode=_SEALED_FILE_MODE,
    )
    _require(
        _sha256_file(development) == OPEN27_DEVELOPMENT_DECISION_FILE_SHA256,
        "open27 development decision lineage changed",
    )
    attempt3_lineage = validate_attempt3_withdrawal_lineage(
        archive_path=lineage["v8_attempt3_archive_integrity"]["path"],
        report_path=lineage["v8_attempt3_withdrawal_report"]["path"],
        pointer_path=lineage["v8_attempt3_withdrawal_pointer"]["path"],
        completion_path=lineage["v8_attempt3_withdrawal_integrity_completion"]["path"],
    )
    _require(
        all(lineage[name] == value for name, value in attempt3_lineage.items()),
        "attempt-3 withdrawal lineage changed",
    )
    disclosed_attempt3 = disclosure["disclosed_v8_attempt3_files"]
    for name in (
        "v8_attempt3_withdrawal_report",
        "v8_attempt3_withdrawal_pointer",
        "v8_attempt3_withdrawal_integrity_completion",
    ):
        disclosed = disclosed_attempt3[name]
        _require(
            {key: disclosed[key] for key in ("path", "sha256", "size_bytes")}
            == lineage[name],
            f"post-withdrawal disclosure {name} lineage changed",
        )
    _require(
        disclosure["v8_attempt3_archive_integrity"]
        == lineage["v8_attempt3_archive_integrity"],
        "post-withdrawal disclosure archive lineage changed",
    )
    _require(
        artifact.get("freshness_and_reuse") == FRESHNESS_AND_REUSE_CONTRACT,
        "held-v8 freshness or reuse contract changed",
    )
    _require(
        artifact.get("information_boundary")
        == {
            "filesystem_case_discovery_permitted": False,
            "barrier_one_requires_physical_online_and_field_for_every_case": True,
            "target_reconstruction_before_barrier_one_permitted": False,
            "future_target_read_before_barrier_two_permitted": False,
            "confirmation_before_calibration_go_permitted": False,
        },
        "held-v8 information boundary changed",
    )
    stage = artifact.get("stage")
    _require(stage in _ROLE_VALUES, "held-v8 lock stage changed")
    if stage == "calibration":
        _require(
            artifact.get("confirmation_access_authorized") is False
            and artifact.get("parent_calibration_lock") is None
            and artifact.get("calibration_gate_evidence") is None
            and _canonical_path(path) == root / "calibration-lock.json",
            "calibration lock prematurely authorizes confirmation",
        )
    else:
        _require(
            artifact.get("confirmation_access_authorized") is True
            and _canonical_path(path) == root / "confirmation-lock.json",
            "confirmation lock is not authorized",
        )
        parent_path = _validate_bound_file(
            artifact.get("parent_calibration_lock"),
            role="parent calibration lock",
            required_mode=_SEALED_FILE_MODE,
            current_lock=artifact,
            required_parent=root,
        )
        parent = validate_protocol_lock(parent_path)
        _require(parent.get("stage") == "calibration", "confirmation parent changed")
        decision_path = _validate_bound_file(
            artifact.get("calibration_gate_evidence"),
            role="calibration GO decision",
            required_mode=_SEALED_FILE_MODE,
            current_lock=artifact,
            required_parent=root / "calibration",
        )
        validate_calibration_gate_decision(decision_path, parent_path)
        for key in (
            "execution_attempt",
            "held_root",
            "cohort",
            "case_whitelist",
            "calibration_case_whitelist",
            "frame_count",
            "update_frames",
            "immutable_bindings",
            "lineage",
            "frozen_field_contract",
            "replacement_source_inventory_contract",
            "primary_method",
            "freshness_and_reuse",
            "information_boundary",
        ):
            _require(
                artifact.get(key) == parent.get(key), f"confirmation changed {key}"
            )
    _require(
        artifact.get("artifact_sha256") == held_artifact_sha256(artifact),
        "held-v8 lock checksum changed",
    )
    return artifact


# Signature-compatible name used by the unchanged numerical workers.
load_held_protocol_lock = validate_protocol_lock


def locked_case_names(
    lock_path: str | Path,
    *,
    role: Literal["calibration", "confirmation"],
) -> tuple[str, ...]:
    lock = validate_protocol_lock(lock_path)
    _authorize_role(lock, role)
    return CALIBRATION_CASE_NAMES if role == "calibration" else CONFIRMATION_CASE_NAMES


def _authorize_role(lock: Mapping[str, Any], role: str) -> None:
    _require(role in _ROLE_VALUES, "unsupported cohort role")
    if role == "calibration":
        _require(lock.get("stage") == "calibration", "calibration requires its lock")
    else:
        _require(
            lock.get("stage") == "confirmation"
            and lock.get("confirmation_access_authorized") is True,
            "confirmation remains inaccessible until calibration GO",
        )


def _authorize_case(
    lock: Mapping[str, Any], case_name: str, role: str
) -> dict[str, Any]:
    _authorize_role(lock, role)
    return _case_identity(case_name, role)


def _replacement_source_barrier_evidence(
    lock_path: str | Path,
) -> CohortBarrierEvidence:
    lock = validate_protocol_lock(lock_path)
    _require(
        lock.get("stage") == "calibration",
        "replacement source acquisition requires the calibration lock",
    )
    _case_identity(FRESH_REPLACEMENT_CASE_NAME, "calibration")
    contract_sha256 = held_contract_sha256(
        replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
    )
    bindings = (
        (
            FRESH_REPLACEMENT_CASE_NAME,
            (("replacement_source_inventory_contract_sha256", contract_sha256),),
        ),
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "barrier_number": 0,
        "role": "calibration",
        "operation": REPLACEMENT_SOURCE_OPERATION,
        "lock_file_sha256": _sha256_file(lock_path),
        "lock_artifact_sha256": lock["artifact_sha256"],
        "case_name": FRESH_REPLACEMENT_CASE_NAME,
        "replacement_source_inventory_contract": (
            replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
        ),
    }
    return CohortBarrierEvidence(
        protocol_id=PROTOCOL_ID,
        barrier_number=0,
        role="calibration",
        operation=REPLACEMENT_SOURCE_OPERATION,
        lock_path=str(_canonical_path(lock_path)),
        lock_file_sha256=payload["lock_file_sha256"],
        lock_artifact_sha256=lock["artifact_sha256"],
        ordered_case_names=(FRESH_REPLACEMENT_CASE_NAME,),
        ordered_artifact_bindings=bindings,
        barrier_sha256=_barrier_digest(payload),
    )


def replacement_source_permit_evidence(lock_path: str | Path) -> dict[str, Any]:
    """Return the deterministic evidence a consumed source permit must emit."""

    evidence = _replacement_source_barrier_evidence(lock_path)
    return {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration",
        "case_name": FRESH_REPLACEMENT_CASE_NAME,
        "operation": REPLACEMENT_SOURCE_OPERATION,
        "lock_file_sha256": evidence.lock_file_sha256,
        "lock_artifact_sha256": evidence.lock_artifact_sha256,
        "cohort_barrier_sha256": evidence.barrier_sha256,
        "replacement_source_inventory_contract": deepcopy(
            replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
        ),
        "replacement_source_inventory_contract_sha256": held_contract_sha256(
            replacement_source.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
        ),
        "single_use_consumed": True,
        "process_local_capability": True,
    }


def authorize_replacement_source_acquisition(lock_path: str | Path) -> object:
    """Issue the only permit that may start the exact 072 acquisition."""

    def revalidate() -> CohortBarrierEvidence:
        return _replacement_source_barrier_evidence(lock_path)

    capabilities = _issue_capabilities(revalidate(), revalidate=revalidate)
    return capabilities[FRESH_REPLACEMENT_CASE_NAME]


def consume_replacement_source_acquisition_capability(
    permit: object,
    *,
    case_name: str,
    operation: str,
) -> dict[str, Any]:
    evidence = consume_case_capability(
        permit,
        case_name=case_name,
        operation=operation,
    )
    expected = replacement_source_permit_evidence(
        _LIVE_CAPABILITIES[id(permit)].revalidate().lock_path
    )
    _require(evidence.items() <= expected.items(), "source permit evidence changed")
    return expected


def validate_frame_zero_bundle_manifest(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Validate a fresh v8 frame-zero manifest and its complete source audit."""

    _require_mode(manifest_path, _SEALED_FILE_MODE, role="frame-zero manifest")
    lock = validate_protocol_lock(lock_path)
    _require_current_execution_path(
        manifest_path,
        lock=lock,
        role="frame-zero manifest",
    )
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_kind") == FRAME_ZERO_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 frame-zero artifact",
    )
    case_name = str(manifest.get("case_name"))
    role = str(manifest.get("role"))
    identity = _authorize_case(lock, case_name, role)
    frame_zero_root = _case_stage_root(
        lock,
        role=role,
        case_name=case_name,
        stage="frame-zero",
    )
    _require_current_execution_path(
        manifest_path,
        lock=lock,
        role="frame-zero manifest",
        required_parent=frame_zero_root,
    )
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "frame-zero binds another case")
    if expected_role is not None:
        _require(role == expected_role, "frame-zero binds another role")
    for key in ("object_id", "episode_id"):
        _require(manifest.get(key) == identity[key], f"frame-zero {key} changed")
    _require(
        manifest.get("lock_sha256") == _sha256_file(lock_path)
        and manifest.get("lock_artifact_sha256") == lock["artifact_sha256"],
        "frame-zero binds another lock",
    )
    config = manifest.get("config")
    _require(isinstance(config, Mapping), "frame-zero config is missing")
    expected_config = lock["immutable_bindings"].get("frame_zero_default_config")
    _require(
        expected_config is not None and held_contract_sha256(config) == expected_config,
        "frame-zero config differs from the immutable lock",
    )
    _require(
        manifest.get("artifact_sha256") == held_artifact_sha256(manifest),
        "frame-zero checksum changed",
    )
    _validate_bound_file(
        manifest.get("bundle"),
        role="frame-zero bundle",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=frame_zero_root,
    )
    action_alignment = manifest.get("action_alignment")
    _require(
        isinstance(action_alignment, Mapping),
        "frame-zero action alignment is missing",
    )
    _validate_bound_file(
        action_alignment.get("selected_action_bundle"),
        role="selected action bundle",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=frame_zero_root,
    )

    # Reuse the source-file, robot-window, camera, mask, and geometry validator,
    # but not its hard-coded v7 identity.  The original v8 bytes were checked
    # above; this transient copy is never materialized as evidence.
    legacy_view = deepcopy(manifest)
    legacy_view["protocol_id"] = _LEGACY_PROTOCOL_ID
    legacy_view["artifact_sha256"] = frame_zero_assets.artifact_sha256(legacy_view)
    frame_zero_assets.validate_frame_zero_bundle_manifest(
        legacy_view,
        require_bounded_subset_audit=True,
    )
    return manifest


def create_physical_prior_seal(
    output_path: str | Path,
    lock_path: str | Path,
    frame_zero_manifest_path: str | Path,
    physical_artifacts: Mapping[str, str | Path],
    *,
    case_name: str,
    role: str = "confirmation",
) -> dict[str, Any]:
    lock = validate_protocol_lock(lock_path)
    identity = _authorize_case(lock, case_name, role)
    physical_root = _case_stage_root(
        lock,
        role=role,
        case_name=case_name,
        stage="physical",
    )
    _require_current_execution_path(
        output_path,
        lock=lock,
        role="physical seal output",
        required_parent=physical_root,
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    _require(
        set(physical_artifacts) == set(PHYSICAL_ARTIFACT_ROLES),
        "physical artifact roles changed",
    )
    artifacts = {
        name: _seal_existing_regular_file(
            path,
            role=name,
            lock=lock,
            required_parent=physical_root,
        )
        for name, path in physical_artifacts.items()
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PHYSICAL_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        **identity,
        "lock": _bound_file(lock_path),
        "frame_zero_manifest": _bound_file(frame_zero_manifest_path),
        "frame_zero_manifest_artifact_sha256": frame_zero["artifact_sha256"],
        "physical_artifacts": artifacts,
        "freshness": {
            "fresh_v8_prediction": True,
            "v7_execution_artifacts_reused": False,
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prior_sealed_before_rgb_frame_gt_zero": True,
        },
    }
    value["artifact_sha256"] = held_artifact_sha256(value)
    _write_new_json(output_path, value)
    return validate_physical_prior_seal(
        output_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )


def validate_physical_prior_seal(
    seal_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    _require_mode(seal_path, _SEALED_FILE_MODE, role="physical seal")
    lock = validate_protocol_lock(lock_path)
    _require_current_execution_path(
        seal_path,
        lock=lock,
        role="physical seal",
    )
    seal = _load_json(seal_path)
    _require(
        seal.get("schema_version") == SCHEMA_VERSION
        and seal.get("artifact_kind") == PHYSICAL_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 physical seal",
    )
    case_name = str(seal.get("case_name"))
    role = str(seal.get("role"))
    identity = _authorize_case(lock, case_name, role)
    physical_root = _case_stage_root(
        lock,
        role=role,
        case_name=case_name,
        stage="physical",
    )
    _require_current_execution_path(
        seal_path,
        lock=lock,
        role="physical seal",
        required_parent=physical_root,
    )
    for key, expected in identity.items():
        _require(seal.get(key) == expected, f"physical seal {key} changed")
    if expected_case_name is not None:
        _require(case_name == expected_case_name, "physical seal binds another case")
    if expected_role is not None:
        _require(role == expected_role, "physical seal binds another role")
    _require(
        seal.get("lock") == _bound_file(lock_path),
        "physical seal binds another lock",
    )
    frame_zero_path = _validate_bound_file(
        seal.get("frame_zero_manifest"),
        role="frame-zero manifest",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=_case_stage_root(
            lock,
            role=role,
            case_name=case_name,
            stage="frame-zero",
        ),
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    _require(
        seal.get("frame_zero_manifest_artifact_sha256")
        == frame_zero["artifact_sha256"],
        "physical seal frame-zero binding changed",
    )
    artifacts = seal.get("physical_artifacts")
    _require(
        isinstance(artifacts, Mapping)
        and set(artifacts) == set(PHYSICAL_ARTIFACT_ROLES),
        "physical artifact set changed",
    )
    for name, record in artifacts.items():
        _validate_bound_file(
            record,
            role=name,
            required_mode=_SEALED_FILE_MODE,
            current_lock=lock,
            required_parent=physical_root,
        )
    _require(
        seal.get("freshness")
        == {"fresh_v8_prediction": True, "v7_execution_artifacts_reused": False},
        "physical seal freshness changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prior_sealed_before_rgb_frame_gt_zero": True,
        },
        "physical seal information boundary changed",
    )
    _require(
        seal.get("artifact_sha256") == held_artifact_sha256(seal),
        "physical seal checksum changed",
    )
    return seal


def create_prefix_stage_authorization(
    output_path: str | Path,
    lock_path: str | Path,
    physical_seal_path: str | Path,
) -> dict[str, Any]:
    physical = validate_physical_prior_seal(physical_seal_path, lock_path)
    lock = validate_protocol_lock(lock_path)
    case_root = _case_root(
        lock,
        role=str(physical["role"]),
        case_name=str(physical["case_name"]),
    )
    _require_current_execution_path(
        output_path,
        lock=lock,
        role="prefix authorization output",
        required_parent=case_root,
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PREFIX_AUTHORIZATION_KIND,
        "protocol_id": PROTOCOL_ID,
        **{
            key: physical[key]
            for key in ("case_name", "object_id", "episode_id", "role")
        },
        "lock": _bound_file(lock_path),
        "physical_prior_seal": _bound_file(physical_seal_path),
        "physical_prior_artifact_sha256": physical["artifact_sha256"],
        "permitted_update_frames": list(UPDATE_FRAMES),
        "maximum_object_rgb_frame_permitted": UPDATE_FRAMES[-1],
        "information_boundary": {
            "physical_prior_validated_before_prefix_authorization": True,
            "causal_prefix_only": True,
            "future_tactile_permitted": False,
            "outcome_creation_permitted": False,
            "outcome_read_permitted": False,
        },
    }
    value["artifact_sha256"] = held_artifact_sha256(value)
    _write_new_json(output_path, value)
    return validate_prefix_stage_authorization(output_path, lock_path)


def validate_prefix_stage_authorization(
    authorization_path: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    _require_mode(authorization_path, _SEALED_FILE_MODE, role="prefix authorization")
    lock = validate_protocol_lock(lock_path)
    _require_current_execution_path(
        authorization_path,
        lock=lock,
        role="prefix authorization",
    )
    authorization = _load_json(authorization_path)
    _require(
        authorization.get("schema_version") == SCHEMA_VERSION
        and authorization.get("artifact_kind") == PREFIX_AUTHORIZATION_KIND
        and authorization.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 prefix authorization",
    )
    case_name = str(authorization.get("case_name"))
    role = str(authorization.get("role"))
    case_root = _case_root(lock, role=role, case_name=case_name)
    _require_current_execution_path(
        authorization_path,
        lock=lock,
        role="prefix authorization",
        required_parent=case_root,
    )
    _require(
        authorization.get("lock") == _bound_file(lock_path),
        "prefix authorization binds another lock",
    )
    physical_path = _validate_bound_file(
        authorization.get("physical_prior_seal"),
        role="physical seal",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=_case_stage_root(
            lock,
            role=role,
            case_name=case_name,
            stage="physical",
        ),
    )
    physical = validate_physical_prior_seal(
        physical_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    for key in ("case_name", "object_id", "episode_id", "role"):
        _require(authorization.get(key) == physical[key], f"prefix {key} changed")
    _require(
        authorization.get("physical_prior_artifact_sha256")
        == physical["artifact_sha256"],
        "prefix physical binding changed",
    )
    _require(
        authorization.get("permitted_update_frames") == list(UPDATE_FRAMES)
        and authorization.get("maximum_object_rgb_frame_permitted")
        == UPDATE_FRAMES[-1],
        "prefix temporal contract changed",
    )
    _require(
        authorization.get("information_boundary")
        == {
            "physical_prior_validated_before_prefix_authorization": True,
            "causal_prefix_only": True,
            "future_tactile_permitted": False,
            "outcome_creation_permitted": False,
            "outcome_read_permitted": False,
        },
        "prefix information boundary changed",
    )
    _require(
        authorization.get("artifact_sha256") == held_artifact_sha256(authorization),
        "prefix authorization checksum changed",
    )
    return authorization


def create_online_prediction_seal(
    output_path: str | Path,
    lock_path: str | Path,
    prefix_authorization_path: str | Path,
    online_artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    authorization = validate_prefix_stage_authorization(
        prefix_authorization_path, lock_path
    )
    lock = validate_protocol_lock(lock_path)
    online_root = _case_stage_root(
        lock,
        role=str(authorization["role"]),
        case_name=str(authorization["case_name"]),
        stage="online",
    )
    _require_current_execution_path(
        output_path,
        lock=lock,
        role="online seal output",
        required_parent=online_root,
    )
    _require(
        set(online_artifacts) == set(ONLINE_ARTIFACT_ROLES),
        "online artifact roles changed",
    )
    artifacts = {
        name: _seal_existing_regular_file(
            path,
            role=name,
            lock=lock,
            required_parent=online_root,
        )
        for name, path in online_artifacts.items()
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ONLINE_SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        **{
            key: authorization[key]
            for key in ("case_name", "object_id", "episode_id", "role")
        },
        "lock": _bound_file(lock_path),
        "prefix_authorization": _bound_file(prefix_authorization_path),
        "prefix_authorization_artifact_sha256": authorization["artifact_sha256"],
        "online_artifacts": artifacts,
        "primary_method": deepcopy(PRIMARY_METHOD),
        "freshness": {
            "fresh_v8_prediction": True,
            "v7_execution_artifacts_reused": False,
        },
        "information_boundary": {
            "maximum_object_rgb_frame_read": UPDATE_FRAMES[-1],
            "object_rgb_frames_read_as_causal_prefixes": True,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "all_frozen_predictions_hashed_before_outcome": True,
        },
    }
    value["artifact_sha256"] = held_artifact_sha256(value)
    _write_new_json(output_path, value)
    return validate_online_prediction_seal(output_path, lock_path)


def validate_online_prediction_seal(
    seal_path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    _require_mode(seal_path, _SEALED_FILE_MODE, role="online seal")
    lock = validate_protocol_lock(lock_path)
    _require_current_execution_path(
        seal_path,
        lock=lock,
        role="online seal",
    )
    seal = _load_json(seal_path)
    _require(
        seal.get("schema_version") == SCHEMA_VERSION
        and seal.get("artifact_kind") == ONLINE_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 online seal",
    )
    case_name = str(seal.get("case_name"))
    role = str(seal.get("role"))
    online_root = _case_stage_root(
        lock,
        role=role,
        case_name=case_name,
        stage="online",
    )
    _require_current_execution_path(
        seal_path,
        lock=lock,
        role="online seal",
        required_parent=online_root,
    )
    _require(seal.get("lock") == _bound_file(lock_path), "online seal changed lock")
    authorization_path = _validate_bound_file(
        seal.get("prefix_authorization"),
        role="prefix authorization",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=_case_root(lock, role=role, case_name=case_name),
    )
    authorization = validate_prefix_stage_authorization(authorization_path, lock_path)
    for key in ("case_name", "object_id", "episode_id", "role"):
        _require(seal.get(key) == authorization[key], f"online seal {key} changed")
    if expected_case_name is not None:
        _require(
            seal["case_name"] == expected_case_name, "online seal binds another case"
        )
    if expected_role is not None:
        _require(seal["role"] == expected_role, "online seal binds another role")
    _require(
        seal.get("prefix_authorization_artifact_sha256")
        == authorization["artifact_sha256"],
        "online seal prefix binding changed",
    )
    artifacts = seal.get("online_artifacts")
    _require(
        isinstance(artifacts, Mapping) and set(artifacts) == set(ONLINE_ARTIFACT_ROLES),
        "online artifact set changed",
    )
    for name, record in artifacts.items():
        _validate_bound_file(
            record,
            role=name,
            required_mode=_SEALED_FILE_MODE,
            current_lock=lock,
            required_parent=online_root,
        )
    _require(seal.get("primary_method") == PRIMARY_METHOD, "online method changed")
    _require(
        seal.get("freshness")
        == {"fresh_v8_prediction": True, "v7_execution_artifacts_reused": False},
        "online seal freshness changed",
    )
    _require(
        seal.get("information_boundary")
        == {
            "maximum_object_rgb_frame_read": UPDATE_FRAMES[-1],
            "object_rgb_frames_read_as_causal_prefixes": True,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "all_frozen_predictions_hashed_before_outcome": True,
        },
        "online seal information boundary changed",
    )
    _require(
        seal.get("artifact_sha256") == held_artifact_sha256(seal),
        "online seal checksum changed",
    )
    return seal


def _call_artifact_validator(
    validator: ArtifactValidator,
    path: str | Path,
    lock_path: str | Path,
    *,
    case_name: str,
    role: str,
) -> Mapping[str, Any]:
    value = validator(
        path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    _require(isinstance(value, Mapping), "artifact validator returned no evidence")
    _require(
        value.get("protocol_id") == PROTOCOL_ID and value.get("case_name") == case_name,
        "artifact validator returned another protocol or case",
    )
    if "role" in value:
        _require(value.get("role") == role, "artifact validator returned another role")
    return value


def _validate_mapping_keys(
    paths: Mapping[str, str | Path],
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    _require(
        set(paths) == set(expected),
        f"{label} remains sealed until every exact cohort artifact is present",
    )
    normalized = [_canonical_path(paths[case]) for case in expected]
    _require(len(set(normalized)) == len(normalized), f"{label} paths are duplicated")


def _barrier_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(payload))).hexdigest()


def validate_first_cohort_barrier(
    lock_path: str | Path,
    *,
    physical_seal_paths: Mapping[str, str | Path],
    online_seal_paths: Mapping[str, str | Path],
    frozen_field_manifest_paths: Mapping[str, str | Path],
    replacement_aligned_source_manifest_path: str | Path | None = None,
    role: Literal["calibration", "confirmation"],
    physical_validator: ArtifactValidator = validate_physical_prior_seal,
    online_validator: ArtifactValidator = validate_online_prediction_seal,
    frozen_field_validator: Callable[..., Mapping[str, Any]] = (
        query_artifacts.validate_preoutcome_frozen_field_manifest
    ),
    replacement_source_validator: Callable[..., Mapping[str, Any]] = (
        replacement_source.validate_aligned_source_manifest
    ),
) -> CohortBarrierEvidence:
    """Purely replay all fresh prediction and pre-outcome-field bindings."""

    lock = validate_protocol_lock(lock_path)
    expected = locked_case_names(lock_path, role=role)
    _validate_mapping_keys(physical_seal_paths, expected, label="physical cohort")
    _validate_mapping_keys(online_seal_paths, expected, label="online cohort")
    _validate_mapping_keys(
        frozen_field_manifest_paths, expected, label="frozen-field cohort"
    )
    source_path: Path | None = None
    if role == "calibration":
        _require(
            replacement_aligned_source_manifest_path is not None,
            "barrier one requires the exact 072 aligned-source manifest",
        )
        source_path = _require_current_execution_path(
            replacement_aligned_source_manifest_path,
            lock=lock,
            role="replacement aligned-source manifest",
            required_parent=_held_root(lock) / "replacement-source" / "manifests",
        )
    else:
        _require(
            replacement_aligned_source_manifest_path is None,
            "confirmation barrier may not substitute a replacement source",
        )
    cohort_paths: dict[str, tuple[Path, Path, Path]] = {}
    for case_name in expected:
        physical_path = _require_current_execution_path(
            physical_seal_paths[case_name],
            lock=lock,
            role="physical seal",
            required_parent=_case_stage_root(
                lock,
                role=role,
                case_name=case_name,
                stage="physical",
            ),
        )
        online_path = _require_current_execution_path(
            online_seal_paths[case_name],
            lock=lock,
            role="online seal",
            required_parent=_case_stage_root(
                lock,
                role=role,
                case_name=case_name,
                stage="online",
            ),
        )
        field_path = _require_current_execution_path(
            frozen_field_manifest_paths[case_name],
            lock=lock,
            role="frozen field",
            required_parent=_case_stage_root(
                lock,
                role=role,
                case_name=case_name,
                stage="frozen-field",
            ),
        )
        cohort_paths[case_name] = (physical_path, online_path, field_path)

    source_binding: tuple[tuple[str, str], ...] = ()
    if source_path is not None:
        expected_permit = replacement_source_permit_evidence(lock_path)
        source_manifest = replacement_source_validator(
            source_path,
            expected_source_permit=expected_permit,
        )
        _require(
            isinstance(source_manifest, Mapping)
            and source_manifest.get("protocol_id") == PROTOCOL_ID
            and source_manifest.get("case_name") == FRESH_REPLACEMENT_CASE_NAME
            and source_manifest.get("source_permit") == expected_permit,
            "replacement aligned-source manifest changed permit or case",
        )
        source_binding = (
            ("replacement_aligned_source_manifest_sha256", _sha256_file(source_path)),
            (
                "replacement_aligned_source_artifact_sha256",
                str(source_manifest["artifact_sha256"]),
            ),
        )

    ordered: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for case_name in expected:
        physical_path, online_path, field_path = cohort_paths[case_name]
        physical = _call_artifact_validator(
            physical_validator,
            physical_path,
            lock_path,
            case_name=case_name,
            role=role,
        )
        online = _call_artifact_validator(
            online_validator,
            online_path,
            lock_path,
            case_name=case_name,
            role=role,
        )
        field_manifest = frozen_field_validator(
            field_path,
            lock_path=lock_path,
            expected_case_name=case_name,
        )
        _require(
            isinstance(field_manifest, Mapping)
            and field_manifest.get("protocol_id") == PROTOCOL_ID
            and field_manifest.get("case_name") == case_name,
            "frozen-field validator returned another protocol or case",
        )
        _require(
            field_manifest.get("online_prediction_seal") == _bound_file(online_path)
            and field_manifest.get("online_prediction_seal_artifact_sha256")
            == online.get("artifact_sha256"),
            "frozen field does not bind the cohort online seal",
        )
        bindings = (
            ("physical_seal_sha256", _sha256_file(physical_path)),
            ("physical_artifact_sha256", str(physical["artifact_sha256"])),
            ("online_seal_sha256", _sha256_file(online_path)),
            ("online_artifact_sha256", str(online["artifact_sha256"])),
            ("frozen_field_sha256", _sha256_file(field_path)),
            (
                "frozen_field_artifact_sha256",
                str(field_manifest["artifact_sha256"]),
            ),
        )
        if case_name == FRESH_REPLACEMENT_CASE_NAME:
            bindings = (*bindings, *source_binding)
        ordered.append(
            (
                case_name,
                bindings,
            )
        )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "barrier_number": 1,
        "role": role,
        "operation": TARGET_RECONSTRUCTION_OPERATION,
        "lock_file_sha256": _sha256_file(lock_path),
        "lock_artifact_sha256": lock["artifact_sha256"],
        "ordered_case_names": list(expected),
        "ordered_artifact_bindings": ordered,
        "complete_cohort": True,
        "fresh_v8_only": True,
    }
    return CohortBarrierEvidence(
        protocol_id=PROTOCOL_ID,
        barrier_number=1,
        role=role,
        operation=TARGET_RECONSTRUCTION_OPERATION,
        lock_path=str(_canonical_path(lock_path)),
        lock_file_sha256=payload["lock_file_sha256"],
        lock_artifact_sha256=lock["artifact_sha256"],
        ordered_case_names=expected,
        ordered_artifact_bindings=tuple(ordered),
        barrier_sha256=_barrier_digest(payload),
    )


def _query_validator_adapter(
    path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str,
    expected_role: str,
) -> Mapping[str, Any]:
    del expected_role
    return query_artifacts.validate_official_frame_zero_query_artifact(
        path, lock_path, expected_case_name=expected_case_name
    )


def _queried_validator_adapter(
    path: str | Path,
    lock_path: str | Path,
    *,
    expected_case_name: str,
    expected_role: str,
) -> Mapping[str, Any]:
    del expected_role
    return query_artifacts.validate_queried_prediction_artifact(
        path,
        lock_path=lock_path,
        expected_case_name=expected_case_name,
    )


def validate_second_cohort_barrier(
    lock_path: str | Path,
    *,
    official_query_manifest_paths: Mapping[str, str | Path],
    queried_prediction_seal_paths: Mapping[str, str | Path],
    role: Literal["calibration", "confirmation"],
    official_query_validator: ArtifactValidator = _query_validator_adapter,
    queried_prediction_validator: ArtifactValidator = _queried_validator_adapter,
) -> CohortBarrierEvidence:
    """Purely replay every x0 reconstruction and pre-score query seal."""

    lock = validate_protocol_lock(lock_path)
    expected = locked_case_names(lock_path, role=role)
    _validate_mapping_keys(
        official_query_manifest_paths, expected, label="official x0 query cohort"
    )
    _validate_mapping_keys(
        queried_prediction_seal_paths, expected, label="queried prediction cohort"
    )
    cohort_paths: dict[str, tuple[Path, Path]] = {}
    for case_name in expected:
        query_path = _require_current_execution_path(
            official_query_manifest_paths[case_name],
            lock=lock,
            role="official x0 query",
            required_parent=(_held_root(lock) / role / "query-inputs" / case_name),
        )
        queried_path = _require_current_execution_path(
            queried_prediction_seal_paths[case_name],
            lock=lock,
            role="queried prediction",
            required_parent=(_held_root(lock) / role / "query-outputs" / case_name),
        )
        cohort_paths[case_name] = (query_path, queried_path)

    ordered: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for case_name in expected:
        query_path, queried_path = cohort_paths[case_name]
        query = _call_artifact_validator(
            official_query_validator,
            query_path,
            lock_path,
            case_name=case_name,
            role=role,
        )
        queried = _call_artifact_validator(
            queried_prediction_validator,
            queried_path,
            lock_path,
            case_name=case_name,
            role=role,
        )
        _require(
            queried.get("official_query_manifest") == _bound_file(query_path)
            and queried.get("official_query_manifest_artifact_sha256")
            == query.get("artifact_sha256"),
            "queried prediction does not bind the cohort x0 query",
        )
        ordered.append(
            (
                case_name,
                (
                    ("official_query_sha256", _sha256_file(query_path)),
                    ("official_query_artifact_sha256", str(query["artifact_sha256"])),
                    ("queried_prediction_sha256", _sha256_file(queried_path)),
                    (
                        "queried_prediction_artifact_sha256",
                        str(queried["artifact_sha256"]),
                    ),
                ),
            )
        )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "barrier_number": 2,
        "role": role,
        "operation": FUTURE_SCORE_OPERATION,
        "lock_file_sha256": _sha256_file(lock_path),
        "lock_artifact_sha256": lock["artifact_sha256"],
        "ordered_case_names": list(expected),
        "ordered_artifact_bindings": ordered,
        "complete_cohort": True,
        "x0_only_queries": True,
    }
    return CohortBarrierEvidence(
        protocol_id=PROTOCOL_ID,
        barrier_number=2,
        role=role,
        operation=FUTURE_SCORE_OPERATION,
        lock_path=str(_canonical_path(lock_path)),
        lock_file_sha256=payload["lock_file_sha256"],
        lock_artifact_sha256=lock["artifact_sha256"],
        ordered_case_names=expected,
        ordered_artifact_bindings=tuple(ordered),
        barrier_sha256=_barrier_digest(payload),
    )


def _issue_capabilities(
    evidence: CohortBarrierEvidence,
    *,
    revalidate: Callable[[], CohortBarrierEvidence],
    predecessor_barrier_sha256: str | None = None,
) -> dict[str, object]:
    key = (
        evidence.lock_file_sha256,
        evidence.role,
        evidence.operation,
        evidence.barrier_sha256,
    )
    _require(key not in _ISSUED_BARRIERS, "cohort capabilities were already issued")
    capabilities: dict[str, object] = {}
    states: list[_CapabilityState] = []
    for case_name in evidence.ordered_case_names:
        capability = _CaseCapability(
            role=evidence.role,
            case_name=case_name,
            operation=evidence.operation,
            lock_file_sha256=evidence.lock_file_sha256,
            lock_artifact_sha256=evidence.lock_artifact_sha256,
            cohort_barrier_sha256=evidence.barrier_sha256,
            predecessor_barrier_sha256=predecessor_barrier_sha256,
            _nonce=object(),
            _authority=_CAPABILITY_AUTHORITY,
        )
        state = _CapabilityState(capability=capability, revalidate=revalidate)
        states.append(state)
        capabilities[case_name] = capability
    _ISSUED_BARRIERS.add(key)
    for state in states:
        _LIVE_CAPABILITIES[id(state.capability)] = state
    return capabilities


def authorize_target_reconstruction_capabilities(
    lock_path: str | Path,
    *,
    physical_seal_paths: Mapping[str, str | Path],
    online_seal_paths: Mapping[str, str | Path],
    frozen_field_manifest_paths: Mapping[str, str | Path],
    replacement_aligned_source_manifest_path: str | Path | None = None,
    role: Literal["calibration", "confirmation"],
    physical_validator: ArtifactValidator = validate_physical_prior_seal,
    online_validator: ArtifactValidator = validate_online_prediction_seal,
    frozen_field_validator: Callable[..., Mapping[str, Any]] = (
        query_artifacts.validate_preoutcome_frozen_field_manifest
    ),
    replacement_source_validator: Callable[..., Mapping[str, Any]] = (
        replacement_source.validate_aligned_source_manifest
    ),
) -> dict[str, object]:
    kwargs = {
        "physical_seal_paths": dict(physical_seal_paths),
        "online_seal_paths": dict(online_seal_paths),
        "frozen_field_manifest_paths": dict(frozen_field_manifest_paths),
        "replacement_aligned_source_manifest_path": (
            replacement_aligned_source_manifest_path
        ),
        "role": role,
        "physical_validator": physical_validator,
        "online_validator": online_validator,
        "frozen_field_validator": frozen_field_validator,
        "replacement_source_validator": replacement_source_validator,
    }

    def revalidate() -> CohortBarrierEvidence:
        return validate_first_cohort_barrier(lock_path, **kwargs)  # type: ignore[arg-type]

    return _issue_capabilities(revalidate(), revalidate=revalidate)


def authorize_future_score_capabilities(
    lock_path: str | Path,
    *,
    official_query_manifest_paths: Mapping[str, str | Path],
    queried_prediction_seal_paths: Mapping[str, str | Path],
    role: Literal["calibration", "confirmation"],
    official_query_validator: ArtifactValidator = _query_validator_adapter,
    queried_prediction_validator: ArtifactValidator = _queried_validator_adapter,
) -> dict[str, object]:
    kwargs = {
        "official_query_manifest_paths": dict(official_query_manifest_paths),
        "queried_prediction_seal_paths": dict(queried_prediction_seal_paths),
        "role": role,
        "official_query_validator": official_query_validator,
        "queried_prediction_validator": queried_prediction_validator,
    }

    def revalidate() -> CohortBarrierEvidence:
        return validate_second_cohort_barrier(lock_path, **kwargs)  # type: ignore[arg-type]

    evidence = revalidate()
    completed = _COMPLETED_RECONSTRUCTION_BARRIERS.get(
        (evidence.lock_file_sha256, role)
    )
    _require(
        completed is not None,
        "future scoring remains sealed until all reconstruction capabilities are consumed",
    )
    return _issue_capabilities(
        evidence,
        revalidate=revalidate,
        predecessor_barrier_sha256=completed,
    )


def consume_case_capability(
    permit: object,
    *,
    case_name: str,
    operation: str,
) -> dict[str, Any]:
    """Validate, spend, and audit one process-local held-v8 capability."""

    _require(isinstance(permit, _CaseCapability), "operation lacks a v8 capability")
    state = _LIVE_CAPABILITIES.get(id(permit))
    _require(
        state is not None
        and state.capability is permit
        and permit._authority is _CAPABILITY_AUTHORITY,
        "capability is not live in this process",
    )
    _require(not state.consumed, "capability was already consumed")
    _require(
        permit.case_name == case_name and permit.operation == operation,
        "capability case or operation changed",
    )
    # Spend before replay: a failed final check must never leave a reusable key.
    state.consumed = True
    evidence = state.revalidate()
    _require(
        evidence.role == permit.role
        and evidence.operation == permit.operation
        and evidence.lock_file_sha256 == permit.lock_file_sha256
        and evidence.lock_artifact_sha256 == permit.lock_artifact_sha256
        and evidence.barrier_sha256 == permit.cohort_barrier_sha256
        and case_name in evidence.ordered_case_names,
        "capability cohort or lock binding changed",
    )
    if permit.predecessor_barrier_sha256 is not None:
        _require(
            _COMPLETED_RECONSTRUCTION_BARRIERS.get(
                (permit.lock_file_sha256, permit.role)
            )
            == permit.predecessor_barrier_sha256,
            "capability predecessor reconstruction barrier changed",
        )
    state.validated_consumption = True
    if permit.operation == TARGET_RECONSTRUCTION_OPERATION:
        siblings = [
            candidate
            for candidate in _LIVE_CAPABILITIES.values()
            if candidate.capability.lock_file_sha256 == permit.lock_file_sha256
            and candidate.capability.role == permit.role
            and candidate.capability.operation == permit.operation
            and candidate.capability.cohort_barrier_sha256
            == permit.cohort_barrier_sha256
        ]
        _require(
            {candidate.capability.case_name for candidate in siblings}
            == set(evidence.ordered_case_names),
            "reconstruction capability cohort changed",
        )
        if all(candidate.validated_consumption for candidate in siblings):
            _COMPLETED_RECONSTRUCTION_BARRIERS[
                (permit.lock_file_sha256, permit.role)
            ] = permit.cohort_barrier_sha256
    audit = {
        "protocol_id": PROTOCOL_ID,
        "role": permit.role,
        "case_name": permit.case_name,
        "operation": permit.operation,
        "lock_file_sha256": permit.lock_file_sha256,
        "lock_artifact_sha256": permit.lock_artifact_sha256,
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "single_use_consumed": True,
        "process_local_capability": True,
    }
    if permit.predecessor_barrier_sha256 is not None:
        audit["predecessor_reconstruction_barrier_sha256"] = (
            permit.predecessor_barrier_sha256
        )
    return audit


def validate_calibration_gate_decision(
    decision_path: str | Path,
    calibration_lock_path: str | Path,
) -> dict[str, Any]:
    _require_mode(decision_path, _SEALED_FILE_MODE, role="calibration gate decision")
    lock = validate_protocol_lock(calibration_lock_path)
    _require(lock.get("stage") == "calibration", "gate parent is not calibration")
    calibration_root = _held_root(lock) / "calibration"
    _require_current_execution_path(
        decision_path,
        lock=lock,
        role="calibration gate decision",
        required_parent=calibration_root,
    )
    decision = _load_json(decision_path)
    _require(
        decision.get("schema_version") == SCHEMA_VERSION
        and decision.get("artifact_kind") == CALIBRATION_DECISION_KIND
        and decision.get("protocol_id") == PROTOCOL_ID
        and decision.get("role") == "calibration",
        "unsupported held-v8 calibration gate decision",
    )
    _require(
        decision.get("lock") == _bound_file(calibration_lock_path)
        and decision.get("ordered_case_names") == list(CALIBRATION_CASE_NAMES),
        "calibration decision binds another lock or cohort",
    )
    _require(
        _valid_sha256(decision.get("barrier_two_sha256")),
        "calibration decision lacks barrier-two evidence",
    )
    _validate_bound_file(
        decision.get("score_evidence"),
        role="calibration score evidence",
        required_mode=_SEALED_FILE_MODE,
        current_lock=lock,
        required_parent=calibration_root,
    )
    gate_result = decision.get("gate_result")
    _require(
        isinstance(gate_result, Mapping)
        and gate_result.get("gate") == "v8-calibration-go-no-go-v1"
        and gate_result.get("passed") is (decision.get("decision") == "GO"),
        "calibration gate result and decision disagree",
    )
    _require(decision.get("decision") in {"GO", "NO-GO"}, "invalid gate decision")
    _require(
        decision.get("artifact_sha256") == held_artifact_sha256(decision),
        "calibration gate decision checksum changed",
    )
    return decision


def create_confirmation_protocol_lock(
    output_path: str | Path,
    calibration_lock_path: str | Path,
    calibration_decision_path: str | Path,
) -> dict[str, Any]:
    calibration = validate_protocol_lock(calibration_lock_path)
    _require(calibration.get("stage") == "calibration", "parent is not calibration")
    decision = validate_calibration_gate_decision(
        calibration_decision_path, calibration_lock_path
    )
    _require(
        decision.get("decision") == "GO",
        "confirmation remains inaccessible after calibration NO-GO",
    )
    root = _canonical_path(calibration["held_root"])
    output = _canonical_path(output_path)
    _require(
        output == root / "confirmation-lock.json", "non-canonical confirmation lock"
    )
    artifact = deepcopy(calibration)
    artifact.pop("artifact_sha256", None)
    artifact["stage"] = "confirmation"
    artifact["confirmation_access_authorized"] = True
    artifact["parent_calibration_lock"] = _bound_file(calibration_lock_path)
    artifact["calibration_gate_evidence"] = _bound_file(calibration_decision_path)
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    _write_new_json(output, artifact)
    return validate_protocol_lock(output)


__all__ = [
    "ATTEMPT3_ARCHIVE_ENTRY_COUNT",
    "ATTEMPT3_ARCHIVE_INVENTORY_SHA256",
    "ATTEMPT3_ARCHIVE_METADATA_INVENTORY_SHA256",
    "ATTEMPT3_ARCHIVE_PATH",
    "ATTEMPT3_WITHDRAWAL_INTEGRITY_COMPLETION_PATH",
    "ATTEMPT3_WITHDRAWAL_POINTER_PATH",
    "ATTEMPT3_WITHDRAWAL_REPORT_PATH",
    "CALIBRATION_CASE_NAMES",
    "CALIBRATION_DECISION_KIND",
    "CONFIRMATION_CASES",
    "CONFIRMATION_CASE_NAMES",
    "CohortBarrierEvidence",
    "EXECUTION_ATTEMPT",
    "FRAME_COUNT",
    "FRESHNESS_AND_REUSE_CONTRACT",
    "FRESH_REPLACEMENT_CASE_NAME",
    "FROZEN_FIELD_CONTRACT",
    "FUTURE_SCORE_OPERATION",
    "LOCK_KIND",
    "ONLINE_ARTIFACT_ROLES",
    "ONLINE_SEAL_KIND",
    "OPEN27_DEVELOPMENT_DECISION_FILE_SHA256",
    "PHYSICAL_ARTIFACT_ROLES",
    "PHYSICAL_SEAL_KIND",
    "PREFIX_AUTHORIZATION_KIND",
    "PRIMARY_METHOD",
    "PROTOCOL_ID",
    "REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT",
    "REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256",
    "RETIRED_V7_CASE_NAME",
    "REPLACEMENT_SOURCE_OPERATION",
    "TARGET_RECONSTRUCTION_OPERATION",
    "UPDATE_FRAMES",
    "V7_WITHDRAWAL_REPORT_FILE_SHA256",
    "authorize_future_score_capabilities",
    "authorize_replacement_source_acquisition",
    "authorize_target_reconstruction_capabilities",
    "consume_case_capability",
    "consume_replacement_source_acquisition_capability",
    "create_calibration_protocol_lock",
    "create_confirmation_protocol_lock",
    "create_online_prediction_seal",
    "create_physical_prior_seal",
    "create_prefix_stage_authorization",
    "held_artifact_sha256",
    "held_contract_sha256",
    "load_held_protocol_lock",
    "locked_case_names",
    "prepare_fresh_held_root",
    "replacement_source_permit_evidence",
    "validate_calibration_gate_decision",
    "validate_first_cohort_barrier",
    "validate_frame_zero_bundle_manifest",
    "validate_online_prediction_seal",
    "validate_physical_prior_seal",
    "validate_post_withdrawal_development_use_disclosure",
    "validate_prefix_stage_authorization",
    "validate_protocol_lock",
    "validate_attempt3_withdrawal_lineage",
    "validate_second_cohort_barrier",
]
