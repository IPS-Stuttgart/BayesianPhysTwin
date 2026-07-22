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
from typing import Any, Literal, Protocol

from . import deform360_frame_zero_assets as frame_zero_assets
from . import deform360_held_v8_replacement_source as replacement_source
from . import deform360_held_v8_query_artifacts as query_artifacts


PROTOCOL_ID = "deform360-held-online-belief-v8"
SCHEMA_VERSION = 1
LOCK_KIND = "Deform360HeldOnlineBeliefLock"
FRAME_ZERO_KIND = "Deform360HeldFrameZeroBundle"
PHYSICAL_SEAL_KIND = "Deform360HeldPhysicalPriorSeal"
PREFIX_AUTHORIZATION_KIND = "Deform360HeldCausalPrefixAuthorization"
ONLINE_SEAL_KIND = "Deform360HeldOnlinePredictionSeal"
CALIBRATION_DECISION_KIND = "Deform360HeldV8CalibrationGateDecision"
POST_WITHDRAWAL_DISCLOSURE_KIND = (
    "Deform360HeldV8PostWithdrawalDevelopmentUseDisclosure"
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


def _require_not_v7_execution_path(path: str | Path, *, role: str) -> Path:
    source = _canonical_path(path)
    _require(
        "held-v7" not in source.parts,
        f"{role} may not reuse a held-v7 execution artifact",
    )
    return source


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


def _seal_existing_regular_file(path: str | Path, *, role: str) -> dict[str, Any]:
    """Freeze one fresh builder output before it enters a v8 seal."""

    source = _require_not_v7_execution_path(path, role=role)
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
    reject_v7_execution_path: bool = False,
) -> Path:
    _require(
        isinstance(record, Mapping) and set(record) == _FILE_RECORD_FIELDS,
        f"{role} file record fields changed",
    )
    path = record.get("path")
    _require(isinstance(path, str) and path, f"{role} path is missing")
    if reject_v7_execution_path:
        _require_not_v7_execution_path(path, role=role)
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
            "post_withdrawal_development",
            "retirement",
            "v8_reuse_boundary",
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
        artifact.get("v8_reuse_boundary")
        == {
            "v7_target_or_staging_reused": False,
            "v7_physical_prediction_reused": False,
            "v7_online_prediction_reused": False,
            "v7_query_or_score_reused": False,
            "v7_execution_artifact_reused": False,
            "v7_withdrawal_report_used_only_as_immutable_lineage": True,
            "all_v8_predictions_targets_queries_and_scores_must_be_fresh": True,
        },
        "v8 reuse boundary changed",
    )
    _require(
        artifact.get("claim_boundary")
        == (
            "This disclosure preserves prospective episode-level evaluation; it "
            "does not turn open development or v8 into an official Deform360 "
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
) -> dict[str, Any]:
    """Create v8 only after proving that its formal root does not exist."""

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
        artifact: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": LOCK_KIND,
            "protocol_id": PROTOCOL_ID,
            "execution_attempt": 3,
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
            "freshness_and_reuse": {
                "held_v8_root_absent_before_lock": True,
                "all_predictions_must_be_fresh_v8_outputs": True,
                "all_targets_queries_and_scores_must_be_fresh_v8_outputs": True,
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
                "full_15_case_fresh_rerun_required": True,
            },
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
        and artifact.get("execution_attempt") == 3,
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
    lineage = artifact.get("lineage")
    _require(
        isinstance(lineage, Mapping)
        and set(lineage)
        == {
            "v7_withdrawal_report",
            "post_withdrawal_development_use_disclosure",
            "open27_development_decision",
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
    _require(
        artifact.get("freshness_and_reuse")
        == {
            "held_v8_root_absent_before_lock": True,
            "all_predictions_must_be_fresh_v8_outputs": True,
            "all_targets_queries_and_scores_must_be_fresh_v8_outputs": True,
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
            "full_15_case_fresh_rerun_required": True,
        },
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
            reject_v7_execution_path=True,
        )
        parent = validate_protocol_lock(parent_path)
        _require(parent.get("stage") == "calibration", "confirmation parent changed")
        decision_path = _validate_bound_file(
            artifact.get("calibration_gate_evidence"),
            role="calibration GO decision",
            required_mode=_SEALED_FILE_MODE,
            reject_v7_execution_path=True,
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

    _require_not_v7_execution_path(manifest_path, role="frame-zero manifest")
    _require_mode(manifest_path, _SEALED_FILE_MODE, role="frame-zero manifest")
    lock = validate_protocol_lock(lock_path)
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
        name: _seal_existing_regular_file(path, role=name)
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
    _require_not_v7_execution_path(seal_path, role="physical seal")
    _require_mode(seal_path, _SEALED_FILE_MODE, role="physical seal")
    lock = validate_protocol_lock(lock_path)
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
        reject_v7_execution_path=True,
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
            reject_v7_execution_path=True,
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
    _require_not_v7_execution_path(authorization_path, role="prefix authorization")
    _require_mode(authorization_path, _SEALED_FILE_MODE, role="prefix authorization")
    authorization = _load_json(authorization_path)
    _require(
        authorization.get("schema_version") == SCHEMA_VERSION
        and authorization.get("artifact_kind") == PREFIX_AUTHORIZATION_KIND
        and authorization.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 prefix authorization",
    )
    _require(
        authorization.get("lock") == _bound_file(lock_path),
        "prefix authorization binds another lock",
    )
    physical_path = _validate_bound_file(
        authorization.get("physical_prior_seal"),
        role="physical seal",
        required_mode=_SEALED_FILE_MODE,
        reject_v7_execution_path=True,
    )
    physical = validate_physical_prior_seal(
        physical_path,
        lock_path,
        expected_case_name=str(authorization.get("case_name")),
        expected_role=str(authorization.get("role")),
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
    _require(
        set(online_artifacts) == set(ONLINE_ARTIFACT_ROLES),
        "online artifact roles changed",
    )
    artifacts = {
        name: _seal_existing_regular_file(path, role=name)
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
    _require_not_v7_execution_path(seal_path, role="online seal")
    _require_mode(seal_path, _SEALED_FILE_MODE, role="online seal")
    seal = _load_json(seal_path)
    _require(
        seal.get("schema_version") == SCHEMA_VERSION
        and seal.get("artifact_kind") == ONLINE_SEAL_KIND
        and seal.get("protocol_id") == PROTOCOL_ID,
        "unsupported held-v8 online seal",
    )
    _require(seal.get("lock") == _bound_file(lock_path), "online seal changed lock")
    authorization_path = _validate_bound_file(
        seal.get("prefix_authorization"),
        role="prefix authorization",
        required_mode=_SEALED_FILE_MODE,
        reject_v7_execution_path=True,
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
            reject_v7_execution_path=True,
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
    source_binding: tuple[tuple[str, str], ...] = ()
    if role == "calibration":
        _require(
            replacement_aligned_source_manifest_path is not None,
            "barrier one requires the exact 072 aligned-source manifest",
        )
        source_path = _require_not_v7_execution_path(
            replacement_aligned_source_manifest_path,
            role="replacement aligned-source manifest",
        )
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
    else:
        _require(
            replacement_aligned_source_manifest_path is None,
            "confirmation barrier may not substitute a replacement source",
        )
    ordered: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for case_name in expected:
        physical_path = _require_not_v7_execution_path(
            physical_seal_paths[case_name], role="physical seal"
        )
        online_path = _require_not_v7_execution_path(
            online_seal_paths[case_name], role="online seal"
        )
        field_path = _require_not_v7_execution_path(
            frozen_field_manifest_paths[case_name], role="frozen field"
        )
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
    ordered: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for case_name in expected:
        query_path = _require_not_v7_execution_path(
            official_query_manifest_paths[case_name], role="official x0 query"
        )
        queried_path = _require_not_v7_execution_path(
            queried_prediction_seal_paths[case_name], role="queried prediction"
        )
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
        reject_v7_execution_path=True,
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
    "CALIBRATION_CASE_NAMES",
    "CALIBRATION_DECISION_KIND",
    "CONFIRMATION_CASES",
    "CONFIRMATION_CASE_NAMES",
    "CohortBarrierEvidence",
    "FRAME_COUNT",
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
    "validate_second_cohort_barrier",
]
