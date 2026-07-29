"""Outcome-blind first-admitted selection ledger for V14."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .deform360_object_exclusion import file_sha256

SELECTION_KIND = "Deform360CausalResponseDirectDepthSelectionLedgerV14"
SELECTION_CONTRACT = "deform360-causal-response-direct-depth-selection-v14"
SELECTION_FILENAME = "deform360_causal_response_direct_depth_selection_v14.json"
FINALIZER_PROTOCOL_KIND = (
    "Deform360CausalResponseDirectDepthSourceFinalizerProtocolV14"
)
FINALIZER_PROTOCOL_CONTRACT = (
    "deform360-causal-response-direct-depth-source-finalizer-v14"
)
FINALIZER_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-source-finalizer"
)
SELECTED_SOURCE_COUNT = 12
_STATUSES = frozenset(
    {"admitted", "preflight_rejected", "technical_preflight_failure"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-selection-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_source_finalizer_protocol(path: str | Path) -> dict[str, Any]:
    """Validate the pre-execution lock for source-panel finalization."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("artifact_kind") == FINALIZER_PROTOCOL_KIND
        and payload.get("contract") == FINALIZER_PROTOCOL_CONTRACT
        and payload.get("protocol_id") == FINALIZER_PROTOCOL_ID
        and payload.get("status") == "locked_before_source_admission_execution",
        "V14 source finalizer identity changed",
    )
    _require(
        payload.get("config_sha256")
        == hashlib.sha256(
            b"deform360-causal-response-direct-depth-source-finalizer-v14\0"
            + json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "config_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "V14 source finalizer checksum changed",
    )
    parents = payload.get("parent_artifacts")
    _require(
        isinstance(parents, Mapping)
        and set(parents)
        == {
            "admission_prelock",
            "exclusion_manifest",
            "method_protocol",
            "staging_queue",
            "synthetic_control",
        }
        and all(
            isinstance(record, Mapping)
            and _valid_digest(record.get("semantic_sha256"))
            and _valid_digest(record.get("file_sha256"))
            for record in parents.values()
        ),
        "V14 source finalizer parent bindings changed",
    )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and set(implementation.get("file_sha256", {}))
        == {
            "admission_module",
            "finalizer_runner",
            "selection_module",
            "source_lock_module",
        }
        and all(
            _valid_digest(value)
            for value in implementation["file_sha256"].values()
        ),
        "V14 source finalizer implementation binding changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("admission_dispositions_read") is True
        and boundary.get("prefix_or_future_object_response_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 source finalizer crossed its information boundary",
    )
    return payload


@dataclass(frozen=True)
class V14SelectionDisposition:
    """One pre-outcome queue disposition."""

    queue_rank: int
    object_hash: str
    case_hash: str
    status: str
    disposition_artifact_sha256: str
    disposition_file_sha256: str
    selected: bool

    def __post_init__(self) -> None:
        _require(self.queue_rank >= 1, "V14 selection rank is invalid")
        _require(self.status in _STATUSES, "V14 selection status is invalid")
        _require(
            all(
                _valid_digest(value)
                for value in (
                    self.object_hash,
                    self.case_hash,
                    self.disposition_artifact_sha256,
                    self.disposition_file_sha256,
                )
            ),
            "V14 selection disposition digest is invalid",
        )
        _require(
            self.selected is (self.status == "admitted"),
            "V14 selected flag differs from admission status",
        )


@dataclass(frozen=True)
class V14SelectionLedger:
    """Immutable queue prefix ending at the twelfth admitted source object."""

    repository_revision: str
    queue_sha256: str
    queue_file_sha256: str
    admission_prelock_config_sha256: str
    admission_prelock_file_sha256: str
    dispositions: tuple[V14SelectionDisposition, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            len(self.repository_revision) == 40
            and all(
                character in "0123456789abcdef"
                for character in self.repository_revision
            ),
            "V14 selection repository revision is invalid",
        )
        _require(
            all(
                _valid_digest(value)
                for value in (
                    self.queue_sha256,
                    self.queue_file_sha256,
                    self.admission_prelock_config_sha256,
                    self.admission_prelock_file_sha256,
                    self.artifact_sha256,
                )
            ),
            "V14 selection ledger digest is invalid",
        )
        ranks = tuple(item.queue_rank for item in self.dispositions)
        _require(
            ranks == tuple(range(1, len(ranks) + 1)),
            "V14 selection dispositions are not a contiguous queue prefix",
        )
        selected = tuple(item for item in self.dispositions if item.selected)
        _require(
            len(selected) == SELECTED_SOURCE_COUNT
            and self.dispositions[-1].selected,
            "V14 selection does not stop at the twelfth admission",
        )
        _require(
            len({item.object_hash for item in selected}) == SELECTED_SOURCE_COUNT
            and len({item.case_hash for item in selected}) == SELECTED_SOURCE_COUNT,
            "V14 selected objects or cases are duplicated",
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": SELECTION_KIND,
            "contract": SELECTION_CONTRACT,
            "repository_revision": self.repository_revision,
            "queue_sha256": self.queue_sha256,
            "queue_file_sha256": self.queue_file_sha256,
            "admission_prelock_config_sha256": (
                self.admission_prelock_config_sha256
            ),
            "admission_prelock_file_sha256": self.admission_prelock_file_sha256,
            "selected_source_count": SELECTED_SOURCE_COUNT,
            "disposition_count": len(self.dispositions),
            "dispositions": [asdict(item) for item in self.dispositions],
            "information_boundary": {
                "selection_rule": "first 12 admitted in immutable queue order",
                "prefix_or_future_object_response_read": False,
                "identity_or_metric_outcome_read": False,
                "target_object_or_outcome_read": False,
                "plaintext_object_or_episode_identity_retained": False,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def build_v14_selection_ledger(
    dispositions: Iterable[V14SelectionDisposition],
    *,
    repository_revision: str,
    queue_sha256: str,
    queue_path: str | Path,
    admission_prelock_config_sha256: str,
    admission_prelock_path: str | Path,
) -> V14SelectionLedger:
    """Build the exact queue prefix through the twelfth admitted case."""

    ordered = tuple(sorted(dispositions, key=lambda item: item.queue_rank))
    provisional = V14SelectionLedger(
        repository_revision=repository_revision,
        queue_sha256=queue_sha256,
        queue_file_sha256=file_sha256(queue_path),
        admission_prelock_config_sha256=admission_prelock_config_sha256,
        admission_prelock_file_sha256=file_sha256(admission_prelock_path),
        dispositions=ordered,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = V14SelectionLedger(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "V14 selection ledger changed after construction",
    )
    return result


def write_v14_selection_ledger(
    path: str | Path,
    ledger: V14SelectionLedger,
) -> None:
    """Write one immutable selection ledger."""

    output = Path(path)
    _require(not output.exists(), "V14 selection ledger already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            ledger.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_v14_selection_ledger(output)


def validate_v14_selection_ledger(path: str | Path) -> V14SelectionLedger:
    """Validate one immutable selection ledger."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == SELECTION_KIND
        and payload.get("contract") == SELECTION_CONTRACT,
        "V14 selection ledger kind or contract changed",
    )
    result = V14SelectionLedger(
        repository_revision=payload["repository_revision"],
        queue_sha256=payload["queue_sha256"],
        queue_file_sha256=payload["queue_file_sha256"],
        admission_prelock_config_sha256=payload[
            "admission_prelock_config_sha256"
        ],
        admission_prelock_file_sha256=payload[
            "admission_prelock_file_sha256"
        ],
        dispositions=tuple(
            V14SelectionDisposition(**record)
            for record in payload["dispositions"]
        ),
        artifact_sha256=payload["artifact_sha256"],
    )
    _require(
        result.descriptor() == payload
        and _canonical_sha256(payload) == result.artifact_sha256,
        "V14 selection ledger checksum or descriptor changed",
    )
    return result


__all__ = [
    "FINALIZER_PROTOCOL_CONTRACT",
    "FINALIZER_PROTOCOL_ID",
    "FINALIZER_PROTOCOL_KIND",
    "SELECTED_SOURCE_COUNT",
    "SELECTION_CONTRACT",
    "SELECTION_FILENAME",
    "SELECTION_KIND",
    "V14SelectionDisposition",
    "V14SelectionLedger",
    "build_v14_selection_ledger",
    "load_v14_source_finalizer_protocol",
    "validate_v14_selection_ledger",
    "write_v14_selection_ledger",
]
