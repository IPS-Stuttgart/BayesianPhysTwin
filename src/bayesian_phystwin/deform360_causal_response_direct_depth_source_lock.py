"""Fresh-object source lock for V14 adaptive causal direct depth."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
)
from .deform360_causal_response_direct_depth_preflight import (
    AdaptiveDirectDepthSourcePreflightConfigV14,
    AdaptiveDirectDepthSourcePreflightV14,
    validate_adaptive_direct_depth_source_preflight_v14,
)
from .deform360_causal_response_direct_depth_synthetic import (
    validate_adaptive_direct_depth_synthetic_v14,
)
from .deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)

CONTRACT = "deform360-causal-response-direct-depth-source-lock-v14"
PROTOCOL_ID = "deform360-causal-response-direct-depth-v14-source"
EXCLUSION_OWNER = "bayesian-phystwin-v14-fresh-source-owner"


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
        b"deform360-causal-response-direct-depth-source-lock-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AdaptiveDirectDepthSourceCaseV14:
    """One fresh source episode and its pre-outcome provenance."""

    case_id: str
    case_hash: str
    object_hash: str
    metadata_sha256: str
    source_preflight_sha256: str
    carrier_artifact_sha256: str
    carrier_arm: str
    fold: int

    def __post_init__(self) -> None:
        _require(bool(self.case_id.strip()), "V14 source case ID is empty")
        _require(
            all(
                _valid_digest(value)
                for value in (
                    self.case_hash,
                    self.object_hash,
                    self.metadata_sha256,
                    self.source_preflight_sha256,
                    self.carrier_artifact_sha256,
                )
            ),
            "V14 source case digest is invalid",
        )
        _require(
            self.carrier_arm in {STRICT_ARM, INFLATED_FALLBACK_ARM},
            "V14 source carrier arm is invalid",
        )
        _require(0 <= self.fold < 3, "V14 source fold is invalid")


@dataclass(frozen=True)
class AdaptiveDirectDepthSourceLockV14:
    """Immutable twelve-object V14 source panel."""

    repository_revision: str
    method_config_sha256: str
    exclusion_manifest_sha256: str
    exclusion_manifest_file_sha256: str
    synthetic_control_result_sha256: str
    synthetic_control_file_sha256: str
    excluded_object_hashes: tuple[str, ...]
    cases: tuple[AdaptiveDirectDepthSourceCaseV14, ...]
    selection_metadata_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            len(self.repository_revision) == 40
            and all(
                character in "0123456789abcdef"
                for character in self.repository_revision
            ),
            "V14 repository revision is invalid",
        )
        _require(
            all(
                _valid_digest(value)
                for value in (
                    self.method_config_sha256,
                    self.exclusion_manifest_sha256,
                    self.exclusion_manifest_file_sha256,
                    self.synthetic_control_result_sha256,
                    self.synthetic_control_file_sha256,
                    self.selection_metadata_sha256,
                    self.artifact_sha256,
                )
            ),
            "V14 source lock digest is invalid",
        )
        excluded = tuple(sorted(set(self.excluded_object_hashes)))
        _require(
            excluded == self.excluded_object_hashes
            and all(_valid_digest(value) for value in excluded),
            "V14 exclusion hashes are invalid or duplicated",
        )
        _require(len(self.cases) == 12, "V14 source panel must contain 12 cases")
        _require(
            len({case.case_id for case in self.cases}) == 12
            and len({case.case_hash for case in self.cases}) == 12
            and len({case.object_hash for case in self.cases}) == 12,
            "V14 source cases or physical objects are duplicated",
        )
        _require(
            not {case.object_hash for case in self.cases}.intersection(excluded),
            "V14 source panel overlaps an excluded physical object",
        )
        _require(
            Counter(case.fold for case in self.cases)
            == Counter({0: 4, 1: 4, 2: 4}),
            "V14 cross-fit folds must contain four objects each",
        )
        object.__setattr__(self, "excluded_object_hashes", excluded)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360AdaptiveDirectDepthSourceLockV14",
            "contract": CONTRACT,
            "protocol_id": PROTOCOL_ID,
            "repository_revision": self.repository_revision,
            "method_config_sha256": self.method_config_sha256,
            "exclusion_manifest_sha256": self.exclusion_manifest_sha256,
            "exclusion_manifest_file_sha256": (
                self.exclusion_manifest_file_sha256
            ),
            "synthetic_control_result_sha256": (
                self.synthetic_control_result_sha256
            ),
            "synthetic_control_file_sha256": self.synthetic_control_file_sha256,
            "excluded_object_count": len(self.excluded_object_hashes),
            "excluded_object_hashes": list(self.excluded_object_hashes),
            "selection_metadata_sha256": self.selection_metadata_sha256,
            "cases": [asdict(case) for case in self.cases],
            "source_gate": {
                "physical_object_count": 12,
                "episode_count_per_object": 1,
                "cross_fit_fold_count": 3,
                "objects_per_fold": 4,
            },
            "information_boundary": {
                "selection_inputs": (
                    "released metadata plus outcome-blind source preflight and "
                    "frame-zero adaptive-carrier admission"
                ),
                "prefix_object_response_read": False,
                "future_object_payload_deserialized": False,
                "identity_or_metric_outcome_read": False,
                "target_cohort_read": False,
                "held_v8_object_ids_or_outcomes_read": False,
                "held_v8_hash_only_exclusion_used": True,
                "accepted_v14_source_preflight_required": True,
                "passed_v14_synthetic_controls_required": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def build_adaptive_direct_depth_source_lock_v14(
    cases: Iterable[AdaptiveDirectDepthSourceCaseV14],
    *,
    repository_revision: str,
    method_config_sha256: str,
    exclusion_manifest_path: str | Path,
    synthetic_control_result_path: str | Path,
    selection_metadata_sha256: str,
    source_preflights: Iterable[AdaptiveDirectDepthSourcePreflightV14],
) -> AdaptiveDirectDepthSourceLockV14:
    """Lock twelve source objects only after complete exclusion and preflight."""

    exclusion_path = Path(exclusion_manifest_path)
    exclusion = load_object_exclusion_manifest(exclusion_path)
    _require(
        exclusion["owner"] == EXCLUSION_OWNER,
        "V14 exclusion manifest owner changed",
    )
    synthetic_path = Path(synthetic_control_result_path)
    synthetic = validate_adaptive_direct_depth_synthetic_v14(synthetic_path)
    ordered_cases = tuple(
        sorted(cases, key=lambda case: (case.fold, case.object_hash, case.case_hash))
    )
    _require(
        not {case.object_hash for case in ordered_cases}.intersection(
            exclusion["object_hashes"]
        ),
        "V14 source panel overlaps an excluded physical object",
    )
    preflights = tuple(source_preflights)
    for preflight in preflights:
        validate_adaptive_direct_depth_source_preflight_v14(preflight)
    _require(
        len(preflights) == len(ordered_cases)
        and len({item.case_hash for item in preflights}) == len(preflights),
        "V14 source preflight set does not match the source panel",
    )
    by_case = {preflight.case_hash: preflight for preflight in preflights}
    for case in ordered_cases:
        preflight = by_case.get(case.case_hash)
        _require(
            preflight is not None
            and preflight.admitted
            and preflight.config == AdaptiveDirectDepthSourcePreflightConfigV14()
            and preflight.object_hash == case.object_hash
            and preflight.artifact_sha256 == case.source_preflight_sha256
            and preflight.carrier_artifact_sha256
            == case.carrier_artifact_sha256
            and preflight.carrier_arm == case.carrier_arm,
            "V14 source case lacks a matching accepted preflight",
        )
    provisional = AdaptiveDirectDepthSourceLockV14(
        repository_revision=repository_revision,
        method_config_sha256=method_config_sha256,
        exclusion_manifest_sha256=exclusion["exclusion_sha256"],
        exclusion_manifest_file_sha256=file_sha256(exclusion_path),
        synthetic_control_result_sha256=synthetic.artifact_sha256,
        synthetic_control_file_sha256=file_sha256(synthetic_path),
        excluded_object_hashes=tuple(exclusion["object_hashes"]),
        cases=ordered_cases,
        selection_metadata_sha256=selection_metadata_sha256,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = AdaptiveDirectDepthSourceLockV14(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "V14 source lock changed after construction",
    )
    return result


def write_adaptive_direct_depth_source_lock_v14(
    path: str | Path,
    lock: AdaptiveDirectDepthSourceLockV14,
) -> None:
    """Write one immutable V14 source lock."""

    output = Path(path)
    _require(not output.exists(), "V14 source lock already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            lock.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_adaptive_direct_depth_source_lock_v14(output)


def validate_adaptive_direct_depth_source_lock_v14(
    path: str | Path,
) -> AdaptiveDirectDepthSourceLockV14:
    """Validate V14 source cardinality, exclusions, folds, and checksum."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "Deform360AdaptiveDirectDepthSourceLockV14"
        and payload.get("contract") == CONTRACT
        and payload.get("protocol_id") == PROTOCOL_ID,
        "V14 source lock kind or contract changed",
    )
    lock = AdaptiveDirectDepthSourceLockV14(
        repository_revision=payload["repository_revision"],
        method_config_sha256=payload["method_config_sha256"],
        exclusion_manifest_sha256=payload["exclusion_manifest_sha256"],
        exclusion_manifest_file_sha256=payload[
            "exclusion_manifest_file_sha256"
        ],
        synthetic_control_result_sha256=payload[
            "synthetic_control_result_sha256"
        ],
        synthetic_control_file_sha256=payload["synthetic_control_file_sha256"],
        excluded_object_hashes=tuple(payload["excluded_object_hashes"]),
        cases=tuple(
            AdaptiveDirectDepthSourceCaseV14(**record)
            for record in payload["cases"]
        ),
        selection_metadata_sha256=payload["selection_metadata_sha256"],
        artifact_sha256=payload["artifact_sha256"],
    )
    _require(
        lock.descriptor() == payload
        and _canonical_sha256(payload) == lock.artifact_sha256,
        "V14 source lock checksum or descriptor changed",
    )
    return lock


__all__ = [
    "CONTRACT",
    "EXCLUSION_OWNER",
    "PROTOCOL_ID",
    "AdaptiveDirectDepthSourceCaseV14",
    "AdaptiveDirectDepthSourceLockV14",
    "build_adaptive_direct_depth_source_lock_v14",
    "validate_adaptive_direct_depth_source_lock_v14",
    "write_adaptive_direct_depth_source_lock_v14",
]
