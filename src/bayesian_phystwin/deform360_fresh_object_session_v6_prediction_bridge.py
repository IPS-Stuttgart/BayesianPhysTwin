"""Target-blind bridge from the sealed v5 source panel to v6 prediction seals.

The protected v5 source workflow publishes 100 nested forecasts before any
opened development suffix is scored. The v6 source gate consumes one held-out
prediction seal per physical object-session and a candidate-specific covariance
portfolio. This module joins those two boundaries without reading outcomes:

* the exact held-out v5 B0/B1 prediction identities are retained;
* one independently content-addressed candidate panel supplies D1 and VT1
  prediction, guard, covariance, and interval identities;
* all four available VT1 covariance variants must share one point prediction,
  source fit, risk score, threshold, and guard decision; and
* exactly ten rebuilt v6 seals are published before source suffix access.

The bridge is custody infrastructure. It neither computes candidate forecasts
nor authorizes source scoring, fresh-target selection, or target access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .deform360_fresh_object_session_source_v6 import (
    AMENDMENT_ID,
    B0,
    B1,
    D1_NATIVE,
    POLICY_ID,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
    build_deform360_v6_source_prediction_batch,
    build_deform360_v6_source_prediction_seal,
    validate_deform360_v6_source_selection,
)
from .deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)

SOURCE_EXECUTION_AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-source-prediction-execution"
)
SOURCE_EXECUTION_AMENDMENT_ID: Final = (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
)
CANDIDATE_PANEL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-candidate-panel"
)
BRIDGE_RECEIPT_SCHEMA: Final = (
    "bayesian-phystwin.deform360-fresh-object-session-v6-prediction-bridge-receipt"
)
SCHEMA_VERSION: Final = 1

_BRIDGE_BOUNDARY = {
    "source_suffix_opened": False,
    "v5_terminal_outcome_used": False,
    "v5_confirmation_payloads_used": False,
    "v5_confirmation_outcomes_used": False,
    "v6_target_selected": False,
    "v6_target_payloads_used": False,
    "v6_target_outcomes_used": False,
    "future_object_observations_used_for_prediction": False,
    "human_selection_used": False,
    "replacement_allowed": False,
}
_PANEL_FIELDS = frozenset(
    {
        "amendment_id",
        "candidate_panel_id",
        "episode_id",
        "implementation_revision",
        "information_boundary",
        "object_id",
        "policy_id",
        "schema",
        "schema_version",
        "selection_artifact_sha256",
        "source_artifacts",
        "source_execution_amendment_id",
        "stratum",
        "v5_execution_lock_id",
        "v5_held_out_prediction_seal_id",
        "v5_prediction_batch_id",
        "variants",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "amendment_id",
        "bridge_receipt_id",
        "bridge_revision",
        "candidate_panel_ids",
        "implementation_revision",
        "information_boundary",
        "policy_id",
        "record_count",
        "schema",
        "schema_version",
        "selection_artifact_sha256",
        "source_execution_amendment_id",
        "v5_execution_lock_id",
        "v5_prediction_batch_id",
        "v6_prediction_batch_id",
        "v6_prediction_seal_ids",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _content_identity(value: Mapping[str, Any], *, id_field: str, name: str) -> None:
    declared = sha256_digest(value.get(id_field), name=id_field)
    identity = {key: item for key, item in value.items() if key != id_field}
    if declared != content_id(identity):
        raise ValueError(f"{name} content identity changed")


def load_deform360_v6_source_execution_amendment(
    path: str | Path,
    *,
    v5_execution_lock_id: str,
) -> Mapping[str, Any]:
    """Load the reviewed-main, target-closed v6 source execution amendment."""

    amendment = load_strict_json_object(
        path,
        label="Deform360 v6 source execution amendment",
    )
    if (
        amendment.get("schema") != SOURCE_EXECUTION_AMENDMENT_SCHEMA
        or amendment.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("v6 source execution amendment schema changed")
    _content_identity(
        amendment,
        id_field="amendment_id",
        name="v6 source execution amendment",
    )
    if amendment.get("amendment_id") != SOURCE_EXECUTION_AMENDMENT_ID:
        raise ValueError("v6 source execution amendment identity changed")
    if amendment.get("v5_source_execution_lock_id") != v5_execution_lock_id:
        raise ValueError("v6 source execution amendment binds another v5 lock")
    boundary = _mapping(amendment.get("information_boundary"), name="boundary")
    forbidden = (
        "development_suffix_opened",
        "v5_confirmation_payloads_opened",
        "v5_confirmation_outcomes_used",
        "v6_fresh_target_selected",
        "v6_target_payloads_opened",
        "v6_target_outcomes_used",
    )
    if any(boundary.get(key) is not False for key in forbidden):
        raise ValueError("v6 source execution amendment crosses its boundary")
    execution = _mapping(amendment.get("execution"), name="execution")
    if (
        execution.get("source_prediction_batch_required_before_suffix_access")
        is not True
        or execution.get("v5_nested_prediction_record_count") != 100
        or execution.get("v6_source_prediction_unit_count") != 10
    ):
        raise ValueError("v6 source execution amendment counts changed")
    return amendment


def _held_out_v5_records(
    batch: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(batch.get("records"), name="v5 records"):
        row = _mapping(raw, name="v5 record")
        if row.get("record_role") != "held_out":
            continue
        object_id = cast(str, row.get("object_id"))
        if row.get("outer_held_out_object_id") != object_id:
            raise ValueError("v5 held-out record has inconsistent outer identity")
        if object_id in records:
            raise ValueError("v5 prediction batch repeats a held-out unit")
        records[object_id] = row
    if len(records) != 10:
        raise ValueError("v5 prediction batch must contain ten held-out records")
    return records


def _validate_shared_vt1_semantics(variants: Mapping[str, Any]) -> None:
    available = [
        _mapping(variants[variant_id], name=variant_id)
        for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH)
        if _mapping(variants[variant_id], name=variant_id).get("available") is True
    ]
    if not available:
        return
    shared_fields = (
        "accepted",
        "prediction_artifact_id",
        "fit_artifact_id",
        "fit_object_ids",
        "guard_artifact_id",
        "risk_score",
        "guard_threshold",
    )
    reference = available[0]
    for row in available[1:]:
        if any(row.get(field) != reference.get(field) for field in shared_fields):
            raise ValueError(
                "available VT1 covariance variants must share one mean, fit, and guard"
            )


def build_deform360_v6_source_candidate_panel(
    *,
    policy: Mapping[str, Any],
    covariance_amendment: Mapping[str, Any],
    source_execution_amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
    v5_execution_lock: Mapping[str, Any],
    v5_prediction_batch: Mapping[str, Any],
    implementation_revision: str,
    object_id: str,
    variants: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one six-variant v6 candidate panel to an exact v5 held-out record."""

    lock_id = sha256_digest(
        v5_execution_lock.get("execution_lock_id"),
        name="v5_execution_lock_id",
    )
    if source_execution_amendment.get("amendment_id") != SOURCE_EXECUTION_AMENDMENT_ID:
        raise ValueError("candidate panel uses another source execution amendment")
    if source_execution_amendment.get("v5_source_execution_lock_id") != lock_id:
        raise ValueError("candidate panel source amendment binds another v5 lock")
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        v5_prediction_batch,
        v5_execution_lock,
    )
    source_selection, cohort = validate_deform360_v6_source_selection(
        selection,
        policy,
    )
    if covariance_amendment.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("candidate panel uses another covariance amendment")
    unit_id = str(object_id)
    if unit_id not in cohort:
        raise ValueError("candidate panel uses an unregistered source unit")
    revision = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    if revision != batch.get("implementation_revision"):
        raise ValueError("candidate panel revision differs from the sealed v5 batch")
    held_out = _held_out_v5_records(batch)
    if set(held_out) != set(cohort):
        raise ValueError("v5 held-out roster differs from the v6 source selection")
    v5_record = held_out[unit_id]

    # Reuse the existing v6 seal constructor as the authoritative variant validator.
    provisional = build_deform360_v6_source_prediction_seal(
        policy=policy,
        amendment=covariance_amendment,
        selection=source_selection,
        implementation_revision=revision,
        object_id=unit_id,
        variants=variants,
        source_artifacts=source_artifacts,
    )
    normalized_variants = cast(dict[str, Any], provisional["variants"])
    v5_methods = _mapping(v5_record.get("methods"), name="v5 held-out methods")
    baseline_mapping = {
        B0: "B0_physical_fallback",
        B1: "B1_last_causal_residual",
    }
    for variant_id, method_id in baseline_mapping.items():
        method = _mapping(v5_methods.get(method_id), name=method_id)
        variant = _mapping(normalized_variants[variant_id], name=variant_id)
        if variant.get("prediction_artifact_id") != method.get("artifact_id"):
            raise ValueError(
                f"{variant_id} prediction identity differs from the sealed v5 held-out method"
            )
    if (
        _mapping(normalized_variants[D1_NATIVE], name=D1_NATIVE).get("available")
        is not True
    ):
        raise ValueError("the frozen D1 native covariance variant must be available")
    _validate_shared_vt1_semantics(normalized_variants)

    episode_id, stratum = cohort[unit_id]
    identity: dict[str, Any] = {
        "schema": CANDIDATE_PANEL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "source_execution_amendment_id": SOURCE_EXECUTION_AMENDMENT_ID,
        "selection_artifact_sha256": source_selection["selection_artifact_sha256"],
        "v5_execution_lock_id": lock_id,
        "v5_prediction_batch_id": batch["prediction_batch_id"],
        "v5_held_out_prediction_seal_id": v5_record["seal_id"],
        "implementation_revision": revision,
        "object_id": unit_id,
        "episode_id": episode_id,
        "stratum": stratum,
        "variants": normalized_variants,
        "source_artifacts": plain_json(
            source_artifact_mapping(source_artifacts, name="source_artifacts")
        ),
        "information_boundary": dict(_BRIDGE_BOUNDARY),
    }
    return {**identity, "candidate_panel_id": content_id(identity)}


def validate_deform360_v6_source_candidate_panel(
    value: object,
    *,
    policy: Mapping[str, Any],
    covariance_amendment: Mapping[str, Any],
    source_execution_amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
    v5_execution_lock: Mapping[str, Any],
    v5_prediction_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically rebuild one target-blind candidate panel."""

    payload = _mapping(value, name="candidate panel")
    require_exact_fields(payload, expected=_PANEL_FIELDS, name="candidate panel")
    if payload.get("schema") != CANDIDATE_PANEL_SCHEMA:
        raise ValueError("candidate panel schema changed")
    if payload.get("information_boundary") != _BRIDGE_BOUNDARY:
        raise ValueError("candidate panel crossed its information boundary")
    rebuilt = build_deform360_v6_source_candidate_panel(
        policy=policy,
        covariance_amendment=covariance_amendment,
        source_execution_amendment=source_execution_amendment,
        selection=selection,
        v5_execution_lock=v5_execution_lock,
        v5_prediction_batch=v5_prediction_batch,
        implementation_revision=cast(str, payload.get("implementation_revision")),
        object_id=cast(str, payload.get("object_id")),
        variants=cast(Mapping[str, Any], payload.get("variants")),
        source_artifacts=cast(Mapping[str, str], payload.get("source_artifacts")),
    )
    if plain_json(payload) != rebuilt:
        raise ValueError("candidate panel content changed")
    return rebuilt


def bridge_deform360_v6_source_prediction_batch(
    *,
    policy: Mapping[str, Any],
    covariance_amendment: Mapping[str, Any],
    source_execution_amendment: Mapping[str, Any],
    selection: Mapping[str, Any],
    v5_execution_lock: Mapping[str, Any],
    v5_prediction_batch: Mapping[str, Any],
    candidate_panels: Sequence[Mapping[str, Any]],
    bridge_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build exactly ten v6 seals and one prediction batch without outcomes."""

    if len(candidate_panels) != 10:
        raise ValueError("v6 prediction bridge requires exactly ten candidate panels")
    validated = [
        validate_deform360_v6_source_candidate_panel(
            panel,
            policy=policy,
            covariance_amendment=covariance_amendment,
            source_execution_amendment=source_execution_amendment,
            selection=selection,
            v5_execution_lock=v5_execution_lock,
            v5_prediction_batch=v5_prediction_batch,
        )
        for panel in candidate_panels
    ]
    by_object: dict[str, dict[str, Any]] = {}
    for panel in validated:
        object_id = cast(str, panel["object_id"])
        if object_id in by_object:
            raise ValueError("v6 prediction bridge repeats a candidate panel")
        by_object[object_id] = panel
    _, cohort = validate_deform360_v6_source_selection(selection, policy)
    if set(by_object) != set(cohort):
        raise ValueError("v6 prediction bridge has an incomplete source roster")
    revisions = {cast(str, panel["implementation_revision"]) for panel in validated}
    if len(revisions) != 1:
        raise ValueError("v6 prediction bridge mixes candidate revisions")

    v5_batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        v5_prediction_batch,
        v5_execution_lock,
    )
    seals: list[dict[str, Any]] = []
    for object_id in sorted(by_object):
        panel = by_object[object_id]
        lineage = {
            **cast(dict[str, str], panel["source_artifacts"]),
            f"lineage/v6-candidate-panel/{object_id}.json": cast(
                str, panel["candidate_panel_id"]
            ),
            "lineage/v5-source-prediction-batch.json": cast(
                str, panel["v5_prediction_batch_id"]
            ),
            f"lineage/v5-held-out-seal/{object_id}.json": cast(
                str, panel["v5_held_out_prediction_seal_id"]
            ),
        }
        seal = build_deform360_v6_source_prediction_seal(
            policy=policy,
            amendment=covariance_amendment,
            selection=selection,
            implementation_revision=cast(str, panel["implementation_revision"]),
            object_id=object_id,
            variants=cast(Mapping[str, Any], panel["variants"]),
            source_artifacts=lineage,
        )
        seals.append(seal)
    batch = build_deform360_v6_source_prediction_batch(
        seals,
        policy=policy,
        amendment=covariance_amendment,
        selection=selection,
    )
    lock_id = sha256_digest(
        v5_execution_lock.get("execution_lock_id"),
        name="v5_execution_lock_id",
    )
    receipt_identity: dict[str, Any] = {
        "schema": BRIDGE_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "amendment_id": AMENDMENT_ID,
        "source_execution_amendment_id": SOURCE_EXECUTION_AMENDMENT_ID,
        "selection_artifact_sha256": selection["selection_artifact_sha256"],
        "v5_execution_lock_id": lock_id,
        "v5_prediction_batch_id": v5_batch["prediction_batch_id"],
        "implementation_revision": next(iter(revisions)),
        "bridge_revision": exact_revision(bridge_revision, name="bridge_revision"),
        "record_count": 10,
        "candidate_panel_ids": [
            cast(str, by_object[object_id]["candidate_panel_id"])
            for object_id in sorted(by_object)
        ],
        "v6_prediction_seal_ids": [cast(str, seal["seal_id"]) for seal in seals],
        "v6_prediction_batch_id": batch["prediction_batch_id"],
        "information_boundary": dict(_BRIDGE_BOUNDARY),
    }
    receipt = {
        **receipt_identity,
        "bridge_receipt_id": content_id(receipt_identity),
    }
    return seals, batch, receipt


def validate_deform360_v6_prediction_bridge_receipt(value: object) -> dict[str, Any]:
    """Validate one content-addressed bridge receipt without reopening inputs."""

    payload = _mapping(value, name="bridge receipt")
    require_exact_fields(payload, expected=_RECEIPT_FIELDS, name="bridge receipt")
    if payload.get("schema") != BRIDGE_RECEIPT_SCHEMA:
        raise ValueError("bridge receipt schema changed")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("bridge receipt version changed")
    if payload.get("information_boundary") != _BRIDGE_BOUNDARY:
        raise ValueError("bridge receipt crossed its information boundary")
    if payload.get("record_count") != 10:
        raise ValueError("bridge receipt record count changed")
    panel_ids = _sequence(
        payload.get("candidate_panel_ids"), name="candidate_panel_ids"
    )
    seal_ids = _sequence(
        payload.get("v6_prediction_seal_ids"),
        name="v6_prediction_seal_ids",
    )
    if len(panel_ids) != 10 or len(seal_ids) != 10:
        raise ValueError("bridge receipt roster is incomplete")
    for index, value_ in enumerate((*panel_ids, *seal_ids)):
        sha256_digest(value_, name=f"bridge receipt digest {index}")
    exact_revision(
        payload.get("implementation_revision"), name="implementation_revision"
    )
    exact_revision(payload.get("bridge_revision"), name="bridge_revision")
    _content_identity(
        payload,
        id_field="bridge_receipt_id",
        name="bridge receipt",
    )
    return cast(dict[str, Any], plain_json(payload))


def publish_deform360_v6_prediction_bridge(
    *,
    seals: Sequence[Mapping[str, Any]],
    batch: Mapping[str, Any],
    receipt: Mapping[str, Any],
    output_directory: str | Path,
) -> Path:
    """Publish the exact ten-seal bridge output without replacement."""

    if len(seals) != 10:
        raise ValueError("bridge publication requires ten seals")
    validate_deform360_v6_prediction_bridge_receipt(receipt)
    output = Path(output_directory)
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError("bridge output must be an ordinary directory")
    output.mkdir(parents=True, exist_ok=True)
    seal_root = output / "source-seals"
    seal_root.mkdir(parents=True, exist_ok=True)
    for seal in seals:
        object_id = cast(str, seal["object_id"])
        write_atomic_json(
            seal,
            seal_root / f"{object_id}.json",
            overwrite=False,
        )
    write_atomic_json(
        batch,
        output / "source-prediction-batch.json",
        overwrite=False,
    )
    write_atomic_json(
        receipt,
        output / "bridge-receipt.json",
        overwrite=False,
    )
    return output


__all__ = [
    "BRIDGE_RECEIPT_SCHEMA",
    "CANDIDATE_PANEL_SCHEMA",
    "SOURCE_EXECUTION_AMENDMENT_ID",
    "SOURCE_EXECUTION_AMENDMENT_SCHEMA",
    "bridge_deform360_v6_source_prediction_batch",
    "build_deform360_v6_source_candidate_panel",
    "load_deform360_v6_source_execution_amendment",
    "publish_deform360_v6_prediction_bridge",
    "validate_deform360_v6_prediction_bridge_receipt",
    "validate_deform360_v6_source_candidate_panel",
]
