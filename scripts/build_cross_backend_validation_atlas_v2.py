#!/usr/bin/env python3
"""Extend validation atlas v1 with frozen native-continuum source evidence."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin.query_competence_certificate_v1 import SimulatorQueryScopeV1
from bayesian_phystwin.simulator_validation_atlas_v1 import (
    SimulatorValidationAtlasV1,
    SimulatorValidationEntryV1,
    ValidationEvidenceReferenceV1,
    ValidationStageAssessmentV1,
    save_simulator_validation_atlas,
)
from scripts.build_cross_backend_validation_atlas_v1 import build_atlas as build_v1

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"

NATIVE_EVIDENCE = {
    "mujoco": Path(
        "results/sota/diagnostics/mujoco_flex_source_gate_v1/failure.json"
    ),
    "jax_fem": Path(
        "results/sota/diagnostics/jax_fem_zebra_source_value_v2/failure.json"
    ),
    "sofa": Path(
        "results/sota/diagnostics/sofa_fem_zebra_source_value_v3/failure.json"
    ),
}
NATIVE_EVIDENCE_SHA256 = {
    "mujoco": "afd35892cdf3c1e5e0e74b19b9902d5365ccbbc210b7891fa314383dfbc0cd44",
    "jax_fem": "73838e359bdab8c7269cdd518be1d3e84e51b6e74ceec3fe66439d0be5d3d55b",
    "sofa": "b8f841c2b76e244b8d08217363e8dd3e1a63a642580fae4a3b946be0c26f7ff6",
}
NATIVE_EVIDENCE_COMMITS = {
    "mujoco": "36106ea74438d248c14f5520b61f16102164961e",
    "jax_fem": "3bd7c3c3b57ee6b8496fac8f8966f0d49bad667b",
    "sofa": "2b323910ba5352245c928fa68dd125c616666dcf",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_id(kind: str, **values: object) -> str:
    return cast(
        str,
        content_id(
            {
                "schema": "bayesian_phystwin.validation_scope_component",
                "schema_version": 1,
                "kind": kind,
                **values,
            }
        ),
    )


def _expect(
    value: Mapping[str, Any], key: str, expected: object, *, label: str
) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{label} field {key!r} changed")


def _load_native_evidence() -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for name, relative in NATIVE_EVIDENCE.items():
        if _sha256(ROOT / relative) != NATIVE_EVIDENCE_SHA256[name]:
            raise ValueError(f"native continuum evidence changed: {relative}")
        result[name] = cast(
            Mapping[str, Any],
            load_strict_json_object(ROOT / relative, label=f"{name} evidence"),
        )
    return result


def _reference(name: str) -> ValidationEvidenceReferenceV1:
    return ValidationEvidenceReferenceV1(
        repository=REPOSITORY,
        commit=NATIVE_EVIDENCE_COMMITS[name],
        path=NATIVE_EVIDENCE[name].as_posix(),
        file_sha256=NATIVE_EVIDENCE_SHA256[name],
    )


def _assessed(
    status: str,
    reason: str,
    *evidence: ValidationEvidenceReferenceV1,
    metrics: Mapping[str, Any] | None = None,
) -> ValidationStageAssessmentV1:
    return ValidationStageAssessmentV1(
        status=cast(Any, status),
        reason=reason,
        evidence=evidence,
        metrics={} if metrics is None else metrics,
    )


def _not_evaluated(reason: str) -> ValidationStageAssessmentV1:
    return _assessed("not_evaluated", reason)


def _not_applicable() -> ValidationStageAssessmentV1:
    return _assessed(
        "not_applicable",
        "the source qualification contains factual actions, not an action bank",
    )


def _phystwin_scope(
    *,
    backend: str,
    protocol_ids: list[str],
    source_group_count: int,
    metric: str,
) -> SimulatorQueryScopeV1:
    return SimulatorQueryScopeV1(
        simulator_id=_semantic_id(
            "simulator", public_dataset="PhysTwin", backend=backend
        ),
        task_id=_semantic_id(
            "task",
            public_dataset="PhysTwin",
            task="registered-source-physical-qualification",
        ),
        observation_policy_id=_semantic_id(
            "observation-policy",
            protocol_ids=protocol_ids,
            description="target-blind physical qualification before outcome scoring",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            protocol_ids=protocol_ids,
            description="registered released factual source actions",
        ),
        metric_id=_semantic_id("metric", description=metric),
        world_distribution_id=_semantic_id(
            "world-distribution",
            public_dataset="PhysTwin",
            split="already-open-public-source",
            source_group_count=source_group_count,
            protocol_ids=protocol_ids,
        ),
        statistical_unit="predeclared public PhysTwin source interaction",
        metadata={
            "public_dataset": "PhysTwin",
            "backend": backend,
            "source_group_count": source_group_count,
        },
    )


def _mujoco_entry(record: Mapping[str, Any]) -> SimulatorValidationEntryV1:
    _expect(
        record,
        "schema",
        "bayesian-phystwin.mujoco-flex-source-gate-failure-v1",
        label="MuJoCo",
    )
    source = cast(Mapping[str, Any], record["source"])
    _expect(source, "source_object_outcomes_read", False, label="MuJoCo source")
    _expect(
        source,
        "target_or_held_out_artifact_read",
        False,
        label="MuJoCo source",
    )
    preflight = cast(Mapping[str, Any], record["synthetic_preflight"])
    _expect(preflight, "passed", True, label="MuJoCo preflight")
    failure = cast(Mapping[str, Any], record["failure"])
    _expect(failure, "stage", "source-physical-orientation-gate", label="MuJoCo")
    _expect(failure, "terminal", True, label="MuJoCo")
    _expect(failure, "retry_performed", False, label="MuJoCo")
    decision = cast(Mapping[str, Any], record["decision"])
    _expect(decision, "source_value_authorized", False, label="MuJoCo decision")
    _expect(decision, "target_evaluation_authorized", False, label="MuJoCo decision")
    reference = _reference("mujoco")
    stopped = "native source-orientation failure stopped horizon qualification"
    return SimulatorValidationEntryV1(
        backend_key="mujoco-flex-source-v1",
        display_name="MuJoCo Flex",
        dataset="PhysTwin",
        query_scope=_phystwin_scope(
            backend="MuJoCo Flex 3.9 volumetric",
            protocol_ids=[str(record["implementation"]["base_revision"])],
            source_group_count=1,
            metric="target-free deformation orientation and constraint admissibility",
        ),
        independent_group_count=1,
        stages={
            "runtime_execution": _assessed(
                "passed", "registered MuJoCo source invocation completed", reference
            ),
            "native_qualification": _assessed(
                "failed",
                "source replay crossed the registered hard orientation floor",
                reference,
                metrics={
                    "observed_minimum_deformation_determinant": failure[
                        "observed_minimum_deformation_determinant"
                    ],
                    "required_minimum_deformation_determinant": failure[
                        "required_minimum_deformation_determinant"
                    ],
                },
            ),
            "full_horizon_qualification": _not_evaluated(stopped),
            "decision_headroom": _not_applicable(),
            "source_value": _not_evaluated(stopped),
            "prospective_value": _not_evaluated(stopped),
        },
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="source-physical-orientation-gate-failed",
        metadata={
            "public_source_outcome_opened": False,
            "target_outcome_opened": False,
            "retry_performed": False,
            "synthetic_preflight_passed": True,
        },
    )


def _qualified_horizon_failure_entry(
    *,
    name: str,
    display_name: str,
    backend_key: str,
    record: Mapping[str, Any],
    expected_schema: str,
    expected_failure_stage: str,
    source_group_count: int,
) -> SimulatorValidationEntryV1:
    _expect(record, "schema", expected_schema, label=display_name)
    _expect(
        record,
        "status",
        "source-physical-rejection-before-outcome-access",
        label=display_name,
    )
    qualification = cast(Mapping[str, Any], record["qualification"])
    _expect(qualification, "qualified", True, label=f"{display_name} qualification")
    failure = cast(Mapping[str, Any], record["failure"])
    _expect(failure, "stage", expected_failure_stage, label=display_name)
    _expect(
        failure,
        "source_physical_admission_failure",
        True,
        label=display_name,
    )
    boundary = cast(Mapping[str, Any], record["information_boundary"])
    for key in (
        "prefix_outcomes_read",
        "future_outcomes_read",
        "target_or_held_out_artifact_read",
        "dlo4_dlo5_access",
        "held_v8_access",
    ):
        _expect(boundary, key, False, label=f"{display_name} boundary")
    decision = cast(Mapping[str, Any], record["decision"])
    _expect(decision, "candidate_admitted", False, label=f"{display_name} decision")
    _expect(
        decision,
        "exact_incumbent_fallback_retained",
        True,
        label=f"{display_name} decision",
    )
    _expect(decision, "retry_authorized", False, label=f"{display_name} decision")
    implementation = cast(Mapping[str, Any], record["implementation"])
    reference = _reference(name)
    stopped = "full-horizon physical rejection kept source outcomes sealed"
    return SimulatorValidationEntryV1(
        backend_key=backend_key,
        display_name=display_name,
        dataset="PhysTwin",
        query_scope=_phystwin_scope(
            backend=display_name,
            protocol_ids=[str(implementation["protocol_sha256"])],
            source_group_count=source_group_count,
            metric="target-free full-horizon orientation qualification",
        ),
        independent_group_count=source_group_count,
        stages={
            "runtime_execution": _assessed(
                "passed", f"registered {display_name} invocation executed", reference
            ),
            "native_qualification": _assessed(
                "passed",
                "frozen source-physics qualification passed before value generation",
                reference,
                metrics={
                    "qualification_artifact_id": qualification["artifact_id"],
                    "qualification_artifact_sha256": qualification["artifact_sha256"],
                },
            ),
            "full_horizon_qualification": _assessed(
                "failed",
                "registered prediction generation violated its hard orientation floor",
                reference,
                metrics={
                    "failure_type": failure["exception_type"],
                    "failure_stage": failure["stage"],
                },
            ),
            "decision_headroom": _not_applicable(),
            "source_value": _not_evaluated(stopped),
            "prospective_value": _not_evaluated(stopped),
        },
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="full-horizon-source-physical-rejection",
        metadata={
            "public_source_outcome_opened": False,
            "target_outcome_opened": False,
            "retry_authorized": False,
            "source_group_count": source_group_count,
        },
    )


def build_atlas() -> SimulatorValidationAtlasV1:
    base = build_v1()
    evidence = _load_native_evidence()
    additions = (
        _mujoco_entry(evidence["mujoco"]),
        _qualified_horizon_failure_entry(
            name="jax_fem",
            display_name="JAX-FEM v2",
            backend_key="jax-fem-hyperelastic-v2",
            record=evidence["jax_fem"],
            expected_schema="bayesian-phystwin.jax-fem-hyperelastic-source-value-failure-v2",
            expected_failure_stage="frozen-native-prediction-grid",
            source_group_count=2,
        ),
        _qualified_horizon_failure_entry(
            name="sofa",
            display_name="SOFA FEM v3",
            backend_key="sofa-fem-canonical-v3",
            record=evidence["sofa"],
            expected_schema="bayesian-phystwin.sofa-fem-canonical-source-value-failure-v3",
            expected_failure_stage="frozen-native-full-horizon-prediction",
            source_group_count=2,
        ),
    )
    metadata = dict(base.metadata)
    bound_sources = dict(cast(Mapping[str, str], metadata["bound_source_sha256"]))
    bound_sources.update(
        {
            path.as_posix(): NATIVE_EVIDENCE_SHA256[name]
            for name, path in NATIVE_EVIDENCE.items()
        }
    )
    metadata.update(
        {
            "public_datasets": ["DLO-Lab", "PhysTwin", "RGBench"],
            "parent_atlas_artifact_id": base.artifact_id,
            "native_continuum_entry_count": len(additions),
            "native_continuum_source_outcomes_read": False,
            "bound_source_sha256": bound_sources,
        }
    )
    atlas = SimulatorValidationAtlasV1(
        entries=(*base.entries, *additions),
        metadata=metadata,
    )
    if (
        len(atlas.entries) != 12
        or len({item.backend_key for item in atlas.entries}) != 8
        or len({item.dataset for item in atlas.entries}) != 3
    ):
        raise ValueError("cross-backend atlas v2 roster changed")
    return atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atlas = build_atlas()
    save_simulator_validation_atlas(args.output, atlas)
    print(f"atlas_id={atlas.artifact_id}")


if __name__ == "__main__":
    main()
