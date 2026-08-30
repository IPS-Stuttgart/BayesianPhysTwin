#!/usr/bin/env python3
"""Build the public-source cross-backend simulator validation atlas v1."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin.query_competence_atlas_v2 import (
    QueryCompetenceStageV2,
    load_query_competence_atlas,
)
from bayesian_phystwin.query_competence_certificate_v1 import SimulatorQueryScopeV1
from bayesian_phystwin.simulator_validation_atlas_v1 import (
    STAGE_NAMES,
    SimulatorValidationAtlasV1,
    SimulatorValidationEntryV1,
    ValidationEvidenceReferenceV1,
    ValidationStageAssessmentV1,
    save_simulator_validation_atlas,
)

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"
ATLAS_IMPLEMENTATION = Path("src/bayesian_phystwin/simulator_validation_atlas_v1.py")
DLO_ATLAS = Path("results/source/dlolab_query_competence_atlas_v4/atlas.json")
EVIDENCE_ROOT = Path("results/source/cross_backend_validation_evidence_v1")

CAPSULES = {
    "arcsim_full": EVIDENCE_ROOT / "rgbbench_arcsim_full_horizon_v12_gate.json",
    "arcsim_source": EVIDENCE_ROOT / "rgbbench_arcsim_source_v13_result.json",
    "codim_native": EVIDENCE_ROOT / "rgbbench_codim_ipc_competence_v5_gate.json",
    "codim_full": EVIDENCE_ROOT / "rgbbench_codim_ipc_full_horizon_v7_gate.json",
    "libuipc_native": EVIDENCE_ROOT / "rgbbench_libuipc_competence_v3_summary.json",
    "libuipc_full": EVIDENCE_ROOT / "rgbbench_libuipc_ensemble_v4_summary.json",
    "matphys_runtime": EVIDENCE_ROOT
    / "rgbbench_matphys_technical_smoke_terminal_v1.json",
}

BOUND_FILES = {
    ATLAS_IMPLEMENTATION: (
        "24acd0a0f753f86622b2e13340991494bffb08d346d74993455adcfca07445a5"
    ),
    DLO_ATLAS: "45890333ac292c0cd2bb5620b1e2bb572e297bddb923d5e570a4cb098adfd94b",
    CAPSULES["arcsim_full"]: (
        "c5cbdd5c00a45fa9ad77606a25a08f2d35fb675718bf7573ad674ee2dd61ee43"
    ),
    CAPSULES["arcsim_source"]: (
        "bd3d3e3f1eaf822d1ac8317a77104ba5796065eae4fcc0d0b10a824a51821082"
    ),
    CAPSULES["codim_native"]: (
        "3e02c039129ea48a154405eac3ed1b9b55134e045a1d57174eb8c079de809f17"
    ),
    CAPSULES["codim_full"]: (
        "c8b44cba46030070a3462fb6c82df56d836c43dce25745b63af62d6479e41577"
    ),
    CAPSULES["libuipc_native"]: (
        "3e339c47d65a87c1ca53fd874607feaba8e61f0cc1f215cf3d65430bb8fc5662"
    ),
    CAPSULES["libuipc_full"]: (
        "9bf6afcc0fbe5da64ebe8ec1da3d19f37401617368ab54b781e48b4149f20fa4"
    ),
    CAPSULES["matphys_runtime"]: (
        "7e210e9ebb096056af1bef05801cad5fe7658e6639c547e493f9fc402f25d36d"
    ),
}

ORIGINAL_EVIDENCE = {
    "arcsim_full": (
        "373682f2d6ac360ffcb28e231312a5a88e37050b",
        "results/sota/rgbbench_arcsim_dirichlet_full_horizon_v12/gate.json",
    ),
    "arcsim_source": (
        "0680d2edb1a14647ffd92f2ddcb811fdc54a37d8",
        "results/sota/rgbbench_arcsim_dirichlet_source_v13/result.json",
    ),
    "codim_native": (
        "17b26b5404600b0ed64024f2e1c3280dc3c6da0c",
        "results/sota/rgbbench_codim_ipc_competence_v5/gate.json",
    ),
    "codim_full": (
        "4975c4b3b852e314d555c5d1c015c6294eee6ad2",
        "results/sota/rgbbench_codim_ipc_full_horizon_v7/gate.json",
    ),
    "libuipc_native": (
        "bcff26fd14960d9beb46cf70b1a4fb5ad7cbdb3a",
        "results/sota/rgbbench_libuipc_competence_v3/summary.json",
    ),
    "libuipc_full": (
        "d07d64605d0ba4cea98286e7027493d345c6c9fb",
        "results/sota/rgbbench_libuipc_ensemble_v4/summary.json",
    ),
    "matphys_runtime": (
        "e42e587ef945585392222f0df4f48169a175486f",
        "evidence/rgbench_matphys_technical_smoke_terminal_v1.json",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bound_files() -> None:
    for relative, expected in BOUND_FILES.items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"bound validation evidence changed: {relative}")


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


def _load_capsules() -> dict[str, Mapping[str, Any]]:
    return {
        name: cast(
            Mapping[str, Any],
            load_strict_json_object(ROOT / path, label=f"{name} evidence"),
        )
        for name, path in CAPSULES.items()
    }


def _reference(
    name: str, *, artifact_id: str | None = None
) -> ValidationEvidenceReferenceV1:
    commit, path = ORIGINAL_EVIDENCE[name]
    return ValidationEvidenceReferenceV1(
        repository=REPOSITORY,
        commit=commit,
        path=path,
        file_sha256=BOUND_FILES[CAPSULES[name]],
        artifact_id=artifact_id,
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


def _not_applicable(reason: str) -> ValidationStageAssessmentV1:
    return _assessed("not_applicable", reason)


def _dlo_reference(entry: QueryCompetenceStageV2) -> ValidationEvidenceReferenceV1:
    return ValidationEvidenceReferenceV1(
        repository=REPOSITORY,
        commit="225de3819b10d97c4984cde7caec023dd457c127",
        path=DLO_ATLAS.as_posix(),
        file_sha256=BOUND_FILES[DLO_ATLAS],
        artifact_id=str(entry.artifact_id),
    )


def _dlo_stage(
    status: str,
    reason: str,
    reference: ValidationEvidenceReferenceV1,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> ValidationStageAssessmentV1:
    if status in {"passed", "failed"}:
        return _assessed(status, reason, reference, metrics=metrics)
    return _assessed(status, reason)


def _dlo_entry(entry: QueryCompetenceStageV2) -> SimulatorValidationEntryV1:
    task = str(entry.query_scope.metadata["task"])
    reference = _dlo_reference(entry)
    native = entry.native_qualification
    if native == "passed":
        full_horizon = "passed"
        full_reason = "registered DLO-Lab query horizon completed under native checks"
    else:
        full_horizon = "not_evaluated"
        full_reason = "native qualification stopped before horizon value admission"
    stages = {
        "runtime_execution": _assessed(
            "passed", "registered public DLO-Lab runtime executed", reference
        ),
        "native_qualification": _dlo_stage(
            native,
            "immutable DLO-Lab atlas native qualification",
            reference,
            metrics=entry.metadata if native in {"passed", "failed"} else None,
        ),
        "full_horizon_qualification": _dlo_stage(full_horizon, full_reason, reference),
        "decision_headroom": _dlo_stage(
            entry.action_headroom,
            "immutable DLO-Lab action-headroom decision",
            reference,
        ),
        "source_value": _dlo_stage(
            entry.source_transfer,
            "immutable DLO-Lab source-transfer decision",
            reference,
        ),
        "prospective_value": _dlo_stage(
            entry.prospective_risk,
            "immutable DLO-Lab prospective value-and-harm decision",
            reference,
            metrics=(
                entry.metadata
                if entry.prospective_risk in {"passed", "failed"}
                else None
            ),
        ),
    }
    return SimulatorValidationEntryV1(
        backend_key="dlolab-public-simulator-v1",
        display_name=f"DLO-Lab {task}",
        dataset="DLO-Lab",
        query_scope=entry.query_scope,
        independent_group_count=entry.independent_group_count,
        stages=stages,
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=entry.protocol_frozen_before_outcomes,
        protected_target_data_read=entry.protected_data_read,
        new_recording_used=False,
        terminal_reason=entry.terminal_reason,
        metadata={
            "task": task,
            "source_query_competence_entry_id": entry.artifact_id,
            "source_query_competence_decision": entry.decision,
            "full_horizon_mapping": (
                "DLO-Lab native qualification covers the complete registered query "
                "rollout; no new outcome is inferred"
            ),
        },
    )


def _rgbbench_scope(
    *, backend: str, protocol_ids: list[str], metric: str
) -> SimulatorQueryScopeV1:
    return SimulatorQueryScopeV1(
        simulator_id=_semantic_id(
            "simulator", public_dataset="RGBench", backend=backend
        ),
        task_id=_semantic_id(
            "task", public_dataset="RGBench", task="green-tshirt-fling-source"
        ),
        observation_policy_id=_semantic_id(
            "observation-policy",
            protocol_ids=protocol_ids,
            description="target-free qualification then sealed source scoring",
        ),
        action_bank_id=_semantic_id(
            "action-bank", description="single released measured factual actuation"
        ),
        metric_id=_semantic_id("metric", description=metric),
        world_distribution_id=_semantic_id(
            "world-distribution",
            public_dataset="RGBench",
            source_case="green_tshirt/fling/01",
            protocol_ids=protocol_ids,
        ),
        statistical_unit="one predeclared public RGBench source case",
        metadata={
            "public_dataset": "RGBench",
            "backend": backend,
            "source_case_count": 1,
        },
    )


def _arcsim_entry(
    capsules: Mapping[str, Mapping[str, Any]],
) -> SimulatorValidationEntryV1:
    full = capsules["arcsim_full"]
    source = capsules["arcsim_source"]
    _expect(full, "qualification_gate_passed", True, label="ARCSim full horizon")
    _expect(full, "status", "passed", label="ARCSim full horizon")
    _expect(full, "point_cloud_coordinates_read", False, label="ARCSim full horizon")
    _expect(source, "decision", "close-arcsim-source-route", label="ARCSim source")
    gates = cast(Mapping[str, Any], source["gates"])
    expected_gates = {
        "all_passed": False,
        "beats_remeshed_physical_baseline": True,
        "beats_selected_dynamic_baseline": False,
        "published_improvement_at_least_5pct": False,
    }
    if dict(gates) != expected_gates:
        raise ValueError("ARCSim source gate roster changed")
    boundary = cast(Mapping[str, Any], source["information_boundary"])
    _expect(boundary, "target_outcomes_read", False, label="ARCSim source boundary")
    full_ref = _reference("arcsim_full")
    source_ref = _reference("arcsim_source")
    stages = {
        "runtime_execution": _assessed(
            "passed", "registered ARCSim executable completed", full_ref
        ),
        "native_qualification": _assessed(
            "passed",
            "two native replays were finite, identity-preserving, and byte-identical",
            full_ref,
            metrics={
                "maximum_pin_target_error_m": full["maximum_pin_target_error_m"],
                "minimum_mean_vertex_displacement_m": full[
                    "minimum_mean_vertex_displacement_m"
                ],
            },
        ),
        "full_horizon_qualification": _assessed(
            "passed",
            "both registered 1636-step replays passed every target-free check",
            full_ref,
        ),
        "decision_headroom": _not_applicable(
            "the RGBench source comparison has one factual rollout, not an action bank"
        ),
        "source_value": _assessed(
            "failed",
            "candidate improved the raw physical prior but failed both stronger comparator gates",
            source_ref,
            metrics={
                "candidate_real_to_sim_l1_m": source["arcsim_real_to_sim_l1_m"],
                "comparators": source["comparators"],
                "relative_improvements": source["relative_improvements"],
            },
        ),
        "prospective_value": _not_evaluated(
            "source gate failure closed calibration and target access"
        ),
    }
    return SimulatorValidationEntryV1(
        backend_key="arcsim-dirichlet-v13",
        display_name="ARCSim Dirichlet",
        dataset="RGBench",
        query_scope=_rgbbench_scope(
            backend="ARCSim Dirichlet",
            protocol_ids=[str(full["protocol_id"]), str(source["protocol_id"])],
            metric="one-sided real-to-sim L1 Chamfer against frozen comparators",
        ),
        independent_group_count=1,
        stages=stages,
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="source-comparator-gates-failed",
        metadata={
            "prediction_sealed_before_source_scoring": True,
            "public_source_outcome_opened": True,
            "target_outcome_opened": False,
            "source_case_count": 1,
        },
    )


def _codim_entry(
    capsules: Mapping[str, Mapping[str, Any]],
) -> SimulatorValidationEntryV1:
    native = capsules["codim_native"]
    full = capsules["codim_full"]
    _expect(native, "competence_gate_passed", True, label="Codim native")
    _expect(native, "status", "passed", label="Codim native")
    _expect(native, "point_cloud_coordinates_read", False, label="Codim native")
    _expect(full, "competence_gate_passed", False, label="Codim full horizon")
    _expect(full, "status", "technical_failure", label="Codim full horizon")
    _expect(full, "source_accuracy_outcomes_read", False, label="Codim full horizon")
    native_ref = _reference("codim_native")
    full_ref = _reference("codim_full")
    stages = {
        "runtime_execution": _assessed(
            "passed", "two isolated Codim-IPC native replays completed", native_ref
        ),
        "native_qualification": _assessed(
            "passed",
            "short native replays were finite, deterministic, and exactly actuated",
            native_ref,
            metrics={
                "maximum_pin_target_error_m": native["maximum_pin_target_error_m"],
                "minimum_mean_vertex_displacement_m": native[
                    "minimum_mean_vertex_displacement_m"
                ],
            },
        ),
        "full_horizon_qualification": _assessed(
            "failed",
            "nonlinear solve stalled on step 48 before the first full replay completed",
            full_ref,
            metrics={"replay_elapsed_seconds": full["replay_elapsed_seconds"]},
        ),
        "decision_headroom": _not_applicable(
            "the RGBench source comparison has one factual rollout, not an action bank"
        ),
        "source_value": _not_evaluated(
            "full-horizon failure kept source point-cloud outcomes sealed"
        ),
        "prospective_value": _not_evaluated(
            "full-horizon failure closed calibration and target access"
        ),
    }
    return SimulatorValidationEntryV1(
        backend_key="codim-ipc-v7",
        display_name="Codim-IPC",
        dataset="RGBench",
        query_scope=_rgbbench_scope(
            backend="Codim-IPC",
            protocol_ids=[str(native["protocol_id"]), str(full["protocol_id"])],
            metric="target-free numerical qualification before source accuracy",
        ),
        independent_group_count=1,
        stages=stages,
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="full-horizon-nonlinear-solver-stagnation",
        metadata={
            "public_source_outcome_opened": False,
            "target_outcome_opened": False,
            "source_case_count": 1,
        },
    )


def _libuipc_entry(
    capsules: Mapping[str, Mapping[str, Any]],
) -> SimulatorValidationEntryV1:
    native = capsules["libuipc_native"]
    full = capsules["libuipc_full"]
    _expect(
        native,
        "status",
        "source_competence_gate_failed",
        label="LibuIPC native",
    )
    native_gate = cast(Mapping[str, Any], native["final_gate"])
    _expect(native_gate, "both_complete", True, label="LibuIPC native gate")
    _expect(
        native_gate,
        "byte_identical_final_vertices",
        False,
        label="LibuIPC native gate",
    )
    _expect(
        full,
        "status",
        "target_free_qualification_failed",
        label="LibuIPC full horizon",
    )
    execution = cast(Mapping[str, Any], full["execution"])
    _expect(execution, "all_replays_complete", True, label="LibuIPC execution")
    boundary = cast(Mapping[str, Any], full["information_boundary"])
    _expect(boundary, "target_outcomes_read", False, label="LibuIPC boundary")
    native_ref = _reference("libuipc_native")
    full_ref = _reference("libuipc_full")
    stages = {
        "runtime_execution": _assessed(
            "passed", "registered LibuIPC native processes completed", full_ref
        ),
        "native_qualification": _assessed(
            "passed",
            "ensemble successor explicitly represents the short-replay process spread",
            native_ref,
            full_ref,
            metrics={"short_replay_difference": native["replay_difference"]},
        ),
        "full_horizon_qualification": _assessed(
            "failed",
            "registered ensemble exceeded replay-spread and pin-tracking limits",
            full_ref,
            metrics={
                "gate_metrics": full["gate_metrics"],
                "gate_limits": full["gate_limits"],
                "failed_checks": full["failed_checks"],
            },
        ),
        "decision_headroom": _not_applicable(
            "the RGBench source comparison has one factual rollout, not an action bank"
        ),
        "source_value": _not_evaluated(
            "full-horizon ensemble failure kept source outcomes sealed"
        ),
        "prospective_value": _not_evaluated(
            "full-horizon ensemble failure closed calibration and target access"
        ),
    }
    return SimulatorValidationEntryV1(
        backend_key="libuipc-ensemble-v4",
        display_name="LibuIPC ensemble",
        dataset="RGBench",
        query_scope=_rgbbench_scope(
            backend="LibuIPC ensemble",
            protocol_ids=[str(native["protocol_id"]), str(full["protocol_id"])],
            metric="target-free replay-spread and kinematic qualification",
        ),
        independent_group_count=1,
        stages=stages,
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="full-horizon-replay-spread-and-pin-gates-failed",
        metadata={
            "public_source_outcome_opened": False,
            "target_outcome_opened": False,
            "source_case_count": 1,
            "deterministic_dirac_claim": False,
        },
    )


def _matphys_entry(
    capsules: Mapping[str, Mapping[str, Any]],
) -> SimulatorValidationEntryV1:
    receipt = capsules["matphys_runtime"]
    _expect(receipt, "technical_smoke_passed", False, label="MatPhys runtime")
    _expect(receipt, "trajectory_produced", False, label="MatPhys runtime")
    _expect(
        receipt,
        "terminal_stage",
        "native-matphys-runtime-import",
        label="MatPhys runtime",
    )
    _expect(receipt, "retry_authorized", False, label="MatPhys runtime")
    boundary = cast(Mapping[str, Any], receipt["information_boundary"])
    _expect(boundary, "target_outcomes_opened", False, label="MatPhys boundary")
    reference = _reference("matphys_runtime", artifact_id=str(receipt["receipt_id"]))
    not_reached = "runtime import failure prevented simulator construction"
    stages = {
        "runtime_execution": _assessed(
            "failed",
            "pinned MatPhys import failed before simulator construction",
            reference,
            metrics={"failure_type": receipt["failure_type"]},
        ),
        "native_qualification": _not_evaluated(not_reached),
        "full_horizon_qualification": _not_evaluated(not_reached),
        "decision_headroom": _not_applicable(
            "the RGBench source comparison has one factual rollout, not an action bank"
        ),
        "source_value": _not_evaluated(not_reached),
        "prospective_value": _not_evaluated(not_reached),
    }
    plan_id = str(receipt["plan_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=_semantic_id(
            "simulator", public_dataset="RGBench", backend="MatPhys pinned runtime"
        ),
        task_id=_semantic_id(
            "task", public_dataset="RGBench", task="frozen-source-risk-screen"
        ),
        observation_policy_id=_semantic_id(
            "observation-policy", plan_id=plan_id, source_frame_zero_only=True
        ),
        action_bank_id=_semantic_id(
            "action-bank", plan_id=plan_id, description="registered factual action"
        ),
        metric_id=_semantic_id(
            "metric", plan_id=plan_id, description="runtime then source-risk gate"
        ),
        world_distribution_id=_semantic_id(
            "world-distribution", plan_id=plan_id, split="public-source"
        ),
        statistical_unit="registered MatPhys technical smoke attempt",
        metadata={
            "public_dataset": "RGBench",
            "backend": "MatPhys",
            "attempt_limit": 1,
        },
    )
    return SimulatorValidationEntryV1(
        backend_key="matphys-pinned-runtime-v1",
        display_name="MatPhys pinned runtime",
        dataset="RGBench",
        query_scope=scope,
        independent_group_count=1,
        stages=stages,
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        protected_target_data_read=False,
        new_recording_used=False,
        terminal_reason="terminal-native-runtime-import-failure",
        metadata={
            "attempt_consumed": True,
            "retry_authorized": False,
            "source_future_outcome_opened": False,
            "target_outcome_opened": False,
        },
    )


def build_atlas() -> SimulatorValidationAtlasV1:
    _verify_bound_files()
    dlo = load_query_competence_atlas(ROOT / DLO_ATLAS)
    if dlo.artifact_id != (
        "842941a296a055c78d17278e671de796a8f413bcb7ea30fb3d4ef4b232c460c2"
    ):
        raise ValueError("DLO-Lab query atlas v4 identity changed")
    capsules = _load_capsules()
    entries = [*(_dlo_entry(entry) for entry in dlo.entries)]
    entries.extend(
        [
            _arcsim_entry(capsules),
            _codim_entry(capsules),
            _libuipc_entry(capsules),
            _matphys_entry(capsules),
        ]
    )
    atlas = SimulatorValidationAtlasV1(
        entries=entries,
        metadata={
            "public_datasets": ["DLO-Lab", "RGBench"],
            "validation_ladder": list(STAGE_NAMES),
            "query_scope": (
                "simulator-task-observation-action-metric-world-distribution"
            ),
            "public_data_and_simulator_evidence_only": True,
            "contains_one_prospective_public_simulator_certificate": True,
            "new_outcomes_read": False,
            "new_recordings": False,
            "protected_target_data_read": False,
            "independent_human_review": False,
            "metrics_rewritten": False,
            "bound_source_sha256": {
                str(path): digest for path, digest in sorted(BOUND_FILES.items())
            },
            "claim_boundary": (
                "validation-domain map only; no backend-wide competence, cross-backend "
                "ranking, official benchmark, or new state-of-the-art claim"
            ),
        },
    )
    if (
        len(atlas.entries) != 9
        or len({item.backend_key for item in atlas.entries}) != 5
    ):
        raise ValueError("cross-backend atlas roster changed")
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
