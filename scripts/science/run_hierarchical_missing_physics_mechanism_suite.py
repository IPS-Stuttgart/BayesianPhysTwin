#!/usr/bin/env python3
"""Controlled falsification suite for hierarchical discrepancy diagnosis.

Each case activates exactly one registered discrepancy mechanism while retaining
all feature blocks. The source-only posterior must identify the active group.
A paired held-out-object/backend panel then tests the registered transfer scope:
shared physics persists, object/backend effects disappear or reverse, contact
and actuation require matching support/calibration, and sensor discrepancy is
never injected into physical rollout state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "hierarchical_missing_physics_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("hmp_core_suite", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {CORE_PATH}")
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)

GROUPS = CORE.REGISTERED_GROUPS
TRANSFERABLE = {"shared_physics"}
EPS = 1.0e-12


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _orthogonal_blocks(
    *, rng: np.random.Generator, n: int, object_index: int, backend_index: int
) -> dict[str, np.ndarray]:
    latent = rng.normal(size=(n, 13))
    # QR makes the blocks nearly orthogonal and avoids accidental attribution.
    q, _ = np.linalg.qr(latent, mode="reduced")
    shared = q[:, 0:3]
    object_block = np.zeros((n, 2))
    object_block[:, object_index] = q[:, 3]
    backend_block = np.zeros((n, 2))
    backend_block[:, backend_index] = q[:, 4]
    contact = np.column_stack((q[:, 5], q[:, 6]))
    actuation = np.column_stack((q[:, 7], q[:, 8]))
    sensor = np.column_stack((q[:, 9], q[:, 10]))
    return {
        "shared_physics": shared,
        "object": object_block,
        "backend": backend_block,
        "contact": contact,
        "actuation": actuation,
        "sensor": sensor,
    }


def _coefficients(active_group: str, output_dim: int = 3) -> np.ndarray:
    widths = {
        "shared_physics": 3,
        "object": 2,
        "backend": 2,
        "contact": 2,
        "actuation": 2,
        "sensor": 2,
    }
    width = widths[active_group]
    base = np.array(
        [
            [1.15, -0.55, 0.38],
            [-0.72, 0.91, -0.44],
            [0.51, 0.36, 0.84],
        ],
        dtype=float,
    )
    return base[:width, :output_dim]


def make_case_panel(
    active_group: str,
    *,
    role: str,
    seed: int,
    matching_support: bool = True,
) -> Any:
    if active_group not in GROUPS:
        raise ValueError(active_group)
    if role not in {"source", "target"}:
        raise ValueError(role)
    rng = np.random.default_rng(seed)
    source = role == "source"
    object_names = ("DLO2", "DLO3") if source else ("DLO4", "DLO5")
    backend_names = ("deform-a", "deform-b") if source else ("alternate-a", "alternate-b")
    y_rows: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    object_ids: list[str] = []
    backend_ids: list[str] = []
    block_rows = {name: [] for name in GROUPS}

    for object_index, object_name in enumerate(object_names):
        for backend_index, backend_name in enumerate(backend_names):
            for trajectory_index in range(5):
                n = 28
                blocks = _orthogonal_blocks(
                    rng=rng,
                    n=n,
                    object_index=object_index,
                    backend_index=backend_index,
                )
                coefficient = _coefficients(active_group)
                target_coefficient = coefficient.copy()
                if not source:
                    if active_group in {"object", "backend"}:
                        target_coefficient *= -0.75
                    elif active_group in {"contact", "actuation"} and not matching_support:
                        target_coefficient *= 0.0
                    elif active_group == "sensor":
                        # Sensor-only discrepancy exists in the measurement residual,
                        # but is never admissible as a physical rollout correction.
                        target_coefficient *= 1.0
                signal = blocks[active_group] @ target_coefficient
                signal += rng.normal(scale=0.035, size=signal.shape)
                for sample_index in range(n):
                    y_rows.append(signal[sample_index])
                    trajectory_ids.append(
                        f"{object_name}:{backend_name}:{active_group}:trajectory-{trajectory_index}"
                    )
                    object_ids.append(object_name)
                    backend_ids.append(backend_name)
                    for name in GROUPS:
                        block_rows[name].append(blocks[name][sample_index])

    y = np.asarray(y_rows)
    blocks_np = {name: np.asarray(values) for name, values in block_rows.items()}
    content = {
        "active_group": active_group,
        "role": role,
        "matching_support": matching_support,
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y).view(np.uint8)).hexdigest(),
        "block_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
            for name, value in blocks_np.items()
        },
    }
    panel_sha = canonical_hash(content)
    return CORE.ResidualPanel(
        y=y,
        blocks=blocks_np,
        trajectory_id=np.asarray(trajectory_ids),
        object_id=np.asarray(object_ids),
        backend_id=np.asarray(backend_ids),
        metadata={
            "panel_sha256": panel_sha,
            "active_group_ground_truth": active_group,
            "role": role,
            "matching_support": matching_support,
            "transfer_eligibility": CORE.DEFAULT_TRANSFER_ELIGIBILITY,
            "minimum_bootstrap_diagnosis_frequency": 0.60,
            "source_gate": {
                "shared_vs_physical_min_relative_improvement": 0.05,
                "minimum_source_bootstrap_shared_diagnosis_frequency": 0.60,
                "minimum_complete_trajectory_win_fraction": 0.70,
                "maximum_worst_trajectory_ratio_vs_physical": 1.05,
            },
        },
    )


def diagnose_case(panel: Any, *, bootstrap: int, seed: int) -> dict[str, Any]:
    model = CORE.fit_group_ard(panel)
    diagnostics = CORE.group_diagnostics(panel, model)
    frequencies = CORE.bootstrap_diagnosis_frequency(
        panel, repetitions=bootstrap, seed=seed
    )
    ranking = sorted(
        diagnostics,
        key=lambda name: (
            diagnostics[name]["negative_log_score_increase_when_removed"],
            diagnostics[name]["full_fit_rmse_increase_when_removed"],
            frequencies[name],
        ),
        reverse=True,
    )
    active = str(panel.metadata["active_group_ground_truth"])
    return {
        "ground_truth_group": active,
        "predicted_group": ranking[0],
        "correct": ranking[0] == active,
        "ranking": ranking,
        "group_diagnostics": diagnostics,
        "bootstrap_diagnosis_frequency": frequencies,
        "model": model,
    }


def evaluate_scope(
    source_panel: Any,
    target_panel: Any,
    diagnosis: Mapping[str, Any],
) -> dict[str, Any]:
    model = diagnosis["model"]
    active_group = diagnosis["predicted_group"]
    physical = np.zeros_like(target_panel.y)
    raw_prediction = model.predict(target_panel.blocks, active_groups=[active_group])
    admissible_for_physical_rollout = active_group in TRANSFERABLE
    emitted = raw_prediction if admissible_for_physical_rollout else physical

    physical_rmse = float(np.sqrt(np.mean(np.square(target_panel.y - physical))))
    raw_rmse = float(np.sqrt(np.mean(np.square(target_panel.y - raw_prediction))))
    emitted_rmse = float(np.sqrt(np.mean(np.square(target_panel.y - emitted))))
    fallback_violations = int(
        not admissible_for_physical_rollout and not np.array_equal(emitted, physical)
    )
    return {
        "diagnosed_group": active_group,
        "admissible_for_cross_object_cross_backend_physical_transfer": admissible_for_physical_rollout,
        "physical_rmse": physical_rmse,
        "raw_transferred_group_rmse": raw_rmse,
        "emitted_rmse": emitted_rmse,
        "raw_relative_improvement_vs_physical": (
            physical_rmse - raw_rmse
        ) / max(physical_rmse, EPS),
        "emitted_relative_improvement_vs_physical": (
            physical_rmse - emitted_rmse
        ) / max(physical_rmse, EPS),
        "fallback_identity_violations": fallback_violations,
        "emitted_is_exact_fallback": bool(np.array_equal(emitted, physical))
        if not admissible_for_physical_rollout
        else False,
    }


def run_suite(*, seed: int, bootstrap: int) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in GROUPS} for expected in GROUPS
    }
    correct = 0
    for case_index, group in enumerate(GROUPS):
        source = make_case_panel(
            group,
            role="source",
            seed=seed + 100 * case_index,
            matching_support=True,
        )
        target = make_case_panel(
            group,
            role="target",
            seed=seed + 100 * case_index + 1,
            matching_support=False,
        )
        diagnosis = diagnose_case(
            source, bootstrap=bootstrap, seed=seed + 1000 + case_index
        )
        scope = evaluate_scope(source, target, diagnosis)
        predicted = diagnosis["predicted_group"]
        confusion[group][predicted] += 1
        correct += int(diagnosis["correct"])
        cases[group] = {
            "ground_truth_group": group,
            "predicted_group": predicted,
            "diagnosis_correct": diagnosis["correct"],
            "ranking": diagnosis["ranking"],
            "bootstrap_diagnosis_frequency": diagnosis[
                "bootstrap_diagnosis_frequency"
            ],
            "group_diagnostics": diagnosis["group_diagnostics"],
            "scope_evaluation": scope,
        }

    shared = cases["shared_physics"]["scope_evaluation"]
    nontransferable = [
        cases[group]["scope_evaluation"] for group in GROUPS if group != "shared_physics"
    ]
    gate = {
        "all_mechanisms_correctly_diagnosed": correct == len(GROUPS),
        "shared_transfers_and_improves": bool(
            shared["admissible_for_cross_object_cross_backend_physical_transfer"]
            and shared["raw_relative_improvement_vs_physical"] >= 0.50
        ),
        "all_nontransferable_mechanisms_rejected": all(
            not value[
                "admissible_for_cross_object_cross_backend_physical_transfer"
            ]
            for value in nontransferable
        ),
        "all_rejected_mechanisms_emit_exact_fallback": all(
            value["emitted_is_exact_fallback"] for value in nontransferable
        ),
        "fallback_identity_violations": sum(
            value["fallback_identity_violations"] for value in nontransferable
        ),
        "sensor_never_changes_physical_rollout": bool(
            cases["sensor"]["scope_evaluation"]["emitted_is_exact_fallback"]
        ),
    }
    gate["passed"] = bool(
        gate["all_mechanisms_correctly_diagnosed"]
        and gate["shared_transfers_and_improves"]
        and gate["all_nontransferable_mechanisms_rejected"]
        and gate["all_rejected_mechanisms_emit_exact_fallback"]
        and gate["fallback_identity_violations"] == 0
        and gate["sensor_never_changes_physical_rollout"]
    )
    result = {
        "schema": "bayesian-phystwin.hierarchical-missing-physics-mechanism-suite-result",
        "schema_version": 1,
        "seed": seed,
        "trajectory_bootstrap_repetitions_per_case": bootstrap,
        "mechanisms": list(GROUPS),
        "correct_diagnoses": correct,
        "total_diagnoses": len(GROUPS),
        "confusion_matrix": confusion,
        "cases": cases,
        "gate": gate,
        "information_boundary": {
            "real_data_read": False,
            "protected_dlo4_dlo5_result_read": False,
            "target_used_to_select_diagnosis": False,
            "target_used_to_fit_coefficients": False,
            "controlled_falsification_only": True,
        },
    }
    result["result_id"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run_suite(seed=arguments.seed, bootstrap=arguments.bootstrap)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "result_id": result["result_id"],
        "correct_diagnoses": result["correct_diagnoses"],
        "total_diagnoses": result["total_diagnoses"],
        "predictions": {
            group: value["predicted_group"] for group, value in result["cases"].items()
        },
        "scope_evaluation": {
            group: value["scope_evaluation"] for group, value in result["cases"].items()
        },
        "gate": result["gate"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    if not result["gate"]["passed"]:
        raise SystemExit("controlled mechanism suite failed")


if __name__ == "__main__":
    main()
