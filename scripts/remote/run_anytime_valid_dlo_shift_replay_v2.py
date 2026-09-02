#!/usr/bin/env python3
"""Replay a fixed DLO3 correction with terminal fallback after revocation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from run_anytime_valid_dlo_shift_replay_v1 import (
    canonical,
    collect_arrays,
    identify_cross_operator_candidates,
    identify_triple,
    load_protocol,
    materialize,
    method_identity,
    sha256_file,
)

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    DeploymentState,
    clipped_gain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dlo45-artifact", type=Path, required=True)
    parser.add_argument("--hierarchical-artifact", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError(f"work root is not empty: {work_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    dlo45_archive = args.dlo45_artifact.resolve(strict=True)
    hierarchical_archive = args.hierarchical_artifact.resolve(strict=True)
    expected_artifacts = protocol["artifacts"]
    for path, identity in (
        (dlo45_archive, expected_artifacts["dlo45"]),
        (hierarchical_archive, expected_artifacts["hierarchical_transfer"]),
    ):
        if path.is_file() and sha256_file(path) != identity["artifact_sha256"]:
            raise ValueError(f"artifact digest differs: {path}")

    dlo45_root = materialize(dlo45_archive, work_root / "dlo45")
    hierarchical_root = materialize(
        hierarchical_archive,
        work_root / "hierarchical",
    )
    records = collect_arrays(dlo45_root, hierarchical_root)
    identities = protocol["registered_aggregate_identities"]

    py_identity = identities["pyelastica"]
    pyelastica = identify_triple(
        records,
        expected_baseline=float(py_identity["fallback_mean_l1_m"]),
        expected_candidate=float(py_identity["candidate_mean_l1_m"]),
        tolerance=5e-4,
        case_count=int(py_identity["trajectory_count"]),
        context_terms=("pyelastica", "source", "transfer", "backend"),
    )
    dlo4_identity = identities["dlo4_procedure"]
    dlo4 = identify_triple(
        records,
        expected_baseline=float(dlo4_identity["fallback_mean_l1_m"]),
        expected_candidate=float(dlo4_identity["candidate_mean_l1_m"]),
        tolerance=5e-7,
        case_count=int(dlo4_identity["trajectory_count"]),
        context_terms=("dlo4", "target", "candidate", "physical"),
    )
    dlo5_identity = identities["dlo5_procedure"]
    dlo5 = identify_triple(
        records,
        expected_baseline=float(dlo5_identity["fallback_mean_l1_m"]),
        expected_candidate=float(dlo5_identity["candidate_mean_l1_m"]),
        tolerance=5e-7,
        case_count=int(dlo5_identity["trajectory_count"]),
        context_terms=("dlo5", "target", "candidate", "physical"),
    )
    cross_identity = identities["dlo3_coefficients_on_dlo45"]
    transfer4, transfer5 = identify_cross_operator_candidates(
        records,
        dlo4=dlo4,
        dlo5=dlo5,
        expected_relative_improvement=float(
            cross_identity["equal_dlo_relative_improvement"]
        ),
    )

    stream = protocol["stream"]
    config = AnytimeAdmissionConfig(
        alpha=float(stream["alpha"]),
        beta=float(stream["beta"]),
        loss_cap=float(stream["loss_cap_m"]),
        gain_margin=float(stream["gain_margin"]),
        harm_margin=float(stream["harm_margin"]),
        allow_reentry=bool(stream["allow_reentry"]),
        lambdas=tuple(float(value) for value in stream["lambdas"]),
    )
    controller = AnytimeAdmissionController(
        config,
        candidate_id=protocol["candidate"]["identity"],
    )
    fallback_token = object()
    candidate_token = object()
    identity_violations = 0
    event_rows: list[dict[str, Any]] = []
    admission: dict[str, Any] | None = None
    revocation: dict[str, Any] | None = None
    terminal_revoked = False
    terminal_epoch: int | None = None
    post_shift_exposures = 0
    post_shift_harmful_exposures = 0
    guarded_post_shift_regret = 0.0
    unguarded_post_shift_regret = 0.0
    reveal_order = 0

    panels = (
        ("PYELASTICA", pyelastica[0], pyelastica[1], pyelastica[2]),
        ("DLO4", dlo4[0], dlo4[1], transfer4),
        ("DLO5", dlo5[0], dlo5[1], transfer5),
    )
    total_post_shift_observations = sum(
        canonical(panel[1].values).shape[0] * canonical(panel[1].values).shape[1]
        for panel in panels[1:]
    )

    for panel_index, (
        domain,
        truth_record,
        fallback_record,
        candidate_record,
    ) in enumerate(panels):
        truth = canonical(truth_record.values)
        fallback = canonical(fallback_record.values)
        candidate = canonical(candidate_record.values)
        if not (truth.shape == fallback.shape == candidate.shape):
            raise ValueError(f"panel arrays do not align for {domain}")
        for case_index in range(truth.shape[0]):
            for frame_index in range(truth.shape[1]):
                reveal_order += 1
                candidate_loss = float(
                    np.mean(
                        np.abs(
                            candidate[case_index, frame_index]
                            - truth[case_index, frame_index]
                        )
                    )
                )
                fallback_loss = float(
                    np.mean(
                        np.abs(
                            fallback[case_index, frame_index]
                            - truth[case_index, frame_index]
                        )
                    )
                )
                raw_gain, normalized_gain, was_clipped = clipped_gain(
                    candidate_loss=candidate_loss,
                    fallback_loss=fallback_loss,
                    loss_cap=config.loss_cap,
                )
                after_shift = panel_index > 0
                regret = candidate_loss - fallback_loss
                if after_shift:
                    unguarded_post_shift_regret += regret

                if terminal_revoked:
                    state_before = DeploymentState.FALLBACK
                    selected = controller.select(
                        fallback=fallback_token,
                        candidate=candidate_token,
                    )
                    if selected is not fallback_token:
                        identity_violations += 1
                    event = "terminal-fallback"
                    epoch = terminal_epoch
                    e_value = None
                    boundary = None
                else:
                    state_before = controller.state
                    selected = controller.select(
                        fallback=fallback_token,
                        candidate=candidate_token,
                    )
                    if state_before is DeploymentState.FALLBACK:
                        if selected is not fallback_token:
                            identity_violations += 1
                    else:
                        if selected is not candidate_token:
                            raise AssertionError("candidate identity changed")
                        if after_shift:
                            post_shift_exposures += 1
                            guarded_post_shift_regret += regret
                            if regret > 0.0:
                                post_shift_harmful_exposures += 1
                    record = controller.observe(
                        candidate_loss=candidate_loss,
                        fallback_loss=fallback_loss,
                    )
                    event = record.event
                    epoch = record.epoch
                    e_value = record.e_value
                    boundary = record.boundary
                    raw_gain = record.raw_gain
                    normalized_gain = record.clipped_normalized_gain

                row = {
                    "reveal_order": reveal_order,
                    "domain": domain,
                    "case_index": case_index,
                    "frame_index": frame_index,
                    "state_before": state_before.value,
                    "event": event,
                    "candidate_loss_m": candidate_loss,
                    "fallback_loss_m": fallback_loss,
                    "raw_gain_m": raw_gain,
                    "clipped_normalized_gain": normalized_gain,
                    "e_value": e_value,
                    "boundary": boundary,
                    "epoch": epoch,
                }
                event_rows.append(row)
                if event == "admit" and admission is None:
                    admission = row
                if event == "revoke" and revocation is None:
                    revocation = row
                    terminal_revoked = not config.allow_reentry
                    terminal_epoch = int(epoch) if epoch is not None else None

    gates = {
        "admitted_during_pyelastica": admission is not None
        and admission["domain"] == "PYELASTICA",
        "revoked_after_operator_shift": revocation is not None
        and revocation["domain"] in {"DLO4", "DLO5"},
        "terminal_fallback_after_revocation": revocation is not None
        and any(row["event"] == "terminal-fallback" for row in event_rows),
        "zero_exact_fallback_identity_violations": identity_violations == 0,
        "guarded_post_shift_exposure_below_unguarded": post_shift_exposures
        < total_post_shift_observations,
        "guarded_post_shift_regret_below_unguarded": guarded_post_shift_regret
        < unguarded_post_shift_regret,
    }
    decision = (
        "retrospective-admit-then-revoke-mechanism-supported"
        if all(gates.values())
        else "retrospective-admit-then-revoke-mechanism-not-supported"
    )
    result = {
        "schema_version": 2,
        "contract": "anytime-valid-dlo-backend-operator-shift-replay-result-v2",
        "status": "complete",
        "decision": decision,
        "gates": gates,
        "candidate_id": protocol["candidate"]["identity"],
        "observation_count": len(event_rows),
        "admission": admission,
        "revocation": revocation,
        "post_shift": {
            "total_observations": total_post_shift_observations,
            "guarded_candidate_exposures": post_shift_exposures,
            "guarded_harmful_candidate_exposures": post_shift_harmful_exposures,
            "terminal_fallback_observations": sum(
                row["event"] == "terminal-fallback"
                and row["domain"] in {"DLO4", "DLO5"}
                for row in event_rows
            ),
            "guarded_cumulative_regret_m": guarded_post_shift_regret,
            "unguarded_cumulative_regret_m": unguarded_post_shift_regret,
            "regret_avoided_m": unguarded_post_shift_regret - guarded_post_shift_regret,
        },
        "exact_fallback_identity_violations": identity_violations,
        "selected_arrays": {
            "pyelastica": method_identity(pyelastica),
            "dlo4_procedure_identity": method_identity(dlo4),
            "dlo5_procedure_identity": method_identity(dlo5),
            "dlo4_exact_transfer_candidate": method_identity((transfer4,)),
            "dlo5_exact_transfer_candidate": method_identity((transfer5,)),
        },
        "artifact_identity": {
            "dlo45": {
                "path": str(dlo45_archive),
                "sha256": sha256_file(dlo45_archive)
                if dlo45_archive.is_file()
                else None,
            },
            "hierarchical_transfer": {
                "path": str(hierarchical_archive),
                "sha256": sha256_file(hierarchical_archive)
                if hierarchical_archive.is_file()
                else None,
            },
        },
        "evidence_class": protocol["evidence_class"],
        "claim_boundary": protocol["claim_boundary"],
        "paper_claim_authorized": False,
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_root / "events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream_handle:
        writer = csv.DictWriter(stream_handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    report = [
        "# Anytime-valid DLO backend-to-operator shift replay",
        "",
        f"- Decision: **{decision}**",
        f"- Observations: **{len(event_rows)}**",
        f"- Admission: **{admission}**",
        f"- Revocation: **{revocation}**",
        f"- Exact fallback identity violations: **{identity_violations}**",
        f"- Guarded post-shift exposures: **{post_shift_exposures}/{total_post_shift_observations}**",
        f"- Guarded post-shift harmful exposures: **{post_shift_harmful_exposures}**",
        (
            "- Post-shift regret avoided: "
            f"**{1000.0 * (unguarded_post_shift_regret - guarded_post_shift_regret):.4f} mm-sum**"
        ),
        "",
        "## Claim boundary",
        "",
        protocol["claim_boundary"],
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
