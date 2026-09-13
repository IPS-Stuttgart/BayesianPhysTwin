#!/usr/bin/env python3
"""Replay sealed paired losses with terminal exact fallback after revocation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_anytime_valid_loss_replay_v1 import load_config, load_rows

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionController,
    DeploymentState,
    clipped_gain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--paired-loss-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--evidence-class",
        choices=("retrospective-replay", "prospectively-ordered-replay"),
        required=True,
    )
    return parser.parse_args()


def write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "stream_id",
        "reveal_order",
        "state_before",
        "deployed_method",
        "candidate_loss",
        "fallback_loss",
        "deployed_loss",
        "raw_gain",
        "clipped_normalized_gain",
        "event",
        "epoch",
        "e_value",
        "boundary",
        "clipped",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    input_path = args.paired_loss_csv.resolve(strict=True)
    protocol, config = load_config(protocol_path)
    streams = load_rows(input_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_identity_violations = 0
    fallback_token = object()
    candidate_token = object()

    for stream_id, rows in sorted(streams.items()):
        controller = AnytimeAdmissionController(
            config,
            candidate_id=f"{args.candidate_id}:{stream_id}",
        )
        terminal_revoked = False
        terminal_epoch: int | None = None
        deployed_loss_sum = 0.0
        fallback_loss_sum = 0.0
        candidate_loss_sum = 0.0
        admissions = 0
        revocations = 0
        candidate_exposures = 0
        harmful_candidate_exposures = 0
        first_admission_order: int | None = None
        first_revocation_order: int | None = None

        for row in rows:
            candidate_loss = float(row["candidate_loss"])
            fallback_loss = float(row["fallback_loss"])
            raw_gain, normalized_gain, was_clipped = clipped_gain(
                candidate_loss=candidate_loss,
                fallback_loss=fallback_loss,
                loss_cap=config.loss_cap,
            )

            if terminal_revoked:
                state_before = DeploymentState.FALLBACK
                selected = controller.select(
                    fallback=fallback_token,
                    candidate=candidate_token,
                )
                if selected is not fallback_token:
                    total_identity_violations += 1
                deployed_method = "fallback"
                deployed_loss = fallback_loss
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
                    deployed_method = "fallback"
                    deployed_loss = fallback_loss
                    if selected is not fallback_token:
                        total_identity_violations += 1
                else:
                    deployed_method = "candidate"
                    deployed_loss = candidate_loss
                    candidate_exposures += 1
                    if candidate_loss > fallback_loss:
                        harmful_candidate_exposures += 1
                    if selected is not candidate_token:
                        raise AssertionError("candidate selection lost object identity")

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
                was_clipped = record.clipped
                if event == "admit":
                    admissions += 1
                    if first_admission_order is None:
                        first_admission_order = int(row["reveal_order"])
                if event == "revoke":
                    revocations += 1
                    terminal_revoked = not config.allow_reentry
                    terminal_epoch = record.epoch
                    if first_revocation_order is None:
                        first_revocation_order = int(row["reveal_order"])

            deployed_loss_sum += deployed_loss
            fallback_loss_sum += fallback_loss
            candidate_loss_sum += candidate_loss
            event_rows.append(
                {
                    "stream_id": stream_id,
                    "reveal_order": row["reveal_order"],
                    "state_before": state_before.value,
                    "deployed_method": deployed_method,
                    "candidate_loss": candidate_loss,
                    "fallback_loss": fallback_loss,
                    "deployed_loss": deployed_loss,
                    "raw_gain": raw_gain,
                    "clipped_normalized_gain": normalized_gain,
                    "event": event,
                    "epoch": epoch,
                    "e_value": e_value,
                    "boundary": boundary,
                    "clipped": was_clipped,
                }
            )

        count = len(rows)
        summaries.append(
            {
                "stream_id": stream_id,
                "observation_count": count,
                "admissions": admissions,
                "revocations": revocations,
                "first_admission_order": first_admission_order,
                "first_revocation_order": first_revocation_order,
                "candidate_exposures": candidate_exposures,
                "harmful_candidate_exposures": harmful_candidate_exposures,
                "terminal_fallback_observations": sum(
                    event["event"] == "terminal-fallback"
                    for event in event_rows
                    if event["stream_id"] == stream_id
                ),
                "fallback_mean_loss": fallback_loss_sum / count,
                "candidate_mean_loss": candidate_loss_sum / count,
                "deployed_mean_loss": deployed_loss_sum / count,
                "deployed_relative_improvement_vs_fallback": (
                    1.0 - deployed_loss_sum / fallback_loss_sum
                    if fallback_loss_sum > 0.0
                    else None
                ),
                "terminal_state": controller.state.value,
                "terminal_epoch": controller.epoch,
                "terminal_revoked": terminal_revoked,
            }
        )

    payload = {
        "schema_version": 2,
        "contract": "anytime-valid-simulator-admission-loss-replay-result-v2",
        "status": "complete",
        "evidence_class": args.evidence_class,
        "candidate_id": args.candidate_id,
        "stream_count": len(summaries),
        "observation_count": len(event_rows),
        "config": protocol["controller"],
        "exact_fallback_identity_violations": total_identity_violations,
        "streams": summaries,
        "aggregate": {
            "admissions": sum(row["admissions"] for row in summaries),
            "revocations": sum(row["revocations"] for row in summaries),
            "candidate_exposures": sum(row["candidate_exposures"] for row in summaries),
            "harmful_candidate_exposures": sum(
                row["harmful_candidate_exposures"] for row in summaries
            ),
            "terminal_fallback_observations": sum(
                row["terminal_fallback_observations"] for row in summaries
            ),
            "mean_deployed_relative_improvement_vs_fallback": sum(
                row["deployed_relative_improvement_vs_fallback"] or 0.0
                for row in summaries
            )
            / len(summaries),
        },
        "claim_boundary": (
            "This replay evaluates a frozen sequential rule on supplied paired "
            "losses. After non-reentrant revocation, every later decision is the "
            "exact fallback and no further evidence test is run. A retrospective "
            "replay is not fresh confirmation. The anytime guarantee requires the "
            "registered conditional-mean null, predictable candidate forecasts, "
            "outcome-independent reveal order, and source-frozen clipping and "
            "margins. It is not a safety claim."
        ),
        "paper_claim_authorized": False,
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_events(output_root / "events.csv", event_rows)
    with (output_root / "stream-summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
