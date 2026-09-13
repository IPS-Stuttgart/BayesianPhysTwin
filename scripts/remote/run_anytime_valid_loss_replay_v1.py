#!/usr/bin/env python3
"""Replay sealed paired physical-twin losses through the anytime-valid guard."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    DeploymentState,
)

REQUIRED_COLUMNS = {
    "stream_id",
    "reveal_order",
    "candidate_loss",
    "fallback_loss",
}


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


def load_config(path: Path) -> tuple[dict[str, Any], AnytimeAdmissionConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract") != "anytime-valid-simulator-admission-validation-v1":
        raise ValueError("unexpected protocol contract")
    raw = payload["controller"]
    return payload, AnytimeAdmissionConfig(
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        loss_cap=float(raw["loss_cap"]),
        gain_margin=float(raw["gain_margin"]),
        harm_margin=float(raw["harm_margin"]),
        allow_reentry=bool(raw["allow_reentry"]),
        lambdas=tuple(float(value) for value in raw["lambdas"]),
    )


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "paired loss CSV must contain " + ", ".join(sorted(REQUIRED_COLUMNS))
            )
        for line_number, raw in enumerate(reader, start=2):
            stream_id = str(raw["stream_id"]).strip()
            if not stream_id:
                raise ValueError(f"line {line_number}: stream_id is empty")
            row = dict(raw)
            row["stream_id"] = stream_id
            row["reveal_order"] = int(raw["reveal_order"])
            row["candidate_loss"] = float(raw["candidate_loss"])
            row["fallback_loss"] = float(raw["fallback_loss"])
            row["line_number"] = line_number
            streams[stream_id].append(row)
    if not streams:
        raise ValueError("paired loss CSV is empty")
    for stream_id, rows in streams.items():
        rows.sort(key=lambda row: row["reveal_order"])
        orders = [int(row["reveal_order"]) for row in rows]
        if len(set(orders)) != len(orders):
            raise ValueError(f"stream {stream_id}: duplicate reveal_order")
        if any(
            current <= previous
            for previous, current in zip(orders, orders[1:], strict=False)
        ):
            raise ValueError(f"stream {stream_id}: reveal order is not increasing")
    return dict(streams)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
            state_before = controller.state
            selected = controller.select(
                fallback=fallback_token,
                candidate=candidate_token,
            )
            if state_before is DeploymentState.FALLBACK:
                deployed_method = "fallback"
                deployed_loss = float(row["fallback_loss"])
                if selected is not fallback_token:
                    total_identity_violations += 1
            else:
                deployed_method = "candidate"
                deployed_loss = float(row["candidate_loss"])
                candidate_exposures += 1
                if float(row["candidate_loss"]) > float(row["fallback_loss"]):
                    harmful_candidate_exposures += 1
                if selected is not candidate_token:
                    raise AssertionError("candidate selection lost object identity")

            record = controller.observe(
                candidate_loss=float(row["candidate_loss"]),
                fallback_loss=float(row["fallback_loss"]),
            )
            if record.event == "admit":
                admissions += 1
                if first_admission_order is None:
                    first_admission_order = int(row["reveal_order"])
            if record.event == "revoke":
                revocations += 1
                if first_revocation_order is None:
                    first_revocation_order = int(row["reveal_order"])

            deployed_loss_sum += deployed_loss
            fallback_loss_sum += float(row["fallback_loss"])
            candidate_loss_sum += float(row["candidate_loss"])
            event_rows.append(
                {
                    "stream_id": stream_id,
                    "reveal_order": row["reveal_order"],
                    "state_before": state_before.value,
                    "deployed_method": deployed_method,
                    "candidate_loss": row["candidate_loss"],
                    "fallback_loss": row["fallback_loss"],
                    "deployed_loss": deployed_loss,
                    "raw_gain": record.raw_gain,
                    "clipped_normalized_gain": record.clipped_normalized_gain,
                    "event": record.event,
                    "epoch": record.epoch,
                    "e_value": record.e_value,
                    "boundary": record.boundary,
                    "clipped": record.clipped,
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
            }
        )

    payload = {
        "schema_version": 1,
        "contract": "anytime-valid-simulator-admission-loss-replay-result-v1",
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
            "mean_deployed_relative_improvement_vs_fallback": sum(
                row["deployed_relative_improvement_vs_fallback"] or 0.0
                for row in summaries
            )
            / len(summaries),
        },
        "claim_boundary": (
            "This replay evaluates a frozen sequential rule on supplied paired "
            "losses. A retrospective replay is not fresh confirmation. The "
            "anytime guarantee requires the registered conditional-mean null, "
            "predictable candidate forecasts, outcome-independent reveal order, "
            "and source-frozen clipping and margins. It is not a safety claim."
        ),
        "paper_claim_authorized": False,
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_rows(output_root / "events.csv", event_rows)
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
