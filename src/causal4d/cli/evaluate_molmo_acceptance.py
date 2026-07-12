"""Evaluate the locked MolmoMotion competence gate before beta selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.molmo_acceptance import (
    MolmoAcceptanceThresholds,
    aggregate_molmo_acceptance,
    evaluate_molmo_acceptance_case,
    molmo_acceptance_result_id,
)
from causal4d.molmo_adapter import load_molmo_forecasts
from causal4d.phystwin_backend import load_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run direct Molmo forecast and physical-ranking acceptance gates."
    )
    parser.add_argument("benchmark_manifest_json")
    parser.add_argument("output_json")
    return parser


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.benchmark_manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Molmo acceptance manifest schema_version must be 1")
    thresholds = MolmoAcceptanceThresholds.from_mapping(
        manifest.get("thresholds", {})
    )
    entries = manifest.get("cases", [])
    if not entries:
        raise ValueError("Molmo acceptance manifest must contain cases")
    case_results = []
    sources = []
    for entry in entries:
        final_data_path = _resolve(manifest_path.parent, entry["final_data"])
        forecast_path = _resolve(manifest_path.parent, entry["molmo_forecast"])
        bank_path = _resolve(manifest_path.parent, entry["physical_rollout_bank"])
        bundle = load_molmo_forecasts(forecast_path)
        bank, bank_manifest = load_rollout_bank(bank_path)
        with final_data_path.open("rb") as handle:
            final_data = pickle.load(handle)
        points = np.asarray(final_data["object_points"], dtype=float)
        validity = np.asarray(final_data["object_visibilities"], dtype=bool) & np.asarray(
            final_data["object_motions_valid"],
            dtype=bool,
        )
        result = evaluate_molmo_acceptance_case(
            case_id=str(entry["case_id"]),
            bundle=bundle,
            object_points_m=points,
            validity=validity,
            bank=bank,
            bank_manifest=bank_manifest,
            primary_forecast_id=str(entry["primary_forecast_id"]),
            paraphrase_forecast_ids=tuple(entry["paraphrase_forecast_ids"]),
            thresholds=thresholds,
        )
        case_results.append(result)
        sources.append(
            {
                "case_id": entry["case_id"],
                "final_data": _artifact_descriptor(final_data_path),
                "molmo_forecast": _artifact_descriptor(forecast_path),
                "physical_rollout_bank": _artifact_descriptor(bank_path),
            }
        )
        del bank
    decision = aggregate_molmo_acceptance(case_results, thresholds)
    payload = {
        "schema_version": 1,
        "experiment": manifest.get(
            "protocol_id",
            "causal4d-molmo-acceptance-v1",
        ),
        "claim_boundary": (
            "MolmoMotion is evaluated before beta selection; failure preserves beta=0."
        ),
        "thresholds": thresholds.as_dict(),
        "cases": case_results,
        "decision": decision,
        "source_manifest": _artifact_descriptor(manifest_path),
        "source_artifacts": sources,
    }
    payload["acceptance_result_id"] = molmo_acceptance_result_id(payload)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
