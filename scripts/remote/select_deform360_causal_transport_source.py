#!/usr/bin/env python3
"""Freeze causal-transport pooling controls and evaluate its source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_causal_transport_method import (
    causal_transport_candidates,
    load_causal_transport_method,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_selection import (
    fit_pooling_controls,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_matrix(values: np.ndarray) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in np.asarray(values, dtype=np.float64)
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_sota_config(args.protocol)
    method = load_causal_transport_method(args.method)
    _require(
        method["config"]["parent_config_sha256"] == protocol["config_sha256"],
        "causal-transport selection uses another parent protocol",
    )
    fit_episode_ids = tuple(
        int(value) for value in method["config"]["selection"]["fit_episode_ids"]
    )
    _require(
        fit_episode_ids
        == tuple(protocol["config"]["dataset"]["fit_episode_ids"]),
        "causal-transport source panel differs from its parent",
    )
    candidates = causal_transport_candidates(method)
    labels = tuple(candidate["label"] for candidate in candidates)
    track = np.empty((len(candidates), len(fit_episode_ids)), dtype=np.float64)
    chamfer = np.empty_like(track)
    persistence_track = np.empty(len(fit_episode_ids), dtype=np.float64)
    persistence_chamfer = np.empty(len(fit_episode_ids), dtype=np.float64)
    bank_inputs: dict[str, dict[str, str]] = {}
    for column, episode_id in enumerate(fit_episode_ids):
        path = (
            args.source_root
            / f"episode_{episode_id:04d}"
            / "causal_transport_source_bank.json"
        )
        bank = json.loads(path.read_text(encoding="utf-8"))
        _require(
            bank.get("artifact_kind")
            == "Deform360CausalTransportSourceCandidateBank"
            and bank.get("result_sha256") == _canonical_sha256(bank)
            and bank.get("method_config_sha256") == method["config_sha256"]
            and bank.get("object_id") == args.object_id
            and int(bank.get("episode_id", -1)) == episode_id
            and bank.get("candidate_order") == list(labels)
            and bank.get("candidate_count") == len(candidates)
            and bank.get("passed") is True,
            f"causal-transport bank is incompatible: episode {episode_id}",
        )
        boundary = bank.get("information_boundary", {})
        _require(
            boundary.get("source_future_outcome_used_for_candidate_scoring") is True
            and boundary.get("source_future_outcome_used_for_prediction") is False
            and boundary.get("held_development_outcome_read") is False
            and boundary.get("confirmatory_object_read") is False
            and boundary.get("pokeflex_target_read") is False,
            f"causal-transport boundary changed: episode {episode_id}",
        )
        persistence_full = bank["persistence_metrics"]["ranges"]["full"]
        persistence_track[column] = float(persistence_full["track_error_m"])
        persistence_chamfer[column] = float(persistence_full["chamfer_m"])
        for row, record in enumerate(bank["records"]):
            full = record["metrics"]["ranges"]["full"]
            track[row, column] = float(full["track_error_m"])
            chamfer[row, column] = float(full["chamfer_m"])
        bank_inputs[str(episode_id)] = {
            "path": str(path.resolve()),
            "file_sha256": _sha256_file(path),
            "result_sha256": str(bank["result_sha256"]),
        }

    selection = fit_pooling_controls(
        labels,
        fit_episode_ids,
        track_error_m=track,
        chamfer_m=chamfer,
        persistence_track_error_m=persistence_track,
        persistence_chamfer_m=persistence_chamfer,
    )
    loo_rows = selection["leave_one_fit_action_out"]
    mean_loo_score = float(
        np.mean([row["pooled_normalized_score"] for row in loo_rows])
    )
    gate = method["config"]["source_gate"]
    gate_checks = {
        "leave_one_action_out_persistence": selection[
            "leave_one_out_persistence_win_fraction"
        ]
        >= float(gate["minimum_leave_one_action_out_persistence_win_fraction"]),
        "leave_one_action_out_single_median": selection[
            "leave_one_out_single_median_win_fraction"
        ]
        >= float(gate["minimum_leave_one_action_out_single_median_win_fraction"]),
        "mean_normalized_score": mean_loo_score
        <= float(gate["maximum_mean_normalized_score"]),
    }
    passed = all(gate_checks.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalTransportSourceSelection",
        "protocol_id": method["config"]["protocol_id"],
        "method_config_sha256": method["config_sha256"],
        "object_id": args.object_id,
        "candidate_parameters_in_locked_order": list(candidates),
        "selection": selection,
        "source_gate": {
            "checks": gate_checks,
            "mean_leave_one_action_out_normalized_score": mean_loo_score,
            "thresholds": gate,
            "passed": passed,
            "failure_action": (
                None
                if passed
                else "sealed development-held outcome remains unopened"
            ),
        },
        "source_bank_inputs": bank_inputs,
        "source_metric_matrices": {
            "track_error_m": _json_matrix(track),
            "chamfer_m": _json_matrix(chamfer),
            "persistence_track_error_m": persistence_track.tolist(),
            "persistence_chamfer_m": persistence_chamfer.tolist(),
        },
        "information_boundary": {
            "fit_episode_outcomes_used": list(fit_episode_ids),
            "held_development_outcome_read": False,
            "confirmatory_object_read": False,
            "pokeflex_target_read": False,
            "selection_refit_after_held_reveal": False,
        },
        "passed": passed,
        "claim_boundary": (
            "source-frozen causal-transport selection only; held transfer and "
            "direct Deform360 Table 4 claims remain unevaluated"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _require(not args.output.exists(), f"source selection exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "pooled_candidate_label": selection["pooled_candidate_label"],
                "gate_checks": gate_checks,
                "mean_leave_one_action_out_normalized_score": mean_loo_score,
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
