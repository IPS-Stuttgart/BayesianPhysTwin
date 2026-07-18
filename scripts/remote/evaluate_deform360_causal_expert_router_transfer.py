#!/usr/bin/env python3
"""Evaluate a frozen causal-expert router on an already-open transfer panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_causal_expert_router import (
    CausalExpertEpisode,
    build_causal_expert_features,
    load_causal_expert_router,
    normalized_candidate_score,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _interval_metrics(record: Mapping[str, Any], name: str) -> tuple[float, float]:
    interval = record["metrics"]["ranges"][name]
    return float(interval["track_error_m"]), float(interval["chamfer_m"])


def main() -> int:
    args = _parse_args()
    model = load_causal_expert_router(args.model)
    rows = []
    bank_inputs = {}
    for episode_id in args.episode_ids:
        path = (
            args.source_root
            / f"episode_{episode_id:04d}"
            / "causal_transport_source_bank.json"
        )
        bank = json.loads(path.read_text(encoding="utf-8"))
        _require(
            bank.get("artifact_kind") == "Deform360CausalTransportSourceCandidateBank"
            and bank.get("result_sha256") == _canonical_sha256(bank)
            and bank.get("object_id") == args.object_id
            and int(bank.get("episode_id", -1)) == episode_id
            and bank.get("passed") is True,
            f"transfer bank is incompatible: episode {episode_id}",
        )
        records = bank["records"]
        labels = tuple(str(record["label"]) for record in records)
        _require(labels == model.candidate_labels, "transfer candidate order changed")
        features_by_candidate = [
            build_causal_expert_features(record) for record in records[1:]
        ]
        _require(
            all(
                set(features) == set(model.feature_names)
                for features in features_by_candidate
            ),
            "transfer feature contract changed",
        )
        features = np.asarray(
            [
                [candidate[name] for name in model.feature_names]
                for candidate in features_by_candidate
            ],
            dtype=np.float64,
        )
        scores = np.asarray(
            [
                normalized_candidate_score(record, bank["persistence_metrics"])
                for record in records
            ]
        )
        episode = CausalExpertEpisode(
            object_id=args.object_id,
            episode_id=episode_id,
            labels=labels,
            features=features,
            normalized_scores=scores,
        )
        decision = model.decide(episode)
        selected = records[decision.selected_index]
        persistence = records[0]
        full_track, full_chamfer = _interval_metrics(selected, "full")
        base_track, base_chamfer = _interval_metrics(persistence, "full")
        late_track, late_chamfer = _interval_metrics(selected, "late")
        late_base_track, late_base_chamfer = _interval_metrics(persistence, "late")
        rows.append(
            {
                "episode_id": episode_id,
                "selected_label": decision.selected_label,
                "accepted": decision.accepted,
                "predicted_log_score": decision.predicted_log_score,
                "upper_log_score": decision.upper_log_score,
                "normalized_joint_score": float(scores[decision.selected_index]),
                "future": {
                    "track_error_m": full_track,
                    "persistence_track_error_m": base_track,
                    "chamfer_m": full_chamfer,
                    "persistence_chamfer_m": base_chamfer,
                },
                "late": {
                    "track_error_m": late_track,
                    "persistence_track_error_m": late_base_track,
                    "chamfer_m": late_chamfer,
                    "persistence_chamfer_m": late_base_chamfer,
                },
                "oracle": {
                    "label": labels[int(np.argmin(scores))],
                    "normalized_joint_score": float(np.min(scores)),
                },
            }
        )
        bank_inputs[str(episode_id)] = {
            "path": str(path.resolve()),
            "file_sha256": _sha256_file(path),
            "result_sha256": bank["result_sha256"],
        }

    def aggregate(interval: str) -> dict[str, float]:
        track = np.asarray([row[interval]["track_error_m"] for row in rows])
        track_base = np.asarray(
            [row[interval]["persistence_track_error_m"] for row in rows]
        )
        chamfer = np.asarray([row[interval]["chamfer_m"] for row in rows])
        chamfer_base = np.asarray(
            [row[interval]["persistence_chamfer_m"] for row in rows]
        )
        return {
            "track_error_m": float(np.mean(track)),
            "persistence_track_error_m": float(np.mean(track_base)),
            "track_improvement_fraction": float(
                1.0 - np.mean(track) / np.mean(track_base)
            ),
            "chamfer_m": float(np.mean(chamfer)),
            "persistence_chamfer_m": float(np.mean(chamfer_base)),
            "chamfer_improvement_fraction": float(
                1.0 - np.mean(chamfer) / np.mean(chamfer_base)
            ),
            "maximum_track_degradation_fraction": float(
                np.max(track / track_base - 1.0)
            ),
            "maximum_chamfer_degradation_fraction": float(
                np.max(chamfer / chamfer_base - 1.0)
            ),
        }

    future = aggregate("future")
    late = aggregate("late")
    checks = {
        "future_track": future["track_improvement_fraction"] >= 0.03,
        "future_chamfer": future["chamfer_improvement_fraction"] >= 0.03,
        "late_track": late["track_improvement_fraction"] >= 0.05,
        "late_chamfer": late["chamfer_improvement_fraction"] >= 0.05,
        "maximum_degradation": max(
            future["maximum_track_degradation_fraction"],
            future["maximum_chamfer_degradation_fraction"],
        )
        <= 0.10,
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalExpertRouterTransferReport",
        "object_id": args.object_id,
        "episode_ids": list(args.episode_ids),
        "model": {
            "path": str(args.model.resolve()),
            "file_sha256": _sha256_file(args.model),
            "result_sha256": model.result_sha256,
        },
        "rows": rows,
        "aggregate": {"future": future, "late": late},
        "transfer_gate": {
            "checks": checks,
            "passed": passed,
            "failure_action": (
                None
                if passed
                else "do not lock an independent evaluation from this router"
            ),
        },
        "input_sha256": {"banks": bank_inputs},
        "information_boundary": {
            "transfer_panel_previously_opened_during_method_development": True,
            "router_decisions_use_outcome": False,
            "outcomes_used_only_after_all_decisions": True,
            "sealed_development_episode_read": False,
            "fresh_or_confirmatory_data_read": False,
            "pokeflex_target_read": False,
        },
        "passed": passed,
        "claim_boundary": (
            "exploratory cross-object transfer only; passing admits a new "
            "prospective lock and does not establish state of the art"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _require(not args.output.exists(), f"transfer report exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "checks": checks,
                "future": future,
                "late": late,
                "accepted_count": sum(row["accepted"] for row in rows),
                "episode_count": len(rows),
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
