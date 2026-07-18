#!/usr/bin/env python3
"""Cross-fit and freeze a competence router on exhausted Deform360 sources."""

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
    cross_fit_causal_expert_router,
    fit_causal_expert_router,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--calibration-level", type=float, default=0.9)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-degradation", type=float, default=0.1)
    return parser.parse_args()


def _load_episode(
    path: Path,
    *,
    expected_result_sha256: str,
    feature_names: tuple[str, ...] | None,
) -> tuple[CausalExpertEpisode, tuple[str, ...], dict[str, Any]]:
    bank = json.loads(path.read_text(encoding="utf-8"))
    _require(
        bank.get("artifact_kind") == "Deform360CausalTransportExhaustedSourceBank"
        and bank.get("result_sha256") == expected_result_sha256
        and bank.get("result_sha256") == _canonical_sha256(bank)
        and bank.get("passed") is True,
        f"source bank is incompatible: {path}",
    )
    records = bank["records"]
    labels = tuple(str(record["label"]) for record in records)
    _require(
        labels == tuple(bank["candidate_order"])
        and len(labels) == int(bank["candidate_count"])
        and labels[0] == "persistence",
        f"candidate order changed: {path}",
    )
    candidate_features = [
        build_causal_expert_features(record) for record in records[1:]
    ]
    names = (
        tuple(sorted(candidate_features[0])) if feature_names is None else feature_names
    )
    _require(
        all(set(features) == set(names) for features in candidate_features),
        f"feature contract changed: {path}",
    )
    matrix = np.asarray(
        [[features[name] for name in names] for features in candidate_features],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            normalized_candidate_score(record, bank["persistence_metrics"])
            for record in records
        ],
        dtype=np.float64,
    )
    episode = CausalExpertEpisode(
        object_id=str(bank["object_id"]),
        episode_id=int(bank["episode_id"]),
        labels=labels,
        features=matrix,
        normalized_scores=scores,
    )
    return episode, names, bank


def main() -> int:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _require(
        manifest.get("artifact_kind")
        == "Deform360CausalTransportExhaustedSourceManifest"
        and manifest.get("result_sha256") == _canonical_sha256(manifest)
        and manifest.get("passed") is True,
        "source manifest is incompatible",
    )
    episodes = []
    feature_names = None
    bank_inputs = {}
    for item in manifest["outputs"]:
        path = Path(item["path"])
        _require(
            path.is_file() and _sha256_file(path) == item["file_sha256"],
            f"source bank file changed: {path}",
        )
        episode, feature_names, bank = _load_episode(
            path,
            expected_result_sha256=str(item["result_sha256"]),
            feature_names=feature_names,
        )
        episodes.append(episode)
        bank_inputs[f"{episode.object_id}-ep{episode.episode_id:04d}"] = {
            "path": str(path.resolve()),
            "file_sha256": item["file_sha256"],
            "result_sha256": bank["result_sha256"],
        }
    assert feature_names is not None
    _require(
        len(episodes) == int(manifest["episode_count"])
        and len({episode.object_id for episode in episodes})
        == int(manifest["object_count"]),
        "source manifest counts changed",
    )
    cross_fit = cross_fit_causal_expert_router(
        episodes,
        feature_names=feature_names,
        calibration_level=args.calibration_level,
        minimum_improvement_fraction=args.minimum_improvement,
        maximum_cross_fitted_degradation_fraction=args.maximum_degradation,
    )
    model, full_fit = fit_causal_expert_router(
        episodes,
        feature_names=feature_names,
        calibration_level=args.calibration_level,
        minimum_improvement_fraction=args.minimum_improvement,
        maximum_cross_fitted_degradation_fraction=args.maximum_degradation,
    )
    object_scores = cross_fit["object_mean_normalized_scores"]
    checks = {
        "mean_improvement": cross_fit["mean_normalized_score"] <= 0.98,
        "maximum_degradation": cross_fit["maximum_normalized_score"]
        <= 1.0 + args.maximum_degradation,
        "no_object_mean_degradation": all(
            float(value) <= 1.0 for value in object_scores.values()
        ),
        "full_fit_inner_safety": bool(full_fit["safety_passed"]),
    }
    passed = all(checks.values())
    model_payload = model.to_payload(
        source={
            "manifest_path": str(args.manifest.resolve()),
            "manifest_file_sha256": _sha256_file(args.manifest),
            "manifest_result_sha256": manifest["result_sha256"],
            "source_episode_count": len(episodes),
            "source_object_count": len({episode.object_id for episode in episodes}),
            "source_bank_inputs": bank_inputs,
            "source_gate_passed": passed,
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalExpertRouterSourceReport",
        "model_result_sha256": model_payload["result_sha256"],
        "cross_fit": cross_fit,
        "full_fit": full_fit,
        "source_gate": {
            "checks": checks,
            "thresholds": {
                "maximum_mean_normalized_score": 0.98,
                "maximum_episode_normalized_score": 1.0 + args.maximum_degradation,
                "maximum_object_mean_normalized_score": 1.0,
            },
            "passed": passed,
            "failure_action": (
                None if passed else "do not promote router to independent evaluation"
            ),
        },
        "input_sha256": {
            "manifest": _sha256_file(args.manifest),
            "banks": bank_inputs,
        },
        "information_boundary": {
            "source_panel_previously_exhausted": True,
            "outer_cross_fit_unit": "object",
            "candidate_decisions_use_outcome": False,
            "source_outcomes_used_for_training_and_scoring": True,
            "fresh_or_confirmatory_data_read": False,
            "pokeflex_target_read": False,
        },
        "passed": passed,
        "claim_boundary": (
            "retrospective source discovery only; a pass admits exploratory "
            "cross-object transfer but is not a state-of-the-art claim"
        ),
    }
    report["result_sha256"] = _canonical_sha256(report)
    for path in (args.model_output, args.report_output):
        _require(not path.exists(), f"output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(
        json.dumps(model_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "gate_checks": checks,
                "mean_normalized_score": cross_fit["mean_normalized_score"],
                "maximum_normalized_score": cross_fit["maximum_normalized_score"],
                "win_fraction": cross_fit["win_fraction"],
                "accepted_fraction": cross_fit["accepted_fraction"],
                "model_result_sha256": model_payload["result_sha256"],
                "result_sha256": report["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
