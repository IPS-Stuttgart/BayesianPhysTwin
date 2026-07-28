#!/usr/bin/env python3
"""Audit target-free observation budgets on an already-open V3 source case."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
    SET_VALUED_MIXTURE_ASSIMILATION,
    BirthAnchoredMeasurements,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    score_assimilation_trajectory,
)
from bayesian_phystwin.observation_belief import file_sha256
from bayesian_phystwin.phystwin_online_belief import (
    deterministic_farthest_point_ids,
)

BUDGETS = (1, 2, 4, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {
            name: np.ascontiguousarray(np.asarray(stored[name]))
            for name in stored.files
        }


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    _require(isinstance(payload, dict), "target payload is invalid")
    target = np.asarray(payload["object_points"], dtype=np.float64)
    visibility = np.asarray(payload["object_visibilities"], dtype=bool)
    validity = np.asarray(payload["object_motions_valid"], dtype=bool)
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "target arrays changed shape",
    )
    return target, visibility, validity


def _budgeted_measurements(
    provider: dict[str, np.ndarray],
    physical: np.ndarray,
    selected_ids: np.ndarray,
) -> BirthAnchoredMeasurements:
    entity_ids = np.asarray(provider["entity_ids"], dtype=np.int64)
    births = np.asarray(provider["birth_frames"], dtype=np.int64)
    updates = np.asarray(provider["update_frames"], dtype=np.int64)
    selected_set = set(map(int, selected_ids))
    selected_rows = np.asarray(
        [
            row
            for row, entity in enumerate(entity_ids)
            if int(entity) in selected_set
        ],
        dtype=np.int64,
    )
    _require(
        len(selected_rows) == len(selected_ids),
        "selected identities do not map one-to-one to provider rows",
    )
    selected_entities = entity_ids[selected_rows]
    measurement = np.full(physical.shape, np.nan, dtype=np.float64)
    covariance = np.full(
        (*physical.shape[:2], 3, 3),
        np.nan,
        dtype=np.float64,
    )
    reliability = np.zeros(physical.shape[:2], dtype=np.float64)
    association = np.zeros(physical.shape[:2], dtype=np.float64)
    available = np.zeros(physical.shape[:2], dtype=bool)
    for row, entity in zip(selected_rows, selected_entities, strict=True):
        birth = int(births[row])
        update = int(updates[row])
        if not (
            bool(provider["accepted_support"][birth, row])
            and bool(provider["accepted_support"][update, row])
        ):
            continue
        observed_displacement = (
            provider["trajectory_world_m"][update, row]
            - provider["trajectory_world_m"][birth, row]
        )
        measurement[update, entity] = (
            physical[birth, entity] + observed_displacement
        )
        covariance[update, entity] = 2.0 * (
            provider["local_covariance_m2"][birth, row]
            + provider["local_covariance_m2"][update, row]
        )
        reliability[update, entity] = np.sqrt(
            provider["prior_reliability"][birth, row]
            * provider["prior_reliability"][update, row]
        )
        association[update, entity] = np.sqrt(
            provider["association_probability"][birth, row]
            * provider["association_probability"][update, row]
        )
        available[update, entity] = True
    return BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=selected_entities,
    )


def _score(
    trajectory: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden: np.ndarray,
) -> dict[str, float]:
    return score_assimilation_trajectory(
        trajectory,
        target,
        visibility,
        validity,
        hidden,
    )


def _distance_bands(
    frame_zero_m: np.ndarray,
    selected_ids: np.ndarray,
    hidden_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    nearest = np.min(
        np.linalg.norm(
            frame_zero_m[hidden_ids, None]
            - frame_zero_m[selected_ids][None],
            axis=2,
        ),
        axis=1,
    )
    first, second = np.quantile(nearest, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "near": hidden_ids[nearest <= first],
        "middle": hidden_ids[(nearest > first) & (nearest <= second)],
        "far": hidden_ids[nearest > second],
    }


def _per_identity_oracle(
    physical: np.ndarray,
    persistence: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    valid = visibility & validity & np.all(np.isfinite(target), axis=2)
    physical_error = np.full(len(hidden), np.inf)
    persistence_error = np.full(len(hidden), np.inf)
    for position, entity in enumerate(hidden):
        mask = valid[:, entity]
        if np.any(mask):
            physical_error[position] = np.sqrt(
                np.mean(
                    np.square(
                        physical[mask, entity] - target[mask, entity]
                    )
                )
            )
            persistence_error[position] = np.sqrt(
                np.mean(
                    np.square(
                        persistence[mask, entity] - target[mask, entity]
                    )
                )
            )
    choose_physical = physical_error < persistence_error
    oracle = persistence.copy()
    oracle[:, hidden[choose_physical]] = physical[:, hidden[choose_physical]]
    return oracle, {
        "physical_identity_count": int(np.sum(choose_physical)),
        "persistence_identity_count": int(np.sum(~choose_physical)),
    }


def main() -> int:
    args = _parse_args()
    run = args.run_dir.resolve()
    report_path = run / "source_development_report.json"
    provider_path = run / "provider_arrays.npz"
    assimilation_path = run / "assimilation_arrays.npz"
    for path in (report_path, provider_path, assimilation_path):
        _require(path.is_file(), f"V3 run artifact is missing: {path.name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("association_semantics") == "set-valued-v3"
        and report.get("status")
        == "post_open_source_development_not_confirmation"
        and report.get("result_sha256") == _canonical_sha256(report),
        "V3 source report is invalid",
    )
    provider = _load_npz(provider_path)
    assimilation = _load_npz(assimilation_path)
    physical = np.asarray(assimilation[PHYSICAL_ARM], dtype=np.float64)
    persistence = np.asarray(assimilation[PERSISTENCE_ARM], dtype=np.float64)
    target, visibility, validity = _load_target(args.target.resolve())
    _require(
        physical.shape == persistence.shape == target.shape,
        "prediction and target shapes differ",
    )

    births = provider["birth_frames"].astype(np.int64)
    updates = provider["update_frames"].astype(np.int64)
    rows = np.arange(len(provider["entity_ids"]), dtype=np.int64)
    supported = (
        provider["accepted_support"][births, rows]
        & provider["accepted_support"][updates, rows]
    )
    supported_ids = provider["entity_ids"][supported].astype(np.int64)
    _require(len(supported_ids) >= max(BUDGETS), "not enough supported identities")

    budgets: list[dict[str, Any]] = []
    all_ids = np.arange(target.shape[1], dtype=np.int64)
    for budget in BUDGETS:
        selected_ids = deterministic_farthest_point_ids(
            physical[0],
            supported_ids,
            budget,
        )
        measurements = _budgeted_measurements(
            provider,
            physical,
            selected_ids,
        )
        candidate_report, candidate_arrays = predict_dynamic_tapnextpp_candidate(
            physical,
            persistence,
            measurements,
            assimilation_mode=SET_VALUED_MIXTURE_ASSIMILATION,
        )
        hidden = np.setdiff1d(
            all_ids,
            selected_ids,
            assume_unique=True,
        )
        scores = {
            name: _score(
                trajectory,
                target,
                visibility,
                validity,
                hidden,
            )
            for name, trajectory in (
                (PHYSICAL_ARM, physical),
                (PERSISTENCE_ARM, persistence),
                (
                    SELECTED_BACKBONE_ARM,
                    candidate_arrays[SELECTED_BACKBONE_ARM],
                ),
                (CANDIDATE_ARM, candidate_arrays[CANDIDATE_ARM]),
            )
        }
        bands = {
            band: {
                name: _score(
                    trajectory,
                    target,
                    visibility,
                    validity,
                    band_ids,
                )
                for name, trajectory in (
                    (PERSISTENCE_ARM, persistence),
                    (
                        SELECTED_BACKBONE_ARM,
                        candidate_arrays[SELECTED_BACKBONE_ARM],
                    ),
                    (CANDIDATE_ARM, candidate_arrays[CANDIDATE_ARM]),
                )
            }
            for band, band_ids in _distance_bands(
                physical[0],
                selected_ids,
                hidden,
            ).items()
        }
        oracle, oracle_counts = _per_identity_oracle(
            physical,
            persistence,
            target,
            visibility,
            validity,
            hidden,
        )
        budgets.append(
            {
                "budget": budget,
                "selected_ids": selected_ids.tolist(),
                "selection_uses_target": False,
                "candidate_report": candidate_report,
                "scores": scores,
                "distance_bands": bands,
                "per_identity_oracle": {
                    "selection_uses_target": True,
                    "counts": oracle_counts,
                    "score": _score(
                        oracle,
                        target,
                        visibility,
                        validity,
                        hidden,
                    ),
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "DynamicTAPNextPPBudgetedHiddenTransferAudit",
        "status": "post_open_source_diagnostic_not_confirmation",
        "source_case_hash": report["case_hash"],
        "source_report_file_sha256": file_sha256(report_path),
        "provider_file_sha256": file_sha256(provider_path),
        "assimilation_file_sha256": file_sha256(assimilation_path),
        "target_file_sha256": file_sha256(args.target.resolve()),
        "supported_identity_count": len(supported_ids),
        "budgets": budgets,
        "information_boundary": {
            "subset_selection_uses_target": False,
            "per_identity_oracle_uses_target": True,
            "already_open_source_target_read": True,
            "fresh_target_read": False,
            "held_v8_artifact_read": False,
            "v1_sealed_target_cohort_read": False,
        },
        "claim_boundary": (
            "Post-open capacity and localization audit on one source case; "
            "not method selection evidence for any opened or sealed cohort."
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    output = args.output.resolve()
    _require(not output.exists(), "audit output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
