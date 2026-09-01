#!/usr/bin/env python3
"""Build the stage-aware simultaneous certificate for the DLO-Lab atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin.query_competence_atlas_v2 import (
    QueryCompetenceStageV2,
    load_query_competence_atlas,
)
from bayesian_phystwin.query_portfolio_certificate_v1 import (
    QueryPortfolioCertificateV1,
    QueryPortfolioMemberV1,
    save_query_portfolio_certificate,
)
from bayesian_phystwin_experiments.deform_state_restart import array_digest

ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = Path("results/source/dlolab_query_competence_atlas_v5/atlas.json")
WRAPPING_SUMMARY_PATH = Path(
    "results/sota/dlolab_wrapping_risk_certified_guard_source_v9/summary.json"
)
SLINGSHOT_SUMMARY_PATH = Path(
    "results/source/dlolab_slingshot_policy_certificate_source_v4/summary.json"
)
ATLAS_ID = "82aef94511f3e0db1746262d4d49ae3ff9e52a587c5c11ce41cc817faa7a7ab9"
ATLAS_SHA256 = "2c9b13c10f6d89ca568bcdcd9fc1cc5b30d9443863323fac7478ccfe8541766a"
WRAPPING_SUMMARY_ID = "d3c577ce1ec215c6d56c4d405e7f9d886f38b7e6d021bb6d62f37da6bd4784b9"
WRAPPING_SUMMARY_SHA256 = (
    "45877e22f1af55ba4f5c7e1a66ab213e148733c9a3da9d82a1dafe545b77a4d1"
)
SLINGSHOT_SUMMARY_ID = (
    "2882809b7265714a93be2d3f1455eeac527adbe681cc990cde762777fcaf3a85"
)
SLINGSHOT_SUMMARY_SHA256 = (
    "cfbab2f371ec606fdbcf844cc8484f543a57829780f893f1b9bf3359dbae2564"
)
WRAPPING_GAIN_SHA256 = (
    "fd564dc627c68be8e8df60b4ad4da8a3983345ba2571bb70b3e0d302fc50b701"
)
SLINGSHOT_GAIN_SHA256 = (
    "adbedc553c6f2694a1beed5b1b538bb0a4129efbf0dc0bd81295bbabcce56469"
)
WORLD_COUNT = 288
FAMILYWISE_CONFIDENCE = 0.95
HARM_RISK_BUDGET = 0.05
FINAL_RISK_QUERY_COUNT = 3
ADJUSTED_LOWER_QUANTILE = (1.0 - FAMILYWISE_CONFIDENCE) / FINAL_RISK_QUERY_COUNT

FloatArray: TypeAlias = NDArray[np.float64]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_json(
    relative_path: Path,
    *,
    expected_sha256: str,
    expected_id: str,
    label: str,
) -> Mapping[str, Any]:
    path = ROOT / relative_path
    if path.is_symlink() or _file_digest(path) != expected_sha256:
        raise ValueError(f"{label} bytes changed")
    value = cast(Mapping[str, Any], load_strict_json_object(path, label=label))
    descriptor = dict(value)
    observed_id = descriptor.pop("artifact_id", None)
    if observed_id != expected_id or observed_id != content_id(descriptor):
        raise ValueError(f"{label} content identity changed")
    return value


def _atlas() -> tuple[QueryCompetenceStageV2, ...]:
    path = ROOT / ATLAS_PATH
    if path.is_symlink() or _file_digest(path) != ATLAS_SHA256:
        raise ValueError("query atlas v5 bytes changed")
    atlas = load_query_competence_atlas(path)
    if atlas.artifact_id != ATLAS_ID or len(atlas.entries) != 6:
        raise ValueError("query atlas v5 identity or roster changed")
    risk_evaluable = tuple(
        entry for entry in atlas.entries if entry.prospective_risk != "not_evaluated"
    )
    certified = tuple(entry for entry in atlas.entries if entry.decision == "certified")
    if (
        len(risk_evaluable) != FINAL_RISK_QUERY_COUNT
        or len(certified) != 2
        or any(not entry.exact_fallback_retained for entry in atlas.entries)
    ):
        raise ValueError("atlas stage or exact-fallback roster changed")
    return tuple(atlas.entries)


def _load_npz(path: Path, *, required: frozenset[str]) -> dict[str, NDArray[Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing ordinary source bundle: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != required:
            raise ValueError(f"source bundle members changed: {path}")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _bootstrap_interval(
    gain: FloatArray,
    *,
    seed: int,
    replicates: int,
    lower_quantile: float,
) -> tuple[float, float, float]:
    if gain.shape != (WORLD_COUNT,) or not np.isfinite(gain).all():
        raise ValueError("one finite gain per evaluation world required")
    indices = np.random.default_rng(seed).integers(
        0,
        WORLD_COUNT,
        size=(replicates, WORLD_COUNT),
    )
    means = gain[indices].mean(axis=1)
    quantiles: FloatArray = np.asarray(
        np.quantile(means, [lower_quantile, 0.025, 0.975]),
        dtype=np.float64,
    )
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def _wrapping_gain(raw_root: Path, summary: Mapping[str, Any]) -> FloatArray:
    decision_bundle = _load_npz(
        raw_root / "decisions/arrays.npz",
        required=frozenset(
            {
                "truth_prefix_m",
                "decisions",
                "guarded_posterior_expected_gain_over_fixed",
                "guarded_posterior_improvement_probability",
                "continuous_posterior_entropy_nats",
                "continuous_posterior_expected_reward",
                "continuous_posterior_improvement_probability",
                "continuous_prior_best_fixed_action",
                "quadrature_normalized_log_material_coordinates",
                "quadrature_prior_weight",
            }
        ),
    )
    generation = _load_npz(
        raw_root / "generation/arrays.npz",
        required=frozenset({"reward"}),
    )
    decisions = np.asarray(decision_bundle["decisions"], dtype=np.int64)
    rewards = np.asarray(generation["reward"], dtype=np.float64)
    if decisions.shape != (WORLD_COUNT, 4096, 7) or rewards.shape != (WORLD_COUNT, 8):
        raise ValueError("complete wrapping source denominator required")
    world_reward = np.take_along_axis(rewards[:, None, :], decisions, axis=2).mean(
        axis=1
    )
    gain: FloatArray = np.ascontiguousarray(
        world_reward[:, 2] - world_reward[:, 0],
        dtype=np.float64,
    )
    if array_digest(gain) != WRAPPING_GAIN_SHA256:
        raise ValueError("wrapping gain vector changed")
    lower, ci95_lower, ci95_upper = _bootstrap_interval(
        gain,
        seed=261912,
        replicates=20_000,
        lower_quantile=ADJUSTED_LOWER_QUANTILE,
    )
    if (
        float(gain.mean()) != summary.get("guard_gain_over_fixed")
        or [ci95_lower, ci95_upper] != summary.get("guard_gain_ci95")
        or int(np.count_nonzero(gain < -0.002)) != summary.get("guard_harmed_worlds")
        or abs(lower - 0.0038302803137030186) > 1e-15
    ):
        raise ValueError("wrapping compact and raw evidence disagree")
    return gain


def _slingshot_gain(raw_root: Path, summary: Mapping[str, Any]) -> FloatArray:
    decision_bundle = _load_npz(
        raw_root / "evaluation-decisions/arrays.npz",
        required=frozenset(
            {
                "truth_prefix_m",
                "observation_m",
                "features",
                "expected_losses",
                "mean_raw_upper",
                "candidate_actions",
                "predicted_gain",
                "neighbor_indices",
                "neighbor_squared_distances",
                "decisions",
                "accepted_mask",
                "simultaneous_accepted_mask",
                "lower_gain_bound",
            }
        ),
    )
    decisions = np.asarray(decision_bundle["decisions"], dtype=np.int64)
    rewards: list[list[float]] = []
    for index in range(WORLD_COUNT):
        path = raw_root / f"evaluation-future-{index:03d}-qualification.json"
        if path.is_symlink():
            raise ValueError("Slingshot qualification cannot be a symlink")
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if (
            value.get("role") != "evaluation"
            or value.get("world", {}).get("index") != index
            or value.get("qa", {}).get("qa_passed") is not True
        ):
            raise ValueError("ordinary Slingshot world qualification required")
        row = value.get("rewards")
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("seven reward-aligned actions required")
        rewards.append([float(item) for item in row])
    reward = np.asarray(rewards, dtype=np.float64)
    if decisions.shape != (WORLD_COUNT, 4) or not np.isfinite(reward).all():
        raise ValueError("complete Slingshot source denominator required")
    selected = np.take_along_axis(reward, decisions, axis=1)
    gain: FloatArray = np.ascontiguousarray(
        selected[:, 3] - reward[:, 5],
        dtype=np.float64,
    )
    if array_digest(gain) != SLINGSHOT_GAIN_SHA256:
        raise ValueError("Slingshot gain vector changed")
    lower, ci95_lower, ci95_upper = _bootstrap_interval(
        gain,
        seed=262054,
        replicates=20_000,
        lower_quantile=ADJUSTED_LOWER_QUANTILE,
    )
    primary = cast(
        Mapping[str, Any], cast(Mapping[str, Any], summary["arms"])["policy_gain_guard"]
    )
    if (
        float(gain.mean()) != primary.get("mean_gain_over_incumbent")
        or [ci95_lower, ci95_upper] != primary.get("mean_gain_ci95")
        or int(np.count_nonzero(gain < -0.002))
        != primary.get("harmful_worlds_beyond_numeric_margin")
        or abs(lower - 0.0013593553944870277) > 1e-15
    ):
        raise ValueError("Slingshot compact and raw evidence disagree")
    return gain


def _task(entry: QueryCompetenceStageV2) -> str:
    value = entry.query_scope.metadata.get("task")
    if not isinstance(value, str):
        raise ValueError("atlas query task changed")
    return value


def build_certificate(
    *,
    wrapping_root: Path,
    slingshot_root: Path,
) -> QueryPortfolioCertificateV1:
    entries = _atlas()
    wrapping_summary = _bound_json(
        WRAPPING_SUMMARY_PATH,
        expected_sha256=WRAPPING_SUMMARY_SHA256,
        expected_id=WRAPPING_SUMMARY_ID,
        label="wrapping v9 summary",
    )
    slingshot_summary = _bound_json(
        SLINGSHOT_SUMMARY_PATH,
        expected_sha256=SLINGSHOT_SUMMARY_SHA256,
        expected_id=SLINGSHOT_SUMMARY_ID,
        label="Slingshot v4 summary",
    )
    wrapping_gain = _wrapping_gain(wrapping_root, wrapping_summary)
    slingshot_gain = _slingshot_gain(slingshot_root, slingshot_summary)
    wrapping_entry = next(
        entry
        for entry in entries
        if _task(entry) == "wrapping" and entry.decision == "certified"
    )
    slingshot_entry = next(
        entry
        for entry in entries
        if _task(entry) == "slingshot"
        and entry.query_scope.metadata.get("version") == "reward-aligned-v4"
    )
    wrapping_lower = _bootstrap_interval(
        wrapping_gain,
        seed=261912,
        replicates=20_000,
        lower_quantile=ADJUSTED_LOWER_QUANTILE,
    )[0]
    slingshot_lower = _bootstrap_interval(
        slingshot_gain,
        seed=262054,
        replicates=20_000,
        lower_quantile=ADJUSTED_LOWER_QUANTILE,
    )[0]
    primary = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], slingshot_summary["arms"])["policy_gain_guard"],
    )
    posterior = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], slingshot_summary["arms"])["posterior_predictive_mean"],
    )
    family_count = sum(entry.prospective_risk != "not_evaluated" for entry in entries)
    per_query_confidence = 1.0 - (1.0 - FAMILYWISE_CONFIDENCE) / family_count
    deployed = {
        wrapping_entry.query_scope.query_id: QueryPortfolioMemberV1(
            query_id=cast(str, wrapping_entry.query_scope.query_id),
            decision="certified",
            prospective_risk_evaluated=True,
            candidate_deployed=True,
            exact_fallback_selected=False,
            evidence_artifact_id=wrapping_entry.evidence_artifact_id,
            evidence_file_sha256=wrapping_entry.evidence_file_sha256,
            independent_groups=WORLD_COUNT,
            harmful_groups=int(wrapping_summary["guard_harmed_worlds"]),
            unguarded_harmful_groups=int(
                wrapping_summary["continuous_bayes_harmed_worlds"]
            ),
            mean_gain_over_fallback=float(wrapping_gain.mean()),
            familywise_gain_lower=wrapping_lower,
            familywise_harm_upper=0.020813273498617287,
            gain_vector_sha256=WRAPPING_GAIN_SHA256,
            metadata={
                "task": "wrapping",
                "source_summary_id": WRAPPING_SUMMARY_ID,
                "source_summary_sha256": WRAPPING_SUMMARY_SHA256,
                "verified_raw_tree_id": wrapping_summary["verified_tree_id"],
                "bootstrap_seed": 261912,
                "bootstrap_replicates": 20_000,
                "adjusted_lower_quantile": ADJUSTED_LOWER_QUANTILE,
                "registered_ci95": wrapping_summary["guard_gain_ci95"],
                "per_query_confidence": per_query_confidence,
            },
        ),
        slingshot_entry.query_scope.query_id: QueryPortfolioMemberV1(
            query_id=cast(str, slingshot_entry.query_scope.query_id),
            decision="certified",
            prospective_risk_evaluated=True,
            candidate_deployed=True,
            exact_fallback_selected=False,
            evidence_artifact_id=slingshot_entry.evidence_artifact_id,
            evidence_file_sha256=slingshot_entry.evidence_file_sha256,
            independent_groups=WORLD_COUNT,
            harmful_groups=int(primary["harmful_worlds_beyond_numeric_margin"]),
            unguarded_harmful_groups=int(
                posterior["harmful_worlds_beyond_numeric_margin"]
            ),
            mean_gain_over_fallback=float(slingshot_gain.mean()),
            familywise_gain_lower=slingshot_lower,
            familywise_harm_upper=0.04706922523142958,
            gain_vector_sha256=SLINGSHOT_GAIN_SHA256,
            metadata={
                "task": "slingshot",
                "version": "reward-aligned-v4",
                "source_summary_id": SLINGSHOT_SUMMARY_ID,
                "source_summary_sha256": SLINGSHOT_SUMMARY_SHA256,
                "verified_raw_tree_id": cast(
                    Mapping[str, Any], slingshot_summary["raw_tree"]
                )["canonical_tree_sha256"],
                "bootstrap_seed": 262054,
                "bootstrap_replicates": 20_000,
                "adjusted_lower_quantile": ADJUSTED_LOWER_QUANTILE,
                "registered_ci95": primary["mean_gain_ci95"],
                "per_query_confidence": per_query_confidence,
            },
        ),
    }
    members: list[QueryPortfolioMemberV1] = []
    for entry in entries:
        query_id = cast(str, entry.query_scope.query_id)
        if query_id in deployed:
            members.append(deployed[query_id])
            continue
        members.append(
            QueryPortfolioMemberV1(
                query_id=query_id,
                decision="rejected",
                prospective_risk_evaluated=(entry.prospective_risk != "not_evaluated"),
                candidate_deployed=False,
                exact_fallback_selected=True,
                evidence_artifact_id=entry.evidence_artifact_id,
                evidence_file_sha256=entry.evidence_file_sha256,
                metadata={
                    "task": _task(entry),
                    "first_failed_stage": entry.first_failed_stage,
                    "terminal_reason": entry.terminal_reason,
                },
            )
        )
    return QueryPortfolioCertificateV1(
        atlas_id=ATLAS_ID,
        atlas_file_sha256=ATLAS_SHA256,
        members=members,
        familywise_confidence=FAMILYWISE_CONFIDENCE,
        harm_risk_budget=HARM_RISK_BUDGET,
        component_trials_prospective=True,
        portfolio_synthesis_posthoc=True,
        selector_must_be_outcome_independent=True,
        metadata={
            "public_simulator": "DLO-Lab",
            "calibration_worlds": 272,
            "evaluation_worlds": 576,
            "risk_evaluable_query_count": FINAL_RISK_QUERY_COUNT,
            "gain_interval_method": (
                "registered-percentile-bootstrap-with-bonferroni-lower-tail-v1"
            ),
            "harm_interval_method": (
                "exact-one-sided-clopper-pearson-with-bonferroni-v1"
            ),
            "query_reward_units_pooled": False,
            "new_recordings": False,
            "protected_data_read": False,
            "held_v8_read": False,
            "dlo4_dlo5_read": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wrapping-root",
        type=Path,
        default=Path(
            "/home/fpfaff/source-only/dlolab-wrapping-risk-certified-guard-source-v9"
        ),
    )
    parser.add_argument(
        "--slingshot-root",
        type=Path,
        default=Path(
            "/home/fpfaff/source-only/dlolab-slingshot-policy-certificate-source-v4"
        ),
    )
    args = parser.parse_args()
    certificate = build_certificate(
        wrapping_root=args.wrapping_root,
        slingshot_root=args.slingshot_root,
    )
    save_query_portfolio_certificate(args.output, certificate)
    print(f"certificate_id={certificate.artifact_id}")


if __name__ == "__main__":
    main()
