"""Materialize accepted and exact-fallback decisions from the installed wheels."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
from causal4d.observation_lineage import load_observation_lineage
from prob4d.provider_v1 import save_observation_belief_export
from test_three_repository_evidence import (
    _bound_counterfactual_artifacts,
    _exact_producer_artifact,
    _manifest,
    _revision,
    _wheel_digest,
)
from test_three_repository_golden_path import _run_bpt_update

from bayesian_phystwin import (
    load_observation_belief,
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.evidence_policy import require_promotable_run_manifest
from bayesian_phystwin.gauge_aware_belief import select_gauge_aware_candidate
from bayesian_phystwin.run_manifest_v2 import write_run_manifest
from bayesian_phystwin.three_repository_golden_path_artifacts_v1 import (
    GoldenPathEvidenceBundleV1,
    build_golden_path_selection_artifact_v1,
    load_golden_path_evidence_bundle_v1,
    write_golden_path_evidence_bundle_v1,
)


@dataclass(frozen=True)
class _RegretGuardDecision:
    selected_value: np.ndarray
    candidate_accepted: bool
    reason: str


def _distinct_physical_arrays(posterior: object) -> tuple[np.ndarray, np.ndarray]:
    trajectories = np.asarray(
        posterior.readout_trajectories_m,  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    if trajectories.ndim < 2 or len(trajectories) < 2:
        raise AssertionError("golden-path posterior has insufficient components")
    baseline = np.ascontiguousarray(trajectories[0])
    for index in range(1, len(trajectories)):
        candidate = np.ascontiguousarray(trajectories[index])
        if candidate.tobytes(order="C") != baseline.tobytes(order="C"):
            return baseline, candidate
    raise AssertionError("golden-path posterior components are byte-identical")


def _component_revisions() -> dict[str, str]:
    return {
        "bayesian_phystwin": _revision("BAYESIAN_PHYSTWIN_REVISION"),
        "prob4d": _revision("PROB4D_REVISION"),
        "causal4d": _revision("CAUSAL4D_REVISION"),
    }


def _component_wheels() -> dict[str, str]:
    return {
        "bayesian_phystwin": _wheel_digest(
            "BAYESIAN_PHYSTWIN_WHEEL_SHA256"
        ),
        "prob4d": _wheel_digest("PROB4D_WHEEL_SHA256"),
        "causal4d": _wheel_digest("CAUSAL4D_WHEEL_SHA256"),
    }


def _component_versions() -> dict[str, str]:
    return {
        "bayesian_phystwin": importlib_metadata.version(
            "bayesian-phystwin"
        ),
        "prob4d": importlib_metadata.version("prob4d"),
        "causal4d": importlib_metadata.version("causal4d"),
    }


def _copy_manifest_artifacts(
    output: Path,
    *paths: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in output.iterdir())
    if unexpected:
        raise FileExistsError(
            f"golden-path evidence output is not empty: {unexpected}"
        )
    for source in paths:
        shutil.copy2(source, output / source.name)


def test_materialize_accepted_and_exact_fallback_artifacts(
    tmp_path: Path,
) -> None:
    producer = _exact_producer_artifact()
    observation_path = tmp_path / "exact-prob4d-observation.npz"
    save_observation_belief_export(observation_path, producer)
    observation = load_observation_belief(observation_path)
    validation = validate_prob4d_causal_observation_belief(observation)
    assert validation["stream_contract_version"] == 2

    lineage = load_observation_lineage(observation_path)
    bpt_result = _run_bpt_update(observation)
    assert bpt_result.inference_admissible

    (
        twin,
        posterior,
        profile_path,
        twin_path,
        posterior_path,
        provider_manifest,
    ) = _bound_counterfactual_artifacts(tmp_path, lineage, bpt_result)
    manifest = _manifest(
        tmp_path,
        observation_path,
        profile_path,
        twin_path,
        posterior_path,
        observation_id=observation.artifact_id,
        twin_id=twin.artifact_id,
        posterior_id=posterior.artifact_id,
        provider_manifest_id=provider_manifest.manifest_id,
    )
    admission = require_promotable_run_manifest(manifest, root=tmp_path)
    assert admission["status"] == "promotable"

    baseline, candidate = _distinct_physical_arrays(posterior)
    accepted_selection = select_gauge_aware_candidate(
        baseline,
        candidate,
        bpt_result,
        regret_decision=_RegretGuardDecision(
            selected_value=candidate.copy(),
            candidate_accepted=True,
            reason="registered-guard-accepted",
        ),
    )
    rejected_selection = select_gauge_aware_candidate(
        baseline,
        candidate,
        bpt_result,
        regret_decision=_RegretGuardDecision(
            selected_value=baseline.copy(),
            candidate_accepted=False,
            reason="registered-guard-rejected",
        ),
    )
    assert accepted_selection.candidate_accepted
    assert not rejected_selection.candidate_accepted
    assert accepted_selection.selected_value.tobytes() == candidate.tobytes()
    assert rejected_selection.selected_value.tobytes() == baseline.tobytes()

    common = {
        "case_id": "three-repository-golden-path",
        "protocol_id": "three-repository-installed-wheel-v1",
        "observation_artifact_id": observation.artifact_id,
        "twin_belief_id": twin.artifact_id,
        "physical_posterior_id": posterior.artifact_id,
        "provider_manifest_id": provider_manifest.manifest_id,
        "run_manifest_id": manifest.manifest_id,
        "evidence_fingerprint": manifest.evidence_fingerprint,
        "repository_revisions": _component_revisions(),
        "wheel_sha256": _component_wheels(),
        "package_versions": _component_versions(),
        "metadata": {
            "run_classification": manifest.classification,
            "statistical_unit": manifest.statistical_unit,
            "claim_authorized": False,
            "selection_stage": "baseline-relative-regret-guard",
        },
    }
    accepted = build_golden_path_selection_artifact_v1(
        selection=accepted_selection,
        baseline=baseline,
        candidate=candidate,
        **common,
    )
    rejected = build_golden_path_selection_artifact_v1(
        selection=rejected_selection,
        baseline=baseline,
        candidate=candidate,
        **common,
    )
    bundle = GoldenPathEvidenceBundleV1(
        accepted=accepted,
        rejected=rejected,
    )
    assert bundle.rejected.exact_fallback_identity == (
        bundle.rejected.baseline_identity.array_id
    )
    assert bundle.accepted.selected_identity == bundle.accepted.candidate_identity

    output_value = os.environ.get("THREE_REPOSITORY_EVIDENCE_OUTPUT")
    if output_value is None:
        return
    output = Path(output_value).resolve()
    _copy_manifest_artifacts(
        output,
        observation_path,
        profile_path,
        twin_path,
        posterior_path,
    )
    write_run_manifest(output / "run-manifest-v2.json", manifest)
    write_golden_path_evidence_bundle_v1(output, bundle)

    require_promotable_run_manifest(manifest, root=output)
    restored = load_golden_path_evidence_bundle_v1(output)
    assert restored == bundle
