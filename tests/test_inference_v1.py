from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.inference._guarded as guarded_module
from bayesian_phystwin.inference.v1 import (
    ClaimBearingProb4DCandidateV1,
    CompleteBeliefGuardDecisionV1,
    GuardedUpdateResultV1,
    ObservationBeliefV1,
    PhysicalLinearizationV1,
    finalize_guarded_update,
    infer_prob4d_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "api/inference-public-api-v1.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DummyBelief:
    artifact_id: str


@dataclass(frozen=True)
class DummyInference:
    candidate_id: str
    inference_admissible: bool


def _decision(
    baseline: DummyBelief,
    candidate: DummyBelief,
    *,
    inference_admissible: bool = True,
    regret_guard_accepted: bool = True,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("domain"),
        certificate_id=_digest("certificate"),
        inference_admissible=inference_admissible,
        regret_guard_accepted=regret_guard_accepted,
        reason="unit-test-decision",
        metadata={"source": "unit-test"},
    )


def test_finalize_guarded_update_reuses_accepted_candidate() -> None:
    baseline = DummyBelief(_digest("baseline"))
    candidate = DummyBelief(_digest("candidate"))
    inference = DummyInference(_digest("inference"), True)

    result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        _decision(baseline, candidate),
        metadata={"case": "accepted"},
    )

    assert isinstance(result, GuardedUpdateResultV1)
    assert result.selected_belief is candidate
    assert result.selected_candidate is True
    assert result.exact_fallback is False
    assert result.to_record()["selected_belief_id"] == candidate.artifact_id
    assert result.to_record()["inference_candidate_id"] == inference.candidate_id
    assert len(result.artifact_id) == 64
    with pytest.raises(TypeError, match="immutable"):
        result.metadata["case"] = "tampered"  # type: ignore[index]


def test_finalize_guarded_update_returns_exact_baseline_on_rejection() -> None:
    baseline = DummyBelief(_digest("baseline"))
    candidate = DummyBelief(_digest("candidate"))
    inference = DummyInference(_digest("inference"), True)

    result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        _decision(
            baseline,
            candidate,
            regret_guard_accepted=False,
        ),
        metadata={"case": "rejected"},
    )

    assert result.selected_belief is baseline
    assert result.selected_candidate is False
    assert result.exact_fallback is True
    assert result.selection.reason == "regret-guard-rejected"


def test_finalize_guarded_update_fails_closed_on_cross_contract_drift() -> None:
    baseline = DummyBelief(_digest("baseline"))
    candidate = DummyBelief(_digest("candidate"))

    with pytest.raises(ValueError, match="inference admissibility"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), False),
            baseline,
            candidate,
            _decision(baseline, candidate, inference_admissible=True),
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        finalize_guarded_update(
            DummyInference("not-a-digest", True),
            baseline,
            candidate,
            _decision(baseline, candidate),
        )

    with pytest.raises(TypeError, match="metadata"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            baseline,
            candidate,
            _decision(baseline, candidate),
            metadata=0,  # type: ignore[arg-type]
        )


def test_infer_prob4d_candidate_rejects_falsey_invalid_configs() -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    prediction = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(TypeError, match="config"):
        infer_prob4d_candidate(
            observation,
            linearization,
            physical_prediction_xyz_m=prediction,
            config=0,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="covariance_semantics"):
        infer_prob4d_candidate(
            observation,
            linearization,
            physical_prediction_xyz_m=prediction,
            covariance_semantics=0,  # type: ignore[arg-type]
        )


def test_infer_prob4d_candidate_delegates_to_the_strict_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    prediction = np.zeros((1, 3), dtype=np.float64)
    candidate = object.__new__(ClaimBearingProb4DCandidateV1)
    captured: dict[str, Any] = {}

    def fake_infer(*args: object, **kwargs: object) -> ClaimBearingProb4DCandidateV1:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return candidate

    monkeypatch.setattr(
        guarded_module,
        "infer_claim_bearing_prob4d_candidate_from_artifacts",
        fake_infer,
    )

    result = infer_prob4d_candidate(
        observation,
        linearization,
        physical_prediction_xyz_m=prediction,
    )

    assert result is candidate
    assert captured["args"] == (observation, linearization)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["physical_prediction_xyz_m"] is prediction
    assert kwargs["config"] is None
    assert kwargs["covariance_semantics"] is None


def test_inference_v1_surface_matches_its_exact_snapshot() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    module = __import__("bayesian_phystwin.inference.v1", fromlist=["*"])

    assert manifest["schema"] == "bayesian-phystwin.inference-public-api-snapshot"
    assert manifest["schema_version"] == 1
    assert manifest["package"] == "bayesian_phystwin.inference.v1"
    assert manifest["compatibility_line"] == "0.4"
    assert manifest["policy"] == "exact-guarded-inference-export-surface"
    assert list(module.__all__) == manifest["symbols"]
    for name in manifest["symbols"]:
        assert getattr(module, name) is not None


def test_inference_v1_import_does_not_load_optional_or_experiment_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent(
                """
                import json
                import sys

                import bayesian_phystwin.inference.v1

                forbidden_external = {
                    "cv2", "h5py", "numpyro", "pymc", "scipy", "torch", "warp"
                }
                forbidden_package_prefixes = (
                    "bayesian_phystwin.deform360_",
                    "bayesian_phystwin.experiments",
                    "bayesian_phystwin.phystwin_",
                )
                leaked = sorted(
                    name
                    for name in sys.modules
                    if name in forbidden_external
                    or name.startswith(forbidden_package_prefixes)
                )
                print(json.dumps({"leaked": leaked}))
                """
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"leaked": []}


def test_inference_v1_assets_are_in_the_source_distribution() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include api/inference-public-api-v1.json" in manifest
    assert "include docs/inference_v1.md" in manifest
    assert "include examples/guarded_inference_v1.py" in manifest
