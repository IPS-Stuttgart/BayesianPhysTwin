from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.inference._guarded as guarded_module
from bayesian_phystwin._validation import lowercase_sha256, optional_instance
from bayesian_phystwin.inference.v1 import (
    ClaimBearingProb4DCandidateV1,
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    GuardedUpdateResultV1,
    ObservationBeliefV1,
    PhysicalLinearizationV1,
    PosteriorCovarianceSemanticsV1,
    PriorAwareGaugeConfigV1,
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


def _accepted_result() -> GuardedUpdateResultV1[DummyBelief]:
    baseline = DummyBelief(_digest("baseline"))
    candidate = DummyBelief(_digest("candidate"))
    return finalize_guarded_update(
        DummyInference(_digest("inference"), True),
        baseline,
        candidate,
        _decision(baseline, candidate),
        metadata={"case": "accepted"},
    )


def _fallback_result(
    *,
    inference_admissible: bool = True,
) -> GuardedUpdateResultV1[DummyBelief]:
    baseline = DummyBelief(_digest("fallback-baseline"))
    candidate = DummyBelief(_digest("fallback-candidate"))
    return finalize_guarded_update(
        DummyInference(_digest("fallback-inference"), inference_admissible),
        baseline,
        candidate,
        _decision(
            baseline,
            candidate,
            inference_admissible=inference_admissible,
            regret_guard_accepted=False,
        ),
    )


def _rebuild_result(
    source: GuardedUpdateResultV1[DummyBelief],
    **changes: object,
) -> GuardedUpdateResultV1[DummyBelief]:
    values: dict[str, object] = {
        field.name: getattr(source, field.name) for field in fields(source)
    }
    values.update(changes)
    return GuardedUpdateResultV1(**values)  # type: ignore[arg-type]


def _forge_selection(
    source: CompleteBeliefSelectionV1,
    **changes: object,
) -> CompleteBeliefSelectionV1:
    forged = object.__new__(CompleteBeliefSelectionV1)
    for field in fields(source):
        object.__setattr__(
            forged,
            field.name,
            changes.get(field.name, getattr(source, field.name)),
        )
    return forged


def test_fail_closed_validation_helpers_cover_all_outcomes() -> None:
    digest = _digest("validation")
    assert lowercase_sha256(digest, name="digest") == digest
    with pytest.raises(TypeError, match="must be a string"):
        lowercase_sha256(0, name="digest")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        lowercase_sha256("short", name="digest")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        lowercase_sha256("A" + digest[1:], name="digest")

    marker = DummyBelief(digest)
    assert optional_instance(None, DummyBelief, name="marker") is None
    assert optional_instance(marker, DummyBelief, name="marker") is marker
    with pytest.raises(TypeError, match="DummyBelief or None"):
        optional_instance(0, DummyBelief, name="marker")


def test_finalize_guarded_update_reuses_accepted_candidate() -> None:
    result = _accepted_result()

    assert isinstance(result, GuardedUpdateResultV1)
    assert result.selected_belief is result.candidate_belief
    assert result.selected_candidate is True
    assert result.exact_fallback is False
    assert (
        result.to_record()["selected_belief_id"] == result.candidate_belief.artifact_id
    )
    assert result.to_record()["inference_candidate_id"] == result.inference_candidate_id
    assert len(result.artifact_id) == 64
    assert result.artifact_id == result.artifact_id
    with pytest.raises(TypeError, match="immutable"):
        result.metadata["case"] = "tampered"  # type: ignore[index]


def test_finalize_guarded_update_returns_exact_baseline_on_both_rejections() -> None:
    regret = _fallback_result()
    inference = _fallback_result(inference_admissible=False)

    assert regret.selected_belief is regret.baseline_belief
    assert regret.selected_candidate is False
    assert regret.exact_fallback is True
    assert regret.selection.reason == "regret-guard-rejected"
    assert inference.selected_belief is inference.baseline_belief
    assert inference.selection.reason == "inference-rejected"


def test_finalize_guarded_update_fails_closed_on_invalid_public_inputs() -> None:
    baseline = DummyBelief(_digest("baseline"))
    candidate = DummyBelief(_digest("candidate"))
    decision = _decision(baseline, candidate)

    with pytest.raises(TypeError, match="must expose candidate_id"):
        finalize_guarded_update(
            cast(Any, object()),
            baseline,
            candidate,
            decision,
        )
    with pytest.raises(ValueError, match="inference admissibility"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), False),
            baseline,
            candidate,
            decision,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        finalize_guarded_update(
            DummyInference("not-a-digest", True),
            baseline,
            candidate,
            decision,
        )
    with pytest.raises(ValueError, match="must be a boolean"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), cast(Any, 1)),
            baseline,
            candidate,
            decision,
        )
    with pytest.raises(TypeError, match="baseline_belief must expose artifact_id"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            cast(Any, object()),
            candidate,
            decision,
        )
    with pytest.raises(TypeError, match="candidate_belief must expose artifact_id"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            baseline,
            cast(Any, object()),
            decision,
        )
    with pytest.raises(TypeError, match="must be a string"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            cast(Any, DummyBelief(cast(Any, 0))),
            candidate,
            decision,
        )
    with pytest.raises(TypeError, match="guard_decision"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            baseline,
            candidate,
            cast(Any, object()),
        )
    with pytest.raises(TypeError, match="metadata"):
        finalize_guarded_update(
            DummyInference(_digest("inference"), True),
            baseline,
            candidate,
            decision,
            metadata=cast(Any, 0),
        )


def test_guarded_update_result_rejects_mismatched_guard_bindings() -> None:
    accepted = _accepted_result()
    fallback = _fallback_result()

    with pytest.raises(TypeError, match="guard_decision"):
        _rebuild_result(accepted, guard_decision=object())
    with pytest.raises(TypeError, match="selection"):
        _rebuild_result(accepted, selection=object())
    with pytest.raises(ValueError, match="inference admissibility"):
        _rebuild_result(
            fallback,
            inference_admissible=not fallback.inference_admissible,
        )
    with pytest.raises(ValueError, match="baseline belief"):
        _rebuild_result(
            accepted,
            guard_decision=replace(
                accepted.guard_decision,
                baseline_belief_id=_digest("other-baseline"),
            ),
        )
    with pytest.raises(ValueError, match="candidate belief"):
        _rebuild_result(
            accepted,
            guard_decision=replace(
                accepted.guard_decision,
                candidate_belief_id=_digest("other-candidate"),
            ),
        )


def test_guarded_update_result_rejects_mismatched_selection_bindings() -> None:
    accepted = _accepted_result()
    fallback = _fallback_result()

    with pytest.raises(ValueError, match="selection does not bind the baseline"):
        _rebuild_result(
            accepted,
            selection=replace(
                accepted.selection,
                baseline_belief_id=_digest("other-selection-baseline"),
            ),
        )
    with pytest.raises(ValueError, match="selection does not bind the candidate"):
        _rebuild_result(
            fallback,
            selection=replace(
                fallback.selection,
                candidate_belief_id=_digest("other-selection-candidate"),
            ),
        )
    with pytest.raises(ValueError, match="selection does not bind the guard"):
        _rebuild_result(
            accepted,
            selection=replace(
                accepted.selection,
                guard_decision_id=_digest("other-guard"),
            ),
        )
    with pytest.raises(ValueError, match="selection does not bind the selected"):
        _rebuild_result(
            accepted,
            selection=_forge_selection(
                accepted.selection,
                selected_belief_id=_digest("other-selected"),
            ),
        )


def test_guarded_update_result_enforces_exact_selected_object_identity() -> None:
    accepted = _accepted_result()
    fallback = _fallback_result()

    with pytest.raises(ValueError, match="exact candidate belief object"):
        _rebuild_result(
            accepted,
            selected_belief=DummyBelief(accepted.candidate_belief.artifact_id),
        )
    with pytest.raises(ValueError, match="exact baseline belief object"):
        _rebuild_result(
            fallback,
            selected_belief=DummyBelief(fallback.baseline_belief.artifact_id),
        )


def test_guarded_update_result_revalidates_constructor_scalars() -> None:
    accepted = _accepted_result()

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _rebuild_result(accepted, inference_candidate_id="bad")
    with pytest.raises(ValueError, match="must be a boolean"):
        _rebuild_result(accepted, inference_admissible=1)
    with pytest.raises(TypeError, match="baseline_belief must expose artifact_id"):
        _rebuild_result(accepted, baseline_belief=object())


def test_infer_prob4d_candidate_rejects_invalid_inputs_before_delegation() -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    prediction = np.zeros((1, 3), dtype=np.float64)

    with pytest.raises(TypeError, match="observation_belief"):
        infer_prob4d_candidate(
            cast(Any, object()),
            linearization,
            physical_prediction_xyz_m=prediction,
        )
    with pytest.raises(TypeError, match="linearization"):
        infer_prob4d_candidate(
            observation,
            cast(Any, object()),
            physical_prediction_xyz_m=prediction,
        )
    with pytest.raises(TypeError, match="config"):
        infer_prob4d_candidate(
            observation,
            linearization,
            physical_prediction_xyz_m=prediction,
            config=cast(Any, 0),
        )
    with pytest.raises(TypeError, match="covariance_semantics"):
        infer_prob4d_candidate(
            observation,
            linearization,
            physical_prediction_xyz_m=prediction,
            covariance_semantics=cast(Any, 0),
        )


def test_infer_prob4d_candidate_delegates_all_inputs_to_strict_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    prediction = np.zeros((1, 3), dtype=np.float64)
    shared = np.zeros((1, 3, 0), dtype=np.float64)
    view = np.zeros((1, 3, 0), dtype=np.float64)
    state_prior = np.eye(1, dtype=np.float64)
    anchor_innovation = np.zeros((0, 3), dtype=np.float64)
    anchor_covariance = np.zeros((0, 3, 3), dtype=np.float64)
    anchor_state = np.zeros((0, 3, 1), dtype=np.float64)
    config = object.__new__(PriorAwareGaugeConfigV1)
    semantics = object.__new__(PosteriorCovarianceSemanticsV1)
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
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        state_prior_covariance_m2=state_prior,
        anchor_innovation_m=anchor_innovation,
        anchor_covariance_m2=anchor_covariance,
        anchor_state_jacobian=anchor_state,
        config=config,
        covariance_semantics=semantics,
        anchor_prior_reliability=np.zeros(0, dtype=np.float64),
    )

    assert result is candidate
    assert captured["args"] == (observation, linearization)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["physical_prediction_xyz_m"] is prediction
    assert kwargs["shared_bias_jacobian"] is shared
    assert kwargs["view_bias_jacobian"] is view
    assert kwargs["state_prior_covariance_m2"] is state_prior
    assert kwargs["anchor_innovation_m"] is anchor_innovation
    assert kwargs["anchor_covariance_m2"] is anchor_covariance
    assert kwargs["anchor_state_jacobian"] is anchor_state
    assert kwargs["config"] is config
    assert kwargs["covariance_semantics"] is semantics
    assert np.array_equal(kwargs["anchor_prior_reliability"], np.zeros(0))


def test_infer_prob4d_candidate_delegates_omitted_options_as_none(
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
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
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
                    "bayesian_phystwin_experiments",
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
