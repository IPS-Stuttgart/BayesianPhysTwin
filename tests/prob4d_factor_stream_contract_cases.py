from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin import GaugeAwareBeliefResult
from bayesian_phystwin.causal4d_belief_provider_v2 import (
    CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES,
    causal4d_belief_provider_v2_manifest,
)
from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.physical_linearization import PhysicalLinearizationV1
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)
from bayesian_phystwin.prob4d_factor_stream import (
    ClaimBearingProb4DStreamRunV1,
    ClaimBearingProb4DStreamStepV1,
    Prob4DObservationFactorStreamUpdateV1,
    Prob4DObservationFactorStreamV1,
    RecursiveNuisancePolicyV1,
    apply_claim_bearing_prob4d_stream_update,
    bind_prob4d_stream_observation,
    load_claim_bearing_prob4d_stream_run,
    load_prob4d_observation_factor_stream,
    prob4d_observation_identity_summary,
    start_claim_bearing_prob4d_stream_run,
    write_claim_bearing_prob4d_stream_run,
    write_prob4d_observation_factor_stream,
)
from bayesian_phystwin.prospective_prob4d_update import (
    ClaimBearingProb4DUpdateV1,
)

REV = "1" * 40
REPO = "FlorianPfaff/Prob4D"
PROVIDER = "2" * 64
CAL = {"gauge": "3" * 64, "point": "4" * 64}
DOMAIN = "d" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(
    *,
    start: int = 0,
    stop: int = 2,
    case_id: str = "case-a",
    stream_id: str = "stream-a",
    repository: str = REPO,
    revision: str = REV,
    windows: tuple[str, ...] = ("window-0",),
) -> ObservationBeliefV1:
    count = stop - start
    groups = np.arange(count, dtype=np.int64)
    frame_ids = np.arange(start, stop, dtype=np.int64)
    mean = np.zeros((count, 3), dtype=np.float64)
    mean[:, 0] = np.arange(count, dtype=np.float64) * 0.01
    mean[:, 2] = 1.0
    return ObservationBeliefV1(
        case_id=case_id,
        stream_id=stream_id,
        causal_frame_stop=stop,
        view_names=("cam-0",),
        window_names=windows,
        factor_names=(),
        source_repository=repository,
        source_revision=revision,
        source_artifact_sha256="9" * 64,
        declared_frame_ids=frame_ids,
        mean_xyz_m=mean,
        frame_ids=frame_ids,
        entity_ids=np.arange(10, 10 + count, dtype=np.int64),
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=groups,
        factor_group_ids=groups,
        prior_reliability=np.full(count, 0.9),
        association_probability=np.ones(count),
        local_covariance_m2=np.repeat(np.eye(3)[None], count, axis=0) * 1e-4,
        low_rank_factor_m=np.zeros((count, 3, 0)),
        group_ids=groups,
        group_prior_nominal_probability=np.full(count, 0.9),
        group_composite_weight=np.ones(count),
        metadata={"causal_source": "factor stream prefix"},
    )


def _stream_update(
    observation: ObservationBeliefV1,
    *,
    index: int,
    start: int,
    manifest_path: str,
    manifest_sha: str,
    payload_sha: str,
    previous: str | None = None,
) -> Prob4DObservationFactorStreamUpdateV1:
    persistent, count, identity_sha = prob4d_observation_identity_summary(observation)
    return Prob4DObservationFactorStreamUpdateV1(
        update_index=index,
        admitted_frame_start=start,
        causal_frame_stop=observation.causal_frame_stop,
        bundle_manifest_path=manifest_path,
        bundle_manifest_sha256=manifest_sha,
        bundle_payload_sha256=payload_sha,
        bundle_sequence_id="sequence-a",
        case_id=observation.case_id,
        stream_id=observation.stream_id,
        source_repository=observation.source_repository,
        source_revision=observation.source_revision,
        factor_count=1,
        observation_count=count,
        persistent_identity_count=persistent,
        observation_identity_sha256=identity_sha,
        gauge_ids=observation.window_names,
        previous_update_id=previous,
    )


def _write_bundle(
    root: Path,
    observation: ObservationBeliefV1,
    *,
    subdir: str,
) -> tuple[str, str, str]:
    directory = root / subdir
    directory.mkdir(parents=True)
    payload = directory / "factors.npz"
    payload.write_bytes(b"portable-payload-" + subdir.encode())
    payload_sha = _sha(payload)
    manifest = directory / "factors.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "prob4d.observation-factor-bundle",
                "schema_version": 4,
                "sequence_id": "sequence-a",
                "case_id": observation.case_id,
                "stream_id": observation.stream_id,
                "source_repository": observation.source_repository,
                "source_revision": observation.source_revision,
                "causal_frame_stop": observation.causal_frame_stop,
                "payload": {
                    "path": "factors.npz",
                    "sha256": payload_sha,
                    "allow_pickle": False,
                },
                "gauge_covariance": {
                    "semantics": "joint-cross-window",
                    "ordered_gauge_ids": list(observation.window_names),
                    "cross_window_covariance_preserved": True,
                },
                "factors": [{"factor_id": f"factor-{subdir}"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest.relative_to(root).as_posix(), _sha(manifest), payload_sha


def _stream_tree(tmp_path: Path, *, two_updates: bool = False):
    first = _observation()
    path0, manifest0, payload0 = _write_bundle(tmp_path, first, subdir="update-0")
    update0 = _stream_update(
        first,
        index=0,
        start=0,
        manifest_path=path0,
        manifest_sha=manifest0,
        payload_sha=payload0,
    )
    updates = [update0]
    observations = [first]
    if two_updates:
        second = _observation(
            start=2,
            stop=4,
        )
        path1, manifest1, payload1 = _write_bundle(
            tmp_path,
            second,
            subdir="update-1",
        )
        update1 = _stream_update(
            second,
            index=1,
            start=2,
            manifest_path=path1,
            manifest_sha=manifest1,
            payload_sha=payload1,
            previous=update0.update_id,
        )
        updates.append(update1)
        observations.append(second)
    stream = Prob4DObservationFactorStreamV1(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="stream-a",
        source_repository=REPO,
        source_revision=REV,
        updates=updates,
        metadata={"protocol": "recursive-test"},
    )
    stream_path = tmp_path / "stream.json"
    write_prob4d_observation_factor_stream(stream, stream_path)
    return stream, observations, stream_path


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def _claim_update(
    observation_id: str,
    linearization_id: str,
    *,
    admissible: bool,
    provider: str = PROVIDER,
) -> ClaimBearingProb4DUpdateV1:
    lineage = {
        "observation_artifact_id": observation_id,
        "linearization_artifact_id": linearization_id,
        "prob4d_claim_bearing_provider_manifest_id": provider,
        "prob4d_claim_bearing_calibration_artifact_ids": CAL,
        "prob4d_claim_bearing_runtime_revision_source": "independent-vcs-check",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }
    result = GaugeAwareBeliefResult(
        inference_admissible=admissible,
        reason="accepted" if admissible else "rejected",
        state_coefficients=np.array([0.1]),
        gauge_delta=np.zeros(0),
        shared_bias_coefficients=np.zeros(0),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        posterior_covariance=np.array([[0.2]]),
        identifiable_state_transform=np.array([[1.0]]),
        identifiable_fractions=np.array([1.0]),
        query_sensitivity_fractions=np.array([1.0]),
        robust_weights=np.ones(2),
        anchor_robust_weights=np.zeros(0),
        diagnostics={"solver": "stream-contract-test"},
        input_lineage=lineage,
    )
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=observation_id,
        linearization_artifact_id=linearization_id,
        provider_manifest_id=provider,
        calibration_artifact_ids=CAL,
        runtime_revision_source="independent-vcs-check",
        runtime_revision_independently_verified=True,
    )


def _policy(
    *,
    mode: str = "persistent_explicit_state",
) -> RecursiveNuisancePolicyV1:
    return RecursiveNuisancePolicyV1(
        mode=mode,  # type: ignore[arg-type]
        state_domain_id=DOMAIN,
        nuisance_family_ids=("prob4d-gauge", "shared-camera-bias"),
        conditional_independence_evidence_id=(
            "6" * 64 if mode == "conditionally_independent_increments" else None
        ),
        metadata={"protocol": "recursive-test"},
    )


def _linearization(
    observation: ObservationBeliefV1,
    baseline_belief_id: str,
    nuisance_policy: RecursiveNuisancePolicyV1,
) -> PhysicalLinearizationV1:
    state_jacobian = np.zeros((observation.observation_count, 3, 1))
    state_jacobian[:, 0, 0] = 1.0
    query_jacobian = np.zeros((1, 3, 1))
    query_jacobian[0, 0, 0] = 1.0
    return PhysicalLinearizationV1(
        observation_artifact_id=observation.artifact_id,
        baseline_belief_id=baseline_belief_id,
        action_prefix_id="7" * 64,
        simulator_revision="simulator-test-revision",
        frame_ids=observation.frame_ids,
        entity_ids=observation.entity_ids,
        view_indices=observation.view_indices,
        window_indices=observation.window_indices,
        state_jacobian=state_jacobian,
        query_state_jacobian=query_jacobian,
        physical_response_m=np.array([[0.01, 0.0, 0.0]]),
        metadata={
            "test": "recursive factor stream",
            "recursive_nuisance_policy_id": nuisance_policy.policy_id,
        },
    )


def test_recursive_nuisance_policy_is_content_addressed_and_fail_closed() -> None:
    policy = _policy()
    restored = RecursiveNuisancePolicyV1.from_mapping(policy.to_record())

    assert restored == policy
    assert policy.policy_id is not None
    with pytest.raises(ValueError, match="allowed only"):
        RecursiveNuisancePolicyV1(
            mode="persistent_explicit_state",
            state_domain_id=DOMAIN,
            nuisance_family_ids=("gauge",),
            conditional_independence_evidence_id="6" * 64,
        )
    conditional = _policy(mode="conditionally_independent_increments")
    assert conditional.conditional_independence_evidence_id == "6" * 64
    with pytest.raises(ValueError, match="conditional_independence_evidence_id"):
        RecursiveNuisancePolicyV1(
            mode="conditionally_independent_increments",
            state_domain_id=DOMAIN,
            nuisance_family_ids=("gauge",),
        )
    with pytest.raises(ValueError, match="mode must be"):
        RecursiveNuisancePolicyV1(
            mode="other",  # type: ignore[arg-type]
            state_domain_id=DOMAIN,
            nuisance_family_ids=("gauge",),
        )
    with pytest.raises(ValueError, match="policy_id"):
        replace(policy, policy_id="f" * 64)
    record = policy.to_record()
    for value, message in (
        (None, "mapping"),
        ({}, "fields"),
        ({**record, "schema": "other"}, "schema"),
        ({**record, "schema_version": 2}, "version"),
    ):
        with pytest.raises(ValueError, match=message):
            RecursiveNuisancePolicyV1.from_mapping(value)


def test_causal4d_provider_v2_advertises_recursive_contracts() -> None:
    manifest = causal4d_belief_provider_v2_manifest(
        provider_revision="a" * 40
    )
    required_capabilities = {
        "claim_bearing_prob4d_recursive_stream",
        "append_only_complete_belief_routing",
        "exact_recursive_complete_belief_fallback",
        "explicit_posterior_covariance_semantics",
        "explicit_recursive_nuisance_policy",
        "stream_member_and_identity_revalidation",
    }
    assert required_capabilities <= set(CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES)
    required_schemas = {
        "Prob4DObservationFactorStream",
        "Prob4DStreamObservationBinding",
        "ClaimBearingProb4DStreamStep",
        "ClaimBearingProb4DStreamRun",
        "PosteriorCovarianceSemantics",
        "RecursiveNuisancePolicy",
    }
    assert required_schemas <= set(
        CAUSAL4D_BELIEF_PROVIDER_V2_ARTIFACT_SCHEMA_VERSIONS
    )
    assert manifest["capabilities"] == list(
        CAUSAL4D_BELIEF_PROVIDER_V2_CAPABILITIES
    )
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["recursive_stream_claim"] == (
        "causal ordering, member bytes, row identities, policy locks, and "
        "exact fallback are validated; provider competence and calibrated "
        "physical benefit remain prospective gates"
    )


def test_covariance_semantics_round_trip_and_policy_identity() -> None:
    semantics = working_irls_covariance_semantics(
        np.eye(2),
        metadata={"source": "test"},
    )
    restored = PosteriorCovarianceSemanticsV1.from_mapping(semantics.to_record())
    wider = working_irls_covariance_semantics(np.eye(3))

    assert restored == semantics
    assert semantics.artifact_id != wider.artifact_id
    assert semantics.policy_id == wider.policy_id
    assert not semantics.calibrated
    with pytest.raises(TypeError, match="immutable"):
        semantics.metadata["source"] = "changed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "method": "irls_working",
                "dimension": 1,
                "likelihood_power_semantics": "power",
                "mixture_curvature_exact": True,
            },
            "contradict",
        ),
        (
            {
                "method": "laplace_observed_information",
                "dimension": 1,
                "likelihood_power_semantics": "power",
            },
            "contradict",
        ),
        (
            {
                "method": "group_sandwich",
                "dimension": 1,
                "likelihood_power_semantics": "power",
            },
            "contradict",
        ),
        (
            {
                "method": "irls_working",
                "dimension": 1,
                "likelihood_power_semantics": "power",
                "calibrated": True,
            },
            "calibration_artifact_id",
        ),
        (
            {
                "method": "irls_working",
                "dimension": 1,
                "likelihood_power_semantics": "power",
                "calibration_artifact_id": "7" * 64,
            },
            "only when calibrated",
        ),
    ],
)
def test_covariance_semantics_rejects_contradictory_claims(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PosteriorCovarianceSemanticsV1(**kwargs)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.zeros((1, 2, 3)), "square"),
        (np.array([[np.nan]]), "finite"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (-np.eye(1), "positive semidefinite"),
    ],
)
def test_working_covariance_semantics_validates_matrix(covariance, message) -> None:
    with pytest.raises(ValueError, match=message):
        working_irls_covariance_semantics(covariance)


def test_stream_load_verifies_members_and_round_trips(tmp_path: Path) -> None:
    stream, _, stream_path = _stream_tree(tmp_path, two_updates=True)
    loaded = load_prob4d_observation_factor_stream(stream_path)

    assert loaded == stream
    assert loaded.causal_frame_stop == 4
    assert loaded.updates[1].previous_update_id == loaded.updates[0].update_id

    record = json.loads(stream_path.read_text())
    record["artifact_id"] = "f" * 64
    stream_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="artifact_id"):
        load_prob4d_observation_factor_stream(
            stream_path,
            verify_member_files=False,
        )


def test_stream_member_tampering_and_symlinks_fail(tmp_path: Path) -> None:
    _, _, stream_path = _stream_tree(tmp_path)
    payload = tmp_path / "update-0" / "factors.npz"
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload checksum"):
        load_prob4d_observation_factor_stream(stream_path)

    payload.unlink()
    target = tmp_path / "outside.npz"
    target.write_bytes(b"portable-payload-update-0")
    payload.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_prob4d_observation_factor_stream(stream_path)


def test_stream_contract_rejects_broken_chain_interval_and_repository(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path, two_updates=True)
    first, second = stream.updates
    with pytest.raises(ValueError, match="hash chain"):
        Prob4DObservationFactorStreamV1(
            sequence_id=stream.sequence_id,
            case_id=stream.case_id,
            stream_id=stream.stream_id,
            source_repository=stream.source_repository,
            source_revision=stream.source_revision,
            updates=(
                first,
                replace(
                    second,
                    previous_update_id="8" * 64,
                    update_id=None,
                ),
            ),
        )
    with pytest.raises(ValueError, match="contiguous"):
        shifted = replace(
            second,
            admitted_frame_start=3,
            update_id=None,
        )
        Prob4DObservationFactorStreamV1(
            sequence_id=stream.sequence_id,
            case_id=stream.case_id,
            stream_id=stream.stream_id,
            source_repository=stream.source_repository,
            source_revision=stream.source_revision,
            updates=(first, shifted),
        )
    with pytest.raises(ValueError, match="Prob4D"):
        replace(first, source_repository="other/repo", update_id=None)
    assert observations[0].causal_frame_stop == first.causal_frame_stop


def test_observation_binding_recomputes_identity_and_rejects_drift(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    observation = observations[0]
    binding = bind_prob4d_stream_observation(stream, 0, observation)

    assert binding.observation_artifact_id == observation.artifact_id
    assert len(binding.binding_id) == 64
    with pytest.raises(ValueError, match="case_id"):
        bind_prob4d_stream_observation(
            stream,
            0,
            replace(observation, case_id="other"),
        )
    with pytest.raises(ValueError, match="window order"):
        bind_prob4d_stream_observation(
            stream,
            0,
            replace(observation, window_names=("other",)),
        )
    with pytest.raises(ValueError, match="identity digest"):
        changed = replace(
            observation,
            entity_ids=np.array([10, 99], dtype=np.int64),
        )
        bind_prob4d_stream_observation(stream, 0, changed)
    with pytest.raises(ValueError, match="outside"):
        bind_prob4d_stream_observation(stream, 4, observation)


def _apply(
    stream,
    observation,
    *,
    accepted: bool,
    run=None,
    baseline=None,
    candidate=None,
    provider=PROVIDER,
):
    baseline = baseline or _Belief("a" * 64)
    candidate = candidate or _Belief("b" * 64)
    policy = _policy()
    run = run or start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )
    linearization = _linearization(
        observation,
        baseline.artifact_id,
        policy,
    )
    update = _claim_update(
        observation.artifact_id,
        linearization.artifact_id,
        admissible=True,
        provider=provider,
    )
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=DOMAIN,
        certificate_id="e" * 64,
        inference_admissible=True,
        regret_guard_accepted=accepted,
        reason="accepted" if accepted else "regret",
    )
    return apply_claim_bearing_prob4d_stream_update(
        stream,
        run,
        baseline=baseline,
        candidate=candidate,
        observation=observation,
        linearization=linearization,
        claim_update=update,
        decision=decision,
        nuisance_policy=policy,
    )


def test_recursive_rejection_reuses_exact_baseline_and_persists(tmp_path: Path) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    baseline = _Belief("a" * 64)
    selected, run, step = _apply(
        stream,
        observations[0],
        accepted=False,
        baseline=baseline,
    )

    assert selected is baseline
    assert step.exact_fallback
    assert not step.selected_candidate
    assert run.final_belief_id == baseline.artifact_id
    assert run.provider_manifest_id == PROVIDER
    assert run.covariance_policy_id == step.covariance_policy_id

    path = tmp_path / "run.json"
    write_claim_bearing_prob4d_stream_run(run, path)
    assert load_claim_bearing_prob4d_stream_run(path) == run
    with pytest.raises(FileExistsError):
        write_claim_bearing_prob4d_stream_run(run, path)


def test_recursive_acceptance_chains_selected_candidate(tmp_path: Path) -> None:
    stream, observations, _ = _stream_tree(tmp_path, two_updates=True)
    initial = _Belief("a" * 64)
    selected0, run0, step0 = _apply(
        stream,
        observations[0],
        accepted=True,
        baseline=initial,
        candidate=_Belief("b" * 64),
    )
    selected1, run1, step1 = _apply(
        stream,
        observations[1],
        accepted=False,
        run=run0,
        baseline=selected0,
        candidate=_Belief("f" * 64),
    )

    assert selected0.artifact_id == "b" * 64
    assert selected1 is selected0
    assert step0.selected_candidate
    assert not step0.exact_fallback
    assert step1.previous_step_id == step0.step_id
    assert run1.final_belief_id == selected0.artifact_id
    assert ClaimBearingProb4DStreamRunV1.from_mapping(run1.to_record()) == run1


def test_recursive_run_rejects_lock_and_lineage_drift(tmp_path: Path) -> None:
    stream, observations, _ = _stream_tree(tmp_path, two_updates=True)
    initial = _Belief("a" * 64)
    selected, run, _ = _apply(
        stream,
        observations[0],
        accepted=True,
        baseline=initial,
        candidate=_Belief("b" * 64),
    )
    with pytest.raises(ValueError, match="provider manifest changed"):
        _apply(
            stream,
            observations[1],
            accepted=True,
            run=run,
            baseline=selected,
            candidate=_Belief("f" * 64),
            provider="9" * 64,
        )
    with pytest.raises(ValueError, match="current belief"):
        _apply(
            stream,
            observations[1],
            accepted=True,
            run=run,
            baseline=_Belief("0" * 64),
            candidate=_Belief("f" * 64),
        )


def test_empty_run_cannot_predeclare_update_locks(tmp_path: Path) -> None:
    stream, _, _ = _stream_tree(tmp_path)
    with pytest.raises(ValueError, match="cannot predeclare"):
        ClaimBearingProb4DStreamRunV1(
            stream_artifact_id=stream.artifact_id,
            initial_belief_id="a" * 64,
            recursive_nuisance_policy_id=_policy().policy_id,
            provider_manifest_id=PROVIDER,
        )


def test_covariance_semantics_mapping_and_identifier_boundaries() -> None:
    valid = working_irls_covariance_semantics(np.eye(1))
    for value, message in (
        (None, "mapping"),
        ({}, "fields"),
        ({**valid.to_record(), "schema": "other"}, "schema"),
        ({**valid.to_record(), "schema_version": 2}, "version"),
    ):
        with pytest.raises(ValueError, match=message):
            PosteriorCovarianceSemanticsV1.from_mapping(value)
    with pytest.raises(ValueError, match="method"):
        PosteriorCovarianceSemanticsV1(
            method="unknown",  # type: ignore[arg-type]
            dimension=1,
            likelihood_power_semantics="power",
        )
    with pytest.raises(ValueError, match="likelihood_power_semantics"):
        PosteriorCovarianceSemanticsV1(
            method="irls_working",
            dimension=1,
            likelihood_power_semantics="",
        )
    with pytest.raises(ValueError, match="artifact_id"):
        replace(valid, artifact_id="f" * 64)
    calibrated = PosteriorCovarianceSemanticsV1(
        method="group_sandwich",
        dimension=1,
        likelihood_power_semantics="group-power",
        mixture_curvature_exact=False,
        group_score_correction=True,
        calibrated=True,
        calibration_artifact_id="7" * 64,
    )
    assert calibrated.calibration_artifact_id == "7" * 64


def test_update_mapping_and_constructor_boundaries(tmp_path: Path) -> None:
    stream, _, _ = _stream_tree(tmp_path)
    update = stream.updates[0]
    with pytest.raises(ValueError, match="exceed"):
        replace(
            update,
            admitted_frame_start=1,
            causal_frame_stop=1,
            update_id=None,
        )
    with pytest.raises(ValueError, match="first"):
        replace(update, previous_update_id="f" * 64, update_id=None)
    with pytest.raises(ValueError, match="later"):
        replace(
            update,
            update_index=1,
            previous_update_id=None,
            update_id=None,
        )
    with pytest.raises(ValueError, match="update_id"):
        replace(update, update_id="f" * 64)
    for value, message in ((None, "mapping"), ({}, "fields")):
        with pytest.raises(ValueError, match=message):
            Prob4DObservationFactorStreamUpdateV1.from_mapping(value)
    with pytest.raises(ValueError, match="sequence"):
        replace(update, gauge_ids="window-0", update_id=None)
    with pytest.raises(ValueError, match="must not be empty"):
        replace(update, gauge_ids=(), update_id=None)
    with pytest.raises(ValueError, match="unique"):
        replace(update, gauge_ids=("w", "w"), update_id=None)


def test_stream_mapping_and_cross_update_boundaries(tmp_path: Path) -> None:
    stream, _, _ = _stream_tree(tmp_path)
    update = stream.updates[0]
    with pytest.raises(ValueError, match="updates must contain"):
        replace(stream, updates=(), artifact_id=None)
    reindexed = replace(
        update,
        update_index=1,
        previous_update_id=update.update_id,
        update_id=None,
    )
    with pytest.raises(ValueError, match="indices"):
        replace(stream, updates=(reindexed,), artifact_id=None)
    changed_sequence = replace(
        update,
        bundle_sequence_id="other",
        update_id=None,
    )
    with pytest.raises(ValueError, match="sequence_id"):
        replace(stream, updates=(changed_sequence,), artifact_id=None)
    changed_case = replace(update, case_id="other", update_id=None)
    with pytest.raises(ValueError, match="case_id"):
        replace(stream, updates=(changed_case,), artifact_id=None)
    record = stream.to_record()
    for value, message in (
        (None, "mapping"),
        ({}, "fields"),
        ({**record, "schema": "other"}, "schema"),
        ({**record, "schema_version": 2}, "version"),
        ({**record, "updates": {}}, "nonempty array"),
    ):
        with pytest.raises(ValueError, match=message):
            Prob4DObservationFactorStreamV1.from_mapping(value)
    assert stream.admitted_frame_start == 0


def _rewrite_stream_member(
    stream_path: Path,
    mutate,
) -> None:
    stream = load_prob4d_observation_factor_stream(
        stream_path,
        verify_member_files=False,
    )
    update = stream.updates[0]
    bundle_path = stream_path.parent / update.bundle_manifest_path
    bundle = json.loads(bundle_path.read_text())
    mutate(bundle)
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n")
    changed = replace(
        update,
        bundle_manifest_sha256=_sha(bundle_path),
        update_id=None,
    )
    changed_stream = replace(stream, updates=(changed,), artifact_id=None)
    write_prob4d_observation_factor_stream(
        changed_stream,
        stream_path,
        overwrite=True,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema="other"), "not a Prob4D"),
        (lambda value: value.update(schema_version=3), "schema v4"),
        (lambda value: value.update(case_id="other"), "case_id differs"),
        (lambda value: value.update(factors=[]), "factor_count"),
        (lambda value: value.update(gauge_covariance=None), "no gauge covariance"),
        (
            lambda value: value["gauge_covariance"].update(
                semantics="marginal-blocks-only"
            ),
            "does not preserve",
        ),
        (
            lambda value: value["gauge_covariance"].update(
                ordered_gauge_ids=["other"]
            ),
            "gauge order",
        ),
        (lambda value: value.update(payload=None), "no payload"),
        (
            lambda value: value["payload"].update(allow_pickle=True),
            "disable pickle",
        ),
        (
            lambda value: value["payload"].update(sha256="f" * 64),
            "digest differs",
        ),
    ],
)
def test_stream_member_descriptor_boundaries(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    _, _, stream_path = _stream_tree(tmp_path)
    _rewrite_stream_member(stream_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_prob4d_observation_factor_stream(stream_path)


def test_stream_manifest_and_missing_member_fail(tmp_path: Path) -> None:
    _, _, stream_path = _stream_tree(tmp_path)
    bundle = tmp_path / "update-0" / "factors.json"
    bundle.write_text(bundle.read_text() + " ")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_prob4d_observation_factor_stream(stream_path)

    _, _, other_path = _stream_tree(tmp_path / "other")
    (other_path.parent / "update-0" / "factors.json").unlink()
    with pytest.raises(ValueError, match="ordinary file"):
        load_prob4d_observation_factor_stream(other_path)


def test_stream_write_and_load_type_boundaries(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="stream"):
        write_prob4d_observation_factor_stream(object(), tmp_path / "x.json")
    _, _, stream_path = _stream_tree(tmp_path)
    with pytest.raises(ValueError, match="boolean"):
        load_prob4d_observation_factor_stream(
            stream_path,
            verify_member_files=1,  # type: ignore[arg-type]
        )


def test_observation_identity_and_binding_count_boundaries(tmp_path: Path) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    observation = observations[0]
    with pytest.raises(TypeError, match="ObservationBeliefV1"):
        prob4d_observation_identity_summary(object())
    duplicate = replace(observation)
    object.__setattr__(duplicate, "frame_ids", np.array([0, 0], dtype=np.int64))
    object.__setattr__(duplicate, "entity_ids", np.array([10, 10], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        prob4d_observation_identity_summary(duplicate)
    with pytest.raises(TypeError, match="stream"):
        bind_prob4d_stream_observation(object(), 0, observation)
    with pytest.raises(TypeError, match="observation"):
        bind_prob4d_stream_observation(stream, 0, object())
    shifted_update = replace(
        stream.updates[0],
        admitted_frame_start=1,
        update_id=None,
    )
    shifted_stream = replace(stream, updates=(shifted_update,), artifact_id=None)
    with pytest.raises(ValueError, match="admitted stream interval"):
        bind_prob4d_stream_observation(shifted_stream, 0, observation)
    changed_persistent = replace(
        stream.updates[0],
        persistent_identity_count=3,
        update_id=None,
    )
    with pytest.raises(ValueError, match="persistent-identity"):
        bind_prob4d_stream_observation(
            replace(stream, updates=(changed_persistent,), artifact_id=None),
            0,
            observation,
        )
    changed_count = replace(
        stream.updates[0],
        observation_count=3,
        update_id=None,
    )
    with pytest.raises(ValueError, match="row count"):
        bind_prob4d_stream_observation(
            replace(stream, updates=(changed_count,), artifact_id=None),
            0,
            observation,
        )
    binding = bind_prob4d_stream_observation(stream, 0, observation)
    assert binding.to_record()["binding_id"] == binding.binding_id
    with pytest.raises(ValueError, match="binding_id"):
        replace(binding, binding_id="f" * 64)


def _valid_step(tmp_path: Path):
    stream, observations, _ = _stream_tree(tmp_path)
    _, run, step = _apply(stream, observations[0], accepted=False)
    return stream, observations[0], run, step


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "admitted_frame_start": 1,
                "causal_frame_stop": 1,
                "step_id": None,
            },
            "exceed",
        ),
        ({"selected_belief_id": "f" * 64, "step_id": None}, "contradicts"),
        ({"exact_fallback": False, "step_id": None}, "exact_fallback"),
        (
            {
                "runtime_revision_independently_verified": False,
                "step_id": None,
            },
            "must be true",
        ),
        ({"previous_step_id": "f" * 64, "step_id": None}, "first"),
        (
            {
                "update_index": 1,
                "previous_step_id": None,
                "step_id": None,
            },
            "later",
        ),
        ({"step_id": "f" * 64}, "step_id"),
    ],
)
def test_recursive_step_contract_boundaries(
    tmp_path: Path,
    changes,
    message: str,
) -> None:
    _, _, _, step = _valid_step(tmp_path)
    with pytest.raises(ValueError, match=message):
        replace(step, **changes)


def test_recursive_step_mapping_boundaries(tmp_path: Path) -> None:
    _, _, _, step = _valid_step(tmp_path)
    record = step.to_record()
    for value, message in (
        (None, "mapping"),
        ({}, "fields"),
        ({**record, "schema": "other"}, "schema"),
        ({**record, "schema_version": 2}, "version"),
    ):
        with pytest.raises(ValueError, match=message):
            ClaimBearingProb4DStreamStepV1.from_mapping(value)


def test_recursive_run_contract_and_mapping_boundaries(tmp_path: Path) -> None:
    _, _, run, step = _valid_step(tmp_path)
    with pytest.raises(ValueError, match="steps must contain"):
        replace(run, steps=(object(),), run_id=None)
    with pytest.raises(ValueError, match="calibration"):
        replace(run, calibration_artifact_ids={}, run_id=None)
    with pytest.raises(ValueError, match="verified runtime"):
        replace(
            run,
            runtime_revision_independently_verified=False,
            run_id=None,
        )
    with pytest.raises(ValueError, match="indices"):
        shifted = replace(
            step,
            update_index=1,
            previous_step_id=step.step_id,
            step_id=None,
        )
        replace(run, steps=(shifted,), run_id=None)
    with pytest.raises(ValueError, match="different Prob4D stream"):
        changed = replace(step, stream_artifact_id="f" * 64, step_id=None)
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="belief chain"):
        changed = replace(
            step,
            prior_belief_id="f" * 64,
            selected_belief_id="f" * 64,
            step_id=None,
        )
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="provider manifest"):
        changed = replace(step, provider_manifest_id="f" * 64, step_id=None)
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="calibration artifacts"):
        changed = replace(
            step,
            calibration_artifact_ids={"gauge": "f" * 64},
            step_id=None,
        )
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="runtime revision source"):
        changed = replace(
            step,
            runtime_revision_source="other",
            step_id=None,
        )
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="covariance interpretation"):
        changed = replace(step, covariance_policy_id="f" * 64, step_id=None)
        replace(run, steps=(changed,), run_id=None)
    with pytest.raises(ValueError, match="run_id"):
        replace(run, run_id="f" * 64)
    record = run.to_record()
    for value, message in (
        (None, "mapping"),
        ({}, "fields"),
        ({**record, "schema": "other"}, "schema"),
        ({**record, "schema_version": 2}, "version"),
        ({**record, "steps": {}}, "array"),
    ):
        with pytest.raises(ValueError, match=message):
            ClaimBearingProb4DStreamRunV1.from_mapping(value)


def test_start_and_run_write_type_boundaries(tmp_path: Path) -> None:
    stream, _, _ = _stream_tree(tmp_path)
    empty = start_claim_bearing_prob4d_stream_run(
        stream,
        _Belief("a" * 64),
        nuisance_policy=_policy(),
    )
    assert empty.final_belief_id == empty.initial_belief_id
    with pytest.raises(TypeError, match="stream"):
        start_claim_bearing_prob4d_stream_run(
            object(),  # type: ignore[arg-type]
            _Belief("a" * 64),
            nuisance_policy=_policy(),
        )
    with pytest.raises(TypeError, match="nuisance_policy"):
        start_claim_bearing_prob4d_stream_run(
            stream,
            _Belief("a" * 64),
            nuisance_policy=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="run"):
        write_claim_bearing_prob4d_stream_run(object(), tmp_path / "run.json")
    path = tmp_path / "run.json"
    write_claim_bearing_prob4d_stream_run(empty, path)
    write_claim_bearing_prob4d_stream_run(empty, path, overwrite=True)
    path.unlink()
    target = tmp_path / "target.json"
    target.write_text("{}")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        write_claim_bearing_prob4d_stream_run(empty, path, overwrite=True)


def _apply_components(tmp_path: Path):
    stream, observations, _ = _stream_tree(tmp_path)
    observation = observations[0]
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )
    linearization = _linearization(
        observation,
        baseline.artifact_id,
        policy,
    )
    update = _claim_update(
        observation.artifact_id,
        linearization.artifact_id,
        admissible=True,
    )
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=DOMAIN,
        certificate_id="e" * 64,
        inference_admissible=True,
        regret_guard_accepted=True,
        reason="accepted",
    )
    return (
        stream,
        observation,
        baseline,
        candidate,
        run,
        linearization,
        update,
        decision,
        policy,
    )


def test_apply_type_and_lineage_boundaries(tmp_path: Path) -> None:
    values = _apply_components(tmp_path)
    (
        stream,
        observation,
        baseline,
        candidate,
        run,
        linearization,
        update,
        decision,
        policy,
    ) = values
    kwargs = dict(
        baseline=baseline,
        candidate=candidate,
        observation=observation,
        linearization=linearization,
        claim_update=update,
        decision=decision,
        nuisance_policy=policy,
    )
    with pytest.raises(TypeError, match="stream"):
        apply_claim_bearing_prob4d_stream_update(object(), run, **kwargs)
    with pytest.raises(TypeError, match="run"):
        apply_claim_bearing_prob4d_stream_update(stream, object(), **kwargs)
    with pytest.raises(ValueError, match="different Prob4D stream"):
        other = replace(run, stream_artifact_id="f" * 64, run_id=None)
        apply_claim_bearing_prob4d_stream_update(stream, other, **kwargs)
    with pytest.raises(TypeError, match="linearization"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "linearization": object()},
        )
    with pytest.raises(TypeError, match="claim_update"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "claim_update": object()},
        )
    with pytest.raises(TypeError, match="decision"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "decision": object()},
        )
    with pytest.raises(TypeError, match="nuisance_policy"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "nuisance_policy": object()},
        )
    different_policy = RecursiveNuisancePolicyV1(
        mode="persistent_explicit_state",
        state_domain_id=DOMAIN,
        nuisance_family_ids=("different-family",),
    )
    with pytest.raises(ValueError, match="differs from the run lock"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "nuisance_policy": different_policy},
        )
    with pytest.raises(ValueError, match="common domain"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{
                **kwargs,
                "decision": replace(decision, common_domain_id="8" * 64),
            },
        )
    without_policy = replace(
        linearization,
        metadata={"test": "missing recursive policy"},
    )
    with pytest.raises(ValueError, match="does not bind the recursive"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **{**kwargs, "linearization": without_policy},
        )
    mismatch_cases = (
        (
            "linearization does not bind the stream observation",
            {"linearization": replace(linearization, observation_artifact_id="f" * 64)},
        ),
        (
            "linearization does not bind the current baseline",
            {"linearization": replace(linearization, baseline_belief_id="f" * 64)},
        ),
        (
            "claim update does not bind the stream observation",
            {
                "claim_update": _claim_update(
                    "f" * 64,
                    linearization.artifact_id,
                    admissible=True,
                )
            },
        ),
        (
            "claim update does not bind the physical linearization",
            {
                "claim_update": _claim_update(
                    observation.artifact_id,
                    "f" * 64,
                    admissible=True,
                )
            },
        ),
        (
            "guard decision does not bind the current baseline",
            {"decision": replace(decision, baseline_belief_id="f" * 64)},
        ),
        (
            "guard decision does not bind the candidate",
            {"decision": replace(decision, candidate_belief_id="f" * 64)},
        ),
        (
            "disagree on inference",
            {
                "decision": replace(
                    decision,
                    inference_admissible=False,
                    regret_guard_accepted=False,
                )
            },
        ),
    )
    for message, changes in mismatch_cases:
        with pytest.raises(ValueError, match=message):
            apply_claim_bearing_prob4d_stream_update(
                stream,
                run,
                **{**kwargs, **changes},
            )


def test_apply_covariance_and_consumption_boundaries(tmp_path: Path) -> None:
    values = _apply_components(tmp_path)
    (
        stream,
        observation,
        baseline,
        candidate,
        run,
        linearization,
        update,
        decision,
        policy,
    ) = values
    kwargs = dict(
        baseline=baseline,
        candidate=candidate,
        observation=observation,
        linearization=linearization,
        claim_update=update,
        decision=decision,
        nuisance_policy=policy,
    )
    with pytest.raises(TypeError, match="covariance_semantics"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **kwargs,
            covariance_semantics=object(),
        )
    wrong_dimension = working_irls_covariance_semantics(np.eye(2))
    with pytest.raises(ValueError, match="dimension"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            run,
            **kwargs,
            covariance_semantics=wrong_dimension,
        )
    _, consumed, _ = apply_claim_bearing_prob4d_stream_update(
        stream,
        run,
        **kwargs,
    )
    with pytest.raises(ValueError, match="no unconsumed"):
        apply_claim_bearing_prob4d_stream_update(
            stream,
            consumed,
            baseline=candidate,
            candidate=_Belief("f" * 64),
            observation=observation,
            linearization=linearization,
            claim_update=update,
            decision=decision,
            nuisance_policy=policy,
        )
