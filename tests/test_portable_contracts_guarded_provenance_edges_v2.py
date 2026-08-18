from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

import bayesian_phystwin.provider_runtime_identity_v1 as runtime_identity_module
from bayesian_phystwin.complete_belief_selection import CompleteBeliefGuardDecisionV1
from bayesian_phystwin.guarded_belief_selection_v2 import (
    CandidateBeliefConstructionReceiptV1,
    GuardedBeliefSelectionReceiptV2,
    bind_guarded_belief_selection_receipt,
    build_candidate_belief_construction_receipt,
)
from bayesian_phystwin.inference.v1 import finalize_guarded_update
from bayesian_phystwin.provider_runtime_identity_v1 import Prob4DRuntimeIdentityV1
from bayesian_phystwin.tree_block_sparse_prob4d import (
    ClaimBearingTreeBlockProb4DUpdateV1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


@dataclass(frozen=True)
class _Inference:
    candidate_id: str
    update_id: str
    admission_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    inference_admissible: bool


def _inference() -> _Inference:
    update_id = _digest("update")
    return _Inference(
        candidate_id=update_id,
        update_id=update_id,
        admission_id=_digest("admission"),
        observation_artifact_id=_digest("observation"),
        linearization_artifact_id=_digest("linearization"),
        inference_admissible=True,
    )


def _guarded() -> tuple[_Inference, _Belief, _Belief, object]:
    inference = _inference()
    baseline = _Belief(_digest("baseline"))
    candidate = _Belief(_digest("candidate"))
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("domain"),
        certificate_id=_digest("certificate"),
        inference_admissible=True,
        regret_guard_accepted=True,
        reason="accepted",
    )
    guarded = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        decision,
    )
    return inference, baseline, candidate, guarded


def _construction() -> tuple[
    _Inference, _Belief, _Belief, object, CandidateBeliefConstructionReceiptV1
]:
    inference, baseline, candidate, guarded = _guarded()
    construction = build_candidate_belief_construction_receipt(
        inference,
        baseline,
        candidate,
        common_domain_id=guarded.guard_decision.common_domain_id,
    )
    return inference, baseline, candidate, guarded, construction


def test_candidate_construction_rejects_malformed_inputs_and_records() -> None:
    inference, baseline, candidate, guarded, construction = _construction()

    with pytest.raises(ValueError, match="nonempty string"):
        replace(construction, construction_method="")
    with pytest.raises(ValueError, match="identity must equal"):
        replace(construction, update_id=_digest("different-update"))
    with pytest.raises(TypeError, match="construction lineage"):
        build_candidate_belief_construction_receipt(
            object(),  # type: ignore[arg-type]
            baseline,
            candidate,
            common_domain_id=guarded.guard_decision.common_domain_id,
        )
    with pytest.raises(ValueError, match="differs from update"):
        build_candidate_belief_construction_receipt(
            replace(inference, candidate_id=_digest("different-candidate")),
            baseline,
            candidate,
            common_domain_id=guarded.guard_decision.common_domain_id,
        )
    with pytest.raises(TypeError, match="artifact_id"):
        build_candidate_belief_construction_receipt(
            inference,
            object(),  # type: ignore[arg-type]
            candidate,
            common_domain_id=guarded.guard_decision.common_domain_id,
        )

    record = construction.to_record()
    with pytest.raises(ValueError, match="string-keyed mapping"):
        CandidateBeliefConstructionReceiptV1.from_record([])  # type: ignore[arg-type]
    non_string_key = dict(record)
    non_string_key[1] = "unexpected"  # type: ignore[index]
    with pytest.raises(ValueError, match="string-keyed mapping"):
        CandidateBeliefConstructionReceiptV1.from_record(non_string_key)  # type: ignore[arg-type]
    missing = dict(record)
    missing.pop("metadata")
    with pytest.raises(ValueError, match="fields do not match schema"):
        CandidateBeliefConstructionReceiptV1.from_record(missing)
    unexpected = dict(record)
    unexpected["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match schema"):
        CandidateBeliefConstructionReceiptV1.from_record(unexpected)
    wrong_schema = dict(record)
    wrong_schema["schema"] = "wrong"
    with pytest.raises(ValueError, match="unsupported candidate construction schema"):
        CandidateBeliefConstructionReceiptV1.from_record(wrong_schema)
    wrong_version = dict(record)
    wrong_version["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        CandidateBeliefConstructionReceiptV1.from_record(wrong_version)
    wrong_identity = dict(record)
    wrong_identity["receipt_id"] = _digest("wrong-receipt")
    with pytest.raises(ValueError, match="identity changed"):
        CandidateBeliefConstructionReceiptV1.from_record(wrong_identity)


def test_guarded_selection_receipt_rejects_invalid_state_and_records() -> None:
    inference, _baseline, candidate, guarded, construction = _construction()
    receipt = bind_guarded_belief_selection_receipt(inference, guarded, construction)
    ids = {
        "guard_certificate_id": guarded.guard_decision.certificate_id,
        "guard_decision_id": guarded.guard_decision.decision_id,
        "selection_id": guarded.selection.selection_id,
    }

    with pytest.raises(TypeError, match="CandidateBeliefConstructionReceiptV1"):
        GuardedBeliefSelectionReceiptV2(
            candidate_construction=object(),  # type: ignore[arg-type]
            guard_kind="guard",
            selected_belief_id=candidate.artifact_id,
            selected_candidate=True,
            exact_fallback=False,
            **ids,
        )
    with pytest.raises(ValueError, match="nonempty string"):
        replace(receipt, guard_kind="")
    with pytest.raises(ValueError, match="contradicts construction"):
        replace(receipt, selected_belief_id=_digest("wrong-selected-belief"))
    with pytest.raises(ValueError, match="complement"):
        replace(receipt, exact_fallback=True)
    inadmissible = replace(construction, inference_admissible=False)
    with pytest.raises(ValueError, match="requires admissible inference"):
        replace(receipt, candidate_construction=inadmissible)

    record = receipt.to_record()
    missing = dict(record)
    missing.pop("metadata")
    with pytest.raises(ValueError, match="fields do not match schema"):
        GuardedBeliefSelectionReceiptV2.from_record(missing)
    wrong_schema = dict(record)
    wrong_schema["schema"] = "wrong"
    with pytest.raises(ValueError, match="unsupported guarded selection schema"):
        GuardedBeliefSelectionReceiptV2.from_record(wrong_schema)
    wrong_version = dict(record)
    wrong_version["schema_version"] = 3
    with pytest.raises(ValueError, match="schema version"):
        GuardedBeliefSelectionReceiptV2.from_record(wrong_version)
    non_mapping_construction = dict(record)
    non_mapping_construction["candidate_construction"] = "wrong"
    with pytest.raises(ValueError, match="must be a mapping"):
        GuardedBeliefSelectionReceiptV2.from_record(non_mapping_construction)
    wrong_identity = dict(record)
    wrong_identity["receipt_id"] = _digest("wrong-selection-receipt")
    with pytest.raises(ValueError, match="identity changed"):
        GuardedBeliefSelectionReceiptV2.from_record(wrong_identity)


def test_guarded_selection_binding_rejects_lineage_substitution() -> None:
    inference, _baseline, _candidate, guarded, construction = _construction()

    with pytest.raises(TypeError, match="construction lineage"):
        bind_guarded_belief_selection_receipt(
            object(),  # type: ignore[arg-type]
            guarded,
            construction,
        )
    with pytest.raises(TypeError, match="GuardedUpdateResultV1"):
        bind_guarded_belief_selection_receipt(
            inference,
            object(),  # type: ignore[arg-type]
            construction,
        )
    with pytest.raises(TypeError, match="CandidateBeliefConstructionReceiptV1"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            object(),  # type: ignore[arg-type]
        )

    other_update = _digest("other-update")
    with pytest.raises(ValueError, match="different inference candidate"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            replace(
                construction,
                inference_candidate_id=other_update,
                update_id=other_update,
            ),
        )
    with pytest.raises(ValueError, match="different update"):
        bind_guarded_belief_selection_receipt(
            replace(inference, update_id=_digest("other-update")),
            guarded,
            construction,
        )
    with pytest.raises(ValueError, match="different admission"):
        bind_guarded_belief_selection_receipt(
            replace(inference, admission_id=_digest("other-admission")),
            guarded,
            construction,
        )
    with pytest.raises(ValueError, match="different observation evidence"):
        bind_guarded_belief_selection_receipt(
            replace(
                inference,
                observation_artifact_id=_digest("other-observation"),
            ),
            guarded,
            construction,
        )
    with pytest.raises(ValueError, match="different linearization"):
        bind_guarded_belief_selection_receipt(
            replace(
                inference,
                linearization_artifact_id=_digest("other-linearization"),
            ),
            guarded,
            construction,
        )
    with pytest.raises(ValueError, match="changed inference admissibility"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            replace(construction, inference_admissible=False),
        )
    with pytest.raises(ValueError, match="guarded baseline"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            replace(construction, baseline_belief_id=_digest("other-baseline")),
        )
    with pytest.raises(ValueError, match="guarded candidate"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            replace(construction, candidate_belief_id=_digest("other-candidate")),
        )
    with pytest.raises(ValueError, match="different common domains"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            replace(construction, common_domain_id=_digest("other-domain")),
        )


def _runtime_identity_kwargs() -> dict[str, object]:
    return {
        "project_id": "prob4d",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "provider_manifest_id": "a" * 64,
        "expected_revision": "1" * 40,
        "observed_revision": "1" * 40,
        "revision_evidence_source": "installed_vcs_metadata",
        "clean_checkout": None,
        "independently_verified": True,
    }


def test_runtime_identity_constructor_rejects_untrusted_evidence() -> None:
    valid = _runtime_identity_kwargs()
    cases = [
        ({"project_id": "other"}, ValueError, "stable Prob4D project"),
        ({"source_repository": "other/repo"}, ValueError, "registered Prob4D"),
        ({"provider_manifest_id": "not-a-digest"}, ValueError, "SHA-256"),
        ({"expected_revision": "not-a-commit"}, ValueError, "Git commit"),
        (
            {"revision_evidence_source": "caller_declared"},
            ValueError,
            "independent VCS evidence",
        ),
        (
            {"revision_evidence_source": "source_checkout", "clean_checkout": None},
            TypeError,
            "declare cleanliness",
        ),
        (
            {"revision_evidence_source": "source_checkout", "clean_checkout": False},
            ValueError,
            "must be clean",
        ),
        ({"clean_checkout": True}, ValueError, "cannot declare checkout cleanliness"),
        ({"independently_verified": 1}, TypeError, "must be a bool"),
    ]
    for changes, error_type, match in cases:
        kwargs = {**valid, **changes}
        with pytest.raises(error_type, match=match):
            Prob4DRuntimeIdentityV1(**kwargs)  # type: ignore[arg-type]


def test_runtime_identity_record_rejects_schema_and_identity_tampering() -> None:
    identity = Prob4DRuntimeIdentityV1(**_runtime_identity_kwargs())  # type: ignore[arg-type]
    record = identity.to_record()

    with pytest.raises(ValueError, match="string-keyed mapping"):
        Prob4DRuntimeIdentityV1.from_record([])  # type: ignore[arg-type]
    non_string_key = dict(record)
    non_string_key[1] = "unexpected"  # type: ignore[index]
    with pytest.raises(ValueError, match="string-keyed mapping"):
        Prob4DRuntimeIdentityV1.from_record(non_string_key)  # type: ignore[arg-type]
    missing = dict(record)
    missing.pop("metadata")
    with pytest.raises(ValueError, match="fields changed"):
        Prob4DRuntimeIdentityV1.from_record(missing)
    unexpected = dict(record)
    unexpected["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        Prob4DRuntimeIdentityV1.from_record(unexpected)
    wrong_schema = dict(record)
    wrong_schema["schema"] = "wrong"
    with pytest.raises(ValueError, match="unsupported runtime identity schema"):
        Prob4DRuntimeIdentityV1.from_record(wrong_schema)
    wrong_version = dict(record)
    wrong_version["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        Prob4DRuntimeIdentityV1.from_record(wrong_version)
    wrong_identity = dict(record)
    wrong_identity["identity_id"] = _digest("wrong-runtime-identity")
    with pytest.raises(ValueError, match="content address changed"):
        Prob4DRuntimeIdentityV1.from_record(wrong_identity)


def test_runtime_identity_attestation_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="attestation must be a mapping"):
        Prob4DRuntimeIdentityV1.from_provider_attestation([])  # type: ignore[arg-type]

    minimal_attestation = {"provider_revision": "1" * 40}
    monkeypatch.setattr(
        runtime_identity_module,
        "validate_prob4d_provider_attestation",
        lambda *_args, **_kwargs: {"runtime_revision": object()},
    )
    with pytest.raises(AssertionError, match="lost mapping type"):
        Prob4DRuntimeIdentityV1.from_provider_attestation(minimal_attestation)

    monkeypatch.setattr(
        runtime_identity_module,
        "validate_prob4d_provider_attestation",
        lambda *_args, **_kwargs: {
            "runtime_revision": {"observed_revision": None},
            "provider_manifest_id": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="omits observed revision"):
        Prob4DRuntimeIdentityV1.from_provider_attestation(minimal_attestation)


def test_tree_block_candidate_id_is_the_bound_update_id() -> None:
    update = object.__new__(ClaimBearingTreeBlockProb4DUpdateV1)
    object.__setattr__(update, "_update_id", "a" * 64)

    assert update.candidate_id == update.update_id == "a" * 64
