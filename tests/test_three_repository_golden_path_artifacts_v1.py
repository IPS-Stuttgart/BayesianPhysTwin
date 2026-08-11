from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import GaugeAwareSelection
from bayesian_phystwin.three_repository_golden_path_artifacts_v1 import (
    ArrayByteIdentityV1,
    GoldenPathEvidenceBundleV1,
    GoldenPathSelectionArtifactV1,
    build_golden_path_selection_artifact_v1,
    load_golden_path_evidence_bundle_v1,
    write_golden_path_evidence_bundle_v1,
)


def _components(value: str) -> dict[str, str]:
    return {
        "bayesian_phystwin": value,
        "prob4d": value,
        "causal4d": value,
    }


def _selection(*, accepted: bool) -> GaugeAwareSelection:
    baseline = np.asarray([[0.0, -0.0], [1.0, 2.0]], dtype=np.float32)
    candidate = np.asarray([[0.2, 0.1], [1.1, 2.2]], dtype=np.float32)
    return GaugeAwareSelection(
        candidate_accepted=accepted,
        inference_admissible=True,
        regret_guard_present=True,
        regret_guard_accepted=accepted,
        reason=(
            "candidate-accepted" if accepted else "regret-guard-exact-baseline-fallback"
        ),
        selected_value=candidate if accepted else baseline,
    )


def _artifact(*, accepted: bool) -> GoldenPathSelectionArtifactV1:
    return build_golden_path_selection_artifact_v1(
        selection=_selection(accepted=accepted),
        baseline=np.asarray(
            [[0.0, -0.0], [1.0, 2.0]],
            dtype=np.float32,
        ),
        candidate=np.asarray(
            [[0.2, 0.1], [1.1, 2.2]],
            dtype=np.float32,
        ),
        case_id="three-repository-golden-path",
        protocol_id="three-repository-installed-wheel-v1",
        observation_artifact_id="a" * 64,
        twin_belief_id="b" * 64,
        physical_posterior_id="c" * 64,
        provider_manifest_id="d" * 64,
        run_manifest_id="e" * 64,
        evidence_fingerprint="f" * 64,
        repository_revisions=_components("1" * 40),
        wheel_sha256=_components("2" * 64),
        package_versions=_components("0.4.0"),
        metadata={"selection_stage": "baseline-relative-regret-guard"},
    )


def _bundle() -> GoldenPathEvidenceBundleV1:
    return GoldenPathEvidenceBundleV1(
        accepted=_artifact(accepted=True),
        rejected=_artifact(accepted=False),
    )


def test_array_identity_binds_dtype_shape_and_exact_bytes() -> None:
    positive_zero = ArrayByteIdentityV1.from_array(np.asarray([0.0], dtype=np.float32))
    negative_zero = ArrayByteIdentityV1.from_array(np.asarray([-0.0], dtype=np.float32))
    wider = ArrayByteIdentityV1.from_array(np.asarray([0.0], dtype=np.float64))
    reshaped = ArrayByteIdentityV1.from_array(np.asarray([[0.0]], dtype=np.float32))

    assert positive_zero.array_id != negative_zero.array_id
    assert positive_zero.array_id != wider.array_id
    assert positive_zero.array_id != reshaped.array_id
    restored = ArrayByteIdentityV1.from_mapping(positive_zero.as_dict())
    assert restored == positive_zero


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray([1.0 + 2.0j]),
        np.asarray(["not-numeric"]),
        np.asarray([], dtype=np.float64),
    ),
)
def test_array_identity_rejects_unsupported_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ArrayByteIdentityV1.from_array(value)


def test_bundle_binds_accepted_candidate_and_rejected_exact_fallback() -> None:
    bundle = _bundle()

    assert bundle.accepted.candidate_accepted
    assert bundle.accepted.selected_identity == bundle.accepted.candidate_identity
    assert bundle.accepted.exact_fallback_identity is None
    assert not bundle.rejected.candidate_accepted
    assert bundle.rejected.selected_identity == bundle.rejected.baseline_identity
    assert bundle.rejected.exact_fallback_identity == (
        bundle.rejected.baseline_identity.array_id
    )
    assert bundle.accepted.artifact_id != bundle.rejected.artifact_id
    assert len(bundle.bundle_id) == 64


def test_bundle_round_trip_and_atomic_no_clobber(tmp_path: Path) -> None:
    bundle = _bundle()
    paths = write_golden_path_evidence_bundle_v1(tmp_path, bundle)

    assert set(paths) == {"accepted", "rejected", "bundle"}
    restored = load_golden_path_evidence_bundle_v1(tmp_path)
    assert restored == bundle

    with pytest.raises(FileExistsError):
        write_golden_path_evidence_bundle_v1(tmp_path, bundle)

    write_golden_path_evidence_bundle_v1(
        tmp_path,
        bundle,
        overwrite=True,
    )
    assert load_golden_path_evidence_bundle_v1(tmp_path) == bundle


def test_selection_records_recursively_freeze_metadata_and_mappings() -> None:
    revisions = _components("1" * 40)
    metadata: dict[str, object] = {"nested": {"values": [1, 2]}}
    artifact = build_golden_path_selection_artifact_v1(
        selection=_selection(accepted=True),
        baseline=np.asarray(
            [[0.0, -0.0], [1.0, 2.0]],
            dtype=np.float32,
        ),
        candidate=np.asarray(
            [[0.2, 0.1], [1.1, 2.2]],
            dtype=np.float32,
        ),
        case_id="case",
        protocol_id="protocol",
        observation_artifact_id="a" * 64,
        twin_belief_id="b" * 64,
        physical_posterior_id="c" * 64,
        provider_manifest_id="d" * 64,
        run_manifest_id="e" * 64,
        evidence_fingerprint="f" * 64,
        repository_revisions=revisions,
        wheel_sha256=_components("2" * 64),
        package_versions=_components("0.4.0"),
        metadata=metadata,
    )
    revisions["prob4d"] = "3" * 40
    cast(dict[str, Any], metadata["nested"])["values"] = []

    assert artifact.repository_revisions["prob4d"] == "1" * 40
    assert artifact.metadata == {"nested": {"values": [1, 2]}}
    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, Any], artifact.metadata)["new"] = True


def test_tampered_selection_artifact_fails_closed() -> None:
    artifact = _artifact(accepted=False)
    payload = copy.deepcopy(artifact.as_dict())
    selected = cast(dict[str, Any], payload["selected_identity"])
    selected["payload_sha256"] = "9" * 64

    with pytest.raises(ValueError):
        GoldenPathSelectionArtifactV1.from_mapping(payload)

    payload = copy.deepcopy(artifact.as_dict())
    payload["exact_fallback_identity"] = "9" * 64
    with pytest.raises(ValueError, match="fallback"):
        GoldenPathSelectionArtifactV1.from_mapping(payload)

    payload = copy.deepcopy(artifact.as_dict())
    payload["candidate_accepted"] = 0
    with pytest.raises(ValueError, match="literal boolean"):
        GoldenPathSelectionArtifactV1.from_mapping(payload)

    payload = copy.deepcopy(artifact.as_dict())
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="selection version"):
        GoldenPathSelectionArtifactV1.from_mapping(payload)

    payload = copy.deepcopy(artifact.as_dict())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields changed"):
        GoldenPathSelectionArtifactV1.from_mapping(payload)


def test_bundle_rejects_context_drift_and_pair_substitution() -> None:
    bundle = _bundle()
    drifted_rejected = replace(
        bundle.rejected,
        run_manifest_id="9" * 64,
    )
    with pytest.raises(ValueError, match="run_manifest_id"):
        GoldenPathEvidenceBundleV1(
            accepted=bundle.accepted,
            rejected=drifted_rejected,
        )

    payload = copy.deepcopy(bundle.as_dict())
    payload["rejected"] = copy.deepcopy(payload["accepted"])
    with pytest.raises(ValueError):
        GoldenPathEvidenceBundleV1.from_mapping(payload)

    payload = copy.deepcopy(bundle.as_dict())
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="bundle version"):
        GoldenPathEvidenceBundleV1.from_mapping(payload)


def test_loader_rejects_pair_that_does_not_match_bundle(tmp_path: Path) -> None:
    bundle = _bundle()
    write_golden_path_evidence_bundle_v1(tmp_path, bundle)
    accepted_path = tmp_path / "accepted-selection.json"
    accepted_payload = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_payload["artifact_id"] = "9" * 64
    accepted_path.write_text(
        json.dumps(accepted_payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_golden_path_evidence_bundle_v1(tmp_path)


def test_bundle_serialization_is_plain_finite_json() -> None:
    payload = _bundle().as_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    restored = GoldenPathEvidenceBundleV1.from_mapping(json.loads(encoded))

    assert restored.as_dict() == payload
    assert isinstance(
        cast(Mapping[str, Any], payload["accepted"])["metadata"],
        Mapping,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"dtype": "not-a-dtype"}, "dtype is invalid"),
        ({"dtype": "float32"}, "canonical finite real storage"),
        ({"shape": cast(Any, "1")}, "integer tuple"),
        ({"shape": (-1,)}, "nonnegative integer"),
        ({"nbytes": 8}, "does not match"),
    ),
)
def test_array_identity_constructor_rejects_noncanonical_descriptors(
    kwargs: Mapping[str, Any],
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "dtype": "<f4",
        "shape": (1,),
        "nbytes": 4,
        "payload_sha256": "0" * 64,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ArrayByteIdentityV1(**arguments)


def test_array_identity_mapping_rejects_non_array_shape() -> None:
    payload = ArrayByteIdentityV1.from_array(np.asarray([1.0])).as_dict()
    payload["shape"] = "not-an-array"

    with pytest.raises(ValueError, match="JSON array"):
        ArrayByteIdentityV1.from_mapping(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"decision": cast(Any, "unknown")}, "unsupported golden-path decision"),
        ({"case_id": " noncanonical"}, "canonical text"),
        (
            {"baseline_identity": cast(Any, np.asarray([0.0]))},
            "ArrayByteIdentityV1",
        ),
        (
            {"candidate_identity": _artifact(accepted=True).baseline_identity},
            "distinct",
        ),
        ({"inference_admissible": False}, "every admission gate"),
        ({"reason": "wrong-accepted-reason"}, "reason changed"),
        (
            {"selected_identity": _artifact(accepted=True).baseline_identity},
            "exact candidate bytes",
        ),
        ({"exact_fallback_identity": "0" * 64}, "cannot claim an exact fallback"),
    ),
)
def test_accepted_selection_rejects_inconsistent_contracts(
    changes: Mapping[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_artifact(accepted=True), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"candidate_accepted": True}, "cannot accept"),
        ({"reason": "regret-guard-rejected"}, "declare exact baseline fallback"),
        (
            {"selected_identity": _artifact(accepted=False).candidate_identity},
            "changed the baseline bytes",
        ),
    ),
)
def test_rejected_selection_rejects_inconsistent_contracts(
    changes: Mapping[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_artifact(accepted=False), **changes)


def test_selection_rejects_incomplete_component_roster() -> None:
    with pytest.raises(ValueError, match="complete component roster"):
        replace(
            _artifact(accepted=True),
            repository_revisions={"bayesian_phystwin": "1" * 40},
        )


def test_selection_mapping_rejects_non_object_and_non_string_keys() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        GoldenPathSelectionArtifactV1.from_mapping(cast(Any, []))

    with pytest.raises(ValueError, match="literal string keys"):
        GoldenPathSelectionArtifactV1.from_mapping(cast(Any, {1: "invalid"}))


def test_selection_mapping_rejects_wrong_schema() -> None:
    payload = copy.deepcopy(_artifact(accepted=True).as_dict())
    payload["schema_name"] = "wrong.schema"

    with pytest.raises(ValueError, match="selection schema"):
        GoldenPathSelectionArtifactV1.from_mapping(payload)


def test_bundle_rejects_invalid_entries_and_tampered_fallback() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="invalid types"):
        GoldenPathEvidenceBundleV1(
            accepted=cast(Any, object()),
            rejected=bundle.rejected,
        )

    tampered_rejected = _artifact(accepted=False)
    object.__setattr__(tampered_rejected, "exact_fallback_identity", "9" * 64)
    with pytest.raises(ValueError, match="exact fallback identity"):
        GoldenPathEvidenceBundleV1(
            accepted=bundle.accepted,
            rejected=tampered_rejected,
        )


def test_bundle_rejects_equal_artifact_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GoldenPathSelectionArtifactV1,
        "artifact_id",
        property(lambda self: "0" * 64),
    )

    with pytest.raises(ValueError, match="artifact IDs must differ"):
        _bundle()


def test_bundle_mapping_rejects_wrong_schema_and_id() -> None:
    payload = copy.deepcopy(_bundle().as_dict())
    payload["schema_name"] = "wrong.schema"
    with pytest.raises(ValueError, match="bundle schema"):
        GoldenPathEvidenceBundleV1.from_mapping(payload)

    payload = copy.deepcopy(_bundle().as_dict())
    payload["bundle_id"] = "9" * 64
    with pytest.raises(ValueError, match="bundle ID"):
        GoldenPathEvidenceBundleV1.from_mapping(payload)


def test_build_and_write_reject_wrong_contract_types(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GaugeAwareSelection"):
        build_golden_path_selection_artifact_v1(
            selection=cast(Any, object()),
            baseline=np.asarray([0.0]),
            candidate=np.asarray([1.0]),
            case_id="case",
            protocol_id="protocol",
            observation_artifact_id="a" * 64,
            twin_belief_id="b" * 64,
            physical_posterior_id="c" * 64,
            provider_manifest_id="d" * 64,
            run_manifest_id="e" * 64,
            evidence_fingerprint="f" * 64,
            repository_revisions=_components("1" * 40),
            wheel_sha256=_components("2" * 64),
            package_versions=_components("0.4.0"),
        )

    with pytest.raises(ValueError, match="GoldenPathEvidenceBundleV1"):
        write_golden_path_evidence_bundle_v1(
            tmp_path,
            cast(Any, object()),
        )


def test_loader_rejects_valid_pair_that_differs_from_bundle(tmp_path: Path) -> None:
    bundle = _bundle()
    write_golden_path_evidence_bundle_v1(tmp_path, bundle)
    altered = replace(
        bundle.accepted,
        metadata={"selection_stage": "different-valid-record"},
    )
    (tmp_path / "accepted-selection.json").write_text(
        json.dumps(altered.as_dict(), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pair does not match"):
        load_golden_path_evidence_bundle_v1(tmp_path)
