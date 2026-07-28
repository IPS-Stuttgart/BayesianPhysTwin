from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin.deform360_official_parity import (
    ARTIFACT_KIND,
    REQUIRED_3D_FIELDS,
    aggregate_metric_sensitivity,
    audit_parity_contract,
    build_public_parity_audit,
    build_public_parity_contract,
    candidate_chamfer_metrics,
    candidate_track_metrics,
    require_official_parity,
    seal_parity_contract,
)


def _complete_author_contract() -> dict[str, object]:
    source_id = "author_contract"
    fields = {
        name: {
            "status": "authoritative",
            "value": f"exact-{name}",
            "source_id": source_id,
            "locator": f"contract.fields.{name}",
            "note": "Exact convention supplied by the benchmark authors.",
        }
        for name in REQUIRED_3D_FIELDS
    }
    fields["benchmark_setting"]["value"] = "multi_episode"
    return seal_parity_contract(
        {
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "benchmark_setting": "multi_episode",
            "sources": {
                source_id: {
                    "kind": "author_confirmed_contract",
                    "authority": "deform360_authoritative",
                    "url": "https://example.test/deform360-evaluator-contract.json",
                    "revision": "contract-v1",
                    "content_sha256": "a" * 64,
                    "bound_files": {},
                }
            },
            "fields": fields,
        }
    )


def test_public_contract_fails_closed_for_every_setting() -> None:
    report = build_public_parity_audit()

    assert report["conclusion"].startswith("No public contract")
    assert (
        report["public_sources"]["deform360_arxiv_v1"]["content_sha256"]
        == "66d2bfecd6ec9b829cd810913238821adc143c4831393704ed4bcc4ccc09e05c"
    )
    assert (
        report["public_sources"]["pgrd_candidate_metric"]["bound_files"][
            "experiments/train/eval.py"
        ]
        == "80a95f1b477bc3852f08d5bd33cc13f33d5152f2798f60c572716d441587c606"
    )
    assert (
        report["public_sources"]["deform360_public_repo"]["revision"]
        == "d8522a4403b766aeb387510c04e89032a56fdf35"
    )
    assert (
        report["public_sources"]["deform360_public_repo"]["bound_files"][
            "deform360/processing/control_points_stage.py"
        ]
        == "9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373"
    )
    contract = report["contracts"]["per_episode"]
    for field_name in (
        "future_frame_manifest",
        "validity_visibility_mask",
        "coordinate_frame",
        "length_unit",
    ):
        assert contract["fields"][field_name]["status"] == "candidate"
        assert contract["fields"][field_name]["source_id"] == "deform360_public_repo"
    for audit in report["audits"].values():
        assert audit["parity_ready"] is False
        assert audit["official_claim_allowed"] is False
        assert audit["allowed_claim_label"] == "candidate_convention_sensitivity_only"
        assert audit["field_counts"] == {
            "authoritative": 1,
            "candidate": 15,
            "missing": 4,
        }
        assert audit["candidate_fields"]
        assert audit["missing_fields"]


def test_public_contract_cannot_be_required_as_official() -> None:
    contract = build_public_parity_contract("per_episode")

    with pytest.raises(ValueError, match="official Deform360 parity is unresolved"):
        require_official_parity(contract)


def test_author_confirmed_complete_contract_allows_official_label() -> None:
    contract = _complete_author_contract()

    audit = audit_parity_contract(contract)

    assert audit["parity_ready"] is True
    assert audit["official_claim_allowed"] is True
    assert audit["candidate_fields"] == []
    assert audit["missing_fields"] == []
    require_official_parity(contract)


def test_contract_tampering_is_rejected() -> None:
    contract = _complete_author_contract()
    contract["fields"]["track_reduction"]["value"] = "changed"

    with pytest.raises(ValueError, match="contract checksum changed"):
        audit_parity_contract(contract)


def test_candidate_source_cannot_be_relabelled_authoritative() -> None:
    contract = build_public_parity_contract("per_episode")
    changed = deepcopy(contract)
    changed["fields"]["track_reduction"]["status"] = "authoritative"
    changed = seal_parity_contract(changed)

    with pytest.raises(
        ValueError,
        match="track_reduction is not backed by an authoritative evaluator contract",
    ):
        audit_parity_contract(changed)


def test_external_released_evaluator_label_cannot_claim_authority() -> None:
    contract = _complete_author_contract()
    contract["sources"]["author_contract"]["kind"] = "released_evaluator"
    contract["sources"]["author_contract"]["authority"] = "external_method_only"
    contract = seal_parity_contract(contract)

    with pytest.raises(
        ValueError,
        match="is not backed by an authoritative evaluator contract",
    ):
        audit_parity_contract(contract)


def test_candidate_metric_variants_expose_direction_and_reduction_ambiguity() -> None:
    prediction = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    chamfer = candidate_chamfer_metrics(prediction, target)
    track = candidate_track_metrics(prediction, target)

    assert np.isclose(chamfer["pred_to_target_mean_euclidean_m"], 0.5)
    assert np.isclose(chamfer["target_to_pred_mean_euclidean_m"], 4.5)
    assert np.isclose(chamfer["symmetric_mean_euclidean_m"], 2.5)
    assert np.isclose(track["coordinate_mse_m2"], 13.5)
    assert np.isclose(track["mean_point_euclidean_m"], 4.5)
    assert not np.isclose(track["coordinate_rmse_m"], track["mean_point_euclidean_m"])


def test_track_metric_requires_explicit_validity_for_nonfinite_rows() -> None:
    prediction = np.asarray([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    target = np.zeros((2, 3))

    with pytest.raises(ValueError, match="non-finite"):
        candidate_track_metrics(prediction, target)

    result = candidate_track_metrics(
        prediction, target, valid_mask=np.asarray([True, False])
    )
    assert result["coordinate_mse_m2"] == 0.0


def test_aggregation_conventions_are_observably_different() -> None:
    result = aggregate_metric_sensitivity(
        {
            "a-1": np.zeros(10),
            "a-2": np.zeros(1),
            "b-1": np.asarray([12.0]),
        },
        {"a-1": "a", "a-2": "a", "b-1": "b"},
    )

    assert np.isclose(result["frame_pooled_mean"], 1.0)
    assert np.isclose(result["episode_balanced_mean"], 4.0)
    assert np.isclose(result["object_balanced_mean"], 6.0)
