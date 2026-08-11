from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "build_deform360_covariance_target_v1.py"
)
_SPEC = importlib.util.spec_from_file_location("_covariance_target_v1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_REPOSITORY = Path(__file__).resolve().parents[1]
_PROTOCOL_V1_4 = (
    _REPOSITORY
    / "protocols"
    / "locks"
    / "deform360_covariance_only_target_v1_4.json"
)
_PROTOCOL_V1_5 = (
    _REPOSITORY
    / "protocols"
    / "locks"
    / "deform360_covariance_only_target_v1_5.json"
)

build_selection = _MODULE.build_selection
load_protocol = _MODULE.load_protocol
select_candidate_panel = _MODULE.select_candidate_panel


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metadata() -> dict[str, Any]:
    rows = [
        ("lift corner", "no"),
        ("lift edge", "yes"),
        ("drag", "no"),
        ("push", "yes"),
        ("fold", "no"),
        ("stretch", "yes"),
    ]
    return {
        "sequences": {
            str(index): {
                "action": action,
                "bimanual": bimanual,
                "nonprehensile": "no",
            }
            for index, (action, bimanual) in enumerate(rows)
        }
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    exclusion_path = repository / "protocols" / "locks" / "exclusion.json"
    exclusion = {
        "artifact_kind": "deform360-covariance-only-target-exclusion-v1",
        "schema_version": 1,
        "hash_namespace": "deform360-fresh-object-exclusion-v1",
        "object_hashes": [],
        "object_hash_count": 0,
    }
    exclusion["exclusion_sha256"] = _canonical(exclusion)
    _write(exclusion_path, exclusion)
    protocol_path = repository / "protocols" / "locks" / "protocol.json"
    protocol = {
        "schema": "bayesian-phystwin/deform360-covariance-only-target-protocol-v1",
        "schema_version": 1,
        "protocol_id": "test-covariance-target",
        "status": "locked-before-target-metadata-access",
        "dataset": {
            "repository": "brownu/deform360",
            "revision": "a" * 40,
            "raw_prefix": "raw",
            "processed_prefix": "processed",
        },
        "implementation_revision": "b" * 40,
        "exclusion": {
            "artifact_path": "protocols/locks/exclusion.json",
            "file_sha256": hashlib.sha256(exclusion_path.read_bytes()).hexdigest(),
            "canonical_sha256": exclusion["exclusion_sha256"],
            "hash_namespace": "deform360-fresh-object-exclusion-v1",
            "object_hash_count": 0,
        },
        "information_boundary": {
            "camera_media_decoded": False,
            "robot_or_tactile_arrays_opened": False,
            "geometry_or_track_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "selection": {
            "seed": "test-covariance-target",
            "roster_size": 24,
            "candidate_objects_per_stratum": 16,
            "metadata_invalid_candidate_policy": (
                "terminate before target payload; do not replace"
            ),
            "action_families": {
                "elevation": ["lift", "wave"],
                "planar_or_contact": ["drag", "push"],
                "shape_change": ["fold", "stretch"],
            },
            "exact_factorial_cells": {
                "object_stratum": ["sheet", "volumetric"],
                "bimanual": ["no", "yes"],
                "sessions_per_cell": 2,
            },
        },
    }
    protocol["protocol_sha256"] = _canonical(protocol)
    _write(protocol_path, protocol)

    sheet = [f"{index:03d}-sheet-{index}-cloth" for index in range(16)]
    volumetric = [f"{index + 100:03d}-volume-{index}" for index in range(16)]
    available = sheet + volumetric
    metadata = {object_id: _metadata() for object_id in available}
    metadata_sha = {
        object_id: _canonical(metadata[object_id]) for object_id in available
    }
    snapshot = {
        "resolved_revision": "a" * 40,
        "raw_objects": available,
        "metadata_by_object": metadata,
        "metadata_sha256_by_object": metadata_sha,
        "opened_paths": [f"raw/{object_id}/metadata.json" for object_id in available],
    }
    return repository, protocol_path, snapshot


def test_factorial_selection_is_deterministic_and_exact(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    first, touched = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )
    shuffled = dict(snapshot)
    shuffled["raw_objects"] = list(reversed(snapshot["raw_objects"]))
    shuffled["metadata_by_object"] = dict(
        reversed(list(snapshot["metadata_by_object"].items()))
    )
    second, _ = build_selection(
        shuffled,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )

    assert first["selection_sha256"] == second["selection_sha256"]
    assert len(first["candidate_panel"]) == 32
    assert len(first["target_roster"]) == 24
    assert len({row["object_id"] for row in first["target_roster"]}) == 24
    cells: dict[tuple[str, str, str], int] = {}
    for row in first["target_roster"]:
        cell = (
            row["stratum"],
            row["bimanual"],
            row["action_family"],
        )
        cells[cell] = cells.get(cell, 0) + 1
    assert len(cells) == 12
    assert set(cells.values()) == {2}
    assert touched["object_hash_count"] == 32
    assert touched["information_boundary"]["target_outcomes_opened"] is False


def test_exclusion_hash_removes_object_before_metadata_panel(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    protocol, _ = load_protocol(protocol_path, repository=repository)
    object_id = snapshot["raw_objects"][0]
    object_hash = _MODULE._object_hash(object_id)

    panel = select_candidate_panel(
        snapshot["raw_objects"] + ["999-extra-sheet-cloth"],
        excluded_hashes={object_hash},
        seed=protocol["selection"]["seed"],
        count_per_stratum=16,
    )

    assert object_id not in {row["object_id"] for row in panel}
    assert len(panel) == 32


def test_malformed_metadata_stops_without_replacement(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    object_id = snapshot["raw_objects"][0]
    snapshot["metadata_by_object"][object_id]["sequences"]["0"]["bimanual"] = (
        "yess"
    )

    with pytest.raises(ValueError, match="bimanual is malformed"):
        build_selection(
            snapshot,
            repository=repository,
            protocol_path=protocol_path,
            implementation_revision="c" * 40,
        )


def test_infeasible_factorial_panel_stops_before_payload(tmp_path: Path) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    for object_id, metadata in snapshot["metadata_by_object"].items():
        if object_id.endswith("-cloth"):
            for sequence in metadata["sequences"].values():
                sequence["bimanual"] = "no"

    with pytest.raises(ValueError, match="factorial cell"):
        build_selection(
            snapshot,
            repository=repository,
            protocol_path=protocol_path,
            implementation_revision="c" * 40,
        )


def test_schema_amendment_records_nonselective_field_without_filtering(
    tmp_path: Path,
) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = "schema-amended-before-target-roster-and-payload-access"
    protocol["selection"]["nonprehensile_selection_policy"] = (
        "record-only-never-used-for-selection"
    )
    protocol["selection"]["metadata_invalid_candidate_policy"] = (
        "malformed action or bimanual terminates; nonprehensile is record-only"
    )
    protocol["amendment"] = {
        "candidate_panel_reused_without_replacement": True,
        "target_roster_created_before_amendment": False,
        "target_payload_opened_before_amendment": False,
        "only_selection_change": (
            "nonprehensile is record-only and cannot affect eligibility or assignment"
        ),
    }
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = _canonical(protocol)
    _write(protocol_path, protocol)
    for metadata in snapshot["metadata_by_object"].values():
        for sequence in metadata["sequences"].values():
            sequence["nonprehensile"] = None

    result, _ = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )

    assert len(result["target_roster"]) == 24
    assert all(row["nonprehensile"] is None for row in result["target_roster"])
    assert all(
        row["nonprehensile_metadata_valid"] is False
        for row in result["target_roster"]
    )


def test_vocabulary_amendment_maps_all_audited_action_tokens(
    tmp_path: Path,
) -> None:
    repository, protocol_path, snapshot = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = (
        "metadata-vocabulary-amended-before-target-roster-and-payload-access"
    )
    protocol["selection"]["nonprehensile_selection_policy"] = (
        "record-only-never-used-for-selection"
    )
    protocol["selection"]["metadata_invalid_candidate_policy"] = (
        "malformed action or bimanual terminates; nonprehensile is record-only"
    )
    protocol["selection"]["action_families"] = {
        "elevation": ["lift", "wave"],
        "planar_or_contact": [
            "drag",
            "flip",
            "move",
            "press",
            "pull",
            "push",
            "roll",
            "rotate",
            "turn",
        ],
        "shape_change": [
            "bend",
            "close",
            "curl",
            "curve",
            "distort",
            "fold",
            "open",
            "squeeze",
            "stretch",
        ],
    }
    protocol["amendment"] = {
        "candidate_panel_reused_without_replacement": True,
        "target_roster_created_before_amendment": False,
        "target_payload_opened_before_amendment": False,
        "target_outcomes_opened_before_amendment": False,
        "only_selection_change": (
            "add pull to planar_or_contact and open/close to shape_change; "
            "no other selection or method change"
        ),
    }
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = _canonical(protocol)
    _write(protocol_path, protocol)
    rows = [
        ("lift corner", "no"),
        ("wave", "yes"),
        ("pull short side", "no"),
        ("drag", "yes"),
        ("open", "no"),
        ("close", "yes"),
    ]
    for metadata in snapshot["metadata_by_object"].values():
        metadata["sequences"] = {
            str(index): {
                "action": action,
                "bimanual": bimanual,
                "nonprehensile": None,
            }
            for index, (action, bimanual) in enumerate(rows)
        }

    result, _ = build_selection(
        snapshot,
        repository=repository,
        protocol_path=protocol_path,
        implementation_revision="c" * 40,
    )

    assert len(result["target_roster"]) == 24
    assert {row["action_family"] for row in result["target_roster"]} == {
        "elevation",
        "planar_or_contact",
        "shape_change",
    }
    assert {row["action"].split()[0] for row in result["target_roster"]} >= {
        "pull",
        "open",
        "close",
    }


def test_protocol_tampering_is_rejected(tmp_path: Path) -> None:
    repository, protocol_path, _ = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["selection"]["roster_size"] = 23
    _write(protocol_path, protocol)

    with pytest.raises(ValueError, match="protocol digest changed"):
        load_protocol(protocol_path, repository=repository)


def test_v1_4_freezes_corrected_source_provider_and_custom_evaluation() -> None:
    protocol = json.loads(_PROTOCOL_V1_4.read_text())
    supplied = protocol["protocol_sha256"]
    canonical = dict(protocol)
    canonical.pop("protocol_sha256")

    assert supplied == _canonical(canonical)
    assert protocol["amendment"]["parent_protocol_id"] == (
        "deform360-covariance-only-target-v1.3"
    )
    parent_path = (
        _REPOSITORY
        / "protocols"
        / "locks"
        / "deform360_covariance_only_target_v1_3.json"
    )
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == protocol[
        "amendment"
    ]["parent_protocol_file_sha256"]
    assert protocol["claim_boundary"]["official_deform360_benchmark_parity_claimed"] is False
    assert protocol["evaluation"]["official_endpoints"] == {
        "official_Chamfer_identity_check": (
            "unavailable-no-official-processed-annotation-in-locked-plan"
        ),
        "official_track_error_identity_check": (
            "unavailable-no-official-processed-annotation-in-locked-plan"
        ),
    }
    assert "joint_energy_score" in protocol["evaluation"][
        "removed_unavailable_or_undefined_endpoints"
    ]
    assert protocol["method"]["support_gate"] == {
        "case_minimum_empirical_identity_fraction": 0.5,
        "case_minimum_observed_prefix_frames": 2,
        "identity_minimum_valid_updates": 2,
        "prior_only_covariance_is_empirical_evidence": False,
        "zero_update_identity_policy": (
            "explicit-prior-only-label-and-exact-fallback-covariance"
        ),
    }
    assert protocol["observation_partition"][
        "provider_and_scoring_camera_sets_disjoint"
    ] is True
    assert protocol["observation_partition"][
        "provider_and_scoring_reconstruction_artifacts_distinct"
    ] is True
    causal = protocol["method"]["causal_residual_history"]
    assert causal["maximum_association_distance_m"] == 0.040
    assert causal["minimum_effective_candidate_support"] == 0.05
    assert causal["innovation_clipping_before_robust_likelihood"] is False
    assert causal["robust_innovation_processing_count"] == 1
    assert "independent of state innovation" in causal["prior_reliability"]
    assert causal["metric_row_covariance_used_by_endpoint_filter"] is True
    assert causal["prior_reliability_used_by_endpoint_filter"] is True
    assert causal["prior_reliability_application_count"] == 1
    assert causal["full_3x3_covariance_preserved_through_horizon"] is True
    assert "R_eff=" in causal["endpoint_likelihood"]
    assert protocol["method"]["mean"][
        "registered_reference_digest_supplied_by_caller"
    ] is True
    record = protocol["implementation"]["provider"]
    assert record["implementation_revision"] == (
        "15d1f53720a6f9f0baa3fcb2baca05c280ab63c4"
    )
    assert record["implementation_file_sha256"] == (
        "3e929bb235c6e2583c19b938151fc3ad0d483f3235a666e1f6aaeea3ef88cae4"
    )


def test_v1_5_freezes_conditional_covariance_semantics() -> None:
    protocol = json.loads(_PROTOCOL_V1_5.read_text())
    supplied = protocol["protocol_sha256"]
    canonical = dict(protocol)
    canonical.pop("protocol_sha256")

    assert supplied == _canonical(canonical)
    amendment = protocol["amendment"]
    assert amendment["parent_protocol_id"] == (
        "deform360-covariance-only-target-v1.4"
    )
    assert amendment["parent_protocol_independent_source_gate_passed"] is False
    assert amendment["parent_protocol_target_decode_authorized"] is False
    assert hashlib.sha256(_PROTOCOL_V1_4.read_bytes()).hexdigest() == amendment[
        "parent_protocol_file_sha256"
    ]
    assert amendment["parent_protocol_sha256"] == json.loads(
        _PROTOCOL_V1_4.read_text()
    )["protocol_sha256"]

    causal = protocol["method"]["causal_residual_history"]
    assert causal["global_loewner_monotonicity_claimed"] is False
    assert causal["posterior_covariance_psd_required"] is True
    assert "need not be Loewner-ordered" in causal[
        "posterior_covariance_semantics"
    ]
    assert causal["metric_row_covariance_used_by_endpoint_filter"] is True
    assert causal["prior_reliability_used_by_endpoint_filter"] is True
    assert causal["robust_innovation_processing_count"] == 1

    record = protocol["implementation"]["provider"]
    assert record["implementation_revision"] == (
        "f5d59e2e73425d5da02d5d5b26576ab7a4bb22f7"
    )
    assert "cannot make" not in record["predecode_blocker_regressions"][
        "downstream_uncertainty"
    ]
    for record_key in (
        "association_implementation_file",
        "implementation_file",
        "source_dry_run_file",
        "source_dry_run_script",
        "test_file",
    ):
        path = _REPOSITORY / record[record_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record[
            f"{record_key}_sha256"
        ]

    dry_run = json.loads((_REPOSITORY / record["source_dry_run_file"]).read_text())
    assert dry_run["dry_run_sha256"] == record["source_dry_run_sha256"]
    assert dry_run["gate_passed"] is True
    assert dry_run["heteroscedastic_endpoint"][
        "global_loewner_monotonicity_claimed"
    ] is False
    assert dry_run["target_roster_read"] is False
    assert dry_run["target_payload_read"] is False
    assert dry_run["target_outcome_read"] is False
