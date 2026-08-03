"""Repository-transfer compatibility and packaging regressions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin import (
    PROB4D_LEGACY_SOURCE_REPOSITORY,
    PROB4D_SOURCE_REPOSITORIES,
    PROB4D_SOURCE_REPOSITORY,
    is_prob4d_source_repository,
)
from bayesian_phystwin.gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.prob4d_causal_lineage import (
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.prob4d_provider_attestation import (
    validate_prob4d_provider_attestation,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "prob4d_joint_observation_v1.json"


def _joint_fixture_belief() -> ObservationBeliefV1:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = payload["descriptor"]
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    return ObservationBeliefV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )


def _provider_attestation(source_repository: str) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.3.0",
        "provider_revision": "d" * 40,
        "provider_api_version": 2,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": source_repository,
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    manifest_id = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {"manifest_id": manifest_id, **descriptor}
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest_id,
        "provider_manifest": manifest,
        "provider_revision": "d" * 40,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "5" * 64,
            "point_artifact_id": "6" * 64,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": "d" * 40,
            "observed_revision": "d" * 40,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def _unfused_belief(source_repository: str) -> ObservationBeliefV1:
    return ObservationBeliefV1(
        case_id="case",
        stream_id="prob4d:unfused",
        causal_frame_stop=3,
        view_names=("camera-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository=source_repository,
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([1, 2]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        frame_ids=np.asarray([1, 2]),
        entity_ids=np.asarray([0, 0]),
        view_indices=np.zeros(2, dtype=np.int64),
        window_indices=np.zeros(2, dtype=np.int64),
        correlation_group_ids=np.asarray([0, 0]),
        factor_group_ids=np.zeros(2, dtype=np.int64),
        prior_reliability=np.ones(2),
        association_probability=np.ones(2),
        local_covariance_m2=np.repeat(np.eye(3)[None] * 1e-4, 2, axis=0),
        low_rank_factor_m=np.zeros((2, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([0.5]),
        metadata={"effective_samples_per_group": 2.0},
    )


def test_repository_identity_constants_distinguish_current_and_frozen_names() -> None:
    assert PROB4D_SOURCE_REPOSITORY == "IPS-Stuttgart/Prob4D"
    assert PROB4D_LEGACY_SOURCE_REPOSITORY == "FlorianPfaff/Prob4D"
    assert PROB4D_SOURCE_REPOSITORIES == {
        PROB4D_SOURCE_REPOSITORY,
        PROB4D_LEGACY_SOURCE_REPOSITORY,
    }
    assert is_prob4d_source_repository(PROB4D_SOURCE_REPOSITORY)
    assert is_prob4d_source_repository(PROB4D_LEGACY_SOURCE_REPOSITORY)
    assert not is_prob4d_source_repository("Example/Prob4D")


def test_frozen_joint_fixture_remains_valid_without_content_rewrite() -> None:
    belief = _joint_fixture_belief()
    validation = validate_prob4d_causal_observation_belief(belief)

    assert belief.source_repository == PROB4D_LEGACY_SOURCE_REPOSITORY
    assert validation["source_repository_is_legacy"] is True
    assert validation["canonical_source_repository"] == PROB4D_SOURCE_REPOSITORY


def test_canonical_joint_fixture_is_validated_through_compatibility_boundary() -> None:
    canonical = replace(
        _joint_fixture_belief(),
        source_repository=PROB4D_SOURCE_REPOSITORY,
    )

    validation = validate_prob4d_causal_observation_belief(canonical)

    assert validation["source_repository"] == PROB4D_SOURCE_REPOSITORY
    assert validation["source_repository_is_legacy"] is False
    assert validation["observation_artifact_id"] == canonical.artifact_id


@pytest.mark.parametrize(
    "source_repository",
    [PROB4D_SOURCE_REPOSITORY, PROB4D_LEGACY_SOURCE_REPOSITORY],
)
def test_provider_attestation_accepts_supported_matching_identity(
    source_repository: str,
) -> None:
    validated = validate_prob4d_provider_attestation(
        _provider_attestation(source_repository),
        source_revision="d" * 40,
        source_repository=source_repository,
        require_claim_bearing=True,
    )

    assert (
        validated["provider_manifest"]["metadata"]["source_repository"]
        == source_repository
    )


def test_provider_attestation_rejects_descriptor_manifest_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="differs from observation source repository"):
        validate_prob4d_provider_attestation(
            _provider_attestation(PROB4D_SOURCE_REPOSITORY),
            source_revision="d" * 40,
            source_repository=PROB4D_LEGACY_SOURCE_REPOSITORY,
            require_claim_bearing=True,
        )


def test_provider_attestation_rejects_unknown_repository_identity() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        validate_prob4d_provider_attestation(
            _provider_attestation(PROB4D_SOURCE_REPOSITORY),
            source_revision="d" * 40,
            source_repository="Example/Prob4D",
            require_claim_bearing=True,
        )


def test_canonical_prob4d_unfused_artifact_keeps_provider_owned_group_power() -> None:
    belief = _unfused_belief(PROB4D_SOURCE_REPOSITORY)
    state = np.zeros((2, 3, 1), dtype=np.float64)
    state[:, 0, 0] = 1.0

    adapted = build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:1],
        physical_response_scale_m=0.05,
    )

    assert adapted.batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    assert adapted.batch.metadata["composite_weight_mode_source"] == (
        "legacy-prob4d-export-metadata"
    )


def test_stable_project_surfaces_use_canonical_repository_urls() -> None:
    checked = (
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "CITATION.cff",
        ROOT / ".github" / "workflows" / "three-repository-golden-path.yml",
        ROOT / "docs" / "three_repository_golden_path.md",
        ROOT / "docs" / "experiment_index.md",
    )
    stale = (
        "https://github.com/FlorianPfaff/Bayesian-PhysTwin",
        "repository: FlorianPfaff/Prob4D",
        "repository: FlorianPfaff/Causal4D",
    )
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for value in stale:
            assert value not in text, f"{path} retains stale repository path {value}"


def test_pep561_marker_is_packaged_with_the_import_root() -> None:
    marker = resources.files("bayesian_phystwin").joinpath("py.typed")
    assert marker.is_file()
