"""Target-closed cohort contract for the RGBench MatPhys risk study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
)

PROTOCOL_SCHEMA: Final = "bayesian-phystwin.rgbench-matphys-cohort-boundary"
PROTOCOL_VERSION: Final = 1
AMENDMENT_SCHEMA: Final = (
    "bayesian-phystwin.rgbench-matphys-preaccess-cohort-amendment"
)
AMENDMENT_VERSION: Final = 1
BASE_POLICY_ID: Final = (
    "13abbe99729a82d58d2a50f3a282abc1ce64b0e068916f39e6f40b2451c45697"
)

RGBENCH_REPOSITORY: Final = "hwk0809/RGBench"
RGBENCH_REVISION: Final = "5cc3d07209362b3bfdbfbc067168dea9a791690a"
RGBENCH_HF_DATASET: Final = "RGBench/RGBench-Cloth-Sim2Real-v1"
RGBENCH_HF_REVISION: Final = "136c00dc5f96b6b3d20427e93875a1c00d7a7cc9"
MATPHYS_REPOSITORY: Final = "Wenqi-Zhao-UESTC/MatPhys"
MATPHYS_REVISION: Final = "c16b858dfb79bf21024ead24b45a710600de7b4f"

GARMENT_SELECTION_SALT: Final = "rgbench-matphys-risk-v1"
CELL_SELECTION_SALT: Final = "rgbench-matphys-cell-v1"
FROZEN_CELL_ROSTER_SHA256: Final = (
    "bc23670b6a3356a8f183a80d1084e972f3890ea9aeb405a11073bb737fd41fa2"
)
ACTIONS: Final = ("fling", "fold", "grasp")

SOURCE_MANIFOLD_GARMENTS: Final = (
    "beige_hoodie",
    "blue_dress",
    "green_tshirt",
    "white_shirt",
)
SOURCE_NONMANIFOLD_GARMENTS: Final = ("grey_sunwear",)
TARGET_MANIFOLD_GARMENTS: Final = (
    "brown_coat",
    "grey_pleat_skirt",
    "white_cakeskirt",
)
TARGET_NONMANIFOLD_GARMENTS: Final = ("khaki_blazer",)

_PROTOCOL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "policy_id",
        "protocol_label",
        "claim_boundary",
        "upstreams",
        "selection",
        "study",
        "source_gate",
        "information_boundary",
    }
)
_UPSTREAM_FIELDS: Final = frozenset(
    {
        "rgbench_repository",
        "rgbench_revision",
        "rgbench_hf_dataset",
        "rgbench_hf_revision",
        "matphys_repository",
        "matphys_revision",
    }
)
_SELECTION_FIELDS: Final = frozenset(
    {
        "garment_selection_salt",
        "cell_selection_salt",
        "cell_roster_sha256",
        "assignment_method",
        "cell_selection_method",
        "actions",
        "source_manifold_garments",
        "source_nonmanifold_garments",
        "target_manifold_garments",
        "target_nonmanifold_garments",
        "source_cells",
        "target_cells",
    }
)
_CELL_FIELDS: Final = frozenset(
    {
        "garment_id",
        "action",
        "sample_id",
        "data_subfolder",
        "selection_key_sha256",
    }
)
_STUDY_FIELDS: Final = frozenset(
    {
        "observation_mode",
        "candidate_family",
        "candidate_scope",
        "source_fit_boundary",
        "comparison_family",
        "risk_signal_family",
        "fallback_policy",
        "target_method_freeze_required",
    }
)
_SOURCE_GATE_FIELDS: Final = frozenset(
    {
        "minimum_source_garments",
        "minimum_source_cells",
        "leave_one_garment_out_required",
        "future_mean_nonregression_required",
        "risk_coverage_dominance_required",
        "nondegenerate_spread_required",
        "exact_fallback_required",
        "separate_target_authorization_artifact_required",
        "deployment_safety_claim_authorized",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "public_metadata_read",
        "source_payload_download_allowed_after_lock",
        "source_payload_decode_allowed_after_lock",
        "source_outcomes_may_be_used_for_development",
        "target_payload_download_allowed",
        "target_payload_decode_allowed",
        "target_outcomes_opened",
        "target_execution_authorized",
        "replacement_allowed",
    }
)
_AMENDMENT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "amendment_id",
        "supersedes_policy_id",
        "protocol_label",
        "claim_boundary",
        "prior_exposure_audit",
        "amended_roles",
        "information_boundary",
    }
)
_PRIOR_AUDIT_FIELDS: Final = frozenset(
    {
        "source_repository_commit",
        "dataset_lock_artifact_sha256",
        "dataset_lock_file_sha256",
        "source_v13_protocol_file_sha256",
        "registered_case_count",
        "registered_garments",
        "target_garments_absent_from_registered_cases",
        "preexisting_full_dataset_cache_declared",
        "audit_scope",
    }
)
_AMENDED_ROLE_FIELDS: Final = frozenset(
    {
        "source_garments",
        "target_garments",
        "source_cell_count",
        "target_cell_count",
        "target_physical_group_count",
        "assignment_method",
    }
)
_AMENDED_BOUNDARY_FIELDS: Final = frozenset(
    {
        "public_metadata_read",
        "prior_source_artifacts_read",
        "preexisting_target_cache_may_exist",
        "source_payload_read_allowed_after_amendment_lock",
        "source_outcomes_may_be_used_for_development",
        "target_payload_read_allowed",
        "target_outcomes_opened",
        "target_execution_authorized",
        "replacement_allowed",
    }
)

_EXPECTED_STUDY: Final = {
    "observation_mode": "rgbench-fixed-point-recorded-end-effector-v1",
    "candidate_family": "native-matphys-warp-spring-mass-ensemble-v1",
    "candidate_scope": "source-development-unfrozen",
    "source_fit_boundary": "causal-prefix-only",
    "comparison_family": "official-rgbench-pybullet-fixed-point-v1",
    "risk_signal_family": "causal-prefix-ensemble-disagreement-v1",
    "fallback_policy": "exact-registered-reference-or-abstain-v1",
    "target_method_freeze_required": True,
}
_EXPECTED_SOURCE_GATE: Final = {
    "minimum_source_garments": 5,
    "minimum_source_cells": 15,
    "leave_one_garment_out_required": True,
    "future_mean_nonregression_required": True,
    "risk_coverage_dominance_required": True,
    "nondegenerate_spread_required": True,
    "exact_fallback_required": True,
    "separate_target_authorization_artifact_required": True,
    "deployment_safety_claim_authorized": False,
}
_EXPECTED_BOUNDARY: Final = {
    "public_metadata_read": True,
    "source_payload_download_allowed_after_lock": True,
    "source_payload_decode_allowed_after_lock": True,
    "source_outcomes_may_be_used_for_development": True,
    "target_payload_download_allowed": False,
    "target_payload_decode_allowed": False,
    "target_outcomes_opened": False,
    "target_execution_authorized": False,
    "replacement_allowed": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(nonempty_string(item, name=f"{name} item") for item in value)
    _require(len(result) == len(set(result)), f"{name} must be unique")
    return result


def _canonical_relative_path(value: object, *, name: str) -> str:
    text = nonempty_string(value, name=name)
    path = PurePosixPath(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require("\\" not in text, f"{name} must use POSIX separators")
    _require(
        path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} must be canonical",
    )
    return text


def _cell_selection_key(
    garment_id: str,
    action: str,
    sample_id: str,
    data_subfolder: str,
) -> str:
    payload = "\0".join(
        (CELL_SELECTION_SALT, garment_id, action, sample_id, data_subfolder)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RGBenchCellV1:
    garment_id: str
    action: str
    sample_id: str
    data_subfolder: str
    selection_key_sha256: str

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return self.garment_id, self.action, self.sample_id, self.data_subfolder


@dataclass(frozen=True, slots=True)
class RGBenchMatPhysProtocolV1:
    value: Mapping[str, Any]
    policy_id: str
    source_cells: tuple[RGBenchCellV1, ...]
    target_cells: tuple[RGBenchCellV1, ...]

    @property
    def target_execution_authorized(self) -> bool:
        boundary = _mapping(self.value["information_boundary"], name="boundary")
        return bool(boundary["target_execution_authorized"])


@dataclass(frozen=True, slots=True)
class AmendedRGBenchMatPhysProtocolV1:
    base: RGBenchMatPhysProtocolV1
    amendment: Mapping[str, Any]
    amendment_id: str
    source_cells: tuple[RGBenchCellV1, ...]
    target_cells: tuple[RGBenchCellV1, ...]

    @property
    def target_execution_authorized(self) -> bool:
        boundary = _mapping(self.amendment["information_boundary"], name="boundary")
        return bool(boundary["target_execution_authorized"])


def _load_cells(
    value: object,
    *,
    name: str,
    expected_garments: frozenset[str],
) -> tuple[RGBenchCellV1, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    cells: list[RGBenchCellV1] = []
    for index, raw in enumerate(value):
        item_name = f"{name}[{index}]"
        cell = _mapping(raw, name=item_name)
        require_exact_fields(cell, expected=_CELL_FIELDS, name=item_name)
        garment_id = nonempty_string(cell["garment_id"], name="garment_id")
        action = nonempty_string(cell["action"], name="action")
        sample_id = nonempty_string(cell["sample_id"], name="sample_id")
        data_subfolder = _canonical_relative_path(
            cell["data_subfolder"], name="data_subfolder"
        )
        digest = sha256_digest(
            cell["selection_key_sha256"], name="selection_key_sha256"
        )
        _require(garment_id in expected_garments, f"{name} garment changed")
        _require(action in ACTIONS, f"{name} action changed")
        _require(
            len(sample_id) == 2 and sample_id.isdigit(),
            f"{name} sample ID must be two decimal digits",
        )
        _require(
            PurePosixPath(data_subfolder).parts[0] == garment_id,
            f"{name} path does not belong to garment",
        )
        _require(
            digest
            == _cell_selection_key(garment_id, action, sample_id, data_subfolder),
            f"{name} selection key changed",
        )
        cells.append(
            RGBenchCellV1(
                garment_id=garment_id,
                action=action,
                sample_id=sample_id,
                data_subfolder=data_subfolder,
                selection_key_sha256=digest,
            )
        )
    identities = tuple(cell.identity for cell in cells)
    _require(
        identities == tuple(sorted(identities)), f"{name} must be canonically sorted"
    )
    expected_pairs = {
        (garment, action) for garment in expected_garments for action in ACTIONS
    }
    actual_pairs = {(cell.garment_id, cell.action) for cell in cells}
    _require(actual_pairs == expected_pairs, f"{name} action coverage changed")
    _require(len(cells) == len(actual_pairs), f"{name} contains duplicate cells")
    return tuple(cells)


def load_rgbench_matphys_protocol_v1(
    path: str | Path,
) -> RGBenchMatPhysProtocolV1:
    value = load_strict_json_object(path, label="RGBench MatPhys cohort protocol")
    require_exact_fields(value, expected=_PROTOCOL_FIELDS, name="protocol")
    _require(value["schema"] == PROTOCOL_SCHEMA, "protocol schema changed")
    _require(value["schema_version"] == PROTOCOL_VERSION, "protocol version changed")
    policy_id = sha256_digest(value["policy_id"], name="policy_id")
    identity = dict(value)
    del identity["policy_id"]
    _require(policy_id == content_id(identity), "policy_id does not match content")
    nonempty_string(value["protocol_label"], name="protocol_label")
    nonempty_string(value["claim_boundary"], name="claim_boundary")

    upstreams = _mapping(value["upstreams"], name="upstreams")
    require_exact_fields(upstreams, expected=_UPSTREAM_FIELDS, name="upstreams")
    _require(
        upstreams["rgbench_repository"] == RGBENCH_REPOSITORY,
        "RGBench repository changed",
    )
    _require(
        exact_revision(upstreams["rgbench_revision"], name="rgbench_revision")
        == RGBENCH_REVISION,
        "RGBench revision changed",
    )
    _require(
        upstreams["rgbench_hf_dataset"] == RGBENCH_HF_DATASET,
        "RGBench dataset changed",
    )
    _require(
        exact_revision(upstreams["rgbench_hf_revision"], name="rgbench_hf_revision")
        == RGBENCH_HF_REVISION,
        "RGBench dataset revision changed",
    )
    _require(
        upstreams["matphys_repository"] == MATPHYS_REPOSITORY,
        "MatPhys repository changed",
    )
    _require(
        exact_revision(upstreams["matphys_revision"], name="matphys_revision")
        == MATPHYS_REVISION,
        "MatPhys revision changed",
    )

    selection = _mapping(value["selection"], name="selection")
    require_exact_fields(selection, expected=_SELECTION_FIELDS, name="selection")
    _require(
        selection["garment_selection_salt"] == GARMENT_SELECTION_SALT,
        "garment selection salt changed",
    )
    _require(
        selection["cell_selection_salt"] == CELL_SELECTION_SALT,
        "cell selection salt changed",
    )
    _require(
        selection["assignment_method"]
        == "metadata-only-stratified-manifold-status-v1",
        "garment assignment changed",
    )
    _require(
        selection["cell_selection_method"]
        == "minimum-salted-sha256-per-garment-action-v1",
        "cell selection method changed",
    )
    actions = _exact_string_list(selection["actions"], name="actions")
    _require(actions == ACTIONS, "action roster changed")
    source_manifold = _exact_string_list(
        selection["source_manifold_garments"], name="source_manifold_garments"
    )
    source_nonmanifold = _exact_string_list(
        selection["source_nonmanifold_garments"],
        name="source_nonmanifold_garments",
    )
    target_manifold = _exact_string_list(
        selection["target_manifold_garments"], name="target_manifold_garments"
    )
    target_nonmanifold = _exact_string_list(
        selection["target_nonmanifold_garments"],
        name="target_nonmanifold_garments",
    )
    _require(source_manifold == SOURCE_MANIFOLD_GARMENTS, "source garments changed")
    _require(
        source_nonmanifold == SOURCE_NONMANIFOLD_GARMENTS,
        "source non-manifold garments changed",
    )
    _require(target_manifold == TARGET_MANIFOLD_GARMENTS, "target garments changed")
    _require(
        target_nonmanifold == TARGET_NONMANIFOLD_GARMENTS,
        "target non-manifold garments changed",
    )
    source_garments = frozenset(source_manifold + source_nonmanifold)
    target_garments = frozenset(target_manifold + target_nonmanifold)
    _require(source_garments.isdisjoint(target_garments), "garment splits overlap")

    source_cells = _load_cells(
        selection["source_cells"],
        name="source_cells",
        expected_garments=source_garments,
    )
    target_cells = _load_cells(
        selection["target_cells"],
        name="target_cells",
        expected_garments=target_garments,
    )
    declared_roster = sha256_digest(
        selection["cell_roster_sha256"], name="cell_roster_sha256"
    )
    computed_roster = content_id(
        {
            "source_cells": selection["source_cells"],
            "target_cells": selection["target_cells"],
        }
    )
    _require(declared_roster == computed_roster, "cell roster digest does not match")
    _require(
        declared_roster == FROZEN_CELL_ROSTER_SHA256, "frozen cell roster changed"
    )

    study = _mapping(value["study"], name="study")
    require_exact_fields(study, expected=_STUDY_FIELDS, name="study")
    _require(study == _EXPECTED_STUDY, "study contract changed")

    source_gate = _mapping(value["source_gate"], name="source_gate")
    require_exact_fields(source_gate, expected=_SOURCE_GATE_FIELDS, name="source_gate")
    _require(source_gate == _EXPECTED_SOURCE_GATE, "source gate changed")

    boundary = _mapping(value["information_boundary"], name="information_boundary")
    require_exact_fields(boundary, expected=_BOUNDARY_FIELDS, name="information_boundary")
    _require(boundary == _EXPECTED_BOUNDARY, "information boundary changed")
    return RGBenchMatPhysProtocolV1(
        value=value,
        policy_id=policy_id,
        source_cells=source_cells,
        target_cells=target_cells,
    )


def load_rgbench_matphys_preaccess_amendment_v1(
    base_path: str | Path,
    amendment_path: str | Path,
) -> AmendedRGBenchMatPhysProtocolV1:
    base = load_rgbench_matphys_protocol_v1(base_path)
    _require(base.policy_id == BASE_POLICY_ID, "base policy changed")
    value = load_strict_json_object(
        amendment_path, label="RGBench MatPhys pre-access amendment"
    )
    require_exact_fields(value, expected=_AMENDMENT_FIELDS, name="amendment")
    _require(value["schema"] == AMENDMENT_SCHEMA, "amendment schema changed")
    _require(
        value["schema_version"] == AMENDMENT_VERSION,
        "amendment version changed",
    )
    amendment_id = sha256_digest(value["amendment_id"], name="amendment_id")
    identity = dict(value)
    del identity["amendment_id"]
    _require(amendment_id == content_id(identity), "amendment_id does not match")
    _require(
        value["supersedes_policy_id"] == BASE_POLICY_ID,
        "superseded policy changed",
    )
    nonempty_string(value["protocol_label"], name="protocol_label")
    nonempty_string(value["claim_boundary"], name="claim_boundary")

    audit = _mapping(value["prior_exposure_audit"], name="prior_exposure_audit")
    require_exact_fields(audit, expected=_PRIOR_AUDIT_FIELDS, name="prior audit")
    _require(
        exact_revision(
            audit["source_repository_commit"], name="source_repository_commit"
        )
        == "0680d2edb1a14647ffd92f2ddcb811fdc54a37d8",
        "prior source repository changed",
    )
    _require(
        sha256_digest(
            audit["dataset_lock_artifact_sha256"],
            name="dataset_lock_artifact_sha256",
        )
        == "3789947cbb9c7c58ccc39b8186b4e30e2a258e615bc2e02c05842bcdafe160e8",
        "prior dataset artifact changed",
    )
    _require(
        sha256_digest(
            audit["dataset_lock_file_sha256"], name="dataset_lock_file_sha256"
        )
        == "7b1db95731c02291031ffb416e456e86cee9d95f164f9198db250ede2612a416",
        "prior dataset lock changed",
    )
    _require(
        sha256_digest(
            audit["source_v13_protocol_file_sha256"],
            name="source_v13_protocol_file_sha256",
        )
        == "6025644cdabd9abd5c106d2c51a012f86f809304b9e7912873d81ca5e56d3e3b",
        "prior source protocol changed",
    )
    prior_garments = _exact_string_list(
        audit["registered_garments"], name="registered_garments"
    )
    expected_prior = tuple(
        sorted(SOURCE_MANIFOLD_GARMENTS + TARGET_MANIFOLD_GARMENTS)
    )
    _require(prior_garments == expected_prior, "prior garment roster changed")
    _require(audit["registered_case_count"] == 63, "prior case count changed")
    _require(
        audit["target_garments_absent_from_registered_cases"] is True,
        "target absence audit changed",
    )
    _require(
        audit["preexisting_full_dataset_cache_declared"] is True,
        "preexisting cache declaration changed",
    )
    nonempty_string(audit["audit_scope"], name="audit_scope")

    roles = _mapping(value["amended_roles"], name="amended_roles")
    require_exact_fields(roles, expected=_AMENDED_ROLE_FIELDS, name="amended roles")
    source_garments = _exact_string_list(
        roles["source_garments"], name="source_garments"
    )
    target_garments = _exact_string_list(
        roles["target_garments"], name="target_garments"
    )
    expected_source = tuple(sorted(expected_prior))
    expected_target = tuple(
        sorted(SOURCE_NONMANIFOLD_GARMENTS + TARGET_NONMANIFOLD_GARMENTS)
    )
    _require(source_garments == expected_source, "amended source garments changed")
    _require(target_garments == expected_target, "amended target garments changed")
    _require(set(source_garments).isdisjoint(target_garments), "amended roles overlap")
    _require(roles["source_cell_count"] == 21, "source cell count changed")
    _require(roles["target_cell_count"] == 6, "target cell count changed")
    _require(
        roles["target_physical_group_count"] == 2,
        "target physical group count changed",
    )
    _require(
        roles["assignment_method"]
        == "prior-exposure-audited-garment-outcome-status-v1",
        "amended assignment method changed",
    )

    cells = base.source_cells + base.target_cells
    source_set = frozenset(source_garments)
    target_set = frozenset(target_garments)
    source_cells = tuple(cell for cell in cells if cell.garment_id in source_set)
    target_cells = tuple(cell for cell in cells if cell.garment_id in target_set)
    _require(len(source_cells) == 21, "derived source cells changed")
    _require(len(target_cells) == 6, "derived target cells changed")
    _require(
        len(source_cells) + len(target_cells) == len(cells),
        "amended roles do not cover the base roster",
    )

    boundary = _mapping(value["information_boundary"], name="information_boundary")
    require_exact_fields(
        boundary,
        expected=_AMENDED_BOUNDARY_FIELDS,
        name="amended information boundary",
    )
    _require(
        boundary
        == {
            "public_metadata_read": True,
            "prior_source_artifacts_read": True,
            "preexisting_target_cache_may_exist": True,
            "source_payload_read_allowed_after_amendment_lock": True,
            "source_outcomes_may_be_used_for_development": True,
            "target_payload_read_allowed": False,
            "target_outcomes_opened": False,
            "target_execution_authorized": False,
            "replacement_allowed": False,
        },
        "amended information boundary changed",
    )
    return AmendedRGBenchMatPhysProtocolV1(
        base=base,
        amendment=value,
        amendment_id=amendment_id,
        source_cells=source_cells,
        target_cells=target_cells,
    )


__all__ = [
    "ACTIONS",
    "AMENDMENT_SCHEMA",
    "AmendedRGBenchMatPhysProtocolV1",
    "CELL_SELECTION_SALT",
    "FROZEN_CELL_ROSTER_SHA256",
    "GARMENT_SELECTION_SALT",
    "MATPHYS_REVISION",
    "RGBENCH_HF_REVISION",
    "RGBENCH_REVISION",
    "RGBenchCellV1",
    "RGBenchMatPhysProtocolV1",
    "load_rgbench_matphys_protocol_v1",
    "load_rgbench_matphys_preaccess_amendment_v1",
]
