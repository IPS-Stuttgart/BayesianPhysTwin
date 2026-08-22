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


__all__ = [
    "ACTIONS",
    "CELL_SELECTION_SALT",
    "FROZEN_CELL_ROSTER_SHA256",
    "GARMENT_SELECTION_SALT",
    "MATPHYS_REVISION",
    "RGBENCH_HF_REVISION",
    "RGBENCH_REVISION",
    "RGBenchCellV1",
    "RGBenchMatPhysProtocolV1",
    "load_rgbench_matphys_protocol_v1",
]
