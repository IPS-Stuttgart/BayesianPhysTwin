"""Prospective boundary for the fresh Deform360 reusable-twin panel."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping


FRESH_PROTOCOL_ID = "deform360-reusable-trust-fresh-v1"
PHYSICS_ADDENDUM_ID = "deform360-reusable-trust-physics-addendum-v1"
EXECUTION_LOCK_ID = "deform360-reusable-trust-execution-v1"
EXPECTED_DENSE_CONFIG_SHA256 = (
    "8a90705dd38c6c90b042ed8f450e2bc7e3cffc54b965765b004d0385999d40ea"
)
EXPECTED_SPLITS = {
    "003-cable": {
        "topology": "1D",
        "fit_episode_ids": (1, 3, 4, 6, 7, 9),
        "held_out_episode_ids": (0, 2, 5, 8),
    },
    "086-cotton-scarf-cloth": {
        "topology": "2D",
        "fit_episode_ids": (1, 3, 4, 6, 7, 9),
        "held_out_episode_ids": (0, 2, 5, 8),
    },
    "171-penguin": {
        "topology": "3D",
        "fit_episode_ids": (1, 3, 4, 6, 7, 9),
        "held_out_episode_ids": (0, 2, 5, 8),
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path.name} must contain an object")
    return payload


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_splits(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = {}
    for object_id, split in payload.items():
        _require(isinstance(split, Mapping), f"invalid split for {object_id}")
        normalized[str(object_id)] = {
            "topology": str(split.get("topology")),
            "fit_episode_ids": tuple(
                int(value) for value in split.get("fit_episode_ids", ())
            ),
            "held_out_episode_ids": tuple(
                int(value) for value in split.get("held_out_episode_ids", ())
            ),
        }
    return normalized


def load_reusable_trust_protocol(
    parent_path: str | Path,
    addendum_path: str | Path,
    execution_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and jointly validate the fresh panel and physical-fit addendum."""

    parent_file = Path(parent_path).resolve()
    addendum_file = Path(addendum_path).resolve()
    parent = _load_json(parent_file)
    addendum = _load_json(addendum_file)
    _require(parent.get("schema_version") == 1, "fresh protocol schema changed")
    _require(
        parent.get("protocol_id") == FRESH_PROTOCOL_ID,
        "fresh protocol identity changed",
    )
    _require(
        parent.get("fresh_object_panel", {}).get(
            "prediction_must_be_hashed_before_any_held_out_outcome_is_opened"
        )
        is True,
        "fresh prediction-first boundary changed",
    )
    _require(
        parent.get("fresh_object_panel", {}).get("all_held_out_episodes_must_be_scored")
        is True,
        "fresh all-episode requirement changed",
    )
    _require(addendum.get("schema_version") == 1, "physics addendum schema changed")
    _require(
        addendum.get("protocol_id") == PHYSICS_ADDENDUM_ID,
        "physics addendum identity changed",
    )
    _require(
        addendum.get("parent_protocol_id") == FRESH_PROTOCOL_ID,
        "physics addendum uses another parent",
    )
    _require(
        addendum.get("parent_protocol_file_sha256") == _sha256_file(parent_file),
        "physics addendum parent hash changed",
    )
    observed_splits = _normalized_splits(addendum.get("objects_and_splits", {}))
    _require(observed_splits == EXPECTED_SPLITS, "fresh object split changed")
    for object_id, split in observed_splits.items():
        _require(
            not set(split["fit_episode_ids"]) & set(split["held_out_episode_ids"]),
            f"fit and held episodes overlap for {object_id}",
        )
    panel = parent.get("fresh_object_panel", {})
    _require(
        tuple(panel.get("object_level_fit_episode_ids", ()))
        == EXPECTED_SPLITS["003-cable"]["fit_episode_ids"],
        "parent fit episodes changed",
    )
    _require(
        tuple(panel.get("held_out_episode_ids", ()))
        == EXPECTED_SPLITS["003-cable"]["held_out_episode_ids"],
        "parent held episodes changed",
    )
    _require(
        set(panel.get("objects", {})) == set(EXPECTED_SPLITS),
        "parent fresh objects changed",
    )
    grid = addendum.get("object_level_physical_grid", {})
    candidates = tuple(
        itertools.product(
            grid.get("init_spring_y", ()),
            grid.get("drag_damping", ()),
            grid.get("dashpot_damping", ()),
        )
    )
    _require(
        len(candidates) == int(grid.get("candidate_count", -1)) == 18,
        "physical candidate grid changed",
    )
    boundary = addendum.get("held_prediction_boundary", {})
    _require(
        boundary.get("all_twelve_predictions_hashed_before_any_held_outcome_is_opened")
        is True
        and boundary.get("closure_rejection_returns_exact_persistence") is True,
        "held prediction boundary changed",
    )
    _require(
        addendum.get("reference_trust_response", {}).get("candidate_result_sha256")
        == parent.get("discovery_evidence", {}).get("candidate_result_sha256"),
        "parent and addendum use different trust candidates",
    )
    result = {
        "parent": parent,
        "addendum": addendum,
        "parent_path": str(parent_file),
        "addendum_path": str(addendum_file),
        "parent_file_sha256": _sha256_file(parent_file),
        "addendum_file_sha256": _sha256_file(addendum_file),
        "splits": observed_splits,
        "physical_candidates": [
            {
                "init_spring_y": float(spring),
                "drag_damping": float(drag),
                "dashpot_damping": float(dashpot),
            }
            for spring, drag, dashpot in candidates
        ],
    }
    if execution_path is not None:
        execution_file = Path(execution_path).resolve()
        execution = _load_json(execution_file)
        _require(execution.get("schema_version") == 1, "execution schema changed")
        _require(
            execution.get("protocol_id") == EXECUTION_LOCK_ID,
            "execution lock identity changed",
        )
        _require(
            execution.get("parent_physics_addendum_id") == PHYSICS_ADDENDUM_ID
            and execution.get("parent_physics_addendum_file_sha256")
            == result["addendum_file_sha256"],
            "execution lock uses another physics addendum",
        )
        canonical = execution.get("canonical_object_graph", {})
        _require(
            canonical.get("one_graph_per_object") is True
            and canonical.get("reference_episode_rule")
            == "lowest frozen fit episode id"
            and int(canonical.get("reference_episode_id", -1)) == 1
            and int(canonical.get("maximum_observed_node_count", -1)) == 384
            and canonical.get("shared_across_fit_and_held_episodes") is True
            and canonical.get("object_springs_and_rest_lengths_rebuilt_per_episode")
            is False
            and canonical.get("controller_or_contact_springs_embedded_in_shared_graph")
            is False
            and canonical.get("registration_uses_simulator_residual") is False
            and canonical.get("registration_uses_post_initial_object_observation")
            is False,
            "canonical reusable-graph execution semantics changed",
        )
        _require(
            canonical.get("registration_config_source_file_sha256")
            == EXPECTED_DENSE_CONFIG_SHA256,
            "canonical registration configuration changed",
        )
        attachment = execution.get("dynamic_controller_attachment", {})
        _require(
            int(attachment.get("controller_group_size", -1)) == 768
            and attachment.get("anchor_rule")
            == "nearest canonical state node to each frame-zero controller group"
            and int(attachment.get("canonical_patch_size_per_anchor", -1)) == 16
            and float(attachment.get("controller_radius_m", -1.0)) == 0.03
            and int(attachment.get("controller_max_neighbours", -1)) == 1
            and attachment.get("known_future_controller_trajectory_used") is True
            and attachment.get("tactile_used") is False,
            "dynamic controller-attachment execution semantics changed",
        )
        preprocessing = execution.get("preprocessing", {})
        _require(
            preprocessing.get("tactile_copied_to_prediction_stage") is False
            and preprocessing.get(
                "post_initial_object_observation_used_before_prediction_seal"
            )
            is False,
            "execution preprocessing crossed the prediction boundary",
        )
        result.update(
            {
                "execution": execution,
                "execution_path": str(execution_file),
                "execution_file_sha256": _sha256_file(execution_file),
            }
        )
    return result


def authorize_reusable_trust_episode(
    protocol: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
    operation: str,
) -> dict[str, Any]:
    """Authorize fit access or outcome-blind held prediction before any I/O."""

    _require(
        protocol.get("parent", {}).get("protocol_id") == FRESH_PROTOCOL_ID
        and protocol.get("addendum", {}).get("protocol_id") == PHYSICS_ADDENDUM_ID,
        "episode authorization uses another protocol",
    )
    split = protocol.get("splits", {}).get(str(object_id))
    _require(split is not None, "object is outside the fresh panel")
    episode = int(episode_id)
    if operation == "fit":
        _require(episode in split["fit_episode_ids"], "episode is not a fit episode")
        role = "object-level-fit"
        outcome_allowed = True
    elif operation == "held-prediction":
        _require(
            episode in split["held_out_episode_ids"],
            "episode is not a held prediction episode",
        )
        role = "held-out-prediction"
        outcome_allowed = False
    else:
        raise ValueError(f"unsupported fresh-panel operation: {operation}")
    authorization = {
        "protocol_id": FRESH_PROTOCOL_ID,
        "physics_addendum_id": PHYSICS_ADDENDUM_ID,
        "parent_file_sha256": str(protocol["parent_file_sha256"]),
        "addendum_file_sha256": str(protocol["addendum_file_sha256"]),
        "object_id": str(object_id),
        "episode_id": episode,
        "episode_key": f"{object_id}/{episode}",
        "topology": str(split["topology"]),
        "role": role,
        "fit_outcome_allowed": outcome_allowed,
        "held_outcome_allowed": False,
    }
    if "execution" in protocol:
        authorization.update(
            {
                "execution_protocol_id": EXECUTION_LOCK_ID,
                "execution_file_sha256": str(protocol["execution_file_sha256"]),
            }
        )
    return authorization


def validate_reusable_trust_prediction(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    verify_archive: bool = True,
) -> dict[str, Any]:
    """Validate one outcome-blind held prediction and its prospective role."""

    _require(payload.get("schema_version") == 1, "prediction schema changed")
    _require(
        payload.get("artifact_kind") == "Deform360ReusableTwinTrustedPrediction",
        "unexpected trusted prediction kind",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "prediction checksum mismatch",
    )
    object_id = str(payload.get("object_id"))
    episode_id = int(payload.get("episode_id", -1))
    expected = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation="held-prediction",
    )
    _require(
        payload.get("prospective_authorization") == expected,
        "prediction authorization changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("post_initial_object_observation_used") is False
        and boundary.get("tactile_used") is False
        and boundary.get("symbolic_action_label_used") is False
        and boundary.get("object_outcome_used") is False
        and boundary.get("trust_inferred_from_fixed_reference_response") is True
        and boundary.get("candidate_physics_cannot_change_trust") is True,
        "prediction crossed its held-out information boundary",
    )
    expected_candidate = protocol["addendum"]["reference_trust_response"][
        "candidate_result_sha256"
    ]
    _require(
        payload.get("model", {}).get("result_sha256") == expected_candidate,
        "prediction uses another trust candidate",
    )
    from .deform360_reusable_physics import validate_reusable_physics_selection

    selection_record = payload.get("physical_selection", {})
    selection_path = Path(str(selection_record.get("path", "")))
    _require(
        selection_path.is_file()
        and _sha256_file(selection_path) == selection_record.get("file_sha256"),
        "prediction physical selection is missing or changed",
    )
    selection = validate_reusable_physics_selection(
        _load_json(selection_path), protocol=protocol
    )
    _require(
        selection["object_id"] == object_id
        and selection["result_sha256"] == selection_record.get("result_sha256")
        and selection["selected_candidate_index"]
        == int(selection_record.get("selected_candidate_index", -1))
        and selection["selected_physical_parameters"]
        == selection_record.get("selected_physical_parameters"),
        "prediction uses another object-level physical selection",
    )
    responses = payload.get("physical_responses", {})
    _require(
        isinstance(responses.get("reference_result_sha256"), str)
        and isinstance(responses.get("application_result_sha256"), str),
        "prediction lacks physical response provenance",
    )
    output = payload.get("output", {})
    _require(
        isinstance(output.get("path"), str)
        and isinstance(output.get("sha256"), str)
        and payload.get("output_sha256") == output.get("sha256"),
        "prediction output record is incomplete",
    )
    if verify_archive:
        archive = Path(output["path"])
        _require(
            archive.is_file() and _sha256_file(archive) == output["sha256"],
            "prediction output archive changed",
        )
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_key": f"{object_id}/{episode_id}",
        "prediction_result_sha256": str(payload["result_sha256"]),
        "prediction_file_sha256": str(output["sha256"]),
        "prediction_path": str(output["path"]),
    }


def build_reusable_trust_prediction_cohort_seal(
    prediction_paths: list[str | Path],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal all twelve held predictions before any held outcome may open."""

    expected_keys = {
        f"{object_id}/{episode_id}"
        for object_id, split in EXPECTED_SPLITS.items()
        for episode_id in split["held_out_episode_ids"]
    }
    entries: dict[str, Any] = {}
    for path_value in prediction_paths:
        path = Path(path_value).resolve()
        payload = _load_json(path)
        record = validate_reusable_trust_prediction(
            payload, protocol=protocol, verify_archive=True
        )
        key = record["episode_key"]
        _require(key not in entries, f"duplicate prediction for {key}")
        entries[key] = {
            **record,
            "prediction_json_path": str(path),
            "prediction_json_sha256": _sha256_file(path),
        }
    _require(
        set(entries) == expected_keys,
        "cohort seal does not contain all held episodes",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinPredictionCohortSeal",
        "protocol_id": FRESH_PROTOCOL_ID,
        "physics_addendum_id": PHYSICS_ADDENDUM_ID,
        "parent_file_sha256": str(protocol["parent_file_sha256"]),
        "addendum_file_sha256": str(protocol["addendum_file_sha256"]),
        "prediction_count": len(entries),
        "predictions": dict(sorted(entries.items())),
        "information_boundary": {
            "all_held_predictions_hashed": True,
            "held_outcomes_read": False,
            "post_initial_held_observations_used": False,
        },
        "claim_boundary": (
            "prediction cohort seal only; no held outcome or state-of-the-art claim"
        ),
    }
    if "execution" in protocol:
        payload.update(
            {
                "execution_protocol_id": EXECUTION_LOCK_ID,
                "execution_file_sha256": str(protocol["execution_file_sha256"]),
            }
        )
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_reusable_trust_prediction_cohort_seal(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    verify_predictions: bool = True,
) -> dict[str, Any]:
    """Validate completeness and immutability of a fresh prediction cohort."""

    _require(payload.get("schema_version") == 1, "cohort seal schema changed")
    _require(
        payload.get("artifact_kind")
        == "Deform360ReusableTwinPredictionCohortSeal",
        "unexpected cohort seal kind",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "cohort seal checksum mismatch",
    )
    _require(
        payload.get("parent_file_sha256") == protocol["parent_file_sha256"]
        and payload.get("addendum_file_sha256")
        == protocol["addendum_file_sha256"],
        "cohort seal uses another protocol",
    )
    if "execution" in protocol:
        _require(
            payload.get("execution_protocol_id") == EXECUTION_LOCK_ID
            and payload.get("execution_file_sha256")
            == protocol["execution_file_sha256"],
            "cohort seal uses another execution lock",
        )
    expected_keys = {
        f"{object_id}/{episode_id}"
        for object_id, split in EXPECTED_SPLITS.items()
        for episode_id in split["held_out_episode_ids"]
    }
    predictions = payload.get("predictions", {})
    _require(
        isinstance(predictions, Mapping)
        and set(predictions) == expected_keys
        and int(payload.get("prediction_count", -1)) == len(expected_keys),
        "cohort seal is incomplete",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("all_held_predictions_hashed") is True
        and boundary.get("held_outcomes_read") is False
        and boundary.get("post_initial_held_observations_used") is False,
        "cohort seal crossed the held-out boundary",
    )
    if verify_predictions:
        for key, record in predictions.items():
            prediction_path = Path(str(record.get("prediction_json_path", "")))
            _require(
                prediction_path.is_file()
                and _sha256_file(prediction_path)
                == record.get("prediction_json_sha256"),
                f"prediction record changed for {key}",
            )
            validated = validate_reusable_trust_prediction(
                _load_json(prediction_path), protocol=protocol, verify_archive=True
            )
            _require(
                validated["episode_key"] == key
                and validated["prediction_result_sha256"]
                == record.get("prediction_result_sha256"),
                f"prediction identity changed for {key}",
            )
    return {
        "prediction_count": len(expected_keys),
        "episode_keys": sorted(expected_keys),
    }


def authorize_reusable_trust_held_outcome(
    protocol: Mapping[str, Any],
    cohort_seal: Mapping[str, Any],
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Permit held scoring only after the complete prediction cohort is sealed."""

    validated = validate_reusable_trust_prediction_cohort_seal(
        cohort_seal, protocol=protocol, verify_predictions=True
    )
    split = protocol.get("splits", {}).get(str(object_id))
    _require(split is not None, "object is outside the fresh panel")
    episode = int(episode_id)
    _require(episode in split["held_out_episode_ids"], "episode is not held out")
    key = f"{object_id}/{episode}"
    _require(key in validated["episode_keys"], "held prediction is not sealed")
    return {
        "protocol_id": FRESH_PROTOCOL_ID,
        "physics_addendum_id": PHYSICS_ADDENDUM_ID,
        "cohort_seal_result_sha256": str(cohort_seal["result_sha256"]),
        "object_id": str(object_id),
        "episode_id": episode,
        "episode_key": key,
        "held_outcome_allowed": True,
        "method_or_hyperparameter_changes_allowed": False,
        **(
            {
                "execution_protocol_id": EXECUTION_LOCK_ID,
                "execution_file_sha256": str(protocol["execution_file_sha256"]),
            }
            if "execution" in protocol
            else {}
        ),
    }


__all__ = [
    "EXPECTED_SPLITS",
    "EXECUTION_LOCK_ID",
    "FRESH_PROTOCOL_ID",
    "PHYSICS_ADDENDUM_ID",
    "authorize_reusable_trust_episode",
    "authorize_reusable_trust_held_outcome",
    "build_reusable_trust_prediction_cohort_seal",
    "load_reusable_trust_protocol",
    "validate_reusable_trust_prediction",
    "validate_reusable_trust_prediction_cohort_seal",
]
