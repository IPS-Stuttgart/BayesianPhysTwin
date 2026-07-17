import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_reusable_trust_protocol import (
    EXPECTED_SPLITS,
    authorize_reusable_trust_episode,
    authorize_reusable_trust_held_outcome,
    build_reusable_trust_prediction_cohort_seal,
    load_reusable_trust_protocol,
    validate_reusable_trust_prediction_cohort_seal,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/causal4d_public/deform360_reusable_trust_fresh_v1.json"
ADDENDUM = (
    ROOT
    / "configs/causal4d_public/deform360_reusable_trust_physics_addendum_v1.json"
)
EXECUTION = (
    ROOT / "configs/causal4d_public/deform360_reusable_trust_execution_v1.json"
)


def test_fresh_reusable_trust_protocol_loads_canonical_lock() -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)

    assert len(protocol["physical_candidates"]) == 18
    assert set(protocol["splits"]) == {
        "003-cable",
        "086-cotton-scarf-cloth",
        "171-penguin",
    }


def test_execution_lock_requires_one_shared_object_graph() -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM, EXECUTION)
    authorization = authorize_reusable_trust_episode(
        protocol, object_id="003-cable", episode_id=1, operation="fit"
    )

    assert protocol["execution"]["canonical_object_graph"]["one_graph_per_object"]
    assert (
        protocol["execution"]["dynamic_controller_attachment"][
            "canonical_patch_size_per_anchor"
        ]
        == 16
    )
    assert authorization["execution_file_sha256"] == protocol[
        "execution_file_sha256"
    ]


def test_execution_lock_rejects_episode_specific_rest_geometry(tmp_path) -> None:
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    execution["canonical_object_graph"][
        "object_springs_and_rest_lengths_rebuilt_per_episode"
    ] = True
    changed = tmp_path / "execution.json"
    changed.write_text(json.dumps(execution), encoding="utf-8")

    with pytest.raises(ValueError, match="execution semantics"):
        load_reusable_trust_protocol(PARENT, ADDENDUM, changed)


def test_fresh_reusable_trust_authorization_separates_fit_and_held() -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)

    fit = authorize_reusable_trust_episode(
        protocol, object_id="003-cable", episode_id=1, operation="fit"
    )
    held = authorize_reusable_trust_episode(
        protocol,
        object_id="003-cable",
        episode_id=0,
        operation="held-prediction",
    )

    assert fit["fit_outcome_allowed"]
    assert not fit["held_outcome_allowed"]
    assert not held["fit_outcome_allowed"]
    assert not held["held_outcome_allowed"]


@pytest.mark.parametrize(
    ("episode_id", "operation"),
    [(0, "fit"), (1, "held-prediction"), (0, "held-outcome")],
)
def test_fresh_reusable_trust_authorization_rejects_boundary_crossing(
    episode_id: int, operation: str
) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)

    with pytest.raises(ValueError):
        authorize_reusable_trust_episode(
            protocol,
            object_id="171-penguin",
            episode_id=episode_id,
            operation=operation,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prediction_paths(tmp_path: Path, protocol: dict) -> list[Path]:
    paths = []
    candidate_hash = protocol["addendum"]["reference_trust_response"][
        "candidate_result_sha256"
    ]
    for object_id, split in EXPECTED_SPLITS.items():
        selection = {
            "schema_version": 1,
            "artifact_kind": "Deform360ReusableTwinPhysicalSelection",
            "parent_file_sha256": protocol["parent_file_sha256"],
            "addendum_file_sha256": protocol["addendum_file_sha256"],
            "object_id": object_id,
            "fit_episode_ids": list(split["fit_episode_ids"]),
            "selected_candidate_index": 0,
            "selected_physical_parameters": protocol["physical_candidates"][0],
            "information_boundary": {
                "held_out_actions_read": False,
                "held_out_initial_geometry_read": False,
                "held_out_outcomes_read": False,
                "trust_inferred_only_from_fixed_reference_response": True,
                "candidate_physics_cannot_change_trust": True,
            },
        }
        selection["result_sha256"] = _result_sha256(selection)
        selection_path = tmp_path / f"{object_id}-selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        for episode_id in split["held_out_episode_ids"]:
            stem = f"{object_id}-ep{episode_id:04d}"
            archive = tmp_path / f"{stem}.npz"
            np.savez_compressed(archive, prediction_m=np.zeros((2, 1, 3)))
            payload = {
                "schema_version": 1,
                "artifact_kind": "Deform360ReusableTwinTrustedPrediction",
                "object_id": object_id,
                "episode_id": episode_id,
                "prospective_authorization": authorize_reusable_trust_episode(
                    protocol,
                    object_id=object_id,
                    episode_id=episode_id,
                    operation="held-prediction",
                ),
                "model": {"result_sha256": candidate_hash},
                "physical_selection": {
                    "path": str(selection_path),
                    "file_sha256": _sha256(selection_path),
                    "result_sha256": selection["result_sha256"],
                    "selected_candidate_index": 0,
                    "selected_physical_parameters": protocol[
                        "physical_candidates"
                    ][0],
                },
                "physical_responses": {
                    "reference_result_sha256": "a" * 64,
                    "application_result_sha256": "b" * 64,
                },
                "output": {"path": str(archive), "sha256": _sha256(archive)},
                "output_sha256": _sha256(archive),
                "information_boundary": {
                    "post_initial_object_observation_used": False,
                    "tactile_used": False,
                    "symbolic_action_label_used": False,
                    "object_outcome_used": False,
                    "trust_inferred_from_fixed_reference_response": True,
                    "candidate_physics_cannot_change_trust": True,
                },
            }
            payload["result_sha256"] = _result_sha256(payload)
            path = tmp_path / f"{stem}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
    return paths


def test_cohort_seal_requires_and_validates_all_twelve_predictions(tmp_path) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)
    paths = _prediction_paths(tmp_path, protocol)

    seal = build_reusable_trust_prediction_cohort_seal(paths, protocol=protocol)
    validated = validate_reusable_trust_prediction_cohort_seal(
        seal, protocol=protocol, verify_predictions=True
    )
    outcome = authorize_reusable_trust_held_outcome(
        protocol, seal, object_id="086-cotton-scarf-cloth", episode_id=5
    )

    assert validated["prediction_count"] == 12
    assert outcome["held_outcome_allowed"]
    assert not outcome["method_or_hyperparameter_changes_allowed"]


def test_cohort_seal_rejects_an_incomplete_prediction_set(tmp_path) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)
    paths = _prediction_paths(tmp_path, protocol)

    with pytest.raises(ValueError, match="all held episodes"):
        build_reusable_trust_prediction_cohort_seal(
            paths[:-1], protocol=protocol
        )
