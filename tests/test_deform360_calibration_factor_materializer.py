from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_prob4d_causal_lineage import _attested_belief, _belief

from bayesian_phystwin.deform360_calibration_factor_materializer import (
    build_deform360_kinematic_contact_anchor,
    materialize_deform360_calibration_factors,
    publish_deform360_calibration_factor_materialization,
)
from bayesian_phystwin.deform360_contact_anchor import (
    Deform360ContactAnchorV1,
    load_deform360_contact_anchor,
    save_deform360_contact_anchor,
)
from bayesian_phystwin.observation_belief import save_observation_belief
from bayesian_phystwin.physical_linearization import (
    PhysicalLinearizationV1,
    save_physical_linearization,
)

_CLI_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/science/materialize_deform360_calibration_factors.py"
)
_CLI_SPEC = importlib.util.spec_from_file_location(
    "deform360_calibration_factor_materializer_cli",
    _CLI_PATH,
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(_CLI)


def _contact_inputs(*, case_id: str = "case") -> dict[str, object]:
    centers = np.asarray(
        [
            [0.00, 0.00, 0.50],
            [0.01, 0.00, 0.50],
            [0.02, 0.00, 0.50],
            [0.03, 0.00, 0.50],
        ]
    )
    offsets = np.asarray(
        [
            [-0.002, 0.000, 0.000],
            [0.002, 0.000, 0.000],
            [0.000, -0.002, 0.000],
            [0.000, 0.002, 0.000],
        ]
    )
    positions = centers[:, None, :] + offsets[None, :, :]
    state = np.zeros((4, 3, 2), dtype=np.float64)
    state[:, 0, 0] = np.asarray([-1.0, 1.0, -1.0, 1.0])
    state[:, 1, 1] = np.asarray([-1.0, -1.0, 1.0, 1.0])
    coefficients = np.asarray([0.006, -0.004])
    innovation = np.einsum("acs,s->ac", state, coefficients)
    return {
        "object_id": "public-calibration-object",
        "observation_case_id": case_id,
        "episode_id": 0,
        "causal_frame_stop": 6,
        "frame_ids": np.asarray([1, 2, 3, 4]),
        "sensor_names": ("gripper-0",) * 4,
        "contact_episode_ids": ("contact-0",) * 4,
        "tactile_response": np.asarray(
            [
                [1.0, 2.0, 1.0, 2.0],
                [2.0, 1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 1.0, 1.0],
            ]
        ),
        "taxel_world_positions_m": positions,
        "physical_patch_prediction_m": centers - innovation,
        "state_jacobian": state,
        "source_reliability": np.asarray([0.9, 0.8, 0.7, 0.6]),
        "source_revision": "d" * 40,
        "source_artifacts": {
            "public/tactile/synced_tactile.npy": "1" * 64,
            "public/robot/robot.npz": "2" * 64,
        },
    }


def _anchor(**updates: object) -> Deform360ContactAnchorV1:
    values = _contact_inputs()
    values.update(updates)
    return build_deform360_kinematic_contact_anchor(**values)


def _linearization() -> PhysicalLinearizationV1:
    belief = _attested_belief()
    state = np.zeros((belief.observation_count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = np.asarray([-1.0, 1.0, -1.0, 1.0])
    state[:, 1, 1] = np.asarray([-1.0, -1.0, 1.0, 1.0])
    query = np.zeros((2, 3, 2), dtype=np.float64)
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    return PhysicalLinearizationV1(
        observation_artifact_id=belief.artifact_id,
        baseline_belief_id="8" * 64,
        action_prefix_id="9" * 64,
        simulator_revision="simulator-revision",
        frame_ids=belief.frame_ids,
        entity_ids=belief.entity_ids,
        view_indices=belief.view_indices,
        window_indices=belief.window_indices,
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=np.asarray(
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        ),
    )


def test_kinematic_anchor_does_not_count_duplicate_taxels_twice() -> None:
    inputs = _contact_inputs()
    original = build_deform360_kinematic_contact_anchor(**inputs)
    duplicated = build_deform360_kinematic_contact_anchor(
        **{
            **inputs,
            "tactile_response": np.concatenate(
                [inputs["tactile_response"], inputs["tactile_response"]],
                axis=1,
            ),
            "taxel_world_positions_m": np.concatenate(
                [
                    inputs["taxel_world_positions_m"],
                    inputs["taxel_world_positions_m"],
                ],
                axis=1,
            ),
        }
    )

    np.testing.assert_allclose(
        duplicated.innovation_m,
        original.innovation_m,
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        duplicated.covariance_m2,
        original.covariance_m2,
        atol=1e-18,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        duplicated.prior_reliability,
        original.prior_reliability,
    )
    assert duplicated.metadata["duplicate_taxel_block_increases_confidence"] is False


def test_state_residual_does_not_change_prior_contact_reliability() -> None:
    inputs = _contact_inputs()
    original = build_deform360_kinematic_contact_anchor(**inputs)
    shifted_prediction = np.asarray(inputs["physical_patch_prediction_m"]) + 0.2
    shifted = build_deform360_kinematic_contact_anchor(
        **{**inputs, "physical_patch_prediction_m": shifted_prediction}
    )

    assert not np.array_equal(shifted.innovation_m, original.innovation_m)
    np.testing.assert_array_equal(shifted.prior_reliability, original.prior_reliability)
    np.testing.assert_array_equal(shifted.composite_weight, original.composite_weight)
    np.testing.assert_array_equal(shifted.covariance_m2, original.covariance_m2)
    assert shifted.metadata["source_reliability_depends_on_state_innovation"] is False


def test_anchor_requires_two_unique_active_taxels() -> None:
    inputs = _contact_inputs()
    response = np.zeros_like(inputs["tactile_response"])
    response[:, 0] = 1.0

    with pytest.raises(ValueError, match="too few unique active taxels"):
        build_deform360_kinematic_contact_anchor(
            **{**inputs, "tactile_response": response}
        )


def test_contact_anchor_archive_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    anchor = _anchor()
    path = tmp_path / "contact-anchor.npz"
    save_deform360_contact_anchor(path, anchor)

    loaded = load_deform360_contact_anchor(path)

    assert loaded.artifact_id == anchor.artifact_id
    np.testing.assert_array_equal(loaded.innovation_m, anchor.innovation_m)
    assert loaded.summary() == anchor.summary()

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["innovation_m"] = payload["innovation_m"].copy()
    payload["innovation_m"][0, 0] += 1.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="artifact ID"):
        load_deform360_contact_anchor(path)


def test_materializer_requires_calibrated_prob4d_before_innovation() -> None:
    exploratory = _belief()
    linearization = replace(
        _linearization(),
        observation_artifact_id=exploratory.artifact_id,
        frame_ids=exploratory.frame_ids,
        entity_ids=exploratory.entity_ids,
        view_indices=exploratory.view_indices,
        window_indices=exploratory.window_indices,
    )

    with pytest.raises(ValueError, match="provider-v2 attestation is required"):
        materialize_deform360_calibration_factors(
            exploratory,
            linearization,
            _anchor(),
            physical_prediction_xyz_m=np.zeros_like(exploratory.mean_xyz_m),
            physical_query_jacobian=np.eye(2),
        )


def test_materializer_binds_case_and_adds_group_capped_contact_information() -> None:
    belief = _attested_belief()
    materialization = materialize_deform360_calibration_factors(
        belief,
        _linearization(),
        _anchor(),
        physical_prediction_xyz_m=belief.mean_xyz_m,
        physical_query_jacobian=np.eye(2),
        state_prior_covariance_m2=np.eye(2) * 1e-3,
    )

    assert materialization.observability_evaluable
    assert materialization.contact_anchor_artifact_id == _anchor().artifact_id
    assert materialization.observation_artifact_id == belief.artifact_id
    assert (
        np.trace(materialization.candidate_marginal_precision)
        > np.trace(materialization.reference_marginal_precision)
    )
    assert materialization.metadata[
        "contact_effective_samples_per_correlation_group"
    ] == pytest.approx(1.0)
    assert materialization.metadata["candidate_diagnostics"][
        "effective_anchor_information_mass"
    ] == pytest.approx(0.75)
    assert materialization.metadata["confirmation_payloads_opened"] is False

    with pytest.raises(ValueError, match="different cases"):
        materialize_deform360_calibration_factors(
            belief,
            _linearization(),
            _anchor(observation_case_id="another-case"),
            physical_prediction_xyz_m=belief.mean_xyz_m,
            physical_query_jacobian=np.eye(2),
        )

    with pytest.raises(ValueError, match="different causal cutoffs"):
        materialize_deform360_calibration_factors(
            belief,
            _linearization(),
            replace(_anchor(), causal_frame_stop=7),
            physical_prediction_xyz_m=belief.mean_xyz_m,
            physical_query_jacobian=np.eye(2),
        )


def test_published_materialization_matches_observability_batch_inputs(
    tmp_path: Path,
) -> None:
    belief = _attested_belief()
    anchor = _anchor()
    materialization = materialize_deform360_calibration_factors(
        belief,
        _linearization(),
        anchor,
        physical_prediction_xyz_m=belief.mean_xyz_m,
        physical_query_jacobian=np.eye(2),
        state_prior_covariance_m2=np.eye(2) * 1e-3,
    )
    output = tmp_path / "materialized"

    published = publish_deform360_calibration_factor_materialization(
        output,
        materialization,
        anchor,
    )

    assert published == output.absolute()
    assert sorted(path.name for path in output.iterdir()) == [
        "SHA256SUMS",
        "candidate-marginal-precision.npy",
        "contact-anchor.npz",
        "materialization.json",
        "physical-query-jacobian.npy",
        "reference-marginal-precision.npy",
    ]
    np.testing.assert_array_equal(
        np.load(output / "reference-marginal-precision.npy", allow_pickle=False),
        materialization.reference_marginal_precision,
    )
    assert (
        load_deform360_contact_anchor(output / "contact-anchor.npz").artifact_id
        == anchor.artifact_id
    )
    with pytest.raises(ValueError, match="already exists"):
        publish_deform360_calibration_factor_materialization(
            output,
            materialization,
            anchor,
        )

    locked_output = tmp_path / "locked-materialized"
    lock = tmp_path / ".locked-materialized.publish.lock"
    lock.write_text("owned elsewhere\n", encoding="ascii")
    with pytest.raises(FileExistsError):
        publish_deform360_calibration_factor_materialization(
            locked_output,
            materialization,
            anchor,
        )
    assert lock.read_text(encoding="ascii") == "owned elsewhere\n"
    assert not locked_output.exists()


def test_cli_ordinary_file_rejects_parent_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    array = source / "array.npy"
    np.save(array, np.asarray([1.0]))
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="path must not contain symlinks"):
        _CLI._ordinary_file(linked / "array.npy", name="test input")


def test_cli_materializes_contact_and_posterior_artifacts(tmp_path: Path) -> None:
    inputs = _contact_inputs()
    array_arguments = {
        "frame-ids": "frame_ids",
        "tactile-response": "tactile_response",
        "taxel-world-positions": "taxel_world_positions_m",
        "physical-patch-prediction": "physical_patch_prediction_m",
        "state-jacobian": "state_jacobian",
        "source-reliability": "source_reliability",
    }
    contact_arguments: list[str] = []
    for option, key in array_arguments.items():
        path = tmp_path / f"{option}.npy"
        np.save(path, np.asarray(inputs[key]))
        contact_arguments.extend((f"--{option}", str(path)))
    json_arguments = {
        "sensor-names": inputs["sensor_names"],
        "contact-episode-ids": inputs["contact_episode_ids"],
        "source-artifacts": inputs["source_artifacts"],
    }
    for option, value in json_arguments.items():
        path = tmp_path / f"{option}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        contact_arguments.extend((f"--{option}", str(path)))
    anchor_path = tmp_path / "contact-anchor.npz"
    anchor_command = [
        "contact-anchor",
        "--object-id",
        str(inputs["object_id"]),
        "--observation-case-id",
        str(inputs["observation_case_id"]),
        "--episode-id",
        str(inputs["episode_id"]),
        "--causal-frame-stop",
        str(inputs["causal_frame_stop"]),
        "--source-revision",
        str(inputs["source_revision"]),
        *contact_arguments,
        "--output",
        str(anchor_path),
    ]

    anchor_path.symlink_to(tmp_path / "missing-anchor.npz")
    assert _CLI.main(anchor_command) == 2
    anchor_path.unlink()

    assert _CLI.main(anchor_command) == 0

    belief = _attested_belief()
    linearization = _linearization()
    belief_path = tmp_path / "observation-belief.npz"
    linearization_path = tmp_path / "physical-linearization.npz"
    prediction_path = tmp_path / "physical-prediction.npy"
    query_path = tmp_path / "physical-query-jacobian.npy"
    prior_path = tmp_path / "state-prior-covariance.npy"
    save_observation_belief(belief_path, belief)
    save_physical_linearization(linearization_path, linearization)
    np.save(prediction_path, belief.mean_xyz_m)
    np.save(query_path, np.eye(2))
    np.save(prior_path, np.eye(2) * 1e-3)
    output = tmp_path / "materialized"

    assert (
        _CLI.main(
            [
                "posterior",
                "--observation-belief",
                str(belief_path),
                "--physical-linearization",
                str(linearization_path),
                "--physical-prediction",
                str(prediction_path),
                "--contact-anchor",
                str(anchor_path),
                "--physical-query-jacobian",
                str(query_path),
                "--state-prior-covariance",
                str(prior_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "materialization.json").is_file()
