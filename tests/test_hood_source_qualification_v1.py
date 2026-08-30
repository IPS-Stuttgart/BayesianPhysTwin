from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.hood_source_qualification_v1 import (
    CLAIM_BOUNDARY,
    REPLACEMENT_CLAIM_BOUNDARY,
    ROLLOUT_STEPS,
    assess_hood_source_replays_v1,
    build_hood_source_result_v1,
    consume_hood_source_attempt_v1,
    consume_hood_source_replacement_attempt_v2,
    load_hood_source_qualification_plan_v1,
    load_hood_source_qualification_replacement_plan_v2,
    save_hood_source_result_v1,
)


def test_registered_plan_is_bound_to_exact_local_sources() -> None:
    root = Path(__file__).parents[1]
    plan = load_hood_source_qualification_plan_v1(
        root / "protocols/execution_requests/hood_mesh_source_qualification_v1.json"
    )
    assert (
        plan.plan_id
        == "fcc5419f1e6dd5196bc39b581fe8fc71f5c064d09bf48e32375f264b185b70f8"
    )
    assert (
        plan.value["implementation"]["revision"]
        == "4b3e80b7243b91c9a744ff9fec4e41fbf8ad99a8"
    )
    assert (
        plan.value["information_boundary"]["certification_execution_authorized"]
        is False
    )
    revision = plan.value["implementation"]["revision"]
    for relative, expected in plan.implementation_source_files.items():
        blob = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"]
        )
        assert hashlib.sha256(blob).hexdigest() == expected


def test_public_terminal_receipt_is_hash_bound_and_nonclaiming() -> None:
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (root / "evidence/hood_mesh_source_qualification_terminal_v1.json").read_text()
    )
    receipt_id = receipt.pop("receipt_id")
    assert (
        receipt_id == "c6f0a658e5bbc969e3e5f1a602ff6dc692d6807efc9701f2695f43cfb2177e85"
    )
    assert content_id(receipt) == receipt_id
    assert receipt["qualification_result_produced"] is False
    assert receipt["source_competence_claim_authorized"] is False
    assert receipt["certification_execution_authorized"] is False
    assert receipt["retry_authorized"] is False


def _plan(tmp_path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "bayesian-phystwin.hood-mesh-source-qualification-plan",
        "schema_version": 1,
        "protocol_label": "test",
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation": {
            "repository": "IPS-Stuttgart/BayesianPhysTwin",
            "revision": "1" * 40,
            "source_archive_sha256": "2" * 64,
            "source_files": {"runner.py": "3" * 64},
        },
        "upstream": {
            "repository": "Dolorousrtur/HOOD",
            "revision": "9bc1076195979ac6c027fdd729c6e960cad62f2a",
            "git_archive_sha256": "4" * 64,
            "config_relative_path": "configs/aux/from_any_pose.yaml",
            "config_sha256": "5" * 64,
        },
        "public_source": {
            "archive_url": (
                "https://drive.google.com/file/d/"
                "1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view?usp=sharing"
            ),
            "archive_sha256": (
                "3b68239bea3f298f9456680e34cf0204c90512ba1e43233febb375a90038a2a4"
            ),
            "archive_byte_count": 604239517,
            "checkpoint_relative_path": "hood_data/trained_models/postcvpr.pth",
            "checkpoint_sha256": (
                "155d2dd25e54756fc04b0d27996ebca3446b2a59d3a715bb1fb73407753ce5ea"
            ),
            "mesh_sequence_relative_path": ("hood_data/fromanypose/mesh_sequence.pkl"),
            "mesh_sequence_sha256": (
                "1ad213334bf1bdb01bcc831f3c579afc063e832f0d7b99407a6f187d07b3059a"
            ),
            "garment_template_relative_path": "hood_data/fromanypose/tshirt.pkl",
            "garment_template_sha256": (
                "57717c231b80d5c6f9eeb26fc350aa6060eb1ad7584a1f4224cda02025d99454"
            ),
            "garment_obj_relative_path": "hood_data/fromanypose/tshirt.obj",
            "garment_obj_sha256": (
                "33d82264415a0bd30faf894ef0c8dcda4ce994d2e8682147d410990f95ae93bc"
            ),
        },
        "runtime": {
            "base_python_path": str(tmp_path / "base-python"),
            "base_python_sha256": "6" * 64,
            "base_freeze_sha256": "7" * 64,
            "python_overlay_path": str(tmp_path / "overlay"),
            "python_overlay_tree_sha256": "8" * 64,
            "cuda_visible_device": 0,
            "torch_version": "2.0.1+cu118",
            "torch_cuda_version": "11.8",
            "torch_geometric_version": "2.4.0",
            "pytorch3d_version": "0.7.4",
        },
        "execution": {
            "output_root": str(tmp_path / "output"),
            "attempt_ledger_path": str(tmp_path / "attempt.json"),
            "attempt_limit": 1,
            "random_seed": 20260830,
            "replay_count": 2,
            "rollout_steps": 30,
            "configuration_name": "aux/from_any_pose",
            "pose_sequence_type": "mesh",
            "source_execution_authorized": True,
        },
        "gates": {
            "maximum_repeat_rmse_m": 1e-7,
            "minimum_cloth_motion_m": 1e-5,
            "minimum_obstacle_motion_m": 1e-5,
            "maximum_absolute_coordinate_m": 10.0,
            "all_values_finite_required": True,
            "exact_frame_count_required": True,
            "topology_identity_required": True,
        },
        "information_boundary": {
            "public_hood_source_read": True,
            "fourddress_payload_read": False,
            "fourddress_participant_roster_read": False,
            "physical_outcomes_read": False,
            "certification_outcomes_read": False,
            "held_v8_read": False,
            "dlo4_or_dlo5_read": False,
            "certification_execution_authorized": False,
            "replacement_allowed": False,
        },
    }
    value["plan_id"] = content_id(value)
    return value


def _replacement_plan(tmp_path: Path) -> dict[str, Any]:
    value = _plan(tmp_path)
    value["schema"] = (
        "bayesian-phystwin.hood-mesh-source-qualification-replacement-plan"
    )
    value["schema_version"] = 2
    value["protocol_label"] = "test replacement"
    value["claim_boundary"] = REPLACEMENT_CLAIM_BOUNDARY
    value["execution"]["smpl_model_override"] = None
    value["information_boundary"] = {
        "public_hood_source_read": True,
        "parent_failure_metadata_read": True,
        "fourddress_payload_read": False,
        "fourddress_participant_roster_read": False,
        "physical_outcomes_read": False,
        "certification_outcomes_read": False,
        "held_v8_read": False,
        "dlo4_or_dlo5_read": False,
        "certification_execution_authorized": False,
        "replacement_execution_authorized": True,
        "further_replacement_allowed": False,
    }
    value["parent_failure"] = {
        "terminal_receipt_relative_path": (
            "evidence/hood_mesh_source_qualification_terminal_v1.json"
        ),
        "terminal_receipt_file_sha256": (
            "e2753bca945adfa7d664eb4255068460c8d9333267979c1c3561b523ea3c8be0"
        ),
        "terminal_receipt_id": (
            "c6f0a658e5bbc969e3e5f1a602ff6dc692d6807efc9701f2695f43cfb2177e85"
        ),
        "parent_plan_id": (
            "fcc5419f1e6dd5196bc39b581fe8fc71f5c064d09bf48e32375f264b185b70f8"
        ),
        "parent_plan_file_sha256": (
            "43b2f6cb52011833bc7f1eecd6e61c3ab6a431ea5fdafe7a8ca2ac7f35dd64a5"
        ),
        "attempt_ledger_path": (
            "/home/florianpfaff/source-only/hood-query-competence-v1/"
            "source-qualification-v1-attempt.json"
        ),
        "attempt_ledger_sha256": (
            "67779503f0a0eb2da195b60d1b8f64eae6609f45db2b6dfd20943f3bfaf32981"
        ),
        "failure_path": (
            "/home/florianpfaff/source-only/hood-query-competence-v1/"
            "source-qualification-v1-4b3e80b7/failure.json"
        ),
        "failure_sha256": (
            "97f3aed2f9261108b7ead948a5324a76a3e6bb1aec12de63941a9289992aa9b9"
        ),
        "terminal_stage": "pre-rollout-runtime-initialization",
        "source_data_decoded": False,
        "rollout_started": False,
        "scientific_score_available": False,
    }
    value["correction"] = {
        "reason": "source-independent-mesh-loader-configuration-defect",
        "scope": "set-dataloader.dataset.from_any_pose.smpl_model-null",
        "upstream_null_path_supported": True,
        "configuration_file_unchanged": True,
        "checkpoint_unchanged": True,
        "public_source_unchanged": True,
        "method_and_gates_unchanged": True,
    }
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    return value


def _write(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _valid_replays() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    time = np.linspace(0.0, 1.0, ROLLOUT_STEPS)[:, None, None]
    cloth = np.zeros((ROLLOUT_STEPS, 4, 3), dtype=np.float64)
    obstacle = np.zeros((ROLLOUT_STEPS, 5, 3), dtype=np.float64)
    cloth[:, :, 0:1] = 0.1 * time
    obstacle[:, :, 1:2] = 0.2 * time
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    return [cloth, cloth.copy()], [obstacle, obstacle.copy()], [faces, faces.copy()]


def test_plan_loads_and_attempt_is_write_once(tmp_path: Path) -> None:
    value = _plan(tmp_path)
    plan = load_hood_source_qualification_plan_v1(_write(tmp_path, value))
    assert plan.cuda_visible_device == 0
    assert plan.value["information_boundary"]["fourddress_payload_read"] is False

    ledger = consume_hood_source_attempt_v1(plan)
    assert ledger["attempt_index"] == 1
    with pytest.raises(FileExistsError):
        consume_hood_source_attempt_v1(plan)


def test_replacement_plan_binds_parent_and_consumes_distinct_attempt(
    tmp_path: Path,
) -> None:
    value = _replacement_plan(tmp_path)
    plan = load_hood_source_qualification_replacement_plan_v2(_write(tmp_path, value))
    assert plan.value["execution"]["smpl_model_override"] is None
    assert plan.value["information_boundary"]["further_replacement_allowed"] is False
    ledger = consume_hood_source_replacement_attempt_v2(plan)
    assert ledger["schema_version"] == 2
    assert ledger["parent_plan_id"] == value["parent_failure"]["parent_plan_id"]
    with pytest.raises(ValueError, match="schema-v1"):
        consume_hood_source_attempt_v1(plan)
    with pytest.raises(FileExistsError):
        consume_hood_source_replacement_attempt_v2(plan)


@pytest.mark.parametrize(
    ("section", "key", "replacement", "message"),
    [
        ("parent_failure", "failure_sha256", "0" * 64, "parent failure"),
        ("correction", "method_and_gates_unchanged", False, "correction"),
        (
            "information_boundary",
            "further_replacement_allowed",
            True,
            "information boundary",
        ),
    ],
)
def test_replacement_plan_fails_closed_on_custody_changes(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    value = _replacement_plan(tmp_path)
    value[section][key] = replacement
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    with pytest.raises(ValueError, match=message):
        load_hood_source_qualification_replacement_plan_v2(_write(tmp_path, value))


def test_runner_applies_only_registered_mesh_loader_correction(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/science/run_hood_mesh_source_qualification_v1.py"
    )
    runner = runpy.run_path(str(script))
    apply_correction = runner["_apply_registered_dataset_correction"]
    replacement = load_hood_source_qualification_replacement_plan_v2(
        _write(tmp_path, _replacement_plan(tmp_path))
    )
    dataset_config = SimpleNamespace(smpl_model="smpl/SMPL_FEMALE.pkl")
    apply_correction(replacement, dataset_config)
    assert dataset_config.smpl_model is None

    v1 = load_hood_source_qualification_plan_v1(_write(tmp_path, _plan(tmp_path)))
    dataset_config = SimpleNamespace(smpl_model="smpl/SMPL_FEMALE.pkl")
    apply_correction(v1, dataset_config)
    assert dataset_config.smpl_model == "smpl/SMPL_FEMALE.pkl"

    changed = SimpleNamespace(smpl_model="smpl/OTHER.pkl")
    with pytest.raises(ValueError, match="smpl_model field changed"):
        apply_correction(replacement, changed)


def test_runner_recomputes_parent_artifact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/science/run_hood_mesh_source_qualification_v1.py"
    )
    runner = runpy.run_path(str(script))
    verify_parent = runner["_verify_parent_failure"]
    plan = load_hood_source_qualification_replacement_plan_v2(
        _write(tmp_path, _replacement_plan(tmp_path))
    )
    parent = plan.value["parent_failure"]
    expected = {
        tmp_path / parent["terminal_receipt_relative_path"]: parent[
            "terminal_receipt_file_sha256"
        ],
        Path(parent["attempt_ledger_path"]): parent["attempt_ledger_sha256"],
        Path(parent["failure_path"]): parent["failure_sha256"],
    }
    observed = dict(expected)
    monkeypatch.setitem(
        verify_parent.__globals__,
        "file_sha256",
        lambda path: observed[Path(path)],
    )
    verify_parent(plan, tmp_path)
    observed[Path(parent["failure_path"])] = "0" * 64
    with pytest.raises(ValueError, match="retained parent artifact"):
        verify_parent(plan, tmp_path)


def test_plan_rejects_resealed_boundary_or_method_changes(tmp_path: Path) -> None:
    value = _plan(tmp_path)
    value["information_boundary"]["fourddress_payload_read"] = True
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    with pytest.raises(ValueError, match="information boundary changed"):
        load_hood_source_qualification_plan_v1(_write(tmp_path, value))

    value = _plan(tmp_path)
    value["execution"]["rollout_steps"] = 29
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    with pytest.raises(ValueError, match="execution policy changed"):
        load_hood_source_qualification_plan_v1(_write(tmp_path, value))


@pytest.mark.parametrize("device", [True, False, "0", "1", 0.0, -1, 2])
def test_plan_rejects_invalid_gpu_values(tmp_path: Path, device: object) -> None:
    value = _plan(tmp_path)
    value["runtime"]["cuda_visible_device"] = device
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    with pytest.raises(ValueError, match="cuda_visible_device"):
        load_hood_source_qualification_plan_v1(_write(tmp_path, value))


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("execution", "attempt_limit", True),
        ("execution", "source_execution_authorized", 1),
        ("gates", "all_values_finite_required", 1),
        ("information_boundary", "fourddress_payload_read", 0),
    ],
)
def test_plan_requires_exact_json_types(
    tmp_path: Path, section: str, key: str, replacement: object
) -> None:
    value = _plan(tmp_path)
    value[section][key] = replacement
    value.pop("plan_id")
    value["plan_id"] = content_id(value)
    with pytest.raises(ValueError):
        load_hood_source_qualification_plan_v1(_write(tmp_path, value))


def test_repeatable_moving_replays_pass() -> None:
    cloth, obstacles, faces = _valid_replays()
    assessment = assess_hood_source_replays_v1(cloth, obstacles, faces, faces)
    assert assessment.passed is True
    assert all(assessment.decisions.values())
    assert assessment.metrics["rollout_steps"] == ROLLOUT_STEPS


def test_repeatability_failure_is_scientific_not_structural() -> None:
    cloth, obstacles, faces = _valid_replays()
    cloth[1][0, 0, 0] += 1e-3
    assessment = assess_hood_source_replays_v1(cloth, obstacles, faces, faces)
    assert assessment.passed is False
    assert assessment.decisions["repeatability"] is False


def test_obstacle_replay_change_is_rejected() -> None:
    cloth, obstacles, faces = _valid_replays()
    obstacles[1][0, 0, 0] += 0.1
    assessment = assess_hood_source_replays_v1(cloth, obstacles, faces, faces)
    assert assessment.passed is False
    assert assessment.decisions["repeatability"] is False


@pytest.mark.parametrize("failure", ["cloth", "obstacle", "topology", "finite"])
def test_gate_failures_are_preserved(failure: str) -> None:
    cloth, obstacles, faces = _valid_replays()
    if failure == "cloth":
        cloth = [np.zeros_like(cloth[0]), np.zeros_like(cloth[1])]
    elif failure == "obstacle":
        obstacles = [np.zeros_like(obstacles[0]), np.zeros_like(obstacles[1])]
    elif failure == "topology":
        faces[1][0, 0] = 3
    else:
        cloth[0][0, 0, 0] = np.nan
        cloth[1][0, 0, 0] = np.nan
    assessment = assess_hood_source_replays_v1(cloth, obstacles, faces, faces)
    assert assessment.passed is False


def test_shape_and_count_substitutions_are_rejected() -> None:
    cloth, obstacles, faces = _valid_replays()
    with pytest.raises(ValueError, match="exactly 2"):
        assess_hood_source_replays_v1(cloth[:1], obstacles, faces, faces)
    with pytest.raises(ValueError, match="shape"):
        assess_hood_source_replays_v1(
            [cloth[0][..., :2], cloth[1][..., :2]],
            obstacles,
            faces,
            faces,
        )


@pytest.mark.parametrize("invalid", ["empty", "fractional", "negative", "out_of_range"])
def test_invalid_meshes_are_rejected(invalid: str) -> None:
    cloth, obstacles, faces = _valid_replays()
    if invalid == "empty":
        cloth = [value[:, :0] for value in cloth]
    elif invalid == "fractional":
        faces = [value.astype(float) + 0.5 for value in faces]
    elif invalid == "negative":
        faces[0][0, 0] = -1
    else:
        faces[0][0, 0] = 4
    with pytest.raises(ValueError):
        assess_hood_source_replays_v1(cloth, obstacles, faces, faces)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e300])
def test_numerical_failure_result_remains_sealable(
    tmp_path: Path, invalid: float
) -> None:
    plan = load_hood_source_qualification_plan_v1(_write(tmp_path, _plan(tmp_path)))
    cloth, obstacles, faces = _valid_replays()
    cloth[1][0, 0, 0] = invalid
    assessment = assess_hood_source_replays_v1(cloth, obstacles, faces, faces)
    assert assessment.passed is False
    result = build_hood_source_result_v1(
        plan=plan,
        assessment=assessment,
        replay_archive_sha256="9" * 64,
        elapsed_seconds=1.0,
    )
    path = save_hood_source_result_v1(result, tmp_path / "result.json")
    saved = json.loads(path.read_text())
    result_id = saved.pop("result_id")
    assert result_id == content_id(saved)
    assert saved["passed"] is False
    with pytest.raises(FileExistsError):
        save_hood_source_result_v1(result, path)


def test_runner_requires_clean_exact_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/science/run_hood_mesh_source_qualification_v1.py"
    )
    runner = runpy.run_path(str(script))
    verify = runner["_verify_checkout"]
    revision = "1" * 40
    digest = "2" * 64
    state = {"revision": revision, "status": "", "archive": digest}

    def fake_output(args: list[str], *, text: bool) -> str:
        assert text
        return state["revision"] if "rev-parse" in args else state["status"]

    monkeypatch.setattr(runner["subprocess"], "check_output", fake_output)
    monkeypatch.setitem(
        verify.__globals__, "_git_archive_digest", lambda root: state["archive"]
    )
    verify(tmp_path, revision, digest)
    for key, changed in (
        ("revision", "3" * 40),
        ("status", " M runner.py\n"),
        ("archive", "4" * 64),
    ):
        original = state[key]
        state[key] = changed
        with pytest.raises(ValueError):
            verify(tmp_path, revision, digest)
        state[key] = original
