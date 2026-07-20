from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from causal4d_public import deform360_object_sam2

from deform360_held_test_helpers import (
    bound_file,
    write_robot_kinematics_fixture,
    write_robot_metadata_fixture,
)

from bayesian_phystwin import deform360_held_outcome_reconstruction as reconstruction


CASE_NAME = "083-blanket-cloth-ep0000"
CAMERAS = ("camera_0", "camera_1")


def _write(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _request(tmp_path: Path, output_dir: Path) -> reconstruction.ReconstructionRequest:
    bundle = _write(tmp_path / "sealed" / "frame-zero.npz")
    manifest = _write(tmp_path / "sealed" / "frame-zero.json", b"{}\n")
    online_seal = _write(tmp_path / "sealed" / "online-seal.json", b"{}\n")
    rgb = np.zeros((len(CAMERAS), 3, 4, 3), dtype=np.uint8)
    masks = np.zeros(rgb.shape[:3], dtype=bool)
    masks[:, 1, 1:3] = True
    arrays = {
        "intrinsics": np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0),
        "camera_to_world": np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
        "rgb_frame0": rgb,
        "mask_frame0": masks,
    }
    aligned_episode = tmp_path / "aligned" / "083-blanket-cloth" / "episode_0000"
    raw_robot, _selected_robot, action_alignment = write_robot_kinematics_fixture(
        aligned_episode / "robot",
        selected_start=62,
    )
    robot_metadata = write_robot_metadata_fixture(
        aligned_episode / "robot" / "robot.meta.json",
        source_frame_count=150,
        cameras=CAMERAS,
    )
    contract = reconstruction._copy_contract()
    return reconstruction.ReconstructionRequest(
        case_name=CASE_NAME,
        object_id="083-blanket-cloth",
        episode_id=0,
        role="calibration",
        cohort_barrier_sha256="b" * 64,
        aligned_episode_dir=aligned_episode,
        output_dir=output_dir,
        source_frame_start=62,
        source_frame_stop=143,
        camera_names=CAMERAS,
        frame_zero_arrays=arrays,
        frame_zero_manifest={
            "bundle": reconstruction._bound_file(bundle),
            "action_inputs": {
                "robot_trajectory": bound_file(raw_robot),
                "robot_metadata": bound_file(robot_metadata),
            },
            "action_alignment": action_alignment,
            "config": {
                "action_candidate_first_frame": 8,
                "action_candidate_stride_frames": 6,
                "action_window_length_frames": 81,
                "prediction_frame_count": 76,
            },
        },
        frame_zero_manifest_path=manifest,
        online_seal_path=online_seal,
        contract=contract,
        immutable_bindings={},
    )


class _SyntheticOfficialBackend:
    def __init__(self) -> None:
        self.calls = 0

    def build(
        self, request: reconstruction.ReconstructionRequest
    ) -> reconstruction.ReconstructionBackendResult:
        self.calls += 1
        episode = request.output_dir / "staged-aligned" / "episode_0000"
        source_record = request.frame_zero_manifest["action_inputs"]["robot_trajectory"]
        selected_record = request.frame_zero_manifest["action_alignment"][
            "selected_robot_kinematics_bundle"
        ]
        source_state = reconstruction.load_robot_kinematics_archive(
            source_record["path"]
        )
        selected_state = reconstruction.load_robot_kinematics_archive(
            selected_record["path"], expected_frame_count=reconstruction.FRAME_COUNT
        )
        staged_state = reconstruction.slice_robot_kinematics(
            source_state,
            start_frame=request.source_frame_start,
            frame_count=reconstruction.TRACKING_CONTEXT_FRAME_COUNT,
        )
        robot_path = episode / "robot" / "robot.npz"
        robot_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(robot_path, **staged_state.archive_arrays())
        source_arrays = source_state.archive_arrays()
        selected_arrays = selected_state.archive_arrays()
        staged_arrays = staged_state.archive_arrays()
        selection = request.frame_zero_manifest["action_alignment"]["selection_audit"]
        exact_slice = request.frame_zero_manifest["action_alignment"][
            "selected_bundle_exact_slice_audit"
        ]
        robot_evidence = {
            "policy_id": reconstruction.ROBOT_KINEMATICS_WINDOW_POLICY_ID,
            "contract_sha256": (reconstruction.ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256),
            "trajectory_semantics": reconstruction.ROBOT_KINEMATICS_WINDOW_CONTRACT[
                "trajectory_semantics"
            ],
            "selection_audit": selection,
            "selected_bundle_exact_slice_audit": exact_slice,
            "temporal_fields_sliced_exactly_81": [
                "actions",
                "T_worlds",
                "openings",
            ],
            "scalar_fields_copied_unchanged": ["format_version", "bimanual"],
            "all_five_fields_first_76_equal_selected_bundle": True,
            "source_array_sha256": {
                name: reconstruction.robot_array_sha256(value)
                for name, value in sorted(source_arrays.items())
            },
            "staged_array_sha256": {
                name: reconstruction.robot_array_sha256(value)
                for name, value in sorted(staged_arrays.items())
            },
            "selected_array_sha256": {
                name: reconstruction.robot_array_sha256(value)
                for name, value in sorted(selected_arrays.items())
            },
            "commanded_control_or_delta_action_used": False,
        }
        robot_source_binding = reconstruction._bound_file(source_record["path"])
        selected_binding = reconstruction._bound_file(selected_record["path"])
        reconstruction_files = {
            str(frame): _write(episode / "splats" / f"splat_{frame}.ply")
            for frame in range(reconstruction.TRACKING_CONTEXT_FRAME_COUNT)
        }
        depth_files = {
            camera: _write(episode / camera / "rendered_depth.h5") for camera in CAMERAS
        }
        tracking_dirs: dict[str, Path] = {}
        mask_files: dict[str, Path] = {}
        for camera in CAMERAS:
            tracking = episode / camera / "tracking"
            _write(tracking / "vel.h5")
            _write(tracking / "visibility.h5")
            tracking_dirs[camera] = tracking
            mask_files[camera] = _write(episode / camera / "mask_refined.h5")
        pcd_files = {
            f"{frame:06d}.npz": _write(episode / "pcd_clean" / f"{frame:06d}.npz")
            for frame in range(reconstruction.FRAME_COUNT)
        }
        runtime_file = _write(episode / "runtime-bound-file.bin")
        bound_runtime = reconstruction._bound_file(runtime_file)
        masks = np.asarray(request.frame_zero_arrays["mask_frame0"])
        camera_masks: dict[str, Any] = {}
        for index, camera in enumerate(CAMERAS):
            mask_sha = reconstruction._sha256_array(masks[index])
            camera_masks[camera] = {
                "mask_archive": reconstruction._bound_file(mask_files[camera]),
                "frame_count": reconstruction.TRACKING_CONTEXT_FRAME_COUNT,
                "sealed_frame_zero_mask_sha256": mask_sha,
                "sam2_raw_frame_zero_mask_sha256": mask_sha,
                "archived_frame_zero_mask_sha256": mask_sha,
                "sam2_future_mask_stack_sha256": mask_sha,
                "archive_future_mask_stack_sha256": mask_sha,
                "initialization": {
                    "case_name": request.case_name,
                    "cohort_barrier_sha256": request.cohort_barrier_sha256,
                    "frame_zero_manifest_sha256": reconstruction._sha256_file(
                        request.frame_zero_manifest_path
                    ),
                    "sealed_mask_sha256": mask_sha,
                    "automatic_initial_mask_selection": False,
                },
            }
        points = np.zeros((reconstruction.FRAME_COUNT, 4, 3), dtype=np.float32)
        points[:, :, 0] = np.arange(4, dtype=np.float32)[None] * np.float32(0.01)
        points[:, :, 1] = np.arange(reconstruction.FRAME_COUNT, dtype=np.float32)[
            :, None
        ] * np.float32(0.001)
        colors = np.full_like(points, np.float32(0.5))
        valid = np.ones(points.shape[:2], dtype=bool)
        audit = {
            "stage_ids": list(reconstruction.STAGE_IDS),
            "contract_sha256": reconstruction._contract_sha256(request.contract),
            "tracking_context_raw_frame_range_half_open": [62, 143],
            "tracking_context_frame_count": 81,
            "prediction_output_frame_range_half_open": [0, 76],
            "tracking_tail_frame_range_half_open": [76, 81],
            "frame_zero_anchor": {
                "bundle": dict(request.frame_zero_manifest["bundle"]),
                "intrinsics_sha256": reconstruction._sha256_array(
                    request.frame_zero_arrays["intrinsics"]
                ),
                "camera_to_world_sha256": reconstruction._sha256_array(
                    request.frame_zero_arrays["camera_to_world"]
                ),
                "rgb_frame0_sha256": reconstruction._sha256_array(
                    request.frame_zero_arrays["rgb_frame0"]
                ),
                "mask_frame0_sha256": reconstruction._sha256_array(masks),
            },
            "staging": {
                "source_inputs": {
                    "robot_kinematics": robot_source_binding,
                    "robot_trajectory": robot_source_binding,
                    "selected_prediction_robot_kinematics": selected_binding,
                    "selected_prediction_action": selected_binding,
                },
                "staged_outputs": {
                    "robot": reconstruction._bound_file(robot_path),
                },
                "robot_kinematics": robot_evidence,
                "source_frame_range_half_open": [62, 143],
                "tracking_context_frame_count": 81,
                "prediction_frame_range_half_open": [0, 76],
                "tracking_tail_frame_range_half_open": [76, 81],
                "video_staging": dict(reconstruction.VIDEO_STAGING_PARAMETERS),
                "tactile_paths_read_or_copied": [],
            },
            "mask_propagation": {
                "stage_id": reconstruction.STAGE_IDS[1],
                "sam2_git": {},
                "sam2_checkpoint": bound_runtime,
                "sam2_model_config": bound_runtime,
                "camera_masks": camera_masks,
                "sam2_initialization_mask_source": "sealed mask_frame0 only",
                "archive_frame_zero_source": "sealed mask_frame0",
                "archive_future_source": "unmodified thresholded SAM2 output",
                "frame_zero_archive_substitution_after_complete_propagation": True,
                "target_dependent_mask_selection_or_tuning": False,
            },
            "official_stages": {
                "runtime": {},
                "cotracker_runtime": {},
                "strict_hull_parameters": dict(reconstruction.STRICT_HULL_PARAMETERS),
                "depth_parameters": dict(reconstruction.DEPTH_PARAMETERS),
                "tracking_parameters": dict(reconstruction.TRACKING_PARAMETERS),
                "pcd_parameters": dict(reconstruction.PCD_PARAMETERS),
                "output_bindings": {
                    "reconstruction": {
                        frame: reconstruction._bound_file(path)
                        for frame, path in reconstruction_files.items()
                    },
                    "depth": {
                        camera: reconstruction._bound_file(path)
                        for camera, path in depth_files.items()
                    },
                    "tracking": {
                        camera: reconstruction._tree_binding(path)
                        for camera, path in tracking_dirs.items()
                    },
                    "pcd": {
                        name: reconstruction._bound_file(path)
                        for name, path in pcd_files.items()
                    },
                },
                "staged_episode_tree": reconstruction._tree_binding(episode),
            },
            "ffmpeg_runtime": {
                "executable": bound_runtime,
                "version_stdout_sha256": "1" * 64,
                "version_stderr_sha256": "2" * 64,
            },
            "tactile_read": False,
            "target_dependent_parameter_selection_or_tuning": False,
            "runtime_seconds": 1.0,
        }
        return reconstruction.ReconstructionBackendResult(
            object_points=points,
            object_colors=colors,
            object_visibilities=valid,
            object_motions_valid=valid.copy(),
            audit=audit,
        )


def _run_immediately(_permit: object, **kwargs: Any) -> Any:
    return kwargs["callback"]()


def test_create_read_and_resume_are_write_once_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    request = _request(tmp_path, output)
    backend = _SyntheticOfficialBackend()
    permit = SimpleNamespace()
    monkeypatch.setattr(reconstruction, "run_outcome_operation", _run_immediately)
    monkeypatch.setattr(
        reconstruction,
        "_load_sealed_request",
        lambda *_args, **_kwargs: request,
    )

    create = reconstruction.make_official_reconstruction_target_operation(
        permit,
        case_name=CASE_NAME,
        aligned_episode_dir=tmp_path / "does-not-need-to-be-opened-by-factory",
        output_dir=output,
        backend=backend,
    )
    assert not output.exists()
    target = create.callback()

    assert backend.calls == 1
    assert target.object_points.shape == (76, 4, 3)
    assert (output / "official_target.npz").is_file()
    assert (output / "held_outcome.json").is_file()
    assert target.provenance["cohort_barrier_sha256"] == "b" * 64
    with pytest.raises(FileExistsError):
        create.callback()

    read = reconstruction.make_official_reconstruction_read_target_operation(
        permit,
        case_name=CASE_NAME,
        output_dir=output,
    )
    resumed = read.callback()
    np.testing.assert_array_equal(resumed.object_points, target.object_points)
    assert backend.calls == 1

    planned = reconstruction.plan_official_reconstruction_target_operation(
        permit,
        case_name=CASE_NAME,
        aligned_episode_dir=tmp_path / "unopened-aligned",
        output_dir=output,
        backend=backend,
    )
    assert planned.operation == "read"
    assert backend.calls == 1

    target_path = output / "official_target.npz"
    target_path.chmod(0o644)
    target_path.write_bytes(target_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="target file binding changed"):
        read.callback()


def test_resume_planner_returns_create_only_for_absent_and_rejects_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permit = SimpleNamespace()
    backend = _SyntheticOfficialBackend()
    monkeypatch.setattr(reconstruction, "run_outcome_operation", _run_immediately)
    absent = tmp_path / "absent"
    operation = reconstruction.plan_official_reconstruction_target_operation(
        permit,
        case_name=CASE_NAME,
        aligned_episode_dir=tmp_path / "aligned",
        output_dir=absent,
        backend=backend,
    )
    assert operation.operation == "create"
    assert not absent.exists()
    assert backend.calls == 0

    partial = tmp_path / "partial"
    partial.mkdir()
    _write(partial / "official_target.npz")
    request = _request(tmp_path / "partial-request", partial)
    monkeypatch.setattr(
        reconstruction,
        "_load_sealed_request",
        lambda *_args, **_kwargs: request,
    )
    with pytest.raises(ValueError, match="partial or invalid"):
        reconstruction.plan_official_reconstruction_target_operation(
            permit,
            case_name=CASE_NAME,
            aligned_episode_dir=tmp_path / "aligned",
            output_dir=partial,
            backend=backend,
        )


def test_operator_paths_are_not_touched_before_live_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BarrierStopped(RuntimeError):
        pass

    class Backend:
        def build(self, _request: object) -> reconstruction.ReconstructionBackendResult:
            raise AssertionError("backend crossed the barrier")

    def stop_before_callback(_permit: object, **_kwargs: Any) -> Any:
        raise BarrierStopped

    monkeypatch.setattr(reconstruction, "run_outcome_operation", stop_before_callback)
    output = tmp_path / "must-remain-absent"
    operation = reconstruction.make_official_reconstruction_target_operation(
        SimpleNamespace(),
        case_name=CASE_NAME,
        aligned_episode_dir=tmp_path / "missing-private-outcome",
        output_dir=output,
        backend=Backend(),
    )
    assert not output.exists()
    with pytest.raises(BarrierStopped):
        operation.callback()
    assert not output.exists()


def test_contract_digest_and_adapter_source_are_both_lock_bound() -> None:
    contract = reconstruction._copy_contract()
    reconstruction._validate_contract_semantics(contract)
    bindings = {
        "outcome_reconstruction_contract": reconstruction._contract_sha256(contract),
        "held_outcome_reconstruction_adapter_source": reconstruction._sha256_file(
            reconstruction.__file__
        ),
    }
    assert (
        reconstruction._validate_locked_adapter_contract(
            {"immutable_bindings": bindings}, contract
        )
        == bindings
    )
    bindings["outcome_reconstruction_contract"] = "0" * 64
    with pytest.raises(ValueError, match="another outcome reconstruction contract"):
        reconstruction._validate_locked_adapter_contract(
            {"immutable_bindings": bindings}, contract
        )


def test_git_runtime_rejects_modified_tracked_bytes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    tracked = _write(repository / "runtime.py", b"released\n")
    subprocess.run(["git", "-C", str(repository), "add", "runtime.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    assert reconstruction._git_runtime_binding(repository)["revision"]
    tracked.write_bytes(b"dirty runtime\n")
    with pytest.raises(ValueError, match="modified tracked files"):
        reconstruction._git_runtime_binding(repository)


def test_cotracker_import_must_resolve_to_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "co-tracker"
    expected_module = _write(repository / "cotracker" / "predictor.py")
    checkpoint = _write(tmp_path / "scaled_offline.pth")
    request = _request(tmp_path / "request", tmp_path / "unused")
    request = reconstruction.ReconstructionRequest(
        **{
            **request.__dict__,
            "immutable_bindings": {
                "cotracker_checkpoint": reconstruction.COTRACKER_CHECKPOINT_SHA256
            },
        }
    )
    backend = reconstruction.PinnedOfficialPipelineBackend(
        deform360_repo="unused",
        sam2_repository="unused",
        sam2_checkpoint="unused",
        cotracker_repo=str(repository),
        cotracker_checkpoint=str(checkpoint),
    )
    monkeypatch.setattr(
        reconstruction,
        "_validate_git_runtime_binding",
        lambda *_args, **_kwargs: {"revision": reconstruction.COTRACKER_COMMIT},
    )
    original_sha = reconstruction._sha256_file
    monkeypatch.setattr(
        reconstruction,
        "_sha256_file",
        lambda path: (
            reconstruction.COTRACKER_CHECKPOINT_SHA256
            if Path(path).resolve() == checkpoint.resolve()
            else original_sha(path)
        ),
    )
    monkeypatch.setattr(reconstruction.sys, "path", list(reconstruction.sys.path))
    wrong_module = _write(tmp_path / "other" / "cotracker" / "predictor.py")
    monkeypatch.setattr(
        reconstruction.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(wrong_module)),
    )
    with pytest.raises(ValueError, match="another repository"):
        reconstruction._verify_cotracker_runtime(request, backend)

    monkeypatch.setattr(
        reconstruction.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(expected_module)),
    )
    observed_checkpoint, _, audit = reconstruction._verify_cotracker_runtime(
        request, backend
    )
    assert observed_checkpoint == checkpoint.resolve()
    assert audit["predictor_module"]["path"] == str(expected_module.resolve())


def test_sam2_hydra_id_and_bound_repository_path_are_distinct(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "sam2"
    config = _write(
        repository / reconstruction.SAM2_MODEL_CONFIG_REPOSITORY_PATH,
        b"model: small\n",
    )
    digest = hashlib.sha256(config.read_bytes()).hexdigest()

    assert reconstruction.SAM2_MODEL_CONFIG == "configs/sam2.1/sam2.1_hiera_s.yaml"
    assert reconstruction.SAM2_MODEL_CONFIG_REPOSITORY_PATH.startswith("sam2/")
    assert (
        reconstruction._validate_sam2_model_config(
            repository, {"sam2_model_config": digest}
        )
        == config.resolve()
    )


def test_sealed_frame_zero_replaces_only_sam2_frame_zero_after_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write(tmp_path / "sam2" / "checkpoint.pt")
    model_config = _write(tmp_path / "sam2" / "model.yaml")
    request = _request(tmp_path / "request", tmp_path / "out")
    request = reconstruction.ReconstructionRequest(
        **{
            **request.__dict__,
            "immutable_bindings": {
                "sam2_checkpoint": reconstruction.SAM2_CHECKPOINT_SHA256
            },
        }
    )
    staged_root = tmp_path / "staged"
    episode = staged_root / f"episode_{reconstruction.STAGED_EPISODE_ID:04d}"
    for camera in request.camera_names:
        _write(episode / camera / "undistorted.mp4", b"video")

    raw_outputs: dict[str, np.ndarray] = {}

    class FakePredictor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        def segment_from_initial_mask(
            self,
            video: Path,
            initial_mask: np.ndarray,
            *,
            initialization: object,
        ) -> object:
            del initialization
            values: list[np.ndarray] = []
            for frame in range(reconstruction.TRACKING_CONTEXT_FRAME_COUNT):
                mask = np.full_like(initial_mask, bool(frame % 2), dtype=bool)
                values.append(mask)
                yield frame, mask
            raw_outputs[video.parent.name] = np.stack(values, axis=0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        deform360_object_sam2,
        "DeformableObjectSam2VideoPredictor",
        FakePredictor,
    )
    monkeypatch.setattr(
        reconstruction,
        "_validate_git_runtime_binding",
        lambda *_args, **_kwargs: {"revision": reconstruction.SAM2_COMMIT},
    )
    monkeypatch.setattr(
        reconstruction,
        "_validate_sam2_model_config",
        lambda *_args, **_kwargs: model_config,
    )
    original_sha256 = reconstruction._sha256_file
    monkeypatch.setattr(
        reconstruction,
        "_sha256_file",
        lambda path: (
            reconstruction.SAM2_CHECKPOINT_SHA256
            if Path(path).resolve() == checkpoint.resolve()
            else original_sha256(path)
        ),
    )
    written: dict[str, np.ndarray] = {}

    def capture_archive(path: Path, values: object) -> None:
        written[path.parent.name] = np.stack(list(values), axis=0)
        _write(path, written[path.parent.name].tobytes())

    monkeypatch.setattr(reconstruction, "_write_mask_h5", capture_archive)
    backend = reconstruction.PinnedOfficialPipelineBackend(
        deform360_repo="unused",
        sam2_repository=str(tmp_path / "sam2"),
        sam2_checkpoint=str(checkpoint),
        cotracker_repo="unused",
        cotracker_checkpoint="unused",
    )

    audit = reconstruction._propagate_sealed_masks(request, backend, staged_root)

    sealed = np.asarray(request.frame_zero_arrays["mask_frame0"], dtype=bool)
    for index, camera in enumerate(request.camera_names):
        assert not np.array_equal(raw_outputs[camera][0], sealed[index])
        assert np.array_equal(written[camera][0], sealed[index])
        assert np.array_equal(written[camera][1:], raw_outputs[camera][1:])
        record = audit["camera_masks"][camera]
        assert (
            record["sealed_frame_zero_mask_sha256"]
            == (record["archived_frame_zero_mask_sha256"])
        )
        assert (
            record["sam2_raw_frame_zero_mask_sha256"]
            != (record["archived_frame_zero_mask_sha256"])
        )
        assert (
            record["sam2_future_mask_stack_sha256"]
            == (record["archive_future_mask_stack_sha256"])
        )
    assert audit == {
        **audit,
        "sam2_initialization_mask_source": "sealed mask_frame0 only",
        "archive_frame_zero_source": "sealed mask_frame0",
        "archive_future_source": "unmodified thresholded SAM2 output",
        "frame_zero_archive_substitution_after_complete_propagation": True,
        "target_dependent_mask_selection_or_tuning": False,
    }


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("frame-order", "SAM2 propagation is incomplete"),
        ("frame-count", "SAM2 propagation is incomplete"),
        ("mask-shape", "SAM2 propagated mask shape changed"),
    ],
)
def test_sam2_propagation_fails_closed_on_malformed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    message: str,
) -> None:
    checkpoint = _write(tmp_path / "sam2" / "checkpoint.pt")
    model_config = _write(tmp_path / "sam2" / "model.yaml")
    request = _request(tmp_path / "request", tmp_path / "out")
    request = reconstruction.ReconstructionRequest(
        **{
            **request.__dict__,
            "immutable_bindings": {
                "sam2_checkpoint": reconstruction.SAM2_CHECKPOINT_SHA256
            },
        }
    )
    staged_root = tmp_path / "staged"
    episode = staged_root / f"episode_{reconstruction.STAGED_EPISODE_ID:04d}"
    for camera in request.camera_names:
        _write(episode / camera / "undistorted.mp4", b"video")

    predictors: list[FakeMalformedPredictor] = []

    class FakeMalformedPredictor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False
            predictors.append(self)

        def segment_from_initial_mask(
            self,
            video: Path,
            initial_mask: np.ndarray,
            *,
            initialization: object,
        ) -> object:
            del video, initialization
            frame_indices = list(range(reconstruction.TRACKING_CONTEXT_FRAME_COUNT))
            if malformation == "frame-order":
                frame_indices[1], frame_indices[2] = (
                    frame_indices[2],
                    frame_indices[1],
                )
            elif malformation == "frame-count":
                frame_indices.pop()
            for frame in frame_indices:
                shape = initial_mask.shape
                if malformation == "mask-shape" and frame == 40:
                    shape = (shape[0] + 1, shape[1])
                yield frame, np.zeros(shape, dtype=bool)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        deform360_object_sam2,
        "DeformableObjectSam2VideoPredictor",
        FakeMalformedPredictor,
    )
    monkeypatch.setattr(
        reconstruction,
        "_validate_git_runtime_binding",
        lambda *_args, **_kwargs: {"revision": reconstruction.SAM2_COMMIT},
    )
    monkeypatch.setattr(
        reconstruction,
        "_validate_sam2_model_config",
        lambda *_args, **_kwargs: model_config,
    )
    original_sha256 = reconstruction._sha256_file
    monkeypatch.setattr(
        reconstruction,
        "_sha256_file",
        lambda path: (
            reconstruction.SAM2_CHECKPOINT_SHA256
            if Path(path).resolve() == checkpoint.resolve()
            else original_sha256(path)
        ),
    )
    monkeypatch.setattr(
        reconstruction,
        "_write_mask_h5",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed SAM2 output reached the mask archive"
        ),
    )
    backend = reconstruction.PinnedOfficialPipelineBackend(
        deform360_repo="unused",
        sam2_repository=str(tmp_path / "sam2"),
        sam2_checkpoint=str(checkpoint),
        cotracker_repo="unused",
        cotracker_checkpoint="unused",
    )

    with pytest.raises(ValueError, match=message):
        reconstruction._propagate_sealed_masks(request, backend, staged_root)

    assert len(predictors) == 1
    assert predictors[0].closed is True
    assert not any(episode.glob("*/mask_refined.h5"))


@pytest.mark.parametrize(
    "audit_field",
    [
        "archived_frame_zero_mask_sha256",
        "archive_future_mask_stack_sha256",
    ],
)
def test_backend_rejects_false_sam2_archive_hash_claims(
    tmp_path: Path, audit_field: str
) -> None:
    request = _request(tmp_path / "request", tmp_path / "out")
    result = _SyntheticOfficialBackend().build(request)
    reconstruction._validate_backend_result(request, result)

    audit = copy.deepcopy(result.audit)
    record = audit["mask_propagation"]["camera_masks"][CAMERAS[0]]
    original = record[audit_field]
    record[audit_field] = ("0" if original[0] != "0" else "1") + original[1:]
    tampered = reconstruction.ReconstructionBackendResult(
        object_points=result.object_points,
        object_colors=result.object_colors,
        object_visibilities=result.object_visibilities,
        object_motions_valid=result.object_motions_valid,
        audit=audit,
    )

    with pytest.raises(ValueError, match="backend changed the sealed mask anchor"):
        reconstruction._validate_backend_result(request, tampered)


def _prepare_staging_inputs(request: reconstruction.ReconstructionRequest) -> None:
    assert request.aligned_episode_dir is not None
    for camera in request.camera_names:
        camera_dir = request.aligned_episode_dir / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        _write(camera_dir / "undistorted.mp4", b"video")
        (camera_dir / "aligned_timestamps.txt").write_text(
            "\n".join(str(frame) for frame in range(150)) + "\n",
            encoding="utf-8",
        )


def test_action_window_staging_slices_all_fields_and_preserves_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, tmp_path / "out")
    _prepare_staging_inputs(request)
    monkeypatch.setattr(
        reconstruction,
        "_decode_video_frame",
        lambda _path, _frame: np.zeros((3, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(reconstruction, "_video_frame_count", lambda _path: 81)

    def fake_trim(
        _ffmpeg: str,
        _source: Path,
        output: Path,
        *,
        start: int,
        frame_count: int,
    ) -> None:
        assert start == 62
        assert frame_count == 81
        _write(output, b"lossless")

    monkeypatch.setattr(reconstruction, "_trim_lossless_video", fake_trim)
    _root, staging = reconstruction._stage_action_window(request, Path("/bin/true"))
    robot_path = Path(staging["staged_outputs"]["robot"]["path"])
    with np.load(robot_path, allow_pickle=False) as stored:
        assert set(stored.files) == {
            "format_version",
            "actions",
            "T_worlds",
            "openings",
            "bimanual",
        }
        assert stored["actions"].shape[0] == 81
        assert stored["T_worlds"].shape[0] == 81
        assert stored["openings"].shape[0] == 81
        assert stored["format_version"].shape == ()
        assert stored["bimanual"].shape == ()
    evidence = staging["robot_kinematics"]
    assert evidence["all_five_fields_first_76_equal_selected_bundle"] is True
    assert evidence["scalar_fields_copied_unchanged"] == [
        "format_version",
        "bimanual",
    ]
    reconstruction._validate_staged_robot_kinematics(request, staging)


def test_action_window_staging_rejects_valid_but_wrong_selected_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, tmp_path / "out")
    _prepare_staging_inputs(request)
    selected_path = Path(
        request.frame_zero_manifest["action_alignment"][
            "selected_robot_kinematics_bundle"
        ]["path"]
    )
    with np.load(selected_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    arrays["T_worlds"][:, 2, 3] += 0.02
    arrays["actions"][:, 0, 2] += 0.02
    np.savez_compressed(selected_path, **arrays)
    manifest = {
        **request.frame_zero_manifest,
        "action_alignment": dict(request.frame_zero_manifest["action_alignment"]),
    }
    selected_record = reconstruction._bound_file(selected_path)
    selected_state = reconstruction.load_robot_kinematics_archive(selected_path)
    manifest["action_alignment"]["selected_robot_kinematics_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_bundle"] = selected_record
    manifest["action_alignment"]["selected_action_arrays"] = (
        reconstruction.robot_kinematics_array_records(selected_state)
    )
    request = reconstruction.ReconstructionRequest(
        **{**request.__dict__, "frame_zero_manifest": manifest}
    )
    with pytest.raises(ValueError, match="exact source slice"):
        reconstruction._stage_action_window(request, Path("/bin/true"))


@pytest.mark.parametrize("linked_component", ["robot", "camera"])
def test_action_window_staging_rejects_symlinked_dataset_ancestor(
    tmp_path: Path, linked_component: str
) -> None:
    request = _request(tmp_path, tmp_path / "out")
    _prepare_staging_inputs(request)
    assert request.aligned_episode_dir is not None
    source = (
        request.aligned_episode_dir / "robot"
        if linked_component == "robot"
        else request.aligned_episode_dir / request.camera_names[0]
    )
    outside = tmp_path / f"outside-{linked_component}"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        reconstruction._stage_action_window(request, Path("/bin/true"))
