from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

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
    contract = reconstruction._copy_contract()
    return reconstruction.ReconstructionRequest(
        case_name=CASE_NAME,
        object_id="083-blanket-cloth",
        episode_id=0,
        role="calibration",
        cohort_barrier_sha256="b" * 64,
        aligned_episode_dir=tmp_path / "083-blanket-cloth" / "episode_0000",
        output_dir=output_dir,
        source_frame_start=62,
        source_frame_stop=143,
        camera_names=CAMERAS,
        frame_zero_arrays=arrays,
        frame_zero_manifest={"bundle": reconstruction._bound_file(bundle)},
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
                "propagated_frame_zero_mask_sha256": mask_sha,
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
                "source_inputs": {},
                "staged_outputs": {},
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
                "sealed_mask_is_only_initialization": True,
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
