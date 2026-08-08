from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import bayesian_phystwin.deform360_calibration_visual_production as production_api
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_calibration_visual_production import (
    PRODUCTION_INFORMATION_BOUNDARY,
    build_deform360_calibration_visual_command,
    build_deform360_calibration_visual_prediction_seal,
    build_deform360_calibration_visual_production_result,
    build_deform360_calibration_visual_technical_failure,
    deform360_calibration_visual_command_descriptor,
    validate_deform360_calibration_visual_prediction_seal,
    validate_deform360_calibration_visual_production_result,
    validate_deform360_motioncrafter_model_set_binding,
    validate_deform360_motioncrafter_prediction_manifest,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
)

IMPLEMENTATION_REVISION = "a" * 40
PROVIDER_REVISION = "b" * 40
MOTIONCRAFTER_REVISION = "c" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _model_binding() -> dict[str, object]:
    revisions = {
        "unet": "1" * 40,
        "vae": "1" * 40,
        "image_vae": "2" * 40,
        "base_pipeline": "3" * 40,
    }
    repositories = {
        "unet": "TencentARC/MotionCrafter",
        "vae": "TencentARC/MotionCrafter",
        "image_vae": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_pipeline": "stabilityai/stable-video-diffusion-img2vid-xt",
    }
    roles = {
        "unet": "unet",
        "vae": "vae",
        "image_vae": "image-vae",
        "base_pipeline": "base-pipeline",
    }
    sources = {
        key: {
            "schema": "prob4d.motioncrafter-model-source.v1",
            "role": roles[key],
            "kind": "huggingface_revision",
            "repository": repositories[key],
            "revision": revisions[key],
        }
        for key in revisions
    }
    manifest = {
        "schema": "prob4d.motioncrafter-model-set.v2",
        "model_type": "determ",
        "sources": sources,
        "loader_module": {
            "module": "prob4d.motioncrafter_models",
            "sha256": "4" * 64,
            "bytes": 12345,
        },
    }
    model_set_id = hashlib.sha256(_canonical(manifest)).hexdigest()
    return {
        "schema": ("bayesian-phystwin/deform360-motioncrafter-model-set-binding-v1"),
        "schema_version": 1,
        "model_set_id": model_set_id,
        "model_set_manifest": manifest,
        "cached_revisions": revisions,
        "cache_path_recorded": False,
        "selected_raw_payloads_opened": False,
        "target_outcomes_used": False,
    }


def _lock(model_set_id: str) -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1(
        provider_revision=PROVIDER_REVISION,
        provider_manifest_id="5" * 64,
        provider_attestation_sha256="6" * 64,
        motioncrafter_revision=MOTIONCRAFTER_REVISION,
        model_set_id=model_set_id,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id="7" * 64,
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
    )


def _job(
    *,
    object_id: str = "object-a",
    camera_id: str = "camera-0",
    seed: int = 101,
) -> dict[str, object]:
    prefix = [20, 78]
    prediction = [20, 96]
    return {
        "job_id": hashlib.sha256(f"{object_id}:{camera_id}".encode()).hexdigest(),
        "object_id": object_id,
        "episode_id": 3,
        "stratum": "sheet",
        "camera_id": camera_id,
        "view_root_seed": seed,
        "prefix_source_frame_range_half_open": prefix,
        "prediction_source_frame_range_half_open": prediction,
        "source_video": {
            "path": f"{object_id}/episode_0000/{camera_id}/undistorted.mp4",
            "sha256": "8" * 64,
            "byte_count": 200,
        },
        "source_timestamps": {
            "path": (f"{object_id}/episode_0000/{camera_id}/aligned_timestamps.txt"),
            "sha256": "9" * 64,
            "byte_count": 100,
        },
        "output_relative_directory": (
            f"objects/{object_id}/episode_3/views/{camera_id}"
        ),
    }


def _admission(lock: Deform360VisualProviderLockV1) -> dict[str, object]:
    jobs = [_job(camera_id="camera-0"), _job(camera_id="camera-1", seed=102)]
    return {
        "admission_id": "a" * 64,
        "visual_provider_lock_id": lock.artifact_id,
        "provider_revision": lock.provider_revision,
        "motioncrafter_revision": lock.motioncrafter_revision,
        "model_set_id": lock.model_set_id,
        "protocol_id": lock.protocol_id,
        "object_count": 1,
        "camera_view_count": 2,
        "jobs": jobs,
    }


def _validated_binding() -> tuple[dict[str, object], dict[str, object]]:
    raw = _model_binding()
    normalized = validate_deform360_motioncrafter_model_set_binding(
        raw,
        expected_model_set_id=str(raw["model_set_id"]),
    )
    return raw, normalized


def _manifest(
    *,
    job: dict[str, object],
    lock: Deform360VisualProviderLockV1,
    binding: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    model_set_id = binding["model_set_id"]
    identity = f"prob4d.motioncrafter-model-set.v2:{model_set_id}"
    config = {
        "model_type": "determ",
        "unet_path": f"{identity}#unet",
        "vae_path": f"{identity}#geometry-motion-vae",
        "base_pipeline_path": f"{identity}#base-video-pipeline",
        "height": 320,
        "width": 640,
        "window_size": 25,
        "overlap": 8,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "decode_chunk_size": 25,
        "seed": job["view_root_seed"],
        "seed_policy": "derived-per-call",
        "low_memory_usage": False,
        "frame_start": 20,
        "frame_stop": 78,
        "frame_stride": 1,
        "model_source_schema": "prob4d.motioncrafter-model-set.v2",
        "model_source_set_sha256": model_set_id,
        "model_source_manifest_json": json.dumps(
            binding["manifest"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "model_loader_module_sha256": binding["loader_sha256"],
        "model_loader_module_bytes": binding["loader_bytes"],
    }
    run_spec = {
        "schema": "prob4d.motioncrafter-run-spec.v1",
        "producer": {},
        "input_video": {
            "sha256": job["source_video"]["sha256"],
            "bytes": job["source_video"]["byte_count"],
        },
        "motioncrafter_upstream": {
            "commit": lock.motioncrafter_revision,
            "clean": True,
            "status_sha256": "0" * 64,
            "status_entry_count": 0,
        },
        "inference_config": config,
    }
    run_spec_sha = hashlib.sha256(_canonical(run_spec)).hexdigest()
    manifest = {
        "format_version": 1,
        "overlap_windows": [
            {
                "window_id": "window-0",
                "start_frame": 20,
                "stop_frame": 45,
                "path": "windows/window-0.npz",
            },
            {
                "window_id": "window-1",
                "start_frame": 53,
                "stop_frame": 78,
                "path": "windows/window-1.npz",
            },
        ],
        "artifact_integrity": {
            "schema": "prob4d.motioncrafter-artifact-integrity.v1",
            "run_spec": run_spec,
            "run_spec_sha256": run_spec_sha,
            "members": [],
        },
    }
    verification = {
        "integrity_bound": True,
        "hashes_verified": True,
        "member_count": 4,
        "run_spec_sha256": run_spec_sha,
    }
    return manifest, verification


def _file(path: str, byte_count: int = 10) -> dict[str, object]:
    return {"path": path, "sha256": "d" * 64, "byte_count": byte_count}


def test_model_binding_command_and_descriptor_are_exact_and_portable() -> None:
    raw, binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    admission = _admission(lock)
    job = _job()

    descriptor = deform360_calibration_visual_command_descriptor(
        admission=admission,
        job=job,
        provider_lock=lock,
        model_binding=binding,
    )
    command = build_deform360_calibration_visual_command(
        executable="/venv/bin/prob4d-motioncrafter",
        source_video_path="/retained/object/video.mp4",
        output_directory="/outputs/job",
        motioncrafter_root="/checkouts/MotionCrafter",
        cache_directory="/cache",
        job=job,
        provider_lock=lock,
        model_binding=binding,
        resume=True,
    )

    assert descriptor["frame_stop"] == 78
    assert descriptor["seed_policy"] == "derived-per-call"
    assert "/retained" not in json.dumps(descriptor)
    assert command[1] == "/retained/object/video.mp4"
    assert command[-1] == "--resume"
    assert command[command.index("--frame-stop") + 1] == "78"
    assert command[command.index("--seed") + 1] == "101"
    assert "96" not in command
    assert command[command.index("--unet-revision") + 1] == "1" * 40
    without_resume = build_deform360_calibration_visual_command(
        executable="prob4d-motioncrafter",
        source_video_path="video.mp4",
        output_directory="output",
        motioncrafter_root="MotionCrafter",
        cache_directory="cache",
        job=job,
        provider_lock=lock,
        model_binding=binding,
        resume=False,
    )
    assert "--resume" not in without_resume


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update(cache_path_recorded=True), "cache_path_recorded"),
        (
            lambda value: value["cached_revisions"].update(unet="f" * 40),
            "cached unet",
        ),
        (
            lambda value: value["model_set_manifest"]["sources"]["unet"].update(
                kind="local_snapshot"
            ),
            "changed kind|digest mismatch",
        ),
        (
            lambda value: value["model_set_manifest"].update(model_type="diff"),
            "manifest changed|digest mismatch",
        ),
    ],
)
def test_model_binding_rejects_drift(mutator, match: str) -> None:
    binding = _model_binding()
    mutator(binding)
    with pytest.raises(ValueError, match=match):
        validate_deform360_motioncrafter_model_set_binding(
            binding,
            expected_model_set_id=str(binding["model_set_id"]),
        )


def test_prediction_manifest_is_bound_to_source_model_seed_and_cutoff() -> None:
    raw, binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    job = _job()
    manifest, verification = _manifest(job=job, lock=lock, binding=binding)

    result = validate_deform360_motioncrafter_prediction_manifest(
        manifest,
        verification=verification,
        job=job,
        provider_lock=lock,
        model_binding=binding,
    )

    assert result["member_count"] == 4
    assert result["run_spec_sha256"] == verification["run_spec_sha256"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda manifest, verification: manifest["overlap_windows"][1].update(
                stop_frame=79
            ),
            "post-cutoff",
        ),
        (
            lambda manifest, verification: manifest["artifact_integrity"]["run_spec"][
                "input_video"
            ].update(sha256="e" * 64),
            "run-spec digest mismatch|input video",
        ),
        (
            lambda manifest, verification: manifest["artifact_integrity"]["run_spec"][
                "motioncrafter_upstream"
            ].update(clean=False),
            "run-spec digest mismatch|checkout differs",
        ),
        (
            lambda manifest, verification: verification.update(hashes_verified=False),
            "did not verify",
        ),
        (
            lambda manifest, verification: manifest["artifact_integrity"]["run_spec"][
                "inference_config"
            ].update(seed=999),
            "run-spec digest mismatch|inference config",
        ),
    ],
)
def test_prediction_manifest_rejects_tampering(change, match: str) -> None:
    raw, binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    job = _job()
    manifest, verification = _manifest(job=job, lock=lock, binding=binding)
    change(manifest, verification)
    with pytest.raises(ValueError, match=match):
        validate_deform360_motioncrafter_prediction_manifest(
            manifest,
            verification=verification,
            job=job,
            provider_lock=lock,
            model_binding=binding,
        )


def test_prediction_seal_closes_evaluation_and_confirmation_boundaries() -> None:
    raw, _binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    admission = _admission(lock)
    job = _job()

    seal = build_deform360_calibration_visual_prediction_seal(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        job=job,
        provider_lock=lock,
        command_id="e" * 64,
        prediction_manifest=_file("predictions.json", 500),
        run_spec_sha256="f" * 64,
        verified_member_count=4,
    )

    assert seal["causal_prefix_frame_range_half_open"] == [20, 78]
    assert seal["reserved_evaluation_frame_range_half_open"] == [78, 96]
    assert seal["information_boundary"] == PRODUCTION_INFORMATION_BOUNDARY
    assert not seal["information_boundary"]["confirmation_payloads_opened"]
    assert not seal["information_boundary"]["target_outcomes_used"]

    tampered = copy.deepcopy(seal)
    tampered["reserved_evaluation_frame_range_half_open"][0] = 77
    with pytest.raises(ValueError, match="frame boundary"):
        validate_deform360_calibration_visual_prediction_seal(tampered)

    tampered = copy.deepcopy(seal)
    tampered["information_boundary"]["reserved_evaluation_frames_opened"] = True
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_calibration_visual_prediction_seal(tampered)


def test_failure_and_complete_result_retain_every_admitted_job() -> None:
    raw, _binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    admission = _admission(lock)
    first = _job(camera_id="camera-0")
    second = _job(camera_id="camera-1", seed=102)
    seal = build_deform360_calibration_visual_prediction_seal(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        job=first,
        provider_lock=lock,
        command_id="e" * 64,
        prediction_manifest=_file("predictions.json", 500),
        run_spec_sha256="f" * 64,
        verified_member_count=4,
    )
    failure = build_deform360_calibration_visual_technical_failure(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        job=second,
        provider_lock=lock,
        command_id="1" * 64,
        stage="motioncrafter-production",
        return_code=7,
        detail=b"synthetic failure",
        stdout=_file("logs/out.bin", 0),
        stderr=_file("logs/err.bin", 12),
    )
    rows = [
        {
            "job_id": first["job_id"],
            "object_id": first["object_id"],
            "camera_id": first["camera_id"],
            "status": "succeeded",
            "receipt": _file("objects/a/seal.json"),
        },
        {
            "job_id": second["job_id"],
            "object_id": second["object_id"],
            "camera_id": second["camera_id"],
            "status": "technical-failure",
            "receipt": _file("failures/b.json"),
        },
    ]

    result = build_deform360_calibration_visual_production_result(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        provider_lock=lock,
        jobs=rows,
    )

    assert seal["seal_id"] != failure["failure_id"]
    assert result["camera_view_count"] == 2
    assert result["succeeded_job_count"] == 1
    assert result["technical_failure_job_count"] == 1
    assert result["completely_succeeded_object_count"] == 0
    assert result["status"] == "technical-failures-retained"
    assert result["information_boundary"]["replacement_allowed"] is False

    forged = copy.deepcopy(result)
    forged["technical_failure_job_count"] = 0
    with pytest.raises(ValueError, match="failure count"):
        validate_deform360_calibration_visual_production_result(forged)

    reordered = copy.deepcopy(result)
    reordered["jobs"].reverse()
    with pytest.raises(ValueError, match="not sorted"):
        validate_deform360_calibration_visual_production_result(reordered)


def test_all_success_result_counts_complete_object() -> None:
    raw, _binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    admission = _admission(lock)
    jobs = [_job(camera_id="camera-0"), _job(camera_id="camera-1", seed=102)]
    rows = [
        {
            "job_id": job["job_id"],
            "object_id": job["object_id"],
            "camera_id": job["camera_id"],
            "status": "succeeded",
            "receipt": _file(f"receipts/{index}.json"),
        }
        for index, job in enumerate(jobs)
    ]
    result = build_deform360_calibration_visual_production_result(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        provider_lock=lock,
        jobs=rows,
    )
    assert result["status"] == "all-jobs-succeeded"
    assert result["completely_succeeded_object_count"] == 1


def test_content_ids_change_with_numerical_or_lineage_fields() -> None:
    raw, _binding = _validated_binding()
    lock = _lock(str(raw["model_set_id"]))
    admission = _admission(lock)
    job = _job()
    base = build_deform360_calibration_visual_technical_failure(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        job=job,
        provider_lock=lock,
        command_id="1" * 64,
        stage="prediction-verification",
        return_code=1,
        detail=b"one",
        stdout=_file("logs/out.bin", 0),
        stderr=_file("logs/err.bin", 0),
    )
    changed = build_deform360_calibration_visual_technical_failure(
        implementation_revision=IMPLEMENTATION_REVISION,
        admission=admission,
        job=job,
        provider_lock=lock,
        command_id="1" * 64,
        stage="prediction-verification",
        return_code=1,
        detail=b"two",
        stdout=_file("logs/out.bin", 0),
        stderr=_file("logs/err.bin", 0),
    )
    assert base["failure_id"] != changed["failure_id"]
    assert (
        content_id({key: value for key, value in base.items() if key != "failure_id"})
        == base["failure_id"]
    )


def test_validation_primitives_reject_malformed_metadata() -> None:
    with pytest.raises(ValueError, match="nonempty literal string"):
        production_api._string("", name="value")
    with pytest.raises(ValueError, match="integer >= 0"):
        production_api._integer(False, name="value")
    with pytest.raises(ValueError, match="JSON object"):
        production_api._file_record([], name="file")
    with pytest.raises(ValueError, match="JSON object"):
        validate_deform360_motioncrafter_model_set_binding(
            [], expected_model_set_id="0" * 64
        )


def test_shared_adapter_factory_reuses_one_loaded_adapter() -> None:
    import runpy
    from dataclasses import dataclass
    from pathlib import Path

    namespace = runpy.run_path(
        "scripts/science/execute_deform360_calibration_visual_production.py"
    )
    shared_factory_type = namespace["_SharedAdapterFactory"]

    @dataclass(frozen=True)
    class Config:
        upstream_root: Path
        video_path: Path
        output_directory: Path
        cache_directory: str
        height: int
        seed: int
        frame_start: int
        frame_stop: int

    class Adapter:
        def __init__(self, config: Config) -> None:
            self.config = config

    created: list[Adapter] = []

    def factory(config: Config) -> Adapter:
        adapter = Adapter(config)
        created.append(adapter)
        return adapter

    shared = shared_factory_type(factory)
    first_config = Config(
        upstream_root=Path("MotionCrafter"),
        video_path=Path("a.mp4"),
        output_directory=Path("a"),
        cache_directory="cache",
        height=320,
        seed=11,
        frame_start=0,
        frame_stop=58,
    )
    second_config = Config(
        upstream_root=Path("MotionCrafter"),
        video_path=Path("b.mp4"),
        output_directory=Path("b"),
        cache_directory="cache",
        height=320,
        seed=12,
        frame_start=20,
        frame_stop=78,
    )

    first = shared(first_config)
    second = shared(second_config)

    assert first is second
    assert len(created) == 1
    assert shared.creation_attempt_count == 1
    assert shared.creation_count == 1
    assert second.config == second_config

    changed_fixed = Config(
        upstream_root=Path("different"),
        video_path=Path("c.mp4"),
        output_directory=Path("c"),
        cache_directory="cache",
        height=320,
        seed=13,
        frame_start=40,
        frame_stop=98,
    )
    with pytest.raises(ValueError, match="fixed fields"):
        shared(changed_fixed)


def test_shared_adapter_factory_latches_initial_creation_failure() -> None:
    import runpy
    from dataclasses import dataclass

    namespace = runpy.run_path(
        "scripts/science/execute_deform360_calibration_visual_production.py"
    )
    shared_factory_type = namespace["_SharedAdapterFactory"]

    @dataclass(frozen=True)
    class Config:
        video_path: Path
        output_directory: Path
        seed: int
        frame_start: int
        frame_stop: int

    attempts = 0

    def factory(_config: Config) -> object:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("synthetic load failure")

    shared = shared_factory_type(factory)
    config = Config(Path("a.mp4"), Path("a"), 1, 0, 58)
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        shared(config)
    with pytest.raises(RuntimeError, match="creation already failed"):
        shared(config)

    assert attempts == 1
    assert shared.creation_attempt_count == 1
    assert shared.creation_count == 0


def test_shared_producer_matches_the_pinned_prob4d_public_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import runpy
    import sys
    import types
    from dataclasses import dataclass
    from types import SimpleNamespace

    namespace = runpy.run_path(
        "scripts/science/execute_deform360_calibration_visual_production.py"
    )
    producer_type = namespace["_SharedMotionCrafterProducer"]
    model_set_id = "a" * 64
    manifest = {"schema": "prob4d.motioncrafter-model-set.v2", "value": 1}
    created_adapters: list[object] = []
    runner_configs: list[object] = []
    verified_paths: list[Path] = []

    @dataclass(frozen=True)
    class Config:
        upstream_root: Path
        video_path: Path
        output_directory: Path
        cache_directory: str
        height: int
        width: int
        window_size: int
        overlap: int
        num_inference_steps: int
        guidance_scale: float
        decode_chunk_size: int
        seed: int
        seed_policy: str
        low_memory_usage: bool
        frame_start: int
        frame_stop: int
        frame_stride: int

    class Adapter:
        def __init__(self, config: Config) -> None:
            self.config = config

    class ModelSet:
        set_sha256 = model_set_id
        manifest_json = json.dumps(manifest, sort_keys=True)

        def build_config(self, **kwargs: object) -> Config:
            return Config(**kwargs)  # type: ignore[arg-type]

        def adapter_factory(self):
            def factory(config: Config) -> Adapter:
                adapter = Adapter(config)
                created_adapters.append(adapter)
                return adapter

            return factory

    class PinnedMotionCrafterModelSet:
        @classmethod
        def inspect(cls, **_kwargs: object) -> ModelSet:
            return ModelSet()

    class SafeMotionCrafterRunner:
        def __init__(self, config: Config, *, adapter_factory) -> None:
            self.config = config
            self.adapter_factory = adapter_factory

        def run(self, *, resume: bool) -> Path:
            assert resume
            if self.config.output_directory.name == "resumed":
                self.config.output_directory.mkdir(parents=True, exist_ok=True)
                return self.config.output_directory / "predictions.json"
            adapter = self.adapter_factory(self.config)
            runner_configs.append(adapter.config)
            self.config.output_directory.mkdir(parents=True, exist_ok=True)
            return self.config.output_directory / "predictions.json"

    def verify(path: Path, *, verify_hashes: bool) -> dict[str, object]:
        assert verify_hashes
        verified_paths.append(path)
        return {"manifest_path": str(path), "hashes_verified": True}

    package = types.ModuleType("prob4d")
    package.__path__ = []  # type: ignore[attr-defined]
    models_module = types.ModuleType("prob4d.motioncrafter_models")
    models_module.PinnedMotionCrafterModelSet = PinnedMotionCrafterModelSet
    runner_module = types.ModuleType("prob4d.motioncrafter_runner")
    runner_module.SafeMotionCrafterRunner = SafeMotionCrafterRunner
    integrity_module = types.ModuleType("prob4d.motioncrafter_integrity")
    integrity_module.verify_motioncrafter_prediction_manifest = verify
    monkeypatch.setitem(sys.modules, "prob4d", package)
    monkeypatch.setitem(sys.modules, "prob4d.motioncrafter_models", models_module)
    monkeypatch.setitem(sys.modules, "prob4d.motioncrafter_runner", runner_module)
    monkeypatch.setitem(sys.modules, "prob4d.motioncrafter_integrity", integrity_module)

    sources = {
        role: {"repository": f"example/{role}", "revision": "b" * 40}
        for role in ("unet", "vae", "image_vae", "base_pipeline")
    }
    producer = producer_type(
        model_binding={
            "model_set_id": model_set_id,
            "manifest": manifest,
            "sources": sources,
        },
        motioncrafter_root=tmp_path / "MotionCrafter",
        cache_directory=tmp_path / "cache",
        provider_lock=SimpleNamespace(height=320, width=640, window_size=25, overlap=8),
    )
    resumed = producer.produce(
        job={"prefix_source_frame_range_half_open": [0, 58], "view_root_seed": 10},
        source_video_path=tmp_path / "resumed.mp4",
        output_directory=tmp_path / "resumed",
        resume=True,
    )
    assert resumed == tmp_path / "resumed" / "predictions.json"
    assert producer.model_load_attempt_count == 0
    assert producer.model_load_count == 0

    first = producer.produce(
        job={"prefix_source_frame_range_half_open": [0, 58], "view_root_seed": 11},
        source_video_path=tmp_path / "a.mp4",
        output_directory=tmp_path / "a",
        resume=True,
    )
    second = producer.produce(
        job={"prefix_source_frame_range_half_open": [20, 78], "view_root_seed": 12},
        source_video_path=tmp_path / "b.mp4",
        output_directory=tmp_path / "b",
        resume=True,
    )

    assert first == tmp_path / "a" / "predictions.json"
    assert second == tmp_path / "b" / "predictions.json"
    assert producer.model_load_count == 1
    assert producer.model_load_attempt_count == 1
    assert len(created_adapters) == 1
    assert len(runner_configs) == 2
    assert runner_configs[-1].video_path == tmp_path / "b.mp4"
    assert producer.verify(second)["hashes_verified"] is True
    assert verified_paths == [second]
