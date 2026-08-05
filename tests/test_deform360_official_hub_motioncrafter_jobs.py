from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_official_hub_causal_windows import (
    load_deform360_official_hub_causal_window_manifest_v2,
)
from bayesian_phystwin.deform360_official_hub_motioncrafter_jobs import (
    build_deform360_motioncrafter_job_manifest,
    load_deform360_motioncrafter_job_manifest,
    motioncrafter_effective_seed,
    validate_deform360_motioncrafter_job_manifest,
)
from bayesian_phystwin.deform360_visual_provider_recovery_lock import (
    load_deform360_visual_provider_recovery_lock,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    repository = _repository()
    causal_path = (
        repository
        / "results/sota/deform360_official_hub_causal_window_feasibility_v2/"
        "causal_window_manifest.json"
    )
    provider_path = (
        repository
        / "protocols/locks/"
        "deform360_official_hub_visuotactile_v2_visual_provider_recovery_v1.json"
    )
    model_path = (
        repository
        / "protocols/locks/"
        "deform360_official_hub_visuotactile_v2_motioncrafter_model_set.json"
    )
    return build_deform360_motioncrafter_job_manifest(
        causal_window_manifest=(
            load_deform360_official_hub_causal_window_manifest_v2(causal_path)
        ),
        causal_window_manifest_file_sha256=_sha256(causal_path),
        provider_lock=load_deform360_visual_provider_recovery_lock(provider_path),
        provider_lock_file_sha256=_sha256(provider_path),
        model_set_manifest=load_strict_json_object(
            model_path,
            label="model set fixture",
        ),
        model_set_manifest_file_sha256=_sha256(model_path),
        implementation_revision="a" * 40,
        runner_source_sha256="b" * 64,
    )


def test_builds_frozen_complete_product_schedule_for_all_cameras() -> None:
    manifest = _manifest()

    assert validate_deform360_motioncrafter_job_manifest(manifest) == manifest[
        "manifest_sha256"
    ]
    assert manifest["object_count"] == 10
    assert manifest["job_count"] == 30
    jobs = manifest["jobs"]
    assert len(jobs) == 30
    assert manifest["smoke_job_id"] == jobs[0]["job_id"]
    assert {job["source_frame_count"] for job in jobs} == {42}
    assert {len(job["windows"]) for job in jobs} == {2}
    assert {len(job["seed_schedule"]) for job in jobs} == {4}
    assert manifest["run_configuration"]["products"] == [
        "disjoint_baseline",
        "latent_linear_baseline",
        "independently_decoded_overlap_windows",
    ]
    model_sources = manifest["motioncrafter"]["model_set_manifest"]["sources"]
    assert set(model_sources) == {"unet", "vae", "image_vae", "base_pipeline"}
    assert model_sources["image_vae"] == {
        "kind": "huggingface_revision",
        "repository": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "role": "image-vae",
        "schema": "prob4d.motioncrafter-model-source.v1",
    }


def test_seed_derivation_matches_prob4d_canonical_descriptor() -> None:
    descriptor = (
        b'{"call_id":"baseline-disjoint",'
        b'"root_seed":20260805,'
        b'"schema":"prob4d.motioncrafter-seed-schedule.v1"}'
    )
    expected = int.from_bytes(hashlib.sha256(descriptor).digest()[:4], "big")

    assert (
        motioncrafter_effective_seed(20260805, call_id="baseline-disjoint")
        == expected
    )


def test_committed_v3_job_manifest_binds_amended_runtime() -> None:
    path = (
        _repository()
        / "protocols/locks/"
        "deform360_official_hub_visuotactile_v3_motioncrafter_jobs.json"
    )

    manifest = load_deform360_motioncrafter_job_manifest(path)

    assert manifest["manifest_sha256"] == (
        "8cf8df7629d4f2a17ec4d5dcb992a65fca638acb8420a7cca79a91c5ecb80682"
    )
    assert manifest["implementation"]["revision"] == (
        "55982e89596ce8a19af977d2d9924d3f7e210809"
    )
    assert manifest["implementation"]["runner_source_sha256"] == (
        "62fdb997ebfcf30ec2906117a02a31cf14777678a023225db70149626c417052"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["jobs"][0]["source_video"].update(
                sha256="0" * 64
            ),
            "job-manifest identity changed",
        ),
        (
            lambda value: value["jobs"][0]["seed_schedule"][0].update(
                effective_seed=0
            ),
            "job-manifest identity changed",
        ),
        (
            lambda value: value["information_boundary"].update(
                calibration_scores_opened=True
            ),
            "job-manifest identity changed",
        ),
    ],
)
def test_manifest_mutations_fail_closed(mutation: object, message: str) -> None:
    manifest = copy.deepcopy(_manifest())
    mutation(manifest)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        validate_deform360_motioncrafter_job_manifest(manifest)


def test_rehashed_seed_mutation_still_fails_semantic_validation() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["jobs"][0]["seed_schedule"][0]["effective_seed"] = 0
    job = dict(manifest["jobs"][0])
    job.pop("job_id")
    from bayesian_phystwin._portable_contracts import content_id

    manifest["jobs"][0]["job_id"] = content_id(job)
    descriptor = dict(manifest)
    descriptor.pop("manifest_sha256")
    manifest["manifest_sha256"] = content_id(descriptor)

    with pytest.raises(ValueError, match="job seed schedule changed"):
        validate_deform360_motioncrafter_job_manifest(manifest)
