from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "build_deform360_visual_provider_freeze.py"
SPEC = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_official_hub_visuotactile_v1_visual_provider_spec.json"
)
POLICY = (
    ROOT
    / "protocols"
    / "locks"
    / "deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json"
)
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-visual-provider-freeze.yml"


def _module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "deform360_visual_provider_freeze", SCRIPT
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _snapshot(
    cache: Path,
    repository: str,
    revision: str,
    members: tuple[str, ...],
) -> None:
    root = cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    for member in members:
        path = root / member
        if "." in path.name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)


def _provider_manifest(revision: str) -> dict[str, Any]:
    module = _module()
    manifest: dict[str, Any] = {
        "provider_name": "prob4d",
        "provider_api_version": 2,
        "provider_revision": revision,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    manifest["manifest_id"] = module.content_id(manifest)
    return manifest


class _FakeProviderV2:
    @staticmethod
    def prob4d_provider_manifest(*, provider_revision: str) -> dict[str, Any]:
        return _provider_manifest(provider_revision)


class _FakeProviderAttestation:
    @staticmethod
    def build_provider_attestation(**arguments: Any) -> dict[str, Any]:
        manifest = arguments["provider_manifest"]
        return {
            "schema_name": "prob4d.provider-attestation",
            "schema_version": 1,
            "provider_api_version": 2,
            "provider_manifest_id": manifest["manifest_id"],
            "provider_manifest": manifest,
            "provider_revision": arguments["provider_revision"],
            "python_import_boundary": "prob4d.provider_v2",
            "export_mode": "exploratory",
            "claim_bearing": False,
            "calibration_compatibility_validated": False,
            "calibration_artifact_ids": {
                "gauge_artifact_id": None,
                "point_artifact_id": None,
            },
            "covariance_root_mode": "canonical_eigenspaces",
            "composition_jacobian_mode": "analytic",
            "runtime_revision": arguments["runtime_revision"],
        }


class _FakeModelSet:
    set_sha256 = "d" * 64
    manifest_json = json.dumps(
        {
            "schema": "prob4d.motioncrafter-model-set.v2",
            "model_type": "determ",
            "sources": {},
            "loader_module": {"sha256": "e" * 64, "bytes": 1},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class _FakeModels:
    class PinnedMotionCrafterModelSet:
        @staticmethod
        def inspect(**arguments: Any) -> _FakeModelSet:
            assert arguments["model_type"] == "determ"
            assert arguments["unet_revision"] == arguments["vae_revision"]
            return _FakeModelSet()


def test_committed_policy_and_spec_are_content_addressed() -> None:
    module = _module()
    policy = module.load_metric_prior_policy(POLICY)
    spec = module.load_preflight_spec(SPEC)

    assert policy["artifact_id"] == (
        "27d26c15784e46c5baedc4fab3d3cf9ac44cbb572799878956c24eaea9312a1d"
    )
    assert spec["artifact_id"] == (
        "65ad9fa1d3a176f663ff4ba720db15ebd92a0f0eccd27818bba5dba0b32ff7bc"
    )
    assert spec["metric_frame_prior_policy"]["artifact_id"] == policy["artifact_id"]
    sources = spec["motioncrafter"]["model_sources"]
    assert sources["unet"]["expected_revision"] == (
        "fc7b18d5657184607bf4501b02d64ada7540b4e3"
    )
    assert sources["vae"]["expected_revision"] == sources["unet"]["expected_revision"]
    assert sources["image_vae"]["expected_revision"] == (
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    assert sources["base_pipeline"]["expected_revision"] == (
        "9e43909513c6714f1bc78bcb44d96e733cd242aa"
    )


def test_cached_snapshot_resolution_is_exact_and_ambiguity_fails(
    tmp_path: Path,
) -> None:
    module = _module()
    repository = "owner/model"
    first = "1" * 40
    second = "2" * 40
    _snapshot(tmp_path, repository, first, ("model_index.json",))

    assert (
        module.resolve_cached_snapshot_revision(
            tmp_path,
            repository=repository,
            required_members=("model_index.json",),
        )
        == first
    )
    assert (
        module.resolve_cached_snapshot_revision(
            tmp_path,
            repository=repository,
            required_members=("model_index.json",),
            expected_revision=first,
        )
        == first
    )

    _snapshot(tmp_path, repository, second, ("model_index.json",))
    with pytest.raises(ValueError, match="unambiguous"):
        module.resolve_cached_snapshot_revision(
            tmp_path,
            repository=repository,
            required_members=("model_index.json",),
        )
    assert (
        module.resolve_cached_snapshot_revision(
            tmp_path,
            repository=repository,
            required_members=("model_index.json",),
            expected_revision=second,
        )
        == second
    )


def test_cached_snapshot_requires_all_declared_members(tmp_path: Path) -> None:
    module = _module()
    repository = "owner/model"
    revision = "3" * 40
    _snapshot(tmp_path, repository, revision, ("model_index.json",))

    with pytest.raises(ValueError, match="found \\[\\]"):
        module.resolve_cached_snapshot_revision(
            tmp_path,
            repository=repository,
            required_members=("model_index.json", "weights/model.safetensors"),
        )


def test_preflight_rejects_mutated_information_boundary(tmp_path: Path) -> None:
    module = _module()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["information_boundary"]["calibration_payloads_opened"] = True
    spec.pop("artifact_id")
    spec["artifact_id"] = module.content_id(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="information boundary"):
        module.load_preflight_spec(path)


def test_metric_policy_rejects_future_frame_use(tmp_path: Path) -> None:
    module = _module()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["future_frames_used"] = True
    policy.pop("artifact_id")
    policy["artifact_id"] = module.content_id(policy)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="future_frames_used"):
        module.load_metric_prior_policy(path)


def test_complete_preflight_builds_and_replays_exact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    provider_revision = spec["provider"]["revision"]
    motion_revision = spec["motioncrafter"]["revision"]

    prob4d = tmp_path / "prob4d"
    motioncrafter = tmp_path / "motioncrafter"
    prob4d.mkdir()
    motioncrafter.mkdir()
    cache = tmp_path / "cache"
    sources = spec["motioncrafter"]["model_sources"]
    for role in ("unet", "vae", "image_vae", "base_pipeline"):
        source = sources[role]
        _snapshot(
            cache,
            source["repository"],
            source["expected_revision"],
            tuple(source["required_members"]),
        )

    monkeypatch.setattr(
        module,
        "git_head",
        lambda checkout, *, name: (
            provider_revision if name == "Prob4D" else motion_revision
        ),
    )
    monkeypatch.setattr(module, "require_clean_checkout", lambda *args, **kwargs: None)

    def fake_import(checkout: Path, module_name: str) -> Any:
        del checkout
        if module_name == "prob4d.provider_v2":
            return _FakeProviderV2
        if module_name == "prob4d.provider_attestation":
            return _FakeProviderAttestation
        if module_name == "prob4d.motioncrafter_models":
            return _FakeModels
        raise AssertionError(module_name)

    monkeypatch.setattr(module, "_import_from_checkout", fake_import)
    first = tmp_path / "first"
    summary = module.build_preflight(
        spec_path=SPEC,
        metric_policy_path=POLICY,
        prob4d_checkout=prob4d,
        motioncrafter_checkout=motioncrafter,
        cache_directory=cache,
        output_directory=first,
    )

    assert summary["visual_provider_lock_id"]
    assert summary["selected_raw_payloads_opened"] is False
    assert (first / "SHA256SUMS").is_file()
    lock = json.loads((first / "visual-provider-lock.json").read_text())
    assert lock["model_set_id"] == "d" * 64
    assert lock["initial_metric_frame_prior_id"] == (
        "27d26c15784e46c5baedc4fab3d3cf9ac44cbb572799878956c24eaea9312a1d"
    )
    assert lock["metadata"]["model_source_revisions"] == {
        role: sources[role]["expected_revision"]
        for role in ("base_pipeline", "image_vae", "unet", "vae")
    }

    second = tmp_path / "second"
    module.build_preflight(
        spec_path=SPEC,
        metric_policy_path=POLICY,
        prob4d_checkout=prob4d,
        motioncrafter_checkout=motioncrafter,
        cache_directory=cache,
        output_directory=second,
        expected_bundle_directory=first,
    )
    assert (second / "visual-provider-lock.json").read_bytes() == (
        first / "visual-provider-lock.json"
    ).read_bytes()


def test_expected_bundle_detects_changed_provider_bytes(tmp_path: Path) -> None:
    module = _module()
    generated = tmp_path / "generated"
    expected = tmp_path / "expected"
    generated.mkdir()
    expected.mkdir()
    for name in module._OUTPUT_FILES:
        (generated / name).write_text("same\n", encoding="utf-8")
        (expected / name).write_text("same\n", encoding="utf-8")
    module._verify_expected_bundle(generated, expected)
    (expected / "provider-attestation.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provider-attestation"):
        module._verify_expected_bundle(generated, expected)


def test_workflow_is_read_only_pinned_and_data_free() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in workflow
    assert spec["provider"]["revision"] in workflow
    assert spec["motioncrafter"]["revision"] in workflow
    assert "git push" not in workflow
    assert "deform360_data_root" not in workflow.lower()
    assert "actions/upload-artifact@" in workflow


def test_cli_exposes_no_dataset_or_target_arguments() -> None:
    module = _module()
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "dataset_root" not in destinations
    assert "target" not in destinations
    assert "confirmation_root" not in destinations
    assert {"spec", "metric_prior_policy", "cache_dir", "output_dir"}.issubset(
        destinations
    )
