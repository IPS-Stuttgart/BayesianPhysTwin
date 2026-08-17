#!/usr/bin/env python3
"""Build the data-free Deform360 visual-provider lock candidate.

This command reads only repository source, immutable model-cache metadata/files,
and target-blind protocol records. It never accepts a Deform360 dataset path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

SPEC_SCHEMA = "bayesian-phystwin/deform360-visual-provider-preflight-spec-v1"
METRIC_POLICY_SCHEMA = "bayesian-phystwin/deform360-metric-frame-prior-policy-v1"
PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
AMENDMENT_ID = "deform360-official-hub-visuotactile-v1-visual-provider-lock"

_SPEC_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "protocol_id",
        "amendment_id",
        "provider",
        "motioncrafter",
        "metric_frame_prior_policy",
        "gauge_covariance",
        "additional_metric_anchor_policy",
        "information_boundary",
        "claim_boundary",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "protocol_id",
        "semantics",
        "statistical_unit",
        "frame_selection",
        "source_family",
        "estimator",
        "per_object_numeric_prior_is_calibration_artifact",
        "allowed_before_prediction",
        "future_frames_used",
        "additional_metric_anchor_policy",
        "confirmation_payloads_opened",
        "target_outcomes_used",
        "claim_boundary",
    }
)
_OUTPUT_FILES = (
    "provider-manifest.json",
    "provider-attestation.json",
    "motioncrafter-model-set.json",
    "visual-provider-lock.json",
    "summary.json",
)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def load_strict_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be a JSON object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_id(value: Mapping[str, Any]) -> str:
    descriptor = dict(value)
    descriptor.pop("artifact_id", None)
    return hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be an ordinary file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any], *, expected: frozenset[str], name: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _require_literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty literal canonical string")
    return value


def _require_revision(value: object, *, name: str) -> str:
    revision = _require_literal_string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase revision")
    return revision


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_literal_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return value


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_probability(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a real number in (0, 1]")
    result = float(value)
    if not (0.0 < result <= 1.0):
        raise ValueError(f"{name} must be a real number in (0, 1]")
    return result


def load_metric_prior_policy(path: str | Path) -> dict[str, Any]:
    policy = load_strict_json(path, name="metric-frame prior policy")
    _require_exact_fields(policy, expected=_POLICY_FIELDS, name="metric-frame policy")
    if policy["schema"] != METRIC_POLICY_SCHEMA or policy["schema_version"] != 1:
        raise ValueError("metric-frame policy schema changed")
    if policy["protocol_id"] != PROTOCOL_ID:
        raise ValueError("metric-frame policy protocol changed")
    declared_id = _require_sha256(
        policy["artifact_id"], name="metric-frame policy artifact_id"
    )
    if content_id(policy) != declared_id:
        raise ValueError("metric-frame policy artifact_id does not match content")
    expected = {
        "statistical_unit": "physical object",
        "frame_selection": "first retained causal frame",
        "per_object_numeric_prior_is_calibration_artifact": True,
        "allowed_before_prediction": True,
        "future_frames_used": False,
        "additional_metric_anchor_policy": "none",
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }
    for field, value in expected.items():
        if policy[field] != value:
            raise ValueError(f"metric-frame policy changed {field}")
    return policy


def load_preflight_spec(path: str | Path) -> dict[str, Any]:
    spec = load_strict_json(path, name="visual-provider preflight spec")
    _require_exact_fields(spec, expected=_SPEC_FIELDS, name="preflight spec")
    if spec["schema"] != SPEC_SCHEMA or spec["schema_version"] != 1:
        raise ValueError("preflight spec schema changed")
    if spec["protocol_id"] != PROTOCOL_ID or spec["amendment_id"] != AMENDMENT_ID:
        raise ValueError("preflight spec protocol boundary changed")
    declared_id = _require_sha256(spec["artifact_id"], name="preflight artifact_id")
    if content_id(spec) != declared_id:
        raise ValueError("preflight artifact_id does not match content")
    boundary = spec["information_boundary"]
    if not isinstance(boundary, Mapping):
        raise ValueError("preflight information_boundary must be an object")
    expected_boundary = {
        "selected_raw_payloads_opened": False,
        "calibration_payloads_opened": False,
        "confirmation_payloads_opened": False,
        "target_outcomes_used": False,
    }
    if dict(boundary) != expected_boundary:
        raise ValueError("preflight information boundary changed")
    if spec["additional_metric_anchor_policy"] != "none":
        raise ValueError(
            "primary visual provider must use no additional metric anchors"
        )
    return spec


def git_head(checkout: str | Path, *, name: str) -> str:
    root = Path(checkout).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"{name} is not a Git checkout: {root}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _require_revision(result.stdout.strip(), name=f"{name} HEAD")


def require_clean_checkout(checkout: str | Path, *, name: str) -> None:
    root = Path(checkout).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError(f"{name} checkout is dirty")


def _snapshot_root(cache_directory: Path, repository: str) -> Path:
    repository_directory = "models--" + repository.replace("/", "--")
    return cache_directory / repository_directory / "snapshots"


def resolve_cached_snapshot_revision(
    cache_directory: str | Path,
    *,
    repository: str,
    required_members: Sequence[str],
    expected_revision: str | None = None,
) -> str:
    cache = Path(cache_directory).expanduser().resolve()
    snapshots = _snapshot_root(cache, repository)
    if not snapshots.is_dir():
        raise ValueError(f"cached model repository is missing: {snapshots}")
    members = tuple(
        _require_literal_string(item, name=f"{repository} required member")
        for item in required_members
    )
    if not members:
        raise ValueError(f"{repository} must declare required members")
    complete: list[str] = []
    for candidate in sorted(snapshots.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        try:
            revision = _require_revision(
                candidate.name, name=f"{repository} cached revision"
            )
        except ValueError:
            continue
        if all((candidate / relative).exists() for relative in members):
            complete.append(revision)
    if expected_revision is not None:
        expected = _require_revision(
            expected_revision, name=f"{repository} expected model revision"
        )
        if expected not in complete:
            raise ValueError(
                f"{repository} expected cached revision {expected} is incomplete or absent"
            )
        return expected
    if len(complete) != 1:
        raise ValueError(
            f"{repository} requires one unambiguous complete cached revision; "
            f"found {complete}"
        )
    return complete[0]


def _import_from_checkout(checkout: Path, module_name: str) -> Any:
    source_root = (checkout / "src").resolve()
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
    module_path = Path(cast(str, module.__file__)).resolve()
    if source_root not in module_path.parents:
        raise ValueError(
            f"{module_name} imported from {module_path}, not verified checkout {source_root}"
        )
    return module


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "SHA256SUMS"
    }


def _write_sha256sums(root: Path) -> None:
    hashes = _relative_hashes(root)
    text = "".join(f"{digest}  {path}\n" for path, digest in sorted(hashes.items()))
    (root / "SHA256SUMS").write_text(text, encoding="utf-8")


def _verify_expected_bundle(generated: Path, expected: Path) -> None:
    if expected.is_symlink() or not expected.is_dir():
        raise ValueError(f"expected bundle must be an ordinary directory: {expected}")
    for filename in _OUTPUT_FILES:
        generated_file = generated / filename
        expected_file = expected / filename
        if not expected_file.is_file() or expected_file.is_symlink():
            raise ValueError(f"expected bundle lacks ordinary file {filename}")
        if generated_file.read_bytes() != expected_file.read_bytes():
            raise ValueError(f"committed visual-provider bundle differs: {filename}")


def _model_source(
    source: Mapping[str, Any],
    *,
    cache_directory: Path,
    label: str,
) -> tuple[str, str]:
    repository = _require_literal_string(
        source.get("repository"), name=f"{label} repository"
    )
    required = source.get("required_members")
    if not isinstance(required, list):
        raise ValueError(f"{label} required_members must be an array")
    expected_value = source.get("expected_revision")
    expected = (
        None
        if expected_value is None
        else _require_revision(expected_value, name=f"{label} expected revision")
    )
    revision = resolve_cached_snapshot_revision(
        cache_directory,
        repository=repository,
        required_members=cast(list[str], required),
        expected_revision=expected,
    )
    return repository, revision


def build_preflight(
    *,
    spec_path: str | Path,
    metric_policy_path: str | Path,
    prob4d_checkout: str | Path,
    motioncrafter_checkout: str | Path,
    cache_directory: str | Path,
    output_directory: str | Path,
    expected_bundle_directory: str | Path | None = None,
) -> dict[str, Any]:
    spec_source = Path(spec_path).resolve()
    policy_source = Path(metric_policy_path).resolve()
    spec = load_preflight_spec(spec_source)
    policy = load_metric_prior_policy(policy_source)
    policy_binding = spec["metric_frame_prior_policy"]
    if not isinstance(policy_binding, Mapping):
        raise ValueError("metric_frame_prior_policy binding must be an object")
    if policy_binding.get("artifact_id") != policy["artifact_id"]:
        raise ValueError("preflight spec binds another metric-frame policy")
    if Path(cast(str, policy_binding.get("path"))).as_posix() != (
        "protocols/locks/"
        "deform360_official_hub_visuotactile_v1_metric_frame_prior_policy.json"
    ):
        raise ValueError("preflight metric-frame policy path changed")

    prob4d_root = Path(prob4d_checkout).resolve()
    motioncrafter_root = Path(motioncrafter_checkout).resolve()
    provider_spec = spec["provider"]
    motion_spec = spec["motioncrafter"]
    if not isinstance(provider_spec, Mapping) or not isinstance(motion_spec, Mapping):
        raise ValueError("provider and motioncrafter specs must be objects")
    provider_revision = _require_revision(
        provider_spec.get("revision"), name="Prob4D revision"
    )
    motioncrafter_revision = _require_revision(
        motion_spec.get("revision"), name="MotionCrafter revision"
    )
    if git_head(prob4d_root, name="Prob4D") != provider_revision:
        raise ValueError("Prob4D checkout differs from the frozen revision")
    if git_head(motioncrafter_root, name="MotionCrafter") != motioncrafter_revision:
        raise ValueError("MotionCrafter checkout differs from the frozen revision")
    require_clean_checkout(prob4d_root, name="Prob4D")
    require_clean_checkout(motioncrafter_root, name="MotionCrafter")

    cache = Path(cache_directory).expanduser().resolve()
    if not cache.is_dir():
        raise ValueError(f"Hugging Face cache is not a directory: {cache}")
    model_sources = motion_spec.get("model_sources")
    if not isinstance(model_sources, Mapping) or set(model_sources) != {
        "unet",
        "vae",
        "image_vae",
        "base_pipeline",
    }:
        raise ValueError("MotionCrafter model source roles changed")
    resolved = {
        role: _model_source(
            cast(Mapping[str, Any], model_sources[role]),
            cache_directory=cache,
            label=role,
        )
        for role in ("unet", "vae", "image_vae", "base_pipeline")
    }
    if resolved["unet"] != resolved["vae"]:
        raise ValueError(
            "MotionCrafter UNet and geometry VAE must share one model revision"
        )

    provider_v2 = _import_from_checkout(prob4d_root, "prob4d.provider_v2")
    provider_attestation = _import_from_checkout(
        prob4d_root, "prob4d.provider_attestation"
    )
    model_module = _import_from_checkout(prob4d_root, "prob4d.motioncrafter_models")

    provider_manifest = provider_v2.prob4d_provider_manifest(
        provider_revision=provider_revision
    )
    attestation = provider_attestation.build_provider_attestation(
        provider_manifest=provider_manifest,
        provider_revision=provider_revision,
        export_mode="exploratory",
        calibration_compatibility_validated=False,
        calibration_artifact_ids={
            "gauge_artifact_id": None,
            "point_artifact_id": None,
        },
        covariance_root_mode=cast(str, provider_spec["covariance_root_mode"]),
        composition_jacobian_mode=cast(str, provider_spec["composition_jacobian_mode"]),
        runtime_revision={
            "expected_revision": provider_revision,
            "observed_revision": provider_revision,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    )
    model_set = model_module.PinnedMotionCrafterModelSet.inspect(
        model_type=cast(str, motion_spec["model_type"]),
        unet_reference=resolved["unet"][0],
        unet_revision=resolved["unet"][1],
        vae_reference=resolved["vae"][0],
        vae_revision=resolved["vae"][1],
        image_vae_reference=resolved["image_vae"][0],
        image_vae_revision=resolved["image_vae"][1],
        base_pipeline_reference=resolved["base_pipeline"][0],
        base_pipeline_revision=resolved["base_pipeline"][1],
    )

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_json(temporary / "provider-manifest.json", provider_manifest)
        _write_json(temporary / "provider-attestation.json", attestation)
        model_record = {
            "schema": "bayesian-phystwin/deform360-motioncrafter-model-set-binding-v1",
            "schema_version": 1,
            "model_set_id": model_set.set_sha256,
            "model_set_manifest": json.loads(model_set.manifest_json),
            "cached_revisions": {
                role: revision for role, (_, revision) in sorted(resolved.items())
            },
            "cache_path_recorded": False,
            "selected_raw_payloads_opened": False,
            "target_outcomes_used": False,
        }
        _write_json(temporary / "motioncrafter-model-set.json", model_record)

        from bayesian_phystwin.deform360_visual_provider_lock import (
            Deform360VisualProviderLockV1,
            load_deform360_visual_provider_lock,
            save_deform360_visual_provider_lock,
        )
        from bayesian_phystwin.prob4d_provider_attestation import (
            validate_prob4d_provider_attestation,
        )

        validate_prob4d_provider_attestation(
            attestation,
            source_revision=provider_revision,
            require_claim_bearing=False,
        )
        attestation_sha256 = file_sha256(temporary / "provider-attestation.json")
        gauge = spec["gauge_covariance"]
        if not isinstance(gauge, Mapping):
            raise ValueError("gauge_covariance must be an object")
        lock = Deform360VisualProviderLockV1(
            provider_revision=provider_revision,
            provider_manifest_id=cast(str, provider_manifest["manifest_id"]),
            provider_attestation_sha256=attestation_sha256,
            motioncrafter_revision=motioncrafter_revision,
            model_set_id=model_set.set_sha256,
            root_seed=_require_integer(
                motion_spec.get("root_seed"), name="root_seed", minimum=0
            ),
            seed_policy=_require_literal_string(
                motion_spec.get("seed_policy"), name="seed_policy"
            ),
            window_size=_require_integer(
                motion_spec.get("window_size"), name="window_size", minimum=2
            ),
            overlap=_require_integer(
                motion_spec.get("overlap"), name="overlap", minimum=0
            ),
            height=_require_integer(
                motion_spec.get("height"), name="height", minimum=1
            ),
            width=_require_integer(motion_spec.get("width"), name="width", minimum=1),
            storage_dtype=cast(str, motion_spec.get("storage_dtype")),
            initial_metric_frame_prior_id=cast(str, policy["artifact_id"]),
            additional_metric_anchor_policy=cast(
                str, spec["additional_metric_anchor_policy"]
            ),
            max_gauge_rank=_require_integer(
                gauge.get("max_rank"), name="max_gauge_rank", minimum=1
            ),
            minimum_retained_gauge_trace=_require_probability(
                gauge.get("minimum_retained_trace"),
                name="minimum_retained_gauge_trace",
            ),
            metadata={
                "preflight_spec_id": spec["artifact_id"],
                "metric_frame_prior_policy_path": cast(str, policy_binding["path"]),
                "model_source_revisions": {
                    role: revision for role, (_, revision) in sorted(resolved.items())
                },
                "provider_attestation_mode": "exploratory-precalibration",
                "selection_role": "calibration-and-confirmation",
                "selected_payload_access": "none",
            },
        )
        save_deform360_visual_provider_lock(
            temporary / "visual-provider-lock.json", lock
        )
        loaded = load_deform360_visual_provider_lock(
            temporary / "visual-provider-lock.json"
        )
        if loaded != lock:
            raise ValueError("visual-provider lock failed independent round trip")

        summary = {
            "schema": "bayesian-phystwin/deform360-visual-provider-preflight-result-v1",
            "schema_version": 1,
            "preflight_spec_id": spec["artifact_id"],
            "metric_frame_prior_policy_id": policy["artifact_id"],
            "provider_revision": provider_revision,
            "provider_manifest_id": provider_manifest["manifest_id"],
            "provider_attestation_sha256": attestation_sha256,
            "motioncrafter_revision": motioncrafter_revision,
            "model_set_id": model_set.set_sha256,
            "model_source_revisions": {
                role: revision for role, (_, revision) in sorted(resolved.items())
            },
            "visual_provider_lock_id": lock.artifact_id,
            "selected_raw_payloads_opened": False,
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "claim_boundary": spec["claim_boundary"],
        }
        _write_json(temporary / "summary.json", summary)
        _write_sha256sums(temporary)

        if expected_bundle_directory is not None:
            _verify_expected_bundle(
                temporary, Path(expected_bundle_directory).resolve()
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return cast(dict[str, Any], summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--prob4d-checkout", type=Path, required=True)
    parser.add_argument("--motioncrafter-checkout", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-bundle-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = build_preflight(
        spec_path=arguments.spec,
        metric_policy_path=arguments.metric_prior_policy,
        prob4d_checkout=arguments.prob4d_checkout,
        motioncrafter_checkout=arguments.motioncrafter_checkout,
        cache_directory=arguments.cache_dir,
        output_directory=arguments.output_dir,
        expected_bundle_directory=arguments.expected_bundle_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
