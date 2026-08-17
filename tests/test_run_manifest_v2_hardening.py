from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import bayesian_phystwin.run_manifest_v2 as run_manifest_v2_module
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import ArtifactDigest, sha256_file
from bayesian_phystwin.run_manifest_v2 import (
    RunManifestV2,
    load_run_manifest,
    load_run_manifest_v2,
    verify_run_manifest_artifacts,
    write_run_manifest,
)


def _artifact(path: Path, *, name: str, role: str, root: Path) -> ArtifactDigest:
    return ArtifactDigest(
        name=name,
        role=role,  # type: ignore[arg-type]
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _manifest(root: Path, **changes: object) -> RunManifestV2:
    source = root / "input.txt"
    result = root / "output.txt"
    source.write_text("input\n", encoding="utf-8")
    result.write_text("output\n", encoding="utf-8")
    values: dict[str, object] = {
        "run_id": "unit-test-v2",
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "revision": "a" * 40,
        "dirty": False,
        "related_repositories": (
            RepositoryState(
                repository="IPS-Stuttgart/Prob4D",
                revision="b" * 40,
                dirty=False,
                role="observation",
            ),
        ),
        "command": ("bpt", "benchmark", "synthetic"),
        "classification": "infrastructure",
        "statistical_unit": "test case",
        "information_boundary": {"causal_frame_stop": 10},
        "configuration": {"nested": {"seeds": [1, 2]}},
        "seeds": (1, 2),
        "inputs": (_artifact(source, name="input", role="input", root=root),),
        "outputs": (_artifact(result, name="output", role="output", root=root),),
        "package_versions": {"bayesian-phystwin": "0.4.0"},
        "runtime_environment": {
            "python_version": "3.13.5",
            "accelerators": ["test"],
        },
        "claim_ids": ("bpt.infrastructure.run_manifest_v2",),
        "method_freeze_id": "method-v1",
        "protocol_id": "protocol-v1",
        "split_id": "split-v1",
        "baseline_id": "baseline-v1",
        "created_utc": "2026-07-26T20:00:00+02:00",
        "notes": "",
    }
    values.update(changes)
    return RunManifestV2(**values)  # type: ignore[arg-type]


def _released_content_id(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_valid_v2_content_ids_keep_the_released_canonical_form(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    assert manifest.evidence_fingerprint == _released_content_id(
        manifest.scientific_descriptor()
    )
    assert manifest.manifest_id == _released_content_id(manifest.descriptor())


def test_round_trip_freezes_nested_metadata_without_rewriting_timestamp(
    tmp_path: Path,
) -> None:
    source_configuration = {"nested": {"seeds": [1, 2]}}
    manifest = _manifest(tmp_path, configuration=source_configuration)
    fingerprint = manifest.evidence_fingerprint

    assert manifest.created_utc == "2026-07-26T20:00:00+02:00"
    source_configuration["nested"]["seeds"].append(3)
    assert manifest.configuration["nested"]["seeds"] == [1, 2]
    assert manifest.evidence_fingerprint == fingerprint

    with pytest.raises(TypeError, match="immutable"):
        manifest.configuration["nested"]["seeds"].append(3)
    with pytest.raises(TypeError, match="immutable"):
        manifest.runtime_environment["new"] = "value"  # type: ignore[index]

    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    expected_bytes = (
        json.dumps(
            manifest.as_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert path.read_bytes() == expected_bytes
    loaded = load_run_manifest_v2(path)
    assert loaded == manifest
    verify_run_manifest_artifacts(loaded, root=tmp_path)


def test_writer_is_no_clobber_by_default_and_atomic_on_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    first = _manifest(tmp_path)
    write_run_manifest(path, first)
    original = path.read_bytes()

    second = replace(first, notes="replacement")
    with pytest.raises(FileExistsError):
        write_run_manifest(path, second)
    assert path.read_bytes() == original

    write_run_manifest(path, second, overwrite=True)
    assert load_run_manifest_v2(path).notes == "replacement"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))

    with pytest.raises(ValueError, match="overwrite must be boolean"):
        write_run_manifest(path, second, overwrite=1)  # type: ignore[arg-type]


def test_loader_rejects_duplicate_keys_before_digest_validation(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest(tmp_path)
    write_run_manifest(path, manifest)
    text = path.read_text(encoding="utf-8")
    duplicate = text.replace(
        '  "run_id": "unit-test-v2",',
        '  "run_id": "unit-test-v2",\n  "run_id": "duplicate",',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key: run_id"):
        load_run_manifest_v2(path)
    with pytest.raises(ValueError, match="duplicate JSON object key: run_id"):
        load_run_manifest(path)

    nonfinite = text.replace(
        '"causal_frame_stop": 10',
        '"causal_frame_stop": NaN',
        1,
    )
    path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant.*NaN"):
        load_run_manifest_v2(path)


def test_loader_rejects_digest_coercion_and_noncanonical_digest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_run_manifest(path, _manifest(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["manifest_id"] = payload["manifest_id"].upper()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_id must be a lowercase SHA-256"):
        load_run_manifest_v2(path)

    payload["manifest_id"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_id must be a non-empty string"):
        load_run_manifest_v2(path)


def test_boolean_seeds_and_other_scalar_coercions_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match=r"seeds\[0\] must be an integer"):
        replace(manifest, seeds=(True,))
    with pytest.raises(ValueError, match=r"command\[1\].*literal string"):
        replace(manifest, command=("bpt", 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="package version name.*literal string"):
        replace(manifest, package_versions={1: "0.4.0"})  # type: ignore[dict-item]


def test_artifact_paths_must_be_canonical_and_repository_relative(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    source = manifest.inputs[0]
    outside = ArtifactDigest(
        name=source.name,
        role=source.role,
        path="../outside.txt",
        sha256=source.sha256,
        size_bytes=source.size_bytes,
    )

    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        replace(manifest, inputs=(outside,))

    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"][0]["path"] = "../outside.txt"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        load_run_manifest_v2(path)


def test_record_subclasses_are_rejected_at_the_public_boundary(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    class ArtifactSubclass(ArtifactDigest):
        pass

    source = manifest.inputs[0]
    subclassed_artifact = ArtifactSubclass(
        name=source.name,
        role=source.role,
        path=source.path,
        sha256=source.sha256,
        size_bytes=source.size_bytes,
    )
    with pytest.raises(ValueError, match="ArtifactDigest values"):
        replace(manifest, inputs=(subclassed_artifact,))

    class RepositorySubclass(RepositoryState):
        pass

    state = manifest.related_repositories[0]
    subclassed_state = RepositorySubclass(
        repository=state.repository,
        revision=state.revision,
        dirty=state.dirty,
        role=state.role,
    )
    with pytest.raises(ValueError, match="RepositoryState values"):
        replace(manifest, related_repositories=(subclassed_state,))

    class ManifestSubclass(RunManifestV2):
        pass

    subclassed_manifest = ManifestSubclass(**manifest.__dict__)
    with pytest.raises(TypeError, match="exact RunManifestV2"):
        write_run_manifest(tmp_path / "subclass.json", subclassed_manifest)


def test_metadata_keys_and_nonfinite_values_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="literal string object keys"):
        replace(manifest, configuration={1: "value"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite JSON values"):
        replace(manifest, runtime_environment={"value": float("nan")})


def test_loaded_artifact_scalars_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_run_manifest(path, _manifest(tmp_path))
    original = json.loads(path.read_text(encoding="utf-8"))

    payload = dict(original)
    payload["inputs"] = [dict(original["inputs"][0])]
    payload["inputs"][0]["sha256"] = payload["inputs"][0]["sha256"].upper()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact sha256 must be a lowercase SHA-256"):
        load_run_manifest_v2(path)

    payload = dict(original)
    payload["outputs"] = [dict(original["outputs"][0])]
    payload["outputs"][0]["size_bytes"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact size_bytes must be an integer"):
        load_run_manifest_v2(path)


def test_loaded_boolean_seed_is_rejected_before_fingerprint_comparison(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_run_manifest(path, _manifest(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seeds"] = [True]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"seeds\[0\] must be an integer"):
        load_run_manifest_v2(path)


def test_direct_builtin_mutation_is_detected_before_rehash(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    dict.__setitem__(manifest.configuration, "tampered", True)

    with pytest.raises(RuntimeError, match="backing storage was mutated"):
        _ = manifest.evidence_fingerprint


def test_exact_field_guard_rejects_nonstring_fields() -> None:
    manifest_from_payload = run_manifest_v2_module._manifest_from_payload

    with pytest.raises(ValueError, match="literal string fields"):
        manifest_from_payload({1: "value"})


def test_fail_closed_constructor_branches_are_covered(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(
        ValueError,
        match="command must contain nonempty strings",
    ):
        replace(manifest, command=())
    with pytest.raises(ValueError, match="repository must use owner/name"):
        replace(
            manifest,
            repository=" IPS-Stuttgart/BayesianPhysTwin",
        )
    with pytest.raises(ValueError, match="unknown run classification"):
        replace(manifest, classification="unknown")
    with pytest.raises(
        ValueError,
        match="created_utc must include a timezone",
    ):
        replace(manifest, created_utc="2026-07-26T20:00:00")
    with pytest.raises(
        ValueError,
        match="claim_ids must be unique nonempty identifiers",
    ):
        replace(manifest, claim_ids=("claim", "claim"))
    with pytest.raises(
        ValueError,
        match="package_versions must be a mapping",
    ):
        replace(manifest, package_versions=())


def test_invalid_artifact_role_is_rejected_by_v2_boundary(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    artifact = replace(manifest.inputs[0])
    object.__setattr__(artifact, "role", "sidecar")

    with pytest.raises(
        ValueError,
        match="artifact role must be 'input' or 'output'",
    ):
        replace(manifest, inputs=(artifact,))


def test_writer_rejects_post_write_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    altered = replace(manifest, notes="altered after publication")
    monkeypatch.setattr(
        run_manifest_v2_module,
        "load_run_manifest_v2",
        lambda _path: altered,
    )

    with pytest.raises(
        RuntimeError,
        match="failed post-write verification",
    ):
        write_run_manifest(tmp_path / "mismatch.json", manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "schema_name",
            "bayesian_phystwin.unsupported",
            "unsupported run-manifest schema",
        ),
        (
            "schema_version",
            3,
            "unsupported run-manifest version",
        ),
    ],
)
def test_loader_rejects_schema_dispatch_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "manifest.json"
    write_run_manifest(path, _manifest(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_run_manifest_v2(path)
