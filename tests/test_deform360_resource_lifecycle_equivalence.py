from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/development/analyze_deform360_resource_lifecycle_equivalence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deform360_resource_lifecycle_equivalence", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
equivalence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = equivalence
SPEC.loader.exec_module(equivalence)


def _structured_vertices(
    offset: float = 0.0,
    *,
    count: int = 6,
    permutation: np.ndarray | None = None,
    quaternion_sign: float = 1.0,
) -> np.ndarray:
    dtype = np.dtype(
        [(name, "<f4") for name in equivalence.EXPECTED_PLY_FIELDS], align=False
    )
    vertices = np.zeros(count, dtype=dtype)
    coordinate = np.linspace(-0.04, 0.04, count, dtype=np.float32)
    vertices["x"] = coordinate + offset
    vertices["y"] = coordinate[::-1] * 0.5
    vertices["z"] = 1.0 + coordinate * 0.25
    for index in range(3):
        vertices[f"f_dc_{index}"] = 0.1 * (index + 1) + coordinate
    for index in range(45):
        vertices[f"f_rest_{index}"] = 0.001 * index + coordinate * 0.01
    vertices["opacity"] = np.linspace(-1.0, 1.0, count)
    vertices["scale_0"] = -3.0 + coordinate
    vertices["scale_1"] = -3.1 - coordinate
    vertices["scale_2"] = -3.2 + coordinate * 0.5
    vertices["rot_0"] = quaternion_sign
    vertices["rot_1"] = quaternion_sign * coordinate
    vertices["rot_2"] = quaternion_sign * coordinate[::-1]
    vertices["rot_3"] = quaternion_sign * 0.02
    if permutation is not None:
        vertices = vertices[permutation].copy()
    return vertices


def _ply_bytes(vertices: np.ndarray, *, rename_last: str | None = None) -> bytes:
    names = list(vertices.dtype.names or ())
    if rename_last is not None:
        names[-1] = rename_last
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment synthetic test",
        f"element vertex {len(vertices)}",
        *(f"property float {name}" for name in names),
        "end_header",
        "",
    ]
    return "\n".join(header).encode("ascii") + vertices.tobytes()


def _binding(payload: bytes, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _global_snapshot(base: int = 1000) -> dict[str, Any]:
    return {
        "event_writers_object_id": base + 1,
        "event_writer_ids": [base + 2],
        "event_storage_object_id": base + 3,
        "event_storage_ids": [base + 4],
        "global_buffer_object_id": base + 5,
        "global_buffer_items": [["'fixed'", base + 6]],
        "profiler_object_id": base + 7,
        "profiler_ids": [base + 8],
        "pytorch_profiler_id": None,
    }


def _write_dataset(dataset: Path) -> None:
    dataset.mkdir()
    (dataset / "images").mkdir()
    for index in range(21):
        (dataset / f"images/camera-{index:02d}.png").write_bytes(
            f"camera-{index:02d}".encode()
        )
    (dataset / "seed.ply").write_bytes(b"fixed seed")
    transforms = json.loads(_canonical_transforms().decode("utf-8"))
    transforms["ply_file_path"] = str((dataset / "seed.ply").resolve())
    (dataset / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")


def _write_fit_evidence(
    tmp_path: Path,
    mode: str,
    ply: Path,
    *,
    profile: str = "historical-0db",
) -> Path:
    dataset = tmp_path / f"{mode}-{ply.stem}-dataset"
    _write_dataset(dataset)
    adapter = (
        Path(__file__).parents[1]
        / "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    )
    smoke = equivalence._signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "contract_sha256": equivalence.PINNED_GSPLAT_SMOKE_CONTRACT_SHA256,
            "physical_gpu_index": equivalence.PROFILE_PHYSICAL_GPU_INDEX[profile],
            "logical_device": "cuda:0",
            "gpu_name": equivalence.PINNED_GPU_NAME,
            "compute_capability": "8.9",
            "python_version": "3.12",
            "torch_version": equivalence.PINNED_TORCH_VERSION,
            "torch_cuda_version": equivalence.PINNED_TORCH_CUDA_VERSION,
            "gsplat_version": equivalence.PINNED_GSPLAT_VERSION,
            "extension_path": str(equivalence.PINNED_GSPLAT_EXTENSION_PATH),
            "extension_sha256": equivalence.PINNED_GSPLAT_EXTENSION_SHA256,
            "extension_loaded_and_retained": True,
            "nvcc_visible": False,
            "ninja_visible": False,
            "target_or_outcome_path_accessed": False,
            "predicates": {
                "render_shape": [1, 16, 16, 3],
                "alpha_shape": [1, 16, 16, 1],
                "positive_radius_count": 2,
                "gradient_groups_finite_and_nonzero": [
                    "colors",
                    "means",
                    "opacities",
                    "quats",
                    "scales",
                ],
                "forward_finite_nonempty_nonzero": True,
                "backward_complete": True,
                "cuda_synchronized": True,
            },
        }
    )
    before = {
        "file_descriptor_count": 10,
        "task_count": 2,
        "rss_kib": 100,
        "rlimit_nofile_soft": 1024,
        "rlimit_nofile_hard": 4096,
    }
    evidence = equivalence._signed(
        {
            "schema_version": 1,
            "artifact_kind": equivalence.FIT_EVIDENCE_KIND,
            "qualification_id": equivalence.FIT_QUALIFICATION_IDS[profile],
            "variant": mode,
            "passed": True,
            "parameters": {
                "iterations": equivalence.FIT_ITERATIONS,
                "seed": equivalence.FIT_SEED,
            },
            "runtime": {
                "seed": 0,
                "python_random_seeded": True,
                "numpy_seeded": True,
                "torch_cpu_seeded": True,
                "torch_cuda_seeded": True,
                "torch_version": equivalence.PINNED_TORCH_VERSION,
                "torch_cuda_version": equivalence.PINNED_TORCH_CUDA_VERSION,
                "cuda_device_name": equivalence.PINNED_GPU_NAME,
                "cuda_device_count": 1,
                "python_version": "3.12.0 synthetic",
            },
            "gsplat_runtime_smoke": {
                "adapter_source": equivalence._bound_file(adapter, label="adapter"),
                "evidence": smoke,
                "evidence_artifact_sha256": smoke["artifact_sha256"],
            },
            "dataset": str(dataset.resolve()),
            "output": equivalence._bound_file(ply, label="fit output"),
            "resource_boundary": {"before": before, "after": dict(before)},
            "global_state": {
                "before": _global_snapshot(),
                "after": _global_snapshot(),
                "restored": True,
            },
            "predicates": {
                "output_created": True,
                "wrapped_fit_requires_global_restoration": True,
                "rlimit_nofile_soft_is_1024": True,
                "rlimit_nofile_unchanged": True,
                "gsplat_runtime_smoke_validated_and_retained": True,
            },
            "formal_held_path_supplied": False,
        }
    )
    path = tmp_path / f"{mode}-{ply.stem}-fit-evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _cloud(
    pairing_id: str,
    mode: str,
    offset: float,
    *,
    permutation: np.ndarray | None = None,
    quaternion_sign: float = 1.0,
) -> Any:
    vertices = _structured_vertices(
        offset,
        permutation=permutation,
        quaternion_sign=quaternion_sign,
    )
    payload = _ply_bytes(vertices)
    return equivalence._parse_gaussian_ply(
        payload,
        mode=mode,
        pairing_id=pairing_id,
        binding=_binding(payload, f"/{mode}-{pairing_id}.ply"),
    )


def _canonical_transforms(*, count: int = 21, distortion: float = 0.0) -> bytes:
    camera_to_world = np.diag([1.0, -1.0, -1.0, 1.0]).tolist()
    payload = {
        "camera_model": "OPENCV",
        "frames": [
            {
                "file_path": f"./images/camera-{index:02d}.png",
                "transform_matrix": camera_to_world,
                "w": 64,
                "h": 48,
                "fl_x": 50.0 + index,
                "fl_y": 51.0 + index,
                "cx": 32.0,
                "cy": 24.0,
                "k1": distortion,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
            }
            for index in range(count)
        ],
    }
    return json.dumps(payload).encode("utf-8")


def _cameras() -> Any:
    return equivalence._load_canonical_cameras(_canonical_transforms())


def _fake_renderer(calls: list[tuple[str, str]]) -> Any:
    def render(cloud: Any, cameras: Any) -> tuple[np.ndarray, np.ndarray]:
        calls.append((cloud.mode, cloud.pairing_id))
        shape = (21, cameras.height, cameras.width)
        center = float(np.mean(cloud.xyz[:, 0]))
        rgb = np.full((*shape, 3), center, dtype=np.float32)
        alpha = np.full((*shape, 1), 0.5, dtype=np.float32)
        return rgb, alpha

    return render


def _cloud_cohort(
    wrapped_order: tuple[int, ...] = (0, 1, 2, 3, 4),
    *,
    wrapped_extra_offset: float = 0.0,
) -> dict[str, list[Any]]:
    offsets = (0.0, 0.01, 0.02, 0.03, 0.04)
    identifiers = tuple(f"repeat-{index}" for index in range(5))
    return {
        "original": [
            _cloud(identifier, "original", offsets[index])
            for index, identifier in enumerate(identifiers)
        ],
        "wrapped": [
            _cloud(
                identifiers[index],
                "wrapped",
                offsets[wrapped_order[index]] + wrapped_extra_offset,
            )
            for index in range(5)
        ],
    }


def test_binary_gaussian_ply_requires_exact_finite_62_field_schema() -> None:
    vertices = _structured_vertices()
    payload = _ply_bytes(vertices)
    cloud = equivalence._parse_gaussian_ply(
        payload,
        mode="original",
        pairing_id="repeat-0",
        binding=_binding(payload, "/repeat-0.ply"),
    )

    assert len(cloud.schema) == 62
    assert cloud.xyz.shape == (6, 3)
    assert cloud.sh.shape == (6, 48)
    assert np.all(np.isfinite(cloud.opacity_probability))
    assert np.allclose(np.linalg.norm(cloud.quaternions, axis=1), 1.0)

    with pytest.raises(ValueError, match="field names or order"):
        equivalence._parse_gaussian_ply(
            _ply_bytes(vertices, rename_last="wrong"),
            mode="original",
            pairing_id="bad-schema",
            binding=_binding(payload, "/bad-schema.ply"),
        )

    vertices["f_rest_17"][2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        equivalence._parse_gaussian_ply(
            _ply_bytes(vertices),
            mode="original",
            pairing_id="nonfinite",
            binding=_binding(payload, "/nonfinite.ply"),
        )

    double_header = payload.replace(b"property float x\n", b"property double x\n", 1)
    with pytest.raises(ValueError, match="literal float/f4"):
        equivalence._parse_gaussian_ply(
            double_header,
            mode="original",
            pairing_id="double-field",
            binding=_binding(double_header, "/double-field.ply"),
        )

    nonzero_normal = _structured_vertices()
    nonzero_normal["ny"][0] = np.float32(1.0e-6)
    nonzero_payload = _ply_bytes(nonzero_normal)
    with pytest.raises(ValueError, match="inert normal field ny"):
        equivalence._parse_gaussian_ply(
            nonzero_payload,
            mode="original",
            pairing_id="nonzero-normal",
            binding=_binding(nonzero_payload, "/nonzero-normal.ply"),
        )


def test_geometry_metrics_are_permutation_and_quaternion_sign_invariant() -> None:
    left = _cloud("repeat-0", "original", 0.0)
    right = _cloud(
        "repeat-0",
        "wrapped",
        0.0,
        permutation=np.asarray([4, 1, 5, 0, 3, 2]),
        quaternion_sign=-1.0,
    )

    metrics = equivalence._pair_geometry_metrics(left, right)

    assert set(metrics) == set(equivalence.PAIR_METRIC_NAMES) - {
        "rgb_rmse",
        "alpha_rmse",
    }
    assert max(metrics.values()) < 2.0e-7


def test_geometry_metrics_include_symmetric_count_and_attribute_differences() -> None:
    left = _cloud("repeat-0", "original", 0.0)
    vertices = _structured_vertices(0.02, count=3)
    vertices["opacity"] += 0.5
    vertices["scale_1"] += 0.2
    vertices["f_dc_2"] += 0.3
    payload = _ply_bytes(vertices)
    right = equivalence._parse_gaussian_ply(
        payload,
        mode="wrapped",
        pairing_id="repeat-0",
        binding=_binding(payload, "/wrapped.ply"),
    )

    metrics = equivalence._pair_geometry_metrics(left, right)

    assert metrics["relative_count_delta"] == 0.5
    assert metrics["xyz_distance_max_m"] > 0.0
    assert metrics["opacity_probability_abs_mean"] > 0.0
    assert metrics["log_scale_vector_l2_mean"] > 0.0
    assert metrics["sh_vector_l2_mean"] > 0.0


def test_canonical_transforms_are_exactly_21_undistorted_downscaled_cameras() -> None:
    cameras = equivalence._load_canonical_cameras(_canonical_transforms())

    assert cameras.viewmats.shape == (21, 4, 4)
    assert cameras.intrinsics.shape == (21, 3, 3)
    assert (cameras.width, cameras.height) == (16, 12)
    assert np.array_equal(cameras.viewmats[0], np.eye(4, dtype=np.float32))
    assert cameras.intrinsics[0, 0, 0] == 12.5

    with pytest.raises(ValueError, match="exactly 21"):
        equivalence._load_canonical_cameras(_canonical_transforms(count=20))
    with pytest.raises(ValueError, match="nonzero distortion"):
        equivalence._load_canonical_cameras(_canonical_transforms(distortion=0.01))


def test_exact_equality_is_primary_and_every_ply_is_rendered_once() -> None:
    calls: list[tuple[str, str]] = []

    result = equivalence._analyze_clouds(
        _cloud_cohort(), _cameras(), _fake_renderer(calls)
    )

    assert len(calls) == 10
    assert len(set(calls)) == 10
    assert result["render_execution"]["each_ply_rendered_exactly_once"] is True
    assert result["secondary_distributional_gate"]["pair_counts"] == {
        "within_original": 10,
        "within_wrapped": 10,
        "cross_mode": 25,
    }
    assert (
        result["decision"]["exact_matched_structured_array_equality_primary_passed"]
        is True
    )
    assert result["decision"]["accepted"] is True
    assert result["decision"]["acceptance_basis"] == ("exact-structured-array-equality")


def test_secondary_gate_accepts_same_distribution_when_pairing_is_permuted() -> None:
    calls: list[tuple[str, str]] = []

    result = equivalence._analyze_clouds(
        _cloud_cohort((1, 2, 3, 4, 0)),
        _cameras(),
        _fake_renderer(calls),
    )

    assert (
        result["decision"]["exact_matched_structured_array_equality_primary_passed"]
        is False
    )
    assert result["secondary_distributional_gate"]["passed"] is True
    assert result["decision"]["accepted"] is True
    assert result["decision"]["acceptance_basis"] == (
        "secondary-distributional-envelope"
    )


def test_secondary_gate_rejects_cross_mode_shift_outside_within_envelope() -> None:
    calls: list[tuple[str, str]] = []

    result = equivalence._analyze_clouds(
        _cloud_cohort(wrapped_extra_offset=1.0),
        _cameras(),
        _fake_renderer(calls),
    )

    assert result["secondary_distributional_gate"]["passed"] is False
    assert result["decision"]["accepted"] is False
    assert result["decision"]["acceptance_basis"] == "rejected"
    assert (
        result["secondary_distributional_gate"]["per_metric"]["xyz_distance_mean_m"][
            "passed"
        ]
        is False
    )


def test_gate_boundary_is_inclusive_and_rejects_nonfinite_metrics() -> None:
    def record(value: float) -> dict[str, Any]:
        return {"metrics": {name: value for name in equivalence.PAIR_METRIC_NAMES}}

    inclusive_boundary = float(
        np.quantile(np.arange(10, dtype=np.float64), 0.95, method="linear")
    )
    groups = {
        "within_original": [record(float(index)) for index in range(10)],
        "within_wrapped": [record(float(index)) for index in range(10)],
        "cross_mode": [record(inclusive_boundary) for _ in range(25)],
    }
    gate, _ = equivalence._evaluate_gate(groups)
    assert gate["passed"] is True

    groups["cross_mode"][0]["metrics"]["rgb_rmse"] = float("nan")
    with pytest.raises(ValueError, match="metric rgb_rmse is invalid"):
        equivalence._evaluate_gate(groups)


def test_signed_manifest_binds_source_files_gpu_and_matching_repeat_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "canonical-source"
    _write_dataset(source)
    monkeypatch.setattr(
        equivalence, "CANONICAL_PUBLIC_SOURCE_DATASET", source.resolve()
    )
    modes: dict[str, list[dict[str, Any]]] = {"original": [], "wrapped": []}
    for mode in modes:
        for index in range(6):
            path = tmp_path / f"{mode}-{index}.ply"
            path.write_bytes(_ply_bytes(_structured_vertices(0.01 * index)))
            binding = equivalence._bound_file(path, label="test PLY")
            evidence_path = _write_fit_evidence(tmp_path, mode, path)
            evidence_binding = equivalence._bound_file(
                evidence_path, label="fit evidence"
            )
            evidence = equivalence._load_signed_json(
                evidence_path, label="fit evidence"
            )
            modes[mode].append(
                {
                    "pairing_id": f"repeat-{index}",
                    "ply": {
                        key: binding[key] for key in ("path", "size_bytes", "sha256")
                    },
                    "fit_evidence": {
                        key: evidence_binding[key]
                        for key in ("path", "size_bytes", "sha256")
                    },
                    "dataset_input_inventory": equivalence._dataset_input_inventory(
                        evidence["dataset"]
                    ),
                }
            )
    source_inventory = equivalence._dataset_input_inventory(source)
    representative = source_inventory["raw_transforms"]
    transform_binding = {
        key: representative[key] for key in ("path", "size_bytes", "sha256")
    }
    representative_path = Path(representative["path"])
    repository = Path(__file__).parents[1].resolve()
    generator_sources = {
        path: {
            **equivalence._bound_file(repository / path, label="generator source"),
            "git_blob_oid": pinned["git_blob_oid"],
        }
        for path, pinned in equivalence.GENERATOR_SOURCE_BINDINGS.items()
    }
    generator_git = {
        "path": str(repository),
        "head": equivalence.GENERATOR_CODE_HEAD,
        "tree": equivalence.GENERATOR_CODE_TREE,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }
    analyzer_git = {
        "path": str(repository),
        "head": "1" * 40,
        "tree": "2" * 40,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }
    manifest = equivalence._signed(
        {
            "schema_version": 1,
            "artifact_kind": equivalence.MANIFEST_KIND,
            "analysis_id": equivalence.ANALYSIS_ID,
            "expected_environment": {
                "generator_profile": "historical-0db",
                "physical_gpu_index": 0,
                "generator_code": {
                    "profile": "historical-0db",
                    "qualification_id": equivalence.FIT_QUALIFICATION_IDS[
                        "historical-0db"
                    ],
                    "physical_gpu_index": 0,
                    "git": generator_git,
                    "sources": generator_sources,
                },
                "analyzer_code": analyzer_git,
                "deform360_git_head": equivalence.PINNED_DEFORM360_REVISION,
                "deform360_git_tree": equivalence.PINNED_DEFORM360_TREE,
                "python_freeze_sha256": equivalence.PINNED_PYTHON_FREEZE_SHA256,
                "python_tree_manifest_sha256": (
                    equivalence.PINNED_PYTHON_TREE_MANIFEST_SHA256
                ),
            },
            "canonical_source_dataset": source_inventory,
            "canonical_transforms": {
                "raw_representative": {
                    key: transform_binding[key]
                    for key in ("path", "size_bytes", "sha256")
                },
                "normalized": equivalence._normalized_transforms_descriptor(
                    representative_path.read_bytes()
                ),
            },
            "modes": modes,
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = equivalence._load_signed_json(manifest_path, label="manifest")
    with monkeypatch.context() as preflight_guard:
        preflight_guard.setattr(
            equivalence,
            "_dataset_input_inventory",
            lambda _path: pytest.fail(
                "dataset census ran during manifest environment preflight"
            ),
        )
        preflight_guard.setattr(
            equivalence,
            "_verified_descriptor",
            lambda *_args, **_kwargs: pytest.fail(
                "cohort descriptor reader ran during manifest environment preflight"
            ),
        )
        preflight = equivalence._validate_manifest_environment(loaded)
        assert preflight["physical_gpu_index"] == 0
    inputs, transform_path, _ = equivalence._validate_manifest(loaded)

    assert transform_path == representative_path.resolve()
    assert [record.pairing_id for record in inputs["original"]] == [
        f"repeat-{index}" for index in range(6)
    ]
    wrong_gpu = copy.deepcopy(manifest)
    wrong_gpu["expected_environment"]["physical_gpu_index"] = 1
    with pytest.raises(ValueError, match="physical GPU differs"):
        equivalence._validate_manifest(equivalence._signed(wrong_gpu))

    source_image = source / "images/camera-00.png"
    source_payload = source_image.read_bytes()
    source_image.write_bytes(b"tampered-source")
    with pytest.raises(ValueError, match="source dataset identity"):
        equivalence._validate_manifest(manifest)
    source_image.write_bytes(source_payload)

    repeated_dataset = Path(modes["wrapped"][0]["dataset_input_inventory"]["root"])
    repeated_image = repeated_dataset / "images/camera-00.png"
    repeated_payload = repeated_image.read_bytes()
    repeated_image.write_bytes(b"tampered-repeat")
    with pytest.raises(ValueError, match="dataset input inventory changed"):
        equivalence._validate_manifest(manifest)
    repeated_image.write_bytes(repeated_payload)

    modes["wrapped"][5]["pairing_id"] = "different"
    invalid = equivalence._signed({**manifest, "modes": modes})
    with pytest.raises(ValueError, match="pairing IDs differ"):
        equivalence._validate_manifest(invalid)
    modes["wrapped"][5]["pairing_id"] = "repeat-5"

    nonmember_path = tmp_path / "nonmember-transforms.json"
    nonmember_path.write_bytes(_canonical_transforms())
    nonmember = equivalence._bound_file(nonmember_path, label="nonmember transforms")
    invalid_canonical = copy.deepcopy(manifest)
    invalid_canonical["canonical_transforms"]["raw_representative"] = {
        key: nonmember[key] for key in ("path", "size_bytes", "sha256")
    }
    invalid_canonical = equivalence._signed(invalid_canonical)
    with pytest.raises(ValueError, match="public source transforms"):
        equivalence._validate_manifest(invalid_canonical)


def test_dataset_inventory_accepts_only_exact_materialized_canonical_seed_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical"
    repeated = tmp_path / "repeated"
    _write_dataset(canonical)
    _write_dataset(repeated)
    canonical_transforms = (canonical / "transforms.json").read_bytes()
    (repeated / "transforms.json").write_bytes(canonical_transforms)
    monkeypatch.setattr(
        equivalence,
        "CANONICAL_PUBLIC_SOURCE_DATASET",
        canonical.resolve(),
    )

    canonical_inventory = equivalence._dataset_input_inventory(canonical)
    repeated_inventory = equivalence._dataset_input_inventory(repeated)
    assert repeated_inventory["seed_reference"]["canonical_absolute_alias_used"] is True
    assert (
        repeated_inventory["content_artifact_sha256"]
        == canonical_inventory["content_artifact_sha256"]
    )
    assert (
        repeated_inventory["seed_reference"]["canonical_target"]["sha256"]
        == repeated_inventory["seed_reference"]["materialized_copy"]["sha256"]
    )

    repeated_seed = repeated / "seed.ply"
    repeated_seed.write_bytes(b"different materialized seed")
    with pytest.raises(ValueError, match="materialized copy differs"):
        equivalence._dataset_input_inventory(repeated)
    repeated_seed.write_bytes((canonical / "seed.ply").read_bytes())

    outside_seed = tmp_path / "outside-seed.ply"
    outside_seed.write_bytes((canonical / "seed.ply").read_bytes())
    transforms = json.loads(canonical_transforms.decode("utf-8"))
    transforms["ply_file_path"] = str(outside_seed.resolve())
    (repeated / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the dataset"):
        equivalence._dataset_input_inventory(repeated)


def test_fit_evidence_rejects_wrong_generator_adapter_path(tmp_path: Path) -> None:
    ply = tmp_path / "fit.ply"
    ply.write_bytes(_ply_bytes(_structured_vertices()))
    evidence = _write_fit_evidence(tmp_path, "original", ply)
    evidence_binding = equivalence._bound_file(evidence, label="fit evidence")
    ply_binding = equivalence._bound_file(ply, label="fit PLY")

    with pytest.raises(ValueError, match="adapter path or content"):
        equivalence._validate_fit_evidence(
            {key: evidence_binding[key] for key in ("path", "size_bytes", "sha256")},
            mode="original",
            ply_binding=ply_binding,
            expected_adapter_path=tmp_path / "wrong-adapter.py",
            generator_profile="historical-0db",
            expected_physical_gpu_index=0,
        )


def test_fit_evidence_rejects_schema_type_and_gpu_tampering(tmp_path: Path) -> None:
    ply = tmp_path / "fit.ply"
    ply.write_bytes(_ply_bytes(_structured_vertices()))
    base_path = _write_fit_evidence(tmp_path, "wrapped", ply)
    base = equivalence._load_signed_json(base_path, label="fit evidence")
    ply_binding = equivalence._bound_file(ply, label="fit PLY")
    adapter = (
        Path(__file__).parents[1]
        / "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ).resolve()

    mutations: list[tuple[str, Any, str]] = []
    extra_top = copy.deepcopy(base)
    extra_top["unexpected"] = True
    mutations.append(("extra-top", extra_top, "fields changed"))
    bool_iterations = copy.deepcopy(base)
    bool_iterations["parameters"]["iterations"] = True
    mutations.append(("bool-iterations", bool_iterations, "strict integer"))
    extra_runtime = copy.deepcopy(base)
    extra_runtime["runtime"]["unexpected"] = 1
    mutations.append(("extra-runtime", extra_runtime, "runtime fields changed"))
    sparse_global = copy.deepcopy(base)
    sparse_global["global_state"]["before"] = {}
    mutations.append(("sparse-global", sparse_global, "globals before fields"))
    wrong_gpu = copy.deepcopy(base)
    wrong_gpu["gsplat_runtime_smoke"]["evidence"]["physical_gpu_index"] = 1
    wrong_gpu["gsplat_runtime_smoke"]["evidence"] = equivalence._signed(
        wrong_gpu["gsplat_runtime_smoke"]["evidence"]
    )
    wrong_gpu["gsplat_runtime_smoke"]["evidence_artifact_sha256"] = wrong_gpu[
        "gsplat_runtime_smoke"
    ]["evidence"]["artifact_sha256"]
    mutations.append(("wrong-gpu", wrong_gpu, "frozen runtime fields changed"))

    for name, changed, message in mutations:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(equivalence._signed(changed)), encoding="utf-8")
        descriptor = equivalence._bound_file(path, label=name)
        with pytest.raises(ValueError, match=message):
            equivalence._validate_fit_evidence(
                {key: descriptor[key] for key in ("path", "size_bytes", "sha256")},
                mode="wrapped",
                ply_binding=ply_binding,
                expected_adapter_path=adapter,
                generator_profile="historical-0db",
                expected_physical_gpu_index=0,
            )


def test_same_as_analyzer_v2_fit_evidence_requires_gpu_one(tmp_path: Path) -> None:
    ply = tmp_path / "fit-v2.ply"
    ply.write_bytes(_ply_bytes(_structured_vertices()))
    evidence_path = _write_fit_evidence(
        tmp_path, "wrapped", ply, profile="same-as-analyzer"
    )
    evidence_binding = equivalence._bound_file(evidence_path, label="v2 evidence")
    ply_binding = equivalence._bound_file(ply, label="v2 PLY")
    adapter = (
        Path(__file__).parents[1]
        / "src/bayesian_phystwin/deform360_held_v8_gsplat_runtime.py"
    ).resolve()
    descriptor = {
        key: evidence_binding[key] for key in ("path", "size_bytes", "sha256")
    }

    _, evidence = equivalence._validate_fit_evidence(
        descriptor,
        mode="wrapped",
        ply_binding=ply_binding,
        expected_adapter_path=adapter,
        generator_profile="same-as-analyzer",
        expected_physical_gpu_index=1,
    )
    assert evidence["qualification_id"].endswith("v2")
    assert evidence["gsplat_runtime_smoke"]["evidence"]["physical_gpu_index"] == 1

    with pytest.raises(ValueError, match="identity changed"):
        equivalence._validate_fit_evidence(
            descriptor,
            mode="wrapped",
            ply_binding=ply_binding,
            expected_adapter_path=adapter,
            generator_profile="historical-0db",
            expected_physical_gpu_index=0,
        )


def test_exact_gsplat_smoke_contract_rejects_resigned_runtime_changes(
    tmp_path: Path,
) -> None:
    ply = tmp_path / "fit.ply"
    ply.write_bytes(_ply_bytes(_structured_vertices()))
    evidence_path = _write_fit_evidence(tmp_path, "wrapped", ply)
    evidence = equivalence._load_signed_json(evidence_path, label="fit evidence")
    smoke = evidence["gsplat_runtime_smoke"]["evidence"]

    assert (
        equivalence._validate_gsplat_smoke(
            smoke, label="smoke", expected_physical_gpu_index=0
        )
        == smoke
    )
    mutations = {
        "contract_sha256": "0" * 64,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "extension_sha256": "f" * 64,
        "nvcc_visible": True,
    }
    for field, value in mutations.items():
        changed = equivalence._signed({**smoke, field: value})
        with pytest.raises(ValueError, match="frozen runtime fields changed"):
            equivalence._validate_gsplat_smoke(
                changed, label="smoke", expected_physical_gpu_index=0
            )
    with pytest.raises(ValueError, match="frozen runtime fields changed"):
        equivalence._validate_gsplat_smoke(
            smoke, label="smoke", expected_physical_gpu_index=1
        )
    bool_schema = equivalence._signed({**smoke, "schema_version": True})
    with pytest.raises(ValueError, match="strict integer"):
        equivalence._validate_gsplat_smoke(
            bool_schema, label="smoke", expected_physical_gpu_index=0
        )


def test_host_and_live_cuda_device_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(equivalence.socket, "gethostname", lambda: "workstation2")
    assert equivalence._validate_execution_host() == "workstation2"
    monkeypatch.setattr(equivalence.socket, "gethostname", lambda: "gpuserver4090")
    with pytest.raises(ValueError, match="not workstation2"):
        equivalence._validate_execution_host()

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(_index: int) -> str:
            return equivalence.PINNED_GPU_NAME

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (8, 9)

    torch = SimpleNamespace(
        cuda=Cuda(),
        __version__=equivalence.PINNED_TORCH_VERSION,
        version=SimpleNamespace(cuda=equivalence.PINNED_TORCH_CUDA_VERSION),
    )
    assert (
        equivalence._validate_live_torch_device(torch)["visible_cuda_device_count"] == 1
    )
    monkeypatch.setattr(Cuda, "device_count", staticmethod(lambda: 2))
    with pytest.raises(ValueError, match="exactly one GPU"):
        equivalence._validate_live_torch_device(torch)


def test_python_flags_environment_and_import_paths_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags = SimpleNamespace(
        isolated=1,
        no_user_site=1,
        dont_write_bytecode=1,
        safe_path=True,
        ignore_environment=1,
    )
    assert equivalence._validate_process_flags(flags)["isolated"] == 1
    assert equivalence._validate_process_flags(flags)["safe_path"] is True
    flags.safe_path = 1
    with pytest.raises(ValueError, match="safe_path"):
        equivalence._validate_process_flags(flags)

    environment = {
        **equivalence.REQUIRED_EXECUTION_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": "1",
        "TMPDIR": str(tmp_path),
    }
    assert (
        equivalence._validate_execution_environment(1, environment)[
            "cuda_visible_devices"
        ]
        == "1"
    )
    assert environment["CUDA_MODULE_LOADING"] == "LAZY"
    eager_environment = {**environment, "CUDA_MODULE_LOADING": "EAGER"}
    with pytest.raises(ValueError, match="CUDA_MODULE_LOADING"):
        equivalence._validate_execution_environment(1, eager_environment)
    with pytest.raises(ValueError, match="manifest physical GPU"):
        equivalence._validate_execution_environment(0, environment)
    with pytest.raises(ValueError, match="forbidden"):
        equivalence._validate_execution_environment(
            1, {**environment, "PYTHONPATH": "/tmp/injected"}
        )

    monkeypatch.setattr(equivalence.sys, "path", ["relative"])
    with pytest.raises(ValueError, match="relative"):
        equivalence._import_path_binding(Path(__file__).parents[1].resolve())


def test_renderer_sys_path_restoration_accepts_only_pinned_transients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    vendor = runtime / "lib/python3.12/site-packages/setuptools/_vendor"
    vendor.mkdir(parents=True)
    temporary_root = tmp_path / "isolated-tmp"
    temporary_root.mkdir()
    temporary_entry = temporary_root / "tmpfixed123"
    temporary_entry.mkdir()
    baseline = ["/validated/source", "/validated/runtime"]
    monkeypatch.setattr(equivalence, "PINNED_PYTHON_RUNTIME", runtime.resolve())
    monkeypatch.setenv("TMPDIR", str(temporary_root.resolve()))
    monkeypatch.setattr(
        equivalence.sys,
        "path",
        [*baseline, str(temporary_entry.resolve()), str(vendor.resolve())],
    )

    evidence = equivalence._restore_renderer_sys_path(
        baseline, require_pinned_mutation=True
    )
    assert evidence["restored_exactly"] is True
    assert [entry["role"] for entry in evidence["transient_entries"]] == [
        "isolated_temporary_import_path",
        "pinned_setuptools_vendor_import_path",
    ]
    assert equivalence.sys.path == baseline

    equivalence.sys.path[:] = [*baseline, str(tmp_path / "unexpected")]
    with pytest.raises(ValueError, match="entry count changed"):
        equivalence._restore_renderer_sys_path(baseline, require_pinned_mutation=True)
    equivalence.sys.path[:] = baseline
    no_mutation = equivalence._restore_renderer_sys_path(
        baseline, require_pinned_mutation=False
    )
    assert no_mutation["transient_entries"] == []


def test_live_pip_freeze_is_canonical_sorted_and_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized = b"alpha==1\nzeta==2\n"
    monkeypatch.setattr(
        equivalence,
        "PINNED_PYTHON_FREEZE_SHA256",
        hashlib.sha256(normalized).hexdigest(),
    )
    monkeypatch.setattr(
        equivalence.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=b"zeta==2\nalpha==1\n"
        ),
    )
    binding = equivalence._live_pip_freeze_binding(Path("/pinned/python"))
    assert binding["normalized_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert binding["normalized_line_count"] == 2

    monkeypatch.setattr(
        equivalence.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"changed==1\n"),
    )
    with pytest.raises(ValueError, match="live pip freeze differs"):
        equivalence._live_pip_freeze_binding(Path("/pinned/python"))


def test_runtime_tree_verifier_hashes_exact_manifest_paths_modes_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    bin_directory = root / "bin"
    lib_directory = root / "lib"
    bin_directory.mkdir(parents=True)
    lib_directory.mkdir()
    frozen_file = lib_directory / "frozen.txt"
    frozen_file.write_bytes(b"frozen-runtime-content\n")
    (bin_directory / "python").symlink_to("/usr/bin/python3")
    (bin_directory / "python3").symlink_to("python")
    (bin_directory / "python3.12").symlink_to("python")
    frozen_file.chmod(0o444)
    bin_directory.chmod(0o555)
    lib_directory.chmod(0o555)
    root.chmod(0o555)

    entries: list[dict[str, Any]] = []
    for relative in (
        "bin",
        "bin/python",
        "bin/python3",
        "bin/python3.12",
        "lib",
        "lib/frozen.txt",
    ):
        path = root / relative
        observed = equivalence.os.lstat(path)
        mode = f"{equivalence.stat.S_IMODE(observed.st_mode):04o}"
        if equivalence.stat.S_ISDIR(observed.st_mode):
            entry = {"path": relative, "mode": mode, "type": "directory"}
        elif equivalence.stat.S_ISLNK(observed.st_mode):
            entry = {
                "path": relative,
                "mode": mode,
                "type": "symlink",
                "target": equivalence.os.readlink(path),
            }
        else:
            payload = path.read_bytes()
            entry = {
                "path": relative,
                "mode": mode,
                "type": "file",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        entries.append(entry)
    entries.sort(key=lambda value: equivalence.os.fsencode(value["path"]))
    freeze_sha = "f" * 64
    manifest = {
        "artifact_kind": equivalence.PINNED_PYTHON_TREE_MANIFEST_KIND,
        "root_path": str(root),
        "python_pip_freeze_sorted_sha256": freeze_sha,
        "entry_counts": {"directory": 2, "file": 1, "symlink": 3},
        "total_regular_file_bytes": frozen_file.stat().st_size,
        "tree_sha256": hashlib.sha256(
            json.dumps(
                entries,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest(),
        "entries": entries,
    }
    manifest_path = root.parent / f"{root.name}.tree-manifest.json"
    raw_manifest = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    manifest_path.write_bytes(raw_manifest)
    manifest_path.chmod(0o400)
    monkeypatch.setattr(equivalence, "PINNED_PYTHON_RUNTIME", root)
    monkeypatch.setattr(equivalence, "PINNED_PYTHON_TREE_MANIFEST", manifest_path)
    monkeypatch.setattr(
        equivalence,
        "PINNED_PYTHON_TREE_MANIFEST_SHA256",
        hashlib.sha256(raw_manifest).hexdigest(),
    )
    monkeypatch.setattr(equivalence, "PINNED_PYTHON_FREEZE_SHA256", freeze_sha)

    try:
        binding = equivalence._validate_runtime_tree()
        assert binding["entry_counts"] == {
            "directory": 2,
            "file": 1,
            "symlink": 3,
        }
        assert binding["all_entry_metadata_and_file_hashes_verified"] is True

        lib_directory.chmod(0o755)
        frozen_file.chmod(0o644)
        frozen_file.write_bytes(b"tampered-runtime-content\n")
        frozen_file.chmod(0o444)
        lib_directory.chmod(0o555)
        with pytest.raises(ValueError, match="metadata changed|content changed"):
            equivalence._validate_runtime_tree()
    finally:
        root.chmod(0o755)
        bin_directory.chmod(0o755)
        lib_directory.chmod(0o755)
        frozen_file.chmod(0o644)
        manifest_path.chmod(0o600)


def test_generator_profiles_are_exact_and_same_profile_is_not_arbitrary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[1].resolve()

    def git_record(head: str, tree: str) -> dict[str, Any]:
        return {
            "path": str(repository),
            "head": head,
            "tree": tree,
            "clean": True,
            "ordinary_untracked_file_count": 0,
            "ignored_untracked_file_count": 0,
        }

    blob_ids = {
        path: value["git_blob_oid"]
        for path, value in equivalence.GENERATOR_SOURCE_BINDINGS.items()
    }
    monkeypatch.setattr(
        equivalence,
        "_git_blob_oid",
        lambda _root, relative: blob_ids.get(relative, "a" * 40),
    )
    real_bound_file = equivalence._bound_file

    def source_binding(path: Path, *, label: str) -> dict[str, Any]:
        try:
            relative = path.resolve().relative_to(repository).as_posix()
        except ValueError:
            return real_bound_file(path, label=label)
        pinned = equivalence.GENERATOR_SOURCE_BINDINGS[relative]
        return {
            "path": str(path.resolve()),
            "size_bytes": pinned["size_bytes"],
            "sha256": pinned["sha256"],
            "mode_octal": "0444",
        }

    monkeypatch.setattr(equivalence, "_bound_file", source_binding)

    historical_git = git_record(
        equivalence.GENERATOR_CODE_HEAD, equivalence.GENERATOR_CODE_TREE
    )
    monkeypatch.setattr(equivalence, "_git_binding", lambda _root: historical_git)
    historical = equivalence._generator_checkout_binding(
        repository,
        profile="historical-0db",
        analyzer_root=repository,
        analyzer_git=historical_git,
    )
    assert historical["qualification_id"].endswith("v1")
    assert historical["physical_gpu_index"] == 0

    final_git = git_record("a" * 40, "b" * 40)
    monkeypatch.setattr(equivalence, "_git_binding", lambda _root: final_git)
    final = equivalence._generator_checkout_binding(
        repository,
        profile="same-as-analyzer",
        analyzer_root=repository,
        analyzer_git=final_git,
    )
    assert final["qualification_id"].endswith("v2")
    assert final["physical_gpu_index"] == 1

    with pytest.raises(ValueError, match="differs from analyzer"):
        equivalence._generator_checkout_binding(
            repository,
            profile="same-as-analyzer",
            analyzer_root=repository.parent,
            analyzer_git=final_git,
        )


def test_formal_paths_and_symlink_inputs_are_rejected(tmp_path: Path) -> None:
    formal = (
        equivalence.FORMAL_HELD_PARENT
        / "held-v8-attempt-4-withdrawn-postbarrier/private.ply"
    )
    with pytest.raises(ValueError, match="formal held root"):
        equivalence._assert_nonheld_path(formal, label="PLY", must_exist=False)

    target = tmp_path / "target.ply"
    target.write_bytes(b"ply")
    linked = tmp_path / "linked.ply"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        equivalence._read_regular_nofollow(linked, label="PLY")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.ply"
    nested.write_bytes(b"ply")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(ValueError, match="symlink ancestor"):
        equivalence._read_regular_nofollow(linked_parent / "nested.ply", label="PLY")

    hardlink = tmp_path / "hardlink.ply"
    try:
        hardlink.hardlink_to(target)
    except OSError:
        pytest.skip("hardlinks unavailable")
    with pytest.raises(ValueError, match="hardlink"):
        equivalence._read_regular_nofollow(hardlink, label="PLY")


def test_prepare_and_analyze_reject_static_protected_output_roots_before_git_or_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1].resolve()
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    canonical_source = tmp_path / "canonical-source"
    canonical_source.mkdir()
    monkeypatch.setattr(
        equivalence, "CANONICAL_PUBLIC_SOURCE_DATASET", canonical_source.resolve()
    )
    monkeypatch.setattr(
        equivalence,
        "_git_binding",
        lambda _root: pytest.fail("Git must not run before output-root rejection"),
    )

    forbidden = repository / "forbidden-equivalence-output.json"
    with pytest.raises(ValueError, match="protected source/input root"):
        equivalence.prepare_manifest(
            original=[],
            wrapped=[],
            canonical_transforms=canonical_source / "transforms.json",
            output_path=forbidden,
            code_root=repository,
            generator_code_root=repository,
            deform360_root=deform360,
            generator_profile="historical-0db",
        )
    with pytest.raises(ValueError, match="protected source/input root"):
        equivalence.analyze(
            tmp_path / "absent-manifest.json",
            forbidden,
            code_root=repository,
            generator_code_root=repository,
            deform360_root=deform360,
        )
    with pytest.raises(ValueError, match="protected source/input root"):
        equivalence._reject_output_within_roots(
            canonical_source,
            [canonical_source],
            label="test output",
        )


def test_prepare_and_analyze_reject_repeat_input_roots_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1].resolve()
    canonical_source = tmp_path / "canonical-source"
    canonical_source.mkdir()
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    repeat_root = tmp_path / "repeat-root"
    repeat_root.mkdir()
    ply = repeat_root / "fit.ply"
    ply.write_bytes(_ply_bytes(_structured_vertices()))
    evidence = _write_fit_evidence(repeat_root, "original", ply)
    fit = equivalence._load_signed_json(evidence, label="fit evidence")
    monkeypatch.setattr(
        equivalence, "CANONICAL_PUBLIC_SOURCE_DATASET", canonical_source.resolve()
    )
    monkeypatch.setattr(
        equivalence,
        "_git_binding",
        lambda _root: pytest.fail("Git must not run before repeat-root rejection"),
    )

    with pytest.raises(ValueError, match="protected source/input root"):
        equivalence.prepare_manifest(
            original=[["repeat-0", str(ply), str(evidence)]],
            wrapped=[],
            canonical_transforms=canonical_source / "transforms.json",
            output_path=repeat_root / "manifest.json",
            code_root=repository,
            generator_code_root=repository,
            deform360_root=deform360,
            generator_profile="historical-0db",
        )

    ply_binding = equivalence._bound_file(ply, label="repeat PLY")
    evidence_binding = equivalence._bound_file(evidence, label="repeat evidence")
    record = {
        "pairing_id": "repeat-0",
        "ply": {key: ply_binding[key] for key in ("path", "size_bytes", "sha256")},
        "fit_evidence": {
            key: evidence_binding[key] for key in ("path", "size_bytes", "sha256")
        },
        "dataset_input_inventory": {"root": str(fit["dataset"])},
    }
    manifest = equivalence._signed(
        {
            "modes": {"original": [record], "wrapped": [record]},
            "expected_environment": {},
        }
    )
    manifest_path = tmp_path / "repeat-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = Path(str(fit["dataset"]))
    with pytest.raises(ValueError, match="protected source/input root"):
        equivalence.analyze(
            manifest_path,
            dataset / "result.json",
            code_root=repository,
            generator_code_root=repository,
            deform360_root=deform360,
        )


def test_signed_json_writer_is_exclusive_nofollow_and_fsyncs_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_flags: list[int] = []
    fsync_modes: list[int] = []
    real_open = equivalence.os.open
    real_fsync = equivalence.os.fsync

    def tracked_open(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        open_flags.append(flags)
        return real_open(path, flags, mode, **kwargs)

    def tracked_fsync(descriptor: int) -> None:
        fsync_modes.append(equivalence.os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(equivalence.os, "open", tracked_open)
    monkeypatch.setattr(equivalence.os, "fsync", tracked_fsync)
    payload = equivalence._signed({"schema_version": 1, "value": "fixed"})
    path = equivalence._write_new_json(tmp_path / "evidence.json", payload)

    assert equivalence._load_signed_json(path, label="evidence") == payload
    assert path.stat().st_mode & 0o222 == 0
    assert all(flags & getattr(equivalence.os, "O_NOFOLLOW", 0) for flags in open_flags)
    assert any(equivalence.stat.S_ISREG(mode) for mode in fsync_modes)
    assert any(equivalence.stat.S_ISDIR(mode) for mode in fsync_modes)
    with pytest.raises(ValueError, match="already exists"):
        equivalence._write_new_json(path, payload)


def test_signed_json_publication_retains_complete_file_if_parent_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = equivalence.os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        if equivalence.stat.S_ISDIR(equivalence.os.fstat(descriptor).st_mode):
            raise OSError("synthetic parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(equivalence.os, "fsync", fail_parent_fsync)
    payload = equivalence._signed({"schema_version": 1, "value": "durable"})
    path = tmp_path / "durable.json"
    with pytest.raises(OSError, match="parent fsync"):
        equivalence._write_new_json(path, payload)
    assert equivalence._load_signed_json(path, label="retained evidence") == payload
    assert not list(tmp_path.glob(".durable.json.partial-*"))


def test_analyze_cli_publishes_signed_rejection_and_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected_analyze(_manifest: Path, output_path: Path, **_kwargs: Any) -> Path:
        return equivalence._write_new_json(
            output_path,
            equivalence._signed(
                {
                    "schema_version": 1,
                    "artifact_kind": equivalence.RESULT_KIND,
                    "decision": {"accepted": False},
                }
            ),
        )

    monkeypatch.setattr(equivalence, "analyze", rejected_analyze)
    output = tmp_path / "rejection.json"
    exit_code = equivalence.main(
        [
            "analyze",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--code-root",
            str(tmp_path),
            "--generator-code-root",
            str(tmp_path),
            "--deform360-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 3
    result = equivalence._load_signed_json(output, label="rejection")
    assert result["decision"]["accepted"] is False


@pytest.mark.parametrize("failure_stage", ["source", "runtime", "execution", "aot"])
def test_invalid_launch_preflight_never_touches_cohort_inputs(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[1].resolve()
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    cohort_ply = cohort / "repeat.ply"
    cohort_ply.write_bytes(b"never read")
    cohort_evidence = cohort / "fit-evidence.json"
    cohort_evidence.write_bytes(b"never read")
    cohort_dataset = cohort / "dataset"
    cohort_dataset.mkdir()

    code_git = {
        "path": str(repository),
        "head": "1" * 40,
        "tree": "2" * 40,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }
    deform_git = {
        **code_git,
        "path": str(deform360),
        "head": equivalence.PINNED_DEFORM360_REVISION,
        "tree": equivalence.PINNED_DEFORM360_TREE,
    }
    generator = {"profile": "synthetic-generator"}
    expected = {
        "generator_profile": "historical-0db",
        "physical_gpu_index": 0,
        "analyzer_code": code_git,
        "generator_code": generator,
        "deform360_git_head": equivalence.PINNED_DEFORM360_REVISION,
        "deform360_git_tree": equivalence.PINNED_DEFORM360_TREE,
        "python_freeze_sha256": equivalence.PINNED_PYTHON_FREEZE_SHA256,
        "python_tree_manifest_sha256": (equivalence.PINNED_PYTHON_TREE_MANIFEST_SHA256),
    }
    manifest = equivalence._signed(
        {
            "expected_environment": expected,
            "modes": {
                mode: [
                    {
                        "pairing_id": "repeat-0",
                        "ply": {"path": str(cohort_ply)},
                        "fit_evidence": {"path": str(cohort_evidence)},
                        "dataset_input_inventory": {"root": str(cohort_dataset)},
                    }
                ]
                for mode in ("original", "wrapped")
            },
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cohort_touches: list[str] = []
    real_reader = equivalence._read_regular_nofollow
    real_bound_file = equivalence._bound_file

    def guarded_reader(path: Any, *, label: str) -> bytes:
        candidate = Path(path).resolve(strict=False)
        if candidate == cohort or cohort in candidate.parents:
            cohort_touches.append(f"reader:{candidate}")
            pytest.fail("cohort reader ran before launch preflight completed")
        return real_reader(path, label=label)

    def git_binding(root: Path) -> dict[str, Any]:
        if failure_stage == "source":
            raise ValueError("synthetic invalid source")
        return code_git if Path(root) == repository else deform_git

    def runtime_binding() -> dict[str, Any]:
        if failure_stage == "runtime":
            raise ValueError("synthetic invalid runtime")
        return {}

    def execution_binding(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if failure_stage == "execution":
            raise ValueError("synthetic invalid execution environment")
        return {"host": "workstation2"}

    def guarded_bound_file(path: Any, *, label: str = "file") -> dict[str, Any]:
        return real_bound_file(path, label=label)

    def aot_binding() -> dict[str, Any]:
        if failure_stage == "aot":
            raise ValueError("synthetic invalid AOT")
        return {
            "path": str(equivalence.PINNED_GSPLAT_EXTENSION_PATH),
            "size_bytes": 1,
            "sha256": equivalence.PINNED_GSPLAT_EXTENSION_SHA256,
            "mode_octal": "0444",
        }

    def cohort_manifest_validation(_manifest: Any, **_kwargs: Any) -> Any:
        cohort_touches.append("manifest-input-validation")
        pytest.fail("cohort manifest phase ran after a failed launch preflight")

    def cohort_census(path: Any) -> Any:
        cohort_touches.append(f"census:{path}")
        pytest.fail("cohort census ran before launch preflight completed")

    monkeypatch.setattr(equivalence, "_read_regular_nofollow", guarded_reader)
    monkeypatch.setattr(
        equivalence, "_validate_manifest_environment", lambda _: expected
    )
    monkeypatch.setattr(equivalence, "_git_binding", git_binding)
    monkeypatch.setattr(equivalence, "_historical_generator_binding", lambda _: {})
    monkeypatch.setattr(
        equivalence, "_generator_checkout_binding", lambda *_args, **_kwargs: generator
    )
    monkeypatch.setattr(
        equivalence, "_validate_analyzer_runtime_record", lambda value, **_: value
    )
    monkeypatch.setattr(equivalence.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(equivalence, "_execution_binding", execution_binding)
    monkeypatch.setattr(equivalence, "_bound_file", guarded_bound_file)
    monkeypatch.setattr(equivalence, "_validated_pinned_aot_binding", aot_binding)
    monkeypatch.setattr(equivalence, "_validate_manifest", cohort_manifest_validation)
    monkeypatch.setattr(equivalence, "_dataset_input_inventory", cohort_census)

    expected_message = {
        "source": "invalid source",
        "runtime": "invalid runtime",
        "execution": "invalid execution environment",
        "aot": "invalid AOT",
    }[failure_stage]
    with pytest.raises(ValueError, match=expected_message):
        equivalence._capture_run_state(
            manifest_file=manifest_path,
            code=repository,
            generator_code_root=repository,
            deform360=deform360,
            runtime_binding=runtime_binding,
        )
    assert cohort_touches == []


def test_pinned_aot_preflight_requires_read_only_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "frozen-aot"
    parent.mkdir()
    extension = parent / "gsplat_cuda.so"
    extension.write_bytes(b"frozen extension")
    extension.chmod(0o444)
    parent.chmod(0o555)
    monkeypatch.setattr(equivalence, "PINNED_GSPLAT_EXTENSION_PATH", extension)
    monkeypatch.setattr(
        equivalence,
        "PINNED_GSPLAT_EXTENSION_SHA256",
        hashlib.sha256(b"frozen extension").hexdigest(),
    )

    try:
        binding = equivalence._validated_pinned_aot_binding()
        assert binding["mode_octal"] == "0444"

        extension.chmod(0o644)
        with pytest.raises(ValueError, match="AOT file mode changed"):
            equivalence._validated_pinned_aot_binding()

        extension.chmod(0o444)
        parent.chmod(0o755)
        with pytest.raises(ValueError, match="AOT parent mode changed"):
            equivalence._validated_pinned_aot_binding()
    finally:
        parent.chmod(0o755)
        extension.chmod(0o644)


def test_analyze_entry_preflight_does_not_read_cohort_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1].resolve()
    generator = tmp_path / "generator"
    deform360 = tmp_path / "deform360"
    cohort = tmp_path / "cohort"
    for directory in (generator, deform360, cohort):
        directory.mkdir()
    cohort_ply = cohort / "repeat.ply"
    cohort_ply.write_bytes(b"never read")
    cohort_evidence = cohort / "fit-evidence.json"
    cohort_evidence.write_bytes(b"never read")
    cohort_dataset = cohort / "dataset"
    cohort_dataset.mkdir()
    manifest = equivalence._signed(
        {
            "expected_environment": {
                "generator_profile": "historical-0db",
                "physical_gpu_index": 0,
            },
            "modes": {
                mode: [
                    {
                        "pairing_id": "repeat-0",
                        "ply": {"path": str(cohort_ply)},
                        "fit_evidence": {"path": str(cohort_evidence)},
                        "dataset_input_inventory": {"root": str(cohort_dataset)},
                    }
                ]
                for mode in ("original", "wrapped")
            },
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    real_reader = equivalence._read_regular_nofollow
    cohort_touches: list[Path] = []

    def guarded_reader(path: Any, *, label: str) -> bytes:
        candidate = Path(path).resolve(strict=False)
        if candidate == cohort or cohort in candidate.parents:
            cohort_touches.append(candidate)
            pytest.fail("public analyze entry read cohort before live preflight")
        return real_reader(path, label=label)

    def failed_live_preflight(**_kwargs: Any) -> Any:
        raise ValueError("synthetic invalid live preflight")

    monkeypatch.setattr(equivalence, "_read_regular_nofollow", guarded_reader)
    monkeypatch.setattr(equivalence, "_install_controlled_code_source", lambda _: None)
    monkeypatch.setattr(equivalence, "_capture_run_state", failed_live_preflight)
    with pytest.raises(ValueError, match="invalid live preflight"):
        equivalence.analyze(
            manifest_path,
            tmp_path / "result.json",
            code_root=repository,
            generator_code_root=generator,
            deform360_root=deform360,
        )
    assert cohort_touches == []


def test_analyze_rejects_any_post_render_validation_state_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1].resolve()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    transforms = tmp_path / "transforms.json"
    transforms.write_bytes(_canonical_transforms())
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    canonical_source = tmp_path / "canonical-source"
    canonical_source.mkdir()
    monkeypatch.setattr(
        equivalence, "CANONICAL_PUBLIC_SOURCE_DATASET", canonical_source.resolve()
    )
    expected = {
        "generator_profile": "historical-0db",
        "physical_gpu_index": 0,
    }
    manifest = {
        "expected_environment": expected,
        "modes": {"original": [], "wrapped": []},
        "artifact_sha256": "a" * 64,
    }
    adapter = {"path": "adapter", "size_bytes": 1, "sha256": "b" * 64}
    live_device = {"gpu_name": equivalence.PINNED_GPU_NAME}
    before = {
        "execution": {"host": "workstation2", "live_device": live_device},
        "source": {"analyzer_gsplat_adapter": adapter},
        "runtime": {"stable": True},
    }
    after = copy.deepcopy(before)
    after["runtime"]["stable"] = False
    captures = iter((before, after))

    def capture(**_kwargs: Any) -> Any:
        return (
            next(captures),
            {"original": [], "wrapped": []},
            transforms,
            expected,
            manifest,
        )

    monkeypatch.setattr(
        equivalence, "_load_signed_json", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(
        equivalence, "_install_controlled_code_source", lambda _code: None
    )
    monkeypatch.setattr(equivalence, "_capture_run_state", capture)
    monkeypatch.setattr(
        equivalence, "_load_canonical_cameras", lambda _payload: object()
    )
    monkeypatch.setattr(equivalence, "_load_clouds", lambda _inputs: {})
    monkeypatch.setattr(
        equivalence,
        "_validate_gsplat_smoke",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        equivalence,
        "_analyze_clouds",
        lambda *_args: {"decision": {"accepted": True}},
    )

    def renderer_factory(_code: Path, gpu: int) -> Any:
        return (lambda *_args: None), {
            "host": "workstation2",
            "physical_gpu_index": gpu,
            "adapter_source": adapter,
            "smoke_evidence": {},
            "live_device": live_device,
        }

    with pytest.raises(ValueError, match="state changed during render"):
        equivalence.analyze(
            manifest_path,
            tmp_path / "result.json",
            code_root=repository,
            generator_code_root=repository,
            deform360_root=deform360,
            renderer_factory=renderer_factory,
            runtime_binding=lambda: {},
        )
