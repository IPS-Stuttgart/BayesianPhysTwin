from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin import deform360_held_v8_protocol as protocol


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/held/seal_deform360_resource_lifecycle_qualification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "resource_qualification_v2_sealer", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)

QUALIFIER_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/development/qualify_deform360_resource_lifecycle.py"
)
QUALIFIER_SPEC = importlib.util.spec_from_file_location(
    "resource_qualification_v2_operator_for_sealer_test", QUALIFIER_SCRIPT
)
assert QUALIFIER_SPEC is not None and QUALIFIER_SPEC.loader is not None
qualification = importlib.util.module_from_spec(QUALIFIER_SPEC)
sys.modules[QUALIFIER_SPEC.name] = qualification
QUALIFIER_SPEC.loader.exec_module(qualification)


def _install_test_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the already-tested local NumPy without weakening production gates."""

    monkeypatch.setattr(sealer, "np", np)
    monkeypatch.setattr(sealer, "_require_pinned_runtime_and_load_numpy", lambda: None)


@pytest.mark.parametrize(
    ("values", "probability"),
    [
        (
            [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.8809702954610409,
                0.9457943091075629,
            ],
            0.50,
        ),
        (
            [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.8809702954610409,
                0.9457943091075629,
            ],
            0.95,
        ),
        ([index / 29.0 for index in range(25)], 0.50),
        ([index / 29.0 for index in range(25)], 0.95),
    ],
)
def test_linear_quantile_is_bit_exact_with_frozen_analyzer(
    values: list[float], probability: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_test_numpy(monkeypatch)
    expected = float(
        np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear")
    )
    assert sealer._linear_quantile(values, probability) == expected
    if len(values) == 10 and probability == 0.95:
        position = (len(values) - 1) * probability
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        old_result = values[lower] + (values[upper] - values[lower]) * (
            position - lower
        )
        assert old_result != expected


@pytest.mark.parametrize(
    "value",
    (
        "/absolute/input.png",
        "../escaped.png",
        "images/../escaped.png",
        "images//camera.png",
        "./images/camera.png",
        "images\\camera.png",
    ),
)
def test_dataset_relative_paths_must_be_canonical_posix(value: str) -> None:
    with pytest.raises(RuntimeError, match="canonical"):
        sealer._canonical_relative_posix(value, role="test dataset path")


def _write(path: Path, payload: bytes, mode: int = 0o444) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode_octal": f"{mode:04o}",
    }


def _write_signed(
    path: Path, value: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = sealer._signed(value)
    record = _write(path, sealer._canonical_json(artifact))
    return artifact, record


def _descriptor(
    record: dict[str, Any], artifact: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = {key: record[key] for key in ("path", "size_bytes", "sha256")}
    if artifact is not None:
        result["artifact_sha256"] = artifact["artifact_sha256"]
    return result


def _global_snapshot() -> dict[str, Any]:
    return {
        "event_writers_object_id": 1,
        "event_writer_ids": [2],
        "event_storage_object_id": 3,
        "event_storage_ids": [4],
        "global_buffer_object_id": 5,
        "global_buffer_items": [["fixed", 6]],
        "profiler_object_id": 7,
        "profiler_ids": [8],
        "pytorch_profiler_id": None,
    }


def _boundary(fd: int = 10, tasks: int = 2) -> dict[str, int]:
    return {
        "file_descriptor_count": fd,
        "task_count": tasks,
        "rss_kib": 1024,
        "rlimit_nofile_soft": 1024,
        "rlimit_nofile_hard": 1_048_576,
    }


def _smoke() -> dict[str, Any]:
    return sealer._signed(
        {
            "schema_version": 1,
            "artifact_kind": "Deform360HeldGsplatRuntimeSmokeV1",
            "contract_sha256": sealer.PINNED_GSPLAT_SMOKE_CONTRACT_SHA256,
            "physical_gpu_index": 1,
            "logical_device": "cuda:0",
            "gpu_name": sealer.PINNED_GPU_NAME,
            "compute_capability": "8.9",
            "python_version": "3.12",
            "torch_version": sealer.PINNED_TORCH_VERSION,
            "torch_cuda_version": sealer.PINNED_TORCH_CUDA_VERSION,
            "gsplat_version": sealer.PINNED_GSPLAT_VERSION,
            "extension_path": str(sealer.PINNED_GSPLAT_EXTENSION_PATH),
            "extension_sha256": sealer.PINNED_GSPLAT_EXTENSION_SHA256,
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


def _runtime() -> dict[str, Any]:
    return {
        "seed": 0,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": True,
        "torch_version": sealer.PINNED_TORCH_VERSION,
        "torch_cuda_version": sealer.PINNED_TORCH_CUDA_VERSION,
        "cuda_device_name": sealer.PINNED_GPU_NAME,
        "cuda_device_count": 1,
        "python_version": "3.12.9",
    }


def _source_dataset(source: Path) -> None:
    source.mkdir()
    (source / "images").mkdir()
    _write(source / "seed.ply", b"seed")
    frames = []
    for index in range(21):
        relative = f"images/camera-{index:02d}.png"
        _write(source / relative, f"image-{index}".encode())
        frames.append({"file_path": relative})
    transforms = {
        "ply_file_path": str((source / "seed.ply").resolve()),
        "frames": frames,
    }
    _write(
        source / "transforms.json",
        (json.dumps(transforms, sort_keys=True) + "\n").encode(),
    )


def _dataset_closure(
    dataset: Path, *, canonical_seed: Path | None = None
) -> dict[str, Any]:
    transforms_path = dataset / "transforms.json"
    transforms_payload = transforms_path.read_bytes()
    transforms = json.loads(transforms_payload)
    refs: list[tuple[str, str, Path]] = [("seed_ply", "seed.ply", dataset / "seed.ply")]
    refs.extend(
        ("frame_image", frame["file_path"], dataset / frame["file_path"])
        for frame in transforms["frames"]
    )
    bindings = []
    rows = []
    for role, relative, path in sorted(refs, key=lambda value: (value[1], value[0])):
        record = sealer._stable_file(path, role="test dataset")
        bindings.append({"role": role, "relative_path": relative, **record})
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    portable = copy.deepcopy(transforms)
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    normalized_payload = sealer._canonical_bytes(portable)
    content = {
        "normalized_transforms": {
            "size_bytes": len(normalized_payload),
            "sha256": hashlib.sha256(normalized_payload).hexdigest(),
        },
        "referenced_files": rows,
    }
    seed_reference: dict[str, Any] = {
        "declared_path": transforms["ply_file_path"],
        "canonical_absolute_alias_used": canonical_seed is not None,
    }
    if canonical_seed is not None:
        seed_reference.update(
            {
                "canonical_target": sealer._stable_file(
                    canonical_seed, role="canonical seed"
                ),
                "materialized_copy": sealer._stable_file(
                    dataset / "seed.ply", role="materialized seed"
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360DeclaredDatasetInputClosureV1",
        "root": str(dataset.resolve()),
        "raw_transforms": sealer._stable_file(transforms_path, role="raw transforms"),
        "transforms_relative_path": "transforms.json",
        "seed_relative_path": "seed.ply",
        "seed_reference": seed_reference,
        "frame_count": 21,
        "regular_input_file_count": 23,
        "referenced_input_bindings": bindings,
        "content_identity": content,
        "content_artifact_sha256": hashlib.sha256(
            sealer._canonical_bytes(content)
        ).hexdigest(),
        "generated_outputs_excluded": True,
        "symlinks_special_files_and_hardlink_aliases_accepted": False,
    }


def _materialize(source: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True)
    (destination / "images").mkdir()
    source_records: dict[str, Any] = {}
    materialized: dict[str, Any] = {}
    for path in [source / "seed.ply", *sorted((source / "images").iterdir())]:
        relative = path.relative_to(source).as_posix()
        source_records[relative] = sealer._stable_file(path, role="source")
        materialized[relative] = _write(destination / relative, path.read_bytes())
    transforms = json.loads((source / "transforms.json").read_text())
    transforms["ply_file_path"] = str((destination / "seed.ply").resolve())
    transforms_payload = (
        json.dumps(transforms, indent=2, sort_keys=True) + "\n"
    ).encode()
    materialized["transforms.json"] = _write(
        destination / "transforms.json", transforms_payload
    )
    portable = copy.deepcopy(transforms)
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    source_content = sealer._content_identity(source_records)
    materialized_content = sealer._content_identity(
        {
            name: record
            for name, record in materialized.items()
            if name != "transforms.json"
        }
    )
    return {
        "source_root": str(source.resolve()),
        "destination_root": str(destination.resolve()),
        "source_transforms": sealer._stable_file(
            source / "transforms.json", role="source transforms"
        ),
        "source_transforms_sha256": hashlib.sha256(
            (source / "transforms.json").read_bytes()
        ).hexdigest(),
        "materialized_transforms_sha256": hashlib.sha256(
            transforms_payload
        ).hexdigest(),
        "portable_transforms_sha256": hashlib.sha256(
            sealer._canonical_bytes(portable)
        ).hexdigest(),
        "rewritten_field": "ply_file_path",
        "source_seed_ply_path": str((source / "seed.ply").resolve()),
        "materialized_seed_ply_path": str((destination / "seed.ply").resolve()),
        "frame_count": 21,
        "copied_regular_file_count": 23,
        "source_records": source_records,
        "materialized_records": materialized,
        "referenced_source_content": source_content,
        "referenced_materialized_content": materialized_content,
        "referenced_source_materialized_content_equal": True,
        "unreferenced_outputs_copied": False,
    }


def _source_record(path: Path, *, git_blob: bool = False) -> dict[str, Any]:
    record = sealer._stable_file(path.resolve(), role="test source")
    if git_blob:
        record["git_blob_oid"] = "f" * 40
    return record


def _code_binding(repo: Path, head: str) -> dict[str, Any]:
    return {
        "path": str(repo.resolve()),
        "head": head,
        "tree": "b" * 40,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }


def _python_binding() -> dict[str, Any]:
    return {
        "lexical_path": str(sealer.PINNED_PYTHON),
        "lexical_mode_octal": "0777",
        "lexical_symlink_target": "/usr/bin/python3",
        "resolved_executable": sealer._stable_file(
            Path(sealer.PINNED_PYTHON_TARGET), role="test Python"
        ),
        "pyvenv_config": sealer._stable_file(
            sealer.PINNED_PYTHON_RUNTIME / "pyvenv.cfg", role="test pyvenv"
        ),
        "frozen_package_inventory": sealer._stable_file(
            sealer.PINNED_PYTHON_FREEZE, role="test freeze"
        ),
        "frozen_runtime_tree_manifest": sealer._stable_file(
            sealer.PINNED_PYTHON_TREE_MANIFEST, role="test runtime tree"
        ),
        "pip_freeze_all": {
            "normalized_sha256": sealer.PINNED_PYTHON_FREEZE_SHA256,
            "normalized_line_count": 1,
            "normalized_size_bytes": 7,
            "equals_frozen_package_inventory": True,
        },
    }


def _cleanup() -> dict[str, Any]:
    raise AssertionError("use _tree_cleanup")


def _tree_cleanup(
    path: Path, parent: Path, *, recreated: bool = False
) -> dict[str, Any]:
    result = {
        "bounded_parent": str(parent.resolve()),
        "pre_cleanup_inventory": {
            "root": str(path.resolve()),
            "entry_count": 1,
            "regular_file_bytes": 1,
            "inventory_sha256": "c" * 64,
        },
        "removed": True,
        "post_cleanup_absent": True,
    }
    if recreated:
        result["recreated_empty"] = True
    return result


def _invocation(
    command: list[str], log: Path, root: Path, return_code: int = 0
) -> dict[str, Any]:
    log_record = _write(log, b"completed\n")
    return {
        "command": command,
        "environment": sealer._expected_environment(root),
        "return_code": return_code,
        "timed_out": False,
        "timeout_error": None,
        "timeout_seconds": (
            sealer.FIT_TIMEOUT_SECONDS
            if "_fit-child" in command
            else sealer.SOAK_TIMEOUT_SECONDS
            if "_soak-child" in command
            else sealer.ANALYZER_TIMEOUT_SECONDS
        ),
        "log": log_record,
    }


def _fit_child(
    *,
    root: Path,
    source: Path,
    repo: Path,
    mode: str,
    pairing_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fit_root = root / "ab" / mode / pairing_id
    audit = _materialize(source, fit_root / "dataset")
    ply = _write(fit_root / "export/splat.ply", b"synthetic retained gaussian ply\n")
    smoke = _smoke()
    adapter = repo / sealer.RELATIVE_GSPLAT_ADAPTER_SOURCE
    child = sealer._signed(
        {
            "schema_version": 1,
            "artifact_kind": sealer.FIT_KIND,
            "qualification_id": sealer.QUALIFICATION_ID,
            "variant": mode,
            "passed": True,
            "parameters": {"iterations": 250, "seed": 0},
            "runtime": _runtime(),
            "gsplat_runtime_smoke": {
                "adapter_source": _source_record(adapter),
                "evidence": smoke,
                "evidence_artifact_sha256": smoke["artifact_sha256"],
            },
            "dataset": str((fit_root / "dataset").resolve()),
            "output": ply,
            "resource_boundary": {"before": _boundary(), "after": _boundary()},
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
    child_record = _write(fit_root / "fit-evidence.json", sealer._canonical_json(child))
    invocation = _invocation(
        sealer._fit_command(
            root=root,
            code=repo.resolve(),
            qualifier=(repo / sealer.RELATIVE_QUALIFIER_SOURCE).resolve(),
            mode=mode,
            pairing_id=pairing_id,
        ),
        fit_root / "fit.log",
        root,
    )
    aggregate = {
        "pairing_id": pairing_id,
        "dataset_key": f"ab_{mode}_{pairing_id.replace('-', '_')}",
        "invocation_key": f"ab_{mode}_{pairing_id.replace('-', '_')}",
        "invocation": invocation,
        "child_evidence": child,
        "child_evidence_record": _descriptor(child_record, child),
        "child_evidence_validation": {
            "loaded_and_signature_valid": True,
            "identity_and_output_binding_valid": True,
            "artifact_sha256": child["artifact_sha256"],
        },
        "retained_output": ply,
        "cleanup": {
            "generated_dataset_outputs": _tree_cleanup(
                fit_root / "dataset/outputs", fit_root / "dataset"
            ),
            "qualification_temporary_cache": _tree_cleanup(
                root / "tmp", root, recreated=True
            ),
        },
    }
    manifest = {
        "pairing_id": pairing_id,
        "ply": _descriptor(ply),
        "fit_evidence": _descriptor(child_record),
        "dataset_input_inventory": _dataset_closure(fit_root / "dataset"),
    }
    return aggregate, manifest, audit


def _pair_record(
    left_mode: str,
    left_id: str,
    right_mode: str,
    right_id: str,
    value: float,
    *,
    exact: bool,
) -> dict[str, Any]:
    return {
        "left": {"mode": left_mode, "pairing_id": left_id},
        "right": {"mode": right_mode, "pairing_id": right_id},
        "matched_pairing_id": left_id == right_id,
        "structured_array_exact": exact,
        "file_sha256_exact": exact,
        "metrics": {name: value for name in sealer.PAIR_METRIC_NAMES},
    }


def _analysis_groups(*, accepted: bool) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for group in sealer.GROUP_COUNTS:
        records = []
        for left_mode, left_id, right_mode, right_id in sealer._expected_pairs(group):
            cross = group == "cross_mode"
            records.append(
                _pair_record(
                    left_mode,
                    left_id,
                    right_mode,
                    right_id,
                    0.0 if accepted or not cross else 1.0,
                    exact=accepted,
                )
            )
        groups[group] = records
    return groups


def _analysis_payload(
    *,
    accepted: bool,
    root: Path,
    repo: Path,
    code: dict[str, Any],
    manifest: dict[str, Any],
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    groups = _analysis_groups(accepted=accepted)
    distributions = {
        metric: {
            group: sealer._distribution(records, metric)
            for group, records in groups.items()
        }
        for metric in sealer.PAIR_METRIC_NAMES
    }
    per_metric = {}
    for metric, item in distributions.items():
        within_p95 = max(item["within_original"]["p95"], item["within_wrapped"]["p95"])
        within_max = max(
            item["within_original"]["maximum"], item["within_wrapped"]["maximum"]
        )
        median = item["cross_mode"]["median"]
        p95 = item["cross_mode"]["p95"]
        per_metric[metric] = {
            "cross_median": median,
            "within_p95_limit": within_p95,
            "cross_median_condition_passed": median <= within_p95,
            "cross_p95": p95,
            "within_max_limit": within_max,
            "cross_p95_condition_passed": p95 <= within_max,
            "passed": median <= within_p95 and p95 <= within_max,
        }
    gate_passed = all(item["passed"] for item in per_metric.values())
    gate = {
        "contract": dict(sealer.GATE_CONTRACT),
        "pair_counts": dict(sealer.GROUP_COUNTS),
        "per_metric": per_metric,
        "all_metrics_finite_and_nonnegative": True,
        "passed": gate_passed,
    }
    decision = {
        "exact_matched_structured_array_equality_primary_passed": accepted,
        "exact_matched_file_bytes_equal": accepted,
        "secondary_distributional_equivalence_passed": gate_passed,
        "accepted": accepted,
        "acceptance_basis": "exact-structured-array-equality"
        if accepted
        else "rejected",
    }
    required_environment = dict(sealer.REQUIRED_EXECUTION_ENVIRONMENT)
    required_environment.pop("CUDA_VISIBLE_DEVICES")
    smoke = _smoke()
    analyzer_source = repo / sealer.RELATIVE_ANALYZER_SOURCE
    source_bindings = {
        "analyzer_git": code,
        "analyzer_source": _source_record(analyzer_source),
    }
    runtime_binding = {
        "frozen_package_inventory": {"sha256": sealer.PINNED_PYTHON_FREEZE_SHA256},
        "frozen_runtime_tree_manifest": {
            "sha256": sealer.PINNED_PYTHON_TREE_MANIFEST_SHA256
        },
        "live_pip_freeze_all": {
            "normalized_sha256": sealer.PINNED_PYTHON_FREEZE_SHA256,
            "equals_frozen_package_inventory": True,
        },
    }
    execution_binding = {
        "host": sealer.EXPECTED_HOST,
        "process_flags": {
            "isolated": 1,
            "no_user_site": 1,
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "safe_path": True,
        },
        "environment": {
            "required": required_environment,
            "cuda_visible_devices": "1",
            "forbidden_absent": list(sealer.FORBIDDEN_EXECUTION_ENVIRONMENT),
            "tmpdir": str(root / "tmp"),
        },
        "live_device": {
            "visible_cuda_device_count": 1,
            "logical_device": "cuda:0",
            "gpu_name": sealer.PINNED_GPU_NAME,
            "compute_capability": "8.9",
            "torch_version": sealer.PINNED_TORCH_VERSION,
            "torch_cuda_version": sealer.PINNED_TORCH_CUDA_VERSION,
        },
        "imports": {},
    }
    result_inputs = copy.deepcopy(manifest["modes"])
    for mode in ("original", "wrapped"):
        for item in result_inputs[mode]:
            child = json.loads(Path(item["fit_evidence"]["path"]).read_text())
            item["fit_evidence"]["artifact_sha256"] = child["artifact_sha256"]
    canonical_transform = {
        **manifest["canonical_transforms"]["raw_representative"],
        "mode_octal": manifest["canonical_source_dataset"]["raw_transforms"][
            "mode_octal"
        ],
    }
    input_manifest = {
        **_descriptor(manifest_record),
        "mode_octal": manifest_record["mode_octal"],
        "artifact_sha256": manifest["artifact_sha256"],
    }
    before_state: dict[str, Any] = {
        "manifest": input_manifest,
        "canonical_source_dataset": manifest["canonical_source_dataset"],
        "canonical_transforms": canonical_transform,
        "transitive_inputs": result_inputs,
        "source": source_bindings,
        "runtime": runtime_binding,
        "execution": execution_binding,
    }
    return sealer._signed(
        {
            "schema_version": 1,
            "artifact_kind": sealer.RESULT_KIND,
            "analysis_id": sealer.ANALYSIS_ID,
            "development_only": True,
            "formal_path_accessed": False,
            "host": sealer.EXPECTED_HOST,
            "generator_profile": "same-as-analyzer",
            "physical_gpu_index": 1,
            "input_manifest": input_manifest,
            "source_bindings": source_bindings,
            "runtime_binding": runtime_binding,
            "execution_binding": execution_binding,
            "gsplat_runtime": {
                "host": sealer.EXPECTED_HOST,
                "physical_gpu_index": 1,
                "smoke_evidence": smoke,
            },
            "renderer_sys_path_restoration": {"restored_exactly": True},
            "canonical_source_dataset": manifest["canonical_source_dataset"],
            "canonical_transforms": canonical_transform,
            "inputs": result_inputs,
            "pre_post_render_stability": {
                "before": before_state,
                "after": before_state,
                "exact_equal": True,
                "analyzer_gsplat_smoke_executed_once_before_render": True,
                "adapter_and_aot_bytes_revalidated_after_render": True,
            },
            "statistical_limitations": ["synthetic test fixture"],
            "schema_validation": {
                "expected_field_count": 62,
                "expected_field_names": [f"field-{index}" for index in range(62)],
                "all_property_declarations_literal_float_f4": True,
                "inert_normal_fields": ["nx", "ny", "nz"],
                "all_inert_normal_values_exactly_zero": True,
                "inert_normal_fields_excluded_from_distribution_metrics": True,
                "identical_schema_across_all_plys": True,
                "all_source_and_derived_values_finite": True,
            },
            "render_execution": {
                "contract": {
                    "camera_count": 21,
                    "integer_downscale": 4,
                    "one_batched_rasterization_call_per_ply": True,
                },
                "calls": [
                    {"mode": mode, "pairing_id": pairing_id}
                    for mode in ("original", "wrapped")
                    for pairing_id in sealer.PAIRING_IDS
                ],
                "render_call_count": 10,
                "each_ply_rendered_exactly_once": True,
            },
            "pair_groups": groups,
            "metric_distributions": distributions,
            "secondary_distributional_gate": gate,
            "decision": decision,
        }
    )


def _manifest_payload(
    *,
    source: Path,
    repo: Path,
    code: dict[str, Any],
    modes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_names = {
        str(sealer.RELATIVE_QUALIFIER_SOURCE),
        str(sealer.RELATIVE_WRAPPER_SOURCE.parent / "deform360_held_gsplat_runtime.py"),
        str(sealer.RELATIVE_WRAPPER_SOURCE),
        str(sealer.RELATIVE_GSPLAT_ADAPTER_SOURCE),
    }
    sources = {
        name: _source_record((repo / name).resolve(), git_blob=True)
        for name in source_names
    }
    canonical = _dataset_closure(source)
    return sealer._signed(
        {
            "schema_version": 1,
            "artifact_kind": sealer.MANIFEST_KIND,
            "analysis_id": sealer.ANALYSIS_ID,
            "expected_environment": {
                "generator_profile": "same-as-analyzer",
                "physical_gpu_index": 1,
                "generator_code": {
                    "profile": "same-as-analyzer",
                    "qualification_id": sealer.QUALIFICATION_ID,
                    "physical_gpu_index": 1,
                    "git": code,
                    "sources": sources,
                },
                "analyzer_code": code,
                "deform360_git_head": sealer.PINNED_DEFORM360_HEAD,
                "deform360_git_tree": sealer.PINNED_DEFORM360_TREE,
                "python_freeze_sha256": sealer.PINNED_PYTHON_FREEZE_SHA256,
                "python_tree_manifest_sha256": sealer.PINNED_PYTHON_TREE_MANIFEST_SHA256,
            },
            "canonical_source_dataset": canonical,
            "canonical_transforms": {
                "raw_representative": _descriptor(canonical["raw_transforms"]),
                "normalized": canonical["content_identity"]["normalized_transforms"],
            },
            "modes": modes,
        }
    )


def _soak_child(dataset: Path) -> dict[str, Any]:
    initial = _global_snapshot()
    fits = []
    for index in range(243):
        output = dataset.parent / "export" / f"splat-{index:04d}.ply"
        output_size = 100 + index
        fits.append(
            {
                "fit_index": index,
                "trainer_reinitialized": index % 81 == 0,
                "output_created": True,
                "dataset_outputs_created": True,
                "output_size_bytes": output_size,
                "cleanup_completed": True,
                "cleanup": {
                    "output_ply": {
                        "bounded_parent": str(output.parent.resolve()),
                        "pre_cleanup_binding": {
                            "path": str(output.resolve()),
                            "size_bytes": output_size,
                            "sha256": "d" * 64,
                            "mode_octal": "0444",
                        },
                        "pre_cleanup_link_count": 1,
                        "removed": True,
                        "post_cleanup_absent": True,
                    },
                    "dataset_outputs": _tree_cleanup(dataset / "outputs", dataset),
                },
                "output_ply_absent_after_cleanup": True,
                "dataset_outputs_absent_after_cleanup": True,
                "resource_boundary_stage": "after_cleanup",
                "resource_boundary": _boundary(12 + index % 2, 3),
                "globals_restored": True,
                "global_state": initial,
            }
        )
    evaluation = sealer._recompute_soak_evaluation(_boundary(), fits)
    smoke = _smoke()
    adapter = Path(__file__).parents[1] / sealer.RELATIVE_GSPLAT_ADAPTER_SOURCE
    return sealer._signed(
        {
            "schema_version": 1,
            "artifact_kind": sealer.SOAK_KIND,
            "qualification_id": sealer.QUALIFICATION_ID,
            "passed": True,
            "parameters": {
                "fit_count": 243,
                "iterations_per_fit": 1,
                "seed": 0,
                "trainer_reinitialization_interval": 81,
            },
            "runtime": _runtime(),
            "gsplat_runtime_smoke": {
                "adapter_source": _source_record(adapter.resolve()),
                "evidence": smoke,
                "evidence_artifact_sha256": smoke["artifact_sha256"],
            },
            "dataset": str(dataset.resolve()),
            "initial_global_state": initial,
            "fits": fits,
            "evaluation": evaluation,
            "formal_held_path_supplied": False,
        }
    )


def _artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    _install_test_numpy(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    head = "a" * 40
    root = base / f"{sealer.ROOT_PREFIX}{head}"
    root.mkdir()
    source = tmp_path / "public-dataset"
    _source_dataset(source)
    repo = Path(__file__).parents[1].resolve()

    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    resolved_python = tmp_path / "usr/bin/python3.12"
    resolved_record = _write(resolved_python, b"python-runtime", 0o555)
    python = runtime / "bin/python"
    python.symlink_to("/usr/bin/python3")
    _write(runtime / "pyvenv.cfg", b"home = /usr\n", 0o444)
    freeze = tmp_path / "runtime.freeze.sorted.txt"
    freeze_record = _write(freeze, b"x==1.0\n", 0o400)
    tree = tmp_path / "runtime.tree-manifest.json"
    tree_record = _write(tree, b"{}\n", 0o400)
    deform360 = tmp_path / "Deform360"
    deform360.mkdir()

    monkeypatch.setattr(sealer, "BASE", base.resolve())
    monkeypatch.setattr(sealer, "FORMAL_HELD_PARENT", (tmp_path / "formal").resolve())
    monkeypatch.setattr(sealer, "PUBLIC_DATASET", source.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON", python.absolute())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_RUNTIME", runtime.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_FREEZE", freeze.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_TREE_MANIFEST", tree.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_TARGET", str(resolved_python.resolve()))
    monkeypatch.setattr(
        sealer, "PINNED_PYTHON_TARGET_SHA256", resolved_record["sha256"]
    )
    monkeypatch.setattr(sealer, "PINNED_PYTHON_FREEZE_SHA256", freeze_record["sha256"])
    monkeypatch.setattr(
        sealer, "PINNED_PYTHON_TREE_MANIFEST_SHA256", tree_record["sha256"]
    )
    monkeypatch.setattr(
        sealer, "PINNED_PYTHON_BASE_PREFIX", str((tmp_path / "usr").resolve())
    )
    monkeypatch.setattr(sealer, "PINNED_DEFORM360", deform360.resolve())
    monkeypatch.setattr(sealer.socket, "gethostname", lambda: sealer.EXPECTED_HOST)

    code = _code_binding(repo, head)
    analyzer = (repo / sealer.RELATIVE_ANALYZER_SOURCE).resolve()
    attempt, attempt_record = _write_signed(
        root / sealer.ATTEMPT_NAME,
        {
            "schema_version": 2,
            "artifact_kind": sealer.ATTEMPT_KIND,
            "qualification_id": sealer.QUALIFICATION_ID,
            "state": "canonical-root-consumed-at-creation",
            "output_root": str(root.resolve()),
            "code_revision": head,
            "generator_profile": "same-as-analyzer",
            "physical_gpu_index": 1,
            "frozen_analyzer_source": _source_record(analyzer),
            "root_consumption_policy": dict(sealer.ROOT_CONSUMPTION_POLICY),
            "formal_held_path_supplied": False,
        },
    )

    repeats: dict[str, list[dict[str, Any]]] = {"original": [], "wrapped": []}
    modes: dict[str, list[dict[str, Any]]] = {"original": [], "wrapped": []}
    audits: dict[str, Any] = {}
    for mode in ("original", "wrapped"):
        for pairing_id in sealer.PAIRING_IDS:
            aggregate, manifest_record, audit = _fit_child(
                root=root,
                source=source,
                repo=repo,
                mode=mode,
                pairing_id=pairing_id,
            )
            repeats[mode].append(aggregate)
            modes[mode].append(manifest_record)
            audits[f"ab_{mode}_{pairing_id.replace('-', '_')}"] = audit

    equivalence_root = root / "equivalence"
    manifest = _manifest_payload(source=source, repo=repo, code=code, modes=modes)
    manifest_record = _write(
        equivalence_root / "repeat-manifest.json", sealer._canonical_json(manifest)
    )
    prepare_invocation = _invocation(
        sealer._prepare_manifest_command(root=root, code=repo, analyzer=analyzer),
        equivalence_root / "prepare-manifest.log",
        root,
    )
    result = _analysis_payload(
        accepted=accepted,
        root=root,
        repo=repo,
        code=code,
        manifest=manifest,
        manifest_record=manifest_record,
    )
    result_record = _write(
        equivalence_root / "analysis-result.json", sealer._canonical_json(result)
    )
    analyze_invocation = _invocation(
        sealer._analyze_command(root=root, code=repo, analyzer=analyzer),
        equivalence_root / "analyze.log",
        root,
        0 if accepted else 3,
    )
    equivalence_cleanup = {
        "after_manifest": _tree_cleanup(root / "tmp", root, recreated=True),
        "after_analysis": _tree_cleanup(root / "tmp", root, recreated=True),
    }
    equivalence = {
        "passed": accepted,
        "manifest": {**manifest_record, "artifact_sha256": manifest["artifact_sha256"]},
        "prepare_manifest_invocation": prepare_invocation,
        "result": {**result_record, "artifact_sha256": result["artifact_sha256"]},
        "analysis_invocation": analyze_invocation,
        "decision": result["decision"],
        "cleanup": equivalence_cleanup,
    }

    soak: dict[str, Any] | None = None
    if accepted:
        soak_root = root / "soak"
        audits["soak"] = _materialize(source, soak_root / "dataset")
        soak_child = _soak_child(soak_root / "dataset")
        soak_child_record = _write(
            soak_root / "soak-evidence.json", sealer._canonical_json(soak_child)
        )
        soak = {
            "passed": True,
            "invocation": _invocation(
                sealer._soak_command(
                    root=root,
                    code=repo,
                    qualifier=(repo / sealer.RELATIVE_QUALIFIER_SOURCE).resolve(),
                ),
                soak_root / "soak.log",
                root,
            ),
            "child_evidence": soak_child,
            "child_evidence_record": _descriptor(soak_child_record, soak_child),
            "child_evidence_validation": {
                "loaded_and_signature_valid": True,
                "identity_sequence_resource_and_cleanup_valid": True,
                "artifact_sha256": soak_child["artifact_sha256"],
            },
            "cleanup": {
                "generated_outputs_absent_after_every_fit": True,
                "empty_export_removed": _tree_cleanup(soak_root / "export", soak_root),
                "final_temporary_cache_removed": _tree_cleanup(root / "tmp", root),
            },
        }

    python = _python_binding()
    deform = {
        "path": str(sealer.PINNED_DEFORM360.resolve()),
        "head": sealer.PINNED_DEFORM360_HEAD,
        "tree": sealer.PINNED_DEFORM360_TREE,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }
    top_invocations = {
        record["invocation_key"]: record["invocation"]
        for mode in ("original", "wrapped")
        for record in repeats[mode]
    }
    top_invocations.update(
        {
            "equivalence_prepare_manifest": prepare_invocation,
            "equivalence_analyze": analyze_invocation,
        }
    )
    if soak is not None:
        top_invocations["soak"] = soak["invocation"]
    cleanup_events = [
        event
        for mode in ("original", "wrapped")
        for record in repeats[mode]
        for event in (
            record["cleanup"]["generated_dataset_outputs"],
            record["cleanup"]["qualification_temporary_cache"],
        )
    ]
    cleanup_events.extend(
        [
            equivalence_cleanup["after_manifest"],
            equivalence_cleanup["after_analysis"],
        ]
    )
    if soak is not None:
        cleanup_events.extend(
            [
                soak["cleanup"]["empty_export_removed"],
                soak["cleanup"]["final_temporary_cache_removed"],
            ]
        )
    else:
        cleanup_events.append(_tree_cleanup(root / "tmp", root))
    evidence = sealer._signed(
        {
            "schema_version": 2,
            "artifact_kind": sealer.QUALIFICATION_KIND,
            "qualification_id": sealer.QUALIFICATION_ID,
            "status": "qualified" if accepted else "admission-inconclusive",
            "passed": accepted,
            "host": sealer.EXPECTED_HOST,
            "phase": "all",
            "generator_profile": "same-as-analyzer",
            "physical_gpu_index": 1,
            "canonical_run_parameters": sealer._canonical_parameters(),
            "parameters": sealer._parameters(),
            "execution_order": [
                "fresh-five-original-and-five-wrapped-fits",
                "equivalence-analyzer",
                "243-fit-soak-only-after-analyzer-acceptance",
            ],
            "runtime_bindings": {
                "python_path": str(sealer.PINNED_PYTHON),
                "python": python,
                "python_after": python,
                "parent_python_process": {
                    "sys_executable": str(sealer.PINNED_PYTHON),
                    "sys_base_executable": sealer.PINNED_PYTHON_TARGET,
                    "sys_prefix": str(sealer.PINNED_PYTHON_RUNTIME),
                    "sys_base_prefix": sealer.PINNED_PYTHON_BASE_PREFIX,
                },
                "code": code,
                "code_after": code,
                "deform360": deform,
                "deform360_after": deform,
                "qualification_source": _source_record(
                    (repo / sealer.RELATIVE_QUALIFIER_SOURCE).resolve()
                ),
                "wrapper_source": _source_record(
                    (repo / sealer.RELATIVE_WRAPPER_SOURCE).resolve()
                ),
                "analyzer_source": _source_record(analyzer),
            },
            "source_dataset": str(source.resolve()),
            "attempt": _descriptor(attempt_record, attempt),
            "root_consumption_policy": dict(sealer.ROOT_CONSUMPTION_POLICY),
            "materialized_datasets": audits,
            "invocations": top_invocations,
            "ab": {
                "passed": accepted,
                "repeat_count_per_mode": 5,
                "pairing_ids": list(sealer.PAIRING_IDS),
                "repeats": repeats,
                "equivalence": equivalence,
                "predicates": sealer._ab_predicates(accepted),
            },
            "soak": soak,
            "cleanup_events": cleanup_events,
            "admission": sealer._expected_admission(accepted),
            "predicates": sealer._top_predicates(accepted),
            "information_boundary": {
                "formal_held_path_accepted": False,
                "formal_target_or_outcome_array_read": False,
                "development_dataset_only": True,
                "unreferenced_source_outputs_copied": False,
                "rlimit_nofile_changed": False,
            },
        }
    )
    _write(root / sealer.MAIN_NAME, sealer._canonical_json(evidence))
    return root, Path(f"{root}-integrity-completion.json"), evidence


@pytest.mark.parametrize(("accepted", "eligible"), [(True, True), (False, False)])
def test_sealer_seals_both_complete_terminal_outcomes_but_only_admits_qualified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
    eligible: bool,
) -> None:
    root, completion, _ = _artifact(tmp_path, monkeypatch, accepted=accepted)

    result = sealer.seal(root, completion)

    assert result["passed"] is True
    assert result["admission_eligible"] is eligible
    assert result["terminal_outcome"] == (
        "qualified" if accepted else "admission-inconclusive"
    )
    assert result["qualification_attempt"]["artifact_sha256"]
    assert result["repeat_manifest"]["artifact_sha256"]
    assert result["equivalence_result"]["artifact_sha256"]
    assert result["analyzer_source"]["sha256"] == sealer.ANALYZER_SOURCE_SHA256
    assert stat.S_IMODE(root.stat().st_mode) == 0o500
    assert stat.S_IMODE(completion.stat().st_mode) == 0o400
    for current, _directories, files in os.walk(root):
        assert stat.S_IMODE(Path(current).stat().st_mode) == 0o500
        for name in files:
            assert stat.S_IMODE((Path(current) / name).stat().st_mode) == 0o400


def _protocol_lineage(
    root: Path,
    completion: Path,
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setattr(protocol, "RESOURCE_LIFECYCLE_QUALIFICATION_BASE", root.parent)
    monkeypatch.setattr(
        protocol,
        "RESOURCE_LIFECYCLE_PUBLIC_DATASET",
        Path(evidence["source_dataset"]),
    )
    return protocol.validate_resource_lifecycle_qualification_lineage(
        evidence_path=root / sealer.MAIN_NAME,
        completion_path=completion,
        verify_content_inventory=True,
    )


def test_protocol_admits_exact_sealed_h1_sources_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, evidence = _artifact(tmp_path, monkeypatch)
    sealer.seal(root, completion)

    lineage = _protocol_lineage(root, completion, evidence, monkeypatch)

    assert (
        lineage["resource_lifecycle_qualification_integrity"]["admission_eligible"]
        is True
    )


def test_protocol_rejects_wrong_canonical_qualification_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, evidence = _artifact(tmp_path, monkeypatch)
    sealer.seal(root, completion)
    monkeypatch.setattr(
        protocol,
        "RESOURCE_LIFECYCLE_QUALIFICATION_BASE",
        root.parent,
    )
    monkeypatch.setattr(
        protocol,
        "RESOURCE_LIFECYCLE_PUBLIC_DATASET",
        tmp_path / "different-public-dataset",
    )

    with pytest.raises(ValueError, match="parameters changed"):
        protocol.validate_resource_lifecycle_qualification_lineage(
            evidence_path=root / sealer.MAIN_NAME,
            completion_path=completion,
            verify_content_inventory=True,
        )


def test_protocol_rejects_resigned_completion_with_wrong_h1_sealer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, evidence = _artifact(tmp_path, monkeypatch)
    sealed = sealer.seal(root, completion)
    changed = copy.deepcopy(sealed)
    changed["executed_integrity_sealer_source"]["sha256"] = "0" * 64
    changed["artifact_sha256"] = protocol.held_artifact_sha256(changed)
    completion.chmod(0o600)
    completion.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    completion.chmod(0o400)

    with pytest.raises(ValueError, match="source cross-link changed"):
        _protocol_lineage(root, completion, evidence, monkeypatch)


def test_sealer_rejects_undeclared_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, _ = _artifact(tmp_path, monkeypatch)
    _write(root / "ab/original/repeat-000/dataset/outputs/cache.bin", b"cache")

    with pytest.raises(
        RuntimeError, match="generated dataset outputs remain|undeclared"
    ):
        sealer.seal(root, completion)

    assert not completion.exists()


def test_sealer_rejects_resigned_invocation_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, evidence = _artifact(tmp_path, monkeypatch)
    changed = copy.deepcopy(evidence)
    record = changed["ab"]["repeats"]["original"][0]
    record["invocation"]["environment"]["CUDA_MODULE_LOADING"] = "EAGER"
    changed["invocations"][record["invocation_key"]] = record["invocation"]
    (root / sealer.MAIN_NAME).chmod(0o600)
    _write(
        root / sealer.MAIN_NAME,
        sealer._canonical_json(sealer._signed(changed)),
    )

    with pytest.raises(RuntimeError, match="invocation changed"):
        sealer.seal(root, completion)


def test_dataset_audit_rejects_relabelled_source_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _completion, evidence = _artifact(tmp_path, monkeypatch)
    audit = copy.deepcopy(evidence["materialized_datasets"]["ab_original_repeat_000"])
    seed_record = audit["source_records"].pop("seed.ply")
    audit["source_records"]["images/relabelled-seed.ply"] = seed_record

    with pytest.raises(RuntimeError, match="source path changed"):
        sealer._validate_dataset_audit(
            audit,
            root=root,
            expected_root=Path(audit["destination_root"]),
            role="mutated materialization",
        )


def test_dataset_audit_rejects_resigned_noncanonical_transform_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "public-dataset"
    _source_dataset(source)
    materialized = tmp_path / "qualification/repeat/dataset"
    audit = _materialize(source, materialized)
    monkeypatch.setattr(sealer, "PUBLIC_DATASET", source.resolve())

    transforms_path = materialized / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    transforms["frames"][0]["transform_matrix"] = [
        [1.0, 0.0, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    payload = (json.dumps(transforms, indent=2, sort_keys=True) + "\n").encode()
    transforms_path.chmod(0o600)
    audit["materialized_records"]["transforms.json"] = _write(transforms_path, payload)
    audit["materialized_transforms_sha256"] = hashlib.sha256(payload).hexdigest()
    portable = copy.deepcopy(transforms)
    portable["ply_file_path"] = "<MATERIALIZED-SEED-PLY>"
    audit["portable_transforms_sha256"] = hashlib.sha256(
        sealer._canonical_bytes(portable)
    ).hexdigest()

    with pytest.raises(RuntimeError, match="portable transforms differ"):
        sealer._validate_dataset_audit(
            audit,
            root=tmp_path.resolve(),
            expected_root=materialized.resolve(),
            role="altered transform matrix",
        )


def test_dataset_closure_rejects_resigned_altered_camera_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "public-dataset"
    _source_dataset(source)
    materialized = tmp_path / "qualification/repeat/dataset"
    _materialize(source, materialized)
    monkeypatch.setattr(sealer, "PUBLIC_DATASET", source.resolve())

    original = materialized / "images/camera-00.png"
    replacement = materialized / "images/camera-alternate.png"
    original.rename(replacement)
    transforms_path = materialized / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    transforms["frames"][0]["file_path"] = "images/camera-alternate.png"
    transforms_path.chmod(0o600)
    _write(
        transforms_path,
        (json.dumps(transforms, indent=2, sort_keys=True) + "\n").encode(),
    )
    resigned_closure = _dataset_closure(materialized)

    with pytest.raises(RuntimeError, match="content differs from the canonical"):
        sealer._validate_dataset_closure(
            resigned_closure,
            root=tmp_path.resolve(),
            expected_root=materialized.resolve(),
            role="altered camera reference",
            require_local=True,
        )


def test_dataset_audit_binds_declared_source_seed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "public-dataset"
    _source_dataset(source)
    materialized = tmp_path / "qualification/repeat/dataset"
    audit = _materialize(source, materialized)
    monkeypatch.setattr(sealer, "PUBLIC_DATASET", source.resolve())
    audit["source_seed_ply_path"] = str((source / "images/camera-00.png").resolve())

    with pytest.raises(RuntimeError, match="seed declaration differ"):
        sealer._validate_dataset_audit(
            audit,
            root=tmp_path.resolve(),
            expected_root=materialized.resolve(),
            role="wrong source seed",
        )


@pytest.mark.parametrize(
    ("missing_name", "message"),
    [
        (sealer.ATTEMPT_NAME, "qualification attempt marker is missing"),
        (sealer.MAIN_NAME, "complete qualification aggregate is missing"),
    ],
)
def test_sealer_rejects_missing_attempt_or_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
    message: str,
) -> None:
    root, completion, _ = _artifact(tmp_path, monkeypatch)
    (root / missing_name).unlink()

    with pytest.raises(RuntimeError, match=message):
        sealer.seal(root, completion)


def test_sealer_rejects_symlink_or_hardlink_tree_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, completion, _ = _artifact(tmp_path, monkeypatch)
    target = root / "qualification-attempt.json"
    alias = root / "alias.json"
    os.link(target, alias)

    with pytest.raises(RuntimeError, match="hard-linked|linked"):
        sealer.seal(root, completion)


def test_tree_sealing_detects_root_substitution_without_touching_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "qualification"
    root.mkdir()
    owned = root / "owned.bin"
    owned.write_bytes(b"owned")
    identity = sealer._inode_identity(os.lstat(root))
    moved = tmp_path / "moved-qualification"
    replacement = root / "replacement.bin"
    real_fchmod = sealer.os.fchmod
    swapped = False

    def substitute_after_first_chmod(descriptor: int, mode: int) -> None:
        nonlocal swapped
        real_fchmod(descriptor, mode)
        if not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
            replacement.write_bytes(b"replacement")

    monkeypatch.setattr(sealer.os, "fchmod", substitute_after_first_chmod)
    with pytest.raises(RuntimeError, match="root changed"):
        sealer._seal_tree(root, expected_identity=identity)

    assert replacement.read_bytes() == b"replacement"
    assert stat.S_IMODE(replacement.stat().st_mode) != 0o400
    moved.chmod(0o700)
    (moved / owned.name).chmod(0o600)


def test_completion_publication_detects_parent_substitution_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "base"
    parent.mkdir()
    completion = parent / "completion.json"
    moved = tmp_path / "moved-base"
    replacement = parent / "replacement.bin"
    real_link = sealer.os.link
    swapped = False

    def substitute_after_link(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        real_link(*args, **kwargs)
        if not swapped:
            swapped = True
            parent.rename(moved)
            parent.mkdir()
            replacement.write_bytes(b"replacement")

    monkeypatch.setattr(sealer.os, "link", substitute_after_link)
    with pytest.raises(RuntimeError, match="parent.*identity changed"):
        sealer._write_completion(completion, {"complete": True})

    assert replacement.read_bytes() == b"replacement"
    assert not completion.exists()
    assert not (moved / completion.name).exists()


def test_completion_publication_rollback_does_not_unlink_filename_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "base"
    parent.mkdir()
    completion = parent / "completion.json"
    replacement_payload = b"replacement must survive"
    real_link = sealer.os.link

    def substitute_completion_after_link(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        completion.unlink()
        completion.write_bytes(replacement_payload)

    monkeypatch.setattr(sealer.os, "link", substitute_completion_after_link)
    with pytest.raises(RuntimeError, match="changed while publishing"):
        sealer._write_completion(completion, {"complete": True})

    assert completion.read_bytes() == replacement_payload
    assert not any(
        path.name.startswith(".completion.json.partial-") for path in parent.iterdir()
    )


def test_postpublication_cleanup_uses_retained_parent_and_file_identities(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "base"
    parent.mkdir()
    completion = parent / "completion.json"
    publication = sealer._write_completion(completion, {"complete": True})
    moved = tmp_path / "moved-base"
    parent.rename(moved)
    parent.mkdir()
    replacement = parent / completion.name
    replacement.write_bytes(b"replacement parent completion")
    try:
        with pytest.raises(RuntimeError, match="parent.*identity changed"):
            sealer._require_published_completion_identity(completion, publication)
        assert sealer._remove_published_completion(completion, publication) is True
    finally:
        os.close(publication["parent_descriptor"])

    assert replacement.read_bytes() == b"replacement parent completion"
    assert not (moved / completion.name).exists()


def test_postpublication_cleanup_does_not_unlink_filename_replacement(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "base"
    parent.mkdir()
    completion = parent / "completion.json"
    publication = sealer._write_completion(completion, {"complete": True})
    displaced = parent / "displaced-original-completion.json"
    completion.rename(displaced)
    completion.write_bytes(b"replacement file completion")
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            sealer._require_published_completion_identity(completion, publication)
        assert sealer._remove_published_completion(completion, publication) is False
    finally:
        os.close(publication["parent_descriptor"])

    assert completion.read_bytes() == b"replacement file completion"
    assert displaced.is_file()


def test_source_contains_no_protected_numerical_deserializer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "numpy.load",
        "np.load",
        "pickle.load",
        "torch.load",
        "h5py.File",
        "PlyData.read",
    ):
        assert forbidden not in source


def test_module_load_is_stdlib_only_before_runtime_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_numpy: list[str] = []
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "numpy" or name.startswith("numpy."):
            imported_numpy.append(name)
            raise AssertionError("NumPy imported during sealer module initialization")
        return original_import(name, globals, locals, fromlist, level)

    name = "resource_qualification_v2_sealer_stdlib_preflight_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)

    assert imported_numpy == []
    assert module.np is None


def test_wrong_runtime_is_rejected_before_numpy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_attempts: list[str] = []

    def unexpected_import(name: str) -> Any:
        import_attempts.append(name)
        raise AssertionError("NumPy import must follow the runtime gate")

    fake_sys = SimpleNamespace(
        flags=SimpleNamespace(
            isolated=1,
            ignore_environment=1,
            no_user_site=1,
            dont_write_bytecode=1,
        ),
        executable="/wrong/python",
        _base_executable=sealer.PINNED_PYTHON_TARGET,
        prefix=str(sealer.PINNED_PYTHON_RUNTIME),
        base_prefix=sealer.PINNED_PYTHON_BASE_PREFIX,
        modules={},
    )
    monkeypatch.setattr(sealer, "np", None)
    monkeypatch.setattr(sealer, "sys", fake_sys)
    monkeypatch.setattr(
        sealer, "importlib", SimpleNamespace(import_module=unexpected_import)
    )

    with pytest.raises(RuntimeError, match="pinned Python launcher"):
        sealer._require_pinned_runtime_and_load_numpy()

    assert import_attempts == []


@pytest.mark.parametrize(
    ("version", "source_kind", "message"),
    [
        ("1.26.3", "expected", "version changed"),
        (sealer.PINNED_NUMPY_VERSION, "foreign", "source changed"),
    ],
)
def test_numpy_loader_rejects_wrong_version_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    source_kind: str,
    message: str,
) -> None:
    runtime = tmp_path / "runtime"
    expected = runtime / sealer.PINNED_NUMPY_SOURCE_RELATIVE
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"pinned numpy source")
    foreign = tmp_path / "foreign/numpy/__init__.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"foreign numpy source")
    module = SimpleNamespace(
        __version__=version,
        __file__=str(expected if source_kind == "expected" else foreign),
    )
    modules: dict[str, Any] = {}

    def import_module(name: str) -> Any:
        assert name == "numpy"
        modules[name] = module
        return module

    monkeypatch.setattr(sealer, "PINNED_PYTHON_RUNTIME", runtime.resolve())
    monkeypatch.setattr(sealer, "np", None)
    monkeypatch.setattr(sealer, "sys", SimpleNamespace(modules=modules))
    monkeypatch.setattr(
        sealer, "importlib", SimpleNamespace(import_module=import_module)
    )

    with pytest.raises(RuntimeError, match=message):
        sealer._load_pinned_numpy()

    assert sealer.np is None
    assert modules == {}


def test_numpy_loader_accepts_only_exact_pinned_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    expected = runtime / sealer.PINNED_NUMPY_SOURCE_RELATIVE
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"pinned numpy source")
    module = SimpleNamespace(
        __version__=sealer.PINNED_NUMPY_VERSION,
        __file__=str(expected),
    )
    modules: dict[str, Any] = {}

    def import_module(name: str) -> Any:
        assert name == "numpy"
        modules[name] = module
        return module

    monkeypatch.setattr(sealer, "PINNED_PYTHON_RUNTIME", runtime.resolve())
    monkeypatch.setattr(sealer, "np", None)
    monkeypatch.setattr(sealer, "sys", SimpleNamespace(modules=modules))
    monkeypatch.setattr(
        sealer, "importlib", SimpleNamespace(import_module=import_module)
    )

    sealer._load_pinned_numpy()

    assert sealer.np is module


def test_actual_qualifier_orchestration_output_is_sealable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real v2 aggregate builder, cleanup, then the integrity seal."""

    _install_test_numpy(monkeypatch)
    repo = Path(__file__).parents[1].resolve()
    source = tmp_path / "public"
    _source_dataset(source)
    base = tmp_path / "qualification-base"
    base.mkdir()
    deform360 = tmp_path / "deform360"
    deform360.mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    python = runtime / "bin/python"
    python.write_bytes(b"launcher")
    resolved_python = tmp_path / "usr/bin/python3.12"
    resolved = _write(resolved_python, b"resolved", 0o555)
    _write(runtime / "pyvenv.cfg", b"home=/usr\n")
    freeze = tmp_path / "runtime.freeze.sorted.txt"
    freeze_record = _write(freeze, b"x==1\n", 0o400)
    tree = tmp_path / "runtime.tree-manifest.json"
    tree_record = _write(tree, b"{}\n", 0o400)
    head = "e" * 40
    output = base / f"{sealer.ROOT_PREFIX}{head}"

    for module in (qualification, sealer):
        monkeypatch.setattr(module, "PINNED_PYTHON", python.resolve())
        monkeypatch.setattr(module, "PINNED_DEFORM360", deform360.resolve())
    monkeypatch.setattr(qualification, "DEFAULT_PUBLIC_DEV_DATASET", source.resolve())
    monkeypatch.setattr(qualification, "QUALIFICATION_BASE", base.resolve())
    monkeypatch.setattr(qualification.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(sealer, "BASE", base.resolve())
    monkeypatch.setattr(sealer, "PUBLIC_DATASET", source.resolve())
    monkeypatch.setattr(sealer.socket, "gethostname", lambda: "workstation2")
    monkeypatch.setattr(sealer, "PINNED_PYTHON_RUNTIME", runtime.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_FREEZE", freeze.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_TREE_MANIFEST", tree.resolve())
    monkeypatch.setattr(sealer, "PINNED_PYTHON_TARGET", str(resolved_python.resolve()))
    monkeypatch.setattr(sealer, "PINNED_PYTHON_TARGET_SHA256", resolved["sha256"])
    monkeypatch.setattr(sealer, "PINNED_PYTHON_FREEZE_SHA256", freeze_record["sha256"])
    monkeypatch.setattr(
        sealer, "PINNED_PYTHON_TREE_MANIFEST_SHA256", tree_record["sha256"]
    )
    monkeypatch.setattr(
        sealer, "PINNED_PYTHON_BASE_PREFIX", str((tmp_path / "usr").resolve())
    )

    code_binding = _code_binding(repo, head)
    deform_binding = {
        "path": str(deform360.resolve()),
        "head": sealer.PINNED_DEFORM360_HEAD,
        "tree": sealer.PINNED_DEFORM360_TREE,
        "clean": True,
        "ordinary_untracked_file_count": 0,
        "ignored_untracked_file_count": 0,
    }
    python_binding = _python_binding()
    parent_python = {
        "sys_executable": str(python.resolve()),
        "sys_base_executable": str(resolved_python.resolve()),
        "sys_prefix": str(runtime.resolve()),
        "sys_base_prefix": str((tmp_path / "usr").resolve()),
    }
    monkeypatch.setattr(
        qualification, "_current_python_process_binding", lambda: parent_python
    )
    monkeypatch.setattr(
        qualification, "_python_runtime_binding", lambda _python: python_binding
    )

    def git_binding(path: Path, *, expected_head: str | None = None) -> dict[str, Any]:
        return deform_binding if expected_head is not None else code_binding

    monkeypatch.setattr(qualification, "_git_binding", git_binding)

    def invoke(
        command: list[str],
        *,
        environment: dict[str, str],
        log_path: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        log_path.write_bytes(b"synthetic child completed\n")
        if "_fit-child" in command:
            mode = command[command.index("--variant") + 1]
            dataset = Path(command[command.index("--dataset") + 1])
            export = Path(command[command.index("--output-dir") + 1])
            result_path = Path(command[command.index("--result") + 1])
            (dataset / "outputs").mkdir()
            (dataset / "outputs/checkpoint.bin").write_bytes(b"checkpoint")
            ply = export / "splat.ply"
            ply.write_bytes(b"retained gaussian ply\n")
            smoke = _smoke()
            adapter = repo / sealer.RELATIVE_GSPLAT_ADAPTER_SOURCE
            child = qualification._signed(
                {
                    "schema_version": 1,
                    "artifact_kind": sealer.FIT_KIND,
                    "qualification_id": sealer.QUALIFICATION_ID,
                    "variant": mode,
                    "passed": True,
                    "parameters": {"iterations": 250, "seed": 0},
                    "runtime": _runtime(),
                    "gsplat_runtime_smoke": {
                        "adapter_source": _source_record(adapter.resolve()),
                        "evidence": smoke,
                        "evidence_artifact_sha256": smoke["artifact_sha256"],
                    },
                    "dataset": str(dataset),
                    "output": qualification._bound_file(ply),
                    "resource_boundary": {
                        "before": _boundary(),
                        "after": _boundary(),
                    },
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
            qualification._write_new_json(result_path, child)
            return_code = 0
        elif "prepare-manifest" in command:
            result_path = Path(command[command.index("--output") + 1])
            modes: dict[str, list[dict[str, Any]]] = {
                "original": [],
                "wrapped": [],
            }
            for mode in modes:
                flag = f"--{mode}"
                cursor = 0
                while flag in command[cursor:]:
                    position = command.index(flag, cursor)
                    pairing_id, ply_value, evidence_value = command[
                        position + 1 : position + 4
                    ]
                    evidence = json.loads(Path(evidence_value).read_text())
                    modes[mode].append(
                        {
                            "pairing_id": pairing_id,
                            "ply": _descriptor(
                                sealer._stable_file(Path(ply_value), role="PLY")
                            ),
                            "fit_evidence": _descriptor(
                                sealer._stable_file(
                                    Path(evidence_value), role="fit evidence"
                                )
                            ),
                            "dataset_input_inventory": _dataset_closure(
                                Path(evidence["dataset"])
                            ),
                        }
                    )
                    cursor = position + 4
            manifest = _manifest_payload(
                source=source,
                repo=repo,
                code=code_binding,
                modes=modes,
            )
            qualification._write_new_json(result_path, manifest)
            return_code = 0
        elif "analyze" in command:
            result_path = Path(command[command.index("--output") + 1])
            manifest_path = Path(command[command.index("--manifest") + 1])
            manifest = json.loads(manifest_path.read_text())
            analysis = _analysis_payload(
                accepted=True,
                root=output,
                repo=repo,
                code=code_binding,
                manifest=manifest,
                manifest_record=sealer._stable_file(manifest_path, role="manifest"),
            )
            qualification._write_new_json(result_path, analysis)
            return_code = 0
        elif "_soak-child" in command:
            dataset = Path(command[command.index("--dataset") + 1])
            result_path = Path(command[command.index("--result") + 1])
            qualification._write_new_json(result_path, _soak_child(dataset))
            return_code = 0
        else:  # pragma: no cover
            raise AssertionError(command)
        return {
            "command": list(command),
            "environment": dict(environment),
            "return_code": return_code,
            "timed_out": False,
            "timeout_error": None,
            "timeout_seconds": timeout_seconds,
            "log": qualification._bound_file(log_path),
        }

    monkeypatch.setattr(qualification, "_invoke_child", invoke)
    arguments = SimpleNamespace(
        code_root=repo,
        deform360_repo=deform360,
        dataset=source,
        output_dir=output,
        python=python,
        phase="all",
        cuda_device=1,
        seed=0,
        ab_iterations=250,
        ab_repeat_count=5,
        soak_fit_count=243,
        soak_iterations=1,
        first_fit_fd_growth_limit=32,
        steady_fd_growth_limit=4,
        steady_task_growth_limit=4,
        fit_timeout_seconds=3_600,
        analyzer_timeout_seconds=86_400,
        soak_timeout_seconds=86_400,
    )

    assert qualification._run(arguments) == 0
    completion = Path(f"{output}-integrity-completion.json")
    sealed = sealer.seal(output, completion)
    assert sealed["admission_eligible"] is True
