"""Fail-closed provenance for deliberately transductive MatPhys controls."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .matphys_causal_bridge import sha256_file
from .matphys_graph_parts import (
    GRAPH_PART_COMPACT_PROXY_CONTRACT,
    GRAPH_PART_PROXY_CONTRACT,
)
from .matphys_part_model import PART_AWARE_MODEL_CONTRACT

MATPHYS_RECONSTRUCTION_AUDIT_SCHEMA_VERSION = 2
MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT = (
    "matphys-per-case-all-frame-part-aware-reconstruction-audit-v2"
)
MATPHYS_RECONSTRUCTION_VIDEO_SCOPE = "uniform-numeric-full-sequence-v1"
MATPHYS_RECONSTRUCTION_TRAINING_SCOPE = "per-case-all-frame-transductive-v1"
MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY = "fixed-terminal-epoch-v1"
MATPHYS_RECONSTRUCTION_WARP_WARNING_COMPATIBILITY = "warp-private-warn-signature-v1"
MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY = (
    "matphys-single-case-provisional-split-v1"
)
MATPHYS_RECONSTRUCTION_OBJECTIVE_GUARD = "exact-full-sequence-objective-v1"
MATPHYS_RECONSTRUCTION_PROXY_CONTRACTS = frozenset(
    (GRAPH_PART_PROXY_CONTRACT, GRAPH_PART_COMPACT_PROXY_CONTRACT)
)
MATPHYS_RECONSTRUCTION_EXPORT_CONTRACT = (
    "matphys-all-frame-part-aware-reconstruction-export-v2"
)
MATPHYS_RECONSTRUCTION_RESULT_CONTRACT = (
    "matphys-all-frame-part-aware-reconstruction-result-v1"
)
MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY = (
    "This checkpoint used future RGB, geometry, and track observations from the "
    "same case. It is an offline reconstruction-capacity control, not a causal "
    "forecast, transfer result, calibration result, or state-of-the-art claim."
)


def _identity(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": sha256_file(source)}


def _validate_identity(value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity must be an object")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != value.get("sha256"):
        raise ValueError(f"{label} bytes changed")
    return path


def _validate_proxy(proxy_path: Path, case_name: str) -> dict[str, object]:
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    if proxy.get("contract") not in MATPHYS_RECONSTRUCTION_PROXY_CONTRACTS:
        raise ValueError("reconstruction proxy uses an unsupported contract")
    records = proxy.get("cases")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("reconstruction proxy must contain exactly one case")
    if records[0].get("name") != case_name:
        raise ValueError("reconstruction proxy case changed")
    mapping_path = _validate_identity(proxy.get("mapping"), label="proxy mapping")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    case_mapping = mapping.get("case_to_material")
    if not isinstance(case_mapping, Mapping) or set(case_mapping) != {case_name}:
        raise ValueError("reconstruction proxy mapping must contain exactly its case")
    for key in ("node_sem", "train_ready"):
        _validate_identity(records[0].get(key), label=f"proxy {key}")
    semantic_dimension = records[0].get("semantic_dimension")
    if (
        isinstance(semantic_dimension, bool)
        or not isinstance(semantic_dimension, int)
        or semantic_dimension < 1
    ):
        raise ValueError("reconstruction proxy omits its semantic dimension")
    return proxy


def _validate_training_configuration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("reconstruction training configuration must be an object")
    configuration = dict(value)
    required = {
        "fit_all_frames": True,
        "video_scope": MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
        "training_scope": MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
        "checkpoint_policy": MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
        "warp_warning_compatibility": (
            MATPHYS_RECONSTRUCTION_WARP_WARNING_COMPATIBILITY
        ),
        "single_case_loader_compatibility": (
            MATPHYS_RECONSTRUCTION_SINGLE_CASE_LOADER_COMPATIBILITY
        ),
        "objective_guard": MATPHYS_RECONSTRUCTION_OBJECTIVE_GUARD,
        "part_model_contract": PART_AWARE_MODEL_CONTRACT,
    }
    for key, expected in required.items():
        if configuration.get(key) != expected:
            raise ValueError(f"reconstruction training configuration violates {key}")
    if (
        configuration.get("proxy_contract")
        not in MATPHYS_RECONSTRUCTION_PROXY_CONTRACTS
    ):
        raise ValueError(
            "reconstruction training configuration violates proxy_contract"
        )
    epochs = configuration.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("reconstruction epochs must be a positive integer")
    semantic_dimension = configuration.get("semantic_dimension")
    if (
        isinstance(semantic_dimension, bool)
        or not isinstance(semantic_dimension, int)
        or semantic_dimension < 1
    ):
        raise ValueError("reconstruction semantic dimension must be positive")
    part_feature_scale = configuration.get("part_feature_scale")
    if (
        isinstance(part_feature_scale, bool)
        or not isinstance(part_feature_scale, (int, float))
        or not math.isfinite(float(part_feature_scale))
        or not 0.0 < float(part_feature_scale)
    ):
        raise ValueError("reconstruction part feature scale must be positive")
    return configuration


def validate_matphys_reconstruction_protocol(
    protocol_path: str | Path,
    *,
    case_name: str,
    source_commit: str,
    training_configuration: Mapping[str, object],
) -> dict[str, object]:
    """Fail before training when a reconstruction request differs from its lock."""

    source = Path(protocol_path).resolve()
    protocol = json.loads(source.read_text(encoding="utf-8"))
    protocol_case = protocol.get("case")
    protocol_implementation = protocol.get("implementation")
    if not isinstance(protocol_case, Mapping) or not isinstance(
        protocol_implementation, Mapping
    ):
        raise ValueError("reconstruction protocol omits case or implementation")
    if protocol_case.get("case_id") != case_name:
        raise ValueError("reconstruction case differs from protocol")
    if protocol_implementation.get("matphys_revision") != source_commit:
        raise ValueError("MatPhys revision differs from reconstruction protocol")
    configuration = _validate_training_configuration(training_configuration)
    protocol_training = {
        "epochs": protocol_implementation.get("epochs"),
        "eval_every": protocol_implementation.get("eval_every"),
        "learning_rate": protocol_implementation.get("learning_rate"),
        "random_seed": protocol_implementation.get("random_seed"),
        "fit_all_frames": protocol_implementation.get("fit_all_frames"),
        "proxy_contract": protocol_implementation.get("proxy_contract"),
        "part_model_contract": protocol_implementation.get("part_model_contract"),
        "part_feature_scale": protocol_implementation.get("part_feature_scale"),
        "warp_warning_compatibility": protocol_implementation.get(
            "warp_warning_compatibility"
        ),
        "single_case_loader_compatibility": protocol_implementation.get(
            "single_case_loader_compatibility"
        ),
        "objective_guard": protocol_implementation.get("objective_guard"),
    }
    audited_training = {key: configuration.get(key) for key in protocol_training}
    if protocol_training != audited_training:
        raise ValueError("reconstruction training settings differ from protocol")
    return protocol


def write_matphys_reconstruction_audit(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    protocol_path: str | Path,
    source_repository: str,
    source_commit: str,
    data_root: str | Path,
    case_name: str,
    split_path: str | Path,
    accessed_frame_indices: Sequence[int],
    accessed_frame_paths: Mapping[int, str | Path],
    objective_end_frame_exclusive: int,
    proxy_summary_path: str | Path,
    training_configuration: Mapping[str, object],
    runtime_access_log_paths: Sequence[str | Path] = (),
    implementation_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Bind a terminal all-frame checkpoint to every future-bearing input."""

    if not source_repository or not source_commit:
        raise ValueError("MatPhys repository and commit are required")
    if not case_name:
        raise ValueError("case name is required")
    root = Path(data_root).resolve()
    split_source = Path(split_path).resolve()
    expected_split = (root / case_name / "split.json").resolve()
    if split_source != expected_split:
        raise ValueError("split must be the selected case's released split")
    split = json.loads(split_source.read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    frame_len = int(split.get("frame_len", split["test"][1]))
    if not 1 < train_end < frame_len:
        raise ValueError("selected case has no distinct fitted-future interval")
    if int(objective_end_frame_exclusive) != frame_len:
        raise ValueError("reconstruction objective must consume the full sequence")

    indices = sorted(set(int(value) for value in accessed_frame_indices))
    if not indices or indices[0] < 0 or indices[-1] >= frame_len:
        raise ValueError("reconstruction video frame ids are outside the sequence")
    if not any(frame_id >= train_end for frame_id in indices):
        raise ValueError("reconstruction control did not access future RGB")
    if set(indices) != {int(value) for value in accessed_frame_paths}:
        raise ValueError("exact reconstruction frame sources are incomplete")
    frame_files = []
    color_root = (root / case_name / "color" / "0").resolve()
    for frame_id in indices:
        frame_path = Path(accessed_frame_paths[frame_id]).resolve()
        if frame_path.parent != color_root or int(frame_path.stem) != frame_id:
            raise ValueError("reconstruction frame source changed identity")
        frame_files.append({"frame_id": frame_id, **_identity(frame_path)})

    configuration = _validate_training_configuration(training_configuration)
    validate_matphys_reconstruction_protocol(
        protocol_path,
        case_name=case_name,
        source_commit=source_commit,
        training_configuration=configuration,
    )

    proxy_path = Path(proxy_summary_path).resolve()
    _validate_proxy(proxy_path, case_name)
    audit = {
        "schema_version": MATPHYS_RECONSTRUCTION_AUDIT_SCHEMA_VERSION,
        "contract": MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT,
        "claim_boundary": MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
        "future_observations_used": True,
        "future_rgb_used": True,
        "future_geometry_and_track_targets_used": True,
        "predictive_use_authorized": False,
        "fit_all_frames": True,
        "frame_id_contract": "numeric-filename-stem-v2",
        "video_sampling": MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
        "training_scope": MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
        "checkpoint_policy": MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "data_root": str(root),
        "protocol": _identity(protocol_path),
        "checkpoint": _identity(checkpoint_path),
        "proxy": _identity(proxy_path),
        "runtime_access_logs": [_identity(path) for path in runtime_access_log_paths],
        "implementation_files": [_identity(path) for path in implementation_paths],
        "training_configuration": configuration,
        "case": {
            "name": case_name,
            "split": {**_identity(split_source), **split},
            "released_train_end_frame_exclusive": train_end,
            "frame_len": frame_len,
            "accessed_frame_indices": indices,
            "accessed_frame_files": frame_files,
            "objective_frame_interval": [1, frame_len],
            "fitted_future_interval": [train_end, frame_len],
        },
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **audit,
        "audit_path": str(destination),
        "audit_sha256": sha256_file(destination),
    }


def validate_matphys_reconstruction_audit(
    audit_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Validate a reconstruction control without granting predictive use."""

    source = Path(audit_path).resolve()
    audit = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "schema_version": MATPHYS_RECONSTRUCTION_AUDIT_SCHEMA_VERSION,
        "contract": MATPHYS_RECONSTRUCTION_AUDIT_CONTRACT,
        "claim_boundary": MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
        "future_observations_used": True,
        "future_rgb_used": True,
        "future_geometry_and_track_targets_used": True,
        "predictive_use_authorized": False,
        "fit_all_frames": True,
        "frame_id_contract": "numeric-filename-stem-v2",
        "video_sampling": MATPHYS_RECONSTRUCTION_VIDEO_SCOPE,
        "training_scope": MATPHYS_RECONSTRUCTION_TRAINING_SCOPE,
        "checkpoint_policy": MATPHYS_RECONSTRUCTION_CHECKPOINT_POLICY,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"unsupported MatPhys reconstruction audit: {key}")
    checkpoint = _validate_identity(audit.get("checkpoint"), label="checkpoint")
    requested_checkpoint = Path(checkpoint_path).resolve()
    if checkpoint != requested_checkpoint:
        raise ValueError("checkpoint is not bound by the reconstruction audit")
    protocol_path = _validate_identity(audit.get("protocol"), label="protocol")
    proxy_path = _validate_identity(audit.get("proxy"), label="proxy")
    case = audit.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("reconstruction audit omits its case")
    case_name = str(case.get("name", ""))
    _validate_proxy(proxy_path, case_name)
    split_path = _validate_identity(case.get("split"), label="split")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_end = int(split["train"][1])
    frame_len = int(split.get("frame_len", split["test"][1]))
    if case.get("fitted_future_interval") != [train_end, frame_len]:
        raise ValueError("reconstruction fitted-future interval changed")
    if case.get("objective_frame_interval") != [1, frame_len]:
        raise ValueError("reconstruction objective interval changed")
    frame_files = case.get("accessed_frame_files")
    indices = [int(value) for value in case.get("accessed_frame_indices", [])]
    if not isinstance(frame_files, list) or len(frame_files) != len(indices):
        raise ValueError("reconstruction frame sources are incomplete")
    bound_ids = []
    for frame in frame_files:
        frame_path = _validate_identity(frame, label="reconstruction frame")
        frame_id = int(frame["frame_id"])
        if int(frame_path.stem) != frame_id:
            raise ValueError("reconstruction frame id changed")
        bound_ids.append(frame_id)
    if sorted(bound_ids) != sorted(indices) or not any(
        frame_id >= train_end for frame_id in indices
    ):
        raise ValueError("reconstruction future RGB binding changed")
    for key, label in (
        ("runtime_access_logs", "runtime access log"),
        ("implementation_files", "implementation file"),
    ):
        values = audit.get(key)
        if not isinstance(values, list):
            raise ValueError(f"reconstruction audit omits {label}s")
        for value in values:
            _validate_identity(value, label=label)
    configuration = _validate_training_configuration(
        audit.get("training_configuration")
    )
    validate_matphys_reconstruction_protocol(
        protocol_path,
        case_name=case_name,
        source_commit=str(audit.get("source_commit", "")),
        training_configuration=configuration,
    )
    return {
        **audit,
        "audit_path": str(source),
        "audit_sha256": sha256_file(source),
    }


def build_matphys_reconstruction_result(
    audit_path: str | Path,
    checkpoint_path: str | Path,
    export_manifest_path: str | Path,
    baseline_metrics_path: str | Path,
) -> dict[str, object]:
    """Recompute the locked capacity decision from byte-bound terminal artifacts."""

    audit = validate_matphys_reconstruction_audit(audit_path, checkpoint_path)
    manifest_path = Path(export_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": 2,
        "contract": MATPHYS_RECONSTRUCTION_EXPORT_CONTRACT,
        "claim_boundary": MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
        "future_observations_used": True,
        "predictive_use_authorized": False,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unsupported MatPhys reconstruction export: {key}")
    method = manifest.get("method")
    if not isinstance(method, Mapping) or method.get("published_matphys_method") is not False:
        raise ValueError("reconstruction export mislabels the method")
    if method.get("part_model_contract") != PART_AWARE_MODEL_CONTRACT:
        raise ValueError("reconstruction export changed the part model")
    if _validate_identity(manifest.get("checkpoint"), label="export checkpoint") != Path(
        checkpoint_path
    ).resolve():
        raise ValueError("reconstruction export did not use the terminal checkpoint")
    if _validate_identity(
        manifest.get("training_audit"), label="training audit"
    ) != Path(audit_path).resolve():
        raise ValueError("reconstruction export changed the training audit")
    case = manifest.get("case")
    if not isinstance(case, Mapping) or case.get("name") != audit["case"]["name"]:
        raise ValueError("reconstruction export changed the case")

    spring = case.get("spring_field")
    if not isinstance(spring, Mapping):
        raise ValueError("reconstruction export omits the spring field")
    _validate_identity(spring, label="spring field")
    spring_values = {
        key: float(spring.get(key, float("nan")))
        for key in ("minimum", "maximum", "geometric_mean")
    }
    spring_count = spring.get("count")
    if (
        isinstance(spring_count, bool)
        or not isinstance(spring_count, int)
        or spring_count < 1
        or not all(math.isfinite(value) and value > 0.0 for value in spring_values.values())
        or not spring_values["minimum"] <= spring_values["maximum"]
    ):
        raise ValueError("reconstruction export has an invalid spring field")
    globals_path = _validate_identity(
        case.get("global_parameters"), label="global parameters"
    )
    global_parameters = json.loads(globals_path.read_text(encoding="utf-8"))
    expected_globals = {
        "collide_elas",
        "collide_fric",
        "collide_object_elas",
        "collide_object_fric",
        "collision_dist",
        "dashpot_damping",
        "drag_damping",
    }
    if set(global_parameters) != expected_globals or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in global_parameters.values()
    ):
        raise ValueError("reconstruction export has invalid global parameters")

    candidate_path = _validate_identity(
        case.get("official_metrics"), label="candidate metrics"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_path = Path(baseline_metrics_path).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for label, payload in (("candidate", candidate), ("baseline", baseline)):
        inputs = payload.get("inputs")
        evaluation = payload.get("evaluation")
        if not isinstance(inputs, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError(f"{label} official metrics are incomplete")
        for split_name in ("train", "test"):
            values = evaluation.get(split_name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{label} official metrics omit {split_name}")
            for metric in ("chamfer_distance_m", "track_error_m"):
                value = values.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"{label} official metric is invalid: {metric}")
    for name in ("final_data", "gt_track_3d", "split"):
        if candidate["inputs"].get(name) != baseline["inputs"].get(name):
            raise ValueError(f"candidate and baseline {name} identities differ")
    if candidate.get("split") != baseline.get("split"):
        raise ValueError("candidate and baseline split metadata differ")

    finiteness = []
    for identity in audit["runtime_access_logs"]:
        path = Path(str(identity["path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract") == "finite-model-and-optimizer-checkpoint-v1":
            finiteness.append(payload)
    if len(finiteness) != 1 or finiteness[0].get("finite") is not True:
        raise ValueError("reconstruction audit lacks one finite checkpoint report")
    if finiteness[0].get("checkpoint") != audit["checkpoint"]:
        raise ValueError("finiteness report changed the terminal checkpoint")

    candidate_test = candidate["evaluation"]["test"]
    baseline_test = baseline["evaluation"]["test"]
    metric_names = ("chamfer_distance_m", "track_error_m")
    improvements = {
        name: 100.0
        * (float(baseline_test[name]) - float(candidate_test[name]))
        / float(baseline_test[name])
        for name in metric_names
    }
    capacity_pass = all(improvements[name] > 0.0 for name in metric_names)
    return {
        "schema_version": 1,
        "contract": MATPHYS_RECONSTRUCTION_RESULT_CONTRACT,
        "claim_boundary": MATPHYS_RECONSTRUCTION_CLAIM_BOUNDARY,
        "future_observations_used": True,
        "predictive_use_authorized": False,
        "published_matphys_method": False,
        "artifacts": {
            "audit": _identity(audit_path),
            "terminal_checkpoint": _identity(checkpoint_path),
            "export_manifest": _identity(manifest_path),
            "candidate_metrics": _identity(candidate_path),
            "released_phystwin_metrics": _identity(baseline_path),
        },
        "case": audit["case"]["name"],
        "terminal_test_metrics_mm": {
            name.removesuffix("_m"): 1000.0 * float(candidate_test[name])
            for name in metric_names
        },
        "released_phystwin_test_metrics_mm": {
            name.removesuffix("_m"): 1000.0 * float(baseline_test[name])
            for name in metric_names
        },
        "improvement_percent": improvements,
        "backend_export": {
            "complete_positive_spring_field": True,
            "spring_count": spring_count,
            "finite_global_parameter_count": len(global_parameters),
        },
        "decision": {
            "capacity_pass": capacity_pass,
            "backend_export_pass": True,
            "advance_to_source_only_causal_design": capacity_pass,
            "authorizes_predictive_use_of_checkpoint": False,
        },
    }
