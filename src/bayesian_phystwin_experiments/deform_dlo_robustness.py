"""Prospective custody and source-split contracts for DEFORM DLO robustness."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from bayesian_phystwin.numerical_linear_algebra_v1 import solve_spd
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _collapse_duplicate_queries,
    _finite_array,
    build_deform_local_residual_features,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

DEFORM_DLO_ROBUSTNESS_CONTRACT = "deform-dlo-robustness-v1"
DEFORM_DLO_ROBUSTNESS_DOMAIN = b"deform-dlo3-robustness-v1\0"
DEFORM_LOCAL_FEATURE_COUNT = 92
Array = np.ndarray[Any, Any]


def deform_local_feature_indices(arm: str) -> tuple[int, ...]:
    """Return the predeclared feature subset for one mechanism arm."""

    if arm in ("full-local", "full-global"):
        return tuple(range(DEFORM_LOCAL_FEATURE_COUNT))
    if arm == "intercept-only":
        return ()
    if arm == "full-no-action":
        action_dependent = set(range(24, 66)) | {69, 70} | set(range(80, 92))
        return tuple(
            index
            for index in range(DEFORM_LOCAL_FEATURE_COUNT)
            if index not in action_dependent
        )
    raise ValueError(f"unsupported DEFORM mechanism arm: {arm}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integers(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an integer sequence")
    result = tuple(int(cast(Any, item)) for item in value)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a string sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{label} must not be empty")
    return result


def load_deform_dlo_robustness_v1_protocol(path: str | Path) -> dict[str, object]:
    """Load the immutable DLO3 transfer and robustness protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DEFORM DLO robustness schema")
    if payload.get("contract") != DEFORM_DLO_ROBUSTNESS_CONTRACT:
        raise ValueError("unsupported DEFORM DLO robustness contract")
    if payload.get("prob4d_used") is not False:
        raise ValueError("DEFORM DLO robustness must keep Prob4D unused")

    upstream = _mapping(payload.get("upstream"), label="upstream")
    parent = _mapping(payload.get("parent_method"), label="parent method")
    if (
        upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or upstream.get("commit") != "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
        or upstream.get("train_script_sha256")
        != "d45abe23a22b0f01fa266833844c4f9b71a2b7e375f8e955e3278b9e969acc55"
        or parent.get("source_revision") != "2cdbff202b2b000a96c6eddf9e750999ee9f6e75"
        or parent.get("source_archive_sha256")
        != "f0e2ac2d1166f1e95e2ca7ba70ee109a63cc7e110687c7a863cefb0cd8322d69"
        or parent.get("arm") != "r1_s0p25"
    ):
        raise ValueError("DEFORM DLO robustness lineage differs")

    freshness = _mapping(payload.get("freshness"), label="freshness")
    data = _mapping(payload.get("data"), label="data")
    if (
        freshness.get("primary_dlo") != "DLO3"
        or _strings(freshness.get("reserve_dlos"), label="reserve DLOs")
        != ("DLO4", "DLO5")
        or freshness.get("project_payload_previously_read") is not False
        or freshness.get("primary_eval_read") is not False
        or freshness.get("reserve_payload_read") is not False
        or freshness.get("reserve_substitution") is not False
        or data.get("train_partition") != "DLO3/train"
        or data.get("eval_partition") != "DLO3/eval"
        or int(cast(Any, data.get("train_trajectory_count", -1))) != 56
        or int(cast(Any, data.get("eval_trajectory_count", -1))) != 14
        or int(cast(Any, data.get("frame_count", -1))) != 500
        or int(cast(Any, data.get("node_count", -1))) != 12
        or data.get("coordinate_transform") != "raw-x-raw-z-raw-y"
        or _integers(data.get("known_action_nodes"), label="action nodes")
        != (0, 1, -2, -1)
    ):
        raise ValueError("DEFORM DLO robustness data boundary differs")

    split = _mapping(payload.get("source_split"), label="source split")
    if (
        split.get("domain_separator") != DEFORM_DLO_ROBUSTNESS_DOMAIN.decode()
        or split.get("identity_operator") != "sha256-domain-separated-utf8-basename-v1"
        or split.get("ordering") != "ascending-hex-digest-v1"
        or int(cast(Any, split.get("fit_count", -1))) != 39
        or int(cast(Any, split.get("calibration_count", -1))) != 9
        or int(cast(Any, split.get("source_test_count", -1))) != 8
        or split.get("manifest_before_payload_read") is not True
    ):
        raise ValueError("DEFORM DLO robustness source split differs")

    training = _mapping(payload.get("physical_training"), label="physical training")
    residual = _mapping(payload.get("local_residual"), label="local residual")
    if (
        training.get("backend") != "official-DEFORM-PBD"
        or int(cast(Any, training.get("primary_seed", -1))) != 42
        or _integers(training.get("audit_seeds"), label="audit seeds") != (42, 43, 44)
        or int(cast(Any, training.get("unroll_horizon_frames", -1))) != 50
        or int(cast(Any, training.get("batch_size", -1))) != 32
        or int(cast(Any, training.get("total_updates", -1))) != 6400
        or _integers(training.get("checkpoint_updates"), label="checkpoints")
        != (0, 280, 640, 1280, 2560, 4000, 5200, 6040, 6400)
        or int(cast(Any, training.get("pbd_iterations", -1))) != 10
        or training.get("source_fit_policy") != "fit-39-only"
        or training.get("final_fit_policy") != "all-56-train"
        or training.get("target_seed_selection") is not False
        or residual.get("operator") != "per-node-trajectory-grouped-bayesian-ridge-v1"
        or float(cast(Any, residual.get("ridge", math.nan))) != 1.0
        or float(cast(Any, residual.get("shrinkage", math.nan))) != 0.25
        or float(cast(Any, residual.get("coordinate_variance_floor_m2", math.nan)))
        != 1e-6
        or residual.get("source_reselection") is not False
        or residual.get("target_reselection") is not False
    ):
        raise ValueError("DEFORM DLO robustness fixed recipe differs")

    source_gate = _mapping(payload.get("source_gate"), label="source gate")
    stability = _mapping(payload.get("training_stability"), label="stability")
    if (
        source_gate.get("required_for_target") is not True
        or float(cast(Any, source_gate.get("minimum_relative_improvement", math.nan)))
        != 0.01
        or int(cast(Any, source_gate.get("minimum_case_wins", -1))) != 6
        or float(cast(Any, source_gate.get("maximum_case_ratio", math.nan))) != 1.10
        or float(cast(Any, source_gate.get("maximum_candidate_l1_m", math.nan)))
        != 0.0077
        or int(cast(Any, stability.get("minimum_seed_source_passes", -1))) != 2
        or float(cast(Any, stability.get("maximum_seed_mean_ratio", math.nan))) != 1.10
        or float(cast(Any, stability.get("maximum_seed_case_ratio", math.nan))) != 1.25
        or stability.get("required_for_target") is not True
        or stability.get("seed_selection") is not False
    ):
        raise ValueError("DEFORM DLO robustness source gates differ")

    ablation = _mapping(payload.get("mechanism_ablation"), label="ablation")
    expected_arms = (
        "physical-only",
        "persistence-plus-full-local",
        "physical-plus-intercept-only",
        "physical-plus-full-no-action",
        "physical-plus-full-global-frame",
        "physical-plus-full-local-unshrunk",
        "physical-plus-full-local-fixed",
    )
    compute = _mapping(payload.get("compute_matched_control"), label="compute")
    sensitivity = _mapping(
        payload.get("physics_solver_sensitivity"), label="sensitivity"
    )
    if (
        _strings(ablation.get("arms"), label="ablation arms") != expected_arms
        or ablation.get("selection_effect") != "none"
        or int(cast(Any, compute.get("start_update", -1))) != 6400
        or int(cast(Any, compute.get("minimum_additional_updates", -1))) != 1
        or int(cast(Any, compute.get("maximum_additional_updates", -1))) != 512
        or compute.get("target_selection") is not False
        or _integers(sensitivity.get("pbd_iteration_values"), label="PBD values")
        != (5, 10, 20)
        or tuple(
            float(value)
            for value in cast(
                Sequence[Any], sensitivity.get("joint_bend_twist_multipliers")
            )
        )
        != (0.9, 1.0, 1.1)
        or sensitivity.get("selection_effect") != "none"
    ):
        raise ValueError("DEFORM DLO robustness diagnostic matrix differs")

    backend = _mapping(payload.get("backend_portability"), label="backend")
    bank = _mapping(backend.get("parameter_bank"), label="backend parameter bank")
    if (
        backend.get("backend") != "PyElastica-CosseratRod"
        or backend.get("version") != "1.0.0"
        or backend.get("commit") != "b087f1399f9be2fdd2fcf3768689f7735a96f7ab"
        or tuple(
            float(value) for value in cast(Sequence[Any], bank.get("youngs_modulus_pa"))
        )
        != (1e5, 1e6, 1e7)
        or tuple(
            float(value) for value in cast(Sequence[Any], bank.get("density_kg_m3"))
        )
        != (900.0, 1200.0)
        or tuple(
            float(value) for value in cast(Sequence[Any], bank.get("damping_constant"))
        )
        != (0.1, 1.0)
        or _integers(bank.get("integration_substeps"), label="backend substeps")
        != (2, 4, 8)
        or backend.get("target_authorized_only_if_source_gate_passes") is not True
    ):
        raise ValueError("DEFORM DLO robustness backend contract differs")

    bayesian = _mapping(payload.get("bayesian_audit"), label="Bayesian audit")
    target = _mapping(payload.get("target_evaluation"), label="target evaluation")
    custody = _mapping(payload.get("custody"), label="custody")
    if (
        bayesian.get("point_mean") != "unchanged-r1-s0p25"
        or bayesian.get("primary_covariance")
        != "trajectory-clustered-full-coordinate-covariance-v1"
        or bayesian.get("cross_coordinate_covariance") is not True
        or bayesian.get("temporal_independence_claimed") is not False
        or int(cast(Any, bayesian.get("calibration_partition_count", -1))) != 9
        or int(cast(Any, bayesian.get("calibration_rank", -1))) != 9
        or target.get("partition") != "DLO3/eval"
        or target.get("one_shot") is not True
        or float(cast(Any, target.get("published_reference_l1_m", math.nan))) != 0.0077
        or _integers(
            target.get("canonical_reference_draw_indices"),
            label="canonical reference draw",
        )
        != (1, 7, 9, 7, 11, 7, 13, 8, 8, 6, 8, 5, 8, 4)
        or target.get("target_selection") is not False
        or target.get("target_calibration") is not False
        or target.get("target_retries") is not False
        or target.get("case_replacement") is not False
        or custody.get("target_authorization_before_target_enumeration") is not True
        or custody.get("target_outcomes_may_not_change_any_arm") is not True
        or custody.get("held_v8_access") is not False
    ):
        raise ValueError("DEFORM DLO robustness Bayesian or target contract differs")

    result = dict(payload)
    result["protocol_path"] = str(source)
    return result


def assign_deform_dlo3_source_partitions(
    names: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Assign source identities before any trajectory payload is loaded."""

    split = _mapping(protocol.get("source_split"), label="source split")
    expected = sum(
        int(cast(Any, split[key]))
        for key in ("fit_count", "calibration_count", "source_test_count")
    )
    normalized = tuple(str(name) for name in names)
    if len(normalized) != expected or len(set(normalized)) != expected:
        raise ValueError("DLO3 source names are incomplete or duplicated")
    if any(
        Path(name).name != name or Path(name).suffix != ".pkl" for name in normalized
    ):
        raise ValueError("DLO3 source identity must be a pickle basename")
    identities = {
        name: hashlib.sha256(
            DEFORM_DLO_ROBUSTNESS_DOMAIN + name.encode("utf-8")
        ).hexdigest()
        for name in normalized
    }
    ordered = sorted(normalized, key=identities.__getitem__)
    fit_count = int(cast(Any, split["fit_count"]))
    calibration_count = int(cast(Any, split["calibration_count"]))
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-source-partitions-v1",
        "payload_read": False,
        "domain_separator_sha256": hashlib.sha256(
            DEFORM_DLO_ROBUSTNESS_DOMAIN
        ).hexdigest(),
        "fit": tuple(ordered[:fit_count]),
        "calibration": tuple(ordered[fit_count : fit_count + calibration_count]),
        "source_test": tuple(ordered[fit_count + calibration_count :]),
        "identity_sha256": identities,
    }


def build_deform_dlo3_source_manifest(
    protocol_path: str | Path,
    data_root: str | Path,
) -> dict[str, object]:
    """Hash and partition DLO3 train files without deserializing trajectories."""

    protocol_source = Path(protocol_path).resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_source)
    root = Path(data_root).resolve()
    train_root = root / "DLO3" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    paths = tuple(sorted(train_root.glob("*.pkl"), key=lambda path: path.name))
    expected = int(
        cast(Any, _mapping(protocol["data"], label="data")["train_trajectory_count"])
    )
    if len(paths) != expected:
        raise ValueError(
            f"DLO3 expected {expected} train trajectories, got {len(paths)}"
        )
    assignment = assign_deform_dlo3_source_partitions(
        [path.name for path in paths], protocol
    )
    identities = {
        path.name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-manifest-v1",
        "protocol": {
            "path": str(protocol_source),
            "sha256": sha256_file(protocol_source),
        },
        "dlo_type": "DLO3",
        "partition": "train",
        "trajectory_deserialized": False,
        "primary_eval_enumerated": False,
        "reserve_payload_enumerated": False,
        "official_eval_read": False,
        "trajectories": identities,
        "split": {
            "fit": list(cast(tuple[str, ...], assignment["fit"])),
            "calibration": list(cast(tuple[str, ...], assignment["calibration"])),
            "source_test": list(cast(tuple[str, ...], assignment["source_test"])),
        },
        "source_identity_sha256": assignment["identity_sha256"],
        "domain_separator_sha256": assignment["domain_separator_sha256"],
    }


def validate_deform_dlo3_source_manifest(
    manifest: Mapping[str, object],
    protocol: Mapping[str, object],
    *,
    protocol_sha256: str,
    verify_files: bool,
) -> dict[str, tuple[str, ...]]:
    """Validate source custody and optionally rehash every bound train file."""

    identity = _mapping(manifest.get("protocol"), label="manifest protocol")
    trajectories = _mapping(manifest.get("trajectories"), label="trajectories")
    split = _mapping(manifest.get("split"), label="manifest split")
    if (
        manifest.get("contract") != "deform-dlo3-robustness-source-manifest-v1"
        or identity.get("sha256") != protocol_sha256
        or manifest.get("dlo_type") != "DLO3"
        or manifest.get("partition") != "train"
        or manifest.get("trajectory_deserialized") is not False
        or manifest.get("primary_eval_enumerated") is not False
        or manifest.get("reserve_payload_enumerated") is not False
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("DLO3 source manifest custody differs")
    partitions = {
        name: _strings(split.get(name), label=f"manifest {name}")
        for name in ("fit", "calibration", "source_test")
    }
    expected = assign_deform_dlo3_source_partitions(
        tuple(str(name) for name in trajectories), protocol
    )
    if any(partitions[name] != expected[name] for name in partitions):
        raise ValueError("DLO3 source manifest partition differs")
    for name, value in trajectories.items():
        record = _mapping(value, label=f"trajectory {name}")
        path = Path(str(record.get("path", ""))).resolve()
        size = int(cast(Any, record.get("size_bytes", -1)))
        digest = str(record.get("sha256", ""))
        if Path(name).name != name or len(digest) != 64 or size <= 0:
            raise ValueError("DLO3 source trajectory identity is invalid")
        if verify_files and (
            not path.is_file()
            or path.name != name
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError(f"DLO3 source trajectory identity changed: {name}")
    return partitions


def fit_deform_local_residual_variant(
    initial_states: Array,
    clamped_action: Array,
    baseline_predictions: Array,
    targets: Array,
    names: Sequence[str],
    *,
    ridge: float,
    arm: str,
) -> dict[str, object]:
    """Fit a predeclared reduced-feature residual arm for mechanism diagnosis."""

    initial = _finite_array(initial_states, ndim=4, label="variant initial states")
    action = _finite_array(clamped_action, ndim=4, label="variant action")
    baseline = _finite_array(baseline_predictions, ndim=4, label="variant baseline")
    observed = _finite_array(targets, ndim=4, label="variant targets")
    if (
        baseline.shape != observed.shape
        or initial.shape[0] != baseline.shape[0]
        or action.shape[:2] != baseline.shape[:2]
        or len(names) != baseline.shape[0]
        or len(set(names)) != len(names)
    ):
        raise ValueError("DEFORM residual variant arrays do not align")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("DEFORM residual variant ridge must be positive")
    feature_indices = deform_local_feature_indices(arm)
    coordinate_frame = (
        "action-centered-global" if arm == "full-global" else "initial-action-local"
    )
    initial, action, baseline, observed, grouped_names = _collapse_duplicate_queries(
        initial,
        action,
        baseline,
        observed,
        names,
    )
    full_features, frames = build_deform_local_residual_features(
        initial,
        action,
        baseline,
        coordinate_frame=coordinate_frame,
    )
    features = full_features[..., feature_indices]
    residual_global = observed - baseline
    residual_canonical = np.einsum("ntvi,nij->ntvj", residual_global, frames)
    internal: Array = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    residual_canonical = residual_canonical[:, :, internal]
    trajectory_count, horizon, internal_count, feature_count = features.shape
    feature_location = np.zeros((internal_count, feature_count), dtype=np.float64)
    feature_scale = np.ones_like(feature_location)
    coefficients = np.zeros((internal_count, feature_count + 1, 3), dtype=np.float64)
    penalty = np.eye(feature_count + 1, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    for node in range(internal_count):
        raw_x = features[:, :, node]
        location = np.mean(raw_x, axis=(0, 1))
        scale = np.std(raw_x, axis=(0, 1))
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = (raw_x - location) / scale
        design = np.concatenate(
            (
                np.ones((trajectory_count, horizon, 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        flat_design = design.reshape(-1, feature_count + 1)
        response = residual_canonical[:, :, node].reshape(-1, 3)
        solved = solve_spd(
            flat_design.T @ flat_design + penalty,
            flat_design.T @ response,
            compute_covariance=False,
        )
        feature_location[node] = location
        feature_scale[node] = scale
        coefficients[node] = solved.solution
    return {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-variant-v1",
        "arm": arm,
        "coordinate_frame": coordinate_frame,
        "node_count": baseline.shape[2],
        "prediction_horizon": baseline.shape[1],
        "full_feature_count": DEFORM_LOCAL_FEATURE_COUNT,
        "feature_indices": feature_indices,
        "trajectory_clusters": grouped_names,
        "feature_location": feature_location,
        "feature_scale": feature_scale,
        "coefficients": coefficients,
        "ridge": ridge,
    }


def predict_deform_local_residual_variant(
    model: Mapping[str, object],
    initial_states: Array,
    clamped_action: Array,
    baseline_predictions: Array,
    *,
    shrinkage: float,
) -> dict[str, Array]:
    """Predict one fixed mechanism arm without changing clamped nodes."""

    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("DEFORM residual variant shrinkage is invalid")
    baseline = _finite_array(
        baseline_predictions, ndim=4, label="variant query baseline"
    )
    if (
        int(cast(Any, model.get("node_count", -1))) != baseline.shape[2]
        or int(cast(Any, model.get("prediction_horizon", -1))) != baseline.shape[1]
    ):
        raise ValueError("DEFORM residual variant shape differs")
    coordinate_frame = str(model.get("coordinate_frame", ""))
    full_features, frames = build_deform_local_residual_features(
        initial_states,
        clamped_action,
        baseline,
        coordinate_frame=coordinate_frame,
    )
    feature_indices = tuple(
        int(value) for value in cast(Sequence[Any], model.get("feature_indices", ()))
    )
    features = full_features[..., feature_indices]
    location = _finite_array(
        np.asarray(model.get("feature_location")),
        ndim=2,
        label="variant feature location",
    )
    scale = _finite_array(
        np.asarray(model.get("feature_scale")),
        ndim=2,
        label="variant feature scale",
    )
    coefficients = _finite_array(
        np.asarray(model.get("coefficients")),
        ndim=3,
        label="variant coefficients",
    )
    internal_count = baseline.shape[2] - 4
    feature_count = len(feature_indices)
    if (
        location.shape != (internal_count, feature_count)
        or scale.shape != location.shape
        or coefficients.shape != (internal_count, feature_count + 1, 3)
    ):
        raise ValueError("DEFORM residual variant model arrays do not align")
    means = []
    for node in range(internal_count):
        standardized = (features[:, :, node] - location[node]) / scale[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        means.append(np.einsum("ntd,dc->ntc", design, coefficients[node]))
    correction_canonical = np.stack(means, axis=2)
    correction_global = np.einsum("ntvj,nij->ntvi", correction_canonical, frames)
    internal: Array = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    candidate = baseline.copy()
    candidate[:, :, internal] += shrinkage * correction_global
    return {
        "predictions": candidate,
        "correction_l2_m": np.sqrt(
            np.mean(np.square(shrinkage * correction_global), axis=(1, 2, 3))
        ),
    }


def augment_deform_local_residual_full_covariance(
    model: Mapping[str, object],
    initial_states: Array,
    clamped_action: Array,
    baseline_predictions: Array,
    targets: Array,
    names: Sequence[str],
) -> dict[str, object]:
    """Add trajectory-clustered cross-coordinate covariance to a fitted model."""

    initial = _finite_array(initial_states, ndim=4, label="covariance initial states")
    action = _finite_array(clamped_action, ndim=4, label="covariance action")
    baseline = _finite_array(baseline_predictions, ndim=4, label="covariance baseline")
    observed = _finite_array(targets, ndim=4, label="covariance targets")
    if (
        baseline.shape != observed.shape
        or initial.shape[0] != baseline.shape[0]
        or action.shape[:2] != baseline.shape[:2]
        or len(names) != baseline.shape[0]
    ):
        raise ValueError("DEFORM full-covariance arrays do not align")
    initial, action, baseline, observed, _ = _collapse_duplicate_queries(
        initial,
        action,
        baseline,
        observed,
        names,
    )
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    location = _finite_array(
        np.asarray(model.get("feature_location")),
        ndim=2,
        label="covariance feature location",
    )
    scale = _finite_array(
        np.asarray(model.get("feature_scale")),
        ndim=2,
        label="covariance feature scale",
    )
    coefficients = _finite_array(
        np.asarray(model.get("coefficients")),
        ndim=3,
        label="covariance coefficients",
    )
    residual_global = observed - baseline
    residual_canonical = np.einsum("ntvi,nij->ntvj", residual_global, frames)
    internal: Array = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    residual_canonical = residual_canonical[:, :, internal]
    trajectory_count, horizon, internal_count, feature_count = features.shape
    if (
        feature_count != DEFORM_LOCAL_FEATURE_COUNT
        or location.shape != (internal_count, feature_count)
        or scale.shape != location.shape
        or coefficients.shape != (internal_count, feature_count + 1, 3)
    ):
        raise ValueError("DEFORM full-covariance model arrays do not align")
    ridge = float(cast(Any, model.get("ridge", math.nan)))
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("DEFORM full-covariance ridge is invalid")
    dimension = feature_count + 1
    coefficient_covariance = np.zeros(
        (internal_count, 3, 3, dimension, dimension), dtype=np.float64
    )
    residual_covariance = np.zeros((internal_count, 3, 3), dtype=np.float64)
    penalty = np.eye(dimension, dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    cluster_correction = trajectory_count / max(1, trajectory_count - 1)
    for node in range(internal_count):
        standardized = (features[:, :, node] - location[node]) / scale[node]
        design = np.concatenate(
            (
                np.ones((trajectory_count, horizon, 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        flat_design = design.reshape(-1, dimension)
        response = residual_canonical[:, :, node].reshape(-1, 3)
        solved = solve_spd(
            flat_design.T @ flat_design + penalty,
            flat_design.T @ response,
            compute_covariance=True,
        )
        bread = solved.covariance
        if bread is None:
            raise RuntimeError("DEFORM full-covariance solve omitted covariance")
        fit_residual = residual_canonical[:, :, node] - np.einsum(
            "ntd,dc->ntc", design, coefficients[node]
        )
        scores = np.einsum("ntd,ntc->ndc", design, fit_residual)
        for left in range(3):
            for right in range(3):
                meat = scores[:, :, left].T @ scores[:, :, right] * cluster_correction
                coefficient_covariance[node, left, right] = bread @ meat @ bread
        flat_residual = fit_residual.reshape(-1, 3)
        residual_covariance[node] = (
            flat_residual.T @ flat_residual / flat_residual.shape[0]
        )
    coefficient_covariance = 0.5 * (
        coefficient_covariance + coefficient_covariance.transpose(0, 2, 1, 4, 3)
    )
    residual_covariance = 0.5 * (
        residual_covariance + residual_covariance.transpose(0, 2, 1)
    )
    result = dict(model)
    result["full_covariance_contract"] = (
        "trajectory-clustered-full-coordinate-covariance-v1"
    )
    result["coefficient_covariance_full"] = coefficient_covariance
    result["residual_covariance_full"] = residual_covariance
    return result


def _project_psd(values: Array) -> Array:
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, 0.0)
    return cast(
        Array,
        np.asarray(
            np.einsum("...ik,...k,...jk->...ij", eigenvectors, clipped, eigenvectors),
            dtype=np.float64,
        ),
    )


def predict_deform_local_residual_full_covariance(
    model: Mapping[str, object],
    initial_states: Array,
    clamped_action: Array,
    baseline_predictions: Array,
    *,
    shrinkage: float,
    variance_mode: str = "conservative",
) -> dict[str, Array]:
    """Predict the unchanged point mean and a full coordinate covariance."""

    if not math.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
        raise ValueError("DEFORM full-covariance shrinkage is invalid")
    if variance_mode not in ("conservative", "shrinkage-propagated"):
        raise ValueError("DEFORM full-covariance variance mode is invalid")
    baseline = _finite_array(
        baseline_predictions, ndim=4, label="full-covariance query baseline"
    )
    point = predict_deform_local_residual(
        dict(model),
        initial_states,
        clamped_action,
        baseline,
        shrinkage=shrinkage,
    )
    features, frames = build_deform_local_residual_features(
        initial_states, clamped_action, baseline
    )
    location = _finite_array(
        np.asarray(model.get("feature_location")),
        ndim=2,
        label="full-covariance feature location",
    )
    scale = _finite_array(
        np.asarray(model.get("feature_scale")),
        ndim=2,
        label="full-covariance feature scale",
    )
    coefficient_covariance = _finite_array(
        np.asarray(model.get("coefficient_covariance_full")),
        ndim=5,
        label="full coefficient covariance",
    )
    residual_covariance = _finite_array(
        np.asarray(model.get("residual_covariance_full")),
        ndim=3,
        label="full residual covariance",
    )
    internal_count = baseline.shape[2] - 4
    feature_count = features.shape[3]
    dimension = feature_count + 1
    if coefficient_covariance.shape != (
        internal_count,
        3,
        3,
        dimension,
        dimension,
    ) or residual_covariance.shape != (internal_count, 3, 3):
        raise ValueError("DEFORM full-covariance model arrays do not align")
    canonical_covariances = []
    for node in range(internal_count):
        standardized = (features[:, :, node] - location[node]) / scale[node]
        design = np.concatenate(
            (
                np.ones((*standardized.shape[:2], 1), dtype=np.float64),
                standardized,
            ),
            axis=2,
        )
        epistemic = np.einsum(
            "ntd,abde,nte->ntab",
            design,
            coefficient_covariance[node],
            design,
        )
        canonical_covariances.append(epistemic + residual_covariance[node])
    canonical = np.stack(canonical_covariances, axis=2)
    global_covariance = np.einsum("nia,ntvab,njb->ntvij", frames, canonical, frames)
    correction = (np.asarray(point["predictions"]) - baseline) / shrinkage
    internal: Array = np.arange(2, baseline.shape[2] - 2, dtype=np.int64)
    correction_internal = correction[:, :, internal]
    unresolved = (1.0 - shrinkage) * correction_internal
    unresolved_covariance = np.einsum("...i,...j->...ij", unresolved, unresolved)
    if variance_mode == "shrinkage-propagated":
        global_covariance = np.square(shrinkage) * global_covariance
    global_covariance = _project_psd(global_covariance + unresolved_covariance)
    floor = float(cast(Any, model.get("variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("DEFORM full-covariance variance floor is invalid")
    global_covariance += floor * np.eye(3, dtype=np.float64)
    coordinate_covariance = np.zeros((*baseline.shape, 3), dtype=np.float64)
    coordinate_covariance[:, :, internal] = global_covariance
    return {
        **point,
        "coordinate_covariance_m2": coordinate_covariance,
        "coordinate_variance_m2": np.diagonal(
            coordinate_covariance, axis1=-2, axis2=-1
        ).copy(),
    }


def calibrate_deform_full_covariance(
    predictions: Array,
    targets: Array,
    coordinate_covariance_m2: Array,
    *,
    nominal_coverage: float = 0.90,
) -> dict[str, object]:
    """Calibrate with the maximum of nine trajectory-level p90 scores."""

    predicted = _finite_array(predictions, ndim=4, label="calibration predictions")
    observed = _finite_array(targets, ndim=4, label="calibration targets")
    covariance = _finite_array(
        coordinate_covariance_m2, ndim=5, label="calibration covariance"
    )
    if (
        predicted.shape != observed.shape
        or covariance.shape != (*predicted.shape, 3)
        or predicted.shape[0] != 9
        or nominal_coverage != 0.90
    ):
        raise ValueError("DEFORM full-covariance calibration contract differs")
    internal = slice(2, -2)
    variance = np.diagonal(covariance[:, :, internal], axis1=-2, axis2=-1)
    if np.any(variance <= 0.0):
        raise ValueError("DEFORM full-covariance calibration variance is nonpositive")
    standardized = np.abs(
        predicted[:, :, internal] - observed[:, :, internal]
    ) / np.sqrt(variance)
    scores = np.quantile(
        standardized.reshape(9, -1),
        nominal_coverage,
        axis=1,
        method="higher",
    )
    radius = float(np.max(scores))
    gaussian_radius = NormalDist().inv_cdf(0.5 + nominal_coverage / 2.0)
    variance_scale = max(1.0, np.square(radius / gaussian_radius))
    return {
        "schema_version": 1,
        "contract": "deform-dlo-full-covariance-calibration-v1",
        "trajectory_scores": scores,
        "rank": 9,
        "order_statistic": "maximum-of-nine",
        "nominal_coordinate_coverage": nominal_coverage,
        "standardized_radius": radius,
        "gaussian_radius": gaussian_radius,
        "variance_scale": float(variance_scale),
        "confidence_increase_forbidden": True,
    }


def scale_deform_coordinate_covariance(
    coordinate_covariance_m2: Array,
    variance_scale: float,
) -> Array:
    """Apply one source-calibrated scalar without changing the point mean."""

    covariance = _finite_array(
        coordinate_covariance_m2, ndim=5, label="coordinate covariance"
    )
    if not math.isfinite(variance_scale) or variance_scale < 1.0:
        raise ValueError("DEFORM covariance scale must not increase confidence")
    return cast(Array, np.asarray(covariance * variance_scale, dtype=np.float64))


def evaluate_deform_predictive_distribution(
    predictions: Array,
    targets: Array,
    coordinate_covariance_m2: Array,
    *,
    sample_count: int = 32,
    sample_seed: int = 0,
) -> dict[str, object]:
    """Evaluate point accuracy, calibration, and a deterministic energy score."""

    predicted = _finite_array(predictions, ndim=4, label="distribution predictions")
    observed = _finite_array(targets, ndim=4, label="distribution targets")
    covariance = _finite_array(
        coordinate_covariance_m2, ndim=5, label="distribution covariance"
    )
    if predicted.shape != observed.shape or covariance.shape != (*predicted.shape, 3):
        raise ValueError("DEFORM predictive distribution arrays do not align")
    if sample_count <= 1 or sample_seed < 0:
        raise ValueError("DEFORM energy-score sampling contract is invalid")
    internal = slice(2, -2)
    mean = predicted[:, :, internal].reshape(-1, 3)
    truth = observed[:, :, internal].reshape(-1, 3)
    matrices = covariance[:, :, internal].reshape(-1, 3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (matrices + matrices.transpose(0, 2, 1))
    )
    if np.any(eigenvalues <= 0.0):
        raise ValueError("DEFORM predictive covariance must be positive definite")
    inverse = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        1.0 / eigenvalues,
        eigenvectors,
    )
    error = truth - mean
    mahalanobis = np.einsum("pi,pij,pj->p", error, inverse, error)
    log_determinant = np.sum(np.log(eigenvalues), axis=1)
    gaussian_nll = 0.5 * (3.0 * math.log(2.0 * math.pi) + log_determinant + mahalanobis)
    variance = np.diagonal(matrices, axis1=-2, axis2=-1)
    coordinate_nees = np.square(error) / variance
    radius = NormalDist().inv_cdf(0.95)
    covered = np.abs(error) <= radius * np.sqrt(variance)
    root = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        np.sqrt(eigenvalues),
        eigenvectors,
    )
    rng = np.random.default_rng(sample_seed)
    standard_a = rng.standard_normal((sample_count, mean.shape[0], 3))
    standard_b = rng.standard_normal((sample_count, mean.shape[0], 3))
    sample_a = mean[None] + np.einsum("pij,spj->spi", root, standard_a)
    sample_b = mean[None] + np.einsum("pij,spj->spi", root, standard_b)
    energy_score = np.mean(
        np.linalg.norm(sample_a - truth[None], axis=2)
    ) - 0.5 * np.mean(np.linalg.norm(sample_a - sample_b, axis=2))
    return {
        "schema_version": 1,
        "contract": "deform-dlo-predictive-distribution-metrics-v1",
        "mean_coordinate_l1_m": float(np.mean(np.abs(error))),
        "gaussian_nll": float(np.mean(gaussian_nll)),
        "coordinate_nees": float(np.mean(coordinate_nees)),
        "multivariate_nees": float(np.mean(mahalanobis) / 3.0),
        "coordinate_coverage_90": float(np.mean(covered)),
        "interval_width_m": float(np.mean(2.0 * radius * np.sqrt(variance))),
        "energy_score": float(energy_score),
        "energy_score_sample_count": sample_count,
        "energy_score_seed": sample_seed,
    }


def evaluate_deform_dlo3_source_gate(
    candidate_predictions: Array,
    baseline_predictions: Array,
    targets: Array,
    names: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the immutable eight-trajectory DLO3 advancement gate."""

    candidate = _finite_array(candidate_predictions, ndim=4, label="source candidate")
    baseline = _finite_array(baseline_predictions, ndim=4, label="source baseline")
    observed = _finite_array(targets, ndim=4, label="source targets")
    normalized_names = tuple(str(name) for name in names)
    if (
        candidate.shape != baseline.shape
        or candidate.shape != observed.shape
        or candidate.shape[0] != 8
        or len(normalized_names) != 8
        or len(set(normalized_names)) != 8
    ):
        raise ValueError("DLO3 source gate requires eight aligned trajectories")
    candidate_errors = np.mean(np.abs(candidate - observed), axis=(1, 2, 3))
    baseline_errors = np.mean(np.abs(baseline - observed), axis=(1, 2, 3))
    if np.any(baseline_errors <= 0.0):
        raise ValueError("DLO3 source baseline error must be positive")
    ratios = candidate_errors / baseline_errors
    candidate_mean = float(np.mean(candidate_errors))
    baseline_mean = float(np.mean(baseline_errors))
    relative_improvement = 1.0 - candidate_mean / baseline_mean
    wins = int(np.count_nonzero(candidate_errors < baseline_errors))
    maximum_ratio = float(np.max(ratios))
    gate = _mapping(protocol.get("source_gate"), label="source gate")
    improvement_passed = relative_improvement >= float(
        cast(Any, gate["minimum_relative_improvement"])
    )
    wins_passed = wins >= int(cast(Any, gate["minimum_case_wins"]))
    ratio_passed = maximum_ratio <= float(cast(Any, gate["maximum_case_ratio"]))
    reference_passed = candidate_mean < float(cast(Any, gate["maximum_candidate_l1_m"]))
    records = [
        {
            "name": name,
            "candidate_l1_m": float(candidate_errors[index]),
            "baseline_l1_m": float(baseline_errors[index]),
            "candidate_to_baseline_ratio": float(ratios[index]),
            "candidate_wins": bool(candidate_errors[index] < baseline_errors[index]),
        }
        for index, name in enumerate(normalized_names)
    ]
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-gate-v1",
        "case_count": 8,
        "candidate_mean_l1_m": candidate_mean,
        "baseline_mean_l1_m": baseline_mean,
        "relative_improvement": relative_improvement,
        "wins": wins,
        "maximum_case_ratio": maximum_ratio,
        "improvement_passed": improvement_passed,
        "wins_passed": wins_passed,
        "maximum_case_ratio_passed": ratio_passed,
        "published_reference_passed": reference_passed,
        "passed": bool(
            improvement_passed and wins_passed and ratio_passed and reference_passed
        ),
        "cases": records,
    }
