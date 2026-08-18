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
DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS = (
    "current-diagonal-conservative-v1",
    "shrinkage-propagated-diagonal",
    "coefficient-only",
    "residual-only",
    "pooled-isotropic",
    "trajectory-clustered-full-coordinate-covariance-v1",
    "calibrated-full-coordinate-covariance-v1",
)
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
        or _strings(
            bayesian.get("ablation_distributions"),
            label="Bayesian ablation distributions",
        )
        != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        or bayesian.get("distribution_selection_from_target") is not False
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


def deform_bayesian_covariance_archive_key(label: str) -> str:
    """Return the deterministic NPZ key for one frozen covariance arm."""

    if label not in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS:
        raise ValueError(f"unsupported DEFORM Bayesian covariance arm: {label}")
    return f"bayesian_covariance_m2__{label.replace('-', '_')}"


def _diagonal_coordinate_covariance(coordinate_variance_m2: Array) -> Array:
    variance = _finite_array(
        coordinate_variance_m2,
        ndim=4,
        label="diagonal coordinate variance",
    )
    covariance = np.zeros((*variance.shape, 3), dtype=np.float64)
    coordinate = np.arange(3)
    covariance[..., coordinate, coordinate] = variance
    return covariance


def build_deform_bayesian_covariance_ablation_v1(
    model: Mapping[str, object],
    initial_states: Array,
    clamped_action: Array,
    baseline_predictions: Array,
    *,
    shrinkage: float,
    variance_scale: float,
) -> dict[str, dict[str, Array]]:
    """Build every frozen covariance arm without changing the point prediction.

    The pooled arm uses only the query covariances, never outcomes. Coefficient-only
    and residual-only retain the unresolved-shrinkage term and variance floor so
    that each arm remains a valid predictive distribution.
    """

    if not math.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
        raise ValueError("DEFORM Bayesian ablation shrinkage is invalid")
    baseline = _finite_array(
        baseline_predictions,
        ndim=4,
        label="Bayesian ablation baseline",
    )
    floor = float(cast(Any, model.get("variance_floor_m2", math.nan)))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("DEFORM Bayesian ablation variance floor is invalid")

    diagonal = predict_deform_local_residual(
        dict(model),
        initial_states,
        clamped_action,
        baseline,
        shrinkage=shrinkage,
    )
    unshrunk_diagonal = predict_deform_local_residual(
        dict(model),
        initial_states,
        clamped_action,
        baseline,
        shrinkage=1.0,
    )
    full = predict_deform_local_residual_full_covariance(
        model,
        initial_states,
        clamped_action,
        baseline,
        shrinkage=shrinkage,
    )
    point_mean = np.asarray(full["predictions"])
    if not np.array_equal(point_mean, np.asarray(diagonal["predictions"])):
        raise RuntimeError("DEFORM Bayesian covariance arms changed the point mean")

    internal = slice(2, -2)
    shrinkage_variance = np.zeros_like(baseline)
    modeled_variance = np.maximum(
        np.asarray(unshrunk_diagonal["coordinate_variance_m2"])[:, :, internal] - floor,
        0.0,
    )
    correction = (point_mean[:, :, internal] - baseline[:, :, internal]) / shrinkage
    unresolved_variance = np.square((1.0 - shrinkage) * correction)
    shrinkage_variance[:, :, internal] = (
        np.square(shrinkage) * modeled_variance + unresolved_variance + floor
    )

    coefficient_model = dict(model)
    coefficient_model["residual_covariance_full"] = np.zeros_like(
        np.asarray(model.get("residual_covariance_full")), dtype=np.float64
    )
    coefficient_only = predict_deform_local_residual_full_covariance(
        coefficient_model,
        initial_states,
        clamped_action,
        baseline,
        shrinkage=shrinkage,
    )
    residual_model = dict(model)
    residual_model["coefficient_covariance_full"] = np.zeros_like(
        np.asarray(model.get("coefficient_covariance_full")), dtype=np.float64
    )
    residual_only = predict_deform_local_residual_full_covariance(
        residual_model,
        initial_states,
        clamped_action,
        baseline,
        shrinkage=shrinkage,
    )

    full_covariance = np.asarray(full["coordinate_covariance_m2"])
    internal_full_covariance = full_covariance[:, :, internal]
    pooled_variance = float(
        np.mean(np.trace(internal_full_covariance, axis1=-2, axis2=-1) / 3.0)
    )
    if not math.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise RuntimeError("DEFORM pooled covariance is nonpositive")
    pooled_covariance = np.zeros_like(full_covariance)
    pooled_covariance[:, :, internal] = pooled_variance * np.eye(3, dtype=np.float64)

    covariances = {
        "current-diagonal-conservative-v1": _diagonal_coordinate_covariance(
            np.asarray(diagonal["coordinate_variance_m2"])
        ),
        "shrinkage-propagated-diagonal": _diagonal_coordinate_covariance(
            shrinkage_variance
        ),
        "coefficient-only": np.asarray(coefficient_only["coordinate_covariance_m2"]),
        "residual-only": np.asarray(residual_only["coordinate_covariance_m2"]),
        "pooled-isotropic": pooled_covariance,
        "trajectory-clustered-full-coordinate-covariance-v1": full_covariance,
        "calibrated-full-coordinate-covariance-v1": (
            scale_deform_coordinate_covariance(full_covariance, variance_scale)
        ),
    }
    if tuple(covariances) != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS:
        raise RuntimeError("DEFORM Bayesian covariance arm order differs")

    predictions = (
        np.asarray(diagonal["predictions"]),
        np.asarray(coefficient_only["predictions"]),
        np.asarray(residual_only["predictions"]),
    )
    if any(not np.array_equal(point_mean, values) for values in predictions):
        raise RuntimeError("DEFORM Bayesian ablation point means differ")
    return {
        label: {
            "predictions": point_mean,
            "coordinate_covariance_m2": covariance,
            "coordinate_variance_m2": np.diagonal(
                covariance, axis1=-2, axis2=-1
            ).copy(),
        }
        for label, covariance in covariances.items()
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


def validate_deform_bayesian_audit_v1(
    result: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Require the complete frozen seven-arm Bayesian audit."""

    if context not in ("source", "evaluator"):
        raise ValueError("unsupported DEFORM Bayesian audit context")
    audit = _mapping(result.get("bayesian_audit"), label="Bayesian audit")
    distributions = _mapping(audit.get("distributions"), label="Bayesian distributions")
    expected = DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    if (
        len(distributions) != len(expected)
        or set(str(name) for name in distributions) != set(expected)
        or audit.get("point_mean_unchanged") is not True
        or audit.get("distribution_selection") != "none"
    ):
        raise ValueError("DEFORM Bayesian audit is incomplete")

    normalized: dict[str, Mapping[str, object]] = {}
    point_errors: list[float] = []
    for name in expected:
        metrics = _mapping(
            distributions.get(name), label=f"Bayesian distribution {name}"
        )
        if (
            metrics.get("contract") != "deform-dlo-predictive-distribution-metrics-v1"
            or int(cast(Any, metrics.get("energy_score_sample_count", -1))) != 32
            or int(cast(Any, metrics.get("energy_score_seed", -1))) != 0
        ):
            raise ValueError("DEFORM Bayesian distribution contract differs")
        values = {
            key: float(cast(Any, metrics.get(key, math.nan)))
            for key in (
                "mean_coordinate_l1_m",
                "gaussian_nll",
                "coordinate_nees",
                "multivariate_nees",
                "coordinate_coverage_90",
                "interval_width_m",
                "energy_score",
            )
        }
        if (
            any(not math.isfinite(value) for value in values.values())
            or values["mean_coordinate_l1_m"] < 0.0
            or values["coordinate_nees"] < 0.0
            or values["multivariate_nees"] < 0.0
            or not 0.0 <= values["coordinate_coverage_90"] <= 1.0
            or values["interval_width_m"] <= 0.0
        ):
            raise ValueError("DEFORM Bayesian distribution metrics are invalid")
        point_errors.append(values["mean_coordinate_l1_m"])
        normalized[name] = metrics
    if any(value != point_errors[0] for value in point_errors[1:]):
        raise ValueError("DEFORM Bayesian distribution point means differ")

    if context == "source":
        calibration = _mapping(audit.get("calibration"), label="Bayesian calibration")
        scores = tuple(
            float(value)
            for value in cast(Sequence[Any], calibration.get("trajectory_scores", ()))
        )
        radius = float(cast(Any, calibration.get("standardized_radius", math.nan)))
        variance_scale = float(cast(Any, calibration.get("variance_scale", math.nan)))
        if (
            calibration.get("contract") != "deform-dlo-full-covariance-calibration-v1"
            or len(scores) != 9
            or any(not math.isfinite(value) or value < 0.0 for value in scores)
            or int(cast(Any, calibration.get("rank", -1))) != 9
            or calibration.get("order_statistic") != "maximum-of-nine"
            or float(
                cast(Any, calibration.get("nominal_coordinate_coverage", math.nan))
            )
            != 0.9
            or not math.isfinite(radius)
            or radius != max(scores)
            or not math.isfinite(variance_scale)
            or variance_scale < 1.0
            or calibration.get("confidence_increase_forbidden") is not True
            or calibration.get("source_test_opened") is not False
            or calibration.get("official_eval_read") is not False
            or audit.get("source_test_outcomes_used_for_covariance_construction")
            is not False
            or audit.get("uncalibrated")
            != normalized["trajectory-clustered-full-coordinate-covariance-v1"]
            or audit.get("calibrated")
            != normalized["calibrated-full-coordinate-covariance-v1"]
        ):
            raise ValueError("DEFORM Bayesian source calibration differs")
    elif (
        audit.get("primary_distribution") != "calibrated-full-coordinate-covariance-v1"
        or audit.get("target_outcomes_used_for_distribution_construction") is not False
        or audit.get("target_outcomes_used_for_distribution_selection") is not False
    ):
        raise ValueError("DEFORM Bayesian evaluator custody differs")

    return {
        "schema_version": 1,
        "contract": "deform-dlo-bayesian-audit-verification-v1",
        "context": context,
        "distribution_count": len(expected),
        "distributions": list(expected),
        "point_mean_unchanged": True,
        "distribution_selection": "none",
    }


def _verified_deform_artifact_path(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    digest = str(identity.get("sha256", ""))
    size = int(cast(Any, identity.get("size_bytes", -1)))
    if (
        len(digest) != 64
        or size <= 0
        or not path.is_file()
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ValueError(f"{label} identity changed")
    return path


def _verify_deform_bayesian_prediction_archive_v1(
    path: Path,
    *,
    expected_case_count: int,
    variance_scale: float,
    source_compatibility_aliases: bool,
) -> dict[str, object]:
    expected_keys = {
        label: deform_bayesian_covariance_archive_key(label)
        for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    }
    with np.load(path, allow_pickle=False) as archive:
        required = {"names", "candidate", *expected_keys.values()}
        if source_compatibility_aliases:
            required.update(
                {"coordinate_covariance_m2", "calibrated_coordinate_covariance_m2"}
            )
        elif "calibrated_coordinate_covariance_m2" not in archive.files:
            raise ValueError("DEFORM Bayesian prediction archive is incomplete")
        if not required.issubset(archive.files):
            raise ValueError("DEFORM Bayesian prediction archive is incomplete")
        names = np.asarray(archive["names"])
        candidate = _finite_array(
            np.asarray(archive["candidate"]), ndim=4, label="Bayesian point mean"
        )
        if (
            names.shape != (expected_case_count,)
            or candidate.shape[0] != expected_case_count
            or candidate.shape[2] < 5
            or candidate.shape[3] != 3
        ):
            raise ValueError("DEFORM Bayesian prediction archive shape differs")
        covariance: dict[str, Array] = {}
        for label, key in expected_keys.items():
            values = _finite_array(
                np.asarray(archive[key]), ndim=5, label=f"Bayesian covariance {label}"
            )
            if values.shape != (*candidate.shape, 3):
                raise ValueError("DEFORM Bayesian covariance shape differs")
            if (
                np.count_nonzero(values[:, :, :2]) != 0
                or np.count_nonzero(values[:, :, -2:]) != 0
                or not np.allclose(
                    values[:, :, 2:-2],
                    values[:, :, 2:-2].swapaxes(-1, -2),
                    rtol=0.0,
                    atol=1e-12,
                )
                or np.min(np.linalg.eigvalsh(values[:, :, 2:-2])) <= 0.0
            ):
                raise ValueError("DEFORM Bayesian covariance is invalid")
            covariance[label] = values
        raw = covariance["trajectory-clustered-full-coordinate-covariance-v1"]
        calibrated = covariance["calibrated-full-coordinate-covariance-v1"]
        if (
            not math.isfinite(variance_scale)
            or variance_scale < 1.0
            or not np.allclose(calibrated, raw * variance_scale, rtol=1e-12, atol=0.0)
            or not np.array_equal(
                np.asarray(archive["calibrated_coordinate_covariance_m2"]),
                calibrated,
            )
        ):
            raise ValueError("DEFORM Bayesian calibrated covariance differs")
        if source_compatibility_aliases and not np.array_equal(
            np.asarray(archive["coordinate_covariance_m2"]), raw
        ):
            raise ValueError("DEFORM Bayesian covariance compatibility alias differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo-bayesian-prediction-archive-verification-v1",
        "case_count": expected_case_count,
        "distribution_count": len(expected_keys),
        "archive_keys": expected_keys,
        "point_mean_count": 1,
    }


def verify_deform_dlo3_seed_bayesian_artifacts_v1(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Rehash and inspect one sealed source-seed Bayesian artifact bundle."""

    if (
        result.get("contract") != "deform-dlo3-robustness-seed-result-v1"
        or result.get("source_test_opened") is not True
        or result.get("primary_eval_enumerated") is not False
        or result.get("primary_eval_read") is not False
        or result.get("target_authorized") is not False
        or result.get("retry_authorized") is not False
        or result.get("held_v8_access") is not False
    ):
        raise ValueError("DEFORM seed Bayesian artifact custody differs")
    audit = validate_deform_bayesian_audit_v1(result, context="source")
    seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="seed prediction seal"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("seed prediction seal must be a JSON object")
    expected_keys = {
        label: deform_bayesian_covariance_archive_key(label)
        for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    }
    seed = int(cast(Any, result.get("seed", -1)))
    method_identity = _mapping(result.get("method_seal"), label="seed method seal")
    seal_method_identity = _mapping(
        seal.get("method_seal"), label="prediction-seal method"
    )
    if (
        seal.get("contract") != "deform-dlo3-robustness-source-prediction-seal-v1"
        or int(cast(Any, seal.get("seed", -1))) != seed
        or _strings(
            seal.get("bayesian_ablation_distributions"),
            label="sealed Bayesian distributions",
        )
        != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        or dict(
            _mapping(
                seal.get("bayesian_covariance_archive_keys"),
                label="sealed Bayesian archive keys",
            )
        )
        != expected_keys
        or seal.get("bayesian_point_means_identical") is not True
        or seal.get("source_outcomes_scored") is not False
        or seal.get("official_eval_read") is not False
        or seal_method_identity.get("sha256") != method_identity.get("sha256")
    ):
        raise ValueError("DEFORM seed Bayesian prediction seal differs")
    _verified_deform_artifact_path(method_identity, label="seed method seal")
    predictions_path = _verified_deform_artifact_path(
        seal.get("predictions"), label="seed Bayesian predictions"
    )
    calibration = _mapping(
        _mapping(result.get("bayesian_audit"), label="Bayesian audit").get(
            "calibration"
        ),
        label="Bayesian calibration",
    )
    archive = _verify_deform_bayesian_prediction_archive_v1(
        predictions_path,
        expected_case_count=8,
        variance_scale=float(cast(Any, calibration.get("variance_scale", math.nan))),
        source_compatibility_aliases=True,
    )
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-seed-bayesian-artifact-verification-v1",
        "seed": seed,
        "prediction_seal_sha256": sha256_file(seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "audit": audit,
        "archive": archive,
        "verified": True,
    }


def verify_deform_dlo3_evaluator_bayesian_artifacts_v1(
    result: Mapping[str, object],
    *,
    expected_mode: str,
) -> dict[str, object]:
    """Rehash and inspect a dry-run or official evaluator Bayesian bundle."""

    if expected_mode not in ("dry-run", "official"):
        raise ValueError("unsupported DEFORM evaluator mode")
    if expected_mode == "dry-run":
        custody_valid = (
            result.get("contract") == "deform-dlo3-robustness-evaluator-dry-run-v1"
            and result.get("primary_eval_read") is False
            and result.get("target_authorized") is False
            and result.get("retry_authorized") is False
            and result.get("held_v8_access") is False
        )
    else:
        custody_valid = (
            result.get("contract") == "deform-dlo3-robustness-official-result-v1"
            and result.get("official_eval_read") is True
            and result.get("retry_authorized") is False
            and result.get("case_replacement") is False
            and result.get("held_v8_access") is False
        )
    if not custody_valid:
        raise ValueError("DEFORM evaluator Bayesian artifact custody differs")
    audit = validate_deform_bayesian_audit_v1(result, context="evaluator")
    seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="evaluator prediction seal"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("evaluator prediction seal must be a JSON object")
    expected_keys = {
        label: deform_bayesian_covariance_archive_key(label)
        for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    }
    if (
        seal.get("contract") != "deform-dlo3-robustness-evaluator-prediction-seal-v1"
        or seal.get("mode") != expected_mode
        or _strings(
            seal.get("bayesian_ablation_distributions"),
            label="evaluator Bayesian distributions",
        )
        != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        or dict(
            _mapping(
                seal.get("bayesian_covariance_archive_keys"),
                label="evaluator Bayesian archive keys",
            )
        )
        != expected_keys
        or seal.get("bayesian_point_means_identical") is not True
        or seal.get("outcomes_scored") is not False
        or seal.get("target_retries") is not False
    ):
        raise ValueError("DEFORM evaluator Bayesian prediction seal differs")
    predictions_path = _verified_deform_artifact_path(
        seal.get("predictions"), label="evaluator Bayesian predictions"
    )
    distribution_count = 8 if expected_mode == "dry-run" else 14
    # The all-train method carries the frozen seed-42 calibration scale in every arm.
    calibrated = np.load(predictions_path, allow_pickle=False)
    try:
        raw = np.asarray(
            calibrated[
                deform_bayesian_covariance_archive_key(
                    "trajectory-clustered-full-coordinate-covariance-v1"
                )
            ]
        )
        scaled = np.asarray(
            calibrated[
                deform_bayesian_covariance_archive_key(
                    "calibrated-full-coordinate-covariance-v1"
                )
            ]
        )
        positive = raw[:, :, 2:-2] > 0.0
        ratios = np.divide(
            scaled[:, :, 2:-2],
            raw[:, :, 2:-2],
            out=np.ones_like(scaled[:, :, 2:-2]),
            where=positive,
        )
        finite_ratios = ratios[positive]
        if finite_ratios.size == 0 or not np.allclose(
            finite_ratios, finite_ratios[0], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("DEFORM evaluator covariance scale differs")
        variance_scale = float(finite_ratios[0])
    finally:
        calibrated.close()
    archive = _verify_deform_bayesian_prediction_archive_v1(
        predictions_path,
        expected_case_count=distribution_count,
        variance_scale=variance_scale,
        source_compatibility_aliases=False,
    )
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-evaluator-bayesian-artifact-verification-v1",
        "mode": expected_mode,
        "prediction_seal_sha256": sha256_file(seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "audit": audit,
        "archive": archive,
        "verified": True,
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


def evaluate_deform_dlo3_stability_gate(
    seed_results: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate the three frozen source seeds without selecting among them."""

    training = _mapping(protocol.get("physical_training"), label="physical training")
    stability = _mapping(protocol.get("training_stability"), label="stability")
    expected_seeds = _integers(training.get("audit_seeds"), label="audit seeds")
    if len(seed_results) != len(expected_seeds):
        raise ValueError("DLO3 stability gate requires every frozen seed")

    normalized: dict[int, Mapping[str, object]] = {}
    bayesian_verifications: dict[int, dict[str, object]] = {}
    protocol_digests: set[str] = set()
    manifest_digests: set[str] = set()
    for raw in seed_results:
        if raw.get("contract") != "deform-dlo3-robustness-seed-result-v1":
            raise ValueError("DLO3 stability input contract differs")
        seed = int(cast(Any, raw.get("seed", -1)))
        if seed in normalized or seed not in expected_seeds:
            raise ValueError("DLO3 stability input seed differs")
        if (
            raw.get("source_test_opened") is not True
            or raw.get("primary_eval_enumerated") is not False
            or raw.get("primary_eval_read") is not False
            or raw.get("target_authorized") is not False
            or raw.get("retry_authorized") is not False
            or raw.get("prob4d_used") is not False
            or raw.get("held_v8_access") is not False
        ):
            raise ValueError("DLO3 stability input custody differs")
        protocol_identity = _mapping(raw.get("protocol"), label="seed protocol")
        manifest_identity = _mapping(
            raw.get("source_manifest"), label="seed source manifest"
        )
        protocol_digests.add(str(protocol_identity.get("sha256", "")))
        manifest_digests.add(str(manifest_identity.get("sha256", "")))
        bayesian_verifications[seed] = validate_deform_bayesian_audit_v1(
            raw, context="source"
        )
        normalized[seed] = raw
    if set(normalized) != set(expected_seeds):
        raise ValueError("DLO3 stability inputs omit a frozen seed")
    if (
        len(protocol_digests) != 1
        or len(manifest_digests) != 1
        or any(len(value) != 64 for value in protocol_digests | manifest_digests)
    ):
        raise ValueError("DLO3 stability input lineage differs")

    seed_records: list[dict[str, object]] = []
    seed_passes = 0
    seed_mean_ratios: list[float] = []
    case_ratios: list[float] = []
    for seed in expected_seeds:
        result = normalized[seed]
        gate = _mapping(result.get("primary_source_gate"), label="primary source gate")
        cases = cast(Sequence[Mapping[str, object]], gate.get("cases"))
        if (
            gate.get("contract") != "deform-dlo3-robustness-source-gate-v1"
            or int(cast(Any, gate.get("case_count", -1))) != 8
            or not isinstance(cases, Sequence)
            or len(cases) != 8
        ):
            raise ValueError("DLO3 seed source gate differs")
        candidate_mean = float(cast(Any, gate.get("candidate_mean_l1_m", math.nan)))
        baseline_mean = float(cast(Any, gate.get("baseline_mean_l1_m", math.nan)))
        if (
            not math.isfinite(candidate_mean)
            or not math.isfinite(baseline_mean)
            or baseline_mean <= 0.0
        ):
            raise ValueError("DLO3 seed source mean is invalid")
        mean_ratio = candidate_mean / baseline_mean
        maximum_case_ratio = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
        if not math.isfinite(maximum_case_ratio) or maximum_case_ratio < 0.0:
            raise ValueError("DLO3 seed source case ratio is invalid")
        case_values = [
            float(cast(Any, case.get("candidate_to_baseline_ratio", math.nan)))
            for case in cases
        ]
        if any(
            not math.isfinite(value) or value < 0.0 for value in case_values
        ) or not math.isclose(maximum_case_ratio, max(case_values), abs_tol=1e-15):
            raise ValueError("DLO3 seed source case records differ")
        passed = gate.get("passed") is True
        seed_passes += int(passed)
        seed_mean_ratios.append(mean_ratio)
        case_ratios.extend(case_values)
        seed_records.append(
            {
                "seed": seed,
                "source_gate_passed": passed,
                "candidate_to_baseline_mean_ratio": mean_ratio,
                "maximum_case_ratio": maximum_case_ratio,
                "bayesian_audit": bayesian_verifications[seed],
            }
        )

    minimum_passes = int(cast(Any, stability["minimum_seed_source_passes"]))
    maximum_seed_mean_ratio = float(cast(Any, stability["maximum_seed_mean_ratio"]))
    maximum_seed_case_ratio = float(cast(Any, stability["maximum_seed_case_ratio"]))
    passes_requirement = seed_passes >= minimum_passes
    mean_requirement = max(seed_mean_ratios) <= maximum_seed_mean_ratio
    case_requirement = max(case_ratios) <= maximum_seed_case_ratio
    primary_seed = int(cast(Any, training["primary_seed"]))
    primary_passed = (
        _mapping(
            normalized[primary_seed].get("primary_source_gate"),
            label="primary-seed source gate",
        ).get("passed")
        is True
    )
    passed = bool(
        primary_passed and passes_requirement and mean_requirement and case_requirement
    )
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-training-stability-gate-v1",
        "protocol_sha256": next(iter(protocol_digests)),
        "source_manifest_sha256": next(iter(manifest_digests)),
        "primary_seed": primary_seed,
        "primary_seed_passed": primary_passed,
        "seed_source_passes": seed_passes,
        "minimum_seed_source_passes": minimum_passes,
        "maximum_seed_mean_ratio": max(seed_mean_ratios),
        "maximum_seed_case_ratio": max(case_ratios),
        "seed_passes_requirement": passes_requirement,
        "seed_mean_ratio_requirement": mean_requirement,
        "seed_case_ratio_requirement": case_requirement,
        "passed": passed,
        "alltrain_fit_authorized": passed,
        "target_authorized": False,
        "target_authorization_requires": [
            "alltrain-fit-and-seal",
            "physics-solver-sensitivity-audit",
            "backend-portability-audit",
            "independent-end-to-end-dry-run",
            "method-and-environment-attestation",
        ],
        "seed_selection": False,
        "bayesian_audit_complete": True,
        "bayesian_distribution_count": len(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS),
        "bayesian_distribution_selection": "none",
        "seeds": seed_records,
        "primary_eval_read": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }


def evaluate_deform_backend_source_gate(
    candidate_predictions: Array,
    backend_predictions: Array,
    targets: Array,
    names: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the fixed PyElastica residual-transfer source gate."""

    candidate = _finite_array(candidate_predictions, ndim=4, label="backend candidate")
    backend = _finite_array(backend_predictions, ndim=4, label="backend baseline")
    observed = _finite_array(targets, ndim=4, label="backend targets")
    normalized_names = tuple(str(name) for name in names)
    if (
        candidate.shape != backend.shape
        or candidate.shape != observed.shape
        or candidate.shape[0] != 8
        or len(normalized_names) != 8
        or len(set(normalized_names)) != 8
    ):
        raise ValueError("DEFORM backend source gate requires eight trajectories")
    candidate_errors = np.mean(np.abs(candidate - observed), axis=(1, 2, 3))
    backend_errors = np.mean(np.abs(backend - observed), axis=(1, 2, 3))
    if np.any(backend_errors <= 0.0):
        raise ValueError("DEFORM backend source error must be positive")
    ratios = candidate_errors / backend_errors
    candidate_mean = float(np.mean(candidate_errors))
    backend_mean = float(np.mean(backend_errors))
    relative_improvement = 1.0 - candidate_mean / backend_mean
    wins = int(np.count_nonzero(candidate_errors < backend_errors))
    maximum_ratio = float(np.max(ratios))
    contract = _mapping(protocol.get("backend_portability"), label="backend")
    improvement_passed = relative_improvement >= float(
        cast(Any, contract["minimum_relative_improvement_over_backend"])
    )
    wins_passed = wins >= int(cast(Any, contract["minimum_source_test_wins"]))
    ratio_passed = maximum_ratio <= float(
        cast(Any, contract["maximum_source_test_ratio"])
    )
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-gate-v1",
        "case_count": 8,
        "candidate_mean_l1_m": candidate_mean,
        "backend_mean_l1_m": backend_mean,
        "relative_improvement": relative_improvement,
        "wins": wins,
        "maximum_case_ratio": maximum_ratio,
        "improvement_passed": improvement_passed,
        "wins_passed": wins_passed,
        "maximum_case_ratio_passed": ratio_passed,
        "passed": bool(improvement_passed and wins_passed and ratio_passed),
        "cases": [
            {
                "name": name,
                "candidate_l1_m": float(candidate_errors[index]),
                "backend_l1_m": float(backend_errors[index]),
                "candidate_to_backend_ratio": float(ratios[index]),
                "candidate_wins": bool(candidate_errors[index] < backend_errors[index]),
            }
            for index, name in enumerate(normalized_names)
        ],
    }


def evaluate_deform_dlo3_target_gate(
    candidate_predictions: Array,
    baseline_predictions: Array,
    targets: Array,
    names: Sequence[str],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the preregistered one-shot DLO3 target claim gate."""

    candidate = _finite_array(candidate_predictions, ndim=4, label="target candidate")
    baseline = _finite_array(baseline_predictions, ndim=4, label="target baseline")
    observed = _finite_array(targets, ndim=4, label="target outcomes")
    normalized_names = tuple(str(name) for name in names)
    if (
        candidate.shape != baseline.shape
        or candidate.shape != observed.shape
        or candidate.shape[0] != 14
        or len(normalized_names) != 14
        or len(set(normalized_names)) != 14
    ):
        raise ValueError("DLO3 target gate requires fourteen unique trajectories")
    candidate_errors = np.mean(np.abs(candidate - observed), axis=(1, 2, 3))
    baseline_errors = np.mean(np.abs(baseline - observed), axis=(1, 2, 3))
    if np.any(baseline_errors <= 0.0):
        raise ValueError("DLO3 target baseline error must be positive")
    ratios = candidate_errors / baseline_errors
    candidate_mean = float(np.mean(candidate_errors))
    baseline_mean = float(np.mean(baseline_errors))
    relative_improvement = 1.0 - candidate_mean / baseline_mean
    wins = int(np.count_nonzero(candidate_errors < baseline_errors))
    maximum_ratio = float(np.max(ratios))
    target = _mapping(protocol.get("target_evaluation"), label="target evaluation")
    draw = _integers(
        target.get("canonical_reference_draw_indices"), label="canonical draw"
    )
    if len(draw) != 14 or any(index < 0 or index >= 14 for index in draw):
        raise ValueError("DLO3 canonical reference draw differs")
    canonical_mean = float(np.mean(candidate_errors[np.asarray(draw, dtype=np.int64)]))
    published_reference = float(cast(Any, target["published_reference_l1_m"]))
    improvement_passed = relative_improvement >= float(
        cast(Any, target["required_primary_relative_improvement"])
    )
    wins_passed = wins >= int(cast(Any, target["required_primary_case_wins"]))
    ratio_passed = maximum_ratio <= float(
        cast(Any, target["maximum_primary_case_ratio"])
    )
    unique_reference_passed = candidate_mean < published_reference
    canonical_reference_passed = canonical_mean < published_reference
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-target-gate-v1",
        "case_count": 14,
        "candidate_mean_l1_m": candidate_mean,
        "baseline_mean_l1_m": baseline_mean,
        "relative_improvement": relative_improvement,
        "wins": wins,
        "maximum_case_ratio": maximum_ratio,
        "canonical_reference_draw_mean_l1_m": canonical_mean,
        "published_reference_l1_m": published_reference,
        "improvement_passed": improvement_passed,
        "wins_passed": wins_passed,
        "maximum_case_ratio_passed": ratio_passed,
        "all_unique_below_published_reference": unique_reference_passed,
        "canonical_draw_below_published_reference": canonical_reference_passed,
        "passed": bool(
            improvement_passed
            and wins_passed
            and ratio_passed
            and unique_reference_passed
            and canonical_reference_passed
        ),
        "cases": [
            {
                "name": name,
                "candidate_l1_m": float(candidate_errors[index]),
                "baseline_l1_m": float(baseline_errors[index]),
                "candidate_to_baseline_ratio": float(ratios[index]),
                "candidate_wins": bool(
                    candidate_errors[index] < baseline_errors[index]
                ),
            }
            for index, name in enumerate(normalized_names)
        ],
    }
