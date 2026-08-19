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
    deserialize_deform_local_residual_model,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_pyelastica import (
    deform_pyelastica_parameter_bank,
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
    # The preregistered intercept-only arm deliberately has zero feature
    # columns.  Its normalization matrices are therefore finite (N, 0)
    # arrays, while the shared non-empty array validator rejects them.
    location = np.asarray(model.get("feature_location"), dtype=np.float64)
    scale = np.asarray(model.get("feature_scale"), dtype=np.float64)
    if location.ndim != 2 or not np.isfinite(location).all():
        raise ValueError("variant feature location must be a finite 2-D array")
    if scale.ndim != 2 or not np.isfinite(scale).all():
        raise ValueError("variant feature scale must be a finite 2-D array")
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


def evaluate_deform_backend_portability_report(
    candidate_predictions: Array,
    backend_predictions: Array,
    targets: Array,
    names: Sequence[str],
) -> dict[str, object]:
    """Report a fixed backend correction without creating a target-side gate."""

    candidate = _finite_array(
        candidate_predictions, ndim=4, label="backend portability candidate"
    )
    backend = _finite_array(
        backend_predictions, ndim=4, label="backend portability baseline"
    )
    observed = _finite_array(targets, ndim=4, label="backend portability outcomes")
    normalized_names = tuple(str(name) for name in names)
    if (
        candidate.shape != backend.shape
        or candidate.shape != observed.shape
        or candidate.shape[0] < 1
        or len(normalized_names) != candidate.shape[0]
        or len(set(normalized_names)) != len(normalized_names)
    ):
        raise ValueError("DEFORM backend portability arrays do not align")
    candidate_errors = np.mean(np.abs(candidate - observed), axis=(1, 2, 3))
    backend_errors = np.mean(np.abs(backend - observed), axis=(1, 2, 3))
    if np.any(backend_errors <= 0.0):
        raise ValueError("DEFORM backend portability error must be positive")
    ratios = candidate_errors / backend_errors
    candidate_mean = float(np.mean(candidate_errors))
    backend_mean = float(np.mean(backend_errors))
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-backend-portability-report-v1",
        "case_count": len(normalized_names),
        "candidate_mean_l1_m": candidate_mean,
        "backend_mean_l1_m": backend_mean,
        "relative_improvement": 1.0 - candidate_mean / backend_mean,
        "wins": int(np.count_nonzero(candidate_errors < backend_errors)),
        "maximum_case_ratio": float(np.max(ratios)),
        "selection_effect": "none",
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


def evaluate_deform_compute_matched_report(
    candidate_predictions: Array,
    registered_physical_predictions: Array,
    compute_matched_predictions: Array,
    targets: Array,
    names: Sequence[str],
) -> dict[str, object]:
    """Report the frozen wall-time-equivalent physical control without selection."""

    candidate = _finite_array(
        candidate_predictions, ndim=4, label="compute-matched candidate"
    )
    registered = _finite_array(
        registered_physical_predictions,
        ndim=4,
        label="registered physical baseline",
    )
    compute_matched = _finite_array(
        compute_matched_predictions,
        ndim=4,
        label="compute-matched physical baseline",
    )
    observed = _finite_array(targets, ndim=4, label="compute-matched outcomes")
    normalized_names = tuple(str(name) for name in names)
    if (
        candidate.shape != registered.shape
        or candidate.shape != compute_matched.shape
        or candidate.shape != observed.shape
        or candidate.shape[0] < 1
        or len(normalized_names) != candidate.shape[0]
        or len(set(normalized_names)) != len(normalized_names)
    ):
        raise ValueError("DEFORM compute-matched arrays do not align")
    candidate_errors = np.mean(np.abs(candidate - observed), axis=(1, 2, 3))
    registered_errors = np.mean(np.abs(registered - observed), axis=(1, 2, 3))
    compute_errors = np.mean(np.abs(compute_matched - observed), axis=(1, 2, 3))
    if np.any(registered_errors <= 0.0) or np.any(compute_errors <= 0.0):
        raise ValueError("DEFORM compute-matched baseline error must be positive")
    candidate_mean = float(np.mean(candidate_errors))
    registered_mean = float(np.mean(registered_errors))
    compute_mean = float(np.mean(compute_errors))
    candidate_ratios = candidate_errors / compute_errors
    compute_ratios = compute_errors / registered_errors
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-compute-matched-report-v1",
        "case_count": len(normalized_names),
        "candidate_mean_l1_m": candidate_mean,
        "registered_physical_mean_l1_m": registered_mean,
        "compute_matched_physical_mean_l1_m": compute_mean,
        "candidate_relative_improvement_over_compute_matched": (
            1.0 - candidate_mean / compute_mean
        ),
        "compute_matched_relative_improvement_over_registered": (
            1.0 - compute_mean / registered_mean
        ),
        "candidate_wins_over_compute_matched": int(
            np.count_nonzero(candidate_errors < compute_errors)
        ),
        "compute_matched_wins_over_registered": int(
            np.count_nonzero(compute_errors < registered_errors)
        ),
        "maximum_candidate_to_compute_matched_ratio": float(np.max(candidate_ratios)),
        "maximum_compute_matched_to_registered_ratio": float(np.max(compute_ratios)),
        "selection_effect": "none",
        "cases": [
            {
                "name": name,
                "candidate_l1_m": float(candidate_errors[index]),
                "registered_physical_l1_m": float(registered_errors[index]),
                "compute_matched_physical_l1_m": float(compute_errors[index]),
                "candidate_to_compute_matched_ratio": float(candidate_ratios[index]),
                "compute_matched_to_registered_ratio": float(compute_ratios[index]),
                "candidate_wins_over_compute_matched": bool(
                    candidate_errors[index] < compute_errors[index]
                ),
                "compute_matched_wins_over_registered": bool(
                    compute_errors[index] < registered_errors[index]
                ),
            }
            for index, name in enumerate(normalized_names)
        ],
    }


def validate_deform_dlo3_alltrain_compute_match_v1(
    value: object,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Validate the target-blind all-train compute-matching arithmetic."""

    record = _mapping(value, label="all-train compute match")
    compute = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    training = _mapping(protocol.get("physical_training"), label="physical training")
    local_seconds = float(
        cast(Any, record.get("local_residual_wall_seconds", math.nan))
    )
    update_seconds = float(
        cast(Any, record.get("median_update_seconds_6301_6400", math.nan))
    )
    additional_updates = int(cast(Any, record.get("additional_updates", -1)))
    start_update = int(cast(Any, record.get("start_update", -1)))
    end_update = int(cast(Any, record.get("end_update", -1)))
    expected_additional = (
        int(math.ceil(local_seconds / update_seconds))
        if math.isfinite(local_seconds)
        and local_seconds > 0.0
        and math.isfinite(update_seconds)
        and update_seconds > 0.0
        else -1
    )
    if (
        record.get("contract") != "deform-dlo3-alltrain-compute-match-v1"
        or int(cast(Any, record.get("seed", -1)))
        != int(cast(Any, training["primary_seed"]))
        or additional_updates != expected_additional
        or additional_updates < int(cast(Any, compute["minimum_additional_updates"]))
        or additional_updates > int(cast(Any, compute["maximum_additional_updates"]))
        or start_update != int(cast(Any, compute["start_update"]))
        or end_update != start_update + additional_updates
        or record.get("selection_effect") != "none"
        or record.get("target_selection") is not False
        or record.get("target_calibration") is not False
        or record.get("target_retries") is not False
        or record.get("primary_eval_read") is not False
    ):
        raise ValueError("DEFORM all-train compute match differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-alltrain-compute-match-verification-v1",
        "seed": int(cast(Any, record["seed"])),
        "local_residual_wall_seconds": local_seconds,
        "median_update_seconds_6301_6400": update_seconds,
        "additional_updates": additional_updates,
        "start_update": start_update,
        "end_update": end_update,
        "selection_effect": "none",
        "verified": True,
    }


def validate_deform_compute_matched_report_v1(
    value: object,
    *,
    expected_case_count: int,
) -> dict[str, object]:
    """Validate a descriptive compute-matched report without making a gate."""

    report = _mapping(value, label="compute-matched report")
    cases_value = report.get("cases")
    if not isinstance(cases_value, Sequence) or isinstance(cases_value, (str, bytes)):
        raise ValueError("DEFORM compute-matched report cases differ")
    cases = tuple(_mapping(case, label="compute-matched case") for case in cases_value)
    candidate_mean = float(cast(Any, report.get("candidate_mean_l1_m", math.nan)))
    registered_mean = float(
        cast(Any, report.get("registered_physical_mean_l1_m", math.nan))
    )
    compute_mean = float(
        cast(Any, report.get("compute_matched_physical_mean_l1_m", math.nan))
    )
    relative_candidate = float(
        cast(
            Any,
            report.get("candidate_relative_improvement_over_compute_matched", math.nan),
        )
    )
    relative_compute = float(
        cast(
            Any,
            report.get(
                "compute_matched_relative_improvement_over_registered", math.nan
            ),
        )
    )
    if (
        report.get("contract") != "deform-dlo3-compute-matched-report-v1"
        or int(cast(Any, report.get("case_count", -1))) != expected_case_count
        or len(cases) != expected_case_count
        or expected_case_count < 1
        or not math.isfinite(candidate_mean)
        or candidate_mean < 0.0
        or not math.isfinite(registered_mean)
        or registered_mean <= 0.0
        or not math.isfinite(compute_mean)
        or compute_mean <= 0.0
        or not math.isclose(
            relative_candidate,
            1.0 - candidate_mean / compute_mean,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            relative_compute,
            1.0 - compute_mean / registered_mean,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or report.get("selection_effect") != "none"
        or "passed" in report
    ):
        raise ValueError("DEFORM compute-matched report summary differs")

    names: list[str] = []
    candidate_values: list[float] = []
    registered_values: list[float] = []
    compute_values: list[float] = []
    candidate_ratios: list[float] = []
    compute_ratios: list[float] = []
    candidate_wins = 0
    compute_wins = 0
    for case in cases:
        name = str(case.get("name", ""))
        candidate_error = float(cast(Any, case.get("candidate_l1_m", math.nan)))
        registered_error = float(
            cast(Any, case.get("registered_physical_l1_m", math.nan))
        )
        compute_error = float(
            cast(Any, case.get("compute_matched_physical_l1_m", math.nan))
        )
        candidate_ratio = float(
            cast(Any, case.get("candidate_to_compute_matched_ratio", math.nan))
        )
        compute_ratio = float(
            cast(Any, case.get("compute_matched_to_registered_ratio", math.nan))
        )
        candidate_won = candidate_error < compute_error
        compute_won = compute_error < registered_error
        if (
            not name
            or not math.isfinite(candidate_error)
            or candidate_error < 0.0
            or not math.isfinite(registered_error)
            or registered_error <= 0.0
            or not math.isfinite(compute_error)
            or compute_error <= 0.0
            or not math.isclose(
                candidate_ratio,
                candidate_error / compute_error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                compute_ratio,
                compute_error / registered_error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or case.get("candidate_wins_over_compute_matched") is not candidate_won
            or case.get("compute_matched_wins_over_registered") is not compute_won
        ):
            raise ValueError("DEFORM compute-matched report case differs")
        names.append(name)
        candidate_values.append(candidate_error)
        registered_values.append(registered_error)
        compute_values.append(compute_error)
        candidate_ratios.append(candidate_ratio)
        compute_ratios.append(compute_ratio)
        candidate_wins += int(candidate_won)
        compute_wins += int(compute_won)
    if (
        len(set(names)) != expected_case_count
        or not math.isclose(
            candidate_mean,
            float(np.mean(candidate_values)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            registered_mean,
            float(np.mean(registered_values)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            compute_mean,
            float(np.mean(compute_values)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or int(cast(Any, report.get("candidate_wins_over_compute_matched", -1)))
        != candidate_wins
        or int(cast(Any, report.get("compute_matched_wins_over_registered", -1)))
        != compute_wins
        or not math.isclose(
            float(
                cast(
                    Any,
                    report.get("maximum_candidate_to_compute_matched_ratio", math.nan),
                )
            ),
            max(candidate_ratios),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(
                cast(
                    Any,
                    report.get("maximum_compute_matched_to_registered_ratio", math.nan),
                )
            ),
            max(compute_ratios),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("DEFORM compute-matched report aggregate differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-compute-matched-report-verification-v1",
        "case_count": expected_case_count,
        "case_names": names,
        "candidate_wins_over_compute_matched": candidate_wins,
        "compute_matched_wins_over_registered": compute_wins,
        "selection_effect": "none",
        "verified": True,
    }


def verify_deform_dlo3_evaluator_compute_matched_artifacts_v1(
    result: Mapping[str, object],
    *,
    expected_mode: str,
) -> dict[str, object]:
    """Rehash the evaluator's sealed compute-matched prediction arm."""

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
        raise ValueError("DEFORM evaluator compute-matched custody differs")
    seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="evaluator prediction seal"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("evaluator prediction seal must be a JSON object")
    if (
        seal.get("contract") != "deform-dlo3-robustness-evaluator-prediction-seal-v1"
        or seal.get("mode") != expected_mode
        or seal.get("outcomes_scored") is not False
        or seal.get("target_retries") is not False
    ):
        raise ValueError("DEFORM evaluator compute-matched seal differs")
    sealed_control = _mapping(
        seal.get("compute_matched_control"), label="sealed compute-matched control"
    )
    result_control = _mapping(
        result.get("compute_matched_control"), label="compute-matched control"
    )
    predictions_path = _verified_deform_artifact_path(
        seal.get("predictions"), label="evaluator predictions"
    )
    expected_case_count = 8 if expected_mode == "dry-run" else 14
    with np.load(predictions_path, allow_pickle=False) as archive:
        names = tuple(str(value) for value in np.asarray(archive["names"]))
        required = ("candidate", "baseline")
        if any(key not in archive.files for key in required):
            raise ValueError("DEFORM evaluator compute-matched archive differs")
        candidate = _finite_array(
            np.asarray(archive["candidate"]), ndim=4, label="sealed candidate"
        )
        baseline = _finite_array(
            np.asarray(archive["baseline"]), ndim=4, label="sealed baseline"
        )
        compute_present = "compute_matched_physical" in archive.files
        compute_values = (
            _finite_array(
                np.asarray(archive["compute_matched_physical"]),
                ndim=4,
                label="sealed compute-matched physical",
            )
            if compute_present
            else None
        )
    if (
        len(names) != expected_case_count
        or len(set(names)) != expected_case_count
        or candidate.shape != baseline.shape
        or candidate.shape[0] != expected_case_count
    ):
        raise ValueError("DEFORM evaluator compute-matched archive differs")

    status = str(result_control.get("status", ""))
    if status == "scored":
        if (
            sealed_control.get("status") != "sealed"
            or sealed_control.get("selection_effect") != "none"
            or sealed_control.get("retry_authorized") is not False
            or result_control.get("selection_effect") != "none"
            or result_control.get("retry_authorized") is not False
            or any(
                result_control.get(field) != sealed_control.get(field)
                for field in (
                    "checkpoint",
                    "compute_match",
                    "compute_match_verification",
                )
            )
            or compute_values is None
            or compute_values.shape != candidate.shape
        ):
            raise ValueError("DEFORM evaluator compute-matched scored arm differs")
        report = validate_deform_compute_matched_report_v1(
            result_control.get("report"), expected_case_count=expected_case_count
        )
        if tuple(cast(Sequence[str], report["case_names"])) != names:
            raise ValueError("DEFORM compute-matched report identity order differs")
    elif status == "technical-failure":
        if (
            expected_mode != "official"
            or dict(result_control) != dict(sealed_control)
            or result_control.get("stage") != "compute-matched-rollout"
            or result_control.get("selection_effect") != "none"
            or result_control.get("retry_authorized") is not False
            or compute_present
        ):
            raise ValueError("DEFORM evaluator compute-matched failure differs")
        report = None
    else:
        raise ValueError("DEFORM evaluator compute-matched status differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-evaluator-compute-matched-artifact-verification-v1",
        "mode": expected_mode,
        "status": status,
        "prediction_seal_sha256": sha256_file(seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "report": report,
        "selection_effect": "none",
        "verified": True,
    }


def _validate_deform_casewise_gate_record_v1(
    value: object,
    protocol: Mapping[str, object],
    *,
    kind: str,
) -> dict[str, object]:
    gate = _mapping(value, label=f"{kind} source gate")
    if kind == "primary":
        expected_contract = "deform-dlo3-robustness-source-gate-v1"
        baseline_mean_key = "baseline_mean_l1_m"
        baseline_case_key = "baseline_l1_m"
        ratio_case_key = "candidate_to_baseline_ratio"
        thresholds = _mapping(protocol.get("source_gate"), label="source gate")
        minimum_improvement = float(
            cast(Any, thresholds["minimum_relative_improvement"])
        )
        minimum_wins = int(cast(Any, thresholds["minimum_case_wins"]))
        maximum_ratio_threshold = float(cast(Any, thresholds["maximum_case_ratio"]))
    elif kind == "backend":
        expected_contract = "deform-dlo3-pyelastica-source-gate-v1"
        baseline_mean_key = "backend_mean_l1_m"
        baseline_case_key = "backend_l1_m"
        ratio_case_key = "candidate_to_backend_ratio"
        thresholds = _mapping(
            protocol.get("backend_portability"), label="backend portability"
        )
        minimum_improvement = float(
            cast(Any, thresholds["minimum_relative_improvement_over_backend"])
        )
        minimum_wins = int(cast(Any, thresholds["minimum_source_test_wins"]))
        maximum_ratio_threshold = float(
            cast(Any, thresholds["maximum_source_test_ratio"])
        )
    else:
        raise ValueError("unsupported DEFORM source gate kind")

    cases_value = gate.get("cases")
    if not isinstance(cases_value, Sequence) or isinstance(cases_value, (str, bytes)):
        raise ValueError("DEFORM source gate cases differ")
    cases = tuple(_mapping(case, label="source gate case") for case in cases_value)
    candidate_mean = float(cast(Any, gate.get("candidate_mean_l1_m", math.nan)))
    baseline_mean = float(cast(Any, gate.get(baseline_mean_key, math.nan)))
    relative_improvement = float(cast(Any, gate.get("relative_improvement", math.nan)))
    maximum_ratio = float(cast(Any, gate.get("maximum_case_ratio", math.nan)))
    if (
        gate.get("contract") != expected_contract
        or int(cast(Any, gate.get("case_count", -1))) != 8
        or len(cases) != 8
        or not math.isfinite(candidate_mean)
        or candidate_mean < 0.0
        or not math.isfinite(baseline_mean)
        or baseline_mean <= 0.0
        or not math.isfinite(relative_improvement)
        or not math.isclose(
            relative_improvement,
            1.0 - candidate_mean / baseline_mean,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("DEFORM source gate summary differs")

    names: list[str] = []
    candidate_values: list[float] = []
    baseline_values: list[float] = []
    ratios: list[float] = []
    wins = 0
    for case in cases:
        name = str(case.get("name", ""))
        candidate_error = float(cast(Any, case.get("candidate_l1_m", math.nan)))
        baseline_error = float(cast(Any, case.get(baseline_case_key, math.nan)))
        ratio = float(cast(Any, case.get(ratio_case_key, math.nan)))
        candidate_wins = candidate_error < baseline_error
        if (
            not name
            or not math.isfinite(candidate_error)
            or candidate_error < 0.0
            or not math.isfinite(baseline_error)
            or baseline_error <= 0.0
            or not math.isfinite(ratio)
            or not math.isclose(
                ratio,
                candidate_error / baseline_error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or case.get("candidate_wins") is not candidate_wins
        ):
            raise ValueError("DEFORM source gate case differs")
        names.append(name)
        candidate_values.append(candidate_error)
        baseline_values.append(baseline_error)
        ratios.append(ratio)
        wins += int(candidate_wins)
    if len(set(names)) != 8:
        raise ValueError("DEFORM source gate case identities differ")
    improvement_passed = relative_improvement >= minimum_improvement
    wins_passed = wins >= minimum_wins
    ratio_passed = max(ratios) <= maximum_ratio_threshold
    passed = improvement_passed and wins_passed and ratio_passed
    if kind == "primary":
        reference_passed = candidate_mean < float(
            cast(Any, thresholds["maximum_candidate_l1_m"])
        )
        passed = passed and reference_passed
        if gate.get("published_reference_passed") is not reference_passed:
            raise ValueError("DEFORM source gate reference decision differs")
    if (
        not math.isclose(
            candidate_mean,
            float(np.mean(candidate_values)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            baseline_mean,
            float(np.mean(baseline_values)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or int(cast(Any, gate.get("wins", -1))) != wins
        or not math.isclose(maximum_ratio, max(ratios), rel_tol=1e-12, abs_tol=1e-15)
        or gate.get("improvement_passed") is not improvement_passed
        or gate.get("wins_passed") is not wins_passed
        or gate.get("maximum_case_ratio_passed") is not ratio_passed
        or gate.get("passed") is not passed
    ):
        raise ValueError("DEFORM source gate decision differs")
    return {
        "contract": expected_contract,
        "case_count": 8,
        "passed": passed,
        "wins": wins,
        "maximum_case_ratio": maximum_ratio,
    }


def verify_deform_dlo3_seed_diagnostic_artifacts_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Rehash the frozen mechanism and compute-matched source controls."""

    if (
        result.get("contract") != "deform-dlo3-robustness-seed-result-v1"
        or result.get("source_test_opened") is not True
        or result.get("primary_eval_enumerated") is not False
        or result.get("primary_eval_read") is not False
        or result.get("target_authorized") is not False
        or result.get("retry_authorized") is not False
        or result.get("prob4d_used") is not False
        or result.get("held_v8_access") is not False
    ):
        raise ValueError("DEFORM seed diagnostic artifact custody differs")
    training = _mapping(protocol.get("physical_training"), label="physical training")
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    compute_contract = _mapping(
        protocol.get("compute_matched_control"), label="compute-matched control"
    )
    expected_arms = _strings(
        _mapping(protocol.get("mechanism_ablation"), label="mechanism ablation").get(
            "arms"
        ),
        label="mechanism arms",
    )
    seed = int(cast(Any, result.get("seed", -1)))
    if seed not in _integers(training.get("audit_seeds"), label="audit seeds"):
        raise ValueError("DEFORM seed diagnostic identity differs")

    result_protocol = _mapping(result.get("protocol"), label="seed protocol")
    result_manifest = _mapping(result.get("source_manifest"), label="seed manifest")
    method_identity = _mapping(result.get("method_seal"), label="seed method seal")
    method_path = _verified_deform_artifact_path(
        method_identity, label="seed diagnostic method seal"
    )
    method = json.loads(method_path.read_text(encoding="utf-8"))
    if not isinstance(method, Mapping):
        raise ValueError("seed diagnostic method seal must be a JSON object")
    if (
        method.get("contract") != "deform-dlo3-robustness-source-method-seal-v1"
        or int(cast(Any, method.get("seed", -1))) != seed
        or _mapping(method.get("protocol"), label="method protocol").get("sha256")
        != result_protocol.get("sha256")
        or _mapping(method.get("source_manifest"), label="method manifest").get(
            "sha256"
        )
        != result_manifest.get("sha256")
        or float(cast(Any, method.get("ridge", math.nan)))
        != float(cast(Any, residual["ridge"]))
        or float(cast(Any, method.get("shrinkage", math.nan)))
        != float(cast(Any, residual["shrinkage"]))
        or method.get("source_test_opened") is not False
        or method.get("official_eval_read") is not False
        or method.get("target_selection") is not False
    ):
        raise ValueError("DEFORM seed diagnostic method seal differs")

    physical_identity = dict(
        _mapping(method.get("physical_checkpoint"), label="physical checkpoint")
    )
    compute_identity = dict(
        _mapping(
            method.get("compute_matched_checkpoint"),
            label="compute-matched checkpoint",
        )
    )
    _verified_deform_artifact_path(physical_identity, label="physical checkpoint")
    _verified_deform_artifact_path(compute_identity, label="compute-matched checkpoint")
    for field, label in (
        ("local_residual_model", "local residual model"),
        ("full_covariance_model", "full covariance model"),
    ):
        _verified_deform_artifact_path(method.get(field), label=label)
    calibration_path = _verified_deform_artifact_path(
        method.get("covariance_calibration"), label="covariance calibration"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_calibration = _mapping(
        _mapping(result.get("bayesian_audit"), label="Bayesian audit").get(
            "calibration"
        ),
        label="source calibration",
    )
    if calibration != source_calibration:
        raise ValueError("DEFORM seed diagnostic calibration differs")

    expected_models = {
        "persistence-plus-full-local": ("full-local", "initial-action-local"),
        "physical-plus-intercept-only": (
            "intercept-only",
            "initial-action-local",
        ),
        "physical-plus-full-no-action": (
            "full-no-action",
            "initial-action-local",
        ),
        "physical-plus-full-global-frame": (
            "full-global",
            "action-centered-global",
        ),
    }
    model_identities = _mapping(
        method.get("mechanism_models"), label="mechanism models"
    )
    if set(str(label) for label in model_identities) != set(expected_models):
        raise ValueError("DEFORM seed diagnostic mechanism model set differs")
    verified_models: dict[str, str] = {}
    for label, (arm, coordinate_frame) in expected_models.items():
        path = _verified_deform_artifact_path(
            model_identities.get(label), label=f"mechanism model {label}"
        )
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "arm",
                "coordinate_frame",
                "node_count",
                "prediction_horizon",
                "feature_indices",
                "feature_location",
                "feature_scale",
                "coefficients",
                "ridge",
            }
            if set(archive.files) != required:
                raise ValueError("DEFORM seed diagnostic mechanism model differs")
            feature_indices = tuple(
                int(value) for value in np.asarray(archive["feature_indices"])
            )
            location = np.asarray(archive["feature_location"], dtype=np.float64)
            scale = np.asarray(archive["feature_scale"], dtype=np.float64)
            coefficients = _finite_array(
                np.asarray(archive["coefficients"]),
                ndim=3,
                label="mechanism coefficients",
            )
            if (
                tuple(str(value) for value in np.asarray(archive["arm"])) != (arm,)
                or tuple(
                    str(value) for value in np.asarray(archive["coordinate_frame"])
                )
                != (coordinate_frame,)
                or tuple(int(value) for value in np.asarray(archive["node_count"]))
                != (12,)
                or tuple(
                    int(value) for value in np.asarray(archive["prediction_horizon"])
                )
                != (498,)
                or feature_indices != deform_local_feature_indices(arm)
                or location.ndim != 2
                or scale.ndim != 2
                or not np.isfinite(location).all()
                or not np.isfinite(scale).all()
                or location.shape != (8, len(feature_indices))
                or scale.shape != location.shape
                or coefficients.shape != (8, len(feature_indices) + 1, 3)
                or np.any(scale <= 0.0)
                or tuple(float(value) for value in np.asarray(archive["ridge"]))
                != (float(cast(Any, residual["ridge"])),)
            ):
                raise ValueError("DEFORM seed diagnostic mechanism model differs")
        verified_models[label] = sha256_file(path)

    compute = _mapping(result.get("compute_match"), label="compute match")
    local_seconds = float(
        cast(Any, compute.get("local_residual_wall_seconds", math.nan))
    )
    update_seconds = float(cast(Any, compute.get("median_update_seconds_6301_6400")))
    additional_updates = int(cast(Any, compute.get("additional_updates", -1)))
    expected_additional = (
        int(math.ceil(local_seconds / update_seconds))
        if math.isfinite(local_seconds)
        and local_seconds > 0.0
        and math.isfinite(update_seconds)
        and update_seconds > 0.0
        else -1
    )
    start_update = int(cast(Any, compute.get("start_update", -1)))
    end_update = int(cast(Any, compute.get("end_update", -1)))
    source_mean = float(cast(Any, compute.get("source_mean_l1_m", math.nan)))
    compute_checkpoint = dict(
        _mapping(compute.get("checkpoint"), label="compute result checkpoint")
    )
    if (
        compute.get("contract") != "deform-dlo3-compute-match-v1"
        or int(cast(Any, compute.get("seed", -1))) != seed
        or not math.isfinite(source_mean)
        or source_mean < 0.0
        or additional_updates != expected_additional
        or additional_updates
        < int(cast(Any, compute_contract["minimum_additional_updates"]))
        or additional_updates
        > int(cast(Any, compute_contract["maximum_additional_updates"]))
        or start_update != int(cast(Any, compute_contract["start_update"]))
        or end_update != start_update + additional_updates
        or compute.get("source_test_opened") is not False
        or compute.get("official_eval_read") is not False
        or compute_checkpoint != compute_identity
        or int(cast(Any, compute_checkpoint.get("update", -1))) != end_update
        or dict(_mapping(result.get("physical_checkpoint"), label="result checkpoint"))
        != physical_identity
        or int(cast(Any, physical_identity.get("update", -1))) != start_update
    ):
        raise ValueError("DEFORM seed compute-matched control differs")

    seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="seed diagnostic prediction seal"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("seed diagnostic prediction seal must be a JSON object")
    if (
        seal.get("contract") != "deform-dlo3-robustness-source-prediction-seal-v1"
        or int(cast(Any, seal.get("seed", -1))) != seed
        or _mapping(seal.get("method_seal"), label="sealed method").get("sha256")
        != method_identity.get("sha256")
        or int(cast(Any, seal.get("source_test_case_count", -1))) != 8
        or seal.get("source_outcomes_scored") is not False
        or seal.get("official_eval_read") is not False
    ):
        raise ValueError("DEFORM seed diagnostic prediction seal differs")
    predictions_path = _verified_deform_artifact_path(
        seal.get("predictions"), label="seed diagnostic predictions"
    )
    required_predictions = {
        "names",
        "physical",
        "compute_matched_physical",
        "candidate",
    } | {f"mechanism_{label}" for label in expected_arms}
    with np.load(predictions_path, allow_pickle=False) as archive:
        if not required_predictions.issubset(archive.files):
            raise ValueError("DEFORM seed diagnostic predictions are incomplete")
        names = tuple(str(value) for value in np.asarray(archive["names"]))
        predictions = {
            key: _finite_array(
                np.asarray(archive[key]), ndim=4, label=f"diagnostic prediction {key}"
            )
            for key in required_predictions
            if key != "names"
        }
    expected_shape = (8, 498, 12, 3)
    if (
        len(names) != 8
        or len(set(names)) != 8
        or any(values.shape != expected_shape for values in predictions.values())
        or not np.array_equal(
            predictions["mechanism_physical-only"], predictions["physical"]
        )
        or not np.array_equal(
            predictions["mechanism_physical-plus-full-local-fixed"],
            predictions["candidate"],
        )
    ):
        raise ValueError("DEFORM seed diagnostic prediction archive differs")

    mechanism_results = _mapping(
        result.get("mechanism_ablation"), label="mechanism results"
    )
    if set(str(label) for label in mechanism_results) != set(expected_arms):
        raise ValueError("DEFORM seed diagnostic mechanism result set differs")
    verified_gates: dict[str, dict[str, object]] = {}
    for label in expected_arms:
        raw_gate = _mapping(mechanism_results.get(label), label=f"mechanism {label}")
        cases = cast(Sequence[Mapping[str, object]], raw_gate.get("cases"))
        if tuple(str(case.get("name", "")) for case in cases) != names:
            raise ValueError("DEFORM seed diagnostic mechanism case order differs")
        verified_gates[label] = _validate_deform_casewise_gate_record_v1(
            raw_gate, protocol, kind="primary"
        )
    physical_gate = _mapping(
        mechanism_results.get("physical-only"), label="physical-only mechanism"
    )
    if float(cast(Any, physical_gate.get("candidate_mean_l1_m", math.nan))) != float(
        cast(Any, physical_gate.get("baseline_mean_l1_m", math.nan))
    ) or any(
        float(cast(Any, case.get("candidate_to_baseline_ratio", math.nan))) != 1.0
        for case in cast(Sequence[Mapping[str, object]], physical_gate.get("cases"))
    ):
        raise ValueError("DEFORM physical-only mechanism control differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-seed-diagnostic-artifact-verification-v1",
        "seed": seed,
        "mechanism_arm_count": len(expected_arms),
        "mechanism_models": verified_models,
        "mechanism_gates": verified_gates,
        "compute_matched_additional_updates": additional_updates,
        "compute_matched_source_mean_l1_m": source_mean,
        "prediction_seal_sha256": sha256_file(seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "verified": True,
    }


def verify_deform_dlo3_stability_artifacts_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the frozen three-seed gate from every rehashed source bundle."""

    identities_value = result.get("seed_results")
    if not isinstance(identities_value, Sequence) or isinstance(
        identities_value, (str, bytes)
    ):
        raise ValueError("DEFORM stability seed identities differ")
    identities = tuple(
        _mapping(value, label="stability seed identity") for value in identities_value
    )
    if len(identities) != 3:
        raise ValueError("DEFORM stability seed identities differ")
    paths = tuple(
        _verified_deform_artifact_path(value, label="stability seed result")
        for value in identities
    )
    seed_results: list[Mapping[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("stability seed result must be a JSON object")
        seed_results.append(payload)
    gate = evaluate_deform_dlo3_stability_gate(seed_results, protocol)
    if any(result.get(key) != value for key, value in gate.items()):
        raise ValueError("DEFORM stability gate replay differs")
    bayesian = [
        verify_deform_dlo3_seed_bayesian_artifacts_v1(seed_result)
        for seed_result in seed_results
    ]
    diagnostics = [
        verify_deform_dlo3_seed_diagnostic_artifacts_v1(seed_result, protocol)
        for seed_result in seed_results
    ]
    if (
        result.get("bayesian_artifacts_verified") is not True
        or result.get("bayesian_artifact_verifications") != bayesian
        or result.get("diagnostic_artifacts_verified") is not True
        or result.get("diagnostic_artifact_verifications") != diagnostics
        or int(cast(Any, result.get("diagnostic_seed_count", -1))) != 3
        or _mapping(result.get("protocol"), label="stability protocol").get("sha256")
        != gate["protocol_sha256"]
    ):
        raise ValueError("DEFORM stability artifact verification differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-stability-artifact-verification-v1",
        "seed_count": 3,
        "seed_result_sha256s": [sha256_file(path) for path in paths],
        "seed_result_sha256_by_seed": {
            str(int(cast(Any, seed_result["seed"]))): sha256_file(path)
            for seed_result, path in zip(seed_results, paths, strict=True)
        },
        "bayesian_artifacts_verified": True,
        "diagnostic_artifacts_verified": True,
        "gate_passed": gate["passed"],
        "verified": True,
    }


def validate_deform_dlo3_sensitivity_result_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Require every frozen solver/material sensitivity arm and no selection."""

    sensitivity = _mapping(
        protocol.get("physics_solver_sensitivity"), label="sensitivity"
    )
    expected = tuple(
        f"pbd-{int(value)}"
        for value in cast(Sequence[Any], sensitivity["pbd_iteration_values"])
    ) + tuple(
        f"stiffness-{float(value):.1f}"
        for value in cast(Sequence[Any], sensitivity["joint_bend_twist_multipliers"])
    )
    variants = _mapping(result.get("variants"), label="sensitivity variants")
    if (
        result.get("contract") != "deform-dlo3-physics-solver-sensitivity-result-v1"
        or len(variants) != len(expected)
        or set(str(name) for name in variants) != set(expected)
        or result.get("selection_effect") != "none"
        or result.get("nominal_replay_exact") is not True
        or result.get("source_test_opened") is not True
        or result.get("primary_eval_enumerated") is not False
        or result.get("primary_eval_read") is not False
        or result.get("target_authorized") is not False
        or result.get("retry_authorized") is not False
        or result.get("prob4d_used") is not False
        or result.get("held_v8_access") is not False
    ):
        raise ValueError("DEFORM sensitivity result differs")
    records = {
        name: _validate_deform_casewise_gate_record_v1(
            variants.get(name), protocol, kind="primary"
        )
        for name in expected
    }
    if variants.get("pbd-10") != variants.get("stiffness-1.0"):
        raise ValueError("DEFORM nominal sensitivity records differ")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-sensitivity-result-verification-v1",
        "variant_count": len(expected),
        "variants": records,
        "selection_effect": "none",
        "verified": True,
    }


def verify_deform_dlo3_sensitivity_artifacts_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Rehash the complete frozen solver/material sensitivity matrix."""

    verification = validate_deform_dlo3_sensitivity_result_v1(result, protocol)
    sensitivity = _mapping(
        protocol.get("physics_solver_sensitivity"), label="sensitivity"
    )
    labels = tuple(
        f"pbd-{int(value)}"
        for value in cast(Sequence[Any], sensitivity["pbd_iteration_values"])
    ) + tuple(
        f"stiffness-{float(value):.1f}"
        for value in cast(Sequence[Any], sensitivity["joint_bend_twist_multipliers"])
    )
    parent_path = _verified_deform_artifact_path(
        result.get("seed_result"), label="sensitivity parent seed result"
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, Mapping):
        raise ValueError("sensitivity parent result must be a JSON object")
    parent_protocol = _mapping(parent.get("protocol"), label="parent protocol")
    parent_manifest = _mapping(parent.get("source_manifest"), label="parent manifest")
    if (
        parent.get("contract") != "deform-dlo3-robustness-seed-result-v1"
        or _mapping(result.get("protocol"), label="sensitivity protocol").get("sha256")
        != parent_protocol.get("sha256")
        or _mapping(result.get("source_manifest"), label="sensitivity manifest").get(
            "sha256"
        )
        != parent_manifest.get("sha256")
    ):
        raise ValueError("DEFORM sensitivity parent lineage differs")
    parent_seal_path = _verified_deform_artifact_path(
        parent.get("prediction_seal"), label="parent prediction seal"
    )
    parent_seal = json.loads(parent_seal_path.read_text(encoding="utf-8"))
    if not isinstance(parent_seal, Mapping):
        raise ValueError("parent prediction seal must be a JSON object")
    parent_predictions_path = _verified_deform_artifact_path(
        parent_seal.get("predictions"), label="parent source predictions"
    )
    with np.load(parent_predictions_path, allow_pickle=False) as archive:
        parent_names = tuple(str(value) for value in np.asarray(archive["names"]))
        parent_candidate = _finite_array(
            np.asarray(archive["candidate"]),
            ndim=4,
            label="parent source candidate",
        )

    seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="sensitivity prediction seal"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("sensitivity prediction seal must be a JSON object")
    if (
        seal.get("contract") != "deform-dlo3-sensitivity-prediction-seal-v1"
        or int(cast(Any, seal.get("variant_count", -1))) != len(labels)
        or seal.get("source_outcomes_scored") is not False
        or seal.get("primary_eval_read") is not False
    ):
        raise ValueError("DEFORM sensitivity prediction seal differs")
    predictions_path = _verified_deform_artifact_path(
        seal.get("predictions"), label="sensitivity predictions"
    )
    required = {"names"} | {
        f"{prefix}_{label}" for prefix in ("physical", "candidate") for label in labels
    }
    with np.load(predictions_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("DEFORM sensitivity prediction matrix differs")
        names = tuple(str(value) for value in np.asarray(archive["names"]))
        arrays = {
            key: _finite_array(
                np.asarray(archive[key]), ndim=4, label=f"sensitivity {key}"
            )
            for key in required
            if key != "names"
        }
    expected_shape = (8, 498, 12, 3)
    if (
        names != parent_names
        or len(set(names)) != 8
        or any(values.shape != expected_shape for values in arrays.values())
        or not np.array_equal(arrays["candidate_pbd-10"], parent_candidate)
        or not np.array_equal(
            arrays["candidate_pbd-10"], arrays["candidate_stiffness-1.0"]
        )
        or not np.array_equal(
            arrays["physical_pbd-10"], arrays["physical_stiffness-1.0"]
        )
    ):
        raise ValueError("DEFORM sensitivity nominal replay artifact differs")
    variants = _mapping(result.get("variants"), label="sensitivity variants")
    nominal = _mapping(variants.get("pbd-10"), label="nominal sensitivity gate")
    nominal_baseline = float(cast(Any, nominal.get("baseline_mean_l1_m", math.nan)))
    nominal_cases = tuple(
        float(cast(Any, case.get("baseline_l1_m", math.nan)))
        for case in cast(Sequence[Mapping[str, object]], nominal.get("cases"))
    )
    for label in labels:
        gate = _mapping(variants.get(label), label=f"sensitivity gate {label}")
        cases = cast(Sequence[Mapping[str, object]], gate.get("cases"))
        if (
            tuple(str(case.get("name", "")) for case in cases) != names
            or float(cast(Any, gate.get("baseline_mean_l1_m", math.nan)))
            != nominal_baseline
            or tuple(
                float(cast(Any, case.get("baseline_l1_m", math.nan))) for case in cases
            )
            != nominal_cases
        ):
            raise ValueError("DEFORM sensitivity score lineage differs")
    return {
        **verification,
        "contract": "deform-dlo3-sensitivity-artifact-verification-v1",
        "parent_seed_result_sha256": sha256_file(parent_path),
        "prediction_seal_sha256": sha256_file(seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "artifact_matrix_verified": True,
    }


def validate_deform_dlo3_backend_result_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Validate the fixed PyElastica source gate and fallback decision."""

    if (
        result.get("contract") != "deform-dlo3-pyelastica-source-result-v1"
        or result.get("selection_effect") != "none-after-fit"
        or result.get("source_test_opened") is not True
        or result.get("primary_eval_enumerated") is not False
        or result.get("primary_eval_read") is not False
        or result.get("retry_authorized") is not False
        or result.get("prob4d_used") is not False
        or result.get("held_v8_access") is not False
        or result.get("primary_target_authorized") is not False
    ):
        raise ValueError("DEFORM backend result differs")
    gate = _validate_deform_casewise_gate_record_v1(
        result.get("source_gate"), protocol, kind="backend"
    )
    if result.get("backend_target_arm_authorized") is not gate["passed"]:
        raise ValueError("DEFORM backend target authorization differs")
    audit = _mapping(result.get("bayesian_audit"), label="backend Bayesian audit")
    raw = _mapping(audit.get("uncalibrated"), label="backend raw distribution")
    calibrated = _mapping(
        audit.get("calibrated"), label="backend calibrated distribution"
    )
    if (
        raw.get("contract") != "deform-dlo-predictive-distribution-metrics-v1"
        or calibrated.get("contract") != "deform-dlo-predictive-distribution-metrics-v1"
        or raw.get("mean_coordinate_l1_m") != calibrated.get("mean_coordinate_l1_m")
        or audit.get("point_mean_unchanged_by_calibration") is not True
    ):
        raise ValueError("DEFORM backend Bayesian audit differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-backend-result-verification-v1",
        "source_gate": gate,
        "backend_target_arm_authorized": gate["passed"],
        "selection_effect": "none-after-fit",
        "verified": True,
    }


def verify_deform_dlo3_backend_artifacts_v1(
    result: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Rehash the fixed PyElastica method needed by an authorized target arm."""

    verification = validate_deform_dlo3_backend_result_v1(result, protocol)
    protocol_identity = _mapping(result.get("protocol"), label="backend protocol")
    method_identity = _mapping(result.get("method_seal"), label="backend method seal")
    method_path = _verified_deform_artifact_path(
        method_identity, label="backend method seal"
    )
    method = json.loads(method_path.read_text(encoding="utf-8"))
    if not isinstance(method, Mapping):
        raise ValueError("backend method seal must be a JSON object")
    method_protocol = _mapping(method.get("protocol"), label="backend method protocol")
    if (
        method.get("contract") != "deform-dlo3-pyelastica-source-method-seal-v1"
        or method_protocol.get("sha256") != protocol_identity.get("sha256")
        or method.get("source_test_opened") is not False
        or method.get("primary_eval_read") is not False
        or method.get("selection_effect_after_fit") != "none"
        or float(cast(Any, method.get("ridge", math.nan))) != 1.0
        or float(cast(Any, method.get("shrinkage", math.nan))) != 0.25
    ):
        raise ValueError("DEFORM backend method seal differs")
    selected_parameters = dict(
        _mapping(method.get("selected_parameters"), label="backend parameters")
    )
    bank = tuple(
        member.to_record() for member in deform_pyelastica_parameter_bank(protocol)
    )
    if selected_parameters not in bank:
        raise ValueError(
            "DEFORM backend selected parameters differ from the frozen bank"
        )

    model_path = _verified_deform_artifact_path(
        method.get("full_covariance_model"), label="backend full covariance model"
    )
    with np.load(model_path, allow_pickle=False) as archive:
        base_model = deserialize_deform_local_residual_model(archive)
        if not {
            "coefficient_covariance_full",
            "residual_covariance_full",
        }.issubset(archive.files):
            raise ValueError("DEFORM backend full covariance model is incomplete")
        coefficient = _finite_array(
            np.asarray(archive["coefficient_covariance_full"]),
            ndim=5,
            label="backend full coefficient covariance",
        )
        residual = _finite_array(
            np.asarray(archive["residual_covariance_full"]),
            ndim=3,
            label="backend full residual covariance",
        )
    internal_count = int(cast(Any, base_model["node_count"])) - 4
    dimension = int(cast(Any, base_model["feature_count"])) + 1
    if (
        int(cast(Any, base_model["node_count"])) != 12
        or int(cast(Any, base_model["prediction_horizon"])) != 498
        or int(cast(Any, base_model["feature_count"])) != DEFORM_LOCAL_FEATURE_COUNT
        or coefficient.shape != (internal_count, 3, 3, dimension, dimension)
        or residual.shape != (internal_count, 3, 3)
        or not np.allclose(
            coefficient,
            coefficient.transpose(0, 2, 1, 4, 3),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(residual, residual.swapaxes(1, 2), rtol=0.0, atol=1e-12)
        or np.min(np.linalg.eigvalsh(residual)) < -1e-12
    ):
        raise ValueError("DEFORM backend full covariance model differs")

    calibration_path = _verified_deform_artifact_path(
        method.get("covariance_calibration"), label="backend covariance calibration"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, Mapping):
        raise ValueError("backend covariance calibration must be a JSON object")
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
        or float(cast(Any, calibration.get("nominal_coordinate_coverage", math.nan)))
        != 0.9
        or not math.isfinite(radius)
        or radius != max(scores)
        or not math.isfinite(variance_scale)
        or variance_scale < 1.0
        or calibration.get("confidence_increase_forbidden") is not True
        or calibration.get("source_test_opened") is not False
        or calibration.get("primary_eval_read") is not False
    ):
        raise ValueError("DEFORM backend covariance calibration differs")

    prediction_seal_path = _verified_deform_artifact_path(
        result.get("prediction_seal"), label="backend prediction seal"
    )
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    if not isinstance(prediction_seal, Mapping):
        raise ValueError("backend prediction seal must be a JSON object")
    sealed_method = _mapping(
        prediction_seal.get("method_seal"), label="backend sealed method"
    )
    if (
        prediction_seal.get("contract")
        != "deform-dlo3-pyelastica-source-prediction-seal-v1"
        or sealed_method.get("sha256") != method_identity.get("sha256")
        or prediction_seal.get("source_outcomes_scored") is not False
        or prediction_seal.get("primary_eval_read") is not False
    ):
        raise ValueError("DEFORM backend prediction seal differs")
    predictions_path = _verified_deform_artifact_path(
        prediction_seal.get("predictions"), label="backend source predictions"
    )
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {
            "names",
            "backend",
            "candidate",
            "coordinate_covariance_m2",
            "calibrated_coordinate_covariance_m2",
        }
        if not required.issubset(archive.files):
            raise ValueError("DEFORM backend source predictions are incomplete")
        names = np.asarray(archive["names"])
        backend = _finite_array(
            np.asarray(archive["backend"]), ndim=4, label="backend source baseline"
        )
        candidate = _finite_array(
            np.asarray(archive["candidate"]), ndim=4, label="backend source candidate"
        )
        raw_covariance = _finite_array(
            np.asarray(archive["coordinate_covariance_m2"]),
            ndim=5,
            label="backend source covariance",
        )
        calibrated_covariance = _finite_array(
            np.asarray(archive["calibrated_coordinate_covariance_m2"]),
            ndim=5,
            label="backend source calibrated covariance",
        )
    if (
        names.shape != (8,)
        or backend.shape != (8, 498, 12, 3)
        or candidate.shape != backend.shape
        or raw_covariance.shape != (*backend.shape, 3)
        or calibrated_covariance.shape != raw_covariance.shape
        or not np.array_equal(candidate[:, :, :2], backend[:, :, :2])
        or not np.array_equal(candidate[:, :, -2:], backend[:, :, -2:])
        or not np.allclose(
            calibrated_covariance,
            raw_covariance * variance_scale,
            rtol=1e-12,
            atol=0.0,
        )
    ):
        raise ValueError("DEFORM backend source prediction archive differs")
    return {
        "schema_version": 1,
        "contract": "deform-dlo3-backend-artifact-verification-v1",
        "backend_target_arm_authorized": verification["backend_target_arm_authorized"],
        "selected_parameters": selected_parameters,
        "full_covariance_model": dict(
            _mapping(method.get("full_covariance_model"), label="backend model")
        ),
        "covariance_calibration": dict(
            _mapping(method.get("covariance_calibration"), label="backend calibration")
        ),
        "variance_scale": variance_scale,
        "prediction_seal_sha256": sha256_file(prediction_seal_path),
        "predictions_sha256": sha256_file(predictions_path),
        "verified": True,
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
