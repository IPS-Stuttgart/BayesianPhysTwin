"""Prospective custody and source-split contracts for DEFORM DLO robustness."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

DEFORM_DLO_ROBUSTNESS_CONTRACT = "deform-dlo-robustness-v1"
DEFORM_DLO_ROBUSTNESS_DOMAIN = b"deform-dlo3-robustness-v1\0"


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
