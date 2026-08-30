from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
    fit_deform_local_residual,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    augment_deform_local_residual_full_covariance,
    build_deform_bayesian_covariance_ablation_v1,
    calibrate_deform_full_covariance,
    deform_bayesian_covariance_archive_key,
    evaluate_deform_predictive_distribution,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file
from bayesian_phystwin_experiments.deform_dlo_upstream import (
    load_deform_dlo_initialization,
)

Array = np.ndarray[Any, Any]
DLOS = ("DLO4", "DLO5")
INTERNAL = slice(2, -2)

__all__ = [
    "DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS",
    "DLOS",
    "INTERNAL",
    "Any",
    "Array",
    "Mapping",
    "Path",
    "Sequence",
    "_assert_upstream_and_initialization",
    "_file_manifest",
    "_identity",
    "_load_named_from_manifest",
    "_load_paths",
    "_load_protocol",
    "_mapping",
    "_parse_args",
    "_partition_names",
    "_paths",
    "_protocol_part",
    "_read_json",
    "_setup_torch",
    "_verified_file",
    "_write_json",
    "argparse",
    "augment_deform_local_residual_full_covariance",
    "build_deform_bayesian_covariance_ablation_v1",
    "calibrate_deform_full_covariance",
    "cast",
    "deform_bayesian_covariance_archive_key",
    "deserialize_deform_local_residual_model",
    "evaluate_deform_predictive_distribution",
    "fit_deform_local_residual",
    "hashlib",
    "json",
    "load_deform_dlo_initialization",
    "local_runtime",
    "math",
    "np",
    "os",
    "posterior_runtime",
    "serialize_deform_local_residual_model",
    "sha256_file",
    "source_runtime",
    "sys",
    "time",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("inventory", "source", "authorize", "predict", "seal", "score"),
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dlo", choices=DLOS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-result-dlo4", type=Path)
    parser.add_argument("--source-result-dlo5", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--prediction-seal-dlo4", type=Path)
    parser.add_argument("--prediction-seal-dlo5", type=Path)
    parser.add_argument("--joint-seal", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    immutable: bool = True,
) -> None:
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if immutable and path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        if immutable:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path, **extra: object) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        **extra,
    }


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity changed")
    return path


def _load_protocol(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != "deform-dlo45-frozen-transfer-v1"
        or payload.get("prob4d_used") is not False
    ):
        raise ValueError("unsupported DLO4/DLO5 protocol")
    parent = _mapping(payload.get("parent_method"), label="parent method")
    upstream = _mapping(payload.get("upstream"), label="upstream")
    data = _mapping(payload.get("data"), label="data")
    split = _mapping(payload.get("source_split"), label="source split")
    training = _mapping(payload.get("physical_training"), label="physical training")
    residual = _mapping(payload.get("local_residual"), label="local residual")
    source_gate = _mapping(payload.get("source_gate"), label="source gate")
    target = _mapping(payload.get("target_evaluation"), label="target evaluation")
    custody = _mapping(payload.get("custody"), label="custody")
    if (
        parent.get("dlo3_method_revision") != "da487c26a0ef1c5c6c4629f6cc32b0964728ad2a"
        or parent.get("dlo3_protocol_sha256")
        != "ef4533e7adcf317ccf0fbe951af2870bf86096ca7cf9bf1d777a84963506c35c"
        or parent.get("point_arm") != "r1_s0p25"
        or upstream.get("repository") != "https://github.com/roahmlab/DEFORM"
        or upstream.get("commit") != "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
        or upstream.get("train_script_sha256")
        != "d45abe23a22b0f01fa266833844c4f9b71a2b7e375f8e955e3278b9e969acc55"
        or tuple(data.get("dlos", ())) != DLOS
        or int(cast(Any, data.get("train_trajectory_count", -1))) != 56
        or int(cast(Any, data.get("eval_trajectory_count", -1))) != 14
        or int(cast(Any, data.get("frame_count", -1))) != 500
        or int(cast(Any, data.get("node_count", -1))) != 12
        or dict(
            _mapping(
                data.get("coordinate_transform_by_dlo"),
                label="coordinate transforms",
            )
        )
        != {
            "DLO4": "raw-x-raw-z-negated-raw-y",
            "DLO5": "raw-x-raw-z-negated-raw-y",
        }
        or tuple(
            int(v) for v in cast(Sequence[Any], data.get("known_action_nodes", ()))
        )
        != (0, 1, -2, -1)
        or int(cast(Any, split.get("fit_count", -1))) != 39
        or int(cast(Any, split.get("calibration_count", -1))) != 9
        or int(cast(Any, split.get("source_test_count", -1))) != 8
        or training.get("backend") != "official-DEFORM-PBD"
        or int(cast(Any, training.get("seed", -1))) != 42
        or int(cast(Any, training.get("unroll_horizon_frames", -1))) != 50
        or int(cast(Any, training.get("batch_size", -1))) != 32
        or int(cast(Any, training.get("total_updates", -1))) != 6400
        or int(cast(Any, training.get("maximum_compute_matched_updates", -1))) != 512
        or int(cast(Any, training.get("pbd_iterations", -1))) != 10
        or training.get("optimizer") != "official-sgd-parameter-groups-v1"
        or training.get("cublas_workspace_config") != ":4096:8"
        or float(cast(Any, residual.get("ridge", math.nan))) != 1.0
        or float(cast(Any, residual.get("shrinkage", math.nan))) != 0.25
        or float(cast(Any, residual.get("coordinate_variance_floor_m2", math.nan)))
        != 1e-6
        or source_gate.get("required_before_target") is not True
        or float(cast(Any, source_gate.get("minimum_relative_improvement", math.nan)))
        != 0.01
        or int(cast(Any, source_gate.get("minimum_case_wins", -1))) != 6
        or float(cast(Any, source_gate.get("maximum_case_ratio", math.nan))) != 1.10
        or target.get("joint_prediction_seal_before_scoring") is not True
        or target.get("one_shot") is not True
        or target.get("target_selection") is not False
        or target.get("target_calibration") is not False
        or target.get("target_retries") is not False
        or target.get("case_replacement") is not False
        or tuple(target.get("primary_datasets", ())) != DLOS
        or custody.get("dlo4_and_dlo5_scored_only_after_joint_seal") is not True
        or custody.get("raw_predictions_uploaded") is not False
    ):
        raise ValueError("DLO4/DLO5 frozen protocol differs")
    return payload


def _protocol_part(protocol: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _mapping(protocol.get(name), label=name.replace("_", " "))


def _partition_names(
    names: Sequence[str],
    *,
    dlo: str,
    protocol: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    split = _protocol_part(protocol, "source_split")
    normalized = tuple(str(name) for name in names)
    if (
        len(normalized) != 56
        or len(set(normalized)) != 56
        or any(not name for name in normalized)
    ):
        raise ValueError(f"{dlo} train roster must contain 56 unique names")
    domain = str(split["domain_separator"]).encode("utf-8")

    def key(name: str) -> tuple[bytes, str]:
        payload = domain + b"\0" + dlo.encode() + b"\0" + name.encode()
        return hashlib.sha256(payload).digest(), name

    ordered = tuple(sorted(normalized, key=key))
    fit_count = int(cast(Any, split["fit_count"]))
    calibration_count = int(cast(Any, split["calibration_count"]))
    return {
        "fit": ordered[:fit_count],
        "calibration": ordered[fit_count : fit_count + calibration_count],
        "source_test": ordered[fit_count + calibration_count :],
    }


def _paths(root: Path, dlo: str, partition: str) -> tuple[Path, ...]:
    paths = tuple(sorted((root / dlo / partition).glob("*.pkl"), key=lambda p: p.name))
    expected = 56 if partition == "train" else 14
    if len(paths) != expected or any(
        not path.is_file() or path.stat().st_size <= 0 for path in paths
    ):
        raise ValueError(
            f"{dlo}/{partition} expected {expected} nonempty trajectories, "
            f"got {len(paths)}"
        )
    return paths


def _file_manifest(paths: Sequence[Path], *, hash_payload: bool) -> dict[str, object]:
    identities: dict[str, object] = {}
    for path in paths:
        record: dict[str, object] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        if hash_payload:
            record["sha256"] = sha256_file(path)
        identities[path.name] = record
    return identities


def _load_paths(
    paths: Sequence[Path],
    *,
    frame_count: int,
    node_count: int,
) -> dict[str, Array]:
    return {
        path.name: source_runtime._load_trajectory(
            path,
            frame_count=frame_count,
            node_count=node_count,
        )
        for path in paths
    }


def _load_named_from_manifest(
    manifest: Mapping[str, object],
    names: Sequence[str],
    *,
    frame_count: int,
    node_count: int,
) -> dict[str, Array]:
    identities = _mapping(manifest.get("trajectories"), label="manifest trajectories")
    result: dict[str, Array] = {}
    for name in names:
        identity = _mapping(identities.get(name), label=f"trajectory {name}")
        path = Path(str(identity.get("path", ""))).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
            or sha256_file(path) != identity.get("sha256")
        ):
            raise RuntimeError(f"trajectory changed after manifesting: {name}")
        result[name] = source_runtime._load_trajectory(
            path,
            frame_count=frame_count,
            node_count=node_count,
        )
    return result


def _assert_upstream_and_initialization(
    protocol: Mapping[str, object],
    upstream_root: Path,
    dlo: str,
) -> dict[str, object]:
    upstream = _protocol_part(protocol, "upstream")
    record = source_runtime._assert_upstream(upstream_root, str(upstream["commit"]))
    initialization = load_deform_dlo_initialization(
        upstream_root.resolve() / "train_DEFORM.py",
        dlo,
    )
    data = _protocol_part(protocol, "data")
    transforms = _mapping(
        data.get("coordinate_transform_by_dlo"), label="coordinate transforms"
    )
    if (
        initialization.node_count != int(cast(Any, data["node_count"]))
        or initialization.coordinate_transform != transforms.get(dlo)
        or initialization.source_sha256 != upstream["train_script_sha256"]
    ):
        raise RuntimeError(f"{dlo} initialization differs from the frozen upstream")
    return {
        "upstream": record,
        "initialization": initialization.to_record(),
    }


def _setup_torch(protocol: Mapping[str, object], device: str) -> Any:
    training = _protocol_part(protocol, "physical_training")
    expected = str(training["cublas_workspace_config"])
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, expected):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("registered DLO4/DLO5 execution requires CUDA")
    index = int(device.split(":", maxsplit=1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {device}")
    return torch
