"""Leakage-safe source protocol helpers for the external DEFORM benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

DEFORM_DLO_SOURCE_SCHEMA_VERSION = 1
DEFORM_DLO_SOURCE_CONTRACT = "deform-dlo-source-reproduction-v1"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one external artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{label} must be a positive integer")
    return number


def load_deform_dlo_source_protocol(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the source-only DEFORM protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEFORM_DLO_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported DEFORM source protocol schema")
    if payload.get("contract") != DEFORM_DLO_SOURCE_CONTRACT:
        raise ValueError("unsupported DEFORM source protocol contract")

    upstream = _require_mapping(payload.get("upstream"), label="upstream")
    if upstream.get("repository") != "https://github.com/roahmlab/DEFORM":
        raise ValueError("DEFORM source protocol names an unexpected repository")
    commit = str(upstream.get("commit", ""))
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("DEFORM source protocol requires a full lowercase commit")
    if upstream.get("code_vendored") is not False:
        raise ValueError("external DEFORM code must not be vendored")

    data = _require_mapping(payload.get("data"), label="data")
    if data.get("development_partition") != "train":
        raise ValueError(
            "DEFORM development must use only the official train partition"
        )
    if data.get("official_eval_metrics_opened") is not False:
        raise ValueError("official DEFORM evaluation metrics must remain unopened")
    if data.get("forbid_eval_reads_during_source_stage") is not True:
        raise ValueError("DEFORM source stage must explicitly forbid eval reads")
    dlo_types = tuple(str(value) for value in data.get("dlo_types", ()))
    if not dlo_types or any(not value.startswith("DLO") for value in dlo_types):
        raise ValueError("DEFORM source protocol has invalid DLO types")
    expected_trajectories = _require_positive_int(
        data.get("expected_train_trajectories_per_dlo"),
        label="expected_train_trajectories_per_dlo",
    )

    split = _require_mapping(payload.get("source_split"), label="source_split")
    split_counts = tuple(
        _require_positive_int(split.get(name), label=f"source_split.{name}")
        for name in ("fit_count", "validation_count", "source_test_count")
    )
    if sum(split_counts) != expected_trajectories:
        raise ValueError("DEFORM source split does not cover every train trajectory")
    if not str(split.get("seed", "")):
        raise ValueError("DEFORM source split requires a seed")

    training = _require_mapping(payload.get("training"), label="training")
    horizon = _require_positive_int(
        training.get("unroll_horizon_frames"),
        label="training.unroll_horizon_frames",
    )
    if (
        horizon
        >= _require_positive_int(
            data.get("expected_frames_per_trajectory"),
            label="expected_frames_per_trajectory",
        )
        - 2
    ):
        raise ValueError("DEFORM training horizon leaves no valid source windows")
    total_updates = _require_positive_int(
        training.get("total_updates"), label="training.total_updates"
    )
    checkpoints = tuple(int(value) for value in training.get("checkpoint_updates", ()))
    if (
        not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != total_updates
        or tuple(sorted(set(checkpoints))) != checkpoints
    ):
        raise ValueError(
            "DEFORM checkpoint schedule must be unique, sorted, and complete"
        )
    _require_positive_int(training.get("batch_size"), label="training.batch_size")
    if training.get("known_action_nodes") != [0, 1, -2, -1]:
        raise ValueError("DEFORM known-action node contract changed")
    if training.get("optimizer") != "official-sgd-parameter-groups-v1":
        raise ValueError("DEFORM source reproduction must use official SGD groups")
    if training.get("cublas_workspace_config") != ":4096:8":
        raise ValueError("DEFORM source reproduction must bind deterministic cuBLAS")

    gate = _require_mapping(payload.get("source_gate"), label="source_gate")
    multiplier = float(gate.get("published_error_multiplier_max", math.nan))
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("DEFORM source gate has an invalid error multiplier")
    minimum_wins = _require_positive_int(
        gate.get("minimum_persistence_wins"),
        label="source_gate.minimum_persistence_wins",
    )
    if minimum_wins > split_counts[2]:
        raise ValueError("DEFORM source gate requires too many persistence wins")
    references = _require_mapping(
        gate.get("published_reference_l1_m"), label="published_reference_l1_m"
    )
    if any(
        not math.isfinite(float(references.get(dlo_type, math.nan)))
        or float(references[dlo_type]) <= 0.0
        for dlo_type in dlo_types
    ):
        raise ValueError("DEFORM source gate omits a positive published reference")

    result = dict(payload)
    result["protocol_path"] = str(source)
    result["dlo_types"] = dlo_types
    return result


def partition_deform_source_names(
    names: Sequence[str],
    *,
    seed: str,
    fit_count: int,
    validation_count: int,
    source_test_count: int,
) -> dict[str, tuple[str, ...]]:
    """Create a stable, exhaustive trajectory split without reading outcomes."""

    normalized = tuple(str(name) for name in names)
    if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("DEFORM trajectory names must be nonempty and unique")
    expected_count = fit_count + validation_count + source_test_count
    if len(normalized) != expected_count:
        raise ValueError(
            f"expected {expected_count} DEFORM trajectories, got {len(normalized)}"
        )

    def key(name: str) -> tuple[bytes, str]:
        payload = (
            b"deform-dlo-source-split-v1\0" + seed.encode() + b"\0" + name.encode()
        )
        return hashlib.sha256(payload).digest(), name

    ordered = tuple(sorted(normalized, key=key))
    fit_end = fit_count
    validation_end = fit_end + validation_count
    return {
        "fit": ordered[:fit_end],
        "validation": ordered[fit_end:validation_end],
        "source_test": ordered[validation_end:],
    }


def build_deform_dlo_source_manifest(
    protocol_path: str | Path,
    data_root: str | Path,
    *,
    dlo_type: str,
) -> dict[str, object]:
    """Bind all source trajectory bytes and their outcome-blind partition."""

    protocol = load_deform_dlo_source_protocol(protocol_path)
    if dlo_type not in protocol["dlo_types"]:
        raise ValueError(f"DLO type is outside the registered source stage: {dlo_type}")
    root = Path(data_root).resolve()
    train_root = root / dlo_type / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    paths = tuple(sorted(train_root.glob("*.pkl"), key=lambda path: path.name))
    expected = int(protocol["data"]["expected_train_trajectories_per_dlo"])
    if len(paths) != expected:
        raise ValueError(
            f"{dlo_type} expected {expected} train trajectories, got {len(paths)}"
        )
    split_config = protocol["source_split"]
    split = partition_deform_source_names(
        [path.name for path in paths],
        seed=str(split_config["seed"]),
        fit_count=int(split_config["fit_count"]),
        validation_count=int(split_config["validation_count"]),
        source_test_count=int(split_config["source_test_count"]),
    )
    identities = {
        path.name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    protocol_source = Path(str(protocol["protocol_path"]))
    return {
        "schema_version": DEFORM_DLO_SOURCE_SCHEMA_VERSION,
        "contract": DEFORM_DLO_SOURCE_CONTRACT,
        "dlo_type": dlo_type,
        "protocol": {
            "path": str(protocol_source),
            "sha256": sha256_file(protocol_source),
        },
        "partition": "train",
        "official_eval_read": False,
        "trajectories": identities,
        "split": {name: list(values) for name, values in split.items()},
    }


def deform_mean_coordinate_l1_m(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return DEFORM's published mean coordinate-wise L1 error in metres."""

    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(target, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim < 2:
        raise ValueError("DEFORM prediction and target shapes must agree")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("DEFORM metric inputs must be finite")
    return float(np.mean(np.abs(predicted - observed)))


def choose_deform_validation_checkpoint(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Choose the lowest validation error, breaking exact ties toward less training."""

    if not records:
        raise ValueError("DEFORM validation checkpoint records are empty")
    normalized = []
    seen_updates: set[int] = set()
    for record in records:
        update = int(record.get("update", -1))
        error = float(record.get("validation_l1_m", math.nan))
        if update < 0 or update in seen_updates or not math.isfinite(error):
            raise ValueError("DEFORM validation checkpoint record is invalid")
        seen_updates.add(update)
        normalized.append({**record, "update": update, "validation_l1_m": error})
    return min(
        normalized,
        key=lambda record: (record["validation_l1_m"], record["update"]),
    )


def evaluate_deform_source_gate(
    records: Sequence[Mapping[str, object]],
    *,
    published_reference_l1_m: float,
    published_error_multiplier_max: float,
    minimum_persistence_wins: int,
) -> dict[str, object]:
    """Evaluate the registered held-out source reproduction gate."""

    if not records:
        raise ValueError("DEFORM source gate requires held-out records")
    model_errors = []
    persistence_errors = []
    names: set[str] = set()
    for record in records:
        name = str(record.get("name", ""))
        model_error = float(record.get("model_l1_m", math.nan))
        persistence_error = float(record.get("persistence_l1_m", math.nan))
        if (
            not name
            or name in names
            or not math.isfinite(model_error)
            or not math.isfinite(persistence_error)
            or model_error < 0.0
            or persistence_error < 0.0
        ):
            raise ValueError("DEFORM source gate record is invalid")
        names.add(name)
        model_errors.append(model_error)
        persistence_errors.append(persistence_error)

    model_mean = float(np.mean(model_errors))
    persistence_mean = float(np.mean(persistence_errors))
    wins = sum(
        model_error < persistence_error
        for model_error, persistence_error in zip(
            model_errors, persistence_errors, strict=True
        )
    )
    threshold = float(published_reference_l1_m) * float(published_error_multiplier_max)
    parity_passed = model_mean <= threshold
    persistence_passed = wins >= minimum_persistence_wins
    return {
        "case_count": len(records),
        "model_mean_l1_m": model_mean,
        "persistence_mean_l1_m": persistence_mean,
        "persistence_wins": wins,
        "published_reference_l1_m": float(published_reference_l1_m),
        "published_error_threshold_l1_m": threshold,
        "parity_passed": parity_passed,
        "persistence_gate_passed": persistence_passed,
        "passed": parity_passed and persistence_passed,
    }
