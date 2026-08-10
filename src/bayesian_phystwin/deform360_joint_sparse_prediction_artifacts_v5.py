"""Prediction-first artifacts for the public Deform360 v5 source study.

The publisher receives an already constructed causal prediction problem and its
raw method result. It has no target or suffix argument. The complete forecast is
written to a deterministic NPZ archive and atomically sealed before any source
suffix is scored.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import GaugeAwareBeliefResult
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)
from .deform360_joint_sparse_prediction_v5 import (
    B0_PHYSICAL_FALLBACK,
    RAW_METHOD_IDS,
    T1_CONTACT_ONLY,
    V1_VISUAL_GUARDED,
    VT2_VISUOTACTILE_UNGUARDED,
    VT3_VISUOTACTILE_ANCHOR_BIAS,
    Deform360JointSparsePredictionInputV5,
    Deform360JointSparsePredictionResultV5,
    _array_sha256,
)

PREDICTION_ARCHIVE_FILENAME: Final = "prediction-arrays.npz"
PREDICTION_SEAL_FILENAME: Final = "prediction-seal.json"
PREDICTION_CHECKSUMS_FILENAME: Final = "SHA256SUMS"
PREDICTION_SEAL_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-prediction-seal"
)
PREDICTION_SEAL_VERSION: Final = 1
PREDICTION_SEAL_SEMANTICS: Final = (
    "causal-prefix-forecast-sealed-before-public-source-suffix-v1"
)
INFERENCE_METHOD_IDS: Final = (
    V1_VISUAL_GUARDED,
    T1_CONTACT_ONLY,
    VT2_VISUOTACTILE_UNGUARDED,
    VT3_VISUOTACTILE_ANCHOR_BIAS,
)
_RESULT_ARRAY_FIELDS: Final = (
    "state_coefficients",
    "gauge_delta",
    "shared_bias_coefficients",
    "view_bias_coefficients",
    "anchor_bias_coefficients",
    "posterior_covariance",
    "identifiable_state_transform",
    "identifiable_fractions",
    "query_sensitivity_fractions",
    "robust_weights",
    "anchor_robust_weights",
)
_ARCHIVE_FIELDS: Final = frozenset(
    {"path", "file_sha256", "byte_count", "array_sha256"}
)
_INFERENCE_FIELDS: Final = frozenset(
    {"inference_admissible", "reason", "diagnostics", "input_lineage"}
)
_SEAL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "prediction_seal_id",
        "execution_lock_id",
        "implementation_revision",
        "object_id",
        "episode_id",
        "stratum",
        "input_id",
        "result_id",
        "physical_mode",
        "factor_admitted",
        "causal_frame_stop",
        "evaluation_frame_range_half_open",
        "prediction_fit_artifact_id",
        "prediction_fit_object_ids",
        "risk_score",
        "predicted_loss_features_m",
        "method_artifact_ids",
        "inference",
        "diagnostics",
        "source_artifact_ids",
        "archive",
        "information_boundary",
    }
)
_INFORMATION_BOUNDARY: Final = {
    "confirmation_outcomes_opened": False,
    "confirmation_payloads_opened": False,
    "development_suffix_opened_before_prediction_seal": False,
    "future_object_observations_used_for_prediction": False,
    "human_approval_used": False,
    "new_measurements_collected": False,
    "public_released_prefix_measurements_used": True,
    "replacement_allowed": False,
    "target_outcomes_used": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be literal strings")
    return cast(Mapping[str, Any], value)


def _exact_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be Boolean")
    return cast(bool, value)


def _finite_nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _expected_array_names() -> frozenset[str]:
    return frozenset(
        {
            *(
                _array_member_name("trajectory", method_id)
                for method_id in RAW_METHOD_IDS
            ),
            *(
                _array_member_name("inference", method_id, field)
                for method_id in INFERENCE_METHOD_IDS
                for field in _RESULT_ARRAY_FIELDS
            ),
        }
    )


def _validate_checksums(root: Path) -> None:
    checksum_path = root / PREDICTION_CHECKSUMS_FILENAME
    names = (PREDICTION_ARCHIVE_FILENAME, PREDICTION_SEAL_FILENAME)
    expected = "".join(
        f"{_file_sha256(root / name)}  {name}\n" for name in sorted(names)
    )
    try:
        actual = checksum_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read prediction checksums") from error
    _require(actual == expected, "prediction SHA256SUMS changed")


def _array_member_name(kind: str, method_id: str, field: str | None = None) -> str:
    suffix = method_id if field is None else f"{method_id}__{field}"
    return f"{kind}__{suffix}"


def _result_arrays(
    result: Deform360JointSparsePredictionResultV5,
) -> dict[str, np.ndarray]:
    arrays = {
        _array_member_name("trajectory", method_id): np.asarray(
            result.trajectories_m[method_id]
        )
        for method_id in RAW_METHOD_IDS
    }
    for method_id in INFERENCE_METHOD_IDS:
        inference = result.inference_results[method_id]
        for field in _RESULT_ARRAY_FIELDS:
            arrays[_array_member_name("inference", method_id, field)] = np.asarray(
                getattr(inference, field)
            )
    return arrays


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        np.ascontiguousarray(value),
        version=(2, 0),
        allow_pickle=False,
    )
    return output.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for name, value in sorted(arrays.items()):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(value))
        raw.flush()
        os.fsync(raw.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                plain_json(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != PREDICTION_CHECKSUMS_FILENAME
    )
    checksums = root / PREDICTION_CHECKSUMS_FILENAME
    with checksums.open("w", encoding="ascii", newline="\n") as stream:
        for path in files:
            stream.write(f"{_file_sha256(path)}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_fit_object_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("prediction_fit_object_ids must be a sequence")
    result = tuple(
        sorted(nonempty_string(value, name="prediction fit object ID") for value in values)
    )
    _require(len(result) == len(set(result)), "prediction fit object IDs repeat")
    _require(all(value.strip() == value for value in result), "fit object ID is not canonical")
    return result


def _inference_descriptor(result: GaugeAwareBeliefResult) -> dict[str, Any]:
    return {
        "inference_admissible": result.inference_admissible,
        "reason": result.reason,
        "diagnostics": plain_json(result.diagnostics),
        "input_lineage": plain_json(result.input_lineage),
    }


def build_deform360_joint_sparse_prediction_seal_v5(
    problem: Deform360JointSparsePredictionInputV5,
    result: Deform360JointSparsePredictionResultV5,
    *,
    execution_lock_id: str,
    implementation_revision: str,
    prediction_fit_artifact_id: str,
    prediction_fit_object_ids: Sequence[str],
    archive_file_sha256: str,
    archive_byte_count: int,
) -> dict[str, Any]:
    """Build one content-addressed prediction seal without writing it."""

    if not isinstance(problem, Deform360JointSparsePredictionInputV5):
        raise TypeError("problem has changed type")
    if not isinstance(result, Deform360JointSparsePredictionResultV5):
        raise TypeError("result has changed type")
    _require(result.input_id == problem.input_id, "result belongs to another input")
    lock_id = sha256_digest(execution_lock_id, name="execution_lock_id")
    revision = exact_revision(implementation_revision, name="implementation_revision")
    fit_id = sha256_digest(
        prediction_fit_artifact_id,
        name="prediction_fit_artifact_id",
    )
    fit_objects = _canonical_fit_object_ids(prediction_fit_object_ids)
    archive_sha = sha256_digest(
        archive_file_sha256,
        name="archive_file_sha256",
    )
    _require(
        type(archive_byte_count) is int and archive_byte_count > 0,
        "archive_byte_count must be a positive integer",
    )
    arrays = _result_arrays(result)
    method_artifact_ids = {
        method_id: _array_sha256(result.trajectories_m[method_id])
        for method_id in RAW_METHOD_IDS
    }
    descriptor: dict[str, Any] = {
        "schema": PREDICTION_SEAL_SCHEMA,
        "schema_version": PREDICTION_SEAL_VERSION,
        "semantics": PREDICTION_SEAL_SEMANTICS,
        "execution_lock_id": lock_id,
        "implementation_revision": revision,
        "object_id": problem.object_id,
        "episode_id": problem.episode_id,
        "stratum": problem.stratum,
        "input_id": problem.input_id,
        "result_id": result.result_id,
        "physical_mode": problem.physical_mode,
        "factor_admitted": problem.factor_admitted,
        "causal_frame_stop": problem.causal_frame_stop,
        "evaluation_frame_range_half_open": list(
            problem.evaluation_frame_range_half_open
        ),
        "prediction_fit_artifact_id": fit_id,
        "prediction_fit_object_ids": list(fit_objects),
        "risk_score": result.risk_score,
        "predicted_loss_features_m": dict(result.predicted_loss_features_m),
        "method_artifact_ids": method_artifact_ids,
        "inference": {
            method_id: _inference_descriptor(result.inference_results[method_id])
            for method_id in INFERENCE_METHOD_IDS
        },
        "diagnostics": plain_json(result.diagnostics),
        "source_artifact_ids": dict(problem.source_artifact_ids),
        "archive": {
            "path": PREDICTION_ARCHIVE_FILENAME,
            "file_sha256": archive_sha,
            "byte_count": archive_byte_count,
            "array_sha256": {
                name: _array_sha256(value) for name, value in sorted(arrays.items())
            },
        },
        "information_boundary": dict(_INFORMATION_BOUNDARY),
    }
    return {"prediction_seal_id": content_id(descriptor), **descriptor}


def publish_deform360_joint_sparse_prediction_v5(
    problem: Deform360JointSparsePredictionInputV5,
    result: Deform360JointSparsePredictionResultV5,
    output_directory: str | Path,
    *,
    execution_lock_id: str,
    implementation_revision: str,
    prediction_fit_artifact_id: str,
    prediction_fit_object_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Atomically publish one no-overwrite source prediction artifact."""

    destination = Path(output_directory).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not os.path.lexists(destination), "prediction output already exists")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        archive = temporary / PREDICTION_ARCHIVE_FILENAME
        _write_deterministic_npz(archive, _result_arrays(result))
        seal = build_deform360_joint_sparse_prediction_seal_v5(
            problem,
            result,
            execution_lock_id=execution_lock_id,
            implementation_revision=implementation_revision,
            prediction_fit_artifact_id=prediction_fit_artifact_id,
            prediction_fit_object_ids=prediction_fit_object_ids,
            archive_file_sha256=_file_sha256(archive),
            archive_byte_count=archive.stat().st_size,
        )
        _write_json(temporary / PREDICTION_SEAL_FILENAME, seal)
        _write_checksums(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_deform360_joint_sparse_prediction_v5(destination)[0]


def _load_arrays(path: Path, declared: Mapping[str, Any]) -> dict[str, np.ndarray]:
    _require(
        set(declared) == _expected_array_names(),
        "declared prediction array roster changed",
    )
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("cannot inspect prediction archive") from error
    expected_members = [f"{name}.npy" for name in sorted(declared)]
    _require(
        [member.filename for member in members] == expected_members,
        "prediction ZIP member roster changed",
    )
    for member in members:
        _require(
            member.compress_type == zipfile.ZIP_STORED
            and member.date_time == (1980, 1, 1, 0, 0, 0)
            and not member.is_dir()
            and not bool(member.flag_bits & 0x1),
            f"prediction ZIP member {member.filename} changed format",
        )
    try:
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except (OSError, ValueError) as error:
        raise ValueError("cannot load prediction archive") from error
    expected = set(declared)
    _require(set(arrays) == expected, "prediction archive member roster changed")
    for name, value in arrays.items():
        _require(value.dtype.kind in "fiu", f"prediction array {name} is not numeric")
        _require(np.all(np.isfinite(value)), f"prediction array {name} is non-finite")
        _require(
            _array_sha256(value)
            == sha256_digest(declared[name], name=f"array SHA-256 {name}"),
            f"prediction array {name} changed",
        )
    return arrays


def _result_from_seal_and_arrays(
    seal: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Deform360JointSparsePredictionResultV5:
    inference_payload = cast(Mapping[str, Any], seal["inference"])
    inference: dict[str, GaugeAwareBeliefResult] = {}
    for method_id in INFERENCE_METHOD_IDS:
        item = cast(Mapping[str, Any], inference_payload[method_id])
        inference[method_id] = GaugeAwareBeliefResult(
            inference_admissible=_exact_boolean(
                item["inference_admissible"],
                name=f"{method_id} inference_admissible",
            ),
            reason=nonempty_string(item["reason"], name=f"{method_id} reason"),
            state_coefficients=arrays[
                _array_member_name("inference", method_id, "state_coefficients")
            ],
            gauge_delta=arrays[
                _array_member_name("inference", method_id, "gauge_delta")
            ],
            shared_bias_coefficients=arrays[
                _array_member_name(
                    "inference", method_id, "shared_bias_coefficients"
                )
            ],
            view_bias_coefficients=arrays[
                _array_member_name("inference", method_id, "view_bias_coefficients")
            ],
            anchor_bias_coefficients=arrays[
                _array_member_name("inference", method_id, "anchor_bias_coefficients")
            ],
            posterior_covariance=arrays[
                _array_member_name("inference", method_id, "posterior_covariance")
            ],
            identifiable_state_transform=arrays[
                _array_member_name(
                    "inference", method_id, "identifiable_state_transform"
                )
            ],
            identifiable_fractions=arrays[
                _array_member_name("inference", method_id, "identifiable_fractions")
            ],
            query_sensitivity_fractions=arrays[
                _array_member_name(
                    "inference", method_id, "query_sensitivity_fractions"
                )
            ],
            robust_weights=arrays[
                _array_member_name("inference", method_id, "robust_weights")
            ],
            anchor_robust_weights=arrays[
                _array_member_name("inference", method_id, "anchor_robust_weights")
            ],
            diagnostics=cast(Mapping[str, Any], item["diagnostics"]),
            input_lineage=cast(Mapping[str, Any], item["input_lineage"]),
        )
    return Deform360JointSparsePredictionResultV5(
        input_id=cast(str, seal["input_id"]),
        trajectories_m={
            method_id: arrays[_array_member_name("trajectory", method_id)]
            for method_id in RAW_METHOD_IDS
        },
        inference_results=inference,
        risk_score=float(seal["risk_score"]),
        predicted_loss_features_m=cast(
            Mapping[str, float], seal["predicted_loss_features_m"]
        ),
        diagnostics=cast(Mapping[str, Any], seal["diagnostics"]),
    )


def load_deform360_joint_sparse_prediction_v5(
    directory: str | Path,
) -> tuple[Mapping[str, Any], Deform360JointSparsePredictionResultV5]:
    """Load and fully revalidate one published prediction artifact."""

    requested_root = Path(directory).absolute()
    _require(
        requested_root.exists()
        and requested_root.is_dir()
        and not requested_root.is_symlink(),
        "prediction root is invalid",
    )
    root = requested_root.resolve(strict=True)
    seal_path = root / PREDICTION_SEAL_FILENAME
    archive_path = root / PREDICTION_ARCHIVE_FILENAME
    checksums_path = root / PREDICTION_CHECKSUMS_FILENAME
    expected_files = {
        PREDICTION_ARCHIVE_FILENAME,
        PREDICTION_SEAL_FILENAME,
        PREDICTION_CHECKSUMS_FILENAME,
    }
    _require(
        {path.name for path in root.iterdir()} == expected_files
        and all(
            path.is_file() and not path.is_symlink()
            for path in (seal_path, archive_path, checksums_path)
        ),
        "prediction artifact is incomplete",
    )
    seal = load_strict_json_object(seal_path, label="v5 prediction seal")
    require_exact_fields(seal, expected=_SEAL_FIELDS, name="prediction seal")
    _require(seal.get("schema") == PREDICTION_SEAL_SCHEMA, "seal schema changed")
    _require(
        seal.get("schema_version") == PREDICTION_SEAL_VERSION,
        "seal version changed",
    )
    _require(
        seal.get("semantics") == PREDICTION_SEAL_SEMANTICS,
        "seal semantics changed",
    )
    declared_seal_id = sha256_digest(
        seal.get("prediction_seal_id"),
        name="prediction_seal_id",
    )
    body = {key: value for key, value in seal.items() if key != "prediction_seal_id"}
    _require(declared_seal_id == content_id(body), "prediction seal ID changed")
    _require(
        seal.get("information_boundary") == _INFORMATION_BOUNDARY,
        "prediction seal crossed its information boundary",
    )
    sha256_digest(seal.get("execution_lock_id"), name="execution_lock_id")
    exact_revision(seal.get("implementation_revision"), name="implementation_revision")
    object_id = nonempty_string(seal.get("object_id"), name="object_id")
    _require(
        object_id.strip() == object_id and "\x00" not in object_id,
        "object_id is not canonical",
    )
    episode_id = seal.get("episode_id")
    _require(
        type(episode_id) is int and episode_id >= 0,
        "episode_id must be a nonnegative integer",
    )
    _require(seal.get("stratum") in {"sheet", "volumetric"}, "stratum changed")
    sha256_digest(seal.get("input_id"), name="input_id")
    sha256_digest(seal.get("result_id"), name="result_id")
    _require(
        seal.get("physical_mode") in {"warp_twin", "persistence_fallback"},
        "physical_mode changed",
    )
    _exact_boolean(seal.get("factor_admitted"), name="factor_admitted")
    causal_stop = seal.get("causal_frame_stop")
    _require(
        type(causal_stop) is int and causal_stop >= 1,
        "causal_frame_stop must be a positive integer",
    )
    evaluation_range = seal.get("evaluation_frame_range_half_open")
    _require(
        type(evaluation_range) is list
        and len(evaluation_range) == 2
        and all(type(value) is int for value in evaluation_range)
        and evaluation_range[0] == causal_stop
        and evaluation_range[0] < evaluation_range[1],
        "evaluation frame range changed",
    )
    sha256_digest(
        seal.get("prediction_fit_artifact_id"),
        name="prediction_fit_artifact_id",
    )
    fit_object_ids = seal.get("prediction_fit_object_ids")
    _require(
        type(fit_object_ids) is list and bool(fit_object_ids),
        "prediction_fit_object_ids must be a nonempty list",
    )
    _require(
        list(_canonical_fit_object_ids(cast(Sequence[str], fit_object_ids)))
        == fit_object_ids,
        "prediction_fit_object_ids changed",
    )
    _finite_nonnegative_float(seal.get("risk_score"), name="risk_score")
    features = _mapping(
        seal.get("predicted_loss_features_m"),
        name="predicted_loss_features_m",
    )
    _require(set(features) == set(RAW_METHOD_IDS), "predicted-loss roster changed")
    for method_id in RAW_METHOD_IDS:
        _finite_nonnegative_float(
            features[method_id],
            name=f"predicted loss {method_id}",
        )
    method_ids = _mapping(
        seal.get("method_artifact_ids"), name="method_artifact_ids"
    )
    _require(set(method_ids) == set(RAW_METHOD_IDS), "method artifact roster changed")
    for method_id in RAW_METHOD_IDS:
        sha256_digest(method_ids[method_id], name=f"method artifact {method_id}")
    inference_payload = _mapping(seal.get("inference"), name="inference")
    _require(
        set(inference_payload) == set(INFERENCE_METHOD_IDS),
        "inference roster changed",
    )
    for method_id in INFERENCE_METHOD_IDS:
        item = _mapping(inference_payload[method_id], name=f"inference {method_id}")
        require_exact_fields(
            item,
            expected=_INFERENCE_FIELDS,
            name=f"inference {method_id}",
        )
        _exact_boolean(
            item["inference_admissible"],
            name=f"{method_id} inference_admissible",
        )
        nonempty_string(item["reason"], name=f"{method_id} reason")
        frozen_finite_json_mapping(
            _mapping(item["diagnostics"], name=f"{method_id} diagnostics"),
            name=f"{method_id} diagnostics",
        )
        frozen_finite_json_mapping(
            _mapping(item["input_lineage"], name=f"{method_id} input_lineage"),
            name=f"{method_id} input_lineage",
        )
    diagnostics = frozen_finite_json_mapping(
        _mapping(seal.get("diagnostics"), name="diagnostics"),
        name="diagnostics",
    )
    source_artifact_mapping(
        _mapping(seal.get("source_artifact_ids"), name="source_artifact_ids"),
        name="source_artifact_ids",
    )
    archive = _mapping(seal.get("archive"), name="archive")
    require_exact_fields(archive, expected=_ARCHIVE_FIELDS, name="archive")
    _require(
        archive.get("path") == PREDICTION_ARCHIVE_FILENAME,
        "prediction archive path changed",
    )
    byte_count = archive.get("byte_count")
    _require(
        type(byte_count) is int and byte_count > 0,
        "prediction archive byte count changed",
    )
    _require(
        _file_sha256(archive_path)
        == sha256_digest(archive.get("file_sha256"), name="archive file SHA-256")
        and archive_path.stat().st_size == byte_count,
        "prediction archive bytes changed",
    )
    declared_arrays = _mapping(archive.get("array_sha256"), name="array_sha256")
    arrays = _load_arrays(archive_path, declared_arrays)
    result = _result_from_seal_and_arrays(seal, arrays)
    _require(result.result_id == seal.get("result_id"), "prediction result ID changed")
    for method_id in RAW_METHOD_IDS:
        _require(
            method_ids[method_id] == _array_sha256(result.trajectories_m[method_id]),
            f"method artifact {method_id} changed",
        )
    inference_diagnostics = _mapping(
        diagnostics.get("inference"), name="diagnostics inference"
    )
    _require(
        set(inference_diagnostics) == set(INFERENCE_METHOD_IDS),
        "diagnostic inference roster changed",
    )
    for method_id in INFERENCE_METHOD_IDS:
        item = _mapping(
            inference_diagnostics[method_id],
            name=f"diagnostics inference {method_id}",
        )
        if "exact_fallback" in item:
            _exact_boolean(
                item["exact_fallback"],
                name=f"{method_id} exact_fallback",
            )
        if item.get("exact_fallback") is True:
            _require(
                method_ids[method_id] == method_ids[B0_PHYSICAL_FALLBACK],
                f"{method_id} does not preserve exact B0 fallback",
            )
    _validate_checksums(root)
    return seal, result


__all__ = [
    "PREDICTION_ARCHIVE_FILENAME",
    "PREDICTION_CHECKSUMS_FILENAME",
    "PREDICTION_SEAL_FILENAME",
    "build_deform360_joint_sparse_prediction_seal_v5",
    "load_deform360_joint_sparse_prediction_v5",
    "publish_deform360_joint_sparse_prediction_v5",
]
