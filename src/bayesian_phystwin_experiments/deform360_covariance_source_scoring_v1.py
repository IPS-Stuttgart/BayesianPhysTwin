"""Source-only scorer for the frozen Deform360 covariance evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin.covariance_only_hybrid_analysis import (
    score_scale_grid,
    score_zero_covariance,
)
from bayesian_phystwin.deform360_covariance_source_producer_v1 import (
    validate_covariance_source_panel_v1,
)
from bayesian_phystwin.strict_json_report_io import load_strict_json_mapping

from .deform360_covariance_only_source_gate_v1 import (
    COVARIANCE_EIGENVALUE_FLOOR_M2,
    OBSERVATION_STD_M,
    PAPER_PROTOCOL_ID,
    SCORES_SCHEMA,
    SOFTWARE_PROTOCOL_ID,
    SOURCE_ROSTER,
    evaluate_source_gate,
    seal_source_scores,
    validate_prediction_batch,
)

SCHEMA_VERSION: Final = 1
OBSERVATIONS_SCHEMA: Final = (
    "bayesian-phystwin.deform360-covariance-source-observations-v1"
)
RECEIPT_SCHEMA: Final = "bayesian-phystwin.deform360-covariance-source-score-receipt-v1"
MARGINAL_COVERAGE_Z: Final = 1.6448536269514722
HORIZONS: Final = (("early", 0, 6), ("middle", 6, 12), ("late", 12, 18))

_OBSERVATION_BOUNDARY: Final = {
    "source_suffix_opened": True,
    "confirmation_payloads_opened": False,
    "confirmation_predictions_run": False,
    "confirmation_outcomes_used": False,
    "target_informed_selection_used": False,
    "replacement_used": False,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _plain_json(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded)


def _content_id(value: Mapping[str, Any], identity_field: str) -> str:
    document = dict(_plain_json(value))
    document.pop(identity_field, None)
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[Any], value)


def _literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty string")
    return cast(str, value)


def _sha256(value: object, *, name: str) -> str:
    result = _literal(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return result


def _revision(value: object, *, name: str) -> str:
    result = _literal(value, name=name)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase Git SHA-1 string")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _canonical_relative_path(value: object, *, name: str) -> str:
    result = _literal(value, name=name)
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or "\\" in result
        or path.as_posix() != result
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    if any(
        part.lower() in {"confirmation", "target", "held-v8"} for part in path.parts
    ):
        raise ValueError(f"{name} enters a forbidden target path")
    return result


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_dir() and not requested.is_symlink(), f"invalid {name}")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical directory")
    return resolved


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(requested.is_file() and not requested.is_symlink(), f"invalid {name}")
    resolved = requested.resolve(strict=True)
    _require(resolved == requested, f"{name} must be a canonical file")
    return resolved


def _resolve_file(root: Path, relative: str, *, name: str) -> Path:
    requested = root.joinpath(*PurePosixPath(relative).parts)
    _require(requested.is_file() and not requested.is_symlink(), f"invalid {name}")
    resolved = requested.resolve(strict=True)
    _require(
        resolved == requested and (root == resolved or root in resolved.parents),
        f"{name} escapes its admitted root",
    )
    return resolved


def _require_disjoint(left: Path, right: Path, *, message: str) -> None:
    _require(
        left != right and right not in left.parents and left not in right.parents,
        message,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    value, _ = load_strict_json_mapping(path, artifact_label=label)
    return dict(value)


def _validate_array_descriptor(
    descriptor: object,
    *,
    name: str,
    expected_dtype: str,
    expected_shape: Sequence[int],
    expected_units: str,
) -> Mapping[str, Any]:
    row = _mapping(descriptor, name=name)
    expected = {"member", "dtype", "shape", "sha256", "units"}
    if set(row) != expected:
        raise ValueError(f"{name} fields changed")
    if row.get("dtype") != expected_dtype or list(row.get("shape", ())) != list(
        expected_shape
    ):
        raise ValueError(f"{name} representation changed")
    _literal(row.get("member"), name=f"{name}.member")
    _sha256(row.get("sha256"), name=f"{name}.sha256")
    if row.get("units") != expected_units:
        raise ValueError(f"{name} units changed")
    return row


def seal_source_observations_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address a complete source-observation manifest."""

    document = dict(_plain_json(payload))
    document.pop("observation_set_id", None)
    document["observation_set_id"] = _content_id(document, "observation_set_id")
    validate_source_observations_v1(document)
    return document


def validate_source_observations_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source-only outcome custody without loading array values."""

    document = dict(_plain_json(_mapping(payload, name="source observations")))
    expected_root = {
        "schema",
        "schema_version",
        "software_protocol_id",
        "paper_protocol_id",
        "prediction_batch_id",
        "scoring_implementation_revision",
        "information_boundary",
        "rows",
        "observation_set_id",
    }
    if set(document) != expected_root:
        raise ValueError("source-observation fields changed")
    if (
        document.get("schema") != OBSERVATIONS_SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("source-observation schema changed")
    if (
        document.get("software_protocol_id") != SOFTWARE_PROTOCOL_ID
        or document.get("paper_protocol_id") != PAPER_PROTOCOL_ID
    ):
        raise ValueError("source-observation protocol identity changed")
    _sha256(document.get("prediction_batch_id"), name="prediction_batch_id")
    _revision(
        document.get("scoring_implementation_revision"),
        name="scoring_implementation_revision",
    )
    declared = _sha256(document.get("observation_set_id"), name="observation_set_id")
    if declared != _content_id(document, "observation_set_id"):
        raise ValueError("observation_set_id does not match document content")
    boundary = _mapping(
        document.get("information_boundary"), name="information_boundary"
    )
    if dict(boundary) != _OBSERVATION_BOUNDARY:
        raise ValueError("source-observation information boundary changed")

    rows = _sequence(document.get("rows"), name="rows")
    if len(rows) != len(SOURCE_ROSTER):
        raise ValueError("source observations must retain all ten source units")
    for index, (raw, expected_unit) in enumerate(zip(rows, SOURCE_ROSTER, strict=True)):
        row = _mapping(raw, name=f"rows[{index}]")
        expected_fields = {
            "object_id",
            "episode",
            "stratum",
            "prediction_id",
            "unit_manifest_id",
            "scoring_reconstruction_id",
            "scoring_configuration_id",
            "scoring_camera_family_ids",
            "scoring_plan_artifact_ids",
            "source_suffix_input_artifacts",
            "future_range_half_open",
            "disposition",
            "technical_failure_reason",
            "artifact",
            "arrays",
        }
        if set(row) != expected_fields:
            raise ValueError(f"source-observation row {index} fields changed")
        if (
            row.get("object_id"),
            row.get("episode"),
            row.get("stratum"),
        ) != expected_unit:
            raise ValueError("source-observation roster or order changed")
        for field in (
            "prediction_id",
            "unit_manifest_id",
            "scoring_reconstruction_id",
            "scoring_configuration_id",
        ):
            _sha256(row.get(field), name=f"rows[{index}].{field}")
        cameras = tuple(
            _literal(item, name="scoring camera")
            for item in _sequence(
                row.get("scoring_camera_family_ids"), name="scoring_camera_family_ids"
            )
        )
        if len(cameras) != 2 or cameras != tuple(sorted(set(cameras))):
            raise ValueError("scoring camera family roster changed")
        plan_ids = tuple(
            _sha256(item, name="scoring plan artifact")
            for item in _sequence(
                row.get("scoring_plan_artifact_ids"), name="scoring_plan_artifact_ids"
            )
        )
        if len(plan_ids) != 2 or plan_ids != tuple(sorted(set(plan_ids))):
            raise ValueError("scoring plan artifact roster changed")
        inputs = _sequence(
            row.get("source_suffix_input_artifacts"),
            name="source_suffix_input_artifacts",
        )
        if not inputs:
            raise ValueError("source suffix input artifact roster is empty")
        roles: list[str] = []
        for raw_artifact in inputs:
            artifact = _mapping(raw_artifact, name="source suffix input artifact")
            if set(artifact) != {"role", "path", "sha256", "size_bytes"}:
                raise ValueError("source suffix input artifact fields changed")
            roles.append(_literal(artifact.get("role"), name="source suffix role"))
            _canonical_relative_path(artifact.get("path"), name="source suffix path")
            _sha256(artifact.get("sha256"), name="source suffix SHA-256")
            if _integer(artifact.get("size_bytes"), name="source suffix size") < 0:
                raise ValueError("source suffix size must be nonnegative")
        if roles != sorted(set(roles)):
            raise ValueError(
                "source suffix input artifacts are duplicated or unordered"
            )
        if list(row.get("future_range_half_open", ())) != [58, 76]:
            raise ValueError("source-observation future range changed")
        disposition = row.get("disposition")
        if disposition not in {"observed", "technical_failure"}:
            raise ValueError("source-observation disposition changed")
        if disposition == "technical_failure":
            _literal(
                row.get("technical_failure_reason"), name="technical_failure_reason"
            )
            if row.get("artifact") is not None or row.get("arrays") is not None:
                raise ValueError("technical source observation carries invented arrays")
            continue
        if row.get("technical_failure_reason") is not None:
            raise ValueError("observed source row has a technical failure reason")
        artifact = _mapping(row.get("artifact"), name="source observation artifact")
        if set(artifact) != {"path", "sha256", "size_bytes"}:
            raise ValueError("source observation artifact fields changed")
        _canonical_relative_path(artifact.get("path"), name="source observation path")
        _sha256(artifact.get("sha256"), name="source observation artifact SHA-256")
        if (
            _integer(
                artifact.get("size_bytes"), name="source observation artifact size"
            )
            <= 0
        ):
            raise ValueError("source observation artifact is empty")
        arrays = _mapping(row.get("arrays"), name="source observation arrays")
        if set(arrays) != {"observation_m", "valid"}:
            raise ValueError("source observation array roster changed")
        observation = _mapping(arrays["observation_m"], name="observation_m descriptor")
        valid = _mapping(arrays["valid"], name="valid descriptor")
        if observation.get("dtype") != "<f8" or valid.get("dtype") != "|b1":
            raise ValueError("source observation dtype changed")
        observation_shape = list(
            _sequence(observation.get("shape"), name="observation shape")
        )
        valid_shape = list(_sequence(valid.get("shape"), name="valid shape"))
        if (
            len(observation_shape) != 3
            or observation_shape[0] != 18
            or observation_shape[2] != 3
            or valid_shape != observation_shape[:2]
        ):
            raise ValueError("source observation shape changed")
        _validate_array_descriptor(
            observation,
            name="observation_m",
            expected_dtype="<f8",
            expected_shape=observation_shape,
            expected_units="m",
        )
        _validate_array_descriptor(
            valid,
            name="valid",
            expected_dtype="|b1",
            expected_shape=valid_shape,
            expected_units="dimensionless",
        )
    return document


def _load_prediction_arrays(
    panel: Path,
    *,
    unit_index: int,
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    object_id, episode, _stratum = SOURCE_ROSTER[unit_index]
    directory = (
        panel / "unit-artifacts" / f"{unit_index:02d}-{object_id}-ep{episode:04d}"
    )
    archive = _ordinary_file(
        directory / "prediction-arrays.npz", name="prediction archive"
    )
    archive_record = _mapping(manifest.get("archive"), name="prediction archive record")
    _require(
        archive_record.get("file_sha256") == _sha256_file(archive)
        and archive_record.get("size_bytes") == archive.stat().st_size,
        "prediction archive identity changed",
    )
    try:
        with np.load(archive, allow_pickle=False) as stored:
            _require(
                set(stored.files) >= {"mean_m", "covariance_m2"},
                "prediction arrays are incomplete",
            )
            mean = np.asarray(stored["mean_m"], dtype=np.float64, order="C")
            covariance = np.asarray(
                stored["covariance_m2"], dtype=np.float64, order="C"
            )
    except (OSError, ValueError) as error:
        raise ValueError("cannot load prediction arrays") from error
    descriptors = _mapping(manifest.get("arrays"), name="prediction array descriptors")
    for name, value in (("mean_m", mean), ("covariance_m2", covariance)):
        descriptor = _mapping(descriptors.get(name), name=f"prediction {name}")
        _require(
            descriptor.get("sha256") == _array_sha256(value)
            and descriptor.get("shape") == list(value.shape)
            and descriptor.get("dtype") == value.dtype.str,
            f"prediction array changed: {name}",
        )
    _require(
        mean.ndim == 3
        and mean.shape[0] == 18
        and mean.shape[-1] == 3
        and covariance.shape == (*mean.shape, 3)
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(covariance)),
        "prediction arrays are malformed",
    )
    return mean, covariance


def _load_observation_arrays(
    root: Path,
    row: Mapping[str, Any],
    *,
    expected_mean_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    artifact = _mapping(row.get("artifact"), name="source observation artifact")
    relative = _canonical_relative_path(
        artifact.get("path"), name="source observation path"
    )
    archive = _resolve_file(root, relative, name="source observation archive")
    _require(
        artifact.get("sha256") == _sha256_file(archive)
        and artifact.get("size_bytes") == archive.stat().st_size,
        "source observation archive identity changed",
    )
    descriptors = _mapping(row.get("arrays"), name="source observation arrays")
    observation_record = _mapping(
        descriptors.get("observation_m"), name="observation_m descriptor"
    )
    valid_record = _mapping(descriptors.get("valid"), name="valid descriptor")
    try:
        with np.load(archive, allow_pickle=False) as stored:
            expected_members = {
                observation_record.get("member"),
                valid_record.get("member"),
            }
            _require(
                set(stored.files) == expected_members,
                "source observation archive member roster changed",
            )
            observation = np.asarray(
                stored[cast(str, observation_record["member"])],
                dtype=np.float64,
                order="C",
            )
            valid = np.asarray(stored[cast(str, valid_record["member"])])
    except (OSError, ValueError) as error:
        raise ValueError("cannot load source observation arrays") from error
    _require(
        observation.dtype == np.dtype(np.float64)
        and observation.shape == expected_mean_shape
        and valid.dtype == np.dtype(np.bool_)
        and valid.shape == expected_mean_shape[:2]
        and np.all(np.isfinite(observation[valid]))
        and not np.any(np.isinf(observation)),
        "source observation arrays are malformed",
    )
    for name, value, descriptor in (
        ("observation_m", observation, observation_record),
        ("valid", valid, valid_record),
    ):
        _require(
            descriptor.get("sha256") == _array_sha256(value)
            and descriptor.get("shape") == list(value.shape)
            and descriptor.get("dtype") == value.dtype.str,
            f"source observation array changed: {name}",
        )
    return observation, valid


def _verify_source_suffix_inputs(
    root: Path,
    row: Mapping[str, Any],
) -> set[str]:
    digests: set[str] = set()
    inputs = _sequence(
        row.get("source_suffix_input_artifacts"),
        name="source_suffix_input_artifacts",
    )
    for index, raw in enumerate(inputs):
        artifact = _mapping(raw, name=f"source suffix input artifact {index}")
        relative = _canonical_relative_path(
            artifact.get("path"),
            name=f"source suffix input path {index}",
        )
        path = _resolve_file(root, relative, name=f"source suffix input {index}")
        digest = _sha256(
            artifact.get("sha256"),
            name=f"source suffix input SHA-256 {index}",
        )
        _require(
            digest == _sha256_file(path)
            and artifact.get("size_bytes") == path.stat().st_size,
            "source suffix input artifact identity changed",
        )
        digests.add(digest)
    return digests


def _score_unit(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    valid: np.ndarray,
    *,
    exact_fallback: bool,
) -> dict[str, Any]:
    error = np.asarray(observation - mean, dtype=np.float64, order="C")
    horizon_rows: dict[str, Any] = {}
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for horizon, start, stop in HORIZONS:
        selected_valid = valid[start:stop]
        if not np.any(selected_valid):
            raise ValueError(f"source outcome has no valid {horizon} events")
        reference_nll, reference_coverage, reference_width = score_zero_covariance(
            error[start:stop],
            selected_valid,
            observation_std_m=OBSERVATION_STD_M,
            eigenvalue_floor_m2=COVARIANCE_EIGENVALUE_FLOOR_M2,
            marginal_coverage_z=MARGINAL_COVERAGE_Z,
        )
        if exact_fallback:
            _require(
                np.array_equal(
                    covariance[start:stop], np.zeros_like(covariance[start:stop])
                ),
                "exact fallback covariance changed",
            )
            candidate_nll = reference_nll
            candidate_coverage = reference_coverage
            candidate_width = reference_width
        else:
            nll, coverage, width = score_scale_grid(
                error[start:stop],
                covariance[start:stop],
                selected_valid,
                scales=(1.0,),
                observation_std_m=OBSERVATION_STD_M,
                eigenvalue_floor_m2=COVARIANCE_EIGENVALUE_FLOOR_M2,
                marginal_coverage_z=MARGINAL_COVERAGE_Z,
            )
            candidate_nll = float(nll[0])
            candidate_coverage = float(coverage[0])
            candidate_width = float(width[0])
        candidate_values.append(candidate_nll)
        reference_values.append(reference_nll)
        horizon_rows[horizon] = {
            "valid_event_count": int(np.sum(selected_valid)),
            "candidate_nll": candidate_nll,
            "reference_nll": reference_nll,
            "candidate_marginal_coverage_90": candidate_coverage,
            "reference_marginal_coverage_90": reference_coverage,
            "candidate_mean_full_interval_width_m": candidate_width,
            "reference_mean_full_interval_width_m": reference_width,
        }
    candidate = float(math.fsum(candidate_values) / len(candidate_values))
    reference = float(math.fsum(reference_values) / len(reference_values))
    return {
        "candidate_nll": candidate,
        "reference_nll": reference,
        "candidate_minus_reference_nll": candidate - reference,
        "horizon_scores": horizon_rows,
        "aggregation": "equal-valid-3d-event-within-horizon-then-equal-three-horizon-mean",
    }


def score_covariance_source_panel_v1(
    *,
    panel_root: str | Path,
    source_observations_path: str | Path,
    source_observation_root: str | Path,
    forbidden_confirmation_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Attach source outcomes once and return scores, decision, and receipt."""

    panel = _ordinary_directory(panel_root, name="source prediction panel")
    observation_root = _ordinary_directory(
        source_observation_root, name="source observation root"
    )
    forbidden = _ordinary_directory(
        forbidden_confirmation_root, name="forbidden confirmation root"
    )
    for admitted in (panel, observation_root):
        _require_disjoint(
            admitted,
            forbidden,
            message="source scorer path overlaps the forbidden confirmation root",
        )
    panel_receipt = validate_covariance_source_panel_v1(panel)
    batch_path = _ordinary_file(
        panel / "source-prediction-batch.json", name="source prediction batch"
    )
    batch = validate_prediction_batch(
        _load_json(batch_path, label="source prediction batch")
    )
    observation_path = _ordinary_file(
        source_observations_path, name="source observation manifest"
    )
    _require(
        observation_root in observation_path.parents,
        "source observation manifest is outside its admitted root",
    )
    observations = validate_source_observations_v1(
        _load_json(observation_path, label="source observations")
    )
    _require(
        observations.get("prediction_batch_id") == batch.get("batch_id"),
        "source observations bind another prediction batch",
    )

    selected = cast(Mapping[str, str], batch["scoring_prediction_by_source_unit"])
    records = cast(Sequence[Mapping[str, Any]], batch["records"])
    record_by_id = {cast(str, record["prediction_id"]): record for record in records}
    score_rows: list[dict[str, Any]] = []
    for unit_index, ((object_id, episode, stratum), raw_observation) in enumerate(
        zip(
            SOURCE_ROSTER,
            cast(Sequence[Mapping[str, Any]], observations["rows"]),
            strict=True,
        )
    ):
        observation_row = _mapping(
            raw_observation, name=f"source observations[{unit_index}]"
        )
        prediction_id = selected[f"{object_id}#{episode}"]
        record = record_by_id[prediction_id]
        unit_directory = (
            panel / "unit-artifacts" / f"{unit_index:02d}-{object_id}-ep{episode:04d}"
        )
        manifest = _load_json(
            unit_directory / "prediction-manifest.json",
            label="unit prediction manifest",
        )
        provenance = _mapping(
            manifest.get("provenance"), name="prediction source provenance"
        )
        _require(
            observation_row.get("prediction_id") == prediction_id
            and observation_row.get("unit_manifest_id") == manifest.get("manifest_id")
            and observation_row.get("scoring_reconstruction_id")
            == provenance.get("scoring_reconstruction_id")
            and observation_row.get("scoring_configuration_id")
            == provenance.get("scoring_configuration_id")
            and observation_row.get("scoring_camera_family_ids")
            == provenance.get("scoring_camera_family_ids")
            and observation_row.get("scoring_plan_artifact_ids")
            == provenance.get("scoring_input_artifact_ids")
            and observations.get("scoring_implementation_revision")
            == provenance.get("scoring_implementation_revision"),
            "source observation provenance differs from the sealed scoring plan",
        )
        provider_inputs = set(
            cast(Sequence[str], provenance.get("provider_input_artifact_ids", ()))
        )
        scoring_inputs = _verify_source_suffix_inputs(observation_root, observation_row)
        _require(
            not provider_inputs.intersection(scoring_inputs),
            "provider and scoring source artifacts overlap",
        )

        if observation_row.get("disposition") == "technical_failure":
            score_rows.append(
                {
                    "object_id": object_id,
                    "episode": episode,
                    "stratum": stratum,
                    "prediction_id": prediction_id,
                    "disposition": "technical_failure",
                    "point_mean_identity": True,
                    "point_metric_difference_m": 0.0,
                    "supported_or_exact_fallback": False,
                    "exact_fallback": False,
                    "candidate_nll": None,
                    "reference_nll": None,
                    "technical_failure_reason": observation_row[
                        "technical_failure_reason"
                    ],
                }
            )
            continue

        mean, covariance = _load_prediction_arrays(
            panel, unit_index=unit_index, manifest=manifest
        )
        _require(
            record.get("mean_sha256")
            == record.get("reference_mean_sha256")
            == _array_sha256(mean)
            and record.get("covariance_sha256") == _array_sha256(covariance)
            and record.get("mean_bytes_identical") is True,
            "selected source prediction changed the registered mean or covariance",
        )
        observation, valid = _load_observation_arrays(
            observation_root,
            observation_row,
            expected_mean_shape=mean.shape,
        )
        exact_fallback = bool(record["exact_fallback"])
        metrics = _score_unit(
            mean,
            covariance,
            observation,
            valid,
            exact_fallback=exact_fallback,
        )
        score_rows.append(
            {
                "object_id": object_id,
                "episode": episode,
                "stratum": stratum,
                "prediction_id": prediction_id,
                "disposition": "exact_fallback" if exact_fallback else "candidate",
                "point_mean_identity": True,
                "point_metric_difference_m": 0.0,
                "supported_or_exact_fallback": True,
                "exact_fallback": exact_fallback,
                **metrics,
            }
        )

    scores = seal_source_scores(
        {
            "schema": SCORES_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch["batch_id"],
            "observation_set_id": observations["observation_set_id"],
            "information_boundary": {
                "source_suffix_opened": True,
                "confirmation_payloads_opened": False,
                "confirmation_predictions_run": False,
                "confirmation_outcomes_used": False,
                "candidate_retuned": False,
                "replacement_used": False,
            },
            "rows": score_rows,
        }
    )
    decision = evaluate_source_gate(batch, scores)
    receipt_identity: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "source-scoring-complete",
        "software_protocol_id": SOFTWARE_PROTOCOL_ID,
        "paper_protocol_id": PAPER_PROTOCOL_ID,
        "panel_receipt_id": panel_receipt["receipt_id"],
        "prediction_batch_id": batch["batch_id"],
        "observation_set_id": observations["observation_set_id"],
        "scoring_implementation_revision": observations[
            "scoring_implementation_revision"
        ],
        "source_scores_id": scores["score_set_id"],
        "source_decision_id": decision["decision_id"],
        "source_decision_status": decision["status"],
        "source_unit_count": len(SOURCE_ROSTER),
        "source_suffix_opened": True,
        "confirmation_payloads_opened": False,
        "confirmation_predictions_run": False,
        "confirmation_outcomes_used": False,
        "claim_authorized": False,
    }
    receipt = cast(
        dict[str, Any],
        _plain_json(
            {
                **receipt_identity,
                "receipt_id": _content_id(receipt_identity, "receipt_id"),
            }
        ),
    )
    return scores, decision, receipt


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(_plain_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def publish_covariance_source_scores_v1(
    *,
    panel_root: str | Path,
    source_observations_path: str | Path,
    source_observation_root: str | Path,
    forbidden_confirmation_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish one complete source-score result without overwriting evidence."""

    output = Path(output_root).absolute()
    _require(not output.exists(), "refusing to overwrite source-score evidence")
    parent = _ordinary_directory(output.parent, name="source-score output parent")
    panel = _ordinary_directory(panel_root, name="source prediction panel")
    observation_root = _ordinary_directory(
        source_observation_root, name="source observation root"
    )
    forbidden = _ordinary_directory(
        forbidden_confirmation_root,
        name="forbidden confirmation root",
    )
    _require_disjoint(
        parent,
        forbidden,
        message="source-score output overlaps the forbidden confirmation root",
    )
    for immutable_input in (panel, observation_root):
        _require_disjoint(
            output,
            immutable_input,
            message="source-score output overlaps an immutable input root",
        )
    temporary = parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        scores, decision, receipt = score_covariance_source_panel_v1(
            panel_root=panel,
            source_observations_path=source_observations_path,
            source_observation_root=observation_root,
            forbidden_confirmation_root=forbidden_confirmation_root,
        )
        _write_json_once(temporary / "source-scores.json", scores)
        _write_json_once(temporary / "source-decision.json", decision)
        receipt = {
            **receipt,
            "source_scores_file_sha256": _sha256_file(temporary / "source-scores.json"),
            "source_decision_file_sha256": _sha256_file(
                temporary / "source-decision.json"
            ),
        }
        receipt.pop("receipt_id")
        receipt["receipt_id"] = _content_id(receipt, "receipt_id")
        _write_json_once(temporary / "source-score-receipt.json", receipt)
        os.replace(temporary, output)
        return cast(dict[str, Any], _plain_json(receipt))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_covariance_source_scores_v1(output_root: str | Path) -> dict[str, Any]:
    """Rehash a published source-score result without opening source arrays."""

    root = _ordinary_directory(output_root, name="source-score evidence")
    receipt_path = _ordinary_file(
        root / "source-score-receipt.json", name="source-score receipt"
    )
    scores_path = _ordinary_file(root / "source-scores.json", name="source scores")
    decision_path = _ordinary_file(
        root / "source-decision.json", name="source decision"
    )
    receipt = _load_json(receipt_path, label="source-score receipt")
    declared = _sha256(receipt.get("receipt_id"), name="receipt_id")
    _require(
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("schema_version") == SCHEMA_VERSION
        and _revision(
            receipt.get("scoring_implementation_revision"),
            name="scoring_implementation_revision",
        )
        == receipt.get("scoring_implementation_revision")
        and declared == _content_id(receipt, "receipt_id")
        and receipt.get("source_scores_file_sha256") == _sha256_file(scores_path)
        and receipt.get("source_decision_file_sha256") == _sha256_file(decision_path)
        and receipt.get("source_suffix_opened") is True
        and receipt.get("confirmation_payloads_opened") is False
        and receipt.get("confirmation_predictions_run") is False
        and receipt.get("confirmation_outcomes_used") is False
        and receipt.get("claim_authorized") is False,
        "source-score receipt changed",
    )
    scores = _load_json(scores_path, label="source scores")
    decision = _load_json(decision_path, label="source decision")
    _require(
        scores.get("score_set_id") == _content_id(scores, "score_set_id")
        and decision.get("decision_id") == _content_id(decision, "decision_id")
        and receipt.get("source_scores_id") == scores.get("score_set_id")
        and receipt.get("source_decision_id") == decision.get("decision_id")
        and receipt.get("source_decision_status") == decision.get("status"),
        "source-score result differs from its receipt",
    )
    return receipt


__all__ = [
    "HORIZONS",
    "MARGINAL_COVERAGE_Z",
    "OBSERVATIONS_SCHEMA",
    "RECEIPT_SCHEMA",
    "publish_covariance_source_scores_v1",
    "score_covariance_source_panel_v1",
    "seal_source_observations_v1",
    "validate_covariance_source_scores_v1",
    "validate_source_observations_v1",
]
