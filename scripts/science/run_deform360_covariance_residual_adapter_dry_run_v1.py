#!/usr/bin/env python3
"""Run the locked source-only Deform360 covariance adapter dry run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from bayesian_phystwin.deform360_covariance_residual_adapter_v1 import (
    Deform360ResidualHistoryAdapterConfigV1,
    Deform360ResidualHistoryIdentityV1,
    adapt_deform360_covariance_residual_history_v1,
)
from bayesian_phystwin.endpoint_model_average import (
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin/deform360-covariance-residual-adapter-dry-run-v1"
)
PROTOCOL_ID: Final = "deform360-covariance-residual-adapter-dry-run-v1"
RESULT_SCHEMA: Final = (
    "bayesian-phystwin/deform360-covariance-residual-adapter-dry-run-result-v1"
)
EXPECTED_PROTOCOL_SHA256: Final = (
    "12fa5e4896f285cbbffb0825e7695fa4"
    "83e292716c1d25502ad96968308f6255"
)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read protocol: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("protocol must contain a JSON object")
    declared = value.get("protocol_sha256")
    descriptor = {key: item for key, item in value.items() if key != "protocol_sha256"}
    computed = _canonical_sha256(descriptor)
    if declared != computed or computed != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 changed")
    if (
        value.get("schema") != PROTOCOL_SCHEMA
        or value.get("schema_version") != 1
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "source-only-contract-dry-run"
    ):
        raise ValueError("unexpected residual-adapter protocol")
    boundary = value.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("protocol information_boundary must be an object")
    required_false = (
        "target_quarantine_path_referenced_by_workflow",
        "target_payload_opened",
        "target_outcomes_opened",
        "camera_media_decoded",
        "sensor_arrays_loaded_from_target",
        "predictions_run_on_target",
        "claim_authorized",
    )
    if boundary.get("source_only") is not True or any(
        boundary.get(field) is not False for field in required_false
    ):
        raise ValueError("protocol source-only boundary changed")
    return value


def _fixture(
    *,
    seed: int,
    prefix_frames: int,
    track_count: int,
    future_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    residual = rng.normal(0.0, 0.002, size=(prefix_frames, track_count, 3))
    valid = np.zeros((prefix_frames, track_count), dtype=bool)
    patterns = (
        tuple(range(prefix_frames)),
        (0, 2, 5, 7),
        (1, 3, 7),
        (0, 6),
        (4,),
    )
    if track_count != len(patterns) or prefix_frames != 8:
        raise ValueError("locked fixture dimensions changed")
    for track, frames in enumerate(patterns):
        valid[np.asarray(frames, dtype=int), track] = True
    residual[~valid] = np.nan
    physical = np.zeros((future_frames, track_count, 3), dtype=np.float64)
    time = np.arange(future_frames, dtype=np.float64)[:, None]
    track = np.arange(track_count, dtype=np.float64)[None, :]
    physical[..., 0] = 0.01 + 0.0002 * time + 0.0001 * track
    physical[..., 1] = -0.02 + 0.0001 * time - 0.00005 * track
    physical[..., 2] = 0.03 - 0.00015 * time + 0.00002 * track
    return residual, valid, np.ascontiguousarray(physical)


def _identity() -> Deform360ResidualHistoryIdentityV1:
    return Deform360ResidualHistoryIdentityV1(
        object_id="source-dry-run-object",
        session_id="source-dry-run-session",
        material_id="source-dry-run-material",
        coordinate_frame="source-canonical-metric-frame",
        provider_camera_ids=("source-camera-00", "source-camera-01"),
        scoring_camera_ids=("source-camera-02", "source-camera-03"),
        provider_artifact_ids=("source-provider-reconstruction-v1",),
        scoring_artifact_ids=("source-scoring-reconstruction-v1",),
    )


def _minimum_eigenvalue(covariance: np.ndarray) -> float:
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    return float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: object, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _expected_donor_covariance(
    residual: np.ndarray,
    valid: np.ndarray,
    horizon_steps: Sequence[int],
    *,
    minimum_valid_observations: int,
) -> np.ndarray:
    canonical = np.zeros(residual.shape, dtype=np.float64)
    canonical[valid] = np.asarray(residual, dtype=np.float64)[valid]
    support = np.sum(valid, axis=0) >= minimum_valid_observations
    provider_valid = valid & support[None, :]
    canonical[~provider_valid] = 0.0
    posterior = infer_model_averaged_endpoint(
        canonical,
        provider_valid,
        end_frame=len(canonical),
    )
    return np.stack(
        [
            predict_model_averaged_endpoint(
                posterior,
                horizon_steps=int(step),
            ).covariance_m2
            for step in horizon_steps
        ],
        axis=0,
    )


def run_dry_run(protocol: Mapping[str, Any]) -> dict[str, object]:
    fixture = protocol.get("dry_run_fixture")
    method = protocol.get("method")
    if not isinstance(fixture, Mapping) or not isinstance(method, Mapping):
        raise ValueError("protocol fixture or method is malformed")
    required_cases = tuple(str(value) for value in fixture["required_cases"])
    if int(fixture["case_count"]) != len(required_cases):
        raise ValueError("dry-run case count changed")
    expected_cases = (
        "ordinary_success_with_missingness",
        "masked_value_invariance",
        "provider_failure_exact_fallback",
        "insufficient_support_exact_physical_fallback",
    )
    if required_cases != expected_cases:
        raise ValueError("dry-run case roster or order changed")
    seed = int(fixture["seed"])
    prefix_frames = int(fixture["prefix_frame_count"])
    track_count = int(fixture["track_count"])
    future_frames = int(fixture["future_frame_count"])
    labels = tuple(str(value) for value in fixture["horizon_labels"])
    steps = tuple(range(1, future_frames + 1))
    residual, valid, physical = _fixture(
        seed=seed,
        prefix_frames=prefix_frames,
        track_count=track_count,
        future_frames=future_frames,
    )
    config = Deform360ResidualHistoryAdapterConfigV1(
        minimum_valid_observations_per_track=int(
            method["minimum_valid_observations_per_track"]
        ),
        horizon_scales=tuple(
            (label, float(method["horizon_covariance_multipliers"][label]))
            for label in ("early", "middle", "late")
        ),
    )
    identity = _identity()
    ordinary = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=labels,
        horizon_steps=steps,
        identity=identity,
        config=config,
    )
    raw_covariance = _expected_donor_covariance(
        residual,
        valid,
        steps,
        minimum_valid_observations=(
            config.minimum_valid_observations_per_track
        ),
    )
    expected_covariance = raw_covariance.copy()
    expected_covariance[:, ~ordinary.supported_track_mask, :, :] = 0.0
    scale_by_label = dict(config.horizon_scales)
    scale = np.asarray([scale_by_label[label] for label in labels])[:, None]
    expected_covariance *= scale[..., None, None]
    horizon_scales_exact = np.array_equal(
        ordinary.covariance_m2,
        expected_covariance,
    )

    alternate = residual.copy()
    alternate[~valid] = 1.0e6
    masked = adapt_deform360_covariance_residual_history_v1(
        alternate,
        valid,
        physical,
        horizon_labels=labels,
        horizon_steps=steps,
        identity=identity,
        config=config,
    )
    masked_values_ignored = (
        ordinary.record.canonical_residual_sha256
        == masked.record.canonical_residual_sha256
        and ordinary.record.artifact_id == masked.record.artifact_id
        and np.array_equal(ordinary.mean_m, masked.mean_m)
        and np.array_equal(ordinary.covariance_m2, masked.covariance_m2)
    )

    def failing_provider(
        _residual: np.ndarray,
        _valid: np.ndarray,
        _steps: tuple[int, ...],
    ) -> np.ndarray:
        raise RuntimeError("intentional source-only dry-run provider failure")

    provider_failure = adapt_deform360_covariance_residual_history_v1(
        residual,
        valid,
        physical,
        horizon_labels=labels,
        horizon_steps=steps,
        identity=identity,
        config=config,
        covariance_provider=failing_provider,
    )
    provider_failure_exact = (
        provider_failure.record.provider_status == "fallback-provider-failure"
        and np.array_equal(provider_failure.mean_m, ordinary.mean_m)
        and np.count_nonzero(provider_failure.covariance_m2) == 0
    )

    insufficient_valid = np.zeros_like(valid)

    def forbidden_provider(
        _residual: np.ndarray,
        _valid: np.ndarray,
        _steps: tuple[int, ...],
    ) -> np.ndarray:
        raise AssertionError("provider must not run without supported tracks")

    insufficient = adapt_deform360_covariance_residual_history_v1(
        residual,
        insufficient_valid,
        physical,
        horizon_labels=labels,
        horizon_steps=steps,
        identity=identity,
        config=config,
        covariance_provider=forbidden_provider,
    )
    insufficient_support_exact = (
        insufficient.record.provider_status == "fallback-no-supported-tracks"
        and np.array_equal(insufficient.mean_m, physical)
        and _array_sha256(insufficient.mean_m) == _array_sha256(physical)
        and np.count_nonzero(insufficient.covariance_m2) == 0
    )

    unsupported = ~ordinary.supported_track_mask
    unsupported_exact = np.array_equal(
        ordinary.mean_m[:, unsupported, :],
        physical[:, unsupported, :],
    )
    gates = {
        "validity_hash_preserved": ordinary.record.validity_sha256
        == masked.record.validity_sha256,
        "masked_values_do_not_change_output": masked_values_ignored,
        "minimum_support_enforced": ordinary.record.supported_track_count == 3
        and ordinary.record.unsupported_track_count == 2,
        "candidate_mean_is_reference_object": (
            ordinary.record.mean_object_identity_preserved
        ),
        "unsupported_tracks_are_byte_identical_to_physical_fallback": (
            unsupported_exact
        ),
        "provider_failure_is_exact_zero_covariance_fallback": (
            provider_failure_exact
        ),
        "insufficient_support_is_exact_physical_fallback": (
            insufficient_support_exact
        ),
        "output_covariance_is_psd": (
            _minimum_eigenvalue(ordinary.covariance_m2)
            >= -config.covariance_psd_tolerance
        ),
        "horizon_scales_are_exact": horizon_scales_exact,
        "camera_sets_are_disjoint": not (
            set(identity.provider_camera_ids) & set(identity.scoring_camera_ids)
        ),
        "artifact_sets_are_disjoint": not (
            set(identity.provider_artifact_ids) & set(identity.scoring_artifact_ids)
        ),
        "target_payload_opened": False,
        "target_outcomes_opened": False,
        "claim_authorized": False,
    }
    cases = {
        "ordinary_success_with_missingness": {
            "artifact_id": ordinary.record.artifact_id,
            "provider_status": ordinary.record.provider_status,
            "supported_track_count": ordinary.record.supported_track_count,
            "unsupported_track_count": ordinary.record.unsupported_track_count,
            "minimum_covariance_eigenvalue_m2": _minimum_eigenvalue(
                ordinary.covariance_m2
            ),
            "reference_mean_sha256": ordinary.record.reference_mean_sha256,
            "output_covariance_sha256": ordinary.record.output_covariance_sha256,
        },
        "masked_value_invariance": {
            "artifact_id": masked.record.artifact_id,
            "passed": masked_values_ignored,
        },
        "provider_failure_exact_fallback": {
            "artifact_id": provider_failure.record.artifact_id,
            "provider_status": provider_failure.record.provider_status,
            "provider_error_type": provider_failure.record.provider_error_type,
            "passed": provider_failure_exact,
        },
        "insufficient_support_exact_physical_fallback": {
            "artifact_id": insufficient.record.artifact_id,
            "provider_status": insufficient.record.provider_status,
            "passed": insufficient_support_exact,
        },
    }
    if tuple(cases) != required_cases:
        raise ValueError("dry-run case roster changed")
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_revision": os.environ.get("GITHUB_SHA", "local-dry-run"),
        "cases": cases,
        "gates": gates,
        "dry_run_passed": all(
            bool(value)
            for key, value in gates.items()
            if key not in {
                "target_payload_opened",
                "target_outcomes_opened",
                "claim_authorized",
            }
        )
        and not bool(gates["target_payload_opened"])
        and not bool(gates["target_outcomes_opened"])
        and not bool(gates["claim_authorized"]),
        "information_boundary": {
            "source_only": True,
            "camera_media_decoded": False,
            "sensor_arrays_loaded_from_target": False,
            "target_payload_opened": False,
            "target_outcomes_opened": False,
            "predictions_run_on_target": False,
            "claim_authorized": False,
        },
    }
    plain = _jsonable(result)
    if not isinstance(plain, dict):
        raise AssertionError("dry-run result did not remain a JSON object")
    plain["result_sha256"] = _canonical_sha256(plain)
    return plain


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "protocols/locks/"
            "deform360_covariance_residual_adapter_dry_run_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = _load_protocol(args.protocol)
    result = run_dry_run(protocol)
    if result["dry_run_passed"] is not True:
        raise SystemExit("source-only residual adapter dry run failed")
    _write_json(args.output, result, force=args.force)
    print(
        json.dumps(
            _jsonable(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
