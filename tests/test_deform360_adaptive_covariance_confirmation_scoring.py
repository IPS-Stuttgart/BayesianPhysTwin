from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_scoring as scoring
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal as seal
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_outcome_adapter import (
    EXTERNAL_AUTHORIZED_OUTCOME_KIND,
    EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
    EXTERNAL_TARGET_ARCHIVE_FILENAME,
    ConfirmationNativeOfficialTarget,
)
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
)


H1 = "a" * 40
H2 = "b" * 40


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _external_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frame_zero() -> np.ndarray:
    ids = np.arange(20, dtype=np.float32)
    return np.stack(
        (
            np.float32(0.02) * ids,
            np.float32(0.01) * (ids % np.float32(4.0)),
            np.float32(0.005) * (ids % np.float32(3.0)),
        ),
        axis=1,
    ).astype(np.float32)


def _target() -> np.ndarray:
    target = np.repeat(_frame_zero()[None, :, :], 76, axis=0)
    target[:, :, 1] += np.arange(76, dtype=np.float32)[:, None] * np.float32(0.0002)
    return target


def _larger_permuted_official_target(target: np.ndarray) -> np.ndarray:
    permutation = np.asarray(
        [7, 2, 18, 11, 0, 15, 4, 19, 9, 13, 6, 1, 17, 10, 5, 14, 3, 16, 8, 12],
        dtype=np.int64,
    )
    matched = target[:, permutation].copy()
    matched[0, :, 0] += np.float32(0.003)
    extras = np.empty((76, 4, 3), dtype=np.float32)
    extras[:, :, 0] = np.float32(1.0) + np.arange(
        4,
        dtype=np.float32,
    )[None, :] * np.float32(0.02)
    extras[:, :, 1] = np.arange(76, dtype=np.float32)[:, None] * np.float32(0.0001)
    extras[:, :, 2] = np.float32(0.25)
    return np.concatenate((matched, extras), axis=1)


def _prediction_arrays(target: np.ndarray) -> dict[str, np.ndarray]:
    def shifted(distance: float) -> np.ndarray:
        value = target.copy()
        value[1:, :, 0] += np.float32(distance)
        value[0] = target[0]
        return value

    physical = shifted(0.001)
    adaptive = shifted(0.0005)
    adaptive[58:76] = physical[58:76]
    return {
        "physical_prior_m": physical,
        "persistence_m": shifted(0.004),
        "adaptive_prediction_m": adaptive,
        "fixed_4_rbf_prediction_m": shifted(0.003),
        "fixed_8_rbf_prediction_m": shifted(0.002),
        "selected_raw_prediction_m": physical.copy(),
    }


def _cameras() -> dict[int, list[str]]:
    eight = [f"camera-{index:02d}" for index in range(8)]
    return {4: eight[:4], 8: eight}


def _budget_record(*, reliable: bool, dispersion: float) -> dict[str, Any]:
    return {
        "valid_covariance_center_count": 8,
        "valid_covariance_center_ids": list(range(8)),
        "normalized_covariance_dispersion": dispersion,
        "reliable": reliable,
    }


def _routing() -> dict[str, Any]:
    cameras = _cameras()
    return {
        "protocol_id": ADAPTIVE_COVARIANCE_PROTOCOL_ID,
        "fallback": {
            "trajectory": "physical_prior",
            "rbf_state_update": False,
            "bit_exact": True,
        },
        "updates": [
            {
                "frame": 19,
                "stop_frame_exclusive": 38,
                "route": "4_view_rbf",
                "selected_camera_budget": 4,
                "tracked_camera_count": 4,
                "tracked_cameras": cameras[4],
                "selected_backbone": "physical_prior",
                "rbf_correction_applied": True,
                "state_updated": True,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=True, dispersion=0.01),
                },
            },
            {
                "frame": 38,
                "stop_frame_exclusive": 57,
                "route": "8_view_rbf",
                "selected_camera_budget": 8,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras[8],
                "selected_backbone": "persistence",
                "rbf_correction_applied": True,
                "state_updated": True,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=False, dispersion=0.02),
                    "8": _budget_record(reliable=True, dispersion=0.012),
                },
            },
            {
                "frame": 57,
                "stop_frame_exclusive": 76,
                "route": "physical_prior_fallback",
                "selected_camera_budget": None,
                "tracked_camera_count": 8,
                "tracked_cameras": cameras[8],
                "selected_backbone": "physical_prior",
                "rbf_correction_applied": False,
                "state_updated": False,
                "budget_diagnostics": {
                    "4": _budget_record(reliable=False, dispersion=0.02),
                    "8": _budget_record(reliable=False, dispersion=0.025),
                },
            },
        ],
    }


@dataclass
class _Fixture:
    lock_path: Path
    barrier_path: Path
    case_id: str
    case_root: Path
    barrier_case: dict[str, Any]
    case_dirs: dict[str, Path]
    measurement_dirs: dict[str, Path]
    future_dirs: dict[str, Path]
    outcome_dirs: dict[str, Path]
    compatibility_root: Path
    target: np.ndarray
    target_archive: Path


def _build_fixture(
    tmp_path: Path,
    *,
    larger_permuted_official_target: bool = False,
) -> _Fixture:
    lock_path = tmp_path / "lock" / "adaptive-confirmation-h2.json"
    lock.write_confirmation_cohort_lock(lock_path, H1)
    lock_payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=H1,
    )
    cases = list(lock_payload["selected_case_ids"])
    case_id = cases[0]
    case_root = tmp_path / "cases" / case_id
    sealed_target = _target()
    barrier_case = seal.seal_confirmation_case(
        lock_path,
        H2,
        case_id,
        case_root,
        _prediction_arrays(sealed_target),
        _cameras(),
        _routing(),
        {
            "status": "prediction_complete",
            "case_retained": True,
            "disposition_based_on_target_or_outcome": False,
            "center_ids": list(range(16)),
            "notes": "target-free scoring fixture",
        },
        expected_h1=H1,
    )
    barrier: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": seal.COHORT_BARRIER_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "status": "complete-target-free-cohort-prediction-barrier",
        "ordered_case_seals": [barrier_case],
    }
    barrier["artifact_sha256"] = seal.artifact_sha256(barrier)
    barrier_path = tmp_path / "barrier" / "prediction-barrier.json"
    _write_json(barrier_path, barrier)

    future_root = tmp_path / "future" / case_id
    outcome_root = tmp_path / "outcome" / case_id
    future_root.mkdir(parents=True)
    outcome_root.mkdir(parents=True)
    future: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "fixture-authorized-future",
    }
    future["result_sha256"] = _result_sha256(future)
    future_path = future_root / "authorized_future_manifest.json"
    _write_json(future_path, future)

    target = (
        _larger_permuted_official_target(sealed_target)
        if larger_permuted_official_target
        else sealed_target
    )
    target_archive = outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME
    visibility = np.ones(target.shape[:2], dtype=bool)
    validity = np.ones(target.shape[:2], dtype=bool)
    with target_archive.open("xb") as stream:
        np.savez_compressed(
            stream,
            target_m=target,
            target_visibility=visibility,
            target_validity=validity,
        )
    diagnostic = json.loads(
        (case_root / seal.DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
    )
    identity = diagnostic["case_identity"]
    outcome: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": EXTERNAL_AUTHORIZED_OUTCOME_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "protocol_config_sha256": lock_payload["artifact_sha256"],
        "case": case_id,
        "object_id": identity["object_id"],
        "episode_id": identity["episode_id"],
        "episode_key": f"{identity['object_id']}/{identity['episode_id']}",
        "stratum": identity["stratum"],
        "role": "calibration",
        "cameras": _cameras()[8],
        "target_frame_count": 76,
        "material_point_count": target.shape[1],
        "inputs_sha256": {
            "authorized_future_manifest": _file_sha256(future_path),
            "prediction_cohort_seal": "c" * 64,
        },
        "authorization": {
            "prediction_cohort_result_sha256": "d" * 64,
        },
        "output": {
            "target_archive": str(target_archive),
            "target_archive_sha256": _file_sha256(target_archive),
            "target_array_sha256": _external_array_sha256(target),
            "frame_zero_bit_exact_to_sealed_baseline": (
                not larger_permuted_official_target
            ),
        },
        "information_boundary": {
            "prediction_cohort_verified_before_target_construction": True,
            "future_tactile_read": False,
            "prediction_metric_computed": False,
        },
    }
    outcome["result_sha256"] = _result_sha256(outcome)
    _write_json(
        outcome_root / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME,
        outcome,
    )

    def bound_paths(stem: str, first: Path) -> dict[str, Path]:
        return {
            selected: (first if selected == case_id else tmp_path / stem / selected)
            for selected in cases
        }

    return _Fixture(
        lock_path=lock_path,
        barrier_path=barrier_path,
        case_id=case_id,
        case_root=case_root,
        barrier_case=barrier_case,
        case_dirs=bound_paths("unopened-cases", case_root),
        measurement_dirs=bound_paths(
            "unopened-measurements",
            tmp_path / "measurements" / case_id,
        ),
        future_dirs=bound_paths("unopened-future", future_root),
        outcome_dirs=bound_paths("unopened-outcome", outcome_root),
        compatibility_root=tmp_path / "compatibility",
        target=target,
        target_archive=target_archive,
    )


def _fake_native_validator(
    *,
    mutate_evidence: Callable[[dict[str, Any]], None] | None = None,
    calls: list[str] | None = None,
) -> Callable[..., ConfirmationNativeOfficialTarget]:
    def validate(
        adapter_repository: Path,
        lock_path: Path,
        h2_commit: str,
        barrier_path: Path,
        case_seal_dirs: Mapping[str, Path],
        nested_measurement_dirs: Mapping[str, Path],
        compatibility_root: Path,
        case_id: str,
        future_root: Path,
        outcome_root: Path,
        *,
        expected_h1: str,
    ) -> ConfirmationNativeOfficialTarget:
        del (
            adapter_repository,
            nested_measurement_dirs,
        )
        if calls is not None:
            calls.append(case_id)
        lock_payload = lock.load_confirmation_cohort_lock(
            lock_path,
            expected_implementation_commit_h1=expected_h1,
        )
        barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
        rows = [
            row for row in barrier["ordered_case_seals"] if row["case_id"] == case_id
        ]
        assert len(rows) == 1
        barrier_case = rows[0]
        case_root = Path(case_seal_dirs[case_id])
        diagnostic = json.loads(
            (case_root / seal.DIAGNOSTIC_FILENAME).read_text(encoding="utf-8")
        )
        with np.load(
            case_root / seal.ARRAY_ARCHIVE_FILENAME,
            allow_pickle=False,
        ) as stored:
            adaptive = np.array(stored["adaptive_prediction_m"], copy=True)
            selected_raw = np.array(
                stored["selected_raw_prediction_m"],
                copy=True,
            )
        target_archive = outcome_root / EXTERNAL_TARGET_ARCHIVE_FILENAME
        with np.load(target_archive, allow_pickle=False) as stored:
            target_arrays = {
                role: np.array(stored[role], copy=True)
                for role in scoring.TARGET_ARRAY_ROLES
            }
        future_path = future_root / "authorized_future_manifest.json"
        outcome_path = outcome_root / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
        future = json.loads(future_path.read_text(encoding="utf-8"))
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": lock.PROTOCOL_ID,
            "case_identity": diagnostic["case_identity"],
            "lock_binding": {
                "implementation_commit_h1": expected_h1,
                "cohort_lock_commit_h2": h2_commit,
                "cohort_lock_artifact_sha256": lock_payload["artifact_sha256"],
                "cohort_lock_file_sha256": _file_sha256(lock_path),
            },
            "prediction_barrier": {
                "path": str(barrier_path),
                "file_sha256": _file_sha256(barrier_path),
                "artifact_sha256": barrier["artifact_sha256"],
            },
            "case_seal": {
                "case_seal_root": str(case_root),
                "case_seal_file_sha256": barrier_case["manifest_file_sha256"],
                "case_seal_artifact_sha256": barrier_case["manifest_artifact_sha256"],
                "prediction_archive_file_sha256": barrier_case[
                    "prediction_archive_sha256"
                ],
                "prediction_arrays": {
                    "adaptive_prediction_m": seal.array_sha256(adaptive),
                    "selected_raw_prediction_m": seal.array_sha256(selected_raw),
                },
                "diagnostic_file_sha256": barrier_case["diagnostic_file_sha256"],
                "diagnostic_artifact_sha256": barrier_case[
                    "diagnostic_artifact_sha256"
                ],
            },
            "nested_measurement": {},
            "identity_persistence_adapter": None,
            "selected_cameras": diagnostic["nested_selected_cameras"]["8"],
            "compatibility_manifest": {
                "path": str(
                    compatibility_root / "confirmation_outcome_compatibility.json"
                ),
                "file_sha256": outcome["inputs_sha256"]["prediction_cohort_seal"],
                "result_sha256": outcome["authorization"][
                    "prediction_cohort_result_sha256"
                ],
            },
            "authorized_future_manifest": {
                "path": str(future_path),
                "file_sha256": _file_sha256(future_path),
                "result_sha256": future["result_sha256"],
            },
            "authorized_outcome_manifest": {
                "path": str(outcome_path),
                "file_sha256": _file_sha256(outcome_path),
                "result_sha256": outcome["result_sha256"],
            },
            "target_archive": {
                "path": str(target_archive),
                "file_sha256": _file_sha256(target_archive),
                "arrays": {
                    role: _external_array_sha256(target_arrays[role])
                    for role in scoring.TARGET_ARRAY_ROLES
                },
            },
            "information_boundary": {
                "native_official_arrays_returned": True,
                "metric_or_score_computed": False,
            },
        }
        if mutate_evidence is not None:
            mutate_evidence(evidence)
        for value in target_arrays.values():
            value.setflags(write=False)
        return ConfirmationNativeOfficialTarget(
            target_m=target_arrays["target_m"],
            target_visibility=target_arrays["target_visibility"],
            target_validity=target_arrays["target_validity"],
            evidence=evidence,
        )

    return validate


def _loader(
    fixture: _Fixture,
    validator: Callable[..., ConfirmationNativeOfficialTarget],
):
    return scoring.build_confirmation_case_target_loader(
        fixture.lock_path.parent.parent / "adapter",
        fixture.lock_path,
        H2,
        fixture.barrier_path,
        fixture.case_dirs,
        fixture.measurement_dirs,
        fixture.compatibility_root,
        fixture.future_dirs,
        fixture.outcome_dirs,
        expected_h1=H1,
        native_target_validator=validator,
    )


def test_case_target_loader_scores_frozen_arms_and_sealed_routes(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    calls: list[str] = []
    callback = _loader(
        fixture,
        _fake_native_validator(calls=calls),
    )

    result = callback(
        fixture.case_id,
        fixture.case_root,
        fixture.barrier_case,
    )

    assert calls == [fixture.case_id]
    assert set(result) == {
        "case_id",
        "diagnostic_file_sha256",
        "diagnostic_artifact_sha256",
        "target_file_sha256",
        "target_arrays_sha256",
        "frame_zero_scale_m",
        "metrics",
        "updates",
    }
    assert result["case_id"] == fixture.case_id
    assert (
        result["diagnostic_file_sha256"]
        == fixture.barrier_case["diagnostic_file_sha256"]
    )
    assert result["target_file_sha256"] == _file_sha256(fixture.target_archive)
    assert len(result["target_arrays_sha256"]) == 64
    assert result["frame_zero_scale_m"] > 0.0
    assert result["updates"] == [
        {
            "update_frame": 19,
            "route": "4_view_rbf",
            "attempted_camera_ids": _cameras()[4],
            "future_visual_update_applied": True,
            "rbf_state_updated": True,
            "fallback_reason": None,
        },
        {
            "update_frame": 38,
            "route": "8_view_rbf",
            "attempted_camera_ids": _cameras()[8],
            "future_visual_update_applied": True,
            "rbf_state_updated": True,
            "fallback_reason": None,
        },
        {
            "update_frame": 57,
            "route": "physical_prior_fallback",
            "attempted_camera_ids": _cameras()[8],
            "future_visual_update_applied": False,
            "rbf_state_updated": False,
            "fallback_reason": "covariance_abstention",
        },
    ]
    adaptive_mean_error = (36 * 0.0005 + 18 * 0.001) / 54
    expected_error = {
        "adaptive": adaptive_mean_error,
        "fixed8": 0.002,
        "fixed4": 0.003,
    }
    for arm, distance in expected_error.items():
        assert result["metrics"][arm][
            "post_update_hidden_identity_rmse_m"
        ] == pytest.approx(distance / np.sqrt(3.0), rel=2e-5)
        assert result["metrics"][arm][
            "post_update_hidden_symmetric_chamfer_m"
        ] == pytest.approx(distance, rel=2e-5)


def test_factory_does_not_open_any_official_target_path(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock.write_confirmation_cohort_lock(lock_path, H1)
    payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=H1,
    )
    cases = payload["selected_case_ids"]
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> ConfirmationNativeOfficialTarget:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("target validator must not run at factory time")

    mappings = {case_id: tmp_path / "does-not-exist" / case_id for case_id in cases}
    callback = scoring.build_confirmation_case_target_loader(
        tmp_path / "adapter-does-not-exist",
        lock_path,
        H2,
        tmp_path / "barrier-does-not-exist",
        mappings,
        mappings,
        tmp_path / "compatibility-does-not-exist",
        mappings,
        mappings,
        expected_h1=H1,
        native_target_validator=forbidden,
    )

    assert callable(callback)
    assert called is False


def test_scoring_factory_attestation_binds_code_closure_and_development_mode(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock.write_confirmation_cohort_lock(lock_path, H1)
    payload = lock.load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=H1,
    )
    mappings = {
        case_id: tmp_path / "opaque" / case_id
        for case_id in payload["selected_case_ids"]
    }
    adapter_repository = Path(scoring.__file__).parents[2]
    callback = scoring.build_confirmation_case_target_loader(
        adapter_repository,
        lock_path,
        H2,
        tmp_path / "barrier-does-not-exist",
        mappings,
        mappings,
        tmp_path / "compatibility-does-not-exist",
        mappings,
        mappings,
        expected_h1=H1,
    )

    attestation = scoring.validate_confirmation_case_target_loader_attestation(
        callback,
        lock_path,
        H2,
        expected_h1=H1,
        require_production=False,
    )
    assert attestation["factory_kind"] == scoring.SCORING_LOADER_FACTORY_KIND
    assert attestation["implementation_commit_h1"] == H1
    assert attestation["cohort_lock_commit_h2"] == H2
    assert attestation["cohort_lock_artifact_sha256"] == payload["artifact_sha256"]
    assert attestation["native_target_validator"]["is_exact_default"] is True
    assert attestation["scoring_source"]["canonical_adapter_source"] is True
    assert attestation["production_eligible"] is False
    assert attestation["ineligibility_reasons"] == [
        "factory_not_bound_to_exact_clean_h2"
    ]
    assert attestation["loader_callable"]["exact_factory_registry_binding_required"]
    assert attestation["repository_provenance"]["validated_exact_clean_h2"] is False

    def forged(*_args: object) -> dict[str, object]:
        return {}

    setattr(
        forged,
        scoring._LOADER_ATTESTATION_ATTRIBUTE,
        getattr(callback, scoring._LOADER_ATTESTATION_ATTRIBUTE),
    )
    with pytest.raises(ValueError, match="not issued by the frozen scoring factory"):
        scoring.validate_confirmation_case_target_loader_attestation(
            forged,
            lock_path,
            H2,
            expected_h1=H1,
        )

    scoring._LOADER_ATTESTATIONS[forged] = scoring._LOADER_ATTESTATIONS[callback]
    with pytest.raises(ValueError, match="code or closure registry binding changed"):
        scoring.validate_confirmation_case_target_loader_attestation(
            forged,
            lock_path,
            H2,
            expected_h1=H1,
        )
    del scoring._LOADER_ATTESTATIONS[forged]

    getattr(callback, scoring._LOADER_ATTESTATION_ATTRIBUTE)["production_eligible"] = (
        True
    )
    with pytest.raises(ValueError, match="attestation attribute changed"):
        scoring.validate_confirmation_case_target_loader_attestation(
            callback,
            lock_path,
            H2,
            expected_h1=H1,
        )


def test_custom_native_validator_loader_is_development_only(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock.write_confirmation_cohort_lock(lock_path, H1)
    payload = lock.load_confirmation_cohort_lock(lock_path)
    mappings = {
        case_id: tmp_path / "opaque" / case_id
        for case_id in payload["selected_case_ids"]
    }
    callback = scoring.build_confirmation_case_target_loader(
        Path(scoring.__file__).parents[2],
        lock_path,
        H2,
        tmp_path / "barrier",
        mappings,
        mappings,
        tmp_path / "compatibility",
        mappings,
        mappings,
        expected_h1=H1,
        native_target_validator=_fake_native_validator(),
    )
    with pytest.raises(ValueError, match="not eligible for production"):
        scoring.validate_confirmation_case_target_loader_attestation(
            callback,
            lock_path,
            H2,
            expected_h1=H1,
            require_production=True,
        )


def test_production_factory_rejects_noncanonical_fake_h2(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    payload = lock.write_confirmation_cohort_lock(lock_path, H1)
    mappings = {
        case_id: tmp_path / "opaque" / case_id
        for case_id in payload["selected_case_ids"]
    }
    with pytest.raises(ValueError):
        scoring.build_confirmation_case_target_loader(
            Path(scoring.__file__).parents[2],
            lock_path,
            H2,
            tmp_path / "barrier",
            mappings,
            mappings,
            tmp_path / "compatibility",
            mappings,
            mappings,
            expected_h1=H1,
            production_mode=True,
        )


def test_loader_transports_larger_permuted_nonidentical_official_frame_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(
        tmp_path,
        larger_permuted_official_target=True,
    )
    original = scoring.transport_official_target
    observed: list[Mapping[str, Any]] = []

    def record_transport(*args: Any, **kwargs: Any):
        transported = original(*args, **kwargs)
        observed.append(transported.diagnostics)
        return transported

    monkeypatch.setattr(
        scoring,
        "transport_official_target",
        record_transport,
    )
    callback = _loader(fixture, _fake_native_validator())

    result = callback(
        fixture.case_id,
        fixture.case_root,
        fixture.barrier_case,
    )

    assert len(observed) == 1
    assert observed[0]["official_identity_count"] == 24
    assert observed[0]["assigned_official_identity_count"] == 20
    assert observed[0]["sealed_point_coverage_fraction"] == 1.0
    assert observed[0]["assigned_official_identity_collision_count"] == 0
    assert observed[0]["observed_maximum_assignment_distance_m"] == (
        pytest.approx(0.003, rel=2e-5)
    )
    assert result["metrics"]["fixed8"][
        "post_update_hidden_symmetric_chamfer_m"
    ] == pytest.approx(0.002, rel=2e-5)
    outcome = json.loads(
        (
            fixture.outcome_dirs[fixture.case_id]
            / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert outcome["material_point_count"] == 24
    assert outcome["output"]["frame_zero_bit_exact_to_sealed_baseline"] is False


def test_loader_rejects_barrier_case_before_target_capability(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    calls: list[str] = []
    callback = _loader(
        fixture,
        _fake_native_validator(calls=calls),
    )
    forged = dict(fixture.barrier_case)
    forged["diagnostic_file_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="not in the sealed barrier"):
        callback(fixture.case_id, fixture.case_root, forged)

    assert calls == []


def test_loader_rejects_changed_target_construction_boundary(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    outcome_path = (
        fixture.outcome_dirs[fixture.case_id]
        / EXTERNAL_AUTHORIZED_OUTCOME_MANIFEST_FILENAME
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["information_boundary"]["prediction_metric_computed"] = True
    outcome["result_sha256"] = _result_sha256(outcome)
    _write_json(outcome_path, outcome)
    callback = _loader(fixture, _fake_native_validator())

    with pytest.raises(ValueError, match="outcome manifest changed"):
        callback(
            fixture.case_id,
            fixture.case_root,
            fixture.barrier_case,
        )


def test_loader_rejects_target_array_evidence_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    def forge(evidence: dict[str, Any]) -> None:
        evidence["target_archive"]["arrays"]["target_visibility"] = "0" * 64

    callback = _loader(
        fixture,
        _fake_native_validator(mutate_evidence=forge),
    )
    with pytest.raises(ValueError, match="target archive evidence changed"):
        callback(
            fixture.case_id,
            fixture.case_root,
            fixture.barrier_case,
        )


def test_loader_rejects_target_mutation_during_metric_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    callback = _loader(fixture, _fake_native_validator())
    original = scoring.score_deform360_hidden_trajectory
    mutated = False

    def mutate_once(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal mutated
        if not mutated:
            with fixture.target_archive.open("ab") as stream:
                stream.write(b"mutation")
            mutated = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        scoring,
        "score_deform360_hidden_trajectory",
        mutate_once,
    )
    with pytest.raises(ValueError, match="changed during official scoring"):
        callback(
            fixture.case_id,
            fixture.case_root,
            fixture.barrier_case,
        )
