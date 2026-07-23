from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

import bayesian_phystwin.deform360_adaptive_covariance_confirmation_evaluation as evaluation
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock


H1 = "a" * 40
H2 = "b" * 40


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_lock(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "confirmation-lock.json"
    payload = lock.write_confirmation_cohort_lock(path, H1)
    return path, payload


def _barrier(lock_path: Path, lock_payload: dict[str, Any]) -> dict[str, Any]:
    records = [
        {
            "case_id": case_id,
            "manifest_file_sha256": _sha256(f"{case_id}:manifest-file"),
            "manifest_artifact_sha256": _sha256(f"{case_id}:manifest-artifact"),
            "prediction_archive_sha256": _sha256(f"{case_id}:predictions"),
            "diagnostic_file_sha256": _sha256(f"{case_id}:diagnostic-file"),
            "diagnostic_artifact_sha256": _sha256(f"{case_id}:diagnostic-artifact"),
        }
        for case_id in lock_payload["selected_case_ids"]
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": evaluation.BARRIER_ARTIFACT_KIND,
        "protocol_id": lock.PROTOCOL_ID,
        "status": evaluation.BARRIER_STATUS,
        "lock_binding": {
            "file_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "artifact_sha256": lock_payload["artifact_sha256"],
            "implementation_commit_h1": H1,
            "cohort_lock_commit_h2": H2,
        },
        "exact_case_ids": list(lock_payload["selected_case_ids"]),
        "case_count": 34,
        "ordered_case_seals": records,
        "information_boundary": {
            "sealer_target_or_outcome_argument_accepted": False,
            "sealer_target_or_outcome_path_opened": False,
            "metric_or_score_computed": False,
            "prediction_content_only": True,
            "all_case_predictions_must_seal_before_barrier": True,
        },
    }
    payload["artifact_sha256"] = evaluation._canonical_sha256(
        payload,
        digest_key="artifact_sha256",
    )
    return payload


def _updates(
    routes: tuple[str, str, str] = (
        "4_view_rbf",
        "4_view_rbf",
        "8_view_rbf",
    ),
) -> list[dict[str, Any]]:
    records = []
    for frame, route in zip(evaluation.EXPECTED_UPDATE_FRAMES, routes, strict=True):
        count = 4 if route == "4_view_rbf" else 8
        fallback = route == "physical_prior_fallback"
        records.append(
            {
                "update_frame": frame,
                "route": route,
                "attempted_camera_ids": [
                    f"camera-{index:02d}" for index in range(count)
                ],
                "future_visual_update_applied": not fallback,
                "rbf_state_updated": not fallback,
                "fallback_reason": "covariance_abstention" if fallback else None,
            }
        )
    return records


def _outcome(
    case_id: str,
    barrier_case: dict[str, Any],
    *,
    adaptive: float = 0.99,
    fixed8: float = 1.0,
    fixed4: float = 1.02,
    routes: tuple[str, str, str] = (
        "4_view_rbf",
        "4_view_rbf",
        "8_view_rbf",
    ),
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "diagnostic_file_sha256": barrier_case["diagnostic_file_sha256"],
        "diagnostic_artifact_sha256": barrier_case["diagnostic_artifact_sha256"],
        "target_file_sha256": _sha256(f"{case_id}:target-file"),
        "target_arrays_sha256": _sha256(f"{case_id}:target-arrays"),
        "frame_zero_scale_m": 0.2,
        "metrics": {
            "adaptive": {metric: adaptive for metric in evaluation.METRICS},
            "fixed8": {metric: fixed8 for metric in evaluation.METRICS},
            "fixed4": {metric: fixed4 for metric in evaluation.METRICS},
        },
        "updates": _updates(routes),
    }


def _point_bootstrap(
    object_values: dict[str, Any],
    *,
    seed_sha256: str | None = None,
) -> dict[str, Any]:
    def arm_mean(arm: str, metric: str) -> float:
        return math.fsum(object_values["metrics"][arm][metric]) / 17

    return {
        "algorithm_id": "test-exact-point-bootstrap",
        "replicate_count": evaluation.BOOTSTRAP_REPLICATE_COUNT,
        "object_count": 17,
        "seed_sha256": seed_sha256 or "d" * 64,
        "resample_index_matrix_sha256": "e" * 64,
        "one_sided_upper_95": {
            "adaptive_vs_fixed8_ratio": {
                metric: arm_mean("adaptive", metric) / arm_mean("fixed8", metric)
                for metric in evaluation.METRICS
            },
            "adaptive_vs_fixed4_ratio": {
                metric: arm_mean("adaptive", metric) / arm_mean("fixed4", metric)
                for metric in evaluation.METRICS
            },
            "scale_normalized_difference": {
                metric: math.fsum(object_values["scale_normalized_difference"][metric])
                / 17
                for metric in evaluation.METRICS
            },
            "mean_counterfactual_policy_charged_camera_streams": math.fsum(
                object_values["mean_charged_cameras"]
            )
            / 17,
        },
    }


def _case_dirs(
    tmp_path: Path,
    lock_payload: dict[str, Any],
    barrier: dict[str, Any],
    *,
    routes: tuple[str, str, str] = (
        "4_view_rbf",
        "4_view_rbf",
        "8_view_rbf",
    ),
) -> dict[str, Path]:
    specifications = evaluation._case_specs(lock_payload)
    records = {record["case_id"]: record for record in barrier["ordered_case_seals"]}
    result: dict[str, Path] = {}
    cameras = [f"camera-{index:02d}" for index in range(8)]
    for specification in specifications:
        case_dir = tmp_path / "sealed-cases" / specification.case_id
        case_dir.mkdir(parents=True)
        routing_updates = []
        for frame, stop, route in zip(
            evaluation.EXPECTED_UPDATE_FRAMES,
            (38, 57, 76),
            routes,
            strict=True,
        ):
            fallback = route == "physical_prior_fallback"
            budget = 4 if route == "4_view_rbf" else 8
            attempted_budgets = (4,) if route == "4_view_rbf" else (4, 8)
            reliability = {
                "4_view_rbf": {4: True},
                "8_view_rbf": {4: False, 8: True},
                "physical_prior_fallback": {4: False, 8: False},
            }[route]
            routing_updates.append(
                {
                    "frame": frame,
                    "stop_frame_exclusive": stop,
                    "route": route,
                    "selected_camera_budget": None if fallback else budget,
                    "tracked_camera_count": budget,
                    "tracked_cameras": cameras[:budget],
                    "selected_backbone": (
                        "physical_prior" if fallback else "persistence"
                    ),
                    "rbf_correction_applied": not fallback,
                    "state_updated": not fallback,
                    "budget_diagnostics": {
                        str(attempted_budget): {
                            "valid_covariance_center_count": 8,
                            "valid_covariance_center_ids": list(range(8)),
                            "reliable": reliability[attempted_budget],
                        }
                        for attempted_budget in attempted_budgets
                    },
                }
            )
        diagnostic: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": evaluation.CASE_DIAGNOSTIC_KIND,
            "protocol_id": lock.PROTOCOL_ID,
            "case_identity": {
                "case_id": specification.case_id,
                "stratum": specification.stratum,
                "object_id": specification.object_id,
                "episode_id": specification.episode_id,
            },
            "nested_selected_cameras": {
                "4": cameras[:4],
                "8": cameras,
            },
            "covariance_routing": {
                "protocol_id": evaluation.ADAPTIVE_COVARIANCE_PROTOCOL_ID,
                "fallback": {
                    "trajectory": "physical_prior",
                    "rbf_state_update": False,
                    "bit_exact": True,
                },
                "updates": routing_updates,
            },
            "technical_disposition": {
                "status": "prediction_complete",
                "case_retained": True,
                "disposition_based_on_target_or_outcome": False,
                "center_ids": list(range(16)),
            },
            "information_boundary": dict(evaluation.TARGET_FREE_BOUNDARY),
        }
        diagnostic["artifact_sha256"] = evaluation._canonical_sha256(
            diagnostic,
            digest_key="artifact_sha256",
        )
        encoded = (
            json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        (case_dir / evaluation.DIAGNOSTIC_FILENAME).write_bytes(encoded)
        record = records[specification.case_id]
        record["diagnostic_file_sha256"] = hashlib.sha256(encoded).hexdigest()
        record["diagnostic_artifact_sha256"] = diagnostic["artifact_sha256"]
        result[specification.case_id] = case_dir
    barrier["artifact_sha256"] = evaluation._canonical_sha256(
        barrier,
        digest_key="artifact_sha256",
    )
    return result


def _sealed_evidence(
    barrier_case: dict[str, Any],
    *,
    routes: tuple[str, str, str] = (
        "4_view_rbf",
        "4_view_rbf",
        "8_view_rbf",
    ),
) -> dict[str, Any]:
    return {
        "diagnostic_file_sha256": barrier_case["diagnostic_file_sha256"],
        "diagnostic_artifact_sha256": barrier_case["diagnostic_artifact_sha256"],
        "center_ids": list(range(16)),
        "updates": _updates(routes),
    }


def test_complete_barrier_precedes_every_target_open_and_result_is_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    events: list[str] = []

    def validate(*_args: object, **_kwargs: object) -> dict[str, Any]:
        events.append("barrier")
        return barrier

    barrier_by_case = {
        record["case_id"]: record for record in barrier["ordered_case_seals"]
    }

    def load_target(
        case_id: str,
        _case_dir: Path,
        barrier_case: dict[str, Any],
    ) -> dict[str, Any]:
        assert barrier_case == barrier_by_case[case_id]
        events.append(f"target:{case_id}")
        return _outcome(case_id, barrier_case)

    original_diagnostic_loader = evaluation._load_sealed_diagnostic_evidence

    def load_diagnostic(
        case_dir: str | Path,
        *,
        specification: evaluation._CaseSpec,
        barrier_case: dict[str, Any],
    ) -> dict[str, Any]:
        events.append(f"diagnostic:{specification.case_id}")
        return original_diagnostic_loader(
            case_dir,
            specification=specification,
            barrier_case=barrier_case,
        )

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        validate,
    )
    monkeypatch.setattr(
        evaluation,
        "_load_sealed_diagnostic_evidence",
        load_diagnostic,
    )
    monkeypatch.setattr(evaluation, "_bootstrap_analysis", _point_bootstrap)
    result = evaluation.evaluate_adaptive_covariance_confirmation(
        lock_path,
        tmp_path / "prediction-barrier.json",
        H2,
        case_dirs,
        expected_h1=H1,
        target_loader=load_target,
    )

    assert events == (
        ["barrier"]
        + [f"diagnostic:{case_id}" for case_id in lock_payload["selected_case_ids"]]
        + [f"target:{case_id}" for case_id in lock_payload["selected_case_ids"]]
    )
    assert result["cohort"]["object_count"] == 17
    assert result["cohort"]["case_count"] == 34
    assert result["artifact_kind"] == evaluation.DEVELOPMENT_RESULT_ARTIFACT_KIND
    assert result["status"] == (
        "complete-target-opened-unattested-development-evaluation"
    )
    assert result["scoring_attestation"] == {
        "status": "unattested-development-only",
        "production_confirmation_authorized": False,
        "attestation": None,
        "evaluator_repository_provenance": None,
    }
    assert (
        "unattested development callback"
        in result["metric_contract"]["case_metric_role"]
    )
    assert result["claim_boundary"].startswith(
        "Development-only target-loader exercise."
    )
    assert "decision" not in result["aggregate"]["primary_confirmation"]
    assert (
        result["aggregate"]["primary_confirmation"][
            "development_statistical_gates_satisfied"
        ]
        is True
    )
    assert "decision" not in result["aggregate"]["fixed4_secondary"]
    assert (
        result["aggregate"]["fixed4_secondary"][
            "development_statistical_gates_satisfied"
        ]
        is True
    )
    assert result["aggregate"]["routes"]["counts"] == {
        "4_view_rbf": 68,
        "8_view_rbf": 34,
        "physical_prior_fallback": 0,
    }
    assert result["result_sha256"] == evaluation._canonical_sha256(
        result,
        digest_key="result_sha256",
    )


def test_production_evaluator_rejects_unattested_or_forged_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    target_calls = 0
    repository_validations = 0

    def custom_loader(*_args: object) -> dict[str, Any]:
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("unattested target loader must never be invoked")

    def validate_barrier(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return barrier

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        validate_barrier,
    )
    monkeypatch.setattr(
        evaluation,
        "_EXACT_PREDICTION_BARRIER_VALIDATOR",
        validate_barrier,
    )

    def validate_repository(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal repository_validations
        repository_validations += 1
        return {}

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_h2_loaded_runtime",
        validate_repository,
    )
    monkeypatch.setattr(
        evaluation,
        "_EXACT_H2_RUNTIME_VALIDATOR",
        validate_repository,
    )
    with pytest.raises(
        ValueError,
        match="not issued by the frozen scoring factory",
    ):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "prediction-barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=custom_loader,
            evaluation_mode=evaluation.PRODUCTION_EVALUATION_MODE,
            scoring_attestation={"forged": True},
            adapter_repository=tmp_path / "adapter",
        )
    assert target_calls == 0
    assert repository_validations == 1


def test_evaluator_pins_production_barrier_validator_capability() -> None:
    parameters = inspect.signature(
        evaluation.evaluate_adaptive_covariance_confirmation
    ).parameters
    assert "barrier_validator" not in parameters
    assert "target_loader" in parameters


@pytest.mark.parametrize("mutation", ("camera_ids", "route"))
def test_target_loader_cannot_invent_sealed_route_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    barrier_by_case = {
        record["case_id"]: record for record in barrier["ordered_case_seals"]
    }
    target_calls = 0

    def target_loader(
        case_id: str,
        _case_dir: Path,
        _barrier_case: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal target_calls
        target_calls += 1
        value = _outcome(case_id, barrier_by_case[case_id])
        if mutation == "camera_ids":
            value["updates"][0]["attempted_camera_ids"][0] = "unsealed-camera"
        else:
            value["updates"][0] = _updates(("8_view_rbf", "4_view_rbf", "8_view_rbf"))[
                0
            ]
        return value

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        lambda *_args, **_kwargs: barrier,
    )
    with pytest.raises(ValueError, match="differs from the sealed"):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=target_loader,
        )
    assert target_calls == 1


def test_diagnostic_hash_change_after_barrier_opens_no_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    first_case = lock_payload["selected_case_ids"][0]
    diagnostic = case_dirs[first_case] / evaluation.DIAGNOSTIC_FILENAME
    diagnostic.write_bytes(diagnostic.read_bytes() + b"changed-after-barrier")
    target_calls = 0

    def target_loader(*_args: object) -> dict[str, Any]:
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("target must remain closed")

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        lambda *_args, **_kwargs: barrier,
    )
    with pytest.raises(ValueError, match="file hash changed after barrier"):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=target_loader,
        )
    assert target_calls == 0


def test_failed_or_malformed_barrier_opens_no_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    target_calls = 0

    def target_loader(*_args: object) -> dict[str, Any]:
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("target must remain closed")

    def failed_validator(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise ValueError("case seal changed")

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        failed_validator,
    )
    with pytest.raises(ValueError, match="case seal changed"):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=target_loader,
        )
    assert target_calls == 0

    malformed = copy.deepcopy(barrier)
    malformed["case_count"] = 33
    malformed["artifact_sha256"] = evaluation._canonical_sha256(
        malformed,
        digest_key="artifact_sha256",
    )

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        lambda *_args, **_kwargs: malformed,
    )

    with pytest.raises(ValueError, match="case count"):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=target_loader,
        )
    assert target_calls == 0


def test_inexact_case_directory_closure_fails_before_barrier_or_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    case_dirs.pop(lock_payload["selected_case_ids"][-1])
    calls: list[str] = []

    def validator(*_args: object, **_kwargs: object) -> dict[str, Any]:
        calls.append("barrier")
        raise AssertionError

    def loader(*_args: object) -> dict[str, Any]:
        calls.append("target")
        raise AssertionError

    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        validator,
    )
    with pytest.raises(ValueError, match="exact 34-case closure"):
        evaluation.evaluate_adaptive_covariance_confirmation(
            lock_path,
            tmp_path / "barrier.json",
            H2,
            case_dirs,
            expected_h1=H1,
            target_loader=loader,
        )
    assert calls == []


def test_physical_fallback_is_charged_eight_and_cannot_update_state() -> None:
    specification = evaluation._CaseSpec(
        case_id="181-belt-ep0001",
        stratum="filament",
        object_id="181-belt",
        episode_id=1,
    )
    barrier_case = {
        "case_id": specification.case_id,
        "manifest_file_sha256": "0" * 64,
        "manifest_artifact_sha256": "1" * 64,
        "prediction_archive_sha256": "2" * 64,
        "diagnostic_file_sha256": "3" * 64,
        "diagnostic_artifact_sha256": "4" * 64,
    }
    value = _outcome(
        specification.case_id,
        barrier_case,
        routes=(
            "4_view_rbf",
            "8_view_rbf",
            "physical_prior_fallback",
        ),
    )
    normalized = evaluation._normalize_case_outcome(
        value,
        specification=specification,
        barrier_case=barrier_case,
        sealed_diagnostic=_sealed_evidence(
            barrier_case,
            routes=(
                "4_view_rbf",
                "8_view_rbf",
                "physical_prior_fallback",
            ),
        ),
    )
    assert [
        update["counterfactual_policy_charged_camera_streams"]
        for update in normalized["updates"]
    ] == [4, 8, 8]

    broken = copy.deepcopy(value)
    broken["updates"][-1]["rbf_state_updated"] = True
    with pytest.raises(ValueError, match="fallback updated visual state"):
        evaluation._normalize_case_outcome(
            broken,
            specification=specification,
            barrier_case=barrier_case,
            sealed_diagnostic=_sealed_evidence(
                barrier_case,
                routes=(
                    "4_view_rbf",
                    "8_view_rbf",
                    "physical_prior_fallback",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"case_id": "replacement-ep0000"}),
            "changed case ID",
        ),
        (
            lambda value: value.update({"diagnostic_file_sha256": "f" * 64}),
            "diagnostic changed after barrier",
        ),
        (
            lambda value: value["metrics"]["adaptive"].update(
                {evaluation.METRICS[0]: float("nan")}
            ),
            "is invalid",
        ),
        (
            lambda value: value["metrics"]["fixed8"].update(
                {evaluation.METRICS[0]: 0.0}
            ),
            "denominator is zero",
        ),
    ],
)
def test_target_loader_cannot_replace_drop_or_corrupt_a_locked_case(
    mutation: object,
    message: str,
) -> None:
    specification = evaluation._CaseSpec(
        case_id="181-belt-ep0001",
        stratum="filament",
        object_id="181-belt",
        episode_id=1,
    )
    barrier_case = {
        "case_id": specification.case_id,
        "manifest_file_sha256": "0" * 64,
        "manifest_artifact_sha256": "1" * 64,
        "prediction_archive_sha256": "2" * 64,
        "diagnostic_file_sha256": "3" * 64,
        "diagnostic_artifact_sha256": "4" * 64,
    }
    value = _outcome(specification.case_id, barrier_case)
    assert callable(mutation)
    mutation(value)
    with pytest.raises(ValueError, match=message):
        evaluation._normalize_case_outcome(
            value,
            specification=specification,
            barrier_case=barrier_case,
            sealed_diagnostic=_sealed_evidence(barrier_case),
        )


def test_two_episodes_are_averaged_before_equal_object_aggregation(
    tmp_path: Path,
) -> None:
    _lock_path, lock_payload = _write_lock(tmp_path)
    specifications = evaluation._case_specs(lock_payload)
    cases = []
    first_object = specifications[0].object_id
    for index, specification in enumerate(specifications):
        barrier_case = {
            "case_id": specification.case_id,
            "manifest_file_sha256": _sha256(f"{index}:manifest-file"),
            "manifest_artifact_sha256": _sha256(f"{index}:manifest-artifact"),
            "prediction_archive_sha256": _sha256(f"{index}:prediction"),
            "diagnostic_file_sha256": _sha256(f"{index}:diagnostic-file"),
            "diagnostic_artifact_sha256": _sha256(f"{index}:diagnostic-artifact"),
        }
        adaptive = (
            1.0 + index
            if specification.object_id == first_object
            else float(10 + index)
        )
        value = _outcome(
            specification.case_id,
            barrier_case,
            adaptive=adaptive,
            fixed8=20.0,
            fixed4=21.0,
        )
        cases.append(
            evaluation._normalize_case_outcome(
                value,
                specification=specification,
                barrier_case=barrier_case,
                sealed_diagnostic=_sealed_evidence(barrier_case),
            )
        )

    _case_rows, object_rows, object_values, _auxiliary = (
        evaluation._object_and_case_summaries(tuple(cases), specifications)
    )
    first_metric = evaluation.METRICS[0]
    expected_first_object_mean = (
        math.fsum(case["metrics"]["adaptive"][first_metric] for case in cases[:2]) / 2
    )
    assert object_rows[0]["metrics"]["adaptive"][first_metric] == (
        expected_first_object_mean
    )
    assert object_values["metrics"]["adaptive"][first_metric][0] == (
        expected_first_object_mean
    )
    assert (
        len(object_rows)
        == len(object_values["metrics"]["adaptive"][first_metric])
        == 17
    )


def _decision_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    object_values: dict[str, Any] = {
        "metrics": {
            "adaptive": {metric: [0.99] * 17 for metric in evaluation.METRICS},
            "fixed8": {metric: [1.0] * 17 for metric in evaluation.METRICS},
            "fixed4": {metric: [1.02] * 17 for metric in evaluation.METRICS},
        },
        "scale_normalized_difference": {
            metric: [-0.01] * 17 for metric in evaluation.METRICS
        },
        "mean_paired_relative_change": {
            comparator: {metric: [-0.01] * 17 for metric in evaluation.METRICS}
            for comparator in ("fixed8", "fixed4")
        },
        "mean_charged_cameras": [6.4] * 17,
    }
    auxiliary = {
        "routes": {
            "fallback_route_count_including_retained_technical_failures": 25,
        },
        "tails": {
            "joint_fixed8_noninferiority_success_ids": [
                f"object-{index}" for index in range(13)
            ],
            "joint_fixed4_strict_win_ids": [f"object-{index}" for index in range(13)],
            "harmful_object_count": 1,
            "severe_case_count": 0,
            "retained_technical_failure_case_count": 0,
        },
    }
    bootstrap = _point_bootstrap(object_values)
    return object_values, auxiliary, bootstrap


@pytest.mark.parametrize(
    "mutation",
    [
        "identity_ni",
        "sign",
        "camera",
        "fallback",
        "harmful",
        "severe",
        "technical_failure",
    ],
)
def test_every_preregistered_primary_gate_is_binding(mutation: str) -> None:
    object_values, auxiliary, bootstrap = _decision_inputs()
    if mutation == "identity_ni":
        bootstrap["one_sided_upper_95"]["adaptive_vs_fixed8_ratio"][
            evaluation.METRICS[0]
        ] = 1.05
    elif mutation == "sign":
        auxiliary["tails"]["joint_fixed8_noninferiority_success_ids"] = [
            f"object-{index}" for index in range(12)
        ]
    elif mutation == "camera":
        object_values["mean_charged_cameras"] = [6.4000001] * 17
    elif mutation == "fallback":
        auxiliary["routes"][
            "fallback_route_count_including_retained_technical_failures"
        ] = 26
    elif mutation == "harmful":
        auxiliary["tails"]["harmful_object_count"] = 2
    elif mutation == "severe":
        auxiliary["tails"]["severe_case_count"] = 1
    elif mutation == "technical_failure":
        auxiliary["tails"]["retained_technical_failure_case_count"] = 1

    result = evaluation._aggregate(
        object_values=object_values,
        auxiliary=auxiliary,
        bootstrap=bootstrap,
    )
    assert result["primary_confirmation"]["passed"] is False
    assert result["primary_confirmation"]["decision"] == "NOT_CONFIRMED"


def test_all_primary_gate_boundaries_pass_and_fixed4_is_secondary_only() -> None:
    object_values, auxiliary, bootstrap = _decision_inputs()
    result = evaluation._aggregate(
        object_values=object_values,
        auxiliary=auxiliary,
        bootstrap=bootstrap,
    )
    assert result["primary_confirmation"]["passed"] is True
    assert result["fixed4_secondary"]["supported"] is True

    bootstrap["one_sided_upper_95"]["adaptive_vs_fixed4_ratio"][
        evaluation.METRICS[0]
    ] = 1.0
    result = evaluation._aggregate(
        object_values=object_values,
        auxiliary=auxiliary,
        bootstrap=bootstrap,
    )
    assert result["primary_confirmation"]["passed"] is True
    assert result["fixed4_secondary"]["supported"] is False


def test_exact_sign_test_has_preregistered_13_of_17_boundary() -> None:
    passing = evaluation._exact_sign_tail(13)
    failing = evaluation._exact_sign_tail(12)
    assert passing == {
        "success_count": 13,
        "trial_count": 17,
        "null_success_probability": 0.5,
        "tail_numerator": 3214,
        "tail_denominator": 131072,
        "one_sided_p_value": 0.0245208740234375,
        "critical_success_count": 13,
        "passed": True,
    }
    assert failing["tail_numerator"] == 9402
    assert failing["one_sided_p_value"] == 0.0717315673828125
    assert failing["passed"] is False


def test_bootstrap_seed_framing_and_small_index_matrix_known_vector() -> None:
    seed = evaluation.bootstrap_seed_sha256(
        h1_commit=H1,
        h2_commit=H2,
        lock_artifact_sha256="c" * 64,
    )
    independent = hashlib.sha256(
        b"\0".join(
            (
                lock.PROTOCOL_ID.encode(),
                H1.encode(),
                H2.encode(),
                ("c" * 64).encode(),
                b"paired-object-bootstrap-v1-B200000",
            )
        )
    ).hexdigest()
    assert (
        seed
        == independent
        == ("af4a76093c5a65bb2d1cdb73c8af86caf30112db7955ab8a2cf6e5d8967dd7d1")
    )
    indices, matrix_sha256 = evaluation._bootstrap_indices(
        seed,
        replicate_count=4,
        object_count=17,
    )
    assert indices.tolist() == [
        [4, 4, 8, 1, 8, 1, 2, 6, 1, 15, 4, 3, 9, 5, 5, 5, 7],
        [15, 3, 6, 8, 8, 7, 6, 2, 0, 8, 16, 15, 1, 7, 6, 14, 14],
        [11, 6, 15, 15, 3, 11, 15, 1, 3, 11, 5, 7, 14, 9, 11, 14, 9],
        [11, 5, 1, 4, 8, 1, 1, 8, 11, 8, 7, 9, 12, 1, 5, 11, 14],
    ]
    assert matrix_sha256 == (
        "cf2106563dc231400fb592c2a25837709a664d8328348914ee53461a3cb8d614"
    )
    assert evaluation.BOOTSTRAP_REPLICATE_COUNT == 200_000
    assert evaluation.BOOTSTRAP_UPPER_INDEX == 189_999


def test_bootstrap_index_uses_rejection_before_modulo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters: list[int] = []

    def digest(*frames: bytes) -> bytes:
        counter = int.from_bytes(frames[-1], "big")
        counters.append(counter)
        return b"\xff" * 32 if counter == 0 else b"\0" * 32

    monkeypatch.setattr(evaluation, "_framed_sha256_bytes", digest)
    assert (
        evaluation._bootstrap_index(
            "c" * 64,
            0,
            0,
            object_count=17,
        )
        == 0
    )
    assert counters == [0, 1]


def test_bootstrap_analysis_requests_exact_200000_object_cluster_replicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def indices(
        _seed: str,
        *,
        replicate_count: int,
        object_count: int,
    ) -> tuple[object, str]:
        calls.append((replicate_count, object_count))
        return object(), "f" * 64

    monkeypatch.setattr(evaluation, "_bootstrap_indices", indices)
    monkeypatch.setattr(evaluation, "_bootstrap_ratio_upper", lambda *_args: 0.9)
    monkeypatch.setattr(evaluation, "_bootstrap_mean_upper", lambda *_args: 0.0)
    object_values, _auxiliary, _bootstrap = _decision_inputs()
    result = evaluation._bootstrap_analysis(
        seed_sha256="c" * 64,
        object_values=object_values,
    )
    assert calls == [(200_000, 17)]
    assert result["replicate_count"] == 200_000
    assert result["object_count"] == 17


def test_tail_distribution_reports_exact_preregistered_thresholds() -> None:
    result = evaluation._tail_distribution([-0.1, 0.0, 0.051, 0.101, 0.201, 0.251])
    assert result["maximum_relative_change"] == 0.251
    assert result["count_strictly_above"] == {
        "0.05": 4,
        "0.10": 3,
        "0.20": 2,
        "0.25": 1,
    }


def test_result_json_is_finite_and_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, lock_payload = _write_lock(tmp_path)
    barrier = _barrier(lock_path, lock_payload)
    case_dirs = _case_dirs(tmp_path, lock_payload, barrier)
    barrier_by_case = {
        record["case_id"]: record for record in barrier["ordered_case_seals"]
    }
    monkeypatch.setattr(evaluation, "_bootstrap_analysis", _point_bootstrap)
    monkeypatch.setattr(
        evaluation,
        "validate_confirmation_prediction_barrier",
        lambda *_args, **_kwargs: barrier,
    )
    result = evaluation.evaluate_adaptive_covariance_confirmation(
        lock_path,
        tmp_path / "barrier.json",
        H2,
        case_dirs,
        expected_h1=H1,
        target_loader=lambda case_id, _path, _record: _outcome(
            case_id,
            barrier_by_case[case_id],
        ),
    )
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert json.loads(encoded)["result_sha256"] == result["result_sha256"]
