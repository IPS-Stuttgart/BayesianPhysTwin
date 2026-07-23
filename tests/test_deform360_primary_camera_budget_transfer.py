from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_primary_camera_budget_transfer as transfer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sign(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["result_sha256"] = transfer.primary_artifact_sha256(result)
    return result


def _config_fixture(tmp_path: Path) -> dict[str, Any]:
    measurements = {
        str(count): {
            "root": str(tmp_path / f"measurements-{count}"),
            "file_count": 56,
            "total_file_bytes": 1000 + count,
            "inventory_sha256": f"{count}" * 64,
        }
        for count in transfer.CAMERA_COUNTS
    }
    return {
        "schema_version": 1,
        "protocol_id": transfer.PROTOCOL_ID,
        "status": transfer.FROZEN_STATUS,
        "freeze": {
            "date_utc": "2026-07-23",
            "low_view_primary_outcome_status_at_freeze": {
                "2": "not produced or inspected",
                "4": "not produced or inspected",
            },
            "eight_view_compatibility_status_at_freeze": (
                "read-only in-memory parity established and inspected for all "
                "27 cases; no fresh 8-view evaluation root materialized"
            ),
            "saved_parity_artifact_status_at_freeze": "not produced or inspected",
            "method_selection": (
                "exact target-free selected-backbone support-gated ungated RBF "
                "implementation transferred without tuning"
            ),
        },
        "parents": {
            "gated_transfer": {
                "path": transfer.PARENT_TRANSFER_CONFIG,
                "sha256": transfer.PARENT_TRANSFER_CONFIG_SHA256,
            },
            "raw_frontier": {
                "path": transfer.RAW_FRONTIER_CONFIG,
                "sha256": transfer.RAW_FRONTIER_CONFIG_SHA256,
            },
        },
        "implementation": {
            "analyzer_commit": "a" * 40,
            "analyzer_source": transfer.ANALYZER_SOURCE,
            "evaluator_commit": "a" * 40,
            "evaluator_source": transfer.EVALUATOR_SOURCE,
            "held_predictor_commit": transfer.HELD_PREDICTOR_COMMIT,
            "held_predictor_source": transfer.HELD_PREDICTOR_SOURCE,
        },
        "runtime": {
            "python_executable": transfer.RUNTIME_PYTHON,
            "python_version": transfer.RUNTIME_PYTHON_VERSION,
            "numpy_version": transfer.RUNTIME_NUMPY_VERSION,
            "scipy_version": transfer.RUNTIME_SCIPY_VERSION,
            "pip_freeze_all_sorted_sha256": transfer.RUNTIME_PIP_FREEZE_SHA256,
            "gpu_required": False,
            "pythonpath_contract": "clean_clone/src only",
            "isolated_mode_forbidden": True,
            "bytecode_writes_forbidden": True,
        },
        "method": {
            "camera_counts": list(transfer.CAMERA_COUNTS),
            "camera_budget_semantics": (
                "dynamic tracked-view count after all-view frame-zero planning; "
                "not a full sensor-count ablation"
            ),
            "primary_evaluation_protocol_id": (transfer.PRIMARY_EVALUATION_PROTOCOL_ID),
            "parity_protocol_id": transfer.PRIMARY_PARITY_PROTOCOL_ID,
            "gated_reference_protocol_id": transfer.GATED_EVALUATION_PROTOCOL_ID,
            "primary_arm": transfer.PRIMARY_ARM,
            "comparators": list(transfer.COMPARATORS),
            "primary_metrics": list(transfer.PRIMARY_METRICS),
            "minimum_selector_support": transfer.MINIMUM_SELECTOR_SUPPORT,
            "insufficient_support_default": "persistence",
            "covariance_or_cpd_used": False,
        },
        "bound_inputs": {
            "measurements": measurements,
            "gated_8_reference": {
                "root": str(tmp_path / "gated-reference"),
                "file_count": 55,
                "total_file_bytes": 1234,
                "inventory_sha256": "b" * 64,
                "summary_file_sha256": "c" * 64,
                "summary_result_sha256": "d" * 64,
            },
        },
        "outputs": {
            "primary_evaluations": {
                str(count): str(tmp_path / f"primary-{count}")
                for count in transfer.CAMERA_COUNTS
            },
            "execution_root": str(tmp_path / "execution"),
            "parity_artifact": str(tmp_path / "execution" / "cam8-parity.json"),
            "analysis_output": str(tmp_path / "analysis"),
        },
        "excluded_partial_roots": dict(transfer.EXCLUDED_PARTIAL_ROOTS),
        "decision": {
            "candidate_camera_count": 4,
            "reference_camera_count": 8,
            "descriptive_camera_counts": [2],
            "go_if_all": {
                (
                    "minimum_fraction_of_8_view_relative_improvement_retained_"
                    "each_primary_metric"
                ): 0.8,
                "minimum_joint_case_wins_vs_physical": 18,
                (
                    "all_five_object_mean_differences_vs_physical_improve_on_"
                    "both_primary_metrics"
                ): True,
                "maximum_case_chamfer_relative_regression_vs_physical": 0.1,
            },
            "secondary_field_value_check": (
                "report whether the 4-view primary arm improves both aggregate "
                "primary metrics over selected_raw_backbone; this does not "
                "override the four preregistered GO gates"
            ),
            "tie_policy": (
                "ties are not improvements or wins; the maximum regression "
                "bound is inclusive"
            ),
        },
        "claim_boundary": "fixture boundary",
    }


def _score_block(primary_score: float) -> dict[str, dict[str, float]]:
    values = {
        "physical_prior": 1.0,
        "persistence": 0.95,
        transfer.SELECTED_BACKBONE_ARM: 0.9,
        transfer.PRIMARY_ARM: primary_score,
    }
    return {
        arm: {metric: value for metric in transfer.PRIMARY_METRICS}
        for arm, value in values.items()
    }


def _primary_report(
    case: str,
    object_id: str,
    episode_id: int,
    measurement_inventory: transfer.TreeInventory,
    manifest: dict[str, Any],
    archive_sha256: str,
    *,
    primary_score: float = 0.8,
) -> dict[str, Any]:
    report = {
        "protocol_id": transfer.PRIMARY_EVALUATION_PROTOCOL_ID,
        "primary_arm": transfer.PRIMARY_ARM,
        "case": case,
        "object_id": object_id,
        "episode_id": episode_id,
        "measurement_manifest_sha256": measurement_inventory.sha256_by_relative_path[
            f"{case}/measurement_manifest.json"
        ],
        "measurement_archive_sha256": measurement_inventory.sha256_by_relative_path[
            f"{case}/measurement.npz"
        ],
        "measurement_result_sha256": manifest["result_sha256"],
        "prediction_seal_sha256": manifest["inputs"]["prediction_seal"]["sha256"],
        "prediction_archive_sha256": manifest["inputs"]["prediction_archive"]["sha256"],
        "trajectory_archive_sha256": archive_sha256,
        "algorithm_binding": {
            "implementation": "predict_support_gated_selected_backbone_rbf",
            "target_argument_accepted_by_predictor": False,
            "uncertainty_argument_accepted_by_predictor": False,
            "held_rbf_config_required": True,
            "primary_trajectory_scored_without_recomputation": True,
        },
        "center_ids": manifest["plan"]["center_ids"],
        "update_frames": list(transfer.UPDATE_FRAMES),
        "scored_frames": [20, 39, 58],
        "rbf_config": {"local_blend": 1.0},
        "support_gate_contract": {
            "minimum_current_observed_centers": transfer.MINIMUM_SELECTOR_SUPPORT,
            "insufficient_support_default": "persistence",
            "covariance_required": False,
        },
        "observed_backbone_selector": {
            "selected_by_update": ["physical_prior"] * 3,
        },
        "updates": [
            {
                "frame": frame,
                "stop_frame_exclusive": frame + 2,
                "available_center_count": 4,
                "selected_backbone": "physical_prior",
                "selector_support_sufficient": True,
                "selector_decision": "current_observed_center_symmetric_chamfer",
                "current_observation_chamfer_m": {
                    "physical_prior": 0.1,
                    "persistence": 0.2,
                },
                "support_gate": {
                    "accepted": True,
                    "decision": "current_observed_center_symmetric_chamfer",
                    "selected_backbone": "physical_prior",
                    "fallback_backbone": "physical_prior",
                    "rbf_correction_applied": True,
                },
            }
            for frame in transfer.UPDATE_FRAMES
        ],
        "scores": _score_block(primary_score),
        "information_boundary": {
            "measurement_verified_before_target_open": True,
            "primary_prediction_completed_before_target_open": True,
            "measurement_builder_target_read": False,
            "uncertainty_sidecar_read": False,
            "target_visible_covariance_calibration_performed": False,
            "target_role": "scoring only",
        },
    }
    return _sign(report)


def _build_primary_output(
    tmp_path: Path,
    *,
    camera_count: int = 4,
) -> tuple[
    transfer.TreeInventory,
    transfer.TreeInventory,
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, str],
    dict[str, int],
]:
    cases = tuple(f"object-{index}-ep0000" for index in range(5))
    objects = {case: f"object-{index}" for index, case in enumerate(cases)}
    episodes = {case: 0 for case in cases}
    measurement_root = tmp_path / "measurements"
    measurement_root.mkdir()
    manifests: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_root = measurement_root / case
        case_root.mkdir()
        (case_root / "measurement.npz").write_bytes(b"measurement")
        manifest = {
            "result_sha256": hashlib.sha256(case.encode()).hexdigest(),
            "plan": {"center_ids": [0, 1]},
            "inputs": {
                "prediction_seal": {
                    "sha256": hashlib.sha256(f"{case}/seal".encode()).hexdigest(),
                },
                "prediction_archive": {
                    "sha256": hashlib.sha256(f"{case}/archive".encode()).hexdigest(),
                },
            },
        }
        _write_json(case_root / "measurement_manifest.json", manifest)
        manifests[case] = manifest
    measurement_inventory = transfer.inventory_tree(measurement_root)

    output = tmp_path / "primary"
    output.mkdir()
    reports: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, str]] = []
    arrays = {
        arm: np.full((4, 2, 3), index, dtype=np.float64)
        for index, arm in enumerate(transfer.PRIMARY_ARMS)
    }
    for case in cases:
        archive = output / f"{case}.npz"
        np.savez_compressed(archive, **arrays)
        report = _primary_report(
            case,
            objects[case],
            episodes[case],
            measurement_inventory,
            manifests[case],
            _sha256(archive),
        )
        report_path = output / f"{case}.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "case": case,
                "report_sha256": _sha256(report_path),
                "report_result_sha256": report["result_sha256"],
                "archive_sha256": _sha256(archive),
            }
        )
        reports[case] = report
    aggregate = {
        arm: {
            metric: float(
                np.mean([reports[case]["scores"][arm][metric] for case in cases])
            )
            for metric in transfer.PRIMARY_METRICS
        }
        for arm in transfer.PRIMARY_ARMS
    }
    comparisons: dict[str, Any] = {}
    for baseline in transfer.COMPARATORS:
        for metric in transfer.PRIMARY_METRICS:
            differences = {
                case: (
                    reports[case]["scores"][transfer.PRIMARY_ARM][metric]
                    - reports[case]["scores"][baseline][metric]
                )
                for case in cases
            }
            record = transfer._physical_object_cluster_bootstrap(differences, objects)
            record["episode_wins"] = int(
                np.sum(np.asarray(list(differences.values())) < 0.0)
            )
            record["per_object_mean_difference_m"] = {
                objects[case]: differences[case] for case in cases
            }
            record["relative_change"] = (
                aggregate[transfer.PRIMARY_ARM][metric] / aggregate[baseline][metric]
                - 1.0
            )
            comparisons[f"{transfer.PRIMARY_ARM}:vs:{baseline}:{metric}"] = record
    summary = _sign(
        {
            "schema_version": 1,
            "protocol_id": transfer.PRIMARY_EVALUATION_PROTOCOL_ID,
            "episode_count": len(cases),
            "physical_object_count": len(objects),
            "primary_arm": transfer.PRIMARY_ARM,
            "comparators": list(transfer.COMPARATORS),
            "aggregate": aggregate,
            "comparisons": comparisons,
            "information_boundary": {
                "all_measurements_verified_before_any_target_open": True,
                "all_primary_predictions_completed_before_any_target_open": True,
                "uncertainty_sidecars_required": False,
                "target_visible_covariance_calibration_performed": False,
            },
            "artifacts": artifacts,
        }
    )
    _write_json(output / "summary.json", summary)
    return (
        transfer.inventory_tree(output),
        measurement_inventory,
        manifests,
        cases,
        objects,
        episodes,
    )


def test_config_freezes_parents_method_outputs_and_exclusions(tmp_path: Path) -> None:
    config = _config_fixture(tmp_path)

    transfer._validate_config(config)

    assert config["method"]["primary_arm"] == transfer.PRIMARY_ARM
    assert config["decision"]["descriptive_camera_counts"] == [2]
    assert config["excluded_partial_roots"] == transfer.EXCLUDED_PARTIAL_ROOTS


def test_config_rejects_interpreting_a_failed_partial_root(tmp_path: Path) -> None:
    config = _config_fixture(tmp_path)
    config["excluded_partial_roots"]["2"] = str(tmp_path / "partial")

    with pytest.raises(ValueError, match="excluded failed partial roots"):
        transfer._validate_config(config)


def test_config_rejects_fresh_path_nested_under_excluded_partial(
    tmp_path: Path,
) -> None:
    config = _config_fixture(tmp_path)
    config["outputs"]["primary_evaluations"]["2"] = (
        transfer.EXCLUDED_PARTIAL_ROOTS["2"] + "/fresh-primary"
    )

    with pytest.raises(ValueError, match="ancestor overlap"):
        transfer._validate_config(config)


def test_config_rejects_excluded_partial_nested_under_fresh_path(
    tmp_path: Path,
) -> None:
    config = _config_fixture(tmp_path)
    runs_root = str(Path(transfer.EXCLUDED_PARTIAL_ROOTS["2"]).parent)
    config["outputs"]["execution_root"] = runs_root
    config["outputs"]["parity_artifact"] = f"{runs_root}/cam8-parity.json"

    with pytest.raises(ValueError, match="ancestor overlap"):
        transfer._validate_config(config)


def test_pip_freeze_digest_uses_c_byte_order_and_one_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = b"zeta==1\r\nAlpha==2\nbeta==3\n"
    monkeypatch.setattr(
        transfer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout),
    )
    expected = hashlib.sha256(b"Alpha==2\nbeta==3\nzeta==1\n").hexdigest()

    observed = transfer._live_sorted_pip_freeze_sha256("/frozen/python")

    assert observed == expected


def test_primary_output_validator_checks_complete_hash_bound_layout(
    tmp_path: Path,
) -> None:
    inventory, measurements, manifests, cases, objects, episodes = (
        _build_primary_output(tmp_path)
    )

    summary, reports, arrays = transfer._validate_primary_evaluation_root(
        inventory,
        measurements,
        manifests,
        cases,
        objects,
        episodes,
        camera_count=4,
    )

    assert summary["episode_count"] == 5
    assert set(reports) == set(cases)
    assert set(arrays[cases[0]]) == set(transfer.PRIMARY_ARMS)


def test_primary_output_validator_rejects_extra_artifact(tmp_path: Path) -> None:
    inventory, measurements, manifests, cases, objects, episodes = (
        _build_primary_output(tmp_path)
    )
    (inventory.root / "unbound.txt").write_text("unexpected", encoding="utf-8")
    changed = transfer.inventory_tree(inventory.root)

    with pytest.raises(ValueError, match="file layout changed"):
        transfer._validate_primary_evaluation_root(
            changed,
            measurements,
            manifests,
            cases,
            objects,
            episodes,
            camera_count=4,
        )


def test_primary_output_validator_rejects_different_sealed_prior(
    tmp_path: Path,
) -> None:
    inventory, measurements, manifests, cases, objects, episodes = (
        _build_primary_output(tmp_path)
    )
    case = cases[0]
    report_path = inventory.root / f"{case}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["prediction_archive_sha256"] = "f" * 64
    report = _sign(report)
    _write_json(report_path, report)
    summary_path = inventory.root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = next(record for record in summary["artifacts"] if record["case"] == case)
    artifact["report_sha256"] = _sha256(report_path)
    artifact["report_result_sha256"] = report["result_sha256"]
    summary = _sign(summary)
    _write_json(summary_path, summary)
    changed = transfer.inventory_tree(inventory.root)

    with pytest.raises(ValueError, match="different sealed prior"):
        transfer._validate_primary_evaluation_root(
            changed,
            measurements,
            manifests,
            cases,
            objects,
            episodes,
            camera_count=4,
        )


def _saved_parity_fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    transfer.TreeInventory,
    tuple[str, ...],
    Path,
]:
    config = _config_fixture(tmp_path)
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_summary = _sign(
        {
            "schema_version": 1,
            "protocol_id": transfer.GATED_EVALUATION_PROTOCOL_ID,
        }
    )
    _write_json(reference_root / "summary.json", reference_summary)
    reference_inventory = transfer.inventory_tree(reference_root)
    parity_path = Path(config["outputs"]["parity_artifact"])
    parity_path.parent.mkdir()
    cases = tuple(f"case-{index:02d}" for index in range(27))
    read_only_contract = {
        "input_artifacts_written": False,
        "output_artifacts_written": False,
        "reference_hashes_verified_before_comparison": True,
        "reference_summary_externally_bound": True,
        "reference_case_files_loaded_from_hash_verified_bytes": True,
        "all_measurements_verified_before_any_target_open": True,
        "all_primary_predictions_completed_before_any_target_open": True,
        "primary_input_hashes_rechecked_at_parity_completion": True,
    }
    case_records = []
    for case in cases:
        trajectory_hashes = {
            arm: hashlib.sha256(f"{case}/{arm}".encode()).hexdigest()
            for arm in transfer.PRIMARY_ARMS
        }
        case_records.append(
            {
                "case": case,
                "all_primary_arrays_byte_exact": True,
                "all_exact_metadata_equal": True,
                "all_support_semantics_equivalent": True,
                "parity_passed": True,
                "trajectory_bit_exact": {arm: True for arm in transfer.PRIMARY_ARMS},
                "metadata_exact": {
                    "center_ids": True,
                    "update_frames": True,
                    "scored_frames": True,
                    "rbf_config": True,
                    "observed_backbone_selector_normalized": True,
                },
                "score_within_absolute_tolerance": {
                    arm: True for arm in transfer.PRIMARY_ARMS
                },
                "score_absolute_tolerance": 1.0e-12,
                "updates": [
                    {
                        "frame": frame,
                        "selection_metadata_bit_exact": True,
                        "canonical_support_decision": (
                            "current_observed_center_symmetric_chamfer"
                        ),
                        "legacy_reference_selector_decision": (
                            "current_observation_chamfer"
                        ),
                        "legacy_reference_gate_decision": (
                            "accepted_without_covariance_gate"
                        ),
                        "support_semantics_equivalent": True,
                    }
                    for frame in transfer.UPDATE_FRAMES
                ],
                "reference_trajectory_sha256": trajectory_hashes,
                "primary_trajectory_sha256": trajectory_hashes,
            }
        )
    parity = _sign(
        {
            "schema_version": 1,
            "protocol_id": transfer.PRIMARY_PARITY_PROTOCOL_ID,
            "reference_protocol_id": transfer.GATED_EVALUATION_PROTOCOL_ID,
            "reference_summary_binding": {
                "file_sha256": reference_inventory.sha256_by_relative_path[
                    "summary.json"
                ],
                "result_sha256": reference_summary["result_sha256"],
            },
            "episode_count": 27,
            "all_27_cases_primary_arrays_byte_exact": True,
            "all_27_cases_parity_passed": True,
            "parity_passed": True,
            "cases": case_records,
            "read_only_contract": read_only_contract,
        }
    )
    _write_json(parity_path, parity)
    return config, reference_summary, reference_inventory, cases, parity_path


def test_saved_parity_requires_complete_passing_evaluator_artifact(
    tmp_path: Path,
) -> None:
    config, reference_summary, reference_inventory, cases, parity_path = (
        _saved_parity_fixture(tmp_path)
    )

    result = transfer._validate_saved_parity(
        config,
        reference_summary,
        reference_inventory,
        cases,
    )

    assert result["parity_passed"] is True
    assert (
        result["result_sha256"]
        == json.loads(parity_path.read_text(encoding="utf-8"))["result_sha256"]
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing_case", "duplicate_case", "failing_case", "altered_contract"),
)
def test_saved_parity_rejects_incomplete_or_weakened_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, reference_summary, reference_inventory, cases, parity_path = (
        _saved_parity_fixture(tmp_path)
    )
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if mutation == "missing_case":
        parity["cases"].pop()
    elif mutation == "duplicate_case":
        parity["cases"][-1]["case"] = parity["cases"][0]["case"]
    elif mutation == "failing_case":
        parity["cases"][0]["parity_passed"] = False
    else:
        parity["read_only_contract"]["output_artifacts_written"] = True
    _write_json(parity_path, _sign(parity))

    with pytest.raises(ValueError, match="saved parity"):
        transfer._validate_saved_parity(
            config,
            reference_summary,
            reference_inventory,
            cases,
        )


def _parity_report(*, legacy_zero_support: bool) -> dict[str, Any]:
    updates = []
    for frame in transfer.UPDATE_FRAMES:
        record = {
            "frame": frame,
            "stop_frame_exclusive": frame + 2,
            "available_center_count": 0,
            "selected_backbone": "persistence",
            "selector_support_sufficient": False,
            "selector_decision": (
                "insufficient_support_persistence_default"
                if legacy_zero_support
                else "insufficient_support_persistence"
            ),
            "current_observation_chamfer_m": {
                "physical_prior": None,
                "persistence": None,
            },
        }
        if legacy_zero_support:
            record["gates"] = {
                "ungated": {
                    "accepted": False,
                    "decision": "insufficient_valid_covariance",
                    "selected_backbone": "persistence",
                    "fallback_backbone": "persistence",
                    "rbf_correction_applied": False,
                }
            }
        else:
            record["support_gate"] = {
                "accepted": False,
                "decision": "insufficient_support_persistence",
                "selected_backbone": "persistence",
                "fallback_backbone": "persistence",
                "rbf_correction_applied": False,
            }
        updates.append(record)
    return {
        "center_ids": [0, 1],
        "update_frames": list(transfer.UPDATE_FRAMES),
        "scored_frames": [20, 39, 58],
        "rbf_config": {"local_blend": 1.0},
        "observed_backbone_selector": {"selected_by_update": ["persistence"] * 3},
        "updates": updates,
        "scores": _score_block(0.8),
    }


def test_independent_parity_normalizes_only_legacy_zero_support_label() -> None:
    case = "fixture"
    fresh = {case: _parity_report(legacy_zero_support=False)}
    reference = {case: _parity_report(legacy_zero_support=True)}
    arrays = {
        case: {
            arm: np.ones((2, 2, 3), dtype=np.float64) for arm in transfer.PRIMARY_ARMS
        }
    }

    result = transfer._independent_eight_view_parity(
        (case,),
        fresh,
        arrays,
        reference,
        arrays,
    )

    assert result["parity_passed"] is True
    assert result["all_primary_arrays_bit_exact"] is True


def test_independent_parity_rejects_trajectory_drift() -> None:
    case = "fixture"
    fresh_report = {case: _parity_report(legacy_zero_support=False)}
    reference_report = {case: _parity_report(legacy_zero_support=True)}
    fresh_arrays = {
        case: {
            arm: np.ones((2, 2, 3), dtype=np.float64) for arm in transfer.PRIMARY_ARMS
        }
    }
    reference_arrays = {
        case: {arm: value.copy() for arm, value in fresh_arrays[case].items()}
    }
    reference_arrays[case][transfer.PRIMARY_ARM][0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="differs from frozen gated reference"):
        transfer._independent_eight_view_parity(
            (case,),
            fresh_report,
            fresh_arrays,
            reference_report,
            reference_arrays,
        )


def _all_go_analyses() -> dict[int, dict[str, Any]]:
    cases = tuple(f"case-{index}" for index in range(20))
    objects = {case: f"object-{index % 5}" for index, case in enumerate(cases)}

    def reports(primary: float) -> dict[str, dict[str, Any]]:
        return {case: {"scores": _score_block(primary)} for case in cases}

    return {
        2: transfer._budget_analysis(2, reports(0.9), cases, objects),
        4: transfer._budget_analysis(4, reports(0.83), cases, objects),
        8: transfer._budget_analysis(8, reports(0.8), cases, objects),
    }


def test_four_view_gate_uses_selected_primary_and_keeps_two_descriptive() -> None:
    analyses = _all_go_analyses()

    decision = transfer._four_view_decision(analyses)

    assert decision["status"] == "GO"
    assert decision["two_view_role"].startswith("descriptive_only")
    assert (
        decision["secondary_field_value_vs_selected_raw_backbone"][
            "overrides_primary_decision"
        ]
        is False
    )


@pytest.mark.parametrize("metric", transfer.PRIMARY_METRICS)
def test_either_retention_metric_independently_forces_no_go(metric: str) -> None:
    analyses = _all_go_analyses()
    analyses[4]["comparisons"]["physical_prior"]["metrics"][metric][
        "relative_improvement"
    ] = 0.15

    decision = transfer._four_view_decision(analyses)

    retention = decision["retains_at_least_80_percent_of_8_view_relative_improvement"]
    assert decision["status"] == "NO_GO"
    assert retention[metric]["passed"] is False
    assert decision["joint_case_wins_vs_physical"]["passed"] is True
    assert decision["all_five_objects_improve_on_both_primary_metrics"]["passed"]
    assert decision["case_chamfer_regression_vs_physical"]["passed"] is True


def test_joint_win_gate_independently_forces_no_go() -> None:
    analyses = _all_go_analyses()
    analyses[4]["comparisons"]["physical_prior"]["joint_case_wins"] = 17

    decision = transfer._four_view_decision(analyses)

    assert decision["status"] == "NO_GO"
    assert decision["joint_case_wins_vs_physical"]["passed"] is False
    assert all(
        record["passed"]
        for record in decision[
            "retains_at_least_80_percent_of_8_view_relative_improvement"
        ].values()
    )
    assert decision["all_five_objects_improve_on_both_primary_metrics"]["passed"]
    assert decision["case_chamfer_regression_vs_physical"]["passed"] is True


@pytest.mark.parametrize("metric", transfer.PRIMARY_METRICS)
def test_any_one_object_metric_non_improvement_independently_forces_no_go(
    metric: str,
) -> None:
    analyses = _all_go_analyses()
    physical = analyses[4]["comparisons"]["physical_prior"]
    object_id = next(iter(physical["metrics"][metric]["per_object"]))
    physical["metrics"][metric]["per_object"][object_id]["mean_difference_m"] = 0.0

    decision = transfer._four_view_decision(analyses)

    objects = decision["all_five_objects_improve_on_both_primary_metrics"]
    assert decision["status"] == "NO_GO"
    assert objects["passed"] is False
    assert objects["objects"][object_id]["passed"] is False
    assert all(
        record["passed"]
        for record in decision[
            "retains_at_least_80_percent_of_8_view_relative_improvement"
        ].values()
    )
    assert decision["joint_case_wins_vs_physical"]["passed"] is True
    assert decision["case_chamfer_regression_vs_physical"]["passed"] is True


def test_maximum_case_chamfer_regression_independently_forces_no_go() -> None:
    analyses = _all_go_analyses()
    chamfer = transfer.PRIMARY_METRICS[1]
    relative = analyses[4]["comparisons"]["physical_prior"]["metrics"][chamfer][
        "per_case_relative_change"
    ]
    relative[next(iter(relative))] = 0.1000000001

    decision = transfer._four_view_decision(analyses)

    assert decision["status"] == "NO_GO"
    assert decision["case_chamfer_regression_vs_physical"]["passed"] is False
    assert all(
        record["passed"]
        for record in decision[
            "retains_at_least_80_percent_of_8_view_relative_improvement"
        ].values()
    )
    assert decision["joint_case_wins_vs_physical"]["passed"] is True
    assert decision["all_five_objects_improve_on_both_primary_metrics"]["passed"]


def test_two_view_mutations_cannot_affect_the_frozen_decision() -> None:
    analyses = _all_go_analyses()
    expected = transfer._four_view_decision(analyses)
    mutated = copy.deepcopy(analyses)
    mutated[2] = {
        "arbitrary": {
            "outcomes": [float("inf"), float("-inf"), float("nan")],
            "decision": "NO_GO",
        }
    }

    observed = transfer._four_view_decision(mutated)

    assert observed == expected
    assert observed["status"] == "GO"


def test_publish_is_self_hashed_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "analysis"
    report = {"schema_version": 1, "protocol_id": "fixture"}

    path = transfer._publish_report(report, destination)

    stored = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(stored)
    claimed = unsigned.pop("result_sha256")
    assert claimed == transfer._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="already exists"):
        transfer._publish_report(report, destination)
