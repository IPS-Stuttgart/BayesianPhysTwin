from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.deform360_raw_camera_budget_frontier as frontier


PRODUCTION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "sota"
    / "deform360_raw_camera_budget_frontier_v1_development.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _case_metadata(config: dict) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    for object_id, episodes in config["panel"]["object_episodes"].items():
        for episode in episodes:
            result.append((f"{object_id}-ep{episode:04d}", object_id, episode))
    return result


def _build_budget(
    config: dict,
    tmp_path: Path,
    camera_count: int,
    *,
    primary_score: float,
    physical_score: float = 1.0,
    centers: list[int] | None = None,
) -> None:
    measurement_root = tmp_path / f"measurements-{camera_count}"
    evaluation_root = tmp_path / f"evaluation-{camera_count}"
    config["roots"][str(camera_count)] = {
        "measurement": str(measurement_root),
        "evaluation": str(evaluation_root),
    }
    measurement_root.mkdir()
    evaluation_root.mkdir()
    case_metadata = _case_metadata(config)
    fixed_centers = list(range(16)) if centers is None else centers
    candidate_ids = list(range(32))
    manifest_hashes: dict[str, str] = {}
    reports: dict[str, dict] = {}
    artifacts: list[dict[str, str]] = []
    expected_config = dict(
        config["observation"]["raw_camera_config_except_selected_camera_count"]
    )
    expected_config["selected_camera_count"] = camera_count

    for case, object_id, episode_id in case_metadata:
        measurement_case = measurement_root / case
        measurement_case.mkdir()
        selected_cameras = [f"camera-{index:02d}" for index in range(camera_count)]
        processed_case = (
            Path(config["observation"]["source_roots"]["processed"])
            / case
            / "episode_0000"
        )
        panel_case = Path(config["observation"]["source_roots"]["panel"]) / case
        selected_camera_inputs = {
            camera: {
                "video": {
                    "path": str(processed_case / camera / "undistorted.mp4"),
                    "decoded_prefix_sha256_by_update": {
                        str(frame): _fake_sha256(f"{case}/{camera}/rgb/{frame}")
                        for frame in frontier.UPDATE_FRAMES
                    },
                    "whole_file_hashed_or_read": False,
                },
                "frame_zero_mask": {
                    "path": str(processed_case / camera / "mask_refined.h5"),
                    "frame_zero_array_sha256": _fake_sha256(f"{case}/{camera}/mask/0"),
                    "only_index_read": 0,
                    "whole_file_hashed_or_read": False,
                },
                "frame_zero_depth": {
                    "path": str(processed_case / camera / "rendered_depth.h5"),
                    "frame_zero_array_sha256": _fake_sha256(f"{case}/{camera}/depth/0"),
                    "only_index_read": 0,
                    "whole_file_hashed_or_read": False,
                },
            }
            for camera in selected_cameras
        }
        archive_path = measurement_case / "measurement.npz"
        np.savez_compressed(
            archive_path,
            candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
            center_ids=np.asarray(fixed_centers, dtype=np.int64),
            selected_cameras=np.asarray(selected_cameras),
            update_frames=np.asarray(frontier.UPDATE_FRAMES, dtype=np.int64),
        )
        manifest = {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalRawCameraMeasurement",
            "protocol_id": frontier.MEASUREMENT_PROTOCOL_ID,
            "case": case,
            "object_id": object_id,
            "episode_id": episode_id,
            "episode_key": f"{object_id}/{episode_id}",
            "config": expected_config,
            "plan": {
                "candidate_count": len(candidate_ids),
                "candidate_ids": candidate_ids,
                "center_ids": fixed_centers,
                "selected_cameras": selected_cameras,
                "selection_score": [16, 16, camera_count * 16, 90.0],
                "selection_inputs": (
                    "sealed frame-zero points, calibration, and HDF5 index zero only"
                ),
            },
            "tracker": dict(config["observation"]["tracker"]),
            "inputs": {
                "prediction_seal": {
                    "path": str(panel_case / "prediction_seal.json"),
                    "sha256": _fake_sha256(f"{case}/prediction_seal"),
                },
                "prediction_archive": {
                    "path": str(panel_case / "prediction.npz"),
                    "sha256": _fake_sha256(f"{case}/prediction_archive"),
                },
                "intrinsics": {
                    "path": str(processed_case / "undistorted_intrinsics.npy"),
                    "sha256": _fake_sha256(f"{case}/intrinsics"),
                },
                "extrinsics": {
                    "path": str(processed_case / "extrinsics.npy"),
                    "sha256": _fake_sha256(f"{case}/extrinsics"),
                },
            },
            "selected_camera_inputs": selected_camera_inputs,
            "updates": [
                {
                    "frame": frame,
                    "maximum_video_frame_read": frame,
                    "prefix_frame_range_half_open": [0, frame + 1],
                    "tracker": [
                        {
                            "camera": camera,
                            "maximum_video_frame_read": frame,
                            "prefix_frame_range_half_open": [0, frame + 1],
                            "decoded_rgb_prefix_sha256": selected_camera_inputs[camera][
                                "video"
                            ]["decoded_prefix_sha256_by_update"][str(frame)],
                        }
                        for camera in selected_cameras
                    ],
                }
                for frame in frontier.UPDATE_FRAMES
            ],
            "output": {
                "measurement_archive_sha256": _sha256(archive_path),
            },
            "information_boundary": {
                "target_data_read": False,
                "outcome_manifest_read": False,
                "future_reconstruction_after_frame_zero_read": False,
                "maximum_video_frame_read_by_update": list(frontier.UPDATE_FRAMES),
                "frame_zero_hdf5_indices_read": [0],
            },
        }
        manifest["result_sha256"] = frontier._canonical_sha256(manifest)
        manifest_path = measurement_case / "measurement_manifest.json"
        _write_json(manifest_path, manifest)
        manifest_hashes[case] = _sha256(manifest_path)

        scores = {
            frontier.PRIMARY_ARM: {
                metric: primary_score for metric in frontier.PRIMARY_METRICS
            },
            "physical_prior": {
                metric: physical_score for metric in frontier.PRIMARY_METRICS
            },
            "persistence": {metric: 0.95 for metric in frontier.PRIMARY_METRICS},
        }
        report = {
            "protocol_id": frontier.MEASUREMENT_PROTOCOL_ID,
            "case": case,
            "object_id": object_id,
            "episode_id": episode_id,
            "measurement_manifest_sha256": _sha256(manifest_path),
            "measurement_archive_sha256": _sha256(archive_path),
            "measurement_result_sha256": manifest["result_sha256"],
            frontier.RAW_STREAM: {
                "protocol_id": "fixture-fixed-raw-belief",
                "center_count": 16,
                "center_ids": fixed_centers,
                "update_frames": list(frontier.UPDATE_FRAMES),
                "scored_frames": [20, 39, 58],
                "belief_config": {"fixed": True},
                "metric_contract": {"fixed": True},
                "updates": [
                    {"frame": frame, "available_center_count": 16, "accepted": True}
                    for frame in frontier.UPDATE_FRAMES
                ],
                "scores": scores,
            },
            "same_support_target_oracle": {"not_used": True},
            "observation_error_by_update": [],
            "information_boundary": {
                "measurement_hashed_before_target_open_in_this_evaluator": True,
                "measurement_builder_target_read": False,
            },
        }
        report_path = evaluation_root / f"{case}.json"
        arrays_path = evaluation_root / f"{case}.npz"
        _write_json(report_path, report)
        np.savez_compressed(arrays_path, primary=np.asarray([primary_score]))
        artifacts.append(
            {
                "case": case,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
        reports[case] = report

    cases = [case for case, _, _ in case_metadata]
    for shard_index in range(frontier.SHARD_COUNT):
        shard_cases = cases[shard_index :: frontier.SHARD_COUNT]
        _write_json(
            measurement_root / f"build-shard-{shard_index:02d}.json",
            {
                "protocol_id": frontier.MEASUREMENT_PROTOCOL_ID,
                "shard_count": frontier.SHARD_COUNT,
                "shard_index": shard_index,
                "case_count": len(shard_cases),
                "cases": shard_cases,
                "measurement_manifest_sha256": {
                    case: manifest_hashes[case] for case in shard_cases
                },
            },
        )

    aggregate = {
        frontier.RAW_STREAM: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            reports[case][frontier.RAW_STREAM]["scores"][arm][metric]
                            for case in cases
                        ]
                    )
                )
                for metric in frontier.PRIMARY_METRICS
            }
            for arm in (frontier.PRIMARY_ARM, *frontier.COMPARATORS)
        }
    }
    summary = {
        "schema_version": 1,
        "protocol_id": frontier.MEASUREMENT_PROTOCOL_ID,
        "episode_count": 27,
        "physical_object_count": 5,
        "aggregate": aggregate,
        "artifacts": artifacts,
        "claim_boundary": "fixture",
    }
    summary["result_sha256"] = frontier._canonical_sha256(summary)
    _write_json(evaluation_root / "summary.json", summary)


def _fixture_config(
    tmp_path: Path,
    *,
    primary_by_budget: dict[int, float] | None = None,
    physical_by_budget: dict[int, float] | None = None,
    centers_by_budget: dict[int, list[int]] | None = None,
) -> Path:
    config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    primary = primary_by_budget or {2: 0.9, 4: 0.83, 8: 0.8}
    physical = physical_by_budget or {2: 1.0, 4: 1.0, 8: 1.0}
    centers = centers_by_budget or {}
    for camera_count in frontier.CAMERA_COUNTS:
        _build_budget(
            config,
            tmp_path,
            camera_count,
            primary_score=primary[camera_count],
            physical_score=physical[camera_count],
            centers=centers.get(camera_count),
        )
    measurement_8 = frontier.inventory_tree(config["roots"]["8"]["measurement"])
    evaluation_8 = frontier.inventory_tree(config["roots"]["8"]["evaluation"])
    config["bound_existing_8_view_baseline"] = {
        "measurement": {
            "file_count": measurement_8.file_count,
            "total_file_bytes": measurement_8.total_file_bytes,
            "inventory_sha256": measurement_8.inventory_sha256,
        },
        "evaluation": {
            "file_count": evaluation_8.file_count,
            "total_file_bytes": evaluation_8.total_file_bytes,
            "inventory_sha256": evaluation_8.inventory_sha256,
            "summary_sha256": evaluation_8.sha256_by_relative_path["summary.json"],
        },
    }
    config["roots"]["frontier_output"] = str(tmp_path / "frontier")
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return config_path


def test_production_preregistration_freezes_fair_older_schema() -> None:
    config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))

    frontier._validate_config(config)

    assert config["evaluation"]["primary_arm"] == "recursive_rbf_ungated"
    assert config["evaluation"]["raw_cpd"]["included"] is False
    assert config["observation"]["camera_counts"] == [2, 4, 8]
    assert config["observation"]["center_count"] == 16
    assert config["observation"]["update_frames"] == [19, 38, 57]
    assert config["observation"]["budget_semantics"] == {
        "all_available_cameras_used_for_frame_zero_planning": True,
        "tracked_views_after_planning_are_budgeted": True,
        "full_sensor_count_ablation": False,
    }
    assert config["observation"]["tracker"]["device"] == "cuda:0"
    assert (
        config["observation"]["raw_camera_config_except_selected_camera_count"][
            "alltracker_max_side"
        ]
        == 512
    )
    assert (
        config["bound_existing_8_view_baseline"]["evaluation"]["summary_sha256"]
        == "92188a6728b4cca104aff76db386d9f1c4bae79dd307786bbb3a3e64797a874d"
    )


def test_inventory_matches_nul_record_contract_and_mode(tmp_path: Path) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    first = root / "b.bin"
    second = root / "a.txt"
    first.write_bytes(b"\x00\x01")
    second.write_bytes(b"abc")
    first.chmod(0o600)
    second.chmod(0o640)
    records = []
    for path in (second, first):
        relative = path.relative_to(root).as_posix()
        records.append(
            f"{relative}\0{path.stat().st_mode & 0o777:o}\0"
            f"{path.stat().st_size}\0{_sha256(path)}\0".encode()
        )
    expected = hashlib.sha256(b"".join(records)).hexdigest()

    observed = frontier.inventory_tree(root)

    assert observed.inventory_sha256 == expected
    assert observed.file_count == 2
    assert observed.total_file_bytes == 5


def test_analyzer_publishes_go_and_object_balanced_report(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)

    result = frontier.analyze_camera_budget_frontier(config_path)

    assert result["decision"]["status"] == "GO"
    assert result["decision"]["joint_case_wins_vs_physical"]["observed"] == 27
    retention = result["decision"][
        "retains_at_least_80_percent_of_8_view_relative_improvement"
    ]
    assert all(record["passed"] for record in retention.values())
    objects = result["decision"]["all_five_objects_improve_on_both_primary_metrics"]
    assert objects["passed"] is True
    assert len(objects["objects"]) == 5
    assert result["budgets"]["2"]["role"] == "descriptive_only"
    assert result["method"]["raw_cpd_comparison_performed"] is False
    output = tmp_path / "frontier" / "frontier.json"
    assert output.is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    unsigned = dict(persisted)
    claimed = unsigned.pop("result_sha256")
    assert claimed == frontier._canonical_sha256(unsigned)


def test_analyzer_emits_no_go_when_four_view_retention_fails(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(
        tmp_path,
        primary_by_budget={2: 0.9, 4: 0.95, 8: 0.8},
    )

    result = frontier.analyze_camera_budget_frontier(config_path)

    assert result["decision"]["status"] == "NO_GO"
    retention = result["decision"][
        "retains_at_least_80_percent_of_8_view_relative_improvement"
    ]
    assert all(record["passed"] is False for record in retention.values())
    assert result["decision"]["joint_case_wins_vs_physical"]["passed"] is True


def test_decision_rejects_case_chamfer_regression_above_ten_percent(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = frontier.build_frontier_report(config)
    analyses = {int(key): value for key, value in report["budgets"].items()}
    chamfer = frontier.PRIMARY_METRICS[1]
    relative = analyses[4]["comparisons"]["physical_prior"]["metrics"][chamfer][
        "per_case_relative_change"
    ]
    relative[next(iter(relative))] = 0.1000000001

    decision = frontier._four_view_decision(analyses)

    assert decision["status"] == "NO_GO"
    assert decision["case_chamfer_regression_vs_physical"]["passed"] is False


def test_decision_requires_each_object_to_improve_both_metrics(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = frontier.build_frontier_report(config)
    analyses = {int(key): value for key, value in report["budgets"].items()}
    metric = frontier.PRIMARY_METRICS[0]
    per_object = analyses[4]["comparisons"]["physical_prior"]["metrics"][metric][
        "per_object"
    ]
    per_object[next(iter(per_object))]["mean_difference_m"] = 0.0

    decision = frontier._four_view_decision(analyses)

    assert decision["status"] == "NO_GO"
    assert (
        decision["all_five_objects_improve_on_both_primary_metrics"]["passed"] is False
    )


def test_analyzer_rejects_center_drift_across_camera_budgets(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(
        tmp_path,
        centers_by_budget={4: list(range(1, 17))},
    )

    with pytest.raises(ValueError, match="center IDs differ across"):
        frontier.analyze_camera_budget_frontier(config_path)


def test_analyzer_rejects_camera_variant_physical_comparator(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(
        tmp_path,
        physical_by_budget={2: 1.0, 4: 1.01, 8: 1.0},
    )

    with pytest.raises(ValueError, match="physical_prior changed across"):
        frontier.analyze_camera_budget_frontier(config_path)


def _minimal_cross_budget_fixture() -> tuple[
    dict[int, dict[str, dict]],
    dict[int, dict[str, dict]],
    tuple[str, ...],
]:
    case = "fixture-case"
    common_camera = {
        "video": {
            "decoded_prefix_sha256_by_update": {
                str(frame): _fake_sha256(f"rgb/{frame}")
                for frame in frontier.UPDATE_FRAMES
            }
        }
    }
    base_manifest = {
        "inputs": {
            "intrinsics": {"sha256": _fake_sha256("intrinsics")},
        },
        "plan": {
            "candidate_count": 16,
            "candidate_ids": list(range(16)),
            "center_ids": list(range(16)),
            "selection_inputs": "fixed",
        },
        "tracker": {"device": "cuda:0"},
        "selected_camera_inputs": {"camera-00": common_camera},
    }
    score_block = {
        arm: {metric: 1.0 for metric in frontier.PRIMARY_METRICS}
        for arm in frontier.COMPARATORS
    }
    report = {
        frontier.RAW_STREAM: {
            "belief_config": {"fixed": True},
            "metric_contract": {"fixed": True},
            "scored_frames": [20, 39, 58],
            "scores": score_block,
        }
    }
    manifests = {
        count: {case: json.loads(json.dumps(base_manifest))}
        for count in frontier.CAMERA_COUNTS
    }
    reports = {
        count: {case: json.loads(json.dumps(report))}
        for count in frontier.CAMERA_COUNTS
    }
    return manifests, reports, (case,)


def test_cross_budget_check_rejects_immutable_input_drift() -> None:
    manifests, reports, cases = _minimal_cross_budget_fixture()
    manifests[4][cases[0]]["inputs"]["intrinsics"]["sha256"] = "1" * 64

    with pytest.raises(ValueError, match="immutable inputs differ"):
        frontier._exact_cross_budget_checks(manifests, reports, cases)


def test_cross_budget_check_rejects_tracker_device_drift() -> None:
    manifests, reports, cases = _minimal_cross_budget_fixture()
    manifests[4][cases[0]]["tracker"]["device"] = "cuda:1"

    with pytest.raises(ValueError, match="tracker execution differs"):
        frontier._exact_cross_budget_checks(manifests, reports, cases)


def test_cross_budget_check_rejects_overlapping_camera_byte_drift() -> None:
    manifests, reports, cases = _minimal_cross_budget_fixture()
    manifests[4][cases[0]]["selected_camera_inputs"]["camera-00"]["video"][
        "decoded_prefix_sha256_by_update"
    ]["19"] = "2" * 64

    with pytest.raises(ValueError, match="camera input bytes differ"):
        frontier._exact_cross_budget_checks(manifests, reports, cases)


def test_analyzer_rejects_changed_bound_8_view_inventory(
    tmp_path: Path,
) -> None:
    config_path = _fixture_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["bound_existing_8_view_baseline"]["measurement"]["inventory_sha256"] = (
        "0" * 64
    )
    _write_json(config_path, config)

    with pytest.raises(ValueError, match="bound 8-view measurement inventory"):
        frontier.analyze_camera_budget_frontier(config_path)


def test_config_rejects_raw_cpd_method_change() -> None:
    config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    config["evaluation"]["raw_cpd"]["included"] = True

    with pytest.raises(ValueError, match="raw CPD inclusion changed"):
        frontier._validate_config(config)
