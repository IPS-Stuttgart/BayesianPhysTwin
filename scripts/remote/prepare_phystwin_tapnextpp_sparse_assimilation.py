#!/usr/bin/env python3
"""Stage the frozen opened-source TAPNext++ sparse-assimilation study."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_tapnextpp_competence import (
    canonical_sha256,
    file_sha256,
)

SOURCE_MANIFEST_FILENAME = "tapnextpp_sparse_assimilation_source_manifest.json"
PREDICTION_INPUT_FILENAME = "prediction_input.npz"
WITHHELD_OUTCOME_FILENAME = "withheld_future_outcome.npz"
PROVIDER_PREDICTION_FILENAME = "tapnextpp_depth_completion_prediction.npz"
PROVIDER_REPORT_FILENAME = "tapnextpp_depth_completion_prediction_report.json"
PROVIDER_SEAL_FILENAME = "tapnextpp_depth_completion_prediction_seal.json"
PROVIDER_RESULT_FILENAME = "tapnextpp_depth_completion_transfer_result.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_protocol(protocol: dict[str, Any]) -> None:
    _require(
        protocol.get("protocol_id")
        == "phystwin-tapnextpp-sparse-assimilation-source-v1",
        "assimilation protocol ID changed",
    )
    _require(
        protocol.get("status") == "locked-before-future-assimilation-outcome",
        "assimilation protocol is not prediction-locked",
    )
    cases = protocol.get("fixed_source_cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 8
        and len(set(cases)) == 8,
        "assimilation case panel changed",
    )
    _require(
        protocol.get("information_boundary", {}).get("held_v8_accessed") is False,
        "held-v8 boundary changed",
    )
    _require(
        protocol.get("retention_policy", {}).get("failed_cases_replaced") is False,
        "failed-case replacement policy changed",
    )


def _validate_provider_case(case_root: Path) -> dict[str, Any]:
    prediction_root = case_root / "depth_completion_prediction"
    archive = prediction_root / PROVIDER_PREDICTION_FILENAME
    report_path = prediction_root / PROVIDER_REPORT_FILENAME
    seal_path = prediction_root / PROVIDER_SEAL_FILENAME
    result_path = case_root / PROVIDER_RESULT_FILENAME
    for path in (archive, report_path, seal_path, result_path):
        _require(path.is_file(), f"provider artifact is missing: {path}")
    seal = _load_json(seal_path)
    _require(
        seal.get("result_sha256") == canonical_sha256(seal),
        "provider prediction seal hash changed",
    )
    _require(
        seal.get("prediction_archive_sha256") == file_sha256(archive)
        and seal.get("prediction_report_sha256") == file_sha256(report_path),
        "provider prediction files changed after sealing",
    )
    report = _load_json(report_path)
    result = _load_json(result_path)
    _require(
        report.get("result_sha256") == canonical_sha256(report),
        "provider report hash changed",
    )
    _require(
        result.get("result_sha256") == canonical_sha256(result),
        "provider transfer result hash changed",
    )
    return {
        "archive": archive,
        "report": report_path,
        "seal": seal_path,
        "result": result_path,
        "provider_gate_passed": bool(result["provider_gate_passed"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--transfer-summary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def stage_assimilation_panel(
    protocol_path: str | Path,
    transfer_summary_path: str | Path,
    data_root: str | Path,
    physical_root: str | Path,
    provider_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create disjoint prediction and future-outcome artifacts for eight cases."""

    protocol_file = Path(protocol_path).resolve()
    summary_file = Path(transfer_summary_path).resolve()
    data = Path(data_root).resolve()
    physical = Path(physical_root).resolve()
    provider = Path(provider_root).resolve()
    output = Path(output_root).resolve()
    _require(not output.exists(), "assimilation staging output already exists")
    protocol = _load_json(protocol_file)
    _validate_protocol(protocol)
    transfer_summary = _load_json(summary_file)
    expected_transfer = protocol["provider_transfer"]
    _require(
        file_sha256(summary_file) == expected_transfer["summary_file_sha256"],
        "provider transfer summary file changed",
    )
    _require(
        transfer_summary.get("result_sha256")
        == expected_transfer["summary_result_sha256"],
        "provider transfer result identity changed",
    )
    _require(
        transfer_summary.get("transfer_gate_passed") is True,
        "provider transfer gate did not pass",
    )
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for case_name in protocol["fixed_source_cases"]:
        case_output = output / "cases" / case_name
        prediction_root = case_output / "prediction_input"
        withheld_root = case_output / "withheld_outcome"
        prediction_root.mkdir(parents=True)
        withheld_root.mkdir(parents=True)
        case_data = data / case_name
        final_data_path = case_data / "final_data.pkl"
        tracks_path = case_data / "gt_track_3d.pkl"
        split_path = case_data / "split.json"
        physical_case = physical / case_name
        optimal_path = physical_case / "optimal_params.pkl"
        physical_path = physical_case / "inference.pkl"
        for path in (
            final_data_path,
            tracks_path,
            split_path,
            optimal_path,
            physical_path,
        ):
            _require(path.is_file(), f"required source artifact is missing: {path}")
        provider_case = _validate_provider_case(provider / "cases" / case_name)
        split = _load_json(split_path)
        train_end = int(split["train"][1])
        future_end = int(split["test"][1])
        final_data = _load_pickle(final_data_path)
        manual_tracks = np.asarray(_load_pickle(tracks_path), dtype=np.float64)
        baseline = np.asarray(_load_pickle(physical_path), dtype=np.float64)
        optimal = _load_pickle(optimal_path)
        object_points = np.asarray(final_data["object_points"], dtype=np.float64)
        object_visible = np.asarray(
            final_data["object_visibilities"],
            dtype=bool,
        )
        motion_valid = np.asarray(
            final_data["object_motions_valid"],
            dtype=bool,
        )
        surface_points = np.asarray(final_data["surface_points"], dtype=np.float64)
        interior_points = np.asarray(final_data["interior_points"], dtype=np.float64)
        _require(
            min(len(object_points), len(manual_tracks), len(baseline)) >= future_end,
            f"{case_name} source arrays do not cover the future split",
        )
        structure_points = np.concatenate(
            (object_points[0], surface_points, interior_points),
            axis=0,
        )
        _require(
            baseline.shape[1:] == structure_points.shape,
            f"{case_name} physical state shape changed",
        )
        with np.load(provider_case["archive"], allow_pickle=False) as stored:
            provider_points = np.asarray(
                stored["completed_points_world_m"],
                dtype=np.float64,
            )
            provider_support = np.asarray(stored["completed_support"], dtype=bool)
            provider_reliability = np.asarray(
                stored["completed_prior_reliability"],
                dtype=np.float64,
            )
            provider_covariance = np.asarray(
                stored["completed_covariance_m2"],
                dtype=np.float64,
            )
            identity_ids = np.asarray(stored["identity_ids"], dtype=np.int64)
        provider_report = _load_json(provider_case["report"])
        source_start = int(provider_report["method_config"]["source_frame_start"])
        source_end = source_start + len(provider_points)
        _require(
            0 <= source_start < source_end <= train_end,
            f"{case_name} provider interval leaves the allowed prefix",
        )
        _require(
            np.all((identity_ids >= 0) & (identity_ids < manual_tracks.shape[1])),
            f"{case_name} provider identities exceed the manual identity table",
        )

        prediction_input = prediction_root / PREDICTION_INPUT_FILENAME
        np.savez_compressed(
            prediction_input,
            prefix_object_points_m=object_points[:train_end].astype(np.float32),
            prefix_object_visibilities=object_visible[:train_end],
            prefix_motion_valid=motion_valid[: max(train_end - 1, 0)],
            structure_points_m=structure_points.astype(np.float32),
            original_point_count=np.asarray(len(object_points[0]), dtype=np.int64),
            surface_point_count=np.asarray(len(surface_points), dtype=np.int64),
            train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
            future_end_frame_exclusive=np.asarray(future_end, dtype=np.int64),
            object_radius=np.asarray(float(optimal["object_radius"])),
            object_max_neighbours=np.asarray(
                int(optimal["object_max_neighbours"]),
                dtype=np.int64,
            ),
            controller_radius=np.asarray(float(optimal["controller_radius"])),
            controller_max_neighbours=np.asarray(
                int(optimal["controller_max_neighbours"]),
                dtype=np.int64,
            ),
            provider_points_world_m=provider_points.astype(np.float32),
            provider_support=provider_support,
            provider_prior_reliability=provider_reliability.astype(np.float32),
            provider_covariance_m2=provider_covariance.astype(np.float32),
            provider_identity_ids=identity_ids,
            provider_source_frame_start=np.asarray(source_start, dtype=np.int64),
            provider_source_frame_end_exclusive=np.asarray(
                source_end,
                dtype=np.int64,
            ),
            provider_gate_passed=np.asarray(
                provider_case["provider_gate_passed"],
                dtype=bool,
            ),
        )
        withheld = withheld_root / WITHHELD_OUTCOME_FILENAME
        np.savez_compressed(
            withheld,
            future_object_points_m=object_points[train_end:future_end].astype(
                np.float32
            ),
            future_object_visibilities=object_visible[train_end:future_end],
            manual_track_frame_zero_m=manual_tracks[0].astype(np.float32),
            future_manual_tracks_m=manual_tracks[train_end:future_end].astype(
                np.float32
            ),
            provider_identity_ids=identity_ids,
            train_end_frame_exclusive=np.asarray(train_end, dtype=np.int64),
            future_end_frame_exclusive=np.asarray(future_end, dtype=np.int64),
        )
        records.append(
            {
                "case": case_name,
                "prediction_input": {
                    "path": str(prediction_input),
                    "sha256": file_sha256(prediction_input),
                },
                "withheld_outcome": {
                    "path": str(withheld),
                    "sha256": file_sha256(withheld),
                },
                "physical_trajectory": {
                    "path": str(physical_path),
                    "sha256": file_sha256(physical_path),
                },
                "provider": {
                    "prediction_archive_sha256": file_sha256(
                        provider_case["archive"]
                    ),
                    "prediction_report_sha256": file_sha256(
                        provider_case["report"]
                    ),
                    "prediction_seal_sha256": file_sha256(provider_case["seal"]),
                    "transfer_result_sha256": file_sha256(
                        provider_case["result"]
                    ),
                    "provider_gate_passed": provider_case[
                        "provider_gate_passed"
                    ],
                    "source_frame_start": source_start,
                    "source_frame_end_exclusive": source_end,
                    "identity_ids": identity_ids.tolist(),
                },
                "source_inputs": {
                    "final_data_sha256": file_sha256(final_data_path),
                    "manual_tracks_sha256": file_sha256(tracks_path),
                    "split_sha256": file_sha256(split_path),
                    "optimal_params_sha256": file_sha256(optimal_path),
                },
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PhysTwinTAPNextPPSparseAssimilationSourceManifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_file),
        "provider_transfer_summary_sha256": file_sha256(summary_file),
        "fixed_case_count": len(records),
        "case_records": records,
        "information_boundary": {
            "prediction_input_contains_future_real_outcome": False,
            "withheld_future_staged_separately": True,
            "future_score_opened": False,
            "held_v8_accessed": False,
            "failed_cases_replaced": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    manifest["result_sha256"] = canonical_sha256(manifest)
    _write_json(output / SOURCE_MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    args = _parse_args()
    result = stage_assimilation_panel(
        args.protocol,
        args.transfer_summary,
        args.data_root,
        args.physical_root,
        args.provider_root,
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
