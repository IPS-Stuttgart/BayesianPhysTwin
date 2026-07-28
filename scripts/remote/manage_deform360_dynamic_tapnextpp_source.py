#!/usr/bin/env python3
"""Build the source barrier and run the two frozen source gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_artifacts import (
    PREDICTION_ARCHIVE_FILENAME,
    PREDICTION_SEAL_FILENAME,
    TECHNICAL_FAILURE_FILENAME,
    authorize_source_scoring,
    build_source_barrier,
    validate_prediction_seal,
    validate_source_admission,
    validate_technical_failure,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    load_dynamic_provider_cohort_lock,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    aggregate_provider_source_gate,
    evaluate_guarded_assimilation_gate,
    load_source_evaluation_protocol,
    score_provider_case_arrays,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_provider import (
    ASSIMILATION_ARCHIVE_FILENAME,
    ASSIMILATION_REPORT_FILENAME,
    PROVIDER_ARCHIVE_FILENAME,
    PROVIDER_REPORT_FILENAME,
    QUERY_SCHEDULE_FILENAME,
    RUNTIME_REPORT_FILENAME,
    validate_query_schedule_artifact,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256
from bayesian_phystwin.observation_belief import (
    array_sha256,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import PROTOCOL_ID

OBSERVATION_BELIEF_FILENAME = "dynamic_tapnextpp_observation_belief.npz"
PROVIDER_RESULT_KIND = "Deform360DynamicTAPNextPPSourceProviderResult"
ASSIMILATION_RESULT_KIND = (
    "Deform360DynamicTAPNextPPSourceAssimilationResult"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "JSON artifact must contain an object")
    return payload


def _canonical_sha256(
    payload: dict[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _require(not path.exists(), "source result already exists")
    payload["result_sha256"] = _canonical_sha256(
        payload,
        digest_key="result_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def _git_revision(repo: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not dirty, "source evaluator repository is dirty")
    return revision


def _admission_path(root: Path, case: str) -> Path:
    exact = root / f"{case}.admission.json"
    _require(exact.is_file(), f"source admission is missing: {case}")
    return exact


def _processed_target_path(root: Path, record: dict[str, Any]) -> Path:
    return (
        root
        / str(record["object_id"])
        / f"episode_{int(record['episode_id']):04d}"
        / "final_data.pkl"
    )


def _validate_protocols(
    *,
    protocol_path: Path,
    evaluation_path: Path,
    cohort: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    evaluation = load_source_evaluation_protocol(evaluation_path)
    _require(
        protocol.get("protocol_id") == PROTOCOL_ID
        and file_sha256(protocol_path)
        == cohort["bindings"]["provider_protocol_file_sha256"],
        "provider protocol differs from the cohort",
    )
    _require(
        file_sha256(evaluation_path)
        == cohort["bindings"][
            "source_evaluation_protocol_file_sha256"
        ],
        "source evaluation protocol differs from the cohort",
    )
    return protocol, evaluation


def _case_auxiliary_paths(
    prediction_dir: Path,
    physical_dir: Path,
) -> dict[str, Path]:
    return {
        "physical_manifest": physical_dir / PHYSICAL_MANIFEST_FILENAME,
        "provider_archive": prediction_dir / PROVIDER_ARCHIVE_FILENAME,
        "provider_report": prediction_dir / PROVIDER_REPORT_FILENAME,
        "assimilation_archive": (
            prediction_dir / ASSIMILATION_ARCHIVE_FILENAME
        ),
        "assimilation_report": (
            prediction_dir / ASSIMILATION_REPORT_FILENAME
        ),
        "runtime_report": prediction_dir / RUNTIME_REPORT_FILENAME,
    }


def _validate_ordinary_case(
    *,
    protocol_path: Path,
    admission_path: Path,
    prediction_dir: Path,
    physical_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    physical_manifest, physical_arrays = (
        validate_dynamic_physical_artifacts(physical_dir)
    )
    schedule_path = prediction_dir / QUERY_SCHEDULE_FILENAME
    schedule = validate_query_schedule_artifact(schedule_path)
    auxiliary = _case_auxiliary_paths(prediction_dir, physical_dir)
    validate_prediction_seal(
        prediction_dir / PREDICTION_SEAL_FILENAME,
        protocol_path=protocol_path,
        admission_path=admission_path,
        query_schedule_path=schedule_path,
        observation_belief_path=(
            prediction_dir / OBSERVATION_BELIEF_FILENAME
        ),
        prediction_dir=prediction_dir,
        additional_input_paths=auxiliary,
    )
    provider_report = _load_json(prediction_dir / PROVIDER_REPORT_FILENAME)
    _require(
        provider_report.get("artifact_kind")
        == "Deform360DynamicTAPNextPPProvider"
        and provider_report.get("case_hash") == schedule["case_hash"]
        and provider_report.get("result_sha256")
        == _canonical_sha256(
            provider_report,
            digest_key="result_sha256",
        ),
        "provider report is invalid",
    )
    with np.load(
        prediction_dir / PROVIDER_ARCHIVE_FILENAME,
        allow_pickle=False,
    ) as stored:
        provider_arrays = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
        }
    _require(
        provider_report["provider_archive"]["array_sha256"]
        == {
            name: array_sha256(value)
            for name, value in sorted(provider_arrays.items())
        },
        "provider archive arrays changed",
    )
    assimilation_report = _load_json(
        prediction_dir / ASSIMILATION_REPORT_FILENAME
    )
    _require(
        assimilation_report.get("artifact_kind")
        == "Deform360DynamicTAPNextPPAssimilation"
        and assimilation_report.get("case_hash") == schedule["case_hash"]
        and assimilation_report.get("result_sha256")
        == _canonical_sha256(
            assimilation_report,
            digest_key="result_sha256",
        ),
        "assimilation report is invalid",
    )
    with np.load(
        prediction_dir / ASSIMILATION_ARCHIVE_FILENAME,
        allow_pickle=False,
    ) as stored:
        assimilation_arrays = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
        }
    _require(
        assimilation_report["assimilation_archive"]["array_sha256"]
        == {
            name: array_sha256(value)
            for name, value in sorted(assimilation_arrays.items())
        },
        "assimilation archive arrays changed",
    )
    _require(
        np.array_equal(
            assimilation_arrays[PHYSICAL_ARM],
            physical_arrays["physical_prediction_m"],
        )
        and np.array_equal(
            assimilation_arrays[PERSISTENCE_ARM],
            physical_arrays["persistence_prediction_m"],
        )
        and physical_manifest["case_hash"] == schedule["case_hash"],
        "assimilation differs from the sealed physical backbone",
    )
    return (
        schedule,
        provider_arrays,
        assimilation_report,
        assimilation_arrays,
    )


def _bind_source_cases(
    *,
    cohort: dict[str, Any],
    barrier: dict[str, Any],
    protocol_path: Path,
    admission_root: Path,
    prediction_root: Path,
    physical_root: Path,
    processed_root: Path,
) -> list[dict[str, Any]]:
    ordinary = set(barrier["ordinary_prediction_case_hashes"])
    failures = set(barrier["technical_failure_case_hashes"])
    _require(
        ordinary | failures
        == {row["case_hash"] for row in cohort["source_cases"]},
        "source barrier differs from the locked source cohort",
    )
    bound: list[dict[str, Any]] = []
    for record_source in cohort["source_cases"]:
        record = dict(record_source)
        case = str(record["case"])
        admission_path = _admission_path(admission_root, case)
        raw_admission = _load_json(admission_path)
        admission = validate_source_admission(raw_admission)
        _require(
            admission["admitted"] is True
            and admission["source_admission_sha256"]
            == record["admission_sha256"]
            and admission["case_hash"] == record["case_hash"],
            f"source admission differs from the cohort: {case}",
        )
        target_path = _processed_target_path(processed_root, record)
        _require(target_path.is_file(), f"source future is missing: {case}")
        _require(
            file_sha256(target_path)
            == raw_admission["source_files"]["future_payload"]["sha256"],
            f"source future checksum changed: {case}",
        )
        prediction_dir = prediction_root / case
        physical_dir = physical_root / case / "sealed_physical"
        if record["case_hash"] in failures:
            validate_technical_failure(
                prediction_dir / TECHNICAL_FAILURE_FILENAME,
                protocol_path=protocol_path,
                admission_path=admission_path,
            )
            prediction = None
        else:
            prediction = _validate_ordinary_case(
                protocol_path=protocol_path,
                admission_path=admission_path,
                prediction_dir=prediction_dir,
                physical_dir=physical_dir,
            )
        bound.append(
            {
                "record": record,
                "admission_path": admission_path,
                "target_path": target_path,
                "prediction_dir": prediction_dir,
                "technical_failure": record["case_hash"] in failures,
                "prediction": prediction,
            }
        )
    return bound


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    _require(isinstance(payload, dict), "source future payload is invalid")
    target = np.asarray(payload["object_points"])
    visibility = np.asarray(payload["object_visibilities"], dtype=bool)
    validity = np.asarray(payload["object_motions_valid"], dtype=bool)
    _require(
        target.ndim == 3
        and target.shape[0] == 76
        and target.shape[2] == 3
        and visibility.shape == validity.shape == target.shape[:2],
        "source future arrays changed shape",
    )
    return target, visibility, validity


def _barrier(args: argparse.Namespace) -> dict[str, Any]:
    cohort = load_dynamic_provider_cohort_lock(args.cohort_lock)
    expected = [row["case_hash"] for row in cohort["source_cases"]]
    predictions: list[Path] = []
    failures: list[Path] = []
    for record in cohort["source_cases"]:
        root = args.prediction_root / str(record["case"])
        prediction = root / PREDICTION_SEAL_FILENAME
        failure = root / TECHNICAL_FAILURE_FILENAME
        _require(
            prediction.is_file() != failure.is_file(),
            f"source case lacks one exclusive disposition: {record['case']}",
        )
        (predictions if prediction.is_file() else failures).append(
            prediction if prediction.is_file() else failure
        )
    return build_source_barrier(
        args.output,
        expected_case_hashes=expected,
        prediction_seals=predictions,
        technical_failures=failures,
    )


def _common_bound(args: argparse.Namespace) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    revision = _git_revision(args.repo.resolve())
    cohort = load_dynamic_provider_cohort_lock(args.cohort_lock)
    _require(
        revision == cohort["bindings"]["provider_commit"],
        "source evaluator revision differs from the cohort",
    )
    _validate_protocols(
        protocol_path=args.protocol.resolve(),
        evaluation_path=args.evaluation_protocol.resolve(),
        cohort=cohort,
    )
    barrier = authorize_source_scoring(args.barrier)
    bound = _bind_source_cases(
        cohort=cohort,
        barrier=barrier,
        protocol_path=args.protocol.resolve(),
        admission_root=args.admission_root.resolve(),
        prediction_root=args.prediction_root.resolve(),
        physical_root=args.physical_root.resolve(),
        processed_root=args.processed_root.resolve(),
    )
    return revision, cohort, barrier, bound


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    revision, cohort, barrier, bound = _common_bound(args)
    case_reports: list[dict[str, Any]] = []
    for item in bound:
        record = item["record"]
        if item["technical_failure"]:
            case_reports.append(
                {
                    "object_hash": record["object_hash"],
                    "case_hash": record["case_hash"],
                    "technical_failure": True,
                    "supported_fraction": 0.0,
                    "provider_rmse_m": None,
                }
            )
            continue
        target, visibility, validity = _load_target(item["target_path"])
        schedule, arrays, _, _ = item["prediction"]
        score = score_provider_case_arrays(
            trajectory_world_m=arrays["trajectory_world_m"],
            accepted_support=arrays["accepted_support"],
            local_covariance_m2=arrays["local_covariance_m2"],
            shared_bias_standard_deviation_m=float(
                _load_json(
                    item["prediction_dir"] / PROVIDER_REPORT_FILENAME
                )["configuration"]["multiview"][
                    "shared_bias_standard_deviation_m"
                ]
            ),
            target_m=target,
            target_visibility=visibility,
            target_validity=validity,
            entity_ids=np.asarray(schedule["entity_ids"]),
            birth_frames=np.asarray(schedule["birth_frames"]),
            update_frames=np.asarray(schedule["update_frames"]),
        )
        case_reports.append(
            {
                "object_hash": record["object_hash"],
                "case_hash": record["case_hash"],
                "technical_failure": False,
                **score,
            }
        )
    gate = aggregate_provider_source_gate(case_reports)
    return _write_result(
        args.output,
        {
            "schema_version": 1,
            "artifact_kind": PROVIDER_RESULT_KIND,
            "protocol_id": PROTOCOL_ID,
            "repository_revision": revision,
            "cohort_lock_sha256": cohort["cohort_lock_sha256"],
            "source_barrier_sha256": barrier["result_sha256"],
            "source_evaluation_protocol_file_sha256": file_sha256(
                args.evaluation_protocol
            ),
            "case_reports": case_reports,
            "gate": gate,
            "information_boundary": {
                "all_source_predictions_validated_before_first_future": True,
                "source_futures_opened_for_provider_scoring": True,
                "assimilation_outcome_opened": False,
                "target_artifact_opened": False,
                "held_v8_access": False,
            },
        },
    )


def _assimilation(args: argparse.Namespace) -> dict[str, Any]:
    revision, cohort, barrier, bound = _common_bound(args)
    provider_result = _load_json(args.provider_result)
    _require(
        provider_result.get("artifact_kind") == PROVIDER_RESULT_KIND
        and provider_result.get("protocol_id") == PROTOCOL_ID
        and provider_result.get("cohort_lock_sha256")
        == cohort["cohort_lock_sha256"]
        and provider_result.get("source_barrier_sha256")
        == barrier["result_sha256"]
        and provider_result.get("result_sha256")
        == _canonical_sha256(
            provider_result,
            digest_key="result_sha256",
        )
        and provider_result.get("gate", {}).get("passed") is True,
        "provider source gate did not authorize assimilation scoring",
    )
    cases: list[dict[str, Any]] = []
    for item in bound:
        record = item["record"]
        if item["technical_failure"]:
            cases.append(
                {
                    "object_hash": record["object_hash"],
                    "technical_failure": True,
                }
            )
            continue
        target, visibility, validity = _load_target(item["target_path"])
        _, _, assimilation_report, arrays = item["prediction"]
        with np.load(
            item["prediction_dir"] / PREDICTION_ARCHIVE_FILENAME,
            allow_pickle=False,
        ) as stored:
            hidden = np.asarray(stored["hidden_entity_ids"]).copy()
        _require(
            set(
                (
                    PHYSICAL_ARM,
                    PERSISTENCE_ARM,
                    SELECTED_BACKBONE_ARM,
                    CANDIDATE_ARM,
                )
            )
            <= set(arrays),
            "assimilation arms are incomplete",
        )
        cases.append(
            {
                "object_hash": record["object_hash"],
                "technical_failure": False,
                "arrays": arrays,
                "assimilation_report": assimilation_report,
                "target_m": target,
                "visibility": visibility,
                "validity": validity,
                "hidden_entity_ids": hidden,
            }
        )
    gate = evaluate_guarded_assimilation_gate(cases)
    return _write_result(
        args.output,
        {
            "schema_version": 1,
            "artifact_kind": ASSIMILATION_RESULT_KIND,
            "protocol_id": PROTOCOL_ID,
            "repository_revision": revision,
            "cohort_lock_sha256": cohort["cohort_lock_sha256"],
            "source_barrier_sha256": barrier["result_sha256"],
            "provider_result_sha256": provider_result["result_sha256"],
            "source_evaluation_protocol_file_sha256": file_sha256(
                args.evaluation_protocol
            ),
            "gate": gate,
            "information_boundary": {
                "provider_gate_passed_before_assimilation_outcome": True,
                "source_futures_opened_for_hidden_identity_scoring": True,
                "target_artifact_opened": False,
                "held_v8_access": False,
            },
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    barrier = commands.add_parser("barrier")
    barrier.add_argument("--cohort-lock", type=Path, required=True)
    barrier.add_argument("--prediction-root", type=Path, required=True)
    barrier.add_argument("--output", type=Path, required=True)
    for name in ("provider", "assimilation"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument(
            "--evaluation-protocol",
            type=Path,
            required=True,
        )
        command.add_argument("--cohort-lock", type=Path, required=True)
        command.add_argument("--barrier", type=Path, required=True)
        command.add_argument("--admission-root", type=Path, required=True)
        command.add_argument("--prediction-root", type=Path, required=True)
        command.add_argument("--physical-root", type=Path, required=True)
        command.add_argument("--processed-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    assimilation = commands.choices["assimilation"]
    assimilation.add_argument("--provider-result", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "barrier":
        result = _barrier(args)
    elif args.command == "provider":
        result = _provider(args)
    else:
        result = _assimilation(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
