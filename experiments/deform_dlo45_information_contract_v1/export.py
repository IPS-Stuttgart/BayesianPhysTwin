"""Replay the frozen DEFORM DLO4/DLO5 decision study as benchmark suites.

The adapter reconstructs per-decision evidence from the exact source model and
pinned public evaluation trajectories. It creates one certificate suite and
forecast-only comparator suites, evaluates no new policy, and never refits or
retunes the completed experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)
from experiments.deform_dlo45_decision_identifiability_v1._common import (
    ATOL,
    DLOS,
    INTERNAL,
    FloatArray,
    Model,
    Protocol,
    extract_observation,
    file_manifest,
    load_protocol,
    load_trajectory,
    read_json,
    sha256_file,
    trajectory_paths,
    window_starts,
    write_json,
)
from experiments.deform_dlo45_decision_identifiability_v1._evaluation import (
    load_models,
)
from experiments.deform_dlo45_decision_identifiability_v1._model import decide

CONTRACT: Final = "deform-dlo45-information-contract-adapter-v1"
REQUEST_CONTRACT: Final = "deform-dlo45-information-contract-request-v1"
SUITE_SCHEMA: Final = "prob4d.information-contract-suite"
SUITE_VERSION: Final = 1
SOURCE_RUN_ID: Final = 33473378340
SOURCE_ARTIFACT_ID: Final = 9787311310
SOURCE_HEAD_SHA: Final = "38b2ea56471923e63b64cfe24bf3f691aad5d5e0"
SOURCE_ARTIFACT_DIGEST: Final = (
    "sha256:fdb35f680e4cf685303f841b8974c16eb9301ad52cbdc8f5af0d87a9dfc358ee"
)
SOURCE_MODEL_SHA256: Final = (
    "a43aed43cd563ee47358e48cab84829dc7eebc77d97725721a11b228f3b6b7f0"
)
SOURCE_RESULT_SHA256: Final = (
    "77332c323ddd09d945f65f57e3b83a12deedd9ea94509e43c13c9ca87f3cc353"
)
SOURCE_SEAL_SHA256: Final = (
    "bcf55e066fdf3b44c29329a8fdffc4f0ccc7e6166ae19a5593ab02f28c9c5e80"
)
REFERENCE_RESULT_SHA256: Final = (
    "d2343114e4c07af4f7126b6779bb65950ebd4cbf89b934b7a2689a7e7106248b"
)
REFERENCE_RESULT_GIT_BLOB: Final = "71f6a2d627c25ba8a3cbfc2ba6cf980842f8bf7b"
DEFORM_REPOSITORY: Final = "roahmlab/DEFORM"
DEFORM_COMMIT: Final = "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
PROB4D_REPOSITORY: Final = "IPS-Stuttgart/Prob4D"
PROB4D_COMMIT: Final = "f25b0cdb0a258e1b2ef276d25a723c2cf3a9fb4f"
METHODS: Final = (
    "fallback",
    "certificate",
    "jeffrey_point",
    "kernel_point",
    "map_point",
    "oracle",
)
CLAIM_BOUNDARY: Final = (
    "This adapter is a retrospective, exact re-expression of the previously "
    "completed source-frozen DLO4/DLO5 result. It opens no new target cohort, "
    "selects no replacement policy, and does not establish unseen-object "
    "generalization, target-domain regret coverage, calibration, robot safety, "
    "or state of the art."
)


class DecisionEvidence(NamedTuple):
    correction: FloatArray
    loss_by_hypothesis: FloatArray
    prior: FloatArray
    classes: NDArray[np.int64]
    quotient_mass: FloatArray
    worst_case_regret: FloatArray
    selected_actions: dict[str, int]
    certificate_admitted: bool


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            value = np.asarray(arrays[name])
            if value.dtype.kind == "O":
                raise ValueError("benchmark payloads must not contain object arrays")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return stream.getvalue()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_request(path: Path) -> dict[str, Any]:
    request = read_json(path)
    expected: dict[str, object] = {
        "contract": REQUEST_CONTRACT,
        "schema_version": 1,
        "status": "authorized-retrospective-replay",
        "source_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_head_sha": SOURCE_HEAD_SHA,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_seal_sha256": SOURCE_SEAL_SHA256,
        "reference_result_sha256": REFERENCE_RESULT_SHA256,
        "reference_result_git_blob": REFERENCE_RESULT_GIT_BLOB,
        "deform_repository": DEFORM_REPOSITORY,
        "deform_commit": DEFORM_COMMIT,
        "prob4d_repository": PROB4D_REPOSITORY,
        "prob4d_commit": PROB4D_COMMIT,
        "target_tuning": False,
        "target_retries": False,
        "new_target_outcomes_opened": False,
        "payload_redistribution": False,
    }
    if set(request) != set(expected):
        raise ValueError("adapter request fields changed")
    for name, value in expected.items():
        if request.get(name) != value:
            raise ValueError(f"adapter request changed: {name}")
    return request


def _validate_sources(
    *,
    protocol_path: Path,
    model_path: Path,
    source_result_path: Path,
    source_seal_path: Path,
    reference_result_path: Path,
) -> tuple[Protocol, dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    seal = read_json(source_seal_path)
    if sha256_file(model_path) != SOURCE_MODEL_SHA256:
        raise ValueError("source model SHA-256 changed")
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source result SHA-256 changed")
    if sha256_file(source_seal_path) != SOURCE_SEAL_SHA256:
        raise ValueError("source seal SHA-256 changed")
    if sha256_file(reference_result_path) != REFERENCE_RESULT_SHA256:
        raise ValueError("reference target result SHA-256 changed")
    if (
        seal.get("contract") != "deform-dlo45-decision-identifiability-v1"
        or seal.get("stage") != "source-seal"
        or seal.get("source_model_sha256") != SOURCE_MODEL_SHA256
        or seal.get("source_result_sha256") != SOURCE_RESULT_SHA256
        or seal.get("protocol_sha256") != sha256_file(protocol_path)
    ):
        raise ValueError("source seal does not bind the supplied inputs")
    source_result = read_json(source_result_path)
    if (
        source_result.get("stage") != "source"
        or source_result.get("target_data_read") is not False
        or source_result.get("all_source_gates_passed") is not True
    ):
        raise ValueError("source result is not the qualified source-only result")
    reference = read_json(reference_result_path)
    if (
        reference.get("contract") != "deform-dlo45-decision-identifiability-v1"
        or reference.get("stage") != "target-result"
        or reference.get("target_tuning") is not False
        or reference.get("target_retries") is not False
        or reference.get("source_model_sha256") != SOURCE_MODEL_SHA256
    ):
        raise ValueError("reference result is not the frozen target result")
    return protocol, reference


def reconstruct_decision_evidence(
    feature: FloatArray,
    model: Model,
    protocol: Protocol,
) -> DecisionEvidence:
    """Reconstruct the exact finite-support record used by ``decide``."""

    query = (feature - model.feature_mean) / model.feature_scale
    pool = (model.features - model.feature_mean) / model.feature_scale
    distance = np.mean(np.square(pool - query[None, :]), axis=1)
    neighbor_count = min(model.neighbors, len(distance))
    selected = np.argpartition(distance, neighbor_count - 1)[:neighbor_count]
    selected = selected[np.lexsort((selected, distance[selected]))]
    selected_distance = distance[selected]
    positive = selected_distance[selected_distance > 0.0]
    base_bandwidth = (
        float(np.median(positive))
        if len(positive)
        else max(float(np.mean(selected_distance)), 1e-12)
    )
    bandwidth = max(base_bandwidth * model.temperature_scale, 1e-12)
    logits = -(selected_distance - float(np.min(selected_distance))) / bandwidth
    kernel_weights = np.exp(logits)
    kernel_weights /= np.sum(kernel_weights)
    global_classes = model.class_labels[selected]
    unique_classes = np.unique(global_classes)
    remap = {int(value): index for index, value in enumerate(unique_classes)}
    classes = np.asarray(
        [remap[int(value)] for value in global_classes], dtype=np.int64
    )
    class_count = len(unique_classes)
    quotient_mass = np.bincount(
        classes,
        weights=kernel_weights,
        minlength=class_count,
    ).astype(np.float64)
    class_sizes = np.bincount(classes, minlength=class_count).astype(np.float64)
    jeffrey_weights = quotient_mass[classes] / class_sizes[classes]
    selected_residuals = model.residuals[selected]
    correction = np.einsum("i,id->d", jeffrey_weights, selected_residuals)
    actions = model.action_scales[:, None] * correction[None, :]
    raw_losses = np.mean(
        np.square(selected_residuals[:, None, :] - actions[None, :, :]),
        axis=2,
    )
    relative_losses = raw_losses / (raw_losses[:, :1] + model.loss_floor)
    prior = np.full(neighbor_count, 1.0 / neighbor_count)
    certificate = query_decision_certificate(
        prior,
        quotient_mass,
        classes,
        relative_losses,
        regret_tolerance=model.regret_tolerance,
    )
    admitted = bool(
        certificate.minimax_worst_case_regret <= model.regret_tolerance + ATOL
    )
    certificate_action = certificate.minimax_action_index if admitted else 0
    jeffrey_action = int(
        np.argmin(np.einsum("i,ia->a", jeffrey_weights, relative_losses))
    )
    kernel_action = int(
        np.argmin(np.einsum("i,ia->a", kernel_weights, relative_losses))
    )
    map_action = int(np.argmin(relative_losses[0]))

    reference = decide(feature, model, protocol)
    if (
        certificate_action != reference.certificate_action
        or jeffrey_action != reference.jeffrey_action
        or kernel_action != reference.kernel_action
        or map_action != reference.map_action
        or not np.allclose(correction, reference.correction, rtol=0.0, atol=1e-15)
        or not np.allclose(
            certificate.worst_case_regret,
            reference.worst_case_regret,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError("adapter reconstruction does not match the frozen policy")
    return DecisionEvidence(
        correction=correction,
        loss_by_hypothesis=relative_losses,
        prior=prior,
        classes=classes,
        quotient_mass=quotient_mass,
        worst_case_regret=certificate.worst_case_regret,
        selected_actions={
            "fallback": 0,
            "certificate": certificate_action,
            "jeffrey_point": jeffrey_action,
            "kernel_point": kernel_action,
            "map_point": map_action,
        },
        certificate_admitted=admitted,
    )


def _case_payload(
    *,
    truth: FloatArray,
    baseline: FloatArray,
    normalized_actions: FloatArray,
    selected_action: int,
    length_scale: float,
    evidence: DecisionEvidence,
    realized_action_loss: FloatArray,
    certificate: bool,
    regret_tolerance: float,
) -> dict[str, NDArray[Any]]:
    prediction = baseline.reshape(-1, 3) + (
        normalized_actions[selected_action].reshape(-1, 3) * length_scale
    )
    arrays: dict[str, NDArray[Any]] = {
        "truth_xyz_m": truth.reshape(-1, 3),
        "prediction_mean_xyz_m": prediction,
    }
    if certificate:
        arrays.update(
            {
                "decision_loss_by_hypothesis": evidence.loss_by_hypothesis,
                "hypothesis_prior": evidence.prior,
                "quotient_class": evidence.classes,
                "quotient_mass": evidence.quotient_mass,
                "reported_worst_case_regret": evidence.worst_case_regret,
                "selected_action": np.asarray(selected_action, dtype=np.int64),
                "fallback_action": np.asarray(0, dtype=np.int64),
                "decision_admitted": np.asarray(evidence.certificate_admitted),
                # The original implementation uses ATOL as a frozen numerical
                # comparison margin. Encoding it in the operational threshold
                # makes the benchmark replay exactly the same decision rule.
                "regret_tolerance": np.asarray(regret_tolerance + ATOL),
                "realized_action_loss": realized_action_loss,
            }
        )
    return arrays


def _suite_claim(method: str) -> str:
    if method == "certificate":
        return (
            CLAIM_BOUNDARY
            + " The decision contract is replayed exactly over the registered "
            "finite support; realized held regret is reported separately."
        )
    if method == "oracle":
        return CLAIM_BOUNDARY + " This comparator uses held outcomes and is diagnostic only."
    return (
        CLAIM_BOUNDARY
        + " This is a forecast-only comparator without a certificate claim."
    )


def export_suites(
    *,
    request_path: Path,
    protocol_path: Path,
    dataset_root: Path,
    model_path: Path,
    source_result_path: Path,
    source_seal_path: Path,
    reference_result_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create deterministic benchmark suites from the exact prior experiment."""

    request = _validate_request(request_path)
    protocol, reference = _validate_sources(
        protocol_path=protocol_path,
        model_path=model_path,
        source_result_path=source_result_path,
        source_seal_path=source_seal_path,
        reference_result_path=reference_result_path,
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to replace output directory: {output_root}")
    output_root.mkdir(parents=True)
    models = load_models(model_path)
    cases_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    dataset_manifests: dict[str, object] = {}
    decisions = 0

    reference_manifests = reference.get("eval_manifest")
    if not isinstance(reference_manifests, dict):
        raise ValueError("reference target result omits the evaluation manifest")

    for dlo in DLOS:
        paths = trajectory_paths(dataset_root, dlo, "eval")
        manifest = file_manifest(paths)
        if manifest != reference_manifests.get(dlo):
            raise ValueError(f"{dlo} public evaluation bytes changed")
        dataset_manifests[dlo] = manifest
        model = models[dlo]
        for path in paths:
            trajectory = load_trajectory(path)
            for current in window_starts(protocol):
                observation = extract_observation(trajectory, current, protocol)
                evidence = reconstruct_decision_evidence(
                    observation.feature,
                    model,
                    protocol,
                )
                truth = trajectory[
                    current + 1 : current + 1 + protocol.horizon_frames,
                    INTERNAL,
                    :,
                ].copy()
                actual_residual = (
                    (truth - observation.baseline).reshape(-1)
                    / observation.length_scale
                )
                normalized_actions = (
                    model.action_scales[:, None] * evidence.correction[None, :]
                )
                normalized_mse = np.mean(
                    np.square(actual_residual[None, :] - normalized_actions),
                    axis=1,
                )
                physical_mse = normalized_mse * observation.length_scale**2
                realized_action_loss = physical_mse
                oracle_action = int(np.argmin(physical_mse))
                selected_actions = {
                    **evidence.selected_actions,
                    "oracle": oracle_action,
                }
                case_id = f"{dlo}/{path.name}/t{current:04d}"
                group_id = f"{dlo}/{path.name}"
                file_stem = f"{dlo.lower()}-{path.stem}-t{current:04d}"
                for method in METHODS:
                    selected_action = selected_actions[method]
                    arrays = _case_payload(
                        truth=truth,
                        baseline=observation.baseline,
                        normalized_actions=normalized_actions,
                        selected_action=selected_action,
                        length_scale=observation.length_scale,
                        evidence=evidence,
                        realized_action_loss=realized_action_loss,
                        certificate=method == "certificate",
                        regret_tolerance=model.regret_tolerance,
                    )
                    payload = _deterministic_npz_bytes(arrays)
                    relative_path = Path("cases") / f"{file_stem}.npz"
                    payload_path = output_root / method / relative_path
                    _write_new(payload_path, payload)
                    cases_by_method[method].append(
                        {
                            "case_id": case_id,
                            "group_id": group_id,
                            "payload": relative_path.as_posix(),
                            "payload_sha256": _sha256_bytes(payload),
                            "tasks": (
                                ["forecast", "decision"]
                                if method == "certificate"
                                else ["forecast"]
                            ),
                            "metadata": {
                                "dataset": "DEFORM",
                                "dataset_commit": DEFORM_COMMIT,
                                "dlo": dlo,
                                "trajectory": path.name,
                                "window_start": current,
                                "method": method,
                                "selected_action": selected_action,
                                "registered_regret_tolerance": model.regret_tolerance,
                                "numerical_margin": ATOL,
                                "uses_held_out_action_selection": method == "oracle",
                                "realized_action_loss_unit": "m2-coordinate-mse",
                                "source_run_id": SOURCE_RUN_ID,
                                "source_artifact_id": SOURCE_ARTIFACT_ID,
                                "target_tuning": False,
                                "target_retries": False,
                            },
                        }
                    )
                decisions += 1

    suite_paths: dict[str, str] = {}
    suite_hashes: dict[str, str] = {}
    for method in METHODS:
        suite = {
            "schema_name": SUITE_SCHEMA,
            "schema_version": SUITE_VERSION,
            "suite_id": f"deform-dlo45-{method}-information-contract-v1",
            "aggregation_unit": "group_id",
            "thresholds": {
                "coverage_probability": 0.9,
                "gauge_sensitivity_tolerance": 1e-12,
                "moment_atol": 1e-12,
                "relative_rank_tolerance": 1e-10,
            },
            "claim_boundary": _suite_claim(method),
            "cases": cases_by_method[method],
        }
        suite_path = output_root / method / "suite.json"
        write_json(suite_path, suite)
        suite_paths[method] = suite_path.relative_to(output_root).as_posix()
        suite_hashes[method] = sha256_file(suite_path)

    dataset_manifest_path = output_root / "dataset_manifest.json"
    write_json(dataset_manifest_path, dataset_manifests)

    adapter_manifest = {
        "contract": CONTRACT,
        "schema_version": 1,
        "classification": "retrospective exact benchmark adapter",
        "claim_boundary": CLAIM_BOUNDARY,
        "request_sha256": sha256_file(request_path),
        "source": {
            "run_id": SOURCE_RUN_ID,
            "artifact_id": SOURCE_ARTIFACT_ID,
            "artifact_digest": SOURCE_ARTIFACT_DIGEST,
            "head_sha": SOURCE_HEAD_SHA,
            "model_sha256": SOURCE_MODEL_SHA256,
            "result_sha256": SOURCE_RESULT_SHA256,
            "seal_sha256": SOURCE_SEAL_SHA256,
        },
        "reference_target": {
            "sha256": REFERENCE_RESULT_SHA256,
            "git_blob_sha1": REFERENCE_RESULT_GIT_BLOB,
        },
        "dataset": {
            "repository": DEFORM_REPOSITORY,
            "commit": DEFORM_COMMIT,
            "eval_manifest_path": (
                dataset_manifest_path.relative_to(output_root).as_posix()
            ),
            "eval_manifest_sha256": _canonical_sha256(dataset_manifests),
            "eval_manifest_file_sha256": sha256_file(dataset_manifest_path),
            "payloads_redistributed": False,
        },
        "benchmark": {
            "repository": PROB4D_REPOSITORY,
            "commit": PROB4D_COMMIT,
            "suite_paths": suite_paths,
            "suite_sha256": suite_hashes,
        },
        "case_count_per_method": decisions,
        "independent_trajectory_count": len(DLOS) * 14,
        "methods": list(METHODS),
        "target_tuning": False,
        "target_retries": False,
        "new_target_outcomes_opened": False,
        "request": request,
    }
    write_json(output_root / "adapter_manifest.json", adapter_manifest)
    return adapter_manifest


def _pooled_rmse_m(cases: Sequence[Mapping[str, Any]]) -> float:
    values = [float(case["metrics"]["forecast"]["rmse_m"]) for case in cases]
    return math.sqrt(float(np.mean(np.square(values))))


def validate_benchmark_results(
    *,
    output_root: Path,
    reference_result_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Verify benchmark outputs reproduce every original aggregate and trajectory."""

    reference = read_json(reference_result_path)
    manifest = read_json(output_root / "adapter_manifest.json")
    if manifest.get("case_count_per_method") != 532:
        raise ValueError("adapter did not produce 532 decisions per method")
    results: dict[str, dict[str, Any]] = {}
    rosters: dict[str, tuple[str, ...]] = {}
    validation_methods: dict[str, Any] = {}
    expected_field = {
        "fallback": "fallback_rmse_mm",
        "certificate": "certificate_rmse_mm",
        "jeffrey_point": "jeffrey_point_rmse_mm",
        "kernel_point": "kernel_point_rmse_mm",
        "map_point": "map_point_rmse_mm",
        "oracle": "oracle_rmse_mm",
    }

    for method in METHODS:
        result_path = output_root / method / "benchmark_result.json"
        result = read_json(result_path)
        results[method] = result
        cases = result.get("cases")
        if not isinstance(cases, list) or len(cases) != 532:
            raise ValueError(f"{method}: benchmark result has wrong case count")
        aggregate = result.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError(f"{method}: benchmark aggregate is missing")
        if aggregate.get("independent_group_count") != 28:
            raise ValueError(f"{method}: wrong trajectory-group count")
        rosters[method] = tuple(str(case["case_id"]) for case in cases)
        per_dlo: dict[str, Any] = {}
        for dlo in DLOS:
            dlo_cases = [case for case in cases if case["metadata"]["dlo"] == dlo]
            observed = 1000.0 * _pooled_rmse_m(dlo_cases)
            expected = float(reference["dlos"][dlo]["aggregate"][method]["rmse_mm"])
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{method}/{dlo}: RMSE {observed} does not reproduce {expected}"
                )
            counts = np.bincount(
                [int(case["metadata"]["selected_action"]) for case in dlo_cases],
                minlength=3,
            ).tolist()
            expected_counts = reference["dlos"][dlo]["aggregate"][method][
                "action_counts"
            ]
            if counts != expected_counts:
                raise ValueError(f"{method}/{dlo}: action counts changed")
            trajectory_rmse: dict[str, float] = {}
            trajectories = {case["metadata"]["trajectory"] for case in dlo_cases}
            for trajectory in sorted(trajectories):
                selected = [
                    case
                    for case in dlo_cases
                    if case["metadata"]["trajectory"] == trajectory
                ]
                trajectory_rmse[trajectory] = 1000.0 * _pooled_rmse_m(selected)
            reference_rows = {
                row["trajectory"]: row
                for row in reference["dlos"][dlo]["per_trajectory"]
            }
            for trajectory, observed_trajectory in trajectory_rmse.items():
                expected_trajectory = float(
                    reference_rows[trajectory][expected_field[method]]
                )
                if not math.isclose(
                    observed_trajectory,
                    expected_trajectory,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"{method}/{dlo}/{trajectory}: trajectory RMSE changed"
                    )
            per_dlo[dlo] = {
                "rmse_mm": observed,
                "action_counts": counts,
                "trajectory_count": len(trajectory_rmse),
            }
        validation_methods[method] = {
            "case_count": len(cases),
            "trajectory_count": aggregate["independent_group_count"],
            "per_dlo": per_dlo,
        }

    reference_roster = rosters["certificate"]
    if any(roster != reference_roster for roster in rosters.values()):
        raise ValueError("method suites do not share one exact case roster")
    certificate = results["certificate"]
    certificate_cases = certificate["cases"]
    if certificate["aggregate"]["contract"]["all_cases_pass"] is not True:
        raise ValueError("certificate suite fails its benchmark contract")
    nonfallback = sum(
        int(case["metrics"]["decision"]["selected_action"] != 0)
        for case in certificate_cases
    )
    harmful = sum(
        int(
            case["metrics"]["decision"]["realized_action_loss"][
                case["metrics"]["decision"]["selected_action"]
            ]
            > case["metrics"]["decision"]["realized_action_loss"][0] + ATOL
        )
        for case in certificate_cases
    )
    if nonfallback != int(reference["aggregate"]["certificate_nonfallback_count"]):
        raise ValueError("certificate nonfallback count changed")
    expected_harmful = sum(
        round(
            float(
                reference["dlos"][dlo]["aggregate"]["certificate"][
                    "harm_fraction_vs_fallback"
                ]
            )
            * int(reference["dlos"][dlo]["decision_count"])
        )
        for dlo in DLOS
    )
    if harmful != expected_harmful:
        raise ValueError("certificate harmful-departure count changed")

    validation = {
        "contract": CONTRACT,
        "schema_version": 1,
        "classification": "exact replay validation",
        "claim_boundary": CLAIM_BOUNDARY,
        "all_reference_metrics_reproduced": True,
        "case_rosters_identical": True,
        "certificate_contract_all_cases_pass": True,
        "certificate_nonfallback_count": nonfallback,
        "certificate_harmful_nonfallback_count": harmful,
        "methods": validation_methods,
        "payloads_retained_for_publication": False,
    }
    write_json(destination, validation)
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--request", type=Path, required=True)
    export.add_argument("--protocol", type=Path, required=True)
    export.add_argument("--dataset-root", type=Path, required=True)
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--source-result", type=Path, required=True)
    export.add_argument("--source-seal", type=Path, required=True)
    export.add_argument("--reference-result", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    validate.add_argument("--reference-result", type=Path, required=True)
    validate.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        export_suites(
            request_path=args.request,
            protocol_path=args.protocol,
            dataset_root=args.dataset_root,
            model_path=args.model,
            source_result_path=args.source_result,
            source_seal_path=args.source_seal,
            reference_result_path=args.reference_result,
            output_root=args.output_root,
        )
        return 0
    validate_benchmark_results(
        output_root=args.output_root,
        reference_result_path=args.reference_result,
        destination=args.destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
