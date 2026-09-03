"""One-attempt, retrospective joint-event experiment on prior source recordings."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deform360_joint_events_v1 as joint
import numpy as np
import run_deform360_action_kernel_v3 as old
import scipy

base = old.base
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols/deform360_joint_events_v1.json"
BOUND_FILES = (
    "protocols/deform360_joint_events_v1.json",
    "protocols/deform360_action_kernel_v3.json",
    "protocols/deform360_action_conditioned_tactile_v2.json",
    "scripts/science/deform360_joint_events_v1.py",
    "scripts/science/run_deform360_joint_events_v1.py",
    "scripts/science/run_deform360_action_kernel_v3.py",
    "scripts/science/run_deform360_action_conditioned_tactile_v2.py",
    "scripts/science/verify_deform360_joint_events_v1.py",
    "protocols/locks/deform360_joint_events_v1_source_access.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bindings() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in BOUND_FILES}


def extract_reference(archive: Path) -> dict[str, Any]:
    protocol = base.read_json(PROTOCOL)
    if sha256(archive) != protocol["parent_artifact_sha256"]:
        raise ValueError("historical development archive digest mismatch")
    with zipfile.ZipFile(archive) as stream:
        historical = json.loads(stream.read("result.json"))
    roster = base.read_json(ROOT / "protocols/deform360_action_kernel_v3.json")[
        "development_object_ids"
    ]
    objects = []
    for row in historical["objects"]:
        if row["object_id"] not in roster:
            raise ValueError(
                "reference contains an object outside original development"
            )
        descriptors = []
        fingerprints = {}
        for episode_id, action, group in zip(
            row["source_episode_ids"],
            row["source_actions"],
            row["source_fingerprints"],
            strict=True,
        ):
            tactile = []
            medians = []
            robot = None
            for item in group["files"]:
                path = Path(item["path"])
                fingerprints[str(path)] = item
                if path.name in {"robot.npy", "robot.npz"}:
                    robot = path
                elif path.name.startswith("median_"):
                    if not tactile or path.parent != Path(tactile[-1]).parent:
                        raise ValueError("historical median pairing invalid")
                    medians[-1] = str(path)
                else:
                    if "tactile" not in path.parent.name:
                        raise ValueError("unexpected historical source carrier")
                    tactile.append(str(path))
                    medians.append(None)
            if robot is None or len(tactile) < 2:
                raise ValueError("incomplete historical source reference")
            descriptors.append(
                {
                    "object_id": row["object_id"],
                    "episode_id": episode_id,
                    "action": action,
                    "robot_path": str(robot),
                    "tactile_paths": tactile,
                    "median_paths": medians,
                }
            )
        objects.append(
            {
                "object_id": row["object_id"],
                "source_descriptors": descriptors,
                "excluded_original_episode_id": row["target_episode_id"],
                "reference_fingerprints": fingerprints,
            }
        )
    if set(roster) != {row["object_id"] for row in objects}:
        raise ValueError("historical source reference does not cover exact roster")
    return {
        "schema": "deform360-joint-events-source-access-v1",
        "parent_artifact_sha256": sha256(archive),
        "parent_run_id": 33329809775,
        "outcome_fields_copied": False,
        "objects": objects,
    }


def split_descriptors(
    descriptors: list[Any], excluded: int
) -> tuple[list[Any], Any, int]:
    ordered = sorted(descriptors, key=lambda item: item.episode_id)
    if (
        len(ordered) < 3
        or len({item.episode_id for item in ordered}) != len(ordered)
        or ordered[-1].episode_id >= excluded
    ):
        raise ValueError(
            "three distinct original source episodes below excluded target required"
        )
    return ordered[:-1], ordered[-1], excluded


def descriptor_dict(descriptor: Any) -> dict[str, Any]:
    result = dataclasses.asdict(descriptor)
    result["robot_path"] = str(descriptor.robot_path)
    for name in ("tactile_paths", "median_paths"):
        result[name] = [None if item is None else str(item) for item in result[name]]
    return result


def descriptor_from(value: dict[str, Any]) -> Any:
    return base.EpisodeDescriptor(
        object_id=value["object_id"],
        episode_id=value["episode_id"],
        action=value["action"],
        robot_path=Path(value["robot_path"]),
        tactile_paths=tuple(map(Path, value["tactile_paths"])),
        median_paths=tuple(
            None if path is None else Path(path) for path in value["median_paths"]
        ),
    )


def paths_for(descriptor: Any) -> list[Path]:
    return list(
        dict.fromkeys(
            [
                descriptor.robot_path,
                *descriptor.tactile_paths,
                *(p for p in descriptor.median_paths if p is not None),
            ]
        )
    )


def inventory() -> dict[str, Any]:
    protocol = base.read_json(PROTOCOL)
    legacy = base.read_json(ROOT / "protocols/deform360_action_kernel_v3.json")
    root = Path(protocol["dataset_root"])
    objects = []
    unavailable = []
    reference = base.read_json(ROOT / protocol["source_access_reference"])
    if reference["parent_artifact_sha256"] != protocol["parent_artifact_sha256"]:
        raise ValueError("historical source binding changed")
    if [item["object_id"] for item in reference["objects"]] != legacy[
        "development_object_ids"
    ]:
        raise ValueError("historical source roster changed")
    for item in reference["objects"]:
        object_id = item["object_id"]
        descriptors = [descriptor_from(value) for value in item["source_descriptors"]]
        training, evaluation, excluded = split_descriptors(
            descriptors, item["excluded_original_episode_id"]
        )
        files = list(
            dict.fromkeys(
                p for item in training + [evaluation] for p in paths_for(item)
            )
        )
        for path in files:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("carrier symlink escapes dataset root")
        if any(not path.is_file() for path in files):
            unavailable.append(
                {"object_id": object_id, "reason": "historical-source-file-missing"}
            )
            continue
        actual = {str(path): base.sampled_fingerprint(path) for path in files}
        if actual != item["reference_fingerprints"]:
            unavailable.append(
                {
                    "object_id": object_id,
                    "reason": "historical-source-fingerprint-mismatch",
                }
            )
            continue
        objects.append(
            {
                "object_id": object_id,
                "reference_fingerprints": actual,
                "training": [descriptor_dict(item) for item in training],
                "evaluation": descriptor_dict(evaluation),
                "excluded_original_episode_id": excluded,
                "file_sizes": {str(path): path.stat().st_size for path in files},
            }
        )
    return {
        "schema": "deform360-joint-events-inventory-v1",
        "created_at": now(),
        "bindings": bindings(),
        "objects": objects,
        "unavailable": unavailable,
        "source_bytes_read_for_hashing_only": True,
        "tactile_robot_arrays_decoded": False,
        "original_highest_episode_payloads_read": False,
        "retrospective_original_development_roster_only": True,
    }


def load_bound(
    descriptor: Any,
    expected_sizes: dict[str, int],
    expected_fingerprints: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    paths = paths_for(descriptor)
    if any(path.stat().st_size != expected_sizes[str(path)] for path in paths):
        raise ValueError("input size changed since metadata inventory")
    if any(
        base.sampled_fingerprint(path) != expected_fingerprints[str(path)]
        for path in paths
    ):
        raise ValueError("original source carrier fingerprint changed")
    before = {str(path): sha256(path) for path in paths}
    episode = base.load_episode(descriptor)
    after = {str(path): sha256(path) for path in paths}
    if before != after:
        raise ValueError("input changed while loading")
    return episode, before


def flatten_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten_arrays(item, f"{prefix}_{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_arrays(item, f"{prefix}_{index}"))
        return result
    return {prefix: np.asarray(value if value is not None else "none")}


def causal_inputs(
    episode: Any,
    transform: Any,
    protocol: dict[str, Any],
    starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = protocol["model"]
    values = base.normalize_tactile(
        episode.tactile,
        transform.feature_scale,
        float(model["normalized_feature_clip"]),
    )
    current = values[starts]
    state = (current - transform.state_mean) @ transform.state_basis.T
    trends = []
    for lag in model["input_trend_lags_frames"]:
        previous = np.maximum(starts - lag, 0)
        trend = (current - values[previous]) @ transform.state_basis.T
        trends.append(trend / np.maximum(starts - previous, 1)[:, None])
    summary = base.simple_tactile_features(
        current, len(episode.descriptor.tactile_paths)
    )
    state_input = np.concatenate((state, *trends, summary), axis=1)
    action = base.action_features(
        episode.robot_actions, starts, 32, episode.descriptor.action
    )
    return state_input, action, current


def fit_source(source: list[Any], protocol: dict[str, Any]) -> dict[str, Any]:
    recipe = base.read_json(ROOT / "protocols/deform360_action_kernel_v3.json")
    predecessor = base.read_json(
        ROOT / "protocols/deform360_action_conditioned_tactile_v2.json"
    )
    transform = base.build_transform(source, predecessor, 32)
    ridges, _, ridge_fits = old.ridge_candidates(source, transform, predecessor, 32)
    config = recipe["kernel_candidates"]
    specs = [
        old.KernelSpec(k, scale)
        for k in config["neighbor_counts"]
        for scale in config["action_distance_scales"]
    ]
    kernels, source_rows = old.kernel_cv_candidates(
        source, transform, predecessor, 32, specs
    )
    candidates = [
        c for c in ridges + kernels if c.family in {"action_ridge", "action_kernel"}
    ]
    weights, temperature = old.generalized_bayes_weights(
        candidates, recipe["model_averaging"]["temperature_floor_fraction"]
    )
    residuals, _, bias = old.ensemble_source_residuals(source, candidates, weights)
    covariance = base.fit_covariance(residuals, 8, 0.9, mean_error=bias)
    queries = joint.query_bank(residuals.shape[1])
    query_errors = residuals @ queries.T
    projected = (queries * covariance.diagonal) @ queries.T
    projected += (queries @ covariance.factor) @ (queries @ covariance.factor).T
    projected *= covariance.multiplier
    source_truth = np.concatenate(
        [candidates[0].cv_truths[e.descriptor.episode_id] for e in source]
    )
    query_truth = source_truth @ queries.T
    source_mean = query_truth - query_errors
    direct_features = np.concatenate(
        [
            np.concatenate(
                [
                    source_rows[e.descriptor.episode_id][0],
                    source_rows[e.descriptor.episode_id][1],
                ],
                axis=1,
            )
            for e in source
        ]
    )
    direct_features = np.column_stack((direct_features, source_mean))
    thresholds = joint.source_thresholds(query_truth, protocol["threshold_quantile"])
    labels = joint.event_values(query_truth, thresholds)
    logistic = joint.fit_direct_logistic(
        direct_features, labels, protocol["direct_logistic_l2"]
    )
    draws, parity = joint.coupled_draws(query_errors, projected, protocol)
    if (
        parity["sorted_query_marginal_max_error"] != 0
        or parity["shared_point_mean_max_error"] > 1e-12
    ):
        raise ValueError("same-query-marginal or shared-mean contract failed")
    return {
        "transform": transform,
        "bias": bias,
        "covariance": covariance,
        "candidates": candidates,
        "ridge_fits": ridge_fits,
        "kernel_fits": old.fit_all_kernel_models(source_rows, kernels),
        "weights": weights,
        "temperature": temperature,
        "queries": queries,
        "query_errors": query_errors,
        "query_truth": query_truth,
        "source_mean": source_mean,
        "projected_covariance": projected,
        "thresholds": thresholds,
        "logistic": logistic,
        "draws": draws,
        "parity": parity,
        "source_event_rate": (labels.sum(axis=0) + 1) / (len(labels) + 2),
        "predecessor": predecessor,
    }


def predict_episode(episode: Any, fit: dict[str, Any]) -> dict[str, np.ndarray]:
    predecessor = fit["predecessor"]
    starts = base.starts_for(len(episode.tactile), 32, 4, 8)
    if len(starts) < 8:
        raise ValueError("fewer than eight source-evaluation windows")
    state, action, current = causal_inputs(
        episode, fit["transform"], predecessor, starts
    )
    design = np.column_stack((state, action))
    predictions = [
        old.target_candidate_prediction(
            candidate,
            fit["ridge_fits"],
            fit["kernel_fits"],
            (state, action),
            design,
            current,
            fit["transform"],
            5.0,
        )
        for candidate in fit["candidates"]
    ]
    field = np.clip(
        np.einsum("k,kwd->wd", fit["weights"], np.stack(predictions)) + fit["bias"],
        0,
        5,
    )
    mean = field @ fit["queries"].T
    result = joint.event_predictions(mean, fit["thresholds"], fit["draws"])
    result["p_direct_logistic"] = joint.direct_predict(
        fit["logistic"], np.column_stack((design, mean))
    )
    result["p_source_event_rate"] = np.broadcast_to(
        fit["source_event_rate"], (len(mean), 5)
    ).copy()
    result["p_point_event"] = joint.event_values(mean, fit["thresholds"]).astype(float)
    residual_draws = fit["draws"]["structured_gaussian"].reshape(-1, 5)
    result.update(
        {
            "mean": mean,
            "point_field": field,
            "starts": starts,
            "thresholds": fit["thresholds"],
            "query_weights": fit["queries"],
            "lower90": mean + np.quantile(residual_draws, 0.05, axis=0),
            "upper90": mean + np.quantile(residual_draws, 0.95, axis=0),
        }
    )
    return result


def run_object(
    item: dict[str, Any], output: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    output.mkdir()
    training = [descriptor_from(value) for value in item["training"]]
    evaluation = descriptor_from(item["evaluation"])
    if (
        evaluation.episode_id in {e.episode_id for e in training}
        or max(e.episode_id for e in training) >= evaluation.episode_id
        or evaluation.episode_id >= item["excluded_original_episode_id"]
    ):
        raise ValueError("recording split changed")
    files = {}
    source = []
    for descriptor in training:
        episode, hashes = load_bound(
            descriptor, item["file_sizes"], item["reference_fingerprints"]
        )
        source.append(episode)
        files.update(hashes)
    fit = fit_source(source, protocol)
    arrays = flatten_arrays(
        {
            key: value
            for key, value in fit.items()
            if key not in {"candidates", "predecessor"}
        }
    )
    np.savez_compressed(output / "source_fit.npz", **arrays)
    write_new(
        output / "source_seal.json",
        {
            "sealed_at": now(),
            "source_fit_sha256": sha256(output / "source_fit.npz"),
            "training_file_sha256": files,
            "bindings": bindings(),
            "candidate_names": [candidate.name for candidate in fit["candidates"]],
            "evaluation_payload_opened": False,
            "parity": fit["parity"],
        },
    )
    episode, evaluation_hashes = load_bound(
        evaluation, item["file_sizes"], item["reference_fingerprints"]
    )
    predictions = predict_episode(episode, fit)
    np.savez_compressed(output / "predictions.npz", **predictions)
    write_new(
        output / "prediction_seal.json",
        {
            "sealed_at": now(),
            "predictions_sha256": sha256(output / "predictions.npz"),
            "source_seal_sha256": sha256(output / "source_seal.json"),
            "evaluation_file_sha256": evaluation_hashes,
            "event_truth_extracted": False,
            "evaluation_payload_loaded_after_source_seal": True,
            "note": "Recording is already-open source; full bytes loaded, but each forecast uses only its own tactile prefix and allowed known robot path.",
        },
    )
    values = base.normalize_tactile(
        episode.tactile, fit["transform"].feature_scale, 5.0
    )
    truth = values[predictions["starts"] + 32] @ fit["queries"].T
    np.savez_compressed(output / "evaluation_truth.npz", query_truth=truth)
    scored = joint.score_predictions(predictions, truth, fit["thresholds"], protocol)
    scored.update(
        {
            "object_id": item["object_id"],
            "window_count": len(truth),
            "parity": fit["parity"],
            "marginal_coverage90": float(
                np.mean(
                    (truth >= predictions["lower90"])
                    & (truth <= predictions["upper90"])
                )
            ),
            "marginal_width90": float(
                np.mean(predictions["upper90"] - predictions["lower90"])
            ),
            "integration_sd_max": {
                arm: float(predictions[f"integration_sd_{arm}"].max())
                for arm in joint.ARMS
            },
            "scored_at": now(),
            "prediction_seal_sha256": sha256(output / "prediction_seal.json"),
            "evaluation_truth_sha256": sha256(output / "evaluation_truth.npz"),
        }
    )
    write_new(output / "scores.json", scored)
    return scored


def run(inventory_path: Path, output: Path) -> None:
    declared = base.read_json(inventory_path)
    if declared["bindings"] != bindings():
        raise ValueError("source or protocol changed after metadata inventory")
    current = inventory()
    if (
        declared["objects"] != current["objects"]
        or declared["unavailable"] != current["unavailable"]
    ):
        raise ValueError("carrier inventory changed; no empirical attempt launched")
    output.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    protocol = base.read_json(PROTOCOL)
    write_new(
        output / "attempt.json",
        {
            "started_at": now(),
            "launch_count": 1,
            "no_retry": True,
            "inventory_sha256": sha256(inventory_path),
            "bindings": bindings(),
            "git_revision": os.environ.get("JOINT_EVENTS_REVISION"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cpu_only": True,
            "boundaries": protocol["boundaries"],
        },
    )
    rows = []
    failures = []
    for index, item in enumerate(declared["objects"]):
        print(
            json.dumps(
                {
                    "stage": "source-fit",
                    "ordinal": index + 1,
                    "total": len(declared["objects"]),
                }
            ),
            flush=True,
        )
        try:
            row = run_object(item, output / item["object_id"], protocol)
            rows.append(row)
        except Exception as error:
            failure = {
                "object_id": item["object_id"],
                "type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            write_new(output / f"failure-{index:02d}.json", failure)
        print(
            json.dumps(
                {
                    "stage": "recording-complete",
                    "completed": len(rows),
                    "technical_failures": len(failures),
                }
            ),
            flush=True,
        )
    summary = joint.aggregate(rows, protocol)
    if failures:
        summary["superiority_gate"] = False
        summary["decision_gate"] = False
    result = {
        "schema": "deform360-joint-events-result-v1",
        "finished_at": now(),
        "elapsed_seconds": time.monotonic() - started,
        "summary": summary,
        "objects": rows,
        "technical_failures": failures,
        "unavailable": declared["unavailable"],
        "attempt_sha256": sha256(output / "attempt.json"),
        "bindings": bindings(),
        "claim_boundary": protocol["interpretation"],
        "boundaries": protocol["boundaries"],
    }
    write_new(output / "result.json", result)
    print(
        json.dumps(
            {
                "stage": "complete",
                "objects": len(rows),
                "failures": len(failures),
                "result_sha256": sha256(output / "result.json"),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "inventory", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--parent-archive", type=Path)
    args = parser.parse_args()
    if args.mode == "reference":
        if args.parent_archive is None:
            parser.error("--parent-archive required")
        write_new(args.output, extract_reference(args.parent_archive))
    elif args.mode == "inventory":
        value = inventory()
        write_new(args.output, value)
        print(
            json.dumps(
                {
                    "objects": len(value["objects"]),
                    "unavailable": len(value["unavailable"]),
                    "arrays_decoded": False,
                    "sha256": sha256(args.output),
                }
            )
        )
    else:
        if args.inventory is None:
            parser.error("--inventory required")
        run(args.inventory, args.output)


if __name__ == "__main__":
    main()
