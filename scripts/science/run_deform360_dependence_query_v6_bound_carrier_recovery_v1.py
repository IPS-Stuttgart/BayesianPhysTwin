#!/usr/bin/env python3
"""Recover the frozen Deform360 same-mean decision study from carrier-superset drift.

The failed v6 execution compared the complete live carrier inventory against the
older readiness snapshot after evaluating 91 of 92 objects. This recovery keeps
the scientific protocol, frozen point predictor, query bank, covariance arms,
calibration, decision rule, bootstrap, and success gates unchanged. It replaces
only the mutable full-tree rediscovery with exact descriptors reconstructed from
the parent confirmation's per-episode sampled fingerprints.

Every bound robot, tactile, and median file must still match its parent receipt.
Actions for every bound episode must also match current metadata. Additional
unbound files or episodes are recorded but never selected or numerically opened.
The exact parent point result must reproduce for all 92 objects before any
same-mean dependence conclusion is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = (
    "bayesian-phystwin/deform360-dependence-query-result-v6-"
    "bound-carrier-recovery-v1"
)
RECOVERY_SCHEMA = (
    "bayesian-phystwin/deform360-dependence-query-bound-carrier-recovery-v1"
)
FAILED_RUN_ID = 33442211234
ORIGINAL_V6_REVISION = "954538832106d8ded13f1101b3a2b2e855b40513"
ORIGINAL_V6_RUNNER_SHA256 = (
    "06c22fc3fe667c4f2f11eddee3dcb1b78b5465b6312136efb010611e1ebab91c"
)
_EPS = 1e-12


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sampled_fingerprint(path: Path) -> dict[str, Any]:
    """Mirror the frozen v2 sampled-file fingerprint for the self-test."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if size > 1024 * 1024:
            stream.seek(max(size - 1024 * 1024, 0))
            digest.update(stream.read(1024 * 1024))
    return {
        "path": str(path),
        "size_bytes": int(size),
        "sampled_sha256": digest.hexdigest(),
        "rule": "sha256(size || first_1MiB || last_1MiB)",
    }


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _bound_path(path_text: object, data_root: Path) -> Path:
    path = Path(str(path_text))
    if not path.is_absolute():
        raise ValueError(f"bound carrier path is not absolute: {path}")
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"bound carrier escapes dataset root: {path}") from exc
    if not path.is_file():
        raise ValueError(f"bound carrier is missing: {path}")
    return path


def _verify_fingerprint(base: Any, expected_value: object, data_root: Path) -> Path:
    expected = dict(_require_mapping(expected_value, "fingerprint file"))
    path = _bound_path(expected.get("path"), data_root)
    actual = base.sampled_fingerprint(path)
    if actual != expected:
        raise ValueError(f"bound carrier fingerprint changed: {path}")
    return path


def descriptor_from_receipt(
    base: Any,
    *,
    data_root: Path,
    object_id: str,
    episode_id: int,
    action: str | None,
    receipt_value: object,
) -> Any:
    """Reconstruct one exact frozen descriptor from its parent file receipt."""
    receipt = _require_mapping(receipt_value, "episode fingerprint receipt")
    files = list(_require_sequence(receipt.get("files"), "fingerprint files"))
    if int(receipt.get("file_count", -1)) != len(files) or not files:
        raise ValueError(f"invalid fingerprint receipt for {object_id} episode {episode_id}")

    paths = [_verify_fingerprint(base, item, data_root) for item in files]
    robot_indices = [
        index
        for index, path in enumerate(paths)
        if path.name in {"robot.npy", "robot.npz"} and path.parent.name == "robot"
    ]
    if robot_indices != [0]:
        raise ValueError(
            f"receipt must start with exactly one robot carrier: {object_id} "
            f"episode {episode_id}"
        )

    tactile_paths: list[Path] = []
    median_paths: list[Path | None] = []
    for path in paths[1:]:
        if path.name.lower().startswith("median_"):
            if (
                not tactile_paths
                or median_paths[-1] is not None
                or path.parent != tactile_paths[-1].parent
            ):
                raise ValueError(
                    f"median ordering changed for {object_id} episode {episode_id}"
                )
            median_paths[-1] = path
            continue
        if path.suffix.lower() != ".npy" or "tactile" not in path.parent.name.lower():
            raise ValueError(
                f"unexpected non-tactile bound carrier for {object_id} "
                f"episode {episode_id}: {path}"
            )
        tactile_paths.append(path)
        median_paths.append(None)

    if len(tactile_paths) < 2:
        raise ValueError(
            f"too few bound tactile carriers for {object_id} episode {episode_id}"
        )
    return base.EpisodeDescriptor(
        object_id=object_id,
        episode_id=int(episode_id),
        action=action,
        robot_path=paths[0],
        tactile_paths=tuple(tactile_paths),
        median_paths=tuple(median_paths),
    )


def readiness_identities(value: Mapping[str, Any]) -> set[tuple[str, int]]:
    result = {
        (str(item["relative_path"]), int(item["size_bytes"]))
        for item in value.get("robot_files", ())
    }
    for group in value.get("tactile_groups", ()):
        result.update(
            (str(item["relative_path"]), int(item["size_bytes"]))
            for item in group.get("recordings", ())
        )
    return result


def current_identities(value: Mapping[str, Any]) -> set[tuple[str, int]]:
    return readiness_identities(value)


def metadata_actions(base: Any, path: Path) -> dict[int, str | None]:
    rows = base.episode_records(base.read_json(path))
    return {int(row["episode_id"]): row.get("action") for row in rows}


def build_bound_descriptors(
    *,
    v3: Any,
    v5: Any,
    audit: Any,
    data_root: Path,
    expected: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    minimum_episodes: int,
) -> tuple[list[Any], dict[str, Any]]:
    object_id = str(expected["object_id"])
    source_ids = [int(value) for value in parent_row["source_episode_ids"]]
    source_actions = list(parent_row["source_actions"])
    if len(source_ids) != len(source_actions):
        raise ValueError(f"source action roster is incomplete: {object_id}")
    target_id = int(parent_row["target_episode_id"])
    target_action = parent_row.get("target_action")
    expected_ids = [int(value) for value in expected["complete_episode_ids"]]
    if source_ids + [target_id] != expected_ids:
        raise ValueError(f"parent episode roster differs from readiness: {object_id}")
    if target_id != int(expected["target_episode_id"]):
        raise ValueError(f"parent target episode differs from readiness: {object_id}")
    if target_action != expected.get("target_action"):
        raise ValueError(f"parent target action differs from readiness: {object_id}")

    source_receipts = list(parent_row["source_fingerprints"])
    if len(source_receipts) != len(source_ids):
        raise ValueError(f"source fingerprint roster is incomplete: {object_id}")
    receipts = source_receipts + [parent_row["target_fingerprint"]]
    actions = source_actions + [target_action]
    descriptors = [
        descriptor_from_receipt(
            v3.base,
            data_root=data_root,
            object_id=object_id,
            episode_id=episode_id,
            action=action,
            receipt_value=receipt,
        )
        for episode_id, action, receipt in zip(
            expected_ids,
            actions,
            receipts,
            strict=True,
        )
    ]
    if len(descriptors) < minimum_episodes:
        raise ValueError(f"bound descriptor roster is too small: {object_id}")

    metadata_path = (
        data_root / "raw-repository" / "raw" / object_id / "metadata.json"
    )
    if not metadata_path.is_file():
        raise ValueError(f"metadata disappeared for bound object: {object_id}")
    actions_by_id = metadata_actions(v3.base, metadata_path)
    expected_actions = dict(zip(expected_ids, actions, strict=True))
    for episode_id, action in expected_actions.items():
        if actions_by_id.get(episode_id) != action:
            raise ValueError(
                f"bound metadata action changed: {object_id} episode {episode_id}"
            )

    current = audit.inspect_object(data_root, object_id, minimum_episodes)
    expected_identity = readiness_identities(expected)
    current_identity = current_identities(current)
    missing = sorted(expected_identity - current_identity)
    if missing:
        raise ValueError(f"bound readiness carriers disappeared: {object_id}: {missing}")
    projection = v5.selection_projection(current)
    extra_ids = sorted(
        set(map(int, current.get("complete_episode_ids", ()))) - set(expected_ids)
    )
    drift = {
        "object_id": object_id,
        "full_readiness_projection_equal": projection == dict(expected),
        "metadata_sha256_equal": (
            sha256_file(metadata_path) == str(expected["metadata_sha256"])
        ),
        "bound_episode_actions_equal": True,
        "bound_readiness_identities_present": True,
        "bound_numeric_fingerprints_equal": True,
        "current_object_eligible_under_live_full_tree": bool(current.get("eligible")),
        "extra_complete_episode_ids": extra_ids,
        "extra_readiness_identity_count": len(current_identity - expected_identity),
        "bound_episode_ids": expected_ids,
        "bound_carrier_receipt_sha256": canonical_digest(
            {
                "object_id": object_id,
                "episode_ids": expected_ids,
                "actions": actions,
                "fingerprints": receipts,
            }
        ),
        "unbound_numeric_payloads_opened": False,
    }
    return descriptors, drift


def run(
    *,
    base_runner_path: Path,
    protocol_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> tuple[Any, dict[str, Any]]:
    base_runner_path = base_runner_path.resolve(strict=True)
    if sha256_file(base_runner_path) != ORIGINAL_V6_RUNNER_SHA256:
        raise ValueError("original v6 runner bytes changed")
    v6 = load_module(base_runner_path, "deform360_dependence_query_v6_original")

    protocol = v6.read_json(protocol_path)
    parent_result = v6.read_json(parent_result_path)
    parent_protocol = v6.read_json(parent_protocol_path)
    data_root = data_root.resolve(strict=True)
    parent_control_root = parent_control_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    v6.validate_protocol(
        protocol,
        parent_control_root=parent_control_root,
        parent_protocol_path=parent_protocol_path,
        data_root=data_root,
    )
    parent_by_object = v6.validate_parent_result(
        parent_result,
        protocol,
        parent_result_path,
    )

    parent_binding = protocol["parent_confirmation"]
    v5_path = parent_control_root / str(parent_binding["runner_path"])
    v5 = v6.load_module(v5_path, "deform360_v5_parent_for_bound_recovery")
    manifest = v5.verify_readiness(
        v6.read_json(readiness_path),
        parent_protocol,
        readiness_path,
    )
    v3, development, base_protocol = v5.validate_frozen_method(
        frozen_root,
        parent_protocol,
    )
    audit_path = parent_control_root / str(parent_binding["audit_path"])
    audit = v6.load_module(audit_path, "deform360_v5_audit_for_bound_recovery")
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])

    evaluation = protocol["evaluation"]
    point_rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    carrier_drift: list[dict[str, Any]] = []
    for index, expected_value in enumerate(manifest, start=1):
        expected = _require_mapping(expected_value, "readiness manifest row")
        object_id = str(expected["object_id"])
        print(
            f"[{index}/{len(manifest)}] bound-carrier dependence-query "
            f"evaluation {object_id}",
            flush=True,
        )
        parent_row = _require_mapping(parent_by_object[object_id], "parent object row")
        descriptors, drift = build_bound_descriptors(
            v3=v3,
            v5=v5,
            audit=audit,
            data_root=data_root,
            expected=expected,
            parent_row=parent_row,
            minimum_episodes=minimum,
        )
        carrier_drift.append(drift)

        row, capture, source_truth, target_truth = v6.evaluate_object_with_capture(
            v3,
            descriptors,
            development,
            base_protocol,
            point_rng,
        )
        exact_point = v6.point_projection(row) == v6.point_projection(parent_row)
        if not exact_point:
            raise RuntimeError(
                f"exact frozen point result did not reproduce: {object_id}"
            )

        target_errors = np.asarray(capture.target_errors, dtype=np.float64)
        source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
        predicted_mean = target_truth - target_errors
        arms = v6.covariance_arms(
            v3.base,
            capture.covariance,
            seed=v6.stable_seed(
                int(evaluation["random_seed"]),
                object_id,
                "scrambled-factor",
            ),
        )
        reference_marginal = v6.marginal_variance(arms["full_low_rank"])
        marginal_parity = float(
            max(
                np.max(
                    np.abs(v6.marginal_variance(model) - reference_marginal)
                )
                for model in arms.values()
            )
        )
        queries: dict[str, Any] = {}
        bank = v6.query_bank(target_truth.shape[1])
        centered_source_errors = source_errors - source_errors.mean(
            axis=0,
            keepdims=True,
        )
        for query_name, (weight, event) in bank.items():
            raw_variances = {
                arm_name: v6.covariance_query_variance(model, weight)
                for arm_name, model in arms.items()
            }
            calibration = v6.source_query_calibration(
                centered_source_errors,
                source_truth,
                weight,
                raw_variances,
                event=event,
                probability=float(evaluation["coverage_probability"]),
                event_quantile=float(evaluation["event_threshold_quantile"]),
            )
            query_record = {
                "event": event,
                "weight_sha256": v6.array_digest(weight),
                "calibration": calibration,
                "arms": {},
            }
            for arm_name, model in arms.items():
                query_record["arms"][arm_name] = v6.query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=float(evaluation["fallback_cost"]),
                    probability_clip=float(evaluation["probability_clip"]),
                )
            queries[query_name] = query_record

        arm_summary: dict[str, dict[str, float]] = {}
        for arm_name in v6.COVARIANCE_ARMS:
            values = [
                queries[query_name]["arms"][arm_name]
                for query_name, _ in v6.QUERY_SPECS
            ]
            arm_summary[arm_name] = {
                metric: float(np.mean([value[metric] for value in values]))
                for metric in (
                    "target_query_nanees",
                    "target_90_coverage",
                    "mean_90_interval_width",
                    "query_nll",
                    "event_brier",
                    "event_log_loss",
                    "decision_loss",
                    "decision_regret",
                    "acceptance_fraction",
                    "harmful_accept_fraction_all",
                    "harmful_accept_rate_given_accept",
                )
            }
            arm_summary[arm_name]["calibration_log_error"] = float(
                np.mean(
                    [
                        abs(math.log(max(value["target_query_nanees"], _EPS)))
                        for value in values
                    ]
                )
            )
            arm_summary[arm_name]["coverage_absolute_error"] = float(
                np.mean(
                    [
                        abs(
                            value["target_90_coverage"]
                            - float(evaluation["coverage_probability"])
                        )
                        for value in values
                    ]
                )
            )

        result_row = {
            "object_id": object_id,
            "source_episode_ids": row["source_episode_ids"],
            "target_episode_id": row["target_episode_id"],
            "target_action": row["target_action"],
            "target_action_family": row["target_action_family"],
            "dimension": int(target_truth.shape[1]),
            "window_count": int(target_truth.shape[0]),
            "predictive_mean_sha256": v6.array_digest(predicted_mean),
            "same_mean_by_construction": True,
            "parent_point_result_exact": exact_point,
            "coordinate_marginal_parity_max_abs": marginal_parity,
            "query_bank_sha256": v6.canonical_digest(
                {
                    name: {
                        "event": event,
                        "weight_sha256": queries[name]["weight_sha256"],
                    }
                    for name, event in v6.QUERY_SPECS
                }
            ),
            "queries": queries,
            "arm_summary": arm_summary,
            "joint_metrics": {
                name: v6.joint_metrics(
                    v3.base,
                    target_errors,
                    model,
                    float(evaluation["coverage_probability"]),
                )
                for name, model in arms.items()
            },
            "bound_carrier_recovery": drift,
        }
        rows.append(result_row)

    summary, decision = v6.aggregate(rows, protocol)
    drift_summary = {
        "object_count": len(carrier_drift),
        "full_projection_equal_count": sum(
            bool(row["full_readiness_projection_equal"]) for row in carrier_drift
        ),
        "metadata_sha256_equal_count": sum(
            bool(row["metadata_sha256_equal"]) for row in carrier_drift
        ),
        "objects_with_extra_complete_episodes": [
            row["object_id"]
            for row in carrier_drift
            if row["extra_complete_episode_ids"]
        ],
        "objects_with_extra_readiness_identities": [
            row["object_id"]
            for row in carrier_drift
            if int(row["extra_readiness_identity_count"]) > 0
        ],
        "all_bound_numeric_fingerprints_equal": all(
            bool(row["bound_numeric_fingerprints_equal"])
            for row in carrier_drift
        ),
        "all_bound_episode_actions_equal": all(
            bool(row["bound_episode_actions_equal"]) for row in carrier_drift
        ),
        "unbound_numeric_payloads_opened": False,
        "carrier_drift_sha256": canonical_digest(carrier_drift),
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "parent_confirmation": protocol["parent_confirmation"],
        "bound_selection_manifest_sha256": parent_protocol["readiness_binding"][
            "selection_manifest_sha256"
        ],
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "failed_run_id": FAILED_RUN_ID,
            "original_v6_revision": ORIGINAL_V6_REVISION,
            "original_v6_runner_sha256": ORIGINAL_V6_RUNNER_SHA256,
            "scientific_protocol_changed": False,
            "point_predictor_changed": False,
            "query_bank_changed": False,
            "covariance_arms_changed": False,
            "calibration_or_decision_rule_changed": False,
            "bootstrap_or_success_gates_changed": False,
            "repair_scope": (
                "replace mutable full-tree rediscovery with exact parent-bound "
                "per-episode carrier receipts"
            ),
        },
        "carrier_drift_summary": drift_summary,
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_original_v6_scientific_protocol_reused": True,
            "exact_original_v6_analysis_functions_reused": True,
            "exact_frozen_v3_point_predictor_reused": True,
            "parent_point_result_reproduced_exactly": all(
                row["parent_point_result_exact"] for row in rows
            ),
            "exact_parent_bound_numeric_carriers_reused": True,
            "live_unbound_superset_used_for_selection": False,
            "unbound_numeric_payloads_opened": False,
            "same_mean_across_covariance_arms": True,
            "coordinate_marginals_matched": True,
            "query_scales_thresholds_and_radii_source_only": True,
            "target_outcomes_used_for_protocol_or_arm_selection": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
        },
        "summary": summary,
        "decision": decision,
        "objects": rows,
        "carrier_drift": carrier_drift,
        "protocol": protocol,
    }
    result["result_sha256"] = v6.canonical_digest(result)
    return v6, result


def make_report(v6: Any, result: dict[str, Any]) -> str:
    drift = result["carrier_drift_summary"]
    recovery = result["recovery"]
    lines = [
        "# Bound-carrier technical recovery",
        "",
        f"- Failed predecessor run: **{recovery['failed_run_id']}**",
        "- Scientific protocol changed: **false**",
        "- Frozen point predictor changed: **false**",
        "- Recovery: exact parent-bound episode carriers and actions",
        f"- Objects with full live-manifest equality: "
        f"**{drift['full_projection_equal_count']}/{drift['object_count']}**",
        f"- Objects with additive complete episodes: "
        f"**{len(drift['objects_with_extra_complete_episodes'])}**",
        "- Every bound numeric fingerprint reproduced: **true**",
        "- Every bound action reproduced: **true**",
        "- Unbound numeric payloads opened: **false**",
        "",
        "The live tree may contain additions made after the original readiness",
        "snapshot. Those additions are inventoried but excluded. All scientific",
        "metrics below are produced by the exact original v6 analysis functions",
        "on the exact parent-bound numeric carriers.",
        "",
    ]
    return "\n".join(lines) + "\n" + v6.make_report(result)


@dataclass(frozen=True)
class _DummyDescriptor:
    object_id: str
    episode_id: int
    action: str | None
    robot_path: Path
    tactile_paths: tuple[Path, ...]
    median_paths: tuple[Path | None, ...]


class _DummyBase:
    EpisodeDescriptor = _DummyDescriptor
    sampled_fingerprint = staticmethod(sampled_fingerprint)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        robot = root / "processed-repository/processed/x/episode_0/robot/robot.npy"
        left = root / "raw-repository/raw/x/tactile_left/left.npy"
        left_median = root / "raw-repository/raw/x/tactile_left/median_left.npy"
        right = root / "raw-repository/raw/x/tactile_right/right.npy"
        right_median = root / "raw-repository/raw/x/tactile_right/median_right.npy"
        for index, path in enumerate(
            (robot, left, left_median, right, right_median),
            start=1,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([index]) * (31 + index))
        files = [
            sampled_fingerprint(path)
            for path in (robot, left, left_median, right, right_median)
        ]
        receipt = {"file_count": len(files), "files": files}
        descriptor = descriptor_from_receipt(
            _DummyBase,
            data_root=root,
            object_id="x",
            episode_id=0,
            action="lift",
            receipt_value=receipt,
        )
        assert descriptor.robot_path == robot
        assert descriptor.tactile_paths == (left, right)
        assert descriptor.median_paths == (left_median, right_median)

        extra = root / "processed-repository/processed/x/episode_1/robot/robot.npy"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"extra")
        repeated = descriptor_from_receipt(
            _DummyBase,
            data_root=root,
            object_id="x",
            episode_id=0,
            action="lift",
            receipt_value=receipt,
        )
        assert repeated == descriptor

        left.write_bytes(b"changed")
        try:
            descriptor_from_receipt(
                _DummyBase,
                data_root=root,
                object_id="x",
                episode_id=0,
                action="lift",
                receipt_value=receipt,
            )
        except ValueError as exc:
            assert "fingerprint changed" in str(exc)
        else:
            raise AssertionError("changed bound file was not rejected")
    print("bound-carrier recovery self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-runner", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    required_names = (
        "base_runner",
        "protocol",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "data_root",
        "parent_control_root",
        "frozen_root",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required_names if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    v6, result = run(
        base_runner_path=args.base_runner,
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        data_root=args.data_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
    )
    v6.write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(v6, result), encoding="utf-8")
    v6.write_object_csv(args.output_csv, result["objects"])
    print(json.dumps(result["carrier_drift_summary"], indent=2, sort_keys=True))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
