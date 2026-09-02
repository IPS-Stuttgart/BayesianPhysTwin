"""Reproduce the frozen Deform360 action TCN and capture query residuals.

The wrapper imports the exact execution revision, verifies the immutable carrier
cache byte-for-byte, reruns the registered training/evaluation protocol, and requires
full scientific equality with the retained TCN artifact.  Window-level query errors
and centered Monte-Carlo-dropout query covariances are exported only after that
reproduction contract has passed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-exact-action-tcn-query-capture-v9"
TCN_RESULT_SCHEMA = "bayesian-phystwin/deform360-tcn-baseline-audit-v6"
TCN_EXECUTION_REVISION = "a32948698fe43e4e52a443a93c9c1604012a21cf"
TCN_RESULT_SHA256 = "4905c9958e1290f60abcae5b3248cbcca1ac1dfb824144ed4b76909cf5fa1333"
TCN_ARTIFACT_SHA256 = "68ff5fb8e2274f3e71a20ef79a7450d0baa0cf57985e88064b31e4c1edc2bca2"
EXPECTED_OBJECT_COUNT = 92


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare_values(
    observed: Any,
    expected: Any,
    *,
    path: str = "root",
    atol: float = 1e-11,
    rtol: float = 1e-11,
) -> float:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"mapping keys changed at {path}")
        return max(
            (
                compare_values(
                    observed[key],
                    expected[key],
                    path=f"{path}.{key}",
                    atol=atol,
                    rtol=rtol,
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"sequence shape changed at {path}")
        return max(
            (
                compare_values(
                    left,
                    right,
                    path=f"{path}[{index}]",
                    atol=atol,
                    rtol=rtol,
                )
                for index, (left, right) in enumerate(
                    zip(observed, expected, strict=True)
                )
            ),
            default=0.0,
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise ValueError(f"value changed at {path}: {observed!r} != {expected!r}")
        return 0.0
    if isinstance(expected, int) and not isinstance(expected, bool):
        if observed != expected:
            raise ValueError(f"integer changed at {path}: {observed!r} != {expected!r}")
        return 0.0
    if isinstance(expected, float):
        left = float(observed)
        right = float(expected)
        if not np.isclose(left, right, rtol=rtol, atol=atol):
            raise ValueError(f"numeric value changed at {path}: {left} != {right}")
        return abs(left - right)
    if observed != expected:
        raise ValueError(f"value changed at {path}")
    return 0.0


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def validate_tcn_artifact(
    artifact_root: Path,
    exact_root: Path,
    data_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = read_json(artifact_root / "result.json")
    protocol = read_json(artifact_root / "effective-protocol.json")
    manifest = read_json(artifact_root / "carrier-manifest.json")
    unsigned = dict(result)
    stored_digest = unsigned.pop("result_sha256", None)
    if result.get("schema") != TCN_RESULT_SCHEMA or result.get("status") != "complete":
        raise ValueError("immutable TCN result is not complete v6 evidence")
    if (
        stored_digest != TCN_RESULT_SHA256
        or canonical_digest(unsigned) != stored_digest
    ):
        raise ValueError("immutable TCN result digest changed")
    if result.get("github_sha") != TCN_EXECUTION_REVISION:
        raise ValueError("immutable TCN execution revision changed")
    if result.get("protocol") != protocol:
        raise ValueError("effective TCN protocol differs from the retained result")
    if Path(str(result["dataset_root"])) != data_root:
        raise ValueError("TCN dataset root differs from the retained execution")
    if manifest.get("status") != "complete" or manifest.get("file_count") != len(
        manifest.get("files", [])
    ):
        raise ValueError("TCN carrier manifest is incomplete")
    if Path(str(manifest["output_root"])) != data_root:
        raise ValueError("TCN carrier root differs from its manifest")
    if len(result.get("objects", [])) != EXPECTED_OBJECT_COUNT:
        raise ValueError("TCN evaluation roster changed")
    if (
        os.popen(f"git -C {exact_root!s} rev-parse HEAD").read().strip()
        != TCN_EXECUTION_REVISION
    ):
        raise ValueError("exact TCN checkout revision changed")
    return result, protocol, manifest


def verify_carrier_cache(
    data_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    total_size = 0
    checked = 0
    for record in manifest["files"]:
        relative = Path(str(record["destination"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe TCN carrier path: {relative}")
        path = data_root / relative
        if not path.is_file():
            raise ValueError(f"TCN carrier is missing: {relative}")
        size = int(path.stat().st_size)
        if size != int(record["size_bytes"]):
            raise ValueError(f"TCN carrier size changed: {relative}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"TCN carrier digest changed: {relative}")
        total_size += size
        checked += 1
    if checked != int(manifest["file_count"]):
        raise ValueError("TCN carrier count changed")
    if total_size != int(manifest["total_size_bytes"]):
        raise ValueError("TCN carrier byte count changed")
    return {
        "file_count": checked,
        "total_size_bytes": total_size,
        "raw_revision": manifest["raw_revision"],
        "processed_revision": manifest["processed_revision"],
        "all_sha256_verified": True,
    }


def _dropout_query_covariance(
    core: Any,
    model: Any,
    samples: Any,
    scaler: Any,
    config: Mapping[str, Any],
    device: Any,
    query: np.ndarray,
    draw_count: int,
    seed: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tactile, action, static, _, _ = core.standardized_arrays(samples, scaler, True)
    dataset = TensorDataset(
        torch.from_numpy(tactile),
        torch.from_numpy(action),
        torch.from_numpy(static),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    draws: list[np.ndarray] = []
    model.train()
    with torch.no_grad():
        for draw in range(draw_count):
            torch.manual_seed(seed + draw)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + draw)
            rows: list[np.ndarray] = []
            for tactile_batch, action_batch, static_batch in loader:
                standardized = (
                    model(
                        tactile_batch.to(device),
                        action_batch.to(device),
                        static_batch.to(device),
                    )
                    .cpu()
                    .numpy()
                )
                delta = (
                    scaler.target_mean[None, :]
                    + standardized * scaler.target_std[None, :]
                )
                rows.append(delta)
            prediction = np.clip(
                samples.current + np.concatenate(rows),
                0.0,
                float(config["normalized_feature_clip"]),
            )
            draws.append(prediction @ query.T)
    model.eval()
    values = np.stack(draws, axis=0)
    centered = values - values.mean(axis=0, keepdims=True)
    return np.einsum("dci,dcj->cij", centered, centered) / float(draw_count - 1)


def reproduce_and_capture(
    artifact_root: Path,
    exact_root: Path,
    data_root: Path,
    query_matrix: np.ndarray,
    draw_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact_root = artifact_root.resolve(strict=True)
    exact_root = exact_root.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    reference, protocol, carrier_manifest = validate_tcn_artifact(
        artifact_root, exact_root, data_root
    )
    carrier_audit = verify_carrier_cache(data_root, carrier_manifest)
    script = exact_root / "scripts" / "science" / "run_deform360_tcn_baseline_v6.py"
    module = load_module(script, "deform360_exact_action_tcn_v9")
    core = module.core
    original_evaluate = core.evaluate_model
    original_predict = core.predict_delta
    captures: list[dict[str, Any]] = []

    def intercept_evaluate(
        model: Any,
        evaluation: Sequence[tuple[Any, Any]],
        scaler: Any,
        use_action: bool,
        config: Mapping[str, Any],
        device: Any,
    ) -> tuple[float, dict[str, float]]:
        result = original_evaluate(
            model, evaluation, scaler, use_action, config, device
        )
        if use_action and len(evaluation) == EXPECTED_OBJECT_COUNT:
            for object_index, (source_object, samples) in enumerate(evaluation):
                delta = original_predict(
                    model,
                    samples,
                    scaler,
                    True,
                    int(config["evaluation_batch_size"]),
                    device,
                )
                prediction = np.clip(
                    samples.current + delta,
                    0.0,
                    float(config["normalized_feature_clip"]),
                )
                residual = (samples.truth - prediction) @ query_matrix.T
                dropout_covariance = _dropout_query_covariance(
                    core,
                    model,
                    samples,
                    scaler,
                    config,
                    device,
                    query_matrix,
                    draw_count,
                    2026090200 + 1000 * object_index,
                )
                captures.append(
                    {
                        "object_id": source_object.object_id,
                        "target_episode_id": int(
                            source_object.target_descriptor.episode_id
                        ),
                        "query_residuals": residual,
                        "dropout_query_covariances": dropout_covariance,
                        "window_indices": np.arange(len(samples), dtype=np.int32),
                    }
                )
        return result

    core.evaluate_model = intercept_evaluate
    saved_environment = {
        key: os.environ.get(key) for key in ("GITHUB_SHA", "RUNNER_NAME")
    }
    os.environ["GITHUB_SHA"] = str(reference["github_sha"])
    os.environ["RUNNER_NAME"] = str(reference["runner_name"])
    try:
        with working_directory(exact_root):
            reproduced = module.run(
                artifact_root / "effective-protocol.json", data_root
            )
    finally:
        for key, value in saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    maximum_difference = compare_values(reproduced, reference)
    if len(captures) != EXPECTED_OBJECT_COUNT:
        raise ValueError("final action-TCN capture did not cover all objects")
    capture_by_object = {row["object_id"]: row for row in captures}
    reference_ids = [str(row["object_id"]) for row in reference["objects"]]
    if set(capture_by_object) != set(reference_ids):
        raise ValueError("action-TCN capture object roster changed")
    ordered = [capture_by_object[object_id] for object_id in reference_ids]
    arrays = {
        "query_residuals": np.concatenate([row["query_residuals"] for row in ordered]),
        "dropout_query_covariances": np.concatenate(
            [row["dropout_query_covariances"] for row in ordered]
        ),
        "object_ids": np.concatenate(
            [np.repeat(row["object_id"], len(row["window_indices"])) for row in ordered]
        ),
        "target_episode_ids": np.concatenate(
            [
                np.repeat(row["target_episode_id"], len(row["window_indices"]))
                for row in ordered
            ]
        ).astype(np.int32),
        "window_indices": np.concatenate([row["window_indices"] for row in ordered]),
        "query_matrix": np.asarray(query_matrix, dtype=np.float64),
    }
    capture_manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 9,
        "status": "exact-tcn-reproduced-and-query-captured",
        "tcn_execution_revision": TCN_EXECUTION_REVISION,
        "tcn_result_sha256": TCN_RESULT_SHA256,
        "tcn_artifact_sha256": TCN_ARTIFACT_SHA256,
        "exact_scientific_result_reproduction": True,
        "maximum_absolute_numeric_difference": maximum_difference,
        "object_count": EXPECTED_OBJECT_COUNT,
        "case_count": int(arrays["query_residuals"].shape[0]),
        "query_dimension": int(arrays["query_residuals"].shape[1]),
        "dropout_draw_count": draw_count,
        "carrier_audit": carrier_audit,
        "final_model_freeze": reference["final_model_freeze"],
        "action_conditioned_tcn_active_field_rmse": reference["summary"]["methods"][
            "action_conditioned_tcn"
        ],
        "paper_claim_authorized": False,
    }
    capture_manifest["result_sha256"] = canonical_digest(capture_manifest)
    return capture_manifest, arrays


def self_test() -> None:
    compare_values({"x": [1.0, True]}, {"x": [1.0 + 1e-13, True]})
    if canonical_digest({"b": 2, "a": 1}) != canonical_digest({"a": 1, "b": 2}):
        raise AssertionError("canonical digest is order dependent")
    print("exact Deform360 action-TCN query capture v9 self-test passed")
