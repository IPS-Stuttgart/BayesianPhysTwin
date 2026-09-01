"""Exact frozen Deform360 confirmation reproduction and scoring capture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import deform360_query_covariance_v7 as query_cov
import numpy as np

CONFIRMATION_SCHEMA = "bayesian-phystwin/deform360-untouched-confirmation-result-v5"
READINESS_SCHEMA = "bayesian-phystwin/deform360-untouched-readiness-v5"
CONFIRMATION_REVISION = "e409527b8499a225bde8bd7f8c532a30e96548c6"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def compare_values(
    observed: Any,
    expected: Any,
    *,
    path: str = "root",
    atol: float = 1e-11,
    rtol: float = 1e-11,
) -> float:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
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
            raise ValueError(
                f"value changed at {path}: {observed!r} != {expected!r}"
            )
        return 0.0
    if isinstance(expected, int) and not isinstance(expected, bool):
        if observed != expected:
            raise ValueError(
                f"integer changed at {path}: {observed!r} != {expected!r}"
            )
        return 0.0
    if isinstance(expected, float):
        left = float(observed)
        right = float(expected)
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            raise ValueError(
                f"numeric value changed at {path}: {left} != {right}"
            )
        return abs(left - right)
    if observed != expected:
        raise ValueError(f"unsupported value changed at {path}")
    return 0.0


def validate_artifact(root: Path) -> tuple[dict[str, Any], Path, Path]:
    result_path = root / "result.json"
    protocol_path = root / "confirmation-protocol.json"
    readiness_path = root / "bound-readiness.json"
    for path in (result_path, protocol_path, readiness_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = read_json(result_path)
    protocol = read_json(protocol_path)
    readiness = read_json(readiness_path)
    if (
        result.get("schema") != CONFIRMATION_SCHEMA
        or result.get("status") != "complete"
    ):
        raise ValueError("immutable confirmation result is not complete v5 evidence")
    unsigned_result = dict(result)
    stored_result_digest = unsigned_result.pop("result_sha256", None)
    if query_cov.canonical_digest(unsigned_result) != stored_result_digest:
        raise ValueError("immutable confirmation result digest is invalid")
    if readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("unexpected readiness schema")
    unsigned_readiness = dict(readiness)
    stored_readiness_digest = unsigned_readiness.pop("result_sha256", None)
    if query_cov.canonical_digest(unsigned_readiness) != stored_readiness_digest:
        raise ValueError("readiness result digest is invalid")
    binding = protocol["readiness_binding"]
    if sha256_file(readiness_path) != binding["readiness_file_sha256"]:
        raise ValueError("bound readiness bytes changed")
    if stored_readiness_digest != binding["readiness_result_sha256"]:
        raise ValueError("bound readiness digest changed")
    if readiness["selection_manifest_sha256"] != binding[
        "selection_manifest_sha256"
    ]:
        raise ValueError("selection manifest binding changed")
    if query_cov.canonical_digest(readiness["selection_manifest"]) != readiness[
        "selection_manifest_sha256"
    ]:
        raise ValueError("selection manifest digest is invalid")
    object_ids = [str(row["object_id"]) for row in result["objects"]]
    if object_ids != list(map(str, protocol["eligible_object_ids"])):
        raise ValueError("confirmation object roster changed")
    if int(result["summary"]["object_count"]) != len(object_ids):
        raise ValueError("confirmation object count changed")
    return result, protocol_path, readiness_path


def reproduce_exact_confirmation(
    artifact_root: Path,
    confirmation_root: Path,
    frozen_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve(strict=True)
    confirmation_root = confirmation_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    reference, protocol_path, readiness_path = validate_artifact(artifact_root)
    binding = reference["development_method_binding"]
    frozen_revision = str(binding["development_source_revision"])
    if git_output(frozen_root, "rev-parse", "HEAD") != frozen_revision:
        raise ValueError("frozen v3 checkout revision changed")
    if git_output(confirmation_root, "rev-parse", "HEAD") != CONFIRMATION_REVISION:
        raise ValueError("confirmation control checkout revision changed")
    confirmation_path = (
        confirmation_root
        / "scripts"
        / "science"
        / "run_deform360_untouched_confirmation_v5.py"
    )
    confirmation = load_module(
        confirmation_path, "deform360_untouched_confirmation_v5_exact_export"
    )
    original_validate = confirmation.validate_frozen_method
    captures: list[dict[str, Any]] = []

    def validate_and_intercept(
        root: Path, protocol: dict[str, Any]
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        v3, development, base_protocol = original_validate(root, protocol)
        original_metrics = v3.base.probabilistic_metrics

        def capture_metrics(
            errors: np.ndarray,
            covariance: Any,
            probability: float,
            rng: np.random.Generator,
            sample_count: int,
        ) -> dict[str, float]:
            captures.append(
                {
                    "errors": np.asarray(errors, dtype=np.float64).copy(),
                    "diagonal": np.asarray(
                        covariance.diagonal, dtype=np.float64
                    ).copy(),
                    "factor": np.asarray(
                        covariance.factor, dtype=np.float64
                    ).copy(),
                    "multiplier": float(covariance.multiplier),
                    "marginal_z": float(covariance.marginal_z),
                }
            )
            return original_metrics(
                errors, covariance, probability, rng, sample_count
            )

        v3.base.probabilistic_metrics = capture_metrics
        return v3, development, base_protocol

    confirmation.validate_frozen_method = validate_and_intercept
    reproduced = confirmation.run(
        protocol_path, readiness_path, data_root, frozen_root
    )
    if len(captures) != len(reproduced["objects"]):
        raise ValueError(
            "expected exactly one final probabilistic scoring call per object"
        )
    maximum_difference = 0.0
    for key in (
        "summary",
        "robust_statistics",
        "action_family_summary",
        "confirmation_decision",
    ):
        maximum_difference = max(
            maximum_difference,
            compare_values(reproduced[key], reference[key], path=key),
        )
    maximum_difference = max(
        maximum_difference,
        compare_values(reproduced["objects"], reference["objects"], path="objects"),
    )
    return {
        "reference": reference,
        "reproduced": reproduced,
        "captures": captures,
        "maximum_absolute_numeric_difference": maximum_difference,
        "confirmation_revision": CONFIRMATION_REVISION,
        "frozen_revision": frozen_revision,
    }


def self_test() -> None:
    compare_values(
        {"x": [1.0, True, "a"]},
        {"x": [1.0 + 1e-13, True, "a"]},
    )
    print("exact Deform360 confirmation capture v7 self-test passed")
