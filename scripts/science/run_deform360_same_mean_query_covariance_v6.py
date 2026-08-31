#!/usr/bin/env python3
"""Audit and cross-fit same-mean Deform360 query covariance arms.

This is a retrospective real-output development tool. It never changes a
predictive mean. It first locates a retained sufficient-statistics bundle by
reading only file metadata and NPZ headers. When exactly one compatible
bundle exists, it compares source-cross-fitted hybrid, full, diagonal, and
marginal-preserving dependence-permuted covariance arms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import numpy as np

ALIASES = {
    "truth": ("truth", "target", "targets", "observed", "observation", "y_true"),
    "mean": (
        "mean",
        "prediction",
        "predictions",
        "ensemble_mean",
        "candidate_prediction",
        "y_pred",
    ),
    "covariance": (
        "covariance",
        "covariances",
        "full_covariance",
        "predictive_covariance",
        "query_covariance",
    ),
    "variance": (
        "variance",
        "variances",
        "diagonal_variance",
        "predictive_variance",
    ),
    "samples": (
        "samples",
        "ensemble",
        "ensemble_predictions",
        "prediction_samples",
    ),
    "groups": ("object_ids", "object_id", "group_ids", "case_ids", "names"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def role_key(names: Sequence[str], role: str) -> str | None:
    lookup = {normalise(name): name for name in names}
    for alias in ALIASES[role]:
        if alias in lookup:
            return lookup[alias]
    return None


def candidate_roles(names: Sequence[str]) -> dict[str, str]:
    roles = {role: role_key(names, role) for role in ALIASES}
    return {role: value for role, value in roles.items() if value is not None}


def is_candidate(roles: dict[str, str]) -> bool:
    return "truth" in roles and "mean" in roles and any(
        role in roles for role in ("covariance", "variance", "samples")
    )


def eligible(path: Path) -> bool:
    text = str(path).lower()
    return any(
        token in text
        for token in (
            "deform360",
            "33335779766",
            "33330455808",
            "33331368970",
            "9738998271",
            "action_kernel",
            "action_forecasting",
        )
    )


def audit_roots(
    roots: Sequence[Path], maximum_files_per_root: int
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for root in roots:
        resolved = root.resolve()
        if not resolved.exists():
            continue
        seen = 0
        for path in resolved.rglob("*"):
            if seen >= maximum_files_per_root:
                break
            if not path.is_file() or not eligible(path):
                continue
            suffix = path.suffix.lower()
            if suffix not in {".npz", ".npy", ".json", ".csv"}:
                continue
            seen += 1
            record: dict[str, Any] = {
                "path": str(path.resolve()),
                "root": str(resolved),
                "size_bytes": path.stat().st_size,
                "suffix": suffix,
            }
            try:
                if suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        arrays = {
                            key: {
                                "shape": list(archive[key].shape),
                                "dtype": str(archive[key].dtype),
                            }
                            for key in archive.files
                        }
                    roles = candidate_roles(list(arrays))
                    record.update(
                        {
                            "arrays": arrays,
                            "roles": roles,
                            "candidate": is_candidate(roles),
                        }
                    )
                elif suffix == ".npy":
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    record.update(
                        {
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                            "candidate": False,
                        }
                    )
                elif suffix == ".csv":
                    with path.open(
                        "r", encoding="utf-8", errors="replace", newline=""
                    ) as stream:
                        columns = next(csv.reader(stream), [])
                    roles = candidate_roles(columns)
                    record.update(
                        {
                            "columns": columns,
                            "roles": roles,
                            "candidate": is_candidate(roles),
                        }
                    )
                else:
                    if path.stat().st_size > 16 * 1024 * 1024:
                        record.update(
                            {
                                "skipped": "json-larger-than-16MiB",
                                "candidate": False,
                            }
                        )
                    else:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        keys = list(payload) if isinstance(payload, dict) else []
                        roles = candidate_roles(keys)
                        record.update(
                            {
                                "top_level_keys": keys,
                                "roles": roles,
                                "candidate": is_candidate(roles),
                            }
                        )
                if path.stat().st_size <= 128 * 1024 * 1024:
                    record["sha256"] = sha256_file(path)
                records.append(record)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(
                    {
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    candidates = [record for record in records if record.get("candidate")]
    decision = (
        "unique-retained-sufficient-statistics-ready"
        if len(candidates) == 1
        else "ambiguous-retained-sufficient-statistics"
        if candidates
        else "no-retained-sufficient-statistics"
    )
    return {
        "schema": "deform360-same-mean-input-audit-v6",
        "information_boundary": {
            "mounted_dataset_numeric_payload_opened": False,
            "retained_target_values_read_during_audit": False,
            "npz_numeric_values_read": False,
            "npz_headers_read": True,
            "csv_rows_read": False,
        },
        "roots": [str(root.resolve()) for root in roots],
        "record_count": len(records),
        "candidate_count": len(candidates),
        "candidate_paths": [record["path"] for record in candidates],
        "decision": decision,
        "records": records,
        "errors": errors,
        "paper_claim_authorized": False,
    }


def nearest_psd(covariance: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("covariance must be a finite square matrix")
    symmetric = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    result = (vectors * values[None, :]) @ vectors.T
    return 0.5 * (result + result.T)


def correlation_shrinkage(covariance: np.ndarray, weight: float) -> np.ndarray:
    covariance = nearest_psd(covariance)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("correlation weight must lie in [0,1]")
    diagonal = np.diag(covariance).copy()
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = covariance * inverse[:, None] * inverse[None, :]
    correlation = weight * correlation + (1.0 - weight) * np.eye(
        covariance.shape[0]
    )
    standard = np.sqrt(diagonal)
    result = correlation * standard[:, None] * standard[None, :]
    np.fill_diagonal(result, diagonal)
    return nearest_psd(result)


def permuted_correlation(
    covariance: np.ndarray, order: Sequence[int]
) -> np.ndarray:
    covariance = nearest_psd(covariance)
    order_array = np.asarray(order, dtype=np.int64)
    if order_array.shape != (covariance.shape[0],) or set(
        order_array.tolist()
    ) != set(range(covariance.shape[0])):
        raise ValueError("invalid dependence permutation")
    diagonal = np.diag(covariance).copy()
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = covariance * inverse[:, None] * inverse[None, :]
    correlation = correlation[np.ix_(order_array, order_array)]
    standard = np.sqrt(diagonal)
    result = correlation * standard[:, None] * standard[None, :]
    np.fill_diagonal(result, diagonal)
    return nearest_psd(result)


def basis_indices(
    shape: tuple[int, ...], maximum: int
) -> list[tuple[int, ...]]:
    limits = [min(axis, 4) for axis in shape]
    values = list(itertools.product(*(range(limit) for limit in limits)))
    values.sort(key=lambda item: (sum(item), max(item), item))
    return values[:maximum]


def cosine_query_matrix(shape: tuple[int, ...], maximum: int) -> np.ndarray:
    coordinates = [np.arange(axis, dtype=np.float64) for axis in shape]
    vectors: list[np.ndarray] = []
    for frequencies in basis_indices(shape, maximum):
        component = np.ones(shape, dtype=np.float64)
        for axis, (coordinate, frequency) in enumerate(
            zip(coordinates, frequencies, strict=True)
        ):
            factor = np.cos(
                np.pi * frequency * (coordinate + 0.5) / shape[axis]
            )
            reshape = [1] * len(shape)
            reshape[axis] = shape[axis]
            component *= factor.reshape(reshape)
        vector = component.reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vectors.append(vector / norm)
    if not vectors:
        raise RuntimeError("empty query basis")
    return np.stack(vectors)


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sample_covariance(samples: np.ndarray) -> np.ndarray:
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    denominator = max(samples.shape[0] - 1, 1)
    return nearest_psd(centered.T @ centered / denominator)


def load_bundle(
    path: Path, maximum_queries: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        roles = candidate_roles(archive.files)
        if not is_candidate(roles):
            raise ValueError("NPZ is not a complete sufficient-statistics bundle")
        truth = np.asarray(archive[roles["truth"]], dtype=np.float64)
        mean = np.asarray(archive[roles["mean"]], dtype=np.float64)
        if truth.shape != mean.shape or truth.ndim < 2 or truth.shape[0] < 5:
            raise ValueError("truth and mean arrays do not align")
        case_count = truth.shape[0]
        field_shape = truth.shape[1:]
        query = cosine_query_matrix(field_shape, maximum_queries)
        flat_truth = truth.reshape(case_count, -1)
        flat_mean = mean.reshape(case_count, -1)
        query_truth = flat_truth @ query.T
        query_mean = flat_mean @ query.T
        residuals = query_truth - query_mean
        if "covariance" in roles:
            covariance = np.asarray(
                archive[roles["covariance"]], dtype=np.float64
            )
            if covariance.shape == (
                case_count,
                flat_truth.shape[1],
                flat_truth.shape[1],
            ):
                query_covariance = np.einsum(
                    "qd,ndk,pk->nqp", query, covariance, query
                )
            elif covariance.shape == (
                case_count,
                query.shape[0],
                query.shape[0],
            ):
                query_covariance = covariance
            else:
                raise ValueError("unsupported covariance shape")
        elif "variance" in roles:
            variance = np.asarray(archive[roles["variance"]], dtype=np.float64)
            if variance.shape != truth.shape:
                raise ValueError("variance does not match truth")
            query_covariance = np.einsum(
                "qd,nd,pd->nqp",
                query,
                variance.reshape(case_count, -1),
                query,
            )
        else:
            samples = np.asarray(archive[roles["samples"]], dtype=np.float64)
            if (
                samples.ndim != truth.ndim + 1
                or samples.shape[0] != case_count
                or samples.shape[2:] != field_shape
            ):
                raise ValueError("sample array does not align")
            projected = samples.reshape(case_count, samples.shape[1], -1) @ query.T
            query_covariance = np.stack(
                [sample_covariance(value) for value in projected]
            )
        if "groups" in roles:
            groups = np.asarray(archive[roles["groups"]]).astype(str).reshape(-1)
            if groups.shape != (case_count,):
                raise ValueError("group identifiers do not align")
        else:
            groups = np.asarray(
                [f"case-{index:06d}" for index in range(case_count)]
            )
    return residuals, query_mean, query_covariance, groups, {
        "roles": roles,
        "field_shape": list(field_shape),
        "query_dimension": int(query.shape[0]),
        "query_matrix_sha256": array_digest(query),
        "mean_sha256": array_digest(query_mean),
    }


def gaussian_terms(
    residual: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, float]:
    covariance = nearest_psd(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("nonpositive determinant")
    distance = float(residual @ np.linalg.solve(covariance, residual))
    nll = 0.5 * (
        residual.size * math.log(2.0 * math.pi) + logdet + distance
    )
    return nll, distance, float(logdet)


def metrics(
    residuals: np.ndarray,
    covariances: np.ndarray,
    probability: float = 0.9,
) -> dict[str, Any]:
    dimension = residuals.shape[1]
    z = NormalDist().inv_cdf(0.5 + probability / 2.0)
    chi = dimension * (
        1.0
        - 2.0 / (9.0 * dimension)
        + NormalDist().inv_cdf(probability)
        * math.sqrt(2.0 / (9.0 * dimension))
    ) ** 3
    nlls: list[float] = []
    distances: list[float] = []
    logdets: list[float] = []
    marginal_hits = 0
    ellipsoid_hits = 0
    widths: list[float] = []
    for residual, covariance in zip(residuals, covariances, strict=True):
        nll, distance, logdet = gaussian_terms(residual, covariance)
        nlls.append(nll)
        distances.append(distance)
        logdets.append(logdet)
        ellipsoid_hits += int(distance <= chi)
        sd = np.sqrt(np.diag(nearest_psd(covariance)))
        marginal_hits += int(np.count_nonzero(np.abs(residual) <= z * sd))
        widths.extend((2.0 * z * sd).tolist())
    return {
        "n_cases": int(residuals.shape[0]),
        "query_dimension": int(dimension),
        "nll_per_dimension": float(np.mean(nlls) / dimension),
        "normalized_anees": float(np.mean(distances) / dimension),
        "ellipsoid_coverage": float(ellipsoid_hits / residuals.shape[0]),
        "marginal_coverage": float(marginal_hits / residuals.size),
        "mean_marginal_width": float(np.mean(widths)),
        "mean_log_determinant": float(np.mean(logdets)),
    }


def transform(
    covariances: np.ndarray, weight: float, scale: float
) -> np.ndarray:
    return np.stack(
        [
            scale * correlation_shrinkage(covariance, weight)
            for covariance in covariances
        ]
    )


def fit(
    residuals: np.ndarray,
    covariances: np.ndarray,
    weights: Iterable[float],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    dimension = residuals.shape[1]
    for weight in weights:
        base = transform(covariances, float(weight), 1.0)
        distances = [
            gaussian_terms(error, covariance)[1]
            for error, covariance in zip(residuals, base, strict=True)
        ]
        scale = float(np.clip(np.mean(distances) / dimension, 1e-4, 1e4))
        score = metrics(
            residuals, transform(covariances, float(weight), scale)
        )["nll_per_dimension"]
        candidate = {
            "correlation_weight": float(weight),
            "scale": scale,
            "source_nll_per_dimension": float(score),
        }
        if best is None or (
            candidate["source_nll_per_dimension"],
            -candidate["correlation_weight"],
        ) < (
            best["source_nll_per_dimension"],
            -best["correlation_weight"],
        ):
            best = candidate
    if best is None:
        raise ValueError("empty weight grid")
    return best


def fold_for(group: str, fold_count: int) -> int:
    digest = hashlib.sha256(("deform360-same-mean-v6\0" + group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def study(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: np.ndarray,
    fold_count: int,
) -> dict[str, Any]:
    unique = sorted(set(groups.tolist()))
    if len(unique) < fold_count:
        raise ValueError("not enough independent groups")
    group_fold = {group: fold_for(group, fold_count) for group in unique}
    if set(group_fold.values()) != set(range(fold_count)):
        raise ValueError("deterministic folds are not all populated")
    rng = np.random.default_rng(20260831)
    order = rng.permutation(residuals.shape[1])
    if np.array_equal(order, np.arange(residuals.shape[1])):
        order = np.roll(order, 1)
    names = ("hybrid", "full", "diagonal", "permuted", "uncalibrated")
    arm_residuals: dict[str, list[np.ndarray]] = {name: [] for name in names}
    arm_covariances: dict[str, list[np.ndarray]] = {name: [] for name in names}
    folds: list[dict[str, Any]] = []
    for index in range(fold_count):
        target_mask = np.asarray([group_fold[group] == index for group in groups])
        source_mask = ~target_mask
        source_residuals = residuals[source_mask]
        target_residuals = residuals[target_mask]
        source_cov = covariances[source_mask]
        target_cov = covariances[target_mask]
        source_perm = np.stack(
            [permuted_correlation(value, order) for value in source_cov]
        )
        target_perm = np.stack(
            [permuted_correlation(value, order) for value in target_cov]
        )
        fits = {
            "hybrid": fit(
                source_residuals, source_cov, np.linspace(0.0, 1.0, 11)
            ),
            "full": fit(source_residuals, source_cov, (1.0,)),
            "diagonal": fit(source_residuals, source_cov, (0.0,)),
            "permuted": fit(source_residuals, source_perm, (1.0,)),
        }
        transformed = {
            "hybrid": transform(
                target_cov,
                fits["hybrid"]["correlation_weight"],
                fits["hybrid"]["scale"],
            ),
            "full": transform(target_cov, 1.0, fits["full"]["scale"]),
            "diagonal": transform(
                target_cov, 0.0, fits["diagonal"]["scale"]
            ),
            "permuted": transform(
                target_perm, 1.0, fits["permuted"]["scale"]
            ),
            "uncalibrated": target_cov,
        }
        for name, covariance in transformed.items():
            arm_residuals[name].append(target_residuals)
            arm_covariances[name].append(covariance)
        folds.append(
            {
                "fold": index,
                "source_groups": len(set(groups[source_mask].tolist())),
                "target_groups": len(set(groups[target_mask].tolist())),
                "target_cases": int(np.count_nonzero(target_mask)),
                "fits": fits,
            }
        )
    arm_metrics = {
        name: metrics(
            np.concatenate(arm_residuals[name]),
            np.concatenate(arm_covariances[name]),
        )
        for name in names
    }
    primary = arm_metrics["hybrid"]
    diagonal = arm_metrics["diagonal"]
    permuted = arm_metrics["permuted"]
    gates = {
        "nll_better_than_diagonal_by_0p02": primary["nll_per_dimension"]
        <= diagonal["nll_per_dimension"] - 0.02,
        "nll_better_than_permuted_by_0p02": primary["nll_per_dimension"]
        <= permuted["nll_per_dimension"] - 0.02,
        "normalized_anees_between_0p8_and_1p2": 0.8
        <= primary["normalized_anees"]
        <= 1.2,
        "marginal_coverage_between_0p87_and_0p93": 0.87
        <= primary["marginal_coverage"]
        <= 0.93,
    }
    return {
        "independent_group_count": len(unique),
        "fold_count": fold_count,
        "case_count": int(residuals.shape[0]),
        "folds": folds,
        "metrics": arm_metrics,
        "contrasts": {
            "hybrid_minus_diagonal_nll_per_dimension": primary[
                "nll_per_dimension"
            ]
            - diagonal["nll_per_dimension"],
            "hybrid_minus_permuted_nll_per_dimension": primary[
                "nll_per_dimension"
            ]
            - permuted["nll_per_dimension"],
        },
        "gates": gates,
        "superior_target_passed": all(gates.values()),
    }


def self_test() -> None:
    rng = np.random.default_rng(73029)
    cases, dimension = 500, 16
    direction = np.ones(dimension) / math.sqrt(dimension)
    covariance = 0.08 * np.eye(dimension) + 1.2 * np.outer(
        direction, direction
    )
    truth = rng.multivariate_normal(
        np.zeros(dimension), covariance, size=cases
    ).reshape(cases, 4, 4)
    mean = np.zeros_like(truth)
    covariances = np.broadcast_to(
        covariance, (cases, dimension, dimension)
    ).copy()
    groups = np.asarray([f"object-{index % 50:02d}" for index in range(cases)])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "deform360-sufficient-statistics.npz"
        np.savez(
            path,
            truth=truth,
            mean=mean,
            covariance=covariances,
            object_ids=groups,
        )
        residuals, query_mean, query_cov, loaded_groups, metadata = load_bundle(
            path, 8
        )
        result = study(residuals, query_cov, loaded_groups, 5)
        assert metadata["mean_sha256"] == array_digest(query_mean)
        assert result["metrics"]["hybrid"]["nll_per_dimension"] < result[
            "metrics"
        ]["diagonal"]["nll_per_dimension"]
        assert (
            0.8
            <= result["metrics"]["hybrid"]["normalized_anees"]
            <= 1.2
        )
        audit = audit_roots([Path(directory)], 100)
        assert audit["candidate_count"] == 1
    covariance3 = np.array(
        [[4.0, 1.2, -0.6], [1.2, 2.0, 0.4], [-0.6, 0.4, 3.0]]
    )
    assert np.allclose(
        np.diag(correlation_shrinkage(covariance3, 0.0)),
        np.diag(covariance3),
    )
    assert np.allclose(
        np.diag(permuted_correlation(covariance3, [2, 0, 1])),
        np.diag(covariance3),
    )
    print("same-mean query covariance self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--maximum-files-per-root", type=int, default=5000)
    parser.add_argument("--maximum-queries", type=int, default=12)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.root or args.output_root is None:
        parser.error(
            "--root and --output-root are required unless --self-test is used"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_roots(args.root, args.maximum_files_per_root)
    (args.output_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if audit["candidate_count"] != 1:
        summary = {
            "schema": "deform360-same-mean-query-covariance-v6",
            "status": audit["decision"],
            "next_action": (
                "add source-frozen sufficient-statistics export to the frozen "
                "v5 predictor"
            ),
            "paper_claim_authorized": False,
        }
        (args.output_root / "result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, sort_keys=True))
        return
    bundle = Path(audit["candidate_paths"][0])
    residuals, query_mean, query_covariance, groups, metadata = load_bundle(
        bundle, args.maximum_queries
    )
    result = {
        "schema": "deform360-same-mean-query-covariance-v6",
        "status": "completed",
        "bundle": {
            "path": str(bundle.resolve()),
            "sha256": sha256_file(bundle),
        },
        "same_mean_contract": {
            "mean_sha256": metadata["mean_sha256"],
            "mean_changed_between_arms": False,
            "only_covariance_changes": True,
        },
        "metadata": metadata,
        "study": study(
            residuals, query_covariance, groups, args.fold_count
        ),
        "classification": "retrospective cross-fitted real-output development",
        "paper_claim_authorized": False,
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "superior_target_passed": result["study"][
                    "superior_target_passed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
