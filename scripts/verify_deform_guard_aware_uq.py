#!/usr/bin/env python3
"""Independent matrix-algebra and statistical verification of fixed-mean UQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ("isotropic", "guard_scaled", "shadow", "fixed_mean_bridge", "rotated_bridge")
FAMILIES = (*RAW, "source_full")
VARIANTS = ("moment", "conformal")


def independent_prediction(incumbent, mean, response, coefficients, posterior, gains):
    shadow_mean = np.empty(mean.shape, dtype=float)
    tangent = np.empty((*mean.shape, 3), dtype=float)
    for case in range(len(mean)):
        j = response[case]
        p = posterior[case, :24, :24]
        shadow_mean[case] = incumbent[case].astype(float) + j @ coefficients[case, :24]
        tangent[case] = (j @ p) @ np.swapaxes(j, -2, -1)
    tangent = 0.5 * (tangent + np.swapaxes(tangent, -2, -1))
    iso = np.broadcast_to(0.003**2 * np.eye(3), tangent.shape).copy()
    unscaled = tangent + iso
    offset = shadow_mean - mean.astype(float)
    bridge = unscaled + offset[..., None] @ offset[..., None, :]
    perm = [1, 2, 0]
    return {
        "mean": mean,
        "shadow_mean": shadow_mean,
        "isotropic": iso,
        "guard_scaled": tangent * gains[:, None, None, None, None] ** 2 + iso,
        "shadow": unscaled,
        "fixed_mean_bridge": bridge,
        "rotated_bridge": bridge[..., perm, :][..., :, perm],
    }


def independent_calibration(error, raw):
    matrices = []
    for start in (0, 40, 80):
        moments = []
        for case in range(13):
            rows = error[case, start : start + 40].reshape(-1, 3)
            moments.append(rows.T @ rows / len(rows))
        matrices.append(np.mean(moments, axis=0) + 1e-12 * np.eye(3))
    full = np.empty((*error.shape, 3))
    for i, matrix in enumerate(matrices):
        full[:, 40 * i : 40 * (i + 1)] = matrix
    scales = {}
    for arm, covariance in {**raw, "source_full": full}.items():
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, error[..., None])[..., 0]
        nees = np.square(whitened).sum(axis=-1)
        values: dict[str, list[float]] = {"moment": [], "conformal": []}
        for start in (0, 40, 80):
            per_case = nees[:, start : start + 40].reshape(13, 160)
            values["moment"].append(
                max(1e-6, float(np.mean(np.mean(per_case, axis=1)) / 3))
            )
            # 'higher' at q=.9 selects ceil(.9 * (160-1)) = index 144.
            upper_scores = np.sort(per_case, axis=1)[:, 144]
            values["conformal"].append(
                max(1e-6, float(max(upper_scores) / 6.251388631170325))
            )
        scales[arm] = values
    return {"source_full_matrices_m2": np.asarray(matrices).tolist(), "scales": scales}


def tree_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            tree_close(a, b)
    elif isinstance(expected, (bool, str)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts/remote"))
    import run_deform_guard_aware_uq as runner
    from verify_deform_weak_constraint_belief import independent_uq

    from bayesian_phystwin_experiments.deform_guard_aware_uq import (
        PROTOCOL,
        load_protocol,
    )
    from bayesian_phystwin_experiments.deform_multiobject_restart import (
        config_for_object,
    )

    receipt = runner.parent_runner.multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    protocol, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    arrays = runner.validate_barrier(args, protocol, parent, receipt)
    calibration = runner.validate_calibration(args, receipt)
    result = json.loads((args.output / "result.json").read_text())
    assert result["source_revision"] == receipt["revision"]
    assert result["source_receipt_sha256"] == args.source_receipt_sha256
    assert result["protocol_sha256"] == file_digest(ROOT / PROTOCOL)
    assert result["prediction_barrier_sha256"] == file_digest(
        args.output / "prediction_barrier.json"
    )
    assert result["calibration_sha256"] == args.calibration_sha256
    assert result["point_mean_byte_identical"] is True
    assert result["original_results_modified"] is False
    assert result["protected_data_access"] is False
    assert result["population_confirmation_or_sota_claim"] is False
    assert result["ordinary_success"] == 30
    assert result["retained_technical_failure"] == result["unsealable"] == 0

    events_checked, records_checked, recomputed = 0, 0, {}
    for item in parent["objects"]:
        name = item["object"]
        model, fit, old, digest = runner.parent_arrays(item, protocol)
        assert (
            array_digest(arrays[name]["mean"])
            == digest
            == array_digest(old["previous_paired_8"])
        )
        expected = independent_prediction(
            model["incumbent"][:, 50:],
            old["previous_paired_8"],
            model["response"][:, 25:, :, :, :24],
            fit["strong_8__coefficients"],
            fit["strong_8__posterior"],
            fit["strong_8__gain"],
        )
        for key, value in expected.items():
            np.testing.assert_allclose(arrays[name][key], value, rtol=1e-10, atol=1e-12)
            if key in RAW:
                np.linalg.cholesky(value)
        truth = runner.parent_runner.truth_for(item, parent)
        rod = config_for_object(parent, item)
        keep = [i for i, case in enumerate(item["names"]) if case != rod.design_case]
        error = old["previous_paired_8"][keep][:, :, rod.hidden_nodes].astype(
            float
        ) - truth[keep][:, :, rod.hidden_nodes].astype(float)
        raw = {key: arrays[name][key][keep][:, :, rod.hidden_nodes] for key in RAW}
        if name == "DLO2":
            tree_close(
                {k: calibration[k] for k in ("source_full_matrices_m2", "scales")},
                independent_calibration(error, raw),
            )
            assert array_digest(truth) == calibration["source_truth_sha256"]
        uq = {}
        for family in FAMILIES:
            if family == "source_full":
                cov = np.empty_like(raw["isotropic"])
                for i, matrix in enumerate(calibration["source_full_matrices_m2"]):
                    cov[:, i * 40 : (i + 1) * 40] = matrix
            else:
                cov = raw[family]
            for variant in VARIANTS:
                scaled = cov.copy()
                for i, scale in enumerate(calibration["scales"][family][variant]):
                    scaled[:, i * 40 : (i + 1) * 40] *= scale
                events = independent_uq(error, scaled)
                per_case = {k: v.mean(axis=(1, 2)).tolist() for k, v in events.items()}
                summary = {k: float(np.mean(v)) for k, v in per_case.items()}
                horizons = {
                    label: {
                        k: float(v[:, i * 40 : (i + 1) * 40].mean())
                        for k, v in events.items()
                    }
                    for i, label in enumerate(("early", "middle", "late"))
                }
                key = family + "__" + variant
                uq[key] = {
                    "summary": summary,
                    "per_case": per_case,
                    "horizons": horizons,
                }
                tree_close(result["objects"][name]["uq"][key], uq[key])
                events_checked += error.shape[0] * 120 * 4
                records_checked += error.shape[0]
        per_case = {
            "coordinate_l1_mm": np.mean(np.abs(error).reshape(len(error), -1), axis=1)
            * 1000,
            "point_rmse_mm": np.sqrt(np.mean(np.sum(error**2, axis=-1), axis=(1, 2)))
            * 1000,
        }
        tree_close(
            result["objects"][name]["point"],
            {k: float(v.mean()) for k, v in per_case.items()},
        )
        tree_close(
            result["objects"][name]["point_per_case"],
            {k: v.tolist() for k, v in per_case.items()},
        )
        assert result["objects"][name]["point_mean_sha256"] == digest
        recomputed[name] = uq

    aggregate = {
        key: {
            metric: float(
                np.mean(
                    [
                        recomputed[name][key]["summary"][metric]
                        for name in ("DLO1", "DLO3")
                    ]
                )
            )
            for metric in recomputed["DLO1"][key]["summary"]
        }
        for key in recomputed["DLO1"]
    }
    tree_close(result["equal_object_transfer"], aggregate)
    decisions = {}
    for name in ("DLO1", "DLO3"):
        candidate = recomputed[name]["fixed_mean_bridge__moment"]
        checks = {
            "point_mean_byte_identical": True,
            "all_30_predictions_accounted": True,
            "coverage_between_80_and_98_percent": 0.8
            <= candidate["summary"]["coverage_90"]
            <= 0.98,
        }
        intervals = {}
        indices = np.random.default_rng(260835).integers(0, 8, (10000, 8))
        for comparator in ("isotropic", "source_full", "shadow"):
            control = recomputed[name][comparator + "__moment"]
            delta = np.array(candidate["per_case"]["nll"]) - control["per_case"]["nll"]
            ci = np.quantile(delta[indices].sum(axis=1) / 8, [0.025, 0.975]).tolist()
            intervals[comparator] = ci
            checks["nll_ci95_upper_negative_vs_" + comparator] = ci[1] < 0
            if comparator != "shadow":
                checks["at_least_five_nll_wins_vs_" + comparator] = (
                    int(sum(delta < 0)) >= 5
                )
                checks["volume_nonincreasing_vs_" + comparator] = (
                    candidate["summary"]["ellipsoid_volume_mm3"]
                    <= control["summary"]["ellipsoid_volume_mm3"]
                )
        decisions[name] = {"checks": checks, "nll_delta_ci95": intervals}
    tree_close(
        result["decision"],
        {
            "primary_arm": "fixed_mean_bridge__moment",
            "objects": decisions,
            "development_advancement_gate_passed": all(
                all(v["checks"].values()) for v in decisions.values()
            ),
            "point_mean_changed": False,
            "secondary_arms_cannot_rescue_primary": True,
            "automatic_target_authorization": False,
            "incumbent_promoted": False,
            "population_confirmation_or_sota_claim": False,
        },
    )
    return {
        "schema": "deform-guard-aware-uq-independent-verification-v1",
        "verified": True,
        "source_revision": receipt["revision"],
        "source_receipt_sha256": args.source_receipt_sha256,
        "prediction_barrier_sha256": file_digest(
            args.output / "prediction_barrier.json"
        ),
        "calibration_sha256": args.calibration_sha256,
        "result_sha256": file_digest(args.output / "result.json"),
        "verifier_sha256": file_digest(Path(__file__)),
        "point_means_byte_identical": 30,
        "covariance_carriers_recomputed": 150,
        "forecast_records_verified": records_checked,
        "marginal_uq_events_verified": events_checked,
        "new_native_rollouts": 0,
        "independent_matrix_algebra": True,
        "independent_source_calibration": True,
        "independent_scores_bootstrap_and_gate": True,
        "protected_data_access": False,
        "new_official_evaluation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args)
    write_json_once(args.report, value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
