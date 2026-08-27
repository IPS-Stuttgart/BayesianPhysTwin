#!/usr/bin/env python3
"""Independent batch, native replay, metric, and calibration verification."""

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
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]


def batch_posterior(
    h: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if h.ndim != 3 or h.shape[:2] != residual.shape or h.shape[1] != 3:
        raise ValueError("batch inference needs aligned 3D prefix rows")
    design = h.reshape(-1, h.shape[-1]) / 0.001
    target = residual.reshape(-1) / 0.001
    precision = np.eye(h.shape[-1]) + design.T @ design
    covariance = np.linalg.solve(precision, np.eye(len(precision)))
    return np.linalg.solve(precision, design.T @ target), (
        covariance + covariance.T
    ) / 2


def independent_design(
    response: np.ndarray, arm: str, nodes: tuple[int, ...]
) -> np.ndarray:
    columns = list(range(24))
    if arm == "weak_16":
        columns += list(range(24, 60))
    elif arm == "weak_8":
        columns += list(range(48, 60))
    rows = []
    for frame in (25, 33, 41, 49):
        for node in nodes:
            rows.append(
                np.column_stack(
                    (response[frame - 25, node][:, columns], 0.005 * np.eye(3))
                )
            )
    return np.stack(rows)


def independent_impulses(
    coefficients: np.ndarray, arm: str, nodes: int, clamps: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots = sorted((2, 4, 6, 8, *clamps))
    values = np.zeros((len(coefficients), 60))
    columns = list(range(24))
    if arm == "weak_16":
        columns += list(range(24, 60))
    elif arm == "weak_8":
        columns += list(range(48, 60))
    values[:, columns] = coefficients[:, :-3]
    pose = np.zeros((len(values), 4, nodes, 3))
    velocity = np.zeros_like(pose)
    for case in range(len(values)):
        for axis in range(3):
            for step in range(4):
                for is_velocity in (False, True):
                    if step == 0:
                        start = 12 if is_velocity else 0
                        factor = 0.1 if is_velocity else 0.01
                    else:
                        start = (step + 1) * 12
                        factor = 0.05 if is_velocity else 0.002
                    knot_values = [
                        values[case, start + (2, 4, 6, 8).index(n) * 3 + axis] * factor
                        if n in (2, 4, 6, 8)
                        else 0.0
                        for n in knots
                    ]
                    (velocity if is_velocity else pose)[case, step, :, axis] = (
                        np.interp(np.arange(nodes), knots, knot_values)
                    )
    pose_max = np.linalg.norm(pose, axis=-1).sum(axis=1).max(axis=1)
    velocity_max = np.linalg.norm(velocity, axis=-1).sum(axis=1).max(axis=1)
    gain = 1 / np.maximum(1, np.maximum(pose_max / 0.03, velocity_max / 0.3))
    return pose * gain[:, None, None, None], velocity * gain[:, None, None, None], gain


def independent_uq(error: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, error[..., None])[..., 0]
    nees = np.square(whitened).sum(axis=-1)
    logdet = 2 * np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=-1)
    radius = 6.251388631170325
    coverage = nees <= radius
    boundary = np.abs(nees - radius) <= 64 * np.finfo(np.float64).eps * radius
    if np.any(boundary):
        # A fitted conformal quantile can be exactly on a binary boundary. Keep
        # independent Cholesky continuous scores, but use the registered solver
        # operation order for membership within float64 roundoff of that boundary.
        solved = np.linalg.solve(covariance, error[..., None])[..., 0]
        canonical_nees = np.einsum("...i,...i->...", error, solved)
        np.testing.assert_allclose(canonical_nees, nees, atol=1e-10, rtol=1e-9)
        coverage = np.where(boundary, canonical_nees <= radius, coverage)
    return {
        "nll": 0.5 * (nees + logdet + 3 * np.log(2 * np.pi)),
        "nees": nees,
        "coverage_90": coverage.astype(float),
        "ellipsoid_volume_mm3": (4 / 3)
        * np.pi
        * np.exp(logdet / 2)
        * radius**1.5
        * 1e9,
        "geometric_full_width_mm": 2000 * np.sqrt(radius) * np.exp(logdet / 6),
    }


def replay_arm(
    rod: Any,
    initial: np.ndarray,
    actions: np.ndarray,
    pose: np.ndarray,
    velocity: np.ndarray,
    torch: Any,
) -> np.ndarray:
    _, state = rod.rollout(
        rod.initialize(initial.copy(order="C")), actions[:, :26].copy(order="C")
    )
    for step, frame in enumerate((25, 33, 41, 49)):
        if step:
            _, state = rod.rollout(
                state, actions[:, frame - 7 : frame + 1].copy(order="C")
            )
        state = update_rod_state(
            state,
            torch.tensor(pose[:, step], dtype=torch.float32),
            torch.tensor(velocity[:, step], dtype=torch.float32),
            gain=1,
            clamped_nodes=rod.config.clamped_nodes,
        )
    future, _ = rod.rollout(state, actions[:, 50:].copy(order="C"))
    return future


def verify(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts/remote"))
    import run_deform_dlo_source as source
    import run_deform_weak_constraint_belief as runner
    import torch
    import verify_deform_multiobject_state_restart as metric_verifier

    from bayesian_phystwin_experiments.deform_multiobject_restart import (
        config_for_object,
    )
    from bayesian_phystwin_experiments.deform_weak_constraint_belief import (
        CALIBRATION_FAMILIES,
        NATIVE_ARMS,
        BeliefConfig,
        load_protocol,
        primary_decision,
    )

    receipt = runner.multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    protocol, parent = load_protocol(ROOT / runner.PROTOCOL, ROOT)
    runner.validate_barrier(
        args.output, protocol, parent, receipt, file_digest(args.source_receipt)
    )
    calibration = runner.validate_calibration(
        args.output / "calibration.json", args.output, args.calibration_sha256, receipt
    )
    result = json.loads((args.output / "result.json").read_text())
    assert result["source_revision"] == receipt["revision"]
    assert result["source_receipt_sha256"] == file_digest(args.source_receipt)
    assert result["protocol_sha256"] == file_digest(ROOT / runner.PROTOCOL)
    assert result["prediction_barrier_sha256"] == file_digest(
        args.output / "prediction_barrier.json"
    )
    assert result["calibration_sha256"] == file_digest(args.output / "calibration.json")
    assert result["protected_data_access"] is False
    assert result["original_results_modified"] is False
    assert result["population_confirmation_or_sota_claim"] is False
    assert result["ordinary_success"] == 30 and result["analysis_case_count"] == 29
    assert result["retained_technical_failure"] == 0 and result["unsealable"] == 0
    source._assert_upstream(Path(parent["upstream_root"]), parent["upstream_commit"])
    modules = source._load_upstream(Path(parent["upstream_root"]))
    torch.set_num_threads(1)
    count, native_count, marginal_events = 0, 0, 0
    for item in parent["objects"]:
        config = config_for_object(parent, item)
        directory = args.output / item["object"]
        seal = json.loads((directory / "prediction_seal.json").read_text())
        arrays = {
            name: runner.previous.verified_arrays(directory / name, spec)
            for name, spec in seal["files"].items()
        }
        model, fits, covariances = (
            arrays[k] for k in ("model.npz", "fits.npz", "covariances.npz")
        )
        predictions = {
            k: v for k, v in arrays["predictions.npz"].items() if k != "names"
        }
        raw_map = source._load_named_trajectories(
            runner.multi.verify_input_files(item),
            item["names"],
            frame_count=500,
            node_count=config.node_count,
        )
        raw = np.stack([raw_map[n] for n in item["names"]])
        truth = raw[:, 52:172]
        initial = raw[:, :2].copy(order="C")
        actions = raw[:, 2:172, config.clamped_nodes].copy(order="C")
        assert array_digest(initial) == array_digest(model["initial"])
        assert array_digest(actions) == array_digest(model["actions"])
        observed = np.stack(
            [raw[:, t + 2, n] for t in (25, 33, 41, 49) for n in config.observed_nodes],
            axis=1,
        ).astype(np.float64)
        reference = np.stack(
            [
                model["incumbent"][:, t, n]
                for t in (25, 33, 41, 49)
                for n in config.observed_nodes
            ],
            axis=1,
        )
        assert array_digest(observed) == array_digest(fits["observations"])
        assert array_digest(reference) == array_digest(fits["reference"])
        checkpoint = torch.load(
            item["checkpoint"]["path"], map_location="cpu", weights_only=True
        )["model_state_dict"]
        rod = runner.multi.MultiObjectRod(
            modules, torch, checkpoint, config, item["object"]
        )
        with torch.no_grad():
            zero = np.zeros((len(observed), 4, config.node_count, 3))
            nominal = replay_arm(rod, initial, actions, zero, zero, torch)
            assert array_digest(nominal) == array_digest(
                model["nominal_from_anchor"][:, 25:]
            )
            for arm in NATIVE_ARMS:
                start = 0 if arm.endswith("16") else 8
                for i in range(len(observed)):
                    design = independent_design(
                        model["response"][i], arm, config.observed_nodes
                    )
                    mean, posterior = batch_posterior(
                        design[start:], (observed[i] - reference[i])[start:]
                    )
                    np.testing.assert_allclose(
                        mean, fits[arm + "__coefficients"][i], atol=1e-9, rtol=1e-8
                    )
                    np.testing.assert_allclose(
                        posterior, fits[arm + "__posterior"][i], atol=1e-10, rtol=1e-8
                    )
                pose, velocity, gain = independent_impulses(
                    fits[arm + "__coefficients"],
                    arm,
                    config.node_count,
                    config.clamped_nodes,
                )
                np.testing.assert_allclose(
                    pose, fits[arm + "__pose"], atol=1e-13, rtol=1e-12
                )
                np.testing.assert_allclose(
                    velocity, fits[arm + "__velocity"], atol=1e-13, rtol=1e-12
                )
                np.testing.assert_allclose(gain, fits[arm + "__gain"], atol=1e-13)
                future = replay_arm(
                    rod,
                    initial,
                    actions,
                    fits[arm + "__pose"],
                    fits[arm + "__velocity"],
                    torch,
                )
                expected = (
                    model["incumbent"][:, 50:]
                    + future.astype(np.float64)
                    - nominal.astype(np.float64)
                )
                np.testing.assert_allclose(
                    predictions[arm], expected, atol=2e-6, rtol=1e-6
                )
                native_count += len(observed)
                columns = list(range(24)) + (
                    list(range(24, 60))
                    if arm == "weak_16"
                    else list(range(48, 60))
                    if arm == "weak_8"
                    else []
                )
                jacobian = model["response"][:, 25:, :, :, columns]
                physical_p = fits[arm + "__posterior"][:, :-3, :-3]
                propagated = (jacobian @ physical_p[:, None, None]) @ jacobian.swapaxes(
                    -2, -1
                )
                expected_cov = propagated * gain[
                    :, None, None, None, None
                ] ** 2 + 9e-6 * np.eye(3)
                np.testing.assert_allclose(
                    covariances[arm], expected_cov, atol=1e-11, rtol=1e-9
                )
        # Separate normal-equation implementation of the matched temporal fit.
        times = (np.asarray((25, 33, 41, 49)) - 49) * 0.01
        design = np.column_stack((np.ones(4), times))
        residual = (observed - reference).reshape(len(observed), 4, 4, 3)
        coefficients = np.linalg.solve(
            design.T @ design, design.T @ residual.transpose(1, 0, 2, 3).reshape(4, -1)
        ).reshape(2, len(observed), 4, 3)
        interpolated = []
        for component in coefficients:
            full = np.zeros((len(observed), config.node_count, 3))
            knots = sorted((*config.observed_nodes, *config.clamped_nodes))
            for i in range(len(observed)):
                for axis in range(3):
                    values = [
                        component[i, config.observed_nodes.index(n), axis]
                        if n in config.observed_nodes
                        else 0
                        for n in knots
                    ]
                    full[i, :, axis] = np.interp(
                        np.arange(config.node_count), knots, values
                    )
            interpolated.append(full)
        dx, dv = interpolated
        gain = 1 / np.maximum(
            1,
            np.maximum(
                np.linalg.norm(dx, axis=-1).max(axis=1) / 0.03,
                np.linalg.norm(dv, axis=-1).max(axis=1) / 0.3,
            ),
        )
        dx, dv = dx * gain[:, None, None], dv * gain[:, None, None]
        np.testing.assert_allclose(fits["ols_pose"], dx, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(fits["ols_velocity"], dv, atol=1e-12, rtol=1e-10)
        expected = (
            model["incumbent"][:, 50:]
            + dx[:, None]
            + np.arange(1, 121)[None, :, None, None] * 0.01 * dv[:, None]
        )
        np.testing.assert_allclose(
            predictions["ols_readout_16"], expected, atol=1e-12, rtol=1e-10
        )
        with torch.no_grad():
            pose = np.zeros((len(observed), 4, config.node_count, 3))
            velocity = np.zeros_like(pose)
            pose[:, -1], velocity[:, -1] = fits["ols_pose"], fits["ols_velocity"]
            future = replay_arm(rod, initial, actions, pose, velocity, torch)
            expected = (
                model["incumbent"][:, 50:]
                + future.astype(np.float64)
                - nominal.astype(np.float64)
            )
            np.testing.assert_allclose(
                predictions["ols_physical_16"], expected, atol=2e-6, rtol=1e-6
            )
            _, state = rod.rollout(rod.initialize(initial), actions[:, :26])
            for step, frame in enumerate((25, 33, 41, 49)):
                if step:
                    _, state = rod.rollout(state, actions[:, frame - 7 : frame + 1])
                node_positions = (
                    state.positions.detach().cpu().numpy().astype(np.float64)
                )
                residual = observed[:, step * 4 : (step + 1) * 4] - (
                    model["incumbent"][:, frame, config.observed_nodes]
                    + node_positions[:, config.observed_nodes]
                    - model["nominal_from_anchor"][:, frame - 25, config.observed_nodes]
                )
                increment = np.zeros_like(node_positions)
                increment[:, config.observed_nodes] = residual
                state = update_rod_state(
                    state,
                    torch.tensor(increment, dtype=torch.float32),
                    torch.zeros_like(state.velocity),
                    gain=1,
                    clamped_nodes=config.clamped_nodes,
                )
            future, _ = rod.rollout(state, actions[:, 50:])
            expected = (
                model["incumbent"][:, 50:]
                + future.astype(np.float64)
                - nominal.astype(np.float64)
            )
            np.testing.assert_allclose(
                predictions["periodic_pose_16"], expected, atol=2e-6, rtol=1e-6
            )
            native_count += 2 * len(observed)
        keep = [
            i
            for i, name in enumerate(item["names"])
            if name != item["excluded_design_case"]
        ]
        count += metric_verifier.verify_rows(
            predictions,
            truth,
            result["objects"][item["object"]]["point"],
            keep,
            list(config.hidden_nodes),
            parent,
        )
        for family, mean_arm in CALIBRATION_FAMILIES.items():
            error = predictions[mean_arm][keep][:, :, config.hidden_nodes].astype(
                np.float64
            ) - truth[keep][:, :, config.hidden_nodes].astype(np.float64)
            unscaled = (
                covariances["weak_16"][keep][:, :, config.hidden_nodes]
                if family.endswith("shaped")
                else np.broadcast_to(9e-6 * np.eye(3), (*error.shape, 3)).copy()
            )
            if item["object"] == "DLO2":
                assert calibration["source_truth_sha256"] == array_digest(truth)
                scores = independent_uq(error, unscaled)["nees"]
                for j, frames in enumerate(np.array_split(np.arange(120), 3)):
                    values = scores[:, frames].reshape(13, -1)
                    moment = max(1e-6, float(np.mean(values.mean(axis=1)) / 3))
                    order = int(np.ceil(0.9 * (values.shape[1] - 1)))
                    conformal = max(
                        1e-6,
                        float(
                            np.max(np.sort(values, axis=1)[:, order])
                            / 6.251388631170325
                        ),
                    )
                    np.testing.assert_allclose(
                        calibration["scales"][family]["moment"][j], moment, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        calibration["scales"][family]["conformal"][j],
                        conformal,
                        rtol=1e-10,
                    )
            for variant, scales in calibration["scales"][family].items():
                covariance = unscaled * np.repeat(scales, 40)[None, :, None, None, None]
                events = independent_uq(error, covariance)
                recorded = result["objects"][item["object"]]["uq"][
                    family + "__" + variant
                ]
                for key, value in events.items():
                    np.testing.assert_allclose(
                        recorded["per_case"][key],
                        value.mean(axis=(1, 2)),
                        rtol=1e-9,
                        atol=1e-9,
                    )
                    np.testing.assert_allclose(
                        recorded["summary"][key],
                        value.mean(axis=(1, 2)).mean(),
                        rtol=1e-9,
                        atol=1e-9,
                    )
                    for label, frames in zip(
                        ("early", "middle", "late"),
                        np.array_split(np.arange(120), 3),
                        strict=True,
                    ):
                        np.testing.assert_allclose(
                            recorded["horizons"][label][key],
                            value[:, frames].mean(),
                            rtol=1e-9,
                            atol=1e-9,
                        )
                marginal_events += int(np.prod(error.shape[:-1]))
    assert result["decision"] == primary_decision(result["objects"], BeliefConfig())
    assert result["aggregate"] == runner.previous.summarize_aggregates(
        {name: {"clean": value["point"]} for name, value in result["objects"].items()}
    )
    return {
        "schema": "deform-weak-constraint-belief-independent-verification-v1",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": file_digest(args.source_receipt),
        "prediction_barrier_sha256": file_digest(
            args.output / "prediction_barrier.json"
        ),
        "calibration_sha256": file_digest(args.output / "calibration.json"),
        "result_sha256": file_digest(args.output / "result.json"),
        "verifier_sha256": file_digest(Path(__file__)),
        "verified": True,
        "forecasts_metric_verified": count,
        "native_forecasts_replayed": native_count,
        "marginal_uq_events_verified": marginal_events,
        "independent_batch_inference": True,
        "independent_physical_increments": True,
        "independent_covariance_and_calibration": True,
        "binary_coverage_boundary_convention": "registered-direct-solve-order-only-within-64-float64-epsilon-of-chi2-threshold",
        "old_means_byte_identical": True,
        "protected_data_access": False,
        "new_official_evaluation": False,
    }


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--prediction-source-root", type=Path)
    args = parser.parse_args()
    if args.prediction_source_root is not None:
        ROOT = args.prediction_source_root.resolve()
    record = verify(args)
    write_json_once(args.verification_output, record)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
