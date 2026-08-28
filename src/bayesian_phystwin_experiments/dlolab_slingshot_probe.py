"""Bounded source-only early-pull information screen; no target policy claims."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dlolab_slingshot_belief import particle_worlds

FRACTIONS = (0.25, 0.5)
FRAMES = (139, 219, 299)


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-slingshot-informative-prefix-source-v1",
        "role": "source_information_and_control_value_screen_not_method_confirmation",
        "frontload_fractions": list(FRACTIONS),
        "redistribution": "add_fraction_of_incumbent_macro2_xyz_to_macro1_subtract_from_macro2",
        "rotation_commands_unchanged": True,
        "macro_count": 3,
        "translation_norm_limit_m": 0.1,
        "angle_component_limit_rad": 1.0,
        "prefix_steps": 300,
        "observation_frames": list(FRAMES),
        "observed_nodes": [3, 6, 8],
        "sphere_center_observed": True,
        "independent_noise_sd_m": 0.002,
        "shared_bias_sd_m": 0.005,
        "prefix_world_indices": list(range(9, 18)),
        "prefix_batches_per_probe": 2,
        "minimum_whitened_stretching_secant_norm": 1.0,
        "probe_selection": "highest_stretching_secant_norm_then_smaller_fraction",
        "full_source_world_count_if_prefix_passes": 27,
        "source_noise_draws_per_world": 8192,
        "source_noise_seed": 260907,
        "minimum_source_information_gain": 0.005,
        "minimum_relative_excess_information_gain": 0.1,
        "minimum_source_gain_over_map": 0.002,
        "require_reward_not_below_original_best_blind": True,
        "same_native_reward_physics_controller_and_release": True,
        "calibration_or_evaluation_worlds_read": False,
        "protected_data_read": False,
        "new_recordings": False,
        "gpu_work": False,
        "retry_authorized": False,
        "method_evaluation_authorized": False,
    }


def probe_controls(original: np.ndarray, index: int) -> np.ndarray:
    if type(index) is not int or index not in range(2):
        raise ValueError("unregistered probe")
    if (
        original.shape != (8, 3, 6)
        or original.dtype != np.float64
        or not np.isfinite(original).all()
        or not np.array_equal(original[5], original[7])
        or not np.all(original[:, 0] == original[5, 0])
    ):
        raise ValueError("complete shared-prefix control bank required")
    result = original.copy()
    shifted = FRACTIONS[index] * original[5, 1, :3]
    result[:, 0, :3] += shifted
    result[:, 1, :3] -= shifted
    if (
        np.max(np.linalg.norm(result[:, :, :3], axis=-1)) > 0.1 + 1e-12
        or np.max(np.abs(result[:, :, 3:])) > 1
    ):
        raise ValueError("probe exceeds frozen native action limits")
    return result


def prefix_task(probe: int, batch: int) -> dict[str, Any]:
    if (
        type(probe) is not int
        or type(batch) is not int
        or probe not in range(2)
        or batch not in range(2)
    ):
        raise ValueError("unregistered prefix task")
    indices = list(range(9 + batch * 8, min(17 + batch * 8, 18)))
    worlds = [particle_worlds()[i] for i in indices]
    return {
        "kind": "prefix",
        "name": f"probe-{probe}-prefix-{batch}",
        "probe": probe,
        "index": batch,
        "world_indices": indices,
        "worlds": worlds + [worlds[-1]] * (8 - len(worlds)),
        "prefix_only": True,
    }


def full_task(probe: int, index: int) -> dict[str, Any]:
    if (
        type(probe) is not int
        or type(index) is not int
        or probe not in range(2)
        or index not in range(27)
    ):
        raise ValueError("unregistered source world")
    return {
        "kind": "source",
        "name": f"source-world-{index:02d}",
        "probe": probe,
        "index": index,
        "world_indices": [index],
        "worlds": [particle_worlds()[index]] * 8,
        "prefix_only": False,
    }


def material_information(history: np.ndarray) -> dict[str, Any]:
    if history.shape != (9, 3, 4, 3) or not np.isfinite(history).all():
        raise ValueError("all nine finite material prefixes required")
    differences = np.stack(
        [(history[7] - history[1]) / 2, (history[5] - history[3]) / 2], axis=-1
    ).reshape(12, 6)
    covariance = 0.002**2 * np.eye(12) + 0.005**2 * np.ones((12, 12))
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), differences).reshape(
        36, 2
    )
    norms = np.linalg.norm(whitened, axis=0)
    return {
        "whitened_bending_secant_norm": float(norms[0]),
        "whitened_stretching_secant_norm": float(norms[1]),
        "whitened_material_secant_singular_values": np.linalg.svd(
            whitened, compute_uv=False
        ).tolist(),
    }


def select_probe(histories: list[np.ndarray], qa: list[bool]) -> dict[str, Any]:
    if len(histories) != 2 or len(qa) != 2 or any(type(v) is not bool for v in qa):
        raise ValueError("complete two-probe screen required")
    information = [material_information(history) for history in histories]
    eligible = [
        i
        for i in range(2)
        if qa[i] and information[i]["whitened_stretching_secant_norm"] >= 1
    ]
    selected = (
        min(
            eligible,
            key=lambda i: (-information[i]["whitened_stretching_secant_norm"], i),
        )
        if eligible and all(qa)
        else None
    )
    return {
        "information": information,
        "native_qa": qa,
        "selected_probe": selected,
        "source_bank_authorized": selected is not None,
        "new_probe_reward_read": False,
        "method_evaluation_authorized": False,
    }


def source_information_value(
    prefix: np.ndarray, reward: np.ndarray, prior: np.ndarray
) -> dict[str, Any]:
    if prefix.shape != (27, 3, 4, 3) or reward.shape != (27, 7) or prior.shape != (27,):
        raise ValueError("complete finite source bank required")
    if (
        any(not np.isfinite(v).all() for v in (prefix, reward, prior))
        or np.any(prior <= 0)
        or not np.isclose(prior.sum(), 1)
    ):
        raise ValueError("invalid source probabilities or arrays")
    rng = np.random.default_rng(260907)
    noise = rng.normal(0, 0.005, (8192, 1, 3)) + rng.normal(0, 0.002, (8192, 12, 3))
    h = prefix.reshape(27, 12, 3)
    covariance = 0.002**2 * np.eye(12) + 0.005**2 * np.ones((12, 12))
    chol = np.linalg.cholesky(covariance)
    white = (
        np.linalg.solve(chol, (h - h[13]).transpose(1, 0, 2).reshape(12, -1))
        .reshape(12, 27, 3)
        .transpose(1, 0, 2)
        .reshape(27, 36)
    )
    white_noise = (
        np.linalg.solve(chol, noise.transpose(1, 0, 2).reshape(12, -1))
        .reshape(12, 8192, 3)
        .transpose(1, 0, 2)
        .reshape(8192, 36)
    )
    norms = np.sum(white**2, axis=1)
    realized = np.zeros((8192, 2))
    for world in range(27):
        for start in range(0, 8192, 256):
            y = white[world] + white_noise[start : start + 256]
            d2 = np.sum(y**2, axis=1)[:, None] + norms - 2 * y @ white.T
            log_weight = np.log(prior) - 0.5 * np.maximum(d2, 0)
            w = np.exp(log_weight - np.max(log_weight, axis=1)[:, None])
            w /= w.sum(axis=1)[:, None]
            choice = np.stack(
                [
                    np.argmax(w @ reward, axis=1),
                    np.argmax(reward[np.argmax(w, axis=1)], axis=1),
                ],
                axis=1,
            )
            realized[start : start + 256] += prior[world] * reward[world, choice]
    blind = float(np.max(prior @ reward))
    return {
        "best_blind_action": int(np.argmax(prior @ reward)),
        "best_blind_reward": blind,
        "perfect_information_reward": float(prior @ np.max(reward, axis=1)),
        "posterior_mean_reward": float(realized[:, 0].mean()),
        "map_reward": float(realized[:, 1].mean()),
        "information_gain": float(realized[:, 0].mean() - blind),
        "posterior_gain_over_map": float(np.mean(realized[:, 0] - realized[:, 1])),
        "monte_carlo_standard_errors": (
            np.std(realized, axis=0, ddof=1) / np.sqrt(8192)
        ).tolist(),
        "integration_only_not_out_of_sample": True,
    }
