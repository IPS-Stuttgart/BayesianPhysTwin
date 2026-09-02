"""Source-only triangle--rod contact prototype for the self-collision panel.

This diagnostic reads repetitions 1 and 2 only.  It never enumerates or opens
repetition-3 outcomes.  Its purpose is to decide whether continuous surface
contact is sufficiently promising to justify a separately reviewed protocol
revision before any confirmation data are touched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import InputView, audit_dataset, prediction_input, scoring_truth
from .model import (
    PHYSICS_ARM,
    grid_edges,
    kinematic_predictions,
    parameter_bank,
    rod_forecast,
    self_collision_pairs,
    trajectory_mse,
)
from .selection import (
    apply_policy,
    fit_cross_material_policies,
    incremental_summary,
    score_case,
    source_gate,
    summarize_policy_rows,
)


@dataclass(frozen=True)
class Variant:
    name: str
    contact_radius_m: float
    integration_substeps: int


def grid_triangles() -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for row in range(4):
        for column in range(3):
            upper_left = row * 4 + column
            upper_right = upper_left + 1
            lower_left = (row + 1) * 4 + column
            lower_right = lower_left + 1
            triangles.append((upper_left, upper_right, lower_right))
            triangles.append((upper_left, lower_right, lower_left))
    return np.asarray(triangles, dtype=int)


def _closest_on_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, float]:
    axis = end - start
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-12:
        raise ValueError("degenerate segment")
    fraction = float(np.clip(np.dot(point - start, axis) / denominator, 0.0, 1.0))
    return start + fraction * axis, fraction


def _closest_on_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point and barycentric weights; Ericson region tests."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0])

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1e-15)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0])

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1e-15)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denominator = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / max(denominator, 1e-15)
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w])

    denominator = va + vb + vc
    if abs(denominator) <= 1e-15:
        raise ValueError("degenerate triangle")
    v = vb / denominator
    w = vc / denominator
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w])


def _closest_segments(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Closest points on two finite segments and their fractions."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return p1, p2, 0.0, 0.0
    if a <= epsilon:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= epsilon:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
                if abs(denominator) > epsilon
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return p1 + s * d1, p2 + t * d2, s, t


def _segment_triangle_pair(
    rod_start: np.ndarray,
    rod_end: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Closest cloth/rod points, cloth barycentrics, and distance."""

    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    direction = rod_end - rod_start
    edge1 = b - a
    edge2 = c - a
    h = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, h))
    if abs(determinant) > 1e-12:
        inverse = 1.0 / determinant
        s = rod_start - a
        u = inverse * float(np.dot(s, h))
        q = np.cross(s, edge1)
        v = inverse * float(np.dot(direction, q))
        t = inverse * float(np.dot(edge2, q))
        if (
            -1e-10 <= u <= 1.0 + 1e-10
            and -1e-10 <= v
            and u + v <= 1.0 + 1e-10
            and -1e-10 <= t <= 1.0 + 1e-10
        ):
            barycentric = np.asarray([1.0 - u - v, u, v])
            point = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
            rod = rod_start + float(np.clip(t, 0.0, 1.0)) * direction
            return point, rod, barycentric, 0.0

    for rod_point in (rod_start, rod_end):
        triangle_point, barycentric = _closest_on_triangle(rod_point, a, b, c)
        distance = float(np.linalg.norm(triangle_point - rod_point))
        candidates.append((distance, triangle_point, rod_point, barycentric))

    vertices = (a, b, c)
    for vertex_index, vertex in enumerate(vertices):
        rod_point, _ = _closest_on_segment(vertex, rod_start, rod_end)
        barycentric = np.zeros(3)
        barycentric[vertex_index] = 1.0
        distance = float(np.linalg.norm(vertex - rod_point))
        candidates.append((distance, vertex, rod_point, barycentric))

    edge_specs = (
        (a, b, np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
        (b, c, np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        (c, a, np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
    )
    for edge_start, edge_end, bary_start, bary_end in edge_specs:
        cloth_point, rod_point, fraction, _ = _closest_segments(
            edge_start, edge_end, rod_start, rod_end
        )
        barycentric = (1.0 - fraction) * bary_start + fraction * bary_end
        distance = float(np.linalg.norm(cloth_point - rod_point))
        candidates.append((distance, cloth_point, rod_point, barycentric))

    distance, cloth_point, rod_point, barycentric = min(
        candidates, key=lambda item: item[0]
    )
    return cloth_point, rod_point, barycentric, distance


def triangle_contact_rollout(
    inputs: InputView,
    parameters: tuple[float, float, float],
    protocol: dict[str, Any],
    variant: Variant,
) -> np.ndarray:
    stiffness, damping, collision_stiffness = parameters
    links, relative_stiffness = grid_edges()
    left, right = links.T
    initial = inputs.cloth_prefix[0]
    rest_lengths = np.linalg.norm(initial[right] - initial[left], axis=1)
    if np.min(rest_lengths) <= 1e-6:
        raise ValueError("degenerate cloth spring")
    pairs = self_collision_pairs()
    pair_left, pair_right = pairs.T
    triangles = grid_triangles()
    predicted_rod = rod_forecast(inputs, protocol)
    output = np.empty((len(inputs.times), 20, 3), dtype=float)
    output[: inputs.cutoff + 1] = inputs.cloth_prefix
    x = inputs.cloth_prefix[-1].copy()
    prefix_times = inputs.times[: inputs.cutoff + 1]
    threshold = prefix_times[-1] - float(protocol["short_velocity_window_seconds"])
    first = min(int(np.searchsorted(prefix_times, threshold, side="left")), len(prefix_times) - 2)
    local_times = prefix_times[first:]
    centered = local_times - local_times.mean()
    velocity = np.einsum(
        "t,tnd->nd", centered, inputs.cloth_prefix[first:]
    ) / float(np.dot(centered, centered))
    gravity = float(protocol["gravity_m_s2"])
    friction_rate = float(protocol["rod_friction_rate"])
    collision_distance = float(protocol["self_collision_distance_m"])
    origin = initial.mean(axis=0)

    for index in range(inputs.cutoff + 1, len(inputs.times)):
        full_dt = float(inputs.times[index] - inputs.times[index - 1])
        dt = full_dt / variant.integration_substeps
        rod_previous = predicted_rod[index - 1]
        rod_current = predicted_rod[index]
        for substep in range(1, variant.integration_substeps + 1):
            fraction = substep / variant.integration_substeps
            rod = (1.0 - fraction) * rod_previous + fraction * rod_current
            delta = x[right] - x[left]
            length = np.linalg.norm(delta, axis=1)
            force = (
                stiffness
                * relative_stiffness
                * (length - rest_lengths)
                / np.maximum(length, 1e-9)
            )[:, None] * delta
            acceleration = -damping * velocity
            acceleration[:, 2] -= gravity
            np.add.at(acceleration, left, force)
            np.add.at(acceleration, right, -force)

            if collision_stiffness > 0 and len(pairs):
                pair_delta = x[pair_right] - x[pair_left]
                pair_length = np.linalg.norm(pair_delta, axis=1)
                active = pair_length < collision_distance
                if np.any(active):
                    normal = pair_delta[active] / np.maximum(
                        pair_length[active, None], 1e-9
                    )
                    repulsion = (
                        collision_stiffness
                        * (collision_distance - pair_length[active])[:, None]
                        * normal
                    )
                    np.add.at(acceleration, pair_left[active], -repulsion)
                    np.add.at(acceleration, pair_right[active], repulsion)

            velocity += dt * acceleration
            x += dt * velocity

            for triangle in triangles:
                a, b, c = x[triangle]
                cloth_point, rod_point, barycentric, distance = _segment_triangle_pair(
                    rod[0], rod[1], a, b, c
                )
                if distance >= variant.contact_radius_m:
                    continue
                if distance > 1e-10:
                    normal = (cloth_point - rod_point) / distance
                else:
                    normal = np.cross(b - a, c - a)
                    norm = float(np.linalg.norm(normal))
                    if norm <= 1e-12:
                        continue
                    normal /= norm
                    if normal[2] < 0.0:
                        normal = -normal
                denominator = float(np.dot(barycentric, barycentric))
                if denominator <= 1e-12:
                    continue
                correction = (variant.contact_radius_m - distance) * normal
                factors = barycentric / denominator
                x[triangle] += factors[:, None] * correction

                contact_velocity = np.einsum(
                    "i,id->d", barycentric, velocity[triangle]
                )
                normal_velocity = float(np.dot(contact_velocity, normal))
                delta_velocity = np.zeros(3)
                if normal_velocity < 0.0:
                    delta_velocity -= normal_velocity * normal
                tangential = contact_velocity - normal_velocity * normal
                delta_velocity += (
                    np.exp(-friction_rate * dt) - 1.0
                ) * tangential
                velocity[triangle] += factors[:, None] * delta_velocity

        if not np.isfinite(x).all() or not np.isfinite(velocity).all():
            raise ValueError("nonfinite triangle-contact rollout")
        if np.max(np.linalg.norm(x - origin, axis=1)) > 5.0:
            raise ValueError("triangle-contact rollout escaped the registered domain")
        output[index] = x
    return output


def _fit_key(case: Any) -> str:
    return f"{case.material}|{case.interaction}"


def evaluate_variant(
    cases: list[Any], protocol: dict[str, Any], variant: Variant
) -> dict[str, Any]:
    bank = parameter_bank(protocol)
    rep1 = [case for case in cases if case.repetition == 1]
    rep2 = [case for case in cases if case.repetition == 2]
    fits: dict[str, dict[str, Any]] = {}
    fit_rows = []
    for case in rep1:
        inputs = prediction_input(case, protocol)
        truth = scoring_truth(case, inputs)
        valid_parameters = []
        losses = []
        rejected = []
        for parameters in bank:
            try:
                prediction = triangle_contact_rollout(
                    inputs, parameters, protocol, variant
                )
                loss = trajectory_mse(prediction, truth, inputs)
            except ValueError as error:
                rejected.append({"parameters": list(parameters), "reason": str(error)})
                continue
            valid_parameters.append(parameters)
            losses.append(loss)
        if not valid_parameters:
            return {
                "variant": variant.__dict__,
                "status": "no-valid-particle",
                "failed_case": case.case_id,
                "rep3_numeric_outcomes_read": False,
            }
        loss_array = np.asarray(losses, dtype=float)
        temperature = max(
            float(np.min(loss_array)),
            float(protocol["measurement_floor_m"]) ** 2,
        )
        logits = -loss_array / (2.0 * temperature)
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
        fits[_fit_key(case)] = {
            "parameters": tuple(valid_parameters),
            "weights": weights,
        }
        fit_rows.append(
            {
                "case_id": case.case_id,
                "valid_particle_count": len(valid_parameters),
                "rejected_particle_count": len(rejected),
                "minimum_loss_m2": float(np.min(loss_array)),
                "maximum_loss_m2": float(np.max(loss_array)),
                "rejected": rejected,
            }
        )

    rep2_rows = []
    rep2_case_rows = []
    for case in rep2:
        inputs = prediction_input(case, protocol)
        truth = scoring_truth(case, inputs)
        fit = fits[_fit_key(case)]
        particle_predictions = []
        for parameters in fit["parameters"]:
            try:
                particle_predictions.append(
                    triangle_contact_rollout(inputs, parameters, protocol, variant)
                )
            except ValueError as error:
                return {
                    "variant": variant.__dict__,
                    "status": "rep2-particle-invalid",
                    "failed_case": case.case_id,
                    "failed_parameters": list(parameters),
                    "error": str(error),
                    "fit_rows": fit_rows,
                    "rep3_numeric_outcomes_read": False,
                }
        predictions = kinematic_predictions(inputs, protocol)
        bank_predictions = np.stack(particle_predictions, axis=0)
        predictions[PHYSICS_ARM] = np.einsum(
            "k,ktnd->tnd", fit["weights"], bank_predictions
        )
        rep2_rows.extend(score_case(predictions, truth, inputs, protocol))
        rep2_case_rows.append(
            {
                "case_id": case.case_id,
                "trajectory_mse_m2": {
                    arm: trajectory_mse(prediction, truth, inputs)
                    for arm, prediction in predictions.items()
                },
            }
        )

    policy = fit_cross_material_policies(rep2_rows, protocol)
    policy_rows = apply_policy(rep2_rows, policy)
    summaries = summarize_policy_rows(policy_rows, protocol)
    incremental = incremental_summary(policy_rows, protocol)
    gate = source_gate(summaries, incremental, protocol)
    return {
        "variant": variant.__dict__,
        "status": "complete",
        "fit_rows": fit_rows,
        "rep2_case_rows": rep2_case_rows,
        "summaries": summaries,
        "incremental": incremental,
        "source_gate": gate,
        "rep3_numeric_outcomes_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    cases, inventory = audit_dataset(args.dataset_root, protocol)
    variants = (
        Variant("triangle-contact-r8mm-s8", 0.008, 8),
        Variant("triangle-contact-r16mm-s8", 0.016, 8),
        Variant("triangle-contact-r24mm-s8", 0.024, 8),
    )
    results = [evaluate_variant(cases, protocol, variant) for variant in variants]
    record = {
        "schema": "bayesian-phystwin.self-collision-triangle-contact-source.v1",
        "dataset_inventory_id": inventory["inventory_id"],
        "audited_repetitions": [1, 2],
        "rep3_numeric_outcomes_read": False,
        "variants": results,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "triangle_contact_source_result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    compact = {
        "schema": record["schema"],
        "dataset_inventory_id": record["dataset_inventory_id"],
        "rep3_numeric_outcomes_read": False,
        "variants": [
            {
                "variant": result["variant"],
                "status": result["status"],
                "source_gate": result.get("source_gate"),
                "incremental": result.get("incremental"),
                "physics_summary": result.get("summaries", {}).get("physics_enabled"),
            }
            for result in results
        ],
    }
    (args.output / "compact.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
