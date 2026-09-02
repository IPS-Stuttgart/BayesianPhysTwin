from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tracking_cloth_cross_action_transport_v1.run import (
    _certificate_for_material,
    _exact_pairing_scores,
    _read_protocol,
    build_pairwise_shape_trajectory,
    canonical_pca,
    constrained_affine_coefficients,
)


def _diameter(points: np.ndarray) -> float:
    first, second = np.triu_indices(points.shape[0], k=1)
    return float(np.max(np.linalg.norm(points[first] - points[second], axis=1)))


def test_pairwise_trajectory_is_rigid_and_scale_invariant() -> None:
    rng = np.random.default_rng(20260902)
    base = rng.normal(size=(20, 3))
    motion = rng.normal(scale=0.05, size=(5, 20, 3))
    cloth = base[None, :, :] + np.cumsum(motion, axis=0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = 3.7 * (cloth @ rotation) + np.asarray([8.0, -3.0, 2.0])

    first = build_pairwise_shape_trajectory(
        cloth,
        cutoff=1,
        initial_diameter_m=_diameter(cloth[0]),
    )
    second = build_pairwise_shape_trajectory(
        transformed,
        cutoff=1,
        initial_diameter_m=_diameter(transformed[0]),
    )

    assert first.shape == (3, 190)
    assert second == pytest.approx(first, abs=1e-12)


def test_canonical_pca_is_deterministic_and_sign_fixed() -> None:
    rows = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, 1.0],
        ]
    )
    first = canonical_pca(rows, n_components=2)
    second = canonical_pca(rows, n_components=2)

    assert first[0] == pytest.approx(second[0])
    assert first[1] == pytest.approx(second[1])
    assert first[2] == pytest.approx(second[2])
    assert first[3] == pytest.approx(second[3])
    for component in first[1]:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] >= 0.0


def test_affine_coefficients_respect_sum_constraint() -> None:
    prototypes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = np.asarray([0.25, 0.5])
    coefficients = constrained_affine_coefficients(
        prototypes,
        query,
        constraint_weight=1e4,
    )

    assert coefficients.sum() == pytest.approx(1.0, abs=1e-8)
    assert coefficients @ prototypes == pytest.approx(query, abs=1e-8)


def test_centered_material_transport_is_identifiable_modulo_gauge() -> None:
    diagnostic = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    source_queries = np.stack(
        (
            diagnostic,
            2.0 * diagnostic,
            diagnostic @ np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
        axis=1,
    )
    source_features = np.zeros((4, 3, 2, 2), dtype=float)

    adequacy, quotient, designs, decisions = _certificate_for_material(
        material="synthetic",
        diagnostic_query=diagnostic[0],
        source_queries=source_queries,
        source_features=source_features,
        diagnostic_index=0,
        target_indices=(1, 2),
        interactions=("diagnostic", "target-a", "target-b"),
        noise_radius=1e-9,
        protocol_id="a" * 64,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
    )

    assert adequacy.status.value == "adequate_set_valued"
    assert adequacy.nullity >= 4
    assert designs == {}
    for target in ("target-a", "target-b"):
        record = quotient.record_for(target)
        assert record.full_transport_permitted
        assert decisions[target]["disposition"] == "transport_without_cause"


def test_identity_material_association_has_exact_one_over_24_p_value() -> None:
    target_bank = np.arange(4, dtype=float).reshape(4, 1, 1)
    source_features = np.stack((np.zeros_like(target_bank), target_bank), axis=1)
    truth = target_bank[:, None, :, :]
    coefficients = np.zeros((4, 7), dtype=float)
    for material in range(4):
        coefficients[material, :4] = -0.25
        coefficients[material, material] += 1.0

    rows, p_value, rank = _exact_pairing_scores(
        truth=truth,
        source_features=source_features,
        coefficients=coefficients,
        target_indices=(1,),
    )

    assert len(rows) == len(list(itertools.permutations(range(4)))) == 24
    assert p_value == pytest.approx(1.0 / 24.0)
    assert rank == 1
    assert rows[0]["identity"]


def test_protocol_preserves_retrospective_and_rep3_boundaries(tmp_path: Path) -> None:
    source = Path(
        "experiments/tracking_cloth_cross_action_transport_v1/protocol.json"
    )
    protocol = _read_protocol(source)
    assert protocol["model_repetition"] == 1
    assert protocol["retrospective_target_repetition"] == 2
    assert protocol["reserved_confirmation_repetition"] == 3
    assert protocol["retrospective_status"]["globally_fresh_target"] is False
    assert protocol["retrospective_status"]["rep3_confirmation_authorized"] is False

    changed = dict(protocol)
    changed["reserved_confirmation_repetition"] = 2
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved confirmation"):
        _read_protocol(path)
