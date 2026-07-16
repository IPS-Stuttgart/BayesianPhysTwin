from __future__ import annotations

import json

import numpy as np
import pytest

from causal4d_public.deform360_material_association import (
    FilamentMaterialAssociationConfig,
    fit_contact_anchored_material_association,
    load_material_association_artifact,
    write_material_association_artifact,
)


def _association_inputs() -> tuple[np.ndarray, ...]:
    coordinate = np.linspace(0.0, 1.0, 72)
    initial = np.column_stack(
        (
            0.6 * coordinate,
            0.002 * np.sin(8.0 * np.pi * coordinate),
            0.002 * np.cos(6.0 * np.pi * coordinate),
        )
    )
    tracks = np.stack(
        [
            initial
            + np.asarray([0.0, 0.001 * frame, 0.0002 * frame])
            + np.column_stack(
                (
                    np.zeros(len(coordinate)),
                    0.0001 * frame * coordinate,
                    np.zeros(len(coordinate)),
                )
            )
            for frame in range(6)
        ]
    )
    support = np.linspace(0.65, 1.0, len(coordinate))
    opacity = np.full(len(coordinate), 0.1)
    complete = np.ones(len(coordinate), dtype=bool)
    offsets = np.linspace(-0.003, 0.003, 12)
    left_taxels = initial[0] + np.column_stack((offsets, np.zeros(12), np.zeros(12)))
    right_taxels = initial[-1] + np.column_stack((offsets, np.zeros(12), np.zeros(12)))
    return tracks, support, opacity, complete, left_taxels, right_taxels


def _fit():
    tracks, support, opacity, complete, left, right = _association_inputs()
    return fit_contact_anchored_material_association(
        tracks,
        support,
        opacity,
        complete,
        (left, right),
        object_id="synthetic-rope",
        episode_id="synthetic-rope/episode_0000",
        source_sha256={"prefix_tracks": "a" * 64, "robot": "b" * 64},
        config=FilamentMaterialAssociationConfig(
            neighbor_count=6,
            node_count=11,
            contact_patch_taxel_count=4,
            minimum_node_contributors=3,
        ),
    )


def test_contact_anchored_association_is_prefix_only_and_metric() -> None:
    association = _fit()

    assert association.prefix_node_tracks_m.shape == (6, 11, 3)
    assert len(association.contact_anchors) == 2
    assert len({value.contact_node_index for value in association.contact_anchors}) == 2
    assert np.all(
        (association.node_reliability >= 0.0) & (association.node_reliability <= 1.0)
    )
    assert np.all(association.node_observation_variance_m2 >= 0.0)
    assert np.max(association.node_observation_variance_m2) < 0.01
    np.testing.assert_allclose(association.slice_weights.sum(axis=1), 1.0)


def test_material_association_round_trip_checksums_archive(tmp_path) -> None:
    association = _fit()
    metadata, archive = write_material_association_artifact(
        tmp_path / "association.json", association
    )

    restored = load_material_association_artifact(metadata)
    np.testing.assert_array_equal(
        restored.selected_initial_ids, association.selected_initial_ids
    )
    np.testing.assert_allclose(
        restored.node_observation_variance_m2,
        association.node_observation_variance_m2,
    )
    payload = json.loads(metadata.read_text())
    assert payload["information_boundary"]["future_splats_read"] is False
    assert (
        payload["information_boundary"]["state_innovation_used_for_prior_reliability"]
        is False
    )

    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        load_material_association_artifact(metadata)


def test_correlated_contributors_do_not_accumulate_probability_mass() -> None:
    association = _fit()

    assert np.max(association.node_reliability) <= 1.0
    assert np.max(association.node_effective_sample_size) > 1.0
    assert (
        "do not accumulate independently"
        in association.graph_diagnostics["uncertainty_policy"]
    )


def test_missing_camera_support_is_zero_reliability() -> None:
    tracks, support, opacity, complete, left, right = _association_inputs()
    support[10] = np.nan

    association = fit_contact_anchored_material_association(
        tracks,
        support,
        opacity,
        complete,
        (left, right),
        object_id="synthetic-rope",
        episode_id="synthetic-rope/episode_0000",
        source_sha256={"prefix_tracks": "a" * 64},
        config=FilamentMaterialAssociationConfig(
            neighbor_count=6,
            node_count=11,
            contact_patch_taxel_count=4,
            minimum_node_contributors=3,
        ),
    )

    assert 10 not in association.selected_initial_ids


def test_association_rejects_noncausal_frame_count() -> None:
    tracks, support, opacity, complete, left, right = _association_inputs()
    with pytest.raises(ValueError, match="wrong shape"):
        fit_contact_anchored_material_association(
            tracks[:-1],
            support,
            opacity,
            complete,
            (left, right),
            object_id="synthetic-rope",
            episode_id="synthetic-rope/episode_0000",
            source_sha256={"prefix_tracks": "a" * 64},
        )
