import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)

GOLDEN_ARTIFACT_ID = "9c02e638f60424cca7738d347d1258acd208eb562f422efacd077db4edb2fe80"


def _belief() -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    return ObservationBeliefV1(
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        view_names=("camera0",),
        window_names=("window0", "window1"),
        factor_names=("gauge_latent_0", "gauge_latent_1"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 1, 0, 1]),
        view_indices=np.zeros(4, dtype=int),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([0, 0, 1, 1]),
        factor_group_ids=np.asarray([0, 0, 1, 1]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=np.ones(4),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={"causal_source": "prefix only"},
    )


def _asymmetric_covariance() -> np.ndarray:
    covariance = _belief().local_covariance_m2.copy()
    covariance[0, 0, 1] = 1e-3
    return covariance


def _write_observation_archive(
    path: Path,
    belief: ObservationBeliefV1,
    *,
    descriptor_changes: dict[str, Any],
) -> None:
    descriptor = {
        **belief._descriptor(),
        "artifact_id": belief.artifact_id,
        **descriptor_changes,
    }
    payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    payload.update(belief._arrays())
    np.savez_compressed(path, **payload)


def test_observation_belief_round_trip_and_digest(tmp_path: Path) -> None:
    belief = _belief()
    path = tmp_path / "belief.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)

    assert belief.artifact_id == GOLDEN_ARTIFACT_ID
    assert restored.artifact_id == belief.artifact_id
    assert restored.summary()["observation_count"] == 4
    assert restored.mean_xyz_m.flags.writeable is False
    np.testing.assert_array_equal(restored.mean_xyz_m, belief.mean_xyz_m)


def test_observation_metadata_is_deeply_immutable_and_digest_stable(
    tmp_path: Path,
) -> None:
    metadata_input = {
        "nested": {
            "items": [1, {"accepted": True}],
            "tuple_items": (2, 3),
        }
    }
    belief = ObservationBeliefV1(
        **{
            **_belief().__dict__,
            "metadata": metadata_input,
        }
    )
    artifact_id = belief.artifact_id

    metadata_input["nested"]["items"][1]["accepted"] = False
    assert belief.metadata["nested"]["items"][1]["accepted"] is True
    assert belief.artifact_id == artifact_id

    frozen_metadata = belief.metadata
    frozen_items = frozen_metadata["nested"]["items"]
    assert isinstance(frozen_metadata, dict)
    assert isinstance(frozen_items, list)
    assert frozen_items.count(1) == 1
    assert set(frozen_metadata.keys()) == {"nested"}

    shallow = copy.copy(frozen_metadata)
    deep = copy.deepcopy(frozen_metadata)
    shallow_items = copy.copy(frozen_items)
    deep_items = copy.deepcopy(frozen_items)
    assert type(shallow) is dict
    assert type(deep) is dict
    assert type(shallow_items) is list
    assert type(deep_items) is list
    assert type(deep["nested"]["items"]) is list
    deep["nested"]["items"].append("copy-only")
    deep_items.append("copy-only")
    assert "copy-only" not in frozen_items

    with pytest.raises(TypeError):
        frozen_metadata["new"] = "mutated"
    with pytest.raises(TypeError):
        frozen_metadata["nested"]["items"][1]["accepted"] = False
    with pytest.raises(TypeError):
        del frozen_metadata["nested"]
    with pytest.raises(TypeError):
        frozen_metadata.update({"new": "mutated"})
    with pytest.raises(TypeError):
        frozen_metadata.__ior__({"new": "mutated"})
    with pytest.raises(TypeError):
        frozen_items.append("mutated")
    with pytest.raises(TypeError):
        frozen_items[0] = 9
    with pytest.raises(TypeError):
        del frozen_items[0]
    with pytest.raises(TypeError):
        frozen_items.__iadd__(["mutated"])
    with pytest.raises(TypeError):
        frozen_items.__imul__(2)

    path = tmp_path / "nested-belief.npz"
    save_observation_belief(path, belief)
    restored = load_observation_belief(path)
    assert restored.artifact_id == artifact_id
    assert restored.metadata == belief.metadata


@pytest.mark.parametrize(
    "metadata",
    (
        {"unsupported": object()},
        {"nonfinite": float("nan")},
    ),
)
def test_observation_metadata_rejects_non_json_values(metadata: object) -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        ObservationBeliefV1(
            **{
                **_belief().__dict__,
                "metadata": metadata,
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("case_id", "", "case_id must be nonempty"),
        ("case_id", cast(str, 1), "case_id must be nonempty"),
        ("stream_id", "", "stream_id must be nonempty"),
        ("stream_id", cast(str, 1), "stream_id must be nonempty"),
        ("source_repository", "", "source repository must be nonempty"),
        ("source_repository", cast(str, 1), "source repository must be nonempty"),
        ("source_revision", "", "source revision must be nonempty"),
        ("source_revision", cast(str, 1), "source revision must be nonempty"),
        ("source_artifact_sha256", "invalid", "lowercase SHA-256"),
        ("source_artifact_sha256", cast(str, 1), "lowercase SHA-256"),
    ),
)
def test_observation_belief_rejects_invalid_descriptor_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    source = _belief()

    with pytest.raises(ValueError, match=message):
        ObservationBeliefV1(
            **{
                **source.__dict__,
                field: value,
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"declared_frame_ids": np.asarray([8, 12])},
            "declared frames must lie before",
        ),
        ({"frame_ids": np.asarray([8, 8, 9])}, "frame_ids must have shape"),
        (
            {"local_covariance_m2": np.zeros((4, 2, 2))},
            "local_covariance_m2 must have shape",
        ),
        (
            {"low_rank_factor_m": np.zeros((4, 3, 1))},
            "low_rank_factor_m must have shape",
        ),
        (
            {"frame_ids": np.asarray([8, 8, 10, 10])},
            "frame_ids must be contained",
        ),
        (
            {"view_indices": np.asarray([0, 0, 0, 1])},
            "view_indices reference unavailable",
        ),
        ({"local_covariance_m2": _asymmetric_covariance()}, "must be symmetric"),
        ({"group_ids": np.asarray([0])}, "group_ids must equal"),
        (
            {"group_prior_nominal_probability": np.asarray([0.85])},
            "group prior and composite weight",
        ),
        (
            {"group_prior_nominal_probability": np.asarray([0.85, 1.1])},
            "group prior nominal probabilities",
        ),
        (
            {"group_composite_weight": np.asarray([0.5, 0.0])},
            "group composite weights",
        ),
    ),
)
def test_observation_belief_validates_reformatted_schema_guards(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ObservationBeliefV1(
            **{
                **_belief().__dict__,
                **changes,
            }
        )


def test_observation_belief_rejects_nonfinite_low_rank_factors() -> None:
    source = _belief()
    factors = source.low_rank_factor_m.copy()
    factors[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="low-rank factors must be finite"):
        ObservationBeliefV1(
            **{
                **source.__dict__,
                "low_rank_factor_m": factors,
            }
        )


@pytest.mark.parametrize(
    "causal_frame_stop",
    (True, 12.0, np.float64(12.0)),
)
def test_observation_belief_rejects_noninteger_causal_cutoff(
    causal_frame_stop: object,
) -> None:
    with pytest.raises(ValueError, match="causal_frame_stop must be an integer"):
        ObservationBeliefV1(
            **{
                **_belief().__dict__,
                "causal_frame_stop": causal_frame_stop,
            }
        )


@pytest.mark.parametrize(
    "field",
    (
        "declared_frame_ids",
        "frame_ids",
        "entity_ids",
        "view_indices",
        "window_indices",
        "correlation_group_ids",
        "factor_group_ids",
        "group_ids",
    ),
)
def test_observation_belief_rejects_float_identity_arrays(field: str) -> None:
    source = _belief()
    values = np.asarray(getattr(source, field), dtype=np.float64)

    with pytest.raises(ValueError, match=f"{field} must contain integers"):
        ObservationBeliefV1(
            **{
                **source.__dict__,
                field: values,
            }
        )


def test_observation_belief_canonicalizes_numpy_integer_and_name_sequences() -> None:
    source = _belief()
    belief = ObservationBeliefV1(
        **{
            **source.__dict__,
            "causal_frame_stop": np.int64(source.causal_frame_stop),
            "view_names": list(source.view_names),
            "window_names": list(source.window_names),
            "factor_names": list(source.factor_names),
        }
    )

    assert type(belief.causal_frame_stop) is int
    assert type(belief.view_names) is tuple
    assert type(belief.window_names) is tuple
    assert type(belief.factor_names) is tuple
    assert belief.artifact_id == source.artifact_id


def test_group_position_requires_a_genuine_integer() -> None:
    belief = _belief()

    with pytest.raises(ValueError, match="group_id must be an integer"):
        belief.group_position(0.0)
    with pytest.raises(KeyError, match="unknown correlation group"):
        belief.group_position(2)
    assert belief.group_position(np.int64(1)) == 1


def test_observation_loader_rejects_schema_version_drift(tmp_path: Path) -> None:
    belief = _belief()
    path = tmp_path / "schema-drift.npz"
    _write_observation_archive(
        path,
        belief,
        descriptor_changes={"schema_version": 2},
    )

    with pytest.raises(ValueError, match="unsupported observation-belief version"):
        load_observation_belief(path)


def test_observation_belief_rejects_future_frame() -> None:
    belief = _belief()
    with pytest.raises(ValueError, match="causal boundary"):
        ObservationBeliefV1(
            **{
                **belief.__dict__,
                "frame_ids": np.asarray([8, 8, 9, 12]),
            }
        )


def test_observation_belief_rejects_duplicate_identity() -> None:
    belief = _belief()
    with pytest.raises(ValueError, match="must be unique"):
        ObservationBeliefV1(
            **{
                **belief.__dict__,
                "entity_ids": np.asarray([0, 0, 0, 1]),
            }
        )


def test_sim3_transform_moves_covariance_and_factors() -> None:
    belief = _belief()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = belief.transformed(
        rotation=rotation,
        translation_m=np.asarray([1.0, 2.0, 3.0]),
        scale=2.0,
        stream_id="world",
    )

    expected = 2.0 * (rotation @ belief.mean_xyz_m[0]) + np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(transformed.mean_xyz_m[0], expected)
    np.testing.assert_allclose(
        transformed.local_covariance_m2[0], 4.0 * belief.local_covariance_m2[0]
    )
    assert transformed.artifact_id != belief.artifact_id
