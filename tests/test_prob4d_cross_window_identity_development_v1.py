from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("prob4d.cross_window_tracklets")
from prob4d.cross_window_tracklets import CrossWindowAssociationConfig  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / "scripts" / "science"
if str(SCIENCE) not in sys.path:
    sys.path.insert(0, str(SCIENCE))

from prob4d_cross_window_identity_development_v1 import (  # noqa: E402
    SOURCE_LINKED,
    AssociationCandidateConfig,
    GroupAssociation,
    _batch_for_method,
    association_configurations,
    association_counts,
    build_association_context,
    load_protocol,
    run_association,
)
from prob4d_bpt_controlled_decisive_core_v1 import generate_group  # noqa: E402


PROTOCOL = ROOT / "protocols" / "prob4d_cross_window_identity_development_v1.json"


def test_development_protocol_has_no_confirmatory_target_seed() -> None:
    protocol = load_protocol(PROTOCOL)

    assert protocol.raw["confirmatory_target"]["seeds_committed"] is False
    assert "seed_start" not in protocol.raw["confirmatory_target"]
    assert {
        protocol.association_partition.seed_start,
        protocol.pilot_guard_partition.seed_start,
        protocol.pilot_evaluation_partition.seed_start,
    } == {71260805, 171260805, 271260805}


def test_association_grid_is_deterministic_and_unique() -> None:
    protocol = load_protocol(PROTOCOL)
    first = association_configurations(protocol.raw)
    second = association_configurations(protocol.raw)

    assert len(first) == 108
    assert [value.configuration_id for value in first] == [
        value.configuration_id for value in second
    ]
    assert len({value.configuration_id for value in first}) == len(first)


def test_nominal_source_association_recovers_permuted_material_ids() -> None:
    protocol = load_protocol(PROTOCOL)
    group = generate_group(
        8801001,
        "nominal_correlated",
        protocol.base_config,
        group_prefix="test",
    )
    context = build_association_context(group)
    candidate = AssociationCandidateConfig(
        use_covariance=False,
        configuration=CrossWindowAssociationConfig(
            minimum_shared_frames=2,
            minimum_effective_support=1.5,
            isotropic_distance_scale_m=0.024,
            covariance_floor_m2=1e-10,
            maximum_weighted_rms_m=0.032,
            maximum_shared_frame_distance_m=0.096,
            maximum_spatial_candidate_pairs=1_000_000,
            minimum_compatibility_score=0.01,
            minimum_score_margin=0.0,
        ),
    )

    result = run_association(context, candidate)
    metrics = association_counts(context, result).metrics()

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] >= 0.80
    assert result.result_id == run_association(context, candidate).result_id


def test_source_linked_batch_uses_only_admitted_cross_window_rows() -> None:
    protocol = load_protocol(PROTOCOL)
    group = generate_group(
        8801002,
        "nominal_correlated",
        protocol.base_config,
        group_prefix="test",
    )
    context = build_association_context(group)
    candidate = AssociationCandidateConfig(
        use_covariance=False,
        configuration=CrossWindowAssociationConfig(
            minimum_shared_frames=2,
            minimum_effective_support=1.5,
            isotropic_distance_scale_m=0.024,
            covariance_floor_m2=1e-10,
            maximum_weighted_rms_m=0.032,
            maximum_shared_frame_distance_m=0.096,
            maximum_spatial_candidate_pairs=1_000_000,
            minimum_compatibility_score=0.01,
            minimum_score_margin=0.0,
        ),
    )
    result = run_association(context, candidate)
    association = GroupAssociation(
        context=context,
        result=result,
        counts=association_counts(context, result),
    )

    batch = _batch_for_method(
        group,
        association,
        SOURCE_LINKED,
        protocol.base_config,
    )

    point_count = protocol.base_config.point_count
    assert len(batch.innovation_m) == 2 * point_count + 2 * len(result.links)
    assert batch.metadata["accepted_cross_window_links"] == len(result.links)
    assert batch.metadata["target_identity_labels_used_for_batch"] is False


def test_protocol_loader_rejects_target_seed_in_development(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["confirmatory_target"]["seed_start"] = 123
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="target seeds"):
        load_protocol(path)
