from __future__ import annotations

from pathlib import Path

from causal4d_public.deform360_causal_transport_method import (
    causal_transport_candidates,
    load_causal_transport_method,
)


CONFIG = (
    Path(__file__).parents[1]
    / "configs/causal4d_public/deform360_causal_contact_transport_source_v1.json"
)


def test_canonical_causal_transport_method_loads() -> None:
    method = load_causal_transport_method(CONFIG)
    candidates = causal_transport_candidates(method)

    assert len(candidates) == 49
    assert candidates[0]["label"] == "persistence"
    assert candidates[0]["initial_contact_gain"] == 0.0
    assert len({candidate["label"] for candidate in candidates}) == len(candidates)
    assert any(
        candidate["base_support_scale_m"] == 0.01
        and candidate["support_growth_per_travel"] == 2.0
        and candidate["initial_contact_gain"] == 1.0
        and candidate["transform_mode"] == "se3"
        for candidate in candidates
    )
