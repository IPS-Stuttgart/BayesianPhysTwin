"""No-growth ratchets for the frozen Causal4D compatibility providers."""

from __future__ import annotations

import hashlib
import importlib
import json


# Provider v1 is retained for frozen scientific and diagnostic consumers. New
# capabilities belong in a role-specific versioned facade. Intentional changes
# require a dedicated compatibility review that updates both count and digest.
_FROZEN_SCIENTIFIC_V1_EXPORT_COUNT = 74
_FROZEN_SCIENTIFIC_V1_EXPORT_SHA256 = (
    "c6491bdbd6fa008453d54309d724d0eedadf9ba024dd692ff48ef8825eb5620b"
)
_FROZEN_PROVIDER_V1_EXPORT_COUNT = 86
_FROZEN_PROVIDER_V1_EXPORT_SHA256 = (
    "530b7ace4afeb8fc497823e3790c4687efeadef4b93f102028e4e3ee73710afa"
)


def _export_digest(exports: list[str]) -> str:
    payload = json.dumps(
        sorted(exports),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_scientific_provider_v1_is_a_no_growth_surface() -> None:
    scientific_v1 = importlib.import_module(
        "bayesian_phystwin.causal4d_scientific_provider_v1"
    )
    exports = scientific_v1.__all__

    assert len(exports) == len(set(exports)) == _FROZEN_SCIENTIFIC_V1_EXPORT_COUNT
    assert _export_digest(exports) == _FROZEN_SCIENTIFIC_V1_EXPORT_SHA256


def test_aggregate_provider_v1_is_a_no_growth_surface() -> None:
    provider_v1 = importlib.import_module("bayesian_phystwin.causal4d_provider_v1")
    exports = provider_v1.__all__

    assert len(exports) == len(set(exports)) == _FROZEN_PROVIDER_V1_EXPORT_COUNT
    assert _export_digest(exports) == _FROZEN_PROVIDER_V1_EXPORT_SHA256
