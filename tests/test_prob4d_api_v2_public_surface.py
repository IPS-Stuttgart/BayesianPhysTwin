from __future__ import annotations

import bayesian_phystwin.v1 as api_v1
from bayesian_phystwin import prob4d_api_v2


def test_versioned_bayesian_phystwin_api_exposes_prob4d_v2_bridge() -> None:
    assert api_v1.Prob4DApiV2Compatibility is (
        prob4d_api_v2.Prob4DApiV2Compatibility
    )
    assert api_v1.inspect_prob4d_api_v2 is prob4d_api_v2.inspect_prob4d_api_v2
    assert api_v1.load_claim_bearing_tree_sparse_prob4d is (
        prob4d_api_v2.load_claim_bearing_tree_sparse_prob4d
    )
    assert api_v1.PROB4D_REQUIRED_API_VERSION == 2
    assert api_v1.PROB4D_REQUIRED_PROVIDER_API_VERSION == 2
    assert api_v1.PROB4D_REQUIRED_FACTOR_API_VERSION == 2
