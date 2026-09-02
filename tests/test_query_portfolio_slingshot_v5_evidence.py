from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.query_portfolio_evidence_v2 import load_component_evidence
from bayesian_phystwin.query_portfolio_replication_v1 import (
    HARM_QUERY_ALPHA,
    REWARD_MARGIN,
    WORLD_COUNT,
    _gain_lower,
    one_sided_binomial_upper_bound,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "results/sota/query_portfolio_replication_v5/slingshot_component_evidence.json"
)
RESULT = ROOT / "results/sota/query_portfolio_replication_v5/slingshot_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_slingshot_v5_component_is_complete_and_passes_registered_gates() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    outcome = load_component_evidence(record)
    harmful = int(np.count_nonzero(outcome.gain < -REWARD_MARGIN))
    assert outcome.ordinary_success.all()
    assert int(outcome.candidate_deployed.sum()) == 65
    assert _gain_lower(outcome.gain, query_index=1) > 0.0
    assert (
        one_sided_binomial_upper_bound(
            harmful, WORLD_COUNT, 1.0 - HARM_QUERY_ALPHA
        )
        <= 0.05
    )


def test_slingshot_v5_evidence_files_retain_registered_hashes() -> None:
    assert _sha256(EVIDENCE) == (
        "c7ff4fa1adfb8f50b5076b00e0265c9c016c478c9d545e8cb8e65c1fed67ea69"
    )
    assert _sha256(RESULT) == (
        "f390e64af7a0236cef36b8c5dc246b8b26a22eac644abf46805f3fd43c0cacfd"
    )
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["artifact_id"] == (
        "7f5622d544d2c8f14c054c28014686b1113af427cae4eb57ee4d92ac6f2cd52d"
    )
    assert result["ordinary_evaluation_worlds"] == 320
    assert result["technical_failures"] == 0
