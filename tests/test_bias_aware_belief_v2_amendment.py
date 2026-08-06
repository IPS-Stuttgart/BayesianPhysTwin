from __future__ import annotations

import hashlib
import json
from pathlib import Path

AMENDMENT = Path("protocols/amendments/bias_aware_belief_spd_v2.json")
FROZEN_V1 = Path("src/bayesian_phystwin/bias_aware_belief.py")
EXPECTED_V1_GIT_BLOB_SHA1 = "80994687a44b798c6b33089bfd4f1858911e0837"


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324


def test_spd_v2_amendment_preserves_the_frozen_v1_boundary() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    assert payload["schema"] == (
        "bayesian-phystwin/bias-aware-belief-numerical-amendment"
    )
    assert payload["schema_version"] == 1
    assert payload["frozen_v1"]["source_bytes_must_remain_unchanged"] is True
    assert payload["frozen_v1"]["git_blob_sha1"] == EXPECTED_V1_GIT_BLOB_SHA1
    assert _git_blob_sha1(FROZEN_V1) == EXPECTED_V1_GIT_BLOB_SHA1
    assert payload["compatibility"]["historical_v1_artifacts_modified"] is False
    assert payload["compatibility"]["v1_and_v2_results_interchangeable"] is False


def test_spd_v2_amendment_is_prospective_and_target_closed() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    access = payload["access_boundary"]
    prospective = payload["prospective_v2"]

    assert access["calibration_outcomes_opened_for_this_amendment"] is False
    assert access["confirmation_outcomes_opened_for_this_amendment"] is False
    assert access["target_outcomes_used_for_numerical_selection"] is False
    assert prospective["status"] == "prospective_unfrozen"
    assert prospective["target_use_authorized"] is False
    assert prospective["implementation_version"] == 2
    assert prospective["numerical_backend_version"] == 1
