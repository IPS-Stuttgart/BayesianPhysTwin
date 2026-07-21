import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DIAGNOSTICS = ROOT / "results" / "sota" / "diagnostics"


def _load_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    expected = unsigned.pop("result_sha256")
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    return payload


def test_static_additional_source_result_is_frozen() -> None:
    result = _load_result(
        DIAGNOSTICS
        / "phystwin_prob4d_bias_guard_additional_v1"
        / "result.json"
    )

    assert result["case_count"] == 11
    assert result["garment_count"] == 6
    assert result["candidate_available_count"] == 11
    assert result["candidate_accepted_count"] == 1
    assert result["accepted_harmful_count"] == 0
    assert result["all_rejections_bit_exact"] is True
    assert result["result_sha256"] == (
        "5a01c24d131219e3897ce4af0081ba1c4ddbba4ac9585e79fe64c89b946872c6"
    )


def test_action_conditioned_source_results_are_frozen() -> None:
    root = DIAGNOSTICS / "phystwin_prob4d_action_guard_source_v1"
    additional = _load_result(root / "additional11_result.json")
    seal = _load_result(root / "exploratory19_prediction_cohort_seal.json")
    exploratory = _load_result(root / "exploratory19_result.json")

    assert additional["candidate_family"] == "action_conditioned"
    assert additional["candidate_accepted_count"] == 1
    assert additional["accepted_harmful_count"] == 0
    assert additional["result_sha256"] == (
        "095f29fa7a835344e353c5917ec1089d372b6feca402caa02b859e524da6d37a"
    )
    assert seal["candidate_family"] == "action_conditioned"
    assert seal["case_count"] == 19
    assert all(not case["candidate_accepted"] for case in seal["cases"])
    assert seal["result_sha256"] == (
        "2891def7d884915cdb64dbc208b56e4f64c5dae7e5de1eea4816556f350a351b"
    )
    assert exploratory["candidate_family"] == "action_conditioned"
    assert exploratory["gates"]["accepted_case_count"] == 0
    assert exploratory["gates"]["all_rejections_bit_exact"] is True
    assert exploratory["decision"] == "reject-prob4d-bias-guard-transfer-family"
    assert exploratory["result_sha256"] == (
        "89859249e6d6dfe68375bcf7d1c7e0d4b68a02c2aa0c6584a9647ce999278966"
    )
