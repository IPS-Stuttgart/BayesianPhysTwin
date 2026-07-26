from __future__ import annotations

from types import SimpleNamespace

import pytest

import bayesian_phystwin.prob4d_causal_lineage as lineage


def _semantic_result(value: str) -> dict[str, object]:
    return {
        "validated": True,
        "covariance_semantics": value,
    }


def test_legacy_stream_version_is_inferred(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda belief: _semantic_result(
            lineage.PROB4D_LEGACY_COVARIANCE_SEMANTICS
        ),
    )

    result = lineage.validate_prob4d_causal_observation_belief(
        SimpleNamespace(metadata={})
    )

    assert result["stream_contract_version"] == 1
    assert result["stream_contract_version_inferred"] is True
    assert result["strict_causal_stream_contract"] is True


def test_explicit_joint_stream_version_is_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda belief: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )

    result = lineage.validate_prob4d_causal_observation_belief(
        SimpleNamespace(
            metadata={"prob4d_causal_stream_contract_version": 2}
        )
    )

    assert result["stream_contract_version"] == 2
    assert result["stream_contract_version_inferred"] is False
    assert result["strict_causal_stream_contract"] is True


def test_joint_stream_rejects_mismatched_explicit_version(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda belief: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )

    with pytest.raises(ValueError, match="disagrees with covariance semantics"):
        lineage.validate_prob4d_causal_observation_belief(
            SimpleNamespace(
                metadata={"prob4d_causal_stream_contract_version": 1}
            )
        )


def test_fixed_lag_is_not_a_strict_stream_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda belief: _semantic_result(
            lineage.PROB4D_FIXED_LAG_GAUGE_MODEL
        ),
    )

    result = lineage.validate_prob4d_causal_observation_belief(
        SimpleNamespace(metadata={})
    )

    assert result["stream_contract_version"] is None
    assert result["stream_contract_version_inferred"] is False
    assert result["strict_causal_stream_contract"] is False


def test_fixed_lag_rejects_an_explicit_strict_version(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda belief: _semantic_result(
            lineage.PROB4D_FIXED_LAG_GAUGE_MODEL
        ),
    )

    with pytest.raises(ValueError, match="fixed-lag covariance cannot declare"):
        lineage.validate_prob4d_causal_observation_belief(
            SimpleNamespace(
                metadata={"prob4d_causal_stream_contract_version": 2}
            )
        )
