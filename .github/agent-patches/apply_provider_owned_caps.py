from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] if ".github" in Path(__file__).parts else Path.cwd()


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if addition.strip() in text:
        raise RuntimeError(f"addition already present in {path}")
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(f"expected one marker in {path}, found {count}: {marker[:80]!r}")
    target.write_text(text.replace(marker, marker + addition, 1), encoding="utf-8")


# Add explicit dependence-weight semantics to the batch contract.
replace_once(
    "src/bayesian_phystwin/_gauge_aware_contracts.py",
    "    return result\n\n\n@dataclass(frozen=True)\nclass GaugeAwareBeliefConfig:",
    """    return result


COMPOSITE_WEIGHT_MODE_CONSUMER_CAP = "consumer-effective-sample-cap-v1"
COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL = "provider-final-per-row-v1"
_COMPOSITE_WEIGHT_MODES = frozenset(
    {
        COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
        COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    }
)


def _validated_composite_weight_mode(value: str, name: str) -> str:
    mode = str(value)
    _require(
        mode in _COMPOSITE_WEIGHT_MODES,
        f"{name} must be one of {sorted(_COMPOSITE_WEIGHT_MODES)}",
    )
    return mode


@dataclass(frozen=True)
class GaugeAwareBeliefConfig:""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_contracts.py",
    "    metadata: Mapping[str, Any] | None = None\n\n    def __post_init__(self) -> None:",
    """    metadata: Mapping[str, Any] | None = None
    composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP
    anchor_composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP

    def __post_init__(self) -> None:""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_contracts.py",
    """                _regularized_precision(
                    anchor_bias_prior,
                    "anchor bias prior covariance",
                    eigenvalue_floor=1e-12,
                )

        for name, value in (
""",
    """                _regularized_precision(
                    anchor_bias_prior,
                    "anchor bias prior covariance",
                    eigenvalue_floor=1e-12,
                )

        composite_weight_mode = _validated_composite_weight_mode(
            self.composite_weight_mode,
            "composite_weight_mode",
        )
        anchor_composite_weight_mode = _validated_composite_weight_mode(
            self.anchor_composite_weight_mode,
            "anchor_composite_weight_mode",
        )

        for name, value in (
""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_contracts.py",
    "        object.__setattr__(self, \"metadata\", _validated_metadata(self.metadata))\n",
    """        object.__setattr__(self, "composite_weight_mode", composite_weight_mode)
        object.__setattr__(
            self,
            "anchor_composite_weight_mode",
            anchor_composite_weight_mode,
        )
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))
""",
)

# Re-export the modes on the stable gauge-aware surface.
replace_once(
    "src/bayesian_phystwin/gauge_aware_belief.py",
    "from ._gauge_aware_contracts import (\n    GaugeAwareBeliefConfig,",
    """from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefConfig,""",
)
replace_once(
    "src/bayesian_phystwin/gauge_aware_belief.py",
    "__all__ = [\n    \"GaugeAwareBeliefConfig\",",
    """__all__ = [
    "COMPOSITE_WEIGHT_MODE_CONSUMER_CAP",
    "COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL",
    "GaugeAwareBeliefConfig",""",
)

# Respect provider-final per-row powers in the standard solver.
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    "from ._gauge_aware_contracts import (\n    GaugeAwareBeliefConfig,",
    """from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefConfig,""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    """    composite_weight: np.ndarray,
    effective_samples_per_group: float,
) -> tuple[np.ndarray, dict[str, int]]:
""",
    """    composite_weight: np.ndarray,
    effective_samples_per_group: float,
    *,
    composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
) -> tuple[np.ndarray, dict[str, int]]:
""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    "        group_scale = min(effective_samples_per_group, float(count)) / count\n",
    """        group_scale = (
            1.0
            if composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
            else min(effective_samples_per_group, float(count)) / count
        )
""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    """        batch.composite_weight,
        cfg.effective_samples_per_correlation_group,
    )
""",
    """        batch.composite_weight,
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    """            batch.anchor_composite_weight,
            cfg.effective_samples_per_anchor_correlation_group,
        )
""",
    """            batch.anchor_composite_weight,
            cfg.effective_samples_per_anchor_correlation_group,
            composite_weight_mode=batch.anchor_composite_weight_mode,
        )
""",
)
replace_once(
    "src/bayesian_phystwin/_gauge_aware_solver.py",
    """        "correlation_treatment": (
            "separate effective-sample caps for observation and anchor groups"
        ),
""",
    """        "observation_composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        "correlation_treatment": (
            "provider-final per-row observation power; no consumer recap"
            if batch.composite_weight_mode
            == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
            else "consumer effective-sample cap after composite weighting"
        ),
""",
)

# Apply the same semantics in the prior-aware grouped-mixture solver.
replace_once(
    "src/bayesian_phystwin/_prior_aware_gauge_math.py",
    "from ._gauge_aware_contracts import (\n    GaugeAwareObservationBatch,",
    """from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,""",
)
replace_once(
    "src/bayesian_phystwin/_prior_aware_gauge_math.py",
    """    composite: np.ndarray,
    cap: float,
) -> tuple[
""",
    """    composite: np.ndarray,
    cap: float,
    *,
    composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
) -> tuple[
""",
)
replace_once(
    "src/bayesian_phystwin/_prior_aware_gauge_math.py",
    """            group_power[position] = (
                float(composite[selected[0]])
                * min(cap, float(len(active)))
                / len(active)
            )
""",
    """            consumer_scale = (
                1.0
                if composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
                else min(cap, float(len(active))) / len(active)
            )
            group_power[position] = float(composite[selected[0]]) * consumer_scale
""",
)
replace_once(
    "src/bayesian_phystwin/prior_aware_gauge_belief.py",
    """        observation_composite,
        cfg.effective_samples_per_correlation_group,
    )
""",
    """        observation_composite,
        cfg.effective_samples_per_correlation_group,
        composite_weight_mode=batch.composite_weight_mode,
    )
""",
)
replace_once(
    "src/bayesian_phystwin/prior_aware_gauge_belief.py",
    """            anchor_composite,
            cfg.effective_samples_per_anchor_correlation_group,
        )
""",
    """            anchor_composite,
            cfg.effective_samples_per_anchor_correlation_group,
            composite_weight_mode=batch.anchor_composite_weight_mode,
        )
""",
)
replace_once(
    "src/bayesian_phystwin/prior_aware_gauge_belief.py",
    """        "group_composite_weight_semantics": "generalized-Bayes likelihood power",
        **basis_diagnostics,
""",
    """        "group_composite_weight_semantics": "generalized-Bayes likelihood power",
        "observation_composite_weight_mode": batch.composite_weight_mode,
        "anchor_composite_weight_mode": batch.anchor_composite_weight_mode,
        **basis_diagnostics,
""",
)

# Detect provider-owned final group weights in the ObservationBelief adapter.
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    "from .gauge_aware_belief import GaugeAwareObservationBatch\n",
    """from .gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
""",
)
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    """def global_translation_bias_jacobian(
""",
    """PROB4D_FINAL_COMPOSITE_WEIGHT_SEMANTICS = (
    "final-per-row-effective-sample-cap-v1"
)


def _observation_composite_weight_mode(
    belief: ObservationBeliefV1,
) -> tuple[str, str]:
    semantics = belief.metadata.get("group_composite_weight_semantics")
    if semantics == PROB4D_FINAL_COMPOSITE_WEIGHT_SEMANTICS:
        return COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL, "artifact-metadata"
    if belief.source_repository == "FlorianPfaff/Prob4D":
        if semantics is not None:
            raise ValueError(
                "unsupported Prob4D group_composite_weight_semantics "
                f"{semantics!r}"
            )
        if "effective_samples_per_group" in belief.metadata:
            return (
                COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
                "legacy-prob4d-export-metadata",
            )
    return COMPOSITE_WEIGHT_MODE_CONSUMER_CAP, "consumer-default"


def global_translation_bias_jacobian(
""",
)
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    """    correlation_groups = tuple(
        f"{belief.stream_id}:correlation-group-{int(group_id)}"
        for group_id in belief.correlation_group_ids
    )
    metadata: dict[str, object] = {
""",
    """    correlation_groups = tuple(
        f"{belief.stream_id}:correlation-group-{int(group_id)}"
        for group_id in belief.correlation_group_ids
    )
    composite_weight_mode, composite_weight_mode_source = (
        _observation_composite_weight_mode(belief)
    )
    metadata: dict[str, object] = {
""",
)
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    """        "low_rank_covariance_double_counted": False,
        "causal_frame_stop_convention": "exclusive",
""",
    """        "low_rank_covariance_double_counted": False,
        "composite_weight_mode": composite_weight_mode,
        "composite_weight_mode_source": composite_weight_mode_source,
        "causal_frame_stop_convention": "exclusive",
""",
)
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    """        composite_weight=row_composite_weight,
        physical_response_scale_m=physical_response_scale_m,
""",
    """        composite_weight=row_composite_weight,
        physical_response_scale_m=physical_response_scale_m,
        composite_weight_mode=composite_weight_mode,
""",
)
replace_once(
    "src/bayesian_phystwin/observation_belief_gauge_adapter.py",
    """            "causal_frame_stop_convention": "exclusive",
            "prob4d_causal_lineage_validated": self.batch.metadata.get(
""",
    """            "causal_frame_stop_convention": "exclusive",
            "composite_weight_mode": self.batch.composite_weight_mode,
            "prob4d_causal_lineage_validated": self.batch.metadata.get(
""",
)

# Strengthen the existing information-mass test suite.
replace_once(
    "tests/test_information_mass_invariants.py",
    """from bayesian_phystwin._gauge_aware_solver import _correlation_group_weights
""",
    """from bayesian_phystwin._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
)
from bayesian_phystwin._gauge_aware_solver import _correlation_group_weights
from bayesian_phystwin._prior_aware_gauge_math import _group_layout
""",
)
append_once(
    "tests/test_information_mass_invariants.py",
    "    assert actual[1] == 0.0\n",
    """


def test_provider_final_per_row_power_is_not_capped_twice() -> None:
    effective = 64.0

    def standard_mass(row_count: int) -> float:
        weights, _ = _correlation_group_weights(
            tuple("provider-group" for _ in range(row_count)),
            np.ones(row_count),
            np.ones(row_count),
            np.full(row_count, effective / row_count),
            effective,
            composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        )
        return float(np.sum(weights))

    assert standard_mass(128) == pytest.approx(effective)
    assert standard_mass(1_024) == pytest.approx(effective)


def test_prior_aware_provider_power_is_duplication_invariant() -> None:
    effective = 32.0

    def layout(row_count: int) -> tuple[np.ndarray, np.ndarray]:
        _, _, base, _, group_power = _group_layout(
            tuple("provider-group" for _ in range(row_count)),
            np.ones(row_count),
            np.ones(row_count),
            np.full(row_count, effective / row_count),
            cap=effective,
            composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
        )
        return base, group_power

    base_small, power_small = layout(64)
    base_large, power_large = layout(640)
    assert float(np.sum(base_small)) == pytest.approx(effective)
    assert float(np.sum(base_large)) == pytest.approx(effective)
    assert power_small[0] == pytest.approx(effective / 64)
    assert power_large[0] == pytest.approx(effective / 640)
""",
)

# Extend the adapter fixture to exercise explicit and legacy Prob4D semantics.
replace_once(
    "tests/test_observation_belief_gauge_adapter.py",
    """from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
""",
    """from bayesian_phystwin.gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefConfig,
""",
)
replace_once(
    "tests/test_observation_belief_gauge_adapter.py",
    """def _belief(
    *,
    association_probability: np.ndarray | None = None,
) -> ObservationBeliefV1:
""",
    """def _belief(
    *,
    association_probability: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> ObservationBeliefV1:
""",
)
replace_once(
    "tests/test_observation_belief_gauge_adapter.py",
    """        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.25]),
    )
""",
    """        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.25]),
        metadata={} if metadata is None else metadata,
    )
""",
)
append_once(
    "tests/test_observation_belief_gauge_adapter.py",
    """    assert result.reason == "no-identifiable-query-state"
""",
    """


def test_adapter_respects_explicit_prob4d_final_group_power() -> None:
    adapted = _adapt(
        _belief(
            metadata={
                "group_composite_weight_semantics": (
                    "final-per-row-effective-sample-cap-v1"
                ),
                "effective_samples_per_group": 64.0,
            }
        )
    )

    assert (
        adapted.batch.composite_weight_mode
        == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    )
    assert adapted.batch.metadata["composite_weight_mode_source"] == (
        "artifact-metadata"
    )
    assert adapted.summary()["composite_weight_mode"] == (
        COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    )


def test_adapter_recognizes_legacy_prob4d_effective_sample_metadata() -> None:
    adapted = _adapt(
        _belief(metadata={"effective_samples_per_group": 64.0})
    )

    assert (
        adapted.batch.composite_weight_mode
        == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    )
    assert adapted.batch.metadata["composite_weight_mode_source"] == (
        "legacy-prob4d-export-metadata"
    )
""",
)

# Document ownership of the cap.
replace_once(
    "docs/gauge_aware_observation_update.md",
    """The adapter maps every shared low-rank covariance factor to an explicit
standard-normal nuisance coefficient. It does not add the same factor to the
conditional point covariance. Association probability remains available in
""",
    """The adapter maps every shared low-rank covariance factor to an explicit
standard-normal nuisance coefficient. It does not add the same factor to the
conditional point covariance. When a provider declares
`group_composite_weight_semantics = final-per-row-effective-sample-cap-v1`, that
weight is treated as the final generalized-Bayes row power: Bayesian-PhysTwin
does not apply a second effective-sample cap. Batches from providers without
that declaration retain the consumer-side cap for backward compatibility.
Association probability remains available in
""",
)

# Remove the temporary automation files from the resulting branch commit.
for relative in (
    ".github/agent-patches/apply_provider_owned_caps.py",
    ".github/workflows/agent-apply-provider-owned-caps.yml",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()
