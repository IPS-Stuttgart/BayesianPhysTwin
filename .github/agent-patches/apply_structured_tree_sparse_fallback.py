from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
SPARSE = ROOT / "src/bayesian_phystwin/sparse_prior_aware_gauge_belief.py"
TESTS = ROOT / "tests/test_tree_sparse_explicit_gauge_prob4d.py"
MANIFEST = ROOT / "MANIFEST.in"
CHANGELOG = ROOT / "CHANGELOG.md"
TREE_DOC = ROOT / "docs/claim-bearing-tree-sparse-prob4d.md"


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{name}: expected one exact replacement target")
    return text.replace(old, new)


sparse = SPARSE.read_text(encoding="utf-8")
sparse = replace_once(
    sparse,
    "from collections.abc import Mapping\nfrom dataclasses import dataclass\n",
    "from collections.abc import Mapping\nfrom contextvars import ContextVar\nfrom dataclasses import dataclass\n",
    name="sparse imports",
)
sparse = replace_once(
    sparse,
    "from ._prior_aware_gauge_math import (\n",
    "from .structured_gauge_aware_result import (\n"
    "    DENSE_COVARIANCE_REPRESENTATION,\n"
    "    DenseCovarianceV1,\n"
    "    PrecisionBackedCovarianceV1,\n"
    "    StructuredGaugeAwareBeliefResultV1,\n"
    ")\n"
    "from ._prior_aware_gauge_math import (\n",
    name="structured imports",
)
sparse = replace_once(
    sparse,
    "SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION = 1\n",
    "SPARSE_PRIOR_AWARE_GAUGE_SOLVER_VERSION = 1\n"
    "_STRUCTURED_RESULT_MODE: ContextVar[bool] = ContextVar(\n"
    "    \"bayesian_phystwin_sparse_structured_result_mode\",\n"
    "    default=False,\n"
    ")\n",
    name="structured context",
)

lazy_pattern = re.compile(
    r"\nclass _LazyPriorCovariance:\n.*?\n\n@dataclass\(frozen=True, slots=True\)\nclass _NuisanceLayout:",
    flags=re.DOTALL,
)
sparse, count = lazy_pattern.subn(
    "\n@dataclass(frozen=True, slots=True)\nclass _NuisanceLayout:",
    sparse,
    count=1,
)
if count != 1:
    raise RuntimeError("lazy prior covariance class target changed")
sparse = sparse.replace("_LazyPriorCovariance", "PrecisionBackedCovarianceV1")

fallback_pattern = re.compile(
    r"def _sparse_fallback_result\(.*?\n\n\ndef _prior_covariances\(",
    flags=re.DOTALL,
)
new_fallback = '''def _sparse_fallback_result(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    reason: str,
    diagnostics: Mapping[str, Any],
    *,
    prior_covariance: PrecisionBackedCovarianceV1,
) -> GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1:
    state_count = batch.state_jacobian.shape[2]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    anchor_bias_count = (
        0 if batch.anchor_bias_jacobian is None else batch.anchor_bias_jacobian.shape[2]
    )
    anchor_count = (
        0 if batch.anchor_innovation_m is None else len(batch.anchor_innovation_m)
    )
    structured = _STRUCTURED_RESULT_MODE.get()
    result_diagnostics = dict(diagnostics)
    result_diagnostics.update(
        {
            "result_covariance_representation": (
                prior_covariance.representation
                if structured
                else DENSE_COVARIANCE_REPRESENTATION
            ),
            "result_dense_covariance_materialized": not structured,
            "result_estimated_dense_covariance_bytes": (
                prior_covariance.estimated_dense_bytes
            ),
            "result_stored_covariance_bytes_before_materialization": (
                prior_covariance.stored_nbytes
            ),
        }
    )
    common = {
        "inference_admissible": False,
        "reason": reason,
        "state_coefficients": np.zeros(state_count),
        "gauge_delta": np.zeros(gauge.gauge_parameter_count),
        "shared_bias_coefficients": np.zeros(shared_count),
        "view_bias_coefficients": np.zeros(view_count),
        "anchor_bias_coefficients": np.zeros(anchor_bias_count),
        "identifiable_state_transform": np.zeros((state_count, 0)),
        "identifiable_fractions": np.zeros(0),
        "query_sensitivity_fractions": np.zeros(0),
        "robust_weights": np.zeros(len(batch.innovation_m)),
        "anchor_robust_weights": np.zeros(anchor_count),
        "diagnostics": result_diagnostics,
        "input_lineage": batch.metadata or {},
    }
    if structured:
        return StructuredGaugeAwareBeliefResultV1(
            covariance=prior_covariance,
            **common,
        )
    return GaugeAwareBeliefResult(
        posterior_covariance=prior_covariance.materialize(),
        **common,
    )


def _prior_covariances('''
sparse, count = fallback_pattern.subn(new_fallback, sparse, count=1)
if count != 1:
    raise RuntimeError("sparse fallback function target changed")

sparse = replace_once(
    sparse,
    "def update_sparse_prior_aware_gauge_belief(\n",
    "def _update_sparse_prior_aware_gauge_belief_impl(\n",
    name="sparse implementation rename",
)
sparse = replace_once(
    sparse,
    ") -> GaugeAwareBeliefResult:\n    \"\"\"Infer a prior-aware state without materializing a dense gauge design.\"\"\"\n",
    ") -> GaugeAwareBeliefResult | StructuredGaugeAwareBeliefResultV1:\n"
    "    \"\"\"Infer a prior-aware state without materializing a dense gauge design.\"\"\"\n",
    name="sparse implementation return annotation",
)

accepted_old = '''    return GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=state_coefficients,
        gauge_delta=nuisance[gauge_slice],
        shared_bias_coefficients=nuisance[shared_slice],
        view_bias_coefficients=nuisance[view_slice],
        anchor_bias_coefficients=nuisance[anchor_bias_slice],
        posterior_covariance=covariance,
        identifiable_state_transform=state_mapping,
        identifiable_fractions=identifiable,
        query_sensitivity_fractions=query_fraction,
        robust_weights=ordinary_robust,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
        input_lineage={} if batch.metadata is None else batch.metadata,
    )
'''
accepted_new = '''    diagnostics.update(
        {
            "result_covariance_representation": DENSE_COVARIANCE_REPRESENTATION,
            "result_dense_covariance_materialized": True,
            "result_estimated_dense_covariance_bytes": int(covariance.nbytes),
            "result_stored_covariance_bytes_before_materialization": int(
                covariance.nbytes
            ),
        }
    )
    legacy_result = GaugeAwareBeliefResult(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=state_coefficients,
        gauge_delta=nuisance[gauge_slice],
        shared_bias_coefficients=nuisance[shared_slice],
        view_bias_coefficients=nuisance[view_slice],
        anchor_bias_coefficients=nuisance[anchor_bias_slice],
        posterior_covariance=covariance,
        identifiable_state_transform=state_mapping,
        identifiable_fractions=identifiable,
        query_sensitivity_fractions=query_fraction,
        robust_weights=ordinary_robust,
        anchor_robust_weights=anchor_robust,
        diagnostics=diagnostics,
        input_lineage={} if batch.metadata is None else batch.metadata,
    )
    if _STRUCTURED_RESULT_MODE.get():
        return StructuredGaugeAwareBeliefResultV1.from_legacy(legacy_result)
    return legacy_result
'''
sparse = replace_once(
    sparse,
    accepted_old,
    accepted_new,
    name="accepted result construction",
)

wrapper = '''def update_sparse_prior_aware_gauge_belief(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> GaugeAwareBeliefResult:
    """Return the historical complete dense-covariance result."""

    token = _STRUCTURED_RESULT_MODE.set(False)
    try:
        result = _update_sparse_prior_aware_gauge_belief_impl(
            batch,
            gauge,
            config=config,
        )
    finally:
        _STRUCTURED_RESULT_MODE.reset(token)
    if not isinstance(result, GaugeAwareBeliefResult):
        raise RuntimeError("dense sparse-solver mode returned a structured result")
    return result


def update_sparse_prior_aware_gauge_belief_structured(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    *,
    config: PriorAwareGaugeConfigV1 | None = None,
) -> StructuredGaugeAwareBeliefResultV1:
    """Return a structured result and avoid dense covariance on rejection."""

    token = _STRUCTURED_RESULT_MODE.set(True)
    try:
        result = _update_sparse_prior_aware_gauge_belief_impl(
            batch,
            gauge,
            config=config,
        )
    finally:
        _STRUCTURED_RESULT_MODE.reset(token)
    if not isinstance(result, StructuredGaugeAwareBeliefResultV1):
        raise RuntimeError("structured sparse-solver mode returned a legacy result")
    return result


'''
sparse = replace_once(
    sparse,
    "__all__ = [\n",
    wrapper + "__all__ = [\n",
    name="public structured wrapper insertion",
)
sparse = replace_once(
    sparse,
    '    "update_sparse_prior_aware_gauge_belief",\n',
    '    "update_sparse_prior_aware_gauge_belief",\n'
    '    "update_sparse_prior_aware_gauge_belief_structured",\n',
    name="structured solver export",
)
SPARSE.write_text(sparse, encoding="utf-8")

append_tests = r'''


def test_structured_rejection_does_not_materialize_dense_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import (
        PRECISION_BACKED_COVARIANCE_REPRESENTATION,
        PrecisionBackedCovarianceV1,
    )

    adapted = _build(_validated_observation())
    unsupported = replace(adapted.batch, prior_reliability=np.zeros(4))

    def fail_materialization(self, *, maximum_bytes=None):
        raise AssertionError("structured rejection materialized covariance")

    monkeypatch.setattr(
        PrecisionBackedCovarianceV1,
        "materialize",
        fail_materialization,
    )
    result = update_sparse_prior_aware_gauge_belief_structured(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )

    assert not result.inference_admissible
    assert result.covariance_representation == (
        PRECISION_BACKED_COVARIANCE_REPRESENTATION
    )
    assert result.dense_covariance_materialized is False
    assert result.diagnostics["result_dense_covariance_materialized"] is False
    assert len(result.result_id) == 64


def test_structured_rejection_materializes_only_through_explicit_conversion() -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )

    adapted = _build(_validated_observation())
    unsupported = replace(adapted.batch, prior_reliability=np.zeros(4))
    structured = update_sparse_prior_aware_gauge_belief_structured(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        unsupported,
        adapted.tree_gauge_design,
        config=_config(),
    )

    with pytest.raises(MemoryError, match="exceeding"):
        structured.materialize_posterior_covariance(
            maximum_bytes=structured.estimated_dense_covariance_bytes - 1
        )
    converted = structured.to_legacy()
    np.testing.assert_allclose(
        converted.posterior_covariance,
        legacy.posterior_covariance,
        atol=1e-12,
        rtol=1e-12,
    )
    assert converted.reason == legacy.reason


def test_structured_acceptance_is_numerically_identical_to_legacy() -> None:
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import DenseCovarianceV1

    adapted = _build(_validated_observation())
    structured = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )

    assert structured.inference_admissible
    assert isinstance(structured.covariance, DenseCovarianceV1)
    assert structured.dense_covariance_materialized is True
    converted = structured.to_legacy()
    for name in (
        "state_coefficients",
        "gauge_delta",
        "posterior_covariance",
        "robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(converted, name),
            getattr(legacy, name),
            atol=1e-12,
            rtol=1e-12,
        )


def test_structured_and_legacy_solver_modes_are_context_local() -> None:
    from bayesian_phystwin._gauge_aware_contracts import GaugeAwareBeliefResult
    from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
        update_sparse_prior_aware_gauge_belief,
        update_sparse_prior_aware_gauge_belief_structured,
    )
    from bayesian_phystwin.structured_gauge_aware_result import (
        StructuredGaugeAwareBeliefResultV1,
    )

    adapted = _build(_validated_observation())
    structured = update_sparse_prior_aware_gauge_belief_structured(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    legacy = update_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=_config(),
    )
    assert isinstance(structured, StructuredGaugeAwareBeliefResultV1)
    assert isinstance(legacy, GaugeAwareBeliefResult)


def test_claim_bearing_structured_rejection_binds_without_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesian_phystwin.structured_gauge_aware_result import (
        PrecisionBackedCovarianceV1,
    )
    from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
        update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts,
    )

    def fail_materialization(self, *, maximum_bytes=None):
        raise AssertionError("claim identity materialized covariance")

    monkeypatch.setattr(
        PrecisionBackedCovarianceV1,
        "materialize",
        fail_materialization,
    )
    update = update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        state_prior_covariance_m2=np.zeros((2, 2), dtype=np.float64),
        config=_config(),
    )

    assert not update.inference_admissible
    assert update.dense_covariance_materialized is False
    assert len(update.admission_id) == 64
    assert len(update.structured_result_id) == 64
    assert len(update.update_id) == 64
    assert update.descriptor()["structured_result_id"] == update.structured_result_id


def test_claim_bearing_structured_conversion_is_explicit_and_budgeted() -> None:
    from bayesian_phystwin.tree_sparse_structured_gauge_prob4d import (
        update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts,
    )

    update = update_claim_bearing_tree_sparse_prob4d_structured_from_artifacts(
        _validated_observation(),
        _linearization(),
        physical_prediction_xyz_m=np.zeros((4, 3), dtype=np.float64),
        state_prior_covariance_m2=np.zeros((2, 2), dtype=np.float64),
        config=_config(),
    )
    with pytest.raises(MemoryError, match="exceeding"):
        update.to_legacy(maximum_covariance_bytes=1)
    legacy = update.to_legacy()
    assert legacy.result.reason == update.result.reason
    assert legacy.observation_artifact_id == update.observation_artifact_id
'''

tests = TESTS.read_text(encoding="utf-8")
if "test_structured_rejection_does_not_materialize_dense_covariance" in tests:
    raise RuntimeError("structured tree-sparse tests already exist")
TESTS.write_text(tests.rstrip() + append_tests + "\n", encoding="utf-8")

manifest = MANIFEST.read_text(encoding="utf-8")
manifest_line = "include docs/structured_tree_sparse_fallback.md"
if manifest_line not in manifest.splitlines():
    manifest = manifest.rstrip() + "\n" + manifest_line + "\n"
MANIFEST.write_text(manifest, encoding="utf-8")

changelog = CHANGELOG.read_text(encoding="utf-8")
changelog_anchor = "### Added\n\n"
changelog_entry = (
    "- An additive structured gauge-aware result and claim-bearing tree-sparse "
    "Prob4D update that preserve rejected prior uncertainty in precision form, "
    "report materialization cost, and require explicit budgeted conversion to "
    "the historical dense result.\n"
)
if changelog_entry not in changelog:
    changelog = replace_once(
        changelog,
        changelog_anchor,
        changelog_anchor + changelog_entry,
        name="changelog insertion",
    )
CHANGELOG.write_text(changelog, encoding="utf-8")

if TREE_DOC.exists():
    tree_doc = TREE_DOC.read_text(encoding="utf-8")
    note = (
        "\n## Structured rejection result\n\n"
        "New claim-bearing runs that may reject large gauge trees should use "
        "[`structured_tree_sparse_fallback.md`](structured_tree_sparse_fallback.md). "
        "The historical dense result remains available only through explicit "
        "compatibility conversion.\n"
    )
    if "## Structured rejection result" not in tree_doc:
        TREE_DOC.write_text(tree_doc.rstrip() + note, encoding="utf-8")
