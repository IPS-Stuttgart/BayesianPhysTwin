from __future__ import annotations

from pathlib import Path


ROOT = Path(".")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one target, found {count}")
    return text.replace(old, new, 1)


def replace_count(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
    expected: int,
) -> str:
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{label}: expected {expected} targets, found {count}")
    return text.replace(old, new)


def patch_contracts() -> None:
    path = ROOT / "src/bayesian_phystwin/_gauge_aware_contracts.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    anchor_composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP

    def __post_init__(self) -> None:
''',
        '''    anchor_composite_weight_mode: str = COMPOSITE_WEIGHT_MODE_CONSUMER_CAP
    association_probability: np.ndarray | None = None

    def __post_init__(self) -> None:
''',
        label="association field",
    )
    text = replace_once(
        text,
        '''        reliability = _probability_vector(
            self.prior_reliability,
            count,
            name="prior_reliability",
            default=1.0,
        )
        nominal_probability = _probability_vector(
''',
        '''        reliability = _probability_vector(
            self.prior_reliability,
            count,
            name="prior_reliability",
            default=1.0,
        )
        association_probability = _probability_vector(
            self.association_probability,
            count,
            name="association_probability",
            default=1.0,
        )
        nominal_probability = _probability_vector(
''',
        label="association validation",
    )
    text = replace_once(
        text,
        '''            ("prior_reliability", reliability),
            ("prior_nominal_probability", nominal_probability),
''',
        '''            ("prior_reliability", reliability),
            ("association_probability", association_probability),
            ("prior_nominal_probability", nominal_probability),
''',
        label="association ownership",
    )
    path.write_text(text, encoding="utf-8")


def patch_dense_solver() -> None:
    path = ROOT / "src/bayesian_phystwin/prior_aware_gauge_belief.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    (
        observation_groups,
        observation_indices,
''',
        '''    association_probability = np.asarray(batch.association_probability)
    observation_row_weight = batch.prior_reliability * association_probability
    (
        observation_groups,
        observation_indices,
''',
        label="dense row power definition",
    )
    text = replace_once(
        text,
        '''        batch.correlation_group_ids,
        batch.prior_reliability,
        observation_nominal,
''',
        '''        batch.correlation_group_ids,
        observation_row_weight,
        observation_nominal,
''',
        label="dense group row power",
    )
    text = replace_count(
        text,
        "selected[batch.prior_reliability[selected] > 0.0]",
        "selected[observation_row_weight[selected] > 0.0]",
        label="dense active rows",
        expected=2,
    )
    text = replace_count(
        text,
        "batch.prior_reliability[active]",
        "observation_row_weight[active]",
        label="dense weighted residual and score",
        expected=2,
    )
    text = replace_once(
        text,
        '''        "association_probability_used_as_reliability": False,
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
''',
        '''        "association_probability_used_as_reliability": False,
        "association_probability_used_as_row_power": True,
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
        "row_association_semantics": "generalized-Bayes-row-power-v1",
''',
        label="dense association diagnostics",
    )
    path.write_text(text, encoding="utf-8")


def patch_sparse_solver() -> None:
    path = ROOT / "src/bayesian_phystwin/sparse_prior_aware_gauge_belief.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    (
        observation_groups,
        observation_indices,
''',
        '''    association_probability = np.asarray(batch.association_probability)
    observation_row_weight = batch.prior_reliability * association_probability
    (
        observation_groups,
        observation_indices,
''',
        label="sparse row power definition",
    )
    text = replace_once(
        text,
        '''        batch.correlation_group_ids,
        batch.prior_reliability,
        observation_nominal,
''',
        '''        batch.correlation_group_ids,
        observation_row_weight,
        observation_nominal,
''',
        label="sparse group row power",
    )
    text = replace_count(
        text,
        "selected[batch.prior_reliability[selected] > 0.0]",
        "selected[observation_row_weight[selected] > 0.0]",
        label="sparse active rows",
        expected=2,
    )
    text = replace_once(
        text,
        '''                    batch.prior_reliability[active]
                    * np.sum(np.square(white_residual[active]), axis=1)
''',
        '''                    observation_row_weight[active]
                    * np.sum(np.square(white_residual[active]), axis=1)
''',
        label="sparse weighted residual",
    )
    text = replace_once(
        text,
        '''            active,
            batch.prior_reliability,
            state_reduced_white,
''',
        '''            active,
            observation_row_weight,
            state_reduced_white,
''',
        label="sparse score row power",
    )
    text = replace_once(
        text,
        '''        "association_probability_used_as_reliability": False,
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
''',
        '''        "association_probability_used_as_reliability": False,
        "association_probability_used_as_row_power": True,
        "row_reliability_semantics": "conditional-covariance-precision-scaling",
        "row_association_semantics": "generalized-Bayes-row-power-v1",
''',
        label="sparse association diagnostics",
    )
    path.write_text(text, encoding="utf-8")


def patch_observed_information() -> None:
    path = ROOT / "src/bayesian_phystwin/observed_information_covariance.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    (
        observation_groups,
        observation_indices,
''',
        '''    association_probability = np.asarray(batch.association_probability)
    observation_row_weight = batch.prior_reliability * association_probability
    (
        observation_groups,
        observation_indices,
''',
        label="observed-information row power definition",
    )
    text = replace_once(
        text,
        '''        batch.correlation_group_ids,
        batch.prior_reliability,
        np.asarray(batch.prior_nominal_probability),
''',
        '''        batch.correlation_group_ids,
        observation_row_weight,
        np.asarray(batch.prior_nominal_probability),
''',
        label="observed-information group row power",
    )
    text = replace_count(
        text,
        "selected[batch.prior_reliability[selected] > 0.0]",
        "selected[observation_row_weight[selected] > 0.0]",
        label="observed-information active rows",
        expected=2,
    )
    text = replace_count(
        text,
        "batch.prior_reliability[active]",
        "observation_row_weight[active]",
        label="observed-information weighted residual and score",
        expected=2,
    )
    path.write_text(text, encoding="utf-8")


def patch_adapter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    row_power_block = '''    row_power = stack["association"] * stack["composite"]
    if np.any(row_power <= 0.0):
        raise ValueError("association-weighted composite power must be positive")
'''
    if row_power_block in text:
        text = text.replace(row_power_block, "", 1)
    else:
        shorter = '''    row_power = stack["association"] * stack["composite"]
'''
        if shorter not in text:
            raise SystemExit(f"{path}: row-power construction not found")
        text = text.replace(shorter, "", 1)
    text = replace_once(
        text,
        '''        prior_reliability=stack["reliability"],
        prior_nominal_probability=stack["nominal"],
        composite_weight=row_power,
''',
        '''        prior_reliability=stack["reliability"],
        prior_nominal_probability=stack["nominal"],
        composite_weight=stack["composite"],
        association_probability=stack["association"],
''',
        label=f"{path} association channel",
    )
    path.write_text(text, encoding="utf-8")


def patch_tree_test() -> None:
    path = ROOT / "tests/test_tree_sparse_explicit_gauge_prob4d.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    parents, transitions, scales = _tree_arrays()
    dense_design = SparseGaugeDesignV1(
''',
        '''    np.testing.assert_allclose(
        adapted.batch.association_probability,
        np.asarray([0.9, 0.8, 0.85, 0.75]),
    )
    np.testing.assert_allclose(
        adapted.batch.composite_weight,
        np.asarray([0.5, 0.5, 0.4, 0.4]),
    )
    parents, transitions, scales = _tree_arrays()
    dense_design = SparseGaugeDesignV1(
''',
        label="tree adapter association assertions",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_contracts()
    patch_dense_solver()
    patch_sparse_solver()
    patch_observed_information()
    for relative in (
        "src/bayesian_phystwin/explicit_gauge_prob4d.py",
        "src/bayesian_phystwin/sparse_explicit_gauge_prob4d.py",
        "src/bayesian_phystwin/tree_sparse_explicit_gauge_prob4d.py",
    ):
        patch_adapter(ROOT / relative)
    patch_tree_test()


if __name__ == "__main__":
    main()
