from __future__ import annotations

from pathlib import Path


TARGET = Path("src/bayesian_phystwin/sparse_prior_aware_gauge_belief.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one replacement target, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    marker = "\n\n@dataclass(frozen=True, slots=True)\nclass _NuisanceLayout:"
    insertion = '''

@dataclass(frozen=True, slots=True)
class TreeSparseGaugeDesignV1:
    """Block-local row design with a causal transition/innovation gauge prior."""

    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    parent_indices: np.ndarray
    transition_matrices: np.ndarray
    innovation_scale_tril: np.ndarray
    gauge_ids: tuple[str, ...]
    prior_id: str

    def __post_init__(self) -> None:
        local = _finite_array(
            self.local_gauge_jacobian,
            "local_gauge_jacobian",
            3,
        )
        _require(
            local.shape[1] == 3 and local.shape[2] >= 1,
            "local_gauge_jacobian must have shape (M, 3, G) with G >= 1",
        )
        raw_indices = np.asarray(self.gauge_indices)
        _require(
            raw_indices.ndim == 1
            and np.issubdtype(raw_indices.dtype, np.integer)
            and raw_indices.dtype.kind != "b",
            "gauge_indices must be an integer vector",
        )
        indices = np.asarray(raw_indices, dtype=np.int64)
        _require(
            indices.shape == (len(local),),
            "gauge_indices must contain one value per observation",
        )
        raw_parents = np.asarray(self.parent_indices)
        _require(
            raw_parents.ndim == 1
            and np.issubdtype(raw_parents.dtype, np.integer)
            and raw_parents.dtype.kind != "b",
            "parent_indices must be an integer vector",
        )
        parents = np.asarray(raw_parents, dtype=np.int64)
        transitions = _finite_array(
            self.transition_matrices,
            "transition_matrices",
            3,
        )
        scales = _finite_array(
            self.innovation_scale_tril,
            "innovation_scale_tril",
            3,
        )
        _require(
            isinstance(self.gauge_ids, tuple)
            and all(isinstance(value, str) for value in self.gauge_ids),
            "gauge_ids must be a tuple of strings",
        )
        gauge_ids = tuple(self.gauge_ids)
        _require(gauge_ids and all(gauge_ids), "gauge_ids must be nonempty")
        _require(
            len(set(gauge_ids)) == len(gauge_ids),
            "gauge_ids must be unique",
        )
        gauge_count = len(gauge_ids)
        block_size = local.shape[2]
        _require(
            parents.shape == (gauge_count,),
            "parent_indices must contain one value per gauge",
        )
        _require(
            transitions.shape == (gauge_count, block_size, block_size),
            "transition_matrices shape changed",
        )
        _require(
            scales.shape == (gauge_count, block_size, block_size),
            "innovation_scale_tril shape changed",
        )
        _require(parents[0] == -1, "the first gauge must be the causal-tree root")
        for index in range(1, gauge_count):
            _require(
                0 <= int(parents[index]) < index,
                "each non-root gauge parent must precede the child",
            )
        _require(
            np.all((indices >= 0) & (indices < gauge_count)),
            "gauge_indices reference an unknown gauge",
        )
        _require(
            np.allclose(scales, np.tril(scales), atol=1e-14, rtol=0.0),
            "innovation_scale_tril must be lower triangular",
        )
        _require(
            np.all(np.diagonal(scales, axis1=1, axis2=2) > 0.0),
            "innovation_scale_tril must have positive diagonal",
        )
        if not isinstance(self.prior_id, str):
            raise TypeError("prior_id must be a string")
        _require(
            len(self.prior_id) == 64
            and all(character in "0123456789abcdef" for character in self.prior_id),
            "prior_id must be a lowercase SHA-256 digest",
        )
        object.__setattr__(self, "local_gauge_jacobian", _readonly(local))
        object.__setattr__(self, "gauge_indices", _readonly(indices, dtype=np.int64))
        object.__setattr__(self, "parent_indices", _readonly(parents, dtype=np.int64))
        object.__setattr__(self, "transition_matrices", _readonly(transitions))
        object.__setattr__(self, "innovation_scale_tril", _readonly(scales))
        object.__setattr__(self, "gauge_ids", gauge_ids)

    @property
    def observation_count(self) -> int:
        return len(self.local_gauge_jacobian)

    @property
    def block_size(self) -> int:
        return self.local_gauge_jacobian.shape[2]

    @property
    def gauge_count(self) -> int:
        return len(self.gauge_ids)

    @property
    def gauge_parameter_count(self) -> int:
        return self.gauge_count * self.block_size

    @property
    def equivalent_dense_design_bytes(self) -> int:
        return (
            self.observation_count
            * 3
            * self.gauge_parameter_count
            * np.dtype(np.float64).itemsize
        )

    @property
    def dense_gauge_prior_avoided_bytes(self) -> int:
        return self.gauge_parameter_count**2 * np.dtype(np.float64).itemsize

    @property
    def tree_factor_storage_nbytes(self) -> int:
        return int(
            self.parent_indices.nbytes
            + self.transition_matrices.nbytes
            + self.innovation_scale_tril.nbytes
        )

    def prior_information_matrix(self) -> np.ndarray:
        """Assemble exact tree precision without constructing prior covariance."""

        dimension = self.gauge_parameter_count
        block_size = self.block_size
        information = np.zeros((dimension, dimension), dtype=np.float64)
        identity = np.eye(block_size, dtype=np.float64)
        for index in range(self.gauge_count):
            inverse_scale = np.linalg.solve(
                self.innovation_scale_tril[index],
                identity,
            )
            innovation_precision = inverse_scale.T @ inverse_scale
            child = slice(index * block_size, (index + 1) * block_size)
            information[child, child] += innovation_precision
            parent_index = int(self.parent_indices[index])
            if parent_index < 0:
                continue
            parent = slice(
                parent_index * block_size,
                (parent_index + 1) * block_size,
            )
            transition = self.transition_matrices[index]
            information[child, parent] -= innovation_precision @ transition
            information[parent, child] -= transition.T @ innovation_precision
            information[parent, parent] += (
                transition.T @ innovation_precision @ transition
            )
        return 0.5 * (information + information.T)


GaugeDesignV1 = SparseGaugeDesignV1 | TreeSparseGaugeDesignV1


class _LazyPriorCovariance:
    """Materialize a dense prior only when a rejected result requires it."""

    def __init__(
        self,
        state_covariance: np.ndarray,
        nuisance_precision: np.ndarray,
        nuisance_covariance: np.ndarray | None,
    ) -> None:
        self._state_covariance = state_covariance
        self._nuisance_precision = nuisance_precision
        self._nuisance_covariance = nuisance_covariance

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        nuisance_covariance = self._nuisance_covariance
        if nuisance_covariance is None:
            identity = np.eye(len(self._nuisance_precision), dtype=np.float64)
            nuisance_covariance = np.linalg.solve(
                self._nuisance_precision,
                identity,
            )
            nuisance_covariance = 0.5 * (
                nuisance_covariance + nuisance_covariance.T
            )
        result = _block_diagonal(
            [self._state_covariance, nuisance_covariance]
        )
        return result if dtype is None else np.asarray(result, dtype=dtype)
'''
    text = replace_once(
        text,
        marker,
        insertion + marker,
        label="tree design insertion",
    )

    text = replace_once(
        text,
        "    gauge: SparseGaugeDesignV1,\n    reason: str,",
        "    gauge: GaugeDesignV1,\n    reason: str,",
        label="fallback gauge annotation",
    )
    text = replace_once(
        text,
        "    prior_covariance: np.ndarray,\n) -> GaugeAwareBeliefResult:",
        "    prior_covariance: np.ndarray | _LazyPriorCovariance,\n) -> GaugeAwareBeliefResult:",
        label="fallback covariance annotation",
    )
    text = replace_once(
        text,
        "        posterior_covariance=prior_covariance,",
        "        posterior_covariance=np.asarray(prior_covariance, dtype=np.float64),",
        label="fallback covariance materialization",
    )

    old_prior = '''def _prior_covariances(
    batch: GaugeAwareObservationBatch,
    gauge: SparseGaugeDesignV1,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_count = batch.state_jacobian.shape[2]
    state = (
        np.eye(state_count) * config.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    nuisance = [np.asarray(gauge.gauge_prior_covariance)]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    if shared_count:
        nuisance.append(np.eye(shared_count) * config.shared_bias_prior_std_m**2)
    if view_count:
        nuisance.append(np.eye(view_count) * config.view_bias_prior_std_m**2)
    if batch.anchor_bias_prior_covariance is not None:
        nuisance.append(np.asarray(batch.anchor_bias_prior_covariance))
    nuisance_covariance = _block_diagonal(nuisance)
    return state, nuisance_covariance, _block_diagonal([state, nuisance_covariance])
'''
    new_prior = '''def _prior_covariances(
    batch: GaugeAwareObservationBatch,
    gauge: GaugeDesignV1,
    config: PriorAwareGaugeConfigV1,
) -> tuple[np.ndarray, np.ndarray, _LazyPriorCovariance]:
    state_count = batch.state_jacobian.shape[2]
    state = (
        np.eye(state_count) * config.state_prior_std_m**2
        if batch.state_prior_covariance_m2 is None
        else np.asarray(batch.state_prior_covariance_m2)
    )
    dense_covariance_blocks: list[np.ndarray] | None
    if isinstance(gauge, TreeSparseGaugeDesignV1):
        gauge_precision = gauge.prior_information_matrix()
        dense_covariance_blocks = None
    else:
        gauge_covariance = np.asarray(gauge.gauge_prior_covariance)
        gauge_precision = _regularized_precision(
            gauge_covariance,
            "gauge prior covariance",
            eigenvalue_floor=config.prior_eigenvalue_floor,
        )
        dense_covariance_blocks = [gauge_covariance]
    precision_blocks = [gauge_precision]
    shared_count = batch.shared_bias_jacobian.shape[2]
    view_count = batch.view_bias_jacobian.shape[2]
    if shared_count:
        shared_covariance = (
            np.eye(shared_count) * config.shared_bias_prior_std_m**2
        )
        precision_blocks.append(
            np.eye(shared_count) / config.shared_bias_prior_std_m**2
        )
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(shared_covariance)
    if view_count:
        view_covariance = np.eye(view_count) * config.view_bias_prior_std_m**2
        precision_blocks.append(
            np.eye(view_count) / config.view_bias_prior_std_m**2
        )
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(view_covariance)
    if batch.anchor_bias_prior_covariance is not None:
        anchor_covariance = np.asarray(batch.anchor_bias_prior_covariance)
        precision_blocks.append(
            _regularized_precision(
                anchor_covariance,
                "anchor bias prior covariance",
                eigenvalue_floor=config.prior_eigenvalue_floor,
            )
        )
        if dense_covariance_blocks is not None:
            dense_covariance_blocks.append(anchor_covariance)
    nuisance_precision = _block_diagonal(precision_blocks)
    nuisance_covariance = (
        None
        if dense_covariance_blocks is None
        else _block_diagonal(dense_covariance_blocks)
    )
    return (
        state,
        nuisance_precision,
        _LazyPriorCovariance(state, nuisance_precision, nuisance_covariance),
    )
'''
    text = replace_once(
        text,
        old_prior,
        new_prior,
        label="precision-form prior construction",
    )

    text = replace_once(
        text,
        "    gauge: SparseGaugeDesignV1,\n) -> tuple[\n",
        "    gauge: GaugeDesignV1,\n) -> tuple[\n",
        label="whitening gauge annotation",
    )
    text = replace_once(
        text,
        "    nuisance_prior: np.ndarray,\n    query: np.ndarray,",
        "    nuisance_prior_precision: np.ndarray,\n    query: np.ndarray,",
        label="basis precision annotation",
    )
    old_basis = '''    nuisance_information = (
        _regularized_precision(
            nuisance_prior,
            "nuisance prior covariance",
            eigenvalue_floor=config.prior_eigenvalue_floor,
        )
        + nuisance_information_from_data
    )
'''
    new_basis = '''    nuisance_information = (
        nuisance_prior_precision + nuisance_information_from_data
    )
'''
    text = replace_once(
        text,
        old_basis,
        new_basis,
        label="basis precision use",
    )

    text = replace_once(
        text,
        "    gauge: SparseGaugeDesignV1,\n    *,\n    config: PriorAwareGaugeConfigV1 | None = None,",
        "    gauge: GaugeDesignV1,\n    *,\n    config: PriorAwareGaugeConfigV1 | None = None,",
        label="solver gauge annotation",
    )
    text = replace_once(
        text,
        '''    if not isinstance(gauge, SparseGaugeDesignV1):
        raise TypeError("gauge must be a SparseGaugeDesignV1")
''',
        '''    if not isinstance(gauge, (SparseGaugeDesignV1, TreeSparseGaugeDesignV1)):
        raise TypeError(
            "gauge must be a SparseGaugeDesignV1 or TreeSparseGaugeDesignV1"
        )
''',
        label="solver gauge type check",
    )
    text = replace_once(
        text,
        "    state_prior, nuisance_prior, full_prior = _prior_covariances(batch, gauge, cfg)",
        "    state_prior, nuisance_prior_precision, full_prior = _prior_covariances(\n        batch, gauge, cfg\n    )",
        label="prior return names",
    )
    text = replace_once(
        text,
        "            nuisance_prior,\n            batch.query_state_jacobian,",
        "            nuisance_prior_precision,\n            batch.query_state_jacobian,",
        label="basis prior argument",
    )
    old_reduced = '''    reduced_prior = _block_diagonal([np.eye(retained), nuisance_prior])
    prior_precision = _regularized_precision(
        reduced_prior,
        "reduced prior covariance",
        eigenvalue_floor=cfg.prior_eigenvalue_floor,
    )
'''
    new_reduced = '''    prior_precision = _block_diagonal(
        [np.eye(retained), nuisance_prior_precision]
    )
'''
    text = replace_once(
        text,
        old_reduced,
        new_reduced,
        label="reduced prior precision",
    )

    old_diagnostics = '''        "dense_gauge_design_avoided_bytes": gauge.equivalent_dense_design_bytes,
        **basis_diagnostics,
'''
    new_diagnostics = '''        "dense_gauge_design_avoided_bytes": gauge.equivalent_dense_design_bytes,
        "gauge_prior_representation": (
            "tree-transition-innovation-information-v1"
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else "dense-covariance-v1"
        ),
        "dense_gauge_prior_covariance_materialized": not isinstance(
            gauge, TreeSparseGaugeDesignV1
        ),
        "dense_gauge_prior_avoided_bytes": (
            gauge.dense_gauge_prior_avoided_bytes
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else 0
        ),
        "tree_factor_storage_nbytes": (
            gauge.tree_factor_storage_nbytes
            if isinstance(gauge, TreeSparseGaugeDesignV1)
            else 0
        ),
        "tree_prior_information_matrix_materialized": isinstance(
            gauge, TreeSparseGaugeDesignV1
        ),
        **basis_diagnostics,
'''
    text = replace_once(
        text,
        old_diagnostics,
        new_diagnostics,
        label="tree prior diagnostics",
    )
    text = replace_once(
        text,
        '''    "SparseGaugeDesignV1",
    "update_sparse_prior_aware_gauge_belief",
''',
        '''    "SparseGaugeDesignV1",
    "TreeSparseGaugeDesignV1",
    "update_sparse_prior_aware_gauge_belief",
''',
        label="public tree design export",
    )

    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
