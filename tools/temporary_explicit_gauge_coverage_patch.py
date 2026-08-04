from pathlib import Path

source_path = Path("src/bayesian_phystwin/explicit_gauge_prob4d.py")
source = source_path.read_text(encoding="utf-8")
redundant = '''    if stack["causal_frame_stop"] != expected["causal_frame_stop"]:
        raise ValueError(
            "sparse factor stack causal_frame_stop differs from validated factor bundle"
        )
'''
if redundant not in source:
    raise SystemExit("redundant causal-cutoff block was not found")
source_path.write_text(source.replace(redundant, "", 1), encoding="utf-8")

test_path = Path("tests/test_explicit_gauge_prob4d.py")
tests = test_path.read_text(encoding="utf-8")
marker = "def test_rederived_stack_rejects_linearized_identity_drift()"
if marker in tests:
    raise SystemExit("coverage tests are already present")
tests += r'''


def _replace_first_linearized(
    validated: SimpleNamespace,
    **changes: object,
) -> None:
    original_linearize = validated.bundle.linearize
    first_factor_id = validated.bundle.factors[0].factor_id

    def changed_linearize(factor: SimpleNamespace) -> SimpleNamespace:
        linearized = original_linearize(factor)
        if factor.factor_id != first_factor_id:
            return linearized
        record = vars(linearized).copy()
        record.update(changes)
        return SimpleNamespace(**record)

    validated.bundle.linearize = changed_linearize


def test_rederived_stack_rejects_linearized_identity_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(validated, view_id="changed-camera")

    with pytest.raises(ValueError, match="identity field view_id"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_rederived_stack_rejects_linearized_point_identity_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(
        validated,
        point_ids=np.asarray([99], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="changed point identities"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("valid_mask", "error", "match"),
    (
        (np.asarray([1], dtype=np.int64), TypeError, "must contain booleans"),
        (np.asarray([False]), ValueError, "changed validity"),
    ),
)
def test_rederived_stack_rejects_linearized_validity_drift(
    valid_mask: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(validated, valid_mask=valid_mask)

    with pytest.raises(error, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("association_probability", np.asarray([0.5])),
        ("prior_reliability", np.asarray([0.5])),
    ),
)
def test_rederived_stack_rejects_linearized_row_probability_drift(
    field: str,
    value: np.ndarray,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(validated, **{field: value})

    with pytest.raises(ValueError, match=f"changed {field}"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prior_nominal_probability", 0.5),
        ("composite_weight", 0.5),
    ),
)
def test_rederived_stack_rejects_linearized_group_probability_drift(
    field: str,
    value: float,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(validated, **{field: value})

    with pytest.raises(ValueError, match="changed group probabilities"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("world_mean_m", np.zeros((1, 2)), "world_mean_m changed shape"),
        (
            "conditional_world_covariance_m2",
            np.zeros((1, 2, 2)),
            "conditional covariance changed shape",
        ),
        (
            "marginal_world_covariance_m2",
            np.zeros((1, 2, 2)),
            "marginal covariance changed shape",
        ),
        (
            "gauge_jacobian",
            np.zeros((1, 3, 6)),
            "local gauge Jacobian changed shape",
        ),
    ),
)
def test_rederived_stack_rejects_linearized_shape_drift(
    field: str,
    value: np.ndarray,
    match: str,
) -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    _replace_first_linearized(validated, **{field: value})

    with pytest.raises(ValueError, match=match):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_rederived_stack_rejects_unknown_factor_gauge() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    factor = validated.bundle.factors[0]
    factor.window_id = "unknown-window"
    factor.gauge_id = "unknown-window"
    _replace_first_linearized(
        validated,
        window_id="unknown-window",
        gauge_id="unknown-window",
    )

    with pytest.raises(ValueError, match="unknown gauge"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_rederived_stack_rejects_an_all_inactive_bundle() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    for factor in validated.bundle.factors:
        factor.valid_mask[:] = False

    with pytest.raises(ValueError, match="no active observation rows"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_rederived_stack_rejects_selected_row_count_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    for field in (
        "world_mean_m",
        "conditional_world_covariance_m2",
        "marginal_world_covariance_m2",
        "local_gauge_jacobian",
        "gauge_indices",
        "association_probability",
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
        "point_ids",
        "frame_indices",
    ):
        setattr(stack, field, np.asarray(getattr(stack, field))[1:])
    for field in ("view_ids", "factor_ids", "correlation_group_ids"):
        setattr(stack, field, tuple(getattr(stack, field)[1:]))

    with pytest.raises(ValueError, match="active row count"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )


def test_rederived_stack_rejects_envelope_total_count_drift() -> None:
    validated, stack, linearization, physical_prediction = _fixture()
    validated.envelope.observation_count += 1

    with pytest.raises(ValueError, match="envelope observation_count"):
        build_claim_bearing_explicit_gauge_batch(
            validated,
            stack,
            linearization,
            physical_prediction_xyz_m=physical_prediction,
        )
'''
test_path.write_text(tests, encoding="utf-8")
