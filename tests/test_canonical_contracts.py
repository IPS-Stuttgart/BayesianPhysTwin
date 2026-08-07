from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import test_deform360_calibration_observability_case_builder as observability_cases
import test_deform360_calibration_observability_case_builder_adversarial as adversarial_cases

import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2
from bayesian_phystwin._canonical_contracts import (
    canonical_relative_posix_path,
    literal_lower_hex,
)

_strict_v2_cases = importlib.import_module("test_prior_aware_gauge_belief_v2")
test_strict_v2_accepts_dense_and_sparse = (
    _strict_v2_cases.test_dense_and_sparse_v2_admit_converged_positive_curvature
)
test_strict_v2_rejects_nonconvergence = (
    _strict_v2_cases.test_v2_fails_closed_when_fixed_point_does_not_converge
)
test_strict_v2_preserves_rejection = (
    _strict_v2_cases.test_v2_preserves_underlying_rejection_reason
)
test_strict_v2_rejects_ill_conditioning = (
    _strict_v2_cases.test_v2_rejects_ill_conditioned_exact_curvature
)
test_strict_v2_rejects_inconsistent_diagnostics = (
    _strict_v2_cases.test_v2_rejects_inconsistent_curvature_diagnostics
)
test_strict_v2_rejects_nonpositive_curvature = (
    _strict_v2_cases.test_v2_rejects_nonpositive_exact_curvature
)
test_strict_v2_rejects_approximate_objective = (
    _strict_v2_cases.test_v2_rejects_precision_floored_approximate_objective
)


class _StringSubclass(str):
    pass


def test_literal_lower_hex_accepts_only_literal_lowercase_strings() -> None:
    assert literal_lower_hex("a" * 40, name="revision", lengths={40, 64}) == ("a" * 40)
    assert literal_lower_hex("1" * 64, name="digest", lengths={64}) == "1" * 64

    rejected = [
        int("1" * 40),
        b"1" * 40,
        _StringSubclass("1" * 40),
        "A" * 40,
        "1" * 39,
        "1" * 40 + " ",
    ]
    for value in rejected:
        with pytest.raises(ValueError):
            literal_lower_hex(value, name="revision", lengths={40, 64})


def test_literal_lower_hex_rejects_invalid_length_contract() -> None:
    for lengths in (set(), {0}, {-1}, {True}):
        with pytest.raises(ValueError, match="positive integers"):
            literal_lower_hex("a", name="digest", lengths=lengths)


def test_canonical_relative_posix_path_is_portable_and_non_normalizing() -> None:
    value = "raw/object-1/tactile.npy"
    assert canonical_relative_posix_path(value, name="artifact path") == value

    rejected = [
        "",
        b"raw/object",
        "/absolute/path",
        "//server/share",
        "C:/windows/path",
        "raw\\windows",
        "raw/../escape",
        "./raw/object",
        "raw/./object",
        "raw//object",
        "raw/object/",
        "raw/\x00object",
    ]
    for value in rejected:
        with pytest.raises(ValueError, match="POSIX|literal"):
            canonical_relative_posix_path(value, name="artifact path")


def test_strict_v2_rejects_nonfinite_diagnostic_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert strict_v2._finite_diagnostic({"value": np.inf}, "value") is None

    dense_batch, _, _ = _strict_v2_cases._batches()
    underlying = _strict_v2_cases._synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics=_strict_v2_cases._diagnostics(
            minimum_eigenvalue=1.0e-308,
            maximum_eigenvalue=1.0e308,
        ),
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    with np.errstate(over="ignore"):
        result = strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            config=_strict_v2_cases._config(),
        )

    assert not result.inference_admissible
    assert result.reason == "strict-v2-invalid-admission-diagnostics"


def test_strict_v2_rejects_invalid_dense_argument_types() -> None:
    dense_batch, _, _ = _strict_v2_cases._batches()

    with pytest.raises(TypeError, match="batch"):
        strict_v2.update_prior_aware_gauge_belief_v2(object())
    with pytest.raises(TypeError, match="config"):
        strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            config=object(),
        )
    with pytest.raises(TypeError, match="admission_config"):
        strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            admission_config=object(),
        )


def test_strict_v2_rejects_invalid_sparse_argument_types() -> None:
    _, sparse_batch, sparse_design = _strict_v2_cases._batches()

    with pytest.raises(TypeError, match="batch"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            object(),
            sparse_design,
        )
    with pytest.raises(TypeError, match="gauge"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            object(),
        )
    with pytest.raises(TypeError, match="config"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            sparse_design,
            config=object(),
        )
    with pytest.raises(TypeError, match="admission_config"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            sparse_design,
            admission_config=object(),
        )


def test_deform360_observability_case_builder_stable_core_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute every claim-bearing producer control in the coverage ratchet."""

    observability_cases.test_evaluated_builder_binds_exact_lineage_and_round_trips(
        tmp_path / "evaluated"
    )
    observability_cases.test_builder_rejects_confirmation_and_unknown_objects(
        tmp_path / "identities"
    )
    observability_cases.test_evaluated_builder_requires_source_prepared_object(
        tmp_path / "prepared"
    )
    substitutions = (
        ("source_protocol_path", "source-lock summary differs"),
        ("plan_path", "plan summary differs"),
        ("download_path", "download summary differs"),
        ("result_path", "result summary differs"),
    )
    for index, (path_name, message) in enumerate(substitutions):
        observability_cases.test_terminal_record_prevents_artifact_file_substitution(
            tmp_path / f"substitution-{index}",
            path_name,
            message,
        )
    observability_cases.test_candidate_information_loss_is_rejected(
        tmp_path / "information-loss"
    )
    observability_cases.test_symlinked_matrix_is_rejected(tmp_path / "symlink")
    observability_cases.test_pickled_query_payload_is_rejected(tmp_path / "pickle")
    observability_cases.test_empty_contact_anchor_is_rejected(tmp_path / "empty")
    observability_cases.test_failure_builder_retains_source_and_analysis_failures(
        tmp_path / "retained-failures"
    )
    observability_cases.test_unsupported_object_failure_records_no_payload_access(
        tmp_path / "unsupported"
    )
    observability_cases.test_cli_publishes_evaluated_case_without_overwrite(
        tmp_path / "cli"
    )

    adversarial_cases.test_ordinary_file_rejects_missing_directory_and_parent_symlink(
        tmp_path / "ordinary-file"
    )
    adversarial_cases.test_read_ordinary_bytes_reports_read_failure(
        tmp_path / "read-error",
        monkeypatch,
    )
    adversarial_cases.test_matrix_loader_rejects_wrong_suffix_scalar_and_non_numeric(
        tmp_path / "matrix-types",
        monkeypatch,
    )
    invalid_matrices = (
        np.asarray([1.0, 2.0]),
        np.empty((0, 2)),
        np.asarray([[1.0, np.nan]]),
    )
    for index, matrix in enumerate(invalid_matrices):
        adversarial_cases.test_matrix_loader_rejects_invalid_shape_empty_and_nonfinite(
            tmp_path / f"matrix-shape-{index}",
            matrix,
        )
    terminal_mutations = (
        ({"status": "failed"}, "did not succeed"),
        ({"exit_code": 1}, "did not succeed"),
        ({"confirmation_boundary_verified": False}, "boundary is unverified"),
        ({"confirmation_payloads_opened": True}, "reports confirmation access"),
        ({"source_locks_valid": False}, "invalid source_locks_valid"),
        ({"plan_valid": False}, "invalid plan_valid"),
        ({"download_valid": False}, "invalid download_valid"),
        ({"result_valid": False}, "invalid result_valid"),
        ({"support_gate": None}, "support gate did not pass"),
        ({"support_gate": {"support_passed": False}}, "support gate did not pass"),
    )
    for mutation, message in terminal_mutations:
        adversarial_cases.test_successful_run_guard_rejects_each_invalid_terminal_state(
            mutation,
            message,
        )
    invalid_digests = (
        ({}, "lacks artifact"),
        ({"artifact": "1" * 63}, "lacks artifact"),
        ({"artifact": "z" * 64}, "invalid artifact"),
    )
    for value, message in invalid_digests:
        adversarial_cases.test_required_digest_rejects_missing_length_and_non_hex(
            value,
            message,
        )
    adversarial_cases.test_required_digest_accepts_literal_sha256()
    adversarial_cases.test_result_row_rejects_malformed_missing_duplicate_and_identity_drift()
    adversarial_cases.test_context_rejects_surrounding_object_whitespace(
        tmp_path / "whitespace-object"
    )
    invalid_stages = (
        ("source", "source locks are invalid"),
        ("plan", "plan is invalid"),
        ("download", "download is invalid"),
        ("result", "result is invalid"),
    )
    for index, (stage, message) in enumerate(invalid_stages):
        adversarial_cases.test_context_rejects_each_invalid_revalidated_stage(
            tmp_path / f"invalid-stage-{index}",
            monkeypatch,
            stage,
            message,
        )
    adversarial_cases.test_context_detects_result_change_after_summary_validation(
        tmp_path / "result-race",
        monkeypatch,
    )
    adversarial_cases.test_failure_builder_rejects_whitespace_reason(
        tmp_path / "failure-reason"
    )
    adversarial_cases.test_cli_publishes_technical_failure_case(
        tmp_path / "technical-cli"
    )
