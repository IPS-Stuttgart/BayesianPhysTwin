from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

_HARDENING_TESTS = (
    "test_valid_v2_content_ids_keep_the_released_canonical_form",
    "test_round_trip_freezes_nested_metadata_without_rewriting_timestamp",
    "test_writer_is_no_clobber_by_default_and_atomic_on_overwrite",
    "test_loader_rejects_duplicate_keys_before_digest_validation",
    "test_loader_rejects_digest_coercion_and_noncanonical_digest",
    "test_boolean_seeds_and_other_scalar_coercions_fail_closed",
    "test_artifact_paths_must_be_canonical_and_repository_relative",
    "test_record_subclasses_are_rejected_at_the_public_boundary",
    "test_metadata_keys_and_nonfinite_values_fail_closed",
    "test_loaded_artifact_scalars_fail_closed",
    "test_loaded_boolean_seed_is_rejected_before_fingerprint_comparison",
    "test_direct_builtin_mutation_is_detected_before_rehash",
)


@pytest.mark.parametrize("test_name", _HARDENING_TESTS)
def test_run_manifest_v2_hardening_is_owned_by_stable_runtime_provenance(
    test_name: str,
    tmp_path: Path,
) -> None:
    """Exercise the focused hardening matrix inside the stable-core suite.

    The stable-core manifest already owns ``test_runtime_provenance*.py`` while
    the focused adversarial module is also run by the repository-wide suite.
    Reusing its exact test functions here prevents the coverage ratchet from
    silently omitting new public provenance branches.
    """

    source = Path(__file__).with_name("test_run_manifest_v2_hardening.py")
    namespace: dict[str, Any] = runpy.run_path(str(source))
    test_function = namespace[test_name]
    assert callable(test_function)

    case_root = tmp_path / test_name
    case_root.mkdir()
    test_function(case_root)
