import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "phystwin_prior_aware_sparse_identity_source_v4.json"
)
V1_FAILURE_PATH = (
    ROOT
    / "results"
    / "sota"
    / "phystwin_prior_aware_sparse_identity_source_v1"
    / "technical_failure.json"
)
V2_FAILURE_PATH = (
    ROOT
    / "results"
    / "sota"
    / "phystwin_prior_aware_sparse_identity_source_v2"
    / "technical_failure.json"
)
V3_FAILURE_PATH = (
    ROOT
    / "results"
    / "sota"
    / "phystwin_prior_aware_sparse_identity_source_v3"
    / "technical_failure.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_lock_binds_every_runtime_implementation_dependency() -> None:
    protocol = _load(PROTOCOL_PATH)
    implementation = protocol["implementation"]
    paths = {
        "runner_sha256": (
            ROOT / "scripts" / "remote" / "run_phystwin_sparse_state_update_source.py"
        ),
        "state_update_module_sha256": (
            ROOT / "src" / "bayesian_phystwin" / "phystwin_sparse_state_update.py"
        ),
        "propagated_state_module_sha256": (
            ROOT / "src" / "bayesian_phystwin" / "propagated_state_correction.py"
        ),
        "state_injection_module_sha256": (
            ROOT / "src" / "bayesian_phystwin" / "phystwin_state_injection.py"
        ),
    }

    assert protocol["schema_version"] == 4
    assert protocol["protocol_id"] == (
        "phystwin-prior-aware-sparse-identity-source-v4"
    )
    for field, path in paths.items():
        assert implementation[field] == _sha256(path)


def test_v4_lock_preserves_hidden_identity_and_exact_fallback_boundaries() -> None:
    protocol = _load(PROTOCOL_PATH)
    source_qa = protocol["source_qa"]

    assert set(source_qa["observed_identity_indices"]).isdisjoint(
        source_qa["hidden_identity_indices"]
    )
    assert protocol["state_update"]["fit_frame_count"] < (
        protocol["state_update"]["response_frame_count"]
    )
    assert protocol["simulator"]["maximum_replay_vector_rmse_m"] == 0.0
    assert protocol["simulator"]["maximum_replay_norm_m"] == 0.0
    assert protocol["simulator"]["self_collision"] is False
    assert protocol["prediction_and_scoring"]["rejection_policy"] == (
        "return the sealed persistence trajectory byte-for-byte"
    )
    assert protocol["predecessor"]["future_identity_outcomes_used_or_scored"] is False


def test_v1_failure_is_archived_as_pre_outcome_technical_evidence() -> None:
    failure = _load(V1_FAILURE_PATH)

    assert failure["registered_attempt"]["prediction_manifest_written"] is False
    assert failure["registered_attempt"]["score_written"] is False
    assert failure["information_boundary"]["state_fit_started"] is False
    assert (
        failure["information_boundary"][
            "future_manual_identity_values_used_for_fit_selection_or_diagnosis"
        ]
        is False
    )
    assert failure["target_free_diagnosis"]["fixed_helper_vector_rmse_m"] == 0.0
    assert failure["disposition"]["replacement_protocol"] == (
        "phystwin-prior-aware-sparse-identity-source-v2"
    )


def test_v2_failure_is_archived_before_array_loading() -> None:
    failure = _load(V2_FAILURE_PATH)

    assert failure["registered_attempt"]["input_arrays_unpickled"] is False
    assert failure["registered_attempt"]["output_directory_created"] is False
    assert failure["information_boundary"]["simulator_initialized"] is False
    assert (
        failure["target_free_diagnosis"]["selected_baseline_vector_rmse_m"] == 0.0
    )
    assert failure["target_free_diagnosis"]["released_trajectory_vector_rmse_m"] > 0.0
    assert failure["disposition"]["replacement_protocol"] == (
        "phystwin-prior-aware-sparse-identity-source-v3"
    )


def test_v3_failure_binds_replay_parity_before_state_fit() -> None:
    failure = _load(V3_FAILURE_PATH)

    assert failure["registered_attempt"]["state_carrier_written"] is False
    assert failure["information_boundary"]["state_fit_started"] is False
    assert failure["information_boundary"]["future_metrics_computed"] is False
    assert failure["registered_replay_parity"]["all_frames"]["vector_rmse_m"] > 0.0
    assert failure["target_free_diagnosis"]["v3_self_collision"] is True
    assert failure["target_free_diagnosis"]["historical_source_self_collision"] is False
    assert failure["disposition"]["replacement_protocol"] == (
        "phystwin-prior-aware-sparse-identity-source-v4"
    )
