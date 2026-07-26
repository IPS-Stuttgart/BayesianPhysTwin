import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "phystwin_prior_aware_sparse_identity_source_v5.json"
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
V4_FAILURE_PATH = (
    ROOT
    / "results"
    / "sota"
    / "phystwin_prior_aware_sparse_identity_source_v4"
    / "technical_failure.json"
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "phystwin_prior_aware_sparse_identity_source_v5"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_lock_binds_every_runtime_implementation_dependency() -> None:
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
        "identity_split_module_sha256": (
            ROOT / "src" / "bayesian_phystwin" / "phystwin_sparse_identity_split.py"
        ),
    }

    assert protocol["schema_version"] == 5
    assert protocol["protocol_id"] == (
        "phystwin-prior-aware-sparse-identity-source-v5"
    )
    for field, path in paths.items():
        assert implementation[field] == _sha256(path)


def test_v5_lock_preserves_hidden_identity_and_exact_fallback_boundaries() -> None:
    protocol = _load(PROTOCOL_PATH)
    source_qa = protocol["source_qa"]

    assert set(source_qa["observed_identity_indices"]).isdisjoint(
        source_qa["hidden_identity_indices"]
    )
    assert source_qa["observed_identity_indices"] == [3, 4, 8, 5]
    assert source_qa["prefix_complete_candidate_indices"] == [3, 4, 5, 6, 8]
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


def test_v4_failure_preserves_exact_state_carrier_before_source_redesign() -> None:
    failure = _load(V4_FAILURE_PATH)

    assert failure["registered_replay_parity"]["all_frames_vector_rmse_m"] == 0.0
    assert failure["registered_attempt"]["state_carrier_path"] == (
        "replay_state_carrier.npz"
    )
    assert failure["information_boundary"]["state_fit_started"] is False
    assert failure["allowed_prefix_support_audit"][
        "replacement_selection_uses_future_frames"
    ] is False
    assert failure["disposition"]["replacement_changes_scientific_method"] is True
    assert failure["disposition"]["replacement_protocol"] == (
        "phystwin-prior-aware-sparse-identity-source-v5"
    )


def test_v5_result_binds_exact_fallback_and_failed_advancement_gate() -> None:
    prediction = _load(RESULT_ROOT / "prediction_manifest.json")
    score = _load(RESULT_ROOT / "score.json")
    summary = _load(RESULT_ROOT / "summary.json")

    hashes = prediction["prediction_array_sha256"]
    assert hashes["candidate_trajectory"] == hashes["persistence_trajectory"]
    assert hashes["state_only_trajectory"] == hashes["persistence_trajectory"]
    assert prediction["accepted_state_update"] is False
    assert score["accepted_state_update"] is False
    assert summary["advancement_gate"]["passed"] is False
    assert summary["advancement_gate"]["future_chamfer_improves"] is True
    assert summary["advancement_gate"]["future_hidden_identity_track_improves"] is False
    assert summary["artifacts"]["score_sha256"] == _sha256(RESULT_ROOT / "score.json")
