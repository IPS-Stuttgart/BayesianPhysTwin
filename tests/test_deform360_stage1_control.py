from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.cli.deform360_stage1_control import main as stage1_cli
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    Deform360CalibrationArtifactRefV1,
    Deform360CalibrationBundleV1,
    save_deform360_calibration_bundle,
)
from bayesian_phystwin.deform360_stage1_control import (
    DEFORM360_STAGE1_PLAN_STATUS,
    Deform360Stage1PlanV1,
    build_deform360_stage1_plan,
    create_deform360_visual_provider_lock,
    derive_deform360_visual_calibration_lock,
    load_deform360_stage1_plan,
    save_deform360_stage1_plan,
    save_deform360_visual_calibration_lock_atomic,
    save_deform360_visual_provider_lock_atomic,
    verify_deform360_calibration_access,
    verify_deform360_confirmation_access,
    verify_deform360_stage1_seal,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _selection_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _amendment_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    )


def _calibration_design_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    )


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _provider_manifest() -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "provider_name": "prob4d",
        "provider_version": "0.3.0",
        "provider_revision": "d" * 40,
        "provider_api_version": 2,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    return {"manifest_id": _canonical_sha256(descriptor), **descriptor}


def _provider_attestation() -> dict[str, Any]:
    manifest = _provider_manifest()
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": "d" * 40,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "5" * 64,
            "point_artifact_id": "6" * 64,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": "d" * 40,
            "observed_revision": "d" * 40,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _provider_lock(tmp_path: Path):
    attestation_path = tmp_path / "prob4d-provider-attestation.json"
    _write_json(attestation_path, _provider_attestation())
    return create_deform360_visual_provider_lock(
        provider_attestation_path=attestation_path,
        motioncrafter_revision="a" * 40,
        model_set_id="b" * 64,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id="c" * 64,
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
        metadata={"selection_role": "calibration-and-confirmation"},
    )


def _plan(tmp_path: Path):
    provider_lock = _provider_lock(tmp_path)
    provider_path = tmp_path / "visual-provider-lock.json"
    save_deform360_visual_provider_lock_atomic(provider_lock, provider_path)
    plan = build_deform360_stage1_plan(
        selection_path=_selection_path(),
        provider_lock_path=provider_path,
        amendment_path=_amendment_path(),
        calibration_design_path=_calibration_design_path(),
        metadata={"prepared_by": "unit-test"},
    )
    return provider_lock, provider_path, plan


def _artifacts(plan: Deform360Stage1PlanV1):
    groups = tuple(unit.object_id for unit in plan.calibration_units)
    return tuple(
        Deform360CalibrationArtifactRefV1(
            role=role,
            artifact_id=f"{index + 1:064x}",
            implementation_revision="e" * 40,
            selection_evidence_id=f"{index + 101:064x}",
            selected_candidate_id=f"candidate-{index}",
            candidate_count=index + 2,
            calibration_group_ids=groups,
            source_artifacts={
                f"calibration/{role}.json": f"{index + 201:064x}"
            },
            metadata={"selection_rule": "external-or-source-only"},
        )
        for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)
    )


def _bundle(plan: Deform360Stage1PlanV1) -> Deform360CalibrationBundleV1:
    return Deform360CalibrationBundleV1(
        selection_artifact_sha256=plan.selection_artifact_sha256,
        content_selection_sha256=plan.content_selection_sha256,
        dataset_revision=plan.dataset_revision,
        processing_revision=plan.processing_revision,
        implementation_revision="e" * 40,
        calibration_units=plan.calibration_units,
        confirmation_units=plan.confirmation_units,
        calibration_artifacts=_artifacts(plan),
        evidence_use_ledger_id="f" * 64,
        source_artifacts={"calibration/evidence-ledger.json": "9" * 64},
        metadata={"sealed_by": "unit-test"},
    )


def _recompute_stage0_identities(value: dict[str, Any]) -> None:
    value["selection_sha256"] = _canonical_sha256(value["selection"])
    content = dict(value)
    for field_name in (
        "content_selection_sha256",
        "implementation_revision",
        "selection_artifact_sha256",
    ):
        content.pop(field_name, None)
    value["content_selection_sha256"] = _canonical_sha256(content)
    artifact = dict(value)
    artifact.pop("selection_artifact_sha256", None)
    value["selection_artifact_sha256"] = _canonical_sha256(artifact)


def test_provider_lock_and_stage1_plan_bind_current_locked_inputs(
    tmp_path: Path,
) -> None:
    provider_lock, _, plan = _plan(tmp_path)
    path = tmp_path / "stage1-plan.json"
    save_deform360_stage1_plan(plan, path)
    loaded = load_deform360_stage1_plan(path)

    assert loaded == plan
    assert loaded.status == DEFORM360_STAGE1_PLAN_STATUS
    assert loaded.visual_provider_lock_id == provider_lock.artifact_id
    assert len(loaded.calibration_units) == 10
    assert len(loaded.confirmation_units) == 12
    assert loaded.summary()["calibration_payloads_opened"] is False
    assert loaded.summary()["confirmation_payloads_opened"] is False
    assert loaded.summary()["target_outcomes_used"] is False
    assert loaded.summary()["replacement_allowed"] is False

    token = verify_deform360_calibration_access(
        loaded,
        expected_plan_id=loaded.plan_id,
        expected_provider_lock_id=provider_lock.artifact_id,
        expected_selection_artifact_sha256=loaded.selection_artifact_sha256,
    )
    assert token == loaded.calibration_access_token


def test_provider_lock_requires_claim_bearing_independently_verified_prob4d(
    tmp_path: Path,
) -> None:
    attestation = _provider_attestation()
    attestation["runtime_revision"]["independently_verified"] = False
    path = tmp_path / "attestation.json"
    _write_json(path, attestation)

    with pytest.raises(ValueError, match="verification flag"):
        create_deform360_visual_provider_lock(
            provider_attestation_path=path,
            motioncrafter_revision="a" * 40,
            model_set_id="b" * 64,
            root_seed=1,
            seed_policy="per-object-derived-seed-v1",
            window_size=25,
            overlap=8,
            height=320,
            width=640,
            storage_dtype="float32",
            initial_metric_frame_prior_id="c" * 64,
            additional_metric_anchor_policy="none",
            max_gauge_rank=64,
            minimum_retained_gauge_trace=0.999,
        )


def test_stage1_plan_rejects_tampered_selection_and_opened_payloads(
    tmp_path: Path,
) -> None:
    provider_lock = _provider_lock(tmp_path)
    provider_path = tmp_path / "provider.json"
    save_deform360_visual_provider_lock_atomic(provider_lock, provider_path)
    selection = json.loads(_selection_path().read_text(encoding="utf-8"))

    tampered = json.loads(json.dumps(selection))
    tampered["selection"]["calibration"][0]["episode_id"] += 1
    tampered_path = tmp_path / "tampered-selection.json"
    _write_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="artifact identity"):
        build_deform360_stage1_plan(
            selection_path=tampered_path,
            provider_lock_path=provider_path,
            amendment_path=_amendment_path(),
            calibration_design_path=_calibration_design_path(),
        )

    opened = json.loads(json.dumps(selection))
    opened["information_boundary"]["camera_media_opened"] = True
    _recompute_stage0_identities(opened)
    opened_path = tmp_path / "opened-selection.json"
    _write_json(opened_path, opened)
    with pytest.raises(ValueError, match="camera_media_opened"):
        build_deform360_stage1_plan(
            selection_path=opened_path,
            provider_lock_path=provider_path,
            amendment_path=_amendment_path(),
            calibration_design_path=_calibration_design_path(),
        )


def test_plan_verification_is_bound_to_all_reviewed_identities(
    tmp_path: Path,
) -> None:
    provider_lock, _, plan = _plan(tmp_path)

    with pytest.raises(ValueError, match="plan identity"):
        verify_deform360_calibration_access(
            plan,
            expected_plan_id="0" * 64,
            expected_provider_lock_id=provider_lock.artifact_id,
            expected_selection_artifact_sha256=plan.selection_artifact_sha256,
        )
    with pytest.raises(ValueError, match="provider lock"):
        verify_deform360_calibration_access(
            plan,
            expected_plan_id=plan.plan_id,
            expected_provider_lock_id="0" * 64,
            expected_selection_artifact_sha256=plan.selection_artifact_sha256,
        )
    with pytest.raises(ValueError, match="selection artifact"):
        verify_deform360_calibration_access(
            plan,
            expected_plan_id=plan.plan_id,
            expected_provider_lock_id=provider_lock.artifact_id,
            expected_selection_artifact_sha256="0" * 64,
        )


def test_complete_calibration_bundle_derives_and_verifies_the_seal(
    tmp_path: Path,
) -> None:
    provider_lock, _, plan = _plan(tmp_path)
    bundle = _bundle(plan)
    calibration_lock = derive_deform360_visual_calibration_lock(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
    )
    summary = verify_deform360_stage1_seal(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
        calibration_lock=calibration_lock,
    )

    assert calibration_lock.visual_provider_lock_id == provider_lock.artifact_id
    assert calibration_lock.selection_lock_id == plan.selection_artifact_sha256
    assert calibration_lock.calibration_group_count == 10
    assert calibration_lock.conformal_rank == 10
    assert summary["stage1_plan_id"] == plan.plan_id
    assert summary["calibration_bundle_id"] == bundle.bundle_id
    assert summary["visual_calibration_lock_id"] == calibration_lock.artifact_id
    assert summary["confirmation_opening_token"] == (
        bundle.confirmation_opening_token
    )
    assert summary["confirmation_payloads_opened"] is False
    assert summary["target_outcomes_used"] is False

    reviewed = verify_deform360_confirmation_access(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
        calibration_lock=calibration_lock,
        expected_plan_id=plan.plan_id,
        expected_provider_lock_id=provider_lock.artifact_id,
        expected_bundle_id=bundle.bundle_id,
        expected_calibration_lock_id=calibration_lock.artifact_id,
        expected_selection_artifact_sha256=plan.selection_artifact_sha256,
        expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
    )
    assert reviewed["reviewed_identity_gate_passed"] is True

    with pytest.raises(ValueError, match="reviewed calibration bundle"):
        verify_deform360_confirmation_access(
            plan=plan,
            provider_lock=provider_lock,
            bundle=bundle,
            calibration_lock=calibration_lock,
            expected_plan_id=plan.plan_id,
            expected_provider_lock_id=provider_lock.artifact_id,
            expected_bundle_id="0" * 64,
            expected_calibration_lock_id=calibration_lock.artifact_id,
            expected_selection_artifact_sha256=(
                plan.selection_artifact_sha256
            ),
            expected_evidence_use_ledger_id=bundle.evidence_use_ledger_id,
        )

    path = tmp_path / "visual-calibration-lock.json"
    save_deform360_visual_calibration_lock_atomic(calibration_lock, path)
    with pytest.raises(FileExistsError):
        save_deform360_visual_calibration_lock_atomic(calibration_lock, path)


def test_seal_rejects_changed_provider_cohort_and_calibration_bundle(
    tmp_path: Path,
) -> None:
    provider_lock, _, plan = _plan(tmp_path)
    bundle = _bundle(plan)

    with pytest.raises(ValueError, match="provider lock"):
        derive_deform360_visual_calibration_lock(
            plan=plan,
            provider_lock=replace(provider_lock, model_set_id="0" * 64),
            bundle=bundle,
        )
    with pytest.raises(ValueError, match="dataset revision"):
        derive_deform360_visual_calibration_lock(
            plan=plan,
            provider_lock=provider_lock,
            bundle=replace(bundle, dataset_revision="0" * 40),
        )
    changed_unit = replace(
        bundle.calibration_units[0],
        episode_id=bundle.calibration_units[0].episode_id + 1,
    )
    with pytest.raises(ValueError, match="calibration cohort"):
        derive_deform360_visual_calibration_lock(
            plan=plan,
            provider_lock=provider_lock,
            bundle=replace(
                bundle,
                calibration_units=(changed_unit, *bundle.calibration_units[1:]),
            ),
        )


def test_stage1_plan_loader_rejects_tampering_duplicate_json_and_type_errors(
    tmp_path: Path,
) -> None:
    provider_lock, _, plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    save_deform360_stage1_plan(plan, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["visual_provider_lock_id"] = "0" * 64
    _write_json(path, payload)
    with pytest.raises(ValueError, match="plan_id"):
        load_deform360_stage1_plan(path)

    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_stage1_plan(path)

    with pytest.raises(TypeError, match="Deform360Stage1PlanV1"):
        save_deform360_stage1_plan(
            object(),  # type: ignore[arg-type]
            tmp_path / "bad.json",
        )
    with pytest.raises(TypeError, match="Deform360Stage1PlanV1"):
        verify_deform360_calibration_access(
            object(),  # type: ignore[arg-type]
            expected_plan_id="0" * 64,
            expected_provider_lock_id=provider_lock.artifact_id,
            expected_selection_artifact_sha256=plan.selection_artifact_sha256,
        )


def test_cli_inputs_can_be_persisted_for_later_seal(tmp_path: Path) -> None:
    _, _, plan = _plan(tmp_path)
    bundle = _bundle(plan)
    path = tmp_path / "calibration-bundle.json"
    save_deform360_calibration_bundle(bundle, path)

    assert path.is_file()
    assert bundle.selection_artifact_sha256 == plan.selection_artifact_sha256


def test_grouped_cli_prepares_verifies_and_seals_stage1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attestation_path = tmp_path / "provider-attestation.json"
    provider_path = tmp_path / "visual-provider-lock.json"
    plan_path = tmp_path / "stage1-plan.json"
    bundle_path = tmp_path / "calibration-bundle.json"
    calibration_lock_path = tmp_path / "visual-calibration-lock.json"
    summary_path = tmp_path / "seal-summary.json"
    _write_json(attestation_path, _provider_attestation())

    assert (
        stage1_cli(
            [
                "provider-lock",
                "--provider-attestation",
                str(attestation_path),
                "--motioncrafter-revision",
                "a" * 40,
                "--model-set-id",
                "b" * 64,
                "--initial-metric-frame-prior-id",
                "c" * 64,
                "--output",
                str(provider_path),
            ]
        )
        == 0
    )
    provider_summary = json.loads(capsys.readouterr().out)

    assert (
        stage1_cli(
            [
                "plan",
                "--provider-lock",
                str(provider_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    plan_summary = json.loads(capsys.readouterr().out)
    plan = load_deform360_stage1_plan(plan_path)

    assert (
        stage1_cli(
            [
                "verify-plan",
                "--plan",
                str(plan_path),
                "--expected-plan-id",
                plan.plan_id,
                "--expected-provider-lock-id",
                provider_summary["artifact_id"],
                "--expected-selection-artifact-sha256",
                plan.selection_artifact_sha256,
            ]
        )
        == 0
    )
    verified_plan = json.loads(capsys.readouterr().out)
    assert verified_plan["verified_calibration_access_token"] == (
        plan.calibration_access_token
    )

    bundle = _bundle(plan)
    save_deform360_calibration_bundle(bundle, bundle_path)
    assert (
        stage1_cli(
            [
                "seal",
                "--plan",
                str(plan_path),
                "--provider-lock",
                str(provider_path),
                "--calibration-bundle",
                str(bundle_path),
                "--output",
                str(calibration_lock_path),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )
    seal_summary = json.loads(capsys.readouterr().out)
    assert seal_summary["stage1_plan_id"] == plan_summary["plan_id"]
    assert summary_path.is_file()

    assert (
        stage1_cli(
            [
                "verify-seal",
                "--plan",
                str(plan_path),
                "--provider-lock",
                str(provider_path),
                "--calibration-bundle",
                str(bundle_path),
                "--calibration-lock",
                str(calibration_lock_path),
                "--expected-plan-id",
                plan.plan_id,
                "--expected-provider-lock-id",
                provider_summary["artifact_id"],
                "--expected-bundle-id",
                bundle.bundle_id,
                "--expected-calibration-lock-id",
                seal_summary["visual_calibration_lock_id"],
                "--expected-selection-artifact-sha256",
                plan.selection_artifact_sha256,
                "--expected-evidence-use-ledger-id",
                bundle.evidence_use_ledger_id,
            ]
        )
        == 0
    )
    verified_seal = json.loads(capsys.readouterr().out)
    assert verified_seal["visual_calibration_lock_id"] == (
        seal_summary["visual_calibration_lock_id"]
    )
