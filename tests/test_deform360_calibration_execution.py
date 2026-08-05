from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    Deform360CalibrationArtifactRefV1,
)
from bayesian_phystwin.deform360_calibration_execution import (
    DEFORM360_CALIBRATION_LEDGER_CASE_ID,
    Deform360CalibrationExecutionArtifactsV1,
    Deform360CalibrationExecutionSealV1,
    build_deform360_calibration_execution_seal,
    deform360_calibration_component_ids,
    load_deform360_calibration_execution_seal,
    load_deform360_stage0_selection,
    save_deform360_calibration_execution_seal,
    verify_deform360_calibration_execution_artifacts,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualProviderLockV1,
    save_deform360_visual_provider_lock,
)
from bayesian_phystwin.evidence_use_ledger import (
    EvidenceUseLedgerV1,
    EvidenceUseV1,
    save_evidence_use_ledger,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage0():
    return load_deform360_stage0_selection(
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _provider() -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1(
        provider_revision="1" * 40,
        provider_manifest_id="2" * 64,
        provider_attestation_sha256="3" * 64,
        motioncrafter_revision="4" * 40,
        model_set_id="5" * 64,
        root_seed=20260805,
        seed_policy="per-object-derived-seed-v1",
        window_size=25,
        overlap=8,
        height=320,
        width=640,
        storage_dtype="float32",
        initial_metric_frame_prior_id="6" * 64,
        additional_metric_anchor_policy="none",
        max_gauge_rank=64,
        minimum_retained_gauge_trace=0.999,
    )


def _artifacts(stage0) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
    return tuple(
        Deform360CalibrationArtifactRefV1(
            role=role,
            artifact_id=f"{index + 1:064x}",
            implementation_revision="a" * 40,
            selection_evidence_id=f"{index + 101:064x}",
            selected_candidate_id=f"candidate-{index}",
            candidate_count=index + 2,
            calibration_group_ids=groups,
            source_artifacts={f"calibration/{role}.json": f"{index + 201:064x}"},
            metadata={"selection_rule": "source-only"},
        )
        for index, role in enumerate(DEFORM360_CALIBRATION_ROLES)
    )


def _entry(
    unit,
    *,
    index: int,
    role: str = "calibration_only",
    object_id: str | None = None,
) -> EvidenceUseV1:
    selected_object = object_id or unit.object_id
    return EvidenceUseV1(
        evidence_artifact_id=f"{index + 301:064x}",
        raw_factor_id=f"{index + 401:064x}",
        raw_factor_sha256=f"{index + 501:064x}",
        source_repository="brownu/deform360",
        source_revision="b" * 40,
        source_artifacts={
            unit.metadata_path: unit.metadata_sha256,
        },
        sensor_family="deform360-calibration-source",
        stream_id=f"{selected_object}:episode-{unit.episode_id}",
        clock_id="deform360-frame-clock",
        causal_frame_start=0,
        causal_frame_stop=10,
        correlation_group_ids=(f"group-{index}",),
        inference_role=role,
        metadata={"object_id": selected_object},
    )


def _ledger(stage0, *, entries=None, case_id=None) -> EvidenceUseLedgerV1:
    selected_entries = entries or tuple(
        _entry(unit, index=index) for index, unit in enumerate(stage0.calibration_units)
    )
    return EvidenceUseLedgerV1(
        protocol_id=stage0.protocol_id,
        case_id=case_id or DEFORM360_CALIBRATION_LEDGER_CASE_ID,
        causal_frame_stop=10,
        entries=selected_entries,
        metadata={"statistical_unit": "physical_object"},
    )


def _sources(stage0) -> dict[str, str]:
    result = {
        "sources/stage0/selection.json": stage0.source_sha256,
        "sources/locks/visual-provider-lock.json": "1" * 64,
        "sources/calibration/evidence-use-ledger.json": "2" * 64,
    }
    for index, role in enumerate(DEFORM360_CALIBRATION_ROLES):
        result[f"sources/calibration/artifacts/{role}.json"] = f"{index + 10:064x}"
    return result


def _products(**updates: Any) -> Deform360CalibrationExecutionArtifactsV1:
    stage0 = updates.pop("stage0_selection", _stage0())
    provider = updates.pop("visual_provider_lock", _provider())
    ledger = updates.pop("evidence_use_ledger", _ledger(stage0))
    artifacts = updates.pop("calibration_artifacts", _artifacts(stage0))
    values: dict[str, Any] = {
        "stage0_selection": stage0,
        "visual_provider_lock": provider,
        "evidence_use_ledger": ledger,
        "calibration_artifacts": artifacts,
        "implementation_revision": "c" * 40,
        "source_artifacts": _sources(stage0),
        "metadata": {"sealed_by": "test"},
    }
    values.update(updates)
    return build_deform360_calibration_execution_seal(**values)


def test_stage0_loader_binds_exact_official_hub_cohort() -> None:
    stage0 = _stage0()

    assert stage0.protocol_id == "deform360-official-hub-visuotactile-v1"
    assert stage0.dataset_revision == "f804696d7a133908c7497ffdab43819d879b5cbc"
    assert stage0.processing_revision == "d8522a4403b766aeb387510c04e89032a56fdf35"
    assert len(stage0.calibration_units) == 10
    assert len(stage0.confirmation_units) == 12
    assert len(stage0.snapshot_id) == 64
    assert set(unit.stratum for unit in stage0.calibration_units) == {
        "sheet",
        "volumetric",
    }


def test_stage0_loader_rejects_boundary_and_selection_drift(
    tmp_path: Path,
) -> None:
    source = (
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )
    base = json.loads(source.read_text(encoding="utf-8"))
    mutations = (
        ("schema", "changed", "schema changed"),
        ("schema_version", 2, "version changed"),
        ("protocol_id", "changed", "protocol changed"),
        (
            "replacement_allowed_after_payload_access",
            True,
            "replacement boundary",
        ),
    )
    for key, value, message in mutations:
        payload = dict(base)
        payload[key] = value
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_deform360_stage0_selection(path)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["information_boundary"]["target_outcomes_opened"] = True
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="information boundary"):
        load_deform360_stage0_selection(path)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["selection"]["calibration"].pop()
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 5"):
        load_deform360_stage0_selection(path)


def test_component_ids_bind_all_eight_roles_and_ignore_input_order() -> None:
    artifacts = _artifacts(_stage0())
    forward = deform360_calibration_component_ids(artifacts)
    reverse = deform360_calibration_component_ids(tuple(reversed(artifacts)))

    assert forward == reverse
    assert set(forward) == {"visual", "contact_anchor", "guard", "interval"}
    assert len(set(forward.values())) == 4
    with pytest.raises(ValueError, match="incomplete"):
        deform360_calibration_component_ids(artifacts[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        deform360_calibration_component_ids((*artifacts[:-1], artifacts[0]))
    with pytest.raises(ValueError, match="must contain"):
        deform360_calibration_component_ids(
            (*artifacts[:-1], object())  # type: ignore[arg-type]
        )


def test_builder_seals_all_cross_artifact_identities() -> None:
    stage0 = _stage0()
    provider = _provider()
    ledger = _ledger(stage0)
    products = _products(
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )

    verify_deform360_calibration_execution_artifacts(
        products,
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )
    assert (
        products.visual_calibration_lock.visual_provider_lock_id == provider.artifact_id
    )
    assert products.visual_calibration_lock.selection_lock_id == stage0.selection_sha256
    assert products.calibration_bundle.calibration_units == (stage0.calibration_units)
    assert products.calibration_bundle.confirmation_units == (stage0.confirmation_units)
    assert products.execution_seal.calibration_payloads_opened is True
    assert products.execution_seal.confirmation_payloads_opened is False
    assert products.execution_seal.target_outcomes_used is False
    assert products.execution_seal.confirmation_opening_token == (
        products.calibration_bundle.confirmation_opening_token
    )


def test_builder_rejects_incomplete_or_leaky_evidence() -> None:
    stage0 = _stage0()
    complete = tuple(
        _entry(unit, index=index) for index, unit in enumerate(stage0.calibration_units)
    )

    with pytest.raises(ValueError, match="does not cover every"):
        _products(
            stage0_selection=stage0,
            evidence_use_ledger=_ledger(stage0, entries=complete[:-1]),
        )
    bad_role = (
        _entry(stage0.calibration_units[0], index=0, role="state_update"),
        *complete[1:],
    )
    with pytest.raises(ValueError, match="calibration_only"):
        _products(
            stage0_selection=stage0,
            evidence_use_ledger=_ledger(stage0, entries=bad_role),
        )
    confirmation = stage0.confirmation_units[0].object_id
    leaked = (
        _entry(
            stage0.calibration_units[0],
            index=0,
            object_id=confirmation,
        ),
        *complete[1:],
    )
    with pytest.raises(ValueError, match="confirmation-object"):
        _products(
            stage0_selection=stage0,
            evidence_use_ledger=_ledger(stage0, entries=leaked),
        )
    with pytest.raises(ValueError, match="case_id"):
        _products(
            stage0_selection=stage0,
            evidence_use_ledger=_ledger(
                stage0,
                case_id="different-calibration-cohort",
            ),
        )


def test_builder_rejects_wrong_types_and_incomplete_sources() -> None:
    stage0 = _stage0()
    with pytest.raises(TypeError, match="stage0_selection"):
        build_deform360_calibration_execution_seal(
            stage0_selection=object(),  # type: ignore[arg-type]
            visual_provider_lock=_provider(),
            evidence_use_ledger=_ledger(stage0),
            calibration_artifacts=_artifacts(stage0),
            implementation_revision="c" * 40,
            source_artifacts=_sources(stage0),
        )
    with pytest.raises(TypeError, match="visual_provider_lock"):
        _products(
            visual_provider_lock=object(),  # type: ignore[arg-type]
        )
    sources = _sources(stage0)
    sources.pop("sources/calibration/evidence-use-ledger.json")
    with pytest.raises(ValueError, match="source artifacts are incomplete"):
        _products(
            stage0_selection=stage0,
            source_artifacts=sources,
        )


def test_execution_seal_roundtrip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    seal = _products().execution_seal
    path = tmp_path / "seal.json"
    save_deform360_calibration_execution_seal(seal, path)

    loaded = load_deform360_calibration_execution_seal(path)
    assert loaded == seal
    assert loaded.seal_id == seal.seal_id
    with pytest.raises(FileExistsError):
        save_deform360_calibration_execution_seal(seal, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["visual_provider_lock_id"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal_id"):
        load_deform360_calibration_execution_seal(path)

    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_calibration_execution_seal(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"calibration_payloads_opened": False}, "acknowledge"),
        ({"confirmation_payloads_opened": True}, "confirmation payloads"),
        ({"target_outcomes_used": True}, "target outcomes"),
        ({"status": "changed"}, "status changed"),
        ({"protocol_id": "changed"}, "protocol changed"),
        ({"calibration_object_ids": ("only-one",)}, "10 independent"),
        ({"confirmation_object_ids": ("only-one",)}, "12 unique"),
        ({"metadata": {"bad": float("nan")}}, "finite JSON"),
    ],
)
def test_execution_seal_rejects_invalid_boundaries(
    updates: dict[str, Any],
    message: str,
) -> None:
    seal = _products().execution_seal
    values = {field: getattr(seal, field) for field in seal.__dataclass_fields__}
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        Deform360CalibrationExecutionSealV1(**values)


def test_independent_verifier_rejects_cross_artifact_substitution() -> None:
    stage0 = _stage0()
    provider = _provider()
    ledger = _ledger(stage0)
    products = _products(
        stage0_selection=stage0,
        visual_provider_lock=provider,
        evidence_use_ledger=ledger,
    )
    substituted_lock = replace(
        products.visual_calibration_lock,
        visual_provider_lock_id="f" * 64,
    )
    substituted = products._replace(visual_calibration_lock=substituted_lock)
    with pytest.raises(ValueError, match="provider identity"):
        verify_deform360_calibration_execution_artifacts(
            substituted,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
        )


def _write_cli_inputs(tmp_path: Path):
    stage0 = _stage0()
    provider_path = tmp_path / "provider.json"
    save_deform360_visual_provider_lock(provider_path, _provider())
    ledger_path = tmp_path / "ledger.json"
    save_evidence_use_ledger(_ledger(stage0), ledger_path)

    artifact_args: list[str] = []
    for artifact in _artifacts(stage0):
        path = tmp_path / f"{artifact.role}.json"
        write_atomic_json(artifact.to_record(), path, overwrite=False)
        artifact_args.extend(("--artifact", f"{artifact.role}={path}"))
    return provider_path, ledger_path, artifact_args


def test_cli_publishes_one_atomic_portable_seal(tmp_path: Path) -> None:
    from bayesian_phystwin.cli.deform360_calibration_execution import main

    root = _repository_root()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    selection = (
        root
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )
    provider, ledger, artifact_args = _write_cli_inputs(tmp_path)
    output = tmp_path / "sealed"
    arguments = [
        "--selection-lock",
        str(selection),
        "--visual-provider-lock",
        str(provider),
        "--evidence-ledger",
        str(ledger),
        *artifact_args,
        "--implementation-revision",
        revision,
        "--repository-root",
        str(root),
        "--output-dir",
        str(output),
        "--calibration-payloads-opened",
    ]

    assert main(arguments) == 0
    required = {
        "calibration-bundle.json",
        "calibration-execution-seal.json",
        "visual-calibration-lock.json",
        "summary.json",
        "STATUS.md",
        "SHA256SUMS",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert (output / "sources" / "stage0" / "selection.json").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["confirmation_payloads_opened"] is False
    assert summary["target_outcomes_used"] is False
    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert "calibration-execution-seal.json" in checksums

    with pytest.raises(FileExistsError):
        main(arguments)
