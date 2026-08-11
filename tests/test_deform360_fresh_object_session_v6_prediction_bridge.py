from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    D1_NATIVE,
    VARIANT_IDS,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
    load_deform360_fresh_object_session_v6_covariance_amendment,
    load_deform360_fresh_object_session_v6_policy,
    load_deform360_v6_source_selection,
)
from bayesian_phystwin.deform360_fresh_object_session_v6_prediction_bridge import (
    bridge_deform360_v6_source_prediction_batch,
    build_deform360_v6_source_candidate_panel,
    load_deform360_v6_source_execution_amendment,
    publish_deform360_v6_prediction_bridge,
    validate_deform360_v6_prediction_bridge_receipt,
    validate_deform360_v6_source_candidate_panel,
)
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    build_deform360_joint_sparse_source_prediction_batch_v5,
    build_deform360_joint_sparse_source_prediction_seal_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    RAW_METHOD_IDS,
    load_deform360_joint_sparse_source_execution_lock_v5,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "protocols/locks/deform360_official_hub_fresh_object_session_v6.json"
)
COVARIANCE_PATH = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_source_covariance.json"
)
EXECUTION_PATH = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_source_prediction_execution.json"
)
SELECTION_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
V5_LOCK_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
REVISION = "a" * 40
BRIDGE_REVISION = "b" * 40


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _locks():
    policy = load_deform360_fresh_object_session_v6_policy(POLICY_PATH)
    covariance = load_deform360_fresh_object_session_v6_covariance_amendment(
        COVARIANCE_PATH,
        policy,
    )
    selection, cohort = load_deform360_v6_source_selection(SELECTION_PATH, policy)
    v5_lock = load_deform360_joint_sparse_source_execution_lock_v5(V5_LOCK_PATH)
    execution = load_deform360_v6_source_execution_amendment(
        EXECUTION_PATH,
        v5_execution_lock_id=v5_lock["execution_lock_id"],
    )
    return policy, covariance, execution, selection, cohort, v5_lock


def _v5_batch(v5_lock, cohort):
    seals = []
    object_ids = sorted(cohort)
    for outer_id in object_ids:
        for object_id in object_ids:
            role = "held_out" if outer_id == object_id else "training"
            excluded = {outer_id} if role == "held_out" else {outer_id, object_id}
            fit_ids = sorted(set(object_ids) - excluded)
            seals.append(
                build_deform360_joint_sparse_source_prediction_seal_v5(
                    lock=v5_lock,
                    implementation_revision=REVISION,
                    outer_held_out_object_id=outer_id,
                    record_role=role,
                    object_id=object_id,
                    factor_admitted=True,
                    technical_failure=False,
                    physical_mode="warp_twin",
                    risk_score=0.1,
                    prediction_fit_artifact_id=_digest(
                        f"v5/{outer_id}/{object_id}/fit"
                    ),
                    prediction_fit_object_ids=fit_ids,
                    methods={
                        method_id: {
                            "artifact_id": _digest(
                                f"v5/{object_id}/{method_id}"
                                if method_id
                                in {
                                    "B0_physical_fallback",
                                    "B1_last_causal_residual",
                                }
                                else f"v5/{outer_id}/{object_id}/{method_id}"
                            ),
                            "predicted_loss_mm": float(index + 1),
                        }
                        for index, method_id in enumerate(RAW_METHOD_IDS)
                    },
                    source_artifacts={
                        f"v5/{outer_id}/{object_id}.json": _digest(
                            f"source/{outer_id}/{object_id}"
                        )
                    },
                )
            )
    return build_deform360_joint_sparse_source_prediction_batch_v5(seals, v5_lock)


def _held_out(batch, object_id: str):
    return next(
        row
        for row in batch["records"]
        if row["record_role"] == "held_out" and row["object_id"] == object_id
    )


def _variant(
    variant_id: str,
    *,
    object_id: str,
    cohort: dict[str, tuple[int, str]],
    prediction_id: str,
    available: bool = True,
) -> dict[str, object]:
    baseline = variant_id in {B0, B1}
    if not available:
        return {
            "available": False,
            "accepted": False,
            "prediction_artifact_id": None,
            "fit_artifact_id": None,
            "fit_object_ids": sorted(set(cohort) - {object_id}),
            "guard_artifact_id": None,
            "guard_threshold": None,
            "covariance_artifact_id": None,
            "interval_artifact_id": None,
            "risk_score": None,
            "unavailable_reason": "source-owned-covariance-unavailable",
        }
    return {
        "available": True,
        "accepted": False if variant_id == B0 else True,
        "prediction_artifact_id": prediction_id,
        "fit_artifact_id": _digest(f"v6/{object_id}/{variant_id}/fit"),
        "fit_object_ids": [] if baseline else sorted(set(cohort) - {object_id}),
        "guard_artifact_id": _digest(f"v6/{object_id}/{variant_id}/guard"),
        "guard_threshold": None if baseline else 0.2,
        "covariance_artifact_id": _digest(f"v6/{object_id}/{variant_id}/covariance"),
        "interval_artifact_id": _digest(f"v6/{object_id}/{variant_id}/interval"),
        "risk_score": None if baseline else 0.1,
        "unavailable_reason": None,
    }


def _variants(batch, object_id: str, cohort):
    held_out = _held_out(batch, object_id)
    variants = {
        B0: _variant(
            B0,
            object_id=object_id,
            cohort=cohort,
            prediction_id=held_out["methods"]["B0_physical_fallback"]["artifact_id"],
        ),
        B1: _variant(
            B1,
            object_id=object_id,
            cohort=cohort,
            prediction_id=held_out["methods"]["B1_last_causal_residual"]["artifact_id"],
        ),
        D1_NATIVE: _variant(
            D1_NATIVE,
            object_id=object_id,
            cohort=cohort,
            prediction_id=_digest(f"v6/{object_id}/d1/prediction"),
        ),
    }
    shared_prediction = _digest(f"v6/{object_id}/vt1/prediction")
    shared_fit = _digest(f"v6/{object_id}/vt1/fit")
    shared_guard = _digest(f"v6/{object_id}/vt1/guard")
    for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH):
        row = _variant(
            variant_id,
            object_id=object_id,
            cohort=cohort,
            prediction_id=shared_prediction,
        )
        row["fit_artifact_id"] = shared_fit
        row["guard_artifact_id"] = shared_guard
        variants[variant_id] = row
    assert set(variants) == set(VARIANT_IDS)
    return variants


def _bundle():
    policy, covariance, execution, selection, cohort, v5_lock = _locks()
    batch = _v5_batch(v5_lock, cohort)
    panels = [
        build_deform360_v6_source_candidate_panel(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            implementation_revision=REVISION,
            object_id=object_id,
            variants=_variants(batch, object_id, cohort),
            source_artifacts={
                f"candidate/{object_id}.json": _digest(f"candidate/{object_id}")
            },
        )
        for object_id in sorted(cohort)
    ]
    return policy, covariance, execution, selection, cohort, v5_lock, batch, panels


def test_bridge_builds_exact_target_closed_ten_unit_batch() -> None:
    policy, covariance, execution, selection, cohort, v5_lock, batch, panels = _bundle()

    seals, v6_batch, receipt = bridge_deform360_v6_source_prediction_batch(
        policy=policy,
        covariance_amendment=covariance,
        source_execution_amendment=execution,
        selection=selection,
        v5_execution_lock=v5_lock,
        v5_prediction_batch=batch,
        candidate_panels=list(reversed(panels)),
        bridge_revision=BRIDGE_REVISION,
    )

    assert len(seals) == v6_batch["record_count"] == receipt["record_count"] == 10
    assert v6_batch["implementation_revision"] == REVISION
    assert receipt["bridge_revision"] == BRIDGE_REVISION
    assert receipt["v5_prediction_batch_id"] == batch["prediction_batch_id"]
    assert receipt["v6_prediction_batch_id"] == v6_batch["prediction_batch_id"]
    assert all(value is False for value in receipt["information_boundary"].values())
    assert validate_deform360_v6_prediction_bridge_receipt(receipt) == receipt
    by_object = {row["object_id"]: row for row in v6_batch["records"]}
    assert set(by_object) == set(cohort)
    for object_id, row in by_object.items():
        held_out = _held_out(batch, object_id)
        assert (
            row["variants"][B0]["prediction_artifact_id"]
            == held_out["methods"]["B0_physical_fallback"]["artifact_id"]
        )
        assert (
            row["variants"][B1]["prediction_artifact_id"]
            == held_out["methods"]["B1_last_causal_residual"]["artifact_id"]
        )


def test_candidate_panel_round_trip_and_tamper_detection() -> None:
    policy, covariance, execution, selection, _, v5_lock, batch, panels = _bundle()
    panel = panels[0]

    assert (
        validate_deform360_v6_source_candidate_panel(
            panel,
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
        )
        == panel
    )
    changed = copy.deepcopy(panel)
    changed["variants"][D1_NATIVE]["risk_score"] = 0.15
    with pytest.raises(ValueError, match="candidate panel content changed"):
        validate_deform360_v6_source_candidate_panel(
            changed,
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
        )


def test_baseline_substitution_is_rejected() -> None:
    policy, covariance, execution, selection, cohort, v5_lock, batch, _ = _bundle()
    object_id = sorted(cohort)[0]
    variants = _variants(batch, object_id, cohort)
    variants[B0]["prediction_artifact_id"] = _digest("substituted")

    with pytest.raises(ValueError, match="sealed v5 held-out method"):
        build_deform360_v6_source_candidate_panel(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            implementation_revision=REVISION,
            object_id=object_id,
            variants=variants,
            source_artifacts={"candidate.json": _digest("candidate")},
        )


def test_d1_unavailability_and_vt1_mean_drift_fail_closed() -> None:
    policy, covariance, execution, selection, cohort, v5_lock, batch, _ = _bundle()
    object_id = sorted(cohort)[0]
    variants = _variants(batch, object_id, cohort)
    variants[D1_NATIVE] = _variant(
        D1_NATIVE,
        object_id=object_id,
        cohort=cohort,
        prediction_id=_digest("unused"),
        available=False,
    )
    with pytest.raises(ValueError, match="D1 native covariance variant"):
        build_deform360_v6_source_candidate_panel(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            implementation_revision=REVISION,
            object_id=object_id,
            variants=variants,
            source_artifacts={"candidate.json": _digest("candidate")},
        )

    variants = _variants(batch, object_id, cohort)
    variants[VT1_OBSERVED]["prediction_artifact_id"] = _digest("different-mean")
    with pytest.raises(ValueError, match="share one mean, fit, and guard"):
        build_deform360_v6_source_candidate_panel(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            implementation_revision=REVISION,
            object_id=object_id,
            variants=variants,
            source_artifacts={"candidate.json": _digest("candidate")},
        )


def test_revision_roster_and_panel_count_are_fail_closed() -> None:
    policy, covariance, execution, selection, cohort, v5_lock, batch, panels = _bundle()
    object_id = sorted(cohort)[0]
    with pytest.raises(ValueError, match="revision differs"):
        build_deform360_v6_source_candidate_panel(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            implementation_revision="c" * 40,
            object_id=object_id,
            variants=_variants(batch, object_id, cohort),
            source_artifacts={"candidate.json": _digest("candidate")},
        )

    with pytest.raises(ValueError, match="exactly ten"):
        bridge_deform360_v6_source_prediction_batch(
            policy=policy,
            covariance_amendment=covariance,
            source_execution_amendment=execution,
            selection=selection,
            v5_execution_lock=v5_lock,
            v5_prediction_batch=batch,
            candidate_panels=panels[:-1],
            bridge_revision=BRIDGE_REVISION,
        )


def test_publication_is_atomic_and_no_clobber(tmp_path: Path) -> None:
    policy, covariance, execution, selection, _, v5_lock, batch, panels = _bundle()
    seals, v6_batch, receipt = bridge_deform360_v6_source_prediction_batch(
        policy=policy,
        covariance_amendment=covariance,
        source_execution_amendment=execution,
        selection=selection,
        v5_execution_lock=v5_lock,
        v5_prediction_batch=batch,
        candidate_panels=panels,
        bridge_revision=BRIDGE_REVISION,
    )

    output = publish_deform360_v6_prediction_bridge(
        seals=seals,
        batch=v6_batch,
        receipt=receipt,
        output_directory=tmp_path / "bridge",
    )
    assert json.loads((output / "bridge-receipt.json").read_text()) == receipt
    assert len(list((output / "source-seals").glob("*.json"))) == 10
    with pytest.raises(FileExistsError):
        publish_deform360_v6_prediction_bridge(
            seals=seals,
            batch=v6_batch,
            receipt=receipt,
            output_directory=output,
        )


def test_bridge_cli_publishes_only_target_closed_artifacts(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    _, _, _, _, _, _, batch, panels = _bundle()
    batch_path = tmp_path / "v5-batch.json"
    batch_path.write_text(json.dumps(batch, sort_keys=True), encoding="utf-8")
    panel_paths = []
    for index, panel in enumerate(panels):
        path = tmp_path / f"panel-{index:02d}.json"
        path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")
        panel_paths.append(path)
    command = [
        sys.executable,
        str(
            ROOT / "scripts/science/"
            "run_deform360_fresh_object_session_v6_prediction_bridge.py"
        ),
        "bridge",
        "--policy",
        str(POLICY_PATH),
        "--covariance-amendment",
        str(COVARIANCE_PATH),
        "--source-execution-amendment",
        str(EXECUTION_PATH),
        "--selection",
        str(SELECTION_PATH),
        "--v5-execution-lock",
        str(V5_LOCK_PATH),
        "--v5-prediction-batch",
        str(batch_path),
        "--bridge-revision",
        BRIDGE_REVISION,
        "--output-directory",
        str(tmp_path / "output"),
    ]
    for path in panel_paths:
        command.extend(("--candidate-panel", str(path)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    summary = json.loads(completed.stdout)
    assert summary["record_count"] == 10
    assert all(value is False for value in summary["information_boundary"].values())
    assert (tmp_path / "output/source-prediction-batch.json").is_file()
    assert not (tmp_path / "output/source-evidence.json").exists()
    assert not (tmp_path / "output/source-result.json").exists()


def test_bridge_script_is_packaged_and_contract_workflow_is_data_closed() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github/workflows/deform360-fresh-object-session-v6-contracts.yml"
    ).read_text(encoding="utf-8")

    assert (
        "include scripts/science/"
        "run_deform360_fresh_object_session_v6_prediction_bridge.py" in manifest
    )
    assert "runs-on: [self-hosted" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "source-evidence.json" not in workflow
    assert "source-result.json" not in workflow
    assert "v6_prediction_bridge.py" in workflow
    assert "test_deform360_fresh_object_session_v6_prediction_bridge.py" in workflow


@pytest.fixture(scope="module")
def bridge_inputs():
    return _bundle()


@pytest.fixture(scope="module")
def bridge_outputs(bridge_inputs):
    return _bridge_from_bundle(bridge_inputs)


def _reidentify(payload, *, id_field: str):
    from bayesian_phystwin._portable_contracts import content_id

    result = copy.deepcopy(payload)
    result[id_field] = content_id(
        {key: value for key, value in result.items() if key != id_field}
    )
    return result


def _write_payload(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _build_panel_from_bundle(bundle, **overrides):
    policy, covariance, execution, selection, _, v5_lock, batch, panels = bundle
    panel = panels[0]
    arguments = {
        "policy": policy,
        "covariance_amendment": covariance,
        "source_execution_amendment": execution,
        "selection": selection,
        "v5_execution_lock": v5_lock,
        "v5_prediction_batch": batch,
        "implementation_revision": REVISION,
        "object_id": panel["object_id"],
        "variants": copy.deepcopy(panel["variants"]),
        "source_artifacts": copy.deepcopy(panel["source_artifacts"]),
    }
    arguments.update(overrides)
    return build_deform360_v6_source_candidate_panel(**arguments)


def _bridge_from_bundle(bundle, *, candidate_panels=None):
    policy, covariance, execution, selection, _, v5_lock, batch, panels = bundle
    return bridge_deform360_v6_source_prediction_batch(
        policy=policy,
        covariance_amendment=covariance,
        source_execution_amendment=execution,
        selection=selection,
        v5_execution_lock=v5_lock,
        v5_prediction_batch=batch,
        candidate_panels=panels if candidate_panels is None else candidate_panels,
        bridge_revision=BRIDGE_REVISION,
    )


def test_source_execution_amendment_rejects_every_contract_drift(
    tmp_path: Path,
    monkeypatch,
    bridge_inputs,
) -> None:
    from bayesian_phystwin import (
        deform360_fresh_object_session_v6_prediction_bridge as bridge_module,
    )

    _, _, execution, _, _, v5_lock, _, _ = bridge_inputs
    lock_id = v5_lock["execution_lock_id"]

    changed = copy.deepcopy(execution)
    changed["schema"] = "changed"
    with pytest.raises(ValueError, match="schema changed"):
        load_deform360_v6_source_execution_amendment(
            _write_payload(tmp_path, "schema.json", changed),
            v5_execution_lock_id=lock_id,
        )

    changed = copy.deepcopy(execution)
    changed["execution"]["v6_source_prediction_unit_count"] = 11
    with pytest.raises(ValueError, match="content identity changed"):
        load_deform360_v6_source_execution_amendment(
            _write_payload(tmp_path, "stale-identity.json", changed),
            v5_execution_lock_id=lock_id,
        )

    changed = copy.deepcopy(execution)
    changed["execution"]["v6_source_prediction_unit_count"] = 11
    changed = _reidentify(changed, id_field="amendment_id")
    with pytest.raises(ValueError, match="identity changed"):
        load_deform360_v6_source_execution_amendment(
            _write_payload(tmp_path, "other-identity.json", changed),
            v5_execution_lock_id=lock_id,
        )

    with pytest.raises(ValueError, match="binds another v5 lock"):
        load_deform360_v6_source_execution_amendment(
            EXECUTION_PATH,
            v5_execution_lock_id="0" * 64,
        )

    changed = copy.deepcopy(execution)
    changed["information_boundary"]["development_suffix_opened"] = True
    changed = _reidentify(changed, id_field="amendment_id")
    monkeypatch.setattr(
        bridge_module,
        "SOURCE_EXECUTION_AMENDMENT_ID",
        changed["amendment_id"],
    )
    with pytest.raises(ValueError, match="crosses its boundary"):
        load_deform360_v6_source_execution_amendment(
            _write_payload(tmp_path, "boundary.json", changed),
            v5_execution_lock_id=lock_id,
        )

    changed = copy.deepcopy(execution)
    changed["execution"]["v6_source_prediction_unit_count"] = 9
    changed = _reidentify(changed, id_field="amendment_id")
    monkeypatch.setattr(
        bridge_module,
        "SOURCE_EXECUTION_AMENDMENT_ID",
        changed["amendment_id"],
    )
    with pytest.raises(ValueError, match="counts changed"):
        load_deform360_v6_source_execution_amendment(
            _write_payload(tmp_path, "counts.json", changed),
            v5_execution_lock_id=lock_id,
        )


def test_held_out_v5_record_guards_reject_inconsistency_duplicates_and_count() -> None:
    from bayesian_phystwin import (
        deform360_fresh_object_session_v6_prediction_bridge as bridge_module,
    )

    with pytest.raises(ValueError, match="inconsistent outer identity"):
        bridge_module._held_out_v5_records(
            {
                "records": [
                    {
                        "record_role": "held_out",
                        "object_id": "object-a",
                        "outer_held_out_object_id": "object-b",
                    }
                ]
            }
        )

    repeated = {
        "record_role": "held_out",
        "object_id": "object-a",
        "outer_held_out_object_id": "object-a",
    }
    with pytest.raises(ValueError, match="repeats a held-out unit"):
        bridge_module._held_out_v5_records(
            {"records": [copy.deepcopy(repeated), copy.deepcopy(repeated)]}
        )

    with pytest.raises(ValueError, match="must contain ten held-out records"):
        bridge_module._held_out_v5_records({"records": [repeated]})


def test_candidate_panel_allows_explicitly_unavailable_vt1(
    bridge_inputs,
) -> None:
    policy, covariance, execution, selection, cohort, v5_lock, batch, panels = (
        bridge_inputs
    )
    panel = panels[0]
    object_id = panel["object_id"]
    variants = copy.deepcopy(panel["variants"])
    for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH):
        variants[variant_id] = _variant(
            variant_id,
            object_id=object_id,
            cohort=cohort,
            prediction_id=_digest(f"unused/{variant_id}"),
            available=False,
        )

    rebuilt = build_deform360_v6_source_candidate_panel(
        policy=policy,
        covariance_amendment=covariance,
        source_execution_amendment=execution,
        selection=selection,
        v5_execution_lock=v5_lock,
        v5_prediction_batch=batch,
        implementation_revision=REVISION,
        object_id=object_id,
        variants=variants,
        source_artifacts=panel["source_artifacts"],
    )

    assert all(
        rebuilt["variants"][variant_id]["available"] is False
        for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH)
    )


def test_candidate_builder_rejects_cross_contract_inputs(
    monkeypatch,
    bridge_inputs,
) -> None:
    from bayesian_phystwin import (
        deform360_fresh_object_session_v6_prediction_bridge as bridge_module,
    )

    _, covariance, execution, _, _, _, batch, _ = bridge_inputs

    changed = copy.deepcopy(execution)
    changed["amendment_id"] = "0" * 64
    with pytest.raises(ValueError, match="another source execution amendment"):
        _build_panel_from_bundle(
            bridge_inputs,
            source_execution_amendment=changed,
        )

    changed = copy.deepcopy(execution)
    changed["v5_source_execution_lock_id"] = "0" * 64
    with pytest.raises(ValueError, match="source amendment binds another v5 lock"):
        _build_panel_from_bundle(
            bridge_inputs,
            source_execution_amendment=changed,
        )

    changed = copy.deepcopy(covariance)
    changed["amendment_id"] = "0" * 64
    with pytest.raises(ValueError, match="another covariance amendment"):
        _build_panel_from_bundle(
            bridge_inputs,
            covariance_amendment=changed,
        )

    with pytest.raises(ValueError, match="unregistered source unit"):
        _build_panel_from_bundle(
            bridge_inputs,
            object_id="unregistered-object",
        )

    changed_batch = copy.deepcopy(batch)
    held_out = next(
        row for row in changed_batch["records"] if row["record_role"] == "held_out"
    )
    held_out["object_id"] = "unregistered-object"
    held_out["outer_held_out_object_id"] = "unregistered-object"
    with monkeypatch.context() as patch:
        patch.setattr(
            bridge_module,
            "validate_deform360_joint_sparse_source_prediction_batch_v5",
            lambda *_: changed_batch,
        )
        with pytest.raises(ValueError, match="held-out roster differs"):
            _build_panel_from_bundle(bridge_inputs)


def test_candidate_validator_rejects_nonobjects_schema_and_boundary(
    bridge_inputs,
) -> None:
    policy, covariance, execution, selection, _, v5_lock, batch, panels = bridge_inputs
    validation = {
        "policy": policy,
        "covariance_amendment": covariance,
        "source_execution_amendment": execution,
        "selection": selection,
        "v5_execution_lock": v5_lock,
        "v5_prediction_batch": batch,
    }

    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_deform360_v6_source_candidate_panel([], **validation)

    changed = copy.deepcopy(panels[0])
    changed["schema"] = "changed"
    with pytest.raises(ValueError, match="schema changed"):
        validate_deform360_v6_source_candidate_panel(changed, **validation)

    changed = copy.deepcopy(panels[0])
    changed["information_boundary"]["v6_target_selected"] = True
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_v6_source_candidate_panel(changed, **validation)


def test_bridge_rejects_duplicate_incomplete_and_mixed_panels(
    monkeypatch,
    bridge_inputs,
) -> None:
    from bayesian_phystwin import (
        deform360_fresh_object_session_v6_prediction_bridge as bridge_module,
    )

    panels = bridge_inputs[-1]
    duplicated = [*panels[:-1], panels[0]]
    with pytest.raises(ValueError, match="repeats a candidate panel"):
        _bridge_from_bundle(bridge_inputs, candidate_panels=duplicated)

    def accept_panel(value, **_):
        return value

    with monkeypatch.context() as patch:
        patch.setattr(
            bridge_module,
            "validate_deform360_v6_source_candidate_panel",
            accept_panel,
        )

        incomplete = copy.deepcopy(panels)
        incomplete[-1]["object_id"] = "unregistered-object"
        with pytest.raises(ValueError, match="incomplete source roster"):
            _bridge_from_bundle(bridge_inputs, candidate_panels=incomplete)

        mixed = copy.deepcopy(panels)
        mixed[-1]["implementation_revision"] = "c" * 40
        with pytest.raises(ValueError, match="mixes candidate revisions"):
            _bridge_from_bundle(bridge_inputs, candidate_panels=mixed)


def test_bridge_receipt_and_publication_guards(
    tmp_path: Path,
    bridge_outputs,
) -> None:
    seals, v6_batch, receipt = bridge_outputs

    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_deform360_v6_prediction_bridge_receipt([])

    changed = copy.deepcopy(receipt)
    changed["schema"] = "changed"
    with pytest.raises(ValueError, match="schema changed"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["schema_version"] = 2
    with pytest.raises(ValueError, match="version changed"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["information_boundary"]["v6_target_selected"] = True
    with pytest.raises(ValueError, match="information boundary"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["record_count"] = 9
    with pytest.raises(ValueError, match="record count changed"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["candidate_panel_ids"] = "not-an-array"
    with pytest.raises(ValueError, match="must be a JSON array"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["candidate_panel_ids"] = changed["candidate_panel_ids"][:-1]
    with pytest.raises(ValueError, match="roster is incomplete"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["bridge_receipt_id"] = "0" * 64
    with pytest.raises(ValueError, match="content identity changed"):
        validate_deform360_v6_prediction_bridge_receipt(changed)

    with pytest.raises(ValueError, match="requires ten seals"):
        publish_deform360_v6_prediction_bridge(
            seals=seals[:-1],
            batch=v6_batch,
            receipt=receipt,
            output_directory=tmp_path / "too-few",
        )

    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="ordinary directory"):
        publish_deform360_v6_prediction_bridge(
            seals=seals,
            batch=v6_batch,
            receipt=receipt,
            output_directory=output_file,
        )
