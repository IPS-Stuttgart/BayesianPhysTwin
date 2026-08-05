from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesian_phystwin.cli import pokeflex_public_evaluation as cli
from bayesian_phystwin.pokeflex_independent_depth_protocol import (
    load_pokeflex_independent_depth_protocol,
)


def test_contract_profile_reports_custody_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--profile", "contracts"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_kind"] == "PokeFlexPublicEvaluationContract"
    assert payload["authorized_profiles"] == ["contracts", "source-validation"]
    assert payload["authorized_take_count"] == 20
    assert payload["causal_history"] == "f-5 through f-1 only"
    assert payload["target_geometry_role"] == "scoring only"
    assert payload["replacement_allowed"] is False
    assert len(payload["analysis_runner_sha256"]) == 64
    assert "retrospective/exploratory" in payload["claim_boundary"]


def test_source_profile_runs_only_registered_source_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_pokeflex_independent_depth_protocol(cli._default_source_protocol())
    take_id = cli._expected_take_ids(source)[0]
    dataset_root = tmp_path / "dataset"
    (dataset_root / take_id).mkdir(parents=True)
    upstream = tmp_path / "upstream"
    (upstream / "models").mkdir(parents=True)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    output = tmp_path / "output"

    expected_upstream = source["payload"]["upstream"]["code_commit"]
    monkeypatch.setattr(
        cli,
        "_git_head",
        lambda path: (
            expected_upstream if path == upstream.resolve() else "bpt-revision"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_verify_checkpoint_files",
        lambda *_: {"checkpoint.pth": "checkpoint-sha"},
    )

    captured: list[str] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        captured.extend(command)
        progress = {
            "schema_version": 1,
            "artifact_kind": "PokeFlexIndependentDepthSourceValidationProgress",
            "protocol_sha256": source["protocol_sha256"],
            "replacement_allowed": False,
            "records": [
                {
                    "take_id": take_id,
                    "status": "failed-no-replacement",
                    "error": "synthetic technical failure",
                }
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "source_validation_progress_v2.json").write_text(
            json.dumps(progress), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert (
        cli.main(
            [
                "--profile",
                "source-validation",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output),
                "--upstream-checkout",
                str(upstream),
                "--checkpoint-root",
                str(checkpoints),
                "--take-id",
                take_id,
            ]
        )
        == 1
    )
    assert "--take-id" in captured
    assert take_id in captured
    manifest = json.loads((output / "execution_manifest.json").read_text())
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert manifest["selected_take_ids"] == [take_id]
    assert manifest["replacement_allowed"] is False
    assert summary["record_count"] == 1
    assert summary["status_counts"]["failed-no-replacement"] == 1
    assert summary["run_complete"] is False
    assert summary["analysis_status"] == "not-run-incomplete"


def test_complete_source_panel_emits_compact_frozen_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_pokeflex_independent_depth_protocol(cli._default_source_protocol())
    take_ids = cli._expected_take_ids(source)
    dataset_root = tmp_path / "dataset"
    for take_id in take_ids:
        (dataset_root / take_id).mkdir(parents=True)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    output = tmp_path / "output"

    expected_upstream = source["payload"]["upstream"]["code_commit"]
    monkeypatch.setattr(
        cli,
        "_git_head",
        lambda path: (
            expected_upstream if path == upstream.resolve() else "bpt-revision"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_verify_checkpoint_files",
        lambda *_: {"checkpoint.pth": "checkpoint-sha"},
    )

    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        output.mkdir(parents=True, exist_ok=True)
        if Path(command[1]) == cli._source_runner():
            records = []
            for take_id in take_ids:
                artifact = output / f"{take_id}_full_template15mm_v2.json"
                artifact.write_text("{}\n", encoding="utf-8")
                records.append(
                    {
                        "take_id": take_id,
                        "status": "completed",
                        "output": str(artifact),
                    }
                )
            progress = {
                "schema_version": 1,
                "artifact_kind": ("PokeFlexIndependentDepthSourceValidationProgress"),
                "protocol_sha256": source["protocol_sha256"],
                "replacement_allowed": False,
                "records": records,
            }
            (output / "source_validation_progress_v2.json").write_text(
                json.dumps(progress), encoding="utf-8"
            )
        else:
            assert Path(command[1]) == cli._analysis_runner()
            analysis_output = Path(command[command.index("--output") + 1])
            analysis_output.write_text(
                json.dumps(
                    {
                        "object_balanced_selector": {
                            "baseline_mean_CD_UL1_mm": 4.6,
                            "selected_mean_CD_UL1_mm": 4.4,
                        },
                        "registered_gate": {"all_passed": False},
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert (
        cli.main(
            [
                "--profile",
                "source-validation",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output),
                "--upstream-checkout",
                str(upstream),
                "--checkpoint-root",
                str(checkpoints),
            ]
        )
        == 0
    )
    assert len(calls) == 2
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["run_complete"] is True
    assert summary["analysis_status"] == "completed"
    assert summary["registered_gate"] == {"all_passed": False}
    assert summary["object_balanced_selector"]["selected_mean_CD_UL1_mm"] == 4.4
    assert len(summary["analysis_sha256"]) == 64


def test_source_profile_rejects_take_outside_frozen_panel(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    upstream = tmp_path / "upstream"
    checkpoints = tmp_path / "checkpoints"
    for path in (dataset_root, upstream, checkpoints):
        path.mkdir()
    with pytest.raises(ValueError, match="outside the frozen source panel"):
        cli.main(
            [
                "--profile",
                "source-validation",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(tmp_path / "output"),
                "--upstream-checkout",
                str(upstream),
                "--checkpoint-root",
                str(checkpoints),
                "--take-id",
                "ReservedObject_T2",
            ]
        )


def test_source_profile_rejects_nonempty_output_root(tmp_path: Path) -> None:
    for name in ("dataset", "upstream", "checkpoints"):
        (tmp_path / name).mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be new or empty"):
        cli.main(
            [
                "--profile",
                "source-validation",
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-root",
                str(output),
                "--upstream-checkout",
                str(tmp_path / "upstream"),
                "--checkpoint-root",
                str(tmp_path / "checkpoints"),
            ]
        )
