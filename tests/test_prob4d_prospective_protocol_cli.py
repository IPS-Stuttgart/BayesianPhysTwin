from __future__ import annotations

import json
from pathlib import Path

from test_prob4d_prospective_protocol import (
    _configuration,
    _passing_statistics,
    _result,
)

from bayesian_phystwin.cli.command_registry import (
    CommandStatus,
    find_command_metadata,
)
from bayesian_phystwin.cli.prob4d_prospective_protocol import main
from bayesian_phystwin.prob4d_prospective_protocol import (
    load_prob4d_prospective_protocol,
)


def test_registered_command_has_current_experiment_lifecycle() -> None:
    command = find_command_metadata("prob4d-prospective-protocol")

    assert command is not None
    assert command.status is CommandStatus.EXPERIMENT
    assert command.owner == "prob4d-bpt-prospective-v1"
    assert command.canonical_command == (
        "bpt experiment run prob4d-prospective-protocol"
    )


def test_cli_freezes_validates_and_decides_protocol(tmp_path: Path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    configuration = tmp_path / "configuration.json"
    configuration.write_text(
        json.dumps(_configuration(artifact_root)),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"

    assert main(["freeze", str(configuration), str(protocol_path)]) == 0
    frozen = json.loads(capsys.readouterr().out)
    protocol = load_prob4d_prospective_protocol(protocol_path)
    assert frozen["protocol_sha256"] == protocol.protocol_sha256

    readiness_path = tmp_path / "readiness.json"
    assert (
        main(
            [
                "validate",
                str(protocol_path),
                "--artifact-root",
                str(artifact_root),
                "--output",
                str(readiness_path),
                "--require-ready",
            ]
        )
        == 0
    )
    readiness = json.loads(capsys.readouterr().out)
    assert readiness["ready_for_target_opening"] is True
    assert json.loads(readiness_path.read_text(encoding="utf-8")) == readiness

    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            _result(
                protocol.protocol_sha256,
                statistics=_passing_statistics(),
                physical_update={
                    "fallback_method_id": "physical_baseline",
                    "fallback_exact": True,
                    "evaluated_group_count": 2,
                    "accepted_update_count": 1,
                    "harmful_accepted_update_count": 0,
                },
            )
        ),
        encoding="utf-8",
    )
    decision_path = tmp_path / "decision.json"
    assert (
        main(
            [
                "decide",
                str(protocol_path),
                str(result_path),
                str(decision_path),
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["prob4d_supported_feeder"] is True
    assert decision["causal4d_evaluation_admissible"] is True
    assert json.loads(decision_path.read_text(encoding="utf-8")) == decision


def test_cli_require_ready_returns_distinct_failure_code(
    tmp_path: Path,
    capsys,
) -> None:
    artifact_root = tmp_path / "artifacts"
    configuration = tmp_path / "configuration.json"
    configuration.write_text(
        json.dumps(_configuration(artifact_root)),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    assert main(["freeze", str(configuration), str(protocol_path)]) == 0
    capsys.readouterr()
    (artifact_root / "method_freeze.json").write_text("tampered\n", encoding="utf-8")

    assert (
        main(
            [
                "validate",
                str(protocol_path),
                "--artifact-root",
                str(artifact_root),
                "--require-ready",
            ]
        )
        == 3
    )
    readiness = json.loads(capsys.readouterr().out)
    assert readiness["ready_for_target_opening"] is False
