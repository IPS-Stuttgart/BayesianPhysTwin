import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols" / "anytime_valid_dlo_replay_v1.json"
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "remote"
    / "run_anytime_valid_dlo_replay_v1.py"
)


def cases(prefix: str, count: int, relative_improvement: float) -> list[dict[str, object]]:
    baseline = 1.0
    candidate = baseline * (1.0 - relative_improvement)
    return [
        {
            "name": f"{prefix}-{index:02d}",
            "candidate_l1_m": candidate,
            "baseline_l1_m": baseline,
        }
        for index in range(count)
    ]


def test_protocol_discloses_retrospective_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["contract"] == "anytime-valid-dlo-retrospective-replay-v1"
    assert protocol["information_boundary"]["outcomes_previously_opened"] is True
    assert protocol["information_boundary"]["retrospective_replay"] is True
    assert (
        protocol["information_boundary"]["fresh_validation_claim_authorized"]
        is False
    )
    assert (
        protocol["information_boundary"]["deployment_safety_claim_authorized"]
        is False
    )
    assert protocol["information_boundary"]["paper_claim_authorized"] is False


def test_synthetic_terminal_artifacts_reproduce_registered_mechanism(
    tmp_path: Path,
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dlo45_contract = protocol["artifacts"]["dlo45_prospective_target"]
    hierarchy = protocol["artifacts"]["hierarchical_transfer"]
    dlo45_root = tmp_path / "dlo45"
    hierarchy_root = tmp_path / "hierarchy"
    output_root = tmp_path / "output"
    dlo45_root.mkdir()
    hierarchy_root.mkdir()

    (dlo45_root / "result.json").write_text(
        json.dumps(
            {
                "procedure": {
                    "cases": cases(
                        "dlo45",
                        dlo45_contract["expected_case_count"],
                        dlo45_contract["expected_relative_improvement"],
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    (hierarchy_root / "result.json").write_text(
        json.dumps(
            {
                "pyelastica": {
                    "cases": cases(
                        "pyelastica",
                        hierarchy["pyelastica"]["expected_case_count"],
                        hierarchy["pyelastica"][
                            "expected_relative_improvement"
                        ],
                    )
                },
                "cross_operator": {
                    "cases": cases(
                        "cross",
                        hierarchy["cross_operator"]["expected_case_count"],
                        hierarchy["cross_operator"][
                            "expected_relative_improvement"
                        ],
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--protocol",
            str(PROTOCOL),
            "--dlo45-root",
            str(dlo45_root),
            "--hierarchy-root",
            str(hierarchy_root),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert result["decision"] == "retrospective-anytime-dlo-mechanism-supported"
    assert result["mechanism_gate"]["passed"] is True
    procedure = result["streams"]["procedure_replication"]
    universal = result["streams"]["universal_coefficient_transport"]
    assert procedure["sign_e_process"]["first_crossing_observation"] == 9
    assert procedure["guard"]["harmful_candidate_deployment_count"] == 0
    assert universal["candidate_active_at_shift_boundary"] is False
    assert universal["cross_operator_harmful_candidate_deployments"] == 0
    assert result["exact_fallback_identity_violations"] == 0


def test_runner_binds_known_terminal_artifact_identities() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert (
        protocol["artifacts"]["dlo45_prospective_target"]["artifact_id"]
        == 9811788200
    )
    assert (
        protocol["artifacts"]["hierarchical_transfer"]["artifact_id"]
        == 9811886089
    )
    assert protocol["streams"]["universal_coefficient_transport"][
        "shift_boundary_after_case"
    ] == 8


def test_changed_aggregate_cannot_be_silently_selected(tmp_path: Path) -> None:
    changed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    changed["artifacts"]["dlo45_prospective_target"][
        "expected_relative_improvement"
    ] = 0.5
    changed_protocol = tmp_path / "changed.json"
    changed_protocol.write_text(json.dumps(changed), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--protocol",
            str(changed_protocol),
            "--dlo45-root",
            str(tmp_path),
            "--hierarchy-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "output" / "result.json").exists()


def test_protocol_expected_counts_are_consistent() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dlo45 = protocol["artifacts"]["dlo45_prospective_target"]
    hierarchy = protocol["artifacts"]["hierarchical_transfer"]

    assert dlo45["expected_wins"] + dlo45["expected_ties"] + dlo45[
        "expected_losses"
    ] == dlo45["expected_case_count"]
    for label in ("pyelastica", "cross_operator"):
        part = hierarchy[label]
        assert part["expected_wins"] + part["expected_ties"] + part[
            "expected_losses"
        ] == part["expected_case_count"]
    assert hierarchy["cross_operator"]["expected_losses"] == 28
    assert hierarchy["pyelastica"]["expected_wins"] == 8


@pytest.mark.parametrize("key", ["fresh_validation_claim_authorized", "paper_claim_authorized"])
def test_no_claim_promotion_flag_is_true(key: str) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["information_boundary"][key] is False
