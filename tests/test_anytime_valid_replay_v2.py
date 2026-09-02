import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "remote" / "run_anytime_valid_loss_replay_v2.py"
BASE_PROTOCOL = (
    REPOSITORY_ROOT / "protocols" / "anytime_valid_simulator_admission_v1.json"
)


def test_terminal_fallback_after_nonreentrant_revocation(tmp_path: Path) -> None:
    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["controller"].update(
        {
            "alpha": 0.2,
            "beta": 0.2,
            "allow_reentry": False,
            "lambdas": [0.5, 0.9],
        }
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    csv_path = tmp_path / "losses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "stream_id",
                "reveal_order",
                "candidate_loss",
                "fallback_loss",
            ),
        )
        writer.writeheader()
        reveal_order = 0
        for candidate_loss, fallback_loss, count in (
            (0.0, 1.0, 30),
            (1.0, 0.0, 30),
            (1.0, 0.0, 20),
        ):
            for _ in range(count):
                reveal_order += 1
                writer.writerow(
                    {
                        "stream_id": "shift-stream",
                        "reveal_order": reveal_order,
                        "candidate_loss": candidate_loss,
                        "fallback_loss": fallback_loss,
                    }
                )

    output = tmp_path / "output"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPOSITORY_ROOT / "src"),
            str(REPOSITORY_ROOT / "scripts" / "remote"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--protocol",
            str(protocol_path),
            "--paired-loss-csv",
            str(csv_path),
            "--output-root",
            str(output),
            "--candidate-id",
            "fixed-candidate-v1",
            "--evidence-class",
            "retrospective-replay",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    stream_result = result["streams"][0]
    assert stream_result["admissions"] == 1
    assert stream_result["revocations"] == 1
    assert stream_result["terminal_revoked"] is True
    assert stream_result["terminal_fallback_observations"] > 0
    assert result["exact_fallback_identity_violations"] == 0

    events = list(csv.DictReader((output / "events.csv").open(encoding="utf-8")))
    revocation_index = next(
        index for index, row in enumerate(events) if row["event"] == "revoke"
    )
    tail = events[revocation_index + 1 :]
    assert tail
    assert {row["event"] for row in tail} == {"terminal-fallback"}
    assert {row["deployed_method"] for row in tail} == {"fallback"}
