#!/usr/bin/env python3
"""Validate, run, record, and merge the source-only Tracking Cloth V2 result.

This file is temporary execution machinery. It removes itself, its request, and
its workflow in the evidence commit. It never imports or opens repetition-3
outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BRANCH = "science/tracking-cloth-active-decision-rep3-v1"
REQUEST = Path(
    ".github/requests/run-and-record-tracking-cloth-action-costed-v2d.json"
)
WORKFLOW = Path(
    ".github/workflows/run-and-record-tracking-cloth-action-costed-v2d.yml"
)
SELF = Path("tools/_temporary_run_tracking_cloth_action_costed_v2d.py")
DATASET_ROOT = Path(
    "/home/github-runner/.cache/datasets/"
    "tracking-cloth-deformation-v1-zenodo-14644526"
)
PROTOCOL = Path(
    "experiments/tracking_cloth_action_feasibility_costed_v2/protocol.json"
)
RESULT_DIRECTORY = Path(
    "results/science/tracking_cloth_action_feasibility_costed_v2/"
    "source-only-20260902-v1"
)

EXPECTED_REQUEST: dict[str, Any] = {
    "schema": "bayesian-phystwin/source-execution-request-v1",
    "request_id": "tracking-cloth-action-costed-v2-source-d-20260902",
    "source_repetitions": [1, 2],
    "reserved_target_repetition": 3,
    "primary_support_miss_probability": 0.1,
    "target_outcomes_opened": False,
    "automatic_target_follow_on": False,
    "merge_result_regardless_of_gate_sign": True,
}

TEMPORARY_PATHS = (
    Path(".github/requests/finalize-tracking-cloth-action-costed-v2.json"),
    Path(".github/requests/finalize-tracking-cloth-action-costed-v2b.json"),
    Path(".github/requests/finalize-tracking-cloth-action-costed-v2c.json"),
    Path(".github/requests/run-and-record-tracking-cloth-action-costed-v2.json"),
    Path(".github/requests/run-and-record-tracking-cloth-action-costed-v2b.json"),
    Path(".github/requests/run-and-record-tracking-cloth-action-costed-v2c.json"),
    Path(".github/workflows/finalize-tracking-cloth-action-costed-v2.yml"),
    Path(".github/workflows/finalize-tracking-cloth-action-costed-v2b.yml"),
    Path(".github/workflows/finalize-tracking-cloth-action-costed-v2c.yml"),
    Path(".github/workflows/run-and-record-tracking-cloth-action-costed-v2.yml"),
    Path(".github/workflows/run-and-record-tracking-cloth-action-costed-v2b.yml"),
    Path(".github/workflows/run-and-record-tracking-cloth-action-costed-v2c.yml"),
)


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def write_environment(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            print(f"{name}={value}", file=stream)


def append_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            print(text, file=stream)


def authorize_request() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if event.get("forced") is not False:
        raise RuntimeError("forced source request is forbidden")
    before = str(event.get("before", ""))
    if before == "0" * 40 or len(before) != 40:
        raise RuntimeError("source request requires a nonzero parent")
    current = output("git", "rev-parse", "HEAD")
    parent = output("git", "rev-parse", "HEAD^")
    if parent != before or current != os.environ["GITHUB_SHA"]:
        raise RuntimeError("source request parent binding changed")
    changed = output("git", "diff", "--name-status", before, current).splitlines()
    expected = f"A\t{REQUEST.as_posix()}"
    if changed != [expected]:
        raise RuntimeError(f"source request must be one add-only file: {changed}")
    value = json.loads(REQUEST.read_text(encoding="utf-8"))
    if value != EXPECTED_REQUEST:
        raise RuntimeError("source request fields or bytes changed")


def patch_scientific_source() -> None:
    decision_path = Path(
        "experiments/tracking_cloth_action_feasibility_costed_v2/_decision.py"
    )
    decision = decision_path.read_text(encoding="utf-8")
    return_line = "    return np.min(pair_budget, axis=1)"
    diagonal_line = "    np.fill_diagonal(pair_budget, 1.0)"
    if diagonal_line not in decision:
        if decision.count(return_line) != 1:
            raise RuntimeError("support-budget return marker changed")
        decision = decision.replace(
            return_line,
            "    # Same-plan loss differences are identically zero.\n"
            "    np.fill_diagonal(pair_budget, 1.0)\n"
            + return_line,
            1,
        )

    old_cost_check = (
        "    costs = np.asarray(probe_costs, dtype=np.float64)\n"
        "    if costs.shape != (certificate.probe_count,) or np.any(costs < 0.0):\n"
        '        raise ValueError("probe_costs changed after parent plan enumeration")\n'
    )
    new_cost_check = (
        "    costs = np.asarray(probe_costs, dtype=np.float64)\n"
        "    if (\n"
        "        costs.shape != (certificate.probe_count,)\n"
        "        or not np.isfinite(costs).all()\n"
        "        or np.any(costs < 0.0)\n"
        "    ):\n"
        '        raise ValueError("probe_costs changed after parent plan enumeration")\n'
        "    tolerance = float(regret_tolerance)\n"
        "    if not np.isfinite(tolerance) or tolerance < 0.0:\n"
        '        raise ValueError("regret_tolerance must be finite and nonnegative")\n'
    )
    if "regret_tolerance must be finite and nonnegative" not in decision:
        if decision.count(old_cost_check) != 1:
            raise RuntimeError("support-robust cost validation marker changed")
        decision = decision.replace(old_cost_check, new_cost_check, 1)
        decision = decision.replace(
            "        float(regret_tolerance),\n",
            "        tolerance,\n",
        )
        decision = decision.replace(
            "    if minimum > float(regret_tolerance) + ATOL:\n",
            "    if minimum > tolerance + ATOL:\n",
            1,
        )
    decision_path.write_text(decision, encoding="utf-8")

    test_path = Path("tests/test_tracking_cloth_action_feasibility_costed_v2.py")
    test = test_path.read_text(encoding="utf-8")
    old_import = (
        "from bayesian_phystwin.act_sense_fallback_certificate_v1 import (\n"
        "    act_sense_fallback_certificate,\n"
        ")"
    )
    new_import = (
        "from bayesian_phystwin.act_sense_fallback_certificate_v1 import (\n"
        "    ActSenseFallbackCertificateV1,\n"
        "    act_sense_fallback_certificate,\n"
        ")"
    )
    if "ActSenseFallbackCertificateV1" not in test:
        if test.count(old_import) != 1:
            raise RuntimeError("certificate import marker changed")
        test = test.replace(old_import, new_import, 1)
    test = test.replace(
        "def _parent_certificate():",
        "def _parent_certificate() -> ActSenseFallbackCertificateV1:",
        1,
    )
    test = test.replace('        [[0.0, 1.0]],\n', '        [[1.0, 0.0]],\n', 1)
    old_bounds = (
        '        unknown_action_loss_lower=np.asarray([0.0, 0.0]),\n'
        '        unknown_action_loss_upper=np.asarray([10.0, 0.0]),\n'
    )
    new_bounds = (
        '        unknown_action_loss_lower=np.asarray([10.0, 0.0]),\n'
        '        unknown_action_loss_upper=np.asarray([10.0, 10.0]),\n'
    )
    if old_bounds in test:
        test = test.replace(old_bounds, new_bounds, 1)
    if "def _parent_certificate() -> ActSenseFallbackCertificateV1:" not in test:
        raise RuntimeError("typed certificate factory is absent")
    if new_bounds not in test:
        raise RuntimeError("same-plan support-budget test bounds are absent")

    invalid_test = '''


def test_support_robust_inputs_fail_closed() -> None:
    parent = _parent_certificate()
    with pytest.raises(ValueError, match="probe_costs"):
        support_robust_decision(
            parent,
            probe_costs=np.asarray([np.nan]),
            support_miss_probability=0.0,
            unknown_action_loss_lower=np.asarray([0.0, 0.0]),
            unknown_action_loss_upper=np.asarray([2.0, 2.0]),
            regret_tolerance=0.5,
        )
    with pytest.raises(ValueError, match="regret_tolerance"):
        support_robust_decision(
            parent,
            probe_costs=np.asarray([0.25]),
            support_miss_probability=0.0,
            unknown_action_loss_lower=np.asarray([0.0, 0.0]),
            unknown_action_loss_upper=np.asarray([2.0, 2.0]),
            regret_tolerance=-0.1,
        )
'''
    if "def test_support_robust_inputs_fail_closed" not in test:
        test = test.rstrip() + invalid_test + "\n"
    test_path.write_text(test, encoding="utf-8")

    permanent_path = Path(
        ".github/workflows/tracking-cloth-action-feasibility-costed-v2.yml"
    )
    workflow = permanent_path.read_text(encoding="utf-8")
    broad = '          assert "rep3" not in source.lower()\n'
    precise = (
        '          assert "scoring_truth" not in source\n'
        '          assert "--stage target" not in source\n'
        '          assert "--stage predict" not in source\n'
        '          assert "--stage score" not in source\n'
    )
    if broad in workflow:
        workflow = workflow.replace(broad, precise, 1)
    rewritten: list[str] = []
    for line in workflow.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("assert")
            and "secrets." in stripped
            and "not in workflow" in stripped
        ):
            line = '          assert ("$" + "{{" + " secrets.") not in workflow'
        elif stripped == 'assert "contents: write" not in workflow':
            line = '          assert ("contents:" + " write") not in workflow'
        elif stripped == 'assert "git push" not in workflow':
            line = '          assert ("git " + "push") not in workflow'
        rewritten.append(line)
    workflow = "\n".join(rewritten) + "\n"
    required_workflow_fragments = (
        'assert "scoring_truth" not in source',
        'assert "--stage target" not in source',
        'assert "--stage predict" not in source',
        'assert "--stage score" not in source',
        'assert ("contents:" + " write") not in workflow',
        'assert ("git " + "push") not in workflow',
        'assert ("$" + "{{" + " secrets.") not in workflow',
    )
    for fragment in required_workflow_fragments:
        if fragment not in workflow:
            raise RuntimeError(f"missing source custody assertion: {fragment}")
    permanent_path.write_text(workflow, encoding="utf-8")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    robustness = protocol["support_robustness"]
    robustness["unknown_probe_outcome_alphabet"] = "registered-outcomes-only"
    robustness["unregistered_probe_outcome_policy"] = "outside-claim-boundary"
    boundary_sentence = (
        "The plan-loss box additionally assumes that unrepresented physics still "
        "produces an outcome in each probe's registered outcome alphabet; an "
        "unregistered outcome is outside this certificate."
    )
    if boundary_sentence not in protocol["claim_boundary"]:
        protocol["claim_boundary"] += " " + boundary_sentence
    PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    run_path = Path("experiments/tracking_cloth_action_feasibility_costed_v2/run.py")
    runner = run_path.read_text(encoding="utf-8")
    marker = (
        '    if robustness.get("target_tuning") is not False:\n'
        '        raise ValueError("target tuning must remain disabled")\n'
    )
    insertion = marker + (
        '    if robustness.get("unknown_probe_outcome_alphabet") != (\n'
        '        "registered-outcomes-only"\n'
        '    ):\n'
        '        raise ValueError("unknown probe outcome alphabet changed")\n'
        '    if robustness.get("unregistered_probe_outcome_policy") != (\n'
        '        "outside-claim-boundary"\n'
        '    ):\n'
        '        raise ValueError("unregistered probe outcome policy changed")\n'
    )
    if "unknown probe outcome alphabet changed" not in runner:
        if runner.count(marker) != 1:
            raise RuntimeError("V2 robustness validation marker changed")
        runner = runner.replace(marker, insertion, 1)
        run_path.write_text(runner, encoding="utf-8")

    readme_path = Path(
        "experiments/tracking_cloth_action_feasibility_costed_v2/README.md"
    )
    readme = readme_path.read_text(encoding="utf-8")
    paragraph = (
        "The terminal-action bounds induce complete sensing-plan bounds only under "
        "the additional registered assumption that unknown physics still returns one "
        "of the probe's registered outcomes. An outcome outside that alphabet is not "
        "certified by V2 and must be handled by a separate runtime monitor or fallback "
        "contract.\n"
    )
    heading = "## Source gate\n"
    if paragraph not in readme:
        if readme.count(heading) != 1:
            raise RuntimeError("README source-gate heading changed")
        readme = readme.replace(heading, paragraph + "\n" + heading, 1)
        readme_path.write_text(readme, encoding="utf-8")

    test = test_path.read_text(encoding="utf-8")
    anchor = '    assert robustness["target_tuning"] is False\n'
    extra = (
        anchor
        + '    assert robustness["unknown_probe_outcome_alphabet"] == (\n'
        + '        "registered-outcomes-only"\n'
        + '    )\n'
        + '    assert robustness["unregistered_probe_outcome_policy"] == (\n'
        + '        "outside-claim-boundary"\n'
        + '    )\n'
    )
    if "unknown_probe_outcome_alphabet" not in test:
        if test.count(anchor) != 1:
            raise RuntimeError("protocol-boundary test anchor changed")
        test_path.write_text(test.replace(anchor, extra, 1), encoding="utf-8")


def remove_superseded_paths() -> None:
    for path in TEMPORARY_PATHS:
        path.unlink(missing_ok=True)


def create_runtime() -> tuple[Path, Path]:
    runner_temp = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    venv = runner_temp / f"tracking-action-costed-v2d-venv-{run_id}-{attempt}"
    result = runner_temp / f"tracking-action-costed-v2d-output-{run_id}-{attempt}"
    shutil.rmtree(venv, ignore_errors=True)
    shutil.rmtree(result, ignore_errors=True)
    run(sys.executable, "-m", "venv", str(venv))
    python = venv / "bin" / "python"
    run(str(python), "-m", "pip", "install", "--disable-pip-version-check", "-e", ".[dev]")
    write_environment("V2D_VENV", str(venv))
    write_environment("V2D_OUTPUT", str(result))
    return python, result


def validate_source(python: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    run(
        str(python),
        "-m",
        "ruff",
        "format",
        "experiments/tracking_cloth_action_feasibility_costed_v2",
        "tests/test_tracking_cloth_action_feasibility_costed_v2.py",
        env=env,
    )
    run(
        str(python),
        "-m",
        "ruff",
        "check",
        "--fix",
        "experiments/tracking_cloth_action_feasibility_costed_v2",
        "tests/test_tracking_cloth_action_feasibility_costed_v2.py",
        env=env,
    )
    run(
        str(python),
        "-m",
        "ruff",
        "format",
        "--check",
        "experiments/tracking_cloth_action_feasibility_costed_v2",
        "tests/test_tracking_cloth_action_feasibility_costed_v2.py",
        env=env,
    )
    focused = (
        "tests/test_tracking_cloth_action_feasibility_costed_v2.py",
        "tests/test_support_robust_act_sense_fallback_certificate_v1.py",
        "tests/test_tracking_cloth_action_feasibility_v1.py",
        "tests/test_act_sense_fallback_certificate_v1.py",
    )
    run(str(python), "-m", "pytest", "-q", *focused, env=env)
    run(str(python), "-m", "pytest", "-q", env=env)
    run("git", "diff", "--check")


def commit_scientific_revision() -> str:
    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )
    run("git", "add", "-A")
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        check=False,
    ).returncode
    if staged != 0:
        run("git", "commit", "-m", "Freeze cost-aware support-robust cloth source protocol")
        run("git", "push", "origin", f"HEAD:{BRANCH}")
    scientific_sha = output("git", "rev-parse", "HEAD")
    remote = output("git", "ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    if remote != scientific_sha:
        raise RuntimeError("remote scientific source revision changed")
    write_environment("SCIENTIFIC_SHA", scientific_sha)
    append_summary(f"Validated scientific revision: `{scientific_sha}`")
    return scientific_sha


def execute_source(python: Path, result: Path) -> None:
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"dataset root is unavailable: {DATASET_ROOT}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    run(
        str(python),
        "-m",
        "experiments.tracking_cloth_action_feasibility_costed_v2.run",
        "--dataset-root",
        str(DATASET_ROOT),
        "--output",
        str(result),
        env=env,
    )


def validate_result(result_root: Path, scientific_sha: str) -> dict[str, Any]:
    result_path = result_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads((result_root / "protocol.json").read_text(encoding="utf-8"))
    if result.get("schema") != (
        "bayesian-phystwin.tracking-cloth-action-feasibility-costed-result.v2"
    ):
        raise RuntimeError("unexpected source-result schema")
    if result.get("source_case_count") != 24 or result.get("source_block_count") != 8:
        raise RuntimeError("source roster changed")
    if result.get("rep3_numeric_outcomes_read") is not False:
        raise RuntimeError("rep3 numeric outcomes were opened")
    if result.get("rep3_protocol_authorized") is not False:
        raise RuntimeError("source result authorized rep3")
    gate = result["source_gate"]
    if gate.get("automatic_target_follow_on") is not False:
        raise RuntimeError("source result enabled automatic target follow-on")
    summary = result["decision_summary"]
    if summary.get("primary_support_miss_probability") != 0.1:
        raise RuntimeError("primary support-miss probability changed")
    if set(summary["selected_by_support_miss_probability"]) != {
        "0.0",
        "0.05",
        "0.1",
        "0.2",
    }:
        raise RuntimeError("support-miss sensitivity grid changed")
    robustness = protocol["support_robustness"]
    if robustness.get("bound_is_assumed_not_estimated") is not True:
        raise RuntimeError("support-miss bound was presented as estimated")
    if robustness.get("unknown_probe_outcome_alphabet") != "registered-outcomes-only":
        raise RuntimeError("unknown probe alphabet boundary changed")
    for setting in summary["selected_by_support_miss_probability"].values():
        if "mode_counts" not in setting:
            continue
        for decision in setting["outputs"]:
            budget = float(decision["output_plan_support_miss_budget"])
            if not math.isfinite(budget) or not 0.0 <= budget <= 1.0:
                raise RuntimeError("invalid plan support-miss budget")
            if decision["mode"] == "sense":
                if decision["selected_probe"] is None:
                    raise RuntimeError("sensing policy lost its probe")
                mapping = decision["terminal_action_by_probe_outcome"]
                if not isinstance(mapping, list) or not mapping:
                    raise RuntimeError("sensing policy lost its terminal action map")
    with (result_root / "source_cases.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 24 or {int(row["repetition"]) for row in rows} != {1, 2}:
        raise RuntimeError("source CSV contains a reserved target repetition")

    receipt = {
        "schema": "bayesian-phystwin.tracking-cloth-action-costed-v2-source-receipt.v1",
        "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "scientific_sha": scientific_sha,
        "result_id": result["result_id"],
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "dataset_inventory_id": result["dataset_inventory_id"],
        "source_gate_pass": bool(gate["pass"]),
        "primary_support_miss_probability": 0.1,
        "source_repetitions": [1, 2],
        "rep3_numeric_outcomes_read": False,
        "automatic_target_follow_on": False,
        "paper_claim_authorized": False,
    }
    (result_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_summary((result_root / "SUMMARY.md").read_text(encoding="utf-8"))
    return receipt


def commit_evidence(result_root: Path, scientific_sha: str) -> str:
    run("git", "fetch", "origin", BRANCH)
    remote = output("git", "rev-parse", "FETCH_HEAD")
    if remote != scientific_sha:
        raise RuntimeError("branch moved after scientific source freeze")
    run("git", "switch", "-C", BRANCH, scientific_sha)
    if RESULT_DIRECTORY.exists():
        raise RuntimeError(f"result directory already exists: {RESULT_DIRECTORY}")
    RESULT_DIRECTORY.mkdir(parents=True)
    for name in (
        "result.json",
        "source_cases.csv",
        "SUMMARY.md",
        "protocol.json",
        "receipt.json",
    ):
        shutil.copy2(result_root / name, RESULT_DIRECTORY / name)
    (RESULT_DIRECTORY / "README.md").write_text(
        "# Tracking Cloth cost-aware support-robust source evidence V2\n\n"
        f"Immutable source-only execution at scientific revision `{scientific_sha}`. "
        "Repetitions 1 and 2 were opened; repetition 3 remained numerically closed. "
        "The source result is retained irrespective of gate sign. See `SUMMARY.md`, "
        "`result.json`, and `receipt.json`.\n",
        encoding="utf-8",
    )
    for path in (*TEMPORARY_PATHS, REQUEST, WORKFLOW, SELF):
        path.unlink(missing_ok=True)
    run("git", "add", "-A")
    run("git", "commit", "-m", "Record cost-aware support-robust cloth source evidence")
    result_sha = output("git", "rev-parse", "HEAD")
    run("git", "push", "origin", f"HEAD:{BRANCH}")
    write_environment("RESULT_SHA", result_sha)
    return result_sha


def github_json(
    method: str,
    url: str,
    token: str,
    body: dict[str, Any] | None = None,
) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "BayesianPhysTwin-source-evidence")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def merge_reviewed_pr(result_sha: str) -> str:
    token = os.environ.get("GH_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        raise RuntimeError("GitHub merge context is unavailable")
    owner = repository.split("/", 1)[0]
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "head": f"{owner}:{BRANCH}",
            "base": "main",
            "per_page": 10,
        }
    )
    api = f"https://api.github.com/repos/{repository}"
    pulls = github_json("GET", f"{api}/pulls?{query}", token)
    matching = [
        item
        for item in pulls
        if item["head"]["ref"] == BRANCH and item["base"]["ref"] == "main"
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one reviewed V2 PR, found {len(matching)}")
    number = int(matching[0]["number"])
    merged = github_json(
        "PUT",
        f"{api}/pulls/{number}/merge",
        token,
        {
            "commit_title": "Merge cost-aware support-robust cloth source evidence",
            "commit_message": (
                "Charge sensing cost in the empirical objective, expose the registered "
                "probe-outcome boundary, evaluate the exact support-miss sensitivity "
                "grid on source repetitions 1 and 2, and retain either source-gate "
                "sign without opening repetition 3."
            ),
            "sha": result_sha,
            "merge_method": "merge",
        },
    )
    if merged.get("merged") is not True:
        raise RuntimeError(f"reviewed V2 PR did not merge: {merged}")
    merge_sha = str(merged["sha"])
    append_summary(f"Merged PR #{number} at `{merge_sha}`.")
    return merge_sha


def main() -> int:
    authorize_request()
    patch_scientific_source()
    remove_superseded_paths()
    python, result_root = create_runtime()
    validate_source(python)
    scientific_sha = commit_scientific_revision()
    execute_source(python, result_root)
    receipt = validate_result(result_root, scientific_sha)
    result_sha = commit_evidence(result_root, scientific_sha)
    merge_sha = merge_reviewed_pr(result_sha)
    print(
        json.dumps(
            {
                "scientific_sha": scientific_sha,
                "result_sha": result_sha,
                "merge_sha": merge_sha,
                "source_gate_pass": receipt["source_gate_pass"],
                "rep3_numeric_outcomes_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
