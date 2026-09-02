from pathlib import Path
import subprocess


def add_after(text, marker, addition, count):
    if text.count(marker) != count:
        raise SystemExit(f"marker count changed: {marker!r}")
    return text.replace(marker, marker + addition)


decision = Path(
    "experiments/tracking_cloth_action_feasibility_costed_v2/_decision.py"
)
text = decision.read_text()
marker = (
    "    unknown_gap = unknown_plan_upper[:, None] - unknown_plan_lower[None, :]\n"
)
if text.count(marker) != 1:
    raise SystemExit("unknown-gap marker changed")
text = text.replace(
    marker,
    marker
    + "    # A complete plan compared with itself has zero loss gap, including\n"
    + "    # on unrepresented physics; its own interval cannot consume epsilon.\n"
    + "    np.fill_diagonal(unknown_gap, 0.0)\n",
)
decision.write_text(text)

path = Path(".github/workflows/tracking-cloth-self-collision-selective-twin-v1.yml")
text = subprocess.check_output(
    ["git", "show", "origin/main:" + str(path)], text=True
)
text = add_after(
    text,
    '      - "experiments/tracking_cloth_action_feasibility_v1/**"\n',
    '      - "experiments/tracking_cloth_action_feasibility_costed_v2/**"\n',
    2,
)
text = add_after(
    text,
    '      - "tests/test_tracking_cloth_action_feasibility_v1.py"\n',
    '      - "tests/test_tracking_cloth_action_feasibility_costed_v2.py"\n',
    2,
)
text = add_after(
    text,
    "  ACTION_PROTOCOL_PATH: "
    "experiments/tracking_cloth_action_feasibility_v1/protocol.json\n",
    "  ACTION_V2_PROTOCOL_PATH: "
    "experiments/tracking_cloth_action_feasibility_costed_v2/protocol.json\n",
    1,
)
text = add_after(
    text,
    "              experiments/tracking_cloth_action_feasibility_v1/*)\n"
    "                action_audit=true\n"
    "                ;;\n",
    "              experiments/tracking_cloth_action_feasibility_costed_v2/*)\n"
    "                action_audit=true\n"
    "                ;;\n",
    1,
)
text = add_after(
    text,
    "              tests/test_tracking_cloth_action_feasibility_v1.py)\n"
    "                action_audit=true\n"
    "                ;;\n",
    "              tests/test_tracking_cloth_action_feasibility_costed_v2.py)\n"
    "                action_audit=true\n"
    "                ;;\n",
    1,
)
text = add_after(
    text,
    "            tests/test_tracking_cloth_action_feasibility_v1.py \\\n",
    "            tests/test_tracking_cloth_action_feasibility_costed_v2.py \\\n",
    3,
)
text = add_after(
    text,
    "            experiments/tracking_cloth_action_feasibility_v1 \\\n",
    "            experiments/tracking_cloth_action_feasibility_costed_v2 \\\n",
    2,
)
fragment = Path("tools/_tracking_cloth_v2_job.ymlfrag").read_text()
if "\n  action_audit_v2:" in text:
    raise SystemExit("V2 job already present")
path.write_text(text.rstrip() + "\n\n" + fragment)

patterns = (
    ".github/workflows/finalize-tracking-cloth-action-costed-v2*.yml",
    ".github/workflows/run-and-record-tracking-cloth-action-costed-v2*.yml",
    ".github/workflows/tracking-cloth-action-feasibility-costed-v2.yml",
    ".github/workflows/repair-tracking-cloth-costed-v2*.yml",
    ".github/requests/finalize-tracking-cloth-action-costed-v2*.json",
    ".github/requests/run-and-record-tracking-cloth-action-costed-v2*.json",
    ".github/requests/repair-tracking-cloth-costed-v2*.json",
    "tools/_temporary_run_tracking_cloth_action_costed_v2*.py",
)
for pattern in patterns:
    for item in Path(".").glob(pattern):
        item.unlink()
