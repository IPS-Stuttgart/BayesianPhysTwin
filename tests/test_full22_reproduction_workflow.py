from pathlib import Path


WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "full22-anchor-reproduction.yml"
)


def test_hosted_full22_lane_installs_scipy_provider() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'python -m pip install -e ".[dev,data,graph]"' in workflow
    assert 'python -m pip install -e ".[dev,data]"' not in workflow
