from pathlib import Path


def test_horizon_workflow_is_read_only_and_pinned() -> None:
    text = Path(".github/workflows/horizon-discrepancy.yml").read_text(
        encoding="utf-8"
    )

    assert "contents: read" in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    assert "tests/test_horizon_discrepancy.py" in text
    assert "python -m ruff check" in text
    assert "python -m pytest -q" in text
