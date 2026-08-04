from importlib.resources import files
from pathlib import Path


def test_source_package_exposes_pep561_marker() -> None:
    marker = files("bayesian_phystwin").joinpath("py.typed")

    assert marker.is_file()
    assert "PEP 561" in marker.read_text(encoding="utf-8")


def test_project_metadata_declares_typed_package_data() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert '"Typing :: Typed"' in pyproject
    assert 'bayesian_phystwin = ["py.typed"]' in pyproject
