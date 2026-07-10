import json
import zipfile
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_data import (
    fetch_phystwin_additional_evaluation_subset,
    fetch_phystwin_evaluation_subset,
)


def _build_archives(root: Path) -> tuple[Path, Path]:
    data_path = root / "data.zip"
    experiments_path = root / "experiments.zip"
    with zipfile.ZipFile(data_path, "w") as archive:
        for case in ("case_a", "case_b"):
            prefix = f"data/different_types/{case}"
            archive.writestr(f"{prefix}/final_data.pkl", case.encode())
            archive.writestr(f"{prefix}/gt_track_3d.pkl", b"tracks")
            archive.writestr(
                f"{prefix}/split.json",
                json.dumps({"train": [0, 3], "test": [3, 4]}),
            )
        archive.writestr("data/different_types/incomplete/split.json", "{}")
    with zipfile.ZipFile(experiments_path, "w") as archive:
        for case in ("case_a", "case_b"):
            archive.writestr(f"experiments/{case}/inference.pkl", b"trajectory")
    return data_path, experiments_path


def test_fetches_selected_evaluation_subset_and_reuses_valid_files(tmp_path: Path):
    data_path, experiments_path = _build_archives(tmp_path)
    factory = lambda source: zipfile.ZipFile(source)
    output = tmp_path / "output"

    first = fetch_phystwin_evaluation_subset(
        output,
        cases=("case_b",),
        data_archive_url=str(data_path),
        experiments_archive_url=str(experiments_path),
        archive_factory=factory,
    )
    second = fetch_phystwin_evaluation_subset(
        output,
        cases=("case_b",),
        data_archive_url=str(data_path),
        experiments_archive_url=str(experiments_path),
        archive_factory=factory,
    )

    assert first["available_cases"] == ["case_a", "case_b"]
    assert first["selected_cases"] == ["case_b"]
    assert (output / "case_b" / "inference.pkl").read_bytes() == b"trajectory"
    assert all(
        record["reused"]
        for record in second["cases"]["case_b"]["files"].values()
    )


def test_rejects_unknown_or_incomplete_case(tmp_path: Path):
    data_path, experiments_path = _build_archives(tmp_path)
    factory = lambda source: zipfile.ZipFile(source)

    with pytest.raises(ValueError, match="incomplete"):
        fetch_phystwin_evaluation_subset(
            tmp_path / "output",
            cases=("incomplete",),
            data_archive_url=str(data_path),
            experiments_archive_url=str(experiments_path),
            archive_factory=factory,
        )


def test_fetches_label_free_additional_subset(tmp_path: Path):
    archive_path = tmp_path / "additional.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        prefix = "additional_data/data/different_types/cloth_blue_fold"
        archive.writestr(f"{prefix}/final_data.pkl", b"data")
        archive.writestr(f"{prefix}/split.json", b"{}")
        archive.writestr(
            "additional_data/experiments/cloth_blue_fold/inference.pkl",
            b"trajectory",
        )
    output = tmp_path / "additional"

    manifest = fetch_phystwin_additional_evaluation_subset(
        output,
        archive_url=str(archive_path),
        archive_factory=lambda source: zipfile.ZipFile(source),
    )

    assert manifest["selected_cases"] == ["cloth_blue_fold"]
    assert manifest["manual_track_labels"] is False
    assert (output / "cloth_blue_fold" / "inference.pkl").read_bytes() == b"trajectory"
