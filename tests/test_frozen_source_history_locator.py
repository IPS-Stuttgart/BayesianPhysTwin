from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.ci.locate_frozen_source_bundle import (
    load_requirements,
    locate_frozen_source_bundle,
    main,
    write_report,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "History Locator Test")
    _git(repository, "config", "user.email", "history-locator@example.invalid")
    return repository


def test_locator_finds_exact_historical_snapshot_and_tag(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = b"frozen first file\n"
    second = b"frozen second file\n"
    (repository / "nested").mkdir()
    (repository / "first.txt").write_bytes(first)
    (repository / "nested/second.txt").write_bytes(b"not frozen\n")
    _commit(repository, "add candidate files")
    (repository / "nested/second.txt").write_bytes(second)
    exact_revision = _commit(repository, "complete exact bundle")
    _git(repository, "tag", "frozen-source-v1", exact_revision)
    (repository / "unrelated.txt").write_text("later\n", encoding="utf-8")
    _commit(repository, "advance unrelated history")

    report = locate_frozen_source_bundle(
        repository,
        {
            "first.txt": _sha256(first),
            "nested/second.txt": _sha256(second),
        },
    )

    assert report["complete_history_searched"] is True
    assert report["candidate_commit_count"] == 2
    assert report["anchor_match_count"] == 2
    assert report["exact_match_count"] == 1
    match = report["exact_matches"][0]
    assert match["revision"] == exact_revision
    assert "refs/tags/frozen-source-v1" in match["refs_pointing_at"]
    assert match["containing_tags"] == ["frozen-source-v1"]
    assert not any(report["information_boundary"].values())
    assert len(report["report_id"]) == 64


def test_locator_reports_no_match_without_guessing(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "source.py").write_text("original\n", encoding="utf-8")
    _commit(repository, "add source")

    report = locate_frozen_source_bundle(
        repository,
        {"source.py": _sha256(b"different\n")},
    )

    assert report["candidate_commit_count"] == 1
    assert report["anchor_match_count"] == 0
    assert report["exact_match_count"] == 0
    assert report["exact_matches"] == []


def test_requirements_loader_accepts_wrapped_mapping_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "requirements.json"
    path.write_text(
        json.dumps(
            {
                "files": {
                    "z.txt": "b" * 64,
                    "a.txt": "a" * 64,
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_requirements(path)

    assert list(loaded) == ["a.txt", "z.txt"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"../escape.txt": "a" * 64},
        {"/absolute.txt": "a" * 64},
        {"windows\\path.txt": "a" * 64},
        {"valid.txt": "A" * 64},
        {"valid.txt": True},
    ],
)
def test_requirements_loader_rejects_ambiguous_inputs(
    tmp_path: Path,
    payload: object,
) -> None:
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_requirements(path)


def test_report_publication_is_no_clobber_and_symlink_refusing(
    tmp_path: Path,
) -> None:
    report = {"schema": "test", "report_id": "a" * 64}
    output = tmp_path / "report.json"
    write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report

    with pytest.raises(FileExistsError):
        write_report(output, report)

    target = tmp_path / "target.json"
    target.write_text("protected\n", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(FileExistsError):
        write_report(link, report)
    assert target.read_text(encoding="utf-8") == "protected\n"


def test_cli_can_require_a_match_while_retaining_report(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "source.py").write_text("original\n", encoding="utf-8")
    _commit(repository, "add source")
    requirements = tmp_path / "requirements.json"
    requirements.write_text(
        json.dumps({"source.py": _sha256(b"different\n")}),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--repository-root",
            str(repository),
            "--requirements-json",
            str(requirements),
            "--output",
            str(output),
            "--require-match",
        ]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["exact_match_count"] == 0


def test_locator_rejects_shallow_repository(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    (source / "source.py").write_text("content\n", encoding="utf-8")
    _commit(source, "add source")
    clone = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            f"file://{source}",
            str(clone),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match="complete Git history"):
        locate_frozen_source_bundle(
            clone,
            {"source.py": _sha256(b"content\n")},
        )
