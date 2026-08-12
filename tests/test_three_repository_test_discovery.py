from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.discover_three_repository_tests import (
    INVENTORY_SCHEMA,
    SourceSpec,
    main,
    stage_integration_tests,
)


def _source(tmp_path: Path, name: str, files: dict[str, bytes]) -> Path:
    root = tmp_path / name
    root.mkdir()
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_stages_tests_from_every_owner_and_preserves_support_files(tmp_path: Path) -> None:
    bpt = _source(
        tmp_path,
        "bpt",
        {
            "integration_tests/test_three_repository_bpt.py": b"def test_bpt(): pass\n",
            "integration_tests/helpers.py": b"VALUE = 1\n",
        },
    )
    prob4d = _source(
        tmp_path,
        "prob4d",
        {
            "integration_tests/nested/test_three_repository_prob4d.py": (
                b"def test_prob4d(): pass\n"
            ),
            "integration_tests/fixture.json": b"{}\n",
        },
    )
    causal4d = _source(tmp_path, "causal4d", {})
    output = tmp_path / "run"
    paths = tmp_path / "paths.txt"
    inventory_path = tmp_path / "inventory.json"

    inventory = stage_integration_tests(
        (
            SourceSpec("prob4d", prob4d),
            SourceSpec("causal4d", causal4d),
            SourceSpec("bayesian_phystwin", bpt),
        ),
        output_root=output,
        path_list=paths,
        inventory_path=inventory_path,
    )

    assert inventory == {
        "schema": INVENTORY_SCHEMA,
        "owners": [
            {
                "owner": "bayesian_phystwin",
                "test_files": ["test_three_repository_bpt.py"],
            },
            {"owner": "causal4d", "test_files": []},
            {
                "owner": "prob4d",
                "test_files": ["nested/test_three_repository_prob4d.py"],
            },
        ],
        "total_test_files": 2,
    }
    assert paths.read_text(encoding="utf-8") == (
        "bayesian_phystwin/test_three_repository_bpt.py\n"
        "prob4d/nested/test_three_repository_prob4d.py\n"
    )
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == inventory
    assert (output / "bayesian_phystwin" / "helpers.py").read_bytes() == b"VALUE = 1\n"
    assert (output / "prob4d" / "fixture.json").read_bytes() == b"{}\n"


def test_rejects_no_matching_tests_without_writing_manifests(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "source",
        {"integration_tests/helper.py": b"VALUE = 1\n"},
    )
    output = tmp_path / "run"
    paths = tmp_path / "paths.txt"
    inventory = tmp_path / "inventory.json"

    with pytest.raises(ValueError, match="no three-repository integration tests"):
        stage_integration_tests(
            (SourceSpec("owner", source),),
            output_root=output,
            path_list=paths,
            inventory_path=inventory,
        )

    assert not output.exists()
    assert not paths.exists()
    assert not inventory.exists()


def test_rejects_duplicate_owners_and_roots(tmp_path: Path) -> None:
    first = _source(
        tmp_path,
        "first",
        {"integration_tests/test_three_repository_a.py": b"def test_a(): pass\n"},
    )
    second = _source(
        tmp_path,
        "second",
        {"integration_tests/test_three_repository_b.py": b"def test_b(): pass\n"},
    )

    with pytest.raises(ValueError, match="owners must be unique"):
        stage_integration_tests(
            (SourceSpec("same", first), SourceSpec("same", second)),
            output_root=tmp_path / "owners",
            path_list=tmp_path / "owners.txt",
            inventory_path=tmp_path / "owners.json",
        )

    with pytest.raises(ValueError, match="roots must be unique"):
        stage_integration_tests(
            (SourceSpec("first", first), SourceSpec("second", first)),
            output_root=tmp_path / "roots",
            path_list=tmp_path / "roots.txt",
            inventory_path=tmp_path / "roots.json",
        )


def test_rejects_nonempty_output_and_existing_manifests(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "source",
        {"integration_tests/test_three_repository_a.py": b"def test_a(): pass\n"},
    )
    output = tmp_path / "run"
    output.mkdir()
    (output / "unexpected.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="output root must be empty"):
        stage_integration_tests(
            (SourceSpec("owner", source),),
            output_root=output,
            path_list=tmp_path / "paths.txt",
            inventory_path=tmp_path / "inventory.json",
        )

    empty = tmp_path / "empty"
    empty.mkdir()
    paths = tmp_path / "existing-paths.txt"
    paths.write_text("existing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path list already exists"):
        stage_integration_tests(
            (SourceSpec("owner", source),),
            output_root=empty,
            path_list=paths,
            inventory_path=tmp_path / "new-inventory.json",
        )


def test_rejects_symlinked_integration_content(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "source",
        {"integration_tests/test_three_repository_a.py": b"def test_a(): pass\n"},
    )
    target = source / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = source / "integration_tests" / "linked.txt"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ValueError, match="contains a symbolic link"):
        stage_integration_tests(
            (SourceSpec("owner", source),),
            output_root=tmp_path / "run",
            path_list=tmp_path / "paths.txt",
            inventory_path=tmp_path / "inventory.json",
        )


def test_cli_rejects_unsafe_owner_and_output_inside_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(
        tmp_path,
        "source",
        {"integration_tests/test_three_repository_a.py": b"def test_a(): pass\n"},
    )

    result = main(
        [
            "--source",
            f"unsafe-owner={source}",
            "--output-root",
            str(tmp_path / "run"),
            "--path-list",
            str(tmp_path / "paths.txt"),
            "--inventory",
            str(tmp_path / "inventory.json"),
        ]
    )
    assert result == 2
    assert "source owner must start" in capsys.readouterr().err

    result = main(
        [
            "--source",
            f"owner={source}",
            "--output-root",
            str(source / "run"),
            "--path-list",
            str(tmp_path / "paths-2.txt"),
            "--inventory",
            str(tmp_path / "inventory-2.json"),
        ]
    )
    assert result == 2
    assert "output path must not be inside" in capsys.readouterr().err


def test_manifest_files_are_created_with_restrictive_default_permissions(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        "source",
        {"integration_tests/test_three_repository_a.py": b"def test_a(): pass\n"},
    )
    paths = tmp_path / "paths.txt"
    inventory = tmp_path / "inventory.json"

    stage_integration_tests(
        (SourceSpec("owner", source),),
        output_root=tmp_path / "run",
        path_list=paths,
        inventory_path=inventory,
    )

    assert os.stat(paths).st_mode & 0o777 == 0o644
    assert os.stat(inventory).st_mode & 0o777 == 0o644
