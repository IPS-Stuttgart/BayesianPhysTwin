"""Custody and separation checks; no recorded dataset or model execution."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "experiments/residual-models"
CATALOG = INDEX / "imports-2026-09-07.json"


def test_imported_source_and_inactive_workflow_custody() -> None:
    catalog = json.loads(CATALOG.read_text())
    assert {e["pr"] for e in catalog["experiments"]} == {
        950,
        952,
        953,
        955,
        956,
        957,
        958,
        959,
        960,
        961,
        962,
        963,
    }
    seen = set()
    inactive = 0
    for study in catalog["experiments"]:
        assert re.fullmatch(r"[0-9a-f]{40}", study["source_revision"])
        for item in study["files"]:
            if "path" not in item:
                assert item["disposition"] in {
                    "historical-document-reference",
                    "historical-outcome-reference",
                }
                assert study["source_revision"] in item["historical_url"]
                continue
            path = ROOT / item["path"]
            assert path.resolve().is_relative_to(ROOT)
            assert path not in seen
            seen.add(path)
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["retained_sha256"]
            if item["disposition"] == "inactive-execution-history":
                inactive += int(path.suffix == ".yml")
                assert item["path"].startswith("archive/research-workflows/")
                assert hashlib.sha256(data).hexdigest() == item["source_sha256"]
            else:
                assert not item["path"].startswith(
                    ("src/bayesian_phystwin/", ".github/")
                )
                if path.suffix == ".py":
                    assert item["numerical_ast_preserved"] is True
    assert inactive == 15


def test_independent_dp_models_do_not_alias() -> None:
    paths = [
        ROOT / "experiments/dp_residual_development_v1/model.py",
        ROOT / "experiments/dp_residual_kinematic_v1/model.py",
    ]
    assert all(p.is_file() for p in paths)
    assert paths[0].read_bytes() != paths[1].read_bytes()
    for name in ("run.py", "test_model.py"):
        text = (paths[1].parent / name).read_text()
        assert "experiments.dp_residual_kinematic_v1" in text
        assert "experiments.dp_residual_development_v1" not in text


def test_research_index_links_resolve() -> None:
    for path in INDEX.glob("*.md"):
        for target in re.findall(r"\[[^\]]*\]\(([^\s()]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            assert resolved.is_relative_to(ROOT)
            assert resolved.exists(), (path, target)
