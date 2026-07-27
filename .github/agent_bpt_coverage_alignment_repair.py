"""Cover the constructor branch attributed to the nearby typing-only edit."""

from __future__ import annotations

from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "tests/test_run_manifest_v2.py"
MARKER = "test_v2_rejects_empty_run_id"
TEST = '''


def test_v2_rejects_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run ID must be nonempty"):
        replace(_manifest(tmp_path), run_id="")
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        raise RuntimeError("coverage-alignment test already present")
    TARGET.write_text(text.rstrip() + TEST, encoding="utf-8")


if __name__ == "__main__":
    main()
