"""One-shot repair hook for generated Deform360 source; deletes itself after use."""

from __future__ import annotations

from pathlib import Path


def _repair_generated_text() -> bool:
    root = Path.cwd()
    paths = (
        root / "scripts/science/run_deform360_normalized_evidence_external.py",
        root / "tests/test_deform360_normalized_evidence_external.py",
        root / ".github/workflows/deform360-normalized-evidence-external.yml",
        root / "docs/deform360_normalized_evidence_external_v1.md",
    )
    changed = False
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        normalized = raw.replace(b"\x00", b"\\0")
        if normalized != raw:
            path.write_bytes(normalized)
            changed = True

    source = paths[0]
    if not source.is_file():
        return changed
    lines = source.read_text(encoding="utf-8").splitlines()
    repaired_lines = 0
    index = 0
    while index + 1 < len(lines):
        if lines[index].strip() == '"' and lines[index + 1].lstrip().startswith(
            '".join('
        ):
            indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines[index] = indentation + '"\\n' + lines[index + 1].lstrip()
            del lines[index + 1]
            repaired_lines += 1
            continue
        index += 1
    if repaired_lines:
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        changed = True
    return changed


if _repair_generated_text():
    try:
        Path(__file__).unlink()
    except OSError:
        pass
