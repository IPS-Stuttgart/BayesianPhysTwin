from __future__ import annotations

from pathlib import Path

import yaml

LAUNCHER = Path(
    ".github/workflows/launch-deform360-calibration-visual-production-once.yml"
)


def test_launcher_records_exact_fresh_resume_failure_and_repair() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    parsed = yaml.load(text, Loader=yaml.BaseLoader)

    assert isinstance(parsed, dict)
    assert "workflow_dispatch:" not in text
    assert "branches: [main]" in text
    assert "execute_authorized: true" in text
    assert "resume: true" in text
    assert "secrets: inherit" in text
    assert r"fresh-resume predecessor: run \`31277475724\`" in text
    assert r"\`0/324\` prediction seals" in text
    assert r"\`324/324\` identical technical failures" in text
    assert r"fresh-resume repair: PR \`#303\`" in text
    assert "starts missing or empty per-job outputs normally" in text
    assert "enables Prob4D resume only for nonempty interrupted bundles" in text
    assert "2026-08-09-fresh-bundle-resume-v6" in text


def test_launcher_preserves_closed_scientific_boundary() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert r"official raw payload opened: \`false\`" in text
    assert r"adaptive-confirmation payloads opened: \`false\`" in text
    assert r"reserved evaluation frames opened: \`false\`" in text
    assert r"confirmation payloads opened: \`false\`" in text
    assert r"target outcomes used: \`false\`" in text
    assert r"replacement allowed: \`false\`" in text
    assert r"Prob4D used: \`true\`, revision \`25d90ef7" in text
    assert r"MotionCrafter used: \`true\`, revision \`9cb4e967" in text
    assert r"admitted camera jobs: \`324\`" in text
