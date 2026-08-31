from __future__ import annotations

from pathlib import Path


REPLACEMENTS = {
    Path("experiments/deform360_tactile_gpuserver6000_v1/run.py"): [
        (
            "from collections.abc import Mapping, Sequence\n",
            "from collections.abc import Mapping\n",
        )
    ],
    Path(
        ".github/workflows/run-deform360-gpuserver6000-tactile-20260901-v1.yml"
    ): [
        (
            "experiments/deform360_tactile_gpuser6000_v1/run.py",
            "experiments/deform360_tactile_gpuserver6000_v1/run.py",
        )
    ],
}

for path, replacements in REPLACEMENTS.items():
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"expected one match in {path}: {old!r}; observed {count}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
