"""One-shot strict-typing fix for PR #187; deleted by its publisher."""

from pathlib import Path

path = Path("src/bayesian_phystwin/prob4d_observation_timestamps.py")
text = path.read_text(encoding="utf-8")
old = "        result = np.zeros(\n"
new = "        result: np.ndarray = np.zeros(\n"
count = text.count(old)
if count != 1:
    raise SystemExit(f"expected one timing-workspace target, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
