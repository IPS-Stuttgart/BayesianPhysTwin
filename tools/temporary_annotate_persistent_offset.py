"""Temporary exact MyPy annotation patch for PR #183."""

from pathlib import Path

PATH = Path("src/bayesian_phystwin/persistent_prob4d_visual_bias.py")
OLD = """    if physical_offset is None:
        offset = np.zeros(physical_dimension, dtype=np.float64)
    else:
        offset = _float64_vector(physical_offset, name=\"physical_offset\")
"""
NEW = """    offset: np.ndarray
    if physical_offset is None:
        offset = np.zeros(physical_dimension, dtype=np.float64)
    else:
        offset = _float64_vector(physical_offset, name=\"physical_offset\")
"""

source = PATH.read_text(encoding="utf-8")
count = source.count(OLD)
if count != 1:
    raise RuntimeError(f"expected one offset annotation target, found {count}")
PATH.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
