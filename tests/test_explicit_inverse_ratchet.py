"""Prevent new explicit matrix inverses in production BayesianPhysTwin code."""

import ast
import textwrap
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "bayesian_phystwin"

# These calls predate the solve-based numerical policy. Some belong to frozen
# experiment paths, while others invert small rigid transforms or reconstruct an
# explicitly exported covariance. The ratchet is exact: removing a call requires
# lowering this baseline in the same change, and adding a call anywhere fails.
EXPECTED_EXPLICIT_INVERSE_CALLS = {
    "bias_aware_belief.py": 2,
    "deform360_bias_aware_prospective_uncertainty.py": 1,
    "deform360_cotracker_bias_source.py": 1,
    "deform360_crossview_2d_guard.py": 1,
    "deform360_crossview_guard.py": 1,
    "deform360_frame_zero_depth_initializer.py": 1,
    "deform360_frame_zero_initializer.py": 1,
    "deform360_fresh_object_session_candidate_v6_1.py": 2,
    "deform360_raw_camera_observation.py": 2,
    "matphys_dino_features.py": 1,
    "phystwin_cotracker3_cues.py": 5,
    "phystwin_motioncrafter_assimilation.py": 3,
    "pokeflex_bayesian_registration.py": 2,
    "pokeflex_independent_depth.py": 1,
}


class _NumpyInverseVisitor(ast.NodeVisitor):
    """Find NumPy inverse calls even when NumPy or ``inv`` is aliased."""

    def __init__(self) -> None:
        self.numpy_aliases: set[str] = set()
        self.linalg_aliases: set[str] = set()
        self.inverse_aliases: set[str] = set()
        self.lines: list[int] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "numpy":
                self.numpy_aliases.add(alias.asname or "numpy")
            elif alias.name == "numpy.linalg":
                if alias.asname:
                    self.linalg_aliases.add(alias.asname)
                else:
                    self.numpy_aliases.add("numpy")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module == "numpy":
            for alias in node.names:
                if alias.name == "linalg":
                    self.linalg_aliases.add(alias.asname or "linalg")
        elif node.module == "numpy.linalg":
            for alias in node.names:
                if alias.name == "inv":
                    self.inverse_aliases.add(alias.asname or "inv")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        function = node.func
        direct_inverse = (
            isinstance(function, ast.Name) and function.id in self.inverse_aliases
        )
        linalg_inverse = False
        numpy_inverse = False
        if isinstance(function, ast.Attribute) and function.attr == "inv":
            owner = function.value
            linalg_inverse = (
                isinstance(owner, ast.Name) and owner.id in self.linalg_aliases
            )
            numpy_inverse = (
                isinstance(owner, ast.Attribute)
                and owner.attr == "linalg"
                and isinstance(owner.value, ast.Name)
                and owner.value.id in self.numpy_aliases
            )
        if direct_inverse or linalg_inverse or numpy_inverse:
            self.lines.append(node.lineno)
        self.generic_visit(node)


def _inverse_lines(source: str) -> tuple[int, ...]:
    visitor = _NumpyInverseVisitor()
    visitor.visit(ast.parse(source))
    return tuple(sorted(visitor.lines))


def _production_inverse_inventory() -> tuple[
    dict[str, int], dict[str, tuple[int, ...]]
]:
    counts: dict[str, int] = {}
    locations: dict[str, tuple[int, ...]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        lines = _inverse_lines(path.read_text(encoding="utf-8"))
        if not lines:
            continue
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        counts[relative] = len(lines)
        locations[relative] = lines
    return counts, locations


def test_explicit_numpy_inverse_inventory_does_not_grow() -> None:
    observed, locations = _production_inverse_inventory()
    assert observed == EXPECTED_EXPLICIT_INVERSE_CALLS, (
        "explicit NumPy inverse inventory changed; use SPDSystem.solve, "
        "np.linalg.solve, whitening, or a factorized operator for new code. "
        f"Expected {EXPECTED_EXPLICIT_INVERSE_CALLS}, observed {observed}, "
        f"locations {locations}"
    )
    assert sum(observed.values()) == 24


def test_inverse_detector_covers_supported_numpy_import_forms() -> None:
    sources = (
        "import numpy as np\nnp.linalg.inv(matrix)\n",
        "import numpy\nnumpy.linalg.inv(matrix)\n",
        "import numpy.linalg as la\nla.inv(matrix)\n",
        "from numpy import linalg as la\nla.inv(matrix)\n",
        "from numpy.linalg import inv as inverse\ninverse(matrix)\n",
    )
    for source in sources:
        assert _inverse_lines(source) == (2,)


def test_inverse_detector_ignores_unrelated_inv_attributes() -> None:
    source = textwrap.dedent(
        """
        class Operator:
            def inv(self, value):
                return value

        Operator().inv(matrix)
        """
    )
    assert _inverse_lines(source) == ()
