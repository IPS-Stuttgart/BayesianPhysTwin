"""Fresh repetition-level selective-digital-twin confirmation."""

from __future__ import annotations

import sys

from . import bounds as _bounds

# The experiment was initially drafted against this import path. Keep the
# internal experiment import working without admitting a new stable API module.
sys.modules.setdefault("bayesian_phystwin.selective_competence_bound_v1", _bounds)
