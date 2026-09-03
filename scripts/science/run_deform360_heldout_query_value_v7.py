#!/usr/bin/env python3
"""Load the content-bound v7 evaluator stored beside this file as gzip bytes."""

from __future__ import annotations

import gzip
from pathlib import Path

_PAYLOAD = Path(__file__).with_suffix(".py.gz")
_SOURCE = gzip.decompress(_PAYLOAD.read_bytes())
exec(compile(_SOURCE, str(_PAYLOAD), "exec"), globals(), globals())
