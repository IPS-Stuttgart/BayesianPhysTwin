#!/usr/bin/env python3
"""Compatibility wrapper for the registered MatPhys backend command."""

from __future__ import annotations

import sys

from bayesian_phystwin.cli.matphys_backend import main as matphys_backend_main


def main() -> int:
    return matphys_backend_main(["materialize", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
