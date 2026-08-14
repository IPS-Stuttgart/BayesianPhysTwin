"""Compatibility module for the canonical external material-backend CLI."""

from .lagrangian_backend import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
