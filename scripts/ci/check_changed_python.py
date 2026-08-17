#!/usr/bin/env python3
"""Run inexpensive source checks on Python files changed between two revisions.

In addition to Ruff and byte compilation, changed package modules are scanned for
high-confidence input-boundary and numerical patterns that previously caused
silent scientific contract violations.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_SIZE = 100
_PACKAGE_PREFIX = "src/bayesian_phystwin/"
_NUMPY_ARRAY_CALLS = frozenset({"array", "asarray"})


@dataclass(frozen=True)
class ScientificBoundaryViolation:
    """One fail-closed source pattern found in a changed package module."""

    path: Path
    line: int
    column: int
    code: str
    message: str

    def render(self) -> str:
        return (
            f"{self.path.as_posix()}:{self.line}:{self.column}: "
            f"{self.code} {self.message}"
        )


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=None,
    )


def _resolve_revision(repository_root: Path, revision: str, *, name: str) -> str:
    if not revision or revision.strip() != revision:
        raise ValueError(f"{name} must be a nonempty canonical revision")
    try:
        result = _run_git(
            repository_root,
            ["rev-parse", "--verify", f"{revision}^{{commit}}"],
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{name} is not an available commit: {revision}") from error
    return result.stdout.decode("ascii").strip()


def changed_python_files(
    repository_root: Path,
    *,
    base_revision: str,
    head_revision: str,
) -> tuple[Path, ...]:
    """Return ordinary repository-local Python files changed in one exact diff."""

    root = repository_root.resolve(strict=True)
    base_sha = _resolve_revision(root, base_revision, name="base_revision")
    head_sha = _resolve_revision(root, head_revision, name="head_revision")
    checkout_sha = _resolve_revision(root, "HEAD", name="repository HEAD")
    if checkout_sha != head_sha:
        raise ValueError(
            "repository HEAD does not match head_revision; "
            "refusing to check files from a different tree"
        )
    result = _run_git(
        root,
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            base_sha,
            head_sha,
            "--",
            "*.py",
        ],
        capture_output=True,
    )
    candidates: set[Path] = set()
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"git reported a nonlocal path: {name}")
        source = root / relative
        try:
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"changed path is not readable: {name}") from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"changed path escapes the repository: {name}") from error
        if source.is_symlink():
            raise ValueError(f"changed Python path must not be a symlink: {name}")
        if not source.is_file():
            raise ValueError(f"changed Python path is not an ordinary file: {name}")
        candidates.add(relative)
    return tuple(sorted(candidates, key=lambda path: path.as_posix()))


def _chunks(values: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _checked_paths(paths: Sequence[Path]) -> list[str]:
    return [f"./{path.as_posix()}" for path in paths]


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _is_builtin_bool(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "bool"


def _function_parameters(arguments: ast.arguments) -> frozenset[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return frozenset(names)


class _ScientificBoundaryVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: Sequence[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.numpy_aliases = {"np", "numpy"}
        self.numpy_array_calls: set[str] = set()
        self.numpy_linalg_aliases: set[str] = set()
        self.numpy_inverse_calls: set[str] = set()
        self.parameter_stack: list[frozenset[str]] = []
        self.violations: list[ScientificBoundaryViolation] = []

    def _suppressed(self, node: ast.AST, code: str) -> bool:
        marker = f"bpt-quality: allow {code}"
        start = max(getattr(node, "lineno", 1) - 2, 0)
        end = min(getattr(node, "end_lineno", start + 1), len(self.source_lines))
        return any(marker in self.source_lines[index] for index in range(start, end))

    def _record(self, node: ast.AST, code: str, message: str) -> None:
        if self._suppressed(node, code):
            return
        self.violations.append(
            ScientificBoundaryViolation(
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                code=code,
                message=message,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            if imported.name == "numpy":
                self.numpy_aliases.add(imported.asname or "numpy")
            elif imported.name == "numpy.linalg":
                self.numpy_linalg_aliases.add(imported.asname or "numpy.linalg")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "numpy":
            for imported in node.names:
                if imported.name in _NUMPY_ARRAY_CALLS:
                    self.numpy_array_calls.add(imported.asname or imported.name)
                elif imported.name == "linalg":
                    self.numpy_linalg_aliases.add(imported.asname or imported.name)
        elif node.module == "numpy.linalg":
            for imported in node.names:
                if imported.name == "inv":
                    self.numpy_inverse_calls.add(imported.asname or imported.name)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.parameter_stack.append(_function_parameters(node.args))
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.parameter_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.parameter_stack.append(_function_parameters(node.args))
        try:
            self.visit(node.body)
        finally:
            self.parameter_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        qualified = _qualified_name(node.func)
        numpy_array_call = qualified in self.numpy_array_calls
        if qualified is not None and "." in qualified:
            prefix, _, name = qualified.rpartition(".")
            numpy_array_call = (
                prefix in self.numpy_aliases and name in _NUMPY_ARRAY_CALLS
            )
        if numpy_array_call:
            dtype_values = [
                keyword.value for keyword in node.keywords if keyword.arg == "dtype"
            ]
            if len(node.args) >= 2:
                dtype_values.append(node.args[1])
            if any(_is_builtin_bool(value) for value in dtype_values):
                self._record(
                    node,
                    "BPTQ001",
                    "dtype=bool silently truth-coerces NaN and arbitrary nonzero "
                    "values; validate the mask dtype first and convert to "
                    "np.bool_ only after admission",
                )

        numpy_inverse_call = qualified in self.numpy_inverse_calls
        if qualified is not None and "." in qualified:
            prefix, _, name = qualified.rpartition(".")
            numpy_inverse_call = name == "inv" and (
                prefix in self.numpy_linalg_aliases
                or any(prefix == f"{alias}.linalg" for alias in self.numpy_aliases)
            )
        if numpy_inverse_call:
            self._record(
                node,
                "BPTQ003",
                "direct numpy.linalg.inv hides solver choice and conditioning; "
                "use an explicit solve, Cholesky factorization, or admitted "
                "eigendecomposition and retain numerical diagnostics",
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if self.parameter_stack and isinstance(node.op, ast.Or):
            parameters = self.parameter_stack[-1]
            for parameter, fallback in zip(node.values, node.values[1:], strict=False):
                if not isinstance(parameter, ast.Name):
                    continue
                if (
                    parameter.id not in parameters
                    or "config" not in parameter.id.lower()
                ):
                    continue
                if not isinstance(fallback, ast.Call):
                    continue
                constructor = _qualified_name(fallback.func)
                if constructor is None or not constructor.split(".")[-1].endswith(
                    "Config"
                ):
                    continue
                self._record(
                    node,
                    "BPTQ002",
                    "parameter-or-Config defaulting silently replaces falsey "
                    "invalid inputs; branch on `is None` and validate the "
                    "configuration type explicitly",
                )
        self.generic_visit(node)


def scientific_boundary_violations(
    repository_root: Path,
    paths: Sequence[Path],
) -> tuple[ScientificBoundaryViolation, ...]:
    """Return high-confidence unsafe patterns in changed package modules."""

    root = repository_root.resolve(strict=True)
    violations: list[ScientificBoundaryViolation] = []
    for relative in paths:
        if not relative.as_posix().startswith(_PACKAGE_PREFIX):
            continue
        source_path = root / relative
        source = source_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative.as_posix())
        except SyntaxError as error:
            line = error.lineno or 1
            column = error.offset or 1
            raise ValueError(
                f"{relative.as_posix()}:{line}:{column}: invalid Python syntax: "
                f"{error.msg}"
            ) from error
        visitor = _ScientificBoundaryVisitor(relative, source.splitlines())
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return tuple(
        sorted(
            violations,
            key=lambda violation: (
                violation.path.as_posix(),
                violation.line,
                violation.column,
                violation.code,
            ),
        )
    )


def run_changed_python_checks(
    repository_root: Path,
    paths: Sequence[Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Run boundary scans, Ruff lint/format, and byte compilation."""

    root = repository_root.resolve(strict=True)
    if not paths:
        print("No changed Python files require the fast source preflight.")
        return
    print(f"Fast source preflight: {len(paths)} changed Python file(s)")
    for path in paths:
        print(f"  {path.as_posix()}")

    violations = scientific_boundary_violations(root, paths)
    if violations:
        print("Scientific-boundary quality violations:", file=sys.stderr)
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        raise ValueError(
            f"{len(violations)} unsafe scientific-boundary pattern(s) detected"
        )

    for chunk in _chunks(tuple(paths), chunk_size):
        rendered = _checked_paths(chunk)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format=github",
                *rendered,
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "--diff",
                *rendered,
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "py_compile", *rendered],
            cwd=root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Exact base commit SHA")
    parser.add_argument("--head", required=True, help="Exact head commit SHA")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root; defaults to the current directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum files passed to each checker invocation",
    )
    args = parser.parse_args()
    try:
        paths = changed_python_files(
            args.repository_root,
            base_revision=args.base,
            head_revision=args.head,
        )
        run_changed_python_checks(
            args.repository_root,
            paths,
            chunk_size=args.chunk_size,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"fast source preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
