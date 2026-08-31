#!/usr/bin/env python3
"""Invoke the existing DEFORM DLO source adapter without guessing its contract.

The invoker parses the adapter's ``argparse`` declarations, maps only a narrow
allow-list of semantic arguments, executes all deterministically derived calls,
and retains the command, help text, output inventory, hashes, and exit codes.
Unknown required arguments stop execution before the adapter touches data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin/deform-dlo-source-adapter-invocation-v1"


class InvocationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvocationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(value: dict[str, Any], field: str = "invocation_id") -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def argument_specs(script: Path) -> list[dict[str, Any]]:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    specs: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        names = [
            value.value
            for value in node.args
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        if not names:
            continue
        keywords = {item.arg: literal(item.value) for item in node.keywords if item.arg}
        specs.append(
            {
                "names": names,
                "required": keywords.get("required") is True,
                "default": keywords.get("default"),
                "action": keywords.get("action"),
                "nargs": keywords.get("nargs"),
                "choices": keywords.get("choices"),
                "type": ast.unparse(
                    next(
                        (item.value for item in node.keywords if item.arg == "type"),
                        ast.Constant(value=None),
                    )
                ),
            }
        )
    return specs


def canonical_name(spec: dict[str, Any]) -> str:
    options = [name for name in spec["names"] if name.startswith("--")]
    name = options[0] if options else spec["names"][0]
    return name.lstrip("-").replace("_", "-").casefold()


def flag_name(spec: dict[str, Any]) -> str:
    options = [name for name in spec["names"] if name.startswith("--")]
    return options[0] if options else spec["names"][0]


def semantic_value(
    spec: dict[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    source: str,
    target: str,
) -> list[str] | None:
    name = canonical_name(spec)
    action = spec.get("action")
    if action in {"store_true", "store_false", "help", "version"}:
        return [] if not spec["required"] else None
    if any(token in name for token in ("output", "result", "artifact")):
        return [str(output_root)]
    if "source" in name and any(token in name for token in ("dlo", "object", "domain", "id", "name")):
        return [source]
    if "target" in name and any(token in name for token in ("dlo", "object", "domain", "id", "name")):
        return [target]
    if name in {"source", "source-dlo", "source-object"}:
        return [source]
    if name in {"target", "target-dlo", "target-object"}:
        return [target]
    if any(token in name for token in ("dataset-root", "data-root", "deform-root")):
        return [str(dataset_root)]
    if name in {"dataset", "data", "root", "path"}:
        return [str(dataset_root)]
    if name in {"seed", "random-seed", "rng-seed"}:
        return ["20260901"]
    if name in {"dlo", "dlo-id", "object", "object-id"}:
        nargs = spec.get("nargs")
        return [source, target] if nargs in {"+", "*"} else [source]
    if name in {"workers", "num-workers", "jobs", "n-jobs"}:
        return ["1"]
    if name in {"device"}:
        return ["cpu"]
    if name in {"repetitions", "runs", "num-runs", "trials"}:
        return ["1"]
    return None


def build_command(
    python: str,
    script: Path,
    specs: list[dict[str, Any]],
    *,
    dataset_root: Path,
    output_root: Path,
    source: str,
    target: str,
) -> tuple[list[str], list[str]]:
    command = [python, str(script)]
    unresolved: list[str] = []
    for spec in specs:
        name = flag_name(spec)
        value = semantic_value(
            spec,
            dataset_root=dataset_root,
            output_root=output_root,
            source=source,
            target=target,
        )
        is_positional = not name.startswith("-")
        if value is None:
            if spec["required"] or (is_positional and spec.get("default") is None):
                unresolved.append(name)
            continue
        if spec.get("action") in {"store_true", "store_false"}:
            continue
        if is_positional:
            command.extend(value)
        else:
            command.append(name)
            command.extend(value)
    return command, unresolved


def inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    adapter_root = args.adapter_root.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    script = adapter_root / "scripts" / "remote" / "run_deform_dlo_source.py"
    require(script.is_file(), f"adapter entry point missing: {script}")
    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True)
    help_run = subprocess.run(
        [args.python, str(script), "--help"],
        cwd=adapter_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    (output / "help.txt").write_text(help_run.stdout, encoding="utf-8")
    specs = argument_specs(script)
    executions: list[dict[str, Any]] = []
    for source, target in (("DLO4", "DLO5"), ("DLO5", "DLO4")):
        run_output = output / f"{source.lower()}-to-{target.lower()}"
        command, unresolved = build_command(
            args.python,
            script,
            specs,
            dataset_root=dataset_root,
            output_root=run_output,
            source=source,
            target=target,
        )
        record: dict[str, Any] = {
            "source": source,
            "target": target,
            "command": command,
            "command_shell": shlex.join(command),
            "unresolved_required_arguments": unresolved,
            "executed": False,
        }
        if unresolved:
            executions.append(record)
            continue
        run_output.mkdir()
        completed = subprocess.run(
            command,
            cwd=adapter_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "DEFORM_DATASET_ROOT": str(dataset_root),
                "DATASET_ROOT": str(dataset_root),
                "DEFORM_OUTPUT_ROOT": str(run_output),
                "OUTPUT_DIR": str(run_output),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        (output / f"{source.lower()}-to-{target.lower()}.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        record.update(
            {
                "executed": True,
                "returncode": completed.returncode,
                "output_inventory": inventory(run_output),
            }
        )
        executions.append(record)
    successful = [
        row for row in executions if row.get("executed") and row.get("returncode") == 0
    ]
    decision = (
        "existing-adapter-produced-realdata-output"
        if successful
        else "existing-adapter-contract-not-yet-executable"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "adapter_root": str(adapter_root),
        "adapter_revision": args.adapter_revision,
        "adapter_script": str(script.relative_to(adapter_root)),
        "adapter_script_sha256": sha256_file(script),
        "dataset_root": str(dataset_root),
        "help_returncode": help_run.returncode,
        "argument_specs": specs,
        "executions": executions,
        "summary": {
            "decision": decision,
            "execution_count": sum(row.get("executed") is True for row in executions),
            "successful_execution_count": len(successful),
            "unresolved_execution_count": sum(
                bool(row.get("unresolved_required_arguments")) for row in executions
            ),
        },
        "information_boundary": {
            "contract_inferred_from_argparse_only": True,
            "unknown_required_argument_guessed": False,
            "target_dependent_tuning": False,
            "paper_claim_authorized": False,
        },
        "claim_boundary": (
            "A successful invocation demonstrates that the existing DEFORM adapter "
            "can consume the verified DLO4/DLO5 release. Scientific interpretation "
            "depends on the adapter's retained outputs and is not granted by this "
            "invocation layer alone."
        ),
    }
    result["invocation_id"] = content_id(result)
    (output / "invocation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Existing DEFORM DLO source adapter invocation",
        "",
        f"- Decision: `{decision}`",
        f"- Invocation ID: `{result['invocation_id']}`",
        f"- Successful executions: `{len(successful)}/2`",
        f"- Help return code: `{help_run.returncode}`",
        "",
    ]
    for row in executions:
        lines.extend(
            [
                f"## {row['source']} to {row['target']}",
                "",
                f"- Executed: `{row['executed']}`",
                f"- Return code: `{row.get('returncode')}`",
                f"- Unresolved arguments: `{row['unresolved_required_arguments']}`",
                f"- Output files: `{len(row.get('output_inventory', []))}`",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", result["claim_boundary"]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"invocation_id": result["invocation_id"], **result["summary"]}, sort_keys=True))
    return 0 if successful else 3


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        adapter = root / "adapter"
        script = adapter / "scripts" / "remote" / "run_deform_dlo_source.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--dataset-root', required=True)\n"
            "p.add_argument('--output-dir', required=True)\n"
            "p.add_argument('--source', required=True)\n"
            "p.add_argument('--target', required=True)\n"
            "a=p.parse_args()\n"
            "o=Path(a.output_dir); o.mkdir(parents=True, exist_ok=True)\n"
            "(o/'result.json').write_text(json.dumps(vars(a)))\n",
            encoding="utf-8",
        )
        dataset = root / "data"
        dataset.mkdir()
        args = argparse.Namespace(
            adapter_root=adapter,
            dataset_root=dataset,
            output=root / "output",
            python=sys.executable,
            adapter_revision="1" * 40,
        )
        code = run(args)
        require(code == 0, "fixture invocation failed")
        value = json.loads((args.output / "invocation.json").read_text())
        require(value["summary"]["successful_execution_count"] == 2, "fixture count")
        require(value["invocation_id"] == content_id(value), "fixture content ID")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--adapter-revision", default="unknown")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.adapter_root is not None, "--adapter-root is required")
    require(args.dataset_root is not None, "--dataset-root is required")
    require(args.output is not None, "--output is required")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InvocationError as error:
        print(f"DEFORM adapter invocation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
