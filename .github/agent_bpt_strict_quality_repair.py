"""Repair the remaining strict-mypy failures on mature public interfaces."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def repair_cli_dispatch() -> None:
    path = "src/bayesian_phystwin/cli/main.py"
    replace_once(
        path,
        "        return commands_main(arguments[1:])\n",
        "        return int(commands_main(arguments[1:]))\n",
    )
    replace_once(
        path,
        "        return catalog_main(namespace, arguments[1:])\n",
        "        return int(catalog_main(namespace, arguments[1:]))\n",
    )
    replace_once(
        path,
        "        return invoke(*resolved)\n",
        "        return int(invoke(*resolved))\n",
    )


def repair_json_mapping(path: str, import_old: str | None = None) -> None:
    if import_old is not None:
        replace_once(path, import_old, import_old.rstrip("\n") + ", cast\n")
    replace_once(
        path,
        "        return json.loads(\n"
        "            json.dumps(dict(value), sort_keys=True, allow_nan=False)\n"
        "        )\n",
        "        return cast(\n"
        "            dict[str, Any],\n"
        "            json.loads(\n"
        "                json.dumps(dict(value), sort_keys=True, allow_nan=False)\n"
        "            ),\n"
        "        )\n",
    )


def repair_run_manifest_v2_mapping() -> None:
    path = "src/bayesian_phystwin/run_manifest_v2.py"
    replace_once(
        path,
        "        return json.loads(\n"
        "            json.dumps(dict(value), sort_keys=True, allow_nan=False)\n"
        "        )\n",
        "        return json.loads(  # type: ignore[no-any-return]\n"
        "            json.dumps(dict(value), sort_keys=True, allow_nan=False)\n"
        "        )\n",
    )


def main() -> None:
    repair_cli_dispatch()
    repair_json_mapping(
        "src/bayesian_phystwin/repository_provenance.py",
        "from typing import Any, Literal\n",
    )
    repair_json_mapping(
        "src/bayesian_phystwin/run_manifest.py",
        "from typing import Any, Literal\n",
    )
    repair_run_manifest_v2_mapping()


if __name__ == "__main__":
    main()
