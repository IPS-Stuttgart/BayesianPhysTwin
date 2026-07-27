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


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        raise RuntimeError(f"{path}: repair marker already present")
    target.write_text(text.rstrip() + "\n\n\n" + addition.strip() + "\n", encoding="utf-8")


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
        "        payload = json.dumps(dict(value), sort_keys=True, allow_nan=False)\n"
        "        result = json.loads(payload)\n"
        "        return cast(dict[str, Any], result)\n",
    )


def repair_run_manifest_v2_tests() -> None:
    append_once(
        "tests/test_run_manifest_v2.py",
        "test_v2_rejects_empty_run_id",
        '''
def test_v2_rejects_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run ID must be nonempty"):
        replace(_manifest(tmp_path), run_id="")
''',
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
    repair_run_manifest_v2_tests()


if __name__ == "__main__":
    main()
