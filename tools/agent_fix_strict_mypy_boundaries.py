from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]

cli_path = root / "src/bayesian_phystwin/cli/main.py"
cli = cli_path.read_text(encoding="utf-8")
cli = replace_once(
    cli,
    "from typing import Final\n",
    "from typing import Final, cast\n",
    label="CLI cast import",
)
cli = replace_once(
    cli,
    "        return commands_main(arguments[1:])\n",
    "        return cast(int, commands_main(arguments[1:]))\n",
    label="commands exit code",
)
cli = replace_once(
    cli,
    "        return catalog_main(namespace, arguments[1:])\n",
    "        return cast(int, catalog_main(namespace, arguments[1:]))\n",
    label="catalog exit code",
)
cli = replace_once(
    cli,
    "        return invoke(*resolved)\n",
    "        return cast(int, invoke(*resolved))\n",
    label="dispatched exit code",
)
cli_path.write_text(cli, encoding="utf-8")

for relative_path in (
    "src/bayesian_phystwin/repository_provenance.py",
    "src/bayesian_phystwin/run_manifest.py",
):
    path = root / relative_path
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "from typing import Any, Literal\n",
        "from typing import Any, Literal, cast\n",
        label=f"{relative_path} cast import",
    )
    source = replace_once(
        source,
        '''        return json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
''',
        '''        return cast(
            dict[str, Any],
            json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)),
        )
''',
        label=f"{relative_path} JSON mapping cast",
    )
    path.write_text(source, encoding="utf-8")

manifest_v2_path = root / "src/bayesian_phystwin/run_manifest_v2.py"
manifest_v2 = manifest_v2_path.read_text(encoding="utf-8")
manifest_v2 = replace_once(
    manifest_v2,
    '''        return json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
''',
    '''        return cast(
            dict[str, Any],
            json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)),
        )
''',
    label="run_manifest_v2 JSON mapping cast",
)
manifest_v2_path.write_text(manifest_v2, encoding="utf-8")
