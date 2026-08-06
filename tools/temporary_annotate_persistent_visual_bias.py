"""Temporary exact typing patch for PR #183; removed by its one-shot workflow."""

from pathlib import Path

SOURCE_PATH = Path("src/bayesian_phystwin/persistent_prob4d_visual_bias.py")


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {name} target, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "    provider_design = np.zeros(\n",
        "    provider_design: np.ndarray = np.zeros(\n",
        name="provider_design annotation",
    )
    source = replace_once(
        source,
        "    if physical_offset is None:\n        offset = np.zeros(\n",
        (
            "    offset: np.ndarray\n"
            "    if physical_offset is None:\n"
            "        offset = np.zeros(\n"
        ),
        name="physical_offset annotation",
    )
    SOURCE_PATH.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
