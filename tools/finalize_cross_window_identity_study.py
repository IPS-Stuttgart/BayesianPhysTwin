#!/usr/bin/env python3
"""One-shot branch finalizer for the cross-window identity development study."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_PROTOCOL_SHA256 = (
    "a9f8514bbf9ff27b393122a7f811d765c8edf4fe6fc867d4b883b5283058a9fd"
)


def _replace_between(path: Path, start_marker: str, stop_marker: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index(start_marker)
    stop = text.index(stop_marker, start)
    path.write_text(text[:start] + value + text[stop:], encoding="utf-8")


def _correct_selected_rows() -> None:
    path = ROOT / "scripts/science/prob4d_cross_window_identity_development_v1.py"
    replacement = '''def _selected_rows(
    group: GroupData,
    association: GroupAssociation,
    method_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = len(np.unique(group.stack.point_ids))
    specifications: list[tuple[int, int, int]] = []
    right_true = association.context.right_local_to_true

    def add_newest_window() -> None:
        for frame in (2, 3):
            specifications.extend((frame, true_id, true_id) for true_id in right_true)

    if method_id == FRAMEWISE:
        specifications.extend((3, true_id, true_id) for true_id in right_true)
    elif method_id == NEWEST_WINDOW:
        add_newest_window()
    elif method_id == NAIVE_MERGE:
        add_newest_window()
        for frame in (0, 1):
            specifications.extend(
                (frame, local_id, true_id)
                for local_id, true_id in enumerate(right_true)
            )
    elif method_id == SOURCE_LINKED:
        add_newest_window()
        for left_id, right_id in association.result.accepted_pairs:
            true_id = right_true[right_id]
            specifications.extend((frame, left_id, true_id) for frame in (0, 1))
    elif method_id == ORACLE_LINKED:
        add_newest_window()
        for true_id in right_true:
            specifications.extend((frame, true_id, true_id) for frame in (0, 1))
    else:
        raise ValueError(f"unknown observation method: {method_id}")

    row_indices = np.asarray(
        [frame * point_count + true_id for frame, true_id, _ in specifications],
        dtype=np.int64,
    )
    assigned_ids = np.asarray(
        [assigned for _, _, assigned in specifications],
        dtype=np.int64,
    )
    return row_indices, assigned_ids


'''
    _replace_between(path, "def _selected_rows(", "def _batch_for_method(", replacement)
    text = path.read_text(encoding="utf-8").replace("import hashlib\n", "")
    path.write_text(text, encoding="utf-8")


def _correct_self_hosted_workflow() -> None:
    path = ROOT / ".github/workflows/prob4d-cross-window-identity-development.yml"
    replacement = '''      - name: Compose exact Prob4D science source
        id: prob4d
        shell: bash
        run: |
          set -euo pipefail
          base=/home/github-runner/src/Prob4D
          additive=/home/github-runner/actions-runner/_work/Prob4D/Prob4D
          destination="${RUNNER_TEMP}/prob4d-cross-window-composed-source"
          test -d "${base}/.git"
          test -d "${additive}/.git"
          test "$(git -C "${base}" rev-parse HEAD)" = \
            aa8ffc6541011d044561e09870569a14ab3f586f
          test -z "$(git -C "${base}" status --porcelain=v1)"
          test "$(git -C "${base}" hash-object \
            src/prob4d/causal_tracklets.py)" = \
            81a44bf6a767464149fe8e60661d7ef699ff3e40
          test "$(git -C "${base}" hash-object src/prob4d/sim3.py)" = \
            1c3f443774f92d7fd580087aa393d793d6704c23
          test "$(git -C "${additive}" hash-object \
            src/prob4d/cross_window_tracklets.py)" = \
            1c90c67c68a42189aabf5fe26ef06a75c6c48065
          rm -rf "${destination}"
          git clone --no-hardlinks --quiet "${base}" "${destination}"
          cp "${additive}/src/prob4d/cross_window_tracklets.py" \
            "${destination}/src/prob4d/cross_window_tracklets.py"
          test "$(git -C "${destination}" hash-object \
            src/prob4d/cross_window_tracklets.py)" = \
            1c90c67c68a42189aabf5fe26ef06a75c6c48065
          {
            echo "controlled_generator_base=aa8ffc6541011d044561e09870569a14ab3f586f"
            echo "cross_window_revision=${PROB4D_SOURCE_REVISION}"
            echo "cross_window_blob=1c90c67c68a42189aabf5fe26ef06a75c6c48065"
            echo "causal_tracklets_blob=81a44bf6a767464149fe8e60661d7ef699ff3e40"
            echo "sim3_blob=1c3f443774f92d7fd580087aa393d793d6704c23"
          } | tee "${destination}/SCIENCE_SOURCE_COMPOSITION.txt"
          echo "path=${destination}" >> "${GITHUB_OUTPUT}"

'''
    _replace_between(
        path,
        "      - name: Resolve exact Prob4D source",
        "      - name: Install exact producer and consumer",
        replacement,
    )
    text = path.read_text(encoding="utf-8")
    import re

    text = re.sub(
        r"PROTOCOL_SHA256: [0-9a-f]{64}",
        f"PROTOCOL_SHA256: {FINAL_PROTOCOL_SHA256}",
        text,
    )
    path.write_text(text, encoding="utf-8")


def _update_documentation() -> None:
    path = ROOT / "docs/prob4d_cross_window_identity_development_v1.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "`P2_source_linked_cross_window_identity` | Merge only unambiguous mutual-best Prob4D links |",
        "`P2_source_linked_cross_window_identity` | Newest-window reference plus only unambiguous mutual-best older-window links; zero links equal `P0` exactly |",
    )
    text = text.replace(
        "resolves the exact Prob4D source revision",
        "composes the clean controlled-study Prob4D base with the exact content-addressed additive association module",
    )
    path.write_text(text, encoding="utf-8")


def _remove_obsolete_workflows() -> None:
    for relative in (
        ".github/workflows/prob4d-cross-window-identity-development-hosted.yml",
        ".github/workflows/temporary-prob4d-local-source-diagnostic.yml",
        ".github/workflows/temporary-format-cross-window-identity-study.yml",
    ):
        (ROOT / relative).unlink(missing_ok=True)


def main() -> None:
    _correct_selected_rows()
    _correct_self_hosted_workflow()
    _update_documentation()
    _remove_obsolete_workflows()


if __name__ == "__main__":
    main()
