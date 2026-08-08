from __future__ import annotations

from pathlib import Path

path = Path(".github/workflows/deform360-calibration-visual-production.yml")
source = path.read_text(encoding="utf-8")
old = '''          python3 - <<'PY'
          import os
          from pathlib import Path

          values = {
              "processed": os.environ["PROCESSED_ROOT_INPUT"],
              "output": os.environ["OUTPUT_ROOT_INPUT"],
              "cache": os.environ["HF_CACHE_INPUT"],
              "workspace": os.environ["GITHUB_WORKSPACE"],
          }
          roots = {
              name: Path(value).expanduser().resolve(strict=False)
              for name, value in values.items()
          }
          if not (roots["processed"] / "aligned").is_dir():
              raise SystemExit("protected calibration aligned root is missing")

          def overlap(left: Path, right: Path) -> bool:
              return left == right or left in right.parents or right in left.parents

          for other in ("processed", "cache", "workspace"):
              if overlap(roots["output"], roots[other]):
                  raise SystemExit(f"output root overlaps {other} root")
          if overlap(roots["processed"], roots["cache"]):
              raise SystemExit("model cache overlaps retained calibration root")

          output = Path(os.environ["GITHUB_ENV"])
          with output.open("a", encoding="utf-8") as stream:
              stream.write(f"PROCESSED_ROOT={roots['processed']}\\n")
              stream.write(f"OUTPUT_ROOT={roots['output']}\\n")
              stream.write(f"HF_CACHE_DIR={roots['cache']}\\n")
              stream.write(
                  "PRODUCTION_RUN_ROOT="
                  f"{roots['output'] / os.environ['ADMISSION_ID']}\\n"
              )
          PY

          mkdir -p -- "${OUTPUT_ROOT}" "${HF_CACHE_DIR}"
'''
new = '''          mapfile -t resolved_roots < <(
            python3 - <<'PY'
          import os
          from pathlib import Path

          values = {
              "processed": os.environ["PROCESSED_ROOT_INPUT"],
              "output": os.environ["OUTPUT_ROOT_INPUT"],
              "cache": os.environ["HF_CACHE_INPUT"],
              "workspace": os.environ["GITHUB_WORKSPACE"],
          }
          roots = {
              name: Path(value).expanduser().resolve(strict=False)
              for name, value in values.items()
          }
          if not (roots["processed"] / "aligned").is_dir():
              raise SystemExit("protected calibration aligned root is missing")

          def overlap(left: Path, right: Path) -> bool:
              return left == right or left in right.parents or right in left.parents

          for other in ("processed", "cache", "workspace"):
              if overlap(roots["output"], roots[other]):
                  raise SystemExit(f"output root overlaps {other} root")
          if overlap(roots["processed"], roots["cache"]):
              raise SystemExit("model cache overlaps retained calibration root")

          for value in (roots["processed"], roots["output"], roots["cache"]):
              rendered = str(value)
              if "\\n" in rendered or "\\r" in rendered:
                  raise SystemExit("resolved root contains a line break")
              print(rendered)
          PY
          )
          if [[ "${#resolved_roots[@]}" -ne 3 ]]; then
            echo "root resolver did not return exactly three paths" >&2
            exit 1
          fi
          PROCESSED_ROOT="${resolved_roots[0]}"
          OUTPUT_ROOT="${resolved_roots[1]}"
          HF_CACHE_DIR="${resolved_roots[2]}"
          PRODUCTION_RUN_ROOT="${OUTPUT_ROOT}/${ADMISSION_ID}"
          export PROCESSED_ROOT OUTPUT_ROOT HF_CACHE_DIR PRODUCTION_RUN_ROOT
          {
            echo "PROCESSED_ROOT=${PROCESSED_ROOT}"
            echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
            echo "HF_CACHE_DIR=${HF_CACHE_DIR}"
            echo "PRODUCTION_RUN_ROOT=${PRODUCTION_RUN_ROOT}"
          } >> "${GITHUB_ENV}"

          mkdir -p -- "${OUTPUT_ROOT}" "${HF_CACHE_DIR}"
'''
if source.count(old) != 1:
    raise SystemExit("protected-root resolution block changed")
path.write_text(source.replace(old, new), encoding="utf-8")

# Push after the finalizer workflow exists so this reviewed transform is executed.
