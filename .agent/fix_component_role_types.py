from pathlib import Path

path = Path("src/bayesian_phystwin/deform360_calibration_execution.py")
text = path.read_text(encoding="utf-8")
replacements = (
    (
        "    Deform360CalibrationArtifactRefV1,\n"
        "    Deform360CalibrationBundleV1,\n",
        "    Deform360CalibrationArtifactRefV1,\n"
        "    Deform360CalibrationBundleV1,\n"
        "    Deform360CalibrationRole,\n",
    ),
    (
        "_COMPONENT_ROLES: Mapping[str, tuple[str, ...]] = {\n",
        "_COMPONENT_ROLES: Mapping[\n"
        "    str, tuple[Deform360CalibrationRole, ...]\n"
        "] = {\n",
    ),
)
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one component-role type anchor, found {count}")
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
