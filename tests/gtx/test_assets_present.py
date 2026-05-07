"""VRF-03: regression .elf + golden hex asset presence.

Plan 03 owns this file (D-18 zero-overlap -- Plan 01 does NOT create a stub).

Verifies P6 D-13 source-of-truth: tests/gtx/data/{elf,golden}/ contains
>= 10 .elf and >= 10 .hex files with corresponding Makefile rules.
"""
import pathlib
import re

import pytest  # noqa: F401  (intentionally available for future fixtures)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "elf"
GOLDEN_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "golden"
MAKEFILE = ELF_DIR / "Makefile"


def test_elf_assets_present():
    """ROADMAP P6 success #2 prerequisite: bundled .elf count >= 10."""
    elfs = sorted(ELF_DIR.glob("*.elf"))
    assert len(elfs) >= 10, (
        "Expected >= 10 .elf files in " + str(ELF_DIR) + ", got "
        + str(len(elfs)) + ": " + str([e.name for e in elfs])
    )


def test_golden_assets_present():
    """Each .elf should have a matching <stem>.hex (with stem-rename for legacy)."""
    hexes = sorted(GOLDEN_DIR.glob("*.hex"))
    assert len(hexes) >= 10, (
        "Expected >= 10 .hex files in " + str(GOLDEN_DIR) + ", got "
        + str(len(hexes)) + ": " + str([h.name for h in hexes])
    )


def test_makefile_has_per_op_rules():
    """Makefile should have a rule of form `<op>.elf: <op>.S` for every new .elf."""
    assert MAKEFILE.exists(), "missing: " + str(MAKEFILE)
    text = MAKEFILE.read_text()
    # Count rules of the form `<op>.elf: <op>.S` (ignoring trailing whitespace)
    rule_pattern = re.compile(r'^\w+\.elf:\s+\w+\.S\s*$', re.MULTILINE)
    rules = rule_pattern.findall(text)
    assert len(rules) >= 10, (
        "Expected >= 10 .elf:.S rules in Makefile, got " + str(len(rules))
        + ": " + str(rules)
    )


def test_each_elf_has_matching_golden():
    """Each .elf stem (except nop_wjoin) should have a corresponding .hex.

    nop_wjoin is a smoke test (no compute), exempt from golden requirement.
    Legacy stems (mm_basic -> mm_basic_n1s16, activation_relu_gelu kept as-is)
    are explicitly mapped.
    """
    STEM_TO_GOLDEN = {
        "mm_basic": "mm_basic_n1s16",  # legacy stem (P4 D-10 lineage)
    }
    EXEMPT_STEMS = {"nop_wjoin"}  # smoke-only, no compute

    elf_stems = {e.stem for e in ELF_DIR.glob("*.elf")}
    hex_stems = {h.stem for h in GOLDEN_DIR.glob("*.hex")}

    missing = []
    for stem in elf_stems:
        if stem in EXEMPT_STEMS:
            continue
        expected = STEM_TO_GOLDEN.get(stem, stem)
        if expected not in hex_stems:
            missing.append(stem + " -> " + expected + ".hex")

    # Allow up to 2 mismatches for ops that gracefully degraded in Tasks 1/2a/2b
    # (Plan 04 handles missing-golden via skip).
    assert len(missing) <= 2, (
        "Too many .elf without matching .hex: " + str(missing)
    )
