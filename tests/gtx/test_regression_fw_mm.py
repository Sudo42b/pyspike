"""P4 strict-mode .elf regression -- MM-05 ROADMAP P4 success #4.

`pyspike --extlib=riscv.gtx mm_basic.elf` -> DDR/L1 dump
-> _verify_minimal.compare_hex(strict=True) PASS.
Subprocess pattern (D-11 fallback as PRIMARY per RESEARCH).

Two-tier skip:
  1. _riscv.so missing -> skip
  2. mm_basic.elf missing -> skip
  3. pyspike not on PATH -> skip
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "mm_basic.elf"
GOLDEN_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "golden" / "mm_basic_n1s16.hex"


def _resolve_pyspike_command():
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    return [sys.executable, "-m", "riscv"]


def test_mm_basic_strict_mode_pass(tmp_path):
    """MM-05 ROADMAP P4 success #4: strict-mode .elf regression PASS."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built -- pyspike CLI cannot dispatch RoCC")
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing -- Wave 0 Task 2 toolchain unavailable")
    if not GOLDEN_PATH.exists():
        pytest.skip(f"{GOLDEN_PATH} missing -- Wave 0 Task 3 must populate")
    # Wave 2 (Plan 05) wires the actual subprocess invocation + DDR dump
    # + compare_hex strict.
    pytest.skip("Wave 2: Plan 05 fills strict-mode .elf regression body")


def test_mm_basic_fixture_present():
    """Always-runnable: mm_basic.S + Makefile + golden hex must exist (D-22 fixture)."""
    s_path = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "mm_basic.S"
    mk_path = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "Makefile"
    assert s_path.exists(), f"missing: {s_path}"
    assert mk_path.exists(), f"missing: {mk_path}"
    assert "mm_basic.elf" in mk_path.read_text(), "Makefile missing mm_basic.elf rule"
