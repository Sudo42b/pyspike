"""Wave 2 strict-mode .elf regression for activation_relu_gelu (test_regression_fw_act.py).

Mirrors test_regression_fw_mm.py shape (P4 04-01 4-tier graceful skip):
  1. _RISCV_AVAILABLE False -> skip
  2. activation_relu_gelu.elf missing -> skip
  3. activation_relu_gelu.hex (golden) missing -> skip
  4. pyspike CLI not on PATH -> skip
Plus 5th-tier graceful skip if subprocess does not honor GTX_DDR_DUMP
(older pyspike build without P3 D-04 dump infrastructure -- same state as P4).

Plan 06 wave 2 GREEN-fills the assertion body. Plan 01 ships the imports +
4-tier skip + pytest.skip stub so the file collects cleanly.
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
ELF_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "activation_relu_gelu.elf"
GOLDEN_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "golden" / "activation_relu_gelu.hex"


def _resolve_pyspike_command():
    """Resolve pyspike CLI: prefer `pyspike` on PATH, fall back to `python -m riscv`.
    Mirrors P4 D-11 + P2 plan-05 pattern."""
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    return [sys.executable, "-m", "riscv"]


def test_act_strict_mode_pass(tmp_path):
    """Plan 06 wave 2 GREEN-fills strict compare. Plan 01 ships 4-tier skip."""
    pytest.skip("Plan 06 wave 2 GREEN-fills strict-mode .elf regression body")
