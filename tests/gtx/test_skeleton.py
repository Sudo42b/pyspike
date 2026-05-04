#
# Copyright 2026 WuXi EsionTech Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Integration test: ROADMAP P2 success criterion 1.

`pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf` exits 0:
    - GtxNpu loads via @isa.register('gtx')
    - reset() initializes sp = 0x80100000
    - addi sp, sp, -16 does NOT trap (sp = 0x800FFFF0 valid DRAM)
    - .insn r 0x2b, 0b101, ... triggers custom1 funct3=0b101 (WJOIN)
    - WJOIN raises SystemExit(0) (GTX_NO_EXIT unset)
    - Spike process exits with code 0

Two-tier skip:
    1. _riscv.so missing -> skip (GtxNpu cannot load).
    2. nop_wjoin.elf missing -> skip (cross-toolchain not available at build time).
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest


# Module-level detection -- self-contained so --noconftest still works.
try:  # pragma: no cover
    # pylint: disable=import-error,no-name-in-module,unused-import
    from riscv.processor import processor_t  # noqa: F401
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "nop_wjoin.elf"


def _resolve_pyspike_command():
    """Prefer `pyspike` on PATH; fall back to `python -m riscv`."""
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    # Fallback: invoke via python -m if the package supports it
    return [sys.executable, "-m", "riscv"]


def test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero():
    """ROADMAP P2 success criterion 1: NOP firmware exits cleanly."""
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built -- pyspike CLI cannot dispatch RoCC")
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing -- build via `make -C tests/gtx/data/elf`")

    cmd = _resolve_pyspike_command() + [
        "--extlib=riscv.gtx",
        str(ELF_PATH),
    ]
    env = os.environ.copy()
    # Ensure GTX_NO_EXIT is not set so WJOIN raises SystemExit -> Spike exits 0
    env.pop("GTX_NO_EXIT", None)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"pyspike CLI not found: {exc}")
    except subprocess.TimeoutExpired:
        pytest.fail("pyspike timed out -- likely WJOIN SystemExit not propagating")

    assert result.returncode == 0, (
        f"pyspike exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )


def test_elf_fixture_exists_or_documented():
    """Always-runnable: at least the .S source must be committed (D-22).

    If .elf is missing, the integration test above skips -- but the source
    + Makefile must be present so CI can build it."""
    s_path = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "nop_wjoin.S"
    mk_path = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "Makefile"
    assert s_path.exists(), f"missing: {s_path}"
    assert mk_path.exists(), f"missing: {mk_path}"
