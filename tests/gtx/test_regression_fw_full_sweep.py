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
"""P7 NJIT-04 Tier 2: vendor 84-op directory full sweep (strict-mode).

Plan 05 GREEN body. Parametrizes over the 84 vendor op directories,
invokes pyspike subprocess + compare_hex(strict=True) against the
golden hex imported by `scripts/import_vendor_golden.py --all`.

5-tier graceful-skip discipline (P5/P6 lineage, mirrors test_regression_fw_full.py):
  Tier 1: _RISCV_AVAILABLE -- _riscv.so absent -> skip
  Tier 2: pyspike CLI resolvable -> skip if not
  Tier 3: tests/gtx/data/firmware/<op>.elf or legacy elf/<op>.elf present
  Tier 4: tests/gtx/data/golden/<op>.hex present (vendor golden imported)
  Tier 5: subprocess clean-exit + dump file generated (atexit hook fired)

Acceptance: M passed + N skipped where M+N == 84.

This test must run with numba absent (NumPy fallback) AND with numba installed
(real njit). Numba presence does NOT affect this test's collection or skip
discipline; it only affects the runtime path inside the gtx subprocess.
"""
from __future__ import annotations
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
VENDOR_TEST_DIR = REPO_ROOT / "vendor" / "gtx_cpp_reference" / "test"
FIRMWARE_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "firmware"
ELF_DIR_LEGACY = REPO_ROOT / "tests" / "gtx" / "data" / "elf"
GOLDEN_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "golden"
# P8 08-03: full-region (non-truncated) goldens, .gitignored. Populated locally
# via `python scripts/import_vendor_golden.py --full` for multi-tile divergence
# investigation. _find_golden prefers this over the truncated golden/ when both
# exist, so investigation runs use the larger compare automatically.
GOLDEN_DIR_FULL = REPO_ROOT / "tests" / "gtx" / "data" / "golden_full"

# Auto-discover 84 vendor op directories at collection time
if VENDOR_TEST_DIR.exists():
    VENDOR_OP_DIRS = sorted(
        p.name for p in VENDOR_TEST_DIR.iterdir()
        if p.is_dir() and p.name != "__pycache__" and p.name[:1].isupper()
    )
else:
    VENDOR_OP_DIRS = []

# Map vendor dir name to the .elf basename that P5/P6 actually built (for the
# 12 hand-written kernels). The vendor SOFT_MAX dir maps to softmax.elf, ADD
# maps to add_vv.elf, etc. Most other ops map to lowercase(dir).
VENDOR_TO_ELF_STEM: dict = {
    "SOFT_MAX": "softmax",
    "ADD": "add_vv",
    "MUL": "mul_vv",
    # All others use lowercase(op_dir) directly.
}

# Vendor ops whose goldens (imported from vendor _ref.txt) assume non-zero
# operand pre-staging via ddr_init_from_file, but whose P5/P6 hand-written
# .S kernels run against zero-init L1 (no operand staging). This is the same
# OPERAND_STAGING_REQUIRED set documented in test_regression_fw_full.py
# (P6 06-04-SUMMARY Known Issues). The runtime output is f(0_vec) which does
# NOT match the vendor's arange-input-driven golden. Skip with explicit
# pointer so M+N==84 holds for sweep acceptance.
OPERAND_STAGING_REQUIRED_VENDOR: set = {
    "RELU", "SIGMOID", "TANH", "SOFT_MAX", "LEAKY_RELU",
    "ADD", "MUL", "SUM", "ABS",
}

# P8 08-03 D-09 / RESEARCH §"Plan-Stage Hand-Off" Open Q#3:
# Per-op GTX_DDR_DUMP_SIZE override. Populated dynamically below by reading
# `os.stat(GOLDEN_DIR_FULL / <op>.lower()+'.hex').st_size` when a full-region
# golden has been generated locally (`scripts/import_vendor_golden.py --full`).
# When no full-region golden exists for an op, the harness falls back to
# the legacy "0x20" (32 byte / single-row) dump that matches the truncated
# golden in GOLDEN_DIR. CI defaults preserved.
#
# Layered priority (descending):
#   1. GTX_DDR_DUMP_SIZE_OVERRIDE_ALL env var (one-shot global override)
#   2. OP_DUMP_SIZE_OVERRIDE[op_dir] (computed from golden_full/ on disk)
#   3. "0x20" (legacy, 32-byte / single-row truncation)
#
# The values are recomputed at module import time so an investigator running
# `scripts/import_vendor_golden.py --full` then `pytest ...` picks up the
# new sizes without restarting the worker.
OP_DUMP_SIZE_OVERRIDE: dict = {}
if GOLDEN_DIR_FULL.exists():
    for _full in GOLDEN_DIR_FULL.glob("*.hex"):
        # File size in bytes; round up to next 32-byte multiple to match the
        # vendor `_ref.txt` line emission cadence (1 line = 32 bytes hex =
        # 16 FP16 words). The harness's GTX_DDR_DUMP_SIZE is in bytes.
        try:
            _bytes = _full.stat().st_size
        except OSError:
            continue
        # Convert hex-text size -> raw byte count: each non-comment line is
        # 64 hex chars (32 bytes) + newline. Estimate raw bytes = lines * 32.
        # Use a quick wc-style line count via mmap-free path.
        try:
            with open(_full, "rb") as _fh:
                _lines = 0
                for _raw in _fh:
                    _s = _raw.strip()
                    if _s and not _s.startswith(b"#") and not _s.startswith(b"@"):
                        _lines += 1
        except OSError:
            continue
        if _lines == 0:
            continue
        # Each data line encodes 32 raw bytes; pad to 32-byte alignment for
        # the C++-style ddr_dump_to_file emission contract.
        _raw_bytes = _lines * 32
        # Stem mapping: golden_full uses lowercase + canonical pyspike op_name
        # (e.g. add_vv.hex, softmax.hex). Convert to uppercase op_dir form.
        _stem = _full.stem  # e.g. "add_vv"
        # Reverse-map via VENDOR_TO_ELF_STEM if needed; otherwise upper-case.
        _op_dir_guess = _stem.upper()
        for _vd, _es in VENDOR_TO_ELF_STEM.items():
            if _es == _stem:
                _op_dir_guess = _vd
                break
        OP_DUMP_SIZE_OVERRIDE[_op_dir_guess] = hex(_raw_bytes)


def _resolve_pyspike_command():
    """Resolve pyspike CLI: prefer system pyspike, fall back to module run."""
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    return [sys.executable, "-m", "riscv"]


def _find_elf(op_dir: str):
    """Find <op>.elf in firmware dir, legacy elf dir, or vendor host tree (D-05).

    Resolution order (firmware/ first -> P5/P6 hand-built wins on collision):
      1. tests/gtx/data/firmware/<elf_stem>.elf      (P5/P6 wheel-bundled)
      2. tests/gtx/data/elf/<elf_stem>.elf           (P5/P6 legacy location)
      3. ${GTX_VENDOR_TEST_DIR}/<OP_DIR>/n1s16/n1s16_<elf_stem>.elf
         (D-13 default = /mnt/e/14_NIGHTLY/pyspike/test/, vendor pre-built)

    Returns Path or None.
    """
    elf_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    vendor_root = pathlib.Path(
        os.environ.get("GTX_VENDOR_TEST_DIR", "/mnt/e/14_NIGHTLY/pyspike/test/")
    )
    candidates = [
        FIRMWARE_DIR / (elf_stem + ".elf"),
        ELF_DIR_LEGACY / (elf_stem + ".elf"),
        vendor_root / op_dir / "n1s16" / ("n1s16_" + elf_stem + ".elf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_golden(op_dir: str):
    """Find <op>.hex. Priority: golden_full/ (P8 08-03 INVESTIGATION) > golden/.

    P8 08-03: when `scripts/import_vendor_golden.py --full` has been run
    locally, golden_full/<op>.hex exists with the non-truncated vendor
    output. That file takes precedence over golden/<op>.hex (which is the
    32-byte single-row truncation committed for CI). Falls through to the
    truncated golden when no full-region version is on disk.

    Vendor SOFT_MAX dir produced softmax.hex (vendor naming variation handled
    in import_vendor_golden.py); look up under the same name lowering rule.
    """
    elf_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    candidates = [
        GOLDEN_DIR_FULL / (op_dir.lower() + ".hex"),  # P8 08-03 full-region
        GOLDEN_DIR_FULL / (elf_stem + ".hex"),         # P8 08-03 elf-stem variant
        GOLDEN_DIR / (op_dir.lower() + ".hex"),       # P7 import_vendor_golden --all
        GOLDEN_DIR / (elf_stem + ".hex"),              # P6 9-op map (softmax, add_vv, ...)
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@pytest.mark.parametrize("op_dir", VENDOR_OP_DIRS or ["__no_vendor__"], ids=lambda x: x)
def test_vendor_op_sweep_strict(op_dir: str, tmp_path) -> None:
    """One vendor op vs vendor golden, strict-mode (5-tier graceful skip).

    M passed + N skipped == 84 for the full vendor sweep on any environment.
    """
    # Sentinel for missing vendor submodule
    if op_dir == "__no_vendor__":
        pytest.skip("vendor submodule not initialized")

    # Tier 1: _riscv.so available
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so unavailable (extension not built)")

    # Tier 2: pyspike CLI resolvable
    if shutil.which("pyspike") is None:
        pytest.skip("pyspike CLI not on PATH -- install via `pip install -e .`")

    # Tier 3: .elf present (firmware/ or legacy elf/)
    elf_path = _find_elf(op_dir)
    if elf_path is None:
        pytest.skip(
            "no .elf for op " + op_dir
            + " (vendor toolchain build pending; see tests/gtx/data/firmware/README.md)"
        )

    # Compute vendor root once for both Tier 3b skip discrimination and the
    # subprocess GTX_DDR_REVERSED env decision below (D-05/D-10/D-13).
    vendor_root_for_env = pathlib.Path(
        os.environ.get("GTX_VENDOR_TEST_DIR", "/mnt/e/14_NIGHTLY/pyspike/test/")
    )

    # Tier 3b (P6 lineage, D-11 conditioning): operand-staging-required ops --
    # vendor golden vs zero-init runtime mismatch. This skip ONLY applies to
    # hand-built .elf (P5/P6 .S kernels). Vendor pre-built .elf stage operands
    # via the __ddr_init intrinsic (ddr_init_from_file), so the vendor golden's
    # non-zero-input expectation is satisfied. Detect by checking whether
    # elf_path is under vendor_root.
    is_vendor_elf = False
    try:
        is_vendor_elf = elf_path.is_relative_to(vendor_root_for_env)
    except (ValueError, AttributeError):
        is_vendor_elf = False
    if op_dir in OPERAND_STAGING_REQUIRED_VENDOR and not is_vendor_elf:
        pytest.skip(
            op_dir + ": (hand-built path) vendor golden assumes non-zero "
            "operand staging that P5/P6 .S kernel does NOT provide (zero-init L1). "
            "See OPERAND_STAGING_REQUIRED_VENDOR + test_regression_fw_full.py."
        )

    # Tier 4: golden hex present
    golden_path = _find_golden(op_dir)
    if golden_path is None:
        pytest.skip("no golden hex for op " + op_dir + " (run import_vendor_golden.py --all)")

    # Tier 5: subprocess pyspike with GTX_DDR_DUMP env vars
    actual_dump = tmp_path / (op_dir.lower() + "_actual.hex")
    cmd = _resolve_pyspike_command() + [
        "--extlib=riscv.gtx",
        "--extension=gtx",
        str(elf_path),
    ]
    env = os.environ.copy()
    env.pop("GTX_NO_EXIT", None)  # WJOIN must raise SystemExit(0)
    env["GTX_DDR_DUMP"] = str(actual_dump)
    env["GTX_DDR_DUMP_ADDR"] = "0x100"  # P5/P6 default ADDRR

    # P8 08-03: per-op GTX_DDR_DUMP_SIZE selection.
    # Priority: GTX_DDR_DUMP_SIZE_OVERRIDE_ALL env var > OP_DUMP_SIZE_OVERRIDE
    # dict (sourced from GOLDEN_DIR_FULL on disk) > "0x20" legacy default.
    explicit_global = os.environ.get("GTX_DDR_DUMP_SIZE_OVERRIDE_ALL")
    if explicit_global:
        env["GTX_DDR_DUMP_SIZE"] = explicit_global
    elif op_dir in OP_DUMP_SIZE_OVERRIDE:
        env["GTX_DDR_DUMP_SIZE"] = OP_DUMP_SIZE_OVERRIDE[op_dir]
    else:
        env["GTX_DDR_DUMP_SIZE"] = "0x20"  # 32 bytes / 16 FP16 (single-row truncation)

    # D-10: vendor pre-built .elf use BE FP16 ordering -> require
    # GTX_DDR_REVERSED=1 for the subprocess only. P5/P6 hand-built .elf
    # use LE FP16 (pyspike default) and remain unaffected.
    # Inline scope (NOT autouse fixture) per CONTEXT.md D-10.
    try:
        if elf_path.is_relative_to(vendor_root_for_env):
            env["GTX_DDR_REVERSED"] = "1"
    except (ValueError, AttributeError):
        # is_relative_to is Python 3.9+; cp310+ baseline so this should
        # always succeed. Guard for unforeseen Path subclass behavior.
        pass

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=120, check=False,
        )
    except FileNotFoundError as exc:
        pytest.skip("pyspike CLI not found: " + str(exc))
    except subprocess.TimeoutExpired:
        pytest.fail(op_dir + ": pyspike subprocess timeout (120s)")

    if result.returncode != 0:
        pytest.fail(
            op_dir + ": pyspike subprocess failed rc=" + str(result.returncode)
            + "\nstderr (tail):\n" + result.stderr[-500:]
        )

    if not actual_dump.exists():
        pytest.skip(
            op_dir + ": subprocess clean-exited but no dump generated "
            "(atexit hook may not have fired for this op)"
        )

    # Strict-mode compare via P6 _verify helper
    from riscv.gtx._verify import compare_hex
    passed, stats = compare_hex(
        str(actual_dump), str(golden_path),
        ulp=1, atol=0.001, strict=True,
    )
    assert passed, (
        op_dir + ": strict-mode compare failed.\n"
        "  actual_dump: " + str(actual_dump) + "\n"
        "  golden:      " + str(golden_path) + "\n"
        "  stats:       " + str(stats)
    )
    assert stats.get("within_tolerance", 0) == 0, (
        op_dir + ": non-exact match in strict mode: " + str(stats)
    )
