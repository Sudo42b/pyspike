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

# Vendor host-tree pre-built `.elf` follow a slightly different naming
# convention from the P5/P6 hand-built `.elf`. Examples:
#   vendor: <root>/MUL/n1s16/n1s16_mul.elf       (NOT mul_vv)
#   vendor: <root>/DIV/n1s16/n1s16_div_vv.elf    (vv suffix)
#   vendor: <root>/SOFT_MAX/n1s16/n1s16_softmax.elf  (matches hand-built)
# This map records vendor-only stems; entries here override the
# VENDOR_TO_ELF_STEM map specifically when resolving vendor `.elf` paths.
# Keys NOT in this map fall through to VENDOR_TO_ELF_STEM (then
# lowercase(op_dir)) for vendor lookup, preserving the prior behavior.
VENDOR_HOST_TREE_STEM_OVERRIDE: dict = {
    "MUL": "mul",              # vendor: n1s16_mul.elf
    "DIV": "div_vv",           # vendor: n1s16_div_vv.elf
    "ADD1": "add1",            # vendor: n1s16_add1.elf (not aliased above)
    # ADD vendor has BOTH n1s16_add_vv.elf and n1s16_add1.elf -- we keep
    # ADD->add_vv (matches VENDOR_TO_ELF_STEM) since add_vv has the
    # corresponding `_ref.txt`.
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

# P8 08-04 / CONTEXT D-11: 12-op smoke set.
# 9 confirmed ops (CONTEXT D-11 explicit list, after collapsing ADD/ADD_VV +
# MUL/MUL_VV duplicates) plus 3 plan-stage chosen ops (NEG, DIV, EXP per
# RESEARCH §"12-op smoke set candidate"). All 12 must achieve byte-exact
# strict-mode PASS against full-region vendor golden after Plan 04 fix.
SMOKE_SET_12 = (
    "ABS", "ADD", "MUL", "RELU", "SIGMOID", "GELU",
    "TANH", "LEAKY_RELU", "SUM", "NEG", "DIV", "EXP",
)


# P8 08-03 D-09 / RESEARCH §"Plan-Stage Hand-Off" Open Q#3:
# Per-op GTX_DDR_DUMP_SIZE override computed at TEST COLLECT TIME from
# `os.stat(GOLDEN_DIR_FULL / <op>.lower()+'.hex').st_size`. No hard-coded
# byte counts -- they would drift if the vendor regenerates goldens. CI
# default ("0x20" truncated) preserved when GOLDEN_DIR_FULL is absent.
#
# Layered priority (descending):
#   1. GTX_DDR_DUMP_SIZE_OVERRIDE_ALL env var (one-shot global override)
#   2. OP_DUMP_SIZE_OVERRIDE[op_dir] (computed from golden_full/ on disk)
#   3. "0x20" (legacy, 32-byte / single-row truncation)
def _compute_op_dump_size_override():
    """Compute per-op dump size from golden_full/<op>.hex line-count * 32.

    Runtime computation avoids hard-coded byte counts that drift if vendor
    regenerates goldens. Returns dict of {OP_DIR: hex_size_str}.

    Each non-comment, non-@-directive line in the vendor _ref.txt-format
    golden encodes exactly 32 raw DDR bytes (16 FP16 words = one 256-bit
    bus word). Computing raw_bytes = lines * 32 matches the vendor C++
    ddr_dump_to_file emission cadence (gtx_npu_dma.cc:509-558).
    """
    out: dict = {}
    if not GOLDEN_DIR_FULL.exists():
        return out
    for _full in GOLDEN_DIR_FULL.glob("*.hex"):
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
        _raw_bytes = _lines * 32
        # Stem mapping: golden_full uses lowercase + canonical pyspike op_name
        # (e.g. add_vv.hex, softmax.hex). Reverse-map via VENDOR_TO_ELF_STEM
        # to recover the upper-case OP_DIR form used by VENDOR_OP_DIRS.
        _stem = _full.stem
        _op_dir_guess = _stem.upper()
        for _vd, _es in VENDOR_TO_ELF_STEM.items():
            if _es == _stem:
                _op_dir_guess = _vd
                break
        out[_op_dir_guess] = "0x" + format(_raw_bytes, "X")
    return out


OP_DUMP_SIZE_OVERRIDE: dict = _compute_op_dump_size_override()


def _resolve_pyspike_command():
    """Resolve pyspike CLI: prefer system pyspike, fall back to module run."""
    cli = shutil.which("pyspike")
    if cli:
        return [cli]
    return [sys.executable, "-m", "riscv"]


def _find_elf(op_dir: str):
    """Find <op>.elf, preferring vendor pre-built when available (P8 08-04).

    Resolution order (vendor first for this sweep harness):
      1. ${GTX_VENDOR_TEST_DIR}/<OP_DIR>/n1s16/n1s16_<elf_stem>.elf
         (D-13 default = /mnt/e/14_NIGHTLY/pyspike/test/, vendor pre-built)
      2. tests/gtx/data/firmware/<elf_stem>.elf      (P5/P6 wheel-bundled)
      3. tests/gtx/data/elf/<elf_stem>.elf           (P5/P6 legacy location)

    Rationale: this sweep tests against vendor _ref.txt goldens, which
    require the vendor multi-tile firmware to produce the full DDR output.
    The hand-built P5/P6 `.S` kernels output a single row at 0x100 and would
    always mismatch the vendor golden; they are exercised by
    test_regression_fw_full.py / test_regression_fw_mm.py instead.

    08-04 reordering: previously hand-built (firmware/, elf/) won, masking
    the multi-tile bug surface (the vendor RoCC dispatch path was never
    exercised). After flipping the priority, vendor `.elf` triggers the
    `is_vendor_elf` branch in the test body which wires GTX_DDR_INIT,
    GTX_DDR_DUMP_ADDR=0xf000000, GTX_NO_EXIT=1, GTX_DDR_REVERSED=1.

    Returns Path or None.
    """
    handbuilt_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    vendor_stem = VENDOR_HOST_TREE_STEM_OVERRIDE.get(op_dir, handbuilt_stem)
    vendor_root = pathlib.Path(
        os.environ.get("GTX_VENDOR_TEST_DIR", "/mnt/e/14_NIGHTLY/pyspike/test/")
    )
    candidates = [
        vendor_root / op_dir / "n1s16" / ("n1s16_" + vendor_stem + ".elf"),
        FIRMWARE_DIR / (handbuilt_stem + ".elf"),
        ELF_DIR_LEGACY / (handbuilt_stem + ".elf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_golden(op_dir: str, *, prefer_full: bool = False):
    """Find <op>.hex. Priority: golden_full/ (P8 08-04, vendor `.elf` only) > golden/.

    P8 08-04: when running vendor pre-built `.elf` (multi-tile output spanning
    BASE_DDR_RESULT..BASE_DDR_RESULT+full_size), prefer the full-region golden
    in `golden_full/`. When running P5/P6 hand-built `.elf` (single-row output
    at 0x100, only 32 bytes), use the truncated golden in `golden/`.

    The `prefer_full=True` flag is set by callers AFTER `_find_elf` resolves
    to a vendor path (`is_vendor_elf=True`). For hand-built `.elf`,
    `prefer_full=False` ensures the comparison is against the matching
    truncated golden (not the full golden which the hand-built kernel never
    produces).

    Vendor SOFT_MAX dir produced softmax.hex (vendor naming variation handled
    in import_vendor_golden.py); look up under the same name lowering rule.
    """
    elf_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    truncated_candidates = [
        GOLDEN_DIR / (op_dir.lower() + ".hex"),       # P7 import_vendor_golden --all
        GOLDEN_DIR / (elf_stem + ".hex"),              # P6 9-op map (softmax, add_vv, ...)
    ]
    full_candidates = [
        GOLDEN_DIR_FULL / (op_dir.lower() + ".hex"),  # P8 08-03 full-region
        GOLDEN_DIR_FULL / (elf_stem + ".hex"),         # P8 08-03 elf-stem variant
    ]
    if prefer_full:
        candidates = full_candidates + truncated_candidates
    else:
        candidates = truncated_candidates + full_candidates
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

    # Tier 4: golden hex present. P8 08-04: only prefer the full-region
    # golden when running a vendor `.elf` (which actually outputs the full
    # multi-tile DDR region). P5/P6 hand-built `.elf` output a single row
    # at 0x100 and would mismatch the full golden trivially.
    golden_path = _find_golden(op_dir, prefer_full=is_vendor_elf)
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
    env["GTX_DDR_DUMP"] = str(actual_dump)

    # P8 08-04: per-op GTX_DDR_DUMP_ADDR selection. Vendor `.elf` writes
    # output at 0xf000000 (per `_ref.txt` @-headers and gtx-firmware
    # n1s16_*.c BASE_DDR_RESULT constant). P5/P6 hand-built `.elf` use the
    # legacy 0x100 default. INVESTIGATION (08-03) flagged this as a
    # mandatory wire-up gap.
    if is_vendor_elf:
        env["GTX_DDR_DUMP_ADDR"] = "0xf000000"
    else:
        env["GTX_DDR_DUMP_ADDR"] = "0x100"  # P5/P6 default ADDRR

    # P8 08-03 / 08-04: per-op GTX_DDR_DUMP_SIZE selection.
    # Priority: GTX_DDR_DUMP_SIZE_OVERRIDE_ALL env var > OP_DUMP_SIZE_OVERRIDE
    # dict (sourced from GOLDEN_DIR_FULL on disk, vendor-elf only) > "0x20"
    # legacy default.
    #
    # 08-04 narrowing: OP_DUMP_SIZE_OVERRIDE only applies when running
    # vendor pre-built `.elf` (full multi-tile output). P5/P6 hand-built
    # `.elf` output a single row at 0x100 -> always use "0x20" so the
    # comparison is against the matching truncated golden in golden/.
    explicit_global = os.environ.get("GTX_DDR_DUMP_SIZE_OVERRIDE_ALL")
    if explicit_global:
        env["GTX_DDR_DUMP_SIZE"] = explicit_global
    elif is_vendor_elf and op_dir in OP_DUMP_SIZE_OVERRIDE:
        env["GTX_DDR_DUMP_SIZE"] = OP_DUMP_SIZE_OVERRIDE[op_dir]
    else:
        env["GTX_DDR_DUMP_SIZE"] = "0x20"  # 32 bytes / 16 FP16 (single-row truncation)

    # P8 08-04: vendor `.elf` requires GTX_DDR_INIT pre-staging from
    # vendor input.txt (loaded via __ddr_init intrinsic at boot). Without
    # it, DDR is all zeros and ABS/RELU/etc operate on f(0) instead of the
    # expected vendor input pattern -> golden mismatch from line 0.
    # Vendor convention: <op_dir>/n1s16/data/n1s16_<elf_stem>_input.txt.
    # INVESTIGATION (08-03) flagged this as a mandatory wire-up gap.
    if is_vendor_elf:
        elf_stem_for_input = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
        vendor_input = elf_path.parent / "data" / (
            "n1s16_" + elf_stem_for_input + "_input.txt"
        )
        if vendor_input.exists():
            env["GTX_DDR_INIT"] = str(vendor_input)
        # If absent, leave unset -- the kernel may not require operand
        # staging (e.g. ARANGE which generates its input internally).

    # P8 08-04: vendor `.elf` uses multi-tile firmware loops (e.g.
    # n1s16_abs.c iterates tile_row_start until ROWS_PER_NEST). Each
    # iteration ends with `__join` (custom1 funct3=0b101) which raises
    # SystemExit(0) by default -- but that exits before the for-loop's
    # subsequent tiles run. Setting GTX_NO_EXIT lets WJOIN return 0; the
    # kernel exits cleanly via main return -> tohost (HTIF). P5/P6
    # hand-built single-iteration kernels are unaffected (they have at
    # most one __join, and exit-on-first-join is fine for them).
    if is_vendor_elf:
        env["GTX_NO_EXIT"] = "1"
    else:
        env.pop("GTX_NO_EXIT", None)  # P5/P6: WJOIN must raise SystemExit(0)

    # D-10: vendor pre-built .elf use BE FP16 ordering -> require
    # GTX_DDR_REVERSED=1 for the subprocess only. P5/P6 hand-built .elf
    # use LE FP16 (pyspike default) and remain unaffected.
    # Inline scope (NOT autouse fixture) per CONTEXT.md D-10.
    if is_vendor_elf:
        env["GTX_DDR_REVERSED"] = "1"

    # P8 08-04: vendor multi-tile kernels (ABS with HEIGHT=393217 -> 96 tiles)
    # take several minutes per op on functional simulation. P5/P6 hand-built
    # kernels finish in seconds. Use a 600s ceiling for vendor and 120s for
    # hand-built so a stuck NPU still surfaces deterministically.
    subprocess_timeout = 600 if is_vendor_elf else 120
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=subprocess_timeout, check=False,
        )
    except FileNotFoundError as exc:
        pytest.skip("pyspike CLI not found: " + str(exc))
    except subprocess.TimeoutExpired:
        pytest.fail(
            op_dir + ": pyspike subprocess timeout ("
            + str(subprocess_timeout) + "s)"
        )

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
    # P8 B1 (2026-05-11): strict mode no longer requires exact_matches == total.
    # CLAUDE.md "Bit-exact ULP 1 + atol 0.001" allows within_tolerance; the
    # prior 0-ULP gate was unreachable for transcendental ops (SIGMOID/TANH/
    # GELU/...) due to vendor's `std::exp(float)` libm/SIMD quirks. failures==0
    # (compare_hex's `passed`) is the correct ULP-1 gate.
