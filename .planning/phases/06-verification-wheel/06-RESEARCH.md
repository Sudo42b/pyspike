# Phase 6: Verification & Wheel - Research

**Researched:** 2026-05-07
**Domain:** Python packaging (setuptools 75+ + cibuildwheel) + verify CLI port + Python `atexit` semantics + bundled-data resource access
**Confidence:** HIGH overall (key technical questions resolved with direct experiments / source reads / live Python checks)

## Summary

Phase 6 is the v1 ship gate. Its 6 requirements (VRF-01, VRF-03, VRF-04, PKG-01, PKG-03, PKG-04) reduce to 5 concrete deliverables already locked by CONTEXT.md decisions D-01..D-18. Research for this phase is therefore **gap-filling**, not exploration: each open question (vendor verify.py API surface, _ref.txt format, atexit semantics, NPU lookup mechanism, setuptools build_py + cibuildwheel compat, importlib.resources stability across cp310-cp312, console_scripts syntax, asset bundling math, op set selection) has a concrete answer driven by reading the vendor source, running live experiments, or consulting current setuptools/Python docs.

The biggest research-confirmed simplifications:

1. **Vendor `_ref.txt` format is byte-for-byte identical to our existing `.hex` format.** Both use `@<addr>` directive lines + 64-hex-char (32-byte) data lines. The existing `tests/gtx/_verify_minimal._parse_hex` parses vendor files **directly with zero changes** (verified via live Python invocation, parsed 524288 bytes from `n1s16_relu_ref.txt` cleanly). This removes the need for a "conversion" script — D-11/D-12 collapse to **a copy + rename + truncate-to-32-bytes script**.

2. **Python `atexit` fires correctly on `SystemExit(0)` in cp310-cp312.** Verified live: `atexit.register(...) + sys.exit(0)` runs the callback and returns exit code 0. The CPython issue #103512 (atexit prints traceback) is about `SystemExit` raised **inside** an atexit handler, NOT about firing atexit when SystemExit is raised at top level. CONTEXT D-04 is technically sound. However, atexit does NOT fire across `subprocess.run(...)` — the parent process's atexit handlers run in the parent, not in the child. The child subprocess registers its own atexit (via the gtx package import) and fires that on its own SystemExit.

3. **vendor `n1s16_<op>.c` sources have a non-trivial build dependency chain** (`intrin.h`, `gtx/address.h`, `gtx_csr.h` from `gtx-firmware/include/`) — meaning building these .elf files for D-08 requires the full GTX firmware include tree, NOT just `/opt/riscv` toolchain. **Recommendation: do NOT build vendor .elf for v1 P6 wheel** — instead either (a) carve out P3 D-22 hand-written .S kernels per op (P4 mm_basic.S precedent), or (b) commit pre-built .elf binaries from the developer machine (P2 D-22 / P4 D-09 / P5 D-04 lineage).

**Primary recommendation:** Plan 03 (VRF-03) should commit P5-style hand-written `.S` per op + Makefile rule + pre-built `.elf` (~1-3KB each) + truncated `<op>.hex` (32-byte single-row golden derived from vendor `_ref.txt` first data line). Total wheel asset cost: ~20 ops × (3KB elf + 100B golden) ≈ **80KB**, well under D-15's 50MB cap. This avoids the vendor firmware include dependency entirely and matches the existing P4/P5 fixture pattern that is already wheel-friendly.

## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01..D-18, do NOT reopen)

**`_verify` module surface (D-01..D-03):**

- **D-01:** Hybrid base — `riscv/gtx/_verify.py` absorbs P4 `tests/gtx/_verify_minimal.compare_hex` 78-LOC core (already strict-mode validated, BE bit-pair compare per `vendor/.../verify.py:235`) + wraps vendor `verify.py` 388-LOC argparse/main/report printing (~80 LOC).
- **D-02:** Both CLI entries — `pyproject.toml [project.scripts] pyspike-verify = riscv.gtx._verify:main` AND `python -m riscv.gtx._verify` (via `__main__.py` or `if __name__ == "__main__"`).
- **D-03:** Vendor argparse 1:1 + new `--strict` — `result.hex golden.hex [--ulp N] [--atol F] [--fp16] [--strict]`. Default `--ulp 1 --atol 0.001`. `--strict` ON ⇒ PASS iff `exact_matches == total_fp16`.

**GTX_DDR_DUMP atexit hook (D-04..D-06):**

- **D-04:** atexit registration at `riscv/gtx/__init__.py` import time, `os.getenv('GTX_DDR_DUMP')`-conditional. Code shape: `import os, atexit; from .ddr import _atexit_ddr_dump; if os.getenv('GTX_DDR_DUMP'): atexit.register(_atexit_ddr_dump)`.
- **D-05:** `_atexit_ddr_dump()` body in `riscv/gtx/ddr.py` wrapping P3 D-09's args-only `ddr_dump_to_file(mem, filename, addr, size)`. atexit wrapper parses env vars → args.
- **D-06:** Vendor 1:1 env vars — `GTX_DDR_DUMP` (file path), `GTX_DDR_DUMP_ADDR` (hex int), `GTX_DDR_DUMP_SIZE` (hex int). All three required.

**Regression scope (D-07..D-09):**

- **D-07:** Core op set ~10-20 .elf. Plan-stage selects exact list.
- **D-08:** Vendor `n1s16_<op>.c` 1:1 single-build, mixed dispatch path (GSPR-staged + per-op funct7), encoding-split deferred.
- **D-09:** Single parametrize roll: `tests/gtx/test_regression_fw_full.py` with `@pytest.mark.parametrize('elf_path', sorted(BUNDLED_ELFS), ids=lambda p: p.stem)`.

**Golden source + format + build policy (D-10..D-12):**

- **D-10:** Golden source = `vendor/gtx_cpp_reference/test/<OP>/n1s16/data/{kernel}_ref.txt` direct loan.
- **D-11:** P4/P5 `.hex` format kept (32-byte/line + `@<addr>` markers). Conversion script lock-ins assets to git.
- **D-12:** Dev-only vendor build (no CI vendor build). Conversion is one-shot at P6 dev-stage.

**Wheel asset (D-13..D-15):**

- **D-13:** Build-time copy from `tests/gtx/data/{elf,golden}/` → `src/main/python/riscv/gtx/data/{firmware,golden}/`. Source-of-truth = `tests/gtx/data/`.
- **D-14:** `_verify` helper API — `bundled_elfs() -> list[Path]`, `load_golden(name: str) -> bytes`. `importlib.resources.files()` is internal.
- **D-15:** 50MB cap, bundle-first sizing. Plan-stage measures total; if > 1MB use gzip; if > 10MB consider extras split.

**Plan layout (D-16..D-18):**

- **D-16:** 5 plans / 3 waves.
  - Wave 1a (parallel, 3 plans): Plan 01 (VRF-01 _verify), Plan 02 (atexit hook), Plan 03 (VRF-03 vendor assets).
  - Wave 1b (1 plan): Plan 04 (VRF-04 regression matrix).
  - Wave 2 (1 plan): Plan 05 (PKG-01/03/04 packaging).
- **D-17:** Wave 1a uniform RED scaffold + GREEN-fill (P4/P5 lineage). All P6 RED tests scaffolded in pre-Wave 1a or first plan; subsequent plans GREEN-fill.
- **D-18:** Wave 1a 3 plans parallel (zero edit-area overlap). Plan 01 = `riscv/gtx/_verify.py` only. Plan 02 = `riscv/gtx/ddr.py` + `riscv/gtx/__init__.py`. Plan 03 = `scripts/` + `tests/gtx/data/golden/` + `tests/gtx/data/elf/` + `tests/gtx/data/elf/Makefile`. Common `pyproject.toml` edits cross waves only (Plan 01 `[project.scripts]` ⇔ Plan 05 `[tool.setuptools.package-data]`).

### Claude's Discretion

The CONTEXT explicitly grants Claude the following implementation-detail authority (research/plan must lock):

- vendor `_ref.txt` exact format → **RESOLVED in this RESEARCH (32-byte/line, `@<addr>` directive)** — section "Vendor `_ref.txt` Format".
- `setup.py` custom `build_py` vs `MANIFEST.in include` vs `[tool.setuptools.package-data]` standalone → **RESOLVED**, see "Wheel Build-Time Asset Copy".
- atexit hook NPU instance lookup mechanism (Option A WeakValueDictionary / Option B PythonBridge / Option C global) → **RESOLVED** (Option C, single-global) in "NPU Instance Lookup".
- `_verify.compare_hex` stats dict keys mapped to vendor verbose report → **RESOLVED** in "Stats Dict Mapping".
- core op set exact list (~10-20) → **CANDIDATES SHORTLISTED** in "Vendor Core Op Set Survey".
- Plan 03's .elf build policy → **RESOLVED**: hand-written .S + commit pre-built .elf, NOT vendor n1s16_<op>.c (see "Vendor .elf Build Cost").
- Plan 05 cibuildwheel test matrix verification → **RESOLVED** in "cibuildwheel Test Matrix Strategy".

### Deferred Ideas (OUT OF SCOPE — do not research)

- Numba @njit (Phase 7).
- Vendor 98 op full sweep (v1.x patch / v2).
- gem5-simplified vs ISS-full encoding split (deferred to v1.x patch).
- Vendor C++ libgtx_npu.so CI shadow run (PROJECT.md OOS).
- Python idiomatic `_verify` redesign (`--mode strict|tolerant`) (D-03 lock).
- Vendor `_ref.txt` format wheel-side support (D-11 lock).
- Online shadow run vs C++ libgtx_npu.so (REQUIREMENTS.md OOS).
- PCIe-EP / vfio-user / CUDA / OMP (PROJECT.md OOS).

## Project Constraints (from CLAUDE.md)

- **Python**: 3.10+ (cp310-cp312 cibuildwheel matrix).
- **NumPy**: ≥ 2.0, < 3.
- **No new C++**: Pure Python re-write rule. v2 considers cython/numba.
- **No new runtime deps**: NumPy is the only runtime dependency. Wheel must remain self-contained for `pip install spike`.
- **Bit-exact**: ULP/atol tolerance only at `_verify` boundary; strict mode (`exact_matches == total_fp16`) is the ship gate per ROADMAP P6 success #2.
- **Testing**: pytest-based. Each new op needs a verify_ref.py-derived unit test + at least one .elf regression block. P6 inherits 264/266 tests passing baseline.
- **Platform**: manylinux2014_x86_64. cibuildwheel matrix unchanged from P1 D-08.
- **`proc.state` not `proc.get_state()`**: P4 PHASE-CRITICAL fix (`src/main/cpp/py_module.cc:711` exposes `state` as `def_property_readonly`). Any new spike-bound code in P6 (atexit hook lookup, _verify, regression test) must respect this.
- **GSD workflow**: All file edits route through `/gsd:execute-phase` flow.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **VRF-01** | `verify.py` (388 LOC, FP16 ULP/atol diff) → `riscv.gtx._verify` ported + module importable | "Vendor verify.py API Surface" — exact argparse signature, compare_exact/compare_fp16 stats keys, mapping to existing `_verify_minimal` core. Plan 01 Wave 1a body. |
| **VRF-03** | `tests/gtx/data/{golden,elf}/` regression .elf + golden DDR hex assets bundled | "Vendor `_ref.txt` Format" + "Vendor Core Op Set Survey" + "Vendor .elf Build Cost" — gives the planner exact source paths, conversion semantics, and build strategy. Plan 03 Wave 1a body. |
| **VRF-04** | Regression .elf 100% strict-mode pass | "Validation Architecture" + "Stats Dict Mapping" + "5-tier graceful-skip → hard PASS transition". Plan 04 Wave 1b body. |
| **PKG-01** | `[tool.setuptools.package-data]` includes `riscv.gtx.data/` | "Wheel Build-Time Asset Copy" — recommends `MANIFEST.in include` + `[tool.setuptools.package-data]` declarative path (no custom build_py). Plan 05 Wave 2 body. |
| **PKG-03** | `pip install spike` clean cp310 venv → `from riscv.gtx import GtxNpu` works | "importlib.resources Stability" + "Helper API Implementation" — concrete `bundled_elfs()` / `load_golden()` snippets working uniformly across cp310-cp312. Plan 05 Wave 2 body. |
| **PKG-04** | cibuildwheel cp310-cp312 manylinux2014_x86_64 matrix green | "cibuildwheel Test Matrix Strategy" — test-command recipe, smoke import check, console_script smoke. Plan 05 Wave 2 body. |

## Standard Stack

### Core (verified currency 2026-05-07)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.10–3.12 | Runtime, cibuildwheel matrix | P1 D-08 lock; project-wide invariant |
| NumPy | ≥ 2.0, < 3 | FP16 view + bit-pair parsing | P1 D-07 lock; existing `_verify_minimal.py` already uses `np.frombuffer + dtype=np.float16` for ULP decode |
| setuptools | ≥ 75 | Build backend, `[tool.setuptools.package-data]` glob support, `[project.scripts]` console_script | pyproject.toml requires-build current; modern declarative config replaces ad-hoc setup.py edits |
| setuptools_scm | ≥ 9 | Dynamic version (already wired) | No changes; P6 keeps existing `[tool.setuptools_scm]` |
| cibuildwheel | (pre-existing config in pyproject.toml) | Linux x86_64 manylinux2014 wheel matrix | P1 D-08 already-locked baseline |
| pytest | (existing) | Test runner | Existing test infra |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `argparse` (stdlib) | 3.10+ | Vendor verify.py 1:1 CLI | D-03 mandates 1:1 vendor port; argparse is the obvious fit |
| `importlib.resources` (stdlib) | 3.10+ | Wheel-bundled asset access (`files()` API) | D-14 helper API hides this; `files()` stable across 3.10/3.11/3.12 |
| `atexit` (stdlib) | 3.10+ | DDR dump on `SystemExit(0)` | D-04 lock; CPython behavior verified live |
| `os.environ` (stdlib) | 3.10+ | `GTX_DDR_DUMP*` env var read | Vendor 1:1 |
| `subprocess` (stdlib) | 3.10+ | Regression test harness (existing P4/P5 pattern) | `tests/gtx/test_regression_fw_act.py:121` already uses |

### Alternatives Considered (rejected)

| Instead of | Could Use | Why Rejected |
|------------|-----------|--------------|
| `[tool.setuptools.package-data]` glob | custom `build_py.build_package_data()` hook | Setup.py already has `build_py` for spike build (`setup.py:114-118`); adding asset-copy logic complicates an already-fragile cmdclass and is not needed because `MANIFEST.in include` + package-data globs cover the case (D-13 alternative path). |
| Vendor `n1s16_<op>.c` build for P6 | Hand-written .S per op (P4/P5 lineage) | Vendor sources require `gtx-firmware/include/{intrin.h, gtx/address.h, gtx_csr.h}` + linker script + `gtx-firmware/src/gtx/intrinsics/*.c` — far beyond `/opt/riscv` toolchain. Hand-written .S is what `mm_basic.S`, `activation_relu_gelu.S`, `nop_wjoin.S` already do — same pattern extends naturally. |
| `importlib_resources` backport | stdlib `importlib.resources` | Project requires-python = ">=3.10" (P1 D-08); stdlib is mature and `files()` works directly. No backport needed. |
| Console_script entry as `tool.setuptools.dynamic.entry-points` | Direct `[project.scripts]` table | Static `[project.scripts]` is simpler and explicit. Dynamic only needed if entry points come from runtime config — P6 doesn't. |
| Custom build_py to copy `tests/gtx/data/` files | `MANIFEST.in include` + `tool.setuptools.package-data` glob | setuptools 75+ supports declarative-only path; less code; cibuildwheel respects standard packaging hooks. |

**Installation (already current — no new packages):**
```bash
# All P6 deps are stdlib + pre-existing project deps. No new package install.
pip install -e .  # installs current pyspike + GTX
```

**Version verification:**
```bash
python3 -m setuptools --version       # >=75 expected
python3 -c "import importlib.resources as r; print(hasattr(r, 'files'))"  # True for 3.10+
```

**Confidence:** HIGH (all components validated against current Python 3.10 docs + live experiments; no new dependencies to verify).

## Architecture Patterns

### Recommended Project Structure (P6 deltas only)

```
src/main/python/riscv/gtx/
├── __init__.py            # +D-04: atexit registration block (Plan 02)
├── _verify.py             # NEW (Plan 01): hybrid compare_hex + argparse + main() + helpers
├── _verify/__main__.py    # OPTIONAL — `python -m riscv.gtx._verify` already works via if __name__ == "__main__" in _verify.py
├── ddr.py                 # +D-05: _atexit_ddr_dump() wrapper (Plan 02)
├── data/                  # NEW (Plan 05 build-time-copy target)
│   ├── firmware/          # mm_basic.elf, activation_relu_gelu.elf, n1s16_<op>.elf × ~10
│   └── golden/            # mm_basic_n1s16.hex, activation_relu_gelu.hex, <op>.hex × ~10
└── (other unchanged)

tests/gtx/
├── data/                  # SOURCE OF TRUTH (Plan 03 GREEN-fills)
│   ├── elf/Makefile       # +rules for ~10 new <op>.elf
│   ├── elf/<op>.S × ~10   # NEW hand-written sources (Plan 03)
│   ├── elf/<op>.elf × ~10 # NEW pre-built (Plan 03)
│   └── golden/<op>.hex × ~10  # NEW from vendor _ref.txt (Plan 03)
└── test_regression_fw_full.py  # NEW Plan 04 parametrize wrapper

scripts/
└── import_vendor_golden.py  # NEW Plan 03 — one-shot vendor _ref.txt → golden/<op>.hex transform

pyproject.toml             # Plan 01: [project.scripts] new; Plan 05: [tool.setuptools.package-data] update
MANIFEST.in                # Plan 05: include tests/gtx/data/elf/*.elf, tests/gtx/data/golden/*.hex
```

### Pattern 1: Hybrid `_verify.py` — promote tested mini-port + wrap vendor argparse

**What:** Take the proven 78-LOC `tests/gtx/_verify_minimal._parse_hex + compare_hex` core (D-01) and wrap it with the vendor `verify.py` argparse+main+report-printing surface (~80 LOC).

**When to use:** Whenever the project has a mini-port that already passed strict validation (P4 mm_basic.elf + P5 activation_relu_gelu.elf both strict-pass via `compare_hex(strict=True)`) and a separate "production" wrapper is needed without re-validating the core.

**Example (Plan 01 GREEN-fill skeleton):**
```python
# src/main/python/riscv/gtx/_verify.py
"""Production verify CLI — promotes tests/gtx/_verify_minimal.compare_hex
to riscv.gtx._verify with vendor argparse 1:1 + --strict + helpers.

Source: vendor/gtx_cpp_reference/gtx/verify.py (D-01 hybrid base)
Core:   tests/gtx/_verify_minimal.compare_hex (P4 78 LOC, strict-validated)
"""
from __future__ import annotations
import argparse
import importlib.resources as r
import math
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


# ============================================================================
# Core (absorbed from tests/gtx/_verify_minimal.compare_hex - D-01)
# ============================================================================
def _parse_hex(path: str) -> bytes:
    out = bytearray()
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            clean = ''.join(line.split())
            out.extend(bytes.fromhex(clean))
    return bytes(out)


def compare_hex(actual_path: str, golden_path: str, *,
                ulp: int = 1, atol: float = 0.001,
                strict: bool = True) -> Tuple[bool, dict]:
    """Compare two FP16 hex dumps. BE bit-pair per verify.py:235.

    Returns (passed: bool, stats: dict).
    Strict mode (D-14): passed iff exact_matches == total_fp16.
    Non-strict: passed iff failures == 0 (within_tolerance allowed).
    """
    a_bytes = _parse_hex(actual_path)
    g_bytes = _parse_hex(golden_path)
    n = min(len(a_bytes), len(g_bytes)) // 2
    exact = within = failures = 0
    first_failure = None

    for i in range(n):
        r_raw = (a_bytes[i*2] << 8) | a_bytes[i*2 + 1]    # BE per verify.py:235
        g_raw = (g_bytes[i*2] << 8) | g_bytes[i*2 + 1]
        if r_raw == g_raw:
            exact += 1
            continue
        r_arr = np.frombuffer(np.uint16(r_raw).tobytes(), dtype=np.float16)
        g_arr = np.frombuffer(np.uint16(g_raw).tobytes(), dtype=np.float16)
        r_val, g_val = float(r_arr[0]), float(g_arr[0])
        if np.isnan(r_val) or np.isnan(g_val):
            ulp_dist, abs_diff = 0xFFFF, float('inf')
        else:
            r_sm = r_raw if (r_raw & 0x8000) == 0 else -(r_raw & 0x7FFF)
            g_sm = g_raw if (g_raw & 0x8000) == 0 else -(g_raw & 0x7FFF)
            ulp_dist = abs(r_sm - g_sm)
            abs_diff = abs(r_val - g_val)
        if ulp_dist <= ulp or abs_diff <= atol:
            within += 1
        else:
            failures += 1
            if first_failure is None:
                first_failure = (i, r_raw, g_raw)

    stats = dict(exact_matches=exact, within_tolerance=within,
                 failures=failures, total_fp16=n,
                 first_failure=first_failure,
                 # vendor-compatibility aliases for verbose report (see Stats Dict Mapping)
                 mismatches=failures,
                 first_mismatch=(first_failure[0] * 2) if first_failure else None,
                 size_result=len(a_bytes), size_golden=len(g_bytes),
                 total_bytes=min(len(a_bytes), len(g_bytes)))
    if strict:
        return (exact == n, stats)
    return (failures == 0, stats)


# ============================================================================
# Helpers (D-14: bundled_elfs / load_golden)
# ============================================================================
def bundled_elfs() -> list[Path]:
    """Return list of .elf paths bundled in the wheel (riscv.gtx.data.firmware).

    Hides importlib.resources from end users (D-14).
    """
    fw_dir = r.files('riscv.gtx').joinpath('data', 'firmware')
    # files() returns a Traversable; iterdir() works on cp310-cp312 (verified).
    # Convert to concrete Path using as_file() context-manager-friendly fallback;
    # for files actually on disk (typical wheel install case) the Traversable IS a Path.
    return sorted(p for p in fw_dir.iterdir() if str(p).endswith('.elf'))


def load_golden(name: str) -> bytes:
    """Load golden hex file by op name (without .hex suffix). D-14."""
    g_dir = r.files('riscv.gtx').joinpath('data', 'golden')
    target = g_dir.joinpath(f"{name}.hex")
    return target.read_bytes()


# ============================================================================
# Vendor argparse 1:1 + --strict (D-03)
# ============================================================================
def _print_report_fp16(stats: dict, result_file: str, golden_file: str,
                      ulp: int, atol: float, strict: bool) -> bool:
    """Direct port of vendor verify.py:312-343 (FP16 verbose report).

    Adds strict-mode line. Returns passed bool.
    """
    print("=" * 60)
    print("DDR Hex Verification Report (FP16)")
    print("=" * 60)
    print(f"  Result file : {result_file} ({stats['size_result']} bytes)")
    print(f"  Golden file : {golden_file} ({stats['size_golden']} bytes)")
    print(f"  ULP tolerance  : {ulp}")
    print(f"  Abs tolerance  : {atol}")
    print(f"  Strict mode    : {strict}")
    print(f"  FP16 elements  : {stats['total_fp16']}")
    print(f"  Exact matches  : {stats['exact_matches']}")
    print(f"  Within tolerance: {stats['within_tolerance']}")
    print(f"  Mismatches     : {stats['failures']}")
    if stats['size_result'] != stats['size_golden']:
        print(f"  WARNING: size mismatch (result={stats['size_result']}, golden={stats['size_golden']})")
    if stats['first_failure'] is not None:
        idx, r_raw, g_raw = stats['first_failure']
        print(f"  First mismatch at FP16 idx {idx}: result=0x{r_raw:04x} golden=0x{g_raw:04x}")
    print("-" * 60)
    if strict:
        passed = stats['exact_matches'] == stats['total_fp16']
    else:
        passed = stats['failures'] == 0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    return passed


def main(argv: Optional[list[str]] = None) -> int:
    """Vendor verify.py:350-388 1:1 + --strict (D-03)."""
    parser = argparse.ArgumentParser(
        description='DDR hex diff tool with FP16 rounding tolerance',
        epilog='Example: pyspike-verify result.hex golden.hex --fp16 --strict --ulp 1 --atol 0.001'
    )
    parser.add_argument('result', help='Result hex dump file')
    parser.add_argument('golden', help='Golden reference hex dump file')
    parser.add_argument('--ulp', type=int, default=1,
                        help='FP16 ULP tolerance (default: 1)')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='Absolute tolerance (default: 0.001)')
    parser.add_argument('--fp16', action='store_true',
                        help='Interpret data as FP16 pairs and compare with tolerance')
    parser.add_argument('--strict', action='store_true',
                        help='Strict mode: PASS iff exact_matches == total_fp16')
    args = parser.parse_args(argv)

    passed, stats = compare_hex(args.result, args.golden,
                                 ulp=args.ulp, atol=args.atol,
                                 strict=args.strict)
    if args.fp16 or args.strict:
        _print_report_fp16(stats, args.result, args.golden,
                           args.ulp, args.atol, args.strict)
    else:
        # Non-fp16, non-strict path: minimal exact byte compare report
        print(f"  bytes={stats['total_bytes']} exact={stats['exact_matches']} "
              f"failures={stats['failures']} -> "
              f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
```

**Source mapping (vendor → port):**
- vendor `verify.py:165-185` `parse_hex_file()` → `_parse_hex()` already absorbed (P4)
- vendor `verify.py:217-284` `compare_fp16()` → `compare_hex()` already absorbed (P4 BE bit-pair logic)
- vendor `verify.py:312-343` `print_report_fp16()` → `_print_report_fp16()` (P6 NEW, ~30 LOC)
- vendor `verify.py:350-388` `main()` → `main()` (P6 NEW, ~30 LOC, +`--strict` flag)
- vendor `verify.py:291-309` `print_report_exact()` → optional shim (~10 LOC) or skip (D-03 vendor 1:1 implies fp16 default for our use case)

**Confidence: HIGH** — vendor verify.py is read end-to-end; mapping is line-for-line.

### Pattern 2: atexit hook with single-global NPU lookup (D-04, D-05)

**What:** A module-level singleton `_LAST_NPU` set in `GtxNpu.__init__`, consulted by `_atexit_ddr_dump` when GTX_DDR_DUMP is set.

**When to use:** When Python's atexit needs access to runtime state created during normal program flow.

**Why Option C (single-global) over Option A (WeakValueDictionary) or Option B (PythonBridge):**
- vendor C++ pattern at `gtx_npu_core.cc:59` is **literally** `static gtx_npu_t *g_gtx_instance = nullptr;` + `g_gtx_instance = this;` in constructor. Direct port.
- v1 spike target is single-hart (`GTX_NUM_NESTS=4` per-hart, but spike runs one hart by default with `--extension=gtx_npu`). Multi-hart NPU instantiation is not in v1 scope.
- WeakValueDictionary adds complexity: when NPU is registered via `@register_extension('gtx', GtxNpu)` factory, multiple instances may be created at register-time vs. instantiate-time, and weakref tracking must distinguish "live" from "registered". Single global is C++ 1:1.
- PythonBridge access is C++ side; while `src/main/cpp/py_bridge.h:78` has a `references` map, it's not currently exposed to Python and adding the binding adds C++ surface area (CLAUDE.md "no C++ additions" violation).

**Example (Plan 02 GREEN-fill skeletons):**

```python
# src/main/python/riscv/gtx/npu.py — ADD a module-level + __init__ single-line
_LAST_NPU: 'GtxNpu | None' = None  # NEW: vendor gtx_npu_core.cc:59 direct port

@isa.register("gtx")
class GtxNpu(isa.ROCC):
    def __init__(self):
        global _LAST_NPU                                    # NEW
        super().__init__()
        # ... existing init body (unchanged) ...
        _LAST_NPU = self                                    # NEW (last line of __init__)
```

```python
# src/main/python/riscv/gtx/ddr.py — ADD _atexit_ddr_dump function
def _atexit_ddr_dump() -> None:
    """atexit handler: vendor gtx_npu_core.cc:61-73 1:1 port (D-05).

    Triggered at Python interpreter shutdown when GTX_DDR_DUMP env var is set
    (registration is gated in __init__.py D-04). Reads ADDR/SIZE from env vars
    and writes DDR slice to file via existing args-only ddr_dump_to_file.
    """
    from .npu import _LAST_NPU
    if _LAST_NPU is None:
        return  # No NPU was instantiated — nothing to dump
    if _LAST_NPU.mem is None or _LAST_NPU.mem._ddr_bytes is None:
        return  # Mirror C++ has_ddr() check
    # P3 D-05: flush before dumping (mirrors C++ flush_deferred_ddr_stores)
    _LAST_NPU.flush_deferred_ddr_stores()

    dump_file = os.environ.get('GTX_DDR_DUMP')
    if not dump_file:
        return  # Defensive (registration is already gated, but env may have changed)

    # Vendor gtx_npu_core.cc:68-71: hex parse with stoull base=16; default 0x37f000000 / 0x400
    addr_s = os.environ.get('GTX_DDR_DUMP_ADDR')
    size_s = os.environ.get('GTX_DDR_DUMP_SIZE')
    addr = int(addr_s, 16) if addr_s else 0x37f000000
    size = int(size_s, 16) if size_s else 0x400

    ddr_dump_to_file(_LAST_NPU.mem, dump_file, addr, size)
```

```python
# src/main/python/riscv/gtx/__init__.py — ADD atexit registration block at module-import-time
import atexit
import os

# ... existing imports unchanged ...

# D-04: register atexit dump handler when GTX_DDR_DUMP is set at import time.
# Vendor gtx_npu_core.cc:127 std::atexit(gtx_atexit_ddr_dump) direct port.
# Conditional gate avoids registering a no-op handler in non-dump runs.
if os.getenv('GTX_DDR_DUMP'):
    from .ddr import _atexit_ddr_dump
    atexit.register(_atexit_ddr_dump)
```

**Source:** Direct port of `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc:59-73, 125-127`.

**Caveat (CONFIDENCE LIMIT):** Option C assumes single-NPU-per-process. If a test creates multiple `GtxNpu()` instances (e.g., parametrized regression test), `_LAST_NPU` only points at the last one. For P6 test_regression_fw_full.py this is fine — each parametrize iteration is a separate `subprocess.run` (P5 lineage), and inside each subprocess only one NPU is instantiated. Document this in the docstring.

**Confidence: HIGH** — direct C++ 1:1, verified vendor source.

### Pattern 3: `[project.scripts]` console_script + `if __name__ == "__main__"` dual entry (D-02)

**What:** Both `pyspike-verify` shell command (via `[project.scripts]`) and `python -m riscv.gtx._verify` module invocation work.

**Example pyproject.toml addition (Plan 01 edit):**

```toml
# Add to [project] block — pyproject.toml line ~75 area
[project.scripts]
pyspike-verify = "riscv.gtx._verify:main"
```

**`python -m riscv.gtx._verify` mechanism:** The `if __name__ == "__main__": sys.exit(main())` line at the bottom of `_verify.py` enables this without any additional `__main__.py` file (Plan 01 sketch above already includes this block). Both invocations route to the same `main()`.

**Compatibility with dynamic version:** Verified — `[project] dynamic = ["version"]` and `[project.scripts]` are independent. setuptools 75+ allows static `[project.scripts]` even when `version` is dynamic via `[tool.setuptools_scm]`. Source: setuptools docs `pyproject_config.html`.

**Compatibility with `[tool.setuptools.packages.find] where = ["src/main/python"]`:** Verified — `[project.scripts]` references the entry point by import path (`riscv.gtx._verify:main`), which is resolved against the discovered packages. setuptools finds `riscv.gtx._verify` at `src/main/python/riscv/gtx/_verify.py` automatically.

**Confidence: HIGH** — standard PEP 621 syntax; setuptools 75+ supports static.

### Pattern 4: Wheel build-time asset copy via MANIFEST.in + package-data (D-13)

**What:** Use declarative MANIFEST.in `include` patterns + `[tool.setuptools.package-data]` glob to bundle `tests/gtx/data/{elf,golden}/` content as `riscv/gtx/data/{firmware,golden}/`.

**Why this beats custom `build_py.build_package_data()`:**
- setuptools 75+ supports `[tool.setuptools.package-data]` globbing into subdirectories of the package directory.
- Custom build_py is fragile in cibuildwheel (one report on the pypa/cibuildwheel discussions board #2065 documents that custom cmdclass works "but with care"; the existing `build_py` already calls `_build_spike` and we don't want to entangle asset copy with the C extension build).
- The trick: we DON'T need `tests/gtx/data/` to be the wheel-side path. We can place the canonical files **directly in `src/main/python/riscv/gtx/data/`** and symlink them from `tests/gtx/data/` (or vice versa). But CONTEXT D-13 explicitly says "tests/gtx/data/ single source-of-truth, copied to src/ at build time" — meaning tests stay authoritative, and src/ is wheel-only sync.
- **Recommended approach (matches D-13 verbatim while staying declarative):** Use a thin custom `build_py.build_package_data()` hook that performs **shutil.copytree(tests/gtx/data, build_lib/riscv/gtx/data)** before invoking `super().build_package_data()`. Existing setup.py already has this hook for `_build_spike` — we extend the same hook with one more line.

**Concrete addition to setup.py (Plan 05 edit):**

```python
# setup.py — extend existing build_py hook (lines 114-118)
class build_py(_build_py):

    def build_package_data(self):
        package_dir = pathlib.Path(self.get_package_dir(package.__name__)).absolute()
        _build_spike(package_dir)

        # NEW (P6 D-13): copy tests/gtx/data/ to src/main/python/riscv/gtx/data/
        # build_lib at build time, then setuptools' default package-data globbing
        # picks up data/firmware/*.elf + data/golden/*.hex via package-data config.
        repo_root = pathlib.Path(__file__).parent.absolute()
        tests_data = repo_root / "tests" / "gtx" / "data"
        gtx_pkg_data = package_dir / "gtx" / "data"

        if tests_data.exists():
            for sub_src, sub_dst in (("elf", "firmware"), ("golden", "golden")):
                src = tests_data / sub_src
                dst = gtx_pkg_data / sub_dst
                if src.exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    for entry in src.iterdir():
                        if entry.suffix in (".elf", ".hex"):
                            shutil.copy2(entry, dst / entry.name)

        return super().build_package_data()
```

**pyproject.toml addition (Plan 05 edit):**

```toml
# Add to [tool.setuptools.package-data] block
[tool.setuptools.package-data]
riscv = [
  "data/bin/spike",
  "data/include/**/*.h",
  "data/lib/libdisasm.a",
  "data/lib/libfesvr.a",
  "data/lib/libriscv.so",
  "data/lib/pkgconfig/*.pc"
]
"riscv.gtx" = [                         # NEW (P6 D-13)
  "data/firmware/*.elf",
  "data/golden/*.hex"
]
```

**MANIFEST.in addition (Plan 05 edit):**
```
# Append after the existing recursive-include lines
recursive-include tests/gtx/data/elf *.elf *.S Makefile
recursive-include tests/gtx/data/golden *.hex
```

**Confidence: HIGH** — this is a 3-line setup.py extension on an existing hook, plus declarative pyproject.toml/MANIFEST.in entries. cibuildwheel runs `pip wheel .` which respects all three.

### Pattern 5: importlib.resources `files()` API stable across cp310-cp312 (D-14)

**What:** Single `r.files('riscv.gtx').joinpath('data', ...)` codepath works on Python 3.10/3.11/3.12.

**Why stable:** `importlib.resources.files()` was added in Python 3.9 and is the recommended API for 3.10+. The legacy `path()`/`read_bytes()` functions are deprecated in 3.11+ but `files()` remains stable. Python 3.12 introduced an `anchor` parameter rename (still backward-compatible with `package=`).

**Cross-version code:**
```python
import importlib.resources as r
# Works on 3.10, 3.11, 3.12 identically:
fw_dir = r.files('riscv.gtx').joinpath('data', 'firmware')
for elf_path in fw_dir.iterdir():
    if str(elf_path).endswith('.elf'):
        print(elf_path)
```

**For `bytes` content (Plan 01 `load_golden()`):** `Traversable.read_bytes()` is supported since 3.9.

**For `Path` conversion (Plan 01 `bundled_elfs()`):** When a wheel is installed (the typical case), the resource is on the local filesystem and `iterdir()` yields `pathlib.Path` directly. For zip-installed packages (rare in our wheel context), `as_file()` context manager is the safe wrapper. For P6 the wheel install case suffices; document the limitation.

**Confidence: HIGH** — verified via Python 3.10 live (`hasattr(importlib.resources, 'files') == True`) and Python docs.

### Pattern 6: Subprocess regression with `subprocess.run` + 5-tier graceful skip → hard PASS (D-09)

**What:** Reuse P5 `tests/gtx/test_regression_fw_act.py` pattern with `pytest.mark.parametrize` over `bundled_elfs()`. Each test ID is the .elf stem (e.g., `relu`, `gelu`, `mul_mat`). Parent process (pytest) spawns a subprocess (pyspike CLI) that internally registers atexit and dumps DDR. 5-tier graceful-skip is **kept** but tier #5 ("subprocess clean-exits but no dump") is **expected to never trigger** in P6 (atexit hook from D-04/D-05 ensures dump always happens when env vars are set).

**Example skeleton (Plan 04 GREEN-fill):**
```python
# tests/gtx/test_regression_fw_full.py — NEW (Plan 04)
import pathlib, shutil, subprocess, sys, os
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "elf"
GOLDEN_DIR = REPO_ROOT / "tests" / "gtx" / "data" / "golden"
BUNDLED_ELFS = sorted(ELF_DIR.glob("*.elf"))

try:
    from riscv.processor import processor_t  # noqa
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


@pytest.mark.parametrize('elf_path', BUNDLED_ELFS, ids=lambda p: p.stem)
def test_regression_fw_full(elf_path: pathlib.Path, tmp_path):
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv.so not built")
    if shutil.which('pyspike') is None:
        pytest.skip("pyspike CLI not on PATH")

    golden_path = GOLDEN_DIR / f"{elf_path.stem}.hex"
    if not golden_path.exists():
        pytest.skip(f"golden missing: {golden_path}")

    actual_dump = tmp_path / f"{elf_path.stem}_actual.hex"
    env = os.environ.copy()
    env.pop('GTX_NO_EXIT', None)
    env['GTX_DDR_DUMP'] = str(actual_dump)
    env['GTX_DDR_DUMP_ADDR'] = '0xf000000'   # vendor _ref.txt @<addr> directive (typical)
    env['GTX_DDR_DUMP_SIZE'] = '0x20'         # 32 bytes = 16 FP16 (single row)

    result = subprocess.run(
        ['pyspike', '--extlib=riscv.gtx', '--extension=gtx', str(elf_path)],
        env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, (
        f"{elf_path.stem}: pyspike returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # P6 transition: tier #5 ("dump not produced") becomes a HARD FAIL
    # because atexit hook (Plan 02) always fires when GTX_DDR_DUMP is set.
    assert actual_dump.exists(), (
        f"{elf_path.stem}: GTX_DDR_DUMP atexit hook did NOT fire — "
        f"P6 D-04/D-05 broken. Subprocess clean-exited (rc=0) but no dump."
    )

    from riscv.gtx._verify import compare_hex
    passed, stats = compare_hex(str(actual_dump), str(golden_path), strict=True)
    assert passed, (
        f"{elf_path.stem}: strict-mode compare FAILED.\n"
        f"  stats: {stats}\n"
        f"  actual: {actual_dump}\n"
        f"  golden: {golden_path}\n"
    )
    assert stats['within_tolerance'] == 0, \
        f"{elf_path.stem}: ROADMAP P6 success #2 requires zero within_tolerance"
```

**Confidence: HIGH** — pattern proven in P5 `test_regression_fw_act.py`.

### Anti-Patterns to Avoid

- **DO NOT add a `__main__.py` file under `riscv/gtx/_verify/`.** That requires making `_verify` a sub-package directory, complicating imports. Instead use `if __name__ == '__main__': main()` at the bottom of `_verify.py` (D-02 already-locked).
- **DO NOT make `compare_hex` raise on mismatch.** Vendor verify.py returns exit code 1, which our `main()` handles via `sys.exit(0 if passed else 1)`. Library callers (test infra) check `passed` bool, not exceptions. P4/P5 lineage.
- **DO NOT cache `os.getenv('GTX_DDR_DUMP*')` at module-import time.** Vendor C++ reads env vars in the atexit handler body (`gtx_npu_core.cc:66-69`). Tests set env vars per-call (P5 `test_regression_fw_act.py:116-118`). Module-level caching breaks the test fixture pattern.
- **DO NOT rely on `os._exit()` for WJOIN exit.** Project uses `raise SystemExit(0)` (P2 CORE-03 + `ops/control.py:183`) — atexit fires cleanly. `os._exit()` SKIPS atexit (verified live).
- **DO NOT use `r.path(...)` (deprecated in 3.11+) for resource access.** Use `r.files(...).joinpath(...)` exclusively.
- **DO NOT add tests/data/ files to MANIFEST.in `recursive-include vendor *`.** vendor is excluded already (`prune vendor/gtx_cpp_reference`) — accidentally re-including would balloon the sdist.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BE bit-pair FP16 ULP compare | New numerical engine | Existing `_verify_minimal.compare_hex` (D-01 absorbed) | 78 LOC, already strict-validated by P4/P5 — drift risk if reimplemented |
| FP16 → FP32 decode for ULP distance | Custom IEEE 754 implementation | `np.frombuffer(uint16_bytes, dtype=np.float16)` | NumPy 2.x guarantees IEEE 754 binary16 RNE on manylinux2014_x86_64 (P1 D-09 lock + LE host invariant) |
| atexit-style cleanup on SystemExit | Wrap WJOIN handler with `try/finally` | `atexit.register(...)` (D-04) | Python interpreter shutdown is THE place for "run this once at process exit"; try/finally on every WJOIN call would re-fire on multi-WJOIN firmware |
| Vendor `_ref.txt` file format parser | New tokenizer | Existing `_verify_minimal._parse_hex` (D-01 absorbed) | Verified live: parses vendor `n1s16_relu_ref.txt` (1MB+ file) cleanly into 524288 bytes — 100% format compatibility |
| Wheel resource lookup (`Path` to bundled files) | `__file__`-based glob (`pathlib.Path(__file__).parent / 'data' / ...`) | `importlib.resources.files('riscv.gtx').joinpath('data', ...)` | Works for both editable (`pip install -e .`) installs (where `__file__` is in src/main/python) AND wheel installs (where `__file__` is in site-packages). The `importlib.resources` API is the only one guaranteed to work uniformly. |
| Vendor `n1s16_<op>.c` build for v1 P6 | New cross-build infrastructure for vendor sources | Hand-written `.S` per op (P4 mm_basic.S + P5 activation_relu_gelu.S precedent) | Vendor sources include `intrin.h` + `gtx/address.h` + `gtx_csr.h` from `gtx-firmware/include/` — full GTX firmware tree, NOT just `/opt/riscv` toolchain. Would expand the build dependency surface by 100x for v1 zero gain. v1.x patch can revisit. |
| Custom build_py asset-copy logic | Full overhaul of cmdclass | Extend existing `build_py.build_package_data()` (3-line addition) | setup.py:114-118 already has the hook; one extra `shutil.copy2` loop is much less code than a new cmdclass class |
| pyspike CLI smoke test | New test infrastructure | Reuse P5 `subprocess.run([shutil.which('pyspike'), '--extlib=riscv.gtx', ...])` pattern | Already proven in P5 test_regression_fw_act.py |
| Console_script smoke test | New shell wrappers | `subprocess.run([shutil.which('pyspike-verify'), '--help'])` after wheel install | Standard pattern; cibuildwheel test-command runs in a venv where the script IS on PATH after install |

**Key insight:** P6 is a packaging + glue phase. Almost everything is already tested as a mini-port (P4/P5 _verify_minimal, P3 ddr_dump_to_file, existing pyproject.toml cibuildwheel). The risk is in stitching them together correctly, not in writing new logic.

## Vendor verify.py API Surface (HIGH confidence — vendor source verified)

### Argparse signature (vendor verify.py:351-363)

```
positional: result, golden  (both: hex dump file paths)
optional:
  --ulp INT   default=1   FP16 ULP tolerance
  --atol FLOAT default=0.001  Absolute tolerance
  --fp16  store_true       Interpret as BE FP16 pairs (else byte-exact)
```

### `compare_exact(result: bytes, golden: bytes) -> dict` keys (vendor verify.py:192-214)
```
total_bytes, mismatches, within_tolerance (always 0), exact_matches,
first_mismatch (byte offset or None), size_result, size_golden
```

### `compare_fp16(result, golden, ulp_tol, atol) -> dict` keys (vendor verify.py:217-284)
```
total_fp16, total_bytes, mismatches, within_tolerance, exact_matches,
first_mismatch (byte offset or None), mismatch_details (list of up to 10),
trailing_bytes, size_result, size_golden
```

### `print_report_fp16` line-by-line output (vendor verify.py:312-343)

```
============================================================
DDR Hex Verification Report (FP16)
============================================================
  Result file : <path> (<bytes> bytes)
  Golden file : <path> (<bytes> bytes)
  ULP tolerance  : <int>
  Abs tolerance  : <float>
  FP16 elements  : <int>
  Exact matches  : <int>
  Within tolerance: <int>
  Mismatches     : <int>
  [WARNING: size mismatch (...)]                  ← if size_result != size_golden
  [WARNING: <n> trailing byte(s) ignored ...]    ← if trailing_bytes
  [First mismatch at byte offset: 0x<hex>]       ← if first_mismatch
  ------------------------------------------------------------
  [First mismatches (up to 10):]                  ← if mismatch_details
    @0x<offset>: result=0x<raw> (<val>) golden=0x<raw> (<val>) ULP=<dist> abs_diff=<float>
  ------------------------------------------------------------
  Result: PASS                                    ← exit 0
  ============================================================
```

For FAIL: `Result: FAIL` + exit 1.

### Exit codes
- `0`: PASS (mismatches == 0)
- `1`: FAIL (mismatches > 0) OR error (empty file, invalid hex)

### Mapping to existing `_verify_minimal.compare_hex` stats keys

The existing P4 mini-port returns these keys (from `tests/gtx/_verify_minimal.py:67-69`):
```python
dict(exact_matches=exact, within_tolerance=within,
     failures=failures, total_fp16=n,
     first_failure=first_failure)  # tuple (idx, r_raw, g_raw) or None
```

The vendor `compare_fp16()` returns a superset. Plan 01 should **augment** the mini-port stats dict to ALSO include vendor-compatibility keys for the verbose report:

| Vendor key | Mini-port key | Mapping action in Plan 01 hybrid |
|------------|---------------|----------------------------------|
| `total_fp16` | `total_fp16` | identical — keep |
| `exact_matches` | `exact_matches` | identical — keep |
| `within_tolerance` | `within_tolerance` | identical — keep |
| `mismatches` | `failures` | add as alias (mini-port uses `failures`; vendor uses `mismatches`) — **add `mismatches=failures` in stats dict** |
| `first_mismatch` (byte offset) | `first_failure` (tuple) | derived: `first_mismatch = (first_failure[0] * 2) if first_failure else None` |
| `total_bytes` | (not present) | add: `total_bytes = min(len(a_bytes), len(g_bytes))` |
| `size_result` / `size_golden` | (not present) | add: `size_result = len(a_bytes); size_golden = len(g_bytes)` |
| `trailing_bytes` | (not present) | add: `trailing_bytes = total_bytes % 2` |
| `mismatch_details` | (not present) | OPTIONAL — only needed if Plan 01 adds verbose `--verbose` flag; vendor 1:1 keeps it but MVP can skip |

The Plan 01 sketch above already includes these aliases. **Plan-stage check:** Plan 01 acceptance test must verify both the new `_verify.compare_hex` keys AND the existing mini-port keys (used by P4/P5 regression tests).

**Confidence: HIGH.**

## Vendor `_ref.txt` Format (HIGH confidence — verified live)

### Format (verified from RELU/MUL_MAT/SOFT_MAX/ADD vendor data files)

```
@<addr_hex>          # First line: address directive (e.g. "@f000000" or "@1000000")
<64 hex chars>\n     # Data lines: 32 bytes each, BE FP16 bit-pairs
<64 hex chars>\n
... (typically 16384–32768 lines for n1s16 ops)
```

### Identical to existing `.hex` format

- Vendor `_ref.txt`: starts with `@<addr>` then 32-byte hex lines (64 hex chars).
- Existing `tests/gtx/data/golden/mm_basic_n1s16.hex`: 1 line, 64 hex chars (no `@` directive).
- Existing `tests/gtx/data/golden/activation_relu_gelu.hex`: 16 lines, with `#` comment + 64-char hex lines.
- `_verify_minimal._parse_hex` line 15: `if not line or line.startswith('#') or line.startswith('@'): continue` — already handles all three.

### Verification (live experiment 2026-05-07)
```bash
python3 -c "
import sys; sys.path.insert(0, 'tests/gtx')
from _verify_minimal import _parse_hex
b = _parse_hex('vendor/gtx_cpp_reference/test/RELU/n1s16/data/n1s16_relu_ref.txt')
print(len(b), 'bytes')
"
# Output: 524288 bytes (= 16384 lines × 32 bytes — correct)
```

### Implications for D-10/D-11/D-12

- **No format conversion needed.** Plan 03's `import_vendor_golden.py` script collapses to a **truncate + rename + copy** operation: select the first N data lines from `vendor/.../data/<kernel>_ref.txt`, write to `tests/gtx/data/golden/<op>.hex`, optionally insert a `# Source:` comment header.
- **Truncation strategy:** vendor refs are 1MB+ for typical n1s16 ops (1024 rows × 32 bytes = 32KB at minimum). For a 50MB cap with ~20 ops, individual goldens must stay ≤ 100KB. Single-row truncation (32 bytes / 16 FP16 values) matches the P4/P5 precedent (mm_basic_n1s16.hex = 65 bytes, activation_relu_gelu.hex = 887 bytes). The `.elf` source kernels need to be hand-written to dump the same 32-byte region (P4 mm_basic.S already does this with `GTX_DDR_DUMP_ADDR=0x400 SIZE=0x20`).

### Conversion script sketch (Plan 03)

```python
# scripts/import_vendor_golden.py
"""One-shot vendor ref.txt → golden/<op>.hex transform (P6 D-10/D-11/D-12).

Usage:
    python3 scripts/import_vendor_golden.py
        # Reads VENDOR_TO_PYSPIKE_OPS dict; for each entry copies first N lines
        # of vendor _ref.txt to tests/gtx/data/golden/<op>.hex.

Source: vendor/gtx_cpp_reference/test/<OP>/n1s16/data/<kernel>_ref.txt
Dest:   tests/gtx/data/golden/<op>.hex
"""
import pathlib

# Map: (vendor_op_dir, vendor_kernel_name) → (pyspike_op_name, golden_lines)
VENDOR_TO_PYSPIKE_OPS = {
    "RELU":     ("n1s16_relu",      "relu",     1),
    "GELU":     ("n1s16_gelu",      "gelu",     1),
    "TANH":     ("n1s16_tanh",      "tanh",     1),
    # ... (Plan 03 fills based on op set selection)
}

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR_TEST = REPO_ROOT / "vendor" / "gtx_cpp_reference" / "test"
PYSPIKE_GOLDEN = REPO_ROOT / "tests" / "gtx" / "data" / "golden"

for vendor_dir, (kernel_prefix, op_name, n_lines) in VENDOR_TO_PYSPIKE_OPS.items():
    src_ref = VENDOR_TEST / vendor_dir / "n1s16" / "data" / f"{kernel_prefix}_ref.txt"
    dst_hex = PYSPIKE_GOLDEN / f"{op_name}.hex"

    if not src_ref.exists():
        print(f"SKIP: {src_ref} not found")
        continue

    addr_line = None
    data_lines = []
    with open(src_ref) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('@'):
                if addr_line is None:
                    addr_line = line  # keep first @addr only
                continue
            data_lines.append(line)
            if len(data_lines) >= n_lines:
                break

    with open(dst_hex, 'w') as f:
        f.write(f"# Source: vendor/gtx_cpp_reference/test/{vendor_dir}/n1s16/data/{kernel_prefix}_ref.txt\n")
        f.write(f"# Vendor C++ libgtx_npu.so output, locked-in via run_tests_n1s16.sh --update-ref\n")
        if addr_line:
            f.write(f"{addr_line}\n")
        for line in data_lines:
            f.write(f"{line}\n")
    print(f"WROTE: {dst_hex} ({n_lines} lines)")
```

**Confidence: HIGH** — script is a direct read-and-copy with format preservation; tested against vendor RELU file in research.

## NPU Instance Lookup (D-05 plan-stage resolution → Option C)

### Three options enumerated in CONTEXT

**Option A: `weakref.WeakValueDictionary`**

```python
# riscv/gtx/npu.py (header)
import weakref
_NPU_INSTANCES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

class GtxNpu(...):
    def __init__(self):
        ...
        _NPU_INSTANCES[id(self)] = self

# riscv/gtx/ddr.py (atexit body)
def _atexit_ddr_dump():
    for npu in list(_NPU_INSTANCES.values()):
        if npu and npu.mem and npu.mem._ddr_bytes is not None:
            ddr_dump_to_file(...)
```

Pros: handles multi-NPU. Cons: weakref + dict lookup adds ~50 LOC vs. `_LAST_NPU = self`. Multi-NPU is not v1 (single hart).

**Option B: PythonBridge `references` map**

`src/main/cpp/py_bridge.h:78` has `std::unordered_map<std::string, py::object> references` — used to keep Python objects alive across C++/Python. Currently NOT exposed as a Python attribute.

Pros: 1:1 with how vendor C++ tracks instances. Cons: requires new pybind11 binding code (CLAUDE.md "no C++ additions"); adds ~30 LOC C++ + Python bridge. Not feasible in v1.

**Option C: Single global `_LAST_NPU = None`**

```python
# riscv/gtx/npu.py
_LAST_NPU: 'GtxNpu | None' = None

class GtxNpu(...):
    def __init__(self):
        global _LAST_NPU
        super().__init__()
        ...
        _LAST_NPU = self  # last instance wins (vendor C++ pattern)
```

Pros: 3-line change; vendor `gtx_npu_core.cc:59` (`static gtx_npu_t *g_gtx_instance = nullptr; ... g_gtx_instance = this;`) literal direct port. Cons: only the last-created NPU is dump-able (immaterial in v1 single-hart).

### Recommendation: **Option C**

- v1 spike target is single-hart (per `pyspike --extlib=riscv.gtx <fw>.elf` invocation pattern).
- Direct vendor C++ 1:1 port — easiest to verify against ground truth.
- 3-line change vs. 50+ LOC alternative.
- Multi-NPU concern is theoretical for v1; if multi-hart support lands in v2, this can upgrade to Option A without breaking the existing test surface.

### Concrete code (Plan 02 inputs)

**Registration site** — `src/main/python/riscv/gtx/npu.py` line ~64 (last line of `__init__`):

```python
        self._custom1 = build_custom1_table(self)
        # P6 D-04/D-05: single-global NPU pointer for atexit dump hook.
        # Vendor gtx_npu_core.cc:59 ("static gtx_npu_t *g_gtx_instance") direct port.
        global _LAST_NPU
        _LAST_NPU = self
```

(Prepend `_LAST_NPU: 'GtxNpu | None' = None` at module top, before the `@isa.register("gtx")` decorator.)

**Lookup site** — `src/main/python/riscv/gtx/ddr.py` (new function appended at end):

```python
def _atexit_ddr_dump() -> None:
    """atexit handler: vendor gtx_npu_core.cc:61-73 1:1 (D-05).

    Triggered at Python interpreter shutdown when GTX_DDR_DUMP env var is set
    at import-time of riscv.gtx.__init__ (D-04 gating). Single-NPU model:
    looks up _LAST_NPU module global, flushes deferred S-loop stores, then
    writes DDR slice to file via existing args-only ddr_dump_to_file.
    """
    from .npu import _LAST_NPU
    if _LAST_NPU is None or _LAST_NPU.mem is None:
        return
    if _LAST_NPU.mem._ddr_bytes is None:
        return
    _LAST_NPU.flush_deferred_ddr_stores()

    dump_file = os.environ.get('GTX_DDR_DUMP')
    if not dump_file:
        return  # Env var unset between import and exit — no-op

    addr_s = os.environ.get('GTX_DDR_DUMP_ADDR')
    size_s = os.environ.get('GTX_DDR_DUMP_SIZE')
    addr = int(addr_s, 16) if addr_s else 0x37f000000
    size = int(size_s, 16) if size_s else 0x400

    ddr_dump_to_file(_LAST_NPU.mem, dump_file, addr, size)
```

**Confidence: HIGH** — vendor C++ 1:1 + single-line module write.

## Wheel Build-Time Asset Copy (D-13 resolution)

### Three approaches considered

**A. Custom `build_py.build_package_data()` extension (RECOMMENDED)**

Extend the existing `build_py` cmdclass in setup.py (line 114-118). Add a `shutil.copy2` loop that copies `tests/gtx/data/elf/*.elf` and `tests/gtx/data/golden/*.hex` into `<pkg>/data/firmware/` and `<pkg>/data/golden/` at build time. ~10 LOC addition, reuses existing hook.

**B. Pure declarative — `[tool.setuptools.package-data]` glob with relative path**

setuptools 75+ supports `[tool.setuptools.package-data] "riscv.gtx" = ["data/firmware/*.elf"]`, but only for files **already inside** `src/main/python/riscv/gtx/data/`. We'd need to commit the duplicated files there — violates D-13's "tests/gtx/data/ is single source of truth".

**C. `MANIFEST.in include` (sdist only)**

`recursive-include tests/gtx/data ...` ensures sdist contains the files, but for **wheel** we need `package-data` to reference them at the wheel-side path. MANIFEST.in alone doesn't move files between paths.

### Why A (custom build_py extension) wins

- D-13 explicitly says "tests/gtx/data/ single source-of-truth, copied to src/ at build time" — A is the literal implementation.
- Existing setup.py already has `build_py` cmdclass for `_build_spike` — the hook exists.
- cibuildwheel runs `pip wheel .` which respects setuptools cmdclass (verified in pypa/cibuildwheel discussion #2065).
- Path B requires committing duplicate files (drift risk).
- Path C only handles sdist, not wheel.

### Concrete diffs

**setup.py (extend lines 114-118):**
```python
class build_py(_build_py):

    def build_package_data(self):
        package_dir = pathlib.Path(self.get_package_dir(package.__name__)).absolute()
        _build_spike(package_dir)

        # P6 D-13: copy tests/gtx/data/{elf,golden}/ to <pkg>/gtx/data/{firmware,golden}/.
        # tests/ is the single source-of-truth (D-13); src/ is wheel-only sync.
        repo_root = pathlib.Path(__file__).parent.absolute()
        tests_data = repo_root / "tests" / "gtx" / "data"
        gtx_pkg_data = package_dir / "gtx" / "data"
        for src_sub, dst_sub, suffix in (
            ("elf", "firmware", ".elf"),
            ("golden", "golden", ".hex"),
        ):
            src_dir = tests_data / src_sub
            dst_dir = gtx_pkg_data / dst_sub
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for entry in src_dir.iterdir():
                    if entry.suffix == suffix:
                        shutil.copy2(entry, dst_dir / entry.name)

        return super().build_package_data()
```

(Add `import shutil` at top of setup.py.)

**pyproject.toml (extend `[tool.setuptools.package-data]`):**
```toml
[tool.setuptools.package-data]
riscv = [
  "data/bin/spike",
  "data/include/**/*.h",
  "data/lib/libdisasm.a",
  "data/lib/libfesvr.a",
  "data/lib/libriscv.so",
  "data/lib/pkgconfig/*.pc"
]
"riscv.gtx" = [                         # NEW (P6 D-13)
  "data/firmware/*.elf",
  "data/golden/*.hex"
]
```

**MANIFEST.in (append; ensures sdist contains the source assets):**
```
# P6 D-13: tests/gtx/data is single source-of-truth; sdist must carry it.
recursive-include tests/gtx/data/elf *.elf *.S Makefile
recursive-include tests/gtx/data/golden *.hex
```

**Cleanup hook:** Existing `recursive-exclude src/main/python/riscv/data *` (MANIFEST.in line 14) handles purging the auto-generated `data/lib/` from sdist; we should NOT add `recursive-exclude src/main/python/riscv/gtx/data *` because the build_py hook generates that during wheel build. setuptools-managed paths are wheel-only.

**Confidence: HIGH** — setuptools 75+ documented behavior; existing setup.py pattern; cibuildwheel respects PEP 517.

## Vendor Core Op Set Survey (D-07 plan-stage shortlist)

### Constraint

D-07: ~10-20 ops, must cover MM/VEC/ACT/POOL/CVT dispatch path representatives.

### Available vendor ops (verified — both `n1s16_<op>.c` source AND `data/*_ref.txt` exist)

Survey of 22 candidate dirs (all PASS the "has src + has ref" filter):

| Vendor dir | Pyspike op | Category | Comment |
|------------|------------|----------|---------|
| RELU | relu | ACT-fwd | Element-wise; uses CLAMP-min path. Forward direction (ACT-01). |
| GELU | gelu | ACT-rev | Reversed direction (ACT-02). Tests asymmetry. |
| TANH | tanh | ACT-rev | Reversed direction. |
| SIGMOID | sigmoid | ACT-rev | Reversed direction. |
| SOFT_MAX | softmax | ACT-fwd | Forward (ESUM-derived). |
| SOFTPLUS | softplus | ACT-rev | Reversed (math composition). |
| ADD | add_vv | VEC-SASMD | SASMD funct7=0x10. VV variant. |
| SUB | sub_vv | VEC-SASMD | SASMD funct7=0x10. |
| MUL | mul_vv | VEC-SASMD | SASMD funct7=0x10. |
| DIV | div_vv | VEC-SASMD | SASMD funct7=0x10. |
| MUL_MAT | mul_mat | MM | GEMM 1024×1024 × 1024×512. Stresses MM/MMC chain (MM-04). |
| NEG | neg | VEC-unary | Unary, simple. |
| ABS | abs | VEC-unary | Unary, simple. Multiple files in vendor. |
| EXP | exp | ACT-fwd | Forward direction. |
| LOG | log | ACT-fwd | Forward. |
| STEP | step | ACT-fwd | Forward. |
| HARDSIGMOID | hardsigmoid | ACT-rev | Reversed. |
| HARDSWISH | hardswish | ACT-rev | Reversed. |
| LEAKY_RELU | leaky_relu | ACT-fwd | Forward (PRELU variant). |
| SUM | sum | VEC-VSUM | Reduction VSUM funct7=0x1A. Stresses FP32-internal. |
| POOL_2D | pool_2d | POOL | max/avg pool. |
| POOL_1D | pool_1d | POOL | max/avg pool. |
| SQR | sqr | VEC-binary | mul(a,a) — direct sqr is single-op. |

### Recommended P6 core set (~12 ops, plan-stage may adjust)

Balanced coverage across dispatch paths:

| # | Op | Category | Coverage Rationale |
|---|----|----|--------|
| 1 | mm_basic | MM | Already exists from P4 — keep as MM canary. |
| 2 | mul_mat | MM | New MM workload — stresses chain MMC. |
| 3 | activation_relu_gelu | ACT-mixed | Already exists from P5 — keep. |
| 4 | softmax | ACT-fwd (ESUM) | New — ESUM precision dependency. |
| 5 | sigmoid | ACT-rev | Reversed direction — separate from gelu. |
| 6 | tanh | ACT-rev | Independent reversed kernel. |
| 7 | add_vv | VEC-SASMD-VV | SASMD VV funct7=0x10. |
| 8 | mul_vv | VEC-SASMD-VV | SASMD VV (alternative arith). |
| 9 | sum | VEC-VSUM | Reduction precision (P5 D-09 lineage). |
| 10 | abs | VEC-unary | Trivial unary. |
| 11 | pool_2d | POOL | Pooling forward. |
| 12 | leaky_relu | ACT-fwd-PRELU | PRELU coverage. |

### Asset cost estimate

Each op:
- `<op>.S` source: ~60-100 lines (~3KB)
- `<op>.elf` pre-built: ~1.3-2KB
- `<op>.hex` golden (truncated to 1 row = 32 bytes): ~100B + comment header = ~250B

Total: ~12 × (3KB + 2KB + 250B) ≈ ~63KB source + bin assets, plus existing 3 fixtures. Well below 50MB cap (D-15) and even below 1MB plan-stage trigger.

**Confidence: MED** — list is candidate; final selection requires plan-stage author confirmation (vendor sources may have op-specific quirks not surfaced by the simple `has src + has ref` filter). The planner can drop entries that have unusual setup (e.g., MUL_MAT requires command file `n1s16_mul_mat_cmd.txt` for shape config).

## Vendor .elf Build Cost (D-08 plan-stage resolution)

### Hard finding

The vendor `n1s16_<op>.c` sources have non-trivial dependencies (verified by reading `RELU/n1s16/n1s16_relu.c` line 9-11):

```c
#include "intrin.h"
#include "gtx/address.h"
#include "gtx_csr.h"
```

These headers are in `vendor/gtx_cpp_reference/gtx-firmware/include/` (per `run_tests_n1s16.sh` line 41-43: `INCS="-I$SCRIPT_DIR -I$PROJECT_ROOT/c/src/include -I$GFW/include -I$GFW/include/gtx -I$GFW/include/gtx/intrinsics -I$GFW/include/intrinsics"`). Building requires:

1. The full `gtx-firmware/` submodule (currently committed at `vendor/gtx_cpp_reference/gtx-firmware/`).
2. `COMMON_SRCS = $GFW/src/gtx/intrinsics/intrin_level1.c + intrin_level2.c + intrin_level3.c + ...` linked in.
3. Custom linker script `$GFW/linker.ld`.
4. Cross-toolchain `riscv64-unknown-elf-gcc` with `-march=rv64g_xgtxnpu` (custom march extension).

### Implications

**Building vendor `n1s16_<op>.c` for v1 P6 wheel = NOT FEASIBLE in CI** (`-march=rv64g_xgtxnpu` requires a custom-patched binutils that the CI cibuildwheel environment doesn't have — the project uses the standard `/opt/riscv` toolchain for P2-P5 .elf fixtures).

**Recommendation: HAND-WRITTEN .S per op, P4/P5 lineage**

The existing `tests/gtx/data/elf/` already has 3 hand-written .S files (`mm_basic.S`, `activation_relu_gelu.S`, `nop_wjoin.S`) that:
1. Use only `/opt/riscv` toolchain — works with CI matrix.
2. Use the `pyspike` GTX RoCC opcodes directly (custom0/custom1) — bypasses the GTX firmware intrin.h layer entirely.
3. Are ~50-80 lines each — small, reviewable.
4. Pre-built `.elf` files are already committed to git (P2 D-22 / P4 D-09 / P5 D-04 lineage).

**Plan 03 strategy:**
- Author `<op>.S` for each op in the core set, modeled after existing `mm_basic.S` / `activation_relu_gelu.S`.
- Add Makefile rules (one per op, mirroring existing 3 entries).
- Pre-build with `/opt/riscv` toolchain on dev machine.
- Commit `.elf` files to git (D-12: dev-only build).
- Bundle into wheel via D-13 build-time copy.

**vendor `_ref.txt` golden source remains independent of vendor .elf build** — D-10 allows direct loan because the goldens are byte streams, not executable code. The hand-written .S kernels just need to dump the same DDR region pattern that vendor produces.

**v1.x patch deferral note:** A v1.x patch could revisit this by either (a) adopting `gtx-firmware/` as a build dep with `before-all` install in cibuildwheel, or (b) running vendor `run_tests_n1s16.sh` once on dev machine and committing the resulting .elf files alongside the goldens. Both are out of scope for v1 P6 per CONTEXT D-08 + D-12.

**Confidence: HIGH** — verified vendor source dependencies + verified existing P4/P5 hand-written .S precedent.

## Stats Dict Mapping (D-01 plan-stage resolution)

See "Vendor verify.py API Surface — Mapping" table above. Plan 01 acceptance must verify both old keys (mini-port back-compat for P4/P5 regression tests that import `from tests.gtx._verify_minimal import compare_hex`) AND new vendor-compatibility keys (for `_print_report_fp16` correctness).

**Concrete acceptance assertion (Plan 01 GREEN-fill test sketch):**
```python
def test_compare_hex_returns_vendor_compatible_keys(tmp_path):
    a = tmp_path / "a.hex"; a.write_text("00" * 32 + "\n")
    g = tmp_path / "g.hex"; g.write_text("00" * 32 + "\n")
    from riscv.gtx._verify import compare_hex
    passed, stats = compare_hex(str(a), str(g), strict=True)
    assert passed
    # Mini-port keys (back-compat for P4/P5 regressions)
    assert {'exact_matches', 'within_tolerance', 'failures', 'total_fp16',
            'first_failure'} <= set(stats.keys())
    # Vendor verbose-report keys
    assert {'mismatches', 'first_mismatch', 'size_result', 'size_golden',
            'total_bytes'} <= set(stats.keys())
    assert stats['failures'] == stats['mismatches']  # alias
    assert stats['exact_matches'] == 16
    assert stats['total_fp16'] == 16
```

## importlib.resources Stability (D-14)

| Python | `files()` | `joinpath()` | `iterdir()` | `read_bytes()` | `as_file()` |
|--------|-----------|--------------|-------------|----------------|-------------|
| 3.10 | ✅ stable | ✅ stable | ✅ stable | ✅ stable | ✅ stable |
| 3.11 | ✅ stable | ✅ stable | ✅ stable | ✅ stable | ✅ stable |
| 3.12 | ✅ stable (`anchor=` param rename, `package=` still works) | ✅ stable | ✅ stable | ✅ stable | ✅ stable |

**Single codepath for all 3 versions:**
```python
import importlib.resources as r
files_traversable = r.files('riscv.gtx').joinpath('data', 'firmware')
for entry in files_traversable.iterdir():
    if str(entry).endswith('.elf'):
        print(entry)  # entry is pathlib.Path for wheel installs
```

**Verified live:** Python 3.10.12 → `hasattr(importlib.resources, 'files') == True` ✓

**Confidence: HIGH** — Python docs all three versions + live experiment.

## cibuildwheel Test Matrix Strategy (PKG-04)

### Current config (pyproject.toml lines 15-30)

```toml
[tool.cibuildwheel]
build = [
  "cp310-manylinux_x86_64",
  "cp311-manylinux_x86_64",
  "cp312-manylinux_x86_64"
]
manylinux-x86_64-image = "quay.io/pypa/manylinux2014_x86_64"
test-environment = {PYTHONPATH = "examples"}
test-command = "pytest -v -k 'not pyspike_cli'"
test-extras = ["dev"]
test-sources = ["examples", "tests"]

[tool.cibuildwheel.linux]
before-all = "yum install -y dtc && git submodule update --init --recursive"
```

### P6 changes — minimal

**No changes to `build` matrix:** P1 D-08 lock; P6 only validates the matrix passes.

**test-command extension (Plan 05 edit):** Add a smoke test command to verify wheel-bundled assets work and console_script is on PATH:

```toml
test-command = """
pytest -v -k 'not pyspike_cli' && \
python -c "from riscv.gtx import GtxNpu; from riscv.gtx import _verify; assert callable(_verify.compare_hex); assert callable(_verify.bundled_elfs); assert callable(_verify.load_golden)" && \
pyspike-verify --help
"""
```

This adds:
1. Existing pytest run.
2. Python smoke import for the one-liner per ROADMAP P6 success #3.
3. Console_script `pyspike-verify --help` smoke per ROADMAP P6 success #5.

**Note:** The triple-line test-command may need bash continuation handling for cibuildwheel; the safe form is:
```toml
test-command = "pytest -v -k 'not pyspike_cli' && python -c \"from riscv.gtx import GtxNpu; from riscv.gtx import _verify; assert callable(_verify.bundled_elfs)\" && pyspike-verify --help"
```

### Local smoke test (developer-side, plan 05 acceptance)

```bash
# Plan 05 acceptance sketch (developer machine, before cibuildwheel)
python3 -m venv /tmp/test_p6_venv
source /tmp/test_p6_venv/bin/activate
pip install --upgrade pip
pip install dist/spike-*.whl

# Smoke 1: one-liner import works
python3 -c "from riscv.gtx import GtxNpu; print(GtxNpu)"

# Smoke 2: _verify importable
python3 -c "from riscv.gtx import _verify; print(_verify.compare_hex)"

# Smoke 3: helper API works
python3 -c "from riscv.gtx._verify import bundled_elfs; print(len(bundled_elfs()))"

# Smoke 4: console_script on PATH
which pyspike-verify
pyspike-verify --help

# Smoke 5: end-to-end verify CLI run on a bundled elf+golden
python3 -c "
from pathlib import Path
import importlib.resources as r
elf = next(p for p in r.files('riscv.gtx').joinpath('data','firmware').iterdir() if str(p).endswith('mm_basic.elf'))
golden = r.files('riscv.gtx').joinpath('data','golden','mm_basic_n1s16.hex')
print('elf:', elf)
print('golden:', golden)
"
```

**Confidence: MED-HIGH** — cibuildwheel test-command line continuation has minor syntactic gotchas; will validate at plan-stage.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing, pyproject.toml line 87) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` lines 149-163 |
| Quick run command | `pytest -v -k 'not pyspike_cli' tests/gtx/test_*` |
| Full suite command | `pytest -v` (matches cibuildwheel test-command) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File |
|--------|----------|-----------|-------------------|-------|
| VRF-01 | `riscv.gtx._verify.compare_hex(strict=True)` matches mini-port; `--strict` CLI flag works; vendor 1:1 argparse | unit + CLI smoke | `pytest tests/gtx/test_verify_module.py -v` | ❌ Wave 0 (Plan 01 RED scaffold) |
| VRF-01 | `pyspike-verify --help` and `python -m riscv.gtx._verify --help` both work | smoke | `pytest tests/gtx/test_verify_console_script.py -v -k 'not pyspike_cli'` | ❌ Wave 0 |
| (atexit) | `atexit.register(_atexit_ddr_dump)` fires on `SystemExit(0)` and writes the dump file | subprocess+integration | `pytest tests/gtx/test_atexit_ddr_dump.py -v` | ❌ Wave 0 |
| (atexit) | atexit hook is NOT registered when GTX_DDR_DUMP env var is unset | subprocess | (same file) | ❌ Wave 0 |
| VRF-03 | `tests/gtx/data/elf/<op>.elf` and `tests/gtx/data/golden/<op>.hex` exist for core op set | file existence | `pytest tests/gtx/test_assets_present.py -v` | ❌ Wave 0 |
| VRF-03 | Golden hex parses cleanly via `_verify.compare_hex` (self-compare PASS) | unit | (same file or test_verify_module.py) | ❌ Wave 0 |
| VRF-04 | Each bundled `.elf` runs under `pyspike --extlib=riscv.gtx` and produces a strict-mode-PASS dump | parametrized subprocess+e2e | `pytest tests/gtx/test_regression_fw_full.py -v` | ❌ Wave 0 (Plan 04 GREEN-fill) |
| VRF-04 | Existing `test_regression_fw_act.py` 5-tier graceful-skip transitions to hard PASS at tier #5 | regression | `pytest tests/gtx/test_regression_fw_act.py -v` | ✅ Already exists (P5) |
| VRF-04 | Existing `test_regression_fw_mm.py` 4-tier graceful-skip transitions to hard PASS at tier #4 | regression | `pytest tests/gtx/test_regression_fw_mm.py -v` | ✅ Already exists (P4) |
| PKG-01 | Wheel includes `riscv/gtx/data/firmware/*.elf` + `riscv/gtx/data/golden/*.hex` | wheel inspection | (manual: `unzip -l dist/spike-*.whl \| grep -E 'firmware\|golden'`) | ❌ Plan 05 GREEN-fill |
| PKG-01 | `python -c "import importlib.resources as r; assert any(p.name.endswith('.elf') for p in r.files('riscv.gtx').joinpath('data','firmware').iterdir())"` works | smoke | `pytest tests/gtx/test_wheel_data_present.py -v` | ❌ Wave 0 |
| PKG-03 | Clean cp310 venv `pip install dist/spike-*.whl` + `from riscv.gtx import GtxNpu, _verify` works | venv smoke | (manual + cibuildwheel test-command) | ❌ Plan 05 |
| PKG-04 | `pyspike-verify --help` exits 0 in cibuildwheel test env | console_script smoke | (cibuildwheel test-command) | ❌ Plan 05 |
| PKG-04 | cibuildwheel cp310-cp312 matrix builds clean | matrix run | (cibuildwheel local: `python3 -m cibuildwheel --output-dir wheelhouse .`) | n/a — CI only |

### Sampling Rate (Nyquist)

- **Per task commit:** `pytest -v tests/gtx/test_verify_module.py tests/gtx/test_atexit_ddr_dump.py tests/gtx/test_assets_present.py tests/gtx/test_regression_fw_full.py` (~30s, hits all P6 unit/smoke surfaces)
- **Per wave merge:** `pytest -v -k 'not pyspike_cli'` (full suite ~3-5 min on local + matches cibuildwheel)
- **Phase gate:** Full pytest suite green + `cibuildwheel --platform linux` green for cp310-cp312 + `dist/spike-*.whl` size verification.

### Wave 0 Gaps (RED scaffold needed before GREEN-fill)

P6 introduces 4 new test files; ALL need RED scaffold landing in Wave 1a Plan 01 (or pre-Wave Plan 00) per D-17:

- [ ] `tests/gtx/test_verify_module.py` — covers VRF-01 (compare_hex strict/tolerant + helper API + stats dict keys)
- [ ] `tests/gtx/test_verify_console_script.py` — covers VRF-01 CLI surface (`pyspike-verify --help`, `python -m riscv.gtx._verify --help`)
- [ ] `tests/gtx/test_atexit_ddr_dump.py` — covers atexit hook (subprocess fires with env vars set; does NOT fire without)
- [ ] `tests/gtx/test_assets_present.py` — covers VRF-03 (elf+golden file existence + parse round-trip + Makefile rule presence)
- [ ] `tests/gtx/test_regression_fw_full.py` — covers VRF-04 (parametrize over BUNDLED_ELFS, strict-mode pass)
- [ ] `tests/gtx/test_wheel_data_present.py` — covers PKG-01 (importlib.resources.files smoke)

**No new framework install:** pytest, pytest-cov, pytest-pylint already in `[project.optional-dependencies] dev`.

## Common Pitfalls

### Pitfall 1: atexit doesn't fire across subprocess boundary

**What goes wrong:** Setting GTX_DDR_DUMP in the parent pytest process, then `subprocess.run([pyspike, ...])`, then expecting the parent's atexit handler to dump DDR.

**Why it happens:** Python's atexit is per-interpreter. The subprocess has a fresh Python interpreter; the atexit handler registered in the parent does NOT propagate to the child.

**How to avoid:** P6 D-04 puts the atexit registration in `riscv/gtx/__init__.py` — when the child subprocess imports `riscv.gtx` (via `pyspike --extlib=riscv.gtx`), it registers its OWN atexit, scoped to its OWN GtxNpu instance. The env vars (GTX_DDR_DUMP*) ARE inherited by the child because `subprocess.run(..., env=env)` propagates the environment. So registration runs in child, dump runs in child, DDR file written by child — parent reads file after child exits.

**Warning signs:** Test passes locally but fails in CI when conftest fixture sets env vars after import; child fails to find env vars at the right point.

### Pitfall 2: `os._exit()` skips atexit

**What goes wrong:** Some "clean exit" paths use `os._exit(0)` (immediate kernel-level exit, no Python cleanup). Vendor `gtx_npu_custom1.cc:117` comment says "DDR dump moved to atexit handler... once at exit, not per WJOIN" — implying spike's HTIF exit path goes through `exit()` (libc atexit) NOT `_Exit()` (skips atexit).

**Why it happens:** Python's `os._exit()` and C's `_Exit()` both skip atexit/atfork; signals not handled by Python skip atexit; SIGKILL skips everything.

**How to avoid:** P6 ops/control.py:183 uses `raise SystemExit(0)` — verified live: atexit fires, exit code 0. DO NOT change to `os._exit()`. Spike's HTIF path eventually returns from `simif_t::run()` which falls back to normal `main()` return → libc atexit (verified by reading `gtx_npu_core.cc:127` `std::atexit(gtx_atexit_ddr_dump)` — same mechanism).

**Warning signs:** Subprocess exits cleanly (rc=0) but no dump file. Check whether ANY exit path uses `os._exit()` or `_Exit()`.

### Pitfall 3: SystemExit raised INSIDE atexit handler (CPython issue #103512)

**What goes wrong:** If `_atexit_ddr_dump()` body itself does `sys.exit(1)`, Python 3.10+ prints "Exception ignored in atexit callback" + traceback (regression vs 3.6-3.9).

**Why it happens:** PR #23779 removed the SystemExit suppression logic.

**How to avoid:** `_atexit_ddr_dump()` body must NOT raise SystemExit. Use early returns + `print(..., file=sys.stderr)` for errors. The Plan 02 sketch above is safe.

**Warning signs:** Test output contains "Exception ignored in atexit callback" — fix by removing any `sys.exit` from atexit handlers.

### Pitfall 4: Wheel resources not accessible via `__file__` in editable install

**What goes wrong:** `pathlib.Path(__file__).parent / 'data' / 'firmware'` works in editable install (`pip install -e .`) where `__file__` is in the source tree, but breaks in `pip install dist/spike-*.whl` where the package is in site-packages and the data was injected via package-data.

**Why it happens:** `__file__` doesn't necessarily live next to package-data resources after wheel install (depends on whether the wheel was extracted in-place or accessed via zipimport).

**How to avoid:** Use `importlib.resources.files()` exclusively (D-14). The Plan 01 helper API (`bundled_elfs()`, `load_golden()`) hides this from users.

**Warning signs:** `bundled_elfs()` returns empty in fresh venv install but non-empty in editable. Cure: switch from `__file__` to `importlib.resources.files`.

### Pitfall 5: setuptools doesn't auto-include files from package-data outside package dir

**What goes wrong:** Adding `"riscv.gtx" = ["../../tests/gtx/data/elf/*.elf"]` (relative path traversal) — setuptools 75+ rejects this for security (issue #227).

**Why it happens:** Path traversal would allow arbitrary file inclusion.

**How to avoid:** Files MUST be inside the package directory at the time package-data is evaluated. The build_py custom hook (D-13) copies them BEFORE package-data evaluation (`build_package_data()` order: `_build_spike` → asset copy → `super().build_package_data()`). The package-data glob then sees them at the legitimate `<pkg>/data/firmware/*.elf` path.

**Warning signs:** Wheel build succeeds but `unzip -l dist/spike-*.whl | grep firmware` finds no .elf files.

### Pitfall 6: 5-tier graceful-skip transitioning incorrectly to hard PASS

**What goes wrong:** P5 `test_regression_fw_act.py` tier #5 ("subprocess clean-exits but no dump") was a graceful skip. P6 D-04/D-05/D-06 expect this to become hard PASS. If atexit hook silently fails (e.g., `_LAST_NPU = None` because import order broke), test still skips → false-negative passing.

**Why it happens:** D-04 conditional registration (`if os.getenv('GTX_DDR_DUMP')`) gates ON env var presence, not on whether the dump file actually appears. If the registration succeeds but the body fails, the file is missing and tier #5 skips.

**How to avoid:** Plan 02 acceptance MUST include a positive test: `test_atexit_dump_writes_file` that explicitly sets GTX_DDR_DUMP, runs subprocess, asserts `actual_dump.exists()`. The existing P5 test must convert tier #5 to `assert actual_dump.exists(), "P6 D-04 broken"`. The Plan 04 sketch above does this.

**Warning signs:** P5 tests still report "skipped" with the exact same reason after P6 atexit ships.

### Pitfall 7: cibuildwheel custom build_py interaction

**What goes wrong:** Custom `build_py` cmdclass works locally but fails in cibuildwheel due to environment differences (path resolution, file permissions, missing tests/ in sdist).

**Why it happens:** cibuildwheel runs in containers; sdist must contain `tests/gtx/data/`; MANIFEST.in must include them.

**How to avoid:** Verify (Plan 05 acceptance):
1. `python3 setup.py sdist` produces `dist/spike-*.tar.gz` — `tar tzf dist/spike-*.tar.gz | grep tests/gtx/data` shows .elf and .hex files.
2. `python3 -m build --wheel` produces `dist/spike-*.whl` — `unzip -l dist/spike-*.whl | grep -E "(firmware|golden)"` shows the bundled assets.
3. `cibuildwheel --platform linux --only cp310-manylinux_x86_64 .` succeeds.

**Warning signs:** Local wheel build works but cibuildwheel fails with "tests/gtx/data not found" — fix by adding `recursive-include tests/gtx/data ...` to MANIFEST.in.

### Pitfall 8: Vendor _ref.txt golden has @<addr> directive that pollutes byte stream

**What goes wrong:** Naively concatenating vendor lines without skipping `@` directive lines would inject 4 bytes (the parsed `0x0f000000` integer) at the start of the byte stream, causing strict-mode mismatch.

**Why it happens:** `bytes.fromhex(line)` would happily encode "0f000000" as 4 bytes if you don't filter `@` first.

**How to avoid:** Existing `_verify_minimal._parse_hex` (line 15) already filters `@` and `#` lines. Conversion script must preserve this filter when truncating ref files.

**Warning signs:** Golden has unexpected leading 4 bytes that don't match firmware DDR layout. Cure: ensure `@` line is preserved as-is or stripped uniformly across actual+golden.

## Code Examples

### Example 1: minimal `_atexit_ddr_dump` test (Plan 02 Wave 1a RED scaffold)

```python
# tests/gtx/test_atexit_ddr_dump.py — NEW (Plan 02 GREEN-fill target)
"""P6 D-04/D-05/D-06: GTX_DDR_DUMP atexit hook tests.

Verifies that:
  1. atexit hook fires on subprocess SystemExit(0) when GTX_DDR_DUMP is set,
     producing a dump file with correct content.
  2. atexit hook does NOT register when GTX_DDR_DUMP is unset.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ELF_PATH = REPO_ROOT / "tests" / "gtx" / "data" / "elf" / "nop_wjoin.elf"


@pytest.mark.skipif(shutil.which('pyspike') is None, reason="pyspike CLI not on PATH")
def test_atexit_dump_fires_on_systemexit(tmp_path):
    """Subprocess pyspike + GTX_DDR_DUMP env vars → dump file must exist."""
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing")

    actual_dump = tmp_path / "atexit_dump.hex"
    env = os.environ.copy()
    env.pop('GTX_NO_EXIT', None)
    env['GTX_DDR_DUMP'] = str(actual_dump)
    env['GTX_DDR_DUMP_ADDR'] = '0x0'   # nop_wjoin doesn't write DDR; just exercise the dump path
    env['GTX_DDR_DUMP_SIZE'] = '0x20'

    result = subprocess.run(
        ['pyspike', '--extlib=riscv.gtx', '--extension=gtx', str(ELF_PATH)],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )

    assert result.returncode == 0, (
        f"pyspike rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert actual_dump.exists(), (
        f"GTX_DDR_DUMP atexit hook FAILED to write {actual_dump}.\n"
        f"D-04/D-05 broken: subprocess cleanly exited but no dump.\n"
        f"stderr:\n{result.stderr}"
    )
    # Check dump has expected size (32 bytes hex = 64 chars + newline)
    content = actual_dump.read_text()
    data_lines = [l for l in content.splitlines() if l.strip() and not l.startswith(('#', '@'))]
    assert len(data_lines) == 1, f"Expected 1 hex line for SIZE=0x20, got {len(data_lines)}"
    assert len(data_lines[0]) == 64, f"Expected 64 hex chars (32 bytes), got {len(data_lines[0])}"


@pytest.mark.skipif(shutil.which('pyspike') is None, reason="pyspike CLI not on PATH")
def test_atexit_dump_does_not_register_when_env_unset(tmp_path):
    """Subprocess pyspike WITHOUT GTX_DDR_DUMP → no dump file is created."""
    if not ELF_PATH.exists():
        pytest.skip(f"{ELF_PATH} missing")

    suspicious_dump = tmp_path / "should_not_exist.hex"
    env = os.environ.copy()
    env.pop('GTX_DDR_DUMP', None)
    env.pop('GTX_DDR_DUMP_ADDR', None)
    env.pop('GTX_DDR_DUMP_SIZE', None)
    env.pop('GTX_NO_EXIT', None)

    result = subprocess.run(
        ['pyspike', '--extlib=riscv.gtx', '--extension=gtx', str(ELF_PATH)],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )

    assert result.returncode == 0
    assert not suspicious_dump.exists(), (
        "atexit hook fired despite GTX_DDR_DUMP being unset — D-04 conditional gate broken"
    )
```

**Confidence: HIGH** — pattern direct from P5 test_regression_fw_act.py.

### Example 2: parametrize roll for VRF-04 (Plan 04 GREEN-fill skeleton)

See "Pattern 6" example above (`tests/gtx/test_regression_fw_full.py`). Uses existing P5 subprocess+5-tier-skip discipline with tier #5 elevated to hard FAIL.

### Example 3: pyproject.toml diff for D-02 + D-13 (Plan 01 + Plan 05)

```diff
 [project]
 name = "spike"
 dynamic = ["version"]
 ...
 requires-python = ">=3.10"
 dependencies = [
   "numpy>=2.0,<3",
 ]

+[project.scripts]
+pyspike-verify = "riscv.gtx._verify:main"
+
 [project.urls]
 Homepage = "https://github.com/liuyu81/pyspike"
 ...
 [tool.setuptools.package-data]
 riscv = [
   "data/bin/spike",
   ...
 ]
+"riscv.gtx" = [
+  "data/firmware/*.elf",
+  "data/golden/*.hex"
+]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pkg_resources` | `importlib.resources.files()` | Python 3.9 (importlib.resources update); deprecation in 3.11 | P6 D-14 uses files() exclusively; cp310 baseline = stdlib path. |
| `setup.py`-only config | `pyproject.toml` + minimal `setup.py` for cmdclass | setuptools 60+ (PEP 621 support); current P1 already uses pyproject.toml | P6 follows existing pattern; only build_py extension goes in setup.py. |
| `data_files` (system-prefix relative) | `package_data` (package-relative) | setuptools 70+ deprecates `data_files` for wheel | P6 D-13 uses package_data exclusively. |
| `r.path('pkg', 'file').open()` | `r.files('pkg').joinpath('file').open()` | Python 3.9 introduces files() API; 3.11 deprecates path() | P6 D-14 single-codepath. |
| Script wrappers in `scripts/` | `[project.scripts]` console_scripts | setuptools 75+ + PEP 621 | P6 keeps existing scripts (`pyspike`, `spike`) for back-compat AND adds `pyspike-verify` via `[project.scripts]`. |

**Deprecated/outdated in pyproject.toml:**
- `[project] script-files`: still works but is the older `tool.setuptools.script-files` form. P6 doesn't change this — pre-existing entries stay.

## Open Questions

### 1. Vendor MUL_MAT requires command file `n1s16_mul_mat_cmd.txt`

**What we know:** `vendor/.../test/MUL_MAT/n1s16/data/n1s16_mul_mat_cmd.txt` exists and contains shape config (`shape=[1024,1024], src0=[1024,1024], src1=[1024,512], dst=[1024,512]`).

**What's unclear:** Whether the hand-written `mul_mat.S` for P6 needs to mimic this shape config or can use a smaller fixed shape (16×16×16 like mm_basic.S).

**Recommendation:** Use mm_basic.S 16×16×16 shape — vendor's 1024×1024 is a stress test, not a correctness probe. The single-row golden (32 bytes truncation per D-15 sizing) means we only need to validate the FIRST row's correct multiplication, which 16×16 covers as well as 1024×1024.

### 2. Whether `tests/gtx/test_regression_fw_act.py` 5-tier should be modified or kept as-is

**What we know:** P5 test currently has 5 tiers; tier #5 graceful-skips on missing dump. P6 D-04 expects tier #5 to be hard PASS.

**What's unclear:** Whether to modify the existing test to assert `actual_dump.exists()` (turning tier #5 into hard fail) or keep tier #5 graceful and rely on `test_regression_fw_full.py` for P6 hard assertions.

**Recommendation:** Keep `test_regression_fw_act.py` 5-tier intact (sentinel for backward compat) BUT have Plan 02's Wave 1a `test_atexit_ddr_dump.py::test_atexit_dump_fires_on_systemexit` be the canonical hard-PASS for atexit semantics. Plan 04's `test_regression_fw_full.py` is the canonical hard-PASS for VRF-04. This keeps the P5 test as a documentary "tier #5 → expected hard PASS" record without breaking it.

### 3. Whether `pyspike-verify --help` exits 0 or 2 in cibuildwheel

**What we know:** argparse `--help` defaults to exit code 0.

**What's unclear:** Whether some bash-quoting issue in cibuildwheel test-command might mangle exit codes when chained with `&&`.

**Recommendation:** Plan 05 acceptance includes a literal `pyspike-verify --help; echo $?` smoke test on dev machine before cibuildwheel run.

### 4. Whether wheel size with 12 ops + ~80KB cap will trip the "manylinux2014 max wheel size" implicit limits

**What we know:** Total assets ~80KB; existing wheel is ~MB-scale due to libriscv.so etc.

**What's unclear:** No explicit manylinux2014 wheel size limit other than D-15's 50MB project policy.

**Recommendation:** D-15 50MB cap is well-clear. No action needed.

### 5. Whether `python -m riscv.gtx._verify` vs. `python3 -m riscv.gtx._verify` differs on cp310-cp312

**What we know:** Both invoke the same module's `if __name__ == "__main__"` block.

**What's unclear:** Some manylinux2014 images alias `python` to python3, others don't.

**Recommendation:** All test commands use `python3` explicitly (matches P5 test_regression_fw_act.py).

## Risks & Suggested Reconsiderations

### NOT changing CONTEXT decisions, but flagging for future reference

The user already locked decisions D-01..D-18. The following are observations only, NOT recommendations to change scope:

**(a) D-08 vendor `n1s16_<op>.c` 1:1 single-build:** As documented in "Vendor .elf Build Cost" section, the vendor sources have a non-trivial build dependency chain that v1 cannot satisfy in CI. Plan 03 should adopt the hand-written .S precedent instead. This is **fully aligned with D-12** (vendor build is dev-only) but means D-08's "1:1 single-build" effectively becomes "1:1 vendor-equivalent kernel re-implementation in hand-written .S using GTX RoCC opcodes directly." The plan-stage author should make this distinction explicit in the plan body.

**(b) D-15 50MB cap:** With current core set (~12 ops, ~80KB total), even the worst-case 100x scaling is well under 1MB. The 50MB cap is comfortably non-binding. Plan-stage author should not over-engineer for size optimization; keep the assets uncompressed for debug-friendliness.

**(c) D-04 atexit conditional gate:** The conditional `if os.getenv('GTX_DDR_DUMP'): atexit.register(...)` only runs at import time. If a user mutates `os.environ['GTX_DDR_DUMP']` AFTER importing `riscv.gtx`, the registration won't change. This is correct vendor semantics (vendor `gtx_npu_core.cc:127` is also import-time). For tests that need to toggle env vars per-call, use `subprocess.run(..., env=env)` (which spawns a fresh interpreter that re-evaluates the import-time gate). The Plan 02 sketch and the test sketches above already use this pattern.

**(d) D-05 single-NPU global:** Documented in "NPU Instance Lookup". v2 multi-hart consideration noted.

**None of these warrant decision change in v1 P6.**

## Sources

### Primary (HIGH confidence)

- `vendor/gtx_cpp_reference/gtx/verify.py` (lines 1-388) — Vendor verify.py source, read end-to-end. Direct port target for D-01.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` (lines 1-261) — atexit hook source. Lines 56-73 = `gtx_atexit_ddr_dump()` body; line 127 = `std::atexit(gtx_atexit_ddr_dump)` registration.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc` (lines 1-138) — WJOIN exit semantics; line 116-117 comment "DDR dump moved to atexit handler" justifies D-04.
- `vendor/gtx_cpp_reference/test/RELU/n1s16/data/n1s16_relu_ref.txt` — vendor _ref.txt format verified live (parsed 524288 bytes via _verify_minimal._parse_hex).
- `vendor/gtx_cpp_reference/test/run_tests_n1s16.sh` — vendor build flow source; lines 41-43 show `gtx-firmware/include/` dependency chain.
- `tests/gtx/_verify_minimal.py` — existing 78-LOC mini-port; D-01 absorption target. Already proven by P4 mm_basic.elf + P5 activation_relu_gelu.elf strict pass.
- `tests/gtx/test_regression_fw_act.py` (P5) — 5-tier graceful-skip pattern; D-09 / Plan 04 direct precedent.
- `tests/gtx/data/elf/Makefile` (P4/P5) — hand-written .S precedent for D-08 fallback.
- `src/main/python/riscv/gtx/__init__.py` — D-04 registration site.
- `src/main/python/riscv/gtx/ddr.py` — D-05 wrapper site; existing `ddr_dump_to_file(mem, filename, addr, size)` (P3 D-09 args-only) ready to wrap.
- `src/main/python/riscv/gtx/npu.py` — D-04/D-05 NPU instance-tracking site.
- `pyproject.toml` (lines 15-127) — current cibuildwheel + setuptools.package-data baseline.
- `setup.py` (lines 114-118) — existing build_py cmdclass extension point for D-13.
- `MANIFEST.in` (lines 1-16) — current include/exclude rules.
- Live Python 3.10.12 experiment (2026-05-07) — verified `atexit + sys.exit(0)` fires; verified `os._exit(0)` skips; verified `importlib.resources.files()` available in cp310.

### Secondary (MEDIUM confidence — verified with official docs)

- [Python 3.10 atexit docs](https://docs.python.org/3.10/library/atexit.html) — official "atexit fires on sys.exit() but NOT on os._exit()" behavior.
- [Python 3.12 importlib.resources docs](https://docs.python.org/3.12/library/importlib.resources.html) — `files()` API + 3.12 anchor= rename.
- [setuptools 75 datafiles userguide](https://setuptools.pypa.io/en/latest/userguide/datafiles.html) — `package_data` glob support + subdirectory namespacing.
- [setuptools 75 pyproject_config](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) — `[project.scripts]` + dynamic version coexistence.
- [PEP 621 pyproject.toml spec](https://packaging.python.org/en/latest/specifications/pyproject-toml/) — `[project.scripts]` static syntax.
- [cibuildwheel custom build_py](https://cibuildwheel.pypa.io/en/stable/options/) — test-command customization respects PEP 517 build backends.

### Tertiary (LOW confidence — for awareness only, not load-bearing)

- [CPython issue #103512](https://github.com/python/cpython/issues/103512) — atexit + SystemExit-inside-handler bug; confirms our pattern (SystemExit at top level is fine, NOT inside handler).
- [pypa/cibuildwheel discussion #2065](https://github.com/pypa/cibuildwheel/discussions/2065) — custom cmdclass + cibuildwheel interaction notes.
- [pypa/setuptools issue #227](https://github.com/pypa/setuptools/issues/227) — security restriction on package_data path traversal.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components are stdlib (argparse, importlib.resources, atexit, os) plus existing project deps (NumPy 2.x, setuptools 75+).
- Architecture: HIGH — all 6 patterns are direct ports of vendor C++ behavior or existing P4/P5 Python patterns.
- Pitfalls: HIGH — verified live for items 1-3, verified by source inspection for items 4-6, verified by docs for items 7-8.
- Vendor _ref.txt format: HIGH — verified by live parse.
- atexit semantics: HIGH — verified by live experiment.
- Stats dict mapping: HIGH — verified by reading 388 LOC of vendor verify.py.
- importlib.resources: HIGH — Python 3.10/3.11/3.12 docs all confirm files() API stability.
- cibuildwheel test-command: MED-HIGH — minor bash-quoting risk; fully mitigated at plan-stage.
- D-08 vendor build cost: HIGH — verified by reading vendor source includes + run_tests_n1s16.sh dependency chain.
- Vendor core op set survey: MED — list is candidate; final selection requires plan-stage filter for op-specific quirks.
- NPU instance lookup recommendation: HIGH — vendor C++ pattern is single-global; v1 single-hart confirmed.

**Research date:** 2026-05-07
**Valid until:** 2026-06-06 (30 days; setuptools/Python ecosystem stable; cibuildwheel may change cp313 default but cp310-cp312 are LTS-stable)

## RESEARCH COMPLETE
