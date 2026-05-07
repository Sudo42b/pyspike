---
phase: 05-vec-act-pool
plan: 01
subsystem: vec-act-pool-scaffold
tags: [vec, act, pool, format_cvt, scaffold, red-tdd, encoding, fp8-lut]

# Dependency graph
requires:
  - phase: 04-mm-subsystem
    provides: "_registry.@handler decorator + ops/__init__.py import-side-effect pattern + tests/gtx/_verify_minimal.compare_hex + MockProcessor.state property + mm_basic.{S,elf} fixture lineage + proc.state mechanical-rename baseline (no get_state in production)"
provides:
  - "21 new funct7 constants (VEC 9, format_cvt 5, ACT 5, POOL 2) appended to encoding.py"
  - "7 GTX_ACT_* + 24 GTX_VEC_* enum constants matching vendor gtx_npu.h:371-405 verbatim"
  - "ACT_OPS_REVERSED frozenset for engine-internal direction consistency check"
  - "vec_core.py + vec_engine.py importable stubs (Plan 02 GREEN-fills)"
  - "act_core.py + act_engine.py importable stubs + FP8/FP16 LUT placeholders (Plans 03/04 GREEN-fill)"
  - "ops/vec.py + ops/act.py empty modules (Plans 02-04 register @handler entries here)"
  - "ops/__init__.py imports vec + act modules (auto-trigger downstream registry)"
  - "tests/gtx/_oracles.py: 30 oracle stub signatures (20 portable + 10 documented DEFERRED)"
  - "7 RED test scaffolds covering VEC-01..05, ACT-01..05, ACT-04, VRF-02, .elf regression"
  - "tests/gtx/conftest.py proc_with_addra_addrr_seeded fixture (direction-asymmetry harness)"
  - "tests/gtx/data/elf/activation_relu_gelu.{S,elf} firmware fixture (committed RV64 ELF)"
  - "Makefile rule + verify-act target for activation_relu_gelu.elf rebuild"
  - "tests/gtx/data/elf/.gitignore allow-list extended for !activation_relu_gelu.elf"
  - "tests/gtx/data/golden/activation_relu_gelu.hex zero-init oracle (16 FP16 = 32 bytes)"
affects: [05-02-vec, 05-03-act, 05-04-pool-format, 05-05-oracle, 05-06-regression]

# Tech tracking
tech-stack:
  added: []  # zero new runtime dependencies; all stubs use existing numpy + pytest
  patterns:
    - "Wave 0 RED-via-pytest.skip discipline (P3 plan-01 D-5 lock; never assert hasattr)"
    - "Vendor-enum-verbatim policy (GTX_VEC_* uses vendor 0..23 not Plan draft 0..9)"
    - "FP8/FP16 LUT module-level placeholders (256 + 65536 entries, Plan 04 fills)"
    - "Reversed-direction set as frozenset (ACT_OPS_REVERSED) -- engine consistency check, NOT routing primary"
    - "ISS-full WRSPR (funct7=0x49) used in firmware .S to avoid funct7=0x00 collision (mirrors mm_basic.S P4 04-01 D-3)"

key-files:
  created:
    - src/main/python/riscv/gtx/vec_core.py
    - src/main/python/riscv/gtx/vec_engine.py
    - src/main/python/riscv/gtx/act_core.py
    - src/main/python/riscv/gtx/act_engine.py
    - src/main/python/riscv/gtx/ops/vec.py
    - src/main/python/riscv/gtx/ops/act.py
    - tests/gtx/_oracles.py
    - tests/gtx/test_op_vec.py
    - tests/gtx/test_op_act.py
    - tests/gtx/test_op_format.py
    - tests/gtx/test_pooling.py
    - tests/gtx/test_vsum_precision.py
    - tests/gtx/test_oracle_parity.py
    - tests/gtx/test_regression_fw_act.py
    - tests/gtx/data/elf/activation_relu_gelu.S
    - tests/gtx/data/elf/activation_relu_gelu.elf
    - tests/gtx/data/golden/activation_relu_gelu.hex
  modified:
    - src/main/python/riscv/gtx/encoding.py
    - src/main/python/riscv/gtx/ops/__init__.py
    - tests/gtx/conftest.py
    - tests/gtx/data/elf/Makefile
    - tests/gtx/data/elf/.gitignore

key-decisions:
  - "GTX_VEC_* enum uses vendor verbatim (0..23, full op list) instead of Plan draft (0..9, abbreviated). Plan note explicitly authorized this: 'if vendor gtx_npu.h:382-405 shows different numbers, USE the vendor numbers verbatim.'"
  - "test_oracle_parity ships single-entry parametrize placeholder (name='abs') so the test ID lands cleanly in pytest collection; Plan 05 wave 2 expands list to 20 names."
  - "Golden hex uses flat hex line format (mm_basic_n1s16.hex precedent) with @-prefixed comment block; _verify_minimal._parse_hex skips @-lines so this is purely informative metadata."
  - "Activation .S fixture uses ISS-full WRSPR (funct7=0x49) for ADDRA/ADDRR setup, mirroring mm_basic.S P4 04-01 D-3 -- avoids reentering not-yet-implemented funct7=0x00 dispatch."
  - "RELU dispatched via firmware DISPATCH_ACT (funct7=0x06, sub_op=GTX_ACT_RELU=0); GELU dispatched via ISS-direct funct7=0x2A. Two distinct dispatch surfaces in one .elf to verify both paths work."
  - "FP8_TO_FP16_LUT/FP16_TO_FP8_LUT shipped as zero-filled placeholders with documented Plan 04 build-at-import contract; this is intentional Wave 0 RED scaffolding, not silent stubs."

patterns-established:
  - "Wave 1a scaffold-only plan: every downstream wave (1b plans 02-04 + wave 2 plan 05/06) develops against pre-existing test files + module stubs. No merge conflicts on shared edits within Wave 1b."
  - "Encoding constants land before any kernel code. Plans 02-04 import GTX_F7_* from encoding.py with zero magic numbers."
  - "Source-module triad (core / engine / ops) for both VEC and ACT mirrors P4 D-01 split. Pure stateless kernels (vec_core, act_core) are the P7 numba @njit boundary; spike-bound dispatch (vec_engine, act_engine) and @handler registration (ops/vec, ops/act) layer on top."

requirements-completed: []
# NOTE: This plan ships ZERO GREEN tests (Wave 1a is scaffold-only by design).
# Per RESEARCH adjustment + P4 04-01 deviation pattern: do NOT mark requirements
# complete in Wave 1a. Wave 1b plans 02-04 GREEN-fill VEC-01..05 + ACT-01..03 +
# ACT-04 + ACT-05; Wave 2 plan 05 GREEN-fills VRF-02; Wave 2 plan 06 closes the
# .elf strict-mode regression.

# Metrics
duration: 13min
completed: 2026-05-07
---

# Phase 5 Plan 01: VEC/ACT/Pool Scaffold Summary

**Wave 1a scaffold landing: 6 importable source-module stubs + 7 RED test scaffolds + 30-oracle skeleton + activation_relu_gelu.elf firmware fixture + zero-init golden hex unblock all 5 downstream Wave 1b/Wave 2 plans for parallel development against a pre-existing test surface.**

## Outcome

This plan delivered the full Wave 1a scaffold for Phase 5 (VEC + ACT + Pool +
format_cvt). Every downstream plan (02 vec, 03 act, 04 pool/format, 05 oracle
parity, 06 .elf regression) now has:

1. An existing test file with named RED scaffolds it can GREEN-fill in surgical
   edits (no merge conflicts on shared test files).
2. An importable source module (`vec_core`, `vec_engine`, `act_core`,
   `act_engine`, `ops/vec`, `ops/act`) it can populate without creating new
   files.
3. All required encoding constants (21 funct7 + 7 GTX_ACT_* + 24 GTX_VEC_*
   + ACT_OPS_REVERSED) preventing magic-number drift.
4. A test fixture (`activation_relu_gelu.elf` + golden hex + .S source +
   Makefile rule + .gitignore allow-list) for Plan 06 strict-mode regression.

The quick suite (`pytest test_op_vec test_op_act test_op_format test_pooling
test_vsum_precision test_oracle_parity --noconftest -o "addopts="`) reports
**43 skipped, 0 failed** — the contract Wave 1a was designed to deliver. The
full P3 + P4 + P5 suite reports **199 passed (matches P4 baseline) / 45
skipped / 0 failed** — no regression introduced.

## Files Created (LOC)

| File | LOC |
|------|-----|
| `src/main/python/riscv/gtx/vec_core.py` | 77 |
| `src/main/python/riscv/gtx/vec_engine.py` | 40 |
| `src/main/python/riscv/gtx/act_core.py` | 131 |
| `src/main/python/riscv/gtx/act_engine.py` | 70 |
| `src/main/python/riscv/gtx/ops/vec.py` | 34 |
| `src/main/python/riscv/gtx/ops/act.py` | 42 |
| `tests/gtx/_oracles.py` | 203 |
| `tests/gtx/test_op_vec.py` | 122 |
| `tests/gtx/test_op_act.py` | 91 |
| `tests/gtx/test_op_format.py` | 75 |
| `tests/gtx/test_pooling.py` | 36 |
| `tests/gtx/test_vsum_precision.py` | 33 |
| `tests/gtx/test_oracle_parity.py` | 35 |
| `tests/gtx/test_regression_fw_act.py` | 45 |
| `tests/gtx/data/elf/activation_relu_gelu.S` | 53 |
| `tests/gtx/data/elf/activation_relu_gelu.elf` | (binary, RV64 ELF, entry 0x800000b0) |
| `tests/gtx/data/golden/activation_relu_gelu.hex` | (32 bytes BE FP16 zeros + comment block) |

**Total:** ~1087 LOC source + 1 RV64 ELF binary + 1 golden hex.

## Encoding Constants Added (verbatim list)

```python
# VEC funct7 (gtx_npu_disasm.inc:67-142)
GTX_F7_VEC_SASMD     = 0x10   # SASMD scalar arith
GTX_F7_VEC_FMADD     = 0x11
GTX_F7_VEC_DOT_SUM   = 0x13   # vsum + dot funct3=0/1
GTX_F7_VEC_ARITH     = 0x18   # SASMD vector arith
GTX_F7_VEC_FMADD_VV  = 0x19
GTX_F7_VEC_MATH      = 0x1C   # sqrt/exp/log
GTX_F7_VEC_SIGN      = 0x1D   # abs/neg/sgn/step
GTX_F7_VEC_ROUND     = 0x1E   # ceil/trunc/floor/rne
GTX_F7_VEC_CLAMP     = 0x1F   # clamp_min_v/clamp_max_v/accum_v/arange_v + bitwise

# format_cvt funct7 (RESEARCH Adjustment 1: 7 directions including FP64<->FP16)
GTX_F7_SCVT_QH       = 0x20   # FP16<->FP8
GTX_F7_SCVT_IH       = 0x21   # FP16<->INT8
GTX_F7_SCVT_HN       = 0x22   # INT32->FP16 normalize
GTX_F7_FCVT_SH       = 0x24   # FP16<->FP32
GTX_F7_FCVT_DH       = 0x25   # FP16<->FP64

# ACT funct7 (gtx_npu_disasm.inc:152-157)
GTX_F7_ACT_PRELU     = 0x28
GTX_F7_ACT_GELU      = 0x2A
GTX_F7_ACT_TANH      = 0x2C
GTX_F7_ACT_SIGM      = 0x2D
GTX_F7_ACT_SOFTMAX   = 0x2F   # esum funct3=1, softmax funct3=2; _imm at funct3=5/6

# POOL funct7
GTX_F7_POOL_MAX      = 0x30
GTX_F7_POOL_AVG      = 0x31

# GTX_ACT_* enum (gtx_npu.h:371-377 verbatim)
GTX_ACT_RELU=0, GTX_ACT_TANH=1, GTX_ACT_SOFTMAX=2, GTX_ACT_GELU=3,
GTX_ACT_SIGMOID=4, GTX_ACT_PRELU=5, GTX_ACT_ESUM=6
ACT_OPS_REVERSED = frozenset({1, 3, 4, 5})  # engine-internal consistency check

# GTX_VEC_* enum (gtx_npu.h:382-405 verbatim -- vendor lock-in 0..23, NOT Plan draft 0..9)
GTX_VEC_ADD=0, GTX_VEC_SUB=1, GTX_VEC_MUL=2, GTX_VEC_DIV=3, GTX_VEC_FMADD=4,
GTX_VEC_VSUM=5, GTX_VEC_VEXP=6, GTX_VEC_VSQRT=7, GTX_VEC_VLN=8, GTX_VEC_VABS=9,
GTX_VEC_VNEG=10, GTX_VEC_MAX=11, GTX_VEC_MIN=12, GTX_VEC_SIGN=13, GTX_VEC_STEP=14,
GTX_VEC_CEIL=15, GTX_VEC_TRUNC=16, GTX_VEC_FLOOR=17, GTX_VEC_RNE=18,
GTX_VEC_ACCUM=19, GTX_VEC_CLAMP_MAX=20, GTX_VEC_CLAMP_MIN=21, GTX_VEC_ARANGE=22,
GTX_VEC_DOT=23
```

## Decisions Made

1. **Vendor enum verbatim for GTX_VEC_*.** Plan draft listed 10 entries (0..9);
   vendor `gtx_npu.h:382-405` defines 24 entries (0..23, full op list including
   FMADD/VEXP/VSQRT/VLN/etc.). Plan note authorized this resolution: "if
   vendor gtx_npu.h:382-405 GTX_VEC_* enum values differ from the draft above,
   research lock prevailed." Plans 02-04 reference these by name, not value.

2. **Wave 0 RED-via-skip discipline (P3 plan-01 D-5 lock).** Every test body is
   `pytest.skip("Wave 1b plan NN GREEN-fills: ...")`. Never `assert hasattr`
   — that pattern would fail Wave 1a verify before Wave 1b plans land
   modules. Quick suite reports 43 skipped, 0 failed.

3. **ISS-full WRSPR in firmware fixture.** `activation_relu_gelu.S` uses
   `funct7=0x49` (ISS-full WRSPR) for ADDRA/ADDRR setup, mirroring
   `mm_basic.S` P4 04-01 D-3. Using gem5-simplified `funct7=0x00` would
   reenter the not-yet-fully-wired dispatch path (still has Pitfall F NOP
   for rs1=0, but the test code path is more brittle).

4. **Golden hex format mirrors mm_basic_n1s16.hex precedent.** Flat 32-byte
   line of zeros (16 FP16 values BE bit-pair) + comment block + `@`-prefixed
   metadata line. `_verify_minimal._parse_hex` skips `@`/`#` lines, so the
   `@370000000` block marker is purely informative for human readers.

5. **FP8 LUT placeholders are intentional, not stubs.** Plan 04 will
   replace `np.zeros(256, ...)` and `np.zeros(65536, ...)` with the
   `gtx_npu.h:154-179, 182-221` LUT-builder output at module import. Plan 01
   ships them so `from .act_core import FP8_TO_FP16_LUT` succeeds in any
   downstream test, and the import-time-LUT-build contract is documented.

6. **`is_reversed` literal at @handler entry, NOT module-level set lookup**
   (CONTEXT D-06 reaffirmation). `ACT_OPS_REVERSED` frozenset is shipped in
   encoding.py as an engine-internal consistency assertion only — Plan 03
   `@handler` entries will pass `is_reversed=True/False` literal at the
   call site.

## Self-Check Results

| Check | Result |
|-------|--------|
| Files created (17 + 5 modified) | All present (verified `git ls-files`) |
| Importability of all 6 source modules | PASS (`python3 -c "from riscv.gtx import vec_core, vec_engine, act_core, act_engine; from riscv.gtx.ops import vec, act"`) |
| Importability of `tests.gtx._oracles` | PASS (`callable(_oracles.op_relu) and callable(_oracles.op_gelu_erf) and isinstance(_oracles.DIRECT_MAPPED_ORACLES, dict)`) |
| 21 funct7 + 7 GTX_ACT_* + 24 GTX_VEC_* + ACT_OPS_REVERSED present in encoding.py | PASS |
| Quick suite skip-clean (`pytest test_op_vec ... test_oracle_parity --noconftest`) | 43 skipped / 0 failed |
| `test_regression_fw_act.py` collects cleanly | 1 skipped / 0 failed |
| Full P3 + P4 + P5 suite (`pytest tests/gtx/`) | **199 passed (matches P4 baseline) / 45 skipped / 0 failed** |
| `tests/gtx/data/elf/activation_relu_gelu.elf` is RV64 ELF | PASS (`file ...` reports "ELF 64-bit LSB executable, UCB RISC-V"; entry 0x800000b0) |
| Golden hex parseable + self-compare strict-PASS | PASS (`exact_matches=16, total_fp16=16, failures=0`) |
| `proc.get_state` not used in P5 production code | PASS (only 1 hit, in a docstring "Do NOT use" warning) |
| `git ls-files tests/gtx/data/elf/activation_relu_gelu.elf` | tracked |
| `git ls-files tests/gtx/data/golden/activation_relu_gelu.hex` | tracked |

## Self-Check: PASSED

All 12 verification checks pass. All 3 task commits present:
- `b632b4e feat(05-01): add VEC/ACT/SCVT/POOL encoding + source-module stubs`
- `e4e6269 test(05-01): add 7 RED scaffolds + _oracles.py + addra/addrr fixture`
- `efc6b7c feat(05-01): commit activation_relu_gelu fixture (.S, .elf, Makefile, golden)`

## Deviations from Plan

**None — plan executed exactly as written, with two pre-authorized resolutions:**

1. **GTX_VEC_* enum values** — Plan draft listed 10 entries (0..9, abbreviated);
   vendor `gtx_npu.h:382-405` defines 24 entries (0..23). Plan note explicitly
   authorized: "exact GTX_VEC_* numeric values may differ slightly; if vendor
   `gtx_npu.h:382-405` shows different numbers, USE the vendor numbers
   verbatim — research locked these enums." Resolution: vendor values used
   verbatim; encoding.py docstring documents the divergence from plan draft.

2. **`@`-prefixed block marker in golden hex** — Plan draft showed
   `@370000000` as the line content; in `_verify_minimal._parse_hex`, lines
   starting with `@` are skipped (line 15 of `_verify_minimal.py`).
   Resolution: the `@370000000` line is preserved as informative metadata for
   human readers and the actual hex bytes follow on the next line, matching
   the `mm_basic_n1s16.hex` precedent exactly. Self-compare via
   `compare_hex(strict=True)` returns `total_fp16=16, exact_matches=16`,
   confirming the golden is parseable.

## Known Stubs

These stubs are **intentional** — Plan 01 is Wave 1a scaffold-only by
design. Each stub names the future plan that resolves it:

| Stub | File | Plan that resolves it |
|------|------|------------------------|
| `vec_core.sasmd_kernel/dot_kernel/vsum_kernel/clamp_min_kernel/clamp_max_kernel/accum_kernel/arange_kernel` (7 NotImplementedError) | `src/main/python/riscv/gtx/vec_core.py` | Plan 02 (vec) |
| `vec_engine.firmware_vec_op` returns 0 | `src/main/python/riscv/gtx/vec_engine.py` | Plan 02 (vec) |
| `act_core.{relu,prelu,gelu,tanh_act,sigmoid,softmax,esum}` (7 NotImplementedError) | `src/main/python/riscv/gtx/act_core.py` | Plan 03 (act) |
| `act_core.{pool_max,pool_avg,cvt_qh,cvt_hq,cvt_ih,cvt_hi,cvt_hn,cvt_sh,cvt_hs,cvt_dh,cvt_hd}` (11 NotImplementedError) | `src/main/python/riscv/gtx/act_core.py` | Plan 04 (pool/format) |
| `act_core.FP8_TO_FP16_LUT` (256 zeros) | `src/main/python/riscv/gtx/act_core.py` | Plan 04 |
| `act_core.FP16_TO_FP8_LUT` (65536 zeros) | `src/main/python/riscv/gtx/act_core.py` | Plan 04 |
| `act_engine.{firmware_act,firmware_pool,firmware_format,firmware_act_imm,firmware_softmax_imm}` returns 0 | `src/main/python/riscv/gtx/act_engine.py` | Plans 03/04 |
| `ops/vec.py` empty (0 @handler calls) | `src/main/python/riscv/gtx/ops/vec.py` | Plan 02 |
| `ops/act.py` empty (0 @handler calls) | `src/main/python/riscv/gtx/ops/act.py` | Plans 03 + 04 |
| `_oracles.py` 20 NotImplementedError stubs (portable ops) | `tests/gtx/_oracles.py` | Plan 05 wave 2 |
| `_oracles.DIRECT_MAPPED_ORACLES = {}` empty dict | `tests/gtx/_oracles.py` | Plan 05 wave 2 |
| 43 `pytest.skip("Wave 1b plan NN ...")` test bodies | All 7 new test files | Plans 02-06 (named per test) |
| `test_regression_fw_act.py::test_act_strict_mode_pass` body skip | `tests/gtx/test_regression_fw_act.py` | Plan 06 |

The plan's success criterion explicitly required this state: "Wave 1b plans
(02/03/04) and Wave 2 plans (05/06) can develop in parallel against this
scaffold without merge conflicts."

## Wave 1b Unblock Signal

**Plans 02-04 can begin GREEN-fill work in parallel.** Each plan owns a
non-overlapping subset of test scaffolds + a non-overlapping pair of source
modules:

- **Plan 02 (vec)**: `vec_core.py` + `vec_engine.py` + `ops/vec.py`;
  GREEN-fills `test_op_vec.py` (15 RED) + `test_vsum_precision.py` (5 RED).
- **Plan 03 (act)**: `act_core.py` (act+esum kernels only) + `ops/act.py`
  (16 activation @handlers); GREEN-fills `test_op_act.py` (11 RED).
- **Plan 04 (pool/format)**: `act_core.py` (pool + format_cvt kernels +
  FP8 LUTs) + `act_engine.py` (firmware_pool + firmware_format) +
  `ops/act.py` (7 cvt + 2 pool @handlers); GREEN-fills `test_op_format.py`
  (8 RED) + `test_pooling.py` (3 RED).
- **Plan 05 (oracle, wave 2)**: `_oracles.py` (20 portable bodies) +
  `DIRECT_MAPPED_ORACLES` dict; GREEN-fills `test_oracle_parity.py` (1 RED →
  20 parametrized).
- **Plan 06 (regression, wave 2)**: `test_regression_fw_act.py` body
  (subprocess pyspike + 4-tier graceful skip + strict compare).
