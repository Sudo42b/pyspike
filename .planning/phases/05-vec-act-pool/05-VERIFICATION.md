---
phase: 05-vec-act-pool
verified: 2026-05-07T00:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification:
  previous_status: none
  previous_score: n/a
  gaps_closed: []
  gaps_remaining: []
  regressions: []
---

# Phase 5: VEC/ACT/Pool Verification Report

**Phase Goal:** Every VEC/ACT/pool/format-cvt op produces FP16 output bit-exact with C++; activation direction asymmetry honored; VSUM/DOT use FP32-internal-accumulate; `verify_ref.py` 32-op oracle suite passes as pytest.
**Verified:** 2026-05-07
**Status:** PASSED
**Re-verification:** No — initial verification
**Test result:** `264 passed, 2 skipped, 0 failed` (`tests/gtx/ -q -p no:pylint -p no:mypy --no-cov -o "addopts="`, 7.93 s).

## Goal Achievement — Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `test_op_vec.py` GREEN; SASMD/DOT/VSUM/CLAMP bit-exact vs oracle | VERIFIED | full suite 264 passed; `_oracles.DIRECT_MAPPED_ORACLES` 20 entries; CLAMP at single funct7=0x1F with 4 funct3 sub-modes (`ops/vec.py:157-178`) |
| 2 | `test_op_act.py` GREEN; ADDRA/ADDRR direction proven; ESUM writes L0 scalar | VERIFIED | `test_direction_asymmetry_table` (`test_op_act.py:345`) parametrizes 7 op_ids; `test_esum_writes_l0_scalar` (`test_op_act.py:164`); ESUM writeback at `act_engine.py:144-150` writes to `(GSPR_OPERAND3 & 0x1F) * 32` |
| 3 | `test_op_format.py` GREEN; 7 cvt directions incl. FP64↔FP16 | VERIFIED | 9 `cvt_*` kernels in `act_core.py:284-352`; 5 @handlers in `ops/act.py:173-227` (3 bidirectional via sub_op&1 + 1 INT32→FP16 + 1 FP64↔FP16 = 7 directions); RESEARCH Adjustment 1 honored (`GTX_F7_FCVT_DH=0x25`) |
| 4 | `test_pooling.py` GREEN; output_len = length//ksize; -0.0 → +0.0 in avg | VERIFIED | `act_core.pool_avg:171` `avg += np.float32(0.0)` canonicalizes signed zero; `act_engine.firmware_pool:314` computes `out_len = length // kernel_size` |
| 5 | `.elf` strict-mode regression PASS or graceful-skip | VERIFIED | `test_regression_fw_act.py:22-27` 5-tier skip; subprocess + `compare_hex(strict=True)` invoked; 1 SKIP observed in env (atexit dump = P6 territory, P3 D-09/P4 04-05 lineage) |

## Required Artifacts

| Artifact | Status | Notes |
|----------|--------|-------|
| `src/main/python/riscv/gtx/vec_core.py` | VERIFIED | 7 stateless kernels; explicit Python `for` FP32 accumulator in `dot_kernel:74-79`, `vsum_kernel:90-93`, `accum_kernel:124-127` |
| `src/main/python/riscv/gtx/vec_engine.py` | VERIFIED | `firmware_vec_op` decodes rs1 (`vec_size = (rs1 & 0xFFFF) or 0x10000`, line 101); funct3-based L0/L1 dispatch (lines 119-188) |
| `src/main/python/riscv/gtx/act_core.py` | VERIFIED | 7 activation kernels + pool_max/pool_avg + 9 cvt kernels + FP8 LUTs precomputed at import (lines 276-277); custom E4M3 (2^-6 subnormal, inf-on-exp=0xF, lines 194-205) |
| `src/main/python/riscv/gtx/act_engine.py` | VERIFIED | 5 firmware entries (act, act_imm, softmax_imm, pool, format); ACT_OPS_REVERSED consistency assert at line 103 |
| `src/main/python/riscv/gtx/ops/vec.py` | VERIFIED | 22 @handlers (8 SASMD scalar + 2 DOT/SUM + 8 SASMD vector + 4 CLAMP) |
| `src/main/python/riscv/gtx/ops/act.py` | VERIFIED | 19 @handlers (6 ISS act + 6 _imm act + 5 cvt + 2 pool) |
| `src/main/python/riscv/gtx/encoding.py` | VERIFIED | All P5 funct constants present; `ACT_OPS_REVERSED = frozenset({TANH, GELU, SIGMOID, PRELU})` at line 169 (single source of truth) |
| `tests/gtx/_oracles.py` | VERIFIED | `DIRECT_MAPPED_ORACLES` (20 ops, line 282) + `DEFERRED_REASONS` (12 entries, line 316); `op_gelu_erf` calls `pytest.skip` for scipy ban |
| `tests/gtx/data/elf/activation_relu_gelu.{S,elf}` | VERIFIED | Both present (1.2K elf, 2.3K .S) |
| `tests/gtx/data/golden/activation_relu_gelu.hex` | VERIFIED | 887 bytes |

## Anti-Pattern Scan

| Check | Result | Evidence |
|-------|--------|----------|
| `np.sum/np.dot/np.einsum/np.matmul` in vec_core/act_core | PASS | 9 grep hits, 100% in docstrings/comments warning "NEVER use" — zero in code |
| `proc.get_state()` in production gtx code | PASS | 0 hits in `src/main/python/riscv/gtx/` (only docstring warnings at `vec_engine.py:24`, `act_engine.py:27`); test files use back-compat method on `MockProcessor` (acceptable, P4 04-05 lineage) |
| scipy / numba imports | PASS | 0 hits across src/ and tests/gtx/ |
| New C++ code for VEC/ACT/format/pool | PASS | No matching files in `src/main/cpp/`; CLAUDE.md "C++ 추가 코드 금지" honored |
| VSUM anti-pattern test uses divergent input | PASS | `test_vsum_precision.py:35` uses `[1024.0] + 5000*[0.4]` (Rule-1 deviation) and asserts `naive_fp16 != expected` to prove distinction |

## Requirements Coverage

| Req | Status | Evidence |
|-----|--------|----------|
| VEC-01..05 | SATISFIED | REQUIREMENTS.md:217-221 marked Complete; vec_core/vec_engine/ops/vec.py present and exercised by `test_op_vec.py` GREEN |
| ACT-01 (forward) | SATISFIED | RELU/SOFTMAX/ESUM @handlers `is_reversed=False`; `test_direction_asymmetry_table` proves ADDRR overwrite |
| ACT-02 (reversed) | SATISFIED | PRELU/GELU/TANH/SIGM @handlers `is_reversed=True`; ACT_OPS_REVERSED frozenset is single source of truth |
| ACT-03 (pooling) | SATISFIED | `pool_max`+`pool_avg` in act_core; signed-zero canon at line 171 |
| ACT-04 (format_cvt) | SATISFIED | 7 directions wired; scale+offset unpack at `act_engine.py:350-351` |
| ACT-05 (_imm L0) | SATISFIED | `firmware_act_imm` + `firmware_softmax_imm` separate engines; 6 _imm @handlers |
| VRF-02 | SATISFIED | 20 oracles parametrized in `test_oracle_parity.py:114`; 12 deferred with documented reasons |

## Behavioral Spot-Checks

| Behavior | Command | Result |
|----------|---------|--------|
| Full gtx suite passes | `python3 -m pytest tests/gtx/ -q -p no:pylint -p no:mypy --no-cov -o "addopts="` | 264 passed / 2 skipped / 0 failed in 7.93 s |

## Gaps

None.

---
_Verified: 2026-05-07_
_Verifier: Claude (gsd-verifier, Opus 4.7 1M)_
