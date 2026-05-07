---
phase: 05-vec-act-pool
plan: 04
subsystem: act
tags: [act, pool, format-cvt, fp8-codec, fp64-fp16, scale-offset, signed-zero-canon, lut-precompute, py-rocc]

# Dependency graph
requires:
  - phase: 05-vec-act-pool
    plan: 01
    provides: "encoding constants (GTX_F7_SCVT_*, GTX_F7_FCVT_*, GTX_F7_POOL_*) + act_core.py + act_engine.py + ops/act.py importable stubs + test_op_format.py / test_pooling.py RED scaffolds"
  - phase: 05-vec-act-pool
    plan: 02
    provides: "vec_engine helpers (_l0_block_view, _l1_view_addr, _fp16_low16/_fp16_high16) -- pattern source for act_engine; FP32-internal explicit-loop reduction precedent (vec_core.vsum_kernel)"
  - phase: 05-vec-act-pool
    plan: 03
    provides: "act_engine helpers (_resolve_nest_spu, _fp16_low16/_fp16_high16) reused by firmware_pool + firmware_format; 12 @handlers occupy funct7 in {0x28, 0x2A, 0x2C, 0x2D, 0x2F} -- no collision with this plan's funct7 in {0x20, 0x21, 0x22, 0x24, 0x25, 0x30, 0x31}"
  - phase: 04-mm-subsystem
    provides: "_registry.@handler decorator + ops/__init__.py auto-import + MockProcessor.state property"
provides:
  - "act_core.py: 2 pool kernels (pool_max + pool_avg with signed-zero canon) + 9 cvt kernels (cvt_qh/hq/ih/hi/hn apply scale+offset; cvt_sh/hs/dh/hd bit-pattern preserving) + 2 LUT builders + populated FP8_TO_FP16_LUT (256) + FP16_TO_FP8_LUT (65536) at module import"
  - "act_engine.firmware_pool full body (forward direction; length+kernel_size from GSPR_OPERAND1/2 & 0xFFFF; output_len = length // kernel_size)"
  - "act_engine.firmware_format full body (9 src/dst routes covering 7 cvt directions including FP64; scale+offset unpacked from GSPR_OPERAND2 [low16=scale, high16=offset]; length from XPR[insn.rs1] & 0xFFFF)"
  - "ops/act.py 7 new @handlers (5 cvt-dispatch at funct7=0x20/0x21/0x22/0x24/0x25 + 2 pool at 0x30/0x31)"
  - "test_pooling.py 3 GREEN tests + test_op_format.py 8 GREEN tests covering ACT-03 + ACT-04"
  - "FP8 codec divergence from NVIDIA E4M3 documented + tested (subnormal 2^-6 base; exp=0xF + frac=0 inf; exp=0xF + frac>0 NaN)"
affects: [05-05-oracle, 05-06-regression]

# Tech tracking
tech-stack:
  added: []  # zero new runtime deps; pure Python + NumPy
  patterns:
    - "FP8 codec LUTs precomputed at module import via dedicated builder functions (D-14, D-15) -- replaces Plan 01 zeros placeholder. Build cost: FP8->FP16 ~0.2 ms; FP16->FP8 ~30 ms; one-time at module import."
    - "Pool kernels use explicit Python for-loop FP32 accumulator (Pitfall 2 lock; same precedent as vec_core.vsum_kernel + act_core.softmax). NEVER np.sum/np.dot/np.einsum on FP16."
    - "Avg-pool signed-zero canonicalization via `avg += np.float32(0.0)` AFTER division (vendor cc:211). Mirrors IEEE 754 (-0.0) + (+0.0) = +0.0; bit-pattern goes 0x8000 -> 0x0000."
    - "FP8 LUT-based encoding hot path: `LUT[fp16_arr.view(uint16).astype(intp)]` with explicit intp cast for safe NumPy 2.x fancy indexing."
    - "Single dispatch @handler per cvt funct7 (mask_funct3=False, None inner key) inspecting `npu.gspr[GSPR_OPCODE] & 1` for direction. Mirrors vendor gtx_npu_act.cc:245. RESEARCH §format_cvt Sub-op direction discrimination authoritative."
    - "scale/offset semantics asymmetry: applied for FP16<->{FP8,INT8,INT32}; NOT applied for FP16<->{FP32,FP64} (bit-pattern preserving). Tested explicitly via test_fp32_fp16_no_scale + test_fp64_fp16_no_scale that set non-trivial scale/offset and assert they are IGNORED."

key-files:
  created: []
  modified:
    - src/main/python/riscv/gtx/act_core.py
    - src/main/python/riscv/gtx/act_engine.py
    - src/main/python/riscv/gtx/ops/act.py
    - tests/gtx/test_pooling.py
    - tests/gtx/test_op_format.py

key-decisions:
  - "act_core.pool_max + pool_avg use explicit FP32 for-loop reduction. pool_max accumulates max via Python `if v > val`; pool_avg adds in FP32 then `avg += 0.0` canonicalizes -0.0 -> +0.0 BEFORE the FP16 cast at writeback. Matches vendor gtx_npu_act.cc:198-213 line-for-line."
  - "_build_fp16_to_fp8_lut is direct port of vendor gtx_npu.h:182-221 (4-case RNE rounding). 64KB at ~30 ms one-time build. The LUT IS the spec; per-call hot path is one line `LUT[fp16_arr.view(uint16)]` (D-15 enabled by NumPy fancy indexing)."
  - "FP8 inf encoding NOT sign-preserving (vendor `sign8 | 0xF8` forces -inf byte regardless of input sign). FP16 +inf (0x7C00) re-encodes to FP8 0xF8, NOT 0x78. test_fp8_roundtrip_identity skips inf bytes (and NaN bytes). Documented divergence; fixture asserts `fp16_to_fp8[0x7C00] == 0xF8` to lock the behavior."
  - "5 cvt @handlers (NOT 7 individual mnemonics) registered at distinct funct7 values; sub_op&1 dispatch at handler entry. SCVT_QH/SCVT_HQ share funct7=0x20; SCVT_IH/SCVT_HI share 0x21; FCVT_SH/FCVT_HS share 0x24; FCVT_DH/FCVT_HD share 0x25. Total disasm entries +7 (qh, ih, hn, sh, dh, pool_m, pool_a) = 85 (was 78)."
  - "kernel_size=0 silently NOPs in firmware_pool (vendor guard `kernel_size > 0` at gtx_npu_act.cc:175). Plan body suggested defaulting kernel_size=0 to 1 but that diverges from vendor. We mirror exactly: if kernel_size==0, return 0 without touching L1."
  - "test_fp8_roundtrip_identity adjusted to skip inf bytes (alongside NaN bytes). Plan body's claim 'all 256 FP8 inputs round-trip' is too strict given the vendor's `sign8 | 0xF8` forced-negative-inf encoding. Round-trip identity holds for finite values + correct sign-zero handling; non-finite has its own equivalence classes documented."

patterns-established:
  - "LUT precompute pattern at module-load: builder function as canonical spec; LUT itself is the cache. _build_fp8_to_fp16_lut (256 entries, ~0.2 ms) + _build_fp16_to_fp8_lut (65536 entries, ~30 ms) build once and stay alive for module lifetime. Replaces Plan 01 zeros placeholder; downstream plans that need fancy-indexed FP8 conversion get vectorized hot paths for free."
  - "Single funct7 + GSPR_OPCODE-based sub_op dispatch at @handler entry: cleaner than registering 2 separate funct3-keyed handlers when funct3 is NOT the discriminator. Use mask_funct3=False (None inner key) and inspect `npu.gspr[GSPR_OPCODE] & 1` per vendor gtx_npu_act.cc:245."
  - "Per-cvt scale/offset semantics encoded in firmware_format dispatch: 5 routes (fp16-fp8, fp8-fp16, fp16-int8, int8-fp16, int32-fp16) call cvt_* with (scale, offset); 4 routes (fp32-fp16, fp16-fp32, fp64-fp16, fp16-fp64) call cvt_* WITHOUT scale/offset. Test asserts non-trivial scale/offset are IGNORED for the bit-pattern-preserving routes -- regression-proof against future drift."

requirements-completed: [ACT-03, ACT-04]

# Metrics
duration: 14min
completed: 2026-05-07
---

# Phase 5 Plan 04: Pool + format_cvt Summary

**ACT-03 + ACT-04 GREEN: 2 pool kernels (max + avg with -0.0 -> +0.0 canon) + 9 cvt kernels (7 directions including FP64↔FP16; FP8 codec via 256-entry + 65536-entry LUTs precomputed at module import) + firmware_pool + firmware_format full bodies + 7 new @handlers (5 cvt-dispatch + 2 pool) + 11 GREEN op-level unit tests, 0 regressions across the 242-test P3+P4+P5 suite.**

## Performance

- **Duration:** ~14 min (4 atomic commits incl. RED prep)
- **Tasks:** 3
- **Files modified:** 5
- **Test surface change:** +11 GREEN (3 test_pooling + 8 test_op_format); 3 skipped overall (down from 14 baseline post-Plan-03).
- **LUT build cost (one-time at module import):**
  - `_build_fp8_to_fp16_lut`: ~0.2 ms (256 iterations, simple branches).
  - `_build_fp16_to_fp8_lut`: ~30 ms (65536 iterations, RNE rounding logic).
  - Total module import overhead: ~30 ms additional vs Plan 03 baseline.

## Accomplishments

- ACT-03 (pool): test_max_pool_output_length + test_avg_pool_signed_zero_canon + test_pool_always_forward GREEN.
- ACT-04 (format_cvt 7 directions): test_scale_offset_packing + test_fp8_roundtrip_identity + test_fp8_subnormal_decode + test_fp8_exp_max + test_int8_fp16_scale_offset + test_int32_fp16_normalize + test_fp32_fp16_no_scale + test_fp64_fp16_no_scale GREEN.
- FP8 LUTs precomputed at module import time (D-14, D-15): `FP8_TO_FP16_LUT` shape (256,) + `FP16_TO_FP8_LUT` shape (65536,) populated and verified.
- 7 new @handlers registered in ops/act.py (5 cvt-dispatchers + 2 pool); collect_disasms() returns 85 entries (was 78 after Plan 03 = +7).
- Pitfall 5 lock: subnormal at 2^-6 base (NOT NVIDIA's 2^-9); exp=0xF+frac=0 inf, exp=0xF+frac>0 NaN -- parametrized over all 16 subnormal patterns + all 16 exp=0xF patterns.
- Pitfall 6 lock: scale = OP2 & 0xFFFF (low 16 FP16); offset = (OP2 >> 16) & 0xFFFF (high 16 FP16). `test_scale_offset_packing` uses asymmetric values (scale=2.0, offset=0.5) so swap would yield drastically different output (3*2+0.5=6.5 vs 3*0.5+2.0=3.5) -- divergence-proof.

## Task Commits

1. **Task 1 prep (RED):** `7be0371` (test) -- 11 RED tests against unbuilt kernels (3 pool + 8 format_cvt). All 11 fail before kernel impl lands.
2. **Task 1 GREEN (act_core kernels):** `ae7ad83` (feat) -- pool_max + pool_avg + 9 cvt kernels + 2 LUT builders + populated FP8_TO_FP16_LUT (256) and FP16_TO_FP8_LUT (65536). 5/11 tests GREEN at this point (kernel-level only; firmware_* still stubs).
3. **Task 2 GREEN (act_engine):** `4dc80cc` (feat) -- firmware_pool + firmware_format full bodies replacing Plan 01 stubs. 11/11 tests GREEN.
4. **Task 3 (ops/act.py @handlers):** `a496f0d` (feat) -- 7 new @handlers registered (5 cvt-dispatch + 2 pool). disasm count 85 (+7 over Plan 03).

**Plan metadata commit:** added below.

## Files Created/Modified

| File | LOC | Role |
|------|-----|------|
| `src/main/python/riscv/gtx/act_core.py` | 352 | +159 LOC: 2 pool kernels + 9 cvt kernels + 2 LUT builders + populated LUTs |
| `src/main/python/riscv/gtx/act_engine.py` | 393 | +104 LOC: firmware_pool + firmware_format full bodies |
| `src/main/python/riscv/gtx/ops/act.py` | 243 | +88 LOC: 7 new @handlers (5 cvt-dispatch + 2 pool) |
| `tests/gtx/test_pooling.py` | 126 | +94 LOC: 3 GREEN tests for ACT-03 |
| `tests/gtx/test_op_format.py` | 359 | +283 LOC: 8 GREEN tests for ACT-04 |

**Total source delta:** +728 LOC across act_core/act_engine/ops/act + tests.

## FP8 Codec Divergence Verification (Pitfall 5)

GTX FP8 is **labeled** "E4M3" but has two intentional divergences from NVIDIA E4M3:

| Aspect | GTX (this impl) | NVIDIA E4M3 | Test |
|--------|----------------|--------------|------|
| Subnormal base | 2^-6 (smallest = (1/8)*2^-6 = 0.001953125) | 2^-9 (smallest ≈ 0.000244) | `test_fp8_subnormal_decode` |
| `exp=0xF, frac=0` | inf (sign-preserved) | not encodable | `test_fp8_exp_max` (asserts inf for ±) |
| `exp=0xF, frac>0` | NaN | NaN (only NaN sentinel) | `test_fp8_exp_max` (asserts NaN for all 14 patterns) |
| Inf round-trip on encode | NOT sign-preserving (`sign8 \| 0xF8` forces 0xF8) | n/a | `test_fp8_roundtrip_identity` skips inf bytes; locks `fp16_to_fp8[0x7C00] == 0xF8` |

The divergences are **vendor HW spec** (not bugs). All bit patterns verified by direct port of `gtx_npu.h:154-221`.

## Avg-pool Signed-Zero Canonicalization (Pitfall test)

Vendor `gtx_npu_act.cc:211` does `avg += 0.0f` AFTER the division. IEEE 754 says `(-0.0) + (+0.0) = +0.0`. Without this canon, an avg-pool of `[0.0, -0.0]` would produce `-0.0` in some FP16 paths, hashing to bit pattern `0x8000`. With the canon, the bit pattern is `0x0000` -- mandatory for golden-hex matching.

`test_avg_pool_signed_zero_canon` directly asserts `int(out.view(np.uint16)[0]) == 0x0000` (NOT `0x8000`) for the input `[0.0, -0.0, -0.0, 0.0]` with `kernel_size=2`. Sanity check on non-zero windows (`[1.0, 2.0, 3.0, 4.0]` -> `[1.5, 3.5]`) confirms the kernel still does correct averages.

## Decisions Made

1. **kernel_size=0 NOP semantics (vendor mirror):** the plan body proposed defaulting `kernel_size=0` to 1 to avoid div-by-zero. Vendor `gtx_npu_act.cc:175` instead has an outer guard `if (... && kernel_size > 0) { ... }` that silently NOPs when `kernel_size==0`. We chose vendor exactness over plan-body suggestion -- in the firmware_pool body, `if kernel_size == 0: return 0` happens before any L1 view is taken or any pool kernel is called. Documented in firmware_pool docstring.
2. **Inf round-trip skip in test_fp8_roundtrip_identity:** plan body said "all 256 FP8 inputs round-trip". Vendor `gtx_fp16_to_8` does `sign8 | 0xF8` on the inf path which forces the sign bit to 1, so FP16 +inf (0x7C00) re-encodes to FP8 0xF8 (=-inf when decoded), NOT 0x78. The strict round-trip claim is impossible. We adjusted the test to skip inf bytes (alongside NaN bytes) and added explicit assertions `fp16_to_fp8[0x7C00] == 0xF8` and `fp16_to_fp8[0xFC00] == 0xF8` to lock the documented divergence.
3. **Single funct7-only @handler per cvt direction (mask_funct3=False):** registered 5 cvt @handlers at distinct funct7 (0x20/0x21/0x22/0x24/0x25); each one inspects `npu.gspr[GSPR_OPCODE] & 1` to pick direction. Plan body's count "9 new @handlers" actually delivered as 7 unique mnemonics in disasm (`scvt.qh`, `scvt.ih`, `scvt.hn`, `fcvt.sh`, `fcvt.dh`, `pool.m`, `pool.a`) -- 5 cvt-dispatch + 2 pool. The `<done>` field of Task 3 documents this resolution.
4. **`_BYTES_PER_ELEM` lookup table:** firmware_format uses a single dict for src/dst byte sizes (`{'fp16':2, 'fp32':4, 'fp64':8, 'fp8':1, 'int8':1, 'int32':4}`). Cleaner than scattering byte-size literals through the if/elif chain. Used to compute `in_size = length * _BYTES_PER_ELEM[src_kind]` BEFORE slicing `l1[addr_a:addr_a + in_size]`. Same idiom holds for the output writeback (we use `len(out_arr)`, computed from the FP/Int byte view).
5. **`np.frombuffer(bytes(in_bytes), dtype=...)` instead of `view`:** required because L1 byte view has odd alignment edge cases when sliced. `bytes(...)` makes a contiguous copy that is safely re-typed via `np.frombuffer(...)`. Avoids dtype-alignment ValueError on some L1 byte ranges.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test Bug] test_fp8_roundtrip_identity claim was too strict on inf bytes.**
- **Found during:** Task 1 GREEN run (test_fp8_roundtrip_identity reported `(120, 31744, 248)`).
- **Issue:** Vendor `gtx_fp16_to_8` does `return h_frac ? (sign8 | 0xF8 | 0x01) : (sign8 | 0xF8);` in the inf branch. The OR with `0xF8` forces the result byte to have sign=1 regardless of the input FP16 sign bit. So FP8 byte `0x78` (positive inf) decodes to FP16 inf (0x7C00); re-encoding 0x7C00 yields `0 | 0xF8 = 0xF8` (-inf in FP8 decode), NOT 0x78. The plan body's claim "all 256 FP8 round-trip" misses this divergence.
- **Fix:** Test now skips inf bytes (alongside NaN bytes) in the round-trip loop, and adds explicit assertions `fp16_to_fp8[0x7C00] == 0xF8` and `fp16_to_fp8[0xFC00] == 0xF8` to lock the vendor inf-encoding behavior. The bug is in the vendor code; we faithfully ported the bug, then documented it.
- **Files modified:** `tests/gtx/test_op_format.py`
- **Verification:** test_fp8_roundtrip_identity GREEN; +2 explicit divergence assertions.
- **Committed in:** `ae7ad83` (Task 1 GREEN, alongside the kernel impl).

---

**Total deviations:** 1 auto-fixed (Rule 1 - Test Bug). The vendor's intentional inf-encoding bit pattern was correctly ported; the test had to be adjusted to acknowledge the divergence as a HW-spec property.

**Impact on plan:** No scope expansion. No new files. No new dependencies. Plan delivered exactly the 11 tests promised, all GREEN, plus +2 explicit-divergence assertions on inf encoding (a refinement beyond the plan body).

## Issues Encountered

None -- apart from the 1 deviation above. Authentication gates: none.

## User Setup Required

None -- pure code change, no env vars / external services.

## Wave 1b -> Wave 2 Unblock Signal

**Plan 05 (oracle parity, Wave 2) and Plan 06 (.elf strict-mode regression) are now unblocked.** Plan 04 closes the last critical compute primitive in Phase 5. The full ACT-* + VEC-* + POOL/FORMAT compute surface is GREEN at unit-test level.

Plans 05/06 readiness:
- **Plan 05** can now wire `_oracles.py` 32-op host-side parity tests against the full ACT (relu/prelu/gelu/tanh/sigmoid/softmax/esum) + format_cvt + pool stack -- every kernel referenced by `verify_ref.py` has a callable Python implementation in `act_core.py` (D-03 lock).
- **Plan 06** can now run `activation_relu_gelu.elf` against the strict-mode `_verify_minimal.compare_hex` because RELU forward + GELU reversed are both wired (Plan 03), and pool + format_cvt are wired (this plan). If the firmware exercises any cvt op, the LUT-based hot path is bit-exact.

**Pattern handoff for Plan 05:** the `_BYTES_PER_ELEM` table in act_engine.py is a clean source-of-truth for any oracle that needs to match firmware_format byte counts.

**Pattern handoff for Plan 06:** the FP8/FP16 LUT-based round-trip pattern (`LUT[arr.view(uint16)]`) is a numba-friendly fancy-index that should JIT cleanly in P7 if profiling identifies cvt as a hot path.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| 4 task commits present (`7be0371`, `ae7ad83`, `4dc80cc`, `a496f0d`) | All in `git log --oneline -6` |
| act_core.py importable + 7 cvt + 2 pool kernels callable | PASS (smoke test FP8_TO_FP16_LUT[0x00]=0; [0x78]=inf; [0x7F]=NaN; pool_max([1,3,2,4,5,7,6,8],4)=[4,8]; pool_avg([0,-0],2)=0x0000) |
| FP8_TO_FP16_LUT shape (256,) + FP16_TO_FP8_LUT shape (65536,) | PASS |
| act_engine.py uses `proc.state.XPR` (not `proc.get_state()`) in production | PASS (`grep -c proc.get_state` = 1, only docstring "Do NOT use" warning) |
| No `np.sum`/`np.dot`/`np.einsum` calls in act_core.py | PASS |
| Pitfall 6: scale = OP2 & 0xFFFF in firmware_format | PASS (via _fp16_low16 which masks 0xFFFF) |
| 7 new @handlers registered (funct7 in {0x20, 0x21, 0x22, 0x24, 0x25, 0x30, 0x31}) | PASS (5 cvt-dispatch + 2 pool) |
| collect_disasms() returns 85 (was 78 in Plan 03) | PASS (+7) |
| All 11 test_pooling + test_op_format tests GREEN with `--noconftest` | PASS (11 passed in 0.84s) |
| Full P3+P4+P5 suite: 242 passed / 3 skipped / 0 failed (was 231/14/0 baseline) | PASS -- no regressions, +11 new GREEN |
| LOC: act_core (352 >= 200) + act_engine (393 >= 220) + ops/act (243 >= 100) | PASS |
| FP8 subnormal divergence (Pitfall 5): 2^-6 base verified for all 14 subnormal patterns | PASS (test_fp8_subnormal_decode parametrize) |
| FP8 exp=0xF semantics: frac=0 -> inf (sign-preserved); frac>0 -> NaN | PASS (test_fp8_exp_max parametrize over 16 patterns) |
| Avg-pool signed-zero canon: pool_avg([0,-0],2) bit pattern is 0x0000 not 0x8000 | PASS (test_avg_pool_signed_zero_canon) |
| Pool always forward (CONTEXT D-08): firmware_pool reads ADDRA, writes ADDRR; ADDRA preserved | PASS (test_pool_always_forward) |

All 15 verification checks pass.

---
*Phase: 05-vec-act-pool*
*Plan: 04 (pool + format_cvt)*
*Completed: 2026-05-07*
