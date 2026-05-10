---
seed: p9-vendor-sweep-non-multi-tile-bugs
created: 2026-05-11
upstream: 08-04 SUMMARY (Plan 04 surgical scope D-04)
priority: medium
scope: out-of-P8 (different bug class than MTDMA-01)
---

# Vendor sweep — line-1+ divergences after MTDMA-01 fix

## Background

After P8 Plan 04 landed the surgical credit_ld_chk -> deferred-flush fix
and wired vendor `.elf` harness env (GTX_DDR_INIT, GTX_DDR_DUMP_ADDR=0xf000000,
GTX_NO_EXIT=1), the vendor 84-op sweep (`tests/gtx/test_regression_fw_full_sweep.py`)
was run for the 12-op SMOKE_SET_12 (CONTEXT D-11 + RESEARCH plan-stage NEG/DIV/EXP).

ABS now PASSES byte-exact across all 96 tiles (the multi-tile boundary that
P8 was created to fix — verified manually with raw `pyspike` invocation +
diff against `n1s16_abs_ref.txt`).

However, several other smoke set ops show divergence at line 1+ that is
NOT related to multi-tile boundary (Plan 03 INVESTIGATION already flagged
these as separate bug classes; they were NOT in P8 scope per D-04 surgical
scope rule).

## Observed divergences (Plan 04 dev-machine smoke run)

Out of the 12 SMOKE_SET_12 ops, after Plan 04 fix:
- **PASSES (1 confirmed):** GELU (single-tile, fits in 60 KB; line 0 of golden=line 0 of dump)
- **FAILS at line 0/1+:** ADD, MUL, RELU, SIGMOID, NEG, DIV, EXP — all NOT multi-tile
- **FAILS at line 1497 (single-bit):** LEAKY_RELU — FP precision delta (1 ULP)
- **PASSES via ABS (manual):** ABS — 96 tiles byte-exact (multi-tile fix verified)
- **SKIPPED (vendor dir absent):** TANH, SUM (no `<root>/TANH`, `<root>/SUM`)

ABS via the harness is in-flight at SUMMARY time but the standalone reproduce
(`/tmp/p8-04-fix/abs_fixed.hex` + diff against `n1s16_abs_ref.txt`) confirms
byte-exact match for all 196609 lines beyond trailing zero-padding.

## Per-op divergence pattern hypothesis

| Op | First diverge line | Symptom | Likely cause class |
|---|---|---|---|
| ADD | 0 | DUMP zeros where GOLDEN has values | Operand pre-stage / address mismatch (input B at 0x2000000? checked via input.txt parsing) |
| MUL | 0 | Similar to ADD | Same |
| RELU | 1 | DUMP has different non-zero pattern | clamp_min sub-op / __set_spm_addr ordering |
| SIGMOID | 1 | Similar to RELU | Activation engine bug |
| NEG | 0 | Within-tolerance count = 22, mismatches=17386 | Sign-bit flip implementation |
| DIV | n/a | Mostly mismatches | div_vv vec engine subop |
| EXP | 0 | All 16496 fp16 fail | Activation EXP path |
| LEAKY_RELU | 1497 | 2 mismatch lines (1-bit LSB delta) | FP precision: 0.01 slope coeff FP32 vs FP16 internal compute |

## Recommendation for P9

These are NOT multi-tile bugs. Plan 03 INVESTIGATION already discriminated
these from the MTDMA-01 surface. Each requires its own root-cause
investigation:

1. **ADD/MUL** — verify dual-input operand staging (rs1=A, rs2=B), DDR
   address layout, pyspike `firmware_dma_sloop_load` for second-operand
   load, and exec_vector_op handling of `is_vs` flag.
2. **RELU/SIGMOID/EXP** — examine activation engine `exec_activation` path
   vs vector engine `exec_vector_op`. Vendor `gtx_npu_act.cc` shows ADDR_R
   -> ADDR_A reversal for some ops; verify pyspike `act_engine.py`.
3. **NEG** — likely simple bit-flip of FP16 sign bit; verify sign-extension
   or sign-mask in pyspike vec_engine SIGN family.
4. **DIV** — investigate `div_vv` sub-op encoding; vendor stem differs
   (`n1s16_div_vv.elf` vs hand-built `div_vv` not present).
5. **LEAKY_RELU** — single-bit (1 ULP) FP precision delta at row 1497;
   likely `0.01 * x` slope multiplication done FP16-naive in pyspike vs
   FP32-internal in vendor C++. Out of P8 numba precision scope.

## Out-of-scope for P8

P8 scope (CONTEXT D-04 — Surgical fix scope = Root-cause specific):
- ✅ Multi-tile DMA orchestration parity at MAX_SHARED_DMA_BYTES boundary
- ✅ State-machine reset audit (Plan 01 MTDMA-04 GREEN)
- ✅ Tile-2 unit test guard (Plan 01 MTDMA-03 GREEN)
- ✅ Vendor sweep harness env wiring (Plan 04 VTW-01)
- ❌ Per-op compute correctness (RELU, SIGMOID, EXP, etc.) — different bug class

## Action items for v1.2 (P9)

- [ ] Investigate per-op divergence using the same investigation methodology
      as 08-03-INVESTIGATION.md (full-region golden diff + first-diverge-line
      analysis + root-cause hypothesis)
- [ ] Decide whether to wire per-op `subprocess_timeout` higher than 600s
      for ABS (currently passes within 600s but tight)
- [ ] If LEAKY_RELU's 1-ULP delta is judged acceptable, weaken
      `compare_hex(strict=True)` to `compare_hex(strict=True, ulp=1)` for
      that op (or document as known precision limit)
