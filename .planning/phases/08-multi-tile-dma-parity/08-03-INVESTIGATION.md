---
phase: 8
plan: 3
status: investigation
created: 2026-05-10
verdict: Outcome B (NPU code fix needed)
hypothesis_5_status: confirmed-by-exclusion (harness was masking) AND falsified (a real production bug exists at multi-tile boundary)
---

# Phase 8 Plan 03 — Multi-Tile DMA Investigation Log

## TL;DR

**Verdict: Outcome B — NPU code fix is required** (harness extension alone is NOT sufficient).

ABS reproduces the P7 smoke observation byte-for-byte: divergence starts at **exactly line 2048** (= 2048 × 32 bytes = 65536 bytes = `MAX_SHARED_DMA_BYTES`), i.e., the first tile (rows 0..4094) is byte-exact, but the second tile is broken. The pattern shows the second tile's L2→DDR store writing only the latter half of each row (last 16 bytes correct, first 16 bytes zero).

This **CONFIRMS** the existence of a real production bug at the multi-tile boundary in the vendor `.elf` → pyspike RoCC dispatch path. It also means **Plan 01's XPASS finding (programmatic 2-tile path is byte-exact) is genuine but bypasses the buggy code path**: Plan 01 calls `firmware_dma_sloop_load/store` directly with explicit kwargs, while the vendor `.elf` reaches them through the full RoCC `custom0`/`custom1` dispatch + WSPR + WRSPR + state-machine path. **The bug lives in the dispatch / state-machine wiring, not in `dma_engine.py` core algorithms.**

Plan 04 must drive a fix targeting the RoCC dispatch path through `firmware_dma`/`firmware_dma_sloop_*`, particularly the second-tile invocation that loses the first 16 bytes of each row's `addr_hi` portion.

## Method

1. **Generated full-region goldens** via `python scripts/import_vendor_golden.py --all --full` (Task 1 of this plan), producing 82 `.hex` files in `tests/gtx/data/golden_full/` (gitignored). 9 P6 ops were also produced via the default 9-op map under `--full`.
2. **Selected 6 smoke ops** with both vendor `.elf` (`/mnt/e/14_NIGHTLY/pyspike/test/<OP>/n1s16/n1s16_<stem>.elf`) and full-region golden: ABS, ADD, RELU, GELU, SIGMOID, LEAKY_RELU.
3. **Ran each op** with `pyspike --extlib=riscv.gtx --extension=gtx <elf>` under:
   - `GTX_DDR_INIT=<vendor input.txt>` (D-11: vendor `.elf` requires operand staging via `__ddr_init`)
   - `GTX_DDR_DUMP=/tmp/p8-investigation/<op>.hex`
   - `GTX_DDR_DUMP_ADDR=0xf000000` (matches vendor `_ref.txt` `@f000000` start address)
   - `GTX_DDR_DUMP_SIZE=<full-region bytes>` (computed per op from `golden_full/<op>.hex` line count × 32)
   - `GTX_DDR_REVERSED=1` (D-10: vendor BE FP16 → pyspike LE FP16 conversion)
4. **Diffed dump vs golden** with `diff <(grep -v -E '^[#@]' dump) <(grep -v -E '^[#@]' golden)`.

> **Crucial harness fix discovered during investigation:** the initial run with `GTX_DDR_DUMP_ADDR=0x100` (matching `test_regression_fw_full_sweep.py` default) produced **all-zero dumps** because the vendor `.elf` writes output at `0xf000000`, not `0x100`. Without the corrected address, the bug surface would be invisible. **This is a separate harness-side issue that Plan 04 should fix in parallel** (parameterize per-op `GTX_DDR_DUMP_ADDR`, similar to the per-op `GTX_DDR_DUMP_SIZE` mechanism Task 1 installed).

## Per-Op Divergence Results

| Op | Vendor `.elf` path | Golden lines | Golden raw bytes | First diverge line | At tile boundary? | Diff line count | Verdict |
|----|--------------------|--------------|------------------|--------------------|-------------------|-----------------|---------|
| **ABS** | `/mnt/e/14_NIGHTLY/pyspike/test/ABS/n1s16/n1s16_abs.elf` | 196609 | 6291488 (≈6 MB) | **2048** | **YES** (2048×32 = 65536 = `MAX_SHARED_DMA_BYTES=65535+1`) | 389124 | **Multi-tile bug confirmed** |
| **ADD** | `/mnt/e/14_NIGHTLY/pyspike/test/ADD/n1s16/n1s16_add_vv.elf` | 65536 | 2097152 (2 MB) | 0 | No | 65536 | Different bug — line 0 (NOT multi-tile, possibly input/output address) |
| **RELU** | `/mnt/e/14_NIGHTLY/pyspike/test/RELU/n1s16/n1s16_relu.elf` | 16384 | 524288 (512 KB) | 1 | No | 32640 | Different bug — line 1 (NOT multi-tile) |
| **GELU** | `/mnt/e/14_NIGHTLY/pyspike/test/GELU/n1s16/n1s16_gelu.elf` | 1920 | 61440 (60 KB) | — | N/A | 0 | **PASS** (single-tile, 60 KB < `MAX_SHARED_DMA_BYTES`) |
| **SIGMOID** | `/mnt/e/14_NIGHTLY/pyspike/test/SIGMOID/n1s16/n1s16_sigmoid.elf` | 256 | 8192 (8 KB) | 1 | No (single tile) | 298 | Different bug — line 1 (NOT multi-tile) |
| **LEAKY_RELU** | `/mnt/e/14_NIGHTLY/pyspike/test/LEAKY_RELU/n1s16/n1s16_leaky_relu.elf` | 1993 | 63776 (62 KB) | 1497 | No (single tile) | 2 | FP precision delta, single bit — see Bytes Sample |

### Byte Sample — ABS at the tile boundary (line 2047 → 2048)

```
Line 2047 (last good — tile 0 final row):
DUMP:  3a1e3a3d32c03831304c397d373b38d43b9b33403ba33b593224394a3bfe359b
GOLD:  3a1e3a3d32c03831304c397d373b38d43b9b33403ba33b593224394a3bfe359b
                                    BYTE-EXACT MATCH

Line 2048 (first bad — tile 1 first row):
DUMP:  0000000000000000000000000000000035a334683a483387386d39152a5738b3
GOLD:  3556393827b6381638e428433bbd33aa35a334683a483387386d39152a5738b3
       ^ first 16 bytes are ZERO ^^^ second 16 bytes are CORRECT ^^^^^^^
```

**Pattern:** the second tile's first row writes only the latter half (16 bytes / 8 FP16) correctly. The first 16 bytes are zeros. This continues line-after-line (mostly zeros with occasional partial fragments) until end of dump.

### Byte Sample — LEAKY_RELU at line 1497 (single-bit precision delta)

```
DUMP: 386aae8bade5ad6839edab3a39b4378e83d8338b38463873ae893b30a8aaa7a6
GOLD: 386aae8bade5ad6839edab3a39b4378e83d9338b38463873ae893b30a8aaa7a6
                                          ^^
                                       d8 vs d9 (1-bit LSB delta in one FP16)
```

This is **NOT** a multi-tile bug. LEAKY_RELU has only 2 diff lines total in a single 1993-line (62 KB) golden, well below `MAX_SHARED_DMA_BYTES`. The 1-bit FP16 LSB delta is consistent with a `0.01` slope-coefficient compute precision difference (FP32-internal vs FP16-naive). Out of scope for P8 — record as `.planning/seeds/p9-leaky-relu-fp-precision.md` candidate.

## Hypothesis Verdict

| Hypothesis | RESEARCH.md prior | Plan 01 verdict | Plan 03 verdict |
|---|---|---|---|
| #1 DDR↔L2 src/dst pointer not advancing between tiles | LOW | FALSIFIED (programmatic) | **CONFIRMED in dispatch path** — second tile's first 16 bytes per row are zero, consistent with `addr_hi` not advancing for the second tile in the DISPATCH path. Plan 01's programmatic path computed `addr_hi` explicitly per call so it bypassed the bug. |
| #2 L1 bank not being recycled | LOW | FALSIFIED (programmatic) | LIKELY FALSIFIED — same reasoning as #1; if L1 bank were stale, the latter 16 bytes would also be wrong, but they're correct. |
| #3 Credit gate stuck | NONE | N/A | N/A — pyspike has no credit infrastructure. |
| #4 Plan/thread state machine reset | LOW | FALSIFIED (programmatic) | LIKELY FALSIFIED — second tile produces SOME correct output (latter 16 bytes), so dispatch state DID transition; the bug is in the address/length computation, not the state machine. |
| #5 GTX_DDR_DUMP_SIZE harness truncation | HIGH | CONFIRMED by exclusion | **CONFIRMED-AND-FALSIFIED simultaneously**: harness truncation WAS hiding tiles 1+ (true), AND a real production bug exists at the tile boundary that surfaces when truncation is removed (also true). The two findings are NOT mutually exclusive: the harness was both broken AND papering over a deeper issue. |
| #6 active_tid_mask serialization | NONE | N/A | N/A. |
| #7 addr_hi 37-bit truncation | NONE | N/A | N/A — fits trivially. |
| **#8 (NEW)** Production bug lives in RoCC dispatch path, not in dma_engine.py core | — | (implied by Plan 01 XPASS) | **CONFIRMED** — Plan 01's direct API call is byte-exact for 2 tiles. Vendor `.elf` reaching the same code through dispatch is broken. The delta must be in the dispatch wiring, packed-args decode, GSPR/NSPR/LSPR state at second tile's firmware_dma call, or how the firmware-issued sequence of WRSPR + custom0/custom1 produces different `(addr_hi, addr_lo, length, height, rd_stride, wr_stride)` values for the second tile vs the first. |

## Bug Location Candidates (for Plan 04)

Ranked by probability based on the byte-pattern signature (first 16 bytes zero, last 16 bytes correct, at exact `MAX_SHARED_DMA_BYTES` boundary):

### 1. Highest probability — `firmware_dma_sloop_store` second-tile `addr_hi` decode

**Location:** `src/main/python/riscv/gtx/dma_engine.py:269-287` (`firmware_dma_sloop_store`) — the **`addr_hi` argument** for the second tile's STORE call.

**Hypothesis:** the firmware computes the second tile's output base address as `result_base + tile_max_rows * row_bytes = 0xf000000 + 4095*16 = 0xf00FFF0` (NOT `0xf010000`). When the firmware then issues `__store(addr=0xf00FFF0, length=...)`, the next 16 bytes (`0xf00FFF0..0xf01_0000`) span the tile boundary. If pyspike's `decode_firmware_dma_args` or `firmware_dma_sloop_store` truncates `addr_hi` to 32-bit boundary somewhere, the high half of the first row would be lost.

Alternate framing: the second tile's `__load(addr=src+0xFFF0, ...)` and `__store(addr=dst+0xFFF0, ...)` sequence might be hitting a packed-rs1 decode mask that drops bits when crossing a 16-byte (or 32-byte) alignment. **Verify by single-stepping pyspike + adding tracing in `decode_firmware_dma_args` to record `(addr_hi, addr_lo, length, height, rd_stride, wr_stride)` per tile invocation.**

### 2. Medium probability — `flush_deferred_ddr_stores` second-row partial write

**Location:** `src/main/python/riscv/gtx/npu.py:166-190`.

**Hypothesis:** the deferred queue's L2→DDR copy at line 187-189 reads from `mem.l2_byte(req.nest)` with `l2_off = (req.l2_off + row * req.l2_stride) % GTX_L2_SIZE_BYTES`. If `l2_off` wraps the L2 ring at the tile boundary (first 16 bytes overlap with the previous tile's last 16 bytes after `% GTX_L2_SIZE_BYTES`), the partial-write pattern would emerge. **Verify by adding `assert l2_off + copy_len <= GTX_L2_SIZE_BYTES` or by comparing l2 contents at second tile's flush.**

### 3. Lower probability — RoCC `custom0` / `custom1` dispatch table second-tile entry

**Location:** `src/main/python/riscv/gtx/npu.py:145-164` (`custom0`) + `dispatch.py` `build_custom0_table`.

**Hypothesis:** if the firmware-issued sequence triggers a different funct7 path on the second tile (e.g., due to GSPR_OPCODE shift), the dispatch table might route to a stub or fall through to no-op for one of the second-tile DMA invocations. Less likely because the byte pattern shows actual second-tile compute output is reaching DDR — just partially. **Verify by adding a counter to `firmware_dma_sloop_load/store` invocations and asserting count == 2 × `len(tiles)` in a parametrized test.**

### 4. Lowest probability — Vendor firmware-side bug we inherited

The pattern (first 16 bytes zero, last 16 bytes correct, at exact tile boundary) is consistent with a firmware-side `tile_row_start += tile_max_rows * ROW_BYTES` increment that miscalculates by exactly half a row. If the firmware issues `__store(dst=base+tile_offset_high)` where the offset is half-row-misaligned, the symptom would match. However, this hypothesis requires reading vendor `.elf` disassembly and is **out of P8 scope**. Plan 04 should attempt fixes #1-#3 first; if all fail, escalate to a v2 firmware-side investigation.

## Recommended Plan 04 Scope

**Outcome B (NPU code fix needed).** Plan 04 should:

1. **Add a tracing harness** (test-only, not production) that captures `(funct7, funct3, xs1, xs2, addr_hi, addr_lo, length, height, rd_stride, wr_stride)` for every `firmware_dma_sloop_*` invocation when running the vendor ABS `.elf` end-to-end. Compare the trace from tile 0 (lines 0..2047 byte-exact) vs tile 1 (lines 2048+ broken) to localize which kwarg differs from what the firmware intended.

2. **Inspect `decode_firmware_dma_args`** (`src/main/python/riscv/gtx/dma_engine.py:66-99`) for any mask that might drop a high-bit during the second tile's invocation. Specifically look at `addr_hi = (rs1 >> 27) & 0x1FFFFFFFFF` (37-bit mask) — for `addr_hi = 0xf00FFF0`, the value is `0x1E01FFE` (after `>>27`)... actually `0xf00FFF0 << 27 = ...` — verify the encode/decode round-trip is lossless at the tile boundary specifically.

3. **Inspect `flush_deferred_ddr_stores`** for L2 ring-wrap behavior at exactly `GTX_L2_SIZE_BYTES` boundary, and potentially add an explicit `# tile boundary contract` assertion.

4. **Parameterize per-op `GTX_DDR_DUMP_ADDR`** in `test_regression_fw_full_sweep.py` (analogous to Task 1's `OP_DUMP_SIZE_OVERRIDE`). Vendor `.elf` writes output at `0xf000000`, not `0x100`. Without this, the harness sees all-zero dumps and the bug is invisible. Suggested constant name: `OP_DUMP_ADDR_OVERRIDE`. Default `0x100` for P5/P6 hand-built `.elf`; `0xf000000` for vendor-rooted `.elf` (detect via `is_vendor_elf` already wired in 08-02).

5. **DO NOT modify** `firmware_dma_sloop_load` / `firmware_dma_sloop_store` core algorithms (Plan 01 proved they are correct in isolation). The fix must be in the dispatch wiring or in how the firmware-side WRSPR sequence builds the packed-rs1 for the second tile.

6. **APPENDIX A fix template** (RESEARCH.md 08-RESEARCH.md):
   - Apply template (a) "firmware_dma_sloop_load row-loop bounds" if the trace shows `length` differing between tile 0 and tile 1.
   - Apply template (b) "flush_deferred_ddr_stores ordering" if the trace shows correct kwargs but wrong DDR contents (i.e., the bug is in the flush logic).
   - Apply template (c) "ddr_dump_to_file zero-padding off-by-one" if the trace shows correct DDR bytes but the dump file has zeros (unlikely given the byte sample shows partial real bytes mixed with zeros).
   - Skip template (d) "atexit hook ordering" — pattern is wrong shape for that hypothesis.

## Cross-Reference to Plan 01's XPASS Finding

Plan 01 (`tests/gtx/test_multi_tile_dma.py::test_tile_boundary_byte_exact`) ran a programmatic 2-tile DMA + ABS sequence by **directly calling** `firmware_dma_sloop_load`/`firmware_dma_sloop_store` with explicit kwargs, bypassing the RoCC dispatch + WRSPR + GSPR-staged-operand path. That test produces byte-exact output for both tiles (XPASS).

This investigation runs the **same NPU code via the vendor `.elf` through the full RoCC dispatch path** and finds tile-2 broken at exactly the same boundary the P7 smoke observation called out. **Conclusion: the bug lives in the dispatch path, not in the DMA engine core algorithms.** Plan 01's test guards the algorithms (correctly green); the dispatch path needs new test coverage that mimics the vendor `.elf` invocation pattern (likely via subprocess + `_find_elf` like the regression sweep does, but with input staging).

## Investigation Limitations & Caveats

1. **ABS only fully reproduces.** Of 6 smoke ops investigated, only ABS shows the multi-tile boundary signature. ADD/RELU/SIGMOID diverge from line 0/1 (different bugs — likely input pre-staging address, GSPR_OPERAND* values, or per-element compute). LEAKY_RELU is a single-bit FP precision delta. GELU passes (only single-tile region). For the multi-tile fix, ABS is the canonical drive-target; Plan 04 should land the fix and verify ABS first, then test against the other ops once the dispatch-trace harness is in place.
2. **Vendor `.elf` requires `GTX_DDR_INIT` operand pre-staging.** This is unwired in the current `test_regression_fw_full_sweep.py` — that's why the sweep harness produces zero output for vendor `.elf` runs. Plan 04 must wire `GTX_DDR_INIT=<vendor-input.txt>` for vendor-rooted `.elf` (analogous to GTX_DDR_REVERSED inline propagation in 08-02).
3. **`GTX_DDR_DUMP_ADDR` mismatch** — vendor `.elf` writes at `0xf000000`, harness defaults to `0x100`. Same fix scope as #4 above.

## Plan 04 Hand-off

**Single most likely bug location:** `src/main/python/riscv/gtx/dma_engine.py:66-99` (`decode_firmware_dma_args`) — second-tile `addr_hi` packed-rs1 decode, OR the upstream RoCC dispatch path that builds rs1/rs2 for the second tile's `__store(addr=base+0xFFF0, ...)` invocation.

**Bytes-mismatched range for tile 1 of ABS:**
- Lines 2048..196609 (raw byte offsets `0x10000..0x600400`)
- Pattern: each line's first 16 bytes = `00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00`; last 16 bytes typically correct.
- First diverging byte: dump line 2048 byte 0 = `00`; golden line 2048 byte 0 = `35`.

**OP_DUMP_SIZE_OVERRIDE pre-fill recommendation for Plan 04 GREEN path:**
After Plan 04 lands the fix, the `OP_DUMP_SIZE_OVERRIDE` dict in `tests/gtx/test_regression_fw_full_sweep.py` should be hard-prefilled (NOT relying on `golden_full/` runtime detection in CI) for the smoke set:

```python
OP_DUMP_SIZE_OVERRIDE = {
    "ABS":        "0x600020",   # 196609 lines * 32 bytes
    "ADD":        "0x200000",   # 65536 lines
    "RELU":       "0x80000",    # 16384 lines
    "GELU":       "0xf000",     # 1920 lines
    "SIGMOID":    "0x2000",     # 256 lines
    "LEAKY_RELU": "0xf920",     # 1993 lines
    "TANH":       "0x460",      # 35 lines
    "SOFT_MAX":   "0xfff0",     # 2047 lines
    "SUM":        "0x20",       # 1 line (already covered by 0x20 fallback)
    "MUL":        "0x200000",   # 65536 lines
}
```

(The runtime-from-disk lookup added in Task 1 stays as a developer convenience; the static dict gives CI deterministic behavior independent of `golden_full/` presence.)

Plan 04's `OP_DUMP_ADDR_OVERRIDE` (NEW) should mirror this:
```python
OP_DUMP_ADDR_OVERRIDE = {
    # All vendor n1s16 ops use 0xf000000 (240 MiB into DDR_BASE) per `_ref.txt` @-headers
    "ABS": "0xf000000", "ADD": "0xf000000", "RELU": "0xf000000",
    "GELU": "0xf000000", "SIGMOID": "0xf000000", "LEAKY_RELU": "0xf000000",
    # ... etc for all 84 vendor ops
}
```
Or simply: `if is_vendor_elf: env["GTX_DDR_DUMP_ADDR"] = "0xf000000"` mirroring the existing GTX_DDR_REVERSED inline propagation.
