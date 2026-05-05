---
phase: 03-dma-ddr-i-o
verified: 2026-05-05T15:07:02Z
status: passed
score: 5/5 must-haves verified
re_verification:
  is_re_verification: false
test_results:
  full_suite: 179 passed in 3.73s
  expected: 179/179
  result: match
requirements_satisfied:
  - DMA-01
  - DMA-02
  - DMA-03
  - DMA-04
  - DMA-05
  - DISP-03
gaps: []
human_verification: []
---

# Phase 3: DMA & DDR I/O Verification Report

**Phase Goal:** Bytes can flow DDR ↔ L2 ↔ L1 ↔ L0 with bit-exact preservation in both `GTX_DDR_REVERSED` modes, with deferred-store semantics matching C++ — enabling all subsequent compute phases to load operands and dump results without a separate fix.

**Verified:** 2026-05-05T15:07:02Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Phase 3 Success Criteria 1–5)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pytest tests/gtx/test_dma_roundtrip.py` passes — write FP16 pattern to `mem.l1_f16(0,0)[0:4096]`, push to L2 via `exec_dma_2d`, push to DDR via `firmware_dma_sloop_store` + flush, dump, reload via `ddr_init_from_file`, run reverse path, byte-exact match | VERIFIED | 3/3 tests pass; full forward + reverse path coded at `tests/gtx/test_dma_roundtrip.py:65-129` (LTR), `:132-174` (REVERSED), `:177-196` (L1→L1 ancillary). Final assertion uses `np.array_equal(...view(np.uint16))` against original `np.arange(4096, dtype=np.float16)` pattern |
| 2 | `pytest tests/gtx/test_ddr_modes.py` passes — same pattern dumped via `ddr_dump_to_file` produces different hex bytes under default LTR vs `GTX_DDR_REVERSED=1` (32-byte bus-word reversal verified) AND each mode round-trips through its own init | VERIFIED | 17/17 tests pass; `test_ddr_dump_modes_differ_and_invert` (line 136) explicitly asserts `ltr_first != rev_first` AND that REV first line is byte-reverse of LTR first line. `test_ddr_round_trip_ltr` + `test_ddr_round_trip_reversed` close the dump→init mode-symmetric round-trip |
| 3 | `firmware_dma_op` decoded for synthetic LOAD with HW conventions `length=0` (decoded as 65536) and `height=0` (decoded as 1) produces same source/destination as `gtx_npu_dma.cc:firmware_dma` | VERIFIED | `decode_firmware_dma_args` at `dma_engine.py:79-99` implements `height = 1 if height_raw == 0 else height_raw` and `length = 0x10000 if length_raw == 0 else length_raw` (L89-90). Tested at `test_dma_engine.py:165` (`test_decode_length_zero_means_65536`) + `:173` (`test_decode_height_zero_means_one`). End-to-end coverage at `test_firmware_dma.py:250` (`test_firmware_dma_length_zero_means_65536_e2e`) — passes captured kwarg `length == 0x10000`, `height == 1`. Pitfall 1 (is_copy carve-out: `addr_hi = (rs1>>32) if is_copy else ((rs1>>27)&0x1FFFFFFFFF)` at `dma_engine.py:82`) covered at `test_firmware_dma.py:test_firmware_dma_copy_tloop_uses_high_32_bit_dst` |
| 4 | S-loop deferred-store: `start_p → start_s → exec_dma_2d(STORE) → end_s → exec_dma_2d(STORE) → end_p` flushes both stores in order at `end_p`; pre-flush DDR is unchanged | VERIFIED | Two-site flush wiring confirmed: (a) `ops/control.py:_do_endp:75-77` calls `npu.flush_deferred_ddr_stores()` when `not npu.warp.wsplit_seen`; (b) `ops/dma.py:_credit_st_chk:322-324` calls flush when `npu.warp.is_sloop`; (c) `dispatch_4mode.py:dispatch_iss_opcode:59-61` calls flush when `funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop` (third call site, validated by `test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop`). Pre-flush vs post-flush divergence asserted at `test_dma_roundtrip.py:96-102` (`pre_flush == bytes(8192)` zeros, `post_flush == l2_bytes`, `pre_flush != post_flush`). Dual-trigger semantics tested at `test_deferred_store.py:142,165,189,201,215,255,302` (10/10 pass) |
| 5 | Mode 1 (no loop, broadcast 64) and Mode 3 (P+S, single NEST DMA) routing in `_dispatch` selects same `(nest_id, spu_id)` set as `gtx_npu_dispatch.cc` for synthesized firmware traces | VERIFIED | `dispatch_4mode.py:91-97` (Mode 1: `for n in range(GTX_NEST_NUM): for s in range(GTX_SPU_NUM): dispatch_iss_opcode(...)` = 4*16 = 64 calls); `:103-115` (Mode 3: `is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)`, `dma_engine.exec_dma_2d(...)` single call). Parameterized routing test at `test_dispatch_4mode.py:67-99` covers Mode 1 (count=64), Mode 2 (count=16), Mode 4 (count=1). Mode 3 OR-rule covered by 3 tests at lines 103, 136, 159; Mode 3 routing-exclusivity at line 178 (`test_dispatch_4mode_mode3_does_not_call_iss_opcode`) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | LOC | Wired | Status |
|----------|----------|-----|-------|--------|
| `src/main/python/riscv/gtx/dma_engine.py` | DeferredDdrStore + decoder + 6 exec_* + 4 firmware_dma_* helpers | 372 | imported by ops/dma.py, dispatch_4mode.py, test files | VERIFIED |
| `src/main/python/riscv/gtx/ops/dma.py` | 16 @handler entries + credit_st_chk flush trigger | 324 | imported by ops/__init__.py → triggers @handler registration | VERIFIED |
| `src/main/python/riscv/gtx/ops/control.py` | end_p flush trigger when !wsplit_seen + WSPLIT setters | 248 | imported by ops/__init__.py | VERIFIED |
| `src/main/python/riscv/gtx/ddr.py` | doubling-grow ensure_ddr + ddr_init_from_file + ddr_dump_to_file | 169 | imported by dma_engine.py, npu.py, test_ddr_modes.py | VERIFIED |
| `src/main/python/riscv/gtx/dispatch_4mode.py` | dispatch_4mode + dispatch_iss_opcode (third flush site) | 121 | imported via dispatch.py re-export (line 55) | VERIFIED |
| `src/main/python/riscv/gtx/npu.py` | GtxNpu.deferred_ddr_stores + flush_deferred_ddr_stores | 177 | concrete consumer of all DMA paths | VERIFIED |
| `src/main/python/riscv/gtx/warp_state.py` | WarpState.wsplit_seen process-lifetime sentinel | 41 | wsplit_seen NOT cleared by reset() (line 41 comment + assertion) | VERIFIED |
| `src/main/python/riscv/gtx/encoding.py` | GTX_ISS_F7_DMA_* + GSPR_GTX_OPERAND* + LSPR_SPM_* AUTHORITATIVE constants | 92 | imported by ops/dma.py, dispatch_4mode.py | VERIFIED |
| `src/main/python/riscv/gtx/params.py` | GTX_DDR_BASE = 0x370000000 | 46 | imported by dma_engine.py, ddr.py | VERIFIED |
| `tests/gtx/test_dma_engine.py` | 27 unit tests for engine helpers | 454 | runnable | VERIFIED |
| `tests/gtx/test_firmware_dma.py` | 15 firmware_dma decoder + dispatch tests | 470 | runnable | VERIFIED |
| `tests/gtx/test_deferred_store.py` | 11 dual-trigger flush tests | 317 | runnable | VERIFIED |
| `tests/gtx/test_ddr_modes.py` | 17 LTR/REV + ensure_ddr + round-trip tests | 291 | runnable | VERIFIED |
| `tests/gtx/test_dma_roundtrip.py` | 3 integration tests | 196 | runnable | VERIFIED |
| `tests/gtx/test_dispatch_4mode.py` | 13 routing tests | 257 | runnable | VERIFIED |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `ops/dma.py @handler` | `dma_engine.firmware_dma_sloop_store/load` | direct call (lines 86-90, 117-121) | WIRED | dma_engine functions invoked from is_sloop branches; rs3 read from `npu.gspr.get(GSPR_GTX_OPERAND3, 0)` |
| `ops/control.py:_do_endp` | `npu.flush_deferred_ddr_stores()` | `if not npu.warp.wsplit_seen` guard (line 76) | WIRED | Site #1 of 2 flush triggers; suppresses when wsplit_seen=True |
| `ops/dma.py:_credit_st_chk` | `npu.flush_deferred_ddr_stores()` | `if npu.warp.is_sloop` guard (line 322) | WIRED | Site #2a of flush triggers (custom0 funct7=0x53 entry path) |
| `dispatch_4mode.py:dispatch_iss_opcode` | `npu.flush_deferred_ddr_stores()` | `if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop` (line 59) | WIRED | Site #2b of flush triggers (dispatch_4mode entry path; cooperative mirror of #2a) |
| `dispatch_4mode.py:dispatch_4mode` (Mode 3) | `dma_engine.exec_dma_2d` | direct call (line 107) | WIRED | Single-NEST DMA path; bypasses dispatch_iss_opcode (verified by exclusion test) |
| `npu.py:GtxNpu.flush_deferred_ddr_stores` | `mem._ddr_bytes` write + `ensure_ddr` | per-row loop (lines 158-170) | WIRED | Drains queue with byte-exact L2→DDR copy |
| `dispatch.py` | `dispatch_4mode.dispatch_4mode` | re-export `from .dispatch_4mode import` (line 55) | WIRED | Public surface preserved |
| `ddr.py:ddr_dump_to_file` | `mem._ddr_bytes` + `chunk[::-1]` | `chunk[::-1]` when `os.environ.get("GTX_DDR_REVERSED")` (line 168) | WIRED | Per-call env read (D-08); 32-byte bus-word reversal |
| `ddr.py:ddr_init_from_file` | `mem._ddr_bytes` + `chunk[::-1]` | `chunk[::-1]` when reversed_mode (line 126) | WIRED | Symmetric reverse for round-trip cancellation |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| `dma_engine.exec_dma_2d` | `l1_buf`, `l2_buf` | `mem.l1_byte()`, `mem.l2_byte()` ndarrays | Yes — slice copy with bounds-clamped `copy_len` | FLOWING |
| `firmware_dma_sloop_store` | `npu.deferred_ddr_stores` | `npu` instance attribute (npu.py:44) | Yes — DeferredDdrStore appended; flush drains via `mem._ddr_bytes` write | FLOWING |
| `firmware_dma_sloop_load` | `mem._ddr_bytes`, `l2_buf` | DDR via `ensure_ddr`, then `l2_buf[off:off+len] = ddr[off:off+len]` | Yes — actual DDR-to-L2 byte copy with row stride | FLOWING |
| `ddr_dump_to_file` | `chunk` | `mem._ddr_bytes[chunk_off+j]` per byte (line 164) | Yes — reads actual ndarray values; zero-pad on out-of-range matches C++ | FLOWING |
| `ddr_init_from_file` | `mem._ddr_bytes[offset:offset+nbytes]` | `bytes.fromhex(line[:nbytes*2])` + reverse | Yes — file-content-driven byte assignment, `ensure_ddr` grows backing | FLOWING |
| `flush_deferred_ddr_stores` | `mem._ddr_bytes[ddr_off:ddr_off+copy_len]` | `mem.l2_byte(req.nest)[l2_off:l2_off+copy_len]` | Yes — byte-exact L2-to-DDR per-row copy from queue | FLOWING |
| `dispatch_4mode` Mode 3 | `dma_engine.exec_dma_2d` return | `npu.warp` flags + op1/op2/op3 from caller | Yes — `width = op3 & 0xFFFF`, `height = (op3 >> 16) & 0xFFFF`, real DMA invoked | FLOWING |

All artifacts verified at Level 4 — data flows from source (memory ndarrays / hex files / RoCC operands) through wiring to actual byte mutations. No HOLLOW state.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full Phase 3 test suite green | `python3 -m pytest tests/gtx/ --noconftest -o "addopts=" -q` | `179 passed in 3.73s` | PASS |
| Round-trip integration tests pass | `pytest tests/gtx/test_dma_roundtrip.py -v` | 3/3 pass (LTR + REVERSED + L1→L1) | PASS |
| DDR mode tests pass (LTR ≠ REV verified) | `pytest tests/gtx/test_ddr_modes.py -v` | 17/17 pass | PASS |
| Deferred-store dual-trigger tests pass | `pytest tests/gtx/test_deferred_store.py -v` | 11/11 pass | PASS |
| 4-mode dispatch routing tests pass | `pytest tests/gtx/test_dispatch_4mode.py -v` | 13/13 pass | PASS |
| Module imports resolve cleanly | `python3 -c "from riscv.gtx import dma_engine; from riscv.gtx.ddr import ddr_dump_to_file, ddr_init_from_file, ensure_ddr; from riscv.gtx.dispatch import dispatch_4mode"` | success (no error during pytest collection) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DMA-01 | 03-01-dma-engine | `exec_dma_2d`, `exec_load_svr`, `exec_store_svr`, `exec_transpose`, `exec_fill` 전체 ops 구현 | SATISFIED | All 6 helpers present in `dma_engine.py` (exec_dma_2d:105, exec_load_svr:142, exec_store_svr:161, exec_transpose:180, exec_transpose_ddr:204, exec_fill:252); 27 test cases at `test_dma_engine.py` |
| DMA-02 | 03-02-ops-dma | `firmware_dma_op` 패킹 인코딩 디코딩 — funct3 = 000(LOAD) / 001(STORE) / 010(COPY) | SATISFIED | `decode_firmware_dma_args` at `dma_engine.py:66-99` with HW conventions + is_copy carve-out; 16 @handler entries in `ops/dma.py`; 15 tests at `test_firmware_dma.py` |
| DMA-03 | 03-05-flush-roundtrip | S-loop L2→DDR 스토어 deferred-store 큐, `endp`에서 일괄 flush | SATISFIED | `npu.deferred_ddr_stores` queue + `flush_deferred_ddr_stores()` (npu.py:44, 146); end_p flush wired (control.py:75-77 via wsplit_seen guard); credit_st_chk dual flush at 2 sites (ops/dma.py:322, dispatch_4mode.py:59); 11 tests at `test_deferred_store.py` |
| DMA-04 | 03-03-ddr-io | DDR hex I/O 두 모드 (`ddr_init_from_file`, `ddr_dump_to_file`) — LTR + `GTX_DDR_REVERSED=1` 모두 동작 | SATISFIED | `ddr.py:97-131` (init), `:134-169` (dump); per-call env read (`os.environ.get("GTX_DDR_REVERSED")` at lines 110, 145); `chunk[::-1]` reversal at lines 126, 168; 17 tests at `test_ddr_modes.py` (incl. `test_ddr_dump_modes_differ_and_invert`, both round-trips) |
| DMA-05 | 03-05-flush-roundtrip | DMA 라운드트립 — L1→DDR→reload→bit-exact 일치 | SATISFIED | 3 integration tests at `test_dma_roundtrip.py`; both LTR + REVERSED close the L1→L2→DDR→file→re-init→L2→L1 cycle with `np.array_equal(final.view(uint16), pattern.view(uint16))` |
| DISP-03 | 03-04-dispatch-4mode | 4-mode dispatch router (Mode 1/2/3/4) NEST/SPU 라우팅 | SATISFIED | `dispatch_4mode.py:69-121`; Mode 1 (4×16=64 broadcast at line 94-97), Mode 2 (16 within tmu_id at line 100-101), Mode 3 (single NEST `exec_dma_2d` at line 103-115 with Pitfall 8 OR-rule), Mode 4 (single tmu_id+curr_id at line 116-120); 13 tests at `test_dispatch_4mode.py` |

**REQUIREMENTS.md cross-check:** All 6 IDs (DMA-01..05, DISP-03) marked `Phase 3 | Complete` at REQUIREMENTS.md:206-211. No orphaned requirements — every plan claims its assigned IDs exactly:
- 01-PLAN: `[DMA-01]`
- 02-PLAN: `[DMA-02]`
- 03-PLAN: `[DMA-04]`
- 04-PLAN: `[DISP-03]`
- 05-PLAN: `[DMA-03, DMA-05]`

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

Zero TODO/FIXME/PLACEHOLDER markers in any of the 8 production source files modified by Phase 3. Single `pass` statement at `ddr.py:152` is the empty-DDR short-circuit branch (matches C++ `has_ddr()` check) — intentional, not a stub. All P3-deferred items (V2 mcast/3D stubs in `ops/dma.py` lines 267-306) are explicitly registered with `v2 deferral` docstrings + return 0 NOPs for disasm parity, matching the Plan 02 design decision. These are out-of-scope for v1 per REQUIREMENTS.md `DMA-V2-01` and not blockers.

### Human Verification Required

None. All Phase 3 acceptance criteria have automated verification (per `03-VALIDATION.md` "Manual-Only Verifications: none"). The first `.elf` strict-mode regression is deferred to Phase 4 success #4 (per CONTEXT D-10) and is not a Phase 3 gate.

### Gaps Summary

No gaps. Phase 3 goal achieved:

1. Bytes flow DDR ↔ L2 ↔ L1 ↔ L0 — verified end-to-end by `test_dma_l1_to_ddr_roundtrip_ltr` (forward path: L1→L2→DDR→file; reverse path: file→DDR→L2→L1; final `np.array_equal` against original pattern).
2. Bit-exact preservation in both `GTX_DDR_REVERSED` modes — verified by `test_dma_l1_to_ddr_roundtrip_reversed` AND `test_ddr_dump_modes_differ_and_invert` (LTR vs REV produce different hex, each mode round-trips through its own init).
3. Deferred-store semantics match C++ — three-call-site flush wiring (end_p when !wsplit_seen + credit_st_chk via custom0 OR via dispatch_iss_opcode when is_sloop) verified by 11 dual-trigger tests; `wsplit_seen` is process-lifetime sentinel NOT cleared by reset (Pitfall 7 — verified at `test_reset_clears_deferred_queue_but_not_wsplit_seen`).
4. Subsequent compute phases enabled — DMA primitives + dispatch_iss_opcode extension point are in place. Phase 4 (MM) inherits a fully working data plane; the dispatch_iss_opcode body has explicit comment markers for funct7=GTX_OP_MM filling.

### Notable Risks for Downstream Phases

While Phase 3 itself is goal-complete, the following observations should inform Phase 4 planning:

1. **Phase 4 .elf strict-mode regression is the next gate.** Per ROADMAP.md Phase 4 success #4, "First .elf GEMM regression passes strict mode" with `verify.py --fp16 --ulp 1 --atol 0.001 --strict` against C++ golden. This is the first end-to-end exercise of Phase 3 plumbing through real firmware. Risk: any subtle off-by-one in DMA stride math (currently bounded by `min(length, GTX_*_SIZE_BYTES - off)` clamp at `dma_engine.py:127-129, :307-309, :339-341`) won't surface until P4 integration. **Mitigation:** P3 unit tests cover the canonical path; the new bug class (DMA bug exposed only by interleaving mm + dma) is what P4 will catch.

2. **`ensure_ddr` divergence from C++ is documented but real.** P3 uses doubling-grow with INITIAL_FLOOR=1MiB (D-13) for CI ergonomics; C++ uses single-shot 4 GiB allocation (`gtx_npu_core.cc:198-203`). For firmware that touches the full 4 GiB the behavior collapses to single-grow-to-cap (semantically identical), but if a firmware mid-workload does sparse DDR writes far apart, doubling-grow may allocate more memory than the C++ path. **Mitigation:** documented in `ddr.py:62-72` docstring; not a correctness risk since DDR contents above current_size are zero-initialized in both paths.

3. **`dispatch_iss_opcode` is a true stub for non-DMA funct7.** P3 NOPs every funct7 except CREDIT_ST_CHK. Phase 4 must fill `GTX_OP_MM=0` branch with the four MM variants. The insertion point is comment-marked at `dispatch_4mode.py:62-65`. Risk: if Phase 4 inadvertently touches the existing CREDIT_ST_CHK branch, deferred-store semantics could regress silently. **Mitigation:** P3 test `test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop` will catch any regression at runtime.

4. **`mxe_accum` 2D state is initialized but never written by P3.** `npu.py:55-58` allocates `(GTX_NEST_NUM, GTX_SPU_NUM)` FP32 ndarray; reset clears it. Phase 4 mm/mmc handlers will be the first writers. Risk: P4 must respect `is_accumulate` flag to chain `mm.s → mmc.s → mmc` correctly per MM-04. P3 cannot pre-validate this. **Mitigation:** noted in `dispatch_4mode.py` extension point; P4 plan-phase should research C++ `mxe_accum` write paths early.

5. **`dispatch_4mode` Mode 4 is wired but untested in firmware context.** Mode 4 (P+T, single tmu_id+curr_id) has unit-test coverage at `test_dispatch_4mode_routing_count` (count=1) but never actually exercises the inner `dispatch_iss_opcode` payload — that payload is a NOP in P3. P4 mm fillers will be the first true Mode 4 exercise. Risk: low — routing logic is straightforward.

---

## Verification Summary

**5/5 truths verified.**
**6/6 requirements satisfied** (DMA-01, DMA-02, DMA-03, DMA-04, DMA-05, DISP-03).
**179/179 tests passing** in `tests/gtx/` full suite (target: 179).
**0 anti-patterns** (no TODO/FIXME/PLACEHOLDER in modified production files).
**0 gaps. 0 human-verification items.**

Phase 3 (DMA & DDR I/O) is goal-complete. The data plane required by all subsequent compute phases (P4 MM, P5 VEC/ACT) is in place and exercised end-to-end by integration tests. Status: **passed**. Ready to proceed to Phase 4.

---

*Verified: 2026-05-05T15:07:02Z*
*Verifier: Claude (gsd-verifier)*
