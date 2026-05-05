---
phase: 03-dma-ddr-i-o
plan: 05
type: execute
wave: 3
depends_on: [03-01, 03-02, 03-03, 03-04]
files_modified:
  - src/main/python/riscv/gtx/ops/control.py
  - src/main/python/riscv/gtx/ops/dma.py
  - src/main/python/riscv/gtx/dispatch_4mode.py
  - tests/gtx/test_deferred_store.py
  - tests/gtx/test_dma_roundtrip.py
  - .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md
autonomous: true
requirements: [DMA-03, DMA-05]

must_haves:
  truths:
    - "WSPLIT handler (custom1 funct3=0b100 AND custom0 funct7=0x02) sets npu.warp.wsplit_seen = True."
    - "end_p handler in ops/control.py calls npu.flush_deferred_ddr_stores() ONLY when !npu.warp.wsplit_seen."
    - "credit_st_chk handler in ops/dma.py calls npu.flush_deferred_ddr_stores() ONLY when npu.warp.is_sloop."
    - "dispatch_iss_opcode in dispatch_4mode.py calls npu.flush_deferred_ddr_stores() when funct7 == GTX_ISS_F7_CREDIT_ST_CHK and is_sloop. Plan 04 split this function into its own module; Plan 05 updates the body there."
    - "Both flush triggers are wired (RESEARCH lock-in: the firmware authoring style determines which fires)."
    - "L1 → L2 → DDR → re-init → L2 → L1 round-trip produces byte-exact match against the original FP16 pattern."
    - "Pre-flush DDR snapshot differs from post-flush DDR (deferred queue is a no-op on DDR until flush)."
    - "Plans 03-01 + 03-02 + 03-03 + 03-04 are integrated end-to-end via this plan's round-trip test."
    - "VALIDATION.md frontmatter is flipped to nyquist_compliant=true and wave_0_complete=true at the end of Task 2 once all earlier sign-off conditions are met."
  artifacts:
    - path: "src/main/python/riscv/gtx/ops/control.py"
      provides: "wsplit_seen=True wired in 2 places (custom0 0x02 + custom1 0b100); flush in end_p when !wsplit_seen"
      contains: "npu.warp.wsplit_seen = True"
      contains_2: "if not npu.warp.wsplit_seen"
      contains_3: "npu.flush_deferred_ddr_stores()"
    - path: "src/main/python/riscv/gtx/ops/dma.py"
      provides: "credit_st_chk body wired with is_sloop check"
      contains: "if npu.warp.is_sloop"
    - path: "src/main/python/riscv/gtx/dispatch_4mode.py"
      provides: "dispatch_iss_opcode body wired for credit_st_chk (Plan 04 split landed this in its own module)"
      contains: "GTX_ISS_F7_CREDIT_ST_CHK"
    - path: "tests/gtx/test_deferred_store.py"
      provides: "DMA-03 tests: dual-assertion (queue + flush), end_p trigger when !wsplit_seen, credit_st_chk trigger when is_sloop, wsplit_seen suppresses end_p flush"
      min_lines: 200
    - path: "tests/gtx/test_dma_roundtrip.py"
      provides: "DMA-05 integration test: full L1→L2→DDR→file→re-init→L2→L1 round-trip bit-exact"
      min_lines: 100
    - path: ".planning/phases/03-dma-ddr-i-o/03-VALIDATION.md"
      provides: "Flipped frontmatter flags + Approval status at the very end of phase 3 work"
      contains: "nyquist_compliant: true"
      contains_2: "wave_0_complete: true"
  key_links:
    - from: "ops/control.py wsplit + wsplit_custom0"
      to: "npu.warp.wsplit_seen = True"
      via: "first-line side effect in both handlers"
      pattern: "npu\\.warp\\.wsplit_seen = True"
    - from: "ops/control.py endp"
      to: "npu.flush_deferred_ddr_stores()"
      via: "if not npu.warp.wsplit_seen branch"
      pattern: "if not npu\\.warp\\.wsplit_seen"
    - from: "ops/dma.py _credit_st_chk"
      to: "npu.flush_deferred_ddr_stores()"
      via: "if npu.warp.is_sloop branch"
      pattern: "if npu\\.warp\\.is_sloop"
    - from: "dispatch_4mode.py dispatch_iss_opcode"
      to: "npu.flush_deferred_ddr_stores()"
      via: "credit_st_chk funct7 check + is_sloop guard"
      pattern: "GTX_ISS_F7_CREDIT_ST_CHK and npu\\.warp\\.is_sloop"
---

<objective>
Wire the deferred-store flush triggers at TWO sites per RESEARCH lock-in
(03-RESEARCH "Deferred-Store Flush Trigger"):

1. **`end_p` (ops/control.py) when `!wsplit_seen`** — simple firmware (no
   WSPLIT/WJOIN) that ends the P-loop flushes here. ROADMAP P3 success #4 path.
2. **`credit_st_chk` (ops/dma.py + dispatch_4mode.py) when `is_sloop`** — plan-style
   firmware (WSPLIT-enabled) flushes mid-execution between S-loop iterations.
   P4 mm_basic.elf path.

Set `npu.warp.wsplit_seen = True` in both WSPLIT handlers (custom1 funct3=0b100
+ custom0 funct7=0x02). Add the integration round-trip test that validates the
full L1↔L2↔DDR chain end-to-end (DMA-05). Finally, flip VALIDATION.md
frontmatter flags now that all sign-off conditions are met.

Purpose: This is the integration plan. It depends on Plan 01 (DeferredDdrStore +
firmware_dma_sloop_store + flush_deferred_ddr_stores body), Plan 02 (npu.deferred_ddr_stores
attribute + ops/dma.py credit_st_chk handler stub), Plan 03 (ddr_dump_to_file
+ ddr_init_from_file), and Plan 04 (dispatch_4mode.py dispatch_iss_opcode stub
in its OWN module). It closes DMA-03 and DMA-05.

Output: 3 source files modified to wire flush triggers; 2 tests populated;
VALIDATION.md flags flipped to ready.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md
@.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md
@.planning/phases/03-dma-ddr-i-o/03-VALIDATION.md
@.planning/phases/03-dma-ddr-i-o/03-01-SUMMARY.md
@.planning/phases/03-dma-ddr-i-o/03-02-SUMMARY.md
@.planning/phases/03-dma-ddr-i-o/03-03-SUMMARY.md
@.planning/phases/03-dma-ddr-i-o/03-04-SUMMARY.md

@src/main/python/riscv/gtx/ops/control.py
@src/main/python/riscv/gtx/ops/dma.py
@src/main/python/riscv/gtx/dispatch.py
@src/main/python/riscv/gtx/dispatch_4mode.py
@src/main/python/riscv/gtx/npu.py
@src/main/python/riscv/gtx/dma_engine.py
@src/main/python/riscv/gtx/ddr.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc
@vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc
@vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc

<interfaces>
From ops/control.py (P2):
```python
@handler(kind='custom1', funct3=0b100, mnemonic='warp_split')
def wsplit(npu, proc, insn, xs1, xs2): ...   # P2: NOP. P3 set wsplit_seen=True.

@handler(kind='custom0', funct7=0x02, mnemonic='wsplit_c0')
def wsplit_custom0(npu, proc, insn, xs1, xs2): ...   # P2: NOP. P3 set wsplit_seen=True.

@handler(kind='custom1', funct3=0b111, mnemonic='warp_end_p')
def endp(npu, proc, insn, xs1, xs2):
    state = proc.get_state()
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    _do_endp(npu, rs1_val, rs2_val)
    return 0
# P3 wires _do_endp OR endp itself to flush.
```

From npu.py (Plan 02):
```python
def flush_deferred_ddr_stores(self) -> None: ...   # body present, called externally
self.deferred_ddr_stores: list = []
```

From ops/dma.py (Plan 02 Task 2b stub):
```python
@handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST_CHK, mnemonic='credit_st_chk')
def _credit_st_chk(npu, proc, insn, xs1, xs2):
    """Plan 05: triggers npu.flush_deferred_ddr_stores() when is_sloop."""
    return 0   # P3-Plan-05 fills the body.
```

From dispatch_4mode.py (Plan 04 — NEW module, not dispatch.py):
```python
def dispatch_iss_opcode(npu, nest_id, spu_id, funct7, op1, op2, op3) -> int:
    # P3 stub. Plan 05 wires credit_st_chk insertion.
```

From dma_engine.py (Plan 01):
```python
def firmware_dma_sloop_store(npu, *, nest, addr_hi, addr_lo, length, height, rd_stride, wr_stride) -> int
def firmware_dma_sloop_load(mem, *, nest, addr_hi, addr_lo, length, height, rd_stride, wr_stride) -> int
def exec_dma_2d(mem, *, nest_id, l2_addr, l1_addr, width, height, is_load, l2_stride=0, spu_id=0) -> int
```

From ddr.py (Plan 03):
```python
def ddr_init_from_file(mem, filename) -> None
def ddr_dump_to_file(mem, filename, addr, size) -> None
def ensure_ddr(mem, end_offset) -> np.ndarray
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire wsplit_seen + end_p flush + credit_st_chk flush + dual-assertion deferred-store tests</name>
  <files>
    src/main/python/riscv/gtx/ops/control.py,
    src/main/python/riscv/gtx/ops/dma.py,
    src/main/python/riscv/gtx/dispatch_4mode.py,
    tests/gtx/test_deferred_store.py
  </files>
  <read_first>
    - src/main/python/riscv/gtx/ops/control.py (existing wsplit, wsplit_custom0, endp, _do_endp)
    - src/main/python/riscv/gtx/ops/dma.py (existing _credit_st_chk stub from Plan 02 Task 2b)
    - src/main/python/riscv/gtx/dispatch_4mode.py (NEW module from Plan 04 with dispatch_iss_opcode stub)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc (line 62 -- `wsplit_seen = true;`)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc (line 76 -- wsplit custom0 variant sets wsplit_seen)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc (lines 52-67 -- endp flush + DDR dump when !wsplit_seen)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc (lines 898-905 -- credit_st_chk in dispatch_iss_opcode when is_sloop)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc (lines 684-694 -- credit_st_chk via custom0 entry path)
    - 03-RESEARCH.md "Deferred-Store Flush Trigger (D-06 Lock-in)" -- full reconciliation matrix
    - 03-RESEARCH.md "Test Patterns" §test_deferred_store.py -- dual-assertion recipe
    - 03-RESEARCH.md "Common Pitfalls" #7 (wsplit_seen reset semantics -- NOT cleared by reset)
  </read_first>
  <behavior>
    - WSPLIT (both forms) sets `npu.warp.wsplit_seen = True`. Both still return 0 (NOP behavior preserved beyond the flag set).
    - end_p handler: after _do_endp clears is_ploop, IF `not npu.warp.wsplit_seen`, call `npu.flush_deferred_ddr_stores()`.
    - credit_st_chk (custom0 funct7=0x53): IF `npu.warp.is_sloop`, call `npu.flush_deferred_ddr_stores()`. Returns 0.
    - dispatch_iss_opcode (in dispatch_4mode.py): when `funct7 == GTX_ISS_F7_CREDIT_ST_CHK` AND `npu.warp.is_sloop`, call `npu.flush_deferred_ddr_stores()`. (Mode 3+ may also reach this -- both trigger paths covered.)
    - **Test test_deferred_store.py** tests:
      * `test_deferred_store_queue_push_shape` -- synthetic firmware_dma STORE (via dma_engine.firmware_dma_sloop_store directly); assert `len(npu.deferred_ddr_stores) == 1`, fields match.
      * `test_deferred_store_flush_diff` -- pre-flush DDR snapshot, flush, post-flush DDR diff matches L2 source bytes.
      * `test_endp_flushes_when_no_wsplit_seen` -- set up S-loop store (push to queue); call endp custom1 handler (not _do_endp directly -- go through the full ops/control.py decorator path); assert flush happened (queue empty + DDR matches L2).
      * `test_endp_does_not_flush_when_wsplit_seen` -- same setup BUT also call wsplit_custom1 first; endp returns without flushing; queue still has the entry.
      * `test_credit_st_chk_flushes_when_is_sloop` -- set is_sloop=True, push deferred store, invoke credit_st_chk handler via custom0; queue is empty post-call; DDR has the bytes.
      * `test_credit_st_chk_no_flush_when_not_sloop` -- push deferred store, set is_sloop=False, invoke credit_st_chk; queue still has entry; DDR untouched.
      * `test_wsplit_custom1_sets_wsplit_seen` -- invoke custom1 funct3=0b100 (warp_split); `npu.warp.wsplit_seen is True`.
      * `test_wsplit_custom0_sets_wsplit_seen` -- invoke custom0 funct7=0x02 (wsplit_c0); `npu.warp.wsplit_seen is True`.
      * `test_dispatch_iss_opcode_credit_st_chk_flushes_when_is_sloop` -- direct call into dispatch_4mode.dispatch_iss_opcode with funct7=0x53; flushes. (Both entry paths work -- dispatch and direct custom0 -- per RESEARCH 3 call sites lock-in.)
      * `test_reset_clears_deferred_queue_but_not_wsplit_seen` -- push to queue, set wsplit_seen=True, call reset; queue empty, wsplit_seen still True (Pitfall 7 e2e -- was set up by Plan 01 Task 1; Plan 05 confirms in integration).
  </behavior>
  <action>
1. Edit `src/main/python/riscv/gtx/ops/control.py`:
   * In `wsplit` (custom1 funct3=0b100): replace body with:
     ```python
     @handler(kind='custom1', funct3=0b100, mnemonic='warp_split')
     def wsplit(npu, proc, insn, xs1, xs2):
         """WSPLIT -- start timing section. P3: sets wsplit_seen sentinel.

         The wsplit_seen flag determines which deferred-store flush trigger
         fires (end_p when !wsplit_seen, credit_st_chk when is_sloop). See
         03-RESEARCH 'Deferred-Store Flush Trigger'.
         """
         npu.warp.wsplit_seen = True
         return 0
     ```
   * In `wsplit_custom0` (custom0 funct7=0x02): same body update -- set `npu.warp.wsplit_seen = True; return 0`.
   * Modify `_do_endp` to add the flush trigger:
     ```python
     def _do_endp(npu, rs1: int, rs2: int) -> None:
         """Port of gtx_npu_t::endp. Clears is_ploop. P3: flushes deferred-store
         queue when !wsplit_seen (RESEARCH 'Deferred-Store Flush Trigger' #1)."""
         npu.warp.is_ploop = False
         # P3: flush deferred S-loop L2->DDR stores at endp for simple firmware
         # (no WSPLIT). Plan-style firmware (with WSPLIT) flushes via
         # credit_st_chk instead -- see ops/dma.py:_credit_st_chk.
         if not npu.warp.wsplit_seen:
             npu.flush_deferred_ddr_stores()
     ```
     (No change needed in `endp` handler itself -- it calls `_do_endp` which now does the flush.)

2. Edit `src/main/python/riscv/gtx/ops/dma.py` `_credit_st_chk` body (replace stub):
   ```python
   @handler(kind='custom0', funct7=GTX_ISS_F7_CREDIT_ST_CHK,
            mnemonic='credit_st_chk')
   def _credit_st_chk(npu, proc, insn, xs1, xs2):
       """Direct port of gtx_npu_dispatch.cc:898-905 + gtx_npu_custom0.cc:684-694.

       Plan-style firmware (uses WSPLIT/WJOIN) flushes deferred S-loop stores
       here mid-execution, after T-loop signals via credit. ROADMAP success #4
       tests the end_p flush; this trigger is for P4 mm_basic.elf.
       """
       if npu.warp.is_sloop:
           npu.flush_deferred_ddr_stores()
       return 0
   ```

3. Edit `src/main/python/riscv/gtx/dispatch_4mode.py` `dispatch_iss_opcode` body — wire the same flush trigger when funct7=0x53. **NOTE: This is the NEW module from Plan 04 (split from dispatch.py to avoid Wave 2 file conflict). The body lives there now, not in dispatch.py.**
   ```python
   def dispatch_iss_opcode(npu, nest_id, spu_id, funct7, op1, op2, op3) -> int:
       """[unchanged docstring]"""
       if nest_id < 0 or nest_id >= GTX_NEST_NUM:
           return 0
       if spu_id < 0 or spu_id >= GTX_SPU_NUM:
           return 0
       # P3: credit_st_chk flush trigger (mirror of ops/dma.py:_credit_st_chk
       # for the dispatch_4mode entry path -- RESEARCH 3 call sites lock-in).
       if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop:
           npu.flush_deferred_ddr_stores()
           return 0
       # P4/P5 will add MM/VEC/ACT cases.
       return 0
   ```

4. Populate `tests/gtx/test_deferred_store.py` per the `<behavior>` block. All tests use `_RISCV_AVAILABLE` self-detect + `pytestmark = pytest.mark.skipif`. Helper:
   ```python
   def _push_deferred_store(npu, *, nest=0, l2_off=100, ddr_off=0x1000,
                              length=64, height=1, l2_stride=64, ddr_stride=64):
       from riscv.gtx.dma_engine import firmware_dma_sloop_store
       firmware_dma_sloop_store(npu, nest=nest, addr_hi=ddr_off, addr_lo=l2_off,
                                  length=length, height=height,
                                  rd_stride=l2_stride, wr_stride=ddr_stride)
   ```
   Each test sets up appropriate warp state (`npu.warp.is_ploop = True` etc.), pre-populates L2 with a known pattern via `npu.mem.l2_byte(0)[100:200] = np.arange(100, dtype=np.uint8)`, then exercises one trigger path.

   For the WSPLIT handler tests, invoke through `npu.custom1(MockProcessor(), MockInsn(...), 0, 0)` and `npu.custom0(MockProcessor(), MockInsn(funct=0x02, ...), 0, 0)`. The handlers run with closure-bound `npu`.

   For the end_p test:
   ```python
   def test_endp_flushes_when_no_wsplit_seen(_make_npu_with_pattern):
       from tests.gtx._mocks import MockProcessor, MockInsn
       npu = _make_npu_with_pattern()  # Pre-populates L2[0][100:200] = arange(100)
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       npu.warp.tmu_id = 0
       npu.warp.wsplit_seen = False
       _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x1000, length=100)

       assert len(npu.deferred_ddr_stores) == 1
       proc = MockProcessor()
       endp_insn = MockInsn(funct=0, xd=1, xs1=1, xs2=1)
       # synthesizes funct3 = (1<<2)|(1<<1)|1 = 7 = 0b111 (warp_end_p)
       npu.custom1(proc, endp_insn, 0, 0)

       # Assertions
       assert len(npu.deferred_ddr_stores) == 0
       # DDR has the L2 bytes
       assert bytes(npu.mem._ddr_bytes[0x1000:0x1000+100]) == bytes(np.arange(100, dtype=np.uint8))
   ```

   For the credit_st_chk test (via custom0 entry path):
   ```python
   def test_credit_st_chk_flushes_when_is_sloop(_make_npu_with_pattern):
       from tests.gtx._mocks import MockProcessor, MockInsn
       npu = _make_npu_with_pattern()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       npu.warp.tmu_id = 0
       _push_deferred_store(npu, nest=0, l2_off=100, ddr_off=0x2000, length=100)
       proc = MockProcessor()
       insn = MockInsn(funct=0x53, xd=0, xs1=0, xs2=0)
       npu.custom0(proc, insn, 0, 0)
       assert len(npu.deferred_ddr_stores) == 0
       assert bytes(npu.mem._ddr_bytes[0x2000:0x2000+100]) == bytes(np.arange(100, dtype=np.uint8))
   ```

   For the dispatch path (note: imports from `riscv.gtx.dispatch_4mode` since Plan 04 split):
   ```python
   def test_dispatch_iss_opcode_credit_st_chk_flushes(_make_npu_with_pattern):
       from riscv.gtx.dispatch_4mode import dispatch_iss_opcode
       from riscv.gtx.encoding import GTX_ISS_F7_CREDIT_ST_CHK
       npu = _make_npu_with_pattern()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       _push_deferred_store(npu, nest=0, ddr_off=0x3000, length=100)
       dispatch_iss_opcode(npu, 0, 0, GTX_ISS_F7_CREDIT_ST_CHK, 0, 0, 0)
       assert len(npu.deferred_ddr_stores) == 0
   ```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_deferred_store.py -x --noconftest -o "addopts="</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E "npu\.warp\.wsplit_seen = True" src/main/python/riscv/gtx/ops/control.py` matches at least 2x (wsplit + wsplit_custom0).
    - `grep -E "if not npu\.warp\.wsplit_seen" src/main/python/riscv/gtx/ops/control.py` matches.
    - `grep -E "npu\.flush_deferred_ddr_stores\(\)" src/main/python/riscv/gtx/ops/control.py` matches (in _do_endp).
    - `grep -E "if npu\.warp\.is_sloop" src/main/python/riscv/gtx/ops/dma.py` matches (in _credit_st_chk).
    - `grep -E "npu\.flush_deferred_ddr_stores\(\)" src/main/python/riscv/gtx/ops/dma.py` matches.
    - `grep -E "GTX_ISS_F7_CREDIT_ST_CHK and npu\.warp\.is_sloop" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - All 10 deferred_store tests pass.
    - No regression: `pytest tests/gtx/test_warp.py` still passes (P2 wsplit/wjoin tests).
  </acceptance_criteria>
  <done>Both flush triggers wired. wsplit_seen sentinel set in 2 places. end_p path covers ROADMAP success #4 (simple firmware). credit_st_chk path covers P4 mm_basic.elf (plan-style firmware). DMA-03 closed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: DMA round-trip integration test (DMA-05) + VALIDATION.md sign-off flag flip</name>
  <files>
    tests/gtx/test_dma_roundtrip.py,
    .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md
  </files>
  <read_first>
    - src/main/python/riscv/gtx/dma_engine.py (Plan 01: exec_dma_2d, firmware_dma_sloop_load/store)
    - src/main/python/riscv/gtx/ddr.py (Plan 03: ddr_init_from_file, ddr_dump_to_file)
    - src/main/python/riscv/gtx/npu.py (GtxNpu + flush_deferred_ddr_stores)
    - .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md (current frontmatter: nyquist_compliant=false, wave_0_complete=false, Approval: pending)
    - 03-RESEARCH.md "Test Patterns" §test_dma_roundtrip.py -- full chain recipe
    - ROADMAP success #1 (test_dma_roundtrip.py -- write FP16 pattern at L1, push L2, push DDR, dump, reload, reverse path, byte-exact match)
  </read_first>
  <behavior>
    Round-trip test exercises the full L1↔L2↔DDR chain end-to-end:
    1. Pre-populate L1 with a known FP16 pattern: `pattern = np.arange(4096, dtype=np.float16)`.
    2. Forward path L1 → L2: `dma_engine.exec_dma_2d(npu.mem, nest_id=0, l2_addr=0, l1_addr=0, width=8192, height=1, is_load=False)` (8192 = 4096 elements × 2 bytes).
    3. Forward path L2 → DDR via deferred store: set is_sloop=True, push via `firmware_dma_sloop_store`, then `npu.flush_deferred_ddr_stores()`. Verify pre-flush DDR is zero, post-flush DDR matches L2 bytes.
    4. Dump DDR to file (LTR mode).
    5. New GtxNpu (`npu2`). `ddr_init_from_file(npu2.mem, dump_file)`. Verify `npu2.mem._ddr_bytes[0:8192]` matches `npu.mem._ddr_bytes[0:8192]`.
    6. Reverse path DDR → L2: `firmware_dma_sloop_load(npu2.mem, nest=0, addr_hi=0, addr_lo=0, length=8192, height=1, rd_stride=8192, wr_stride=8192)`. Verify L2 matches original.
    7. Reverse path L2 → L1: `exec_dma_2d(npu2.mem, nest_id=0, l2_addr=0, l1_addr=0, width=8192, height=1, is_load=True)`. Verify L1 matches.
    8. Final assertion: `npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16) == pattern.view(np.uint16)` byte-exact.

    Additionally, a REVERSED-mode round-trip variant: same chain but with `monkeypatch.setenv('GTX_DDR_REVERSED', '1')` for both dump and re-init; final L1 still matches original (the reversal cancels out across dump+init since the in-memory DDR bytes are bit-identical pre-dump and post-init).

    **Final sign-off step (LAST step in this task, after all other steps complete):**
    Flip VALIDATION.md frontmatter:
    - `nyquist_compliant: false` → `nyquist_compliant: true`
    - `wave_0_complete: false` → `wave_0_complete: true`
    - `Approval: pending` → `Approval: ready`
    These flips are the LAST action so all `<automated>` blocks and Wave 0
    scaffolds are confirmed in place + green first.
  </behavior>
  <action>
1. Populate `tests/gtx/test_dma_roundtrip.py`. Use `_RISCV_AVAILABLE` skipif. Tests:

   ```python
   import numpy as np
   import pytest

   try:
       from riscv.processor import processor_t  # noqa: F401
       from riscv.extension import rocc_insn_t  # noqa: F401
       _RISCV_AVAILABLE = True
   except ImportError:
       _RISCV_AVAILABLE = False

   pytestmark = pytest.mark.skipif(
       not _RISCV_AVAILABLE,
       reason="GtxNpu round-trip requires _riscv.so",
   )


   def _make_npu():
       from riscv.gtx.npu import GtxNpu
       return GtxNpu()


   def test_dma_l1_to_ddr_roundtrip_ltr(tmp_path, monkeypatch):
       """DMA-05: L1 -> L2 -> DDR -> file -> re-init -> L2 -> L1 byte-exact."""
       monkeypatch.delenv("GTX_DDR_REVERSED", raising=False)
       from riscv.gtx import dma_engine
       from riscv.gtx.ddr import ddr_dump_to_file, ddr_init_from_file, ensure_ddr

       npu = _make_npu()
       pattern = np.arange(4096, dtype=np.float16)
       npu.mem.l1_f16(0, 0)[0:4096] = pattern

       # L1 -> L2 (no warp loop needed; using exec_dma_2d direct)
       dma_engine.exec_dma_2d(npu.mem, nest_id=0,
                                l2_addr=0, l1_addr=0,
                                width=8192, height=1, is_load=False)
       # Verify L2 bytes match L1
       l1_bytes = bytes(npu.mem.l1_byte(0, 0)[0:8192])
       l2_bytes = bytes(npu.mem.l2_byte(0)[0:8192])
       assert l1_bytes == l2_bytes

       # L2 -> DDR via deferred store
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       npu.warp.tmu_id = 0
       dma_engine.firmware_dma_sloop_store(
           npu, nest=0, addr_hi=0, addr_lo=0,
           length=8192, height=1, rd_stride=8192, wr_stride=8192,
       )
       assert len(npu.deferred_ddr_stores) == 1
       # Pre-flush snapshot -- DDR untouched
       ensure_ddr(npu.mem, 8192)
       pre_flush = bytes(npu.mem._ddr_bytes[0:8192])
       assert pre_flush == bytes(8192)   # all zeros

       npu.flush_deferred_ddr_stores()
       post_flush = bytes(npu.mem._ddr_bytes[0:8192])
       assert post_flush == l2_bytes

       # Dump
       hexf = tmp_path / "rt.hex"
       ddr_dump_to_file(npu.mem, str(hexf), 0, 8192)
       assert hexf.exists()

       # Re-init in fresh NPU
       npu2 = _make_npu()
       ddr_init_from_file(npu2.mem, str(hexf))
       assert bytes(npu2.mem._ddr_bytes[0:8192]) == post_flush

       # Reverse: DDR -> L2
       dma_engine.firmware_dma_sloop_load(
           npu2.mem, nest=0, addr_hi=0, addr_lo=0,
           length=8192, height=1, rd_stride=8192, wr_stride=8192,
       )
       assert bytes(npu2.mem.l2_byte(0)[0:8192]) == l2_bytes

       # Reverse: L2 -> L1
       dma_engine.exec_dma_2d(npu2.mem, nest_id=0,
                                l2_addr=0, l1_addr=0,
                                width=8192, height=1, is_load=True)

       # Final byte-exact assertion (FP16 view)
       assert np.array_equal(
           npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16),
           pattern.view(np.uint16),
       )


   def test_dma_l1_to_ddr_roundtrip_reversed(tmp_path, monkeypatch):
       """DMA-05 + DMA-04: round-trip in REVERSED mode also bit-exact."""
       monkeypatch.setenv("GTX_DDR_REVERSED", "1")
       from riscv.gtx import dma_engine
       from riscv.gtx.ddr import ddr_dump_to_file, ddr_init_from_file

       npu = _make_npu()
       pattern = np.arange(4096, dtype=np.float16)
       npu.mem.l1_f16(0, 0)[0:4096] = pattern

       dma_engine.exec_dma_2d(npu.mem, nest_id=0, l2_addr=0, l1_addr=0,
                                width=8192, height=1, is_load=False)
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       dma_engine.firmware_dma_sloop_store(
           npu, nest=0, addr_hi=0, addr_lo=0,
           length=8192, height=1, rd_stride=8192, wr_stride=8192,
       )
       npu.flush_deferred_ddr_stores()
       hexf = tmp_path / "rt_rev.hex"
       ddr_dump_to_file(npu.mem, str(hexf), 0, 8192)

       npu2 = _make_npu()
       ddr_init_from_file(npu2.mem, str(hexf))
       # DDR bytes should match -- REVERSED dump + REVERSED init cancel out
       assert bytes(npu2.mem._ddr_bytes[0:8192]) == bytes(npu.mem._ddr_bytes[0:8192])

       dma_engine.firmware_dma_sloop_load(
           npu2.mem, nest=0, addr_hi=0, addr_lo=0,
           length=8192, height=1, rd_stride=8192, wr_stride=8192,
       )
       dma_engine.exec_dma_2d(npu2.mem, nest_id=0, l2_addr=0, l1_addr=0,
                                width=8192, height=1, is_load=True)
       assert np.array_equal(
           npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16),
           pattern.view(np.uint16),
       )


   def test_dma_l1_to_l1_copy_via_firmware_dma_tloop_copy():
       """DMA-05 ancillary: L1 -> L1 same-SPU copy bit-exact."""
       from riscv.gtx import dma_engine
       npu = _make_npu()
       src_pattern = np.arange(2048, dtype=np.float16)
       # Write to L1 starting at offset 0; copy to L1 offset 4096 (2048 fp16s = 4096 bytes)
       npu.mem.l1_f16(0, 0)[0:2048] = src_pattern
       dma_engine.firmware_dma_tloop_copy(
           npu.mem, nest=0, spu=0,
           src_addr=0, dst_addr=4096, length=4096, height=1,
       )
       assert np.array_equal(
           npu.mem.l1_f16(0, 0)[2048:4096].view(np.uint16),
           src_pattern.view(np.uint16),
       )
   ```

2. Run the full P3 suite to confirm everything is green BEFORE flipping the VALIDATION.md flags:
   ```bash
   cd /mnt/e/14_NIGHTLY/pyspike && pytest tests/gtx/ -x
   ```
   This is a manual sanity gate -- if any P3 test fails, do NOT proceed to step 3.

3. **FINAL STEP — VALIDATION.md sign-off flag flip (must be the very last action of Plan 05).**
   Edit `.planning/phases/03-dma-ddr-i-o/03-VALIDATION.md`:
   * In the YAML frontmatter, change:
     ```yaml
     nyquist_compliant: false
     wave_0_complete: false
     ```
     to:
     ```yaml
     nyquist_compliant: true
     wave_0_complete: true
     ```
   * In the body's "Validation Sign-Off" section, update the closing line from:
     ```
     **Approval:** pending
     ```
     to:
     ```
     **Approval:** ready

     *Sign-off conditions met (Plan 05 Task 2 final step):*
     - All 5 plan PLANs have <automated> blocks in every task <verify>
     - All 6 Wave 0 test scaffolds exist and are populated by their owning plans
     - Full P3 suite (`pytest tests/gtx/ -x`) is green
     - All 6 requirement IDs (DMA-01..05, DISP-03) closed
     ```

   This step MUST be the last action because earlier steps create the testable
   surface that backs the flag flip. Flipping prematurely would falsely advertise
   compliance.
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_dma_roundtrip.py -x --noconftest -o "addopts=" &amp;&amp; grep -E "^nyquist_compliant: true$" .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md &amp;&amp; grep -E "^wave_0_complete: true$" .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md &amp;&amp; grep -E "\*\*Approval:\*\* ready" .planning/phases/03-dma-ddr-i-o/03-VALIDATION.md</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E "test_dma_l1_to_ddr_roundtrip_ltr|test_dma_l1_to_ddr_roundtrip_reversed" tests/gtx/test_dma_roundtrip.py` returns both names.
    - `grep -E "np\.array_equal\(.*\.view\(np\.uint16\)" tests/gtx/test_dma_roundtrip.py` matches (byte-exact uint16 comparison).
    - `grep -E "firmware_dma_sloop_store" tests/gtx/test_dma_roundtrip.py` matches.
    - `grep -E "ddr_dump_to_file.*ddr_init_from_file" tests/gtx/test_dma_roundtrip.py` (in close vicinity, both used).
    - All 3 round-trip tests pass.
    - VALIDATION.md frontmatter shows `nyquist_compliant: true` and `wave_0_complete: true`.
    - VALIDATION.md body shows `**Approval:** ready` (Blocker 4 fix).
  </acceptance_criteria>
  <done>DMA-05 closed: full L1↔L2↔DDR round-trip is bit-exact in both LTR and REVERSED modes. The integration validates Plans 01 (dma_engine helpers), 03 (DDR I/O), 02 (deferred queue + flush API), and 04 (4-mode dispatch + iss_opcode credit_st_chk wiring) all working together end-to-end. VALIDATION.md is flipped to ready -- Phase 3 is signed off.</done>
</task>

</tasks>

<verification>
- Plan 5 closes both DMA-03 (deferred-store flush dual-trigger) and DMA-05 (round-trip).
- Pitfall 7 (wsplit_seen NOT cleared by reset) verified end-to-end via integration scenario.
- Both flush trigger paths exercised (end_p when !wsplit_seen, credit_st_chk when is_sloop).
- Full P3 suite green: `pytest tests/gtx/ -x` succeeds.
- VALIDATION.md flags flipped to true + Approval: ready (Blocker 4 fix).
</verification>

<success_criteria>
- `pytest tests/gtx/test_deferred_store.py tests/gtx/test_dma_roundtrip.py -x --noconftest -o "addopts="` returns 0.
- `pytest tests/gtx/ -x` (full suite) passes -- no P2 regression.
- `wsplit_seen = True` set in 2 handler bodies (custom0 0x02 + custom1 0b100).
- `npu.flush_deferred_ddr_stores()` invoked from 3 distinct sites (control.py end_p, dma.py credit_st_chk, dispatch_4mode.py dispatch_iss_opcode credit_st_chk case).
- VALIDATION.md frontmatter `nyquist_compliant: true` and `wave_0_complete: true`; body Approval: ready.
</success_criteria>

<output>
After completion, create `.planning/phases/03-dma-ddr-i-o/03-05-SUMMARY.md` documenting:
- All 3 flush trigger sites + their guard conditions.
- DMA-05 round-trip test outcome (LTR + REVERSED both pass).
- Confirmation Pitfall 7 (wsplit_seen persists reset) verified at integration.
- VALIDATION.md sign-off complete (nyquist_compliant=true, wave_0_complete=true, Approval=ready).
- P3 phase status: all 6 requirement IDs (DMA-01..05, DISP-03) closed; ready for `/gsd:verify-work 3`.
</output>
</content>
