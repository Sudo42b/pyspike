---
phase: 03-dma-ddr-i-o
plan: 04
type: execute
wave: 2
depends_on: [03-01]
files_modified:
  - src/main/python/riscv/gtx/dispatch_4mode.py
  - src/main/python/riscv/gtx/dispatch.py
  - tests/gtx/test_dispatch_4mode.py
autonomous: true
requirements: [DISP-03]

must_haves:
  truths:
    - "dispatch_4mode(npu, *, opcode, op1, op2, op3, sub_op=0) -> int routes by warp loop state."
    - "Mode 1 (!is_ploop): broadcast all 4 NEST × 16 SPU = 64 (n, s) pairs to dispatch_iss_opcode."
    - "Mode 2 (is_ploop && !is_sloop && !is_tloop): 16 (tmu_id, s) pairs."
    - "Mode 3 (is_ploop && is_sloop): single dma_engine.exec_dma_2d call on tmu_id; is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)."
    - "Mode 4 (is_ploop && is_tloop): single (tmu_id, curr_id) pair."
    - "dispatch_iss_opcode is a stub-callable function in dispatch_4mode.py that handles only DMA-relevant funct7s (load_svr_l1=0x43, store_svr_l1=0x45, credit_st_chk=0x53) -- all NOPs in P3."
    - "firmware_dma (funct7=0x40) bypasses dispatch_4mode entirely; only dispatch_dma (funct7=0x07, gem5 simplified) reaches Mode 3 via dispatch_4mode."
    - "dispatch_4mode + dispatch_iss_opcode live in their OWN module (dispatch_4mode.py) per Plan 04 split decision; this avoids the Wave 2 file-write conflict with Plan 02 that also modifies dispatch.py. dispatch.py imports and re-exports them for callers."
  artifacts:
    - path: "src/main/python/riscv/gtx/dispatch_4mode.py"
      provides: "NEW MODULE -- dispatch_4mode + dispatch_iss_opcode (DMA-only stub) + Mode 3 exec_dma_2d call"
      contains: "def dispatch_4mode"
      contains_2: "def dispatch_iss_opcode"
      contains_3: "is_load = (sub_op == 0) or"
      min_lines: 90
    - path: "src/main/python/riscv/gtx/dispatch.py"
      provides: "Re-exports dispatch_4mode + dispatch_iss_opcode for callers via `from .dispatch_4mode import ...`. Keeps existing build_custom0_table / build_custom1_table builders untouched."
      contains: "from .dispatch_4mode import"
    - path: "tests/gtx/test_dispatch_4mode.py"
      provides: "DISP-03 unit tests: 4 modes parametrized + Mode 3 sub_op/opcode disambiguation + iss_opcode stub"
      min_lines: 150
  key_links:
    - from: "dispatch_4mode Mode 3"
      to: "dma_engine.exec_dma_2d"
      via: "single function call with width=op3 & 0xFFFF, height=(op3 >> 16) & 0xFFFF"
      pattern: "dma_engine\\.exec_dma_2d"
    - from: "dispatch_4mode Mode 1/2/4"
      to: "dispatch_iss_opcode"
      via: "for n in range(GTX_NEST_NUM): for s in range(GTX_SPU_NUM): dispatch_iss_opcode(npu, n, s, ...)"
      pattern: "for n in range\\(GTX_NEST_NUM\\)"
    - from: "dispatch.py"
      to: "dispatch_4mode.py"
      via: "module-level re-export so callers can do `from .dispatch import dispatch_4mode`"
      pattern: "from \\.dispatch_4mode import"
---

<objective>
Add `dispatch_4mode` to a NEW module `riscv/gtx/dispatch_4mode.py` — the 4-mode
router that all firmware operands pass through (DMA-relevant in P3;
MM/VEC/ACT-relevant in P4/P5). Direct port of `gtx_npu_dispatch.cc:25-143`. Add
`dispatch_iss_opcode` as a stub that handles only the DMA-relevant funct7s in
P3 (NOP for load_svr_l1/store_svr_l1/credit_st_chk; future-extensible for
P4/P5).

**Why a new file (Plan 04 split decision):** Plan 02 (Wave 2) modifies
`dispatch.py` to upgrade the dispatch tables to 2-level. Plan 04 (Wave 2) also
needs to add the 4-mode router. If both plans wrote to `dispatch.py` they would
race in Wave 2 parallel execution. Splitting `dispatch_4mode` + `dispatch_iss_opcode`
into their own module removes the conflict entirely AND aligns with CONTEXT
"Defer to user follow-up" §"dispatch.py 단일 파일 vs `dispatch_4mode.py` 분리"
which permitted re-evaluation if line count grows. dispatch.py keeps a small
`from .dispatch_4mode import ...` re-export for any callers that import via
`riscv.gtx.dispatch` (Plan 05 happens to import via dispatch.py;
this re-export keeps that interface stable).

Purpose: This is the routing fabric. P3 success #5 explicitly tests Mode 1
(broadcast 64) and Mode 3 (single-NEST DMA) routing. P4 will reuse Mode 4 for
MM dispatch. Plan 04 establishes the contract — Plans 05+ extend `dispatch_iss_opcode`
with funct7 cases without touching `dispatch_4mode`.

Output: `dispatch_4mode.py` (~90 LOC, NEW), `dispatch.py` gains a single import
line, populated `test_dispatch_4mode.py`.
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

@src/main/python/riscv/gtx/dispatch.py
@src/main/python/riscv/gtx/dma_engine.py
@src/main/python/riscv/gtx/encoding.py
@src/main/python/riscv/gtx/params.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc

<interfaces>
From src/main/python/riscv/gtx/dma_engine.py (Plan 01 output):
```python
def exec_dma_2d(mem, *, nest_id, l2_addr, l1_addr, width, height, is_load,
                l2_stride=0, spu_id=0) -> int
```

From src/main/python/riscv/gtx/encoding.py (Plan 01 added):
```python
GTX_OP_MM = 0; GTX_OP_VECTOR = 1; GTX_OP_ACTIVATION = 2; GTX_OP_DMA = 3
GTX_ISS_F7_DMA_LD_SVR_L1 = 0x43
GTX_ISS_F7_DMA_ST_SVR_L1 = 0x45
GTX_ISS_F7_CREDIT_ST_CHK = 0x53
```

From src/main/python/riscv/gtx/params.py:
```python
GTX_NEST_NUM = 4
GTX_SPU_NUM = 16
```

From src/main/python/riscv/gtx/warp_state.py (Plan 01 + P2):
```python
class WarpState:
    is_ploop: bool; is_tloop: bool; is_sloop: bool
    tmu_id: int; curr_id: int
    wsplit_seen: bool   # P3 added
```

Existing dispatch.py (P2 — and Plan 02 upgrades to 2-level dict; Plan 04 adds
ONE import line to re-export from dispatch_4mode):
```python
def build_custom0_table(npu) -> Dict[int, Dict]   # 2-level after Plan 02
def build_custom1_table(npu) -> Dict[int, Callable]
def _bind(fn, npu) -> Callable
# Plan 04 appends: from .dispatch_4mode import dispatch_4mode, dispatch_iss_opcode  # noqa: F401
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: dispatch_4mode.py (new module) + dispatch.py re-export + 8 routing tests</name>
  <files>
    src/main/python/riscv/gtx/dispatch_4mode.py,
    src/main/python/riscv/gtx/dispatch.py,
    tests/gtx/test_dispatch_4mode.py
  </files>
  <read_first>
    - src/main/python/riscv/gtx/dispatch.py (existing P2 builders + Plan 02's 2-level upgrade -- preserve unchanged; Plan 04 ONLY appends an `from .dispatch_4mode import` line)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc (lines 25-143 dispatch() -- ports verbatim minus thread-pool branches)
    - vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc (lines 145-200 dispatch_iss_opcode() -- P3 stubs only DMA-relevant cases)
    - 03-RESEARCH.md "Pattern 3: dispatch_4mode Signature" -- exact function shape
    - 03-RESEARCH.md "4-Mode Dispatch Router (DISP-03)" -- table of 4 modes + parametrized test fixture
    - 03-RESEARCH.md "Common Pitfalls" #8 (Mode 3 op-encoding ambiguity -- `is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)`)
    - 03-RESEARCH.md "Open Questions" Q2 (firmware_dma vs dispatch_dma orthogonality -- confirms dispatch_4mode is for the gem5-simplified dispatch_dma path, NOT firmware_dma which is direct)
    - 03-CONTEXT.md D-14 (dispatch_4mode lives in dispatch.py) AND CONTEXT "Defer to user follow-up" §"dispatch.py 단일 파일 vs dispatch_4mode.py 분리" (PERMITS the split if line count or wave-conflict pressure justifies it). Plan 04 chooses the split for wave-conflict avoidance. dispatch.py keeps the public surface via re-export.
  </read_first>
  <behavior>
    - Mode 1 (`!is_ploop`): all 64 (n, s) pairs visited.
    - Mode 2 (`is_ploop && !is_sloop && !is_tloop`): all 16 SPU at tmu_id.
    - Mode 3 (`is_ploop && is_sloop`): NO `dispatch_iss_opcode` calls; instead one `dma_engine.exec_dma_2d` call.
    - Mode 4 (`is_ploop && is_tloop`): exactly 1 dispatch_iss_opcode call at (tmu_id, curr_id).
    - Mode 3 sub_op/opcode disambiguation: `(sub_op=0, any opcode) → is_load=True`; `(sub_op=1, opcode=GTX_OP_DMA) → is_load=True` (the OR rule); `(sub_op=1, opcode=GTX_OP_VECTOR) → is_load=False`.
    - Mode 3 width/height extraction: `width = op3 & 0xFFFF`, `height = (op3 >> 16) & 0xFFFF` -- and tests assert this against op3 = (height << 16) | width fixtures.
    - dispatch_iss_opcode in P3 handles 3 funct7 stubs: `0x43 load_svr_l1`, `0x45 store_svr_l1`, `0x53 credit_st_chk` -- all NOP, return 0 (Plan 05 fills credit_st_chk with the flush trigger).
    - **Public surface preservation**: callers can import `from riscv.gtx.dispatch import dispatch_4mode` (via re-export) AND `from riscv.gtx.dispatch_4mode import dispatch_4mode` (direct). Both work.
  </behavior>
  <action>
1. Create NEW file `src/main/python/riscv/gtx/dispatch_4mode.py`:
   ```python
   #
   # Copyright 2026 WuXi EsionTech Co., Ltd.
   # ... (standard license header copy from any existing module)
   #
   """4-mode warp dispatch router + ISS opcode stub.

   Direct port of vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:25-200.

   Lives in its own module (not dispatch.py) to keep Plan 04 file ownership
   distinct from Plan 02's dispatch.py table-builder upgrades, and to honor
   CONTEXT 'Defer to user follow-up' §dispatch.py vs dispatch_4mode.py split.
   dispatch.py re-exports both functions so the public import surface
   (`riscv.gtx.dispatch.dispatch_4mode`) remains unchanged.
   """
   from .params import GTX_NEST_NUM, GTX_SPU_NUM
   from .encoding import (
       GTX_OP_DMA,
       GTX_ISS_F7_DMA_LD_SVR_L1, GTX_ISS_F7_DMA_ST_SVR_L1,
       GTX_ISS_F7_CREDIT_ST_CHK,
   )
   from . import dma_engine


   def dispatch_iss_opcode(npu, nest_id: int, spu_id: int, funct7: int,
                            op1: int, op2: int, op3: int) -> int:
       """Unified opcode router. Direct port of gtx_npu_dispatch.cc:151-1100+ --
       P3 stubs only DMA-relevant funct7s; P4 fills MM (funct7=GTX_OP_MM=0),
       P5 fills VEC/ACT.

       In P3 the only firmware paths that reach this are:
         - dispatch_4mode Mode 1/2/4 (broadcasting MM/VEC/ACT -- all NOP in P3)
         - load_svr_l1 (funct7=0x43) -- disasm-only stub here
         - store_svr_l1 (funct7=0x45) -- disasm-only stub here
         - credit_st_chk (funct7=0x53) -- Plan 05 wires the flush trigger here

       Returns 0 (cycles vestigial). NEVER raises -- invalid funct7 silently NOPs.
       """
       if nest_id < 0 or nest_id >= GTX_NEST_NUM:
           return 0
       if spu_id < 0 or spu_id >= GTX_SPU_NUM:
           return 0
       # P3 NOPs for everything. P4/P5 will dispatch to op modules here.
       # Plan 05 will replace this body with a credit_st_chk flush trigger:
       #   if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop:
       #       npu.flush_deferred_ddr_stores()
       return 0


   def dispatch_4mode(npu, *, opcode: int, op1: int, op2: int, op3: int,
                       sub_op: int = 0) -> int:
       """4-mode warp router. Direct port of gtx_npu_dispatch.cc:79-139.

       Mode 1: !is_ploop                            → broadcast all NEST × SPU
       Mode 2: is_ploop && !is_sloop && !is_tloop   → broadcast SPU within tmu_id
       Mode 3: is_ploop && is_sloop                 → DDR↔L2 via dma_engine.exec_dma_2d
       Mode 4: is_ploop && is_tloop                 → single (tmu_id, curr_id)

       Args:
         npu: GtxNpu instance (for .warp, .mem)
         opcode: GTX_OP_MM | GTX_OP_VECTOR | GTX_OP_ACTIVATION | GTX_OP_DMA
         op1, op2, op3: read by caller from npu.gspr[GSPR_GTX_OPERAND1/2/3]
         sub_op: low byte of npu.gspr[GSPR_GTX_OPCODE]

       Returns: vestigial cycle count (0 in functional model).
       """
       w = npu.warp
       if not w.is_ploop:
           # Mode 1: broadcast all
           for n in range(GTX_NEST_NUM):
               for s in range(GTX_SPU_NUM):
                   dispatch_iss_opcode(npu, n, s, opcode, op1, op2, op3)
           return 0
       if w.is_ploop and not w.is_sloop and not w.is_tloop:
           # Mode 2: broadcast within tmu_id
           for s in range(GTX_SPU_NUM):
               dispatch_iss_opcode(npu, w.tmu_id, s, opcode, op1, op2, op3)
           return 0
       if w.is_ploop and w.is_sloop:
           # Mode 3: DDR↔L2 single-NEST DMA via dma_engine.exec_dma_2d
           # Pitfall 8: is_load = (sub_op == 0) || (opcode == GTX_OP_DMA)
           is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)
           return dma_engine.exec_dma_2d(
               npu.mem,
               nest_id=w.tmu_id,
               l2_addr=op1 & 0xFFFFFFFF,
               l1_addr=op2 & 0xFFFFFFFF,
               width=op3 & 0xFFFF,
               height=(op3 >> 16) & 0xFFFF,
               is_load=is_load,
           )
       if w.is_ploop and w.is_tloop:
           # Mode 4: single (tmu_id, curr_id)
           return dispatch_iss_opcode(
               npu, w.tmu_id, w.curr_id, opcode, op1, op2, op3
           )
       return 0
   ```

2. Edit `src/main/python/riscv/gtx/dispatch.py` — APPEND a single re-export line
   at the bottom of the file (after Plan 02's 2-level builders). Do NOT touch the
   existing `build_custom0_table` / `build_custom1_table` / `_bind` functions:
   ```python
   # ----- 4-mode dispatch router (Plan 04) ---------------------------------
   # Defined in a sibling module to avoid Wave 2 file-write conflict with
   # Plan 02's table builders. Re-exported here so callers can import via
   # `from riscv.gtx.dispatch import dispatch_4mode` (stable public surface).
   from .dispatch_4mode import dispatch_4mode, dispatch_iss_opcode  # noqa: F401
   ```

3. Populate `tests/gtx/test_dispatch_4mode.py`. Use `_RISCV_AVAILABLE` self-detect block + `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)`. **Tests import directly from `riscv.gtx.dispatch_4mode`** (the new module) AND verify the re-export works via `riscv.gtx.dispatch`.

   ```python
   from typing import List, Tuple
   import pytest

   try:
       from riscv.processor import processor_t  # noqa: F401
       from riscv.extension import rocc_insn_t  # noqa: F401
       _RISCV_AVAILABLE = True
   except ImportError:
       _RISCV_AVAILABLE = False

   pytestmark = pytest.mark.skipif(
       not _RISCV_AVAILABLE,
       reason="GtxNpu requires _riscv.so",
   )


   def _make_npu():
       from riscv.gtx.npu import GtxNpu
       return GtxNpu()


   def test_dispatch_4mode_reexport_via_dispatch_module():
       """Public surface preservation: `from riscv.gtx.dispatch import dispatch_4mode`
       still works after Plan 04 split."""
       from riscv.gtx.dispatch import dispatch_4mode as via_dispatch
       from riscv.gtx.dispatch_4mode import dispatch_4mode as via_dispatch_4mode
       assert via_dispatch is via_dispatch_4mode


   @pytest.mark.parametrize("loop_state,expected_count", [
       ((False, False, False), 64),  # Mode 1: broadcast 64
       ((True,  False, False), 16),  # Mode 2: broadcast 16 in tmu_id
       ((True,  True,  False), 1),   # Mode 4: single (tmu, curr)
   ])
   def test_dispatch_4mode_routing_count(loop_state, expected_count, monkeypatch):
       """Mode 1/2/4 broadcast counts (Mode 3 verified separately -- DMA path)."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_VECTOR
       npu = _make_npu()
       is_ploop, is_tloop, is_sloop = loop_state
       npu.warp.is_ploop = is_ploop
       npu.warp.is_tloop = is_tloop
       npu.warp.is_sloop = is_sloop
       npu.warp.tmu_id = 1
       npu.warp.curr_id = 5

       seen: List[Tuple[int, int]] = []
       monkeypatch.setattr(
           d4, "dispatch_iss_opcode",
           lambda npu, n, s, opc, o1, o2, o3: seen.append((n, s)) or 0,
       )

       d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR, op1=0, op2=0, op3=0)

       if expected_count == 64:
           assert sorted(seen) == [(n, s) for n in range(4) for s in range(16)]
       elif expected_count == 16:
           assert sorted(seen) == [(1, s) for s in range(16)]
       elif expected_count == 1:
           assert seen == [(1, 5)]


   def test_dispatch_4mode_mode3_calls_exec_dma_2d_load_when_sub_op_zero(monkeypatch):
       """Mode 3 sub_op=0 → is_load=True regardless of opcode."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_VECTOR
       calls = []
       monkeypatch.setattr(
           d4.dma_engine, "exec_dma_2d",
           lambda mem, **kw: calls.append(kw) or 0,
       )
       npu = _make_npu()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       npu.warp.tmu_id = 2

       d4.dispatch_4mode(
           npu, opcode=GTX_OP_VECTOR,
           op1=0x100, op2=0x200,
           op3=(0x10 << 16) | 0x40,  # height=0x10, width=0x40
           sub_op=0,
       )
       assert len(calls) == 1
       assert calls[0]["nest_id"] == 2
       assert calls[0]["l2_addr"] == 0x100
       assert calls[0]["l1_addr"] == 0x200
       assert calls[0]["width"] == 0x40
       assert calls[0]["height"] == 0x10
       assert calls[0]["is_load"] is True


   def test_dispatch_4mode_mode3_or_rule_opcode_dma(monkeypatch):
       """Mode 3 opcode=GTX_OP_DMA → is_load=True even when sub_op != 0.
       ALSO verifies that width=op3 & 0xFFFF and height=(op3 >> 16) & 0xFFFF
       extraction is correct."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_DMA
       calls = []
       monkeypatch.setattr(
           d4.dma_engine, "exec_dma_2d",
           lambda mem, **kw: calls.append(kw) or 0,
       )
       npu = _make_npu()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True

       # op3 = 0x10001 -> width=0x0001, height=0x0001
       d4.dispatch_4mode(npu, opcode=GTX_OP_DMA,
                          op1=0, op2=0, op3=0x10001, sub_op=1)
       assert len(calls) == 1
       assert calls[0]["is_load"] is True
       # Warning 8 fix: also assert width / height extraction is correct.
       assert calls[0]["width"] == 1
       assert calls[0]["height"] == 1


   def test_dispatch_4mode_mode3_store_when_sub_op_nonzero_non_dma(monkeypatch):
       """Mode 3 sub_op≠0 AND opcode≠GTX_OP_DMA → is_load=False (store)."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_VECTOR
       calls = []
       monkeypatch.setattr(
           d4.dma_engine, "exec_dma_2d",
           lambda mem, **kw: calls.append(kw) or 0,
       )
       npu = _make_npu()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True

       d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR,
                          op1=0, op2=0, op3=0x10001, sub_op=1)
       assert calls[0]["is_load"] is False


   def test_dispatch_4mode_mode3_does_not_call_iss_opcode(monkeypatch):
       """Mode 3 routes EXCLUSIVELY through dma_engine.exec_dma_2d, NOT
       dispatch_iss_opcode."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_DMA
       iss_calls = []
       monkeypatch.setattr(
           d4, "dispatch_iss_opcode",
           lambda *a, **k: iss_calls.append(a) or 0,
       )
       monkeypatch.setattr(
           d4.dma_engine, "exec_dma_2d", lambda mem, **kw: 0,
       )
       npu = _make_npu()
       npu.warp.is_ploop = True
       npu.warp.is_sloop = True
       d4.dispatch_4mode(npu, opcode=GTX_OP_DMA,
                          op1=0, op2=0, op3=0x10001)
       assert iss_calls == []


   def test_dispatch_iss_opcode_oob_nest_returns_zero():
       from riscv.gtx import dispatch_4mode as d4
       npu = _make_npu()
       assert d4.dispatch_iss_opcode(npu, 99, 0, 0x40, 0, 0, 0) == 0
       assert d4.dispatch_iss_opcode(npu, 0, 99, 0x40, 0, 0, 0) == 0


   def test_dispatch_iss_opcode_unknown_funct7_returns_zero():
       """P3 stub: every funct7 NOPs. Plan 05 wires credit_st_chk."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_ISS_F7_CREDIT_ST_CHK
       npu = _make_npu()
       assert d4.dispatch_iss_opcode(
           npu, 0, 0, GTX_ISS_F7_CREDIT_ST_CHK, 0, 0, 0
       ) == 0
       assert d4.dispatch_iss_opcode(npu, 0, 0, 0xFF, 0, 0, 0) == 0


   def test_dispatch_4mode_mode2_uses_tmu_id_not_zero(monkeypatch):
       """Mode 2 broadcasts within tmu_id, NOT nest 0."""
       from riscv.gtx import dispatch_4mode as d4
       from riscv.gtx.encoding import GTX_OP_VECTOR
       seen: List[Tuple[int, int]] = []
       monkeypatch.setattr(
           d4, "dispatch_iss_opcode",
           lambda npu, n, s, opc, o1, o2, o3: seen.append((n, s)) or 0,
       )
       npu = _make_npu()
       npu.warp.is_ploop = True
       npu.warp.tmu_id = 3
       d4.dispatch_4mode(npu, opcode=GTX_OP_VECTOR, op1=0, op2=0, op3=0)
       assert all(n == 3 for n, _ in seen)
       assert sorted(s for _, s in seen) == list(range(16))
   ```
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_dispatch_4mode.py -x --noconftest -o "addopts="</automated>
  </verify>
  <acceptance_criteria>
    - File `src/main/python/riscv/gtx/dispatch_4mode.py` exists.
    - `grep -E "^def dispatch_4mode" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - `grep -E "^def dispatch_iss_opcode" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - `grep -E "is_load = \(sub_op == 0\) or \(opcode == GTX_OP_DMA\)" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - `grep -E "for n in range\(GTX_NEST_NUM\)" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - `grep -E "dma_engine\.exec_dma_2d\(" src/main/python/riscv/gtx/dispatch_4mode.py` matches.
    - `grep -E "for s in range\(GTX_SPU_NUM\)" src/main/python/riscv/gtx/dispatch_4mode.py` matches at least twice (Mode 1 inner + Mode 2).
    - `grep -E "from \.dispatch_4mode import dispatch_4mode, dispatch_iss_opcode" src/main/python/riscv/gtx/dispatch.py` matches (re-export line landed).
    - The new test `test_dispatch_4mode_reexport_via_dispatch_module` PASSES (confirms `riscv.gtx.dispatch.dispatch_4mode is riscv.gtx.dispatch_4mode.dispatch_4mode`).
    - All ~9 dispatch_4mode tests pass — INCLUDING the new width/height assertion in `test_dispatch_4mode_mode3_or_rule_opcode_dma` (Warning 8 fix).
    - Existing P2 build_custom0_table / build_custom1_table tests still pass (`pytest tests/gtx/test_dispatch.py`).
  </acceptance_criteria>
  <done>4-mode router lives in its own module `dispatch_4mode.py` (no Wave 2 collision with Plan 02). Mode 1 broadcasts 64; Mode 2 broadcasts 16 in tmu_id; Mode 3 invokes a single exec_dma_2d call with verified width/height extraction; Mode 4 routes to single (tmu, curr). The OR-rule for is_load (sub_op==0 OR opcode==GTX_OP_DMA) is exercised. dispatch_iss_opcode is a stub with the credit_st_chk insertion point clearly comment-marked for Plan 05. dispatch.py re-exports both functions so `from riscv.gtx.dispatch import dispatch_4mode` continues to work.</done>
</task>

</tasks>

<verification>
- All 4 modes route to correct (n, s) sets via parametrized test.
- Mode 3 hits dma_engine.exec_dma_2d, NOT dispatch_iss_opcode.
- The OR-rule for is_load is exercised in 3 separate tests covering all 4 truth-table corners.
- Mode 3 width/height extraction asserted (Warning 8 fix).
- dispatch_iss_opcode is the future extension point — Plan 05 fills credit_st_chk; P4 fills MM; P5 fills VEC/ACT.
- Public import surface preserved: `from riscv.gtx.dispatch import dispatch_4mode` still works via re-export.
</verification>

<success_criteria>
- `pytest tests/gtx/test_dispatch_4mode.py -x --noconftest -o "addopts="` returns 0 with all 9 tests green.
- DISP-03 covered: Mode 1 broadcast 64 + Mode 3 single-NEST DMA both verified.
- No regression in `pytest tests/gtx/test_dispatch.py`.
- Plan 02 + Plan 04 can run in parallel in Wave 2 with no file-write conflict (Plan 02 owns dispatch.py body; Plan 04 owns dispatch_4mode.py + a single import-line append in dispatch.py).
</success_criteria>

<output>
After completion, create `.planning/phases/03-dma-ddr-i-o/03-04-SUMMARY.md` documenting:
- Final dispatch_4mode.py LOC.
- All 4 mode entry/exit branches.
- Pitfall 8 (Mode 3 OR-rule) test coverage + Mode 3 width/height extraction assertion.
- Plan 05 contract: credit_st_chk insertion point in dispatch_iss_opcode.
- Confirmation that dispatch.py public surface (dispatch_4mode + dispatch_iss_opcode names) is preserved via re-export.
</output>
</content>
