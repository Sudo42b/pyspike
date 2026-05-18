---
quick_id: 260518-ibf
phase: quick-260518-ibf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/unit/context/dma_engine.py
  - src/main/python/riscv/gtx/unit/context/dma.py
  - tests/gtx/test_mcast_copy_mem.py
autonomous: true
requirements:
  - TODO-A1   # mcast.s2l firmware-path body
  - TODO-A2   # mcast.g2s firmware-path body
  - TODO-A3   # mcast.s2s body (port; reachability TBD)
  - TODO-A4   # copy.mem firmware-path body (incl. mandatory flush)
  - DOC-FIX-1 # 3 stub docstrings (s2l rs1 layout / g2s zero-fill fiction / s2s self-broadcast fiction)
  - BASELINE-ABS
  - BASELINE-GELU
user_setup: []

must_haves:
  truths:
    - "4 vendor C++ ops (mcast.s2l, mcast.g2s, mcast.s2s, copy.mem) execute the vendor-canonical firmware-path body — not `return 0`."
    - "`#!TODO: 구현` 마커가 dma.py 4 stub 위치에서 완전 제거되고, vendor cite comment(`vendor/gtx_cpp_reference/gtx/...:<lines>`)로 대체된다."
    - "copy.mem DDR-path 첫 줄이 `npu.flush_deferred_ddr_stores()`; L2↔L2 same-NEST else-branch는 flush 호출 없음(asymmetry 보존)."
    - "3 stub docstrings 정정: s2l rs1 = (l2<<32)|l1 (high=src/low=dst), g2s zero-fill 제거, s2s self-broadcast guard 제거 + `tgt_mask = (op3 >> 32)` flat."
    - "신규 단위 테스트 4개(test_mcast_s2l/g2s/s2s/copy_mem)가 PASS — synthetic insn encoding으로 handler가 fire하고 메모리가 vendor 기대값과 byte-exact 일치."
    - "ABS strict + GELU strict regression이 baseline(94.82s, byte-exact) 그대로 유지 — 회귀 1개라도 깨지면 REVERT(fix-forward 금지)."
  artifacts:
    - path: "src/main/python/riscv/gtx/unit/context/dma_engine.py"
      provides: "4 new engine functions: firmware_mcast_s2l, firmware_mcast_g2s, firmware_mcast_s2s, firmware_copy_mem (vendor verbatim port, torch 2D-view pattern)"
      contains: "def firmware_mcast_s2l\ndef firmware_mcast_g2s\ndef firmware_mcast_s2s\ndef firmware_copy_mem"
    - path: "src/main/python/riscv/gtx/unit/context/dma.py"
      provides: "4 stub bodies replaced with shim handlers that decode operands + call engine; 3 docstrings corrected; vendor cite comments added"
      contains: "vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc"
    - path: "tests/gtx/test_mcast_copy_mem.py"
      provides: "4 unit tests (s2l/g2s/s2s/copy_mem) + 1 flush-assertion test for copy.mem"
      min_lines: 120
  key_links:
    - from: "src/main/python/riscv/gtx/unit/context/dma.py:_mcast_s2l"
      to: "dma_engine.firmware_mcast_s2l"
      via: "operand decode (rs1=(l2<<32)|l1, rs2=h|len|stride, rs3=gspr[OPERAND3] bitmask) then engine call"
      pattern: "firmware_mcast_s2l\\("
    - from: "src/main/python/riscv/gtx/unit/context/dma.py:_copy_mem"
      to: "npu.flush_deferred_ddr_stores"
      via: "engine first line on DDR-path branch (vendor dispatch.cc:784)"
      pattern: "flush_deferred_ddr_stores"
    - from: "src/main/python/riscv/gtx/unit/context/dma_engine.py:firmware_copy_mem"
      to: "mem.ddr.read / mem.ddr.write"
      via: "DDR-path 4-case dispatch (DDR↔DDR / DDR→L2 / L2→DDR); L2↔L2 same-NEST uses temp-buffer overlap-safe copy"
      pattern: "mem\\.ddr\\.(read|write)"
    - from: "tests/gtx/test_mcast_copy_mem.py::test_copy_mem"
      to: "npu.deferred_ddr_stores assertion"
      via: "pre-push DeferredDdrStore, run copy.mem, assert queue is empty post-call"
      pattern: "deferred_ddr_stores"
---

<objective>
Replace the 4 `#!TODO: 구현` stubs in `src/main/python/riscv/gtx/unit/context/dma.py:223-272`
(`mcast.s2l` / `mcast.g2s` / `mcast.s2s` / `copy.mem`) with vendor C++ verbatim ports.
Research (260518-ibf-RESEARCH.md) has line-by-line mapped every byte of the port and
flagged 3 docstring drifts vs vendor — this plan implements per research findings.

Purpose: Unblock GEMM-class workloads (`MUL_MAT`, `MUL_MAT_ID`, `SET_ROWS`, `WIN_UNPART`
firmware emits `__mcast_g2s` + `__mcast_s2l` + `__copy_mem` macros — confirmed by grep).
Maintain Category-A invariant: zero `#!TODO` markers in dma.py after this task.

Output:
  - 4 new engine functions in `dma_engine.py` (~30-50 LOC each, no proc/insn deps)
  - 4 shim handler bodies in `dma.py` (~15 LOC each, replacing `return 0` stubs)
  - 3 stub docstrings corrected to match vendor semantics
  - 1 new test file `tests/gtx/test_mcast_copy_mem.py` (~150 LOC, 4+1 tests)
  - ABS strict (94.82s) + GELU strict baselines preserved byte-exact
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260518-ibf-12-todo-part2-4-mcast-copy-mem-stubs-ven/260518-ibf-RESEARCH.md
@CLAUDE.md
@src/main/python/riscv/gtx/unit/context/dma.py
@src/main/python/riscv/gtx/unit/context/dma_engine.py
@src/main/python/riscv/gtx/unit/memory.py
@src/main/python/riscv/gtx/unit/ins/encoding.py
@src/main/python/riscv/gtx/unit/config_params.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc
@vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc

<interfaces>
<!-- Key types and contracts the executor needs. Extracted from research + codebase. -->
<!-- All four ports share these existing APIs — no new helper functions required. -->

From src/main/python/riscv/gtx/unit/context/dma_engine.py (existing template to mirror):
```python
def exec_dma_2d(mem: 'GtxMemory', *, nest_id: int, l2_addr: int, l1_addr: int,
                width: int, height: int, is_load: bool,
                l2_stride: int = 0, spu_id: int = 0) -> int
def firmware_dma_sloop_load(mem: 'GtxMemory', *, nest_id: int, l2_addr: int, l1_addr: int,
                            ...) -> int
# Pattern: invariant asserts, then ONE 2D view + single copy_() (no row loop)
```

From src/main/python/riscv/gtx/unit/memory.py (reuse-only):
```python
class GtxMemory:
    def l1_byte(self, nest_id: int, spu_id: int) -> torch.Tensor      # zero-copy view
    def l2_byte(self, nest_id: int) -> torch.Tensor                   # zero-copy view
    ddr: DDR_MEMORY                                                    # CPU-resident grow-on-demand
class DDR_MEMORY:
    def read(self, offset: int, n: int) -> torch.Tensor               # raw byte read
    def write(self, offset: int, t: torch.Tensor) -> None             # raw byte write
```

From src/main/python/riscv/gtx/unit/ins/encoding.py:
```python
GTX_ISS_F7_MCAST_S2L: int = 0b1000010   # 0x42  (mcast.s2l)
GTX_ISS_F7_MCAST_G2S: int = 0b1000100   # 0x44  (mcast.g2s / mcast.s2s / copy.mem branch by funct3)
```

From src/main/python/riscv/gtx/unit/config_params.py (use these, NOT GTX_DDR_BASE for DDR-discrimination):
```python
GTX_L2_SIZE_BYTES = 0x1000000   # 16 MiB — DDR-vs-L2 boundary per vendor dispatch.cc:779
GTX_L1_SIZE_BYTES = 384 * 1024
GTX_NEST_NUM      = 4
GTX_SPU_NUM       = 16
```

From src/main/python/riscv/gtx/unit/context/control.py (existing flush API):
```python
class GtxNpu:
    def flush_deferred_ddr_stores(self) -> None    # control.py:75, 228
    deferred_ddr_stores: list                       # queue inspected by tests
```

From dma.py shim helpers (existing, reuse):
```python
def _select_nest(npu) -> int                       # dma.py:33-49 — is_ploop ? tmu_id : 0
def _select_spu(npu)  -> int                       # dma.py:33-49 — SPU selector
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Port 4 vendor engine functions + replace 4 shim handler bodies</name>
  <files>
    src/main/python/riscv/gtx/unit/context/dma_engine.py
    src/main/python/riscv/gtx/unit/context/dma.py
  </files>
  <behavior>
    <!-- Test expectations (written first in Task 2; this task's <verify> is the import-level smoke test) -->
    - Engine: `firmware_mcast_s2l(mem, *, nest, l2_addr, l1_addr, height, length, rd_stride, target_spu_mask)` reads L2 source span ONCE (single 2D view), then `copy_()` into each SPU L1 bit-set in `target_spu_mask`.
    - Engine: `firmware_mcast_g2s(mem, *, ddr_addr, l2_addr, height, length, rd_stride, target_nest_mask)` reads DDR row span ONCE, `copy_()` into each NEST L2 bit-set in `target_nest_mask`. NO zero-fill special case (Pitfall 1).
    - Engine: `firmware_mcast_s2s(mem, *, src_tmu, src_addr, dst_addr, src_stride, dst_stride, length, height, target_nest_mask)` per-row temp-buffer L2→L2 across NESTs. NO self-broadcast guard (Pitfall 3). `src_tmu >= GTX_NEST_NUM` clamps to 0 per vendor.
    - Engine: `firmware_copy_mem(npu, *, src_addr_raw, dst_addr_raw, src_stride, dst_stride, length, height)` — branches on `src_addr_raw >= GTX_L2_SIZE_BYTES` / `dst_addr_raw >= GTX_L2_SIZE_BYTES`. If src OR dst is DDR: FIRST LINE calls `npu.flush_deferred_ddr_stores()`. Else same-NEST L2↔L2 with temp buffer.
    - length==0 normalisation per Pitfall 6: s2l/g2s → 0x10000; s2s/copy.mem → as-is (vendor literal).
    - Handler shim: each `_mcast_s2l` / `_mcast_g2s` / `_mcast_s2s` / `_copy_mem` decodes operands per vendor-correct bitfield (NOT current docstring), calls engine, returns 0.
    - Handler shim docstrings: vendor cite line `vendor/gtx_cpp_reference/gtx/<file>:<lines>`, vendor-correct operand layout. `#!TODO: 구현` marker REMOVED.
    - mcast.s2l rs1 decode: `l2_addr = (rs1 >> 32) & 0xFFFFFFFF`, `l1_addr = rs1 & 0xFFFFFFFF` (Pitfall 5). NOT the current docstring's `dst_addr[23:0], src_addr[58:32]`.
    - rs3 source for mcast.s2l / mcast.g2s: `npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)` (vendor pulls operand3 from GSPR).
    - For mcast.s2s funct3=2 / copy.mem funct3=3: operand decode follows `dispatch.cc:732-846` bitfield layout (op1/op2/op3 from XPR or GSPR per current handler pattern — check existing similar handlers for source).
  </behavior>
  <action>
    **Step A — Add 4 engine functions to `src/main/python/riscv/gtx/unit/context/dma_engine.py`** (append after existing `exec_load_svr` block, keep file ordered top-to-bottom by complexity).

    Each engine function MUST:
    1. Start with a vendor cite docstring: `"""Direct port of vendor/gtx_cpp_reference/gtx/<file>:<lines>."""`
    2. Apply length/height normalisation EXACTLY per vendor (see Pitfall 6 table in RESEARCH.md):
       - `firmware_mcast_s2l`: `if height == 0: height = 1; if length == 0: length = 0x10000` (vendor custom0.cc:248-249)
       - `firmware_mcast_g2s`: same (vendor custom0.cc:561)
       - `firmware_mcast_s2s`: `if height == 0: height = 1` only (vendor dispatch.cc:741-742, NO length normalisation)
       - `firmware_copy_mem`: `if height == 0: height = 1` only (vendor dispatch.cc:777, NO length normalisation)
    3. Use existing `mem.l2_byte(nest)` / `mem.l1_byte(nest, spu)` / `mem.ddr.read` / `mem.ddr.write` only — NO new helper functions.
    4. Mirror `exec_dma_2d` invariant-assert style: assert bounds (l2/l1 ranges within size), assert non-negative.
    5. **s2l / g2s reuse the `firmware_dma_sloop_load` torch-2D-view template** — single source-span snapshot, then per-target `copy_()` (NO Python row loop). Build source view as `src_buf[base : base + (h-1)*rd_stride + length].view(h, rd_stride)[:, :length]`.
    6. **s2s / copy.mem use per-row temp-buffer** (vendor dispatch.cc:752-760, 832-844) — these have distinct src/dst strides, so unified 2D view is not safe; do per-row `read → write` via small `torch.empty(length, dtype=uint8)` temp.
    7. **copy.mem DDR-path first line MUST be** `npu.flush_deferred_ddr_stores()` (vendor dispatch.cc:784). The else-branch (L2↔L2 same-NEST) MUST NOT call flush — preserve asymmetry per Pitfall 2.
    8. **copy.mem 4-case dispatch** per vendor dispatch.cc:785-830:
       - `src_is_ddr AND dst_is_ddr` → `ddr.read(src_off, length).to(target_device)` → `ddr.write(dst_off, ...)` per row
       - `src_is_ddr AND not dst_is_ddr` → `ddr.read` → `mem.l2_byte(dst_nest)[dst_off:].copy_(...)` (dst_nest = derived; check vendor for NEST selection)
       - `not src_is_ddr AND dst_is_ddr` → read L2 → `ddr.write`
       - `src_is_ddr AND dst_is_ddr` covered above; `not src AND not dst` falls through to else (L2↔L2 same-NEST temp-buffer branch)
       - For NEST selection when one side is L2: vendor dispatch.cc:790-825 uses the current dispatch NEST (read `tmu_id` from npu/dispatch state). Check existing `firmware_dma_sloop_load` for NEST-derivation pattern; if unclear, use `_select_nest(npu)` from dma.py.

    **Step B — Replace 4 stub bodies in `src/main/python/riscv/gtx/unit/context/dma.py:223-272`**.

    For each of the 4 handlers (`_mcast_s2l_stub` → `_mcast_s2l`, etc.):
    1. **Rename function** by removing `_stub` suffix (preserves handler registration; mnemonic and funct7/funct3 unchanged).
    2. **Delete the `#!TODO: 구현` line entirely** and rewrite the docstring per vendor:
       - `mcast.s2l`: `"""firmware mcast.s2l (funct7=0x42): L2 → L1 broadcast to selected SPUs.\n\n    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273.\n    rs1 = (L2_src << 32) | L1_dst (high=src/low=dst — NOT OPSET layout).\n    rs2 = (height<<48) | (length<<32) | read_stride.\n    rs3 = target_spu_bitmask (from GSPR_GTX_OPERAND3).\n    """`
       - `mcast.g2s`: `"""firmware mcast.g2s (funct7=0x44, f3=0): DDR → L2 broadcast to selected NESTs.\n\n    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:545-583.\n    rs1 = (DDR_src << 27) | L2_dst (37-bit DDR / 27-bit L2).\n    rs2 = (height<<48) | (length<<32) | read_stride.\n    rs3 = target_nest_bitmask.\n    NOTE: NO zero-fill special case (vendor has none — earlier docstring was fiction).\n    """`
       - `mcast.s2s`: `"""firmware mcast.s2s (funct7=0x44, f3=2; reachable via OPSET sub_op=0x22): L2 → L2 across NESTs.\n\n    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:732-762.\n    op1[26:0]=src_addr, op1[53:27]=dst_addr, op1[61:56]=src_tmu.\n    op2[31:0]=src_stride, op2[47:32]=length, op2[63:48]=height.\n    op3[31:0]=dst_stride, op3[63:32]=target_nest_bitmask (FLAT — no self-broadcast guard, no select bit).\n    NOTE: funct3=2 firmware reachability uncertain — see RESEARCH Pitfall 4.\n    """`
       - `copy.mem`: `"""firmware copy.mem (funct7=0x44, f3=3; OPSET sub_op=0x23): DDR↔DDR (and L2↔DDR, L2↔L2).\n\n    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:509-543 (decode)\n            + vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:763-846 (execution).\n    op1[36:0]=src_addr_raw, op3[36:0]=dst_addr_raw (37-bit; ≥ GTX_L2_SIZE_BYTES → DDR).\n    op2[31:0]=src_stride, op2[47:32]=length, op2[63:48]=height.\n    dst_stride = (op1[63:48] low 16) | (op3[63:48] << 16) — split layout.\n    DDR-path MUST call npu.flush_deferred_ddr_stores() first (vendor dispatch.cc:784).\n    """`
    3. **Body**: decode operands per vendor-correct bitfield (see Pattern: `dma.py` shim in RESEARCH.md), call corresponding `dma_engine.firmware_*` function, return its result.

    **Decision-traceability**: Each handler body's call site MUST include a 1-line `# Vendor: <file>:<lines>` comment above the engine call.

    **CRITICAL — DO NOT**:
    - DO NOT extend `decode_firmware_dma_args` (bitfields differ; per-op decoders only — Pitfall 6).
    - DO NOT add `if src_tmu == k: continue` to s2s (no self-broadcast guard in vendor — Pitfall 3).
    - DO NOT add `if src_addr == 0xFFFFFFFF: zero_fill` to g2s (no zero-fill in vendor — Pitfall 1).
    - DO NOT call `flush_deferred_ddr_stores` from any handler other than `copy.mem` DDR-path.
    - DO NOT use `GTX_DDR_BASE` for the DDR-vs-L2 boundary check; use `GTX_L2_SIZE_BYTES` (Pitfall 7).
    - DO NOT auto-remove debug prints (`feedback_debug_prints` memory).
    - DO NOT modify any other handler / unrelated code (Surgical Changes — CLAUDE.md).

    **Open-question annotation**: In the `mcast.s2s` handler docstring, include the line `NOTE: funct3=2 firmware reachability uncertain — see RESEARCH Pitfall 4.` (Task 2 will add the reachability test.)

    **Decision references**: TODO-A1/A2/A3/A4 (implement bodies per research). DOC-FIX-1 (3 docstring corrections per research findings 1).
  </action>
  <verify>
    <automated>
      uv run python -c "from riscv.gtx.unit.context import dma, dma_engine; assert hasattr(dma_engine, 'firmware_mcast_s2l'); assert hasattr(dma_engine, 'firmware_mcast_g2s'); assert hasattr(dma_engine, 'firmware_mcast_s2s'); assert hasattr(dma_engine, 'firmware_copy_mem'); print('OK')" \
      &amp;&amp; ! grep -n '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py \
      &amp;&amp; grep -c 'vendor/gtx_cpp_reference' src/main/python/riscv/gtx/unit/context/dma.py | awk '$1 >= 4 {exit 0} {exit 1}'
    </automated>
  </verify>
  <done>
    - All 4 engine functions importable from `dma_engine`.
    - `grep '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py` returns nothing (0 markers — Category A complete).
    - dma.py contains ≥4 vendor cite references (one per handler).
    - 4 stub bodies replaced with real implementations + corrected docstrings.
    - No collateral edits to other handlers.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add unit test file + run acceptance gate (baselines + new tests)</name>
  <files>
    tests/gtx/test_mcast_copy_mem.py
  </files>
  <behavior>
    <!-- Per RESEARCH §"Recommended new test" -->
    - `test_mcast_s2l_broadcast_to_2_spus`: seed NEST-0 L2 with deterministic pattern, build `rs1=(l2<<32)|l1`, `rs2=(1<<48)|(64<<32)|64`, set `gspr[GSPR_GTX_OPERAND3]=0b101` (SPU 0 + SPU 2), dispatch synthetic insn. Assert L1[0][0] and L1[0][2] match source bytes; L1[0][1] unchanged.
    - `test_mcast_g2s_broadcast_to_2_nests`: pre-seed DDR with pattern, target NESTs 0 + 2 (mask=0b101), dispatch. Assert L2[0] and L2[2] match; L2[1] unchanged.
    - `test_mcast_s2s_l2_to_l2`: seed NEST-0 L2, target NESTs 1+2+3 (mask=0b1110), dispatch via synthetic insn. Assert L2[1], L2[2], L2[3] match source. If handler does NOT fire (Pitfall 4 confirmed), mark `pytest.xfail("funct3=2 firmware reachability — needs OPSET routing")` with reason — do NOT remove the test.
    - `test_copy_mem_ddr_to_ddr_flushes_first`: pre-populate `npu.deferred_ddr_stores` with one synthetic `DeferredDdrStore`, seed DDR src, dispatch copy.mem (src_addr ≥ GTX_L2_SIZE_BYTES, dst_addr ≥ GTX_L2_SIZE_BYTES). Assert (a) dst DDR bytes match src, AND (b) `npu.deferred_ddr_stores == []` post-call (proves flush ran).
    - `test_copy_mem_l2_to_l2_no_flush`: seed L2[0] with pattern, copy to L2[0] different offset, pre-populate `deferred_ddr_stores` with sentinel. Assert dst L2 bytes match, AND `npu.deferred_ddr_stores` STILL contains sentinel (proves flush was correctly SKIPPED for L2↔L2 path — Pitfall 2 asymmetry).
  </behavior>
  <action>
    **Step A — Create `tests/gtx/test_mcast_copy_mem.py`**.

    Mirror the existing test structure of `tests/gtx/test_custom_dispatch_chain.py` and `tests/gtx/test_deferred_store.py`:
    1. Import fixtures from `tests/gtx/conftest.py` (npu/proc/mem builders).
    2. Use `pytest.fixture` for synthetic insn building — reuse any existing `make_custom0_insn(funct7, funct3, rs1, rs2, ...)` helper. If none exists, build a minimal local helper that constructs a `rocc_insn_t` proxy or directly invokes the handler via the dispatch table.
    3. Each test: setup → seed memory with deterministic pattern (e.g., `torch.arange(N, dtype=torch.uint8)`) → invoke handler → assert byte-equal via `torch.equal`.

    **Test 4 (`test_copy_mem_ddr_to_ddr_flushes_first`)** is the most critical — it validates Pitfall 2 (mandatory flush). Pre-push a `DeferredDdrStore` (check `control.py` for the dataclass; fabricate one with arbitrary fields), then post-call `assert npu.deferred_ddr_stores == []`.

    **Test 5 (`test_copy_mem_l2_to_l2_no_flush`)** is the asymmetry test — proves the L2↔L2 same-NEST path does NOT call flush (vendor dispatch.cc:832-844 branch). Pre-push a sentinel, assert it survives.

    **For mcast.s2s funct3=2 reachability** (Pitfall 4):
    - First try synthetic dispatch via the dispatch table directly (`dispatch[funct7=0x44, funct3=2]`).
    - If the handler is genuinely unreachable from synthetic insn, mark `pytest.xfail` rather than skip — leaves a permanent record that this op needs OPSET-routing follow-up. RESEARCH §"Open Questions" recommendation 1 is satisfied.

    **Test sizing**: ~30 LOC per test, ~150 LOC total.

    **Decision references**: TODO-A1..A4 (one test per op, plus 1 extra for flush-asymmetry).
  </action>
  <verify>
    <automated>
      uv run pytest tests/gtx/test_mcast_copy_mem.py --no-cov -v
    </automated>
  </verify>
  <done>
    - All 5 tests in test_mcast_copy_mem.py PASS (or test_mcast_s2s_l2_to_l2 XFAIL with documented reason — acceptable).
    - File ≥ 120 LOC (4+1 tests).
    - No skipped tests (xfail is OK for s2s; skip is NOT).
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3 (Checkpoint): Acceptance gate — ABS + GELU regression baselines PASS, byte-exact preserved</name>
  <what-built>
    - 4 vendor C++ ops ported (mcast.s2l, mcast.g2s, mcast.s2s, copy.mem)
    - 4 stubs removed, docstrings corrected per vendor
    - 5 new unit tests PASS
  </what-built>
  <how-to-verify>
    Run the acceptance gate in 3 stages — REVERT on any FAIL (fix-forward forbidden per CLAUDE.md "회귀 1개라도 깨지면 출하 보류"):

    **Stage 1 — Unit tests (must PASS, no SKIP)**:
    ```bash
    uv run pytest tests/gtx/test_mcast_copy_mem.py --no-cov -v
    ```
    Expect: 4 PASS + 1 PASS/XFAIL (s2s reachability). 0 SKIP, 0 FAIL.

    **Stage 2 — Baseline regression (byte-exact preserved)**:
    ```bash
    uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]' --no-cov -v --timeout=900
    uv run pytest 'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' --no-cov -v --timeout=180
    ```
    Expect: ABS PASS within ≤ 100s (94.82s baseline; +5s tolerance for noise). GELU PASS within timeout. Both byte-exact (strict).

    **Stage 3 — Bonus regression (optional — report only, do NOT block on SKIP)**:
    ```bash
    uv run pytest tests/gtx/test_regression_fw_full_sweep.py --no-cov -v -k 'mul_mat or set_rows or win_unpart' --timeout=300 || true
    ```
    Expect: PASS if vendor .elf is present (now newly exercised by these ports). SKIP if .elf absent — surface in summary, do NOT block.

    **Stage 4 — Marker audit**:
    ```bash
    grep -nE '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py && echo "FAIL: TODO markers remain" || echo "OK: 0 TODO markers"
    grep -cE 'vendor/gtx_cpp_reference' src/main/python/riscv/gtx/unit/context/dma.py
    ```
    Expect: "OK: 0 TODO markers", ≥ 4 vendor cite references.

    **Verdict criteria**:
    - PASS → commit per Task 2 atomic-commit-per-task rule.
    - FAIL (any stage 1 or 2) → REVERT the dma.py + dma_engine.py changes (`git checkout HEAD -- src/main/python/riscv/gtx/unit/context/dma.py src/main/python/riscv/gtx/unit/context/dma_engine.py`), keep test file as a record, surface in summary, escalate to user — do NOT attempt fix-forward.
  </how-to-verify>
  <resume-signal>Type "approved" if all stages PASS (or stage 3 SKIP with stage 1+2 PASS); otherwise report which stage failed and stop.</resume-signal>
</task>

</tasks>

<verification>
**Phase-level checks** (combine task-level verify outputs):

1. `grep -c '#!TODO' src/main/python/riscv/gtx/unit/context/dma.py` returns 0 (Category A complete).
2. `grep -c 'def firmware_mcast_s2l\|def firmware_mcast_g2s\|def firmware_mcast_s2s\|def firmware_copy_mem' src/main/python/riscv/gtx/unit/context/dma_engine.py` returns 4.
3. `uv run pytest tests/gtx/test_mcast_copy_mem.py --no-cov -v` → all PASS (or s2s XFAIL with documented reason).
4. `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU' --no-cov -v` → 2 PASS, both byte-exact, ABS within ≤ 100s walltime.
5. dma.py contains ≥ 4 references to `vendor/gtx_cpp_reference/gtx/` (one cite per handler).
6. No collateral edits to unrelated handlers (audit with `git diff --stat src/main/python/riscv/gtx/unit/context/dma.py` — expect only ~50-line localized churn in 223-272 region).
</verification>

<success_criteria>
- 4 `#!TODO: 구현` stubs replaced with vendor C++ verbatim ports (zero markers remaining in dma.py).
- 3 stub docstrings corrected per RESEARCH finding 1 (s2l rs1 layout, g2s zero-fill fiction removed, s2s self-broadcast fiction removed).
- copy.mem DDR-path calls `flush_deferred_ddr_stores()` as first line (vendor dispatch.cc:784); L2↔L2 path does NOT (asymmetry preserved).
- 4+1 new unit tests in `tests/gtx/test_mcast_copy_mem.py` PASS (s2s XFAIL acceptable with documented reason).
- ABS strict regression PASS, byte-exact, walltime ≤ 100s (94.82s baseline + tolerance).
- GELU strict regression PASS, byte-exact.
- vendor cite comment present on every handler call site and every engine function (file:line traceable).
- Atomic commit per task (Task 1: implementation; Task 2: tests; Task 3: not a commit, gate only).
- REVERT-on-FAIL discipline observed: if Stage 1 or Stage 2 of acceptance gate fails, revert source changes; do NOT fix-forward.
</success_criteria>

<output>
After completion, create `.planning/quick/260518-ibf-12-todo-part2-4-mcast-copy-mem-stubs-ven/260518-ibf-SUMMARY.md` with:
- Vendor ports completed (4 ops with vendor cite lines)
- Docstring corrections applied (3 per research finding 1)
- Test results (5 unit tests + ABS/GELU baselines + optional MUL_MAT/SET_ROWS bonus)
- Open question status: `mcast.s2s` funct3=2 reachability (XFAIL recorded → follow-up: route via OPSET)
- Bonus regression outcome (MUL_MAT/SET_ROWS/WIN_UNPART — PASS / SKIP / N/A)
- Confirmation: ABS walltime within baseline, `#!TODO` count = 0 in dma.py
</output>
