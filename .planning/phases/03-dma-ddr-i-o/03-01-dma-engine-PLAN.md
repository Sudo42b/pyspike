---
phase: 03-dma-ddr-i-o
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/dma_engine.py
  - src/main/python/riscv/gtx/params.py
  - src/main/python/riscv/gtx/encoding.py
  - src/main/python/riscv/gtx/warp_state.py
  - tests/gtx/test_dma_engine.py
  - tests/gtx/test_firmware_dma.py
  - tests/gtx/test_deferred_store.py
  - tests/gtx/test_ddr_modes.py
  - tests/gtx/test_dma_roundtrip.py
  - tests/gtx/test_dispatch_4mode.py
autonomous: true
requirements: [DMA-01]
gap_closure: false

must_haves:
  truths:
    - "All 6 DMA engine pure helpers (exec_dma_2d, exec_load_svr, exec_store_svr, exec_transpose, exec_transpose_ddr, exec_fill) are byte-exact with C++ gtx_npu_dma.cc."
    - "DeferredDdrStore dataclass has exactly 7 fields in order: nest, l2_off, ddr_off, length, height, l2_stride, ddr_stride."
    - "decode_firmware_dma_args produces correct bit fields for LOAD/STORE/COPY including the is_copy carve-out (rs1>>32 vs (rs1>>27)&0x1FFFFFFFFF)."
    - "HW conventions length=0->65536 and height=0->1 are applied at decode."
    - "WarpState gains wsplit_seen field; reset() does NOT clear it (process-lifetime sentinel)."
    - "params.GTX_DDR_BASE = 0x370000000 exists; encoding.py exposes GSPR_GTX_OPERAND1/2/3 + GSPR_GTX_OPCODE at AUTHORITATIVE addresses 0x001/0x002/0x003/0x004 (per gtx_params.h:38-41)."
    - "Wave 0: 6 empty test scaffolds exist on disk so downstream plans can target their <verify> commands. All scaffolds use pytest.skip() placeholders so collection passes."
  artifacts:
    - path: "src/main/python/riscv/gtx/dma_engine.py"
      provides: "DeferredDdrStore + decode_firmware_dma_args + 6 exec_* helpers + 3 firmware_dma_* branch helpers"
      contains: "class DeferredDdrStore"
      contains_2: "def exec_dma_2d"
      contains_3: "def decode_firmware_dma_args"
      min_lines: 250
    - path: "src/main/python/riscv/gtx/params.py"
      provides: "GTX_DDR_BASE = 0x370000000 (per gtx_params.h:24)"
      contains: "GTX_DDR_BASE"
    - path: "src/main/python/riscv/gtx/encoding.py"
      provides: "GSPR_GTX_OPERAND1/2/3/OPCODE at 0x001/0x002/0x003/0x004 + funct7 0x40/0x41/0x38/0x39 + LSPR_SPM_ADDRA/B/C/R at 0x900/0x901/0x902/0x903 constants"
      contains: "GSPR_GTX_OPERAND3"
      contains_2: "GTX_ISS_F7_DMA"
      contains_3: "LSPR_SPM_ADDRR"
    - path: "src/main/python/riscv/gtx/warp_state.py"
      provides: "wsplit_seen field added; not reset() touched"
      contains: "wsplit_seen"
    - path: "tests/gtx/test_dma_engine.py"
      provides: "DMA-01 unit tests: 6 helpers + DeferredDdrStore field-shape + decode helper"
      min_lines: 200
    - path: "tests/gtx/test_firmware_dma.py"
      provides: "Wave 0 scaffold (filled by Plan 02)"
    - path: "tests/gtx/test_deferred_store.py"
      provides: "Wave 0 scaffold (filled by Plan 05)"
    - path: "tests/gtx/test_ddr_modes.py"
      provides: "Wave 0 scaffold (filled by Plan 03)"
    - path: "tests/gtx/test_dma_roundtrip.py"
      provides: "Wave 0 scaffold (filled by Plan 05)"
    - path: "tests/gtx/test_dispatch_4mode.py"
      provides: "Wave 0 scaffold (filled by Plan 04)"
  key_links:
    - from: "dma_engine.py exec_dma_2d"
      to: "memory.GtxMemory.l1_byte / l2_byte"
      via: "byte-level slice assignment"
      pattern: "l1_buf\\[l1_off : l1_off \\+ copy_len\\] = l2_buf"
    - from: "dma_engine.py decode_firmware_dma_args"
      to: "is_copy carve-out branch"
      via: "(rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)"
      pattern: "rs1 >> 32.*is_copy.*rs1 >> 27"
    - from: "warp_state.WarpState"
      to: "wsplit_seen"
      via: "field default False, NOT touched by reset()"
      pattern: "wsplit_seen: bool = False"
---

<objective>
Land the spike-independent DMA engine (`dma_engine.py`) — 6 pure DMA helpers, the
`DeferredDdrStore` dataclass, and the `decode_firmware_dma_args` rs1/rs2/rs3 decoder
— directly ported from `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc`. Add the
constants and `WarpState.wsplit_seen` field that downstream plans (02/04/05)
depend on. Create the 6 empty Wave 0 test scaffolds VALIDATION.md requires so
later plans can wire their `<verify>` commands without a chicken-and-egg.

Purpose: This is the byte-level memcpy bedrock. Plans 02/04/05 all import from
`dma_engine` — without it, no other DMA work compiles. Per CONTEXT D-01, this
is the spike-independent layer (no `proc`/`insn` deps), so unit tests run with
the offline mock stack alone.

Output: `dma_engine.py` (~280 LOC), 4 modified files for constants/state, 1 fully
populated test (`test_dma_engine.py`), 5 empty test scaffolds.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md
@.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md
@.planning/phases/03-dma-ddr-i-o/03-VALIDATION.md

@src/main/python/riscv/gtx/memory.py
@src/main/python/riscv/gtx/params.py
@src/main/python/riscv/gtx/encoding.py
@src/main/python/riscv/gtx/warp_state.py
@tests/gtx/_mocks.py
@vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc
@vendor/gtx_cpp_reference/gtx/gtx_npu.h
@vendor/gtx_cpp_reference/gtx/gtx_params.h

<interfaces>
<!-- Key types and contracts the executor needs. Extracted from codebase. -->

From src/main/python/riscv/gtx/memory.py:
```python
class GtxMemory:
    _l0_bytes: np.ndarray   # shape (NEST, SPU, GTX_L0_SIZE_BYTES) uint8
    _l1_bytes: np.ndarray   # shape (NEST, SPU, GTX_L1_SIZE_BYTES) uint8
    _l2_bytes: np.ndarray   # shape (NEST, GTX_L2_SIZE_BYTES) uint8
    _ddr_bytes: Optional[np.ndarray]   # None until ensure_ddr() called
    spr: dict[int, int]
    def l0_byte(self, nest: int, spu: int) -> np.ndarray
    def l1_byte(self, nest: int, spu: int) -> np.ndarray
    def l2_byte(self, nest: int) -> np.ndarray
```

From src/main/python/riscv/gtx/params.py (existing constants):
```python
GTX_NEST_NUM = 4
GTX_SPU_NUM = 16
GTX_L0_SIZE_BYTES = 1024
GTX_L1_SIZE_BYTES = 384 * 1024
GTX_L2_SIZE_BYTES = 16 * 1024 * 1024
GTX_DDR_DEFAULT_SIZE_BYTES = 4 * 1024 * 1024 * 1024
GTX_DDR_BUS_WORD_BYTES = 32
```

From src/main/python/riscv/gtx/encoding.py (existing constants):
```python
CUSTOM0_OPCODE = 0x0b
GTX_OP_MM = 0; GTX_OP_VECTOR = 1; GTX_OP_ACTIVATION = 2; GTX_OP_DMA = 3
GSPR_STARTP = 0x100   # ... (no GSPR_GTX_OPERAND* yet — P3 adds)
```

From src/main/python/riscv/gtx/warp_state.py (existing):
```python
@dataclass
class WarpState:
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False
    tmu_id: int = 0
    curr_id: int = 0
    def reset(self) -> None: ...   # P3: do NOT reset wsplit_seen
```

From C++ ground-truth (gtx_params.h:24, 38-41, 64-67) — AUTHORITATIVE addresses
verified by orchestrator (revision iteration 1, 2026-05-05):
```c++
static constexpr uint64_t GTX_DDR_BASE = 0x370000000ULL;  // line 24
static constexpr uint16_t GSPR_GTX_OPERAND1 = 0x001;      // line 38
static constexpr uint16_t GSPR_GTX_OPERAND2 = 0x002;      // line 39
static constexpr uint16_t GSPR_GTX_OPERAND3 = 0x003;      // line 40
static constexpr uint16_t GSPR_GTX_OPCODE   = 0x004;      // line 41
static constexpr uint16_t LSPR_SPM_ADDRA = 0x900;         // line 64
static constexpr uint16_t LSPR_SPM_ADDRB = 0x901;         // line 65
static constexpr uint16_t LSPR_SPM_ADDRC = 0x902;         // line 66
static constexpr uint16_t LSPR_SPM_ADDRR = 0x903;         // line 67
```

From C++ ground-truth (gtx_npu.h:1257-1266) — DeferredDdrStore exact field order:
```c++
struct deferred_ddr_store_t {
    int nest;          // -> Python: int
    uint32_t l2_off;   // -> Python: int
    uint64_t ddr_off;  // -> Python: int
    uint32_t length;   // -> Python: int
    uint16_t height;   // -> Python: int
    uint32_t l2_stride;// -> Python: int
    uint32_t ddr_stride;// -> Python: int
};
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Constants, WarpState.wsplit_seen, and 6 Wave 0 test scaffolds</name>
  <files>
    src/main/python/riscv/gtx/params.py,
    src/main/python/riscv/gtx/encoding.py,
    src/main/python/riscv/gtx/warp_state.py,
    tests/gtx/test_dma_engine.py,
    tests/gtx/test_firmware_dma.py,
    tests/gtx/test_deferred_store.py,
    tests/gtx/test_ddr_modes.py,
    tests/gtx/test_dma_roundtrip.py,
    tests/gtx/test_dispatch_4mode.py
  </files>
  <read_first>
    - **AUTHORITATIVE GROUND TRUTH (read FIRST, before any edit):**
      `vendor/gtx_cpp_reference/gtx/gtx_params.h` — lines 24, 38-41, 64-67 are the
      LOCKED HW addresses. Use these EXACTLY. Do not invent or copy from older
      drafts. Orchestrator verified these in revision iteration 1 (2026-05-05).
    - src/main/python/riscv/gtx/params.py (existing GTX_NEST_NUM etc.)
    - src/main/python/riscv/gtx/encoding.py (existing GTX_F7_*, GSPR_STARTP/ENDP)
    - src/main/python/riscv/gtx/warp_state.py (existing dataclass)
    - vendor/gtx_cpp_reference/gtx/gtx_npu.h (lines 1240-1266 — wsplit_seen + deferred_ddr_store_t struct)
    - 03-RESEARCH.md "Open Questions" Q3 (GTX_DDR_BASE belongs in params.py) and "Common Pitfalls" Pitfall 7 (wsplit_seen NOT reset)
    - 03-CONTEXT.md D-13 (ensure_ddr doubling-grow location) and D-04 (DeferredDdrStore 7 fields)
  </read_first>
  <behavior>
    - Constants land in canonical locations (params.py for HW topology, encoding.py for ISA).
    - WarpState gains `wsplit_seen: bool = False` field but `reset()` is unchanged (does NOT touch wsplit_seen).
    - Six Wave 0 test scaffold files exist on disk with `_RISCV_AVAILABLE` self-detect block + `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, ...)` UNLESS noted otherwise. Each scaffold's placeholder body uses `pytest.skip("Filled by Plan NN — placeholder")` so collection works AND the placeholder does not fail before the downstream plan fills the body.
    - Test 1: `params.GTX_DDR_BASE == 0x370000000`.
    - Test 2: `encoding.GSPR_GTX_OPERAND1 == 0x001`, `GSPR_GTX_OPERAND2 == 0x002`, `GSPR_GTX_OPERAND3 == 0x003`, `GSPR_GTX_OPCODE == 0x004` (per gtx_params.h:38-41 — AUTHORITATIVE).
    - Test 3: `encoding.LSPR_SPM_ADDRA == 0x900`, `LSPR_SPM_ADDRB == 0x901`, `LSPR_SPM_ADDRC == 0x902`, `LSPR_SPM_ADDRR == 0x903` (per gtx_params.h:64-67 — AUTHORITATIVE).
    - Test 4: `encoding.GTX_ISS_F7_DMA_LD_ST == 0x40`, `GTX_ISS_F7_DMA_3D == 0x41`, `GTX_ISS_F7_DMA_TPOSE == 0x38`, `GTX_ISS_F7_DMA_FILL == 0x39`, `GTX_ISS_F7_CREDIT_ST_CHK == 0x53`.
    - Test 5: `WarpState().wsplit_seen is False`. After `w.wsplit_seen = True; w.reset()`, `w.wsplit_seen is True` (persistence assertion — Pitfall 7).
    - Test 6: All 6 scaffold test files exist via `pathlib.Path(__file__).parent / "test_*.py"` exists check.
  </behavior>
  <action>
1. Edit `src/main/python/riscv/gtx/params.py`. Append after `GTX_DDR_BUS_WORD_BYTES`:
   ```python
   # DDR base physical address (firmware GTX_MAIN_BASE -- gtx_params.h:24)
   GTX_DDR_BASE: int = 0x370000000
   ```

2. Edit `src/main/python/riscv/gtx/encoding.py`. Append after the existing `GSPR_*` block:
   ```python
   # ----- ISS funct7 (custom0) -- DMA section, P3 -----
   GTX_ISS_F7_DMA_TPOSE: int = 0x38      # tpose (transpose)
   GTX_ISS_F7_DMA_FILL: int = 0x39       # fill
   GTX_ISS_F7_DMA_LD_ST: int = 0x40      # firmware DMA load/store/copy
   GTX_ISS_F7_DMA_3D: int = 0x41         # SVR + 3D variants (load_svr/store_svr/load_3d/store_3d)
   GTX_ISS_F7_DMA_MCAST_S2L: int = 0x42  # disasm-only stub in P3
   GTX_ISS_F7_DMA_LD_SVR_L1: int = 0x43  # load_svr_l1 alias
   GTX_ISS_F7_DMA_MCAST_GS: int = 0x44   # disasm-only stub (mcast_g2s/mcast_s2s/copy_mem share funct7)
   GTX_ISS_F7_DMA_ST_SVR_L1: int = 0x45  # store_svr_l1 alias
   GTX_ISS_F7_CREDIT_ST_CHK: int = 0x53  # credit_st_chk -- flush trigger when is_sloop

   # ----- GSPR addresses for firmware operand staging --
   # AUTHORITATIVE: gtx_params.h:38-41 (verified by orchestrator, revision iter 1).
   # Do NOT copy these from older drafts -- the original drafted 0x110..0x113 values
   # were WRONG and would silently break GSPR-staged operand reads.
   GSPR_GTX_OPERAND1: int = 0x001
   GSPR_GTX_OPERAND2: int = 0x002
   GSPR_GTX_OPERAND3: int = 0x003
   GSPR_GTX_OPCODE:   int = 0x004

   # ----- LSPR per-SPU SPM addresses --
   # AUTHORITATIVE: gtx_params.h:64-67. Used by tpose / fill / future LSPR-staged ops.
   LSPR_SPM_ADDRA: int = 0x900
   LSPR_SPM_ADDRB: int = 0x901
   LSPR_SPM_ADDRC: int = 0x902
   LSPR_SPM_ADDRR: int = 0x903
   ```
   (No "cross-check against gtx_npu.h" caveat — these values are LOCKED by the
    orchestrator-verified gtx_params.h read. Use them verbatim.)

3. Edit `src/main/python/riscv/gtx/warp_state.py`. Add the field and a comment but do NOT change reset():
   ```python
   @dataclass
   class WarpState:
       is_ploop: bool = False
       is_tloop: bool = False
       is_sloop: bool = False
       tmu_id: int = 0
       curr_id: int = 0
       # P3: process-lifetime sentinel -- set True by WSPLIT, NOT cleared by reset()
       # (matches C++ gtx_npu.h:1251 field initializer; see 03-RESEARCH Pitfall 7)
       wsplit_seen: bool = False

       def reset(self) -> None:
           self.is_ploop = False
           self.is_tloop = False
           self.is_sloop = False
           self.tmu_id = 0
           self.curr_id = 0
           # NOTE: wsplit_seen intentionally NOT reset -- see field comment above.
   ```

4. Create 6 Wave 0 test scaffold files. Each gets a license header (copy from any existing tests/gtx/test_*.py), the standard `_RISCV_AVAILABLE` self-detect block (5 lines: try import riscv.processor / except ImportError → False), and `pytestmark = pytest.mark.skipif(not _RISCV_AVAILABLE, reason="...")` UNLESS noted. **All placeholder bodies use `pytest.skip(...)` (NOT `assert hasattr`) so they pass cleanly until downstream plans fill them in.** This addresses Warning 6 from revision iteration 1.
   - `tests/gtx/test_dma_engine.py` — NO skipif (tests pure-python dma_engine module). Single placeholder:
     ```python
     def test_dma_engine_module_imports():
         pytest.skip("Filled by Task 2 -- placeholder")
     ```
   - `tests/gtx/test_firmware_dma.py` — skipif. Body: `def test_placeholder(): pytest.skip("Filled by Plan 02")`.
   - `tests/gtx/test_deferred_store.py` — skipif. Body: `def test_placeholder(): pytest.skip("Filled by Plan 05")`.
   - `tests/gtx/test_ddr_modes.py` — NO skipif (DDR I/O is pure-python). Body: `def test_placeholder(): pytest.skip("Filled by Plan 03")`.
   - `tests/gtx/test_dma_roundtrip.py` — skipif (uses GtxNpu). Body: `def test_placeholder(): pytest.skip("Filled by Plan 05")`.
   - `tests/gtx/test_dispatch_4mode.py` — skipif. Body: `def test_placeholder(): pytest.skip("Filled by Plan 04")`.

5. Add the const + WarpState assertions to `tests/gtx/test_dma_engine.py` (it has no skipif, so these run always). NOTE: Plan 01 Task 2 will APPEND additional tests to this same file — do not delete or move these Task 1 tests when Task 2 runs.
   ```python
   def test_gtx_ddr_base_constant():
       from riscv.gtx.params import GTX_DDR_BASE
       assert GTX_DDR_BASE == 0x370000000

   def test_gspr_operand_addresses():
       # AUTHORITATIVE values from gtx_params.h:38-41 (orchestrator-verified).
       from riscv.gtx.encoding import (
           GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3, GSPR_GTX_OPCODE)
       assert GSPR_GTX_OPERAND1 == 0x001
       assert GSPR_GTX_OPERAND2 == 0x002
       assert GSPR_GTX_OPERAND3 == 0x003
       assert GSPR_GTX_OPCODE   == 0x004

   def test_lspr_spm_addresses():
       # AUTHORITATIVE values from gtx_params.h:64-67.
       from riscv.gtx.encoding import (
           LSPR_SPM_ADDRA, LSPR_SPM_ADDRB, LSPR_SPM_ADDRC, LSPR_SPM_ADDRR)
       assert LSPR_SPM_ADDRA == 0x900
       assert LSPR_SPM_ADDRB == 0x901
       assert LSPR_SPM_ADDRC == 0x902
       assert LSPR_SPM_ADDRR == 0x903

   def test_iss_dma_funct7_constants():
       from riscv.gtx.encoding import (
           GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
           GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
           GTX_ISS_F7_CREDIT_ST_CHK)
       assert (GTX_ISS_F7_DMA_TPOSE, GTX_ISS_F7_DMA_FILL,
               GTX_ISS_F7_DMA_LD_ST, GTX_ISS_F7_DMA_3D,
               GTX_ISS_F7_CREDIT_ST_CHK) == (0x38, 0x39, 0x40, 0x41, 0x53)

   def test_warp_wsplit_seen_persists_through_reset():
       from riscv.gtx.warp_state import WarpState
       w = WarpState()
       assert w.wsplit_seen is False
       w.wsplit_seen = True
       w.is_ploop = True
       w.reset()
       assert w.is_ploop is False        # normal field reset
       assert w.wsplit_seen is True      # process-lifetime sentinel persists

   def test_wave0_scaffolds_exist():
       import pathlib
       td = pathlib.Path(__file__).parent
       for name in ("test_firmware_dma.py", "test_deferred_store.py",
                    "test_ddr_modes.py", "test_dma_roundtrip.py",
                    "test_dispatch_4mode.py"):
           assert (td / name).exists(), f"missing wave 0 scaffold: {name}"
   ```
  </action>
  <verify>
    <!-- Warning 9 fix: broaden the verify command to run the entire test file
         after Warning 6 fix (placeholder uses pytest.skip, not assert hasattr,
         so the placeholder no longer fails). -->
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_dma_engine.py -x --noconftest -o "addopts="</automated>
  </verify>
  <acceptance_criteria>
    - `grep -E '^GTX_DDR_BASE: int = 0x370000000' src/main/python/riscv/gtx/params.py` matches.
    - `grep -E '^GSPR_GTX_OPERAND1: int = 0x001' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E '^GSPR_GTX_OPERAND3: int = 0x003' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E '^GSPR_GTX_OPCODE:\s+int = 0x004' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E '^LSPR_SPM_ADDRA: int = 0x900' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E '^LSPR_SPM_ADDRR: int = 0x903' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E '^GTX_ISS_F7_DMA_LD_ST: int = 0x40' src/main/python/riscv/gtx/encoding.py` matches.
    - `grep -E 'wsplit_seen: bool = False' src/main/python/riscv/gtx/warp_state.py` matches.
    - `grep -c 'wsplit_seen' src/main/python/riscv/gtx/warp_state.py` returns >= 2 (field + comment in reset()).
    - `ls tests/gtx/test_firmware_dma.py tests/gtx/test_deferred_store.py tests/gtx/test_ddr_modes.py tests/gtx/test_dma_roundtrip.py tests/gtx/test_dispatch_4mode.py tests/gtx/test_dma_engine.py` all exist.
    - All Task 1 pytest tests above pass (or skip cleanly for placeholder).
  </acceptance_criteria>
  <done>All P3 constants land at canonical locations using AUTHORITATIVE gtx_params.h values; WarpState.wsplit_seen exists and survives reset(); 6 Wave 0 test files exist (one with const tests + skip-placeholder, 5 with skip-placeholders). Downstream plans can now reference these constants and write to these test files.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: dma_engine.py — DeferredDdrStore + decode_firmware_dma_args + 6 exec_* helpers + 3 firmware_dma branch helpers</name>
  <files>
    src/main/python/riscv/gtx/dma_engine.py,
    tests/gtx/test_dma_engine.py
  </files>
  <read_first>
    - vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc (lines 25-90 exec_dma_2d, 97-113 exec_load_svr, 118-136 exec_store_svr, 143-167 exec_transpose, 175-225 exec_transpose_ddr, 230-246 exec_fill, 256-397 firmware_dma decode + branches, 415-435 flush)
    - vendor/gtx_cpp_reference/gtx/gtx_npu.h (lines 1257-1266 deferred_ddr_store_t -- verify field order matches)
    - 03-RESEARCH.md "C++ Function Signature Locks (DMA-01)" table -- exact Python signatures required
    - 03-RESEARCH.md "Code Examples" §1 (decode_firmware_dma_args), §4 (exec_dma_2d), §6 (firmware_dma_sloop_store)
    - 03-RESEARCH.md "firmware_dma Encoding (DMA-02 Lock-in)" -- bit-level decode rules
    - 03-RESEARCH.md "Common Pitfalls" #1 (is_copy carve-out), #2 (length=0/height=0 conventions), #4 (DeferredDdrStore frozen + 7 fields), #8 (dispatch Mode 3 op-encoding ambiguity)
    - 03-CONTEXT.md D-04 (DeferredDdrStore 7-field), D-01 (spike-independent helpers), D-13 (doubling-grow ensure_ddr in P3 -- used here)
    - src/main/python/riscv/gtx/memory.py (GtxMemory raw byte accessors -- l1_byte/l2_byte/l0_byte)
    - src/main/python/riscv/gtx/params.py (GTX_NEST_NUM=4, GTX_SPU_NUM=16, GTX_L0_SIZE_BYTES=1024, GTX_L1_SIZE_BYTES=384*1024, GTX_L2_SIZE_BYTES=16*1024*1024, GTX_DDR_BASE=0x370000000)
  </read_first>
  <behavior>
    Test plan covers every helper. All tests use `GtxMemory()` directly + `np.zeros`/`np.arange` patterns, NO spike deps.
    - DeferredDdrStore: instantiate with 7 fields by name; `dataclasses.fields(DeferredDdrStore)` length == 7; tuple of `(.name for f in fields)` == `('nest','l2_off','ddr_off','length','height','l2_stride','ddr_stride')`. Frozen — `pytest.raises(dataclasses.FrozenInstanceError): req.nest = 99`.
    - decode_firmware_dma_args:
      * LOAD `funct3=0` (xs2=0,xs1=0,xd=0): rs1=0x1234567_89ABCDEF -> addr_hi = (0x1234567_89ABCDEF >> 27) & 0x1FFFFFFFFF, addr_lo = 0x89ABCDEF & 0x7FFFFFF. is_store=False, is_copy=False.
      * STORE `funct3=1` (xs2=1): same rs1, is_store=True, is_copy=False, wr_stride=rs2_low, rd_stride=rs3_low.
      * COPY `funct3=2` (xs2=0,xs1=1): addr_hi = rs1 >> 32 (32-bit dst, NOT 27-shift). is_store=False, is_copy=True.
      * length=0 -> 0x10000 (65536); height=0 -> 1.
      * height=5, length=256: passes through.
    - exec_dma_2d: 4 NEST × known byte pattern. LOAD path: l2 -> l1. STORE path: l1 -> l2. width=64, height=4, l2_stride=128 (sparse). Verify byte-for-byte using `assert (l1_buf[off:off+64] == l2_buf[off:off+64]).all()` after copy.
    - exec_load_svr: spu.l1[l1_addr % L1] 32 bytes (8 × 4-byte words) copied to spu.l0[l0_reg*32]. Pattern: `np.arange(32, dtype=np.uint8) + 100` in L1, assert L0 matches.
    - exec_store_svr: reverse of load_svr. L0 -> L1. Same pattern.
    - exec_transpose: 4×8 FP16 matrix at L1 addr_a, transposed 8×4 at addr_r. Build pattern as `np.arange(32, dtype=np.float16).reshape(4,8)` written byte-wise to L1, expect transposed byte layout at addr_r.
    - exec_transpose_ddr: 2×3×4 FP16 tensor in DDR, permute (2,1,0)->(0,1,2) (identity sanity check for permutation logic) and (2,1,0)->(2,0,1) (full perm, indices verified by hand).
    - exec_fill: length=10, fill_val=0x3C00 (FP16 1.0), addr_r=0x80. Verify `l1_byte(0,0)[0x80:0x80+20]` is alternating `[0x00, 0x3C] * 10`.
    - firmware_dma_sloop_store: pushes single DeferredDdrStore with .nest, .l2_off, .ddr_off (= addr_hi - GTX_DDR_BASE if addr_hi >= GTX_DDR_BASE else addr_hi), .length, .height, .l2_stride=rd_stride, .ddr_stride=wr_stride.
    - firmware_dma_sloop_load: per-row immediate DDR -> L2 memcpy (uses ensure_ddr + l2_byte). Test: pre-populate ddr[0:64] = arange, call with addr_hi=0, addr_lo=0, length=64, height=1; assert l2_byte(0)[0:64] == arange.
    - firmware_dma_tloop_load_store: L2 ↔ L1 strided per-row. Test LOAD direction (l2 -> l1) and STORE direction (l1 -> l2).
    - firmware_dma_tloop_copy: L1 -> L1 same-SPU. Use `np.memmove`-equivalent (numpy slice assignment handles overlap correctly when source and dest don't alias the SAME slice, and Python uses byte-domain so OK).
  </behavior>
  <action>
**APPEND ONLY -- do not overwrite Task 1 placeholders or test functions in
`tests/gtx/test_dma_engine.py`.** Task 1 added `test_gtx_ddr_base_constant`,
`test_gspr_operand_addresses`, `test_lspr_spm_addresses`,
`test_iss_dma_funct7_constants`, `test_warp_wsplit_seen_persists_through_reset`,
`test_wave0_scaffolds_exist`, and a `test_dma_engine_module_imports` placeholder.
Task 2 must:
  (a) Replace ONLY the `test_dma_engine_module_imports` placeholder (which uses
      `pytest.skip(...)`) with a real assertion that imports succeed.
  (b) Append the ~20 new tests below.
  (c) Leave all other Task 1 tests intact.

Create `src/main/python/riscv/gtx/dma_engine.py`. Direct-port `gtx_npu_dma.cc` lines 25-435. License header (copy from any existing module). Then:

1. **Imports & constants**:
   ```python
   from __future__ import annotations
   from dataclasses import dataclass
   from typing import TYPE_CHECKING
   import numpy as np
   from .params import (
       GTX_NEST_NUM, GTX_SPU_NUM,
       GTX_L0_SIZE_BYTES, GTX_L1_SIZE_BYTES, GTX_L2_SIZE_BYTES,
       GTX_DDR_BASE,
   )
   from .ddr import ensure_ddr  # Plan 03 upgrades to doubling-grow; Plan 01 keeps stub
   if TYPE_CHECKING:
       from .memory import GtxMemory
   ```

2. **DeferredDdrStore** — `@dataclass(frozen=True)` with fields exactly in this order:
   `nest: int`, `l2_off: int`, `ddr_off: int`, `length: int`, `height: int`, `l2_stride: int`, `ddr_stride: int`. Add docstring citing `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1257-1266`.

3. **decode_firmware_dma_args** — direct port of `gtx_npu_dma.cc:262-288`. Signature:
   ```python
   def decode_firmware_dma_args(rs1: int, rs2: int, rs3: int, *, xd: int, xs1: int, xs2: int) -> dict:
   ```
   Body matches 03-RESEARCH "Code Examples §1" exactly. Returns dict with keys: `addr_hi, addr_lo, height, length, rd_stride, wr_stride, is_store, is_copy, funct3`. **Critical bit-mask values (do not invent — use these exact values)**:
   - `funct3 = (xd << 2) | (xs1 << 1) | xs2`
   - `is_store = bool(funct3 & 1)`
   - `is_copy = (not is_store) and bool(funct3 & 2)`
   - `addr_hi = (rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)` (NOTE: the carve-out is critical — Pitfall 1)
   - `addr_lo = rs1 & 0x7FFFFFF` (27 bits)
   - `height_raw = (rs2 >> 48) & 0xFFFF`; `length_raw = (rs2 >> 32) & 0xFFFF`
   - `rs2_low = rs2 & 0xFFFFFFFF`; `rs3_low = rs3 & 0xFFFFFFFF`
   - HW conventions: `height = 1 if height_raw == 0 else height_raw`; `length = 0x10000 if length_raw == 0 else length_raw`
   - Stride direction: `if is_store: wr_stride, rd_stride = rs2_low, rs3_low` else `rd_stride, wr_stride = rs2_low, rs3_low`

4. **exec_dma_2d** — direct port of `gtx_npu_dma.cc:25-90`. Signature exactly:
   ```python
   def exec_dma_2d(mem: 'GtxMemory', *, nest_id: int, l2_addr: int, l1_addr: int,
                   width: int, height: int, is_load: bool,
                   l2_stride: int = 0, spu_id: int = 0) -> int:
   ```
   Drop the `ctx` parameter (Python doesn't need the trace branches; comment as such). Logic:
   - Early return 0 if `nest_id >= GTX_NEST_NUM` or `width == 0` or `height == 0`.
   - If `l2_stride == 0`: `l2_stride = width`.
   - Get `l1_buf = mem.l1_byte(nest_id, spu_id)` and `l2_buf = mem.l2_byte(nest_id)`.
   - For each row in range(height): compute `l2_off = (l2_addr + row * l2_stride) % GTX_L2_SIZE_BYTES`, `l1_off = (l1_addr + row * width) % GTX_L1_SIZE_BYTES`, `copy_len = min(width, GTX_L2_SIZE_BYTES - l2_off, GTX_L1_SIZE_BYTES - l1_off)`. Skip if `copy_len <= 0`. Then conditional: `if is_load: l1_buf[l1_off:l1_off+copy_len] = l2_buf[l2_off:l2_off+copy_len]` else reverse.
   - Return 0 (cycles vestigial).

5. **exec_load_svr / exec_store_svr** — direct ports of `gtx_npu_dma.cc:97-136`. Signatures:
   ```python
   def exec_load_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int, l1_addr: int, l0_reg: int) -> None
   def exec_store_svr(mem: 'GtxMemory', *, nest_id: int, spu_id: int, l1_addr: int, l0_reg: int) -> None
   ```
   - Early return if nest/spu OOB.
   - `l1_off = l1_addr % GTX_L1_SIZE_BYTES`; `l0_off = (l0_reg & 0x1F) * 32`.
   - Loop `j in range(8)`: 4-byte word copy. For load_svr: `l0[l0_off+j*4 : l0_off+j*4+4] = l1[(l1_off+j*4) % GTX_L1_SIZE_BYTES : ...+4]`. Use `mem.l0_byte(nest_id, spu_id)` and `mem.l1_byte(nest_id, spu_id)`. Modulo-wrap each src/dst index by GTX_L1_SIZE_BYTES / GTX_L0_SIZE_BYTES.
   - exec_store_svr: same pattern, l0 -> l1.

6. **exec_transpose** — direct port of `gtx_npu_dma.cc:143-167`. Signature:
   ```python
   def exec_transpose(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                       rows: int, cols: int, addr_a: int, addr_r: int) -> int
   ```
   - Early return 0 if OOB or rows/cols == 0.
   - `l1 = mem.l1_byte(nest_id, spu_id)`.
   - Double loop: `for i in range(rows): for j in range(cols): s_off = (addr_a + (j + cols * i) * 2) % GTX_L1_SIZE_BYTES; d_off = (addr_r + (i + rows * j) * 2) % GTX_L1_SIZE_BYTES; l1[d_off:d_off+2] = l1[s_off:s_off+2]`.
   - Return 0.

7. **exec_transpose_ddr** — direct port of `gtx_npu_dma.cc:175-225`. Signature:
   ```python
   def exec_transpose_ddr(mem: 'GtxMemory', *, src_addr: int, dst_addr: int,
                           dim2: int, dim1: int, dim0: int,
                           p2: int, p1: int, p0: int) -> None
   ```
   - Compute `src_off = (src_addr - GTX_DDR_BASE) if src_addr >= GTX_DDR_BASE else src_addr`. Same for dst_off.
   - Treat dim==0 -> 1 sentinel (matches C++ line 185-187).
   - `old_dims = [dim0, dim1, dim2]`; `new_dim1 = old_dims[p1]`; `new_dim0 = old_dims[p0]`.
   - Strides: `old_s1 = dim0`, `old_s2 = dim1 * dim0`, `new_s1 = new_dim0`, `new_s2 = new_dim1 * new_dim0`.
   - Triple loop over (i2, i1, i0). For each: `src_idx = i2*old_s2 + i1*old_s1 + i0`, `oi = [i0, i1, i2]`, `dst_idx = oi[p2]*new_s2 + oi[p1]*new_s1 + oi[p0]`. Each element is 2 bytes (FP16): `s = src_off + src_idx*2; d = dst_off + dst_idx*2`. Use `ensure_ddr(mem, max(s, d) + 2)` once before the loop. Bounds-check `s+1 < cap` and `d+1 < cap`. Then `mem._ddr_bytes[d:d+2] = mem._ddr_bytes[s:s+2]`.

8. **exec_fill** — direct port of `gtx_npu_dma.cc:230-246`. Signature:
   ```python
   def exec_fill(mem: 'GtxMemory', *, nest_id: int, spu_id: int,
                  length: int, fill_val: int, addr_r: int) -> int
   ```
   - Early return 0 if OOB.
   - `l1 = mem.l1_byte(nest_id, spu_id)`. Loop `i in range(length)`: `off = (addr_r + i*2) % GTX_L1_SIZE_BYTES`. `l1[off] = fill_val & 0xFF; l1[off+1] = (fill_val >> 8) & 0xFF`.

9. **firmware_dma_sloop_store** — direct port of `gtx_npu_dma.cc:319-326`. Signature:
   ```python
   def firmware_dma_sloop_store(npu, *, nest: int, addr_hi: int, addr_lo: int,
                                  length: int, height: int,
                                  rd_stride: int, wr_stride: int) -> int
   ```
   - `ddr_off = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi`.
   - Append `DeferredDdrStore(nest=nest, l2_off=addr_lo, ddr_off=ddr_off, length=length, height=height, l2_stride=rd_stride, ddr_stride=wr_stride)` to `npu.deferred_ddr_stores`.
   - Return 0.

10. **firmware_dma_sloop_load** — direct port of `gtx_npu_dma.cc:294-318` LOAD branch. Signature:
    ```python
    def firmware_dma_sloop_load(mem: 'GtxMemory', *, nest: int, addr_hi: int, addr_lo: int,
                                  length: int, height: int,
                                  rd_stride: int, wr_stride: int) -> int
    ```
    - `ddr_off_base = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi`.
    - Compute the maximum DDR offset touched: `max_off = ddr_off_base + (height-1)*rd_stride + length`. `ensure_ddr(mem, max_off)`.
    - For each row in range(height): `ddr_off = ddr_off_base + row * rd_stride` (NOT modulo); `l2_off = (addr_lo + row * wr_stride) % GTX_L2_SIZE_BYTES`; `copy_len = min(length, mem._ddr_bytes.size - ddr_off, GTX_L2_SIZE_BYTES - l2_off)`. Skip if <= 0. `mem.l2_byte(nest)[l2_off:l2_off+copy_len] = mem._ddr_bytes[ddr_off:ddr_off+copy_len]`.
    - Return 0.

11. **firmware_dma_tloop_load_store** — direct port of `gtx_npu_dma.cc:349-391`. Signature:
    ```python
    def firmware_dma_tloop_load_store(mem: 'GtxMemory', *, nest: int, spu: int,
                                        is_store: bool,
                                        addr_hi: int, addr_lo: int,
                                        length: int, height: int,
                                        rd_stride: int, wr_stride: int) -> int
    ```
    - `l1 = mem.l1_byte(nest, spu)`; `l2 = mem.l2_byte(nest)`.
    - For each row in range(height):
      * If not is_store (LOAD L2->L1): `hi_off = (addr_hi + row * rd_stride) % GTX_L2_SIZE_BYTES`, `lo_off = (addr_lo + row * length) % GTX_L1_SIZE_BYTES`.
      * If is_store (STORE L1->L2): `hi_off = (addr_hi + row * wr_stride) % GTX_L2_SIZE_BYTES`, `lo_off = (addr_lo + row * length) % GTX_L1_SIZE_BYTES`.
      * `copy_len = min(length, GTX_L2_SIZE_BYTES - hi_off, GTX_L1_SIZE_BYTES - lo_off)`. Skip if <= 0.
      * `if not is_store: l1[lo_off:lo_off+copy_len] = l2[hi_off:hi_off+copy_len]` else reverse.
    - Return 0.

12. **firmware_dma_tloop_copy** — direct port of `gtx_npu_dma.cc:334-348`. Signature:
    ```python
    def firmware_dma_tloop_copy(mem: 'GtxMemory', *, nest: int, spu: int,
                                  src_addr: int, dst_addr: int,
                                  length: int, height: int) -> int
    ```
    - `l1 = mem.l1_byte(nest, spu)`.
    - For each row in range(height): `s_off = (src_addr + row * length) % GTX_L1_SIZE_BYTES`, `d_off = (dst_addr + row * length) % GTX_L1_SIZE_BYTES`, `copy_len = min(length, GTX_L1_SIZE_BYTES - s_off, GTX_L1_SIZE_BYTES - d_off)`. Skip if <= 0. **Use `l1[d_off:d_off+copy_len] = l1[s_off:s_off+copy_len].copy()`** — explicit `.copy()` is essential because the source/dest may overlap; numpy slice assignment without copy can corrupt overlapping ranges (matches C++ `std::memmove`).

13. Update `tests/gtx/test_dma_engine.py` with all helpers' tests per the `<behavior>` block above (APPEND only). Each test uses `from riscv.gtx.memory import GtxMemory` and `from riscv.gtx import dma_engine`. Use `GtxMemory()` directly — no spike, no MockProcessor needed (Plan 01 helpers are spike-independent per CONTEXT D-01).

    **First**, REPLACE the `test_dma_engine_module_imports` placeholder body (which currently uses `pytest.skip(...)`) with the real import-check assertion:
    ```python
    def test_dma_engine_module_imports():
        from riscv.gtx import dma_engine
        assert hasattr(dma_engine, 'exec_dma_2d')
        assert hasattr(dma_engine, 'decode_firmware_dma_args')
        assert hasattr(dma_engine, 'DeferredDdrStore')
    ```

    **Then APPEND** the following tests. Required test names (each verifies one branch / pitfall):
    - `test_deferred_ddr_store_has_seven_fields_in_order` (Pitfall 4)
    - `test_deferred_ddr_store_is_frozen`
    - `test_decode_load_basic` (LOAD funct3=0)
    - `test_decode_store_swaps_strides` (STORE funct3=1)
    - `test_decode_copy_uses_high_32_bits` (Pitfall 1 — is_copy carve-out)
    - `test_decode_length_zero_means_65536` (Pitfall 2)
    - `test_decode_height_zero_means_one` (Pitfall 2)
    - `test_exec_dma_2d_l2_to_l1_load`
    - `test_exec_dma_2d_l1_to_l2_store`
    - `test_exec_dma_2d_strided`
    - `test_exec_dma_2d_zero_height_returns_zero`
    - `test_exec_load_svr_32_bytes`
    - `test_exec_store_svr_round_trip` (load_svr followed by store_svr returns same bytes)
    - `test_exec_transpose_4x8_to_8x4`
    - `test_exec_fill_writes_le_byte_pair` (asserts L1 contains [lo, hi, lo, hi, ...] for 0x3C00)
    - `test_exec_transpose_ddr_identity_perm` (perm = (2,1,0) input dims, p2=2,p1=1,p0=0 -> identity copy)
    - `test_firmware_dma_sloop_store_pushes_one_request` (uses a dummy npu with `deferred_ddr_stores = []` attribute)
    - `test_firmware_dma_sloop_load_immediate_copy` (DDR -> L2 verified)
    - `test_firmware_dma_tloop_load_store_l2_l1`
    - `test_firmware_dma_tloop_copy_l1_to_l1` (overlapping src/dst chunk to verify .copy() guard)
  </action>
  <verify>
    <automated>cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; pytest tests/gtx/test_dma_engine.py -x --noconftest -o "addopts="</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "^def \|^class " src/main/python/riscv/gtx/dma_engine.py` returns >= 11 (1 dataclass + 1 decoder + 6 exec_* + 4 firmware_dma_*).
    - `grep -E "^@dataclass\(frozen=True\)$" src/main/python/riscv/gtx/dma_engine.py` matches.
    - `grep -E "rs1 >> 32.*if is_copy.*else.*rs1 >> 27" src/main/python/riscv/gtx/dma_engine.py` matches (Pitfall 1).
    - `grep -E "0x10000 if length_raw == 0" src/main/python/riscv/gtx/dma_engine.py` matches (Pitfall 2).
    - `grep -E "1 if height_raw == 0" src/main/python/riscv/gtx/dma_engine.py` matches.
    - `grep -E "ddr_off.*= .*addr_hi - GTX_DDR_BASE" src/main/python/riscv/gtx/dma_engine.py` matches.
    - `grep -E "= l1\[s_off:s_off\+copy_len\]\.copy\(\)" src/main/python/riscv/gtx/dma_engine.py` matches (firmware_dma_tloop_copy uses .copy()).
    - All ~25+ `pytest tests/gtx/test_dma_engine.py` tests pass (Task 1's 7 + Task 2's ~20).
  </acceptance_criteria>
  <done>`dma_engine.py` is the spike-independent DMA bedrock. All 6 C++ exec_* helpers + 3 firmware_dma branch helpers + decode helper + DeferredDdrStore are byte-exact ports of `gtx_npu_dma.cc:25-435`. Plan 02 can now register `@handler` entry points that delegate to these. Plan 04 can call `exec_dma_2d` from Mode 3. Plan 05 can use `firmware_dma_sloop_store` to test the deferred queue.</done>
</task>

</tasks>

<verification>
- All 6 Wave 0 test files exist on disk (5 placeholders + 1 populated).
- WarpState.wsplit_seen survives reset() (Pitfall 7 lock-in).
- All exec_* helpers byte-match C++ source.
- DeferredDdrStore is frozen with exactly 7 fields in spec order.
- decode_firmware_dma_args handles is_copy carve-out (rs1>>32) AND HW conventions (length=0->65536, height=0->1).
- AUTHORITATIVE constants from gtx_params.h:24, 38-41, 64-67 land in encoding.py / params.py at correct values.
</verification>

<success_criteria>
- `pytest tests/gtx/test_dma_engine.py -x --noconftest -o "addopts="` returns 0 with ≥ 25 passing tests (7 from Task 1 + ~20 from Task 2).
- 5 Wave 0 placeholder files exist with `pytest.skip(...)` bodies.
- `git diff --stat` shows 4 modified source files + 6 new test files + 1 new module file.
</success_criteria>

<output>
After completion, create `.planning/phases/03-dma-ddr-i-o/03-01-SUMMARY.md` documenting:
- Final dma_engine.py LOC and helper count.
- Any deviations from research-prescribed signatures (should be zero).
- Confirmation that GSPR_GTX_OPERAND1/2/3=0x001/0x002/0x003 + GSPR_GTX_OPCODE=0x004 (per gtx_params.h:38-41) and LSPR_SPM_ADDRA/B/C/R=0x900/0x901/0x902/0x903 (per gtx_params.h:64-67).
- Confirmation that Pitfall 1/2/4/7 were exercised by tests.
</output>
</content>
