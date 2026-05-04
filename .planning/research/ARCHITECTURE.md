# Architecture Research — Python GTX NPU on pyspike

**Domain:** RoCC functional model (NumPy-backed Python port of `gtx_npu_t : rocc_t`)
**Researched:** 2026-05-04
**Confidence:** HIGH (C++ reference is the spec; pyspike trampoline surface is mapped and validated)

> Scope note: This document describes ONLY the new Python NPU component sitting on top of pyspike. The pyspike binding layers (Python user / pybind11 / trampoline / runtime bootstrap / CLI) are already mapped in `.planning/codebase/ARCHITECTURE.md` and are not re-described here.

---

## 1. Module Layout

### Recommended Package: `riscv.gtx`

A subpackage under `src/main/python/riscv/`, NOT under `examples/`. Rationale:

- `examples/xhuimt/`, `examples/xthead/` are **out-of-tree user code** demonstrating the API. They are added to `pythonpath` in `pyproject.toml` for tests but do not ship in the wheel.
- GTX NPU is the v1 product feature ("`pip install spike` → `from riscv.gtx import GtxNpu`" per PROJECT.md GTX-PKG-01). It must ship inside the `riscv` package.
- Sitting alongside `riscv.dev`, `riscv.isa`, `riscv.csrs` etc. matches the existing convention (canonical building blocks live under `riscv.*`; example/user packages live under `examples/*`).

### Concrete Directory Tree

```
src/main/python/riscv/gtx/
├── __init__.py              # Public re-exports: GtxNpu, register-on-import
├── npu.py                   # GtxNpu (riscv.isa.ROCC) — top-level class, custom0/1/2/3 entry
├── params.py                # Mirror of gtx_params.h: GTX_NUM_NESTS, GTX_L1_SIZE, SPR addresses
├── encoding.py              # funct7/funct3 constants, ucode_to_funct7 table, GTX_OP_*, GTX_MM_*
├── memory.py                # GtxMemory: NumPy-backed GSPR/NSPR/LSPR + L0/L1/L2/DDR holder
├── fp.py                    # FP16/FP8/INT8 conversion helpers (numpy-based)
├── dispatch.py              # 4-mode router (`_dispatch`) + `_dispatch_iss_opcode` 75-case switch
├── loop.py                  # P/S/T loop state machine: startp/endp/starts/ends/startt/endt
├── spr.py                   # wr_spr / rd_spr (range-routed to gspr/nspr/lspr)
├── disasm.py                # Disassembly table: arg_t subclasses + get_disasms() builder
├── ddr.py                   # ddr_init_from_file, ddr_dump_to_file, GTX_DDR_REVERSED handling
├── ops/
│   ├── __init__.py          # Op registry (funct7 → handler) — built once at import
│   ├── mm.py                # exec_mm, exec_mm_s, exec_mm_o, exec_mm_v, exec_mm_t, mxe_accum, firmware_mm_op
│   ├── vec.py               # exec_vector_op, exec_vec_scalar, exec_*_imm (scalar/vector/vfunc/bitwise)
│   ├── act.py               # exec_activation, exec_act_imm, exec_softmax_imm
│   ├── dma.py               # exec_dma_2d, mcast_g2s, mcast_s2l, copy_mem, exec_load_svr/store_svr
│   ├── pool.py              # exec_pooling, 2D pooling firmware path
│   ├── conv.py              # IM2COL_N / IM2COL_D
│   ├── tpose.py             # exec_transpose, exec_transpose_ddr, exec_fill
│   ├── format.py            # exec_format_cvt (FP8/INT8/INT32/FP32/FP64 ↔ FP16)
│   └── mexec.py             # Microcode fetch/decode/execute (calls back into dispatch)
└── data/                    # OPTIONAL (Phase 5+): bundled .elf regression assets, golden hex
```

### Why this split (not a single 4000-line `npu.py`)

| Concern | Single-file risk | Split benefit |
|---|---|---|
| Python import overhead | All 30+ ops loaded eagerly | Same — `riscv.gtx` always pulls them all (registry needs them at import time). Net: equal. |
| Circular imports | None possible | **Real risk:** `ops/mm.py` needs `GtxMemory`, `dispatch.py` needs `ops/*`. Solution below. |
| Diff churn | Every op edit touches one giant file | Each file ~300-700 lines — tractable, mirrors C++ split (`gtx_npu_mm.cc`, `gtx_npu_vec.cc`, etc.) |
| Test isolation | Mocking is harder | Per-op tests can `from riscv.gtx.ops.mm import exec_mm` and pass a fresh `GtxMemory` |
| Bit-exact debugging | Stack traces opaque | Exception with `riscv/gtx/ops/mm.py:147` immediately localises |

### Circular import discipline

Op modules MUST NOT import `npu.GtxNpu`. Instead:

```python
# riscv/gtx/ops/mm.py
from riscv.gtx.memory import GtxMemory
from riscv.gtx.params import GTX_NUM_NESTS, ...

def exec_mm(mem: GtxMemory, nest_id: int, spu_id: int, ...) -> int:
    """Pure function on the memory snapshot. Returns cycles."""
```

The class `GtxNpu` (in `npu.py`) only stores state and delegates:
```python
def custom0(self, proc, insn, xs1, xs2):
    return _dispatch_custom0(self.mem, self.loop, proc, insn)
```

This makes ops directly unit-testable without instantiating spike.

---

## 2. Memory Representation

### NumPy ndarray view discipline (CRITICAL for bit-exactness)

The C++ memory is byte-arrays (`std::vector<uint8_t>`). Python firmware via `riscv.dev`-style scalar paths writes individual FP16 halfwords as little-endian byte pairs (per CLAUDE.md "GTX-MEM-01: 모든 L1/L0 FP16 접근을 little-endian"). NumPy must give us BOTH views **without copying**.

**Rule:** every memory region is allocated **once** as `np.zeros(N, dtype=np.uint8)` and all other views (`view(np.uint16)`, `.view(np.float16)`, slicing) are derived via `.view()` — never `.copy()`, never `np.frombuffer()` of a temp.

### GtxMemory structure (one allocation per region, multiple views)

```python
# riscv/gtx/memory.py
class GtxMemory:
    def __init__(self):
        # GSPR / NSPR / LSPR: dict[int, int] is fine (sparse, ~few dozen keys touched)
        # NOT ndarray — ISS uses unordered_map<uint16_t, uint64_t> for the same reason.
        self.gspr: dict[int, int] = {}
        self.nspr: list[dict[int, int]] = [dict() for _ in range(GTX_NUM_NESTS)]
        self.lspr: list[list[dict[int, int]]] = [
            [dict() for _ in range(GTX_SPUS_PER_NEST)] for _ in range(GTX_NUM_NESTS)
        ]

        # L2: 16MB × 4 NEST. Single contiguous allocation.
        self._l2_bytes = np.zeros((GTX_NUM_NESTS, GTX_L2_SIZE), dtype=np.uint8)
        # NumPy bank views (cheap — no copy):
        self._l2_banks = self._l2_bytes.reshape(
            GTX_NUM_NESTS, GTX_L2_NUM_BANKS, GTX_L2_BANK_SIZE
        )

        # L1: 384KB × 16 SPU × 4 NEST. Single contiguous allocation.
        self._l1_bytes = np.zeros(
            (GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L1_SIZE), dtype=np.uint8
        )
        # Halfword views — same memory, different dtype.
        # 384KB / 2 = 196608 fp16 elements per SPU.
        self._l1_u16  = self._l1_bytes.view(np.uint16).reshape(
            GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L1_SIZE // 2
        )
        self._l1_f16  = self._l1_bytes.view(np.float16).reshape(
            GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L1_SIZE // 2
        )

        # L0: 1KB × 16 SPU × 4 NEST.
        self._l0_bytes = np.zeros(
            (GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L0_SIZE), dtype=np.uint8
        )
        self._l0_u16 = self._l0_bytes.view(np.uint16).reshape(
            GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L0_SIZE // 2
        )
        self._l0_f16 = self._l0_bytes.view(np.float16).reshape(
            GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L0_SIZE // 2
        )

        # DDR: lazy. 4GB max — allocate on first ensure_ddr() call.
        self._ddr_bytes: np.ndarray | None = None
        self.ddr_reversed: bool = False  # GTX_DDR_REVERSED env

    # ── Byte-addressed access (firmware MMIO writes) ─────────────────
    def l1_byte(self, nest, spu) -> np.ndarray:
        return self._l1_bytes[nest, spu]   # 384KB uint8 — direct LE write target

    # ── Halfword math access (NPU ops) ───────────────────────────────
    def l1_u16(self, nest, spu) -> np.ndarray:
        return self._l1_u16[nest, spu]      # 196608 uint16

    def l1_f16(self, nest, spu) -> np.ndarray:
        return self._l1_f16[nest, spu]      # 196608 float16
```

### Bit-exactness safeguards

1. **Native byte order.** `view(np.float16)` only matches LE FP16 if the host is little-endian (x86_64 — guaranteed by manylinux2014_x86_64 baseline in PROJECT.md). Add an `assert sys.byteorder == 'little'` in `riscv/gtx/__init__.py` to fail fast on hypothetical non-x86 platforms.

2. **Resolve the BE/LE inconsistency.** The C++ source has a contradiction:
   - `gtx/CLAUDE.md` says "L1/L0 FP16 access = little-endian (SystemC TLM 일치)"
   - `gtx_spu_mem_t::wr16_be` in `gtx_npu.h:770` writes `l1[off] = (fp16 >> 8) & 0xFF` (big-endian)
   - The shadow buffer `gtx_l1_device_t::sync_from_spu` byte-swaps SPU L1 → CPU view

   **Recommendation:** use **little-endian internally** for the Python port (matches CLAUDE.md and the shadow-after-swap CPU view, which is what firmware actually sees). Implement a one-time **byte-swap layer at the C++/Python boundary** if/when comparing against C++ libgtx_npu.so memory dumps. Document this decision in `riscv/gtx/memory.py` docstring. This is the SIMPLEST bit-exact target and matches NumPy's natural `view(np.float16)`.

3. **Slicing must NEVER copy.** Op handlers must use `mem.l1_f16(n, s)[base:base+rows]` (returns a view) not `mem.l1_f16(n, s)[base:base+rows].copy()`. NumPy is generally safe here, but watch for `astype(np.float32)` (always copies) — only do that for the FP32-accumulate temp, never to write back.

4. **FP32 accumulator pattern (VSUM, GEMM):**
   ```python
   # CORRECT: read FP16 view → accumulate in FP32 → cast back ONCE
   a = mem.l1_f16(n, s)[a_base:a_base + length]   # view, no copy
   b = mem.l1_f16(n, s)[b_base:b_base + length]   # view, no copy
   acc = np.float32(a).dot(np.float32(b))         # FP32 scalar
   out = np.float16(acc)                          # single rounding
   mem.l0_f16(n, s)[r_off] = out                  # writes back via view
   ```

### GSPR/NSPR/LSPR — dict not ndarray

The C++ uses `unordered_map<uint16_t, uint64_t>`. Although addressable by 1024 keys, only ~10–20 are actually used per kernel. An ndarray (`np.zeros(1024, dtype=np.uint64)`) is fine too but offers no semantic gain and silently returns 0 for un-set keys — which **matches the C++ fallback** (`unordered_map::operator[]` default-constructs to 0). Either works; **dict mirrors source structure 1:1** which is preferred for bit-exact diffing.

---

## 3. Dispatch Architecture

### Layered dispatch — three levels, no Python overhead surprises

```
RoCC opcode (custom-0 = 0x0b, custom-1 = 0x2b)
        │
        ▼
GtxNpu.custom0 / custom1     ← thin trampoline (1 line each)
        │
        ▼
_dispatch_custom0(mem, loop, proc, insn)    ← funct7 switch
        │ (firmware paths handled inline; ISS paths fall through)
        ▼
_dispatch(mem, loop, p, rs1, rs2, op_class)   ← 4-mode router (Mode 1/2/3/4)
        │
        ▼
_dispatch_iss_opcode(mem, n, s, funct7, op1, op2, op3, cycles)
        │
        ▼
ops.mm.exec_mm_*  /  ops.vec.exec_*  /  ops.act.exec_*  /  ops.dma.exec_*
```

### funct7 dispatch in `custom0` — dict-of-handlers (NOT match-statement)

The C++ `gtx_npu_custom0.cc` is a 950-line `switch(funct7)` with ~40 cases, mostly using rs1=0 vs rs1≠0 to distinguish firmware vs ISS encoding. Recommendation:

```python
# riscv/gtx/dispatch.py

# Each handler signature: (mem, loop, proc, insn) -> reg_t
_CUSTOM0_HANDLERS: dict[int, callable] = {}

def _handler(funct7):
    def deco(fn):
        _CUSTOM0_HANDLERS[funct7] = fn
        return fn
    return deco

@_handler(0x00)  # GTX_F7_WRSPR / firmware MM (rs1≠0)
def _h_wrspr_or_mm(mem, loop, proc, insn):
    if insn.rs1 != 0:
        return firmware_mm_op(mem, loop, proc, insn, is_accumulate=False)
    val_rs1 = proc.get_state().XPR[insn.rs1]
    val_rs2 = proc.get_state().XPR[insn.rs2]
    wr_spr(mem, val_rs1 & 0xFFFF, val_rs2)
    return 0

# ...
def dispatch_custom0(mem, loop, proc, insn) -> int:
    funct7 = insn.funct
    handler = _CUSTOM0_HANDLERS.get(funct7)
    if handler is not None:
        return handler(mem, loop, proc, insn)
    # ISS opcode passthrough (funct7 ≥ 0x08)
    if funct7 >= 0x08:
        f3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        mem.gspr[GSPR_GTX_OPCODE] = f3
        ...
        return _dispatch(mem, loop, proc, val_rs1, val_rs2, funct7)
    illegal_instruction(proc)  # match C++ default branch
    return 0
```

**Why dict not match:**
- Python `match` is 3.10+; PROJECT.md targets Python 3.8+
- Dict lookup is O(1), match is O(n) cases (compiler unrolls but not as predictably)
- Adding/removing handlers is a one-line decorator change, not a switch case edit
- Per-op test isolation: `from riscv.gtx.dispatch import _CUSTOM0_HANDLERS; _CUSTOM0_HANDLERS[0x00](...)` lets you directly invoke a handler

### `_dispatch` 4-mode router lives in `dispatch.py`

It's the same routing logic as C++ (Mode 1: !is_ploop → broadcast all; Mode 2: P only; Mode 3: P+S → exec_dma_2d on selected NEST; Mode 4: P+T → single SPU). Pure function on `mem` and `loop` state. Returns cycles.

### `_dispatch_iss_opcode` lives in `dispatch.py`

The 75-case ISS opcode router. Same dict-of-handlers pattern as `custom0`, keyed on funct7. Each entry is a thin shim that unpacks op1/op2/op3 and calls into `ops/<subsys>.py`.

### P/S/T warp state machine lives in `loop.py`

```python
# riscv/gtx/loop.py
class GtxLoop:
    def __init__(self):
        self.is_ploop = False
        self.is_sloop = False
        self.is_tloop = False
        self.tmu_id   = 0   # selected NEST when is_ploop
        self.curr_id  = 0   # selected SPU within NEST when is_tloop
        self.wsplit_cycle = 0
        # Credit counters mirror per-SPU/NEST C++ state
        self.scredit_flag = [False] * GTX_SPUS_PER_NEST
        self.tfull_flag   = [False] * GTX_NUM_NESTS

    def current_context(self) -> int:
        # CTX_C1 / C2 / C3 / C4
        if not self.is_ploop: return 1
        if self.is_sloop:     return 2
        if self.is_tloop:     return 3
        return 4
```

`custom1` (in `dispatch.py`, called from `npu.py`) is a tiny funct3 switch that calls `loop.startp/endp/starts/ends/startt/endt`. WJOIN logic (L1_DUMP, L2_DUMP, profile reporting) stays here since it touches `mem` and `loop` together.

---

## 4. Op Handler Pattern

Every `exec_*` function follows ONE signature and returns cycles (or 0 if `GTX_FUNCTIONAL_ONLY` mode):

```python
def exec_<op_name>(
    mem: GtxMemory,
    nest_id: int,
    spu_id: int,
    *op_specific_args,
) -> int:
    """One-line description, mirror of C++ gtx_npu_<subsys>.cc:exec_<op_name>."""
    # 1. Read SPRs (LSPR addresses A/B/C/R)
    spu_lspr = mem.lspr[nest_id][spu_id]
    addr_a = spu_lspr.get(LSPR_SPM_ADDRA, 0)
    addr_r = spu_lspr.get(LSPR_SPM_ADDRR, 0)

    # 2. Get the right view of memory (NEVER copy)
    l1 = mem.l1_f16(nest_id, spu_id)
    a_view = l1[addr_a // 2 : addr_a // 2 + length]
    r_view = l1[addr_r // 2 : addr_r // 2 + length]

    # 3. Compute (FP32 accumulate if reduction, FP16 in-place otherwise)
    np.add(a_view.astype(np.float32), scalar, out=tmp)  # FP32 temp
    r_view[:] = tmp.astype(np.float16)                  # bit-exact write back

    # 4. Update mxe_accum if MM_O / MM_V variant
    if accumulate:
        mem.mxe_accum[nest_id, spu_id] += result_acc

    # 5. Return cycle count (matches C++ formula or 0)
    return gtx_cycles.relu(length)  # or 0 in functional-only
```

### `mxe_accum` — class state, not function-local

The C++ `gtx_npu_t` has per-(NEST,SPU) FP32 accumulators for MM_O / MM_V chains. In Python:

```python
# riscv/gtx/memory.py — add to GtxMemory.__init__
self.mxe_accum = np.zeros(
    (GTX_NUM_NESTS, GTX_SPUS_PER_NEST, MAX_ACCUM_LEN),
    dtype=np.float32,
)
```

`exec_mm_o` reads/writes `mem.mxe_accum[nest_id, spu_id]`. NEVER cache via local variable across calls (the dispatch broadcast loop calls the same handler many times with different `(nest_id, spu_id)` — local cache would cross streams).

### SPR routing via `spr.py`

```python
# riscv/gtx/spr.py
def wr_spr(mem: GtxMemory, loop: GtxLoop, addr: int, value: int) -> None:
    if   GSPR_BASE <= addr <= GSPR_END:  mem.gspr[addr] = value
    elif NSPR_BASE <= addr <= NSPR_END:  mem.nspr[loop.current_nest()][addr] = value
    elif LSPR_BASE <= addr <= LSPR_END:  mem.lspr[loop.current_nest()][loop.current_spu()][addr] = value
    # else: silently ignore — matches C++ default behaviour
```

---

## 5. Reference Cross-Checking (Shadow Mode)

**Recommendation: NO online shadow run against libgtx_npu.so.**

Reasons:
1. **PROJECT.md explicitly excludes** "C++ libgtx_npu.so를 wheel에 동봉하기" (Out of Scope). Bundling for shadow mode would violate the wheel-distribution simplicity goal.
2. The verification objective is **DDR result equality post-firmware**, not lockstep instruction tracing. SystemC HW already gives ULP-equality with C++ libgtx_npu.so per PROJECT.md. So the Python port only needs to match libgtx_npu.so DDR output — done **offline** by `verify.py`.
3. Online shadow would 2× the runtime AND introduce wheel/dev-environment divergence (libgtx_npu.so build needs autoconf'd Spike).

**Recommended verification flow:**

```
Python NPU run → result.hex
                   │
                   ▼ verify.py --fp16 --ulp 1 --atol 0.001
C++ NPU run → golden.hex
```

Cross-checking happens in `tests/gtx/test_regression.py` (described in §10), which:
1. Has pre-recorded golden hex files in `tests/gtx/data/` (committed, generated once from C++).
2. Runs Python NPU on each .elf.
3. Compares result.hex vs golden.hex with `verify.py` logic.

For ad-hoc debugging during op porting, `tests/gtx/test_op_*.py` directly imports `verify_ref.py` (which is already pure NumPy — see §10) for per-op scalar oracles. **No libgtx_npu.so needed for ongoing development**, only for re-generating golden when the spec changes.

The only caveat: keep a `vendor/gtx_cpp_reference/` source snapshot (per PROJECT.md GTX-REF-01) so a developer CAN build libgtx_npu.so on demand to regenerate golden after spec changes — but it's not in the wheel, not on the dev's hot path.

---

## 6. Disassembly Registration

### Pattern: mirror `xhuimt`'s `get_disasms()` delegate-and-collect

```python
# riscv/gtx/disasm.py
from riscv.disasm import disasm_insn_t
from riscv.isa import arg

@arg
def gtx_xrd(insn): return _xpr_name(insn.rd)

@arg
def gtx_xrs1(insn): return _xpr_name(insn.rs1)

@arg
def gtx_xrs2(insn): return _xpr_name(insn.rs2)

def _add_r(insns: list, name: str, funct7: int) -> None:
    match = (funct7 << 25) | 0x0b
    mask  = (0x7f << 25) | 0x7f
    insns.append(disasm_insn_t(name, match, mask, [gtx_xrd, gtx_xrs1, gtx_xrs2]))

def _add_rf3(insns: list, name: str, funct7: int, funct3: int) -> None:
    match = (funct7 << 25) | (funct3 << 12) | 0x0b
    mask  = (0x7f << 25) | (0x07 << 12) | 0x7f
    insns.append(disasm_insn_t(name, match, mask, [gtx_xrd, gtx_xrs1, gtx_xrs2]))

def build_disasm_table() -> list[disasm_insn_t]:
    insns = []
    # Direct port of gtx_npu_disasm.inc
    _add_rf3(insns, "mm_s",  0x00, 0)
    _add_rf3(insns, "mm_o",  0x00, 1)
    _add_rf3(insns, "mm",    0x00, 2)
    _add_rf3(insns, "mm_v",  0x00, 3)
    _add_rf3(insns, "mm_t",  0x00, 7)
    # ... ~140 entries total
    return insns
```

```python
# riscv/gtx/npu.py
class GtxNpu(isa.ROCC):
    def get_disasms(self, proc):
        return build_disasm_table()
```

Note: `riscv.isa.arg` decorator (existing, isa.py:73-86) returns an `arg_t` instance — perfect for our use. Define each operand formatter once at module level (as singletons), pass references into the disasm table just like `gtx_xrd`/`gtx_xrs1`/`gtx_xrs2` are static structs in C++.

### What about `get_instructions()`?

Default behaviour from `rocc_t` returns 4 stubs (one per custom0/1/2/3 opcode) — visible in the test pattern (`test_extension.py` expects `n_insn=4` for `dummy_rocc`). For GTX we override `custom0/1` and **leave `custom2/3` returning 0** (default). This means `len(get_instructions()) == 4` is preserved without us having to override it. **Don't override `get_instructions()`** — the disasm table provides human-readable names; spike's RoCC dispatch uses opcode-level routing.

---

## 7. CSR Exposure

**Recommendation: do NOT expose SPRs via `get_csrs()`. Keep them as internal `GtxMemory` state, accessed via `wr_spr` / `rd_spr`.**

Reasons:
1. **Address space mismatch.** Standard RISC-V CSRs are 12-bit (0x000–0xFFF) and accessed via `csrrw`/`csrrs`/`csrrc`. GTX SPRs are NPU-internal, accessed via the WRSPR/RDSPR custom-0 instructions (funct7=0x00/0x01) — they're parameters to a coprocessor, not CPU state.
2. **Volume.** 1024 GSPRs + 1024 NSPRs × 4 + 1024 LSPRs × 16 × 4 = ~70K possible SPRs. Wrapping each as a `csr_t` is overkill and bloats the spike CSR registry.
3. **No firmware uses `csrrw` to access them.** The xgtxnpu assembler emits `wrspr` (custom0 funct7=0x00), not `csrw`.
4. **C++ source uses `unordered_map<uint16_t, uint64_t>` directly** — no `csr_t` wrapper. Mirroring this is bit-exact-correct.

Stage state in `GtxMemory`. If a future debugging need arises (e.g., GDB integration to inspect NPU SPRs), a thin `csr_t` shim CAN be added later for select diagnostic registers without changing the operational path.

---

## 8. Error Handling

**Match C++ behaviour exactly. Bit-exactness extends to fault behaviour.**

### Decision matrix

| Event | C++ behaviour | Python behaviour | Where |
|---|---|---|---|
| Unknown funct7 (custom0 default branch) | `GTX_TRACE("ERROR..."); illegal_instruction(*p); return 0;` | Call `proc.illegal_instruction()` (exposed via pybind11), return 0 | `dispatch.py:dispatch_custom0` default |
| Out-of-range L1 access | `off + 1 >= GTX_L0_SIZE` short-circuit return | Same — guard with `if off + 1 >= GTX_L0_SIZE: return` | `ops/vec.py`, `ops/act.py` |
| Bank conflict in L2 (D-03) | `check_bank_conflict()` returns true → block | NumPy doesn't model conflicts in functional mode — leave as no-op (C++ also no-ops outside cycle-accurate builds) | `ops/dma.py` |
| Illegal SPR write (out of GSPR/NSPR/LSPR range) | Silently ignored | Silently ignored (else branch in `wr_spr`) | `spr.py` |
| Context violation (`is_valid_in_context` fails) | `GTX_TRACE; illegal_instruction; return 0;` | Same | `dispatch.py:_dispatch` |
| Python op handler raises | n/a | Caught at pybind11 trampoline boundary (existing pyspike pattern, `riscv_extension.cc:34`); printed to stderr; return 0 | (handled by pyspike) |

**Important:** `illegal_instruction` is exposed in `extension_t::illegal_instruction` (publicized in `py_extension_t` per `riscv_extension.h:188`). For ROCC, pyspike does NOT currently expose it via `py_rocc_t`. **Action item for the porter:** verify `using rocc_t::illegal_instruction;` is reachable from Python; if not, this is a one-line pyspike binding addition (out of scope for the architecture but worth flagging in PITFALLS.md).

### `reset()` parity

```python
# riscv/gtx/npu.py
class GtxNpu(isa.ROCC):
    def reset(self, proc):
        super().reset(proc)
        # GTX-RST-01: sp = 0x80100000
        proc.get_state().XPR.write(2, 0x80100000)
        # FPU enable: mstatus.FS = Initial (01)
        mstatus = proc.get_state().mstatus.read()
        mstatus = (mstatus & ~0x6000) | 0x2000
        proc.put_csr(0x300, mstatus)
        # GTX-DMA-01: ddr init from env
        ddr_init = os.environ.get("GTX_DDR_INIT")
        if ddr_init:
            self.mem.ddr_init_from_file(ddr_init)
        # atexit DDR dump (replaces C++ std::atexit + g_gtx_instance pattern)
        atexit.register(self._atexit_ddr_dump)
```

### WJOIN exit semantics (GTX-RST-01)

C++ `WJOIN` calls `exit(0)` if `GTX_NO_EXIT` unset (firmware infinite-loops otherwise). Python:

```python
# Inside custom1 funct3=0b101 (JOIN) handler
if "GTX_NO_EXIT" not in os.environ:
    sys.exit(0)
```

`sys.exit` raises `SystemExit` which pybind11 propagates back to spike — spike treats it as clean termination. Verified pattern from existing `examples/xhuimt/mylrsc.py` style behaviour.

---

## 9. Build Order & Bit-Exact Validation Checkpoints

PROJECT.md mandates **MM-first**. The phasing below maximises bit-exact validation signal earliest by ordering ops so each phase's regression set strictly extends the prior.

### Phase 0 — Skeleton (no compute)
**Lands:** `riscv/gtx/__init__.py`, `npu.py` (empty `custom0/1/2/3` returning 0), `params.py`, `encoding.py`, `memory.py` (allocations only, no ops), `disasm.py` (full table — needed by spike on init), `loop.py`, `spr.py`.
**Validates:** `pytest tests/gtx/test_lifecycle.py` — extension instantiates, register/find round-trip works, `get_disasms()` returns ~140 entries, `reset()` sets sp=0x80100000.
**Checkpoint:** spike loads the extension and a NOP firmware.elf runs to completion without crash.

### Phase 1 — Memory + DMA (data movement only)
**Lands:** `ops/dma.py` (load, store, copy, mcast_g2s, mcast_s2l, copy_mem), `ddr.py`, `loop.py` full P/S/T state machine, `dispatch.py` Mode 3 (P+S DMA).
**Validates:** Per-op test using `verify_ref.py`-style oracle; one synthetic firmware that does DDR→L2→L1→L2→DDR round-trip and dumps DDR. Verify with `verify.py` against C++.
**Checkpoint:** **DMA round-trip preserves bytes exactly** (including `GTX_DDR_REVERSED` mode). Without this, all subsequent ops will silently fail on input mismatch.

### Phase 2 — MM core (the value driver) ⭐
**Lands:** `ops/mm.py` complete (`exec_mm`, `_s`, `_o`, `_v`, `_t`, both MM and MMC variants), `firmware_mm_op` path (custom0 funct7=0x00/0x01 rs1≠0 disambiguation), `mxe_accum` plumbing in `GtxMemory`.
**Validates:** `tests/gtx/test_op_mm.py` — small (16×16×16, 32×16×32) GEMM in FP16 vs `verify_ref.py.MM_VS()` oracle. Then full GEMM regression .elf vs golden hex.
**Checkpoint:** **One full-stack firmware regression passes bit-exact.** This is the strongest possible signal — if MM is right, the SPR/dispatch/DMA plumbing was right too. Per PROJECT.md GTX-MM-01 this is "NPU 핵심" and other ops indirectly depend on operand staging which MM exercises.

### Phase 3 — VEC subsystem
**Lands:** `ops/vec.py` (SASMD add/sub/mul/div, FMADD, MIN/MAX, DOT, VSUM, math/sign/round/clamp, vector_imm L0 paths). 30+ ops.
**Validates:** Each op against the matching `verify_ref.py` reference (already NumPy — perfect oracle). Then 2-3 vector firmware regressions.
**Checkpoint:** VSUM matches FP16 with FP32-internal-accumulate convention (PROJECT.md "VSUM은 FP32 내부 누적 후 1회 FP16 변환").

### Phase 4 — ACT + POOL + CONV + format conversion
**Lands:** `ops/act.py` (RELU, SOFTMAX, ESUM, PRELU, GELU, TANH, SIGM — including direction reversal per PROJECT.md), `ops/pool.py` (2D windowed), `ops/conv.py` (IM2COL_N/D), `ops/format.py` (FP8/INT8/INT32/FP32/FP64 ↔ FP16), `ops/tpose.py`.
**Validates:** Per-op vs `verify_ref.py`; activation regression.elf passes bit-exact. **PRELU/GELU/TANH/SIGM direction (ADDRR→ADDRA) is the most likely bug source** — flag prominently.
**Checkpoint:** Full activation/conv firmware passes.

### Phase 5 — Microcode & sync
**Lands:** `ops/mexec.py` (DDR microcode fetch via `proc.get_mmu()`), credit counters (`credit_ld`/`credit_st`), barriers (BAR/WAIT/MBAR/MSYNC/EOM/HALT), CPSVR/MVSVR/OPSET.
**Validates:** Full firmware regression suite (`run_tests_n1s16.sh` equivalent).
**Checkpoint:** **GTX-FW-01: 100% of existing .elf regressions pass bit-exact.** Ship gate.

### Phase 6 — Packaging & polish
**Lands:** wheel inclusion of `data/` regression assets, `pyspike` CLI integration test (`pyspike --extlib=riscv.gtx fw.elf`), examples documentation.
**Validates:** `pip install` of built wheel into clean venv → `python -c "from riscv.gtx import GtxNpu"` works → CLI runs an .elf end-to-end.

---

## 10. Test Architecture

### Directory layout

```
tests/
├── conftest.py                    # existing — provides mock_sim
├── gtx/                           # NEW
│   ├── conftest.py                # GTX-specific fixtures: gtx_npu, gtx_mem, fixture_elf
│   ├── test_lifecycle.py          # Phase 0 — register/find/reset/disasm count
│   ├── test_memory.py             # GtxMemory view discipline (no copies on slice)
│   ├── test_loop.py               # P/S/T state machine, context computation
│   ├── test_spr.py                # GSPR/NSPR/LSPR routing
│   ├── test_disasm.py             # Disasm table count + sample decode
│   ├── test_op_mm.py              # Per-op MM oracle vs verify_ref
│   ├── test_op_vec.py             # Per-op vector oracle vs verify_ref
│   ├── test_op_act.py             # Per-op activation oracle vs verify_ref
│   ├── test_op_dma.py             # DMA round-trip
│   ├── test_op_format.py          # FP8/INT8/FP32/FP64 ↔ FP16
│   ├── test_regression_fw.py      # Full firmware .elf regression with golden hex
│   └── data/
│       ├── golden/                # Pre-recorded golden DDR hex (committed)
│       │   ├── mm_basic_n1s16.hex
│       │   ├── activation_relu_n1s16.hex
│       │   └── ...
│       └── elf/                   # Test firmware .elf files (committed)
│           ├── mm_basic.elf
│           ├── activation_relu.elf
│           └── ...
```

### Pytest discovery & fixture pattern

`pytest_asyncio` is already configured with session scope (per `pyproject.toml:163`). Add a session-scoped fixture for the .elf data dir:

```python
# tests/gtx/conftest.py
import pathlib, pytest
from riscv.gtx import GtxNpu
from riscv.gtx.memory import GtxMemory

@pytest.fixture(scope="session")
def gtx_data_dir():
    return pathlib.Path(__file__).parent / "data"

@pytest.fixture
def gtx_mem():
    """Fresh memory per test — avoids cross-test pollution."""
    return GtxMemory()

@pytest.fixture
def gtx_npu_inst(mock_sim):
    """Fresh NPU instance bound to mock_sim's hart 0."""
    p = mock_sim.get_core(0)
    p.reset()
    npu = GtxNpu()
    npu.reset(p)
    return npu, p
```

Per `pyproject.toml:154-160` `pythonpath` includes `src/main/python`, so `from riscv.gtx import ...` works without further config. **No conftest hacks needed.**

### Per-op oracle pattern (ULP tolerance)

Reuse `gtx/verify_ref.py` directly — it's already pure NumPy. Add it to `tests/gtx/data/oracle.py` as a wrapper:

```python
# tests/gtx/test_op_vec.py
import numpy as np
from tests.gtx.data import oracle  # mirrors verify_ref.py
from riscv.gtx.memory import GtxMemory
from riscv.gtx.ops.vec import exec_vec_scalar
from riscv.gtx.encoding import GTX_IMM_ADD

def test_add_vs_basic(gtx_mem):
    # Setup
    n, s = 0, 0
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    scalar = np.float16(5.0)
    gtx_mem.l1_f16(n, s)[0:4] = a
    gtx_mem.lspr[n][s] = {LSPR_SPM_ADDRA: 0, LSPR_SPM_ADDRR: 8}  # ADDRA=0, ADDRR=8 bytes

    # Execute
    exec_vec_scalar(gtx_mem, n, s, GTX_IMM_ADD, vector_size=4,
                    scalar_val=int(scalar.view(np.uint16)), scalar_val2=0)

    # Verify
    expected = oracle.ADD_VS(a, scalar)  # NumPy reference
    actual = gtx_mem.l1_f16(n, s)[4:8].copy()  # ADDRR=8B → fp16 idx 4
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    # Bit-exact equality on uint16 — no ULP needed for scalar add
```

### ULP comparison for golden hex (regression tests)

```python
# tests/gtx/test_regression_fw.py
import subprocess, pathlib, pytest
from .verify_lib import compare_hex  # extracted from gtx/verify.py

REGRESSIONS = [
    ("mm_basic.elf",        "mm_basic_n1s16.hex"),
    ("activation_relu.elf", "activation_relu_n1s16.hex"),
    # ...
]

@pytest.mark.parametrize("elf,golden", REGRESSIONS, ids=[e for e, _ in REGRESSIONS])
def test_firmware_regression(elf, golden, gtx_data_dir, tmp_path):
    elf_path = gtx_data_dir / "elf" / elf
    golden_path = gtx_data_dir / "golden" / golden
    result_path = tmp_path / "result.hex"

    # Run pyspike with GtxNpu extension
    subprocess.run([
        "pyspike", "--extlib=riscv.gtx", str(elf_path),
    ], env={
        **os.environ,
        "GTX_DDR_DUMP": str(result_path),
        "GTX_DDR_DUMP_ADDR": "0x37f000000",
        "GTX_DDR_DUMP_SIZE": "0x400",
    }, check=True)

    # Compare with ULP=1, atol=0.001 (matches PROJECT.md success criterion)
    diffs = compare_hex(result_path, golden_path, fp16=True, ulp=1, atol=0.001)
    assert diffs == [], f"Regression failed: {diffs[:5]}"
```

`verify_lib.py` is `gtx/verify.py` repackaged as importable functions instead of `__main__`. One-time port — bundle into `riscv/gtx/_verify.py` so it ships in the wheel and tests import it as `from riscv.gtx._verify import compare_hex`.

### .elf asset bundling

In `pyproject.toml`, add to `[tool.setuptools.package-data]`:
```toml
[tool.setuptools.package-data]
"riscv.gtx" = ["data/golden/*.hex", "data/elf/*.elf"]
```

This makes assets accessible at runtime via `importlib.resources.files("riscv.gtx") / "data"` after `pip install`. Per GTX-PKG-01 requirement.

---

## Open Questions / Action Items for Roadmap

1. **BE/LE inconsistency in C++ source** — port should pick LE internally and document. (See §2.)
2. **`illegal_instruction` exposure on `py_rocc_t`** — verify reachable from Python; may need pyspike binding patch. (See §8.)
3. **`get_instructions()` default — keep at 4 (default)** — verified compatible with `test_extension.py` expectation. No override needed.
4. **`riscv.dev.MMIO` integration** — out of scope for v1 (PCIe-EP excluded), but future re-add via `riscv.gtx.dev.PcieEP(MMIO)` slots cleanly into existing `examples/amba/uart_lite.py` pattern.
5. **Performance target** — PROJECT.md says "한 세션 내(≤ 수십 분 수준)". MM at 32×128×32 FP16 in pure NumPy is ~1ms/call. Full firmware regression with 10K MM calls = 10s. Comfortable margin. **No need for cython/C++ in v1.** Re-evaluate at Phase 5.

## Summary: Conventions to Mirror from Existing Pyspike

| Pyspike convention | GTX adoption |
|---|---|
| `examples/xhuimt/__init__.py` delegates to sub-modules via `*self.lrsc.get_instructions()` | `riscv/gtx/npu.py` delegates to `build_disasm_table()` from `disasm.py` |
| `@isa.register("name")` decorator | `@isa.register("gtx_npu")` on `GtxNpu` class in `npu.py` |
| `MyLRSC` class encapsulates a sub-feature | `GtxMemory`, `GtxLoop` encapsulate orthogonal concerns |
| Tests under `tests/test_<feature>.py` | `tests/gtx/test_<area>.py` (subdirectory because of volume) |
| `pyproject.toml:pythonpath` exposes `src/main/python` | Already covers `riscv.gtx` automatically — no change |
| `extension_t.get_csrs(proc) → []` default | Don't override (§7) |
| pybind11 `processor_t` / `rocc_insn_t` passed as-is | Read regs via `proc.get_state().XPR[insn.rs1]` exactly like C++ workaround for xs1=0 (PROJECT.md noted footgun) |

---

*Architecture research: 2026-05-04*
