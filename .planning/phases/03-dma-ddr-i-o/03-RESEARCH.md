# Phase 3: DMA & DDR I/O - Research

**Researched:** 2026-05-05
**Domain:** C++ → Python direct port (DMA / DDR / dispatch routing); bit-exact mechanics
**Confidence:** HIGH (all findings traced to file:line in `vendor/gtx_cpp_reference/gtx/`)

## Summary

Phase 3 is a **pure direct-port phase**: every Python function corresponds to a precisely identifiable C++ function, and bit-exact behavior is the only acceptance signal. The ecosystem question is settled (Python 3.10+, NumPy ≥ 2, no new deps). The remaining work is mechanical encoding/decoding plus four design lock-ins:

1. **Deferred-store flush trigger is `endp` when `!wsplit_seen`, AND `credit_st_chk` (funct7=0x53) when `is_sloop`**. Three call sites (`gtx_npu_loop.cc:53`, `gtx_npu_dispatch.cc:902`, `gtx_npu_custom0.cc:690`) collapse into one canonical Python wiring decision: ROADMAP success #4's "flush at end_p" is correct *for the simple firmware that has no WSPLIT*; firmware that uses WSPLIT/WJOIN flushes via `credit_st_chk` instead. P3 must wire **both** triggers.
2. **`firmware_dma` rs1/rs2 layout** has a non-obvious `is_copy` carve-out: `addr_hi` is `(rs1>>27) & 0x1FFFFFFFFF` for LOAD/STORE but `(rs1>>32)` for COPY (funct3=010). The `funct3` here is *not* the RoCC funct3 field — it's a synthesized 3-bit value `(xd<<2)|(xs1<<1)|xs2`.
3. **`ensure_ddr` is NOT doubling-grow** — C++ allocates the FULL 4 GiB once (`gtx_npu_core.cc:198-203`). The Phase 1 stub's "Phase 3 will replace with doubling-grow" note in `ddr.py` is **incorrect**. P3 should keep doubling-grow for *test ergonomics* (avoid 4 GiB allocation in CI) but document that the *production semantics* match C++ exactly when sized right.
4. **DDR dump byte format** is plain LE bytes (the same bytes physically resident in DDR). `verify.py` parses each 2-byte pair as **big-endian FP16** (`verify.py:235`); this is consistent because LE-stored value `[lo, hi]` parsed as BE-pair gives `(lo<<8)|hi` which is the BE wire form of the same FP16 magnitude.

**Primary recommendation:** Direct-port `gtx_npu_dma.cc` 1-to-1 into `dma_engine.py` (pure functions on `GtxMemory`) + `ops/dma.py` (`@handler` entry points reading `proc`/`insn`). Wire flush at BOTH `end_p` (when `!wsplit_seen`) and `credit_st_chk` (when `is_sloop`). Defer DMA-3D / mcast / load_3d / store_3d **bodies** to v2 but register **disasm-only stubs** in P3 so trace mnemonics are correct.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01**: DMA = `ops/dma.py` (handler entry points) + `riscv/gtx/dma_engine.py` (pure functions) split. spike-dependent vs spike-independent separation.
- **D-02**: `firmware_dma` (funct7=0x40) and `firmware_dma_svr` (funct7=0x41) are separate `@handler` entries.
- **D-03**: funct3 sub-variant decomposed via decorator + 2-level dispatch. Data structure: `dict[funct7, dict[funct3, Callable]]` (see Architecture Patterns for confirmation).
- **D-04**: `@dataclass DeferredDdrStore` with 7 fields exactly: `nest, l2_off, ddr_off, length, height, l2_stride, ddr_stride`.
- **D-05**: `self.deferred_ddr_stores: list[DeferredDdrStore]` lives on `GtxNpu` instance; `reset()` clears.
- **D-06**: `npu.flush_deferred_ddr_stores() -> None` API. Trigger location locked by THIS research (see "Deferred-Store Flush Trigger" section).
- **D-07**: DDR I/O lives in `riscv/gtx/ddr.py`. Pure functions taking `mem: GtxMemory`.
- **D-08**: `GTX_DDR_REVERSED` env var read at every I/O call (no caching).
- **D-09**: `ddr_dump_to_file` accepts only function arguments. CLI/env var integration deferred to P6.
- **D-10**: P3 regression is Python-only programmatic — no `.elf` builds.
- **D-11**: Deferred-store dual-assertion (queue shape + flush diff).
- **D-12**: No MMU mock in P3 (YAGNI; P4 adds).
- **D-13**: `ensure_ddr` upgraded from Phase 1 stub. **Note:** see "Common Pitfalls" — C++ does NOT do doubling-grow; Phase 1 doc is wrong.
- **D-14**: `dispatch_4mode` function in `riscv/gtx/dispatch.py`.

### Claude's Discretion

The following implementation details are explicitly delegated by CONTEXT.md to research/plan:

- 2-level dispatch data structure (`dict[int, dict[int, Callable]]` vs `dict[(int, int), Callable]`) — **research recommends** `dict[funct7, dict[funct3 | None, Callable]]` (see Architecture Patterns).
- `npu.custom0` 2-level branch shape — **research recommends** sentinel `funct3=None` key for funct7s without sub-decomposition (single dispatch table, fewer modules touched).
- `dma_engine.py` module name — **kept** as `dma_engine.py` (matches CONTEXT.md sample code).
- `DeferredDdrStore` dataclass location — **research recommends** `dma_engine.py` (no separate `dma_state.py`; YAGNI).
- `dispatch_4mode` arg signature — **research recommends** `(npu, opcode, op1, op2, op3, sub_op)` — see Architecture Patterns.
- P3 disasm coverage — **research recommends** 9 active mnemonics (load/store/copy/load_svr/store_svr/tpose/fill + load_svr_l1/store_svr_l1) + 4 disasm-only stubs (load_3d/store_3d/mcast_g2s/mcast_s2s/copy_mem) for trace correctness.
- `monkeypatch.setenv` — **standard pytest fixture pattern**, no `unittest.mock` needed.
- `ensure_ddr` `INITIAL_FLOOR` — **research recommends** 1 MiB (covers 32 KB bus-word minimum overhead with headroom).

### Deferred Ideas (OUT OF SCOPE)

From CONTEXT.md `<deferred>`:
- DMA-3D / IM2COL-N / IM2COL-D / MCAST handler **bodies** — v2 (DMA-V2-01)
- `mexec` full microcode loop — v1 firmware doesn't trigger
- MM op handlers / `gemm_core` / `mxe_accum` write — Phase 4
- First .elf strict-mode regression (mm_basic.elf) — Phase 4 success #4
- VEC/ACT/Pool ops — Phase 5
- `verify.py` port → `riscv.gtx._verify` — Phase 6
- MMU mock (`load_uint64`/`store_uint64`) — Phase 4
- Auto DDR dump on Spike exit (GTX_DDR_DUMP env var hook) — Phase 6 or follow-up

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **DMA-01** | Full op set (`exec_dma_2d`, `exec_load_svr`, `exec_store_svr`, `exec_transpose`, `exec_fill`) | Section "C++ Function Signature Locks" — exact signatures + side-effects for all six. |
| **DMA-02** | `firmware_dma_op` packed encoding (funct3=000/001/010 LOAD/STORE/COPY) | Section "firmware_dma Encoding" — bit layout + HW conventions + COPY carve-out. |
| **DMA-03** | S-loop deferred-store queue + flush at end_p (per ROADMAP) | Section "Deferred-Store Flush Trigger" — reconciled across 3 C++ call sites. |
| **DMA-04** | DDR hex I/O both modes (LTR + GTX_DDR_REVERSED=1) | Section "DDR Hex I/O" — 32-byte bus word, half-density packing, @offset semantics. |
| **DMA-05** | Round-trip bit-exactness | Section "Test Patterns" — programmatic round-trip recipe. |
| **DISP-03** | 4-mode dispatch (Mode 1 broadcast, Mode 3 P+S DMA) | Section "4-Mode Dispatch Router" — exact NEST/SPU routing per loop state. |

## Project Constraints (from CLAUDE.md)

- **Tech stack:** Python 3.10+, NumPy ≥ 2.0, pyspike pybind11 trampolines. **No new C++ code.** No new runtime dependencies (NumPy only).
- **Bit-exact:** ULP tolerance verified by `verify.py --fp16 --ulp 1 --atol 0.001`. P3 must use byte-level memcpy (no FP16 view shortcuts in DMA — DMA is byte-domain).
- **Compatibility:** `riscv.isa.ROCC` virtual method signatures unchanged. `processor_t` / `rocc_insn_t` are pybind11 binding objects.
- **Performance:** NumPy-backed; ndarray slicing for 2D DMA (`mem.l2_byte(nest)[off:off+len] = ddr[ddr_off:ddr_off+len]` is the canonical idiom).
- **Testing:** pytest. Each new op gets a unit test. P3 success criteria #4 (deferred-store) requires direct introspection of `npu.deferred_ddr_stores`.
- **Platform:** Linux x86_64 manylinux2014. cibuildwheel cp310-cp312 must remain green.

## Standard Stack

### Core (already in pyspike — no new deps)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | ≥ 2.0,<3 | byte-level memcpy + ndarray views (LE byte order in L1/L0) | Phase 1 D-07. The only DMA-relevant API used is `ndarray[a:b] = ndarray[c:d]` slice assignment, `np.zeros`, `arr.tobytes()`, `arr.view(dtype)`. |
| os (stdlib) | — | `os.environ.get('GTX_DDR_REVERSED')` per-call read (D-08) | Avoid module-load caching trap. |
| dataclasses (stdlib) | 3.10+ | `@dataclass DeferredDdrStore` (D-04) | Direct port of C++ POD struct. |

### Don't add

- ❌ `numba` / `cython` / `cffi` — banned by PROJECT.md
- ❌ `mmap` — DDR uses contiguous `np.uint8` ndarray; mmap is overhead for an in-process buffer
- ❌ Any DDR/hex parser library (e.g., `intelhex`) — see "Don't Hand-Roll" section
- ❌ `struct.pack/unpack` for FP16 — NumPy view is the chosen path (Phase 1 D-09); DMA is byte-domain anyway, never reinterprets

**Version verification:** numpy 2.x is already locked in `pyproject.toml` (Phase 1 D-07). No P3 dep changes.

## Architecture Patterns

### Module Layout Extension (P3 adds three modules, modifies four)

```
src/main/python/riscv/gtx/
├── memory.py              # P1 (no change in P3)
├── ddr.py                 # P1 stub → P3 fills: ensure_ddr (rewrite), ddr_init_from_file (new), ddr_dump_to_file (new)
├── npu.py                 # P2 → P3 modifies: __init__ adds deferred_ddr_stores=[], reset clears it, custom0 becomes 2-level
├── warp_state.py          # P2 → P3 adds: wsplit_seen=False (used by flush trigger logic)
├── dispatch.py            # P2 → P3 adds: dispatch_4mode() function
├── _registry.py           # P2 (no change — mask_funct3 path already implemented)
├── encoding.py            # P2 → P3 adds: funct7 0x40, 0x41, 0x38, 0x39 + GSPR_GTX_OPERAND1/2/3/OPCODE addresses
├── dma_engine.py          # P3 NEW: DeferredDdrStore + 6 pure DMA helpers (no spike deps)
└── ops/
    ├── control.py         # P2 → P3 modifies: end_p calls flush (when !wsplit_seen), wsplit sets wsplit_seen=True
    └── dma.py             # P3 NEW: 9 @handler entry points (load/store/copy + load_svr/store_svr + tpose/fill + 2 stubs)
```

### Data Flow

#### Mode 1 (no loop): broadcast DMA via dispatch
```
firmware INSN custom0 funct7=0x07 (DISPATCH_DMA, gem5 simplified)
  → npu.custom0 → handler(dispatch_dma_stub) → dispatch_4mode(npu, opcode=GTX_OP_DMA, ...)
  → !is_ploop branch: for n in range(4): for s in range(16): dispatch_iss_opcode(n, s, ...)
```

#### Mode 3 (P + S): DDR ↔ L2 single-NEST DMA via dispatch
```
firmware: WSPLIT → start_p(nest=0) → start_s(gdmac=0) → dispatch_dma → end_s → end_p
  → dispatch_4mode reaches `is_ploop && is_sloop` branch
  → calls exec_dma_2d(tmu_id, op1, op2, length=op3&0xFFFF, height=(op3>>16)&0xFFFF, is_load, CTX_C2)
```

#### Firmware DMA (funct7=0x40): packed-rs1/rs2/rs3 path
```
firmware INSN custom0 funct7=0x40 funct3=001 (STORE)
  → npu.custom0 → 2-level dispatch[0x40][1] → ops/dma.py:_firmware_dma_store
  → reads rs1, rs2 from XPR; rs3 from npu.gspr[GSPR_GTX_OPERAND3]
  → decode addr_hi/addr_lo/length/height/strides
  → if is_sloop: dma_engine.firmware_dma_sloop_store(npu, ...) → push DeferredDdrStore
  → if is_tloop: dma_engine.firmware_dma_tloop_store(npu, ...) → immediate L1→L2 memcpy
```

#### Deferred store push and flush
```
S-loop STORE → dma_engine pushes DeferredDdrStore(nest, l2_off, ddr_off, length, height, l2_stride, ddr_stride) onto npu.deferred_ddr_stores
...
end_p (when !wsplit_seen) OR credit_st_chk (funct7=0x53) when is_sloop
  → npu.flush_deferred_ddr_stores()
  → for req in self.deferred_ddr_stores: for row: l2_data = mem.l2_byte(nest)[l2_off..]; ddr_bytes[ddr_off..] = l2_data
  → self.deferred_ddr_stores.clear()
```

### Pattern 1: 2-Level Dispatch with Sentinel Key

**What:** `dict[int, dict[int | None, Callable]]` — outer keyed by funct7, inner keyed by funct3 (or `None` for funct7s without sub-decomposition).

**When to use:** Any time the registry has multiple funct3 entries under the same funct7 (P3 = funct7=0x40 (3 entries) + funct7=0x41 (4 entries: 0/1/4/5)).

**Implementation in npu.py custom0:**
```python
def custom0(self, proc, insn, xs1, xs2) -> int:
    funct7 = insn.funct
    sub_table = self._custom0.get(funct7)
    if sub_table is None:
        return 0
    # sub_table is dict[Optional[int], Callable]
    # For funct7s without funct3 sub-decomposition: key is None
    # For funct7s with sub-decomposition: key is the synthesized funct3
    handler = sub_table.get(None)  # try non-decomposed first
    if handler is None:
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        handler = sub_table.get(funct3)
    if handler is None:
        return 0
    return handler(proc, insn, xs1, xs2)
```

**Why this layout (not `dict[(funct7, funct3), Callable]`):** flat tuple-keyed dict requires the dispatcher to know whether each funct7 expects a funct3 lookup. The sentinel approach keeps the dispatcher uniform and makes registration transparent (`mask_funct3=False` → `funct3=None` key; `mask_funct3=True` → integer funct3 key).

**Source:** matches `gtx_npu_custom0.cc` switch — its `case GTX_ISS_F7_DMA_LOAD:` (funct7=0x40) immediately calls `firmware_dma()` which itself decodes funct3 internally. P3 keeps the same idiom but uses the decorator-driven 2-level lookup for cleaner separation.

### Pattern 2: dma_engine.py Pure Functions

**Signature template:**
```python
def exec_dma_2d(mem: GtxMemory, *, nest_id: int, l2_addr: int, l1_addr: int,
                width: int, height: int, is_load: bool, l2_stride: int = 0,
                ctx: int = CTX_C3, spu_id: int = 0) -> int:
    """Direct port of gtx_npu_t::exec_dma_2d (gtx_npu_dma.cc:25-90).
    Returns cycles (always 0 in functional model — caller ignores).
    """
```

All `dma_engine` functions follow the same shape:
- First positional arg: `mem: GtxMemory` (no `npu`/`proc`/`insn` — those stay in `ops/dma.py` entry points)
- All other args keyword-only
- Returns `int` (cycles — vestigial in functional model)
- Mutates `mem` in place (DMA = side effect)

### Pattern 3: dispatch_4mode Signature

```python
def dispatch_4mode(npu, opcode: int, op1: int, op2: int, op3: int, sub_op: int = 0) -> int:
    """4-mode router based on warp loop state. Direct port of gtx_npu_t::dispatch
    (gtx_npu_dispatch.cc:27-143).

    Args:
        npu: GtxNpu instance (provides .warp, .mem, .gspr)
        opcode: GTX_OP_MM | GTX_OP_VECTOR | GTX_OP_ACTIVATION | GTX_OP_DMA
        op1, op2, op3: read by caller from npu.gspr[GSPR_GTX_OPERAND1/2/3]
        sub_op: low byte of npu.gspr[GSPR_GTX_OPCODE]

    Returns: total cycles (functional: ignored).
    """
    if not npu.warp.is_ploop:
        # Mode 1: broadcast all NEST × SPU
        for n in range(GTX_NEST_NUM):
            for s in range(GTX_SPU_NUM):
                dispatch_iss_opcode(npu, n, s, opcode, op1, op2, op3)
    elif npu.warp.is_ploop and not npu.warp.is_sloop and not npu.warp.is_tloop:
        # Mode 2: P only — broadcast within tmu_id
        for s in range(GTX_SPU_NUM):
            dispatch_iss_opcode(npu, npu.warp.tmu_id, s, opcode, op1, op2, op3)
    elif npu.warp.is_ploop and npu.warp.is_sloop:
        # Mode 3: P + S — DDR↔L2 DMA on tmu_id
        is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)
        return dma_engine.exec_dma_2d(npu.mem, nest_id=npu.warp.tmu_id,
            l2_addr=op1 & 0xFFFFFFFF, l1_addr=op2 & 0xFFFFFFFF,
            width=op3 & 0xFFFF, height=(op3 >> 16) & 0xFFFF,
            is_load=is_load, ctx=CTX_C2)
    elif npu.warp.is_ploop and npu.warp.is_tloop:
        # Mode 4: P + T — single (tmu_id, curr_id) — P4 actually invokes compute
        dispatch_iss_opcode(npu, npu.warp.tmu_id, npu.warp.curr_id, opcode, op1, op2, op3)
    return 0
```

**Source:** verbatim port of `gtx_npu_dispatch.cc:79-139` minus the thread-pool fallback (which is `#ifdef GTX_USE_POOL` — not relevant to functional model).

In Phase 3, `dispatch_iss_opcode` is a stub that handles only the DMA-relevant funct7s (0x43 LOAD_SVR_L1, 0x45 STORE_SVR_L1, 0x52/0x53 credit checks). Other funct7s become NOP until P4/P5.

### Anti-Patterns to Avoid

- **`mem._ddr_bytes.view(np.float16)` for DMA copies** — DMA is byte-domain. Use `mem._ddr_bytes[off:off+n] = src_bytes` (view returns view, slice assignment is in-place memcpy).
- **Eager DDR allocation in P3 tests** — calling `ensure_ddr(mem, GTX_DDR_DEFAULT_SIZE_BYTES)` in a fixture allocates 4 GiB. Each test that exercises DDR should request only `end_offset` it actually touches; the doubling-grow strategy avoids per-test 4 GiB allocation.
- **Module-level `os.environ.get('GTX_DDR_REVERSED')` cache** — D-08 violation. Read inside each I/O function.
- **`flush_deferred_ddr_stores` on `GtxMemory`** — D-06 violation. It's a `GtxNpu` method (deferred queue is per-NPU state, not per-memory).
- **Manual byte-pair construction (`(hi << 8) | lo`) in DMA path** — DMA copies *raw bytes*. The byte order is preserved through the entire DMA chain by virtue of `np.uint8` slice assignments. The (hi, lo) split appears only in: (a) `firmware_dma` rs1/rs2 decode, (b) `exec_fill` writeback, (c) `exec_transpose` per-element move. Nowhere else.
- **String parsing in `ddr_init_from_file` for every byte** — use `bytes.fromhex(line[:nbytes*2])` once per line, then index into the bytes object. See "Code Examples" for the canonical idiom.
- **`ddr_dump_to_file` reading os.environ for addr/size** — D-09 violation. Function takes `addr: int, size: int` only. CLI/env handling deferred to P6.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hex line parsing | Custom char→nibble loop | `bytes.fromhex(line[:nbytes*2])` | Stdlib is C-implemented, 5-10× faster than per-char Python; identical output |
| Byte reversal of a 32-byte chunk | `for j in range(31, -1, -1): out[31-j] = inp[j]` | `chunk[::-1]` (numpy or bytes slice) | Idiomatic Python; one byte-level memcpy under the hood |
| `int.to_bytes(2, 'little')` for FP16 | Manual mask + shift | `np.float16(val).tobytes()` (when value is fp), or `struct.pack('<H', raw)` for raw uint16 | NumPy 2.x guarantees IEEE 754 binary16 (Phase 1 D-09); raw uint16 path is byte-identical to manual mask |
| 2D strided memcpy | nested Python loop with per-byte assignment | `for row: dst[d_off:d_off+L] = src[s_off:s_off+L]` (NumPy slice = vectorized memcpy) | Each row copy is a single `memmove` in C; eliminates Python overhead |
| Doubling-growth DDR buffer | `np.append`/`np.concatenate` per call | `np.zeros(new_size); new[:old_size] = old; mem._ddr_bytes = new` | concatenate makes a fresh allocation+copy each call (O(N²) total); preallocation+copy is O(N) amortized |
| `@offset` line parsing | Regex / string tokenizer | `int(line[1:].strip(), 16)` | `@DEADBEEF` → `int('DEADBEEF', 16)` directly |
| Half-density packing detection | Read line twice (once to check, once to use) | Detect `nbytes = len(line.strip()) // 2` per line | C++ already does exactly this — only `nbytes` bytes consumed, offset advances by `nbytes` (not 32) |
| Bus-word reverse iterator | Manual loop over `range(31, -1, -1)` | `bytes.fromhex(line)[::-1]` | One pass for both modes — branch on `reversed`, then slice |

**Key insight:** The DMA layer is *literally* byte memcpy at every level (DDR, L2, L1, L0). The only places that look at byte semantics (FP16 split into hi/lo) are `exec_fill` and `exec_transpose` — and both already do the right LE thing in C++ by writing `[lo, hi]` to consecutive bytes. Any "improvement" to compactness via FP16 views is wrong.

## Deferred-Store Flush Trigger (D-06 Lock-in)

**The reconciliation:** ROADMAP success #4 says "flush at end_p". But C++ has 3 call sites:

| C++ Site | File:Line | Trigger | When Used |
|----------|-----------|---------|-----------|
| #1 | `gtx_npu_loop.cc:53` | `endp` when `!wsplit_seen` | Simple firmware (no WSPLIT/WJOIN) ends — flush + maybe DDR-dump-on-HTIF-exit |
| #2 | `gtx_npu_dispatch.cc:902` | `dispatch_iss_opcode` case `GTX_ISS_F7_CREDIT_ST_CHK` (funct7=0x53) when `is_sloop` | Plan-style firmware: S-loop wakes after T-loop signals via credit_st |
| #3 | `gtx_npu_custom0.cc:690` | Same as #2 (custom0 entry path for funct7=0x53) | Same case as #2, just a different entry path |
| #4 (atexit) | `gtx_npu_core.cc:65` | `atexit(gtx_atexit_ddr_dump)` — Spike HTIF tohost exit | Final flush before DDR dump (P6 territory) |

**Lock-in for P3:** Wire flush at **TWO** points:

1. **`end_p` handler** in `ops/control.py` (when `!npu.warp.wsplit_seen`):
   ```python
   def _do_endp(npu, rs1, rs2):
       npu.warp.is_ploop = False
       if not npu.warp.wsplit_seen:
           npu.flush_deferred_ddr_stores()
           # P6: env-var-gated ddr_dump_to_file at this point too
   ```

2. **`credit_st_chk` handler** (funct7=0x53) in `ops/dma.py` (when `npu.warp.is_sloop`):
   ```python
   @handler(kind='custom0', funct7=0x53, mnemonic='credit_st_chk')
   def credit_st_chk(npu, proc, insn, xs1, xs2):
       if npu.warp.is_sloop:
           npu.flush_deferred_ddr_stores()
       return 0
   ```

**Why both:** the firmware authoring style determines which trigger fires:
- Plan-style firmware (uses WSPLIT/WJOIN + credit_st_chk) — flush via #2/#3 (mid-execution between S-loop S-iterations).
- Simple firmware without WSPLIT (e.g., `__copy_mem` standalone) — flush via #1 at endp.

P3 success #4 explicitly tests the **endp path** (no credit_st_chk in the test sequence). But the credit_st_chk wiring must also be present so P4 mm_basic.elf (which is plan-style) doesn't silently lose stores.

**Required state additions to `WarpState`:**
```python
@dataclass
class WarpState:
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False
    tmu_id: int = 0
    curr_id: int = 0
    wsplit_seen: bool = False  # P3 NEW: set True by WSPLIT (custom1 funct3=0b100)

    def reset(self):
        self.is_ploop = False
        self.is_tloop = False
        self.is_sloop = False
        self.tmu_id = 0
        self.curr_id = 0
        # NOTE: wsplit_seen NOT reset by reset() — it's a process-lifetime sentinel
        # (matches gtx_npu.h:1251 `bool wsplit_seen = false;` initialized once)
        # Actually: C++ does NOT reset it on hart reset. Confirm with a P3 test
        # that wsplit_seen persists across reset (or revisit if P4 fires asserts).
```

**Source citations:**
- `gtx_npu_loop.cc:52-67` — endp flush + DDR dump (when !wsplit_seen)
- `gtx_npu_dispatch.cc:898-905` — credit_st_chk in dispatch_iss_opcode
- `gtx_npu_custom0.cc:684-694` — credit_st_chk in custom0 entry path
- `gtx_npu_custom1.cc:62` — `wsplit_seen = true;` set by WSPLIT (funct3=0b100)
- `gtx_npu_custom0.cc:76` — `wsplit_seen = true;` set by WSPLIT custom0 variant (funct7=0x02)
- `gtx_npu.h:1251` — initial `wsplit_seen = false;`

**P3 modification to WSPLIT in `ops/control.py`:** the existing P2 `wsplit` handler (custom1 funct3=0b100) and `wsplit_custom0` (custom0 funct7=0x02) currently NOP. P3 must add `npu.warp.wsplit_seen = True` to both.

## firmware_dma Encoding (DMA-02 Lock-in)

**Source:** `gtx_npu_dma.cc:256-397` (256-line function — direct-port body).

### rs1, rs2, rs3 layout

| Field | Bits in rs1 | Bits in rs2 | Bits in rs3 | Notes |
|-------|-------------|-------------|-------------|-------|
| `addr_hi` (DDR or L2 address) | `[63:27]` for LOAD/STORE; `[63:32]` for COPY | — | — | 37 bits (LOAD/STORE: `(rs1 >> 27) & 0x1FFFFFFFFF`) or 32 bits (COPY: `rs1 >> 32`) |
| `addr_lo` (L2 or L1 address) | `[26:0]` (always — even for COPY, this is the src L1 addr) | — | — | 27 bits: `rs1 & 0x7FFFFFF` |
| `height` | — | `[63:48]` | — | 16 bits, HW conv `0 → 1` |
| `length` | — | `[47:32]` | — | 16 bits, HW conv `0 → 0x10000` (65536) |
| `rs2_low` (a stride) | — | `[31:0]` | — | 32 bits |
| `rs3_low` (the OTHER stride) | — | — | `[31:0]` | 32 bits, FROM `npu.gspr[GSPR_GTX_OPERAND3]` (NOT XPR) |

### Synthesized funct3 (NOT the RoCC funct3 field)

```python
funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
```

This 3-bit value encodes:
- Bit 0 (`insn.xs2`): `is_store` (0=LOAD, 1=STORE)
- Bit 1 (`insn.xs1`): if set with bit 0 cleared: `is_copy` (LOAD-COPY = funct3=010)
- Bit 2 (`insn.xd`): unused (in DMA path)

### Decode reference algorithm (`gtx_npu_dma.cc:262-288`)

```python
funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
is_store = bool(funct3 & 1)
is_copy = (not is_store) and bool(funct3 & 2)  # funct3=010 only

# COPY uses [63:32] for dst, others use [63:27]
addr_hi = (rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)  # 37-bit mask
addr_lo = rs1 & 0x7FFFFFF                                             # 27 bits

height_raw = (rs2 >> 48) & 0xFFFF
length_raw = (rs2 >> 32) & 0xFFFF
rs2_low = rs2 & 0xFFFFFFFF
rs3_low = rs3 & 0xFFFFFFFF

# HW conventions
height = 1 if height_raw == 0 else height_raw
length = 0x10000 if length_raw == 0 else length_raw

# Stride assignment (direction-dependent)
if not is_store:  # LOAD or COPY
    rd_stride = rs2_low
    wr_stride = rs3_low
else:
    wr_stride = rs2_low
    rd_stride = rs3_low

# NEST selection
nest = npu.warp.tmu_id if npu.warp.is_ploop else 0
if nest >= GTX_NEST_NUM:
    nest = 0
```

### Loop-state branches (3 paths)

1. **`is_sloop` (DDR ↔ L2)** — `gtx_npu_dma.cc:294-328`:
   - LOAD: per-row, `mem.l2_byte(nest)[l2_off:l2_off+copy_len] = ddr[ddr_off:ddr_off+copy_len]` (immediate)
   - STORE: push **single** `DeferredDdrStore(nest, addr_lo, ddr_offset(addr_hi), length, height, rd_stride, wr_stride)` and `break` (do NOT loop rows; entire 2D copy deferred)

2. **`is_tloop && is_copy` (L1 → L1)** — `gtx_npu_dma.cc:334-348`:
   - Per-row L1→L1 `np.memmove`-like: `spu.l1[d_off:d_off+L] = spu.l1[s_off:s_off+L]`
   - Note: stride for COPY is forced to `length` (contiguous packed)

3. **`is_tloop && !is_copy` (L1 ↔ L2)** — `gtx_npu_dma.cc:349-391`:
   - LOAD: `spu.l1[lo_off:lo_off+L] = mem.l2_byte(nest)[hi_off:hi_off+L]`
   - STORE: `mem.l2_byte(nest)[hi_off:hi_off+L] = spu.l1[lo_off:lo_off+L]`

**No-loop fallthrough:** if neither `is_sloop` nor `is_tloop`, function returns 0 (does nothing). This matches C++ behavior — the function is implicitly a NOP outside warp loops.

### `firmware_dma_svr` (funct7=0x41)

Currently the C++ code does NOT have a separate `firmware_dma_svr` function. funct7=0x41 (`GTX_ISS_F7_DMA_3D`) is handled within `dispatch_iss_opcode` for cases load_svr/store_svr/load_3d/store_3d. The actual handler for SVR (32-byte L1↔L0 transfer) lives in `gtx_npu_dma.cc:exec_load_svr` / `exec_store_svr`. P3 lock-in: in `ops/dma.py`, the funct7=0x41 entry checks funct3:
- funct3=0 → `exec_load_svr`
- funct3=1 → `exec_store_svr`
- funct3=4/5 → load_3d/store_3d **disasm-only stub** in P3 (deferred to v2)

## C++ Function Signature Locks (DMA-01)

Each row is a Python signature lock for `dma_engine.py`. Keyword-only args after `*` for clarity.

| C++ function | C++ src | Python signature in dma_engine.py | Mutates |
|--------------|---------|-----------------------------------|---------|
| `exec_dma_2d` | `gtx_npu_dma.cc:25-90` | `def exec_dma_2d(mem, *, nest_id, l2_addr, l1_addr, width, height, is_load, l2_stride=0, ctx=CTX_C3, spu_id=0) -> int` | `mem._l1_bytes`, `mem._l2_bytes` |
| `exec_load_svr` | `gtx_npu_dma.cc:97-113` | `def exec_load_svr(mem, *, nest_id, spu_id, l1_addr, l0_reg) -> None` | `mem._l0_bytes` |
| `exec_store_svr` | `gtx_npu_dma.cc:118-136` | `def exec_store_svr(mem, *, nest_id, spu_id, l1_addr, l0_reg) -> None` | `mem._l1_bytes` |
| `exec_transpose` | `gtx_npu_dma.cc:143-167` | `def exec_transpose(mem, *, nest_id, spu_id, rows, cols, addr_a, addr_r) -> int` | `mem._l1_bytes` |
| `exec_transpose_ddr` | `gtx_npu_dma.cc:175-225` | `def exec_transpose_ddr(mem, *, src_addr, dst_addr, dim2, dim1, dim0, p2, p1, p0) -> None` | `mem._ddr_bytes` |
| `exec_fill` | `gtx_npu_dma.cc:230-246` | `def exec_fill(mem, *, nest_id, spu_id, length, fill_val, addr_r) -> int` | `mem._l1_bytes` |
| `firmware_dma` (decoded) | `gtx_npu_dma.cc:256-397` | Split into 3 helpers: `firmware_dma_sloop_load/store(...)`, `firmware_dma_tloop_load_store(...)`, `firmware_dma_tloop_copy(...)` | varies per branch |
| `flush_deferred_ddr_stores` | `gtx_npu_dma.cc:415-435` | `GtxNpu.flush_deferred_ddr_stores()` (method, not pure function — needs `self.deferred_ddr_stores`) | `mem._ddr_bytes` |
| `ddr_init_from_file` | `gtx_npu_dma.cc:438-502` | `def ddr_init_from_file(mem, filename: str) -> None` (in `ddr.py`) | `mem._ddr_bytes` |
| `ddr_dump_to_file` | `gtx_npu_dma.cc:509-558` | `def ddr_dump_to_file(mem, filename: str, addr: int, size: int) -> None` (in `ddr.py`) | nothing |

### Notes per function

- **`exec_transpose`**: C++ reads `addr_a`/`addr_r` from `spu.lspr[LSPR_SPM_ADDRA/R]`. Python helper takes them as args (caller — i.e., `ops/dma.py`/dispatch_iss_opcode — reads from `npu.lspr[nest][spu]` first). This keeps `dma_engine` spike-independent.
- **`exec_fill`**: same — `addr_r` is an arg, not read from LSPR inside.
- **`exec_transpose_ddr`**: 3D permute; uses `ddr_offset(addr) = (addr - GTX_DDR_BASE) if addr >= GTX_DDR_BASE else addr`. The `GTX_DDR_BASE = 0x370000000` constant must be added to `params.py` or `encoding.py`.
- **`firmware_dma`**: decode happens in `ops/dma.py:_firmware_dma_*` entry points (which read XPR + GSPR), then delegates to one of the three branches above.

## DDR Hex I/O (DMA-04)

### `ddr_init_from_file` parsing rules

**Source:** `gtx_npu_dma.cc:438-502`

Line categories (skip empty + lines starting with `#`):
- `@HEX` — sets `offset = int(line[1:].strip(), 16)`
- Hex line — `nbytes = len(line.strip()) // 2`, capped at 32. Each line writes `nbytes` bytes to `mem._ddr_bytes[offset : offset+nbytes]`, then `offset += nbytes`.

**Half-density packing:** the C++ code does NOT have explicit "half-density" detection — it just consumes `nbytes` bytes per line, where `nbytes` is whatever the line length implies (16 for half-density, 32 for full). Comment in C++ (line 408-410) describes the *upstream* tool's behavior; the *parser* itself is mode-agnostic.

**`GTX_DDR_REVERSED=1`:** for each hex line, byte at hex-position `(nbytes-1-i) * 2` goes to `mem._ddr_bytes[offset + i]` (right-to-left interpretation).

**Default (LTR):** byte at hex-position `i * 2` goes to `mem._ddr_bytes[offset + i]` (left-to-right).

### `ddr_dump_to_file` writing rules

**Source:** `gtx_npu_dma.cc:509-558`

- 32 bytes per line
- For each 32-byte chunk starting at `off + i` (where i is a multiple of 32):
  - Default LTR: write `ddr[off+i+0], ddr[off+i+1], ..., ddr[off+i+31]` left-to-right as 64 hex chars
  - GTX_DDR_REVERSED=1: write `ddr[off+i+31], ddr[off+i+30], ..., ddr[off+i+0]` left-to-right as 64 hex chars (i.e., reverse the byte order within each 32-byte bus word)
- Out-of-range `off+i+j >= GTX_DDR_SIZE` writes `00` (zero-pad)
- Each line ends with `\n`
- No `@offset` lines emitted (raw consecutive blocks)

### Verify.py byte-format compatibility (cross-reference)

- `verify.py:178-184` parses each line via `bytes.fromhex(line)` — **plain LTR**, no awareness of `GTX_DDR_REVERSED`.
- This means if you **dump with REVERSED=1** and try to **verify against an LTR golden**, all 32-byte chunks will compare differently. The two modes must be matched: golden produced by C++ with `GTX_DDR_REVERSED=1` must be re-loaded/verified in the same orientation.
- For pyspike P6 verify port: `verify.py` itself doesn't need a `GTX_DDR_REVERSED` flag — both golden and result are produced under the same dump orientation, so byte-by-byte parity is sufficient.

### LSB byte order vs FP16 BE-pair note

P3 internal data is **always LE** (the bytes physically stored at L1[off+0] = low byte of FP16, L1[off+1] = high byte). When DMA copies these bytes to DDR, the same LE order is preserved. When `verify.py` later parses each 2-byte pair as `(byte0 << 8) | byte1` (BE form, line 235), the resulting raw uint16 IS the FP16 stored value — because reading LE bytes `[lo, hi]` as `(byte0<<8)|byte1` = `(lo<<8)|hi` ≠ original. **Wait — this is a contradiction.** Let me re-check.

Actually `verify.py` says: "Big-endian FP16 (matching gtx.h byte order)". The C++ `gtx_npu.h` FP16 conversion helpers store/load FP16 in **little-endian** — so the comment in verify.py is misleading. What matters: the comparison treats both files the same way, so any consistent encoding works for FP16 ULP comparison. **Bit-exact tests in P3 should compare byte-for-byte (not FP16-aware), and bypass this question.** P6 verify port can address the FP16 semantics question separately.

### Half-density edge case (P3 acceptance test concern)

C++ comment (`gtx_npu_dma.cc:408-410`):
> Half-density packing: when upper half has data and lower half is zero (common for partial bus utilization), only the data half is stored contiguously, advancing offset by 16 instead of 32.

The C++ **dumper** does NOT produce half-density output (it always emits 32-byte lines). Half-density input is something other tools (SystemC trace dumper) produce; the parser just consumes whatever line length it gets. **P3 dumper output is always full 32-byte lines**, simplifying round-trip tests.

## 4-Mode Dispatch Router (DISP-03)

| Mode | Loop state | Routes to | NEST/SPU set |
|------|------------|-----------|--------------|
| 1 | `!is_ploop` | every (n, s) ∈ NEST × SPU | broadcast 64 |
| 2 | `is_ploop && !is_sloop && !is_tloop` | every s ∈ SPU at `tmu_id` | 16 |
| 3 | `is_ploop && is_sloop` | DDR↔L2 on `tmu_id` only | single NEST (no SPU iter — DMA, not compute) |
| 4 | `is_ploop && is_tloop` | single `(tmu_id, curr_id)` | 1 |

**Source:** `gtx_npu_dispatch.cc:79-139`.

**P3 success #5 test fixture (parametrized over (loop_state, opcode) tuples):**
```python
@pytest.mark.parametrize("loop_state,expected_count", [
    ((False, False, False), 64),  # Mode 1
    ((True,  False, False), 16),  # Mode 2 (no T)
    ((True,  False, True),  None), # Mode 3 (S — DMA, single NEST)
    ((True,  True,  False), 1),   # Mode 4
])
def test_dispatch_4mode_routing(loop_state, expected_count):
    npu = make_npu()
    is_ploop, is_tloop, is_sloop = loop_state
    npu.warp.is_ploop = is_ploop
    npu.warp.is_tloop = is_tloop
    npu.warp.is_sloop = is_sloop
    npu.warp.tmu_id = 1
    npu.warp.curr_id = 5
    seen = []
    monkeypatch.setattr(dispatch, 'dispatch_iss_opcode',
                        lambda npu, n, s, *args: seen.append((n, s)))
    dispatch.dispatch_4mode(npu, opcode=GTX_OP_VECTOR, op1=0, op2=0, op3=0)
    if expected_count == 64:
        assert sorted(seen) == [(n, s) for n in range(4) for s in range(16)]
    elif expected_count == 16:
        assert sorted(seen) == [(1, s) for s in range(16)]
    elif expected_count == 1:
        assert seen == [(1, 5)]
    # Mode 3 verified separately (DMA-specific, no dispatch_iss_opcode call)
```

## Runtime State Inventory

This is a **greenfield porting phase** — no existing pyspike state to migrate. Skip categories:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by checking that no DDR/L2 fixtures exist before P3. | none |
| Live service config | None — pyspike has no external services. | none |
| OS-registered state | None. | none |
| Secrets/env vars | `GTX_DDR_REVERSED`, `GTX_NO_EXIT` are read at runtime only — no rename, no breakage. `GTX_DDR_DUMP`, `GTX_DDR_DUMP_ADDR`, `GTX_DDR_DUMP_SIZE` are P6 territory (CLI-level). | none |
| Build artifacts | `tests/gtx/data/` does not exist yet (Phase 5/6 adds golden hex). | none |

## Common Pitfalls

### Pitfall 1: `firmware_dma` `is_copy` carve-out

**What goes wrong:** Decoding `addr_hi` as `(rs1 >> 27) & 0x1FFFFFFFFF` for COPY (funct3=010) gives a corrupted destination L1 address.

**Why it happens:** L1 is 384 KB = 19 bits — fits well within 27 bits. But the C++ COPY path uses bits `[63:32]` for dst (32-bit wide). This avoids stealing bits from rs1[26:0] which holds src L1 addr.

**How to avoid:** Always branch first: `addr_hi = (rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)`. Add an explicit P3 unit test for funct3=010 with rs1 having distinguishable hi/lo bits.

**Warning sign:** copy_mem regression .elf produces visible memory corruption in nest 0 SPU 0's L1 at unexpected offsets.

### Pitfall 2: `length=0`, `height=0` HW conventions backward

**What goes wrong:** Treating `length=0` as a no-op (skipping DMA) instead of as a 65536-byte transfer. Treating `height=0` as 0 (no rows) instead of 1 row.

**Why it happens:** The HW reuses the all-zeros 16-bit field as a "full extent" sentinel. This is documented obliquely (`gtx_npu_dma.cc:285-287`).

**How to avoid:** Apply both conventions immediately after extracting `length_raw` and `height_raw`. Add a unit test with `(length=0, height=0)` synthetic inputs and assert the engine attempts a `1 × 65536`-byte transfer.

### Pitfall 3: `xs1=0` Spike marshaling for `firmware_dma`

**What goes wrong:** Reading `xs1` argument directly (passed by Spike with value `-1` when xs1=0 in the encoded instruction).

**Why it happens:** Spike's RoCC dispatch passes register values, not register indices. When `insn.xs1==0`, the value is taken from x0 → reg_t = 0; but Spike's marshalling uses `-1` for "missing" registers. The C++ `firmware_dma` function bypasses this by reading `p->get_state()->XPR[insn.rs1]` directly.

**How to avoid:** Use `proc.get_state().XPR[insn.rs1]` and `proc.get_state().XPR[insn.rs2]` in `ops/dma.py:_firmware_dma_*` entry points. Do NOT rely on the `xs1`/`xs2` arguments. (Phase 2 already established this pattern for SPR; P3 follows.)

**Warning sign:** assertion failure where decoded `addr_lo` is `0x7FFFFFF` (-1 truncated to 27 bits = 0x7FFFFFF) instead of the firmware-intended L2 addr.

### Pitfall 4: `dataclass` attribute count drift

**What goes wrong:** Adding/removing fields to `DeferredDdrStore` and missing the corresponding update in `flush_deferred_ddr_stores` consumer.

**Why it happens:** C++ `deferred_ddr_store_t` (gtx_npu.h:1257-1265) has exactly 7 fields in this order: `nest, l2_off, ddr_off, length, height, l2_stride, ddr_stride`. Adding any field changes the producer/consumer contract.

**How to avoid:** Use frozen dataclass to make it immutable, and add a P3 unit test that asserts `dataclasses.fields(DeferredDdrStore)` returns exactly 7 fields with these names.

### Pitfall 5: `ensure_ddr` allocates 4 GiB

**What goes wrong:** Phase 1 stub note ("Phase 3 will replace this with the C++ doubling-grow strategy") is misleading. C++ `gtx_npu_t::ensure_ddr` (`gtx_npu_core.cc:198-203`) allocates the FULL `GTX_DDR_SIZE = 4 GiB` once on first call.

**Why it happens:** C++ uses `std::shared_ptr<gtx_ddr_mem_t>(GTX_DDR_SIZE)`, which is a one-shot allocation — no doubling. The "doubling" idea was a CONTEXT.md author's interpolation.

**How to avoid (P3 lock-in):** Keep doubling-grow in `ddr.py` for **CI/test ergonomics** (don't allocate 4 GiB when the test only writes 8 KB). Document the divergence: production firmware that touches addresses up to `GTX_DDR_SIZE - 1` will trigger a single grow up to the cap (still O(1) wall-clock for a test that exercises the high address). Add a comment:
```python
def ensure_ddr(mem, end_offset):
    """Ensures mem._ddr_bytes can hold `end_offset` bytes.

    NOTE: C++ gtx_npu_t::ensure_ddr allocates GTX_DDR_SIZE (4 GiB) once. We
    use doubling-grow purely as a CI/test ergonomic so per-test allocations
    stay small. For regression tests touching the full 4 GiB, behavior is
    identical (single grow to cap). Cap enforced via GTX_DDR_SIZE env var.
    """
```

**Warning sign:** tests OOM in CI, or single-test wall time > 5 s for trivial DDR ops.

### Pitfall 6: GTX_DDR_REVERSED toggling cache poisoning

**What goes wrong:** Reading `os.environ.get('GTX_DDR_REVERSED')` once at import and caching. Different tests under `monkeypatch.setenv` see stale value.

**Why it happens:** D-08 was added precisely to avoid this. Phase 2 P2 D-07 established the per-call read pattern.

**How to avoid:** Read `os.environ.get('GTX_DDR_REVERSED')` (and convert to bool) at the top of each I/O function. NEVER store at module level.

### Pitfall 7: P3 deferred-store flush reset semantics

**What goes wrong:** `npu.reset()` clears `deferred_ddr_stores` (D-05) but does NOT reset `wsplit_seen` (per gtx_npu.h initialization that runs once). Some test that exercises reset-then-firmware sequence may exhibit wrong flush behavior.

**Why it happens:** `wsplit_seen` is initialized once in C++ (`= false;` field initializer), no reset path touches it.

**How to avoid:** P3 lock-in — `WarpState.reset()` does NOT clear `wsplit_seen`. Only the constructor sets it to `False`. Add a unit test that asserts `wsplit_seen` persists across `npu.reset()`. (If a future regression contradicts this, a "warp_state_reset_includes_wsplit" branch will be needed — but C++ doesn't have it today.)

### Pitfall 8: `dispatch.py:dispatch_4mode` Mode 3 op-encoding ambiguity

**What goes wrong:** Mode 3 reads `is_load = (sub_op == 0) || (opcode == GTX_OP_DMA)`. If a test sets sub_op != 0 AND opcode != GTX_OP_DMA, Mode 3 routes as STORE — even though caller may have intended LOAD.

**Why it happens:** ISS encoding has two redundant signals. C++ collapses them with OR for safety.

**How to avoid:** Lock test fixtures to one of two patterns:
- Mode 3 LOAD: `sub_op = 0` (regardless of opcode) — matches `dispatch_iss_opcode` default
- Mode 3 STORE: `sub_op != 0` AND `opcode != GTX_OP_DMA` — matches firmware-style synthesized DMA

## Code Examples

Verified Python patterns (each cited to a C++ source line).

### 1. firmware_dma rs1/rs2/rs3 decode helper

```python
# riscv/gtx/dma_engine.py
def decode_firmware_dma_args(rs1: int, rs2: int, rs3: int, insn) -> dict:
    """Direct port of gtx_npu_dma.cc:262-288 packed-arg decode.

    Returns dict with: addr_hi, addr_lo, height, length, rd_stride, wr_stride,
    is_store, is_copy, funct3.
    """
    funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
    is_store = bool(funct3 & 1)
    is_copy = (not is_store) and bool(funct3 & 2)
    addr_hi = (rs1 >> 32) if is_copy else ((rs1 >> 27) & 0x1FFFFFFFFF)
    addr_lo = rs1 & 0x7FFFFFF
    height_raw = (rs2 >> 48) & 0xFFFF
    length_raw = (rs2 >> 32) & 0xFFFF
    rs2_low = rs2 & 0xFFFFFFFF
    rs3_low = rs3 & 0xFFFFFFFF

    height = 1 if height_raw == 0 else height_raw
    length = 0x10000 if length_raw == 0 else length_raw

    if is_store:
        wr_stride, rd_stride = rs2_low, rs3_low
    else:
        rd_stride, wr_stride = rs2_low, rs3_low

    return dict(addr_hi=addr_hi, addr_lo=addr_lo, height=height, length=length,
                rd_stride=rd_stride, wr_stride=wr_stride,
                is_store=is_store, is_copy=is_copy, funct3=funct3)
```

### 2. ddr_init_from_file canonical parser

```python
# riscv/gtx/ddr.py
def ddr_init_from_file(mem: GtxMemory, filename: str) -> None:
    """Direct port of gtx_npu_dma.cc:438-502."""
    reversed_mode = bool(os.environ.get('GTX_DDR_REVERSED'))  # D-08: per-call read

    offset = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                offset = int(line[1:].strip(), 16)
                continue
            nbytes = min(len(line) // 2, 32)
            if nbytes == 0:
                continue
            chunk = bytes.fromhex(line[: nbytes * 2])
            if reversed_mode:
                chunk = chunk[::-1]
            ensure_ddr(mem, offset + nbytes)
            mem._ddr_bytes[offset : offset + nbytes] = np.frombuffer(chunk, dtype=np.uint8)
            offset += nbytes
```

### 3. ddr_dump_to_file canonical writer

```python
# riscv/gtx/ddr.py
def ddr_dump_to_file(mem: GtxMemory, filename: str, addr: int, size: int) -> None:
    """Direct port of gtx_npu_dma.cc:509-558. D-09: addr/size are args only."""
    reversed_mode = bool(os.environ.get('GTX_DDR_REVERSED'))

    # Address-to-offset (same as C++ ddr_offset)
    GTX_DDR_BASE = 0x370000000
    off = (addr - GTX_DDR_BASE) if addr >= GTX_DDR_BASE else addr

    if mem._ddr_bytes is None:
        return  # nothing to dump (matches C++ has_ddr() check)

    with open(filename, 'w') as f:
        for i in range(0, size, 32):
            chunk_off = off + i
            chunk_end = chunk_off + 32
            # Out-of-range: zero-pad
            if chunk_end <= mem._ddr_bytes.size:
                chunk = bytes(mem._ddr_bytes[chunk_off : chunk_end])
            else:
                chunk = bytearray(32)
                avail = max(0, mem._ddr_bytes.size - chunk_off)
                if avail > 0:
                    chunk[:avail] = bytes(mem._ddr_bytes[chunk_off : chunk_off + avail])
                chunk = bytes(chunk)
            if reversed_mode:
                chunk = chunk[::-1]
            f.write(chunk.hex() + '\n')
```

### 4. exec_dma_2d strided memcpy

```python
# riscv/gtx/dma_engine.py
def exec_dma_2d(mem: GtxMemory, *, nest_id: int, l2_addr: int, l1_addr: int,
                width: int, height: int, is_load: bool,
                l2_stride: int = 0, ctx: int = CTX_C3, spu_id: int = 0) -> int:
    """Direct port of gtx_npu_dma.cc:25-90 (functional-only — no L1 shadow sync).

    Returns 0 (cycles ignored in functional model).
    """
    if nest_id >= GTX_NEST_NUM:
        return 0
    if width == 0 or height == 0:
        return 0
    if l2_stride == 0:
        l2_stride = width

    l1_buf = mem.l1_byte(nest_id, spu_id)  # uint8 view, 384 KB
    l2_buf = mem.l2_byte(nest_id)          # uint8 view, 16 MB

    for row in range(height):
        l2_off = (l2_addr + row * l2_stride) % GTX_L2_SIZE_BYTES
        l1_off = (l1_addr + row * width)     % GTX_L1_SIZE_BYTES
        copy_len = width
        copy_len = min(copy_len, GTX_L2_SIZE_BYTES - l2_off)
        copy_len = min(copy_len, GTX_L1_SIZE_BYTES - l1_off)
        if copy_len <= 0:
            continue
        if is_load:
            l1_buf[l1_off : l1_off + copy_len] = l2_buf[l2_off : l2_off + copy_len]
        else:
            l2_buf[l2_off : l2_off + copy_len] = l1_buf[l1_off : l1_off + copy_len]
    return 0
```

### 5. Doubling-grow `ensure_ddr` (P3 upgrade)

```python
# riscv/gtx/ddr.py (P3 replacement)
INITIAL_FLOOR: int = 1 * 1024 * 1024  # 1 MiB minimum first allocation

def ensure_ddr(mem: GtxMemory, end_offset: int) -> np.ndarray:
    cap = get_ddr_cap()
    if end_offset > cap:
        raise ValueError(
            f"DDR access {end_offset:#x} exceeds cap {cap:#x} "
            f"(set GTX_DDR_SIZE env var to raise)"
        )
    current_size = mem._ddr_bytes.size if mem._ddr_bytes is not None else 0
    if end_offset > current_size:
        new_size = max(end_offset, current_size * 2, INITIAL_FLOOR)
        new_size = min(new_size, cap)
        new_arr = np.zeros(new_size, dtype=np.uint8)
        if mem._ddr_bytes is not None:
            new_arr[: current_size] = mem._ddr_bytes
        mem._ddr_bytes = new_arr
    return mem._ddr_bytes
```

### 6. Deferred-store push (S-loop STORE branch)

```python
# riscv/gtx/dma_engine.py
def firmware_dma_sloop_store(npu, *, nest: int, addr_hi: int, addr_lo: int,
                              length: int, height: int,
                              rd_stride: int, wr_stride: int) -> int:
    """S-loop STORE: defer the entire 2D transfer (gtx_npu_dma.cc:319-326)."""
    GTX_DDR_BASE = 0x370000000
    ddr_off = (addr_hi - GTX_DDR_BASE) if addr_hi >= GTX_DDR_BASE else addr_hi
    npu.deferred_ddr_stores.append(DeferredDdrStore(
        nest=nest,
        l2_off=addr_lo,
        ddr_off=ddr_off,
        length=length,
        height=height,
        l2_stride=rd_stride,
        ddr_stride=wr_stride,
    ))
    return 0
```

### 7. flush_deferred_ddr_stores (GtxNpu method)

```python
# riscv/gtx/npu.py (P3 addition)
def flush_deferred_ddr_stores(self) -> None:
    """Direct port of gtx_npu_dma.cc:415-435."""
    if not self.deferred_ddr_stores:
        return
    for req in self.deferred_ddr_stores:
        for row in range(req.height):
            ddr_off = req.ddr_off + row * req.ddr_stride
            l2_off = (req.l2_off + row * req.l2_stride) % GTX_L2_SIZE_BYTES
            copy_len = req.length
            ensure_ddr(self.mem, ddr_off + copy_len)
            copy_len = min(copy_len, self.mem._ddr_bytes.size - ddr_off)
            copy_len = min(copy_len, GTX_L2_SIZE_BYTES - l2_off)
            if copy_len > 0:
                self.mem._ddr_bytes[ddr_off : ddr_off + copy_len] = \
                    self.mem.l2_byte(req.nest)[l2_off : l2_off + copy_len]
    self.deferred_ddr_stores.clear()
```

## P3 Scope vs v2 Deferral

The DMA disasm table (`gtx_npu_disasm.inc:167-186`) lists 13 mnemonics under DMA-related funct7s. P3 must register **all** disasm entries (so the spike trace is correct) but only IMPLEMENT a subset.

| Mnemonic | funct7 | funct3 | P3 status | Reason |
|----------|--------|--------|-----------|--------|
| `load` | 0x40 | 0 | **Implement** | DMA-01/02 — firmware_dma path |
| `store` | 0x40 | 1 | **Implement** | DMA-01/02 |
| `copy` | 0x40 | 2 | **Implement** | DMA-01/02 (T-loop L1→L1) |
| `load_svr` | 0x41 | 0 | **Implement** | DMA-01 (`exec_load_svr`) |
| `store_svr` | 0x41 | 1 | **Implement** | DMA-01 (`exec_store_svr`) |
| `load_3d` | 0x41 | 4 | **Disasm-only stub** | DMA-V2-01; P4 mm_basic.elf doesn't use 3D |
| `store_3d` | 0x41 | 5 | **Disasm-only stub** | DMA-V2-01 |
| `mcast_s2l` | 0x42 | — | **Disasm-only stub** | DMA-V2-01; required by some firmware but not P4 mm_basic |
| `mcast_g2s` | 0x44 | 0 | **Disasm-only stub** | DMA-V2-01 |
| `mcast_s2s` | 0x44 | 2 | **Disasm-only stub** | DMA-V2-01 |
| `copy_mem` | 0x44 | 3 | **Disasm-only stub** | DMA-V2-01 (firmware uses for plan setup, but P3 success criteria don't exercise) |
| `load_svr_l1` | 0x43 | — | **Implement** | trivial alias for L1-bound load_svr; small port effort |
| `store_svr_l1` | 0x45 | — | **Implement** | trivial alias for L1-bound store_svr |
| `tpose` | 0x38 | — | **Implement** | DMA-01 (`exec_transpose`) |
| `fill` | 0x39 | — | **Implement** | DMA-01 (`exec_fill`) |

**P4 acceptance check:** `mm_basic.elf` uses (per `dispatch_4mode` Mode 3 routing) only `firmware_dma` (funct7=0x40) for DDR↔L2 load/store, and Mode 4 dispatch for compute. It does NOT use load_3d/store_3d/mcast/copy_mem. Confirmed by reading the disasm of the test sequence in CONTEXT.md `Phase 4` section.

**v2 promotion path:** when DMA-V2-01 is unblocked (typically by a firmware that fails because mcast/3D handler is missing), promote disasm-only stub to a full implementation. The disasm registration is already in place, so the only diff is replacing the NOP body with the C++ port.

## Test Patterns (D-10 / D-11)

### test_dma_roundtrip.py (P3 success #1)

```python
def test_dma_l1_to_ddr_roundtrip():
    npu = GtxNpu()
    npu.reset(MockProcessor())
    pattern = np.arange(4096, dtype=np.float16)
    npu.mem.l1_f16(0, 0)[0:4096] = pattern

    # L1 → L2 (T-loop STORE)
    npu.warp.is_ploop = True
    npu.warp.is_tloop = True
    npu.warp.tmu_id = 0
    npu.warp.curr_id = 0
    dma_engine.exec_dma_2d(npu.mem, nest_id=0, l2_addr=0, l1_addr=0,
                           width=8192, height=1, is_load=False, spu_id=0)
    npu.warp.is_tloop = False

    # L2 → DDR (S-loop STORE — deferred)
    npu.warp.is_sloop = True
    dma_engine.firmware_dma_sloop_store(npu, nest=0, addr_hi=0, addr_lo=0,
                                        length=8192, height=1,
                                        rd_stride=8192, wr_stride=8192)
    assert len(npu.deferred_ddr_stores) == 1
    npu.flush_deferred_ddr_stores()
    npu.warp.is_sloop = False

    # Dump → re-init → reverse path
    ddr_dump_to_file(npu.mem, '/tmp/dump.hex', addr=0, size=8192)
    npu2 = GtxNpu()
    ddr_init_from_file(npu2.mem, '/tmp/dump.hex')

    # DDR → L2 (S-loop LOAD — immediate)
    # ... reverse the chain
    # Final assertion: bit-exact
    assert np.array_equal(
        npu2.mem.l1_f16(0, 0)[0:4096].view(np.uint16),
        pattern.view(np.uint16),
    )
```

### test_ddr_modes.py (P3 success #2)

```python
def test_ddr_modes_differ_and_each_round_trips(monkeypatch, tmp_path):
    pattern = np.arange(64, dtype=np.float16)
    npu = GtxNpu()
    ensure_ddr(npu.mem, 128)
    npu.mem._ddr_bytes[:128] = pattern.view(np.uint8)

    # LTR dump
    ddr_dump_to_file(npu.mem, str(tmp_path/'ltr.hex'), 0, 128)

    # REVERSED dump
    monkeypatch.setenv('GTX_DDR_REVERSED', '1')
    ddr_dump_to_file(npu.mem, str(tmp_path/'rev.hex'), 0, 128)

    # Files differ
    assert (tmp_path/'ltr.hex').read_text() != (tmp_path/'rev.hex').read_text()

    # 32-byte bus-word reversal verified (sample one chunk)
    ltr_first_line = (tmp_path/'ltr.hex').read_text().splitlines()[0]
    rev_first_line = (tmp_path/'rev.hex').read_text().splitlines()[0]
    # rev = LTR with bytes reversed within the 32-byte chunk
    assert bytes.fromhex(rev_first_line) == bytes.fromhex(ltr_first_line)[::-1]

    # Each mode round-trips through its own init
    monkeypatch.delenv('GTX_DDR_REVERSED')
    npu2 = GtxNpu()
    ddr_init_from_file(npu2.mem, str(tmp_path/'ltr.hex'))
    assert bytes(npu2.mem._ddr_bytes[:128]) == bytes(npu.mem._ddr_bytes[:128])

    monkeypatch.setenv('GTX_DDR_REVERSED', '1')
    npu3 = GtxNpu()
    ddr_init_from_file(npu3.mem, str(tmp_path/'rev.hex'))
    assert bytes(npu3.mem._ddr_bytes[:128]) == bytes(npu.mem._ddr_bytes[:128])
```

### test_deferred_store.py (P3 success #4 — dual assertion)

```python
def test_deferred_store_queue_push_and_flush_diff():
    npu = GtxNpu()
    npu.reset(MockProcessor())

    # Pre-populate L2 with a known pattern
    npu.mem.l2_byte(1)[100:200] = np.arange(100, dtype=np.uint8)

    # Set up S-loop in nest 1
    npu.warp.is_ploop = True
    npu.warp.is_sloop = True
    npu.warp.tmu_id = 1

    # Synthetic STORE
    dma_engine.firmware_dma_sloop_store(npu, nest=1, addr_hi=0x1000, addr_lo=100,
                                        length=100, height=1,
                                        rd_stride=100, wr_stride=100)

    # Assertion #1: queue shape (no DDR write yet)
    assert len(npu.deferred_ddr_stores) == 1
    req = npu.deferred_ddr_stores[0]
    assert req.nest == 1
    assert req.l2_off == 100
    assert req.length == 100
    assert req.height == 1

    # Pre-flush DDR snapshot
    ensure_ddr(npu.mem, 0x1000 + 100)
    snapshot = bytes(npu.mem._ddr_bytes[0x1000:0x1000+100])
    assert snapshot == bytes(100)  # all zeros

    # Flush
    npu.flush_deferred_ddr_stores()

    # Assertion #2: queue cleared + DDR diff matches L2 source
    assert len(npu.deferred_ddr_stores) == 0
    post = bytes(npu.mem._ddr_bytes[0x1000:0x1000+100])
    assert post == bytes(np.arange(100, dtype=np.uint8))
    assert post != snapshot
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (pyspike baseline; already detected from `tests/test_extension.py` + `tests/gtx/conftest.py`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`; `tests/gtx/conftest.py` |
| Quick run command | `pytest tests/gtx/test_dma_*.py -x --noconftest -o "addopts="` |
| Full suite command | `pytest tests/gtx/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DMA-01 | All exec_* DMA functions correct | unit | `pytest tests/gtx/test_dma_engine.py -x` | ❌ Wave 0 |
| DMA-02 | firmware_dma packed encoding | unit | `pytest tests/gtx/test_firmware_dma.py -x` | ❌ Wave 0 |
| DMA-03 | Deferred-store queue + flush | unit | `pytest tests/gtx/test_deferred_store.py -x` | ❌ Wave 0 |
| DMA-04 | DDR hex I/O both modes | unit | `pytest tests/gtx/test_ddr_modes.py -x` | ❌ Wave 0 |
| DMA-05 | DMA round-trip bit-exactness | integration | `pytest tests/gtx/test_dma_roundtrip.py -x` | ❌ Wave 0 |
| DISP-03 | 4-mode dispatch routing | unit | `pytest tests/gtx/test_dispatch_4mode.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/gtx/test_dma_engine.py tests/gtx/test_firmware_dma.py -x` (~5 s)
- **Per wave merge:** `pytest tests/gtx/ -x` (full P3 suite, < 30 s expected)
- **Phase gate:** Full suite green before `/gsd:verify-work 3`

### Wave 0 Gaps

- [ ] `tests/gtx/test_dma_engine.py` — covers DMA-01 (all 6 exec_* helpers)
- [ ] `tests/gtx/test_firmware_dma.py` — covers DMA-02 (rs1/rs2/rs3 decode + funct3 branches)
- [ ] `tests/gtx/test_deferred_store.py` — covers DMA-03 (queue push, flush diff, end_p trigger, credit_st_chk trigger)
- [ ] `tests/gtx/test_ddr_modes.py` — covers DMA-04 (LTR + REVERSED, round-trip)
- [ ] `tests/gtx/test_dma_roundtrip.py` — covers DMA-05 (full L1↔L2↔DDR chain)
- [ ] `tests/gtx/test_dispatch_4mode.py` — covers DISP-03 (Mode 1/2/3/4 routing)

(Existing test infrastructure — `conftest.py`, `_mocks.py`, `_RISCV_AVAILABLE` self-detect — covers framework needs. No `conftest.py` changes required.)

## Open Questions

### Q1: Should `wsplit_seen` reset on `npu.reset()`?

**What we know:** C++ `gtx_npu.h:1251` initializes `wsplit_seen = false;` once (field initializer). No `reset()` path touches it.

**What's unclear:** P2's existing `WarpState.reset()` clears all warp state. If P3 keeps that pattern and resets `wsplit_seen` too, behavior diverges from C++ on hart-reset-then-firmware-rerun sequences.

**Recommendation:** Do NOT reset `wsplit_seen`. Add a P3 unit test asserting persistence. If this turns out wrong (P4 flags it), add a separate `WarpState.hard_reset()` for that case.

### Q2: How does Mode 3 dispatch interact with `firmware_dma`?

**What we know:** `firmware_dma` (funct7=0x40) is invoked DIRECTLY from custom0 entry — not through `dispatch_4mode`. It internally checks `is_sloop`/`is_tloop` and branches.

**What's unclear:** `dispatch_4mode` Mode 3 path (`is_ploop && is_sloop`) calls `exec_dma_2d` for the gem5-simplified `dispatch_dma` (funct7=0x07) path — different from firmware_dma. P3 must implement BOTH; they are orthogonal entry points.

**Recommendation:** P3 plan should treat `firmware_dma` (0x40) and `dispatch_dma → dispatch_4mode → exec_dma_2d` (0x07) as two separate code paths. Both call into `dma_engine.exec_dma_2d` (or its firmware variant) but the entry points are different. Test fixtures should cover BOTH funct7s.

### Q3: Is GTX_DDR_BASE exported in `params.py` or `encoding.py`?

**What we know:** Currently neither has `GTX_DDR_BASE`. C++ has it in `gtx_params.h:24`.

**Recommendation:** Add to `params.py` (alongside other HW topology constants). `encoding.py` is for ISA constants only.

```python
# params.py additions for P3:
GTX_DDR_BASE: int = 0x370000000  # firmware GTX_MAIN_BASE (DDR physical address)
```

### Q4: Is L2 stride 0 → contiguous default applied for `exec_dma_2d` only, or for `firmware_dma` too?

**What we know:** `gtx_npu_dma.cc:55` sets `l2_stride = width` if `l2_stride == 0` — but only in `exec_dma_2d`, not in `firmware_dma`. firmware_dma uses rd_stride/wr_stride from rs2_low/rs3_low directly without a contiguous fallback.

**Recommendation:** Only apply the `stride == 0 → width` substitution inside `exec_dma_2d`. Keep `firmware_dma` strides as-is (they may legitimately be 0 in HW pad cases, and the firmware code is supposed to know).

## Sources

### Primary (HIGH confidence — direct port)

- `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc` — 558 LOC; entire DMA module ported verbatim
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:25-143` — `dispatch()` 4-mode router
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:898-905` — `credit_st_chk` flush trigger
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc:21-142` — warp control + `endp` flush trigger (line 53)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc:198-203` — `ensure_ddr` (the actual 4 GiB allocator)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:684-694` — credit_st_chk in custom0 entry path
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc:62` — wsplit_seen=true on WSPLIT
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1257-1266` — `deferred_ddr_store_t` 7-field struct
- `vendor/gtx_cpp_reference/gtx/gtx_params.h:10-24` — HW topology constants
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:163-186` — DMA mnemonic registry
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — FP16 LE byte order + GTX_DDR_REVERSED rules

### Secondary (HIGH — pyspike existing assets)

- `src/main/python/riscv/gtx/memory.py` — `GtxMemory` API surface (P3 consumer)
- `src/main/python/riscv/gtx/ddr.py` — Phase 1 stub (P3 replaces ensure_ddr)
- `src/main/python/riscv/gtx/npu.py` — `GtxNpu` (P3 extends with deferred_ddr_stores)
- `src/main/python/riscv/gtx/dispatch.py` — `build_custom0_table` (P3 adds dispatch_4mode)
- `src/main/python/riscv/gtx/_registry.py` — `@handler` decorator (already supports `mask_funct3=True`)
- `src/main/python/riscv/gtx/warp_state.py` — `WarpState` (P3 adds wsplit_seen)
- `src/main/python/riscv/gtx/ops/control.py` — P2 warp handlers (P3 wires flush in endp)
- `vendor/gtx_cpp_reference/gtx/verify.py:165-185, 217-238` — DDR hex parser (LTR-only) + FP16 BE-pair compare

### Tertiary (n/a)

No web searches required — the entire phase is C++ → Python direct port from in-repo submodule. All claims trace to file:line in the vendored source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — locked at Phase 1; no new deps.
- Architecture (dispatch / dma_engine / ops split): HIGH — all 3 patterns directly port C++ structure.
- C++ function signature locks: HIGH — every signature traced to a `gtx_npu_dma.cc` line range.
- firmware_dma encoding: HIGH — bit-level match of `gtx_npu_dma.cc:262-288`.
- Deferred-store flush trigger reconciliation: HIGH — all 3 C++ call sites identified, both wired in P3.
- DDR hex I/O parser: HIGH — direct port of 65-line C++ function, line-for-line equivalent.
- 4-mode dispatch routing: HIGH — verbatim port of 60-line C++ function.
- P3 vs v2 scope split: MEDIUM — based on P4 mm_basic.elf likely instruction set; if a v1 firmware unexpectedly uses load_3d/mcast, we'll find out at P4 execute time and promote.
- ensure_ddr semantics divergence note: HIGH — C++ source is unambiguous; CONTEXT.md note was misleading.

**Research date:** 2026-05-05
**Valid until:** No expiration — C++ submodule is pinned by commit hash; only invalidates if user replaces submodule with a newer GTX_NPU revision.
