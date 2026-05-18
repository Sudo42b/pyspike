# Quick Task: 12 TODO part2 — 4 mcast/copy.mem Stubs Vendor Port — Research

**Researched:** 2026-05-18
**Domain:** GTX NPU RoCC DMA broadcast/copy ops (firmware-emitted multicast)
**Confidence:** HIGH (vendor C++ direct quote available for every byte of the port)

## Summary

The four `#!TODO` stubs in `src/main/python/riscv/gtx/unit/context/dma.py:223-272`
(`mcast.s2l`, `mcast.g2s`, `mcast.s2s`, `copy.mem`) currently `return 0` for
every firmware-emitted broadcast/copy instruction. Vendor C++ canonical
implementations exist verbatim in
`vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273` (s2l),
`:503-585` (g2s + s2s + copy_mem firmware decode), and
`vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:728-856` (s2s + copy_mem
execution body). Production reference at
`/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:231-748` (line-shifted
but byte-identical body — both files were spotchecked).

**Three of the four stubs are critical for GEMM workloads** (`MUL_MAT`,
`MUL_MAT_ID`, `SET_ROWS`, `WIN_UNPART` firmware emits `__mcast_g2s` +
`__mcast_s2l` + `__copy_mem`). The 10 P9 deferred ops in
`p9-vendor-sweep-non-multi-tile-bugs.md` (RELU/SIGMOID/ADD_VV/MUL_VV/NEG/
DIV/EXP/LEAKY_RELU) do **NOT** emit any of these four — these stubs unblock a
**different** test class (matrix multiplication + format conversion). The
`mcast.s2s` registration at `funct7=0x44, funct3=2` may be unreachable from
real firmware — see Pitfall 5.

**Primary recommendation:** Implement all four as direct ports of the vendor
firmware-path C++ bodies into `dma_engine.py` (engine functions, no
proc/insn deps) + thin shim handlers in `dma.py` that mirror the existing
`_firmware_dma_load/_store/_copy` shape. Reuse `mem.l2_byte()`,
`mem.l1_byte()`, `mem.ddr.read/write` for memory access; reuse
`npu.flush_deferred_ddr_stores()` before `copy.mem` DDR-path. Three of four
docstrings are wrong vs vendor — the rewrites MUST use vendor C++ bitfield
layout, not the current docstring.

## User Constraints (from task focus)

### Locked Decisions
- Pure Python + torch (current backend) — NO C++ additions per CLAUDE.md.
- Vendor C++ (`vendor/gtx_cpp_reference/gtx/`) is the authoritative reference.
- ABS strict byte-exact baseline (94.82s) MUST remain PASS. These ops are
  not on the ABS path (`__abs` uses VEC engine, not mcast), so by
  construction non-regression — but still must be verified.
- GELU strict PASS baseline preserved.
- Implement (not delete) the stubs — "기능구현 안한것" instruction.

### Claude's Discretion
- Whether `mcast.s2s` is implemented as a no-op-with-assert vs full port
  (decision: full port, with discovery-test to verify firmware reach).
- Whether to extend `decode_firmware_dma_args` or write fresh decoders
  (recommendation: separate decoders — bitfields differ enough that sharing
  is misleading).

### Deferred Ideas (OUT OF SCOPE)
- ISS-encoding path through `dispatch_iss_opcode` (the `rs1 == 0` branch
  in vendor `custom0.cc:273`). Current stubs only fire on the firmware
  `rs1 != 0` branch. Wiring the ISS path requires an OPSET routing chain
  that is a separate plan.
- `gtx_cycles::dmac()` cycle accounting — current Python implementation
  does not track cycles for any DMA op. Match existing behaviour.

## Standard Stack

Reuse-only — no new dependencies.

| Component | Location | Why Standard |
|-----------|----------|--------------|
| `torch.uint8` row-strided 2D views | `dma_engine.exec_dma_2d` pattern | Single CUDA launch; vendor row-loop becomes one `copy_()` |
| `mem.l2_byte(nest)` / `mem.l1_byte(nest, spu)` | `memory.GtxMemory` | Existing per-(NEST,SPU) views, zero-copy |
| `mem.ddr.read(off, n)` / `mem.ddr.write(off, t)` | `DDR_MEMORY` | CPU-resident grow-on-demand DDR |
| `npu.flush_deferred_ddr_stores()` | `control.py:75,228` | Already-wired flush API — `copy.mem` DDR-path MUST call this first per vendor |
| `_select_nest(npu)` / `_select_spu(npu)` helpers | `dma.py:33-49` | NEST/SPU selection mirrors vendor `is_ploop ? tmu_id : 0` |

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Operand decode | A 4th universal `decode_*_args` | One small decoder per op | Bitfields are op-specific; sharing obscures the cite line |
| DDR offset conversion | New helper | `mem.ddr.write/read` accept raw offsets; use `(addr - GTX_DDR_BASE)` inline | Matches existing `firmware_dma_sloop_load:339` pattern |
| Row loop | Python `for row in range(height)` | 2D `.view(height, length)` + single `copy_()` | Existing `exec_dma_2d`-class pattern; CUDA launches matter |
| Cross-device DDR↔scratchpad transfers | `.cpu()` per row | `mem.ddr.read(...).to(l2.device)` once | DMA-boundary device crossing per `DDR_MEMORY` contract |

## Vendor C++ Reference Map (Authoritative)

### 1. `mcast.s2l` — L2 → L1 multicast (funct7=0x42)

**File:** `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273`
**Production mirror:** `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:231`

```cpp
// rs1 = (L2_src << 32) | L1_dst      ← NOT what current docstring says
// rs2 = (height << 48) | (length << 32) | read_stride
// rs3 = target_spu_bitmask (from opset → gspr[GSPR_GTX_OPERAND3])
uint32_t l2_addr  = static_cast<uint32_t>(rs1 >> 32);
uint32_t l1_addr  = static_cast<uint32_t>(rs1 & 0xFFFFFFFF);
uint16_t height   = (rs2 >> 48) & 0xFFFF;
uint32_t length   = (rs2 >> 32) & 0xFFFF;
uint32_t rd_stride = rs2 & 0xFFFFFFFF;
uint32_t tgt_mask = rs3 & 0xFFFF;
if (height == 0) height = 1;
if (length == 0) length = 0x10000;
int nest = is_ploop ? tmu_id : 0;
for (int s = 0; s < GTX_SPUS_PER_NEST; s++) {       // 16 SPUs
    if (!((tgt_mask >> s) & 1)) continue;
    for (uint16_t row = 0; row < height; row++) {
        uint32_t l2_off = (l2_addr + row * rd_stride) % GTX_L2_SIZE;
        uint32_t l1_off = (l1_addr + row * length)    % GTX_L1_SIZE;
        ... bounds-clip ...
        nests[nest].l2_read(l2_off, &spu.l1[l1_off], copy_len);
    }
}
```

**Pattern:** ONE source NEST L2 → up-to-16 selected SPU L1s; same source
bytes broadcast. Python: read L2 row span once, then `copy_()` into each
selected SPU L1.

### 2. `mcast.g2s` — DDR → L2 multicast (funct7=0x44, f3=0)

**File:** `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:545-583`

```cpp
// rs1 = (DDR_src_addr << 27) | L2_dst_addr   ← 27-bit L2 + 37-bit DDR
// rs2 = (height << 48) | (length << 32) | read_stride
// rs3 = target_nest_bitmask
uint64_t ddr_addr = (rs1 >> 27) & 0x1FFFFFFFFFULL;  // 37 bits
uint32_t l2_addr  = rs1 & 0x7FFFFFF;                // 27 bits
uint16_t height   = (rs2 >> 48) & 0xFFFF;
uint32_t length   = (rs2 >> 32) & 0xFFFF;
uint32_t rd_stride = rs2 & 0xFFFFFFFF;
uint32_t tgt_mask = rs3 & 0xFFFF;
if (height == 0) height = 1;
if (length == 0) length = 0x10000;
ensure_ddr();
for (uint16_t row = 0; row < height; row++) {
    uint64_t ddr_off = ddr_offset(ddr_addr + (uint64_t)row * rd_stride);
    uint32_t l2_off  = (l2_addr + row * length) % GTX_L2_SIZE;
    ... bounds-clip ...
    for (int k = 0; k < GTX_NUM_NESTS; k++) {       // 4 NESTs
        if ((tgt_mask >> k) & 1)
            nests[k].l2_write(l2_off, &ddr[ddr_off], copy_len);
    }
}
```

**Pattern:** ONE DDR source → up-to-4 selected NEST L2s. Python: snapshot
DDR row span once (CPU→GPU), then `copy_()` into each selected NEST L2.
**No zero-fill special case** despite the current docstring claim — see
Pitfall 1.

### 3. `mcast.s2s` — L2 → L2 multicast across NESTs (sub_op=0x22)

**File:** `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:732-762`
(reached only via OPSET → `dispatch_iss_opcode`; NOT in `custom0.cc`)

```cpp
// op1[26:0]=src_addr, op1[53:27]=dst_addr, op1[61:56]=src_tmu
// op2[47:32]=length, op2[63:48]=height, op2[31:0]=src_stride
// op3[31:0]=dst_stride, op3[63:32]=target_tmu_bitmask
uint32_t src_addr  = op1 & 0x7FFFFFF;
uint32_t dst_addr  = (op1 >> 27) & 0x7FFFFFF;
int      src_tmu   = (op1 >> 56) & 0x3F;
uint16_t length    = (op2 >> 32) & 0xFFFF;
uint16_t height    = (op2 >> 48) & 0xFFFF;
uint32_t src_stride = op2 & 0xFFFFFFFF;
uint32_t dst_stride = op3 & 0xFFFFFFFF;
uint32_t tgt_mask  = (op3 >> 32) & 0xFFFFFFFF;
if (src_tmu >= GTX_NUM_NESTS) src_tmu = 0;
for (uint16_t row = 0; row < height; row++) {
    uint32_t s_off = (src_addr + row * src_stride) % GTX_L2_SIZE;
    uint32_t d_off = (dst_addr + row * dst_stride) % GTX_L2_SIZE;
    std::vector<uint8_t> tmp(copy_len);
    nests[src_tmu].l2_read(s_off, tmp.data(), copy_len);
    for (int k = 0; k < GTX_NUM_NESTS; k++)
        if ((tgt_mask >> k) & 1)
            nests[k].l2_write(d_off, tmp.data(), copy_len);
}
```

**Pattern:** Source NEST L2 → up-to-4 target NEST L2s. No self-broadcast
guard in vendor — current docstring claim of "if target_nest_sel == 0,
target_nest[31:0] else target_nest[63:32]" does NOT match vendor (vendor
takes `tgt_mask = (op3 >> 32)`, no select-bit). See Pitfall 3.

### 4. `copy.mem` — DDR↔DDR (and L2↔L2, DDR↔L2 corner cases) (sub_op=0x23)

**Firmware decode:** `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:509-543`
re-stages operands into GSPR and calls `dispatch_iss_opcode(..., sub_op=0x23)`.

**Execution body:** `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:763-846`

```cpp
// op1[36:0] = src_addr (37-bit raw — addr >= GTX_L2_SIZE ⇒ DDR)
// op2[31:0] = src_stride (read_stride)
// op2[47:32] = length, op2[63:48] = height
// op3[36:0] = dst_addr (37-bit raw)
// dst_stride = (op1[63:48] low 16) | (op3[63:48] << 16)
uint64_t src_addr_raw = op1 & 0x1FFFFFFFFFULL;
uint64_t dst_addr_raw = op3 & 0x1FFFFFFFFFULL;
uint32_t src_stride   = op2 & 0xFFFFFFFF;
uint16_t length       = (op2 >> 32) & 0xFFFF;
uint16_t height       = (op2 >> 48) & 0xFFFF;
uint32_t dst_stride   = ((op1 >> 48) & 0xFFFF) | (((op3 >> 48) & 0xFFFF) << 16);
if (height == 0) height = 1;
// Note: vendor does NOT special-case length==0 here (unlike s2l/g2s/firmware_dma).

bool src_is_ddr = (src_addr_raw >= GTX_L2_SIZE);  // i.e. >= 16 MiB
bool dst_is_ddr = (dst_addr_raw >= GTX_L2_SIZE);

if (src_is_ddr || dst_is_ddr) {
    flush_deferred_ddr_stores();   // ★ MUST flush before copy_mem
    ensure_ddr();
    ... 4 cases: DDR↔DDR, DDR→L2, L2→DDR, L2↔L2 (last falls through to else)
} else {
    // L2-to-L2 same-NEST, temp buffer for overlap safety
}
```

**Pattern:** Decides DDR-vs-L2 from the raw address (≥ `GTX_L2_SIZE` = DDR).
ALL DDR-touching paths must call `flush_deferred_ddr_stores()` first.

## Architecture Patterns (Python Port)

### Recommended Layout

```
src/main/python/riscv/gtx/unit/context/
├── dma.py              # 4 new @handler shims (replace stubs)
├── dma_engine.py       # 4 new engine functions (no proc/insn deps)
```

### Pattern: `dma.py` shim (replace each stub)

```python
@handler(kind='custom0', funct7=GTX_ISS_F7_MCAST_S2L, funct3=0,
         mnemonic='mcast.s2l')
def _mcast_s2l(npu, proc, insn, xs1, xs2):
    """firmware mcast.s2l (funct7=0x42): L2 → L1 broadcast to selected SPUs.

    Vendor: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:230-273.
    rs1 = (L2_src << 32) | L1_dst, rs2 = (h<<48)|(len<<32)|rd_stride,
    rs3 = target_spu_bitmask.
    """
    state = proc.state
    rs1 = state.XPR[insn.rs1]
    rs2 = state.XPR[insn.rs2]
    rs3 = npu.gspr.get(GSPR['GSPR_GTX_OPERAND3'].address, 0)
    nest = _select_nest(npu)
    return dma_engine.firmware_mcast_s2l(
        npu.mem, nest=nest,
        l2_addr=(rs1 >> 32) & 0xFFFFFFFF,
        l1_addr=rs1 & 0xFFFFFFFF,
        height=(rs2 >> 48) & 0xFFFF,
        length=(rs2 >> 32) & 0xFFFF,
        rd_stride=rs2 & 0xFFFFFFFF,
        target_spu_mask=rs3 & 0xFFFF)
```

### Pattern: `dma_engine.py` engine fn (direct vendor port)

```python
def firmware_mcast_s2l(mem, *, nest, l2_addr, l1_addr,
                        height, length, rd_stride, target_spu_mask):
    """Direct port of gtx_npu_custom0.cc:230-273.

    HW conventions: height==0 → 1, length==0 → 0x10000 (vendor :248-249).
    """
    if height == 0: height = 1
    if length == 0: length = 0x10000
    if rd_stride == 0: rd_stride = length  # NB: vendor does NOT do this for s2l
    # — but for the strided=length common case, both behave identically.
    # CHECK: vendor `(l2_addr + row * rd_stride)` with rd_stride=0 collapses
    # to the same row every iteration. Match vendor literally — do NOT
    # normalise rd_stride==0 to length.
    l2 = mem.l2_byte(nest)
    # Single contiguous source span snapshot (one tensor view)
    src_2d = l2[l2_addr : l2_addr + (height - 1) * rd_stride + length] \
                .view(...)[:, :length]  # match firmware_dma_sloop_load pattern
    for s in range(GTX_SPU_NUM):
        if not ((target_spu_mask >> s) & 1):
            continue
        l1 = mem.l1_byte(nest, s)
        dst_2d = l1[l1_addr : l1_addr + height * length].view(height, length)
        dst_2d.copy_(src_2d)
    return 0
```

(Use the same invariant-assert pattern as `dma_engine.firmware_dma_sloop_load`
— `rd_stride >= length`, `l2_end <= GTX_L2_SIZE_BYTES`, etc.)

### Anti-Patterns to Avoid

- **Calling `flush_deferred_ddr_stores` from `mcast.g2s`** — vendor does NOT
  flush before `mcast.g2s` (only `copy.mem` does).
- **Reusing `decode_firmware_dma_args`** — bitfield layouts differ
  (`mcast.s2l` uses `rs1>>32` for the 32-bit L2 source; `mcast.g2s` uses
  `rs1>>27` for the 37-bit DDR source; `copy.mem` uses `rs1 & 0x1FFFFFFFFF`
  for the 37-bit DDR source, and a split-stride layout). Each needs its own
  decoder.
- **Treating `mcast.s2s` arg layout per current docstring** — the
  "src_nest_id[61:56], target_nest_sel[63]" decoder in the current stub is
  wrong vs vendor `dispatch.cc:732-745`. Vendor uses
  `src_tmu = (op1 >> 56) & 0x3F` and `tgt_mask = (op3 >> 32) & 0xFFFFFFFF`,
  no select bit.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — these are pure ops, no persistent state | None |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | None | None |
| Build artifacts | None — Python source only | None |

These are stateless function-level ops; no cleanup or migration required.

## Common Pitfalls

### Pitfall 1: `mcast.g2s` "zero-fill if src all 1s" is FICTION

**What goes wrong:** Current stub docstring says
"broadcast DDR data to selected L2SPM (zero fill if src address is all 1)".
The vendor C++ has **NO such special case**. Adding zero-fill logic
silently breaks any firmware that legitimately uses
`ddr_addr == 0x1FFFFFFFFF` as a real address (unlikely but legal).
**Why it happens:** Docstring drift from an older or hypothetical HW spec.
**How to avoid:** Implement per vendor `custom0.cc:545-583` only. Update
the docstring to match vendor.
**Warning signs:** Any conditional branch in the port that checks
`src_addr == 0xFFFFFFFF...` should not exist.

### Pitfall 2: `copy.mem` requires `flush_deferred_ddr_stores()` first

**What goes wrong:** If `copy.mem` reads from DDR (the common case — it
exists *because* it bridges DDR-to-DDR), the deferred-store queue holds
unflushed L2→DDR writes from a previous S-loop. Without the flush,
`copy.mem` reads stale DDR.
**Why it happens:** Vendor `dispatch.cc:784` explicitly calls
`flush_deferred_ddr_stores()` at the top of the DDR-path branch — this is
NOT optional.
**How to avoid:** First line of the DDR-touching branch must be
`npu.flush_deferred_ddr_stores()`. The L2↔L2 same-NEST else-branch
(`dispatch.cc:832-844`) does NOT need the flush — preserve that asymmetry.
**Warning signs:** Pre-existing tests pass but `MUL_MAT_ID`, `SET_ROWS`,
or `WIN_UNPART` show "line 0 byte-exact wrong, partial-match later" — that
is the classic stale-DDR signature.

### Pitfall 3: `mcast.s2s` self-broadcast guard is FICTION

**What goes wrong:** Current stub docstring says "self broadcast is not
supported (use copy instead), if target_nest_sel == 0, target_nest[31:0]
else target_nest[63:32]". Vendor `dispatch.cc:732-762` has **NO select-bit
and NO self-broadcast guard** — it iterates all 4 NESTs and writes wherever
`tgt_mask` says, even if `src_tmu == k`.
**Why it happens:** Hypothetical-HW-spec docstring drift.
**How to avoid:** Port vendor literally — no guards. If `src_tmu` is in
`tgt_mask`, vendor reads-then-writes the same NEST L2 at distinct
(src_addr, dst_addr) offsets, which is a legitimate intra-NEST permute.
**Warning signs:** Any `if src_tmu == k: continue` in the port.

### Pitfall 4: `mcast.s2s` funct3=2 dispatch may be unreachable

**What goes wrong:** The current Python registration at
`funct7=0x44, funct3=2, mnemonic='mcast.s2s'` may never fire. Vendor
`custom0.cc:507` only branches `f3 == 3 → copy_mem` and `f3 == 0 →
mcast_g2s`; **there is no `f3 == 2` firmware branch**. `mcast.s2s` is
reached ONLY via the ISS path (`dispatch_iss_opcode` with `sub_op == 0x22`),
which requires OPSET to set `GSPR_GTX_OPCODE = 0x22` and then a normal
`funct7=0x44 rs1==0` instruction to call `dispatch()`.
**Why it happens:** RoCC `funct3 = (xd<<2)|(xs1<<1)|xs2` encodes register
plumbing flags, not a sub-opcode. The Python `@handler` `funct3=2` filter
only matches when xs2=0, xs1=1, xd=0, which is a peculiar firmware-side
encoding that real `__mcast_s2s` calls may not produce.
**How to avoid:**
  (a) Keep the `funct3=2` registration but make the body call into the
      same engine fn the ISS path would use (`firmware_mcast_s2s`).
  (b) Add a discovery test that exercises a synthetic `funct3=2`
      `funct7=0x44` instruction and asserts the handler fires; if it
      doesn't, route via OPSET path instead. This is a follow-up — for the
      current quick task, implement the body and document the dispatch
      uncertainty in the docstring.
**Warning signs:** Test calls `mcast.s2s` directly via insn encoding and
the handler doesn't fire (returns 0 from default Spike path).

### Pitfall 5: `mcast.s2l` rs1 layout — docstring is wrong

**What goes wrong:** Current stub says "operand1: dst_addr[23:0],
src_addr[58:32]" — that is the **OPSET / ISS-encoding** layout. The
firmware-emitted layout (`custom0.cc:241-242`) is
`rs1 = (l2_addr << 32) | l1_addr`, i.e. dst (L1) in low 32, src (L2) in
high 32. They are different paths.
**How to avoid:** Use vendor `custom0.cc:241-242` decode in the
firmware-path handler. Update docstring.

### Pitfall 6: `length == 0` semantics differ across the 4 ops

**What goes wrong:** Vendor `custom0.cc:249, 561` set `length = 0x10000` if
`length_raw == 0` for `mcast.s2l` and `mcast.g2s`. Vendor
`dispatch.cc:777` does **NOT** apply this normalisation for `copy.mem`
(only `height == 0 → 1`). The `mcast.s2s` path in `dispatch.cc:741-742`
similarly does NOT special-case `length == 0`.
**How to avoid:** Match vendor per-op:
  - `mcast.s2l`: length 0 → 0x10000
  - `mcast.g2s`: length 0 → 0x10000
  - `mcast.s2s`: length 0 → use as-is (vendor literal)
  - `copy.mem`:  length 0 → use as-is (vendor literal)
**Warning signs:** A unified `decode_firmware_dma_args`-style helper would
hide this divergence — keep per-op decoders.

### Pitfall 7: GTX_L2_SIZE constant for DDR address discrimination

**Vendor `dispatch.cc:779-780`:** `bool src_is_ddr = (src_addr_raw >= GTX_L2_SIZE);`
where `GTX_L2_SIZE` is from `gtx_params.h` and matches Python
`GTX_L2_SIZE_BYTES` (16 MiB = 16 \* 1024 \* 1024 = 0x1000000).
**How to avoid:** Use `from ...config_params import GTX_L2_SIZE_BYTES` for
the DDR-discrimination check, not `GTX_DDR_BASE` (different constant).
The vendor `ddr_offset()` helper subtracts `GTX_DDR_BASE` only AFTER the
DDR-vs-L2 decision; Python `mem._ddr_offset()` does the same.

## Test Coverage Strategy

### Existing coverage

| Test file | Hits these 4 ops? | Notes |
|-----------|--------------------|-------|
| `tests/gtx/test_regression_fw_full_sweep.py` | Partial — parametrised over 84 ops, but `MUL_MAT*`/`SET_ROWS`/`WIN_UNPART` may be SKIPPED if vendor dir absent | Most likely currently SKIPPED |
| `tests/gtx/test_regression_elf_n1s16.py` | Same caveat | Same |
| `tests/gtx/test_custom_dispatch_chain.py` | No — tests handler registration only | Will need extension |

`grep -lE "mcast|copy_mem|copy\.mem" tests/gtx/*.py` ⇒ no matches: there
is **no existing unit test** for the four ops.

### Recommended new test (1 file, 4 small tests)

Add `tests/gtx/test_mcast_copy_mem.py` with one test per op using
synthetic insn encoding (mirror `test_custom_dispatch_chain.py` style):

1. `test_mcast_s2l_broadcast_to_2_spus`: pre-seed NEST 0 L2 with known
   pattern, build `rs1=(l2<<32)|l1`, `rs2=(1<<48)|(64<<32)|64`,
   `gspr[OPERAND3]=0b101` (SPUs 0 and 2). Assert L1[NEST 0][SPU 0] and
   L1[NEST 0][SPU 2] match L2 source; SPU 1 unchanged.
2. `test_mcast_g2s_broadcast_to_2_nests`: pre-seed DDR, target NESTs 0+2,
   assert L2[0] and L2[2] match, L2[1] unchanged.
3. `test_mcast_s2s_l2_to_l2`: NEST 0 → NEST 1+2+3. (Caveat: may need to
   exercise via OPSET path if Pitfall 4 holds.)
4. `test_copy_mem_ddr_to_ddr`: pre-seed DDR src, run copy.mem, assert dst
   bytes match. ALSO push a `DeferredDdrStore` first, assert the flush
   happened (verify `npu.deferred_ddr_stores == []` after).

For **regression validation**: after implementing, run
`pytest tests/gtx/test_regression_fw_full_sweep.py -k 'mul_mat or set_rows'`
— these were the SKIPped ops most likely to now PASS. Also run
`-k abs` and `-k gelu` to confirm no baseline regression.

## Code Examples (Verified Patterns)

### Pattern: Vendor row-loop → torch 2D view

Vendor `custom0.cc:257-266`:
```cpp
for (uint16_t row = 0; row < height; row++) {
    uint32_t l2_off = (l2_addr + row * rd_stride) % GTX_L2_SIZE;
    uint32_t l1_off = (l1_addr + row * length)    % GTX_L1_SIZE;
    nests[nest].l2_read(l2_off, &spu.l1[l1_off], copy_len);
}
```

Python (matches existing `firmware_dma_sloop_load:339-368`):
```python
l2 = mem.l2_byte(nest)
src_2d = l2[l2_addr : l2_addr + (height - 1) * rd_stride + length] \
            .view(height, rd_stride)[:, :length]
for s in target_spus:
    l1 = mem.l1_byte(nest, s)
    l1[l1_addr : l1_addr + height * length] \
        .view(height, length).copy_(src_2d)
```

### Pattern: `copy.mem` DDR-path with mandatory flush

```python
def firmware_copy_mem(npu, *, src_addr_raw, dst_addr_raw,
                       src_stride, dst_stride, length, height):
    """Direct port of dispatch.cc:763-846 (sub_op=0x23)."""
    if height == 0:
        height = 1
    src_is_ddr = src_addr_raw >= GTX_L2_SIZE_BYTES
    dst_is_ddr = dst_addr_raw >= GTX_L2_SIZE_BYTES
    if src_is_ddr or dst_is_ddr:
        npu.flush_deferred_ddr_stores()              # ★ mandatory
        # 4-case dispatch: DDR↔DDR / DDR→L2 / L2→DDR / L2↔L2
        ...
    else:
        # Same-NEST L2↔L2 — temp buffer for overlap
        ...
```

## State of the Art

No external libraries — pure vendor C++ port. The "current approach" is
the existing `dma_engine.py` torch-2D-view pattern (post Plan 04 / sloop
fix). All four ports must adopt that pattern.

## Open Questions

1. **Does any current pyspike-bundled firmware exercise `mcast.s2s`?**
   - What we know: Production C tests (`/mnt/e/14_NIGHTLY/gtx_spike/test/`)
     contain `__mcast_g2s`, `__mcast_s2l`, `__copy_mem` macros but NO
     `__mcast_s2s` macro grep hit.
   - What's unclear: The `funct3=2, funct7=0x44` encoding may only be
     reachable via OPSET (vendor `dispatch.cc:732`, sub_op=0x22), in which
     case the current `@handler(funct3=2)` registration is dead.
   - Recommendation: Implement the engine function regardless; add a
     pytest that asserts handler firing — if the test reveals dead
     registration, route through OPSET in a follow-up.

2. **Does `MUL_MAT`/`SET_ROWS` firmware exist in the pyspike test corpus?**
   - What we know: `/mnt/e/14_NIGHTLY/gtx_spike/test/MUL_MAT/n1s16/` is the
     production source.
   - What's unclear: Whether `test_regression_fw_full_sweep.py`
     parametrisation reaches them after `vendor/` import.
   - Recommendation: After implementation, run the sweep filtered on
     `mul_mat`/`set_rows`/`win_unpart` and report pass/fail. If they
     SKIP for missing `.elf`/`_ref.txt`, surface in task summary — no
     additional implementation work is blocked.

## Environment Availability

Skipped — no external dependencies (pure Python + existing torch).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (uv venv per memory `reference_test_runner`) |
| Config file | `tests/gtx/conftest.py` (exists) |
| Quick run command | `uv run pytest tests/gtx/test_mcast_copy_mem.py -xvs` |
| Full suite command | `uv run pytest tests/gtx/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TODO-A1 | mcast.s2l broadcasts L2→selected SPU L1s | unit | `uv run pytest tests/gtx/test_mcast_copy_mem.py::test_mcast_s2l -xvs` | ❌ Wave 0 |
| TODO-A2 | mcast.g2s broadcasts DDR→selected NEST L2s | unit | `uv run pytest tests/gtx/test_mcast_copy_mem.py::test_mcast_g2s -xvs` | ❌ Wave 0 |
| TODO-A3 | mcast.s2s broadcasts L2→selected NEST L2s | unit | `uv run pytest tests/gtx/test_mcast_copy_mem.py::test_mcast_s2s -xvs` | ❌ Wave 0 |
| TODO-A4 | copy.mem DDR→DDR (and L2↔DDR) with prior flush | unit | `uv run pytest tests/gtx/test_mcast_copy_mem.py::test_copy_mem -xvs` | ❌ Wave 0 |
| BASELINE | ABS strict byte-exact PASS preserved | regression | `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k abs -xvs` | ✅ exists |
| BASELINE | GELU strict PASS preserved | regression | `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k gelu -xvs` | ✅ exists |
| OPTIONAL | MUL_MAT / SET_ROWS now exercisable | regression | `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'mul_mat or set_rows or win_unpart' -xvs` | ✅ (may SKIP) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/gtx/test_mcast_copy_mem.py -xvs`
- **Per wave merge:** `uv run pytest tests/gtx/ -x` (full unit suite)
- **Phase gate:** Add MUL_MAT/SET_ROWS regression PASS before
  `/gsd:verify-work`; ABS+GELU baselines unchanged.

### Wave 0 Gaps
- [ ] `tests/gtx/test_mcast_copy_mem.py` — covers TODO-A1..A4 (new file)
- [ ] No new fixtures needed — reuse `tests/gtx/conftest.py`
- [ ] No framework install — pytest+uv already wired per
  `reference_test_runner` memory.

## Project Constraints (from CLAUDE.md)

- **C++ 추가 코드 금지** — pure Python + torch; no C++ extensions.
- **NumPy ≥1.20** target backend per CLAUDE.md, **but current code uses
  torch** (see `dma_engine.py:21` `import torch`). Implementation should
  follow current torch pattern (NumPy migration is later).
- **Bit-exact** ULP 1, atol 0.001 vendor parity required for ANY op that
  ends up in DDR.
- **No new runtime dependencies** beyond NumPy/torch — must reuse existing
  `GtxMemory` / `mem.ddr.read/write` / `mem.l2_byte/l1_byte` APIs.
- **NEST × SPU × L1 limits:** `GTX_NEST_NUM = 4`, `GTX_SPU_NUM = 16`,
  `GTX_L1_SIZE_BYTES = 384 KB`, `GTX_L2_SIZE_BYTES = 16 MB`.
- **No print debug auto-removal** per `feedback_debug_prints` memory.
- **Vendor C++ as authoritative reference** per `reference_vendor_cpp`
  memory — first stop is `vendor/gtx_cpp_reference/gtx/` (this research
  cited it 11+ times) then `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/` for
  production cross-check.

## Sources

### Primary (HIGH confidence — direct file reads)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:220-273` — mcast.s2l
  firmware decode + body (verbatim)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:503-585` — mcast.g2s +
  copy_mem firmware decode + body (verbatim)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc:728-856` — mcast.s2s +
  copy_mem execution body (verbatim, sub_op=0x22 and 0x23)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h:338-339,589-592,704,715` —
  funct7 constants (0x42, 0x44) and OPSET sub_op mappings
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx/src/gtx_npu_custom0.cc:231,708` —
  production reference (byte-identical body, line-shifted)
- `src/main/python/riscv/gtx/unit/context/dma.py:1-272` — current stubs
  and existing handler patterns
- `src/main/python/riscv/gtx/unit/context/dma_engine.py:1-440` — existing
  engine pattern (torch 2D views, invariant asserts)
- `src/main/python/riscv/gtx/unit/memory.py:183-340` — `GtxMemory` /
  `DDR_MEMORY` API surface

### Secondary (MEDIUM confidence — grep + cross-reference)
- `/mnt/e/14_NIGHTLY/gtx_spike/test/{MUL_MAT,MUL_MAT_ID,SET_ROWS,WIN_UNPART}/n1s16/*.c` —
  firmware grep confirms which ops emit these instructions
- `.planning/seeds/p9-vendor-sweep-non-multi-tile-bugs.md` — confirms
  these 4 stubs do NOT unblock P9 deferred ops (different bug class)
- `src/main/python/riscv/gtx/unit/ins/encoding.py:89-90` — funct7
  constants (`GTX_ISS_F7_MCAST_S2L = 0x42`, `GTX_ISS_F7_MCAST_G2S = 0x44`)

### Tertiary (LOW confidence — assumed, flag for validation)
- `mcast.s2s` Python `funct3=2` registration reachability (Pitfall 4) —
  requires a discovery test to validate.

## Metadata

**Confidence breakdown:**
- Vendor C++ port mapping: HIGH — every byte of the 4 ports has a direct
  cite line in `vendor/gtx_cpp_reference/`.
- Test coverage: HIGH — confirmed no existing tests for these 4 ops;
  pattern to follow is `test_deferred_store.py` shape.
- P9 unblock claim: HIGH — grep confirms 10 P9 ops do NOT emit mcast/copy.
- `mcast.s2s` dispatch reachability: LOW — needs runtime confirmation.

**Research date:** 2026-05-18
**Valid until:** 2026-06-17 (30 days — vendor C++ is stable)
