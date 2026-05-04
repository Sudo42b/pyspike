# Feature Research — Python GTX NPU on pyspike

**Domain:** Python-bindable RISC-V RoCC NPU functional model (port of `gtx_spike/gtx/`'s C++ `gtx_npu_t` to a `riscv.isa.ROCC` subclass)
**Researched:** 2026-05-04
**Confidence:** HIGH (ground truth = existing C++ reference at `~/NIGHTLY/gtx_spike/gtx/`; pyspike trampoline surface already verified in `.planning/codebase/ARCHITECTURE.md`)
**Mode:** Brownfield port — features are derived from the existing C++ implementation, NOT from generic NPU literature

## Scope Note

Features pyspike already provides (`riscv.isa.ROCC`, `custom0..3` trampolines, `@register` decorator, `riscv.dev.MMIO` base, `PYSPIKE_LIBS` bootstrap, manylinux wheel pipeline, `get_instructions/get_disasms/get_csrs`) are NOT enumerated here — see `.planning/codebase/ARCHITECTURE.md`. This file lists features the **GtxNpu subclass** must add.

---

## 1. ISA Surface

### Table Stakes (firmware regression won't pass without these)

| Feature | Op-level breakdown | Why required | Complexity |
|---|---|---|---|
| **ISA-FW-DISPATCH** — Firmware funct7=0x04–0x07 dispatch (gem5 simplified) | `dispatch_mm` (0x04), `dispatch_vec` (0x05), `dispatch_act` (0x06), `dispatch_dma` (0x07); operands read from `GSPR_GTX_OPERAND1/2/3`, sub-op from `GSPR_GTX_OPCODE` | Used by every firmware kernel that targets gem5-compatible encoding. `run_tests_n1s16.sh` invokes these | MEDIUM |
| **ISA-FW-SPR** — `WRSPR` (f7=0x00) / `RDSPR` (f7=0x01) gem5 path | `WRSPR` writes `SPR[rs1]=rs2` (xs1=1,xs2=1,xd=0); `RDSPR` writes `rd=SPR[rs1]` (xs1=1,xs2=0,xd=1). Disambiguated from MM (also f7=0x00/0x01) by `insn.rs1!=0` heuristic | All kernels stage operands via SPR before dispatch; without this nothing else can run | LOW |
| **ISA-FW-WSPLIT/WJOIN** — Plan boundary markers (f7=0x02/0x03) | `wsplit` starts timing section, sets `wsplit_seen=true`; `wjoin` returns elapsed cycles in `rd` and (if `GTX_NO_EXIT` unset) calls `exit(0)` to break firmware infinite-loop scaffolding | Firmware uses `WJOIN` as termination signal. Without auto-exit, .elf hangs | LOW |
| **ISA-ISS-FULL** — Full ISS encoding (funct7 = 0x00–0x7F) | 64+ distinct funct7 values, see `gtx_npu_disasm.inc` for the canonical list. Examples: MM=0x00, MMC=0x01, IM2COL=0x08/0x09, scalar arith=0x10, FMADD_S=0x11, MIN/MAX_S=0x13, vec arith=0x18, FMADD_V=0x19, DOT/SUM=0x1A, math=0x1C, sign=0x1D, round=0x1E, clamp/accum/logic=0x1F, format cvt=0x20–0x25, activation=0x28–0x2F, pooling=0x30/0x31, transpose/fill=0x38/0x39, DMA=0x40–0x45, SPR/OPSET=0x48–0x4C, credit=0x50–0x53, IMM=0x54–0x5D, sync=0x70–0x7F | gem5 path is a fast-path subset; full ISS path is what the SystemC HW sim used and what most regression .elf binaries assume. Coexistence is required | HIGH |
| **ISA-ROCC-XS1-WORKAROUND** — When `xs1=0`, Spike passes `-1` instead of `rs1` value | C++ uses `p->get_state()->XPR[insn.rs1]` to bypass; Python must do the same via `proc.state.XPR[insn.rs1]` | RISC-V RoCC convention. Required for FW-DMA, FW-VEC, FW-MM where firmware sets xs1=0 to skip register read | LOW |
| **ISA-DISASM** — `get_disasms()` returning ~120 instruction entries | Mirror `gtx_npu_disasm.inc`: helpers `add_r(name, funct7)` and `add_rf3(name, funct7, funct3)` over custom-0 (0x0b) and custom-1 (0x2b, warp control: `warp_start_p/t/s`, `warp_end_p/t/s`, `warp_split`, `warp_join` at funct3=0–7) | Without disasm, `--log-instructions` and Spike trace are unreadable; debugging firmware is impossible. ~5 minor sub-tables (warp control on custom-1 are funct3-discriminated) | MEDIUM |
| **ISA-FUNCT3-IS-FLAG** — `funct3 = {xd, xs1, xs2}` is NOT a generic field | Used by gem5 to encode operand presence flags AND (under firmware encoding) sub-variant selection. `firmware_mm_op` reads `funct3 = (xd<<2)|(xs1<<1)|xs2` to pick between mm.s/mm.o/mm/mm.v/mm.t (variant 0/1/2/3/7) | Without this decoding all firmware MM variants collapse into one op | LOW |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **ISA-PY-TABLE** — Pure-Python funct7 dispatch table (`{0x18: self._handle_vec_arith, …}`) instead of C++ switch | User can `instance.dispatch_table[0x10] = my_custom_handler` to live-patch a single op without rebuild — biggest win over `libgtx_npu.so` | LOW |
| **ISA-DECODE-INTROSPECT** — `instance.disasm(insn)` exposes Python-side decoded form (funct7/funct3/rs1/rs2/rd) for live REPL debugging | Spike's C++ disasm is pretty-printed text only; we can return a structured dict | LOW |

### Anti-Features

| Feature | Why NOT building | PROJECT.md reference |
|---|---|---|
| **GTX commitlog** (`--enable-gtxcommitlog`, GPR commit log) | User explicitly excluded — "회귀/검증과 직교한 부가 기능" | Out of Scope §3 |
| **Custom GPR-commit hooks** beyond standard RoCC `xd` writeback | Spike already handles RoCC writeback via `custom0` return value; reinventing risks divergence | — |

---

## 2. Memory Hierarchy

### Table Stakes

| Feature | Op-level breakdown | Why required | Complexity |
|---|---|---|---|
| **MEM-GSPR** — Global SPR map (0x000–0x3FF) | `gspr: dict[int, int]` keyed by 16-bit addr; key entries: `GSPR_GTX_RUN(0x000)`, `GSPR_GTX_OPERAND1(0x001)`, `GSPR_GTX_OPERAND2(0x002)`, `GSPR_GTX_OPERAND3(0x003)`, `GSPR_GTX_OPCODE(0x004)` | Every dispatch reads operands from these. `wr_spr/rd_spr` route by addr range | LOW |
| **MEM-NSPR** — Per-NEST SPR map (0x400–0x7FF), 4 banks | `nspr[nest_id]: dict[int,int]`. Key entries: `NSPR_THREAD_MASK(0x400)`, `NSPR_SHARED_MASK(0x401)`, `NSPR_TYPE(0x402)`, `NSPR_OP_MODE(0x403)`, `NSPR_CLEAR(0x700)`, `NSPR_SDLE_STATUS(0x780)`, `NSPR_CREDIT_COUNT(0x781)`, `NSPR_CREDIT_ERROR(0x782)` | Used by mexec, credit system, type tagging | LOW |
| **MEM-LSPR** — Per-SPU SPR map (0x800–0xBFF), 64 banks (4×16) | `lspr[nest_id][spu_id]: dict[int,int]`. Address operands `LSPR_SPM_ADDRA(0x900)`, `_ADDRB(0x901)`, `_ADDRC(0x902)`, `_ADDRR(0x903)` are read by every MM/VEC/ACT op | All compute ops dereference LSPR addresses to L1 byte offsets — without this, no compute lands at the right place | LOW |
| **MEM-L0** — 1 KB scratchpad per SPU (`np.uint8` array, shape `(4, 16, 1024)`) | Stores 32×32-byte SVR registers (FP16 little-endian). Read/write helpers `rd_l0(off)/wr_l0(off, v)` matching SystemC TLM byte order. Used by `_imm` ops and `MM_O/MM_V` scalar writeback | Scalar reductions (DOT, VSUM, MM_O, ESUM) target L0 with rest-zeroed semantics | LOW |
| **MEM-L1** — 384 KB SPM per SPU (`np.uint8` array, shape `(4, 16, 384*1024)`) | FP16 little-endian byte store. Helpers `rd16(base, idx)/wr16(base, idx, v)` with `(base + idx*2) % GTX_L1_SIZE` modular addressing | Largest hot path — every compute op reads/writes L1. **bit-exact byte order required (SystemC TLM)** | LOW |
| **MEM-L2** — 16 MB per NEST (`np.uint8`, shape `(4, 16*1024*1024)`), 16 banks of 1 MB | DMA target for DDR↔L2 (S-loop) and L2↔L1 (T-loop). `l2_read(off, dst, len)` / `l2_write(off, src, len)` helpers. Bank conflict tracking (`bank_busy[16]`) optional for v1 | DMA pipeline staging — every firmware kernel uses L2 | LOW |
| **MEM-DDR** — 4 GB DDR (`np.uint8`, length `0x100000000`), base `0x370000000` | Backing store for tensors. `ensure_ddr()` lazy-allocates. `ddr_offset(addr)` translates physical→buffer offset. Must be contiguous (NumPy ndarray) | Firmware reads inputs from DDR, writes outputs to DDR | LOW |
| **MEM-DDR-INIT** — `ddr_init_from_file(path)` honoring two byte orders | LTR (default, objcopy/standard hex) and `GTX_DDR_REVERSED=1` (right-to-left, SystemC HW sim convention). 32-byte bus-word lines. Skip lines starting `#` or `@` (address marker) | HW-sim-derived golden hex files require reversed mode; firmware-emitted files use LTR. Both must work | MEDIUM |
| **MEM-DDR-DUMP** — `ddr_dump_to_file(path, addr, size)` with both byte orders | Mirror init — emit 32-byte hex lines, reversed when `GTX_DDR_REVERSED=1`. `endp` calls this when `GTX_DDR_DUMP` env is set, with `GTX_DDR_DUMP_ADDR` / `GTX_DDR_DUMP_SIZE` overrides | Required by `verify.py` flow: dump → diff vs golden | LOW |
| **MEM-LE-BYTE-ORDER** — All L0/L1 FP16 access strictly little-endian | `raw = buf[off] | (buf[off+1] << 8)` on read, `buf[off]=v&0xFF; buf[off+1]=(v>>8)&0xFF` on write — SystemC TLM x86 host convention | Bit-exact comparison fails if BE — this is the single biggest porting trap | LOW (but easy to forget) |
| **MEM-MMU-ACCESS** — `read_from_dram<T>(addr) / write_to_dram<T>(addr, T)` via Spike `mmu_t` | C++ uses `processor_t::get_mmu()` for individual element access (firmware CPU-side), `ddr_data()` DMI host pointer for bulk DMA. Python equivalent: pybind11-bound `proc.get_mmu()` for scalar access, NumPy slice for bulk | Some firmware mexec paths read DDR via mmu — required for instruction fetch from DDR microcode | MEDIUM |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **MEM-PY-NDARRAY-VIEWS** — `gtx.l1_view(nest, spu, dtype=np.float16)` returns a NumPy view, not a copy | User can `assert np.allclose(gtx.l1_view(0,0)[100:200], golden_slice)` directly without reading hex files | LOW |
| **MEM-PY-SLICING** — `gtx.ddr[0x370010000:0x370020000]` Python slice returns NumPy view | Bypasses dump-then-diff round trip during interactive debugging | LOW |
| **MEM-PY-TYPED-ACCESSORS** — `gtx.read_fp16(l1, off) / read_int8 / read_fp8` matching `gtx_fp16_to_32` etc. | Avoids re-implementing IEEE 754 conversion in user notebooks | LOW |
| **MEM-PY-SNAPSHOT** — `gtx.snapshot()` returns dict of NumPy arrays + SPR dict; `gtx.restore(snap)` | Replay debugging — capture state before flaky op, replay with modified inputs | MEDIUM |

### Anti-Features

| Feature | Why NOT building | Reference |
|---|---|---|
| **MMIO-via-libvfio-user for L1** | Wheel build complexity; PROJECT.md Out of Scope explicitly excludes vfio-user | Out of Scope §2 |
| **CUDA L1 mirror** (`gtx_cuda_get_l1`, `gtx_cuda_sync_to_gpu`) | NumPy backend decision makes GPU mirroring meaningless; excludes CUDA toolkit dependency | Out of Scope §1 |
| **`gtx_l1_device_t` MMIO bus shadow** | This is a Spike-internal optimization for CPU-volatile-pointer access; firmware regression doesn't need it for v1. Defer to v2 if `cpu-l1` tests are needed | — |

---

## 3. Compute Operations

### Table Stakes — MM Subsystem (NPU core, GTX-MM-01)

| Op | Function | Inputs | Output | Complexity |
|---|---|---|---|---|
| **MM-CORE** — `gemm_core(spu, M, K, N, has_bias, gspr_ref)` static helper | Phase 1: load A (MxK) and B (KxN, transposed) from L1 ADDRA/ADDRB FP16→FP32. Phase 2: GEMM C=A·B as FP32. Phase 3: optional FP32 bias from ADDRC | A,B at L1 ADDRA/ADDRB | C: `np.float32` (M*N,) | MEDIUM (use `np.matmul`) |
| **MM-EXEC** — `exec_mm(nest, spu, M, K, N, has_bias)` | MM/MMC: C=A·B [+ FP32 bias from ADDRC] → FP16 to ADDRR | LSPR ADDRA/B/[C] | L1 at ADDRR (FP16) | LOW |
| **MM-EXEC-S** — `exec_mm_s(nest, spu, M, K, N, is_accumulate)` | MM_S/MMC_S: C=A·B [+prior FP32 bias from ADDRC] → FP32 to ADDRC | LSPR ADDRA/B/[C] | L1 at ADDRC (FP32) | LOW |
| **MM-EXEC-O** — `exec_mm_o(nest, spu, A_row, A_col, B_col, l0_addr, has_bias)` | MM_O/MMC_O: row-sum of A (B is implicit ones) [+ mxe_accum] → FP16 scalar to L0[l0_addr*32], rest-zeroed; updates `mxe_accum[nest][spu]` | LSPR ADDRA, mxe_accum | L0 scalar | LOW |
| **MM-EXEC-V** — `exec_mm_v(nest, spu, vec_len, l0_addr, has_bias)` | MM_V/MMC_V: dot(A,B) [+mxe_accum] → FP16 scalar to L0[l0_addr*32] | LSPR ADDRA/B, mxe_accum | L0 scalar | LOW |
| **MM-EXEC-T** — `exec_mm_t(nest, spu, M, K, N, has_bias)` | MM_T/MMC_T: (A·B [+bias])^T → FP16 to ADDRR (transposed write) | LSPR ADDRA/B/[C] | L1 at ADDRR (transposed) | LOW |
| **MM-FW** — `firmware_mm_op(p, insn, is_accumulate)` (f7=0x00/0x01) | Decode `funct3` → variant: 0=mm.s, 1=mm.o, 2=mm, 3=mm.v, 7=mm.t. Pack rs1 = `colB[63:48] | colA[31:16] | rowA[15:0]`. HW convention: 0 in 16-bit field means 65536. l0_addr from `GSPR_GTX_OPERAND3 & 0x1F` | rs1 packed dims, GSPR | calls one of the exec_mm_* | LOW |
| **MM-MXE-ACCUM** — `mxe_accum[GTX_NUM_NESTS][GTX_SPUS_PER_NEST]` FP32 per-SPU accumulator | Stores sum across MM_S → MMC_S → … chain. Read by MMC_O/MMC_V variants for "+accum" | per-SPU FP32 | scalar | LOW |

### Table Stakes — VEC Subsystem (GTX-VEC-01)

| Op group | Op-level breakdown | L1 (vv) path | L0 (ii) path | Notes |
|---|---|---|---|---|
| **VEC-ARITH** — `exec_vector_op(nest, spu, op, length)` | 4 base ops: `GTX_VEC_ADD/SUB/MUL/DIV` (codes 0–3) | reads L1[ADDRA], L1[ADDRB], writes L1[ADDRR] | n/a (II uses `exec_vector_imm`) | LOW |
| **VEC-FMADD** | `GTX_VEC_FMADD` (4) — `r=a*b+c` | L1: ADDRA, ADDRB, ADDRC → ADDRR | II: `exec_vector_imm` with `GTX_IMM_FMADD` reading c from b_reg[9:5] | LOW |
| **VEC-VSUM** — reduction | `GTX_VEC_VSUM` (5): sum L1[ADDRA], scalar to L1[ADDRR][0] AND L0 SVR[0] | scalar reduction | n/a | **FP32 internal accum; one final FP16 cast** |
| **VEC-DOT** — reduction | `GTX_VEC_DOT` — sum(a*b), scalar to L1[ADDRR][0] AND L0 SVR[0] | yes | n/a | LOW |
| **VEC-MATH** — `VEXP/VSQRT/VLN` (codes 6,7,8) | Element-wise `np.exp/np.sqrt/np.log` (LN: 0 if a≤0) | L1 path | `exec_vfunc_imm` codes `GTX_IMM_SQRT/EXP/LN` (in-place when no opset) | LOW |
| **VEC-SIGN** — `VABS/VNEG/SIGN/STEP` (9,10,13,14) | abs/neg/(0<a)−(a<0)/step(a>0) | L1 path | `exec_vfunc_imm` `GTX_IMM_ABS/NEG/SIGN/STEP` | LOW |
| **VEC-MINMAX** — `MAX/MIN` (11, 12) | element-wise max/min | L1 path | `exec_vec_scalar` reduction `GTX_IMM_MAX/MIN` writes scalar to L0[l0_reg*32] rest-zeroed | LOW |
| **VEC-ROUND** — `CEIL/TRUNC/FLOOR/RNE` (15–18) | `np.ceil/trunc/floor/rint` | L1 path | `exec_vfunc_imm` IMM_CEIL/TRUNC/FLOOR/RNE | LOW |
| **VEC-CLAMP** — `CLAMP_MAX/MIN/ACCUM/ARANGE` (f7=0x1F) | min(a,sv) / max(a,sv) / cumulative sum / arange(start,step) | L1 path; sv from `GSPR_GTX_OPERAND2 & 0xFFFF` (FP16); arange uses [16:31] for step | n/a | LOW |
| **VEC-SASMD** — `exec_vec_scalar(nest, spu, sub_op, vector_size, scalar, scalar2)` for `_VS` variants | GTX_IMM_ADD/SUB/MUL/DIV/FMADD/MAX/MIN. MAX/MIN are L1→L0 reductions (rest-zeroed) | reads ADDRA, writes ADDRR with FP16 broadcast scalar | n/a | LOW |
| **VEC-IMM-SCALAR** — `exec_scalar_imm(nest, spu, sub_op, in_reg, out_reg, scalar, scalar2)` | _IS path: 16 FP16 elements per L0 register | n/a | L0[in_reg*32] → L0[out_reg*32] | LOW |
| **VEC-IMM-VECTOR** — `exec_vector_imm(nest, spu, sub_op, a_reg, b_reg, r_reg)` | _II path: element-wise across 16 elements; FMADD reads c from b_reg[9:5] | n/a | L0 only | LOW |
| **VEC-IMM-VFUNC** — `exec_vfunc_imm(nest, spu, sub_op, in_reg, out_reg)` | Math/sign/round on 16-elem L0 reg. SQRT/EXP/LN cy=4 | n/a | L0 only | LOW |
| **VEC-IMM-BITWISE** — `exec_bitwise_imm(nest, spu, sub_op, a_reg, b_reg, r_reg)` | AND/OR/NOT/SHIFT on raw uint16 in L0. SHIFT: `b_reg[3:0]=amt`, `b_reg[4]=dir(1=left)` | n/a | L0 raw uint16 | LOW |
| **VEC-FW-DISPATCH** — `firmware_vec_op(p, insn)` | Maps (funct7, funct3) → `vec_op`. `funct3 & 4` selects L0 path. F7 0x18 (arith), 0x19 (FMADD), 0x1A (DOT/SUM), 0x1C (math), 0x1D (sign), 0x1E (round), 0x1F (clamp). Stashes rs2 to `GSPR_GTX_OPERAND2` for clamp/arange | Firmware uses these heavily | MEDIUM |

### Table Stakes — ACT Subsystem (GTX-ACT-01)

| Op | Sub-op code | Direction | Notes |
|---|---|---|---|
| **ACT-RELU** | `GTX_ACT_RELU=0` | ADDRA→ADDRR | `max(0, a)` |
| **ACT-TANH** | `GTX_ACT_TANH=1` | **ADDRR→ADDRA (reversed)** | `np.tanh` |
| **ACT-SOFTMAX** | `GTX_ACT_SOFTMAX=2` | ADDRA→ADDRR | numerically stable: subtract max, exp, divide by sum |
| **ACT-GELU** | `GTX_ACT_GELU=3` | **ADDRR→ADDRA (reversed)** | tanh approx: `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` |
| **ACT-SIGMOID** | `GTX_ACT_SIGMOID=4` | **ADDRR→ADDRA (reversed)** | `1/(1+exp(-x))` |
| **ACT-PRELU** | `GTX_ACT_PRELU=5` | **ADDRR→ADDRA (reversed)** | slope from `GSPR_GTX_OPERAND2 & 0xFFFF` (FP16) |
| **ACT-ESUM** | `GTX_ACT_ESUM=6` | reads ADDRR | sum(exp(x − max_val)) + accum, max_val from op2[15:0], accum from op2[31:16]; writes scalar to L0 at op3 register |
| **ACT-IMM** — `exec_act_imm(nest, spu, sub_op, in_reg, out_reg, param)` | L0 16-element path | per-sub-op | mirrors L1 set |
| **ACT-SOFTMAX-IMM** — `exec_softmax_imm(nest, spu, sub_op, in_reg, out_reg, max_val, accum_val)` | L0 reduction | — | ESUM/SOFTMAX on 16-elem L0 register |
| **ACT-POOL** — `exec_pooling(nest, spu, is_max, length, kernel_size)` (f7=0x30/0x31) | reads ADDRA, writes ADDRR | output length = length/kernel_size; avg-pool canonicalizes signed zero | LOW |
| **ACT-FORMAT-CVT** — `exec_format_cvt(nest, spu, cvt_type, length)` (f7=0x20–0x25) | scvt_qh/hq (FP8↔FP16), scvt_ih/hi (INT8↔FP16), scvt_hn (INT32→FP16), fcvt_sh/hs (FP32↔FP16), fcvt_dh/hd (FP64↔FP16) | scale+offset packed in `GSPR_GTX_OPERAND2`: `[offset:16 | scale:16]` | MEDIUM |

### Table Stakes — DMA Subsystem (GTX-DMA-01)

| Op | Function | Notes |
|---|---|---|
| **DMA-2D** — `exec_dma_2d(nest, l2_addr, l1_addr, width, height, is_load, ctx, l2_stride=0)` | Generic 2D DMA used by Mode 3 (P+S) | LOW |
| **DMA-LOAD-SVR** — `exec_load_svr(nest, spu, l1_addr, l0_reg)` | L1 → L0 transfer of one 32-byte SVR register | LOW |
| **DMA-STORE-SVR** — `exec_store_svr(nest, spu, l1_addr, l0_reg)` | L0 → L1 transfer of one 32-byte SVR register | LOW |
| **DMA-TRANSPOSE-L1** — `exec_transpose(nest, spu, rows, cols)` | In-L1 FP16 matrix transpose | LOW |
| **DMA-TRANSPOSE-DDR** — `exec_transpose_ddr(src, dst, dim2, dim1, dim0, p2, p1, p0)` | DDR-to-DDR 3D tensor transpose/permute (FP16) | MEDIUM |
| **DMA-FILL** — `exec_fill(nest, spu, length, fill_val)` | Fill L1 region with constant FP16 | LOW |
| **DMA-FW** — `firmware_dma(p, insn)` (f7=0x40) | LOAD (xs2=0), STORE (xs2=1), COPY (xs1=1, !xs2 → L1↔L1). Packs: `addr_hi[63:27] | addr_lo[27:0]` in rs1; `height[63:48] | length[47:32] | stride[31:0]` in rs2; rs3 from `GSPR_GTX_OPERAND3` (opset). S-loop: DDR↔L2; T-loop: L2↔L1; HW convention: length=0 means 65536, height=0 means 1 | HIGH |
| **DMA-DEFERRED-STORE** — Deferred L2→DDR store queue | C++ defers L2→DDR until `endp`/`flush_deferred_ddr_stores`. Snapshot vs ref-based per `plan_has_tloop` | Required to match firmware semantics where T-loop produces L2 results consumed by S-loop store at endp | HIGH |
| **DMA-LOAD-3D / STORE-3D** (f7=0x41 funct3=4/5) | 3D strided DMA | MEDIUM |
| **DMA-MCAST** (f7=0x42, 0x44) | mcast.s2l (SRAM→L1), mcast.g2s/s2s/copy_mem | MEDIUM |
| **DMA-IM2COL-N/D** (f7=0x08/0x09) | im2col normal & depthwise; reads ADDRR, writes ADDRA. op1=out_h/out_w, op2=kh/kw/strides, op3=pad/C | MEDIUM |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **CMP-PY-OVERRIDE** — Per-op override hooks: `instance.before_mm(insn)`, `instance.after_mm(C, addr_r)` | User can intercept C between gemm and FP16 cast — drop-in numerical experimentation without rebuild | LOW |
| **CMP-NUMPY-FAST-MM** — Use `np.matmul(A.astype(f32), B.T.astype(f32))` instead of triple loop | C++ MM has ~3 backend tiers (cuBLAS, OpenBLAS, manual). NumPy's BLAS wrapper gets us ~OpenBLAS speed for free | LOW |
| **CMP-PY-OP-TRACE** — `instance.trace_ops` list captures (insn_funct7, addr_r, output_view) for each compute | One-line replacement for `--enable-gtxcommitlog`, decoupled from build flags | LOW |

### Anti-Features

| Feature | Why NOT building | Reference |
|---|---|---|
| **CUDA backend** (`gtx_cuda_vector_op`, `gtx_cuda_activation`, etc.) | Pure-Python decision excludes GPU. NumPy is the chosen perf floor | Out of Scope §1 |
| **OpenMP parallel for / thread pool** (`GTX_USE_POOL`, `gtx_thread_pool_t`) | NumPy already parallel via BLAS; Python GIL constrains gains; v1 perf target is "regression in tens of minutes" — adequate without | — |
| **Bank-conflict modeling** (`bank_busy[GTX_L2_NUM_BANKS]`) | Functional model only; cycle-accurate behavior is non-goal in v1 | — |
| **Cycle-accurate `gtx_cycles::*` counter family** | C++ has detailed per-op cycle formulas. v1 returns 0 (or rough estimate) — sufficient since `GTX_FUNCTIONAL_ONLY` semantics are normal mode | — |

---

## 4. Loop Control (P/S/T Warp State Machine)

### Table Stakes (GTX-CORE-02)

| Feature | Op-level breakdown | Why required | Complexity |
|---|---|---|---|
| **LOOP-PSTART** — `startp(rs1, rs2)` | NEST id from `(rs2 & 0x400) ? (rs2 & 0x3F) : rs1`. Bounds: id < 4. Sets `tmu_id=id, is_ploop=true` | Selects target NEST for following ops | LOW |
| **LOOP-PEND** — `endp(rs1, rs2)` | Validates id matches `tmu_id`; clears `is_ploop`. Calls `flush_deferred_ddr_stores()`. Resets `plan_has_tloop=false`. Optional `GTX_DDR_DUMP` env-driven dump if `!wsplit_seen` | Plan boundary. Without DDR flush, S-loop stores never persist | MEDIUM |
| **LOOP-SSTART** — `starts(rs1, rs2)` | GDMAC id (max `GTX_GDMAC_NUM=4`). Must be inside P-loop. Sets `is_sloop=true, curr_id=id` | Selects DMA context for DDR↔L2 | LOW |
| **LOOP-SEND** — `ends(rs1, rs2)` | Clears `is_sloop` if both `is_sloop && is_ploop` | LOW |
| **LOOP-TSTART** — `startt(rs1, rs2)` | SPU id (max `GTX_SPUS_PER_NEST=16`). Must be inside P-loop. Sets `is_tloop=true, curr_id=id, plan_has_tloop=true` | Selects SPU for compute + L2↔L1 DMA | LOW |
| **LOOP-TEND** — `endt(rs1, rs2)` | Clears `is_tloop` | LOW |
| **LOOP-CUSTOM1** — `custom1()` warp control entry, funct3=0–7 | `funct3=0:start_t, 1:end_t, 2:start_s, 3:end_s, 4:wsplit, 5:wjoin, 6:start_p, 7:end_p` (custom-1 opcode 0x2b). `funct3 = (insn.xd<<2) | (insn.xs1<<1) | insn.xs2`. ISS encoding | LOW |
| **DISPATCH-MODE-1** — No loop active | Broadcast op to ALL 4 NESTs × 16 SPUs (64 invocations) | Initial state, used by reset/init kernels | LOW |
| **DISPATCH-MODE-2** — P only | Broadcast within selected NEST (16 SPUs) | Used for NEST-local SPM init | LOW |
| **DISPATCH-MODE-3** — P + S | DMA on selected NEST (DDR↔L2 via `exec_dma_2d`); is_load determined by sub_op==0 or opcode==GTX_OP_DMA | Mode 3 IS the DMA path during plan execution | LOW |
| **DISPATCH-MODE-4** — P + T | Compute on selected (NEST, SPU) only | Hot path for all MM/VEC/ACT | LOW |
| **CONTEXT-CHECK** — `is_valid_in_context(opcode, ctx)` BLOCK rule (D-01) | C1 (no loop), C2 (P+S), C3 (P+T), C4 (P only). Some opcodes illegal outside their context. Check raises `illegal_instruction(*p)` | Required for D-01 conformance; Spike already has `illegal_instruction()` exposed | MEDIUM |
| **CREDIT-STALL** — `use_spu_queue/use_tmu_queue` queue infrastructure (D-06) | Functional model: queues stay empty (DMA instantaneous), but push/pop infrastructure exists for compute-stall semantics on credit-flag set | Functional regression doesn't trip queues — but inserting credit stall MUST not corrupt state | LOW (stub for v1) |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **LOOP-PY-CONTEXT-MANAGER** — `with gtx.ploop(0): with gtx.tloop(5): gtx.dispatch_mm(...)` | User scripts can drive NPU directly without writing firmware — useful for op-level unit testing | LOW |
| **LOOP-PY-STATE-DUMP** — `gtx.loop_state` returns `{'p':True,'s':False,'t':True,'tmu_id':0,'curr_id':5}` | Debugging "why didn't my op run on SPU 5" | LOW |

### Anti-Features

| Feature | Why NOT | Reference |
|---|---|---|
| **Cycle-accurate credit modeling** with stall queue draining | Functional model only. `spu_queue/tmu_queue` stay empty in v1 | — |

---

## 5. Verification & Debugging

### Table Stakes

| Feature | Op-level breakdown | Why required | Complexity |
|---|---|---|---|
| **VRF-DDR-DIFF** — `verify.py` integration (GTX-VERIFY-01) | Bundle existing 388-LOC `verify.py` (FP16 ULP/atol comparison, big-endian FP16 parsing, mismatch report). Either ship as `riscv.gtx.verify` module or as a `pyspike-verify` CLI | The single contractually mandated success metric: "회귀가 한 세션 내 끝나야" + bit-exact match. Without `verify.py`, CI can't gate | LOW |
| **VRF-REF-PY** — `verify_ref.py` integration (GTX-VERIFY-02) | Bundle 378-LOC `verify_ref.py`. 32 host-side scalar ops registry: `ABS/NEG/SQR/SQRT/EXP/LOG/CEIL/FLOOR/TRUNC/ROUND/STEP/SGN/SIN/COS/RELU/SILU/SIGMOID/TANH/GELU/GELU_ERF/GELU_QUICK/ELU/SOFTPLUS/LEAKY_RELU/HARDSIGMOID/HARDSWISH/ADD/SUB/MUL/DIV/ADD1/SCALE/FILL`. Reads `_input.txt`/`_ref.txt` from `test/<OP>/n1s16/data/`. Comparison: ULP=1 OR abs_diff<0.01 | Per-op host-side validation. Used during porting to confirm "this op produces the right answer for THIS input" | LOW |
| **VRF-FP16-CONV** — Python fp16↔fp32 helpers matching `gtx_fp16_to_32` / `gtx_fp32_to_16` | Already in `verify.py` (lines 27–129). Mirror behavior of C++ helpers for RNE rounding, subnormals, NaN/Inf | Test harnesses need this for golden generation | LOW |
| **VRF-OPLEVEL-TESTS** — pytest suite with one test per op | Mirror `verify_ref.py` op set + edge cases (subnormal, NaN, ±Inf). Compare Python NPU result to numpy scalar reference | Must run in <30s to keep dev loop tight | MEDIUM |
| **VRF-ELF-REGRESSION** — `.elf` regression harness (GTX-FW-01) | Mirror existing `run_tests_n1s16.sh` (per-NEST 1, per-SPU 16) and `run_llext_tests.sh`. Each test: load `.elf`, run, dump DDR, diff vs golden hex with `verify.py --fp16 --ulp 1 --atol 0.001` | The ONE acceptance criterion ("Core Value": firmware regression bit-exact). Bundle .elf + .hex assets into wheel | HIGH (asset packaging + harness) |
| **VRF-FIRST-MISMATCH** — Detailed mismatch diagnostics | Already in `verify.py`: stores up to 10 first mismatches with `(offset, result_raw, golden_raw, fp16_val, ulp_dist, abs_diff)`. Print on FAIL | Without this, "regression failed" is debugging hell | LOW (already exists) |
| **VRF-REVERSED-DDR** — Both byte orders supported in dump/init | `GTX_DDR_REVERSED=1` for SystemC HW sim hex; LTR for objcopy/standard. Cached into `ddr_reversed` at reset() | HW-sim-derived golden hex requires reversed; firmware-emitted requires LTR | LOW |
| **VRF-DEBUG-INSN** — `debug_wr` (f7=0x7D), `debug_rd` (f7=0x7E) instructions | Direct FP16 read/write to L1 from CPU side; useful for firmware self-test scaffolding | Some firmware uses these for instrumented assertions | LOW |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **VRF-PY-NUMPY-CMP** — `gtx.compare_to_numpy(reference_fn)` runs reference on snapshot inputs and compares output | One-step alternative to dump→hex→`verify.py` round trip during interactive debugging | LOW |
| **VRF-PY-ULP-NDARRAY** — `gtx.ulp_diff(a_ndarray, b_ndarray)` returns ULP distance ndarray | Visual heatmaps of error in Jupyter | LOW |
| **VRF-PY-OP-OVERRIDE-DIFF** — Override one op with reference implementation, run regression, see WHICH op breaks | Bisect-style debugging unique to Python port | MEDIUM |

### Anti-Features

| Feature | Why NOT | Reference |
|---|---|---|
| **GTX commitlog** | User-excluded debug trace mode | Out of Scope §3 |
| **GTX_QUIET stderr suppression** | Debug-only build flag, not a runtime feature; Python uses `logging` instead | — |
| **`-DGTX_PROFILE` cycle profiler** | C++ chrono-based; Python ports Profile via `cProfile` if needed — not a feature | — |

---

## 6. Python Ergonomics

### Table Stakes

| Feature | Op-level breakdown | Complexity |
|---|---|---|
| **PY-ROCC-SUBCLASS** — `class GtxNpu(riscv.isa.ROCC)` with `@register("gtx")` | Required pyspike pattern. Implement `name`, `custom0/1/2/3`, `get_disasms`, `get_csrs`, `reset` | LOW |
| **PY-CUSTOM0-DISPATCH** — `custom0(proc, insn, xs1, xs2) -> reg_t` decoding `insn.funct` | Switch on funct7; route to `_handle_wrspr / _handle_rdspr / _handle_dispatch_mm / …` | LOW |
| **PY-CUSTOM1-WARP** — `custom1(proc, insn, xs1, xs2) -> reg_t` for warp control | funct3 → start_t/end_t/start_s/end_s/wsplit/wjoin/start_p/end_p | LOW |
| **PY-RESET** — `reset(proc)` initializing `sp = 0x80100000`, clearing all SPRs/L0/L1/L2/DDR, caching env vars (`GTX_DDR_REVERSED`, `GTX_NO_EXIT`, `GTX_DDR_DUMP*`) | sp init avoids `addi sp,sp,-16` trap from sp=0 | LOW |
| **PY-WJOIN-EXIT** — `WJOIN` invokes `sys.exit(0)` (or `raise SystemExit(0)`) when `GTX_NO_EXIT` unset | Mirrors C++ `exit(0)`. Firmware enters infinite loop after WJOIN — without this, .elf hangs | LOW |
| **PY-CSR-INJECTION** — `get_csrs(proc) -> List[csr_t]` exposing GTX-specific CSRs if any | Currently the C++ implementation uses SPRs (custom address space 0x000–0xBFF), not CSRs. v1 may return `[]` unless ISA needs control CSRs | LOW |

### Differentiators (THIS is where Python wins over C++)

| Feature | Value Proposition | Complexity |
|---|---|---|
| **PY-OVERRIDE-HOOK** — Per-op `before_<op>/after_<op>` hooks (e.g., `before_mm(self, insn, A, B) -> Optional[A_modified]`) | One-line numerical experiments without rebuild. Killer feature for ISA research | LOW |
| **PY-OP-INSTRUMENTATION** — `gtx.enable_trace(); gtx.run(...); records = gtx.trace` returns list of `{op:'mm', funct7:0x00, addr_r:..., shape:...}` | Replaces `--enable-gtxcommitlog` build flag with runtime knob | LOW |
| **PY-CUSTOM-FUNCT7** — `gtx.register_funct7(0x7E, my_handler_fn)` lets user add unimplemented op without subclassing | ISA experimentation made trivial | LOW |
| **PY-CSR-LIVE-WRITE** — `gtx.gspr[GSPR_GTX_OPERAND1] = 0xDEADBEEF` from REPL between ops | Manual operand staging for op-level unit tests without firmware | LOW |
| **PY-SNAPSHOT-RESTORE** — `snap = gtx.snapshot(); gtx.restore(snap)` deep-copies state (gspr/nspr/lspr/L0/L1/L2/DDR/loop_state/mxe_accum) | Replay debugging — pin snapshot before flaky op, retry with mods. Impossible in C++ libgtx_npu.so | MEDIUM |
| **PY-OP-MOCK** — `gtx.mock_op('mm', return_value=fixed_result)` short-circuits compute for isolation | Isolating "is the bug in MM or in the dispatch?" | MEDIUM |
| **PY-NUMPY-VIEWS** — `gtx.l1(nest, spu, dtype=np.float16)` returns NumPy view of L1 buffer | Direct ndarray manipulation for test setup; no hex round trip | LOW |
| **PY-DISASM-PY** — `gtx.disasm_insn(insn) -> {'name':'mm', 'funct7':0x00, 'rs1':1, ...}` | Programmatic decode for test assertions ("did this trigger MM?") | LOW |

### Anti-Features

| Feature | Why NOT | Reference |
|---|---|---|
| **`asyncio` / threading for parallel SPU dispatch** | Python GIL + NumPy already parallel; complicates determinism. v1 is single-threaded | — |
| **Auto-reload on file change** | Out of scope; users use `importlib.reload` if needed | — |
| **GUI debugger / waveform viewer** | Way out of v1 scope | — |

---

## 7. Distribution

### Table Stakes (GTX-PKG-01)

| Feature | Op-level breakdown | Why required | Complexity |
|---|---|---|---|
| **PKG-PIP-INSTALL** — `pip install spike` includes `riscv.gtx` subpackage | Existing pyspike wheel pipeline (PYS-EXT-06, manylinux2014_x86_64, Python 3.8–3.12) just adds `riscv/gtx/__init__.py` | LOW |
| **PKG-ENTRY** — `from riscv.gtx import GtxNpu` works after install | One import surface | LOW |
| **PKG-FW-ASSETS** — Bundle `.elf` regression binaries + `.hex` golden files in wheel | Use `pyproject.toml` `[tool.setuptools.package-data]` to include `riscv/gtx/firmware/*.elf` + `riscv/gtx/golden/*.hex`. Total size: estimate <50 MB; if larger, split optional `[regression]` extra | MEDIUM (size budget + asset listing) |
| **PKG-VERIFY-CLI** — `pyspike-verify result.hex golden.hex --fp16 --ulp 1 --atol 0.001` shipped as console_script | Replaces `python3 gtx/verify.py` shell pattern; integrates with `--extlib` flow | LOW |
| **PKG-CLI-DEFAULT** — `pyspike --extlib=riscv.gtx test.elf` works as one-liner | Existing pyspike `--extlib` accepts module names; just need `riscv.gtx` to register on import | LOW |
| **PKG-DOC-EXAMPLE** — `examples/gtx/run_mm_smoke.py` showing `with gtx.ploop(0): with gtx.tloop(0): gtx.dispatch_mm(...)` | First-run sanity check for users | LOW |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **PKG-PYPROJECT-EXTRA** — `pip install spike[gtx-regression]` for the .elf bundle (split if size matters) | Keeps base wheel small; full regression dataset opt-in | LOW |
| **PKG-NOTEBOOK** — Jupyter `examples/gtx/walkthrough.ipynb` (NPU 101) | Lowers learning curve dramatically | MEDIUM |

### Anti-Features

| Feature | Why NOT | Reference |
|---|---|---|
| **Bundle `libgtx_npu.so` (C++) in wheel** | Decision: C++ snapshot stays in `vendor/gtx_cpp_reference/` for golden compare only — not wheel-shipped | PROJECT.md Key Decisions row 1; Out of Scope §5 |
| **Windows / macOS / aarch64 wheels** | manylinux2014_x86_64 baseline only | Out of Scope §4 |
| **Bundle `libvfio-user.so` / vfio-user adapter** | Wheel build complexity disproportionate to v1 value | Out of Scope §2 |
| **CUDA toolkit dependency in wheel** | NumPy backend; CUDA path excluded | Out of Scope §1 |

---

## 8. CSR/SPR Access

### Table Stakes (GTX-SPR-01)

| Feature | Op-level breakdown | Complexity |
|---|---|---|
| **SPR-WR** — `wr_spr(addr, value)` routing 0x000–0x3FF→GSPR, 0x400–0x7FF→NSPR (NEST-scoped via `current_nest()`), 0x800–0xBFF→LSPR (NEST,SPU-scoped) | LOW |
| **SPR-RD** — `rd_spr(addr) -> uint64` mirror of wr_spr | LOW |
| **SPR-FW-WRSPR** — Firmware path (f7=0x00, xs1=1, xs2=1, xd=0): `SPR[rs1] = rs2` | Disambiguated from MM by `insn.rs1!=0` (firmware) or `xs1=xs2=1` (gem5) | LOW |
| **SPR-FW-RDSPR** — Firmware path (f7=0x01, xs1=1, xs2=0, xd=1): `rd = SPR[rs1]` returned via custom0 return value (Spike writes to rd because xd=1) | The pybind11 `custom0` return value gets stored to `rd` — works automatically | LOW |
| **SPR-ISS-WRSPR** — ISS path (f7=0x49) — same semantics as firmware WRSPR but via ISS encoding | LOW |
| **SPR-ISS-RDSPR** — ISS path (f7=0x48) — same as firmware RDSPR | LOW |
| **SPR-OPSET** — OPSET (f7=0x4A): operand staging; writes `GSPR_GTX_OPERAND1/2/3` based on rs2 selector | Used heavily by firmware kernels before dispatch (e.g., set ADDRA/B/C/R into LSPR via WRSPR, then OPSET sub-fields) | LOW |
| **SPR-CPSVR / MVSVR** — CPSVR (0x4B), MVSVR (0x4C): copy/move SVR register data | LOW |
| **SPR-CREDIT** — credit_ld/_st/_ld_chk/_st_chk (f7=0x50–0x53) | Functional v1: stub return 0 (queues unused) | LOW |
| **SPR-SYNC-FAMILY** — mexec/mbar/msync/eom/bar/wait/intr/flush/halt (f7=0x70–0x7F) | mexec runs DDR-microcode loop (HIGH complexity); others are NOPs in functional model except `halt` (raises) and `eom` (return like wjoin) | MEDIUM (mexec) / LOW (rest) |

### Differentiators

| Feature | Value Proposition | Complexity |
|---|---|---|
| **SPR-PY-DICT** — Plain Python `dict[int,int]` for gspr/nspr/lspr (vs C++ `unordered_map`) | Trivially inspectable: `print(gtx.gspr)` shows everything | LOW |
| **SPR-PY-NAMED** — `gtx.gspr_named['GTX_OPERAND1']` accessor | Removes magic-number burden | LOW |
| **SPR-PY-WATCH** — `gtx.watch_spr(LSPR_SPM_ADDRR, callback)` fires on writes | Unit-testing "did the firmware actually set ADDRR before dispatch?" | LOW |

### Anti-Features

| Feature | Why NOT | Reference |
|---|---|---|
| **CSR commit log via `--enable-gtxcommitlog`** | User-excluded; Python `watch_spr` replaces it cleanly | Out of Scope §3 |

---

## Feature Dependency Graph

```
[ISA-FUNCT3-IS-FLAG]                      (foundational decoding rule)
        │
        ├──> [ISA-FW-DISPATCH] + [ISA-ISS-FULL] + [ISA-DISASM]
        │           │                                  │
        │           v                                  v
        │      [ISA-FW-SPR / SPR-WR / SPR-RD]    [PY-DISASM-PY]
        │           │
        │           v
        │      [SPR-OPSET]──┐
        │                   v
        │           (operand staging in GSPR)
        │                   │
        v                   v
[MEM-LSPR / NSPR / GSPR]   [LOOP-PSTART → SSTART → TSTART]
        │                          │
        v                          v
[MEM-L0 / L1 / L2 / DDR]    [DISPATCH-MODE-1..4]
[MEM-LE-BYTE-ORDER]                │
        │                          v
        ├────────> [MM-CORE] ──> [MM-EXEC, EXEC-S, EXEC-O, EXEC-V, EXEC-T] ──> [MM-FW]
        │                              │
        │                              v
        │                        [MM-MXE-ACCUM]    (consumed by MMC variants)
        │
        ├────────> [VEC-ARITH/FMADD/VSUM/DOT/MATH/SIGN/MINMAX/ROUND/CLAMP/SASMD/IMM-*]
        │              │
        │              v
        │         [VEC-FW-DISPATCH]
        │
        ├────────> [ACT-RELU/TANH/SOFTMAX/GELU/SIGMOID/PRELU/ESUM]    (note dir reversal)
        │              │                                                ^^^
        │              v                                                requires
        │         [ACT-IMM / SOFTMAX-IMM / POOL / FORMAT-CVT]           [LSPR ADDRA/R]
        │
        └────────> [DMA-2D / LOAD-SVR / STORE-SVR / FW / DEFERRED-STORE / TRANSPOSE / FILL]
                          │
                          v
                  [LOOP-PEND triggers DMA-DEFERRED-STORE flush] ──> [MEM-DDR-DUMP]

[LOOP-CUSTOM1] ──> [LOOP-PSTART/SSTART/TSTART/etc.]    (entry point)

[VRF-DDR-DIFF + VRF-REF-PY + VRF-OPLEVEL-TESTS] ──> [VRF-ELF-REGRESSION]
        │                          │                       │
        │                          │                       v
        │                          v                  [PKG-FW-ASSETS]
        v                     [VRF-FP16-CONV]              │
[PKG-VERIFY-CLI]                                           v
                                                    [PKG-PIP-INSTALL] ──> [PKG-ENTRY]

[PY-ROCC-SUBCLASS] ─requires─> [pyspike trampolines] (already exists)
        │
        ├──> [PY-CUSTOM0-DISPATCH] ──> all funct7 handlers
        ├──> [PY-CUSTOM1-WARP] ──> [LOOP-CUSTOM1]
        ├──> [PY-RESET] ──> sp init + env caching + state zeroing
        └──> [PY-WJOIN-EXIT] ──> sys.exit(0)
```

### Critical Dependency Notes

- **`MM-FW` requires `SPR-FW-WRSPR/RDSPR` first.** Firmware sequence is "WRSPR ADDRA/B/C/R, OPSET, DISPATCH_MM/firmware MM". Without SPR routing, addresses are zero and MM corrupts L1[0].
- **`DISPATCH-MODE-3` requires `LOOP-PEND` deferred flush.** S-loop L2→DDR stores are NOT applied immediately; they go into `deferred_ddr_stores` and flush at `endp`. Without this, golden hex compare fails.
- **`ACT-PRELU/GELU/TANH/SIGMOID` reverse-direction** is a runtime invariant, not a separate feature. Must be encoded in `exec_activation`'s direction logic. Common porting trap.
- **`VEC-VSUM` precision rule:** FP32 internal accumulator, single FP16 cast at end. For row-split execution, partial sums re-accumulated in FP16 (matches C++ comment). Required for ULP-1 match.
- **`MEM-LE-BYTE-ORDER` underlies every compute op.** All L0/L1 reads and writes must preserve `byte0=lo, byte1=hi`. Single biggest correctness trap.
- **`xs1=0 → -1` workaround** is mandatory for FW-DMA / FW-VEC / FW-MM dispatch — they always set xs1=0 and rely on the extension reading rs1 from `proc.state.XPR[insn.rs1]` directly.

---

## MVP Definition

### Launch With (v1) — Acceptance Gate: 100 % .elf regression pass + bit-exact DDR vs C++

**Phase A — Foundation (ISA + Memory + SPR):**
- [ ] PY-ROCC-SUBCLASS (`GtxNpu(riscv.isa.ROCC)` skeleton + `@register("gtx")`)
- [ ] PY-RESET (sp=0x80100000, state zero, env cache)
- [ ] MEM-GSPR / NSPR / LSPR (dict-based)
- [ ] MEM-L0 / L1 / L2 / DDR (NumPy ndarray)
- [ ] MEM-LE-BYTE-ORDER (helper functions)
- [ ] MEM-DDR-INIT / DDR-DUMP (both byte orders)
- [ ] SPR-WR / SPR-RD / SPR-FW-WRSPR / SPR-FW-RDSPR / SPR-OPSET
- [ ] ISA-ROCC-XS1-WORKAROUND
- [ ] ISA-DISASM (full table from `gtx_npu_disasm.inc`)

**Phase B — Loop Control + Dispatch:**
- [ ] LOOP-PSTART/PEND/SSTART/SEND/TSTART/TEND
- [ ] LOOP-CUSTOM1 (custom-1 funct3 dispatch)
- [ ] DISPATCH-MODE-1/2/3/4 router
- [ ] PY-WJOIN-EXIT
- [ ] CONTEXT-CHECK (D-01)

**Phase C — MM Subsystem (NPU core):**
- [ ] MM-CORE (gemm_core via `np.matmul` FP32)
- [ ] MM-EXEC / EXEC-S / EXEC-O / EXEC-V / EXEC-T
- [ ] MM-MXE-ACCUM
- [ ] MM-FW (firmware MM dispatch with packed rs1 dim decoding)

**Phase D — VEC + ACT + DMA:**
- [ ] VEC-ARITH / FMADD / VSUM / DOT / MATH / SIGN / MINMAX / ROUND / CLAMP
- [ ] VEC-SASMD / IMM-SCALAR / IMM-VECTOR / IMM-VFUNC / IMM-BITWISE
- [ ] VEC-FW-DISPATCH
- [ ] ACT-RELU / TANH / SOFTMAX / GELU / SIGMOID / PRELU / ESUM (with reversed-direction handling)
- [ ] ACT-IMM / SOFTMAX-IMM / POOL / FORMAT-CVT
- [ ] DMA-2D / LOAD-SVR / STORE-SVR / TRANSPOSE / FILL
- [ ] DMA-FW (firmware packed encoding, S-loop and T-loop paths, copy)
- [ ] DMA-DEFERRED-STORE (queue + flush at endp)

**Phase E — Verification + Distribution:**
- [ ] VRF-FP16-CONV + VRF-DDR-DIFF (port `verify.py`)
- [ ] VRF-REF-PY (port `verify_ref.py`)
- [ ] VRF-OPLEVEL-TESTS (pytest, ≥1 per op)
- [ ] VRF-ELF-REGRESSION (.elf harness, ≥1 batch matching `run_tests_n1s16.sh`)
- [ ] PKG-PIP-INSTALL / PKG-ENTRY / PKG-FW-ASSETS / PKG-VERIFY-CLI

### Add After Validation (v1.x)

- [ ] PY-OVERRIDE-HOOK / PY-OP-INSTRUMENTATION / PY-CUSTOM-FUNCT7 (the differentiating ergonomics)
- [ ] PY-SNAPSHOT-RESTORE
- [ ] LOOP-PY-CONTEXT-MANAGER
- [ ] DMA-LOAD-3D / STORE-3D / MCAST / IM2COL-N/D (op coverage expansion if regression demands)
- [ ] mexec full microcode loop (currently only stub if regression doesn't trip it)
- [ ] CMP-PY-OP-TRACE (replaces `--enable-gtxcommitlog`)

### Future (v2+)

- [ ] PCIe-EP / vfio-user reintroduction via `riscv.dev.MMIO`
- [ ] Cycle-accurate timing (port `gtx_cycles::*`)
- [ ] Bank conflict modeling
- [ ] Cython/C extension for hot paths if NumPy is too slow
- [ ] Windows / macOS / aarch64 wheels
- [ ] CUDA backend (only if user explicitly asks; currently excluded)

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---|---|---|---|
| MEM-LE-BYTE-ORDER | HIGH (correctness gate) | LOW | P1 |
| ISA-ROCC-XS1-WORKAROUND | HIGH (correctness gate) | LOW | P1 |
| MM-CORE / MM-EXEC family | HIGH (NPU core) | MEDIUM | P1 |
| MM-FW (firmware encoding) | HIGH | LOW | P1 |
| DMA-FW + DMA-DEFERRED-STORE | HIGH | HIGH | P1 |
| LOOP P/S/T + DISPATCH 4-mode | HIGH | LOW | P1 |
| ACT reversed-direction handling | HIGH (silent corruption otherwise) | LOW | P1 |
| VEC-VSUM FP32-accum precision | HIGH | LOW | P1 |
| VRF-DDR-DIFF (verify.py) | HIGH (CI gate) | LOW | P1 |
| VRF-ELF-REGRESSION harness | HIGH (acceptance) | HIGH | P1 |
| ISA-DISASM full table | MEDIUM | MEDIUM | P1 |
| PKG-FW-ASSETS in wheel | HIGH | MEDIUM | P1 |
| PY-OVERRIDE-HOOK | HIGH (differentiator) | LOW | P2 |
| PY-OP-INSTRUMENTATION | HIGH (differentiator) | LOW | P2 |
| PY-SNAPSHOT-RESTORE | MEDIUM | MEDIUM | P2 |
| LOOP-PY-CONTEXT-MANAGER | MEDIUM | LOW | P2 |
| DMA-LOAD-3D / IM2COL / MCAST | MEDIUM (regression-driven) | MEDIUM | P2 (or P1 if regression needs) |
| mexec full impl | MEDIUM | HIGH | P2 |
| Cycle-accurate timing | LOW | HIGH | P3 |
| CUDA / vfio-user / commitlog | NONE (Out of Scope) | — | NEVER (v1) |

---

## Sources

- **Primary (HIGH confidence — read directly):**
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu.h` (1382 LOC — class declaration, encoding tables, conversion helpers)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_dispatch.cc` (4-mode dispatch + ISS opcode router)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_mm.cc` (gemm_core, exec_mm/_s/_o/_v/_t, firmware_mm_op)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_vec.cc` (757 LOC — vector ops + firmware_vec_op)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_act.cc` (487 LOC — activation/pool/format-cvt + L0 IMM variants)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_dma.cc` (600 LOC — exec_dma_2d/load_svr/store_svr/transpose/transpose_ddr/fill, firmware_dma, deferred-store flush, ddr_init/dump)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_loop.cc` (147 LOC — startp/endp/starts/ends/startt/endt)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_npu_disasm.inc` (244 LOC — full instruction disasm table)
  - `~/NIGHTLY/gtx_spike/gtx/gtx_params.h` (HW constants)
  - `~/NIGHTLY/gtx_spike/gtx/verify.py` (388 LOC — ULP/atol diff)
  - `~/NIGHTLY/gtx_spike/gtx/verify_ref.py` (378 LOC — 32 host-side scalar ops)
  - `~/NIGHTLY/gtx_spike/gtx/CLAUDE.md` (memory hierarchy + encoding overview)

- **Project context:**
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/PROJECT.md` (Active reqs GTX-CORE-01…GTX-REF-01, Out of Scope items, Key Decisions)
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/codebase/ARCHITECTURE.md` (existing pyspike trampoline surface)
  - `/mnt/e/14_NIGHTLY/pyspike/.planning/codebase/STRUCTURE.md` (where new code lands)
  - `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/isa.py` (ISA / ROCC base classes, register decorator)
  - `/mnt/e/14_NIGHTLY/pyspike/examples/xhuimt/__init__.py` (extension subclass pattern reference)
  - `/mnt/e/14_NIGHTLY/pyspike/examples/amba/uart_lite.py` (`riscv.dev.MMIO` device pattern reference)

---
*Feature research for: pyspike + GTX NPU Python port*
*Researched: 2026-05-04*
