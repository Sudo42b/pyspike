# Project Research Summary

**Project:** pyspike + GTX NPU (Pure-Python NumPy-backed RoCC Port)
**Domain:** Bit-exact functional model port — C++ `gtx_npu_t : rocc_t` → Python `riscv.isa.ROCC` subclass on top of pyspike's existing pybind11 trampolines
**Researched:** 2026-05-04
**Confidence:** HIGH

---

## Executive Summary

This is a **brownfield port** of an already-validated C++ NPU functional model (`~/NIGHTLY/gtx_spike/gtx/`, 11 .cc + .h files, single class `gtx_npu_t : rocc_t` covering NEST(4)×SPU(16) memory hierarchy + MM/VEC/ACT/DMA subsystems) into pure Python on top of pyspike's existing `riscv.isa.ROCC` trampoline surface. The C++ reference has already been ULP-equivalence-validated against SystemC HW sim, so the Python port's only acceptance gate is **"every existing .elf firmware regression produces a DDR dump that passes `verify.py --fp16 --ulp 1 --atol 0.001` against the C++ golden"** — there is no design freedom, only fidelity.

All four research streams converge on the same recommended approach: **NumPy 1.26.x (pinned `>=1.20,<2.0` to preserve pyspike's cp38 wheel matrix), FP16 stored as `np.uint8`/`np.uint16` byte arrays in little-endian layout, FP32 internal accumulation with a single trailing `np.float16` cast** for every reduction (VSUM, DOT, MM_O/V, SOFTMAX, ESUM). The package lands at `src/main/python/riscv/gtx/` (not `examples/`, because it ships in the wheel as a v1 product feature) split into `npu.py` + `memory.py` + `dispatch.py` + `loop.py` + `spr.py` + `disasm.py` + `ops/{mm,vec,act,dma,...}.py` to mirror the C++ file split. No new C++ code, no numba/cython/JAX/torch — pure NumPy gets us to the "regression in tens of minutes" budget.

The risk surface is dominated by **bit-exactness traps that look like correct code**: little-endian-vs-big-endian byte order at the verify.py↔L1 seam (Pitfall 1), per-element FP16 casts that secretly truncate reductions (Pitfall 2), `mxe_accum` continuity across MM_O/MMC_O chains (Pitfall 3), the Spike `xs1=0 → -1` quirk that requires reading `proc.get_state().XPR[insn.rs1]` directly (Pitfall 4), the funct7=0x00 collision between gem5-simplified WRSPR and ISS-full MM (Pitfall 5), and ACT direction reversal where PRELU/GELU/TANH/SIGM read ADDRR→write ADDRA while RELU/SOFTMAX/ESUM go the other way (Pitfall 9). Mitigation is uniform: implement explicit FP16↔FP32 bit-manipulation helpers (port `gtx_fp16_to_32`/`gtx_fp32_to_16` from `gtx_npu.h:89-151`) and treat every C++ source comment as authoritative spec, not commentary.

---

## Key Findings

### Recommended Stack

NumPy is the single new runtime dependency. Everything else (pybind11 trampolines, `@register` decorator, `PYSPIKE_LIBS` bootstrap, `--extlib=` CLI, cibuildwheel manylinux2014 matrix, pytest+pytest-asyncio) already exists in pyspike and is validated. Test layer additions: `pytest-xdist` (parallel .elf regression) and `hypothesis` (FP16 property tests). cp38 backport: `importlib_resources>=5.0; python_version < "3.9"` for asset access.

**Core technologies:**
- **NumPy 1.26.4 (pinned `>=1.20,<2.0`)** — last 1.x line; preserves cp38 wheel; `np.float16` is IEEE 754 binary16 RNE-conformant since 1.20
- **NumPy uint8 byte arrays for L0/L1/L2/DDR** — one allocation per region, halfword views via `view(np.uint16)` / `view(np.float16)` — zero-copy slicing
- **Python dict for GSPR/NSPR/LSPR** — mirrors C++ `unordered_map<uint16_t, uint64_t>` 1:1; only ~10–20 keys touched per kernel
- **`np.matmul` / `np.einsum` with `dtype=np.float32`** — calls OpenBLAS for GEMM hot path; 10–100× faster than Python loops at the dimensions GTX uses
- **Existing pyspike `riscv.disasm.disasm_insn_t` + `arg` decorators** — direct port of `gtx_npu_disasm.inc` (~140 entries) into a Python list returned from `get_disasms()`
- **stdlib `bytes.fromhex` + `np.frombuffer(buf, dtype='<f2')`** — explicit little-endian for L1/L0; reuse existing `verify.py` parser as black box for DDR hex (which is BE-paired by convention)
- **`importlib.resources.files()` (3.9+) + backport** — bundle `.elf`/`.hex` assets in the `riscv.gtx.data/` subpackage via `[tool.setuptools.package-data]`

**Anti-stack (explicitly rejected by all 4 research files):** NumPy 2.x (drops cp38), scipy (only used for `erf`; reimplement or skip GELU_ERF), ml_dtypes, numba, cython, JAX/torch, capstone, `struct.pack('<e', ...)` for hot paths.

### Expected Features

The feature set is fully determined by the C++ reference — there is no greenfield product discovery. FEATURES.md enumerates ~120 distinct ops/sub-ops grouped into 8 areas: ISA Surface, Memory Hierarchy, Compute (MM/VEC/ACT/DMA), Loop Control, Verification, Python Ergonomics, Distribution, SPR Access.

**Must have (table stakes — 100% of .elf regression depends on these):**
- **MEM-LE-BYTE-ORDER** — every L0/L1 FP16 access strictly little-endian
- **ISA-ROCC-XS1-WORKAROUND** — read register operands via `proc.get_state().XPR[insn.rs1]`
- **ISA-FW-DISPATCH + ISA-ISS-FULL coexistence** — gem5 simplified (funct7=0x04–0x07) AND ISS-full (funct7=0x00–0x7F); collision at 0x00/0x01 disambiguated by `insn.rs1!=0`
- **MM-CORE / MM-EXEC{,_S,_O,_V,_T} + MM-FW + MM-MXE-ACCUM** — NPU core; firmware_mm_op packs dims into rs1; mxe_accum is per-(NEST,SPU) FP32 persistent state
- **DMA-FW + DMA-DEFERRED-STORE** — S-loop L2→DDR stores deferred until `endp`
- **LOOP P/S/T state machine + DISPATCH 4-mode** — Mode 1 (broadcast 64), Mode 2 (NEST 16), Mode 3 (P+S DMA), Mode 4 (P+T compute)
- **ACT direction asymmetry** — RELU/SOFTMAX/ESUM forward; PRELU/GELU/TANH/SIGM reversed (ADDRR→ADDRA)
- **VEC VSUM FP32-internal-accumulate + single FP16 cast**
- **VRF-DDR-DIFF** — port `verify.py` (388 LOC, ULP/atol diff)
- **VRF-ELF-REGRESSION** — `.elf` harness mirroring `run_tests_n1s16.sh` + `run_llext_tests.sh`
- **PY-RESET + PY-WJOIN-EXIT** — sp=0x80100000 init; `SystemExit(0)` on WJOIN unless `GTX_NO_EXIT` set

**Should have (Python differentiators — the v1 value-add over `libgtx_npu.so`):**
- **PY-OVERRIDE-HOOK** (per-op `before_<op>`/`after_<op>` for live numerical experiments)
- **PY-CUSTOM-FUNCT7** (`gtx.register_funct7(0x7E, my_handler)` for ISA experimentation)
- **PY-NUMPY-VIEWS** (`gtx.l1_view(nest, spu, dtype=np.float16)` returns ndarray view)
- **PY-OP-INSTRUMENTATION** (runtime `gtx.enable_trace()` replaces `--enable-gtxcommitlog` build flag)

**Defer (v2+):** Cycle-accurate timing, bank-conflict modeling, mexec full microcode loop, DMA-LOAD-3D / IM2COL / MCAST (regression-driven), PY-SNAPSHOT-RESTORE.

**Anti-features (PROJECT.md Out of Scope):** PCIe-EP/vfio-user, CUDA backend, GTX commitlog, non-Linux/non-x86_64, bundling `libgtx_npu.so` in the wheel.

### Architecture Approach

Ten-file Python package at `src/main/python/riscv/gtx/` mirroring the C++ file split. The class `GtxNpu(riscv.isa.ROCC)` in `npu.py` is a thin trampoline; `custom0/1/2/3` delegate to pure functions in `dispatch.py` that take `(mem: GtxMemory, loop: GtxLoop, proc, insn)` and return cycles. Op modules in `ops/{mm,vec,act,dma,pool,conv,tpose,format,mexec}.py` are pure functions on `GtxMemory` views — directly unit-testable without spike. Dispatch uses dict-of-handlers (`@_handler(0x00)` decorator) rather than `match` (Python 3.10+, PROJECT.md targets 3.8+).

**Major components:**
1. **`GtxNpu` (npu.py)** — `riscv.isa.ROCC` subclass; thin: every method body is one `_dispatch_*` call.
2. **`GtxMemory` (memory.py)** — single `np.uint8` allocation per region (L0: 64KB total, L1: 24MB total, L2: 64MB, DDR: lazy 4GB) with non-copying halfword views; `mxe_accum: np.float32`; gspr/nspr/lspr as Python dicts.
3. **`GtxLoop` (loop.py)** — P/S/T flag tuple + tmu_id/curr_id; `current_context() → 1..4` selects dispatch mode.
4. **Dispatch layer (dispatch.py)** — three levels: funct7→handler dict, 4-mode router, ISS opcode router. Mirrors C++ structure to preserve disambiguation at funct7=0x00/0x01.
5. **Op handlers (ops/*.py)** — uniform signature `exec_<op>(mem, nest_id, spu_id, *args) → cycles`. Read SPRs → get views → FP32 compute → single FP16 cast → update `mxe_accum` if MMC.
6. **Disasm table (disasm.py)** — direct port of `gtx_npu_disasm.inc` via `_add_r` / `_add_rf3` helpers.
7. **Hex I/O (ddr.py / `_verify.py`)** — `ddr_init_from_file` / `ddr_dump_to_file` honoring `GTX_DDR_REVERSED`; `verify.py` repackaged as importable `compare_hex`.
8. **Test architecture** — `tests/gtx/{test_lifecycle,test_memory,test_loop,test_spr,test_op_*,test_regression_fw}.py` with `tests/gtx/data/{golden,elf}/`; oracles drawn from `verify_ref.py`.

### Critical Pitfalls

1. **`verify.py` BE-pair vs L1/L0 LE byte order** — `verify.py:235` reads DDR as BE pair while every C++ memory write uses LE. Treat verify.py as black box; write LE in L0/L1; DMA path is the explicit translation seam. Use `np.frombuffer(buf, dtype='<f2')` — never `arr.view(np.float16)` (host-native).
2. **Per-element FP16 cast in reductions** — `np.add.reduce(arr_f16)` accumulates in FP16. Mandatory: `arr_f16.astype(np.float32).sum(dtype=np.float32)` then single `np.float16(...)` cast. Applies to VSUM, DOT, MM_O, MM_V, SOFTMAX, ESUM, gemm bias chain.
3. **`mxe_accum` continuity across MM chains** — allocate at construction, only `reset()` zeros; honor `is_accumulate` from `funct7==0x01`. Test: `mm.s → mmc.s → mmc` chain.
4. **xs1=0 quirk** — when xs1=0, Spike marshals `xs1 = -1`. Always read register operands as `proc.get_state().XPR[insn.rs1]` directly.
5. **funct7 collision (gem5 vs ISS-full)** — funct7=0x00 is both WRSPR and MM; disambiguated by `insn.rs1 != 0`. Mirror C++ dispatch order; cover both `run_tests_n1s16.sh`-style and `run_llext_tests.sh`-style suites.
6. **Activation direction asymmetry** — REVERSED_ACTIVATIONS = {PRELU, GELU, TANH, SIGMOID} read ADDRR, write ADDRA; others (RELU, SOFTMAX, ESUM) are forward.

Other: WJOIN must `SystemExit(0)` unless `GTX_NO_EXIT` set; `reset()` must `XPR.write(2, 0x80100000)`; pybind11 GIL means stage operands once at top of `custom0`; DDR `GTX_DDR_REVERSED` reverses 32-byte bus words; pin spike commit hash at module init.

---

## Implications for Roadmap

PROJECT.md mandates **MM-first** (GTX-MM-01 is "NPU 핵심"). All 4 research files independently derive the same 6-phase order, and the bit-exact validation gate strictly extends each phase from the prior.

### Phase 1 — Foundation: FP helpers, memory layer, package skeleton
**Rationale:** Everything downstream rounds-trips through FP16↔FP32 conversion and L1/L0 byte writes. Implement `gtx_fp16_to_32`/`gtx_fp32_to_16` as pure-Python bit manipulation (NOT `np.float16(x)`) so subnormal/NaN/-0.0 behavior is identical across NumPy 1.20–1.26 and Python 3.8–3.12.
**Delivers:** `riscv/gtx/{__init__.py,params.py,encoding.py,fp.py,memory.py}`; round-trip test for all 65536 FP16 values; LE byte-order assertion test.
**Maps to requirements:** GTX-CORE-01, GTX-MEM-01, GTX-REF-01, GTX-PKG-01 (skeleton).
**Avoids pitfalls:** 1, 8, 13, 16, 18.

### Phase 2 — Skeleton + SPR + Disasm + Reset/WJOIN
**Rationale:** Spike calls `get_disasms()` once at init; firmware's first instruction (`addi sp,sp,-16`) runs immediately after `reset()`. SPR routing is prerequisite because firmware stages operands via WRSPR before every dispatch.
**Delivers:** `npu.py` (empty `custom0/1/2/3`), `disasm.py` (~140-entry table), `spr.py`, `loop.py`, reset (sp=0x80100000), WJOIN→SystemExit. NOP firmware survives end-to-end.
**Maps to requirements:** GTX-CORE-02, GTX-SPR-01, GTX-DISASM-01, GTX-RST-01.
**Avoids pitfalls:** 4, 5 (scaffold), 6, 7, 11, 17, 19.

### Phase 3 — DMA + DDR I/O (data movement, no compute)
**Rationale:** MM cannot be tested without first being able to load FP16 into L1 and dump out. `GTX_DDR_REVERSED` mode required for HW-sim-derived golden hex.
**Delivers:** `ops/dma.py` (full set including deferred-store flush), `ddr.py` (both byte orders), Mode 3 (P+S) dispatch. DMA round-trip test.
**Maps to requirements:** GTX-DMA-01, GTX-DISP-01 (Mode 3 + Mode 1).
**Avoids pitfalls:** 10, 13, 19.

### Phase 4 — MM Subsystem (the value driver)
**Rationale:** Once MM-FW dispatches correctly through the funct7=0x00 collision and `gemm_core` produces bit-exact FP16 from FP32-internal `np.matmul`, the entire SPR→dispatch→DMA→compute→writeback plumbing is validated. First full firmware regression passes here.
**Delivers:** `ops/mm.py` (MM/MMC variants), `firmware_mm_op` with funct3 decode + packed-rs1 dim decoding (0=65536), Mode 4 dispatch, mxe_accum chain. First .elf regression.
**Maps to requirements:** GTX-MM-01, GTX-DISP-01 (Mode 4).
**Avoids pitfalls:** 2, 3, 5, 12.

### Phase 5 — VEC + ACT + Pool + Conv + Format
**Rationale:** Mechanical ports — but VSUM FP32-precision rule and ACT direction asymmetry are silent-corruption traps. `verify_ref.py`'s 32 host-side scalar oracles serve as test references.
**Delivers:** `ops/{vec,act,pool,conv,tpose,format}.py` complete; per-op pytest with `verify_ref.py` oracle; activation regression .elf passes.
**Maps to requirements:** GTX-VEC-01, GTX-ACT-01, GTX-VERIFY-02.
**Avoids pitfalls:** 2, 9, 18.

### Phase 6 — Verification harness + .elf regression + wheel ship
**Rationale:** Final acceptance. Port `verify.py` as `riscv.gtx._verify`; bundle `.elf` + `.hex` via `[tool.setuptools.package-data]`; integrate `pyspike --extlib=riscv.gtx` CLI. Ship gate: 100% strict-mode pass.
**Delivers:** `riscv/gtx/_verify.py`, parametrized `test_regression_fw.py`, `pyspike-verify` CLI, wheel includes `riscv/gtx/data/{firmware,golden}/*`. CI matrix Python 3.8 × oldest NumPy.
**Maps to requirements:** GTX-VERIFY-01, GTX-FW-01, GTX-PKG-01, GTX-DISASM-01 (final validation).
**Avoids pitfalls:** 14, 15, 16, 20.

### Phase Ordering Rationale

- **Bit-exact dependency chain:** FP helpers → memory views → SPR → DMA → MM → VEC/ACT → regression. Each phase strictly extends prior, so failures localise.
- **Risk-front-loading:** all 6 critical pitfalls have explicit acceptance tests in Phases 1–5; none defer to Phase 6.
- **MM-first per PROJECT.md:** Phase 4 proves the project's "Core Value." Phases 1–3 are infrastructure.
- **Verify-last (Phase 6) NOT verify-throughout:** per-op oracle tests run from Phase 4 onward via `verify_ref.py`; Phase 6 only adds .elf+golden harness and packaging.

### Research Flags

**Phases needing deeper research (`/gsd:research-phase`):**
- **Phase 4 (MM):** firmware_mm_op packed-rs1 encoding (`colB[63:48] | colA[31:16] | rowA[15:0]`, with HW convention 0=65536); mxe_accum tuple shape.
- **Phase 5 (ACT):** direction asymmetry table; format_cvt scale+offset packing; FP8 codec (`gtx_fp8_to_32`) bit patterns.
- **Phase 3 (DMA):** firmware_dma packed encoding; DMA-DEFERRED-STORE queue ordering (snapshot vs ref-based per `plan_has_tloop`).
- **Phase 6 (Wheel):** size budget (split if >50MB); cibuildwheel + numpy/importlib_resources interaction.

**Standard patterns (skip research-phase):**
- **Phase 1 (Foundation):** FP16 bit-manipulation is direct port from C++.
- **Phase 2 (Skeleton):** disasm/SPR/reset patterns mirror existing pyspike `examples/xhuimt`.

### Cross-cutting decisions to record before roadmap kickoff

1. **Internal byte order = little-endian.** `gtx/CLAUDE.md` is authoritative; `gtx_npu.h:770 wr16_be` is the SPU↔CPU shadow seam. DDR hex I/O uses verify.py's BE-pair format unchanged.
2. **`mxe_accum` is class state**, zeroed only by `reset()`, honoring `is_accumulate` from `funct7==0x01`.
3. **Disasm built once at import**; no override of `get_instructions()`.
4. **No CSR exposure for SPRs** — they live in `GtxMemory`, accessed via WRSPR/RDSPR custom0 funct7=0x00/0x01.
5. **No online shadow run** against `libgtx_npu.so` — verification is offline diff against pre-recorded golden hex.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | NumPy version pinning verified vs cibuildwheel matrix; FP16 RNE conformance verified vs `gtx_fp32_to_16`; no hypothetical libraries |
| Features | HIGH | Brownfield port — set fully determined by C++ reference (~120 ops cataloged); no greenfield discovery |
| Architecture | HIGH | C++ file split is the spec; pyspike trampoline surface validated; only open question (BE/LE) resolved above |
| Pitfalls | HIGH | 16/20 pitfalls source-verified; 4 speculative are well-known NumPy/wheel patterns |

**Overall confidence:** HIGH

### Gaps to Address

In-flight verification items, NOT roadmap blockers:

- **`illegal_instruction` exposure on `py_rocc_t`** — verify Phase 2; one-line pyspike binding patch if missing.
- **`XPR.write(idx, value)` mutability** — verify Phase 2; would block sp init if read-only.
- **Disasm `arg` callable acceptance** — verify Phase 2; fallback: pre-format mnemonic strings.
- **mexec full microcode loop** — Phase 5+ deferred; v1 only if a regression trips it.
- **Wheel size budget** — Phase 6 flag: split into `spike[gtx-regression]` extra if >50MB.
- **Strict-mode acceptance gate** — Pitfall 20: `exact_matches == total_fp16` required, not just tolerance pass.

---

## Sources

### Primary (HIGH — direct source read)
- `~/NIGHTLY/gtx_spike/gtx/CLAUDE.md` — memory hierarchy, encoding, byte-order rule, VSUM precision, sp init, WJOIN exit
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu.h` (1382 LOC) — class declaration, FP16↔FP32 helpers (lines 89–151)
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_{dispatch,mm,vec,act,dma,loop}.cc` — op semantics, firmware dispatch, mxe_accum, ACT direction, DDR mode handling
- `~/NIGHTLY/gtx_spike/gtx/gtx_npu_disasm.inc` (244 LOC) — full disasm table (~140 entries)
- `~/NIGHTLY/gtx_spike/gtx/verify.py` (388 LOC) + `verify_ref.py` (378 LOC, 32 oracles)
- `~/NIGHTLY/gtx_spike/gtx/gtx_params.h` — HW constants

### Primary (HIGH — pyspike codebase)
- `/mnt/e/14_NIGHTLY/pyspike/.planning/codebase/{ARCHITECTURE,STACK,STRUCTURE,INTEGRATIONS,CONVENTIONS,TESTING,CONCERNS}.md`
- Commit `c9cf7c4 docs: map RoCC extension surface in pyspike binding layer`
- `/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/isa.py` — `riscv.isa.ROCC`, `@register`
- `/mnt/e/14_NIGHTLY/pyspike/examples/xhuimt/__init__.py` — extension subclass pattern
- `/mnt/e/14_NIGHTLY/pyspike/.planning/PROJECT.md` — Active reqs, Out of Scope, Key Decisions

### Primary (HIGH — external)
- NumPy 2.0 / 2.3 release notes (Python 3.8 drop, manylinux migration)
- NumPy 1.26.4 PyPI wheel matrix (cp38–cp312 manylinux2014)
- NumPy `numpy.finfo` / dtype.byteorder docs
- Python `importlib.resources.files()` (3.9+) + backport
- pytest `pytest_generate_tests` parametrize discovery

### Secondary (MEDIUM)
- pyspike disasm `arg` callable acceptance (verify Phase 2; fallback documented)
- `XPR.write(idx, value)` from Python (verify Phase 2)

### Per-document references
- `.planning/research/STACK.md` — NumPy pinning, FP16 storage/compute, memory, endianness, hex I/O, disasm, perf, testing, packaging
- `.planning/research/FEATURES.md` — ~120 op-level breakdowns + dependency graph + MVP + prioritization
- `.planning/research/ARCHITECTURE.md` — module layout, dispatch, op handler pattern, error handling, build order, test architecture
- `.planning/research/PITFALLS.md` — 20 pitfalls + tech debt + integration gotchas + perf traps + checklist + pitfall-to-phase mapping

---

*Research synthesis completed: 2026-05-04*
*Ready for roadmap: yes*
