# Roadmap: pyspike + GTX NPU (Python RoCC Port)

**Created:** 2026-05-04
**Granularity:** standard (5–8 phases)
**Parallelization:** enabled
**Coverage:** 42/42 v1 requirements mapped

**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로
통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된
golden)와 ULP 허용오차 내로 일치한다.

**Acceptance Gate (every phase contributes toward this):**
`pyspike --extlib=riscv.gtx <fw>.elf` produces a DDR dump that
`verify.py --fp16 --ulp 1 --atol 0.001` reports as **strict-mode pass**
(`exact_matches == total_fp16`) against the C++ golden hex.

---

## Phases

- [x] **Phase 1: Foundation** — FP16 helpers, NumPy memory layer, package skeleton, NumPy/cp38 packaging baseline (completed 2026-05-04)
- [ ] **Phase 2: Skeleton & Disasm** — `GtxNpu(ROCC)` skeleton, SPR routing, disasm table, reset/WJOIN, custom0/1 dispatch shells
- [ ] **Phase 3: DMA & DDR I/O** — Full DMA op set, deferred-store flush, DDR hex I/O (LTR + reversed), Mode 1/3 dispatch
- [ ] **Phase 4: MM Subsystem** — `gemm_core`, all MM/MMC variants, `firmware_mm_op` packed-rs1, `mxe_accum` chain, Mode 4 dispatch, **first .elf regression passes strict mode**
- [ ] **Phase 5: VEC/ACT/Pool** — Vector ops (SASMD/DOT/VSUM/CLAMP), forward+reversed activations, pooling, format_cvt (incl. FP8 codec), per-op `verify_ref` oracle suite
- [ ] **Phase 6: Verification & Wheel** — `verify.py` ported as `riscv.gtx._verify`, full .elf regression harness (strict mode), `pip install spike` ship gate, cibuildwheel matrix green

---

## Phase Details

### Phase 1: Foundation

**Goal**: Pure-Python FP16↔FP32 helpers, NumPy-backed L0/L1/L2/DDR memory layer, and the `riscv.gtx` package skeleton land in the wheel — ready to host the rest of the port without further packaging churn.

**Depends on**: Nothing (first phase)

**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, PKG-02

**Success Criteria** (what must be TRUE):
  1. `pytest tests/gtx/test_fp_roundtrip.py` passes — all 65536 FP16 values round-trip through `riscv.gtx.fp.fp16_to_fp32` / `fp32_to_fp16` (NumPy 2.x `np.float16` view per D-09). Idempotent: `f16 → f32 → f16 == f16` for every non-NaN value; NaN inputs produce NaN with stable bit pattern. C++ `gtx_npu.h:89-151` strict comparison deferred to P4/P5 (Phase 1 risk — see STATE.md "In-flight Verification Items").
  2. `pytest tests/gtx/test_memory_layout.py` passes — writing `0x3C00` to `mem.l1_byte(0,0)[off]` produces bytes `[0x00, 0x3C]` (LE), and `mem.l1_f16(0,0)[off//2]` reads back as `np.float16(1.0)` with `arr.base is not None` (view, not copy).
  3. `python -c "from riscv.gtx import fp, memory; from riscv.gtx.params import GTX_NEST_NUM"` succeeds in a clean **cp310** venv with `numpy>=2.0,<3` resolved (D-07/D-08; `GtxNpu` re-export is Phase 2 work).
  4. `vendor/gtx_cpp_reference/` is registered as a git submodule (D-04) pointing to `https://github.com/Sudo42b/gtx_spike`, scope = `gtx/` + spike patches (D-05). `git submodule status` shows it as initialized; `MANIFEST.in` excludes it from wheel (D-06).
  5. `pyproject.toml` declares `numpy>=2.0,<3` runtime dep, `requires-python = ">=3.10"` (D-07/D-08), and `[tool.cibuildwheel].build` lists only cp310/cp311/cp312 (cp38/cp39 lines removed). `pip wheel .` produces a valid manylinux2014_x86_64 wheel.

**Plans:** 5/5 plans complete
- [ ] 01-foundation/01-skeleton-PLAN.md — riscv.gtx 패키지 스켈레톤 (`__init__.py`, `params.py`, `encoding.py`, `ops/__init__.py`, `tests/gtx/__init__.py`)
- [ ] 01-foundation/02-fp-PLAN.md — FP16↔FP32 헬퍼 + 65536 round-trip 테스트 (FOUND-01)
- [ ] 01-foundation/03-memory-PLAN.md — GtxMemory(L0/L1/L2 + SPR + DDR lazy) + memory layout 테스트 (FOUND-02)
- [ ] 01-foundation/04-packaging-PLAN.md — pyproject.toml 5-stanza 패치 + wheel 빌드 검증 (PKG-02 + FOUND-03 wheel ship)
- [ ] 01-foundation/05-submodule-PLAN.md — vendor/gtx_cpp_reference git submodule + MANIFEST.in prune (FOUND-04 + D-06)
**UI hint**: no

---

### Phase 2: Skeleton & Disasm

**Goal**: A NOP-only firmware can be loaded under `pyspike --extlib=riscv.gtx`, reach WJOIN, and exit cleanly — with full disasm coverage in the trace, SPR routing wired, and the custom0/custom1 dispatch shells ready to host op handlers.

**Depends on**: Phase 1

**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, SPR-01, SPR-02, DISASM-01, DISP-01, DISP-02

**Success Criteria** (what must be TRUE):
  1. `pyspike --extlib=riscv.gtx tests/gtx/data/elf/nop_wjoin.elf` returns exit code 0 and `addi sp,sp,-16` does NOT trap (sp init = 0x80100000 confirmed by `proc.get_state().XPR[2]` post-reset).
  2. `GtxNpu().get_disasms()` returns a list whose length equals the per-op registry sum: in **Phase 2** that is ~10 entries (SPR/control: `wrspr`, `rdspr`, `wsplit`, `wjoin`, `start_p`, `end_p`, `start_t`, `end_t`, plus a small no-op set). The full ~140 entries matching `gtx_npu_disasm.inc` accumulate progressively as P3 (`dma_load`/`dma_store`/...), P4 (`mm`/`mm_s`/`mm_t`/...), and P5 (VEC/ACT) op modules register. A sample of 5 P2-available instructions (`wrspr`, `rdspr`, `wsplit`, `wjoin`, `start_p`) decode to expected mnemonics in the spike trace; full-set verification deferred to phases that introduce the remaining ops. (Per-op registry decision: see Phase 2 CONTEXT.md D-09.)
  3. `pytest tests/gtx/test_spr.py` passes a sequence `WRSPR(LSPR_SPM_ADDRA, 0xCAFE) → RDSPR(LSPR_SPM_ADDRA) → assert XPR[rd] == 0xCAFE`, exercising both gem5-simplified (funct7=0x00 with `xs1=xs2=1`) and ISS-full (funct7=0x49) encodings, plus the `xs1=0 → -1` workaround via `proc.get_state().XPR[insn.rs1]`.
  4. `pytest tests/gtx/test_warp.py` exercises `start_p → start_t → end_t → end_p` via custom1 funct3 dispatch and the loop state machine ends in `(is_ploop=False, is_tloop=False)` with no leak across calls.
  5. `GTX_NO_EXIT` unset → WJOIN raises `SystemExit(0)`; `GTX_NO_EXIT=1` → WJOIN returns 0 and firmware loop continues (asserted by direct `custom1` invocation in unit test).

**Plans:** 5 plans (5/5 complete) -- gap-closure 02-06 attempted, build path validated, post-build regressions deferred
- [x] 02-skeleton-disasm/02-01-PLAN.md — Wave 0 scaffold (package skeleton + test infra + nop_wjoin.elf fixture) — completed 2026-05-04 (`2170e6d` `cd7c042` `01e9737`)
- [x] 02-skeleton-disasm/02-02-PLAN.md — SPR routing + WRSPR/RDSPR handlers (Wave 1) — completed 2026-05-04 (`9391242` `7eaa054` `849e840`)
- [x] 02-skeleton-disasm/02-03-PLAN.md — Loop state machine + custom1 + WJOIN (Wave 1) — completed 2026-05-04 (`1cb2cba` `ad41713` `ef9a659`)
- [x] 02-skeleton-disasm/02-04-PLAN.md — Disasm registration (Wave 1) — completed 2026-05-04 (`e6c28bb` `3babd10` `7d4e76f`)
- [x] 02-skeleton-disasm/02-05-PLAN.md — Skeleton tests + integration (Wave 2)
**UI hint**: no

---

### Phase 3: DMA & DDR I/O

**Goal**: Bytes can flow DDR ↔ L2 ↔ L1 ↔ L0 with bit-exact preservation in both `GTX_DDR_REVERSED` modes, with deferred-store semantics matching C++ — enabling all subsequent compute phases to load operands and dump results without a separate fix.

**Depends on**: Phase 2

**Note for plan-phase**: Flagged for `/gsd:research-phase` before `/gsd:plan-phase` — `firmware_dma_op` packed encoding (`addr_hi[63:27] | addr_lo[27:0]` in rs1; `height[63:48] | length[47:32] | stride[31:0]` in rs2; HW convention 0=65536 for length, 0=1 for height) and DMA-DEFERRED-STORE queue ordering (snapshot-vs-ref per `plan_has_tloop`) are non-obvious encodings that warrant deep research before planning.

**Requirements**: DMA-01, DMA-02, DMA-03, DMA-04, DMA-05, DISP-03

**Success Criteria** (what must be TRUE):
  1. `pytest tests/gtx/test_dma_roundtrip.py` passes — write a known FP16 pattern at `mem.l1_f16(0,0)[0:4096]`, dispatch `exec_store_svr` chain to push it to L2, then `exec_dma_2d` (S-loop) to push to DDR, dump, reload via `ddr_init_from_file`, run reverse path, and assert byte-exact match against the original pattern.
  2. `pytest tests/gtx/test_ddr_modes.py` passes — same pattern dumped via `ddr_dump_to_file` produces different hex bytes under default LTR vs `GTX_DDR_REVERSED=1` (32-byte bus-word reversal verified) AND each mode round-trips through its own init.
  3. `firmware_dma_op` decoded for a synthetic instruction with funct3=000 (LOAD), HW convention `length=0` (decoded as 65536) and `height=0` (decoded as 1) produces the same source/destination address pair as `gtx_npu_dma.cc:firmware_dma`.
  4. S-loop deferred-store: a sequence `start_p → start_s → exec_dma_2d(STORE) → end_s → exec_dma_2d(STORE) → end_p` flushes both stores in order at `end_p`, and a DDR dump taken before `end_p` shows the pre-store contents (deferred queue verified by direct inspection of `mem.deferred_ddr_stores`).
  5. Mode 1 (no loop, broadcast 64) and Mode 3 (P+S, single NEST DMA) routing in `_dispatch` selects the same `(nest_id, spu_id)` set as `gtx_npu_dispatch.cc` for synthesized firmware traces (parametrized test over a fixture of (loop_state, opcode) tuples).

**Plans:** 5 plans
- [ ] 02-skeleton-disasm/02-01-PLAN.md — Wave 0 scaffold (package skeleton + test infra + nop_wjoin.elf fixture)
- [ ] 02-skeleton-disasm/02-02-PLAN.md — SPR routing + WRSPR/RDSPR handlers (Wave 1)
- [ ] 02-skeleton-disasm/02-03-PLAN.md — Loop state machine + custom1 + WJOIN (Wave 1)
- [ ] 02-skeleton-disasm/02-04-PLAN.md — Disasm registration (Wave 1)
- [x] 02-skeleton-disasm/02-05-PLAN.md — Skeleton tests + integration (Wave 2)
**UI hint**: no

---

### Phase 4: MM Subsystem

**Goal**: `gemm_core` produces FP16 results bit-exact with C++ `libgtx_npu.so` for every MM/MMC variant, the `firmware_mm_op` dispatch path correctly disambiguates the funct7=0x00 collision with WRSPR, `mxe_accum` chains across `mm.s → mmc.s → mmc` reproduce C++ behavior, and the **first full .elf GEMM regression passes strict mode** — proving the entire SPR→dispatch→DMA→compute→writeback plumbing is correct.

**Depends on**: Phase 3 (operand loading), Phase 2 (dispatch + SPR), Phase 1 (FP helpers + memory)

**Note for plan-phase**: Flagged for `/gsd:research-phase` before `/gsd:plan-phase` — `firmware_mm_op` packed-rs1 encoding (`colB[63:48] | colA[31:16] | rowA[15:0]` with HW convention 0=65536 in each 16-bit field), `mxe_accum` per-(NEST,SPU) FP32 state shape, and the `funct7=0x00` `insn.rs1!=0` heuristic for WRSPR-vs-MM disambiguation need deep verification against C++ source before planning.

**Requirements**: MM-01, MM-02, MM-03, MM-04, MM-05

**Success Criteria** (what must be TRUE):
  1. `pytest tests/gtx/test_op_mm.py` passes — for every MM variant (`mm`, `mm_s`, `mm_o`, `mm_v`, `mm_t`, `mmc`, `mmc_s`, `mmc_o`, `mmc_v`, `mmc_t`), a 16×16×16 GEMM with random FP16 operands produces output bit-exact (`assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))`) to a NumPy FP32-internal `np.matmul` + single `np.float16` cast oracle.
  2. `pytest tests/gtx/test_mm_chain.py` passes — `mm.s → mmc.s → mmc` chain on the same (nest, spu) accumulates `mxe_accum` correctly: final FP16 result equals `np.float16(A1@B1 + A2@B2 + A3@B3)` computed in FP32, while a single fused `mm` on the concatenated inputs gives a different (and known) value.
  3. `firmware_mm_op` with `insn.rs1=0` routes to WRSPR (gem5 path); with `insn.rs1!=0` routes to MM and decodes `rs1=0` 16-bit field as `65536` per HW convention. Coverage matrix `(funct7, funct3, has_rs1) → handler` parametrized test passes for funct7∈{0x00, 0x01}.
  4. **First .elf GEMM regression passes strict mode**: `pyspike --extlib=riscv.gtx tests/gtx/data/elf/mm_basic.elf` with `GTX_DDR_DUMP` set produces a hex file that `verify.py --fp16 --ulp 1 --atol 0.001 --strict` (where strict requires `exact_matches == total_fp16`) reports as PASS against `tests/gtx/data/golden/mm_basic_n1s16.hex`.
  5. Mode 4 (P+T) dispatch routes a synthesized `firmware_mm_op` to exactly the `(tmu_id, curr_id)` SPU when `is_ploop=True, is_tloop=True`; `mxe_accum[other_nest][other_spu]` remains untouched (verified by snapshot diff).

**Plans:** 5 plans
- [ ] 02-skeleton-disasm/02-01-PLAN.md — Wave 0 scaffold (package skeleton + test infra + nop_wjoin.elf fixture)
- [ ] 02-skeleton-disasm/02-02-PLAN.md — SPR routing + WRSPR/RDSPR handlers (Wave 1)
- [ ] 02-skeleton-disasm/02-03-PLAN.md — Loop state machine + custom1 + WJOIN (Wave 1)
- [ ] 02-skeleton-disasm/02-04-PLAN.md — Disasm registration (Wave 1)
- [x] 02-skeleton-disasm/02-05-PLAN.md — Skeleton tests + integration (Wave 2)
**UI hint**: no

---

### Phase 5: VEC/ACT/Pool

**Goal**: Every VEC/ACT/pool/format-cvt op produces FP16 output bit-exact with C++ for op-level inputs, the activation-direction asymmetry (RELU/SOFTMAX/ESUM forward; PRELU/GELU/TANH/SIGM reversed) is honored, VSUM/DOT honor the FP32-internal-accumulate-with-single-FP16-cast precision rule, and the `verify_ref.py` 32-op oracle suite passes as pytest unit tests.

**Depends on**: Phase 4 (memory + dispatch + FP discipline proven by MM)

**Note for plan-phase**: Flagged for `/gsd:research-phase` before `/gsd:plan-phase` — activation-direction asymmetry table, `format_cvt` scale+offset packing (`GSPR_GTX_OPERAND2 = [offset:16 | scale:16]`), and FP8 codec (`gtx_fp8_to_32` bit patterns from C++ reference, since FP8 has a custom encoding not standard E4M3/E5M2) are non-obvious and need deep research before planning.

**Requirements**: VEC-01, VEC-02, VEC-03, VEC-04, VEC-05, ACT-01, ACT-02, ACT-03, ACT-04, ACT-05, VRF-02

**Success Criteria** (what must be TRUE):
  1. `pytest tests/gtx/test_op_vec.py` passes — for SASMD (add/sub/mul/div, IS+VS variants), DOT, VSUM, CLAMP (min/max/arange/accum), and L0/L1 paths, bit-exact match against the matching `verify_ref.py` oracle. Specifically: `np.float16([1.0, 1e-4]*1000).sum()` via VSUM produces the FP32-internal-accumulate result (≈0.1) not the FP16-truncated result (inf), enforced by an explicit anti-pattern test.
  2. `pytest tests/gtx/test_op_act.py` passes — every activation runs with **distinct** ADDRA and ADDRR pre-loaded with different known patterns, and the assertion confirms which buffer was overwritten matches the direction table: forward activations (RELU/SOFTMAX/ESUM) overwrite ADDRR, reversed activations (PRELU/GELU/TANH/SIGM) overwrite ADDRA.
  3. `pytest tests/gtx/test_op_format.py` passes — `format_cvt` with FP16↔FP32, FP16↔FP8, FP16↔INT8, FP16↔INT32 round-trips with scale+offset packed in `GSPR_GTX_OPERAND2`, bit-exact against C++ reference for representative values including subnormals.
  4. `pytest tests/gtx/test_pooling.py` passes — `exec_pooling` (max + avg) produces output of length `length/kernel_size` bit-exact with NumPy oracle (avg-pool canonicalizes signed zero to +0.0).
  5. **Activation regression .elf passes strict mode**: `pyspike --extlib=riscv.gtx tests/gtx/data/elf/activation_relu_gelu.elf` with `GTX_DDR_DUMP` produces a hex that `verify.py --fp16 --ulp 1 --atol 0.001 --strict` reports as PASS against the C++ golden.

**Plans:** 5 plans
- [ ] 02-skeleton-disasm/02-01-PLAN.md — Wave 0 scaffold (package skeleton + test infra + nop_wjoin.elf fixture)
- [ ] 02-skeleton-disasm/02-02-PLAN.md — SPR routing + WRSPR/RDSPR handlers (Wave 1)
- [ ] 02-skeleton-disasm/02-03-PLAN.md — Loop state machine + custom1 + WJOIN (Wave 1)
- [ ] 02-skeleton-disasm/02-04-PLAN.md — Disasm registration (Wave 1)
- [x] 02-skeleton-disasm/02-05-PLAN.md — Skeleton tests + integration (Wave 2)
**UI hint**: no

---

### Phase 6: Verification & Wheel

**Goal**: `verify.py` is shipped as `riscv.gtx._verify` (importable + CLI), the full .elf regression suite (gem5-simplified + ISS-full encoding sweeps) is bundled in the wheel and passes 100% strict-mode, and `pip install spike` in a clean cp38–cp312 manylinux2014 environment delivers a working `from riscv.gtx import GtxNpu` one-liner. **This is the project's ship gate.**

**Depends on**: Phase 5 (all ops working), Phase 4 (MM regression already passing), Phase 1 (packaging baseline)

**Requirements**: VRF-01, VRF-03, VRF-04, PKG-01, PKG-03, PKG-04

**Success Criteria** (what must be TRUE):
  1. `python -m riscv.gtx._verify result.hex golden.hex --fp16 --ulp 1 --atol 0.001 --strict` (and `from riscv.gtx._verify import compare_hex` from a notebook) both work — verify.py logic is fully ported and importable, with `--strict` mode requiring `exact_matches == total_fp16`.
  2. **Full .elf regression passes 100% strict mode**: `pytest tests/gtx/test_regression_fw.py` with parametrize over every bundled .elf (covering both `run_tests_n1s16.sh`-style gem5-simplified encoding and `run_llext_tests.sh`-style ISS-full encoding) reports zero failures and zero `within_tolerance` matches (every byte exact).
  3. `pip install dist/spike-*.whl` in a fresh **cp310** venv (no developer tooling) succeeds, and `python -c "from riscv.gtx import GtxNpu; from riscv.gtx import _verify; import importlib.resources as r; assert any(p.name.endswith('.elf') for p in r.files('riscv.gtx').joinpath('data','firmware').iterdir())"` confirms wheel includes `.elf` + `.hex` assets via `[tool.setuptools.package-data]`.
  4. `cibuildwheel` matrix builds green for **cp310–cp312** manylinux2014_x86_64 (Phase 1 D-08; cp38/cp39 dropped) with no regressions vs the pre-GTX baseline; wheel size is ≤50MB (or split into `spike[gtx-regression]` extra if exceeded).
  5. `pyspike --extlib=riscv.gtx tests/gtx/data/elf/<any>.elf` from a `pip install`-ed wheel produces a DDR dump that the bundled `pyspike-verify` console script accepts as strict-mode PASS — proving the user's "한 줄 실행" path works end-to-end.

**Plans:** 5 plans
- [ ] 02-skeleton-disasm/02-01-PLAN.md — Wave 0 scaffold (package skeleton + test infra + nop_wjoin.elf fixture)
- [ ] 02-skeleton-disasm/02-02-PLAN.md — SPR routing + WRSPR/RDSPR handlers (Wave 1)
- [ ] 02-skeleton-disasm/02-03-PLAN.md — Loop state machine + custom1 + WJOIN (Wave 1)
- [ ] 02-skeleton-disasm/02-04-PLAN.md — Disasm registration (Wave 1)
- [x] 02-skeleton-disasm/02-05-PLAN.md — Skeleton tests + integration (Wave 2)
**UI hint**: no

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 5/5 | Complete   | 2026-05-04 |
| 2. Skeleton & Disasm | 0/5 | Not started | - |
| 3. DMA & DDR I/O | 0/? | Not started | - |
| 4. MM Subsystem | 0/? | Not started | - |
| 5. VEC/ACT/Pool | 0/? | Not started | - |
| 6. Verification & Wheel | 0/? | Not started | - |

---

## Phase Ordering Rationale

- **Bit-exact dependency chain:** FP helpers (P1) → memory views (P1) → SPR + dispatch (P2) → DMA (P3) → MM (P4) → VEC/ACT (P5) → regression + wheel (P6). Each phase strictly extends the prior so failures localise to the most recent phase.
- **MM-first per PROJECT.md:** P4 proves the project's "Core Value" (first .elf regression). P1–P3 are infrastructure that exist solely to unblock P4.
- **Risk-front-loading:** All 6 critical pitfalls (LE byte order, FP32 accumulate rule, mxe_accum continuity, xs1=0 quirk, funct7 collision, ACT direction asymmetry) have explicit acceptance criteria in P1–P5; none defer to P6.
- **Verify-throughout, not verify-last:** Per-op `verify_ref.py` oracles are integrated from P4 onward (MM) and P5 (VEC/ACT). P6 only adds the full `.elf` harness and packaging — not new verification logic.

---

## Coverage Summary

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1. Foundation | FOUND-01, FOUND-02, FOUND-03, FOUND-04, PKG-02 | 5 |
| 2. Skeleton & Disasm | CORE-01, CORE-02, CORE-03, CORE-04, SPR-01, SPR-02, DISASM-01, DISP-01, DISP-02 | 9 |
| 3. DMA & DDR I/O | DMA-01, DMA-02, DMA-03, DMA-04, DMA-05, DISP-03 | 6 |
| 4. MM Subsystem | MM-01, MM-02, MM-03, MM-04, MM-05 | 5 |
| 5. VEC/ACT/Pool | VEC-01, VEC-02, VEC-03, VEC-04, VEC-05, ACT-01, ACT-02, ACT-03, ACT-04, ACT-05, VRF-02 | 11 |
| 6. Verification & Wheel | VRF-01, VRF-03, VRF-04, PKG-01, PKG-03, PKG-04 | 6 |
| **Total** | | **42 / 42** |

---

*Roadmap created: 2026-05-04*
*Last updated: 2026-05-04 after Phase 1 discuss (NumPy 2.x / cp310 / FP16 view pivot — D-07/D-08/D-09)*
*Granularity: standard*
*Coverage: 100%*
