---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-05-04T09:00:49.511Z"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 10
  completed_plans: 8
  percent: 70
---

# State: pyspike + GTX NPU (Python RoCC Port)

**Last updated:** 2026-05-04 after Phase 2 plan 01 (skeleton-disasm — GtxNpu shell + per-op registry + nop_wjoin.elf fixture; CORE-01 + CORE-02 complete)

## Project Reference

**Name:** pyspike + GTX NPU (Python RoCC Port)

**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로
통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된
golden)와 ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.

**Current Focus:** Phase 02 — skeleton-disasm

**Acceptance Gate:** `pyspike --extlib=riscv.gtx <fw>.elf` → DDR dump that
`verify.py --fp16 --ulp 1 --atol 0.001` reports as **strict-mode pass**
(`exact_matches == total_fp16`) against C++ golden.

## Current Position

Phase: 02 (skeleton-disasm) — EXECUTING
Plan: 2 of 5 (1 complete)

- **Phase:** 2
- **Plan:** 02-01 complete; 02-02 next (Wave 1 — runs in parallel with 02-03 + 02-04)
- **Status:** Executing Phase 02
- **Progress:** [███████░░░] 70%

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| v1 requirement coverage | 42/42 | 42/42 ✓ |
| Phases completed | 6 | 0 |
| .elf regressions passing strict | 100% | 0% |
| Wheel size | ≤50MB | TBD |
| cp310–cp312 cibuildwheel matrix (D-08, cp38/cp39 dropped) | green | TBD — Phase 1 will adjust matrix |
| Phase 01-foundation P04-packaging | 36m22s | 2 tasks | 1 files |
| Phase 02-skeleton-disasm P01-skeleton | 7m44s | 3 tasks | 16 files |
| Phase 02-skeleton-disasm P02 | 4m52s | 3 tasks | 3 files |
| Phase 02 P03 | 5m30s | 3 tasks | 3 files |

## Accumulated Context

### Phase 2 Plan 01 Decisions (locked during execution)

1. **`mxe_accum` is 2D `(GTX_NEST_NUM, GTX_SPU_NUM)` float32** verbatim per `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1254`. Supersedes CONTEXT.md D-06 which incorrectly stated 4D `(NEST, SPU, M_TILE, N_TILE)`. P4 MM op should reference this not D-06.
2. **`riscv.gtx.GtxNpu = None` when `_riscv.so` is absent** — package import wraps `from . import npu` in try/except so Phase 1 tests still pass without rebuilding the C extension. Plans 02-05 unit tests use mocks; plan 05 integration test gates on `_RISCV_AVAILABLE`.
3. **Makefile `CC = …` (not `?=`)** — Make sets `CC=cc` implicitly so `?=` is a no-op. Use explicit assignment; callers can still override via `make CC=/path/to/gcc`.

### Key Decisions (from PROJECT.md, locked at planning)

1. **Pure Python + NumPy backend.** No new C++ code; no numba/cython/JAX/torch in v1.
2. **Bit-exact target = C++ libgtx_npu.so DDR output** (offline diff). No online shadow run.
3. **MM-first.** P4 is the project's value driver; P1–P3 exist to unblock P4.
4. **PCIe-EP / vfio-user / CUDA / GTX commitlog excluded from v1.** Wheel-distribution simplicity > those features for v1.
5. **C++ gtx sources kept in `vendor/gtx_cpp_reference/`** as ground-truth snapshot, NOT bundled in the wheel.
6. **NumPy version pin: `numpy>=2.0,<3`** + `requires-python = ">=3.10"` (Phase 1 D-07/D-08; cp38/cp39 dropped from cibuildwheel matrix). Reverses earlier research recommendation.
7. **FP16 conversion = `np.float16` view** (Phase 1 D-09; not pure-Python bit manipulation). NumPy 2.x IEEE 754 binary16 RNE assumed. Risk of edge-case divergence vs C++ `gtx_fp32_to_16` to be measured in P4/P5 strict mode.
8. **C++ ground-truth = git submodule at `https://github.com/Sudo42b/gtx_spike`** mounted at `vendor/gtx_cpp_reference/` (Phase 1 D-04/D-05/D-06; not bundled in wheel).
9. **DDR allocation = lazy `ensure_ddr` + `GTX_DDR_SIZE` env var** (Phase 1 D-01/D-02; default 4GB). DDR_REVERSED handled at I/O boundary only (D-03).
10. **Memory API = layered (raw view + named accessor) + single-dict SPR + non-copying view guarantee** (Phase 1 D-10/D-11/D-12).

### Architecture Conventions (from research/ARCHITECTURE.md)

- Package lands at `src/main/python/riscv/gtx/` (NOT `examples/`) — it ships in the wheel as a v1 product feature.
- Module split mirrors C++ file split: `npu.py`, `memory.py`, `dispatch.py`, `loop.py`, `spr.py`, `disasm.py`, `fp.py`, `ddr.py`, `_verify.py`, `ops/{mm,vec,act,dma,pool,conv,tpose,format,mexec}.py`.
- Op handlers are pure functions on `GtxMemory` — directly unit-testable without spike.
- Dispatch uses `dict[funct7, handler]` with `@_handler(0x00)` decorator (NOT `match`, since cp38 target).
- L0/L1/L2/DDR are `np.uint8` ndarray with halfword views (`view(np.uint16)` / `view(np.float16)`); GSPR/NSPR/LSPR are Python `dict[int,int]`.
- FP discipline: load FP16 → upcast to FP32 → compute → single FP16 cast at write-back. Never accumulate in FP16.
- Internal byte order = little-endian (matches `gtx/CLAUDE.md` and CPU-shadow view); DDR hex I/O uses verify.py's BE-pair format.
- `mxe_accum` is class state, zeroed only by `reset()`, honoring `is_accumulate` from `funct7==0x01`.
- No CSR exposure for SPRs — they live in `GtxMemory`, accessed via WRSPR/RDSPR custom0 funct7=0x00/0x01.

### Critical Pitfalls Surfaced (from research/PITFALLS.md)

Each phase's success criteria explicitly defends against the following:

1. **verify.py BE-pair vs L1/L0 LE byte order** → P1 success criterion 2 (LE byte assertion).
2. **Per-element FP16 cast in reductions** → P4 (`np.matmul` FP32) and P5 (VSUM/DOT/SOFTMAX/ESUM FP32-accumulate) criteria.
3. **`mxe_accum` continuity across MM chains** → P4 success criterion 2 (mm.s→mmc.s→mmc chain test).
4. **xs1=0 quirk** (Spike marshals -1) → P2 success criterion 3 (`proc.get_state().XPR[insn.rs1]` direct read).
5. **funct7=0x00 collision (gem5 WRSPR vs ISS MM)** → P4 success criterion 3 (`insn.rs1!=0` heuristic).
6. **Activation direction asymmetry** → P5 success criterion 2 (distinct ADDRA/ADDRR test).

Other tracked: WJOIN `SystemExit(0)` (P2 #5), reset sp=0x80100000 (P2 #1), DDR_REVERSED both modes (P3 #2), wheel size budget (P6 #4), strict-mode acceptance gate (P4/P5/P6).

### In-flight Verification Items (from research/SUMMARY.md "Gaps" + Phase 1 D-09 risk)

NOT roadmap blockers — to be confirmed during phase execution:

- **`illegal_instruction` exposure on `py_rocc_t`** — verify in P2; one-line pyspike binding patch if missing.
- **`XPR.write(idx, value)` mutability from Python** — verify in P2; would block sp init if read-only.
- **Disasm `arg` callable acceptance** — verify in P2; fallback: pre-format mnemonic strings.
- **mexec full microcode loop** — defer to P5+ only if a regression trips it.
- **Wheel size budget** — P6 flag: split into `spike[gtx-regression]` extra if >50MB.
- **★ NumPy 2.x `np.float16` vs C++ `gtx_fp32_to_16` edge cases (D-09)** — measure in P4/P5 strict mode. Differences in subnormal handling, NaN payload preservation, halfway-rounding (RNE half-to-even) may surface only at .elf regression. Fallback: port C++ bit operations as `gtx/fp_strict.py` if strict gate fails.

### Todos / Open Items

- [ ] Pre-P1: `git submodule add https://github.com/Sudo42b/gtx_spike vendor/gtx_cpp_reference` (FOUND-04, D-04). Scope: gtx/ + spike patches (D-05). Wheel exclude (D-06).
- [ ] P3: `/gsd:research-phase 3` before `/gsd:plan-phase 3` (firmware_dma_op packing + deferred-store ordering)
- [ ] P4: `/gsd:research-phase 4` before `/gsd:plan-phase 4` (firmware_mm_op packed-rs1 + mxe_accum tuple shape + funct7=0x00 disambiguation)
- [ ] P5: `/gsd:research-phase 5` before `/gsd:plan-phase 5` (ACT direction table + format_cvt scale/offset + FP8 codec)
- [ ] P6: decide wheel split strategy if `du -sh dist/*.whl > 50MB`

### Blockers

None. All 4 research streams converge on a HIGH-confidence approach; coverage is 100%.

## Session Continuity

### Last Action

Phase 2 plan 01 (skeleton-disasm Wave 0 scaffold) executed solo on main. Three atomic commits:
`2170e6d` (test, mock infrastructure + D-18 hybrid fallback), `cd7c042` (feat, riscv.gtx package
skeleton — GtxNpu shell + per-op registry + dispatch builders + WarpState + ops stubs), and
`01e9737` (chore, nop_wjoin.elf test fixture — assembly + Makefile + prebuilt 5KB binary).
SUMMARY at `.planning/phases/02-skeleton-disasm/02-01-SUMMARY.md`. CORE-01 + CORE-02 marked
complete in REQUIREMENTS.md. Two auto-fix deviations during T3 (Rule 3 blocking issues): Makefile
`CC ?=` → `CC =` to override Make's implicit cc default; local `tests/gtx/data/elf/.gitignore`
negation rules to override project-level `*.elf` and `Makefile` patterns. CRITICAL: `mxe_accum`
locked as 2D `(GTX_NEST_NUM, GTX_SPU_NUM)` float32 per `gtx_npu.h:1254` — supersedes CONTEXT.md
D-06 which stated 4D.

### Next Action

Wave 1 of Phase 2 ready: plans 02-02 (SPR), 02-03 (warp/control), 02-04 (disasm) can land in
parallel. Each plan uses `from ..gtx._registry import handler` + `tests.gtx._mocks.MockProcessor`
provided by Wave 0. After Wave 1, plan 02-05 (integration) closes Phase 2.

### Resumption Notes

If resuming work in a new session:

1. Read `.planning/PROJECT.md` for core value + constraints
2. Read `.planning/ROADMAP.md` for phase structure + success criteria
3. Read `.planning/REQUIREMENTS.md` for full v1 requirement list with phase mappings
4. Read this STATE.md for current position
5. Per-phase research artifacts live under `.planning/research/` (already populated for the project; per-phase research is generated on-demand by `/gsd:research-phase <N>`)

---

*State initialized: 2026-05-04 after roadmap creation*
