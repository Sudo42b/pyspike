---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-05-04T05:40:41Z"
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 5
  completed_plans: 1
---

# State: pyspike + GTX NPU (Python RoCC Port)

**Last updated:** 2026-05-04 after Phase 1 discuss (NumPy 2.x / cp310 / FP16 view pivot)

## Project Reference

**Name:** pyspike + GTX NPU (Python RoCC Port)

**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로
통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된
golden)와 ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.

**Current Focus:** Phase 01 — foundation

**Acceptance Gate:** `pyspike --extlib=riscv.gtx <fw>.elf` → DDR dump that
`verify.py --fp16 --ulp 1 --atol 0.001` reports as **strict-mode pass**
(`exact_matches == total_fp16`) against C++ golden.

## Current Position

Phase: 01 (foundation) — EXECUTING
Plan: 2 of 5

- **Phase:** Phase 1 — Plan 01-skeleton complete; Wave 1 sibling plans 02-fp / 03-memory next
- **Plan:** 01-skeleton committed (d55a82a, 30a50d6, 7284080); SUMMARY at `.planning/phases/01-foundation/01-skeleton-SUMMARY.md`
- **Status:** Executing Phase 01 (1/5 plans complete in this phase)
- **Progress:** [▰▱▱▱▱▱] 0/6 phases complete (1/5 plans within Phase 1)

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| v1 requirement coverage | 42/42 | 42/42 ✓ |
| Phases completed | 6 | 0 |
| .elf regressions passing strict | 100% | 0% |
| Wheel size | ≤50MB | TBD |
| cp310–cp312 cibuildwheel matrix (D-08, cp38/cp39 dropped) | green | TBD — Phase 1 will adjust matrix |

### Plan Execution Log

| Phase | Plan | Duration | Tasks | Files | Commits |
|-------|------|----------|-------|-------|---------|
| 01-foundation | 01-skeleton | 4 min | 3 | 5 | d55a82a, 30a50d6, 7284080 |

## Accumulated Context

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

`/gsd:execute-phase 1 plan 01-skeleton` complete (Wave 1, parallel worktree).
Created `riscv.gtx` package skeleton: `__init__.py` (LE guard + 5 re-exports),
`params.py` (HW topology + memory + SPR addresses, 17 constants), `encoding.py`
(8 funct7 stubs), `ops/__init__.py` (P2-P5 marker), `tests/gtx/__init__.py`
(pytest collection root). 3 atomic commits: d55a82a, 30a50d6, 7284080.
1 auto-fix deviation (Rule 1: docstring paraphrase to satisfy GtxNpu absence
acceptance criterion). SUMMARY at `.planning/phases/01-foundation/01-skeleton-SUMMARY.md`.

### Next Action

Wave 1 sibling plans `02-fp-PLAN.md` and `03-memory-PLAN.md` execute in parallel
worktrees (consume `riscv.gtx.params.GTX_NEST_NUM/SPU/L1` + `from . import fp/memory/ddr`
re-export wiring). After Wave 1 closes (all 3 plans merged), `python -c "import riscv.gtx"`
will work standalone. Then Wave 2 plans `04-packaging-PLAN.md` (NumPy 2.x / cp310 pin)
and `05-submodule-PLAN.md` (vendor/gtx_cpp_reference/ git submodule).

### Resumption Notes

If resuming work in a new session:

1. Read `.planning/PROJECT.md` for core value + constraints
2. Read `.planning/ROADMAP.md` for phase structure + success criteria
3. Read `.planning/REQUIREMENTS.md` for full v1 requirement list with phase mappings
4. Read this STATE.md for current position
5. Per-phase research artifacts live under `.planning/research/` (already populated for the project; per-phase research is generated on-demand by `/gsd:research-phase <N>`)

---

*State initialized: 2026-05-04 after roadmap creation*
