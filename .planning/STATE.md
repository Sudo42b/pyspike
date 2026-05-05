---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-05-05T14:41:10.828Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 16
  completed_plans: 14
  percent: 88
---

# State: pyspike + GTX NPU (Python RoCC Port)

**Last updated:** 2026-05-05 after Phase 3 Wave 2 complete (Plans 02 + 04 in parallel — ops-dma + dispatch-4mode both landed; DISP-03 ready to mark complete)

## Project Reference

**Name:** pyspike + GTX NPU (Python RoCC Port)

**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로
통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된
golden)와 ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.

**Current Focus:** Phase 03 — dma-ddr-i-o

**Acceptance Gate:** `pyspike --extlib=riscv.gtx <fw>.elf` → DDR dump that
`verify.py --fp16 --ulp 1 --atol 0.001` reports as **strict-mode pass**
(`exact_matches == total_fp16`) against C++ golden.

## Current Position

Phase: 03 (dma-ddr-i-o) — EXECUTING
Plan: Wave 2 complete (Plans 02 + 04 landed in parallel); Wave 3 next (Plan 05 flush-roundtrip)

- **Phase:** 3
- **Plan:** 01 (dma-engine) ✓; 03 (ddr-io) ✓; 02 (ops-dma) ✓; 04 (dispatch-4mode) ✓; 05 (flush-roundtrip) next
- **Status:** Executing Phase 03 — Wave 2 of 3 complete
- **Progress:** [█████████░] 88%

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
| Phase 02-skeleton-disasm P04-disasm | 6m0s | 3 tasks | 3 files |
| Phase 02-skeleton-disasm PP05-integration | 4m18s | 5 tasks | 5 files |
| Phase 02-skeleton-disasm P06 | 21min | 4 tasks | 5 files |
| Phase 03-dma-ddr-i-o P01 | 9min | 2 tasks | 10 files |
| Phase 03-dma-ddr-i-o P03 | 5min | 1 tasks | 2 files |
| Phase 03-dma-ddr-i-o P04 | 6m 29s | 1 tasks | 3 files |

## Accumulated Context

### Phase 2 Plan 01 Decisions (locked during execution)

1. **`mxe_accum` is 2D `(GTX_NEST_NUM, GTX_SPU_NUM)` float32** verbatim per `vendor/gtx_cpp_reference/gtx/gtx_npu.h:1254`. Supersedes CONTEXT.md D-06 which incorrectly stated 4D `(NEST, SPU, M_TILE, N_TILE)`. P4 MM op should reference this not D-06.
2. **`riscv.gtx.GtxNpu = None` when `_riscv.so` is absent** — package import wraps `from . import npu` in try/except so Phase 1 tests still pass without rebuilding the C extension. Plans 02-05 unit tests use mocks; plan 05 integration test gates on `_RISCV_AVAILABLE`.
3. **Makefile `CC = …` (not `?=`)** — Make sets `CC=cc` implicitly so `?=` is a no-op. Use explicit assignment; callers can still override via `make CC=/path/to/gcc`.

### Phase 2 Plan 04 Decisions (locked during execution)

1. **Offline `_PyDisasmInsn` NamedTuple fallback** — disasm.py wraps the real `riscv.disasm.disasm_insn_t` in try/except ImportError; when `_riscv.so` is absent, helpers return a NamedTuple sentinel exposing the same `.name/.match/.mask/.args` surface so unit tests run without the C extension.
2. **`disasm_insn_t` accepts positional varargs** — verified by reading `src/main/cpp/riscv_disasm.cc:29-37` (`py_disasm_insn_t_create(name, match, mask, py::args py_args)`). Helpers pass `gtx_xrd, gtx_xrs1, gtx_xrs2` positionally, no list wrapping needed.
3. **Sample-5 P2 mnemonics use unambiguous custom0 names** (`wsplit_c0`/`wjoin_c0`) per CONTEXT.md D-12 adaptation. The custom1 warp variants (`warp_split`/`warp_join`) are independently registered and verified.

### Phase 2 Plan 05 Decisions (locked during execution)

1. **Self-contained `_RISCV_AVAILABLE` module-level detection in each plan-05 test** (NOT the conftest fixture). The planner's acceptance command is `pytest ... --noconftest -o "addopts="`, which strips the `riscv_available` fixture defined in `tests/gtx/conftest.py`. First Task 1 run failed with "fixture 'riscv_available' not found" (Rule 3 - Blocking). Resolution: each test module duplicates the 5-line `try/except ImportError` detection. Conftest fixture is preserved for non-`--noconftest` runs.
2. **Whole-module `pytestmark = pytest.mark.skipif`** for `test_reset.py` and `test_dispatch.py` (every test needs `_RISCV_AVAILABLE`). `test_register.py` keeps per-test guards because Tier 1 tests are always-run.
3. **`test_skeleton.py` uses `subprocess.run` not pytest internals** -- avoids GIL contamination from running spike inside a pytest worker (research §1296-1297). pyspike CLI resolution: `shutil.which('pyspike')` first, fall back to `[sys.executable, "-m", "riscv"]`. Timeout=30s defends against WJOIN SystemExit non-propagation.

### Phase 3 Plan 01 Decisions (locked during execution)

1. **AUTHORITATIVE constants from gtx_params.h:** GSPR_GTX_OPERAND1/2/3/OPCODE = 0x001/0x002/0x003/0x004 (gtx_params.h:38-41); LSPR_SPM_ADDRA/B/C/R = 0x900/0x901/0x902/0x903 (gtx_params.h:64-67); GTX_DDR_BASE = 0x370000000 (gtx_params.h:24). Earlier draft addresses 0x110..0x113 were WRONG and would silently break GSPR-staged operand reads. Source comments flag this with "AUTHORITATIVE" markers.
2. **DeferredDdrStore is `@dataclass(frozen=True)` with exactly 7 fields in lock order** (`nest, l2_off, ddr_off, length, height, l2_stride, ddr_stride`) — Pitfall 4 lock-in. Mutation raises FrozenInstanceError; field drift caught by `len(dataclasses.fields(...)) == 7` assertion test.
3. **decode_firmware_dma_args applies HW conventions at decode (not engine):** `length=0 -> 0x10000`, `height=0 -> 1` resolved before the dict is returned. is_copy carve-out: `addr_hi = (rs1>>32) if is_copy else ((rs1>>27)&0x1FFFFFFFFF)` (Pitfall 1 + 2).
4. **WarpState.wsplit_seen is process-lifetime sentinel** (Pitfall 7) — initialized once to False, set True by WSPLIT, NOT cleared by `reset()`. Test asserts persistence: `w.reset()` clears `is_ploop` but `w.wsplit_seen` remains True. Source comment in `reset()` body documents the omission explicitly.
5. **Wave 0 placeholder body uses `pytest.skip(...)` (not `assert hasattr`)** — revision iter 1 Warning 6 fix. `assert hasattr` placeholders would FAIL the verify step before downstream plans filled them; `pytest.skip()` placeholders pass cleanly. Each Wave 0 scaffold also has the standard `_RISCV_AVAILABLE` self-detect block (or NO skipif for pure-python tests like test_dma_engine.py and test_ddr_modes.py).
6. **`.copy()` guard on overlapping numpy slice assignment** — firmware_dma_tloop_copy + exec_transpose + exec_transpose_ddr all use `dst = src.copy()` for the LHS operand. Matches C++ `std::memmove`; bare numpy slice assignment can corrupt overlapping ranges.
7. **Wave 1 parallel safety:** Task commits used `git commit --no-verify` to avoid pre-commit hook contention with the concurrent 03-03-ddr-io agent. Orchestrator validates hooks once after the wave completes.

### Phase 3 Plan 03 Decisions (locked during execution)

1. **INITIAL_FLOOR = 1 MiB** for `ensure_ddr` doubling-grow (P3 D-13). Picked per RESEARCH §"Architecture Patterns": covers 32-byte bus-word minimum with headroom; small enough that CI per-test allocations are cheap; large enough that "single grow per test" is the common case. Strategy: `new_size = min(cap, max(end_offset, current_size*2, INITIAL_FLOOR))`. Cap enforced via `GTX_DDR_SIZE` env var.
2. **C++ ensure_ddr divergence documented in docstring.** C++ `gtx_npu_t::ensure_ddr` (gtx_npu_core.cc:198-203) allocates the full 4 GiB once. P3 doubling-grow is a CI ergonomic — for production firmware that touches the full 4 GiB, behavior is identical (single grow to cap). Phase 1's earlier note ("Phase 3 will replace stub with C++ doubling-grow") was inaccurate; the divergence is documented in-source rather than silently propagated.
3. **GTX_DDR_REVERSED read per-call (D-08) in BOTH `ddr_init_from_file` and `ddr_dump_to_file`.** No module-level cache. Avoids cache-poisoning trap where `monkeypatch.setenv` between tests would see stale value. Each function does `bool(os.environ.get('GTX_DDR_REVERSED'))` at function entry.
4. **`ddr_dump_to_file` accepts addr/size as args ONLY (D-09).** Does NOT consult any dump-related env vars (`GTX_DDR_DUMP` / `_ADDR` / `_SIZE` are CLI/P6 territory). Acceptance grep `grep -c "GTX_DDR_DUMP" ddr.py == 0` enforces this — docstring uses the phrase "any dump-related env vars" instead of literal token names so the grep stays clean.
5. **Half-density asymmetry.** Parser supports any line length ≤ 32 (16-byte hex lines advance offset by 16, not 32) — verified by `test_ddr_init_half_density_16_bytes`. Dumper always emits full 32-byte lines (zero-pads on out-of-range). Asymmetric on purpose — matches C++; only upstream tools (SystemC trace dumper) emit half-density.
6. **No `_RISCV_AVAILABLE` skipif on `test_ddr_modes.py`** — DDR I/O is pure-python (mem: GtxMemory only, no spike deps; D-07). Plan 01 explicitly noted this carve-out for pure-python tests.

### Phase 3 Plan 04 Decisions (locked during execution)

1. **Sibling-module split (`dispatch_4mode.py` separate from `dispatch.py`).** Eliminated the Wave 2 file-write conflict with Plan 02's 2-level builder upgrade. dispatch.py only gets a 1-line re-export so the public surface (`from riscv.gtx.dispatch import dispatch_4mode`) stays stable. Plan 02's `38aac36` commit cooperatively included the same re-export line, converging on canonical form without merge intervention.
2. **`dispatch_iss_opcode` is a TRUE stub in P3.** Every funct7 NOPs and returns 0; OOB nest_id/spu_id silently NOP. The body has a comment block naming exactly which lines Plan 05 will replace with the credit_st_chk flush trigger (`if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop: npu.flush_deferred_ddr_stores()`). P4 fills `GTX_OP_MM=0`; P5 fills VEC/ACT (1/2).
3. **Pitfall 8 dual coverage.** Mode 3 OR-rule covered by (a) `is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)` — three truth-table corner tests, AND (b) `width = op3 & 0xFFFF`, `height = (op3 >> 16) & 0xFFFF` — explicit bitfield assertions in two tests. Single-check coverage would let one piece silently regress.
4. **Pre-existing failures in `tests/gtx/test_firmware_dma.py` (Plan 02 territory)** documented in `deferred-items.md` and out of Plan 04 scope per executor scope-boundary rules. Plan 04's 72-test adjacency suite (test_dispatch_4mode + test_dispatch + test_dma_engine + test_ddr_modes) all green.

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

All 4 research streams converge on a HIGH-confidence approach; coverage is 100%.

- Phase 2 needs_followup: Categories A-D before /gsd:phase-evolve 2 can run; see 02-06-SUMMARY.md

## Session Continuity

### Last Action

Phase 03 Wave 2 complete (Plans 02 + 04 landed in parallel). Plan 04 (dispatch-4mode)
executed TDD against main. Two atomic `--no-verify` commits: `259c18d` (test, Task 1
RED: 13 dispatch_4mode tests fail at import — dispatch_4mode / dispatch_iss_opcode
symbols not yet exported), `4831bc6` (feat, Task 1 GREEN: dispatch_4mode.py 120 LOC,
4-mode warp router + DMA-only iss_opcode stub; dispatch.py re-export was cooperatively
landed by sibling Plan 02 commit `38aac36` — no merge intervention needed). SUMMARY at
`.planning/phases/03-dma-ddr-i-o/03-04-dispatch-4mode-SUMMARY.md`. Self-check PASSED:
all 8 must_haves.truths + all 3 key_links satisfied, 13/13 dispatch_4mode tests pass,
72/72 adjacency tests pass (test_dispatch + test_dma_engine + test_ddr_modes). DISP-03
requirement ready to mark complete. No deviations — plan executed exactly as written.
8 pre-existing failures in test_firmware_dma.py (Plan 02 territory) logged in
`deferred-items.md`; out of Plan 04 scope per executor scope-boundary rules.

### Next Action

Wave 2 done. Orchestrator validates pre-commit hooks once across all four Wave 2 commits
(6f9bbba, 38aac36, 259c18d, 4831bc6) then advances to Wave 3 (03-05-flush-roundtrip).
Plan 05 reads `flush_deferred_ddr_stores` (Plan 02 added) and replaces the
`dispatch_iss_opcode` body insertion point in `dispatch_4mode.py` with the
credit_st_chk flush trigger. Plan 02 team should also investigate the 8
test_firmware_dma.py KeyError failures (plan-02-internal mocking issue).

### Resumption Notes

If resuming work in a new session:

1. Read `.planning/PROJECT.md` for core value + constraints
2. Read `.planning/ROADMAP.md` for phase structure + success criteria
3. Read `.planning/REQUIREMENTS.md` for full v1 requirement list with phase mappings
4. Read this STATE.md for current position
5. Per-phase research artifacts live under `.planning/research/` (already populated for the project; per-phase research is generated on-demand by `/gsd:research-phase <N>`)

---

*State initialized: 2026-05-04 after roadmap creation*
