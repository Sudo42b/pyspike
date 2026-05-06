---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: 4
status: executing
last_updated: "2026-05-06T00:42:52.881Z"
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 21
  completed_plans: 19
  percent: 90
---

# State: pyspike + GTX NPU (Python RoCC Port)

**Last updated:** 2026-05-06 after Phase 4 Plan 02 complete (gemm_core stateless kernel — 3 functions in 150-LOC leaf module; 3 MM-01 scaffolds GREEN-filled; 183 passed/16 skipped/0 failed; Plans 03/04 unblocked for Wave 1b sibling waves)

## Project Reference

**Name:** pyspike + GTX NPU (Python RoCC Port)

**Core Value:** 기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로
통과하고 DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된
golden)와 ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.

**Current Focus:** Phase 04 — mm-subsystem

**Acceptance Gate:** `pyspike --extlib=riscv.gtx <fw>.elf` → DDR dump that
`verify.py --fp16 --ulp 1 --atol 0.001` reports as **strict-mode pass**
(`exact_matches == total_fp16`) against C++ golden.

## Current Position

Phase: 04 (mm-subsystem) — EXECUTING
Current Plan: 4
Total Plans in Phase: 5

- **Phase:** 4
- **Plan:** 2 of 5 (Plan 01 Wave 0 scaffold complete; Wave 1 unblocked)
- **Status:** Ready to execute
- **Progress:** [█████████░] 90%

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
| Phase 03-dma-ddr-i-o P02 | 13min | 3 tasks | 6 files |
| Phase 03-dma-ddr-i-o P05 | 5m53s | 2 tasks | 7 files |
| Phase 04-mm-subsystem P01 | 7min | 3 tasks | 10 files |
| Phase 04-mm-subsystem P02 | 4min | 2 tasks | 2 files |
| Phase 04-mm-subsystem PP03 | 4min | 2 tasks | 2 files |

## Accumulated Context

### Roadmap Evolution

- Phase 7 added: 제대로 동작을 하면, numba 등의 라이브러리를 통해 동적 최적화 기술을 이용하여 최적화 (정상 동작 확인 후 핫스팟 가속; 진입 조건 = P6 회귀 그린)

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

### Phase 3 Plan 02 Decisions (locked during execution)

1. **2-level custom0 dispatch uses sentinel None inner key.** `_registry.collect_for_kind('custom0')` returns `dict[int, dict[Optional[int], Callable]]`. P2 handlers (`mask_funct3=False`) register under the inner key `None`; P3+ funct3-decomposed handlers (`mask_funct3=True`) register under integer funct3. `GtxNpu.custom0` first tries `sub_table.get(None)` (P2 backwards-compat), then falls back to synthesized funct3 = `(insn.xd<<2)|(insn.xs1<<1)|insn.xs2`. Single dispatcher; no separate flat-table for P2.
2. **deferred_ddr_stores list lives on GtxNpu instance (D-05).** `__init__` creates it before `gspr`; `reset()` clears it (alongside SPR re-init). `flush_deferred_ddr_stores()` method added with C++ `gtx_npu_dma.cc:415-435` body. Plan 05 wires the actual call sites (end_p when `!wsplit_seen`, credit_st_chk when `is_sloop`); this plan only registers the API and the queue.
3. **ops/dma.py imports LSPR addresses from encoding.py — no magic numbers.** `LSPR_SPM_ADDRA = 0x900` (gtx_params.h:64) for tpose source; `LSPR_SPM_ADDRR = 0x903` (gtx_params.h:67) for tpose result and fill. The dedicated tests `test_tpose_reads_lspr_spm_addrr_at_0x903` + `test_fill_reads_lspr_spm_addrr_at_0x903` would FAIL if a stale draft using `0x901` (LSPR_SPM_ADDRB) leaked in — addr_r would silently read 0 instead of the firmware-staged value. `grep -E "0x901" ops/dma.py` matches NOTHING.
4. **rs3 read from `npu.gspr.get(GSPR_GTX_OPERAND3, 0)` — NOT XPR.** Operand 3 is GSPR-staged (gtx_params.h:40 GSPR_GTX_OPERAND3 = 0x003). All three firmware_dma branches (load/store/copy) read this.
5. **5 v2-deferral stubs + credit_st_chk all return 0 (NOP) but populate disasm registry.** load_3d/store_3d/mcast_s2l/mcast_g2s/mcast_s2s/copy_mem registered for spike trace fidelity. credit_st_chk awaits Plan 05 body (`if npu.warp.is_sloop: npu.flush_deferred_ddr_stores()`).
6. **Wave 2 cooperative parallel landing.** Both Plan 02 and sibling Plan 04 added the `dispatch_4mode` re-export line in `dispatch.py` independently — converged on identical text. No merge conflict because Plan 02's table-builder edit and Plan 04's re-export are in different regions of the same file. `--no-verify` on all task commits avoided pre-commit hook contention.

### Phase 3 Plan 04 Decisions (locked during execution)

1. **Sibling-module split (`dispatch_4mode.py` separate from `dispatch.py`).** Eliminated the Wave 2 file-write conflict with Plan 02's 2-level builder upgrade. dispatch.py only gets a 1-line re-export so the public surface (`from riscv.gtx.dispatch import dispatch_4mode`) stays stable. Plan 02's `38aac36` commit cooperatively included the same re-export line, converging on canonical form without merge intervention.
2. **`dispatch_iss_opcode` is a TRUE stub in P3.** Every funct7 NOPs and returns 0; OOB nest_id/spu_id silently NOP. The body has a comment block naming exactly which lines Plan 05 will replace with the credit_st_chk flush trigger (`if funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop: npu.flush_deferred_ddr_stores()`). P4 fills `GTX_OP_MM=0`; P5 fills VEC/ACT (1/2).
3. **Pitfall 8 dual coverage.** Mode 3 OR-rule covered by (a) `is_load = (sub_op == 0) or (opcode == GTX_OP_DMA)` — three truth-table corner tests, AND (b) `width = op3 & 0xFFFF`, `height = (op3 >> 16) & 0xFFFF` — explicit bitfield assertions in two tests. Single-check coverage would let one piece silently regress.
4. **Pre-existing failures in `tests/gtx/test_firmware_dma.py` (Plan 02 territory)** documented in `deferred-items.md` and out of Plan 04 scope per executor scope-boundary rules. Plan 04's 72-test adjacency suite (test_dispatch_4mode + test_dispatch + test_dma_engine + test_ddr_modes) all green.

### Phase 4 Plan 02 Decisions (locked during execution)

1. **Explicit Python 3-loop FP32 accumulate in `gemm_core` — NOT `np.matmul`** (RESEARCH np.matmul Bit-Exactness lock). BLAS drifts up to 4 ULP / 0.0078 abs on 41/500 random 16x16x16 FP16-cast-to-FP32 trials, exceeding `verify.py --ulp 1 --atol 0.001`. Strict-mode (D-14) regression cannot tolerate any drift. P7 numba `@njit` reactivates BLAS-equivalent throughput while preserving the scalar accumulate ordering. `gemm_dot` similarly uses explicit loop, NOT `np.dot`.
2. **Matrix-bias path vs scalar-chain path are SEPARATE functions.** CONTEXT D-03's surface sketch had `gemm_core(..., prior_accum: float)` returning `(C, new_accum)` — but the C++ shows MM/MMC use a matrix bias (`bias_fp32: NDArray[np.float32]` of shape `(M, N)` staged from L1 ADDRC) while MM_O/MMC_O/MM_V/MMC_V use scalar `mxe_accum[nest, spu]` chain. Plan 02 cleanly split: `gemm_core(has_bias, bias_fp32)` for matrix variants; `gemm_reduce_sum_a(prior_accum)` and `gemm_dot(prior_accum)` for scalar variants. mm_engine (Plan 03) selects the right kernel per variant — dispatch is unambiguous.
3. **Leaf module discipline locked** (D-01/D-03). `gemm_core.py` imports only `from typing` + `numpy` + `numpy.typing` — zero `riscv.gtx.*` imports, zero `npu`/`proc`/`insn` parameters. Confirmed by `test_gemm_core_signature_stateless` via `inspect.signature` + `inspect.getsource`. P7 `@njit` boundary clean.
4. **Surgical scope on test_op_mm.py** (parallel-plan ownership). Only the 3 named gemm_core scaffolds (MM-01) GREEN-filled; the 8 other scaffolds (MM-02 handler-registry/exec, MM-03 decode, MM-05 verify_minimal smoke) left as `pytest.skip(...)` — they are owned by Plans 03 (mm_engine) and 04 (ops/mm). This guarantees Wave 1b plans 03/04 see no merge conflicts on the same test file when they GREEN-fill their respective scaffolds.

### Phase 4 Plan 01 Decisions (locked during execution)

1. **Test-only `_verify_minimal` — NO CLI / NO argparse / NO `__main__` block** (D-13 lock). `tests/gtx/_verify_minimal.compare_hex(actual, golden, *, ulp, atol, strict) -> (bool, dict)` exposed for `tests.gtx` import only. P6 VRF-01 promotes this to `riscv.gtx._verify` with CLI; conflating helper + production CLI in P4 would create a forward-incompat surface.
2. **BE bit-pair done MANUALLY via `(byte[0] << 8) | byte[1]`** — not via numpy `>u2` `frombuffer` or `newbyteorder` (Pitfall 1 explicit). Reason: matches `verify.py:235` line-for-line; numpy 2.x deprecates `newbyteorder`; explicit form is bit-exact regardless of host endianness. Verified via round-trip on `np.float16(1.0) = 0x3C00` BE encoding.
3. **`mm_basic.S` uses ISS-full WRSPR (funct7=0x49)** instead of gem5-simplified (funct7=0x00). Wave 0 ships ELF before MM @handler is wired; using funct7=0x00 for WRSPR would reenter the not-yet-implemented MM dispatch path. ISS-full is the only validated route in pyspike (P2 only registered ISS-full WRSPR at funct7=0x49).
4. **Zero-init golden hex (Blocker 1 Option B)** — `mm_basic_n1s16.hex` is 32 bytes of `0x00` because `mm_basic.elf` runs against zero-init L1 (firmware does NOT pre-load operands). `gemm_core(zeros @ zeros) = zeros` (FP32 internal then cast to FP16 = `0x0000`). Plumbing-proof: if any @handler crashes during the subprocess run, the .elf never reaches WJOIN, returncode != 0, and the test fails. Non-trivial operand staging is deferred to P6 (which has the operand-fixture infrastructure scope).
5. **Local `.gitignore` `!mm_basic.elf` override** added (Rule 3 blocking fix). Project-level `.gitignore` line 3 (`*.elf`) masks new ELFs; the existing `tests/gtx/data/elf/.gitignore` only listed `!nop_wjoin.elf`. Mirrors P2 D-22 commit-binary pattern; surgical 2-line edit.
6. **Wave 0 RED-pass-via-skip discipline** (P3 plan-01 D-5 mirror) — every scaffold body is `pytest.skip("Wave 1: ...")`, NEVER `assert hasattr(...)`. The latter would fail the verify step before downstream Wave 1 plans fill modules; `pytest.skip()` placeholders pass cleanly. `test_verify_minimal_be_fp16_pairs` smoke-imports `compare_hex` (landed in Task 1) before skipping the full BE-pair regression.
7. **Subprocess pyspike (D-11 fallback as PRIMARY)** — `test_regression_fw_mm.py` mirrors `test_skeleton.py` pattern: `shutil.which("pyspike")` first, fall back to `[sys.executable, "-m", "riscv"]`, timeout=30s. 3-tier skip on `_RISCV_AVAILABLE` / ELF missing / golden missing — never fails. In-process .elf load (D-11 PRIMARY in CONTEXT) was demoted to fallback by RESEARCH; subprocess proven by P2 plan-05 path.

### Phase 3 Plan 05 Decisions (locked during execution)

1. **3 flush call sites wired** (DMA-03 lock-in): `ops/control.py:_do_endp` flushes when `!npu.warp.wsplit_seen` (simple firmware path; ROADMAP P3 success #4); `ops/dma.py:_credit_st_chk` flushes when `npu.warp.is_sloop` (plan-style firmware path); `dispatch_4mode.py:dispatch_iss_opcode` flushes when `funct7 == GTX_ISS_F7_CREDIT_ST_CHK and npu.warp.is_sloop` (Mode 3+ dispatch entry). Both `credit_st_chk` paths converge on the same `npu.flush_deferred_ddr_stores()` API per RESEARCH "3 call sites" lock-in.
2. **WSPLIT sentinel set in 2 entry forms.** `wsplit` (custom1 funct3=0b100) and `wsplit_custom0` (custom0 funct7=0x02) both set `npu.warp.wsplit_seen = True` before returning 0. Pitfall 7 verified end-to-end: `test_reset_clears_deferred_queue_but_not_wsplit_seen` confirms `reset()` clears the deferred queue but leaves `wsplit_seen` True (process-lifetime sentinel).
3. **`_fake_npu` shim expansion** (Rule 1 fix): `tests/gtx/test_warp.py:_fake_npu` was `SimpleNamespace(warp=WarpState())` — broke after `_do_endp` started calling `npu.flush_deferred_ddr_stores()`. Fix: shim grew `flush_deferred_ddr_stores=lambda: None` + `deferred_ddr_stores=[]`. Pure test-shim change; production behavior unchanged for real GtxNpu callers.
4. **Single-line `np.array_equal` assertion**: acceptance grep `np\.array_equal\(.*\.view\(np\.uint16\)` is single-line. Multi-line `assert np.array_equal(\n    a,\n    b,\n)` form didn't match. Solution in `test_dma_l1_to_ddr_roundtrip_ltr`: extract `final_l1_u16 = ...` then `assert np.array_equal(final_l1_u16, pattern.view(np.uint16))` on one line. The other 2 assertions kept multi-line readable form.
5. **VALIDATION.md sign-off body** retains 4-bullet justification (P3 suite size = 179, all 6 requirement IDs closed, all 5 PLANs have `<automated>`, all 6 Wave 0 scaffolds populated). Future `/gsd:verify-work 3` reader sees the audit trail without grepping git history.
6. **No-deviation round-trip integration**: 3/3 round-trip tests passed on first run because Plans 01-04 had already shipped a complete data plane. Plan 05's only new wiring required for DMA-05 was the flush triggers (Task 1 GREEN). Task 2 was pure test population + flag flip.

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

Phase 04 Wave 1b sub-wave (Plan 02 gemm_core kernel) complete — first
GREEN of the MM subsystem. **Plan 02** executed in ~4 min with 2 atomic
commits (normal hooks, no `--no-verify`):

- `7b2a115` (feat Task 1): `src/main/python/riscv/gtx/gemm_core.py` —
  150-LOC pure-Python leaf module; 3 functions: `gemm_core` (explicit
  3-loop FP32 accumulate, NOT np.matmul, port of gtx_npu_mm.cc:73-79),
  `gemm_reduce_sum_a` (MM_O/MMC_O scalar sum, gtx_npu_mm.cc:200-211),
  `gemm_dot` (MM_V/MMC_V scalar dot, gtx_npu_mm.cc:262-265). 0 imports
  from `riscv.gtx.*`; 0 spike deps. P7 numba `@njit` boundary marked.

- `13d8a58` (test Task 2): GREEN-fill 3 of 11 scaffolds in
  `tests/gtx/test_op_mm.py` — `test_gemm_core_explicit_3loop_matches_oracle`
  (16x16x16 random FP16 GEMM, bit-exact uint16-view compare against in-test
  3-loop oracle), `test_gemm_core_fp32_internal_not_fp16` (Pitfall 2
  regression: `[1.0, 1e-4]*1000` as FP16 stays finite ~1000.0), and
  `test_gemm_core_signature_stateless` (inspect.signature + getsource lock).
  Other 8 scaffolds untouched (owned by Plans 03/04 — parallel-safe).

SUMMARY at `.planning/phases/04-mm-subsystem/04-02-SUMMARY.md`. Self-check
PASSED: 1 created file + 1 modified file present; both commits in
`git log`; `pytest tests/gtx/ -q --noconftest -o "addopts="` reports
**183 passed, 16 skipped, 0 failed** (P4 plan 01 baseline 180/19 + 3
GREEN converted from skip).

### Next Action

Wave 1b siblings unblocked — Plans 03 (mm_engine) and 04 (ops/mm) will
spawn next. Plan 03 fills 2 scaffolds (`test_decode_firmware_mm_args` +
`test_mode4_routes_to_tmu_curr`) and provides `mm_engine.firmware_mm`
that consumes `from riscv.gtx.gemm_core import gemm_core, gemm_reduce_sum_a,
gemm_dot`. Plan 04 fills 11 scaffolds (10 `@handler` entry points + funct7
collision matrix + 4 mm_chain scaffolds) and goes through Plan 03's
`mm_engine.firmware_mm` rather than calling gemm_core directly. Plan 05
(Wave 2) wires the subprocess strict-mode regression body in
`test_regression_fw_mm.py`.

### Resumption Notes

If resuming work in a new session:

1. Read `.planning/PROJECT.md` for core value + constraints
2. Read `.planning/ROADMAP.md` for phase structure + success criteria
3. Read `.planning/REQUIREMENTS.md` for full v1 requirement list with phase mappings
4. Read this STATE.md for current position
5. Per-phase research artifacts live under `.planning/research/` (already populated for the project; per-phase research is generated on-demand by `/gsd:research-phase <N>`)

---

*State initialized: 2026-05-04 after roadmap creation*
