---
phase: 04-mm-subsystem
plan: 02
subsystem: compute
tags: [gemm, fp16, fp32-accumulate, numpy, stateless-kernel, p7-numba-boundary]

requires:
  - phase: 04-mm-subsystem
    provides: "Wave 0 RED scaffolds (test_op_mm.py 11 tests including 3 gemm_core scaffolds); _verify_minimal BE FP16 compare; .elf fixture"
provides:
  - "riscv.gtx.gemm_core leaf module: 3 stateless functions (gemm_core, gemm_reduce_sum_a, gemm_dot)"
  - "Explicit Python 3-loop FP32-accumulate GEMM kernel (NOT np.matmul) -- bit-exact match to gtx_npu_mm.cc:73-79 scalar accumulate ordering"
  - "Pitfall 2 anti-pattern regression coverage: 1000-element FP16 vector with 1e-4 tail stays finite via FP32 internal"
  - "P7 numba @njit boundary marked: array-in/scalar-in/array-out/scalar-out, zero spike imports"
affects: [04-03-mm-engine, 04-04-ops-mm, 04-05-regression, 07-numba-optimization]

tech-stack:
  added: []
  patterns:
    - "Pure stateless leaf module (no riscv.gtx.* imports) per D-01/D-03 -- mirrors P3 dma_engine.py boundary"
    - "Explicit 3-loop FP32 accumulate (NOT np.matmul) -- locked by RESEARCH np.matmul Bit-Exactness analysis (4 ULP drift on 8.2% of trials)"
    - "Bit-exact unit test via view(np.uint16) compare (D-15) -- avoids FP16 NaN comparison pitfall"

key-files:
  created:
    - "src/main/python/riscv/gtx/gemm_core.py"
  modified:
    - "tests/gtx/test_op_mm.py"

key-decisions:
  - "Explicit 3-loop FP32 accumulate -- NOT np.matmul (RESEARCH lock: BLAS drift up to 4 ULP exceeds verify.py --ulp 1)"
  - "gemm_dot uses explicit Python loop, NOT np.dot (BLAS dispatch may drift like matmul for long vectors)"
  - "bias added in FP32 before single FP16 cast (Pitfall 2 lock)"
  - "Surgical scope: only the 3 named gemm_core scaffolds GREEN-filled; the 8 other MM-02/03/05 scaffolds left for Plans 03/04 (parallel-safe)"

patterns-established:
  - "Leaf module discipline: gemm_core.py imports only `from typing` + `numpy` + `numpy.typing` -- enables P7 @njit wrap with zero refactor"
  - "Bit-exact FP16 oracle: in-test 3-loop FP32 oracle compared via view(np.uint16); deterministic regardless of host BLAS version"
  - "Pitfall 2 regression idiom: long mixed-magnitude FP16 vector (1.0 + 1e-4 alternating) traps FP16-internal accumulate as inf or saturation"

requirements-completed: [MM-01]

duration: 4min
completed: 2026-05-06
---

# Phase 4 Plan 02: gemm_core Stateless Kernel Summary

**3 spike-independent FP32-accumulate GEMM functions (gemm_core, gemm_reduce_sum_a, gemm_dot) directly ported from gtx_npu_mm.cc:27-94/200-211/262-265 -- explicit 3-loop locks bit-exactness against C++ scalar accumulate ordering; Plans 03/04 unblocked.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-06T00:26:31Z
- **Completed:** 2026-05-06T00:30:37Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

- `riscv.gtx.gemm_core` lands as a 150-LOC pure-Python leaf module -- 0 imports from `riscv.gtx.*`, 0 dependencies on `npu`/`proc`/`insn`. Importable in offline (no `_riscv.so`) configurations.
- 3 named MM-01 scaffolds in `tests/gtx/test_op_mm.py` transition skip -> pass; the other 8 scaffolds (MM-02/03/05 territory of Plans 03/04) untouched and still skip cleanly.
- Pitfall 2 regression locked end-to-end: `gemm_reduce_sum_a([1.0, 1e-4]*1000 as FP16)` returns `~999.95` (finite). FP16-internal accumulate would saturate or inf.
- 16x16x16 random-FP16 GEMM unit test: bit-exact match (per-cell uint16 view equality) against in-test explicit 3-loop FP32 oracle.
- Full test suite delta: 180 passed -> **183 passed**, 19 skipped -> **16 skipped**, 0 failed (P3+P4 baseline preserved).

## Task Commits

1. **Task 1: gemm_core.py module (3 stateless functions)** -- `7b2a115` (feat)
2. **Task 2: GREEN-fill 3 gemm_core scaffolds in test_op_mm.py** -- `13d8a58` (test)

_Note: TDD-style RED was already laid by Plan 01 (Wave 0 scaffolds via `pytest.skip`). This plan's "RED" was therefore the inherited skip state; Tasks 1+2 are GREEN steps. No refactor commit needed -- the 3-loop body is C++-direct, no shape adjustments triggered._

## Files Created/Modified

- `src/main/python/riscv/gtx/gemm_core.py` (NEW, 150 LOC) -- module-level docstring cites C++ source line numbers, RESEARCH np.matmul lock, Pitfall 2; 3 functions: `gemm_core`, `gemm_reduce_sum_a`, `gemm_dot`; type-hinted via `NDArray[np.float16]` and `Optional[NDArray[np.float32]]`; raises `ValueError`/`TypeError` for shape/dtype/bias-missing mismatches.
- `tests/gtx/test_op_mm.py` (MODIFIED, 3 of 11 scaffolds GREEN-filled, +44/-3 lines) -- `test_gemm_core_explicit_3loop_matches_oracle`, `test_gemm_core_fp32_internal_not_fp16`, `test_gemm_core_signature_stateless`. Other 8 scaffold bodies untouched.

## Decisions Made

- **Explicit Python 3-loop, NOT np.matmul** -- RESEARCH np.matmul Bit-Exactness Analysis lock: BLAS drifts up to 4 ULP / 0.0078 abs on 41/500 random 16x16x16 FP16-cast-to-FP32 trials, which exceeds `verify.py --ulp 1 --atol 0.001`. Strict-mode (P4 D-14) regression cannot tolerate any drift. P7 numba `@njit` will reactivate BLAS-equivalent throughput while preserving the scalar accumulate ordering.
- **gemm_dot uses explicit Python loop, NOT np.dot** -- np.dot also dispatches to BLAS for long vectors and may drift. Vector lengths in MM_V/MMC_V are short (col_A typically <= 16 per gtx_npu_mm.cc), so loop overhead is negligible.
- **bias_fp32 (Optional[NDArray[np.float32]]) explicit kwarg** -- the surface sketch in CONTEXT D-03 mentioned `prior_accum: float`, but the C++ MM_O/MMC_O scalar `mxe_accum[nest, spu]` chain semantics are scalar (single FP32 per (nest, spu) cell), while the MM/MMC matrix bias would need an (M, N) array. Plan 02 cleanly separated: matrix bias goes through `gemm_core(has_bias, bias_fp32)` while scalar chain accumulation goes through `gemm_reduce_sum_a(prior_accum)` / `gemm_dot(prior_accum)`. mm_engine (Plan 03) selects the right kernel per variant.
- **Surgical edit discipline** -- only the 3 named gemm_core scaffolds (MM-01) were modified. The 8 other scaffolds in test_op_mm.py (test_handler_registry_*, test_exec_mm_*, test_decode_firmware_mm_args, test_verify_minimal_be_fp16_pairs) are owned by Plans 03 and 04. This guarantees Wave 1b plans 03 and 04 will not see merge conflicts on the same test file.

## Deviations from Plan

None -- plan executed exactly as written. All 3 plan-supplied test bodies were applied verbatim (with one minor character substitution: literal `≈` in the assertion message reverted to ASCII `~=` to keep the file pure-ASCII consistent with the rest of the test suite).

## Issues Encountered

None. The smoke verification command in `<verify>` for Task 1 passed first-run (2x2 GEMM `[[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]`). The 3 targeted pytest cases passed first-run. Full P3+P4 suite ran clean (183 passed, 16 skipped, 0 failed).

## Known Stubs

None introduced by this plan. The 8 other scaffolds in test_op_mm.py remain `pytest.skip(...)` placeholders (intentional, owned by Plans 03/04). No code stubs in `gemm_core.py` -- every function has a complete implementation.

## Self-Check: PASSED

**Created files exist:**
- `src/main/python/riscv/gtx/gemm_core.py` ✓ (FOUND)

**Modified files exist:**
- `tests/gtx/test_op_mm.py` ✓ (FOUND, contains the 3 GREEN-filled function bodies via grep)

**Commits exist (verified via `git log --oneline`):**
- `7b2a115` ✓ (FOUND)
- `13d8a58` ✓ (FOUND)

**Verification commands all pass:**
- `python3 -c "from riscv.gtx.gemm_core import gemm_core, gemm_reduce_sum_a, gemm_dot"` -> succeeds
- `grep -E "np\\.matmul\\(|np\\.dot\\(" src/main/python/riscv/gtx/gemm_core.py` -> NO MATCHES (BLAS lock)
- `pytest tests/gtx/test_op_mm.py::test_gemm_core_*` -> 3 passed
- `pytest tests/gtx/ -q --noconftest -o "addopts="` -> 183 passed, 16 skipped, 0 failed

## Next Wave Readiness

Wave 1b plans 03 (mm_engine) and 04 (ops/mm) can now consume `gemm_core` directly:

- **Plan 03 (mm_engine)** will `from riscv.gtx.gemm_core import gemm_core, gemm_reduce_sum_a, gemm_dot` inside `firmware_mm()`. The variant-to-kernel mapping is fully unambiguous now: `mm`/`mmc` -> `gemm_core(has_bias=is_accumulate, bias_fp32=mxe_matrix_or_None)`; `mm_o`/`mmc_o` -> `gemm_reduce_sum_a(prior_accum=...)`; `mm_v`/`mmc_v` -> `gemm_dot(prior_accum=...)`.
- **Plan 04 (ops/mm)** does not import gemm_core directly -- it goes through Plan 03's `mm_engine.firmware_mm`. So Plan 04 only needs Plan 03 to land first; it does not block on gemm_core's exact surface.
- **Plan 05 (Wave 2 regression)** will exercise gemm_core indirectly through the `.elf` subprocess path. Zero-init operand staging means `gemm_core(zeros @ zeros) -> zeros` is the bit-exact match against the zero-init golden hex from Plan 01 -- this verification path is now end-to-end ready.

No blockers. The leaf-module boundary is locked, Pitfall 2 regression coverage is in place, and the 3-loop explicit accumulate ordering matches `gtx_npu_mm.cc:73-79` line-for-line.

---
*Phase: 04-mm-subsystem*
*Completed: 2026-05-06*
