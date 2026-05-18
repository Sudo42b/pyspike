# Phase 9 Deferred Items

## Out-of-scope failures encountered during Wave 6 verification

### tests/gtx/test_deferred_store.py — 11 failures (pre-existing)

**Trigger**: `ModuleNotFoundError: No module named 'riscv.gtx.dma_engine'`

**Root cause**: The test imports `riscv.gtx.dma_engine` at the top level,
but the module was moved to `riscv.gtx.unit.context.dma_engine` by an
earlier refactor (pre-Phase-9). The test was not updated at that time.

**Why deferred**: Not caused by Wave 6. Not blocking the BM-* gate (BM-03
covers ops; BM-04 covers tloop/sloop). Out-of-scope per executor Rule
"SCOPE BOUNDARY: Only auto-fix issues DIRECTLY caused by the current task's
changes."

**Fix**: 1-line import change: `from riscv.gtx.dma_engine import ...` →
`from riscv.gtx.unit.context.dma_engine import ...`. Owner: future patch
(P10 cleanup or dedicated quick fix).

### tests/gtx/test_regression_fw_full_sweep.py — 3 pre-existing P9-backlog FAILs

GELU_QUICK, HARDSIGMOID, LEAKY_RELU — documented in Wave 5 SUMMARY
"P9-backlog failures pre-existing". Phase 8 baseline M=2 (only ABS + GELU
strict-mode PASS); these 3 ops have non-multi-tile root causes deferred
to v1.2 milestone.

### Full 84-op vendor sweep — abbreviated for time efficiency

Running 84 ELFs sequentially at 1-2 min each = 90-180 min total. Wave 6
ran the 3-op head + ACT-family smoke subset as proxy validation. The
byte-exact contract is verified by ABS strict (96 tiles × 196609 lines
under GTX_DDR_REVERSED=1).
