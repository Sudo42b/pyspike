# Wave 2a Gate Results

Date: 2026-05-18
Commit (HEAD after Task 5 + shim removal): 8b35f7c
Gate Status: **GREEN**

## Summary

Wave 2a (plan 09-02a-ops) ported the 4 op-handler modules
(`unit/ins/ops/{spr,mm,vec,act}.py`) and `unit/csr/register.py`
docstring from torch to xp. The 6-op smoke gate is GREEN with 4 PASS +
1 SKIP per Wave 0 convention, matching the Wave 1 baseline exactly.
ABS strict walltime improved from 110.97s (Wave 1) to 96.68s
(post-Wave-2a) — a 13% reduction, back inside the D-08 85-105s budget.

Additionally, the 3 f16 WAVE-1-SHIM accessor sites in `memory.py`
(`l0_f16` / `l1_f16` / `l2_f16`) were removed: all consumers in
`unit/ins/ops/*.py` were ported and now bypass these accessors by
reading raw `npu.mem.l[012][nest, spu].view(xp.float16)` storage
directly (same pattern Wave 1b's `flush_deferred_ddr_stores` adopted).

## Smoke Set (D-07, literal Wave-0 convention)

The plan filter `-k 'ABS or GELU or RELU or SIGMOID or TANH or
SOFTMAX'` is ambiguous (substring widens to 9 ops; SOFTMAX absent
from vendor sweep — only SOFTPLUS exists). Literal 6-op smoke =
ABS + GELU + RELU + SIGMOID + TANH (5 ops, TANH skip).

Command:
```
uv run pytest \
  "tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[ABS]" \
  "tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]" \
  "tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[RELU]" \
  "tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[SIGMOID]" \
  "tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[TANH]" \
  --no-cov -v
```

Result: **PASS** (4 passed + 1 skipped TANH)
Stats: 4 passed (ABS, GELU, RELU, SIGMOID) / 0 failed / 1 skipped
       (TANH — vendor `.elf` absent; same skip status as Wave 1/0)
Wall: 137.81s (entire 5-op subset; Wave 1 baseline 153.30s → 10%
      improvement)

The 3 pre-existing P9-backlog regressions (GELU_QUICK, HARDSIGMOID,
LEAKY_RELU) surface only when using the broader `-k` substring
expansion. Per Wave 0 / Wave 1 gate convention, the literal 6-op
smoke is the gate; substring collateral is acknowledged but out of
scope (tracked in `.planning/seeds/p9-vendor-sweep-non-multi-tile-bugs.md`).

## Tile-2 Unit Test (P8 MTDMA-03)

Status: **N/A** — `tests/gtx/test_multi_tile_dma.py` removed by
commit 6bc2c3f (2026-05-14) pre-Phase-9. Same as Wave 0 / Wave 1
gates. ABS strict PASS through all 96 tiles (196609 lines byte-exact)
IS the multi-tile invariant proxy.

## ABS Strict Walltime (D-08 budget: 85-105s)

Wall: **96.68s** (test wall via `uv run pytest`; subprocess wall
including ~10s pytest collection: 112.43s).

| Stage | ABS wall (test) | In D-08 budget (85-105s)? |
|-------|-----------------|---------------------------|
| Phase 8 baseline (commit 2b0c66e) | 94.82s | YES |
| Wave 0 baseline | 144.16s | NO (pre-existing perf drift) |
| Wave 1b post-shim | 110.97s | NO (6% over) |
| **Wave 2a post-port** | **96.68s** | **YES** |

The ~14s reduction vs Wave 1 comes from two factors:
1. Removing per-call `torch.from_numpy(arr)` shim wrap on the hot
   memory accessor paths (`mem.l*_byte` / `mem.l*_f16` hit thousands
   of times per ABS tile across 96 tiles).
2. ABS exercises `vec.py` heavily (`abs_v` mnemonic via the
   `_exec_vec_unary` handler → `_apply_unary` → `xp.abs`). The
   xp.abs path avoids the torch.from_numpy round-trip entirely
   because vec.py now reads raw xp storage via
   `npu.mem.l1[nest,spu].view(xp.float16)`.

Marginal improvements expected when Wave 5 (dma_engine.py) lands —
the remaining `mem.l1_byte` / `mem.l2_byte` / `mem.ddr.read` shim
sites are exactly the hot DMA paths.

## Bridge-Shim Site Table (post Wave 2a)

The shim was originally landed at 7 accessor sites in Wave 1b. After
Wave 2a (this plan), 3 of those sites are removed. The 4 surviving
shims are inherited by Waves 5 / 6 per the 09-01b-SUMMARY removal
table:

| Accessor | Status | Removal owner |
|----------|--------|---------------|
| `GtxMemory.l0_byte` | **SHIMMED** | Wave 5 (09-02b-engines) — dma_engine.py L155/179 |
| `GtxMemory.l1_byte` | **SHIMMED** | Wave 6 (09-03-finalize) — tloop_buffer.py L483 |
| `GtxMemory.l2_byte` | **SHIMMED** | Wave 6 (09-03-finalize) — tloop_buffer.py L459/467/477/485 |
| `GtxMemory.l0_f16` | **REMOVED (this plan)** | n/a |
| `GtxMemory.l1_f16` | **REMOVED (this plan)** | n/a |
| `GtxMemory.l2_f16` | **REMOVED (this plan)** | n/a |
| `DDR_MEMORY.read` | **SHIMMED** | Wave 5 (09-02b-engines) — dma_engine.py L266/345/534/647/664 |

After Wave 5 (plans 09-02b-engines): `l0_byte` + `ddr.read` shims can
be removed. Only `l1_byte` + `l2_byte` survive (tloop_buffer.py).
After Wave 6 (plan 09-03-finalize): all shims gone; `_torch_view`
helper + module-level `import torch` deleted; `memory.py` returns to
its pure-xp form.

## D-10 DDR-on-GPU Verification

xp=numpy: 96.68s ABS walltime (no VRAM concern).
xp=cupy: **SKIP — no GPU available** in this CI/dev environment.

## D-11 SPR-on-GPU Perf Verification

xp=numpy: 96.68s ABS walltime (SPR access cost on the FAST path).
xp=cupy: **SKIP — no GPU available**.

## Wave 2a Sign-Off

- [x] `unit/ins/ops/spr.py` torch-free; xp imported for `xp.tile` /
  byte-array primitives in CPSVR/MVSVR (H-3 partially honored — pure
  Python ints for arithmetic; xp.tile for byte-stream replication
  which was required to replace torch.Tensor.repeat tile semantics)
- [x] `unit/ins/ops/mm.py` torch-free (44 → 0 refs); xp.matmul / xp.dot
  / xp.sum preserve BLAS-equivalent semantics; FP32-internal-accumulate
  discipline preserved
- [x] `unit/ins/ops/vec.py` torch-free (52 → 0 refs); `_apply_unary` +
  sasmd/dot/vsum/clamp/cumsum/arange all on xp; cumsum uses `axis=`
  not `dim=`; clamp → xp.clip
- [x] `unit/ins/ops/act.py` torch-free (81 → 0 refs); 7 activations +
  2 pool + 9 cvt + FP8 LUTs; FP8 path is single deterministic LUT-only
  implementation (Option-B, H-1)
- [x] `unit/csr/register.py` docstring updated for xp
- [x] 3 f16 shim sites removed from `unit/memory.py`
- [x] **Smoke set GREEN** — literal 4 PASS + 1 SKIP per Wave 0 convention
- [/] Tile-2 GREEN — file removed pre-Phase-9 (same as Wave 0/1); ABS
  multi-tile (96 tiles × 196609 lines byte-exact) is the invariant proxy
- [x] **ABS walltime in 85-105s band** — 96.68s (back INSIDE budget;
  13% improvement vs Wave 1's 110.97s)
- [/] D-10 verification — SKIP (no GPU available)
- [/] D-11 verification — partial (xp design landed; perf measurement
  deferred to plan 09-03 Task 7 / BM-04)

## Unit-Level Evidence

Wave 1 baseline + Wave 2a updates: **73 / 73 GREEN**.

```
tests/gtx/test_memory_layout.py        15 passed (Wave 1a + shim-aware updates)
tests/gtx/test_dma_roundtrip.py         6 passed (Wave 1a + shim-aware updates)
tests/gtx/test_register_file_xp.py     10 passed (Wave 1b)
tests/gtx/test_csr_registry_chain.py   10 passed (Wave 1b)
tests/gtx/test_xp_alias.py              4 passed (Wave 0)
tests/gtx/test_memory_torch_shim.py    17 passed (Wave 1b + Wave 2a f16-flip)
tests/gtx/test_custom_dispatch_chain.py 11 passed (pre-Phase-9)
                                       ——
                                       73 passed in 34.89s
```

(test_npu_xp.py has 1 pre-existing failure — `test_no_torch_in_npu_source`
trips on a docstring in `flush_deferred_ddr_stores` that mentions
torch.Tensor as part of describing what's NOT in use. This is a
Wave-1b regression not caused by Wave 2a; npu.py code is unchanged.
Documented as deferred-items below.)

## Deferred Issues (out of Wave 2a scope)

1. **`test_no_torch_in_npu_source` false positive** —
   `tests/gtx/test_npu_xp.py:23-28` parses npu.py line-by-line for
   "torch.", excluding `#` comments but NOT excluding docstring text.
   `npu.py:350` is inside a `flush_deferred_ddr_stores` docstring
   ("Bypasses ``mem.l2_byte()`` (which is Wave-1-SHIM-wrapped to return
   a torch.Tensor for un-ported Wave 2/3 callers)..."). The docstring
   is informational and describes the Wave-1-SHIM bridge contract;
   removing it would lose useful context. The test should exclude
   docstrings (triple-quote-delimited blocks) before flagging "torch.".
   Pre-existing Wave 1b commit 6072b37 — not caused by Wave 2a.

2. **3 substring-match collateral failures** (GELU_QUICK, HARDSIGMOID,
   LEAKY_RELU) — same status as Wave 1. Pre-existing P9-backlog
   regressions in vec.py:339 path (tracked in
   `.planning/seeds/p9-vendor-sweep-non-multi-tile-bugs.md`).

## Next Phase Readiness

**Plan 09-02b-engines (Wave 5: dma_engine.py + tloop_buffer.py) entry
unblocked.** Per the shim site table, 09-02b is responsible for:
- Porting `dma_engine.py` off torch (lines 21, 117-118, 154-155, 178-
  206, 221-222, 266, 289, 344-345, 387-388, 424, 474, 491, 534, 546,
  578, 591, 647, 651, 664, 669, 682, 687).
- Removing the `l0_byte` shim from memory.py.
- Removing the `ddr.read` shim from memory.py.
- Verifying the smoke gate stays GREEN.

Plan 09-03-finalize (Wave 6) then handles tloop_buffer.py + npu.py
residual + __init__.py + _verify.py + the final `l1_byte` / `l2_byte`
shim removal + `_torch_view` helper + module-level torch import
deletion.

---

## Task Commits (this plan)

- Task 1: `de291ff` — feat(09-02a): port spr.py from torch to xp
- Task 2: `cfc2677` — feat(09-02a): port mm.py from torch to xp
- Task 3: `d62ba27` — feat(09-02a): port vec.py from torch to xp
- Task 4: `6b2e3c1` — feat(09-02a): port act.py from torch to xp (FP8 LUT-only)
- Task 5: `020ebb9` — docs(09-02a): update csr/register.py docstring
- Shim removal: `8b35f7c` — refactor(09-02a): remove l0/l1/l2_f16 shim

Plan metadata commit (this SUMMARY + STATE + ROADMAP): recorded after
SUMMARY.md self-check.
