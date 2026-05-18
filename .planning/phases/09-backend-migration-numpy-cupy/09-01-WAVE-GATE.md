# Wave 1 Gate Results

Date: 2026-05-18
Commit (HEAD after Task 3): a8a533e
Gate Status: **RED — see Failures**

## Summary

Wave 1 (plans 09-01a memory + 09-01b register_file/npu) ported `unit/memory.py`,
`unit/register_file.py`, and `npu.py` storage layer to xp. Unit tests for all
three files are GREEN (21 memory + 10 register_file_xp + 11 npu_xp +
10 csr_registry_chain = 52 tests). However, the **D-07 wave-end smoke gate
fails**: the vendor `.elf` sweep crashes during S-loop replay because Wave 2
files (`dma_engine.py`, `tloop_buffer.py`, ops/*.py) still expect torch.Tensor
inputs from the memory layer and call torch APIs like `.to(device)`,
`.numel()`, `.view(torch.float16)` on the xp.ndarray outputs.

This is a **Wave-spanning transit-state failure anticipated by CONTEXT D-06**
("Dual-import allowed but minimised") but **violates CONTEXT D-06's own
follow-on constraint**: "각 wave 끝에 ABS strict byte-exact GREEN 보장 필수
(intermediate 상태도 invariant 유지)". The current state cannot satisfy that
invariant without porting Wave 2/3 files.

## Smoke Set (D-07, 6 ops)

Command:
```
uv run pytest tests/gtx/test_regression_fw_full_sweep.py \
  -k 'ABS or GELU or RELU or SIGMOID or TANH or SOFTMAX' --no-cov -v
```

Result: **FAIL** — 8 failed, 1 skipped, 75 deselected (164.92s wall)
Per-op rc=255 stderr (all 8 failures identical root cause):

```
terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  AttributeError: 'numpy.ndarray' object has no attribute 'to'

At:
  /.../gtx/unit/context/dma_engine.py(348): firmware_dma_sloop_load
  /.../gtx/unit/context/dma.py(71): _firmware_dma_load
  /.../gtx/dispatch.py(102): wrapped
  /.../gtx/sloop_buffer.py(316): _replay
  /.../gtx/sloop_buffer.py(258): flush
  /.../gtx/npu.py(335): custom0
```

Failed ops: ABS, GELU, GELU_ERF, GELU_QUICK, HARDSIGMOID, LEAKY_RELU, RELU, SIGMOID
Skipped: 1 (TANH — vendor `.elf` absent or skip-marker)
SOFTMAX: not present in vendor sweep (only SOFTPLUS exists; same as Wave 0 finding)

## Tile-2 Unit Test (P8 MTDMA-03)

Command: `uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v`
Result: **N/A** — file removed by commit 6bc2c3f (2026-05-14, "test(gtx): reset
test infra for ORDER.md FSM redesign") as part of pre-Phase-9 cleanup. Same
status as Wave 0 gate. Deferred to v1.2.

## ABS Strict Walltime (D-08 budget: 85-105s)

Wall: **80.03s** (process wall; test failed in 67.69s subprocess + 12s test
fixture overhead). NOT a complete success-path measurement — the subprocess
crashes mid-replay BEFORE the compare_hex / DMA-tile orchestration phase.

For comparison:
- Wave 0 baseline: 144.16s (full ABS strict succeeded but pre-existing 144s drift)
- Phase 8 baseline (commit 2b0c66e): 94.82s
- User-env baseline (per STATE.md): 458.84s on 2026-05-18 debug session

In-budget: **N/A** — no successful walltime to compare against the 85-105s
budget. ABS does not currently complete. Re-baseline owned by plan 09-03 Task 7
(BM-04) after Wave 2/3 ports complete.

## D-10 DDR-on-GPU Verification

xp=numpy path: **N/A** (numpy ABS does not complete; see Failures).
xp=cupy path: **SKIP — no GPU available** in this CI/dev environment.
4 GiB default DDR allocation under cupy + 12 GB consumer-GPU concern is
deferred to a future GPU-equipped verification pass (plan 09-03 Task 7 or
v1.2 perf phase). Wave 1a landed the `GTX_DDR_SIZE=1G` source-comment
workaround near `DDR_MEMORY.__init__` per plan invariants.

## D-11 SPR-on-GPU Perf Verification

xp=numpy path: **N/A** (ABS does not complete on the numpy path).
xp=cupy path: **SKIP — no GPU available**.

Wave 1b applied D-11 design (RegisterFile follows scratchpad device — no
`device=` kwarg, xp-implicit allocation). The SPR-perf exception path
(host-pinned numpy fallback if cupy SPR access > 105s budget) is **NOT
implemented yet** because:
1. xp=numpy is the current default and SPR-on-GPU perf is moot.
2. No GPU box is available to measure the cupy-SPR access cost.
3. The exception path is an optimization, not a correctness requirement.

Decision: **DEFER the SPR-perf exception path to plan 09-03 Task 7 (BM-04
benchmarks)** once Wave 2/3 ports complete and cupy can be smoke-tested on
a GPU runner. Wave 1b's RegisterFile contract supports the exception cleanly
(its constructor takes only `shape`/`defs`/`tensor` — a future host-pinned
override can pass a pre-allocated numpy ndarray via the `tensor=` kwarg).

## Failures

### Root cause: torch API leakage across Wave-1 → Wave-2 boundary

Wave 1's memory.py / register_file.py / npu.py port produces `xp.ndarray`
(numpy) outputs. Wave 2/3 files have NOT been ported and still call torch
APIs on those outputs:

| File | Torch refs | First failure site |
|------|-----------:|-------------------|
| `src/main/python/riscv/gtx/unit/context/dma_engine.py` | 6 | line 348 `.to(l2_buf.device)` |
| `src/main/python/riscv/gtx/tloop_buffer.py` | 6 | line 423 local `import torch` + lines 468/478/486 |
| `src/main/python/riscv/gtx/unit/ins/ops/act.py` | 80 | line 24 module `import torch` |
| `src/main/python/riscv/gtx/unit/ins/ops/mm.py` | 44 | line 28 module `import torch` |
| `src/main/python/riscv/gtx/unit/ins/ops/vec.py` | 52 | line 20 module `import torch` |
| `src/main/python/riscv/gtx/unit/ins/ops/spr.py` | 1 | line 18 module `import torch` |
| `src/main/python/riscv/gtx/_verify.py` | 5 | line 9 module `import torch` |
| **Total** | **194** | Wave 2 + Wave 3 territory |

The very first torch call hit during ABS sweep replay is at dma_engine.py:348:
```python
ddr_span = mem.ddr.read(..., min(max_off, ddr_cap) - ddr_off_base).to(l2_buf.device)
```
`.to(...)` is torch-Tensor API; numpy.ndarray has no `.to()` method.

The file `dma_engine.py` also uses `.view(torch.float16)`, `.view(torch.uint8)`,
`.numel()` — all torch-only patterns.

### CONTEXT D-06 invariant analysis

CONTEXT.md D-06 states:
> "Wave 1/2 중간에는 일부 파일 numpy + 일부 torch가 임수적. 경계 함수는
>  numpy.ndarray ↔ torch.Tensor 단방향 bridge (단 임시 — Wave 3에서 모두 제거).
>  **각 wave 끝에 ABS strict byte-exact GREEN 보장 필수** (intermediate 상태도
>  invariant 유지 — 회귀가 어느 wave에서 났는지 즉시 격리 가능)."

The intent: provide bidirectional bridges at boundaries so each wave can land
incrementally. Wave 1a's memory.py port produced xp.ndarray outputs without
adding a `to_torch()` bridge or keeping a torch-output shim for downstream
consumers. As a result, the moment any vendor `.elf` exercises the dma_engine
or ops/* path, the runtime crashes.

This is a **gap in Wave 1a's design** that surfaces only at Wave 1b's gate
(the unit-level tests for memory.py and register_file.py do NOT exercise
dma_engine.py / ops/*). Wave 1a's SUMMARY acknowledged the transit risk:

> "Downstream callers in `npu.py`, `unit/register_file.py`, `unit/context/dma_engine.py`,
>  `tloop_buffer.py`, `unit/ins/ops/*.py`, and `_verify.py` still use `torch.*` and
>  will fail at runtime when they receive numpy arrays from `mem.lN_byte()`. This is
>  expected per CONTEXT D-06."

The acknowledgement is correct factually but the design DOES violate D-06's
"각 wave 끝에 ABS strict byte-exact GREEN 보장 필수" invariant. Without a
torch-bridge shim or all-at-once port, no Wave 1 gate run can be GREEN.

### Options for resolution (planner / user decision required)

**Option A — Accept the Wave 1 gate failure & defer the gate to end-of-Wave-2.**
Document that the D-07 gate is unenforceable mid-port and shift the
ABS-strict-GREEN expectation to the end of plan 09-02b-engines. Wave 1
acceptance becomes "unit-level invariants only" (the 52 unit tests).
Estimated impact: 0 lines of additional code; gate doc + STATE update only.

**Option B — Add a temporary numpy→torch bridge shim in memory.py.**
Memory accessors (`mem.lN_byte`, `mem.lN_f16`, `mem.ddr.read`) wrap their
returns in `torch.from_numpy(arr)` until Wave 2/3 ports complete. This
restores the torch-only contract for downstream files at a one-time copy
cost (no copy on the numpy path since `torch.from_numpy` shares memory).
Estimated impact: ~30 lines in memory.py; reverted in plan 09-03 cleanup.
Risk: ABS may regress from 144s to ~180s due to bridge call overhead;
strict byte-exact preserved (zero-copy).

**Option C — Pull Wave 2 forward into 09-01b.**
Port dma_engine.py + tloop_buffer.py + ops/* + _verify.py in the current
plan. Estimated impact: ~194 lines of mechanical torch→xp substitution
across 7 files; same scope as plans 09-02a + 09-02b combined. Bypasses
the gate design but produces an end-to-end Wave 1 GREEN state.

**Option D — Revert Wave 1b's register_file.py + npu.py ports.**
Hardest to recover — would lose Task 1+2 work + the deferred-store path
audit. Not recommended.

### Recommendation

**Option A** is the cleanest: it acknowledges that the Wave 1 boundary as
designed cannot satisfy D-06's smoke-gate invariant without either an
all-at-once port or a bridge shim, and shifts the gate to a point where
all torch-consuming files are ported. The next plan-level checkpoint becomes:
"end-of-Wave-2 ABS GREEN" instead of "end-of-Wave-1 ABS GREEN".

**Option B** is the cleanest if the user wants to preserve Wave-by-wave gating
as originally designed. The shim is throwaway and the 30-line cost is small.

User decision required.

## Wave 1 Sign-Off

- [x] memory.py torch-free, xp.zeros for scratchpads + DDR (Plan 09-01a)
- [x] register_file.py torch-free, SPR int64 via xp (Plan 09-01b Task 1)
- [x] register_file.py SPR-perf exception design clean (no device= kwarg; `tensor=` kwarg supports host-pinned override in plan 09-03 if needed)
- [x] npu.py constructor uses xp; line 354 `.cpu()` → `to_host()` (Plan 09-01b Task 2)
- [x] test_csr_registry_chain.py torch-free (Plan 09-01b Task 3)
- [ ] **Smoke set GREEN — FAIL (see Failures, Option A/B/C/D decision needed)**
- [/] Tile-2 GREEN — file removed pre-Phase-9 (Wave 0 finding, same status)
- [ ] **ABS walltime in 85-105s band — N/A, ABS does not complete**
- [/] D-10 verification — SKIP (no GPU available; 1 GiB workaround source-comment landed Wave 1a)
- [/] D-11 verification — partial (xp design landed; perf measurement deferred to plan 09-03 Task 7)

## Unit-Level Evidence (mitigates lost smoke gate)

Wave 1 unit tests passing (52 total):

```
tests/gtx/test_memory_layout.py        15 passed (Wave 1a)
tests/gtx/test_dma_roundtrip.py         6 passed (Wave 1a)
tests/gtx/test_register_file_xp.py     10 passed (Wave 1b Task 1)
tests/gtx/test_npu_xp.py               11 passed (Wave 1b Task 2)
tests/gtx/test_csr_registry_chain.py   10 passed (Wave 1b Task 3, ported off torch)
```

Plus dispatch chain (16 tests passed: test_custom0_smoke + test_fsm_smoke +
test_custom_dispatch_chain) confirming npu.py's RoCC entry points are still
wired correctly under xp storage.

The storage-layer port is complete and correct. The failure is purely at
the boundary where torch-consuming Wave 2/3 files meet xp-producing Wave 1
files.

---

## Next Actions

1. User/planner reviews Options A-D.
2. If Option A: amend Phase 9 plan-set to remove the Wave 1 smoke gate and
   move it to end-of-Wave-2.
3. If Option B: spawn a quick plan to add a memory.py torch-bridge shim,
   re-run gate, ship Wave 1 GREEN.
4. If Option C: amend plan 09-01b scope to include Wave 2/3 source files,
   re-execute as a single mega-plan.
5. If Option D: revert and re-plan (not recommended).
