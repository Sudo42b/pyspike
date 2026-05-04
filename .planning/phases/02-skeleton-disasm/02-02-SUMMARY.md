---
phase: 02-skeleton-disasm
plan: 02
subsystem: dispatch
tags: [spr, rocc, riscv, gtx, custom0, encoding-collision, xs1-workaround]

# Dependency graph
requires:
  - phase: 02-skeleton-disasm-01
    provides: "@handler decorator (riscv.gtx._registry), encoding constants (GTX_F7_*, GSPR_*), WarpState dataclass, MockProcessor/MockInsn fixtures"
provides:
  - "riscv.gtx.spr_router.wr_spr / rd_spr — 3-zone SPR routing (GSPR/NSPR/LSPR) verbatim port of gtx_npu_spr.cc"
  - "riscv.gtx.ops.spr — 4 @handler entries (funct7=0x00/0x01/0x48/0x49) for WRSPR/RDSPR (gem5 + ISS encodings)"
  - "tests/gtx/test_spr.py — 16 tests covering SPR-01 routing + SPR-02 handlers + CORE-04 xs1=0 workaround proof"
affects: [02-03-warp-control, 02-04-disasm, 02-05-integration, phase-04-mm, phase-03-dma]

# Tech tracking
tech-stack:
  added: []  # No new external dependencies
  patterns:
    - "Lazy import inside function body (avoids plan 02 → plan 03 circular dep): `from .ops import control as _ctrl` only at the call site for GSPR_STARTP..GSPR_ENDT loop-control side effects"
    - "Direct GPR read via state.XPR[insn.rs1] in every handler — bypasses Spike's xs1=0 → -1 marshalling sentinel (CORE-04). Pattern is now locked for plans 03/04 and Phases 3-5"
    - "D-02 collision heuristic codified: gem5 funct7=0x00/0x01 with insn.rs1 != 0 returns 0 stub (P4 firmware_mm_op placeholder); rs1==0 falls through to wr_spr/rd_spr verbatim"
    - "Handler signature contract: (npu, proc, insn, xs1, xs2) → int; npu argument bound by dispatch.build_custom0_table closure"

key-files:
  created:
    - "src/main/python/riscv/gtx/spr_router.py — wr_spr/rd_spr (110 lines, port of gtx_npu_spr.cc)"
    - "tests/gtx/test_spr.py — 16 tests (246 lines)"
  modified:
    - "src/main/python/riscv/gtx/ops/spr.py — replaced plan-01 stub (24 lines) with 4 @handler entries (89 lines)"

key-decisions:
  - "Verbatim C++ port for D-02 collision: when insn.rs1==0 and funct7=0x00, addr=XPR[0]=0 is preserved (writes GSPR_GTX_RUN as a side effect — matches C++ exactly, NOT trying to outsmart it). Open question 1 from research is documented but not resolved here."
  - "Lazy `from .ops import control` inside each loop-control branch (GSPR_STARTP..GSPR_ENDT) chosen over module-level conditional import — keeps spr_router.py importable in isolation when plan 03 is still landing."
  - "Test infrastructure choice: SimpleNamespace `_fake_npu()` instead of GtxNpu instantiation — sidesteps the `_riscv.so` requirement (Wave 0 graceful-degradation pattern carries through here)."

patterns-established:
  - "SPR storage shape contract for downstream: gspr=dict[int,int], nspr=list[dict[int,int]] (len GTX_NEST_NUM=4), lspr=list[list[dict[int,int]]] (NEST × SPU = 4 × 16). Plan 02-03 (warp/control) reads/writes these via spr_router; plan 02-05 integration test verifies via GtxNpu.reset() default values."
  - "Handler authoring convention: 1) `from .._registry import handler`; 2) `from ..spr_router import wr_spr, rd_spr` (or future `from ..mm_engine import ...`); 3) `@handler(kind='custom0', funct7=0xNN, mnemonic='name')` decorator; 4) function reads via `state = proc.get_state(); val = state.XPR[insn.rs1]`."
  - "xs1=0 workaround test proof: pass `xs1=0xFFFFFFFFFFFFFFFF` (Spike's reg_t -1) at handler call; assert XPR-derived address wins over the bogus xs1 arg. This pattern is reused in plan 02-03 for warp ops."

requirements-completed: [SPR-01, SPR-02]

# Metrics
duration: 4m52s
completed: 2026-05-04
---

# Phase 02 Plan 02: SPR Routing & WRSPR/RDSPR Handlers Summary

**3-zone SPR routing (GSPR/NSPR/LSPR per loop state) + 4 WRSPR/RDSPR handlers (gem5 funct7=0x00/0x01 with D-02 collision heuristic + ISS-full funct7=0x48/0x49) — all 16 tests pass on pure-Python mock fallback.**

## Performance

- **Duration:** 4m52s
- **Started:** 2026-05-04T08:52:37Z
- **Completed:** 2026-05-04T08:57:29Z
- **Tasks:** 3 (all auto, all TDD-flavored — verify block IS the test for T1/T2; T3 builds the persisted pytest suite)
- **Files modified:** 3 (1 modified, 2 created)

## Accomplishments

- **`riscv.gtx.spr_router.wr_spr / rd_spr`** — verbatim port of `gtx_npu_spr.cc`. 3-zone routing: GSPR (0x000-0x3FF) flat, NSPR (0x400-0x7FF) per-NEST (ploop tmu_id else NEST 0), LSPR (0x800-0xBFF) per-(NEST, SPU) (tloop targets curr_id; ploop broadcasts across 16 SPUs; fallback to (0,0) outside loop context). Loop-control GSPR addresses 0x100..0x105 forward to plan 03's `_do_*` helpers via lazy import (broken circular dep at function-call time).
- **`riscv.gtx.ops.spr`** — 4 `@handler` entries (funct7 ∈ {0x00, 0x01, 0x48, 0x49}; mnemonics `wrspr_gem5`, `rdspr_gem5`, `wrspr`, `rdspr`). All 4 read register values via `proc.get_state().XPR[insn.rs1]` to bypass xs1=0 → -1 marshalling (CORE-04). gem5 0x00/0x01 honor D-02 collision heuristic (rs1!=0 → P4 MM/MMC stub returning 0; rs1==0 → verbatim wr_spr/rd_spr).
- **`tests/gtx/test_spr.py`** — 16 tests on pure-Python mocks (no `_riscv.so` required): 7 routing tests covering all GSPR/NSPR/LSPR loop combinations + 16-bit addr mask, 7 handler tests covering both encoding paths + collision heuristic + RDSPR force-write to rd, 2 ROADMAP P2 success criterion 3 tests (WRSPR→RDSPR roundtrip in BOTH ISS and gem5 encodings).

## Task Commits

Each task was committed atomically (with `--no-verify` per Wave 1 parallel execution protocol):

1. **Task 1: spr_router.py — port of gtx_npu_spr.cc** — `9391242` (feat)
2. **Task 2: ops/spr.py — 4 WRSPR/RDSPR handlers (gem5 + ISS)** — `7eaa054` (feat)
3. **Task 3: tests/gtx/test_spr.py — 16 tests (routing + handlers + ROADMAP)** — `849e840` (test)

**Plan metadata commit:** to follow (this SUMMARY + STATE/ROADMAP updates).

## Files Created/Modified

### Created (2)
- `src/main/python/riscv/gtx/spr_router.py` — `wr_spr(npu, addr, value)` + `rd_spr(npu, addr) -> int` + `_in_range` helper. 3-zone routing + 6 loop-control side-effect hooks (lazy imports).
- `tests/gtx/test_spr.py` — 16 pytest functions. Uses `SimpleNamespace`-based `_fake_npu()` to avoid `_riscv.so` dependency. Tests gated nowhere — all 16 always run.

### Modified (1)
- `src/main/python/riscv/gtx/ops/spr.py` — replaced plan-01 stub (24-line documentation-only file) with 4 `@handler`-decorated functions: `wrspr_iss`, `rdspr_iss`, `wrspr_gem5`, `rdspr_gem5`.

## Decisions Made

1. **D-02 collision heuristic implemented exactly as plan-stated, not "improved"** — When `insn.rs1==0` and `funct7=0x00`, the C++ does `wr_spr(XPR[0] & 0xFFFF, XPR[rs2])` which means addr=0 → writes GSPR_GTX_RUN. The plan flagged research's open question 1 (this could be a C++ bug, since rs1==0 is a "marker" indicating gem5 simplified encoding, not a real address source). Per plan instruction "port verbatim", I preserved this behavior. Future P4 work that revisits gem5 encoding may revise; for now, the test `test_wrspr_gem5_rs1_zero_writes_to_addr_xpr0` codifies the verbatim contract.
2. **Lazy import inside each loop-control branch, not at module top** — This was prescribed by the plan and confirmed correct: plan 02 lands ahead of plan 03, so `from .ops import control as _ctrl` at module top would fail (control.py is still a stub). Putting the import inside each `if addr == GSPR_STARTP:` branch defers resolution to call time, by which point plan 03 will have populated `_do_*` helpers. Test suite never exercises these branches (no test calls wr_spr with addr=0x100..0x105), so the lazy imports are dormant in plan 02 — but they're proven syntactically correct because `import riscv.gtx.spr_router` succeeds.
3. **Test infrastructure: SimpleNamespace fake_npu, not GtxNpu** — `GtxNpu` requires `_riscv.so` (the `from riscv import isa` chain pulls in `_riscv`). The plan-prescribed `_fake_npu()` returns a `SimpleNamespace` with the exact 4 attributes routing/handlers need (`gspr`, `nspr`, `lspr`, `warp`). This keeps the test suite executable in any dev environment.
4. **`_FakeProc` wraps MockProcessor instead of subclassing** — A small composition helper inside the test file (not exposed elsewhere) so that handler-call sites read `proc.get_state().XPR[idx]` against the real MockXPR (which honors x0 = 0 hardwiring). This pattern is reusable in plans 02-03/02-05.

## Deviations from Plan

None — plan executed exactly as written.

All 3 task action blocks were copied verbatim into the corresponding files. All `<verify>` blocks pass on first run. All `<acceptance_criteria>` greps match the expected counts. The plan's pre-supplied code was already aligned with C++ ground-truth (I cross-checked `gtx_npu_custom0.cc:56-113` and `gtx_npu_spr.cc` to confirm the port is faithful).

**Total deviations:** 0
**Impact on plan:** None — clean execution.

## Issues Encountered

- `pytest tests/gtx/test_spr.py --collect-only` (the `rtk` proxy variant) returned "No tests collected" while `python3 -m pytest ... --collect-only` returned the proper 16-test breakdown. Root cause is the rtk proxy's optimistic short-circuiting when run with `--noconftest -o "addopts="`; pre-existing in the dev environment, not specific to this plan. Workaround documented: use `python3 -m pytest` explicitly. Mock-fallback path unaffected.
- `PYTHONPATH=src/main/python` is required to run pytest before `_riscv.so` is built. This is the same workaround Phase 1 tests use; not a new issue.

## Acceptance / Verification Snapshot

```
$ python3 -m pytest tests/gtx/test_spr.py -x -q --noconftest -o "addopts="
................                                                         [100%]
16 passed in 0.41s
```

```
$ python3 -c "from riscv.gtx._registry import _HANDLER_REGISTRY; \
    assert {0x00, 0x01, 0x48, 0x49} <= \
    {e['funct7'] for e in _HANDLER_REGISTRY \
        if e['kind']=='custom0' and e.get('funct7') is not None}; \
    print('all 4 funct7 registered')"
all 4 funct7 registered
```

```
$ grep -cE "GSPR_STARTP|GSPR_ENDP|GSPR_STARTT|GSPR_ENDT" \
    src/main/python/riscv/gtx/spr_router.py
6   # >= 4 required (loop control hooks for 6 GSPR loop addresses)

$ grep -cE "from \.ops import control" \
    src/main/python/riscv/gtx/spr_router.py
6   # >= 6 required (lazy import per loop-control branch)
```

## `_riscv.so` Availability

`_riscv.so` was NOT built during this run. The hybrid mock-fallback path carried through:
- `riscv.gtx.spr_router` and `riscv.gtx.ops.spr` are pure-Python — they import only `params.py`, `encoding.py`, `_registry.py`, all of which are pure-Python and do not depend on `_riscv`.
- Tests use `SimpleNamespace` + `MockProcessor` from `tests/gtx/_mocks.py` (Wave 0 plan 01 task 1).
- The "Missing `riscv._riscv`" `UserWarning` printed at module import is benign — pyspike's package `__init__.py` always emits it when the C++ extension isn't built. Nothing in this plan's code-paths needs `_riscv`.

When `_riscv.so` IS built (CI or `python setup.py build_ext --inplace`), the same tests still pass because: (a) handlers are decorated functions registered at module-load time (no runtime difference), (b) `MockProcessor` and the real `processor_t` expose the same `get_state().XPR[i]` / `XPR.write(i, val)` surface (D-19 mock spec). Plan 02-05 integration tests (`test_register.py`, `test_dispatch.py`) will verify the full GtxNpu path end-to-end.

## Cross-Plan Contract for Wave 1 Siblings (02-03 / 02-04)

This summary locks the API surface for plans 02-03 and 02-04 to consume:

- **02-03 (warp/control):** Will import `from ..spr_router import wr_spr, rd_spr` to read/write SPR addresses for warp loop state (e.g., `wr_spr(npu, 0x100, value)` should land in `_do_startp`). The lazy-import side-effect chain is now closed: when plan 02-03's `ops/control.py` defines `_do_startp/_do_endp/_do_starts/_do_ends/_do_startt/_do_endt`, the existing branches in `spr_router.py` will resolve them at call time. **No edits to spr_router.py needed by plan 02-03.**
- **02-04 (disasm):** Will read `_HANDLER_REGISTRY` for the 4 entries this plan added (mnemonics `wrspr`, `rdspr`, `wrspr_gem5`, `rdspr_gem5`). Disasm builder should produce `disasm_insn_t` for at least the ISS variants (`wrspr` 0x49, `rdspr` 0x48). gem5 variants may be omitted from public disasm tables since they're "internal" gem5-simplified — Claude's discretion in plan 02-04.

## Stub Tracking

- **`from .ops import control as _ctrl`** (lazy, in `spr_router.py`) — calls `_ctrl._do_startp/endp/starts/ends/startt/endt`. These are NOT yet defined in `ops/control.py` (still a stub from plan 02-01). The lazy import means: if the test never writes to GSPR 0x100..0x105, the import never fires. Plan 02-03 will populate `_do_*`, at which point any future test or runtime call hitting these branches will resolve. **Not a blocker for plan 02-02 acceptance.** Documented for plan 02-03 to wire.
- **gem5 collision rs1!=0 path returns 0 stub** — placeholder for P4 firmware_mm_op (D-02 future MM dispatch). Documented in handler docstrings. Not a stub that prevents plan 02-02's goal — the goal is to land WRSPR/RDSPR with the collision heuristic correctly classifying the two paths; the MM path will be filled in Phase 4.

## Next Phase Readiness

- **Plan 02-03 (warp/control)** unblocked — `ops/control.py` is a clean stub awaiting `_do_*` helpers. spr_router.py's lazy imports will resolve once plan 02-03 fills them.
- **Plan 02-04 (disasm)** unblocked — 4 new entries in `_HANDLER_REGISTRY` ready to produce disasm output. Plan 02-04 will iterate the registry and build `disasm_insn_t` records.
- **Plan 02-05 (integration)** unblocked — when `_riscv.so` is built, GtxNpu instantiation will trigger `ops/__init__.py → from . import spr` → 4 `@handler` decorations → `_HANDLER_REGISTRY` populated → `dispatch.build_custom0_table(self)` exposes them as `self._custom0[0x00/0x01/0x48/0x49]`. The integration test in plan 02-05 should be able to dispatch a real custom0 insn to wrspr_iss.
- **No new blockers.**

## Self-Check: PASSED

Verified files exist:
- `src/main/python/riscv/gtx/spr_router.py` — FOUND (110 lines)
- `src/main/python/riscv/gtx/ops/spr.py` — FOUND (89 lines, modified from plan-01 stub)
- `tests/gtx/test_spr.py` — FOUND (246 lines)

Verified commits exist (`git log --oneline | grep`):
- `9391242` feat(02-02): add SPR router (wr_spr/rd_spr) with GSPR/NSPR/LSPR routing — FOUND
- `7eaa054` feat(02-02): add WRSPR/RDSPR handlers (gem5 + ISS encodings) — FOUND
- `849e840` test(02-02): add SPR routing + WRSPR/RDSPR handler tests (16 tests) — FOUND

Verified all acceptance commands pass:
- Task 1 inline verify: `spr_router routing ok` printed (offline routing semantics)
- Task 2 verify: all 4 funct7 (0x00/0x01/0x48/0x49) registered, all 4 mnemonics matched
- Task 3 verify: `pytest tests/gtx/test_spr.py -x -q --noconftest -o "addopts="` → 16 passed in 0.41s
- Plan-level verify: imports of all 4 handlers + spr_router pair succeed; `_HANDLER_REGISTRY` contains the expected funct7 superset

---
*Phase: 02-skeleton-disasm*
*Completed: 2026-05-04*
