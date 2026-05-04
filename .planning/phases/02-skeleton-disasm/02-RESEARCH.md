# Phase 2: Skeleton & Disasm — Research

**Researched:** 2026-05-04
**Domain:** pyspike RoCC extension authoring + GTX NPU dispatch shell (pure-Python NumPy backend)
**Confidence:** HIGH

## Summary

Phase 2 builds the **dispatch / control shell** of the GTX NPU as a `riscv.isa.ROCC` subclass, using only Python + NumPy. The work is structural — no compute ops, no DMA — just enough plumbing for `pyspike --extlib=riscv.gtx nop_wjoin.elf` to load, hit `WJOIN`, and exit cleanly with a populated disasm trace.

Two C++ files are the unambiguous ground truth: `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc` (funct7 dispatch, every encoding case) and `gtx_npu_custom1.cc` (warp-control funct3 dispatch). The full ~140-entry disasm table is committed verbatim in `gtx_npu_disasm.inc` (244 lines). All numeric constants — funct7 opcodes, SPR address ranges, sp init value, DDR base — are already authoritative in `gtx_npu.h` and `gtx_params.h`. No interpretation needed: copy the values, port the switch-case to dict-of-handlers.

The pyspike extension API is fully verified: `ROCC` (in `riscv/isa.py`) inherits both `rocc_t` and `ISA`; `@isa.register("name")` decorator factory wraps the class and registers via `register_extension`; the trampoline (`py_rocc_t::custom0..3`) dispatches via `PYBIND11_OVERRIDE`. `examples/xhuimt/__init__.py` is the canonical RoCC example to mirror — but RoCC subclasses (unlike `ISA`) do NOT register custom RISC-V instruction descriptors via `get_instructions()` (RoCC opcodes 0x0b/0x2b/0x5b/0x7b are pre-bound by Spike). Disasm registration is via `get_disasms() -> List[disasm_insn_t]`, with `disasm_insn_t(name, match, mask, *args)` (varargs of `arg_t` instances).

**Primary recommendation:** Replicate C++ structure 1:1 — `npu.py` (the `GtxNpu` class), `dispatch.py` (custom0/custom1 dict tables built in `__init__`), `spr_router.py` (wr_spr/rd_spr matching `gtx_npu_spr.cc`), `warp_state.py` (start/end P/S/T matching `gtx_npu_loop.cc`), `disasm.py` (per-op registry + `add_r`/`add_rf3` Python helpers), `ops/spr.py` + `ops/control.py` (P2 op modules). Mock-based unit tests with `MockProcessor` and `MockInsn` in `tests/gtx/_mocks.py` (D-19/D-20). The integration test (`pyspike --extlib=riscv.gtx nop_wjoin.elf`) requires the `_riscv` build to land first; gate it with `pytest.mark.skipif(not _RISCV_AVAILABLE)`.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Dispatch table structure:**
- **D-01:** Single dict-of-handlers — `self._custom0_handlers: dict[int, Callable]` (funct7 keyed), `self._custom1_handlers: dict[int, Callable]` (funct3 keyed). Mirrors xhuimt/mylrsc.py pattern, direct port of C++ switch-case.
- **D-02:** `funct7=0x00` collision heuristic = `insn.rs1 != 0` → WRSPR (gem5 marker), else → MM/no-op fallback (DISP-01 verbatim).
- **D-03:** WRSPR/RDSPR register address extraction is Claude's discretion — verify in plan stage against C++ `gtx_npu_spr.cc`.

**Warp loop state machine + xs1=0 workaround:**
- **D-04:** `WarpState` dataclass — `is_ploop`, `is_tloop`, `is_sloop` (all `bool`, default `False`). P2 uses `is_ploop`/`is_tloop` only; `is_sloop` activated in P3+.
- **D-05:** xs1=0 workaround = decorator auto-wrap (CORE-04). All `custom0`/`custom1` handlers wrapped by a decorator that auto-replaces `xs1` with `proc.get_state().XPR[insn.rs1]` when `xs1 == 0`. Implementation detail (4-arg signature wrap) decided at plan stage.
- **D-06:** `mxe_accum` layout locked early in P2 — `np.ndarray` field, `shape=(NEST_NUM, SPU_NUM, M_TILE, N_TILE)`, `dtype=np.float32`. P4 MM-04 reads/writes; `reset()` calls `.fill(0.0)`. M_TILE/N_TILE values extracted from C++ at plan stage.

**WJOIN / SystemExit:**
- **D-07:** WJOIN reads `GTX_NO_EXIT` env var on every call (no caching). Unset/falsy → `raise SystemExit(0)`. Set/truthy → return 0.
- **D-08:** WJOIN unit test exercises both modes (default raise + `monkeypatch.setenv` continue).

**Disasm table (DISASM-01):**
- **D-09:** Per-op registry — each op module (`ops/spr.py`, `ops/control.py`, ...) provides its funct7 handlers + disasm entries together. P2: ~10 SPR/control entries; P3+ adds DMA/MM/VEC/ACT modules incrementally.
- **D-10:** New file `src/main/python/riscv/gtx/disasm.py` — separate from encoding.py, provides accumulate/query API.
- **D-11:** `encoding.py` expanded — full funct7 set (gem5 0x04-0x07 + ISS 0x00-0x7F), funct3 (custom1 start_p/end_p/start_t/end_t/wsplit/wjoin), Mode 1-4 constants.
- **D-12:** Sample 5 disasm test — adjusted in P2 to use `['wrspr','rdspr','wsplit','wjoin','start_p']` (only P2-available ops).

**Per-op registry protocol (P3-P5):**
- **D-13:** Decorator-based registry — `@gtx.handler(funct7=0x49, funct3=None, mnemonic='wrspr', mask=..., kind='custom0')`. Registers function in dispatch dict + disasm list automatically. Internal API; PY-FUNCT7-01 (v2 deferred) is separate public API.
- **D-14:** op module locations — `src/main/python/riscv/gtx/ops/spr.py` (P2), `ops/control.py` (P2 warp), `ops/dma.py` (P3), `ops/mm.py` (P4), `ops/vec.py` / `ops/act.py` (P5).

**`_riscv.so` build + submodule:**
- **D-15:** `pybind11<3.0.4` pin in `[build-system].requires` (Phase 1 deferred resolved). pyproject.toml: `"pybind11>=3,<3.0.4"`.
- **D-16:** Submodule SHA re-verification + re-init = first P2 task. `git submodule sync` + `git submodule update --init`.

**Test strategy:**
- **D-17:** Hybrid mock — `tests/gtx/conftest.py` tries `from riscv.processor import processor_t`, falls back to `MockProcessor`. Same test code runs in both environments.
- **D-18:** `tests/conftest.py` wrapped try/except for `riscv.cfg`/`riscv.sim` imports (separate P2 plan).
- **D-19:** Mock spec (P2): `MockProcessor.get_state().XPR.read(i)/write(i, val)`, `MockInsn.rs1/rs2/funct/xs1/xs2/xd`, `MockState.XPR` array. MMU added P3.
- **D-20:** Mocks live only in `tests/gtx/_mocks.py` — never exposed via `riscv.gtx._test_helpers`.
- **D-21:** `@isa.register('gtx')` validation = design contract (always) + skipif `_RISCV_AVAILABLE` for actual `register_extension`/factory.
- **D-22:** `nop_wjoin.elf` prebuilt binary committed at `tests/gtx/data/elf/nop_wjoin.elf` + source `nop_wjoin.S` for reproducibility. Package-data registration deferred to PKG-01 (P5/P6).

**Test patterns:**
- **D-23:** Warp/SPR tests call `npu.custom1(proc, insn, xs1, xs2)` directly + assert `WarpState` field values directly. No CPU step-through in unit tests.

### Claude's Discretion

- Decorator implementation details (D-05, D-13) — 4-arg signature wrap mechanics, class- vs instance-level
- `mxe_accum` shape M_TILE/N_TILE values (D-06)
- WRSPR/RDSPR rs1/rs2 semantics (D-03)
- Mock class method signatures (D-19) matching real `riscv.processor` API
- xs1=0 decorator vs helper trade-off (P4 hot-path measurement)
- ELF build script + source (`nop_wjoin.S`, Makefile vs single command)
- `test_disasm_table.py` sample list (D-12) — only P2-available ops

### Deferred Ideas (OUT OF SCOPE)

- ROADMAP/REQUIREMENTS sync commit (D-09 success criterion 2 wording)
- v2 milestones: PY-FUNCT7-01, CYC-01/02, MEXEC-01
- Other phases: DMA (P3), MM (P4), VEC/ACT (P5), verify.py (P6), PKG-01 (P5/P6), DISP-03 (P3+)
- pybind11<3.0.4 pin impact on cibuildwheel matrix — verify in first P2 plan
- Submodule SHA push verification (D-16) — first P2 task

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **CORE-01** | `riscv.isa.ROCC` subclass `GtxNpu` + `@isa.register("gtx")` so `pyspike --extlib=riscv.gtx` loads it | `examples/xhuimt/__init__.py` is canonical pattern; `riscv/isa.py:42-70` shows ROCC base + register decorator. `_riscv` exposes `rocc_t` with `custom0..3` overrides via `PYBIND11_OVERRIDE` (`riscv_extension.cc:90-108`). |
| **CORE-02** | `reset()`: `XPR.write(2, 0x80100000)` sp init + `mxe_accum`/SPR/L0/L1/L2 zero-init | C++ `gtx_npu_core.cc:144-189` shows exact sp value. `state.XPR.write(idx, val)` exposed in `py_module.cc:619-622` (`xpr_regfile_t::write`). FPU enable (mstatus.FS=01) is C++-only — Python equivalent via `proc.put_csr(0x300, ...)` (`py_module.cc:706`). |
| **CORE-03** | WJOIN raises `SystemExit(0)` if `GTX_NO_EXIT` unset | C++ uses `exit(0)` after dump; Python uses `raise SystemExit(0)` (CONTEXT.md D-07). Trigger location: custom1 funct3=0b101 (JOIN). |
| **CORE-04** | xs1=0 workaround — `proc.get_state().XPR[insn.rs1]` direct read | Confirmed in C++: `gtx_npu_custom0.cc:49-50` reads `p->get_state()->XPR[rs1_num]` directly. Spike's `define_custom_func` macro marshals -1 when `xs1==0` (rocc.h:42). Decorator auto-wrap (D-05). |
| **SPR-01** | GSPR (0x000-0x3FF) / NSPR (0x400-0x7FF) / LSPR (0x800-0xBFF) routing in `wr_spr` / `rd_spr` | Address ranges in `gtx_params.h:29-34`. Routing logic in `gtx_npu_spr.cc:16-107` — port verbatim. P2 minimum: route by range; full NEST/SPU broadcast (lines 42-52) deferred to P3+. |
| **SPR-02** | WRSPR (gem5 `funct7=0x00`, `xs1=xs2=1`) + RDSPR (ISS-full `funct7=0x49`) writeback | gem5 path: `gtx_npu_custom0.cc:56-72`. ISS-full path: `gtx_npu_custom0.cc:96-113` (RDSPR=0x48, WRSPR=0x49). RDSPR explicitly writes to rd via `state->XPR.write(insn.rd, val)` when `xd=0` (the ISS encoding case). |
| **DISASM-01** | `disasm_insn_t` list returned by `get_disasms()` (P2: ~10 entries) | `disasm_insn_t(name, match, mask, *args)` constructor in `riscv_disasm.cc:29-37`. Match formula: `(funct7 << 25) \| 0x0b` for custom0; mask `(0x7f << 25) \| 0x7f`. funct3 sub-variants add `(funct3 << 12)` to match and `(0x7 << 12)` to mask (gtx_npu_disasm.inc:23-36). |
| **DISP-01** | custom0 funct7 dispatch dict + funct7=0x00 collision heuristic | Full table in `gtx_npu_custom0.cc:56-823`. P2 wires SPR control + WJOIN; later phases register more handlers via decorator. |
| **DISP-02** | custom1 funct3 warp control dispatch + P/S/T state machine | `gtx_npu_custom1.cc:43-136` — 8 funct3 cases. P2 implements all 8 (only `start_p`/`end_p`/`start_t`/`end_t`/`wsplit`/`wjoin` exercised; `start_s`/`end_s` is no-op stub for P3 DMA). |

## Project Constraints (from CLAUDE.md)

- **Pure Python only.** No new C++ code. Performance hot spots reconsidered in v2 (cython/C extension).
- **`riscv.isa.ROCC` virtual signature must be exact:** `custom0/1/2/3(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`. `processor_t` and `rocc_insn_t` are pybind11 binding objects.
- **NumPy ≥ 2.0** (Phase 1 D-07). Confirmed installed: 2.2.6 in dev environment.
- **`requires-python = ">=3.10"`** — cp310/cp311/cp312 only. Confirmed: Python 3.10.12 in dev.
- **No new runtime deps** beyond NumPy. Validation tools (libgtx_npu.so) used in dev only.
- **Bit-exact ULP target** — but P2 is structural; bit-exact testing kicks in P4+.
- **manylinux2014_x86_64 / glibc 2.17+** — dev env Linux x86_64 confirmed. Don't break cibuildwheel.
- **GSD workflow enforcement** — Phase 2 file edits flow through `/gsd:execute-phase`.
- **Style: max line 120, type hints required, docstrings optional** (Phase 1 established).
- **Module naming: lowercase + underscore** (Phase 1 established).
- **TDD workflow established in Phase 1** — `test_*.py` RED → module GREEN → refactor.

## Standard Stack

### Core (already in repo — Phase 1)

| Library / Module | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| NumPy | 2.2.6 (verified `pip show`) | All memory backed by `np.uint8` ndarrays + halfword views (Phase 1 D-09/D-10) | Sole runtime dep. P2 only uses for `mxe_accum` zero-init. |
| Python `dataclasses` | stdlib | `WarpState` (D-04) | Mutable, named fields, ergonomic for pytest assertions. |
| `os.environ` | stdlib | `GTX_NO_EXIT` read in WJOIN | Per-call read (D-07). `monkeypatch.setenv` test-friendly. |
| `riscv.isa.ROCC` | from `_riscv` (pyspike's pybind11 binding) | Base class for `GtxNpu` | Verified at `src/main/python/riscv/isa.py:42`. Inherits `rocc_t` (C++) + `ISA`. |
| `@riscv.isa.register("gtx")` | from `_riscv` | Auto-register factory | `riscv/isa.py:48-70`. Wraps class with hardcoded `name` property + `register_extension`. |
| `riscv.disasm.disasm_insn_t` | from `_riscv` | Disasm entries returned by `get_disasms()` | `py_module.cc:350-357`. Constructor: `disasm_insn_t(name, match, mask, *args)` with varargs of `arg_t`. |
| `riscv.processor.processor_t` | from `_riscv` | First arg to `custom0/1` | `py_module.cc:697-741`. Has `state` property → `state_t.XPR` (`xpr_regfile_t` with `__getitem__` and `write(i, val)`). |
| `riscv.extension.rocc_insn_t` | from `_riscv` | Second arg to `custom0/1` | `py_module.cc:391-409`. Read-only properties: `opcode`, `rd`, `xs2`, `xs1`, `xd`, `rs1`, `rs2`, `funct`. |
| `pytest` | 9.0.1 (verified) | Test framework | Phase 1 established. `monkeypatch` for env vars; `pytest.raises(SystemExit)` for WJOIN. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy.testing` | with NumPy | `assert_array_equal` for `mxe_accum` zero-init | reset() unit test |
| `riscv.disasm.arg_t` (subclass via `@isa.arg`) | from `_riscv` | Custom operand formatters | Optional in P2 — `disasm_insn_t` accepts `arg_t` instances; default is to omit args (mnemonic-only output). The C++ ref uses `gtx_xrd`/`gtx_xrs1`/`gtx_xrs2` (formatters showing register names). |
| `riscv.csrs.csr_t` | from `_riscv` | `get_csrs()` return type | P2 returns `[]` for now — SPRs are NOT exposed as CSRs (Phase 1 STATE.md "No CSR exposure for SPRs"). |
| `riscv.processor.insn_desc_t` | from `_riscv` | `get_instructions()` return type | P2 returns `[]` — RoCC opcodes 0x0b/0x2b/0x5b/0x7b are pre-bound by Spike. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dict-of-handlers | `match` statement (PEP 634) | cp310+ baseline allows match, but Phase 1 STATE.md explicitly recommends dict for consistency with project pattern. Locked: dict (D-01). |
| dataclass `WarpState` | bit flags in single `int` | C++ uses 3 bool flags. dataclass is clearer + asserts trivial (D-04 locked). |
| Per-op registry decorator (D-13) | central dispatch in `npu.py` | Locked: per-op modules with co-located handler+disasm. Better cohesion + future-proof for P3-P5. |
| Mock fallback (D-17) | only run tests when `_riscv` builds | Locked: hybrid for local dev velocity. Production wheel never sees mocks. |

**Installation (no new deps required):**
```bash
# Already installed via Phase 1
# numpy>=2.0,<3 — verified 2.2.6 in dev
# pytest — verified 9.0.1
# Phase 2 adds NO new runtime/test deps.
```

**Version verification:**
- `numpy 2.2.6` (current) — verified `python3 -c "import numpy; print(numpy.__version__)"` → 2.2.6
- `pytest 9.0.1` (current) — verified
- `riscv64-unknown-elf-gcc 15.2.0` (toolchain at `/opt/riscv/`) — verified for `nop_wjoin.elf` build

## Architecture Patterns

### Recommended Project Structure

```
src/main/python/riscv/gtx/
├── __init__.py            # P1: existing — extend to re-export GtxNpu (D-14)
├── params.py              # P1: existing — HW constants
├── encoding.py            # P1: existing — extend per D-11 (full funct7/funct3/Mode set)
├── fp.py                  # P1: existing — untouched in P2
├── memory.py              # P1: existing — GtxNpu holds GtxMemory()
├── ddr.py                 # P1: existing — untouched in P2
├── npu.py                 # P2: NEW — GtxNpu class (@isa.register("gtx"))
├── warp_state.py          # P2: NEW — WarpState dataclass (D-04)
├── spr_router.py          # P2: NEW — wr_spr/rd_spr (port of gtx_npu_spr.cc)
├── dispatch.py            # P2: NEW — build_custom0_table, build_custom1_table
├── disasm.py              # P2: NEW — per-op registry + helpers (D-10)
├── _registry.py           # P2: NEW — internal handler decorator (D-13)
└── ops/
    ├── __init__.py        # P1: existing — package marker
    ├── spr.py             # P2: NEW — WRSPR/RDSPR handlers + disasm entries
    └── control.py         # P2: NEW — WSPLIT/WJOIN/start_p/end_p/start_t/end_t

tests/gtx/
├── __init__.py            # P1: existing
├── conftest.py            # P2: NEW (D-17 hybrid mock fallback)
├── _mocks.py              # P2: NEW (D-19/D-20 mock spec, internal-only)
├── test_register.py       # P2: NEW (D-21 register validation)
├── test_spr.py            # P2: NEW (SPR-01/02 — WRSPR/RDSPR roundtrip)
├── test_warp.py           # P2: NEW (DISP-02 — start_p/end_p/start_t/end_t)
├── test_dispatch.py       # P2: NEW (DISP-01 — funct7=0x00 collision)
├── test_disasm.py         # P2: NEW (DISASM-01 — get_disasms structure + sample)
├── test_wjoin.py          # P2: NEW (CORE-03 — SystemExit + GTX_NO_EXIT)
├── test_reset.py          # P2: NEW (CORE-02 — sp + zero-init + mxe_accum)
├── test_xs1_zero.py       # P2: NEW (CORE-04 — decorator workaround)
├── test_nop_elf.py        # P2: NEW (CORE-01 integration — skipif _RISCV)
└── data/
    └── elf/
        ├── nop_wjoin.S    # P2: NEW (assembly source)
        ├── nop_wjoin.elf  # P2: NEW (prebuilt; ~1KB; D-22)
        └── Makefile       # P2: NEW (build instructions for repro)
```

### Pattern 1: ROCC subclass skeleton

**What:** Class inherits from `isa.ROCC` (which inherits both `rocc_t` and `ISA`). Decorator `@isa.register("gtx")` wraps with hardcoded `name` and registers.

**When to use:** Always, for the GtxNpu top-level class. Mirror `examples/xhuimt/__init__.py:27-55`.

**Example:**
```python
# Source: examples/xhuimt/__init__.py:24-55 + riscv/isa.py:42-70
from typing import List
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

@isa.register("gtx")
class GtxNpu(isa.ROCC):
    """GTX NPU functional model — Phase 2 dispatch shell."""

    def __init__(self):
        super().__init__()
        # ... GtxMemory, WarpState, mxe_accum, dispatch tables

    def get_instructions(self, proc: processor_t) -> List[insn_desc_t]:
        return []  # RoCC opcodes 0x0b/0x2b are pre-bound by Spike

    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]:
        return list(self._disasm_entries)  # accumulated by per-op registry

    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []  # SPRs are not CSRs (STATE.md decision)

    def reset(self, proc: processor_t) -> None:
        super().reset(proc)
        proc.get_state().XPR.write(2, 0x80100000)  # sp init
        # FPU enable: mstatus.FS = Initial (0x2000)
        # ... mxe_accum.fill(0.0), mem zero-init, SPR clear

    def custom0(self, proc, insn, xs1, xs2) -> int:
        funct7 = insn.funct
        handler = self._custom0.get(funct7)
        if handler is None:
            return 0  # unknown funct7 — silent NOP for now
        return handler(proc, insn, xs1, xs2)

    def custom1(self, proc, insn, xs1, xs2) -> int:
        funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
        handler = self._custom1.get(funct3)
        if handler is None:
            return 0
        return handler(proc, insn, xs1, xs2)
```

### Pattern 2: Reading rs1/rs2 directly via processor state (CORE-04)

**What:** Spike marshals `xs1=0` as `-1` to `custom0/custom1`. C++ workaround reads register values directly via `proc.get_state().XPR[insn.rs1]`.

**When to use:** Anywhere a handler needs the actual rs1/rs2 value (always, for SPR/warp/MM/firmware ops). The decorator (D-05) can wrap this transparently.

**Example:**
```python
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc:49-50, 99-110
def _wrspr_handler(proc, insn, xs1, xs2):
    # Direct register read — bypasses xs1/xs2 marshalling
    val_rs1 = proc.get_state().XPR[insn.rs1]
    val_rs2 = proc.get_state().XPR[insn.rs2]
    addr = val_rs1 & 0xFFFF
    npu.spr_router.wr_spr(addr, val_rs2)
    return 0

# Decorator-wrap variant (D-05):
def gpr_safe(handler):
    """Auto-replace xs1/xs2 with proc.get_state().XPR[insn.rs1/rs2] when xs* == 0."""
    def wrapped(proc, insn, xs1, xs2):
        state = proc.get_state()
        if xs1 == 0 or xs1 == (1 << 64) - 1:  # spike marshals as -1
            xs1 = state.XPR[insn.rs1]
        if xs2 == 0 or xs2 == (1 << 64) - 1:
            xs2 = state.XPR[insn.rs2]
        return handler(proc, insn, xs1, xs2)
    return wrapped
```

### Pattern 3: disasm_insn_t registration

**What:** `disasm_insn_t(name, match, mask, *args)` returns one entry. `get_disasms()` returns a `List[disasm_insn_t]`.

**When to use:** Once per recognizable mnemonic. P2 registers ~10; per-op modules add more in P3+.

**Example:**
```python
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc:23-36 (helpers ported to Python)
# C++:
#   auto match = (funct7 << 25) | 0x0b;
#   auto mask  = (0x7fU << 25) | 0x7f;
#   insns.push_back(new disasm_insn_t(name, match, mask, {&gtx_xrd, &gtx_xrs1, &gtx_xrs2}));
from riscv.disasm import disasm_insn_t, xpr_name
from riscv import isa as _isa

CUSTOM0_OPCODE = 0x0b
CUSTOM1_OPCODE = 0x2b

@_isa.arg
def gtx_xrd(insn): return xpr_name[insn.rd]
@_isa.arg
def gtx_xrs1(insn): return xpr_name[insn.rs1]
@_isa.arg
def gtx_xrs2(insn): return xpr_name[insn.rs2]

def add_r_custom0(name: str, funct7: int) -> disasm_insn_t:
    """R-type custom0: match on funct7+opcode only."""
    match = (funct7 << 25) | CUSTOM0_OPCODE
    mask = (0x7f << 25) | 0x7f
    return disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)

def add_rf3_custom0(name: str, funct7: int, funct3: int) -> disasm_insn_t:
    """R-type custom0 with funct3 sub-variant."""
    match = (funct7 << 25) | (funct3 << 12) | CUSTOM0_OPCODE
    mask = (0x7f << 25) | (0x7 << 12) | 0x7f
    return disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)

def add_warp(name: str, funct3: int) -> disasm_insn_t:
    """custom1 warp control: match on funct3+opcode."""
    match = (funct3 << 12) | CUSTOM1_OPCODE
    mask = (0x7 << 12) | 0x7f
    return disasm_insn_t(name, match, mask, gtx_xrd, gtx_xrs1, gtx_xrs2)
```

### Anti-Patterns to Avoid

- **Custom dispatch via `if/elif` chains** instead of dict — locked against (D-01). xhuimt mylrsc uses different patterns; we use dict.
- **Reading `proc.state.XPR[i]` then writing to a stale local** — XPR is a live regfile. Always pass values back via `proc.get_state().XPR.write(i, val)` (verified at `py_module.cc:619-622`).
- **Importing `from . import npu` in `riscv/gtx/__init__.py`** — heavy import. STATE.md established lazy import in P1. P2 D-14 says `__init__.py` re-exports `GtxNpu`; do this with deferred import (`from .npu import GtxNpu` is OK at module level but costs ~50ms+ NumPy import; consider `__getattr__` lazy).
- **Subclassing `extension_t` instead of `rocc_t` for GtxNpu** — `extension_t` only registers RISC-V instructions via `get_instructions()`. RoCC opcode 0x0b/0x2b dispatch is owned by Spike core; only `rocc_t` subclasses receive `custom0/1/2/3` calls.
- **Returning `disasm_insn_t` instances from `get_disasms()` without keeping references alive** — `disasm_insn_t` is tracked via `PythonBridge` (`riscv_extension.cc:39-54`). Store in `self._disasm_entries: List[disasm_insn_t]` so they outlive the function call.
- **Accessing `.illegal_instruction` from `ROCC`** — `py_extension_t::illegal_instruction` is on `extension_t` (`py_module.cc:424-425`). It IS reachable through `self.illegal_instruction(proc)` since `ROCC` inherits from `rocc_t : extension_t`. But P2 should NOT raise illegal_instruction for unknown funct7 (return 0 silently — handlers add up incrementally per phase). Reserve `illegal_instruction` for P5/P6 strict mode.
- **Reading `funct3` from `insn.funct`** — `insn.funct` is funct7 (bits 31:25). funct3 is reconstructed as `(insn.xd << 2) | (insn.xs1 << 1) | insn.xs2`. C++ does this verbatim at `gtx_npu_custom0.cc:45` and `gtx_npu_custom1.cc:35`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| custom0 funct7 dispatch | switch-style if/elif | `dict[int, Callable]` (D-01) | Hash lookup is O(1), trivially extensible by op modules. C++ uses `switch` because no Python-equivalent dispatch exists in C; Python has dict native. |
| Spike-side RoCC opcode 0x0b binding | Manual `register_base_insn` | Inherit from `isa.ROCC` and Spike auto-binds | `_riscv` already binds opcodes 0x0b/0x2b/0x5b/0x7b → custom0/1/2/3 via `define_custom_func` macro (`vendor/spike/riscv/rocc.h:35-50`). Subclassing `rocc_t` is sufficient. |
| Disasm match/mask formula | Hand-compute hex | Helper `add_r(name, funct7)` / `add_rf3(name, funct7, funct3)` | Each formula is a one-liner: `(funct7 << 25) \| 0x0b` etc. The C++ `add_r`/`add_rf3` lambdas at `gtx_npu_disasm.inc:23-36` are already validated; port verbatim. |
| Register decorator | Re-implement `register_extension` | Use `@riscv.isa.register("gtx")` | `riscv/isa.py:48-70` already wraps + calls `register_extension`. |
| `arg_t` formatter | Subclass `arg_t` directly | `@riscv.isa.arg` decorator | `riscv/isa.py:73-86` wraps callable as `arg_t` subclass. Identical to `gtx_xrd` C++ struct. |
| Mock processor | Real Spike step-through | `MockProcessor` with `get_state().XPR.read/write` (D-19) | Phase 2 is structural; CPU-step integration only needed for `nop_wjoin.elf` test. Unit tests run faster + work without `_riscv.so`. |
| ELF loader | Hand-write ELF parser | RISC-V cross-toolchain produces standard ELF — let Spike load it | `/opt/riscv/bin/riscv64-unknown-elf-gcc` (verified GCC 15.2.0) produces ELF that `pyspike --extlib=` consumes. |
| WarpState bit flags | Pack into one int | `@dataclass class WarpState` (D-04) | dataclass has `__init__`, `__eq__`, `__repr__`, default values for free. Pytest assertion `assert npu.warp.is_ploop` is trivial. |
| funct7 collision resolution | New funct7 invented | `insn.rs1 != 0` heuristic (D-02) | C++ reference `gtx_npu_custom0.cc:60-72` does exactly this. Locked. |

**Key insight:** The C++ reference is the entire dispatch logic, already debugged and matching SystemC golden. P2 is a 1:1 port — every funct7 case, every SPR address, every WJOIN side-effect is in `vendor/gtx_cpp_reference/gtx/`. Don't invent; port.

## SPR Encoding & Routing Reference (HIGH confidence — from C++ source)

### SPR Address Layout (gtx_params.h:29-67)

| Range | Type | Routing scope |
|-------|------|---------------|
| 0x000-0x3FF | GSPR | Global (single-instance NPU state) |
| 0x100-0x105 | GSPR loop control | `wr_spr` to these calls `startp/endp/starts/ends/startt/endt` directly (not stored as SPR value) |
| 0x400-0x7FF | NSPR | Per-NEST (4 NESTs total) — routed by `tmu_id` if `is_ploop` else NEST 0 |
| 0x800-0xBFF | LSPR | Per-SPU per-NEST (16 SPUs × 4 NESTs) — routed by `(tmu_id, curr_id)` if `is_tloop`, broadcast across NEST if `is_ploop`, else (0,0) |

### Named SPR addresses — at minimum required by P2

| Name | Address | Type | P2 use |
|------|---------|------|--------|
| `GSPR_GTX_RUN` | 0x000 | GSPR | reset() init to 0 |
| `GSPR_GTX_OPERAND1` | 0x001 | GSPR | reset() init to 0 |
| `GSPR_GTX_OPERAND2` | 0x002 | GSPR | reset() init to 0 |
| `GSPR_GTX_OPERAND3` | 0x003 | GSPR | reset() init to 0 (also rs3 staging in OPSET) |
| `GSPR_GTX_OPCODE` | 0x004 | GSPR | reset() init to 0 |
| `GSPR_STACK_INFO` | 0x010 | GSPR | reset() init to 0 |
| `GSPR_STACK_SAVE` | 0x011 | GSPR | reset() init to 0 |
| `GSPR_STARTP/ENDP/STARTS/ENDS/STARTT/ENDT` | 0x100-0x105 | GSPR | wr_spr triggers loop-control side-effect |
| `NSPR_THREAD_MASK` | 0x400 | NSPR | reset() init to 0xFFFF (all SPUs active) |
| `NSPR_SHARED_MASK` | 0x401 | NSPR | reset() 0 |
| `NSPR_TYPE` | 0x402 | NSPR | reset() 1 (FP16 default) |
| `NSPR_OP_MODE` | 0x403 | NSPR | reset() 0 |
| `NSPR_CLEAR` | 0x700 | NSPR | reset() 0 |
| `NSPR_SDLE_STATUS` | 0x780 | NSPR | reset() 0 |
| `NSPR_CREDIT_COUNT` | 0x781 | NSPR | reset() 0 |
| `NSPR_CREDIT_ERROR` | 0x782 | NSPR | reset() 0 |
| `LSPR_SPM_ADDRA` | 0x900 | LSPR | reset() 0 (REQUIRED for ROADMAP P2 success criterion 3) |
| `LSPR_SPM_ADDRB` | 0x901 | LSPR | reset() 0 |
| `LSPR_SPM_ADDRC` | 0x902 | LSPR | reset() 0 |
| `LSPR_SPM_ADDRR` | 0x903 | LSPR | reset() 0 |

### SPR storage shape — recommendation

Phase 1 already locked **single dict pattern** (D-11): `mem.spr: dict[int, int]`. This works for GSPR but does NOT model NEST/SPU routing. Two options:

**Option A (minimal, P2):** Keep `mem.spr: dict[int, int]` as a flat dict. Encode NEST/SPU into the key for NSPR/LSPR routing (e.g., `mem.spr[(0x900, nest, spu)] = val`). **Problem:** Phase 1 tests directly do `mem.spr[0x900] = 0xF00D` — would break.

**Option B (recommended):** Layered SPR — `mem.spr` for GSPR, `mem.nspr[nest]: dict` and `mem.lspr[nest][spu]: dict` introduced in P2. This matches C++ `gtx_npu_t::gspr` + `nests[n].nspr` + `nests[n].spus[s].lspr` (gtx_npu.h:757) verbatim.

**Decision recommendation for plan stage:** Option B. Add `nspr: List[dict[int,int]]` (length GTX_NEST_NUM) and `lspr: List[List[dict[int,int]]]` (NEST × SPU) to `GtxMemory` in a backwards-compatible way (Phase 1 single-dict tests still pass — they exercised `mem.spr[0x900]` which becomes a routing call to LSPR(0,0) by default when `is_tloop=False`). Alternative: keep single dict at the GtxMemory layer, but route via `spr_router.wr_spr` which maintains separate dicts internally. The router in C++ is the **only** place address-range → physical-storage mapping happens; making this a router function (not a memory-layer field) is cleaner.

**Strong recommendation:** Add **`spr_router.py`** with module-level `gspr: dict`, `nspr: list[dict]`, `lspr: list[list[dict]]` owned BY THE GtxNpu instance (passed as `npu.gspr`, `npu.nspr`, `npu.lspr`), not by GtxMemory. GtxMemory.spr can stay as a Phase 1 placeholder (or be removed in P2 if no one reads it). This keeps Phase 1 tests stable and matches C++ ownership.

### gem5-simplified vs ISS-full encoding (verified from gtx_npu.h:265-282 + custom0.cc:56-113)

| Encoding | funct7 | RoCC bits | Behavior |
|----------|--------|-----------|----------|
| gem5 WRSPR | 0x00 | xs1=1, xs2=1, xd=0 | Spike passes rs1/rs2 register values as args; `funct3 = 011 = 3` |
| gem5 RDSPR | 0x01 | xs1=1, xs2=0, xd=1 | Spike passes rs1 only; result written to rd; `funct3 = 110 = 6` |
| ISS WRSPR | 0x49 | xs1=0, xs2=0, xd=0 | Spike passes -1 args; **MUST** read rs1/rs2 directly via `XPR[insn.rs1]` |
| ISS RDSPR | 0x48 | xs1=0, xs2=0, xd=0 | Same — direct read; result written to rd via `XPR.write(insn.rd, val)` even if `xd=0` |

**Key:** ISS-full encodings (0x48/0x49) use the xs1=0 workaround. gem5-simplified encodings (0x00/0x01) get xs1/xs2 marshalled normally — but the C++ code still uses `p->get_state()->XPR[rs1_num]` for them too (line 49-50), so the decorator can apply uniformly without breaking either encoding.

### funct7=0x00 collision resolution (DISP-01, locked D-02)

`gtx_npu_custom0.cc:56-72`:
```cpp
case GTX_F7_WRSPR:   // funct7=0
    if (insn.rs1 != 0) {
        return firmware_mm_op(p, insn, /*is_accumulate=*/false);  // → P4
    }
    wr_spr(static_cast<uint16_t>(val_rs1 & 0xFFFF), val_rs2);
    return 0;
```

In **P2**, MM is not implemented. Recommendation: handler returns 0 (no-op) on `insn.rs1 != 0` path, with a `# P4: firmware_mm_op` TODO comment. Same for funct7=0x01 (RDSPR / MMC).

## custom0 funct7 dispatch table — concrete (HIGH confidence)

Sourced from `vendor/gtx_cpp_reference/gtx/gtx_npu.h:266-353` and `gtx_npu_custom0.cc:56-823`.

### Phase 2 wires (~10 funct7 entries — concrete handlers)

| funct7 | C++ constant | Mnemonic | P2 action |
|--------|--------------|----------|-----------|
| 0x00 | `GTX_F7_WRSPR` | wrspr (gem5) | If `insn.rs1==0`: call `wr_spr(val_rs1 & 0xFFFF, val_rs2)`; else NOP (P4 MM stub) |
| 0x01 | `GTX_F7_RDSPR` | rdspr (gem5) | If `insn.rs1==0`: return `rd_spr(val_rs1 & 0xFFFF)`; else NOP (P4 MMC stub) |
| 0x02 | `GTX_F7_WSPLIT` | wsplit | NOP in P2 (timing — `wsplit_cycle = 0; wsplit_seen = True`) |
| 0x03 | `GTX_F7_WJOIN` | wjoin (custom0 variant!) | **WJOIN exit semantics** — see Section "WJOIN + GTX_NO_EXIT semantics" below |
| 0x04-0x07 | `GTX_F7_DISPATCH_*` | dispatch_mm/vec/act/dma | NOP stubs (P3-P5 fill) |
| 0x48 | `GTX_ISS_F7_RDSPR_ISS` | rdspr (ISS) | Read rs1 via XPR, return `rd_spr(addr & 0xFFFF)`, write to rd via XPR.write(insn.rd, val) |
| 0x49 | `GTX_ISS_F7_WRSPR_ISS` | wrspr (ISS) | Read rs1, rs2 via XPR; call `wr_spr(addr & 0xFFFF, val)`; return 0 |

**Note:** WJOIN appears in BOTH custom0 (funct7=0x03) AND custom1 (funct3=0b101) in the C++ reference. The custom0 path (`gtx_npu_custom0.cc:79-81`) is the **simple firmware case** — just returns elapsed cycles, NO env var check, NO `exit(0)`. The custom1 path (`gtx_npu_custom1.cc:65-122`) is the **rich case** with L1/L2 dump and is the one referenced by ROADMAP P2 success criterion 5 + CORE-03. Confirmed: only custom1 should `raise SystemExit`.

### Out of scope for P2 (handlers added by later phases)

| funct7 range | Op family | Phase |
|--------------|-----------|-------|
| 0x00-0x01 (rs1!=0 path) | MM/MMC variants | P4 |
| 0x08-0x09 | IM2COL_N/D | P5 |
| 0x10-0x1F | Scalar/vector arith, FMADD, MIN/MAX, DOT/SUM, math, sign, round, clamp | P5 |
| 0x20-0x25 | Format conversion (FP8/INT8/FP32/FP64) | P5 |
| 0x28-0x2F | Activation (PRELU/GELU/TANH/SIGM/SOFTMAX/ESUM) | P5 |
| 0x30-0x31 | Pooling (max/avg) | P5 |
| 0x38-0x39 | Transpose, fill | P5 |
| 0x40-0x45 | DMA (load/store/copy/SVR/multicast) | P3 |
| 0x4A-0x4C | OPSET, CPSVR, MVSVR | P3 (OPSET is needed for DMA staging) |
| 0x50-0x53 | Credit (LD/ST/LD_CHK/ST_CHK) | P3 |
| 0x54-0x5D | _IMM variants (L0 paths) | P5 |
| 0x70-0x7F | mexec/mbar/msync/eom/bar/wait/intr/flush/halt | v2 (mostly NOPs in functional model) |
| 0x7D, 0x7E | debug_wr, debug_rd | tests-only (P5/P6) |

## custom1 funct3 dispatch table — concrete (HIGH confidence, full table)

Sourced verbatim from `vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc:43-136`.

| funct3 (binary) | funct3 (dec) | Mnemonic | C++ handler | P2 action |
|-----------------|--------------|----------|-------------|-----------|
| 0b000 | 0 | START_T (warp_start_t) | `startt(rs1, rs2)` | **Implement** — set `is_tloop=True`, `curr_id=spu_id` |
| 0b001 | 1 | END_T (warp_end_t) | `endt(rs1, rs2)` | **Implement** — clear `is_tloop` |
| 0b010 | 2 | START_S (warp_start_s) | `starts(rs1, rs2)` | Stub (NOP — `is_sloop=True` only used by P3 DMA) |
| 0b011 | 3 | END_S (warp_end_s) | `ends(rs1, rs2)` | Stub (NOP) |
| 0b100 | 4 | SPLIT (warp_split / wsplit) | record `wsplit_cycle = total_npu_cycles` | NOP — timing; P2 doesn't track cycles |
| 0b101 | 5 | JOIN (warp_join / wjoin) | dump L1/L2/DDR, return elapsed | **`raise SystemExit(0)` if `GTX_NO_EXIT` unset, else return 0** |
| 0b110 | 6 | START_P (warp_start_p) | `startp(rs1, rs2)` | **Implement** — set `is_ploop=True`, `tmu_id=nest_id` |
| 0b111 | 7 | END_P (warp_end_p) | `endp(rs1, rs2)` | **Implement** — clear `is_ploop` |

### Loop ID extraction formula (from gtx_npu_loop.cc:21-23)

```python
# C++ pattern repeated in startp/endp/starts/ends/startt/endt
def extract_id(rs1: int, rs2: int) -> int:
    return (rs2 & 0x3F) if (rs2 & 0x400) else (rs1 & 0xFFFFFFFF)
```

This dual-mode addressing (rs1 path OR rs2 marker bit) is preserved across all 6 loop ops. P2 must implement it correctly even if some firmware uses only the rs1 path.

### Concrete dispatch builder example

```python
# src/main/python/riscv/gtx/dispatch.py — recommended skeleton
from typing import Callable, Dict
from .ops import spr as _spr_ops
from .ops import control as _ctrl_ops

def build_custom0_table(npu) -> Dict[int, Callable]:
    """Build funct7 → handler dict. Per-op modules contribute via decorator."""
    return {
        0x00: _spr_ops.wrspr_gem5,  # collision-aware (rs1!=0 → MM stub)
        0x01: _spr_ops.rdspr_gem5,
        0x02: _ctrl_ops.wsplit_custom0,
        0x03: _ctrl_ops.wjoin_custom0,  # NO exit (custom1 variant has exit)
        0x04: _ctrl_ops.dispatch_mm_stub,
        0x05: _ctrl_ops.dispatch_vec_stub,
        0x06: _ctrl_ops.dispatch_act_stub,
        0x07: _ctrl_ops.dispatch_dma_stub,
        0x48: _spr_ops.rdspr_iss,
        0x49: _spr_ops.wrspr_iss,
        # P3-P5 add more via @gtx.handler decorator
    }

def build_custom1_table(npu) -> Dict[int, Callable]:
    """Build funct3 → handler dict for warp control."""
    return {
        0b000: _ctrl_ops.startt,
        0b001: _ctrl_ops.endt,
        0b010: _ctrl_ops.starts,  # NOP in P2
        0b011: _ctrl_ops.ends,    # NOP in P2
        0b100: _ctrl_ops.wsplit,  # NOP in P2
        0b101: _ctrl_ops.wjoin_with_exit,  # SystemExit semantics
        0b110: _ctrl_ops.startp,
        0b111: _ctrl_ops.endp,
    }
```

## Disasm registration pattern with worked examples (HIGH confidence)

Source: `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` (full 244-line table verbatim).

### Mask/match formulas

| Type | Match | Mask | Notes |
|------|-------|------|-------|
| custom0 R-type (funct7 only) | `(funct7 << 25) \| 0x0b` | `(0x7f << 25) \| 0x7f` | matches funct7 + opcode |
| custom0 R-type with funct3 | `(funct7 << 25) \| (funct3 << 12) \| 0x0b` | `(0x7f << 25) \| (0x7 << 12) \| 0x7f` | matches funct7 + funct3 + opcode |
| custom1 warp control | `(funct3 << 12) \| 0x2b` | `(0x7 << 12) \| 0x7f` | matches funct3 + opcode (funct7 ignored) |

### Three concrete worked examples (P2-available)

**1. wrspr (ISS-full, funct7=0x49) — custom0:**
```python
match = (0x49 << 25) | 0x0b      # = 0x9200000B
mask  = (0x7f << 25) | 0x7f      # = 0xFE00007F
disasm_insn_t("wrspr", 0x9200000B, 0xFE00007F, gtx_xrd, gtx_xrs1, gtx_xrs2)
# Trace output for `wrspr a0, t0, t1`: "wrspr   a0, t0, t1"
```

**2. wjoin (custom0 firmware variant, funct7=0x03):**
```python
match = (0x03 << 25) | 0x0b      # = 0x0600000B
mask  = (0x7f << 25) | 0x7f      # = 0xFE00007F
disasm_insn_t("wjoin", 0x0600000B, 0xFE00007F, gtx_xrd, gtx_xrs1, gtx_xrs2)
```

**3. warp_start_p (custom1, funct3=0b110):**
```python
match = (0b110 << 12) | 0x2b     # = 0x0000602B
mask  = (0x7 << 12) | 0x7f       # = 0x0000707F
disasm_insn_t("warp_start_p", 0x0000602B, 0x0000707F, gtx_xrd, gtx_xrs1, gtx_xrs2)
```

### Per-op registry decorator (D-13) — concrete shape

```python
# src/main/python/riscv/gtx/_registry.py
from typing import Callable, Optional, List
from riscv.disasm import disasm_insn_t

# Module-level accumulator — ops register at import time
_HANDLER_REGISTRY: List[dict] = []

def handler(*, kind: str, funct7: Optional[int] = None, funct3: Optional[int] = None,
            mnemonic: Optional[str] = None, mask_funct3: bool = False):
    """Register an op handler + (optional) disasm entry.

    kind: 'custom0' or 'custom1'
    funct7: required for custom0 (or 7-bit value for ignored-funct7 custom1)
    funct3: optional for custom0 (variant); required for custom1
    mnemonic: if provided, contributes to disasm table
    mask_funct3: if True, disasm match includes funct3 (matters for MM variants)
    """
    def decorator(fn: Callable):
        _HANDLER_REGISTRY.append({
            'fn': fn, 'kind': kind, 'funct7': funct7, 'funct3': funct3,
            'mnemonic': mnemonic, 'mask_funct3': mask_funct3,
        })
        return fn
    return decorator

def collect_for_kind(kind: str) -> dict:
    """Build dispatch dict for a given kind ('custom0' or 'custom1')."""
    out = {}
    for entry in _HANDLER_REGISTRY:
        if entry['kind'] != kind:
            continue
        key = entry['funct7'] if kind == 'custom0' else entry['funct3']
        out[key] = entry['fn']
    return out

def collect_disasms() -> List[disasm_insn_t]:
    """Build disasm list from all registered handlers with mnemonics."""
    from .disasm import add_r_custom0, add_rf3_custom0, add_warp
    out = []
    for entry in _HANDLER_REGISTRY:
        if not entry['mnemonic']:
            continue
        if entry['kind'] == 'custom0' and not entry['mask_funct3']:
            out.append(add_r_custom0(entry['mnemonic'], entry['funct7']))
        elif entry['kind'] == 'custom0' and entry['mask_funct3']:
            out.append(add_rf3_custom0(entry['mnemonic'], entry['funct7'], entry['funct3']))
        elif entry['kind'] == 'custom1':
            out.append(add_warp(entry['mnemonic'], entry['funct3']))
    return out
```

```python
# src/main/python/riscv/gtx/ops/spr.py — sample op module
from .._registry import handler

@handler(kind='custom0', funct7=0x49, mnemonic='wrspr')
def wrspr_iss(npu, proc, insn, xs1, xs2):
    """ISS-full WRSPR — funct7=0x49 — read rs1 (addr), rs2 (val) from XPR."""
    state = proc.get_state()
    addr = state.XPR[insn.rs1] & 0xFFFF
    val = state.XPR[insn.rs2]
    npu.spr_router.wr_spr(addr, val)
    return 0

@handler(kind='custom0', funct7=0x48, mnemonic='rdspr')
def rdspr_iss(npu, proc, insn, xs1, xs2):
    """ISS-full RDSPR — funct7=0x48 — read addr from rs1, return val (also write to rd)."""
    state = proc.get_state()
    addr = state.XPR[insn.rs1] & 0xFFFF
    val = npu.spr_router.rd_spr(addr)
    if insn.rd != 0:
        state.XPR.write(insn.rd, val)
    return val

@handler(kind='custom0', funct7=0x00, mnemonic='wrspr_gem5')
def wrspr_gem5(npu, proc, insn, xs1, xs2):
    """gem5-simplified WRSPR — collision: rs1!=0 → MM (P4 stub)."""
    if insn.rs1 != 0:
        return 0  # P4: firmware_mm_op stub
    state = proc.get_state()
    addr = state.XPR[insn.rs1] & 0xFFFF  # always 0 here
    # NOTE: actual gem5 marker is xs1=xs2=1, val read directly from XPR.
    val_rs1 = state.XPR[insn.rs1]  # =0 (since insn.rs1==0 → XPR[0]=0)
    val_rs2 = state.XPR[insn.rs2]
    npu.spr_router.wr_spr(val_rs1 & 0xFFFF, val_rs2)
    return 0
```

**Important note on wrspr_gem5 above:** The gem5 marker pattern is `insn.rs1 == 0` (the convention, per DISP-01). When insn.rs1==0, `XPR[0]` is always 0, so `val_rs1 & 0xFFFF == 0`. This means the gem5 path `wrspr` writes to SPR address 0 (`GSPR_GTX_RUN`). **Verify this with C++ behavior at plan stage** — looking at C++ line 63: `wr_spr(static_cast<uint16_t>(val_rs1 & 0xFFFF), val_rs2)` — `val_rs1` IS `XPR[insn.rs1]=XPR[0]=0`. So gem5 WRSPR always writes address 0. This may be a deliberate quirk (GSPR_GTX_RUN trigger), or it may indicate gem5 firmware passes the address in a different register. Either way, P2 ports the C++ behavior verbatim.

## WJOIN + GTX_NO_EXIT semantics (HIGH confidence)

### C++ reference (gtx_npu_custom1.cc:65-122)

The C++ custom1 funct3=0b101 (WJOIN) handler:
1. Computes `elapsed = total_npu_cycles - wsplit_cycle`
2. Optionally dumps L1 (per-SPU) to `${GTX_L1_DUMP}_n${N}_s${S}.hex`
3. Optionally dumps L2 (per-NEST) to `${GTX_L2_DUMP}_n${N}.hex`
4. **DDR dump moved to atexit** (line 116) — done at HTIF exit
5. Returns elapsed (writeback to rd via Spike custom1 macro)

**Where is `exit(0)` called?** Surprisingly, NOT in `custom1` directly. The C++ reference removed the `exit(0)` call from custom1 (commented-out reference at gtx_npu_custom1.cc:67: `"L1_DUMP, L2_DUMP 사라져야함. 꼭 split join이 아니여도 동작할 수 있음."`). Exit happens via firmware `_Exit()` writing to HTIF tohost, which Spike intercepts. The atexit handler then dumps DDR.

**HOWEVER**, the project's CORE-03 requirement and CONTEXT.md D-07 explicitly state: **WJOIN MUST raise SystemExit(0) if GTX_NO_EXIT is unset.** This is a Python-side simplification (no firmware HTIF cycle in unit tests). At plan stage, decide:

- **Recommended:** Python WJOIN path implements both. In integration test (`pyspike --extlib=riscv.gtx nop_wjoin.elf`), Spike's HTIF will exit when firmware does `_Exit()`. In unit tests, calling `npu.custom1(proc, insn, ...)` with funct3=0b101 raises SystemExit so test can `pytest.raises(SystemExit)` directly. The env var `GTX_NO_EXIT=1` lets tests that want to continue do so.

```python
# src/main/python/riscv/gtx/ops/control.py
import os
from .._registry import handler

@handler(kind='custom1', funct3=0b101, mnemonic='warp_join')
def wjoin_with_exit(npu, proc, insn, xs1, xs2):
    """custom1 funct3=0b101 — JOIN with optional SystemExit (CORE-03, D-07).

    Per D-07: read GTX_NO_EXIT every call (no caching).
    Unset → raise SystemExit(0). Set/truthy → return 0.
    """
    # Future: dump L1/L2/DDR per env vars (GTX_L1_DUMP, GTX_L2_DUMP, GTX_DDR_DUMP).
    # P2: skip dumps. P3+ implement.
    if not os.environ.get('GTX_NO_EXIT'):
        raise SystemExit(0)
    return 0
```

### How tests exercise both modes (D-08)

```python
# tests/gtx/test_wjoin.py
import pytest
from riscv.gtx.npu import GtxNpu
from ._mocks import MockProcessor, MockInsn

def test_wjoin_default_raises_systemexit(monkeypatch):
    monkeypatch.delenv('GTX_NO_EXIT', raising=False)
    npu = GtxNpu()
    proc = MockProcessor()
    insn = MockInsn(funct=0x00, xd=1, xs1=0, xs2=1, rs1=0, rs2=0, rd=0)  # funct3=0b101
    with pytest.raises(SystemExit) as exc_info:
        npu.custom1(proc, insn, xs1=0, xs2=0)
    assert exc_info.value.code == 0

def test_wjoin_with_no_exit_returns_zero(monkeypatch):
    monkeypatch.setenv('GTX_NO_EXIT', '1')
    npu = GtxNpu()
    proc = MockProcessor()
    insn = MockInsn(funct=0x00, xd=1, xs1=0, xs2=1, rs1=0, rs2=0, rd=0)
    ret = npu.custom1(proc, insn, xs1=0, xs2=0)
    assert ret == 0
```

## Loop state machine spec (HIGH confidence)

### State diagram (sourced from gtx_npu_loop.cc verbatim)

```
                  ┌──────────────────────┐
                  │  WarpState           │
                  │  is_ploop = False    │
                  │  is_tloop = False    │
                  │  is_sloop = False    │
                  └──────────┬───────────┘
                             │ start_p(rs1, rs2)
                             v
                ┌────────────────────────┐
                │  is_ploop = True       │
                │  tmu_id  = extract_id  │
                └─────┬──────────────────┘
                      │
       ┌──────────────┼─────────────────┐
       │              │                 │
       v              v                 v
   start_t       start_s             end_p
   (in P-loop)   (in P-loop)         clear is_ploop
       │              │                 │
       v              v                 ↓
   is_tloop=True  is_sloop=True     return to base
   curr_id=...   curr_id=...
       │              │
       v              v
    end_t           end_s
   clear t-loop   clear s-loop
       │              │
       └──────┬───────┘
              v
        back to P-loop only
```

### Invariants (must hold; pytest assertions)

1. **start_p sets `is_ploop=True`** and stores `tmu_id`.
2. **start_t requires `is_ploop=True`** (warning logged if violated; flag still set).
3. **start_s requires `is_ploop=True`** (warning logged if violated).
4. **end_t requires `is_tloop=True && is_ploop=True`** (warning logged if violated; flag still cleared).
5. **end_p requires `is_ploop=True`** (warning logged if violated; flag still cleared).
6. **end_p with no prior wsplit** triggers DDR dump (P3+ semantics; P2 skip).
7. **End state** after `start_p → start_t → end_t → end_p` MUST be `(False, False, False)`.

### Storage: per-instance, not per-hart

C++ stores `tmu_id`, `curr_id`, `is_ploop`, `is_tloop`, `is_sloop` as fields on `gtx_npu_t` (one instance per Spike process; `processor_t::extensions` maps name to one extension). Pyspike registers ONE extension per name → ONE GtxNpu instance per process. So per-hart state = per-instance state. Multi-hart correctness is NOT a P2 concern (RoCC extensions are inherently per-process-singleton in this model).

```python
# src/main/python/riscv/gtx/warp_state.py
from dataclasses import dataclass

@dataclass
class WarpState:
    """P/S/T loop state machine — port of gtx_npu_t loop fields."""
    is_ploop: bool = False
    is_tloop: bool = False
    is_sloop: bool = False  # P3+ only — DMA paths
    tmu_id: int = 0   # NEST id selected by start_p
    curr_id: int = 0  # SPU id (T-loop) or GDMAC id (S-loop)

    def reset(self) -> None:
        self.is_ploop = False
        self.is_tloop = False
        self.is_sloop = False
        self.tmu_id = 0
        self.curr_id = 0
```

```python
# src/main/python/riscv/gtx/ops/control.py — concrete loop handlers
def _extract_id(rs1: int, rs2: int) -> int:
    """gtx_npu_loop.cc:21-23 dual-mode addressing."""
    return (rs2 & 0x3F) if (rs2 & 0x400) else (rs1 & 0xFFFFFFFF)

@handler(kind='custom1', funct3=0b110, mnemonic='warp_start_p')
def startp(npu, proc, insn, xs1, xs2):
    state = proc.get_state()
    rs1_val = state.XPR[insn.rs1]
    rs2_val = state.XPR[insn.rs2]
    new_id = _extract_id(rs1_val, rs2_val)
    if new_id >= GTX_NEST_NUM:
        new_id = 0
    npu.warp.tmu_id = new_id
    npu.warp.is_ploop = True
    return 0

@handler(kind='custom1', funct3=0b111, mnemonic='warp_end_p')
def endp(npu, proc, insn, xs1, xs2):
    npu.warp.is_ploop = False
    return 0  # P3+ adds DDR dump on no-WSPLIT firmware

# Similar for startt/endt/starts/ends (port from gtx_npu_loop.cc verbatim)
```

## Test fixture inventory (HIGH confidence)

### `nop_wjoin.elf` — assembly source

Per CONTEXT.md "specifics" section + ROADMAP success criterion 1:

```asm
# tests/gtx/data/elf/nop_wjoin.S
.section .text._start
.global _start
_start:
    addi sp, sp, -16        # sp = 0x80100000 - 16 = 0x800FFFF0 (must be valid DRAM)
    # No body — just exit via WJOIN
    .insn r 0x2b, 0b101, 0, x0, x0, x0   # custom1 funct3=0b101 = WJOIN
    j .                                   # safety loop (WJOIN raises SystemExit, never reached)
```

`_start` ELF entry point conventions: spike uses 0x80000000 by default (the standard RISC-V boot address). `nop_wjoin.elf` must be linked at this address.

### Build command (Makefile or single command)

```bash
# Single command (verified /opt/riscv/bin/riscv64-unknown-elf-gcc 15.2.0):
cd tests/gtx/data/elf
riscv64-unknown-elf-gcc -nostdlib -nostartfiles -static \
    -Ttext=0x80000000 \
    -o nop_wjoin.elf nop_wjoin.S
```

Or `Makefile`:
```makefile
# tests/gtx/data/elf/Makefile
CC = riscv64-unknown-elf-gcc
nop_wjoin.elf: nop_wjoin.S
	$(CC) -nostdlib -nostartfiles -static -Ttext=0x80000000 -o $@ $<
```

### sp init = 0x80100000 — provenance

Set in `gtx_npu_t::reset()` at `gtx_npu_core.cc:144-156`:
```cpp
proc.get_state()->XPR.write(2, 0x80100000ULL);
```

The Python equivalent (in `GtxNpu.reset()`) writes the same value to XPR[2] (sp = x2). For `addi sp, sp, -16` to NOT trap, the address `0x80100000 - 16 = 0x800FFFF0` must be in valid memory. Spike's default DRAM mapping at 0x80000000 size 256MB covers this — no extra setup needed.

**Important:** P2 reset() must call BEFORE the firmware `_start` — Spike's processor reset cycle calls `extension_t::reset()` BEFORE jumping to `_start` (this is verified by the existing `examples/xhuimt/__init__.py::HuiMtISA::reset` pattern + `riscv_extension.cc:76-81` which calls `PYBIND11_OVERRIDE(void, extension_t, reset, ...)`).

### FPU enable (mstatus.FS)

The C++ reset() also enables FPU:
```cpp
reg_t mstatus = proc.get_state()->mstatus->read();
mstatus = (mstatus & ~(reg_t)0x6000) | (reg_t)0x2000;  // FS = 01 (Initial)
proc.put_csr(0x300 /*mstatus*/, mstatus);
```

For `nop_wjoin.elf` (which has no FP instructions), this is technically not required for P2. But for forward compatibility with other firmware (P4 GEMM uses FP), the Python reset() should match. **Plan-stage decision:** Either include FPU enable in `GtxNpu.reset()` (matches C++) or document as deferred to P3.

Verified Python access: `proc.put_csr(0x300, mstatus)` works (`py_module.cc:706`). `mstatus->read()` is more involved — `state.mstatus` may not be directly exposed; use `proc.get_csr(0x300)` if available (`py_module.cc:700-705`).

## Hybrid mock strategy (D-17/D-19/D-20)

### Mock spec (minimum required for P2 unit tests)

```python
# tests/gtx/_mocks.py — INTERNAL ONLY (D-20: never exposed via riscv.gtx)
"""Mocks for unit tests that must run without _riscv.so being built.

D-19: minimal spec — only what P2 tests need.
- MockProcessor.get_state().XPR.read(i) / write(i, val)
- MockInsn fields: rs1, rs2, funct, xs1, xs2, xd, rd
- MMU added in P3 (DMA paths)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockXPR:
    _regs: List[int] = field(default_factory=lambda: [0] * 32)

    def __getitem__(self, i: int) -> int:
        return self._regs[i]

    def write(self, i: int, val: int) -> None:
        if i != 0:  # x0 is hardwired zero
            self._regs[i] = val & 0xFFFFFFFFFFFFFFFF


@dataclass
class MockState:
    XPR: MockXPR = field(default_factory=MockXPR)


@dataclass
class MockProcessor:
    _state: MockState = field(default_factory=MockState)

    def get_state(self) -> MockState:
        return self._state

    # Optional for P2 reset() if testing FPU enable:
    def put_csr(self, which: int, val: int) -> None:
        pass

    def get_csr(self, which: int) -> int:
        return 0


@dataclass
class MockInsn:
    """Mirrors rocc_insn_t fields exposed by py_module.cc:391-409."""
    opcode: int = 0x0b
    rd: int = 0
    xs2: int = 0
    xs1: int = 0
    xd: int = 0
    rs1: int = 0
    rs2: int = 0
    funct: int = 0  # this is funct7
```

### conftest.py hybrid fallback

```python
# tests/gtx/conftest.py
import pytest

try:
    from riscv.processor import processor_t
    from riscv.extension import rocc_insn_t
    _RISCV_AVAILABLE = True
except ImportError:
    from ._mocks import MockProcessor as processor_t  # type: ignore
    from ._mocks import MockInsn as rocc_insn_t  # type: ignore
    _RISCV_AVAILABLE = False


@pytest.fixture
def riscv_available() -> bool:
    return _RISCV_AVAILABLE


@pytest.fixture
def proc():
    if _RISCV_AVAILABLE:
        # Real spike processor needs a sim_t — use the mock_sim fixture from
        # tests/conftest.py (D-18: ensure that conftest is import-safe).
        pytest.skip("real proc fixture requires sim_t setup — covered by integration test")
    from ._mocks import MockProcessor
    return MockProcessor()


@pytest.fixture
def insn_factory():
    from ._mocks import MockInsn
    return MockInsn
```

### tests/conftest.py D-18 try/except guard

The existing `tests/conftest.py` does `from riscv.cfg import cfg_t, mem_cfg_t` etc. at module level. If `_riscv.so` is not built, this import fails and ALL tests under `tests/gtx/` fail to collect. D-18 fix:

```python
# tests/conftest.py — patched form
import os
import pathlib
# ... existing code ...

try:
    # pylint: disable=import-error,no-name-in-module
    from riscv.cfg import cfg_t, mem_cfg_t
    from riscv.debug_module import debug_module_config_t
    from riscv.sim import sim_t
    _RISCV_AVAILABLE = True
except ImportError:
    _RISCV_AVAILABLE = False


@pytest.fixture(scope="session")
def mock_sim():
    if not _RISCV_AVAILABLE:
        pytest.skip("_riscv not built — sim_t fixture unavailable")
    yield sim_t(...)  # existing definition
```

## Common Pitfalls

### Pitfall 1: funct3 reconstruction (NOT funct.bits[14:12])

**What goes wrong:** `insn.funct` returns funct7 (bits 31:25), not funct3. Code that does `if insn.funct == 0b101: # JOIN` will never match.

**Why it happens:** RoCC's `{xd, xs1, xs2}` bit pattern is in bits[14:12] — these are NOT `funct3` semantically (they're register-pass flags). But for custom1 dispatch, the C++ code REUSES them as funct3 by reconstructing the value: `funct3 = (xd << 2) | (xs1 << 1) | xs2`.

**How to avoid:** Always reconstruct funct3 in custom1 dispatcher:
```python
funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2
```

**Warning signs:** custom1 tests passing for one funct3 only (the one that happens to match `funct.bits[14:12]==0`).

### Pitfall 2: xs1=0 → -1 marshalling (CORE-04)

**What goes wrong:** Direct read `state.XPR[insn.rs1]` works. Use of `xs1` arg directly fails when `insn.xs1 == 0` (Spike marshals -1).

**Why it happens:** `define_custom_func` macro at `vendor/spike/riscv/rocc.h:42`:
```cpp
reg_t xs1 = u.r.xs1 ? state->XPR[insn.rs1()] : -1;
```

**How to avoid:** Wrap all custom0/custom1 handlers with a decorator (D-05) that detects `xs1 == 0` (or, equivalently, `xs1 == -1` cast to unsigned reg_t = 0xFFFFFFFFFFFFFFFF in 64-bit) and replaces with `state.XPR[insn.rs1]`. ALL ISS-full encodings (funct7=0x48, 0x49, etc.) require this. Many gem5 encodings (xs1=1) don't — but the decorator is no-op then.

**Warning signs:** Test passes with `MockInsn(xs1=1)`, fails with `xs1=0`.

### Pitfall 3: GtxMemory.spr is single-dict but routing is layered

**What goes wrong:** `mem.spr[0x900] = val` writes to ONE dict, but C++ routes to NEST/SPU based on loop state.

**Why it happens:** Phase 1 D-11 said "single dict" — that was for GSPR. NSPR/LSPR routing was deferred to P2.

**How to avoid:** Introduce `nspr: list[dict]` and `lspr: list[list[dict]]` storage in P2. Either on GtxNpu directly (recommended — matches C++ `nests[n].nspr` / `nests[n].spus[s].lspr` exactly) or extend GtxMemory backward-compatibly. `spr_router` is the single mapping authority.

**Warning signs:** `wr_spr(0x900, val)` then `rd_spr(0x900)` returns wrong value when loop state changes.

### Pitfall 4: WJOIN appears in BOTH custom0 (funct7=0x03) and custom1 (funct3=0b101)

**What goes wrong:** Implementing only one path makes some firmware fail.

**Why it happens:** The C++ reference defines wjoin in two places — custom0 funct7=0x03 (firmware shorthand, no exit) AND custom1 funct3=0b101 (full warp control with dump+exit semantics).

**How to avoid:** P2 implements both. ONLY custom1 funct3=0b101 raises SystemExit. custom0 funct7=0x03 returns 0.

**Warning signs:** `nop_wjoin.elf` uses `.insn r 0x2b, 0b101, 0, x0, x0, x0` (custom1 form) — that's the variant that exits.

### Pitfall 5: get_disasms() entries garbage-collected

**What goes wrong:** `get_disasms()` builds and returns a list, but if it doesn't keep references, `disasm_insn_t` instances are GC'd before Spike consumes them.

**Why it happens:** `riscv_extension.cc:39-54` tracks each entry via `PythonBridge::track`, but the Python-side list must outlive the C++ vector building.

**How to avoid:** Store the entries on `self._disasm_entries: List[disasm_insn_t]` in `__init__` and return `list(self._disasm_entries)` (copy of list, but elements are persistent). Verified pattern: xhuimt builds `disasm_insn_t` inline in `get_disasms()` — no leak issue because pybind11 `track()` extends lifetime to bridge level. **Either pattern works**, but caching avoids rebuild on every call.

### Pitfall 6: pyspike --extlib resolution for `riscv.gtx`

**What goes wrong:** `pyspike --extlib=riscv.gtx ...` doesn't load.

**Why it happens:** `riscv/__main__.py:30-58` parses `--extlib=name`, distinguishes C++ shared libs (`.so` suffix) from Python modules. For Python modules, the path must resolve via `pathlib.Path("riscv.gtx")` — but that's not how Python imports work. Inspect `__main__.py:40-48`: if the arg is not a `.so` file, it's appended to `pylibs` and serialized into `PYSPIKE_LIBS` env var. Then `py_bridge.cc` at runtime imports each entry via `importlib.import_module()`. So `--extlib=riscv.gtx` → adds `riscv.gtx` to PYSPIKE_LIBS → `importlib.import_module('riscv.gtx')` happens → `riscv/gtx/__init__.py` runs → triggers `from . import npu` → `@isa.register("gtx")` fires → factory registered.

**How to avoid:** Ensure `riscv/gtx/__init__.py` imports `npu` (or sets up `__getattr__` lazy import) so the decorator fires. Phase 1's `__init__.py` does NOT import npu (it's not yet built). P2 must add `from . import npu  # noqa: F401  -- triggers @isa.register("gtx")`.

**Warning signs:** `pyspike --extlib=riscv.gtx` runs but firmware crashes with "extension 'gtx' not found".

## Code Examples

### Reset() — full P2 implementation skeleton

```python
# Source: examples/xhuimt/__init__.py:53-55 + vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc:144-189
def reset(self, proc) -> None:
    super().reset(proc)
    # CORE-02: sp init
    proc.get_state().XPR.write(2, 0x80100000)

    # FPU enable (forward-compat for P4 GEMM)
    # mstatus.FS = 01 (Initial). Bits[14:13] = 0x6000 mask, value 0x2000.
    try:
        mstatus = proc.get_csr(0x300)
        mstatus = (mstatus & ~0x6000) | 0x2000
        proc.put_csr(0x300, mstatus)
    except Exception:
        pass  # if get_csr unsupported in mock, skip silently

    # mxe_accum zero-init (D-06)
    self._mxe_accum.fill(0.0)

    # Memory zero-init (CORE-02)
    self.mem._l0_bytes.fill(0)
    self.mem._l1_bytes.fill(0)
    self.mem._l2_bytes.fill(0)
    # DDR is lazy — leave as None unless allocated

    # SPR zero-init + defaults (gtx_npu_core.cc:80-109)
    self.gspr.clear()
    self.gspr[0x000] = 0  # GSPR_GTX_RUN
    self.gspr[0x001] = 0
    self.gspr[0x002] = 0
    self.gspr[0x003] = 0
    self.gspr[0x004] = 0
    self.gspr[0x010] = 0
    self.gspr[0x011] = 0
    for n in range(GTX_NEST_NUM):
        self.nspr[n].clear()
        self.nspr[n][0x400] = 0xFFFF  # NSPR_THREAD_MASK
        self.nspr[n][0x401] = 0
        self.nspr[n][0x402] = 1  # NSPR_TYPE = FP16
        self.nspr[n][0x403] = 0
        self.nspr[n][0x700] = 0
        self.nspr[n][0x780] = 0
        self.nspr[n][0x781] = 0
        self.nspr[n][0x782] = 0
        for s in range(GTX_SPU_NUM):
            self.lspr[n][s].clear()
            self.lspr[n][s][0x900] = 0  # LSPR_SPM_ADDRA
            self.lspr[n][s][0x901] = 0
            self.lspr[n][s][0x902] = 0
            self.lspr[n][s][0x903] = 0

    # Warp state reset
    self.warp.reset()
```

### SPR router — full P2 implementation

```python
# Source: vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc:16-107 (port verbatim)
# src/main/python/riscv/gtx/spr_router.py
from .params import (GSPR_BASE, GSPR_END, NSPR_BASE, NSPR_END,
                     LSPR_BASE, LSPR_END, GTX_NEST_NUM, GTX_SPU_NUM)
from .ops import control as _ctrl  # for startp/endp/etc. trigger

# GSPR loop control addresses
GSPR_STARTP, GSPR_ENDP = 0x100, 0x101
GSPR_STARTS, GSPR_ENDS = 0x102, 0x103
GSPR_STARTT, GSPR_ENDT = 0x104, 0x105


def wr_spr(npu, addr: int, value: int) -> None:
    """Write SPR — port of gtx_npu_t::wr_spr (gtx_npu_spr.cc:16-78)."""
    # Loop control side-effects (gem5 v2.0 convention)
    if addr == GSPR_STARTP:
        return _ctrl._do_startp(npu, value, 0)
    if addr == GSPR_ENDP:
        return _ctrl._do_endp(npu, value, 0)
    if addr == GSPR_STARTS:
        return _ctrl._do_starts(npu, value, 0)
    if addr == GSPR_ENDS:
        return _ctrl._do_ends(npu, value, 0)
    if addr == GSPR_STARTT:
        return _ctrl._do_startt(npu, value, 0)
    if addr == GSPR_ENDT:
        return _ctrl._do_endt(npu, value, 0)

    if LSPR_BASE <= addr <= LSPR_END:
        if npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM and \
           npu.warp.curr_id < GTX_SPU_NUM:
            npu.lspr[npu.warp.tmu_id][npu.warp.curr_id][addr] = value
        elif npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            for s in range(GTX_SPU_NUM):
                npu.lspr[npu.warp.tmu_id][s][addr] = value
        else:
            # No loop context — fallback to NEST 0, SPU 0 (warning log)
            npu.lspr[0][0][addr] = value
    elif NSPR_BASE <= addr <= NSPR_END:
        if npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM:
            npu.nspr[npu.warp.tmu_id][addr] = value
        else:
            npu.nspr[0][addr] = value
    elif GSPR_BASE <= addr <= GSPR_END:
        npu.gspr[addr] = value
    # else: warning, drop


def rd_spr(npu, addr: int) -> int:
    """Read SPR — port of gtx_npu_t::rd_spr (gtx_npu_spr.cc:83-107)."""
    if LSPR_BASE <= addr <= LSPR_END:
        if npu.warp.is_tloop and npu.warp.tmu_id < GTX_NEST_NUM and \
           npu.warp.curr_id < GTX_SPU_NUM:
            return npu.lspr[npu.warp.tmu_id][npu.warp.curr_id].get(addr, 0)
        return npu.lspr[0][0].get(addr, 0)
    if NSPR_BASE <= addr <= NSPR_END:
        nid = npu.warp.tmu_id if (npu.warp.is_ploop and npu.warp.tmu_id < GTX_NEST_NUM) else 0
        return npu.nspr[nid].get(addr, 0)
    if GSPR_BASE <= addr <= GSPR_END:
        return npu.gspr.get(addr, 0)
    return 0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single SPR dict (Phase 1 D-11) | Layered GSPR/NSPR/LSPR (P2 router) | P2 plan stage | Phase 1 tests still pass — dict for GSPR exists; NSPR/LSPR added |
| `match` statement (PEP 634) | `dict[int, Callable]` dispatch | Phase 1 STATE.md | Locked. cp310+ supports match, but project-wide consistency wins. |
| C++ `switch(funct7)` mega-function | Per-op modules with `@gtx.handler` registry (D-13) | P2 plan | Better cohesion. Each op is one source file — handler + disasm together. |
| `exit(0)` from C++ WJOIN | `raise SystemExit(0)` from Python WJOIN | P2 plan (D-07) | Test-friendly via `pytest.raises(SystemExit)`. |
| Spike's marshal-with--1 for xs1=0 | Decorator auto-wrap → direct XPR read | P2 plan (D-05) | Single decorator handles all op handlers; signatures stay clean. |

**Deprecated/outdated:**
- C++ `gtx_npu_t` mexec full microcode loop — never triggered by current firmware. v2 deferred (MEXEC-01).
- Spike commitlog flag (`--enable-gtxcommitlog`) — explicitly excluded from v1 (REQUIREMENTS.md "Out of Scope").

## Open Questions

1. **gem5 WRSPR/RDSPR address — does insn.rs1==0 imply addr=XPR[0]=0 always?**
   - What we know: C++ at `gtx_npu_custom0.cc:60-72` does `wr_spr(val_rs1 & 0xFFFF, val_rs2)` after the rs1!=0 check. With rs1==0, `val_rs1 = XPR[0] = 0`, so addr is always 0.
   - What's unclear: This means gem5 WRSPR always writes to GSPR_GTX_RUN (0x000). Is that intentional? Or is the gem5 firmware passing the address differently (e.g., as an immediate)?
   - **Recommendation:** Plan stage — search `vendor/gtx_cpp_reference/test/` or related firmware for actual gem5 WRSPR usage. If indeed addr=0 always, document and port verbatim. If not, the C++ may have a bug we shouldn't replicate.

2. **WJOIN exit ownership — custom1 directly or atexit-via-tohost?**
   - What we know: C++ `gtx_npu_custom1.cc:65-122` does NOT call `exit(0)` — it returns elapsed cycles. DDR dump is in atexit (`gtx_npu_core.cc:61-73`).
   - What's unclear: Project's CORE-03 says WJOIN raises SystemExit. C++ doesn't. Are they semantically equivalent?
   - **Recommendation:** The Python-side simplification (raise SystemExit on WJOIN) is acceptable for unit-test ergonomics (D-08). For the integration test (`pyspike --extlib=riscv.gtx nop_wjoin.elf`), Spike's HTIF tohost path will exit independently. Both paths converge on "process exits cleanly." Document this in test_wjoin.py.

3. **`mxe_accum` shape M_TILE / N_TILE values?**
   - What we know: D-06 says shape `(NEST_NUM, SPU_NUM, M_TILE, N_TILE)`. Phase 1 lists `GTX_NEST_NUM=4`, `GTX_SPU_NUM=16`. M_TILE/N_TILE not yet defined.
   - What's unclear: Match what unit? L0=1KB/SPU = 512 FP16s. L1=384KB/SPU. Likely tile is 16×16 (matrix block size for GEMM) — but C++ source authoritative.
   - **Recommendation:** Plan stage — search C++ for `mxe_accum` shape declaration. Likely in `gtx_npu.h` or `gtx_npu_mm.cc`. From the disasm table inferring (mm operates on row/col 16-bit fields with HW convention 0=65536), tile is bounded by L1 capacity. **Probable answer: 16×16 (matches typical GEMM block).** Verify before locking.

4. **Should `GtxNpu.reset()` raise warnings on startp-without-prior-reset etc.?**
   - What we know: C++ does `GTX_TRACE` warnings (stderr). Python equivalent could be `warnings.warn`.
   - **Recommendation:** Skip warnings in P2 — they generate test noise. Add a `_strict_mode` flag in P5/P6 that makes them assertions.

5. **`_riscv.so` build availability for CI vs local — what's the actual current state?**
   - What we know: deferred-items.md mentions pybind11 3.0.4 / csr_t binding incompatibility. STATE.md says "canonical wheel build deferred to next CI cibuildwheel run."
   - What's unclear: Can `python setup.py build_ext --inplace` produce a working `_riscv.so` locally? D-15 pins pybind11<3.0.4 — does this resolve?
   - **Recommendation:** P2 plan should include a "smoke build verification" task as either Wave 0 or a precondition: `pip install -e .` then `python -c "from riscv.gtx import GtxNpu"`. If it fails, hybrid mock is the safety net (D-17).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All P2 work | Yes | 3.10.12 | — |
| NumPy 2.0+ | reset() mxe_accum, Phase 1 carryover | Yes | 2.2.6 | — |
| pytest | All tests | Yes | 9.0.1 | — |
| `_riscv.so` (pyspike binding) | Integration test (CORE-01) | UNKNOWN | — | Mock fallback (D-17) for unit tests; integration test uses `skipif` |
| `riscv64-unknown-elf-gcc` | Build `nop_wjoin.elf` | Yes | 15.2.0 | Pre-build .elf and commit (D-22) |
| dtc (device-tree-compiler) | cibuildwheel before-all | Yes | 1.6.1 | — |
| `vendor/gtx_cpp_reference/gtx/` submodule | C++ ground-truth refs | Yes | initialized at SHA 80d524293 (verified on disk: 11 .cc files present) | — |
| RISC-V cross-toolchain `/opt/riscv/` | ELF build | Yes | — | — |

**Missing dependencies with no fallback:**
- None critical — `_riscv.so` is the only uncertainty, and the hybrid mock strategy (D-17) is the explicit plan.

**Missing dependencies with fallback:**
- `_riscv.so` build status → unit tests use mocks; integration test uses `skipif`.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.1 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (Phase 1) |
| Quick run command | `pytest tests/gtx/ -x --noconftest -o "addopts="` (mock fallback, fast) |
| Full suite command | `pytest tests/ -v` (after `_riscv.so` builds) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORE-01 | `@isa.register("gtx")` registers; `pyspike --extlib=riscv.gtx nop_wjoin.elf` exits 0 | integration | `pytest tests/gtx/test_register.py tests/gtx/test_nop_elf.py -x` | ❌ Wave 0 |
| CORE-02 | `reset()` sets sp=0x80100000; mxe_accum/L0/L1/L2/SPR zero-init | unit | `pytest tests/gtx/test_reset.py -x` | ❌ Wave 0 |
| CORE-03 | WJOIN raises SystemExit when GTX_NO_EXIT unset; returns 0 when set | unit | `pytest tests/gtx/test_wjoin.py -x` | ❌ Wave 0 |
| CORE-04 | xs1=0 workaround — handler reads via XPR[insn.rs1] regardless | unit | `pytest tests/gtx/test_xs1_zero.py -x` | ❌ Wave 0 |
| SPR-01 | GSPR/NSPR/LSPR routing in wr_spr/rd_spr | unit | `pytest tests/gtx/test_spr.py::test_routing -x` | ❌ Wave 0 |
| SPR-02 | WRSPR(0xCAFE)→RDSPR roundtrip via gem5 (funct7=0x00) AND ISS (funct7=0x49) | unit | `pytest tests/gtx/test_spr.py::test_wrspr_rdspr_roundtrip -x` | ❌ Wave 0 |
| DISASM-01 | get_disasms() returns ≥10 entries; sample 5 P2-available ops decode correctly | unit | `pytest tests/gtx/test_disasm.py -x` | ❌ Wave 0 |
| DISP-01 | custom0 funct7=0x00 routes to WRSPR (rs1==0) or NOP/MM-stub (rs1!=0) | unit | `pytest tests/gtx/test_dispatch.py::test_funct7_collision -x` | ❌ Wave 0 |
| DISP-02 | custom1 funct3 dispatch — start_p→start_t→end_t→end_p ends in (False, False) | unit | `pytest tests/gtx/test_warp.py::test_loop_state_machine -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/gtx/test_<task>.py -x --noconftest -o "addopts="` (mock fallback, ~1s)
- **Per wave merge:** `pytest tests/gtx/ -x` (full P2 suite, ~5s)
- **Phase gate:** Full suite green (`pytest tests/`) BEFORE `/gsd:verify-work`. Integration test (`test_nop_elf.py`) requires `_riscv.so` — gate handled by `skipif` if not built; rerun in CI cibuildwheel context.

### Wave 0 Gaps

- [ ] `tests/gtx/conftest.py` — hybrid mock fallback (D-17)
- [ ] `tests/gtx/_mocks.py` — MockProcessor / MockState / MockXPR / MockInsn (D-19/D-20)
- [ ] `tests/gtx/test_register.py` — `@isa.register('gtx')` validation
- [ ] `tests/gtx/test_reset.py` — sp init, zero-init, mxe_accum
- [ ] `tests/gtx/test_spr.py` — GSPR/NSPR/LSPR routing + WRSPR/RDSPR roundtrip
- [ ] `tests/gtx/test_warp.py` — start_p/end_p/start_t/end_t state machine
- [ ] `tests/gtx/test_dispatch.py` — funct7 collision heuristic
- [ ] `tests/gtx/test_disasm.py` — get_disasms structure + sample mnemonics
- [ ] `tests/gtx/test_wjoin.py` — SystemExit + GTX_NO_EXIT both modes (D-08)
- [ ] `tests/gtx/test_xs1_zero.py` — decorator workaround (D-05)
- [ ] `tests/gtx/test_nop_elf.py` — integration via pyspike CLI (skipif _RISCV)
- [ ] `tests/gtx/data/elf/nop_wjoin.S` — assembly source
- [ ] `tests/gtx/data/elf/nop_wjoin.elf` — prebuilt binary (D-22)
- [ ] `tests/gtx/data/elf/Makefile` — reproduce build
- [ ] `tests/conftest.py` patch — try/except guard for `riscv.cfg`/`riscv.sim` (D-18)

**Mocking strategy:** Unit tests instantiate `GtxNpu()` and call `custom*` directly with MockProcessor + MockInsn. Integration test (`test_nop_elf.py`) uses `subprocess.run(['pyspike', '--extlib=riscv.gtx', 'tests/gtx/data/elf/nop_wjoin.elf'])` and asserts `returncode == 0`.

**Bit-exact comparison for P2:** None required (no compute). State assertions only:
- `npu.warp.is_ploop is True` after start_p
- `proc.get_state().XPR[2] == 0x80100000` after reset
- `npu.gspr[0xCAFE & 0xFFFF] == 0xDEADBEEF` after WRSPR

The ULP comparison machinery (`verify.py`) doesn't activate until P4.

## Sources

### Primary (HIGH confidence — verbatim from project files)

- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — Class declaration, FP helpers, ALL funct7/funct3 constants, mxe_accum location reference
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom0.cc` (847 lines, fully read) — custom0 dispatch table, every funct7 case, xs1=0 workaround pattern
- `vendor/gtx_cpp_reference/gtx/gtx_npu_custom1.cc` (138 lines, fully read) — custom1 funct3 warp control, WJOIN dump+exit
- `vendor/gtx_cpp_reference/gtx/gtx_npu_spr.cc` (108 lines, fully read) — wr_spr/rd_spr routing
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc` (142 lines, fully read) — startp/endp/starts/ends/startt/endt
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` (260 lines, fully read) — reset(), constructor, REGISTER_EXTENSION
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` (244 lines, fully read) — full disasm table to port
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` — HW constants (NEST=4, SPU=16, L0/L1/L2/DDR sizes, GSPR/NSPR/LSPR addresses)
- `vendor/spike/riscv/rocc.h` — rocc_insn_t bitfield, define_custom_func macro (xs1=0→-1)
- `src/main/python/riscv/isa.py` — ISA, ROCC, @register, @arg decorator
- `src/main/cpp/py_module.cc` — pybind11 bindings (rocc_insn_t, processor_t, state_t, xpr_regfile_t, disasm_insn_t)
- `src/main/cpp/riscv_extension.cc` — py_rocc_t / py_extension_t trampolines
- `src/main/cpp/riscv_disasm.cc` — py_disasm_insn_t_create
- `examples/xhuimt/__init__.py` — Canonical RoCC extension pattern
- `examples/xhuimt/mylrsc.py` — get_instructions / get_disasms / get_csrs / reset patterns
- `examples/xhuimt/arg.py` — @isa.arg decorator usage
- `examples/xthead/__init__.py`, `theadba.py` — Alternative ISA pattern
- `tests/test_extension.py` — Extension lifecycle tests
- `tests/test_disasm.py` — disasm_insn_t API usage
- `tests/conftest.py` — mock_sim fixture pattern (and D-18 issue)
- `src/main/python/riscv/gtx/__init__.py`, `params.py`, `encoding.py`, `memory.py` — Phase 1 outputs
- `tests/gtx/test_memory_layout.py` — Phase 1 test pattern
- `.planning/phases/01-foundation/05-submodule-SUMMARY.md` — Submodule SHA verified

### Secondary (MEDIUM confidence — derived from primary cross-reference)

- C++ comment annotations (`gtx_npu.h:255-262`) — RoCC bit-field semantics for gem5 vs ISS encoding (cross-verified with custom0.cc behavior)
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — High-level architectural narrative (cross-verified against source)
- WJOIN exit semantics — C++ does NOT call `exit(0)` directly; project Python-side adds `SystemExit` for test ergonomics (CORE-03 + D-07 specify this divergence explicitly)

### Tertiary (LOW confidence — flagged for plan-stage validation)

- `mxe_accum` shape M_TILE/N_TILE values — not directly read; inferred to be 16×16 from disasm row/col 16-bit field convention
- gem5 WRSPR `insn.rs1 == 0 → addr = XPR[0] = 0` interpretation — likely correct port; verify with actual gem5 firmware
- Whether `proc.put_csr(0x300, mstatus)` works reliably in mock vs real proc — verify in plan
- pybind11<3.0.4 pin (D-15) impact on `_riscv.so` build success — verify in first P2 plan

## Metadata

**Confidence breakdown:**

- **Standard stack:** HIGH — all libraries already in repo, versions verified
- **Architecture (per-op registry, dispatch dict):** HIGH — pattern locked in CONTEXT.md, mirrors C++ structure, examples in xhuimt/
- **SPR encoding/routing:** HIGH — port verbatim from gtx_npu_spr.cc + gtx_params.h
- **funct7 dispatch table:** HIGH — full table read from gtx_npu.h:266-353 + custom0.cc verified case-by-case
- **funct3 warp dispatch:** HIGH — full table from custom1.cc verified
- **Disasm registration:** HIGH — formula and entries verbatim from gtx_npu_disasm.inc
- **WJOIN exit semantics:** MEDIUM — Python diverges from C++ intentionally; design rationale in D-07/D-08 is sound but novel for the project
- **Loop state machine:** HIGH — C++ source fully read
- **Pitfalls:** HIGH — direct from C++ comments + Phase 1 STATE.md
- **mxe_accum shape:** LOW — inferred, needs plan-stage verification
- **`_riscv.so` build readiness:** LOW — deferred-items mentions pybind11 3.0.4 issue

**Research date:** 2026-05-04
**Valid until:** 2026-06-03 (30 days for stable infrastructure-tier research; underlying C++ source is frozen submodule SHA 80d524293)
