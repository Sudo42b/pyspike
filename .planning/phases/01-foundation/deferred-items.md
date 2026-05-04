# Phase 01 — Deferred Items

Items discovered during Phase 01 execution that are out-of-scope for the originating plan but warrant tracking.

## Pre-existing Build Environment Issues

### pybind11 3.0.4 / csr_t binding inaccessibility (logged 2026-05-04 by 04-packaging)

**Symptom:** Local `pip wheel . -w /tmp/wheel-test/ --no-deps` fails at C++ compile step with:
```
error: static assertion failed: Cannot bind an inaccessible base class method;
       use a lambda definition instead
       (pybind11/pybind11.h:2006, instantiated from py_module.cc:90-91)
error: cannot convert ‘bool (py_csr_t::*)(long unsigned int) noexcept’ to
       ‘bool (csr_t::*)(long unsigned int) noexcept’ in return
```

**Root cause:** pybind11 3.0.4 (latest) tightened `is_accessible_base_of` template checks. The existing `py::class_<csr_t, py_csr_t, ...>::def(..., &py_csr_t::method, ...)` bindings in `src/main/cpp/py_module.cc:90-91` rely on the old laxer member-pointer cast behavior.

**Owner:** pyspike core (NOT the GTX port — out of scope for CLAUDE.md "no new C++ code" mandate)

**Possible fixes (not applied here):**
1. Wrap affected bindings in lambdas: `.def("name", [](csr_t& self, ...) { return self.method(...); })`
2. Pin pybind11 in `[build-system].requires` to `pybind11[global]>3,<3.0.4`
3. Verify cibuildwheel manylinux2014_x86_64 image's pybind11 version — CI may already have a working pin

**Why deferred from Plan 04:** Plan 04's scope is `pyproject.toml` packaging metadata only. The plan's correctness was canonically verified via:
- `setuptools.find_packages(where='src/main/python', include=['riscv', 'riscv.*'])` returns `['riscv', 'riscv.gtx', 'riscv.gtx.ops']` (vs `['riscv']` with old glob)
- sdist build succeeds; tarball contains all 7 `riscv.gtx` files; `vendor/gtx_cpp_reference` count = 0

Once the C++ binding fix lands (separate plan), the canonical `pip wheel .` build path will validate end-to-end.

**Recorded by:** 04-packaging plan execution (commit `f3c3b7a`)

---

## ~~RoCC Subclass Dispatch Lifecycle~~ — RESOLVED 2026-05-05

**Status:** RESOLVED inline during Phase 2 wrap-up per user directive
(commits `611c222`, `be91d2f`, `51dee8d`). Re-investigation surfaced the
real two-part root cause and a missing CLI flag, all of which were
straightforward to fix:

- **Missing `--extension=gtx`** — spike's `--extlib` loads the library
  but `--extension=<name>` is what activates the RoCC extension on the
  core. The 02-06 plan and downstream tests were missing this flag.
  Fixed in `tests/gtx/test_skeleton.py` (`51dee8d`).
- **D1 — `py_rocc_t` extension_t hook trampolines missing** —
  `py_rocc_t` inherits from `rocc_t` directly without trampolining
  `get_instructions / get_disasms / get_csrs / reset / set_debug`.
  Without the disasm trampoline, spike's `register_extension` saw
  `rocc_t::get_disasms() == {}` and rendered custom opcodes as
  `unknown` in `--log` even though dispatch was correct. Added the
  five trampolines in `src/main/cpp/riscv_extension.{h,cc}` with
  `if (!py_method)` fallback to the C++ base (`be91d2f`).
- **D2 — `pybind11::error_already_set` propagating from custom* on
  SystemExit** — Wrapped each `py_rocc_t::custom*` trampoline in
  try/catch and added a `[[noreturn]] exit_from_systemexit(...)` helper
  that translates SystemExit to `std::exit(code)`, matching D-08's
  WJOIN exit semantics (`be91d2f`).

After the fix:
```
$ scripts/pyspike --extlib=riscv.gtx --extension=gtx -l --log=t.log \
    tests/gtx/data/elf/nop_wjoin.elf; echo $?
0
$ grep -c "warp\.join" t.log
1
```

Phase 2 test suite: **87 passed / 0 failed / 0 skipped / 0 xfailed**.

The original deferral note below is preserved for historical context.

---

### Original deferral note (logged 2026-05-05 by 02-06 gap-closure)

### Symptom

After Phase 2 lands and `_riscv.so` is built, running:

```bash
pyspike --extlib=riscv.gtx --log=trace.log tests/gtx/data/elf/nop_wjoin.elf
```

times out at 30s instead of exiting 0. Spike trace shows:

```
core   0: 0x00000000800000b0 (0x00001141) c.addi  sp, -16
core   0: 3 0x00000000800000b0 (0x1141) x2  0xfffffffffffffff0       <-- sp wraps from 0
core   0: 0x00000000800000b2 (0x0000502b) unknown
core   0: exception trap_illegal_instruction, epc 0x00000000800000b2
```

Two production-side issues are suspected:

1. **sp not initialized.** After `addi sp, sp, -16`, sp = `0xfffffffffffffff0`,
   meaning sp was 0 at entry — NOT `0x80100000` as `GtxNpu.reset()` writes via
   `proc.get_state().XPR.write(2, 0x80100000)`. Spike likely re-initializes XPR
   AFTER extension `reset()` runs, or sp init must hook a different lifecycle
   point.
2. **custom1 funct3=0b101 (WJOIN) returns illegal.** `find_extension('gtx')`
   resolves to the factory and `_RISCV_AVAILABLE` is true, but the instruction
   at `0x800000b2 0x0000502b` doesn't dispatch to `GtxNpu.custom1`. The
   `@isa.register('gtx')` decorator may produce a wrapper class
   (`MyISA`) that doesn't expose `custom1` to spike's RoCC trampoline through
   pybind11's `PYBIND11_OVERRIDE` chain.

### Reproduction

- `tests/gtx/test_skeleton.py::test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero` —
  fails with `pyspike timed out -- WJOIN SystemExit not propagating`.
- `tests/gtx/test_skeleton.py::test_full_trace_mnemonics_present` — fails for the
  same reason (no trace output to grep).
- Both gated on `_RISCV_AVAILABLE` + `nop_wjoin.elf` + `pyspike` on PATH; skip
  cleanly when env not ready.
- ELF fixture is correct (LOAD at `0x80000000`, entry `0x800000b0`, verified by
  `riscv64-unknown-elf-readelf -l`).

### Owner

pyspike core (`riscv.isa.register` + `py_rocc_t` trampoline integration). NOT the
GTX port — handler logic itself is verified by 86 unit tests on the mock-fallback
path.

### Investigation hooks

- Check whether spike re-initializes XPR after `extension_t::reset()`. Likely
  candidate: `processor_t::reset()` order in `vendor/spike/riscv/processor.cc`.
- Inspect what `@isa.register("gtx")` synthesizes. The decorator at
  `src/main/python/riscv/isa.py` produces `MyISA` (verified by Phase 2:
  `<class 'riscv.isa.register.<locals>.isa_decorator.<locals>.MyISA'>`).
  If `MyISA` doesn't override `custom0/1/2/3`, the wrapper steals dispatch
  before `GtxNpu` sees it.
- Check `py_module.cc` registration of `rocc_t` / `extension_t` — specifically
  whether `PYBIND11_OVERRIDE_PURE_NAME("custom1", ...)` chains through wrapper
  subclasses correctly.
- Compare with `examples/xhuimt/__init__.py` — does that example's RoCC dispatch
  actually run when invoked via `pyspike --extlib=...`?

### Why deferred from Plan 02-06 / Phase 2

Phase 2's scope was the GTX-side code (handlers, dispatch tables, disasm, tests).
The dispatch wiring failure is in pyspike's pybind11 trampoline / decorator
machinery, which CLAUDE.md explicitly forbids modifying ("no new C++ code").
Per the user's directive (2026-05-05), Categories A/B/C were fixed inline as
mechanical changes; Category D requires investigation that belongs with Phase 1
foundation.

### Acceptance for closure

- `pyspike --extlib=riscv.gtx --log=t.log nop_wjoin.elf; echo $?` outputs `0`
- `grep -cE '(wjoin|wrspr|rdspr)' t.log` returns ≥1 (or ≥3 with richer ELF
  fixture in P3+)
- Both `test_skeleton.py::test_pyspike_extlib_riscv_gtx_nop_wjoin_exits_zero`
  and `::test_full_trace_mnemonics_present` pass

**Recorded by:** 02-06 gap-closure follow-up (commits `107e646`, `87f8d2a`,
`8f75991` fixed Categories A/B/C; D logged here)
