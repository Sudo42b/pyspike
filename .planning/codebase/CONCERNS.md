# Codebase Concerns: RoCC Binding Layer

**Analysis Date:** 2026-05-04

## Critical Gaps

### Test Coverage for RoCC Virtual Overrides

**Issue:** No end-to-end tests of `custom0`, `custom1`, `custom2`, `custom3` methods
- **Files:** `src/main/cpp/riscv_extension.cc` (lines 90-108), `tests/test_extension.py` (lines 54-58, 92)
- **Risk:** High — PYBIND11_OVERRIDE macros in `py_rocc_t` are untested. Python overrides may not work correctly at runtime due to pybind11 bridging issues.
- **Symptom:** A Python RoCC subclass that overrides `custom0` could silently use the base class stub (returning 0) instead of the Python implementation.
- **Impact:** Users cannot implement custom RoCC instructions in Python; binding is broken but undetected.
- **Test data:** Test suite has `MyDummyROCC` (line 54, 92) but it does not override any custom methods — it only inherits the base stubs.
- **Safe modification:** 
  1. Create `TestROCC(isa.ROCC)` that overrides `custom0..3` with meaningful implementations
  2. Register it and exercise with RoCC instructions (0x0b, 0x2b, 0x5b, 0x7b opcodes)
  3. Verify return values flow through to Python side correctly
- **Priority:** High — Core API contract is unvalidated

---

### No Examples of Custom RoCC in Python

**Issue:** README and examples show ISA extensions but no RoCC coprocessor example
- **Files:** `/mnt/e/14_NIGHTLY/pyspike/README.md` (lines 47-65), `examples/` directory
- **Risk:** Medium — Users have no working reference. RoCC binding API surface is opaque; docstrings in `src/main/python/riscv/isa.py` (lines 42-45) are minimal.
- **What's exposed:** `riscv.isa.ROCC` class (line 42 in isa.py) is a blank trampoline inheriting from `rocc_t` and `ISA`. No guidance on overriding `custom0..3`.
- **Missing documentation:**
  - How to define `custom0..3(self, processor_t, rocc_insn_t, xs1, xs2) -> reg_t`
  - How instruction operands map to `rocc_insn_t.rd`, `.xs1`, `.xs2`, etc.
  - When state writes happen (before/after custom method returns?)
  - Example of reading/writing processor registers from Python RoCC
- **Safe modification:** Create `examples/rocc/simple_counter.py` demonstrating a minimal RoCC that increments a counter on custom0 and returns it.
- **Priority:** Medium — Blocks adoption but not functionality

---

## Fragility Points

### Upstream API Dependency: `rocc.h` Interface

**Issue:** pyspike's `py_rocc_t` trampoline is tightly coupled to spike's `rocc_t` virtual interface
- **Files:** `src/main/cpp/riscv_extension.h` (lines 56-72), `vendor/spike/riscv/rocc.h` (lines 24-33)
- **Breakage risk:** High — Spike API is not stable. Historical precedent:
  - Commit `42c1ebb` ("bumped upstream spike to 591cff16"): Required edits to `src/main/cpp/py_module.cc` (16 lines changed, 9 deleted) — churn in the binding layer
  - Upstream recently removed fields like `p_imm2..p_imm6` and `start_pc_t` (noted in recent bindings cleanup)
  - `rocc_t::custom0..3` signatures have not changed, but they could (e.g., adding return type, parameter changes)
- **What could break:**
  1. Signature change: e.g., `custom0(processor_t *, rocc_insn_t, reg_t xs1, reg_t xs2) -> reg_t` → `custom0(...) -> void` or adds parameters
  2. New virtual: Spike adds `custom4()` or removes `name()` override requirement
  3. Base class change: `rocc_t` inherits from something other than `extension_t`
- **Detection:** Build will fail with pybind11 override errors or C++ compile errors, but then pyspike is stuck in non-buildable state until upstream is matched.
- **Mitigation:** Version pin (not yet done). `pyproject.toml` / `setup.py` should enforce compatible Spike version range (currently bundled, no version check).
- **Safe modification:** Add version check in `src/main/cpp/py_module.cc` initialization to assert Spike API level matches expectation.
- **Priority:** High — Entire binding could silently break on upstream bump

---

### Pybind11 Trampoline + Virtual Interface Interaction

**Issue:** `py_rocc_t` inherits from two bases with virtual methods (`rocc_t` and `trampoline_self_life_support`)
- **Files:** `src/main/cpp/riscv_extension.h` (lines 56-72), `src/main/cpp/riscv_extension.cc` (lines 90-108)
- **Visibility warning:** During build (noted in scope), compiler warns: `py_rocc_t declared with greater visibility than its base 'pybind11::trampoline_self_life_support'`
  - This warning indicates potential memory management or vtable lookup issues
  - Harmless in practice (visibility attrs are ignored in GCC-style ELF), but signals template complexity
- **Hidden assumption:** pybind11's `trampoline_self_life_support` assumes sole ownership of object lifetime. If `py_rocc_t` is ever stored in `processor_t`'s extension list and later accessed via C++ as raw `rocc_t*`, the trampoline metadata may be inaccessible.
- **Pattern risk:** Lines 90-108 of `riscv_extension.cc` use `PYBIND11_OVERRIDE(reg_t, rocc_t, custom0, ...)` — this macro acquires the Python GIL and calls back into Python. If spike's `processor_t::step()` calls `custom0()` from a hot loop, GIL contention could serialize execution.
- **Safe modification:** Add comment explaining that PYBIND11_OVERRIDE implicitly holds GIL during custom* calls. If performance becomes critical, consider py::call_guard<py::gil_scoped_release> to release between instruction execution (requires careful thread safety review).
- **Priority:** Medium — Architectural, not immediately breaking

---

### Lifetime / Ownership Model Undefined

**Issue:** Who owns `py_rocc_t` instances? Python side or spike's `processor_t`?
- **Files:** `src/main/cpp/riscv_extension.cc` (lines 114-122), `src/main/cpp/py_bridge.h` (lines 53-80), `src/main/cpp/py_module.cc` (lines 431-447)
- **Pattern:** `py_register_extension()` (lines 114-122) creates a lambda that calls `py_ctor()`, casts to `rocc_t*`, and tracks via `PythonBridge::track()` (line 118).
  ```cpp
  void py_register_extension(const std::string &name, py::function py_ctor) {
    register_extension(name.c_str(), [py_ctor]() -> extension_t * {
      auto py_ext = py_ctor();  // Python object created
      if (py::isinstance<rocc_t>(py_ext)) {
        return PythonBridge::getInstance().track<rocc_t *>(py_ext);
      }
      return PythonBridge::getInstance().track<extension_t *>(py_ext);
    });
  }
  ```
- **Ownership hole:** 
  - `PythonBridge::track()` increments refcount (line 76: `py_obj.inc_ref()`) and stores handle
  - But when does refcount decrement? `references` map (line 87) is never cleared in destructor.
  - If Python GC runs before C++ side is done with the RoCC, use-after-free is possible (though pybind11's smart_holder mitigates via holder types)
- **Double-free risk:** If user keeps a Python reference and also passes to `processor_t::register_extension()`, object lifetime becomes ambiguous. Explicit documentation needed.
- **Safe modification:** 
  1. Add assertion in `py_rocc_t` constructor that `this` is heap-allocated (pybind11 smart_holder guarantee)
  2. Document in README: "RoCC instances are owned by the simulator; do not rely on Python references after passing to `register_extension()`"
- **Priority:** Medium — Potential for silent heap corruption under edge cases

---

### Missing API Surface: RoCC Reset and Teardown

**Issue:** `rocc_t` interface includes `reset()` (inherited from `extension_t`) and potentially `get_instructions()`, `get_csrs()`, but no explicit cleanup hook
- **Files:** `src/main/cpp/riscv_extension.h` (lines 56-72), `vendor/spike/riscv/rocc.h`, `src/main/cpp/py_module.cc` (lines 431-447)
- **Exposed in bindings:** `extension_t::reset()` is exposed (line 420 in py_module.cc). `rocc_t::custom0..3` are exposed (lines 435-442).
- **Not exposed:** 
  - Custom CSR access methods (if rocc_t defines any)
  - Reset behavior specific to RoCC state
  - Any finalization hook
- **Impact:** If a Python RoCC stores state in instance variables, `reset()` must be overridden to clear them. Current test (line 54 of test_extension.py: `MyDummyROCC`) has no state and no reset override, so this is never exercised.
- **Safe modification:** Document that Python RoCC should override `reset()` if it maintains per-processor state. Add test case.
- **Priority:** Low — Not a blocker, but state management is implicit

---

## GIL / Threading Concerns

### PYBIND11_OVERRIDE Holds GIL During Instruction Execution

**Issue:** Each RoCC custom instruction (`custom0..3`) crosses from C++ into Python via `PYBIND11_OVERRIDE`, which holds the GIL
- **Files:** `src/main/cpp/riscv_extension.cc` (lines 90-108)
- **Pattern:** 
  ```cpp
  reg_t py_rocc_t::custom0(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) {
    PYBIND11_OVERRIDE(reg_t, rocc_t, custom0, proc, insn, xs1, xs2);
  }
  ```
- **Risk:** If spike's execution loop is multithreaded (multiple processors), GIL contention will serialize all RoCC instruction execution. Not immediately a problem for single-core simulation, but blocks multi-core scalability.
- **Observation:** No explicit `py::call_guard` or `py::gil_scoped_release` in the bindings. The GIL is held throughout `custom0..3` execution.
- **Mitigation path:** Not needed now, but document as a future scaling limitation. If multi-processor simulation becomes critical, consider:
  1. Move GIL release to outer loop (before instruction dispatch)
  2. Use thread-local storage to track which processor is currently executing
  3. Only acquire GIL when entering Python, release immediately after return
- **Safe modification:** Add comment in `riscv_extension.cc` lines 90-108 noting GIL behavior.
- **Priority:** Low — Future concern, not current issue

---

## Dependency Versioning

### No Explicit Spike Version Pinning

**Issue:** pyspike bundles Spike via git submodule (`vendor/spike/`) but does not validate version match at runtime
- **Files:** `src/main/cpp/py_module.cc` (initialization), `setup.py` (lines 98-111 build Spike), `vendor/spike/` (submodule)
- **Risk:** Medium — User may have old wheel built with Spike commit X, upgrade repo, rebuild with Spike commit Y, and encounter silent API mismatches if RoCC interface changed.
- **Example:** Commit `42c1ebb` bumped Spike to `591cff16`, requiring binding changes. Commit `5d4348e` bumped to `20feb9c2`. No version check prevents accidental mismatches.
- **Safe modification:** 
  1. Add Spike commit hash to version string (already done in setup.py lines 66-77, local_scheme)
  2. At runtime, check that Spike's version string matches expected hash. Add assertion in `py_module.cc` module initialization.
- **Priority:** Medium — Prevents silent breakage

---

## API Design Issues

### Incomplete RoCC Instruction Information in Binding

**Issue:** `rocc_insn_t` is exposed as read-only struct with properties (py_module.cc lines 391-409), but instruction decoding is opaque
- **Files:** `src/main/cpp/py_module.cc` (lines 391-409), `vendor/spike/riscv/rocc.h` (lines 6-16)
- **Exposed fields:** `opcode`, `rd`, `xs2`, `xs1`, `xd`, `rs1`, `rs2`, `funct` (all read-only properties)
- **User perspective:** A Python RoCC subclass receives `rocc_insn_t` with these fields set. But full instruction encoding (immediate values, etc.) is not exposed.
- **Risk:** Medium — If RoCC instruction format requires immediates or custom bit fields, user must manually parse raw instruction bits (not exposed either). The define_custom_func macro in spike's rocc.h shows how upstream does it, but this pattern is not replicated in Python.
- **Impact:** Python RoCC implementations are limited to simple fixed-format instructions. Any instruction with embedded immediates requires C++ extension.
- **Safe modification:** Expose raw instruction bits in rocc_insn_t binding (similar to insn_t.bits property in py_module.cc line 239).
- **Priority:** Medium — Limits feature expressiveness but not correctness

---

## Build-Time Issues

### Visibility Warning Not Documented

**Issue:** Compiler warning during build about `py_rocc_t` visibility
- **Files:** `src/main/cpp/riscv_extension.h` (lines 56-57)
- **Warning message (observed):** 
  ```
  py_rocc_t declared with greater visibility than its base 'pybind11::trampoline_self_life_support'
  ```
- **Root cause:** `py_rocc_t` is implicitly public visibility (no `__attribute__` or pragma), inheriting from pybind11 trampoline which may have stricter visibility.
- **Assessment:** Harmless in GCC/Clang ELF (visibility attrs mainly for DSO symbol export). But warning indicates potential portability issue on other platforms (e.g., Apple Clang with different visibility defaults).
- **Safe modification:** Add explicit `__attribute__((visibility("default")))` to `py_rocc_t` class definition, or suppress warning with pragma comment.
- **Priority:** Low — Cosmetic, but improves build cleanliness

---

## Summary of Fix Priorities

| Area | Priority | Effort | Impact |
|------|----------|--------|--------|
| RoCC virtual override test coverage | High | Medium | Unvalidated API contract |
| Upstream API dependency version pinning | High | Low | Prevent silent breakage |
| RoCC example in Python | Medium | Low | Unblock adoption |
| Lifetime/ownership documentation | Medium | Low | Prevent use-after-free |
| GIL/threading comments | Low | Minimal | Clarify scalability limits |
| Visibility warning cleanup | Low | Minimal | Build hygiene |

---

*Concerns audit: 2026-05-04*
