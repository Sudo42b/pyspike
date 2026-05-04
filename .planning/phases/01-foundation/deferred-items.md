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
