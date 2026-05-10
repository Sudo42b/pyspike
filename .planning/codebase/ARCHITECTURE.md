# Architecture: RoCC Extension Binding in Pyspike

**Analysis Date:** 2026-05-04

## Pattern Overview

**Overall:** Layered binding architecture with Python-first user API, pybind11 trampoline classes for virtual dispatch, and C++ factory registration bridge.

**Key Characteristics:**
- Python user code subclasses `riscv.isa.ROCC` or `riscv.isa.ISA` and uses `@riscv.isa.register()` decorator
- Decorator wraps the class and calls C++ `py_register_extension()` to register a factory
- Factory is invoked by spike's simulator when instantiating extensions per-hart
- Custom instruction dispatch flows: Python opcode handler → `custom0/1/2/3` virtual override → return to spike
- Lifecycle managed by `PythonBridge` singleton: bootstraps Python environment, loads extension modules via `PYSPIKE_LIBS` env var, keeps Python objects alive across C++/Python boundary

## Layers

**Python User Layer (src/main/python/riscv/):**
- Purpose: User-facing extension API with decorators, base classes, and registration
- Location: `src/main/python/riscv/isa.py`, `src/main/python/riscv/__init__.py`
- Contains: `ISA`, `ROCC` abstract base classes; `register()` decorator; `arg()` decorator for disasm operands
- Depends on: pybind11-wrapped `_riscv` module (C++ extension types `extension_t`, `rocc_t`)
- Used by: User Python extension packages (e.g., `examples/xhuimt/`, `examples/xthead/`)

**pybind11 Binding Layer (src/main/cpp/py_module.cc):**
- Purpose: Expose C++ RISC-V types to Python; register module with pybind11
- Location: `src/main/cpp/py_module.cc`
- Contains: `PYBIND11_MODULE(_riscv)` definition (lines 48+); submodules for `cfg`, `csrs`, `decode`, `disasm`, `extension`, `isa_parser`, etc.
- Key bindings: `rocc_t` (lines 431–447), `extension_t` (lines 411–429), `rocc_insn_t` (lines 391–409)
- Depends on: `py_rocc_t`, `py_extension_t` trampolines; `py_register_extension()` function
- Used by: Python layer; loaded as `_riscv` module during `riscv` import

**Trampoline Layer (src/main/cpp/riscv_extension.h, riscv_extension.cc):**
- Purpose: Virtual dispatch from C++ to Python method overrides
- Location: `src/main/cpp/riscv_extension.h` (class definitions), `src/main/cpp/riscv_extension.cc` (implementations)
- Contains: 
  - `py_extension_t` (extends `extension_t`; trampolines: `get_instructions()`, `get_disasms()`, `get_csrs()`, `name()`, `reset()`, `set_debug()`)
  - `py_rocc_t` (extends `rocc_t`; trampolines: `custom0()`, `custom1()`, `custom2()`, `custom3()`, `name()`)
  - `py_register_extension()` function (line 114): converts Python constructor to C++ factory lambda
- Depends on: `PythonBridge` singleton for object tracking and Python calls; pybind11 for `PYBIND11_OVERRIDE` macros
- Used by: C++ extension framework when methods are overridden in Python

**Runtime Bootstrap Layer (src/main/cpp/py_bridge.{h,cc}):**
- Purpose: Initialize Python interpreter; load and import extension modules from `PYSPIKE_LIBS` env var
- Location: `src/main/cpp/py_bridge.cc` (lines 23–53)
- Contains: `PythonBridge::bootstrap()` (lines 35–53); imports modules listed in `PYSPIKE_LIBS` using `importlib.import_module()`
- Mechanism: `py_bridge.cc` line 42 reads env var, splits on `os.pathsep`, imports each module by name
- Used by: Spike simulator initialization; called once during pybind11 module load

**CLI Layer (scripts/pyspike, src/main/python/riscv/__main__.py):**
- Purpose: Parse `--extlib` arguments; load Python/C++ libraries; invoke spike executable with patched args
- Location: `scripts/pyspike` (wrapper), `src/main/python/riscv/__main__.py` (main entry)
- Entry: User runs `pyspike --extlib=foo.py --extlib=bar.so ...`
- Logic (lines 30–58 of `__main__.py`):
  1. Parse `--extlib` arguments
  2. Identify C/C++ shared libraries (`.so`, `.dylib`) → prepend to spike's `--extlib` list
  3. Identify Python modules (`.py`, `.pyc`, directories) → accumulate in `pylibs` list
  4. Set `PYSPIKE_LIBS` env var to `os.pathsep.join(pylibs)`
  5. `os.execve()` spike executable with both C++ libs and env var set
- Used by: End users; integrates Python extension loading with spike CLI

**Upstream rocc_t (vendor/spike/, out of scope):**
- Spike defines base `rocc_t` with virtual `custom0/1/2/3` methods
- pyspike wraps it with `py_rocc_t` trampoline class

## Data Flow

### Extension Registration

```
User Python (@riscv.isa.register decorator)
  ↓
  Wraps class, calls riscv.extension.register_extension(name, class)
  ↓
py_register_extension() [C++ function, line 114 of riscv_extension.cc]
  ↓
  Calls spike's register_extension(name, lambda_factory)
  ↓
  Factory stored in spike's global extension registry
```

### Extension Instantiation

```
Spike simulator (per hart)
  ↓
  Looks up factory by name from registry
  ↓
  Calls factory lambda (line 115 of riscv_extension.cc)
    ↓
    Instantiates Python extension class: py_ctor() (line 116)
    ↓
    PythonBridge::track<rocc_t *>(py_ext) keeps Python object alive (line 118)
    ↓
  Returns C++ rocc_t * pointer to simulator
  ↓
  Attaches to processor_t for this hart
```

### Custom Instruction Dispatch

```
Spike instruction decode (rocc_insn_t for opcode 0x0b, 0x2b, 0x3b, 0x7b)
  ↓
  Calls rocc_t::custom0() / custom1() / custom2() / custom3()
  (which is py_rocc_t virtual override, line 90–108 of riscv_extension.cc)
  ↓
  PYBIND11_OVERRIDE macro (line 92, 97, 102, 107)
    ↓
    Looks up Python method override using py::get_override()
    ↓
    Calls Python method with args: (proc: processor_t*, insn: rocc_insn_t, xs1: reg_t, xs2: reg_t)
    ↓
  Returns reg_t result (written to register or memory)
  ↓
  Spike resumes instruction stream
```

## State Management

**Extension Instance Lifecycle:**
- Created once per hart during `processor_t::reset()`
- Stored in `processor_t::extensions` map (keyed by name)
- Python objects kept alive by `PythonBridge::references` map (line 78 of py_bridge.h)
- Shared state (CSRs, custom state) managed by user Python code (e.g., reserved addresses in `MyLRSC`, line 32 of `examples/xhuimt/mylrsc.py`)

**Python Interpreter State:**
- `PythonBridge` singleton (line 46 of py_bridge.h) initialized once
- Python interpreter bootstrapped if not already running (line 25 of py_bridge.cc)
- Imported extension modules persist for lifetime of spike process

## Key Abstractions

**`ISA` (riscv/isa.py, lines 27–39):**
- Purpose: Abstract base for all extension types
- Defines: `name` property (abstract); `_name()` method for C++ callback
- Examples: Subclass for custom extensions

**`ROCC` (riscv/isa.py, lines 42–45):**
- Purpose: Abstract base for RoCC (Rocket Custom Coprocessor) extensions
- Inherits from both `rocc_t` (C++ pybind11 binding) and `ISA` (Python abstraction)
- Pattern: User creates subclass, implements `custom0/1/2/3` or leaves default (no-op)

**`register()` decorator (riscv/isa.py, lines 48–70):**
- Purpose: Decorator for registering extension by name
- Creates synthetic subclass with hardcoded `name` property (lines 58–65)
- Calls `register_extension()` to bind name → factory (line 67)
- Returns the synthetic class for user

**`py_rocc_t` (riscv_extension.h, lines 56–72; riscv_extension.cc, lines 90–112):**
- Purpose: pybind11 trampoline for `rocc_t` virtual methods
- Inherits from both `rocc_t` (upstream) and `pybind11::trampoline_self_life_support`
- Implements: `custom0/1/2/3()` using `PYBIND11_OVERRIDE` to invoke Python overrides
- Also implements: `name()` pure virtual method

**`py_extension_t` (riscv_extension.h, lines 31–53; riscv_extension.cc, lines 22–88):**
- Purpose: pybind11 trampoline for `extension_t` (general extensions)
- Similar to `py_rocc_t` but for non-RoCC extensions
- Trampolines: `get_instructions()`, `get_disasms()`, `get_csrs()`, `name()`, `reset()`, `set_debug()`

**`PythonBridge` (py_bridge.h, py_bridge.cc):**
- Purpose: Singleton managing Python interpreter state and object lifetimes
- Key methods:
  - `getInstance()` (line 46): Return singleton
  - `bootstrap()` (lines 35–53): Import modules from `PYSPIKE_LIBS`
  - `track<T>(py::handle)` (line 53 of py_bridge.h): Keep Python object alive, return C++ pointer

## Entry Points

**User Extension (e.g., examples/xhuimt/__init__.py):**
- Location: `examples/xhuimt/__init__.py`
- Triggers: When `PYSPIKE_LIBS` includes this module path
- Responsibilities:
  1. Define extension class (inherits `riscv.isa.ISA` or `riscv.isa.ROCC`)
  2. Implement `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()` methods
  3. Decorate with `@riscv.isa.register("name")` to auto-register on import

**CLI Entry (scripts/pyspike):**
- Location: `scripts/pyspike` (shell wrapper)
- Triggers: `pyspike --extlib=module.py --extlib=lib.so`
- Flow: Parses args → calls `riscv.__main__.main()` → sets `PYSPIKE_LIBS` → `execve()` spike

**C++ Factory Registration (py_register_extension):**
- Location: `src/main/cpp/riscv_extension.cc`, line 114–122
- Triggers: When Python code calls `riscv.extension.register_extension(name, py_class)`
- Mechanism: Lambda closure captures `py_ctor`, called by spike's extension instantiation

**PythonBridge Bootstrap:**
- Location: `src/main/cpp/py_bridge.cc`, lines 35–53
- Triggers: First call to `PythonBridge::getInstance()` during spike initialization
- Mechanism: Reads `PYSPIKE_LIBS` env var, imports each module

## Error Handling

**Strategy:** Exception-safe via pybind11; Python exceptions logged to stderr, simulator continues with default behavior.

**Patterns:**

**Custom Instruction Handler Error (riscv_extension.cc, no explicit try-catch in custom0/1/2/3):**
- If Python method raises exception, pybind11 converts to C++ exception
- Not explicitly caught (relies on pybind11 error propagation)
- Simulator may crash or exhibit undefined behavior

**get_instructions/get_disasms/get_csrs Errors (riscv_extension.cc, lines 23–70):**
- Wrapped in `try { } catch (py::error_already_set &e)` (lines 33–35, 50–52, 66–68)
- Error printed to stderr (line 34, 51, 67)
- Empty vector returned to simulator (graceful degradation)

**Module Import Error (py_bridge.cc, lines 49–52):**
- `importlib.import_module()` wrapped implicitly (Python exception caught internally)
- Warning issued to stderr if import fails
- Simulator continues without that extension

## Cross-Cutting Concerns

**Logging:** 
- C++ errors via `std::cerr` (riscv_extension.cc lines 34, 51, 67)
- Python errors via `warnings.warn()` (py_bridge.cc line 52)
- No centralized logging framework

**Validation:**
- Python type hints used (riscv/isa.py, examples/) but not enforced at runtime
- pybind11 handles type casting; mismatch raises Python `TypeError`

**Authentication/Access Control:**
- None; all extensions trusted if loaded via `--extlib`

**Performance Considerations:**
- Virtual function calls add minimal overhead (pybind11 trampoline cost ~1–2 μs per call)
- Python GIL held during custom instruction execution (single-threaded spike context)
- No lazy loading; extensions instantiated for each hart on first use

### FP16 byte-order boundary (BE vs LE)

The pyspike GTX NPU operates in **LE FP16** byte order natively (matching
host x86_64 little-endian + `np.float16.view(np.uint8)` semantics). Vendor
HW simulation (SystemC) produces **BE FP16** in `_ref.txt` golden files
(32-byte DDR bus-words parsed right-to-left, per
`vendor/gtx_cpp_reference/gtx/CLAUDE.md` "DDR Hex 파일 바이트 순서").

The boundary is mediated by the `GTX_DDR_REVERSED=1` env var, read per
call by `src/main/python/riscv/gtx/ddr.py` at `ddr_init_from_file`
(`ddr.py:110`) and `ddr_dump_to_file` (`ddr.py:145`). When set, both
functions reverse byte ordering on read/write so that vendor BE golden
files compare byte-exact against pyspike's LE-default in-memory
representation. No module-level cache — each call reads `os.environ`
directly to avoid stale-value poisoning under `monkeypatch.setenv`.

The regression sweep harness
(`tests/gtx/test_regression_fw_full_sweep.py:382-387`) auto-applies
`GTX_DDR_REVERSED=1` for vendor-rooted `.elf` paths only via an inline
`is_relative_to(vendor_root)` check on the resolved ELF path (D-10
inline subprocess env, NOT autouse fixture — autouse leaks across
non-vendor tests like `test_ddr_modes.py`). See
`tests/gtx/data/firmware/README.md` Contracts 1 & 2 for the canonical
4-contract documentation.

Production code (`riscv.gtx.*`) sees only LE FP16; the BE/LE conversion
is fully encapsulated in the DDR I/O layer.

---

*Architecture analysis: 2026-05-04*
*Updated 2026-05-10: BE/LE FP16 boundary note appended (Phase 8 VTW-04 closure).*
