# Codebase Structure: RoCC Extension Binding

**Analysis Date:** 2026-05-04

## Directory Layout

```
pyspike/
├── scripts/
│   ├── pyspike                          # CLI entry point wrapper
│   ├── spike                            # Wrapper for spike executable
│   └── lcov-report                      # Coverage report tool
├── src/
│   ├── main/
│   │   ├── cpp/
│   │   │   ├── riscv_extension.h        # Trampoline class definitions
│   │   │   ├── riscv_extension.cc       # Trampoline implementations
│   │   │   ├── py_module.cc             # pybind11 module definition
│   │   │   ├── py_bridge.h              # PythonBridge singleton
│   │   │   ├── py_bridge.cc             # PythonBridge implementation
│   │   │   ├── riscv_processor.h        # Processor binding
│   │   │   ├── riscv_csrs.h             # CSR binding
│   │   │   ├── riscv_cfg.h              # Config binding
│   │   │   ├── riscv_decode.h           # Decode binding
│   │   │   ├── riscv_disasm.h           # Disasm binding
│   │   │   └── [other bindings]
│   │   └── python/
│   │       └── riscv/
│   │           ├── __init__.py          # Module initialization
│   │           ├── __main__.py          # CLI entry (pyspike command)
│   │           ├── isa.py               # ISA/ROCC base classes
│   │           ├── _utils.py            # Library discovery functions
│   │           └── dev.py               # Dev utilities
│   └── test/
│       └── asm/                         # Assembly test code (spike upstream)
├── examples/
│   ├── xhuimt/                          # HuiMt custom ISA extension
│   │   ├── __init__.py                  # HuiMtISA class + @register
│   │   ├── mylrsc.py                    # LR/SC instruction implementation
│   │   ├── mycsrs.py                    # Custom CSR definitions
│   │   └── arg.py                       # Custom operand formatters
│   ├── xthead/                          # T-Head custom ISA extension
│   │   ├── __init__.py                  # TheadISA class + @register
│   │   ├── theadba.py                   # th.addsl instruction
│   │   └── arg.py                       # Custom operand formatters
│   └── amba/                            # AMBA device examples
│       ├── uart_lite.py                 # UART device interface
│       └── uart_lite_impl.py            # UART implementation
├── tests/
│   ├── test_extension.py                # Test registration & lifecycle
│   └── data/                            # Test data/binaries
└── vendor/
    └── spike/                           # Upstream spike submodule (out of scope)
```

## Directory Purposes

**scripts/:**
- **Purpose:** CLI wrappers for user-facing commands
- **Contains:** Executable entry points
- **Key file:** `pyspike` — parses `--extlib`, sets `PYSPIKE_LIBS` env, invokes spike

**src/main/cpp/:**
- **Purpose:** pybind11 bindings and C++ layer
- **Contains:** Trampoline classes, pybind11 module definition, Python bridge
- **Core RoCC files:** `riscv_extension.h/cc`, `py_module.cc`, `py_bridge.h/cc`

**src/main/python/riscv/:**
- **Purpose:** User-facing Python API and module initialization
- **Contains:** Abstract base classes, decorators, library discovery
- **Key files:** `isa.py` (base classes), `__init__.py` (module init), `__main__.py` (CLI)

**tests/:**
- **Purpose:** Unit and integration tests
- **Contains:** Test cases for registration, extension lifecycle, mocking
- **Key file:** `test_extension.py` — tests both built-in and Python-registered extensions

**examples/:**
- **Purpose:** Reference implementations of RoCC extensions
- **Pattern:** Each subdirectory is a complete extension package
- **Subdirs:** `xhuimt/` (instructions+CSRs), `xthead/` (instructions), `amba/` (devices)

**vendor/spike/:**
- **Purpose:** Upstream Spike RISC-V simulator source (Git submodule)
- **Not mapped:** Out of scope per requirements
- **References in binding:** Only via header includes (e.g., `#include <riscv/rocc.h>`)

## Key File Locations

| File | Role | Key Symbols |
|------|------|-------------|
| `src/main/cpp/riscv_extension.h` | Trampoline class definitions | `py_rocc_t`, `py_extension_t`, `py_register_extension()` |
| `src/main/cpp/riscv_extension.cc` | Trampoline implementations | `py_rocc_t::custom0/1/2/3()`, `py_extension_t::get_instructions/disasms/csrs()`, `py_register_extension()` |
| `src/main/cpp/py_module.cc` | pybind11 module binding | `PYBIND11_MODULE(_riscv)`, rocc_t binding (lines 431–447), extension_t binding (lines 411–429) |
| `src/main/cpp/py_bridge.h` | PythonBridge singleton decl | `PythonBridge::getInstance()`, `track<T>()`, `bootstrap()` |
| `src/main/cpp/py_bridge.cc` | PythonBridge implementation | Module import logic (lines 35–53), Python interpreter init (lines 23–29) |
| `src/main/python/riscv/__init__.py` | Module initialization | Loads Spike library, imports _riscv submodule, bootstraps |
| `src/main/python/riscv/__main__.py` | CLI entry point | `main()`, `entry()`, argument parsing, spike invocation |
| `src/main/python/riscv/isa.py` | Extension base classes & decorators | `ISA` class, `ROCC` class, `register()` decorator, `arg()` decorator |
| `src/main/python/riscv/_utils.py` | Library discovery | `find_spike_library()`, `find_bridge_library()`, `find_python_library()`, `load_spike_library()` |
| `scripts/pyspike` | CLI wrapper | Shell script that bootstraps sys.path and calls riscv.__main__.entry() |
| `tests/test_extension.py` | Extension tests | Test classes `MyCFlush`, `MyDummyROCC`; test functions `test_register_extension()`, `test_find_extension()` |
| `examples/xhuimt/__init__.py` | HuiMt extension class | `HuiMtISA` class, `@isa.register("huimt")` decorator |
| `examples/xhuimt/mylrsc.py` | LR/SC instruction handler | `MyLRSC` class, `get_instructions()`, `get_disasms()`, LR/SC handlers |
| `examples/xthead/__init__.py` | T-Head extension class | — (to be mapped) |
| `examples/xthead/theadba.py` | th.addsl instruction | `TheadBa` class, `_do_th_addsl()` handler |

## Naming Conventions

**File Names:**
- Python extensions: `lowercase_with_underscores.py` (e.g., `mylrsc.py`, `mycsrs.py`)
- C++ headers: `lowercase_with_underscores.h` (e.g., `riscv_extension.h`, `py_bridge.h`)
- C++ sources: `lowercase_with_underscores.cc` (e.g., `riscv_extension.cc`, `py_bridge.cc`)
- Entry scripts: `lowercase` no extension (e.g., `pyspike`, `spike`)

**Classes/Types:**
- Python base classes: `CamelCase`, suffixed with descriptor (e.g., `ISA`, `ROCC`, `MyLRSC`)
- C++ trampoline classes: `py_` prefix + class name (e.g., `py_rocc_t`, `py_extension_t`)
- Internal C++ helper classes: `_` prefix or normal name based on scope (e.g., `PythonBridge`)
- Extension-specific classes: Descriptive names (e.g., `TheadBa`, `HuiMtISA`)

**Functions/Methods:**
- Python: `snake_case` (e.g., `get_instructions()`, `register_extension()`)
- C++: `snake_case` (e.g., `py_register_extension()`, `find_spike_library()`)
- Decorators: `snake_case` (e.g., `@register()`, `@arg()`)
- Handlers/Callbacks: `_do_<operation>` (e.g., `_do_th_addsl()`, `_do_lr_32()`)

**Environment Variables:**
- `PYSPIKE_LIBS` — colon-separated paths to Python extension modules (set by `__main__.py`, read by `py_bridge.cc`)
- `PYSPIKE_EXTS` — reserved for future use (not currently implemented)
- `RISCV` — Path to RISC-V toolchain (used by `_utils.py` as fallback)

**File Extensions:**
- Python: `.py`, `.pyc`
- C++: `.cc` (source), `.h` (header)
- Shared objects: `.so` (Linux), `.dylib` (macOS), `.pyd` (Windows)
- Test data: `.bin`, `.elf`, `.hex` (in `tests/data/`)

## Where to Add New Code

**New RoCC Extension (e.g., `examples/myext/`):**

1. **Create directory:** `examples/myext/`
2. **Main module:** `examples/myext/__init__.py`
   - Subclass `riscv.isa.ROCC` (for RoCC) or `riscv.isa.ISA` (for regular extension)
   - Implement: `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()`, `set_debug()`
   - For RoCC: Implement `custom0()`, `custom1()`, etc. or leave default
   - Decorate with `@riscv.isa.register("myext_name")`

3. **Instruction handlers:** `examples/myext/myhandlers.py` (optional)
   - Create classes or functions implementing `insn_desc_t` handlers
   - Methods: `handler(proc: processor_t, insn: insn_t, pc: int) -> int`
   - Return: next PC after instruction execution

4. **Custom operand formatters:** `examples/myext/arg.py` (optional)
   - Create `arg_t` subclasses for disassembly operand formatting
   - Use `@riscv.isa.arg(lambda insn: ...)` decorator for simple lambdas

5. **CSR definitions:** `examples/myext/csrs.py` (optional)
   - Subclass `riscv.csrs.csr_t` for custom CSRs
   - Return from `get_csrs()`

6. **Usage:** `pyspike --extlib=examples/myext myprogram`

**New C++ Binding (for new Spike types):**

1. **Trampoline class:** Add to `src/main/cpp/riscv_extension.h` if method overriding needed
   - Subclass Spike type and `pybind11::trampoline_self_life_support`
   - Implement virtual methods using `PYBIND11_OVERRIDE` macro

2. **Implementation:** Add to `src/main/cpp/riscv_extension.cc`
   - Use `pybind11::get_override()` to dispatch to Python if overridden

3. **pybind11 binding:** Add to `src/main/cpp/py_module.cc`
   - Create `py::class_<MyType, PyMyType, py::smart_holder>(mod, "MyType")`
   - Bind public methods and properties

**New Test:**

1. **File:** `tests/test_<feature>.py` following existing patterns
2. **Pattern:** Use `pytest` fixtures (see `test_extension.py` for `mock_sim` fixture)
3. **Extension tests:** Create test extension class (e.g., `MyTestExt(riscv.isa.ISA)`), register with `register_extension()`, then verify behavior

**Utilities:**

- **Shared helpers:** `src/main/python/riscv/_utils.py` or `src/main/python/riscv/_helpers.py`
- **Shared device base:** `src/main/python/riscv/devices.py` (if extending beyond pybind11 bindings)

## Special Directories

**vendor/spike/:**
- **Purpose:** Upstream Spike submodule
- **Generated:** No (checked in as Git submodule)
- **Committed:** Yes
- **Note:** Out of scope for RoCC binding analysis; referred to via `#include <riscv/rocc.h>` only

**build/:**
- **Purpose:** CMake build artifacts
- **Generated:** Yes (build time)
- **Committed:** No (in `.gitignore`)
- **Contains:** Object files, shared libraries (`_riscv.so`), executables

**tests/data/:**
- **Purpose:** Test binaries and reference files
- **Generated:** Some (compiled test programs)
- **Committed:** Some (expected output files)

**.pytest_cache/, __pycache__/:**
- **Purpose:** Python runtime caches
- **Generated:** Yes (runtime)
- **Committed:** No (in `.gitignore`)

---

*Structure analysis: 2026-05-04*
