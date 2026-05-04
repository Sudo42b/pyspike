# External Integrations

**Analysis Date:** 2026-05-04

## APIs & External Services

**Upstream Spike (RISC-V ISA Simulator):**
- Spike library (libriscv.so) - Provides base classes and instruction execution infrastructure
  - SDK/Client: `-lriscv` linked at build time (`setup.py:55`)
  - Headers: rocc.h, extension.h, processor.h from vendor/spike/riscv/
  - Auth: None (built from source via `_build_spike()` in `setup.py:98-111`)

**Dynamic Extension Loading:**
- libcustomext.so (optional) - Upstream spike custom extension library for testing
  - Used in: `tests/test_extension.py:66` (ctypes.util.find_library check)
  - Auth: None

## Data Storage

**None used** - RoCC extensions do not interface with persistent storage or databases. All state is in-memory within processor_t and rocc_t instances.

## File Storage

**Local filesystem only:**
- RISC-V toolchain installation at `${RISCV:-/opt/riscv}/` (read-only during build and runtime)
- Compiled spike libraries bundled in wheel at `riscv/data/lib/` (libriscv.so, libdisasm.a, libfesvr.a)

## Caching

None detected.

## Authentication & Identity

**Not applicable** - No user authentication or identity system. pybind11-based C++ ↔ Python bridging uses internal C++ object tracking via PythonBridge::getInstance().track<>() template.

## Monitoring & Observability

**Error Tracking:**
- None (errors bubble up as C++ exceptions or Python exceptions)

**Logs:**
- stdout/stderr via C++ std::cerr in error handlers (`src/main/cpp/riscv_extension.cc:34, 50, 66`)
- Optional commit logging at sim level (not RoCC-specific)

## CI/CD & Deployment

**Hosting:**
- GitHub (repo: github.com/liuyu81/pyspike)

**CI Pipeline:**
- cibuildwheel on GitHub Actions (inferred from pyproject.toml)
  - Builds wheels for cp38-cp312 on manylinux2014_x86_64
  - Before-all: yum install dtc
  - Test command: pytest -v -k 'not pyspike_cli' with PYTHONPATH=examples

## Environment Configuration

**Required env vars:**
- `RISCV` - Path to RISC-V toolchain (default: `/opt/riscv`)
  - Used during build: `setup.py:33, 53, 60`
  - Used during `_build_spike()`: subprocess configure with `--prefix=${dest_dir}`

**Secrets location:**
- No secrets. Environment configuration is public.

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

## RoCC-Specific Integration Points

### 1. Upstream Spike rocc_t Interface

**Integration:** pyspike wraps spike's rocc_t base class (defined in `vendor/spike/riscv/rocc.h`).

**Binding location:** `src/main/cpp/riscv_extension.h:56-72` defines `py_rocc_t` trampoline class:
```cpp
class py_rocc_t : public rocc_t, 
                  public pybind11::trampoline_self_life_support {
  // Overrides 4 virtual custom*() methods + name()
  virtual reg_t custom0/1/2/3(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2)
  virtual const char *name() const
};
```

**Virtual method trampolining:** Each custom* override in `riscv_extension.cc:90-108` uses `PYBIND11_OVERRIDE` macro to dispatch Python-defined rocc_t subclasses back to their Python implementations.

**Implementation:**
- `src/main/cpp/riscv_extension.cc:90-108` - py_rocc_t::custom0/1/2/3 method bodies using PYBIND11_OVERRIDE
- `src/main/cpp/py_module.cc:431-447` - pybind11 binding of rocc_t class, exposes custom0-3 and name()

### 2. Extension Registration & Factory

**Registration mechanism:** `py_register_extension()` in `src/main/cpp/riscv_extension.h:75` and `src/main/cpp/riscv_extension.cc:114-122`

**Flow:**
1. Python calls `riscv.extension.register_extension(name, py_class)` → pybind11 calls `py_register_extension()`
2. `py_register_extension()` wraps the py_class in a C++ lambda that calls `register_extension()` from upstream spike
3. When spike instantiates the extension via factory, the lambda returns a C++ pointer to the py_rocc_t/py_extension_t instance
4. PythonBridge::track<>() keeps the Python object alive (`src/main/cpp/riscv_extension.cc:118-120`)

**Binding:** `src/main/cpp/py_module.cc:450-451` - pybind11 def of register_extension

**Upstream call:** `src/main/cpp/riscv_extension.cc:115` calls upstream `register_extension(name.c_str(), ...)`

### 3. Processor Integration

**How rocc_t is attached to processor_t:**
- Via `processor_t::register_extension(extension_t *x)` in `src/main/cpp/py_module.cc:719-720`
- User code instantiates a rocc_t subclass, passes to proc.register_extension()
- Processor owns the extension and invokes it during instruction dispatch

**Binding:** `src/main/cpp/py_module.cc:719-720` - pybind11 binding of processor_t.register_extension

### 4. Python User Code ↔ C++ Binding

**Python user defines:**
- Subclass of `riscv.isa.ROCC` (defined in `src/main/python/riscv/isa.py:42-45`)
- Override `name`, `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()`, and optionally `custom0/1/2/3()`

**Example:** `tests/test_extension.py:54-58` shows MyDummyROCC inheriting from isa.ROCC

**Python isa.ROCC base:**
- `src/main/python/riscv/isa.py:42-45` defines ROCC(rocc_t, ISA) which:
  - Inherits from both rocc_t (C++ class via pybind11) and ISA (Python ABC)
  - ISA defines abstract name property and _name() method for C++ → Python dispatch
  - ROCC adds no additional functionality; exists to declare "this extension supports RoCC"

**Registration decorator:**
- `src/main/python/riscv/isa.py:48-70` defines @register(ext_name) decorator
- Wraps user class in a name-providing subclass and calls register_extension()

**Example usage:**
```python
@register("my_rocc")
class MyRoCC(isa.ROCC):
    def name(self): return "my_rocc"
    def custom0(self, proc, insn, xs1, xs2): ...
```

### 5. PythonBridge Object Lifetime Management

**Location:** `src/main/cpp/py_bridge.h:33-100`

**Purpose:** Keep Python objects alive while C++ code holds pointers to them.

**Template specialization:** `PythonBridge::track<T>()` allows C++ to register Python-created extension/rocc instances:
- `src/main/cpp/py_bridge.h:53-80` - Generic track<T>() template
- Static assert checks if T is rocc_t or extension_t (line 66-68)
- Stores py_obj.inc_ref() + mapping of C++ ptr → py::handle (line 76-78)

**Usage in RoCC context:**
- `src/main/cpp/riscv_extension.cc:118` - track<rocc_t*>(py_ext) keeps Python rocc_t alive
- `src/main/cpp/riscv_extension.cc:120` - track<extension_t*>(py_ext) for non-RoCC extensions
- `src/main/cpp/py_module.cc:105` - track<py_csr_t*>() for CSRs defined by RoCC

### 6. pybind11 Trampoline Pattern

**Base trampoline class:** `py_extension_t` and `py_rocc_t` in `src/main/cpp/riscv_extension.h:31-72`

**Why needed:** Allows Python to override virtual C++ methods. When spike C++ code calls extension_t::name(), it can reach a Python-implemented override.

**Key lines:**
- `src/main/cpp/riscv_extension.h:32` - py_extension_t inherits `pybind11::trampoline_self_life_support`
- `src/main/cpp/riscv_extension.h:56` - py_rocc_t inherits `pybind11::trampoline_self_life_support`
- `src/main/cpp/riscv_extension.cc:73` - `PYBIND11_OVERRIDE_PURE_NAME` for name() method
- `src/main/cpp/riscv_extension.cc:92-107` - `PYBIND11_OVERRIDE` for each custom* method

**trampoline_self_life_support detail:** Ensures the Python object backing the trampoline instance is never freed while C++ code still holds a pointer, and handles Python exception propagation.

### 7. Instruction Decoding & RoCC Instruction Constants

**RoCC opcode constants exposed to Python:**
- `src/main/cpp/py_module.cc:385-389` - Expose ROCC_OPCODE0/1/2/3 and ROCC_OPCODE_MASK
- `src/main/cpp/py_module.cc:391-410` - Define rocc_insn_t binding with fields: opcode, rd, xs2, xs1, xd, rs1, rs2, funct

**Binding location:** `mod_extension` submodule of _riscv module (line 383)

**Use case:** Python RoCC implementation can inspect rocc_insn_t fields to decode the instruction and dispatch to custom0-3.

---

*Integration audit: 2026-05-04*
