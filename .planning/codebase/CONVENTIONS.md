# Coding Conventions

**Analysis Date:** 2025-05-04

## RoCC Extension Pattern Overview

RoCC (Rocket Custom Coprocessor) extensions in pyspike follow a two-layer pattern:
- **Python side**: Base classes (`rocc_t`, `ISA`, `ROCC`) with required method overrides
- **C++ side**: Pybind11 trampolines (`py_rocc_t`, `py_extension_t`) that mediate Python → C++ → Python calls via `PYBIND11_OVERRIDE*` macros

## Naming Patterns

**Files:**
- Extension modules follow lowercase with underscores: `examples/xhuimt/__init__.py`, `examples/xthead/__init__.py`
- Test files: `tests/test_extension.py` for extension test suite
- Custom implementations: descriptive names like `MyLRSC`, `MyDummyROCC` for test classes

**Classes:**
- Base classes for user extension: `class MyCustomROCC(isa.ROCC)` (inherits from `isa.ROCC`)
- Base classes for ISA extension: `class MyISA(isa.ISA)` (inherits from `isa.ISA`)
- Test dummy class pattern: `class MyDummyROCC(isa.ROCC)` with minimal implementation
- Name property implementation: `@property def name(self) -> str: return "extension_name"` (required override)

**Factory/Registration:**
- Extension registration key matches class name: `@isa.register("my_extension_name")`
- Key convention: lowercase, underscore-separated (e.g., `"huimt"`, `"my_cflush"`, `"my_dummy_rocc"`)

**C++ Trampoline Class Naming:**
- Pattern: `py_rocc_t` (inherits from `rocc_t` and `pybind11::trampoline_self_life_support`)
- Pattern: `py_extension_t` (inherits from `extension_t` and `pybind11::trampoline_self_life_support`)

**Functions:**
- RoCC instruction handlers: `custom0`, `custom1`, `custom2`, `custom3` (required overrides)
- Extension API: `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()`
- pybind11 macro naming: `PYBIND11_OVERRIDE_PURE_NAME` for pure virtual, `PYBIND11_OVERRIDE` for overridable

**Variables and Parameters:**
- Instruction field extraction: `proc`, `insn`, `xs1`, `xs2` (from RoCC ISA specification)
- rocc_insn_t properties: `opcode`, `rd`, `xs2`, `xs1`, `xd`, `rs1`, `rs2`, `funct` (read-only)
- Processor references: `proc: processor_t` (type-hinted as pointer in C++, reference in Python)

**Type Hints:**
- Python return types: `-> str`, `-> List[insn_desc_t]`, `-> List[disasm_insn_t]`, `-> List[csr_t]`
- Python function args: explicit type hints required (`proc: processor_t`)
- C++ trampoline args: exact spike types (`processor_t *proc`, `rocc_insn_t insn`, `reg_t xs1`, `reg_t xs2`)

## Python API Conventions

### Base Class Inheritance

**For RoCC extensions**, always inherit from `isa.ROCC`:

```python
from riscv import isa

class MyROCC(isa.ROCC):
    @property
    def name(self) -> str:
        return "my_rocc_name"
```

The `isa.ROCC` base class is defined at `src/main/python/riscv/isa.py:42-45` and includes four required handler methods (inherited from `rocc_t`):
- `custom0(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom1(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom2(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom3(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`

Default implementations (from `rocc_t` base in spike) return 0.

**For non-RoCC extensions**, inherit from `isa.ISA`:

```python
from riscv import isa

class MyCFlush(isa.ISA):
    @property
    def name(self) -> str:
        return "my_cflush"
    
    def get_instructions(self, proc: processor_t) -> List[insn_desc_t]:
        return []
    
    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]:
        return []
    
    def get_csrs(self, proc: processor_t) -> List[csr_t]:
        return []
    
    def reset(self, proc: processor_t):
        super().reset(proc)
```

### Registration Decorator

Use `@isa.register(extension_name)` decorator on class definition:

```python
@isa.register("my_rocc")
class MyROCC(isa.ROCC):
    @property
    def name(self) -> str:
        return "my_rocc"  # matches decorator argument
```

**Decorator behavior** (from `src/main/python/riscv/isa.py:48-70`):
- Creates a wrapped class with the same `__name__` and `__doc__` as original
- Sets the `name` property to return the decorator argument (the `ext_name`)
- Calls `register_extension(ext_name, WrappedClass)` automatically
- Returns the wrapped class for use in module

### Custom Instruction Definition Pattern

Extensions export instruction definitions via `get_instructions()`. Example from `examples/xhuimt/__init__.py:37-40`:

```python
def get_instructions(self, proc: processor_t) -> List[insn_desc_t]:
    return [
        *self.lrsc.get_instructions(proc),  # delegate to sub-extension
    ]
```

Return type is always `List[insn_desc_t]`.

### Custom Instruction Formatting

For custom disassembly output, use the `@isa.arg` decorator at `src/main/python/riscv/isa.py:73-86`:

```python
from riscv import isa
from riscv.decode import insn_t

@isa.arg
def my_operand(insn: insn_t) -> str:
    # inspect insn fields and format output
    return f"x{insn.rd}"
```

Returns an `arg_t` object usable in `get_disasms()`.

## C++ pybind11 Conventions

### Trampoline Class Structure

**py_rocc_t** at `src/main/cpp/riscv_extension.h:55-72`:

```cpp
class py_rocc_t : public rocc_t, public pybind11::trampoline_self_life_support {
public:
  using rocc_t::rocc_t;  // inherit constructors

public:
  // Override the four custom handlers
  virtual reg_t custom0(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  virtual reg_t custom1(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  virtual reg_t custom2(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  virtual reg_t custom3(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) override;

public:
  virtual const char *name() const override;
};
```

**py_extension_t** at `src/main/cpp/riscv_extension.h:31-53`:

```cpp
class py_extension_t : public extension_t,
                       public pybind11::trampoline_self_life_support {
public:
  using extension_t::extension_t;

public:
  virtual std::vector<insn_desc_t> get_instructions(const processor_t &proc) override;
  virtual std::vector<disasm_insn_t *> get_disasms(const processor_t *proc = nullptr) override;
  virtual std::vector<csr_t_p> get_csrs(processor_t &proc) const override;

public:
  virtual const char *name() const override;
  virtual void reset(processor_t &proc) override;
  virtual void set_debug(bool value, const processor_t &proc) override;

public:
  // expose protected base members for Python access
  using extension_t::clear_interrupt;
  using extension_t::illegal_instruction;
  using extension_t::raise_interrupt;
};
```

Both inherit from `pybind11::trampoline_self_life_support` to maintain Python object lifetime across C++→Python calls.

### Override Macro Usage

In implementation file (`src/main/cpp/riscv_extension.cc`), use `PYBIND11_OVERRIDE*` macros:

**For pure virtual methods** (those with no default implementation in C++ base):

```cpp
const char *py_rocc_t::name() const {
  PYBIND11_OVERRIDE_PURE_NAME(const char *, rocc_t, "_name", name);
}
```

Macro signature: `PYBIND11_OVERRIDE_PURE_NAME(return_type, base_class, python_method_name, cpp_method_name)`
- `"_name"` is the Python method name (underscore prefix indicates C++ implementation detail, user calls `@property name`)

**For overridable methods** (those with default implementation):

```cpp
reg_t py_rocc_t::custom0(processor_t *proc, rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  PYBIND11_OVERRIDE(reg_t, rocc_t, custom0, proc, insn, xs1, xs2);
}
```

Macro signature: `PYBIND11_OVERRIDE(return_type, base_class, method_name, args...)`

**For non-override virtual methods** (methods added purely for Python):

```cpp
std::vector<insn_desc_t> py_extension_t::get_instructions(const processor_t &proc) {
  std::vector<insn_desc_t> instructions;
  auto &bridge = PythonBridge::getInstance();
  try {
    py::function py_method = py::get_override(this, "get_instructions");
    py::object py_proc = py::cast(&proc);
    py::sequence py_seq = py_method(*bridge.track<processor_t *>(py_proc));
    for (const auto &py_obj : py_seq) {
      instructions.push_back(*bridge.track<insn_desc_t *>(py_obj));
    }
  } catch (py::error_already_set &e) {
    std::cerr << e.what() << std::endl;
  }
  return instructions;
}
```

Pattern:
1. Get the override from the Python object: `py::get_override(this, "method_name")`
2. Use `PythonBridge` singleton to track lifetime and convert pointer types
3. Wrap all in try-catch for `py::error_already_set`
4. Print exception to stderr if caught (do not re-throw by default)

### Pybind11 Binding Declaration

In `src/main/cpp/py_module.cc:431-447`:

```cpp
py::class_<rocc_t, py_rocc_t, extension_t, py::smart_holder>(mod_extension,
                                                             "rocc_t")
    .def(py::init())
    // rocc_t members
    .def("custom0", &rocc_t::custom0, py::arg("proc"), py::arg("insn"),
         py::arg("xs1"), py::arg("xs2"))
    .def("custom1", &rocc_t::custom1, py::arg("proc"), py::arg("insn"),
         py::arg("xs1"), py::arg("xs2"))
    .def("custom2", &rocc_t::custom2, py::arg("proc"), py::arg("insn"),
         py::arg("xs1"), py::arg("xs2"))
    .def("custom3", &rocc_t::custom3, py::arg("proc"), py::arg("insn"),
         py::arg("xs1"), py::arg("xs2"))
    // extension_t members
    .def("get_instructions", &rocc_t::get_instructions, py::arg("proc"))
    .def("get_disasms", &rocc_t::get_disasms, py::arg("proc"))
    .def("get_csrs", &rocc_t::get_csrs, py::arg("proc"))
    .def_property_readonly("name", &rocc_t::name);
```

**Declaration pattern**:
- Template args: `py::class_<BaseType, TrampolineType, ParentTypes..., py::smart_holder>(module, "PythonName")`
- `py_rocc_t` as trampoline type enables Python overrides of rocc_t methods
- `py::smart_holder` manages lifetime across language boundaries
- `.def(py::init())` allows Python `rocc_t()` instantiation (required)
- All public virtual methods exposed with argument names: `py::arg("name")`

### Python Bridge and Lifetime

The `PythonBridge` singleton (header: `src/main/cpp/py_bridge.h:33-91`) manages Python object lifetime:

```cpp
template <typename T> T track(pybind11::handle py_obj) {
  py_obj.inc_ref();
  T obj = pybind11::cast<T>(py_obj);
  references.emplace(reinterpret_cast<uint64_t>(obj), py_obj);
  return obj;
}
```

**Usage**:
- Converts pybind11 Python handle to C++ pointer
- Stores reference to prevent garbage collection while C++ holds the object
- Supported types (from validation check): `rocc_t`, `extension_t`, `processor_t`, `csr_t`, and others

**Always use in custom0/1/2/3 implementations when converting processor_t**:

```cpp
auto py_proc = py::cast(&proc);
PYBIND11_OVERRIDE(reg_t, rocc_t, custom0, 
                  *bridge.track<processor_t *>(py_proc), insn, xs1, xs2);
```

This ensures the processor object remains alive during the Python call.

## Error Handling

**Python → C++ exceptions propagate** via try-catch in `py_extension_t` methods:

```cpp
try {
    py::function py_method = py::get_override(this, "get_instructions");
    // ... call python method ...
} catch (py::error_already_set &e) {
    std::cerr << e.what() << std::endl;  // print to stderr
}
return instructions;  // return empty/default if Python threw
```

**Exception behavior**:
- Python exceptions are caught at the C++ trampoline boundary
- Error message printed to stderr (not re-thrown)
- Method returns empty/default value (e.g., empty vector, 0 for custom0-3)
- Execution continues in the simulator

This is intentional: RoCC instruction handlers should not crash the simulator if Python raises an exception; the exception is logged and execution continues.

## Style & Formatting

**Python code** (from `pyproject.toml:149`):
- Line length: max 120 characters (pylint setting)
- Type hints: explicit, required for method signatures
- Docstrings: optional (disabled in pylint config), but comments on complex logic recommended

**C++ code**:
- Consistent with spike upstream conventions
- pybind11 modules use lowercase with underscores: `mod_extension`, `mod_decode`, etc.
- Class member access: public/protected/private separation

**Linting & Type Checking** (from `pyproject.toml:149-191`):
- `pytest --pylint`: static code analysis with disabled messages for abstract methods
- `pytest --mypy`: type checking with `check_untyped_defs = 1`
- Coverage: lcov for C++ code (see `conftest.py:77-107`)

## Import Organization (Python)

**Standard order**:
1. `import abc` and other standard library
2. `from typing import ...`
3. `from riscv import isa` and other pyspike imports
4. Project-specific imports (local example modules)

Example from `src/main/python/riscv/isa.py:16-21`:

```python
import abc
from typing import Callable, Type

from riscv.decode import insn_t
from riscv.disasm import arg_t
from riscv.extension import extension_t, rocc_t, register_extension
```

---

*Convention analysis: 2025-05-04*
