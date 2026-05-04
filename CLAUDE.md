<!-- GSD:project-start source:PROJECT.md -->
## Project

**pyspike + GTX NPU (Python RoCC Port)**

기존 C++ Spike RoCC 확장으로 구현된 GTX NPU functional model
(`~/NIGHTLY/gtx_spike/gtx/`)을 **pyspike의 `riscv.isa.ROCC` 서브클래스**로
**순수 Python(NumPy 백엔드)으로 재작성**하는 프로젝트. 결과물은 pyspike wheel
패키지에 동봉되어, 사용자가 `pip install spike` 후 한 줄로 GTX NPU 시뮬레이션을
띄우고 ISA/op를 Python에서 자유롭게 변형·검증할 수 있게 한다.

**Core Value:** **기존 NPU 펌웨어(.elf) 회귀 테스트가 pyspike+Python NPU에서도 그대로 통과하고
DDR 결과가 C++ libgtx_npu.so(SystemC HW sim과 ULP 내 일치 검증 완료된 golden)와
ULP 허용오차 내로 일치한다 — 이게 안 되면 다른 어떤 기능도 의미가 없다.**

### Constraints

- **Tech stack**: Python 3.8+ / NumPy(≥1.20 권장) / pyspike의 pybind11 트램폴린.
  C++ 추가 코드 금지(순수 Python 재작성이라는 사용자 결정) — 성능 핫스팟이 발견되면
  v2에서 cython/C 확장 검토
- **Compatibility**: `riscv.isa.ROCC` 가상 메서드 시그니처(`custom0/1/2/3(proc, insn,
  xs1, xs2) -> reg_t`)를 정확히 따라야 함. processor_t/rocc_insn_t는 pybind11
  바인딩 객체 그대로 사용
- **Performance**: NumPy 백엔드 가정. NEST(4)×SPU(16)×L1(384KB) 메모리 표현은
  ndarray로, FP16 연산은 `np.float16` 또는 FP32 누적 후 캐스트로. 회귀가 한
  세션 내(≤ 수십 분 수준) 끝나야 실용
- **Dependencies**: NumPy 외부 추가 런타임 의존성 신규 도입 금지(wheel 배포 단순성).
  검증 단계에서만 기존 C++ libgtx_npu.so 참조(개발 환경)
- **Bit-exact**: ULP 허용오차 내(`verify.py --fp16 --ulp 1 --atol 0.001` 수준)
  C++ 결과와 일치 필수. 회귀 1개라도 깨지면 출하 보류
- **Testing**: pytest 기반(이미 구축됨). 신규 op마다 verify_ref.py 대응 단위 테스트
  + 적어도 1개의 .elf 회귀 통과 묶음 추가
- **Platform**: Linux x86_64 / glibc 2.17+ (manylinux2014). cibuildwheel
  파이프라인을 깨지 않아야 함
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- **C++** (C++20 / C++2a) - RoCC trampoline and pybind11 binding layer; all py_* wrapper classes and glue code
- **Python** (3.8+) - User-facing extension API and RoCC subclass definitions
- **C** - Spike ISA simulator core (upstream, in `vendor/spike/`)
## Runtime
- Python 3.8, 3.9, 3.10, 3.11, 3.12 (tested via cibuildwheel on manylinux2014_x86_64)
- RISC-V toolchain at `/opt/riscv` (default, overridable via `RISCV` env var)
- pip / setuptools (Python side)
- Leverages pybind11 for C++ → Python bridging
- Lockfile: `pyproject.toml` and `setup.py`
## Frameworks
- **pybind11** [>3] - Binds C++ extension_t/rocc_t classes to Python; provides trampoline_self_life_support for virtual method overriding
- **Spike RISC-V ISA Simulator** (upstream submodule at `vendor/spike/`) - Provides rocc_t base class and ISA infrastructure
- **setuptools** [>=75] - Build system
- **setuptools_scm** [>=9] - Version management (git-based)
- **pybind11.setup_helpers** - CMake-free pybind11 extension building
- **pytest** - Test framework (referenced in `pyproject.toml`)
- **pytest-cov** - Coverage collection
## Key Dependencies
- **libriscv.so** - Spike's RISC-V simulator library; linked at build time via `-lriscv` flag in `setup.py:55`
- **libdisasm.a** - Spike disassembler (static, bundled)
- **libfesvr.a** - Front-end server (static, bundled)
- **pybind11** [>3] - Mandatory for binding extension_t and rocc_t classes; supports trampoline_self_life_support for virtual method dispatch
- **auditwheel** - Binary wheel auditing (manylinux compliance)
- **patchelf** - ELF manipulation for bundled libraries
- **dtc** (device-tree-compiler) - Required before-all in cibuildwheel
## Configuration
- `RISCV` env var sets toolchain prefix (default: `/opt/riscv`)
- `PYSPIKE_LIBS` - Name of env var for loading spike libraries dynamically (defined in `src/main/python/riscv/__init__.py:30`)
- `PYSPIKE_EXTS` - Name of env var for extension library paths (defined in `src/main/python/riscv/__init__.py:32`)
- C++ compilation with `-std=c++2a` (C++20) - `setup.py:45`
- pybind11 detailed error messages enabled via `PYBIND11_DETAILED_ERROR_MESSAGES=1` macro - `setup.py:49`
- Runtime library paths: `-Wl,-rpath,$ORIGIN/data/lib` and `-Wl,-rpath,{RISCV}/lib` - `setup.py:52-53`
## Platform Requirements
- GCC/Clang with C++20 support
- dtc (device-tree-compiler)
- RISC-V cross-toolchain at `$RISCV` (typically from github.com/riscv-collab/riscv-gnu-toolchain)
- Linux x86_64 (manylinux2014 baseline)
- Python 3.8+ shared library
- glibc 2.17+
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## RoCC Extension Pattern Overview
- **Python side**: Base classes (`rocc_t`, `ISA`, `ROCC`) with required method overrides
- **C++ side**: Pybind11 trampolines (`py_rocc_t`, `py_extension_t`) that mediate Python → C++ → Python calls via `PYBIND11_OVERRIDE*` macros
## Naming Patterns
- Extension modules follow lowercase with underscores: `examples/xhuimt/__init__.py`, `examples/xthead/__init__.py`
- Test files: `tests/test_extension.py` for extension test suite
- Custom implementations: descriptive names like `MyLRSC`, `MyDummyROCC` for test classes
- Base classes for user extension: `class MyCustomROCC(isa.ROCC)` (inherits from `isa.ROCC`)
- Base classes for ISA extension: `class MyISA(isa.ISA)` (inherits from `isa.ISA`)
- Test dummy class pattern: `class MyDummyROCC(isa.ROCC)` with minimal implementation
- Name property implementation: `@property def name(self) -> str: return "extension_name"` (required override)
- Extension registration key matches class name: `@isa.register("my_extension_name")`
- Key convention: lowercase, underscore-separated (e.g., `"huimt"`, `"my_cflush"`, `"my_dummy_rocc"`)
- Pattern: `py_rocc_t` (inherits from `rocc_t` and `pybind11::trampoline_self_life_support`)
- Pattern: `py_extension_t` (inherits from `extension_t` and `pybind11::trampoline_self_life_support`)
- RoCC instruction handlers: `custom0`, `custom1`, `custom2`, `custom3` (required overrides)
- Extension API: `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()`
- pybind11 macro naming: `PYBIND11_OVERRIDE_PURE_NAME` for pure virtual, `PYBIND11_OVERRIDE` for overridable
- Instruction field extraction: `proc`, `insn`, `xs1`, `xs2` (from RoCC ISA specification)
- rocc_insn_t properties: `opcode`, `rd`, `xs2`, `xs1`, `xd`, `rs1`, `rs2`, `funct` (read-only)
- Processor references: `proc: processor_t` (type-hinted as pointer in C++, reference in Python)
- Python return types: `-> str`, `-> List[insn_desc_t]`, `-> List[disasm_insn_t]`, `-> List[csr_t]`
- Python function args: explicit type hints required (`proc: processor_t`)
- C++ trampoline args: exact spike types (`processor_t *proc`, `rocc_insn_t insn`, `reg_t xs1`, `reg_t xs2`)
## Python API Conventions
### Base Class Inheritance
- `custom0(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom1(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom2(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
- `custom3(self, proc: processor_t, insn: rocc_insn_t, xs1: int, xs2: int) -> int`
### Registration Decorator
- Creates a wrapped class with the same `__name__` and `__doc__` as original
- Sets the `name` property to return the decorator argument (the `ext_name`)
- Calls `register_extension(ext_name, WrappedClass)` automatically
- Returns the wrapped class for use in module
### Custom Instruction Definition Pattern
### Custom Instruction Formatting
## C++ pybind11 Conventions
### Trampoline Class Structure
### Override Macro Usage
- `"_name"` is the Python method name (underscore prefix indicates C++ implementation detail, user calls `@property name`)
### Pybind11 Binding Declaration
- Template args: `py::class_<BaseType, TrampolineType, ParentTypes..., py::smart_holder>(module, "PythonName")`
- `py_rocc_t` as trampoline type enables Python overrides of rocc_t methods
- `py::smart_holder` manages lifetime across language boundaries
- `.def(py::init())` allows Python `rocc_t()` instantiation (required)
- All public virtual methods exposed with argument names: `py::arg("name")`
### Python Bridge and Lifetime
- Converts pybind11 Python handle to C++ pointer
- Stores reference to prevent garbage collection while C++ holds the object
- Supported types (from validation check): `rocc_t`, `extension_t`, `processor_t`, `csr_t`, and others
## Error Handling
- Python exceptions are caught at the C++ trampoline boundary
- Error message printed to stderr (not re-thrown)
- Method returns empty/default value (e.g., empty vector, 0 for custom0-3)
- Execution continues in the simulator
## Style & Formatting
- Line length: max 120 characters (pylint setting)
- Type hints: explicit, required for method signatures
- Docstrings: optional (disabled in pylint config), but comments on complex logic recommended
- Consistent with spike upstream conventions
- pybind11 modules use lowercase with underscores: `mod_extension`, `mod_decode`, etc.
- Class member access: public/protected/private separation
- `pytest --pylint`: static code analysis with disabled messages for abstract methods
- `pytest --mypy`: type checking with `check_untyped_defs = 1`
- Coverage: lcov for C++ code (see `conftest.py:77-107`)
## Import Organization (Python)
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Python user code subclasses `riscv.isa.ROCC` or `riscv.isa.ISA` and uses `@riscv.isa.register()` decorator
- Decorator wraps the class and calls C++ `py_register_extension()` to register a factory
- Factory is invoked by spike's simulator when instantiating extensions per-hart
- Custom instruction dispatch flows: Python opcode handler → `custom0/1/2/3` virtual override → return to spike
- Lifecycle managed by `PythonBridge` singleton: bootstraps Python environment, loads extension modules via `PYSPIKE_LIBS` env var, keeps Python objects alive across C++/Python boundary
## Layers
- Purpose: User-facing extension API with decorators, base classes, and registration
- Location: `src/main/python/riscv/isa.py`, `src/main/python/riscv/__init__.py`
- Contains: `ISA`, `ROCC` abstract base classes; `register()` decorator; `arg()` decorator for disasm operands
- Depends on: pybind11-wrapped `_riscv` module (C++ extension types `extension_t`, `rocc_t`)
- Used by: User Python extension packages (e.g., `examples/xhuimt/`, `examples/xthead/`)
- Purpose: Expose C++ RISC-V types to Python; register module with pybind11
- Location: `src/main/cpp/py_module.cc`
- Contains: `PYBIND11_MODULE(_riscv)` definition (lines 48+); submodules for `cfg`, `csrs`, `decode`, `disasm`, `extension`, `isa_parser`, etc.
- Key bindings: `rocc_t` (lines 431–447), `extension_t` (lines 411–429), `rocc_insn_t` (lines 391–409)
- Depends on: `py_rocc_t`, `py_extension_t` trampolines; `py_register_extension()` function
- Used by: Python layer; loaded as `_riscv` module during `riscv` import
- Purpose: Virtual dispatch from C++ to Python method overrides
- Location: `src/main/cpp/riscv_extension.h` (class definitions), `src/main/cpp/riscv_extension.cc` (implementations)
- Contains: 
- Depends on: `PythonBridge` singleton for object tracking and Python calls; pybind11 for `PYBIND11_OVERRIDE` macros
- Used by: C++ extension framework when methods are overridden in Python
- Purpose: Initialize Python interpreter; load and import extension modules from `PYSPIKE_LIBS` env var
- Location: `src/main/cpp/py_bridge.cc` (lines 23–53)
- Contains: `PythonBridge::bootstrap()` (lines 35–53); imports modules listed in `PYSPIKE_LIBS` using `importlib.import_module()`
- Mechanism: `py_bridge.cc` line 42 reads env var, splits on `os.pathsep`, imports each module by name
- Used by: Spike simulator initialization; called once during pybind11 module load
- Purpose: Parse `--extlib` arguments; load Python/C++ libraries; invoke spike executable with patched args
- Location: `scripts/pyspike` (wrapper), `src/main/python/riscv/__main__.py` (main entry)
- Entry: User runs `pyspike --extlib=foo.py --extlib=bar.so ...`
- Logic (lines 30–58 of `__main__.py`):
- Used by: End users; integrates Python extension loading with spike CLI
- Spike defines base `rocc_t` with virtual `custom0/1/2/3` methods
- pyspike wraps it with `py_rocc_t` trampoline class
## Data Flow
### Extension Registration
```
```
### Extension Instantiation
```
```
### Custom Instruction Dispatch
```
```
## State Management
- Created once per hart during `processor_t::reset()`
- Stored in `processor_t::extensions` map (keyed by name)
- Python objects kept alive by `PythonBridge::references` map (line 78 of py_bridge.h)
- Shared state (CSRs, custom state) managed by user Python code (e.g., reserved addresses in `MyLRSC`, line 32 of `examples/xhuimt/mylrsc.py`)
- `PythonBridge` singleton (line 46 of py_bridge.h) initialized once
- Python interpreter bootstrapped if not already running (line 25 of py_bridge.cc)
- Imported extension modules persist for lifetime of spike process
## Key Abstractions
- Purpose: Abstract base for all extension types
- Defines: `name` property (abstract); `_name()` method for C++ callback
- Examples: Subclass for custom extensions
- Purpose: Abstract base for RoCC (Rocket Custom Coprocessor) extensions
- Inherits from both `rocc_t` (C++ pybind11 binding) and `ISA` (Python abstraction)
- Pattern: User creates subclass, implements `custom0/1/2/3` or leaves default (no-op)
- Purpose: Decorator for registering extension by name
- Creates synthetic subclass with hardcoded `name` property (lines 58–65)
- Calls `register_extension()` to bind name → factory (line 67)
- Returns the synthetic class for user
- Purpose: pybind11 trampoline for `rocc_t` virtual methods
- Inherits from both `rocc_t` (upstream) and `pybind11::trampoline_self_life_support`
- Implements: `custom0/1/2/3()` using `PYBIND11_OVERRIDE` to invoke Python overrides
- Also implements: `name()` pure virtual method
- Purpose: pybind11 trampoline for `extension_t` (general extensions)
- Similar to `py_rocc_t` but for non-RoCC extensions
- Trampolines: `get_instructions()`, `get_disasms()`, `get_csrs()`, `name()`, `reset()`, `set_debug()`
- Purpose: Singleton managing Python interpreter state and object lifetimes
- Key methods:
## Entry Points
- Location: `examples/xhuimt/__init__.py`
- Triggers: When `PYSPIKE_LIBS` includes this module path
- Responsibilities:
- Location: `scripts/pyspike` (shell wrapper)
- Triggers: `pyspike --extlib=module.py --extlib=lib.so`
- Flow: Parses args → calls `riscv.__main__.main()` → sets `PYSPIKE_LIBS` → `execve()` spike
- Location: `src/main/cpp/riscv_extension.cc`, line 114–122
- Triggers: When Python code calls `riscv.extension.register_extension(name, py_class)`
- Mechanism: Lambda closure captures `py_ctor`, called by spike's extension instantiation
- Location: `src/main/cpp/py_bridge.cc`, lines 35–53
- Triggers: First call to `PythonBridge::getInstance()` during spike initialization
- Mechanism: Reads `PYSPIKE_LIBS` env var, imports each module
## Error Handling
- If Python method raises exception, pybind11 converts to C++ exception
- Not explicitly caught (relies on pybind11 error propagation)
- Simulator may crash or exhibit undefined behavior
- Wrapped in `try { } catch (py::error_already_set &e)` (lines 33–35, 50–52, 66–68)
- Error printed to stderr (line 34, 51, 67)
- Empty vector returned to simulator (graceful degradation)
- `importlib.import_module()` wrapped implicitly (Python exception caught internally)
- Warning issued to stderr if import fails
- Simulator continues without that extension
## Cross-Cutting Concerns
- C++ errors via `std::cerr` (riscv_extension.cc lines 34, 51, 67)
- Python errors via `warnings.warn()` (py_bridge.cc line 52)
- No centralized logging framework
- Python type hints used (riscv/isa.py, examples/) but not enforced at runtime
- pybind11 handles type casting; mismatch raises Python `TypeError`
- None; all extensions trusted if loaded via `--extlib`
- Virtual function calls add minimal overhead (pybind11 trampoline cost ~1–2 μs per call)
- Python GIL held during custom instruction execution (single-threaded spike context)
- No lazy loading; extensions instantiated for each hart on first use
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
