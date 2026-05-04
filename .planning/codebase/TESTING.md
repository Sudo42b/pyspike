# Testing Patterns

**Analysis Date:** 2025-05-04

## Test Framework Setup

**Test Runner:**
- Framework: `pytest` (from `pyproject.toml:87`)
- Config file: `pyproject.toml:148-162`
- Run command: `pytest -v` (or `pytest -v -k 'not pyspike_cli'` for CI)

**Assertion Library:**
- Standard: Python's `assert` statement (no pytest-assert-rewrite modifications)
- Equality checks: `assert obj == expected_value`
- Type checks: `assert isinstance(obj, ClassName)`

**Test Commands:**
```bash
pytest -v                          # Run all tests
pytest -v tests/test_extension.py  # Run extension tests only
pytest --cov                       # With code coverage (pytest-cov)
```

**Plugins enabled** (from `pyproject.toml:79-98`):
- `pytest-pylint`: static analysis on test files
- `pytest-mypy`: type checking on test files
- `pytest-cov`: coverage reporting with lcov output
- `pytest-asyncio`: async test support
- `pytest-timeout`: timeout protection
- `pytest-repeat`: test repetition for flakiness detection

## Test File Organization

**Location and Naming:**
- Test directory: `tests/`
- Test files: `tests/test_*.py` (e.g., `tests/test_extension.py`)
- Fixture file: `tests/conftest.py` (session-scoped setup)

**RoCC-related tests** are located in:
- `tests/test_extension.py`: Tests for `extension_t` and `rocc_t` base classes and registration mechanism

## Test Structure Patterns

### Fixture: Session-Scoped Simulator

From `tests/conftest.py:40-53`:

```python
@pytest.fixture(scope="session")
def mock_sim():
    yield sim_t(
        cfg=cfg_t(
            isa="rv32gc",
            priv="m",
            mem_layout=[
                mem_cfg_t(0x9000_0000, 0x4_0000)
            ],
            start_pc=0x9000_0000
        ),
        halted=True,
        plugin_device_factories=[],
        args=["pk"],
        dm_config=debug_module_config_t())
```

**Purpose**:
- Provides a single `sim_t` instance for all extension tests
- Configured with 32-bit RISC-V (rv32gc) ISA, machine privilege, 256 KB memory starting at 0x9000_0000
- Created once per test session (not per test function) for performance
- Passed to test functions as `mock_sim` parameter

**Usage in tests**:

```python
def test_find_extension(mock_sim, ...):
    p: processor_t = mock_sim.get_core(0)  # Get first core
    p.reset()  # Reset processor state
    # ... test extension ...
```

### Test Pattern: Extension Registration and Instantiation

From `tests/test_extension.py:61-88` (test_find_extension):

```python
@pytest.mark.parametrize("name,cls,n_insn,n_disasm", [
    pytest.param("cflush", extension_t, 3, 3, id="cflush"),
    pytest.param("dummy_rocc", extension_t, 4, 0, id="dummy_rocc"),
])
def test_find_extension(mock_sim, name, cls, n_insn, n_disasm):
    if not find_library("customext"):
        pytest.skip("libcustomext.so not found in this build")
    p: processor_t = mock_sim.get_core(0)
    p.reset()
    # lookup
    ext_ctor = find_extension(name)
    assert ext_ctor is not None
    # instantiate
    ext = ext_ctor()
    assert isinstance(ext, cls)
    # instructions
    all_insn = ext.get_instructions(p)
    assert len(all_insn) == n_insn
    for this_insn in all_insn:
        assert isinstance(this_insn, insn_desc_t)
    # disasms
    all_disasm = ext.get_disasms(p)
    assert len(all_disasm) == n_disasm
    for disasm in all_disasm:
        assert isinstance(disasm, disasm_insn_t)
    # reset
    ext.reset(p)
```

**Test structure**:
1. **Parameterization**: Multiple test cases via `@pytest.mark.parametrize` with `pytest.param(..., id="..."`
2. **Skip condition**: Skip if external extension library not compiled (`find_library("customext")`)
3. **Setup**: Get processor from mock_sim, reset state
4. **Test steps**:
   - `find_extension(name)`: Lookup extension by name → returns callable/class
   - Instantiate extension
   - Type assertion on result
   - Call `get_instructions()` → verify count and types
   - Call `get_disasms()` → verify count and types
   - Call `reset()`

**RoCC specific**: `dummy_rocc` test case expects 4 instructions (inherited default from rocc_t).

### Test Pattern: Custom Extension Registration

From `tests/test_extension.py:54-59` and `90-117`:

```python
class MyDummyROCC(isa.ROCC):
    @property
    def name(self) -> str:
        return "my_dummy_rocc"

@pytest.mark.parametrize("name,cls,n_insn,n_disasm", [
    pytest.param("my_dummy_rocc", MyDummyROCC, 4, 0, id="my_dummy_rocc"),
])
def test_register_extension(mock_sim, name, cls, n_insn, n_disasm):
    p: processor_t = mock_sim.get_core(0)
    p.reset()
    # register
    register_extension(name, cls)
    # lookup
    ext_ctor = find_extension(name)
    assert ext_ctor is not None
    # instantiate
    ext = ext_ctor()
    assert isinstance(ext, cls)
    assert ext.name == name
    # instructions
    all_insn = ext.get_instructions(p)
    assert len(all_insn) == n_insn
    for this_insn in all_insn:
        assert isinstance(this_insn, insn_desc_t)
    # disasms
    all_disasm = ext.get_disasms(p)
    assert len(all_disasm) == n_disasm
    for disasm in all_disasm:
        assert isinstance(disasm, disasm_insn_t)
    # reset
    ext.reset(p)
```

**Difference from find_extension test**:
1. **Define test class** (not loaded from external library)
2. **Call `register_extension(name, cls)`** before lookup (C++ registers constructor function)
3. **Verify name property** after instantiation: `assert ext.name == name`
4. RoCC dummy implementation: inherits custom0-3, returns 0 by default (not called in test)

**MyDummyROCC behavior**:
- Inherits from `isa.ROCC` which inherits from `rocc_t`
- Only overrides `name` property
- `get_instructions()` returns inherited default (4 instructions for RoCC, from rocc_t base)
- `get_disasms()` returns empty list (0 disassemblies)
- Handlers (custom0-3) use inherited defaults (return 0)

### Test Pattern: Python ISA Extension (Non-RoCC)

From `tests/test_extension.py:30-51`:

```python
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

    def set_debug(self, value: bool, proc: processor_t):
        super().set_debug(value, proc)
```

**Requirements for `isa.ISA` subclasses**:
- Implement `name` property (required, no default)
- Implement or inherit `get_instructions()`, `get_disasms()`, `get_csrs()`, `reset()`, `set_debug()`
- Can delegate to parent: `super().reset(proc)` (does nothing in base but good practice)
- Return types: `List[insn_desc_t]`, `List[disasm_insn_t]`, `List[csr_t]`, `None`

## Mocking Patterns

**Current practice**: No mock objects used. Tests use:
- Real `sim_t` instance (albeit halted)
- Real `processor_t` from `sim.get_core(0)`
- Real extension classes (both built-in and test-defined)

**Why not mocked**:
- Testing the real pybind11 trampoline behavior requires actual C++ ↔ Python calls
- Mocking would defeat the purpose (validating language boundary crossing)

## Fixtures and Test Data

**Session-level fixture**: `mock_sim` (described above)

**Per-module scope fixture**: `import_from_data_dir` (unused in extension tests)

```python
@pytest.fixture(scope="session")
def import_from_data_dir():
    path = pathlib.Path(__file__).parent / "data"
    sys.path.insert(0, path.as_posix())
    return path
```

**Test data files** in `tests/data/`:
- `huimt_lr_sc.elf`: ELF binary for HuiMt extension with LRSC instructions
- `huimt_msctlr.elf`: ELF binary for HuiMt with CSR control register
- `libc-printf_hello.elf`: General test binary
- `plic-uart_echo.elf`: Peripheral + UART test

These ELF files are used by other test modules (e.g., `test_sim.py`) but not directly by extension tests.

## Test Execution and Coverage

**From pyproject.toml:148-162**:

```toml
[tool.pytest.ini_options]
addopts = "--pylint --mypy --cov-report=lcov"
filterwarnings = [
  "ignore::UserWarning"
]
pythonpath = [
  "src/main/python",
  "examples"
]
testpaths = [
  "tests"
]
asyncio_default_fixture_loop_scope = "session"
```

**Execution**:
- Default addopts: runs pylint, mypy, coverage simultaneously
- Coverage output: `riscv.lcov` (Python) and `_riscv.lcov` (C++, if GCOV=1)
- Pythonpath includes: `src/main/python/` (for `from riscv import ...`) and `examples/` (for example extensions)

**Coverage report generation** (from `tests/conftest.py:77-107`):

```python
def _lcov_report(terminalreporter, verbosity: int):
    # ... generate C++ trace file with lcov ...
    subprocess.run([
        "lcov", "--capture", "--test-name", project_dir.name, "--no-external",
        "--directory", project_dir.joinpath("build").as_posix(), 
        "--base-directory", project_dir.as_posix(),
        "--demangle-cpp", "-o", lcov_cpp.as_posix()
    ], ...)
    # ... generate report with custom lcov-report script ...
```

**Coverage requirements** (from `pyproject.toml:212-222`):
- Source files: `examples/`, `src/main/python/`, `tests/`
- Excluded lines: namespace package init, `if __name__ == "__main__"`, abstract methods, pass statements
- No line coverage threshold enforced (minimum coverage not specified)

## Test Types

**Unit Tests:**
- Scope: Individual extension class registration and method call validation
- Approach: Direct instantiation → call methods → assert results
- Example: `test_find_extension` validates that find_extension returns correct class

**Integration Tests:**
- Scope: Extension registered in simulator → get_instructions() → called with real processor_t
- Approach: Use mock_sim, test extension lifecycle with actual processor state
- Example: Calling `ext.reset(p)` after instantiation (processor state management)

**No E2E Tests:**
- E2E testing (RoCC instruction execution in assembled binaries) not in test_extension.py
- Could be added to validate custom0-3 handlers execute correctly with real instruction binaries

## Common Test Assertions

**Type validation**:
```python
assert isinstance(ext, MyDummyROCC)
assert isinstance(this_insn, insn_desc_t)
```

**Collection validation**:
```python
all_insn = ext.get_instructions(p)
assert len(all_insn) == 4  # RoCC default
for insn in all_insn:
    assert isinstance(insn, insn_desc_t)
```

**Lookup validation**:
```python
ext_ctor = find_extension(name)
assert ext_ctor is not None  # callable found
```

**Property validation**:
```python
assert ext.name == name  # registered name matches
```

## Error Testing

**No explicit error tests** in current test_extension.py. Pattern for testing error handling:

```python
def test_custom0_exception():
    class FailingROCC(isa.ROCC):
        def custom0(self, proc, insn, xs1, xs2):
            raise ValueError("test error")
    
    register_extension("failing", FailingROCC)
    ext = find_extension("failing")()
    # Behavior: exception caught at C++ boundary, logged to stderr, returns 0
    # (no exception re-raised to Python caller)
```

This tests that Python exceptions in custom0-3 don't crash the simulator (they're caught in py_rocc_t::custom0 at `src/main/cpp/riscv_extension.cc:92`).

## Coverage Gaps

**Areas not tested** in test_extension.py:

1. **RoCC instruction handlers**: `custom0`, `custom1`, `custom2`, `custom3` behavior
   - Dummy implementation doesn't call these, so trampoline behavior not validated
   - Would need a test extension that implements custom0-3 and validates return values

2. **get_disasms() implementation details**: Only validates length and type
   - Could add assertions on disasm structure, field values, string output

3. **get_csrs() implementation**: Not tested
   - Could add CSR registration and validation test

4. **Error propagation**: Exception handling in Python → C++ boundary
   - Could add test that raises exception in extension method and validates logging

5. **Extension reuse**: Registering multiple extensions, registering same name twice
   - Could add conflict detection test

**Testing recommendation**: Add RoCC handler tests with custom0-3 implementations that:
1. Modify processor register state (e.g., write to x-registers via rocc_insn_t)
2. Return non-zero values
3. Assert modifications are visible to test code after custom0-3 call

---

*Testing analysis: 2025-05-04*
