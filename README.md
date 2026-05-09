# Python Bindings of Spike RISC-V ISA Simulator

```text
LIU Yu <liuy@huimtlab.org>
2026/3/1 (v0.0.5)
```

## Introduction

This project provides Python bindings for the [Spike RISC-V ISA Simulator](https://github.com/riscv-software-src/riscv-isa-sim). The Pythonic Spike (or PySpike) opens up Spike's C++ internals (such as RISC-V disassembler, processors, controllers, peripherals, etc.) for interoperation with Python scripts. It enables users to write ISA / RoCC extensions and MMIO device models in Python, and plug them into vanilla Spike for (co-)simulating complex hardware systems. Through integrating Spike more seamlessly into the Python ecosystem, PySpike aims to boost the agility of Python-based hardware verification tools and workflows.

PyPI package: [`spike`](https://pypi.org/project/spike/)


## Getting Started

PySpike requires: Python 3.8+.

Install the wheel package with `pip`.

```shell
$ pip install --pre spike
```

PySpike ships the original command-line tool `spike`, a.k.a *vanilla Spike*, within its wheel package. You can confirm its availability using,

```shell
$ spike --help
Spike RISC-V ISA Simulator 0.0.5...
...
```

There is also a 100%-compatible command-line wrapper called `pyspike`, with additional support for Python-based ISA / MMIO / RoCC extensions via `--extlib=<name>`.

```shell
$ pyspike \
    --isa=rv32imc_xmyisa --priv=m \
    --pc=0x90000000 \
    -m0x90000000:0x4000000 \
    --extlib=myisa.py \
    --extlib=mydev.py \
    --device=mydev,0x20000000 \
    tests/data/libc-printf_hello.elf
Hello, World!
```

## Performance acceleration (optional)

`pip install spike` ships with a NumPy-only backend that passes the full
bit-exact regression suite. For wall-clock acceleration on GTX NPU compute
kernels, install the optional `[fast]` extras:

```shell
$ pip install spike[fast]
```

This installs [numba](https://numba.pydata.org/) (LLVM JIT compiler) and
enables `@njit(cache=True)` on the 28 stateless GTX kernels (`gemm_core`,
`vec_core` 7 ops, `act_core` 18 ops). The integration is **transparent** —
the same Python API works whether numba is installed or not. The first
invocation pays a one-shot compile cost (~640 ms per kernel; ~17.9 s
aggregate); subsequent runs hit the disk cache (`__pycache__/*.nbi`).

Empirical speedups (verified on x86_64 manylinux2014):
- `gemm_core` 16x16x16 FP16: ~455x (910 µs → 2 µs)
- Full vendor 84-op `n1s16` regression sweep: >= 5x

**Bit-exactness is preserved** — `fastmath=False` (numba default) +
explicit FP32 Python for-loop accumulate + `with numba.objmode(...)`
escape for 5 transcendental kernels (gelu, tanh_act, sigmoid, softmax,
esum) match NumPy oracle byte-for-byte (ULP-0 parity). Per-kernel parity
test: `pytest tests/gtx/test_njit_parity.py -v`.

**Disable acceleration** without uninstalling:

```shell
# Either uninstall numba, or set the env var (next major version):
$ pip uninstall numba
# The `HAS_NUMBA = False` path takes over automatically.
```

### Quick ISA Extension

An ISA extension implements one or more custom instructions and / or control-state registers (CSRs) for Spike's RISC-V processor models. With PySpike, an ISA extension is a Python class that inherits `riscv.isa.ISA`. It should implement a minimum of two methods: `get_instructions` and `get_disasms`. The former provides functional models of one or more custom instructions, and the latter provides their disassemblers. Other optional methods include `get_csrs` and `reset`, for providing custom CSRs and resetting extension states, respectively. Use decorator `@isa.register("myisa")` to register the extension under the name `myisa`.

```python
from typing import List
from riscv import isa
from riscv.csrs import csr_t
from riscv.disasm import disasm_insn_t
from riscv.processor import insn_desc_t, processor_t

@isa.register("myisa")
class MyISA(isa.ISA):
    def __init__(self): ...
    def get_instructions(self, proc: processor_t) -> List[insn_desc_t]: ...
    def get_disasms(self, proc: processor_t) -> List[disasm_insn_t]: ...
    def get_csrs(self, proc: processor_t) -> List[csr_t]: ...
    def reset(self, proc: processor_t) -> None: ...
```

### Quick Device Model

Likewise to the ISA extension, a device model implements a custom *memory-mapped input/output* (MMIO) peripheral for Spike's simulated system bus. With PySpike, a device model is a Python class that inherits `riscv.dev.MMIO`. It should implement a minimum of three methods: `__init__`, `load`, and `store`. The former initializes the model, the latter two handle memory read and write operations. Other optional methods include `size` and `tick`, for obtaining the size of memory-mapped address space, and shifting device states, respectively. Use decorator `@dev.register("mydev")` to register the model under the name `mydev`.

```python
from typing import Optional
from riscv import dev
from riscv.sim import sim_t

@dev.register("mydev")
class MyDEV(dev.MMIO):
    def __init__(self, sim: sim_t, args: Optional[str] = None): ...
    def load(self, addr: int, size: int) -> bytes: ...
    def store(self, addr: int, data: bytes) -> None: ...
    def size(self) -> int: ...
    def tick(self, rtc_ticks: int) -> None:
```

## Development

### Getting Source Code

```shell
$ git clone --recurse-submodules https://github.com/huimtlab/pyspike
$ cd pyspike
```

### Setting Up Develop Environment

Install with `pip` in *editable* mode. This will setup development dependencies as well.

```shell
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ python -m pip install -e '.[dev]'
```

### Running Tests

Run the built-in test suite with `pytest`.

```shell
(.venv) $ python -m pytest -v
```

### Running Tests with Coverage

To enable coverage report, rebuild the extension module with `--cov` option, then re-run the test suite also with `--cov` option. You should see the coverage report by the end of the test output.

Note, if leaving out the first step, the C++ code will *not* show up in the coverage report.

```shell
(.venv) $ python setup.py build_ext --inplace --cov
(.venv) $ python -m pytest -v --cov
```

Optionally, you can generate HTML coverage report from the `lcov` data files.

```shell
(.venv) $ genhtml -o coverage --substitute "s#^#$PWD/#g" *.lcov
```

### Packaging

```shell
(.venv) $ python -m build
```
