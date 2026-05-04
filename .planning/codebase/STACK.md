# Technology Stack

**Analysis Date:** 2026-05-04

## Languages

**Primary:**
- **C++** (C++20 / C++2a) - RoCC trampoline and pybind11 binding layer; all py_* wrapper classes and glue code
- **Python** (3.8+) - User-facing extension API and RoCC subclass definitions

**Secondary:**
- **C** - Spike ISA simulator core (upstream, in `vendor/spike/`)

## Runtime

**Environment:**
- Python 3.8, 3.9, 3.10, 3.11, 3.12 (tested via cibuildwheel on manylinux2014_x86_64)
- RISC-V toolchain at `/opt/riscv` (default, overridable via `RISCV` env var)

**Package Manager:**
- pip / setuptools (Python side)
- Leverages pybind11 for C++ → Python bridging
- Lockfile: `pyproject.toml` and `setup.py`

## Frameworks

**Core:**
- **pybind11** [>3] - Binds C++ extension_t/rocc_t classes to Python; provides trampoline_self_life_support for virtual method overriding
  - Located: `src/main/cpp/riscv_extension.h:32` (py_extension_t), `src/main/cpp/riscv_extension.h:56` (py_rocc_t)
- **Spike RISC-V ISA Simulator** (upstream submodule at `vendor/spike/`) - Provides rocc_t base class and ISA infrastructure

**Build/Dev:**
- **setuptools** [>=75] - Build system
- **setuptools_scm** [>=9] - Version management (git-based)
- **pybind11.setup_helpers** - CMake-free pybind11 extension building

**Testing:**
- **pytest** - Test framework (referenced in `pyproject.toml`)
- **pytest-cov** - Coverage collection

## Key Dependencies

**Critical:**
- **libriscv.so** - Spike's RISC-V simulator library; linked at build time via `-lriscv` flag in `setup.py:55`
- **libdisasm.a** - Spike disassembler (static, bundled)
- **libfesvr.a** - Front-end server (static, bundled)
- **pybind11** [>3] - Mandatory for binding extension_t and rocc_t classes; supports trampoline_self_life_support for virtual method dispatch
  - Used in: `src/main/cpp/riscv_extension.h:26-28`, `src/main/cpp/py_module.cc:22-25`

**Infrastructure:**
- **auditwheel** - Binary wheel auditing (manylinux compliance)
- **patchelf** - ELF manipulation for bundled libraries
- **dtc** (device-tree-compiler) - Required before-all in cibuildwheel

## Configuration

**Environment:**
- `RISCV` env var sets toolchain prefix (default: `/opt/riscv`)
  - Used in `setup.py:33`, `setup.py:53, 60`
- `PYSPIKE_LIBS` - Name of env var for loading spike libraries dynamically (defined in `src/main/python/riscv/__init__.py:30`)
- `PYSPIKE_EXTS` - Name of env var for extension library paths (defined in `src/main/python/riscv/__init__.py:32`)

**Build:**
- C++ compilation with `-std=c++2a` (C++20) - `setup.py:45`
- pybind11 detailed error messages enabled via `PYBIND11_DETAILED_ERROR_MESSAGES=1` macro - `setup.py:49`
- Runtime library paths: `-Wl,-rpath,$ORIGIN/data/lib` and `-Wl,-rpath,{RISCV}/lib` - `setup.py:52-53`

## Platform Requirements

**Development:**
- GCC/Clang with C++20 support
- dtc (device-tree-compiler)
- RISC-V cross-toolchain at `$RISCV` (typically from github.com/riscv-collab/riscv-gnu-toolchain)

**Production:**
- Linux x86_64 (manylinux2014 baseline)
- Python 3.8+ shared library
- glibc 2.17+

---

*Stack analysis: 2026-05-04*
