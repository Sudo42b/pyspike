---
plan: 02-06
type: build-log
created: 2026-05-04T15:49:05Z
host: DESKTOP-ADRHA0T
kernel: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
status: in-progress
---

# Plan 02-06 Gap-Closure Build & Test Log

Captures evidence for the 4 tasks in `02-06-PLAN.md` (Wave 3 gap closure).
This log is the source of truth that 02-VERIFICATION.md, 02-HUMAN-UAT.md, and
02-06-SUMMARY.md cite by reference.

---

## Task 1 — Build `_riscv.so` via `pip install -e .`

**Started:** 2026-05-04T15:49:05Z
**Outcome:** SUCCESS

### Step 1.1 — Pre-flight checks

| Check | Command | Result |
|-------|---------|--------|
| Python | `python3 --version` | `Python 3.10.12` |
| pybind11 | `pip show pybind11` | `Version: 3.0.1` (NOT the broken 3.0.4) |
| RISC-V toolchain | `/opt/riscv/bin/riscv64-unknown-elf-gcc --version` | `riscv64-unknown-elf-gcc (GCC) 15.2.0` |
| libriscv.so | `ls src/main/python/riscv/data/lib/libriscv.so` | present (292.9M; built by Phase 1) |

All four pre-flight checks pass. Phase-1 spike core was already built — `_build_spike()` in setup.py:101 short-circuits on existing `data/include/riscv/encoding.h`.

### Step 1.2 — Build via editable install

**Initial attempt failed** with `ModuleNotFoundError: No module named 'setuptools_scm'` because `--no-build-isolation` skips fetching build dependencies. Resolved with:

```bash
python3 -m pip install --user setuptools_scm
# Successfully installed setuptools_scm-10.0.5 vcs-versioning-1.1.1
```

(Treated as a Rule 3 - Blocking deviation: build dependency missing.)

**Successful build command:**
```bash
python3 -m pip install -e . --no-build-isolation --user
```

**Final 16 lines of `/tmp/02-06-pip-install.log`:**
```
Obtaining file:///mnt/e/14_NIGHTLY/pyspike
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: numpy<3,>=2.0 in /home/sw.lee/.local/lib/python3.10/site-packages (from spike==0.0.5.dev85) (2.2.6)
Building wheels for collected packages: spike
  Building editable for spike (pyproject.toml): started
  Building editable for spike (pyproject.toml): still running...
  Building editable for spike (pyproject.toml): still running...
  Building editable for spike (pyproject.toml): finished with status 'done'
  Created wheel for spike: filename=spike-0.0.5.dev85-0.editable-cp310-cp310-linux_x86_64.whl size=13518 sha256=e9b2b26676ce49879772aadc6323b78b9ff54bffb306afd340c2a4ed4509a5bd
  Stored in directory: /tmp/pip-ephem-wheel-cache-ez6fzogi/wheels/df/18/fd/bc17a9ba44b4d263f3ccdd9d6b65319ddaa51e9cdff7baeb22
Successfully built spike
Installing collected packages: spike
Successfully installed spike-0.0.5.dev85
```

**Exit code:** 0

**Build artifact:** `src/main/python/riscv/_riscv.cpython-310-x86_64-linux-gnu.so` (1.5M)

### Step 1.3 — Verify `_riscv` import resolves

```bash
$ python3 -c "from riscv import _riscv; print(_riscv.__file__)"
/mnt/e/14_NIGHTLY/pyspike/src/main/python/riscv/_riscv.cpython-310-x86_64-linux-gnu.so
```
Exit code 0.

### Step 1.4 — Verify `GtxNpu` hydrates

```bash
$ python3 -c "from riscv.gtx import GtxNpu; assert GtxNpu is not None; print(GtxNpu)"
<class 'riscv.isa.register.<locals>.isa_decorator.<locals>.MyISA'>
```
Exit code 0. The class is the decorator-synthesized wrapper around `GtxNpu` produced by `@isa.register('gtx')` — confirming the registration path works end-to-end.

### Step 1.5 — Failure-mode discriminant

| Mode | Description | Triggered? |
|------|-------------|------------|
| F1   | pybind11 csr_t static_assert (deferred-items.md issue resurface) | NO |
| F2   | Linker error `cannot find -lriscv` | NO |
| F3   | Import OK but `GtxNpu is None` | NO |

**Note:** pybind11 3.0.4 issue (deferred-items.md) was avoided because the system has 3.0.1 installed and `--no-build-isolation` reused it. CI / cibuildwheel still needs a `pyproject.toml` `[build-system].requires` pin if the latest pybind11 is breaking — Phase-1 deferred-items concern remains valid for reproducibility, but for this gap-closure cycle the local dev environment is unblocked.

### Task 1 Summary

- `_riscv.cpython-310-x86_64-linux-gnu.so` built successfully (1.5M).
- `from riscv import _riscv` resolves.
- `from riscv.gtx import GtxNpu` resolves to a real class (was `None` due to graceful-degradation fallback before build).
- One auto-fixed deviation (Rule 3 - Blocking: missing `setuptools_scm` build dep — fix-attempt count: 1).

---

## Task 2 — Run all skipif-gated tests + zero-regression check

(populated after Task 2 execution)

---

## Task 3 — Subprocess CLI integration + trace inspection

(populated after Task 3 execution)

---

## Task 4 — UAT/VERIFICATION/ROADMAP doc-sync

(populated after Task 4 execution)
