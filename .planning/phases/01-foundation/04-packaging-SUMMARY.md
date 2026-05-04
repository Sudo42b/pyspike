---
phase: 01-foundation
plan: 04
subsystem: packaging
tags: [pyproject-toml, cibuildwheel, numpy, cp310, packages-find, wheel-discovery]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: D-07 (numpy>=2.0,<3) / D-08 (cp310+ baseline) / D-09 (FP16 view) lock-in from CONTEXT.md; Plan 05 vendor/gtx_cpp_reference submodule registered
provides:
  - "pyproject.toml `[project].dependencies = ['numpy>=2.0,<3']` (D-07 lock-in)"
  - "pyproject.toml `[project].requires-python = '>=3.10'` (D-08 lock-in)"
  - "pyproject.toml `[tool.cibuildwheel].build` matrix reduced to cp310/cp311/cp312"
  - "pyproject.toml `[tool.setuptools.packages.find].include = ['riscv', 'riscv.*']` — riscv.gtx subpackage discovery (RESEARCH.md Critical Finding)"
  - "pyproject.toml `[tool.cibuildwheel.linux].before-all` chains `git submodule update --init --recursive`"
affects:
  - 02-rocc-dispatch / Phase 2-6 — wheel will now contain riscv.gtx (silent-fail averted)
  - cibuildwheel CI runs — manylinux container will initialize both vendor/spike and vendor/gtx_cpp_reference before build

# Tech tracking
tech-stack:
  added:
    - "numpy>=2.0,<3 (declared runtime dep in pyproject.toml [project].dependencies; D-07)"
  patterns:
    - "PEP 621 ordering: requires-python -> blank line -> dependencies -> blank line -> [project.urls]"
    - "[tool.setuptools.packages.find].include uses ['riscv', 'riscv.*'] glob (NOT bare ['riscv']; setuptools 80.9.0 does NOT auto-discover subpackages from a single name)"
    - "cibuildwheel before-all: shell `&&` chain (yum install + git submodule update --init --recursive) — minimal addition, no wrapper script"

key-files:
  created: []
  modified:
    - pyproject.toml

key-decisions:
  - "Adopted RESEARCH.md 'Critical Finding' fix exactly: `include = ['riscv', 'riscv.*']` — empirically verified that `find_packages(where='src/main/python', include=['riscv'])` returns only `['riscv']` while `include=['riscv', 'riscv.*']` returns `['riscv', 'riscv.gtx', 'riscv.gtx.ops']`. Old glob would have shipped wheels without riscv.gtx, silently breaking Phase 2-6 user-facing import."
  - "Locked NumPy as runtime dep with `numpy>=2.0,<3` (upper bound prevents NumPy 3.x ABI break; lower bound matches D-07 NumPy 2.x FP16 RNE requirement)."
  - "cibuildwheel linux before-all chained via `&&` (single-line). No new wrapper script needed; both vendor submodules are initialized by `git submodule update --init --recursive` (recursive flag handles nested submodules within vendor/spike if any)."
  - "Verified pyproject.toml correctness via Python 3.11 tomllib (system Python 3.10 lacks tomllib; used /home/sw.lee/.local/bin/python3.11 — note for future executors)."
  - "Wheel build acceptance verified via sdist + setuptools.find_packages introspection (canonical wheel build blocked by pre-existing C++ extension issue; documented as known limitation)."

patterns-established:
  - "Pattern A (Critical) — packages.find recursive include glob: `['riscv', 'riscv.*']` is the canonical setuptools 80.9.0 form. Bare `['riscv']` does NOT auto-discover subpackages."
  - "Pattern B — PEP 621 dependency block placement: `dependencies = [...]` placed between `requires-python` and `[project.urls]` with blank-line separators (matches RESEARCH.md Open Questions #1 recommendation)."

requirements-completed: [PKG-02, FOUND-03]

# Metrics
duration: 36m22s
completed: 2026-05-04
---

# Phase 01 Plan 04: Packaging Summary

**Five-stanza pyproject.toml patch to lock in NumPy 2.x runtime dep, cp310+ baseline, cibuildwheel matrix reduction (cp310-cp312), classifiers cleanup, the canonical `packages.find.include = ['riscv', 'riscv.*']` glob fix (RESEARCH.md Critical Finding), and the cibuildwheel before-all chain enabling vendor/gtx_cpp_reference submodule init in CI.**

## Performance

- **Duration:** 36m22s
- **Started:** 2026-05-04T05:49:39Z
- **Completed:** 2026-05-04T06:26:01Z
- **Tasks:** 2 / 2
- **Files modified:** 1 (`pyproject.toml`)

## Accomplishments

- **Edit 1 (Task 04-01):** `[tool.cibuildwheel].build` reduced from 5 wheels (cp38..cp312) to 3 wheels (cp310/cp311/cp312). cp38/cp39 lines removed.
- **Edit 2 (Task 04-01):** `[tool.cibuildwheel.linux].before-all` extended with `&& git submodule update --init --recursive` so both `vendor/spike` (existing) and `vendor/gtx_cpp_reference` (Plan 05) initialize inside the manylinux container.
- **Edit 3 (Task 04-01):** `[project].classifiers` cleaned — Python 3.8 / 3.9 lines removed, leaving 3.10/3.11/3.12.
- **Edit 4 (Task 04-01):** `requires-python = ">=3.10"` (was `">=3.8"`); new `dependencies = ["numpy>=2.0,<3"]` block added between `requires-python` and `[project.urls]`.
- **Edit 5 (Task 04-01) — ★ CRITICAL:** `[tool.setuptools.packages.find].include` changed from `["riscv"]` to `["riscv", "riscv.*"]`. Empirically the most important single line in Phase 1 — without it, `pip install spike` would deliver wheels missing `riscv.gtx` and `riscv.gtx.ops`, breaking every downstream user import.
- **Verified (Task 04-02):** sdist build (`python3.11 -m build --sdist`) succeeds and `dist/spike-0.0.5.dev52.tar.gz` (852K) contains all 7 `riscv.gtx` files; `vendor/gtx_cpp_reference` count = 0.

## Task Commits

Each task committed atomically with normal commits (pre-commit hooks ran):

1. **Task 04-01: pyproject.toml 5-stanza patch (NumPy 2.x / cp310 / packages.find glob fix)** — `cbd1487` (chore)
2. **Task 04-02: wheel discovery verification record (sdist + find_packages introspection)** — `f3c3b7a` (chore, --allow-empty)

_Plan metadata commit follows this SUMMARY._

## Files Created/Modified

### Modified

- **`pyproject.toml`** (5 stanza patches; net `+8 -7` lines):

#### Diff 1 — cibuildwheel build matrix
```diff
 [tool.cibuildwheel]
 build = [
-  "cp38-manylinux_x86_64",
-  "cp39-manylinux_x86_64",
   "cp310-manylinux_x86_64",
   "cp311-manylinux_x86_64",
   "cp312-manylinux_x86_64"
 ]
```

#### Diff 2 — cibuildwheel before-all chain
```diff
 [tool.cibuildwheel.linux]
-before-all = "yum install -y dtc"
+before-all = "yum install -y dtc && git submodule update --init --recursive"
```

#### Diff 3 — classifiers
```diff
   "Programming Language :: Python :: 3",
-  "Programming Language :: Python :: 3.8",
-  "Programming Language :: Python :: 3.9",
   "Programming Language :: Python :: 3.10",
   "Programming Language :: Python :: 3.11",
   "Programming Language :: Python :: 3.12",
```

#### Diff 4 — requires-python + new dependencies block
```diff
-requires-python = ">=3.8"
+requires-python = ">=3.10"
+
+dependencies = [
+  "numpy>=2.0,<3",
+]

 [project.urls]
```

#### Diff 5 — packages.find recursive glob (★ CRITICAL)
```diff
 [tool.setuptools.packages.find]
 where = [
   "src/main/python"
 ]
 include = [
-  "riscv"
+  "riscv",
+  "riscv.*"
 ]
```

## Verification Outputs

### Task 04-01: tomllib assertions (combined + per-criterion, all PASS)

```
$ python3.11 -c "import tomllib; t=tomllib.load(open('pyproject.toml','rb')); ..."
OK pyproject.toml validation passed

$ python3.11 -c "... assert any('numpy>=2.0' in d for d in t['project']['dependencies']) ..."
1/4 numpy OK
2/4 requires-python OK
3/4 cibuildwheel matrix OK
4/4 packages.find include OK
```

### Task 04-01: grep assertions (all PASS)

```
$ grep -F '"riscv.*"' pyproject.toml             # exit 0 (Edit 5 applied)
$ grep -F 'git submodule update --init --recursive' pyproject.toml  # exit 0 (Edit 2)
$ grep -F '"cp38-manylinux_x86_64"' pyproject.toml  # exit 1 (cp38 removed)
$ grep -F '"cp39-manylinux_x86_64"' pyproject.toml  # exit 1 (cp39 removed)
$ grep -F 'Python :: 3.8' pyproject.toml         # exit 1 (classifier removed)
$ grep -F 'Python :: 3.9' pyproject.toml         # exit 1 (classifier removed)
$ python3.11 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"  # TOML OK
```

### Task 04-02: setuptools.find_packages empirical verification

The single most important verification — proves Edit 5 is correct on disk:

```
$ python3.11 -c "from setuptools import find_packages; print(sorted(find_packages(where='src/main/python', include=['riscv', 'riscv.*'])))"
['riscv', 'riscv.gtx', 'riscv.gtx.ops']

$ python3.11 -c "from setuptools import find_packages; print(sorted(find_packages(where='src/main/python', include=['riscv'])))"
['riscv']
```

The new glob discovers all three packages; the old glob (research's RESEARCH.md "Critical Finding") only finds the top-level `riscv` package — which would silent-fail Phase 2-6 since `riscv.gtx` would be missing from the wheel.

### Task 04-02: sdist build acceptance

```
$ python3.11 -m build --sdist
... Successfully built spike-0.0.5.dev52.tar.gz

$ tar tzf dist/spike-0.0.5.dev52.tar.gz | grep -E 'riscv/gtx/'
spike-0.0.5.dev52/src/main/python/riscv/gtx/
spike-0.0.5.dev52/src/main/python/riscv/gtx/__init__.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/ddr.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/encoding.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/fp.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/memory.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/ops/
spike-0.0.5.dev52/src/main/python/riscv/gtx/ops/__init__.py
spike-0.0.5.dev52/src/main/python/riscv/gtx/params.py

$ tar tzf dist/spike-0.0.5.dev52.tar.gz | grep -c gtx_cpp_reference
0
```

All 7 `riscv.gtx` artifacts present in sdist. `vendor/gtx_cpp_reference` excluded (D-06 prune from MANIFEST.in working as designed; Plan 05 deliverable).

## Decisions Made

- **Used the recursive include glob `["riscv", "riscv.*"]` (not `["riscv*"]`).** Both forms work for our case but the explicit two-element form mirrors the RESEARCH.md "Example 3: Full pyproject.toml deltas" snippet exactly and is more legible to humans reviewing the diff. Either form would satisfy the acceptance regex (`'riscv.*' in inc or 'riscv*' in inc`).
- **NumPy bound: `numpy>=2.0,<3`.** Adopted `,<3` upper bound (CONTEXT.md "Claude's Discretion" recommended `numpy>=2.0,<3` over bare `numpy>=2.0`). Defends against NumPy 3.x C-API break that would invalidate downstream FP16 view assumptions.
- **Python 3.11 used for tomllib verification.** System python3 is Python 3.10 (no tomllib). `/home/sw.lee/.local/bin/python3.11` (uv-managed) was used for `python3.11 -c "import tomllib; ..."`. Note for future executors: this verification step requires Python 3.11+; on Python 3.10-only systems use `pip install tomli; python3 -c "import tomli; ..."` instead.
- **PEP 621 dependency block placement.** Placed `dependencies = [...]` between `requires-python` and `[project.urls]` per CONTEXT.md Open Questions #1 / RESEARCH.md recommendation. Preserves blank-line separation and keeps related metadata grouped at the top of the `[project]` table.

## Deviations from Plan

### [Rule 3 - Blocking] Wheel build environment limitation

- **Found during:** Task 04-02 (canonical `pip wheel . -w /tmp/wheel-test/` execution)
- **Issue:** `pip wheel . -w /tmp/wheel-test/ --no-deps` fails at the C++ extension compile step with pybind11 3.0.4 static-assertion errors:
  ```
  error: static assertion failed: Cannot bind an inaccessible base class method;
         use a lambda definition instead
         (in pybind11/pybind11.h:2006, instantiated from py_module.cc:90)
  ```
  This is a pre-existing pybind11 3.0.4 / `csr_t` binding inaccessibility in `src/main/cpp/py_module.cc` (lines 90-91). The pybind11 version chosen by the build's PEP 517 isolated env is 3.0.4 (latest), which has stricter `is_accessible_base_of` checks than the bound-protected `py_csr_t::*` member-pointer cast pattern in the existing C++ binding code expected.
- **Why out of scope for this plan:** This is a pre-existing C++ binding/pybind11-version incompatibility unrelated to the pyproject.toml patches. Plan 04's scope is **packaging metadata**, not C++ extension compile fixes (CLAUDE.md "no new C++ code" + plan's `<files>pyproject.toml</files>` scope marker).
- **Workaround applied (Rule 3 alternative verification):** Verified the canonical Plan 04 outcome using two equivalent paths that do not require the C++ extension to compile:
  1. `setuptools.find_packages(where='src/main/python', include=['riscv', 'riscv.*'])` empirically returns `['riscv', 'riscv.gtx', 'riscv.gtx.ops']` — direct proof Edit 5 is correct.
  2. `python3.11 -m build --sdist` succeeds and sdist contains all 7 `riscv.gtx` files (FOUND-03 wheel-discovery requirement met at the source-distribution layer; the wheel layer behaves identically once the C++ build succeeds).
- **Plan-specific guidance acknowledged this:** "If `pip wheel` fails due to environment ... document in SUMMARY.md as a known environment limitation; the static tomllib checks above are sufficient to validate the patch correctness. The full wheel build is canonical-verified by CI (cibuildwheel) on the next push."
- **Logged for follow-up (deferred):** Add to phase deferred-items: "pybind11 3.0.4 csr_t binding inaccessibility blocks local pip wheel build; needs lambda-wrapper fix in src/main/cpp/py_module.cc:90-91 OR pybind11 pin to <3.0.4 — CI cibuildwheel chain may already handle this if the manylinux2014 image has a different pybind11."

### Auto-fixed Issues

None — all 5 pyproject.toml edits applied exactly as specified in the plan's `<action>` block. No additional bug fixes were necessary; no missing functionality was introduced; no architectural changes proposed.

**Total deviations:** 1 environmental limitation (out-of-scope, documented and worked around). 0 in-scope auto-fixes applied.

## Issues Encountered

- **System python3 lacks tomllib.** Python 3.10 is the system default (`/usr/bin/python3 → python3.10`); tomllib was added in Python 3.11. Used `/home/sw.lee/.local/bin/python3.11` (uv-managed) for verification. Documented in Decisions Made section as a hint for future executors.
- **System pip 22.0.2 has stale typing_extensions.** First wheel-build attempt with `python3 -m pip wheel` failed because the system pip 22's PEP 517 isolated build env imported `Self` from typing_extensions at the wrong version. Switched to `python3.11 -m pip` (pip 26.0.1) which bypassed that. Then hit the pybind11 csr_t issue (separate, pre-existing).
- **Pre-existing pybind11 3.0.4 / csr_t binding incompatibility** — the actual blocker for the canonical wheel build path. Logged as out-of-scope deviation above.

## Authentication Gates

None — all packaging operations are local (no network credentials required).

## Known Stubs

None. All five pyproject.toml edits are production-ready declarations; no placeholder or TODO patterns. The "deferred" pybind11 csr_t fix is a pre-existing issue **outside this plan's scope** (CLAUDE.md "no new C++ code" + plan's `<files>` scope is `pyproject.toml` only).

## User Setup Required

None — `numpy` is fetched transparently by `pip install spike` once published, and `git submodule update --init --recursive` runs anonymously inside the cibuildwheel manylinux container against the public `https://github.com/Sudo42b/gtx_spike` URL (Plan 05).

## Deferred Items (logged for phase deferred-items.md)

- **pybind11 3.0.4 / csr_t binding inaccessibility** in `src/main/cpp/py_module.cc:90-91`. Local `pip wheel .` fails with `static_assert "Cannot bind an inaccessible base class method"` errors. Either (a) wrap the affected `py_csr_t::*` member-pointer bindings in lambdas, or (b) pin `pybind11<3.0.0` in build-system requires. CI cibuildwheel may have this baked into the manylinux2014 image at a known-working pybind11 version; verification deferred to next CI push. **Tracking ticket:** to be opened against pyspike core (not GTX); not blocking Phase 1 acceptance because the pyproject.toml fix is empirically correct via sdist + setuptools.find_packages.

## Next Phase / Plan Readiness

- **Phase 1 wave 2 conclusion:** With Plan 04 (this plan) + Plan 05 (vendor submodule + MANIFEST.in prune) complete, Phase 1 packaging baseline is locked. ROADMAP success criterion 5 ("`pyproject.toml` 선언 + 유효한 manylinux2014_x86_64 wheel") is satisfied at the metadata layer; the binary wheel layer is gated on the pre-existing pybind11 issue and CI cibuildwheel run.
- **Phase 2 readiness:** Plan 04 silently unblocks Phase 2-6 by making `riscv.gtx` discoverable. Without Edit 5, all subsequent phases would silently fail at user-import time post-`pip install spike`. RESEARCH.md "Critical Finding" risk fully neutralized.
- **CI verification (manual, deferred):** On next `git push`, the cibuildwheel GitHub Actions matrix runs cp310/cp311/cp312 wheel builds in the manylinux2014_x86_64 container. The chained `before-all` will execute `yum install -y dtc && git submodule update --init --recursive`, initializing both vendor/spike and vendor/gtx_cpp_reference. Post-CI: confirm wheel contents include `riscv/gtx/__init__.py` via `auditwheel show`.

## Self-Check: PASSED

- File `pyproject.toml` modified: VERIFIED (`git diff HEAD~2..HEAD~1 -- pyproject.toml` shows the 5-stanza diff)
- Commit `cbd1487` (Task 04-01): FOUND in `git log --oneline`
- Commit `f3c3b7a` (Task 04-02): FOUND in `git log --oneline`
- All 4 tomllib per-criterion assertions: PASS
- All 7 grep acceptance checks: PASS
- `setuptools.find_packages(include=['riscv', 'riscv.*'])` returns 3 packages including `riscv.gtx`: VERIFIED
- sdist contains all 7 `riscv.gtx` files: VERIFIED
- `vendor/gtx_cpp_reference` count in sdist == 0 (D-06): VERIFIED
- TOML syntax valid: VERIFIED (`python3.11 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` exit 0)

---

*Phase: 01-foundation*
*Plan: 04-packaging*
*Completed: 2026-05-04*
