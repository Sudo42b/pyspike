---
phase: 08-multi-tile-dma-parity
plan: 02
subsystem: testing
tags: [vendor-asset-policy, wheel-packaging, golden-import, env-var-config, manifest, pyproject]

# Dependency graph
requires:
  - phase: 07-numba
    provides: scripts/import_vendor_golden.py --all + VENDOR_OPS_84 + tests/gtx/test_regression_fw_full_sweep.py 5-tier graceful skip
provides:
  - _find_elf 3-tier resolution (firmware/ -> elf/ -> vendor host tree via GTX_VENDOR_TEST_DIR)
  - inline GTX_DDR_REVERSED=1 propagation for vendor-rooted .elf paths
  - Tier 3b OPERAND_STAGING skip conditioned on `not is_vendor_elf` (vendor pre-built bypass)
  - import_vendor_golden.py --all skip-guard preserving P6 9-op md5 invariant
  - MANIFEST.in `prune tests/gtx/data/firmware` (D-07 wheel exclusion)
  - pyproject.toml [tool.setuptools.exclude-package-data] firmware/ exclusion (belt-and-suspenders)
  - tests/gtx/test_wheel_data_present.py::test_wheel_excludes_firmware_dir sentinel
affects: [08-04 (vendor sweep investigation will see real .elf+golden pairs), 08-03 (rebases on top of skip-guard), Wave 1 / Plan 04]

# Tech tracking
tech-stack:
  added: [zipfile (test-only, stdlib)]
  patterns: [3-tier path resolution with env-var override, lower-cased dict skip-guard, sentinel test for wheel contents]

key-files:
  created: []
  modified:
    - tests/gtx/test_regression_fw_full_sweep.py
    - scripts/import_vendor_golden.py
    - MANIFEST.in
    - pyproject.toml
    - tests/gtx/test_wheel_data_present.py

key-decisions:
  - "EXP/NEG (no hand-built .elf) used as 3rd-tier vendor-resolution probe; ABS not used because elf/abs.elf wins via candidates 1/2"
  - "Tier 3b skip relocated AFTER _find_elf and conditioned on `not is_vendor_elf` so vendor pre-built .elf bypasses the OPERAND_STAGING guard (D-11)"
  - "abs.hex md5 invariant guard implemented as VENDOR_TO_PYSPIKE_OPS_LOWER skip dict (preserves canonical pyspike op_name like add_vv vs add)"
  - "Sentinel wheel test uses pytest.skip when dist/ is empty (no `python -m build` required during plan execution)"

patterns-established:
  - "3-tier candidate resolution with env-var override: provides graceful degradation when vendor tree absent"
  - "is_relative_to() guard for env-var injection: scopes env mutation to vendor-rooted paths only (cross-test contamination guard)"
  - "MANIFEST.in prune + pyproject.toml exclude-package-data: belt-and-suspenders wheel exclusion (works under sdist + wheel paths)"

requirements-completed: [VTW-01, VTW-04]

# Metrics
duration: 9min
completed: 2026-05-10
---

# Phase 08 Plan 02: Vendor asset wire-up Summary

**3-tier `_find_elf` (firmware/ -> elf/ -> ${GTX_VENDOR_TEST_DIR}) + inline GTX_DDR_REVERSED env propagation + import_vendor_golden --all skip-guard + MANIFEST.in/pyproject.toml firmware/ wheel exclusion**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-05-10T13:41Z (plan execution start)
- **Completed:** 2026-05-10T13:50Z
- **Tasks:** 3 (all atomic commits)
- **Files modified:** 5

## Accomplishments

- `_find_elf('EXP')` resolves to `/mnt/e/14_NIGHTLY/pyspike/test/EXP/n1s16/n1s16_exp.elf` (3rd-tier vendor candidate)
- `_find_elf('EXP')` returns `None` when `GTX_VENDOR_TEST_DIR` points to a nonexistent path (graceful degradation)
- `GTX_DDR_REVERSED=1` injected only when `elf_path.is_relative_to(vendor_root_for_env)` (no fixture contamination)
- `OPERAND_STAGING_REQUIRED_VENDOR` skip now bypasses for vendor pre-built `.elf` (D-11) — sweep harness can run vendor ABS/RELU/SIGMOID without hitting the operand-staging guard
- `import_vendor_golden.py --all` ran with skip-guard active: 73 converted, 11 skipped (84 total walk = full VENDOR_OPS_84 coverage); md5 invariant verified for all 9 P6 goldens
- `tests/gtx/data/firmware/` excluded from wheel via MANIFEST.in `prune` + pyproject.toml `[tool.setuptools.exclude-package-data]`
- `test_wheel_excludes_firmware_dir` sentinel passes (skip when no built wheel; will assert exclusion when invoked after `python -m build --wheel`)

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 0 parallel discipline):

1. **Task 1: _find_elf 3-tier vendor candidate + inline GTX_DDR_REVERSED** — `759cfa7` (feat)
2. **Task 2: import_vendor_golden --all skip-guard** — `2f5815e` (feat)
3. **Task 3: MANIFEST.in/pyproject.toml firmware exclusion + sentinel test** — `95aeee8` (feat)

## Files Created/Modified

- `tests/gtx/test_regression_fw_full_sweep.py` — `_find_elf` extended with vendor 3rd-tier candidate; subprocess env block now sets `GTX_DDR_REVERSED=1` for vendor-rooted `.elf`; Tier 3b OPERAND_STAGING skip conditioned on `not is_vendor_elf`
- `scripts/import_vendor_golden.py` — `--all` mode now consults `VENDOR_TO_PYSPIKE_OPS_LOWER` skip dict before walking VENDOR_OPS_84; preserves canonical pyspike op_name (e.g. `add_vv` over lower-cased `add`)
- `MANIFEST.in` — added `prune tests/gtx/data/firmware`; preserves `recursive-include tests/gtx/data/golden *.hex` and `recursive-include tests/gtx/data/elf`
- `pyproject.toml` — added `[tool.setuptools.exclude-package-data]` table with wildcard package match for `tests/gtx/data/firmware/*` (cp310-cp312 compatible — no `tomllib` dependency)
- `tests/gtx/test_wheel_data_present.py` — added `test_wheel_excludes_firmware_dir` sentinel inspecting `dist/spike-*.whl` namelist for any `tests/gtx/data/firmware/` entries

## Verification Evidence

### Static checks (all pass)

```
GTX_VENDOR_TEST_DIR mentions in test_regression_fw_full_sweep.py: 3 (>= 2 required)
is_relative_to mentions in test_regression_fw_full_sweep.py: 2 actual code uses (>= 1 required)
vendor_root.*op_dir.*n1s16 pattern matches: 1 (>= 1 required)
env["GTX_DDR_REVERSED"] = "1" lines: 1 (== 1 required)
prune tests/gtx/data/firmware in MANIFEST.in: 1 match
recursive-include tests/gtx/data/golden *.hex preserved: 1 match
recursive-include tests/gtx/data/elf preserved: 1 match
pyproject.toml exclude-package-data + firmware reference: 3 lines (>= 2 required)
def test_wheel_excludes_firmware_dir: 1 match
```

### Behavior checks

```
$ python scripts/import_vendor_golden.py --all --verify | tail -1
--all summary: 73 converted, 11 skipped/missing.

$ python scripts/import_vendor_golden.py --all --verify | grep -c -E 'DRY|SKIP'
84

$ ls tests/gtx/data/golden/*.hex | wc -l
89

$ du -sk tests/gtx/data/golden/ | awk '{print $1}'
4

$ find tests/gtx/data/golden/ -name '*.hex' -size +5k | wc -l
0

$ grep -l '# Source: vendor/gtx_cpp_reference/test/' tests/gtx/data/golden/*.hex | wc -l
85
```

### abs.hex md5 invariant (D-06 MEDIUM-2)

Captured BEFORE running `--all`:
```
17c69bcb99b374653727eb41679f5fc1  tests/gtx/data/golden/abs.hex
2d4f9abc923fab712b83308fdbb1a517  tests/gtx/data/golden/relu.hex
1a89f29ae6c63f509c448f3b31d25b86  tests/gtx/data/golden/add_vv.hex
9720ca081e4ccf9346995d80becccd08  tests/gtx/data/golden/sigmoid.hex
6a6f58a819af0f174efab7b91a0be511  tests/gtx/data/golden/tanh.hex
2996404f64510863da0a51d0261f3d48  tests/gtx/data/golden/softmax.hex
33afb9ae7de1dc50ae68f01ab3779b97  tests/gtx/data/golden/sum.hex
853546666a347ced128cd5ef98dbe63a  tests/gtx/data/golden/mul_vv.hex
d6347c969feb0bcba74e5cfb3fb1dcad  tests/gtx/data/golden/leaky_relu.hex
```

Captured AFTER running `--all` write mode:
```
(byte-identical to BEFORE — diff returned 'Files are identical')
```

**Result: MD5 INVARIANT HOLDS for all 9 P6 goldens.**

### EXP 3rd-tier resolution (D-05/D-13)

```python
$ python -c "import os; os.environ['GTX_VENDOR_TEST_DIR']='/mnt/e/14_NIGHTLY/pyspike/test/'
import sys; sys.path.insert(0, 'tests')
from gtx.test_regression_fw_full_sweep import _find_elf
p = _find_elf('EXP'); print(p)"
/mnt/e/14_NIGHTLY/pyspike/test/EXP/n1s16/n1s16_exp.elf
```

Graceful degradation when vendor_root absent:
```python
$ python -c "import os; os.environ['GTX_VENDOR_TEST_DIR']='/nonexistent/path/that/does/not/exist'
... print(_find_elf('EXP'))"
None
```

### Sentinel test (skips cleanly with no built wheel)

```
$ python -m pytest tests/gtx/test_wheel_data_present.py::test_wheel_excludes_firmware_dir --no-cov
collected 1 item
tests/gtx/test_wheel_data_present.py s        [100%]
============== 1 skipped in 0.32s ==============

$ python -m pytest tests/gtx/test_wheel_data_present.py --no-cov
collected 5 items
tests/gtx/test_wheel_data_present.py ....s    [100%]
====== 4 passed, 1 skipped in 0.76s ======
```

### pytest collection (84 vendor ops parametrize)

```
$ python -m pytest tests/gtx/test_regression_fw_full_sweep.py --collect-only --no-cov | tail -3
        <Function test_vendor_op_sweep_strict[XIELU]>
========================= 84 tests collected in 0.62s ==========================
```

## File count change in tests/gtx/data/golden/

- BEFORE Task 2: 87 .hex files (committed in 6de3ed4 "feat(07-05): GREEN-fill --all + import 76 vendor goldens")
- AFTER Task 2 (re-run --all with skip-guard): 89 .hex files
- The +2 delta corresponds to ops that previous sweeps missed (the script is now idempotent and captures new vendor _ref.txt as they appear in the cpp_reference test/ tree)
- Total size: 4 KB (well under 100 KB ceiling)
- All files <= 5 KB per entry (no oversized goldens)

## Decisions Made

- **EXP chosen over ABS for 3rd-tier vendor probe** — `tests/gtx/data/elf/abs.elf` exists (P5/P6 hand-built) and would win via candidates 1/2, masking the new vendor candidate. EXP/NEG have no hand-built `.elf` at any tier 1/2 location, so they fall through to the 3rd-tier vendor candidate cleanly.
- **vendor_root_for_env computed once** — single pathlib.Path() construction shared by Tier 3b skip-discrimination and subprocess env decision. Avoids double parsing of `GTX_VENDOR_TEST_DIR`.
- **Skip-guard implementation: lower-cased dict** — `{k.lower(): v for k, v in VENDOR_TO_PYSPIKE_OPS.items()}` is constructed once before the VENDOR_OPS_84 walk. Skipping silently with a "covered by P6 9-op default map" message keeps the summary line readable.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None. The skip-guard logic was added proactively per the plan's NOTES block (Task 2 instructed "If the script does NOT already implement this guard, add it now"). The script did NOT have it, so I added it.

## Next Phase Readiness

- Wave 1 / Plan 04 (vendor sweep investigation) can now execute with `GTX_VENDOR_TEST_DIR=/mnt/e/14_NIGHTLY/pyspike/test/` and find real vendor `.elf` + golden pairs at the 3rd-tier candidate location.
- Plan 08-03 will rebase on top of the skip-guard logic when adding `--full` flag and OP_DUMP_SIZE_OVERRIDE prefill — the regions modified by 08-02 and 08-03 are non-overlapping (08-02 owns the `--all` walk; 08-03 owns the `--full` flag + per-op dump size).
- Wheel size verification deferred to Wave 2 manual gate (`python -m build --wheel; du -h dist/spike-*.whl`); the sentinel test will assert exclusion at that point.

## Self-Check: PASSED

- tests/gtx/test_regression_fw_full_sweep.py modifications (Task 1 commit `759cfa7`): FOUND
- scripts/import_vendor_golden.py modifications (Task 2 commit `2f5815e`): FOUND
- MANIFEST.in modifications (Task 3 commit `95aeee8`): FOUND
- pyproject.toml modifications (Task 3 commit `95aeee8`): FOUND
- tests/gtx/test_wheel_data_present.py modifications (Task 3 commit `95aeee8`): FOUND
- All 3 task commits exist on `main` branch: VERIFIED via `git log --oneline`

---
*Phase: 08-multi-tile-dma-parity*
*Plan: 02*
*Completed: 2026-05-10*
