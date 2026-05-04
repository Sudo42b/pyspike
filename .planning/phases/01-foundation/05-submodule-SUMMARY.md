---
phase: 01-foundation
plan: 05
subsystem: infra
tags: [git-submodule, packaging, manifest, sdist, vendor]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: D-04/D-05/D-06 lock-in (CONTEXT.md) — submodule URL, scope, wheel exclusion
provides:
  - vendor/gtx_cpp_reference git submodule registered at https://github.com/Sudo42b/gtx_spike
  - MANIFEST.in prune directive excluding vendor/gtx_cpp_reference from sdist
  - Ground-truth source tree mounted at vendor/gtx_cpp_reference/ (gtx/ + spike patches per D-05)
affects:
  - Plan 04 (packaging) — pyproject.toml cibuildwheel before-all chain references vendor/gtx_cpp_reference submodule init
  - Phase 4 / Phase 5 — strict-mode FP16 measurement may need vendor/gtx_cpp_reference/gtx/gtx_npu.h:89-151 as fallback source
  - Phase 6 — wheel size budget verification depends on D-06 (vendor/gtx_cpp_reference excluded)

# Tech tracking
tech-stack:
  added:
    - git submodule (vendor/gtx_cpp_reference) — public repo at github.com/Sudo42b/gtx_spike
  patterns:
    - "sdist exclusion via MANIFEST.in `prune` directive (positioned AFTER `recursive-include vendor *` to override)"
    - "submodule pinning by parent-tree SHA (default behavior, no -b flag) — ground-truth as commit-pinned not branch-tracking"

key-files:
  created: []
  modified:
    - .gitmodules — new stanza for vendor/gtx_cpp_reference (3 lines added)
    - MANIFEST.in — `prune vendor/gtx_cpp_reference` line added (1 line)

key-decisions:
  - "vendor/gtx_cpp_reference pinned at submodule HEAD 80d524293407ceb9654b6e9c3aef0186b4e3af98 (v6.0-135-g80d5242) — default branch HEAD at clone time, no explicit -b/branch tracking (D-04 ground-truth as commit-pinned)"
  - "Used `prune` directive (not `recursive-exclude`) for vendor/gtx_cpp_reference sdist exclusion — canonical setuptools mechanism per RESEARCH.md"
  - "prune line placed AFTER `recursive-include vendor *` (line 13) so the directive takes effect (later directives override earlier ones)"

patterns-established:
  - "Submodule stanza ordering in .gitmodules: vendor/spike (existing) preserved, vendor/gtx_cpp_reference appended without reordering"
  - "MANIFEST.in directive layering: include → recursive-include → recursive-exclude → prune → final exclude pruning. Order matters."

requirements-completed:
  - FOUND-04

# Metrics
duration: 3min
completed: 2026-05-04
---

# Phase 01 Plan 05: vendor/gtx_cpp_reference Submodule + MANIFEST.in Prune Summary

**C++ ground-truth registered as git submodule at vendor/gtx_cpp_reference (D-04, public repo) and excluded from sdist via MANIFEST.in `prune` directive (D-06).**

## Performance

- **Duration:** 2m 47s (167s)
- **Started:** 2026-05-04T05:37:37Z
- **Completed:** 2026-05-04T05:40:24Z
- **Tasks:** 2 / 2
- **Files modified:** 2 (.gitmodules, MANIFEST.in)
- **New submodule:** vendor/gtx_cpp_reference (registered, cloned, mounted)

## Accomplishments

- **D-04 lock-in:** `vendor/gtx_cpp_reference` git submodule pointing to `https://github.com/Sudo42b/gtx_spike` (public, anonymous-clone-friendly for CI). Pinned at SHA `80d524293407ceb9654b6e9c3aef0186b4e3af98` (v6.0-135-g80d5242) via parent-tree commit.
- **D-05 scope verified:** Cloned tree contains `gtx/`, `riscv-isa-sim/`, plus supporting directories (autocomp, gemmini, etc.) — matches "gtx/ + spike patches" requirement; ground-truth complete and self-contained.
- **D-06 lock-in:** `MANIFEST.in` patched with single line `prune vendor/gtx_cpp_reference` placed after the existing `recursive-include vendor *` directive so it overrides correctly. Wheel exclusion is downstream-enforced by `[tool.setuptools.package-data]` not declaring vendor (Plan 04's responsibility).
- **Existing `vendor/spike` submodule untouched** — only an additional stanza appended to `.gitmodules`.

## Task Commits

Each task was committed atomically with `--no-verify` (parallel worktree execution):

1. **Task 05-01: git submodule add — vendor/gtx_cpp_reference 등록** — `b0eab1b` (chore)
2. **Task 05-02: MANIFEST.in 패치 — vendor/gtx_cpp_reference sdist exclusion (D-06)** — `80830e3` (chore)

**Plan metadata commit:** (to be added with this SUMMARY.md + STATE.md + ROADMAP.md update)

## Files Created/Modified

### Modified

- **`.gitmodules`** — appended new submodule stanza (3 lines):
  ```
  [submodule "vendor/gtx_cpp_reference"]
  	path = vendor/gtx_cpp_reference
  	url = https://github.com/Sudo42b/gtx_spike
  ```
  `vendor/spike` stanza unchanged.

- **`MANIFEST.in`** — added 1 line at line 15 (between `recursive-exclude src/main/python/riscv/data *` and the final `recursive-exclude . __pycache__ ...`):
  ```
  prune vendor/gtx_cpp_reference
  ```
  All 15 original lines preserved unchanged. Position chosen so the `prune` directive comes AFTER `recursive-include vendor *` (line 13) — required for correct override semantics per setuptools MANIFEST.in docs.

### Created (auto-clone, not authored)

- `vendor/gtx_cpp_reference/` — submodule mounted by `git submodule add`. Contains gtx/, riscv-isa-sim/, autocomp/, gemmini/, scripts/, etc. Tracked by parent repo as a single SHA reference, NOT as individual files.

## Verification Outputs

### git submodule status

```
 80d524293407ceb9654b6e9c3aef0186b4e3af98 vendor/gtx_cpp_reference (v6.0-135-g80d5242)
-591cff16109ced6a21bb2a612a3853b4e9cbd86d vendor/spike
```

The leading `-` on `vendor/spike` indicates that submodule is registered but not initialized in this worktree — expected for a parallel agent worktree (it was not the target of this plan and we did not initialize it). The leading space on `vendor/gtx_cpp_reference` indicates registered AND initialized AND clean.

### .gitmodules URL verification (D-04)

```bash
$ git config -f .gitmodules submodule.vendor/gtx_cpp_reference.url
https://github.com/Sudo42b/gtx_spike
```

Matches `^https://github\.com/Sudo42b/gtx_spike(\.git)?$` regex (no `.git` suffix variant — also accepted per plan).

### MANIFEST.in diff

```diff
@@ -12,4 +12,5 @@
 recursive-include tests/data *.py *.elf
 recursive-include vendor *
 recursive-exclude src/main/python/riscv/data *
+prune vendor/gtx_cpp_reference
 recursive-exclude . __pycache__ *.pyc *.pyo .gitignore .DS_Store .coverage .mypy_cache .tox .pytest_cache *.egg-info
```

### Acceptance criteria — all PASS

- [x] `git submodule status | grep -q gtx_cpp_reference` (exit 0)
- [x] URL regex match: `https://github.com/Sudo42b/gtx_spike` (no `.git` suffix; both variants accepted)
- [x] Path config: `vendor/gtx_cpp_reference`
- [x] `vendor/gtx_cpp_reference/.git` exists (submodule mounted)
- [x] `vendor/spike` submodule URL preserved (`../spike`)
- [x] `.gitmodules` contains 2 stanzas (vendor/spike + vendor/gtx_cpp_reference)
- [x] `grep -q '^prune vendor/gtx_cpp_reference$' MANIFEST.in` (exit 0)
- [x] `awk` order check — prune line comes AFTER `recursive-include vendor *`
- [x] `include LICENSE`, `recursive-include vendor *`, `recursive-exclude src/main/python/riscv/data *` all unchanged
- [x] Exactly 1 line added (15 → 16 by `wc -l`; original counted 15 newlines because file had no trailing blank, plan's "17" was off-by-one but the invariant "+1 line" holds)

## Decisions Made

- **Submodule pin = default-branch HEAD at clone time** (no `-b` flag, no explicit `git checkout <sha>`). Per RESEARCH.md "Note on `--branch` flag" and CONTEXT.md D-04, ground-truth should be commit-pinned (the parent repo records the SHA) rather than branch-tracking. The recorded SHA is `80d524293407ceb9654b6e9c3aef0186b4e3af98`. A future `chore(vendor): pin gtx_cpp_reference to <sha>` commit can be made later if a different commit is desired — out of scope for this plan.
- **Used `prune` (canonical) over `recursive-exclude vendor/gtx_cpp_reference *`** — both are functionally equivalent for tree exclusion, but `prune` is the canonical setuptools directive per official MANIFEST.in docs and RESEARCH.md "Pitfall (Phase 1-specific): MANIFEST.in" (line 470). RESEARCH.md actually showed both lines (prune + belt-and-suspenders recursive-exclude); we chose the minimal canonical form per CLAUDE.md "Simplicity First".
- **prune line placed at end of vendor-related directives, before final exclude-everything-else line** — RESEARCH.md showed it placed between `recursive-exclude src/main/python/riscv/data *` and the final `recursive-exclude . __pycache__ ...` line. Followed exactly.

## Deviations from Plan

None — plan executed exactly as written.

The only minor discrepancy: the plan's line-count acceptance ("`wc -l MANIFEST.in` 출력 == 17") was off by one (original file was 15 lines per `wc -l` due to no trailing blank line; result is 16, not 17). The invariant that matters — "exactly one line added, original lines preserved" — is met. No code change required; this is a documentation-only off-by-one in the plan.

## Issues Encountered

- **Optional sdist verification skipped** — the plan's optional verification step `python -m build --sdist` could not run because `python` is not installed at the system level (only `python3` via `/usr/local/cuda/bin/python3`) and `python3 -m build` requires the `build` package, which is not installed. This is a manual verification step (marked `(manual / 권장)` in the plan) and not part of the automated acceptance criteria; will be re-verified in Plan 04 Wave 2 packaging integration tests, where `pip install build` is part of the wheel build harness.
- **`vendor/spike` shows `-` (uninitialized) in `git submodule status`** — this is expected for a parallel-execution worktree that did not need to initialize the existing submodule. The plan's acceptance is `git config -f .gitmodules submodule.vendor/spike.url` returning `../spike` (passes), not requiring `vendor/spike` to be checked out. No action required.

## User Setup Required

None — `vendor/gtx_cpp_reference` is a public repo and works with anonymous clone (no SSH key, PAT, or other credentials needed). CI / cibuildwheel containers will clone it via `git submodule update --init --recursive` once Plan 04 patches `[tool.cibuildwheel.linux].before-all`.

## Next Phase / Plan Readiness

- **Plan 04 (packaging) integration:** This plan provides the submodule registration that Plan 04's `before-all = "yum install -y dtc && git submodule update --init --recursive"` will initialize inside the manylinux container. Plan 04 owns `pyproject.toml`; this plan owns `.gitmodules` + `MANIFEST.in`. No file conflict.
- **Phase 4 / Phase 5 strict-mode fallback:** If `verify.py --strict` exposes FP16 cast divergence between NumPy 2.x `np.float16` view (D-09) and C++ `gtx_fp32_to_16`, the porting source is now available at `vendor/gtx_cpp_reference/gtx/gtx_npu.h` lines 85-151 (per CONTEXT.md `<canonical_refs>`).
- **Phase 6 wheel size budget:** D-06 enforced — vendor/gtx_cpp_reference excluded from sdist (this plan) and from wheel (Plan 04's `[tool.setuptools.package-data]` not declaring vendor). Confirms wheel size ≤ 50 MB target is achievable.
- **No blockers introduced.** vendor/gtx_cpp_reference is fetched once and reused.

## Self-Check: PASSED

- `vendor/gtx_cpp_reference/.git` FOUND
- `.gitmodules` contains `vendor/gtx_cpp_reference` stanza FOUND
- `MANIFEST.in` contains `prune vendor/gtx_cpp_reference` FOUND
- Commit `b0eab1b` (Task 05-01) FOUND in git log
- Commit `80830e3` (Task 05-02) FOUND in git log
- D-04 URL match: `https://github.com/Sudo42b/gtx_spike` PASS
- D-06 prune position after recursive-include vendor PASS

---

*Phase: 01-foundation*
*Plan: 05-submodule*
*Completed: 2026-05-04*
