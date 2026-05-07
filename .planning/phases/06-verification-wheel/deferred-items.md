# Phase 06 Deferred Items (Plan 01)

## Out-of-scope discoveries during Plan 01 execution

### tests/gtx/test_regression_fw_act.py — pre-existing fail
- **Discovered:** during Plan 01 broader regression sweep
- **Status:** WIP modification by Plan 02 parallel agent (Wave 1a sibling)
- **Detail:** Plan 02 has already edited test_regression_fw_act.py to convert
  tier #5 graceful-skip into a hard PASS gate (per P6 D-04 transition).
  Without Plan 02's atexit hook landed, the assertion at line 156 trips.
- **Owner:** Plan 02 (atexit hook + npu.py _LAST_NPU registration).
- **Not a Plan 01 regression:** stashing my Plan 01 changes leaves this test
  unaffected — fail is purely from Plan 02's WIP edit.
- **Action:** None for Plan 01. Plan 02 will GREEN this on its own commit.
