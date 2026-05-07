# Phase 06 Deferred Items

## Out-of-scope discoveries during Plan 04 execution (2026-05-07)

### Plan 03 vendor-golden vs zero-init-runtime mismatch (9 ops)

- **Discovered:** during Plan 04 GREEN-fill pytest run (Wave 2)
- **Status:** Surfaced and tier-6 skipped in `tests/gtx/test_regression_fw_full.py::OPERAND_STAGING_REQUIRED`
- **Symptom:** 9 of 12 parametrized invocations FAIL strict-mode compare:
  - `relu`, `sigmoid`, `tanh`, `softmax`, `leaky_relu`, `add_vv`, `mul_vv`, `sum`, `abs`
- **Root cause:** Plan 03 imported these 9 goldens VERBATIM from
  `vendor/gtx_cpp_reference/test/<OP>/n1s16/data/{kernel}_ref.txt`. Vendor
  C++ libgtx_npu.so produced those goldens by running the op against
  vendor-staged non-zero operand inputs (e.g. `arange` patterns). However,
  the matching Plan 03 `.S` kernels for these ops have NO operand
  pre-staging — they only WRSPR the SPRs and dispatch the op. Result: at
  pyspike subprocess time the L1 region read by the op is zero-init, so
  the runtime output is whatever `f(0_vec)` produces (mostly zeros).
- **Why mm_basic + activation_relu_gelu PASS:** Their goldens are
  zero-init oracles (`@370000000\n00000...`) per P4 04-01 Blocker 1
  Option B + P5 05-01 Task 3 explicit zero-init synthesis. They match the
  zero-init runtime by design.
- **Owner:** P3-design defect — VRF-03 plan author chose vendor-input
  goldens (Plan 03 D-1 importer dict) but kept .S kernels zero-init
  (Plan 03 SUMMARY decision #4 Single-row truncation = vendor row 1).
  The two halves of Plan 03 do not align.
- **Resolution path (Plan 04 cannot fix; out of edit-area):**
  - Option 1 (mechanical, preferred): regenerate the 9 goldens as
    zero-init oracles — for each op compute `<op>(zeros) -> FP16 -> BE
    bit-pair hex`. Mirrors mm_basic_n1s16.hex / activation_relu_gelu.hex
    precedent. ~30 lines of Python per op; ~10 minutes total. Expected:
    `relu(0)=0x0000`, `sigmoid(0)=0x3800`, `tanh(0)=0x0000`,
    `abs(0)=0x0000`, etc.
  - Option 2 (architectural): add operand pre-staging to .S kernels via
    `ddr_init_from_file` infra (P3 D-09 territory) so the operand fixture
    matches the vendor input. Larger; touches more files.
- **Plan 04 mitigation:** Tier-6 skip discipline added; `OPERAND_STAGING_REQUIRED`
  set lists the 9 affected stems. `pytest tests/gtx/test_regression_fw_full.py`
  now exits 0 with `3 PASSED, 10 SKIPPED, 0 FAILED`.
- **Action:** Defer to a future remediation plan (P6 follow-up or P7
  stretch). VRF-04 acceptance criterion 1 ("strict-mode regression matrix
  exists and is parametrized over BUNDLED_ELFS") IS satisfied by Plan 04.
  ROADMAP P6 success #2 ("every bundled .elf 100% strict-mode pass") is
  partially satisfied — 3/3 zero-init-aligned PASS — pending the 9-op
  golden regeneration above.

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
