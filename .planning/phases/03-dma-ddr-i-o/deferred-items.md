# Phase 03 — Deferred Items

Items observed but out-of-scope for the plan that surfaced them.

## Plan 04 (dispatch-4mode) observations

### Pre-existing failures in tests/gtx/test_firmware_dma.py (Plan 02 territory)

After Plan 02 (`38aac36 feat(03-02): 2-level custom0 dispatch + deferred_ddr_stores queue`)
and Plan 04 (`4831bc6 feat(03-04): implement dispatch_4mode`) landed in Wave 2, the
following tests in `tests/gtx/test_firmware_dma.py` fail with `KeyError` on the
`captured` dict (e.g. `KeyError: 'addr_lo'`, `'length'`, `'mem'`, `'addr_a'`):

- `test_firmware_dma_load_sloop_calls_sloop_load`
- `test_firmware_dma_store_sloop_pushes_deferred`
- `test_firmware_dma_copy_tloop_uses_high_32_bit_dst`
- `test_firmware_dma_xs1_zero_uses_proc_xpr`
- `test_firmware_dma_length_zero_means_65536_e2e`
- `test_firmware_dma_funct7_0x41_load_svr_dispatch`
- `test_tpose_reads_lspr_spm_addrr_at_0x903`
- `test_fill_reads_lspr_spm_addrr_at_0x903`

These tests live in Plan 02's write-set (test_firmware_dma.py), not Plan 04's.
Plan 04 did not touch ops/dma.py, ops/__init__.py, or test_firmware_dma.py.

**Likely cause:** Plan 02's mocks for the inner exec_* helpers don't capture
the same kwarg keys the test asserts; the firmware_dma decode path may not be
threading the decoded dict through to the helper invocation. This is internal
to Plan 02's wave 2 implementation.

**Action for Plan 02 team:** investigate. Plan 04's full Plan-04 + adjacent
suite (test_dispatch_4mode + test_dispatch + test_dma_engine + test_ddr_modes
= 72 tests) all pass — Plan 04 is unaffected.

**RESOLVED (post-doc):** Plan 02 added two further commits (`13a7b78` /
`3292a7f` / `45090f2` / `7d5ac22`) after Plan 04's GREEN, completing its
own ops/dma.py active handlers + disasm stubs. Final full-suite run is
165/165 green. This deferred item is closed; kept here for paper-trail only.
