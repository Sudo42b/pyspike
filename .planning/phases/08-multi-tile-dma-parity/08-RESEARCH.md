# Phase 8: Multi-tile DMA Parity - Research

**Researched:** 2026-05-10
**Domain:** Multi-tile DMA orchestration parity between vendor C++ NPU and pyspike Python NPU
**Confidence:** HIGH (every claim cited to file:line in vendor sources or pyspike code on disk)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 Strategy = Hybrid (diff → hypothesis 순결).** Wave 0 vendor C++ ↔ pyspike Python 1:1 diff → 구조적 누락은 verbatim port, 남은 cross-tile 상태 버그는 hypothesis test로 분리.
- **D-02 Diff scope = DMA + dispatch + loop + compute engines.** Vendor → pyspike 매핑:
  - `gtx_npu_dma.cc` ↔ `dma_engine.py` + `ops/dma.py`
  - `gtx_npu_dispatch.cc` ↔ `dispatch.py` + `dispatch_4mode.py`
  - `gtx_npu_loop.cc` ↔ `warp_state.py` + `ops/control.py`
  - `gtx_npu_mm.cc` ↔ `mm_engine.py` + `gemm_core.py` + `ops/mm.py`
  - `gtx_npu_vec.cc` ↔ `vec_engine.py` + `vec_core.py` + `ops/vec.py`
  - `gtx_npu_act.cc` ↔ `act_engine.py` + `act_core.py` + `ops/act.py`
  - `gtx_npu_core.cc` ↔ `npu.py` (state lifecycle)
- **D-03 Instrumentation = 테스트단 일회성 스냅샷 only.** No `GTX_DEBUG_TILE_TRACE` env var, no `_debug.py` module. production 코드 변경 0.
- **D-04 Fix scope = Root-cause에 지역화 (수술적).** 발견된 무관 잠재 누락은 `.planning/seeds/p9-*.md`로 기록하고 P8에서 fix하지 않음.
- **D-05 ELF policy = `_find_elf` multi-path search 확장.** `firmware/<stem>.elf` > `elf/<stem>.elf` > `${GTX_VENDOR_TEST_DIR}/<OP>/n1s16/n1s16_<stem>.elf` (default `/mnt/e/14_NIGHTLY/pyspike/test/`).
- **D-06 REF policy = `import_vendor_golden.py --all` → 84 op .hex 생성** (≤32 byte truncated, ~10 KB total commit).
- **D-07 Wheel = `tests/gtx/data/firmware/`는 wheel에서 제외.** P6 PKG-01 결정 reverse. `tests/gtx/data/golden/<op>.hex`는 wheel에 포함.
- **D-08 Doc = `tests/gtx/data/firmware/README.md` 단일 파일에 4 contract 통합** (BE/LE FP16, GTX_DDR_REVERSED 자동 적용, vendor `.elf` import, `_find_elf` 우선순위).
- **D-09 Tile-2 test = Python-programmatic + MockProcessor + ABS compute + HEIGHT≥SHARED_TILE_MAX_ROWS+1.** 형태: `tests/gtx/test_multi_tile_dma.py`. assert 대상: `npu.warp.tmu_id`, `npu.warp.curr_id`, `npu.lspr[][LSPR_SPM_ADDR*]`, `npu.deferred_ddr_stores` 길이.
- **D-10 GTX_DDR_REVERSED 자동화 = `test_regression_fw_full_sweep.py` 인라인 set.** vendor 경로 `.elf`만 `subprocess.run(env=...)`에 inline 주입. autouse fixture / pytest marker ❌ (cross-test contamination 방지).
- **D-11 Smoke set 12개 = vendor `.elf` 경로 활성화로 OPERAND_STAGING skip 자동 해소.** 명시 6개 (ABS, ADD_VV, MUL_VV, RELU, SIGMOID, GELU) + 추가 6개 (TANH, LEAKY_RELU, ADD, MUL, SUM + 1개 plan-stage 결정).
- **D-12 VTW-03 = HAS_NUMBA=False baseline 재기록.** MTDMA-01 fix 후 `tests/gtx/data/baseline_walltime.txt` 재기록 → `test_vendor_sweep_walltime_5x` PASS (`mean*5 ≤ baseline`). 30s skip threshold 안 걸리도록 baseline > 30s 확보.
- **D-13 vendor 경로 의존 = `GTX_VENDOR_TEST_DIR` env var 기본값 + override.** `_find_elf` 안에서 `os.environ.get('GTX_VENDOR_TEST_DIR', '/mnt/e/14_NIGHTLY/pyspike/test/')`. v1.2에서 submodule 정식화 검토 (deferred).

### Claude's Discretion

- 1:1 diff의 구체 형식 (markdown table vs 인라인 주석 vs 별도 부록) — plan-stage 결정.
- tile-2 test의 정확한 HEIGHT 값과 numba 적용 여부 (D-09 옵션 a/b/c 중 택).
- `import_vendor_golden.py --all` 확장 시 기존 9-op 매핑 보존 vs 단순화.
- VTW-04 README.md의 정확한 섹션 구성 (4 contract 순서/깊이).
- 12 op smoke set의 12번째 op 선정 (vendor 디렉토리 카운트 후 plan-stage).

### Deferred Ideas (OUT OF SCOPE)

- multi-hart 지원 (single-hart 전제, v2 이상).
- vendor 펌웨어 build chain 통합 (gtx-firmware/ cmake → pyspike CI), v2.
- submodule로 vendor 자산 정식화 (D-13 dev path 우선, v1.2 결정).
- `mxe_accum` 4D 확장 (현재 (NEST, SPU) 2D — vendor도 동일, v2).
- cibuildwheel matrix에 vendor 자산 통합 (v1.2).
- `GTX_DEBUG_TILE_TRACE` 환경변수, `_debug.py` 모듈 (D-03 reject).
- 광역 verbatim re-port 11 .cc 전체 (D-01 reject).
- conftest autouse fixture로 GTX_DDR_REVERSED 자동화 (D-10 reject).
- @pytest.mark.vendor_be_fp16 marker (D-10 reject).
- wheel에 vendor `.elf` 직접 포함 (D-07 reject).

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **MTDMA-01** | vendor `gtx_npu_dma.cc` tile loop의 DDR↔L2 + L2↔L1 multi-tile orchestration을 pyspike에 1:1 포팅 | Section "Vendor C++ ↔ pyspike Python Diff Matrix" — DMA & npu.py rows show **NO STRUCTURAL DIFF**; root cause is in REF FORMAT MISCOMPARE (Hypothesis #5 — see updated table). |
| **MTDMA-02** | `GTX_DDR_REVERSED=1` 시맨틱이 vendor BE FP16 ↔ pyspike LE FP16 변환을 정확히 처리함을 회귀 게이트에 자동 적용 + 문서화 | Section "Hypothesis Validation" — `ddr.py:110, :145` reads env-var per call (already correct); inline subprocess env (D-10) wires it. |
| **MTDMA-03** | `tests/gtx/test_multi_tile_dma.py` — vendor `.elf` 의존 없이 tile-1↔tile-2 경계 회귀 방지 unit test | Section "Validation Architecture" — exact fixture shape, RED/GREEN assertions specified. |
| **MTDMA-04** | `__split` / `__start_plan` / `__start_thread` / `__credit_chk` 상태 머신의 tile-경계 reset 검증 | Section "State-Machine Reset Audit" — pyspike state-machine MATCHES vendor behavior at every reset point; verify-only requirement. |
| **VTW-01** | `pyspike/test/<OP>/n1s16/n1s16_<op>.elf` (79개) + `_ref.txt` (79개 — see Vendor Asset Inventory) untracked 자산을 정식 fixture로 wire-up | Section "`_find_elf` Extension Plan" — D-05 multi-path patch shape locked. |
| **VTW-02** | P7 HUMAN-UAT #1 종결 — `M ≥ 12` PASS | Section "Hypothesis Validation" Outcome — 12 op smoke set (D-11) achievable once REF FORMAT issue resolved. |
| **VTW-03** | `tests/gtx/data/baseline_walltime.txt` `HAS_NUMBA=False` 재기록 → `test_vendor_sweep_walltime_5x` PASS | D-12 procedural — section "Plan-Stage Hand-Off". |
| **VTW-04** | vendor `.elf` git 자산화 결정 — commit / symlink / 별도 데이터 레포 + `MANIFEST.in` / wheel size 영향 평가 | D-07 + D-08 합쳐 firmware 디렉토리 wheel 제외 + README single-file 4 contract. |

</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Tech stack:** Pure Python ≥3.10 + NumPy ≥2.0 (manylinux2014_x86_64 wheel). **No new C++ code.** No new runtime deps.
- **Bit-exact:** ULP/atol verified by `verify.py --fp16 --ulp 1 --atol 0.001`. Strict mode requires `exact_matches == total_fp16`.
- **Compatibility:** `riscv.isa.ROCC` virtual signatures unchanged.
- **Testing:** pytest. New ops get unit tests. Direct `npu.deferred_ddr_stores` introspection allowed for verification.
- **Performance:** NumPy backend; ndarray slicing canonical. P8 `tile_max_rows=4095 × 96 tiles ≈ 3M elements` per ABS sweep — must complete in CI window.
- **GSD workflow:** P8 plan-phase will follow `/gsd:plan-phase 8` after this RESEARCH.md lands.

## Executive Summary

Three findings drive the entire P8 plan:

1. **The pyspike Python NPU's DMA + state-machine layers are structurally identical to vendor C++.** Every cross-tile state path (deferred-store queue producer/consumer, mxe_accum FP32, warp.tmu_id/curr_id/wsplit_seen, lspr SPM_ADDR*) has a 1:1 source mapping with no behavioral diff. Hypotheses #1 (DDR↔L2 pointer non-advance), #2 (L1 stale bank), #3 (credit gate stuck), and #4 (plan/thread reset) are all **REJECTED** by reading the code. Vendor and pyspike both:
   - Recompute `ddr_off = addr_hi + row * stride` and `l2_off = (addr_lo + row * stride) % L2_SIZE` per row inside `firmware_dma_sloop_load/store` (not per tile — the firmware itself reissues `__load(...)` per tile via `GTX_TILE_LOOP`).
   - Push one `DeferredDdrStore` per S-loop STORE call (firmware issues N store calls = N tiles, queue grows linearly).
   - Flush the entire queue at `end_p` when `!wsplit_seen` OR at `credit_st_chk` when `is_sloop` (matched 1:1).
   - Have empty/idle credit infrastructure in functional model (`is_load = (sub_op == 0) || (opcode == GTX_OP_DMA)` in both).

2. **The actual root cause is the GTX_DDR_DUMP region size, not multi-tile orchestration.** P7 HUMAN-UAT and the ABS smoke baseline both observed "first 2047 lines (~64 KB) byte-exact, lines 2048+ diverge." But:
   - The vendor `_ref.txt` for ABS at `test/ABS/n1s16/data/n1s16_abs_ref.txt` is **13 MB / 196610 lines** — covering the full 393217-row × 16-byte/row output (Source: file size + `n1s16_abs.c` `total_rows = 393217u`, `tile_max_rows = 4095u`, ROW_BYTES=16 ⇒ 96 tiles).
   - `test_regression_fw_full_sweep.py:179-180` sets `GTX_DDR_DUMP_SIZE=0x20` (32 bytes / 16 FP16 = single row only).
   - The atexit hook in `ddr.py:_atexit_ddr_dump` will produce 32 bytes — covering only tile 0 — and `compare_hex(strict=True)` against the truncated golden (the `import_vendor_golden.py` script also truncates `_ref.txt` to single-row `.hex`, line 25-39 `n_data_lines=1`).
   - Therefore the "tile 1 byte-exact, tile 2+ diverge" observation came from a **different** test path (manual smoke run on the full 13 MB `_ref.txt`, not the truncated one). The actual full-region divergence is REAL but not yet measured by the harness; D-09 tile-2 unit test is required to lock it down.

3. **The "84 vs 79" question is a path-confusion artifact.** ROADMAP's `M+N=84` invariant references **`vendor/gtx_cpp_reference/test/`** (84 ALL_CAPS dirs, used by `_discover_vendor_ops`). The HUMAN-UAT's "79 .elf" references **`/mnt/e/14_NIGHTLY/pyspike/test/`** (79 ALL_CAPS dirs, the dev's pre-built host directory — D-13 default `GTX_VENDOR_TEST_DIR`). The two trees overlap on 68 ops with mismatch:
   - Vendor-only (16 ops, no `.elf` available): SILU, SIN, SOFTPLUS, SOFT_MAX, SOLVE_TRI, SQR, STEP, SUB, SUM, SWIGLU_OAI, TANH, TIMESTEP_EMBEDDING, TRI, TRUNC, WIN_PART, WIN_UNPART, XIELU.
   - Host-only (11 ops, no vendor golden mapping in `_ref.txt`): ARGMAX, ARGSORT, CEIL, CONT, CONV_2D_DW, CONV_3D, COUNT_EQUAL, FLASH_ATTN_EXT, GLU, PERMUTE, RESHAPE.
   - Overlap (68 ops with `.elf` AND `_ref.txt`).

   `M+N=84` invariant remains **valid** because parametrize iterates over the 84-element `vendor/gtx_cpp_reference/test/` directory tree (which is the source of golden hex). Skip occurs for vendor-only ops (no `.elf` from D-13 path → Tier 3 skip with documented reason) until vendor-host overlap is fully populated. After D-05 `_find_elf` patch, the 68-overlap becomes available, M climbs from 0 to up to 68 (subject to D-11 skip set + correctness fixes).

**Primary recommendation:** Land 4 surgical changes in this order: (a) D-05 `_find_elf` 3rd-tier vendor candidate; (b) D-06 `import_vendor_golden.py --all` (must NOT truncate to 1 row — extend to full output region or add a `--full` flag); (c) D-09 `test_multi_tile_dma.py` (RED-then-GREEN validates the actual bug exists, since current dump-size truncation hides it); (d) re-record baseline (D-12). The current code path has NO multi-tile orchestration bug **as observed by the running test harness** — but the harness measures only 32 bytes, masking the real divergence on tiles 1..95 that the dev observed manually. P8 must surface it via D-09 first.

## Vendor C++ ↔ pyspike Python Diff Matrix

> Goal: every D-02 vendor file mapped to pyspike target, with cross-tile state behavior verdict. **MATCH** = 1:1 verbatim port. **DIFF** = behavioral difference (must specify). **MISSING** = vendor has it, pyspike lacks it. **EXTRA** = pyspike has it, vendor lacks it.

### `gtx_npu_dma.cc` ↔ `dma_engine.py` + `ops/dma.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| `firmware_dma()` LOAD S-loop entry | `gtx_npu_dma.cc:294-318` | `dma_engine.py:293-313` (`firmware_dma_sloop_load`) | **MATCH** — same per-row recompute, same `% GTX_L2_SIZE`, same `ensure_ddr` + `ddr.size - ddr_off` clamp |
| `firmware_dma()` STORE S-loop branch | `gtx_npu_dma.cc:319-326` | `dma_engine.py:269-287` (`firmware_dma_sloop_store`) | **MATCH** — both push exactly one `DeferredDdrStore` per call with `(addr_hi, addr_lo, length, height, rd_stride, wr_stride)` and `break` after first row |
| `firmware_dma()` T-loop LOAD/STORE | `gtx_npu_dma.cc:349-391` | `dma_engine.py:319-348` (`firmware_dma_tloop_load_store`) | **MATCH** (with one **EXTRA** absent: vendor has post-loop `l1_device->sync_region_from_spu(l1_min, l1_max)` shadow sync at lines 388-390; pyspike has NO L1 shadow because there is no separate CPU-visible L1 device — pyspike L1 is just `np.uint8` slice, single source of truth; this is **correct by design** per CONTEXT P3 D-12 "no MMU mock") |
| `firmware_dma()` T-loop COPY | `gtx_npu_dma.cc:334-348` | `dma_engine.py:354-372` (`firmware_dma_tloop_copy`) | **MATCH** — explicit `.copy()` on slice (Python equivalent of `std::memmove` per Plan 03 dispatch.cc) |
| Packed-rs1/rs2 decode | `gtx_npu_dma.cc:262-288` | `dma_engine.py:66-99` (`decode_firmware_dma_args`) | **MATCH** — same `funct3 = (xd<<2)\|(xs1<<1)\|xs2`, same `is_copy → addr_hi = rs1>>32`, same HW-conv 0→65536 + 0→1 |
| `flush_deferred_ddr_stores()` | `gtx_npu_dma.cc:415-435` | `npu.py:166-190` | **MATCH** — same per-row loop, same clamps, same `clear()` at end |
| Deferred store queue datatype | `gtx_npu.h:1257-1266` (struct, 7 fields) | `dma_engine.py:46-60` (frozen dataclass, 7 fields, exact field order) | **MATCH** |
| Cross-tile state in DMA path | None (each `firmware_dma` call is stateless given queue + npu state) | None | **MATCH** |

### `gtx_npu_dispatch.cc` ↔ `dispatch.py` + `dispatch_4mode.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| 4-mode router | `gtx_npu_dispatch.cc:79-139` | `dispatch_4mode.py:69-121` | **MATCH** — Mode 1/2/3/4 selection, `is_load = (sub_op==0) \|\| (opcode==GTX_OP_DMA)` lockin Pitfall 8 |
| Credit queue check | `gtx_npu_dispatch.cc:42-61` (uses `use_spu_queue`/`use_tmu_queue` arrays, both empty in functional model — never queues) | NOT PORTED (pyspike doesn't have credit queue infrastructure at all) | **EXTRA-VENDOR but DEAD CODE** — vendor "always pass" by design (line 42 comment); pyspike is correct to omit. **Hypothesis 3 (credit gate stuck) is therefore IMPOSSIBLE.** |
| MEXEC interception | `gtx_npu_dispatch.cc:63-69` | dispatch_4mode does not handle MEXEC; firmware-DMA path goes through `@handler` directly | **MATCH** — vendor only triggers exec_mexec for `GTX_ISS_F7_MEXEC` opcode; pyspike's `firmware_dma` (funct7=0x40) never reaches dispatch_4mode (it's a custom0 @handler). 4-mode router is only for gem5-simplified `dispatch_dma` (funct7=0x07) |
| `dispatch_iss_opcode()` switch (1100+ lines) | `gtx_npu_dispatch.cc:151-1100+` | `dispatch_4mode.py:38-66` (P3-only stubs for `CREDIT_ST_CHK` + `LD/ST_SVR_L1`) | **PARTIAL MATCH** — only the funct7s reachable via dispatch_4mode are wired (P3 D-03); MM/VEC/ACT bypass via direct @handler. **No diff in cross-tile concerns.** |
| `flush_deferred_ddr_stores()` at `credit_st_chk` | `gtx_npu_dispatch.cc:898-905` (dispatch_iss_opcode case) + `gtx_npu_custom0.cc:684-694` (direct custom0 entry) | `ops/dma.py:312-324` (`@handler` for funct7=0x53) + `dispatch_4mode.py:57-61` (mirror inside iss_opcode) | **MATCH** — both flush sites wired (3 vendor call sites collapsed to 2 pyspike sites since pyspike has no separate `dispatch_iss_opcode` callable, but the single `@handler` covers it) |

### `gtx_npu_loop.cc` ↔ `warp_state.py` + `ops/control.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| `startp(rs1, rs2)` | `gtx_npu_loop.cc:21-35` | `ops/control.py:57-63` (`_do_startp`) | **MATCH** — same `extract_id` + clamp + `is_ploop = true` |
| `endp(rs1, rs2)` flush | `gtx_npu_loop.cc:37-69` (flush + DDR-dump-during-WJOIN-firmware path) | `ops/control.py:66-77` (`_do_endp`) | **MATCH for flush** (`if !wsplit_seen: flush_deferred_ddr_stores()`); **DIFF for DDR-dump** — vendor dumps DDR inside endp when no WSPLIT (lines 56-62, alternate-path firmware), pyspike defers all DDR-dump to atexit hook (`ddr.py:_atexit_ddr_dump` per P6 D-04/D-05). This is **intentional and CORRECT** for pyspike's `subprocess.run` model: atexit fires once at interpreter shutdown, equivalent to vendor's `std::atexit(gtx_atexit_ddr_dump)` registered at construction (`gtx_npu_core.cc:127`). Vendor dumps in `endp` are a redundant 2nd path for HTIF early-exit — Spike calls `_Exit()` from firmware, which triggers atexit. |
| `startt`/`endt`/`starts`/`ends` | `gtx_npu_loop.cc:74-142` | `ops/control.py:80-109` | **MATCH** — every state transition mirrored, including `is_sloop`/`is_tloop`/`curr_id` write |
| `wsplit_seen` lifetime | `gtx_npu.h:1251` (bool initializer; **never cleared in reset()**) | `warp_state.py:33` (bool field, comment "process-lifetime sentinel — set True by WSPLIT, NOT cleared by reset()") | **MATCH** — pyspike explicitly preserves vendor semantics (comment cites RESEARCH Pitfall 7) |
| `WarpState.reset()` | Vendor `gtx_npu_t::reset()` does NOT touch `is_ploop/is_tloop/is_sloop/tmu_id/curr_id` (these reset implicitly because no firmware leaves them set after WJOIN) | `warp_state.py:35-41` resets `is_ploop=is_tloop=is_sloop=False; tmu_id=curr_id=0` (explicit per Phase 2) | **EXTRA-PYSPIKE but BENIGN** — pyspike resets these explicitly for test cleanliness; vendor relies on firmware to leave them False at WJOIN. Both produce same firmware behavior. |

### `gtx_npu_core.cc` ↔ `npu.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| Constructor: SPR + L1 shadow + DDR init + atexit registration | `gtx_npu_core.cc:78-134` | `npu.py:45-83` | **MATCH** — same SPR defaults, same `_init_ddr_from_env` symmetric pair, same `_LAST_NPU` global (= `g_gtx_instance`) |
| `reset()` lifecycle | `gtx_npu_core.cc:144-189` (sp init + FPU enable + DDR bus register + L1 shadow registration) | `npu.py:101-143` (sp init + FPU + mxe zero + L0/L1/L2 zero + SPR zero + deferred queue clear + warp.reset) | **MATCH for state reset**; **DIFF for bus registration** — pyspike does NOT call `sim->add_device(GTX_DDR_BASE, ddr_mem)` because there's no Spike `sim_t` to register against from Python (the DMA path uses `mem._ddr_bytes` directly). **Correct by design** — pyspike's CPU firmware never accesses DDR through MMU; it goes through `proc.state.XPR[]` then through the `@handler`. |
| `mxe_accum` lifecycle | `gtx_npu.h:1254` (`float[4][16]`); zero-init via `gtx_npu_t::reset()` doesn't clear it (it's per-construction zero — vendor relies on first `MM_O`/`MM_V` overwriting before read) | `npu.py:64-66` (`np.zeros((4,16), dtype=np.float32)`); `npu.py:114` `reset()` calls `self._mxe_accum.fill(0.0)` | **EXTRA-PYSPIKE but BENIGN** — pyspike clears mxe_accum on reset, vendor doesn't. For ABS / vec-only firmware (D-09 ABS path), `mxe_accum` is **never read or written**; both produce zero result. Hypothesis 4 reset path is **immaterial** to ABS smoke set. For MM regression `mm_basic.elf` (Phase 4 already PASSed strict-mode), the chain `mm_s → mmc_s → mmc` works in both because firmware writes before reading on every chain. |
| `flush_deferred_ddr_stores()` API | `gtx_npu.h:1173` + `gtx_npu_dma.cc:415-435` | `npu.py:166-190` | **MATCH** — exact 1:1 port |
| `deferred_ddr_stores` storage | `gtx_npu.h:1266` (`std::vector<deferred_ddr_store_t>`) | `npu.py:52` (`list`) | **MATCH** — both grow append-only, both cleared on reset |

### `gtx_npu_mm.cc` ↔ `mm_engine.py` + `gemm_core.py` + `ops/mm.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| `gemm_core` (SystemC parity) | `gtx_npu_mm.cc:27-94` | `gemm_core.py` (Phase 4 land) — `gemm_core`, `gemm_reduce_sum_a`, `gemm_dot` | **MATCH** (Phase 4 strict-mode `mm_basic.elf` PASSes — already verified) |
| `mxe_accum` per-(NEST, SPU) FP32 | `gtx_npu.h:1254` (`mxe_accum[4][16]`) | `npu.py:64-66` (`np.zeros((4,16), dtype=np.float32)`) | **MATCH** in shape and dtype |
| `mxe_accum` r/w sites | `gtx_npu_mm.cc:210-212` (MM_O), `:267-269` (MM_V) — write-then-read pattern with `has_bias` flag | `mm_engine.py` Plan 04 — same pattern | **MATCH** — verified by Phase 4 GREEN tests |
| Cross-tile mxe_accum lifecycle | Vendor: NEVER cleared except per-construction; firmware MUST init via `MM_O` (not `MMC_O`) on first iteration | pyspike: `reset()` clears explicitly — **EXTRA-PYSPIKE clear**, but firmware never relies on stale state, so both behave identically. **For ABS (vec-only), mxe_accum is untouched throughout sweep.** | **DIFF (benign)** |

### `gtx_npu_vec.cc` ↔ `vec_engine.py` + `vec_core.py` + `ops/vec.py`

| Concern | Vendor C++ | pyspike Python | Verdict |
|---------|-----------|-----------------|---------|
| `firmware_vec_op` SIGN→ABS path | `gtx_npu_vec.cc:572-754`, `:283-291` (SIGN op funct7=0x1D, sub_op=0 = ABS via `np.abs`) | `vec_engine.py:91-191` `firmware_vec_op` + `:270-302` `_apply_unary` | **MATCH** — both: read FP16 view from `LSPR_SPM_ADDRA`, apply `np.abs(f32).astype(np.float16)`, write to `LSPR_SPM_ADDRR`. **Verified against P7 ABS smoke (4.8 s; tile 0 byte-exact)**. |
| LSPR `SPM_ADDRA`/`SPM_ADDRR` cross-tile | Read fresh from `spu.lspr[LSPR_SPM_ADDRA]` per call (firmware writes via `wr_spr` before each tile via WRSPR) | Same — `npu.lspr[nest][spu].get(LSPR_SPM_ADDRA, 0)` per call | **MATCH** |
| Cross-tile state in vec path | None | None | **MATCH** |

## Hypothesis Validation Table

> Each hypothesis from `.planning/seeds/p8-multi-tile-dma.md` evaluated against the read code.

| # | Hypothesis | Evidence FOR | Evidence AGAINST | Likelihood | Validation steps for plan-stage |
|---|-----------|--------------|------------------|-----------|-------------------------------|
| 1 | DDR↔L2 src/dst pointer not advancing between tiles | None — symptom (lines 2048+ diverge) is consistent with this if it were true | `dma_engine.py:304-307` recomputes `ddr_off = ddr_off_base + row * rd_stride` per row INSIDE the call; `ddr_off_base` is freshly computed from `addr_hi` (= `rs1>>27` at firmware ENTRY of every tile call). **Each tile reissues `__load(...)` with new `addr_hi` from the firmware-side `tile_row_start`-shifted base address**. Vendor `gtx_npu_dma.cc:300-303` does identical recompute. There is no NPU-internal "current pointer" state to drift. | **LOW** | Plan-stage: assert `npu.deferred_ddr_stores[i].ddr_off` advances by `tile_max_rows * stride` for each tile's STORE entry. If this DOES advance, hypothesis 1 falsified. |
| 2 | L1 bank not being recycled (stale compute-side state) | None | T-loop COPY/LOAD path (`firmware_dma_tloop_load_store`) writes L1 directly via `np.uint8` slice assignment (`l1[lo_off:lo_off+copy_len] = l2[hi_off:...]`) — no read-side cache. ABS firmware does T-loop LOAD (DDR→L2→L1), runs `firmware_vec_op SIGN-ABS` (overwrites L1), then T-loop STORE (L1→L2). Each tile reissues all three steps. There is no "compute-side state" surviving across the LOAD because L1 is just bytes. **Hypothesis assumes a shadow buffer; pyspike has no shadow buffer (CONTEXT P3 D-12 `no MMU mock`).** | **LOW** | Plan-stage: D-09 assert `npu.mem.l1_byte(0,0)[addr_a:addr_a+tile_max_bytes]` post-LOAD-tile2 differs from post-LOAD-tile1 (assuming distinct DDR contents per tile). If it does, hypothesis 2 falsified. |
| 3 | Credit gate stuck | None | pyspike has NO credit-queue infrastructure at all (the EXTRA-VENDOR row in dispatch.cc table — vendor's queue infrastructure is dead code in functional model since `use_spu_queue`/`use_tmu_queue` are zero-init false). pyspike `credit_st_chk @handler` (`ops/dma.py:312-324`) only flushes the deferred queue when `is_sloop` is True — it never blocks compute. | **NONE** | Mark hypothesis 3 as IMPOSSIBLE in plan SUMMARY. No validation needed. |
| 4 | Plan/thread state machine reset (NEST/SPU dispatch context not refreshed at tile 2) | None | pyspike `_do_startp` always overwrites `tmu_id`, `_do_startt` always overwrites `curr_id`. Each tile's `__start_plan(NEST_ID)` + `__start_thread(SPU_ID)` issues these writes. Vendor `gtx_npu_loop.cc:21-142` does same. **The state machine is overwritten on every tile entry — there is no "stale" path.** | **LOW** | Plan-stage: D-09 assert `npu.warp.tmu_id == expected_nest` and `npu.warp.curr_id == expected_spu` immediately after each tile's start_p/start_t calls. If true → hypothesis 4 falsified. |
| **5 (NEW)** | **GTX_DDR_DUMP_SIZE in test harness is too small to detect divergence past tile 0** | `test_regression_fw_full_sweep.py:179-180` sets `GTX_DDR_DUMP_SIZE=0x20` (32 bytes). Vendor ABS `_ref.txt` is 13 MB / 196610 lines (full output). `import_vendor_golden.py:31-41` uses `n_data_lines=1` (truncates `.hex` to single 32-byte row). **The harness compares 32 bytes vs 32 bytes — a single-tile dump.** The "tile 1 byte-exact, tile 2 diverges" observation came from a manual full-region run, not the harness. | The dev confirmed manual full-region divergence — so a real bug DOES exist somewhere; just not surfaceable via current harness. | **HIGH** (most likely) | Plan-stage: extend GTX_DDR_DUMP_SIZE to full output region (e.g., `0x600000` for ABS = 6 MB) OR write D-09 unit test that directly inspects `npu.mem._ddr_bytes` slice for tile 0 vs tile 1. **D-09 unit test's RED state is the actual bug surfacing.** |
| **6 (NEW)** | **`active_tid_mask = 0xFFFF` (16 SPUs in parallel) — pyspike's per-tile T-loop iterates SPUs sequentially, but firmware code expects parallel SPU dispatch** | `kernel_common.h:55,75` `active_tid_mask` field; firmware writes it via `wr_spr NSPR_THREAD_MASK = 0xFFFF`; vendor `dispatch_4mode` Mode 4 routes to single `(tmu_id, curr_id)` per call — firmware issues one `__start_thread + ops + __end_thread` per SPU (16 calls per tile). pyspike does same. **Both vendor and pyspike serialize 16 SPUs.** | NSPR_THREAD_MASK is read-only from NPU's perspective in functional model — both pyspike and vendor honor firmware-issued sequential dispatch. **No diff.** | **NONE** | Mark as IMPOSSIBLE in plan SUMMARY. |
| **7 (NEW)** | **`addr_hi` 37-bit truncation drops bits when `total_rows × ROW_BYTES > 2^37` overflows** | `dma_engine.py:82` masks `addr_hi = (rs1 >> 27) & 0x1FFFFFFFFF` (37 bits = 128 GiB). ABS full region `0x1000000` (16 MiB input) and `0xf000000` (240 MiB result) — both fit in 37 bits trivially. | Math check: 0xf000000 + 393217 × 16 = 0xf000000 + 0x600400 = 0xf600400 ≪ 2^37. No overflow. | **NONE** | Mark as IMPOSSIBLE. |

**Conclusion: Hypothesis 5 (test harness truncation hides the divergence) is highest probability. Hypotheses 1, 2, 4 are LOW (verifiable false). Hypotheses 3, 6, 7 are IMPOSSIBLE.** Plan-stage MUST start by extending the dump-size and adding D-09 unit test FIRST — because once the bug is surfaced, the actual root cause may lie outside the diff scope (e.g., in `ddr_dump_to_file` itself, in `_atexit_ddr_dump` ordering, or in something firmware-side like `MAX_SHARED_DMA_BYTES = 65535` boundary handling that the firmware itself bug-handles).

## Vendor Asset Inventory

> Resolves the "84 vs 79" question. ROADMAP says 84; HUMAN-UAT says 79. **Both are correct — they reference different paths.**

### Path A — `vendor/gtx_cpp_reference/test/` (the "84" path, used by ROADMAP / `import_vendor_golden.py:43-69`)

86 entries total in directory listing; **84** are op directories after excluding `CLAUDE.md` + `README.md`. Sorted ALL_CAPS:

```
ABS, ACC, ADD, ADD1, ADD_ID, ADD_REL_POS, ARANGE, CLAMP, CONCAT, CONV_2D,
CONV_TRANSPOSE_1D, CONV_TRANSPOSE_2D, COS, CPY, CUMSUM, DIAG, DIAG_MASK_INF,
DIAG_MASK_ZERO, DIV, DUP, ELU, EXP, EXPM1, FILL, FLOOR, GATED_LINEAR_ATTN,
GEGLU, GEGLU_ERF, GEGLU_QUICK, GELU, GELU_ERF, GELU_QUICK, GET_REL_POS,
GET_ROWS, GROUP_NORM, HARDSIGMOID, HARDSWISH, IM2COL, IM2COL_3D, L2_NORM,
LEAKY_RELU, LOG, MEAN, MUL, MUL_MAT, MUL_MAT_ID, NEG, NORM, OUT_PROD, PAD,
PAD_REFLECT_1D, POOL_1D, POOL_2D, REGLU, RELU, REPEAT, RMS_NORM, ROLL,
ROPE, ROUND, RWKV_WKV6, RWKV_WKV7, SCALE, SET, SET_ROWS, SGN, SIGMOID,
SILU, SIN, SOFTPLUS, SOFT_MAX, SOLVE_TRI, SQR, STEP, SUB, SUM, SWIGLU_OAI,
TANH, TIMESTEP_EMBEDDING, TRI, TRUNC, WIN_PART, WIN_UNPART, XIELU
```

This is the **golden-source** tree. Every dir has `n1s16/data/n1s16_<stem>_ref.txt`. NO `.elf` here — these are reference data only.

### Path B — `/mnt/e/14_NIGHTLY/pyspike/test/` (the "79" path, D-13 default `GTX_VENDOR_TEST_DIR`)

79 ALL_CAPS dirs. Sorted:

```
ABS, ACC, ADD, ADD1, ADD_ID, ADD_REL_POS, ARANGE, ARGMAX, ARGSORT, CEIL,
CLAMP, CONCAT, CONT, CONV_2D, CONV_2D_DW, CONV_3D, CONV_TRANSPOSE_1D,
CONV_TRANSPOSE_2D, COS, COUNT_EQUAL, CPY, CUMSUM, DIAG, DIAG_MASK_INF,
DIAG_MASK_ZERO, DIV, DUP, ELU, EXP, EXPM1, FILL, FLASH_ATTN_EXT, FLOOR,
GATED_LINEAR_ATTN, GEGLU, GEGLU_ERF, GEGLU_QUICK, GELU, GELU_ERF,
GELU_QUICK, GET_REL_POS, GET_ROWS, GLU, GROUP_NORM, HARDSIGMOID,
HARDSWISH, IM2COL, IM2COL_3D, L2_NORM, LEAKY_RELU, LOG, MEAN, MUL, MUL_MAT,
MUL_MAT_ID, NEG, NORM, OUT_PROD, PAD, PAD_REFLECT_1D, PERMUTE, POOL_1D,
POOL_2D, REGLU, RELU, REPEAT, RESHAPE, RMS_NORM, ROLL, ROPE, ROUND,
RWKV_WKV6, RWKV_WKV7, SCALE, SET, SET_ROWS, SGN, SIGMOID
```

Each dir has `n1s16/n1s16_<stem>.elf` (79 `.elf` confirmed by `find` count) + `n1s16/data/n1s16_<stem>_ref.txt` (e.g., 13 MB for ABS).

### Diff between Path A and Path B

**Path B has but Path A lacks (11 ops):** ARGMAX, ARGSORT, CEIL, CONT, CONV_2D_DW, CONV_3D, COUNT_EQUAL, FLASH_ATTN_EXT, GLU, PERMUTE, RESHAPE.

**Path A has but Path B lacks (16 ops):** SILU, SIN, SOFTPLUS, SOFT_MAX, SOLVE_TRI, SQR, STEP, SUB, SUM, SWIGLU_OAI, TANH, TIMESTEP_EMBEDDING, TRI, TRUNC, WIN_PART, WIN_UNPART, XIELU.

**Overlap (68 ops):** all the rest. Both `.elf` and `_ref.txt` available.

### Implications for `M+N=84` invariant

- `test_regression_fw_full_sweep.py:60-64` parametrizes over `VENDOR_TEST_DIR = vendor/gtx_cpp_reference/test/` → 84 ops. **Invariant holds.**
- For each op:
  - Tier 3 `_find_elf` checks `firmware/` then `elf/`. With D-05 it adds Path B as 3rd candidate → 68 overlap ops + 12 Path-B-only that overlap with hand-built (currently zero).
  - Tier 4 `_find_golden` finds golden `.hex` if `import_vendor_golden.py --all` was run.
  - 16 Path-A-only ops have golden but no `.elf` → permanent skip with documented reason ("vendor-only op; no host pre-built `.elf`").
- After D-05 + D-06 land, **M = up to 68** (subject to D-11 `OPERAND_STAGING_REQUIRED_VENDOR` and correctness fixes); N = 16 (vendor-only) + Tier-3-skips for hand-built mismatches. M ≥ 12 floor is highly achievable.

### 12-op smoke set (D-11) candidate

D-11 names 11 ops + leaves 1 to plan-stage:

- **Confirmed 11**: ABS, ADD_VV (= ADD), MUL_VV (= MUL), RELU, SIGMOID, GELU, TANH, LEAKY_RELU, ADD, MUL, SUM.
  - Wait — `test_regression_fw_full_sweep.py:69-74` `VENDOR_TO_ELF_STEM` maps `"ADD" → "add_vv"` (single entry; ADD_VV is not a separate op_dir). So D-11's "ADD_VV" and "ADD" are the same op. Same for MUL_VV / MUL. Effective unique smoke set: **9 ops** (ABS, ADD, MUL, RELU, SIGMOID, GELU, TANH, LEAKY_RELU, SUM).
- **Plan-stage chooses 3 more** to reach 12. Recommended candidates (have `.elf` AND `_ref.txt` AND simple compute):
  - **NEG** (vec_engine SIGN sub_op=1, simple unary).
  - **DIV** (vec_engine SASMD VV sub_op=3, simple binary).
  - **EXP** (vec_engine MATH funct7=0x1C sub_op=1, transcendental — exercises P7 objmode path).
  - Alternatives if any fail: SUB, SQR, LOG, FLOOR, CEIL.

## `_find_elf` Extension Plan

> Exact patch shape (file:line, before/after).

**File:** `tests/gtx/test_regression_fw_full_sweep.py`

**Current code (lines 97-107):**

```python
def _find_elf(op_dir: str):
    """Find <op>.elf in firmware dir or legacy elf dir. Returns Path or None."""
    elf_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    candidates = [
        FIRMWARE_DIR / (elf_stem + ".elf"),
        ELF_DIR_LEGACY / (elf_stem + ".elf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
```

**Patched code (D-05):**

```python
def _find_elf(op_dir: str):
    """Find <op>.elf in firmware dir, legacy elf dir, or vendor host tree (D-05).

    Resolution order (firmware/ first → P5/P6 hand-built wins on collision):
      1. tests/gtx/data/firmware/<elf_stem>.elf      (P5/P6 wheel-bundled)
      2. tests/gtx/data/elf/<elf_stem>.elf           (P5/P6 legacy location)
      3. ${GTX_VENDOR_TEST_DIR}/<OP_DIR>/n1s16/n1s16_<elf_stem>.elf
         (D-13 default = /mnt/e/14_NIGHTLY/pyspike/test/, vendor pre-built)

    Returns Path or None.
    """
    elf_stem = VENDOR_TO_ELF_STEM.get(op_dir, op_dir.lower())
    vendor_root = pathlib.Path(
        os.environ.get("GTX_VENDOR_TEST_DIR", "/mnt/e/14_NIGHTLY/pyspike/test/")
    )
    candidates = [
        FIRMWARE_DIR / (elf_stem + ".elf"),
        ELF_DIR_LEGACY / (elf_stem + ".elf"),
        vendor_root / op_dir / "n1s16" / ("n1s16_" + elf_stem + ".elf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
```

**Non-breaking proof:** firmware/ + elf/ are checked FIRST; vendor only used as fallback. P5/P6 hand-built `.elf` (e.g., `tests/gtx/data/elf/abs.elf`) takes precedence over `/mnt/e/14_NIGHTLY/pyspike/test/ABS/n1s16/n1s16_abs.elf`. If both exist (current state), hand-built is used — **identical** to today's behavior for the 12 ops with hand-built `.elf`. New ops (e.g., NEG, EXP) have no hand-built `.elf` → hit candidate 3, get vendor `.elf`.

**Subprocess env var (D-10) inline:** `test_regression_fw_full_sweep.py:176-181` `env = os.environ.copy()` block needs to add:

```python
# D-10: vendor-derived .elf needs GTX_DDR_REVERSED=1 (BE FP16 vs LE pyspike)
if elf_path.is_relative_to(vendor_root):
    env["GTX_DDR_REVERSED"] = "1"
```

(`is_relative_to` is Python 3.9+; cp310+ baseline so fine.)

## State-Machine Reset Audit

> For each loop transition, what does vendor C++ reset and what does pyspike Python reset? **Delta** column flags any state pyspike fails to reset (zero entries means MTDMA-04 is verify-only, not fix-needed).

| Reset point | Vendor C++ resets | pyspike Python resets | Delta (pyspike does NOT reset) |
|-------------|---------------------|--------------------------|-----|
| `start_p(rs1, rs2)` | `tmu_id = id` (clamped); `is_ploop = true` (`gtx_npu_loop.cc:21-35`) | `npu.warp.tmu_id = nest_id`; `npu.warp.is_ploop = True` (`ops/control.py:57-63`) | **none** |
| `end_p(rs1, rs2)` | `is_ploop = false`; flushes deferred queue when `!wsplit_seen`; alternate-path DDR dump (lines 50-68) | `npu.warp.is_ploop = False`; `npu.flush_deferred_ddr_stores()` when `!wsplit_seen` (`ops/control.py:66-77`) | **none for state**; pyspike defers DDR dump to atexit (intentional, see Diff Matrix npu.py row) |
| `start_s(rs1, rs2)` | `is_sloop = true`; `curr_id = gdmac_id` (clamped) (`gtx_npu_loop.cc:74-90`) | `npu.warp.is_sloop = True`; `npu.warp.curr_id = gdmac_id` (`ops/control.py:94-104`) | **none** |
| `end_s(rs1, rs2)` | `is_sloop = false` (rs1/rs2 ignored, `gtx_npu_loop.cc:92-102`) | `npu.warp.is_sloop = False` (`ops/control.py:107-109`) | **none** |
| `start_t(rs1, rs2)` | `is_tloop = true`; `curr_id = id` (clamped); shadow sync deferred to dispatch (`gtx_npu_loop.cc:107-125`) | `npu.warp.is_tloop = True`; `npu.warp.curr_id = spu_id` (`ops/control.py:80-86`) | **none** |
| `end_t(rs1, rs2)` | `is_tloop = false`; explicit anti-sync comment (lines 130-132 — does NOT shadow→L1 sync because shadow may be stale) | `npu.warp.is_tloop = False` (`ops/control.py:89-91`) | **none** |
| `wsplit` | `wsplit_seen = true` (lifetime sentinel, `gtx_npu_custom1.cc:62`, `gtx_npu_custom0.cc:76`) | `npu.warp.wsplit_seen = True` (`ops/control.py:155-165`, `:207-214`) | **none** |
| `wjoin` (custom1 funct3=0b101) | calls `_Exit(0)` if `GTX_NO_EXIT` not set (vendor implementation) | `raise SystemExit(0)` if env unset (`ops/control.py:168-183`) | **none** |
| `reset()` (extension reset, fires once per processor reset) | sp init; FPU enable; bus register DDR; L1 shadow register (`gtx_npu_core.cc:144-189`). Notable: does NOT clear `wsplit_seen`, `mxe_accum`, `deferred_ddr_stores`, `is_ploop`, `tmu_id`, `gspr/nspr/lspr`. | sp init; FPU enable; mxe_accum zero; L0/L1/L2 zero; gspr/nspr/lspr zero w/ defaults; deferred queue clear; warp.reset (clears is_*loop, tmu_id, curr_id; preserves wsplit_seen) (`npu.py:101-143`) | **EXTRA-PYSPIKE clears** mxe_accum + L0/L1/L2 + SPRs + deferred queue + warp loop flags. **Benign** — same per-firmware behavior because vendor relies on firmware to either set or never read these. P8 should NOT change reset() (D-04 surgical scope). |

**Conclusion: MTDMA-04 is verify-only.** D-09 unit test asserts the 8 transition rows above. If asserts pass at tile 1→2 boundary, MTDMA-04 closes.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-cov + pytest-benchmark (P7 introduced for VTW-03) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/gtx/test_multi_tile_dma.py -v --no-cov` |
| Full suite command | `pytest tests/gtx/ -v --no-cov` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MTDMA-01 | Vendor 84-op n1s16 sweep PASSes strict-mode end-to-end past tile 0 | integration | `pytest tests/gtx/test_regression_fw_full_sweep.py -v --no-cov` | ✅ exists; needs D-05 + D-06 + dump-size patch |
| MTDMA-02 | `GTX_DDR_REVERSED=1` auto-applied for vendor `.elf` | integration | (subset of above; D-10 inline subprocess env) | ✅ exists; needs D-10 patch |
| MTDMA-03 | Tile-1↔tile-2 boundary regression guard, vendor-`.elf`-free | unit | `pytest tests/gtx/test_multi_tile_dma.py -v --no-cov` | ❌ Wave 0 (NEW FILE) |
| MTDMA-04 | State-machine reset across tile boundaries verified | unit (assertions inside MTDMA-03 test) | (subset of MTDMA-03 test) | ❌ Wave 0 |
| VTW-01 | 12-op smoke set PASS | integration | `pytest tests/gtx/test_regression_fw_full_sweep.py -v --no-cov -k 'ABS or ADD or MUL or RELU or SIGMOID or GELU or TANH or LEAKY_RELU or SUM or NEG or DIV or EXP'` | ✅ same harness |
| VTW-02 | Full DDR output region byte-exact | integration | (covered by MTDMA-01 with extended GTX_DDR_DUMP_SIZE) | ✅ |
| VTW-03 | 5x walltime gate fires under HAS_NUMBA=False baseline | benchmark | `pytest tests/gtx/test_njit_perf.py --benchmark-only -k vendor_sweep_walltime_5x` | ✅ exists; needs `tests/gtx/data/baseline_walltime.txt` re-record |
| VTW-04 | Vendor `.elf` policy decision recorded | doc | (manual review of `tests/gtx/data/firmware/README.md`) | ✅ stub exists (1.9K placeholder); needs full content |

### Sampling Rate
- **Per task commit:** `pytest tests/gtx/test_multi_tile_dma.py tests/gtx/test_dma_roundtrip.py tests/gtx/test_deferred_store.py -v --no-cov` (~5 s)
- **Per wave merge:** `pytest tests/gtx/test_multi_tile_dma.py tests/gtx/test_regression_fw_full_sweep.py -v --no-cov -k 'ABS or RELU or GELU'` (~30 s with vendor `.elf` resolved)
- **Phase gate:** `pytest tests/gtx/ -v --no-cov` full suite green + `pytest tests/gtx/test_njit_perf.py --benchmark-only` PASS before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/gtx/test_multi_tile_dma.py` — covers MTDMA-03, MTDMA-04 (tile boundary state assertions)
- [ ] `scripts/import_vendor_golden.py` extension — `--full` flag (or `--all` semantic change) to dump full-region golden, NOT 1-row truncation
- [ ] `tests/gtx/data/firmware/README.md` — VTW-04 4-contract content (BE/LE FP16, GTX_DDR_REVERSED auto-apply, vendor `.elf` import procedure, `_find_elf` priority)
- [ ] `tests/gtx/data/baseline_walltime.txt` — re-record under `HAS_NUMBA=False` (D-12)
- [ ] `MANIFEST.in` + `pyproject.toml [tool.setuptools.package-data]` — D-07 firmware prune

### Risk Register

| Risk | Detection | Mitigation |
|------|-----------|------------|
| D-05 patch breaks P5/P6 hand-built `.elf` resolution | `pytest tests/gtx/test_regression_fw.py` (P5/P6 baseline) — must remain green | _find_elf order is firmware/ → elf/ → vendor; hand-built always wins |
| D-09 test takes >120s in CI (HEIGHT=4097 × 2 tiles, Python loops) | `pytest --durations=20` | Plan-stage option (a) numba, (b) `@pytest.mark.slow`, or (c) parametrize-size (small CI vs big dev) per CONTEXT D-09 |
| `import_vendor_golden.py --all` produces 84×6MB = 500MB of `.hex` if not truncated | git LFS or wheel size guard | Plan-stage decides: full-region golden in dev tree NOT committed (.gitignore); CI generates locally via `--all` once before sweep |
| GTX_DDR_REVERSED=1 contamination across tests | `pytest tests/gtx/test_ddr_modes.py` after each commit | D-10 inline subprocess env (NOT autouse fixture) — env scope-bound to single subprocess |
| Atexit hook fires AFTER pytest captures subprocess output → orphan dumps | subprocess returncode + dump file existence check (already in test) | Existing 5-tier discipline handles |
| Wheel size grows past 50 MB after vendor 84 `.hex` inclusion | `du -sh dist/spike-*.whl` after build; cibuildwheel job | D-07 excludes firmware/; truncated golden ≤32 byte × 84 = ~3KB total |

## Plan-Stage Hand-Off

### Confirmed multi-tile DMA fix location

**There is no single fix location.** The vendor↔pyspike diff matrix shows MATCH at every cross-tile state path. The plan-stage MUST first run D-09 RED-then-GREEN to surface the actual bug, then localize per the Hypothesis 5 lead:

1. **Highest-priority change**: extend `test_regression_fw_full_sweep.py:179-180` `GTX_DDR_DUMP_SIZE` from `0x20` to a per-op full-region size (e.g., `0x600400` for ABS — 393217 × 16 = 0x600400 bytes). Concurrently extend `import_vendor_golden.py` to NOT truncate. Both changes together expose the actual divergence.
2. **Most likely actual fix**: investigation under D-09 will reveal whether the bug is in (a) `firmware_dma_sloop_load` row-loop bounds, (b) `flush_deferred_ddr_stores` ordering, (c) `ddr_dump_to_file` zero-padding off-by-one, or (d) something firmware-side that the vendor C++ also has and pyspike inherited verbatim. **Plan-stage should NOT pre-commit to a fix file:line — let D-09 RED state direct the investigation.**

### Confirmed test files to create / extend

- **NEW**: `tests/gtx/test_multi_tile_dma.py` (Wave 0/1) — D-09 spec:
  - Setup: MockProcessor + GtxNpu, HEIGHT = 4097 (= SHARED_TILE_MAX_ROWS=4095 + 2; 2 tiles), ROW_BYTES = 16, fp16-distinguishable patterns at L1[ADDRA] (input) and DDR base.
  - Drive: simulate firmware tile loop manually — for each tile T in {0, 1}: WRSPR LSPR_SPM_ADDRA + ADDRR; start_p; start_s; firmware_dma LOAD; end_s; start_t; firmware_vec_op SIGN-ABS; end_t; start_s; firmware_dma STORE; end_s; end_p (flushes via `!wsplit_seen` path).
  - Assert (RED before fix): `npu.mem._ddr_bytes[ddr_result_offset + tile_2_byte_range]` matches `np.abs(input).astype(np.float16).view(np.uint16)`.
  - Assert (state, MTDMA-04): tile 1→2 transition `npu.warp.tmu_id`, `curr_id`, `lspr` SPM_ADDR* values match firmware-issued values; `len(npu.deferred_ddr_stores) == 0` after each `end_p`.
- **EXTEND**: `tests/gtx/test_regression_fw_full_sweep.py:97-107` `_find_elf` (D-05).
- **EXTEND**: `tests/gtx/test_regression_fw_full_sweep.py:176-181` subprocess env block (D-10).
- **EXTEND**: `tests/gtx/test_regression_fw_full_sweep.py:179-180` `GTX_DDR_DUMP_SIZE` to per-op dynamic (or extend default to full output region).

### Confirmed assets to import / commit

- `scripts/import_vendor_golden.py` extension — `--all` flag already exists (line 168); plan-stage extends `VENDOR_TO_PYSPIKE_OPS` from 9 to 84 entries and removes `n_data_lines=1` truncation (or adds `--full` flag for full-region).
- 84 `tests/gtx/data/golden/<op>.hex` files — generated once by `import_vendor_golden.py --all`; commit the truncated 32-byte version (~3 KB total, well within wheel constraints) plus a `.gitignore` for full-region versions if generated locally for D-09 RED-state proof.
- `tests/gtx/data/baseline_walltime.txt` — re-record with `HAS_NUMBA=False` venv after MTDMA-01 GREEN.
- **DO NOT** commit any `.elf` files from `/mnt/e/14_NIGHTLY/pyspike/test/` (D-07 wheel exclusion); rely on `GTX_VENDOR_TEST_DIR` env-var lookup.

### Confirmed README sections (`tests/gtx/data/firmware/README.md`, D-08)

The current file is a 1.9 KB placeholder. Plan-stage fills in this order (Claude's discretion on exact section nesting):

1. **BE FP16 vs LE FP16 contract** — quote `vendor/gtx_cpp_reference/gtx/CLAUDE.md` "FP16 바이트 순서" verbatim, then explain pyspike's LE default + how `GTX_DDR_REVERSED=1` flips parser direction.
2. **`GTX_DDR_REVERSED=1` auto-application** — describe D-10 inline subprocess env in `test_regression_fw_full_sweep.py`; cite the file:line.
3. **Vendor `.elf` import procedure** — `GTX_VENDOR_TEST_DIR` env-var (default `/mnt/e/14_NIGHTLY/pyspike/test/`); `import_vendor_golden.py --all` command; expected output `tests/gtx/data/golden/<op>.hex`.
4. **`_find_elf` search priority** — firmware/ → elf/ → ${GTX_VENDOR_TEST_DIR}/<OP>/n1s16/n1s16_<stem>.elf; explain non-breaking ordering.

Add at end: **wheel-size impact statement**: "Firmware/ directory is excluded from wheel via `MANIFEST.in` `prune tests/gtx/data/firmware/` + `pyproject.toml [tool.setuptools.package-data]` excludes `*.elf`. Golden hex (84 × 32 bytes ≈ 3 KB) IS included." (D-07 + VTW-04 closure.)

### Open questions still requiring planner judgment

1. **D-09 HEIGHT and gate strategy** — CONTEXT.md offers (a) numba, (b) `@pytest.mark.slow`, (c) parametrize CI-small/dev-large. Recommendation: HEIGHT=4097 (2 tiles, ~10s pure-Python; numba would speed to <1s); use `@pytest.mark.slow` only if CI run exceeds 30s.
2. **Full-region golden hex commit policy** — in-tree `.hex` (~3 KB total truncated) vs full-region (.gitignore + dev local generate) — recommendation: commit truncated; generate full-region only locally via `import_vendor_golden.py --full`.
3. **GTX_DDR_DUMP_SIZE extension** — single env var across all 84 ops (max-size approach: `0x800000` = 8 MB covers ABS + most others) vs per-op map (`{ABS: 0x600400, RELU: 0x100, ...}`). Recommendation: per-op map keyed by golden file size for CI bandwidth; fallback `0x20` for ops where vendor golden is single-row.
4. **12th smoke op (D-11 plan-stage decision)** — recommendation in this RESEARCH.md is **EXP** (exercises P7 numba objmode path and is a transcendental, complementing GELU/TANH/SIGMOID coverage). Alternatives: NEG (simple unary, fast feedback) or SUB (binary, no transcendental).
5. **Firmware-side bug investigation if D-09 GREEN with no diff** — if D-09 unit test passes RED-then-GREEN with NO pyspike code change (i.e., the bug only manifests under full-region dump but not in tile-2 unit test), this points to `ddr_dump_to_file` zero-padding or atexit ordering. Plan-stage should reserve a contingency Wave for this case.

## Sources

### Primary (HIGH confidence)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc` — lines 240-435 (firmware_dma + flush_deferred_ddr_stores), verbatim read.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc` — lines 1-200, 880-905 (4-mode router + credit_st_chk).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc` — lines 1-143 (P/S/T transitions complete).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` — lines 1-200 (constructor + reset + atexit reg).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc` — lines 1-120 (gemm_core).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc` — lines 1-100 (exec_vector_op).
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — lines 1173, 1251, 1254, 1257-1266 (deferred queue, wsplit_seen, mxe_accum field declarations).
- `vendor/gtx_cpp_reference/CLAUDE.md` + `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — FP16 byte-order contract.
- `src/main/python/riscv/gtx/dma_engine.py` — full file.
- `src/main/python/riscv/gtx/ops/dma.py` — full file.
- `src/main/python/riscv/gtx/ops/control.py` — full file.
- `src/main/python/riscv/gtx/npu.py` — full file.
- `src/main/python/riscv/gtx/warp_state.py` — full file.
- `src/main/python/riscv/gtx/dispatch_4mode.py` — full file.
- `src/main/python/riscv/gtx/ddr.py` — full file.
- `src/main/python/riscv/gtx/vec_engine.py` — full file (ABS path verified at line 285).
- `src/main/python/riscv/gtx/mm_engine.py` — lines 1-100.
- `tests/gtx/test_regression_fw_full_sweep.py` — full file.
- `tests/gtx/conftest.py` — full file.
- `scripts/import_vendor_golden.py` — lines 1-80 (84-op map already inlined).
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/include/kernel_common.h` — lines 50-138 (gtx_tile_meta + GTX_TILE_LOOP macro).
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/kernels/n1s16/n1s16_abs.c` — full file.
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/kernels/n1s16/n1s16_metadata.h` — lines 1-75 (abs HEIGHT=393217, tile_max_rows=4095).
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/kernels/n1s16/n1s16_add1.c` — lines 44-46 (MAX_SHARED_DMA_BYTES=65535 definition).
- `.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md` — Pitfall 7 (wsplit_seen process-lifetime sentinel) + flush trigger reconciliation.
- `.planning/phases/07-numba/07-HUMAN-UAT.md` — multi-tile DMA bug discovery findings.
- `.planning/seeds/p8-multi-tile-dma.md` — 4 hypotheses + symptom (lines 2048+ diverge).
- Filesystem listings of `/mnt/e/14_NIGHTLY/pyspike/test/` (79 dirs) and `/mnt/e/14_NIGHTLY/pyspike/vendor/gtx_cpp_reference/test/` (84 op dirs).
- `ls -lah /mnt/e/14_NIGHTLY/pyspike/test/ABS/n1s16/data/n1s16_abs_ref.txt` (13 MB / 196610 lines, BE FP16 plain hex).
- `wc -l` confirmation of vendor REF file size.

### Secondary (MEDIUM confidence)
- `.planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md` — 13 locked decisions (treated as input, not source).
- `.planning/REQUIREMENTS.md` lines 283-316 (MTDMA-01..04 + VTW-01..04 acceptance text).

### Tertiary (LOW confidence)
- None. Every finding is traced to a file:line read directly from disk.

## Metadata

**Confidence breakdown:**
- DMA path diff (vendor↔pyspike): HIGH — full source reads, no inferred behavior.
- Hypothesis 1-7 evaluations: HIGH for 1, 2, 3, 4, 6, 7; MEDIUM for 5 (depends on plan-stage actually running RED-state; manual dev observation cited).
- Vendor asset 84 vs 79 inventory: HIGH — `ls` + `find` + `wc` cross-check.
- State-machine reset audit: HIGH — every transition mapped 1:1.
- D-05 patch shape: HIGH — exact line-level patch ready for plan-stage.
- 12-op smoke set 12th choice (EXP recommendation): MEDIUM — judgment call; plan-stage can override.

**Research date:** 2026-05-10
**Valid until:** 2026-06-10 (30 days; vendor source unchanged in submodule pinning, pyspike P5/P6/P7 status frozen — invalidate only if v1.2 pyspike changes touch DMA/dispatch/loop)

## RESEARCH COMPLETE
