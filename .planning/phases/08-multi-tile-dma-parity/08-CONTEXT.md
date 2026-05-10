# Phase 8: Multi-tile DMA Parity - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning
**Milestone:** v1.1 — Post-Ship Polish (Multi-tile DMA Orchestration Parity)

<domain>
## Phase Boundary

Vendor 84-op `n1s16` regression sweep (`pyspike/test/<OP>/n1s16/n1s16_<op>.elf`)이
strict-mode `compare_hex(strict=True)`로 vendor `_ref.txt` 골든과 byte-exact PASS,
첫 `MAX_SHARED_DMA_BYTES=65535` tile 경계 너머 **전체 출력 영역**까지 일치.
M ≥ 12 PASS (대표 6 op + 추가 6) 달성. P7 HUMAN-UAT #1 (M ≥ 12) + #2 (5x walltime
HAS_NUMBA=False baseline)을 `/gsd:verify-work 7`로 종결. vendor `.elf`-free
tile-2 unit test로 회귀 방지 가드.

**Out of scope (다른 페이즈):**
- 새로운 op 추가, 새로운 ISA 인코딩 — v2
- mxe_accum 4D 확장, multi-hart 지원 — v2
- vendor 펌웨어 build chain (gtx-firmware/) 직접 통합 — v2
- numba 추가 최적화 (correctness만; 5x는 자연 도달 예상) — v2

</domain>

<decisions>
## Implementation Decisions

### Area 1: 디버깅 vs 포팅 전략 (D-01 ~ D-04)

- **D-01 Strategy = Hybrid (diff → hypothesis 순결).**
  Wave 0: vendor C++ ↔ pyspike Python 1:1 diff 수행 → 구조적 누락은 verbatim port,
  남은 cross-tile 상태 버그는 hypothesis test로 분리. 시간 효율과 root-cause
  확신을 모두 확보. P3/P4/P5/P6 패턴(verbatim port 우선, 검증 분리)과 일관.

- **D-02 Diff scope = DMA + dispatch + loop + compute engines.**
  vendor 파일 → pyspike 모듈 매핑 (1:1 diff 대상):
  - `gtx_npu_dma.cc` ↔ `dma_engine.py` + `ops/dma.py`
  - `gtx_npu_dispatch.cc` ↔ `dispatch.py` + `dispatch_4mode.py`
  - `gtx_npu_loop.cc` ↔ `warp_state.py` + `ops/control.py`
  - `gtx_npu_mm.cc` ↔ `mm_engine.py` + `gemm_core.py` + `ops/mm.py`
  - `gtx_npu_vec.cc` ↔ `vec_engine.py` + `vec_core.py` + `ops/vec.py`
  - `gtx_npu_act.cc` ↔ `act_engine.py` + `act_core.py` + `ops/act.py`
  - `gtx_npu_core.cc` ↔ `npu.py` (state lifecycle)
  컴퓨트 path도 cross-tile 상태(mxe_accum, deferred queue 소비자) 의심되므로 포함.

- **D-03 Instrumentation = 테스트단 일회성 스냅샷 only.**
  `tests/gtx/test_multi_tile_dma.py`에서 `npu.mem._ddr_bytes` / `l2` / `l1` /
  `npu.deferred_ddr_stores` / `npu.warp` 직접 스냅샷 + assert. production 코드
  변경 없음 (`GTX_DEBUG_TILE_TRACE` 환경변수 추가 ❌, `_debug.py` 모듈 추가 ❌).
  fix가 끝난 후에도 스냅샷 코드는 회귀 가드로 유지.

- **D-04 Fix scope = Root-cause에 지역화 (수술적).**
  diff/hypothesis가 지목한 구체 라인에만 패치. 다른 테스트 영향 최소.
  diff에서 발견된 잠재 누락(현재 문제와 무관한 것)은 `.planning/seeds/p9-*.md`로
  기록 후 **이번 페이즈에서는 fix하지 않음**. P8 종결 조건 = vendor sweep
  M ≥ 12 PASS + tile-2 unit test GREEN (그 이상의 코드 변경은 P9/v1.2).

### Area 2: vendor `.elf` / `_ref.txt` 자산 정책 (D-05 ~ D-08)

- **D-05 ELF policy = `_find_elf` multi-path search 확장.**
  `tests/gtx/test_regression_fw_full_sweep.py:_find_elf`에 vendor 경로 후보
  추가 (3번째 candidate). 환경변수 `GTX_VENDOR_TEST_DIR` 지원하되 기본값
  `/mnt/e/14_NIGHTLY/pyspike/test/`. wheel 증가 0, dev/CI 모두 override 가능.
  검색 우선순위: `firmware/<stem>.elf` > `elf/<stem>.elf` > `${GTX_VENDOR_TEST_DIR}/<OP>/n1s16/n1s16_<stem>.elf`.

- **D-06 REF policy = `import_vendor_golden.py --all` → 84 op .hex 생성.**
  기존 9 op 한정 도구를 84 op 전체 대상으로 확장. vendor `_ref.txt`(BE FP16
  plain text) → `tests/gtx/data/golden/<op>.hex`(P7 _verify 호환 LE FP16
  binary) 변환. dev가 한 번 실행. 결과 .hex는 GTX_DDR_DUMP_SIZE=0x20 (32 byte)
  분량이라 op당 ~100 byte → 84 op 전체 commit해도 ~10KB. git 자산화.

- **D-07 Wheel = `tests/gtx/data/firmware/`는 wheel에서 제외.**
  P6 PKG-01의 .elf wheel 포함 결정을 v1.1에서 되돌림. MANIFEST.in /
  setup.py package-data에서 firmware/ 디렉토리 prune. 사유: vendor .elf는
  `/mnt/e/14_NIGHTLY/pyspike/test/`에서 multi-path로 찾고, P5/P6 hand-built
  .elf는 dev source checkout에만 필요. wheel size ≤50MB 제약 안전 마진 확대.
  `tests/gtx/data/golden/<op>.hex`는 wheel에 **포함**(작고, dev/CI 모두 필요).

- **D-08 Doc = `tests/gtx/data/firmware/README.md` 단일 파일에 4 contract 통합.**
  포함 내용:
  1. BE FP16 (vendor) ↔ LE FP16 (pyspike default) 바이트 순서 contract
  2. `GTX_DDR_REVERSED=1` 자동 적용 조건 (D-10 `test_regression_fw_full_sweep.py` 인라인)
  3. vendor `.elf` import 절차 (`GTX_VENDOR_TEST_DIR` env var, `import_vendor_golden.py --all` 실행)
  4. `_find_elf` 검색 우선순위 (firmware/ → elf/ → vendor)
  - VTW-04 closure는 이 README가 land됨 + MANIFEST.in 변경 + `.planning/codebase/ARCHITECTURE.md`에 BE/LE 노트 추가하면 충족.

### Area 3: Tile-2 unit test 설계 (D-09)

- **D-09 Tile-2 test = Python-programmatic + MockProcessor + ABS compute + HEIGHT≥SHARED_TILE_MAX_ROWS+1.**
  - **형태**: `tests/gtx/test_multi_tile_dma.py`에 MockProcessor / MockInsn 사용,
    `_riscv.so` 의존 없음. P3 `test_dma_roundtrip` / `test_deferred_store` 패턴 확장.
    vendor `.elf` 의존 0.
  - **scope**: tile boundary state reset (Hypothesis #4 우선). assert 대상:
    `npu.warp.tmu_id` / `npu.warp.curr_id` / `npu.lspr[][LSPR_SPM_ADDR*]` /
    `npu.deferred_ddr_stores` 길이 — tile 1 종료 ↔ tile 2 진입 사이 의도된 reset
    상태 일치. RED 조건: tile 2 LOAD가 tile 1과 동일 주소를 읽거나 stale L1
    bank를 참조해 byte-mismatch.
  - **fixture 크기**: HEIGHT = `SHARED_TILE_MAX_ROWS + small extra` (예: 32769 rows,
    실제 tile loop 발동) — vendor 펌웨어 시나리오 충실 재현. ⚠️ Python 루프 +
    MockProcessor에서는 수십 초~수 분 가능 → plan-stage에서 옵션 검토:
    (a) numba JIT 적용 (이미 P7 인프라 land), (b) `@pytest.mark.slow` 마커,
    (c) CI에서는 작은 사이즈 + dev local에서 큰 사이즈로 parametrize.
  - **컴퓨트 커널**: ABS (P7 smoke set과 동형) — `vec_engine._apply_unary` SIGN→ABS
    경로 재사용. golden = NumPy `np.abs(input).astype(np.float16)`. 실전 발생
    가능성과 1:1 대응.
  - **RED→GREEN 증명**: plan SUMMARY에 fix 적용 전(RED, byte-mismatch 라인 번호)
    + fix 적용 후(GREEN, byte-exact) 증거 명시.

### Area 4: Smoke set 12개 op + GTX_DDR_REVERSED 자동화 (D-10 ~ D-13)

- **D-10 GTX_DDR_REVERSED 자동화 = `test_regression_fw_full_sweep.py` 인라인 set.**
  vendor `.elf` 경로(D-05 multi-path)에서 발견된 fixture만 `subprocess.run(env=...)`
  에 `'GTX_DDR_REVERSED': '1'` 주입. P5/P6 hand-built `.elf`는 영향 없음. conftest
  autouse fixture / pytest marker 채택 안 함 (cross-test contamination 방지).

- **D-11 Smoke set 12개 = vendor `.elf` 경로 활성화로 OPERAND_STAGING skip 자동 해소.**
  vendor `.elf`는 `ddr_init_from_file`로 자체 input 스테이징 → P7
  `OPERAND_STAGING_REQUIRED_VENDOR` skip 사유 무효. _find_elf가 vendor 경로
  fixture를 찾으면 skip 마킹 우회 (skip 조건 = "vendor `.elf`도 hand-built도 둘
  다 없을 때"). 결과:
  - **명시 6개** (ROADMAP P8 success #1): ABS, ADD_VV, MUL_VV, RELU, SIGMOID, GELU
  - **추가 6개**: TANH, LEAKY_RELU, ADD, MUL, SUM (vendor `.elf` 활성화로 자연
    PASS 가능 후보) + 1개 plan-stage 결정 (vendor 디렉토리 카운트 검증 후).
  - 이상적: 79 vendor `.elf` 전체가 PASS → M ≥ 60 도달, 그 위에 floor 12 충족.

- **D-12 VTW-03 = HAS_NUMBA=False baseline 재기록 (REQUIREMENTS 원안 유지).**
  MTDMA-01 fix 후 vendor sweep PASS 확정 → `tests/gtx/data/baseline_walltime.txt`
  를 `HAS_NUMBA=False`(numba 미설치 venv)로 재기록 → `pytest tests/gtx/test_njit_perf.py
  --benchmark-only`가 `test_vendor_sweep_walltime_5x` PASS (`mean*5 ≤ baseline`).
  P7 HUMAN-UAT #2 closure. 30s skip threshold 안 걸리도록 baseline > 30s 확보.

- **D-13 vendor 경로 의존 = `GTX_VENDOR_TEST_DIR` env var 기본값 + override.**
  `_find_elf` 안에서 `os.environ.get('GTX_VENDOR_TEST_DIR',
  '/mnt/e/14_NIGHTLY/pyspike/test/')` 패턴. README에 명시. CI/다른 dev 환경
  대응. v1.2에서 submodule 정식화 검토 (deferred).

### Claude's Discretion

- **다음 사항은 plan-stage / executor 재량**:
  - 1:1 diff의 구체 형식 (markdown table vs 인라인 주석 vs 별도 RESEARCH.md
    부록) — diff 결과를 누가 어떻게 보관하느냐는 plan 단계에서.
  - tile-2 test의 정확한 HEIGHT 값과 numba 적용 여부 (D-09 옵션 a/b/c 중 택).
  - `import_vendor_golden.py --all` 확장 시 기존 9-op 매핑 보존 vs 단순화.
  - VTW-04 README.md의 정확한 섹션 구성 (4 contract 순서/깊이).
  - 12 op smoke set의 12번째 op 선정 (vendor 디렉토리 카운트 후 plan-stage).

### Folded Todos

해당 없음 (todo cross-reference 매칭 없음).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 8 핵심 ROADMAP / REQ 자료

- `.planning/ROADMAP.md` §"### Phase 8: Multi-tile DMA Parity" (lines 239-256) — Goal,
  Depends on, Requirements (MTDMA-01..04, VTW-01..04), 5 Success Criteria.
- `.planning/REQUIREMENTS.md` §"## Milestone v1.1 Post-Ship Polish" (lines 283-316) —
  MTDMA-01..04 + VTW-01..04 정의 (한국어 acceptance 텍스트).
- `.planning/STATE.md` lines 1-44 — milestone v1.1 frontmatter, Phase 8 current
  position, Performance Metrics (M=0 → M ≥ 12 target).

### P8 trigger artifact (P7 HUMAN-UAT)

- `.planning/seeds/p8-multi-tile-dma.md` — Goal/Symptom/4 Hypotheses/Investigation
  steps/Acceptance Criteria. P7 ABS smoke test가 발견한 정확한 증상 (lines 2048+
  diverge, 첫 ~64KB byte-exact).
- `.planning/phases/07-numba/07-HUMAN-UAT.md` Findings (lines 27-34) — vendor
  asset 위치, numba speed (4.8s), endianness root cause, multi-tile DMA bug
  발견 경위.

### Vendor C++ source (1:1 diff target — D-02)

- `vendor/gtx_cpp_reference/gtx/gtx_npu_dma.cc` (특히 lines 249-397: firmware_dma
  + S/T loop 분기, lines 415-435: flush_deferred_ddr_stores).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc` — 4-mode dispatch (Mode 1
  brodcast, Mode 3 P+S, Mode 4 P+T) + dispatch_iss_opcode 라우팅.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_loop.cc` — startp/endp/starts/ends/startt/endt
  state transitions (extract_id 포함).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc` — gemm_core, exec_mm*, mxe_accum
  (FP32 누적 상태가 cross-tile 영향 가능).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc` — exec_vector_op, firmware_vec_op
  (ABS는 SIGN 분기에 흐름).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc` — exec_activation 방향성 (D-09
  ABS는 vec path지만 RELU/SIGMOID/GELU는 act path → smoke set 검증).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_core.cc` (lines 80-141) — 생성자/reset
  state lifecycle, sp init, mxe_accum zero-init, deferred queue clear.
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — gtx/ 디렉토리 구조, FP16 byte
  order regs (LE for L1/L0, DDR is BE under GTX_DDR_REVERSED), Activation
  방향성 표.

### Firmware tile loop reference (debug 이해용)

- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/include/kernel_common.h` (lines 60-107)
  — `struct gtx_tile_meta` + `GTX_TILE_LOOP` macro (firmware-side tile 반복).
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/kernels/n1s16/n1s16_abs.c` —
  ABS 펌웨어 어댑터 (vendor `.elf` source).
- `/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/kernels/n1s16/n1s16_add1.c` (lines
  44-46) — `MAX_SHARED_DMA_BYTES=65535u`, `SHARED_TILE_MAX_ROWS = 65535/ROW_BYTES`
  정의.

### Prior phase contexts (decision precedent)

- `.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md` — D-01 DMA 모듈 분리,
  D-04 DeferredDdrStore dataclass, D-08 GTX_DDR_REVERSED env var read 정책,
  D-10 Python-only programmatic 회귀, D-13 ensure_ddr doubling-grow.
- `.planning/phases/03-dma-ddr-i-o/03-RESEARCH.md` "Pitfall 7" — wsplit_seen
  process-lifetime sentinel (reset에서 안 지움). MTDMA-04 가설 검증과 직결.
- `.planning/phases/04-mm-subsystem/04-CONTEXT.md` — mxe_accum FP32 (NEST,SPU)
  shape 정의 (4D 아님!). cross-tile 상태 의심 시 검증 필수.
- `.planning/phases/05-vec-act-pool/05-CONTEXT.md` — VEC/ACT engine 형태,
  ABS는 vec_engine SIGN 분기.
- `.planning/phases/07-numba/07-CONTEXT.md` D-09 (objmode 5 transcendentals)
  — numba 적용 시 IEEE 754 보장 메커니즘. tile-2 test에 numba 적용 검토 시 참조.
- `.planning/phases/07-numba/07-VERIFICATION.md` — VTW-03 baseline 재기록
  맥락 (HAS_NUMBA=False).

### Code context (P3/P5/P6/P7 already-landed)

- `src/main/python/riscv/gtx/dma_engine.py` (`firmware_dma_sloop_load/store`,
  `firmware_dma_tloop_load_store/copy`, `decode_firmware_dma_args`,
  `DeferredDdrStore`).
- `src/main/python/riscv/gtx/ops/dma.py` (`@handler` entry points + `_select_nest`
  / `_select_spu` 헬퍼 — Mode 3/4 라우팅).
- `src/main/python/riscv/gtx/ops/control.py` (`_do_startp/endp/starts/ends/startt/endt`
  + WSPLIT/WJOIN handlers).
- `src/main/python/riscv/gtx/npu.py` (`flush_deferred_ddr_stores`,
  `reset` lifecycle, `_LAST_NPU` global, `_init_ddr_from_env`).
- `src/main/python/riscv/gtx/warp_state.py` (`WarpState.reset` — `wsplit_seen`
  NOT cleared).
- `src/main/python/riscv/gtx/dispatch_4mode.py` — 4-mode dispatch entry.
- `src/main/python/riscv/gtx/_verify.py` — `compare_hex(strict=True)` API.
- `tests/gtx/test_regression_fw_full_sweep.py` — 5-tier graceful skip,
  `_find_elf` (D-05 확장 대상), `_find_golden`,
  `OPERAND_STAGING_REQUIRED_VENDOR` (D-11 자연 해소 대상).
- `tests/gtx/conftest.py` — `MockProcessor`/`MockInsn` 패턴, `baseline_walltime`
  fixture (VTW-03 재기록 대상).
- `tests/gtx/_mocks.py` — D-09 Python-programmatic test 의존.
- `tests/gtx/_njit_helpers.py` — numba JIT 적용 시 D-09 옵션 (a) 인프라.
- `tests/gtx/data/firmware/` (wheel 제외 결정 D-07) — `*.elf` 자산.
- `tests/gtx/data/golden/` (wheel 포함, D-06 확장 대상) — `*.hex` 자산.
- `scripts/import_vendor_golden.py` (현 9 op → D-06로 84 op 확장).

### Build / Distribution

- `pyproject.toml` `[tool.setuptools.package-data]` — D-07 firmware 제외, golden
  포함 명시.
- `MANIFEST.in` — D-07 prune `tests/gtx/data/firmware/`.
- `setup.py` — package-data resolution.

### Project documents (locked context)

- `CLAUDE.md` — Project / Constraints (Pure Python + NumPy, NumPy>=1.20,
  Bit-exact ULP/atol, manylinux2014).
- `vendor/gtx_cpp_reference/CLAUDE.md` (line 158-160) — GTX_DDR_REVERSED 사용
  조건 (HW sim 데이터 사용 시 반드시 1).
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` "FP16 바이트 순서" / "DDR Hex 파일
  바이트 순서" — D-08 README 작성 시 직접 인용.

### v1.1 backlog seed (out-of-P8 follow-up)

- `.planning/seeds/p9-*.md` — D-04 단계에서 발견된 잠재 누락은 여기에 기록.
  P8 plan-stage가 만들 예정 (현재 비존재).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (P3/P5/P6/P7 산출물 — P8가 직접 사용/확장)

- **`tests/gtx/_mocks.py:MockProcessor` + `MockInsn`** — D-09 Python-programmatic
  test 핵심. `_riscv.so` 의존 없이 custom0/custom1 시뮬레이션. P3 `test_deferred_store.py`
  에서 동일 패턴 사용 검증 완료.
- **`src/main/python/riscv/gtx/dma_engine.py:firmware_dma_sloop_load/store`** —
  vendor C++ verbatim 포팅 (lines 294-313, 269-287). 함수 시그니처 안정 — D-09
  test가 직접 호출 가능.
- **`src/main/python/riscv/gtx/_verify.py:compare_hex(strict=True)`** — VTW-01 +
  VTW-02 검증 핵심. 그대로 재사용.
- **`scripts/import_vendor_golden.py`** — 현재 9 op 매핑 (P6 land). D-06이 84 op
  확장. CLI 옵션 `--all` 기존에 존재할 가능성 — plan-stage에서 확인.
- **`tests/gtx/test_regression_fw_full_sweep.py:_find_elf` + `_find_golden`** —
  D-05 multi-path search 확장 대상. 5-tier skip discipline 그대로 유지.
- **`tests/gtx/conftest.py:baseline_walltime`** — D-12 fixture (이미 land).
  값만 재기록.
- **`tests/gtx/_njit_helpers.py`** — D-09 옵션 (a) numba 가속 인프라.

### Established Patterns

- **vendor verbatim port + 전용 RESEARCH.md** — P3/P4/P5 모두 사용. D-01 hybrid
  diff 결과를 P8 RESEARCH.md에 기록 (혹은 인라인 주석).
- **Python-only programmatic 회귀 우선** — Phase 3 D-10. D-09가 직접 계승.
  `.elf` fixture는 P5/P6 + vendor에 한정.
- **`@handler` decorator + 2-level dispatch** (Phase 3 D-03). P8가 새로 추가하지
  않음 — 기존 라우팅 그대로 사용.
- **Lazy `ensure_ddr` doubling-grow** (Phase 3 D-13). cross-tile 큰 DDR 접근에서
  자동 확장 — D-09 HEIGHT=32K+ 시나리오에서 검증.
- **graceful skip 5-tier** (P6 06-04 → P7 07-05 계승). D-05/D-06가 깨지지 않게
  보존.

### Integration Points

- `_find_elf`: P3 test_dma_roundtrip + P4/P5 hand-built .elf + **P8 vendor .elf
  multi-path** 의 합집합 검색.
- `import_vendor_golden.py`: P6 9 op + **P8 84 op** 합집합. CLI flag로 분기.
- `flush_deferred_ddr_stores` 트리거 = `end_p` (when !wsplit_seen) + `credit_st_chk`
  (when is_sloop). D-04 fix가 이 트리거를 건드리면 P3 D-06 / P3 RESEARCH "Deferred
  Store Flush Trigger" 재검증 필수.
- MANIFEST.in / pyproject.toml: D-07 firmware 제외 — P6 PKG-01 결정 reverse.
  P6 cibuildwheel matrix(cp310-cp312)는 영향 없음.

### Anti-patterns to avoid (D-04 root-cause 지역화 원칙 + P3 PITFALLS)

- **diff 결과 기반 광역 코드 재작성 금지** (D-04). P5/P6에서 검증된 verbatim port
  를 P8가 흔들면 회귀 위험. 발견된 잠재 누락은 seed에 기록만.
- **production 코드에 debug 환경변수 추가 금지** (D-03). `GTX_DEBUG_TILE_TRACE`,
  `_debug.py` 모듈 추가 모두 ❌. 회귀 가드는 test-side에 둠.
- **conftest autouse fixture로 GTX_DDR_REVERSED set 금지** (D-10). cross-test
  contamination — vendor sweep만 인라인 set.
- **HAS_NUMBA=True baseline 기록 금지** (D-12). 5x assert 의도 깨짐. REQUIREMENTS
  원안 고수.

</code_context>

<specifics>
## Specific Ideas

### "1:1 diff 결과 보관 형태" (plan-stage 결정)
diff 결과(vendor C++ vs pyspike Python)를 어디에 기록하느냐는 plan-stage 결정
(Claude 재량). 후보:
1. `08-RESEARCH.md` 부록 (P3/P4/P5 패턴 — 가장 일관적)
2. 인라인 주석 (`vendor/gtx_npu_dma.cc:NNN` 형식 라인 매핑)
3. 별도 `08-DIFF-AUDIT.md` (audit 우선)

### "tile boundary state reset" 의 구체 검증 항목 (D-09 RED/GREEN 지표)
- `npu.warp.tmu_id == nest_id_at_tile_2_start` (after start_p)
- `npu.warp.curr_id == spu_id_at_tile_2_start` (after start_t)
- `npu.lspr[nest][spu][LSPR_SPM_ADDRA]` / `LSPR_SPM_ADDRR` / `LSPR_SPM_ADDRB`:
  tile 2 시점 수정된 값 (tile 1 잔존값 ❌)
- `len(npu.deferred_ddr_stores) == 0` (after end_p flush at tile 1 boundary)
- `npu._mxe_accum`: ABS 시나리오 무관 (compute가 mxe 사용 안 함) → 0 유지 확인.

### "vendor 디렉토리 카운트 84 vs 79" 모순
- ROADMAP P7: vendor 디렉토리 84개 op
- HUMAN-UAT: pre-built `.elf` 79개
- 차이 = 5 op는 vendor `.elf`도 hand-built도 없음 (예: DIAG, CONCAT 등 — plan-stage
  에서 카운트 검증). M+N=84 invariant가 깨질 가능성 → P8 plan-stage에서
  test_regression_fw_full_sweep parametrize 카운트 점검.

### `.elf` 검색 우선순위 의미
firmware/ 우선 → P5/P6 hand-built가 vendor 보다 우선 → vendor 펌웨어
업데이트로 hand-built 깨질 위험 ❌. dev intent: hand-built `.elf`는 P5/P6
specific test (.S 단순 커널), vendor `.elf`는 vendor 펌웨어 회귀 → 다른 케이스를
다른 fixture로 검증. 매핑 충돌은 거의 없음 (op 이름 lowercase 대 op_dir naming
규칙).

### MAX_SHARED_DMA_BYTES=65535 정확한 의미
firmware-side 상수 (Zephyr kernel `n1s16_*.c`). NPU 모델 입장에서는 단순히
"firmware가 길이 ≤ 65535 byte의 DMA만 발행한다" 제약. NPU는 length 매개변수
값을 그대로 처리 — pyspike DMA에 hard-coded된 65535 분기 ❌. tile 경계는
firmware가 결정.

</specifics>

<deferred>
## Deferred Ideas

### Out of P8 scope (다른 페이즈로)

- **multi-hart 지원** (현재 single-hart 전제. tile loop는 single-hart 내 firmware
  반복 — multi-hart는 v2 이상).
- **vendor 펌웨어 build chain 통합** (`/mnt/e/14_NIGHTLY/gtx_spike/gtx-firmware/`
  의 cmake 빌드를 pyspike CI가 직접 호출하는 통합 — v2).
- **submodule로 vendor 자산 정식화** (D-13 dev path 우선, v1.2 결정).
- **mxe_accum 4D 확장** (현재 (NEST, SPU) 2D — vendor C++도 동일. v2에서 batch 차원).
- **cibuildwheel matrix에 vendor 자산 통합** (CI에서 vendor sweep 활성화 — v1.2).

### Within-domain ideas surfaced but not selected for discussion

- **GTX_DEBUG_TILE_TRACE 환경변수** (D-03이 reject — production 코드 오염).
- **`_debug.py` 모듈** (D-03이 reject — 동상).
- **광역 verbatim re-port 11 .cc 전체** (D-01이 reject — 시간 폭발 + P5/P6 검증
  종결 부분 흔들림 위험).
- **conftest autouse fixture로 GTX_DDR_REVERSED 자동화** (D-10이 reject —
  contamination).
- **@pytest.mark.vendor_be_fp16 marker** (D-10이 reject — registry 복잡도).
- **wheel에 vendor `.elf` 직접 포함** (D-07이 reject — wheel size).

### Reviewed Todos (not folded)

해당 없음.

### Defer to user follow-up

- **VTW-04 README.md 정확한 섹션 구성** — D-08이 4 contract 통합으로 결정,
  세부 섹션 순서/깊이는 plan-stage 재량.
- **12 op smoke set의 12번째 op** — D-11이 11개 명시 + 1개 plan-stage 결정.
- **diff 결과 보관 형태** (RESEARCH.md 부록 / 인라인 / 별도 DIFF-AUDIT.md) —
  plan-stage 재량.
- **D-09 tile-2 test 가속 옵션** (numba / @pytest.mark.slow / parametrize 사이즈)
  — plan-stage 재량.

</deferred>

---

*Phase: 08-multi-tile-dma-parity*
*Context gathered: 2026-05-10*
*Milestone: v1.1 — Post-Ship Polish*
*Requirements: MTDMA-01, MTDMA-02, MTDMA-03, MTDMA-04, VTW-01, VTW-02, VTW-03, VTW-04*
