# Phase 8: Multi-tile DMA Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 08-multi-tile-dma-parity
**Areas discussed:** Debug-vs-Port Strategy, Vendor `.elf`/`_ref.txt` Asset Policy, Tile-2 Unit Test Design, Smoke Set + GTX_DDR_REVERSED Automation

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| 디버깅 vs 포팅 전략 | (1) hypothesis 검증 (2) verbatim re-port (3) hybrid 사이 선택 | ✓ |
| vendor `.elf`/_ref.txt 자산 정책 (VTW-04) | 472MB의 vendor assets를 어떻게 wire-up | ✓ |
| Tile-2 unit test 설계 (MTDMA-03) | vendor `.elf`-free 회귀 가드 형태 | ✓ |
| Smoke set 12개 op + GTX_DDR_REVERSED 자동화 (VTW-02 + MTDMA-02) | M ≥ 12 floor + env 자동 적용 | ✓ |

**User's choice:** All four areas selected for discussion (multi-select).

---

## Area 1: 디버깅 vs 포팅 전략

### Q1: 디버깅 수서를 어떻게 잡을까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 하이브리드: diff → hypothesis 순결 (추천) | Wave 0 vendor diff, 남은 가설은 instrumentation으로. 구조적 누락은 verbatim, 상태 버그는 hypothesis test로 분리 | ✓ |
| Hypothesis-first: 4개 가설 순서대로 검증 | P8 seed의 가설 1→4 순서 print/trace, fix는 마지막 | |
| Vendor Re-port: dma.cc + dispatch.cc 전체 verbatim 재이식 | 11개 .cc 중 multi-tile 경로 닿는 파일 전체 verbatim. lurking divergence 일괄 제거 | |
| Hypothesis #4 단독 fix | seed 가설 4 (plan/thread state machine reset) 만 검증, 시간 최소 | |

**User's choice:** 하이브리드: diff → hypothesis 순결 (추천)
**Notes:** D-01에 캡처. Wave 0 diff 결과로 구조적 누락 식별 + 잔여 상태 버그를 hypothesis test로 분리.

### Q2: 1:1 diff 수행 시 어디까지 커버할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| DMA + dispatch + loop 제한 (추천) | 3개 파일만 diff (multi-tile orchestration 경로) | |
| + MM/VEC/ACT engine까지 | compute path도 cross-tile state 의심 → 포함 | ✓ |
| DMA 단독 (최소) | dma_engine.py + ops/dma.py만 | |
| 전체 11 .cc 파일 | 모든 .cc 파일 명시적 diff | |

**User's choice:** + MM/VEC/ACT engine까지
**Notes:** D-02에 캡처. mxe_accum, deferred queue 소비자 등 cross-tile 상태가 compute path에도 있을 수 있어 포함. diff 범위 확대 → 시간 증가하지만 회귀 안정성 우선.

### Q3: 디버깅 계측기 (instrumentation)는 어떻게?

| Option | Description | Selected |
|--------|-------------|----------|
| 테스트단 일회성 스냅샷 (추천) | test_multi_tile_dma.py에서 직접 npu.mem._ddr_bytes 등 스냅샷. production 코드 변경 ❌ | ✓ |
| GTX_DEBUG_TILE 환경변수 추가 | production 코드에 환경변수 훅, v1.1 이후도 재사용 가능하지만 production 코드 오염 | |
| snapshot helper module | src/.../_debug.py 추가, 재사용용이하지만 source에 새 파일 | |
| 계측기 없이 fix | diff 결과만으로 충분 시 직접 fix | |

**User's choice:** 테스트단 일회성 스냅샷 (추천)
**Notes:** D-03에 캡처. 회귀 가드는 test-side에 둠.

### Q4: fix를 어디에 넣을까요?

| Option | Description | Selected |
|--------|-------------|----------|
| Root-cause에 지역화 (추천) | diff/hypothesis가 지목한 구체 라인에만 패치, 다른 테스트 영향 최소 | ✓ |
| 광역 포팅 (탄당 계수 포함) | 잠재 누락도 같이 fix, 잠재 버그 대비 | |
| P8는 root-cause만, 나머지 도우 인레 등록 후 v1.2 | 발견된 잠재 누락은 seeds에 기록, 종결 조건 명확 | |
| Claude의 재량 — plan-stage에서 결정 | diff 결과 본 뒤 결정 | |

**User's choice:** Root-cause에 지역화 (추천)
**Notes:** D-04에 캡처. 발견된 잠재 누락은 `.planning/seeds/p9-*.md`로 기록만, 이번 페이즈는 fix하지 않음.

---

## Area 2: vendor `.elf` / `_ref.txt` 자산 정책 (VTW-04)

### Q1: .elf 자산을 어떻게 wire-up할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| _find_elf multi-path search 확장 (추천) | 기존 _find_elf에 vendor 경로 후보 추가, GTX_VENDOR_TEST_DIR env var 지원 | ✓ |
| import_vendor_elf.py 확장 | dev이 명시적으로 .elf만 firmware/ 로 복사 | |
| vendor submodule 확장 | vendor/gtx_npu_test_assets/ 추가 submodule | |
| git LFS 로 직접 커밋 | tests/gtx/data/firmware/vendor/ 아래 79 .elf 직접 LFS 커밋 | |

**User's choice:** _find_elf multi-path search 확장 (추천)
**Notes:** D-05에 캡처. wheel 증가 0, dev/CI 모두 override 가능.

### Q2: _ref.txt (golden, 각 ~12MB) 정책은?

| Option | Description | Selected |
|--------|-------------|----------|
| import_vendor_golden.py 확장해 .hex 생성 (추천) | --all 옵션으로 84 op 전체 _ref.txt → .hex 변환, .hex commit | ✓ |
| .hex commit (압축) | dump 적은 부분만이라 100바이트 수준, commit 가능 | |
| _ref.txt 원본 multi-path search | .hex 변환 안 하고 _ref.txt를 fly로 파싱 | |
| VTW-04는 .hex 변환만 언구, 원본 .ref는 manual reproduce | dev 로컬/vendor 레포에 남김 | |

**User's choice:** import_vendor_golden.py 확장해 .hex 생성 (추천)
**Notes:** D-06에 캡처. op당 ~100 byte → 84 op 전체 commit해도 ~10KB. git 자산화.

### Q3: wheel size 제약을 어떻게 맞출까요?

| Option | Description | Selected |
|--------|-------------|----------|
| tests/gtx/data/firmware/ 는 wheel에서 제외 (추천) | MANIFEST.in / package-data에서 .elf, .hex 제외. dev test는 source checkout에서만 | ✓ |
| wheel에 .elf 따로, _ref/.hex는 제외 | .elf 포함 하되 .hex는 제외 | |
| wheel에 전부 포함 (압축된 .hex 작으면 OK) | .hex가 몇 KB 이하면 .elf + .hex 모두 wheel에 | |
| wheel size 측시 안 함 (P6 결정 답른) | tests/gtx/data/firmware/ 내 자산 추가 시 current rule대로 | |

**User's choice:** tests/gtx/data/firmware/ 는 wheel에서 제외 (추천)
**Notes:** D-07에 캡처. P6 PKG-01의 .elf wheel 포함 결정을 v1.1에서 되돌림. golden .hex는 wheel에 **포함**(작고, dev/CI 모두 필요).

### Q4: tests/gtx/data/firmware/README.md 문서화 범위는?

| Option | Description | Selected |
|--------|-------------|----------|
| BE↔LE FP16 + GTX_DDR_REVERSED + ELF 자산 import 절차 (추천) | MTDMA-04 요구사항 — 4 contract 단일 README에 모두 조광 | ✓ |
| VTW-04 결정 명시 석섭 수준으로 | wheel 포함/제외 정책 + .elf 자산 소스에 집중 | |
| 상세 README + ARCH.md 단계적 업데이트 | README는 간략, codebase/ARCHITECTURE.md에 BE/LE 세부 | |
| Claude의 재량 | Phase 8 plan-stage에서 더 자세한 구조를 설계 | |

**User's choice:** BE↔LE FP16 + GTX_DDR_REVERSED + ELF 자산 import 절차 (추천)
**Notes:** D-08에 캡처. 4 contract: BE↔LE byte order, GTX_DDR_REVERSED 자동 적용 조건, vendor `.elf` import 절차, `_find_elf` 검색 우선순위.

---

## Area 3: Tile-2 unit test 설계 (MTDMA-03)

### Q1: tile-2 테스트의 의존 형태는?

| Option | Description | Selected |
|--------|-------------|----------|
| Python-programmatic + MockProcessor (추출) | _riscv.so 없이 firmware_dma 핸들러 직접 호출, deferred queue 및 L1/L2/DDR 검증 | ✓ |
| Python-prog. + .elf 메타 머실림 | MockProcessor에서 custom0 명령 시퀀스 직접 재생 | |
| 인라인 .S 빌드 + 소용 ELF 주입 | tests/gtx/data/firmware/multi_tile_2.elf 수동 .S에서 빌드 | |
| Hybrid: 주테스트 = programmatic, 결정한을 설명제로 .S | 둘 다 | |

**User's choice:** Python-programmatic + MockProcessor (추출)
**Notes:** D-09에 캡처. P3 패턴 확장. vendor `.elf` 의존 0.

### Q2: 테스트가 잡아야 하는 경우(case)는?

| Option | Description | Selected |
|--------|-------------|----------|
| tile boundary state reset (추천) | Hypothesis #4 우선 — warp.tmu_id, curr_id, lspr 검증 | ✓ |
| DDR↔L2 포인터 advance | 가설 #1 — ddr_off 증가 검증 | |
| L1 bank recycle + deferred queue ordering | 가설 #2/#3 — STORE 후 LOAD stale 검증 | |
| All four hypotheses (파러메트라이즈된 suite) | 4개 가설 모두 케이스로 | |

**User's choice:** tile boundary state reset (추천)
**Notes:** D-09에 캡처. Hypothesis #4 우선 + 보조로 다른 가설.

### Q3: in-memory fixture 크기 (HEIGHT 수준)는?

| Option | Description | Selected |
|--------|-------------|----------|
| tile_max_rows=2, total_rows=4 (추천) | 가장 작은 multi-tile, 1초 내 종료 | |
| MAX_SHARED_DMA_BYTES 실제 시나리오 (HEIGHT=64K+1 rows) | 65535 byte 경계 재현, 수십 초~수 분 | ✓ |
| 중간: tile_max_rows=512, total_rows=1024 | 중간 ground | |
| Programmatic: HEIGHT 파라메터라이즈 | parametrize {4, 16, 256, 1024} | |

**User's choice:** MAX_SHARED_DMA_BYTES 실제 시나리오 (HEIGHT=64K+1 rows)
**Notes:** D-09에 캡처. 실제 vendor 시나리오 충실 재현. ⚠️ Python 루프 + MockProcessor 시 실행 시간 길 수 있음 → plan-stage에서 numba 적용 / @pytest.mark.slow / parametrize 검토.

### Q4: tile-2 test의 컴퓨트 커널은?

| Option | Description | Selected |
|--------|-------------|----------|
| Identity (컴퓨트 없이 LOAD→STORE) (추천) | 순수 DMA 경로만 테스트, 100% DMA 문제 분리 | |
| ABS (실제 P7 smoke set과 동형) | vec_engine ABS 절출, 실전 시나리오 1:1 대응 | ✓ |
| Synthetic: tile_idx 대입 fingerprint | tile-distinct 입력으로 cross-tile data leak 즉시 감지 | |
| Multiple kernels parametrized (ABS + ADD_VV) | 단일 kernel 의조 제거, coverage 높음 | |

**User's choice:** ABS (실제 P7 smoke set과 동형)
**Notes:** D-09에 캡처. 실전 발생 시나리오와 1:1 대응. golden = NumPy `np.abs(input).astype(np.float16)`.

---

## Area 4: Smoke set 12개 op + GTX_DDR_REVERSED 자동화

### Q1: M≥12 smoke set은 어떻게 구성할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| vendor .elf 경로 결정, OPERAND_STAGING skip마킹 제거 (추천) | vendor .elf 자체에 ddr_init_from_file 스테이징 포함 → skip 자동 해소 | ✓ |
| ROADMAP의 명시 6개 + 6개 수동 선정 | 명시 6 + 수동 6 plan-stage에서 추가 선정 | |
| Floor M≥12만 조수, 실제는 동적 discovery | floor만 강제, 동적으로 PASS 카운트 | |
| Smoke set 결정은 plan-phase로 이잘 (CONTEXT 안 잡아멐) | discuss에서는 vendor 경로 사용 시 PASS 가능성만 | |

**User's choice:** vendor .elf 경로 결정, OPERAND_STAGING skip마킹 제거 (추천)
**Notes:** D-11에 캡처. vendor `.elf`는 자체 input staging 포함 → P7 OPERAND_STAGING_REQUIRED_VENDOR skip 사유 무효. 79 vendor `.elf` 전체가 PASS 가능성, 그 위에 floor 12 충족.

### Q2: GTX_DDR_REVERSED=1 자동화 방식은?

| Option | Description | Selected |
|--------|-------------|----------|
| test_regression_fw_full_sweep.py 인라인 set (추천) | vendor .elf 경로 발견 시만 subprocess.run env에 주입 | ✓ |
| conftest.py autouse fixture | 세션 fixture가 자동 set, finalize에서 unset | |
| @pytest.mark.vendor_be_fp16 마커 | marker 기반 conftest hook | |
| 항상 on | tests/gtx/data/firmware/ 하위 자산 = vendor .elf이면 모두 BE FP16 그렉 | |

**User's choice:** test_regression_fw_full_sweep.py 인라인 set (추천)
**Notes:** D-10에 캡처. cross-test contamination 방지. P5/P6 hand-built `.elf`는 영향 없음.

### Q3: 5x walltime gate (VTW-03) baseline 재기록 절차는?

| Option | Description | Selected |
|--------|-------------|----------|
| MTDMA-01 fix 후 HAS_NUMBA=False로 재레코딩 (추천) | Fix 이후 vendor sweep PASS → baseline_walltime.txt를 HAS_NUMBA=False로 재기록 → mean*5 ≤ baseline assert PASS | (initial pick was misclick) |
| Phase 8에서 별도 plan으로 명시 관리 | wave 순서 명시 — Wave 2 fixed-after | |
| Plan-stage에서 계측 / acceptance 도구 결정 | discuss에서는 closure 경로만 | |
| VTW-03 상프레임는 P9로 (P8 scope 축소) | P8는 MTDMA-01 fix 절 출석, VTW-03는 P9 이동 | |

**Initial pick:** "NUMBA로만 기록" (Q-Q3 클릭 보였음 — 이는 5x assert 깨뜨림)

**Clarification follow-up:** "VTW-03 baseline_walltime.txt 재기록 의도를 명확해주세요."

| Clarify Option | Description | Selected |
|----------------|-------------|----------|
| HAS_NUMBA=False로 재기록 (REQUIREMENTS 원안 유지) | numba 끄고 NumPy fallback path로 vendor sweep 실행, 그 시간이 baseline → 그 후 numba 켜고 재실행이 mean*5 ≤ baseline 충족. P7 HUMAN-UAT #2 종결 조건 그대로 | ✓ |
| NUMBA 켜고 baseline 기록 (5x assert 롤백 / 제거) | VTW-03 acceptance 수정 — "5x 빠름" 검증 안 함 | |
| 둘 다 기록 (HAS_NUMBA=False + HAS_NUMBA=True) | baseline_walltime.txt = NumPy, baseline_walltime_njit.txt = NUMBA | |
| Plan-stage에 이자 | discuss에서는 "VTW-03은 P7 HUMAN-UAT #2 closure 경로"만 | |

**User's final choice:** HAS_NUMBA=False로 재기록 (REQUIREMENTS 원안 유지)
**Notes:** D-12에 캡처. 초기 misclick 명확화 완료.

### Q4: vendor /mnt/e/14_NIGHTLY/gtx_spike/ 경로 의존은?

| Option | Description | Selected |
|--------|-------------|----------|
| GTX_VENDOR_TEST_DIR env var 기본값 + override (추천) | _find_elf에 env var 지원, 기본값 = /mnt/e/14_NIGHTLY/pyspike/test/ | ✓ |
| Hard-coded path /mnt/e/14_NIGHTLY/pyspike/test/ | env var 없이 그대로 hard-code | |
| Submodule로 vendor 자산 관리 | vendor/gtx_npu_test_assets/ 별도 submodule | |
| Phase 8에서는 dev path만, CI VTW-04 도우 | P8은 dev 머신에 집중, CI 경로는 P9/v1.2 | |

**User's choice:** GTX_VENDOR_TEST_DIR env var 기본값 + override (추천)
**Notes:** D-13에 캡처. CI 환경 대응. README에 명시.

---

## Done Check

| Option | Description | Selected |
|--------|-------------|----------|
| CONTEXT 작성 | 4개 영역 결정 충분 — 08-CONTEXT.md + 08-DISCUSSION-LOG.md 작성, commit, STATE 업데이트 | ✓ |
| 추가 탐시 gray area | mxe_accum reset, vec_engine state 등 추가 논의 | |
| 기존 영역 재검토 | Area 1~4 중 하나 다시 열기 | |

**User's choice:** CONTEXT 작성

---

## Claude's Discretion

다음 사항은 plan-stage / executor 재량으로 명시:
- 1:1 diff 결과 보관 형태 (RESEARCH.md 부록 / 인라인 / 별도 DIFF-AUDIT.md)
- tile-2 test의 numba 적용 여부, @pytest.mark.slow 처리, parametrize 사이즈
- import_vendor_golden.py --all 확장 시 기존 9-op 매핑 보존 vs 단순화
- VTW-04 README.md 정확한 섹션 구성 (4 contract 순서/깊이)
- 12 op smoke set의 12번째 op 선정 (vendor 디렉토리 카운트 후)

---

## Deferred Ideas (CONTEXT.md `<deferred>` 섹션 참조)

- multi-hart 지원, vendor 펌웨어 build chain 통합, submodule 정식화 등은
  v1.2/v2 이상 페이즈로 연기.
- `GTX_DEBUG_TILE_TRACE` env var, `_debug.py` 모듈, autouse fixture 등은
  결정 단계에서 reject.
