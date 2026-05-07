# Phase 6: Verification & Wheel - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 06-verification-wheel
**Areas discussed:** _verify base, atexit hook, regression scope, golden source, wheel asset, wave/plan structure (전체 6개 영역)

---

## Area 1: `_verify` 모듈 표면 (VRF-01)

### Q1: VRF-01 `_verify` 모듈은 어떤 base에서 출발할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 하이브리드 (Recommended) | `_verify_minimal.compare_hex` 78 LOC 코어 유지 + vendor verify.py argparse + verbose report 래퍼 ~80 LOC 추가. 코어 안정성 + CLI 호환성 양쪽. | ✓ |
| vendor verify.py 풀 직역 | 388 LOC 1:1 직역. 단점: BE bit-pair compare 로직 이중 구현. | |
| _verify_minimal 그대로 승격 | 78 LOC rename. 단점: vendor verbose report 손실. | |

**User's choice:** 하이브리드 (Recommended)
**Notes:** 사용자 default 추천 수용 — `_verify_minimal`이 P4/P5에서 검증된 자산이라는 점을 인정.

### Q2: CLI 진입점은 어떤 형태로 노출할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 둘 다 (Recommended) | `pyspike-verify` console_script + `python -m riscv.gtx._verify`. ROADMAP success #1과 #5 둘 다 만족. | ✓ |
| console_script만 | 모듈 진입점 없음. ROADMAP #1 미충족 위험. | |
| `python -m`만 | console_script 없음. ROADMAP #5 미충족. | |

**User's choice:** 둘 다 (Recommended)

### Q3: argparse 매개변수는 vendor verify.py와 어떻게 호환할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 1:1 호환 + --strict 추가 (Recommended) | vendor argparse 그대로 + `--strict` 신규 (ROADMAP #1 명시). drop-in replacement. | ✓ |
| --strict default + 단순화 | strict 기본 ON, default ULP/atol. vendor 호환성 약화. | |
| Python-idiomatic 재설계 | `--mode {strict,tolerant}`. 가장 깨끗하나 vendor 비호환. | |

**User's choice:** 1:1 호환 + --strict 추가 (Recommended)

---

## Area 2: GTX_DDR_DUMP atexit hook

### Q1: atexit hook은 언제 어디서 등록할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| import 시점 (Recommended) | `riscv.gtx.__init__.py`에서 `if os.getenv('GTX_DDR_DUMP'):` 조건부 `atexit.register(...)`. vendor `std::atexit` 직역. | ✓ |
| GtxNpu 인스턴스 생성 시 | per-instance. 다중 NPU 충돌 위험. | |
| WJOIN 핸들러 직접 | atexit 미사용, inline dump. vendor pattern 비대칭. | |

**User's choice:** import 시점 (Recommended)

### Q2: atexit dump 함수 본체는 어느 모듈에 둘까요?

| Option | Description | Selected |
|--------|-------------|----------|
| `riscv.gtx.ddr`에 (Recommended) | P3 D-09 `ddr_dump_to_file(args-only)` 옆에 `_atexit_ddr_dump()` 래퍼. env var 파싱 + args 변환. | ✓ |
| `riscv.gtx.npu`에 | GtxNpu 라이프사이클과 묶음. ddr-related 코드 분산. | |
| `riscv.gtx._verify`에 | compare와 dump가 symmetric. dump는 _verify의 자연 책임이 아님. | |

**User's choice:** `riscv.gtx.ddr`에 (Recommended)

### Q3: 환경변수 시맨틱은 vendor C++과 어떻게 맞출까요?

| Option | Description | Selected |
|--------|-------------|----------|
| vendor 1:1 (Recommended) | 3개 env vars 정확 동일 이름/시맨틱: `GTX_DDR_DUMP` (path), `GTX_DDR_DUMP_ADDR` (hex), `GTX_DDR_DUMP_SIZE` (hex). | ✓ |
| ADDR/SIZE는 default 관대 | DUMP path만 필수. P5 테스트 가정 깨짐. | |
| 단일 env var로 압축 | `GTX_DDR_DUMP=path:addr:size`. vendor 비호환. | |

**User's choice:** vendor 1:1 (Recommended)

---

## Area 3: Regression matrix scope (VRF-04)

### Q1: .elf 회귀 범위는 어디까지 들일까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 코어 op 셋 ~10-20개 (Recommended) | MM/VEC/ACT/POOL/CVT 대표. 50MB cap fit + dispatch coverage. | ✓ |
| 현재 3개만 | 최소. ROADMAP #2 의도 미달. | |
| vendor 98개 전부 | 최대 coverage. wheel size 위험 + v2 영역 op 포함. | |

**User's choice:** 코어 op 셋 ~10-20개 (Recommended)

### Q2: 인코딩 스윗은 한 스위트 모두 가져갈까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 양 스위트 둘 다 (Recommended) | gem5-simplified + ISS-full 각각 빌드. ROADMAP #2 명시. | |
| gem5-simplified만 | run_tests_n1s16.sh 단일. | |
| ISS-full만 | run_llext_tests.sh 단일. | |
| **(free-text follow-up)** vendor `n1s16_<op>.c` 1:1 직역 단일 빌드 | 사용자 질문 ("gem5 사라진 지 오래") 후 코드베이스 재확인 결과: `run_llext_tests.sh`는 vendor에 없음. ROADMAP "gem5-simplified + ISS-full" 표현은 historical artifact. 단일 .elf이 양 dispatch path 자연 mix (P5 activation_relu_gelu.elf 검증된 패턴). | ✓ |

**User's choice:** "(a)로 일단하고 나중에 다시 볼게" → vendor 1:1 직역 단일 빌드, deferred 재검토 가능
**Notes:** 사용자가 "gem5 사라진 지 오래"를 정확히 지적. 표면 용어와 실제 dispatch path 두 종류(GSPR-staged funct7=0x04-0x07 + per-op funct7)를 명확화한 후 단일 빌드로 양 path 자연 mix하기로 잠금. v1.x patch 시점에 명시 분리 재검토 가능.

### Q3: 우회적 regression test는 어떤 구조로?

| Option | Description | Selected |
|--------|-------------|----------|
| 하나의 parametrize 롤 (Recommended) | `test_regression_fw_full.py` + `@pytest.mark.parametrize('elf_path', ...)`. 실패 격리. | ✓ |
| op-family별 파일 | mm/vec/act/pool 별도. 파일 수 증가. | |
| 단일 함수에 모두 | 가장 간단. 실패 식별 불가. | |

**User's choice:** 하나의 parametrize 롤 (Recommended)

---

## Area 4: Golden hex 출처 + 포맷

### Q1: golden hex는 어디서 올 것으로 결정할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| vendor generate_n1s16_tests.py (Recommended) | vendor가 작성한 Python script로 input+ref 생성. dev 시점 실행 후 wheel lock-in. | |
| Python NPU self-golden | NPU 자체 실행 결과 lock-in. self-referential 위험. | |
| vendor libgtx_npu.so capture | C++ binary 1회 실행. 외부 의존성. | |
| **(free-text)** vendor `test/<OP>/n1s16/data/{kernel}_ref.txt` 직접 차용 | vendor가 이미 ISS-captured해 lock-in한 ref 자산을 그대로 차용. vendor C++ 신규 빌드 zero. P6 dev-stage 변환 스크립트만. | ✓ |

**User's choice:** "test/{OP}/data에 있음."
**Notes:** 사용자가 vendor 디렉토리에 이미 캡처된 ref 자산이 존재함을 직접 지적. `vendor/.../test/CONCAT/n1s16/data/n1s16_concat_result2.hex` 등 실재 확인됨. generate_n1s16_tests.py 신규 invoke 없이 자산 직접 차용으로 가장 단순화. CONTEXT.md D-10 lock-in.

### Q2: golden 파일 포맷은 어느 쪽으로 잡을까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 현 P4/P5 포맷 유지 (Recommended) | `<elf>.hex` 단일 파일. vendor `_ref.txt` → `.hex` 변환 스크립트. | ✓ |
| vendor `_ref.txt` 직접 | 변환 zero. P4/P5 자산과 형식 불일치. | |

**User's choice:** 현 P4/P5 포맷 유지 (Recommended)

### Q3: vendor C++ 빌드 의존성은 어디에 둘까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 개발 시점만 (Recommended) | dev 머신에 한 번 변환 + 자산 git lock-in. CI에는 결과만. | ✓ |
| cibuildwheel CI에 vendor 빌드 | 매 런 재현. 빌드 시간 폭증. | |
| vendor 의존 zero | self-golden만. Q1 답변과 충돌. | |

**User's choice:** 개발 시점만 (Recommended)

---

## Area 5: Wheel 자산 위치 + 사이즈 (PKG-01, PKG-04)

### Q1: .elf+golden 자산을 wheel에 어떻게 넣을까요?

| Option | Description | Selected |
|--------|-------------|----------|
| src로 이동 (Recommended) | tests/gtx/data → src/main/python/riscv/gtx/data 이동. tests 경로 업데이트 필요. | |
| src에 신설 + tests에 사본 유지 | 자산 두 곳. drift 위험. | |
| 빌드 시점 복사 | tests 단일 source-of-truth + setup.py custom build 복사. drift zero. | ✓ |

**User's choice:** 빌드 시점 복사
**Notes:** 사용자가 default 추천(이동) 대신 빌드 시점 복사 선택 — drift zero 우선. CONTEXT.md D-13 lock-in. cibuildwheel 호환성은 P6 plan-stage에서 검증 (대안: MANIFEST.in include + package-data 직접 포함).

### Q2: 번들된 자산 접근 API는 어떤 표면?

| Option | Description | Selected |
|--------|-------------|----------|
| _verify helper API (Recommended) | `bundled_elfs() -> list[Path]`, `load_golden(name) -> bytes`. 사용자 친화적. | ✓ |
| importlib.resources 그대로 | 사용자가 표준 API 직접 호출. 버전별 시그니처 변동 노출. | |

**User's choice:** _verify helper API (Recommended)

### Q3: wheel size 50MB cap 시나리오는 어떻게?

| Option | Description | Selected |
|--------|-------------|----------|
| 우선 합침 검증 (Recommended) | 측정 후 근접 시 gzip 또는 extras split. | ✓ |
| extras 분리 선공 | `spike[gtx-regression]`. 사용자 추가 의식 필요. | |
| 자산 포함만 우선 | cap soft. download 부담. | |

**User's choice:** 우선 합침 검증 (Recommended)

---

## Area 6: Wave / Plan 분할 + 의존성

### Q1: P6 총 plan 수와 wave 구조는?

| Option | Description | Selected |
|--------|-------------|----------|
| 5 plans / 3 wave (Recommended) | Wave 1a parallel 3 (VRF-01 / atexit / VRF-03) + Wave 1b 1 (VRF-04) + Wave 2 1 (PKG 통합). | ✓ |
| 6 plans / 3 wave | PKG 2 plans으로 분리. P5 동수. | |
| 7 plans / 세분화 | VRF-04를 op-family별 분할. | |

**User's choice:** 5 plans / 3 wave (Recommended)

### Q2: RED/GREEN scaffold pattern을 그대로 쓸까요?

| Option | Description | Selected |
|--------|-------------|----------|
| Wave 1a RED scaffold uniform (Recommended) | 첫 plan이 모든 P6 테스트 stub RED. P4/P5 lineage. | ✓ |
| Plan당 RED → GREEN | 각 plan 자체 완결. 일관성 약화. | |

**User's choice:** Wave 1a RED scaffold uniform (Recommended)

### Q3: Wave 1a 3 plans 병렬 실행 허용?

| Option | Description | Selected |
|--------|-------------|----------|
| 예, 병렬 (Recommended) | 편집 면적 zero overlap. P5 D-04 mirror. | ✓ |
| 아니요, 순차 | 안전하지만 시간 3배. | |

**User's choice:** 예, 병렬 (Recommended)

---

## Claude's Discretion

다음은 implementation detail로 Claude/research/plan 단계 결정:

- vendor `_ref.txt` 정확 포맷 디코딩
- `setup.py` build hook vs `MANIFEST.in` vs `package-data` 단독 — cibuildwheel 호환성 검증 후 선택
- atexit hook NPU 인스턴스 lookup 메커니즘 (WeakValueDictionary vs PythonBridge vs 단일 글로벌)
- `_verify.compare_hex` stats dict 키 명세 (vendor verbose report 호환)
- 코어 op 셋 정확 리스트 (~10-20개)
- vendor → `.hex` 변환 스크립트 정확 위치 (`scripts/import_vendor_golden.py` 등)
- Plan 03 .elf 빌드 정책 (vendor 산출물 차용 vs P6 신규 빌드)
- Plan 05 cibuildwheel test 매트릭스 verification 방법

## Deferred Ideas

- **Numba @njit 동적 최적화** → Phase 7
- **vendor 98개 op 풀 sweep** → v1.x patch 또는 v2
- **gem5-simplified vs ISS-full 별도 빌드 매트릭스** → v1.x patch ("나중에 다시 볼게")
- **vendor C++ libgtx_npu.so CI shadow run** → REQUIREMENTS.md OOS, v2
- **`_verify` Python idiomatic 재설계** → v2
- **`pyspike-verify --json` output mode** → P6 plan-stage 추가 검토
- **자산 hash check (sha256)** → v2
- **VRF-04 cross-verify against vendor live shadow** → v2
- **PCIe-EP / vfio-user / CUDA / OMP** → PROJECT.md Out of Scope
