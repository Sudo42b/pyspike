# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 1-Foundation
**Areas discussed:** DDR allocation, C++ snapshot, NumPy version pin, Memory class API surface

---

## Area Selection (multi-select)

| Option | Description | Selected |
|--------|-------------|----------|
| DDR 할당 전략 | Eager 4GB vs Lazy ensure_ddr vs page-banked | ✓ |
| C++ 스냅샷 메커니즘 | Static copy vs git submodule vs symlink | ✓ |
| NumPy 버전 핀 | `>=1.20,<2.0` (cp38 보존) vs cp38 드롭+NumPy 2.x | ✓ |
| Memory 클래스 API 표면 | Minimal byte view vs Rich abstraction vs Layered | ✓ |

**User's choice:** All four areas selected.

---

## DDR 할당 전략

### Q1: DDR 버퍼는 언제 할당합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| Lazy `ensure_ddr` | C++ 패턴 그대로, 첫 접근 시 필요 범위만 할당 — 시작 시간 짧고 메모리 절약 | ✓ |
| Eager 4GB | GtxNpu 생성 시 전체 4GB 프리알록 — 단순하지만 테스트당 4GB 소모 | |
| Page-banked | 4KB 페이지 dict로 동적 관리 — 트레이드오프지만 dict lookup 오버헤드 | |

**User's choice:** Lazy `ensure_ddr` (Recommended).
**Notes:** C++ 동등 패턴 채택. CI 안정성 + 테스트 시작 시간 개선.

### Q2: DDR 최대 크기 제한?

| Option | Description | Selected |
|--------|-------------|----------|
| 환경변수 `GTX_DDR_SIZE` | 기본 4GB, 재정의 가능 — C++ 동등 유연성 | ✓ |
| 고정 4GB | 하드코딩, 단순 | |
| DDR hex 파일 기반 동적 | 파일 크기로 자동 확장 | |

**User's choice:** 환경변수 (Recommended).
**Notes:** CI 압박 시 다운사이즈 가능 (예: `GTX_DDR_SIZE=64M`).

### Q3: GTX_DDR_REVERSED=1 처리?

| Option | Description | Selected |
|--------|-------------|----------|
| I/O 경계에서만 변환 | 내부는 항상 LE, ddr_init/ddr_dump에서만 32-byte word 역순 | ✓ |
| 저장도 reversed | DDR 메모리에 reversed로 저장, dump 빠르지만 모든 op이 reversed 고려 | |

**User's choice:** I/O 경계에서만 변환 (Recommended).
**Notes:** 내부 일관성 — 모든 op이 단일 LE 가정.

---

## C++ 스냅샷 메커니즘

### Q1: 어떻게 vendor/gtx_cpp_reference/에 넣을까?

| Option | Description | Selected |
|--------|-------------|----------|
| Static copy | 파일 복사, 외부 git 의존 없음, 수동 동기화 | |
| git submodule | 자동 동기화, repo 접근 필요, .gitmodules 관리 | ✓ |
| git subtree | 히스토리 흡수, 완전 독립 + 업스트림 경로 | |

**User's choice:** git submodule.
**Notes:** 사용자가 권장(static copy)보다 자동 업스트림 동기화를 우선시.

### Q2: 스냅샷 범위?

| Option | Description | Selected |
|--------|-------------|----------|
| gtx/ 디렉토리 전체 | 11 .cc + .h들 + .inc + verify*.py + CLAUDE.md + .ac/.mk.in/.patch | |
| .cc/.h/.inc/.py만 | 빌드 스크립트와 패치 제외, 가벼우나 ground-truth 부족 | |
| gtx/ + spike 패치 | gtx/와 함께 ../riscv-isa-sim/ patch와 주요 수정 지점도 캡처 | ✓ |

**User's choice:** gtx/ + spike 패치 포함.
**Notes:** 완전한 빌드 재현. P4/P5 차이 분석 시 C++ 빌드 가능해야 효과적.

### Q3: wheel 동봉 여부?

| Option | Description | Selected |
|--------|-------------|----------|
| 제외 | 개발자 레퍼런스 전용, MANIFEST.in/package-data exclude | ✓ |
| 동봉 | 최종 사용자가 소스 참조 가능, ~500KB 추가 | |

**User's choice:** 제외 (Recommended).
**Notes:** wheel 사이즈 절약 + 라이선스/IP 단순화.

### Q4: submodule 원격?

| Option | Description | Selected |
|--------|-------------|----------|
| 공개 GitHub/GitLab 레포 있음 | URL 제공 가능, CI 익명 clone OK | ✓ |
| 사설 레포 + deploy key/PAT | secret 설정 필요 | |
| 보류 | 일단 static copy로 시작 | |

**User's choice:** 공개 GitHub/GitLab 레포 있음.

### Q5: URL 제공?

| Option | Description | Selected |
|--------|-------------|----------|
| 지금 제공 | URL을 'Other'로 입력 | ✓ |
| Placeholder + 추후 설정 | 안전한 폴백 | |

**User's choice:** 지금 제공.
**Free-text response:** `https://github.com/Sudo42b/gtx_spike`

---

## NumPy 버전 핀

### Q1: NumPy 버전 핀?

| Option | Description | Selected |
|--------|-------------|----------|
| `>=1.20,<2.0` cp38 보존 | research 권장 — 기존 cp38-cp312 매트릭스 유지 | |
| `>=1.26` (cp39+, 1.x) | cp38 드롭 | |
| `>=2.0` (cp310+) | NumPy 2.x, FP16 시맨틱 깔끔, cp38/cp39 드롭 | ✓ |
| 제약 없이 (`>=1.20`) | 사용자 환경에 맡김 | |

**User's choice:** `>=2.0` (cp310+).
**Notes:** 사용자 결정으로 research 권장사항 번복. cp38/cp39 드롭 + cibuildwheel 매트릭스 변경 결과.

### Q2: FP16 연산 방식?

| Option | Description | Selected |
|--------|-------------|----------|
| 순수 비트 변환 + np.float32 | gtx_npu.h:89-151 그대로 포팅, NumPy 무관, 결정적 | |
| `np.float16` view 사용 | 메모리 = uint16 + .view(float16), 빠르지만 NumPy 버전 의존 | ✓ |

**User's choice:** np.float16 view 사용.
**Notes:** Pitfall 2 위험 인지 후에도 단순성 우선. P4/P5에서 strict 모드 측정 필요.

### Q3: 이 결정으로 PROJECT.md/REQUIREMENTS.md 업데이트?

| Option | Description | Selected |
|--------|-------------|----------|
| 둘 다 적용, 상위 문서 변경 | 모든 상위 문서 cp 버전/numpy/FP16 가정 동기화 | ✓ |
| NumPy만 변경, FP16은 비트 조작 | 안전한 모더니제이션 부분 채택 | |
| FP16 view만, NumPy는 cp38 보존 | 부분 채택의 다른 형태 | |
| 재고 — research 권장으로 되돌림 | 안전한 폴백 | |

**User's choice:** 둘 다 적용, 상위 문서 변경.
**Notes:** PROJECT.md / REQUIREMENTS.md / STATE.md / ROADMAP.md 모두 업데이트 필요. discuss-phase 종료 후 일괄 처리.

---

## Memory 클래스 API 표면

### Q1: Op handler 접근 인터페이스?

| Option | Description | Selected |
|--------|-------------|----------|
| Layered (raw + named accessor) | 두 계층 노출 — helper 우선, raw로 fall through | ✓ |
| Minimal byte view만 | mem.l1[n][s] uint8 view만, op이 직접 인덱싱 | |
| Rich 완전 abstraction | mem.l1_at(n,s,a).f16() 체이닝 — 표현력 풍부, 오버헤드/복잡도 | |

**User's choice:** Layered (Recommended).
**Notes:** Op handler 가독성 + edge case 유연성 모두 확보.

### Q2: SPR 접근?

| Option | Description | Selected |
|--------|-------------|----------|
| 한 개 dict + 주소 라우팅 | mem.spr[0x123], 주소 범위로 GSPR/NSPR/LSPR 자동 분기 | ✓ |
| 세 독립 dict | mem.gspr / mem.nspr[n] / mem.lspr[n][s] — 의도 명시적이지만 인덱싱 실수 위험 | |

**User's choice:** 한 개 dict (Recommended).
**Notes:** C++ unordered_map 직접 매칭. WRSPR/RDSPR 구현 단순.

### Q3: halfword view 메모리 안전?

| Option | Description | Selected |
|--------|-------------|----------|
| Always non-copying view | arr.base is not None 보장, in-place 쓰기 원본 반영 | ✓ |
| Read-only (writeable=False) | 명시적 store helper 강제, 안전하지만 op 코드 장황 | |

**User's choice:** Always non-copying view (Recommended).
**Notes:** np.copyto / arr[i]=x 양 형태 in-place 작동 — bit-exact 유지.

---

## Done check

### Q: Phase 1 의론을 마칠까요?

| Option | Description | Selected |
|--------|-------------|----------|
| I'm ready for context | CONTEXT.md 작성 + 상위 문서 일괄 업데이트 진행 | ✓ |
| Explore more gray areas | FP 헬퍼 노출 방식, 테스트 위치 등 추가 논의 | |

**User's choice:** I'm ready for context.

---

## Claude's Discretion

다음은 implementation detail로 Claude 판단 (CONTEXT.md `<decisions>` §"Claude's Discretion" 참조):
- np.float16 view 헬퍼 내부 구현
- params.py 상수 명명 규칙 (C++ 매크로 그대로 권장)
- encoding.py 포함 범위 (P1은 funct7 상수만)
- MANIFEST.in 정확한 exclude 패턴
- pyproject.toml NumPy 표기 (`numpy>=2.0,<3` 권장)
- cibuildwheel build 항목 정확한 라인

## Deferred Ideas

- FP16 비트 포팅 fallback 라이브러리 — D-09 위험 발생 시 P4/P5에서 추가
- `@isa.register("gtx")` 데코레이터 적용 — Phase 2 책임
- 상위 문서(PROJECT/REQUIREMENTS/STATE/ROADMAP) cp 버전/numpy/FP16 가정 동기화 — discuss-phase 직후 별도 커밋으로 일괄 처리
