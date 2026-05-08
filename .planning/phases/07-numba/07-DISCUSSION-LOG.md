# Phase 7: Numba Dynamic Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-08
**Phase:** 07-numba
**Areas discussed:** Scope & wheel 배포 전략 / Library 선택 + JIT 적용 범위 / Bit-exactness 보장 + Fallback / 성능 목표 + acceptance gate

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Scope & wheel 배포 전략 | REQ Out of Scope vs ROADMAP P7 충돌 + wheel 배포 전략 (hard dep / extras / lazy fallback) | ✓ |
| Library 선택 + JIT 적용 범위 | numba default 잠그고 대안 점검 + 적용 대상 식별 (gemm 핵심만 / stateless cores / engine layer) | ✓ |
| Bit-exactness 보장 + Fallback | FP32 누적 보존 + JIT/NumPy 전환 메커니즘 + per-kernel ULP 테스트 + env toggle | ✓ |
| 성능 목표 + acceptance gate | speedup 목표 + 측정 도구 + .elf strict-mode 유지 + per-op profile | ✓ |

**User's choice:** All four areas selected.

---

## Scope & wheel 배포 전략 (Area 1)

### Q1: Phase 7은 어느 마일스톤에 속합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| v1 ship gate 내부 (추가 결정) | P6 종료 시점에 numba 포함된 상태로 v1.0 릴리즈. REQ Out of Scope 조정 필요 | ✓ |
| v1.1 patch (P6 후 follow-up) | v1.0은 NumPy로만 ship, v1.1에서 P7 추가 | |
| v2 milestone으로 이동 | Phase 7을 현 마일스톤에서 제거하고 v2 backlog로 재배치 | |
| v1 내 optional extras (Recommended) | v1.0에 P7 포함하되 numba는 hard dep 아닌 `pip install spike[fast]` extras | |

**User's choice:** v1 ship gate 내부 (추가 결정). 권장 옵션(v1 내 optional extras)을 거부하고 더 강력한 v1 통합 선택.
**Notes:** ROADMAP Phase 7이 명시되어 있고 P4/P5 docstring이 P7 boundary를 미리 예고함 → v1 ship gate 통합이 자연스러움.

### Q2: numba 설치별 조건에서 P7 hot path가 어떻게 동작해야 합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| Lazy import + auto fallback (Recommended) | numba 부재 시 NumPy 메서드 유지, 사용자 투명 | ✓ |
| Hard fail when missing | numba 부재 시 ImportError raise, 명시적 `spike[fast]` 유도 | |
| Hard dep (numba 항상 설치) | `pyproject.toml dependencies`에 직접 추가 | |
| Env toggle (`GTX_NO_JIT=1`) | env var로 JIT 끄기 (단독 아닌 추가 옵션) | |

**User's choice:** Lazy import + auto fallback (Recommended).
**Notes:** zero-friction UX. base wheel 사용자가 numba 설치 안 해도 동작.

### Q3: Wheel 패키징 구조는 어떻게 갈까요?

| Option | Description | Selected |
|--------|-------------|----------|
| 단일 wheel + extras (Recommended) | `pyproject.toml [project.optional-dependencies] fast = ["numba>=0.59"]`. `pip install spike[fast]`로 선택적 설치 | ✓ |
| Optional + ENV 토글 결합 | extras + `GTX_NO_JIT` 둘 다 제공 | |
| Separate wheel (`spike-fast`) | 별도 PyPI 패키지 분리 | |
| Hard dep (단일 wheel 증량) | numba 포함 → ~150MB+ | |

**User's choice:** 단일 wheel + extras (Recommended).
**Notes:** cibuildwheel 매트릭스 zero-impact + base wheel 50MB cap 안전.

### Q4: REQUIREMENTS.md Out of Scope `numba` 항목을 어떻게 처리합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| Optional extras 명시로 조절 (Recommended) | "v1 hard dep 제외 (Phase 7의 optional `spike[fast]` extras는 허용)"으로 재문구 | ✓ |
| Out of Scope에서 제거 | 완전 제거 (NumPy 단독 의도 회석) | |
| Out of Scope 유지 + P7 v2 이동 | REQ 그대로, P7만 v2로 | |
| Defer to plan-stage | CONTEXT는 잠그고 REQ 판은 완료 시점에 | |

**User's choice:** Optional extras 명시로 조절 (Recommended).
**Notes:** ROADMAP P7 ↔ REQ 충돌의 통상 해결. wheel 배포 단순성 원칙 유지.

---

## Library 선택 + JIT 적용 범위 (Area 2)

### Q1: 어떤 동적 최적화 라이브러리를 채택합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| numba (LLVM JIT, Recommended) | ROADMAP 명시 + P4/P5 docstring 준비. extras 설치 시 ~50MB transitive | ✓ |
| Cython AOT | wheel 빌드 시 컴파일, 런타임 numba 부재 OK | |
| C extension (pybind11 추가) | PROJECT.md "C++ 추가 코드 금지" 위반 | |
| numba + Cython hybrid | 이중 파이프라인 복잡도 | |

**User's choice:** numba (LLVM JIT, Recommended).

### Q2: JIT 적용 범위는 어디까지입니까?

| Option | Description | Selected |
|--------|-------------|----------|
| Hot 3-loop만 (gemm_core.gemm_core) | 가장 작은 변경 면적 + 가장 큰 입증 | |
| Stateless cores (Recommended) | gemm_core + vec_core 7 + act_core 7+2+9 ≈ 25개 | ✓ |
| + Engine layer | mm_engine/vec_engine/act_engine까지 (numba 비호환) | |
| Profile 기반 점진 적용 | py-spy/cProfile 후 hand-pick | |

**User's choice:** Stateless cores (Recommended).
**Notes:** P4 D-01 / P5 D-01의 stateless 설계가 이미 P7 boundary로 만들어짐. 명확 boundary + 최대 ROI.

### Q3: JIT signature 전략은?

| Option | Description | Selected |
|--------|-------------|----------|
| Eager AOT-like (typed signature pre-compile, Recommended) | `@njit("f4[:,::1](...)")` 명시 서명 | |
| Lazy first-call dispatch | `@njit(cache=True)`만, 첫 호출에서 type 추론+컴파일 | ✓ |
| AOT compile to .so | `numba.pycc`, deprecated | |
| Mixed (per-kernel 옵션) | parallel=True 차도화 | |

**User's choice:** Lazy first-call dispatch.
**Notes:** 25 kernel × 다양한 dtype 변형 → signature 폭발 회피. type drift는 D-12 parity 테스트로 검출.

### Q4: JIT 컴파일 결과 캐싱 전략은?

| Option | Description | Selected |
|--------|-------------|----------|
| Disk cache (`@njit(cache=True)`, Recommended) | numba 기본 `__pycache__/.nbi`/`.nbc` 자동 | ✓ |
| Eager pre-compile at import | import 시점에 더미 inputs로 강제 컴파일 | |
| No caching | 매번 첫 호출에서 재컴파일 | |
| Hybrid (cache + eager warmup) | 디스크 캐시 + 워밍업 | |

**User's choice:** Disk cache (`@njit(cache=True)`, Recommended).

---

## Bit-exactness 보장 + Fallback (Area 3)

### Q1: JIT FP32 누적 순서를 어떻게 보장합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| fastmath=False + explicit FP32 loop (Recommended) | numba 기본 (fastmath=False, error_model='numpy'). 현 explicit for-loop 보존 | ✓ |
| fastmath=False + array reshape | JIT 내부에서 `np.dot` 사용 (BLAS drift 위험) | |
| fastmath=True (속도 우선) | bit-exact 위반 → 거부 | |
| Explicit numba decorator with type signature | `@njit("...", fastmath=False)` 명시 | |

**User's choice:** fastmath=False + explicit FP32 loop (Recommended).
**Notes:** P4 RESEARCH np.matmul drift 거부 lineage. 현 코드 그대로 유지.

### Q2: P7 acceptance gate는?

| Option | Description | Selected |
|--------|-------------|----------|
| P6 자산 100% 재통과 + per-kernel ULP-0 (Recommended) | P6 회귀 + 25 kernel parity 이중 방어막 | |
| .elf 회귀만 | 강력하지만 per-kernel divergence 미검출 | |
| ULP 결과 + benchmark 타깃 | ULP-0 + speedup 둘 다 한계 | |
| Just P6 baseline | per-kernel 부재 | |

**User's choice (Other):** "test/{OP} 103개 모두 통과. (데이터가 없는 경우 skip)"
**Notes:** 사용자가 권장보다 더 야심찬 gate를 명시 — P6 코어 op 셋 ~10-20개에서 vendor 103-op 풀 sweep으로 확장. P6 deferred ("vendor 98개 풀 sweep → v1.x/v2")가 P7으로 흡수됨. 자산 미보유 op은 graceful skip. **이는 보일러 D-10 기반.**

### Q3: NumPy fallback 경로는 어떻게 관리합니까?

| Option | Description | Selected |
|--------|-------------|----------|
| Same module, dual export (Recommended) | `gemm_core_numpy` + `gemm_core_njit` + 자동 dispatcher | ✓ |
| Separate file (`*_njit.py`) | 원본 파일 numpy, 별도 파일 njit | |
| Decorator-based hot-swap | `@gtx.maybe_njit(...)` 데코레이터 | |
| Import-time monkey-patch | `__init__.py`에서 점프 함수 교체 | |

**User's choice:** Same module, dual export (Recommended).

### Q4: 검증 시 소수 접근도 보장은 어떻게?

| Option | Description | Selected |
|--------|-------------|----------|
| Parametrize per-kernel ULP (Recommended) | `tests/gtx/test_njit_parity.py` 25-kernel ULP-0 | ✓ |
| Random fuzz over fixed seeds | 100개 random input fuzz | |
| Vendor verify_ref oracle dual run | P5 VRF-02 oracle 32개 dual run | |
| Just .elf regression sufficient | per-kernel 부재 | |

**User's choice:** Parametrize per-kernel ULP (Recommended).

---

## 성능 목표 + acceptance gate (Area 4)

### Q1: P6 baseline 대비 성능 목표는?

| Option | Description | Selected |
|--------|-------------|----------|
| Wall-clock 5× 이상 (Recommended) | 103-op vendor sweep 전체 런 기준 5× 단축 | ✓ |
| Wall-clock 10× 이상 | 더 공격적 | |
| BLAS-equivalent on gemm_core | gemm_core가 np.matmul 수준 | |
| Defer to plan-stage | P6 baseline 측정 후 결정 | |

**User's choice:** Wall-clock 5× 이상 (Recommended).

### Q2: 성능 측정 도구는?

| Option | Description | Selected |
|--------|-------------|----------|
| pytest-benchmark (Recommended) | pytest 인프라 자연 통합 | ✓ |
| 직접 timeit + 타임스탬프 | 추가 종속성 zero | |
| asv (airspeed velocity) | 전용 benchmark suite + history | |
| py-spy / cProfile (프로파일 전용) | 프로파일링용 일회성 | |

**User's choice:** pytest-benchmark (Recommended).

### Q3: Wheel 종속 설치 증가량 허용 범위는?

| Option | Description | Selected |
|--------|-------------|----------|
| Transitive 100MB까지 수용 (Recommended) | base 50MB + extras ~50-80MB | |
| Strict 50MB 이내 (extras 포함) | extras 설치 시도 50MB | |
| Base wheel만 cap, extras 무제한 | `[fast]` 특별 무제한 | |
| Defer to plan-stage | 실측 후 결정 | |

**User's choice (Other):** "wheel 종속 설치 고려하지말고 진행."
**Notes:** 사용자 명시: extras transitive size 제약 없음. base wheel 50MB cap만 유지. **D-15 직접 lineage.**

### Q4: P7 테스트 시나리오 단위는?

| Option | Description | Selected |
|--------|-------------|----------|
| 3-tier (per-kernel ULP + 103-op sweep + perf benchmark, Recommended) | parity + sweep + perf 독립 실패 격리 | ✓ |
| 2-tier (per-kernel + regression sweep) | perf benchmark 제외 | |
| 1-tier (regression sweep만) | per-kernel 부재 | |
| Tier choice in plan-stage | CONTEXT는 acceptance만 잠금 | |

**User's choice:** 3-tier (per-kernel ULP + 103-op sweep + perf benchmark, Recommended).

---

## Final Confirmation

### Q: 추가로 탐색할 결정적으로 불명한 영역이 남아 있습니까?

| Option | Description | Selected |
|--------|-------------|----------|
| I'm ready for context | 4개 영역 결정 잠금 완료. CONTEXT.md 작성 진행 | ✓ |
| Explore more gray areas | vendor 103-op 정확 식별 / mxe_accum boundary / cibuildwheel 재검증 / 손 timeit 등 | |

**User's choice:** I'm ready for context.

---

## Claude's Discretion

다음 항목은 사용자가 plan-stage 또는 Claude 판단으로 위임:

- `@njit` decorator 직접 적용 vs 재호출 패턴 (D-06/D-07 plan-stage 정확화)
- vec_core.py / act_core.py 정확 kernel 카운트 (D-06 plan-stage 정확화)
- vendor `test/<OP>/` 정확 디렉토리 카운트 (98 vs 103 — D-10 plan-stage 정확화)
- vendor 자산 → `.hex` 변환 스크립트 확장 (D-10 plan-stage)
- NumPy fallback 활성화 분기점 (module-top vs 중앙 `_jit.py` — D-02 plan-stage)
- 첫 컴파일 시간 측정 + eager warmup 트리거 (D-08 plan-stage)
- numba 정확 버전 핀 (D-03 plan-stage)
- `@njit(parallel=True)` per-kernel 적용 (D-07/D-08 plan-stage benchmark)

---

## Deferred Ideas (out of P7 scope)

- engine layer JIT 가속 → v2 (numba 비호환)
- Cython AOT / C extension / PyPy → 거부 lock-in
- CUDA / GPU acceleration → PROJECT.md Out of Scope (v2)
- fastmath=True / FP 재결합 → 영구 거부 (bit-exact)
- numba.pycc AOT compile → 거부 (deprecated)
- mxe_accum FP32 state numba 통합 → engine 영역 (v2)
- asv benchmark suite → 거부 (pytest-benchmark default)
- `@njit(parallel=True)` 적극 사용 → plan-stage hot path만 fine-tune
- PyArrow zero-copy view → v2
- `@njit(cuda=True)` → PROJECT.md Out of Scope (v2)
- Multi-process pytest-xdist sweep 병렬 → plan-stage 검토 가능

---

## Project Document Sync (P7 Plan 01 또는 첫 스텝)

- **REQUIREMENTS.md `Out of Scope`**: numba 항목 재문구 ("v1 hard dependency 제외; Phase 7 optional `spike[fast]` extras는 허용"). cython / JAX / torch / scipy의 거부는 유지.
- **PROJECT.md Constraints**: "wheel size ≤50MB" → "**base** wheel size ≤50MB. `[fast]` extras transitive size 무제한" 명시화.
- **ROADMAP.md Phase 7**: TBD goal/requirements 자리에 P7 success criteria 추가 (P7 plan 시점).
