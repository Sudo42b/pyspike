# Phase 9: Backend Migration — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 09-backend-migration-numpy-cupy
**Areas discussed:** xp alias scaffold, Migration strategy, CuPy device placement, Numba × xp compatibility

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| xp alias scaffold & GTX_USE_CUDA contract | 모듈 위치, resolve 시점, fail policy, DEVICE 심볼 처리 | ✓ |
| Migration strategy & PR shape | Wave vs Big-bang, dual-import, gate, perf budget | ✓ |
| CuPy device placement model | Scratchpads / DDR / RegisterFile 위치 + transfer API | ✓ |
| Numba × xp compatibility + tloop fusion fate | njit 28-kernel 운명 + fusion 보존 | ✓ |

---

## Area 1: xp alias scaffold & GTX_USE_CUDA contract

### Q1.1: `xp` alias를 어디에 정의?

| Option | Description | Selected |
|--------|-------------|----------|
| config_params.py에 확장 (Recommended) | 기존 DEVICE SSOT 패턴 재사용 | ✓ |
| 신규 backend.py 모듈 | 관심사 분리, 테스트 격리 | |

**User's choice:** config_params.py에 확장 → **D-01**

### Q1.2: Backend resolve 시점?

| Option | Description | Selected |
|--------|-------------|----------|
| Import 시점 eager + frozen (Recommended) | module-level alloc과 자연 일치 | ✓ |
| Lazy property | monkey-patch 가능, atexit ordering 위험 | |

**User's choice:** Import 시점 eager + frozen → **D-02**

### Q1.3: `GTX_USE_CUDA=1` + cupy 미설치?

| Option | Description | Selected |
|--------|-------------|----------|
| Fail-loud RuntimeError (Recommended) | 260518-ffr 교훈, 명시적 opt-in | ✓ |
| Silent fallback + warning | DX 부드러움, 5x regression 위험 재현 | |

**User's choice:** Fail-loud RuntimeError → **D-03**

### Q1.4: 기존 `DEVICE` 심볼 처리?

| Option | Description | Selected |
|--------|-------------|----------|
| 제거(깨끗하게) + xp로만 표현 (Recommended) | torch 완전 제거 목표, shim 금지 | ✓ |
| Backwards-compat 셈 유지 | 외부 import 보호 | |

**User's choice:** 제거 → **D-04**

---

## Area 2: Migration strategy & PR shape

### Q2.1: PR 구조?

| Option | Description | Selected |
|--------|-------------|----------|
| Wave 구조 (Recommended) | Wave 0/1/2/3 + 각 wave 끝 byte-exact gate | ✓ |
| Big-bang 단일 PR | 13 src + 3 test 동시, rollback 쉬움 | |
| Hybrid: 1 PR + atomic commits | intra-PR wave + CI는 통합 | |

**User's choice:** Wave 구조 → **D-05**

### Q2.2: 중간 dual-import 허용?

| Option | Description | Selected |
|--------|-------------|----------|
| 허용되지만 최소화 (Recommended) | numpy.ndarray ↔ torch.from_numpy() 한정 | ✓ |
| Hard cut — 파일당 하나만 | 더 깔끔, wave 경계 복잡 | |

**User's choice:** 허용되지만 최소화 → **D-06**

### Q2.3: Wave gate?

| Option | Description | Selected |
|--------|-------------|----------|
| 명시 6 op + tile-2 unit test (Recommended) | ABS+GELU+RELU+SIGMOID+TANH+SOFTMAX+tile2 | ✓ |
| ABS 단일 (최소) | 빠름, 회귀 감지 늦음 | |
| 전체 84-op vendor sweep | 최대 안전, ~10분+ 사이클 | |

**User's choice:** 명시 6 op + tile-2 → **D-07**

### Q2.4: Perf budget?

| Option | Description | Selected |
|--------|-------------|----------|
| ±10% (85-105s) | 타이트, torch overhead 제거 기대 | ✓ |
| +50% OK (≤140s) (Recommended) | 1차 correctness-first | |
| 추조 baseline만 기록 | gate 없음 | |

**User's choice:** ±10% (85-105s) → **D-08**

---

## Area 3: CuPy device placement model

### Q3.1: Scratchpads (L0/L1/L2) 위치?

| Option | Description | Selected |
|--------|-------------|----------|
| GPU (cupy.ndarray) (Recommended) | 현재 torch 패턴 유지 | ✓ |
| CPU 고정 (numpy) | cupy benefit 사라짐 | |
| Plan-stage 재량 | 결정 미루기 | |

**User's choice:** GPU → **D-09**

### Q3.2: DDR 위치?

| Option | Description | Selected |
|--------|-------------|----------|
| CPU 고정 유지 (Recommended) | 현재 DMA boundary 모델 유지 | |
| GPU로 이동 (전체 GPU 통일) | 4 GiB VRAM 점유 위험 | ✓ |

**User's choice:** GPU로 이동 → **D-10**
**Notes:** Plan-stage가 GPU VRAM budget, ddr_dump_to_file asnumpy 전환, doubling-grow GPU 동작 검증 필수.

### Q3.3: RegisterFile (SPR int64) 위치?

| Option | Description | Selected |
|--------|-------------|----------|
| CPU 고정 (Recommended) | scalar dispatch overhead 회피 | |
| scratchpads와 같은 device | xp 결정 자동 따름, 5x regression 위험 | ✓ |

**User's choice:** scratchpads와 같은 device → **D-11**
**Notes:** Wave 1 끝에 ABS perf 측정으로 검증. 위반 시 RegisterFile만 host-pinned 예외.

### Q3.4: Cross-device transfer API?

| Option | Description | Selected |
|--------|-------------|----------|
| 헬퍼 두 개: to_host + to_device (Recommended) | xp=numpy면 no-op, 명시적 boundary | ✓ |
| Raw API 직접 호출 | 더 명시적이지만 읽기 어려움 | |

**User's choice:** 헬퍼 두 개 → **D-12**

---

## Area 4: Numba × xp compatibility + tloop fusion fate

### Q4.1: 28개 @njit kernel cupy 경로 처리?

| Option | Description | Selected |
|--------|-------------|----------|
| cupy 모드에서 numba 비활성화 + raw cupy 호출 (Recommended) | cupy native vectorized | |
| `.get()`으로 host bouncing + njit | 전송 overhead 손해 | |
| Plan-stage 결정 | 커널 별 거동 다르게 | |
| **자유 입력**: "from numba import cuda; @cuda.jit 사용" | numba.cuda.jit dual-impl | ✓ |

**User's choice (free text):** numba.cuda backend로 28개 dual-impl → **D-13**

### Q4.1-follow-up: Cuda kernel scope?

| Option | Description | Selected |
|--------|-------------|----------|
| Hot-path 몇 개만 (Recommended) | 5-7 kernel + 나머지 cupy native | |
| 28개 전부 dual-impl | 최대 일관성, scope 위험 | ✓ |
| Plan-stage 의결 | 커버 범위 결정 미루기 | |

**User's choice:** 28개 전부 → **D-13 (scope 확장)**
**Notes:** Plan-stage가 1주 추정 후 옵션 A/B/C 제시 (P10 분할 가능성 포함).

### Q4.1-follow-up: kernel 소스 공유?

| Option | Description | Selected |
|--------|-------------|----------|
| 모듈 분리: _njit_kernels + _cuda_kernels (Recommended) | 별도 관리 | |
| Universal source + jit fallback (numba.guvectorize) | 같은 소스, target 스위치 | ✓ |

**User's choice:** Universal source → **D-14**
**Notes:** Plan-stage가 guvectorize-convertible audit 필수 (nested loop / state mutation 패턴은 예외 가능).

### Q4.2: tloop_buffer.py `_execute_fused` 전략?

| Option | Description | Selected |
|--------|-------------|----------|
| 1:1 drop-in (torch.abs→np.abs 등) (Recommended) | fusion benefit 유지 | ✓ |
| Fusion 비활성화 (replay only) | 더 안전, perf 손실 | |
| numba njit fused kernel로 자동 승격 | 광범위 재설계 | |

**User's choice:** 1:1 drop-in → **D-15**

### Q4.3: 테스트 포팅 범위?

| Option | Description | Selected |
|--------|-------------|----------|
| tests/gtx/ 전체 포팅 (Recommended) | 3 파일 + conftest 일관성 | ✓ |
| Source만 포팅, test torch 유지 | dual-import 장기, shim 위험 | |

**User's choice:** tests/gtx/ 전체 포팅 → **D-16**

### Q4.4: pyproject.toml deps 전략?

| Option | Description | Selected |
|--------|-------------|----------|
| torch 완전 제거 + [cuda] extras (Recommended) | ROADMAP success #5 일치 | ✓ |
| torch를 dev/test extras로 잔존 | CI bridge, design intent 약화 | |

**User's choice:** torch 완전 제거 + [cuda] extras → **D-17**

---

## Claude's Discretion

- `to_host()` / `to_device()` 헬퍼 정확한 시그니처 (dtype 보존, view vs copy)
- guvectorize-convertible audit 형식 (markdown table vs RESEARCH 부록)
- Wave 0 backend fixture 위치 (conftest.py vs 별도 helper)
- `[cuda-jit]` extras 분리 여부 (numba를 base dep 유지)
- cuda kernel 단위 테스트의 mock vs real GPU 정책
- 28-kernel scope 옵션 A/B/C 중 plan-stage가 1주 추정 후 사용자 컨펌

## Deferred Ideas

- CUDA kernel 성능 최적화 (shared memory, warp shuffle, RawKernel) — v1.2 perf phase
- pybind11 C++ 측 torch::Tensor 제거 — 별도 phase
- Wheel multi-arch (cp313+) — v1.2
- CuPy memory pool tuning — D-10 검증 후 결정
- P10 신설: 28-kernel dual-impl + cuda smoke test가 v1.1 milestone 초과 시
