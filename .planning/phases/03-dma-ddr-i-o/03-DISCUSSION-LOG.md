# Phase 3: DMA & DDR I/O - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05
**Phase:** 03-dma-ddr-i-o
**Mode:** discuss (interactive)
**Areas discussed:** DMA module organization, deferred-store queue, DDR hex I/O placement, .elf regression strategy, ensure_ddr upgrade, 4-mode dispatch routing

---

## Area 1: DMA 모듈 구성

### Q1.1 — DMA 핸들러 파일 구조

| Option | Description | Selected |
|--------|-------------|----------|
| A. 단일 `ops/dma.py` (C++ 직역) | 800+ LOC 단일 파일, P2 ops/control.py 7.8K 선례 | |
| B. 기능별 분할 | dma_firmware.py / dma_xform.py / dma_iss.py 분산 | |
| C. 단일 + 헬퍼 분리 (추천) | ops/dma.py(@handler 진입점) + dma_engine.py(순수 함수) | ✓ |

**User's choice:** C
**Rationale:** spike 의존 0의 dma_engine.py가 단위 테스트 + 회귀 디버깅 용이성 확보

### Q1.2 — funct7=0x40/0x41 핸들러 분리

| Option | Description | Selected |
|--------|-------------|----------|
| A. 별도 두 핸들러 (추천) | `@handler(funct7=0x40)`/`(funct7=0x41)` 분리 | ✓ |
| B. 단일 dispatch + 내부 funct7 분기 | one wrapper handles both | |

**User's choice:** 별도

### Q1.3 — funct3 sub-variant (LOAD/STORE/COPY) 처리

| Option | Description | Selected |
|--------|-------------|----------|
| A. 단일 핸들러 내부 분기 (C++ 패턴, 추천) | funct3에 따라 if-else | |
| B. 데코레이터로 funct3까지 분해 | mask_funct3=True 활성화, 함수 3개 | ✓ |

**User's choice:** 데코레이터
**Notes:** `_registry.py`의 `mask_funct3=True` 경로 사용. `npu.custom0` 2-level dispatch 확장 필요 (plan에서 정확화).

---

## Area 2: deferred-store 큐 위치/형태

### Q2.1 — 데이터 클래스 형태

| Option | Description | Selected |
|--------|-------------|----------|
| A. `@dataclass DeferredDdrStore` (추천) | P2 WarpState 패턴 일치, mypy/pylint 친화 | ✓ |
| B. `NamedTuple` | frozen, hash 가능, mutation 불가 | |
| C. plain dict / tuple | 가벼움, type safety 손해 | |

**User's choice:** A

### Q2.2 — 큐 보유 위치

| Option | Description | Selected |
|--------|-------------|----------|
| A. `GtxNpu.deferred_ddr_stores: list` (추천) | P2 self.warp 위치 패턴 일치 | ✓ |
| B. `GtxMemory.deferred_ddr_stores: list` | 메모리 상태와 함께 | |
| C. 별도 `GtxDmaQueue` 클래스 | 캡슐화 ↑, 분리 비용 > 이득 | |

**User's choice:** A

### Q2.3 — flush API 시그니처

| Option | Description | Selected |
|--------|-------------|----------|
| A. `npu.flush_deferred_ddr_stores() -> None` (추천) | C++ 직역 | ✓ |
| B. `GtxMemory.flush_deferred_ddr_stores(npu)` static | 메모리 모듈 응집도 | |

**User's choice:** A
**Notes:** 호출 트리거 위치(end_p / end_s / credit_st_chk)는 research가 잠금. ROADMAP P3 success #4는 end_p 명시. C++ 실제 호출 3곳 발견 (`gtx_npu_loop.cc:53`, `gtx_npu_dispatch.cc:902`, `:784`).

---

## Area 3: DDR hex I/O 배치

### Q3.1 — DDR I/O 모듈 위치

| Option | Description | Selected |
|--------|-------------|----------|
| A. `ddr.py`에 채움 (P1 의도, 추천) | P1 D-13 layout 준수, 순수 함수 | ✓ |
| B. `ops/dma.py`로 통합 (C++ 직역) | DMA op과 같은 파일 | |
| C. `ddr_io.py` 신설 + `ddr.py`는 alloc만 | 모듈 분리 | |

**User's choice:** A

### Q3.2 — `GTX_DDR_REVERSED` 환경변수 read 정책

| Option | Description | Selected |
|--------|-------------|----------|
| A. 매 호출 시 read (추천) | monkeypatch 친화, P2 D-07 GTX_NO_EXIT 패턴 | ✓ |
| B. 모듈 로드 시 1회 cache | 성능, 변경 무시 | |

**User's choice:** 매호출

### Q3.3 — `GTX_DDR_DUMP_*` 환경변수 처리 위치

| Option | Description | Selected |
|--------|-------------|----------|
| A. 함수 인자만 받음, 환경변수 처리는 외부 | 라이브러리 깨끗 | |
| B. 인자 None 시 환경변수 fallback (C++ 직역, 추천) | behavior parity | |
| C. CLI 진입점에서만 환경변수 처리 | 라이브러리는 깨끗 + CLI가 책임 | ✓ |

**User's choice:** C
**Notes:** `riscv/gtx/__main__.py` vs 별도 dump-hook 모듈 vs P6 `pyspike-verify` console script — 위치 plan에서 결정. 자동 DDR dump (spike 종료 시 GTX_DDR_DUMP 환경변수 처리)는 P6 또는 별도 follow-up으로 미룸.

---

## Area 4: .elf 회귀 테스트 전략

### Q4.1 — P3 .elf 픽스처 범위

| Option | Description | Selected |
|--------|-------------|----------|
| A. Python-only programmatic만 (추천) | ROADMAP success 정합, .elf 인프라 비용 ↓ | ✓ |
| B. Python + 최소 dma_roundtrip.elf 1개 | smoke test, P4 .elf 인프라 사전 검증 | |
| C. .elf는 P4부터 (= A와 동일) | | |

**User's choice:** A
**Notes:** ROADMAP P3 success #1~#5 모두 programmatic. 첫 .elf strict-mode는 P4 success #4 (mm_basic.elf)에서 첫 등장.

### Q4.2 — Deferred-store success #4 assertion 형태

| Option | Description | Selected |
|--------|-------------|----------|
| A. dataclass 속성 직접 비교 | 큐 모양만 검증 | |
| B. 스냅샷 비교 | flush 동작만 검증 | |
| C. 둘 다 (추천) | 큐 push + flush 별도 검증, 진단성 ↑ | ✓ |

**User's choice:** C

### Q4.3 — Mock MMU 확장

| Option | Description | Selected |
|--------|-------------|----------|
| A. P2 D-19 약속 이행 — P3에서 추가 | P4/P5 사전 작업 | |
| B. P4 MM까지 미루기 (추천) | YAGNI, P3는 GPR mock만 | ✓ |

**User's choice:** B
**Notes:** P3 firmware_dma는 GPR만 사용 (`proc.get_state().XPR[insn.rs1]`). MMU `load_uint64/store_uint64`는 P4 firmware_mm_op이 실제 호출 시점에 추가.

---

## 보조 Area: ensure_ddr 업그레이드 + Mode 1/3 dispatch

### Q5.1 — `ensure_ddr` 업그레이드 전략

| Option | Description | Selected |
|--------|-------------|----------|
| A. Doubling-grow로 교체 (추천) | C++ parity, alloc 횟수 ↓ | ✓ |
| B. Stub 유지 | 성능 미미, GTX_DDR_SIZE=64M로 감당 | |

**User's choice:** A
**Notes:** Phase 1 D-01에 "Phase 3 will replace this with the C++ doubling-grow strategy"로 미리 약속됨.

### Q5.2 — Mode 1/3 4-mode dispatch 라우팅 위치

| Option | Description | Selected |
|--------|-------------|----------|
| A. `riscv/gtx/dispatch.py`에 `dispatch_4mode` 함수 (추천) | 모든 op 공유, dispatch.py 자연 위치 | ✓ |
| B. `npu.py:GtxNpu._dispatch()` 메서드 | 캡슐화 | |
| C. `ops/dma.py` 내부 분기 | DMA 외 op도 사용하므로 부적합 | |

**User's choice:** A
**Notes:** `firmware_dma`/`firmware_mm`(P4)/`firmware_vec`(P5) 모두 호출. ROADMAP P3 success #5 정합.

---

## Claude's Discretion

다음은 implementation detail로 plan/research 단계에서 정확화:

- `_registry.py` 2-level dispatch 자료 구조 (`dict[int, dict[int, Callable]]` vs `dict[(int, int), Callable]`)
- `npu.custom0` funct3 sub-decomposition 없는 funct7 처리 (sentinel `funct3=None` vs 별도 dict)
- `dma_engine.py` 모듈명 (`dma_engine.py` vs `dma_kernels.py` vs `dma_helpers.py`)
- `DeferredDdrStore` 정의 위치 (`dma_engine.py` vs `dma_state.py`)
- `dispatch_4mode` 인자 시그니처 정확한 형태
- P3 disasm 항목 정확한 목록 (9 active + mcast 4 stub-or-defer)
- `monkeypatch.setenv` fixture vs `os.environ`/`unittest.mock.patch.dict` 스타일
- `ensure_ddr` doubling-grow `INITIAL_FLOOR` 값 (1MB / 64KB / 32B 단위)
- DDR hex parser 구현 디테일 (line iteration vs mmap, half-density packing)
- `firmware_dma_op` rs1/rs2 인코딩의 inverse helper 제공 여부 (테스트 친화)

## Deferred Ideas

- DMA-3D / IM2COL-N/D / MCAST 본격 구현 → v2 (DMA-V2-01)
- `mexec` full microcode 페치-디코드 → v1 펌웨어 미요구 시 stub (P5+)
- 자동 DDR dump (spike 종료 시 GTX_DDR_DUMP 처리) → P6 또는 별도 follow-up
- MMU mock 확장 → P4 plan
- 첫 .elf strict-mode 회귀 → P4 success #4 (mm_basic.elf)
- `verify.py` 포팅 → P6 VRF-01
