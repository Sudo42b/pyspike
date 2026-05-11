# GtxNpu FSM 실행 순서

**Source of truth**: `src/main/python/riscv/context_map.yaml`

GtxNpu는 **두 개의 직교 FSM**으로 구성:

- **Context FSM** — NPU 전역, instruction 사이에 persistent. Warp 마커로 transition.
- **Instruction FSM** — `custom0`/`custom1` 호출 1회당. 매번 IDLE → … → IDLE 한 사이클.

DISPATCH 단계에서 두 FSM이 만남: "현재 context에서 이 instruction이 valid한가" 체크.

---

## 1. Context FSM (NPU 전역, persistent)

### 상태

| State | 정의 (context_map.yaml) |
|---|---|
| **C1** `PLAN_OUTSIDE` | plan outside — before `START_P` (또는 `END_P` 후 복귀) |
| **C4** `PLAN_INSIDE` | plan inside, shared/thread outside — inside `START_P`, outside `S/T` |
| **C2** `SHARED` | plan inside + shared inside — inside `START_P + START_S` |
| **C3** `THREAD` | plan inside + thread inside — inside `START_P + START_T` |

초기 상태: **C1** (NPU reset 직후).

### 전이 (warp 마커 트리거)

```
                       ┌───────────────────────┐
                       │                       │
                       v                       │
                ┌────────────┐                 │
       ┌────────│  C1        │<────END_P───────┤
       │        │ PLAN_OUTSIDE                 │
       │        └────────────┘                 │
       │             │                         │
       │       START_P                         │
       │             v                         │
       │        ┌────────────┐                 │
       │  ┌─────│  C4        │─────────────────┘
       │  │     │ PLAN_INSIDE
       │  │     └────────────┘
       │  │      ↑    ↑   ↑
       │  │      │    │   │
       │  │   END_S END_T │
       │  │      │    │   │
       │  │      │    │   │
       │ START_S │ START_T│
       │  │      │    │   │
       │  v      │    v   │
       │ ┌──────────┐  ┌──────────┐
       │ │   C2     │  │   C3     │
       │ │ SHARED   │  │ THREAD   │
       │ └──────────┘  └──────────┘
       │
       (초기 상태)
```

| 마커 | From → To |
|---|---|
| `GTX_WARP_START_P` | C1 → C4 |
| `GTX_WARP_END_P` | C4 → C1 |
| `GTX_WARP_START_S` | C4 → C2 |
| `GTX_WARP_END_S` | C2 → C4 |
| `GTX_WARP_START_T` | C4 → C3 |
| `GTX_WARP_END_T` | C3 → C4 |
| `GTX_WARP_SPLIT` / `GTX_WARP_JOIN` | 변경 없음 (구조적 마커) |

### Context별 valid 그룹 (요약)

| Context | Valid 그룹 (context_map.yaml `contexts.{Cx}.valid_groups`) |
|---|---|
| **C1** | `type_a`, `c1_only`, `tpose_fill`, `shared_mcast`, `all_context` |
| **C2** | `dma`, `tpose_fill`, `shared_mcast`, `credit_ld`, `all_context`, `credit_ld_chk` |
| **C3** | `type_a`, `dma`, `credit_ld`, `all_context`, `credit_ld_chk` |
| **C4** | `type_a`, `tpose_fill`, `shared_mcast`, `all_context` |

`type_a`(95개)는 C1·C3·C4에서, `dma`(7개)는 C2·C3에서만 valid 등. 자세한 매핑은 YAML 참고.

---

## 2. Instruction FSM (per `custom0`/`custom1` 호출)

한 instruction이 들어올 때마다 아래 순서로 1 사이클.

```
[1] IDLE              호출 전/후 대기
       ↓
[2] DECODE            funct7 = insn.funct
                      funct3 = (xd<<2) | (xs1<<1) | xs2
                      → mnemonic 결정 (registry에서 mnemonic lookup)
       ↓
[3] DISPATCH          (a) handler lookup:
                          custom0: 2-level (sub_table[None] → sub_table[funct3])
                          custom1: flat self._custom1[funct3]
                      (b) context validity 검사:
                          group  ← context_map.yaml의 instruction → group 매핑
                          valid? ← contexts[npu._context].valid_groups에 group ∈
                          invalid → 정책에 따라 NOP 또는 illegal raise
                      (c) warp marker 인 경우:
                          context transition 트리거 (아래 WRITEBACK 단계에서 적용)
       ↓
[4] EXECUTE           handler(proc, insn, xs1, xs2) 호출
                      ctx["rd"] ← 반환값
                      warp marker는 보통 no-op (context 변경만 한다면)
       ↓
[5] WRITEBACK         (a) context FSM 업데이트 (warp marker였으면)
                      (b) OPSET staging 정리:
                          custom0 ∧ funct7 ≠ 0x4A 일 때
                              gspr[0x003] = 0   (OPERAND3)
                              gspr[0x005] = 0   (OPERAND4)
                          OPSET(0x4A) 자체는 staging 유지 (vendor parity)
       ↓
[1] IDLE              ctx["rd"] 리턴
```

---

## 3. 두 FSM의 상호작용

```
   Instruction 진입
        │
        v
   DECODE  ─────────────────────────────────────────┐
        │                                           │
        │ funct7/funct3                             │
        v                                           │
   DISPATCH  ──── ① handler lookup                  │
        │   ──── ② context validity 검사 ◄──────  npu._context (current)
        │   ──── ③ warp marker 식별                 │
        │                                           │
        v                                           │
   EXECUTE   ──── handler(...) 호출                 │
        │                                           │
        v                                           │
   WRITEBACK ──── ④ context transition 적용 ─────►  npu._context (next)
        │   ──── ⑤ OPSET staging clear              │
        v                                           │
   IDLE                                             │
                                                    │
   (다음 instruction이 들어오면 다시 DECODE) ───────┘
```

핵심: `npu._context` 필드 1개가 instruction 경계를 넘어 살아남고, DISPATCH에서 읽고 WRITEBACK에서 갱신.

---

## 4. 파일·메서드 매핑 (현재 코드 기준 — 모든 정리 완료)

### `npu.py` — Instruction FSM (5 states)

| # | State | 메서드 | 라인 |
|---|---|---|---|
| 1 | `IDLE` | (대기, 메서드 없음) | — |
| 2 | `DECODE` | `_state_decode` | 252 |
| 3 | `DISPATCH` | `_state_dispatch` (context-aware 3/2-level lookup) | 259 |
| 4 | `EXECUTE` | `_state_execute` (handler 호출) | 307 |
| 5 | `WRITEBACK` | `_state_writeback` (OPSET clear + context transition) | 323 |

### 보조 메서드 / 필드

| 항목 | 위치 | 역할 |
|---|---|---|
| `class _NpuState(Enum)` | npu.py:28 | 5-state enum |
| `self._state` | `__init__` / `reset` | 현재 FSM state |
| `self._ctx: dict` | `__init__` / `reset` | per-instruction transient data |
| `self._context: NpuContext` | `__init__` / `reset` | NPU 전역 context (C1 초기) |
| `_run_pipeline` | npu.py | ctx 초기화 + state machine loop |
| `_step` | npu.py | state → 메서드 dispatch |
| `custom0` / `custom1` | npu.py:162 / 207 | RoCC 엔트리 → `_run_pipeline` 호출 |

### `npu_context.py` — Context FSM (모듈 레벨)

| 항목 | 역할 |
|---|---|
| `class NpuContext(Enum)` | `C1`, `C2`, `C3`, `C4` |
| `INITIAL_CONTEXT = NpuContext.C1` | reset 시 진입할 초기 context |
| `WARP_TRANSITIONS` | `{warp_marker_mnemonic: (from, to)}` |
| `WARP_MARKERS_NO_TRANSITION` | `{GTX_WARP_SPLIT, GTX_WARP_JOIN}` |
| `GROUPS` | 9 그룹 (`type_a`, `dma`, ...) → instruction tuple |
| `_CONTEXT_VALID_GROUPS` | `{context: frozenset(group_names)}` |
| `EXCLUDED_FROM_CONTEXT` | warp markers + control/sync (context 검사 면제) |
| `get_group(mnemonic)` | mnemonic → group name |
| `is_valid_in_context(mn, ctx)` | bool |
| `is_warp_marker(mn)` | bool |
| `apply_transition(ctx, mn)` | next context (or unchanged) |
| `is_legal_transition(ctx, mn)` | strict check |

### `_registry.py` — handler 데코레이터 확장

| 항목 | 변경 |
|---|---|
| `handler(...)` 시그니처 | `context: NpuContext \| Iterable[NpuContext] \| None = None` 파라미터 추가 |
| 등록 시 mnemonic 부착 | `fn.gtx_mnemonic = mnemonic` (dispatch에서 추출용) |
| `collect_for_kind("custom0")` | 3-level: `{funct7: {context: {funct3-or-None: fn}}}` |
| `collect_for_kind("custom1")` | 2-level: `{funct3: {context: fn}}` |
| `collect_disasms` | `(kind, funct7, funct3, mnemonic)`로 dedupe (context 다중 등록 무시) |

### `dispatch.py` — 테이블 빌더 + 바인딩

| 항목 | 변경 |
|---|---|
| `build_custom0_table` | 3-level 구조로 closure bind |
| `build_custom1_table` | 2-level 구조로 closure bind |
| `_bind` | `wrapped.gtx_mnemonic = getattr(fn, "gtx_mnemonic", None)` 전파 |

---

## 5. ctx 키 라이프사이클 (per-instruction)

| 키 | 설정되는 단계 | 사용되는 단계 |
|---|---|---|
| `kind` ("custom0"/"custom1") | `_run_pipeline` 진입 | DISPATCH, WRITEBACK |
| `proc` | `_run_pipeline` 진입 | EXECUTE |
| `insn` | `_run_pipeline` 진입 | DECODE, EXECUTE |
| `xs1`, `xs2` | `_run_pipeline` 진입 | EXECUTE |
| `funct7` | DECODE | DISPATCH, WRITEBACK |
| `funct3` | DECODE | DISPATCH |
| `mnemonic` (추가 권장) | DECODE | DISPATCH (group lookup) |
| `group` (추가 권장) | DISPATCH | (디버깅용) |
| `is_warp_marker` (추가 권장) | DISPATCH | WRITEBACK |
| `context_transition` (추가 권장) | DISPATCH | WRITEBACK |
| `handler` | DISPATCH | EXECUTE |
| `rd` | EXECUTE | `_run_pipeline` 리턴 |

---

## 6. Vendor parity invariant (변경 금지)

1. `funct7 = insn.funct`
2. `funct3 = (insn.xd << 2) | (insn.xs1 << 1) | insn.xs2`
3. **custom0** 2-level dispatch:
   - 먼저 `sub_table[None]` (P2 back-compat)
   - 없으면 `sub_table[funct3]` (P3+ mask_funct3=True 경로)
4. **custom1** flat dispatch: `self._custom1[funct3]`, miss=0
5. handler 호출 인자: `(proc, insn, xs1, xs2)` 정확히 4개
6. **OPSET post-clear**: custom0 ∧ funct7 ≠ 0x4A 일 때만
   - `gspr[0x003] = 0`
   - `gspr[0x005] = 0`
   - OPSET(funct7=0x4A) 자체는 staging 유지
7. Handler miss → `rd = 0` (silent NOP)
8. **NEW (context_map 도입 후)**: context invalid → 정책 결정 필요
   - 옵션 A: silent NOP (현재 vendor 동작과 일치하는지 회귀로 확인)
   - 옵션 B: 경고 + NOP (디버깅 친화적)
   - 옵션 C: illegal instruction trap (엄격, ISS와 매칭 가능)
   - **결정 보류** — vendor `nsu.cpp` 동작 확인 후 결정

---

## 7. 작업 진행 상태

| 단계 | 항목 | 상태 |
|---|---|---|
| 1 | Context FSM 인프라 (`npu_context.py`) | ✅ 완료 |
| 2 | Registry 확장 (`_registry.py`에 `context=` 파라미터) | ✅ 완료 |
| 3 | Dispatch table 3/2-level 빌더 (`dispatch.py`) | ✅ 완료 |
| 4 | npu.py FSM 5-state로 정리 (EXEC_* 4-sub-state 제거) | ✅ 완료 |
| 5 | `self._context` 필드 + `_state_dispatch` 컨텍스트 lookup + `_state_writeback` 전이 | ✅ 완료 |
| 6 | 12개 multi-context 명령어 핸들러 분기 (Style C) | 🟡 사용자 작업 |
| 7 | Warp marker 핸들러와 FSM context transition 간 single-source-of-truth 정리 | 🟡 사용자 작업 |
| 8 | Vendor regression 검증 (`libcustomext.so` 복구 후) | ⛔ 빌드 의존 |
| 9 | `gtx/` 다른 파일들의 sweep damage 정리 | 🟡 사용자 작업 |

### 남은 사용자 작업 (단계 6)

12개 multi-context 명령어를 패턴 4로 분기 (Style C):

```python
from .npu_context import NpuContext

@handler(funct7=GTX_LOAD_F7, context=NpuContext.C2, mnemonic="GTX_LOAD")
def gtx_load_c2(npu, proc, insn, xs1, xs2):
    # DDR → L2SPM
    ...

@handler(funct7=GTX_LOAD_F7, context=NpuContext.C3, mnemonic="GTX_LOAD")
def gtx_load_c3(npu, proc, insn, xs1, xs2):
    # L2SPM → L1SPM
    ...
```

대상 명령어 (`context_map.yaml` notes 기준):
`GTX_LOAD`, `GTX_STORE`, `GTX_COPY`, `GTX_TPOSE`, `GTX_FILL`,
`GTX_MCAST_S2L`, `GTX_MCAST_S2S`, `GTX_RDSPR`, `GTX_WRSPR`,
`GTX_CREDIT_LD`, `GTX_CREDIT_ST`, `GTX_CREDIT_LD_CHK`, `GTX_CREDIT_ST_CHK`

---

## 8. 검증 스모크 (정정 후)

```bash
# Context FSM
.venv/bin/python -c "
from riscv.gtx.npu import GtxNpu, _NpuContext, _NpuState
n = GtxNpu()
n.reset(None)  # MockProcessor 필요할 수 있음
assert n._context == _NpuContext.C1
print('context init OK')
print('contexts:', [c.name for c in _NpuContext])
print('states:',  [s.name for s in _NpuState])
"

# Vendor regression (build 복구 후)
PATH=$PWD/.venv/bin:$PATH GTX_VENDOR_TEST_DIR=$PWD/test/ \
  .venv/bin/python -m pytest \
  'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[RELU]' \
  --no-cov --timeout=180 -v
```

---

## 9. 디버깅 진입점

| 증상 | 진입점 |
|---|---|
| context transition 누락 | `_state_writeback` — warp marker 식별 + 전이 로그 |
| 잘못된 context에서 instruction 실행 | `_state_dispatch` — `(mnemonic, group, ctx, valid?)` 로그 |
| handler lookup miss | `_state_dispatch` — `(kind, f7, f3, sub_table)` 로그 |
| OPSET staging 누수 | `_state_writeback` clear 전/후 `gspr[0x003], gspr[0x005]` |
| handler 인자 의심 | `_state_execute` — `(proc, insn.funct, xs1, xs2)` |

---

## 10. 참고

- `src/main/python/riscv/context_map.yaml` — 9 그룹 + 4 context, 132 instruction 매핑
- `gtx_doxygen/src/intrinsics/1.1.4.1/mainpage.md` — 모드별 사용 가능 인트린식 (cross-ref)
- `gtx-risc-vp/vp/src/platform/gtx/nsu.cpp` — ISS dispatch (vendor 동작 검증용)
- `src/main/python/riscv/gtx/ops/control.py` — `begin_p`/`end_p` 등 warp 마커 핸들러 (현재 위치)
