---
quick_id: 260515-0ro
date: 2026-05-15
status: complete
test_gate: 26/26 PASS
commits:
  - b464bb4 (Step 2 — single-source SPR addresses via csr, no aliases)
files_changed:
  - src/main/python/riscv/gtx/unit/csr/gspr.py (OPERAND4 → OPCODE @ 0x004)
  - src/main/python/riscv/gtx/unit/ins/encoding.py (re-export block 제거)
  - src/main/python/riscv/gtx/npu.py (_GSPR_OP3/_OP5 alias 제거)
  - src/main/python/riscv/gtx/tloop_buffer.py (alias 제거)
  - src/main/python/riscv/gtx/unit/context/dma.py (alias 제거)
  - src/main/python/riscv/gtx/unit/ins/ops/mm.py (14 KeyError sites)
  - src/main/python/riscv/gtx/unit/ins/ops/act.py (alias 제거 + GSPR_GTX_OPCODE 활성화)
  - src/main/python/riscv/gtx/unit/ins/ops/vec.py (alias 제거)
loc_total: "+78 -91 net = -13"
---

# Quick Task 260515-0ro: Step 2/4 KeyError/NameError fix + single-source SPR addresses

## What changed

**역할 변경:** csr/gspr.py + csr/lspr.py + csr/nspr.py 의 `@csr` 데코레이터
선언이 **유일한** SPR address 정의 위치가 됨. encoding.py 의 re-export 블록
(260514-vjk 가 land 한 10 줄) + module-level alias (npu.py / tloop_buffer.py /
act.py / vec.py / dma.py) 모두 제거. 사용처는 `GSPR['name'].address` /
`LSPR['name'].address` 로 인라인 호출.

**vendor 매핑 정합:** `GSPR_GTX_OPERAND4 @ 0x004` (Python 확장, dead) 를
`GSPR_GTX_OPCODE @ 0x004` (vendor `gtx_params.h:42`) 로 교체. OPERAND4 사용처
없었음. OPCODE 는 act.py:598/609/627/638 가 사용 중 — KeyError 해소.

**KeyError fix:**

| 파일 | 잘못된 키 | 정합된 키 |
|---|---|---|
| mm.py × 14 | `CSR_LSPR['LSPR_SPM_ADDR_A/_B/_C/_R']` | `CSR_LSPR['SPM_ADDRA/B/C/R']` |
| act.py × 4 | `GSPR['GSPR_GTX_OPCODE']` (등록 안 됨) | csr/gspr.py 등록 후 → `GSPR['GSPR_GTX_OPCODE'].address` |
| act.py × 다수 | `LSPR['LSPM_*'] / LSPR['LSPMSPM_*'] / LSPR['LSPR_*']` (typo) | 이미 다른 editor 가 bare-int 변환했음 → `LSPR['SPM_ADDRA/R'].address` inline |

**NameError fix:**

- `dma.py:70/101/132/195-215` 의 `GSPR_GTX_OPERAND3` / `LSPR_SPM_ADDRA/R`
  — 이전엔 encoding.py 의 re-export 또는 commented import 에 의존, 이제
  `GSPR['...'].address` / `LSPR['...'].address` inline.
- `vec.py:191/192/198/212/223/250/261/262/263` 동일하게 인라인.
- `tloop_buffer.py:213/214/513/514/523/524` 동일.
- `npu.py:238/240/264/265/269/270` (T-loop fast-path) 동일.

## Why

사용자 명시:
1. "csr 부분 제대로 register 참고 안 했고, 재정의한 부분이 있어"
2. "GSPR/LSPR/NSPR Dictionary 참조도 잘못되어 있는 부분이 있어. 확인 후 제대로 고쳐"
3. "alias 만들지마. 이전에 그런게 있다면 다 바꿔"
4. "왜 두번 선언해. 오히려 헷갈려"
5. "그냥 .address 로 바로 접근해"

→ csr 가 single source of truth, 모든 alias / re-export 제거, 사용처 inline.

## Test gate

```bash
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py \
              tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v
```

**26/26 PASS** (회귀 없음).

## Out of scope (Step 3/4 로 미룸)

- **Step 3 — RegisterView attribute access 패턴 통일** (`npu.gspr.GSPR_GTX_OPERAND3`
  형태). 사용자가 vjk 응답에서 추천한 패턴. 현재는 dict-lookup 방식이 살아 있으며
  Step 3 가 그 위에 attribute API 를 얹는 형태로 진행할 수 있음.
- **Step 4 — silent-clamp → assert** (vec.py:245, dma.py:42).
- D-4 stubs (mcast_s2l/g2s/s2s/copy_mem) — 다음 세션.

## Open notes for successor

- `GSPR.get(...)` / `LSPR.get(...)` 같이 RegisterFile.get + Register 객체
  조합이 동작은 하지만 (Register `__index__` 자동 변환), 의미가 모호. Step 3
  의 RegisterView attribute access 가 들어오면 자연스럽게 사라질 패턴.
- vendor `gtx_params.h:42` 에 OPCODE @ 0x004 가 명시. Python 의 OPERAND4 슬롯
  은 이제 사라짐 — Python-확장 코드가 OPERAND4 를 가정하면 안 됨.
- mm.py 가 module-level 에서 `CSR_GSPR` / `CSR_LSPR` 을 import 하는 패턴 —
  Step 3 에서 RegisterView 로 옮기면 이 import 도 줄일 수 있음.
