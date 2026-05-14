# Quick Task 260514-vjk: GSPR_GTX_OPERAND0..5 register 복원 - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning (decision locked by user)

<domain>
## Task Boundary

d6f73f9 "Architecture Refactoring" 가 `src/main/python/riscv/gtx/unit/ins/encoding.py`
에서 `GSPR_GTX_OPERAND0..5` + `GSPR_GTX_OPCODE` bare-int constants 를 삭제했고,
`src/main/python/riscv/gtx/unit/csr/gspr.py` 에도 동일한 register를 `@csr`
데코레이터로 다시 등록하지 않았다. 결과로 **5 개 파일에 잠재된 NameError /
KeyError** 가 남아 있고 — T-loop fast-path 와 MM/VEC/ACT 핸들러가 실행되는
순간 발동한다.

이 quick task 는 csr/gspr.py 를 source of truth 로 정해 6 개 register 를
등록하고, encoding.py 에서 bare-int 별칭을 re-export 하며, 5 개 사용 site 의
import 를 정합시킨다.

</domain>

<decisions>
## Implementation Decisions

### Source of truth — csr/gspr.py 에 @csr 등록

vendor `vendor/gtx_cpp_reference/gtx/gtx_params.h:36-44` 가 진짜 source of
truth (GSPR_GTX_RUN/OPERAND1/2/3 + GSPR_GTX_OPCODE@0x004). 단 d6f73f9 **이전**
의 Python `encoding.py:51-57` 은 vendor 보다 확장된 매핑을 사용했고
(OPERAND0..5 모두 정의, OPCODE@0x012), 이 매핑이 ops/spr.py opset handler
(slot=0 → 0x003, slot=1 → 0x005) 의 기반이다. **이 quick task 는 Python 확장
매핑을 보존**한다 — vendor 와 부분적으로 다르지만 d6f73f9 이전 동작 보존.

따라서 csr/gspr.py 에 다음 6 개를 추가 (block 위치는 STACK_INFO 바로 위,
"64-bit PIPE Registers" 섹션 헤더 아래):

```python
@csr(name="GSPR_GTX_OPERAND0", address=0x000, width=64, rw_type="RW")
class GSPR_GTX_OPERAND0:
    value = bits(0, 63)

@csr(name="GSPR_GTX_OPERAND1", address=0x001, width=64, rw_type="RW")
class GSPR_GTX_OPERAND1:
    value = bits(0, 63)

@csr(name="GSPR_GTX_OPERAND2", address=0x002, width=64, rw_type="RW")
class GSPR_GTX_OPERAND2:
    value = bits(0, 63)

@csr(name="GSPR_GTX_OPERAND3", address=0x003, width=64, rw_type="RW")
class GSPR_GTX_OPERAND3:
    value = bits(0, 63)

@csr(name="GSPR_GTX_OPERAND4", address=0x004, width=64, rw_type="RW")
class GSPR_GTX_OPERAND4:
    value = bits(0, 63)

@csr(name="GSPR_GTX_OPERAND5", address=0x005, width=64, rw_type="RW")
class GSPR_GTX_OPERAND5:
    value = bits(0, 63)
```

`value = bits(0, 63)` 단일 field. 이건 `@csr` 데코레이터 요구 (`if not
fields: raise`) 만족시키기 위한 최소 declaration. callsite 가 raw int 만
사용하므로 의미 있는 sub-field 분할 불필요.

### encoding.py re-export

`encoding.py` 끝에 다음을 추가 (csr에서 derived; 단일 source of truth):

```python
# ============================================================================
# GSPR address constants — re-exported from csr/gspr.py for raw int access
# (bare-int access patterns: npu.py T-loop fast-path, tloop_buffer.py,
# act.py, vec.py). MM uses CSR_GSPR['GSPR_GTX_OPERAND3'] dict lookup directly.
# ============================================================================
from ..csr.gspr import GSPR as _GSPR_REGS
GSPR_GTX_OPERAND0: int = _GSPR_REGS['GSPR_GTX_OPERAND0'].address
GSPR_GTX_OPERAND1: int = _GSPR_REGS['GSPR_GTX_OPERAND1'].address
GSPR_GTX_OPERAND2: int = _GSPR_REGS['GSPR_GTX_OPERAND2'].address
GSPR_GTX_OPERAND3: int = _GSPR_REGS['GSPR_GTX_OPERAND3'].address
GSPR_GTX_OPERAND4: int = _GSPR_REGS['GSPR_GTX_OPERAND4'].address
GSPR_GTX_OPERAND5: int = _GSPR_REGS['GSPR_GTX_OPERAND5'].address
```

`OPCODE` 는 이번 task 에서 필요한 callsite 없음 — out of scope.

### 사용 site 정합 (5 파일)

| 파일 | 현재 상태 | 수정 |
|---|---|---|
| `src/main/python/riscv/gtx/npu.py` | `_GSPR_OP3`/`_GSPR_OP5` 정의 없이 line 238/240/264/265/269/270 사용 | line 27-29 import 블록에 `GSPR_GTX_OPERAND3 as _GSPR_OP3, GSPR_GTX_OPERAND5 as _GSPR_OP5` 추가 |
| `src/main/python/riscv/gtx/tloop_buffer.py` | line 35 `# from .unit.ins.encoding import GSPR_GTX_OPERAND3, GSPR_GTX_OPERAND5` 주석 | 주석 해제 |
| `src/main/python/riscv/gtx/unit/ins/ops/act.py` | line ~32 `# GSPR_GTX_OPERAND1, GSPR_GTX_OPERAND2, GSPR_GTX_OPERAND3,` 주석 | 주석 해제 (정확한 import block 위치는 executor 가 확인) |
| `src/main/python/riscv/gtx/unit/ins/ops/mm.py` | `CSR_GSPR['GSPR_GTX_OPERAND3']` dict lookup | csr/gspr.py 등록만으로 자동 동작 — 수정 불필요 |
| `src/main/python/riscv/gtx/unit/ins/ops/vec.py` | `npu.gspr.get("GSPR_GTX_OPERAND3", ...)` string key + `npu.gspr[GSPR_GTX_OPERAND2]` int key | string key 는 csr 등록만으로 동작. int key (`GSPR_GTX_OPERAND2` 변수) 는 encoding.py import 추가 필요 |

### Test 검증

기존 23/23 PASS 회귀 유지 필수:
```bash
uv run pytest tests/gtx/test_fsm_smoke.py tests/gtx/test_custom0_smoke.py \
              tests/gtx/test_csr_registry_chain.py tests/gtx/test_custom_dispatch_chain.py -v
```

추가로 **T-loop fast-path NameError 가 정말 사라졌는지** 검증하는 1 개 unit
test 를 `test_custom_dispatch_chain.py` 끝에 append:

```python
def test_tloop_fast_path_opset_no_nameerror(gtx_npu, mock_proc, dummy_insn):
    """custom0 T-loop fast-path inline OPSET no longer raises NameError
    on _GSPR_OP3 / _GSPR_OP5 (260514-vjk: GSPR_GTX_OPERAND restored)."""
    from riscv.gtx.unit.ins.encoding import GTX_ISS_F7_OPSET
    gtx_npu.warp.is_tloop = True
    gtx_npu._tloop_buf = []
    mock_proc.state.XPR.write(1, 0)  # slot=0 -> OPERAND3
    mock_proc.state.XPR.write(2, 0xCAFE)
    dummy_insn.funct = GTX_ISS_F7_OPSET
    dummy_insn.rs1, dummy_insn.rs2 = 1, 2
    assert gtx_npu.custom0(mock_proc, dummy_insn, 0, 0) == 0
    assert int(gtx_npu.gspr.tensor[0x003]) == 0xCAFE
```

### Out of scope

- `OverflowError @ register_file.py:188` (2번 task 로 분리)
- `GSPR_GTX_OPCODE` (이번 callsite 에 없음)
- mm.py / vec.py 의 string vs int key 불일치 코딩 스타일 통일 (정합만 보장하면 충분)
- act.py 의 numerous other GSPR_GTX_OPERAND 사용 (이미 OPERAND1/2/3 만 import 하면 모두 정합)

### Claude's Discretion

- act.py 의 정확한 import block 위치 (line ~32 의 주석 처리된 import 가 어디서 정의되는지 executor 가 read 로 확인)
- vec.py 의 string vs int key 패턴 차이는 그대로 두기 (코딩 스타일 정합은 별도 task)

</decisions>

<specifics>
## Specific Ideas

- csr/gspr.py 의 declaration block 순서 보존 — STACK_INFO 가 0x010 으로 시작
  하므로 6 개 OPERAND register 를 0x000..0x005 로 declaration 블록 맨 위에
  insert. comment header 추가: `# Operand staging slots (OPSET writes 0x003
  on slot=0, 0x005 on slot=1; see ops/spr.py:opset)`.
- encoding.py 의 re-export 블록은 파일 끝에 추가 (다른 ISS_F7_* / VEC / ACT
  constants 와 함께 module-level int constants 가 모여 있음).
- npu.py 의 import 블록 (line 27-29) 은:
  ```python
  from .unit.ins.encoding import (
      GTX_ISS_F7_OPSET,
      GSPR_GTX_OPERAND3 as _GSPR_OP3,
      GSPR_GTX_OPERAND5 as _GSPR_OP5,
  )
  ```
  순서 유지.

</specifics>

<canonical_refs>
## Canonical References

- `vendor/gtx_cpp_reference/gtx/gtx_params.h:36-44` (vendor source of truth)
- d6f73f9 이전 `encoding.py:51-57` (Python 확장 매핑 — `git show d6f73f9~1:src/main/python/riscv/gtx/unit/ins/encoding.py`)
- `src/main/python/riscv/gtx/unit/csr/__init__.py` (CSR_GSPR PIPE-only view)
- `src/main/python/riscv/gtx/unit/csr/register.py` (@csr decorator + make_csr)
- `src/main/python/riscv/gtx/unit/ins/ops/spr.py:96-107` (opset handler, hard-coded 0x003/0x005 inline)
- `.planning/quick/260514-ti0-csr-custom0-1-dispatch-test-tests-gtx/260514-ti0-SUMMARY.md` (Open Notes #2 — original flag)

</canonical_refs>
