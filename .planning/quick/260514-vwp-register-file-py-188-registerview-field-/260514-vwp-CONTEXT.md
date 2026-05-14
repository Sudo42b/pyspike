# Quick Task 260514-vwp: register_file.py:188 RegisterView field setter OverflowError fix - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning (decision locked by user)

<domain>
## Task Boundary

`src/main/python/riscv/gtx/unit/register_file.py:188` 의 `RegisterView.__setattr__`
가 너비 64 bit 인 field 를 broadcast write 하려 할 때 `OverflowError: can't
convert negative int to unsigned` 를 발생시킨다. 

이는 quick task 260514-ti0 분석 중 발견되어 "Open Notes for Successor" 로
flag 되었고, 사용자가 명시적으로 다음 fix 대상으로 지정했다.

이 quick task 는 `__setattr__` 로직만 surgical 하게 수정하고, RegisterView 의
다른 행동(`value` 쓰기, 비-field attribute 쓰기, getter, repr 등)은 손대지
않는다.

</domain>

<root_cause>
## Root cause

원본 코드 `register_file.py:182-188`:

```python
if name in self._reg.fields:
    field = self._reg.fields[name]
    mask = field.mask
    shift = field.shift
    
    # (tensor & ~(mask << shift)) | ((value & mask) << shift)
    new_val = torch.as_tensor(value, dtype=torch.int64) & mask
    self._tensor.copy_((self._tensor & ~(mask << shift)) | (new_val << shift))
```

64-bit field (예: `SGPR0.gpr = bits(0, 63)`) 일 때:

- `field.mask` = `(1 << 64) - 1` = `0xFFFFFFFFFFFFFFFF`
- `field.shift` = `0`
- `mask << shift` = `0xFFFFFFFFFFFFFFFF` (Python int, 양수 무한 정밀도)
- `~(mask << shift)` = `-0x10000000000000000` (Python int, 음수 무한 정밀도)
- `self._tensor & (Python negative int outside int64 range)` → torch 가 음수 Python int 를 int64 로 변환 시도, range 초과로 `OverflowError`

</root_cause>

<decisions>
## Implementation Decisions

### Fix 방식 — 64-bit signed wrap via Python int (inline)

`shifted_mask` 를 64-bit unsigned 로 먼저 truncate 후, top bit 가 켜져 있으면
Python int 의 음수 signed 표현으로 변환. 그 다음 Python `~` 를 적용하면 결과
도 int64 범위 안에 머무름.

```python
if name in self._reg.fields:
    field = self._reg.fields[name]
    mask = field.mask
    shift = field.shift

    # Reinterpret the shifted mask as a signed int64 to avoid Python's
    # arbitrary-precision negative result from `~(mask << shift)`, which
    # torch cannot cast back into int64 (OverflowError).
    u64 = (mask << shift) & ((1 << 64) - 1)
    shifted_mask = u64 - (1 << 64) if u64 >> 63 else u64

    new_val = torch.as_tensor(value, dtype=torch.int64) & mask
    self._tensor.copy_((self._tensor & ~shifted_mask) | (new_val << shift))
    return
```

**왜 다른 방식 안 쓰는가:**

- *helper fn `_wrap_int64`*: 한 곳에서만 쓰므로 인라인이 가독성 더 좋음.
- *torch bitwise_not on tensor*: `torch.tensor(mask << shift, dtype=int64)`
  자체가 `mask << shift >= 2^63` 일 때 또 OverflowError. tensor 만들기 전에
  Python 측에서 wrap 필수.
- *두 step (clear then set) bitwise*: `tensor & ~mask` 가 두 번 copy_ 호출
  필요. 비효율.
- *`mask == 0xFFFFFFFFFFFFFFFF and shift == 0` special-case overwrite*:
  특수 케이스 분기 추가 대신 통합 식 한 줄로 통과 — Karpathy §2 (단순).

### `(new_val << shift)` 의 잠재 overflow

`new_val` 은 torch tensor (int64). torch tensor `<<` Python int 는
int64 도메인에서 wrap-around 처리되어 OverflowError 없음.

예: shift=48, value=0xFFFF → `new_val << 48 = 0xFFFF000000000000`. torch 가
이를 int64 signed 로 wrap 해서 부호 비트 켜진 음수로 저장. `|` 결과는
원래 비트 패턴과 동일. tensor read 시 다시 unsigned 로 해석하면 정확.

따라서 이 줄은 손대지 않는다.

### Regression test 추가

`tests/gtx/test_csr_registry_chain.py` 끝에 2 개 test append:

1. **`test_register_view_64bit_field_broadcast_write_no_overflow`** —
   `lspr.SGPR0.gpr = 0xCAFEBABEDEADBEEF` 가 OverflowError 없이 실행되는지,
   그리고 lspr.tensor 의 SGPR0 슬롯(주소 0x800 → mask 0x000) 모든
   `(NEST, SPU)` 에 그 값이 broadcast 되는지 확인.

2. **`test_register_view_partial_field_high_bits_preserves_low_bits`** —
   기존 동작 회귀 가드: 16-bit field 같이 부분 field write 가 기존처럼
   다른 비트를 보존하는지. `THREAD_MASK.mask = 0xABCD` 후 `mask` 부분만
   변경됨을 검증.

기존 24/24 PASS 회귀 유지 필수.

### Out of scope

- `RegisterView.value` setter (라인 176-178) — 이미 동작 정상.
- `RegisterFile.__setitem__/__setattr__` — 다른 코드 path.
- vec.py 등에서 64-bit field 쓰기 callsite 정합 — fix 후 자동으로 동작.
- mm.py `CSR_GSPR['GSPR_GTX_OPERAND3']` 같은 dict-key 패턴 — 무관.

</decisions>

<specifics>
## Specific Ideas

- 변경되는 줄은 register_file.py 의 line 180-189 영역 (10 줄 안쪽).
- 새 test 2 개는 ~25-30 LOC.
- 변경 후 fixture (`lspr` 64-bit SGPR0 attribute access) 가 동작하는지 spot
  check 까지 verify.

</specifics>

<canonical_refs>
## Canonical References

- `src/main/python/riscv/gtx/unit/register_file.py:171-191` (current buggy code)
- `src/main/python/riscv/gtx/unit/csr/lspr.py:22-30` (SGPR0..127 declarations
  with 64-bit `gpr` field — primary case that triggers the bug)
- `src/main/python/riscv/gtx/unit/csr/register.py:48-72` (`Field` /
  `_Bits` semantics — `mask = (1 << (end-start+1)) - 1`)
- `.planning/quick/260514-ti0-csr-custom0-1-dispatch-test-tests-gtx/260514-ti0-SUMMARY.md`
  "Open Notes for Successor" #1 (original bug flag)

</canonical_refs>
