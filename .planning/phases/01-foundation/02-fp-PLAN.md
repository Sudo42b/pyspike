---
phase: 01-foundation
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - src/main/python/riscv/gtx/fp.py
  - tests/gtx/test_fp_roundtrip.py
autonomous: true
requirements:
  - FOUND-01
must_haves:
  truths:
    - "All 65536 FP16 bit patterns round-trip exactly: `fp32_to_fp16(fp16_to_fp32(x)) == x` (bitwise uint16 equality)"
    - "All 2046 NaN bit patterns produce NaN outputs with stable bit pattern (no canonicalization to 0x7E00)"
    - "All FP16 subnormals (exp==0, mantissa!=0) round-trip exactly"
    - "Negative zero (0x8000) preserves sign bit through round-trip"
    - "Known cases: np.float16(1.0) == 0x3C00, 2.0 == 0x4000, 0.5 == 0x3800, -1.0 == 0xBC00 round-trip exactly"
  artifacts:
    - path: "src/main/python/riscv/gtx/fp.py"
      provides: "fp16_to_fp32 / fp32_to_fp16 helpers via np.float16 view (D-09 IEEE 754 binary16 RNE)"
      exports: ["fp16_to_fp32", "fp32_to_fp16"]
    - path: "tests/gtx/test_fp_roundtrip.py"
      provides: "5 acceptance tests covering FOUND-01 — exhaustive idempotency, NaN, subnormals, -0.0, known values"
      contains: "test_all_65536_fp16_values_idempotent"
  key_links:
    - from: "tests/gtx/test_fp_roundtrip.py"
      to: "src/main/python/riscv/gtx/fp.py"
      via: "from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16"
      pattern: "from riscv.gtx.fp import"
    - from: "src/main/python/riscv/gtx/fp.py"
      to: "numpy.ndarray.astype"
      via: "np.float16 view (D-09)"
      pattern: "astype\\(np\\.float(16|32)\\)"
---

<objective>
NumPy 2.x `np.float16` view 기반 FP16↔FP32 변환 헬퍼(`fp16_to_fp32`, `fp32_to_fp16`)를
`src/main/python/riscv/gtx/fp.py`에 작성하고, 65536개 FP16 값 전수 round-trip 검증
+ NaN bit-pattern preservation + subnormal + negative zero를 다루는 5개 테스트 함수를
`tests/gtx/test_fp_roundtrip.py`에 작성한다.

Purpose: D-09 lock-in. NumPy 2.x IEEE 754 binary16 RNE 시맨틱이 C++ `gtx_fp32_to_16`과
helper-level에서 일치하는지 보증 (P4/P5 strict mode에서 op-level 차이는 별도 측정).
PROJECT.md PITFALL #2/#8 (FP16 cast precision)을 직접 방어한다.

Output: 두 파일. RESEARCH.md Example 4(테스트) + Pattern 2(헬퍼)의 정확한 코드를 사용 —
empirically 2026-05-04 NumPy 2.2.6에서 cp310 venv에서 검증됨 (RESEARCH.md "Summary" #2).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/01-foundation/01-CONTEXT.md
@.planning/phases/01-foundation/01-RESEARCH.md
@.planning/phases/01-foundation/01-VALIDATION.md
@CLAUDE.md
</context>

<interfaces>
<!-- 이 plan이 만드는 contract — Phase 4/5 op 핸들러가 사용할 -->

```python
# src/main/python/riscv/gtx/fp.py exports:
def fp16_to_fp32(x: ArrayLike) -> np.ndarray:
    """Widen FP16 → FP32. Lossless. Returns NEW array (astype copies)."""
def fp32_to_fp16(x: ArrayLike) -> np.ndarray:
    """Narrow FP32 → FP16 with IEEE 754 binary16 RNE."""

# ArrayLike = Union[np.ndarray, np.float16, np.float32, float]
```

D-09 risk: subnormal/NaN payload/halfway-rounding 차이는 P4/P5 strict mode에서 측정.
차이 발생 시 `gtx/fp_strict.py` fallback 추가 — Phase 1 scope 밖.
</interfaces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 02-01: tests/gtx/test_fp_roundtrip.py — RED phase</name>
  <files>tests/gtx/test_fp_roundtrip.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-09 view 채택, D-15 test 위치, D-16 65536 round-trip)
    - .planning/phases/01-foundation/01-RESEARCH.md "Example 4: FP roundtrip test (D-16)" (정확한 5개 테스트 함수)
    - .planning/phases/01-foundation/01-VALIDATION.md "Per-Task Verification Map" (4개 자동 verify 명령)
    - tests/test_extension.py (기존 pytest 컨벤션: import 스타일, 타입 힌트)
  </read_first>
  <behavior>
    - test_all_65536_fp16_values_idempotent: 0x0000~0xFFFF 모든 값에 대해 view → fp32 → fp16 → uint16 == 원본 uint16
    - test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern: 2046 NaN bit patterns 보존 (assert_array_equal)
    - test_subnormals_roundtrip: 0x0001~0x03FF + 0x8001~0x83FF 정확히 round-trip
    - test_negative_zero_preserved: 0x8000 sign bit 보존
    - test_known_values: 1.0/2.0/0.5/-1.0 의 hex 값 보존
  </behavior>
  <action>
    `tests/gtx/test_fp_roundtrip.py` 파일을 새로 만들고 RESEARCH.md Example 4의
    정확한 코드를 그대로 작성. 라이선스 헤더는 `tests/__init__.py`와 동일 톤:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    # ... (Apache 2.0 헤더)
    #
    """Phase 1 acceptance: 65536 FP16 values round-trip exactly through fp.fp16_to_fp32 / fp32_to_fp16.

    D-09 risk acknowledgment: NumPy 2.x np.float16 RNE may differ from C++ gtx_fp32_to_16
    on subnormal/NaN payload/halfway-rounding edge cases. Phase 1 verifies the *helper-level*
    round-trip; full strict-mode comparison vs C++ is deferred to P4/P5.
    """
    import numpy as np

    from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16


    def test_all_65536_fp16_values_idempotent():
        """For every FP16 bit pattern x: fp32_to_fp16(fp16_to_fp32(x)) == x (bitwise)."""
        all_u16 = np.arange(65536, dtype=np.uint16)
        all_f16 = all_u16.view(np.float16)

        fp32 = fp16_to_fp32(all_f16)
        back_f16 = fp32_to_fp16(fp32)
        back_u16 = back_f16.view(np.uint16)

        # Empirically verified on NumPy 2.2.6 (cp310 on x86_64 LE):
        # ALL 65536 values round-trip exactly, including all 2046 NaN bit patterns.
        np.testing.assert_array_equal(back_u16, all_u16)


    def test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern():
        """NaN inputs produce NaN outputs; bit pattern is preserved (NumPy 2.x behavior)."""
        all_u16 = np.arange(65536, dtype=np.uint16)
        all_f16 = all_u16.view(np.float16)
        nan_mask = np.isnan(all_f16)
        nan_count = int(nan_mask.sum())
        assert nan_count == 2046, f"Expected 2046 NaN bit patterns, got {nan_count}"

        back_u16 = fp32_to_fp16(fp16_to_fp32(all_f16)).view(np.uint16)
        # All NaN inputs produce NaN outputs:
        assert np.all(np.isnan(back_u16.view(np.float16)[nan_mask]))
        # Bit pattern is preserved (HIGH-confidence on NumPy 2.x):
        np.testing.assert_array_equal(back_u16[nan_mask], all_u16[nan_mask])


    def test_subnormals_roundtrip():
        """All FP16 subnormals (exp == 0, mantissa != 0) round-trip exactly."""
        # FP16 subnormals: 0x0001..0x03FF and 0x8001..0x83FF
        subnormal_pos = np.arange(0x0001, 0x0400, dtype=np.uint16)
        subnormal_neg = np.arange(0x8001, 0x8400, dtype=np.uint16)
        subs = np.concatenate([subnormal_pos, subnormal_neg]).view(np.float16)

        back = fp32_to_fp16(fp16_to_fp32(subs)).view(np.uint16)
        expected = np.concatenate([subnormal_pos, subnormal_neg])
        np.testing.assert_array_equal(back, expected)


    def test_negative_zero_preserved():
        """fp32_to_fp16(fp16_to_fp32(np.float16(-0.0))) preserves -0.0 (sign bit)."""
        neg_zero_u16 = np.array([0x8000], dtype=np.uint16)
        neg_zero_f16 = neg_zero_u16.view(np.float16)
        back = fp32_to_fp16(fp16_to_fp32(neg_zero_f16)).view(np.uint16)
        np.testing.assert_array_equal(back, neg_zero_u16)


    def test_known_values():
        """Sanity-check known FP16 <-> FP32 conversions."""
        cases = [
            (np.float16(1.0), np.float32(1.0), 0x3C00),
            (np.float16(2.0), np.float32(2.0), 0x4000),
            (np.float16(0.5), np.float32(0.5), 0x3800),
            (np.float16(-1.0), np.float32(-1.0), 0xBC00),
        ]
        for f16, f32, raw in cases:
            assert fp16_to_fp32(f16) == f32
            assert fp32_to_fp16(f32) == f16
            assert int(fp32_to_fp16(f32).view(np.uint16)) == raw
    ```

    중요한 점:
    - TDD RED phase: 이 task 단독으로는 `riscv.gtx.fp` 가 없어서 모든 테스트가
      `ModuleNotFoundError`로 실패해야 정상. Task 02-02가 GREEN을 만든다.
    - `from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16` 정확한 import 라인 사용 —
      Task 02-02가 이 두 심볼을 export해야 한다 (contract).
    - `np.testing.assert_array_equal`은 정확한 동등성 (NaN은 NaN과 같지 않으므로 NaN
      마스크는 분리해서 검증 — RESEARCH.md Example 4 그대로).
    - 테스트는 vectorized (`for i in range(65536)` 사용 안 함 — RESEARCH.md
      "Anti-Patterns" #4 방어).
  </behavior>
  <verify>
    <automated>test -f tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'def test_all_65536_fp16_values_idempotent' tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'def test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern' tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'def test_subnormals_roundtrip' tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'def test_negative_zero_preserved' tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'def test_known_values' tests/gtx/test_fp_roundtrip.py &amp;&amp; grep -q 'from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16' tests/gtx/test_fp_roundtrip.py &amp;&amp; python -c "import ast; ast.parse(open('tests/gtx/test_fp_roundtrip.py').read())"</automated>
  </verify>
  <acceptance_criteria>
    - 위 grep 명령들 모두 종료코드 0 (5개 test 함수 + import 라인 + syntax valid)
    - `grep -c 'def test_' tests/gtx/test_fp_roundtrip.py` 출력 == 5
    - `grep -q 'np.testing.assert_array_equal' tests/gtx/test_fp_roundtrip.py` 종료코드 0 (정확한 동등성 비교 사용)
    - `grep -q 'for i in range(65536)' tests/gtx/test_fp_roundtrip.py` 종료코드 1 (vectorized — Python loop 금지, RESEARCH.md "Anti-Patterns" #4)
    - `python -c "import ast; ast.parse(open('tests/gtx/test_fp_roundtrip.py').read())"` 종료코드 0
  </acceptance_criteria>
  <done>5개 테스트 함수가 RESEARCH.md Example 4 그대로 작성됨; `from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16` 라인 존재; syntax valid. 이 시점에는 `riscv.gtx.fp` 모듈이 없어서 RED 상태 (Task 02-02가 GREEN으로 전환).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 02-02: src/main/python/riscv/gtx/fp.py — GREEN phase</name>
  <files>src/main/python/riscv/gtx/fp.py</files>
  <read_first>
    - .planning/phases/01-foundation/01-CONTEXT.md (D-09: np.float16 view, NOT 비트 조작)
    - .planning/phases/01-foundation/01-RESEARCH.md "Pattern 2: FP16 conversion via NumPy view (D-09)" (정확한 헬퍼 코드)
    - tests/gtx/test_fp_roundtrip.py (Task 02-01 출력 — `from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16` contract)
  </read_first>
  <behavior>
    - fp16_to_fp32(arr) → np.ndarray (dtype=np.float32) — `astype(np.float16)` 정규화 후 `astype(np.float32)`
    - fp32_to_fp16(arr) → np.ndarray (dtype=np.float16) — `astype(np.float32)` 정규화 후 `astype(np.float16)` (NumPy 2.x IEEE 754 binary16 RNE)
    - 두 함수 모두 scalar (np.float16/np.float32/float), np.ndarray 모두 받아들임
    - 두 함수 모두 NEW array 반환 (`astype` 항상 copy — D-12 view-base 불변량 적용 안 됨; helper-level은 copy 허용)
  </behavior>
  <action>
    `src/main/python/riscv/gtx/fp.py` 파일을 새로 만들고 RESEARCH.md Pattern 2의
    정확한 코드를 작성:

    ```python
    #
    # Copyright 2026 WuXi EsionTech Co., Ltd.
    #
    # Licensed under the Apache License, Version 2.0 (the "License"); ...
    # (라이선스 헤더 — 다른 모듈과 동일 톤)
    #
    """FP16 / FP32 conversion helpers — D-09 (np.float16 view via astype, NOT bit manipulation).

    NumPy 2.x guarantees IEEE 754 binary16 RNE for astype(np.float16). All 65536 FP16
    values round-trip exactly; NaN bit patterns are preserved (empirically verified on
    NumPy 2.2.6, see tests/gtx/test_fp_roundtrip.py).

    Risk acknowledgment (D-09): subnormal/NaN payload/halfway-rounding edge cases vs C++
    gtx_fp32_to_16 are deferred to P4/P5 strict-mode measurement. If discrepancies arise,
    fp_strict.py (bit-manipulation port of gtx_npu.h:89-151) will be added as fallback.
    """
    from typing import Union

    import numpy as np

    ArrayLike = Union[np.ndarray, np.float16, np.float32, float]


    def fp16_to_fp32(x: ArrayLike) -> np.ndarray:
        """Widen FP16 -> FP32. Lossless (widening cast).

        Note: returns a NEW array (astype always copies). Caller MUST NOT expect
        base preservation — D-12 (view-base invariant) applies to memory accessors,
        not to FP conversion helpers.

        For zero-copy FP32 reduction over FP16 storage, use mem.l1_f16(...) directly
        and pass to NumPy reductions with dtype=np.float32 keyword (Phases 4/5 pattern).
        """
        return np.asarray(x, dtype=np.float16).astype(np.float32)


    def fp32_to_fp16(x: ArrayLike) -> np.ndarray:
        """Narrow FP32 -> FP16 with IEEE 754 binary16 RNE (NumPy 2.x default).

        Empirically verified on NumPy 2.2.6: idempotent for all 65536 FP16 values
        (including NaN bit-pattern preservation, subnormals, negative zero).
        """
        return np.asarray(x, dtype=np.float32).astype(np.float16)
    ```

    중요한 점:
    - `np.asarray(x, dtype=np.float16)`로 입력을 정규화 (scalar/list/ndarray 모두 처리).
    - `astype(np.float32)` / `astype(np.float16)` 만 사용 — 비트 조작 금지 (D-09).
    - 함수는 `np.ndarray` 반환. scalar 입력에도 0-d array 반환 (`assert ==` 시 NumPy
      broadcasting으로 동작).
    - 타입 힌트는 cp310+이므로 `Union` 사용 (PEP 604 `|`도 가능하지만 RESEARCH.md
      Pattern 2 그대로 — `Union` 유지).
    - NO C++ code (CLAUDE.md "C++ 추가 코드 금지").
    - NO scipy, ml_dtypes 등 추가 의존성 (CLAUDE.md "NumPy 외부 추가 런타임 의존성
      신규 도입 금지").
  </action>
  <verify>
    <automated>test -f src/main/python/riscv/gtx/fp.py &amp;&amp; cd /mnt/e/14_NIGHTLY/pyspike &amp;&amp; PYTHONPATH=src/main/python python -c "from riscv.gtx.fp import fp16_to_fp32, fp32_to_fp16; import numpy as np; assert fp32_to_fp16(np.float32(1.0)) == np.float16(1.0); assert fp16_to_fp32(np.float16(1.0)) == np.float32(1.0); all_u16 = np.arange(65536, dtype=np.uint16); back = fp32_to_fp16(fp16_to_fp32(all_u16.view(np.float16))).view(np.uint16); np.testing.assert_array_equal(back, all_u16); print('OK')" &amp;&amp; PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py -x -p no:pylint -p no:mypy --no-header -q</automated>
  </verify>
  <acceptance_criteria>
    - `test -f src/main/python/riscv/gtx/fp.py` 종료코드 0
    - `grep -q 'def fp16_to_fp32' src/main/python/riscv/gtx/fp.py` 종료코드 0
    - `grep -q 'def fp32_to_fp16' src/main/python/riscv/gtx/fp.py` 종료코드 0
    - `grep -q 'astype(np.float32)' src/main/python/riscv/gtx/fp.py` 종료코드 0 (D-09 view 패턴)
    - `grep -q 'astype(np.float16)' src/main/python/riscv/gtx/fp.py` 종료코드 0
    - `grep -E '(struct\.pack|int\.from_bytes|<<|>>) ' src/main/python/riscv/gtx/fp.py` 종료코드 1 (D-09: 비트 조작 금지)
    - `python -c "import ast; ast.parse(open('src/main/python/riscv/gtx/fp.py').read())"` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py::test_all_65536_fp16_values_idempotent -x -p no:pylint -p no:mypy --no-header -q` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py::test_nan_inputs_produce_nan_outputs_with_stable_bit_pattern -x -p no:pylint -p no:mypy --no-header -q` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py::test_subnormals_roundtrip -x -p no:pylint -p no:mypy --no-header -q` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py::test_negative_zero_preserved -x -p no:pylint -p no:mypy --no-header -q` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py::test_known_values -x -p no:pylint -p no:mypy --no-header -q` 종료코드 0
    - `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py -p no:pylint -p no:mypy --no-header -q` 전체 통과 (5/5)
  </acceptance_criteria>
  <done>fp.py 존재 + 5개 테스트 모두 GREEN. 65536 round-trip + NaN + subnormal + -0.0 + known values 모두 bit-exact.</done>
</task>

</tasks>

<verification>
**Plan-level verification:**
- `PYTHONPATH=src/main/python pytest tests/gtx/test_fp_roundtrip.py -v` → 5/5 PASS
- `grep -c 'astype' src/main/python/riscv/gtx/fp.py` 출력 >= 2 (D-09 view 패턴)
- `grep -E '(struct|<<|int\.from_bytes)' src/main/python/riscv/gtx/fp.py` 출력 비어있음 (비트 조작 금지)
- 65536 round-trip 테스트가 1초 내에 완료 (D-16 expectation)
</verification>

<success_criteria>
1. `fp.py`에 `fp16_to_fp32`, `fp32_to_fp16` 두 함수만 export
2. `tests/gtx/test_fp_roundtrip.py`에 5개 테스트 함수 (idempotent, nan, subnormal, neg-zero, known)
3. `pytest tests/gtx/test_fp_roundtrip.py -x` 5/5 PASS (FOUND-01 fully covered)
4. 비트 조작 코드 NO (`astype` only)
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation/02-fp-SUMMARY.md` with:
- fp.py 정확한 LOC + exported 함수 목록
- 5개 테스트 결과 (PASS/FAIL count + 타이밍)
- D-09 risk: NumPy 2.2.6에서 65536/65536 round-trip 통과 확인 (P4/P5 strict 측정 대기)
- 비트 조작 사용 안 함 — fp_strict.py fallback은 P4/P5에서 필요 시 추가
</output>
