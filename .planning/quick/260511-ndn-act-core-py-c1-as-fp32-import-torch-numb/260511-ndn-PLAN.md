---
phase: quick-260511-ndn-act-core-c1
plan: 01
type: execute
wave: 1
quick_id: 260511-ndn
description: act_core.py C1 — imports/helpers/ReLU/PReLU/FP8 LUT torch migration
mode: quick
scope: chunk C1 of 4-chunk act_core torch migration
created: 2026-05-11
must_haves:
  truths:
    - act_core.py 내부에서 numpy/numba 의존 제거 시작 (이번 청크 범위만)
    - 공개 함수 시그니처 호환 (act_engine 호출 측 무수정) — 입력 ndarray|Tensor 수용, 출력 torch.Tensor
    - FP8 dual-path — GTX-custom LUT(수치 그대로, storage만 torch) + torch.float8_e4m3fn 코덱 병존
    - vendor regression RELU/LEAKY_RELU strict-mode PASS (ULP=1)
  artifacts:
    - src/main/python/riscv/gtx/act_core.py 수정
    - helpers `_as_fp32`, `_to_torch_fp16`
    - e4m3 codec `fp8_e4m3_to_fp16`, `fp16_to_fp8_e4m3`
  key_links:
    - src/main/python/riscv/gtx/vec_core.py (phase 1 reference 53eb670)
    - src/main/python/riscv/gtx/act_engine.py:39-43 (caller import list)
    - act_core.py:77-110 (GTX-custom FP8 spec 보존)
---

# Quick Task 260511-ndn — act_core.py C1

## Goal

`act_core.py` 18-kernel 모듈을 torch backend로 4단계 청크 마이그레이션. 이번 **C1**: imports + helpers + ReLU/PReLU(2 kernels) + FP8 LUT 2종. 나머지 16 kernel(Pool 2 + Transcendentals 5 + CVT 9)는 후속 청크.

## Scope decisions (locked)

| 영역 | 결정 |
|---|---|
| numba | 이번 청크에서 건드리는 함수(`_relu_impl/_njit`, `_prelu_impl/_njit`)만 제거. transcendentals/Pool/CVT는 다음 청크. |
| numpy | C1 범위 함수에서만 numpy 제거. 다른 함수가 still numpy 사용 → import는 잔류. |
| 외부 API | caller가 ndarray 넘기는 것은 수용 — `_as_fp32`가 torch.as_tensor로 흡수 |
| FP8 표현 | dual-path. (1) GTX-custom LUT 경로 보존 (수치 로직 동일, storage `np.ndarray` → `torch.Tensor` 변환) (2) 신규 e4m3 codec 경로 (`torch.float8_e4m3fn` 네이티브). |
| 반환 타입 | C1 범위 공개 함수는 `torch.Tensor` 반환 (vec_core 패턴 일치) |

## Out-of-scope (다음 청크)

- Pool max/avg (C2)
- Transcendentals: gelu/tanh_act/sigmoid/softmax/esum (C3 — **SIGMOID 회복**)
- CVT 9종 (C4)
- act_engine.py 호출 측 수정 (caller-side ndarray→tensor 어댑테이션은 별도 청크 / 필요시 verify에서 surface)
- `_jit.py` 파일 자체 삭제

## Tasks

### Task 1: act_core.py C1 migration (single atomic edit)

**Files:**
- `src/main/python/riscv/gtx/act_core.py`

**Action:**

1. **Header import 갱신** (라인 ~58-68 영역)
   - **추가**: `import torch`
   - **유지**: `import numpy as np`, `from numpy.typing import NDArray`, `from ._jit import njit, HAS_NUMBA` (transcendentals/Pool/CVT 함수가 아직 사용 중)

2. **모듈 helpers 추가** (FP8 LUT 빌더 직전 / 또는 import 블록 뒤)
   ```python
   def _as_fp32(a) -> torch.Tensor:
       """ndarray | torch.Tensor | scalar → FP32 torch.Tensor.
       gtx 외부 API ingestion 포인트. caller가 ndarray 넘겨도 흡수."""
       if isinstance(a, torch.Tensor):
           return a.to(torch.float32)
       return torch.as_tensor(a, dtype=torch.float32)

   def _to_torch_fp16(t: torch.Tensor) -> torch.Tensor:
       """FP32 torch.Tensor → FP16 torch.Tensor (single cast site)."""
       return t.to(torch.float16)
   ```

3. **FP8 LUT 빌더 — storage만 torch로 (수치 로직 그대로)**
   - `_build_fp8_to_fp16_lut() -> torch.Tensor`: 기존 256-iter Python loop 그대로, 최종 `torch.tensor(vals, dtype=torch.float16)`. negative zero bit pattern `lut[0x80] = torch.tensor(-0.0, dtype=torch.float16)` 유지.
   - `_build_fp16_to_fp8_lut() -> torch.Tensor`: 기존 65536-iter Python loop 그대로, 최종 `torch.tensor(vals, dtype=torch.uint8)`.
   - 모듈 globals 갱신:
     ```python
     FP8_TO_FP16_LUT: torch.Tensor = _build_fp8_to_fp16_lut()
     FP16_TO_FP8_LUT: torch.Tensor = _build_fp16_to_fp8_lut()
     ```
   - **주의**: cvt_qh/cvt_hq의 _impl 함수가 여전히 `FP8_TO_FP16_LUT[uint8_index]` 형태로 인덱싱하므로, torch.Tensor 인덱싱도 동일하게 동작함을 verify 단계에서 확인. 만약 깨지면 임시로 `FP8_TO_FP16_LUT_NP = FP8_TO_FP16_LUT.numpy()` 호환 alias 추가 (C4 CVT 마이그레이션 때 정리).

4. **e4m3 신규 코덱 추가** (FP8 LUT 빌더 뒤)
   ```python
   def fp8_e4m3_to_fp16(t_e4m3: torch.Tensor) -> torch.Tensor:
       """NVIDIA E4M3 native codec (torch.float8_e4m3fn → FP16).
       GTX-custom LUT과 별개 경로. vendor parity 필요시 LUT 경로 사용."""
       return t_e4m3.to(torch.float16)

   def fp16_to_fp8_e4m3(t_fp16: torch.Tensor) -> torch.Tensor:
       """FP16 → NVIDIA E4M3 native (torch.float8_e4m3fn).
       GTX-custom encoding과 일치 보장 안 함 — 신규 학습 경로용."""
       if not isinstance(t_fp16, torch.Tensor):
           t_fp16 = torch.as_tensor(t_fp16, dtype=torch.float16)
       return t_fp16.to(torch.float8_e4m3fn)
   ```

5. **ReLU/PReLU public 함수 본문 교체**

   **삭제 대상 (C1 한정):**
   - `_relu_impl` (FP32 numpy)
   - `_prelu_impl` (FP32 numpy)
   - `_relu_njit = njit(cache=True)(_relu_impl)` 정의
   - `_prelu_njit = njit(cache=True)(_prelu_impl)` 정의

   **신규 본문:**
   ```python
   def relu(arr) -> torch.Tensor:
       """RELU: max(0, x). FP32 internal, FP16 output.
       Source: gtx_npu_act.cc:60-67 (forward)."""
       a_f32 = _as_fp32(arr)
       return _to_torch_fp16(torch.relu(a_f32))

   def prelu(arr, slope) -> torch.Tensor:
       """PRELU: x if x>=0 else slope*x.
       Source: gtx_npu_act.cc:118-131 (reversed: vendor C++ `(a<0)?slope*a:a`)."""
       a_f32 = _as_fp32(arr)
       slope_f32 = (
           slope.to(torch.float32) if isinstance(slope, torch.Tensor)
           else float(slope)
       )
       out = torch.where(a_f32 < 0.0, slope_f32 * a_f32, a_f32)
       return _to_torch_fp16(out)
   ```

**Verify:**

```bash
# 1. Import smoke
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -c "
import numpy as np, torch
from riscv.gtx import act_core
assert act_core._as_fp32(np.array([1.0, 2.0])).dtype == torch.float32
assert act_core._as_fp32(torch.tensor([1.0, 2.0])).dtype == torch.float32
assert isinstance(act_core.FP8_TO_FP16_LUT, torch.Tensor)
assert act_core.FP8_TO_FP16_LUT.dtype == torch.float16
assert act_core.FP8_TO_FP16_LUT.shape == (256,)
assert isinstance(act_core.FP16_TO_FP8_LUT, torch.Tensor)
assert act_core.FP16_TO_FP8_LUT.dtype == torch.uint8
assert act_core.FP16_TO_FP8_LUT.shape == (65536,)
t = torch.tensor([1.5, -2.0], dtype=torch.float16)
e4m3 = act_core.fp16_to_fp8_e4m3(t)
assert e4m3.dtype == torch.float8_e4m3fn
back = act_core.fp8_e4m3_to_fp16(e4m3)
assert back.dtype == torch.float16
r = act_core.relu(np.array([-1.0, 0.0, 2.5], dtype=np.float16))
assert r.dtype == torch.float16
assert torch.allclose(r, torch.tensor([0.0, 0.0, 2.5], dtype=torch.float16))
p = act_core.prelu(np.array([-2.0, 3.0], dtype=np.float16), 0.25)
assert torch.allclose(p, torch.tensor([-0.5, 3.0], dtype=torch.float16))
# negative zero in LUT
import struct
neg_zero_bits = act_core.FP8_TO_FP16_LUT[0x80].view(torch.uint16).item()
assert neg_zero_bits == 0x8000, f'neg-zero bit pattern lost: {neg_zero_bits:#06x}'
print('C1 import smoke OK')
"

# 2. FP8 LUT 자체-일관성
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -c "
import torch
from riscv.gtx import act_core
n_stable = 0
for code in range(256):
    fp16_val = act_core.FP8_TO_FP16_LUT[code]
    if torch.isnan(fp16_val) or torch.isinf(fp16_val):
        continue
    u16 = fp16_val.view(torch.uint16).item()
    fp8_back = act_core.FP16_TO_FP8_LUT[u16].item()
    n_stable += 1
assert n_stable > 200, f'expected most FP8 codes stable, got {n_stable}'
print(f'FP8 LUT consistency: {n_stable}/256 stable')
"

# 3. RELU vendor regression
PATH=$PWD/.venv/bin:$PATH GTX_VENDOR_TEST_DIR=$PWD/test/ \
  .venv/bin/python -m pytest \
  'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[RELU]' \
  --no-cov --timeout=180 -v

# 4. LEAKY_RELU vendor regression (PReLU slope path)
PATH=$PWD/.venv/bin:$PATH GTX_VENDOR_TEST_DIR=$PWD/test/ \
  .venv/bin/python -m pytest \
  'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[LEAKY_RELU]' \
  --no-cov --timeout=180 -v

# 5. 회귀 방지 (transcendentals 본문 미변경 — 깨지면 안 됨)
PATH=$PWD/.venv/bin:$PATH GTX_VENDOR_TEST_DIR=$PWD/test/ \
  .venv/bin/python -m pytest \
  'tests/gtx/test_regression_fw_full_sweep.py::test_vendor_op_sweep_strict[GELU]' \
  --no-cov --timeout=180 -v
```

**Done criteria:**
- [ ] act_core.py import smoke 통과 (FP8 LUT torch.Tensor + helpers + e4m3 codec + RELU + PRELU + 0x8000 bit pattern 보존)
- [ ] FP8 LUT 자체-일관성 점검 통과 (n_stable > 200)
- [ ] `RELU` strict-mode PASS (ULP=1)
- [ ] `LEAKY_RELU` strict-mode PASS (ULP=1)
- [ ] `GELU` PASS (회귀 방지)
- [ ] act_core.py 변경 라인 ≤ 200

## Risk register

| 위험 | 완화 |
|---|---|
| cvt_qh/cvt_hq의 `_impl`이 `FP8_TO_FP16_LUT[idx]` 인덱싱 — torch.Tensor index → torch.Tensor 결과 (scalar tensor) vs 기존 numpy scalar 일치성 차이로 cvt 깨질 가능성 | verify 5번(GELU)는 안전. 하지만 cvt 회귀를 따로 확인하지 않으면 C4까지 잠복. 응급 대안: `FP8_TO_FP16_LUT_NP = FP8_TO_FP16_LUT.numpy()` alias 추가 + _impl이 이걸 쓰도록. 필요시 verify에 cvt 회귀 추가. |
| ReLU/PReLU 반환이 `np.ndarray` → `torch.Tensor` — caller(act_engine)에서 `arr.astype(np.float16).view(np.uint16)` 같은 numpy-only 호출이 있으면 깨짐 | RELU/LEAKY_RELU vendor regression이 잡아냄. 깨지면 act_engine에 `np.asarray(t)` 어댑터 추가 (executor 재량). |
| FP8 LUT의 negative zero (0x8000) bit pattern 손실 | 명시적 후처리 `lut[0x80] = torch.tensor(-0.0, dtype=torch.float16)` + verify 1번에서 비트패턴 assert |
| @njit 제거로 ReLU/PReLU 성능 저하 | torch.relu/torch.where는 vectorized — 측정은 후속에서. 기능 회귀 우선. |

## Commit message (executor 참고용)

```
feat(torch): act_core C1 — imports/helpers/ReLU/PReLU/FP8 LUT migration

- Add `import torch` + `_as_fp32` / `_to_torch_fp16` helpers
- ReLU/PReLU: drop @njit/_impl/_njit split, use torch.relu / torch.where
- FP8 LUTs: storage np.ndarray → torch.Tensor (GTX-custom math preserved)
  - Negative-zero bit pattern (0x8000) explicit
- Add e4m3 dual-path codec (fp8_e4m3_to_fp16 / fp16_to_fp8_e4m3)
- Transcendentals/Pool/CVT bodies unchanged (numba/numpy still active there)

Verified: import smoke + FP8 LUT consistency + RELU/LEAKY_RELU/GELU vendor
regression strict PASS (ULP=1)

Refs: Phase 1 (53eb670) torch hybrid pattern; C1 of 4-chunk act_core migration
```
