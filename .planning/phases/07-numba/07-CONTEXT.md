# Phase 7: Numba Dynamic Optimization - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7은 **v1 ship gate의 마지막 가속 레이어**다. P6 회귀가 strict-mode로 그린이 된 시점에서 이미 검증된 stateless NumPy 커널(`gemm_core`, `vec_core`, `act_core`)에 numba `@njit`을 얹어 회귀 시간 예산을 wall-clock 5× 이상 단축하면서, **bit-exact 보장과 fallback 안전망을 동시에 유지**한다. 구체적으로:

1. **JIT boundary는 stateless cores 한정** — `gemm_core.py` (3-loop FP32 accumulate), `vec_core.py` 7 kernels (sasmd / dot / vsum / clamp / accum_v / arange_v / format), `act_core.py` 7 활성화 + 2 pool + 9 cvt 합쳐 약 25개. 모두 P4/P5 단계에서 이미 `npu`/`proc`/`insn` 의존 zero로 설계 완료(docstring: "P7 numba @njit boundary"). engine layer(mm_engine/vec_engine/act_engine)는 spike pybind11 객체 의존으로 numba 호환 불가 → JIT 경계 밖.

2. **Acceptance gate = vendor 103-op sweep 풀 회귀** — P6에서 코어 op 셋 ~10-20개로 deferred됐던 "vendor 98개 op 디렉토리 풀 sweep"을 P7이 흡수한다. `vendor/gtx_cpp_reference/test/<OP>/n1s16/data/` 디렉토리 ~98-103개를 모두 strict-mode로 통과시키되, golden 자산이 없는 op은 graceful skip. 진입 조건은 P6 그린.

3. **Wheel 배포 = base + extras hybrid** — base `spike-*.whl`은 NumPy-only 그대로(P1-P6 baseline), `pip install spike[fast]`로만 numba 설치. lazy import + auto fallback으로 사용자가 numba 미설치라도 P7 코드는 NumPy 메서드로 투명 동작. base wheel 50MB cap은 유지, extras transitive size는 무제한 허용.

4. **Bit-exactness 절대 양보 불가** — `fastmath=False` (numba 기본) + explicit FP32 Python for-loop 보존. `np.dot`/`np.matmul`/`np.einsum`/`np.sum`(FP16 위) 등 BLAS pairwise summation 사용 금지(P4 RESEARCH lock-in: 16×16×16에서 4 ULP / 0.0078 abs drift 41/500 trials). per-kernel ULP-0 parity 단위 테스트가 모든 25개 kernel을 강제.

5. **REQUIREMENTS.md `Out of Scope` 재문구** — 기존 "numba / cython / JAX / torch / scipy ... 추가 시 wheel 빌드 복잡도/사이즈 폭증"을 "v1 hard dependency 제외 (Phase 7의 optional `spike[fast]` extras는 허용)"로 조절. ROADMAP P7과의 충돌 해결.

다음 모두는 **Phase 7 비범위(out-of-scope)** — 다른 페이즈/마일스톤 또는 Claude 판단:

- **engine layer (mm/vec/act_engine) JIT 가속** → v2. `proc`/`insn` pybind11 객체 의존 + dataclass/dict 사용은 numba 비호환.
- **Cython AOT / C extension / PyPy** → 거부. PROJECT.md "C++ 추가 코드 금지" + cibuildwheel 파이프라인 안정 우선.
- **fastmath=True** → 거부. bit-exact 위반.
- **numba.pycc AOT compile to .so** → 거부. deprecated + manylinux 재현성 이슈.
- **mxe_accum FP32 state numba 통합** → engine layer에 속함, JIT 경계 밖.
- **CUDA / GPU acceleration** → PROJECT.md Out of Scope (v2).
- **벤치마크 history tracking (asv)** → 과도. pytest-benchmark가 P7 default.

</domain>

<decisions>
## Implementation Decisions

### Scope & wheel 배포 전략 (D-01 ~ D-04)

- **D-01:** **Phase 7 = v1 ship gate 내부.** P6 회귀가 strict-mode 그린이 되는 시점에 P7 진입, P7 완료 후 v1.0 release.
  - **이유:** 사용자가 "v1 ship gate 내부"를 명시 선택. P4/P5 docstring이 이미 "P7 numba @njit boundary"를 가정하고 코드를 작성해 둠 → v2로 미루면 코드 ↔ 의도 drift.
  - **REQ 수정 동반:** D-04와 함께 REQUIREMENTS.md `Out of Scope` numba 항목을 재문구.

- **D-02:** **Lazy import + auto NumPy fallback.** numba 부재 시 P7 hot kernel은 자동으로 P4/P5의 NumPy 메서드로 fallback. 사용자가 numba 설치 안 해도 완전 동작 (투명).
  - 코드 패턴 (plan-stage에서 정확화):
    ```python
    try:
        from numba import njit
        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        def njit(*args, **kwargs):
            def decorator(fn): return fn
            return decorator if not callable(args[0]) else args[0]
    ```
  - **이유:** zero-friction UX. base wheel = NumPy-only, P7 전혀 없는 사용자도 깨짐 zero. 회귀 .elf은 NumPy fallback 경로에서도 strict-mode pass(이미 P6 검증됨).

- **D-03:** **단일 wheel + `[project.optional-dependencies]` extras.** `pyproject.toml`에 `[project.optional-dependencies]` 섹션 신설 + `fast = ["numba>=0.59"]` (정확 버전은 plan-stage에서 manylinux2014_x86_64 + cp310-cp312 호환 확인 후).
  - 사용자 설치: `pip install spike` (base) → NumPy-only, `pip install spike[fast]` → numba 가속 활성화.
  - **이유:** cibuildwheel 매트릭스 zero impact (extras는 wheel 빌드 영향 없음). base wheel 50MB cap 안전.

- **D-04:** **REQUIREMENTS.md `Out of Scope` numba 항목 재문구.**
  - 변경 전: "numba / cython / JAX / torch / scipy — NumPy 단독으로 회귀 시간 예산 충족. 추가 시 wheel 빌드 복잡도/사이즈 폭증"
  - 변경 후 (제안): "numba는 v1 hard dependency 제외 (Phase 7의 optional `spike[fast]` extras 통한 lazy 가속은 허용). cython / JAX / torch / scipy는 v1 전 영역에서 hard 또는 optional dep 모두 제외."
  - **이유:** ROADMAP P7 ↔ REQ 충돌 해결. 다른 라이브러리(cython 등)의 거부는 유지. P7 plan-stage 또는 Plan 01 첫 스텝에서 REQ 패치.

### Library 선택 + JIT 적용 범위 (D-05 ~ D-08)

- **D-05:** **라이브러리 = numba (LLVM JIT, ≥0.59).** ROADMAP 명시 + P4/P5 docstring 준비 완료 + cp310-cp312 + manylinux2014 호환.
  - **거부 대안:**
    - Cython AOT → cibuildwheel 파이프라인 재구성 비용. PROJECT.md "C++ 추가 코드 금지" 원칙 회색 영역.
    - C extension (pybind11) → PROJECT.md Constraints "C++ 추가 코드 금지" 직접 위반.
    - PyPy → wheel 호환성 + manylinux2014 매트릭스 깨짐.
    - numba + Cython hybrid → 이중 파이프라인 복잡도 정당화 부족.

- **D-06:** **JIT 적용 범위 = stateless cores 한정.** 정확 대상:
  - `riscv/gtx/gemm_core.py`: `gemm_core`, `gemm_reduce_sum_a`, `gemm_dot` (3 kernels)
  - `riscv/gtx/vec_core.py`: `sasmd_kernel`, `dot_kernel`, `vsum_kernel`, `clamp_kernel`, 그리고 P5에서 추가된 부속 kernel (~7개; plan-stage에서 vec_core.py 전수 식별)
  - `riscv/gtx/act_core.py`: `relu`, `prelu`, `gelu`, `tanh_act`, `sigmoid`, `softmax`, `esum` (7 활성화) + `pool_max`, `pool_avg` (2 pool) + 7+ format_cvt 함수 (FP16↔FP32, FP16↔FP8 LUT 활용, FP16↔INT8, FP16↔INT32 등; plan-stage에서 act_core.py 전수 식별).
  - **총 약 25개 kernel.** (정확 카운트는 plan-stage에서 vec_core.py + act_core.py 전수 검사 후 lock-in.)
  - **engine layer 비포함:** mm_engine/vec_engine/act_engine는 `proc`/`insn` pybind11 객체 + dataclass + dict + spike state lookup 의존 → numba 비호환. JIT 경계는 stateless boundary에서 자연 stop.
  - **이유:** P4 D-01 / P5 D-01의 stateless 설계가 이미 P7 boundary로 만들어짐. 최대 ROI + 명확한 경계 + 기존 코드 변경 면적 최소.

- **D-07:** **JIT signature = lazy first-call dispatch.** `@njit(cache=True)`만 작성. signature 명시 없음 → 첫 호출에서 type 추론 + 컴파일.
  - **이유:** signature 명시(`@njit("f4[:,::1](f2[:,::1], f2[:,::1])")`)는 type drift 차단에 유리하지만, FP16/FP32/INT8/INT32 + scalar 변형 + 1D/2D shape × 25 kernel = 시그니처 폭발. lazy 추론이 25 kernel 다 처리하는 것이 enkele 설정 단순.
  - **risk mitigation:** D-12 per-kernel ULP-0 parity 테스트가 type drift 자동 검출.

- **D-08:** **컴파일 캐싱 = `@njit(cache=True)` 디스크 자동.** numba 기본 `__pycache__/<module>.<func>-<hash>.nbi`/`.nbc`.
  - 첫 import (after cache miss) → 컴파일 후 디스크 lock-in. 이후 import → cache hit, 실행 즉시.
  - **이유:** 사용자 첫 .elf 회귀에서 컴파일 오버헤드 한 번 발생, 이후는 zero. CI 첫 실행도 동일 — caching 전략으로 충분.
  - **재고 시점:** plan-stage benchmark에서 첫 컴파일 시간이 5분 이상이라면 import-time eager warmup 추가 검토.

### Bit-exactness 보장 + Fallback (D-09 ~ D-12)

- **D-09:** **`fastmath=False` + explicit FP32 for-loop 보존.** numba 기본값 사용 (fastmath=False, error_model='numpy'). `gemm_core.py:73-79` 등 explicit Python `for k in range(K): s += np.float32(A[i,k]) * np.float32(B[k,j])` 패턴 그대로.
  - **금지 사항:** `np.dot` / `np.matmul` / `np.einsum` / `np.sum`(FP16) → BLAS pairwise summation drift (P4 RESEARCH "np.matmul Bit-Exactness" 41/500 trials 4 ULP / 0.0078 abs).
  - **이유:** PROJECT.md Core Value (bit-exact w/ C++) 절대 보장. fastmath=True는 결격.
  - **검증:** D-12 per-kernel ULP-0 parity가 모든 25 kernel을 잠금.

- **D-10:** **Acceptance gate = vendor `test/<OP>/n1s16/` 풀 sweep.** ~98-103개 op 디렉토리(plan-stage에서 vendor 정확 카운트):
  - 각 op 디렉토리에 `data/<kernel>_ref.txt` 또는 `_result.hex` 자산 보유 여부 확인.
  - **자산 보유 op:** P6 변환 스크립트 lineage(`scripts/import_vendor_golden.py`) 확장 → `tests/gtx/data/golden/<op>.hex`로 일괄 변환 + git lock-in.
  - **자산 미보유 op:** `pytest.skip(reason=f"vendor {op} 자산 없음")` graceful skip.
  - **strict-mode 통과 기준:** P6 D-09 `tests/gtx/test_regression_fw_full.py` parametrize 롤 확장 → `test_regression_fw_full_sweep.py` 신규 (또는 기존 fw_full.py 확장). 모든 자산 보유 op이 `compare_hex(strict=True)` PASS.
  - **이유:** 사용자가 "test/{OP} 103개 모두 통과 (데이터가 없는 경우 skip)"으로 명시. P6 D-07 "코어 op 셋 ~10-20개" + P6 deferred "vendor 98개 풀 sweep → v1.x patch 또는 v2"가 P7으로 흡수됨.
  - **plan-stage 정확화 필요:** vendor 디렉토리 정확 카운트(P6 CONTEXT는 98로, 사용자는 103 언급). 디렉토리 sweep 후 자산 보유 분포 측정.

- **D-11:** **Fallback 관리 = same module dual export + 자동 dispatcher.** 각 stateless core 모듈이 NumPy 원본 + numba JIT 버전을 공존:
  ```python
  # gemm_core.py (예시)
  def gemm_core_numpy(A, B, ...): ...  # 원본 NumPy 메서드 (P4 lineage 그대로)

  if HAS_NUMBA:
      gemm_core_njit = njit(cache=True)(gemm_core_numpy)  # 또는 별도 정의
      gemm_core = gemm_core_njit
  else:
      gemm_core = gemm_core_numpy
  ```
  - **이유:** 단일 파일에서 두 버전 명확. 외부 호출자(mm_engine 등)는 `from gemm_core import gemm_core` 단일 import만 알면 됨 — JIT 여부 투명.
  - **plan-stage 정확화:** numba 데코레이터를 재호출(`njit(cache=True)(fn)`) vs 별도 정의 둘 중 어느 패턴이 25 kernel에 더 일관적인지 검증.

- **D-12:** **per-kernel ULP-0 parity 단위 테스트.** `tests/gtx/test_njit_parity.py` 신규 — 25 kernel 모두 NumPy vs JIT delta_ulp == 0:
  ```python
  @pytest.mark.parametrize("kernel_name", ALL_NJIT_KERNELS)  # 25 entries
  def test_kernel_parity(kernel_name):
      numpy_fn, njit_fn = NUMPY_REGISTRY[kernel_name], NJIT_REGISTRY[kernel_name]
      inputs = generate_test_inputs(kernel_name)  # fixed seed for reproducibility
      out_numpy = numpy_fn(*inputs)
      out_njit = njit_fn(*inputs)
      assert np.array_equal(out_numpy.view(np.uint16), out_njit.view(np.uint16)), \
          f"{kernel_name}: delta_ulp != 0"
  ```
  - **이유:** D-07 lazy first-call이 type drift 위험을 수반하지만 이 테스트가 자동 검출 + 정확한 kernel 식별. 실패 시 정확히 어느 kernel이 깨졌는지 즉각 보고.
  - **plan-stage 정확화:** generate_test_inputs는 P5 VRF-02 oracle 패턴 직접 활용 (fixed-seed FP16 random).

### 성능 목표 + acceptance gate (D-13 ~ D-16)

- **D-13:** **성능 목표 = wall-clock 5× 이상.** vendor 103-op sweep 전체 walltime 기준, P6 NumPy baseline 대비.
  - **베이스라인 측정:** P6 종료 시점 또는 P7 plan-stage 첫 스텝에서 `pytest tests/gtx/test_regression_fw_full_sweep.py --no-cov` walltime을 P6 baseline으로 lock-in.
  - **P7 통과 기준:** 같은 sweep walltime이 baseline의 1/5 이하.
  - **이유:** 5×는 ROI 합리적 + numba가 small kernel에서 부진할 수 있는 vec/act까지 평균 끌어올리기 가능. gemm_core (3-loop)가 numba 핵심 사용자.
  - **재고 시점:** plan-stage P6 baseline 측정 후 5× 비현실적이면 사용자와 재논의. 단, 현 시점은 5× lock-in.

- **D-14:** **측정 도구 = pytest-benchmark.** `tests/gtx/test_njit_perf.py` 신규.
  - per-kernel benchmark + 회귀 .elf walltime 둘 다 추적.
  - dev-only dependency: `pyproject.toml [project.optional-dependencies]` `dev = ["pytest-benchmark>=4.0", ...]` (P6 lineage 확장).
  - **이유:** pytest 인프라 자연 통합 + CI 재현성. base wheel 영향 zero (dev extras only).

- **D-15:** **Wheel size 정책 = base 50MB cap만 유지, extras transitive size 비고려.** 사용자 명시: "wheel 종속 설치 고려하지말고 진행".
  - base `spike-*.whl`: ≤50MB (PROJECT.md ROADMAP success #4 그대로).
  - `pip install spike[fast]` 후 transitive size: 무제한 허용. numba (~10MB) + llvmlite (~30MB) + 기타 의존 = 약 50-80MB 추가 가능.
  - PROJECT.md "wheel size ≤50MB" 문구는 base wheel에 한정한다는 명시 추가 (P7 plan-stage 또는 Plan 첫 스텝에서 PROJECT.md 동기화).
  - **이유:** 사용자 결정. extras는 opt-in이므로 사용자가 size 의식 후 선택.

- **D-16:** **3-tier 테스트 구조.**
  - **Tier 1 — `test_njit_parity.py`** (D-12): 25 kernel × NumPy vs JIT ULP-0. JIT 정확성 가드.
  - **Tier 2 — `test_regression_fw_full_sweep.py`** (D-10): vendor 103-op 디렉토리 풀 strict-mode sweep + skip-on-missing-data. JIT × .elf end-to-end.
  - **Tier 3 — `test_njit_perf.py`** (D-14): pytest-benchmark per-kernel + walltime. wall-clock 5× 보증.
  - 독립 실행 가능 (한 tier 실패가 다른 tier 차단 zero). CI 리포트에서 어느 tier 어느 op 실패인지 즉각 식별.
  - **이유:** P5/P6 lineage (parametrize 롤 + 5-tier graceful-skip 패턴)와 일관. 3-tier 구조로 acceptance signal 분리: 정확성(Tier 1+2) vs 성능(Tier 3).

### Claude's Discretion

다음은 implementation detail로 Claude 판단 (research/plan 단계에서 정확화):

- **`@njit` decorator 적용 패턴** (D-06/D-07):
  - 직접 적용 (`@njit(cache=True)\ndef gemm_core(...): ...`) vs 재호출(`gemm_core_njit = njit(cache=True)(gemm_core_numpy)`).
  - dual export 구조에 어느 쪽이 25 kernel에 일관적인지 plan-stage 검증.
- **vec_core.py / act_core.py 정확 kernel 카운트** (D-06): plan-stage에서 두 모듈 전수 검사. ~25 추정이 정확값으로 lock-in.
- **vendor `test/<OP>/` 정확 디렉토리 카운트** (D-10): P6 CONTEXT는 98로, 사용자는 103 언급. plan-stage에서 `find vendor/gtx_cpp_reference/test -mindepth 1 -maxdepth 1 -type d | wc -l`로 정확화.
- **vendor 자산 → `.hex` 변환 스크립트 확장** (D-10): P6 plan-stage `scripts/import_vendor_golden.py`가 코어 셋 ~10-20개 처리한 것을 풀 103개로 확장. 누락 op 식별 + skip 자동화.
- **NumPy fallback 활성화 분기점** (D-02): 모든 stateless core가 module-top에서 `HAS_NUMBA` 검사 vs 중앙 `riscv/gtx/_jit.py` 모듈에서 단일 정의. plan-stage에서 import 의존 그래프 검토 후 결정.
- **첫 컴파일 시간 측정 + eager warmup 트리거** (D-08): plan-stage benchmark에서 25 kernel 첫 컴파일 누적 시간이 일정 임계(예: 30초) 초과 시 import-time pre-compile 추가 검토.
- **numba 버전 핀** (D-03): `numba>=0.59`만 명시 vs 정확 핀 `numba==0.59.x`. cp310-cp312 + manylinux2014 호환성 매트릭스 검증 후 lock-in.
- **`@njit(parallel=True)`** (D-07/D-08): 25 kernel 중 어느 것이 parallel reduction 이득 있는지 plan-stage benchmark. small kernel(<1024 elem)은 thread spawn cost 손해, large gemm + 큰 vec sum은 이득. 기본 `parallel=False`로 시작 후 hot path만 fine-tune.

### Folded Todos

None — `gsd-tools todo match-phase 7`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 7 핵심 ROADMAP / REQ 자료

- `.planning/ROADMAP.md` Phase 7 섹션 — "정상 동작 확인 후 numba 등 동적 최적화 라이브러리로 핫스팟 가속 (P6 회귀 그린이 진입 조건)". P7 진입 조건 명시.
- `.planning/REQUIREMENTS.md` `Out of Scope` 표 — "numba" 항목 재문구 대상 (D-04 lineage). D-04 완료 시점에 P7 plan-stage 또는 Plan 01에서 패치.
- `.planning/PROJECT.md` Constraints — "Performance: NumPy 백엔드 가정. ... 성능 핫스팟이 발견되면 v2에서 cython/C 확장 검토" (D-05 numba 채택은 이 절을 P7으로 advance). "wheel size ≤50MB" (D-15: base wheel 한정으로 명시 추가).

### Vendor C++ source (회귀 .elf + golden 자산)

- `vendor/gtx_cpp_reference/test/<OP>/n1s16/` — **D-10 핵심 source.** 약 98-103개 op 디렉토리 (plan-stage에서 정확 카운트). 각 디렉토리는 `n1s16_<op>.c` (펌웨어 source) + `data/<kernel>_ref.txt` (golden) + `data/<kernel>_input*.txt` (input) 패턴.
- `vendor/gtx_cpp_reference/test/run_tests_n1s16.sh` — vendor의 회귀 빌드 + 실행 스크립트. P7 plan-stage에서 변환 스크립트 (`scripts/import_vendor_golden.py` 확장)의 vendor flow 참조.
- `vendor/gtx_cpp_reference/test/generate_n1s16_tests.py` — vendor input/ref 생성 스크립트. P7에서 신규 op 추가 시 참조.
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — vendor-side guidelines (`GTX_USE_OMP` / `GTX_USE_CUBLAS` 무시, P5/P6 lineage 그대로).

### Prior phase contexts (decision precedent)

- `.planning/phases/04-mm-subsystem/04-CONTEXT.md` — **D-01 (gemm_core stateless 설계 = P7 numba boundary), D-13 (`_verify_minimal` 핵심), RESEARCH np.matmul Bit-Exactness 거부 lineage.** P7 D-09의 직접 부모 (BLAS pairwise summation drift = bit-exact 위반).
- `.planning/phases/04-mm-subsystem/04-VERIFICATION.md` — `proc.state` is property (pybind11) 확인. P7 stateless boundary 외 어떤 코드도 spike-bound면 numba 비호환.
- `.planning/phases/05-vec-act-pool/05-CONTEXT.md` — **D-01 (vec_core stateless 3-way module split), D-02 (act_core bundled module + FP8 LUT), D-09~D-12 (VSUM/DOT explicit FP32 for-loop FP32 accumulate 규약).** P7 D-06 (JIT 적용 범위) + D-09 (FP32 누적 보존) 직접 mirror. P5 deferred → P7 흡수.
- `.planning/phases/06-verification-wheel/06-CONTEXT.md` — **D-07 (코어 op 셋 ~10-20개), D-09 (parametrize 롤), D-10 (vendor `_ref.txt` golden source), D-12 (vendor C++ 빌드 dev-stage only). Deferred (numba @njit → P7, vendor 98개 풀 sweep → v1.x/v2).** P7 D-10이 P6 deferred 흡수.
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-07/D-08 (NumPy ≥ 2.0 + cp310-cp312 cibuildwheel). P7 D-03 extras 추가가 cibuildwheel zero-impact임을 보장하는 baseline.

### Code context (P7 JIT boundary 코드)

- `src/main/python/riscv/gtx/gemm_core.py` — **D-06 핵심 boundary.** docstring 명시: "P7 numba `@njit` boundary." 3 kernel: `gemm_core` (3-loop FP32), `gemm_reduce_sum_a`, `gemm_dot`. 모두 explicit Python for-loop 사용 (P4 D-01).
- `src/main/python/riscv/gtx/vec_core.py` — **D-06 핵심 boundary.** docstring 명시: "P7 numba @njit boundary." 7+ kernels: `sasmd_kernel`, `dot_kernel`, `vsum_kernel`, `clamp_kernel` 등 (plan-stage 전수 카운트).
- `src/main/python/riscv/gtx/act_core.py` — **D-06 핵심 boundary.** docstring 명시: "Pure stateless ACT/Pool/Format kernels + FP8 LUTs." 7 활성화 + 2 pool + 9 cvt + FP8 LUT (256B + 64KB precomputed). FP8 LUT는 module import 시점에 numpy로 precompute → numba 함수에서는 array view로만 접근.
- `src/main/python/riscv/gtx/mm_engine.py` / `vec_engine.py` / `act_engine.py` — **JIT boundary 밖.** `proc`/`insn`/dataclass/dict 의존 → numba 비호환. 이들 engine은 stateless core 호출자로만 동작. P7에서 변경 zero.
- `tests/gtx/_oracles.py` — P5 VRF-02 oracle suite. P7 D-12 per-kernel parity test의 input generation 패턴 직접 활용.
- `tests/gtx/test_regression_fw_full.py` (P6) — **D-10 직접 확장.** P6 코어 op 셋 parametrize 롤 → P7 vendor 103-op sweep으로 확장 (`test_regression_fw_full_sweep.py` 신규 또는 기존 확장).
- `scripts/import_vendor_golden.py` (P6 P03 산출) — **D-10 직접 확장.** 코어셋 ~10-20개 변환 → 풀 103개 변환 + 자산 미보유 op 자동 skip 신호.
- `pyproject.toml` — **D-03 추가 위치.** `[project.optional-dependencies]` 섹션 신설 + `fast = ["numba>=0.59"]`. P6에서 추가된 `[project.scripts] pyspike-verify`와 별도 섹션.

### 신규 P7 코드 (plan-stage에서 lock-in)

- `tests/gtx/test_njit_parity.py` (신규) — D-12 per-kernel ULP-0 parametrize.
- `tests/gtx/test_regression_fw_full_sweep.py` (신규 or P6 fw_full.py 확장) — D-10 vendor 103-op sweep with skip-on-missing-data.
- `tests/gtx/test_njit_perf.py` (신규) — D-14 pytest-benchmark per-kernel + walltime.
- `riscv/gtx/_jit.py` (검토 후보; D-02 plan-stage) — `HAS_NUMBA` 감지 + njit shim의 중앙 정의 위치 후보. 또는 각 core 모듈에 module-top 검사로 분산. plan-stage 결정.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`gemm_core.py:gemm_core` 3-loop** (P4 D-01) — docstring "P7 reactivates BLAS-equivalent perf via @njit" 명시. NumPy → @njit decorator만 추가하면 즉시 hot kernel.
- **`vec_core.py:dot_kernel` / `vsum_kernel` explicit FP32 for-loop** (P5 D-09~D-12) — 동일 패턴. for-loop가 numba에서 가장 잘 컴파일.
- **`act_core.py:softmax` / `esum` explicit FP32 sum** (P5 RESEARCH §VSUM/DOT Precision) — 동일 패턴.
- **`tests/gtx/test_regression_fw_full.py` parametrize 롤** (P6 D-09) — D-10 풀 sweep의 직접 확장 대상.
- **`scripts/import_vendor_golden.py` 변환 스크립트** (P6 P03) — D-10 vendor 103-op asset import의 직접 확장 대상.
- **`tests/gtx/_oracles.py` 32-op verify_ref oracle** (P5 VRF-02) — D-12 parity test input generation 재활용.
- **`tests/gtx/conftest.py` `_RISCV_AVAILABLE` 패턴** — P7 신규 테스트도 동일 fixture (`pytest --noconftest` mode 위해 module-level self-detect도 P5 lineage대로).

### Established Patterns

- **stateless core ↔ engine layer 분리** (P4 D-01 / P5 D-01 / D-02) — D-06 JIT boundary가 자연스럽게 stateless line으로 정착.
- **Explicit FP32 for-loop = bit-exact insurance** (P4 RESEARCH np.matmul drift / P5 RESEARCH VSUM/DOT) — D-09 numba 적용 시에도 그대로 보존.
- **parametrize 롤로 acceptance gate stress** (P6 D-09) — D-10 / D-12 / D-14 모두 동일 패턴.
- **`pyproject.toml [project.optional-dependencies]` 추가 패턴** — P6 D-02 `[project.scripts]`와 같은 pyproject 표면. cibuildwheel zero-impact (P1 D-08 lineage).
- **`HAS_<feature>` module-top 검사 + graceful import** (P3 03-01 D-1 `_RISCV_AVAILABLE` 직계 lineage) — D-02 numba lazy import + auto fallback.

### Integration Points

- **`pyproject.toml`** — D-03 `[project.optional-dependencies]` `fast = ["numba>=0.59"]` 추가. D-04 REQ 패치는 별도 파일이지만 plan-stage에서 동기화.
- **`src/main/python/riscv/gtx/gemm_core.py`, `vec_core.py`, `act_core.py`** — D-06 JIT 적용 대상. 변경 면적: import 줄 + (decorator 추가 OR 재호출 패턴) + dual export 모듈-수준 alias.
- **`src/main/python/riscv/gtx/mm_engine.py`, `vec_engine.py`, `act_engine.py`** — D-06에서 명시적 비포함. 호출자 측면에서 JIT/NumPy 전환 투명 (단일 import만 사용).
- **`tests/gtx/test_regression_fw_full.py`** — D-10 확장 대상 (또는 신규 _sweep.py). 변경 시 P6 acceptance와 호환 유지 필수.
- **`scripts/import_vendor_golden.py`** — D-10 확장 대상. P6 코어셋 lineage + 자산 미보유 op skip 시그널 추가.
- **`.planning/REQUIREMENTS.md`** — D-04 `Out of Scope` 표 numba 항목 재문구 (P7 Plan 01 또는 첫 스텝).
- **`.planning/PROJECT.md`** — D-15 "wheel size ≤50MB" → "base wheel size ≤50MB" 명시화 (P7 Plan 01 또는 첫 스텝).

### Creative Options Enabled / Constrained

- **Lazy fallback enables zero-friction UX** (D-02) — base wheel 사용자도 P7 코드가 ImportError 없이 NumPy로 동작. CI / cibuildwheel 매트릭스 별도 numba 검증 부담 zero.
- **JIT boundary가 P4/P5 단계에서 미리 마련됨** (D-06) — 25 kernel 모두 docstring에 "P7 numba @njit boundary" 명시 → 코드 reorganize 비용 zero, decorator만 추가.
- **vendor 자산 직접 차용** (D-10) — 사용자 vendor가 이미 ISS run 결과를 ref로 lock-in. P7 신규 binary 빌드 zero. P6 D-10/D-12 lineage 그대로 확장.
- **Constraint: bit-exact 절대 양보 불가** — fastmath=True 거부 (D-09). 5× speedup 미달 시 더 공격적 최적화 유혹 있겠지만 거부.
- **Constraint: extras 통한 base wheel 격리** (D-03) — base wheel cap 50MB / cibuildwheel 매트릭스 zero-regression. extras 의존이 클수록 사용자 선택 가시.
- **Constraint: stateless boundary 고정** (D-06) — engine layer JIT 시도 거부 (numba 비호환 + 변경 면적 폭증). 대신 stateless 25개에 집중 → 단순 명료.

</code_context>

<specifics>
## Specific Ideas

### "P4/P5 docstring이 P7을 미리 예고함" 패턴

`gemm_core.py`/`vec_core.py`/`act_core.py` 모두 module-level docstring에 "P7 numba @njit boundary"를 명시. 이는 P4/P5 단계에서 stateless 설계를 의식적으로 도입한 결과 — P7은 자연스러운 "decorator만 추가" 작업이 됨. 변경 면적이 25 kernel × ~3-5 LOC = 약 75-125 LOC. 보일러플레이트 패턴이 일관 → review/audit 단순.

### "P6 deferred → P7 흡수"의 이중 의미

P6 CONTEXT D-07 deferred 두 가지 모두 P7으로 흡수됨:
1. **Numba @njit 동적 최적화** (P5 → P7 명시 deferred). P7 D-05/D-06이 직접 처리.
2. **vendor 98개 op 디렉토리 풀 sweep** (P6 → v1.x/v2 deferred였음). P7 D-10 acceptance gate가 이를 풀 sweep 형태로 흡수.

이는 사용자 "test/{OP} 103개 모두 통과 (데이터가 없는 경우 skip)" 답변에서 발생한 P6 → P7 boundary shift. P6 plan은 코어 op 셋 ~10-20개로 lock-in했으나, P7이 acceptance gate를 v1 sunset으로 끌어올림.

### vendor 디렉토리 카운트 불일치 (98 vs 103)

P6 CONTEXT는 "vendor 98개 op 디렉토리"로 표기, 사용자 답변은 "103개". 5개 차이는 plan-stage에서 정확화:
```bash
find vendor/gtx_cpp_reference/test -mindepth 1 -maxdepth 1 -type d -name "[A-Z]*" | wc -l
```
또는
```bash
ls vendor/gtx_cpp_reference/test/ | grep -E '^[A-Z]' | wc -l
```
실제 카운트 후 D-10 acceptance gate에 lock-in. 자산 보유 op 분포(`<op>/n1s16/data/*ref.txt` 또는 `*result*.hex` 존재 여부) 동시 측정.

### numba IEEE 754 보장 메커니즘

numba 0.59+ 기본:
- `fastmath=False` (default): IEEE 754 conformant. associative reordering 금지. 사용자 explicit FP32 for-loop 그대로 컴파일.
- `error_model='numpy'` (default): division by zero, sqrt(-1) 등이 NumPy처럼 inf/nan 반환 (raise 안 함). FP8 codec / vec_core div-by-zero (P5 D-?) 와 호환.
- `cache=True`: 디스크 캐시 자동 — 첫 호출 후 두 번째 import부터 재컴파일 없음.

이 세 default 조합이 D-09 (bit-exact 보장) + D-08 (디스크 캐시) 직접 충족. P7 코드는 추가 옵션 명시 없이 `@njit(cache=True)`만 작성하면 됨.

### `@njit` decorator 직접 vs 재호출 패턴 (plan-stage 결정)

**Option A — 직접 decorator** (plan-stage 추천 가능):
```python
# gemm_core.py
def _gemm_core_impl(A, B, has_bias, bias_fp32):
    # 원본 NumPy 구현 (P4 lineage 그대로)
    ...

if HAS_NUMBA:
    gemm_core = njit(cache=True)(_gemm_core_impl)
else:
    gemm_core = _gemm_core_impl
```
장점: 단일 함수 정의 + 한 줄로 dispatch. 단점: numba가 일부 NumPy API(특정 `.astype()` 변형 등) 비호환 시 조용히 fallback 안 되고 컴파일 fail.

**Option B — 별도 njit 함수**:
```python
# gemm_core.py
def gemm_core_numpy(A, B, has_bias, bias_fp32): ...

def _gemm_core_njit_impl(A, B, has_bias, bias_fp32):
    # numba-friendly로 약간 재작성 (필요 시)
    ...

gemm_core = njit(cache=True)(_gemm_core_njit_impl) if HAS_NUMBA else gemm_core_numpy
```
장점: numba 비호환 API를 piecewise 회피 가능. 단점: 두 정의 동기화 부담.

plan-stage에서 25 kernel 중 numba 비호환 NumPy API 쓰는 kernel 식별 → Option A 가능 카운트 vs Option B 필요 카운트 측정.

### "Wall-clock 5×" 측정 정확화

P7 acceptance Tier 3:
```python
# tests/gtx/test_njit_perf.py (예시)
@pytest.mark.benchmark(group="gemm")
def test_gemm_core_njit_speedup(benchmark, baseline_walltime):
    inputs = make_gemm_inputs(M=16, K=16, N=16)
    result = benchmark(gemm_core, *inputs)  # JIT version (HAS_NUMBA=True)
    # baseline_walltime은 fixture가 P6 NumPy-only run 시간 회귀로부터 제공
    assert benchmark.stats['mean'] * 5 <= baseline_walltime, \
        f"5x speedup 미달: {benchmark.stats['mean']:.4f}s vs baseline {baseline_walltime:.4f}s"
```
plan-stage에서 baseline_walltime fixture 정의 + cibuildwheel CI 매트릭스 (cp310-cp312 × 4 hardware)에서 5× 일관 보장 검증.

### `pip install spike[fast]` 사용자 시나리오

```bash
# 새 사용자 (numba 미설치)
pip install spike
python -c "from riscv.gtx import GtxNpu; npu = GtxNpu(); ..."  # NumPy fallback 동작 (투명)

# 가속 활성화
pip install spike[fast]
python -c "from riscv.gtx import GtxNpu; npu = GtxNpu(); ..."  # numba @njit 활성화
```

P7 README 업데이트(plan-stage Plan 마지막 스텝 후보):
- "성능 가속" 섹션에 `pip install spike[fast]` 명시.
- "기본 설치는 NumPy로만 동작하며, 회귀 시간이 우려되면 fast extras로 numba 가속 활성화" 메시지.

</specifics>

<deferred>
## Deferred Ideas

### Out of P7 scope (explicit deferrals to other phases / milestones)

- **engine layer (mm_engine / vec_engine / act_engine) JIT 가속** → v2. `proc`/`insn` pybind11 객체 + dataclass + dict 의존 → numba 비호환. 우회하려면 spike state shim 필요 → 변경 면적 폭증.
- **Cython AOT / C extension 경로** → 거부 lock-in (D-05). PROJECT.md "C++ 추가 코드 금지" + cibuildwheel 파이프라인 안정 우선. v2에서도 재고 가능성 낮음.
- **PyPy 호환** → 거부 (D-05). manylinux2014 + cp310-cp312 fix.
- **CUDA / GPU acceleration** → PROJECT.md Out of Scope (v2 reconsider).
- **fastmath=True / FP 재결합 허용** → bit-exact 위반으로 영구 거부 (D-09). v2에서도 PROJECT.md Core Value 변경 없는 한 거부.
- **numba.pycc AOT compile to .so** → 거부 (D-07/D-08). deprecated + manylinux 재현성 이슈.
- **mxe_accum FP32 state numba 통합** → engine layer 영역 (D-06 boundary 밖). v2.
- **asv (airspeed velocity) benchmark suite** → 거부 (D-14). pytest-benchmark가 P7 default, asv는 과도.
- **`@njit(parallel=True)` 적극 사용** → plan-stage에서 hot path만 fine-tune (Claude's Discretion). 기본 `parallel=False`. 적극 사용은 v1.x patch 또는 v2.

### Within-domain ideas surfaced but not selected for discussion

- **PyArrow/Apache Arrow 통한 zero-copy view** — JIT 후속 가속 가능성. v2.
- **`spike[fast]` 외 `spike[bench]` extras 분리** — pytest-benchmark는 dev extras에 포함시켜 주력 사용자에 불노출 (plan-stage 정확화).
- **per-kernel 정확 numba 옵션 매트릭스** — `parallel`, `nogil`, `boundscheck=False` 등 25 kernel 별 fine-tuning. 기본 default 조합으로 시작, plan-stage benchmark 후 hot path만 추가.
- **GPU acceleration via numba.cuda** → 거부. PROJECT.md Out of Scope.
- **Multi-process pytest-xdist 통한 sweep 병렬** → P7 Tier 2 sweep 시간 단축. plan-stage 검토 가능. CI 매트릭스 영향 확인 필요.

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 7`은 매칭 0건.

### Defer to user follow-up

- **v1.0 ship announcement** — P7 완료 후 `pip install spike` + `pip install spike[fast]` 둘 다 검증된 상태에서 사용자가 직접 announce.
- **PROJECT.md / REQUIREMENTS.md 동기화** — D-04 (numba Out of Scope 재문구) + D-15 (50MB cap base 한정 명시)는 P7 Plan 01 또는 첫 스텝에서 처리. `/gsd:complete-milestone v1.0` 흐름에서 자동 처리도 가능.
- **README extras 안내 추가** — plan-stage Plan 마지막 또는 P7 종료 후 사용자 직접 추가.

</deferred>

---

*Phase: 07-numba*
*Context gathered: 2026-05-08*
