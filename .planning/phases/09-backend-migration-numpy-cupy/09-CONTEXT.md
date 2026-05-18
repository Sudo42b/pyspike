# Phase 9: Backend Migration — PyTorch → NumPy + CuPy opt-in - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning
**Milestone:** v1.1 — Post-Ship Polish (Backend migration follow-up)

<domain>
## Phase Boundary

`src/main/python/riscv/gtx/*`의 모든 `torch.Tensor` 사용을 `numpy.ndarray`로 치환하되
`xp` alias 패턴(`xp = numpy` 기본 / `xp = cupy` opt-in via `GTX_USE_CUDA=1`)으로
드롭-인 GPU 가속을 가능케 한다. 28개 P7 numba @njit 핫패스 커널은
`numba.guvectorize` universal source + target switching으로 `xp=cupy` 시
`@cuda.jit` 경로로 자동 컴파일된다. PyTorch는 런타임 의존성 + dev/test 모두에서
완전 제거하고, CuPy는 `pip install spike[cuda]` extras로만 노출한다.

**핵심 invariant (절대 위반 금지):**
1. **ABS strict byte-exact** — `test_vendor_op_sweep_strict[ABS]` 96 tiles × 196609 hex lines
   `compare_hex(strict=True)` PASS. Phase 8이 land한 baseline (94.82s on torch CPU).
2. **GELU/RELU/SIGMOID/TANH/SOFTMAX/ESUM strict** — P8 D-11 smoke set 5 op + ABS 총 6 op
   PASS 유지.
3. **Tile-2 unit test (P8 MTDMA-03)** — `tests/gtx/test_multi_tile_dma.py` GREEN 유지.
4. **LE byte-order assertion (D-17 from Phase 1)** — non-LE host tripwire 그대로.

**Out of scope (다른 페이즈 / v1.2):**
- 새로운 op 추가, 새로운 ISA 인코딩
- pybind11 트램폴린 (C++ binding 계층)의 `torch::Tensor` 사용 — Phase 9는 Python 측만
- Numba CUDA kernel **추가 최적화** (예: shared memory, warp shuffle, ElementwiseKernel) —
  v1.2 perf phase
- 28개 dual-impl 중 일부가 v1.1 milestone 범위 초과 시 **plan-stage에서 phase 분할
  옵션 검토** (예: P9 = numpy default 완료, P10 = cupy + cuda kernel)

</domain>

<decisions>
## Implementation Decisions

### Area 1: xp alias scaffold & GTX_USE_CUDA contract (D-01 ~ D-04)

- **D-01 xp alias 위치 = `config_params.py` 확장.**
  기존 `DEVICE` SSOT 단일 메커니즘 재사용. config_params.py:9-23의 cuda regression
  주석이 이미 "Future opt-in: GTX_USE_CUDA env-var gate"를 명시 — 그 future
  hook를 실제로 구현. import 경로 마이그레이션 중 한 파일만 쇼프레이스
  (`from ...config_params import xp`). 별도 `backend.py` 모듈 신설 안 함
  (중복 위험).

- **D-02 Backend resolve = import-time eager + frozen.**
  config_params.py 모듈 최상단에서 `xp = numpy if not GTX_USE_CUDA else _import_cupy()`
  단일 결정. memory.py의 `_L2_GLOBAL` 등 module-level allocation이 이미
  import-time이므로 eager와 자연스럽게 일치. 테스트 프로세스 경계로 설정 고정
  (lazy property → atexit ordering 버그가 테스트별로 다르게 나타날 위험 제거).

- **D-03 `GTX_USE_CUDA=1` 인데 cupy 미설치 = Fail-loud RuntimeError.**
  Silent fallback 금지. 260518-ffr regression(`torch.cuda.is_available()` 자동
  활성화로 5x ABS slowdown)이 정확히 silent auto-fallback으로 일어난 사건 —
  명시적 opt-in은 명시적으로 실패해야 함. 에러 메시지에 "`pip install spike[cuda]`"
  recovery hint 포함.

- **D-04 기존 `DEVICE` 심볼 = 제거 (clean cut).**
  torch 완전 제거가 목표 — `torch.device` 디포 자르기. config_params.py:25 +
  `__init__.py:88` 두 곳의 re-export 제거. 외부에서 `from riscv.gtx import DEVICE`
  쓰는 코드가 있으면 ImportError로 surface (numpy/cupy device 모델은 torch와
  의미론 다름 → 같은 이름 유지가 더 혼선). backwards-compat shim 금지 (CLAUDE.md
  "No backwards-compatibility shims" 원칙).

### Area 2: Migration strategy & PR shape (D-05 ~ D-08)

- **D-05 PR shape = Wave 구조 (4 wave).**
  - **Wave 0** (scaffold): config_params.py xp alias + `to_host()`/`to_device()` 헬퍼 +
    DEVICE 제거 준비. test infra (`tests/gtx/conftest.py`의 backend fixture).
  - **Wave 1** (저장소): `memory.py` (DDR + scratchpads module-level alloc) +
    `register_file.py` (SPR int64) numpy 포팅. RegisterFile bit-field op 검증.
  - **Wave 2** (연산): `ops/spr.py`, `ops/mm.py`, `ops/vec.py`, `ops/act.py` +
    engines (`dma_engine.py`, mm_engine, vec_engine, act_engine). FP8 LUT 빌드
    경로 (`act.py:45-117`)도 import-time numpy 캐스트.
  - **Wave 3** (마무리): `tloop_buffer.py`, `_verify.py`, `npu.py`,
    `__init__.py` torch import 제거. `tests/gtx/*.py` 3 파일 포팅. `pyproject.toml`에서
    torch 제거 + `[cuda]` extras 추가. CLAUDE.md "Dependencies" 섹션 업데이트.
  - Plan-stage가 각 wave를 PLAN으로 분해.

- **D-06 Dual-import = 허용되지만 최소화.**
  Wave 1/2 중간에는 일부 파일 numpy + 일부 torch가 임수적. 경계 함수는
  `numpy.ndarray ↔ torch.Tensor` 단방향 bridge (단 임시 — Wave 3에서 모두 제거).
  **각 wave 끝에 ABS strict byte-exact GREEN 보장 필수** (intermediate 상태도
  invariant 유지 — 회귀가 어느 wave에서 났는지 즉시 격리 가능).

- **D-07 Test gate (매 wave 끝) = 6 vendor op + tile-2 unit test.**
  - 6 op: ABS, GELU, RELU, SIGMOID, TANH, SOFTMAX (P8 D-11 smoke set 5 + ABS) —
    `uv run pytest tests/gtx/test_regression_fw_full_sweep.py -k 'ABS or GELU or RELU
    or SIGMOID or TANH or SOFTMAX' --no-cov -v`
  - tile-2: `uv run pytest tests/gtx/test_multi_tile_dma.py --no-cov -v` (P8 MTDMA-03).
  - PR final: 전체 84-op vendor sweep + tile-2 + ABS perf 재측정.

- **D-08 Perf budget = ABS strict ±10% (target window: 85–105s).**
  baseline 94.82s (commit 2b0c66e). Wave 3 끝에 재측정. 위반 시 plan-stage가
  hot-path 조사 (numba @njit 재적용 / cupy 경로 RawKernel / numpy strided op
  최적화). 80s 이하 (10%+ 개선)는 OK (torch dispatch overhead 제거 기대 효과).

### Area 3: CuPy device placement model (D-09 ~ D-12)

- **D-09 Scratchpads (L0/L1/L2) = GPU (cupy.ndarray) when xp=cupy.**
  현재 torch 경로와 동형 (DEVICE-resident). memory.py module-level `_L2_GLOBAL` 등이
  `xp.zeros(..., dtype=xp.uint8)` 패턴으로 import-time GPU 메모리 사전 확보.
  4-NEST × (L0 + L1 + L2) ≈ ~25 MB → 일반 GPU에서 부담 없음.

- **D-10 DDR = GPU (전체 통일).**
  현재 명시적 CPU(`_DDR_DEVICE = torch.device("cpu")`)를 v1.1에서 되돌림.
  **⚠️ Plan-stage 검증 항목 (필수)**:
  1. 일반적 consumer GPU(8GB VRAM)에서 기본 `GTX_DDR_DEFAULT_SIZE_BYTES = 4 GiB`가
     50% VRAM 점유 — `GTX_DDR_SIZE` env var override 권장 + README 명시.
  2. `ddr_dump_to_file` / `ddr_init_from_file` 경로는 `xp.asnumpy()` 사전 변환
     필수 (file I/O는 host 전용).
  3. `DDR_MEMORY.ensure()` doubling-grow path가 GPU에서도 동작 (`xp.zeros` +
     copy) — vendor `n1s16_<op>.elf` 실행 중 4 GiB까지 grow하지 않는지 baseline 확인.

- **D-11 RegisterFile (SPR int64) = scratchpads와 같은 device.**
  현재 npu.py:94-96 `device=DEVICE` 명시. xp 결정에 따라 자동 이동.
  **⚠️ Plan-stage 검증 항목 (필수)**: SPR access 빈도 측정. dispatch마다 N×다발성
  read/write 발생 — 작은 scalar GPU op이 5x ABS regression 원인이었던 패턴과 유사.
  Wave 1 끝에 ABS perf 측정으로 검증 (≤105s gate). 위반 시 RegisterFile만
  host-pinned 예외 처리 (plan-stage 결정).

- **D-12 Cross-device transfer API = 헬퍼 두 개.**
  `to_host(arr)` / `to_device(arr)` 헬퍼를 config_params.py에 정의.
  xp=numpy면 no-op (return arr), xp=cupy면 `cp.asnumpy(arr)` / `cp.asarray(arr)`.
  현재 torch `.cpu()` / `.to(device)` 사이트 (npu.py:354, dma_engine.py:682)를
  헬퍼로 1:1 치환. DMA 경계와 file I/O 경계에서만 사용 (산발적 호출 금지).

### Area 4: Numba × xp compatibility + tloop fusion fate (D-13 ~ D-17)

- **D-13 P7의 28개 @njit kernel = 28개 전부 dual-impl + numba CUDA backend.**
  scope 확장 선택. 사용자 명시: "`from numba import cuda; @cuda.jit` 사용".
  **scope 경고 (plan-stage가 명시적으로 다룸)**: 28개 dual-impl + 테스트 fixture가
  v1.1 milestone 범위 초과 가능. plan-stage에서 분할 옵션 검토:
  - 옵션 A: Phase 9 = numpy default + cupy raw mode (fallback path)만, P10 신설
    (cuda.jit dual-impl 28 kernel)
  - 옵션 B: Phase 9 단일로 모두 land, milestone 일정 + UAT 부담 흡수
  - 옵션 C: Hot-path 우선 (5-7 kernel만 cuda.jit, 나머지는 cupy native
    vectorized cp.* 호출로 충분)
  → plan-stage가 1주 추정 후 사용자에게 옵션 선택 요청.

- **D-14 Kernel 소스 = `numba.guvectorize` universal source + target switching.**
  사용자 명시. 단일 소스 + target='cuda' or 'cpu' (또는 'parallel') 키워드로
  스위치. **caveat**: P7의 njit 커널 28개 중 일부는 nested loop / state mutation
  pattern으로 guvectorize signature ('(n)->(n)', '(m,k),(k,n)->(m,n)' 등)에 안 맞을
  수 있음. plan-stage가 audit:
  - guvectorize-convertible 커널: 단일 소스로 통일
  - 비-convertible 커널: 별도 `_njit_kernels.py` + `_cuda_kernels.py` dual-source
    (이 경우 D-14를 일부 양보).
  - 결과를 plan SUMMARY에 명시.

- **D-15 tloop_buffer.py `_execute_fused` = 1:1 drop-in.**
  `torch.abs` → `xp.abs` / `torch.neg` → `xp.negative` / `torch.exp` → `xp.exp`.
  Bulk op은 ndarray에서도 동일 의미. fusion fast path 유지 (replay-only 회귀
  방지). Wave 3 끝에 ABS perf 재측정으로 fusion benefit 보존 확인.
  TRANSPARENT_MNEMONICS / BUFFERABLE_MNEMONICS / _VEC_UNARY_MNEMONICS 컨트랙트
  변경 없음.

- **D-16 Test 포팅 범위 = `tests/gtx/` 3 파일 전체 + conftest.**
  conftest.py + test_csr_registry_chain.py + test_mcast_copy_mem.py 총 54개 torch
  참조 모두 numpy로 치환. `examples/` (RoCC 계층 데모)는 영향 없음. tests에서
  torch fixture는 backend-agnostic xp fixture로 교체 (`xp = numpy if not
  GTX_USE_CUDA else cupy`).

- **D-17 pyproject.toml = torch 완전 제거 + `[cuda]` extras.**
  - `[project] dependencies`에서 torch 제거
  - `[project.optional-dependencies] cuda = ["cupy-cuda12x>=13.0"]` 추가
  - `[project.optional-dependencies] cuda-jit = ["numba>=0.58", "cupy-cuda12x>=13.0"]`
    분리 검토 (numba는 cpu-only도 의미 있어 base에 유지 가능 — plan-stage 결정)
  - `cibuildwheel.test-extras` 정리: base test는 numpy만, cuda test는 별도
    matrix entry (CI에 GPU runner 필요 — 일반 cibuildwheel cloud runner에서는 skip)
  - wheel size delta 측정 후 STATE.md 기록.

### Claude's Discretion

- **다음 사항은 plan-stage / executor 재량**:
  - `to_host()` / `to_device()` 헬퍼의 정확한 시그니처 (dtype 보존, view vs copy
    의미)
  - guvectorize-convertible audit의 정확한 형식 (markdown table vs RESEARCH 부록)
  - Wave 0 scaffold 안에 backend fixture를 conftest.py에 두냐 별도 helper 모듈로
    두냐
  - `[cuda-jit]` extras 분리 여부 — numba를 base dep로 유지할지 결정
  - cuda kernel 단위 테스트의 mock vs real GPU 정책 (GPU runner 없는 CI 환경 대응)
  - 28-kernel dual-impl scope 옵션 (A/B/C) 중 plan-stage가 1주 추정 후 사용자 컨펌

### Folded Todos

해당 없음 (todo cross-reference 매칭 없음 — `gsd-tools todo match-phase 9` 결과 0건).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 9 핵심 ROADMAP / REQ 자료

- `.planning/ROADMAP.md` §"### Phase 9: Backend migration — PyTorch → NumPy + CuPy
  opt-in" — Goal, Depends on (Phase 8), Requirements (BM-01..06), 6 Success Criteria.
  **참고**: REQUIREMENTS.md에는 BM-01..06이 아직 미정의 — plan-stage가 ROADMAP success
  criteria에서 transcribe해야 함 (또는 별도 plan task로 REQUIREMENTS.md 업데이트).
- `.planning/REQUIREMENTS.md` §"## Milestone v1.1 Post-Ship Polish" (lines 283+) —
  MTDMA/VTW가 있음, BM-* 미정의. plan-stage에서 BM-01..06 transcription 필요.
- `.planning/STATE.md` lines 1-44 — milestone v1.1 frontmatter (Phase 8 complete 후
  Phase 9 enter 상태).

### Phase 8 invariant (회귀 방지 게이트)

- `.planning/phases/08-multi-tile-dma-parity/08-CONTEXT.md` §D-01 ~ D-13 — strict
  byte-exact gate 설계 (D-09 tile-2 unit test + D-11 12-op smoke set + D-12
  HAS_NUMBA=False perf baseline).
- `tests/gtx/test_regression_fw_full_sweep.py` — vendor 84-op sweep harness +
  `_find_elf` multi-path search + GTX_DDR_REVERSED 자동 적용. Phase 9 마이그레이션이
  보존해야 할 핵심 회귀.
- `tests/gtx/test_multi_tile_dma.py` — P8 MTDMA-03 tile-boundary unit test (vendor
  `.elf` 의존 없음). 매 wave 끝 gate 중 하나.
- `tests/gtx/data/baseline_walltime.txt` — HAS_NUMBA=False ABS 5x walltime baseline
  (P8 VTW-03 결과). Phase 9 perf gate(±10%) 비교 기준.

### Phase 7 numba 자산

- `.planning/phases/07-numba/07-CONTEXT.md` — 28개 njit kernel 등록 결정, P7
  HUMAN-UAT closure 조건.
- `.planning/phases/07-numba/07-HUMAN-UAT.md` Findings (lines 27-34) — vendor
  asset 위치 + multi-tile DMA bug 발견 경위 + endianness root cause (LE 강제 필요).

### 코드 사이트 (torch 사용 13 파일 — D-05 Wave 매핑)

- `src/main/python/riscv/gtx/config_params.py` (lines 1-25, 특히 8-23 cuda
  regression history 주석) — **Wave 0 핵심**. D-04 DEVICE 제거 + D-01 xp 정의.
- `src/main/python/riscv/gtx/__init__.py` (lines 80, 87-88) — **Wave 3**. torch
  import + DEVICE re-export 제거.
- `src/main/python/riscv/gtx/npu.py` (lines 12, 19, 94-106, 354) — **Wave 1**.
  `_mxe_accum`, `_credit_ld`, `_credit_st`, SPR RegisterFile 인스턴스화. line 354의
  `.cpu()` 호출 → `to_host()` 치환.
- `src/main/python/riscv/gtx/unit/memory.py` (lines 6, 16, 22, 48-56, 79-145) —
  **Wave 1**. module-level `_L2/L1/L0_GLOBAL` + `DDR_MEMORY` (line 79 _DDR_DEVICE,
  line 145 doubling-grow). D-10 GPU placement 검증의 중심지.
- `src/main/python/riscv/gtx/unit/register_file.py` (lines 19, ~80) — **Wave 1**.
  `torch.zeros(shape, dtype=torch.int64, device=device)` → xp.zeros. D-11 SPR
  device placement.
- `src/main/python/riscv/gtx/unit/context/dma_engine.py` (lines 21, 682) — **Wave 2**.
  `.cpu()` 호출 → `to_host()` 치환.
- `src/main/python/riscv/gtx/unit/ins/ops/act.py` (lines 24-25, 45-181) — **Wave 2**.
  FP8 LUT 빌드 (`_build_fp8_to_fp16_lut` 256-byte + `_build_fp16_to_fp8_lut` 64KB
  precomputed). cvt_qh/hq/ih/hi/hn/sh/hs/dh 변환 함수 9개.
- `src/main/python/riscv/gtx/unit/ins/ops/mm.py` (lines 28, 79) — **Wave 2**. gemm
  유사 FP32-accumulate + FP16 cast pattern.
- `src/main/python/riscv/gtx/unit/ins/ops/spr.py` (line 18) — **Wave 2**.
- `src/main/python/riscv/gtx/unit/ins/ops/vec.py` (lines 20, 67-102) — **Wave 2**.
  `torch.arange` (line 102) + DOT/VSUM FP32-internal-accumulate.
- `src/main/python/riscv/gtx/tloop_buffer.py` (lines 415-440 `_execute_fused`) —
  **Wave 3**. D-15 1:1 drop-in 대상. `_VEC_UNARY_MNEMONICS` (ABS/NEG/EXP)
  bulk-fusion fast path.
- `src/main/python/riscv/gtx/_verify.py` (line 9) — **Wave 3**.
- `src/main/python/riscv/gtx/unit/csr/register.py` (line 95 주석) — **Wave 2 보조**.
  RegisterFile bit-field 의 torch.Tensor 도큐먼테이션.

### Test 코드 사이트 (D-16)

- `tests/gtx/conftest.py` — backend fixture 신설 가능 위치 (D-04 DEVICE 제거 후).
- `tests/gtx/test_csr_registry_chain.py` — RegisterFile 단위 테스트.
- `tests/gtx/test_mcast_copy_mem.py` — Wave 2 의존성 (260518-ibf에서 land한 5 unit
  test 포함).

### Vendor C++ reference (선택적 — 변환 의미 검증용)

- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` (lines 89-151) — FP16↔FP32 IEEE 754
  binary16 RNE 시맨틱 ground truth. D-15 fusion path가 보존해야 할 정밀도 규약.
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — gtx/ 디렉토리 구조 + FP16 byte order
  regs (LE for L1/L0, DDR is BE under GTX_DDR_REVERSED).
- `/mnt/e/14_NIGHTLY/RISCV-GTX_pk/src+inc/` — SystemC TLM 풀시뮬 (SPU.cpp 178K
  본체) — 회귀 검증 시 first stop (memory `reference_vendor_cpp.md` 참조).

### 배경 자료

- `CLAUDE.md` (project root) §"Constraints" — "NumPy 백엔드 가정" 명시 (Phase 9의
  설계 원천).
- `CLAUDE.md` (project root) §"Technology Stack > Languages" — Python 3.8+ runtime
  baseline (단 Phase 1 D-08으로 cp310-cp312로 축소).
- Memory `reference_test_runner.md` — `uv run pytest` 강제 사용 (시스템 torch
  libcusparseLt 누락 회피). Phase 9 작업 중에도 `uv run` 경로로 검증.
- Memory `project_gtx_extension_silent_import_failure.md` — `riscv/gtx/__init__.py:62-68`의
  ImportError silent swallow 패턴 주의 (Wave 3에서 torch 제거하면서 이 try/except도
  audit 필요).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`config_params.py:25 DEVICE SSOT 패턴**: 이미 backend 결정의 단일 위치 —
  xp alias 확장에 직접 활용 (D-01).
- **`memory.py` module-level allocation 패턴**: `_L2_GLOBAL/L1/L0` import-time alloc
  + per-(NEST, SPU) view 디스패치. xp.zeros로 1:1 치환 (Wave 1).
- **`act.py` FP8 LUT precompute (256 + 64KB at import-time)**: xp 추상화 후에도
  동일 효율 (numpy/cupy 둘 다 module-level 캐시 지원). gpu에서는 GPU memory에 64KB
  추가 점유.
- **`tloop_buffer.py` BUFFERABLE/TRANSPARENT/VEC_UNARY mnemonic frozenset**:
  backend 무관 dispatch — backend 변경에 영향 없음 (frozen string set).
- **P7의 28개 njit 커널 (location TBD by plan-stage)**: D-14 guvectorize universal
  source 변환 대상.

### Established Patterns

- **DEVICE 경계 함수 (npu.py:354, dma_engine.py:682의 `.cpu()`)**: cross-device
  transfer는 DMA 경계에서만. D-12 `to_host()` 헬퍼가 동일 위치를 점유.
- **module-level allocation + import-time eager**: 모든 scratchpad/LUT가 import-time.
  D-02 eager resolve가 자연스럽게 일치.
- **FP32-internal-accumulate + 단일 FP16 cast (mm.py, vec.py VSUM/DOT)**:
  `np.matmul(..., dtype=np.float32).astype(np.float16)` 동등 패턴으로 1:1 치환 가능.
- **xs1=0 우회 (CORE-04, ROADMAP)**: dispatch level이라 backend 무관.

### Integration Points

- **`pyproject.toml` `[project.dependencies]`**: torch entry 제거 + numpy 유지
  (이미 P1 D-08에서 `numpy>=2.0,<3` 명시).
- **`pyproject.toml` `[project.optional-dependencies]`**: 신설 `cuda = [...]`.
- **`tests/gtx/conftest.py`**: backend fixture 신설로 모든 테스트 자동 xp-aware
  (D-16).
- **`cibuildwheel` test-extras**: GPU test가 cloud runner에서 SKIP 처리되도록 marker
  관리.
- **`__init__.py:62-68` ImportError silent swallow**: torch import 실패 처리도 함께
  audit. xp 결정 후 ImportError 의미 변경 (numpy도 못 import이면 catastrophic).

</code_context>

<specifics>
## Specific Ideas

- **사용자 명시 — `numba.cuda` backend**: "from numba import cuda; @cuda.jit 사용"
  (Q1 자유 입력) + "Universal source + jit fallback (numba.guvectorize)" (Q2 선택).
  → D-13 + D-14 결합. 28 kernel 전부 dual-impl. plan-stage가 scope 분할 옵션 제시.

- **DDR full-GPU 선호**: D-10에서 사용자가 "전체 GPU 통일" 선택 (현재 explicit
  CPU 계약을 되돌림). 4 GiB DDR alloc + VRAM budget 위험을 plan-stage가 명시적으로
  검증.

- **RegisterFile follow-device 선호**: D-11에서 사용자가 "scratchpads와 같은 device"
  선택. 5x ABS regression과 비슷한 패턴 위험 — plan-stage Wave 1 끝 perf 검증
  필수.

- **fail-loud 절대 선호**: D-03 silent fallback 절대 금지. 사용자의 일관된 design
  intent (config_params.py:9-23 주석 + 본 결정).

- **Wheel 사이즈 추적**: ROADMAP success #5 "Wheel size delta vs pre-migration
  ≤ 0 MB". PyTorch 제거가 wheel 50+ MB 감소 효과 — 정확한 측정 필요.

</specifics>

<deferred>
## Deferred Ideas

### V1.2 (Phase 9 종결 후 검토)

- **CUDA kernel 성능 최적화**: shared memory, warp shuffle, cupy ElementwiseKernel
  (RawKernel) — Phase 9는 correctness-first. perf phase는 별도.
- **pybind11 트램폴린 (C++) torch::Tensor 사용 제거**: Phase 9는 Python-side만.
  C++ 측은 별도 phase.
- **Wheel multi-arch (cp310 + 추가 Python 버전)**: P1 D-08의 cp310-cp312 결정
  유지. cp313+ 확장은 v1.2.
- **Numba dispatch overhead 최적화**: P7에서 inline 가능했지만 도려뽑힌 패턴 →
  v1.2 perf phase.
- **CuPy memory pool tuning**: 4 GiB DDR alloc과 충돌 시 cupy.cuda.MemoryPool
  설정. D-10 검증 결과에 따라 plan-stage 또는 v1.2.

### Plan-stage 분할 시 (D-13 scope warning):

- **P10 신설 가능성**: 28-kernel dual-impl + cuda smoke test infrastructure가 v1.1
  milestone 범위 초과 시. P9 = numpy default + cupy raw, P10 = cuda kernel 작업.

### Reviewed Todos (not folded)

해당 없음 (cross_reference 0건).

</deferred>

---

*Phase: 09-backend-migration-numpy-cupy*
*Context gathered: 2026-05-18*
