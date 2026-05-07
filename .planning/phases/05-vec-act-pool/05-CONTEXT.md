# Phase 5: VEC/ACT/Pool - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5는 GTX NPU의 **두 번째 "compute" 레이어**를 구축한다. P4가 `gemm_core` + MM/MMC dispatch로 plumbing 정합성을 strict mode로 증명한 위에, **vector ops + activations + pooling + format_cvt + 32-op host-side oracle suite**를 올려서 NPU 연산 표면을 완성한다. 구체적으로:

1. **VEC 5종 op 구현** — SASMD (add/sub/mul/div × IS/VS = 8 variants, funct7=0x10) + DOT/VSUM (funct7=0x1A) + CLAMP min/max/arange/accum (funct7=0x18..0x1F, L0/L1 분기) + scalar/immediate variants + `firmware_vec_op` packed-rs1 디코드 (VEC-01..05)
2. **활성화 8종 + 방향성 비대칭** — RELU/SOFTMAX/ESUM forward (ADDRA→ADDRR), PRELU/GELU/TANH/SIGM reversed (ADDRR→ADDRA). C++ `gtx_npu_act.cc:18-20`의 ISS pattern 직역 (ACT-01, ACT-02)
3. **`exec_pooling`** — max + avg, kernel_size 기반 length/kernel_size 출력, avg-pool signed-zero canonicalization (-0.0 → +0.0). 항상 forward 방향 (ADDRA→ADDRR per `gtx_npu_act.cc:199`) (ACT-03)
4. **`exec_format_cvt`** — FP16↔FP32, FP16↔FP8 (custom E4M3-layout, non-NVIDIA semantics), FP16↔INT8, FP16↔INT32. scale+offset packed in `GSPR_GTX_OPERAND2 = [offset_fp16:16 | scale_fp16:16]`. 항상 forward 방향 (ACT-04)
5. **`_imm` L0 path 변형 활성화** — funct7=0x28/0x2A/0x2C/0x2D `& 4` bit selects L0(immediate) path. 같은 forward/reverse 비대칭 유지 (ACT-05)
6. **VRF-02 oracle suite** — `vendor/gtx_cpp_reference/gtx/verify_ref.py` (32-op host-side scalar 검증)을 pytest 파라미터화 단위 테스트로 흡수. 각 op이 oracle과 ULP 1 내 일치
7. **첫 ACT `.elf` 회귀 strict mode 통과** — `activation_relu_gelu.elf` 실행 → DDR dump → `_verify_minimal.compare_hex(strict=True)` PASS. forward(RELU) + reversed(GELU) 양 방향 동시 검증

다음 모두는 **Phase 5 비범위(out-of-scope)** — 다른 페이즈가 다룬다:

- **Production `riscv.gtx._verify` (CLI 포함)** → Phase 6 VRF-01 (P5는 P4 `_verify_minimal` 재사용)
- **`tests/gtx/data/{golden,elf}/` 패키지 데이터 등록** → Phase 6 PKG-01
- **`pyspike-verify` console script** → Phase 6 PKG-03
- **자동 DDR dump (`GTX_DDR_DUMP` atexit hook)** → Phase 6 또는 별도 follow-up (P3 D-09 / P4 D-12 상점 유지; P5 회귀도 테스트 내부 명시 dump)
- **Full .elf regression matrix (gem5+ISS)** → Phase 6 VRF-04 (P5는 단일 `activation_relu_gelu.elf` 1개)
- **CUDA / OMP / BLAS 가속 경로** — PROJECT.md Out of Scope (CUDA), `vendor/gtx_cpp_reference/gtx/CLAUDE.md`의 `GTX_USE_OMP`/`GTX_USE_CUBLAS` 코드 분기는 무시 (Python 직역만)
- **Numba @njit on vec_core/act_core hot kernels** → Phase 7 (P5는 numba-friendly 구조만 마련 — P4 D-01 lineage)
- **DMA-3D / IM2COL / MCAST** → v2 (Phase 3 deferred)
- **PCIe EP / VFIO** → PROJECT.md Out of Scope (v2 reconsider)

</domain>

<decisions>
## Implementation Decisions

### 모듈 구성 (D-01 ~ D-04)

- **D-01:** **VEC 3-way module split** — `riscv/gtx/ops/vec.py` (@handler 진입점, thin forwarders) + `riscv/gtx/vec_engine.py` (firmware_vec_op decode + variant dispatcher + L0/L1 path branch + 메모리 view 생성) + `riscv/gtx/vec_core.py` (순수 stateless NumPy 커널: `sasmd_kernel`, `dot_kernel`, `vsum_kernel`, `clamp_kernel`).
  - **이유:** P4 D-01 패턴 직접 mirror. VSUM/DOT는 P7 numba `@njit` 핫 후보 — JIT boundary 명확화. spike-bound 디코드/state는 `vec_engine`에 격리, array-in/array-out 순수성은 `vec_core`에 보장.
  - **위험:** 파일 3개. P4에서 검증된 패턴이라 위험 낮음.

- **D-02:** **ACT bundled module** — `riscv/gtx/ops/act.py` (모든 activation+pool+format_cvt @handler 진입점) + `riscv/gtx/act_engine.py` (single engine: 활성화 direction dispatch + pool kernel/stride decode + format_cvt 타입쌍 decode + scale/offset unpack from GSPR_GTX_OPERAND2) + `riscv/gtx/act_core.py` (순수 NumPy: 활성화 + pool + format_cvt + FP8 LUT 8종 함수).
  - **이유:** P4 D-01 single-engine 패턴 일관 + ACT 도메인이 (activation, pool, format_cvt) 모두 같은 ADDRA/ADDRR 인터페이스를 공유한다는 정합성. 파일 3개로 ACT 전체 커버 → 인지 부하 적음. `act_core.py`는 ~400 LOC 예상 — P4 `mm_engine.py` (342 LOC)와 비슷.
  - **trade-off:** FP8 LUT가 `act_core.py` 내부에 들어가면서 import-time 64KB+256B 메모리 사용. 무시 가능 (mm_basic.elf 1.3KB보다 작은 이슈).
  - **재고 시점:** P5 plan-phase에서 `act_core.py`가 600 LOC 넘어가면 `format_core.py` / `pool_core.py` 분리 검토.

- **D-03:** **VRF-02 oracle 위치 = `tests/gtx/_oracles.py`** (test-only tier, `_verify_minimal.py` / `_mocks.py`와 동급).
  - **이유:** wheel 미포함 (PROJECT.md "wheel size ≤50MB" 정합). 단위 테스트가 `from ._oracles import sasmd_ref, dot_ref, ...` import. P6 VRF-01 production `riscv.gtx._verify` 승격 라인과 별도 — oracles는 dev-only.
  - **단일 파일 vs 분할:** P5 plan-phase에서 32 oracle이 한 파일 200 LOC 넘어가면 `tests/gtx/_oracles/{vec,act,format,pool}.py` 분할 검토. 첫 시도는 단일 파일.

- **D-04:** **Wave structure mirror P4** —
  - **Wave 1a (scaffold, 1 plan, sequential alone):** RED test stubs for all ~30 P5 tests (`test_op_vec.py` SASMD/DOT/VSUM/CLAMP, `test_op_act.py` 8 활성화 + _imm, `test_op_format.py` FP16/FP32/FP8/INT8/INT32, `test_pooling.py` max/avg, `test_regression_fw_act.py` strict mode) + `_oracles.py` 32-op stub skeletons + `activation_relu_gelu.elf` fixture (vendor or hand-written .S — D-09 lineage).
  - **Wave 1b (impl, 4 plans, sequential within wave due to test_op_*.py shared edit surface):** plan 02 = `vec_core` + GREEN-fill VEC unit tests, plan 03 = `vec_engine` + `ops/vec.py` + GREEN-fill VEC dispatch tests, plan 04 = `act_core` activations + GREEN-fill activation unit tests, plan 05 = `act_engine` + `ops/act.py` (활성화/pool/format_cvt 모두) + GREEN-fill remaining tests.
  - **Wave 2 (integration, 1 plan):** `_oracles.py` 32-op full GREEN + activation_relu_gelu.elf strict-mode regression + cross-op verify_ref parity sweep.
  - **총 6 plans.**
  - **이유:** P4 패턴이 검증됨. test_op_*.py 공유 편집 면적 (Wave 1b 모든 plan이 같은 테스트 파일 수정) 때문에 sequential within Wave 1b 필수 — P4 06번 lesson.

### 활성화 방향 비대칭 (D-05 ~ D-08)

- **D-05:** **`is_reversed` 파라미터를 `ops/act.py` 진입점에서 명시 전달** (P4 D-05 `is_accumulate` 패턴 직접 mirror).
  - 8 활성화 @handler가 각자 `is_reversed=False/True` 명시: `_exec_relu(...)` → `act_engine.firmware_act(npu, proc, insn, is_reversed=False, op='relu')`. `_exec_gelu(...)` → `act_engine.firmware_act(npu, proc, insn, is_reversed=True, op='gelu')`.
  - **이유:** 진입점에서 분기 → routing 사고 0. spike trace에서 funct7 → @handler → is_reversed 모두 한 줄로 추적. C++ `gtx_npu_act.cc:18-20` ISS pattern 직역.

- **D-06:** **방향 진실의 위치 = @handler 본문 inline.**
  - 각 reversed @handler는 docstring/comment로 `# Reversed direction: ADDRR→ADDRA, see gtx_npu_act.cc:18-20` 자기 문서화.
  - 모듈 레벨 `REVERSED_OPS` set 등 indirection 금지 — direction claim이 dispatch 사이트 옆에 살아 있어야 함.
  - **이유:** 4-byte indirection cost > 8-line @handler explicit cost. 활성화 8개라는 작은 set에 set lookup overhead는 의미 없음.

- **D-07:** **ACT-05 _imm 변형도 같은 비대칭 유지, 별도 @handler.**
  - L0 path (immediate): RELU_imm / SOFTMAX_imm / ESUM_imm forward, PRELU_imm / GELU_imm / TANH_imm / SIGM_imm reversed.
  - **총 16 활성화 @handler:** 8 ISS path + 8 L0 immediate path.
  - **funct7 정확값은 research가 잠금** — `gtx_npu_disasm.inc`에서 `& 4` bit semantics + 0x28/0x2A/0x2C/0x2D 정확 매핑.
  - **위험:** 16 @handler 보일러플레이트. P4 10 MM/MMC variants 패턴 일관 — 검증된 cost.

- **D-08:** **`format_cvt` 와 `exec_pooling`은 direction state 안 가짐.**
  - 항상 forward (ADDRA→ADDRR). C++ `gtx_npu_act.cc:225-227` (format_cvt) + `:199` (pooling) 직역.
  - 해당 @handler들은 `is_reversed` 파라미터 자체를 안 받음 — `act_engine.firmware_format(...)` / `act_engine.firmware_pool(...)`은 별도 함수 (D-02 single-engine 안에서 3개 entry point).
  - **이유:** clean engine surface — `is_reversed` 파라미터가 의미 없는 dispatch path에 잡음으로 들어가지 않음.

### VSUM/DOT 정밀도 (D-09 ~ D-12)

- **D-09:** **VSUM 듀얼 모드 — 커널은 mode A만, mode B는 firmware composition.**
  - **Mode A (kernel):** `vec_core.vsum(view: ndarray) -> np.float16` — 항상 FP32 internal accumulate + single FP16 cast. `gtx_npu_vec.cc:103-108` 직역 (`float sum = 0.0f; for (...) sum += rd16(...); wr16(addr_r, 0, sum); spu.l0[0..1] = gtx_fp32_to_16(sum)`).
  - **Mode B (firmware):** firmware가 row-by-row VSUM을 N번 호출하고 N개 FP16 결과를 별도로 합산하는 패턴. **커널 자체는 mode-agnostic** — 커널 호출자(firmware ELF)가 mode B를 만든다.
  - **이유:** 벤더 `CLAUDE.md` "VSUM 정밀도" 문구 ("FP32 내부 누적 후 1회 FP16 변환. 레퍼런스 매칭 필요 시 행별 분할 후 FP16 부분합 재합산.") 직접 매핑. 두 모드는 contradiction이 아니라 **단일-호출 vs N-호출 composition** 차이. P4 D-03 "stateless 커널 = composition primitive" 정합.

- **D-10:** **VSUM 두 테스트 패밀리.**
  - `test_vsum_fp32_internal_anti_pattern` (ROADMAP success #1 직역): `np.float16([1.0, 1e-4]*1000).sum()` via `vec_core.vsum(...)` ≈ 0.1, **inf 아님** (FP16 truncated would saturate). 커널의 mode A 검증.
  - `test_vsum_row_split_matches_cpp` (parametrized over row counts ∈ {2, 4, 8, 16}): firmware 모드 시뮬레이션 — `vec_core.vsum`을 row 별로 N번 호출 + 각 결과를 `np.float16`로 캐스트 + N개 FP16 부분합을 다시 FP32 누적 + final FP16 cast. C++ golden hex (P5 plan-phase에서 source 잠금 — research가 row-split exercise하는 mini .elf 식별 또는 scaffold)와 bit-exact 일치.
  - **이유:** 두 테스트 = 두 distinct claim. mode A 안티패턴 + mode B firmware 정합성 분리 보장.

- **D-11:** **DOT 정밀도 = VSUM single-call 동등.**
  - `vec_core.dot(view_a, view_b) -> np.float16` = FP32 internal accumulate + single FP16 cast (`gtx_npu_vec.cc:251-262` 직역).
  - 안티패턴 테스트: `vec_core.dot([1.0]*1000, [1e-4]*1000)` ≈ 0.1, inf 아님.

- **D-12:** **FP32→FP16 cast on overflow = IEEE round-to-nearest, inf on overflow.**
  - `np.float16(fp32_value)` 직접 사용 (NumPy 2.x IEEE conversion). C++ `gtx_fp32_to_16` 정확한 의미와 일치 — research가 plan-stage에서 `vendor/gtx_cpp_reference/gtx/gtx_npu.h:89-151` 코드 직접 비교로 확정.
  - **명시 검증 테스트:** `vec_core.vsum(np.full(70000, 1.0, dtype=np.float16))` = `np.float16(inf)` (correct overflow behavior). 문서화 가치 높은 anti-pattern.

### format_cvt + FP8 codec (D-13 ~ D-16)

- **D-13:** **format_cvt = 6 @handler (1 per direction).**
  - `_exec_scvt_qh` (FP16→FP8), `_exec_scvt_hq` (FP8→FP16) — funct7=0x20, sub_op LSB로 분기되던 C++을 진입점에서 분리.
  - `_exec_scvt_ih` (FP16→INT8), `_exec_scvt_hi` (INT8→FP16) — funct7=0x21.
  - `_exec_scvt_hn` (FP16→INT32 normalize), `_exec_scvt_nh` (INT32→FP16) — funct7=0x?? (research lock from `gtx_npu_disasm.inc`).
  - **이유:** P4 D-04 lesson — 분리된 @handler가 disasm clarity (`scvt_hq` vs `scvt_qh` 명확) + P7 numba boundary + 디버깅 정확성을 모두 잡음. C++ `act.cc`의 sub_op LSB 분기는 NPU HW의 인코딩 절약 트릭일 뿐 — Python 측에서는 진입점에서 분리하는 것이 깨끗.

- **D-14:** **FP8→FP16 codec = 256-byte LUT precomputed at module load.**
  - `act_core.FP8_TO_FP16_LUT: np.ndarray[256, dtype=np.float16]` — 모듈 import 시 `gtx_fp8_to_32` (vendor `gtx_npu.h:154`) 직역 함수를 256개 모든 입력에 대해 호출하여 LUT 빌드.
  - 디코드 hot path: `LUT[fp8_byte_array]` (NumPy fancy indexing, 벡터화).
  - **bit-twiddle source-of-truth는 LUT-builder 함수 docstring에 위치** — LUT은 cache, builder는 spec.

- **D-15:** **FP16→FP8 codec = 64KB LUT precomputed at module load.**
  - `act_core.FP16_TO_FP8_LUT: np.ndarray[65536, dtype=np.uint8]` — 모듈 import 시 `gtx_fp16_to_8` (vendor `gtx_npu.h:182`) 직역 함수를 65536개 모든 FP16 입력에 대해 호출하여 LUT 빌드.
  - 인코드 hot path: `LUT[fp16_array.view(np.uint16)]` (벡터화).
  - **이유:** 64KB는 mm_basic.elf (1.3KB)보다 작음. 한 번 빌드 + 영구 cache. RNE rounding logic은 builder에 한 번만 산다 — per-call 호출에서 다시 풀 필요 없음.

- **D-16:** **FP8 codec 명시 테스트 커버리지.**
  - **Subnormal 테스트:** `test_fp8_subnormal_decode` — parametrize over `(h_sign ∈ {0,1}, h_exp=0, h_frac ∈ {1,2,3,4,5,6,7})`. `gtx_fp8_to_32` 직역에 따라 `(h_frac/8) * 2^(-6) * (-1)^sign` 와 일치.
  - **exp=0xF 테스트:** `test_fp8_exp_max_is_inf` — `h_exp=0xF, h_frac=0` 입력 → FP32 inf. `h_exp=0xF, h_frac=4` → FP32 inf (벤더의 `f_exp = 0xFF << 23` semantics는 inf 매핑이지 NaN 아님). **NVIDIA standard E4M3와 의도적 divergence — 문서화 가치 높음.**
  - **256-input round-trip identity:** `test_fp8_roundtrip_identity` — 모든 256 FP8 입력에 대해 `LUT_FP8_TO_FP16[in]`로 디코드 후 `LUT_FP16_TO_FP8[decoded.view(uint16)]` 인코드 → 원본 256 값 복원 (단, decoded가 FP16에서 표현 가능한 한). 양방향 LUT 일관성 invariant.

### Claude's Discretion

다음은 implementation detail로 Claude 판단 (research/plan 단계에서 정확화):

- 10개 SASMD variant funct7 sub-encoding: IS vs VS bit position, exact sub_op encoding (research lock from `gtx_npu_disasm.inc`)
- CLAMP funct7=0x18..0x1F → (op × L0/L1 path) 정확 매핑 (research lock)
- `firmware_vec_op` packed-rs1 인코딩 (P4 `firmware_mm_op` 패턴 analog — research가 `gtx_npu_dispatch.cc` line search로 잠금)
- `_imm` 활성화 immediate operand 디코딩 위치 (instruction의 어느 필드, research lock)
- VSUM row-split 테스트 golden hex source: 기존 vendor .elf에서 capture 가능한지, 아니면 P5에서 mini .elf scaffold 필요한지 (research가 vendor `test/run_tests_n1s16.sh` 매핑 + `tests/gtx/data/elf/`에 적합한 fixture 식별)
- `activation_relu_gelu.elf` fixture sourcing: vendor에서 차용 가능한지 (P4 D-09 패턴) 또는 P5 Wave 1a에서 hand-written `.S` + Makefile (P4 mm_basic.S lineage). research가 vendor 자산 매핑 후 결정.
- `gtx_fp32_to_16` 정확 IEEE 시맨틱: `np.float16(fp32)` cast가 bit-exact 일치하는지 verify (research가 256개 boundary 값 `±0`, `±inf`, NaN, subnormals, fp16-overflow 비교)
- pool kernel/stride 패킹 위치 (firmware → instruction 어느 필드, research lock)
- `vec_engine` / `act_engine` 내부 함수 분리 정도 (`firmware_vec_op` 단일 함수 vs `_decode_args` + `_dispatch_variant` + `_writeback` 분리 — P4 D-01의 plan-stage 결정 lineage)
- CLAMP `accum` variant의 정확한 시맨틱 (per-element accumulate vs reset — research가 C++ source 잠금)
- `exec_format_cvt` SCVT_HN (INT32 normalize) 의 정확 normalize 공식 (research lock from `gtx_npu_act.cc:301+`)
- VRF-02 oracle 단일 파일 vs 분할 임계: `_oracles.py`가 ~200 LOC 넘으면 plan-stage에서 분할 결정

### Folded Todos

None — `gsd-tools todo match-phase 5`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Vendor C++ source (operational truth — bit-exact 직역 대상)

- `vendor/gtx_cpp_reference/gtx/gtx_npu_vec.cc` — VEC ops (SASMD/DOT/VSUM/CLAMP). 30KB, 632+ LOC.
  - **VSUM impl:** lines 102-113 — FP32 accumulate confirmed (`float sum = 0.0f; for (...) sum += rd16(...); wr16; gtx_fp32_to_16`).
  - **DOT impl:** lines 251-262 — same pattern as VSUM.
  - **SASMD/CLAMP:** main switch around line 50+; `firmware_vec_op` decode around line 600+.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_act.cc` — Activations + pooling + format_cvt. 19KB.
  - **Direction asymmetry:** lines 18-20 + 35 (ISS pattern comments).
  - **exec_pooling:** lines 167-220 — reads ADDRA, writes ADDRR (forward only).
  - **exec_format_cvt:** lines 223-272+ — scale/offset unpack from `GSPR_GTX_OPERAND2`.
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — FP/Format codec helpers + GSPR layout.
  - **`gtx_fp8_to_32`:** line 154 — FP8 (E4M3-layout) → FP32 with custom subnormal `(h_frac/8) * 2^(-6)` and inf-on-exp=0xF semantics.
  - **`gtx_fp16_to_8`:** line 182 — FP16 → FP8 with RNE rounding.
  - **`gtx_fp32_to_16` / `gtx_fp16_to_32`:** lines ~89-151 — FP16↔FP32 helpers (P5 D-12 verify against `np.float16` cast).
  - **`GSPR_GTX_OPERAND2`:** packed `[offset_fp16:16 | scale_fp16:16]` for format_cvt.
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` — VEC/ACT mnemonic ↔ funct3/funct7 LUT (research source for D-01/D-02/D-13).
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc` — `dispatch_iss_opcode` VEC/ACT funct7 fillers (P5 plumbs into existing `dispatch_4mode` per P4 D-04 lineage).
- `vendor/gtx_cpp_reference/gtx/verify_ref.py` — 32-op host-side oracle (VRF-02 source). 12KB.
- `vendor/gtx_cpp_reference/gtx/verify.py` — DDR FP16 ULP/atol 비교 (P5는 P4 `_verify_minimal` mini-port 재사용; P6 VRF-01에서 production 승격).
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — vendor-side guidelines.
  - **★ VSUM 정밀도** 섹션: D-09 dual-mode lock의 source-of-truth.
  - **★ Activation 방향성** 섹션: D-05/D-06 source-of-truth.
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` (symlink) — GTX_NUM_NESTS, GTX_SPUS_PER_NEST, GTX_L1_SIZE 등 토폴로지 상수.

### 레퍼런스 폄웨어 / 골든 자산 (research 잠금 대상)

- `vendor/gtx_cpp_reference/test/` — 103-op GGML 커널 테스트 스위트 (vendor `test/CLAUDE.md` 참조). research가 활성화 / format_cvt 관련 .elf + golden hex 식별.
- `vendor/gtx_cpp_reference/test/run_tests_n1s16.sh` — gem5-simplified 인코딩 펌웨어 스위트 source.

### Project documents (locked context)

- `.planning/PROJECT.md` — Core Value (bit-exact w/ C++ libgtx_npu.so), Constraints (Python+NumPy only, no C++ additions), Out of Scope (CUDA / OMP 가속, PCIe EP), Validated requirements (P4 GTX-MM-01 closed).
- `.planning/REQUIREMENTS.md` — VEC-01..05, ACT-01..05, VRF-02 v1 acceptance criteria.
- `.planning/ROADMAP.md` Phase 5 섹션 — 5 success criteria + research-flag 노트 (asymmetry table, scale+offset packing, FP8 codec).
- `.planning/STATE.md` — 현재 진행 (P1, P3, P4 완료; P5 ready to plan).

### Prior phase contexts (decision precedent)

- `.planning/phases/04-mm-subsystem/04-CONTEXT.md` — D-01 (3-way split), D-02 (FP32 accumulate), D-04 (variant @handlers), D-05 (param at @handler entry), D-09 (vendor .elf 차용 + .S 폴백), D-13 (_verify_minimal mini-port), D-14 (strict mode), D-15 (op-level np.array_equal 직접). **P5 D-01/D-02/D-04~08/D-13~16의 lineage source.**
- `.planning/phases/04-mm-subsystem/04-VERIFICATION.md` — `proc.state` is property (not method) per pybind11 binding `src/main/cpp/py_module.cc:711`. **P5의 모든 spike-bound code (vec_engine, act_engine, ops/{vec,act}.py)는 `proc.state` 사용 필수.**
- `.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md` — DMA dispatch + DDR mode lessons (relevant to format_cvt INT32 normalization). LE byte order invariant carry-forward.
- `.planning/phases/02-skeleton-disasm/02-CONTEXT.md` — disasm registration patterns + per-op registry (P5 `ops/vec.py` + `ops/act.py` @handler 등록은 같은 메커니즘).

### Code context (existing pyspike assets to mirror)

- `src/main/python/riscv/gtx/mm_engine.py` — **pattern source for `vec_engine.py` and `act_engine.py`.** 342 LOC, 5 variant helpers + decode + dispatch.
- `src/main/python/riscv/gtx/gemm_core.py` — **pattern source for `vec_core.py` and `act_core.py`.** 150 LOC, pure stateless NumPy, P7 numba candidate.
- `src/main/python/riscv/gtx/ops/mm.py` — **pattern source for `ops/vec.py` and `ops/act.py`.** 148 LOC, 10 thin @handler forwarders.
- `src/main/python/riscv/gtx/ops/spr.py` — WRSPR/RDSPR re-dispatch lesson (P4 04-04 deviation #3). P5 may need analogous re-dispatch if any VEC/ACT funct7 collides with existing P2 None-key handlers.
- `src/main/python/riscv/gtx/encoding.py` — funct3/funct7 constants. P5 will add `GTX_F3_VEC_*`, `GTX_F3_ACT_*`, `GTX_F7_SCVT_*` constants here.
- `tests/gtx/_verify_minimal.py` — **reuse for `activation_relu_gelu.elf` regression in Wave 2** (D-04). BE FP16 bit-pair compare per Pitfall 1.
- `tests/gtx/_mocks.py` — `MockProcessor` exposes both `state` property AND `get_state()` method (P4 04-05 back-compat). **Use `state` property in P5 production code.**
- `tests/gtx/data/elf/Makefile` — pattern source if `activation_relu_gelu.S` hand-written fallback needed (P4 mm_basic.S lineage).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`riscv.gtx.mm_engine.firmware_mm` and `gemm_core.gemm_core`** (P4) — direct architectural template for P5 VEC/ACT engines + cores. Same shape (decode → variant dispatch → result writeback), same boundary discipline (pure kernel + spike-bound engine).
- **`tests/gtx/_verify_minimal.compare_hex`** (P4) — strict-mode FP16 BE comparator. Used for `activation_relu_gelu.elf` regression in Wave 2. No re-implementation needed.
- **`tests/gtx/_mocks.MockProcessor`** (P4 04-05 patched) — has both `state` property and `get_state()` method. P5 unit tests reuse without modification.
- **`tests/gtx/data/elf/Makefile`** (P2 → P4) — proven .elf scaffold pattern. Add `activation_relu_gelu.elf` rule mirroring `mm_basic.elf` rule.
- **`src/main/python/riscv/gtx/ops/__init__.py`** (P4) — module-level `from . import mm` triggers @handler registration via PythonBridge. P5 adds `from . import vec` and `from . import act`.

### Established Patterns

- **3-way module split for compute layers** (P4 D-01) — P5 D-01 (VEC) follows directly; P5 D-02 (ACT) uses single-engine variant of the same pattern.
- **Variant disambiguation at @handler entry, not in dispatcher** (P4 D-05) — P5 D-05 (`is_reversed` for activations) is the direct application.
- **Bit-exact strict mode for first .elf regression** (P4 D-14) — P5 carries forward; activation_relu_gelu.elf must PASS strict (no `within_tolerance > 0`).
- **Pure stateless cores as P7 numba JIT boundary** (P4 D-01 + D-03) — vec_core.py and act_core.py kernels stay array-in/array-out, no `npu` instance dependency.
- **Surgical edits within shared test files** (P4 sequencing lesson) — P5 Wave 1b plans must own non-overlapping subsets of `test_op_vec.py` / `test_op_act.py` / `test_op_format.py` / `test_pooling.py`.
- **`gsd-tools requirements mark-complete` deferral** (P4 04-01 deviation) — DON'T mark requirements complete in Wave 1a (RED scaffolds). Mark only when GREEN tests land.

### Integration Points

- **`src/main/python/riscv/gtx/ops/__init__.py`** — P5 adds `from . import vec` and `from . import act` at line ~8 (mirror existing `from . import mm` line).
- **`src/main/python/riscv/gtx/encoding.py`** — P5 appends new VEC/ACT/SCVT funct constants (P4 03-04 added `GTX_F3_MM_*`).
- **`src/main/python/riscv/gtx/dispatch_4mode.py`** (or wherever the funct7 router lives) — P5 may need to add VEC funct7=0x10/0x18..0x1F + ACT funct7=0x20/0x21/0x28/0x2A/0x2C/0x2D fillers. Investigate during research whether the routing happens via @handler decorator alone (P4 ops/mm.py pattern) or via explicit funct7 fillers in `dispatch_iss_opcode`.
- **`tests/gtx/conftest.py`** — P5 may need a `@pytest.fixture proc_with_addra_addrr_seeded` that pre-loads distinct known FP16 patterns at ADDRA and ADDRR (ROADMAP success #2 setup pattern). Place at conftest level since multiple test files use it.

### Creative Options Enabled / Constrained

- **LUT precomputation at import time** (D-14, D-15) is enabled by NumPy's vectorized fancy indexing — Python boundary cost negligible. C++ couldn't do this as cleanly because of static initialization order rules.
- **Mode B VSUM as firmware composition** (D-09) is enabled by P4 D-03 stateless kernel discipline. C++ has the same VSUM impl but firmware must explicitly orchestrate row-split loops; in Python we can document the same composition without building it into the kernel.
- **Constraint: no SIMD/AVX intrinsics** — Python+NumPy backend; CLAUDE.md "C++ 추가 코드 금지". Performance optimization deferred to P7 numba @njit on vec_core/act_core.

</code_context>

<specifics>
## Specific Ideas

### Vendor source as ground truth — do not re-derive

When research/plan/executor encounters ambiguity in any P5 op, the resolution path is:
1. Read the C++ source at `vendor/gtx_cpp_reference/gtx/gtx_npu_*.cc` for that op.
2. Mirror the math line-by-line, only changing types (`float` → `np.float32`, `uint16_t` → `np.uint16` view of `np.float16`).
3. Test via direct `np.array_equal(actual.view(uint16), expected.view(uint16))` against `verify_ref.py` oracle (P4 D-15 pattern).
4. If `verify_ref.py` doesn't cover the case, capture from a vendor-side .elf run and store as a hex fixture.

### FP8 is custom — document the divergence

The codec is **labeled** "E4M3" in C++ comments (`gtx_npu.h:153`) but has two intentional divergences from NVIDIA standard E4M3:
1. **Subnormal handling:** `h_exp == 0 && h_frac != 0` decodes as `(h_frac/8) * 2^(-6) * (-1)^sign` — uses 2^-6 base, not standard E4M3's 2^-9.
2. **exp=0xF semantics:** Maps to FP32 inf via `f_exp = 0xFF << 23`, NOT NaN as standard E4M3 (which has no inf, only NaN at the all-ones-except-mantissa pattern).

These are not bugs — they're the GTX HW spec. P5 LUT builder must match them exactly. P5 D-16 tests document the divergence.

### "First ACT .elf regression" mirror P4 mm_basic strategy

`activation_relu_gelu.elf` plays the same role for P5 that `mm_basic.elf` played for P4: prove the SPR→dispatch→DMA→compute→writeback chain works for activations specifically (with both forward + reversed exercised). Same `_verify_minimal.compare_hex(strict=True)` gate. Same vendor-borrow-or-handwrite-`.S`-fallback decision.

### VSUM dual-mode is a P5 thesis, not a footnote

The "two modes" insight (D-09 + D-10) is the most non-obvious technical decision in P5. Wave 1a scaffold should include both `test_vsum_fp32_internal_anti_pattern` and `test_vsum_row_split_matches_cpp` as RED stubs. Wave 1b plan 02 (vec_core + GREEN-fill VEC tests) makes both GREEN. Without the row-split test, a future firmware regression that exercises mode B could silently diverge from C++.

</specifics>

<deferred>
## Deferred Ideas

### Out of P5 scope (explicit deferrals to other phases)

- **Production `riscv.gtx._verify` CLI + `--strict` argument** → Phase 6 VRF-01.
- **Wheel `package_data` for `tests/gtx/data/{golden,elf}/`** → Phase 6 PKG-01.
- **`pyspike-verify` console script** → Phase 6 PKG-03.
- **Full .elf regression matrix (gem5 simplified + ISS full encoding sweeps, ALL 103 GGML kernels)** → Phase 6 VRF-04.
- **`GTX_DDR_DUMP` atexit auto-flush hook** → Phase 6 (or P4-deferred follow-up — see P4 04-05 SUMMARY).
- **Numba @njit acceleration on `vec_core.vsum`, `vec_core.dot`, `act_core.softmax`, `act_core.gelu`** → Phase 7 (P5 D-01/D-02 leaves these as stateless kernels = trivial @njit candidates).
- **CUDA / OMP / cuBLAS GEMM acceleration** → PROJECT.md Out of Scope (PCIe EP / cuBLAS in v2 reconsider).
- **DMA-3D / IM2COL / MCAST** → v2 (P3 deferred).

### Within-domain ideas surfaced but not selected for discussion

- **CLAMP `arange` variant** — generates a sequence (`v += step` per element, `gtx_npu_vec.cc:248`). Easily testable but its firmware use case is non-obvious. Lock impl in P5; defer "is this even exercised by any .elf?" to research.
- **VSUM/DOT scalar-result-to-L0 SVR[0] writeback** (`gtx_npu_vec.cc:108-110`) — IS variants need this; VS variants don't. Make sure `vec_core.vsum` returns the FP16 scalar and `vec_engine` writes it to BOTH L1 (addr_r) AND L0 SVR[0] for IS path. Document during Wave 1b.
- **`gtx_npu_pool.cc` is the CUDA thread pool, NOT NPU pooling** — NPU pooling lives in `gtx_npu_act.cc:167-220`. Do NOT read `gtx_npu_pool.cc` for NPU pooling logic — it's irrelevant to v1 (CUDA out of scope).
- **CONTAINMENT decision for `vec_engine.firmware_vec_op` granularity** (single function vs `_decode_args` + `_dispatch` + `_writeback` split) — defer to P5 plan-stage. P4 D-01 left analogous question to plan-stage and it worked fine.

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 5` returned 0 matches.

</deferred>

---

*Phase: 05-vec-act-pool*
*Context gathered: 2026-05-07*
