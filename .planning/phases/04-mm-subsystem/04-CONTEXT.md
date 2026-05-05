# Phase 4: MM Subsystem - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4는 GTX NPU의 **첫 번째 "compute" 레이어**를 구축한다. P3 데이터 평면(DMA + DDR)이 깔린 위에 처음으로 실제 연산을 올리고, **첫 `.elf` 펌웨어 회귀를 strict mode로 통과**시켜 `SPR → dispatch → DMA → compute → writeback` 전체 plumbing 정합성을 증명한다. 구체적으로:

1. **`gemm_core` FP32-internal accumulate + single FP16 cast** — `np.matmul(A.astype(f32), B.astype(f32)).astype(f16)` 기반의 순수 stateless 컴퓨트 커널 (MM-01)
2. **10 MM/MMC variant 전 구현** — `mm`, `mm_s`, `mm_o`, `mm_v`, `mm_t` (funct7=0x00) + `mmc`, `mmc_s`, `mmc_o`, `mmc_v`, `mmc_t` (funct7=0x01). 모두 별도 `@handler` 등록 (MM-02)
3. **`firmware_mm_op` packed-rs1 디코드** — `colB[63:48] | colA[31:16] | rowA[15:0]` with HW conv 0=65536 in EACH 16-bit field. `funct7=0x00` collision은 `insn.rs1 != 0` heuristic으로 WRSPR/MM 분기 (MM-03)
4. **`mxe_accum` 체인 동작** — 클래스 state `npu._mxe_accum: ndarray((4, 16), float32)`을 mmc 계열이 read+add+write. mm 계열은 write only. `mm.s → mmc.s → mmc` 체인이 C++과 동등 (MM-04)
5. **첫 `.elf` 펌웨어 회귀 strict mode 통과** — `mm_basic.elf` (vendor 차용) 실행 → DDR dump → `_verify_minimal.compare_hex(strict=True)` PASS (MM-05)
6. **Mode 4 (P+T) dispatcher inner payload 채우기** — P3에서 routing은 wired, dispatch_iss_opcode의 funct7=0x00/0x01 case가 P4 차례

다음 모두는 **Phase 4 비범위(out-of-scope)** — 다른 페이즈가 다룬다:

- **VEC/ACT/Pool op 핸들러** → Phase 5 VEC-01..05, ACT-01..05
- **`verify.py` production 포팅 + CLI** → Phase 6 VRF-01 (P4는 mini 직역만)
- **`tests/gtx/data/{golden,elf}/` 패키지 데이터 등록** → Phase 6 PKG-01
- **`pyspike-verify` console script** → Phase 6 PKG-03
- **자동 DDR dump (atexit hook)** → Phase 6 또는 별도 follow-up (P3 D-09 상점 유지)
- **format_cvt + FP8 codec** → Phase 5
- **DMA-3D / IM2COL / MCAST** → v2 (Phase 3 deferred)
- **dispatch_iss_opcode VEC/ACT funct7 fillers** → Phase 5
- **Numba @njit 가속 적용** → Phase 7 (P4는 numba-friendly 구조만 마련)

</domain>

<decisions>
## Implementation Decisions

### MM 모듈 구성 (D-01 ~ D-04)

- **D-01:** **3-way module split** — `riscv/gtx/ops/mm.py` (@handler 진입점) + `riscv/gtx/mm_engine.py` (firmware_mm decode + variant dispatcher + mxe_accum read/write 책임) + `riscv/gtx/gemm_core.py` (순수 stateless NumPy GEMM 커널 1개 함수).
  - **이유:** P3 `dma_engine` 2-way split이 잘 작동했으나 P4는 P7 numba 동적 최적화 페이즈를 염두에 두고 더 명확한 boundary 필요. `gemm_core.py`는 P7에서 `@njit`로 감쌀 단일 hot 함수 — JIT boundary 명확화. `mm_engine.py`는 spike-bound 디코드/state, `gemm_core.py`는 array-in/array-out 순수.
  - **위험:** 파일 3개 늘어남 → import 파이프 살짝 복잡. 측정 후 P5에서 비슷한 결정 시 재고 가능.
- **D-02:** **`gemm_core` 구현 = `np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np.float16)`** — single-line, FP32 internal accumulate (BLAS 자동), single FP16 cast.
  - **이유:** PITFALLS Pitfall 2 정합 (FP32-internal). NumPy BLAS 호출이 explicit 3-loop보다 100x↑ 빠르며 P4에서 baseline 확보 후 P7에서 numba로 가속하는 그림. C++ scalar 3-loop 대비 bit-exact 여부는 research가 검증 (BLAS 누적 순서 차이 가능성 있음 — research가 확인 후 plan에서 explicit 3-loop fallback 옵션 제공할 수도).
  - **위험:** BLAS implementation defined 누적 순서가 C++ scalar 3-loop과 다르면 ULP-level drift 발생. 그 경우 explicit 3-loop으로 fallback (P7에서 numba 가속) — research가 lock.
- **D-03:** **`gemm_core` stateless API:** `gemm_core(A: ndarray, B: ndarray, *, has_bias: bool = False, prior_accum: float = 0.0) -> tuple[ndarray, float]` — array-in/scalar-in/array-out/scalar-out. mxe_accum은 호출자(`mm_engine`)가 read/write 책임.
  - **이유:** P7 numba `@njit`가 그대로 감쌀 수 있는 형태. npu/GtxMemory 인스턴스 의존 0. 단위 테스트가 mock 없이 가능. mxe_accum의 클래스 state 책임은 `mm_engine`에만 있음 → scope 격리.
  - **시그니처 정확화는 plan 단계에서** — `prior_accum` scalar vs ndarray, `has_bias` 필요성, 반환 tuple 순서.
- **D-04:** **10개 MM/MMC variant 모두 별도 `@handler` 함수 등록** — `_exec_mm`, `_exec_mm_s`, `_exec_mm_o`, `_exec_mm_v`, `_exec_mm_t`, `_exec_mmc`, `_exec_mmc_s`, `_exec_mmc_o`, `_exec_mmc_v`, `_exec_mmc_t`.
  - **이유:** P3 dma 패턴 (9 active @handler) 일관. 각 variant 고유 funct3 → @handler가 disasm 자동 등록. spike trace에 정확한 mnemonic 노출. P7 numba JIT은 closure factory보다 individual function이 더 잘 가속됨.
  - **mnemonic ↔ funct3 매핑 정확값은 research가 잠금** — `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc`에서 추출.

### `is_accumulate` 분기 + mxe_accum read/write 위치 (D-05 ~ D-08)

- **D-05:** **`is_accumulate` 분기는 `ops/mm.py` 진입점에서.** mmc 계열 (funct7=0x01)은 `is_accumulate=True` 명시 전달, mm 계열 (funct7=0x00)은 `False`. `_exec_mmc_o(npu, proc, insn, xs1, xs2)`가 `mm_engine.firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc_o')` 호출.
  - **이유:** PITFALLS Pitfall 3 정합 (`funct7==0x01` driven). C++ `gtx_npu_mm.cc:333` 직역. 진입점에서 분기 → routing 사고 0. funct7 collision 디버깅도 진입점에서 명확.
- **D-06:** **`mxe_accum` read/write는 `ops/mm.py` 진입점.** `_exec_mmc_o`는:
  1. `nest = npu.tmu_id; spu = npu.curr_id` (정확 필드명은 P2 WarpState — research lock)
  2. `prior = float(npu._mxe_accum[nest, spu])` read
  3. `result, new_accum = gemm_core(A, B, has_bias=True, prior_accum=prior)` 호출
  4. `npu._mxe_accum[nest, spu] = new_accum` write back
  5. result를 L1으로 store
  - **이유:** mxe_accum 클래스 state read/write를 한 layer에 격리 → scope 사고 0. `gemm_core`는 npu 의존 0 (D-03). PITFALLS Pitfall 3의 3가지 실수 (per-call wipe / dtype slip / NEST,SPU scope 누락) 모두 여기서 잡힘.
- **D-07:** **Pitfall 3 anti-pattern test = `mm.s → mmc.s → mmc` chain 단일 테스트** (ROADMAP success #2 그대로).
  - 동일한 `(nest, spu)`에서 3-step 체인 실행 → final FP16 결과가 `np.float16(A1@B1 + A2@B2 + A3@B3)` (FP32 internal)와 일치. 단일 fused `mm`이 concatenated 입력에서 다른 (known) 값 반환.
  - 추가 검증 (D-11 dual-assertion 패턴 mirror): pre-call `mxe_accum.copy()` snapshot → mmc.s on (1, 5) → `mxe_accum[1, 5]`만 변경, 나머지 (4×16-1) 셀 unchanged.
- **D-08:** **funct7=0x00 collision (WRSPR vs MM) parametrized matrix test** (ROADMAP success #3 그대로). parametrize over `funct7 ∈ {0x00, 0x01}`, `has_rs1 ∈ {True, False}`로 4 케이스. `(0x00, False)` → WRSPR 라우팅 / `(0x00, True) | (0x01, *)` → MM/MMC 라우팅. P2 D-02 heuristic 잠금 검증.

### `.elf` 회귀 fixture 전략 (D-09 ~ D-12)

- **D-09:** **`mm_basic.elf`는 vendor/gtx_cpp_reference/에서 차용** (정확한 path는 research lock). **fallback:** vendor에 적합한 fixture 없으면 P2 D-22 패턴 (`tests/gtx/data/elf/{mm_basic.S, Makefile, mm_basic.elf}` 사전 빌드 커밋) — `/opt/riscv/` 툴체인 가진 개발자만 `make` 가능.
  - **이유:** vendor에 이미 C++ libgtx_npu.so SystemC HW sim ULP 일치 검증 완료된 .elf 자산이 있을 가능성 ↑. 빌드 인프라 도입 부담 0. SHA 동기화 이슈 (P2 D-16)는 P2 plan-02에서 해결됨.
- **D-10:** **golden hex `mm_basic_n1s16.hex`도 vendor에서 차용.**
  - **이유:** ROADMAP P4 success #4가 `tests/gtx/data/golden/mm_basic_n1s16.hex` 명시. C++ 출력이 SystemC HW sim과 ULP 일치 검증 완료된 자산이라 `--strict`의 ground-truth로 정합. NumPy oracle을 P4 자체에서 합성하면 'C++ libgtx_npu.so 일치'를 증명 못 함.
- **D-11:** **`.elf` 구동은 pytest in-process** — `tests/gtx/test_regression_mm.py` 내부에서 subprocess 없이 GtxNpu 인스턴스 + ELF 메모리 로드 + 펌웨어 진입점 호출.
  - **fallback:** GIL/single-hart/spike pthread 도석 이슈 발견 시 P2 `test_skeleton.py` 패턴(`subprocess.run([pyspike, '--extlib=riscv.gtx', mm_basic.elf])`)으로 스위치.
  - **이유:** 빠르고, debug 단순, GIL 안전. spike는 hart 제어하지만 P4 .elf는 single-hart firmware라 in-process로 충분할 가능성 ↑.
- **D-12:** **`GTX_DDR_DUMP` 처리는 P4에서 처음 도입 — but 테스트 내부 명시 호출만.** P4 회귀 테스트가 펌웨어 종료 후 `ddr_dump_to_file(npu.mem, '/tmp/dump.hex', addr=DUMP_ADDR, size=DUMP_SIZE)` 명시 호출. **production atexit hook은 P6.**
  - **이유:** P3 D-09 상점 유지 (라이브러리 함수 깨끗 + CLI/진입점 책임은 P6). pyspike CLI subprocess 모드(D-11 fallback)에서는 process 종료 후 명시 dump 불가하므로 in-process 모드 우선 선택과 정합.

### Strict-mode 검증 인프라 (D-13 ~ D-15)

- **D-13:** **`tests/gtx/_verify_minimal.py` mini 직역** (~30 LOC).
  - 시그니처: `def compare_hex(actual_path: str, golden_path: str, *, ulp: int = 1, atol: float = 0.001, strict: bool = True) -> tuple[bool, dict]` — bool은 PASS/FAIL, dict는 {`exact_matches`, `within_tolerance`, `failures`, `total_fp16`, `first_failure: tuple[int, int, int] | None`}.
  - 위치: `tests/gtx/_verify_minimal.py` (production 미포함 — `tests/gtx/_mocks.py`와 같은 등급).
  - 직역 source: `verify.py` (vendor 또는 ROADMAP 명시 인자 시그니처).
  - **P6에서 `riscv.gtx._verify` (CLI 포함)로 승격** — P6 VRF-01.
- **D-14:** **P4도 strict 강제** — `within_tolerance > 0` (정확 일치는 아니지만 ULP 1 내)도 failure로 보고. ROADMAP success #4 그대로.
  - **이유:** drift 이슈는 첫 .elf 회귀에서 잡는 게 가장 저렴. ULP 허용 모드로 통과시키면 P5/P6에서 누적 drift가 더 큰 비용.
- **D-15:** **Op-level unit test는 `np.array_equal(actual.view(np.uint16), expected.view(np.uint16))` 직접** (ROADMAP success #1 그대로). `_verify_minimal`은 `.elf` 회귀에만 사용.
  - **이유:** unit test는 16×16×16 GEMM이 NumPy oracle과 직접 일치하는지 검증 — hex round-trip 불필요. error message가 어느 셀이 깨졌는지 명확 (op-level dev experience).

### Claude's Discretion

다음은 implementation detail로 Claude 판단 (research/plan 단계에서 정확화):

- `gemm_core` 정확한 signature (prior_accum scalar vs ndarray view, has_bias 필요 여부, 반환 tuple 순서)
- `np.matmul` BLAS 누적 순서가 C++ scalar 3-loop과 bit-exact인지 — research lock. 차이 발견 시 explicit 3-loop으로 fallback
- 10 MM/MMC variant funct3 정확한 매핑 (research lock from `gtx_npu_disasm.inc`)
- `mm_engine.py` 내 함수 분리 정도 (`firmware_mm` 단일 함수 vs `_decode_args` + `_dispatch_variant` + `_writeback` 분리)
- `MM_T` (transposed B) / `MM_V` (vector) variant의 정확한 데이터 경로 — research가 C++ 직역 source 잠금
- `mxe_accum[nest, spu]` 정확한 nest/spu 추출 (npu.tmu_id / npu.warp.tmu_id / npu._tmu_id — P2 WarpState 필드 정확명 plan 단계)
- `dispatch_iss_opcode`에서 funct7=0x00/0x01 케이스 추가 정확한 위치 (`dispatch_4mode.py` body 확장 vs ops/mm.py 진입점이 직접 dispatch_4mode 호출 우회)
- `_verify_minimal.py` 정확한 구현 (FP16 BE-packing 처리 — PITFALLS Pitfall 1 정합)
- `mm_basic.elf` 위치가 vendor에 없을 시 fallback 방식 (.S 직접 작성 vs cross-toolchain 외부 의존)
- in-process .elf 로드의 정확한 진입점 (Python에서 ELF 로드 + entry point 호출 메커니즘 — research lock)

### Folded Todos

None — `gsd-tools todo match-phase 4`에서 매칭 0건.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project documents (locked context)
- `.planning/PROJECT.md` — Core Value (bit-exact w/ C++ libgtx_npu.so), Constraints (Python+NumPy only, no C++ additions), Out of Scope, Key Decisions
- `.planning/REQUIREMENTS.md` — MM-01..05 v1 acceptance criteria
- `.planning/ROADMAP.md` — Phase 4 success criteria 1-5 (특히 #2 chain test, #3 funct7 routing, #4 strict mode, #5 Mode 4 isolation) + research-flag 노트
- `.planning/STATE.md` — 현재 진행 (Phase 1, 2, 3 완료; Phase 4 ready to plan). Phase 2 STATE 정정: mxe_accum은 2D `(NEST, SPU)` float32
- `.planning/research/PITFALLS.md` — **§Pitfall 2 (VSUM/MM_O FP32-internal accumulate, single FP16 cast)**, **§Pitfall 3 (mxe_accum continuity across MM_O / MM_V chains, scope per-(NEST,SPU))**
- `.planning/research/SUMMARY.md` — high-level synthesis
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-09 NumPy view, D-10..D-12 layered API + view 보장, D-13 모듈 layout
- `.planning/phases/02-skeleton-disasm/02-CONTEXT.md` — D-02 funct7=0x00 collision heuristic, D-04 WarpState dataclass, D-06 mxe_accum (정정 in STATE), D-13/D-14 per-op decorator registry, D-19 mock spec, D-22 .elf fixture 패턴
- `.planning/phases/03-dma-ddr-i-o/03-CONTEXT.md` — D-01 ops + engine 분리 패턴 (P4가 mirror), D-03 2-level dispatch, D-09 GTX_DDR_DUMP 위치 (P4가 D-12에서 'tests 내부만' 유지), D-11 dual-assertion 테스트 (P4 mxe_accum 격리 테스트가 mirror), D-14 dispatch_4mode 구조

### Phase 4 research (생성 예정)
- `.planning/phases/04-mm-subsystem/04-RESEARCH.md` — `/gsd:research-phase 4`가 plan 전에 생성. 다음을 잠금:
  - `firmware_mm_op` rs1 packed bit layout 정확한 마스크 + HW conv 0=65536 in EACH 16-bit field (`colB[63:48] | colA[31:16] | rowA[15:0]`)
  - 10 MM/MMC variant funct3 정확한 매핑 from `gtx_npu_disasm.inc`
  - `np.matmul` BLAS 누적 순서가 C++ scalar 3-loop과 bit-exact인지 (실측 또는 ground-truth 비교)
  - `funct7==0x00 insn.rs1!=0` heuristic이 실제 mm_basic.elf 어셈블리에서 발화되는 방식
  - `mm_basic.elf` + golden hex의 vendor 내 정확한 path
  - in-process .elf 로드 메커니즘 (subprocess fallback 필요 여부)
  - `dispatch_iss_opcode` body 확장 위치 (dispatch_4mode.py vs ops/mm.py 진입점)
  - WarpState `tmu_id`/`curr_id` 정확한 필드 위치/접근

### C++ ground-truth (via submodule, Phase 1 D-04)

**Primary (P4 핵심 직역 대상):**
- `vendor/gtx_cpp_reference/gtx/gtx_npu_mm.cc` — ~400 LOC. P4 직역 대상:
  - `firmware_mm` (firmware_mm_op 진입점) — rs1 packed decode + variant dispatch
  - `gemm_core` (line 60-) — FP32 std::vector C 누적 + FP16 cast (D-02 source)
  - `exec_mm` / `exec_mm_s` / `exec_mm_o` / `exec_mm_v` / `exec_mm_t` (전체 5개 mm)
  - `exec_mmc` / `exec_mmc_s` / `exec_mmc_o` / `exec_mmc_v` / `exec_mmc_t` (5개 mmc)
  - mxe_accum read/write site (line ~209-212): MM_O 초기화, MMC_O continuation
  - `is_accumulate` 결정 로직 (line ~333): funct7==0x01 driven (D-05 source)
- `vendor/gtx_cpp_reference/gtx/gtx_npu_disasm.inc` — MM 관련 mnemonic 영역 (D-04 funct3 매핑 source)
- `vendor/gtx_cpp_reference/gtx/gtx_npu.h` — `mxe_accum[GTX_NUM_NESTS][GTX_SPUS_PER_NEST]` 정의 (line ~1254 — STATE 정정 source), `firmware_mm_op` packed-rs1 매크로 정의 가능성
- `vendor/gtx_cpp_reference/gtx/gtx_npu_dispatch.cc` — `dispatch_iss_opcode` 본체 (line 151-): funct7=0x00/0x01 케이스 (D-claudemissing source)
- `vendor/gtx_cpp_reference/gtx/gtx_params.h` — GTX_NUM_NESTS=4, GTX_SPUS_PER_NEST=16, M_TILE/N_TILE 상수 (P2 D-06 정정 source)

**Secondary (참고):**
- `vendor/gtx_cpp_reference/gtx/CLAUDE.md` — 메모리 계층, FP16 LE 바이트 순서 규약, MM operand orientation 명세
- `vendor/gtx_cpp_reference/.../verify.py` — D-13 mini 직역 source (인자 시그니처 + ULP/atol/strict 로직)
- `vendor/gtx_cpp_reference/.../mm_basic.elf` — D-09 fixture 차용 source (정확한 path는 research lock)
- `vendor/gtx_cpp_reference/.../mm_basic_n1s16.hex` — D-10 golden hex 차용 source

### Existing pyspike + Phase 1/2/3 산출물 (이미 wheel/repo에 land)

**Phase 1 자산:**
- `src/main/python/riscv/gtx/{__init__.py, params.py, encoding.py, fp.py, memory.py, ddr.py}` — P4가 직접 수정 안 함 (단 `encoding.py`에 funct7=0x00/0x01 + funct3 MM/MMC 매핑 추가 가능)
- `src/main/python/riscv/gtx/memory.py` — `GtxMemory.l1_f16(nest, spu)` / `.l2_f16(nest)` view API (D-12 P1) — P4 GEMM의 A/B/C 메모리 접근 경로

**Phase 2 자산:**
- `src/main/python/riscv/gtx/npu.py` — `GtxNpu._mxe_accum: np.ndarray((4, 16), float32)` 이미 land됨 (P4가 read/write). `reset()`이 `_mxe_accum.fill(0.0)` 이미 land됨
- `src/main/python/riscv/gtx/_registry.py` — `@handler` 데코레이터 (P3에서 2-level mask_funct3=True 활성화). P4 ops/mm.py가 동일 패턴 사용
- `src/main/python/riscv/gtx/dispatch.py` — `build_custom0_table` 2-level (P3). P4가 funct7=0x00, 0x01 추가 자동 등록
- `src/main/python/riscv/gtx/warp_state.py` — `WarpState(is_ploop, is_tloop, is_sloop, ...)` (P2 D-04 + P3 wsplit_seen). P4가 `tmu_id`/`curr_id` 필드 read (정확한 이름은 plan lock)
- `src/main/python/riscv/gtx/ops/spr.py` — P2 SPR 핸들러 (P4 직접 수정 안 함, 단 funct7=0x00 collision 발화 시 통과 경로 검증)
- `src/main/python/riscv/gtx/ops/control.py` — P2/P3 warp 제어 + flush 트리거 (P4 직접 수정 안 함)

**Phase 3 자산:**
- `src/main/python/riscv/gtx/ops/dma.py` — DMA @handler 16개 entry (P4 직접 수정 안 함, MM이 DMA를 firmware-level로 호출만)
- `src/main/python/riscv/gtx/dma_engine.py` — DMA exec_* 6 helpers + `decode_firmware_dma_args` 직역 패턴 (D-03 mirror source — `decode_firmware_mm_args`도 동일 패턴)
- `src/main/python/riscv/gtx/ddr.py` — doubling-grow ensure_ddr + ddr_init/dump_to_file (P4 .elf 회귀 dump path)
- `src/main/python/riscv/gtx/dispatch_4mode.py` — 4-mode 라우터 (P3 D-14). Mode 4 routing wired, **dispatch_iss_opcode body의 funct7=0x00/0x01 case가 P4 차례** (D-claudemissing — Claude's Discretion: dispatch_4mode.py 확장 vs ops/mm.py 진입점이 dispatch_4mode 호출 우회)

**Phase 2/3 테스트 자산:**
- `tests/gtx/conftest.py` + `tests/gtx/_mocks.py` — Hybrid mock (D-17/D-19 P2). P4가 GPR mock + 가능 시 단순 ELF 메모리 로더 mock 추가 (D-11 in-process)
- `tests/gtx/test_register.py` / `test_reset.py` / `test_dispatch.py` / `test_warp.py` / `test_skeleton.py` — P4 신규 테스트는 동일 패턴 (`_RISCV_AVAILABLE` self-detect)
- P3 회귀 (test_dma_*, test_ddr_*, test_dispatch_4mode, test_deferred_store, test_dma_roundtrip) — P4 도입 시 회귀 0 (179/179 그린 유지)

### Build / Distribution
- `pyproject.toml` — P4 직접 수정 없음 (PKG-01 ELF/golden hex 등록은 P6)
- `MANIFEST.in` — Phase 1 vendor prune 적용됨, P4 추가 작업 없음

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 1/2/3 산출물 — P4가 직접 사용/확장)

- **`riscv.gtx.memory.GtxMemory`** — `l1_f16(nest, spu)` / `l2_f16(nest)` view API. P4 `mm_engine`이 A/B 입력은 L1 (또는 L2) view로 read, C 출력은 L1 view에 write. `arr.base is not None` view 보장 (Phase 1 D-12).
- **`riscv.gtx.npu.GtxNpu._mxe_accum`** — `(4, 16)` float32 ndarray. P4 `_exec_mmc_*`가 `npu._mxe_accum[nest, spu]` read/write. `reset()`만 zero (PITFALLS Pitfall 3).
- **`riscv.gtx.npu.GtxNpu.tmu_id` / `curr_id`** — P2 WarpState 필드 (정확명 plan lock). P4 진입점이 `nest = npu.tmu_id; spu = npu.curr_id` 추출 후 mxe_accum index.
- **`riscv.gtx._registry.handler`** — P2 D-13 + P3 mask_funct3=True. P4 `ops/mm.py` 모든 핸들러가 `@handler(kind='custom0', funct7=0x00, funct3=N, mnemonic='mm_*', mask_funct3=True)`로 등록.
- **`riscv.gtx.dispatch.build_custom0_table`** — P3 2-level (`dict[funct7, dict[funct3 or None, Callable]]`). P4가 funct7=0x00/0x01 + funct3 0..4 추가 자동 등록.
- **`riscv.gtx.dma_engine.decode_firmware_dma_args`** — `(rs1, rs2, rs3) -> dict` 패턴. P4의 `mm_engine.decode_firmware_mm_args`가 동일 패턴 mirror (`rs1 packed → {rowA, colA, colB}` with HW conv 0=65536).
- **`riscv.gtx.dispatch_4mode.dispatch_iss_opcode`** — P3 DMA-only 스텁. P4가 funct7=0x00/0x01 case body 채움 — **OR** ops/mm.py가 dispatch_4mode를 직접 호출하고 dispatch_iss_opcode는 `dispatch_4mode → 미통과` (Claude's Discretion).
- **`riscv.gtx.dma_engine.exec_dma_2d`** — Mode 3 caller. P4 firmware는 일반적으로 DMA로 operand 로드 후 MM 실행 → MM op 자체는 직접 dma_engine 호출 안 함 (단 firmware가 직접 dma instruction emit).
- **`tests/gtx/_mocks.py` `MockProcessor.get_state().XPR.read/write`** — P4 firmware_mm rs1/rs2 read 경로. ELF 로드 mock은 D-11 in-process 모드면 추가 가능 (research lock).

### Established Patterns

- **모듈 명명:** lowercase + underscore (`mm_engine.py`, `gemm_core.py` 후보 + `ops/mm.py`).
- **Type hints:** explicit, mypy-checked (`pytest --mypy`).
- **TDD:** test_*.py RED → 모듈 GREEN → 픽스 (P1, P2, P3에서 확립).
- **Test 격리:** `--noconftest -o "addopts="` 우회 + 모듈 레벨 `_RISCV_AVAILABLE` self-detect (P2 plan 05 D-1).
- **`@handler(kind='custom0', funct7=, funct3=, mnemonic=, mask_funct3=)` 데코레이터** — P3 mask_funct3=True 표준화. P4 동일 패턴.
- **TDD pattern (P3 확립):** test_op_mm.py 작성 → mm_engine.py + gemm_core.py 함수 작성 → ops/mm.py @handler 진입점 작성 → 통합 테스트.
- **C++ 직역 + 명시적 byte 조작:** P3 LE 가정 유지. MM 결과 L1 write도 동일.
- **dual-assertion 테스트** (P3 D-11 mirror): mxe_accum chain test가 (a) 최종 FP16 일치 + (b) per-cell scope 격리 둘 다 검증.
- **Parallel wave 커밋:** P3 입증 — `--no-verify` for parallel agents, post-wave hook 한 번 검증.
- **3-way 모듈 split:** P4가 처음 도입 — P3 2-way에서 진화. P5 VEC/ACT도 같은 패턴 따를지는 P5 discuss에서 재고.

### Integration Points

- **`riscv.gtx.ops.mm`** (NEW MODULE) — 10 @handler 진입점 (D-04). `dispatch_4mode.dispatch_iss_opcode`가 funct7=0x00/0x01 routing 시 호출하거나, mm.py 진입점이 dispatch_4mode를 직접 호출 (Claude's Discretion).
- **`riscv.gtx.mm_engine`** (NEW MODULE) — `firmware_mm`, `decode_firmware_mm_args`, MM/MMC variant 분기 helpers. spike-bound (npu/proc/insn 인자 받음).
- **`riscv.gtx.gemm_core`** (NEW MODULE) — `gemm_core(A, B, *, has_bias, prior_accum) -> tuple[ndarray, float]`. 순수 stateless. P7 numba `@njit` 후보.
- **`riscv.gtx.dispatch_4mode`** — D-claudemissing: `dispatch_iss_opcode` body에 funct7=0x00 (collision check after rs1!=0), funct7=0x01 case 추가 (or 우회).
- **`riscv.gtx.encoding`** — P4가 funct7 0x00/0x01 + funct3 0..4 (mm/mm_s/mm_o/mm_v/mm_t) 상수 + `GTX_OP_MM` 매크로 추가.
- **`tests/gtx/test_op_mm.py`** (NEW) — 10 variant unit + chain test (D-07) + funct7 routing matrix (D-08).
- **`tests/gtx/test_mm_chain.py`** (NEW or merged into test_op_mm.py) — ROADMAP success #2 mxe_accum chain.
- **`tests/gtx/test_regression_fw_mm.py`** (NEW) — `.elf` strict mode 회귀 (D-09 ~ D-15).
- **`tests/gtx/_verify_minimal.py`** (NEW) — D-13 mini 직역. P6에서 production `riscv.gtx._verify`로 승격.
- **`tests/gtx/data/elf/mm_basic.elf`** (vendor 차용 또는 사전 빌드 — D-09).
- **`tests/gtx/data/golden/mm_basic_n1s16.hex`** (vendor 차용 — D-10).

### Anti-patterns to avoid (PITFALLS.md / P4 추가)

- **`np.float16` array에서 직접 GEMM accumulate** — Pitfall 2 위배. 항상 FP32 internal (D-02).
- **`mxe_accum`을 메서드 local 또는 per-call wipe** — Pitfall 3 위배. 클래스 state, `reset()`만 zero (D-06).
- **`mxe_accum` dtype을 float16 또는 float64로 슬립** — Pitfall 3 위배. 항상 float32 (npu.py:54-58 lock).
- **`mxe_accum[nest, spu]` 대신 flat index** — Pitfall 3 위배. `(nest, spu)` 2-tuple 명시 access.
- **`is_accumulate`를 call count로 판단** — Pitfall 3 위배. `funct7 == 0x01` driven (D-05).
- **`gemm_core`에 npu/GtxMemory 인자 전달** — D-03 위배 (stateless API). array-in/scalar-in/array-out/scalar-out만.
- **`np.matmul(A.float16, B.float16)` 후 `.astype(float16)`** — FP16-internal accumulate 트랩. 반드시 `.astype(np.float32)` 후 matmul.
- **`@handler` 10개 대신 closure factory loop** — D-04 위배. P7 numba JIT 친화도 ↓.
- **`_verify_minimal`에 production CLI 추가** — D-13 위배. P4는 helper만, P6에서 promote.
- **op-level test에서 `_verify_minimal` 통한 hex round-trip** — D-15 위배. `np.array_equal(...view(uint16))` 직접.
- **`GTX_DDR_DUMP` env var를 `ddr_dump_to_file` 안에서 read** — P3 D-09 위배. 명시 호출만.
- **`mm_basic.elf` build script를 P4 conftest에 추가** — D-09 fallback. 사전 빌드 .elf 커밋 우선.
- **subprocess pyspike를 unit test에서 사용** — D-11 위배 (in-process 우선). integration-only fallback.
- **mm_engine을 spike-independent로 만들려고 npu/proc/insn 인자 분해 추상화** — over-engineering. dma_engine은 bound, gemm_core만 pure (D-03).
- **MMC_O `is_accumulate` 결정을 위해 npu.last_mm_call 같은 spy attribute 추가** — D-08 위배. funct7만으로 결정.

</code_context>

<specifics>
## Specific Ideas

### 모듈 표면 시안 (참고용 — plan/research 단계에서 정확화)

#### `riscv/gtx/gemm_core.py` (~30 LOC, P7 numba @njit 후보)

```python
"""Pure stateless GEMM kernel. P4 baseline; P7 numba @njit boundary."""
import numpy as np
from numpy.typing import NDArray


def gemm_core(
    A: NDArray[np.float16],
    B: NDArray[np.float16],
    *,
    has_bias: bool = False,
    prior_accum: float = 0.0,
) -> tuple[NDArray[np.float16], float]:
    """C = (A @ B + prior_accum if has_bias else 0) cast to FP16.

    FP32 internal accumulate (PITFALLS Pitfall 2). Single FP16 cast on output.
    Returns: (C, new_accum_scalar)
        - C: FP16 (M, N) result
        - new_accum_scalar: FP32 scalar for mxe_accum chain (== sum of all elements
          for MMC_O lineage; exact semantics locked by research from gtx_npu_mm.cc)
    """
    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)
    C_f32 = np.matmul(A_f32, B_f32)
    if has_bias:
        C_f32 += prior_accum  # broadcast scalar
    new_accum = float(C_f32.sum())  # research locks exact semantics
    return C_f32.astype(np.float16), new_accum
```

#### `riscv/gtx/mm_engine.py` 표면 시안

```python
"""Spike-bound MM dispatcher. Decodes firmware_mm_op, reads/writes mxe_accum."""
from . import gemm_core


def decode_firmware_mm_args(rs1: int, rs2: int) -> dict:
    """rs1 = colB[63:48] | colA[31:16] | rowA[15:0]
    HW conv: 0 → 65536 in EACH 16-bit field.
    """
    rowA = rs1 & 0xFFFF or 0x10000
    colA = (rs1 >> 16) & 0xFFFF or 0x10000
    colB = (rs1 >> 48) & 0xFFFF or 0x10000
    return {'rowA': rowA, 'colA': colA, 'colB': colB, 'rs2': rs2}


def firmware_mm(npu, proc, insn, *, is_accumulate: bool, variant: str) -> int:
    """All 10 MM/MMC variants flow through here. variant ∈
    {'mm', 'mm_s', 'mm_o', 'mm_v', 'mm_t', 'mmc', 'mmc_s', 'mmc_o', 'mmc_v', 'mmc_t'}.
    """
    rs1 = proc.get_state().XPR[insn.rs1]
    rs2 = proc.get_state().XPR[insn.rs2]
    args = decode_firmware_mm_args(rs1, rs2)
    nest, spu = npu.tmu_id, npu.curr_id  # exact field name plan-lock

    A, B = _read_operands(npu.mem, nest, spu, variant, args)
    prior = float(npu._mxe_accum[nest, spu]) if is_accumulate else 0.0
    C, new_accum = gemm_core.gemm_core(A, B, has_bias=is_accumulate, prior_accum=prior)
    if is_accumulate:
        npu._mxe_accum[nest, spu] = new_accum

    _write_result(npu.mem, nest, spu, variant, C)
    return 0
```

#### `riscv/gtx/ops/mm.py` 표면 시안

```python
"""Custom0 funct7=0x00/0x01 MM/MMC handler entry points."""
from .._registry import handler
from .. import mm_engine


@handler(kind='custom0', funct7=0x00, funct3=0, mnemonic='mm', mask_funct3=True)
def _exec_mm(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm')


@handler(kind='custom0', funct7=0x00, funct3=1, mnemonic='mm_s', mask_funct3=True)
def _exec_mm_s(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn, is_accumulate=False, variant='mm_s')


# ... mm_o, mm_v, mm_t ...

@handler(kind='custom0', funct7=0x01, funct3=0, mnemonic='mmc', mask_funct3=True)
def _exec_mmc(npu, proc, insn, xs1, xs2):
    return mm_engine.firmware_mm(npu, proc, insn, is_accumulate=True, variant='mmc')


# ... mmc_s, mmc_o, mmc_v, mmc_t ...
```

### `_verify_minimal.py` 표면 시안 (~30 LOC)

```python
"""Test-only strict-mode verify. P6 promotes to riscv.gtx._verify with CLI."""
import numpy as np


def compare_hex(actual_path, golden_path, *, ulp=1, atol=0.001, strict=True):
    """Return (passed: bool, stats: dict).

    PITFALLS Pitfall 1: hex compare uses BE FP16 packing
    (data[i*2] << 8) | data[i*2+1].
    """
    def parse(path):
        return bytes.fromhex(open(path).read().replace('\n', '').replace(' ', ''))

    a_bytes, e_bytes = parse(actual_path), parse(golden_path)
    n = min(len(a_bytes), len(e_bytes)) // 2
    a_u16 = np.frombuffer(a_bytes[:n*2], dtype='>u2')  # BE
    e_u16 = np.frombuffer(e_bytes[:n*2], dtype='>u2')
    exact = int(np.sum(a_u16 == e_u16))

    a_f16 = a_u16.view(np.uint16).newbyteorder().view(np.float16).astype(np.float64)
    e_f16 = e_u16.view(np.uint16).newbyteorder().view(np.float16).astype(np.float64)
    within = int(np.sum(np.abs(a_f16 - e_f16) <= atol)) - exact

    failures = n - exact - within
    first_fail = next((i for i in range(n) if a_u16[i] != e_u16[i]
                       and np.abs(a_f16[i] - e_f16[i]) > atol), None)
    stats = {'exact_matches': exact, 'within_tolerance': within,
             'failures': failures, 'total_fp16': n, 'first_failure': first_fail}
    if strict:
        return exact == n, stats
    return failures == 0, stats
```

### test_mm_chain.py assertion 골자 (D-07)

```python
def test_mm_chain_mxe_accum_continuity():
    """mm.s → mmc.s → mmc on same (nest=1, spu=5).
    Final FP16 result == np.float16(A1@B1 + A2@B2 + A3@B3) FP32 internal.
    """
    npu = GtxNpu()
    npu.reset(MockProcessor())
    nest, spu = 1, 5
    npu._tmu_id = nest  # plan-lock attribute name
    npu._curr_id = spu

    A1, B1 = _make_random_fp16((16, 16)), _make_random_fp16((16, 16))
    A2, B2 = _make_random_fp16((16, 16)), _make_random_fp16((16, 16))
    A3, B3 = _make_random_fp16((16, 16)), _make_random_fp16((16, 16))

    # Step 1: mm.s — initialize
    snapshot_before = npu._mxe_accum.copy()
    _load_operands_to_l1(npu.mem, nest, spu, A1, B1)
    npu.custom0(_synthetic_insn(funct7=0x00, funct3=1), 0, 0)  # mm_s

    # Step 2: mmc.s — accumulate
    _load_operands_to_l1(npu.mem, nest, spu, A2, B2)
    npu.custom0(_synthetic_insn(funct7=0x01, funct3=1), 0, 0)  # mmc_s

    # Step 3: mmc — accumulate again
    _load_operands_to_l1(npu.mem, nest, spu, A3, B3)
    npu.custom0(_synthetic_insn(funct7=0x01, funct3=0), 0, 0)  # mmc

    # FP32 internal oracle
    expected = np.float16(
        np.matmul(A1.astype(np.float32), B1.astype(np.float32))
        + np.matmul(A2.astype(np.float32), B2.astype(np.float32))
        + np.matmul(A3.astype(np.float32), B3.astype(np.float32))
    )
    actual = _read_result_from_l1(npu.mem, nest, spu)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

    # Per-(NEST,SPU) isolation: only [1, 5] cell changed
    other_cells_before = np.delete(snapshot_before.flatten(), nest * 16 + spu)
    other_cells_after = np.delete(npu._mxe_accum.flatten(), nest * 16 + spu)
    np.testing.assert_array_equal(other_cells_after, other_cells_before)


def test_funct7_zero_collision_routing():
    """ROADMAP success #3: funct7=0x00 + insn.rs1!=0 → MM, else WRSPR."""
    cases = [
        # (funct7, insn_rs1, expected_route)
        (0x00, 0,    'wrspr'),
        (0x00, 1,    'mm'),
        (0x01, 0,    'mmc'),
        (0x01, 1,    'mmc'),
    ]
    for funct7, rs1, route in cases:
        npu = GtxNpu()
        npu.reset(MockProcessor())
        insn = _synthetic_insn(funct7=funct7, funct3=0, rs1=rs1)
        # Spy: record routing
        routes = []
        # ... call dispatch ...
        assert routes == [route]
```

### 사용자 강하게 선호한 패턴 (Phase 1/2/3/4 일관)

- **순수 함수 + spike 의존 0의 단위 테스트** (D-01 dma_engine.py / D-03 gemm_core.py)
- **C++ 직역 우선, 그러나 module split은 Python 응집도 + P7 JIT 친화도 고려** (D-01 3-way split, D-04 10 individual handlers)
- **실제 사용 시점까지 confer 미룸** (D-13 _verify_minimal mini, D-12 GTX_DDR_DUMP atexit hook은 P6)
- **테스트가 양 모드/이중 검증** (D-07 chain + per-cell isolation; D-08 funct7 matrix 4 cases; D-15 op-level np.array_equal + .elf 회귀 _verify_minimal)

</specifics>

<deferred>
## Deferred Ideas

### Phase 4 plan/research가 정확화 (이번 discuss 비범위)

- `firmware_mm_op` rs1 packed-bit 정확한 마스크 + HW conv 0=65536 in EACH 16-bit field — `/gsd:research-phase 4`가 잠금
- 10 MM/MMC variant 정확한 funct3 매핑 — research가 `gtx_npu_disasm.inc`에서 추출
- `np.matmul` BLAS 누적 순서가 C++ scalar 3-loop과 bit-exact인지 — research lock. Drift 발견 시 explicit 3-loop fallback (P7에서 numba 가속 plan 단계 결정)
- `mm_basic.elf` + golden hex의 vendor 내 정확한 path — research lock
- in-process .elf 로드 메커니즘 (Python에서 ELF 메모리 + entry point 호출 방식) — research lock
- `dispatch_iss_opcode` body에서 funct7=0x00/0x01 case 정확한 위치 (dispatch_4mode.py 확장 vs ops/mm.py 진입점 우회) — Claude's Discretion at plan
- `WarpState` `tmu_id`/`curr_id` 정확한 필드 이름/접근 경로 — plan lock
- `mm_engine.firmware_mm` 내부 helper 분리 정도 (`_decode_args` / `_load_operands` / `_write_result`) — plan lock
- `MM_T` (transposed B) / `MM_V` (vector) variant의 정확한 데이터 경로 — research lock
- `gemm_core` `new_accum` scalar 정확한 semantics (sum vs single-cell vs row-sum) — research가 `gtx_npu_mm.cc:200-205` 직역해서 잠금

### Out of scope for Phase 4 (다른 페이즈로)

- **VEC op 핸들러 (SASMD/DOT/VSUM/CLAMP)** → P5 VEC-01..05
- **ACT op 핸들러 (RELU/SOFTMAX/ESUM forward + PRELU/GELU/TANH/SIGM reversed) + format_cvt + FP8 codec** → P5 ACT-01..05, VRF-02
- **`verify.py` 정식 포팅 → `riscv.gtx._verify` with CLI + console script** → P6 VRF-01, PKG-03
- **전체 .elf 회귀 (run_tests_n1s16 + run_llext_tests 시리즈) 100% strict mode** → P6 VRF-04
- **`tests/gtx/data/{golden,elf}/` package_data 등록 + wheel 동봉** → P6 PKG-01
- **자동 DDR dump (atexit hook)** → P6 또는 별도 follow-up (P3 D-09 + P4 D-12 일관)
- **Numba @njit 가속 적용 (gemm_core 등)** → P7 (D-01/D-03가 numba-friendly 구조 마련만)
- **DMA-3D / IM2COL-N / IM2COL-D / MCAST 본격 구현** → v2 (P3 deferred)
- **`mexec` full microcode loop** → v1 펌웨어 미요구 시 stub
- **Mode 4 inner payload의 VEC/ACT funct7** → P5 (P4는 MM만)

### v2 / 향후 마일스톤

- **PY-OVRD-01** (per-op before/after 후크 — MM에도 적용 가능)
- **PY-FUNCT7-01** (외부 funct7 등록 API — 사용자 정의 MM op)
- **MM-V2-01** (MM-3D / MM-CONV / sparse MM 풀 구현)
- **CYC-01/02** (사이클 카운팅 — 현재 모든 MM이 instantaneous)
- **MM-NUMBA-01** (P7) — `gemm_core.py`에 `@njit` 적용 + `mxe_accum` chain test에서 가속비 측정

### Reviewed Todos (not folded)

None — `gsd-tools todo match-phase 4`에서 매칭 0건.

### Defer to user follow-up

- **D-09 fallback (.S build script vs vendor 차용)** — research가 vendor 검사 후 plan에서 자동 routing
- **D-13 `_verify_minimal` BE FP16 packing 처리 (PITFALLS Pitfall 1)** — research가 vendor verify.py 정확한 로직 잠금
- **gemm_core `new_accum` semantics** — `MM_O` row-sum (gtx_npu_mm.cc:200-205) vs `MMC_O` element-sum, research가 잠금
- **`dispatch_iss_opcode` 확장 vs 우회** — plan 단계에서 line count + 책임 분리 측정 후 결정

</deferred>

---

*Phase: 04-mm-subsystem*
*Context gathered: 2026-05-05 via /gsd:discuss-phase 4*
