# Phase 4: MM Subsystem - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05
**Phase:** 04-mm-subsystem
**Areas discussed:** MM 모듈 분리 + numba 친화 구조, is_accumulate 분기 위치, 첫 .elf 회귀 fixture 전략, strict-mode 검증 인프라 P4 vs P6

---

## Area 1: MM 모듈 분리 + numba 친화 구조

### Q1.1 MM 모듈 구성을 어떻게 할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| **3-way split** | ops/mm.py (@handler 진입점) + mm_engine.py (firmware_mm decode + variant dispatcher + mxe_accum read/write) + gemm_core.py (순수 NumPy GEMM 커널 1개 함수). P7 numba는 gemm_core.py만 @njit로 감싸면 됨 — JIT boundary 명확 | ✓ (Recommended) |
| 2-way split (P3 패턴) | ops/mm.py + mm_engine.py. gemm_core는 mm_engine.py 안의 module-level 함수 | |
| 1-way (단일 파일) | 모든 것을 ops/mm.py에. C++ gtx_npu_mm.cc(약 400 LOC)도 단일이니 직역이 자연 | |

**User's choice:** 3-way split (Recommended)
**Notes:** P7 numba 동적 최적화 페이즈를 명시적으로 염두 — gemm_core.py가 JIT boundary가 됨

### Q1.2 gemm_core 함수 구현은 어떤 NumPy 관용구어 기반으로 하나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **np.matmul(f32) + cast** | C = np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np.float16). 한 줄, FP32 internal accumulate 자동 보장(BLAS 호출), single FP16 cast | ✓ (Recommended) |
| Explicit 3-loop (FP32 accumulator) | C++ 직역, bit-exact 강한 보장. 광어마어리 느림 — 16x16x16 GEMM도 NumPy 루프 vs BLAS 차이 100x. P7 numba가 이 패턴을 제일 잘 최적화 | |
| np.einsum('ik,kj->ij') | 수식적 명확하지만 np.matmul과 동등이고 더 느린 경향 | |
| You decide | Claude가 research+plan으로 정확한 선택 | |

**User's choice:** np.matmul(f32) + cast (Recommended)
**Notes:** BLAS 누적 순서가 C++ scalar 3-loop과 bit-exact인지는 research가 검증 — drift 발견 시 explicit 3-loop fallback (P7에서 numba 가속). CONTEXT D-02 위험 항목으로 기록

### Q1.3 gemm_core를 P4에서부터 numba @njit 키워드 함수에 골골하게 쓸 만한 '순수 stateless' 계약을 서명할까요?

| Option | Description | Selected |
|--------|-------------|----------|
| **예—P4에서부터 일관** | gemm_core(A: ndarray, B: ndarray, *, has_bias: bool = False, prior_accum: float = 0.0) -> tuple[ndarray, float]으로 stateless API. mxe_accum은 출력만 반환, 쓰기는 mm_engine이 | ✓ (Recommended) |
| 아니오—mxe_accum direct write 허용 | gemm_core(A, B, mxe_accum_view: ndarray) -> None 같이 view를 통해 in-place write. P3 dma_engine도 GtxMemory를 받아 in-place | |
| Hybrid — stateless inner + stateful wrapper | gemm_core(A, B) -> ndarray (순수) + mm_engine이 mxe_accum 추가 처리 | |

**User's choice:** 예—P4에서부터 일관 (Recommended)
**Notes:** stateless 계약이 P7 numba JIT 적용 시 zero-friction. CONTEXT D-03

### Q1.4 10개 MM/MMC variant의 @handler 등록 방식은?

| Option | Description | Selected |
|--------|-------------|----------|
| **10개 별도 @handler 함수** | 각 variant 고유 funct3, P3 dma 패턴 동일. _exec_mm, _exec_mm_s, ... 10개 명시적. is_accumulate는 mmc 계열에서 자동 결정. spike trace에 정확한 mnemonic 노출. LOC 늘어 ~150 | ✓ (Recommended) |
| 2개 master + funct3 dispatch | _exec_firmware_mm + _exec_firmware_mmc 둘만 @handler 등록, 내부에서 funct3로 variant 분기 | |
| Loop registration | for variant in MM_VARIANTS: handler(...)(_exec_factory(variant)) | |

**User's choice:** 10개 별도 @handler 함수 (Recommended)
**Notes:** P7 numba JIT은 closure factory보다 individual function이 더 잘 가속됨. CONTEXT D-04

---

## Area 2: is_accumulate 분기 위치

### Q2.1 is_accumulate 분기는 어느 layer에서 일어나야 하나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **ops/mm.py 진입점** | mmc 계열은 is_accumulate=True 명시 전달, mm 계열은 False. C++ 패턴 직역 | ✓ (Recommended) |
| mm_engine.firmware_mm wrapper | 진입점은 단순 전달, mm_engine 내부에서 funct7로 결정 | |
| gemm_core(accumulate=) parameter | gemm_core 자체 signature에 accumulate 변수 | |

**User's choice:** ops/mm.py 진입점 (Recommended)
**Notes:** routing 사고 0, funct7 collision 디버깅 명확. CONTEXT D-05

### Q2.2 MMC 계열 (funct7=0x01)의 mxe_accum read 은 언제 일어나나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **ops/mm.py의 진입점** | _exec_mmc_o가 npu._mxe_accum[nest, spu] read → gemm_core 호출 → 반환값 write back | ✓ (Recommended) |
| mm_engine.firmware_mm 내부 | mm_engine이 npu 도메인 알고 mxe_accum read/write | |
| gemm_core에서 npu 알면 안 됨 — mm_engine이 read 후 전달 | (2)와 비슷, prior_accum read 후 전달 | |

**User's choice:** ops/mm.py의 진입점 (Recommended)
**Notes:** scope 격리 + gemm_core stateless 계약(D-03) 일치. CONTEXT D-06

### Q2.3 Pitfall 3 anti-pattern test 설계?

| Option | Description | Selected |
|--------|-------------|----------|
| **Chain mm.s→mmc.s→mmc 단일 테스트** | ROADMAP success #2 그대로. 3-step chain → final FP16 == np.float16(A1@B1 + A2@B2 + A3@B3) FP32 internal | ✓ (Recommended) |
| Per-(NEST,SPU) 격리 테스트 추가 | (1) + snapshot.copy() → mmc.s on (1, 5) → 다른 (4*16-1) cell unchanged | |
| Funct7 routing test | (1) + parametrized funct7 → spy attribute로 is_accumulate 확인 | |
| (1) + (2) 둘 다 | Chain test + per-cell isolation test 모두 추가 | |

**User's choice:** Chain mm.s→mmc.s→mmc 단일 테스트 (Recommended)
**Notes:** CONTEXT D-07. 단 dual-assertion 패턴 (P3 D-11 mirror)으로 chain 테스트 안에 per-cell isolation도 함께 검증하는 형태로 강화 (CONTEXT D-07 본문 참조)

### Q2.4 funct7=0x00 collision (WRSPR vs MM) 검증 테스트는?

| Option | Description | Selected |
|--------|-------------|----------|
| **Parametrized matrix test** | ROADMAP success #3 그대로. (funct7=0x00, insn.rs1=0) → WRSPR; (funct7=0x00, insn.rs1!=0) → MM. 4 cases | ✓ (Recommended) |
| Integration test only (.elf) | mm_basic.elf 통과로 funct7=0x00 routing 간접 검증 | |
| Skip — P2에서 이미 land | P2 D-02 그대로 신뢰 | |

**User's choice:** Parametrized matrix test (Recommended)
**Notes:** CONTEXT D-08

---

## Area 3: 첫 .elf 회귀 fixture 전략

### Q3.1 mm_basic.elf는 어떻게 마련하나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **Vendor 에서 차용** | vendor/gtx_cpp_reference/ 내부에 이미 펌웨어 테스트 자산이 있다면 차용. golden hex도 같이 받을 가능성 ↑ | ✓ (Recommended) |
| 새 .S 소스 + Makefile + 사전 빌드 .elf 커밋 (P2 D-22 패턴) | tests/gtx/data/elf/{mm_basic.S, Makefile, mm_basic.elf} | |
| Vendor build script 호출 | Makefile target이 vendor의 .elf를 도출 → tests/gtx/data/elf로 symlink/copy | |
| You decide | research가 vendor 검사 후 plan에서 정확한 선택 | |

**User's choice:** Vendor 에서 차용 (Recommended)
**Notes:** vendor에 적합 fixture 없으면 P2 D-22 패턴(.S + Makefile)으로 fallback. CONTEXT D-09

### Q3.2 Golden hex (mm_basic_n1s16.hex) 출처는?

| Option | Description | Selected |
|--------|-------------|----------|
| **Vendor 에서 차용** | vendor/gtx_cpp_reference/ 내 golden hex (C++ libgtx_npu.so SystemC HW sim ULP 일치 검증 완료된 자산) | ✓ (Recommended if ELF too) |
| C++ libgtx_npu.so 실행 결과를 P4에서 생성 | dev 환경에 libgtx_npu.so 필요, CI는 사전 빌드 필요 | |
| Synthetic NumPy oracle (P4 self-contained) | 16x16x16 random GEMM → np.float32 → .float16 cast → hex format. 'C++ libgtx_npu.so 일치' 증명은 안 됨 | |

**User's choice:** Vendor 에서 차용 (Recommended)
**Notes:** ROADMAP P4 success #4의 'libgtx_npu.so와 ULP 내 일치' 정합. CONTEXT D-10

### Q3.3 .elf 구동 방식: pyspike CLI subprocess vs in-process pytest?

| Option | Description | Selected |
|--------|-------------|----------|
| **pytest in-process 실행** | tests/gtx/test_op_mm.py 내부에서 subprocess 없이 GtxNpu + .elf 메모리 로드 + 펌웨어 진입점 호출. 빠르고, debug 단순, GIL 안전. 필요 시 P2 test_skeleton.py 패턴(subprocess)으로 스위치 | ✓ (Recommended) |
| subprocess pyspike (서온하게 실행) | subprocess.run([pyspike, '--extlib=riscv.gtx', 'mm_basic.elf']) + GTX_DDR_DUMP env var | |
| (1) + (2) 둘 다 | 10개 MM variant 단위는 in-process, .elf 회귀 하나만 subprocess | |

**User's choice:** pytest in-process 실행 (Recommended)
**Notes:** GIL/single-hart 이슈 시 subprocess fallback. CONTEXT D-11

### Q3.4 GTX_DDR_DUMP 환경변수 디렉티브 처리는 P4에서 처음 도입?

| Option | Description | Selected |
|--------|-------------|----------|
| **P4에서 처음 도입 (테스트 내부만)** | P4 회귀 테스트가 펌웨어 종료 후 ddr_dump_to_file 명시 호출. production atexit hook은 P6 | ✓ (Recommended) |
| P4에서 production hook까지 | riscv/gtx/__main__.py 진입점에 GTX_DDR_DUMP env var atexit hook 추가 | |
| P4도 미룸 (테스트에서 만드는 dump path) | 테스트가 프로그램 종료 후 ddr_dump_to_file() 명시 호출, env var 안 씀 | |

**User's choice:** P4에서 처음 도입 (Recommended)
**Notes:** P3 D-09 ('자동 dump는 P6 또는 별도 follow-up') 상점 유지. CONTEXT D-12

---

## Area 4: strict-mode 검증 인프라 P4 vs P6

### Q4.1 P4 첫 .elf 검증 시 strict-mode 도구는 어느 수준으로 준비하나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **P4 mini 직역** | tests/gtx/_verify_minimal.py 같은 테스트 전용 helper. compare_hex(actual_path, golden_path, ulp=1, atol=0.001, strict=True) -> bool. ~30 LOC. P6에서 riscv.gtx._verify (production CLI)로 승격 | ✓ (Recommended) |
| P4는 np.array_equal만, hex 경경 부레아 안 음 | actual_bytes vs expected_bytes 직접 비교. ROADMAP success #4 'verify.py --strict' 명시 부합 부족 | |
| Vendor verify.py subprocess | vendor 의존 더 김어짐, P6에서 결국 다시 포팅 | |

**User's choice:** P4 mini 직역 (Recommended)
**Notes:** CONTEXT D-13. PITFALLS Pitfall 1 (BE FP16 packing) 처리 포함

### Q4.2 Strict mode의 'exact_matches == total_fp16' 수증은 P4에서 자동 엔돈 되나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **P4도 strict 강제** | within_tolerance match 하나도 이용 없이 'failed' 보고. ROADMAP success #4 그대로 | ✓ (Recommended) |
| P4는 ULP=1 관용, P6에서 strict 스위치 | drift 누적 위험 | |
| Strict 되, P4는 fixture 축소해 'within ULP=1' 일이수 최소화 | 16x16x16 GEMM 하나만 fixture | |

**User's choice:** P4도 strict 강제 (Recommended)
**Notes:** drift 이슈는 첫 .elf 회귀에서 잡는 게 가장 저렴. CONTEXT D-14

### Q4.3 P4 테스트가 아닌 op-level unit test도 hex 파일 비교 경로를 쓰나요?

| Option | Description | Selected |
|--------|-------------|----------|
| **Op-level은 np.array_equal 직접** | ROADMAP success #1 그대로. assert_array_equal(actual.view(np.uint16), expected.view(np.uint16)). NumPy oracle과 직접 비교 | ✓ (Recommended) |
| Op-level도 _verify_minimal | dump → hex → compare_hex 경로 | |
| Op-level np.array_equal + hex을 병행 | hex compare를 디버깅 보조 helper로 | |

**User's choice:** Op-level은 np.array_equal 직접 (Recommended)
**Notes:** error message 직관성 우선. _verify_minimal은 .elf 회귀에만. CONTEXT D-15

---

## Closing

### Q5 더 논의할 영역?

| Option | Description | Selected |
|--------|-------------|----------|
| **I'm ready for context** | 15개 결정 충분. CONTEXT.md 작성 후 /gsd:plan-phase 4 진행 | ✓ (Recommended) |
| Explore more gray areas | (a) MM L1/L2 read 점극 패턴 (b) 테스트 데이터 generation strategy (c) MM-V transposed B (d) tmu_id/curr_id 필드명 등 — plan에서 정확화해도 됨 | |

**User's choice:** I'm ready for context (Recommended)

## Claude's Discretion

[CONTEXT.md `<decisions>` "Claude's Discretion" 섹션 참조 — 10개 항목 plan/research가 정확화]

## Deferred Ideas

[CONTEXT.md `<deferred>` 섹션 참조 — Phase 4 plan/research 잠금 항목 + Out of scope (P5/P6/v2) + 사용자 follow-up]
